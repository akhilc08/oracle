"""Neo4j knowledge graph client with schema management."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog
from neo4j import AsyncGraphDatabase, AsyncDriver

logger = structlog.get_logger()

# Schema definitions
NODE_TYPES = {
    "Person": {
        "properties": ["name", "role", "party", "organization", "sentiment_score"],
        "unique_key": "name",
    },
    "Organization": {
        "properties": ["name", "type", "sector", "country"],
        "unique_key": "name",
    },
    "Event": {
        "properties": ["name", "date", "status", "outcome"],
        "unique_key": "name",
    },
    "Policy": {
        "properties": ["name", "status", "sponsors", "affected_sectors"],
        "unique_key": "name",
    },
    "Market": {
        "properties": [
            "polymarket_id",
            "question",
            "current_price",
            "volume",
            "resolution_date",
            "category",
            "active",
        ],
        "unique_key": "polymarket_id",
    },
    "Location": {
        "properties": ["name", "country", "region", "geopolitical_significance"],
        "unique_key": "name",
    },
    "LegalCase": {
        "properties": ["case_name", "court", "status", "parties"],
        "unique_key": "case_name",
    },
}

EDGE_TYPES = [
    ("LEADS", "Person", "Organization"),
    ("AFFECTS", "Event", "Market"),
    ("SPONSORS", "Person", "Policy"),
    ("OPPOSES", "Person", "Policy"),
    ("ALLIED_WITH", "Organization", "Organization"),
    ("PRECEDED_BY", "Event", "Event"),
    ("IMPACTS_SECTOR", "Policy", "Organization"),
    ("RULES_ON", "LegalCase", "Policy"),
    ("LOCATED_IN", "Organization", "Location"),
    ("RELATED_TO", "Market", "Event"),
    ("MENTIONS", "Market", "Person"),
]


class Neo4jClient:
    """Async Neo4j client for Oracle knowledge graph."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        self._uri = uri

    @property
    def driver(self) -> AsyncDriver:
        """Expose underlying async driver for advanced queries."""
        return self._driver

    async def setup_schema(self) -> None:
        """Create constraints, indexes, and verify schema."""
        async with self._driver.session() as session:
            # Uniqueness constraints (also create implicit indexes)
            for label, config in NODE_TYPES.items():
                key = config["unique_key"]
                constraint_name = f"unique_{label.lower()}_{key}"
                query = (
                    f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE n.{key} IS UNIQUE"
                )
                await session.run(query)
                logger.info("neo4j.constraint_created", label=label, key=key)

            # Composite indexes for common query patterns
            indexes = [
                ("idx_market_active", "Market", ["active", "category"]),
                ("idx_event_status", "Event", ["status", "date"]),
                ("idx_person_org", "Person", ["organization", "role"]),
                ("idx_policy_status", "Policy", ["status"]),
                ("idx_legalcase_status", "LegalCase", ["status", "court"]),
            ]
            for name, label, props in indexes:
                props_str = ", ".join(f"n.{p}" for p in props)
                query = (
                    f"CREATE INDEX {name} IF NOT EXISTS FOR (n:{label}) ON ({props_str})"
                )
                await session.run(query)
                logger.info("neo4j.index_created", name=name, label=label)

            # Full-text search index on names
            await session.run(
                "CREATE FULLTEXT INDEX entity_names IF NOT EXISTS "
                "FOR (n:Person|Organization|Event|Policy|Location|LegalCase) "
                "ON EACH [n.name]"
            )

            # Timestamp tracking — add created_at/updated_at to all nodes
            logger.info("neo4j.schema_setup_complete", node_types=len(NODE_TYPES))

    async def verify_connectivity(self) -> bool:
        """Check Neo4j is reachable."""
        try:
            await self._driver.verify_connectivity()
            return True
        except Exception:
            return False

    async def get_stats(self) -> dict[str, Any]:
        """Get knowledge graph statistics."""
        async with self._driver.session() as session:
            result = await session.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count "
                "ORDER BY count DESC"
            )
            node_counts = {record["label"]: record["count"] async for record in result}

            result = await session.run(
                "MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count "
                "ORDER BY count DESC"
            )
            edge_counts = {record["type"]: record["count"] async for record in result}

            return {
                "total_nodes": sum(node_counts.values()),
                "total_edges": sum(edge_counts.values()),
                "node_counts": node_counts,
                "edge_counts": edge_counts,
            }

    async def merge_entity(
        self, label: str, properties: dict[str, Any]
    ) -> dict[str, Any]:
        """Create or update an entity node. Uses MERGE on unique key."""
        if label not in NODE_TYPES:
            raise ValueError(f"Unknown node type: {label}")

        config = NODE_TYPES[label]
        unique_key = config["unique_key"]

        if unique_key not in properties:
            raise ValueError(f"Missing required property: {unique_key}")

        now = datetime.now(timezone.utc).isoformat()
        properties["updated_at"] = now

        # Build MERGE query
        set_clauses = ", ".join(f"n.{k} = ${k}" for k in properties if k != unique_key)
        query = (
            f"MERGE (n:{label} {{{unique_key}: ${unique_key}}}) "
            f"ON CREATE SET n.created_at = $now, {set_clauses} "
            f"ON MATCH SET {set_clauses} "
            f"RETURN n"
        )
        params = {**properties, "now": now}

        async with self._driver.session() as session:
            result = await session.run(query, params)
            record = await result.single()
            return dict(record["n"]) if record else {}

    async def create_relationship(
        self,
        from_label: str,
        from_key_value: str,
        rel_type: str,
        to_label: str,
        to_key_value: str,
        properties: dict[str, Any] | None = None,
    ) -> bool:
        """Create a relationship between two entities."""
        from_config = NODE_TYPES[from_label]
        to_config = NODE_TYPES[to_label]
        from_key = from_config["unique_key"]
        to_key = to_config["unique_key"]

        props_str = ""
        params: dict[str, Any] = {"from_val": from_key_value, "to_val": to_key_value}

        if properties:
            props_str = " {" + ", ".join(f"{k}: ${k}" for k in properties) + "}"
            params.update(properties)

        now = datetime.now(timezone.utc).isoformat()
        params["now"] = now

        query = (
            f"MATCH (a:{from_label} {{{from_key}: $from_val}}) "
            f"MATCH (b:{to_label} {{{to_key}: $to_val}}) "
            f"MERGE (a)-[r:{rel_type}{props_str}]->(b) "
            f"ON CREATE SET r.created_at = $now "
            f"RETURN r"
        )

        async with self._driver.session() as session:
            result = await session.run(query, params)
            record = await result.single()
            return record is not None

    async def get_entities(self, entity_type: str, limit: int = 50) -> list[dict[str, Any]]:
        """Get entities of a given type."""
        if entity_type not in NODE_TYPES:
            raise ValueError(f"Unknown entity type: {entity_type}")

        async with self._driver.session() as session:
            result = await session.run(
                f"MATCH (n:{entity_type}) RETURN n ORDER BY n.updated_at DESC LIMIT $limit",
                {"limit": limit},
            )
            return [dict(record["n"]) async for record in result]

    async def get_markets(
        self, limit: int = 50, active_only: bool = True
    ) -> list[dict[str, Any]]:
        """Get tracked markets."""
        where = "WHERE n.active = true" if active_only else ""
        async with self._driver.session() as session:
            result = await session.run(
                f"MATCH (n:Market) {where} "
                f"RETURN n ORDER BY n.volume DESC LIMIT $limit",
                {"limit": limit},
            )
            return [dict(record["n"]) async for record in result]

    async def get_market_detail(self, market_id: str) -> dict[str, Any]:
        """Get market with related entities."""
        async with self._driver.session() as session:
            # Get market node
            result = await session.run(
                "MATCH (m:Market {polymarket_id: $id}) "
                "OPTIONAL MATCH (m)-[r]-(related) "
                "RETURN m, collect({type: type(r), node: related, labels: labels(related)}) AS connections",
                {"id": market_id},
            )
            record = await result.single()
            if not record:
                return {"error": "Market not found"}

            market = dict(record["m"])
            connections = []
            for conn in record["connections"]:
                if conn["node"] is not None:
                    connections.append({
                        "relationship": conn["type"],
                        "entity_type": conn["labels"][0] if conn["labels"] else "Unknown",
                        "entity": dict(conn["node"]),
                    })

            return {"market": market, "connections": connections}

    async def graph_search(
        self, entity_name: str, max_depth: int = 2, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Traverse graph from an entity to find connected information."""
        async with self._driver.session() as session:
            result = await session.run(
                "CALL db.index.fulltext.queryNodes('entity_names', $name) "
                "YIELD node, score "
                "WITH node, score ORDER BY score DESC LIMIT 1 "
                "CALL apoc.path.subgraphAll(node, {maxLevel: $depth, limit: $limit}) "
                "YIELD nodes, relationships "
                "RETURN nodes, relationships",
                {"name": entity_name, "depth": max_depth, "limit": limit},
            )
            record = await result.single()
            if not record:
                return []

            nodes = [
                {"labels": list(n.labels), "properties": dict(n)}
                for n in record["nodes"]
            ]
            return nodes

    async def close(self) -> None:
        """Close the driver connection."""
        await self._driver.close()
