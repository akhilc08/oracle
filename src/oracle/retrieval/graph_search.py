"""Graph traversal retrieval — entity resolution from query → Cypher → expand context."""

from __future__ import annotations

from oracle.ingestion.entity_resolver import EntityResolver
from oracle.knowledge.neo4j_client import Neo4jClient
from oracle.models import RetrievalQuery, RetrievalResult


class GraphSearchStrategy:
    """Retrieve relevant chunks by traversing the knowledge graph.

    Pipeline:
    1. Extract entities from query text via NER
    2. Resolve entities to canonical names
    3. Full-text search in Neo4j for each entity
    4. Expand 2 hops from matched nodes
    5. Collect connected chunk references and entity context
    """

    def __init__(self, neo4j: Neo4jClient, entity_resolver: EntityResolver) -> None:
        self.neo4j = neo4j
        self.entity_resolver = entity_resolver

    async def search(self, query: RetrievalQuery) -> list[RetrievalResult]:
        """Run graph-based retrieval from query text."""
        # Step 1: Extract and resolve entities from the query
        entities = self.entity_resolver.extract_and_resolve(query.text)

        if not entities:
            # Fallback: use raw query words for full-text search
            return await self._fulltext_fallback(query)

        results: list[RetrievalResult] = []
        seen_ids: set[str] = set()

        for entity in entities:
            entity_name = entity["properties"].get("name", "")
            if not entity_name:
                continue

            # Step 2: Find entity in graph via full-text search
            graph_results = await self._expand_entity(entity_name, max_depth=2)

            for item in graph_results:
                item_id = f"graph_{item['name']}_{item.get('type', 'unknown')}"
                if item_id in seen_ids:
                    continue
                seen_ids.add(item_id)

                # Build a text representation from graph node properties
                text = self._node_to_text(item)
                # Score based on path length (closer = higher score)
                depth = item.get("depth", 0)
                score = 1.0 / (1.0 + depth)

                results.append(
                    RetrievalResult(
                        chunk_id=item_id,
                        text=text,
                        score=score,
                        source="graph",
                        metadata={
                            "entity_name": item.get("name", ""),
                            "entity_type": item.get("type", ""),
                            "depth": depth,
                            "relationships": item.get("relationships", []),
                            "properties": item.get("properties", {}),
                        },
                    )
                )

        # Sort by score descending, limit to top_k
        results.sort(key=lambda r: r.score, reverse=True)
        return results[: query.top_k]

    async def _expand_entity(self, entity_name: str, max_depth: int = 2) -> list[dict]:
        """Find entity in Neo4j and expand 2 hops."""
        cypher = """
        CALL db.index.fulltext.queryNodes('entity_names', $name) YIELD node, score
        WHERE score > 0.5
        WITH node, score
        ORDER BY score DESC
        LIMIT 3
        CALL apoc.path.subgraphAll(node, {maxLevel: $max_depth})
        YIELD nodes, relationships
        UNWIND nodes AS n
        WITH DISTINCT n, score,
             [r IN relationships WHERE startNode(r) = n OR endNode(r) = n |
                {type: type(r),
                 start: properties(startNode(r)).name,
                 end: properties(endNode(r)).name}
             ] AS rels
        RETURN labels(n)[0] AS type,
               properties(n) AS properties,
               properties(n).name AS name,
               rels AS relationships,
               CASE WHEN properties(n).name = $name THEN 0
                    WHEN any(r IN rels WHERE r.start = $name OR r.end = $name) THEN 1
                    ELSE 2 END AS depth
        ORDER BY depth ASC
        LIMIT 50
        """
        try:
            async with self.neo4j.driver.session() as session:
                result = await session.run(
                    cypher, name=entity_name, max_depth=max_depth
                )
                records = await result.data()
                return records
        except Exception:
            # Fallback to simpler query without APOC
            return await self._simple_expand(entity_name, max_depth)

    async def _simple_expand(
        self, entity_name: str, max_depth: int = 2
    ) -> list[dict]:
        """Simpler graph expansion without APOC dependency."""
        cypher = """
        CALL db.index.fulltext.queryNodes('entity_names', $name) YIELD node, score
        WHERE score > 0.5
        WITH node, score
        ORDER BY score DESC
        LIMIT 3
        OPTIONAL MATCH (node)-[r1]-(hop1)
        OPTIONAL MATCH (hop1)-[r2]-(hop2)
        WHERE hop2 <> node
        WITH COLLECT(DISTINCT {
            name: node.name,
            type: labels(node)[0],
            properties: properties(node),
            depth: 0,
            relationships: []
        }) +
        COLLECT(DISTINCT {
            name: hop1.name,
            type: labels(hop1)[0],
            properties: properties(hop1),
            depth: 1,
            relationships: [{type: type(r1), start: node.name, end: hop1.name}]
        }) +
        COLLECT(DISTINCT {
            name: hop2.name,
            type: labels(hop2)[0],
            properties: properties(hop2),
            depth: 2,
            relationships: [{type: type(r2), start: hop1.name, end: hop2.name}]
        }) AS all_nodes
        UNWIND all_nodes AS n
        WHERE n.name IS NOT NULL
        RETURN DISTINCT n.name AS name, n.type AS type,
               n.properties AS properties, n.depth AS depth,
               n.relationships AS relationships
        ORDER BY n.depth ASC
        LIMIT 50
        """
        try:
            async with self.neo4j.driver.session() as session:
                result = await session.run(cypher, name=entity_name)
                return await result.data()
        except Exception:
            return []

    async def _fulltext_fallback(self, query: RetrievalQuery) -> list[RetrievalResult]:
        """When no entities extracted, search full-text index with raw query."""
        cypher = """
        CALL db.index.fulltext.queryNodes('entity_names', $query) YIELD node, score
        WHERE score > 0.3
        RETURN labels(node)[0] AS type,
               properties(node) AS properties,
               properties(node).name AS name,
               score
        ORDER BY score DESC
        LIMIT $limit
        """
        try:
            async with self.neo4j.driver.session() as session:
                result = await session.run(
                    cypher, query=query.text, limit=query.top_k
                )
                records = await result.data()

            return [
                RetrievalResult(
                    chunk_id=f"graph_{r['name']}_{r['type']}",
                    text=self._node_to_text(r),
                    score=r["score"],
                    source="graph",
                    metadata={
                        "entity_name": r.get("name", ""),
                        "entity_type": r.get("type", ""),
                        "depth": 0,
                        "properties": r.get("properties", {}),
                    },
                )
                for r in records
            ]
        except Exception:
            return []

    @staticmethod
    def _node_to_text(item: dict) -> str:
        """Convert a graph node to a text representation for downstream use."""
        props = item.get("properties", {})
        node_type = item.get("type", "Entity")
        name = props.get("name", item.get("name", "Unknown"))

        parts = [f"{node_type}: {name}"]

        # Add relevant properties
        skip_keys = {"name", "created_at", "updated_at"}
        for key, value in props.items():
            if key in skip_keys or value is None or value == "":
                continue
            parts.append(f"  {key}: {value}")

        # Add relationship info
        rels = item.get("relationships", [])
        if rels:
            rel_strs = []
            for rel in rels[:5]:  # Limit to 5 relationships
                if isinstance(rel, dict):
                    rel_strs.append(
                        f"{rel.get('start', '?')} -[{rel.get('type', '?')}]-> {rel.get('end', '?')}"
                    )
            if rel_strs:
                parts.append("  Connections: " + "; ".join(rel_strs))

        return "\n".join(parts)
