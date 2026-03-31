"""Contextual expansion — enrich top results with surrounding chunks and graph context."""

from __future__ import annotations

from oracle.knowledge.neo4j_client import Neo4jClient
from oracle.knowledge.qdrant_client import QdrantManager
from oracle.models import ExpandedContext, FusedResult


class ContextualExpander:
    """Expand each top result with surrounding chunks and graph neighbors.

    For each result:
    1. Fetch surrounding chunks (same article, adjacent indices) from Qdrant
    2. Fetch graph node properties for mentioned entities
    3. Fetch immediate graph neighbors of those entities
    """

    def __init__(self, qdrant: QdrantManager, neo4j: Neo4jClient) -> None:
        self.qdrant = qdrant
        self.neo4j = neo4j

    async def expand(
        self,
        results: list[FusedResult],
        collection: str = "news_articles",
    ) -> list[FusedResult]:
        """Expand each result with surrounding context."""
        for result in results:
            context = ExpandedContext()

            # 1. Surrounding chunks from same source
            surrounding = await self._fetch_surrounding_chunks(
                result.chunk_id, result.metadata, collection
            )
            context.surrounding_chunks = surrounding

            # 2. Graph properties and neighbors for mentioned entities
            entity_ids = result.metadata.get("entity_ids", [])
            if entity_ids:
                props, neighbors = await self._fetch_graph_context(entity_ids)
                context.graph_properties = props
                context.graph_neighbors = neighbors

            result.expanded_context = context

        return results

    async def _fetch_surrounding_chunks(
        self,
        chunk_id: str,
        metadata: dict,
        collection: str,
    ) -> list[str]:
        """Fetch adjacent chunks from the same source article."""
        source_url = metadata.get("source_url")
        if not source_url:
            return []

        try:
            # Search for chunks from the same source
            results, _ = await self.qdrant.client.scroll(
                collection_name=collection,
                scroll_filter={
                    "must": [
                        {"key": "source_url", "match": {"value": source_url}},
                    ]
                },
                limit=10,
                with_payload=True,
                with_vectors=False,
            )

            # Sort by chunk index if available, return texts
            chunks = []
            for point in results:
                if str(point.id) != chunk_id:
                    chunks.append(point.payload.get("text", ""))

            return chunks[:4]  # Return up to 4 surrounding chunks

        except Exception:
            return []

    async def _fetch_graph_context(
        self, entity_ids: list[str]
    ) -> tuple[dict, list[dict]]:
        """Fetch graph node properties and immediate neighbors."""
        properties: dict = {}
        neighbors: list[dict] = []

        for entity_id in entity_ids[:5]:  # Limit to 5 entities
            try:
                cypher_props = """
                CALL db.index.fulltext.queryNodes('entity_names', $name)
                YIELD node, score
                WHERE score > 0.5
                WITH node
                LIMIT 1
                RETURN labels(node)[0] AS type, properties(node) AS props
                """
                async with self.neo4j.driver.session() as session:
                    result = await session.run(cypher_props, name=entity_id)
                    records = await result.data()
                    if records:
                        properties[entity_id] = {
                            "type": records[0]["type"],
                            "properties": records[0]["props"],
                        }

                # Fetch immediate neighbors
                cypher_neighbors = """
                CALL db.index.fulltext.queryNodes('entity_names', $name)
                YIELD node, score
                WHERE score > 0.5
                WITH node
                LIMIT 1
                MATCH (node)-[r]-(neighbor)
                RETURN labels(neighbor)[0] AS type,
                       neighbor.name AS name,
                       type(r) AS relationship,
                       properties(neighbor) AS props
                LIMIT 10
                """
                async with self.neo4j.driver.session() as session:
                    result = await session.run(cypher_neighbors, name=entity_id)
                    records = await result.data()
                    for rec in records:
                        neighbors.append(
                            {
                                "entity": entity_id,
                                "neighbor_name": rec["name"],
                                "neighbor_type": rec["type"],
                                "relationship": rec["relationship"],
                                "properties": rec.get("props", {}),
                            }
                        )

            except Exception:
                continue

        return properties, neighbors
