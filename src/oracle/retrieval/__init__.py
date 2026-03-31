"""Oracle hybrid retrieval engine.

Orchestrates 4 retrieval strategies:
1. Vector similarity search (Qdrant HNSW + BGE-large)
2. BM25 keyword search (in-memory Okapi BM25)
3. Graph traversal (Neo4j entity expansion, 2 hops)
4. Recency-weighted vector search (exponential decay)

Results are merged via Reciprocal Rank Fusion (k=60),
re-ranked by BGE-reranker-v2-m3 cross-encoder,
and expanded with surrounding chunks + graph context.
"""
