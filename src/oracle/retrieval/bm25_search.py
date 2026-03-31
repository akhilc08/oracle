"""BM25 keyword search for exact matching of names, tickers, and terms."""

from __future__ import annotations

import math
import re
from collections import defaultdict

from oracle.models import RetrievalQuery, RetrievalResult


class BM25Index:
    """In-memory BM25 index with Okapi BM25 scoring.

    Uses rank_bm25-style scoring without external dependencies.
    Documents are added incrementally and the index supports filtering.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_texts: list[str] = []
        self.doc_metadata: list[dict] = []
        self.doc_lengths: list[int] = []
        self.avg_dl: float = 0.0
        self.doc_count: int = 0
        # term -> {doc_index: term_frequency}
        self.inverted_index: dict[str, dict[int, int]] = defaultdict(dict)
        # term -> document frequency
        self.df: dict[str, int] = defaultdict(int)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer with lowercasing."""
        return re.findall(r"\b\w+\b", text.lower())

    def add_documents(
        self,
        doc_ids: list[str],
        texts: list[str],
        metadata: list[dict] | None = None,
    ) -> None:
        """Add documents to the BM25 index."""
        if metadata is None:
            metadata = [{}] * len(doc_ids)

        for doc_id, text, meta in zip(doc_ids, texts, metadata):
            idx = self.doc_count
            self.doc_ids.append(doc_id)
            self.doc_texts.append(text)
            self.doc_metadata.append(meta)

            tokens = self.tokenize(text)
            self.doc_lengths.append(len(tokens))

            seen_terms: set[str] = set()
            for token in tokens:
                self.inverted_index[token][idx] = (
                    self.inverted_index[token].get(idx, 0) + 1
                )
                if token not in seen_terms:
                    self.df[token] += 1
                    seen_terms.add(token)

            self.doc_count += 1

        total = sum(self.doc_lengths)
        self.avg_dl = total / self.doc_count if self.doc_count > 0 else 0.0

    def search(
        self,
        query: str,
        top_k: int = 20,
        allowed_ids: set[str] | None = None,
    ) -> list[tuple[str, float, str, dict]]:
        """Search the index, returning (doc_id, score, text, metadata) tuples."""
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores: dict[int, float] = defaultdict(float)

        for token in query_tokens:
            if token not in self.inverted_index:
                continue

            doc_freq = self.df[token]
            idf = math.log(
                (self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0
            )

            for doc_idx, tf in self.inverted_index[token].items():
                if allowed_ids is not None and self.doc_ids[doc_idx] not in allowed_ids:
                    continue

                dl = self.doc_lengths[doc_idx]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
                scores[doc_idx] += idf * numerator / denominator

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            (self.doc_ids[idx], score, self.doc_texts[idx], self.doc_metadata[idx])
            for idx, score in ranked
        ]


class BM25SearchStrategy:
    """BM25 keyword search strategy for the retrieval pipeline."""

    def __init__(self) -> None:
        self.index = BM25Index()
        self._indexed = False

    def build_index(
        self,
        doc_ids: list[str],
        texts: list[str],
        metadata: list[dict] | None = None,
    ) -> None:
        """Build or rebuild the BM25 index from documents."""
        self.index = BM25Index()
        self.index.add_documents(doc_ids, texts, metadata)
        self._indexed = True

    async def search(self, query: RetrievalQuery) -> list[RetrievalResult]:
        """Run BM25 keyword search."""
        if not self._indexed:
            return []

        raw_results = self.index.search(query.text, top_k=query.top_k)

        return [
            RetrievalResult(
                chunk_id=doc_id,
                text=text,
                score=score,
                source="bm25",
                metadata=meta,
            )
            for doc_id, score, text, meta in raw_results
        ]

    async def search_with_qdrant_docs(
        self,
        query: RetrievalQuery,
        qdrant_manager: "QdrantManager",
    ) -> list[RetrievalResult]:
        """Fetch documents from Qdrant, build index on-the-fly, and search.

        This is the primary entry point — builds a fresh BM25 index from the
        Qdrant collection, then runs keyword search over it.
        """
        from qdrant_client.models import ScrollRequest

        # Scroll through all points in the collection to build the index
        all_points = []
        offset = None
        while True:
            results, next_offset = await qdrant_manager.client.scroll(
                collection_name=query.collection,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_points.extend(results)
            if next_offset is None:
                break
            offset = next_offset

        if not all_points:
            return []

        doc_ids = [str(p.id) for p in all_points]
        texts = [p.payload.get("text", "") for p in all_points]
        metadata = [p.payload for p in all_points]

        self.build_index(doc_ids, texts, metadata)
        return await self.search(query)
