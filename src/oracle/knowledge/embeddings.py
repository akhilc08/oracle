"""Embedding service using BGE-large-en-v1.5."""

from __future__ import annotations

import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

logger = structlog.get_logger()


class EmbeddingService:
    """Manages BGE-large-en-v1.5 embedding model for Oracle."""

    _instance: EmbeddingService | None = None
    _model: SentenceTransformer | None = None

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5") -> None:
        self._model_name = model_name

    @classmethod
    def get_instance(cls, model_name: str = "BAAI/bge-large-en-v1.5") -> EmbeddingService:
        """Singleton to avoid loading model multiple times."""
        if cls._instance is None:
            cls._instance = cls(model_name)
        return cls._instance

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("embeddings.loading_model", model=self._model_name)
            self._model = SentenceTransformer(self._model_name)
            logger.info("embeddings.model_loaded", dim=self._model.get_sentence_embedding_dimension())
        return self._model

    def embed(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed a list of texts. BGE models use 'Represent this sentence:' prefix for retrieval."""
        model = self._load_model()

        # BGE instruction prefix for better retrieval quality
        prefixed = [f"Represent this sentence: {t}" for t in texts]

        embeddings = model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query. BGE uses different prefix for queries."""
        model = self._load_model()

        # BGE query prefix
        prefixed = f"Represent this sentence for searching relevant passages: {query}"

        embedding = model.encode(
            [prefixed],
            normalize_embeddings=True,
        )

        return embedding[0].tolist()

    @property
    def dimension(self) -> int:
        return 1024
