"""Cross-encoder reranking (optional)."""

from __future__ import annotations

from typing import Protocol

from anvil.schemas.retrieval import RetrievedChunk


class Reranker(Protocol):
    def rerank(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]: ...


class CrossEncoderReranker:
    """Thin wrapper around `sentence_transformers.CrossEncoder`."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        max_length: int = 512,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is required for reranking."
            ) from e
        self.model = CrossEncoder(model_name, max_length=max_length)

    def rerank(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        pairs = [[query, c.content] for c in chunks]
        scores = self.model.predict(pairs)
        for c, s in zip(chunks, scores, strict=True):
            c.scores.rerank = float(s)
            c.score = float(s)
            c.retrieval_source = "rerank"
        return sorted(chunks, key=lambda c: c.score, reverse=True)
