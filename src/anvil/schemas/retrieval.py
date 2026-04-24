"""Retrieval query/result schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievalQuery(BaseModel):
    """A query sent through the retrieval pipeline."""

    text: str
    top_k: int = Field(default=10, ge=1, le=100)
    element_type_filter: list[str] | None = Field(
        default=None, description="Optional filter on ElementType values"
    )
    enable_graph_expansion: bool = True
    enable_rerank: bool = False


class HybridScores(BaseModel):
    """Component scores produced by the hybrid retriever before fusion."""

    bm25: float = 0.0
    vector: float = 0.0
    graph_hops: int = Field(default=0, ge=0)
    rerank: float | None = None
    fused: float = 0.0


class RetrievedChunk(BaseModel):
    """A retrieved document element with provenance and scoring info."""

    element_id: str
    paragraph_ref: str | None = None
    element_type: str
    content: str
    page_number: int
    score: float = Field(description="Primary score used for ranking (fused score)")
    scores: HybridScores = Field(default_factory=HybridScores)
    retrieval_source: str = Field(
        default="hybrid",
        description="'bm25' | 'vector' | 'graph' | 'hybrid' | 'rerank'",
    )

    def covers(self, entity_ref: str) -> bool:
        """Return True if this chunk can plausibly cover the given entity ref."""
        if self.paragraph_ref and entity_ref.lower() in self.paragraph_ref.lower():
            return True
        return entity_ref.lower() in self.content.lower()
