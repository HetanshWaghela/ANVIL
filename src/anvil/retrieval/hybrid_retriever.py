"""Hybrid retrieval: BM25 + vector + graph expansion + RRF fusion."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np

from anvil import RetrievalError
from anvil.knowledge.graph_store import GraphStore
from anvil.logging_config import get_logger
from anvil.retrieval.embedder import _STOPWORDS, Embedder
from anvil.retrieval.graph_retriever import GraphRetriever
from anvil.retrieval.vector_store import VectorStore
from anvil.schemas.document import DocumentElement
from anvil.schemas.retrieval import HybridScores, RetrievalQuery, RetrievedChunk

log = get_logger(__name__)

_CODE_REF_PATTERN = (
    r"(?:[A-Z]{1,5}-\d+(?:\.\d+)?(?:-\d+)?(?:\([a-z0-9]+\))?"
    r"|[0-9]{1,2}-[0-9]+(?:\.\d+)?(?:\([a-z]\))?)"
)
_QUERY_TABLE_REF_RE = re.compile(rf"\bTable\s+({_CODE_REF_PATTERN})\b", re.IGNORECASE)
_QUERY_PARA_REF_RE = re.compile(rf"\b({_CODE_REF_PATTERN})\b", re.IGNORECASE)


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]], k: int = 60
) -> dict[str, float]:
    """RRF (Cormack et al., 2009): score = Σ 1 / (k + rank_i).

    k=60 is the recommended constant. Higher k smooths scores across
    lists; lower k emphasizes top-ranked items. Returns a dict of
    element_id → fused RRF score, sorted descending by score.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


class HybridRetriever:
    """End-to-end retrieval: BM25 + vector → RRF → graph expansion → (optional) rerank."""

    def __init__(
        self,
        elements: list[DocumentElement],
        embedder: Embedder,
        vector_store: VectorStore,
        graph_store: GraphStore,
        rrf_k: int = 60,
        graph_hops: int = 1,
        mode: str = "hybrid",
    ) -> None:
        """`mode` selects which retrieval signals participate in fusion.

        Recognized values (see `evaluation/ablation.py`):

        - ``"hybrid"`` — baseline: BM25 + vector + graph expansion + RRF.
        - ``"bm25_only"`` — drop the vector list before RRF.
        - ``"vector_only"`` — drop the BM25 list before RRF.
        - ``"hybrid_no_graph"`` — keep BM25+vector but disable graph expansion.

        Any other value raises at construction so a typo is caught at
        the configuration site (a silent fallback would let an ablation
        run as the baseline and silently misreport its result).
        """
        valid_modes = {"hybrid", "bm25_only", "vector_only", "hybrid_no_graph"}
        if mode not in valid_modes:
            raise RetrievalError(
                f"Unknown HybridRetriever mode {mode!r}. Valid: {sorted(valid_modes)}."
            )
        self.elements = elements
        self.element_by_id: dict[str, DocumentElement] = {e.element_id: e for e in elements}
        self.embedder = embedder
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.rrf_k = rrf_k
        self.graph_hops = graph_hops
        self.mode = mode

        # Build BM25 index
        self._corpus_texts = [self._element_text(e) for e in elements]
        self._bm25 = self._init_bm25(self._corpus_texts)

        # Build a vocabulary of "content" tokens from the corpus for the
        # OOD lexical-overlap check.
        self._corpus_vocab: set[str] = set()
        for text in self._corpus_texts:
            self._corpus_vocab |= _content_tokens(text)

        # Index of all chunks for graph expansion (needs RetrievedChunk shape)
        self._chunk_index: dict[str, RetrievedChunk] = {
            e.element_id: self._element_to_chunk(e, score=0.0, source="graph")
            for e in elements
        }
        self._graph_retriever = GraphRetriever(
            graph_store=graph_store,
            element_index=self._chunk_index,
            max_hops=graph_hops,
        )

    # ---- core API ----------------------------------------------------------

    def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        """Run the full hybrid pipeline for a single query."""
        # 1) Vector search (skipped for bm25_only mode — keeps the
        #    vector-similarity signal at zero so the OOD lexical guard
        #    below does not also disable the ablation accidentally).
        if self.mode == "bm25_only":
            vec_ranked: list[str] = []
            vec_scores: dict[str, float] = {}
        else:
            q_vec = self.embedder.encode([query.text])[0]
            vec_k = max(query.top_k * 5, 20)
            vec_hits = self.vector_store.knn(q_vec, k=vec_k)
            vec_ranked = [h.element_id for h in vec_hits]
            vec_scores = {h.element_id: h.score for h in vec_hits}

        # 2) BM25 search (skipped for vector_only mode).
        if self.mode == "vector_only":
            bm25_ranked: list[str] = []
            bm25_scores: dict[str, float] = {}
        else:
            bm25_ranked, bm25_scores = self._bm25_search(
                query.text, top_k=max(query.top_k * 5, 20)
            )

        # 3) RRF fusion (normalized so top chunk scores 1.0 for a meaningful
        #    relevance threshold in the refusal gate). Single-list ablations
        #    (bm25_only / vector_only) RRF-fuse over a single ranked list,
        #    which produces ranks identical to that list — what we want.
        ranked_lists = [r for r in (vec_ranked, bm25_ranked) if r]
        fused = reciprocal_rank_fusion(ranked_lists, k=self.rrf_k)
        max_fused = max(fused.values(), default=0.0) or 1.0
        # We cap the final score by two signals to detect OOD queries:
        #   (a) top vector cosine similarity (semantic signal)
        #   (b) lexical overlap: fraction of non-stopword query tokens that
        #       actually appear in the corpus vocabulary.
        # Hash-based embedders produce false positives from hash collisions,
        # so (b) is the hard guard: if zero query terms are in the corpus,
        # we KNOW the query is out of domain regardless of vector similarity.
        top_vec_sim = max(vec_scores.values(), default=0.0)
        q_tokens = _content_tokens(query.text)
        overlap_frac = (
            len(q_tokens & self._corpus_vocab) / len(q_tokens) if q_tokens else 0.0
        )
        # In bm25_only mode there's no vector signal — fall back to
        # lexical overlap as the sole OOD guard. Otherwise the cap
        # would always be 0 and ablation runs would falsely "refuse"
        # every query.
        if self.mode == "bm25_only":
            signal_cap = max(0.0, min(1.0, overlap_frac))
        else:
            signal_cap = max(0.0, min(1.0, top_vec_sim * overlap_frac))
        fused_ids = list(fused.keys())[: query.top_k]
        seeds: list[RetrievedChunk] = []
        for eid in fused_ids:
            el = self.element_by_id.get(eid)
            if el is None:
                continue
            norm_score = (fused[eid] / max_fused) * signal_cap
            chunk = self._element_to_chunk(el, score=norm_score, source="hybrid")
            chunk.scores = HybridScores(
                bm25=bm25_scores.get(eid, 0.0),
                vector=vec_scores.get(eid, 0.0),
                graph_hops=0,
                fused=norm_score,
            )
            seeds.append(chunk)

        # 4) Graph expansion (gated by query flag AND retriever mode).
        graph_enabled = (
            query.enable_graph_expansion and self.mode != "hybrid_no_graph"
        )
        if graph_enabled and seeds:
            expanded = self._graph_retriever.expand(seeds)
            # Preserve rich scores from seeds; graph-only chunks keep their source
            result = expanded[: max(query.top_k, 10)]
        else:
            result = seeds[: query.top_k]

        result = self._promote_exact_reference_matches(query.text, result)

        # 5) Element type filter (post)
        if query.element_type_filter:
            etypes = set(query.element_type_filter)
            result = [c for c in result if c.element_type in etypes]

        return result

    # ---- helpers -----------------------------------------------------------

    def _promote_exact_reference_matches(
        self, query_text: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Promote elements whose canonical ref is explicitly named in the query.

        Hybrid lexical/vector search can rank a nearby heading above the exact
        table or paragraph mentioned by the user. In standards retrieval,
        explicit references such as "Table UW-12" and "UG-27" are high-precision
        identifiers, so they should be honored before softer semantic signals.
        """
        table_refs = {m.group(1).upper() for m in _QUERY_TABLE_REF_RE.finditer(query_text)}
        para_refs = {m.group(1).upper() for m in _QUERY_PARA_REF_RE.finditer(query_text)}
        if not table_refs and not para_refs:
            return chunks

        existing = {chunk.element_id for chunk in chunks}
        promoted: list[RetrievedChunk] = []
        seen: set[str] = set()
        max_score = max((chunk.score for chunk in chunks), default=1.0)

        for el in self.elements:
            if not (el.table and el.table.table_id.upper() in table_refs):
                continue
            chunk = self._element_to_chunk(el, score=max(max_score, 1.0), source="exact_ref")
            promoted.append(chunk)
            seen.add(chunk.element_id)

        for el in self.elements:
            is_exact_section = (
                el.element_type.value == "section"
                and el.paragraph_ref is not None
                and el.paragraph_ref.upper() in para_refs
            )
            if not is_exact_section or el.element_id in seen:
                continue
            chunk = self._element_to_chunk(el, score=max(max_score, 1.0), source="exact_ref")
            promoted.append(chunk)
            seen.add(chunk.element_id)

        for chunk in chunks:
            if chunk.element_id in seen:
                continue
            if chunk.element_id in existing:
                promoted.append(chunk)
        return promoted

    @staticmethod
    def _element_text(el: DocumentElement) -> str:
        parts = [el.title or "", el.paragraph_ref or "", el.content]
        return " ".join(p for p in parts if p)

    @staticmethod
    def _element_to_chunk(
        el: DocumentElement, score: float, source: str
    ) -> RetrievedChunk:
        return RetrievedChunk(
            element_id=el.element_id,
            paragraph_ref=el.paragraph_ref,
            element_type=el.element_type.value,
            content=el.content,
            page_number=el.page_number,
            score=score,
            retrieval_source=source,
        )

    def _init_bm25(self, corpus: list[str]) -> tuple[str, Any]:
        """Initialize BM25 via bm25s, failing loudly on unexpected errors.

        Silent fallback policy mirrors VectorStore: a missing `bm25s` package
        is a soft failure (log + use in-repo fallback). An installed `bm25s`
        that raises at index time is a real bug — propagate it. The fallback
        BM25 implementation is noticeably slower and does not use the same
        tokenization as `bm25s`, so silently swapping would give subtly
        different retrieval results in production.
        """
        try:
            import bm25s
        except ImportError:  # pragma: no cover
            log.warning(
                "hybrid_retriever.bm25s_unavailable",
                reason="bm25s package not installed",
                using_backend="pure_python_fallback",
                hint="install with `uv add bm25s` for production quality",
            )
            return ("fallback", _FallbackBM25(corpus))

        try:
            retriever = bm25s.BM25()
            retriever.index(bm25s.tokenize(corpus, stopwords="en"))
        except Exception as exc:
            raise RetrievalError(
                f"bm25s is installed but indexing failed: {exc!r}. "
                f"This is a bug (or a malformed corpus) — fix the input or "
                f"the environment, do not silently degrade the index."
            ) from exc
        return ("bm25s", retriever)

    def _bm25_search(self, query: str, top_k: int) -> tuple[list[str], dict[str, float]]:
        kind, retriever = self._bm25
        ids: list[str] = []
        scores: dict[str, float] = {}
        if kind == "bm25s":
            import bm25s

            tokens = bm25s.tokenize([query], stopwords="en")
            results, sc = retriever.retrieve(
                tokens, k=min(top_k, len(self._corpus_texts))
            )
            # results is (1, k) of corpus indices; sc (1, k) of scores
            for idx, score in zip(
                results[0].tolist(), sc[0].tolist(), strict=True
            ):
                eid = self.elements[int(idx)].element_id
                ids.append(eid)
                scores[eid] = float(score)
        else:  # pragma: no cover - used only if bm25s missing
            ranking = retriever.search(query, top_k=top_k)
            for idx, s in ranking:
                eid = self.elements[idx].element_id
                ids.append(eid)
                scores[eid] = s
        return ids, scores


_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]+")


def _content_tokens(text: str) -> set[str]:
    """Return the set of non-stopword alphanumeric tokens (len≥2) from `text`."""
    return {
        t
        for t in (m.group(0).lower() for m in _TOKEN_RE.finditer(text))
        if len(t) >= 2 and t not in _STOPWORDS
    }


class _FallbackBM25:  # pragma: no cover - only used when bm25s is absent
    """Tiny BM25 implementation as a dependency-free fallback."""

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        import math

        self.k1 = k1
        self.b = b
        self.tokens = [self._tokenize(c) for c in corpus]
        self.doc_len = [len(t) for t in self.tokens]
        self.avgdl = float(np.mean(self.doc_len)) if self.doc_len else 0.0
        self.N = len(self.tokens)
        self.df: dict[str, int] = {}
        for toks in self.tokens:
            for t in set(toks):
                self.df[t] = self.df.get(t, 0) + 1
        self.idf = {
            t: math.log((self.N - d + 0.5) / (d + 0.5) + 1)
            for t, d in self.df.items()
        }

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        import re

        return re.findall(r"[A-Za-z0-9\-]+", text.lower())

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        q_tokens = self._tokenize(query)
        scores = np.zeros(self.N, dtype=np.float32)
        for i, doc in enumerate(self.tokens):
            if not doc:
                continue
            tf: dict[str, int] = {}
            for t in doc:
                tf[t] = tf.get(t, 0) + 1
            score = 0.0
            for qt in q_tokens:
                if qt not in tf:
                    continue
                f = tf[qt]
                idf = self.idf.get(qt, 0.0)
                denom = f + self.k1 * (
                    1 - self.b + self.b * (self.doc_len[i] / (self.avgdl or 1))
                )
                score += idf * ((f * (self.k1 + 1)) / denom)
            scores[i] = score
        idxs = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in idxs if scores[i] > 0]
