"""Vector store backed by sqlite-vec with a NumPy fallback.

The sqlite-vec backend is the production path. The NumPy fallback is used
ONLY when `sqlite_vec` is not installed at all (ImportError). Every other
failure mode — bad db path, permission error, loadable-extension support
disabled, sqlite-vec load failure — raises loudly. A silent fallback to the
NumPy backend would downgrade production retrieval quality without telling
the operator; we refuse to do that.
"""

from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from anvil import RetrievalError
from anvil.logging_config import get_logger
from anvil.schemas.document import DocumentElement

log = get_logger(__name__)


@dataclass
class VectorHit:
    element_id: str
    paragraph_ref: str
    element_type: str
    content: str
    page_number: int
    score: float  # cosine similarity in [-1, 1], higher = better


def _serialize_f32(vec: np.ndarray) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec.astype(np.float32).tolist())


class VectorStore:
    """Stores embeddings for DocumentElements and supports KNN queries."""

    def __init__(self, dim: int, db_path: str | Path = ":memory:") -> None:
        self.dim = dim
        self.db_path = str(db_path)
        self._use_sqlite_vec = self._try_load_sqlite_vec()
        # NumPy fallback state
        self._np_ids: list[str] = []
        self._np_meta: list[tuple[str, str, str, int]] = []  # (para_ref, etype, content, page)
        self._np_mat: np.ndarray | None = None

        if self._use_sqlite_vec:
            self._init_sqlite_vec_schema()

    def _try_load_sqlite_vec(self) -> bool:
        """Load sqlite-vec if installed. Fail loudly on any other error.

        Silent fallback policy: if `sqlite_vec` is not installed (`ImportError`),
        we log a loud WARNING and use the NumPy backend — that is the only
        "soft" failure permitted. Every other exception (bad db path, sqlite
        refusing to load extensions, sqlite_vec.load failing) is a broken
        environment and MUST be surfaced to the caller. Hiding it would mean
        shipping silently-degraded retrieval quality.
        """
        try:
            import sqlite_vec  # noqa: F401
        except ImportError:
            log.warning(
                "vector_store.sqlite_vec_unavailable",
                reason="sqlite_vec package not installed",
                using_backend="numpy_fallback",
                hint="install with `uv add sqlite-vec` for production retrieval",
            )
            return False

        # sqlite_vec is installed — any failure from here on is a real
        # environment problem that must not be silently masked.
        try:
            db = sqlite3.connect(self.db_path, check_same_thread=False)
            db.enable_load_extension(True)
            sqlite_vec.load(db)
            db.enable_load_extension(False)
        except Exception as exc:
            raise RetrievalError(
                f"sqlite-vec is installed but failed to load: {exc!r}. "
                f"db_path={self.db_path!r}. This is almost always a broken "
                f"environment (loadable-extension support disabled in your "
                f"Python's sqlite3 build, or a bad db_path). Fix the "
                f"environment rather than silently falling back."
            ) from exc

        self._db = db
        import threading
        self._lock = threading.Lock()
        return True

    def _init_sqlite_vec_schema(self) -> None:
        self._db.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                chunk_id TEXT PRIMARY KEY,
                embedding float[{self.dim}],
                paragraph_ref TEXT,
                element_type TEXT,
                +content TEXT,
                +page_number INTEGER
            )
            """
        )

    # ---- ingestion ---------------------------------------------------------

    def add(
        self,
        elements: list[DocumentElement],
        embeddings: np.ndarray,
    ) -> None:
        """Insert element embeddings. `embeddings` must be shape (len(elements), dim)."""
        if embeddings.shape != (len(elements), self.dim):
            raise ValueError(
                f"Embedding shape mismatch: got {embeddings.shape}, "
                f"expected ({len(elements)}, {self.dim})"
            )
        if self._use_sqlite_vec:
            with self._lock:
                for el, vec in zip(elements, embeddings, strict=True):
                    self._db.execute(
                        "INSERT OR REPLACE INTO vec_chunks(chunk_id, embedding, paragraph_ref, "
                        "element_type, content, page_number) VALUES (?, ?, ?, ?, ?, ?)",
                        [
                            el.element_id,
                            _serialize_f32(vec),
                            el.paragraph_ref or "",
                            el.element_type.value,
                            el.content,
                            el.page_number,
                        ],
                    )
                self._db.commit()
        else:
            self._np_ids.extend(e.element_id for e in elements)
            self._np_meta.extend(
                (
                    e.paragraph_ref or "",
                    e.element_type.value,
                    e.content,
                    e.page_number,
                )
                for e in elements
            )
            if self._np_mat is None:
                self._np_mat = embeddings.copy()
            else:
                self._np_mat = np.vstack([self._np_mat, embeddings])

    # ---- query -------------------------------------------------------------

    def knn(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        element_types: list[str] | None = None,
    ) -> list[VectorHit]:
        """Top-k nearest neighbors by cosine similarity."""
        if query_embedding.shape != (self.dim,):
            raise ValueError(
                f"Query shape {query_embedding.shape} != ({self.dim},)"
            )

        if self._use_sqlite_vec:
            where = ""
            params: list[object] = [_serialize_f32(query_embedding), k]
            if element_types:
                placeholders = ",".join("?" * len(element_types))
                where = f"AND element_type IN ({placeholders})"
                params.extend(element_types)
            with self._lock:
                rows = self._db.execute(
                    f"""
                    SELECT chunk_id, paragraph_ref, element_type, content,
                           page_number, distance
                    FROM vec_chunks
                    WHERE embedding MATCH ? AND k = ? {where}
                    ORDER BY distance
                    """,
                    params,
                ).fetchall()
            # sqlite-vec returns L2 distance; with normalized vecs,
            # cosine_similarity = 1 - (L2^2 / 2). Convert to be consistent.
            return [
                VectorHit(
                    element_id=r[0],
                    paragraph_ref=r[1],
                    element_type=r[2],
                    content=r[3],
                    page_number=r[4],
                    score=float(1.0 - (r[5] ** 2) / 2.0),
                )
                for r in rows
            ]

        # NumPy fallback
        if self._np_mat is None or self._np_mat.shape[0] == 0:
            return []
        scores = self._np_mat @ query_embedding  # cosine similarity
        # Apply element_type filter
        idxs = np.argsort(-scores)
        hits: list[VectorHit] = []
        for idx in idxs:
            para_ref, etype, content, page = self._np_meta[idx]
            if element_types and etype not in element_types:
                continue
            hits.append(
                VectorHit(
                    element_id=self._np_ids[idx],
                    paragraph_ref=para_ref,
                    element_type=etype,
                    content=content,
                    page_number=page,
                    score=float(scores[idx]),
                )
            )
            if len(hits) >= k:
                break
        return hits
