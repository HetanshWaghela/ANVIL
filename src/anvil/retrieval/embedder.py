"""Embedding backends.

Two implementations are provided:

1. `SentenceTransformerEmbedder` — the production backend. Requires a one-time
   model download (~130MB for BAAI/bge-small-en-v1.5 by default).
2. `DeterministicHashEmbedder` — a hash-based embedder with no dependencies.
   Used in tests and in environments without model downloads available. Its
   embeddings are deterministic and roughly lexically-sensitive (token hashing
   + normalization), good enough to validate the retrieval pipeline end-to-end
   without real semantic quality.

Select between them via the `ANVIL_EMBEDDER` env variable
(`sentence_transformer` | `hash`). The default is `hash` for reproducibility
in CI — production code should set `ANVIL_EMBEDDER=sentence_transformer`.
The sentence-transformer model and cache directory are configured with
`ANVIL_ST_MODEL` and `ANVIL_ST_CACHE_DIR`, matching `.env.example`.
"""

from __future__ import annotations

import hashlib
import os
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """Minimal embedder interface."""

    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (N, dim) float32 array of L2-normalized embeddings."""
        ...


_STOPWORDS: frozenset[str] = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "but", "by", "do", "does", "for", "from", "have", "has", "in", "is", "it", "its", "of", "on", "or", "that", "the", "their", "this", "to", "was", "were", "will", "with", "what", "which", "who", "why", "how", "when", "where", "here", "there", "if", "then", "so", "not", "no", "just", "about", "but", "an", "any", "all", "some", "can", "could", "should", "would", "may", "might", "must", "shall", "do", "does", "done", "each", "every", "some", "such", "than", "them", "they", "us", "we", "you", "your", "my", "me", "i", "he", "she", "his", "her", "our", "today", "tomorrow", "yesterday", "now", "also", "get", "give", "got", "much", "many", "more", "most", "less", "least", "does", "done", "such", "very", "will", "would", "into", "over", "onto", "off", "upon", "among", "between", "without", "within"]
)


class DeterministicHashEmbedder:
    """Dependency-free deterministic embedder (for tests and reproducible CI).

    Tokenizes on whitespace/punctuation, hashes each token to a column in a
    fixed-width vector, applies sublinear weighting, and L2-normalizes.
    Filters common English stopwords so spurious matches on "is/the/in"
    don't make out-of-domain queries appear relevant.
    Surprisingly effective for keyword-heavy engineering text.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _tokens(self, text: str) -> list[str]:
        out: list[str] = []
        buf: list[str] = []
        for ch in text.lower():
            if ch.isalnum() or ch == "-":
                buf.append(ch)
            else:
                if buf:
                    out.append("".join(buf))
                    buf = []
        if buf:
            out.append("".join(buf))
        return out

    def _hash_index(self, token: str) -> int:
        h = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
        return int.from_bytes(h, "little") % self.dim

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            toks = self._tokens(text)
            if not toks:
                continue
            # Term frequency with sublinear scaling; drop stopwords and 1-char tokens
            counts: dict[int, float] = {}
            for tok in toks:
                if len(tok) < 2 or tok in _STOPWORDS:
                    continue
                idx = self._hash_index(tok)
                counts[idx] = counts.get(idx, 0.0) + 1.0
            for idx, c in counts.items():
                vecs[i, idx] = 1.0 + np.log(c)
            # L2 normalize so dot product = cosine similarity
            norm = float(np.linalg.norm(vecs[i]))
            if norm > 0:
                vecs[i] /= norm
        return vecs


class SentenceTransformerEmbedder:
    """Thin wrapper around sentence-transformers for production quality."""

    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", cache_folder: str | None = None
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is required. "
                "Install with `uv add sentence-transformers`."
            ) from e
        self.model = SentenceTransformer(model_name, cache_folder=cache_folder)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        # sentence-transformers is intentionally untyped in mypy config →
        # explicit annotation keeps the public signature honest.
        result: np.ndarray = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)
        return result


def get_default_embedder() -> Embedder:
    """Return the embedder selected by `ANVIL_EMBEDDER` env var.

    Recognized values:

      * `hash` (default) — `DeterministicHashEmbedder`, zero-dependency, no
        semantic quality. Logs a WARNING: this is only for CI/tests.
      * `sentence_transformer` — `SentenceTransformerEmbedder`. If the
        import fails we raise instead of silently falling back to hash —
        a prod deploy that wanted real embeddings must fail loudly.

    Any other value raises `RetrievalError`.
    """
    from anvil import RetrievalError
    from anvil.logging_config import get_logger

    log = get_logger(__name__)
    choice = os.environ.get("ANVIL_EMBEDDER", "hash").lower().strip()
    if choice == "sentence_transformer":
        model_name = os.environ.get(
            "ANVIL_ST_MODEL",
            os.environ.get("ANVIL_EMBEDDER_MODEL", "BAAI/bge-small-en-v1.5"),
        )
        cache_dir = os.environ.get("ANVIL_ST_CACHE_DIR")
        cache_folder = os.path.expanduser(cache_dir) if cache_dir else None
        log.info("embedder.selected", embedder="sentence_transformer", model=model_name)
        return SentenceTransformerEmbedder(
            model_name=model_name,
            cache_folder=cache_folder,
        )  # pragma: no cover
    if choice != "hash":
        raise RetrievalError(
            f"Unknown ANVIL_EMBEDDER={choice!r}. Supported: hash, sentence_transformer."
        )
    log.warning(
        "embedder.hash_selected",
        hint=(
            "DeterministicHashEmbedder is in use — lexical-only, no semantic "
            "quality. Set ANVIL_EMBEDDER=sentence_transformer for production."
        ),
    )
    return DeterministicHashEmbedder()
