"""ANVIL — Compliance-grade retrieval over engineering standards."""

from __future__ import annotations

__version__ = "0.1.0"


class AnvilError(Exception):
    """Root of the anvil exception hierarchy."""


class ParsingError(AnvilError):
    """Raised when document parsing fails."""


class RetrievalError(AnvilError):
    """Raised when retrieval fails."""


class GenerationError(AnvilError):
    """Raised when generation fails or produces invalid output."""


class CalculationError(AnvilError):
    """Raised when a deterministic calculation cannot be completed."""


class EvaluationError(AnvilError):
    """Raised when evaluation fails."""


__all__ = [
    "AnvilError",
    "CalculationError",
    "EvaluationError",
    "GenerationError",
    "ParsingError",
    "RetrievalError",
    "__version__",
]
