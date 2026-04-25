"""Structured logging configuration.

Logs are written to **stderr**, never stdout. This keeps the CLI's
machine-readable output (`anvil ... --json`) free of log noise so it can
be piped into `jq`, parsed by CI artifacts, or compared in run-logger
diffs without preprocessing.
"""

from __future__ import annotations

import logging
import os
import sys

import structlog


def configure_logging(production: bool | None = None, level: int = logging.INFO) -> None:
    """Configure structlog for console (dev) or JSON (production) output.

    Args:
        production: If True, emit JSON logs. If None, infer from ANVIL_ENV env var.
        level: Minimum log level (default INFO).
    """
    if production is None:
        production = os.environ.get("ANVIL_ENV", "").lower() == "production"

    shared: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    if production:
        processors = shared + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = shared + [structlog.dev.ConsoleRenderer()]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        # Always send logs to stderr so stdout stays available for the
        # CLI's structured (JSON) output. Without this, `anvil nim-check
        # --json | jq` breaks on the first log line.
        logger_factory=structlog.WriteLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structlog bound logger (configured lazily on first use)."""
    # `structlog.get_logger` is dynamically typed; the BoundLogger annotation
    # holds at runtime but mypy needs the explicit cast.
    logger: structlog.stdlib.BoundLogger = (
        structlog.get_logger(name) if name else structlog.get_logger()
    )
    return logger


# Configure lazily on import — tests can re-configure as needed.
configure_logging()
