"""FastAPI middleware: structured request logging."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response

from anvil.logging_config import get_logger

log = get_logger(__name__)


async def structured_log_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Log every request with method, path, status, and duration in ms."""
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:  # noqa: BLE001
        duration_ms = (time.perf_counter() - start) * 1000
        log.error(
            "http.error",
            method=request.method,
            path=request.url.path,
            error=str(exc),
            duration_ms=round(duration_ms, 2),
        )
        raise
    duration_ms = (time.perf_counter() - start) * 1000
    log.info(
        "http.request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=round(duration_ms, 2),
    )
    return response
