"""FastAPI entrypoint. Wires modules together, contains no business logic."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.errors import AppError, ErrorResponse
from app.orchestration.graph import build_graph

logger = logging.getLogger("reposage")
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Builds the graph ONCE at server startup, not per-request. build_graph()
    itself is cheap (just wires Runnables together, no I/O) -- the
    genuinely expensive resources it depends on (Chroma + embedding model,
    the call graph, the LLM client, risk scores) are already lazy-cached
    via lru_cache (Steps 5-7) and only trigger on the FIRST real
    graph.invoke() call, not here. Using FastAPI's lifespan (rather than a
    bare module-level global) is still the right pattern: it integrates
    with the app's actual startup/shutdown lifecycle and lets tests swap
    in a fake graph cleanly via dependency overrides, without needing to
    reason about what happens at import time.
    """
    app.state.graph = build_graph()
    yield
    # No explicit shutdown cleanup needed -- the lru_cache'd resources
    # (Chroma client, LLM client, etc.) don't hold anything requiring
    # an explicit close.


app = FastAPI(title="RepoSage", lifespan=lifespan)


@app.exception_handler(AppError)
async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    """
    Global handler: catches ANY AppError (RetrievalError, RiskDataError,
    CallGraphError, LLMUnavailableError, InternalError, ...) raised
    ANYWHERE during request handling -- graph nodes, chains, routes --
    without every route needing its own try/except. This is the DRY,
    idiomatic FastAPI pattern: error->HTTP translation lives in ONE place,
    keyed off the error's OWN declared http_status/error_code/retryable,
    not duplicated per endpoint.
    """
    log_level = logging.WARNING if exc.retryable else logging.ERROR
    logger.log(log_level, "%s: %s", exc.error_code, exc.message, exc_info=exc.cause)
    return JSONResponse(
        status_code=exc.http_status,
        content=ErrorResponse(error_code=exc.error_code, message=exc.message, retryable=exc.retryable).model_dump(),
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """
    Anything NOT already one of our own AppError types is, by definition,
    a bug we didn't anticipate -- log the full traceback server-side (so
    it's debuggable), but never leak internal details to the client.
    """
    logger.exception("Unhandled, unclassified error")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error_code="internal_error",
            message="Something went wrong answering your question.",
            retryable=False,
        ).model_dump(),
    )


# Order matters: /ask must be registered BEFORE the catch-all static mount
# at "/", or the static mount would shadow it (StaticFiles with html=True
# serves index.html for any unmatched path under "/", including "/ask" if
# it were mounted first).
app.include_router(router)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")