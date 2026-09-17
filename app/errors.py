"""
Shared error taxonomy for RepoSage.

Three-tier, industry-standard pattern:
  1. Third-party/library errors (chromadb, subprocess, langchain_google_genai,
     etc.) are NEVER let to propagate raw across a module boundary -- caught
     and WRAPPED into one of the types below, at the exact point we call
     into that library. The original exception is kept as `.cause` for
     logging, never shown to the client.
  2. A small, reusable set of APP-WIDE error types, each with a FIXED HTTP
     status code and a `retryable` flag -- used across the whole project,
     not redefined per module.
  3. Narrow, domain-specific subclasses (RetrievalError, RiskDataError,
     CallGraphError, LLMUnavailableError) for the rare cases where extra
     context matters -- each subclasses one of the common types, so
     generic handling still works via isinstance up the hierarchy without
     every consumer needing to know every narrow subtype.

This hierarchy is HTTP-agnostic on purpose: graph nodes, chains, and
scripts (invoked directly, not just via the API -- e.g. `python -m
app.orchestration.graph`) raise and catch these without any knowledge of
HTTP status codes. Only main.py's global exception handlers (the one
place with HTTP concerns) map error type -> status code.

`retryable` means "the CALLER might reasonably retry later, this isn't a
permanent mistake on their part" -- NOT "we already retried internally."
Internal retry (llm_utils.py's backoff) is separate and already exhausted
by the time one of these is raised.
"""

from __future__ import annotations

from pydantic import BaseModel


class AppError(Exception):
    """Base for every RepoSage-specific error. Never raised directly -- always one of its subclasses below."""
    http_status: int = 500
    error_code: str = "internal_error"
    retryable: bool = False

    def __init__(self, message: str, *, cause: Exception | None = None):
        super().__init__(message)
        self.message = message
        self.cause = cause  # the original library exception, if any -- for logging only, never sent to the client


# --- Common, app-wide error types -------------------------------------------

class ValidationError(AppError):
    """The request was semantically invalid in a way Pydantic's own schema validation wouldn't catch."""
    http_status = 422
    error_code = "validation_error"
    retryable = False


class ServiceUnavailableError(AppError):
    """A backend dependency is down, misconfigured, or degraded. Not the client's fault; retrying LATER might help once the dependency recovers."""
    http_status = 503
    error_code = "service_unavailable"
    retryable = True


class RateLimitedError(AppError):
    """Reserved for if/when RepoSage rate-limits its OWN callers directly (not currently used -- distinct from LLMUnavailableError, which is about an UPSTREAM provider rate-limiting us)."""
    http_status = 429
    error_code = "rate_limited"
    retryable = True


class InternalError(AppError):
    """An unexpected or misconfigured failure -- a real bug (bad API key, wrong model name, etc.), not a transient condition. NOT retryable -- retrying the same bug won't fix it."""
    http_status = 500
    error_code = "internal_error"
    retryable = False


# --- Narrow, domain-specific subclasses -------------------------------------

class RetrievalError(ServiceUnavailableError):
    """The vector store (Chroma + embedding model) failed to answer a query."""
    error_code = "retrieval_unavailable"


class RiskDataError(ServiceUnavailableError):
    """Risk scoring (git log / import graph) failed."""
    error_code = "risk_data_unavailable"


class CallGraphError(ServiceUnavailableError):
    """The function-level call graph (or class-to-file map) failed to build."""
    error_code = "call_graph_unavailable"


class LLMUnavailableError(ServiceUnavailableError):
    """Both the primary AND fallback LLM failed (llm_utils.py's FallbackRunnable exhausted every option)."""
    error_code = "llm_unavailable"


# --- HTTP-boundary response shape -------------------------------------------

class ErrorResponse(BaseModel):
    """
    Structured error body returned by the API -- built from an AppError's
    own attributes by main.py's global exception handler. Lets the
    frontend distinguish failure types (e.g. show a "try again" hint only
    when retryable=True) instead of parsing a generic string message.
    """
    error_code: str
    message: str
    retryable: bool