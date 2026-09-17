"""
Shared LLM call wrapper: retry-with-backoff for transient errors, and
automatic fallback to a lighter model for rate-limit/quota errors.

Design decisions:
  - Different failure TYPES get different STRATEGIES, not one blanket
    retry-everything approach:
      * ModelRateLimitError (quota exhausted) -- retrying the SAME model
        is pointless (a daily quota won't reset in seconds); instead we
        immediately switch to a lighter fallback model for THIS call.
      * ServerError (transient network/5xx blips) -- genuinely worth
        retrying a few times with exponential backoff, same model.
      * ModelAuthenticationError / ModelPermissionDeniedError /
        ModelInvalidRequestError / ModelNotFoundError -- NOT retried and
        NOT a fallback target. These represent real bugs (bad API key,
        malformed request, a model name that no longer exists) that need
        a human fix. Silently retrying or falling back would hide a real
        problem rather than surface it -- these propagate immediately.

  - Fallback is STATELESS: every call tries the primary model first,
    with no "cooldown" or "revert after N minutes" tracking needed. If
    the primary is still rate-limited on the NEXT call, it falls back
    again; the moment its quota resets, it just starts succeeding again
    on its own. No extra state to manage or get out of sync.

  - FallbackRunnable wraps two Runnables behind a single .invoke() so
    call sites (classifier.py, code_explanation_chain.py) don't need to
    change how they call it -- only how the underlying LLM is *built*.
"""

from __future__ import annotations

import logging

from langchain_google_genai.chat_models import (
    ModelAuthenticationError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ServerError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.errors import AppError, InternalError, LLMUnavailableError

logger = logging.getLogger("reposage")

FALLBACK_MODEL_NAME = "gemini-3.1-flash-lite"  # same generation as gemini-3.5-flash-lite
# (Google's July 2026 "dual launch"), with its OWN per-model quota bucket --
# confirmed via the real quota error we hit: quotaId is scoped per model.

# These represent real bugs (bad API key, malformed request, a model name
# that no longer exists) -- not transient conditions. Wrapped into
# InternalError (not retryable) rather than LLMUnavailableError, since
# retrying or falling back won't fix a misconfiguration.
_MISCONFIGURATION_ERRORS = (
    ModelAuthenticationError,
    ModelPermissionDeniedError,
    ModelInvalidRequestError,
    ModelNotFoundError,
)


@retry(
    retry=retry_if_exception_type(ServerError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def _invoke_with_backoff(runnable, input):
    """Retries ONLY on transient ServerError -- any other exception (rate limit, misconfiguration) propagates immediately, no retry."""
    return runnable.invoke(input)


def _classify_and_wrap(exc: Exception, model_label: str) -> AppError:
    """Turns any raw langchain_google_genai exception into one of our own AppError types -- nothing raw ever leaves this module."""
    if isinstance(exc, _MISCONFIGURATION_ERRORS):
        return InternalError(f"{model_label} misconfigured: {exc}", cause=exc)
    return LLMUnavailableError(f"{model_label} unavailable: {exc}", cause=exc)


class FallbackRunnable:
    """
    Wraps a primary and fallback Runnable (anything with .invoke()) behind
    one .invoke() call. On ModelRateLimitError specifically, switches to
    the fallback for that one call. EVERY other outcome (including the
    fallback itself failing, or a misconfiguration) is classified and
    wrapped into a proper AppError before leaving this class -- nothing
    raw from langchain_google_genai ever propagates further up.
    """

    def __init__(self, primary, fallback, fallback_model_name: str = FALLBACK_MODEL_NAME):
        self._primary = primary
        self._fallback = fallback
        self._fallback_model_name = fallback_model_name

    def invoke(self, input):
        try:
            return _invoke_with_backoff(self._primary, input)
        except ModelRateLimitError:
            logger.warning(
                "Primary model rate-limited; falling back to %s for this call.",
                self._fallback_model_name,
            )
            try:
                return _invoke_with_backoff(self._fallback, input)
            except Exception as exc:
                raise _classify_and_wrap(exc, "fallback model") from exc
        except Exception as exc:
            raise _classify_and_wrap(exc, "primary model") from exc