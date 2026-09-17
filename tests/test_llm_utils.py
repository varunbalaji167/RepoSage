"""
Tests for app.llm_utils.

Uses the REAL langchain_google_genai exception types (ModelRateLimitError,
ServerError, ModelAuthenticationError) rather than generic fakes -- the
whole point of this module is classifying real error types into the
right strategy, so the tests need to raise the real types to prove that
classification actually works, not just that "some exception" triggers
"some behavior."
"""

from __future__ import annotations

from langchain_google_genai.chat_models import (
    ModelAuthenticationError,
    ModelRateLimitError,
    ServerError,
)

from app.errors import InternalError, LLMUnavailableError
from app.llm_utils import FallbackRunnable


class _FakeRunnable:
    """Records calls; can be set to raise a specific exception N times before succeeding, or always."""

    def __init__(self, result=None, raise_error=None, fail_count=None):
        self.result = result
        self.raise_error = raise_error
        self.fail_count = fail_count  # None = always fail; int = fail this many times then succeed
        self.call_count = 0

    def invoke(self, input):
        self.call_count += 1
        if self.raise_error is not None:
            if self.fail_count is None or self.call_count <= self.fail_count:
                raise self.raise_error
        return self.result


def test_primary_success_never_touches_fallback():
    primary = _FakeRunnable(result="primary answer")
    fallback = _FakeRunnable(result="fallback answer")
    runnable = FallbackRunnable(primary, fallback)

    result = runnable.invoke("question")
    assert result == "primary answer"
    assert fallback.call_count == 0


def test_rate_limit_falls_back_to_secondary():
    primary = _FakeRunnable(raise_error=ModelRateLimitError("quota exceeded"))
    fallback = _FakeRunnable(result="fallback answer")
    runnable = FallbackRunnable(primary, fallback)

    result = runnable.invoke("question")
    assert result == "fallback answer"
    assert primary.call_count == 1  # rate limit -> immediate fallback, no same-model retry loop


def test_misconfiguration_error_wraps_to_internal_error_not_raw_exception():
    """
    Auth/permission/invalid-request/model-not-found errors are real bugs.
    They must be WRAPPED into our own InternalError (not retryable) --
    never let the raw langchain_google_genai exception leak past this
    module, and never silently mask them with a fallback attempt.
    """
    primary = _FakeRunnable(raise_error=ModelAuthenticationError("bad API key"))
    fallback = _FakeRunnable(result="should never be reached")
    runnable = FallbackRunnable(primary, fallback)

    try:
        runnable.invoke("question")
        assert False, "expected InternalError"
    except InternalError as exc:
        assert exc.retryable is False
        assert isinstance(exc.cause, ModelAuthenticationError)  # original exception preserved for logging

    assert fallback.call_count == 0


def test_transient_server_error_retries_on_same_model_then_succeeds(monkeypatch):
    """A ServerError that clears within 3 attempts should succeed on the SAME model -- no fallback needed."""
    from app.llm_utils import _invoke_with_backoff
    monkeypatch.setattr(_invoke_with_backoff.retry, "sleep", lambda seconds: None)  # skip real backoff delays in tests

    primary = _FakeRunnable(raise_error=ServerError(500, {"error": "transient"}), fail_count=2)
    fallback = _FakeRunnable(result="should never be reached")
    runnable = FallbackRunnable(primary, fallback)

    result = runnable.invoke("question")
    assert result == primary.result
    assert primary.call_count == 3  # failed twice, succeeded on the 3rd
    assert fallback.call_count == 0


def test_transient_server_error_exhausts_retries_and_wraps_to_llm_unavailable(monkeypatch):
    """A ServerError that never clears exhausts its 3 attempts and becomes LLMUnavailableError (503-equivalent, retryable) -- not a raw exception, doesn't hang forever, doesn't silently fall back either."""
    from app.llm_utils import _invoke_with_backoff
    monkeypatch.setattr(_invoke_with_backoff.retry, "sleep", lambda seconds: None)  # skip real backoff delays in tests

    primary = _FakeRunnable(raise_error=ServerError(500, {"error": "persistent"}))  # always fails
    fallback = _FakeRunnable(result="should never be reached")
    runnable = FallbackRunnable(primary, fallback)

    try:
        runnable.invoke("question")
        assert False, "expected LLMUnavailableError"
    except LLMUnavailableError as exc:
        assert exc.retryable is True
        assert isinstance(exc.cause, ServerError)

    assert primary.call_count == 3  # stop_after_attempt(3)
    assert fallback.call_count == 0


def test_both_primary_and_fallback_rate_limited_wraps_to_llm_unavailable():
    """If the fallback is ALSO rate-limited, that's genuinely 'we have no working model right now' -- LLMUnavailableError, not a silent failure or infinite fallback chain."""
    primary = _FakeRunnable(raise_error=ModelRateLimitError("primary quota exceeded"))
    fallback = _FakeRunnable(raise_error=ModelRateLimitError("fallback quota exceeded"))
    runnable = FallbackRunnable(primary, fallback)

    try:
        runnable.invoke("question")
        assert False, "expected LLMUnavailableError"
    except LLMUnavailableError as exc:
        assert exc.retryable is True
        assert isinstance(exc.cause, ModelRateLimitError)


def test_fallback_misconfiguration_wraps_to_internal_error():
    """If primary is rate-limited but the FALLBACK model itself is misconfigured (e.g. wrong model name), that's still InternalError, not LLMUnavailableError."""
    primary = _FakeRunnable(raise_error=ModelRateLimitError("primary quota exceeded"))
    fallback = _FakeRunnable(raise_error=ModelAuthenticationError("fallback misconfigured"))
    runnable = FallbackRunnable(primary, fallback)

    try:
        runnable.invoke("question")
        assert False, "expected InternalError"
    except InternalError as exc:
        assert exc.retryable is False