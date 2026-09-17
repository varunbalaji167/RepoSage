"""
Step 9 — genuine end-to-end test.

Unlike every other test in this suite (which uses fakes/mocks
deliberately, for speed and independence from real infra), THIS test hits
the real system: the real Chroma index, the real call graph, the real
Gemini API, through the real (non-overridden) FastAPI app.

This is the one thing no amount of mocked testing can ever catch: whether
our code's ASSUMPTIONS about an external dependency's actual behavior are
correct. Mocks only prove "if the dependency behaves like I assumed, my
code handles it correctly" -- they can't catch a WRONG assumption. We've
been bitten by exactly this before, more than once: Chroma's real `where`
clause needing $and (not a flat dict), a Gemini model name we assumed was
current but had been deprecated, a `temperature` parameter that worked
fine until it silently stopped mattering. A real end-to-end run is the
only thing that would have caught any of those.

Opt-in only: skipped by default (`pytest tests/` never touches your real
API key, Chroma index, or LLM quota). Run explicitly with:
    RUN_E2E=1 pytest tests/test_e2e.py -v

Also a good moment to confirm LangSmith tracing is wired up: if
LANGCHAIN_TRACING_V2=true, LANGCHAIN_API_KEY, and LANGCHAIN_PROJECT are
set in your .env, every run of this test produces a full trace tree at
smith.langchain.com -- no code changes needed for that beyond the env
vars, since LCEL Runnables and LangGraph both run through the same
callback-manager system tracing plugs into automatically.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from app.main import app

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_E2E"),
    reason="Real end-to-end test -- costs real LLM quota and needs a live API key. Run explicitly with RUN_E2E=1.",
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_code_explanation_end_to_end(client):
    """Real retrieval + real LLM generation. Confirms the whole retrieval->call-graph-expansion->LLM->citation pipeline actually works together, not just each piece in isolation."""
    response = client.post("/ask", json={"question": "What does Client.request do?"})
    assert response.status_code == 200

    body = response.json()
    assert body["intent"] == "code_explanation"
    assert len(body["answer"]) > 0
    assert len(body["citations"]) > 0  # real retrieval must have actually found something
    assert "Client.request" not in body["answer"] or "not included" not in body["answer"].lower()
    # ^ regression guard for the real bug found in Step 8: this exact
    # question used to fail to retrieve Client.request at all.


def test_risk_assessment_end_to_end(client):
    """Real risk computation (git log + import graph) for a known file. Result should match our confirmed real data table."""
    response = client.post("/ask", json={"question": "How risky is it to change _exceptions.py?"})
    assert response.status_code == 200

    body = response.json()
    assert body["intent"] == "risk_assessment"
    assert body["risk_details"] is not None
    assert body["risk_details"][0]["file"] == "_exceptions"
    assert body["risk_details"][0]["risk_level"] == "high"  # matches our confirmed real data table


def test_clarification_path_end_to_end(client):
    """No file mentioned -- must ask for clarification, not guess or dump all 9 files' scores."""
    response = client.post("/ask", json={"question": "Is it risky to change anything?"})
    assert response.status_code == 200

    body = response.json()
    assert body["intent"] == "risk_assessment"
    assert body["risk_details"] is None
    assert "which file" in body["answer"].lower()