"""
Tests for app.main / app.api.routes.

Uses FastAPI's dependency_overrides to swap in a fake graph for get_graph
-- this is the idiomatic FastAPI testing pattern for exactly this
situation, cleaner than monkeypatching internals. The real lifespan still
runs (it's cheap and side-effect-free, per Steps 5-7's lazy-loading), but
the override means the real graph is never actually invoked in these
tests -- no API key or network access needed.

We test what's actually OUR wiring here: does a valid request reach the
graph and return its output correctly shaped, does an invalid request
get rejected with 422 before ever reaching the graph, and does an
unexpected graph failure turn into a clean 500 rather than leaking
internals. We don't test that FastAPI's own validation machinery works
-- that's the framework's job, already covered by its own test suite.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.routes import get_graph
from app.main import app
from app.schemas import QueryResponse


class _FakeGraph:
    def __init__(self, output: QueryResponse | None = None, raise_error: bool = False):
        self._output = output
        self._raise_error = raise_error
        self.last_input = None

    def invoke(self, input_dict):
        self.last_input = input_dict
        if self._raise_error:
            raise RuntimeError("simulated failure")
        return {"final_output": self._output}


def test_ask_returns_response_model_shape():
    fake_output = QueryResponse(intent="code_explanation", answer="It sends things.", citations=["_client.py :: send"])
    fake_graph = _FakeGraph(output=fake_output)
    app.dependency_overrides[get_graph] = lambda: fake_graph

    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "What does send do?"})

    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "code_explanation"
    assert body["answer"] == "It sends things."
    assert body["risk_details"] is None
    assert fake_graph.last_input == {"question": "What does send do?"}


def test_ask_rejects_empty_question():
    fake_graph = _FakeGraph(output=QueryResponse(intent="code_explanation", answer="x"))
    app.dependency_overrides[get_graph] = lambda: fake_graph

    with TestClient(app) as client:
        response = client.post("/ask", json={"question": ""})

    app.dependency_overrides.clear()
    assert response.status_code == 422


def test_ask_rejects_missing_question_field():
    fake_graph = _FakeGraph(output=QueryResponse(intent="code_explanation", answer="x"))
    app.dependency_overrides[get_graph] = lambda: fake_graph

    with TestClient(app) as client:
        response = client.post("/ask", json={})

    app.dependency_overrides.clear()
    assert response.status_code == 422


def test_ask_returns_clean_500_for_unclassified_failure():
    """An UNEXPECTED, unclassified failure (not one of our AppError types) must become a clean, structured 500 -- not leak the raw exception to the client."""
    fake_graph = _FakeGraph(raise_error=True)
    app.dependency_overrides[get_graph] = lambda: fake_graph

    # raise_server_exceptions=False: TestClient's default (True) re-raises
    # exceptions caught only by the bare Exception handler into the TEST
    # process itself (a debugging convenience) -- specific AppError-type
    # handlers don't have this quirk (see the passing tests above), only
    # the catch-all does. False here gets the REAL response a production
    # client would actually receive.
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/ask", json={"question": "anything"})

    app.dependency_overrides.clear()
    assert response.status_code == 500
    assert "simulated failure" not in response.text  # the raw exception message must not leak
    body = response.json()
    assert body["error_code"] == "internal_error"
    assert body["retryable"] is False


def test_ask_maps_retrieval_error_to_503():
    """A RetrievalError (vector store down) must map to 503, not a generic 500 -- this is a degraded dependency, not a bug."""
    from app.errors import RetrievalError

    class _RaisingGraph:
        def invoke(self, input_dict):
            raise RetrievalError("vector store unavailable")

    app.dependency_overrides[get_graph] = lambda: _RaisingGraph()

    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "anything"})

    app.dependency_overrides.clear()
    assert response.status_code == 503
    body = response.json()
    assert body["error_code"] == "retrieval_unavailable"
    assert body["retryable"] is True


def test_ask_maps_internal_error_to_500_not_retryable():
    """An InternalError (e.g. LLM misconfiguration) must map to 500 AND be marked non-retryable -- distinct from a 503 service issue."""
    from app.errors import InternalError

    class _RaisingGraph:
        def invoke(self, input_dict):
            raise InternalError("bad API key configured")

    app.dependency_overrides[get_graph] = lambda: _RaisingGraph()

    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "anything"})

    app.dependency_overrides.clear()
    assert response.status_code == 500
    body = response.json()
    assert body["error_code"] == "internal_error"
    assert body["retryable"] is False


def test_ask_with_risk_details_populated():
    """Confirms risk_assessment-shaped responses (with nested RiskScore data) serialize correctly through the API."""
    from app.schemas import RiskScore

    score = RiskScore(file="_exceptions", dependents=5, dependent_files=["_api"], commit_count=30,
                       risk_level="high", rationale="test")
    fake_output = QueryResponse(intent="risk_assessment", answer="HIGH risk", risk_details=[score])
    fake_graph = _FakeGraph(output=fake_output)
    app.dependency_overrides[get_graph] = lambda: fake_graph

    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "How risky is _exceptions.py?"})

    app.dependency_overrides.clear()
    assert response.status_code == 200
    body = response.json()
    assert body["risk_details"][0]["file"] == "_exceptions"
    assert body["risk_details"][0]["risk_level"] == "high"