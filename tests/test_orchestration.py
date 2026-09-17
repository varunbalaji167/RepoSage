"""
Tests for app.orchestration.graph.

Individual node functions are tested directly with real or fake inputs
(risk_assessment_node uses REAL git-derived data via a monkeypatched
get_risk_by_file_stem for speed/determinism -- matches the pattern used
in test_risk_scoring.py: mock the slow/external part, test OUR logic).

The end-to-end test at the bottom monkeypatches classify_query,
extract_target_files, and code_explanation_chain (the only pieces needing
a real API/network) and drives the ACTUAL compiled graph via build_graph().
This is the one thing individual node tests can't prove: that the
conditional edges are wired to the correct node names and both branches
genuinely converge at format_response_node.
"""

from __future__ import annotations

from app.orchestration.graph import (
    build_graph,
    code_explanation_node,
    format_response_node,
    risk_assessment_node,
    route_by_intent,
)
from app.schemas import CodeExplanationResult, RiskScore


# ---------------------------------------------------------------------------
# route_by_intent
# ---------------------------------------------------------------------------

def test_route_by_intent_picks_risk_assessment_node():
    assert route_by_intent({"intent": "risk_assessment"}) == "risk_assessment_node"


def test_route_by_intent_picks_code_explanation_node():
    assert route_by_intent({"intent": "code_explanation"}) == "code_explanation_node"


# ---------------------------------------------------------------------------
# risk_assessment_node
# ---------------------------------------------------------------------------

def _fake_risk_score(file_stem: str, risk_level: str) -> RiskScore:
    return RiskScore(
        file=file_stem, dependents=3, dependent_files=["_a", "_b", "_c"],
        commit_count=10, risk_level=risk_level, rationale="test rationale",
    )


def test_risk_assessment_node_returns_clarification_when_no_target_files():
    result = risk_assessment_node({"question": "is this risky?", "target_files": []})
    assert result["risk_assessment_result"]["needs_clarification"] is True
    assert "message" in result["risk_assessment_result"]


def test_risk_assessment_node_returns_data_for_each_target_file(monkeypatch):
    import app.orchestration.graph as graph_module
    fake_scores = {
        "_exceptions": _fake_risk_score("_exceptions", "high"),
        "_client": _fake_risk_score("_client", "low"),
    }
    monkeypatch.setattr(graph_module, "get_risk_by_file_stem", lambda: fake_scores)

    result = risk_assessment_node({"question": "x", "target_files": ["_exceptions", "_client"]})
    files = result["risk_assessment_result"]["files"]
    assert result["risk_assessment_result"]["needs_clarification"] is False
    assert [f["file"] for f in files] == ["_exceptions", "_client"]
    assert files[0]["risk_level"] == "high"


def test_risk_assessment_node_reports_unrecognized_files_instead_of_silently_dropping(monkeypatch):
    """
    Regression test: a partial match (some files known, some not) must
    still answer for what it recognized AND explicitly say what it
    didn't -- never silently narrow the answer with no explanation.
    """
    import app.orchestration.graph as graph_module
    monkeypatch.setattr(graph_module, "get_risk_by_file_stem", lambda: {"_client": _fake_risk_score("_client", "low")})

    result = risk_assessment_node({"question": "x", "target_files": ["_client", "_nonexistent"]})
    files = result["risk_assessment_result"]["files"]
    assert len(files) == 1
    assert files[0]["file"] == "_client"
    assert result["risk_assessment_result"]["unrecognized_files"] == ["_nonexistent"]


def test_risk_assessment_node_asks_for_clarification_when_nothing_recognized(monkeypatch):
    """If NONE of the mentioned files are known, this is a full clarification, same as the empty-list case."""
    import app.orchestration.graph as graph_module
    monkeypatch.setattr(graph_module, "get_risk_by_file_stem", lambda: {"_client": _fake_risk_score("_client", "low")})

    result = risk_assessment_node({"question": "x", "target_files": ["_totally_made_up"]})
    assert result["risk_assessment_result"]["needs_clarification"] is True
    assert "_totally_made_up" in result["risk_assessment_result"]["message"]


# ---------------------------------------------------------------------------
# format_response_node
# ---------------------------------------------------------------------------

def test_format_response_folds_risk_notes_into_code_explanation_answer():
    state = {
        "intent": "code_explanation",
        "code_explanation_result": CodeExplanationResult(
            answer="Client.send sends a request.",
            citations=["_client.py :: Client.send (lines 1-2)"],
            risk_notes=["Note: _client.py is HIGH risk -- test rationale"],
            risk_details=[_fake_risk_score("_client", "high")],
        ),
    }
    result = format_response_node(state)["final_output"]
    assert "Client.send sends a request." in result.answer
    assert "HIGH risk" in result.answer
    # risk_details now carries the SAME finding, structured (Step 7 change) --
    # no longer hardcoded to None just because this is the code_explanation branch.
    assert result.risk_details == [_fake_risk_score("_client", "high")]
    assert len(result.citations) == 1


def test_format_response_code_explanation_without_risk_notes():
    state = {
        "intent": "code_explanation",
        "code_explanation_result": CodeExplanationResult(answer="It sends things.", citations=[], risk_notes=[], risk_details=[]),
    }
    result = format_response_node(state)["final_output"]
    assert result.answer == "It sends things."
    # [] here means "risk WAS checked, nothing notable" -- distinct from
    # None, which is reserved for "risk wasn't computed at all" (the
    # risk_assessment clarification path).
    assert result.risk_details == []


def test_format_response_risk_assessment_clarification():
    state = {
        "intent": "risk_assessment",
        "risk_assessment_result": {"needs_clarification": True, "message": "Which file?"},
    }
    result = format_response_node(state)["final_output"]
    assert result.answer == "Which file?"
    assert result.risk_details is None
    assert result.citations == []


def test_format_response_risk_assessment_with_data():
    state = {
        "intent": "risk_assessment",
        "risk_assessment_result": {
            "needs_clarification": False,
            "files": [_fake_risk_score("_exceptions", "high").model_dump()],
        },
    }
    result = format_response_node(state)["final_output"]
    assert "_exceptions.py" in result.answer
    assert "HIGH" in result.answer
    assert result.risk_details == [_fake_risk_score("_exceptions", "high")]


def test_format_response_risk_assessment_surfaces_unrecognized_files_note():
    """The partial-match case must be VISIBLE in the final answer text, not just present in internal state."""
    state = {
        "intent": "risk_assessment",
        "risk_assessment_result": {
            "needs_clarification": False,
            "files": [_fake_risk_score("_client", "low").model_dump()],
            "unrecognized_files": ["_nonexistent"],
        },
    }
    result = format_response_node(state)["final_output"]
    assert "_nonexistent" in result.answer
    assert "didn't recognize" in result.answer


# ---------------------------------------------------------------------------
# Full graph wiring (the one thing individual node tests can't prove)
# ---------------------------------------------------------------------------

class _FakeClassification:
    def __init__(self, intent):
        self.intent = intent


class _FakeChain:
    def invoke(self, question):
        return CodeExplanationResult(answer="fake answer", citations=[], risk_notes=[], risk_details=[])


def test_graph_routes_to_code_explanation_end_to_end(monkeypatch):
    import app.orchestration.graph as graph_module
    monkeypatch.setattr(graph_module, "classify_query", lambda classifier, q: _FakeClassification("code_explanation"))
    monkeypatch.setattr(graph_module, "extract_target_files", lambda q: [])
    monkeypatch.setattr(graph_module, "get_classifier", lambda: None)
    monkeypatch.setattr(graph_module, "code_explanation_chain", _FakeChain())

    app = graph_module.build_graph()
    result = app.invoke({"question": "What does Client.send do?"})
    assert result["final_output"].intent == "code_explanation"
    assert result["final_output"].answer == "fake answer"


def test_graph_routes_to_risk_assessment_end_to_end(monkeypatch):
    import app.orchestration.graph as graph_module
    monkeypatch.setattr(graph_module, "classify_query", lambda classifier, q: _FakeClassification("risk_assessment"))
    monkeypatch.setattr(graph_module, "extract_target_files", lambda q: ["_exceptions"])
    monkeypatch.setattr(graph_module, "get_classifier", lambda: None)
    monkeypatch.setattr(
        graph_module, "get_risk_by_file_stem",
        lambda: {"_exceptions": _fake_risk_score("_exceptions", "high")},
    )

    app = graph_module.build_graph()
    result = app.invoke({"question": "How risky is _exceptions.py?"})
    assert result["final_output"].intent == "risk_assessment"
    assert result["final_output"].risk_details is not None


def test_graph_routes_to_clarification_end_to_end(monkeypatch):
    """Proves the risk_assessment branch's clarification path is reachable through the REAL compiled graph, not just the node in isolation."""
    import app.orchestration.graph as graph_module
    monkeypatch.setattr(graph_module, "classify_query", lambda classifier, q: _FakeClassification("risk_assessment"))
    monkeypatch.setattr(graph_module, "extract_target_files", lambda q: [])
    monkeypatch.setattr(graph_module, "get_classifier", lambda: None)

    app = graph_module.build_graph()
    result = app.invoke({"question": "is this risky?"})
    assert result["final_output"].risk_details is None
    assert "Which file" in result["final_output"].answer