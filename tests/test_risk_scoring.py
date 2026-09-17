"""
Tests for app.graph_analysis.risk_scoring.compute_risk_scores.

We monkeypatch build_import_graph and get_commit_count (both bound into
this module) so tests are deterministic and don't depend on git history or
real file imports -- those are covered by test_dependency_graph.py and are
also just slow (git log) to run repeatedly. Here we test OUR classification
logic: the impact-gate-then-churn decision (explicitly NOT a weighted
formula, per the design decision), median-based thresholds, and sort order.
"""

from __future__ import annotations

import networkx as nx

from app.errors import RiskDataError

import app.graph_analysis.risk_scoring as risk_scoring


def _patch(monkeypatch, graph: nx.DiGraph, commits: dict[str, int], target_files: set[str]):
    monkeypatch.setattr(risk_scoring, "TARGET_FILES", target_files)
    monkeypatch.setattr(risk_scoring, "build_import_graph", lambda: graph)
    monkeypatch.setattr(risk_scoring, "get_commit_count", lambda stem: commits[stem])


def _make_graph(edges: list[tuple[str, str]], nodes: set[str]) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


def test_high_impact_low_churn_is_classified_high(monkeypatch):
    """Many dependents + few commits = HIGH risk (impact gate passes, churn stays low)."""
    files = {"_types", "_client"}
    # _types has 1 dependent (_client), _client has 0 -- with only 2 files,
    # median dependents = 0.5, so _types (1 dependent) clears the high-impact bar.
    graph = _make_graph([("_client", "_types")], files)
    _patch(monkeypatch, graph, {"_types": 5, "_client": 100}, files)

    scores = {s.file: s for s in risk_scoring.compute_risk_scores()}
    assert scores["_types"].risk_level == "high"


def test_high_impact_high_churn_is_medium(monkeypatch):
    """Many dependents but ALSO frequently changed = MEDIUM, not HIGH (well-exercised code)."""
    files = {"_types", "_client"}
    graph = _make_graph([("_client", "_types")], files)
    # Commits median across {5, 100} note we need churn to be ABOVE median to avoid "low_churn" gate.
    _patch(monkeypatch, graph, {"_types": 100, "_client": 5}, files)

    scores = {s.file: s for s in risk_scoring.compute_risk_scores()}
    assert scores["_types"].risk_level == "medium"


def test_low_impact_is_classified_low(monkeypatch):
    """Few/no dependents = LOW risk regardless of churn."""
    files = {"_types", "_client"}
    graph = _make_graph([("_client", "_types")], files)
    _patch(monkeypatch, graph, {"_types": 5, "_client": 5}, files)

    scores = {s.file: s for s in risk_scoring.compute_risk_scores()}
    assert scores["_client"].risk_level == "low"


def test_dependent_files_field_matches_actual_ancestors(monkeypatch):
    files = {"_types", "_client", "_api"}
    graph = _make_graph([("_client", "_types"), ("_api", "_client")], files)
    _patch(monkeypatch, graph, {"_types": 5, "_client": 5, "_api": 5}, files)

    scores = {s.file: s for s in risk_scoring.compute_risk_scores()}
    # _types is depended on by BOTH _client (direct) and _api (transitive).
    assert scores["_types"].dependent_files == ["_api", "_client"]
    assert scores["_types"].dependents == 2


def test_results_are_sorted_high_to_low(monkeypatch):
    files = {"_types", "_client", "_api"}
    graph = _make_graph([("_client", "_types"), ("_api", "_types")], files)
    _patch(monkeypatch, graph, {"_types": 1, "_client": 100, "_api": 100}, files)

    scores = risk_scoring.compute_risk_scores()
    risk_order = {"high": 0, "medium": 1, "low": 2}
    levels = [risk_order[s.risk_level] for s in scores]
    assert levels == sorted(levels)


def test_missing_git_binary_raises_risk_data_error(monkeypatch):
    """If git itself isn't installed/on PATH, this must surface as a clear RiskDataError, not an unhandled FileNotFoundError."""
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("git not found")
    monkeypatch.setattr(risk_scoring.subprocess, "run", fake_run)

    try:
        risk_scoring.get_commit_count("_client")
        assert False, "expected RiskDataError"
    except RiskDataError as exc:
        assert "git" in exc.message.lower()
        assert exc.retryable is True  # a missing binary might be a deploy-in-progress issue -- worth letting the caller retry later


def test_failed_git_command_raises_instead_of_silently_returning_zero(monkeypatch):
    """
    Regression test for a real bug found while adding error handling:
    subprocess.run does NOT raise on a non-zero exit code by default --
    a genuinely failed git command was silently reported as "0 commits"
    instead of surfacing as an error. This could make a file wrongly
    look low-churn (and therefore possibly HIGH risk) simply because git
    failed, not because it truly has no history.
    """
    class FakeResult:
        returncode = 128
        stdout = ""
        stderr = "fatal: not a git repository"

    def fake_run(*args, **kwargs):
        return FakeResult()
    monkeypatch.setattr(risk_scoring.subprocess, "run", fake_run)

    try:
        risk_scoring.get_commit_count("_client")
        assert False, "expected RiskDataError instead of silently returning a commit count"
    except RiskDataError as exc:
        assert "not a git repository" in exc.message