"""
Shared pytest fixtures.

Testing philosophy for this suite (read this before adding new tests):
  - We test OUR logic: chunking decisions, call resolution algorithms
    (CHA, super(), constructors), risk classification, reassembly, and
    citation/risk-note wiring.
  - We do NOT test that third-party libraries work (ast.parse parses
    Python correctly, networkx.DiGraph.successors() returns successors,
    chromadb stores what you put in it, pydantic validates a BaseModel).
    That's the library maintainers' job, already covered by their own
    test suites -- re-testing it here would be redundant weight that
    doesn't catch OUR bugs.
  - Where a function reads real files (build_call_graph, build_import_graph,
    compute_risk_scores), we point it at small synthetic temp files via
    monkeypatch rather than depending on the real httpx clone -- keeps
    tests fast, deterministic, and independent of the target repo's
    contents ever changing.
"""

from __future__ import annotations

import pytest


class FakeCollection:
    """
    Minimal stand-in for a chromadb.Collection, implementing only the two
    methods code_explanation_chain.py actually calls (.get, .query), with
    the same return shapes real Chroma uses.

    Constructed from a flat list of (id, document_text, metadata) tuples --
    tests populate this directly instead of needing a real embedding model.
    """

    def __init__(self, rows: list[tuple[str, str, dict]], query_results: list[tuple[str, str, dict]] | None = None):
        self._rows = rows
        # query_results lets a test say "THESE specific rows are the top-k
        # semantic match" independently of what .get() can find -- real
        # Chroma ranks by embedding similarity, which we can't fake
        # meaningfully, so tests specify the intended ranking directly.
        # Defaults to `rows` unchanged if not given.
        self._query_results = query_results if query_results is not None else rows

    def get(self, where: dict | None = None) -> dict:
        matched = [r for r in self._rows if self._matches(r[2], where)]
        return {
            "ids": [r[0] for r in matched],
            "documents": [r[1] for r in matched],
            "metadatas": [r[2] for r in matched],
        }

    def query(self, query_texts: list[str], n_results: int) -> dict:
        top = self._query_results[:n_results]
        return {
            "documents": [[r[1] for r in top]],
            "metadatas": [[r[2] for r in top]],
        }

    @staticmethod
    def _matches(metadata: dict, where: dict | None) -> bool:
        if where is None:
            return True
        if "$and" in where:
            return all(FakeCollection._matches(metadata, cond) for cond in where["$and"])
        return all(metadata.get(k) == v for k, v in where.items())


@pytest.fixture
def fake_collection():
    """Returns a FakeCollection factory so each test builds its own rows."""
    return FakeCollection
