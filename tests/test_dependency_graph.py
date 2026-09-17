"""
Tests for app.graph_analysis.dependency_graph.build_import_graph.

We monkeypatch extract_imported_names (as bound INTO this module via
`from ... import extract_imported_names`) so these tests are deterministic
and don't depend on real file contents -- ast_parser's own correctness is
already covered by test_ast_parser.py. Here we test OUR graph-building
decisions: which imports become edges, and which are filtered out.
"""

from __future__ import annotations

import app.graph_analysis.dependency_graph as dependency_graph


def _patch_target_files(monkeypatch, target_files: set[str], imports_by_file: dict[str, list[str]]):
    monkeypatch.setattr(dependency_graph, "TARGET_FILES", target_files)

    def fake_extract(file_path):
        stem = file_path.stem
        return imports_by_file.get(stem, [])

    monkeypatch.setattr(dependency_graph, "extract_imported_names", fake_extract)


def test_edge_added_for_in_scope_import(monkeypatch):
    _patch_target_files(
        monkeypatch,
        {"_client", "_models"},
        {"_client": ["_models"], "_models": []},
    )
    graph = dependency_graph.build_import_graph()
    assert graph.has_edge("_client", "_models")


def test_self_import_does_not_create_self_loop(monkeypatch):
    """A file 'importing itself' shouldn't be possible in practice, but guard against it anyway."""
    _patch_target_files(
        monkeypatch,
        {"_client"},
        {"_client": ["_client"]},
    )
    graph = dependency_graph.build_import_graph()
    assert not graph.has_edge("_client", "_client")


def test_out_of_scope_import_is_excluded(monkeypatch):
    """Imports of files NOT in our 9-file scope (e.g. _content.py) must be silently dropped."""
    _patch_target_files(
        monkeypatch,
        {"_client", "_models"},
        {"_client": ["_content", "_models"], "_models": []},
    )
    graph = dependency_graph.build_import_graph()
    assert graph.has_edge("_client", "_models")
    assert not graph.has_node("_content")


def test_isolated_file_is_a_node_with_no_edges(monkeypatch):
    _patch_target_files(
        monkeypatch,
        {"_client", "_isolated"},
        {"_client": [], "_isolated": []},
    )
    graph = dependency_graph.build_import_graph()
    assert graph.has_node("_isolated")
    assert graph.out_degree("_isolated") == 0
    assert graph.in_degree("_isolated") == 0
