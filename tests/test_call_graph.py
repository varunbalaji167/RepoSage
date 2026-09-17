"""
Tests for app.graph_analysis.call_graph.build_call_graph.

Unlike dependency_graph, we use REAL temp files here rather than mocking
extract_imported_names -- call resolution genuinely depends on real AST
structure (class bodies, decorators, inheritance), so faking it would mean
not actually testing the resolution logic. These scenarios mirror the
exact cases manually verified against real httpx code during Step 5a
development (CHA fan-out, super(), constructors, unresolved attribute
calls) -- now locked in as permanent regression tests.
"""

from __future__ import annotations

import app.graph_analysis.call_graph as call_graph_module
from app.errors import CallGraphError


def _build_repo(tmp_path, files: dict[str, str]):
    """Write {filename: source} into tmp_path and return it as the repo root."""
    for filename, source in files.items():
        (tmp_path / filename).write_text(source)
    return tmp_path


def _patch_repo(monkeypatch, repo_path, target_files: set[str]):
    monkeypatch.setattr(call_graph_module, "REPO_PATH", repo_path)
    monkeypatch.setattr(call_graph_module, "TARGET_FILES", target_files)


def test_self_call_within_same_class_resolves(tmp_path, monkeypatch):
    source = (
        "class Client:\n"
        "    def send(self, req):\n"
        "        return self.build(req)\n"
        "    def build(self, req):\n"
        "        return req\n"
    )
    _build_repo(tmp_path, {"_client.py": source})
    _patch_repo(monkeypatch, tmp_path, {"_client"})

    graph, unresolved = call_graph_module.build_call_graph()
    assert graph.has_edge("_client.py::Client.send", "_client.py::Client.build")


def test_cha_fans_out_to_every_overriding_subclass(tmp_path, monkeypatch):
    """
    self.merge_url() called inside BaseClient.send() must resolve to EVERY
    class's implementation (base + all overriding subclasses), since we
    can't know the caller's actual runtime type statically. This is the
    core Class Hierarchy Analysis behavior -- a wrong single guess here
    would silently corrupt retrieval context.
    """
    source = (
        "class BaseClient:\n"
        "    def merge_url(self, url):\n"
        "        return url\n"
        "    def send(self, request):\n"
        "        return self.merge_url(request)\n"
        "\n"
        "class Client(BaseClient):\n"
        "    def merge_url(self, url):\n"
        "        return url.strip()\n"
        "\n"
        "class AsyncClient(BaseClient):\n"
        "    def merge_url(self, url):\n"
        "        return url.lower()\n"
    )
    _build_repo(tmp_path, {"_client.py": source})
    _patch_repo(monkeypatch, tmp_path, {"_client"})

    graph, unresolved = call_graph_module.build_call_graph()
    successors = set(graph.successors("_client.py::BaseClient.send"))
    assert successors == {
        "_client.py::BaseClient.merge_url",
        "_client.py::Client.merge_url",
        "_client.py::AsyncClient.merge_url",
    }


def test_super_call_resolves_to_parent_only_not_fan_out(tmp_path, monkeypatch):
    """
    super().__init__(...) must resolve to a FIXED lookup at the parent --
    the opposite direction from CHA's self-call fan-out. Also verifies the
    inner super() call (visited separately by ast.walk) doesn't produce a
    spurious duplicate unresolved entry (regression test for the
    consumed_node_ids fix).
    """
    source = (
        "class BaseClient:\n"
        "    def __init__(self, url):\n"
        "        self.url = url\n"
        "\n"
        "class Client(BaseClient):\n"
        "    def __init__(self, url, extra):\n"
        "        super().__init__(url)\n"
        "        self.extra = extra\n"
    )
    _build_repo(tmp_path, {"_client.py": source})
    _patch_repo(monkeypatch, tmp_path, {"_client"})

    graph, unresolved = call_graph_module.build_call_graph()
    assert graph.has_edge("_client.py::Client.__init__", "_client.py::BaseClient.__init__")
    # The inner `super()` call must NOT show up as its own unresolved entry.
    assert not any(u.call_expression == "super(...)" for u in unresolved)


def test_bare_constructor_call_resolves_to_init(tmp_path, monkeypatch):
    source = (
        "class Headers:\n"
        "    def __init__(self, data):\n"
        "        self.data = data\n"
        "\n"
        "class Client:\n"
        "    def __init__(self, base_url):\n"
        "        self.headers = Headers(base_url)\n"
    )
    _build_repo(tmp_path, {"_client.py": source})
    _patch_repo(monkeypatch, tmp_path, {"_client"})

    graph, unresolved = call_graph_module.build_call_graph()
    assert graph.has_edge("_client.py::Client.__init__", "_client.py::Headers.__init__")


def test_constructor_call_falls_back_to_parent_init_when_not_overridden(tmp_path, monkeypatch):
    """A class with no __init__ of its own inherits its parent's -- constructor resolution must find it via MRO."""
    source = (
        "class BaseClient:\n"
        "    def __init__(self, url):\n"
        "        self.url = url\n"
        "\n"
        "class Client(BaseClient):\n"
        "    def request(self):\n"
        "        return self.url\n"
        "\n"
        "def make_client():\n"
        "    return Client('http://example.com')\n"
    )
    _build_repo(tmp_path, {"_client.py": source})
    _patch_repo(monkeypatch, tmp_path, {"_client"})

    graph, unresolved = call_graph_module.build_call_graph()
    assert graph.has_edge("_client.py::make_client", "_client.py::BaseClient.__init__")


def test_cross_file_function_call_resolves(tmp_path, monkeypatch):
    models_source = "def parse_response(resp):\n    return resp\n"
    client_source = (
        "from _models import parse_response\n"
        "\n"
        "def handle(resp):\n"
        "    return parse_response(resp)\n"
    )
    _build_repo(tmp_path, {"_models.py": models_source, "_client.py": client_source})
    _patch_repo(monkeypatch, tmp_path, {"_models", "_client"})

    graph, unresolved = call_graph_module.build_call_graph()
    assert graph.has_edge("_client.py::handle", "_models.py::parse_response")


def test_unresolvable_attribute_call_is_logged_not_guessed(tmp_path, monkeypatch):
    """response.json() where response's type is unknown -- must be logged, NEVER given a guessed edge."""
    source = (
        "def handle(response):\n"
        "    data = response.json()\n"
        "    return data\n"
    )
    _build_repo(tmp_path, {"_client.py": source})
    _patch_repo(monkeypatch, tmp_path, {"_client"})

    graph, unresolved = call_graph_module.build_call_graph()
    assert graph.out_degree("_client.py::handle") == 0
    assert any("response.json" in u.call_expression for u in unresolved)


def test_unresolvable_bare_name_is_logged_not_guessed(tmp_path, monkeypatch):
    """A call to a name that's neither a local function, an imported one, nor a known class."""
    source = "def handle(x):\n    return some_unknown_function(x)\n"
    _build_repo(tmp_path, {"_client.py": source})
    _patch_repo(monkeypatch, tmp_path, {"_client"})

    graph, unresolved = call_graph_module.build_call_graph()
    assert graph.out_degree("_client.py::handle") == 0
    assert any("some_unknown_function" in u.call_expression for u in unresolved)


def test_missing_target_file_raises_call_graph_error(tmp_path, monkeypatch):
    """A TARGET_FILES entry pointing at a file that doesn't exist must surface as a clear CallGraphError, not a raw FileNotFoundError."""
    _patch_repo(monkeypatch, tmp_path, {"_nonexistent"})  # no file actually written for this stem

    try:
        call_graph_module.build_call_graph()
        assert False, "expected CallGraphError"
    except CallGraphError as exc:
        assert exc.retryable is True