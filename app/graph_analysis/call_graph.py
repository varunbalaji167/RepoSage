"""
Step 5a — Function-level call graph.

Design decisions locked in for this step:
  - Nodes are functions/methods (e.g. "_client.py::Client.send"), NOT files.
    Separate, more granular graph than Step 2's file-level import graph.

  - self.method(...) resolution uses Class Hierarchy Analysis (CHA), the
    same principle production static-analysis tools use for this exact
    ambiguity: since we don't track the caller's actual runtime type, a
    self-call inside class C could dispatch to C's own implementation OR
    any subclass of C that overrides the method. Rather than guessing one
    (which could silently be wrong and corrupt retrieval context later),
    we add an edge to EVERY possible receiver implementation across C and
    all of C's descendants. This is "sound but imprecise": it never misses
    the true edge, at the cost of occasionally including one that doesn't
    apply to a specific call site.

  - bare_name(...) resolution: same-file top-level function, OR a
    top-level function in another target file that THIS file's import
    list (from Step 2's extract_imported_names) includes. Step 2's
    function returns imported MODULE names, not specific imported names,
    so this is deliberately approximate: "this file imports module _models,
    and _models defines a top-level function called X" is treated as a
    match. Good enough for our closed 9-file scope; would need tightening
    (tracking specific `from X import Y` names) if this scaled further.

  - Anything else (e.g. `response.json()`, unknown variable type) is
    UNRESOLVABLE and logged, never guessed -- same reasoning as before:
    a wrong edge silently corrupts LLM context later; a missing edge is a
    visible, safe gap.

  - Inheritance stored as a plain dict {class_name: parent_class_name},
    not graph edges -- CHA needs both directions (who's my parent, who
    are my descendants), so we also build the reverse {parent: [children]}
    map once, but neither is part of the call graph itself.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

from app.config import REPO_PATH, TARGET_FILES
from app.errors import CallGraphError
from app.graph_analysis.ast_parser import extract_imported_names


def _normalized_target_files() -> list[str]:
    return [f if f.endswith(".py") else f"{f}.py" for f in TARGET_FILES]


def _module_name(file_path: str) -> str:  # DEPRECATED: unused since bare-call resolution
    # was rewritten to compare against imported MODULE names directly (see
    # _resolve_bare_call) rather than needing this conversion. Left in place
    # per project convention of marking, not removing, dead code.
    """'_models.py' -> '_models', to compare against extract_imported_names' output."""
    return file_path[:-3] if file_path.endswith(".py") else file_path


@dataclass
class ClassInfo:
    file_path: str
    method_names: set[str] = field(default_factory=set)
    parent_class: str | None = None


@dataclass
class UnresolvedCall:
    caller_node_id: str
    call_expression: str
    reason: str


def _node_id(file_path: str, name: str, class_name: str | None = None) -> str:
    qualified = f"{class_name}.{name}" if class_name else name
    return f"{file_path}::{qualified}"


def _call_repr(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return f"{func.id}(...)"
    if isinstance(func, ast.Attribute):
        base = func.value.id if isinstance(func.value, ast.Name) else "<expr>"
        return f"{base}.{func.attr}(...)"
    return "<unknown call>"


class _FileRegistry:
    def __init__(self):
        self.top_level_functions: dict[str, set[str]] = {}
        self.classes: dict[str, ClassInfo] = {}
        self.imported_modules_by_file: dict[str, list[str]] = {}
        self.children_of: dict[str, list[str]] = {}  # parent_class -> [child classes], built after all files registered

    def register_file(self, file_path: str, tree: ast.Module, imported_modules: list[str]):
        self.top_level_functions.setdefault(file_path, set())
        self.imported_modules_by_file[file_path] = imported_modules

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.top_level_functions[file_path].add(node.name)

            elif isinstance(node, ast.ClassDef):
                parent_name = None
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        parent_name = base.id
                        break
                method_names = {
                    n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                self.classes[node.name] = ClassInfo(
                    file_path=file_path, method_names=method_names, parent_class=parent_name
                )

    def build_children_map(self):
        """Reverse of parent_class -- needed for CHA's downward search. Run once, after ALL files registered."""
        for class_name, info in self.classes.items():
            if info.parent_class:
                self.children_of.setdefault(info.parent_class, []).append(class_name)


def _mro_lookup(registry: _FileRegistry, start_class: str, method_name: str) -> str | None:
    """Walk UP the single-inheritance chain from start_class until method_name is found."""
    current = start_class
    seen = set()
    while current and current not in seen:
        seen.add(current)
        info = registry.classes.get(current)
        if info is None:
            return None
        if method_name in info.method_names:
            return _node_id(info.file_path, method_name, current)
        current = info.parent_class
    return None


def _descendants(registry: _FileRegistry, class_name: str) -> set[str]:
    """All classes reachable by walking DOWN children_of from class_name (excludes class_name itself)."""
    result: set[str] = set()
    stack = list(registry.children_of.get(class_name, []))
    while stack:
        child = stack.pop()
        if child not in result:
            result.add(child)
            stack.extend(registry.children_of.get(child, []))
    return result


def _resolve_self_call_cha(registry: _FileRegistry, current_class: str, method_name: str) -> set[str]:
    """
    CHA: the true receiver could be current_class itself, or any subclass of it
    (never a superclass -- a self-call inside class C's method body can only be
    invoked on a C-or-more-derived instance). For each candidate class, resolve
    via normal MRO lookup, then dedupe the resulting set of implementation nodes.
    """
    candidates = {current_class} | _descendants(registry, current_class)
    resolved: set[str] = set()
    for cls in candidates:
        node = _mro_lookup(registry, cls, method_name)
        if node:
            resolved.add(node)
    return resolved


def _resolve_bare_call(registry: _FileRegistry, current_file: str, called_name: str) -> str | None:
    """Same-file/imported top-level FUNCTION, e.g. helper(x) or parse_response(r)."""
    if called_name in registry.top_level_functions.get(current_file, set()):
        return _node_id(current_file, called_name)

    imported_modules = registry.imported_modules_by_file.get(current_file, [])
    for module_name in imported_modules:
        candidate_file = f"{module_name}.py"
        if candidate_file != current_file and called_name in registry.top_level_functions.get(candidate_file, set()):
            return _node_id(candidate_file, called_name)

    return None


def _resolve_constructor_call(registry: _FileRegistry, called_name: str) -> str | None:
    """
    Bare call to a CLASS name, e.g. URL(base_url) or Headers(headers) -- a
    constructor call, not a function call. Resolves to that class's __init__,
    walking up the inheritance chain if the class doesn't define its own
    (e.g. it relies on its parent's __init__). Classes are tracked as one
    flat, globally-addressable set across our 9 files (same assumption CHA
    already relies on for self-calls), so no per-file import check is needed
    here -- if called_name isn't a class we know about at all (e.g. it's a
    builtin like `list()` or `dict()`), this correctly returns None.
    """
    if called_name not in registry.classes:
        return None
    return _mro_lookup(registry, called_name, "__init__")


def _resolve_super_call(registry: _FileRegistry, current_class: str, method_name: str) -> str | None:
    """
    super().method(...) -- deliberately skips current_class's OWN
    implementation and starts the search at its parent, unlike a self-call
    which starts at current_class itself. This is the opposite direction
    from CHA's fan-out: a single, fixed lookup upward, not multiple
    candidates downward, because `super()` unambiguously means "my parent,"
    not "whichever subclass happens to be calling this."
    """
    info = registry.classes.get(current_class)
    if info is None or info.parent_class is None:
        return None
    return _mro_lookup(registry, info.parent_class, method_name)


def _walk_function_body(
    node: ast.AST,
    registry: _FileRegistry,
    file_path: str,
    caller_node_id: str,
    current_class: str | None,
    graph: nx.DiGraph,
    unresolved: list[UnresolvedCall],
):
    consumed_node_ids: set[int] = set()  # nodes already handled as part of a larger pattern (e.g. super() inside super().method())

    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if id(child) in consumed_node_ids:
            continue  # e.g. the inner `super()` call within `super().method(...)` -- already accounted for

        func = child.func

        # self.method(...) -- CHA fan-out to every possible receiver.
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
            resolved_set = _resolve_self_call_cha(registry, current_class, func.attr) if current_class else set()
            if resolved_set:
                for target in resolved_set:
                    graph.add_edge(caller_node_id, target)
            else:
                unresolved.append(
                    UnresolvedCall(caller_node_id, _call_repr(child), "self-call not found via CHA")
                )
            continue

        # super().method(...) -- fixed single lookup starting at the parent.
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Call)
            and isinstance(func.value.func, ast.Name)
            and func.value.func.id == "super"
        ):
            consumed_node_ids.add(id(func.value))  # the inner `super()` call -- don't log it separately
            resolved = _resolve_super_call(registry, current_class, func.attr) if current_class else None
            if resolved:
                graph.add_edge(caller_node_id, resolved)
            else:
                unresolved.append(
                    UnresolvedCall(caller_node_id, _call_repr(child), "super() call not resolvable (no parent or method not found)")
                )
            continue

        # bare_name(...) -- either a plain function call or a class constructor.
        if isinstance(func, ast.Name):
            resolved = _resolve_bare_call(registry, file_path, func.id)
            if resolved is None:
                resolved = _resolve_constructor_call(registry, func.id)
            if resolved:
                graph.add_edge(caller_node_id, resolved)
            else:
                unresolved.append(
                    UnresolvedCall(caller_node_id, _call_repr(child), "unresolved bare name")
                )
            continue

        # Any other attribute call (e.g. response.json()) -- unresolvable.
        unresolved.append(
            UnresolvedCall(
                caller_node_id, _call_repr(child), "attribute call on non-self variable of unknown type"
            )
        )


def build_class_to_file_map() -> dict[str, str]:
    """
    Lightweight: parses all target files and returns {class_name: file_stem}
    (e.g. {"Client": "_client", "Headers": "_models"}), WITHOUT the
    expensive full call-resolution walk that build_call_graph() does.

    Added for classifier.py's extract_target_files(), which needs to
    resolve a mentioned class name (e.g. "Client") to its containing file
    -- reusing THIS instead of build_call_graph() avoids walking every
    function body in all 9 files just to answer a much smaller question.
    """
    try:
        registry = _FileRegistry()
        for relative_path in _normalized_target_files():
            full_path = Path(REPO_PATH) / relative_path
            source = full_path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            imported_modules = extract_imported_names(full_path)
            registry.register_file(relative_path, tree, imported_modules)

        return {
            class_name: info.file_path[:-3] if info.file_path.endswith(".py") else info.file_path
            for class_name, info in registry.classes.items()
        }
    except Exception as exc:
        # Real failure modes here: a target file missing/unreadable, or
        # genuinely invalid Python (SyntaxError) -- a configuration
        # problem, not something retrying fixes on its own, but the
        # CALLER might reasonably retry once it's fixed (hence
        # CallGraphError's retryable=True via ServiceUnavailableError).
        raise CallGraphError(f"Failed to build class-to-file map: {exc}", cause=exc) from exc


def build_call_graph() -> tuple[nx.DiGraph, list[UnresolvedCall]]:
    try:
        graph = nx.DiGraph()
        unresolved: list[UnresolvedCall] = []
        registry = _FileRegistry()
        parsed_files: dict[str, ast.Module] = {}

        for relative_path in _normalized_target_files():
            full_path = Path(REPO_PATH) / relative_path
            source = full_path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            parsed_files[relative_path] = tree

            imported_modules = extract_imported_names(full_path)  # matches real signature: takes a Path
            registry.register_file(relative_path, tree, imported_modules)

        registry.build_children_map()  # requires ALL classes to be registered first

        for file_path, tree in parsed_files.items():
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    caller_id = _node_id(file_path, node.name)
                    graph.add_node(caller_id)
                    _walk_function_body(node, registry, file_path, caller_id, None, graph, unresolved)

                elif isinstance(node, ast.ClassDef):
                    for sub_node in node.body:
                        if isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            caller_id = _node_id(file_path, sub_node.name, node.name)
                            graph.add_node(caller_id)
                            _walk_function_body(
                                sub_node, registry, file_path, caller_id, node.name, graph, unresolved
                            )

        return graph, unresolved
    except Exception as exc:
        raise CallGraphError(f"Failed to build call graph: {exc}", cause=exc) from exc


if __name__ == "__main__":
    call_graph, unresolved_calls = build_call_graph()
    print(f"Call graph: {call_graph.number_of_nodes()} nodes, {call_graph.number_of_edges()} edges")
    print(f"Unresolved calls: {len(unresolved_calls)}")
    for u in unresolved_calls[:10]:
        print(f"  {u.caller_node_id} -> {u.call_expression}  ({u.reason})")