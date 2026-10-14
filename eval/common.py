"""Ground-truth helpers. Reads the httpx source directly (ast, git); nothing imports app/*
so a bug in the system under test can't leak into the answer key."""
from __future__ import annotations

import ast
import builtins
import keyword
import pathlib
import re
import statistics
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
GIT_ROOT = ROOT / "target_repos/httpx"
PKG = GIT_ROOT / "httpx"
STEMS = ["_api", "_client", "_models", "_config", "_auth", "_urls", "_exceptions", "_types", "_utils"]


# ---- gold nodes: "file.py::Class.name" -> (file, start_line, end_line, source) ------------------
def load_nodes() -> dict[str, tuple[str, int, int, str]]:
    out = {}
    for stem in STEMS:
        fname = f"{stem}.py"
        src = (PKG / fname).read_text()
        for n in ast.parse(src).body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out[f"{fname}::{n.name}"] = (fname, n.lineno, n.end_lineno, ast.get_source_segment(src, n))
            if isinstance(n, ast.ClassDef):
                for m in n.body:
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        out[f"{fname}::{n.name}.{m.name}"] = (fname, m.lineno, m.end_lineno, ast.get_source_segment(src, m))
    return out


def covers(doc_file: str, doc_start: int, doc_end: int, gold: tuple[str, int, int, str]) -> bool:
    """True if the retrieved span contains the gold node's full line range (so a class chunk covers its methods)."""
    gfile, gs, ge, _ = gold
    return doc_file == gfile and doc_start <= gs and doc_end >= ge


# ---- vocabulary for the fabricated-identifier check ---------------------------------------------
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text))


def repo_vocabulary() -> set[str]:
    vocab = set(dir(builtins)) | set(keyword.kwlist)
    for p in PKG.rglob("*.py"):
        vocab |= tokens(p.read_text())
    return vocab


# ---- independent risk recomputation (spec: impact = transitive importers, churn = commits) ------
def _is_type_checking_guard(node: ast.If) -> bool:
    t = node.test
    return (isinstance(t, ast.Name) and t.id == "TYPE_CHECKING") or (
        isinstance(t, ast.Attribute) and t.attr == "TYPE_CHECKING")


def _runtime_imports(tree: ast.AST):
    """Yield ImportFrom nodes that run at import time; skip TYPE_CHECKING bodies."""
    stack = [tree]
    while stack:
        node = stack.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If) and _is_type_checking_guard(child):
                stack.extend(child.orelse)  # only the else-branch runs
                continue
            if isinstance(child, ast.ImportFrom):
                yield child
            stack.append(child)


def independent_risk() -> dict[str, dict]:
    direct = {s: set() for s in STEMS}  # importer -> imported
    for stem in STEMS:
        tree = ast.parse((PKG / f"{stem}.py").read_text())
        for imp in _runtime_imports(tree):
            if imp.level == 1 and imp.module in STEMS and imp.module != stem:
                direct[stem].add(imp.module)
            elif imp.level == 1 and imp.module is None:  # "from . import _x"
                direct[stem] |= {a.name for a in imp.names if a.name in STEMS and a.name != stem}

    def dependents(target: str) -> set[str]:
        seen, frontier = set(), {target}
        while frontier:
            frontier = {s for s, imported in direct.items() if imported & frontier} - seen
            seen |= frontier
        seen.discard(target)
        return seen

    deps = {s: dependents(s) for s in STEMS}
    commits = {}
    for s in STEMS:
        r = subprocess.run(["git", "rev-list", "--count", "HEAD", "--", f"httpx/{s}.py"],
                           cwd=GIT_ROOT, capture_output=True, text=True, check=True)
        commits[s] = int(r.stdout.strip())

    dep_med = statistics.median(len(d) for d in deps.values())
    com_med = statistics.median(commits.values())
    out = {}
    for s in STEMS:
        high_impact, low_churn = len(deps[s]) >= dep_med, commits[s] <= com_med
        level = "high" if (high_impact and low_churn) else "medium" if high_impact else "low"
        out[s] = {"dependents": len(deps[s]), "dependent_files": sorted(deps[s]),
                  "commit_count": commits[s], "risk_level": level}
    return out


# ---- stats ---------------------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval; honest at small n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def pct(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100*k/n:.0f}% (95% CI {100*lo:.0f}–{100*hi:.0f}%)"


def percentile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = max(0, min(len(xs) - 1, int(round(q * (len(xs) - 1)))))
    return xs[i]
