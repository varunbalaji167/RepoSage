"""Combine graph metrics + git history into a RiskScore."""
import statistics
import subprocess
from functools import lru_cache

import networkx as nx

from app.config import TARGET_FILES, GIT_REPO_ROOT
from app.errors import RiskDataError
from app.graph_analysis.dependency_graph import build_import_graph
from app.schemas import RiskScore


def get_commit_count(file_stem: str) -> int:
    """
    Count commits touching this file across FULL history.
    Requires the non-shallow clone we chose in Step 1 -- a shallow
    clone would make every file report ~1 commit, breaking this entirely.
    """
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "--", f"httpx/{file_stem}.py"],
            cwd=GIT_REPO_ROOT,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        # git itself isn't installed / not on PATH -- a real environment
        # problem, worth a specific message rather than a generic wrap.
        raise RiskDataError("git is not installed or not on PATH.", cause=exc) from exc

    if result.returncode != 0:
        # subprocess.run does NOT raise on a non-zero exit code by default
        # -- without this check, a genuinely failed git command (bad repo
        # path, corrupted .git, wrong cwd) was silently returning "0
        # commits" instead of surfacing as an error. That's a real
        # data-integrity bug: a file could be wrongly reported as
        # low-churn (and therefore possibly HIGH risk) simply because git
        # failed, not because it truly has no history.
        raise RiskDataError(
            f"git log failed for {file_stem}.py (exit {result.returncode}): {result.stderr.strip()}"
        )

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return len(lines)


def compute_risk_scores() -> list[RiskScore]:
    try:
        graph = build_import_graph()

        # Store the actual set of ancestors, not just the length
        dependents_by_file = {f: nx.ancestors(graph, f) for f in TARGET_FILES}
        commits_by_file = {f: get_commit_count(f) for f in TARGET_FILES}
    except RiskDataError:
        raise  # already our own type (e.g. from get_commit_count) -- don't double-wrap
    except Exception as exc:
        # Defensive boundary: anything else unexpected (e.g. build_import_graph
        # failing to read a target file) becomes RiskDataError too, so
        # nothing raw leaks past this module.
        raise RiskDataError(f"Failed to compute risk scores: {exc}", cause=exc) from exc

    # Thresholds computed from the actual data, not hardcoded --
    # a design choice, not something forced on us by the tools.
    # Updated to use the length of the ancestor sets
    dependents_median = statistics.median(len(deps) for deps in dependents_by_file.values())
    commits_median = statistics.median(commits_by_file.values())

    scores = []
    for file_stem in TARGET_FILES:
        dependent_set = dependents_by_file[file_stem]
        dependents = len(dependent_set)
        commits = commits_by_file[file_stem]

        high_impact = dependents >= dependents_median
        low_churn = commits <= commits_median

        if high_impact and low_churn:
            risk_level = "high"
            rationale = (
                f"{dependents} files depend on this, but it only has "
                f"{commits} commits -- little recent proof changes here are safe."
            )
        elif high_impact:
            risk_level = "medium"
            rationale = (
                f"{dependents} files depend on this, and it changes often "
                f"({commits} commits) -- presumably well-exercised."
            )
        else:
            risk_level = "low"
            rationale = f"Only {dependents} file(s) depend on this."

        scores.append(RiskScore(
            file=file_stem,
            dependents=dependents,
            dependent_files=sorted(dependent_set),
            commit_count=commits,
            risk_level=risk_level,
            rationale=rationale,
        ))

    risk_order = {"high": 0, "medium": 1, "low": 2}
    return sorted(scores, key=lambda s: risk_order[s.risk_level])


@lru_cache(maxsize=1)
def get_risk_by_file_stem() -> dict[str, RiskScore]:
    """
    Shared, cached lookup: {file_stem: RiskScore}. Cached because
    compute_risk_scores() runs `git log` per file -- expensive enough that
    it shouldn't be recomputed by every caller. Added so code_explanation_chain.py
    (Step 5's risk-note join) and orchestration/graph.py's risk_assessment_node
    (Step 7) can share ONE cache instead of each keeping a private copy,
    which would mean git log running twice in the same process for the
    exact same data.
    """
    return {score.file: score for score in compute_risk_scores()}