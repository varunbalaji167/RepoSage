"""Build a networkx.DiGraph from imports across target files."""
import networkx as nx

from app.config import TARGET_FILES, REPO_PATH
from app.graph_analysis.ast_parser import extract_imported_names


def build_import_graph() -> nx.DiGraph:
    """
    Build a directed graph where an edge A -> B means "file A imports file B",
    restricted to files in TARGET_FILES. External/third-party imports and
    out-of-scope httpx files (like _content.py) are silently excluded.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(TARGET_FILES)  # ensures isolated files still show up as nodes

    for file_stem in TARGET_FILES:
        file_path = REPO_PATH / f"{file_stem}.py"
        imported_names = extract_imported_names(file_path)

        for name in imported_names:
            if name in TARGET_FILES and name != file_stem:
                graph.add_edge(file_stem, name)
                # (Removed: a debug `print(f"{file_stem} -> {name}")` used to
                # fire here during Step 2 development. Unlike the deprecated
                # items marked elsewhere in this codebase, this wasn't unused
                # code being preserved -- it was live debug output that had
                # started firing on every import of this module, including
                # in production via risk_scoring.py's call chain. Removed
                # outright rather than commented-and-kept.)

    return graph