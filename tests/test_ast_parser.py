"""
Tests for app.graph_analysis.ast_parser.extract_imported_names.

We don't test that ast.parse() works (that's Python's job) -- we test OUR
decisions: which imports count as "real" (runtime-reachable) vs. which get
excluded, and the specific edge cases that were hand-verified during Step 2
development (the else-branch bug fix in particular is a real regression
worth locking in permanently).
"""

from __future__ import annotations

from app.graph_analysis.ast_parser import extract_imported_names


def _write(tmp_path, source: str):
    path = tmp_path / "sample.py"
    path.write_text(source)
    return path


def test_plain_import_and_from_import_are_captured(tmp_path):
    source = "import os\nfrom typing import Optional\n"
    result = extract_imported_names(_write(tmp_path, source))
    assert "os" in result
    assert "typing" in result


def test_bare_type_checking_block_is_excluded(tmp_path):
    source = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from _models import Response\n"
    )
    result = extract_imported_names(_write(tmp_path, source))
    assert "_models" not in result


def test_dotted_type_checking_block_is_excluded(tmp_path):
    """httpx uses BOTH `if TYPE_CHECKING:` and `if typing.TYPE_CHECKING:` across files."""
    source = (
        "import typing\n"
        "if typing.TYPE_CHECKING:\n"
        "    from _models import Response\n"
    )
    result = extract_imported_names(_write(tmp_path, source))
    assert "_models" not in result


def test_else_branch_of_type_checking_is_still_captured(tmp_path):
    """
    Regression test for the fixed bug described in the progress summary:
    else-branch imports were being silently skipped. This must never
    regress -- the else-branch runs at runtime and its imports are real.
    """
    source = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from _urls import URL\n"
        "else:\n"
        "    from _urls import URL\n"
    )
    result = extract_imported_names(_write(tmp_path, source))
    assert result.count("_urls") == 1


def test_function_body_imports_are_found(tmp_path):
    """Imports aren't only at module level -- a recursive walker must find nested ones too."""
    source = (
        "def helper():\n"
        "    import json\n"
        "    return json.dumps({})\n"
    )
    result = extract_imported_names(_write(tmp_path, source))
    assert "json" in result


def test_httpx_prefix_is_stripped_from_from_import(tmp_path):
    source = "from httpx._models import Response\n"
    result = extract_imported_names(_write(tmp_path, source))
    assert "_models" in result
    assert "httpx._models" not in result


def test_nested_if_else_non_type_checking_still_finds_imports(tmp_path):
    """Regular (non-TYPE_CHECKING) nested if/else blocks must still be walked."""
    source = (
        "import sys\n"
        "if sys.version_info >= (3, 8):\n"
        "    import importlib.metadata\n"
        "else:\n"
        "    import importlib_metadata\n"
    )
    result = extract_imported_names(_write(tmp_path, source))
    assert "importlib.metadata" in result
    assert "importlib_metadata" in result
