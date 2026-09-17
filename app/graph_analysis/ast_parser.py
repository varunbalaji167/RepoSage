"""Parse a .py file into a list of imported module names."""
import ast
from pathlib import Path


def _is_type_checking_test(test: ast.expr) -> bool:
    """
    Matches both `if TYPE_CHECKING:` (bare name, imported directly)
    and `if typing.TYPE_CHECKING:` (attribute access) — httpx uses both
    styles across different files.
    """
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
        return True
    return False


def _walk_body(stmts: list, in_type_checking: bool, found: list) -> None:
    """
    Walk a list of statements (module body, if-body, function body, etc.),
    checking EACH statement directly rather than relying on a parent's
    iter_child_nodes() to hand it to us.
    """
    for stmt in stmts:
        if isinstance(stmt, ast.If) and _is_type_checking_test(stmt.test):
            _walk_body(stmt.body, True, found)                 # never runs at runtime
            _walk_body(stmt.orelse, in_type_checking, found)   # does run
            continue

        if isinstance(stmt, ast.Import) and not in_type_checking:
            for alias in stmt.names:
                found.append(alias.name)

        elif isinstance(stmt, ast.ImportFrom) and not in_type_checking:
            if stmt.module is not None:
                name = stmt.module
                if name.startswith("httpx."):
                    name = name[len("httpx."):]
                found.append(name)

        for _, value in ast.iter_fields(stmt):
            if isinstance(value, list):
                nested = [v for v in value if isinstance(v, ast.stmt)]
                if nested:
                    _walk_body(nested, in_type_checking, found)


def extract_imported_names(file_path: Path) -> list:
    """
    Return the names a file imports AT RUNTIME — imports inside
    `if TYPE_CHECKING:` blocks are excluded since they never execute.
    """
    source = file_path.read_text()
    tree = ast.parse(source, filename=str(file_path))
    found: list = []
    _walk_body(tree.body, in_type_checking=False, found=found)
    return found