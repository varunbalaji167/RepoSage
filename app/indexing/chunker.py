"""
Step 4 — AST-based code chunking.

Design decisions locked in for this step:
  - Chunk at function/class granularity (not whole-file, not fixed-token windows),
    because chunk boundaries should respect code semantics.
  - This decision rule is per-AST-node, not per-file, so it's scale-invariant:
    the same function works whether TARGET_FILES has 9 entries or 900.
  - Any single function/class whose source exceeds MAX_CHUNK_TOKENS gets a
    sliding-window fallback split, so no chunk sent to the embedding model
    is unreasonably large.
  - Leftover top-level code (imports, module-level constants, `if __name__`
    guards) is NOT discarded — it's collected into one "module_level" chunk
    per file, since it's usually small and can carry useful context.
"""

from __future__ import annotations

import ast

from app.schemas import CodeChunk

# Rough heuristic: ~4 characters per token. Good enough for a chunking
# threshold — we don't need exact tokenizer parity here, just a consistent
# way to decide "is this chunk too big."
CHARS_PER_TOKEN_ESTIMATE = 4
MAX_CHUNK_TOKENS = 400
MAX_CHUNK_CHARS = MAX_CHUNK_TOKENS * CHARS_PER_TOKEN_ESTIMATE

# When we fall back to sliding-window splitting an oversized function/class,
# split by line and overlap a little so we don't sever context at the seam.
WINDOW_OVERLAP_LINES = 3

# NOTE: CodeChunk used to be defined here as a @dataclass. Moved to
# schemas.py and converted to a Pydantic BaseModel -- it crosses a real
# module boundary (consumed by vector_store.py), same category as this
# project's other shared schemas (RiskScore, QueryClassification, etc.).


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)


def _sliding_window_split(text: str, max_chars: int, overlap_lines: int) -> list[str]:
    """Fallback for a single AST node whose source is too large for one chunk."""
    lines = text.splitlines()
    windows: list[str] = []
    start = 0
    while start < len(lines):
        # Grow the window until we'd exceed max_chars, then cut.
        chunk_lines: list[str] = []
        char_count = 0
        i = start
        while i < len(lines) and char_count < max_chars:
            chunk_lines.append(lines[i])
            char_count += len(lines[i]) + 1  # +1 for the newline
            i += 1
        windows.append("\n".join(chunk_lines))
        if i >= len(lines):
            break
        start = max(i - overlap_lines, start + 1)  # ensure forward progress
    return windows


def _node_source(source: str, node: ast.AST) -> str | None:
    return ast.get_source_segment(source, node)


def _chunk_function_or_class(
    source: str,
    node: ast.AST,
    file_path: str,
    chunk_type: str,
    parent_class: str | None = None,
) -> list[CodeChunk]:
    text = _node_source(source, node)
    if text is None:
        return []

    name = getattr(node, "name", "<unknown>")

    if _estimate_tokens(text) <= MAX_CHUNK_TOKENS:
        return [
            CodeChunk(
                file_path=file_path,
                chunk_type=chunk_type,
                name=name,
                start_line=node.lineno,
                end_line=getattr(node, "end_lineno", node.lineno),
                text=text,
                parent_class=parent_class,
            )
        ]

    # Fallback: oversized node -> sliding-window split.
    windows = _sliding_window_split(text, MAX_CHUNK_CHARS, WINDOW_OVERLAP_LINES)
    chunks = []
    running_line = node.lineno
    for idx, window_text in enumerate(windows):
        window_line_count = window_text.count("\n") + 1
        chunks.append(
            CodeChunk(
                file_path=file_path,
                chunk_type="split",
                name=f"{name}[part {idx + 1}/{len(windows)}]",
                start_line=running_line,
                end_line=running_line + window_line_count - 1,
                text=window_text,
                parent_class=parent_class,
            )
        )
        running_line += window_line_count - WINDOW_OVERLAP_LINES
    return chunks


def chunk_file(source: str, file_path: str) -> list[CodeChunk]:
    """
    Parse one file's source and return its list of CodeChunks.

    Top-level functions and classes each become one chunk (or several, if
    oversized). A class's methods are NOT split out individually unless the
    whole class is too large to fit in one chunk — in that case we recurse
    into the class body and chunk each method the same way a top-level
    function would be chunked.
    """
    tree = ast.parse(source)
    chunks: list[CodeChunk] = []

    # DEPRECATED: superseded by `covered_lines` (a set of line numbers,
    # populated below) which does the same job more usefully -- `covered_lines`
    # lets us compute leftover lines via set subtraction in one step, whereas
    # this list was an earlier, abandoned approach. Left here rather than
    # deleted per project convention of marking (not removing) dead code.
    # module_level_lines: list[int] = []

    covered_lines: set[int] = set()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunk_type = "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
            new_chunks = _chunk_function_or_class(source, node, file_path, chunk_type)
            chunks.extend(new_chunks)
            covered_lines.update(range(node.lineno, getattr(node, "end_lineno", node.lineno) + 1))

        elif isinstance(node, ast.ClassDef):
            class_text = _node_source(source, node)
            covered_lines.update(range(node.lineno, getattr(node, "end_lineno", node.lineno) + 1))

            if class_text is not None and _estimate_tokens(class_text) <= MAX_CHUNK_TOKENS:
                chunks.append(
                    CodeChunk(
                        file_path=file_path,
                        chunk_type="class",
                        name=node.name,
                        start_line=node.lineno,
                        end_line=getattr(node, "end_lineno", node.lineno),
                        text=class_text,
                    )
                )
            else:
                # Class too big as one chunk -> chunk each method individually.
                for sub_node in node.body:
                    if isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        sub_type = "async_function" if isinstance(sub_node, ast.AsyncFunctionDef) else "method"
                        chunks.extend(
                            _chunk_function_or_class(
                                source, sub_node, file_path, sub_type, parent_class=node.name
                            )
                        )

    # Anything not inside a top-level function/class (imports, constants,
    # `if __name__ == "__main__":`, etc.) -> one module_level chunk.
    all_lines = set(range(1, len(source.splitlines()) + 1))
    leftover_lines = sorted(all_lines - covered_lines)
    if leftover_lines:
        source_lines = source.splitlines()
        leftover_text = "\n".join(
            source_lines[ln - 1] for ln in leftover_lines if source_lines[ln - 1].strip()
        )
        if leftover_text.strip():
            chunks.append(
                CodeChunk(
                    file_path=file_path,
                    chunk_type="module_level",
                    name="<module>",
                    start_line=leftover_lines[0],
                    end_line=leftover_lines[-1],
                    text=leftover_text,
                )
            )

    return chunks