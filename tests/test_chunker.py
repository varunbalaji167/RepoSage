"""
Tests for app.indexing.chunker.chunk_file.

Pure function, no I/O -- these are fast and exercise the full fallback
chain: function/class-level chunking, oversized-class -> per-method
recursion, and oversized-function/method -> sliding-window split. These
mirror the exact scenarios manually verified against real httpx code
during Step 4 development, now locked in as regression tests.
"""

from __future__ import annotations

from app.indexing.chunker import chunk_file


def test_small_function_is_one_chunk():
    source = "def helper(x):\n    return x + 1\n"
    chunks = chunk_file(source, "sample.py")
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "function"
    assert chunks[0].name == "helper"


def test_async_function_gets_async_chunk_type():
    source = "async def fetch(url):\n    return await get(url)\n"
    chunks = chunk_file(source, "sample.py")
    assert chunks[0].chunk_type == "async_function"


def test_small_class_is_one_chunk():
    source = "class Client:\n    def __init__(self, url):\n        self.url = url\n"
    chunks = chunk_file(source, "sample.py")
    class_chunks = [c for c in chunks if c.chunk_type == "class"]
    assert len(class_chunks) == 1
    assert class_chunks[0].name == "Client"


def test_oversized_function_falls_back_to_sliding_window_split():
    body = "\n".join(f"    x{i} = {i}" for i in range(300))
    source = f"def big():\n{body}\n    return x0\n"
    chunks = chunk_file(source, "sample.py")
    split_chunks = [c for c in chunks if c.chunk_type == "split"]
    assert len(split_chunks) >= 2
    # Part numbering must be sequential and correctly labeled.
    assert split_chunks[0].name == "big[part 1/{}]".format(len(split_chunks))
    assert split_chunks[-1].name == f"big[part {len(split_chunks)}/{len(split_chunks)}]"


def test_oversized_class_recurses_into_per_method_chunks():
    method_body = "\n".join(f"        y{i} = {i}" for i in range(300))
    source = (
        f"class Big:\n"
        f"    def small_method(self):\n"
        f"        return 1\n"
        f"    def huge_method(self):\n{method_body}\n        return y0\n"
    )
    chunks = chunk_file(source, "sample.py")
    # Class itself should NOT appear as a single "class" chunk -- it's too big.
    assert not any(c.chunk_type == "class" for c in chunks)
    small = [c for c in chunks if c.name == "small_method"]
    assert len(small) == 1
    assert small[0].chunk_type == "method"
    assert small[0].parent_class == "Big"
    # The oversized method must ALSO trigger the sliding-window fallback --
    # this is the nested-fallback case (class -> methods -> split).
    huge_parts = [c for c in chunks if c.chunk_type == "split" and c.name.startswith("huge_method")]
    assert len(huge_parts) >= 2
    assert all(c.parent_class == "Big" for c in huge_parts)


def test_leftover_module_level_code_becomes_one_chunk():
    source = (
        "import os\n"
        "MAX_RETRIES = 3\n"
        "\n"
        "def helper():\n"
        "    return 1\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    print(helper())\n"
    )
    chunks = chunk_file(source, "sample.py")
    module_chunks = [c for c in chunks if c.chunk_type == "module_level"]
    assert len(module_chunks) == 1
    assert "import os" in module_chunks[0].text
    assert "MAX_RETRIES" in module_chunks[0].text
    assert '__main__' in module_chunks[0].text
    # The function's own body must NOT leak into the module-level chunk.
    assert "return 1" not in module_chunks[0].text


def test_file_with_only_module_level_code_does_not_crash():
    source = "import os\nVALUE = 42\n"
    chunks = chunk_file(source, "sample.py")
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "module_level"


def test_blank_lines_are_excluded_from_module_level_chunk():
    """The chunker filters blank leftover lines via `.strip()` -- verify it actually does."""
    source = "import os\n\n\nVALUE = 1\n\ndef f():\n    return 1\n"
    chunks = chunk_file(source, "sample.py")
    module_chunk = next(c for c in chunks if c.chunk_type == "module_level")
    assert "" not in module_chunk.text.splitlines()
