"""ChromaDB client setup + collection helpers.
Step 4 — ChromaDB indexing.

Design decisions locked in for this step:
  - Embedding model: jina-embeddings-v2-base-code, run locally via
    sentence-transformers. Free, no API key, and trained on code, so it
    should place semantically-similar code (not just textually-similar
    code) closer together than a general-purpose text embedder would.
  - Metadata stored per chunk: file_path, chunk_type, name, start_line,
    end_line, parent_class. This is NOT for pre-filtering the query (the
    query text itself won't mention a file path) — it's for two things
    downstream, in later steps:
      1. Citing the source of the retrieved chunk in the final answer.
      2. Joining against risk_scoring.py's output by file_path, so a
         code_explanation answer can surface a risk note ONLY when that
         file's risk is HIGH (per our Step 4 design decision) — silent
         otherwise.
  - Collection is a single flat `code_chunks` collection (per the design
    doc), persisted to disk so we don't re-embed on every run.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from app.config import GIT_REPO_ROOT, REPO_PATH, TARGET_FILES
from app.indexing.chunker import chunk_file
from app.schemas import CodeChunk

CHROMA_PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "code_chunks"
EMBEDDING_MODEL_NAME = "jinaai/jina-embeddings-v2-base-code"


class JinaCodeEmbeddingFunction(EmbeddingFunction):
    """Wraps jina-embeddings-v2-base-code for use as a Chroma embedding function."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        # Imported lazily so the rest of this module can be imported/tested
        # without requiring sentence-transformers + the model download.
        from sentence_transformers import SentenceTransformer

        # trust_remote_code=True is required: this model ships custom
        # modeling code from its Hugging Face repo, not a stock architecture.
        self._model = SentenceTransformer(model_name, trust_remote_code=True)

    def __call__(self, input: Documents) -> Embeddings:
        vectors = self._model.encode(list(input), convert_to_numpy=True)
        return vectors.tolist()


def _chunk_id(chunk: CodeChunk, index: int) -> str:
    # Stable, human-readable ID: helps when debugging what got indexed.
    safe_name = chunk.name.replace("/", "_").replace(" ", "_")
    return f"{chunk.file_path}:{safe_name}:{chunk.start_line}:{index}"


def _chunk_metadata(chunk: CodeChunk) -> dict:
    # Chroma metadata values must be str/int/float/bool -- None is not
    # allowed, so we normalize parent_class to "" when absent.
    return {
        "file_path": chunk.file_path,
        "chunk_type": chunk.chunk_type,
        "name": chunk.name,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "parent_class": chunk.parent_class or "",
    }


def load_target_file_chunks() -> list[CodeChunk]:
    """Read and chunk every file in TARGET_FILES."""
    all_chunks: list[CodeChunk] = []
    for raw_entry in TARGET_FILES:
        # TARGET_FILES entries may or may not already include the .py suffix
        # depending on how config.py defines them -- normalize here rather
        # than assuming one format.
        relative_path = raw_entry if raw_entry.endswith(".py") else f"{raw_entry}.py"
        full_path = Path(REPO_PATH) / relative_path
        source = full_path.read_text(encoding="utf-8")
        all_chunks.extend(chunk_file(source, file_path=relative_path))
    return all_chunks


def build_chunk_index(reset: bool = False) -> chromadb.Collection:
    """
    Chunk all TARGET_FILES and load them into the persistent `code_chunks`
    Chroma collection. Safe to re-run: pass reset=True to wipe and rebuild
    from scratch, otherwise existing IDs are upserted.
    """
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    embedding_fn = JinaCodeEmbeddingFunction()

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except (ValueError, chromadb.errors.NotFoundError):
            pass  # collection didn't exist yet

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    chunks = load_target_file_chunks()

    ids = [_chunk_id(chunk, i) for i, chunk in enumerate(chunks)]
    documents = [chunk.text for chunk in chunks]
    metadatas = [_chunk_metadata(chunk) for chunk in chunks]

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    print(f"Indexed {len(chunks)} chunks from {len(TARGET_FILES)} files into '{COLLECTION_NAME}'.")
    return collection


if __name__ == "__main__":
    build_chunk_index(reset=True)