"""
Step 5 — code_explanation LCEL chain.

Design decisions locked in for this step:
  - k=3 primary chunks from vector retrieval (chunk granularity is already
    function/class-level, so 3 gives room for a related-but-not-top-match
    chunk without flooding the LLM with noise).
  - Split chunks (chunk_type="split") are reassembled -- by file_path +
    parent_class + base name (the part before "[part") -- BEFORE being used
    anywhere, so the LLM never sees half a method.
  - From each (reassembled) primary chunk, expand 1 hop via Step 5a's call
    graph -- pull in what it directly calls. Visited-tracking avoids
    duplicate chunks if a call cycles back.
  - Citations are built from Document.metadata AFTER the LLM/parser step,
    using RunnableParallel to carry the retrieved Documents alongside the
    LLM's text -- a plain `prompt | llm | parser` pipe would discard the
    Documents the moment they're embedded into prompt text, so metadata
    would no longer exist by the time we need it for citations.
  - Both primary AND expanded chunks get cited (per design decision).

Testability note (added after initial build):
  Resource construction (Chroma collection + embedding model, call graph,
  LLM client, risk scores) is LAZY and CACHED via _get_*() functions below,
  rather than running at import time. Pure logic functions (_fetch_full_chunk,
  _retrieve_and_expand, _build_final_response) take these resources as
  explicit arguments rather than reading module globals. This means:
    1. Importing this module has zero side effects (no API key, no model
       download, no disk access required just to import it).
    2. The pure logic can be unit-tested with fake collections/graphs/scores,
       without needing real infrastructure -- see tests/test_code_explanation_chain.py.
  Actual resource loading only happens on first real chain invocation.
"""

from __future__ import annotations

import re
from functools import lru_cache
# from dataclasses import dataclass  # DEPRECATED: unused -- carried over by
# habit from call_graph.py's pattern; this module never defines a dataclass.

from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root; must run before ChatGoogleGenerativeAI is constructed

import chromadb
import networkx as nx
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI

from app.errors import RetrievalError
from app.graph_analysis.call_graph import build_call_graph
from app.graph_analysis.risk_scoring import get_risk_by_file_stem
from app.indexing.vector_store import CHROMA_PERSIST_DIR, COLLECTION_NAME, JinaCodeEmbeddingFunction
from app.llm_utils import FALLBACK_MODEL_NAME, FallbackRunnable
from app.schemas import CodeExplanationResult

TOP_K = 3
OVERFETCH_MULTIPLIER = 4  # fetch TOP_K * this many raw chunks, dedupe by
# function identity, THEN cut to TOP_K -- see _retrieve_and_expand's
# docstring for why (a split function's parts must not each compete for
# their own k-slot).
LLM_MODEL_NAME = "gemini-3.5-flash-lite"


# ---------------------------------------------------------------------------
# Lazy, cached resource getters. Nothing here runs until first CALLED, and
# each is cached (via lru_cache) so repeated calls don't redo expensive work
# (model loading, AST-walking 9 files, git log, API client construction).
# Importing this module triggers NONE of this -- deferred to first real use.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_collection() -> chromadb.Collection:
    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        return client.get_collection(name=COLLECTION_NAME, embedding_function=JinaCodeEmbeddingFunction())
    except Exception as exc:
        # Real failure modes: chroma_db/ missing or corrupted, the
        # "code_chunks" collection was never built (Step 4 never ran), or
        # the Jina embedding model failed to load.
        raise RetrievalError(f"Failed to initialize the vector store: {exc}", cause=exc) from exc


@lru_cache(maxsize=1)
def _get_call_graph() -> nx.DiGraph:
    graph, _unresolved_calls = build_call_graph()
    return graph


@lru_cache(maxsize=1)
def _get_llm() -> FallbackRunnable:
    # NOTE: no `temperature` parameter here. Gemini 3.6 Flash and later
    # deprecate temperature/top_p/top_k entirely -- not just ignored today,
    # but documented to start returning an HTTP 400 error for them in
    # future model generations. Determinism is controlled instead via the
    # structured output schema + explicit system instructions we already
    # use, per Google's own migration guidance.
    #
    # Wrapped in FallbackRunnable so a rate-limited primary model falls
    # back to a lighter one automatically -- see llm_utils.py. The
    # existing call site (RunnableLambda(lambda x: _get_llm().invoke(x)))
    # doesn't need to change, since FallbackRunnable exposes the same
    # .invoke() interface as a plain ChatGoogleGenerativeAI.
    primary = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME)
    fallback = ChatGoogleGenerativeAI(model=FALLBACK_MODEL_NAME)
    return FallbackRunnable(primary, fallback, FALLBACK_MODEL_NAME)

# NOTE: risk-by-file-stem lookup used to be a private duplicate here.
# Moved to app.graph_analysis.risk_scoring.get_risk_by_file_stem() (Step 7)
# so this chain and orchestration/graph.py's risk_assessment_node share
# ONE cache instead of each re-running git log for the same data.


# ---------------------------------------------------------------------------
# Reassembly + node-id helpers (shared by both primary reassembly and
# call-graph-expansion chunk lookup, since both need "give me the full text
# for this function, whether it's one chunk or several split parts").
#
# All take `collection`/`call_graph` as explicit arguments rather than
# reading module globals -- this is what makes them unit-testable with a
# fake collection object, no real Chroma/model needed.
# ---------------------------------------------------------------------------

_PART_PATTERN = re.compile(r"^(.*)\[part (\d+)/(\d+)\]$")


def _base_name(name: str) -> str:
    """'request[part 1/2]' -> 'request'; a plain name is returned unchanged."""
    match = _PART_PATTERN.match(name)
    return match.group(1) if match else name


def _part_number(name: str) -> int:
    """'request[part 2/2]' -> 2; a plain (non-split) name is treated as part 1."""
    match = _PART_PATTERN.match(name)
    return int(match.group(2)) if match else 1


def _node_id(file_path: str, name: str, class_name: str | None) -> str:
    """Build a call-graph node id, e.g. '_client.py::Client.send' or '_utils.py::helper'."""
    qualified = f"{class_name}.{name}" if class_name else name
    return f"{file_path}::{qualified}"


def _parse_node_id(node_id: str) -> tuple[str, str | None, str]:
    """Inverse of _node_id: 'file.py::Class.method' -> ('file.py', 'Class', 'method')."""
    file_path, qualified = node_id.split("::", 1)
    if "." in qualified:
        class_name, name = qualified.split(".", 1)
        return file_path, class_name, name
    return file_path, None, qualified


def _fetch_full_chunk(
    collection: chromadb.Collection, file_path: str, class_name: str | None, base_name: str
) -> Document | None:
    """
    Fetch and reassemble ALL chunks belonging to one logical function/method
    -- whether it was stored as a single chunk or split into [part N/M]
    pieces. Works for both primary-chunk reassembly and call-graph-expansion
    lookups, since both boil down to "give me this function's full text."
    Returns None if no matching chunk exists (e.g. a call-graph target that
    was never indexed).
    """
    where = {"$and": [{"file_path": file_path}, {"parent_class": class_name or ""}]}
    results = collection.get(where=where)

    matching = [
        (name, text, meta)
        for name, text, meta in zip(results["ids"], results["documents"], results["metadatas"])
        if _base_name(meta["name"]) == base_name
    ]
    if not matching:
        return None

    matching.sort(key=lambda item: _part_number(item[2]["name"]))
    full_text = "\n".join(text for _, text, _ in matching)
    first_meta, last_meta = matching[0][2], matching[-1][2]

    return Document(
        page_content=full_text,
        metadata={
            "file_path": file_path,
            "name": base_name,
            "parent_class": class_name or "",
            "start_line": first_meta["start_line"],
            "end_line": last_meta["end_line"],
        },
    )


def _reassemble(collection: chromadb.Collection, docs: list[Document]) -> list[Document]:  # DEPRECATED
    # No longer called anywhere -- superseded by the dedupe-by-identity-
    # before-cutoff logic now built directly into _retrieve_and_expand,
    # which needed to reassemble at the SAME point it deduplicates (before
    # the k cutoff), not as a separate pass afterward like this function
    # did. Left here per project convention of marking, not removing,
    # dead code.
    """Reassemble any split chunks among the raw retriever results."""
    reassembled = []
    for doc in docs:
        meta = doc.metadata
        full = _fetch_full_chunk(collection, meta["file_path"], meta.get("parent_class") or None, _base_name(meta["name"]))
        reassembled.append(full if full else doc)
    return reassembled


# ---------------------------------------------------------------------------
# Retrieval + call-graph expansion
# ---------------------------------------------------------------------------

def _retrieve_and_expand(collection: chromadb.Collection, call_graph: nx.DiGraph, question: str) -> dict:
    """
    Retrieves the top-k most relevant FUNCTIONS (not chunks), then expands
    1 hop via the call graph.

    Found via real testing (not a hypothetical): if a function is split
    into multiple chunks (Step 4's sliding-window fallback for oversized
    functions), each part is a SEPARATE embedded vector that competes
    independently for one of only TOP_K retrieval slots. This has two bad
    consequences if left naive: (1) a single split function can win
    MULTIPLE k-slots against itself (both its parts scoring well), wasting
    k on one function instead of spreading it across distinct relevant
    ones; (2) a split function competing against a near-duplicate WHOLE
    chunk (e.g. Client.request vs AsyncClient.request, structurally
    almost identical) can lose entirely if neither of its individual
    parts outscores the whole competitor, even though the split function
    -- reassembled -- would have been the better overall match.

    Fix (part 1): overfetch raw chunks (TOP_K * OVERFETCH_MULTIPLIER), then
    deduplicate by FUNCTION IDENTITY (file_path, parent_class, base name)
    across the ENTIRE overfetched window before cutting to TOP_K -- not
    stopping the instant TOP_K distinct identities are found. An earlier
    version of this fix stopped too early: if the first few raw hits
    already happened to be TOP_K distinct functions (no redundant split
    parts among them), the loop broke immediately, never scanning further
    into the window even though a MORE relevant function was sitting a
    few ranks lower. Confirmed via real data: "Client.request" ranked
    4th, one place past where the old loop stopped.

    Fix (part 2): a small, narrowly-scoped exact-identifier boost. Pure
    semantic similarity can rank a near-duplicate SIBLING (AsyncClient.request)
    ahead of the EXACT method literally named in the question
    (Client.request), since their content is nearly identical -- the only
    reliable disambiguator is the literal name, which embeddings don't
    specially privilege. If the question contains an exact "ClassName.method"
    substring matching a candidate found anywhere in the overfetched
    window, that candidate is guaranteed inclusion as bonus context, even
    if its raw semantic rank fell outside the natural top-k. Deliberately
    narrow: only the dotted Class.method pattern (case-sensitive, low
    false-positive risk -- same reasoning as extract_target_files' class-
    name matching in classifier.py), NOT bare method names alone (too
    ambiguous/generic, real false-positive risk without a class qualifier
    to disambiguate). A full fix (proper hybrid dense+lexical retrieval or
    reranking) is real Phase 2 work -- explicitly out of scope per the
    design doc's Non-Goals ("Reranking, hybrid multi-collection retrieval").
    """
    try:
        raw_results = collection.query(query_texts=[question], n_results=TOP_K * OVERFETCH_MULTIPLIER)
        raw_metas = raw_results["metadatas"][0]

        seen_identities = set()
        ordered_identities = []
        for meta in raw_metas:
            identity = (meta["file_path"], meta.get("parent_class") or None, _base_name(meta["name"]))
            if identity not in seen_identities:
                seen_identities.add(identity)
                ordered_identities.append(identity)
            # NOTE: no early break here -- the whole overfetched window is
            # scanned so a later-but-more-relevant distinct function isn't
            # missed just because the first few ranks already happened to
            # be TOP_K distinct functions.

        selected_identities = ordered_identities[:TOP_K]

        # Exact-identifier boost: does the question literally name a
        # "ClassName.method" pair matching something in the FULL scanned
        # window (not just the natural top-k)? If so, and it isn't already
        # selected, include it as one extra piece of context rather than
        # displacing a natural top-k pick -- we have no principled basis
        # for deciding which natural pick to drop, so we err toward more
        # grounding, not less.
        for file_path, parent_class, base_name in ordered_identities:
            if parent_class and f"{parent_class}.{base_name}" in question:
                identity = (file_path, parent_class, base_name)
                if identity not in selected_identities:
                    selected_identities.append(identity)
                break  # at most one boosted addition -- stay lightweight

        primary_docs = []
        for file_path, parent_class, base_name in selected_identities:
            doc = _fetch_full_chunk(collection, file_path, parent_class, base_name)
            if doc:
                primary_docs.append(doc)

        visited_node_ids = {
            _node_id(d.metadata["file_path"], d.metadata["name"], d.metadata.get("parent_class") or None)
            for d in primary_docs
        }

        expanded_docs: list[Document] = []
        for doc in primary_docs:
            node_id = _node_id(doc.metadata["file_path"], doc.metadata["name"], doc.metadata.get("parent_class") or None)
            if node_id not in call_graph:
                continue
            for target_id in call_graph.successors(node_id):  # 1 hop only, per design decision
                if target_id in visited_node_ids:
                    continue
                visited_node_ids.add(target_id)
                file_path, class_name, base_name = _parse_node_id(target_id)
                expanded_doc = _fetch_full_chunk(collection, file_path, class_name, base_name)
                if expanded_doc:
                    expanded_docs.append(expanded_doc)

        return {"question": question, "primary_docs": primary_docs, "expanded_docs": expanded_docs}
    except Exception as exc:
        # Boundary catch: EVERYTHING chroma-dependent in this function
        # (the top-k query, every _fetch_full_chunk call during
        # reassembly/expansion) funnels through here. Call-graph traversal
        # itself (call_graph.successors) is pure in-memory networkx and
        # already defensively guarded (`if node_id not in call_graph`),
        # so it's not a realistic failure surface -- this is honestly a
        # retrieval-layer boundary, not a call-graph one.
        raise RetrievalError(f"Failed to retrieve relevant code for the question: {exc}", cause=exc) from exc


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are RepoSage, explaining code from the httpx library. "
            "Answer ONLY using the code provided below. If the context doesn't "
            "contain enough information to answer, say so explicitly rather "
            "than guessing. Reference specific file names when relevant.\n\n"
            "{context}",
        ),
        ("human", "{question}"),
    ]
)


def _format_context(inputs: dict) -> dict:
    def block(label: str, docs: list[Document]) -> str:
        if not docs:
            return ""
        parts = [f"--- {label} ---"]
        for d in docs:
            m = d.metadata
            loc = f"{m['file_path']}" + (f"::{m['parent_class']}.{m['name']}" if m.get("parent_class") else f"::{m['name']}")
            parts.append(f"[{loc}, lines {m['start_line']}-{m['end_line']}]\n{d.page_content}")
        return "\n\n".join(parts)

    context = block("PRIMARY MATCH", inputs["primary_docs"]) + "\n\n" + block("RELATED CODE (called by primary match)", inputs["expanded_docs"])
    return {"question": inputs["question"], "context": context}


# ---------------------------------------------------------------------------
# Citation + risk-note construction (after the LLM -- see module docstring)
# ---------------------------------------------------------------------------

def _build_final_response(risk_by_file_stem: dict, result: dict) -> CodeExplanationResult:
    def citation(d: Document) -> str:
        m = d.metadata
        loc = f"{m['parent_class']}.{m['name']}" if m.get("parent_class") else m["name"]
        return f"{m['file_path']} :: {loc} (lines {m['start_line']}-{m['end_line']})"

    citations = [citation(d) for d in result["primary_docs"]] + [citation(d) for d in result["expanded_docs"]]

    # Risk note: ONLY for primary-match files (not expanded ones -- the
    # question is "about" the primary file, expanded chunks are supporting
    # context), and ONLY when risk_level == "high" (silent otherwise, per
    # the Step 4 design decision -- don't show risk for every answer, only
    # when it's important enough not to miss).
    #
    # We collect BOTH a human-readable text note (for folding into the
    # answer) AND the structured RiskScore data (risk_details) -- added in
    # Step 7 so API consumers get a machine-readable risk payload here too,
    # not just prose they'd have to parse.
    risk_notes = []
    risk_details = []
    seen_stems = set()
    for doc in result["primary_docs"]:
        file_stem = doc.metadata["file_path"].removesuffix(".py")
        if file_stem in seen_stems:
            continue
        seen_stems.add(file_stem)
        score = risk_by_file_stem.get(file_stem)
        if score and score.risk_level == "high":
            risk_notes.append(f"Note: {file_stem}.py is HIGH risk to change -- {score.rationale}")
            risk_details.append(score)  # the real RiskScore object -- no dump/reconstruct round-trip needed

    return CodeExplanationResult(
        answer=result["llm_answer"],
        citations=citations,
        risk_notes=risk_notes,
        risk_details=risk_details,  # [] when no HIGH-risk primary file, never None -- see format_response_node
    )



# ---------------------------------------------------------------------------
# Full chain. Constructing this object is CHEAP (just wiring Runnables
# together) -- the lambdas below only call the lazy _get_*() getters when
# the chain is actually invoked, not when this module is imported.
# ---------------------------------------------------------------------------

code_explanation_chain = (
    RunnableLambda(lambda question: _retrieve_and_expand(_get_collection(), _get_call_graph(), question))
    | RunnableParallel(
        llm_answer=RunnableLambda(_format_context) | _PROMPT | RunnableLambda(lambda x: _get_llm().invoke(x)) | StrOutputParser(),
        primary_docs=RunnableLambda(lambda x: x["primary_docs"]),
        expanded_docs=RunnableLambda(lambda x: x["expanded_docs"]),
    )
    | RunnableLambda(lambda result: _build_final_response(get_risk_by_file_stem(), result))
)


if __name__ == "__main__":
    response = code_explanation_chain.invoke("What does Client.send do?")
    print(response.answer)
    if response.risk_notes:
        print()
        for note in response.risk_notes:
            print(note)
    print("\nSources:")
    for c in response.citations:
        print(" -", c)