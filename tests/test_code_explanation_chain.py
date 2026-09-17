"""
Tests for the pure logic in app.chains.code_explanation_chain: reassembly,
node-id parsing, call-graph expansion, and citation/risk-note construction.

These use FakeCollection (see conftest.py) and small real networkx graphs
instead of real Chroma/embedding model/LLM -- we're testing OUR wiring
logic (does reassembly sort parts correctly? does expansion respect the
1-hop limit and dedupe visited nodes? does the risk note only appear for
HIGH?), not whether Chroma or an LLM API works.
"""

from __future__ import annotations

import networkx as nx

from app.errors import RetrievalError
from app.schemas import RiskScore

from app.chains.code_explanation_chain import (
    _base_name,
    _build_final_response,
    _fetch_full_chunk,
    _node_id,
    _parse_node_id,
    _part_number,
    _retrieve_and_expand,
)


# ---------------------------------------------------------------------------
# Node-id / naming helpers
# ---------------------------------------------------------------------------

def test_base_name_strips_split_suffix():
    assert _base_name("request[part 1/2]") == "request"
    assert _base_name("send") == "send"


def test_part_number_extracts_correctly():
    assert _part_number("request[part 2/3]") == 2
    assert _part_number("send") == 1  # non-split names default to part 1


def test_node_id_round_trip_for_method():
    node_id = _node_id("_client.py", "send", "Client")
    assert node_id == "_client.py::Client.send"
    assert _parse_node_id(node_id) == ("_client.py", "Client", "send")


def test_node_id_round_trip_for_bare_function():
    node_id = _node_id("_utils.py", "helper", None)
    assert node_id == "_utils.py::helper"
    assert _parse_node_id(node_id) == ("_utils.py", None, "helper")


# ---------------------------------------------------------------------------
# Reassembly
# ---------------------------------------------------------------------------

def test_fetch_full_chunk_reassembles_split_parts_in_order(fake_collection):
    rows = [
        ("id2", "part two text", {"file_path": "_client.py", "parent_class": "Client",
                                   "name": "request[part 2/2]", "start_line": 10, "end_line": 15}),
        ("id1", "part one text", {"file_path": "_client.py", "parent_class": "Client",
                                   "name": "request[part 1/2]", "start_line": 1, "end_line": 9}),
    ]
    collection = fake_collection(rows)
    doc = _fetch_full_chunk(collection, "_client.py", "Client", "request")
    assert doc is not None
    # Must be reassembled IN ORDER (part 1 before part 2), regardless of storage order.
    assert doc.page_content == "part one text\npart two text"
    assert doc.metadata["start_line"] == 1
    assert doc.metadata["end_line"] == 15


def test_fetch_full_chunk_handles_non_split_single_chunk(fake_collection):
    rows = [("id1", "def send(): pass", {"file_path": "_client.py", "parent_class": "Client",
                                          "name": "send", "start_line": 5, "end_line": 6})]
    collection = fake_collection(rows)
    doc = _fetch_full_chunk(collection, "_client.py", "Client", "send")
    assert doc.page_content == "def send(): pass"


def test_fetch_full_chunk_returns_none_when_no_match(fake_collection):
    collection = fake_collection([])
    doc = _fetch_full_chunk(collection, "_client.py", "Client", "missing")
    assert doc is None


# ---------------------------------------------------------------------------
# Retrieval + call-graph expansion
# ---------------------------------------------------------------------------

def test_retrieve_and_expand_pulls_in_direct_call_graph_successors(fake_collection):
    rows = [
        ("id1", "def send(): pass", {"file_path": "_client.py", "parent_class": "Client",
                                      "name": "send", "start_line": 1, "end_line": 2}),
        ("id2", "def build(): pass", {"file_path": "_client.py", "parent_class": "Client",
                                       "name": "build", "start_line": 3, "end_line": 4}),
    ]
    collection = fake_collection(rows, query_results=[rows[0]])  # only "send" is the top-k match
    graph = nx.DiGraph()
    graph.add_edge("_client.py::Client.send", "_client.py::Client.build")

    result = _retrieve_and_expand(collection, graph, "what does send do?")
    assert len(result["primary_docs"]) == 1
    assert result["primary_docs"][0].metadata["name"] == "send"
    assert len(result["expanded_docs"]) == 1
    assert result["expanded_docs"][0].metadata["name"] == "build"


def test_retrieve_and_expand_does_not_duplicate_a_chunk_already_primary(fake_collection):
    """If a call graph edge points BACK to a chunk that's already a primary match, don't add it again."""
    rows = [
        ("id1", "def send(): pass", {"file_path": "_client.py", "parent_class": "Client",
                                      "name": "send", "start_line": 1, "end_line": 2}),
    ]
    collection = fake_collection(rows)
    graph = nx.DiGraph()
    graph.add_edge("_client.py::Client.send", "_client.py::Client.send")  # self-cycle, edge case

    result = _retrieve_and_expand(collection, graph, "what does send do?")
    assert len(result["expanded_docs"]) == 0  # send is already primary -- must not be re-added


def test_retrieve_and_expand_handles_node_with_no_call_graph_entry(fake_collection):
    """A retrieved chunk that isn't in the call graph at all (e.g. never called by/calling anything) must not crash."""
    rows = [
        ("id1", "def isolated(): pass", {"file_path": "_utils.py", "parent_class": "",
                                          "name": "isolated", "start_line": 1, "end_line": 2}),
    ]
    collection = fake_collection(rows)
    graph = nx.DiGraph()  # empty -- isolated's node doesn't exist in it

    result = _retrieve_and_expand(collection, graph, "what does isolated do?")
    assert result["expanded_docs"] == []


def test_retrieve_and_expand_wraps_chroma_failure_as_retrieval_error(fake_collection):
    """A genuine Chroma failure (corrupted DB, embedding call failure, etc.) must surface as RetrievalError, not a raw exception."""

    class BrokenCollection:
        def query(self, *args, **kwargs):
            raise RuntimeError("simulated chroma connection failure")

    graph = nx.DiGraph()
    try:
        _retrieve_and_expand(BrokenCollection(), graph, "what does send do?")
        assert False, "expected RetrievalError"
    except RetrievalError as exc:
        assert exc.retryable is True
        assert isinstance(exc.cause, RuntimeError)


def test_retrieve_and_expand_scans_full_overfetch_window_not_just_first_k_distinct(fake_collection):
    """
    Regression test for a real production bug, reproduced using the
    ACTUAL ranking order observed live: "Client.request" ranked 4th, but
    the first 3 raw ranks already happened to be 3 DISTINCT functions
    (no redundant split parts among them to skip past), so an earlier
    version of the fix broke immediately at rank 3 -- never scanning far
    enough to see rank 4 at all, even though it was well within the
    12-chunk overfetch window. The fix must scan the FULL window before
    cutting to TOP_K, not stop the instant TOP_K distinct identities
    appear.
    """
    rows = [
        ("id1", "async _send_single_request", {"file_path": "_client.py", "parent_class": "AsyncClient",
                                                 "name": "_send_single_request", "start_line": 1717, "end_line": 1749}),
        ("id2", "sync _send_single_request", {"file_path": "_client.py", "parent_class": "Client",
                                                "name": "_send_single_request", "start_line": 1001, "end_line": 1034}),
        ("id3", "async request part 1", {"file_path": "_client.py", "parent_class": "AsyncClient",
                                          "name": "request[part 1/2]", "start_line": 1485, "end_line": 1529}),
        ("id4", "sync request part 1", {"file_path": "_client.py", "parent_class": "Client",
                                         "name": "request[part 1/2]", "start_line": 771, "end_line": 815}),
        ("id5", "sync request part 2", {"file_path": "_client.py", "parent_class": "Client",
                                         "name": "request[part 2/2]", "start_line": 813, "end_line": 825}),
    ]
    collection = fake_collection(rows, query_results=rows)
    graph = nx.DiGraph()

    result = _retrieve_and_expand(collection, graph, "What does Client.request do?")
    identities = [(d.metadata["parent_class"], d.metadata["name"]) for d in result["primary_docs"]]

    assert ("Client", "request") in identities  # must be found -- was completely missing before this fix


def test_exact_identifier_boost_stays_scoped_to_dotted_class_method_pattern(fake_collection):
    """The boost must NOT trigger on a bare method name alone (no class qualifier) -- too ambiguous/generic, same false-positive risk class as the classifier's underscore-matching guard."""
    rows = [
        ("id1", "unrelated 1", {"file_path": "_client.py", "parent_class": "AsyncClient",
                                 "name": "_send_single_request", "start_line": 1, "end_line": 2}),
        ("id2", "unrelated 2", {"file_path": "_client.py", "parent_class": "Client",
                                 "name": "_send_single_request", "start_line": 3, "end_line": 4}),
        ("id3", "unrelated 3", {"file_path": "_client.py", "parent_class": "AsyncClient",
                                 "name": "send", "start_line": 5, "end_line": 6}),
        ("id4", "the target, unboosted", {"file_path": "_client.py", "parent_class": "Client",
                                           "name": "request", "start_line": 7, "end_line": 8}),
    ]
    collection = fake_collection(rows, query_results=rows)
    graph = nx.DiGraph()

    # No class prefix ("Client.") in this question -- just the bare word "request".
    result = _retrieve_and_expand(collection, graph, "What does the request method do?")
    identities = [(d.metadata["parent_class"], d.metadata["name"]) for d in result["primary_docs"]]

    assert len(result["primary_docs"]) == 3  # natural top-3 only, no boosted extra
    assert ("Client", "request") not in identities  # correctly NOT force-included without an exact "Client.request" mention


def test_retrieve_and_expand_does_not_waste_k_slots_on_one_split_functions_parts(fake_collection):
    """
    Regression test for a real bug found in production: a single function
    split into multiple chunks (Step 4's sliding-window fallback) used to
    have EACH part compete independently for a k-slot, meaning both parts
    of ONE function could win 2 of the 3 slots, crowding out a genuinely
    distinct, relevant THIRD function that ranked just below them.

    Rows are pre-ordered by "relevance" (matching FakeCollection.query's
    contract): request's two parts rank highest, build_request third,
    send fourth -- a naive top-3-raw-chunks approach would return
    [request part1, request part2, build_request] (only 2 DISTINCT
    functions), silently dropping `send` even though TOP_K=3 was
    configured. The fix must instead return 3 DISTINCT functions by
    continuing into the overfetch window.
    """
    rows = [
        ("id1", "part one", {"file_path": "_client.py", "parent_class": "Client",
                              "name": "request[part 1/2]", "start_line": 1, "end_line": 5}),
        ("id2", "part two", {"file_path": "_client.py", "parent_class": "Client",
                              "name": "request[part 2/2]", "start_line": 6, "end_line": 10}),
        ("id3", "def build_request(): pass", {"file_path": "_client.py", "parent_class": "Client",
                                               "name": "build_request", "start_line": 20, "end_line": 21}),
        ("id4", "def send(): pass", {"file_path": "_client.py", "parent_class": "Client",
                                      "name": "send", "start_line": 30, "end_line": 31}),
    ]
    collection = fake_collection(rows, query_results=rows)  # all 4 available in the overfetch window
    graph = nx.DiGraph()

    result = _retrieve_and_expand(collection, graph, "what does Client.request do?")
    primary_names = [d.metadata["name"] for d in result["primary_docs"]]

    assert len(result["primary_docs"]) == 3  # TOP_K distinct functions, not TOP_K raw chunks
    assert primary_names == ["request", "build_request", "send"]  # request reassembled to ONE entry, "send" wasn't crowded out


# ---------------------------------------------------------------------------
# Citations + risk notes
# ---------------------------------------------------------------------------

class _FakeDoc:
    def __init__(self, metadata):
        self.metadata = metadata


def _make_score(risk_level, file_stem="_test", rationale="test rationale"):
    """Real RiskScore instance -- required now that CodeExplanationResult.risk_details
    validates entries as list[RiskScore]; a loose duck-typed stand-in would fail validation."""
    return RiskScore(file=file_stem, dependents=0, dependent_files=[], commit_count=0,
                      risk_level=risk_level, rationale=rationale)


def test_risk_note_appears_only_for_high_risk_primary_file():
    primary = [_FakeDoc({"file_path": "_types.py", "parent_class": "", "name": "helper",
                          "start_line": 1, "end_line": 2})]
    risk_by_stem = {"_types": _make_score("high", "_types")}
    result = _build_final_response(risk_by_stem, {"primary_docs": primary, "expanded_docs": [], "llm_answer": "answer"})
    assert len(result.risk_notes) == 1
    assert "_types.py" in result.risk_notes[0]
    # risk_details carries the SAME finding, structured -- added in Step 7
    # so an API consumer isn't forced to parse the text note.
    assert len(result.risk_details) == 1
    assert result.risk_details[0].risk_level == "high"


def test_no_risk_note_for_medium_or_low_risk():
    primary = [_FakeDoc({"file_path": "_client.py", "parent_class": "", "name": "send",
                          "start_line": 1, "end_line": 2})]
    risk_by_stem = {"_client": _make_score("low", "_client")}
    result = _build_final_response(risk_by_stem, {"primary_docs": primary, "expanded_docs": [], "llm_answer": "answer"})
    assert result.risk_notes == []
    assert result.risk_details == []


def test_no_risk_note_for_expanded_only_high_risk_file():
    """Risk notes are ONLY for primary-match files, per the design decision -- expanded chunks don't trigger them."""
    primary = [_FakeDoc({"file_path": "_client.py", "parent_class": "", "name": "send",
                         "start_line": 1, "end_line": 2})]
    expanded = [_FakeDoc({"file_path": "_types.py", "parent_class": "", "name": "helper",
                          "start_line": 1, "end_line": 2})]
    risk_by_stem = {"_client": _make_score("low", "_client"), "_types": _make_score("high", "_types")}
    result = _build_final_response(
        risk_by_stem, {"primary_docs": primary, "expanded_docs": expanded, "llm_answer": "answer"}
    )
    assert result.risk_notes == []
    assert result.risk_details == []


def test_citations_include_both_primary_and_expanded_docs():
    primary = [_FakeDoc({"file_path": "_client.py", "parent_class": "Client", "name": "send",
                         "start_line": 1, "end_line": 2})]
    expanded = [_FakeDoc({"file_path": "_client.py", "parent_class": "Client", "name": "build",
                          "start_line": 3, "end_line": 4})]
    result = _build_final_response({}, {"primary_docs": primary, "expanded_docs": expanded, "llm_answer": "answer"})
    assert len(result.citations) == 2
    assert any("Client.send" in c for c in result.citations)
    assert any("Client.build" in c for c in result.citations)