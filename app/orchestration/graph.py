"""
Step 7 — LangGraph nodes, conditional routing, and graph assembly.

Design decisions locked in for this step:
  - Each node returns only the fields it writes (a PARTIAL state update),
    not the full state -- see state.py's docstring for why. This is what
    keeps nodes decoupled: code_explanation_node doesn't need to know
    anything about target_files, risk_assessment_node doesn't need to
    know anything about citations.

  - route_by_intent is a small, SEPARATE function from classify_node
    itself -- it translates state["intent"] into a graph node NAME
    (a string LangGraph understands), which is a detail of how THIS
    graph happens to be wired, not something the classifier should need
    to know about. Keeps classify_query/extract_target_files reusable
    and testable independent of graph topology.

  - format_response_node is its OWN graph node (not something done later
    in Step 8's FastAPI layer), because normalizing two structurally
    different result shapes into one consistent final_output is still a
    business-logic decision (what the unified shape MEANS), not a
    presentation concern. Keeps main.py's stated convention true:
    "contains no business logic."

  - risk_assessment_node returns a CLARIFICATION message (not an error,
    not a default-to-all-9-files dump) when no target file was extracted
    from the question -- per the explicit design decision made before
    writing this code.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.chains.code_explanation_chain import code_explanation_chain
from app.classification.classifier import classify_query, extract_target_files, get_classifier
from app.config import TARGET_FILES
from app.graph_analysis.risk_scoring import get_risk_by_file_stem
from app.orchestration.state import GraphState
from app.schemas import QueryResponse, RiskScore


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def classify_node(state: GraphState) -> dict:
    """Entry node: classify intent + extract any mentioned target files."""
    classification = classify_query(get_classifier(), state["question"])
    target_files = extract_target_files(state["question"])
    return {"intent": classification.intent, "target_files": target_files}


def route_by_intent(state: GraphState) -> str:
    """
    The actual routing decision. Returns a NODE NAME (graph wiring detail),
    translated from state["intent"] (a classifier concept) -- kept separate
    from classify_node so the classifier itself never needs to know what
    the graph's nodes happen to be called.
    """
    if state["intent"] == "risk_assessment":
        return "risk_assessment_node"
    return "code_explanation_node"


def code_explanation_node(state: GraphState) -> dict:
    """Invokes Step 5's chain exactly as built -- no changes needed here."""
    result = code_explanation_chain.invoke(state["question"])
    return {"code_explanation_result": result}


def risk_assessment_node(state: GraphState) -> dict:
    """
    NEW logic for this step: compute_risk_scores() returns ALL 9 files --
    this node filters to just the file(s) the question actually asked
    about, using the (possibly multi-file, per Step 6's fix) target_files
    list.

    In practice, extract_target_files only ever returns members of
    TARGET_FILES (both its filename-matching and class-name-matching paths
    are scoped to that same closed set), so target_files SHOULD always be
    a clean subset of what get_risk_by_file_stem() knows about. But this
    node's job is to behave correctly given ANY list[str] it's handed --
    not to assume its only current caller is well-behaved -- so an
    unrecognized entry is surfaced explicitly rather than silently
    dropped, matching the same "ask, don't guess or hide" principle behind
    the empty-target_files clarification path below.
    """
    target_files = state.get("target_files", [])

    if not target_files:
        return {
            "risk_assessment_result": {
                "needs_clarification": True,
                "message": (
                    "Which file would you like a risk assessment for? "
                    "For example: 'How risky is it to change _exceptions.py?'"
                ),
            }
        }

    risk_by_stem = get_risk_by_file_stem()
    known = [stem for stem in target_files if stem in risk_by_stem]
    unknown = [stem for stem in target_files if stem not in risk_by_stem]

    if not known:
        # NONE of the requested files were recognized -- can't answer at
        # all, so this is a full clarification, same as the empty-list case.
        suggestions = ", ".join(f"{f}.py" for f in sorted(TARGET_FILES))
        return {
            "risk_assessment_result": {
                "needs_clarification": True,
                "message": (
                    f"I don't recognize {', '.join(unknown)} as one of the "
                    f"files I track. Did you mean one of: {suggestions}?"
                ),
            }
        }

    result = {
        "needs_clarification": False,
        "files": [risk_by_stem[stem].model_dump() for stem in known],
    }
    if unknown:
        # PARTIAL match: answer for what we recognized, but say so --
        # never silently narrow the answer without telling the user.
        result["unrecognized_files"] = unknown

    return {"risk_assessment_result": result}


def format_response_node(state: GraphState) -> dict:
    """
    Normalizes EITHER branch's result into one consistent QueryResponse
    object -- this is the ONLY place that needs to know both branches'
    internal shapes, AND the one place internal dicts (risk_assessment_result,
    code_explanation_result) get converted into the real schema. Step 8's
    FastAPI layer can just return final_output as-is (it's already a
    QueryResponse -- directly usable as a response_model).
    """
    if state["intent"] == "code_explanation":
        result = state["code_explanation_result"]
        answer = result.answer
        if result.risk_notes:
            # risk_notes are already short, human-readable text (HIGH-risk-only,
            # per Step 4's design decision) -- fold them into the answer text
            # in addition to (not instead of) the structured risk_details below.
            answer = answer + "\n\n" + "\n".join(result.risk_notes)
        return {
            "final_output": QueryResponse(
                intent="code_explanation",
                answer=answer,
                citations=result.citations,
                # [] (not None) when risk WAS checked but nothing was HIGH --
                # None is reserved for "risk wasn't computed at all" (the
                # risk_assessment clarification path below). result.risk_details
                # is ALREADY list[RiskScore] here -- no reconstruction needed,
                # since CodeExplanationResult carries the real objects through.
                risk_details=result.risk_details,
            )
        }

    # intent == "risk_assessment"
    result = state["risk_assessment_result"]
    if result["needs_clarification"]:
        return {
            "final_output": QueryResponse(
                intent="risk_assessment",
                answer=result["message"],
                citations=[],
                risk_details=None,
            )
        }

    scores = [RiskScore(**d) for d in result["files"]]
    answer = "\n\n".join(f"{s.file}.py: {s.risk_level.upper()} risk -- {s.rationale}" for s in scores)
    if result.get("unrecognized_files"):
        # Partial match: never silently narrow the answer -- tell the user
        # what we couldn't find, rather than just answering for less than
        # they asked.
        unrecognized = ", ".join(result["unrecognized_files"])
        answer += f"\n\n(Note: I didn't recognize {unrecognized} as a tracked file, so it isn't included above.)"
    return {
        "final_output": QueryResponse(
            intent="risk_assessment",
            answer=answer,
            citations=[],
            risk_details=scores,
        )
    }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("classify", classify_node)
    graph.add_node("code_explanation_node", code_explanation_node)
    graph.add_node("risk_assessment_node", risk_assessment_node)
    graph.add_node("format_response", format_response_node)

    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify",
        route_by_intent,
        {
            "code_explanation_node": "code_explanation_node",
            "risk_assessment_node": "risk_assessment_node",
        },
    )
    graph.add_edge("code_explanation_node", "format_response")
    graph.add_edge("risk_assessment_node", "format_response")
    graph.add_edge("format_response", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    for q in [
        "What does Client.send do?",
        "How risky is it to change _exceptions.py?",
        "Is it risky to change anything?",  # no target file -> clarification path
    ]:
        result = app.invoke({"question": q})
        print(f"\n{q!r}")
        print(result["final_output"].model_dump())