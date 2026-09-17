"""
Step 7 — LangGraph state schema.

TypedDict, not a Pydantic model: LangGraph's node-return pattern is
PARTIAL updates (a node returns only the fields it's writing, e.g.
{"intent": "code_explanation"}), which are then merged into the running
state. TypedDict's `total=False` matches this naturally -- every field is
optional until some node actually writes it. A Pydantic model would need
every field satisfied (or defaulted) on every partial return, which
doesn't fit LangGraph's merge-as-you-go design.

Each field is written by exactly one node (see graph.py) and read by
whichever node(s) run after it -- this is how nodes stay decoupled from
each other: a node only needs to know the shape of the fields IT reads
and writes, not the full state's history.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from app.schemas import CodeExplanationResult, QueryResponse


class GraphState(TypedDict, total=False):
    question: str
    """The raw user question. Set once, at graph entry, never modified."""

    intent: Literal["code_explanation", "risk_assessment"]
    """Written by classify_node. Read by route_by_intent to pick the next node."""

    target_files: list[str]
    """
    Written by classify_node (via extract_target_files). Read by
    risk_assessment_node. May be an empty list -- code_explanation_node
    doesn't use this at all (it relies on semantic retrieval, not a
    target-file lookup).
    """

    code_explanation_result: CodeExplanationResult
    """
    Written by code_explanation_node. The exact CodeExplanationResult
    object Step 5's code_explanation_chain returns, unchanged -- crosses
    the module boundary from chains/ into orchestration/ as a real
    schema, not an untyped dict (see schemas.py for why).
    """

    risk_assessment_result: dict
    """
    Written by risk_assessment_node. Either a clarification request
    ({"needs_clarification": True, "message": str}) when no target file
    was found, or {"needs_clarification": False, "files": list[dict]}
    with one entry per requested file's risk data.
    """

    final_output: QueryResponse
    """
    Written by format_response_node (last node in both branches). The
    ONE consistent QueryResponse object returned regardless of which
    intent was served -- this is what Step 8's FastAPI layer returns to
    the user (directly usable as a response_model) without needing any
    intent-specific branching of its own.
    """