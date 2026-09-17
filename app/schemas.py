"""Shared Pydantic contracts used across modules."""
from typing import Literal

from pydantic import BaseModel


class CodeChunk(BaseModel):
    """
    Output of Step 4's chunk_file() (chunker.py), consumed by
    vector_store.py's indexing pipeline -- crosses a real module boundary,
    same as this file's other schemas. Moved here from chunker.py, where
    it started as a @dataclass, for the same reason CodeExplanationResult
    and QueryResponse are here rather than living in the module that
    produces them.
    """
    file_path: str
    chunk_type: str  # "function" | "async_function" | "class" | "method" | "module_level" | "split"
    name: str         # function/class name, or "<module>" for module-level chunk
    start_line: int
    end_line: int
    text: str
    parent_class: str | None = None  # set when chunk_type == "method"


class RiskScore(BaseModel):
    file: str
    dependents: int
    dependent_files: list[str]
    commit_count: int
    risk_level: str  # "low" | "medium" | "high"
    rationale: str


class QueryClassification(BaseModel):
    """
    Output schema for Step 6's intent classifier.

    Deliberately narrow -- intent ONLY, no target filename/function
    extraction bundled in here. Classification (genuinely ambiguous
    natural language) and extraction (matching against a small, known,
    closed set of 9 filenames) are different kinds of problems needing
    different tools -- see extract_target_file() in classifier.py, which
    handles extraction without an LLM call at all.
    """
    intent: Literal["code_explanation", "risk_assessment"]


class CodeExplanationResult(BaseModel):
    """
    Output schema for Step 5's code_explanation_chain.invoke(). Crosses a
    real module boundary -- built and tested independently in
    code_explanation_chain.py, consumed by orchestration/graph.py (Step 7)
    -- same category as RiskScore/QueryClassification/QueryResponse, so it
    gets the same treatment rather than staying an untyped dict.

    risk_details reuses RiskScore directly (the SAME objects
    risk_by_file_stem already holds -- no dump-then-reconstruct needed,
    unlike the risk_assessment_node path which serializes through a plain
    dict internally).
    """
    answer: str
    citations: list[str] = []
    risk_notes: list[str] = []
    risk_details: list[RiskScore] = []


class QueryResponse(BaseModel):
    """
    Output schema for Step 7's orchestration graph (format_response_node)
    and Step 8's FastAPI response model -- the ONE consistent shape
    returned regardless of which intent branch actually ran.

    risk_details reuses RiskScore rather than a bare list[dict], since
    every entry here genuinely IS a RiskScore.model_dump() under the
    hood (from either risk_assessment_node's lookup or
    code_explanation_chain's own HIGH-risk check) -- typing it loosely
    would throw away information we already have in a proper schema.

    risk_details being None vs [] is a deliberate distinction (not
    interchangeable): None means risk wasn't computed at all for this
    response (the risk_assessment clarification path, when no target
    file was identified); [] means risk WAS checked and nothing was
    high-risk. An API consumer can tell "we don't know" apart from
    "we checked, all clear."
    """
    intent: Literal["code_explanation", "risk_assessment"]
    answer: str
    citations: list[str] = []
    risk_details: list[RiskScore] | None = None