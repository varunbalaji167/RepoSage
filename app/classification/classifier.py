"""
Step 6 — structured-output intent classifier.

Design decisions locked in for this step:
  - Schema is narrow: intent ONLY (see QueryClassification in schemas.py).
    No target filename/function extraction bundled into the same LLM call
    -- classification (ambiguous natural language, needs real judgment)
    and extraction (matching against a small, known, closed set of 9
    filenames) are different kinds of problems. Combining them into one
    schema means one validation failure can't tell you which part broke.

  - Filename extraction uses NO LLM at all -- plain string matching
    against TARGET_FILES. An LLM call is the right tool when the answer
    space is open-ended and needs judgment; it's overkill (slower, costs
    money, can hallucinate a file that doesn't exist) when the answer
    space is a small, fixed, enumerable list we already have.

  - Uses Literal["code_explanation", "risk_assessment"] rather than a bare
    str, so an invalid/drifted LLM output fails validation immediately, at
    the classification step -- loud and early, not silently passed
    downstream into the router where the failure would be harder to trace
    back to its real cause.

  - Follows the same lazy-resource pattern as code_explanation_chain.py:
    the LLM client is built once, on first real use, and cached -- not at
    import time. classify_query() takes the classifier as an explicit
    argument rather than reading a module global, so it can be unit-tested
    with a fake/stubbed classifier, no real API call required.
"""

from __future__ import annotations

from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import TARGET_FILES
from app.graph_analysis.call_graph import build_class_to_file_map
from app.llm_utils import FALLBACK_MODEL_NAME, FallbackRunnable
from app.schemas import QueryClassification

LLM_MODEL_NAME = "gemini-3.5-flash-lite"


@lru_cache(maxsize=1)
def _get_class_to_file_map() -> dict[str, str]:
    return build_class_to_file_map()

_CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Classify the user's question about the httpx codebase into exactly "
            "one of two intents:\n\n"
            "code_explanation -- the user wants to understand what a piece of "
            "code does or how it works. Examples: 'What does Client.send do?', "
            "'Explain the merge_url method', 'How does redirect handling work?'\n\n"
            "risk_assessment -- the user wants to know how safe or risky it "
            "would be to CHANGE a file. Examples: 'How risky is it to change "
            "_exceptions.py?', 'Is it safe to modify _client.py?', 'What's the "
            "impact of touching _types.py?'\n\n"
            "If the question could plausibly be either, prefer code_explanation "
            "unless it explicitly asks about risk, safety, or impact of a change.",
        ),
        ("human", "{question}"),
    ]
)


@lru_cache(maxsize=1)
def get_classifier():
    """
    Lazy, cached: nothing here runs until first CALLED, so importing this
    module has no side effects (no API key/network required just to import).

    Wrapped in FallbackRunnable so a rate-limited primary model falls
    back to a lighter one automatically for this call -- see llm_utils.py.
    classify_query()'s call site (classifier.invoke(...)) doesn't need to
    change, since FallbackRunnable exposes the same .invoke() interface
    as a plain LCEL chain.
    """
    # NOTE: no `temperature` parameter -- see code_explanation_chain.py's
    # _get_llm() for why (deprecated on Gemini 3.6+, will error in future
    # model generations). Determinism here comes from the structured
    # output schema (Literal-typed intent) and the explicit system prompt.
    primary_llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME)
    fallback_llm = ChatGoogleGenerativeAI(model=FALLBACK_MODEL_NAME)

    primary_chain = _CLASSIFICATION_PROMPT | primary_llm.with_structured_output(QueryClassification)
    fallback_chain = _CLASSIFICATION_PROMPT | fallback_llm.with_structured_output(QueryClassification)

    return FallbackRunnable(primary_chain, fallback_chain, FALLBACK_MODEL_NAME)


def classify_query(classifier, question: str) -> QueryClassification:
    """
    Classify a question's intent. `classifier` is passed explicitly (not
    read from a module global) so this is directly unit-testable with a
    fake/stub classifier -- no real LLM call needed to test the wiring.
    """
    return classifier.invoke({"question": question})


def extract_target_files(question: str) -> list[str]:
    """
    Find ALL of our 9 known target files mentioned in the question -- by
    filename (e.g. "_client.py") OR by a class name defined in that file
    (e.g. "Client" -> _client) -- via plain substring matching, no LLM.

    Results are ordered by where each file's FIRST mention (filename or
    class name, whichever comes earlier) appears in the question text --
    not alphabetically, and deliberately not by iterating TARGET_FILES
    directly (it's a set; Python randomizes string hashing per process,
    so set iteration order isn't reliable for a stable result).

    Class-name matching is case-SENSITIVE (unlike filename matching),
    since class names are capitalized by convention -- this avoids, e.g.,
    an unrelated lowercase word coincidentally matching a class name.

    Filenames are matched in three forms: "_client" (bare, underscore-
    prefixed -- unambiguous on its own), "_client.py", and "client.py"
    (underscore DROPPED but ".py" REQUIRED). A bare word like "types" is
    NOT matched without ".py" -- "_types" is unambiguous, but "types"
    alone is a generic English word that could appear in unrelated
    questions ("what types of arguments does send take?"). Requiring
    ".py" when the underscore is dropped is what buys back the confidence
    lost from dropping it, without needing a per-file exception list.
    """
    question_lower = question.lower()
    best_position: dict[str, int] = {}

    def record(file_stem: str, position: int):
        if position != -1 and (file_stem not in best_position or position < best_position[file_stem]):
            best_position[file_stem] = position

    for file_stem in TARGET_FILES:
        stem = file_stem.lower()  # e.g. "_client"
        bare = stem.lstrip("_")   # e.g. "client"
        record(file_stem, question_lower.find(stem))
        record(file_stem, question_lower.find(f"{stem}.py"))
        record(file_stem, question_lower.find(f"{bare}.py"))  # underscore dropped, but .py required

    for class_name, file_stem in _get_class_to_file_map().items():
        record(file_stem, question.find(class_name))

    return [file_stem for file_stem, _ in sorted(best_position.items(), key=lambda item: item[1])]


if __name__ == "__main__":
    classifier = get_classifier()
    for q in [
        "What does Client.send do?",
        "How risky is it to change _exceptions.py?",
        "Is it safe to modify _client.py?",
        "Does _client.py depend on _models.py?",
    ]:
        result = classify_query(classifier, q)
        targets = extract_target_files(q)
        print(f"{q!r} -> intent={result.intent}, target_files={targets}")