"""
Tests for app.classification.classifier.

classify_query is tested with a FakeClassifier stand-in (no real LLM call
needed -- we're testing OUR wiring: does classify_query pass the question
through correctly and return what the classifier gives back?). Real LLM
classification quality (does Gemini actually pick the right intent for a
given question) isn't something a unit test can meaningfully assert against
anyway -- that's a judgment call, not a deterministic function, and belongs
in manual/eval testing, not this suite.

extract_target_file is pure logic with a real, testable contract (which of
our 9 known filenames appears in the question), so it's tested thoroughly.
"""

from __future__ import annotations

from app.classification.classifier import classify_query, extract_target_files
from app.schemas import QueryClassification


class _FakeClassifier:
    """Stands in for the real (prompt | structured_llm) Runnable."""

    def __init__(self, intent: str):
        self._intent = intent
        self.last_input = None

    def invoke(self, input_dict):
        self.last_input = input_dict
        return QueryClassification(intent=self._intent)


def test_classify_query_passes_question_through_and_returns_result():
    fake = _FakeClassifier(intent="code_explanation")
    result = classify_query(fake, "What does Client.send do?")
    assert result.intent == "code_explanation"
    assert fake.last_input == {"question": "What does Client.send do?"}


def test_classify_query_returns_risk_assessment_intent():
    fake = _FakeClassifier(intent="risk_assessment")
    result = classify_query(fake, "How risky is it to change _exceptions.py?")
    assert result.intent == "risk_assessment"


def test_extract_target_files_finds_bare_filename():
    assert extract_target_files("How risky is it to change _exceptions?") == ["_exceptions"]


def test_extract_target_files_finds_filename_with_py_suffix():
    assert extract_target_files("Is _exceptions.py safe to modify?") == ["_exceptions"]


def test_extract_target_files_is_case_insensitive():
    assert extract_target_files("Is _CLIENT risky to touch?") == ["_client"]


def test_extract_target_files_returns_empty_list_when_no_known_file_mentioned():
    assert extract_target_files("What does the send method do?") == []


def test_extract_target_files_returns_empty_list_for_unrelated_question():
    assert extract_target_files("What's the weather like today?") == []


def test_extract_target_files_finds_multiple_files_in_order_mentioned():
    """
    Regression test for the fixed bug: the old single-match version could
    only return ONE of two mentioned files, non-deterministically (set
    iteration order). Must now return BOTH, in the order they appear.
    """
    result = extract_target_files("Does _client.py depend on _models.py?")
    assert result == ["_client", "_models"]


def test_extract_target_files_order_follows_question_not_alphabetical():
    """_urls is mentioned before _auth in the question -- order must reflect that, not alphabetical sort."""
    result = extract_target_files("Does _urls.py get used by _auth.py?")
    assert result == ["_urls", "_auth"]


def test_extract_target_files_resolves_class_name_to_its_file():
    """'Client' (class) must resolve to '_client' (the file it's defined in), via build_class_to_file_map."""
    result = extract_target_files("Is the Client class risky to change?")
    assert result == ["_client"]


def test_extract_target_files_matches_filename_without_leading_underscore_if_py_qualified():
    """People naturally drop the leading underscore in prose ('client.py' not '_client.py') -- must still match."""
    result = extract_target_files("Does client.py depend on models.py?")
    assert result == ["_client", "_models"]


def test_extract_target_files_does_not_false_positive_on_generic_bare_word():
    """
    Regression guard: 'types' alone (no leading underscore, no .py) is a
    common English word and must NOT match _types -- the .py suffix is
    what disambiguates a real file reference from ordinary prose.
    """
    result = extract_target_files("What types of arguments does send take?")
    assert result == []


def test_extract_target_files_matches_generic_word_when_py_qualified():
    """The SAME generic word ('types') DOES match once qualified with .py -- proves the disambiguation rule works both ways."""
    result = extract_target_files("How risky is it to change types.py?")
    assert result == ["_types"]