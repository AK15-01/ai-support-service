import pytest

from rag.config import PROJECT_ROOT
from rag.models import REFUSAL_PREFIX
from rag.pipeline import load_questions
from rag.validation import normalize_expected_behavior

QUESTIONS = load_questions(PROJECT_ROOT / "questions.json")
UNSUPPORTED = [q for q in QUESTIONS if normalize_expected_behavior(q.get("expected_behavior")) == "refuse"]
ANSWERABLE = [q for q in QUESTIONS if normalize_expected_behavior(q.get("expected_behavior")) == "answer"]


def assert_refused(record):
    assert record["supported"] is False
    assert record["answer"].startswith(REFUSAL_PREFIX)
    assert record["citations"] == [] and record["chunk_citations"] == []
    assert record["refusal_reason"]
    assert record["validation"]["passed"], record["validation"]["checks"]


@pytest.mark.parametrize("q", UNSUPPORTED, ids=[q["question_id"] for q in UNSUPPORTED])
def test_labelled_unsupported_questions_are_refused(assistant, q):
    assert_refused(assistant.answer(q["question"]))


@pytest.mark.parametrize(
    "question",
    [
        "What is the capital of France?",
        "Who won the football world cup in 2018?",
        "What is it?",
        "Can I pay my withdrawal fees with gift cards?",
    ],
)
def test_out_of_domain_questions_are_refused(assistant, question):
    assert_refused(assistant.answer(question))


def test_partially_supported_question_with_known_terms_is_refused(assistant):
    # Every word exists in the corpus, but no single passage covers withdrawals *and* API keys.
    record = assistant.answer("What is the withdrawal limit for API keys?")
    assert_refused(record)
    assert record["refusal_reason"] == "insufficient_coverage"


@pytest.mark.parametrize("q", ANSWERABLE, ids=[q["question_id"] for q in ANSWERABLE])
def test_answerable_questions_are_supported_with_retrieved_citations(assistant, q):
    record = assistant.answer(q["question"])
    assert record["supported"] is True
    assert record["citations"] and record["chunk_citations"]
    assert set(record["chunk_citations"]) <= set(record["context_source_ids"]) <= set(record["retrieved_source_ids"])
    assert set(record["citations"]) <= set(record["retrieved_doc_ids"])
    assert set(record["citations"]) & set(q["expected_sources"])
    assert record["validation"]["passed"], record["validation"]["checks"]


def test_example_question_returns_documented_limit(assistant):
    record = assistant.answer("How many password reset attempts are allowed per hour?")
    assert record["supported"]
    assert "5 password reset attempts per hour" in record["answer"]
    assert record["citations"] == ["authentication_password_reset.md"]
    assert record["chunk_citations"][0].startswith("authentication_password_reset#")


def test_answer_sentences_are_verbatim_from_cited_chunks(assistant):
    record = assistant.answer("How long does identity verification review usually take?")
    for item in record["evidence"]:
        assert item["text"] in assistant.corpus[item["chunk_id"]].text
