"""Regression tests for the review findings: output schema, label vocabularies, paraphrases, and the
negation / numeric / identifier distinctions that must survive the paraphrase support."""

import json

import pytest

import run_pipeline
from rag.lexicon import TermMatcher, identifier_pattern, is_exact_term, of_phrases
from rag.models import GenerationResult
from rag.pipeline import SupportAssistant
from rag.validation import normalize_expected_behavior, validate_answer


@pytest.fixture(scope="module")
def artefacts(tmp_path_factory):
    out = tmp_path_factory.mktemp("artefacts")
    assert run_pipeline.main(["--output-dir", str(out), "--log-level", "WARNING"]) == 0
    return {name: json.loads((out / f"{name}.json").read_text(encoding="utf-8"))
            for name in ("retrieval_results", "answers", "validation_report")}


# --- Priority 1: required output schema -------------------------------------------------------------

def test_retrieval_results_schema(artefacts, assistant):
    results = artefacts["retrieval_results"]["results"]
    assert results
    for row in results:
        assert row["question_id"] and isinstance(row["retrieved_chunks"], list)
        for chunk in row["retrieved_chunks"]:
            assert {"doc_id", "chunk_id", "score", "text"} <= set(chunk)
            assert chunk["doc_id"] == assistant.corpus[chunk["chunk_id"]].source  # source filename
            assert chunk["doc_id"].endswith(".md")
            assert chunk["text"] == assistant.corpus[chunk["chunk_id"]].text
            assert isinstance(chunk["score"], float)


def test_answers_schema_uses_document_citations(artefacts, assistant):
    answers = artefacts["answers"]["answers"]
    for row in answers:
        assert {"question_id", "answer", "citations", "supported"} <= set(row)
        assert isinstance(row["supported"], bool) and row["answer"].strip()
        derived = list(dict.fromkeys(assistant.corpus[c].source for c in row["chunk_citations"]))
        assert row["citations"] == derived
        assert all(c.endswith(".md") for c in row["citations"])
        assert set(row["chunk_citations"]) <= set(row["retrieved_source_ids"])
    assert any(row["supported"] for row in answers) and any(not row["supported"] for row in answers)


def test_validation_report_includes_document_citation_check(artefacts):
    for row in artefacts["validation_report"]["per_question"]:
        assert "question_id" in row
        assert "document_citations_consistent" in {c["name"] for c in row["checks"]}


def test_wrong_document_citation_fails_validation(assistant):
    chunk_id = "payments_withdrawals#fees-0"
    result = GenerationResult(answer=f"{assistant.corpus[chunk_id].text.splitlines()[0]} [{chunk_id}]",
                              citations=[chunk_id], supported=True)
    good = validate_answer(result, [chunk_id], assistant.corpus, document_citations=["payments_withdrawals.md"])
    bad = validate_answer(result, [chunk_id], assistant.corpus, document_citations=["api_rate_limits.md"])
    assert good["passed"]
    assert not {c["name"]: c["passed"] for c in bad["checks"]}["document_citations_consistent"]


# --- Priority 2: label vocabularies ------------------------------------------------------------------

@pytest.mark.parametrize("label, expected", [
    ("answerable", "answer"), ("answer", "answer"), ("Answerable", "answer"),
    ("unanswerable", "refuse"), ("refuse", "refuse"), ("unsupported", "refuse"),
    ("partially supported", "refuse"), ("maybe", None), (None, None),
])
def test_expected_behavior_vocabularies(label, expected):
    assert normalize_expected_behavior(label) == expected


def test_mixed_label_vocabularies_evaluate_correctly(tmp_path):
    questions = [
        {"question_id": "a", "question": "How many password reset attempts are allowed per hour?",
         "expected_behavior": "answerable"},
        {"question_id": "b", "question": "How many password reset attempts are allowed per hour?",
         "expected_behavior": "answer"},
        {"question_id": "c", "question": "What are the opening hours of the London office?",
         "expected_behavior": "unanswerable"},
        {"id": "d", "question": "What are the opening hours of the London office?", "expected_behavior": "refuse"},
        {"question_id": "e", "question": "What are the opening hours of the London office?",
         "expected_behavior": "sometimes"},
    ]
    path = tmp_path / "mixed.json"
    path.write_text(json.dumps(questions), encoding="utf-8")
    assert run_pipeline.main(["--questions", str(path), "--output-dir", str(tmp_path), "--log-level", "WARNING"]) == 0
    evaluation = json.loads((tmp_path / "validation_report.json").read_text(encoding="utf-8"))["evaluation"]
    assert evaluation["labelled_questions"] == 4
    assert evaluation["behavior_accuracy"] == 1.0
    assert evaluation["unrecognised_labels"] == [{"question_id": "e", "label": "sometimes"}]


# --- Priority 3: paraphrases, negation, numbers, identifiers ----------------------------------------

@pytest.mark.parametrize("question, must_contain, lexicon", [
    ("Can I submit a mobile phone bill as proof of address?", "not accepted as proof of address", "submit"),
    ("Is two-factor authentication optional for API accounts?", "mandatory for accounts with API access", "optional"),
    ("Do I pay for my second withdrawal in a month?", "each additional withdrawal in the same month costs 2.50 USD",
     "second"),
    ("How long are verification documents kept after an account is closed?", "retained for 5 years", "kept"),
    ("What happens to the money if a withdrawal fails?", "funds are returned to the account balance", "money"),
])
def test_paraphrases_answer_with_verbatim_evidence(assistant, question, must_contain, lexicon):
    record = assistant.answer(question)
    assert record["supported"], record["answer"]
    assert must_contain in record["answer"]  # verbatim, so negations and figures are preserved
    assert lexicon in record["diagnostics"]["lexicon_matches"]
    assert record["validation"]["passed"]


@pytest.mark.parametrize("question, expected, excluded", [
    ("How often is the status page updated during a SEV2 incident?", "During a SEV2 incident it is updated every 2 hours",
     "SEV1"),
    ("How often is the status page updated during a SEV1 incident?", "updated every 30 minutes", "SEV2"),
    ("When is a SEV2 incident escalated to the Engineering Manager?", "after 4 hours", "SEV1"),
])
def test_identifier_questions_exclude_sibling_identifiers(assistant, question, expected, excluded):
    record = assistant.answer(question)
    assert record["supported"]
    assert expected in record["answer"]
    assert excluded not in record["answer"]


def test_sibling_specifier_is_not_used_as_evidence(assistant):
    record = assistant.answer("Are screenshots accepted as proof of identity?")
    assert "proof of address" not in record["answer"]


@pytest.mark.parametrize("question, reason", [
    ("How often is the status page updated during a SEV3 incident?", None),   # only SEV1/SEV2 documented
    ("Is the minimum withdrawal 20 USD?", "missing_exact_terms"),            # docs say 10 USD
    ("What happens after 5 rejected submissions?", None),                   # docs say 3
    ("Is there a refund for the withdrawal fee?", "unknown_terms"),
    ("Is 2FA optional for Basic plan accounts?", None),                      # 2FA and plans are separate docs
    ("Can I submit a selfie as proof of address?", None),
    ("Is a passport optional for withdrawals?", None),
    ("What is the fee for a third withdrawal in a week?", "unknown_terms"),
])
def test_paraphrase_support_does_not_open_false_answers(assistant, question, reason):
    record = assistant.answer(question)
    assert record["supported"] is False, record["answer"]
    assert record["citations"] == []
    if reason:
        assert record["refusal_reason"] == reason


def test_lexicon_is_replaceable(settings):
    plain = SupportAssistant(settings, matcher=TermMatcher(expansions={}))
    record = plain.answer("How long are verification documents kept after an account is closed?")
    assert record["supported"] is False and record["refusal_reason"] == "unknown_terms"


def test_lexicon_match_counts_less_than_literal_match():
    matcher = TermMatcher()
    assert matcher.strength("kept", {"kept"}) == 1.0
    assert matcher.strength("kept", {"retained"}) == matcher.expansion_credit < 1.0
    assert matcher.strength("kept", {"deleted"}) == 0.0


def test_exact_terms_and_conflicts():
    matcher = TermMatcher()
    assert is_exact_term("sev2") and is_exact_term("2.50") and not is_exact_term("hours")
    assert identifier_pattern("sev2") == identifier_pattern("sev1") and identifier_pattern("30") is None
    assert matcher.strength("sev2", {"sev1"}) == 0.0 and matcher.strength("20", {"2.50"}) == 0.0
    assert matcher.conflicting_identifiers(["sev2"], {"sev1", "status"}) == {"sev1"}
    assert matcher.conflicting_identifiers(["sev2"], {"sev1", "sev2"}) == {"sev1"}
    assert of_phrases("Not accepted as proof of address.") == {("proof", "address")}
    assert matcher.conflicting_specifiers({("proof", "identity")}, "accepted as proof of address")
    assert not matcher.conflicting_specifiers({("proof", "identity")}, "Proof of identity: a passport")
