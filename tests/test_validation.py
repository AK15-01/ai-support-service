import pytest

from rag.models import REFUSAL_PREFIX, Chunk, GenerationResult
from rag.validation import validate_answer


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, source=chunk_id.split("#")[0] + ".md", doc_title="Doc", section="Doc > Section",
                 chunk_index=0, text=text, word_count=len(text.split()), content_hash="x")


CORPUS = {
    "fees#fees-0": make_chunk("fees#fees-0", "The first withdrawal each month is free; later ones cost 2.50 USD."),
    "limits#daily-0": make_chunk("limits#daily-0", "Verified accounts can withdraw up to 50,000 USD per day."),
}
RETRIEVED = ["fees#fees-0"]


def checks(report):
    return {c["name"]: c["passed"] for c in report["checks"]}


def supported(answer, citations):
    return GenerationResult(answer=answer, citations=citations, supported=True)


def test_valid_cited_answer_passes():
    result = supported("The first withdrawal each month is free. [fees#fees-0]", ["fees#fees-0"])
    report = validate_answer(result, RETRIEVED, CORPUS)
    assert report["passed"], report["checks"]


def test_citation_outside_retrieved_set_fails():
    # The chunk exists in the corpus but was not retrieved for this question.
    result = supported("Verified accounts can withdraw up to 50,000 USD per day. [limits#daily-0]", ["limits#daily-0"])
    report = validate_answer(result, RETRIEVED, CORPUS)
    assert checks(report)["citations_known"]
    assert not checks(report)["citations_in_retrieved"]
    assert not report["passed"]


def test_fabricated_citation_fails():
    result = supported("Withdrawals are free. [fees#made-up-0]", ["fees#made-up-0"])
    report = validate_answer(result, RETRIEVED, CORPUS)
    assert not checks(report)["citations_known"]
    assert not checks(report)["citations_in_retrieved"]


def test_supported_answer_without_citations_fails():
    report = validate_answer(supported("The first withdrawal each month is free.", []), RETRIEVED, CORPUS)
    assert not checks(report)["citations_present"]
    assert not checks(report)["inline_citations_consistent"]
    assert not report["passed"]


@pytest.mark.parametrize("answer", ["", "   "])
def test_empty_answer_fails(answer):
    report = validate_answer(supported(answer, ["fees#fees-0"]), RETRIEVED, CORPUS)
    assert not checks(report)["answer_nonempty"]


def test_inline_markers_must_match_citation_list():
    result = supported("The first withdrawal each month is free. [fees#fees-0]", ["fees#fees-0", "limits#daily-0"])
    assert not checks(validate_answer(result, RETRIEVED + ["limits#daily-0"], CORPUS))["inline_citations_consistent"]


def test_proper_refusal_passes():
    result = GenerationResult(answer=f"{REFUSAL_PREFIX} The documents do not mention: crypto.", citations=[],
                              supported=False, refusal_reason="unknown_terms")
    assert validate_answer(result, RETRIEVED, CORPUS)["passed"]


def test_refusal_with_citations_fails():
    result = GenerationResult(answer=REFUSAL_PREFIX, citations=["fees#fees-0"], supported=False)
    assert not checks(validate_answer(result, RETRIEVED, CORPUS))["refusal_marked"]


def test_unsupported_answer_must_be_explicitly_marked():
    result = GenerationResult(answer="Probably 3 days.", citations=[], supported=False)
    assert not checks(validate_answer(result, RETRIEVED, CORPUS))["refusal_marked"]


def test_supported_answer_phrased_as_refusal_fails():
    result = GenerationResult(answer=f"{REFUSAL_PREFIX} [fees#fees-0]", citations=["fees#fees-0"], supported=True)
    assert not checks(validate_answer(result, RETRIEVED, CORPUS))["refusal_marked"]


def test_valid_citation_does_not_imply_support():
    """Citation provenance passes, yet the sentence is not grounded in the cited text."""
    result = supported("Bitcoin withdrawals settle instantly through the lightning network. [fees#fees-0]",
                       ["fees#fees-0"])
    report = validate_answer(result, RETRIEVED, CORPUS)
    c = checks(report)
    assert c["citations_known"] and c["citations_in_retrieved"] and c["inline_citations_consistent"]
    assert not c["lexical_grounding"]
    assert not report["passed"]
    assert report["semantic_support"] == "not_verified"


def test_uncited_trailing_claim_is_flagged():
    result = supported("The first withdrawal each month is free. [fees#fees-0] Refunds take 90 days.", ["fees#fees-0"])
    report = validate_answer(result, RETRIEVED, CORPUS)
    assert not checks(report)["lexical_grounding"]
    assert report["grounding"][-1]["cited"] == []
