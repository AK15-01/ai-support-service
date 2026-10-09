import pytest

from rag.config import PROJECT_ROOT
from rag.ingestion import ingest_directory
from rag.pipeline import load_questions
from rag.retrieval import TfidfRetriever
from rag.validation import normalize_expected_behavior

from .conftest import SYNTHETIC_DOCS

ANSWERABLE = [q for q in load_questions(PROJECT_ROOT / "questions.json")
              if normalize_expected_behavior(q.get("expected_behavior")) == "answer"]


@pytest.mark.parametrize("q", ANSWERABLE, ids=[q["question_id"] for q in ANSWERABLE])
def test_answerable_questions_rank_expected_source_first(assistant, q):
    results = assistant.retriever.search(q["question"], top_k=4)
    assert results, "no results"
    assert results[0].chunk.source in q["expected_sources"]
    assert results[0].above_threshold


def test_results_are_sorted_bounded_and_ranked(assistant):
    results = assistant.retriever.search("withdrawal fees and processing times", top_k=3)
    assert len(results) == 3
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert [r.rank for r in results] == [1, 2, 3]


def test_retrieval_is_deterministic(assistant):
    query = "how do I escalate an incident"
    first = [(r.chunk.chunk_id, r.score) for r in assistant.retriever.search(query, top_k=4)]
    second = [(r.chunk.chunk_id, r.score) for r in assistant.retriever.search(query, top_k=4)]
    assert first == second


def test_stopword_only_query_returns_nothing(assistant):
    assert assistant.retriever.search("what is it and how?", top_k=4) == []


def test_unrelated_query_does_not_clear_threshold(assistant):
    results = assistant.retriever.search("What is the capital of France?", top_k=4)
    assert not any(r.above_threshold for r in results)


def test_query_inflections_reach_the_index(make_docs):
    """TF-IDF matches tokens exactly; "refund" must still retrieve a passage that only says "Refunds"."""
    docs = make_docs({"returns.md": "# Returns\n\nRefunds are issued within 7 business days.\n",
                      "shipping.md": "# Shipping\n\nExpress shipping costs 9.90 EUR.\n"})
    retriever = TfidfRetriever(ingest_directory(docs))
    assert "refunds" in retriever.expand_query("When will I get my refund?")
    assert retriever.search("When will I get my refund?", top_k=1)[0].chunk.source == "returns.md"


def test_works_on_a_replacement_corpus(make_docs):
    retriever = TfidfRetriever(ingest_directory(make_docs(SYNTHETIC_DOCS)))
    top = retriever.search("How often should I change the engine oil?", top_k=2)[0]
    assert top.chunk.source == "car_maintenance.md"
    assert top.chunk.chunk_id == "car_maintenance#engine-oil-0"
    top = retriever.search("when to prune roses", top_k=2)[0]
    assert top.chunk.chunk_id == "garden_care#pruning-roses-0"
