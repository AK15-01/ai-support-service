"""Orchestration shared by app.py and run_pipeline.py: ingest -> retrieve -> prompt -> generate -> validate."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Settings
from .generation import Generator, create_generator
from .ingestion import chunk_lookup, ingest_directory
from .lexicon import TermMatcher
from .logging_utils import get_logger, log_event
from .models import Prompt
from .prompting import build_prompt
from .retrieval import TfidfRetriever
from .validation import validate_answer

logger = get_logger("pipeline")


class QuestionFileError(ValueError):
    pass


def load_questions(path: Path) -> List[Dict[str, Any]]:
    """Accept a list of strings, a list of {"question": ...} objects, or {"questions": [...]}.
    Each question gets a `question_id` (taken from "question_id" or "id", else generated)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise QuestionFileError(f"Questions file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise QuestionFileError(f"Questions file is not valid JSON: {exc}") from exc
    if isinstance(data, dict):
        data = data.get("questions")
    if not isinstance(data, list) or not data:
        raise QuestionFileError("Questions file must contain a non-empty list of questions")

    questions: List[Dict[str, Any]] = []
    for n, item in enumerate(data, start=1):
        entry = {"question": item} if isinstance(item, str) else dict(item) if isinstance(item, dict) else None
        if not entry or not isinstance(entry.get("question"), str) or not entry["question"].strip():
            raise QuestionFileError(f"Question #{n} has no 'question' text")
        entry["question_id"] = str(entry.get("question_id") or entry.pop("id", None) or f"q{n:02d}")
        questions.append(entry)
    ids = [q["question_id"] for q in questions]
    if len(ids) != len(set(ids)):
        raise QuestionFileError("Question IDs must be unique")
    return questions


def unique(values: List[str]) -> List[str]:
    return list(dict.fromkeys(values))


class SupportAssistant:
    def __init__(self, settings: Optional[Settings] = None, generator: Optional[Generator] = None,
                 matcher: Optional[TermMatcher] = None):
        self.settings = settings or Settings()
        self.matcher = matcher or TermMatcher(expansion_credit=self.settings.lexicon_credit)
        self.chunks = ingest_directory(self.settings.docs_dir, self.settings.max_chunk_words)
        self.corpus = chunk_lookup(self.chunks)
        self.retriever = TfidfRetriever(self.chunks, min_score=self.settings.min_retrieval_score, matcher=self.matcher)
        self.generator = generator or create_generator(
            self.settings.generator, self.retriever.unigram_idf, self.settings, self.matcher
        )

    def build_prompt(self, question: str) -> Prompt:
        retrieved = self.retriever.search(question, self.settings.top_k)
        return build_prompt(question, retrieved, self.settings.max_context_chars)

    def answer(self, question: str) -> Dict[str, Any]:
        """Answer one question. Only the question text is used; labels never reach this method."""
        started = time.perf_counter()
        retrieved = self.retriever.search(question, self.settings.top_k)
        prompt = build_prompt(question, retrieved, self.settings.max_context_chars)
        result = self.generator.generate(prompt)

        # Public citations are document filenames; chunk-level citations are kept for validation and audit.
        doc_citations = unique([self.corpus[c].source for c in result.citations if c in self.corpus])
        validation = validate_answer(
            result, prompt.allowed_citations, self.corpus, self.settings.min_grounding_overlap,
            document_citations=doc_citations,
        )
        record = {
            "question": question,
            "answer": result.answer,
            "citations": doc_citations,
            "supported": result.supported,
            "chunk_citations": result.citations,
            "refusal_reason": result.refusal_reason,
            "retrieved_source_ids": [item.chunk.chunk_id for item in retrieved],
            "retrieved_doc_ids": unique([item.chunk.source for item in retrieved]),
            "context_source_ids": prompt.allowed_citations,
            "retrieved": [item.to_dict() for item in retrieved],
            "evidence": result.evidence,
            "diagnostics": result.diagnostics,
            "generator": result.generator,
            "validation": validation,
        }
        log_event(
            logger,
            "pipeline.answered",
            question=question,
            supported=result.supported,
            refusal_reason=result.refusal_reason,
            citations=doc_citations,
            chunk_citations=result.citations,
            validation_passed=validation["passed"],
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return record
