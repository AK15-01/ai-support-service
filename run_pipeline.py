"""Batch run: answer every question in questions.json and write the three JSON artefacts.

    python run_pipeline.py [--docs docs] [--questions questions.json] [--output-dir .]
"""

from __future__ import annotations

import argparse
import logging
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag.config import Settings
from rag.generation import GenerationError
from rag.ingestion import IngestionError
from rag.logging_utils import configure_logging, get_logger, log_event
from rag.pipeline import QuestionFileError, SupportAssistant, load_questions
from rag.validation import build_validation_report

logger = get_logger("run_pipeline")

RETRIEVAL_FILE = "retrieval_results.json"
ANSWERS_FILE = "answers.json"
REPORT_FILE = "validation_report.json"

# `citations` are document filenames; `chunk_citations` keep the chunk-level provenance used by validation.
ANSWER_FIELDS = ("question_id", "question", "answer", "citations", "supported", "chunk_citations", "refusal_reason",
                 "retrieved_source_ids", "retrieved_doc_ids", "context_source_ids", "evidence", "diagnostics")


def parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    defaults = Settings()
    parser = argparse.ArgumentParser(description="Run the RAG support assistant over a question set.")
    parser.add_argument("--docs", type=Path, default=defaults.docs_dir, help="directory of .md/.txt documents")
    parser.add_argument("--questions", type=Path, default=defaults.questions_path, help="questions JSON file")
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir, help="where to write JSON artefacts")
    parser.add_argument("--top-k", type=int, default=defaults.top_k)
    parser.add_argument("--generator", default=defaults.generator, choices=["extractive", "anthropic"])
    parser.add_argument("--strict", action="store_true", help="also fail when outcomes disagree with labels")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(settings: Settings) -> Dict[str, Any]:
    questions = load_questions(settings.questions_path)
    assistant = SupportAssistant(settings)

    records: List[Dict[str, Any]] = []
    for q in questions:
        record = assistant.answer(q["question"])
        # Labels are attached only after answering, for the evaluation section of the report.
        record = {"question_id": q["question_id"], **record,
                  "expected_behavior": q.get("expected_behavior"), "expected_sources": q.get("expected_sources")}
        records.append(record)

    retrieval_payload = {
        "corpus": {
            "documents": len({c.source for c in assistant.chunks}),
            "chunks": len(assistant.chunks),
        },
        "settings": {"top_k": settings.top_k, "min_retrieval_score": settings.min_retrieval_score},
        "results": [
            {
                "question_id": r["question_id"],
                "question": r["question"],
                "retrieved_chunks": r["retrieved"],  # each: doc_id, chunk_id, score, text, rank, section, ...
                "expected_sources": r["expected_sources"],
                "expected_source_hit": (any(s in r["expected_sources"] for s in r["retrieved_doc_ids"])
                                        if r["expected_sources"] else None),
            }
            for r in records
        ],
    }
    answers_payload = {
        "generator": assistant.generator.name,
        "answers": [
            {key: r[key] for key in ANSWER_FIELDS}
            for r in records
        ],
    }
    report = build_validation_report(records, settings.to_dict())

    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / RETRIEVAL_FILE, retrieval_payload)
    write_json(output_dir / ANSWERS_FILE, answers_payload)
    write_json(output_dir / REPORT_FILE, report)
    return {"records": records, "report": report, "output_dir": output_dir}


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)
    settings = Settings(
        docs_dir=args.docs,
        questions_path=args.questions,
        output_dir=args.output_dir,
        top_k=args.top_k,
        generator=args.generator,
    )
    started = time.perf_counter()
    try:
        outcome = run(settings)
    except (IngestionError, QuestionFileError, GenerationError, ValueError) as exc:
        log_event(logger, "pipeline.failed", level=logging.ERROR, error=str(exc))
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = outcome["report"]
    summary, evaluation = report["summary"], report["evaluation"]
    for rec in outcome["records"]:
        status = "ANSWERED" if rec["supported"] else f"REFUSED ({rec['refusal_reason']})"
        check = "valid" if rec["validation"]["passed"] else "INVALID"
        print(f"{rec['question_id']:<6} {status:<36} {check:<8} {rec['question']}")
    print(
        f"\n{summary['questions']} questions: {summary['answered']} answered, {summary['refused']} refused, "
        f"{summary['validation_passed']} passed validation."
    )
    if evaluation["labelled_questions"]:
        line = f"Label agreement: {evaluation['behavior_accuracy']:.0%}"
        if evaluation["retrieval_topk_hit_rate"] is not None:
            line += f"  |  retrieval hit@{settings.top_k}: {evaluation['retrieval_topk_hit_rate']:.0%}"
        print(line)
    if evaluation["unrecognised_labels"]:
        print(f"Unrecognised expected_behavior labels (ignored): {evaluation['unrecognised_labels']}")
    print(f"Wrote {RETRIEVAL_FILE}, {ANSWERS_FILE}, {REPORT_FILE} to {outcome['output_dir'].resolve()}")
    log_event(logger, "pipeline.complete", duration_ms=round((time.perf_counter() - started) * 1000, 2), **summary)

    if not summary["all_passed"]:
        return 1
    if args.strict and evaluation["behavior_mismatches"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
