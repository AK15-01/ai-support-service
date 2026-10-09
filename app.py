"""Single-question CLI. Prints one JSON object to stdout; logs go to stderr.

    python app.py --question "How many password reset attempts are allowed per hour?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from rag.config import Settings
from rag.generation import GenerationError
from rag.ingestion import IngestionError
from rag.logging_utils import configure_logging
from rag.pipeline import SupportAssistant


def parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    defaults = Settings()
    parser = argparse.ArgumentParser(description="Ask the local support assistant a question.")
    parser.add_argument("--question", "-q", required=True)
    parser.add_argument("--docs", type=Path, default=defaults.docs_dir)
    parser.add_argument("--top-k", type=int, default=defaults.top_k)
    parser.add_argument("--generator", default=defaults.generator, choices=["extractive", "anthropic"])
    parser.add_argument("--show-prompt", action="store_true", help="include the constructed prompt in the output")
    parser.add_argument("--log-level", default="WARNING", help="stderr log level (INFO shows pipeline events)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)
    if not args.question.strip():
        print("error: --question must not be empty", file=sys.stderr)
        return 2
    settings = Settings(docs_dir=args.docs, top_k=args.top_k, generator=args.generator)
    try:
        assistant = SupportAssistant(settings)
        record = assistant.answer(args.question)
    except (IngestionError, GenerationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output = {
        "question": record["question"],
        "answer": record["answer"],
        "citations": record["citations"],  # document filenames
        "supported": record["supported"],
        "retrieved_source_ids": record["retrieved_source_ids"],  # chunk IDs, ranked
        "chunk_citations": record["chunk_citations"],
        "retrieved_doc_ids": record["retrieved_doc_ids"],
        "refusal_reason": record["refusal_reason"],
        "retrieved": record["retrieved"],
        "evidence": record["evidence"],
        "validation": {"passed": record["validation"]["passed"], "checks": record["validation"]["checks"],
                       "semantic_support": record["validation"]["semantic_support"]},
    }
    if args.show_prompt:
        prompt = assistant.build_prompt(args.question)
        output["prompt"] = {"system": prompt.system, "user": prompt.user}
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
