"""Central configuration. Every tunable threshold lives here, not in module code."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    docs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "docs")
    questions_path: Path = field(default_factory=lambda: PROJECT_ROOT / "questions.json")
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT)

    # Ingestion
    max_chunk_words: int = 120

    # Retrieval
    top_k: int = 4
    min_retrieval_score: float = 0.08

    # Prompting
    max_context_chars: int = 6000

    # Extractive generation / refusal
    generator: str = "extractive"
    min_sentence_score: float = 0.35
    min_coverage: float = 0.9
    max_answer_sentences: int = 2
    relative_sentence_cutoff: float = 0.6
    lexicon_credit: float = 0.8  # a lexicon (synonym) match counts less than a literal match

    # Validation
    min_grounding_overlap: float = 0.6

    def to_dict(self) -> dict:
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, Path):  # project-relative where possible, so artefacts carry no local paths
                try:
                    data[key] = value.resolve().relative_to(PROJECT_ROOT).as_posix() or "."
                except ValueError:
                    data[key] = value.as_posix()
        return data
