from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

import pytest

from rag.config import PROJECT_ROOT, Settings
from rag.logging_utils import configure_logging
from rag.pipeline import SupportAssistant, load_questions

configure_logging("WARNING")


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="session")
def assistant(settings: Settings) -> SupportAssistant:
    return SupportAssistant(settings)


@pytest.fixture(scope="session")
def questions():
    return load_questions(PROJECT_ROOT / "questions.json")


@pytest.fixture
def make_docs(tmp_path: Path) -> Callable[[Dict[str, str]], Path]:
    """Write a throwaway corpus: {"name.md": "markdown"} -> docs directory."""

    def _make(files: Dict[str, str]) -> Path:
        docs = tmp_path / "docs"
        docs.mkdir(exist_ok=True)
        for name, text in files.items():
            (docs / name).write_text(text, encoding="utf-8")
        return docs

    return _make


SYNTHETIC_DOCS = {
    "garden_care.md": (
        "# Garden Care\n\n## Watering tomatoes\n\nWater tomato plants deeply twice a week. "
        "Seedlings need daily misting during the first ten days.\n\n"
        "## Pruning roses\n\nPrune roses in late winter, cutting stems just above an outward facing bud.\n"
    ),
    "car_maintenance.md": (
        "# Car Maintenance\n\n## Engine oil\n\nChange the engine oil every 8,000 kilometres or every 6 months, "
        "whichever comes first.\n\n## Tyre pressure\n\nCheck tyre pressure monthly; the recommended pressure "
        "is printed inside the driver door.\n"
    ),
}
