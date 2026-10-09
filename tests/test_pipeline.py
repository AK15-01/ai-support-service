import json
import subprocess
import sys

import app
import run_pipeline
from rag.config import PROJECT_ROOT

from .conftest import SYNTHETIC_DOCS

REQUIRED_APP_KEYS = {"answer", "citations", "supported", "retrieved_source_ids", "chunk_citations"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_pipeline_runs_on_replacement_corpus_and_questions(make_docs, tmp_path):
    docs = make_docs(SYNTHETIC_DOCS)
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps(["How often should I change the engine oil?", "What is the capital of Peru?"]),
                         encoding="utf-8")
    out = tmp_path / "out"
    code = run_pipeline.main(["--docs", str(docs), "--questions", str(questions), "--output-dir", str(out),
                              "--log-level", "WARNING"])
    assert code == 0
    answers = read(out / "answers.json")["answers"]
    assert [a["question_id"] for a in answers] == ["q01", "q02"]
    assert answers[0]["supported"] and answers[0]["citations"] == ["car_maintenance.md"]
    assert answers[0]["chunk_citations"] == ["car_maintenance#engine-oil-0"]
    assert not answers[1]["supported"]
    assert read(out / "retrieval_results.json")["corpus"] == {"documents": 2, "chunks": 4}
    assert read(out / "validation_report.json")["summary"]["all_passed"] is True


def test_artifacts_are_reproducible(tmp_path):
    for name in ("a", "b"):
        assert run_pipeline.main(["--output-dir", str(tmp_path / name), "--log-level", "WARNING"]) == 0
    for artefact in ("retrieval_results.json", "answers.json", "validation_report.json"):
        a = (tmp_path / "a" / artefact).read_text(encoding="utf-8")
        b = (tmp_path / "b" / artefact).read_text(encoding="utf-8")
        assert a.replace((tmp_path / "a").as_posix(), "") == b.replace((tmp_path / "b").as_posix(), "")


def test_labels_never_influence_answers(tmp_path):
    """Flip every expected_behavior label: answers must be identical, only the evaluation changes."""
    original = read(PROJECT_ROOT / "questions.json")
    flip = {"answerable": "unanswerable", "unanswerable": "answerable", "answer": "refuse", "refuse": "answer"}
    flipped = [dict(q, expected_behavior=flip[q["expected_behavior"]]) for q in original]
    flipped_path = tmp_path / "flipped.json"
    flipped_path.write_text(json.dumps(flipped), encoding="utf-8")

    run_pipeline.main(["--output-dir", str(tmp_path / "orig"), "--log-level", "WARNING"])
    run_pipeline.main(["--questions", str(flipped_path), "--output-dir", str(tmp_path / "flip"), "--log-level", "WARNING"])
    assert read(tmp_path / "orig" / "answers.json") == read(tmp_path / "flip" / "answers.json")
    assert read(tmp_path / "orig" / "validation_report.json")["evaluation"]["behavior_accuracy"] == 1.0
    assert read(tmp_path / "flip" / "validation_report.json")["evaluation"]["behavior_accuracy"] == 0.0


def test_app_returns_required_fields(capsys):
    assert app.main(["--question", "How many password reset attempts are allowed per hour?"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert REQUIRED_APP_KEYS <= set(output)
    assert output["supported"] is True
    assert output["citations"] == ["authentication_password_reset.md"]
    assert set(output["chunk_citations"]) <= set(output["retrieved_source_ids"])
    assert set(output["citations"]) <= set(output["retrieved_doc_ids"])


def test_app_runs_as_a_script_with_clean_stdout():
    proc = subprocess.run(
        [sys.executable, "app.py", "--question", "What compensation do customers receive after a SEV1 outage?"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    output = json.loads(proc.stdout)  # stdout must be pure JSON
    assert output["supported"] is False and output["citations"] == []


def test_missing_docs_directory_exits_with_error(tmp_path, capsys):
    assert run_pipeline.main(["--docs", str(tmp_path / "nope"), "--output-dir", str(tmp_path)]) == 2
    assert "Docs directory not found" in capsys.readouterr().err
