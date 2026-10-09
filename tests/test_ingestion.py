import hashlib
import re

import pytest

from rag.ingestion import IngestionError, ingest_directory
from rag.text import slugify

CHUNK_ID_RE = re.compile(r"^[a-z0-9_]+#[a-z0-9-]+-\d+$")


def test_every_chunk_has_complete_metadata(assistant, settings):
    doc_files = {p.name for p in settings.docs_dir.iterdir() if p.suffix == ".md"}
    for chunk in assistant.chunks:
        assert CHUNK_ID_RE.match(chunk.chunk_id), chunk.chunk_id
        assert chunk.source in doc_files
        assert chunk.chunk_id.startswith(slugify(chunk.source.rsplit(".", 1)[0], sep="_") + "#")
        assert chunk.section.startswith(chunk.doc_title)
        assert chunk.text.strip()
        assert 0 < chunk.word_count <= settings.max_chunk_words
        assert chunk.content_hash == hashlib.sha1(chunk.text.encode("utf-8")).hexdigest()[:12]


def test_every_document_is_chunked(assistant, settings):
    doc_files = {p.name for p in settings.docs_dir.iterdir() if p.suffix == ".md"}
    assert {c.source for c in assistant.chunks} == doc_files
    assert len(doc_files) >= 5


def test_chunk_ids_are_unique_and_stable_across_runs(settings):
    first = ingest_directory(settings.docs_dir, settings.max_chunk_words)
    second = ingest_directory(settings.docs_dir, settings.max_chunk_words)
    ids = [c.chunk_id for c in first]
    assert len(ids) == len(set(ids))
    assert [(c.chunk_id, c.content_hash) for c in first] == [(c.chunk_id, c.content_hash) for c in second]


def test_editing_one_section_does_not_renumber_others(make_docs):
    original = "# Guide\n\n## Alpha\n\nAlpha text here.\n\n## Beta\n\nBeta text here.\n"
    docs = make_docs({"guide.md": original})
    before = {c.chunk_id: c.text for c in ingest_directory(docs)}
    (docs / "guide.md").write_text(original.replace("# Guide\n", "# Guide\n\n## New intro\n\nInserted.\n"),
                                   encoding="utf-8")
    after = {c.chunk_id: c.text for c in ingest_directory(docs)}
    assert before["guide#alpha-0"] == after["guide#alpha-0"]
    assert before["guide#beta-0"] == after["guide#beta-0"]
    assert "guide#new-intro-0" in after


def test_tables_become_self_describing_rows(make_docs):
    docs = make_docs({"plans.md": "# Plans\n\n## Limits\n\n| Plan | Limit |\n|---|---|\n| Free | 10 |\n| Pro | 99 |\n"})
    (chunk,) = ingest_directory(docs)
    assert chunk.text.splitlines() == ["Plan: Free; Limit: 10.", "Plan: Pro; Limit: 99."]


def test_long_sections_split_without_crossing_sections(make_docs):
    sentences = " ".join(f"Sentence number {i} talks about refunds." for i in range(60))
    docs = make_docs({"long.md": f"# Long\n\n## Refunds\n\n{sentences}\n\n## Other\n\nShort section.\n"})
    chunks = ingest_directory(docs, max_words=50)
    refund_chunks = [c for c in chunks if c.section == "Long > Refunds"]
    assert len(refund_chunks) > 1
    assert [c.chunk_id for c in refund_chunks] == [f"long#refunds-{i}" for i in range(len(refund_chunks))]
    assert all(c.word_count <= 50 for c in chunks)
    assert all("Short section" not in c.text for c in refund_chunks)


def test_headings_inside_code_fences_are_ignored(make_docs):
    docs = make_docs({"code.md": "# Code\n\n## Usage\n\n```\n# not a heading\nrun --fast\n```\n"})
    chunks = ingest_directory(docs)
    assert [c.section for c in chunks] == ["Code > Usage"]


def test_missing_or_empty_docs_directory_raises(tmp_path):
    with pytest.raises(IngestionError):
        ingest_directory(tmp_path / "does-not-exist")
    (tmp_path / "empty").mkdir()
    with pytest.raises(IngestionError):
        ingest_directory(tmp_path / "empty")
