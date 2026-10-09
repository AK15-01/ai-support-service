"""Load local documents, normalise Markdown, and split into section-aware chunks with stable IDs."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from .logging_utils import get_logger, log_event
from .models import Chunk
from .text import slugify, split_sentences

logger = get_logger("ingestion")

SUPPORTED_SUFFIXES = (".md", ".markdown", ".txt")

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^(```|~~~)")
_LIST_ITEM_RE = re.compile(r"^(?:[-*+]|\d+[.)])\s+")
_TABLE_SEPARATOR_RE = re.compile(r"^\|?[\s:|-]+\|?$")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


class IngestionError(RuntimeError):
    pass


@dataclass
class Section:
    headings: List[str]  # heading path below the document title
    lines: List[str] = field(default_factory=list)  # logical lines: paragraphs, list items, table rows


@dataclass
class Document:
    source: str
    doc_id: str
    title: str
    sections: List[Section]


def _clean_inline(text: str) -> str:
    text = _LINK_RE.sub(r"\1 (\2)", text)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    return re.sub(r"\s+", " ", text).strip()


def _table_to_lines(rows: List[str]) -> List[str]:
    """Turn a Markdown table into one self-describing line per data row."""
    cells = [[_clean_inline(c) for c in row.strip().strip("|").split("|")] for row in rows]
    cells = [row for row, raw in zip(cells, rows) if not _TABLE_SEPARATOR_RE.match(raw.strip())]
    if len(cells) < 2:
        return ["; ".join(row) for row in cells if any(row)]
    header, body = cells[0], cells[1:]
    lines = []
    for row in body:
        pairs = [f"{h}: {v}" if h else v for h, v in zip(header, row) if v]
        if pairs:
            lines.append("; ".join(pairs) + ".")
    return lines


def _block_to_lines(block: List[str]) -> List[str]:
    """Join wrapped paragraph lines, keep list items and table rows as separate lines."""
    out: List[str] = []
    table: List[str] = []
    can_continue = False
    for raw in block:
        stripped = raw.strip()
        if stripped.startswith("|"):
            table.append(stripped)
            can_continue = False
            continue
        if table:
            out.extend(_table_to_lines(table))
            table = []
        if _LIST_ITEM_RE.match(stripped):
            out.append("- " + _clean_inline(_LIST_ITEM_RE.sub("", stripped)))
            can_continue = True
        elif can_continue and out:
            out[-1] = f"{out[-1]} {_clean_inline(stripped)}"
        else:
            out.append(_clean_inline(stripped))
            can_continue = True
    if table:
        out.extend(_table_to_lines(table))
    return [line for line in out if line.strip("- ")]


def parse_markdown(text: str, source: str, doc_id: str) -> Document:
    title = ""
    sections: List[Section] = [Section(headings=[])]
    heading_stack: List[str] = []
    block: List[str] = []
    in_code = False

    def flush_block() -> None:
        if block:
            sections[-1].lines.extend(_block_to_lines(block))
            block.clear()

    for raw in text.splitlines():
        if _FENCE_RE.match(raw.strip()):
            in_code = not in_code
            block.append(raw)
            continue
        match = None if in_code else _HEADING_RE.match(raw)
        if match:
            flush_block()
            level, heading = len(match.group(1)), _clean_inline(match.group(2))
            if level == 1 and not title:
                title = heading
                continue
            depth = max(level - 1, 1)
            heading_stack = heading_stack[: depth - 1] + [heading]
            sections.append(Section(headings=list(heading_stack)))
        elif raw.strip():
            block.append(raw)
        else:
            flush_block()
    flush_block()

    title = title or Path(source).stem.replace("_", " ").replace("-", " ").title()
    return Document(source=source, doc_id=doc_id, title=title, sections=[s for s in sections if s.lines])


def _split_long_line(line: str, max_words: int) -> List[str]:
    if len(line.split()) <= max_words:
        return [line]
    pieces, current = [], []
    for sentence in split_sentences(line):
        if current and len(" ".join(current + [sentence]).split()) > max_words:
            pieces.append(" ".join(current))
            current = []
        current.append(sentence)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _pack_lines(lines: List[str], max_words: int) -> List[str]:
    chunks, current, current_words = [], [], 0
    for line in lines:
        for piece in _split_long_line(line, max_words):
            words = len(piece.split())
            if current and current_words + words > max_words:
                chunks.append("\n".join(current))
                current, current_words = [], 0
            current.append(piece)
            current_words += words
    if current:
        chunks.append("\n".join(current))
    return chunks


def chunk_document(doc: Document, max_words: int = 120) -> List[Chunk]:
    chunks: List[Chunk] = []
    used_slugs: Dict[str, int] = {}
    for section in doc.sections:
        base_slug = slugify(section.headings[-1]) if section.headings else "overview"
        used_slugs[base_slug] = used_slugs.get(base_slug, 0) + 1
        slug = base_slug if used_slugs[base_slug] == 1 else f"{base_slug}-{used_slugs[base_slug]}"
        section_path = " > ".join([doc.title] + section.headings)
        for n, text in enumerate(_pack_lines(section.lines, max_words)):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{slug}-{n}",
                    source=doc.source,
                    doc_title=doc.title,
                    section=section_path,
                    chunk_index=len(chunks),
                    text=text,
                    word_count=len(text.split()),
                    content_hash=hashlib.sha1(text.encode("utf-8")).hexdigest()[:12],
                )
            )
    return chunks


def discover_files(docs_dir: Path) -> List[Path]:
    if not docs_dir.is_dir():
        raise IngestionError(f"Docs directory not found: {docs_dir}")
    return sorted(
        (p for p in docs_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES),
        key=lambda p: p.name,
    )


def load_documents(docs_dir: Path) -> List[Document]:
    documents: List[Document] = []
    used_ids: Dict[str, str] = {}
    for path in discover_files(docs_dir):
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError) as exc:
            log_event(logger, "ingestion.skip_file", level=logging.WARNING, source=path.name, error=str(exc))
            continue
        doc_id = slugify(path.stem, sep="_")
        if doc_id in used_ids:  # e.g. faq.md and faq.txt
            doc_id = slugify(path.name, sep="_")
        used_ids[doc_id] = path.name
        documents.append(parse_markdown(text, source=path.name, doc_id=doc_id))
    return documents


def ingest_directory(docs_dir: Path, max_words: int = 120) -> List[Chunk]:
    started = time.perf_counter()
    documents = load_documents(Path(docs_dir))
    chunks = [chunk for doc in documents for chunk in chunk_document(doc, max_words)]
    if not chunks:
        raise IngestionError(f"No non-empty documents ({', '.join(SUPPORTED_SUFFIXES)}) found in {docs_dir}")
    ids = [c.chunk_id for c in chunks]
    if len(ids) != len(set(ids)):
        raise IngestionError("Duplicate chunk IDs produced; check for duplicate document names")
    per_doc: Dict[str, int] = {}
    for chunk in chunks:
        per_doc[chunk.source] = per_doc.get(chunk.source, 0) + 1
    log_event(
        logger,
        "ingestion.complete",
        docs_dir=str(docs_dir),
        documents=len(documents),
        chunks=len(chunks),
        chunks_per_document=per_doc,
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
    )
    return chunks


def chunk_lookup(chunks: List[Chunk]) -> Dict[str, Chunk]:
    return {c.chunk_id: c for c in chunks}

