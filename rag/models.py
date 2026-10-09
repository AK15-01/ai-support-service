"""Data types shared between pipeline stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# Every refusal answer starts with this exact text; validation relies on it.
REFUSAL_PREFIX = "I don't have enough information in the provided documents to answer this question."


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str  # filename relative to the docs directory
    doc_title: str
    section: str  # heading path, e.g. "Payments and Withdrawals > Fees"
    chunk_index: int  # position of the chunk within its document
    text: str
    word_count: int
    content_hash: str

    @property
    def index_text(self) -> str:
        """Text used for retrieval and coverage: headings add topical signal to short chunks."""
        return f"{self.doc_title}\n{self.section}\n{self.text}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int
    above_threshold: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "doc_id": self.chunk.source,  # source filename
            "chunk_id": self.chunk.chunk_id,
            "score": round(self.score, 4),
            "text": self.chunk.text,
            "section": self.chunk.section,
            "above_threshold": self.above_threshold,
        }


@dataclass(frozen=True)
class Prompt:
    question: str
    system: str
    user: str
    context: List[RetrievedChunk]  # chunks actually included, in rank order

    @property
    def allowed_citations(self) -> List[str]:
        return [item.chunk.chunk_id for item in self.context]


@dataclass
class GenerationResult:
    answer: str
    citations: List[str]
    supported: bool
    refusal_reason: Optional[str] = None
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    generator: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
