"""TF-IDF retrieval over chunks (scikit-learn)."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .logging_utils import get_logger, log_event
from .lexicon import TermMatcher
from .models import Chunk, RetrievedChunk
from .text import content_terms, terms_match

logger = get_logger("retrieval")


class TfidfRetriever:
    def __init__(self, chunks: Sequence[Chunk], min_score: float = 0.08, ngram_range=(1, 2),
                 matcher: Optional[TermMatcher] = None):
        if not chunks:
            raise ValueError("Cannot build a retriever over an empty corpus")
        self.chunks: List[Chunk] = list(chunks)
        self.min_score = min_score
        self.matcher = matcher or TermMatcher()
        self.vectorizer = TfidfVectorizer(
            tokenizer=content_terms,
            token_pattern=None,
            lowercase=False,  # content_terms lowercases
            ngram_range=ngram_range,
            sublinear_tf=True,
            norm="l2",
        )
        try:
            self.matrix = self.vectorizer.fit_transform([c.index_text for c in self.chunks])
        except ValueError as exc:  # every document was stop words only
            raise ValueError(f"Corpus has no indexable terms: {exc}") from exc
        vocab = self.vectorizer.vocabulary_
        self._unigram_idf: Dict[str, float] = {
            term: float(self.vectorizer.idf_[idx]) for term, idx in vocab.items() if " " not in term
        }
        log_event(logger, "retrieval.index_built", chunks=len(self.chunks), vocabulary=len(vocab))

    @property
    def unigram_idf(self) -> Dict[str, float]:
        return self._unigram_idf

    @property
    def chunk_ids(self) -> List[str]:
        return [c.chunk_id for c in self.chunks]

    def expand_query(self, query: str) -> List[str]:
        """Inflections the index actually contains ("refund" -> "refunds"), plus lexicon alternatives
        ("kept" -> "retained"). TF-IDF matches tokens exactly, so without this a plural mismatch scores 0."""
        expansions: List[str] = []
        for term in dict.fromkeys(content_terms(query)):
            if term not in self._unigram_idf:
                expansions.extend(v for v in self._unigram_idf if terms_match(term, v))
            expansions.extend(self.matcher.alternatives(term))
        return expansions

    def search(self, query: str, top_k: int = 4) -> List[RetrievedChunk]:
        started = time.perf_counter()
        if top_k <= 0 or not content_terms(query):
            log_event(logger, "retrieval.empty_query", query=query)
            return []
        expansions = self.expand_query(query)
        scores = linear_kernel(self.vectorizer.transform([" ".join([query] + expansions)]), self.matrix).ravel()
        # Sort by score desc, then corpus order, so ties are deterministic.
        order = np.lexsort((np.arange(len(scores)), -scores))
        results = [
            RetrievedChunk(
                chunk=self.chunks[i],
                score=float(scores[i]),
                rank=rank,
                above_threshold=bool(scores[i] >= self.min_score),
            )
            for rank, i in enumerate((i for i in order if scores[i] > 0), start=1)
            if rank <= top_k
        ]
        log_event(
            logger,
            "retrieval.search",
            query=query,
            expansions=expansions,
            top_k=top_k,
            results=[(r.chunk.chunk_id, round(r.score, 4)) for r in results],
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return results
