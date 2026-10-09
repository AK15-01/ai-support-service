"""Answer generation behind a replaceable interface.

ExtractiveGenerator (default) is local and deterministic: it copies the best-supported sentences from the
retrieved context and refuses conservatively. AnthropicGenerator is optional and only imported on demand.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Protocol, Tuple

from .config import Settings
from .lexicon import TermMatcher, is_exact_term, of_phrases
from .logging_utils import get_logger, log_event
from .models import REFUSAL_PREFIX, GenerationResult, Prompt
from .text import content_terms, extract_citation_markers, split_units, terms_match, unique_terms

logger = get_logger("generation")


class GenerationError(RuntimeError):
    pass


class Generator(Protocol):
    name: str

    def generate(self, prompt: Prompt) -> GenerationResult: ...


def refusal(reason: str, detail: str, generator: str, diagnostics: Dict[str, Any]) -> GenerationResult:
    return GenerationResult(
        answer=f"{REFUSAL_PREFIX} {detail}".strip(),
        citations=[],
        supported=False,
        refusal_reason=reason,
        diagnostics=diagnostics,
        generator=generator,
    )


class ExtractiveGenerator:
    """Select verbatim evidence sentences; refuse unless the evidence covers the question."""

    name = "extractive"

    def __init__(self, idf: Mapping[str, float], settings: Optional[Settings] = None,
                 matcher: Optional[TermMatcher] = None):
        self.settings = settings or Settings()
        self.idf = dict(idf)
        self.max_idf = max(self.idf.values()) if self.idf else 1.0
        self.matcher = matcher or TermMatcher(expansion_credit=self.settings.lexicon_credit)

    def _vocab_weights(self, term: str) -> List[float]:
        if term in self.idf:
            return [self.idf[term]]
        if is_exact_term(term):
            return []
        return [w for vocab_term, w in self.idf.items() if terms_match(term, vocab_term)]

    def term_weight(self, term: str) -> Tuple[float, bool]:
        """IDF of the term, an inflection, or (failing those) a lexicon alternative.
        Terms the corpus never mentions get the maximum IDF and known=False."""
        weights = self._vocab_weights(term)
        if not weights:
            for alt in self.matcher.alternatives(term):
                weights.extend(self._vocab_weights(alt))
        return (max(weights), True) if weights else (self.max_idf, False)

    def generate(self, prompt: Prompt) -> GenerationResult:
        s = self.settings
        terms = unique_terms(prompt.question)
        exact = [t for t in terms if is_exact_term(t)]
        diagnostics: Dict[str, Any] = {
            "question_terms": terms,
            "exact_terms": exact,
            "related_sources": sorted({item.chunk.source for item in prompt.context}),
        }

        if not terms:
            return self._refuse("no_content_terms", "The question has no specific terms to look up.", diagnostics)
        if not prompt.context:
            return self._refuse("no_relevant_documents", "No document passage is relevant to this question.", diagnostics)

        weights: Dict[str, float] = {}
        unknown: List[str] = []
        for term in terms:
            weight, known = self.term_weight(term)
            weights[term] = weight
            if not known:
                unknown.append(term)
        diagnostics["unknown_terms"] = unknown
        if unknown:
            return self._refuse("unknown_terms", f"The documents do not mention: {', '.join(unknown)}.", diagnostics)

        total = sum(weights.values())
        candidates = self._score_windows(prompt, weights, total)
        if not candidates:
            if exact:
                detail = f"No retrieved passage states this for: {', '.join(exact)}."
                return self._refuse("missing_exact_terms", detail, diagnostics)
            return self._refuse("weak_evidence", "No usable sentence was found in the retrieved passages.", diagnostics)

        best = candidates[0]
        missing = best["chunk_missing"]
        diagnostics.update(
            best_unit_score=round(best["score"], 4),
            best_chunk_coverage=round(best["coverage"], 4),
            terms_missing_from_best_chunk=missing,
            best_window_matched_terms=best["matched"],
            lexicon_matches=best["lexicon"],
        )

        if best["score"] < s.min_sentence_score:
            return self._refuse("weak_evidence", "The retrieved passages do not directly address the question.", diagnostics)
        if best["coverage"] < s.min_coverage:
            detail = "The closest passage does not cover: " + ", ".join(missing) + "."
            return self._refuse("insufficient_coverage", detail, diagnostics)

        # Start from the best window, then add single supporting sentences that are on-topic for every term
        # the best window covers (so a "Basic plan" row is not appended to a "Pro plan" answer).
        cutoff = max(s.min_sentence_score, s.relative_sentence_cutoff * best["score"])
        selected = [(best["chunk_id"], best["pos"] + i, unit) for i, unit in enumerate(best["units"])]
        meta = {best["chunk_id"]: best}
        for cand in candidates[1:]:
            if len(selected) >= s.max_answer_sentences:
                break
            key = (cand["chunk_id"], cand["pos"], cand["units"][0])
            if (len(cand["units"]) == 1 and key not in selected and cand["score"] >= cutoff
                    and cand["coverage"] >= s.min_coverage and best["covered"] <= cand["covered"]):
                selected.append(key)
                meta.setdefault(cand["chunk_id"], cand)
        selected.sort(key=lambda k: (meta[k[0]]["rank"], k[1]))  # present in reading order

        parts, citations, evidence = [], [], []
        for chunk_id, _, unit in selected:
            sentence = unit if unit.endswith((".", "!", "?")) else unit + "."
            parts.append(f"{sentence} [{chunk_id}]")
            if chunk_id not in citations:
                citations.append(chunk_id)
            evidence.append({"chunk_id": chunk_id, "doc_id": meta[chunk_id]["source"], "text": unit})

        log_event(logger, "generation.answered", question=prompt.question, citations=citations,
                  best_unit_score=diagnostics["best_unit_score"], coverage=diagnostics["best_chunk_coverage"],
                  lexicon_matches=best["lexicon"])
        return GenerationResult(
            answer=" ".join(parts),
            citations=citations,
            supported=True,
            evidence=evidence,
            diagnostics=diagnostics,
            generator=self.name,
        )

    def _score_windows(self, prompt: Prompt, weights: Dict[str, float], total: float) -> List[Dict[str, Any]]:
        """Score windows of 1-2 consecutive answer units per chunk.

        - A term split across adjacent sentences still counts.
        - Exact terms (numbers, identifiers such as "sev2") must appear in the window or its heading.
        - A sentence naming a different identifier of the same kind ("sev1" when the question asks about
          "sev2"), or a sibling specifier ("proof of address" when it asks about "proof of identity"), is
          never used as evidence.
        """
        terms = list(weights)
        question_phrases = of_phrases(prompt.question)
        exact = [t for t in terms if is_exact_term(t)]
        match = self.matcher.strength
        candidates: List[Dict[str, Any]] = []
        for item in prompt.context:
            chunk = item.chunk
            chunk_tokens = set(content_terms(chunk.index_text))
            chunk_strength = {t: match(t, chunk_tokens) for t in terms}
            coverage = sum(weights[t] * chunk_strength[t] for t in terms) / total
            heading_tokens = set(content_terms(chunk.section))
            # Heading context supports a sentence, but counts half as much as the sentence itself.
            heading_strength = {t: 0.5 * match(t, heading_tokens) for t in terms}
            units = split_units(chunk.text)
            unit_tokens = [set(content_terms(u)) for u in units]
            unit_strength = [{t: match(t, toks) for t in terms} for toks in unit_tokens]
            conflicted = [
                bool(self.matcher.conflicting_identifiers(terms, toks)
                     or self.matcher.conflicting_specifiers(question_phrases, unit))
                for unit, toks in zip(units, unit_tokens)
            ]
            for pos in range(len(units)):
                for length in (1, 2):
                    span = range(pos, pos + length)
                    if pos + length > len(units) or any(conflicted[i] for i in span):
                        continue
                    in_units = {t: max(unit_strength[i][t] for i in span) for t in terms}
                    combined = {t: max(in_units[t], heading_strength[t]) for t in terms}
                    if any(combined[t] == 0 for t in exact):
                        continue
                    window_tokens = set().union(*(unit_tokens[i] for i in span))
                    lexicon = {}
                    for t in terms:
                        via = self.matcher.match(t, window_tokens)[1]
                        if via:
                            lexicon[t] = via
                    candidates.append(
                        {
                            "chunk_id": chunk.chunk_id,
                            "source": chunk.source,
                            "units": units[pos:pos + length],
                            "score": sum(weights[t] * combined[t] for t in terms) / total,
                            "matched": [t for t in terms if in_units[t] > 0],
                            "covered": {t for t in terms if combined[t] > 0},
                            "lexicon": lexicon,
                            "coverage": coverage,
                            "chunk_missing": [t for t in terms if chunk_strength[t] == 0],
                            "rank": item.rank,
                            "pos": pos,
                        }
                    )
        # Highest score first; prefer the shorter window on ties, then retrieval rank, then position.
        candidates.sort(key=lambda c: (-round(c["score"], 9), len(c["units"]), c["rank"], c["pos"]))
        return candidates

    def _refuse(self, reason: str, detail: str, diagnostics: Dict[str, Any]) -> GenerationResult:
        log_event(logger, "generation.refused", reason=reason, detail=detail)
        return refusal(reason, detail, self.name, diagnostics)


class AnthropicGenerator:
    """Optional LLM backend. Requires `pip install anthropic` and Anthropic credentials."""

    name = "anthropic"

    def __init__(self, model: Optional[str] = None, max_tokens: int = 16000):
        try:
            import anthropic  # noqa: F401  (imported lazily so the default path never needs it)
        except ImportError as exc:
            raise GenerationError("The anthropic package is not installed: pip install anthropic") from exc
        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("RAG_LLM_MODEL", "claude-opus-5-5")
        self.max_tokens = max_tokens

    def generate(self, prompt: Prompt) -> GenerationResult:
        diagnostics: Dict[str, Any] = {"model": self.model, "allowed_citations": prompt.allowed_citations}
        if not prompt.context:
            return refusal("no_relevant_documents", "No document passage is relevant to this question.", self.name, diagnostics)
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                output_config={"effort": "medium"},
                system=prompt.system,
                messages=[{"role": "user", "content": prompt.user}],
            )
        except self._anthropic.APIError as exc:
            raise GenerationError(f"LLM request failed: {exc}") from exc

        diagnostics["stop_reason"] = response.stop_reason
        if response.stop_reason == "refusal":
            return refusal("model_refusal", "The model declined to answer.", self.name, diagnostics)
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        if not text or text.startswith(REFUSAL_PREFIX):
            return refusal("model_insufficient_context", "", self.name, diagnostics)
        citations: List[str] = []
        for marker in extract_citation_markers(text):
            if marker not in citations:
                citations.append(marker)
        # Citations are reported as produced; validation decides whether they are legitimate.
        return GenerationResult(answer=text, citations=citations, supported=True, diagnostics=diagnostics, generator=self.name)


def create_generator(name: str, idf: Mapping[str, float], settings: Settings,
                     matcher: Optional[TermMatcher] = None) -> Generator:
    if name == "extractive":
        return ExtractiveGenerator(idf, settings, matcher)
    if name == "anthropic":
        return AnthropicGenerator()
    raise ValueError(f"Unknown generator '{name}'. Choose 'extractive' or 'anthropic'.")
