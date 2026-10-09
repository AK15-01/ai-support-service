"""Deterministic answer validation.

These checks establish provenance and format (citations exist, were retrieved, refusals are explicit) plus a
lexical-overlap heuristic. They do NOT establish that the cited text semantically entails the answer.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .models import REFUSAL_PREFIX, Chunk, GenerationResult
from .text import content_terms, extract_citation_markers, term_in

SEMANTIC_DISCLAIMER = (
    "Validation verifies answer format, citation provenance (cited chunk IDs exist and were retrieved for the "
    "question), explicit refusals, and lexical overlap between answer sentences and cited text. It does not "
    "verify that the cited text semantically supports or entails the answer."
)

_SEGMENT_RE = re.compile(r"(.*?)((?:\s*\[[^\[\]\s]+#[^\[\]\s]+\])+)", re.DOTALL)


def _check(name: str, passed: bool, detail: str = "") -> Dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def lexical_grounding(answer: str, corpus: Mapping[str, Chunk]) -> List[Dict[str, Any]]:
    """Per answer sentence: share of its content terms found in the chunk(s) it cites."""
    segments: List[Dict[str, Any]] = []
    consumed = 0
    for match in _SEGMENT_RE.finditer(answer):
        consumed = match.end()
        text, ids = match.group(1).strip(), extract_citation_markers(match.group(2))
        cited_tokens = set()
        for cid in ids:
            if cid in corpus:
                cited_tokens.update(content_terms(corpus[cid].index_text))
        terms = content_terms(text)
        overlap = sum(1 for t in terms if term_in(t, cited_tokens)) / len(terms) if terms else 1.0
        segments.append({"sentence": text, "cited": ids, "overlap": round(overlap, 4)})
    tail = answer[consumed:].strip()
    if content_terms(tail):  # trailing claim with no citation at all
        segments.append({"sentence": tail, "cited": [], "overlap": 0.0})
    return segments


def validate_answer(
    result: GenerationResult,
    retrieved_ids: Sequence[str],
    corpus: Mapping[str, Chunk],
    min_grounding_overlap: float = 0.6,
    document_citations: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """`citations` on the result are chunk IDs. `document_citations`, if given, are the public
    filename-level citations and must be exactly the documents of the cited chunks."""
    answer = result.answer if isinstance(result.answer, str) else ""
    citations = list(result.citations or [])
    supported = result.supported
    retrieved = set(retrieved_ids)
    markers = extract_citation_markers(answer)
    is_refusal_text = answer.strip().startswith(REFUSAL_PREFIX)
    checks = [_check("answer_nonempty", bool(answer.strip()), "answer text must not be blank")]
    checks.append(_check("supported_is_bool", isinstance(supported, bool), f"type={type(supported).__name__}"))

    if supported:
        checks.append(_check("citations_present", bool(citations), f"{len(citations)} citation(s)"))
    else:
        checks.append(_check("citations_present", True, "not required for a refusal"))

    unknown = [c for c in citations if c not in corpus]
    checks.append(_check("citations_known", not unknown, f"unknown: {unknown}" if unknown else "all cited IDs exist"))

    outside = [c for c in citations if c not in retrieved]
    checks.append(
        _check("citations_in_retrieved", not outside,
               f"not retrieved: {outside}" if outside else "all cited IDs were retrieved for this question")
    )

    if document_citations is not None:
        expected_docs = list(dict.fromkeys(corpus[c].source for c in citations if c in corpus))
        retrieved_docs = {corpus[c].source for c in retrieved if c in corpus}
        ok = list(document_citations) == expected_docs and set(document_citations) <= retrieved_docs
        checks.append(_check("document_citations_consistent", ok,
                             f"documents={list(document_citations)} derived_from_chunks={expected_docs}"))

    if supported:
        consistent = bool(markers) and set(markers) == set(citations)
        detail = f"inline={sorted(set(markers))} listed={sorted(set(citations))}"
    else:
        consistent = not markers
        detail = "refusal contains no inline citations" if consistent else f"refusal has markers {markers}"
    checks.append(_check("inline_citations_consistent", consistent, detail))

    if supported:
        checks.append(_check("refusal_marked", not is_refusal_text, "supported answer is not phrased as a refusal"))
    else:
        ok = is_refusal_text and not citations
        checks.append(_check("refusal_marked", ok, "refusal starts with the refusal prefix and cites nothing"
                             if ok else "unsupported answer must start with the refusal prefix and cite nothing"))

    grounding: List[Dict[str, Any]] = []
    if supported:
        grounding = lexical_grounding(answer, corpus)
        low = [g for g in grounding if g["overlap"] < min_grounding_overlap]
        checks.append(
            _check("lexical_grounding", bool(grounding) and not low,
                   f"min sentence overlap {min((g['overlap'] for g in grounding), default=0.0):.2f} "
                   f"(threshold {min_grounding_overlap}); heuristic, not entailment")
        )

    return {
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
        "grounding": grounding,
        "semantic_support": "not_verified",
    }


_LABELS = {
    "answer": "answer", "answerable": "answer", "supported": "answer",
    "refuse": "refuse", "unanswerable": "refuse", "unsupported": "refuse", "partially_supported": "refuse",
    "partially_unsupported": "refuse",
}


def normalize_expected_behavior(label: Any) -> Optional[str]:
    """Map label vocabularies ("answerable"/"unanswerable", "answer"/"refuse", ...) to "answer"/"refuse"."""
    if not isinstance(label, str):
        return None
    return _LABELS.get(label.strip().lower().replace("-", "_").replace(" ", "_"))


def evaluate_against_labels(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Compare outcomes with optional labels in questions.json. Used for reporting only, never for answering."""
    behaviour_rows, retrieval_rows, unrecognised = [], [], []
    for rec in records:
        raw = rec.get("expected_behavior")
        expected = normalize_expected_behavior(raw)
        if raw is not None and expected is None:
            unrecognised.append({"question_id": rec["question_id"], "label": raw})
        if expected:
            actual = "answer" if rec["supported"] else "refuse"
            behaviour_rows.append({"question_id": rec["question_id"], "label": raw, "expected": expected,
                                   "actual": actual, "match": expected == actual})
        expected_sources = rec.get("expected_sources") or []
        if expected_sources:
            retrieved_sources = rec.get("retrieved_doc_ids", [])
            retrieval_rows.append({
                "question_id": rec["question_id"],
                "expected_sources": expected_sources,
                "top1_hit": bool(retrieved_sources) and retrieved_sources[0] in expected_sources,
                "topk_hit": any(src in expected_sources for src in retrieved_sources),
            })

    def rate(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
        return round(sum(r[key] for r in rows) / len(rows), 4) if rows else None

    return {
        "labelled_questions": len(behaviour_rows),
        "behavior_accuracy": rate(behaviour_rows, "match"),
        "behavior_mismatches": [r for r in behaviour_rows if not r["match"]],
        "unrecognised_labels": unrecognised,
        "retrieval_top1_hit_rate": rate(retrieval_rows, "top1_hit"),
        "retrieval_topk_hit_rate": rate(retrieval_rows, "topk_hit"),
        "retrieval_misses": [r for r in retrieval_rows if not r["topk_hit"]],
        "behavior": behaviour_rows,
        "retrieval": retrieval_rows,
    }


def build_validation_report(records: List[Mapping[str, Any]], settings: Mapping[str, Any]) -> Dict[str, Any]:
    check_failures: Dict[str, int] = {}
    for rec in records:
        for check in rec["validation"]["checks"]:
            if not check["passed"]:
                check_failures[check["name"]] = check_failures.get(check["name"], 0) + 1
    passed = sum(1 for r in records if r["validation"]["passed"])
    return {
        "summary": {
            "questions": len(records),
            "validation_passed": passed,
            "validation_failed": len(records) - passed,
            "answered": sum(1 for r in records if r["supported"]),
            "refused": sum(1 for r in records if not r["supported"]),
            "check_failures": check_failures,
            "all_passed": passed == len(records),
        },
        "semantic_support": "not_verified",
        "disclaimer": SEMANTIC_DISCLAIMER,
        "evaluation": evaluate_against_labels(records),
        "per_question": [
            {
                "question_id": r["question_id"],
                "question": r["question"],
                "supported": r["supported"],
                "refusal_reason": r.get("refusal_reason"),
                "passed": r["validation"]["passed"],
                "checks": r["validation"]["checks"],
                "grounding": r["validation"]["grounding"],
            }
            for r in records
        ],
        "settings": dict(settings),
    }
