"""Tokenisation and lexical matching shared by retrieval, generation and validation."""

from __future__ import annotations

import os
import re
from typing import Iterable, List

# Decimal / thousands numbers stay whole ("2.50", "10,000") so they are not confused with "2" or "50".
_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)+|[a-z0-9]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_LIST_MARKER_RE = re.compile(r"^(?:[-*+]|\d+[.)])\s+")
_CITATION_RE = re.compile(r"\[([^\[\]\s]+#[^\[\]\s]+)\]")

# Standard English function words (NLTK-style list; sklearn's list is avoided because it
# contains content words such as "bill", "amount" and "system").
_ENGLISH_STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had has have
    having he her here hers herself him himself his how i if in into is it its itself just me more most
    my myself no nor not now of off on once only or other our ours ourselves out over own same she should
    so some such than that the their theirs them themselves then there these they this those through to
    too under until up very was we were what when where which while who whom why will with would you your
    yours yourself yourselves s t d ll m o re ve y ain aren couldn didn doesn don hadn hasn haven isn ma
    mightn mustn needn shan shouldn wasn weren won wouldn
    """.split()
)

# Words that frame a support question but carry no topical content ("how long does X take",
# "am I allowed to", "what happens if"). Kept small and generic on purpose.
_QUESTION_FRAMING_WORDS = frozenset(
    """
    also may might must shall per via please tell know explain happen happens happened long many much
    allowed permitted able possible usually typically generally normally currently exactly need needs
    take takes taken get gets allow allows often ever way ways mean means use using
    """.split()
)

STOPWORDS = _ENGLISH_STOPWORDS | _QUESTION_FRAMING_WORDS

# Inflectional endings accepted when one term is a prefix of the other.
_SUFFIXES = frozenset(
    ["s", "es", "d", "ed", "ing", "er", "ers", "al", "als", "ly", "ion", "ions", "ation", "ations", "ment", "ments"]
)


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def content_terms(text: str) -> List[str]:
    """Lowercase tokens without stop words; single letters dropped, digits kept."""
    return [tok for tok in tokenize(text) if tok not in STOPWORDS and (len(tok) > 1 or tok.isdigit())]


def unique_terms(text: str) -> List[str]:
    seen: dict = {}
    for term in content_terms(text):
        seen.setdefault(term, None)
    return list(seen)


def terms_match(a: str, b: str) -> bool:
    """Exact match or light morphological match (hour/hours, verify/verification, exceed/exceeding)."""
    if a == b:
        return True
    if any(ch.isdigit() for ch in a + b):  # numbers and identifiers (sev1, 2.50) only match exactly
        return False
    short, long_ = sorted((a, b), key=len)
    if long_.startswith(short):  # enter/entered match, enter/enterprise must not
        rest = long_[len(short):]
        doubled = len(rest) > 1 and rest[0] == short[-1] and rest[1:] in _SUFFIXES  # cancel/cancelled
        return len(short) >= 3 and (rest in _SUFFIXES or doubled)
    common = len(os.path.commonprefix([a, b]))
    return common >= 5 and common >= 0.7 * len(short)


def term_in(term: str, tokens: Iterable[str]) -> bool:
    return any(terms_match(term, tok) for tok in tokens)


def split_sentences(text: str) -> List[str]:
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]


def split_units(text: str) -> List[str]:
    """Split chunk text into answer units: lines (list items, table rows), then sentences."""
    units: List[str] = []
    for line in text.split("\n"):
        line = _LIST_MARKER_RE.sub("", line.strip())
        if not line:
            continue
        for sentence in split_sentences(line):
            if len(tokenize(sentence)) >= 3:
                units.append(sentence)
    return units


def extract_citation_markers(text: str) -> List[str]:
    return _CITATION_RE.findall(text)


def strip_citation_markers(text: str) -> str:
    return _CITATION_RE.sub("", text).strip()


def slugify(text: str, sep: str = "-") -> str:
    slug = re.sub(r"[^a-z0-9]+", sep, text.lower()).strip(sep)
    return slug or "section"
