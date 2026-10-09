"""Generic support-language lexicon and term matching.

QUERY_EXPANSIONS maps a word a customer might use to words a policy document typically uses for the same
concept or property ("kept" -> "retained", "optional" -> "mandatory": both describe whether something is
required). Entries are generic support vocabulary and name no document, product, plan or entity; replace or
extend the table for another domain. Expansions are one-directional and are only consulted when the literal
word (or an inflection of it) is absent, and a lexicon match earns less credit than a literal match.

Tokens containing digits ("sev2", "30", "2.50") are exact terms: they never match morphologically or via
the lexicon, so numeric and identifier distinctions are preserved.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

from .text import term_in, terms_match

QUERY_EXPANSIONS: Dict[str, Tuple[str, ...]] = {
    # retention and deletion
    "kept": ("retained", "stored", "held"),
    "keep": ("retain", "store", "hold"),
    "keeps": ("retains", "stores", "holds"),
    "store": ("retain", "keep"),
    "delete": ("remove", "erase", "deletion"),
    "deleted": ("removed", "erased", "deletion"),
    "remove": ("delete", "deletion"),
    # money and cost
    "money": ("funds", "balance"),
    "cash": ("funds", "balance"),
    "cost": ("fee", "fees", "charge", "free"),
    "costs": ("fee", "fees", "charge", "free"),
    "charge": ("fee", "fees", "cost", "costs", "free"),
    "charged": ("fee", "fees", "cost", "costs", "free"),
    "pay": ("fee", "fees", "cost", "costs", "charge", "free"),
    "price": ("fee", "fees", "cost", "costs"),
    "free": ("fee", "fees", "cost", "costs", "charge"),
    # requirement status: "is X optional?" is answered by a statement that X is mandatory/required
    "optional": ("mandatory", "required", "requirement"),
    "mandatory": ("required", "optional", "requirement"),
    "required": ("mandatory", "requirement"),
    "compulsory": ("mandatory", "required"),
    "obligatory": ("mandatory", "required"),
    # ordinals and limits
    "second": ("additional", "subsequent", "extra"),
    "third": ("additional", "subsequent", "extra"),
    "extra": ("additional",),
    "another": ("additional",),
    "maximum": ("limit", "limited", "capped", "cap"),
    "max": ("maximum", "limit", "capped"),
    "minimum": ("least", "min"),
    "daily": ("day",),
    "weekly": ("week",),
    "monthly": ("month",),
    "hourly": ("hour",),
    "yearly": ("year",),
    "annual": ("year",),
    # actions: "can I submit X?" is answered by whether X is accepted
    "submit": ("accepted", "upload", "provide", "send"),
    "send": ("submit", "upload"),
    "upload": ("submit", "send"),
    "login": ("sign",),
    "log": ("sign",),
    "blocked": ("locked", "suspended"),
    "banned": ("suspended", "locked"),
    "photo": ("selfie", "image"),
    "picture": ("photo", "selfie", "image"),
    "id": ("identity", "identification"),
}

_DIGIT_RE = re.compile(r"\d")
_OF_PHRASE_RE = re.compile(r"\b([a-z][a-z0-9]+) of (?:the |a |an |your |my |their )?([a-z0-9][a-z0-9.,]*[a-z0-9]|[a-z0-9])\b")


def of_phrases(text: str) -> Set[Tuple[str, str]]:
    """Specifier phrases: "proof of address" -> ("proof", "address")."""
    return set(_OF_PHRASE_RE.findall(text.lower()))


def is_exact_term(term: str) -> bool:
    return bool(_DIGIT_RE.search(term))


def identifier_pattern(term: str) -> Optional[str]:
    """'sev2' -> 'sev#'. Only alphanumeric identifiers have a pattern; plain numbers do not."""
    if is_exact_term(term) and re.search(r"[a-z]", term):
        return re.sub(r"\d+(?:[.,]\d+)*", "#", term)
    return None


class TermMatcher:
    def __init__(self, expansions: Optional[Mapping[str, Sequence[str]]] = None, expansion_credit: float = 0.8):
        self.expansions = QUERY_EXPANSIONS if expansions is None else expansions
        self.expansion_credit = expansion_credit

    def alternatives(self, term: str) -> Sequence[str]:
        return () if is_exact_term(term) else self.expansions.get(term, ())

    def match(self, term: str, tokens: Iterable[str]) -> Tuple[float, Optional[str]]:
        """Return (strength, matched_via): 1.0 literal/inflection, expansion_credit via lexicon, else 0."""
        tokens = tokens if isinstance(tokens, (set, frozenset)) else set(tokens)
        if is_exact_term(term):
            return (1.0, None) if term in tokens else (0.0, None)
        if term_in(term, tokens):
            return 1.0, None
        for alt in self.alternatives(term):
            if term_in(alt, tokens):
                return self.expansion_credit, alt
        return 0.0, None

    def strength(self, term: str, tokens: Iterable[str]) -> float:
        return self.match(term, tokens)[0]

    def conflicting_identifiers(self, question_terms: Iterable[str], tokens: Set[str]) -> Set[str]:
        """Identifiers in `tokens` that look like a question identifier but differ (sev1 vs sev2)."""
        wanted = {t for t in question_terms if identifier_pattern(t)}
        patterns = {identifier_pattern(t) for t in wanted}
        return {tok for tok in tokens if tok not in wanted and identifier_pattern(tok) in patterns}

    @staticmethod
    def conflicting_specifiers(question_phrases: Set[Tuple[str, str]], text: str) -> Set[Tuple[str, str]]:
        """"proof of address" in `text` when the question asks about "proof of identity" (and `text` never
        says "proof of identity"): the sentence is about a sibling concept."""
        found = of_phrases(text)
        conflicts = set()
        for head, spec in question_phrases:
            same_head = [(h, s) for h, s in found if terms_match(h, head)]
            if same_head and not any(terms_match(s, spec) for _, s in same_head):
                conflicts.update(same_head)
        return conflicts
