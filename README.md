# AI Support Service — local, citation-first RAG assistant

Answers support questions **only** from the Markdown files in `docs/`, cites the chunks it used, refuses when
the evidence is insufficient, and validates every answer deterministically. The default path needs no network,
no API key, and no model download. Full design: [SPEC.md](SPEC.md).

## Architecture

```
docs/*.md → ingestion → chunks ─┐
                                ▼
question ─────────────→ retrieval (TF-IDF) → prompting → generation → validation → answer + report
```

| Stage | Module | What it does |
|---|---|---|
| Ingestion | `rag/ingestion.py` | Parses Markdown into section-aware chunks (≤120 words). Tables become `Header: value;` lines. Stable IDs like `api_rate_limits#exceeding-the-limit-0`, plus source filename, section path and content hash. |
| Retrieval | `rag/retrieval.py` | scikit-learn `TfidfVectorizer` (uni+bigrams, sublinear tf) with cosine scoring and lexicon query expansion. Returns top-k with an above-threshold flag. |
| Matching | `rag/lexicon.py` | Generic paraphrase lexicon (`kept`→`retained`, `optional`→`mandatory`, …) at reduced credit. Exact matching for numbers and identifiers (`sev2`, `2.50`). Rejects sibling evidence (`sev1` for a `sev2` question, "proof of address" for a "proof of identity" question). |
| Prompting | `rag/prompting.py` | Builds labelled context blocks, the citation whitelist and the instructions. Every backend gets the same prompt. |
| Generation | `rag/generation.py` | `Generator` protocol. Default `ExtractiveGenerator` copies the best-supported 1–2 sentences verbatim with `[chunk_id]` markers and refuses conservatively. Optional `AnthropicGenerator` (opt-in, never required). |
| Validation | `rag/validation.py` | Deterministic checks: non-empty answer, citations present, known, in the retrieved set and matching the inline markers; filename citations consistent with chunk citations; explicit refusal; lexical grounding. Also evaluates against optional labels. |
| Orchestration | `rag/pipeline.py` | `SupportAssistant`, shared by both CLIs. |
| Config / logging | `rag/config.py`, `rag/logging_utils.py` | All thresholds in one dataclass. JSON-lines logs go to stderr. |

## Setup

Requires Python 3.9+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run

One command regenerates `retrieval_results.json`, `answers.json` and `validation_report.json` in the project root:

```bash
python run_pipeline.py
```

Ask a single question. It prints JSON with `answer`, `citations` (document filenames), `supported`, `retrieved_source_ids`
(chunk IDs), `chunk_citations`, scores and validation:

```bash
python app.py --question "How many password reset attempts are allowed per hour?"
```

Useful flags: `--docs DIR`, `--questions FILE`, `--output-dir DIR`, `--top-k N`, `--log-level INFO|WARNING`,
`--show-prompt` (app), `--strict` (pipeline: also fail if outcomes disagree with labels).
`run_pipeline.py` exits 0 when every answer passes validation, 1 on a validation failure, and 2 on input or config errors.

Run the tests:

```bash
python -m pytest -q
```

## Output schema

- `retrieval_results.json` → `results[]`: `question_id`, `question` and `retrieved_chunks[]`, each with `doc_id`
  (source filename), `chunk_id`, `score`, `text`, `rank`, `section` and `above_threshold`.
- `answers.json` → `answers[]`: `question_id`, `question`, `answer`, `citations` (source filenames) and `supported`,
  plus `chunk_citations` (chunk IDs used for validation), `refusal_reason`, `retrieved_source_ids`, `evidence` and
  `diagnostics` (including `lexicon_matches`).
- `validation_report.json` → summary, per-question checks, label evaluation, settings, and the disclaimer that semantic
  support is not verified.

## Using your own documents

Replace `docs/` with any `.md`, `.markdown` or `.txt` files, and replace `questions.json` with either a list of
strings or a list of `{"question_id" (or "id"), "question", "expected_behavior"?, "expected_sources"?}` objects.
`expected_behavior` may be `answerable`/`unanswerable` or `answer`/`refuse`. No code changes are needed. Labels are optional and only feed the evaluation section of the report. They are attached
*after* answering, and a test flips every label to prove the answers don't change.

## Refusal policy (conservative by design)

The extractive generator refuses, with an answer that starts with a fixed prefix and has no citations, when:
no chunk clears the retrieval threshold; the question has no content words; a question term never appears in
the corpus (`unknown_terms`, e.g. "Enterprise plan", "cryptocurrency"); the best sentence is weak; or the best
passage covers less than 90% of the IDF-weighted question terms (`insufficient_coverage`, e.g. "withdrawal
limit for API keys", where every word exists in the docs but no single passage covers them together).
It also refuses when a number or identifier in the question isn't stated in any eligible passage
(`missing_exact_terms`, e.g. "minimum withdrawal 20 USD" when the docs say 10 USD).

Paraphrases are handled by a small generic lexicon (`rag/lexicon.py`). A lexicon match only counts when the literal
word is absent, and it earns 0.8 of a literal match's credit. Thresholds were **not** lowered. Answers stay verbatim,
so "Can I submit a mobile phone bill as proof of address?" returns *"Mobile phone bills and screenshots are not
accepted as proof of address."*: the negation is quoted, not inferred.

## Optional LLM backend

`pip install anthropic`, set `ANTHROPIC_API_KEY`, then run `python app.py --generator anthropic --question "..."`.
It uses model `claude-opus-5-5` (override with `RAG_LLM_MODEL`) and the same prompt and citation whitelist.
Its citations go through the same validator, which catches invented or unretrieved IDs.
This path was **not** exercised in this submission (no key or SDK in the test environment). Everything else runs without it.

## Limitations

- Lexical matching: paraphrases outside the lexicon with no shared word ("old password" vs "last 5
  passwords", "the number of requests") are refused, not answered. A deliberate trade of recall for precision.
- The lexicon is hand-written. Every new entry widens matching, so re-run the near-miss refusal tests.
- No negation reasoning and no yes/no verdicts: answers quote the evidence ("…not accepted…", "…mandatory…"),
  so the reader sees the real wording.
- A 2-sentence evidence window can include a loosely related neighbour sentence.
- Extractive answers can't combine facts across passages, compute, or rephrase.
- Thresholds were calibrated by hand on a few dozen questions. They are config values, not learned.
- Validation proves provenance, format and lexical overlap, **not** semantic entailment. The report says so
  (`"semantic_support": "not_verified"`).
- In-memory index rebuilt on each run. Fine for hundreds of documents, not millions.

## Tradeoffs

| Choice | Why | Cost |
|---|---|---|
| TF-IDF over embeddings | Deterministic, instant, no model download, explainable scores | Misses paraphrases |
| Extractive over generative default | Can't hallucinate facts, fully offline, reproducible artefacts | Terse, sometimes awkward answers |
| Refuse on unknown terms | Strong guard against questions about things the docs never mention | Refuses harmless synonyms the lexicon lacks |
| Small explicit lexicon over embeddings | Auditable, offline, each match shown in `diagnostics` | Hand-maintained, limited coverage |
| Exact numbers / identifiers | `sev2` never borrows `sev1` evidence; "20 USD" never matches "10 USD" | Refuses where a human would correct the number |
| Section-aware chunks, no overlap | Each chunk's heading describes its text; IDs stay stable when other sections change | Facts that span sections aren't co-located |
| Validator separate from generator | Same checks apply to any backend, including an LLM | Checks are structural, not semantic |
