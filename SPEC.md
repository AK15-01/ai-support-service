# SPEC — Local RAG Support Assistant

Status: design spec, written before the implementation and revised after review (v1.1: public output schema,
label vocabularies, paraphrase lexicon, exact-term / sibling rules). The README summarises it; this document is
the contract.

## 1. Goal and non-goals

**Goal.** A runnable Python support assistant that answers customer-support questions **only** from local
Markdown documents, cites the chunks it used, refuses when the evidence is insufficient, and validates
every answer deterministically.

**Non-goals.** No network calls in the default path, no vector database, no embeddings model, no
fine-tuning, no claim of semantic fact-checking.

## 2. Architecture

```
docs/*.md ──► ingestion ──► chunks (id, source, section, text, hash)
                                │
                                ▼
question ──► retrieval (TF-IDF, cosine) ──► top-k scored chunks
                                │
                                ▼
                prompting (context blocks + citation whitelist + instructions)
                                │
                                ▼
                generation  (Generator interface)
                   ├─ ExtractiveGenerator  (default, local, deterministic)
                   └─ AnthropicGenerator   (optional, needs SDK + API key)
                                │
                                ▼
                validation (deterministic checks) ──► answer record
```

| Module | Responsibility | Public surface |
|---|---|---|
| `rag/config.py` | All tunables in one dataclass | `Settings` |
| `rag/models.py` | Shared data types and constants | `Chunk`, `RetrievedChunk`, `Prompt`, `GenerationResult`, `REFUSAL_PREFIX` |
| `rag/text.py` | Tokenisation, stop words, term matching, sentence splitting | `content_terms`, `terms_match`, `split_units` |
| `rag/lexicon.py` | Generic paraphrase lexicon, exact terms, sibling conflicts | `TermMatcher`, `QUERY_EXPANSIONS` |
| `rag/ingestion.py` | Load + normalise Markdown, chunk with metadata | `load_documents`, `chunk_document`, `ingest_directory` |
| `rag/retrieval.py` | TF-IDF index + search | `TfidfRetriever` |
| `rag/prompting.py` | Build the prompt / context bundle | `build_prompt` |
| `rag/generation.py` | Replaceable generator interface + implementations | `Generator`, `ExtractiveGenerator`, `AnthropicGenerator`, `create_generator` |
| `rag/validation.py` | Deterministic answer checks + report | `validate_answer`, `build_validation_report` |
| `rag/pipeline.py` | Orchestration used by both CLIs | `SupportAssistant` |
| `rag/logging_utils.py` | Structured JSON logs to stderr | `configure_logging`, `log_event` |
| `run_pipeline.py` | Batch run over `questions.json`, writes 3 JSON artefacts | CLI |
| `app.py` | Single question CLI, JSON to stdout | CLI |

Design rules:
- Modules communicate through the dataclasses in `models.py`, never through globals.
- Nothing in code refers to specific document names, sections, or question wording; the corpus and the
  question set are inputs. Replacing `docs/` and `questions.json` must not require code changes.
- Logs go to **stderr** as JSON lines; machine output (answers) goes to **stdout** / files.

## 3. Ingestion and chunking

1. Discover `*.md`, `*.markdown`, `*.txt` in the docs directory, **sorted by filename** (determinism).
   Empty directory or no readable documents → clear error, non-zero exit.
2. Normalise Markdown per line:
   - `# H1` → document title (fallback: filename stem). `##`/`###` → section heading path.
   - Headings inside fenced code blocks are ignored.
   - Tables → one line per data row: `Header1: v1; Header2: v2; ...` so each row is a self-contained
     extractable fact.
   - Strip emphasis markers (`**`, `__`, backticks); links `[t](u)` → `t (u)`.
3. Chunking: section-aware packing. Paragraph/list/table-row blocks of one section are packed until
   `max_chunk_words` (default 120). An oversized block is split by lines, then by sentences.
   Chunks never cross a section boundary, so a chunk's heading always describes its text.
4. Chunk metadata:
   - `chunk_id = "{doc_id}#{section_slug}-{n}"`, e.g. `authentication_password_reset#password-reset-limits-0`.
     `doc_id` = slugified filename stem; `section_slug` = slug of the deepest heading (deduplicated per doc);
     `n` = index within the section. IDs are **stable**: same input → same IDs, and editing one section does not
     renumber chunks in other sections.
   - `source` (filename), `doc_title`, `section` (heading path), `chunk_index` (global order in doc),
     `text`, `word_count`, `content_hash` (sha1 prefix of text, for change detection).
5. Retrieval text = `doc_title + section + text` (headings add topical signal to short chunks).

## 4. Retrieval

- `sklearn.feature_extraction.text.TfidfVectorizer` with the shared tokenizer from `text.py`
  (lowercase alphanumerics, English + question-framing stop words), `ngram_range=(1, 2)`,
  `sublinear_tf=True`, L2 norm.
- Score = cosine similarity (`linear_kernel` on L2-normalised vectors).
- Return the `top_k` (default 4) chunks with score > 0, ordered by score desc, ties broken by corpus order
  (deterministic). Each result carries rank, score and `above_threshold = score >= min_retrieval_score`
  (default 0.08).
- Query expansion: lexicon alternatives of question words are appended to the query (§6a), so "kept" also
  retrieves passages that say "retained".
- The retriever exposes unigram IDF weights so the generator can weight question terms by specificity.

## 5. Prompt construction

`build_prompt(question, retrieved, settings) → Prompt`:
- Keeps only chunks with `above_threshold`, in rank order, within `max_context_chars` (default 6000).
- Each context block is labelled with its chunk ID and source:
  `[chunk_id] (source › section)\ntext`.
- System instructions: answer only from context; cite `[chunk_id]` after every sentence; use only the listed
  IDs; if the context is insufficient, reply starting with the exact refusal prefix.
- `allowed_citations` = IDs of included blocks. The prompt object is used by **every** generator (the extractive
  generator reads the context blocks; an LLM generator sends `system`/`user`), so the citation whitelist is
  identical regardless of backend.

## 6. Answer generation

Interface (`typing.Protocol`):
```python
class Generator(Protocol):
    name: str
    def generate(self, prompt: Prompt) -> GenerationResult: ...
```
`GenerationResult = {answer, citations, supported, refusal_reason, evidence[], diagnostics{}}`.

**ExtractiveGenerator (default).** Fully local and deterministic:
1. Question content terms = tokens minus stop words, deduplicated. Each term is weighted by corpus IDF
   (terms absent from the corpus get the maximum IDF — they are maximally specific).
2. Split context chunks into answer units (lines / list items / table rows, then sentences).
3. Score **windows of 1–2 consecutive units** inside a chunk: IDF-weighted fraction of question terms matched
   in the window (terms that only occur in the section heading count half). Windows let a discriminating term
   in the next sentence count ("During a SEV1 incident … every 30 minutes. During a SEV2 incident … every 2
   hours."). Ties prefer the shorter window. Term matching is exact or light morphological
   (`hour`/`hours`, `verify`/`verification`, `cancel`/`cancelled`, but not `enter`/`enterprise`; see `terms_match`).
4. Apply refusal logic (§8). If supported, start from the best window and add single supporting units, up to
   `max_answer_sentences` (default 2), that score ≥ `min_sentence_score` and ≥ 60 % of the best, **and** cover
   every question term the best window covers (so a "Basic plan" row is never appended to a "Pro plan" answer).
   Emit them verbatim in reading order, each followed by its `[chunk_id]`. Citations = unique chunk IDs used.

The answer is copied text, never paraphrased, so the extractive backend cannot invent facts — it can only
select the wrong sentence, which validation and refusal thresholds aim to catch.

### 6a. Term matching rules (shared by retrieval and generation via `TermMatcher`)

| Rule | Example | Effect |
|---|---|---|
| Literal / inflection match | `hour`/`hours`, `cancel`/`cancelled` | full credit (1.0) |
| Lexicon match, only if no literal match | `kept`→`retained`, `optional`→`mandatory`, `second`→`additional`, `submit`→`accepted`, `money`→`funds` | `lexicon_credit` (0.8). Recorded in `diagnostics.lexicon_matches` |
| Exact terms: any token with a digit | `sev2`, `30`, `2.50`, `10,000` | never fuzzy or lexicon-matched; must appear in the evidence window or its heading, else the window is ineligible |
| Sibling identifier | question `sev2`, sentence mentions only `sev1` | sentence never used as evidence |
| Sibling specifier ("A of B") | question "proof of identity", sentence "proof of address" | sentence never used as evidence |

The lexicon (`rag/lexicon.py`) is generic support vocabulary. It names no document, product, plan or entity,
its expansions are one-directional, and it can be replaced (`SupportAssistant(matcher=TermMatcher(expansions=...))`).
It answers paraphrases by **quoting** the evidence, so negations ("not accepted") and figures ("2.50 USD") reach
the user unchanged. The extractive generator never synthesises a yes or no.

**AnthropicGenerator (optional).** Sends the same prompt to an LLM via the `anthropic` SDK, parses `[chunk_id]`
markers into citations, and marks `supported=False` when the reply starts with the refusal prefix. Only loaded
when `--generator anthropic` is passed; the SDK and API key are never required otherwise.

## 7. Validation (deterministic)

Per answer, each check yields `{name, passed, detail}`:

| Check | Rule |
|---|---|
| `answer_nonempty` | answer text is non-blank |
| `supported_is_bool` | `supported` is a real boolean |
| `citations_present` | supported ⇒ ≥ 1 citation |
| `citations_known` | every chunk citation is a chunk ID that exists in the corpus |
| `document_citations_consistent` | the public filename `citations` equal the documents of the chunk citations, and each was retrieved |
| `citations_in_retrieved` | every citation is among the chunk IDs retrieved for **this** question and passed to the generator (above threshold) |
| `inline_citations_consistent` | supported ⇒ set of inline `[id]` markers == citation list; refused ⇒ no markers |
| `refusal_marked` | unsupported ⇒ answer starts with `REFUSAL_PREFIX` **and** has no citations; supported ⇒ answer does not start with it |
| `lexical_grounding` | supported ⇒ every answer sentence has ≥ 60 % of its content tokens present in its cited chunk(s) |

Important distinction: passing these checks proves **provenance and format**, not truth. A citation can be
valid (it exists and was retrieved) while the cited text does not entail the answer. `lexical_grounding` is
a heuristic overlap check, not entailment. The report therefore states `"semantic_support": "not_verified"`.

Separately, when `questions.json` carries labels (`expected_behavior`, `expected_sources`), the report includes
an **evaluation** section: refusal-decision accuracy and retrieval hit@k. Labels are optional; unlabelled
question files still run. `expected_behavior` accepts `answerable`/`unanswerable`, `answer`/`refuse` (and
`supported`/`unsupported`/`partially supported`), case-insensitive. Unrecognised labels are listed, not guessed.
Labels are attached to records only **after** answering, and `SupportAssistant.answer()` takes only the question
text, so labels can't influence answers. A test flips every label and asserts the answers are byte-identical.

## 8. Refusal logic (conservative)

The generator refuses (answer starts with `REFUSAL_PREFIX`, `supported=false`, no citations) when any holds,
checked in order:

1. `no_relevant_documents` — no retrieved chunk reaches `min_retrieval_score`.
2. `no_content_terms` — the question has no content words after stop-word removal.
3. `unknown_terms` — a question content term does not occur anywhere in the corpus (e.g. a product, plan, or
   payment rail the docs never mention). The KB cannot support a claim about something it never names.
4. `missing_exact_terms` — the question contains a number or identifier and no eligible window states it
   (e.g. "minimum withdrawal 20 USD" when the docs say 10 USD).
5. `weak_evidence` — best window score < `min_sentence_score` (default 0.35).
6. `insufficient_coverage` — the chunk holding the best window covers < `min_coverage` (default 0.9) of the
   IDF-weighted question terms (catches partially answerable questions whose terms all exist in the corpus but
   not in one passage, e.g. "withdrawal limit for API keys").

`no_content_terms` is checked before `no_relevant_documents` so the reason is the most specific one.
Question-framing words ("how long", "allowed", "what does X mean") are stop words, not topic terms.

The refusal text names the reason and unmatched terms; the closest sources are kept in `diagnostics` for a
human agent, not presented as citations. Preference: a false refusal is cheaper than a confident wrong answer.

## 9. Outputs

`run_pipeline.py` regenerates, in the output directory (default: project root):
- `retrieval_results.json` — `results[]`, each with `question_id`, `question` and `retrieved_chunks[]`. Each chunk has
  `doc_id` (source filename), `chunk_id`, `score`, `text`, `rank`, `section` and `above_threshold`, plus
  `expected_sources` and `expected_source_hit` when labelled.
- `answers.json` — `answers[]`, each with `question_id`, `question`, `answer`, `citations` (**document filenames**) and
  `supported`. It also keeps `chunk_citations` (chunk IDs, the provenance used by validation), `refusal_reason`,
  `retrieved_source_ids` (chunk IDs), `retrieved_doc_ids`, `context_source_ids`, `evidence[]` and `diagnostics`.
- `validation_report.json` — per-question checks (keyed by `question_id`), summary counts, evaluation vs labels,
  settings, disclaimer.

Artefacts contain no timestamps, so reruns on the same inputs are byte-identical (diffable).
Exit code: 0 if all validation checks pass, 1 otherwise (`--strict` also fails on evaluation mismatches).

`app.py --question "..."` prints one JSON object: `question, answer, citations (filenames), supported,
retrieved_source_ids (chunk IDs), chunk_citations, retrieved_doc_ids, refusal_reason, retrieved[], evidence[], validation`.

## 10. Limitations (known and accepted)

- Lexical retrieval and matching: paraphrases outside the small lexicon that share no vocabulary with the
  docs ("old password" vs "last 5 passwords", "number of requests") are refused rather than answered.
- The `unknown_terms` rule refuses any question containing a word the docs never use and the lexicon cannot
  map, including harmless synonyms. Deliberate conservatism.
- The lexicon is hand-written and English-only. Each entry widens what can match, so new entries need the
  near-miss refusal tests re-run (`test_regressions.py`).
- Extractive answers are verbatim sentences: they can be terse, and cannot combine or compute over facts.
- Thresholds are calibrated on a small hand-written set plus ad-hoc probe questions; they are configuration,
  not learned values.
- No negation reasoning: "X is not accepted" scores like "X is accepted". Answers are verbatim, so the
  user sees the negation, but the system never turns it into a yes or no.
- Yes/no questions get the evidence sentence, not a "Yes"/"No" verdict.
- A 2-sentence window can add a loosely related neighbouring sentence (e.g. the status-page wording rule
  alongside the SEV2 cadence).
- Validation checks provenance/format and lexical overlap; it does not verify semantic entailment.
- Single-process, in-memory index rebuilt per run (fine for hundreds of docs, not millions).

## 11. Testing strategy

`pytest`, no network:
- **Chunk metadata** (`test_ingestion.py`): every chunk has id/source/section/hash; IDs unique and stable across
  runs; sources map to real files; table normalisation; chunk size bound; section boundaries respected.
- **Retrieval relevance** (`test_retrieval.py`): labelled answerable questions hit their expected source in
  top-k; synthetic corpus in `tmp_path` ranks the right doc first (proves no dependence on sample docs);
  deterministic ordering; empty-query behaviour.
- **Citation validation** (`test_validation.py`): valid answer passes; citation outside retrieved set fails;
  unknown citation fails; supported answer without citations fails; empty answer fails; refusal with citations
  fails; inline/list mismatch fails; validity of citations does not imply grounding.
- **Refusal** (`test_refusal.py`): labelled unsupported questions are refused; out-of-domain and stop-word-only
  questions are refused; answerable questions are supported with citations inside the retrieved set.
- **Review regressions** (`test_regressions.py`): output schema of all three artefacts; filename citations derived
  from chunk citations; both label vocabularies; paraphrases answered verbatim (negation and figures kept); SEV2
  answers exclude SEV1; numeric, identifier and sibling-specifier near-misses refused; lexicon replaceable.
- **End-to-end** (`test_pipeline.py`): `run_pipeline` on a replacement corpus + question file writes the three
  artefacts; `app.py` returns the required keys.
