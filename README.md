# MultiRAG

Retrieval-augmented generation over PDFs **as images**. Pages are never parsed into
text: each one is rendered whole, so tables, figures, equations and layout survive into
retrieval instead of being flattened by a text extractor. The retrieved page images go
straight to a vision model, which answers with page-level citations.

## How it works

```
PDF ──► ingest ──► embed ──► index ──► retrieve ──► generate ──► answer + citations
       PNG/page   ColPali    Qdrant    MaxSim       any vision
       @200 DPI   multi-     on disk   top-k pages  model
                  vectors
```

| Stage | What happens | Code |
|---|---|---|
| Ingest | PyMuPDF renders every page to a 200-DPI PNG, plus a manifest | `multirag/ingest/` |
| Embed | A ColPali-family encoder turns each page into ~1,000 patch vectors (128-d) | `multirag/embed/` |
| Index | Qdrant (embedded, on disk) stores each page's matrix with late-interaction MaxSim | `multirag/index/` |
| Retrieve | The question is encoded per token; each token takes its best-matching patch on a page, and those maxima are summed into the page's score | `multirag/retrieve/` |
| Generate | The top pages are downscaled and sent with the question to a vision model | `multirag/generate/` |
| UI | Streamlit app over retrieve + generate | `multirag/ui/` |
| Eval | Scores retrieval against hand-checked questions | `multirag/eval/` |

## Setup

Python 3.11+.

```bash
git clone git@github.com:Ali-Abdullahi/MultiRAG.git
cd MultiRAG
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # makes `multirag` importable from any launcher
```

The encoder downloads from Hugging Face the first time a page is embedded.

## Usage

**1. Add documents** - files or whole folders (searched recursively):

```bash
python -m multirag add paper.pdf ~/Documents/course-notes/
```

Embedding runs at about 8 s per page on an Apple M1 CPU, so a 200-page folder takes
around half an hour. Every stage is resumable: re-run the same command after an
interruption and only the missing work is done. On a Mac, prefix long runs with
`caffeinate -i` - a sleeping laptop pauses the job.

**2. Ask questions** in the app:

```bash
streamlit run multirag/ui/app.py
```

Retrieval is free and local. Generation runs only when you press **Generate**, using
the backend chosen in the sidebar. The `echo` backend calls no model and shows which
pages would be sent - the quickest way to check retrieval.

**3. Measure retrieval** against a gold set (see [Evaluation](#evaluation)):

```bash
python -m multirag eval <name>
```

> Embedded Qdrant allows **one process at a time**. Stop the Streamlit app before
> running `add` or `eval`; the CLI says so if you forget.

## Answer backends

Pick one in the app's sidebar. Bring your own key - nothing is bundled.

| Backend | Use for | Needs |
|---|---|---|
| `echo` | Checking retrieval; no model is called | nothing |
| `claude` | Anthropic's Claude | `pip install anthropic` and an Anthropic API key |
| `openai-compatible` | OpenAI, Gemini, Groq, Mistral, OpenRouter, a local Ollama, ... | base URL + model id + key |

The model must accept images. Example `openai-compatible` settings:

| Field | Gemini (Google AI Studio has a free tier) | Claude, without installing `anthropic` |
|---|---|---|
| Base URL | `https://generativelanguage.googleapis.com/v1beta/openai/` | `https://api.anthropic.com/v1/` |
| Model | the bare id, e.g. `gemini-…-flash-lite` - not `models/gemini-…` | e.g. `claude-opus-5-5` |
| API key | from Google AI Studio | from console.anthropic.com |

Anthropic describes its OpenAI-compatible endpoint mainly for testing and comparison;
the `claude` backend uses Anthropic's own SDK and supports Claude's full feature set.

Free tiers can be congested: if requests time out or report "provider busy", try a
`lite` variant or raise the timeout in the sidebar.

**Keys stay with you.** A key typed into the sidebar is held only in that browser
session's memory - never written to disk, never logged, never shared across sessions,
and redacted from any error shown. Alternatively set `ANTHROPIC_API_KEY` or
`OPENAI_API_KEY` in your environment (see `.env.example`); `.env` is gitignored.
The app listens on `localhost` only, so it is not reachable from your network.

A chat subscription (Claude Pro, Google AI Pro) does not include API usage; API
calls are billed separately by each provider.

To add a provider that is not OpenAI-compatible: write one class with an
`answer(question, hits)` method (see `multirag/generate/base.py`) and register it in
`multirag/generate/factory.py`. Nothing else needs to know about it.

## Evaluation

A gold set is a JSON file in `evalsets/` pairing questions with the page(s) that
answer them. Pages are cited the way a person would - filename and 1-based page:

```json
{
  "schema_version": 1,
  "questions": [
    {
      "id": "late-policy",
      "question": "What happens if I submit the report late?",
      "relevant": [{ "source_name": "Syllabus.pdf", "page": 2 }]
    }
  ]
}
```

List several pages when more than one is a correct answer. Write questions the way
you would actually ask them, not by copying the page's wording - that measures string
overlap, not retrieval. A reference to a page that is not indexed fails at load time
rather than silently scoring zero.

`python -m multirag eval <name>` prints a line per question - with what ranked first
whenever the expected page did not - then:

| Metric | Meaning |
|---|---|
| top-1 | right page ranked first |
| hit@k | a right page anywhere in the top k (what the vision model actually receives) |
| recall@k | share of all right pages found in the top k |
| MRR | how near the top the first right page lands |

If the corpus has no more pages than k, every page is always returned and hit@k is
meaningless; the report warns when that happens.

For reference: on a 208-page lecture-and-homework corpus with 35 hand-checked
questions, the default `colSmol-256M` encoder scored top-1 0.600, hit@5 0.886,
MRR 0.718. Most misses were dense homework or exam pages outranking the slide that
explained the concept.

**Unanswerable questions.** Mark a question `"answerable": false` (with no
`relevant` pages) when the documents deliberately cannot answer it. These stay out
of the ranking metrics and instead check the app's weak-match warning: each run
reports how many unanswerable questions it fired on, and how many answerable ones
it wrongly fired on (which should be 0).

**Saved runs.** Every `eval` is saved under `data/evals/<name>/`. Compare them:

```bash
python -m multirag runs <name>             # numbered list with headline metrics
python -m multirag compare <name> [old new] # default: the last two runs
```

`compare` shows metric deltas and per-question rank changes. A different encoder is
reported as what changed; an edited gold set, a different k, or a different corpus
size is flagged as a warning, because each moves the numbers for reasons that have
nothing to do with retrieval quality.

Gold sets and saved runs built from private documents quote them; keep them out of
version control (`data/` is already ignored, and so is every `evalsets/*.json` except
files named `*.public.json`).

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite runs in about 10 seconds with no model, no network and no API key: it
uses a fake encoder, a local stand-in for the provider, and a temporary data
directory, so it never touches your documents or index. It covers content
identity, idempotent ingest and indexing, every staleness check, crash
resumption, Qdrant's MaxSim against an independent numpy computation, the store
lock, provider error handling, and that API keys never appear in anything shown.

## Design notes

- **Identity is content, not filename.** A document's id is a prefix of its file's
  sha256, so the same PDF under two names is one document. Each page's Qdrant id is a
  deterministic UUID derived from that, so re-indexing overwrites instead of
  duplicating.
- **Every stage records what it was built from, and checks it.** Images record the
  render settings; vectors record the encoder and a fingerprint of those settings; the
  collection records which encoder it belongs to. Changing the DPI or the model is
  detected and reported rather than silently mixing incompatible data - mixing two
  encoders' vectors raises no error, it just ranks nonsense.
- **Writes are atomic** (temp file, fsync, rename), so a crash never leaves a
  half-written manifest or vector file behind.
- **Search is exact.** HNSW is disabled: at this scale, brute-force MaxSim over every
  page is cheap and never misses. Exact still means semantic - vectors are compared,
  not words.
- **Pages are rendered as PNG**, not JPEG: compression artifacts hurt small text and
  thin table lines most, which is what this pipeline exists to read.

## Hardware

The default encoder is `vidore/colSmol-256M`, chosen to run on an 8 GB Apple M1 on the
CPU. Larger ColPali-family models (ColPali, ColQwen) need roughly 6 GB in half
precision; set `EMBED_MODEL_ID` in `multirag/config.py` on bigger hardware. Changing
the model means re-embedding and re-indexing - the pipeline refuses to mix them.

On torch 2.13, Apple's MPS backend hung or crashed loading this model, so the encoder
uses the CPU unless `EMBED_DEVICE` is set explicitly.

## Limitations

- Retrieval always returns the top k pages, even for a question the documents cannot
  answer. The app warns when the best page is a weak match (score per query token
  below `WEAK_MATCH_THRESHOLD`), and the vision model is instructed to say so rather
  than guess. The threshold is calibrated per encoder - re-check it with `eval`
  after changing models.
- One process can open the index at a time (embedded Qdrant takes a file lock).
- The `claude` backend has not yet been exercised against the live API; the
  `openai-compatible` backend has (via Gemini).
