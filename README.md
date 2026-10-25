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

The model must accept images. Example settings for Gemini (Google AI Studio offers a
free tier):

| Field | Value |
|---|---|
| Base URL | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| Model | the bare id, e.g. `gemini-…-flash-lite` - not `models/gemini-…` |
| API key | from Google AI Studio |

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

Gold sets built from private documents quote them; keep those out of version
control (see `.gitignore`).
