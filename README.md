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
