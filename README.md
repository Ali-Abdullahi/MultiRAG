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
