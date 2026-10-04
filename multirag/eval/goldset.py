"""The gold set: questions paired with the pages that should answer them.

A gold set is written by a person, so it refers to pages the way a person does -
by filename and printed page number, the same form Hit.citation prints. Nothing
downstream can use that: retrieval returns content-addressed page_ids. Resolving
one to the other is this module's whole job, and it is deliberately strict.

A reference that does not resolve must raise. The alternative - treating an
unknown page as simply never retrieved - scores it zero, which is
indistinguishable from a retriever that got it wrong. That turns "the corpus
moved" into "the model regressed", which is the single most expensive way for an
eval harness to lie.
"""

from dataclasses import dataclass
from pathlib import Path

from multirag import config, jsonio

# Bump when the fixture layout changes incompatibly, so an old gold set is
# reported rather than half-read.
SCHEMA_VERSION = 1


class GoldSetError(ValueError):
    """Raised when a gold set is malformed or refers to pages that do not exist."""


@dataclass(frozen=True)
class PageRef:
    """One page as a human wrote it: a filename and a 1-based page number."""

    source_name: str
    page: int

    @property
    def citation(self) -> str:
        """Matches Hit.citation, so expected and retrieved print identically."""
        return f"{self.source_name} p.{self.page}"


@dataclass(frozen=True)
class GoldQuestion:
    """One question, with the pages a correct retrieval must surface.

    Carries both forms on purpose: page_ids are what metrics compare against,
    and refs are what a failure report has to print. A report that can only say
    "missed b58fc132ae32166f-p0000" is unreadable by the person who wrote it.
    """

    qid: str
    question: str
    relevant: frozenset[str]
    refs: tuple[PageRef, ...]


@dataclass(frozen=True)
class GoldSet:
    """A named collection of questions, loaded and fully resolved."""

    name: str
    questions: tuple[GoldQuestion, ...]

    def __len__(self) -> int:
        return len(self.questions)


def goldset_path(name: str) -> Path:
    """Location of a named gold set. One JSON file per set."""
    return config.EVALSETS_DIR / f"{name}.json"


@dataclass(frozen=True)
class _Corpus:
    """What is ingested, indexed by the two keys a gold set needs.

    page_ids maps (source_name, 0-based page_index) -> page_id. doc_ids maps a
    source_name to every doc_id carrying it, which is what makes ambiguity
    detectable: identity is the sha256 of the bytes, so the same PDF under two
    filenames collapses to one document - but two *different* PDFs under one
    filename do not. source_name is a label, not a key.
    """

    page_ids: dict[tuple[str, int], str]
    doc_ids: dict[str, list[str]]


def _load_corpus() -> _Corpus:
    """Read every ingest manifest into a lookup keyed by how people cite pages.

    Staleness is not checked here. Whether the images still match RENDER_DPI is
    the pipeline's question; this layer only needs to know which page_id a
    citation names. Sorted for determinism, so a duplicate filename reports the
    same pair of doc_ids on every run.
    """
    page_ids: dict[tuple[str, int], str] = {}
    doc_ids: dict[str, list[str]] = {}

    for path in sorted(config.MANIFESTS_DIR.glob("*.json")):
        manifest = jsonio.read_or_none(path)
        if manifest is None:
            continue
        source_name = manifest["source_name"]
        doc_ids.setdefault(source_name, []).append(manifest["doc_id"])
        for page in manifest["pages"]:
            page_ids[(source_name, page["page_index"])] = page["page_id"]

    return _Corpus(page_ids=page_ids, doc_ids=doc_ids)


def _resolve(ref: PageRef, corpus: _Corpus, qid: str) -> str:
    """Turn one human page reference into a page_id, or explain why it cannot be.

    Three distinct failures, each with a different remedy, so each says which:
    the document was never ingested, the filename is ambiguous across documents,
    or the page number is outside the document.
    """
    candidates = corpus.doc_ids.get(ref.source_name)
    if not candidates:
        known = ", ".join(sorted(corpus.doc_ids)) or "nothing ingested"
        raise GoldSetError(
            f"{qid}: {ref.citation} names a document that is not ingested "
            f"(known: {known})"
        )
    if len(candidates) > 1:
        raise GoldSetError(
            f"{qid}: {ref.citation} is ambiguous - {len(candidates)} documents "
            f"share that filename ({', '.join(candidates)}); two different PDFs "
            "cannot be told apart by name, so rename one and re-ingest"
        )

    page_id = corpus.page_ids.get((ref.source_name, ref.page - 1))
    if page_id is None:
        pages = sum(1 for name, _ in corpus.page_ids if name == ref.source_name)
        raise GoldSetError(
            f"{qid}: {ref.citation} is out of range - that document has "
            f"{pages} page(s). Page numbers in a gold set are 1-based."
        )
    return page_id
