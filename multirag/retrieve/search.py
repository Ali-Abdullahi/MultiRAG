"""Query the index and return pages a VLM can read.

Retrieval is where the two sides of the pipeline finally have to agree. Pages
were encoded by one model into one collection; a query encoded by any other
model lands in a different space and ranks nonsense without erroring. That
agreement is checked once when the retriever opens, not per query.
"""

from dataclasses import dataclass
from pathlib import Path

from multirag import config
from multirag.index import client as qc
from multirag.index import manifest as index_manifest


class RetrievalMismatchError(RuntimeError):
    """Raised when the query encoder does not match the indexed collection."""


@dataclass(frozen=True)
class Hit:
    """One retrieved page, ready to cite or to hand to a VLM."""

    score: float
    doc_id: str
    source_name: str
    page_index: int
    page_id: str
    image_path: Path

    @property
    def citation(self) -> str:
        """Human-readable provenance. Page numbers are 1-based for people."""
        return f"{self.source_name} p.{self.page_index + 1}"
