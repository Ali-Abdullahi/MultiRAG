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


class Retriever:
    """An open encoder and an open index, checked to agree with each other.

    Both are expensive: the encoder costs ~8s to load and the embedded Qdrant
    store takes an exclusive file lock. So a Retriever is built once and reused,
    and closed when done - a leaked one blocks every later run.

    Use as a context manager to guarantee that.
    """

    def __init__(self, encoder, client):
        self._encoder = encoder
        self._client = client

    @classmethod
    def open(cls, encoder=None) -> "Retriever":
        """Load the encoder, open the index, and verify they match.

        The check happens here rather than per query so a mismatch surfaces at
        startup, when it is cheap to act on, instead of as quietly wrong results.
        """
        contract = index_manifest.read_manifest()
        if contract is None:
            raise RetrievalMismatchError("nothing indexed yet; run index_document first")

        if encoder is None:
            from multirag.embed.encoder import PageEncoder

            encoder = PageEncoder.load()

        info = encoder.info
        for key, want in (
            ("model_id", info.model_id),
            ("model_revision", info.revision),
            ("dim", info.dim),
        ):
            got = contract.get(key)
            if got != want:
                raise RetrievalMismatchError(
                    f"index was built with {key} {got!r}, querying with {want!r}: "
                    "re-index, or point at the collection that matches"
                )

        return cls(encoder, qc.open_client())

    def close(self) -> None:
        """Release the Qdrant file lock."""
        self._client.close()

    def __enter__(self) -> "Retriever":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
