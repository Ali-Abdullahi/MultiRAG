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
    # How many vectors the query encoded to. Defaulted so a Hit built outside
    # search() (tests, fixtures) stays valid; match is then unknown.
    query_tokens: int = 0

    @property
    def match(self) -> float | None:
        """Score per query token: the mean best-patch similarity.

        The raw MaxSim score is a sum over query tokens, so it grows with question
        length and cannot be compared across questions. Dividing by the token
        count gives a number that can - measured on a 208-page course corpus, it
        separated answerable from unanswerable questions where the raw score
        overlapped (raw 10.9-20.5 vs 8.1-11.8; per-token 0.53-0.78 vs 0.42-0.55).
        """
        return self.score / self.query_tokens if self.query_tokens else None

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

        # Store first, encoder second: a held lock is the common failure and
        # costs milliseconds to detect, so it must not wait behind an 8s load.
        client = qc.open_client()
        try:
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
        except BaseException:
            # Opened before anything that can fail, so release it on any failure:
            # a leaked client keeps the lock until this process exits.
            client.close()
            raise

        return cls(encoder, client)

    def close(self) -> None:
        """Release the Qdrant file lock."""
        self._client.close()

    def __enter__(self) -> "Retriever":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    @property
    def page_count(self) -> int:
        """How many pages are indexed.

        Exposed because this client holds the store's exclusive lock, so a
        caller cannot open a second one to ask. Eval needs it to know whether
        its own numbers mean anything: when the whole corpus fits inside the
        result list, recall@k is 1.0 by construction, not by merit.
        """
        return qc.collection_size(self._client)

    def search(self, query: str, limit: int | None = None) -> list[Hit]:
        """Rank pages against one query, best first.

        Queries go through encode_queries, never encode_pages: the model is
        trained with a different prefix per side, and crossing them silently
        degrades every result.

        Scores are MaxSim sums and are comparable *within* one query only - they
        scale with query token count, so a longer question scores higher against
        the same page. Rank them; do not threshold them.
        """
        [vectors] = self._encoder.encode_queries([query])
        response = self._client.query_points(
            collection_name=config.QDRANT_COLLECTION,
            query=vectors.tolist(),
            limit=limit or config.RETRIEVE_TOP_K,
            with_payload=True,
        )
        return [
            Hit(
                score=point.score,
                doc_id=point.payload["doc_id"],
                source_name=point.payload["source_name"],
                page_index=point.payload["page_index"],
                page_id=point.payload["page_id"],
                image_path=config.DATA_DIR / point.payload["image"],
                query_tokens=int(vectors.shape[0]),
            )
            for point in response.points
        ]
