"""The single entry point for Phase 3: stored vectors in, indexed points out.

Indexing is the first stage that writes to shared state rather than to files it
owns. That shifts the failure model: earlier stages could recompute their way out
of a mismatch, but a populated collection belongs to every document in it, so a
conflict is refused and handed back to the caller instead of resolved silently.

Idempotent by construction. Point ids are the uuid5s from Phase 1, so re-indexing
a document overwrites its points rather than duplicating them.
"""

from qdrant_client import models

from multirag import config
from multirag.embed import manifest as embed_manifest
from multirag.embed import store
from multirag.index import client as qc
from multirag.index import manifest as index_manifest
from multirag.ingest import ids
from multirag.ingest import manifest as render_manifest_mod


class IndexConflictError(RuntimeError):
    """Raised when embeddings do not match the collection they would join."""


class NotEmbeddedError(RuntimeError):
    """Raised when a document has no complete set of vectors to index."""


def _payload(embed_page: dict, render_page: dict, render: dict) -> dict:
    """What a retrieved point needs to carry.

    Phase 5 hands page images to a VLM, so the image path travels with the point:
    a hit becomes a file to open without a second trip through the manifests.
    source_name and page_index are here to cite the answer - "page 3 of X.pdf" is
    the difference between an answer and a verifiable one.
    """
    return {
        "doc_id": render["doc_id"],
        "source_name": render["source_name"],
        "page_index": embed_page["page_index"],
        "page_id": embed_page["page_id"],
        "image": render_page["image"],
        "patches": embed_page["patches"],
    }


def _points(document_id: str, embedded: dict, render: dict) -> list[models.PointStruct]:
    """Build every point for one document.

    Vectors are loaded as fp32 and handed over as nested lists - Qdrant's client
    wants plain Python here, and the collection stores them back as fp16 anyway.
    """
    by_index = {page["page_index"]: page for page in render["pages"]}
    points = []
    for page in embedded["pages"]:
        index = page["page_index"]
        vectors = store.load_vectors(document_id, index)
        points.append(
            models.PointStruct(
                id=ids.page_point_id(page["page_id"]),
                vector=vectors.tolist(),
                payload=_payload(page, by_index[index], render),
            )
        )
    return points
