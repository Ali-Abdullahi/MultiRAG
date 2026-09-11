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


def index_document(document_id: str, *, client=None, force: bool = False, on_batch=None) -> dict:
    """Index one embedded document into the collection. Returns a summary.

    Refuses rather than repairs on three counts: vectors that do not exist, an
    incomplete embedding run, and embeddings that disagree with the collection's
    model. The first two are recoverable by running the earlier phase; the third
    is a decision only the caller can make.

    Pass a client to index a corpus under one open store - embedded Qdrant takes
    an exclusive lock, so opening one per document would serialize badly and
    fail outright if anything else holds it.
    """
    embedded = embed_manifest.read_manifest(document_id)
    if embedded is None:
        raise NotEmbeddedError(f"{document_id} has no embeddings; run embed_document first")
    if not embedded.get("complete"):
        raise NotEmbeddedError(
            f"{document_id} was embedded only partially ({embedded['page_count']} pages); "
            "finish embed_document before indexing"
        )

    render = render_manifest_mod.read_manifest(document_id)
    reason = embed_manifest.stale_reason(embedded, render_manifest=render)
    if reason:
        raise NotEmbeddedError(f"re-embed {document_id}: {reason}")

    owns_client = client is None
    client = client or qc.open_client()
    try:
        contract = index_manifest.read_manifest()
        if contract is None:
            contract = index_manifest.build_manifest(embedded)
            index_manifest.write_manifest(contract)
        else:
            conflict = index_manifest.mismatch_reason(contract, embedded)
            if conflict:
                raise IndexConflictError(
                    f"{document_id} does not belong in collection "
                    f"{config.QDRANT_COLLECTION!r}: {conflict}"
                )

        qc.ensure_collection(client, embedded["dim"])

        points = _points(document_id, embedded, render)
        batch_size = config.QDRANT_UPSERT_BATCH
        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            client.upsert(collection_name=config.QDRANT_COLLECTION, points=batch, wait=True)
            if on_batch:
                on_batch(min(start + len(batch), len(points)), len(points))

        return {
            "doc_id": document_id,
            "collection": config.QDRANT_COLLECTION,
            "points": len(points),
            "model_id": contract["model_id"],
            "dim": contract["dim"],
        }
    finally:
        if owns_client:
            client.close()
