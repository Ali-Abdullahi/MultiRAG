"""Embedded Qdrant: opening the store and shaping the collection.

Local mode, not a server. QdrantClient(path=...) holds an exclusive file lock on
the directory, so exactly one process may have it open. That is a real operational
constraint, not a detail: an indexing run and a retrieval run cannot overlap, and
a forgotten open client in a notebook will block the next one.
"""

from qdrant_client import QdrantClient, models

from multirag import config


class StoreLockedError(RuntimeError):
    """Raised when another process already holds the embedded store's lock."""


def open_client() -> QdrantClient:
    """Open the embedded store. Caller is responsible for closing it.

    qdrant-client signals a held lock with a bare RuntimeError, so the message is
    the only way to tell it apart. It is re-raised as a named error because this
    is the one failure a user can fix in seconds, and they should be told how.
    """
    config.QDRANT_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        return QdrantClient(path=str(config.QDRANT_PATH))
    except RuntimeError as error:
        if "already accessed" not in str(error):
            raise
        raise StoreLockedError(
            f"{config.QDRANT_PATH} is in use by another process - most likely the "
            "Streamlit app. Stop it (Ctrl-C) and retry."
        ) from error


def _vector_params(dim: int) -> models.VectorParams:
    """Collection geometry for late-interaction page vectors.

    MAX_SIM is the whole point: Qdrant scores a query matrix against a page
    matrix by taking, for each query row, its best-matching page row and summing
    those maxima. Without multivector_config a collection expects one vector per
    point and a page would have to be pooled into one - discarding exactly the
    per-patch detail this architecture exists to keep.

    HNSW is disabled (m=0), which makes search exact rather than approximate.
    A page is ~1139 vectors, so building a proximity graph over a corpus of them
    costs far more memory than brute force saves at this scale, and the answers
    are exact. The production path for a large corpus is two-stage - a cheap
    pooled prefetch, then MAX_SIM rerank on the survivors - which is an
    optimization to make when a corpus actually demands it, not before.
    """
    return models.VectorParams(
        size=dim,
        distance=models.Distance(config.QDRANT_DISTANCE),
        multivector_config=models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM
        ),
        datatype=models.Datatype(config.QDRANT_DATATYPE),
        hnsw_config=models.HnswConfigDiff(m=0),
        on_disk=True,
    )


def ensure_collection(client: QdrantClient, dim: int) -> bool:
    """Create the collection if absent. Returns True if it was created.

    Never recreates an existing collection: that would silently drop every
    indexed page. A collection built for the wrong model is caught upstream by
    the index manifest, which can then say so instead of destroying data.
    """
    if client.collection_exists(config.QDRANT_COLLECTION):
        return False
    client.create_collection(
        collection_name=config.QDRANT_COLLECTION,
        vectors_config=_vector_params(dim),
    )
    return True


def collection_dim(client: QdrantClient) -> int | None:
    """The vector width the collection was built for, or None if it is absent."""
    if not client.collection_exists(config.QDRANT_COLLECTION):
        return None
    info = client.get_collection(config.QDRANT_COLLECTION)
    return info.config.params.vectors.size


def collection_size(client: QdrantClient) -> int:
    """Number of indexed pages, or 0 if the collection is absent.

    Exact rather than estimated: the caller is eval, which uses this to decide
    whether a metric is meaningful, and an approximate corpus size would make
    that decision approximate too.
    """
    if not client.collection_exists(config.QDRANT_COLLECTION):
        return 0
    return client.count(config.QDRANT_COLLECTION, exact=True).count
