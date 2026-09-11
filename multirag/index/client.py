"""Embedded Qdrant: opening the store and shaping the collection.

Local mode, not a server. QdrantClient(path=...) holds an exclusive file lock on
the directory, so exactly one process may have it open. That is a real operational
constraint, not a detail: an indexing run and a retrieval run cannot overlap, and
a forgotten open client in a notebook will block the next one.
"""

from qdrant_client import QdrantClient, models

from multirag import config


def open_client() -> QdrantClient:
    """Open the embedded store. Caller is responsible for closing it."""
    config.QDRANT_PATH.parent.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(config.QDRANT_PATH))


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
