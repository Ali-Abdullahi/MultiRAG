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
