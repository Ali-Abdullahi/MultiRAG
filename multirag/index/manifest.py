"""The collection contract: which model this Qdrant collection belongs to.

Third link in the staleness chain. A Qdrant collection has no place to record
what produced its vectors, so it is recorded beside it.

This guard matters more than the earlier two. Mixing vectors from two models in
one collection does not fail - MaxSim happily scores them - it just ranks
nonsense, and nothing in the index would ever reveal why. Deliberately no
per-document state here: Qdrant is the single source of truth for what is
indexed, and a second copy of that would only drift.
"""

from pathlib import Path

from multirag import config, jsonio

SCHEMA_VERSION = 1


def manifest_path() -> Path:
    """One contract per store, beside the Qdrant directory it describes."""
    return config.DATA_DIR / "index.json"


def build_manifest(embed_manifest: dict) -> dict:
    """Derive the collection contract from the embeddings that seeded it."""
    return {
        "schema_version": SCHEMA_VERSION,
        "collection": config.QDRANT_COLLECTION,
        "model_id": embed_manifest["model_id"],
        "model_revision": embed_manifest["model_revision"],
        "dim": embed_manifest["dim"],
        "distance": config.QDRANT_DISTANCE,
        "datatype": config.QDRANT_DATATYPE,
    }


def read_manifest() -> dict | None:
    """Load the collection contract, or None if nothing has been indexed yet."""
    return jsonio.read_or_none(manifest_path())


def write_manifest(manifest: dict) -> Path:
    """Write the collection contract atomically."""
    return jsonio.write_atomic(manifest_path(), manifest)


def mismatch_reason(manifest: dict, embed_manifest: dict) -> str | None:
    """Explain why these embeddings do not belong in this collection.

    Not called "stale": the earlier manifests describe artifacts that can simply
    be rebuilt, but a populated collection is shared state. A mismatch here means
    the caller must choose - rebuild the collection, or point at a different one -
    so the answer is a reason to refuse, not a trigger to recompute.
    """
    found = manifest.get("schema_version")
    if found != SCHEMA_VERSION:
        return f"schema_version {found} != {SCHEMA_VERSION}"

    expected = {
        "model_id": embed_manifest["model_id"],
        "model_revision": embed_manifest["model_revision"],
        "dim": embed_manifest["dim"],
        "collection": config.QDRANT_COLLECTION,
    }
    for key, want in expected.items():
        got = manifest.get(key)
        if got != want:
            return f"{key} {got!r} != {want!r}"

    return None
