"""Per-document embedding manifests: what was embedded, by which model.

The second link in the staleness chain. The render manifest records the settings
the images were made under; this one records the model that turned those images
into vectors, *and* the render manifest it was built against.

Both links matter. New weights invalidate the vectors. So does a re-render at a
different DPI, even with the model untouched - the pixels the vectors describe
no longer exist. Neither failure raises an exception on its own; both quietly
degrade retrieval, which is why they are checked rather than assumed.
"""

from datetime import datetime, timezone
from pathlib import Path

from multirag import config, jsonio
from multirag.embed import store
from multirag.embed.identity import EncoderInfo, resolved_revision

# Bump when this layout changes incompatibly.
SCHEMA_VERSION = 1


# The render settings that determine the pixels. Rendering is deterministic, so
# identical settings mean byte-identical images: a re-render under the same
# config leaves existing vectors valid. That is why the fingerprint compares
# settings rather than timestamps - a fresh created_at is not a reason to
# recompute 8 seconds of work per page.
RENDER_FINGERPRINT_KEYS = (
    "schema_version",
    "render_dpi",
    "max_pixels_long_edge",
    "image_format",
)


def render_fingerprint(render_manifest: dict) -> dict:
    """The part of a render manifest that decides whether vectors are still valid."""
    return {key: render_manifest.get(key) for key in RENDER_FINGERPRINT_KEYS}


def page_record(page_index: int, page_id: str, patches: int, document_id: str) -> dict:
    """Serialize one embedded page. Paths stay relative to DATA_DIR, as in Phase 1."""
    path = store.vector_path(document_id, page_index)
    return {
        "page_index": page_index,
        "page_id": page_id,
        "vectors": str(path.relative_to(config.DATA_DIR)),
        "patches": patches,
    }


def build_manifest(
    render_manifest: dict,
    info: EncoderInfo,
    pages: list[dict],
    *,
    complete: bool = True,
) -> dict:
    """Assemble the embedding manifest for one document.

    Carries the encoder's identity and the render fingerprint it was built
    against, so both links in the chain can be checked later.

    complete=False marks a checkpoint written mid-run: the pages listed are
    genuinely done, but the document is not. A consumer that indexes only
    complete documents can tell the difference.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "complete": complete,
        "doc_id": render_manifest["doc_id"],
        "model_id": info.model_id,
        "model_revision": info.revision,
        "dim": info.dim,
        "storage_dtype": str(store.STORAGE_DTYPE.__name__),
        # Provenance only: which box produced this. Never checked for staleness -
        # cpu and cuda give numerically different vectors, but not differently
        # *meaningful* ones, and re-embedding a corpus to change device is absurd.
        "device": info.device,
        "compute_dtype": info.dtype,
        "render": render_fingerprint(render_manifest),
        "page_count": len(pages),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pages": pages,
    }


def manifest_path(document_id: str) -> Path:
    """Location of a document's embedding manifest.

    Co-located with the vectors it describes, unlike the render manifest which
    sits in its own directory. Deleting data/embeddings/<doc_id> then removes
    the manifest and the vectors together - there is no way to end up with a
    manifest promising vectors that are no longer on disk.
    """
    return store.document_dir(document_id) / "manifest.json"


def write_manifest(manifest: dict) -> Path:
    """Write an embedding manifest atomically. Returns the path written."""
    return jsonio.write_atomic(manifest_path(manifest["doc_id"]), manifest)


def read_manifest(document_id: str) -> dict | None:
    """Load an embedding manifest, or None if the document was never embedded."""
    return jsonio.read_or_none(manifest_path(document_id))


def stale_reason(
    manifest: dict,
    *,
    render_manifest: dict | None = None,
    info: EncoderInfo | None = None,
) -> str | None:
    """Explain why these vectors no longer match, or None if they still do.

    Checks both links. Pass render_manifest to check the images the vectors were
    built from; pass info to check against a loaded encoder. With neither, the
    model is checked against config - which resolves the revision from the local
    HF cache, so the cheap check needs no model load and no network.

    Schema first: an older layout may not have the other keys at all.
    """
    found = manifest.get("schema_version")
    if found != SCHEMA_VERSION:
        return f"schema_version {found} != {SCHEMA_VERSION}"

    if render_manifest is not None:
        want = render_fingerprint(render_manifest)
        got = manifest.get("render")
        if got != want:
            return f"pages were re-rendered: {got} != {want}"

    if info is None:
        model_id = config.EMBED_MODEL_ID
        revision = resolved_revision(model_id, config.EMBED_MODEL_REVISION)
        dim = manifest.get("dim")
    else:
        model_id, revision, dim = info.model_id, info.revision, info.dim

    for key, want in (("model_id", model_id), ("model_revision", revision), ("dim", dim)):
        got = manifest.get(key)
        if got != want:
            return f"{key} {got!r} != {want!r}"

    return None
