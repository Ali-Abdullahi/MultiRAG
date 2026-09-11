"""Per-document manifests: the metadata record for an ingested PDF.

The manifest is the contract between ingestion and everything downstream. It
records the settings the images were produced under, so a later stage can refuse
to run against stale artifacts instead of silently embedding mismatched pages.
"""

from datetime import datetime, timezone
from pathlib import Path

from multirag import config, jsonio
from multirag.ingest.render import RenderedPage

# Bump when the manifest layout changes incompatibly. Old manifests are then
# detectable rather than crashing on a missing key deep in a later stage.
SCHEMA_VERSION = 1


def _page_record(page: RenderedPage) -> dict:
    """Serialize one rendered page.

    The image path is stored relative to DATA_DIR, never absolute. An absolute
    path bakes in this machine's home directory and breaks the moment the same
    data is read from a clone, a container, or Colab.
    """
    return {
        "page_index": page.page_index,
        "page_id": page.page_id,
        "image": str(page.image_path.relative_to(config.DATA_DIR)),
        "width": page.width,
        "height": page.height,
    }


def build_manifest(
    document_id: str, pdf_path: Path, pages: list[RenderedPage]
) -> dict:
    """Assemble the manifest for one ingested document.

    Only the source *name* is kept, not its path: the bytes are the identity,
    the filename is a human label. Every setting that shaped the images is
    recorded alongside them so Phase 2 can compare rather than assume.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "doc_id": document_id,
        "source_name": Path(pdf_path).name,
        "page_count": len(pages),
        "render_dpi": config.RENDER_DPI,
        "max_pixels_long_edge": config.MAX_PIXELS_LONG_EDGE,
        "image_format": config.PAGE_IMAGE_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pages": [_page_record(page) for page in pages],
    }


def manifest_path(document_id: str) -> Path:
    """Location of a document's manifest. One file per document."""
    return config.MANIFESTS_DIR / f"{document_id}.json"


def write_manifest(manifest: dict) -> Path:
    """Write a manifest atomically. Returns the path written."""
    return jsonio.write_atomic(manifest_path(manifest["doc_id"]), manifest)


def read_manifest(document_id: str) -> dict | None:
    """Load a manifest, or None if the document has never been ingested."""
    return jsonio.read_or_none(manifest_path(document_id))


def page_image_path(page_record: dict) -> Path:
    """Resolve a manifest page record back to an absolute image path.

    The inverse of the relative path stored by _page_record, and the accessor
    Phase 2 should use rather than rebuilding the path convention itself.
    """
    return config.DATA_DIR / page_record["image"]


def stale_reason(manifest: dict) -> str | None:
    """Explain why a manifest no longer matches current config, or None if it does.

    This is the check the recorded settings exist for. Embeddings are a function
    of the pixels, so a manifest rendered under different settings describes
    images that no longer correspond to the vectors in the index. Returning the
    reason rather than a bare bool means the caller can say what to re-run.

    Schema is checked first: an older layout may not even have the other keys.
    """
    found = manifest.get("schema_version")
    if found != SCHEMA_VERSION:
        return f"schema_version {found} != {SCHEMA_VERSION}"

    expected = {
        "render_dpi": config.RENDER_DPI,
        "max_pixels_long_edge": config.MAX_PIXELS_LONG_EDGE,
        "image_format": config.PAGE_IMAGE_FORMAT,
    }
    for key, want in expected.items():
        got = manifest.get(key)
        if got != want:
            return f"{key} {got!r} != {want!r}"

    return None
