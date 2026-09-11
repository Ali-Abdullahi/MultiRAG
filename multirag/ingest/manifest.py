"""Per-document manifests: the metadata record for an ingested PDF.

The manifest is the contract between ingestion and everything downstream. It
records the settings the images were produced under, so a later stage can refuse
to run against stale artifacts instead of silently embedding mismatched pages.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from multirag import config
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
