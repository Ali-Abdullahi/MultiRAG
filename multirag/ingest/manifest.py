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
