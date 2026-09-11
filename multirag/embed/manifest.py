"""Per-document embedding manifests: what was embedded, by which model.

The second link in the staleness chain. The render manifest records the settings
the images were made under; this one records the model that turned those images
into vectors, *and* the render manifest it was built against.

Both links matter. New weights invalidate the vectors. So does a re-render at a
different DPI, even with the model untouched - the pixels the vectors describe
no longer exist. Neither failure raises an exception on its own; both quietly
degrade retrieval, which is why they are checked rather than assumed.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from multirag import config
from multirag.embed import store
from multirag.embed.encoder import EncoderInfo

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


def _page_record(page_index: int, page_id: str, patches: int, document_id: str) -> dict:
    """Serialize one embedded page. Paths stay relative to DATA_DIR, as in Phase 1."""
    path = store.vector_path(document_id, page_index)
    return {
        "page_index": page_index,
        "page_id": page_id,
        "vectors": str(path.relative_to(config.DATA_DIR)),
        "patches": patches,
    }
