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
