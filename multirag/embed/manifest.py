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
