"""Central configuration.

Values here are baked into artifacts on disk. RENDER_DPI in particular is a
contract: embeddings are computed from images rendered at this DPI, so changing
it invalidates every vector already in the index. It is written into each
manifest so downstream stages can detect the mismatch instead of silently
retrieving against stale vectors.
"""

from pathlib import Path

# Repository root, resolved from this file so the package works from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PAGES_DIR = DATA_DIR / "pages"
MANIFESTS_DIR = DATA_DIR / "manifests"

# Rendering. 200 DPI puts a US-letter page at ~1700x2200px: comfortably legible
# for a vision-language model reading a dense table, without exploding disk use.
RENDER_DPI = 200

# Hard ceiling on the long edge. Guards against pathological page sizes (posters,
# CAD sheets, A0 plans) blowing up memory when rendered at RENDER_DPI.
MAX_PIXELS_LONG_EDGE = 3000

# PNG is lossless. JPEG artifacts land hardest on small text and thin table rules,
# which is exactly the content this pipeline exists to read.
PAGE_IMAGE_FORMAT = "png"
