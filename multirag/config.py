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

# --- Embedding (Phase 2) ---------------------------------------------------

EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# ColPali-family encoder. colSmol-256M is the deliberate default: ColPali
# (PaliGemma-3B) and ColQwen2.5 are ~6GB in fp16 and will thrash or OOM on an
# 8GB machine. Swap this for a larger checkpoint on bigger hardware or Colab.
EMBED_MODEL_ID = "vidore/colSmol-256M"

# Pin to a specific commit once verified. "main" is a moving target: the same
# model id can silently change weights, which would make old vectors and new
# queries land in different spaces. Resolved value is recorded per document.
EMBED_MODEL_REVISION = "main"

# Late-interaction models emit one vector per image patch, so a single page is
# already a batch-sized workload. Raise only if memory headroom proves it safe.
EMBED_BATCH_SIZE = 1

# Device for the encoder. "auto" picks CUDA when present and CPU otherwise.
#
# MPS is deliberately NOT auto-selected. Measured on this M1 with torch 2.13.0
# and colSmol-256M: bfloat16 spun for 38 minutes of CPU across 3 cores with a
# 9.5MB resident set and never finished loading, and float32 segfaulted (139)
# partway through the weights. CPU loads in seconds at under 1GB. Set this to
# "mps" explicitly to retry on a newer torch - the failure is in the backend,
# not the model, so it is worth rechecking after an upgrade.
EMBED_DEVICE = "auto"
