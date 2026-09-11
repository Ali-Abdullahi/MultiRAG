"""Central configuration.

Values here are baked into artifacts on disk. RENDER_DPI in particular is a
contract: embeddings are computed from images rendered at this DPI, so changing
it invalidates every vector already in the index. It is written into each
manifest so downstream stages can detect the mismatch instead of silently
retrieving against stale vectors.
"""

import os
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

# --- Index (Phase 3) -------------------------------------------------------

# Embedded Qdrant: a directory, not a server. One less moving part to run, and
# the storage is content the rest of data/ already treats as regenerable.
QDRANT_PATH = DATA_DIR / "qdrant"
QDRANT_COLLECTION = "pages"

# Vectors leave the encoder L2-normalized, so cosine and dot are equivalent here.
# Cosine is the safer declaration: it stays correct if a future encoder emits
# unnormalized vectors, where dot would silently start ranking by magnitude.
QDRANT_DISTANCE = "Cosine"

# Storing the index as float16 mirrors how vectors are already kept on disk, and
# halves what a multivector collection holds in memory. A page is ~1139 vectors,
# so a 10k-page corpus is ~11M vectors: the factor of two is not cosmetic on 8GB.
QDRANT_DATATYPE = "float16"

# Pages per upsert. A page carries ~1139 x 128 floats, so a batch of 8 moves
# roughly 4.7MB - large enough to amortize round trips, small enough that a
# failed batch is cheap to retry.
QDRANT_UPSERT_BATCH = 8

# --- Retrieval (Phase 4) ---------------------------------------------------

# Pages returned per query. Small on purpose: these go to a VLM as images, and
# every extra page is real tokens and real latency in Phase 5.
RETRIEVE_TOP_K = 5

# --- Generation (Phase 5) --------------------------------------------------

# Long edge, in pixels, for images sent to a vision model. Pages are rendered at
# 200 DPI (~2200px) for retrieval quality, but most hosted vision models downscale
# above ~1568px anyway. Sending the full-size page spends upload bandwidth and
# image tokens on detail the model discards. Retrieval keeps the big version.
VLM_MAX_IMAGE_EDGE = 1568

# Backend name and model, both overridable by environment so a clone can point
# at its own provider without editing code. No credentials live in config.
GENERATOR_BACKEND = os.environ.get("MULTIRAG_GENERATOR", "claude")
GENERATOR_MODEL = os.environ.get("MULTIRAG_GENERATOR_MODEL", "claude-opus-5")
