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
