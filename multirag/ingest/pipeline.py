"""The single entry point for Phase 1: a PDF path in, a manifest out.

Callers should use ingest_pdf rather than driving render and manifest by hand,
so the skip-and-refresh decision lives in exactly one place.
"""

from pathlib import Path

from multirag.ingest import ids, manifest
from multirag.ingest.render import render_pdf
