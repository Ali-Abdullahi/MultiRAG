"""Rasterize PDF pages to images with PyMuPDF.

Deterministic and CPU-only: the same PDF always yields byte-identical images.
That is what lets the expensive embedding stage treat this output as a cache.
"""

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

from multirag import config
from multirag.ingest import ids

# PDF user-space units are 1/72 inch, so this is the denominator that turns a
# target DPI into a scale factor.
PDF_NATIVE_DPI = 72


@dataclass(frozen=True)
class RenderedPage:
    """One rasterized page and the facts Phase 2 needs about it."""

    page_index: int
    page_id: str
    image_path: Path
    width: int
    height: int
