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


def _zoom_for_page(page: fitz.Page) -> float:
    """Scale factor to render this page at RENDER_DPI, clamped for huge pages.

    DPI is relative to physical size, so an A0 plan at 200 DPI is ~62M pixels
    and ~186MB uncompressed. Oversized pages are scaled down to keep memory
    bounded; ordinary letter and A4 pages never hit the ceiling.
    """
    zoom = config.RENDER_DPI / PDF_NATIVE_DPI
    long_edge_points = max(page.rect.width, page.rect.height)
    if long_edge_points * zoom > config.MAX_PIXELS_LONG_EDGE:
        zoom = config.MAX_PIXELS_LONG_EDGE / long_edge_points
    return zoom
