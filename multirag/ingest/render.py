"""Rasterize PDF pages to images with PyMuPDF.

Deterministic and CPU-only: the same PDF always yields byte-identical images.
That is what lets the expensive embedding stage treat this output as a cache.
"""

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

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


def _render_one(
    page: fitz.Page, page_index: int, document_id: str, out_dir: Path, force: bool
) -> RenderedPage:
    """Rasterize a single page, reusing the file on disk unless force is set."""
    image_path = out_dir / f"page_{page_index:04d}.{config.PAGE_IMAGE_FORMAT}"

    if image_path.exists() and not force:
        # Resumable: a run that died midway skips everything already written.
        # PIL reads only the header here, not the pixel data.
        with Image.open(image_path) as existing:
            width, height = existing.size
    else:
        zoom = _zoom_for_page(page)
        # alpha=False drops the unused transparency channel: 25% less memory
        # per page, smaller files, and RGB is what the encoders expect anyway.
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        pixmap.save(image_path)
        width, height = pixmap.width, pixmap.height

    return RenderedPage(
        page_index=page_index,
        page_id=ids.page_id(document_id, page_index),
        image_path=image_path,
        width=width,
        height=height,
    )


def render_pdf(pdf_path: Path, *, force: bool = False) -> tuple[str, list[RenderedPage]]:
    """Rasterize every page of a PDF. Returns its document id and page records.

    Safe to call repeatedly: the document id is content-derived and existing
    images are reused, so a second call over the same PDF is close to free.
    """
    pdf_path = Path(pdf_path)
    document_id = ids.doc_id(pdf_path)
    out_dir = config.PAGES_DIR / document_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with fitz.open(pdf_path) as document:
        return document_id, [
            _render_one(page, index, document_id, out_dir, force)
            for index, page in enumerate(document)
        ]
