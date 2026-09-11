"""Turning retrieved pages into model input, without naming a provider.

Everything here is shared by every backend: loading page images, downscaling
them, and writing the instruction. Only the final wire format differs per
provider, and that belongs in the backend.
"""

import io

from PIL import Image

from multirag import config
from multirag.generate.base import PageImage
from multirag.retrieve import Hit

SYSTEM_PROMPT = (
    "You answer questions about document pages shown to you as images. "
    "Use only what is visible in the pages provided. "
    "If the pages do not contain the answer, say so plainly rather than guessing. "
    "Cite the page label given with each image for every claim you make."
)


def _downscaled(image: Image.Image) -> Image.Image:
    """Shrink to VLM_MAX_IMAGE_EDGE on the long edge, never enlarge.

    BOX, not LANCZOS. Measured on a 1700x2200 page: both cut image tokens from
    ~4986 to ~2533, but LANCZOS ringing around text adds gray levels that PNG
    cannot pack, inflating the file from 371KB to 580KB - larger than the
    original it replaced. BOX is plain area averaging, the standard choice for
    downscaling, and lands at 350KB with no ringing halos around small text.
    """
    longest = max(image.size)
    if longest <= config.VLM_MAX_IMAGE_EDGE:
        return image
    scale = config.VLM_MAX_IMAGE_EDGE / longest
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.BOX)


def load_page_images(hits: list[Hit]) -> list[PageImage]:
    """Load each hit's page image, downscaled, in rank order.

    Kept as PNG bytes: these are document scans where JPEG artifacts land hardest
    on small text and thin table rules - the same reason Phase 1 renders PNG.
    """
    pages = []
    for hit in hits:
        with Image.open(hit.image_path) as image:
            prepared = _downscaled(image.convert("RGB"))
            buffer = io.BytesIO()
            prepared.save(buffer, format="PNG", optimize=True)
        pages.append(
            PageImage(
                data=buffer.getvalue(),
                media_type="image/png",
                citation=hit.citation,
            )
        )
    return pages


def build_instruction(question: str, pages: list[PageImage]) -> str:
    """The text that accompanies the images.

    The page labels are repeated here so the model can refer to them by name.
    Images alone carry no filenames, and a citation the model invents is worse
    than no citation at all.
    """
    labels = "\n".join(f"- Page {i + 1}: {page.citation}" for i, page in enumerate(pages))
    return (
        f"Pages provided, in order:\n{labels}\n\n"
        f"Question: {question}\n\n"
        "Answer using only these pages, and cite the page label for each claim."
    )
