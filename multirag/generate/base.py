"""The generator contract: one interface, any vision model behind it.

Nothing in this package assumes a particular provider. A backend receives page
images and a question and returns an answer; where it runs, what it costs, and
whose account pays are entirely its own business.

Credentials are never read here, never defaulted, and never written to disk.
A clone of this repo talks to nobody until its owner supplies a backend and a
key from their own environment.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from multirag.retrieve import Hit


@dataclass(frozen=True)
class PageImage:
    """One retrieved page, loaded and ready to send to a vision model."""

    data: bytes
    media_type: str
    citation: str


@dataclass(frozen=True)
class Answer:
    """A generated answer and the pages it was grounded in."""

    text: str
    model: str
    hits: tuple[Hit, ...]

    @property
    def sources(self) -> list[str]:
        """Citations for the pages the model was shown, in rank order."""
        return [hit.citation for hit in self.hits]


@runtime_checkable
class Generator(Protocol):
    """Anything that can turn a question plus page images into an answer."""

    name: str

    def answer(self, question: str, hits: list[Hit]) -> Answer:
        """Answer the question using only the given pages."""
        ...
