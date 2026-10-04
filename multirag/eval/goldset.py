"""The gold set: questions paired with the pages that should answer them.

A gold set is written by a person, so it refers to pages the way a person does -
by filename and printed page number, the same form Hit.citation prints. Nothing
downstream can use that: retrieval returns content-addressed page_ids. Resolving
one to the other is this module's whole job, and it is deliberately strict.

A reference that does not resolve must raise. The alternative - treating an
unknown page as simply never retrieved - scores it zero, which is
indistinguishable from a retriever that got it wrong. That turns "the corpus
moved" into "the model regressed", which is the single most expensive way for an
eval harness to lie.
"""

from dataclasses import dataclass
from pathlib import Path

from multirag import config, jsonio

# Bump when the fixture layout changes incompatibly, so an old gold set is
# reported rather than half-read.
SCHEMA_VERSION = 1


class GoldSetError(ValueError):
    """Raised when a gold set is malformed or refers to pages that do not exist."""


@dataclass(frozen=True)
class PageRef:
    """One page as a human wrote it: a filename and a 1-based page number."""

    source_name: str
    page: int

    @property
    def citation(self) -> str:
        """Matches Hit.citation, so expected and retrieved print identically."""
        return f"{self.source_name} p.{self.page}"
