"""Run a gold set against the retriever and score what comes back.

Retrieval-only: no VLM is called and nothing here costs money, which is the
point. Ranking quality is the upstream variable - a generator cannot answer from
a page that was never retrieved - so it is worth measuring on its own, often,
before anything expensive is wired in.
"""

from dataclasses import dataclass
from datetime import datetime, timezone

from multirag import config
from multirag.eval import metrics
from multirag.eval.goldset import GoldSet, PageRef
from multirag.index import manifest as index_manifest
from multirag.retrieve import Hit


@dataclass(frozen=True)
class QuestionResult:
    """One question's retrieval, scored.

    Keeps the hits themselves, not just the numbers. A score of 0 is a question,
    not an answer: the next thing anyone wants to know is what *was* returned
    instead, and reconstructing that means re-running the query.
    """

    qid: str
    question: str
    relevant: frozenset[str]
    refs: tuple[PageRef, ...]
    hits: tuple[Hit, ...]
    rank: int | None
    recall: float
    reciprocal_rank: float

    @property
    def found(self) -> bool:
        """Whether any relevant page made the top k."""
        return self.rank is not None

    @property
    def expected(self) -> str:
        """Expected pages as a person wrote them, for a failure line."""
        return ", ".join(ref.citation for ref in self.refs)
