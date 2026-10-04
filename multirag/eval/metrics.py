"""Retrieval metrics: ranked page_ids in, numbers out.

Deliberately pure. Nothing here opens a file, loads a model, or knows what a
page is - it compares a ranked list of ids against a set of ids. That is what
makes the arithmetic checkable by hand, which matters more here than anywhere
else in the pipeline: a bug in a metric does not crash, it just reports a
number, and a wrong number is worse than no number because it gets acted on.

k is always an explicit argument. A default would let a caller measure at one
depth and report another, and the discrepancy would never surface.
"""

from collections.abc import Sequence


def rank_of_first_relevant(
    retrieved: Sequence[str], relevant: frozenset[str], k: int
) -> int | None:
    """1-based rank of the first relevant page in the top k, or None if absent.

    Ranks are 1-based because that is how a result list is read and how the
    reciprocal rank is defined. Everything else in this codebase indexes pages
    from 0; this is the one place that must not.
    """
    for rank, page_id in enumerate(retrieved[:k], start=1):
        if page_id in relevant:
            return rank
    return None


def reciprocal_rank(
    retrieved: Sequence[str], relevant: frozenset[str], k: int
) -> float:
    """1/rank of the first relevant page, or 0.0 if none made the cut.

    Rewards putting a right answer at the top rather than merely somewhere in
    the list: rank 1 scores 1.0, rank 2 scores 0.5, rank 5 scores 0.2. For a
    question with several relevant pages it only sees the first one found -
    recall_at_k is what notices the rest.
    """
    rank = rank_of_first_relevant(retrieved, relevant, k)
    return 0.0 if rank is None else 1.0 / rank
