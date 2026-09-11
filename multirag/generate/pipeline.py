"""The single entry point for Phase 5: a question in, a cited answer out.

Ties retrieval to generation and nothing more. Both halves are injectable so a
caller can hold one open Retriever across many questions - reopening per
question would reload the encoder and fight for the Qdrant lock.
"""

from multirag.generate.base import Answer
from multirag.generate.factory import load_generator
from multirag.retrieve import Retriever


def ask(
    question: str,
    *,
    retriever: Retriever | None = None,
    generator=None,
    limit: int | None = None,
) -> Answer:
    """Retrieve pages for the question and answer from them.

    With no retriever, one is opened and closed around this call - convenient
    for a one-off, wasteful in a loop.
    """
    generator = generator or load_generator()

    if retriever is not None:
        return generator.answer(question, retriever.search(question, limit=limit))

    with Retriever.open() as opened:
        return generator.answer(question, opened.search(question, limit=limit))
