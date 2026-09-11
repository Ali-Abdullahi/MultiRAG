"""A backend that calls nothing. Useful without credentials.

Exists so the retrieval-to-answer path can be exercised in tests, in CI, and by
anyone who has cloned this repo without a provider account. It reports exactly
what would have been sent, which also makes it the fastest way to see whether
retrieval put the right pages in front of the model.
"""

from multirag.generate import prompt as prompt_mod
from multirag.generate.base import Answer
from multirag.retrieve import Hit


class EchoGenerator:
    """Describes the request instead of answering it. No network, no credentials."""

    name = "echo"

    def answer(self, question: str, hits: list[Hit]) -> Answer:
        pages = prompt_mod.load_page_images(hits)
        lines = [
            f"[echo] no model was called. {len(pages)} page(s) would be sent:",
            *(
                f"  {i + 1}. {page.citation} ({len(page.data) / 1024:.0f} KB PNG)"
                for i, page in enumerate(pages)
            ),
            f"[echo] question: {question}",
        ]
        return Answer(text="\n".join(lines), model="echo", hits=tuple(hits))
