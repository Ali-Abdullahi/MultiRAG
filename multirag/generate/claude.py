"""A Claude backend for the generator contract. One option, not the interface.

The anthropic SDK is imported lazily and is not a hard dependency: the rest of
MultiRAG imports and runs without it. Credentials are resolved by the SDK from
the caller's own environment - this module never reads, stores, or defaults a
key, so a clone of this repo has no path to anyone else's account.
"""

import base64

from multirag import config
from multirag.generate import prompt as prompt_mod
from multirag.generate.base import Answer, PageImage
from multirag.retrieve import Hit

# Non-streaming, so keep this under the SDK's HTTP timeout.
MAX_TOKENS = 16000


class MissingCredentialsError(RuntimeError):
    """Raised when no usable credentials are configured for this backend."""


def _image_block(page: PageImage) -> dict:
    """One page as a base64 image block."""
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": page.media_type,
            "data": base64.standard_b64encode(page.data).decode("ascii"),
        },
    }


class ClaudeGenerator:
    """Sends page images to a Claude vision model and returns a cited answer."""

    def __init__(self, model: str | None = None):
        self.model = model or config.GENERATOR_MODEL
        self.name = f"claude:{self.model}"
        self._client = None

    def _get_client(self):
        """Build the SDK client on first use, not at import time."""
        if self._client is None:
            import anthropic

            # No api_key argument on purpose: the SDK resolves the caller's own
            # credentials from their environment.
            self._client = anthropic.Anthropic()
        return self._client

    def answer(self, question: str, hits: list[Hit]) -> Answer:
        """Answer from the given pages. Images precede text, as the API expects."""
        import anthropic

        pages = prompt_mod.load_page_images(hits)
        content = [_image_block(page) for page in pages]
        content.append({"type": "text", "text": prompt_mod.build_instruction(question, pages)})

        try:
            response = self._get_client().messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=prompt_mod.SYSTEM_PROMPT,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": content}],
            )
        except anthropic.AuthenticationError as error:
            raise MissingCredentialsError(
                "no Anthropic credentials found; export ANTHROPIC_API_KEY in your "
                "own environment, or set MULTIRAG_GENERATOR to a different backend"
            ) from error

        text = "\n".join(block.text for block in response.content if block.type == "text")
        return Answer(text=text, model=self.model, hits=tuple(hits))
