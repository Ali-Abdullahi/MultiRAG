"""A Claude backend for the generator contract. One option, not the interface.

The anthropic SDK is imported lazily and is not a hard dependency: the rest of
MultiRAG imports and runs without it. Credentials are resolved by the SDK from
the caller's own environment - this module never reads, stores, or defaults a
key, so a clone of this repo has no path to anyone else's account.
"""

import base64

from multirag import config
from multirag.generate import prompt as prompt_mod
from multirag.generate.base import Answer, MissingCredentialsError, PageImage
from multirag.retrieve import Hit

# Non-streaming, so keep this under the SDK's HTTP timeout.
MAX_TOKENS = 16000


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


def _sdk():
    """Import the anthropic SDK, or say how to get it.

    It is an optional dependency - echo and openai-compatible run without it - so
    a missing install is an expected state, not a crash to show as a traceback.
    """
    try:
        import anthropic
    except ImportError:
        raise RuntimeError(
            "the claude backend needs the anthropic package: pip install anthropic"
        ) from None
    return anthropic


class ClaudeGenerator:
    """Sends page images to a Claude vision model and returns a cited answer."""

    def __init__(
        self, model: str | None = None, api_key: str | None = None, timeout: float | None = None
    ):
        self.model = model or config.GENERATOR_MODEL
        self.name = f"claude:{self.model}"
        self._api_key = (api_key or "").strip() or None
        # None keeps the SDK's own default (10 minutes).
        self._timeout = timeout
        self._client = None

    def __repr__(self) -> str:
        # Keeps a session-supplied key out of logs and rendered objects.
        return f"ClaudeGenerator(model={self.model!r})"

    def _get_client(self):
        """Build the SDK client on first use, not at import time."""
        if self._client is None:
            anthropic = _sdk()

            # A key typed into the UI is passed through; otherwise api_key=None
            # lets the SDK resolve the caller's own credentials from their
            # environment. Nothing is ever read from or written to disk here.
            options = {"timeout": self._timeout} if self._timeout else {}
            self._client = anthropic.Anthropic(api_key=self._api_key, **options)
        return self._client

    def answer(self, question: str, hits: list[Hit]) -> Answer:
        """Answer from the given pages. Images precede text, as the API expects."""
        anthropic = _sdk()

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
