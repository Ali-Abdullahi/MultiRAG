"""A backend for any provider that speaks the OpenAI chat-completions format.

OpenAI, Gemini's compatibility endpoint, Groq, Mistral, OpenRouter, a local
Ollama - one class covers them, because they differ only in base URL, model name
and key. Writing a backend per provider would not scale and adds nothing.

Plain HTTP via httpx rather than a provider SDK: httpx is already installed, and
the request is small enough that an SDK would be a dependency for its own sake.

The key is held privately, never logged, never put in an error message, and
never written anywhere. It comes from the constructor (the UI, per session) or
the OPENAI_API_KEY environment variable - nowhere else.
"""

import base64
import os

from multirag.generate import prompt as prompt_mod
from multirag.generate.base import PageImage

DEFAULT_BASE_URL = "https://api.openai.com/v1"

# Generous: several page images plus a model that may think before answering.
TIMEOUT_SECONDS = 120
MAX_TOKENS = 4096


def _image_part(page: PageImage) -> dict:
    """One page as an inline data-URI image part - the format these APIs share."""
    data = base64.standard_b64encode(page.data).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{page.media_type};base64,{data}"}}


class OpenAICompatGenerator:
    """Sends page images to any OpenAI-compatible vision model."""

    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None):
        # No default model on purpose: a hardcoded name goes stale, and a
        # text-only model would fail on the images. The user names one they have.
        self.model = (model or os.environ.get("MULTIRAG_OPENAI_MODEL") or "").strip()
        self.base_url = (base_url or os.environ.get("MULTIRAG_OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self._api_key = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
        self.name = f"openai-compatible:{self.model or '?'}"

    def __repr__(self) -> str:
        # Defined so the key can never surface through a debugger, a log line,
        # or Streamlit rendering the object.
        return f"OpenAICompatGenerator(model={self.model!r}, base_url={self.base_url!r})"

    def _payload(self, question: str, pages: list[PageImage]) -> dict:
        content = [_image_part(page) for page in pages]
        content.append({"type": "text", "text": prompt_mod.build_instruction(question, pages)})
        return {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "messages": [
                {"role": "system", "content": prompt_mod.SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        }
