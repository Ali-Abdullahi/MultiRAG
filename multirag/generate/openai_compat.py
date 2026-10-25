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
from multirag.generate.base import Answer, MissingCredentialsError, PageImage
from multirag.retrieve import Hit

DEFAULT_BASE_URL = "https://api.openai.com/v1"

# Generous: several page images plus a model that may think before answering.
TIMEOUT_SECONDS = 120
MAX_TOKENS = 4096


class ProviderError(RuntimeError):
    """The provider refused or failed the request. Message is safe to display."""


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

    def _redact(self, text: str) -> str:
        """Strip the key from provider text before it can be displayed.

        Some providers echo part or all of a rejected key in their error body.
        Errors are rendered in the UI, so this runs on everything that is shown.
        """
        if self._api_key:
            text = text.replace(self._api_key, "[redacted]")
        return text[:300]

    def answer(self, question: str, hits: list[Hit]) -> Answer:
        """Answer from the given pages, or raise an error that is safe to show."""
        import httpx

        if not self.model:
            raise ValueError("no model set; enter a vision-capable model name for this provider")
        if not self._api_key and "localhost" not in self.base_url and "127.0.0.1" not in self.base_url:
            # Local servers such as Ollama need no key; hosted ones always do.
            raise MissingCredentialsError("no API key; enter one in the sidebar or set OPENAI_API_KEY")

        pages = prompt_mod.load_page_images(hits)
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                json=self._payload(question, pages),
                headers=headers,
                timeout=TIMEOUT_SECONDS,
            )
        except httpx.TimeoutException:
            raise ProviderError(f"{self.base_url} did not answer within {TIMEOUT_SECONDS}s") from None
        except httpx.HTTPError as error:
            # `from None`: the chained exception holds the request, headers included.
            raise ProviderError(f"could not reach {self.base_url}: {type(error).__name__}") from None

        # Gemini reports a bad key as 400 rather than 401, so the body is the only
        # signal. Checked before the generic 400 so the user is told to fix the key.
        bad_key = response.status_code == 400 and "api key" in response.text.lower()
        if response.status_code in (401, 403) or bad_key:
            raise MissingCredentialsError("the provider rejected this API key")
        if response.status_code in (429, 503):
            # Measured on Gemini's free tier: 503 "high demand", and one-word
            # replies taking 157s. The provider is the bottleneck, not this app.
            raise ProviderError(
                f"provider busy ({response.status_code}) - try again shortly, or pick a "
                "less loaded model (e.g. a 'lite' variant)"
            )
        if response.status_code == 404:
            raise ProviderError(f"not found - check the base URL and that model {self.model!r} exists")
        if response.status_code >= 400:
            raise ProviderError(
                f"provider returned {response.status_code} (a text-only model fails on page "
                f"images): {self._redact(response.text)}"
            )

        message = response.json()["choices"][0]["message"]["content"]
        # Most providers return a string; some return a list of content parts.
        if isinstance(message, list):
            message = "\n".join(part.get("text", "") for part in message if isinstance(part, dict))
        return Answer(text=message or "", model=self.model, hits=tuple(hits))
