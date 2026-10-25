"""Phase 5: backends, provider error handling, and keeping keys out of everything shown.

A local HTTP server stands in for the provider, so these run offline and can
make it misbehave on purpose - including echoing the key back.
"""

import base64
import io
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from PIL import Image

from multirag.generate import MissingCredentialsError, UnknownBackendError, load_generator
from multirag.generate.base import redact
from multirag.generate.openai_compat import OpenAICompatGenerator, ProviderError
from multirag.retrieve import Hit

KEY = "sk-test-0123456789abcdef"

# path prefix -> (status, body); "{auth}" is replaced with the received header.
RESPONSES = {
    "/ok": (200, {"choices": [{"message": {"content": "Answer citing Page 1"}}]}),
    "/parts": (200, {"choices": [{"message": {"content": [{"type": "text", "text": "in parts"}]}}]}),
    "/unauthorized": (401, {"error": "nope"}),
    "/gemini-bad-key": (400, {"error": {"message": "Please pass a valid API key"}}),
    "/busy": (503, {"error": "This model is currently experiencing high demand"}),
    "/missing": (404, {"error": "not found"}),
    "/leak": (400, {"error": "rejected request for {auth}"}),
}


@pytest.fixture(scope="module")
def provider():
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen["auth"] = self.headers.get("Authorization")
            status, body = RESPONSES[self.path.split("/chat")[0]]
            data = json.dumps(body).replace("{auth}", seen["auth"] or "").encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", seen
    server.shutdown()


@pytest.fixture
def hits(tmp_path):
    image = tmp_path / "page.png"
    Image.new("RGB", (1700, 2200), "white").save(image)
    return [Hit(5.0, "doc", "Notes.pdf", 0, "doc-p0000", image, query_tokens=10)]


def ask(provider, path, hits, key=KEY):
    url, _ = provider
    return OpenAICompatGenerator(model="vision-model", base_url=url + path, api_key=key).answer("Q?", hits)


def test_a_successful_answer_carries_its_pages(provider, hits):
    answer = ask(provider, "/ok", hits)
    assert answer.text == "Answer citing Page 1"
    assert answer.sources == ["Notes.pdf p.1"]

    body = provider[1]["body"]
    parts = body["messages"][1]["content"]
    assert [p["type"] for p in parts] == ["image_url", "text"]  # images, then the question
    assert provider[1]["auth"] == f"Bearer {KEY}"


def test_list_shaped_content_is_joined(provider, hits):
    assert ask(provider, "/parts", hits).text == "in parts"


def test_pages_are_downscaled_before_sending(provider, hits):
    ask(provider, "/ok", hits)
    url = provider[1]["body"]["messages"][1]["content"][0]["image_url"]["url"]
    image = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert max(image.size) == 1568  # VLM_MAX_IMAGE_EDGE; retrieval keeps full size


@pytest.mark.parametrize(
    "path, error, message",
    [
        ("/unauthorized", MissingCredentialsError, "rejected this API key"),
        ("/gemini-bad-key", MissingCredentialsError, "rejected this API key"),
        ("/busy", ProviderError, "provider busy"),
        ("/missing", ProviderError, "check the base URL"),
    ],
)
def test_provider_failures_say_what_to_do(provider, hits, path, error, message):
    with pytest.raises(error, match=message):
        ask(provider, path, hits)


def test_a_key_echoed_by_the_provider_is_redacted(provider, hits):
    with pytest.raises(ProviderError) as raised:
        ask(provider, "/leak", hits)
    assert KEY not in str(raised.value)
    assert "[redacted]" in str(raised.value)
    assert raised.value.__cause__ is None  # nothing chained that could carry headers


def test_an_unreachable_provider_is_reported_without_the_request(hits):
    generator = OpenAICompatGenerator(model="m", base_url="http://127.0.0.1:1", api_key=KEY)
    with pytest.raises(ProviderError, match="could not reach") as raised:
        generator.answer("Q?", hits)
    assert raised.value.__cause__ is None


def test_keys_never_appear_in_a_generators_repr():
    for backend in ("claude", "openai-compatible"):
        assert KEY not in repr(load_generator(backend, api_key=KEY))


def test_redact_ignores_strings_too_short_to_be_keys():
    assert redact("or pick a model", "k") == "or pick a model"
    assert redact(f"bad {KEY} here", KEY) == "bad [redacted] here"
    assert redact("text", None) == "text"


def test_missing_settings_are_explained_before_any_request(hits, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # else the env key fills in
    with pytest.raises(ValueError, match="no model set"):
        OpenAICompatGenerator(api_key=KEY).answer("Q?", hits)
    with pytest.raises(MissingCredentialsError, match="no API key"):
        OpenAICompatGenerator(model="m", api_key="", base_url="https://example.invalid/v1").answer("Q?", hits)


def test_claude_backend_explains_a_missing_sdk(hits, monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)  # import now fails
    with pytest.raises(RuntimeError, match="pip install anthropic"):
        load_generator("claude", api_key=KEY).answer("Q?", hits)


def test_echo_calls_nothing_and_reports_the_pages(hits):
    answer = load_generator("echo").answer("Q?", hits)
    assert "no model was called" in answer.text
    assert answer.sources == ["Notes.pdf p.1"]


def test_unknown_backends_list_the_known_ones():
    with pytest.raises(UnknownBackendError, match="openai-compatible"):
        load_generator("nope")
