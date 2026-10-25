"""Streamlit front end for MultiRAG.

    streamlit run multirag/ui/app.py

Two things shape this file more than any layout choice.

Streamlit re-executes the whole script on every widget interaction. Loading the
encoder takes ~8s, so anything expensive must live behind st.cache_resource or
the app reloads a model every time a slider moves.

Embedded Qdrant holds an exclusive lock on data/qdrant. While this app runs it
owns that lock, so indexing from another process will fail until it stops. That
is a property of running the database in-process, not a bug to work around here.
"""

import os

import streamlit as st

from multirag import config
from multirag.generate import UnknownBackendError, load_generator
from multirag.generate.factory import BACKENDS
from multirag.index import manifest as index_manifest
from multirag.retrieve import RetrievalMismatchError, Retriever

st.set_page_config(page_title="MultiRAG", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading encoder and opening the index...")
def get_retriever() -> Retriever:
    """One retriever for the life of the process.

    Never closed on purpose: Streamlit has no reliable shutdown hook, and the
    lock is released when the process exits. Closing it on a rerun would just
    reopen it on the next one.
    """
    return Retriever.open()


def backend_settings(backend: str) -> dict:
    """Provider fields for the chosen backend, as load_generator keyword args.

    The key field is a password input whose value lives only in this browser
    session's st.session_state - in this process's memory, never on disk, never
    in the shared cache, gone when the tab closes or the app stops. An empty
    field falls back to the provider's environment variable.
    """
    if backend == "echo":
        return {}

    if backend == "claude":
        settings = {"model": st.text_input("Model", value=config.GENERATOR_MODEL)}
        env_var = "ANTHROPIC_API_KEY"
    else:
        settings = {
            "base_url": st.text_input(
                "Base URL",
                value="https://api.openai.com/v1",
                help="Any OpenAI-compatible endpoint: OpenAI, Gemini, Groq, Mistral, "
                "OpenRouter, or a local server such as Ollama (http://localhost:11434/v1).",
            ),
            "model": st.text_input(
                "Model", placeholder="a vision-capable model", help="Must accept images."
            ),
        }
        env_var = "OPENAI_API_KEY"

    settings["api_key"] = st.text_input(
        "API key",
        type="password",
        key=f"api_key_{backend}",
        help="Kept only in this browser session's memory - never saved or logged.",
    )
    if not settings["api_key"] and os.environ.get(env_var):
        st.caption(f"Using `{env_var}` from your environment.")
    return settings


def sidebar() -> tuple[str, int, dict]:
    """Backend, provider and top-k controls. Returns the current selection."""
    with st.sidebar:
        st.header("Settings")

        contract = index_manifest.read_manifest()
        if contract:
            st.caption(f"Index: `{contract['model_id']}` · dim {contract['dim']}")
            st.caption(f"Revision `{contract['model_revision'][:8]}`")
        else:
            st.warning("Nothing indexed yet.")

        backend = st.selectbox(
            "Answer backend",
            BACKENDS,
            help=(
                "echo calls no model and needs no credentials - it reports which "
                "pages would be sent. The others call a provider with your own key."
            ),
        )
        settings = backend_settings(backend)
        top_k = st.slider("Pages to retrieve", 1, 10, config.RETRIEVE_TOP_K)

        st.divider()
        st.caption(
            "This app holds an exclusive lock on the Qdrant store. Stop it before "
            "indexing new documents."
        )
    return backend, top_k, settings


def show_hits(hits) -> None:
    """Render retrieved pages as images with their scores.

    Scores are shown as plain numbers with no bar or percentage. MaxSim sums
    scale with query length, so they rank pages within one query and mean
    nothing across queries - a progress bar would imply a confidence this
    number does not carry.
    """
    for column, hit in zip(st.columns(len(hits)), hits):
        with column:
            st.image(str(hit.image_path), use_container_width=True)
            st.caption(f"**{hit.citation}** · score {hit.score:.2f}")


def main() -> None:
    st.title("MultiRAG")
    st.caption("Ask a question; pages are retrieved as images, never as extracted text.")

    backend, top_k, settings = sidebar()

    question = st.text_input("Question", placeholder="When is the homework due?")
    if not question:
        return

    try:
        retriever = get_retriever()
    except RetrievalMismatchError as error:
        st.error(str(error))
        return

    with st.spinner("Searching..."):
        hits = retriever.search(question, limit=top_k)

    if not hits:
        st.info("No pages matched.")
        return

    st.subheader(f"{len(hits)} pages retrieved")
    show_hits(hits)

    # Generation is behind its own button on purpose: retrieval is free and
    # instant, generation costs money and seconds. Nobody should pay for a VLM
    # call just because they moved a slider.
    st.divider()
    if not st.button(f"Generate answer with `{backend}`", type="primary"):
        return

    # Built per click, never cached: st.cache_resource is shared by every session
    # connected to this server, so a generator holding one user's key must not
    # live there. Construction is cheap; the expensive part is the API call.
    try:
        generator = load_generator(backend, **settings)
    except UnknownBackendError as error:
        st.error(str(error))
        return

    with st.spinner(f"Asking {backend}..."):
        try:
            answer = generator.answer(question, hits)
        except Exception as error:  # backends fail in provider-specific ways
            st.error(f"{type(error).__name__}: {error}")
            return

    st.subheader("Answer")
    st.markdown(answer.text)
    st.caption("Sources: " + " · ".join(answer.sources))


main()
