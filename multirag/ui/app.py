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

import streamlit as st

from multirag import config
from multirag.generate import UnknownBackendError, load_generator
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


@st.cache_resource(show_spinner=False)
def get_generator(name: str):
    """Cached per backend name, so switching backends does not rebuild both."""
    return load_generator(name)
