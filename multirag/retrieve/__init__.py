"""Retrieval: a text query in, ranked page images out."""

from multirag.retrieve.search import Hit, RetrievalMismatchError, Retriever

__all__ = ["Hit", "Retriever", "RetrievalMismatchError"]
