"""Embedding: page images in, per-page multi-vectors on disk. One model, one contract."""

from multirag.embed.pipeline import embed_document, is_embedded

__all__ = ["embed_document", "is_embedded"]
