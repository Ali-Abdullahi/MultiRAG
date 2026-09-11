"""Ingestion: PDF files in, page images and manifests out. No ML in this layer."""

from multirag.ingest.pipeline import ingest_pdf

__all__ = ["ingest_pdf"]
