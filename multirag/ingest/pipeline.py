"""The single entry point for Phase 1: a PDF path in, a manifest out.

Callers should use ingest_pdf rather than driving render and manifest by hand,
so the skip-and-refresh decision lives in exactly one place.
"""

from pathlib import Path

from multirag.ingest import ids, manifest
from multirag.ingest.render import render_pdf


def ingest_pdf(pdf_path: Path, *, force: bool = False) -> dict:
    """Render a PDF and write its manifest. Returns the manifest.

    Idempotent by design. Hashing the file is cheap; rendering is not. So the
    document id is computed first and an existing, current manifest short-
    circuits the whole stage - re-ingesting a known corpus touches no pixels.

    A stale manifest forces a re-render even when force is False: its images
    were produced under settings that no longer apply, and reusing them is the
    silent-corruption case this pipeline is built to avoid.
    """
    pdf_path = Path(pdf_path)
    document_id = ids.doc_id(pdf_path)

    existing = manifest.read_manifest(document_id)
    reason = manifest.stale_reason(existing) if existing else None
    if existing and not force and reason is None:
        return existing

    _, pages = render_pdf(pdf_path, force=force or reason is not None)
    record = manifest.build_manifest(document_id, pdf_path, pages)
    manifest.write_manifest(record)
    return record
