"""The single entry point for Phase 2: a rendered document in, vectors out.

Resumability is the design driver here. At ~8s per page on CPU, a 100-page PDF
is a quarter of an hour, and a run that dies at page 90 must not start over.

The manifest is therefore rewritten after every page, not once at the end, and
carries a `complete` flag. That makes it a checkpoint as well as a record: a
resumed run trusts the vectors it already lists, because the manifest also names
the model and render settings they were produced under. Orphan .npy files with
no manifest entry are never trusted - they could be from any model.
"""

from pathlib import Path

from multirag.embed import manifest as embed_manifest
from multirag.embed import store
from multirag.ingest import manifest as render_manifest_mod


class StaleRenderError(RuntimeError):
    """Raised when the page images themselves no longer match the render config."""


def _resumable_pages(document_id: str, existing: dict | None) -> dict[int, dict]:
    """Page records from a previous run that can be trusted and skipped.

    A record is only reusable if its .npy is still on disk. The manifest having
    been checked for staleness by the caller is what makes the rest safe: these
    vectors came from the model and render settings currently in force.
    """
    if not existing:
        return {}
    return {
        record["page_index"]: record
        for record in existing["pages"]
        if store.has_vectors(document_id, record["page_index"])
    }


def embed_document(
    document_id: str,
    *,
    encoder=None,
    force: bool = False,
    on_page=None,
) -> dict:
    """Embed every page of an already-rendered document. Returns the manifest.

    Refuses to run against stale page images rather than embedding them: vectors
    built from images that no longer match the render config are the silent
    corruption this pipeline exists to prevent.

    Pass an encoder to reuse one across a corpus - loading costs ~8s. Omit it and
    one is loaded lazily, which means a fully-cached document costs no model load
    at all. on_page(index, total, page_id) is called after each page, because a
    quarter-hour of silence is not an acceptable interface.
    """
    render = render_manifest_mod.read_manifest(document_id)
    if render is None:
        raise FileNotFoundError(f"{document_id} has not been ingested")

    reason = render_manifest_mod.stale_reason(render)
    if reason:
        raise StaleRenderError(f"re-run ingestion for {document_id}: {reason}")

    existing = None if force else embed_manifest.read_manifest(document_id)
    if existing and embed_manifest.stale_reason(
        existing, render_manifest=render, info=encoder.info if encoder else None
    ):
        existing = None  # nothing in it can be trusted

    done = _resumable_pages(document_id, existing)
    if existing and existing.get("complete") and len(done) == render["page_count"]:
        return existing

    if encoder is None:
        from multirag.embed.encoder import PageEncoder

        encoder = PageEncoder.load()
        # The cheap pre-check compared against config; now that the real model is
        # loaded, re-check before trusting a single vector from the last run.
        if existing and embed_manifest.stale_reason(
            existing, render_manifest=render, info=encoder.info
        ):
            existing, done = None, {}

    from PIL import Image

    records: list[dict] = []
    for page in render["pages"]:
        index = page["page_index"]
        if index in done:
            records.append(done[index])
            continue

        with Image.open(render_manifest_mod.page_image_path(page)) as image:
            [vectors] = encoder.encode_pages([image.convert("RGB")])
        store.save_vectors(document_id, index, vectors)
        records.append(
            embed_manifest.page_record(index, page["page_id"], int(vectors.shape[0]), document_id)
        )

        # Checkpoint after every page. One small atomic write against eight
        # seconds of compute is a trade worth making every time.
        embed_manifest.write_manifest(
            embed_manifest.build_manifest(render, encoder.info, records, complete=False)
        )
        if on_page:
            on_page(index, render["page_count"], page["page_id"])

    final = embed_manifest.build_manifest(render, encoder.info, records, complete=True)
    embed_manifest.write_manifest(final)
    return final
