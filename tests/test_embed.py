"""Phase 2: vector storage, the two-link staleness chain, and resumability.

The encoder is faked: these tests are about the contracts around the model,
which is what can silently break, not about the model's numbers.
"""

import numpy as np
import pytest
from conftest import fake_info, unit_rows

from multirag import config
from multirag.embed import embed_document, is_embedded, store
from multirag.embed import manifest as embed_manifest
from multirag.embed.pipeline import StaleRenderError
from multirag.ingest import ingest_pdf


class FakeEncoder:
    def __init__(self, info=None):
        self.info = info or fake_info()
        self.pages_encoded = 0

    def encode_pages(self, images):
        self.pages_encoded += len(images)
        return [unit_rows(20, self.info.dim, seed=self.pages_encoded) for _ in images]


def test_vectors_store_as_fp16_and_load_as_fp32():
    vectors = unit_rows(50)
    store.save_vectors("doc", 0, vectors)

    loaded = store.load_vectors("doc", 0)
    assert loaded.dtype == np.float32
    assert np.load(store.vector_path("doc", 0)).dtype == np.float16
    assert np.allclose(loaded, vectors, atol=1e-3)  # unit-norm values survive fp16


def test_a_flat_vector_is_refused():
    # A page is a matrix (one row per patch). A 1D array means something upstream
    # pooled it, and MaxSim over it would be meaningless.
    with pytest.raises(ValueError, match="2D"):
        store.save_vectors("doc", 0, np.ones(128, dtype=np.float32))


def test_staleness_names_a_different_model(make_pdf, fake_embed):
    render = ingest_pdf(make_pdf())
    embedded = fake_embed(render)

    assert embed_manifest.stale_reason(embedded, render_manifest=render, info=fake_info()) is None
    reason = embed_manifest.stale_reason(embedded, info=fake_info(model_id="other/model"))
    assert reason == "model_id 'test/encoder' != 'other/model'"


def test_staleness_follows_a_rerender_at_new_settings(make_pdf, fake_embed, monkeypatch):
    pdf = make_pdf()
    embedded = fake_embed(ingest_pdf(pdf))

    monkeypatch.setattr(config, "RENDER_DPI", 100)
    rerendered = ingest_pdf(pdf)
    reason = embed_manifest.stale_reason(embedded, render_manifest=rerendered, info=fake_info())
    assert reason.startswith("pages were re-rendered")


def test_rerender_under_same_settings_keeps_vectors_valid(make_pdf, fake_embed):
    # Settings, not timestamps: an identical re-render must not trigger ~8s/page
    # of recomputation just because created_at moved.
    pdf = make_pdf()
    embedded = fake_embed(ingest_pdf(pdf))
    again = ingest_pdf(pdf, force=True)
    assert embed_manifest.stale_reason(embedded, render_manifest=again, info=fake_info()) is None


def test_a_crashed_run_resumes_with_only_the_missing_pages(make_pdf, fake_embed):
    render = ingest_pdf(make_pdf())
    fake_embed(render, pages=2, complete=False)  # died after page 2 of 3

    encoder = FakeEncoder()
    result = embed_document(render["doc_id"], encoder=encoder)

    assert encoder.pages_encoded == 1
    assert result["complete"] and result["page_count"] == 3


def test_a_complete_document_is_not_reembedded(make_pdf, fake_embed):
    render = ingest_pdf(make_pdf())
    fake_embed(render)
    encoder = FakeEncoder()
    embed_document(render["doc_id"], encoder=encoder)
    assert encoder.pages_encoded == 0


def test_stale_page_images_are_refused_not_embedded(make_pdf, monkeypatch):
    render = ingest_pdf(make_pdf())
    monkeypatch.setattr(config, "RENDER_DPI", 100)
    with pytest.raises(StaleRenderError, match="re-run ingestion"):
        embed_document(render["doc_id"], encoder=FakeEncoder())


def test_is_embedded_answers_without_a_model(make_pdf, fake_embed, monkeypatch):
    monkeypatch.setattr(config, "EMBED_MODEL_ID", "test/encoder")
    monkeypatch.setattr(config, "EMBED_MODEL_REVISION", "rev1")
    render = ingest_pdf(make_pdf())

    assert not is_embedded(render["doc_id"])
    fake_embed(render, pages=2, complete=False)
    assert not is_embedded(render["doc_id"])
    fake_embed(render)
    assert is_embedded(render["doc_id"])
