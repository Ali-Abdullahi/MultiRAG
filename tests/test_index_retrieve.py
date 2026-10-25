"""Phases 3-4: idempotent indexing, the collection contract, exact MaxSim, and the lock.

Runs a real embedded Qdrant in a temp directory - fast, and the only way to
test what the index actually computes rather than what we assume it does.
"""

import pytest
from conftest import fake_info, unit_rows

from multirag import config
from multirag.embed import store
from multirag.index import client as qc
from multirag.index import index_document
from multirag.index.pipeline import IndexConflictError, NotEmbeddedError
from multirag.ingest import ingest_pdf
from multirag.retrieve import RetrievalMismatchError, Retriever


class FakeQueryEncoder:
    def __init__(self, info=None, query=None):
        self.info = info or fake_info()
        self.query = unit_rows(6, seed=99) if query is None else query

    def encode_queries(self, queries):
        return [self.query for _ in queries]


@pytest.fixture(autouse=True)
def configured_for_fake_encoder(monkeypatch):
    # Indexing checks vectors against the *configured* encoder too (guard 2 of
    # the chain), so config must name the fake one these tests embed with.
    monkeypatch.setattr(config, "EMBED_MODEL_ID", "test/encoder")
    monkeypatch.setattr(config, "EMBED_MODEL_REVISION", "rev1")


@pytest.fixture
def indexed(make_pdf, fake_embed):
    render = ingest_pdf(make_pdf())
    fake_embed(render)
    index_document(render["doc_id"])
    return render


def test_reindexing_overwrites_instead_of_duplicating(indexed):
    index_document(indexed["doc_id"])
    client = qc.open_client()
    try:
        assert qc.collection_size(client) == 3
    finally:
        client.close()


def test_vectors_from_another_model_are_refused(make_pdf, fake_embed, indexed, monkeypatch):
    # The realistic way in: the collection was built with one encoder, then
    # config moved to another and new documents were embedded with it.
    monkeypatch.setattr(config, "EMBED_MODEL_ID", "other/model")
    other = ingest_pdf(make_pdf("other.pdf", texts=("another document",)))
    fake_embed(other, info=fake_info(model_id="other/model", revision="rev1"))
    # Mixing encoders raises nothing in Qdrant - MaxSim just ranks nonsense -
    # which is why this check has to exist on our side.
    with pytest.raises(IndexConflictError, match="model_id"):
        index_document(other["doc_id"])


def test_partial_embeddings_are_not_indexed(make_pdf, fake_embed):
    render = ingest_pdf(make_pdf())
    fake_embed(render, pages=1, complete=False)
    with pytest.raises(NotEmbeddedError, match="partially"):
        index_document(render["doc_id"])


def test_qdrant_maxsim_matches_an_independent_numpy_maxsim(indexed):
    encoder = FakeQueryEncoder()
    with Retriever.open(encoder=encoder) as retriever:
        hits = retriever.search("anything", limit=3)

    def maxsim(page_index):
        page = store.load_vectors(indexed["doc_id"], page_index)
        return float((encoder.query @ page.T).max(axis=1).sum())

    expected = sorted((maxsim(i) for i in range(3)), reverse=True)
    # fp16 storage on both sides; agreement to ~3 decimal places per term.
    assert [h.score for h in hits] == pytest.approx(expected, abs=1e-2)
    assert hits[0].match == pytest.approx(hits[0].score / 6)


def test_retriever_refuses_an_encoder_the_index_was_not_built_with(indexed):
    with pytest.raises(RetrievalMismatchError, match="model_id"):
        Retriever.open(encoder=FakeQueryEncoder(info=fake_info(model_id="other/model")))
    # The failed open must not leak the lock.
    qc.open_client().close()


def test_the_store_admits_one_client_at_a_time(indexed):
    first = qc.open_client()
    try:
        with pytest.raises(qc.StoreLockedError, match="Streamlit"):
            qc.open_client()
    finally:
        first.close()
    qc.open_client().close()


def test_hits_carry_what_generation_and_citations_need(indexed):
    with Retriever.open(encoder=FakeQueryEncoder()) as retriever:
        hit = retriever.search("anything", limit=1)[0]
    assert hit.image_path.exists()
    assert hit.citation.startswith("doc.pdf p.")
    assert hit.query_tokens == 6
