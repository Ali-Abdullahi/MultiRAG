"""Shared fixtures. Every test runs against a throwaway data directory.

autouse is the point: a test that forgot to ask for isolation must not be able
to write into the real data/ - or take the real Qdrant lock away from a running
app. Modules read config attributes at call time, so patching them is enough.
"""

import fitz
import pytest

from multirag import config


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    data = tmp_path / "data"
    for name, path in {
        "DATA_DIR": data,
        "PAGES_DIR": data / "pages",
        "MANIFESTS_DIR": data / "manifests",
        "EMBEDDINGS_DIR": data / "embeddings",
        "QDRANT_PATH": data / "qdrant",
        "EVAL_RUNS_DIR": data / "evals",
        "EVALSETS_DIR": tmp_path / "evalsets",
    }.items():
        monkeypatch.setattr(config, name, path)
    return data


@pytest.fixture
def make_pdf(tmp_path):
    """Build a small US-letter PDF. Same texts in, byte-identical file out."""

    def build(name="doc.pdf", texts=("page one", "page two", "page three")):
        document = fitz.open()
        for text in texts:
            document.new_page(width=612, height=792).insert_text((72, 100), text, fontsize=14)
        path = tmp_path / name
        # Fixed metadata: PyMuPDF otherwise stamps creation times, and identity
        # tests depend on identical inputs producing identical bytes.
        document.set_metadata({"creationDate": "", "modDate": ""})
        document.save(path, garbage=4, deflate=True, no_new_id=True)
        return path

    return build


def fake_info(model_id="test/encoder", revision="rev1", dim=128):
    from multirag.embed.identity import EncoderInfo

    return EncoderInfo(model_id=model_id, revision=revision, dim=dim, device="cpu", dtype="float32")


def unit_rows(rows, dim=128, seed=0):
    """Random L2-normalized rows - the shape and norm real page vectors have."""
    import numpy as np

    matrix = np.random.default_rng(seed).normal(size=(rows, dim)).astype(np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


@pytest.fixture
def fake_embed():
    """Embed an ingested document without a model: real storage code, fake vectors.

    Exercises everything downstream of the encoder - manifests, staleness,
    indexing - in milliseconds and without torch.
    """
    from multirag.embed import manifest as embed_manifest
    from multirag.embed import store

    def embed(render, info=None, patches=20, pages=None, complete=True):
        info = info or fake_info()
        records = []
        for page in render["pages"][:pages]:
            index = page["page_index"]
            store.save_vectors(render["doc_id"], index, unit_rows(patches, info.dim, seed=index))
            records.append(embed_manifest.page_record(index, page["page_id"], patches, render["doc_id"]))
        manifest = embed_manifest.build_manifest(render, info, records, complete=complete)
        embed_manifest.write_manifest(manifest)
        return manifest

    return embed
