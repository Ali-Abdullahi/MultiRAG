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
