"""Phase 1: identity, rendering, idempotency, and the render staleness check."""

import shutil
import uuid

from multirag import config
from multirag.ingest import ids, ingest_pdf
from multirag.ingest import manifest as render_manifest


def test_doc_id_comes_from_bytes_not_filename(make_pdf, tmp_path):
    original = make_pdf("a.pdf")
    renamed = tmp_path / "completely different name.pdf"
    shutil.copy(original, renamed)
    other = make_pdf("b.pdf", texts=("different content",))

    assert ids.doc_id(original) == ids.doc_id(renamed)
    assert ids.doc_id(original) != ids.doc_id(other)


def test_point_ids_are_deterministic_uuids():
    page = ids.page_id("abc123", 7)
    assert page == "abc123-p0007"
    assert ids.page_point_id(page) == ids.page_point_id(page)
    uuid.UUID(ids.page_point_id(page))  # Qdrant accepts only ints or UUIDs


def test_page_namespace_never_changes():
    # Changing this turns every re-index into a duplicate insert. If this test
    # fails, the change is the bug - not the test.
    assert str(ids.PAGE_NAMESPACE) == "6f1d4f22-2b3a-5c7e-9a10-8c4b2e6d0f31"


def test_ingest_renders_every_page_at_the_contract_dpi(make_pdf):
    manifest = ingest_pdf(make_pdf())

    assert manifest["page_count"] == 3
    assert manifest["render_dpi"] == config.RENDER_DPI
    for page in manifest["pages"]:
        assert (page["width"], page["height"]) == (1700, 2200)  # US letter at 200 DPI
        assert not page["image"].startswith("/")  # stored relative, for portability
        assert render_manifest.page_image_path(page).exists()


def test_reingest_is_idempotent_and_touches_no_pixels(make_pdf):
    pdf = make_pdf()
    first = ingest_pdf(pdf)
    images = [render_manifest.page_image_path(p) for p in first["pages"]]
    mtimes = [image.stat().st_mtime_ns for image in images]

    second = ingest_pdf(pdf)

    assert second == first
    assert [image.stat().st_mtime_ns for image in images] == mtimes


def test_changed_render_settings_are_detected_and_rerendered(make_pdf, monkeypatch):
    pdf = make_pdf()
    old = ingest_pdf(pdf)

    monkeypatch.setattr(config, "RENDER_DPI", 100)
    assert render_manifest.stale_reason(old) == "render_dpi 200 != 100"

    new = ingest_pdf(pdf)
    assert new["render_dpi"] == 100
    assert new["pages"][0]["width"] == 850  # half the width, at half the DPI


def test_manifest_writes_leave_no_temp_files(make_pdf, isolated_data):
    ingest_pdf(make_pdf())
    assert not list(isolated_data.rglob("*.tmp"))
