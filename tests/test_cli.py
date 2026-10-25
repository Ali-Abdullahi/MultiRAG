"""The CLI's input handling, and a guard on what the suite imports."""

import subprocess
import sys

import pytest

from multirag.__main__ import _collect_pdfs, main


def test_files_and_folders_expand_to_sorted_unique_pdfs(make_pdf, tmp_path):
    a = make_pdf("a.pdf")
    nested = tmp_path / "sub" / "deeper"
    nested.mkdir(parents=True)
    b = make_pdf("sub/deeper/b.PDF")  # case-insensitive suffix
    (tmp_path / "notes.txt").write_text("not a pdf")

    found = _collect_pdfs([str(tmp_path), str(a)])  # a.pdf named twice
    assert found == sorted({a.resolve(), b.resolve()})


def test_a_path_that_is_not_a_pdf_is_refused(tmp_path):
    text = tmp_path / "notes.txt"
    text.write_text("x")
    with pytest.raises(SystemExit, match="not a PDF or folder"):
        _collect_pdfs([str(text)])


def test_eval_of_a_missing_gold_set_fails_fast(capsys):
    assert main(["eval", "does-not-exist"]) == 1
    assert "no gold set" in capsys.readouterr().err


def test_compare_needs_two_runs(capsys):
    assert main(["compare", "nothing-saved"]) == 1
    assert "need at least 2" in capsys.readouterr().out


def test_nothing_past_the_encoder_imports_torch():
    # Indexing, retrieval plumbing, eval and generation are meant to run without
    # torch installed. Checked in a fresh interpreter: this test process may
    # have imported torch for reasons of its own.
    code = (
        "import sys; import multirag.index, multirag.retrieve, multirag.eval, multirag.generate, "
        "multirag.embed, multirag.__main__; print('torch' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
