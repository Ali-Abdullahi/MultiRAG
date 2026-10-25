"""Phase 7: gold set validation, metrics, the weak-match check, saved-run comparison."""

import json
from pathlib import Path

import pytest

from multirag import config
from multirag.eval import GoldSetError, load_goldset, metrics, store
from multirag.eval.runner import QuestionResult, RunReport
from multirag.ingest import ingest_pdf


def write_goldset(questions, name="g"):
    config.EVALSETS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.EVALSETS_DIR / f"{name}.json"
    path.write_text(json.dumps({"schema_version": 1, "questions": questions}))
    return path


def q(qid, page=1, source="doc.pdf", **extra):
    return {"id": qid, "question": f"question {qid}?", "relevant": [{"source_name": source, "page": page}], **extra}


@pytest.fixture(autouse=True)
def corpus(make_pdf):
    ingest_pdf(make_pdf())  # doc.pdf, 3 pages


def test_page_references_resolve_to_page_ids():
    write_goldset([q("a", page=2)])
    [question] = load_goldset("g").questions
    assert len(question.relevant) == 1
    assert next(iter(question.relevant)).endswith("-p0001")  # 1-based in, 0-based id


@pytest.mark.parametrize(
    "questions, message",
    [
        ([q("a", source="missing.pdf")], "not ingested"),
        ([q("a", page=9)], "out of range"),
        ([q("a"), q("a")], "duplicate question id"),
        ([{"id": "a", "question": "x?"}], "no relevant pages"),
        ([{"id": "a", "question": "x?", "answerable": False,
           "relevant": [{"source_name": "doc.pdf", "page": 1}]}], "unanswerable but lists"),
    ],
)
def test_bad_references_fail_at_load_instead_of_scoring_zero(questions, message):
    # A silent zero is indistinguishable from a retriever that got it wrong.
    write_goldset(questions)
    with pytest.raises(GoldSetError, match=message):
        load_goldset("g")


def test_editing_a_gold_set_changes_its_fingerprint():
    write_goldset([q("a")])
    before = load_goldset("g").fingerprint
    write_goldset([q("a", page=2)])
    assert load_goldset("g").fingerprint != before


def test_metrics():
    retrieved = ["p1", "p2", "p3"]
    assert metrics.rank_of_first_relevant(retrieved, {"p2"}, 3) == 2
    assert metrics.rank_of_first_relevant(retrieved, {"p9"}, 3) is None
    assert metrics.reciprocal_rank(retrieved, {"p3"}, 3) == pytest.approx(1 / 3)
    assert metrics.recall_at_k(retrieved, {"p1", "p9"}, 3) == 0.5
    assert metrics.rank_of_first_relevant(retrieved, {"p3"}, 2) is None  # beyond k


def result(qid, rank, match, answerable=True):
    return QuestionResult(
        qid, "?", frozenset(), (), (), rank, 1.0 if rank else 0.0,
        1 / rank if rank else 0.0, answerable, match,
    )


def report(results, created="2026-10-25T10:00:00+00:00", **identity):
    fields = dict(goldset_name="g", goldset_fingerprint="fp", k=5, model_id="m",
                  model_revision="r", collection="pages", corpus_pages=208)
    fields.update(identity)
    return RunReport(created_at=created, results=tuple(results), **fields)


def test_unanswerable_questions_stay_out_of_ranking_metrics():
    run = report([result("a", 1, 0.7), result("b", None, 0.6), result("x", None, 0.4, answerable=False)])
    assert run.hit_rate == 0.5  # 1 of the 2 answerable, not 1 of 3
    assert [r.qid for r in run.misses] == ["b"]


def test_weak_match_check_counts_both_kinds_of_failure():
    run = report([
        result("a", 1, 0.70), result("b", 1, 0.50),             # b: false alarm
        result("x", None, 0.40, False), result("y", None, 0.60, False),  # y: missed
    ])
    check = run.weak_match_check(threshold=0.53)
    assert (check["answerable_warned"], check["unanswerable_missed"]) == (1, 1)


def test_saved_runs_list_in_order_and_compare_by_question():
    store.save_run(report([result("a", 3, 0.6)], created="2026-10-25T10:00:00+00:00"))
    store.save_run(report([result("a", 1, 0.6)], created="2026-10-25T11:00:00+00:00", model_id="m2"))

    old, new = (store.load_run(p) for p in store.list_runs("g"))
    diff = store.compare(old, new)
    assert diff["variables"] == ["encoder: m -> m2"]  # the experiment, not a warning
    assert diff["invalidating"] == []
    assert diff["rank_changes"] == [("a", 3, 1)]


def test_comparison_flags_what_makes_deltas_meaningless():
    old = store.report_record(report([result("a", 1, 0.6)]))
    new = store.report_record(report([result("a", 1, 0.6)], corpus_pages=300, goldset_fingerprint="fp2"))
    flagged = store.compare(old, new)["invalidating"]
    assert any("gold set edited" in line for line in flagged)
    assert any("corpus size" in line for line in flagged)


def test_runs_are_saved_under_data_not_beside_the_code(isolated_data):
    path = store.save_run(report([result("a", 1, 0.6)]))
    # Runs repeat every question; they are as private as the documents.
    assert Path(path).is_relative_to(isolated_data)
