"""Saved eval runs, and the rules for comparing two of them.

A metric is a number about one gold set, at one k, from one encoder, over one
corpus. Change any of those and the number moves for reasons that have nothing
to do with retrieval quality. So every run is saved with all four, and a
comparison names any that differ instead of presenting the deltas as progress.

One JSON file per run under data/evals/<goldset>/, named by its UTC timestamp
so a directory listing is already in chronological order.
"""

from datetime import datetime
from pathlib import Path

from multirag import config, jsonio
from multirag.eval.runner import RunReport

SCHEMA_VERSION = 1

# Ordered as printed. Each is a RunReport property.
METRICS = ("top1_rate", "hit_rate", "mean_recall", "mrr")


def _run_dir(goldset_name: str) -> Path:
    return config.EVAL_RUNS_DIR / goldset_name


def report_record(report: RunReport) -> dict:
    """Serialize a run. Keeps what each question retrieved, not just its score:
    a regression is only actionable if you can see what took the right page's place."""
    return {
        "schema_version": SCHEMA_VERSION,
        "goldset": report.goldset_name,
        "goldset_fingerprint": report.goldset_fingerprint,
        "k": report.k,
        "model_id": report.model_id,
        "model_revision": report.model_revision,
        "collection": report.collection,
        "corpus_pages": report.corpus_pages,
        "created_at": report.created_at,
        "saturated": report.saturated,
        "metrics": {name: getattr(report, name) for name in METRICS},
        "questions": [
            {
                "qid": r.qid,
                "rank": r.rank,
                "recall": r.recall,
                "reciprocal_rank": r.reciprocal_rank,
                "expected": [ref.citation for ref in r.refs],
                "retrieved": [hit.citation for hit in r.hits],
            }
            for r in report.results
        ],
    }


def save_run(report: RunReport) -> Path:
    """Write a run atomically. Returns its path."""
    stamp = datetime.fromisoformat(report.created_at).strftime("%Y%m%dT%H%M%SZ")
    path = _run_dir(report.goldset_name) / f"{stamp}.json"
    return jsonio.write_atomic(path, report_record(report))


def list_runs(goldset_name: str) -> list[Path]:
    """Saved runs for a gold set, oldest first."""
    return sorted(_run_dir(goldset_name).glob("*.json"))


def load_run(path: Path) -> dict:
    record = jsonio.read_or_none(path)
    if record is None:
        raise FileNotFoundError(path)
    return record
