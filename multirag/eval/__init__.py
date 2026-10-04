"""Eval: questions with known answer pages in, measured retrieval quality out."""

from multirag.eval.goldset import (
    GoldQuestion,
    GoldSet,
    GoldSetError,
    PageRef,
    load_goldset,
)
from multirag.eval.runner import QuestionResult, RunReport, run_retrieval

__all__ = [
    "GoldQuestion",
    "GoldSet",
    "GoldSetError",
    "PageRef",
    "QuestionResult",
    "RunReport",
    "load_goldset",
    "run_retrieval",
]
