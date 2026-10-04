"""Eval: questions with known answer pages in, measured retrieval quality out."""

from multirag.eval.goldset import (
    GoldQuestion,
    GoldSet,
    GoldSetError,
    PageRef,
    load_goldset,
)

__all__ = [
    "GoldQuestion",
    "GoldSet",
    "GoldSetError",
    "PageRef",
    "load_goldset",
]
