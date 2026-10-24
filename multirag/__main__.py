"""Command line entry point: `python -m multirag add <pdf-or-folder>...`.

Chains ingest -> embed -> index, which until now had to be driven from Python by
hand. Every stage is already idempotent, so the CLI adds no state of its own:
re-running `add` over a folder only does the work that is missing.
"""

import argparse
import sys
from pathlib import Path


def _collect_pdfs(paths: list[str]) -> list[Path]:
    """Expand files and folders into a sorted, de-duplicated list of PDFs.

    Sorted so a run over the same folder always processes documents in the same
    order - progress output and partial failures are then reproducible.
    """
    found: set[Path] = set()
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            found.update(p.resolve() for p in path.rglob("*") if p.suffix.lower() == ".pdf")
        elif path.is_file() and path.suffix.lower() == ".pdf":
            found.add(path.resolve())
        else:
            raise SystemExit(f"not a PDF or folder: {raw}")
    return sorted(found)
