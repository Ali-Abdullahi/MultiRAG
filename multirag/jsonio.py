"""Atomic JSON reads and writes.

Every stage records what it produced in a JSON sidecar, and every one of them
needs the same guarantee: a crash mid-write must leave the previous file intact,
never a truncated one. A half-written manifest that still parses is the worst
outcome - it reports fewer pages than exist and nothing errors.
"""

import json
import os
from pathlib import Path


def write_atomic(path: Path, payload: dict) -> Path:
    """Write JSON via temp file, fsync, and rename. Returns the path written.

    os.replace is atomic on POSIX, so a reader either sees the whole old file or
    the whole new one. fsync before the rename is what makes that survive a
    power loss rather than just a process crash.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")

    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)

    return path


def read_or_none(path: Path) -> dict | None:
    """Load JSON, or None if the file does not exist.

    Absence is a normal answer across this pipeline - it means a stage has not
    run yet - so it is not an error.
    """
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
