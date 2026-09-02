"""Content-addressed identity for documents and pages.

Identity derives from file bytes, never from filename or path. The same PDF
uploaded twice under different names collapses to one document, and re-running
ingestion overwrites the same records instead of accumulating duplicates.
"""

import hashlib
from pathlib import Path

# 16 hex chars = 64 bits of the digest. Collision risk is negligible at
# document-corpus scale, and short ids keep paths and log lines readable.
DOC_ID_LENGTH = 16

# Hash in chunks so a 500MB scanned PDF never lands in memory all at once.
_CHUNK_BYTES = 1024 * 1024


def sha256_file(path: Path) -> str:
    """Return the full hex sha256 of a file, read incrementally."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()
