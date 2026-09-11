"""On-disk storage for per-page multi-vectors.

Late-interaction encoders emit a matrix per page, not a single vector: one row
per image patch. That matrix is what MaxSim scores against, so it is stored
whole, one .npy per page, mirroring the layout of data/pages.

Per-page files rather than one array per document keep embedding resumable at
page granularity and let Phase 3 stream a corpus into Qdrant without loading a
whole document at once.
"""

import os
from pathlib import Path

import numpy as np

from multirag import config

# Vectors are stored float16. These are L2-normalized values in [-1, 1], where
# fp16 carries ~3 decimal digits: the MaxSim score shift is far below retrieval
# noise, and it halves disk and read time across a corpus.
STORAGE_DTYPE = np.float16
