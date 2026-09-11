"""The page encoder: images and queries in, late-interaction multi-vectors out.

Wraps colpali-engine so the rest of the pipeline never imports torch. Everything
crossing this boundary is a numpy array, which keeps Phase 3 and Phase 4 free of
device, dtype, and autograd concerns.
"""

from dataclasses import dataclass

import numpy as np
import torch

from multirag import config


def select_device() -> str:
    """Pick the best available device.

    MPS is Apple's GPU backend and is the difference between seconds and minutes
    per page on this machine. CUDA is here so the same code runs unchanged on a
    Colab box; CPU is the honest fallback.
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def select_dtype(device: str) -> torch.dtype:
    """Pick a compute dtype for the device.

    bfloat16 halves the weight footprint, which matters on 8GB of unified memory
    shared with the OS. It keeps fp32's exponent range, so it degrades far more
    gracefully than fp16 - no rescaling, no inf on an unlucky activation. CPU
    stays fp32 because bf16 on CPU is emulated and slower, not faster.
    """
    return torch.float32 if device == "cpu" else torch.bfloat16


@dataclass(frozen=True)
class EncoderInfo:
    """Identity of the model that produced a set of vectors.

    Recorded alongside the vectors for the same reason RENDER_DPI is recorded
    alongside the images: a query encoded by a different model lands in a
    different space, and the failure mode is bad rankings, not an exception.
    """

    model_id: str
    revision: str
    dim: int
    device: str
    dtype: str


def _resolved_revision(model, requested: str) -> str:
    """The commit actually loaded, not the tag that was asked for.

    transformers records the resolved commit on the config, so this needs no
    extra network call and stays correct offline. Falls back to the requested
    string when unavailable - better an honest "main" than a fabricated sha.
    """
    return getattr(model.config, "_commit_hash", None) or requested
