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
