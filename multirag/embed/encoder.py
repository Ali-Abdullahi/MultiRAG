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


class PageEncoder:
    """A loaded late-interaction encoder, pinned to one model and device.

    Construct with PageEncoder.load(). Loading is expensive - weights, processor,
    and a device transfer - so one instance should embed a whole corpus rather
    than being rebuilt per document.
    """

    def __init__(self, model, processor, info: EncoderInfo):
        self._model = model
        self._processor = processor
        self.info = info

    @classmethod
    def load(
        cls,
        model_id: str | None = None,
        revision: str | None = None,
        device: str | None = None,
    ) -> "PageEncoder":
        """Load weights and processor onto the best available device."""
        from colpali_engine.models import ColIdefics3, ColIdefics3Processor

        model_id = model_id or config.EMBED_MODEL_ID
        revision = revision or config.EMBED_MODEL_REVISION
        device = device or select_device()
        dtype = select_dtype(device)

        model = ColIdefics3.from_pretrained(
            model_id, revision=revision, dtype=dtype, device_map=device
        ).eval()
        processor = ColIdefics3Processor.from_pretrained(model_id, revision=revision)

        info = EncoderInfo(
            model_id=model_id,
            revision=_resolved_revision(model, revision),
            dim=int(model.dim),
            device=device,
            dtype=str(dtype).removeprefix("torch."),
        )
        return cls(model, processor, info)
