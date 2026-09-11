"""The page encoder: images and queries in, late-interaction multi-vectors out.

Wraps colpali-engine so the rest of the pipeline never imports torch. Everything
crossing this boundary is a numpy array, which keeps Phase 3 and Phase 4 free of
device, dtype, and autograd concerns.
"""

from dataclasses import dataclass
from pathlib import Path

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


def _resolved_revision(model_id: str, requested: str) -> str:
    """The commit of the adapter repo actually on disk, not the tag asked for.

    Deliberately not model.config._commit_hash: colSmol is a LoRA adapter over a
    separate base repo, and the loaded config carries the *base* model's commit.
    Recording that would pin the wrong repo - the adapter holds the projection
    head, so it is the adapter's commit that decides what space vectors land in.

    The HF cache stores snapshots under .../snapshots/<sha>/, so the resolved
    commit can be read from the path with no network call. Falls back to the
    requested string - an honest "main" beats a fabricated sha.
    """
    from huggingface_hub import try_to_load_from_cache

    for filename in ("adapter_config.json", "config.json"):
        cached = try_to_load_from_cache(model_id, filename, revision=requested)
        if isinstance(cached, str):
            parts = Path(cached).parts
            if "snapshots" in parts:
                return parts[parts.index("snapshots") + 1]
    return requested


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
            revision=_resolved_revision(model_id, revision),
            dim=int(model.dim),
            device=device,
            dtype=str(dtype).removeprefix("torch."),
        )
        return cls(model, processor, info)

    @torch.no_grad()
    def _encode(self, batch) -> list[np.ndarray]:
        """Run one prepared batch and split it back into per-item matrices.

        Padded positions are dropped using the attention mask. This matters more
        than it looks: MaxSim takes a max over patches, so a leftover pad vector
        can win that max and inflate a score with a position that does not exist.

        bfloat16 has no numpy equivalent, so results are cast to float32 on the
        way out - which is also the dtype every downstream consumer wants.
        """
        batch = {key: value.to(self._model.device) for key, value in batch.items()}
        embeddings = self._model(**batch).to(torch.float32).cpu().numpy()

        mask = batch.get("attention_mask")
        if mask is None:
            return list(embeddings)
        keep = mask.cpu().numpy().astype(bool)
        return [item[rows] for item, rows in zip(embeddings, keep)]

    def encode_pages(self, images) -> list[np.ndarray]:
        """Embed page images. Returns one (patches, dim) matrix per image."""
        return self._encode(self._processor.process_images(images))

    def encode_queries(self, queries: list[str]) -> list[np.ndarray]:
        """Embed text queries. Returns one (tokens, dim) matrix per query.

        Queries go through a different processor path than images - the model is
        trained with distinct prefixes for each - so Phase 4 must call this, not
        encode_pages, or the two sides land in mismatched spaces.
        """
        return self._encode(self._processor.process_queries(queries))
