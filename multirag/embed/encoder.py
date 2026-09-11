"""The page encoder: images and queries in, late-interaction multi-vectors out.

Wraps colpali-engine so the rest of the pipeline never imports torch. Everything
crossing this boundary is a numpy array, which keeps Phase 3 and Phase 4 free of
device, dtype, and autograd concerns.
"""


import numpy as np
import torch

from multirag import config
from multirag.embed.identity import EncoderInfo, resolved_revision


def select_device() -> str:
    """Resolve the device to load onto.

    An explicit config.EMBED_DEVICE always wins, including "mps" for anyone who
    wants to retry it. Otherwise CUDA when present, CPU as the honest default.

    MPS is not auto-selected on purpose; see the note in config.EMBED_DEVICE for
    the measurements. Picking a device that hangs or segfaults is worse than
    picking a slow one, and "the GPU is right there" is not a reason to default
    to a backend that does not work.
    """
    if config.EMBED_DEVICE != "auto":
        return config.EMBED_DEVICE
    return "cuda" if torch.cuda.is_available() else "cpu"


def select_dtype(device: str) -> torch.dtype:
    """Pick a compute dtype for the device.

    bfloat16 halves the weight footprint on CUDA and keeps fp32's exponent
    range, so it degrades far more gracefully than fp16 - no loss scaling, no
    inf on an unlucky activation. CPU stays fp32 because bf16 there is emulated
    and slower, not faster. MPS stays fp32 because its bf16 kernels for this
    model do not work at all (see config.EMBED_DEVICE).
    """
    return torch.bfloat16 if device == "cuda" else torch.float32


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
            revision=resolved_revision(model_id, revision),
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
