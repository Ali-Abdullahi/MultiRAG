"""Encoder identity, deliberately free of torch.

Which model produced a set of vectors is a metadata question, not a compute one.
Keeping it in its own module means Phase 3 can read manifests and push vectors
into Qdrant on a machine with no torch installed at all - only embedding itself
needs the heavy stack.
"""

from dataclasses import dataclass
from pathlib import Path


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


def resolved_revision(model_id: str, requested: str) -> str:
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
