"""The single entry point for Phase 2: a rendered document in, vectors out.

Resumability is the design driver here. At ~8s per page on CPU, a 100-page PDF
is a quarter of an hour, and a run that dies at page 90 must not start over.

The manifest is therefore rewritten after every page, not once at the end, and
carries a `complete` flag. That makes it a checkpoint as well as a record: a
resumed run trusts the vectors it already lists, because the manifest also names
the model and render settings they were produced under. Orphan .npy files with
no manifest entry are never trusted - they could be from any model.
"""

from pathlib import Path

from multirag.embed import manifest as embed_manifest
from multirag.embed import store
from multirag.ingest import manifest as render_manifest_mod


class StaleRenderError(RuntimeError):
    """Raised when the page images themselves no longer match the render config."""
