"""Central cache paths for downloaded models and generated data."""

from __future__ import annotations

import os
from pathlib import Path


def cache_dir(*subdirs: str) -> Path:
    """Return Mosaic's cache root (or a subdirectory of it) without creating it."""
    root = Path(os.environ.get("MOSAIC_CACHE_DIR", "~/.cache/mosaic")).expanduser()
    return root.joinpath(*subdirs)


def resolve_cache(override, *subdirs: str, create: bool = False) -> Path:
    """Resolve a user-supplied cache path, falling back to a cache subdirectory.

    ``override`` of ``None`` uses ``cache_dir(*subdirs)``; otherwise the override
    is expanded (``~``). With ``create=True`` the directory is created.
    """
    path = cache_dir(*subdirs) if override is None else Path(override).expanduser()
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
