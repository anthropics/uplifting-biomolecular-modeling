"""Digests of things this repo does NOT track: a downloaded model weight file, a delivery tarball, any external archive fetched
at install or run time. For an out-of-git artifact ONLY -- never for a file or directory this repo already tracks. Equality
inside the repo is the git commit (see ``opt_core.kernels``, ``opt_core.gates``): a digest of repo-tracked content restates
what git already identifies, and is the pattern this module exists to replace, not extend. A kit needing to fingerprint one of
its own out-of-git artifacts (a fetched weights file, an extracted delivery tree) uses these, rather than writing its own
sha256 walk -- one implementation, every kit's digest of the same shape of thing computed the same way.
"""
import fnmatch
import hashlib
import os
from typing import Iterable, List, Optional

from .gates import sha256_file as digest_file  # the one chunked-sha256-of-a-file implementation; re-exported under this module's name

IGNORED_DIRS = ("__pycache__", "*.egg-info", ".venv", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".ipynb_checkpoints", ".idea", ".vscode",
                "do_not_commit", ".git")
IGNORED_FILES = ("*.py[cod]", ".python-version", ".env", ".env.*", "*.pem", "*.key", ".DS_Store")


def tree_files(root: str) -> List[str]:
    """The sorted relative paths of every file under ``root``, ``IGNORED_DIRS``/``IGNORED_FILES`` pruned (build caches, VCS
    metadata, OS litter -- never content that would change what the artifact IS)."""
    out = []
    for d, dns, fns in os.walk(root):
        dns[:] = sorted(x for x in dns if not any(fnmatch.fnmatchcase(x, p) for p in IGNORED_DIRS))
        out += [os.path.relpath(os.path.join(d, f), root) for f in fns if not any(fnmatch.fnmatchcase(f, p) for p in IGNORED_FILES)]
    return sorted(out)


def digest_tree(root: str, files: Optional[Iterable[str]] = None) -> str:
    """sha256 (hex) of a whole directory tree: every file's ``digest_file`` result paired with its relative path, as sorted
    ``<sha256>  <relpath>`` lines, joined and hashed once more -- so the tree digest changes if any file's bytes OR the file
    set itself changes. ``files`` restricts the digest to that subset of relative paths (default: every file under ``root``,
    via ``tree_files``)."""
    paths = sorted(files) if files is not None else tree_files(root)
    lines = [f"{digest_file(os.path.join(root, p))}  {p}" for p in paths]
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()
