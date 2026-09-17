"""Where a kit lives: the engine directory (tree home), its ``opt/`` directory and the kit files beside the package.

Contract. A kit package sits at ``<engine>/opt/<pkg>/``; the kit's carried files sit beside it under ``<engine>/opt/<class>/<kit>/``
and its pins under ``<engine>/stock/``. :func:`tree_home` resolves the engine directory in this order: the kit's own home variable
(``env_home``, a wheel installed into site-packages with no kit beside it), the tree convention ``MODEL_OPT`` (the engine directory,
the variable ``run.sh`` exports), else ``levels`` directories above the package file. When ``require`` names entries the home must
contain and they are absent, the resolution is a named refusal (:class:`HomeError` says which variables to set) — never a wrong
directory. :func:`place_on_sys_path` puts kit directories on ``sys.path`` idempotently — an entry already present is moved to the
requested position, never duplicated.
"""
from __future__ import annotations

import os
import sys
from typing import Iterable, Mapping, Optional, Sequence

ENV_TREE = "MODEL_OPT"


class HomeError(RuntimeError):
    """The engine directory could not be resolved to a directory with the required entries."""


def tree_home(package_file: str, *, env_tree: str = ENV_TREE, env_home: Optional[str] = None, levels: int = 2,
              require: Iterable[str] = (), environ: Optional[Mapping[str, str]] = None) -> str:
    """The engine directory for the package whose ``__file__`` is ``package_file`` (see the module contract for the precedence)."""
    environ = os.environ if environ is None else environ
    if env_home and environ.get(env_home):
        home, how = os.path.abspath(environ[env_home]), f"${env_home}"
    elif env_tree and environ.get(env_tree):
        home, how = os.path.abspath(environ[env_tree]), f"${env_tree}"
    else:
        d = os.path.dirname(os.path.abspath(package_file))
        for _ in range(levels):
            d = os.path.dirname(d)
        home, how = d, f"{levels} levels above the package"
    missing = [r for r in require if not os.path.exists(os.path.join(home, r))]
    if missing:
        variables = " or ".join(v for v in (env_home, env_tree) if v)
        raise HomeError(f"engine directory {home} ({how}) lacks {', '.join(missing)}: set {variables} to the engine directory")
    return home


def opt_home(package_file: str, **kw) -> str:
    """``<tree_home>/opt``."""
    return os.path.join(tree_home(package_file, **kw), "opt")


def kit_path(tree: str, *rel: str) -> str:
    return os.path.join(tree, *rel)


def place_on_sys_path(entries: Sequence[str], *, after: Optional[str] = None, path: Optional[list] = None) -> list:
    """Put ``entries`` (absolute paths, in order) on ``sys.path`` — at the front, or right after ``after`` when that entry is present.
    An entry already on the path is moved to the position. Returns the path list (``sys.path`` itself unless ``path`` is given)."""
    path = sys.path if path is None else path
    entries = [os.path.abspath(e) for e in entries]
    for e in entries:
        while e in path:
            path.remove(e)
    at = 0
    if after is not None and after in path:
        at = path.index(after) + 1
    for i, e in enumerate(entries):
        path.insert(at + i, e)
    return path
