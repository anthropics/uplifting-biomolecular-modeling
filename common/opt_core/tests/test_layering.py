"""Layering: the opt_core ROOT never imports a sub-namespace. The sub-namespaces — ``opt_core.seq`` (sequence-model kits),
``opt_core.jax_design`` / ``opt_core.diffusion_loop`` / ``opt_core.host_cache`` (design kits) — import the root; the reverse edge would
make installing one engine family require another's code paths and would break the rule that a core minor which touches only one
sub-namespace leaves every other consumer's bytes unchanged. Held statically over the source tree (AST of every import, relative
imports resolved), no framework needed."""
from __future__ import annotations

import ast
import os

import opt_core

SUB_NAMESPACES = ("seq", "jax_design", "host_cache", "diffusion_loop")
PKG = os.path.dirname(opt_core.__file__)


def _reverse_edges():
    hits = []
    for dp, _dn, fn in os.walk(PKG):
        rel = [x for x in os.path.relpath(dp, PKG).split(os.sep) if x not in (".", "")]
        if rel and rel[0] in SUB_NAMESPACES:
            continue                                        # a sub-namespace may import anything of the root
        if "__pycache__" in rel:
            continue
        for f in fn:
            if not f.endswith(".py"):
                continue
            p = os.path.join(dp, f)
            tree = ast.parse(open(p, encoding="utf-8").read(), p)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        parts = a.name.split(".")
                        if parts[0] == "opt_core" and len(parts) > 1 and parts[1] in SUB_NAMESPACES:
                            hits.append(f"{os.path.relpath(p, PKG)}:{node.lineno}: import {a.name}")
                elif isinstance(node, ast.ImportFrom):
                    names = [n.name for n in node.names]
                    if node.level == 0:
                        target = (node.module or "").split(".")
                    else:
                        base = ["opt_core"] + rel
                        up = node.level - 1
                        target = base[:len(base) - up] + ((node.module or "").split(".") if node.module else [])
                    if target and target[0] == "opt_core":
                        if len(target) > 1 and target[1] in SUB_NAMESPACES:
                            hits.append(f"{os.path.relpath(p, PKG)}:{node.lineno}: from {'.'.join(target)} import {names}")
                        elif len(target) == 1 and any(n in SUB_NAMESPACES for n in names):
                            hits.append(f"{os.path.relpath(p, PKG)}:{node.lineno}: from opt_core import {names}")
    return hits


def test_the_root_never_imports_a_sub_namespace():
    hits = _reverse_edges()
    assert hits == [], "root -> sub-namespace imports (sub-namespaces import the root, never the reverse):\n" + "\n".join(hits)


def test_the_scan_sees_a_planted_reverse_edge(tmp_path, monkeypatch):
    # the scanner is live: a root module importing opt_core.seq is reported (planted in a copy of the layout, not in the package)
    import shutil
    fake = tmp_path / "opt_core"
    shutil.copytree(PKG, str(fake), ignore=shutil.ignore_patterns("__pycache__", "*.so", "*.cu", "*.cpp"))
    (fake / "planted.py").write_text("from .seq import det_torch\nimport opt_core.jax_design\n")
    import sys
    monkeypatch.setattr(sys.modules[__name__], "PKG", str(fake))
    hits = _reverse_edges()
    assert any(h.startswith("planted.py:1:") for h in hits) and any(h.startswith("planted.py:2:") for h in hits), hits
