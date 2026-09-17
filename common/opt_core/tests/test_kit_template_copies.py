"""Every kit's ``_build_backend.py`` and ``_core_gate.py`` is byte-identical to ``common/opt_core/kit_template``'s copy — a live
read-and-compare, no digest: git already identifies the bytes, so a drifted kit is caught by diffing against the template file
itself, not against a restated hash of it. One test per kit copy found (named by its path), so a drifted kit is named directly
in the failure; skips by name when no sibling kit ``opt/`` directories are checked out beside ``common/`` (e.g. an sdist-only
install of this package alone).

This is the one shared test for the property: every kit relies on this file for it, and carries no per-kit
``test_*_core_gate_matches_template`` (or similarly-named sha256 comparison) test of its own.
"""
import glob
import os

import pytest

CORE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))            # common/opt_core
RELEASE_DIR = os.path.dirname(os.path.dirname(CORE_DIR))                          # .../model-opt-release
TEMPLATE_DIR = os.path.join(CORE_DIR, "kit_template")
TEMPLATE_NAMES = ("_build_backend.py", "_core_gate.py")


def _kit_copies():
    """(template_name, kit_copy_path) for every matching file under ``<release>/*/opt/**``, sorted by path."""
    out = []
    for name in TEMPLATE_NAMES:
        out += [(name, p) for p in glob.glob(os.path.join(RELEASE_DIR, "*", "opt", "**", name), recursive=True)]
    return sorted(out, key=lambda t: t[1])


_COPIES = _kit_copies()

if not _COPIES:
    def test_kit_copies_present():
        pytest.skip("no sibling kit opt/ directories checked out beside common/ (e.g. an sdist-only install of opt_core alone)")
else:
    @pytest.mark.parametrize("name,path", _COPIES, ids=[os.path.relpath(p, RELEASE_DIR) for _, p in _COPIES])
    def test_kit_copy_matches_template(name, path):
        with open(os.path.join(TEMPLATE_DIR, name), "rb") as fh:
            want = fh.read()
        with open(path, "rb") as fh:
            got = fh.read()
        assert got == want, f"{path} has drifted from kit_template/{name}"
