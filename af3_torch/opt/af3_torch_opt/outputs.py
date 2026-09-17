"""The one file hasher of the package (the parameters digest, the tree tests, the carry gate)."""
from __future__ import annotations

from opt_core.gates import sha256_file as sha256   # noqa: F401 — the one file hasher (the tree tests and the carry gate read it from here)
