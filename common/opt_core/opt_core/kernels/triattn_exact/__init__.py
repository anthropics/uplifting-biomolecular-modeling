"""triattn_exact — bit-identical triangle attention (cuEquivariance-exact tier).

Public face: triattn_exact.face.triangle_attention (library signature and semantics).
Routes live in subpackages (cuda_mma, hopper); each exposes attention(...) and supports(...).
"""
from __future__ import annotations


class Refused(RuntimeError):
    """Raised BY NAME whenever an input falls outside a proven (version, device, shape-class, env) cell.
    Never a silent fallback: the caller decides what to do (e.g. call the library)."""

    def __init__(self, reason: str, cell: dict | None = None):
        self.reason = reason
        self.cell = cell or {}
        super().__init__(f"triattn_exact refused: {reason} cell={self.cell}")
