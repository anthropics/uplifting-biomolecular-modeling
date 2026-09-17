"""The add-on's install() on a part without a tile table: the requested kernels are HELD by name (patch.HELD, status()['held']), their
classes stay stock, nothing raises — so the package import completes and the hoist installs. Needs the add-on's imports (haiku, jax)."""
import os
import sys
import types

import pytest

pytest.importorskip("haiku"); pytest.importorskip("jax")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "forward", "flashpairformer"))
os.environ["AF3_FLASHPAIRFORMER"] = "off"                                   # import the package without installing (no alphafold3 here)
from af3_flashpairformer import patch  # noqa: E402


class StockTM: pass
class StockGA: pass


def test_no_tiles_holds_each_requested_kernel_by_name(monkeypatch):
    modules = types.SimpleNamespace(TriangleMultiplication=StockTM, GridSelfAttention=StockGA, __name__="stub_modules")
    monkeypatch.setattr(patch, "NO_TILES", patch.S.Refusal(patch.S.NO_TILES, "no-tiles:8.0 (tile tables: ['10.0', '9.0'])"))
    monkeypatch.setattr(patch, "CC", "8.0")
    patch.STATE.update(modules=None, stock_tm=None, stock_ga=None)
    assert patch.install("both", modules=modules) == "both"
    assert modules.TriangleMultiplication is StockTM and modules.GridSelfAttention is StockGA          # nothing rebound
    assert patch.HELD == {"trimul": "fallback:no_tiles_cc80(trimul)", "triatt": "fallback:no_tiles_cc80(triattn)"} and patch.status()["held"] == patch.HELD
    assert patch.install("trimul", modules=modules) == "trimul" and patch.HELD == {"trimul": "fallback:no_tiles_cc80(trimul)"}   # per kernel: only the requested one is held
    patch.uninstall(); assert patch.HELD == {} and patch.status()["mode"] == "off"
