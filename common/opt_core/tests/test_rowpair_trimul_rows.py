"""opt_core.kernels.fpf_trimul_rows + the row-block provider's width gate (opt_core.mem.rowpair.trimul_fused) — CPU tests.

* the rows-only table is well-formed; its 9.0 and 8.0 (384,384) / (64,64) keys are MEASURED (an H100 sweep / the A100 cells pass); the (256,256)
  keys are floor-only entries; a card without a key (e.g. cc 12.0) keeps the decline word;
* the ONE selection statement: measured -> served; an unmeasured (PLACEHOLDER) key -> not served (the provider's decline word ``c=<Cz>/<Ch>`` — unchanged
  from before the unit existed) unless ``ROWPAIR_TRIMUL_ROWS_UNMEASURED=1`` -> served:unmeasured; no key -> no_row, a v4 width -> v4, a width outside the
  kernels' -> width;
* the provider's ladder on a (simulated) device: cc 12.0 C 384 declines ``c=384/384`` with fact ``rows_table=-:no_row``; cc 9.0 / 8.0 C 384 are served (the gate
  opens: fact ``rows_table=<cc>|C384|H384:served``); the size floor of C 256 on cc 9.0 / 8.0 is the table's 0 (fact ``min_tokens_source``), a kit's explicit
  ``min_tokens=`` wins, C 128 keeps 2048, C 384 reads 1400;
* the driver seam ``b_transposed`` (``_b_plane``): a provider answering True gets the GEMM-B layout view and the update is byte-identical to the plain one;
* the kernels' host-side layout words / refusals (no launch), the width rule (chunking);
* under triton's INTERPRETER (subprocess, CPU tensors): ``_k1r`` == fpf_trimul_v4 ``_k1c`` and ``_k3r`` == ``_k3c`` BIT FOR BIT at one channel chunk
  (C 64 / 128), and at C 384 the a-layout == the b-layout planes and the in-place window epilogue == the epilogue on a contiguous copy (addressing);
* invariance: the whole-plane tables (fpf_trimul_v4/table.json, trimul/TRIMUL_CELLS.json) are byte-identical to their pinned digests and no
  whole-plane provider source names the rows unit or its table.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import textwrap

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.dirname(HERE)
if CORE not in sys.path:
    sys.path.insert(0, CORE)

torch = pytest.importorskip("torch")

from opt_core.kernels import fpf_trimul_rows as R                            # noqa: E402
from opt_core.mem.rowpair import RowpairRefused                             # noqa: E402
from opt_core.mem.rowpair import trimul as TM, trimul_fused as RF           # noqa: E402
from opt_core.mem.rowpair.dist import Layout, all_gather_rows               # noqa: E402
from opt_core.testing import run_ranks                                      # noqa: E402

KDIR = os.path.join(CORE, "opt_core", "kernels")
# sha256 of the WHOLE-PLANE tables the rows members must never touch (rule: no single-GPU TriMul resolution may change). A release
# owner who edits either file on purpose updates the digest here in the same commit — the pin exists so an edit is never incidental to the rows unit.
PINNED = {
    "fpf_trimul_v4/table.json": None,
    "trimul/TRIMUL_CELLS.json": None,
}
PINS_PATH = os.path.join(HERE, "fixtures", "rows_untouched_tables.json")


def _sha(rel):
    with open(os.path.join(KDIR, rel), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# ----------------------------------------------------------------------------------------------------------------- the table + selection
def test_rows_table_is_well_formed_and_the_keys_are_measured():
    t = R.load_table()
    assert t["schema"] == R.SCHEMA and R.validate(t) == [], R.validate(t)
    keys = R.keys(t)
    assert {"9.0|C384|H384", "9.0|C64|H64", "9.0|C256|H256", "8.0|C384|H384", "8.0|C64|H64", "8.0|C256|H256"} <= set(keys), keys
    assert not any(k.startswith("12.0|") for k in keys), keys                                  # an unmeasured card keeps declining by name
    want = {"9.0|C384|H384": ({"BM": 64, "BN": 128, "num_warps": 4, "num_stages": 1}, {"BM": 64, "BN": 128, "num_warps": 4, "num_stages": 1},
                              {"BM": 64, "BN": 128, "num_warps": 8, "num_stages": 1}, 1400),
            "9.0|C64|H64": ({"BM": 128, "BN": 32, "num_warps": 4, "num_stages": 2}, {"BM": 64, "BN": 32, "num_warps": 4, "num_stages": 2},
                            {"BM": 128, "BN": 32, "num_warps": 4, "num_stages": 1}, 2048),
            "8.0|C384|H384": ({"BM": 64, "BN": 64, "num_warps": 4, "num_stages": 1}, {"BM": 64, "BN": 64, "num_warps": 4, "num_stages": 1},
                              {"BM": 64, "BN": 128, "num_warps": 8, "num_stages": 1}, 1400),
            "8.0|C64|H64": ({"BM": 128, "BN": 32, "num_warps": 4, "num_stages": 1}, {"BM": 128, "BN": 32, "num_warps": 4, "num_stages": 1},
                            {"BM": 128, "BN": 32, "num_warps": 4, "num_stages": 1}, 2048)}
    box = {"9.0": ("run-A", "0ff63df1b970"), "8.0": ("run-B", "a10892493479")}
    for k, (k1, k1b, k3, floor) in want.items():
        e = t["cells"][k]
        assert e["kernels"] == R.KERNELS_WORD and e["measured"] is True and e["status"].startswith("MEASURED_OP") and R.is_measured(e), e["status"]
        assert R.tiles(e) == (k1, k1b, k3) and R.b_direct(e) is True and e["min_tokens"] == floor
        sb, tree = box[k.split("|")[0]]
        assert sb in e["evidence"] and "triton 3.6.0" in e["evidence"] and tree in e["evidence"]                  # provenance: box / stack / tree
    for cc in ("9.0", "8.0"):
        f = t["cells"][cc + "|C256|H256"]
        assert f["kernels"] == R.V4_WORD and "k1" not in f and f["min_tokens"] == 0 and f["measured"] is True  # floor-only: the v4 widths keep their own tiles
        assert R.safe_cells(t, cc)["k1"] == {"BM": 32, "BN": 32, "num_warps": 4, "num_stages": 1}
    assert R.safe_cells(t, "12.0") == t["safe"]["*"] and R.safe_cells({"safe": {}}, "9.0") is None


def test_select_is_the_one_statement(monkeypatch):
    t = R.load_table()
    monkeypatch.delenv(R.ENV_UNMEASURED, raising=False)
    s = R.select(t, "9.0", 384, 384)
    assert (s.served, s.reason, s.key) == (True, "served", "9.0|C384|H384") and R.tiles(s.cell)[2] == {"BM": 64, "BN": 128, "num_warps": 8, "num_stages": 1}
    assert R.select(t, (9, 0), 64, 64)[:2] == (True, "served")
    s8 = R.select(t, "8.0", 384, 384)
    assert (s8.served, s8.reason, s8.key) == (True, "served", "8.0|C384|H384") and R.tiles(s8.cell)[0] == {"BM": 64, "BN": 64, "num_warps": 4, "num_stages": 1}
    assert R.select(t, "8.0", 64, 64)[:2] == (True, "served")
    assert R.select(t, "12.0", 384, 384)[:3] == (False, "no_row", None)                     # no key on this card: the decline word, unchanged
    assert R.select(t, "12.0", 64, 64).reason == "no_row" and R.select(t, "10.0", 384, 384).reason == "no_row"
    assert R.select(t, "9.0", 256, 256).reason == "v4"                                       # a v4 width is never this unit's
    assert R.select(t, "9.0", 64, 128).reason == "no_row" and R.select(t, "9.0", 512, 512).reason == "width" and R.select(t, "9.0", 32, 32).reason == "width"
    # an UNMEASURED (placeholder) key — the state every new (cc, C) starts in — declines unless the engineering switch admits it
    ph = {"schema": R.SCHEMA, "cells": {"12.0|C384|H384": dict(t["cells"]["9.0|C384|H384"], measured=False, status="PLACEHOLDER (test)")}}
    assert R.select(ph, "12.0", 384, 384)[:2] == (False, "unmeasured")
    assert R.select(ph, "12.0", 384, 384, allow_unmeasured=True)[:2] == (True, "served:unmeasured")
    monkeypatch.setenv(R.ENV_UNMEASURED, "1")
    assert R.select(ph, "12.0", 384, 384)[:2] == (True, "served:unmeasured")                # the switch, read when the caller passes no override
    assert R.select(ph, "12.0", 384, 384, allow_unmeasured=False).reason == "unmeasured"
    monkeypatch.delenv(R.ENV_UNMEASURED)
    contradiction = {"schema": R.SCHEMA, "cells": {"12.0|C384|H384": dict(t["cells"]["9.0|C384|H384"], measured=True, status="PLACEHOLDER (test)")}}
    assert R.select(contradiction, "12.0", 384, 384).reason == "unmeasured" and R.validate(contradiction) != []   # measured=true with a PLACEHOLDER status is not measured
    assert R.select({"cells": {"9.0|C384|H384": {"kernels": R.KERNELS_WORD, "measured": True, "status": "X"}}}, "9.0", 384, 384).reason == "no_row"   # a key without tiles serves nothing
    assert R.min_tokens(t, "9.0", 256, 256, 2048) == 0 and R.min_tokens(t, "8.0", 256, 256, 2048) == 0 and R.min_tokens(t, "12.0", 256, 256, 2048) == 2048
    assert R.min_tokens(t, "9.0", 128, 128, 2048) == 2048 and R.min_tokens(t, "9.0", 384, 384, 2048) == 1400 and R.min_tokens(t, "9.0", 64, 64, 999) == 2048
    assert R.load_table("/nonexistent/table_rows.json")["cells"] == {}                        # a missing table = empty: every beside-width declines by name
    assert R.chunking(64) == (64, 1) and R.chunking(128) == (128, 1) and R.chunking(256) == (256, 1) and R.chunking(384) == (128, 3)
    for bad in (0, 48, 100, 640, 1024):
        with pytest.raises(R.RowsUnsupported):
            R.chunking(bad)


def test_resolve_rows_and_widths_words(monkeypatch):
    monkeypatch.delenv(R.ENV_UNMEASURED, raising=False)
    assert RF.ROWS_UNIT == "fpf_trimul_" + "rows" and RF.KERNEL == "fpf_trimul_" + "v4" and RF.SUPPORTED_C == (128, 256)
    assert RF.rows_widths(128, 128) == RF.rows_widths(256, 256) == RF.rows_widths(128, 256) == "v4"
    assert RF.rows_widths(384, 384) == RF.rows_widths(64, 64) == RF.rows_widths(64, 128) == "rows" and RF.rows_widths(512, 512) == RF.rows_widths(32, 32) == "none"
    sel, k1a, k1b, k3, bdir = RF.resolve_rows("12.0", 384, 384)
    assert not sel.served and sel.reason == "no_row" and (k1a, k1b, k3, bdir) == (None, None, None, False)
    sel, k1a, k1b, k3, bdir = RF.resolve_rows("8.0", 384, 384)
    assert sel.served and k1a == {"BM": 64, "BN": 64, "num_warps": 4, "num_stages": 1} and k3 == {"BM": 64, "BN": 128, "num_warps": 8, "num_stages": 1} and bdir is True
    sel, k1a, k1b, k3, bdir = RF.resolve_rows("9.0", 384, 384)
    assert sel.served and sel.reason == "served" and k1a == {"BM": 64, "BN": 128, "num_warps": 4, "num_stages": 1} and k3 == {"BM": 64, "BN": 128, "num_warps": 8, "num_stages": 1} and bdir is True
    sel, k1a, k1b, k3, bdir = RF.resolve_rows("9.0", 64, 64)
    assert sel.served and k1a["BM"] == 128 and k1b == {"BM": 64, "BN": 32, "num_warps": 4, "num_stages": 2} and k3["num_stages"] == 1 and bdir is True


# ----------------------------------------------------------------------------------------------------------------- the provider's ladder (simulated card)
N, C, C_H = 40, 16, 8


class _FakeZ(object):
    """Stands in for a CUDA pair block in the decision ladder (which reads is_cuda / device / dtype / shape only)."""
    is_cuda = True

    def __init__(self, n=4096, c=384, dtype=None, device="cuda:0"):
        self.device = torch.device(device)
        self.dtype = dtype or torch.bfloat16
        self.shape = (8, n, c)


def _weights(g, c, ch):
    r = lambda *s: torch.randn(*s, generator=g) * 0.3  # noqa: E731
    return {"ln_in_w": 1 + r(c), "ln_in_b": r(c), "w_ag": r(ch, c), "w_ap": r(ch, c), "w_bg": r(ch, c), "w_bp": r(ch, c),
            "ln_out_w": 1 + r(ch), "ln_out_b": r(ch), "w_o": r(c, ch), "w_og": r(c, c)}


def _stock_fns(w):
    c, ch = int(w["w_ap"].shape[1]), int(w["w_ap"].shape[0])
    ln = lambda x, wt, b: torch.nn.functional.layer_norm(x, (x.shape[-1],), wt, b, 1e-5)  # noqa: E731

    def proj(zb, m, is_a):
        x = ln(zb, w["ln_in_w"], w["ln_in_b"])
        wg, wp = (w["w_ag"], w["w_ap"]) if is_a else (w["w_bg"], w["w_bp"])
        return torch.sigmoid(x @ wg.t()) * (x @ wp.t()) * m

    def out(x):
        return ln(x, w["ln_out_w"], w["ln_out_b"]) @ w["w_o"].t()

    def gate(zb):
        return torch.sigmoid(ln(zb, w["ln_in_w"], w["ln_in_b"]) @ w["w_og"].t())
    return TM.TriMulFns(proj, out, gate, ch)


@pytest.mark.parametrize("cc", ["9.0", "8.0", "12.0"])
def test_ladder_by_card(cc, monkeypatch):
    """C 384: on cc 12.0 (no key) the decision is the decline word ``c=384/384`` — what this width answered before the rows unit — with the reason recorded
    (``rows_table=-:no_row``); on cc 9.0 / 8.0 (MEASURED key) the gate opens: the ladder proceeds to the kernels (on this CPU-only process the weight pack cannot
    move to the fake device, which surfaces as an ordinary exception AFTER the fact ``rows_table=9.0|C384|H384:served`` is recorded — proof the gate opened)."""
    monkeypatch.delenv(R.ENV_UNMEASURED, raising=False)
    monkeypatch.delenv(RF.ENV_KERNELS, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: tuple(int(x) for x in cc.split(".")))
    g = torch.Generator().manual_seed(1)
    w = _weights(g, 384, 384)
    fns = RF.fused_trimul_fns(w, _stock_fns(w), ledger=RF.Ledger("t"))
    z = _FakeZ(4096, 384)
    if cc == "12.0":
        assert fns.operand_dtype(z) is None                                         # declined -> the statements' dtype
        assert fns._decide(z) == (None, None, "c=384/384")
        assert fns.facts["rows_table"] == "-:no_row" and fns.ledger.get("rows_table") == "-:no_row"
        assert fns.b_transposed(z) is False
        monkeypatch.setenv(R.ENV_UNMEASURED, "1")                                   # the switch admits PLACEHOLDER keys only; no key stays no key
        fns2 = RF.fused_trimul_fns(w, _stock_fns(w), ledger=RF.Ledger("t2"))
        assert fns2._decide(z) == (None, None, "c=384/384") and fns2.facts["rows_table"] == "-:no_row"
    else:
        opened = False
        try:
            d = fns._decide(z)
            opened = d[0] is not None
        except RowpairRefused:
            raise
        except Exception:                                                           # noqa: BLE001 — .to('cuda') of the weight pack on a CPU-only torch
            opened = True
        assert opened and fns.facts["rows_table"] == cc + "|C384|H384:served"
        monkeypatch.setenv(RF.ENV_KERNELS, "torch")                                 # the opt-out still wins over a measured key
        assert RF.fused_trimul_fns(w, _stock_fns(w), ledger=RF.Ledger("t3"))._decide(z) == (None, None, "env_torch")


def test_ladder_size_floor_per_card_and_width(monkeypatch):
    monkeypatch.delenv(RF.ENV_KERNELS, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    g = torch.Generator().manual_seed(2)
    w256, w128, w384 = _weights(g, 256, 256), _weights(g, 128, 128), _weights(g, 384, 384)
    f = RF.fused_trimul_fns(w256, _stock_fns(w256), ledger=RF.Ledger("a"))
    assert f.min_tokens == RF.DEFAULT_MIN_TOKENS == 2048                            # at construction (no device yet): the default, as before
    assert f._floor(_FakeZ(700, 256)) == 0 and f.min_tokens == 0                   # cc 9.0, C 256: the table's floor (every sharded size)
    assert f.facts["min_tokens_source"] == "rows_table:9.0|C256|H256" and f.ledger.get("min_tokens") == 0 and f.ledger.get("min_tokens_source") == "rows_table:9.0|C256|H256"
    f._gate_n = 700
    d = None
    try:
        d = f._decide(_FakeZ(700, 256))
    except Exception:                                                               # noqa: BLE001 — past the size gate the v4 loader wants a real device; the point is: not below_gate
        d = ("past-gate",)
    assert d != (None, None, "below_gate")
    fk = RF.fused_trimul_fns(w256, _stock_fns(w256), ledger=RF.Ledger("b"), min_tokens=3000)     # a kit's explicit floor wins
    assert fk._floor(_FakeZ(700, 256)) == 3000 and "min_tokens_source" not in fk.facts
    fk._gate_n = 2999
    assert fk._decide(_FakeZ(2999, 256)) == (None, None, "below_gate")
    f128 = RF.fused_trimul_fns(w128, _stock_fns(w128), ledger=RF.Ledger("c"))
    assert f128._floor(_FakeZ(700, 128)) == 2048                                    # C 128 on 9.0: no entry, the default
    f128._gate_n = 2047
    assert f128._decide(_FakeZ(2047, 128)) == (None, None, "below_gate")
    f384 = RF.fused_trimul_fns(w384, _stock_fns(w384), ledger=RF.Ledger("d"))
    assert f384._floor(_FakeZ(1399, 384)) == 1400 and f384.facts["min_tokens_source"] == "rows_table:9.0|C384|H384"
    f384._gate_n = 1399
    assert f384._decide(_FakeZ(1399, 384)) == (None, None, "below_gate")           # the measured key's own floor (1400 = its smallest measured N) holds below it
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))
    f8 = RF.fused_trimul_fns(w256, _stock_fns(w256), ledger=RF.Ledger("e"))
    assert f8._floor(_FakeZ(700, 256, device="cuda:1")) == 0 and f8.facts["min_tokens_source"] == "rows_table:8.0|C256|H256"   # cc 8.0: the A100 pass's floor
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (12, 0))
    f12 = RF.fused_trimul_fns(w256, _stock_fns(w256), ledger=RF.Ledger("e12"))
    assert f12._floor(_FakeZ(700, 256, device="cuda:2")) == 2048                   # a card without an entry: the default
    cpu = torch.zeros(2, N, 256)
    assert RF.fused_trimul_fns(w256, _stock_fns(w256), ledger=RF.Ledger("f"))._floor(cpu) == 2048    # CPU tensors never read the table


# ----------------------------------------------------------------------------------------------------------------- the driver seam: b_transposed / _b_plane
class _TransposedHooks(TM.TriMulFns):
    """A provider whose K1 'kernel' (torch) writes whatever [C_h, rows, N] destination it is handed — strided or not — and asks for the GEMM-B layout."""

    def __init__(self, stock, transposed):
        TM.TriMulFns.__init__(self, stock.proj, stock.out, stock.gate, stock.C_h)
        self.transposed = transposed
        self.layouts = []

    def b_transposed(self, z_shard):
        return self.transposed

    def proj_into(self, dst, z_block, mask_block, is_a):
        m = mask_block.unsqueeze(-1) if mask_block is not None else 1.0
        self.layouts.append("a" if dst.stride(2) == 1 else ("b" if dst.stride(1) == 1 else "?"))
        dst.copy_(self.proj(z_block, m, is_a).permute(2, 0, 1))
        return True


def _rank_bt(rank, P, transposed, outgoing):
    g = torch.Generator().manual_seed(7)
    z = torch.randn(N, N, C, generator=g)
    mask = (torch.rand(N, N, generator=g) > 0.15).float()
    w = _weights(g, C, C_H)
    stock = _stock_fns(w)
    fns = stock if transposed is None else _TransposedHooks(stock, transposed)
    lay = Layout(N, P, rank, 8)
    zs = z[lay.r0:lay.r1].clone()
    TM.trimul_update_(fns, zs, mask[lay.r0:lay.r1].clone(), lay, outgoing=outgoing, add=True, inplace_chunk=8, RB=8)
    return all_gather_rows(zs, lay), (list(fns.layouts) if transposed is not None else [])


@pytest.mark.parametrize("outgoing", [True, False])
def test_b_transposed_seam_is_byte_identical(outgoing):
    plain = run_ranks(2, _rank_bt, None, outgoing)
    hooks_a = run_ranks(2, _rank_bt, False, outgoing)
    hooks_b = run_ranks(2, _rank_bt, True, outgoing)
    for r in range(2):
        assert torch.equal(hooks_a[r][0], plain[r][0]) and torch.equal(hooks_b[r][0], plain[r][0])
        assert "b" not in hooks_a[r][1] and "a" in hooks_a[r][1]                      # False: today's [C_h, w, N] planes only
        assert "b" in hooks_b[r][1]                                                    # True: b sub-blocks arrive as the GEMM-B layout view
    view = TM._b_plane(lambda z: True, None, 8, 3, 20, torch.float32, "cpu")
    assert tuple(view.shape) == (8, 3, 20) and view.stride() == (60, 1, 3) and view.transpose(1, 2).is_contiguous()
    flat = TM._b_plane(None, None, 8, 3, 20, torch.float32, "cpu")
    assert flat.is_contiguous() and tuple(flat.shape) == (8, 3, 20)
    assert TM._b_plane(lambda z: False, None, 8, 3, 20, torch.float32, "cpu").is_contiguous()


# ----------------------------------------------------------------------------------------------------------------- kernels: host-side words (no launch)
def test_kernel_launchers_refuse_foreign_layouts_by_name():
    triton = pytest.importorskip("triton")  # noqa: F841
    from opt_core.kernels.fpf_trimul_rows import kernels as RK
    assert RK.WIDTHS == R.WIDTHS and RK.chunking is R.chunking and RK.RowsUnsupported is R.RowsUnsupported
    D, rows, n = 384, 6, 10
    a = torch.empty(D, rows, n, dtype=torch.bfloat16)
    b = torch.empty(D, n, rows, dtype=torch.bfloat16).transpose(1, 2)
    assert RK.k1_layout(a, rows, n, D) == RK.LAYOUT_A and RK.k1_layout(b, rows, n, D) == RK.LAYOUT_B
    assert RK.k1_layout(torch.empty(D, rows + 4, n, dtype=torch.bfloat16)[:, :rows], rows, n, D) == RK.LAYOUT_A     # a rows-slice of a bigger A block
    for bad in (torch.empty(D, rows, n, dtype=torch.float32), torch.empty(D, rows, 2 * n, dtype=torch.bfloat16)[:, :, ::2], torch.empty(D, n, rows, dtype=torch.bfloat16)):
        with pytest.raises(RK.RowsUnsupported):
            RK.k1_layout(bad, rows, n, D)
    z = torch.zeros(rows, n, 384, dtype=torch.bfloat16)
    kw = dict(ln_w=torch.zeros(384), ln_b=torch.zeros(384), wgT=torch.zeros(384, D, dtype=torch.bfloat16), wpT=torch.zeros(384, D, dtype=torch.bfloat16), bg=None, bp=None,
              has_bias=False, cell={"BM": 64, "BN": 128, "num_warps": 4, "num_stages": 1}, eps=1e-5)
    with pytest.raises(RK.RowsUnsupported, match="k1_z"):
        RK.launch_k1(z[:, :, ::2], None, torch.empty(D, rows, n, dtype=torch.bfloat16), **dict(kw, wgT=torch.zeros(192, D, dtype=torch.bfloat16)))
    with pytest.raises(RK.RowsUnsupported, match="k1_mask"):
        RK.launch_k1(z, torch.zeros(rows, n, dtype=torch.bool), a, **kw)
    with pytest.raises(RK.RowsUnsupported, match="k1_cell"):
        RK.launch_k1(z, None, a, **dict(kw, cell={"BM": 64, "BN": 256, "num_warps": 4, "num_stages": 1}))
    assert RK.launch_k1(z[:0], None, a[:, :0], **kw) == RK.LAYOUT_A                                                  # empty block: the word, no launch
    T = torch.zeros(D, rows, 5, dtype=torch.bfloat16)
    win = torch.zeros(rows, n, 384, dtype=torch.bfloat16)[:, 2:7]
    k3 = dict(ln_out_w=torch.zeros(D), ln_out_b=torch.zeros(D), ln_in_w=torch.zeros(384), ln_in_b=torch.zeros(384), wz=torch.zeros(384, D, dtype=torch.bfloat16),
              wg_out=torch.zeros(384, 384, dtype=torch.bfloat16), bo=None, bg=None, has_bias=False, cell={"BM": 128, "BN": 128, "num_warps": 8, "num_stages": 1},
              residual=True, stock_round=True, eps=1e-5)
    with pytest.raises(RK.RowsUnsupported, match="k3_T"):
        RK.launch_k3(T.float(), win, win, **k3)
    with pytest.raises(RK.RowsUnsupported, match="k3_z"):
        RK.launch_k3(T, win[:, :, ::2], win[:, :, ::2], **dict(k3, ln_in_w=torch.zeros(192)))
    with pytest.raises(RK.RowsUnsupported, match="k3_out"):
        RK.launch_k3(T, win, win.float(), **k3)
    RK.launch_k3(T[:, :0], win[:0], win[:0], **k3)                                                                       # empty window: no launch


# ----------------------------------------------------------------------------------------------------------------- kernels under triton's interpreter (CPU)
_INTERP = textwrap.dedent(r'''
    import json, sys
    sys.path.insert(0, %(core)r)
    import torch, triton
    from opt_core.kernels.fpf_trimul_v4 import kernels as K
    from opt_core.kernels.fpf_trimul_rows import kernels as RK
    g = torch.Generator().manual_seed(0)
    out = {}
    def raw(C, D):
        r = lambda *s: torch.randn(*s, generator=g) * 0.2
        return dict(ln_in_w=1 + r(C), ln_in_b=r(C), w_ag=r(D, C), w_ap=r(D, C), w_bg=r(D, C), w_bp=r(D, C), ln_out_w=1 + r(D), ln_out_b=r(D), w_o=r(C, D), w_og=r(C, C),
                    b_ag=r(D), b_ap=r(D), b_bg=r(D), b_bp=r(D), b_o=r(C), b_og=r(C))
    for C, D, bias in ((64, 64, False), (128, 128, True), (384, 384, True)):
        rw = raw(C, D)
        if not bias:
            rw = {k: v for k, v in rw.items() if not k.startswith("b_")}
        pack = K.pack_generic(cdt=torch.bfloat16, **rw)
        hb = bool(pack["has_bias"])
        rows, N, w = 3, 32, 24
        for proj in ("a", "b"):
            sl = slice(0, D) if proj == "a" else slice(D, 2 * D)
            h = {"wgT": pack["wgT_in"][:, sl].contiguous(), "wpT": pack["wpT_in"][:, sl].contiguous(),
                 "bg": pack["bg_in"][sl].contiguous() if hb else None, "bp": pack["bp_in"][sl].contiguous() if hb else None}
            for zdt in (torch.bfloat16, torch.float32):
                big = torch.randn(rows, N + 5, C, generator=g).to(zdt)
                zv = big[:, :N]                                                   # a row block that is NOT contiguous (row stride (N+5)*C)
                mask = (torch.rand(rows, N, generator=g) > 0.2).float()
                cell = {"BM": 16, "BN": 32, "num_warps": 4, "num_stages": 1}
                A = torch.zeros(D, rows, N, dtype=torch.bfloat16)
                RK.launch_k1(zv, mask, A, ln_w=pack["ln_in_w"], ln_b=pack["ln_in_b"], wgT=h["wgT"], wpT=h["wpT"], bg=h["bg"], bp=h["bp"], has_bias=hb, cell=cell, eps=1e-5)
                B = torch.zeros(D, N, rows, dtype=torch.bfloat16).transpose(1, 2)
                RK.launch_k1(zv, mask.to(torch.bfloat16), B, ln_w=pack["ln_in_w"], ln_b=pack["ln_in_b"], wgT=h["wgT"], wpT=h["wpT"], bg=h["bg"], bp=h["bp"], has_bias=hb, cell=cell, eps=1e-5)
                key = "k1 C%%d %%s %%s%%s" %% (C, proj, str(zdt).replace("torch.", ""), " bias" if hb else "")
                out[key + " a==b"] = bool(torch.equal(A, B))
                if RK.chunking(C)[1] == 1:                                        # one chunk: the fpf_trimul_v4 kernel on a contiguous copy is the reference, bit for bit
                    zc = zv.contiguous()
                    V = torch.zeros(D, rows, N, dtype=torch.bfloat16)
                    grid = (triton.cdiv(N, 16), rows, 1)
                    K._k1c[grid](zc, pack["ln_in_w"], pack["ln_in_b"], h["wgT"], h["wpT"], h["bg"] if hb else pack["ln_in_w"], h["bp"] if hb else pack["ln_in_w"], mask, V,
                                 N, N, V.stride(0), 1e-5, rows * N, 0, 0, C=C, D2=D, BM=16, BN=32, HAS_MASK=True, HAS_BIAS=hb, IN_F32=(zdt == torch.float32), num_warps=4, num_stages=1)
                    out[key + " ==v4"] = bool(torch.equal(A, V))
        for zdt in (torch.bfloat16, torch.float32):
            for add in (True, False):
                T = torch.randn(D, rows, w, generator=g).to(torch.bfloat16)
                zfull = torch.randn(rows, N, C, generator=g).to(zdt)
                ref_full = zfull.clone()
                win = zfull[:, 3:3 + w]
                cwin = ref_full[:, 3:3 + w].contiguous()                          # the same window staged contiguous (today's statement)
                cell = {"BM": 16, "BN": 32, "num_warps": 4, "num_stages": 1}
                kw = dict(ln_out_w=pack["ln_out_w"], ln_out_b=pack["ln_out_b"], ln_in_w=pack["ln_in_w"], ln_in_b=pack["ln_in_b"], wz=pack["wz"], wg_out=pack["wg_out"],
                          bo=pack.get("b_o"), bg=pack.get("b_og"), has_bias=hb, cell=cell, residual=add, stock_round=True, eps=1e-5)
                before = zfull.clone()
                RK.launch_k3(T, win, win, **kw)
                RK.launch_k3(T, cwin, cwin, **kw)
                key = "k3 C%%d %%s add=%%s%%s" %% (C, str(zdt).replace("torch.", ""), add, " bias" if hb else "")
                out[key + " inplace==staged"] = bool(torch.equal(win, cwin))
                out[key + " outside untouched"] = bool(torch.equal(zfull[:, :3], before[:, :3]) and torch.equal(zfull[:, 3 + w:], before[:, 3 + w:]))
                if RK.chunking(C)[1] == 1:
                    vwin = ref_full[:, 3:3 + w].contiguous()
                    grid = (triton.cdiv(w, 16), rows, 1)
                    K._k3c[grid](T, vwin, pack["ln_out_w"], pack["ln_out_b"], pack["ln_in_w"], pack["ln_in_b"], pack["wz"], pack["wg_out"],
                                 pack["b_o"] if hb else pack["ln_in_w"], pack["b_og"] if hb else pack["ln_in_w"], vwin, w, T.stride(1), T.stride(0), 1e-5, rows * w,
                                 C=C, CH=D, BM=16, BN=32, RESIDUAL=add, STOCK_ROUND=True, HAS_BIAS=hb, IN_F32=(zdt == torch.float32), num_warps=4, num_stages=1)
                    out[key + " ==v4"] = bool(torch.equal(win, vwin))
    print("JSON" + json.dumps(out))
''')


def test_kernels_under_the_triton_interpreter_match_fpf_trimul_v4_bit_for_bit():
    """CPU, TRITON_INTERPRET=1 (a fresh interpreter process: the variable must be set before triton loads): every ``==v4`` (one-chunk widths vs the
    certified fpf_trimul_v4 kernels on contiguous copies), ``a==b`` (the two K1 layouts), ``inplace==staged`` (the window epilogue vs a staged copy)
    and ``outside untouched`` entry is True. The interpreter's bf16 MMA is not IEEE-faithful, so this is a statement-and-addressing identity test, not a
    numerics test (numerics: the GPU battery / tests marked gpu)."""
    pytest.importorskip("triton")
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    p = subprocess.run([sys.executable, "-c", _INTERP % {"core": CORE}], env=env, capture_output=True, text=True, timeout=170)
    assert p.returncode == 0, p.stderr[-3000:]
    line = [l for l in p.stdout.splitlines() if l.startswith("JSON")]
    assert line, p.stdout[-2000:]
    res = json.loads(line[-1][4:])
    bad = {k: v for k, v in res.items() if v is not True}
    assert len(res) >= 30 and not bad, bad
    assert any(k.endswith("==v4") for k in res) and any("C384" in k for k in res)


# ----------------------------------------------------------------------------------------------------------------- invariance of the whole-plane lines
def test_whole_plane_tables_are_byte_identical_to_their_pins():
    """The rows members touch NO whole-plane table: fpf_trimul_v4/table.json and trimul/TRIMUL_CELLS.json are byte-identical to the digests pinned in
    tests/fixtures/rows_untouched_tables.json (recorded when the members were written). A deliberate edit of either table by its owner
    updates the fixture in the same commit."""
    pins = json.load(open(PINS_PATH, encoding="utf-8"))
    for rel, want in pins["sha256"].items():
        assert _sha(rel) == want, "%s changed: sha256 %s != pinned %s (update tests/fixtures/rows_untouched_tables.json if the edit is deliberate)" % (rel, _sha(rel), want)


def test_no_whole_plane_provider_names_the_rows_unit():
    """Static closure: nothing under kernels/trimul/**, kernels/fpf_trimul_v4/** or kernels/trimul_esm_shapes/** names the rows unit or its table — the
    single-GPU lines cannot consult it; only mem/rowpair/trimul_fused.py (P > 1) does."""
    words = ("fpf_trimul_" + "rows", "table_rows.json", "ROWPAIR_TRIMUL_ROWS_UNMEASURED")
    offenders = []
    for sub in ("trimul", "fpf_trimul_v4", "trimul_esm_shapes", "fpf_trimul"):
        root = os.path.join(KDIR, sub)
        for r_, _ds, fs in os.walk(root):
            for f in fs:
                if f.endswith((".py", ".json")):
                    txt = open(os.path.join(r_, f), encoding="utf-8", errors="replace").read()
                    offenders += ["%s: %s" % (os.path.relpath(os.path.join(r_, f), KDIR), w_) for w_ in words if w_ in txt]
    assert offenders == [], offenders
    rf = open(os.path.join(CORE, "opt_core", "mem", "rowpair", "trimul_fused.py"), encoding="utf-8").read()
    assert "fpf_trimul_rows" in rf and "ROWS_UNIT" in rf


def test_v4_whole_plane_selection_sweep_is_independent_of_the_rows_table(monkeypatch, tmp_path):
    """fpf_trimul_v4's ONE selection statement over every row of its table x representative tritons x its (C, D, bias) shapes answers the same with the
    rows table present, empty, or pointing at garbage (it never reads it)."""
    from opt_core.kernels.fpf_trimul_v4 import table as V4T
    table = V4T.load_table()

    def sweep():
        res = {}
        for key in sorted(k for k in table if "|" in k):
            cc = key.split("|")[0]
            for mm in ("3.3", "3.4", "3.6", "3.7", "9.9"):
                for has_desc in (True, False):
                    try:
                        sel = V4T.select(table, cc, mm, has_desc=has_desc)
                    except TypeError:
                        sel = V4T.select(table, cc, mm)
                    for C_, D_, b_ in ((256, 256, False), (128, 128, False), (128, 128, True), (256, 256, True), (64, 128, False)):
                        cfg = None
                        if sel is not None and getattr(sel, "cfg", None) is not None:
                            try:
                                cfg = V4T.resolve_cfg(sel.cfg, C_, D_, b_)
                            except Exception as e:  # noqa: BLE001
                                cfg = repr(e)[:60]
                        res[(key, mm, has_desc, C_, D_, b_)] = (repr(sel)[:400], repr(cfg))
        return res
    base = sweep()
    garbage = tmp_path / "table_rows.json"
    garbage.write_text("{not json")
    monkeypatch.setenv("FPF_TRIMUL_ROWS_TABLE", str(garbage))
    assert sweep() == base
    monkeypatch.setenv("FPF_TRIMUL_ROWS_TABLE", str(tmp_path / "absent.json"))
    assert sweep() == base and len(base) > 0
