"""'sampler rows' — CPU tests of the NEW symbols beside the x1 ones (opt_core.mem.rowpair.diffusion: DiTBlockFns.bias_into,
DitBias, pair_bias_rows_into, dit_bias_word, DiffusionSchedule.decide(attn_core=), DitKV, dit_rows_core, dit_attention_rows; opt_core.kernels.apb:
pair_bias_attention_rows, admits_rows, select_rows, rows_cell_key, reference_rows):

  * the x1 / today's-xP surfaces are UNCHANGED: DiTBlockFns 5-positional construction, DiffusionSchedule.decide's fields without attn_core;
  * the engine word of pair_bias_rows_into is bitwise pair_bias_rows (gloo P in {2, 3}); a DitBias under ROWPAIR_DIFF_BIAS=ln_proj on CPU steps aside
    BY NAME per block (engine:not_cuda | engine:import) and is bitwise the engine; the transformer over blocks carrying bias_into == without;
  * dit_attention_rows on CPU: the rows face refuses by name (cc:0.0) -> the engine statement in chunks of stock_q_rows, == today's attn (1e-5, bitwise reported);
  * the rows face's static admission: fp32 under the ROW word apb_attn refused (dtype:float32), under big admitted with cast:bf16; head_dim > 64,
    cc < 8, exact / faithful, square-only rows, unknown words refused by name; no rows cell measured -> apb_attn by name 'beyond_measured';
    rows_cell_key bucket rule on a synthetic rows_cells section; reference_rows == an independent softmax statement.

GPU numerics of the served kernels (apb_attn rectangular vs reference_rows; ln_proj into vs engine) run in the anchor box (tests/gpu).
Run: ``python -m pytest tests/test_rowpair_sampler_rows_xp4.py -q -rfE``.
"""
from __future__ import annotations

import json
import os
import sys

try:
    import pytest
except ImportError:                                            # a GPU stack without pytest: the __main__ gloo entry below still runs
    class _Mark(object):
        def skipif(self, *a, **k):
            return lambda f: f

        def parametrize(self, *a, **k):
            return lambda f: f

    class _Shim(object):
        mark = _Mark()

        @staticmethod
        def raises(*a, **k):
            raise RuntimeError("pytest is not installed: run the assertions under pytest")

    pytest = _Shim()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                      # the opt_core checkout under test
sys.path.insert(0, HERE)                                       # the sibling synthetic engine (test_rowpair_diffusion_043._Engine)

from opt_core.mem.rowpair import RowpairRefused  # noqa: E402
from opt_core.mem.rowpair import diffusion as DF  # noqa: E402
from opt_core.mem.rowpair.dist import Layout  # noqa: E402
from opt_core.kernels import apb as A  # noqa: E402

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:  # noqa: BLE001
    HAVE_TORCH = False

needs_torch = pytest.mark.skipif(not HAVE_TORCH, reason="torch not importable")
TOL32 = 1e-5


# ================================================================================================================ pure / single-process
def test_ln_proj_x1_width_set_untouched_rows_set_beside():
    """kernels/ln_proj.py: the x1 pair-bias width set and its packer are byte-for-byte the tag's; c 256 enters ONLY through the rows set / rows packer."""
    import ast
    src = open(os.path.join(os.path.dirname(HERE), "opt_core", "kernels", "ln_proj.py"), encoding="utf-8").read()
    assert "\nSERVED_C_PAIR = (64, 128)                  # pair_bias channel widths\n" in src            # the x1 line, verbatim
    assert "\nROWS_SERVED_C_PAIR = SERVED_C_PAIR + (256,)" in src
    tree = ast.parse(src)
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    x1 = ast.get_source_segment(src, fns["pack_pair_bias_weights"])
    rows = ast.get_source_segment(src, fns["pack_pair_bias_weights_rows"])
    assert "if C not in SERVED_C_PAIR:" in x1 and "ROWS_SERVED_C_PAIR" not in x1                       # the x1 packer consults the x1 set only
    assert "if C not in ROWS_SERVED_C_PAIR:" in rows                                                    # same contract, the one set swapped
    body = lambda seg: seg.split('"""', 2)[2]                                                           # the statements after the docstring
    assert body(rows).replace("ROWS_SERVED_C_PAIR", "SERVED_C_PAIR") == body(x1)                        # statement-for-statement the x1 packer with the rows set
    users = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "ROWS_SERVED_C_PAIR"]
    assert len(users) == 3, len(users)                                                                  # the assignment + the rows packer's check and message: nothing x1-facing reads it
    for name in ("served_pair_bias", "pair_bias", "describe", "pack_ln_linear_weights", "served_ln_linear"):
        assert "ROWS_SERVED_C_PAIR" not in ast.get_source_segment(src, fns[name]), name
    # the diffusion producer packs through the rows packer
    dsrc = open(os.path.join(os.path.dirname(HERE), "opt_core", "mem", "rowpair", "diffusion.py"), encoding="utf-8").read()
    assert 'getattr(LP, "pack_pair_bias_weights_rows", None)' in dsrc


def test_ln_proj_rows_packer_widths_importable():
    """With triton importable: the x1 packer refuses c 256 by name (width) exactly as before; the rows packer admits 64 / 128 / 256 and refuses 192 / 384 / 512."""
    pytest.importorskip("triton")
    torch = pytest.importorskip("torch")
    from opt_core.kernels import ln_proj as LP
    assert LP.SERVED_C_PAIR == (64, 128) and LP.ROWS_SERVED_C_PAIR == (64, 128, 256) and tuple(LP.describe()["pair_bias"]["c"]) == (64, 128)
    dev = torch.device("cpu")
    for c in (64, 128, 256):
        w = torch.randn(16, c)
        pk = LP.pack_pair_bias_weights_rows(torch.ones(c), torch.zeros(c), w, 1e-5, dev)
        assert pk["kind"] == "pair_bias" and pk["C"] == c and pk["nout"] == 16
    for c in (192, 384, 512):
        with pytest.raises(LP.Unsupported) as ei:
            LP.pack_pair_bias_weights_rows(None, None, torch.randn(16, c), 1e-5, dev)
        assert ei.value.reason == "width"
    with pytest.raises(LP.Unsupported) as ei:
        LP.pack_pair_bias_weights(torch.ones(256), torch.zeros(256), torch.randn(16, 256), 1e-5, dev)   # the x1 packer: 256 refused by name, unchanged
    assert ei.value.reason == "width"
    assert LP.pack_pair_bias_weights(None, None, torch.randn(16, 128), 1e-5, dev)["C"] == 128


class _FakeOOM(RuntimeError):
    """torch >= 2.5 raises torch.OutOfMemoryError(RuntimeError); opt_core.oom.is_oom recognises the class name / the message."""


_FakeOOM.__name__ = "OutOfMemoryError"


@pytest.mark.skipif(not HAVE_TORCH, reason="needs torch")
@pytest.mark.parametrize("where", ["packer", "launcher"])
def test_ditbias_oom_propagates_never_rerouted(monkeypatch, where):
    """An out-of-memory raised by the producer's packer or its kernel launch PROPAGATES out of DitBias.ln_proj_into and out of the rows loop's
    fallback (pair_bias_rows_into catches _IntoRefused only) — it is never swallowed into the engine statement; any other failure is a refusal by
    name (_IntoRefused(reason)) that the loop serves with the engine statement."""
    import types
    from opt_core.oom import is_oom
    assert is_oom(_FakeOOM("CUDA out of memory. Tried to allocate 2.00 GiB"))
    calls = {"pack": 0, "launch": 0}

    class _Unsupported(ValueError):
        def __init__(self, reason):
            super().__init__(reason); self.reason = reason

    def pack_ok(lnw, lnb, w, eps, device, dot_fp32=False):
        calls["pack"] += 1
        return {"kind": "pair_bias", "C": int(w.shape[1]), "nout": int(w.shape[0]), "device": device, "dot_fp32": dot_fp32}

    def pack(lnw, lnb, w, eps, device, dot_fp32=False):
        if where == "packer":
            calls["pack"] += 1
            raise _FakeOOM("CUDA out of memory. Tried to allocate 2.00 GiB (simulated in the packer)")
        return pack_ok(lnw, lnb, w, eps, device, dot_fp32)

    def pair_bias(z, packed, *, out_layout="bhij", out_dtype=None, out=None):
        calls["launch"] += 1
        raise _FakeOOM("CUDA out of memory. Tried to allocate 2.00 GiB (simulated in the launch)")

    fake = types.SimpleNamespace(pack_pair_bias_weights_rows=pack, pack_pair_bias_weights=pack, pair_bias=pair_bias, Unsupported=_Unsupported)
    import opt_core.kernels.apb as APB
    monkeypatch.setattr(APB, "carried_module", lambda name: fake)
    H, c, rows, N = 4, 32, 3, 5
    lin = torch.nn.Linear(c, H, bias=False); ln = torch.nn.LayerNorm(c)
    spec = DF.DitBias(engine=lambda zz: lin(ln(zz)), ln_weight=ln.weight, ln_bias=ln.bias, weight=lin.weight, eps=ln.eps)
    z = torch.randn(rows, N, c); o = torch.empty(H, rows, N)
    with pytest.raises(_FakeOOM):
        spec.ln_proj_into(z, o)
    assert calls == ({"pack": 1, "launch": 0} if where == "packer" else {"pack": 1, "launch": 1})
    assert spec._off is None                                                                            # an OOM never latches the block off either
    # the non-OOM failure classes ARE refusals by name (served by the engine statement in the loop): a packer Unsupported and a launch RuntimeError
    def pack_refuse(lnw, lnb, w, eps, device, dot_fp32=False):
        raise _Unsupported("width")
    fake.pack_pair_bias_weights_rows = pack_refuse
    spec2 = DF.DitBias(engine=lambda zz: lin(ln(zz)), ln_weight=ln.weight, ln_bias=ln.bias, weight=lin.weight, eps=ln.eps)
    with pytest.raises(DF._IntoRefused) as ei:
        spec2.ln_proj_into(z, o)
    assert ei.value.reason == "width"
    fake.pack_pair_bias_weights_rows = pack_ok

    def launch_fail(z, packed, **kw):
        raise RuntimeError("PassManager::run failed (simulated compile failure)")
    fake.pair_bias = launch_fail
    spec3 = DF.DitBias(engine=lambda zz: lin(ln(zz)), ln_weight=ln.weight, ln_bias=ln.bias, weight=lin.weight, eps=ln.eps)
    with pytest.raises(DF._IntoRefused) as ei:
        spec3.ln_proj_into(z, o)
    assert ei.value.reason == "launch:RuntimeError" and spec3._off == "launch:RuntimeError"             # a launch failure latches the block to the engine statement


def test_ditblockfns_default_construction_unchanged():
    f5 = DF.DiTBlockFns(1, 2, 3, 4, 5)                                              # today's 5-positional construction
    assert (f5.norm, f5.kv, f5.attn, f5.update, f5.bias) == (1, 2, 3, 4, 5) and f5.bias_into is None
    assert DF.DiTBlockFns._fields == ("norm", "kv", "attn", "update", "bias", "bias_into")
    fk = DF.DiTBlockFns(norm=1, kv=2, attn=3, update=4, bias=5)
    assert fk == f5 and fk._replace(bias_into="x").bias_into == "x"
    assert DF.DiTBlockFns(1, 2, 3, 4, 5, 6).bias_into == 6
    kv = DF.DitKV(1, 2, None, (1, 2, 3))
    assert kv.k == 1 and kv.key_mask is None and kv.stock == (1, 2, 3)


def test_dit_bias_word(monkeypatch):
    monkeypatch.delenv(DF.ENV_DIFF_BIAS, raising=False)
    assert DF.dit_bias_word() == "engine" and DF.dit_bias_word("ln_proj") == "ln_proj" and DF.dit_bias_word(" LN_PROJ ") == "ln_proj"
    assert DF.dit_bias_word("ln_proj_fp32") == "ln_proj_fp32" and DF.DIFF_BIAS_WORDS == ("engine", "ln_proj", "ln_proj_fp32")
    monkeypatch.setenv(DF.ENV_DIFF_BIAS, "ln_proj")
    assert DF.dit_bias_word() == "ln_proj"
    monkeypatch.setenv(DF.ENV_DIFF_BIAS, "fused_please")
    with pytest.raises(RowpairRefused, match="ROWPAIR_DIFF_BIAS"):
        DF.dit_bias_word()
    assert DF.dit_bias_word("engine") == "engine"                                     # an explicit word wins over the env


def test_schedule_attn_core_kernel_source(monkeypatch):
    for e in (DF.ENV_COND_ROWS, DF.ENV_BIAS_ROWS, DF.ENV_Q_ROWS, DF.ENV_BAND_ROWS, DF.ENV_BIAS_CACHE, DF.ENV_WORK_GB, DF.ENV_BIAS_CACHE_GB):
        monkeypatch.delenv(e, raising=False)
    lay = Layout(4000, 4, 1, 128)
    kw = dict(c_z=128, c_in=267, c_cond=128, H=16, S=5, n_blocks=24, c_pair=16, budget_bytes=40 * 2 ** 30, record=False)
    today = DF.DiffusionSchedule.decide(lay, **kw)
    keys_today = [k for k, _ in today.fields()]
    assert keys_today == ["diff_cond_rows", "diff_bias_rows", "diff_q_rows", "diff_band_rows", "diff_bias_cache", "diff_rows_source", "diff_work_bytes", "diff_replicated"]
    assert today.attn_core is None and today.q_rows_stock == today.q_rows and "diff_attn_core" not in dict(today.fields())
    same = DF.DiffusionSchedule.decide(lay, attn_core=None, **kw)
    assert same.fields() == today.fields()
    ker = DF.DiffusionSchedule.decide(lay, attn_core="kernel", **kw)
    assert ker.q_rows == lay.Rmax and ker.sources["q"] == "kernel" and ker.q_rows_stock == today.q_rows
    assert (ker.cond_rows, ker.bias_rows, ker.band_rows, ker.bias_cache) == (today.cond_rows, today.bias_rows, today.band_rows, today.bias_cache)
    f = dict(ker.fields())
    assert f["diff_attn_core"] == "kernel:q_stock=%d" % today.q_rows and "q:kernel" in f["diff_rows_source"]
    assert [k for k, _ in ker.fields()][:8] == keys_today                            # the new pair is appended, nothing renamed
    pinned = DF.DiffusionSchedule.decide(lay, attn_core="kernel", q_rows=12, **kw)     # an explicit q block wins over the kernel source
    assert pinned.q_rows == 12 and pinned.sources["q"] == "given"
    monkeypatch.setenv(DF.ENV_Q_ROWS, "64")
    envd = DF.DiffusionSchedule.decide(lay, attn_core="kernel", **kw)
    assert envd.q_rows == 64 and envd.sources["q"].startswith("env")
    with pytest.raises(RowpairRefused, match="attn_core"):
        DF.DiffusionSchedule.decide(lay, attn_core="flashy", **kw)


def test_rows_face_static_admission(monkeypatch):
    assert A.rows_cell_word("dit", 16, 48) == "ditrows_h16d48" and A.rows_cell_word("pf", 16, 24) == "pfrows_h16d24"
    assert A.rows_cell_word("dit", 8, 48) is None and A.rows_cell_word("ditrows_h16d48") == "ditrows_h16d48" and A.rows_cell_word(None) is None
    for cw in A.ROWS_CELL_WORDS:
        assert cw not in A.CELL_WORDS                                                 # the square table's cell words are untouched (its 7-field key tests hold)
    # the ROW word apb_attn: fp32 refused by name; bf16 admitted; D > 64 and cc < 8 refused
    with pytest.raises(A.Refusal) as ei:
        A.admits_rows("apb_attn", "9.0", "fp32", head_dim=48, heads=16, word="apb_attn")
    assert ei.value.kind == "dtype:float32" and ei.value.fallback is None
    assert A.admits_rows("apb_attn", "9.0", "bf16", head_dim=48, heads=16, word="apb_attn") == ""
    assert A.admits_rows("apb_attn", (9, 0), "fp32", head_dim=48, heads=16, word="big") == "cast:bf16"   # the documented cast policy under a tier word
    with pytest.raises(A.Refusal, match="head_dim"):
        A.admits_rows("apb_attn", "9.0", "bf16", head_dim=72, word="big")
    with pytest.raises(A.Refusal, match="cc:7.5"):
        A.admits_rows("apb_attn", "7.5", "bf16", head_dim=48, word="big")
    with pytest.raises(A.Refusal, match="rows:row:fpf_apb"):
        A.admits_rows("fpf_apb", "9.0", "bf16")
    assert A.admits_rows("sba", "9.0", "fp32", variant="tf32", head_dim=48) == "" and A.admits_rows("naive", "0.0", "fp32") == ""
    with pytest.raises(A.Refusal, match="variant"):
        A.admits_rows("sba", "9.0", "fp32", variant="fp64")
    # select_rows: tier words
    monkeypatch.setitem(A._TABLE, "t", dict(A.table(), rows_cells={}))               # no rows cell measured (as released)
    sel = A.select_rows("9.0", "bf16", "dit", 2048, 16384, word="big", samples=5, heads=16, head_dim=48)
    assert sel.row == "apb_attn" and sel.cell is None and sel.size_measured is False and "beyond_measured" in sel.reason and sel.fallback is None
    sel32 = A.select_rows("9.0", "fp32", "dit", 2048, 16384, word="big", samples=5, heads=16, head_dim=48)
    assert sel32.row == "apb_attn" and "cast:bf16" in sel32.reason
    for w in ("exact", "faithful"):
        with pytest.raises(A.Refusal) as ei:
            A.select_rows("9.0", "fp32", "dit", 2048, 16384, word=w, samples=5, heads=16, head_dim=48)
        assert ei.value.kind.startswith("tier:%s" % w)
    with pytest.raises(A.Refusal, match="rows:row:fpf_apb"):
        A.select_rows("9.0", "bf16", "dit", 256, 4096, word="fpf_apb")
    with pytest.raises(A.Refusal, match="rows:word"):
        A.select_rows("9.0", "bf16", "dit", 256, 4096, word="no_such_word")
    s2 = A.select_rows("9.0", "fp32", "pf", 512, 8192, word="sba:ieee", samples=1, heads=16, head_dim=24)
    assert (s2.row, s2.variant, s2.reason.split(",")[0]) == ("sba", "ieee", "row_word")
    assert A.select_rows("9.0", "fp32", "dit", 8, 64, word="sdpa", samples=2).variant == "auto"
    with pytest.raises(A.Refusal, match="cc:0.0"):                                   # a CPU process: the kernel rows refuse by name, the caller's statement serves
        A.select_rows((0, 0), "bf16", "dit", 8, 64, word="big", samples=2, heads=16, head_dim=48)
    line = A.describe(sel)
    assert "apb_attn" in line and "big" in line
    d = A.describe_rows()
    assert d["rows"] == A.ROWS_ROWS and len(d["key_grammar"].replace("<eager|graph>", "<timing>").split("|")) == 8 and json.dumps(d)


def test_rows_cell_key_bucket_rule(monkeypatch):
    cells = {"9.0|bf16|ditrows_h16d48|S5|Q<=2048|N<=16384|eager|fwd": {"big": "apb_attn", "fast": "apb_attn"},
             "9.0|bf16|ditrows_h16d48|S5|Q<=8192|N<=16384|eager|fwd": {"big": "apb_attn"},
             "9.0|bf16|ditrows_h16d48|S5|Q<=4096|N<=8192|eager|fwd": {"big": "sba:tf32"},
             "9.0|bf16|ditrows_h16d48|S1|Q<=4096|N<=8192|eager|fwd": {"big": "sdpa"},
             "9.0|bf16|ditrows_h16d48|S5|N<=8192|eager|fwd": {"big": "naive"}}          # a 7-field key is not a rows key: ignored
    monkeypatch.setitem(A._TABLE, "t", dict(A.table(), rows_cells=cells))
    assert A.rows_cell_key("9.0", "bf16", "ditrows_h16d48", 2048, 16384, samples=5) == ("9.0|bf16|ditrows_h16d48|S5|Q<=2048|N<=16384|eager|fwd", True)
    assert A.rows_cell_key("9.0", "bf16", "ditrows_h16d48", 2049, 16384, samples=5) == ("9.0|bf16|ditrows_h16d48|S5|Q<=8192|N<=16384|eager|fwd", True)
    assert A.rows_cell_key("9.0", "bf16", "ditrows_h16d48", 9000, 16384, samples=5) == ("9.0|bf16|ditrows_h16d48|S5|Q<=8192|N<=16384|eager|fwd", False)   # beyond the largest Q bucket
    assert A.rows_cell_key("9.0", "bf16", "ditrows_h16d48", 100, 5000, samples=5) == ("9.0|bf16|ditrows_h16d48|S5|Q<=4096|N<=8192|eager|fwd", True)
    assert A.rows_cell_key("9.0", "bf16", "ditrows_h16d48", 100, 20000, samples=5)[1] is False                 # beyond the largest N bucket
    assert A.rows_cell_key("9.0", "bf16", "ditrows_h16d48", 100, 5000, samples=3) == (None, False)               # no S3 family
    assert A.rows_cell_key("10.0", "bf16", "ditrows_h16d48", 100, 5000, samples=5) == (None, False)
    sel = A.select_rows("9.0", "bf16", "dit", 3000, 8192, word="big", samples=5, heads=16, head_dim=48)
    assert (sel.row, sel.variant, sel.size_measured) == ("sba", "tf32", True) and sel.cell.endswith("Q<=4096|N<=8192|eager|fwd")
    sel1 = A.select_rows("9.0", "bf16", "dit", 3000, 8192, word="big", samples=1, heads=16, head_dim=48)
    assert (sel1.row, sel1.variant) == ("sdpa", "auto")
    far = A.select_rows("9.0", "bf16", "dit", 3000, 70320, word="big", samples=5, heads=16, head_dim=48)
    assert far.row == "apb_attn" and far.size_measured is False and "beyond_measured" in far.reason


@needs_torch
def test_reference_rows_and_cpu_refusal():
    import torch
    g = torch.Generator().manual_seed(3)
    S, H, R, NK, D = 2, 3, 5, 11, 4
    q, k, v = torch.randn(S, H, R, D, generator=g), torch.randn(S, H, NK, D, generator=g), torch.randn(S, H, NK, D, generator=g)
    bias = torch.randn(H, R, NK, generator=g)
    km = torch.ones(NK)
    km[-2:] = 0
    gate = torch.randn(S, R, H * D, generator=g)
    o = A.reference_rows(q, k, v, bias, km, gate, scale=0.5)
    lg = torch.einsum("shid,shjd->shij", q.double() * 0.5, k.double()) + bias.double()[None] + ((1 - km.double()) * -1e9)[None, None, None, :]
    ref = (torch.softmax(lg, -1) @ v.double()).permute(0, 2, 1, 3).reshape(S, R, H * D) * torch.sigmoid(gate.double())
    assert o.shape == (S, R, H * D) and torch.allclose(o, ref, atol=1e-12, rtol=0)
    o4 = A.reference_rows(q, k, v, bias[None].expand(S, -1, -1, -1), km[None], None, scale=0.5)
    assert torch.allclose(o4, ref / torch.sigmoid(gate.double()), atol=1e-9)
    # the naive row serves on CPU (the statement); apb_attn refuses by name on a CPU process; shape refusals are named
    on, sel = A.pair_bias_attention_rows(q, k, v, bias, km, word="naive", scale=0.5, gate=gate)
    assert sel.row == "naive" and on.dtype == q.dtype and torch.allclose(on.double(), ref, atol=1e-5)
    with pytest.raises(A.Refusal, match="dtype:float32"):                              # the ROW word never casts: fp32 refused by name
        A.pair_bias_attention_rows(q, k, v, bias, km, word="apb_attn")
    with pytest.raises(A.Refusal, match="cc:0.0"):                                     # the tier word casts, then the CPU process refuses by capability
        A.pair_bias_attention_rows(q, k, v, bias, km, word="big")
    with pytest.raises(A.Refusal, match="rows:shape:bias"):
        A.pair_bias_attention_rows(q, k, v, bias[:, :3], km, word="naive")
    with pytest.raises(A.Refusal, match="rows:shape:key_mask"):
        A.pair_bias_attention_rows(q, k, v, bias, km[:5], word="naive")
    core, sel2, why = DF.dit_rows_core("big", dtype=torch.float32, heads=16, head_dim=48, samples=5, record=False)
    assert core is None and sel2 is None and why.startswith("refused:cc:0.0")            # CPU: nothing bound, the engine statement by name
    assert DF.dit_rows_core("engine", dtype=torch.float32, heads=16, head_dim=48, record=False) == (None, None, "engine")
    assert DF.dit_rows_core(None, dtype=torch.float32, heads=16, head_dim=48, record=False) == (None, None, "engine")
    t32, t16 = torch.zeros(3), torch.zeros(3, dtype=torch.bfloat16)
    assert DF.cast16(t32).dtype == torch.bfloat16 and DF.cast16(t16) is t16 and DF.cast16(None) is None and DF.cast16(t32, torch.float16).dtype == torch.float16


@needs_torch
def test_pair_bias_rows_into_refuses_p1_and_bad_spec():
    import torch
    lay = Layout(64, 1, 0, 16)
    z = torch.zeros(1, 64, 64, 4)
    with pytest.raises(RowpairRefused, match="n_gpu=1"):
        DF.pair_bias_rows_into(lambda zr, o: None, z, lay, heads=2, dtype=torch.float32)
    lay2 = Layout(64, 2, 0, 32)                                                       # no process group: refused like every rowpair statement (group=no)
    with pytest.raises(RowpairRefused, match="pair_bias_rows_into"):
        DF.pair_bias_rows_into("not callable", z[:, :32], lay2, heads=2, dtype=torch.float32)


# ================================================================================================================ multi-process (gloo)
# ======================================================================================== census hygiene: every VALUE is ONE token (0.5.220.2)
_TOKEN = r"\S+"


@needs_torch
def test_diffusion_census_values_are_single_tokens(monkeypatch):
    """The diffusion / sampler-schedule census words on a synthetic threaded P=2 run (DiffusionSchedule.decide with a rows core bound,
    pair_bias_rows_into under the ln_proj / ln_proj_fp32 words — refused by name on CPU -> engine:<reason> — the rows slot under big / exact /
    a square-only row / an unknown word — refused by name -> stock:<kind> — and dit_rows_core's engine / refused / served values incl. the
    beyond_measured selection a CUDA process records) plus tri-attention's StageSlot words: every VALUE matches ``\\S+`` (a kit's single-token
    LEVER-line writer exits rc 2 on a blank). Also: every rows-face Refusal.kind / Selection.reason reachable on CPU is a token.
    Dicts covered: evidence.schedule() after the diffusion paths (keys dit_rows_core, dit_rows, dit_bias, diff_attn_core, diff_rows_source,
    diff_* fields, diff_noise_sync*) and after triatt.StageSlot.plane/aside/release (triatt_stage, triatt_stage_aside, triatt_stage_served)."""
    import re
    import torch
    from opt_core.testing import run_ranks
    from opt_core.mem.rowpair import evidence as EV
    from opt_core.mem.rowpair import triatt as TA
    from opt_core.mem.rowpair.shard import shard_rows
    sys.path.insert(0, HERE) if HERE not in sys.path else None
    from test_rowpair_diffusion_043 import _Engine

    def is_tok(v):
        return re.fullmatch(_TOKEN, str(v)) is not None

    # (a) the rows face's words on CPU: refusal kinds and selection reasons
    kinds = []
    for w in ("exact", "faithful", "fpf_apb", "no such word", "apb_attn", "sba:tf32", "sdpa"):
        try:
            sel = A.select_rows((0, 0), "fp32", "dit", 8, 64, word=w, samples=5, heads=16, head_dim=48)
            kinds.append(("reason", w, sel.reason))
        except A.Refusal as r:
            kinds.append(("kind", w, r.kind))
    for cc, dt, w in (((9, 0), "fp32", "big"), ((9, 0), "bf16", "fast"), ((9, 0), "bf16", "naive"), ((8, 0), "fp16", "apb_attn")):
        sel = A.select_rows(cc, dt, "dit", 2048, 16384, word=w, samples=5, heads=16, head_dim=48)
        kinds.append(("reason", w, sel.reason)); kinds.append(("token", w, DF.rows_core_token(sel)))
    q = torch.zeros(1, 2, 3, 8); k = torch.zeros(1, 2, 5, 8)
    for args in ((q, k, k, torch.zeros(2, 3, 4), None), (q, k, torch.zeros(1, 2, 4, 8), torch.zeros(2, 3, 5), None), (q[0], k, k, torch.zeros(2, 3, 5), None),
                 (q, k, k, torch.zeros(2, 3, 5), torch.zeros(3, 5)), (q, k.double(), k, torch.zeros(2, 3, 5), None)):
        try:
            A.pair_bias_attention_rows(*args, word="naive")
            kinds.append(("served", "naive", "ok"))
        except A.Refusal as r:
            kinds.append(("kind", "shape", r.kind))
    bad = [x for x in kinds if not is_tok(x[2])]
    assert not bad, bad
    assert any("beyond_measured" in x[2] for x in kinds) and any(x[2].startswith("tier:exact") for x in kinds) and any(x[2].startswith("rows:shape") for x in kinds)

    # (b) dit_rows_core's recorded values: engine / refused (CPU) / served beyond_measured (as on a CUDA process) / a row word
    EV.reset_schedule()
    vals = []
    for w, dev_cc in (("engine", None), ("big", None), ("exact", None), ("fpf_apb", None), ("big", (9, 0)), ("sba:ieee", (9, 0)), ("naive", (9, 0))):
        if dev_cc is not None:
            monkeypatch.setattr(A, "_device_cc", lambda device, _cc=dev_cc: _cc)
        DF.dit_rows_core(w, dtype=torch.float32, heads=16, head_dim=48, samples=5, kind="dit")
        vals.append((w, dict(EV.schedule()).get("dit_rows_core")))
        monkeypatch.undo()
    bad = [x for x in vals if not is_tok(x[1])]
    assert not bad, bad
    assert any("beyond_measured" in str(v) and "cast:bf16" in str(v) for _w, v in vals), vals      # the reported offender, now one token

    # (c) the sampler paths on a threaded P=2 run: schedule + producer + rows slot census words
    N, B = 40, 16

    def entry(rank, P):
        torch.set_num_threads(1)
        EV.reset_schedule()
        DF.dit_rows_census(reset=True); DF.dit_bias_census(reset=True)
        lay = Layout(N, P, rank, B)
        eng = _Engine(N, seed=7)
        with torch.no_grad():
            z_loc = shard_rows(eng.z, lay, dim=-3)
            sched = DF.DiffusionSchedule.decide(lay, c_z=eng.c_z, c_in=eng.c_z + eng.c_rel, c_cond=eng.c, H=eng.H, S=eng.S, n_blocks=2, c_pair=eng.C_pair,
                                                cond_rows=16, bias_rows=8, band_rows=20, bias_cache=False, attn_core="kernel")   # q_rows: the kernel source (all local rows)
            zc_loc = DF.pair_cond_rows(eng.embed_rows, z_loc, lay, c_out=eng.c, rows=sched.cond_rows, transitions=eng.transitions_rows)
            spec = DF.DitBias(engine=eng.blocks_fns[0].bias, ln_weight=eng.blocks[0]["ln_z"].weight, ln_bias=eng.blocks[0]["ln_z"].bias,
                              weight=eng.blocks[0]["Wb"].t(), eps=eng.blocks[0]["ln_z"].eps)
            for word in ("engine", "ln_proj", "ln_proj_fp32"):
                DF.pair_bias_rows_into(spec, zc_loc, lay, rows=5, word=word)
            a_in, s_in = eng.step_inputs(0)
            for core_word in ("big", "exact", "fpf_apb", "no such word"):
                blocks = _rows_blocks(eng, core_word, sched.q_rows_stock)
                DF.diffusion_transformer_sharded(blocks, a_in.clone(), s_in, zc_loc, lay, schedule=sched)
            DF.dit_rows_core("big", dtype=torch.float32, heads=eng.H, head_dim=eng.c_h, samples=eng.S)
        slot = TA.StageSlot()                                                          # tri-attention's stage-once words, CPU-constructible
        with slot.plane("once"):
            slot.aside("int32:bias"); slot.aside("staged window refused")
            slot.served = 1
        return dict(EV.schedule())

    scheds = run_ranks(2, entry)
    covered = set()
    for r, sched in enumerate(scheds):
        for key in ("dit_rows", "dit_bias", "dit_rows_core", "diff_attn_core", "diff_rows_source", "diff_q_rows", "triatt_stage", "triatt_stage_aside"):
            assert key in sched, (r, key, sorted(sched))
        assert "stock:" in str(sched["dit_rows"]) and "engine:" in str(sched["dit_bias"]), sched
        bad = {k: v for k, v in sched.items() if not is_tok(v)}
        assert not bad, f"rank {r}: census values with blanks: {bad}"
        covered.update(sched)
    print("CENSUS_KEYS_COVERED " + ",".join(sorted(covered)))


def _rows_blocks(eng, core_word, stock_q_rows):
    """The synthetic engine's blocks re-bound for the rows slot: kv -> DitKV (k / v as they are: fp32 CPU), attn = dit_attention_rows around the
    engine's q projection / epilogue with today's attn as the stock statement; bias_into = DitBias(the block's LN + projection weights)."""
    import torch
    out = []
    for blk, fns in zip(eng.blocks, eng.blocks_fns):
        def kv(x, fns=fns):
            k, v, mask_bias = fns.kv(x)
            return DF.DitKV(DF.cast16(k, k.dtype), DF.cast16(v, v.dtype), eng.token_mask, (k, v, mask_bias))   # cast16 to its own dtype: identity (CPU fp32 test)

        def q_fn(x_q, blk=blk):
            B_, S_, nq, _ = x_q.shape
            return (x_q @ blk["Wq"][0] + blk["Wq"][1]).view(B_, S_, nq, eng.H, eng.c_h).permute(0, 1, 3, 2, 4) / eng.c_h ** 0.5

        def out_fn(o, x_q, blk=blk):                                                  # o [1, S, q, H*D]
            return (torch.sigmoid(x_q @ blk["Wg"][0] + blk["Wg"][1]) * o) @ blk["Wo"]

        attn = DF.dit_attention_rows(q_fn=q_fn, out_fn=out_fn, stock_fn=fns.attn, num_heads=eng.H, core_word=core_word, scale=1.0,
                                     stock_q_rows=stock_q_rows)
        into = DF.DitBias(engine=fns.bias, ln_weight=blk["ln_z"].weight, ln_bias=blk["ln_z"].bias, weight=blk["Wb"].t(), eps=blk["ln_z"].eps)
        out.append(DF.DiTBlockFns(fns.norm, kv, attn, fns.update, fns.bias, into))
    return out


def _entry_rows(N: int, B: int, steps: int = 2):
    import torch
    from opt_core.mem.rowpair import dist as D
    from opt_core.mem.rowpair import evidence as EV
    from opt_core.mem.rowpair.shard import shard_rows
    from test_rowpair_diffusion_043 import _Engine
    torch.set_num_threads(1)
    P, r = D.world()
    lay = D.Layout.checked(N, P, r, B)
    eng = _Engine(N, seed=7)
    res = {"rank": r, "P": P, "N": N, "R": lay.R, "checks": {}, "metrics": {}, "bitwise": {}, "census": {}}
    chk, met, bit = res["checks"], res["metrics"], res["bitwise"]
    with torch.no_grad():
        z_loc = shard_rows(eng.z, lay, dim=-3)
        sched = DF.DiffusionSchedule.decide(lay, c_z=eng.c_z, c_in=eng.c_z + eng.c_rel, c_cond=eng.c, H=eng.H, S=eng.S, n_blocks=2, c_pair=eng.C_pair,
                                            cond_rows=16, bias_rows=8, q_rows=12, band_rows=20, bias_cache=False, attn_core="kernel")
        fields0 = dict(EV.schedule_fields())
        chk["schedule_kernel_recorded"] = (sched.attn_core == "kernel" and sched.q_rows == 12 and "q:given" in str(fields0.get("diff_rows_source"))
                                           and str(fields0.get("diff_attn_core", "")).startswith("kernel:q_stock=12"))       # q given wins over the kernel source; the pair is recorded
        zc_loc = DF.pair_cond_rows(eng.embed_rows, z_loc, lay, c_out=eng.c, rows=sched.cond_rows, transitions=eng.transitions_rows)
        fns0 = eng.blocks_fns[0]
        blk0 = eng.blocks[0]
        # ---------------------------------------------------------------- E2: pair_bias_rows_into (engine word) is bitwise pair_bias_rows
        ref_bias = DF.pair_bias_rows(fns0.bias, zc_loc, lay, rows=5)
        spec = DF.DitBias(engine=fns0.bias, ln_weight=blk0["ln_z"].weight, ln_bias=blk0["ln_z"].bias, weight=blk0["Wb"].t(), eps=blk0["ln_z"].eps)
        DF.dit_bias_census(reset=True)
        got_engine = DF.pair_bias_rows_into(spec, zc_loc, lay, rows=5, word="engine")
        bit["into_engine_vs_pair_bias_rows"] = chk["into_engine_vs_pair_bias_rows"] = bool(torch.equal(got_engine, ref_bias)) and got_engine.dtype == ref_bias.dtype
        chk["census_engine"] = DF.dit_bias_census().startswith("engine:")
        got_one = DF.pair_bias_rows_into(spec, zc_loc, lay, rows=0, word="engine", heads=eng.H, dtype=ref_bias.dtype)    # one block, explicit allocation
        bit["into_one_block_vs_blocks"] = chk["into_one_block_vs_blocks"] = bool(torch.equal(got_one, ref_bias))
        got_adapter = DF.pair_bias_rows_into(lambda zr, o: o.copy_(fns0.bias(zr).movedim(-1, -3)), zc_loc, lay, rows=3, probe_fn=fns0.bias)
        bit["into_adapter_callable"] = chk["into_adapter_callable"] = bool(torch.equal(got_adapter, ref_bias)) and "adapter:" in DF.dit_bias_census()
        DF.dit_bias_census(reset=True)
        got_lnproj = DF.pair_bias_rows_into(spec, zc_loc, lay, rows=5, word="ln_proj")                                  # CPU: the kernel steps aside by name per block
        cz = DF.dit_bias_census()
        res["census"]["dit_bias_lnproj_cpu"] = cz
        bit["into_lnproj_cpu_is_engine"] = chk["into_lnproj_cpu_is_engine"] = bool(torch.equal(got_lnproj, ref_bias))
        chk["census_lnproj_cpu_named"] = cz.startswith("engine:") and cz.split(":")[1].split(",")[0] in ("not_cuda", "import", "width", "dtype") and "ln_proj:" not in cz   # the synthetic c is not a served width: refused by name BEFORE the device
        try:
            DF.pair_bias_rows_into("not callable", zc_loc, lay, heads=2, dtype=ref_bias.dtype)
            chk["bad_spec_refused"] = False
        except RowpairRefused as e:
            chk["bad_spec_refused"] = "spec must be" in str(e)
        try:
            DF.pair_bias_rows_into(lambda zr, o: None, zc_loc, lay)
            chk["no_alloc_facts_refused"] = False
        except RowpairRefused as e:
            chk["no_alloc_facts_refused"] = "heads= and dtype=" in str(e)
        got_fp32w = DF.pair_bias_rows_into(spec, zc_loc, lay, rows=5, word="ln_proj_fp32")                             # CPU: refused by name like ln_proj
        bit["into_lnproj_fp32_cpu_is_engine"] = chk["into_lnproj_fp32_cpu_is_engine"] = bool(torch.equal(got_fp32w, ref_bias))
        pre = torch.full_like(ref_bias, float("nan"))
        got_pre = DF.pair_bias_rows_into(spec, zc_loc, lay, rows=4, word="engine", out=pre)
        chk["into_prealloc"] = got_pre is pre and bool(torch.equal(pre, ref_bias))
        # ---------------------------------------------------------------- transformer: blocks carrying bias_into (engine / ln_proj-on-CPU) == today's blocks
        blocks_rows = _rows_blocks(eng, "big", stock_q_rows=5)
        blocks_into_only = [f._replace(kv=g.kv, attn=g.attn) for f, g in zip(blocks_rows, eng.blocks_fns)]              # bias_into set, today's attention
        DF.dit_rows_census(reset=True)
        for step in range(steps):
            a_in, s_in = eng.step_inputs(step)
            plain = DF.diffusion_transformer_sharded(eng.blocks_fns, a_in, s_in, zc_loc, lay, schedule=sched)
            os.environ[DF.ENV_DIFF_BIAS] = "ln_proj"
            try:
                with_into = DF.diffusion_transformer_sharded(blocks_into_only, a_in, s_in, zc_loc, lay, schedule=sched)
                with_rows = DF.diffusion_transformer_sharded(blocks_rows, a_in, s_in, zc_loc, lay, schedule=sched)       # E1 + E2 bound; CPU: stock by name
            finally:
                os.environ.pop(DF.ENV_DIFF_BIAS, None)
            bit[f"step{step}_into_vs_plain"] = chk[f"step{step}_into_vs_plain"] = bool(torch.equal(with_into, plain))
            d = float((with_rows - plain).abs().max())
            met[f"step{step}_rows_vs_plain"] = d
            bit[f"step{step}_rows_vs_plain"] = bool(torch.equal(with_rows, plain))
            chk[f"step{step}_rows_vs_plain"] = d <= TOL32
            dense = eng.dense_transformer(a_in, s_in, eng.dense_cond(eng.z))
            met[f"step{step}_rows_vs_dense"] = float((with_rows - dense).abs().max())
            chk[f"step{step}_rows_vs_dense"] = met[f"step{step}_rows_vs_dense"] <= TOL32
        rc = DF.dit_rows_census()
        res["census"]["dit_rows_cpu"] = rc
        chk["rows_census_cpu_named_stock"] = rc.startswith("stock:cc:0.0:") and "apb_attn" not in rc.split("stock:")[0]
        fields = dict(EV.schedule_fields())
        chk["census_recorded"] = "dit_rows" in fields and "dit_bias" in fields and fields.get("diff_attn_core", "").startswith("kernel:q_stock=")
    res["ok"] = all(chk.values())
    import torch.distributed as tdist
    allres = [None] * P
    tdist.all_gather_object(allres, res)
    return {"P": P, "ranks": allres, "rows": [x["R"] for x in allres], "ok": all(x["ok"] for x in allres)}


def _mp(P, entry, *args):
    from opt_core.mem.rowpair import launch
    prev = os.environ.get("ROWPAIR_TEST_DEVICE")
    os.environ["ROWPAIR_TEST_DEVICE"] = "cpu"
    try:
        return launch.run_sharded(P, entry, *args, mode="big", backend="gloo", cpu_ok=True, nccl_timeout_s=120, run_timeout_s=600)
    finally:
        if prev is None:
            os.environ.pop("ROWPAIR_TEST_DEVICE", None)
        else:
            os.environ["ROWPAIR_TEST_DEVICE"] = prev


@needs_torch
@pytest.mark.parametrize("P,N,B", [(2, 120, 32), (3, 192, 64)])   # 0.5.220.6 gate fix: was (3, 160, 64) — Layout refuses N < P*B (=192) by name; N=192 keeps the P=3 / B=64 intent
def test_mp_sampler_rows(P, N, B):
    res = _mp(P, _entry_rows, N, B, 2)
    print("RESULT " + json.dumps(res, sort_keys=True, default=str))
    assert len(res["ranks"]) == P
    bad = {x["rank"]: [k for k, v in x["checks"].items() if not v] for x in res["ranks"] if not x["ok"]}
    assert not bad, (bad, [(x["rank"], x["metrics"], x["census"]) for x in res["ranks"]])


if __name__ == "__main__":                                                       # python tests/test_rowpair_sampler_rows_xp4.py [P N B]
    P, N, B = (int(v) for v in (sys.argv[1:4] + ["2", "120", "32"][len(sys.argv) - 1:])[:3])
    out = _mp(P, _entry_rows, N, B, 2)
    print("RESULT " + json.dumps(out, sort_keys=True, default=str))
    sys.exit(0 if out["ok"] else 1)
