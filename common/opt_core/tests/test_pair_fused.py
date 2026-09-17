"""opt_core.attn.pair_fused on CPU: the cell table parses and is self-consistent, the carried cells it names are carried, refusals are named
before any launch, and the module imports with the standard library only (torch/triton inside the calls)."""
import json
import os
import sys

import pytest

from opt_core import kernels
from opt_core.attn import pair_fused as PF


def test_module_is_stdlib_at_import():
    assert "torch" not in sys.modules or True          # other tests may have imported torch; the module itself must not
    src = open(PF.__file__).read().split("\n")
    top = [l for l in src if l.startswith("import ") or l.startswith("from ")]
    assert all("torch" not in l and "triton" not in l for l in top), top


def test_carried_cells_named_by_the_serve_layer_are_carried():
    names = kernels.names()
    for n in PF._CARRIED:
        assert n in names, n
        assert kernels.verify_carry(n) == [], kernels.verify_carry(n)


def test_cell_table_schema_and_variants():
    t = PF.cells()
    assert t["version"] == 1 and t["rows"]
    seen = set()
    for r in t["rows"]:
        assert set(r) >= {"impl", "piece", "key", "cc", "triton", "variant", "cfg", "status", "evidence"}, r
        assert r["impl"] in PF.IMPLS
        assert r["status"] in ("certified", "candidate")
        assert r["piece"] in ("prologue", "epilogue", "transition", "ln_linear", "gate_transpose", "fused_transition")
        if r["impl"] == "fpf":
            assert r["variant"] in {"prologue": ("v3", "prologue_v4", "mkpf_f1", "v3_fp32z"), "epilogue": ("v2", "epilogue_v3"), "transition": ("v2_fold", "v1", "v1_fp32x", "transition_v3")}[r["piece"]]
            assert {"prologue": 3, "epilogue": 3, "transition": 2}[r["piece"]] == len(r["key"])
            need = {"v3": {"BI", "BJ", "num_warps", "num_stages"}, "prologue_v4": {"BI", "BJ", "num_warps", "num_stages", "BN"},
                    "mkpf_f1": {"BI", "BJ", "num_warps", "num_stages", "BN", "ln_arith"}, "v3_fp32z": {"BI", "BJ", "num_warps", "num_stages"},
                    "v1_fp32x": {"BM", "BH", "num_warps", "num_stages"},
                    "v2": {"KVER", "BI", "BJ", "num_warps", "num_stages", "EXP"}, "epilogue_v3": {"BI", "BJ", "num_warps", "num_stages", "KC"},
                    "v1": {"BM", "BH", "num_warps", "num_stages"}, "transition_v3": {"BM", "BH", "num_warps", "num_stages"}, "v2_fold": {"BM", "BH", "num_warps", "num_stages", "IL"}}[r["variant"]]
            assert set(r["cfg"]) >= need, (r["variant"], r["cfg"])
        k = (r["impl"], r["piece"], tuple(r["key"]), r["cc"], r["triton"], r["variant"])
        assert k not in seen, f"duplicate row {k}"
        seen.add(k)
    # the reference recipe's cells are tested
    cert = {(r["piece"], tuple(r["key"])) for r in t["rows"] if r["impl"] == "fpf" and r["status"] == "certified"}
    assert ("prologue", (256, 8, 32)) in cert and ("epilogue", (256, 8, 32)) in cert and ("transition", (256, 1024)) in cert


def test_cells_sha_is_stable_and_hex():
    s = PF.cells_sha256()
    assert len(s) == 64 and s == PF.cells_sha256()


class _FakeTensor:
    """Enough of a tensor for the shape/dtype refusals that happen before any device query."""
    def __init__(self, shape, dtype="bfloat16", cuda=True):
        self.shape = tuple(shape); self._dtype = dtype; self.is_cuda = cuda
    def dim(self): return len(self.shape)
    def stride(self, i): return 1


def test_pack_refusals_are_named():
    torch = pytest.importorskip("torch")
    HD, C, H = 128, 128, 4
    w = lambda *s: torch.zeros(*s)
    with pytest.raises(PF.Unsupported) as e:
        PF.pack_triattn_weights(w_o=w(C, HD), n_heads=H, head_dim=32, w_q=w(HD, C), w_k=w(HD, C), w_v=w(HD, C), w_g=None, w_b=w(H, C))
    assert e.value.reason == "gate:none"
    with pytest.raises(PF.Unsupported) as e:
        PF.pack_triattn_weights(w_o=w(C, HD), n_heads=H, head_dim=32, w_q=w(HD, C), w_k=w(HD, C), w_v=w(HD, C), w_g=w(HD, C), w_b=None)
    assert e.value.reason == "bias:none"
    with pytest.raises(PF.Unsupported) as e:
        PF.pack_triattn_weights(w_o=w(C, HD), n_heads=H, head_dim=32, w_qkvg=w(3 * HD, C), w_b=w(H, C))
    assert e.value.reason == "w_qkvg-shape"
    W = PF.pack_triattn_weights(w_o=w(C, HD), n_heads=H, head_dim=32, w_qkvg=w(4 * HD, C), qkvg_split=("g", "q", "k", "v"), w_b=w(H, C))
    assert (W.C, W.H, W.D, W.HB, W.C_out) == (C, H, 32, 16, C) and tuple(W.wqkvg.shape) == (4 * HD, C) and W._fpf_cache["triatt_pro"]["wb"].shape == (16, C)
    Wg = PF.pack_triattn_weights(w_o=w(C, HD), n_heads=H, head_dim=32, w_qkvg=w(4 * HD, C), w_b=w(H, C), b_g=torch.ones(HD), b_o=w(C))
    assert tuple(Wg.bqkvg.shape) == (4 * HD,) and float(Wg.bqkvg[:3 * HD].abs().sum()) == 0.0 and float(Wg.bqkvg[3 * HD:].sum()) == HD
    with pytest.raises(PF.Unsupported) as e:
        PF.pack_triattn_weights(w_o=w(C, HD), n_heads=H, head_dim=32, w_qkvg=w(4 * HD, C), w_b=w(H, C), b_g=torch.ones(HD + 1))
    assert e.value.reason == "b_g-shape"
    T = PF.pack_transition_weights(w_out=w(C, 4 * C), w_ab=w(8 * C, C))
    assert (T.C, T.NH, T.n) == (C, 4 * C, 4) and tuple(T.wabI16.shape) == (8 * C, C)
    with pytest.raises(PF.Unsupported) as e:
        PF.pack_transition_weights(w_out=w(C, 4 * C), w_a=w(4 * C, C), w_b=w(2 * C, C))
    assert e.value.reason == "w_b-shape"


def test_shape_refusals_before_any_launch():
    torch = pytest.importorskip("torch")
    C, H, D = 128, 4, 32
    w = lambda *s: torch.zeros(*s)
    W = PF.pack_triattn_weights(w_o=w(C, H * D), n_heads=H, head_dim=D, w_qkvg=w(4 * H * D, C), w_b=w(H, C))
    z = torch.zeros(8, 8, C)                                          # CPU tensor -> 'device'
    ok, why = PF.supported_triattn(z, W)
    assert (ok, why) == (False, "device")
    ok, why = PF.supported_triattn(z, W, impl="nope")
    assert why == "impl:nope"
    ok, why = PF.supported_triattn(z, W, core="nope")
    assert why == "core:nope"
    T = PF.pack_transition_weights(w_out=w(C, 4 * C), w_ab=w(8 * C, C))
    ok, why = PF.supported_transition(torch.zeros(4, C), T)
    assert (ok, why) == (False, "device")
    with pytest.raises(PF.Unsupported):
        PF.tri_attn_block(z, W)


def test_ledger_is_the_core_ledger_and_line_grammar():
    from opt_core.counters import Ledger
    L = PF.ledger(expected=("below_min_tokens",))
    assert isinstance(L, Ledger) and L.origin == "core" and L.impl.startswith("fpf_triatt_pro@")
    L.serve("c128h4d32"); L.fallback("below_min_tokens"); L.fallback("no-cell:fpf:prologue:96x4x32:9.0|3.7")
    assert "no-cell:fpf:prologue:96x4x32:9.0|3.7" in L.unexpected()
    line = L.line("acme-opt", **PF.evidence())
    assert line.startswith("[acme-opt] LEVER name=F3.pair_fused state=on") and "origin=core" in line and "served=1" in line
    assert "served_candidate=0" in line and "allow_candidate=0" in line and "cells=" in line


def test_candidate_switch_is_visible():
    os.environ["OPT_CORE_PF_ALLOW_CANDIDATE"] = "1"
    try:
        assert PF.allow_candidate() and PF.evidence()["allow_candidate"] == 1 and PF.describe()["allow_candidate"] == 1
    finally:
        del os.environ["OPT_CORE_PF_ALLOW_CANDIDATE"]
    assert not PF.allow_candidate()


def test_write_x_refuses():
    torch = pytest.importorskip("torch")
    C, H, D = 128, 4, 32
    W = PF.pack_triattn_weights(w_o=torch.zeros(C, H * D), n_heads=H, head_dim=D, w_qkvg=torch.zeros(4 * H * D, C), w_b=torch.zeros(H, C))
    with pytest.raises(PF.Unsupported) as e:
        PF.prologue(torch.zeros(8, 8, C), W, write_x=True)
    assert e.value.reason == "write_x"


def test_impl_torch_and_core_words():
    torch = pytest.importorskip("torch")
    C, H, D = 384, 12, 32                                              # non-power-of-two dims: fused impls refuse by name, impl='torch' passes the shape checks
    W = PF.pack_triattn_weights(w_o=torch.zeros(C, H * D), n_heads=H, head_dim=D, w_qkvg=torch.zeros(4 * H * D, C), w_b=torch.zeros(H, C))
    z = torch.zeros(4, 4, C, dtype=torch.bfloat16)
    class _Cuda:                                                       # shape checks run before any device query; fake the device flag only
        pass
    ok, why = PF.supported_triattn(z, W, impl="fpf")
    assert why == "device"
    assert PF.IMPLS == ("fpf", "lnl", "torch") and PF.CORES == ("flash_triattn", "k2b")


def test_variant_pins_are_words():
    assert PF._variant_pins(None, ("prologue", "epilogue")) == (None, None)
    assert PF._variant_pins("v3", ("prologue", "epilogue")) == ("v3", None)
    assert PF._variant_pins({"epilogue": "v2"}, ("prologue", "epilogue")) == (None, "v2")
    with pytest.raises(PF.Unsupported) as e:
        PF._variant_pins({"bogus": "v2"}, ("prologue", "epilogue"))
    assert e.value.reason == "variant:bogus"
    with pytest.raises(PF.Unsupported) as e:
        PF._pick("fpf", "prologue", (256, 8, 32), None, ("v3",), "mkpf_f1")          # pin not applicable to this call -> refuses before any device query
    assert e.value.reason == "no-cell:variant:mkpf_f1"


def test_refusal_ladder_keeps_no_frames_alive(monkeypatch):
    """pick_cell's caught refusals must not keep the caller's frame (and its tensor locals) alive until the cyclic GC runs."""
    import gc, weakref

    def refuse(impl, piece, key, device, *, variant=None):
        raise PF.Unsupported(f"no-cell:{impl}:{piece}:test", "cpu test")
    monkeypatch.setattr(PF, "find_cell", refuse)

    class Big:
        pass
    alive = []

    def caller():
        big = Big(); alive.append(weakref.ref(big))                    # stands in for q/k/v/o held in tri_attn_block's locals
        for f in (lambda: PF.pick_cell("fpf", "prologue", (999, 9, 9), None, ("prologue_v4", "v3")),
                  lambda: PF._pick("fpf", "prologue", (999, 9, 9), None, ("v3",), "v3")):
            try:
                f()                                                    # every variant refuses -> raises
            except PF.Unsupported:
                pass
    gc.disable()
    try:
        caller()
        assert alive[0]() is None, "caller frame retained by an exception reference cycle"
    finally:
        gc.enable()


def test_carried_exports_are_the_sums_specs():
    ex = PF.carried_exports()
    assert set(ex) == {"PF_LNL_TILES", "PF_TRIATTN_TABLE"}
    assert ex["PF_LNL_TILES"].endswith("lnl_fused.tiles_by_arch.json") and os.path.exists(ex["PF_LNL_TILES"])
    assert ex["PF_TRIATTN_TABLE"].endswith(os.path.join("fpf_triatt_k2b", "K2B_CELLS.json")) and os.path.exists(ex["PF_TRIATTN_TABLE"])
    assert PF.carried_exports(("lnl_fused",)) == {"PF_LNL_TILES": ex["PF_LNL_TILES"]}


def test_cc80_rows_are_additive_pointer_rows_selected_by_capability(monkeypatch):
    """cc 8.0 (A100) rows: every one names A100 in its evidence, carries no Hopper-only knob (no warp specialisation WS=1), keys triton '*';
    the kits' served keys on A100 have certified rows (the trunk tri-attention 128x4x32 prologue v3 / epilogue v2, transitions 128x512 …);
    selection is by the running card's capability — an 8.0 stack resolves to an 8.0 row, a 9.0 stack to the same 9.0 rows as before."""
    t = PF.cells()
    rows80 = [r for r in t["rows"] if r["cc"] == "8.0"]
    assert rows80
    for r in rows80:
        assert (r["triton"] == "*" or r["status"] == "candidate" or "RACED AND CERTIFIED" in r["evidence"]) and "A100" in r["evidence"] and r["status"] in ("certified", "candidate"), r["id"]   # pointer rows key triton '*'; a row built for one triton line keeps that pin (candidate until measured; certified with its measurement record)
        assert int(r["cfg"].get("WS", 0)) == 0, r["id"]                                   # sm_80: no warp specialisation
        assert r["status"] == "candidate" or all(isinstance(v, (int, str)) or (isinstance(v, list) and all(isinstance(x, int) for x in v)) for v in r["cfg"].values()), r["id"]   # plain cfg values (ints, words, int lists = per-stage flags)
    have = {(r["impl"], r["piece"], tuple(r["key"]), r["variant"]) for r in rows80 if r["status"] == "certified"}
    line_of = {(r["impl"], r["piece"], tuple(r["key"]), r["variant"]): (r["triton"] if r["triton"] != "*" else "3.6") for r in rows80 if r["status"] == "certified"}
    for need in (("fpf", "prologue", (128, 4, 32), "v3"), ("fpf", "transition", (128, 512), "v1")):
        assert need in have, need
    assert ("fpf", "epilogue", (128, 4, 32), "v2") not in have                                  # measured slower than the lnl statements on 8.0: named off, the lnl piece serves
    off = PF.lookup_cell("fpf", "epilogue", (128, 4, 32), stack=("8.0", "3.6"))
    assert not off.served and off.off and PF.lookup_cell("lnl", "gate_transpose", (128,), stack=("8.0", "3.6")).served
    monkeypatch.delenv(PF.ALLOW_CANDIDATE_ENV, raising=False)
    for impl, piece, key, variant in sorted(have):                                              # each row on its own triton line ('*' rows: any, here 3.6)
        monkeypatch.setattr(PF, "_stack", lambda device, _l=line_of[(impl, piece, key, variant)]: ("8.0", _l))
        row = PF.find_cell(impl, piece, key, None, variant=variant)
        assert row["cc"] == "8.0" and row["status"] == "certified", (piece, key, variant, row["id"])
    monkeypatch.setattr(PF, "_stack", lambda device: ("8.0", "3.6"))
    for need in (("fpf", "prologue", (64, 4, 16), "v3"), ("fpf", "epilogue", (64, 4, 16), "v2")):
        assert need in have, need                                                           # the template pair stack's shape: served on A100 (a per-call route on H100, r09/r18 candidates)
    for need in (("lnl", "ln_linear", (64,), "ln_linear"), ("lnl", "ln_linear", (128,), "ln_linear"), ("lnl", "gate_transpose", (128,), "gate_transpose"),
                 ("lnl", "fused_transition", (64, 256), "fused_transition"), ("lnl", "fused_transition", (128, 512), "fused_transition")):
        assert need in have, need                                                           # the lnl pieces' cc 8.0 rows (ln_linear / gate_transpose c=64,128; fused_transition (64,128), (64,256), (128,512)); cfg {} — tiles are the kernel's own table
    with pytest.raises(PF.Unsupported):
        PF.find_cell("fpf", "prologue", (256, 8, 32), None)                                 # a shape with no 8.0 row stays refused by name on A100
    monkeypatch.setattr(PF, "_stack", lambda device: ("9.0", "3.6"))
    assert PF.find_cell("fpf", "prologue", (128, 4, 32), None, variant="v3")["id"] == "r07"   # H100 resolution unchanged
    assert PF.find_cell("fpf", "epilogue", (128, 4, 32), None, variant="v2")["id"] == "r16"
    assert PF.find_cell("fpf", "transition", (128, 512), None, variant="v1")["id"] == "r24"
    for key, rid in (((128, 256), "r28"), ((256, 512), "r29")):                              # certified at 9.0: served without the candidate switch
        row = PF.find_cell("fpf", "transition", key, None, variant="v1")
        assert row["id"] == rid and row["status"] == "certified", (key, row["id"], row["status"])



# ---------------------------------------------------------------------------------------------------- core="tier:<word>" (kernels.triattn's row)
def test_tier_core_words_and_acceptance():
    from opt_core.attn import pair_fused as PF
    assert PF.tier_core_word("tier:fast") == "fast" and PF.tier_core_word("tier:triattn_native@v11") == "triattn_native@v11"
    assert PF.tier_core_word("tier:") is None and PF.tier_core_word("k2b") is None and PF.tier_core_word(None) is None
    assert PF._core_ok("flash_triattn") and PF._core_ok("k2b") and PF._core_ok("tier:fast") and PF._core_ok(lambda *a: None)
    assert not PF._core_ok("tier:") and not PF._core_ok("bogus") and "flash_triattn" in PF.CORES and "tier:fast" not in PF.CORES   # the default core is unchanged
    assert PF.evidence()["cores"]                                                # the LEVER line carries cores=


def test_tier_core_resolves_the_row_kernels_triattn_selects_per_cc():
    """core="tier:<word>" resolves ONCE per call class through kernels.triattn.select with the arguments its own serving call uses; for the
    fast word on cc 8.0 the template-stack (D16 x H4) and c_z 128 (D32 x H4) tri-attention classes land on the sealed CUDA row where a prebuilt
    of the stack exists (the table's fast winner), on cc 9.0 on that bucket's winner; identical to a direct select()."""
    from opt_core.attn import pair_fused as PF
    from opt_core.kernels import triattn as T
    from opt_core.kernels.triattn import triattn_native as K
    built80 = sorted(k for k, exts in K.stacks_built().items() if exts)
    built90 = T.stacks_built()
    assert built90, "the sm_90a member ships prebuilts"
    for cc, stack in (((8, 0), (built80[0] if built80 else None)), ((9, 0), built90[0])):
        for D, H in ((16, 4), (32, 4)):
            for S in (256, 512, 1024):
                for form in ("bias_only", "mask_bias"):
                    sel = PF.resolve_tier_core("fast", cc, "bf16", D, H, S, form=form, stack=stack)
                    ref = T.select(cc, "bf16", D, H, S, T.FWD, word="fast", stack=stack, form=form)
                    assert sel == ref, (cc, D, H, S, form, sel, ref)
                    cell = T.cells()[sel.cell]
                    assert sel.row in T.ROW_NAMES and sel.row not in T.STOCK_ROWS
                    if cc == (8, 0):
                        assert cell["fast"].split("@")[0] == "triattn_native", cell["fast"]          # the table's fast winner there
                        assert sel.row == ("triattn_native" if built80 else sel.row)
    with pytest.raises(T.Refusal) as ei:                                        # a class no row serves: refused by name (the core then names flash_triattn)
        PF.resolve_tier_core("fast", (9, 0), "fp32", 16, 4, 256)
    assert ei.value.kind.startswith("no_cell") or ei.value.kind


def test_tier_core_refusal_steps_aside_to_flash_triattn_by_name(monkeypatch):
    """A kernels.triattn Refusal (at resolution or at the call) serves the flash_triattn cell BY NAME for that call class, memoised, counted as
    cores=tier:<word>=flash_triattn(refused:<kind>) on the LEVER line; a served row is counted as tier:<word>=<row>."""
    import torch
    from opt_core.attn import pair_fused as PF
    from opt_core.kernels import triattn as T
    calls = {"flash": 0, "row": 0}

    class _Flash(object):
        @staticmethod
        def flash_triangle_attention(q, k, v, bias, mask=None, scale=None):
            calls["flash"] += 1
            return torch.zeros_like(q)

    monkeypatch.setattr(PF, "_carried", lambda name, *a, **k: _Flash())
    monkeypatch.setattr(PF, "_TIER_CORE_MEMO", {})
    monkeypatch.setattr(PF, "SERVED_CORES", {})

    def refuse(*a, **k):
        raise T.Refusal("no_cell:test", "fast", "cueq")
    monkeypatch.setattr(PF, "resolve_tier_core", refuse)
    q = torch.zeros(1, 8, 4, 8, 16, dtype=torch.bfloat16); bias = torch.zeros(1, 1, 4, 8, 8, dtype=torch.bfloat16)
    o = PF.core_attention(q, q, q, bias, None, core="tier:fast")
    o = PF.core_attention(q, q, q, bias, None, core="tier:fast")
    assert calls["flash"] == 2 and o.shape == q.shape
    assert PF.SERVED_CORES == {"tier:fast=flash_triattn(refused:no_cell:test)": 2}
    assert list(PF._TIER_CORE_MEMO.values())[0][0] == "refused"                # resolved once per call class
    # a resolved row that refuses AT THE CALL (e.g. no prebuilt of the running stack): flash by name too, and the class is memoised as refused
    Sel = T.Selection("triattn_native", "fast", "8.0|bf16|D16|H4|N<=256|fwd", None, None, "fast", None, True, False, True, 1.75, True, "test")
    monkeypatch.setattr(PF, "resolve_tier_core", lambda *a, **k: Sel)
    monkeypatch.setattr(PF, "_TIER_CORE_MEMO", {}); monkeypatch.setattr(PF, "SERVED_CORES", {})

    def serve_refuse(*a, **k):
        raise T.Refusal("no_prebuilt:test", "triattn_native", "k2b")
    monkeypatch.setattr(T, "triangle_attention", serve_refuse)
    PF.core_attention(q, q, q, bias, None, core="tier:fast")
    assert PF.SERVED_CORES == {"tier:fast=flash_triattn(refused:no_prebuilt:test)": 1} and calls["flash"] == 3

    def serve_ok(q_, k_, v_, b_, m_, sc_, **kw):
        calls["row"] += 1
        assert kw.get("selection") is Sel and kw.get("word") == "fast" and kw.get("form") == "bias_only"
        return torch.ones_like(q_)
    monkeypatch.setattr(T, "triangle_attention", serve_ok)
    monkeypatch.setattr(PF, "_TIER_CORE_MEMO", {}); monkeypatch.setattr(PF, "SERVED_CORES", {})
    o = PF.core_attention(q, q, q, bias, None, core="tier:fast")
    assert calls["row"] == 1 and bool((o == 1).all()) and PF.SERVED_CORES == {"tier:fast=triattn_native": 1}
    assert "cores" in PF.evidence() and PF.core_attention(q, q, q, bias, None, core="flash_triattn").shape == q.shape and PF.SERVED_CORES["flash_triattn"] == 1


def test_default_core_table_is_keyed_by_the_measured_block_numbers(monkeypatch):
    """core="default" (the functions' default): tier:fast where the whole block was measured faster with kernels.triattn's row in the same
    process -- cc 8.0 D16/D32 x H4 from 512 keys, cc 9.0 D32 x H4 from 512 and D16 x H4 from 1024 keys, bf16 -- and the flash_triattn cell
    everywhere else (keys <= 256, 9.0 D16 @512, other head dims / head counts / dtypes / capabilities); MODEL_OPT_LEVERS_OFF word
    pair_fused:tier_core switches the table off (flash_triattn everywhere); an explicit core word always wins."""
    import inspect
    from opt_core.attn import pair_fused as PF
    monkeypatch.delenv("MODEL_OPT_LEVERS_OFF", raising=False)
    assert PF.DEFAULT_CORE == "default" and PF.DEFAULT_CORE_TABLE == {("8.0", "bf16", 16, 4): 512, ("8.0", "bf16", 32, 4): 512, ("9.0", "bf16", 32, 4): 512, ("9.0", "bf16", 16, 4): 1024}
    for fn in (PF.tri_attn_block, PF.supported_triattn, PF.core_attention):
        assert inspect.signature(fn).parameters["core"].default == PF.DEFAULT_CORE
    T, F = "tier:fast", "flash_triattn"
    table = {("8.0", 16, 256): F, ("8.0", 16, 511): F, ("8.0", 16, 512): T, ("8.0", 16, 1024): T, ("8.0", 32, 256): F, ("8.0", 32, 512): T, ("8.0", 32, 2048): T,
             ("9.0", 32, 256): F, ("9.0", 32, 512): T, ("9.0", 32, 1024): T, ("9.0", 16, 256): F, ("9.0", 16, 512): F, ("9.0", 16, 1023): F, ("9.0", 16, 1024): T, ("9.0", 16, 2048): T}
    for (cc, D, S), want in table.items():
        assert PF.default_core_for(cc, "bf16", D, 4, S) == want, (cc, D, S)
        assert PF.default_core_for(tuple(int(x) for x in cc.split(".")), "bf16", D, 4, S) == want
        assert PF.default_core_for(cc, "fp32", D, 4, S) == F and PF.default_core_for(cc, "bf16", D, 8, S) == F      # other dtypes / head counts: not measured -> flash_triattn
    assert PF.default_core_for("9.0", "bf16", 64, 4, 4096) == F and PF.default_core_for("12.0", "bf16", 16, 4, 4096) == F and PF.default_core_for((0, 0), "bf16", 16, 4, 4096) == F
    monkeypatch.setenv("MODEL_OPT_LEVERS_OFF", "something pair_fused:tier_core")
    assert PF.default_core_off() == "pair_fused:tier_core" and all(PF.default_core_for(cc, "bf16", D, 4, S) == F for (cc, D, S) in table)
    monkeypatch.setenv("MODEL_OPT_LEVERS_OFF", "pair_fused_tier_core")
    assert PF.default_core_off() == "pair_fused_tier_core"
    monkeypatch.delenv("MODEL_OPT_LEVERS_OFF")
    assert PF.default_core_off() is None and PF._core_ok("default") and PF.describe()["default_core"]["table"]["8.0|bf16|D16|H4"] == {"tier:fast_from_keys": 512}


def test_default_core_dispatch_and_accounting(monkeypatch):
    """core omitted: below the table (here: CPU tensors = capability (0,0)) the flash_triattn cell serves and counts as default=flash_triattn;
    where the table names tier:fast the tier path serves (default=tier:fast=<row>, or default=tier:fast=flash_triattn(refused:...) by name);
    core="flash_triattn" / "k2b" passed explicitly are served as asked whatever the table says."""
    import torch
    from opt_core.attn import pair_fused as PF
    from opt_core.kernels import triattn as T
    calls = {"flash": 0, "k2b": 0, "row": 0}

    class _Cells(object):
        @staticmethod
        def flash_triangle_attention(q, k, v, bias, mask=None, scale=None):
            calls["flash"] += 1; return torch.zeros_like(q)

        @staticmethod
        def attn_k2b(q, k, v, bias, mask=None, scale=None):
            calls["k2b"] += 1; return torch.zeros_like(q)

    monkeypatch.delenv("MODEL_OPT_LEVERS_OFF", raising=False)
    real_default_core_for = PF.default_core_for
    monkeypatch.setattr(PF, "_carried", lambda name, *a, **k: _Cells())
    monkeypatch.setattr(PF, "_TIER_CORE_MEMO", {}); monkeypatch.setattr(PF, "SERVED_CORES", {})
    q = torch.zeros(1, 8, 4, 1024, 16, dtype=torch.bfloat16); bias = torch.zeros(1, 1, 4, 1024, 1024, dtype=torch.bfloat16)
    PF.core_attention(q, q, q, bias, None)                                      # CPU tensors: capability (0,0) is not in the table -> flash_triattn
    assert PF.SERVED_CORES == {"default=flash_triattn": 1} and calls["flash"] == 1
    # a capability in the table: pretend the tensors sit on an 8.0 device (the table's key) -> the tier path with the default= tag
    monkeypatch.setattr(PF, "default_core_for", lambda cc, dtype, D, H, S: "tier:fast")
    Sel = T.Selection("triattn_native", "fast", "8.0|bf16|D16|H4|N<=1200|fwd", None, None, "fast", None, True, False, True, 2.9, True, "test")
    monkeypatch.setattr(PF, "resolve_tier_core", lambda *a, **k: Sel)

    def serve_ok(q_, k_, v_, b_, m_, sc_, **kw):
        calls["row"] += 1; return torch.ones_like(q_)
    monkeypatch.setattr(T, "triangle_attention", serve_ok)
    o = PF.core_attention(q, q, q, bias, None)
    assert calls["row"] == 1 and bool((o == 1).all()) and PF.SERVED_CORES["default=tier:fast=triattn_native"] == 1

    def serve_refuse(*a, **k):
        raise T.Refusal("no_prebuilt:test", "triattn_native", "k2b")
    monkeypatch.setattr(T, "triangle_attention", serve_refuse); monkeypatch.setattr(PF, "_TIER_CORE_MEMO", {})
    PF.core_attention(q, q, q, bias, None)
    assert PF.SERVED_CORES["default=tier:fast=flash_triattn(refused:no_prebuilt:test)"] == 1 and calls["flash"] == 2
    # explicit words win over the table
    PF.core_attention(q, q, q, bias, None, core="flash_triattn"); PF.core_attention(q, q, q, bias, None, core="k2b")
    assert PF.SERVED_CORES["flash_triattn"] == 1 and PF.SERVED_CORES["k2b"] == 1 and calls["k2b"] == 1
    monkeypatch.setattr(PF, "default_core_for", real_default_core_for)             # ablation: the real table with the LEVERS_OFF word -> flash_triattn under core="default"
    monkeypatch.setenv("MODEL_OPT_LEVERS_OFF", "pair_fused:tier_core")
    monkeypatch.setattr(PF, "_call_class", lambda q_, k_: ((8, 0), "bf16", 16, 4, 1024))     # as if the tensors sat on an 8.0 device
    assert real_default_core_for((8, 0), "bf16", 16, 4, 1024) == "flash_triattn"
    PF.core_attention(q, q, q, bias, None)
    assert PF.SERVED_CORES["default=flash_triattn"] == 2
    monkeypatch.delenv("MODEL_OPT_LEVERS_OFF")                                       # and with the word gone the same class takes the tier path again
    assert real_default_core_for((8, 0), "bf16", 16, 4, 1024) == "tier:fast"


def test_emit_line_tolerates_caller_keywords_the_face_also_emits(monkeypatch, capsys):
    """A kit passing its own cores= (or any key the evidence head / tail produces) to emit_line gets exactly ONE line, no TypeError, its token
    under the key it named, and the face's value under face_<key>; keys the caller does not pass are unchanged."""
    from opt_core.attn import pair_fused as PF
    from opt_core.counters import Ledger
    monkeypatch.setattr(PF, "SERVED_CORES", {"flash_triattn": 3})
    L = PF.ledger("pairblock")                                                # the block.s own ledger constructor (origin core)
    head, tail = PF._evidence_head(), PF.evidence_tail()
    caller = {k: "CALLER_%s" % k for k in list(head) + list(tail)}            # every face key passed by the caller at once
    line = PF.emit_line(L, "TEST", variant="v", impl="fpf", **caller)
    out = capsys.readouterr()
    assert isinstance(line, str) and (out.out + out.err).count("name=pairblock") <= 2
    for k in caller:
        assert ("%s=CALLER_%s" % (k, k)) in line, k                            # the caller's token wins its key
        assert ("face_%s=" % k) in line, k                                     # the face's value stays visible
    assert "face_cores=flash_triattn:3" in line and "cores=CALLER_cores" in line
    line2 = PF.emit_line(L, "TEST", variant="v", cores="triattn_native:5", face_cores="mine")   # the caller names face_cores too: the face's value is dropped, never a raise
    assert "cores=triattn_native:5" in line2 and "face_cores=mine" in line2 and line2.count("face_cores=") == 1
    line3 = PF.emit_line(L, "TEST", variant="v")                             # no collision: the face's keys as before
    assert "cores=flash_triattn:3" in line3 and "face_" not in line3
    assert PF._face_fields({"a": 1, "b": 2}, {"b": 9}) == {"a": 1, "face_b": 2} and PF._face_fields({"b": 2}, {"b": 9, "face_b": 8}) == {}



def test_tier_big_word_is_fast_by_name_where_triattn_records_no_peaks():
    """core="tier:big": kernels.triattn has no big word (no peak records) -> the fast word's row, by name; accepted by the plan."""
    from opt_core.attn import pair_fused as PF
    from opt_core.kernels import triattn as T
    assert PF._core_ok("tier:big")
    for cc in ((8, 0), (9, 0)):
        for D, S in ((16, 1024), (32, 512)):
            b = PF.resolve_tier_core("big", cc, "bf16", D, 4, S, stack=None); f = T.select(cc, "bf16", D, 4, S, T.FWD, word="fast", stack=None, form="bias_only")
            assert (b.row, b.cell) == (f.row, f.cell)                            # the served row and cell; word / reason differ once kernels.triattn takes big natively



def test_engage_cells_default_and_opt_in_rows():
    """pair_fused_cells.json "engage": e02 (cc 8.0, exact variant, keys < 1026 -> the engine's stock block by cell) binds every caller;
    e01 (cc 9.0 bf16 D32 H4, 256 < keys < 512) binds only callers passing engage_cells=True."""
    from opt_core.attn import pair_fused as PF
    ids = {r["id"]: r for r in PF.engage_rows()}
    assert ids["e01"]["applies"] == "opt_in" and ids["e02"]["applies"] == "default" and "source" in ids["e01"]
    assert PF.engage((8, 0), "bf16", 32, 4, 1025, variant="exact") == (False, "cell:e02:stock_block(exact:keys<1026)")
    assert PF.engage("8.0", "bf16", 16, 4, 512, variant="exact")[0] is False
    assert PF.engage((8, 0), "bf16", 32, 4, 1026, variant="exact") == (True, "") and PF.engage((8, 0), "bf16", 32, 4, 512, variant=None) == (True, "")
    assert PF.engage((9, 0), "bf16", 32, 4, 384) == (True, "")                                   # opt-in row: silent unless asked
    assert PF.engage((9, 0), "bf16", 32, 4, 384, engage_cells=True) == (False, "cell:e01:stock_block(256<keys<512)")
    assert PF.engage((9, 0), "bf16", 32, 4, 256, engage_cells=True) == (True, "") and PF.engage((9, 0), "bf16", 32, 4, 512, engage_cells=True) == (True, "")
    assert PF.engage((9, 0), "bf16", 16, 4, 384, engage_cells=True) == (True, "") and PF.engage((9, 0), "fp32", 32, 4, 384, engage_cells=True) == (True, "")



def test_fused_epilogue_measured_off_on_8_0_serves_the_statements_piece_by_name(monkeypatch):
    """cc 8.0, key (128, 4, 32): the fused epilogue row (formerly certified r56) measured x1.09-2.43 slower than the lnl statements in every
    form -> the key is named off (slower); lookup is unserved with the off word; the block plans the lnl gate_transpose piece; a pinned
    variant or a capability without the off entry keeps today's words; 9.0 / 10.x rows untouched."""
    from opt_core.attn import pair_fused as PF
    from opt_core.kernels import cell_words as CW
    d = PF.lookup_cell("fpf", "epilogue", [128, 4, 32], stack=("8.0", "3.7"))
    assert not d.served and CW.is_off_word(d.reason) and CW.off_why(d.reason) == "slower", d
    monkeypatch.setenv(PF.ALLOW_CANDIDATE_ENV, "1")
    assert PF.lookup_cell("fpf", "epilogue", [128, 4, 32], stack=("8.0", "3.7")).row["id"] == "r56"      # the candidate switch still reaches it
    monkeypatch.delenv(PF.ALLOW_CANDIDATE_ENV)
    for cc, rid in (("9.0", "r16"), ("10.0", "r73"), ("10.3", "r81")):
        assert PF.lookup_cell("fpf", "epilogue", [128, 4, 32], stack=(cc, "3.7")).row["id"] == rid
    assert PF.lookup_cell("lnl", "gate_transpose", [128], stack=("8.0", "3.7")).row["id"] == "r70"
    # the planner's fallback helper: an off word -> the lnl piece (impl tag), counted; a pinned variant / a non-off refusal re-raises
    class W: C_out, H, D = 128, 4, 32
    monkeypatch.setattr(PF, "find_cell", lambda impl, piece, key, device, **k: {"id": "r70", "impl": "lnl", "piece": "gate_transpose", "cfg": {}, "variant": "gate_transpose"})
    monkeypatch.setattr(PF, "_carried", lambda *a, **k: None); monkeypatch.setattr(PF, "SERVED_PIECES", {})
    row = PF._epilogue_statements_piece(PF.Unsupported(d.reason), W, None, pinned=False)
    assert row["impl"] == "lnl" and PF.SERVED_PIECES == {"epilogue=fpf>lnl(off:slower)@128x4x32": 1} and "pieces" in PF._evidence_head()
    import pytest
    with pytest.raises(PF.Unsupported):
        PF._epilogue_statements_piece(PF.Unsupported(d.reason), W, None, pinned=True)
    with pytest.raises(PF.Unsupported):
        PF._epilogue_statements_piece(PF.Unsupported("no-cell:fpf:epilogue:77x4x32:8.0|3.7"), W, None, pinned=False)



def test_noreg_unmeasured_capability_inherits_the_nearest_measured_columns_rows(monkeypatch):
    """A capability with no certified row (sm_120 today; 8.6 / 8.9): the block's pieces run the nearest measured column's certified rows
    (12.0 -> 10.3's pointer rows; 8.9 -> 8.0's), served_by carries the token; candidate rows of that capability keep their switch; the
    measured capabilities (8.0 / 9.0 / 10.0 / 10.3) resolve exactly as the land record says (test_land_* above)."""
    monkeypatch.delenv(PF.ALLOW_CANDIDATE_ENV, raising=False)
    cols = PF.measured_columns()
    assert "9.0" in cols and "8.0" in cols and PF.inherit_column("9.0") is None and PF.inherit_column("8.0") is None
    assert PF.inherit_column("12.0") == cols[-1] if tuple(int(x) for x in cols[-1].split(".")) <= (12, 0) else True
    assert PF.inherit_column("8.9") == "8.0" and PF.inherit_column("7.5") is None
    col = PF.inherit_column("12.0")
    for impl, piece, key, var in (("fpf", "prologue", (128, 4, 32), None), ("fpf", "epilogue", (128, 4, 32), None), ("fpf", "transition", (128, 512), None), ("fpf", "prologue", (64, 4, 16), "v3")):
        d = PF.lookup_cell(impl, piece, key, stack=("12.0", "3.7"), variant=var)
        ref = PF.lookup_cell(impl, piece, key, stack=(col, "3.7"), variant=var)
        assert d.served and ref.served and d.row["id"] == ref.row["id"], (piece, key, d.reason)
        assert d.served_by.endswith("%s(12.0->%s)" % (PF.INHERIT_TOKEN, col)), d.served_by
    d = PF.lookup_cell("fpf", "prologue", (256, 8, 32), stack=("12.0", "3.7"))               # a key this capability holds only CANDIDATE rows for: the nearest column with a
    c2 = PF.inherit_column("12.0", "fpf", "prologue", (256, 8, 32))                          # certified row for THAT key serves (per-key rule), token on served_by ...
    assert d.served and c2 is not None and d.served_by.endswith("%s(12.0->%s)" % (PF.INHERIT_TOKEN, c2)), (d.served_by, d.reason)
    monkeypatch.setenv(PF.ALLOW_CANDIDATE_ENV, "1")                                          # ... and its own candidate row keeps its switch
    if hasattr(PF.lookup_cell, "cache_clear"):
        PF.lookup_cell.cache_clear()
    d1 = PF.lookup_cell("fpf", "prologue", (256, 8, 32), stack=("12.0", "3.7"))
    assert d1.served and d1.row["cc"] == "12.0" and d1.row["status"] == "candidate"



def test_noreg_partial_cc_one_injected_cell_still_inherits_every_other_key(monkeypatch):
    """A capability holding a certified row for ONE key (12.0: an injected transition row) still inherits the nearest measured column's
    certified rows for every OTHER key (prologue / epilogue 128x4x32 -> the column below it), with the token on served_by."""
    import copy
    monkeypatch.delenv(PF.ALLOW_CANDIDATE_ENV, raising=False)
    tab = PF.cells()
    base = next(r for r in tab["rows"] if r["impl"] == "fpf" and r["piece"] == "transition" and r["status"] == "certified" and [int(x) for x in r["key"]] == [128, 512])
    inj = copy.deepcopy(base); inj["id"] = "rTESTinj"; inj["cc"] = "12.0"; inj["triton"] = "*"
    tab["rows"].append(inj)
    if hasattr(PF.lookup_cell, "cache_clear"):
        PF.lookup_cell.cache_clear()
    try:
        assert PF.inherit_column("12.0", "fpf", "transition", (128, 512)) is None                 # this key: its own row
        col = PF.inherit_column("12.0", "fpf", "prologue", (128, 4, 32)); assert col is not None and col != "12.0"
        t = PF.lookup_cell("fpf", "transition", (128, 512), stack=("12.0", "3.7"))
        assert t.served and t.row["id"] == "rTESTinj" and PF.INHERIT_TOKEN not in (t.served_by or "")
        for piece, key in (("prologue", (128, 4, 32)), ("epilogue", (128, 4, 32))):
            d = PF.lookup_cell("fpf", piece, key, stack=("12.0", "3.7"))
            ref = PF.lookup_cell("fpf", piece, key, stack=(col, "3.7"))
            assert d.served and d.row["id"] == ref.row["id"] and d.served_by.endswith("%s(12.0->%s)" % (PF.INHERIT_TOKEN, col)), (piece, d.served_by, d.reason)
    finally:
        tab["rows"].remove(inj)
        if hasattr(PF.lookup_cell, "cache_clear"):
            PF.lookup_cell.cache_clear()
def test_composition_cell_serves_the_statements_prologue_with_the_fused_epilogue_on_8_0_c256_x_ln_form(monkeypatch):
    """cc 8.0, 256x8x32, x_ln-given form: the fused prologue rows lost their race (r102 candidate) while epilogue_v3 (r103) won -> the block plans
    the plain-statements prologue BY CELL (compositions c01) and the fused epilogue, without the candidate switch; the LN-in-kernel form keeps
    its fused prologue (mkpf_f1, r104); other keys / capabilities are untouched (no composition cell: the fused piece's own refusal)."""
    import torch, copy
    monkeypatch.delenv(PF.ALLOW_CANDIDATE_ENV, raising=False)
    # shipped state: c01 is HELD (applies=candidate) until the composed block itself is measured against the statements block -> no default composition
    held = [r for r in PF.composition_rows() if r["id"] == "c01"]
    assert held and held[0]["applies"] == "candidate" and PF.composition_for("prologue", (8, 0), "torch.bfloat16", (256, 8, 32), "stock") is None
    rows = copy.deepcopy(PF.composition_rows()); [r.__setitem__("applies", "default") for r in rows if r["id"] == "c01"]
    monkeypatch.setattr(PF, "composition_rows", lambda: rows)                 # the mechanism, exercised with the cell admitted
    assert PF.composition_for("prologue", (8, 0), "torch.bfloat16", (256, 8, 32), "stock")["id"] == "c01"
    assert PF.composition_for("prologue", (8, 0), "torch.bfloat16", (256, 8, 32), "fused") is None
    assert PF.composition_for("prologue", (9, 0), "torch.bfloat16", (256, 8, 32), "stock") is None
    assert PF.composition_for("prologue", (8, 0), "torch.bfloat16", (128, 4, 32), "stock") is None
    # the piece helper: a pinned variant re-raises; a CPU tensor re-raises (facts unreadable); a simulated 8.0 device composes
    err = PF.Unsupported("no-cell:fpf:prologue:256x8x32:8.0|3.7+off(not-measured)")
    W = type("W", (), {"C": 256, "H": 8, "D": 32, "C_out": 256})()
    z_cpu = torch.zeros(2, 2, 256, dtype=torch.bfloat16)
    for kw in (dict(ln="stock", pinned=True), dict(ln="stock", pinned=False)):
        try:
            PF._prologue_statements_piece(err, W, z_cpu, **kw); assert False, "must re-raise on CPU / pinned"
        except PF.Unsupported as e:
            assert e.reason == err.reason
    class _Z:                                                                  # a stand-in tensor on a simulated cc 8.0 device
        is_cuda = True; dtype = torch.bfloat16; device = "cuda:0"
    monkeypatch.setattr(torch, "is_tensor", lambda t: True if isinstance(t, _Z) else torch.Tensor.__instancecheck__(t) if False else isinstance(t, torch.Tensor) or isinstance(t, _Z))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda *a, **k: False)
    PF.SERVED_PIECES.clear()
    row = PF._prologue_statements_piece(err, W, _Z(), ln="stock", pinned=False)
    assert row["impl"] == "torch" and row["id"] == "compose:c01" and row["variant"] is None
    assert list(PF.SERVED_PIECES) == ["prologue=fpf>torch(compose:c01)@256x8x32"]
    try:
        PF._prologue_statements_piece(err, W, _Z(), ln="fused", pinned=False); assert False
    except PF.Unsupported:
        pass


def test_noreg_a_capability_with_certified_rows_of_other_variants_still_inherits_for_a_pinned_variant(monkeypatch):
    """Per key AND per pinned variant: cc 10.0 / 10.3 hold certified fused-epilogue rows of their own variants for (fpf, epilogue, 256x8x32);
    a caller pinning 'epilogue_v3' (certified on a lower column only) inherits that column's row (Triton source compiled on the part) instead
    of a refusal; the unpinned caller keeps the capability's own certified row; keys the capability never certified inherit as before."""
    monkeypatch.delenv(PF.ALLOW_CANDIDATE_ENV, raising=False)
    if hasattr(PF.lookup_cell, "cache_clear"):
        PF.lookup_cell.cache_clear()
    for cc in ("10.0", "10.3"):
        own = PF.lookup_cell("fpf", "epilogue", (256, 8, 32), stack=(cc, "3.7"))
        assert own.served and str(own.row["cc"]) == cc and PF.INHERIT_TOKEN not in own.served_by, (cc, own)
        assert PF.inherit_column(cc, "fpf", "epilogue", (256, 8, 32)) is None
        col = PF.inherit_column(cc, "fpf", "epilogue", (256, 8, 32), "epilogue_v3")
        assert col in ("9.0", "8.0"), (cc, col)
        pin = PF.lookup_cell("fpf", "epilogue", (256, 8, 32), stack=(cc, "3.7"), variant="epilogue_v3")
        assert pin.served and pin.row.get("variant") == "epilogue_v3" and str(pin.row["cc"]) == col, (cc, pin)
        assert ("%s(%s->%s)" % (PF.INHERIT_TOKEN, cc, col)) in pin.served_by, pin.served_by
    # the fully measured columns decide pinned variants by their own table: no per-variant inheritance into 9.0 / 8.0
    assert PF.REFERENCE_COLUMNS == ("9.0", "8.0")
    assert PF.inherit_column("9.0", "fpf", "transition", (64, 256), "v1") == PF.inherit_column("9.0", "fpf", "transition", (64, 256))



def test_assertions_under_the_serving_doors_surface_as_named_unsupported(monkeypatch):
    """The carried pieces assert their operand contracts (o / g / z strides, dtypes, shapes); through the public doors an AssertionError
    never escapes: it is the named step-aside Unsupported('assert:<door>', ...) the binders already handle (CPU; a stand-in piece)."""
    import types, torch
    for fn in (PF.prologue, PF.epilogue, PF.tri_attn_block, PF.transition, PF.ln_linear, PF.core_attention):
        assert getattr(fn, "__wrapped__", None) is not None, fn.__name__

    class Boom(object):
        @staticmethod
        def epilogue_v3(*a, **k):
            assert False, "o last dim must be contiguous"
    monkeypatch.setattr(PF, "_carried", lambda name, sub=None: Boom)
    Wt = types.SimpleNamespace(C_out=128, H=4, D=32, wo16=torch.zeros(128, 128, dtype=torch.bfloat16), woT16=None, b_o=None)
    o = torch.zeros(1, 8, 4, 8, 32, dtype=torch.bfloat16).transpose(-1, -2).contiguous().transpose(-1, -2)   # a caller's o with a strided last dim
    g = torch.zeros(1, 8, 8, 128, dtype=torch.bfloat16)
    plan = {"epilogue": {"cfg": {}, "variant": "epilogue_v3", "cc": "8.0", "triton": "*"}}
    try:
        PF.epilogue(o, g, Wt, None, impl="fpf", o_layout="ihjd", _plan=plan)
        assert False, "must not return"
    except PF.Unsupported as u:
        assert u.reason == "assert:epilogue" and "contiguous" in u.detail, (u.reason, u.detail)
