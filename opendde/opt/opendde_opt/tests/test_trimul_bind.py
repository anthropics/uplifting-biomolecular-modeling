"""levers/ARMT/odde_trimul_bind.py — the two TriMul routes through the core provider (opt_core.kernels.trimul): the word, the registry / modes /
floor / ablation rows, the install record, the evidence tokens (CPU); the served row's bytes through the provider face (GPU, skipped without CUDA)."""
import importlib
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(os.path.dirname(HERE))
ARMT = os.path.join(OPT, "forward", "fast_inference", "levers", "ARMT")
if ARMT not in sys.path:
    sys.path.insert(0, ARMT)


def _fresh(word):
    old = os.environ.get("ODDE_TRIMUL")
    if word is None:
        os.environ.pop("ODDE_TRIMUL", None)
    else:
        os.environ["ODDE_TRIMUL"] = word
    try:
        sys.modules.pop("odde_trimul_bind", None)
        return importlib.import_module("odde_trimul_bind")
    finally:
        if old is None:
            os.environ.pop("ODDE_TRIMUL", None)
        else:
            os.environ["ODDE_TRIMUL"] = old


def test_word_parsing():
    for w, exp in ((None, None), ("", None), ("0", None), ("off", None), ("stock", None), ("fast", "fast"), ("exact", "exact"), ("big", "big"), ("tmk3_exact", "tmk3_exact")):
        m = _fresh(w)
        assert m.WORD == exp and m.active() == (exp is not None) and m.COUNTS["active"] == (exp is not None)
    from opt_core.kernels import trimul as KT
    assert tuple(m.TIER_WORDS) == tuple(KT.TIER_WORDS) == ("fast", "exact", "big")   # the unit's tier words are the provider's (the big lines bind by `big`, 0.2.56)
    for cc in ((9, 0), (8, 0)):                                                         # the big word resolves for OpenDDE's shapes: SOME served row or a refusal by name (class contract)
        for C in (384, 64):
            try:
                sel = KT.select(cc, "bf16", C, C, 800, "outgoing", word="big", has_cueq=True)
                assert sel.row in KT.ROW_NAMES, (cc, C, sel.row)
            except KT.Refusal as r:
                assert r.row and r.kind, (cc, C)
    sys.modules.pop("odde_trimul_bind", None)


def test_provider_rows_and_cells_for_opendde_shapes():
    """CLASS contract over OpenDDE's TriMul shapes (c_z 384 pair stacks, c 64 template stack; 400 / 800 / 1200-row buckets; cc 9.0 and 8.0; both
    residencies): a tier word resolves to SOME row the provider will serve (admit) or refuses BY NAME (KT.Refusal carries .row / .kind) — never a
    particular winner, never a cell's numbers (the core flips winners between races).  The exact word admits only bitwise-class rows (KT.EXACT_ROWS)
    or names the stock op; the fast word any kernel row or the stock op by the cell's word."""
    from opt_core.kernels import trimul as KT
    for cc in ((9, 0), (8, 0)):
        for C in (384, 64):
            for n in (400, 800, 1200):
                for w in ("exact", "fast"):
                    for residency in (None, "fp32"):
                        kw = {} if residency is None else {"residency": residency}
                        try:
                            s = KT.select(cc, "bf16", C, C, n, "outgoing", word=w, has_cueq=True, **kw)
                        except KT.Refusal as r:                                  # a refusal is by name: the row and the kind are words
                            assert r.row and r.kind, (cc, C, n, w, residency, r)
                            continue
                        assert s.row in KT.ROW_NAMES, (cc, C, n, w, residency, KT.describe(s))
                        if w == "exact":
                            assert s.row in KT.EXACT_ROWS or s.row in KT.STOCK_ROWS, (cc, C, n, w, residency, KT.describe(s))   # bitwise class or the stock op itself
    m = _fresh("exact")
    assert m.N_MIN == 101 and not hasattr(m, "KIT_ROW") and not hasattr(m, "MIN_ROWS")        # no kit row, no row floor: the tier word decides at every row count (0.2.57)
    sys.modules.pop("odde_trimul_bind", None)


def test_registry_modes_floor_and_ablation_rows():
    from opendde_opt import modes, ran, registry, smalln
    for lv, tier, line in (("trimul_core", "tier2", "LSTAR2A"), ("trimul_exact", "exact", "S1")):
        assert registry.LEVERS[lv].tier == tier and registry.PIN_STATUS[lv][0] == "tested" and lv in ran.COUNTERS and lv in smalln.FLOOR_LEVERS
        assert lv in modes.LINES[line].levers and modes.LINES[line].exports.get("ODDE_TRIMUL") == ("fast" if lv == "trimul_core" else "exact")
        assert lv in modes.LEVER_SWITCHES and "ODDE_TRIMUL" in modes.LEVER_SWITCHES[lv]
        w = modes.line_without(modes.LINES[line], (lv,))
        assert "ODDE_TRIMUL" not in w.exports and lv not in w.levers
        assert lv in registry.ENGAGEMENT and registry.ENGAGEMENT[lv].stepped_aside is not None
    assert "odde_trimul_bind:trimul_out" in modes.LINES["S1"].exports["FPF_IMPL"] and "odde_trimul_bind:trimul_in" in modes.LINES["S1"].exports["FPF_IMPL"]
    for n in modes.BIG_LINES:                                                          # the memory mode's lines bind by their own tier word (0.2.56; fast's up to 0.2.55)
        assert "trimul_core" in modes.LINES[n].levers and modes.LINES[n].exports.get("ODDE_TRIMUL") == modes.BIG_TRIMUL_WORD == "big", n
        assert "ODDE_TRIMUL" not in modes.line_without(modes.LINES[n], ("trimul_core",)).exports
    assert "trimul_core" in modes.LEVER_DEPENDENTS["arm_u23"] and "trimul_exact" in modes.LEVER_DEPENDENTS["fpf_trimul_exact"]
    w = modes.line_without(modes.LINES["S1"], ("fpf_trimul_exact",))
    assert "trimul_exact" not in w.levers and "FPF_IMPL" not in w.exports and "ODDE_TRIMUL" not in w.exports
    for site in ("cueq_trimul",):
        assert "trimul_core" in registry.KERNEL_SITE_OWNERS[site] and "trimul_exact" in registry.KERNEL_SITE_OWNERS[site]
    assert registry.validate() == []


def test_adapter_is_the_stock_forward_by_name_when_the_word_is_unset(monkeypatch):
    """FPF_IMPL names the unit's callables on the exact line; with ODDE_TRIMUL unset (the lever left out by name) every call is fpf_engines'
    FPFFallback -- the adapter runs the stock forward, counted (no kit construction behind the word since 0.2.57)."""
    import types
    m = _fresh(None)
    fb = type("FPFFallback", (Exception,), {})
    monkeypatch.setitem(sys.modules, "fpf_engines", types.SimpleNamespace(FPFFallback=fb))
    mod = types.SimpleNamespace(c_z=384, c_hidden=384, training=False, _outgoing=True)
    with pytest.raises(fb) as e:
        m.trimul_out(mod, "z", None, True, True, 256, "cuequivariance")
    assert str(e.value) == "provider:no_word" and m.COUNTS["calls"] == 0 and m.COUNTS["stock_calls"] == 1 and m.COUNTS["asides"] == {"no_word": 1}
    with pytest.raises(fb) as e2:
        m.trimul_in(mod, "z")                                                            # triangle_multiplicative="torch": the torch path is the stock forward's own
    assert str(e2.value) == "torch_path_class"
    sys.modules.pop("odde_trimul_bind", None)


def test_refresh_marks_the_binding_applied_from_its_routes(monkeypatch):
    import types
    from opendde_opt import stack
    fake_hook = types.SimpleNamespace(STATE={"installs": [{"levers": {"odde_arm_t": {"arm": "U"}}}], "errors": {}})
    arm = types.SimpleNamespace(COUNTS={"arm": "U", "installed": True, "att_mode": "k2b_bf16", "u2_trimul": True, "u3": True, "trimul_bind": "fast"})
    bind = types.SimpleNamespace(COUNTS={"active": True, "word": "fast"}, WORD="fast")
    monkeypatch.setitem(sys.modules, "odde_served_levers", fake_hook)
    monkeypatch.setitem(sys.modules, "odde_arm_t", arm)
    monkeypatch.setitem(sys.modules, "odde_trimul_bind", bind)
    saved = stack._REPORT
    try:
        stack._REPORT = {"active": True, "levers_planned": ["served_levers_hook", "arm_u", "arm_u23", "trimul_core"], "levers": ["served_levers_hook", "arm_u", "arm_u23", "trimul_core"]}
        rep = stack.refresh()
        assert "trimul_core" in rep["levers_applied"] and not rep["partial"], rep
        arm.COUNTS["trimul_bind"] = None                                             # the arm's route did not bind the unit: a fallback BY NAME
        rep = stack.refresh()
        assert "trimul_core" not in rep["levers_applied"] and rep["partial"] and any(f.startswith("trimul_core:arm route not bound") for f in rep["levers_fallback"]), rep
        # the exact line: fpf_engines bound the trimul ops, FPF_IMPL names the unit
        fe = types.SimpleNamespace(_ORIG={("opendde", "trimul_out"): 1, ("opendde", "trimul_in"): 2})
        monkeypatch.setitem(sys.modules, "fpf_engines", fe)
        monkeypatch.setenv("FPF_IMPL", "trimul_out=odde_trimul_bind:trimul_out,trimul_in=odde_trimul_bind:trimul_in")
        bind.COUNTS["word"] = bind.WORD = "exact"
        stack._REPORT = {"active": True, "levers_planned": ["fpf_trimul_exact", "trimul_exact"], "levers": ["fpf_trimul_exact", "trimul_exact"], "levers_applied": []}
        rep = stack.refresh()
        assert {"fpf_trimul_exact", "trimul_exact"} <= set(rep["levers_applied"]) and not rep["partial"], rep
        monkeypatch.setenv("FPF_IMPL", "trimul_out=elsewhere:trimul_out,trimul_in=elsewhere:trimul_in")
        rep = stack.refresh()
        assert "trimul_exact" not in rep["levers_applied"] and any(f.startswith("trimul_exact:FPF_IMPL") for f in rep["levers_fallback"]), rep
    finally:
        stack._REPORT = saved


def test_evidence_tokens():
    from opendde_opt import report
    stats = {"trimul_prov": {"word": "fast", "calls": 960, "rows": {"tmk3_exact": 960}, "precisions": {"bf16": 960}, "channels": {"C384": 940, "C64": 20},
                             "sites": {"arm_u2": 960}, "stock_calls": 2, "asides": {"cell_stock": 2}, "refusals": {}, "buckets": {"le400:tmk3_exact": 960}, "excluded": [], "errors": {},
                             "resolved_from": "core", "cells": {"trimul row=tmk3_exact word=fast cell=9.0|bf16|C384|H384|N<=400|out|fwd": 480}}}
    ev = report.lever_evidence("trimul_core", stats)
    assert ev["word"] == "fast" and ev["calls"] == 960 and ev["rows"] == "tmk3_exact:960" and ev["asides"] == "cell_stock:2" and ev["channels"] == "C384:940,C64:20"
    assert ev["buckets"] == "le400:tmk3_exact:960" and "kit_row" not in ev                                # served rows per row-count bucket; no kit-row substitution field (0.2.57)
    assert report.CORE_LEVERS["trimul_core"] == "opt_core.kernels.trimul" and report.STRATEGY["trimul_exact"].startswith("F2.")


def test_the_providers_per_module_cast_memo_is_released_after_each_call():
    """0.2.63: the core face memoises (fp32 z object, its autocast-dtype copy) per module when a cast row serves (compute_input's `_z_cast` in the
    module's provider cache); this engine updates z in place on one tensor object and builds a fresh z per confidence sample, so the binding
    releases that memo after EVERY call (release_cast_memo, in serve()'s finally): through the face's public helper when the core carries one,
    else the module's cache dict -- a no-op when nothing is memoised (native / exact rows, a core without the memo). LEVER token memo_pops."""
    import types
    from opendde_opt import report
    m = _fresh("fast")
    m.reset_counts(); m.COUNTS["memo_api"] = None
    face = types.SimpleNamespace(_CACHE_ATTR="_opt_core_trimul")                              # the face as the core ships it today: a per-module dict, no helper
    mod = types.SimpleNamespace()
    z0, zc = object(), object()
    setattr(mod, "_opt_core_trimul", {("by_word", "odde.fast"): {"_z_cast": (z0, "bf16", zc), ("pack", "bf16"): "weights"}, "_z_cast": (z0, "bf16", zc)})
    assert m.release_cast_memo(face, mod) == 2
    cache = getattr(mod, "_opt_core_trimul")
    assert "_z_cast" not in cache and "_z_cast" not in cache[("by_word", "odde.fast")] and cache[("by_word", "odde.fast")][("pack", "bf16")] == "weights"   # packs stay
    assert m.COUNTS["memo_api"] == "module._opt_core_trimul[*]._z_cast"
    assert m.release_cast_memo(face, mod) == 0                                                  # nothing memoised: 0, no error
    assert m.release_cast_memo(face, types.SimpleNamespace()) == 0                              # a module the face never touched
    assert m.release_cast_memo(types.SimpleNamespace(), mod) == 0                               # a core without the cache attribute name: no-op
    calls = []
    helper_face = types.SimpleNamespace(release_cast_memo=lambda module: calls.append(module) or True, _CACHE_ATTR="_opt_core_trimul")
    cache[("by_word", "odde.fast")]["_z_cast"] = (z0, "bf16", zc)
    assert m.release_cast_memo(helper_face, mod) == 1 and calls == [mod] and m.COUNTS["memo_api"] == "opt_core.trimul.release_cast_memo"   # the public helper wins when present
    m._bump("memo_pops", n=3)
    assert m.describe()["memo_pops"] == 3
    ev = report.lever_evidence("trimul_core", {"trimul_prov": m.describe()})
    assert ev["memo_pops"] == 3
    ev0 = report.lever_evidence("trimul_core", {"trimul_prov": {**m.describe(), "memo_pops": 0}})
    assert ev0["memo_pops"] == 0                                                                # printed as 0 where no cast row serves (H100 native rows)
    m.reset_counts()
    assert m.describe()["memo_pops"] == 0
    sys.modules.pop("odde_trimul_bind", None)


# ------------------------------------------------------------------------------------------------------------------ GPU: the served row through the provider face
cuda = pytest.mark.skipif(not os.environ.get("ODDE_TEST_CUDA") and not (importlib.util.find_spec("torch") and __import__("torch").cuda.is_available()),
                          reason="needs CUDA (+ cuequivariance_ops_torch, triton)")


def _module(C, outgoing, device, dtype):
    import torch
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c_z = self.c_hidden = C; self._outgoing = outgoing
            self.layer_norm_in = torch.nn.LayerNorm(C); self.layer_norm_out = torch.nn.LayerNorm(C)
            for n in ("linear_a_p", "linear_b_p", "linear_a_g", "linear_b_g", "linear_g"):
                setattr(self, n, torch.nn.Linear(C, C, bias=False))
            self.linear_z = torch.nn.Linear(C, C, bias=False)
    torch.manual_seed(0)
    m = M().to(device=device, dtype=dtype).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0.0, 0.05)
        m.layer_norm_in.weight.fill_(1.0).add_(torch.randn_like(m.layer_norm_in.weight) * 0.02); m.layer_norm_out.weight.fill_(1.0)
    return m


@cuda
@pytest.mark.parametrize("C,N", [(384, 384), (384, 800), (64, 400)])
def test_the_served_row_is_a_kernel_row_and_its_bytes_are_the_providers(C, N):
    """bf16-resident z (ARM U's planes): the row the unit serves under the fast word is a kernel row of the provider and its bytes are that row's
    through the provider face (pinned by the name the unit's census reports), torch.equal, both directions -- class contract, no winner named here."""
    import torch
    m_exact = _fresh("fast")
    from opt_core.kernels import trimul as KT
    dev = torch.device("cuda")
    for outgoing in (True, False):
        mod = _module(C, outgoing, dev, torch.float32)
        torch.manual_seed(1)
        z = (torch.randn(N, N, C, device=dev) * 0.5).to(torch.bfloat16)
        mask = (torch.rand(N, N, device=dev) > 0.05).to(torch.bfloat16)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out_p = m_exact.serve(mod, z, mask, residual=False, site="test")
            served = sorted(m_exact.COUNTS["rows"])[-1] if len(m_exact.COUNTS["rows"]) == 1 else None
            assert served is not None and served in KT.ROW_NAMES and served not in KT.STOCK_ROWS, m_exact.COUNTS["rows"]   # a kernel row (fast word: tolerance class or better)
            ref = KT.triangle_multiplication(z, mask, direction="outgoing" if outgoing else "incoming", weights=m_exact.weights_of(mod), word=served, residual=False)
        assert out_p.shape == ref.shape and torch.equal(out_p, ref), (C, N, outgoing, (out_p.float() - ref.float()).abs().max().item())
    assert m_exact.COUNTS["calls"] >= 2 and not (set(m_exact.COUNTS["rows"]) & set(KT.STOCK_ROWS))
    sys.modules.pop("odde_trimul_bind", None)


def test_no_row_floor_only_upstreams_small_n_regime(monkeypatch):
    """No kit row floor: a tier word's call at ANY row count above upstream's small-N regime goes to the provider's select() (no row count above that regime
    diverts a call). At and below 100 rows the stock module runs cuEquivariance's torch algorithm: the
    stock forward by name (Aside n_le_100_stock_small_n_path); a tensor that is not CUDA / not rank 3-4 likewise (not_cuda_or_rank)."""
    import types
    for word in ("exact", "big", "fast"):
        m = _fresh(word)
        z_small = types.SimpleNamespace(is_cuda=True, shape=(64, 64, 384), dim=lambda: 3)
        with pytest.raises(m.Aside) as e2:
            m.serve(types.SimpleNamespace(_outgoing=True), z_small, None, residual=False)
        assert e2.value.reason == "n_le_100_stock_small_n_path", word
        z_cpu = types.SimpleNamespace(is_cuda=False, shape=(400, 400, 384), dim=lambda: 3)
        with pytest.raises(m.Aside) as e3:
            m.serve(types.SimpleNamespace(_outgoing=True), z_cpu, None, residual=False)
        assert e3.value.reason == "not_cuda_or_rank", word
        assert [m.bucket_word(n) for n in (101, 256, 300, 400, 512, 799, 800, 1200, 1400, 3000)] == ["le256", "le256", "le400", "le400", "le512", "le800", "le800", "le1200", "le1556", "gt2048"]
    sys.modules.pop("odde_trimul_bind", None)
