"""The per-call dtype gate (dtype_gate.py; protenix_opt 0.3.46): stock's `--dtype fp32|fp16` runs every mode — the kit's bf16-only fused
kernels (opm_fused / pwa_fused / pf_attn) step aside PER CALL, BY NAME, to the module's own statement outside a bf16 autocast region, counted
(`aside=dtype_fp32:<n>` on the LEVER line); atom_attn_exact's per-call `Refused` is caught at the kit's binding and answered with the stock
statement, counted under the refusal's word; the verdict (stack.served_census / reconcile / the CLI's exit rule) treats a DECLARED aside as by
design (lever applied, partial false, rc 0) and an undeclared served-0 as a fallback by name (exit 3). CPU tensors + stub kernels throughout
(the carried packages import triton / protenix)."""
import functools
import sys
import types

import pytest

from protenix_opt import apb_levers as apb, cli, dtype_gate as G, report, runner_seam, sampler_levers as sl, stack
from protenix_opt.tests.test_cli_passthrough import CoreStub, fake_stock            # noqa: F401  (fixture: the in-process stock stand-in)

torch = pytest.importorskip("torch")

REFUSED_FULL = "atom_attn_exact: cuBLAS numerics for full chunks (1280) not tf32-sequential (qk=None, pv=None)"
REFUSED_REM = "atom_attn_exact: cuBLAS numerics for remainder chunk batch 260 not reproducible (qk=None, pv=tf32)"


@pytest.fixture
def clean(monkeypatch):
    """Fresh lever / seam state (the gate tallies included); the process's own modules back afterwards."""
    monkeypatch.setattr(apb, "_STATE", {"on": [], "patch": None, "installed": {}, "errors": {}, "named": {}, "precision": None, "models": 0, "sm80": {}, "gate": {}})
    monkeypatch.setattr(sl, "_STATE", {"on": [], "patch": None, "installed": {}, "errors": {}, "reports": {}, "lowp": None, "models": 0, "gate": {}})
    monkeypatch.setattr(runner_seam, "_INSTALLERS", []); monkeypatch.setattr(runner_seam, "_STATE", {"patch": None, "runners": 0, "ran": []})
    yield


# ------------------------------------------------------------------------------------------------ the words
def test_autocast_word_follows_the_autocast_state_not_the_tensor_dtype():
    """No autocast (stock `--dtype fp32`) -> dtype_fp32; an fp16 region (`--dtype fp16`) -> dtype_fp16; a bf16 region -> None (the kernels' cell).
    Read for the device of the call (CPU here; `cuda` on the box)."""
    assert G.autocast_word(torch, "cpu") == G.WORD_FP32 == "dtype_fp32"
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert G.autocast_word(torch, "cpu") is None
        assert G.autocast_word(torch, "cuda") == "dtype_fp32", "another device's region does not engage the kernels"
    with torch.autocast("cpu", dtype=torch.float16):
        assert G.autocast_word(torch, "cpu") == G.WORD_FP16 == "dtype_fp16"
    x = torch.zeros(3, dtype=torch.float32)
    assert G.device_type((x,), {}) == "cpu" and G.device_type((), {"z": x}) == "cpu" and G.device_type((1, "a"), {}) == "cuda"


def test_refused_word_names_the_chunk_form_without_digits():
    assert G.refused_word(RuntimeError(REFUSED_FULL)) == "cublas_route_full_chunk"
    assert G.refused_word(RuntimeError(REFUSED_REM)) == "cublas_route_remainder"
    assert G.refused_word(RuntimeError("atom_attn_exact: something else")) == G.REFUSED == "refused"


def test_census_reads_the_tally():
    """None = no call reached the site; True = served, or every call a declared aside; False = served 0 with unaccounted calls."""
    assert G.census(None) == (None, "no call reached the site: calls=0 served=0 aside=none")
    assert G.census({"calls": 8, "served": 8, "aside": {}}) == (True, "calls=8 served=8 aside=none")
    assert G.census({"calls": 8, "served": 0, "aside": {"dtype_fp32": 8}}) == (True, "every call stepped aside by name: calls=8 served=0 aside=dtype_fp32:8")
    assert G.census({"calls": 8, "served": 2, "aside": {"cublas_route_full_chunk": 6}})[0] is True
    ok, detail = G.census({"calls": 8, "served": 0, "aside": {"dtype_fp32": 3}})
    assert ok is False and detail.startswith("served 0 of 8 calls, 5 not accounted for by a declared aside")
    assert G.aside_token({"aside": {"dtype_fp32": 3, "dtype_fp16": 1}}) == "dtype_fp16:1,dtype_fp32:3" and G.aside_token({"aside": {}}) is None


# ------------------------------------------------------------------------------------------------ the instance-level gate (opm_fused / pwa_fused / pf_attn)
class _OPM:
    """A stand-in OuterProductMean: the class's forward is the module's own statement (what stock / the mode runs without the fused kernel)."""
    calls = []

    def forward(self, m, chunk_size=None):
        _OPM.calls.append(("stock", m.dtype, chunk_size)); return ("stock", m)


class _Model:
    def __init__(self, *mods):
        self._mods = list(mods)

    def modules(self):
        return iter([self] + self._mods)


def _kernel_forward(self, cell, m, chunk_size=None):
    """The package's instance forward (install: functools.partial(_opm_forward, module, cell) with the ownership marks)."""
    assert m.dtype == torch.bfloat16 or torch.is_autocast_enabled("cpu"), "the kernel serves the bf16-autocast statement only"
    _OPM.calls.append(("kernel", m.dtype, chunk_size)); return ("kernel", m)


def _install_like_the_package(mod, lever="opm_fused"):
    fwd = functools.partial(_kernel_forward, mod, {"ln_rows": 64})
    fwd.__wrapped__ = type(mod).forward; fwd._fpf_msa = lever
    mod.forward = fwd
    return fwd


def test_gate_method_routes_fp32_calls_to_the_class_statement_and_bf16_calls_to_the_kernel(clean):
    """CPU fp32 tensors: outside an autocast region the class's forward answers (counted dtype_fp32); inside a bf16 region the package's
    callable does (counted served); an fp16 region is dtype_fp16. The package's ownership marks ride the gate; gating twice changes nothing."""
    _OPM.calls = []
    mod = _OPM(); kernel = _install_like_the_package(mod)
    assert apb.gate_installed("opm_fused", _Model(mod)) == 1 and apb.gate_installed("opm_fused", _Model(mod)) == 0, "idempotent"
    assert apb.gate_installed("dit_attn", _Model(mod)) == 0, "a lever outside DTYPE_GATED gates nothing"
    g = mod.__dict__["forward"]
    assert g._ptx_dtype_gate == "opm_fused" and g._fpf_msa == "opm_fused" and g.__wrapped__ is _OPM.forward and g._ptx_kernel_fn is kernel
    m32 = torch.ones(4, 5, 8, dtype=torch.float32)
    assert mod.forward(m32, chunk_size=4)[0] == "stock"                          # `--dtype fp32`: no autocast region
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert mod.forward(m32, chunk_size=4)[0] == "kernel"                     # the kernels' cell (an fp32 tensor under bf16 autocast is served: stock's Linears cast it)
        assert mod.forward(m32.to(torch.bfloat16))[0] == "kernel"
    with torch.autocast("cpu", dtype=torch.float16):
        assert mod.forward(m32)[0] == "stock"                                    # `--dtype fp16`
    t = apb.gate_tally("opm_fused")
    assert t == {"calls": 4, "served": 2, "aside": {"dtype_fp32": 1, "dtype_fp16": 1}}
    assert [c[0] for c in _OPM.calls] == ["stock", "kernel", "kernel", "stock"] and _OPM.calls[0] == ("stock", torch.float32, 4)
    assert G.census(t) == (True, "calls=4 served=2 aside=dtype_fp16:1,dtype_fp32:1")


def test_the_aside_target_is_the_class_method_at_the_call(clean):
    """The statement a stepped-aside call runs is the class's method resolved AT THE CALL: a class-level lever of the mode installed later
    (pwa_zcache's class forward) keeps its place; the gate never pins the function it saw at install."""
    mod = _OPM(); _install_like_the_package(mod, "pwa_fused"); apb.gate_installed("pwa_fused", _Model(mod))
    seen = []
    orig = _OPM.forward
    try:
        _OPM.forward = lambda self, m, chunk_size=None: (seen.append("class_lever") or ("class_lever", m))
        assert mod.forward(torch.ones(2, dtype=torch.float32))[0] == "class_lever"
    finally:
        _OPM.forward = orig
    assert seen == ["class_lever"] and apb.gate_tally("pwa_fused")["aside"] == {"dtype_fp32": 1}


def _stub_msa_package(monkeypatch, tmp_path):
    """A stub protenix_fpf_msa whose installs bind instance forwards the way the carried package does (partial + marks) and report calls."""
    pkg = types.ModuleType(apb.MSA_IMPORT_NAME); pkg.__version__ = "0.1.0"
    pkg.STATE = {"opm_fused": {"installed_on": 0, "calls": 0, "chunks": 0, "cell": None, "cell_key": None, "named": None, "left_alone": [], "error": None},
                 "pwa_fused": {"installed_on": 0, "calls": 0, "zcache_hits": 0, "cell": None, "cell_key": None, "named": None, "left_alone": [], "error": None}}

    def _fwd(self, cell, m, *a, _lever=None, **k):
        pkg.STATE[_lever]["calls"] += 1; return ("kernel", _lever)

    def _install(model, _lever=None):
        n = 0
        for mod in model.modules():
            if isinstance(mod, _OPM) and getattr(mod, "kind", None) == _lever:
                fwd = functools.partial(_fwd, mod, {"cell": 1}, _lever=_lever); fwd.__wrapped__ = type(mod).forward; fwd._fpf_msa = _lever
                mod.forward = fwd; n += 1
        pkg.STATE[_lever].update(installed_on=n, cell_key="9.0", cell={"ln_rows": 64})
        return dict(pkg.STATE[_lever])

    pkg.install_opm_fused = functools.partial(_install, _lever="opm_fused"); pkg.install_pwa_fused = functools.partial(_install, _lever="pwa_fused")
    pkg.report = lambda: {k: dict(v) for k, v in pkg.STATE.items()}
    monkeypatch.setitem(sys.modules, apb.MSA_IMPORT_NAME, pkg)
    monkeypatch.setattr(apb, "triton_note", lambda: None)
    return pkg


def test_the_seam_installer_gates_what_the_msa_package_bound(clean, monkeypatch, tmp_path, capsys):
    """apb_levers' runner-seam installer: the package's install runs, its marker line prints as before, and every forward it bound is re-wrapped
    with the gate; under no autocast the package's kernel serves nothing (its `calls` stays 0) and the tally names every call dtype_fp32; the
    LEVER evidence then carries `aside=dtype_fp32:<n>` — and carries no `aside` pair on a run where nothing stepped aside."""
    pkg = _stub_msa_package(monkeypatch, tmp_path)
    opms = [_OPM() for _ in range(4)]
    for m in opms: m.kind = "opm_fused"
    pwas = [_OPM() for _ in range(3)]
    for m in pwas: m.kind = "pwa_fused"
    runner = types.SimpleNamespace(model=_Model(*opms, *pwas))
    apb._STATE["on"] = ["opm_fused", "pwa_fused"]
    apb._installer("opm_fused")(runner); apb._installer("pwa_fused")(runner)
    err = capsys.readouterr().err
    assert "[protenix-opt] OPM_FUSED:on(4 modules; cell=9.0 ln_rows=64; fpf_msa 0.1.0)" in err and "PWA_FUSED:on(3 modules;" in err
    assert all(m.__dict__["forward"]._ptx_dtype_gate == "opm_fused" for m in opms) and all(m.__dict__["forward"]._fpf_msa == "pwa_fused" for m in pwas)
    x = torch.ones(2, 3, dtype=torch.float32)
    for _cycle in range(10):                                                       # a `--dtype fp32` run: 10 recycles x the MSA module's 4 OPM / 3 PWA calls
        for m in opms + pwas:
            assert m.forward(x)[0] == "stock"
    assert pkg.STATE["opm_fused"]["calls"] == 0 and pkg.STATE["pwa_fused"]["calls"] == 0, "the kernels served nothing"
    assert apb.gate_tally("opm_fused") == {"calls": 40, "served": 0, "aside": {"dtype_fp32": 40}}
    assert apb.gate_tally("pwa_fused") == {"calls": 30, "served": 0, "aside": {"dtype_fp32": 30}}
    monkeypatch.setattr(runner_seam, "patched", lambda: True)
    ev = dict(apb.evidence("opm_fused"))
    assert ev["aside"] == "dtype_fp32:40" and ev["calls"] == 0 and ev["modules"] == 4
    assert apb.gate_census() == {"opm_fused": (True, "every call stepped aside by name: calls=40 served=0 aside=dtype_fp32:40"),
                                 "pwa_fused": (True, "every call stepped aside by name: calls=30 served=0 aside=dtype_fp32:30")}
    for t in apb._STATE["gate"].values():                                          # the bf16 run (same process, tallies zeroed in place): served, the package counts, no aside pair on the line
        t.update(calls=0, served=0); t["aside"].clear()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        for m in opms: assert m.forward(x)[0] == "kernel"
    assert pkg.STATE["opm_fused"]["calls"] == 4 and apb.gate_tally("opm_fused") == {"calls": 4, "served": 4, "aside": {}}
    assert "aside" not in dict(apb.evidence("opm_fused")), "a run where nothing stepped aside prints the LEVER line as before"


# ------------------------------------------------------------------------------------------------ atom_attn_exact: the Refused catch at the binding
def _stub_atomx(monkeypatch, refuse_when=lambda q: q.shape[0] == 1):
    """protenix primitives (the site) + a stub protenix_fpf_atom_attn_exact whose install binds a fused callable that raises its kernel.Refused
    BY NAME for `refuse_when(q)` (the un-probed chunk batch) and otherwise serves, counting like the carried package (kernel += 1 BEFORE the
    route check)."""
    for name in ("protenix", "protenix.model", "protenix.model.modules"):
        m = types.ModuleType(name); m.__path__ = []; monkeypatch.setitem(sys.modules, name, m)
    PR = types.ModuleType("protenix.model.modules.primitives")
    stock_calls = []

    def _local_attention(q, k, v, n_queries, n_keys, attn_bias=None, trunked_attn_bias=None, inf=1e10, use_efficient_implementation=True, inplace_safe=False, chunk_size=None):
        stock_calls.append((tuple(q.shape), chunk_size)); return ("stock", q)
    PR._local_attention = _local_attention; PR.stock_calls = stock_calls
    monkeypatch.setitem(sys.modules, "protenix.model.modules.primitives", PR)
    K = types.ModuleType("protenix_opt.apb_atom_exact.kernel")

    class Refused(RuntimeError):
        pass
    K.Refused = Refused; K.STATE = {"calls": {"kernel": 0, "original": 0}, "why_original": {}}
    pkg = types.ModuleType("protenix_opt.apb_atom_exact"); pkg.__version__ = "1.0.0"; pkg.__path__ = []; pkg.kernel = K

    def install(model=None):
        if getattr(PR._local_attention, "_atom_attn_exact", False):               # the carried install's idempotence: a site already bound is left as it is
            return {"installed": True}
        original = PR._local_attention

        def fused(q, k, v, n_queries, n_keys, **kw):
            if q.dtype != torch.float32:
                K.STATE["calls"]["original"] += 1; K.STATE["why_original"]["dtype"] = K.STATE["why_original"].get("dtype", 0) + 1
                return original(q, k, v, n_queries, n_keys, **kw)
            K.STATE["calls"]["kernel"] += 1
            if refuse_when(q):
                raise Refused(REFUSED_FULL)
            return ("kernel", q)
        fused._atom_attn_exact = True
        PR._local_attention = fused
        return {"installed": True, "routes": "qk:tf32[4..256] pv:tf32[4..256]", "loadcheck": ["a", "b", "c", "d"]}
    pkg.install = install
    pkg.report = lambda: {"installed": True, "routes": "qk:tf32[4..256]", "calls": dict(K.STATE["calls"]), "original_path_reasons": dict(K.STATE["why_original"])}
    monkeypatch.setitem(sys.modules, "protenix_opt.apb_atom_exact", pkg); monkeypatch.setitem(sys.modules, "protenix_opt.apb_atom_exact.kernel", K)
    return PR, K, pkg


def test_gate_refusable_answers_a_refused_call_with_the_original(clean):
    class Refused(RuntimeError):
        pass
    seen = []

    def fused(q, *a, **k):
        if q == "unprobed":
            raise Refused(REFUSED_REM)
        return "kernel"
    fused._atom_attn_exact = True
    tally = G.new_tally()
    g = G.gate_refusable(fused, lambda q, *a, **k: "stock", (Refused,), tally, on_aside=seen.append)
    assert g("probed") == "kernel" and g("unprobed") == "stock" and g._atom_attn_exact is True and g._ptx_dtype_gate == "atom_attn_exact"
    assert tally == {"calls": 2, "served": 1, "aside": {"cublas_route_remainder": 1}} and seen == ["cublas_route_remainder"]
    with pytest.raises(ValueError):                                                # anything but the package's Refused propagates as before (no blanket catch)
        G.gate_refusable(lambda *a, **k: (_ for _ in ()).throw(ValueError("x")), lambda *a, **k: None, (Refused,), G.new_tally())("q")


def test_the_binding_catches_atom_attn_exact_refused_per_call_and_keeps_the_package_census_truthful(clean, monkeypatch, capsys):
    """sampler_levers' installer: the package binds its kernel on primitives._local_attention, the kit re-wraps it; a call the kernel refuses BY
    NAME (the input embedder's un-chunked fp32 call under `--dtype fp32`) runs the stock statement the install displaced, counted
    `cublas_route_full_chunk`; the diffusion calls stay on the kernel; the package's kernel count is what the kernel served; the LEVER evidence
    carries `aside=`; a bf16 call keeps the package's own counted original path (no aside)."""
    PR, K, pkg = _stub_atomx(monkeypatch)
    stock = PR._local_attention
    sl._STATE["on"] = ["atom_attn_exact"]
    sl._installer("atom_attn_exact")(types.SimpleNamespace(model=object()))
    err = capsys.readouterr().err
    assert "[protenix-opt] ATOMATTNEXACT:installed(1 sites; kernels.apb.atom_exact 1.0.0)" in err
    site = PR._local_attention
    assert site._ptx_dtype_gate == "atom_attn_exact" and site._atom_attn_exact is True and site.__wrapped__ is stock
    assert sl._STATE["reports"]["atom_attn_exact"]["gated"] is True
    q1 = torch.zeros(1, 4, 96, 32); q5 = torch.zeros(5, 4, 96, 32); qb = torch.zeros(5, 4, 96, 32, dtype=torch.bfloat16)
    assert site(q1, q1, q1, 32, 128, trunked_attn_bias=q1, chunk_size=None)[0] == "stock"      # refused by name -> the stock statement, this call only
    for _step in range(6):
        assert site(q5, q5, q5, 32, 128, trunked_attn_bias=q5, chunk_size=256)[0] == "kernel"
    assert site(qb, qb, qb, 32, 128, trunked_attn_bias=qb)[0] == "stock"                     # the package's own envelope miss (dtype): its counted original path, not an aside
    assert PR.stock_calls[0] == ((1, 4, 96, 32), None) and len(PR.stock_calls) == 2
    assert K.STATE["calls"] == {"kernel": 6, "original": 1} and K.STATE["why_original"] == {"dtype": 1}, "kernel = calls the kernel served (the refused attempt taken back)"
    assert sl.gate_tally("atom_attn_exact") == {"calls": 8, "served": 7, "aside": {"cublas_route_full_chunk": 1}}
    monkeypatch.setattr(runner_seam, "patched", lambda: True)
    ev = dict(sl.evidence("atom_attn_exact"))
    assert ev["aside"] == "cublas_route_full_chunk:1" and ev["kernel_calls"] == 6 and ev["original_calls"] == 1 and ev["original_reasons"] == "dtype:1"
    assert sl.gate_census() == {"atom_attn_exact": (True, "calls=8 served=7 aside=cublas_route_full_chunk:1")}
    # a second runner in the process: the site is already gated — nothing stacks
    sl._installer("atom_attn_exact")(types.SimpleNamespace(model=object()))
    assert PR._local_attention is site


# ------------------------------------------------------------------------------------------------ the verdict
def _fp32_run_state(monkeypatch, opm_aside, opm_calls=None, atom_aside=None):
    """The end-of-run state of a `--dtype fp32` run as the levers record it: a runner built, the MSA kernels installed on their modules with every
    call stepped aside (the tallies given), atom_attn_exact's un-chunked call refused by name; no package loaded (report() absent)."""
    monkeypatch.setattr(runner_seam, "patched", lambda: True)
    monkeypatch.setattr(runner_seam, "_STATE", {"patch": None, "runners": 1, "ran": ["opm_fused", "pwa_fused", "pf_attn", "atom_attn_exact"]})
    calls = opm_aside if opm_calls is None else opm_calls
    monkeypatch.setattr(apb, "_STATE", {"on": ["pf_attn", "opm_fused", "pwa_fused"], "patch": None, "installed": {"pf_attn": 52, "opm_fused": 4, "pwa_fused": 3}, "errors": {},
                                        "named": {}, "precision": None, "models": 1, "sm80": {},
                                        "gate": {"opm_fused": {"calls": calls, "served": 0, "aside": ({"dtype_fp32": opm_aside} if opm_aside else {})},
                                                 "pwa_fused": {"calls": 30, "served": 0, "aside": {"dtype_fp32": 30}},
                                                 "pf_attn": {"calls": 520, "served": 0, "aside": {"dtype_fp32": 520}}}})
    monkeypatch.setattr(sl, "_STATE", {"on": ["atom_attn_exact"], "patch": None, "installed": {"atom_attn_exact": 1}, "errors": {},
                                       "reports": {"atom_attn_exact": {"installed": True, "gated": True}}, "lowp": None, "models": 1,
                                       "gate": {"atom_attn_exact": {"calls": 61, "served": 60, "aside": dict(atom_aside or {"cublas_route_full_chunk": 1})}}})


def test_served_census_and_reconcile_treat_a_declared_aside_as_by_design_and_an_undeclared_served_0_as_a_fallback(clean, monkeypatch):
    _fp32_run_state(monkeypatch, opm_aside=40)
    served = stack.served_census()
    assert served["opm_fused"] == (True, "every call stepped aside by name: calls=40 served=0 aside=dtype_fp32:40")
    assert served["pf_attn"][0] is True and served["pwa_fused"][0] is True and served["atom_attn_exact"] == (True, "calls=61 served=60 aside=cublas_route_full_chunk:1")
    rep = {"mode": "fast", "active": True, "levers_applied": ["pf_attn", "opm_fused", "pwa_fused", "atom_attn_exact", "t1_fused_transition"], "levers_fallback": [], "fallback_reasons": {}}
    r = stack.reconcile(rep, {})
    assert r["partial"] is False and r["levers_fallback"] == [] and set(r["levers_applied"]) == set(rep["levers_applied"])
    assert r["asides"] == {"opm_fused": {"dtype_fp32": 40}, "pwa_fused": {"dtype_fp32": 30}, "pf_attn": {"dtype_fp32": 520}, "atom_attn_exact": {"cublas_route_full_chunk": 1}}
    assert r["reconciled"]["moves"] == {}
    # the same run with opm_fused's calls at the site NOT accounted for (served 0, no declared aside): a fallback by name
    _fp32_run_state(monkeypatch, opm_aside=0, opm_calls=40)
    assert stack.served_census()["opm_fused"] == (False, "served 0 of 40 calls, 40 not accounted for by a declared aside: calls=40 served=0 aside=none")
    r = stack.reconcile(rep, {})
    assert r["partial"] is True and r["levers_fallback"] == ["opm_fused"] and r["reconciled"]["moves"] == {"opm_fused": "applied -> fallback (census)"}
    assert r["fallback_reasons"]["opm_fused"].startswith("on but served 0 calls: served 0 of 40 calls")
    # no call reached the site (an input without MSA): nothing to judge, applied as before
    _fp32_run_state(monkeypatch, opm_aside=0, opm_calls=0)
    assert stack.served_census()["opm_fused"][0] is None and stack.reconcile(rep, {})["partial"] is False


@pytest.mark.parametrize("mode,applied", [
    ("fast", ["pf_attn", "opm_fused", "pwa_fused", "t1_fused_transition"]),
    ("exact", ["atom_attn_exact", "t1_fused_transition"]),
    ("big", ["opm_fused", "pwa_fused", "drop_bond_mask", "msa_zfree", "t1_fused_transition"]),
])
def test_verdict_replay_of_an_fp32_run_exits_0_with_partial_false(mode, applied, clean, monkeypatch, fake_stock, tmp_path, capsys):
    """THE REPLAY: `pred --mode <m> … --dtype fp32` on the recorded end-of-run state of an fp32 run (every fused MSA / pair-bias call aside
    dtype_fp32, atom_attn_exact's un-chunked call aside cublas_route_full_chunk, big's line levers marked as on any completed trunk — the
    drop_bond_mask mark is the trunk-end seam's, reached once the MSA module no longer raises): FINAL partial=false, fallbacks=none, rc 0, the
    LEVER lines carry the aside counts, `--dtype fp32` reaches the stock parser verbatim."""
    from protenix_opt import big
    _fp32_run_state(monkeypatch, opm_aside=40)
    monkeypatch.setattr(big, "_record", lambda: None); monkeypatch.setattr(big, "_STATE", dict(big._STATE, install_error=None, unit=None))
    stub = CoreStub(applied=applied)
    monkeypatch.setattr(cli, "activate", lambda core, m, dry_run=False, det_level=None: stub.enable(m))
    monkeypatch.setattr(cli, "check_mode", lambda core, m: m)
    monkeypatch.setattr(report, "register_exit_tally", lambda: True)
    monkeypatch.delenv("PTX_LEVER_REPORT", raising=False)
    out = tmp_path / "o"
    rc = cli.main(["pred", "--mode", mode, "--input", "x", "--out_dir", str(out), "--dtype", "fp32"])
    err = capsys.readouterr().err
    assert rc == 0, err[-3000:]
    assert fake_stock["pred"][-1]["dtype"] == "fp32", "the stock knob passes through verbatim"
    assert f"[protenix-opt] FINAL mode={mode} n_gpu=1 sharding=none levers={','.join(applied)} fallbacks=none partial=false" in err
    assert "NOT ACTIVE" not in err and "[protenix-opt] OUTPUTS complete " in err
    if "opm_fused" in applied:
        assert " aside=dtype_fp32:40 " in next(l for l in err.splitlines() if l.startswith("[protenix-opt] LEVER ") and l.endswith(" lever=opm_fused"))
        assert " aside=dtype_fp32:30 " in next(l for l in err.splitlines() if l.startswith("[protenix-opt] LEVER ") and l.endswith(" lever=pwa_fused"))
    if "atom_attn_exact" in applied:
        assert " aside=cublas_route_full_chunk:1 " in next(l for l in err.splitlines() if l.startswith("[protenix-opt] LEVER ") and l.endswith(" lever=atom_attn_exact"))


def test_verdict_replay_of_an_undeclared_served_0_exits_3_by_name(clean, monkeypatch, fake_stock, tmp_path, capsys):
    """The exit rule holds: a fused lever whose calls reached the site with none served and none declared aside is a fallback by name after the
    run — FINAL partial=true, NOT ACTIVE names it with the census, exit 3 (outputs in place)."""
    _fp32_run_state(monkeypatch, opm_aside=0, opm_calls=40)
    stub = CoreStub(applied=["pf_attn", "opm_fused", "pwa_fused", "t1_fused_transition"])
    monkeypatch.setattr(cli, "activate", lambda core, m, dry_run=False, det_level=None: stub.enable(m))
    monkeypatch.setattr(cli, "check_mode", lambda core, m: m)
    monkeypatch.setattr(report, "register_exit_tally", lambda: True)
    monkeypatch.delenv("PTX_LEVER_REPORT", raising=False)
    rc = cli.main(["pred", "--mode", "fast", "--input", "x", "--out_dir", str(tmp_path / "o"), "--dtype", "fp32"])
    err = capsys.readouterr().err
    assert rc == cli.EXIT_NOT_ACTIVE == 3, err[-3000:]
    assert "[protenix-opt] FINAL mode=fast n_gpu=1 sharding=none levers=pf_attn,pwa_fused,t1_fused_transition fallbacks=opm_fused partial=true" in err
    assert "NOT ACTIVE: partial activation refused (after the run: the kit's end-of-run records) — opm_fused: on but served 0 calls: served 0 of 40 calls" in err
    assert "[protenix-opt] OUTPUTS complete " in err


def test_release_words():
    """The README states the rule in the release vocabulary; the sealed packages are untouched by it (the gate is kit glue)."""
    import os
    readme = " ".join(open(os.path.join(stack.tree_home(), "README.md"), encoding="utf-8").read().split())   # the words, however the README wraps its lines
    assert "`--dtype`" in readme and "step aside per call to the stock computation, counted by name" in readme
    for mod in (apb, sl):
        src = open(mod.__file__, encoding="utf-8").read()
        assert "dtype_gate" in src
    assert apb.DTYPE_GATED == ("pf_attn", "opm_fused", "pwa_fused") and sl.GATED == ("atom_attn_exact",)
    assert set(stack.GATED_SERVED) == set(apb.DTYPE_GATED) | set(sl.GATED) and set(stack.GATED_SERVED) <= set(stack.SERVED)
