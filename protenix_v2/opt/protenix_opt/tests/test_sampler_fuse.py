"""The diffusion transformer's fused elementwise kernels (sampler_fuse.py, lever sampler_fuse): the switch parses by name, every kit
mode's row lists it and exports its switch, a switch set under a row without the lever is refused by name, the install wraps InferenceRunner.__init__ through
the core's fail-closed import patch and runs the carried add-on's install on the built model, the record and the LEVER evidence follow,
and the add-on tree the lever imports is the carried one."""
import os
import sys
import types

import pytest

from protenix_opt import kits, modes, registry, report, sampler_fuse as sf, stack


STUBBED = ("runner", "runner.inference", "dit_fuse", "dit_fuse_patch")   # the modules a test replaces by stubs: removed before, dropped and restored after


@pytest.fixture
def clean(monkeypatch):
    """A clean lever state, no AttrPatch of an earlier test on the seam, and the process's own `runner` / add-on modules back afterwards
    (a stub left in sys.modules would serve every later import of the stock runner)."""
    from opt_core import autoload as A
    from protenix_opt import runner_seam
    monkeypatch.setattr(A, "_PATCHES", {})
    monkeypatch.setattr(runner_seam, "_INSTALLERS", []); monkeypatch.setattr(runner_seam, "_STATE", {"patch": None, "runners": 0, "ran": []})
    monkeypatch.setattr(sf, "_STATE", dict(sf._STATE, on=False, patch=None, patched=False, models=0, report=None, error=None))
    saved = {m: sys.modules.pop(m) for m in STUBBED if m in sys.modules}
    yield
    for m in STUBBED:
        sys.modules.pop(m, None)
    sys.modules.update(saved)
    if sf._STATE.get("patch") is not None:                                        # an armed hook of this test never fired: withdraw it
        sf._STATE["patch"]._unhook()


def test_switch_parses_by_name():
    assert sf.from_env({}) is False and sf.from_env({sf.ENV: ""}) is False and sf.from_env({sf.ENV: "0"}) is False
    assert sf.from_env({sf.ENV: "1"}) is True
    with pytest.raises(ValueError, match="PTX_SAMPLER_FUSE='2'"):
        sf.from_env({sf.ENV: "2"})


def test_rows_and_exports():
    assert all("sampler_fuse" in modes.MODES[m] for m in ("exact", "fast", "big")) and "sampler_fuse" not in modes.BIG_DROPPED and sf.ENV not in modes.BIG_POST_DROPPED
    assert set(modes.PACKAGE_POST) == {"exact", "fast", "big"} and all(modes.PACKAGE_POST[m][sf.ENV] == "1" for m in modes.PACKAGE_POST)   # + pred_release and the sampler admission words per mode (test_sampler_admit pins those)
    lv = registry.LEVERS["sampler_fuse"]
    assert lv.env_keys == (sf.ENV,) and lv.tier == registry.EXACT and lv.probe == "marker" and lv.extra
    assert report.STRATEGY_IDS["sampler_fuse"] == sf.STRATEGY == "LOCAL.dit_fused_kernels" and stack.MARKERS["sampler_fuse"] == (sf.MARK, (sf.MARK,))
    for mode in ("exact", "fast"):
        res = modes.resolve(mode, {"PATH": os.environ["PATH"]}, stack.kit_home(), compute_cap="9.0", triton="3.7.1")
        assert res.exports.get(sf.ENV) == "1" and res.extras.get(sf.ENV) == "1", mode
        assert modes.resolve(mode, {"PATH": os.environ["PATH"], sf.ENV: "0"}, stack.kit_home(), compute_cap="9.0", triton="3.7.1").exports.get(sf.ENV) is None, "a caller's own value wins"
    b = modes.resolve("big", {"PATH": os.environ["PATH"]}, stack.kit_home(), compute_cap="9.0", triton="3.7.1")   # big through its base
    assert b.exports.get(sf.ENV) == "1" and b.extras.get(sf.ENV) == "1"


def test_switch_under_a_row_without_the_lever_is_refused_by_name(monkeypatch):
    assert stack._sampler_fuse_on("fast", {sf.ENV: "1"}) is True and stack._sampler_fuse_on("exact", {}) is False and stack._sampler_fuse_on("big", {sf.ENV: "1", "PTX_GUARD_LIFT": "1"}) is True
    monkeypatch.setitem(modes.MODES, "fast", [n for n in modes.MODES["fast"] if n != "sampler_fuse"])
    with pytest.raises(stack.ActivationError, match="PTX_SAMPLER_FUSE set under mode fast: the sampler_fuse lever is not in this mode's row"):
        stack._sampler_fuse_on("fast", {sf.ENV: "1"})


def test_the_lever_imports_the_carried_add_on():
    d = sf.tools_dir()
    assert d == os.path.join(kits.kit_dir("DIT_FUSE"), "tools") and sorted(os.listdir(d)) == ["dit_fuse.py", "dit_fuse_patch.py"]
    assert kits.labels()["DIT_FUSE"] == "DIT_FUSE_ADDON_v0"
    assert not os.path.exists(os.path.join(kits.kit_dir("DIT_FUSE"), "tools", "dit_sched.py")), "the S/P stream-scheduling module is not part of the tree"


def test_install_wraps_the_runner_and_runs_the_add_on(clean, monkeypatch, tmp_path):
    """runner.inference imported after install(): InferenceRunner.__init__ runs whole, then the add-on's install(model, levers=all four) — here a
    stub add-on in a stub tools dir (the carried one imports torch / triton / protenix); the record and the evidence carry its report."""
    tools = tmp_path / "tools"; tools.mkdir()
    (tools / "dit_fuse_patch.py").write_text(
        "import dit_fuse as DF\nimport torch\nCALLS = []\n"
        "def _dm(model): return torch.nn.Module()\n"
        "def install(model, levers):\n    CALLS.append((model, levers)); DF.STATS['calls']['ada'] += 3\n"
        "    return {'installed': True, 'n_ada': 66, 'n_gate': 24, 'n_res_apb': 24, 'n_ctb': 30, 'n_blocks': 30, 'cfg': {'x': 1}}\n")
    (tools / "dit_fuse.py").write_text("STATS = {'calls': {'ada': 0, 'gate': 0, 'res': 0, 'swiglu': 0}, 'recheck': 0, 'recheck_fail': 0, 'fallback': 0}\n"
                                     "def ada_tail(a, x1, x2): return 'ada'\ndef res_gate(g, x, res=None, _kind='res'): return 'res'\ndef swiglu(x, y): return 'swiglu'\ndef attn_gate(o, g): return 'gate'\n")
    monkeypatch.setattr(sf, "tools_dir", lambda: str(tools))
    pkg = tmp_path / "runner"; pkg.mkdir(); (pkg / "__init__.py").write_text("")
    (pkg / "inference.py").write_text("class InferenceRunner:\n    def __init__(self, configs):\n        self.configs = configs; self.model = ('model', configs)\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    marker = sf.install()
    assert marker == "DIT_FUSE:ada,gate,res,swiglu(armed)" and sf.state()["patched"] is False
    import runner.inference as RI                                                        # the import lands the patch
    r = RI.InferenceRunner({"n": 1})
    assert r.configs == {"n": 1} and sys.modules["dit_fuse_patch"].CALLS == [(("model", {"n": 1}), "ada,gate,res,swiglu")]
    st = sf.state()
    assert st["patched"] is True and st["models"] == 1 and st["sites"] == {"ada": 66, "gate": 24, "res": 24, "swiglu": 30} and st["calls"]["ada"] == 3 and st["error"] is None
    assert "cfg" not in st["report"] and st["keyed"] is True and sys.modules["dit_fuse"].ada_tail._sampler_fuse_keyed and st["regime"] == "fp32"
    assert sf.evidence()[:5] == [("models", 1), ("ada", 66), ("gate", 24), ("res", 24), ("swiglu", 30)] and ("calls_ada", 3) in sf.evidence()


def test_an_install_error_propagates(clean, monkeypatch, tmp_path):
    tools = tmp_path / "tools"; tools.mkdir()
    (tools / "dit_fuse_patch.py").write_text("def install(model, levers):\n    raise RuntimeError('no libdevice')\n")
    monkeypatch.setattr(sf, "tools_dir", lambda: str(tools))
    pkg = tmp_path / "runner"; pkg.mkdir(); (pkg / "__init__.py").write_text("")
    (pkg / "inference.py").write_text("class InferenceRunner:\n    def __init__(self):\n        self.model = object()\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    sf.install()
    import runner.inference as RI
    with pytest.raises(RuntimeError, match="no libdevice"):
        RI.InferenceRunner()
    assert sf.state()["models"] == 0 and "no libdevice" in sf.state()["error"]


def test_a_missing_tree_is_named(monkeypatch, tmp_path):
    monkeypatch.setattr(sf, "tools_dir", lambda: str(tmp_path / "absent"))
    with pytest.raises(RuntimeError, match="dit_fuse_patch.py missing"):
        sf._import_patch_module()


def test_reconcile_names_a_runner_never_imported(monkeypatch):
    monkeypatch.setattr(sf, "_STATE", dict(sf._STATE, on=True, patch=types.SimpleNamespace(state="armed"), patched=False, models=0, report=None, error=None))
    rep = {"active": True, "mode": "fast", "levers_applied": ["sampler_fuse", "deadskip"], "levers_fallback": []}
    r = stack.reconcile(rep, {})
    assert "sampler_fuse" in r["levers_fallback"] and "sampler_fuse" not in r["levers_applied"]
    assert r["fallback_reasons"]["sampler_fuse"].startswith("runner.inference was never imported")


def test_the_multi_gpu_line_tallies_the_kernel_calls_per_rank():
    """tp.TALLY counts the lever per rank from the package record's sampler_fuse calls (executed: > 0 in every rank; a rank with zero calls:
    inert_under_tp; no record: no_record) — never worded env_only."""
    from protenix_opt import tp
    assert "sampler_fuse" in tp.TALLY and tp.TALLY["sampler_fuse"][0] == "protenix_opt"
    calls = {"ada": 9600, "gate": 4800, "res": 12000, "swiglu": 6000}
    rec = lambda c: {"protenix_opt": {"levers_applied": ["sampler_fuse"], "sampler_fuse": {"patched": True, "models": 1, "calls": dict(c)}}}
    ex = tp.execution({11: rec(calls), 22: rec(calls)}, ["sampler_fuse"])
    assert ex["executed"] == {"sampler_fuse": {"11": 32400, "22": 32400}} and ex["env_only"] == [] and ex["no_record"] == []
    ex = tp.execution({11: rec(calls), 22: rec({k: 0 for k in calls})}, ["sampler_fuse"])
    assert ex["inert_under_tp"] == {"sampler_fuse": {"11": 32400, "22": 0}} and ex["executed"] == {}
    assert tp.execution({11: {"protenix_opt": {}}, 22: {"deadskip": {"n": 3}}}, ["sampler_fuse"])["no_record"] == ["sampler_fuse"]


def test_a_callers_zero_is_a_recorded_opt_out_not_a_fallback():
    """PTX_SAMPLER_FUSE=0 under a mode whose row lists the lever: not installed (stack._sampler_fuse_on is False) and classified `off by flag`
    (report: skipped / off_by_flag) — never a fallback that refuses the activation, never silent."""
    from protenix_opt import stack, report
    from protenix_opt.modes import readme_row
    env = {"PTX_SAMPLER_FUSE": "0"}
    assert stack._sampler_fuse_on("fast", env) is False
    on, fb, skipped, why = stack._classify("fast", [], env, row=readme_row("other", "fast"), row_key="other")
    assert "sampler_fuse" in skipped and "sampler_fuse" not in fb and why["sampler_fuse"].startswith(report.OFF_BY_FLAG)


def test_the_written_process_record_carries_the_kernel_tally(monkeypatch):
    """report.write_record's protenix_opt record carries the reconciled sampler_fuse state (the calls tp.TALLY counts per rank)."""
    import json as _json
    from protenix_opt import report, stack
    rep = {"mode": "fast", "active": True, "reconciled": {"records": []}, "levers_applied": ["sampler_fuse"],
           "sampler_fuse": {"patched": True, "models": 1, "calls": {"ada": 9600, "gate": 0, "res": 2400, "swiglu": 1200}}}
    monkeypatch.setattr(stack, "status", lambda: dict(rep))
    rec = report._activation_record()
    assert rec["sampler_fuse"]["calls"] == {"ada": 9600, "gate": 0, "res": 2400, "swiglu": 1200}
    from protenix_opt import tp
    assert tp.execution({7: {"protenix_opt": rec}}, ["sampler_fuse"])["executed"] == {"sampler_fuse": {"7": 13200}}


def test_gate_calls_on_the_stock_chain_are_named_stock_chain_calls_never_fallback(monkeypatch):
    """The add-on's exit line counts gate calls that ran the stock chain under its own word; the kit's state / LEVER evidence / record name them
    stock_chain_calls, and no key or evidence pair of the kit's own says fallback."""
    stub = types.ModuleType("dit_fuse"); stub.STATS = {"calls": {"ada": 4, "gate": 0, "res": 2, "swiglu": 1}, "fallback": 7}
    monkeypatch.setitem(sys.modules, "dit_fuse", stub)
    st = sf.state(); ev = dict(sf.evidence())
    assert st["stock_chain_calls"] == 7 and ev["stock_chain_calls"] == 7 and ev["calls_gate"] == 0
    assert not any("fallback" in k for k in st) and not any("fallback" in k for k in ev)


# ---- the precision-regime dispatch on the add-on's kernel entry points (sampler_fuse.regime_dispatch / key_kernels)

def _fake_addon(monkeypatch):
    """A stand-in ``dit_fuse`` module: STATS and four kernel entry points that count their calls like the add-on's kernels and return a marker."""
    fake = types.ModuleType("dit_fuse"); fake.STATS = {"calls": {"ada": 0, "gate": 0, "res": 0, "swiglu": 0}, "fallback": 0}; fake.seen = []
    def mk(name, kind):
        def kern(*a, **k):
            fake.STATS["calls"][k.get("_kind", kind) if name == "res_gate" else kind] += 1; fake.seen.append(name); return ("fused", name)
        kern.__name__ = name; return kern
    for name, (kind, _) in sf.STOCK_EXPRESSIONS.items():
        setattr(fake, name, mk(name, kind))
    monkeypatch.setitem(sys.modules, "dit_fuse", fake)
    monkeypatch.setattr(sf, "_REGIME", {"keyed": False, "gated": 0, "stock_calls": {k: 0 for k in sf.SITE_KINDS}, "offsig_calls": {k: 0 for k in sf.KINDS}})
    assert sf.key_kernels(fake) is True and sf.key_kernels(fake) is False          # idempotent
    return fake


def test_regime_dispatch_fp32_operands_run_the_fused_kernel(monkeypatch):
    torch = pytest.importorskip("torch")
    fake = _fake_addon(monkeypatch)
    for n in (4, 7):                                                                # two token counts in one process: no per-shape state anywhere
        a, x1, x2 = (torch.randn(5, n, 8) for _ in range(3))
        assert fake.ada_tail(a, x1, x2) == ("fused", "ada_tail")
        assert fake.res_gate(x1, a, x2) == ("fused", "res_gate") and fake.res_gate(x1, a, None, _kind="gate") == ("fused", "res_gate")
        assert fake.swiglu(x1, x2) == ("fused", "swiglu")
        assert fake.attn_gate(torch.randn(2, n, 4).contiguous(), torch.randn(n, 8)) == ("fused", "attn_gate")
    assert fake.STATS["calls"] == {"ada": 2, "gate": 4, "res": 2, "swiglu": 2} and sf.offsig_calls() == {"ada": 0, "gate": 0, "res": 0, "swiglu": 0}
    st = sf.state(); ev = dict(sf.evidence()); assert st["fused_calls"] == 10 and st["regime"] == "fp32" and ev["regime_stock_calls"] == 0 and ev["offsig_calls"] == 0


def test_regime_dispatch_bf16_and_mixed_operands_run_the_stock_expression(monkeypatch):
    torch = pytest.importorskip("torch")
    F = pytest.importorskip("torch.nn.functional")
    fake = _fake_addon(monkeypatch)
    bf = torch.bfloat16
    a32 = torch.randn(5, 6, 8); x1, x2 = torch.randn(5, 6, 8, dtype=bf), torch.randn(5, 6, 8, dtype=bf)     # the bf16 regime's AdaLN site: LN output fp32, linear outputs bf16
    out = fake.ada_tail(a32, x1, x2)
    assert torch.equal(out, torch.sigmoid(x1) * a32 + x2) and out.dtype == torch.float32
    g, x, r = (torch.randn(5, 6, 8, dtype=bf) for _ in range(3))
    assert torch.equal(fake.res_gate(g, x, r), torch.sigmoid(g) * x + r) and torch.equal(fake.res_gate(g, x, None, _kind="gate"), torch.sigmoid(g) * x)
    assert torch.equal(fake.swiglu(x, r), F.silu(x) * r)
    o = torch.randn(3, 2, 6, 4, dtype=bf); gl = torch.randn(3, 6, 8, dtype=bf)                            # o [B, H, N, C], g [B, N, H*C]
    ref = (o.transpose(-2, -3) * torch.sigmoid(gl).view(3, 6, 2, 4)).reshape(3, 6, 8)
    assert torch.equal(fake.attn_gate(o, gl), ref)
    nc = torch.randn(5, 8, 6).transpose(-1, -2)                                                             # fp32 but not contiguous: out of signature too
    assert torch.equal(fake.swiglu(nc, nc), F.silu(nc) * nc)
    assert fake.seen == [] and fake.STATS["calls"] == {"ada": 0, "gate": 0, "res": 0, "swiglu": 0}          # no fused kernel ran
    assert sf.offsig_calls() == {"ada": 1, "gate": 2, "res": 1, "swiglu": 2}
    st = sf.state(); ev = dict(sf.evidence())
    assert st["regime"] == "bf16" and ev["regime"] == "bf16" and ev["offsig_calls"] == 6 and ev["fused_calls"] == 0 and ev["regime_stock_calls"] == 0
    assert not any("fallback" in k for k in st) and not any("fallback" in k for k in ev)
    fake.ada_tail(a32, a32, a32)                                                                            # an fp32 item in the same process
    assert sf.state()["regime"] == "fp32+bf16"


def test_the_multi_gpu_tally_counts_fused_and_regime_stock_calls():
    from protenix_opt import tp
    paths = tp.TALLY["sampler_fuse"][1]
    assert ("sampler_fuse", "calls", "ada") in paths and ("sampler_fuse", "regime_stock_calls", "ada") in paths
    assert ("sampler_fuse", "regime_stock_calls", "block") in paths and ("sampler_fuse", "offsig_calls", "swiglu") in paths
    rec = {"protenix_opt": {"sampler_fuse": {"calls": {"ada": 0, "gate": 0, "res": 0, "swiglu": 0}, "regime_stock_calls": {"ada": 9600, "gate": 4800, "block": 6000}, "offsig_calls": {}}}}
    assert tp.execution({3: rec}, ["sampler_fuse"])["executed"] == {"sampler_fuse": {"3": 20400}}          # a rank whose item ran the bf16 regime: served by the stock methods, counted


def test_the_site_gate_runs_the_add_on_site_in_fp32_and_the_stock_method_in_bf16(monkeypatch):
    """gate_sites wraps every dit_fuse_patch function the add-on put on the diffusion module's instances: fp32 sampler -> the add-on's site; bf16
    sampler (autocast) -> the class's own method bound to the instance, counted per site kind; another module's instance forward (the hoist's) is kept."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(sf, "_REGIME", {"keyed": False, "gated": 0, "stock_calls": {k: 0 for k in sf.SITE_KINDS}, "offsig_calls": {k: 0 for k in sf.KINDS}})

    class AdaptiveLayerNorm(torch.nn.Module):
        def forward(self, a, s): return ("stock_ada", a, s)

    class Attention(torch.nn.Module):
        def _wrap_up(self, o, q): return ("stock_wrap", o, q)
        def forward(self, x): return x

    class DiffusionTransformerBlock(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.ln = AdaptiveLayerNorm(); self.att = Attention(); self.hoisted = AdaptiveLayerNorm()
        def forward(self, a, s, z, n_queries=None): return ("stock_block", a)

    def addon_fn(tag):
        f = lambda *a, **k: (tag,) + a          # noqa: E731
        f.__module__ = sf.SITE_MODULE; return f
    hoist_fn = lambda a, s: ("hoist", a)         # noqa: E731  (another module's site: not gated)
    hoist_fn.__module__ = "dit_hoist"
    dm = torch.nn.Module(); dm.blocks = torch.nn.ModuleList([DiffusionTransformerBlock(), DiffusionTransformerBlock()])
    for b in dm.blocks:
        b.forward = addon_fn("addon_block"); b.ln.forward = addon_fn("addon_ada"); b.att._wrap_up = addon_fn("addon_wrap"); b.hoisted.forward = hoist_fn
    assert sf.gate_sites(dm) == 6 and sf.gate_sites(dm) == 0                                     # 3 sites x 2 blocks; idempotent
    b = dm.blocks[0]
    monkeypatch.setattr(sf, "bf16_regime", lambda: False)
    assert b(1, 2, 3) == ("addon_block", 1, 2, 3) and b.ln(1, 2) == ("addon_ada", 1, 2) and b.att._wrap_up(7, 8) == ("addon_wrap", 7, 8) and b.hoisted(1, 2) == ("hoist", 1)
    assert sf.regime_stock_calls() == {"ada": 0, "gate": 0, "block": 0}
    monkeypatch.setattr(sf, "bf16_regime", lambda: True)
    assert b(1, 2, 3) == ("stock_block", 1) and b.ln(1, 2) == ("stock_ada", 1, 2) and b.att._wrap_up(7, 8) == ("stock_wrap", 7, 8) and b.hoisted(1, 2) == ("hoist", 1)
    assert b(1, 2, z=3, n_queries=4) == ("stock_block", 1)                                        # keyword arguments pass through to the stock method
    assert sf.regime_stock_calls() == {"ada": 1, "gate": 1, "block": 2}


def test_bf16_regime_reads_autocast(monkeypatch):
    torch = pytest.importorskip("torch")
    assert sf.bf16_regime() is False
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cpu_inside = sf.bf16_regime()                                                            # the CUDA query: a CPU autocast region does not count
    assert cpu_inside is False
