"""The sampler attention levers (apb_levers.py: dit_attn, atom_attn) and the runner seam (runner_seam.py): the switches parse by name,
the T* rows of cc 9.0 list them and export their switches (the 8.0 rows list the sampler sites dit_attn / dit_attn_fp16 / atom_attn on the
kit's cc-8.0 binding through the shared core's provider by word, protenix_opt.apb_sm80 — protenix_opt 0.3.37; the 10.x rows list none: not in the arm there, never refused), a switch set
under a row without the lever is refused by name, the install registers on the ONE runner seam after sampler_fuse and runs the carried
package's install functions on the built model in that order, the record / LEVER evidence / census re-basing follow, and the package the
levers import is the carried one."""
import io
import os
import re
import sys
import types
from contextlib import redirect_stderr
from unittest import mock

import pytest

from protenix_opt import apb_levers as apb, kits, modes, registry, report, runner_seam, sampler_fuse as sf, stack, tp

STUBBED = ("runner", "runner.inference", "apb_stub_calls", "dit_fuse", "dit_fuse_patch")
H100 = {"name": "NVIDIA H100 80GB HBM3", "sm": "sm90", "cc": "9.0", "probe": "nvidia-smi"}
A100 = {"name": "NVIDIA A100-SXM4-80GB", "sm": "sm80", "cc": "8.0", "probe": "nvidia-smi"}


@pytest.fixture
def clean(monkeypatch):
    """Fresh seam / lever state, no AttrPatch of an earlier test, the process's own runner / package modules back afterwards."""
    from opt_core import autoload as A
    monkeypatch.setattr(A, "_PATCHES", {})
    monkeypatch.setattr(runner_seam, "_INSTALLERS", []); monkeypatch.setattr(runner_seam, "_STATE", {"patch": None, "runners": 0, "ran": []})
    monkeypatch.setattr(apb, "_STATE", {"on": [], "patch": None, "installed": {}, "errors": {}, "named": {}, "precision": None, "models": 0})
    from protenix_opt import apb_core
    apb_core._reset_for_tests()
    monkeypatch.setattr(sf, "_STATE", dict(sf._STATE, on=False, patch=None, patched=False, models=0, report=None, error=None))
    saved = {m: sys.modules.pop(m) for m in STUBBED if m in sys.modules}
    yield
    apb_core._reset_for_tests()                                   # the stub installers fill apb_core.STATE: nothing of it leaks into later tests (surface lines)
    for m in STUBBED:
        sys.modules.pop(m, None)
    sys.modules.update(saved)
    p = runner_seam._STATE.get("patch")
    if p is not None:
        p._unhook()


def _stub_runner(tmp_path, monkeypatch, body="        self.configs = configs; self.model = ('model', configs)\n"):
    pkg = tmp_path / "runner"; pkg.mkdir(); (pkg / "__init__.py").write_text("")
    (pkg / "inference.py").write_text("class InferenceRunner:\n    def __init__(self, configs):\n" + body)
    monkeypatch.syspath_prepend(str(tmp_path))


def _stub_package(tmp_path, monkeypatch, fail=None):
    """Stub installers on protenix_opt.apb_core (the real ones need torch / protenix / a GPU): they record the model on ``apb_stub_calls.CALLS``
    and fill apb_core.STATE the way the real ones do; the tier word is ``fast`` (no activation record in this process)."""
    from protenix_opt import apb_core as core
    (tmp_path / "apb_stub_calls.py").write_text("CALLS = []\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    import apb_stub_calls
    monkeypatch.setattr(core, "tier_word", lambda mode=None: "fast")
    monkeypatch.setattr(core, "core_version", lambda: "0.5.111.0")

    def dit(model, *, word, fp16=False):
        if fail:
            raise RuntimeError(fail)
        apb_stub_calls.CALLS.append(("dit_attn", model, fp16))
        st = core.STATE["dit_attn"]
        st.update(installed_on=24, cell_key="9.0", precision=("per_cell" if fp16 else "fp32_class"),
                  cell={"word": word, "face": "kernels.apb", "geometry": "dit_h16d48", "fp16_arms": ("admissible" if fp16 else "off")})
        core.STATE["dit_attn_fp16"]["on"] = bool(fp16)
        return dict(st, aside=None, left_alone=0)

    def atom(model, *, word):
        apb_stub_calls.CALLS.append(("atom_attn", model))
        st = core.STATE["atom_attn"]
        st.update(installed_on=6, cell_key="9.0", cell={"word": word, "face": "kernels.apb", "geometry": "atom_h4d32w32x128"})
        return dict(st, aside=None, left_alone=0)

    monkeypatch.setattr(core, "INSTALLERS", {"dit_attn": dit, "atom_attn": atom, "pf_attn": core.install_pf_attn})
    return apb_stub_calls


# ------------------------------------------------------------------------------------------------------------------ switches / rows
def test_switches_parse_by_name():
    assert apb.ENVS == {"dit_attn": "PTX_DIT_ATTN", "dit_attn_fp16": "PTX_DIT_ATTN_FP16", "atom_attn": "PTX_ATOM_ATTN", "pf_attn": "PTX_PF_ATTN",
                        "opm_fused": "PTX_OPM_FUSED", "pwa_fused": "PTX_PWA_FUSED"}
    assert apb.LEVERS == ("dit_attn", "dit_attn_fp16", "atom_attn", "pf_attn", "opm_fused", "pwa_fused") and apb.KERNEL_LEVERS == ("dit_attn", "atom_attn", "pf_attn", "opm_fused", "pwa_fused") and apb.PRECISION == {"dit_attn_fp16": "dit_attn"}
    for lever, env in apb.ENVS.items():
        assert apb.from_env(lever, {}) is False and apb.from_env(lever, {env: ""}) is False and apb.from_env(lever, {env: "0"}) is False
        assert apb.from_env(lever, {env: "1"}) is True
        with pytest.raises(ValueError, match=re.escape(f"{env}='2': expected 0 or 1")):          # the package's own grammar (INTEGRATION §10)
            apb.from_env(lever, {env: "2"})


def test_the_precision_lever_requires_its_kernel_lever_by_name():
    apb.check_requires(["dit_attn", "dit_attn_fp16"]); apb.check_requires(["atom_attn"]); apb.check_requires([])
    with pytest.raises(RuntimeError, match=re.escape("dit_attn_fp16: requires dit_attn (PTX_DIT_ATTN=1) — the precision lever selects the operands of dit_attn's kernel")):
        apb.check_requires(["dit_attn_fp16", "atom_attn"])
    assert registry.LEVERS["dit_attn_fp16"].requires == ("dit_attn",) and not registry.LEVERS["dit_attn"].requires and not registry.LEVERS["atom_attn"].requires
    from protenix_opt import ablation
    assert ablation.validate("fast", ["dit_attn_fp16"]) == ["dit_attn_fp16"]                       # ablated alone: the kernel stays on at tf32x3
    assert ablation.validate("fast", ["dit_attn", "dit_attn_fp16", "dit_fused", "dit_lowp"]) == ["dit_attn", "dit_attn_fp16", "dit_fused", "dit_lowp"]   # dit_fused requires dit_attn (and dit_lowp rides dit_fused): the whole chain goes together
    with pytest.raises(ablation.AblationError, match=re.escape("dit_attn: required by dit_attn_fp16, which stays on (ablate dit_attn_fp16 with it)")):
        ablation.validate("fast", ["dit_attn"])                                                      # never implied: refused by name
    with pytest.raises(ablation.AblationError, match=re.escape("dit_attn: required by dit_fused, which stays on (ablate dit_fused with it)")):
        ablation.validate("fast", ["dit_attn", "dit_attn_fp16"])
    with pytest.raises(ablation.AblationError, match="required by dit_attn_fp16"):
        ablation.validate("big", ["dit_attn", "atom_attn"])
    with pytest.raises(ablation.AblationError, match=re.escape("atom_attn: required by atom_fused, which stays on (ablate atom_fused with it)")):
        ablation.validate("big", ["atom_attn"])


def test_untested_triton_is_named_not_refused(monkeypatch):
    import opt_core.gates as G
    monkeypatch.setattr(G, "dist_version", lambda name: "3.3.1")
    assert apb.triton_note() == "untested triton 3.3.1: engaged with the 3.7 cells"
    monkeypatch.setattr(G, "dist_version", lambda name: "3.7.1")
    assert apb.triton_note() is None
    monkeypatch.setattr(G, "dist_version", lambda name: None)
    assert apb.triton_note() is None


def test_marker_grammar():
    rx = re.compile(r"^(DIT_ATTN|ATOM_ATTN|PF_ATTN):on\((\d+) modules; cell=(\d+\.\d+) [^)]*; opt_core\.kernels\.apb (\d+(?:\.\d+)+)(; NAMED [^)]*)?\)$")   # protenix_opt 0.3.51: the provider + the shared core's version
    line = apb.on_line("dit_attn", {"installed_on": 24, "cell_key": "9.0", "cell": {"word": "fast", "face": "kernels.apb", "geometry": "dit_h16d48", "fp16_arms": "admissible"}}, "0.5.111.0", [])
    assert line == "DIT_ATTN:on(24 modules; cell=9.0 face=kernels.apb fp16_arms=admissible geometry=dit_h16d48 word=fast; opt_core.kernels.apb 0.5.111.0)" and rx.match(line)
    named = apb.on_line("atom_attn", {"installed_on": 6, "cell_key": "8.0", "cell": {"word": "big", "face": "kernels.apb"}}, "0.5.111.0", ["untested triton 3.9.0: engaged with the 3.7 cells"])
    assert named == "ATOM_ATTN:on(6 modules; cell=8.0 face=kernels.apb word=big; opt_core.kernels.apb 0.5.111.0; NAMED untested triton 3.9.0: engaged with the 3.7 cells)" and rx.match(named)
    assert apb.on_line("opm_fused", {"installed_on": 4, "cell_key": "9.0", "cell": {"ln_rows": 64}}, "0.1.0", []) == "OPM_FUSED:on(4 modules; cell=9.0 ln_rows=64; fpf_msa 0.1.0)"
    assert apb.MARKS == {"dit_attn": "DIT_ATTN:", "dit_attn_fp16": "DIT_ATTN_FP16:", "atom_attn": "ATOM_ATTN:", "pf_attn": "PF_ATTN:",
                         "opm_fused": "OPM_FUSED:", "pwa_fused": "PWA_FUSED:"}


def test_registry_rows_and_mode_membership():
    assert apb.LEVERS == ("dit_attn", "dit_attn_fp16", "atom_attn", "pf_attn", "opm_fused", "pwa_fused") and apb.KERNEL_LEVERS == ("dit_attn", "atom_attn", "pf_attn", "opm_fused", "pwa_fused")
    for lever in apb.LEVERS:
        lv = registry.LEVERS[lever]
        assert lv.tier == registry.TOLERANCE and lv.probe == "marker" and lv.extra and lv.row_dependent and lv.env_keys == (apb.ENVS[lever],)
        assert lever in modes.MODES["fast"] and lever not in modes.MODES["exact"]
        assert lever in modes.MODES["big"] and lever in modes.big_levers()          # pf_attn too (carried in big: it composes with apb_bias_chunk's chunked producer)
        assert lever not in modes.BIG_DROPPED and apb.ENVS[lever] not in modes.BIG_POST
        impl = (("opt_core.kernels.apb", "core") if apb.LEVER_PKG[lever] == "core"                       # protenix_opt 0.3.51: the shared core's provider by tier word (apb_core.py)
                else ("forward/flashpairformer/third_party/" + apb.PACKAGES[apb.LEVER_PKG[lever]][0], "kit"))
        assert report.STRATEGY_IDS[lever] == apb.STRATEGIES[lever] and report.IMPL[lever] == impl
        fam = apb.MARKS[lever]
        assert stack.MARKERS[lever] == (fam, (fam,)), lever          # the family alone: ':patched' at activation (seam armed, runner not built yet) must classify as on, ':unavailable(' carries BAD
        assert lever in tp.TALLY
    assert apb.STRATEGIES == {"dit_attn": "F5.flash_attn_dense", "dit_attn_fp16": "F4.autocast_policy", "atom_attn": "F5.flash_attn_dense",
                              "pf_attn": "LOCAL.protenix_v2.pf_attn", "opm_fused": "LOCAL.protenix_v2.opm_fused", "pwa_fused": "LOCAL.protenix_v2.pwa_fused"}
    assert "PRECISION" in registry.LEVERS["dit_attn_fp16"].description and "TIER WORD" in registry.LEVERS["dit_attn"].cells and "TIER WORD" in registry.LEVERS["atom_attn"].cells
    assert registry.LEVERS["dit_attn"].subsumes == (("sampler_fuse", "calls_gate", "0"),)
    assert registry.LEVERS["atom_attn"].subsumes == (("sampler_graph", "glue_padbias_builds", "0"), ("sampler_graph", "glue_padbias_hits", "0"))
    for lv in registry.LEVERS.values():                                          # every declared census interaction names a registry lever
        assert all(l in registry.LEVERS for l, _, _ in lv.subsumes)


def test_rows_export_the_switches_of_the_cells_they_have_and_no_other_row_does():
    SAMPLER = {"PTX_DIT_ATTN", "PTX_DIT_ATTN_FP16", "PTX_ATOM_ATTN"}                                   # fpf_apb's dit_attn / atom_attn sites: the package's 9.0 cells, the kit's cc-8.0 binding (apb_sm80)
    for key, row in modes.README_ROWS.items():
        listed = {k for part in row["fast"].values() for k in part} & set(apb.ENVS.values())
        if key.startswith("9.0|"):
            assert listed == SAMPLER | {"PTX_PF_ATTN", "PTX_OPM_FUSED", "PTX_PWA_FUSED"} and all(row["fast"]["post"][k] == "1" for k in listed), key
        elif key.startswith("8.0|"):
            assert listed == SAMPLER and all(row["fast"]["post"][k] == "1" for k in listed), f"{key}: the 8.0 rows list exactly the sampler attention levers (the kit's cc-8.0 binding, apb_sm80)"
        else:
            assert not listed, f"{key}: the levers have no cells there — the row must not list them"
        assert not ({k for part in row["exact"].values() for k in part} & set(apb.ENVS.values())), f"{key}: exact never lists a TOLERANCE lever"
    assert modes.readme_row("9.0|3.7", "big")["post"]["PTX_DIT_ATTN"] == "1"                      # big through its base
    assert modes.readme_row("8.0|3.7", "big")["post"]["PTX_DIT_ATTN"] == "1" and "PTX_PF_ATTN" not in modes.readme_row("8.0|3.7", "fast")["post"]


def test_switch_under_a_row_without_the_lever_is_refused_by_name():
    with pytest.raises(stack.ActivationError, match="PTX_DIT_ATTN set under mode exact: the dit_attn lever is not in this mode's row"):
        stack._apb_levers_on("exact", {"PTX_DIT_ATTN": "1"})
    assert stack._apb_levers_on("fast", {"PTX_DIT_ATTN": "1", "PTX_ATOM_ATTN": "1"}) == ["dit_attn", "atom_attn"]
    assert stack._apb_levers_on("fast", {"PTX_ATOM_ATTN": "1"}) == ["atom_attn"] and stack._apb_levers_on("fast", {}) == []
    assert stack._apb_levers_on("fast", {"PTX_DIT_ATTN": "1", "MODEL_OPT_LEVERS_OFF": "dit_attn"}) == []          # ablated: left off, not refused


def test_on_a_card_whose_row_lacks_them_the_levers_are_not_in_the_arm_not_refused():
    on, fb, skipped, why = stack._classify("fast", ["DIT_ATTN:armed", "ATOM_ATTN:armed"], {}, row=modes.readme_row("10.0|3.7", "fast"), row_key="10.0|3.7", sm="sm100")
    assert "dit_attn" in skipped and "atom_attn" in skipped and not {"dit_attn", "atom_attn"} & (set(on) | set(fb))
    assert why["dit_attn"].startswith(stack.NOT_IN_ROW)
    on8, fb8, sk8, _ = stack._classify("fast", ["DIT_ATTN:armed", "ATOM_ATTN:armed"], {}, row=modes.readme_row("8.0|3.7", "fast"), row_key="8.0|3.7", sm="sm80")
    assert {"dit_attn", "atom_attn"} <= set(on8) and not {"dit_attn", "atom_attn"} & (set(sk8) | set(fb8))      # cc 8.0: the row lists them (bound through apb_sm80)
    on, fb, skipped, why = stack._classify("fast", ["DIT_ATTN:armed", "ATOM_ATTN:armed"], {"PTX_DIT_ATTN": "1", "PTX_ATOM_ATTN": "1"},
                                           row=modes.readme_row("9.0|3.7", "fast"), row_key="9.0|3.7", sm="sm90")
    assert "dit_attn" in on and "atom_attn" in on
    on, fb, skipped, why = stack._classify("fast", ["DIT_ATTN:armed"], {}, row=modes.readme_row("9.0|3.7", "fast"), row_key="9.0|3.7", sm="sm90")
    assert "atom_attn" in fb and why["atom_attn"] == "no marker in the kit's applied list"                       # in the row, no marker: a fallback by name


# ------------------------------------------------------------------------------------------------------------------ the seam
def test_the_seam_runs_installers_in_registration_order_after_the_stock_constructor(clean, monkeypatch, tmp_path):
    _stub_runner(tmp_path, monkeypatch)
    order = []
    runner_seam.add("first", lambda r: order.append(("first", r.configs))); runner_seam.add("second", lambda r: order.append(("second", r.configs)))
    runner_seam.add("first", lambda r: order.append(("first*", r.configs)))                    # idempotent by name: replaces the callable, keeps the position
    assert runner_seam.names() == ["first", "second"]
    p = runner_seam.arm(); assert p is runner_seam.arm() and runner_seam.patched() is False    # one AttrPatch per process
    import runner.inference as RI
    r = RI.InferenceRunner({"n": 2})
    assert runner_seam.patched() and r.model == ("model", {"n": 2}) and order == [("first*", {"n": 2}), ("second", {"n": 2})]
    assert runner_seam.state() == {"patched": True, "installers": ["first", "second"], "runners": 1, "ran": ["first", "second"]}


def test_an_installer_error_propagates_and_stops_the_later_ones(clean, monkeypatch, tmp_path):
    _stub_runner(tmp_path, monkeypatch)
    later = []
    def boom(r): raise RuntimeError("first lever failed by name")
    runner_seam.add("boom", boom); runner_seam.add("later", lambda r: later.append(1)); runner_seam.arm()
    import runner.inference as RI
    with pytest.raises(RuntimeError, match="first lever failed by name"):
        RI.InferenceRunner({})
    assert later == [] and runner_seam.state()["runners"] == 0


def test_sampler_fuse_and_the_apb_levers_share_the_seam_in_order(clean, monkeypatch, tmp_path):
    """sampler_fuse.install() then apb_levers.install(): ONE patch, installers [sampler_fuse, dit_attn, atom_attn]; on construction the
    DIT_FUSE add-on installs first, then fpf_apb's two install functions on the same model."""
    tools = tmp_path / "tools"; tools.mkdir()
    (tools / "dit_fuse_patch.py").write_text("import dit_fuse as DF\nimport apb_stub_calls as fpf_apb\n"
                                              "class _M:\n    def modules(self): return []\n"
                                              "def _dm(model): return _M()\n"
                                              "def install(model, levers):\n    fpf_apb.CALLS.append(('sampler_fuse', model)); return {'installed': True, 'n_ada': 66, 'n_gate': 24, 'n_res_apb': 24, 'n_ctb': 30}\n")
    (tools / "dit_fuse.py").write_text("STATS = {'calls': {'ada': 0, 'gate': 0, 'res': 0, 'swiglu': 0}, 'fallback': 0}\n"
                                     "def ada_tail(a, x1, x2): return 0\ndef res_gate(g, x, res=None, _kind='res'): return 0\ndef swiglu(x, y): return 0\ndef attn_gate(o, g): return 0\n")
    monkeypatch.setattr(sf, "tools_dir", lambda: str(tools)); monkeypatch.setattr(apb, "triton_note", lambda: None)   # the image's own triton is not this test's subject
    _stub_package(tmp_path, monkeypatch); _stub_runner(tmp_path, monkeypatch)
    m1 = sf.install(); m2 = apb.install(["dit_attn", "dit_attn_fp16", "atom_attn"])
    assert m1 == "DIT_FUSE:ada,gate,res,swiglu(armed)" and m2 == ["DIT_ATTN:armed", "DIT_ATTN_FP16:requested", "ATOM_ATTN:armed"]
    assert runner_seam.names() == ["sampler_fuse", "dit_attn", "atom_attn"] and sf._STATE["patch"] is runner_seam._STATE["patch"] is apb._STATE["patch"]   # the precision lever has no installer of its own
    err = io.StringIO()
    with redirect_stderr(err):
        import runner.inference as RI
        r = RI.InferenceRunner({"n": 1})
    import apb_stub_calls as fpf_apb
    assert fpf_apb.CALLS == [("sampler_fuse", r.model), ("dit_attn", r.model, True), ("atom_attn", r.model)]     # the order the levers need; fp16=True = dit_attn_fp16 on
    st = apb.state()
    assert st["patched"] is True and st["models"] == 1 and st["dit_attn"]["installed_on"] == 24 and st["atom_attn"]["installed_on"] == 6
    assert st["package"] == "opt_core.kernels.apb"
    assert st["dit_attn"]["cell"] == {"word": "fast", "face": "kernels.apb", "geometry": "dit_h16d48", "fp16_arms": "admissible"} and st["dit_attn"]["precision"] == "per_cell" and st["atom_attn"]["named"] is None
    assert st["dit_attn_fp16"] == {"on": True, "engaged": True, "engaged_calls": 0, "kernel": "dit_attn", "error": None}
    lines = err.getvalue().splitlines()
    assert "[protenix-opt] DIT_ATTN:on(24 modules; cell=9.0 face=kernels.apb fp16_arms=admissible geometry=dit_h16d48 word=fast; opt_core.kernels.apb 0.5.111.0)" in lines and "[protenix-opt] DIT_ATTN_FP16:on" in lines
    assert "[protenix-opt] ATOM_ATTN:on(6 modules; cell=9.0 face=kernels.apb geometry=atom_h4d32w32x128 word=fast; opt_core.kernels.apb 0.5.111.0)" in lines
    ev = dict(apb.evidence("dit_attn"))
    assert ev["modules"] == 24 and ev["cell"] == "9.0" and ev["word"] == "fast" and ev["precision"] == "per_cell" and ev["fp16_arms"] == "admissible" and (ev["named"] == "none" or ev["named"].startswith("untested_triton"))
    assert ev["served"] == 0 and ev["rows"] == "none" and list(ev)[-2:] == ["served", "rows"]                        # the provider tallies trail the line (nothing served in this process)
    assert dict(apb.evidence("atom_attn"))["named"] == "none"
    assert dict(apb.evidence("dit_attn_fp16")) == {"models": 1, "kernel": "dit_attn", "engaged": 1, "opd": "fp16", "kernel_modules": 24, "kernel_calls": 0, "fp16_calls": 0}


def test_dit_attn_alone_installs_with_the_fp16_arms_off(clean, monkeypatch, tmp_path):
    """MODEL_OPT_LEVERS_OFF=dit_attn_fp16 leaves dit_attn on: install_dit_attn(model, word=<tier>, fp16=False) -- the provider's fp16-operand arms
    are inadmissible to the tier word -- and DIT_ATTN_FP16:off on the stream."""
    monkeypatch.setattr(apb, "triton_note", lambda: None)
    _stub_package(tmp_path, monkeypatch); _stub_runner(tmp_path, monkeypatch)
    assert apb.install(["dit_attn", "atom_attn"]) == ["DIT_ATTN:armed", "ATOM_ATTN:armed"]
    err = io.StringIO()
    with redirect_stderr(err):
        import runner.inference as RI
        r = RI.InferenceRunner({"n": 1})
    import apb_stub_calls as fpf_apb
    assert fpf_apb.CALLS == [("dit_attn", r.model, False), ("atom_attn", r.model)]
    st = apb.state()
    assert st["dit_attn"]["cell"]["fp16_arms"] == "off" and st["dit_attn"]["precision"] == "fp32_class" and st["dit_attn_fp16"]["on"] is False and st["dit_attn_fp16"]["engaged"] is False
    assert "[protenix-opt] DIT_ATTN:on(24 modules; cell=9.0 face=kernels.apb fp16_arms=off geometry=dit_h16d48 word=fast; opt_core.kernels.apb 0.5.111.0)" in err.getvalue().splitlines() and "[protenix-opt] DIT_ATTN_FP16:off" in err.getvalue().splitlines()
    with pytest.raises(RuntimeError, match="dit_attn_fp16: requires dit_attn"):
        apb.install(["dit_attn_fp16"]) if False else apb.check_requires(["dit_attn_fp16"])


def test_an_install_error_steps_aside_by_name(clean, monkeypatch, tmp_path):
    """protenix_opt 0.3.53: a lever that cannot engage at install steps aside BY NAME -- `<MARK>aside(install: ...)` on the stream, the reason
    recorded (reconcile names the lever a fallback with it), the stock statement left in place, the runner built (no raise); the other levers install."""
    _stub_package(tmp_path, monkeypatch, fail="dit_attn: expected 24 primitives.Attention token modules, found 0"); _stub_runner(tmp_path, monkeypatch)
    monkeypatch.setattr(apb, "triton_note", lambda: None)
    apb.install(["dit_attn", "dit_attn_fp16", "atom_attn"])
    err = io.StringIO()
    with redirect_stderr(err):
        import runner.inference as RI
        r = RI.InferenceRunner({})
    lines = err.getvalue().splitlines()
    assert "[protenix-opt] DIT_ATTN:aside(install: RuntimeError('dit_attn: expected 24 primitives.Attention token modules, found 0'); the modules keep the stock statement)" in lines, lines
    assert "[protenix-opt] DIT_ATTN_FP16:off" in lines and any(l.startswith("[protenix-opt] ATOM_ATTN:on(6 modules;") for l in lines), lines   # atom_attn still installs
    st = apb.state()
    assert "expected 24" in st["dit_attn"]["error"] and st["dit_attn"]["installed_on"] == 0 and st["atom_attn"]["installed_on"] == 6
    assert st["dit_attn_fp16"]["engaged"] is False
    rep = {"active": True, "mode": "fast", "levers_applied": ["dit_attn", "dit_attn_fp16", "atom_attn"], "levers_fallback": []}
    rec = stack.reconcile(dict(rep), {})                                                               # the exit record names the aside: a fallback BY NAME with its reason
    assert "dit_attn" in rec["levers_fallback"] and "dit_attn_fp16" in rec["levers_fallback"] and "atom_attn" in rec["levers_applied"]
    assert "expected 24" in rec["fallback_reasons"]["dit_attn"]


def test_a_missing_package_is_named(clean, monkeypatch):
    import importlib as _il
    real = _il.import_module
    def no_pkg(name, *a, **k):
        if name == apb.MSA_IMPORT_NAME: raise ImportError("No module named 'protenix_fpf_msa'")
        return real(name, *a, **k)
    monkeypatch.setattr(apb.importlib, "import_module", no_pkg)
    with pytest.raises(RuntimeError, match="protenix_fpf_msa \\(fpf_msa\\) is not importable"):
        apb._package("opm_fused")
    assert apb._package("dit_attn").__name__ == "protenix_opt.apb_core"                       # the attention levers' installer module is the kit's own tier-word binding


def test_the_binding_is_the_shared_core_provider_by_tier_word():
    """protenix_opt 0.3.51: the attention levers bind opt_core.kernels.apb by the mode's tier word on every card (apb_core.py); no kit kernel,
    cell table, tile table or size floor; there is no per-card binding module (apb_sm80) and no carried dit_attn_exact package."""
    from protenix_opt import apb_core as core
    tp_dir = os.path.join(kits.kit_dir("flashpairformer"), "third_party")
    assert not os.path.exists(os.path.join(tp_dir, "protenix_fpf_dit_attn_exact")), "the exact row is the provider's dit_exact (its carried package), not a kit copy"
    assert not os.path.exists(os.path.join(os.path.dirname(apb.__file__), "apb_sm80.py")), "one binding for every card"
    assert core.TIER_WORDS == {"exact": "exact", "fast": "fast", "big": "big"} and core.LEVERS == ("dit_attn", "atom_attn", "pf_attn") and core.EXACT_LEVER == "dit_attn_exact"
    assert core.CELLS == {"dit_attn": "dit_h16d48", "atom_attn": "atom_h4d32w32x128", "pf_attn": "pf_h16d24"}
    assert set(core.INSTALLERS) == set(core.LEVERS)
    for mode in ("exact", "fast", "big"):
        assert core.tier_word(mode) == mode
    with pytest.raises(RuntimeError, match="no provider tier word for mode 'off'"):
        core.tier_word("off")
    src = open(core.__file__, encoding="utf-8").read()
    body = src.split('"""', 2)[2]
    assert "third_party" not in body and "protenix_fpf_apb" not in body and "protenix_fpf_dit_attn_exact" not in body   # the binding imports the core's face, never a carried kit package
    assert not re.search(r"BLOCK_M|BLOCK_N|num_warps|min_tokens|FLOOR", body), "no tile table / size floor in the kit: the provider's table decides"
    assert sorted(set(re.findall(r"""environ(?:\.get)?\s*[\[(]\s*["']([A-Z][A-Z0-9_]+)["']""", body))) == ["PROTENIX_OPT", "PTX_DIT_ATTN_EXACT"], "reads the mode (env route) and dit_attn_exact's switch, nothing else"
    from opt_core.kernels import apb as A                                              # the provider surface the binding calls, by name
    assert all(hasattr(A, n) for n in core.FACE_NAMES) and set(core.TIER_WORDS.values()) <= set(A.TIER_WORDS) and set(core.CELLS.values()) <= set(A.CELL_WORDS)
    assert "dit_exact" in A.EXACT_ROWS and "dit_exact" in A.ROW_NAMES and {"fpf_apb", "fpf_atom", "fpf_pf_bias", "ln_proj", "sdpa", "sdpa_gather"} <= set(A.ROW_NAMES)
    m = os.path.join(tp_dir, "protenix_fpf_msa")                                        # the MSA package stays carried (opm_fused / pwa_fused)
    msrc = open(os.path.join(m, "install.py"), encoding="utf-8").read(); minit = open(os.path.join(m, "__init__.py"), encoding="utf-8").read()
    assert "def install_opm_fused(model)" in msrc and "def install_pwa_fused(model)" in msrc and "def report()" in msrc and '__version__ = "0.1.0"' in minit


def test_the_tier_word_resolves_through_the_provider_for_our_call_classes():
    """Class contract (CPU, the provider's table): for this kit's geometry cells the three tier words resolve to a row on both cards at the ladder
    sizes -- never an uncovered cell; exact = the provider's dit_exact row on a vouched cc-9.0 stack (with its ABI key), a stock-family arm by name
    elsewhere; fast / big name carried kernels for the DiT graph classes and the atom windows."""
    from opt_core.kernels import apb as A
    import opt_core.cell_census as census
    census.reset()                                                                         # table reads only: the provider's process census of these probe classes is not this kit's record
    try:
        _tier_word_classes(A)
    finally:
        census.reset()


def _tier_word_classes(A):
    stacks = A.table().get("stacks") or {}
    h100 = next((k for k in stacks if k.startswith("H100:torch2.13")), None)
    for cc in ("9.0", "8.0"):
        for word in ("exact", "fast", "big"):
            for n in (20, 199, 400, 800, 1200):
                for capture in (False, True):
                    s = A.select(cc, "fp32", "dit_h16d48", n, word=word, samples=5, capture=capture, heads=16, head_dim=48)
                    assert s.row in A.ROW_NAMES and (word != "exact" or s.row in A.STOCK_ROWS or s.row in A.EXACT_ROWS), (cc, word, n, capture, s)
                a = A.select(cc, "fp32", "atom_h4d32w32x128", n, word=word, samples=5, heads=4, head_dim=32)   # the windowed cell's token measure = atoms // 8
                assert a.row in ("fpf_atom", "dtk_window", "sdpa_gather") and (word != "exact" or a.row == "sdpa_gather"), (cc, word, n, a)   # no exact-class windowed row: the exact word names the gather statement
                p = A.select(cc, "bf16", "bias_c128h16", n, word=word, samples=1, heads=16, c_z=128)
                c = A.select(cc, "bf16", "pf_h16d24", n, word=word, samples=1, heads=16, head_dim=24)
                assert p.row in A.ROW_NAMES and c.row in A.ROW_NAMES, (cc, word, n, p, c)
        g = A.select(cc, "fp32", "dit_h16d48", 800, word="fast", samples=5, capture=True, heads=16, head_dim=48)
        assert g.row not in A.STOCK_ROWS, (cc, g)                                          # the sampler's captured DiT step: a carried kernel row on both cards
    if h100:                                                                               # exact on the vouched H100 stack: the provider's dit_exact row (bitwise class)
        e = A.select("9.0", "fp32", "dit_h16d48", 800, word="exact", samples=5, capture=True, heads=16, head_dim=48, stack=h100, abi="torch2.13.0-cu130-sm90")
        assert e.row == "dit_exact" and e.cls == "bitwise", e
        u = A.select("9.0", "fp32", "dit_h16d48", 800, word="exact", samples=5, capture=True, heads=16, head_dim=48, stack="H100:torch9.9.9+cu130/3.7.1/cueq0.11.1")
        assert u.row in A.STOCK_ROWS, u                                                    # an unvouched stack: the statement's own arm by name, never the kernel
    e80 = A.select("8.0", "fp32", "dit_h16d48", 800, word="exact", samples=5, capture=True, heads=16, head_dim=48)
    assert e80.row in A.STOCK_ROWS, e80                                                    # cc 8.0: no exact-class DiT kernel row; the exact rows of 8.0 do not list dit_attn_exact


def test_reconcile_names_a_runner_never_imported(clean):
    apb._STATE["on"] = ["dit_attn", "dit_attn_fp16", "atom_attn"]
    rep = {"active": True, "mode": "fast", "levers_applied": ["dit_attn", "dit_attn_fp16", "atom_attn", "deadskip"], "levers_fallback": []}
    r = stack.reconcile(rep, {})
    assert {"dit_attn", "dit_attn_fp16", "atom_attn"} <= set(r["levers_fallback"]) and not {"dit_attn", "dit_attn_fp16", "atom_attn"} & set(r["levers_applied"])
    assert r["fallback_reasons"]["dit_attn"].startswith("runner.inference was never imported") and r["partial"] is True
    assert r["fallback_reasons"]["dit_attn_fp16"].startswith("rides dit_attn: runner.inference was never imported")     # the precision lever leaves the applied list with its kernel lever, by name
    assert r["apb_levers"]["patched"] is False


def test_lever_lines_carry_evidence_and_the_census_rebase(clean, monkeypatch):
    monkeypatch.setattr(apb, "state", lambda: {"patched": True, "models": 1, "package": "opt_core.kernels.apb", "version": "0.5.111.0",
                                               "dit_attn": {"on": True, "installed_on": 24, "calls": 288, "cell_key": "9.0", "cell": {"word": "fast", "geometry": "dit_h16d48"}, "named": None, "error": None, "precision": "per_cell",
                                                            "served": {"fpf_apb": 48, "fpf_apb:fp16": 240}, "aside_kinds": {}},
                                               "dit_attn_fp16": {"on": True, "engaged": True, "engaged_calls": 240, "kernel": "dit_attn", "error": None},
                                               "atom_attn": {"on": True, "installed_on": 6, "calls": 72, "cell_key": "9.0", "cell": {"word": "fast"}, "named": None, "error": None, "served": {"fpf_atom": 72}, "aside_kinds": {}}})
    monkeypatch.setattr(sf, "evidence", lambda: [("models", 1), ("ada", 66), ("gate", 24), ("res", 24), ("swiglu", 30), ("calls_ada", 10), ("calls_gate", 0), ("calls_res", 5), ("calls_swiglu", 5)])
    rep = {"active": True, "mode": "fast", "levers_applied": [n for n in modes.MODES["fast"] if n not in ("dit_fused", "dit_lowp", "atom_fused", "cond_dedupe", "atom_attn_exact")], "levers_fallback": [], "reconciled": {"records": [], "moves": {}}}
    lines = {l.rsplit(" lever=", 1)[1]: l for l in report.lever_lines(rep)}
    d, a, fz, sg = lines["dit_attn"], lines["atom_attn"], lines["sampler_fuse"], lines["sampler_graph"]
    assert " state=on " in d and " name=F5.flash_attn_dense " in d and " modules=24 calls=288 cell=9.0 precision=per_cell geometry=dit_h16d48 word=fast named=none served=288 rows=fpf_apb:48+fpf_apb:fp16:240 subsumes=sampler_fuse:calls_gate " in d
    assert " impl=opt_core.kernels.apb origin=core " in d and " impl=opt_core.kernels.apb origin=core " in a
    p16 = lines["dit_attn_fp16"]
    assert " name=F4.autocast_policy state=on " in p16 and " strategy=F4.autocast_policy models=1 kernel=dit_attn engaged=1 opd=fp16 kernel_modules=24 kernel_calls=288 fp16_calls=240 " in p16
    assert " modules=6 calls=72 cell=9.0 word=fast named=none served=72 rows=fpf_atom:72 subsumes=sampler_graph:glue_padbias_builds,sampler_graph:glue_padbias_hits " in a
    assert " calls_gate=0 " in fz and " subsumed_by=dit_attn:calls_gate(expected=0) calls_gate_expected=0 calls_gate_census=ok " in fz      # re-based, not a fallback
    assert " subsumed_by=atom_attn:glue_padbias_builds(expected=0),atom_attn:glue_padbias_hits(expected=0) " in sg
    monkeypatch.setattr(sf, "evidence", lambda: [("models", 1), ("calls_gate", 480)])
    fz2 = {l.rsplit(" lever=", 1)[1]: l for l in report.lever_lines(rep)}["sampler_fuse"]
    assert " calls_gate_census=rebased_mismatch " in fz2                                                         # named on the line, the run's own evidence
    off = {l.rsplit(" lever=", 1)[1]: l for l in report.lever_lines(dict(rep, levers_applied=[x for x in modes.MODES["fast"] if x not in ("dit_attn", "dit_attn_fp16")], levers_ablated=["dit_attn", "dit_attn_fp16"]))}
    assert " subsumed_by=" not in off["sampler_fuse"] and " state=off reason=ablated " in off["dit_attn"] and " state=off reason=ablated " in off["dit_attn_fp16"]   # nothing re-based when the subsuming lever is off


def test_the_written_process_record_and_the_tp_tally_carry_the_levers(clean, monkeypatch):
    st = {"patched": True, "models": 1, "package": "fpf_apb", "version": "0.2.0",
          "dit_attn": {"on": True, "installed_on": 24, "calls": 0, "cell_key": "9.0", "cell": {}, "named": None, "error": None, "precision": "fp16"},
          "dit_attn_fp16": {"on": True, "engaged": True, "kernel": "dit_attn", "error": None},
          "atom_attn": {"on": True, "installed_on": 6, "calls": 72, "cell_key": "9.0", "cell": {}, "named": None, "error": None}}
    key, paths = tp.TALLY["dit_attn"]; assert key == "protenix_opt" and paths == (("apb_levers", "dit_attn", "calls"),)
    rec = {"protenix_opt": {"apb_levers": st}}
    assert tp._count(rec, *tp.TALLY["dit_attn"]) == 0                                           # the line's block functions never call Attention.forward: inert by count
    assert tp._count(rec, *tp.TALLY["atom_attn"]) == 72


# ------------------------------------------------------------------------------------------------------------------ dry run end to end
def _dry_run(mode, gpu, triton="3.7.1", env=None):
    err = io.StringIO()
    with mock.patch.dict(os.environ, env or {}), mock.patch.object(stack, "protenix_version", return_value="2.0.0"), \
         mock.patch.object(modes, "triton_version", return_value=triton), mock.patch.object(stack, "gpu_probe_smi", return_value=dict(gpu)), redirect_stderr(err):
        r = stack.activate(mode, dry_run=True)
    return r, err.getvalue()


class TestDryRun:
    def setup_method(self):
        stack._REPORT = None; self._env = dict(os.environ)

    def teardown_method(self):
        stack._REPORT = None; os.environ.clear(); os.environ.update(self._env)

    def test_fast_on_h100_plans_both_levers_and_exports_their_switches(self):
        r, err = _dry_run("fast", H100)
        assert "dit_attn" in r["levers_applied"] and "atom_attn" in r["levers_applied"]
        assert r["env"]["PTX_DIT_ATTN"] == "1" and r["env"]["PTX_ATOM_ATTN"] == "1"
        line = [l for l in err.splitlines() if l.startswith("[protenix-opt] DRY-RUN mode=fast")][0]
        assert ",dit_attn," in line + "," and "atom_attn" in line

    def test_exact_never_lists_them(self):
        r, _ = _dry_run("exact", H100)
        assert not {"dit_attn", "atom_attn"} & set(r["levers_applied"]) and "PTX_DIT_ATTN" not in r["env"]

    def test_fast_on_a100_plans_the_sampler_sites_and_names_pf_attn_out_of_the_arm(self):
        r, err = _dry_run("fast", A100)
        assert [n for n in r["levers_applied"] if n in ("dit_attn", "dit_attn_fp16", "atom_attn")] == ["dit_attn", "dit_attn_fp16", "atom_attn"]
        assert r["env"]["PTX_DIT_ATTN"] == "1" == r["env"]["PTX_DIT_ATTN_FP16"] == r["env"]["PTX_ATOM_ATTN"] and "PTX_PF_ATTN" not in r["env"]
        assert "pf_attn" in r["levers_not_in_arm"] and not {"dit_attn", "atom_attn"} & set(r["levers_not_in_arm"])
        assert "CARD LEVER SET kernel_key=8.0|3.7" in err and "pf_attn" in err
        b, _ = _dry_run("big", A100)
        assert [n for n in b["levers_applied"] if n in ("dit_attn", "dit_attn_fp16", "atom_attn")] == ["dit_attn", "dit_attn_fp16", "atom_attn"]   # big through its base row

    def test_ablation_removes_exactly_the_switch(self):
        plain, _ = _dry_run("fast", H100); r, err = _dry_run("fast", H100, env={"MODEL_OPT_LEVERS_OFF": "dit_attn_fp16"})
        assert r["levers_ablated"] == ["dit_attn_fp16"] and "PTX_DIT_ATTN_FP16" not in r["env"] and r["env"]["PTX_DIT_ATTN"] == "1" == r["env"]["PTX_ATOM_ATTN"]
        assert [n for n in plain["levers_applied"] if n != "dit_attn_fp16"] == r["levers_applied"]                  # the kernel lever stays on (tf32x3)
        both, _ = _dry_run("fast", H100, env={"MODEL_OPT_LEVERS_OFF": "dit_attn,dit_attn_fp16,dit_fused,dit_lowp"})
        assert both["levers_ablated"] == ["dit_attn", "dit_attn_fp16", "dit_fused", "dit_lowp"] and "PTX_DIT_ATTN" not in both["env"] and "PTX_DIT_ATTN_FP16" not in both["env"] and "PTX_DIT_FAST" not in both["env"]

    def test_ablating_the_kernel_lever_alone_is_refused_by_name(self):
        r, err = _dry_run("fast", H100, env={"MODEL_OPT_LEVERS_OFF": "dit_attn"})
        assert r["active"] is False and "dit_attn: required by dit_attn_fp16, which stays on (ablate dit_attn_fp16 with it)" in r["reason"] and "NOT ACTIVE" in err

    def test_fast_on_h100_plans_all_three(self):
        r, err = _dry_run("fast", H100)
        assert [n for n in r["levers_applied"] if n in apb.LEVERS] == ["dit_attn", "dit_attn_fp16", "atom_attn", "pf_attn", "opm_fused", "pwa_fused"] and r["env"]["PTX_DIT_ATTN_FP16"] == "1"
        b, _ = _dry_run("big", H100)
        assert [n for n in b["levers_applied"] if n in apb.LEVERS] == ["dit_attn", "dit_attn_fp16", "atom_attn", "pf_attn", "opm_fused", "pwa_fused"] and b["env"]["PTX_PF_ATTN"] == "1"   # pf_attn: the fast row's export, carried by big (0.3.33)


def test_every_kernel_lever_line_carries_its_census(monkeypatch):
    """report.evidence dispatches all six levers to apb_levers.evidence: the LEVER line names models / modules / calls / cell (+ chunks /
    zcache_hits for the MSA levers) — a lever line without its census cannot show the kernel actually served."""
    st = {"patched": True, "models": 1, "package": "fpf_apb", "version": "0.2.2", "msa_version": "0.1.0", "dit_attn_fp16": {"on": True, "engaged": True, "error": None}}
    for lever, n, extra in (("dit_attn", 24, {}), ("atom_attn", 6, {}), ("pf_attn", 52, {}), ("opm_fused", 4, {"chunks": 88}), ("pwa_fused", 3, {"zcache_hits": 30})):
        st[lever] = {"on": True, "installed_on": n, "calls": 7, "cell_key": "9.0", "cell": {"bias_tma": True}, "named": None, "error": None, "precision": "fp16", "extra": extra}
    monkeypatch.setattr(apb, "state", lambda: st)
    rep = {"active": True, "mode": "fast", "levers_applied": list(modes.MODES["fast"]), "levers_fallback": [], "reconciled": {"records": [], "moves": {}}}
    lines = {l.rsplit("lever=", 1)[1]: l for l in report.lever_lines(rep)}
    for lever, n in (("pf_attn", 52), ("opm_fused", 4), ("pwa_fused", 3), ("dit_attn", 24), ("atom_attn", 6)):
        assert f" modules={n} " in lines[lever] and " calls=7 " in lines[lever] and " cell=9.0 " in lines[lever], lines[lever]
    assert " chunks=88 " in lines["opm_fused"] and " zcache_hits=30 " in lines["pwa_fused"]
    # the real packages report left_alone as a LIST of module names (and may report any extra as a non-int): the pair is its length / a token, never a crash at FINAL
    st["pf_attn"]["extra"] = {"left_alone": ["pairformer_stack.blocks.0.attention", "x.y"], "chunks": "n/a"}
    lines = {l.rsplit("lever=", 1)[1]: l for l in report.lever_lines(rep)}
    assert " left_alone=2 " in lines["pf_attn"] and " chunks=n/a " in lines["pf_attn"], lines["pf_attn"]
