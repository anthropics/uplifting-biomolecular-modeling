"""The activation rules: tree state first, pins, no contradicting switch, a GPU; the row exported; idempotent; late refused by name;
the APPLIED check against the kit's own describe(); NOT ACTIVE never silent."""
import io
import os
import sys
import types

import pytest

import rosettafold3_opt
from .. import modes, stack, tree
from . import _stubs


@pytest.fixture
def stock_tree(tmp_path, monkeypatch):
    root = _stubs.make_tree(str(tmp_path / "sp"), "stock")
    monkeypatch.syspath_prepend(root)
    _stubs.make_dist(root, route="vcs")
    import importlib
    importlib.invalidate_caches()
    monkeypatch.setattr(stack, "gpu_info", lambda: dict(_stubs.FAKE_GPU))
    return root


def _stderr(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stderr", buf)
    return buf


def test_exact_activates_on_a_patched_tree_and_exports_the_row(patched_tree, monkeypatch):
    buf = _stderr(monkeypatch)
    rep = rosettafold3_opt.enable("exact")
    assert rep["active"] and rep["mode"] == "exact" and rep["tree_state"] == "patched" and rep["applied"] == "deferred"
    assert os.environ["RF3_CUDAGRAPH"] == "1" and os.environ["RF3_HOIST"] == "1"
    assert "RF3_CUDAGRAPH_WARMUP" not in os.environ and "RF3_GRAPH_SAFE_OPS" not in os.environ
    lines = buf.getvalue().strip().splitlines()
    line = next(ln for ln in lines if ln.startswith("[rosettafold3-opt] ACTIVE "))                    # the ACTIVE line, then the KERNELS and MEM lines
    assert lines[-1] == "[rosettafold3-opt] MEM policy=capped seams=rf3.graph_flags.hoist_begin,rf3.graph_flags.hoist_end alloc_conf=expandable_segments:True,garbage_collection_threshold:0.5 fpf_tg_max=1"
    assert rep["mem"] == {"policy": "capped", "trigger": "rf3.graph_flags", "seams": ["hoist_begin", "hoist_end"], "applied": "deferred", "alloc_conf": "expandable_segments:True,garbage_collection_threshold:0.5", "fpf_tg_max": "1"}
    assert rep["fpf"]["arm"] == EXACT_ARM and rep["fpf"]["applied"] == "deferred"                  # exact carries the graph arm on stock kernels
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True,garbage_collection_threshold:0.5"                    # exported for the kit process at activation
    assert line.startswith("[rosettafold3-opt] ACTIVE mode=exact row=RF3_CUDAGRAPH=1,RF3_HOIST=1 fpf=tg+sapb+xatt+xmul.eager+xln+smsa@L1.warm tree=patched(5/5) foundry=0.2.1.dev13+g4010e3e2e@4010e3e2 ")
    assert "gpu=NVIDIA H100 80GB HBM3(sm90) levers=graph,graph_safe_ops,hoist,fpf_tg,fpf_sapb,fpf_xatt,fpf_xmul,fpf_xln,fpf_smsa,warm,xtr,confhoist,hostlean,prefetch,awrite applied=deferred" in line
    assert rosettafold3_opt.status()["active"]
    # idempotent; a different mode is refused
    assert rosettafold3_opt.enable("exact")["active"]
    with pytest.raises(rosettafold3_opt.ActivationError, match="already active"):
        rosettafold3_opt.enable("fast")


def test_exact_refused_on_a_stock_tree(stock_tree, monkeypatch):
    buf = _stderr(monkeypatch)
    rep = rosettafold3_opt.enable("exact")
    assert not rep["active"] and "not patched" in rep["reason"] and rep["tree_state"] == "stock"
    assert "[rosettafold3-opt] NOT ACTIVE: this interpreter's tree is stock(5/5), not patched" in buf.getvalue()
    assert "RF3_CUDAGRAPH" not in os.environ
    with pytest.raises(rosettafold3_opt.ActivationError):
        rosettafold3_opt.enable("exact", strict=True)


def test_tree_gate_reads_the_rows_tree_state_column(stock_tree, monkeypatch):
    """The state a row needs is its tree_state column (modes.py), not a literal in the gate: flipping the column moves the gate."""
    import dataclasses
    from .. import modes
    monkeypatch.setitem(modes.KIT_MODES, "exact", dataclasses.replace(modes.KIT_MODES["exact"], tree_state="stock"))
    rep = stack.activate("exact", dry_run=True)
    assert rep["tree_state"] == "stock" and rep["reason"] is None and rep["applied"] == "dry-run"
    monkeypatch.setitem(modes.KIT_MODES, "exact", dataclasses.replace(modes.KIT_MODES["exact"], tree_state="patched"))
    rep = stack.activate("exact", dry_run=True)
    assert rep["reason"] and "not patched: row exact runs only on the patched interpreter" in rep["reason"]
    assert tree.STATES == ("stock", "patched")


def test_target_gpu_is_recorded_and_a_mismatch_noted(patched_tree, monkeypatch):
    monkeypatch.setenv("MODEL_OPT_TARGET_GPU", "H100")
    rep = stack.activate("exact", dry_run=True)
    assert rep["target_gpu"] == "H100" and not any("MODEL_OPT_TARGET_GPU" in n for n in rep["notes"])
    monkeypatch.setenv("MODEL_OPT_TARGET_GPU", "B200")
    rep = stack.activate("exact", dry_run=True)
    assert rep["target_gpu"] == "B200" and any(n.startswith("MODEL_OPT_TARGET_GPU=B200 but the GPU is NVIDIA H100 80GB HBM3") for n in rep["notes"])
    monkeypatch.delenv("MODEL_OPT_TARGET_GPU")
    assert stack.activate("exact", dry_run=True)["target_gpu"] is None


def test_off_is_never_an_error_and_exports_nothing(patched_tree):
    rep = rosettafold3_opt.enable("off")
    assert not rep["active"] and rep["reason"].startswith("off: nothing to activate")
    assert "RF3_CUDAGRAPH" not in os.environ and rosettafold3_opt.status()["active"] is False


def test_late_activation_refused_by_name(patched_tree, monkeypatch):
    sys.modules["rf3.graph_flags"] = types.ModuleType("rf3.graph_flags")
    with pytest.raises(stack.LateActivationError, match="already imported"):
        rosettafold3_opt.enable("exact", strict=True)
    assert "RF3_CUDAGRAPH" not in os.environ
    sys.modules.pop("rf3.graph_flags")
    sys.modules["rf3.diffusion_samplers.inference_sampler"] = types.ModuleType("x")
    rep = rosettafold3_opt.enable("exact")
    assert not rep["active"] and "rf3.diffusion_samplers.inference_sampler is already imported" in rep["reason"]


def test_contradicting_switch_refused(patched_tree, monkeypatch):
    monkeypatch.setenv("RF3_HOIST", "0")
    rep = rosettafold3_opt.enable("exact")
    assert not rep["active"] and "contradicts row exact" in rep["reason"] and "RF3_HOIST='0'" in rep["reason"]
    monkeypatch.setenv("RF3_HOIST", "1")                       # an agreeing value is fine
    rep = rosettafold3_opt.enable("exact")
    assert rep["active"]


def test_no_gpu_refused_but_dry_run_reports(patched_tree, monkeypatch):
    monkeypatch.setattr(stack, "gpu_info", lambda: None)
    rep = rosettafold3_opt.enable("exact")
    assert not rep["active"] and "no CUDA GPU visible" in rep["reason"]
    rep = stack.activate("exact", dry_run=True)
    assert rep["reason"] is None and rep["applied"] == "dry-run" and any("no CUDA GPU visible here" in n for n in rep["notes"])
    assert "RF3_CUDAGRAPH" not in os.environ and stack.STATE["report"] is None


def test_pin_failure_refused(patched_tree, monkeypatch):
    monkeypatch.setattr(stack, "pin_check", lambda python=None: {"ok": False, "detail": "commit differs"})
    rep = rosettafold3_opt.enable("exact")
    assert not rep["active"] and "stock pin check failed: commit differs" in rep["reason"]


def test_missing_kit_refused(patched_tree, monkeypatch):
    monkeypatch.setattr(stack, "kit_home", lambda: str(patched_tree))   # a directory without the kit files
    rep = rosettafold3_opt.enable("exact")
    assert not rep["active"] and "kit files missing" in rep["reason"]


def test_applied_line_from_the_kits_describe(patched_tree, monkeypatch):
    buf = _stderr(monkeypatch)
    rep = rosettafold3_opt.enable("exact")
    assert rep["active"]
    _stubs.graph_flags_stub(patched_tree)                     # the watch reads describe() when rf3.graph_flags is imported
    import importlib
    importlib.invalidate_caches()
    gf = importlib.import_module("rf3.graph_flags")
    assert gf.describe()["RF3_CUDAGRAPH"] == "1" and gf.describe()["RF3_HOIST"] is True
    st = rosettafold3_opt.status()
    assert st["applied"] == "configured" and st["levers_applied"] == ["graph", "graph_safe_ops", "hoist"]
    assert "[rosettafold3-opt] APPLIED rf3.graph_flags RF3_CUDAGRAPH=1 RF3_GRAPH_SAFE_OPS=True" in buf.getvalue()


def test_applied_mismatch_stops(patched_tree, monkeypatch):
    rosettafold3_opt.enable("exact")
    _stubs.graph_flags_stub(patched_tree)
    os.environ["RF3_HOIST"] = "0"                             # something changed the row after activation
    import importlib
    importlib.invalidate_caches()
    with pytest.raises(rosettafold3_opt.ActivationError, match="different row"):
        importlib.import_module("rf3.graph_flags")
    assert rosettafold3_opt.status()["applied"] == "mismatch"


# ------------------------------------------------------------------------------------------------------ the fast mode (FPF arm)
FPF_ARM = "fast.fast+gflash+ttr+apb.fast+res+tg+xmul.eager+xln+msa@L1.warm"
FPF_LEVERS = ["fpf_trimul", "fpf_gflash", "fpf_ttr", "fpf_apb", "fpf_res"]   # + fpf_tg, then the provider-row components (fpf_xmul, fpf_xln) and warm, in arm order
EXACT_ARM = "tg+sapb+xatt+xmul.eager+xln+smsa@L1.warm"
EXACT_FPF_LEVERS = ["fpf_tg", "fpf_sapb"]
EXACT_KIT_LEVERS = ["xtr"]


@pytest.fixture
def fpf_stub(tmp_path, monkeypatch):
    root = _stubs.fpf_stub(str(tmp_path / "fpf"))
    monkeypatch.setattr(stack, "fpf_home", lambda: root)
    _stubs.dtk_stub(monkeypatch)
    _stubs.mkdit_stub(monkeypatch)
    _stubs.tgb_stub(monkeypatch)
    _stubs.xtr_stub(monkeypatch)
    for m in ("fpf_rf3_adapter", "rf3.model.RF3_structure", "rf3.model", "rf3.graph_flags", "rf3"):
        sys.modules.pop(m, None)
    yield root
    for m in ("fpf_rf3_adapter", "rf3.model.RF3_structure", "rf3.model", "rf3.graph_flags", "rf3"):
        sys.modules.pop(m, None)
    d = os.path.join(root, "rf3fpf")
    while d in sys.path:
        sys.path.remove(d)
    for f in list(sys.meta_path):
        if (isinstance(f, stack._ImportWatch) and f.trigger == stack.FPF_TRIGGER):
            sys.meta_path.remove(f)


def _import_after_activation(patched_tree):
    _stubs.graph_flags_stub(patched_tree)
    _stubs.rf3_structure_stub(patched_tree)
    import importlib
    importlib.invalidate_caches()
    importlib.import_module("rf3.graph_flags")
    return importlib.import_module("rf3.model.RF3_structure")


def test_fast_activates_exports_the_row_and_applies_the_arm_after_rf3_structure(patched_tree, fpf_stub, monkeypatch):
    buf = _stderr(monkeypatch)
    rep = rosettafold3_opt.enable("fast")
    assert rep["active"] and rep["mode"] == "fast" and rep["applied"] == "deferred" and rep["tree_state"] == "patched"
    assert os.environ["RF3_CUDAGRAPH"] == "1" and os.environ["RF3_HOIST"] == "1"           # the exact row, exported the same way
    from opt_core import kernels as core_kernels
    assert {k: v for k, v in os.environ.items() if k.startswith("FPF_RF3_")} == {"FPF_RF3_TG_MAX": "1"}   # the one add-on KNOB: the trunk-graph cache size (fast carries tg under its token budget)
    assert os.environ["PF_LNL_TILES"] == os.path.join(core_kernels.KERNELS_DIR, "lnl_fused.tiles_by_arch.json")   # the routed lnl_fused kernel's shipped tile table (the core's carried copy)
    assert os.environ["FPF_TRIMUL_V4_CELLS"] == stack.CORE_TRIMUL_TABLE == os.path.join(core_kernels.KERNELS_DIR, "fpf_trimul_v4", "table.json")   # no kit cell table: the cells evidence the kernel asks for IS the core's one table
    kernels = dict(rep["fpf"]["kernels"])                                                        # the live report keeps it: the dtk lever reads its routed core copy at apply time
    assert list(stack.KERNELS) == ["fpf_trimul_v4", "flash_triattn", "lnl_fused"]
    assert list(kernels) == list(stack.kernels_of(["mkdit"])) == list(stack.KERNELS)                     # the arm's three kernel modules, routed to the core; the mkdit lever's kernels are carried in the kit, not routed
    for name, k in kernels.items():                                                                 # routed by name, held before any import
        assert k["ok"] and k["reason"] is None and k["routed"] and not k["already_imported"], (name, k)
        assert k["resolved"] == k["core_copy"] == core_kernels.carried_path(name)
    assert kernels["fpf_trimul_v4"]["runtime_imports"] == {}                                                 # fpf_trimul_v4 imports no other carried kernel at run time
    assert core_kernels.routed() == sorted(stack.kernels_of(["mkdit"]))
    assert {k: v for k, v in rep["fpf"].items() if k != "kernels"} == {"arm": FPF_ARM, "components": ["fast.fast", "gflash", "ttr", "apb.fast", "res", "tg", "xmul.eager", "xln", "msa"], "lever_state": True, "home": fpf_stub,
                          "sys_path": [os.path.join(fpf_stub, "rf3fpf")],                                     # the adapter's directory; the kernels are routed names
                          "knobs": {"FPF_RF3_TG_MAX": "6"},
                          "trigger": "rf3.model.RF3_structure", "applied": "deferred", "cfg": None, "reason": None}
    assert rep["levers"] == ["graph", "graph_safe_ops", "hoist"] + FPF_LEVERS + ["fpf_tg", "fpf_xmul", "fpf_xln", "fpf_msa", "warm", "mkdit", "confhoist", "confln", "hostlean", "prefetch", "awrite"] and rep["levers_off"] == ["fpf_sapb", "fpf_dattn", "fpf_xatt", "fpf_smsa", "dtk", "xtr"]
    lines = buf.getvalue().strip().splitlines()
    line = lines[-3]                                                                                # the ACTIVE line, then the KERNELS line, then the MEM line
    assert lines[-1].startswith("[rosettafold3-opt] MEM policy=capped ") and lines[-1].endswith(" fpf_tg_max=1")      # fast carries the trunk graph under its token budget: one graph held
    assert line.startswith(f"[rosettafold3-opt] ACTIVE mode=fast row=RF3_CUDAGRAPH=1,RF3_HOIST=1 fpf={FPF_ARM} tree=patched(5/5) ")
    assert "levers=graph,graph_safe_ops,hoist,fpf_trimul,fpf_gflash,fpf_ttr,fpf_apb,fpf_res,fpf_tg,fpf_xmul,fpf_xln,fpf_msa,warm,mkdit,confhoist,confln,hostlean,prefetch,awrite applied=deferred" in line
    routed = stack.kernels_of(["mkdit"])                                                   # the arm's kernels + the dtk lever's, in route order
    assert lines[-2] == "[rosettafold3-opt] KERNELS routed=" + ",".join(routed) + " " + " ".join(f"{n}={core_kernels.carried_path(n)}" for n in routed)
    assert "fpf_rf3_adapter" not in sys.modules                                                 # nothing of the add-on imported yet
    assert any((isinstance(f, stack._ImportWatch) and f.trigger == stack.FPF_TRIGGER) for f in sys.meta_path)
    m = _import_after_activation(patched_tree)
    assert m.STUB is True
    adp = sys.modules["fpf_rf3_adapter"]
    assert adp.CALLS == [FPF_ARM] and adp.__file__.startswith(fpf_stub)
    assert sys.path[0] == os.path.join(fpf_stub, "rf3fpf") and [p for p in sys.path if p.startswith(fpf_stub)] == [sys.path[0]]   # the adapter's directory only
    assert rosettafold3_opt.status()["fpf"]["kernels_imported"] == {n: None for n in stack.kernels_of(["mkdit"])}   # the stub adapter (and the mkdit stand-in) import no kernel
    st = rosettafold3_opt.status()
    assert st["applied"] == "configured" and st["fpf"]["applied"] == "configured" and st["fpf"]["reason"] is None
    assert st["fpf"]["cfg"] == {"triattn": "gflash", "transition": "triton", "apb": "triton", "levers": True, "warm": True, "trunk_graph": True, "dattn": False,
                                "res": True, "msa": "card", "msa_units": "['opm', 'pwa']", "msa_aside": "{}", "smsa": False, "apb_word": "fast", "xmul": "eager", "xln": "exact", "trimul": "fast"}
    assert st["levers_applied"] == ["graph", "graph_safe_ops", "hoist"] + FPF_LEVERS + ["fpf_tg", "fpf_xmul", "fpf_xln", "fpf_msa", "mkdit"] and st["mkdit"]["on"] is True
    gf = sys.modules["rf3.graph_flags"]
    assert gf.CUDAGRAPH_MODE == "1" and gf.HOIST is True and gf.GRAPH_SAFE_OPS is True        # @L1: the adapter's set_kit_levers on the kit module
    out = buf.getvalue()
    assert "[rosettafold3-opt] APPLIED rf3.graph_flags RF3_CUDAGRAPH=1" in out
    assert " levers=graph,graph_safe_ops,hoist\n" in out                                        # the kit's APPLIED line: the switch levers only
    assert (f"[rosettafold3-opt] FPF APPLIED arm={FPF_ARM} triattn=gflash transition=triton apb=triton levers=True warm=True trunk_graph=True dattn=False "
            f"res=True msa=card msa_units=['opm', 'pwa'] msa_aside={{}} smsa=False apb_word=fast xmul=eager xln=exact trimul=fast fpf_levers=fpf_trimul,fpf_gflash,fpf_ttr,fpf_apb,fpf_res,fpf_tg,fpf_xmul,fpf_xln") in out
    assert not any((isinstance(f, stack._ImportWatch) and f.trigger == stack.FPF_TRIGGER) for f in sys.meta_path)                     # fired once, removed itself
    # the exit tally: the adapter's own counters, every FPF lever of the row named as acted or silent
    from .. import report
    t = report.tally(st)
    assert t["fpf"]["ok"] is False and "silent lever(s): fpf_trimul,fpf_gflash,fpf_ttr,fpf_apb,fpf_res" in t["fpf"]["reason"]
    adp.fold()
    t = report.tally(st)
    assert t["fpf"]["ok"] is True and t["fpf"]["served"] == 96 and t["fpf"]["tg_replays"] == 10 and t["fpf"]["levers_acted"] == FPF_LEVERS + ["fpf_tg", "fpf_xmul", "fpf_xln", "fpf_msa"]   # the trunk graph replayed (the stub's fold: 10 replays)
    assert t["fpf"]["levers_silent"] == [] and t["fpf"]["errors"] == 0
    ln = report.tally_line(t)
    assert (f" fpf={FPF_ARM} served=96 fallback={{}} errors=0 tg_replays=10 tg_fallbacks={{}} kernel_errors={{}} levers_acted={','.join(FPF_LEVERS + ['fpf_tg', 'fpf_xmul', 'fpf_xln', 'fpf_msa'])} "
            f"levers_silent=none ok=True") in ln


def test_fast_refused_without_the_add_on_files(patched_tree, fpf_stub, monkeypatch):
    os.remove(os.path.join(fpf_stub, "rf3fpf", "fpf_rf3_adapter.py"))
    rep = rosettafold3_opt.enable("fast")
    assert not rep["active"] and "FPF add-on files missing" in rep["reason"] and "rf3fpf/fpf_rf3_adapter.py" in rep["reason"]
    assert "RF3_CUDAGRAPH" not in os.environ


def test_fast_refused_when_the_adapter_is_already_imported(patched_tree, fpf_stub, monkeypatch):
    sys.modules["fpf_rf3_adapter"] = types.ModuleType("fpf_rf3_adapter")
    with pytest.raises(stack.LateActivationError, match="fpf_rf3_adapter is already imported"):
        rosettafold3_opt.enable("fast", strict=True)
    assert "RF3_CUDAGRAPH" not in os.environ
    assert stack.activate("exact", dry_run=True)["reason"] is None


def test_fpf_apply_failure_stops_the_process(patched_tree, fpf_stub, monkeypatch):
    buf = _stderr(monkeypatch)
    monkeypatch.setenv("FPF_STUB_FAIL", "raise")
    assert rosettafold3_opt.enable("fast")["active"]
    with pytest.raises(rosettafold3_opt.ActivationError, match="apply_arm.*failed: RuntimeError: stub adapter refused"):
        _import_after_activation(patched_tree)
    st = rosettafold3_opt.status()
    assert st["active"] is False and st["fpf"]["applied"] == "error" and "NOT ACTIVE: FPF apply_arm" in buf.getvalue()


def test_fpf_configuration_mismatch_stops_the_process(patched_tree, fpf_stub, monkeypatch):
    monkeypatch.setenv("FPF_STUB_FAIL", "mismatch")
    assert rosettafold3_opt.enable("fast")["active"]
    with pytest.raises(rosettafold3_opt.ActivationError, match="not the arm's: transition: adapter='stock' arm='triton'"):
        _import_after_activation(patched_tree)
    assert rosettafold3_opt.status()["fpf"]["applied"] == "mismatch"


def test_fpf_knob_in_the_environment_is_noted_not_exported(patched_tree, fpf_stub, monkeypatch):
    monkeypatch.setenv("FPF_RF3_TG_MAX", "8")
    rep = rosettafold3_opt.enable("fast")
    assert rep["active"] and rep["fpf_knobs_set"] == {"FPF_RF3_TG_MAX": "8"} and any("FPF knobs set" in n for n in rep["notes"])


def test_fast_survives_a_find_spec_probe_before_the_import(patched_tree, fpf_stub, monkeypatch):
    """A spec lookup without an import (the usual 'is it installed?' probe) leaves both watches armed for the real import."""
    import importlib, importlib.util
    _stderr(monkeypatch)
    assert rosettafold3_opt.enable("fast")["active"]
    _stubs.graph_flags_stub(patched_tree)
    _stubs.rf3_structure_stub(patched_tree)
    importlib.invalidate_caches()
    assert importlib.util.find_spec("rf3.graph_flags") is not None and importlib.util.find_spec("rf3.model.RF3_structure") is not None
    assert "rf3.graph_flags" not in sys.modules and "fpf_rf3_adapter" not in sys.modules
    assert {"rf3.graph_flags", "rf3.model.RF3_structure", "rf3.utils.inference"} <= {f.trigger for f in sys.meta_path if isinstance(f, stack._ImportWatch)}   # still armed: the apply watch, the FPF watch, the template gate's watch (+ the package levers' own)
    importlib.import_module("rf3.graph_flags")
    importlib.import_module("rf3.model.RF3_structure")
    st = rosettafold3_opt.status()
    assert st["applied"] == "configured" and st["fpf"]["applied"] == "configured" and sys.modules["fpf_rf3_adapter"].CALLS == [FPF_ARM]
    assert [f.trigger for f in sys.meta_path if isinstance(f, stack._ImportWatch)] == ["rf3.utils.inference"]   # the template gate's watch stays armed until its module imports


@pytest.mark.parametrize("order", ["watch_first", "probe_site_first"])
def test_watch_fires_once_when_a_probe_site_finder_hooks_the_same_module(tmp_path, monkeypatch, order):
    """The fold process's arrangement on ``rf3.model.RF3``: the kit's ``_ImportWatch`` (big's levers, the row-sharding install chained
    at P > 1) AND an attached ``sitecustomize``'s post-import hook (a model site's ``_After``: resolves the module through
    ``importlib.util.find_spec``, which re-walks ``sys.meta_path``) both armed for one module. The module executes once and EVERY callback
    fires once — in either finder order (the kit's watch is inserted ahead of the probe site's finder at activation) — and the watch is gone."""
    import importlib, importlib.abc, importlib.util
    name = "rf3opt_watch_probe_mod"
    (tmp_path / f"{name}.py").write_text("import itertools\nEXECUTED = next(COUNTER)\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    import builtins
    counter = iter(range(100))
    calls = {"apply": 0, "install": 0, "site_hook": 0}
    rep = {}

    class ProbeSiteFinder(importlib.abc.MetaPathFinder):       # the model site's _After pattern: resolve via importlib.util.find_spec (busy-guarded), wrap the loader,
        busy = False                                          # run the hook after the body, hook each name once
        hooked = False

        def find_spec(self, fullname, path=None, target=None):
            if fullname != name or self.busy or self.hooked:
                return None
            ProbeSiteFinder.busy = True
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                ProbeSiteFinder.busy = False
            if spec is None or spec.loader is None:
                return None
            inner = spec.loader

            class _Loader(importlib.abc.Loader):
                def create_module(self, spec_):
                    return None

                def exec_module(self, module):
                    module.COUNTER = counter
                    inner.exec_module(module)
                    ProbeSiteFinder.hooked = True
                    calls["site_hook"] += 1
            spec.loader = _Loader()
            return spec

    site = ProbeSiteFinder()

    def arm_watch():
        stack._install_watch(name, lambda m: calls.__setitem__("apply", calls["apply"] + 1), rep)       # where big.apply hooks in
        stack._install_watch(name, lambda m: calls.__setitem__("install", calls["install"] + 1), rep)   # where rowpair.install hooks in (chained)

    sys.modules.pop(name, None)
    try:
        if order == "watch_first":                            # meta_path = [watch, site, ...]: the arrangement that wrapped the loader twice
            sys.meta_path.insert(0, site)
            arm_watch()
        else:                                                 # meta_path = [site, watch, ...]
            arm_watch()
            sys.meta_path.insert(0, site)
        watches = [f for f in sys.meta_path if isinstance(f, stack._ImportWatch) and f.trigger == name]
        assert len(watches) == 1 and len(watches[0].callbacks) == 2
        if order == "probe_site_first":
            assert sys.meta_path.index(site) < sys.meta_path.index(watches[0])
        else:
            assert sys.meta_path.index(watches[0]) < sys.meta_path.index(site)
        importlib.invalidate_caches()
        mod = importlib.import_module(name)
        assert mod.EXECUTED == 0 and next(counter) == 1                                     # the module body ran ONCE
        assert calls == {"apply": 1, "install": 1, "site_hook": 1}, calls                   # every callback ONCE (apply-count == 1, install-count == 1)
        assert not any(isinstance(f, stack._ImportWatch) and f.trigger == name for f in sys.meta_path) and watches[0].fired
        assert rep.get("watch_nested") is None                                               # the nested walk was not answered: the loader was wrapped once
        importlib.reload(mod)                                                                # a later re-execution: the watch is gone, nothing fires again
        assert calls == {"apply": 1, "install": 1, "site_hook": 1}
    finally:
        sys.modules.pop(name, None)
        for f in list(sys.meta_path):
            if f is site or (isinstance(f, stack._ImportWatch) and f.trigger == name):
                sys.meta_path.remove(f)


def test_fpf_lever_step_that_did_not_reach_the_kit_is_a_mismatch(patched_tree, fpf_stub, monkeypatch):
    """@L1 whose set_kit_levers found no kit module: the adapter records levers='unavailable'; the check names it."""
    monkeypatch.setenv("FPF_STUB_FAIL", "levers")
    _stderr(monkeypatch)
    assert rosettafold3_opt.enable("fast")["active"]
    with pytest.raises(rosettafold3_opt.ActivationError, match="levers: adapter='unavailable' arm=@L1"):
        _import_after_activation(patched_tree)
    assert rosettafold3_opt.status()["fpf"]["applied"] == "mismatch"


def test_fpf_tally_gates_the_adapters_own_degradations(patched_tree, fpf_stub, monkeypatch):
    """On a row that carries the trunk graph (exact: tg+sapb): the adapter's own degradations gate the tally."""
    from .. import report
    monkeypatch.setenv("FPF_STUB_TG_FALLBACKS", "capture_failed=1,evictions=2")
    monkeypatch.setenv("FPF_STUB_KERNEL_ERROR", "1")
    _stderr(monkeypatch)
    assert rosettafold3_opt.enable("exact")["active"]
    assert os.environ["FPF_RF3_TG_MAX"] == "1"                                                          # the row carries tg: the default policy budgets its cache
    _import_after_activation(patched_tree)
    sys.modules["fpf_rf3_adapter"].fold()
    t = report.tally(rosettafold3_opt.status())["fpf"]
    assert t["tg_fallbacks"] == {"capture_failed": 1, "evictions": 2} and list(t["kernel_errors"]) == ["triattn:kernel-error->contiguous:RuntimeError('stub')"]
    assert t["ok"] is False and t["reason"] == ("trunk-graph fallback(s): capture_failed=1; kernel error(s): triattn:kernel-error->contiguous:RuntimeError('stub')=1")
    assert 'tg_fallbacks={"capture_failed": 1, "evictions": 2}' in report.tally_line(report.tally(rosettafold3_opt.status()))


def test_fpf_cfg_mismatch_grammar():
    cfg = {"triattn": "gflash", "transition": "triton", "apb": "triton", "trunk_graph": True, "dattn": False, "res": True}
    assert stack.fpf_cfg_mismatch(["fast", "gflash", "ttr", "apb", "tg", "res"], cfg, trimul_mode="fast") == []
    assert stack.fpf_cfg_mismatch(["fast", "gflash", "ttr", "apb", "tg", "res"], dict(cfg, apb="safe"), trimul_mode="fast") == ["apb: adapter='safe' arm='triton'"]
    tg = {"triattn": "stock", "transition": "stock", "apb": "stock", "trunk_graph": True, "dattn": False, "res": False}
    assert stack.fpf_cfg_mismatch(["fast", "tg"], dict(tg, levers=True), trimul_mode="fast", lever_state=True) == []
    assert stack.fpf_cfg_mismatch(["fast", "tg"], dict(tg, levers="unavailable"), trimul_mode="fast", lever_state=True) == ["levers: adapter='unavailable' arm=@L1 (True)"]
    assert stack.fpf_cfg_mismatch(["fast", "tg"], tg, trimul_mode="fast", lever_state=False) == ["levers: adapter=None arm=@L0 (False)"]
    assert stack.fpf_cfg_mismatch(["tg", "sapb"], {"apb": "safe", "trunk_graph": True}, trimul_mode="stock") == []
    assert stack.fpf_cfg_mismatch(["fast", "tg"], {"trunk_graph": False}, trimul_mode="stock") == ["trimul: adapter='stock' arm='fast'", "trunk_graph: adapter=False arm=True"]
    assert stack.fpf_cfg_mismatch(["fast", "dattn"], {}, trimul_mode="fast") == ["dattn: absent from the adapter's cfg, arm wants True"]


def test_expected_describe_grammar():
    assert stack.expected_describe({"RF3_CUDAGRAPH": "1", "RF3_HOIST": "1"}) == {"RF3_CUDAGRAPH": "1", "RF3_HOIST": True}
    assert stack.expected_describe({"RF3_CUDAGRAPH": "0", "RF3_HOIST": "1"}) == {"RF3_CUDAGRAPH": "0", "RF3_HOIST": True}
    assert stack.expected_describe({"RF3_CUDAGRAPH": "replay"}) == {"RF3_CUDAGRAPH": "replay"}


def test_activation_refuses_an_undeclared_env_name(patched_tree, monkeypatch):
    """A mistyped ROSETTAFOLD3_OPT_* name (ROSETTAFOLD3_OPT_MODE for the mode variable) is a named refusal, never ignored."""
    buf = _stderr(monkeypatch)
    monkeypatch.setenv("ROSETTAFOLD3_OPT_MODE", "exact")
    with pytest.raises(rosettafold3_opt.ActivationError, match="undeclared ROSETTAFOLD3_OPT_\\* / ROSETTAFOLD3_BIG_\\* names in the environment"):
        rosettafold3_opt.enable("exact", strict=True)
    assert "[rosettafold3-opt] NOT ACTIVE: undeclared ROSETTAFOLD3_OPT_* / ROSETTAFOLD3_BIG_* names in the environment (mistyped? this tree reads ROSETTAFOLD3_OPT_STOCK_PYTHON, " in buf.getvalue()
    assert "ROSETTAFOLD3_OPT_MODE='exact'" in buf.getvalue()


def test_two_watches_on_one_trigger_are_chained_in_order():
    """big's levers and the row-sharding adapter both fire on rf3.model.RF3: the second _install_watch for a trigger chains after the
    first (never replaces it) and both run once, in installation order, when the module executes."""
    import importlib, sys as _sys, types
    fired = []
    rep = {}
    for f in [f for f in _sys.meta_path if isinstance(f, stack._ImportWatch) and f.trigger == "rf3_watch_probe"]:
        _sys.meta_path.remove(f)
    stack._install_watch("rf3_watch_probe", lambda m: fired.append(("big", m.__name__)), rep)
    stack._install_watch("rf3_watch_probe", lambda m: fired.append(("rowpair", m.__name__)), rep)
    assert sum(isinstance(f, stack._ImportWatch) and f.trigger == "rf3_watch_probe" for f in _sys.meta_path) == 1
    import tempfile, os
    d = tempfile.mkdtemp(); open(os.path.join(d, "rf3_watch_probe.py"), "w").write("X = 1\n")
    _sys.path.insert(0, d)
    try:
        importlib.invalidate_caches()
        importlib.import_module("rf3_watch_probe")
    finally:
        _sys.path.remove(d); _sys.modules.pop("rf3_watch_probe", None)
    assert fired == [("big", "rf3_watch_probe"), ("rowpair", "rf3_watch_probe")]
    assert not any(isinstance(f, stack._ImportWatch) and f.trigger == "rf3_watch_probe" for f in _sys.meta_path)   # disarmed after firing
