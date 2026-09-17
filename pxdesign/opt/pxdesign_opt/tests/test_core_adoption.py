"""The kit runs on the shared core (common/opt_core): the pin names the core installed — the core pin gate's facts — the build backend and the
gate are the core's kit_template files byte-for-byte, the shipped .pth is the backend's generated text, a core that is not the pinned one is
refused by name with exit status 3, no core or kit module is copied into the package; `rows` is refused by name on the activation route (not a mode: select `big`); `big` exports its switch and applies the package's size levers through the hook."""
import hashlib
import os
import sys

import pytest

from . import _stubs

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.abspath(os.path.join(HERE, "..", ".."))                      # pxdesign/opt
PYPROJECT = os.path.join(OPT, "pyproject.toml")


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def test_pin_names_the_core_installed():
    """[tool.opt_core] = the core's path and MINIMUM version (a floor), and the core this interpreter imports satisfies that floor: the gate of
    record (`pxdesign_opt.core_gate` = `_core_gate.gate(__file__, tag=TAG)`) passes and its facts name both sides; the core's own live-tree
    comparison (`opt_core.gates.core_pin_check`, the tests' and tools' cross-check) agrees."""
    import pxdesign_opt
    from pxdesign_opt._core_gate import version_tuple
    facts = pxdesign_opt.core_gate()
    assert facts["tag"] == "pxdesign-opt" == pxdesign_opt.TAG
    pinned, installed = facts["pinned"], facts["installed"]
    assert pinned["path"] == "../../common/opt_core" and os.path.samefile(pinned["pyproject"], PYPROJECT)
    assert version_tuple(installed["version"]) >= version_tuple(pinned["version"]), facts   # floor: installed >= pinned, not necessarily equal
    import opt_core
    assert os.path.samefile(installed["package_dir"], os.path.dirname(opt_core.__file__)) and installed["version"] == opt_core.__version__
    from opt_core import gates
    g = gates.core_pin_check(PYPROJECT)
    assert g.ok, g.reason


def test_build_backend_and_gate_are_the_core_templates():
    """opt/_build_backend.py and pxdesign_opt/_core_gate.py are the installed core's kit_template files byte-for-byte (located through the
    gate's own facts, not a relative guess), and the gate is carried exactly once, inside the package."""
    import glob
    import pxdesign_opt
    root = pxdesign_opt.core_gate()["installed"]["root"]
    assert _sha(os.path.join(OPT, "_build_backend.py")) == _sha(os.path.join(root, "kit_template", "_build_backend.py"))
    copies = sorted(glob.glob(os.path.join(OPT, "**", "_core_gate.py"), recursive=True))
    assert copies == [os.path.join(OPT, "pxdesign_opt", "_core_gate.py")], copies
    assert _sha(copies[0]) == _sha(os.path.join(root, "kit_template", "_core_gate.py"))


def test_pth_is_the_backends_generated_text():
    """opt/pxdesign_opt_autoload.pth == `python opt/_build_backend.py pxdesign_opt PXDESIGN_OPT pxdesign-opt` (the backend refuses to build a
    wheel around any other text); its fields round-trip to this kit's package, variable, tag and exit code."""
    import importlib.util
    import pxdesign_opt
    from pxdesign_opt import _autoload
    spec = importlib.util.spec_from_file_location("_pxdesign_build_backend_under_test", os.path.join(OPT, "_build_backend.py"))
    bb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bb)
    shipped = open(os.path.join(OPT, "pxdesign_opt_autoload.pth"), encoding="utf-8").read()
    assert shipped == bb.pth_text("pxdesign_opt", _autoload.ENV, pxdesign_opt.TAG)
    assert bb.pth_fields(shipped) == ("pxdesign_opt", "PXDESIGN_OPT", "pxdesign-opt", _autoload.EXIT_NOT_ACTIVE) == ("pxdesign_opt", _autoload.ENV, pxdesign_opt.TAG, 3)


def test_one_spelling_of_the_tag():
    """`[pxdesign-opt]` on every line: report.PREFIX, the autoload spec, the gate's tag and the .pth guard all read pxdesign_opt.TAG."""
    import pxdesign_opt
    from pxdesign_opt import _autoload, report
    assert pxdesign_opt.TAG == "pxdesign-opt" and report.TAG is pxdesign_opt.TAG and _autoload.TAG is pxdesign_opt.TAG
    assert report.PREFIX == "[pxdesign-opt]" and _autoload.spec().tag == "pxdesign-opt" and pxdesign_opt.core_gate()["tag"] == "pxdesign-opt"


def _installed(**over):
    from pxdesign_opt import _core_gate
    real = _core_gate.installed_core()
    assert real is not None
    return dict(real, **over)


@pytest.mark.parametrize("entry", ["core_gate", "enable", "status"])
def test_a_core_that_is_not_the_pinned_one_is_refused_by_name(monkeypatch, capsys, fresh_stack, entry):
    """An installed core whose __init__.py names an older version than the pin: `core_gate()` — and so `enable()` / `status()`, whose
    first statement it is — raise SystemExit(3) after ONE `[pxdesign-opt] NOT ACTIVE: reason=core_mismatch: …` line naming both sides; the
    activation module is never consulted (no report, no override: PXDESIGN_OPT_FORCE does not apply to the core pin)."""
    import pxdesign_opt
    from pxdesign_opt import _core_gate
    stale = _installed(version="0.2.5")                                        # older than the pin (floor semantics)
    monkeypatch.setattr(_core_gate, "installed_core", lambda: stale)
    monkeypatch.setenv("PXDESIGN_OPT_FORCE", "1")
    call = {"core_gate": pxdesign_opt.core_gate, "enable": lambda: pxdesign_opt.enable("exact"), "status": pxdesign_opt.status}[entry]
    with pytest.raises(SystemExit) as ex:
        call()
    assert ex.value.code == 3
    err = capsys.readouterr().err
    pin = _core_gate.read_table(PYPROJECT, "tool.opt_core")
    assert err.count("NOT ACTIVE") == 1 and err.strip() == (
        "[pxdesign-opt] NOT ACTIVE: reason=core_mismatch: opt_core pinned >= v%s at %s, installed v0.2.5 at %s"
        % (pin["version"], pin["path"], stale["root"])), err
    assert fresh_stack.status()["active"] is False and "has not run" in fresh_stack.status()["reason"]   # activation never ran


@pytest.mark.parametrize("case", ["absent", "unversioned"])
def test_absent_and_unversioned_cores_are_refused_by_name(monkeypatch, capsys, case):
    """ABSENT: installed_core() returns None. UNVERSIONED: installed_core() locates opt_core but its __init__.py has no __version__
    literal (version=None), displayed as `v?` in the refusal line."""
    from pxdesign_opt import _core_gate
    import pxdesign_opt
    fake = {"absent": None, "unversioned": _installed(version=None)}[case]
    monkeypatch.setattr(_core_gate, "installed_core", lambda: fake)
    with pytest.raises(SystemExit) as ex:
        pxdesign_opt.core_gate()
    assert ex.value.code == 3
    err = capsys.readouterr().err
    assert err.count("NOT ACTIVE") == 1 and err.startswith("[pxdesign-opt] NOT ACTIVE: reason="), err
    expect = {"absent": "reason=core_missing:opt_core (pinned >= v", "unversioned": "installed v? at "}[case]
    assert expect in err and ("reason=core_mismatch: opt_core pinned >= v" in err) == (case != "absent"), err



def test_activation_report_carries_the_gates_facts(fresh_stack, monkeypatch):
    """The report's core_pin block is the gate's facts (one producer): the installed core + ok + the pin; check --json and opt_manifest.json list it."""
    stack = fresh_stack
    _stubs.install_torch(); _stubs.install_upstream(); _stubs.fake_box(monkeypatch, stack)
    rep = stack.activate("exact", dry_run=True)
    cp = rep["core_pin"]
    import pxdesign_opt
    facts = pxdesign_opt.core_gate()
    assert cp["ok"] is True and cp["pinned"] == facts["pinned"] and {k: cp[k] for k in facts["installed"]} == facts["installed"]
    assert "forced" not in cp and not any("pin mismatch" in n and "opt_core" in n for n in rep["notes"])


def test_no_core_module_is_vendored():
    """The package imports the core; it carries no copy of a core module (one registry per concept)."""
    import pxdesign_opt
    pkg = os.path.join(OPT, "pxdesign_opt")
    core_pkg = pxdesign_opt.core_gate()["installed"]["package_dir"]          # the installed (pinned) core's package directory
    core_hashes = {_sha(os.path.join(core_pkg, f)) for f in os.listdir(core_pkg) if f.endswith(".py")}
    for f in os.listdir(pkg):
        if f.endswith(".py"):
            assert _sha(os.path.join(pkg, f)) not in core_hashes, f


def test_big_applies_the_hoist_and_the_size_levers_through_the_hook(fresh_stack, monkeypatch, capsys):
    """big: the hoist installed with PXD_HOIST_MODE=rows exported, then the package's three size levers and fast's two tolerance levers applied
    by the same hook on the intact base — the APPLIED line keeps its bytes and a PACKAGE line names featdiet,padmask,rowpipe,tf32,sdedup applied
    with one evidence field per patch point; a second runner in the process is classified again without a fallback (the h5 record accepts the
    padmask-seeded wrapper)."""
    from pxdesign_opt import sdedup
    sdedup.reset_for_tests()
    stack = fresh_stack
    rep, runner, err = _runner_after(stack, monkeypatch, "big", capsys)
    assert os.environ.get("PXD_HOIST_MODE") == "rows" and rep["tier"] == "2" and rep["package_levers"] == ["featdiet", "padmask", "rowpipe", "tf32", "sdedup"] and rep["precision_policy"] == "tf32"
    assert "[pxdesign-opt] APPLIED model#1 levers=h1,h2,h3,h4,h5 fallbacks=none\n" in err, err
    assert "[pxdesign-opt] PACKAGE model#1 levers=featdiet,padmask,rowpipe,tf32,sdedup applied=featdiet,padmask,rowpipe,tf32,sdedup fallbacks=none " in err, err
    assert "featdiet_predict_wrapped=True" in err and "padmask_rearrange_qk_to_dense_trunk=True" in err and "padmask_h5_seeded=True" in err and "rowpipe_prepare_cache_rebound=True" in err and "tf32_applied=True" in err and "sdedup_hook=True" in err, err
    assert "applied=deferred package_levers=featdiet,padmask,rowpipe,tf32,sdedup tier=2" in err
    st = stack.status()
    assert st["package_levers_applied"] == ["featdiet", "padmask", "rowpipe", "tf32", "sdedup"] and st["partial"] is False and st["levers_applied"] == ["h1", "h2", "h3", "h4", "h5"]
    import protenix.model.modules.primitives as PR
    from pxdesign.runner.inference import InferenceRunner
    assert getattr(PR.rearrange_qk_to_dense_trunk, "_pxdesign_opt_lever", None) == "padmask" and getattr(InferenceRunner.predict, "_pxdesign_opt_lever", None) == "featdiet"
    InferenceRunner(configs=None)                                               # a second runner: the hook classifies again
    err2 = capsys.readouterr().err
    assert "APPLIED model#2 levers=h1,h2,h3,h4,h5 fallbacks=none" in err2 and "PACKAGE model#2 levers=featdiet,padmask,rowpipe,tf32,sdedup applied=featdiet,padmask,rowpipe,tf32,sdedup " in err2, err2


def test_big_exports_the_rows_switch(fresh_stack, monkeypatch, capsys):
    stack = fresh_stack
    rep, runner, err = _runner_after(stack, monkeypatch, "big", capsys)
    assert os.environ["PXD_HOIST_MODE"] == "rows" and rep["tier"] == "2"
    assert "ACTIVE mode=big" in err and "package_levers=featdiet,padmask,rowpipe,tf32,sdedup tier=2" in err and "env=PXD_HOIST=1,PXD_HOIST_MODE=rows,PXD_HOIST_MASK=1 " in err
    assert "[pxdesign-opt] APPLIED model#1 levers=h1,h2,h3,h4,h5 fallbacks=none\n" in err


def _runner_after(stack, monkeypatch, mode, capsys):
    _stubs.install_torch(); _stubs.install_upstream(); _stubs.fake_box(monkeypatch, stack)
    rep = stack.activate(mode)
    assert rep["active"], rep.get("reason")
    from pxdesign.runner.inference import InferenceRunner
    runner = InferenceRunner(configs=None)
    return rep, runner, capsys.readouterr().err


def test_exact_applies_the_hoist_through_the_hook(fresh_stack, monkeypatch, capsys):
    """exact: the hoist is installed when load_checkpoint returns; the APPLIED line keeps its bytes (levers=h1..h5 fallbacks=none, end of line);
    no PACKAGE line (exact plans no package lever); the stock initialiser runs as upstream's (fast_init is not wired); the TF32 policy is untouched."""
    stack = fresh_stack
    rep, runner, err = _runner_after(stack, monkeypatch, "exact", capsys)
    import protenix.openfold_local.model.primitives as OP
    assert OP.calls == [1.0]                                                    # the stub's construction-time initialiser ran (nothing suppresses it)
    assert "[pxdesign-opt] APPLIED model#1 levers=h1,h2,h3,h4,h5 fallbacks=none\n" in err and "PACKAGE model#" not in err
    assert "applied=deferred package_levers=none tier=1" in err
    st = stack.status()
    assert st["package_levers_applied"] == [] and st["partial"] is False and st["levers_applied"] == ["h1", "h2", "h3", "h4", "h5"]
    import torch
    assert torch.backends.cuda.matmul.allow_tf32 is False                       # exact holds the stock numerics policy: TF32 off (only the `tf32` package lever of fast / big sets it)


def test_rows_is_not_a_mode_on_the_activation_route(fresh_stack, monkeypatch, capsys):
    """`enable("rows")` / `stack.activate("rows")` is refused by name before anything is gated, exported or armed (the table's one sentence;
    PXD_HOIST_MODE=rows is a lever value `big` exports, not a mode); the process stays activatable as exact afterwards."""
    stack = fresh_stack
    _stubs.install_torch(); _stubs.install_upstream(); _stubs.fake_box(monkeypatch, stack)
    monkeypatch.delenv("PXD_HOIST_MODE", raising=False)
    for kw in ({}, {"dry_run": True}, {"strict": True}):
        with pytest.raises(ValueError) as e:
            stack.activate("rows", **kw)
        assert str(e.value) == "unknown mode 'rows'; expected one of off|exact|fast|big", kw
    assert "PXD_HOIST_MODE" not in os.environ and stack.status()["active"] is False and capsys.readouterr().err == ""
    rep, runner, err = _runner_after(stack, monkeypatch, "exact", capsys)
    assert os.environ["PXD_HOIST_MODE"] == "shape" and "ACTIVE mode=exact" in err and "tier=1" in err


def test_stock_process_holds_no_core_module_beyond_the_proof():
    """det.py and options.py — imported by the stock caller — import nothing of the core at module level; stock_infer imports only opt_core.stock_proof."""
    import ast
    for name, allowed in (("det.py", set()), ("options.py", set()), ("infer_loop.py", set()), ("outputs.py", set()), ("stock_infer.py", {"opt_core"})):
        tree = ast.parse(open(os.path.join(OPT, "pxdesign_opt", name)).read())
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        mods = {(n.module or "").split(".")[0] if isinstance(n, ast.ImportFrom) else a.name.split(".")[0] for n in top for a in (n.names if isinstance(n, ast.Import) else [n])}
        assert not ({m for m in mods if m == "opt_core"} - allowed), (name, mods)
    src = open(os.path.join(OPT, "pxdesign_opt", "stock_infer.py")).read()
    assert "from opt_core import stock_proof" in src and "from opt_core import gates" not in src and "import opt_core.gates" not in src


def test_registry_module_prefixes_are_the_stock_callers():
    from pxdesign_opt import registry, stock_infer
    assert tuple(stock_infer.KIT_MODULE_PREFIXES) == tuple(registry.KIT_MODULE_PREFIXES)


def test_jit_cache_key_is_the_cores_rule():
    """The key string of the pinned stack under opt_core.jit_cache (configs/h100.env's TORCH_EXTENSIONS_DIR is keyed by it)."""
    from opt_core.jit_cache import key
    from pxdesign_opt.modes import jit_cache_key
    assert jit_cache_key("2.3.1+cu121", "12.1", "9.0") == key("2.3.1+cu121", "12.1", "9.0") == "torch2.3.1-cu121-sm90"


def test_a_key_part_that_cannot_be_established_is_refused_by_name(monkeypatch):
    """No compute capability (no visible GPU) and none passed: the derivation raises `StackKeyUnknown` naming the part — `configs/h100.env`
    refuses then (rc 3) instead of keying a cache directory by `unknown`; `strict=False` renders `unknown` for display, and the manifest's
    box-as-found block uses that form (a refused activation on a GPU-less box still writes its manifest)."""
    from opt_core import gates
    from opt_core.jit_cache import StackKeyUnknown
    from pxdesign_opt import manifest
    from pxdesign_opt.modes import jit_cache_key
    monkeypatch.setattr(gates, "nvidia_smi_probe", lambda *a, **k: {})
    with pytest.raises(StackKeyUnknown) as ex:
        jit_cache_key("2.3.1+cu121", "12.1", None)
    assert "cc" in str(ex.value)
    assert jit_cache_key("2.3.1+cu121", "12.1", None, strict=False) == "torch2.3.1-cu121-smunknown"
    monkeypatch.delenv("MODEL_OPT_STACK_KEY", raising=False)
    blk = manifest.stack_block({"gpu": {"torch": "2.3.1+cu121", "cuda": "12.1", "cc": None, "name": None}})
    assert blk["stack_key"] == "torch2.3.1-cu121-smunknown"
    monkeypatch.setenv("MODEL_OPT_STACK_KEY", "torch2.3.1-cu121-sm90")
    assert manifest.stack_block({"gpu": {"torch": "2.3.1+cu121", "cuda": "12.1", "cc": None}})["stack_key"] == "torch2.3.1-cu121-sm90"   # the configured key, when set, is listed as is


def test_pin_table_reads_through_the_gates_fallback_reader(monkeypatch):
    """A python without tomllib (< 3.11) reads [tool.opt_core] through the gate's own line reader: the table's header line is exactly
    `[tool.opt_core]` (no trailing text), the two pin lines carry no comment, and path / version come back equal to
    tomllib's reading and to the core's reader (`opt_core.gates.read_pin_table`)."""
    import builtins
    from pxdesign_opt import _core_gate
    lines = open(PYPROJECT, encoding="utf-8").read().splitlines()
    header = [ln for ln in lines if ln.lstrip().startswith("[tool.opt_core]")]
    assert header == ["[tool.opt_core]"], header
    i = lines.index("[tool.opt_core]")
    assert [ln.split(" = ")[0] for ln in lines[i + 1:i + 3]] == ["path", "version"] and not any("#" in ln for ln in lines[i + 1:i + 3]), lines[i:i + 3]
    real_import = builtins.__import__

    def no_tomllib(name, *a, **k):
        if name == "tomllib":
            raise ModuleNotFoundError("No module named 'tomllib'")
        return real_import(name, *a, **k)

    monkeypatch.delitem(sys.modules, "tomllib", raising=False)
    monkeypatch.setattr(builtins, "__import__", no_tomllib)
    fallback = _core_gate.read_table(PYPROJECT, "tool.opt_core")
    monkeypatch.setattr(builtins, "__import__", real_import)
    assert set(fallback) == {"path", "version"}, fallback
    assert fallback["path"] == "../../common/opt_core" and fallback["version"]
    from opt_core import gates
    assert {k: gates.read_pin_table(PYPROJECT)[k] for k in fallback} == fallback
    try:
        import tomllib  # noqa: F401
    except ModuleNotFoundError:
        return
    assert _core_gate.read_table(PYPROJECT, "tool.opt_core") == fallback


def test_fast_applies_the_hoist_rows_and_its_package_levers_through_the_hook(fresh_stack, monkeypatch, capsys):
    """fast: the hoist installed with PXD_HOIST_MODE=rows exported, then featdiet + padmask + tf32 + sdedup applied by the same hook — the
    PACKAGE line names all four applied with their evidence fields (tf32_applied=True: the policy set through opt_core.precision; sdedup_hook=True:
    the carried lever consults pxd_xattempt.fuse and the binding is the package's module); the stub torch's TF32 switches read on afterwards."""
    stack = fresh_stack
    from pxdesign_opt import sdedup
    sdedup.reset_for_tests()
    rep, runner, err = _runner_after(stack, monkeypatch, "fast", capsys)
    assert os.environ.get("PXD_HOIST_MODE") == "rows" and rep["tier"] == "2" and rep["package_levers"] == ["featdiet", "padmask", "tf32", "sdedup"] and rep["precision_policy"] == "tf32"
    assert "ACTIVE mode=fast" in err and "package_levers=featdiet,padmask,tf32,sdedup tier=2" in err and "withheld=" not in err
    assert "[pxdesign-opt] APPLIED model#1 levers=h1,h2,h3,h4,h5 fallbacks=none\n" in err, err
    assert "[pxdesign-opt] PACKAGE model#1 levers=featdiet,padmask,tf32,sdedup applied=featdiet,padmask,tf32,sdedup fallbacks=none " in err, err
    assert "tf32_policy=tf32" in err and "tf32_matmul=high" in err and "tf32_applied=True" in err and "sdedup_hook=True" in err and "sdedup_class=tolerance" in err, err
    import torch
    assert torch.backends.cuda.matmul.allow_tf32 is True and torch.backends.cudnn.allow_tf32 is True and torch.get_float32_matmul_precision() == "high"
    import pxd_xattempt
    assert pxd_xattempt.fuse is sdedup and sdedup.state()["sdedup"] is True
    st = stack.status()
    assert st["package_levers_applied"] == ["featdiet", "padmask", "tf32", "sdedup"] and st["partial"] is False
    assert stack.runtime_gate() == []                                                        # nothing has run yet: no dedup-able call, nothing demanded (an N_sample 1 run reads the same)
    D = sdedup.dedup(); D.stats_["calls"] += 3; D.stats_["single"] += 1                     # a run with dedup-able calls (N_sample > 1) that served none of them: NOT ACTIVE by name
    assert stack.runtime_gate() and any("served" in p for p in stack.runtime_gate()), stack.runtime_gate()
    D.stats_["calls"] -= 3; D.stats_["single"] -= 1
    pc = stack.precision_check()                                                             # the live switches re-read against the APPLIED plan: tf32 applied -> policy tf32, live high/on
    assert pc["precision_planned"] == "tf32" and pc["precision_ok"] == 1 and stack.precision_gate() == [], pc
    import torch as _t
    _t.set_float32_matmul_precision("highest")                                               # something reset the switch after the hook: the exit gate names it
    assert stack.precision_gate() and "lever tf32 requested" in stack.precision_gate()[0] and "--mode exact" in stack.precision_gate()[0]   # the exit gate names the lever and its one-flag escape
    _t.set_float32_matmul_precision("high")
    lines = stack.lever_lines()
    assert lines[0].startswith("[pxdesign-opt] LEVER name=tf32 state=on") and "strategy=F4.tf32_matmul" in lines[0] and "policy=tf32" in lines[0]
    assert lines[1].startswith("[pxdesign-opt] LEVER name=sdedup state=on") and "impl=row_dedup" in lines[1] and "strategy=F7.row_dedup" in lines[1] and "class=tolerance" in lines[1]
    assert len(lines) == 2                                                                   # the two tolerance-class levers' census lines: tf32, sdedup
    sdedup.reset_for_tests()


def test_a_tf32_library_override_is_reported_never_refused(fresh_stack, monkeypatch, capsys):
    """NVIDIA_TF32_OVERRIDE / TORCH_ALLOW_TF32_CUBLAS_OVERRIDE in the environment: activation proceeds on exact, fast and big alike with one note
    naming the variable (the KERNELS line and the LEVER tf32 line print the live switches); a stock-policy mode whose live switches an override moved is
    reported in the manifest's precision record (ok=0), not refused — only a REQUESTED tf32 lever that did not hold is NOT ACTIVE (precision.exit_problems)."""
    from pxdesign_opt import precision
    stack = fresh_stack
    _stubs.install_torch(); _stubs.install_upstream(); _stubs.fake_box(monkeypatch, stack)
    for var in ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"):
        monkeypatch.setenv(var, "1")
        for mode in ("exact", "fast", "big"):
            rep = stack.activate(mode)
            assert rep["active"] is True and any(f"TF32 library override in the environment ({var}=1)" in n for n in rep["notes"]), (var, mode, rep.get("notes"))
            stack.reset_for_tests()
        assert precision.override_words() == f"{var}=1"
        monkeypatch.delenv(var)
    assert precision.override_words() == "" and stack.activate("exact")["active"] is True
    assert precision.exit_problems({"precision_planned": "stock", "precision_ok": 0, "precision_live": "high/True/True", "precision_mismatch": "matmul:high!=highest"}) == []
    assert precision.exit_problems({"precision_planned": "tf32", "precision_ok": 1}) == []
    assert "--mode exact" in precision.exit_problems({"precision_planned": "tf32", "precision_ok": 0, "precision_live": "highest/False/False", "precision_mismatch": "m"})[0]


