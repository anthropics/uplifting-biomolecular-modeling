"""enable(): the activation gates, the late-activation rule, idempotence, the refusal of a second mode, and the hook route on CPU
(the hook files run without torch; their finders are installed for the line's add-ons)."""
import os
import sys

import pytest

from openfold3_opt.tests import _stubs


@pytest.fixture
def pkg():
    d = _stubs.stub_dist()
    pkg = _stubs.reset_package()
    yield pkg
    _stubs.reset_package()
    _stubs.unstub_dist(d)


def test_version_gate_refuses_an_unpinned_openfold3():
    d = _stubs.stub_dist("0.5.0")
    try:
        pkg = _stubs.reset_package()
        rep = pkg.enable("fast")
        assert not rep["active"] and "0.5.0" in rep["reason"] and "0.4.1" in rep["reason"]
        with pytest.raises(pkg.ActivationError):
            pkg.enable("exact", strict=True)
    finally:
        _stubs.reset_package()
        _stubs.unstub_dist(d)


def test_off_is_inactive_and_exports_nothing(pkg):
    rep = pkg.enable("off")
    assert rep["mode"] == "off" and not rep["active"] and rep["env"] == {} and rep["hooks"] == []
    assert not any(k.startswith("OF3") for k in os.environ)


def test_unknown_mode(pkg):
    rep = pkg.enable("faster")
    assert not rep["active"] and "unknown mode" in rep["reason"]


def test_preset_conflicting_switch_refuses(pkg):
    os.environ["OF3T_TRIATT"] = "stock"
    rep = pkg.enable("fast")
    assert not rep["active"] and "OF3T_TRIATT" in rep["reason"]


def test_late_activation_refused_once_a_hook_target_is_imported(pkg):
    _stubs.stub_module("openfold3.core.model.structure.diffusion_module")
    rep = pkg.enable("fast")
    assert not rep["active"] and "late activation" in rep["reason"] and "diffusion_module" in rep["reason"]


def test_late_activation_refused_under_the_kits_own_route(pkg):
    class _LeverFinder:                                   # the kit's finder already on sys.meta_path: the PYTHONPATH route is in use
        def find_spec(self, *a, **k):
            return None
    sys.meta_path.insert(0, _LeverFinder())
    rep = pkg.enable("exact")
    assert not rep["active"] and "already installed" in rep["reason"]


def test_fast_activates_and_installs_both_hooks(pkg, capsys):
    from openfold3_opt import hooks, modes
    rep = pkg.enable("fast")
    assert rep["active"], rep.get("reason")
    assert rep["mode"] == "fast" and rep["hooks"] == ["cells", "trunk_kernels", "fast_inference"]
    assert "OF3T_TRIMUL" not in os.environ and os.environ["OPENFOLD3_OPT_TRIMUL_TIER"] == "fast" and os.environ["OF3_FAST_INIT"] == "1" and os.environ["OPENFOLD3_OPT"] == "fast"   # the TriMul is the pair cells' through the core's provider on the line's tier word
    home = _stubs.tree_home()
    assert os.environ[modes.KIT_LEVERS_ENV] == os.path.join(home, modes.KITS["fast_inference"], "of3_levers")     # the fast line: the fast-inference kit's levers
    inst = hooks.installed()
    assert {h["kit"] for h in inst} == {"cells", "trunk_kernels", "fast_inference"}                            # the package\'s cells hook + the two add-on hooks
    assert {h["cls"] for h in inst} == {"_PostImportFinder", "_Finder", "_LeverFinder"}
    assert sys.path[:2] == [os.path.join(home, modes.CELLS_HOOK_DIR), os.path.join(home, modes.KITS["trunk_kernels"], "of3t_hook")]   # the fast line: the cells hook first, then the trunk-kernels hook
    assert rep["hooks_spelling"] == "opt/openfold3_opt/hooks/cells>of3t_hook>of3_levers"
    err = capsys.readouterr().err
    assert "[openfold3-opt] ACTIVE mode=fast" in err
    assert "levers_requested=fast_init,cuda_graphs,graphs_strict,templ_distinct,paircache," in err
    assert "levers=" not in err.replace("levers_requested=", "")                              # the line claims nothing applied
    assert rep["levers_applied"] == [] and rep["levers_pending"] == rep["levers_requested"] and rep["arm_complete"] is None
    assert rep["carrier"] == os.environ[modes.KIT_LEVERS_ENV]
    st = pkg.status()
    assert st["active"] and st["levers_pending"] == rep["levers_requested"] and st["partial"] is False
    again = pkg.enable("fast")
    assert again["active"] and again["hooks_installed"] == rep["hooks_installed"]        # idempotent
    other = pkg.enable("exact")
    assert not other["active"] and "one mode per process" in other["reason"]


def test_exact_activates_the_trunk_kernels_hook_chained_to_the_kits_eager(pkg, capsys):
    """The exact line (cueq): stock's cuEquivariance kernels are the runner yaml's (cli.row_yaml); the process carries fast init and the trunk-kernels
    add-on's two exact levers, no graph switch, no kernel swap."""
    from openfold3_opt import hooks
    rep = pkg.enable("exact")
    assert rep["active"] and rep["line"] == "cueq" and rep["hooks"] == ["cells", "trunk_kernels", "fast_inference"]
    assert os.environ["OF3_FAST_INIT"] == "1" and os.environ["OF3T_TEMPL_DISTINCT"] == "1" and os.environ["OF3T_PAIRCACHE"] == "1" and os.environ["OPENFOLD3_OPT_ATOM_HOIST"] == "1"
    assert all(k not in os.environ for k in ("OF3_CUDA_GRAPHS", "OF3_GRAPHS_STRICT", "OF3T_TRIATT", "OF3T_TRIMUL", "OF3T_APB", "OPENFOLD3_OPT_PAIR", "OPENFOLD3_OPT_DIT", "OPENFOLD3_OPT_ROLLOUT"))
    assert os.environ["OF3T_KIT_LEVERS"].endswith(os.path.join("fast_inference", "of3_levers"))
    assert {h["kit"] for h in hooks.installed()} == {"cells", "trunk_kernels", "fast_inference"}   # the package's cells hook (atom_hoist) + the two add-on hooks it chains to
    assert "line=cueq:" in capsys.readouterr().err
    assert pkg.modes.line_arg("exact") == "cueq"                                       # the one exact composition: no selector


def test_late_activation_refused_once_the_model_module_is_imported(pkg):
    m = _stubs.stub_model_module()                          # inserted directly: no finder fires, the class is wrapped on activation
    rep = pkg.enable("fast")
    assert not rep["active"] and "late activation" in rep["reason"] and "already imported" in rep["reason"]   # refused by the target's name


def test_instance_count_refuses_by_name(pkg, monkeypatch):
    """The instance-count refusal on its own: the counter armed directly (the count is opt_core.instances'), one instance built, the
    imported-target check silenced (no kit hook installed) — `enable()` is refused by the instance count and says so."""
    from openfold3_opt import stack, hooks
    m = _stubs.stub_model_module()
    assert stack.register_instance_counter() == "wrapped"
    m.OpenFold3()
    assert stack.instances() == 1
    monkeypatch.setattr(hooks, "targets_imported", lambda: [])                 # isolate the instance-count sentence from the imported-target one
    assert stack.late_activation_gate() == f"late activation: 1 {stack.MODEL_CLASS} instance(s) already built in this process"
    rep = pkg.enable("fast")
    assert not rep["active"] and "1 OpenFold3 instance(s) already built" in rep["reason"]
    with pytest.raises(pkg.ActivationError):
        pkg.enable("fast", strict=True)


class _ReentrantKitFinder:
    """The add-on's finder shape (of3t_hook/sitecustomize.py:59-98): marks itself done, then asks
    every OTHER finder on sys.meta_path — the package's counter finder included — and wraps the spec's exec_module."""
    def __init__(self, target):
        self.target, self.done, self.wraps = target, False, 0

    def find_spec(self, name, path=None, target=None):
        if name != self.target or self.done:
            return None
        self.done = True
        for finder in sys.meta_path:
            if finder is self:
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None and spec.loader is not None:
                orig = spec.loader.exec_module

                def exec_module(module, _orig=orig):
                    _orig(module)
                    module.KIT_WRAPS = getattr(module, "KIT_WRAPS", 0) + 1
                spec.loader.exec_module = exec_module
                self.wraps += 1
                return spec
        return None


def test_counter_finder_is_consulted_once_under_reentrant_kit_finders(pkg, tmp_path, monkeypatch):
    """Two kit-shaped finders inserted at sys.meta_path[0] after the counter finder (the composed line's order): the model module's
    import goes through all three, each wraps exec_module once, the counter finder is consulted once, instances are counted once."""
    from openfold3_opt import stack
    src = open(os.path.join(_stubs.tree_home(), "opt", "forward", "trunk_kernels", "of3t_hook", "sitecustomize.py"), encoding="utf-8").read()
    assert "for finder in sys.meta_path:" in src and "if finder is self:" in src         # the shape the stub reproduces
    root = tmp_path / "site"
    (root / "openfold3" / "projects" / "of3_all_atom").mkdir(parents=True)
    for rel in ("openfold3/__init__.py", "openfold3/projects/__init__.py", "openfold3/projects/of3_all_atom/__init__.py"):
        (root / rel).write_text("")
    (root / "openfold3" / "projects" / "of3_all_atom" / "model.py").write_text("class OpenFold3:\n    def __init__(self, *a, **k):\n        pass\n    def forward(self, batch):\n        return batch\n")
    for name in [n for n in list(sys.modules) if n == "openfold3" or n.startswith("openfold3.")]:
        del sys.modules[name]
    monkeypatch.syspath_prepend(str(root))
    assert stack.register_instance_counter() == "armed"
    finders = [_ReentrantKitFinder(stack.MODEL_MODULE), _ReentrantKitFinder(stack.MODEL_MODULE)]
    for f in finders:
        sys.meta_path.insert(0, f)
    try:
        import importlib
        m = importlib.import_module(stack.MODEL_MODULE)
    finally:
        for f in finders:
            sys.meta_path.remove(f)
    assert stack._COUNTER_FINDER.wraps == 1 and stack._COUNTER_FINDER.done
    assert [f.wraps for f in finders] == [1, 1] and m.KIT_WRAPS == 2
    m.OpenFold3()
    m.OpenFold3().forward({"x": 1})
    assert stack.instances() == 2


def _fn_from(path):
    """A function whose code file is `path` (what the add-ons' patches look like to `stack._patched_from`)."""
    ns = {}
    exec(compile("def f(self, *a, **k):\n    return None\n", path, "exec"), ns)
    return ns["f"]


def test_levers_record_reads_the_kits_own_records(pkg, capsys):
    """The exact line with the add-ons' records stubbed as the add-ons leave them: every lever pending before the targets are imported;
    applied / unavailable afterwards from the add-ons' flags, state dicts and patched methods — never from the mode table; a lever whose
    patch is not in place makes the arm partial and not complete, in status(), the exit tally and the manifest; a lever module loaded
    from another copy than the line's carrier does the same."""
    from openfold3_opt import manifest, registry, report, stack
    home = _stubs.tree_home()
    rep = pkg.enable("exact")
    assert rep["active"] and rep["line"] == "cueq"
    requested = list(rep["levers_requested"])
    # no token count on the enable() route: the exact line (graphed up to its own 512-token cap) runs EAGER, fail-closed — its graph levers are not requested
    assert list(pkg.modes.LINES[("exact", "cueq")].levers) == ["fast_init", "cuda_graphs", "graphs_strict", "templ_distinct", "paircache", "atom_hoist", "castcache", "exactln", "apb_hoist", "triatt_exact", "trimul_exact", "transition_exact", "post_release", "sync_hoist", "postfwd_mem", "loader_workers", "fastjson", "writer_overlap", "hostfeat", "ckpt_mmap"]
    assert requested == [l for l in pkg.modes.LINES[("exact", "cueq")].levers if l not in pkg.modes.GRAPHS_LEVERS] and rep["size_gate"].endswith("n_tok_unknown")
    assert stack.levers_record()["levers_pending"] == requested
    carrier = rep["carrier"]
    assert carrier == os.path.join(home, "opt", "forward", "fast_inference", "of3_levers")
    tk_hook = pkg.modes.hook_dir(home, "trunk_kernels")
    # the hook targets imported (stubs), the add-ons' records as the add-ons write them
    for t in (registry.T_LINEAR, registry.T_DIFFUSION, registry.T_MODEL, registry.T_WRITER, registry.T_MSA_IO, registry.T_CKPT):
        _stubs.stub_module(t)
    _stubs.stub_module("of3_fastinit", _ENABLED=True, __file__=os.path.join(carrier, "of3_fastinit.py"))
    _stubs.stub_module("of3t_levers")
    _stubs.stub_module("of3t_paircache", STATS={"rollouts": 5})
    tm = _stubs.stub_module("openfold3.core.model.latent.template_module")
    tm.TemplatePairStack = type("TemplatePairStack", (), {"forward": _fn_from(os.path.join(tk_hook, "of3t_levers.py"))})
    dc = _stubs.stub_module("openfold3.core.model.layers.diffusion_conditioning")
    dc.DiffusionConditioning = type("DiffusionConditioning", (), {"forward": _fn_from(os.path.join(tk_hook, "of3t_paircache.py"))})
    from openfold3_opt.cells import apb_hoist as _ap, atom_hoist as _ah, castcache as _cc, exactln as _xl, post_release as _po, sync_hoist as _sh, fastjson as _fj, postfwd_mem as _pm, loader_workers as _lw, triatt_exact as _te, trimul_exact as _tm, trimul_form as _tmf, transition_exact as _tx, writer_overlap as _wo, hostfeat as _hf, ckpt_mmap as _cm   # the cells' own records, as their installs leave them
    _prev = [(m, dict(m.STATE)) for m in (_ah, _cc, _xl, _ap, _po, _sh, _fj, _pm, _lw, _te, _tm, _tmf, _tx, _wo, _hf, _cm)]
    for m, _ in _prev:
        m.STATE.update(installed=True, state="on")
    _ah_prev = dict(_prev[0][1])
    try:
        r = stack.levers_record()
        assert r["levers_pending"] == [] and r["levers_unavailable"] == [] and r["levers_applied"] == requested
        assert r["partial"] is False and r["arm_complete"] is True and r["off_carrier"] == []
        assert r["lever_evidence"]["templ_distinct"].startswith("applied") and r["lever_evidence"]["paircache"].startswith("applied") and r["lever_evidence"]["fast_init"].startswith("applied")
        line = report.exit_tally_line(pkg.status(), stack.instances())
        assert " arm_complete=true " in line and "PARTIAL" not in line and "paircache_rollouts=5" in line and "levers_applied=" + ",".join(requested) in line
        m = manifest.build(pkg.status(), command="pred")
        assert m["levers_unavailable"] == [] and m["partial"] is False and m["arm_complete"] is True and m["levers_requested"] == requested
        # the pair-cache patch not in place (upstream's own forward): the lever is unavailable, the arm partial and not as shipped — everywhere
        dc.DiffusionConditioning = type("DiffusionConditioning", (), {"forward": lambda self: None})
        r = stack.levers_record()
        assert r["levers_unavailable"] == ["paircache"] and r["levers_applied"] == ["fast_init", "templ_distinct", "atom_hoist", "castcache", "exactln", "apb_hoist", "triatt_exact", "trimul_exact", "transition_exact", "post_release", "sync_hoist", "postfwd_mem", "loader_workers", "fastjson", "writer_overlap", "hostfeat", "ckpt_mmap"] and r["partial"] is True and r["arm_complete"] is False
        assert r["lever_evidence"]["paircache"].startswith("unavailable")
        st = pkg.status()
        assert st["levers_unavailable"] == ["paircache"] and st["arm_complete"] is False
        line = report.exit_tally_line(st, stack.instances())
        assert " levers_unavailable=paircache " in line and " arm_complete=false " in line and "PARTIAL: unavailable paircache" in line
        m = manifest.build(pkg.status(), command="pred")
        assert m["levers_unavailable"] == ["paircache"] and m["partial"] is True and m["arm_complete"] is False
        dc.DiffusionConditioning = type("DiffusionConditioning", (), {"forward": _fn_from(os.path.join(tk_hook, "of3t_paircache.py"))})
        assert stack.levers_record()["arm_complete"] is True
        # of3_fastinit from another copy than the line's carrier: off the carrier, not as shipped
        sys.modules["of3_fastinit"].__file__ = os.path.join(home, "opt", "forward", "elsewhere", "of3_levers", "of3_fastinit.py")
        r = stack.levers_record()
        assert r["off_carrier"] and "not the line's carrier" in r["off_carrier"][0] and r["partial"] and r["arm_complete"] is False
        assert "PARTIAL: of3_fastinit loaded from" in report.exit_tally_line(pkg.status(), 0)
    finally:
        for m, prev in _prev:
            m.STATE.clear(); m.STATE.update(prev)
        for n in ("of3_fastinit", "of3t_levers", "of3t_paircache", "openfold3.core.model.latent.template_module", "openfold3.core.model.layers.diffusion_conditioning",
                  registry.T_LINEAR, registry.T_DIFFUSION, registry.T_MODEL):
            sys.modules.pop(n, None)


def test_exit_tally_content_is_the_record(pkg):
    """The exit line is more than its prefix: mode, line, the instance count and the lever verdicts are all in it."""
    from openfold3_opt import report
    rep = pkg.enable("fast")
    line = report.exit_tally_line(pkg.status(), 2)
    assert line.startswith("[openfold3-opt] exit mode=fast line=- instances=2 levers_applied=none levers_unavailable=none levers_pending=")
    first = line.splitlines()[0].split(" ")                    # the EXIT line's own fields (the LEVER lines follow it); field order past the lever verdicts is the counters' own
    for field in ["levers_pending=" + ",".join(rep["levers_requested"]), "arm_complete=pending", "n_gpu=1", "sharding=none",
                  "templ_guard=on", "templ_policy=ignore(upstream_0.4.1)", "templ_parsed=0", "templ_ignored=0", "templ_sampled=0", "templ_dropped=0"]:   # an active process carries the template guard's census (templ_guard.counters)
        assert field in first, (field, line)
    from openfold3_opt import templ_guard
    templ_guard.uninstall()                                   # a process that never activated (mode off) carries no guard census
    off = report.exit_tally_line({"active": False, "mode": "off"}, None)      # `counters=none`, or the CUDA peak-memory counters alone on a GPU box — never a lever or guard field
    assert off.startswith("[openfold3-opt] exit mode=off line=- instances=- ") and "levers_" not in off and "templ_" not in off and "arm_complete" not in off
    assert off.endswith(" counters=none") or set(f.split("=")[0] for f in off.split(" instances=- ", 1)[1].split(" ")) <= {"cuda_max_reserved_mib", "cuda_max_allocated_mib"}, off


def test_instance_counter_counts_and_ready_line(pkg, capsys):
    from openfold3_opt import stack
    rep = pkg.enable("fast")
    assert rep["active"] and rep["instance_counter"] == "armed"
    m = _stubs.stub_model_module()
    assert stack.register_instance_counter() == "wrapped"
    a, b = m.OpenFold3(), m.OpenFold3()
    assert stack.instances() == 2
    b.forward({"x": 1})
    b.forward({"x": 2})
    err = capsys.readouterr().err
    assert err.count("[openfold3-opt] ready t=") == 1
    fwd = [l for l in err.splitlines() if "[openfold3-opt] Model forward time: " in l]
    assert len(fwd) == 2 and all("route=kit" in l for l in fwd)                                         # the per-item forward line, every add-on route (report.wrap_forward_timer)
    assert err.count("[openfold3-opt] PHASE item=") == 2                                                 # and its phase line (report.phase_line)
    assert err.index("ready t=") < err.index("Model forward time")
    rep2 = pkg.enable("exact")
    assert not rep2["active"]                                                # a second mode after activation: refused


def test_dry_run_applies_nothing(pkg):
    from openfold3_opt import hooks, stack
    rep = stack.activate("fast", dry_run=True)
    assert rep["dry_run"] and not rep["active"] and rep["reason"] == "dry run: nothing applied"
    assert "OF3T_TRIMUL" not in rep["env"] and "OF3T_TRIATT" not in rep["env"]          # the fast line carries neither trunk-kernels swap: the pair cells serve both ops through the core's providers
    assert "OF3T_TRIMUL" not in os.environ and hooks.installed() == [] and pkg.status()["active"] is False


def test_instance_counter_wraps_the_model_module_at_its_import(pkg, tmp_path, monkeypatch):
    """The counter is armed before the model module exists; the module is then imported through the path finder (a real file on
    sys.path, like the installed package) and the class built from it must be counted — the finder has to sit ahead of the path finder."""
    from openfold3_opt import stack
    root = tmp_path / "site"
    (root / "openfold3" / "projects" / "of3_all_atom").mkdir(parents=True)
    (root / "openfold3" / "__init__.py").write_text("")
    (root / "openfold3" / "projects" / "__init__.py").write_text("")
    (root / "openfold3" / "projects" / "of3_all_atom" / "__init__.py").write_text("")
    (root / "openfold3" / "projects" / "of3_all_atom" / "model.py").write_text("class OpenFold3:\n    def __init__(self, *a, **k):\n        pass\n    def forward(self, batch):\n        return batch\n")
    for name in [n for n in list(sys.modules) if n == "openfold3" or n.startswith("openfold3.")]:
        del sys.modules[name]
    monkeypatch.syspath_prepend(str(root))
    assert stack.register_instance_counter() == "armed"
    assert sys.meta_path[0] is stack._COUNTER_FINDER
    import importlib
    m = importlib.import_module(stack.MODEL_MODULE)
    assert stack.instances() == 0
    m.OpenFold3()
    assert stack.instances() == 1
    m.OpenFold3().forward({"x": 1})
    assert stack.instances() == 2
