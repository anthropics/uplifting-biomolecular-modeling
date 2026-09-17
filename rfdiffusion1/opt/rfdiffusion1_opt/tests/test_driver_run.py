"""driver_run: the kit line runs under the launch's values — the drivers' constant strings on inference.write_trajectory / inference.cautious /
inference.deterministic (DRIVER_FIXED) are replaced by the --fixed values (typed, else upstream's defaults), the case's startnum is composed as
inference.design_startnum (the row found by its output prefix, which every row of the cases file carries; the token verbatim), every other
override passes verbatim in the driver's order; the constants are literally in the kit drivers' sources at the cited
lines (the lock against drift)."""
import json
import os
import subprocess
import sys
import textwrap

import pytest

from rfdiffusion1_opt import driver_run, modes, registry, stack

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))                 # the rfdiffusion1/ tree
DRIVER_OV = ["inference.input_pdb=/in/t.pdb", "inference.output_prefix=/out/caseA/des", "inference.model_directory_path=/w", "inference.num_designs=5",
             "contigmap.contigs=[A1-115/0 80-80]", "ppi.hotspot_res=[A59,A83,A91]", "inference.deterministic=True", "inference.write_trajectory=True",
             "inference.cautious=False"]                                     # rfd_bench.py:99-103 (--no-traj 0)


KIT_LITERALS = ("inference.deterministic=True", "inference.write_trajectory=", "inference.cautious=False")   # the driver's own strings on the three keys (rfd_bench.py:103)


def test_fixed_values_are_typed_or_upstreams():
    assert modes.DRIVER_FIXED == ("inference.write_trajectory", "inference.cautious", "inference.deterministic")
    assert modes.settings_of(attach="driver").compose_overrides == ("inference.write_trajectory=True", "inference.cautious=True", "inference.deterministic=False") == tuple(driver_run.fixed_overrides())   # base.yaml:13,17,21
    kc = modes.settings_of(["inference.write_trajectory=False", "inference.cautious=False"], "driver")
    assert kc.compose_overrides == ("inference.write_trajectory=False", "inference.cautious=False", "inference.deterministic=False") and kc.stock_overrides == ("inference.write_trajectory=False", "inference.cautious=False")
    assert driver_run.fixed_overrides(["inference.cautious=False"]) == ["inference.write_trajectory=True", "inference.cautious=False", "inference.deterministic=False"]
    assert driver_run.fixed_overrides(["inference.deterministic=True"]) == ["inference.write_trajectory=True", "inference.cautious=True", "inference.deterministic=True"] == list(modes.settings_of(attach="driver", det=True).compose_overrides)   # --det 1's composition
    with pytest.raises(SystemExit, match="--fixed takes"):
        driver_run.fixed_overrides(["inference.final_step=2"])                # a key the driver does not hard-code is --compose's, never --fixed's
    for kit, rel, line in ((KIT, "opt/forward/fast_inference/drivers/rfd_bench.py", 103),):
        src = open(os.path.join(kit, rel), encoding="utf-8").read().splitlines()[line - 1]
        assert all(lit in src for lit in KIT_LITERALS), (rel, line, src)     # the constants driver_run replaces are literally there (modes.DRIVER_FIXED_DOC)


def test_merge_rule_upstream_defaults():
    m = driver_run.merge_overrides(DRIVER_OV, modes.settings_of(attach="driver").compose_overrides, "5")
    assert m["dropped"] == ["inference.deterministic=True", "inference.write_trajectory=True", "inference.cautious=False"]   # the driver's three constants, in the driver's order
    assert m["appended"] == ["inference.write_trajectory=True", "inference.cautious=True", "inference.deterministic=False", "inference.design_startnum=5"]
    assert m["overrides"] == DRIVER_OV[:6] + m["appended"]                   # every other override verbatim, in the driver's order
    keys = [o.split("=", 1)[0] for o in m["overrides"]]
    assert len(keys) == len(set(keys))                                        # one value per key: no duplicate the last-wins rule would have to resolve
    md = driver_run.merge_overrides(DRIVER_OV, driver_run.fixed_overrides(["inference.deterministic=True"]), "5")   # --det 1: the driver's own True string dropped and the launch's True composed — one entry either way
    assert md["overrides"].count("inference.deterministic=True") == 1 and "inference.deterministic=False" not in md["overrides"]


def test_merge_rule_keeps_other_keys_and_handles_cli_path_startnum():
    ov = DRIVER_OV[:7] + ["inference.design_startnum=0", "inference.write_trajectory=False", "inference.cautious=False", "inference.final_step=5", "denoiser.noise_scale_ca=0.5"]
    m = driver_run.merge_overrides(ov, modes.settings_of(attach="driver").compose_overrides, "10")
    assert "inference.final_step=5" in m["overrides"] and "denoiser.noise_scale_ca=0.5" in m["overrides"]   # keys outside the driver's constants and the start number, verbatim
    assert m["overrides"].count("inference.design_startnum=10") == 1 and "inference.design_startnum=0" in m["dropped"]
    assert driver_run.merge_overrides(ov, driver_run.fixed_overrides(), "007")["appended"][-1] == "inference.design_startnum=007"   # the token is composed verbatim (the stock command line carries the typed string too)
    m0 = driver_run.merge_overrides(ov, modes.settings_of(["inference.write_trajectory=False", "inference.cautious=False"], "driver").compose_overrides, None)        # no startnum known: the driver's own stays
    assert "inference.design_startnum=0" in m0["overrides"] and m0["appended"] == ["inference.write_trajectory=False", "inference.cautious=False", "inference.deterministic=False"]
    mx = driver_run.merge_overrides(DRIVER_OV, driver_run.fixed_overrides(), "5", extra=["inference.final_step=2", "denoiser.noise_scale_ca=0.5"])   # --compose: the launch's other typed keys, appended after the startnum; a driver string on such a key would be dropped
    assert mx["appended"][-2:] == ["inference.final_step=2", "denoiser.noise_scale_ca=0.5"] and mx["overrides"][-2:] == mx["appended"][-2:]


def test_case_of_by_output_prefix(tmp_path):
    """The composition's row is read from its inference.output_prefix: every row of the driver's cases file carries its `prefix`
    (design.write_cases: `<out>/<case>/des` on the cases form, upstream's typed prefix on the hydra form) and case_startnums keys the
    startnum token by it — the row's startnum, or on the hydra form the typed inference.design_startnum token verbatim (None = untyped:
    nothing composed, as upstream's own command line would carry nothing)."""
    from rfdiffusion1_opt import design
    rows = [{"name": "caseA", "startnum": 5, "num_designs": 1}, {"name": "caseB", "num_designs": 1},
            {"name": "design", "startnum": 7, "num_designs": 1, "prefix": str(tmp_path / "run" / "samples" / "design"), "startnum_compose": "7"},
            {"name": "d2", "startnum": 0, "num_designs": 1, "prefix": str(tmp_path / "run2" / "d2"), "startnum_compose": None}]
    cases = design.write_cases(rows, str(tmp_path / "work"), "/out")                          # the ONE writer of the driver's cases file
    on_disk = json.load(open(cases))
    assert [r["prefix"] for r in on_disk] == ["/out/caseA/des", "/out/caseB/des", str(tmp_path / "run" / "samples" / "design"), str(tmp_path / "run2" / "d2")]
    sn = driver_run.case_startnums(cases)
    assert sn == {"/out/caseA/des": "5", "/out/caseB/des": "0", str(tmp_path / "run" / "samples" / "design"): "7", str(tmp_path / "run2" / "d2"): None}
    assert driver_run.STARTNUM_TOKEN == "startnum_compose" and not hasattr(driver_run, "case_prefixes")
    assert driver_run.case_of(DRIVER_OV, sn) == "/out/caseA/des" and driver_run.case_of(["inference.output_prefix=/out/other/des"], sn) is None
    hydra_ov = ["inference.input_pdb=/in/t.pdb", f"inference.output_prefix={tmp_path}/run/samples/./design", "inference.num_designs=3"]
    assert driver_run.case_of(hydra_ov, sn) == str(tmp_path / "run" / "samples" / "design")   # compared as absolute paths
    m = driver_run.merge_overrides(hydra_ov + ["inference.deterministic=True"], driver_run.fixed_overrides(), sn[driver_run.case_of(hydra_ov, sn)])
    assert m["appended"][-1] == "inference.design_startnum=7"
    m2 = driver_run.merge_overrides([f"inference.output_prefix={tmp_path}/run2/d2", "inference.design_startnum=0"], driver_run.fixed_overrides(), sn[str(tmp_path / "run2" / "d2")])
    assert "inference.design_startnum=0" in m2["overrides"] and not any(a.startswith("inference.design_startnum") for a in m2["appended"])   # untyped on the hydra form: the driver's own string stays, nothing composed
    assert driver_run.case_startnums(None) == {}


def test_live_wrap_through_module_entry(tmp_path):
    """python -m rfdiffusion1_opt.driver_run on a stub hydra + a stub driver that composes the drivers' literal list: the composed config carries
    the --fixed values, the case's design_startnum (cases form by layout, hydra form by the row's prefix) and the --compose strings."""
    site = tmp_path / "site"; (site / "hydra").mkdir(parents=True)
    (site / "hydra" / "__init__.py").write_text(textwrap.dedent('''
        def compose(config_name=None, overrides=None, *a, **k):
            return {"config_name": config_name, "overrides": list(overrides or [])}
        class initialize_config_dir:
            def __init__(self, **k): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False
    '''))
    drv = tmp_path / "stub_driver.py"
    drv.write_text(textwrap.dedent('''
        import json, os, sys
        from hydra import compose, initialize_config_dir
        out = sys.argv[sys.argv.index("--out") + 1]
        cases = json.load(open(sys.argv[sys.argv.index("--cases") + 1]))
        res = {}
        for c in cases:
            prefix = os.path.abspath(c["prefix"]) if c.get("prefix") else f"{out}/{c['name']}/des"      # rfd_bench.py:113
            ov = [f"inference.input_pdb={c['pdb']}", f"inference.output_prefix={prefix}", "inference.model_directory_path=/w",
                  f"inference.num_designs={c['num_designs']}", f"contigmap.contigs={c['contigs']}", f"ppi.hotspot_res={c['hotspots']}",
                  "inference.deterministic=True", "inference.write_trajectory=True", "inference.cautious=False"]
            with initialize_config_dir(config_dir="/x", version_base=None):
                res[c["name"]] = compose(config_name="base", overrides=ov)
        json.dump({"argv": sys.argv, "composed": res}, open(f"{out}/composed.json", "w"))
    '''))
    from rfdiffusion1_opt import design
    hydra_prefix = str(tmp_path / "elsewhere" / "samples" / "binder")
    out = tmp_path / "out"; out.mkdir(); log = tmp_path / "compose.jsonl"
    cases = design.write_cases([{"name": "caseA", "pdb": "/in/t.pdb", "contigs": "[A1-10/0 5-5]", "hotspots": "[A1]", "num_designs": 5, "startnum": 5},
                                {"name": "binder", "pdb": "/in/t.pdb", "contigs": "[A1-10/0 5-5]", "hotspots": "null", "num_designs": 2, "startnum": 3, "prefix": hydra_prefix, "startnum_compose": "3"}],
                               str(tmp_path / "work"), str(out))                                           # the package's own writer: every row carries its prefix
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(site), os.path.join(KIT, "opt")] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]))
    fixed = ["--fixed", "inference.write_trajectory=True", "--fixed", "inference.cautious=True", "--fixed", "inference.deterministic=False"]
    cmd = [sys.executable, "-m", "rfdiffusion1_opt.driver_run", *fixed, "--cases", str(cases), "--log", str(log), "--compose", "inference.final_step=2", str(drv),
           "--mode", "time", "--cases", str(cases), "--out", str(out), "--tag", "t"]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    got = json.load(open(out / "composed.json"))
    assert got["argv"][0] == str(drv) and got["argv"][1:] == ["--mode", "time", "--cases", str(cases), "--out", str(out), "--tag", "t"]   # the driver sees its own argv
    ov = got["composed"]["caseA"]["overrides"]
    assert ov[-5:] == ["inference.write_trajectory=True", "inference.cautious=True", "inference.deterministic=False", "inference.design_startnum=5", "inference.final_step=2"]
    assert "inference.cautious=False" not in ov and "inference.deterministic=True" not in ov
    assert ov[:6] == ["inference.input_pdb=/in/t.pdb", f"inference.output_prefix={out}/caseA/des", "inference.model_directory_path=/w", "inference.num_designs=5",
                      "contigmap.contigs=[A1-10/0 5-5]", "ppi.hotspot_res=[A1]"]
    hv = got["composed"]["binder"]["overrides"]                                            # the hydra-form row: its prefix as typed, its startnum composed
    assert hv[1] == f"inference.output_prefix={hydra_prefix}" and "inference.design_startnum=3" in hv and hv[-1] == "inference.final_step=2"
    recs = [json.loads(l) for l in log.read_text().splitlines()]
    assert recs == [{"case": f"{out}/caseA/des", "fixed": fixed[1::2], "dropped": ["inference.deterministic=True", "inference.write_trajectory=True", "inference.cautious=False"],
                     "appended": fixed[1::2] + ["inference.design_startnum=5", "inference.final_step=2"]},
                    {"case": hydra_prefix, "fixed": fixed[1::2], "dropped": ["inference.deterministic=True", "inference.write_trajectory=True", "inference.cautious=False"],
                     "appended": fixed[1::2] + ["inference.design_startnum=3", "inference.final_step=2"]}]   # the row named by its prefix


# ------------------------------------------------------------------------------------------- the SE(3) add-on's arming in the driver process

def _fake_addon(tmp_path, applied=True, triton="3.0.0", fallback=None):
    """A stand-in add-on directory: rfd_se3fast.apply() returns the given stats and records that it ran; it imports no model."""
    d = tmp_path / "se3fast_addon"; (d / "rfd_se3fast").mkdir(parents=True)
    (d / "rfd_se3fast" / "__init__.py").write_text(textwrap.dedent(f"""
        __version__ = "0.3.0"
        CALLS = []
        def apply(mode=None, scope=None, verbose=True):
            CALLS.append(mode)
            return {{"applied": {applied!r}, "triton": {triton!r}, "fallback_reason": {fallback!r}, "mode": mode}}
        def stats():
            return {{"mode": "t2", "n_calls": 0, "n_full": 0, "n_topk": 0, "triton": {triton!r}, "fallback_reason": {fallback!r}, "verify_fail": 0}}
    """))
    return str(d)


@pytest.fixture
def clean_modules(monkeypatch):
    for name in ("rfd_se3fast", "triton", "rfdiffusion"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield
    for name in ("rfd_se3fast", "triton"):
        sys.modules.pop(name, None)


def test_arm_se3fast_is_inert_without_the_row(tmp_path, clean_modules):
    """No RFD_SE3FAST in the driver environment (every mode but base/fast), or the add-on's own off value: nothing imported, nothing on sys.path."""
    before = list(sys.path); d = _fake_addon(tmp_path)
    for v in (None, "", " ", "0"):
        assert driver_run.arm_se3fast(v, addon_dir=d) is None
    assert sys.path == before and "rfd_se3fast" not in sys.modules


@pytest.mark.parametrize("value", ["t2torch", "exact", "T2Torch", "1"])
def test_arm_se3fast_refuses_values_other_than_the_rows(tmp_path, capsys, clean_modules, value):
    """The one value a mode row sets is t2; the add-on's torch reference (t2torch) and reserved (exact) settings are refused by name, exit 3."""
    with pytest.raises(SystemExit) as ei:
        driver_run.arm_se3fast(value, addon_dir=_fake_addon(tmp_path))
    assert ei.value.code == 3
    out = capsys.readouterr().out
    assert out.startswith("[rfdiffusion1-opt] REFUSED: RFD_SE3FAST=") and "'t2' (registry lever T2)" in out and "rfd_se3fast" not in sys.modules


def test_arm_se3fast_refuses_a_missing_addon_directory(tmp_path, capsys, clean_modules):
    with pytest.raises(SystemExit) as ei:
        driver_run.arm_se3fast("t2", addon_dir=str(tmp_path / "nowhere"))
    assert ei.value.code == 3 and "does not hold rfd_se3fast/" in capsys.readouterr().out and "rfd_se3fast" not in sys.modules


def test_arm_se3fast_refuses_without_triton(tmp_path, capsys, monkeypatch, clean_modules):
    """triton not importable in the driver process: refused by name before the add-on is imported — its own t2 -> t2torch degradation never runs."""
    monkeypatch.setitem(sys.modules, "triton", None)                     # `import triton` raises ImportError
    with pytest.raises(SystemExit) as ei:
        driver_run.arm_se3fast("t2", addon_dir=_fake_addon(tmp_path))
    assert ei.value.code == 3
    out = capsys.readouterr().out
    assert "REFUSED: RFD_SE3FAST=t2 needs triton in this process" in out and "rfd_se3fast" not in sys.modules


@pytest.mark.parametrize("stats", [dict(applied=False), dict(triton=None), dict(fallback="triton unavailable: ImportError()")])
def test_arm_se3fast_refuses_off_the_triton_line(tmp_path, capsys, monkeypatch, clean_modules, stats):
    """The add-on imported and applied, but not on its Triton line (not applied / no triton version / a fallback reason): refused by name, exit 3."""
    import types
    monkeypatch.setitem(sys.modules, "triton", types.ModuleType("triton"))
    with pytest.raises(SystemExit) as ei:
        driver_run.arm_se3fast("t2", addon_dir=_fake_addon(tmp_path, **stats))
    assert ei.value.code == 3 and "rfd_se3fast did not apply its Triton line" in capsys.readouterr().out


def test_arm_se3fast_arms_the_triton_line(tmp_path, capsys, monkeypatch, clean_modules):
    import types
    monkeypatch.setitem(sys.modules, "triton", types.ModuleType("triton"))
    d = _fake_addon(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    registered = []
    monkeypatch.setattr(driver_run, "__name__", driver_run.__name__)     # (no-op; keeps monkeypatch in scope for atexit below)
    import atexit
    monkeypatch.setattr(atexit, "register", lambda f, *a, **k: registered.append(f) or f)
    st = driver_run.arm_se3fast("t2", addon_dir=d)
    assert st["applied"] is True and st["triton"] == "3.0.0" and sys.path[0] == d and sys.modules["rfd_se3fast"].CALLS == ["t2"]
    assert len(registered) == 1
    registered[0]()                                                        # the exit line: the add-on's counters by name
    assert capsys.readouterr().out.strip().endswith("fallback_reason=None verify_fail=0") and "rfdiffusion" not in sys.modules


# ------------------------------------------------------------------------------------------- lever K2's kernel route in the driver process

def test_route_core_kernels_serves_k2_from_the_core_copy(monkeypatch):
    """The drivers' `import rfd_layernorm` (rfd_bench.py under --triton-ln 1) resolves to the shared core's carried
    opt_core/kernels/rfd_layernorm.py through the core's route; the tree carries no copy beside the drivers that import it."""
    import importlib.util
    from opt_core import kernels
    monkeypatch.delitem(sys.modules, "rfd_layernorm", raising=False)
    core = kernels.carried_path("rfd_layernorm")
    try:
        assert driver_run.route_core_kernels() == {"rfd_layernorm": core}
        assert core.replace(os.sep, "/").endswith("/opt_core/kernels/rfd_layernorm.py") and os.path.isfile(core)
        assert importlib.util.find_spec("rfd_layernorm").origin == core
        assert kernels.route_check("rfd_layernorm").ok and "rfd_layernorm" in kernels.routed()
    finally:
        kernels.unroute("rfd_layernorm")
    assert not os.path.exists(os.path.join(stack.kit_dir(registry.KIT_BASE), "drivers", "rfd_layernorm.py"))
    k2 = registry.LEVERS["K2"]
    with open(os.path.join(stack.kit_dir(k2.kit), k2.kit_file), encoding="utf-8") as fh:
        assert "import rfd_layernorm" in fh.read()                            # the switch site the registry names imports the routed name
    assert driver_run.CORE_KERNELS == {"rfd_layernorm": "K2"} and "rfd_layernorm" in stack.KIT_LEVER_MODULES


def test_route_core_kernels_refuses_by_name(monkeypatch, capsys):
    """A name that would resolve to bytes other than the core copy's: refused by name, exit 3, before any driver runs."""
    from opt_core import kernels
    from opt_core.gates import Gate
    monkeypatch.setattr(kernels, "route_check", lambda name, *a, **k: Gate(name=f"kernel_route:{name}", ok=False, reason=f"{name} resolves to /elsewhere/{name}.py, not the carried bytes"))
    try:
        with pytest.raises(SystemExit) as ei:
            driver_run.route_core_kernels()
    finally:
        kernels.unroute("rfd_layernorm")
    assert ei.value.code == 3
    assert "[rfdiffusion1-opt] REFUSED: kernel rfd_layernorm (lever K2): rfd_layernorm resolves to /elsewhere/rfd_layernorm.py, not the carried bytes" in capsys.readouterr().out


def test_module_entry_refuses_before_the_driver_runs(tmp_path):
    """python -m rfdiffusion1_opt.driver_run under RFD_SE3FAST=t2torch: exit 3 with the named fact, and the driver script never starts (no model import)."""
    site = tmp_path / "site"; (site / "hydra").mkdir(parents=True)
    (site / "hydra" / "__init__.py").write_text("def compose(*a, **k): return {}\nclass initialize_config_dir:\n    def __init__(self, **k): pass\n")
    drv = tmp_path / "stub_driver.py"; marker = tmp_path / "driver_ran"
    drv.write_text(f"open({str(marker)!r}, 'w').write('ran')\n")
    env = dict(os.environ, RFD_SE3FAST="t2torch", PYTHONPATH=os.pathsep.join([str(site), os.path.join(KIT, "opt")] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]))
    p = subprocess.run([sys.executable, "-m", "rfdiffusion1_opt.driver_run", str(drv)], env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 3, p.stdout + p.stderr
    assert "[rfdiffusion1-opt] REFUSED: RFD_SE3FAST='t2torch'" in p.stdout and not marker.exists()
    env["RFD_SE3FAST"] = "0"                                               # the add-on's own off value: the driver runs
    p = subprocess.run([sys.executable, "-m", "rfdiffusion1_opt.driver_run", str(drv)], env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    assert marker.exists()
