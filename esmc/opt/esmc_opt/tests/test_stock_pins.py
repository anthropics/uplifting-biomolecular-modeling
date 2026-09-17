"""stock/: PINS.json names the upstream commit, the stack, the weights digests and the forbidden environment; check_pins.py refuses
where the pin is absent and names the stack from the installed packages."""
import importlib.util
import json
import os
import subprocess
import sys

import pytest

from ._paths import STOCK


def _pins():
    return json.load(open(os.path.join(STOCK, "PINS.json")))


def test_pins_shape():
    p = _pins()
    assert p["model"] == "esmc"
    assert set(p["variants"]) == {"300m", "600m", "6b"}
    up = p["upstream"]["esm"]
    assert up["commit"] == "43ccece2ad485f27db46afdb67da2a9601e8f106" and up["version"] == "3.4.0" and up["tag"] is None
    assert p["upstream"]["transformers"]["commit"] == "ef32577f55da19a4989cd7b22e004dc43a4998cb"
    assert set(p["stacks"]) == {"accel"}                                       # one stack: every mode runs on it
    assert p["stacks"]["accel"]["flash_attn"] and p["stacks"]["accel"]["transformer_engine"]
    assert p["pins"]["flash_attn"] == p["stacks"]["accel"]["flash_attn"] and p["pins"]["transformer_engine"] == p["stacks"]["accel"]["transformer_engine"]
    assert p["pins"]["xformers"] is None
    for repo in ("biohub/ESMC-300M", "biohub/ESMC-600M", "biohub/ESMC-6B"):
        w = p["weights"][repo]
        assert w["snapshot_commit"] and w["weights_files"] and w["total_weights_bytes"] > 0
        for f in w["weights_files"]:
            assert w["files"][f]["sha256"], (repo, f)
    assert "ESMC_OPT" in p["stock_environment"]["must_be_absent"] and "ESMC_KIT" in p["stock_environment"]["must_be_absent"] and not any(n.startswith("MO_SDK_") for n in p["stock_environment"]["must_be_absent"])
    assert "CUBLAS_WORKSPACE_CONFIG" not in p["stock_environment"]["must_be_absent"]          # the deterministic recipe's variable
    assert not any(n.endswith("_") for n in p["stock_environment"]["must_be_absent"]), "exact names, never prefixes"
    assert p["entry_point"]["route"].startswith("route_B")
    assert p["entry_point"]["cli"] is None


def test_src_subset_present():
    for rel in ("esm/models/esmc/compatibility.py", "esm/models/esmc/model.py", "esm/models/hub.py", "esm/sdk/api.py", "LICENSE.md", "pyproject.toml"):
        assert os.path.isfile(os.path.join(STOCK, "src", "esm", rel)), rel


def test_check_pins_refuses_without_upstream():
    probe = subprocess.run([sys.executable, "-I", "-c", "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('esm') is None else 1)"], capture_output=True)
    if probe.returncode != 0:
        pytest.skip("the upstream esm package is installed on this interpreter: the refusal without upstream is not observable here (check_pins checks that install instead)")
    out = subprocess.run([sys.executable, "-I", os.path.join(STOCK, "check_pins.py")], capture_output=True, text=True)
    assert out.returncode == 3
    assert "esm" in out.stderr


def test_stack_naming():
    spec = importlib.util.spec_from_file_location("cp", os.path.join(STOCK, "check_pins.py"))
    cp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cp)
    p = _pins()
    orig = cp._version
    try:
        cp._version = lambda n: {"flash_attn": p["stacks"]["accel"]["flash_attn"], "transformer_engine": p["stacks"]["accel"]["transformer_engine"]}.get(n)
        assert cp.stack(p)[0] == "accel"
        cp._version = lambda n: None
        name, _, drift = cp.stack(p)
        assert name is None and drift and "not the pinned stack" in drift[0]       # nothing installed: named as drift (never a refusal; no second stack)
        cp._version = lambda n: {"flash_attn": "9.9.9"}.get(n)
        name, _, drift = cp.stack(p)
        assert name is None and drift
        cp._version = lambda n: {"xformers": "0.0.35"}.get(n)
        name, _, drift = cp.stack(p)
        assert drift and "xformers" in drift[0]
    finally:
        cp._version = orig


def _check_pins_module():
    spec = importlib.util.spec_from_file_location("cp_probe", os.path.join(STOCK, "check_pins.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_import_probe_names_a_broken_accelerator_and_a_version_off_the_pin():
    """The stack is proven by IMPORT and every finding sorted: torch not importing (or a silent probe) is ``bad`` — nothing can run, refused;
    an accelerator module that fails to import in the fresh child, or a package whose `__version__` differs from the stack's pin, is
    ``drift`` — NAMED with its error, never refused (a lever that needs it steps aside by name; the KERNELS line says what engages)."""
    cp = _check_pins_module()
    pins = json.load(open(os.path.join(STOCK, "PINS.json")))
    accel = pins["stacks"]["accel"]

    def runner_for(report):
        return lambda cmd: (0, "noise\nPROBE " + json.dumps(report) + "\n", "")

    ok = {"torch": {"ok": True, "version": pins["framework"]["version"] if isinstance(pins.get("framework"), dict) and pins["framework"].get("version") else "x", "cuda": "13.0"}}
    for a, mods in cp.ACCEL_IMPORTS.items():
        for mo in mods:
            ok[mo] = {"ok": True, "version": accel[a] if mo == a else None, "file": f"/site/{mo}.py"}
    bad, drift, detail = cp.import_probe("accel", pins, runner=runner_for(ok))
    assert bad == [] and drift == [] and detail["stack"] == "accel" and set(detail["modules"]) == set(ok)
    broken = dict(ok, **{"flash_attn.flash_attn_interface": {"ok": False, "error": "ImportError: libcudart.so.13: cannot open shared object file"}})
    bad, drift, _ = cp.import_probe("accel", pins, runner=runner_for(broken))
    assert bad == [] and len(drift) == 1 and drift[0].startswith("flash_attn.flash_attn_interface: import fails on this stack (ImportError: libcudart.so.13") and "stack accel pins it usable" in drift[0]
    offpin = dict(ok, transformer_engine={"ok": True, "version": "9.9.9", "file": "/site/te.py"})
    bad, drift, _ = cp.import_probe("accel", pins, runner=runner_for(offpin))
    assert bad == [] and drift == [f"transformer_engine: imports as version 9.9.9; stack accel pins {accel['transformer_engine']}"]
    localtag = dict(ok, flash_attn={"ok": True, "version": accel["flash_attn"] + "+cu130torch2.11", "file": "/site/fa.py"})
    assert cp.import_probe("accel", pins, runner=runner_for(localtag))[:2] == ([], [])      # a local +tag on the pinned version is the pinned version
    notorch = dict(ok, torch={"ok": False, "error": "ImportError: libtorch_cuda.so"})
    bad, drift, _ = cp.import_probe("accel", pins, runner=runner_for(notorch))
    assert len(bad) == 1 and bad[0].startswith("torch does not import") and drift == []    # nothing can run: the one refusal of the probe
    silent, drift, d = cp.import_probe("accel", pins, runner=lambda cmd: (1, "", "Segmentation fault"))
    assert len(silent) == 1 and "did not report (rc=1)" in silent[0] and "Segmentation fault" in silent[0] and drift == []


def test_main_refuses_only_the_upstream_commit_and_names_stack_drift(monkeypatch, capsys):
    """check_pins.main: the upstream packages at their commits + torch importing = exit 0 whatever the stack drift (each drift line printed
    `check_pins: drift (named, not refused): …`); an upstream package off its commit (or absent) = exit 3 with the refusal named."""
    cp = _check_pins_module()
    pins = json.load(open(os.path.join(STOCK, "PINS.json")))
    probe_ok = {"torch": {"ok": True, "version": "x", "cuda": "13.0"}}
    monkeypatch.setattr(cp, "import_probe", lambda name, p, **kw: ([], ["transformer_engine: imports as version 9.9.9; stack accel pins X"], {"stack": name, "modules": probe_ok, "rc": 0}))
    monkeypatch.setattr(cp, "_version", lambda n: None)                                   # no accelerator installed: drift, named
    monkeypatch.setattr(cp, "check", lambda up: ([], {n: {"version": "v", "pinned": True, "source": "git"} for n in up}))   # the upstream commits pinned
    monkeypatch.setattr(sys, "argv", ["check_pins.py"])
    cp.main()                                                                              # returns: exit 0
    err = capsys.readouterr().err
    assert "check_pins: drift (named, not refused): not the pinned stack: flash_attn=None transformer_engine=None" in err
    assert "check_pins: drift (named, not refused): transformer_engine: imports as version 9.9.9" in err
    monkeypatch.setattr(cp, "check", lambda up: (["esm: installed at commit deadbeef, pinned 43ccece2"], {}))    # the stock off its commit: refused
    with pytest.raises(SystemExit) as ei:
        cp.main()
    assert ei.value.code == 3 and "check_pins: esm: installed at commit deadbeef" in capsys.readouterr().err


