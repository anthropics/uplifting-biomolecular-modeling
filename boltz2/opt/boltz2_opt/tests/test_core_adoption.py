"""The core this kit stands on (opt_core, pinned in opt/pyproject.toml [tool.opt_core]): the pin holds the imported core, the gate passes on
it, the autoload module's restated constants equal the core's, the manifest's shape gained exactly the core block; the routed kernel is the
kit's own bytes served by name from the core, in the gate and in the worker."""
import json
import os
import subprocess
import sys

import opt_core
from opt_core import gates as core_gates, manifest as core_manifest, report as core_report

from .. import _autoload, manifest as mf, modes, stack

OPT = os.path.dirname(os.path.dirname(os.path.abspath(stack.__file__)))


def _v(text):
    return tuple(int(x) for x in text.split("."))


def test_core_pin_holds_the_imported_core():
    """The pin is a minimum (opt_core.gates / _core_gate: `pinned >= v<want>`): the tree's core is at or above it, never below."""
    g = core_gates.core_pin_check(stack.PYPROJECT)
    assert g.ok, g.reason
    assert g.details["imported"]["version"] == opt_core.__version__ and _v(g.details["pinned"]["version"]) <= _v(opt_core.__version__), g.details


def test_core_gate_passes_on_the_pinned_core():
    from .._core_gate import gate
    facts = gate(os.path.join(OPT, "boltz2_opt", "__init__.py"), "boltz2-opt")
    assert facts["installed"]["version"] == opt_core.__version__ and _v(facts["pinned"]["version"]) <= _v(opt_core.__version__), facts


def test_autoload_restated_constants_equal_the_core_and_the_mode_table():
    assert _autoload.EXIT_NOT_ACTIVE == core_report.EXIT_NOT_ACTIVE == 3
    assert set(_autoload.MODES) == set(modes.MODE_NAMES)


def test_manifest_shape_is_the_kit_schema_plus_the_core_block(monkeypatch):
    monkeypatch.setattr(stack, "gpu_probe", lambda: None)
    m = mf.build("pred", "exact", report={"active": True}, evidence={}, inputs=[], outputs={}, rc=0)
    assert m["core"] == core_manifest.core_block()
    assert m["core"]["version"] == core_gates.core_pin_check(stack.PYPROJECT).details["imported"]["version"]
    expected = ["package", "package_version", "core", "written_utc", "route", "mode", "tier", "levers", "env_row", "kernels", "evidence_env", "pinned_route",
                "python", "platform", "gpu", "target_gpu", "gpu_class", "gpu_supported", "boltz_cache", "report", "evidence", "settings", "det",
                "staged_kit_files", "inputs", "outputs", "command", "rc", "kernels_census", "tally"]
    assert list(m.keys()) == expected, [k for k in m if k not in expected] + [k for k in expected if k not in m]


def test_routed_kernels_are_the_cores_and_the_worker_runs_under_the_route(tmp_path, monkeypatch):
    """The core serves `flash_triattn` to the big worker (and fast's block core) by name: this tree carries no copy of a kernel the core ships,
    the file is never staged beside the worker, the launch goes through worker_launch (every worker mode: the template guard; these modes: the routes), and the pre-launch gate holds."""
    from opt_core import kernels as core_kernels
    sums = stack.read_sums()
    for mode in modes.MODE_NAMES:
        for name in modes.routed_kernels(mode):
            doc = core_kernels.sums(name)
            assert doc["kind"] in ("module", "package"), (mode, name)
            shipped = {name + ".py"} if doc["kind"] == "module" else {f"{name}/{rel}" for rel in doc["files"]}
            assert not any(os.path.basename(f) == name + ".py" for f in modes.stage_files(mode)), "a routed kernel is never staged beside the worker"
            carried = {rel for rel in sums if os.path.basename(rel) == name + ".py" or any(rel.endswith("/" + s) for s in shipped)}
            assert carried == set(), f"the kit carries no copy of a kernel the core ships: {sorted(carried)}"
            for param, rel in modes.kernel_exports(name).items():            # the kit's own data for the kernel is carried and named by the sums file's exports
                assert rel in sums and any(spec["param"] == param for spec in doc["exports"].values()), (name, param, rel)
    assert modes.routed_kernels("fast") == [] and modes.routed_kernels("big") == ["flash_triattn"], "the TriMul kernels of every row are the core provider's, reached by their core names (no route, no export of this tree's); big routes the flash kernel for the n_gpu > 1 line"
    assert modes.routed_kernels("off") == [] and modes.routed_kernels("exact") == ["fpf_trimul"], "exact routes the TM-K3 package by name (the core provider's tmk3_exact row reaches it by that name or its core name)"
    _split = lambda c: (c[:c.index("--kernels-route")], c[c.index("--"):])   # noqa: E731 — (launcher head before the KERNELS options, the script line from "--")
    assert _split(stack.worker_command("b.json", "big"))[0] == [sys.executable, "-m", "boltz2_opt.worker_launch", "--route", "flash_triattn", "--attach", "templ,trimul,transition,pairblock,pairfuse,sampler,exactln,waste,msa,msa2,templskip,atom,precision,xl,writer,prefetch"] and _split(stack.worker_command("b.json", "big"))[1] == _split(stack.worker_command("b.json", "fast"))[1], \
        "the memory mode's worker runs under fast's routes (+ the flash kernel by name for the n_gpu > 1 line) and the attachments of the fast levers it carries plus the precision unit and the xl attach hook (before the featurizer, after every Transition patch), with fast's own worker line"
    assert [a for a in modes.attachments("big") if a not in ("xl", "precision")] == [a for a in modes.attachments("fast") if a in modes.attachments("big")], "big attaches fast's adapters in fast's order (+ its own precision unit and the xl hook)"
    assert stack.worker_command("b.json", "exact") == [sys.executable, "-m", "boltz2_opt.worker_launch", "--route", "fpf_trimul", "--attach", ",".join(modes.attachments("exact")),
                                                    "--kernels-route", "exact", "--kernels-expect", "cueq_triatt=engaged,cueq_trimul=engaged", "--kernels-settings", "defaults", "--kernels-ngpu", "1", "--kernels-mode", "exact",
                                                    "--", "bz_worker.py", "--batch", "b.json", "--mode", "fast", "--kernels", "on", "--num_workers", "1", "--pipeline", "1", "--keep_on_gpu", "1"], "exact = the kernels-on row under the exact TriMul route and the three core adapters, the KERNELS census expecting both cuEquivariance accelerators engaged"
    assert stack.kernels_opts("exact") == stack.worker_command("b.json", "exact")[7:17], "the KERNELS options are stack.kernels_opts, between the attachments and the script"
    assert _split(stack.worker_command("b.json", "fast"))[0] == [sys.executable, "-m", "boltz2_opt.worker_launch", "--attach", ",".join(modes.attachments("fast"))] and _split(stack.worker_command("b.json", "fast"))[1] == _split(stack.worker_command("b.json", "exact"))[1], "fast = the exact row's worker line (kernels on) under the Triton TriMul route, the same attachments and the fused MSA-module kernels' attach hook"
    assert "FPF_TRIMUL_V4_CELLS" not in os.environ and modes.KERNEL_EXPORTS == {}, "no cell table of this tree's: nothing exported for the core's Triton TriMul unit (its own table serves every part)"
    r = stack.route_kernels(["flash_triattn"])
    assert r["flash_triattn"]["resolved"] == r["flash_triattn"]["core_copy"] == core_kernels.carried_path("flash_triattn"), "route_check already held the resolved bytes to the core's own copy live; nothing left to restate here"
    assert "flash_triattn" not in sys.modules, "the gate resolves without importing"
    core_kernels.unroute("flash_triattn")
    # the launcher: the route installed, then the script as __main__ with its own argv; a refused route exits 3 by name
    import pytest
    pytest.importorskip("torch", reason="the routed kernel module (flash_triattn) imports torch when the script imports it; the route and its gate above are torch-free")
    from .. import worker_launch
    script = tmp_path / "w.py"; script.write_text("import sys, json, flash_triattn\nprint(json.dumps({'argv': sys.argv, 'kernel': flash_triattn.__file__}))\n")
    r = subprocess.run([sys.executable, "-m", "boltz2_opt.worker_launch", "--route", "flash_triattn", "--", str(script), "--x", "1"], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["argv"] == [str(script), "--x", "1"] and os.path.realpath(out["kernel"]) == os.path.realpath(core_kernels.carried_path("flash_triattn"))
    assert "[boltz2-opt route] flash_triattn served from" in r.stderr
    assert worker_launch.main(["--route", "no_such_kernel", "--", str(script)]) == 3
