"""Every mode line's carrier PROVENANCE verdict over its executed hook chain: arm_complete true, nothing PARTIAL (CPU; a clean interpreter per line).

The exit tally judges a run PARTIAL (rc 3 through cli.exit_rule) when a carrier lever module (stack.CARRIER_MODULES: of3_fastinit,
of3_graphs — the fast-inference add-on's `of3_levers`) was loaded from anywhere but the line's carrier directory. Activation derived that
directory from the line's CHAIN variables (OF3T_KIT_LEVERS, OF3O_KIT_LEVERS, the *_CHAIN exports) — each names a chaining hook's NEXT directory;
on the `big --n_gpu P` (tp) line, whose chain is cells > tp_rowpair > fast_inference, the only chain variable names
tp_rowpair/hook, so a COMPLETE ×2 H100 run (ACTIVE on both ranks, every lever applied, 5/5 structures) printed `PARTIAL: of3_fastinit loaded from
…/fast_inference/of3_levers/of3_fastinit.py, not the line's carrier …/tp_rowpair/hook` and exited rc 3. The
carrier is now the line's own `fast_inference` hook directory (stack.carrier_dir). This test resolves EVERY line as the package does, executes its
full hook chain in a fresh interpreter exactly as tests/test_hook_chain_exec.py does, imports the carrier modules by name the way the levers do
(through the sys.path the chain leaves), takes the carrier from the package's own activation report (stack.activate dry-run — the code path the
run's exit tally reads) and evaluates stack.levers_record over it: `off_carrier` empty, `partial` false, `arm_complete` true, and the carrier IS the
line's fast_inference hook directory."""
import json
import os
import subprocess
import sys
import textwrap

import pytest

from openfold3_opt import modes
from openfold3_opt.tests import _stubs

HOME = _stubs.tree_home()
TP_ENV = {"OF3TP_RANK": "0", "OF3TP_WORLD": "2", "OPENFOLD3_OPT_N_GPU": "2"}

CHILD = textwrap.dedent('''
    import json, os, runpy, sys
    hook_dirs = json.loads(os.environ["_PROV_TEST_HOOK_DIRS"])
    entry = os.environ["_PROV_TEST_ENTRY"]
    mode = os.environ["_PROV_TEST_MODE"]
    kw = json.loads(os.environ["_PROV_TEST_KW"])
    home = os.environ["_PROV_TEST_HOME"]
    sys.setrecursionlimit(200)
    sys.path[:0] = hook_dirs                                   # activation: the line's hook directories first, in the line's order
    runpy.run_path(entry, run_name="openfold3_opt_provenance_test")     # the entry hook chains the rest (hooks.run)
    import of3_fastinit, of3_graphs                            # the carrier lever modules, resolved through the path the chain leaves — as fast_init / cuda_graphs import them
    from openfold3_opt import stack
    rep = stack.activate(mode, home=home, environ=dict(os.environ), dry_run=True, log=False, **kw)      # the activation report the exit tally reads (its `carrier`)
    rec = stack.levers_record({"levers_requested": [], "carrier": rep.get("carrier"), "home": home}, home)
    print("VERDICT " + json.dumps({"carrier": rep.get("carrier"), "off_carrier": rec.get("off_carrier"), "partial": rec.get("partial"),
                                  "arm_complete": rec.get("arm_complete"), "levers_pending": rec.get("levers_pending"),
                                  "files": {"of3_fastinit": of3_fastinit.__file__, "of3_graphs": of3_graphs.__file__}}))
''')


def _cases():
    out = []
    for (mode, el), ln in modes.LINES.items():
        if "fast_inference" not in tuple(ln.hooks or ()):
            continue
        if el in modes.TP_LINES:
            out.append(pytest.param(mode, el, dict(TP_ENV), {"n_gpu": 2}, id=f"{mode}/{el}"))
        elif mode == "big":
            out.append(pytest.param(mode, el, {}, {"n_tokens": 6000}, id=f"{mode}/{el}@6000"))
            out.append(pytest.param(mode, el, {}, {"n_tokens": 300}, id=f"{mode}/{el}@300"))
        else:
            out.append(pytest.param(mode, el, {}, {}, id=f"{mode}/{el or '-'}"))
    return out


def test_every_line_with_levers_carries_fast_inference_last():
    for (mode, el), ln in modes.LINES.items():
        if ln.hooks:
            assert tuple(ln.hooks)[-1] == "fast_inference", (mode, el, ln.hooks)


@pytest.mark.parametrize("mode,el,environ,kw", _cases())
def test_carrier_provenance_verdict_is_complete_on_every_line(mode, el, environ, kw):
    res = modes.resolve(mode, HOME, environ=environ, **kw)
    assert "fast_inference" in res.hooks
    want = os.path.realpath(res.hook_dirs[list(res.hooks).index("fast_inference")])
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OPENFOLD3_OPT", "OF3T_", "OF3O_", "OF3TP_", "OF3_", "MODEL_OPT_LEVERS_OFF"))}
    env.update({k: str(v) for k, v in res.exports.items()})
    env.update(environ)
    env.update({"_PROV_TEST_HOOK_DIRS": json.dumps(res.hook_dirs), "_PROV_TEST_ENTRY": res.entry_hook, "_PROV_TEST_MODE": mode,
                "_PROV_TEST_KW": json.dumps(kw), "_PROV_TEST_HOME": HOME, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)})
    r = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"{mode}/{el}: rc={r.returncode}\n{r.stderr[-3000:]}"
    line = [x for x in r.stdout.splitlines() if x.startswith("VERDICT ")]
    assert line, r.stdout[-2000:] + r.stderr[-1500:]
    v = json.loads(line[-1][len("VERDICT "):])
    assert v["carrier"] and os.path.realpath(v["carrier"]) == want, f"{mode}/{el}: carrier {v['carrier']} is not the line's fast_inference hook {want}"
    for mn, f in v["files"].items():
        assert os.path.realpath(f).startswith(want + os.sep), f"{mode}/{el}: {mn} resolved from {f}, not {want} (the chain left another of3_levers first on sys.path)"
    assert v["off_carrier"] == [], f"{mode}/{el}: PARTIAL by provenance: {v['off_carrier']}"
    assert v["partial"] is False and v["levers_pending"] == [] and v["arm_complete"] is True, f"{mode}/{el}: {v}"
