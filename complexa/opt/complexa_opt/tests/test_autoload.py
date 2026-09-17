"""The autoload hook (`_autoload.py`, run by complexa_opt_autoload.pth at interpreter start): inert without a kit mode — nothing of the
core, no torch, no finder (what a stock child holds is exactly the declared pair stack.PTH_MODULES); armed under a kit mode with the trigger
`proteinfoundation.proteina`; a variable under the package prefix the package does not read is refused in every process (exit 3, one line);
the shipped .pth is the build backend's generated text for this package."""
import os
import subprocess
import sys

from complexa_opt import _core_gate, stack

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(os.path.dirname(HERE))                                   # complexa/opt


def probe(code, env_extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("COMPLEXA_OPT")}
    env.update(env_extra or {})
    r = subprocess.run([sys.executable, "-c", "import complexa_opt._autoload as A\n" + code], capture_output=True, text=True, env=env)
    return r.returncode, r.stdout.strip(), r.stderr


def test_inert_without_a_kit_mode():
    code = ("import sys\n"
            "held = sorted(m for m in sys.modules if m == 'complexa_opt' or m.startswith('complexa_opt.') or m == 'opt_core' or m.startswith('opt_core.') or m == 'torch')\n"
            "print(A.FINDER, ','.join(held))")
    for env in ({}, {"COMPLEXA_OPT": ""}, {"COMPLEXA_OPT": "off"}, {"COMPLEXA_OPT": " OFF "}):
        rc, out, err = probe(code, env)
        assert rc == 0, err
        assert out == "None complexa_opt,complexa_opt._autoload", (env, out)          # exactly stack.PTH_MODULES: no core, no torch, no finder
    assert stack.PTH_MODULES == ("complexa_opt", "complexa_opt._autoload")


def test_armed_under_a_kit_mode_with_the_model_module_as_trigger():
    code = "import sys; print(A.FINDER is not None and A.FINDER.armed, A.SPEC.triggers, A.SPEC.modes, 'torch' in sys.modules)"
    for mode in ("exact", "fast", "big", " Exact "):
        rc, out, err = probe(code, {"COMPLEXA_OPT": mode})
        assert rc == 0, err
        assert out == "True ('proteinfoundation.proteina',) ('off', 'exact', 'fast', 'big') False", (mode, out)


def test_a_variable_the_package_does_not_read_is_refused_in_every_process():
    for env in ({"COMPLEXA_OPT_LEVERS": "x"}, {"COMPLEXA_OPT": "exact", "COMPLEXA_OPTS": "1"}):
        rc, out, err = probe("print('reached')", env)
        assert rc == 3 and "reached" not in out, (env, out, err)
        assert err.count("[complexa-opt] NOT ACTIVE: reason=") == 1 and "is not a variable this package reads" in err, err
    rc, out, err = probe("print('reached')", {"COMPLEXA_OPT": "exact", "COMPLEXA_OPT_RECORD": "/tmp/r"})   # the record directory IS read (modes.ENV_RECORD)
    assert rc == 0 and out == "reached", err


def test_an_unknown_mode_word_is_refused_at_the_trigger_not_silently():
    """The core's contract for a selection outside the table: the finder refuses when the trigger is imported (here simulated by asking it)."""
    code = ("from opt_core import autoload\n"
            "sel = autoload.selection(A.SPEC)\n"
            "print(bool(sel[2]), 'turbo' in (sel[2] or ''))")
    rc, out, err = probe(code, {"COMPLEXA_OPT": "turbo"})
    assert rc == 0, err
    assert out == "True True", out                                                    # a refusal line naming the word is prepared for the trigger


def test_the_shipped_pth_is_the_backends_generated_text():
    sys.path.insert(0, OPT)
    try:
        import _build_backend as bb
    finally:
        sys.path.remove(OPT)
    with open(os.path.join(OPT, stack.PTH_FILE), encoding="utf-8") as fh:
        text = fh.read()
    assert bb.PTH == stack.PTH_FILE
    assert bb.pth_fields(text) == {"package": "complexa_opt", "env": "COMPLEXA_OPT", "tag": "complexa-opt", "exit_code": 3} or text == bb.pth_text("complexa_opt", "COMPLEXA_OPT", "complexa-opt")
    assert _core_gate.EXIT_NOT_ACTIVE == 3
