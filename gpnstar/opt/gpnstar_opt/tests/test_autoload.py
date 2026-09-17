"""The autoload: the shipped .pth is the generated text; unset / off installs nothing; a set word arms the core's finder; a word outside the
table and a gate refusal end the process with exit 3 at the trigger's import — stock never runs silently under GPNSTAR_OPT."""
import os
import subprocess
import sys
import textwrap

from gpnstar_opt.tests._paths import OPT, PY, env_clean

sys.path.insert(0, OPT)
import _build_backend   # noqa: E402


def test_the_pth_is_the_generated_text():
    text = open(os.path.join(OPT, "gpnstar_opt_autoload.pth"), encoding="utf-8").read()
    assert text == _build_backend.pth_text("gpnstar_opt", "GPNSTAR_OPT", "gpnstar-opt", 3)
    assert _build_backend.pth_fields(text) == ("gpnstar_opt", "GPNSTAR_OPT", "gpnstar-opt", 3)


def _py(code, **env):
    if "PYTHONPATH" in env and os.environ.get("PYTHONPATH"):
        env["PYTHONPATH"] = env["PYTHONPATH"] + os.pathsep + os.environ["PYTHONPATH"]
    return subprocess.run([PY, "-c", code], capture_output=True, text=True, env=env_clean(**env), timeout=120)


def test_unset_or_off_installs_nothing():
    for env in ({}, {"GPNSTAR_OPT": "off"}, {"GPNSTAR_OPT": " OFF "}):
        p = _py("import sys, gpnstar_opt._autoload as a; print(a.FINDER); print('opt_core.autoload' in sys.modules)", **env)
        assert p.returncode == 0, p.stderr
        assert p.stdout.split() == ["None", "False"], p.stdout


def test_a_word_arms_the_finder_and_imports_nothing_else():
    p = _py("import sys, gpnstar_opt._autoload as a; print(type(a.FINDER).__name__, a.FINDER.armed, a.FINDER.mode); "
            "print('torch' in sys.modules, 'gpnstar_opt.stack' in sys.modules)", GPNSTAR_OPT="exact")
    assert p.returncode == 0, p.stderr
    assert p.stdout.split() == ["Finder", "True", "exact", "False", "False"], p.stdout


def _stub_trigger(tmp_path):
    d = tmp_path / "stub"
    (d / "gpn" / "star").mkdir(parents=True)
    (d / "gpn" / "__init__.py").write_text("")
    (d / "gpn" / "star" / "__init__.py").write_text("")
    (d / "gpn" / "star" / "inference.py").write_text(textwrap.dedent("""
        BODY_RAN = True
        class MLMforVEPModel:            # the names the arm patches; the stub carries them so a patch never finds them missing
            def __init__(self, *a, **k): pass
        class MLMforLogitsModel(MLMforVEPModel): pass
        class ModelCenterEmbedding(MLMforVEPModel): pass
    """))
    return str(d)


def test_an_unknown_word_is_refused_at_the_trigger_with_exit_3(tmp_path):
    stub = _stub_trigger(tmp_path)
    p = _py("import gpnstar_opt._autoload; import gpn.star.inference; print('stock ran')", GPNSTAR_OPT="warp9", PYTHONPATH=stub)
    assert p.returncode == 3, (p.stdout, p.stderr)
    assert "[gpnstar-opt] NOT ACTIVE: unknown GPNSTAR_OPT='warp9' (expected exact|off)" in p.stderr and "stock ran" not in p.stdout


def test_the_old_mode_words_are_unknown_now(tmp_path):
    stub = _stub_trigger(tmp_path)
    for word in ("fast", "exact-no-dedup", "big"):
        p = _py("import gpnstar_opt._autoload; import gpn.star.inference; print('stock ran')", GPNSTAR_OPT=word, PYTHONPATH=stub)
        assert p.returncode == 3, (word, p.stdout, p.stderr)
        assert f"[gpnstar-opt] NOT ACTIVE: unknown GPNSTAR_OPT='{word}' (expected exact|off)" in p.stderr and "stock ran" not in p.stdout, word


def test_a_gate_refusal_at_the_trigger_is_exit_3_never_silent_stock(tmp_path):
    stub = _stub_trigger(tmp_path)
    # this box has no pinned gpn (the stub is not an installed distribution) and usually no GPU: enable() refuses by name at the trigger
    p = _py("import gpnstar_opt._autoload; import gpn.star.inference; print('stock ran')", GPNSTAR_OPT="exact", PYTHONPATH=stub, CUDA_VISIBLE_DEVICES="-1")
    assert p.returncode == 3, (p.stdout, p.stderr)
    assert "[gpnstar-opt] NOT ACTIVE:" in p.stderr and "stock ran" not in p.stdout
