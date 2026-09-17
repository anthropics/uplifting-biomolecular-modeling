"""run.sh --config <cfg>: the sanctioned flag selects configs/<cfg>.env, an unknown name exits 2 by
name, install refuses it, the ATLASFOLD_KIT_CONFIG=<path> route still works, and nothing after the stock separator `--` is consumed (CPU, bash)."""
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
RUN = os.path.join(KIT, "run.sh")
PROBE = 'echo "cfg=${ATLASFOLD_KIT_CONFIG:-} gpu=${AFO_TARGET_GPU:-} argv=$*"'


def sh(*args, env=None):
    """run.sh with the python entry replaced by a probe that prints the config in force and the argv the package would get."""
    src = open(RUN).read().replace('exec python3 -m atlasfold_opt "$verb" "$@"', PROBE + '; exit 0')
    e = dict(os.environ); e.pop("ATLASFOLD_KIT_CONFIG", None); e.pop("AFO_TARGET_GPU", None); e.update(env or {})
    p = subprocess.run(["bash", "-c", src, RUN, *args], capture_output=True, text=True, env=e, cwd=KIT)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def test_config_flag_selects_the_card_file_before_or_after_the_verb():
    for argv in (["check", "--config", "a100", "--mode", "fast"], ["--config", "a100", "check", "--mode", "fast"], ["check", "--config=a100", "--mode", "fast"]):
        rc, out, err = sh(*argv)
        assert rc == 0, err
        assert out.endswith("configs/a100.env gpu=A100(sm80) argv=--mode fast".replace("configs/a100.env", os.path.join(KIT, "configs", "a100.env"))), out


def test_default_and_env_route():
    rc, out, err = sh("check", "--mode", "exact")
    assert rc == 0 and f"cfg={os.path.join(KIT, 'configs', 'h100.env')} gpu=H100(sm90) argv=--mode exact" in out, out
    rc, out, err = sh("check", env={"ATLASFOLD_KIT_CONFIG": os.path.join(KIT, "configs", "b300.env")})
    assert rc == 0 and "configs/b300.env gpu=" in out, out
    rc, out, err = sh("check", "--config", "a100", env={"ATLASFOLD_KIT_CONFIG": os.path.join(KIT, "configs", "b300.env")})   # the flag wins over the env route
    assert rc == 0 and "configs/a100.env gpu=A100(sm80)" in out, out


def test_unknown_config_and_install_refuse_by_name():
    rc, out, err = sh("check", "--config", "zz9", "--mode", "fast")
    assert rc == 2 and "run.sh: no such config: zz9" in err
    rc, out, err = sh("check", "--config")
    assert rc == 2 and "--config takes a name" in err
    rc, out, err = sh("install", "--config", "a100")
    assert rc == 2 and "install takes no --config" in err


def test_flags_after_the_stock_separator_are_the_stock_clis():
    rc, out, err = sh("pred", "--mode", "off", "--", "multimer", "--config", "whatever", "--seed", "1")
    assert rc == 0 and out.endswith("argv=--mode off -- multimer --config whatever --seed 1") and "configs/h100.env" in out, out
