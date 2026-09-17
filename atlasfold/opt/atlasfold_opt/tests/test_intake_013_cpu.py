import os, subprocess, sys, tempfile

def test_cli_has_four_verbs_only():
    from atlasfold_opt import cli
    src = open(cli.__file__).read()
    assert 'choices=("pred", "check", "warm", "install")' in src

def test_install_with_missing_weights_is_information_exit_0():
    d = tempfile.mkdtemp(prefix="afo_w_")                      # empty weights root: everything missing
    env = dict(os.environ); env.pop("ATLASFOLD_OPT", None)
    r = subprocess.run([sys.executable, "-m", "atlasfold_opt", "install", "--weights", d], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-400:]
    assert "WEIGHTS NOTE" in r.stderr and "not a gate" in r.stderr

def test_check_pins_weights_leg_never_gates():
    here = os.path.dirname(os.path.abspath(__file__))
    kit = os.path.abspath(os.path.join(here, "..", "..", ".."))
    d = tempfile.mkdtemp(prefix="afo_w_")
    r = subprocess.run([sys.executable, os.path.join(kit, "stock", "check_pins.py"), "--weights", d], capture_output=True, text=True)
    assert "stock/src tree sha256 OK" in r.stdout, r.stdout[-300:]
    assert r.returncode == 0 and "WEIGHTS missing (information)" in r.stdout
