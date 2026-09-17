"""The weights boot gate (kit.frozen_weights_check), on every route (cli pred before anything runs, the activation gates, configs/h100.env's
probe): a MISSING root / checkpoint / data cache — what the upstream's boot-time download fallback would otherwise fetch
(runner/inference.py:291-347) — is refused by name; a PRESENT checkpoint is always accepted, digested against stock.checkpoint_sha256 and
announced once per process as `weights=<name> sha256=<12> (pinned)` or `weights sha256=<12> NOT PINNED — …`, and the run proceeds."""
import os
import sys

import pytest

from protenix_v1_opt import kit as K
from protenix_v1_opt import report as R
from protenix_v1_opt import stack


def _pin():
    return stack.pins()["stock"]


def _url_basenames():
    import importlib.metadata, importlib.util
    try:
        p = K.stock_url_module()
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("the stock package `protenix` is not installed here (the gate reads its dependency_url module)")
    ms = importlib.util.spec_from_file_location("u", p); m = importlib.util.module_from_spec(ms); ms.loader.exec_module(m)
    return {n: os.path.basename(m.URL[n]) for n in _pin()["data_files"] if n in m.URL}


def _root(tmp_path, ckpt_bytes=4096, caches=True, payload=None):
    st = _pin(); root = tmp_path / "w"; ck = root / st["checkpoint"]; ck.parent.mkdir(parents=True)
    with open(ck, "wb") as fh:
        if payload is not None:
            fh.write(payload)
        else:
            fh.truncate(ckpt_bytes)                                            # a stand-in checkpoint: present, not the pinned digest
    if caches:
        (root / "common").mkdir()
        for n, base in _url_basenames().items():
            (root / "common" / base).write_bytes(b"x")
    return str(root)


def test_missing_root_is_refused_by_name(monkeypatch):
    monkeypatch.delenv(_pin()["root_env"], raising=False)
    with pytest.raises(K.FrozenWeightsError, match="is not a directory — the upstream would download the weights at boot"):
        K.frozen_weights_check()


def test_missing_checkpoint_is_refused_by_name(tmp_path):
    with pytest.raises(K.FrozenWeightsError, match="is absent — the upstream would download the checkpoint at boot"):
        K.frozen_weights_check(str(tmp_path))


def test_an_unknown_checkpoint_runs_and_is_named_not_pinned(tmp_path, capsys, monkeypatch):
    """A user checkpoint is ALWAYS accepted — digested, named NOT PINNED once per process, and the run proceeds (no flag)."""
    monkeypatch.setattr(K, "_ANNOUNCED", set())
    root = _root(tmp_path, ckpt_bytes=1234)
    w = K.frozen_weights_check(root)
    assert w["pinned"] is False and w["checkpoint_bytes"] == 1234 and len(w["sha256"]) == 64
    err = capsys.readouterr().err.splitlines()
    assert err == [f"{R.PREFIX} weights sha256={w['sha256'][:12]} NOT PINNED — the kit's numerics and speed statements hold for the pinned weights only"]
    K.frozen_weights_check(root)                                             # the same digest is announced once per process
    assert capsys.readouterr().err == ""


def test_the_pinned_checkpoint_is_recognised_by_digest(tmp_path, capsys, monkeypatch):
    import hashlib
    monkeypatch.setattr(K, "_ANNOUNCED", set())
    payload = b"the pinned checkpoint, for the test"
    monkeypatch.setitem(_pin(), "checkpoint_sha256", hashlib.sha256(payload).hexdigest())   # the pin names this file's digest
    root = _root(tmp_path, payload=payload)
    w = K.frozen_weights_check(root)
    assert w["pinned"] is True and w["weights"] == _pin()["model_name"]
    assert capsys.readouterr().err.splitlines() == [f"{R.PREFIX} weights={_pin()['model_name']} sha256={hashlib.sha256(payload).hexdigest()[:12]} (pinned)"]


def test_the_checkpoint_the_stock_arguments_resolve_to_is_the_one_gated(tmp_path, capsys, monkeypatch):
    """`--model_name <n>` makes the upstream load <root>/checkpoint/<n>.pt: that file is the one digested (absent = refused by name;
    present = accepted, not pinned — the kit's statements are the default model's)."""
    monkeypatch.setattr(K, "_ANNOUNCED", set())
    root = _root(tmp_path)
    assert K.stock_model_name([]) == _pin()["model_name"] and K.stock_model_name(["--model_name", "protenix_mini_default_v0.5.0"]) == "protenix_mini_default_v0.5.0"
    assert K.stock_model_name(["--model_name=protenix_tiny_default_v0.5.0", "--use_msa", "true"]) == "protenix_tiny_default_v0.5.0"
    with pytest.raises(K.FrozenWeightsError, match="protenix_mini_default_v0.5.0.pt is absent"):
        K.frozen_weights_check(root, argv=["--model_name", "protenix_mini_default_v0.5.0"])
    (tmp_path / "w" / "checkpoint" / "protenix_mini_default_v0.5.0.pt").write_bytes(b"mini")
    w = K.frozen_weights_check(root, argv=["--model_name", "protenix_mini_default_v0.5.0"])
    assert w["pinned"] is False and w["weights"] == "protenix_mini_default_v0.5.0" and w["checkpoint"].endswith("protenix_mini_default_v0.5.0.pt")
    assert "NOT PINNED" in capsys.readouterr().err


def test_pred_proceeds_past_the_gate_on_an_unknown_checkpoint(tmp_path, monkeypatch, capsys):
    """The exit-0 path is unchanged: `pred` on a checkpoint that is not the pinned one prints the weights line and goes on to the route (here the stock
    route's launcher, stubbed), never NOT ACTIVE."""
    from protenix_v1_opt import cli
    monkeypatch.setattr(K, "_ANNOUNCED", set())
    monkeypatch.setenv(_pin()["root_env"], _root(tmp_path))
    seen = {}
    monkeypatch.setattr(cli, "_run_stock", lambda stock_args, det: seen.setdefault("rc", 0))
    rc = cli.main(["pred", "--mode", "off", "--input", "x.json", "--out_dir", str(tmp_path / "o")])
    err = capsys.readouterr().err
    assert rc == 0 and seen == {"rc": 0} and " NOT PINNED — the kit's numerics and speed statements hold for the pinned weights only" in err and "NOT ACTIVE" not in err


def test_missing_cache_is_refused_by_name(tmp_path):
    _url_basenames()                                                         # the cache names are the stock package's URL module's: skipped by name without it, like every cache test here
    root = _root(tmp_path, caches=False)
    with pytest.raises(K.FrozenWeightsError, match="is absent — the upstream would download it at boot"):
        K.frozen_weights_check(root)


def test_the_pinned_weights_pass(tmp_path, monkeypatch):
    root = _root(tmp_path)
    w = K.frozen_weights_check(root)
    assert w["root"] == root and w["checkpoint_bytes"] == 4096 and w["pinned"] is False and set(w["caches"]) == set(_url_basenames())
    monkeypatch.setenv(_pin()["root_env"], root)
    assert K.frozen_weights_check()["root"] == root                  # the root from the environment (stock.root_env)


def test_the_probe_prints_the_line_and_exits_3(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(K, "_ANNOUNCED", set())
    monkeypatch.setenv(_pin()["root_env"], str(tmp_path))
    with pytest.raises(SystemExit) as ex:
        K.frozen_weights_main()
    assert ex.value.code == R.EXIT_NOT_ACTIVE == 3
    err = capsys.readouterr().err
    assert err.startswith(R.PREFIX + " NOT ACTIVE: frozen weights: ") and "would download the checkpoint at boot" in err
    monkeypatch.setenv(_pin()["root_env"], _root(tmp_path))
    K.frozen_weights_main()                                            # a present checkpoint: the weights line, no exit
    assert "weights sha256=" in capsys.readouterr().err


def test_the_gate_is_an_activation_gate_and_a_pred_gate():
    src = open(os.path.join(os.path.dirname(stack.__file__), "stack.py"), encoding="utf-8").read()
    assert "frozen_weights" in src[src.index("return G.run_gates(["):src.index("\n", src.index("return G.run_gates(["))]
    cli = open(os.path.join(os.path.dirname(stack.__file__), "cli.py"), encoding="utf-8").read()
    i = cli.index("def cmd_pred("); assert cli.index("K.frozen_weights_check(", i) < cli.index('if mode == "off":', i)   # before either route runs
    env = open(os.path.join(K.tree_home(), "configs", "h100.env"), encoding="utf-8").read()
    assert 'kit.frozen_weights_main()" || { return 3 2>/dev/null || exit 3; }' in env


def test_the_config_names_the_root_and_carries_no_default_for_it():
    """configs/h100.env: PROTENIX_ROOT_DIR is read from the environment — no baked path default; unset, the config refuses by name (rc 3)
    before the boot gate runs; the exported value is the caller's own."""
    env = open(os.path.join(K.tree_home(), "configs", "h100.env"), encoding="utf-8").read()
    assert "${PROTENIX_ROOT_DIR:-/" not in env and 'export PROTENIX_ROOT_DIR="$PROTENIX_ROOT_DIR"' in env      # no `${VAR:-/path}` default
    i = env.index('[ -n "${PROTENIX_ROOT_DIR:-}" ] || { echo "[protenix-v1-opt] NOT ACTIVE: PROTENIX_ROOT_DIR is not set')
    assert "return 3 2>/dev/null || exit 3; }" in env[i:env.index("\n", i)] and i < env.index("kit.frozen_weights_main()")


def test_a_missing_stock_distribution_is_a_named_refusal(tmp_path, monkeypatch, capsys):
    """m9: with the weights root present but the stock distribution absent, the gate refuses by name (no traceback): the pred entry prints
    the NOT ACTIVE line and exits 3."""
    import importlib.metadata
    root = _root(tmp_path)
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: (_ for _ in ()).throw(importlib.metadata.PackageNotFoundError(name)))
    with pytest.raises(K.FrozenWeightsError, match="is not installed in this interpreter — nothing to gate"):
        K.frozen_weights_check(root)
    from protenix_v1_opt import cli
    monkeypatch.setenv(_pin()["root_env"], root)
    rc = cli.main(["pred", "--mode", "off", "--input", "x.json", "--out_dir", str(tmp_path / "o")])
    err = capsys.readouterr().err
    assert rc == R.EXIT_NOT_ACTIVE == 3 and "Traceback" not in err and (R.PREFIX + " NOT ACTIVE: frozen weights: the stock package protenix is not installed") in err


def test_load_checkpoint_dir_is_honoured_by_the_resolver(tmp_path, capsys, monkeypatch):
    """`--load_checkpoint_dir <d>` makes the upstream load <d>/<model_name>.pt (runner/inference.py:338): that file is the one gated —
    the pinned digest there is named pinned, other bytes there are NOT PINNED and run, an absent file there is refused by name."""
    import hashlib
    monkeypatch.setattr(K, "_ANNOUNCED", set())
    payload = b"the pinned checkpoint, in a caller's directory"
    monkeypatch.setitem(_pin(), "checkpoint_sha256", hashlib.sha256(payload).hexdigest())
    root = _root(tmp_path)                                                   # the root's own checkpoint is a stand-in (not the pin's digest)
    d = tmp_path / "elsewhere"; d.mkdir()
    argv = ["--load_checkpoint_dir", str(d), "--use_msa", "true"]
    assert K.stock_checkpoint(root, argv) == str(d / (_pin()["model_name"] + ".pt")) and K.stock_checkpoint(root, []) == os.path.join(root, _pin()["checkpoint"])
    with pytest.raises(K.FrozenWeightsError, match="elsewhere/" + _pin()["model_name"] + r"\.pt is absent"):
        K.frozen_weights_check(root, argv=argv)
    (d / (_pin()["model_name"] + ".pt")).write_bytes(payload)
    w = K.frozen_weights_check(root, argv=argv)
    assert w["pinned"] is True and w["checkpoint"] == str(d / (_pin()["model_name"] + ".pt"))
    assert capsys.readouterr().err.splitlines() == [f"{R.PREFIX} weights={_pin()['model_name']} sha256={hashlib.sha256(payload).hexdigest()[:12]} (pinned)"]
    (d / (_pin()["model_name"] + ".pt")).write_bytes(b"someone else's weights")
    w = K.frozen_weights_check(root, argv=["--load_checkpoint_dir=" + str(d)])
    assert w["pinned"] is False and "NOT PINNED" in capsys.readouterr().err          # other bytes in the caller's directory: named, and the run proceeds
