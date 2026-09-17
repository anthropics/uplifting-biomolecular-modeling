"""The z_dtype lever: the trunk hand-off's dtype word per mode (bf16 on fast / big = OpenFold3 0.4.1's hand-off, fp32 = upstream 0.5.0's
`.float()` on exact), the caller's preset or `--z-dtype` winning by name (source=env), the hand-off statement following the word on CPU
tensors (the kit's own trunks) and through the wrapped OpenFold3.run_trunk (upstream's class), the census words, the tp launcher passing the
knob to its ranks, the refusal of fp32 heads over a bf16 hand-off, and the seams' existence in stock and in the kit's trunks."""
import os
import sys
import types

import pytest
import torch

from openfold3_ob0_opt import cli, modes, stack, tp, z_dtype as ZD
from openfold3_ob0_opt.registry import LEVERS
from openfold3_ob0_opt.tests import _stubs

HERE = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
TP_ENV = {"OF3TP_RANK": "0", "OF3TP_WORLD": "2", "OPENFOLD3_OB0_OPT_N_GPU": "2"}


def _res(mode, environ=None, line=None):
    environ = dict(environ or {})
    if line == "tp":
        environ.update(TP_ENV)
    return modes.resolve(mode, HOME, environ=environ, n_gpu=2 if line == "tp" else None)


def test_the_lines_words():
    for mode, line in (("fast", None), ("big", None), ("big", "tp")):
        r = _res(mode, line=line)
        assert not r.conflicts, (mode, line, r.conflicts)
        assert r.exports[modes.ENV_Z_DTYPE] == "bf16" and r.exports[modes.ENV_Z_DTYPE_SOURCE] == "mode" and r.z_dtype == "z_dtype=bf16 source=mode", (mode, line)
        assert "z_dtype" in r.levers, (mode, line, r.levers)
    r = _res("exact")
    assert not r.conflicts and r.exports[modes.ENV_Z_DTYPE] == "fp32" and r.z_dtype == "z_dtype=fp32 source=mode" and "z_dtype" not in r.levers
    r = _res("off")
    assert modes.ENV_Z_DTYPE not in r.exports and r.z_dtype is None
    assert "z_dtype" in LEVERS and LEVERS["z_dtype"].env_keys == (modes.ENV_Z_DTYPE,) and "z_dtype" in stack._PROBES and "z_dtype" in stack.ACTIVATION_MODULES


def test_a_preset_word_wins_and_is_named():
    r = _res("fast", {modes.ENV_Z_DTYPE: "fp32", modes.ENV_CONF_DTYPE: "fp32"})                       # upstream's pair of words on the fast line: adopted, source=env, both levers leave the set
    assert not r.conflicts and r.z_dtype == "z_dtype=fp32 source=env" and "z_dtype" not in r.levers and "conf_dtype" not in r.levers
    r = _res("fast", {modes.ENV_Z_DTYPE: "fp32"})                                                      # fp32 hand-off with the line's bf16 heads: fine (0.5.0 hand-off, explicit bf16 heads)
    assert not r.conflicts and r.z_dtype == "z_dtype=fp32 source=env" and r.conf_dtype == "conf_dtype=bf16 source=mode"
    r = _res("exact", {modes.ENV_Z_DTYPE: "bf16"})                                                     # bf16 hand-off under exact's fp32 heads: refused by name (cast_dtype follows the hand-off)
    assert r.conflicts and modes.ENV_CONF_DTYPE in r.conflicts[0] and modes.ENV_Z_DTYPE in r.conflicts[0]
    r = _res("big", {modes.ENV_Z_DTYPE: "bf16", modes.ENV_Z_DTYPE_SOURCE: "mode"}, line="tp")         # a rank: the launcher's word and source pass through
    assert not r.conflicts and r.z_dtype == "z_dtype=bf16 source=mode"
    r = _res("fast", {modes.ENV_Z_DTYPE: "half"})
    assert r.conflicts and modes.ENV_Z_DTYPE in r.conflicts[0]


def test_the_cli_flag_sets_the_variable(monkeypatch):
    monkeypatch.delenv(modes.ENV_Z_DTYPE, raising=False); monkeypatch.delenv(modes.ENV_Z_DTYPE_SOURCE, raising=False)
    p = cli.build_parser() if hasattr(cli, "build_parser") else None
    a = types.SimpleNamespace(conf_dtype=None, z_dtype="fp32") if p is None else p.parse_args(["pred", "--z-dtype", "fp32", "--query-json", "q.json", "--output-dir", "o"])
    cli.apply_conf_dtype_flag(a)
    assert os.environ[modes.ENV_Z_DTYPE] == "fp32" and os.environ[modes.ENV_Z_DTYPE_SOURCE] == "env"
    monkeypatch.setenv(modes.ENV_Z_DTYPE, "bf16")
    with pytest.raises(ValueError):
        cli.apply_conf_dtype_flag(types.SimpleNamespace(conf_dtype=None, z_dtype="fp32"))


def test_the_rank_keeps_the_knob():
    from openfold3_ob0_opt import _autoload, manifest
    base = {modes.ENV_Z_DTYPE: "fp32", modes.ENV_Z_DTYPE_SOURCE: "env", "OPENFOLD3_OB0_OPT_ROLLOUT": "bf16", "OPENFOLD3_OB0_OPT_HOME": HOME, manifest.RECORDS_ENV: "/tmp/x"}
    env = tp.rank_env(base, 1, 2, "1", 29500, 16)
    assert env[modes.ENV_Z_DTYPE] == "fp32" and env[modes.ENV_Z_DTYPE_SOURCE] == "env" and "OPENFOLD3_OB0_OPT_ROLLOUT" not in env
    assert {modes.ENV_Z_DTYPE, modes.ENV_Z_DTYPE_SOURCE} <= set(_autoload.DECLARED)


def test_the_handoff_dtype_follows_the_word(monkeypatch):
    """The kit's own trunks' statement: bf16 → 0.4.1's `return s_input, s, z` (the very tensors), fp32 → 0.5.0's `.float()` copies."""
    ZD.uninstall()
    s_input, s, z = torch.ones(2, 3, dtype=torch.bfloat16), torch.ones(2, 3, dtype=torch.bfloat16), torch.ones(2, 2, 4, dtype=torch.bfloat16)
    monkeypatch.setenv(modes.ENV_Z_DTYPE, "bf16"); monkeypatch.setenv(modes.ENV_Z_DTYPE_SOURCE, "mode")
    out = ZD.handoff(s_input, s, z)
    assert out[0] is s_input and out[1] is s and out[2] is z and ZD.STATE["in"] == "bfloat16" and ZD.STATE["out"] == "bfloat16" and ZD.STATE["handoffs"] == 1
    monkeypatch.setenv(modes.ENV_Z_DTYPE, "fp32"); ZD.uninstall()
    out = ZD.handoff(s_input, s, z)
    assert all(t.dtype == torch.float32 for t in out) and ZD.STATE["out"] == "float32"
    monkeypatch.delenv(modes.ENV_Z_DTYPE); ZD.uninstall()
    assert ZD.word() == "fp32" and all(t.dtype == torch.float32 for t in ZD.handoff(s_input, s, z))          # unset: upstream's statement
    ZD.uninstall()


def test_the_wrapped_run_trunk_hands_bf16(monkeypatch):
    """Upstream's class route: the wrapper the lever puts on OpenFold3.run_trunk hands the `.float()` outputs on as bfloat16 — exact
    values — and install() under bf16 registers that ONE patch site (fail-closed by name in the core); under fp32 nothing is patched."""
    ZD.uninstall()

    class OpenFold3:                                                                               # upstream's statement, shape only
        def run_trunk(self, batch, num_cycles, inplace_safe=False):
            z = torch.full((1, 2, 2, 4), 1.0 / 3.0, dtype=torch.bfloat16)
            return z[..., 0, :].clone().float(), z[..., 0, :].clone().float(), z.float()
    wrapped = ZD._make_run_trunk(OpenFold3.run_trunk)
    assert getattr(wrapped, "_of3ob0_z_dtype", False) and wrapped.__wrapped__ is OpenFold3.run_trunk
    s_input, s, z = wrapped(OpenFold3(), {}, 1)
    assert z.dtype == torch.bfloat16 and s.dtype == torch.bfloat16 and s_input.dtype == torch.bfloat16
    assert torch.equal(z.float(), torch.full((1, 2, 2, 4), 1.0 / 3.0, dtype=torch.bfloat16).float())   # bf16 -> fp32 -> bf16: the trunk's own values
    assert ZD.STATE["handoffs"] == 1 and ZD.STATE["in"] == "float32" and ZD.STATE["out"] == "bfloat16"
    monkeypatch.setenv(modes.ENV_Z_DTYPE, "bf16"); monkeypatch.setenv(modes.ENV_Z_DTYPE_SOURCE, "mode")
    st = ZD.install()
    assert st["installed"] and st["word"] == "bf16" and ZD.PATCH is not None and (ZD.PATCH.target, ZD.PATCH.attr) == (ZD.TARGET, ZD.ATTR)
    assert "state=on dtype=bf16 source=mode" in ZD.census_line()
    ZD.uninstall()
    monkeypatch.setenv(modes.ENV_Z_DTYPE, "fp32")
    assert ZD.install()["installed"] is False and ZD.PATCH is None and "state=off dtype=fp32" in ZD.census_line()   # upstream's word: nothing patched
    ZD.uninstall()


def test_the_seams_exist():
    """Upstream 0.5.0's statement and the kit's two trunks end in the hand-off the lever governs."""
    model = open(_stubs.stock_src("openfold3", "projects", "of3_all_atom", "model.py")).read()
    assert "class OpenFold3(" in model and "def run_trunk(" in model and "return s_input.float(), s.float(), z.float()" in model
    assert "cast_dtype = torch.float32 if self.training else si_trunk.dtype" in model              # the confidence phase follows the hand-off (the conf/z coupling refusal rests on it)
    trunk = open(os.path.join(HERE, "..", "tp_rowpair", "trunk.py")).read()
    assert "ZD.handoff(s_input, out.s, out.z)" in trunk
    port = open(os.path.join(HOME, "opt", "forward", "offload", "of3o", "of3_offload.py")).read()
    assert "ZD.handoff(s_input, s, z)" in port and "return s_input.float(), s.float(), z.float()" not in port
