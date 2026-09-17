"""The conf_dtype lever: the confidence phase's dtype word per mode (bf16 on fast / big, fp32 = upstream on exact), the caller's preset
or `--conf-dtype` winning by name (source=env), the ONE wrapped seam (AuxiliaryHeadsAllAtom.forward under bf16 autocast with
pairformer_dtype=bfloat16), the row-sharded line's dtype pair, the census words, and the tp launcher passing the knob to its ranks."""
import os
import sys
import types

import pytest

from openfold3_ob0_opt import cli, conf_dtype as CD, modes, report, stack, tp
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
    """fast, big/resident, big/tp: bf16 from the mode, the lever in the call's set; exact: upstream's fp32, no lever; off: nothing."""
    for mode, line in (("fast", None), ("big", None), ("big", "tp")):
        r = _res(mode, line=line)
        assert not r.conflicts, (mode, line, r.conflicts)
        assert r.exports[modes.ENV_CONF_DTYPE] == "bf16" and r.exports[modes.ENV_CONF_DTYPE_SOURCE] == "mode", (mode, line)
        assert r.conf_dtype == "conf_dtype=bf16 source=mode" and "conf_dtype" in r.levers, (mode, line)
    r = _res("exact")
    assert not r.conflicts and r.exports[modes.ENV_CONF_DTYPE] == "fp32" and r.conf_dtype == "conf_dtype=fp32 source=mode" and "conf_dtype" not in r.levers
    r = _res("off")
    assert modes.ENV_CONF_DTYPE not in r.exports and r.conf_dtype is None
    assert "conf_dtype" in LEVERS and LEVERS["conf_dtype"].env_keys == (modes.ENV_CONF_DTYPE,) and "conf_dtype" in stack._PROBES
    assert set(LEVERS["conf_dtype"].modes) == {"fast", "big/resident", "big/tp"}


def test_a_preset_word_wins_and_is_named_not_refused():
    r = _res("fast", {modes.ENV_CONF_DTYPE: "fp32", modes.ENV_Z_DTYPE: "fp32"})                      # upstream's pair of roll-out dtypes on the fast line: adopted, source=env
    assert not r.conflicts and r.exports[modes.ENV_CONF_DTYPE] == "fp32" and r.conf_dtype == "conf_dtype=fp32 source=env" and "conf_dtype" not in r.levers
    r = _res("fast", {modes.ENV_CONF_DTYPE: "fp32"})                                                 # fp32 heads over the line's bf16 hand-off: refused by name (test_z_dtype)
    assert r.conflicts and modes.ENV_Z_DTYPE in r.conflicts[0]
    r = _res("exact", {modes.ENV_CONF_DTYPE: "bf16"})
    assert not r.conflicts and r.conf_dtype == "conf_dtype=bf16 source=env" and "conf_dtype" not in r.levers      # the exact line carries no such lever: the word applies, the line's lever set is the line's
    r = _res("big", {modes.ENV_CONF_DTYPE: "bf16", modes.ENV_CONF_DTYPE_SOURCE: "mode"}, line="tp")            # a rank: the launcher's words passed down (tp.RANK_KNOBS) keep their source
    assert not r.conflicts and r.conf_dtype == "conf_dtype=bf16 source=mode" and "conf_dtype" in r.levers
    r = _res("fast", {modes.ENV_CONF_DTYPE: "half"})
    assert r.conflicts and modes.ENV_CONF_DTYPE in r.conflicts[0]
    line = report.activation_line({"active": True, "mode": "fast", "conf_dtype": "conf_dtype=bf16 source=mode", "n_gpu": 1})
    assert " conf_dtype=bf16 source=mode " in line + " "


def test_cli_flag_is_the_variables_spelling(monkeypatch):
    monkeypatch.delenv(modes.ENV_CONF_DTYPE, raising=False); monkeypatch.delenv(modes.ENV_CONF_DTYPE_SOURCE, raising=False)
    cli.apply_conf_dtype_flag(types.SimpleNamespace(conf_dtype=None))
    assert modes.ENV_CONF_DTYPE not in os.environ
    cli.apply_conf_dtype_flag(types.SimpleNamespace(conf_dtype="fp32"))
    assert os.environ[modes.ENV_CONF_DTYPE] == "fp32" and os.environ[modes.ENV_CONF_DTYPE_SOURCE] == "env"
    with pytest.raises(ValueError):
        cli.apply_conf_dtype_flag(types.SimpleNamespace(conf_dtype="bf16"))                                        # disagrees with the preset value: refused by name
    p = cli.build_parser() if hasattr(cli, "build_parser") else None
    if p is not None:
        a = p.parse_args(["pred", "--conf-dtype", "bf16", "--query-json", "q.json", "--output-dir", "o"])
        assert a.conf_dtype == "bf16"


def test_rank_env_keeps_the_knob(tmp_path):
    from openfold3_ob0_opt import manifest
    base = {modes.ENV_CONF_DTYPE: "fp32", modes.ENV_CONF_DTYPE_SOURCE: "env", "OPENFOLD3_OB0_OPT_ROLLOUT": "bf16", "OPENFOLD3_OB0_OPT_HOME": HOME, manifest.RECORDS_ENV: str(tmp_path)}
    env = tp.rank_env(base, 1, 2, "1", 29500, 64)
    assert env[modes.ENV_CONF_DTYPE] == "fp32" and env[modes.ENV_CONF_DTYPE_SOURCE] == "env" and "OPENFOLD3_OB0_OPT_ROLLOUT" not in env   # the knob passes; other route variables do not (the rank takes its route from argv)
    from openfold3_ob0_opt import _autoload
    assert {modes.ENV_CONF_DTYPE, modes.ENV_CONF_DTYPE_SOURCE} <= set(_autoload.DECLARED)                                                       # declared package variables: a rank (or any interpreter) holding them activates


class _Heads:                                                  # the shape of head_modules.AuxiliaryHeadsAllAtom.forward: pairformer_dtype defaults to float32
    def forward(self, batch, si_input, output, pairformer_dtype=None, **kw):
        import torch
        return {"pairformer_dtype": pairformer_dtype if pairformer_dtype is not None else torch.float32,
                "autocast_gpu_dtype": torch.get_autocast_gpu_dtype() if hasattr(torch, "get_autocast_gpu_dtype") else None, "kw": kw}


@pytest.fixture
def heads_module(monkeypatch):
    mod = types.ModuleType(CD.TARGET)
    mod.AuxiliaryHeadsAllAtom = type("AuxiliaryHeadsAllAtom", (), {"forward": _Heads.forward})
    monkeypatch.setitem(sys.modules, CD.TARGET, mod)
    yield mod
    CD.uninstall()


def test_bf16_wraps_the_one_forward(heads_module, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv(modes.ENV_CONF_DTYPE, "bf16"); monkeypatch.setenv(modes.ENV_CONF_DTYPE_SOURCE, "mode")
    CD.uninstall()
    st = CD.install()
    assert st["installed"] and st["word"] == "bf16" and st["source"] == "mode"
    fwd = heads_module.AuxiliaryHeadsAllAtom.forward
    assert getattr(fwd, "_of3ob0_conf_dtype", False) and stack._p_conf_dtype(HOME) is True
    out = heads_module.AuxiliaryHeadsAllAtom().forward({}, None, {}, chunk_size=4)
    assert out["pairformer_dtype"] == torch.bfloat16 and out["kw"] == {"chunk_size": 4}                              # the caller passed none: bf16; other kwargs untouched
    out = heads_module.AuxiliaryHeadsAllAtom().forward({}, None, {}, pairformer_dtype=torch.float32)
    assert out["pairformer_dtype"] == torch.float32                                                                    # an explicit caller is honoured and counted
    line = CD.census_line()
    assert "LEVER name=conf_dtype state=on dtype=bf16 source=mode calls=2 explicit=1" in line, line
    assert CD.head_dtypes(torch.float32) == (torch.bfloat16, torch.bfloat16)                                         # the row-sharded line's pair
    assert CD.fallbacks() == {}


def test_fp32_patches_nothing(heads_module, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv(modes.ENV_CONF_DTYPE, "fp32"); monkeypatch.setenv(modes.ENV_CONF_DTYPE_SOURCE, "env")
    CD.uninstall()
    orig = heads_module.AuxiliaryHeadsAllAtom.forward
    st = CD.install()
    assert not st["installed"] and st["word"] == "fp32" and st["source"] == "env"
    assert heads_module.AuxiliaryHeadsAllAtom.forward is orig and stack._p_conf_dtype(HOME) is False                 # upstream's forward, byte-untouched
    assert CD.head_dtypes(torch.float32) == (torch.float32, torch.float32) and CD.head_dtypes(torch.bfloat16) == (torch.bfloat16, torch.float32)
    assert "state=off dtype=fp32 source=env" in CD.census_line()
    monkeypatch.delenv(modes.ENV_CONF_DTYPE); CD.uninstall()
    assert CD.setting() == (None, None) and CD.install()["word"] == "fp32" and not CD.STATE["installed"]           # unset: upstream's word, no lever


def test_the_seam_exists_in_stock():
    """The wrapped attribute is upstream 0.5.0's: head_modules.AuxiliaryHeadsAllAtom.forward takes pairformer_dtype (default float32)."""
    import ast
    src = open(_stubs.stock_src("openfold3", "core", "model", "heads", "head_modules.py")).read()
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "AuxiliaryHeadsAllAtom")
    fwd = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    names = [a.arg for a in fwd.args.args + fwd.args.kwonlyargs]
    assert "pairformer_dtype" in names
    model = open(_stubs.stock_src("openfold3", "projects", "of3_all_atom", "model.py")).read()
    assert "cast_dtype = torch.float32 if self.training else si_trunk.dtype" in model and "return s_input.float(), s.float(), z.float()" in model
