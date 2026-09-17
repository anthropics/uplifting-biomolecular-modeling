"""`ckpt_mmap` (cells/of3_ckptmmap.py + cells/ckpt_mmap.py): the engine's `load_checkpoint` text (sliced from stock/src: 0.4.1 `torch.load(path)`)
run against a real checkpoint file written by torch.save — the port's dict equals the engine's (same keys, dtypes, devices, every element), the
directory branch and a file torch cannot map go to the engine's own call (counted), the by-name importer's global is re-pointed through a real
import of a file-backed package, a text the port does not re-state is refused by name; the kit wiring (every kit line but off)."""
import ast
import hashlib
import importlib
import os
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")

from openfold3_opt.tests import _stubs

HOME = _stubs.tree_home()
SRC = os.path.join(HOME, "stock", "src", "openfold3", "core", "utils", "checkpoint_loading_utils.py")


def _live():
    g = globals()
    g["_autoload"], g["modes"], g["registry"], g["stack"] = (importlib.import_module("openfold3_opt." + n) for n in ("_autoload", "modes", "registry", "stack"))
    g["core"] = importlib.import_module("openfold3_opt.cells.of3_ckptmmap")
    g["cell"] = importlib.import_module("openfold3_opt.cells.ckpt_mmap")


_live()


@pytest.fixture(autouse=True)
def live_modules():
    _live()
    core.uninstall()
    yield
    core.uninstall()


def _fn_text(path, name):
    src = open(path, encoding="utf-8").read(); lines = src.splitlines(keepends=True)
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return "".join(lines[(n.decorator_list[0].lineno if n.decorator_list else n.lineno) - 1:n.end_lineno])
    raise AssertionError(name)


@pytest.fixture()
def engine_pkg(tmp_path, monkeypatch):
    """A file-backed package `nfc_of3` with `ckpt_utils` (the ENGINE'S load_checkpoint text) and `runner` (imports it by name), a checkpoint file."""
    d = tmp_path / "nfc_of3"; d.mkdir()
    (d / "__init__.py").write_text("")
    (d / "ckpt_utils.py").write_text("from pathlib import Path\nimport torch\n\ndef load_model_state_dict_from_ds_checkpoint(p):\n    return {'ds_dir': str(p)}\n\n\n" + textwrap.dedent(_fn_text(SRC, "load_checkpoint")))
    (d / "runner.py").write_text("from nfc_of3.ckpt_utils import load_checkpoint\n\ndef setup(p):\n    return load_checkpoint(p)\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    for n in [k for k in sys.modules if k.startswith("nfc_of3")]:
        del sys.modules[n]
    ckpt = {"a.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4), "b.bias": torch.tensor([1.5, -2.0], dtype=torch.float64), "n.count": torch.tensor(7)}
    f = tmp_path / "model.pt"; torch.save(ckpt, f)
    legacy = tmp_path / "legacy.pt"; torch.save(ckpt, legacy, _use_new_zipfile_serialization=False)
    saved = {k: getattr(core, k) for k in core.CONFIGURABLE}
    core.configure(ENV="NFC_CKPT_MMAP", PREFIX="[nfc/ckpt_mmap]", M_CKPT="nfc_of3.ckpt_utils", REBIND=("nfc_of3.runner",), DIGESTS=dict(saved["DIGESTS"]))
    class NS: pass
    ns = NS(); ns.tmp, ns.ckpt, ns.file, ns.legacy = tmp_path, ckpt, f, legacy
    yield ns
    core.uninstall(); core.configure(**saved)
    for n in [k for k in sys.modules if k.startswith("nfc_of3")]:
        del sys.modules[n]


def _same_dict(a, b):
    assert list(a) == list(b)
    for k in a:
        x, y = a[k], b[k]
        assert x.dtype == y.dtype and x.device == y.device and x.shape == y.shape and torch.equal(x, y), k


def test_the_mapped_read_hands_the_engine_the_same_dict_and_the_by_name_importer_is_repointed(engine_pkg):
    from pathlib import Path
    st = core.install({"NFC_CKPT_MMAP": "1"})
    assert st["state"] == "armed" and core.FINDER in sys.meta_path
    R = importlib.import_module("nfc_of3.runner"); U = sys.modules["nfc_of3.ckpt_utils"]
    assert getattr(U.load_checkpoint, core.MARK, False) and R.load_checkpoint is U.load_checkpoint and st["state"] == "on" and st["variant"] == "v041"
    orig = U.load_checkpoint.__wrapped__
    got, ref = R.setup(Path(engine_pkg.file)), orig(Path(engine_pkg.file))
    _same_dict(got, ref); _same_dict(got, engine_pkg.ckpt)
    assert st["loads"] == 1 and st["mmap"] == 1 and st["fallback"] == 0
    d = engine_pkg.tmp / "ds_ckpt"; d.mkdir()
    assert R.setup(d) == {"ds_dir": str(d)} and st["mmap"] == 1 and st["loads"] == 2                # a directory checkpoint: the engine's branch
    got2 = R.setup(Path(engine_pkg.legacy))                                                         # the legacy (non-zip) format: torch cannot map it -> the engine's own call, counted
    _same_dict(got2, engine_pkg.ckpt); assert st["fallback"] == 1 and st["mmap"] == 1
    line = core.census_line()
    for w in ("LEVER name=ckpt_mmap", "state=on", "loads=3", "mmap=1", "fallback=1", "load_s="):
        assert w in line, (w, line)
    with pytest.raises(ValueError):
        R.setup(engine_pkg.tmp / "missing.pt")                                                      # the engine's own error for a path that is neither


def test_install_after_the_engine_imported_its_reader_patches_in_place(engine_pkg):
    R = importlib.import_module("nfc_of3.runner"); U = sys.modules["nfc_of3.ckpt_utils"]
    orig = U.load_checkpoint
    st = core.install({"NFC_CKPT_MMAP": "1"})
    assert st["state"] == "on" and U.load_checkpoint is not orig and R.load_checkpoint is U.load_checkpoint and U.load_checkpoint.__wrapped__ is orig
    core.uninstall()
    assert U.load_checkpoint is orig and R.load_checkpoint is orig


def test_switch_words_and_a_text_the_port_does_not_restate(engine_pkg):
    assert core.requested({}) is False and core.requested({"NFC_CKPT_MMAP": "1"}) is True
    with pytest.raises(ValueError):
        core.requested({"NFC_CKPT_MMAP": "on"})
    st = core.install({"NFC_CKPT_MMAP": "2"})
    assert st["state"] == "refused" and st["reason"].startswith("bad_word:") and core.FINDER not in sys.meta_path
    core.uninstall()
    core.configure(DIGESTS={"0" * 16: "v041"})
    st = core.install({"NFC_CKPT_MMAP": "1"})
    importlib.import_module("nfc_of3.runner"); U = sys.modules["nfc_of3.ckpt_utils"]
    assert st["state"] == "refused" and st["reason"].startswith("digest:load_checkpoint=") and not getattr(U.load_checkpoint, core.MARK, False)


def test_the_kits_digest_pin_is_the_engines_text_and_every_kit_line_but_off_carries_the_lever():
    d = hashlib.sha256(textwrap.dedent(_fn_text(SRC, "load_checkpoint")).encode("utf-8")).hexdigest()[:16]
    assert cell._core.DIGESTS == {d: "v041"} and cell._core.KWARGS["v041"] == {} and "torch.load(ckpt_path)" in _fn_text(SRC, "load_checkpoint")
    for key, ln in modes.LINES.items():
        carries = key[0] != "off"
        assert (ln.env.get(cell.ENV) == "1") is carries and ("ckpt_mmap" in ln.levers) is carries, key
    lv = registry.LEVERS["ckpt_mmap"]
    assert (lv.kit, lv.cls, lv.tier, lv.env_keys, lv.target) == ("cells", registry.FORWARD, registry.EXACT, modes.CKPT_MMAP_ENVS, registry.T_CKPT)
    assert set(lv.modes) == {"exact", "fast", "big/resident", "big/tp"} and lv.marker == "[openfold3-opt/ckpt_mmap] installed" and lv.source == "opt/openfold3_opt/cells/ckpt_mmap.py"
    assert registry.tested_sm("ckpt_mmap") == ("sm90",) and "ckpt_mmap" in registry.ARCH_NOTES
    assert modes.CKPT_MMAP_ENVS == (cell.ENV,) and modes.CKPT_MMAP_ENVS in modes.CELL_FAMILIES and modes.CKPT_MMAP_ENVS in modes.ACTIVATION_FAMILIES
    assert "ckpt_mmap" in modes.ACTIVATION_CELLS and "ckpt_mmap" in stack._PROBES and modes.LEVER_SWITCHES["ckpt_mmap"] == modes.CKPT_MMAP_ENVS
    assert cell.ENV in _autoload.DECLARED and cell._core.M_CKPT == registry.T_CKPT and cell._core.REBIND == ("openfold3.entry_points.experiment_runner",)
    r = modes.resolve("exact", HOME, environ={modes.ENV_LEVERS_OFF: "ckpt_mmap"}, n_tokens=400)
    assert "ckpt_mmap" not in r.levers and cell.ENV not in r.exports and "hostfeat" in r.levers
