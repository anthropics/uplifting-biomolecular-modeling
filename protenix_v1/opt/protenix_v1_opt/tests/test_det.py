"""The deterministic recipe (det.py): the kit's detpatch files, refused once scatter_utils is imported, the modules swapped in-process.
The swap itself needs torch (the detpatch files import it) — that part runs where torch is installed (the stack image), else skips."""
import os
import sys
import types

import pytest

from .conftest import KIT
from protenix_v1_opt import det as D
from protenix_v1_opt import kit as K


def test_recipe_constants():
    assert D.ENV == {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PROTENIX_DET_SCATTER": "1"}
    assert D.MODULES == ("protenix.utils.det_segment_reduce", "protenix.utils.scatter_utils")
    assert [os.path.basename(p) for p in K.DETPATCH_RELPATHS] == ["det_segment_reduce.py", "scatter_utils.py"]
    for rel in K.DETPATCH_RELPATHS:
        assert os.path.isfile(os.path.join(KIT, rel))


def test_refused_once_the_stock_module_is_imported(monkeypatch):
    monkeypatch.setattr(D, "_REPORT", None)
    monkeypatch.setitem(sys.modules, D.STOCK_MODULE, types.ModuleType(D.STOCK_MODULE))
    with pytest.raises(RuntimeError, match="already imported"):
        D.install(KIT)


def test_swap_in_process(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    monkeypatch.setattr(D, "_REPORT", None)
    for m in list(sys.modules):
        if m == "protenix" or m.startswith("protenix."):
            monkeypatch.delitem(sys.modules, m)
    (tmp_path / "protenix" / "utils").mkdir(parents=True)
    (tmp_path / "protenix" / "__init__.py").write_text("")
    (tmp_path / "protenix" / "utils" / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    before = {k: os.environ.get(k) for k in D.ENV}
    try:
        rep = D.install(KIT)
        assert rep["installed"] and rep["det_scatter_active"] is True
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        mod = sys.modules[D.STOCK_MODULE]
        assert os.path.realpath(mod.__file__) == os.path.realpath(os.path.join(KIT, K.DETPATCH_RELPATHS[1]))
        import protenix.utils
        assert protenix.utils.scatter_utils is mod
    finally:                                                  # the recipe's variables are process state: leave the environment as found
        for k, v in before.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
