"""workers — the stock --gpu-ids workers activate the kit's mode themselves (CPU: no spawn, no GPU): install rebinds the stock entry to a
picklable partial carrying mode / det / partial policy, the partial round-trips through pickle by reference, the worker-side entry
activates BEFORE the stock entry and exits 3 on a refused gate, gpu_ids_of reads the stock argv, and the pred route prints WORKERS."""
import functools
import pickle
import sys
import types

import pytest

from atlasfold_opt import workers as W


def _fake_multigpu(monkeypatch, calls):
    mod = types.ModuleType(W.SITE)
    def worker_entry(worker_index, args, assignments, gpu_ids, model_type):
        calls.append(("stock", worker_index, gpu_ids[worker_index], model_type))
    mod.worker_entry = worker_entry
    monkeypatch.setitem(sys.modules, W.SITE, mod)
    return mod


def test_install_rebinds_the_stock_entry_to_a_picklable_partial(monkeypatch):
    calls = []
    mod = _fake_multigpu(monkeypatch, calls)
    stock = mod.worker_entry
    r = W.install("fast", 1, False)
    assert r == {"installed": True, "reason": None}
    assert isinstance(mod.worker_entry, functools.partial) and mod.worker_entry.func is W.worker_entry
    assert mod.worker_entry.keywords == {"mode": "fast", "det": 1, "allow_partial": False} and mod.worker_entry.__wrapped_stock__ is stock
    assert W.install("fast", 1, False)["reason"] == "already"                       # idempotent
    blob = pickle.dumps(functools.partial(W.worker_entry, mode="exact", det=0, allow_partial=True))   # by reference: what mp.spawn ships to the worker
    back = pickle.loads(blob)
    assert back.func is W.worker_entry and back.keywords["mode"] == "exact"


def test_install_names_a_missing_multigpu_module(monkeypatch):
    monkeypatch.setitem(sys.modules, W.SITE, None)                                   # import raises ImportError
    r = W.install("fast", 0, False)
    assert r["installed"] is False and r["reason"].startswith(("ImportError", "ModuleNotFoundError"))


def test_worker_entry_activates_first_then_runs_stock_and_folds_the_gates(monkeypatch, capsys):
    calls = []
    _fake_multigpu(monkeypatch, calls)
    import atlasfold_opt, atlasfold_opt.phase_timing as PT, atlasfold_opt.stack as S
    monkeypatch.setattr(atlasfold_opt, "enable", lambda mode, **k: calls.append(("enable", mode, k["det"], k["trigger"])))
    monkeypatch.setattr(PT, "install", lambda: calls.append(("phase",)) or {"installed": True})
    monkeypatch.setattr(S, "gates_verdict", lambda: {"ok": True, "refused": []})
    W.worker_entry(0, types.SimpleNamespace(), [["t1"]], [3], "multimer", mode="fast", det=1, allow_partial=False)
    assert calls == [("enable", "fast", 1, "worker:gpu3"), ("phase",), ("stock", 0, 3, "multimer")]
    assert "WORKER gpu=3" in capsys.readouterr().err
    monkeypatch.setattr(S, "gates_verdict", lambda: {"ok": False, "refused": [("trimul_v4", "cell")]})
    with pytest.raises(SystemExit) as x:
        W.worker_entry(0, types.SimpleNamespace(), [["t1"]], [3], "multimer", mode="fast", det=0, allow_partial=False)
    assert x.value.code == 3 and "gates=refused:trimul_v4:cell" in capsys.readouterr().err


def test_gpu_ids_of_reads_the_stock_argv():
    assert W.gpu_ids_of(["multimer", "--gpu-ids", "0", "1", "--input-fasta", "x"]) == ["0", "1"]
    assert W.gpu_ids_of(["multimer", "--input-fasta", "x", "--gpu-ids", "2"]) == ["2"]
    assert W.gpu_ids_of(["multimer", "--input-fasta", "x"]) == []
