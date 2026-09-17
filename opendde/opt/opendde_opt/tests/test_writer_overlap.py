"""writer_overlap: upstream's DataDumper.dump runs on one background writer thread over a host copy taken in the loop — the files are the
stock dump's bytes, the writer is drained in InferenceRunner.close before any reader, a failed write is loud (printed, reported in the
runner's per-item error file, counted, raised), a call outside the pin's signature is written synchronously by name; the lever rides every
single-card line, is off the row-sharded line, and composes out by name (modes.LEVER_SWITCHES / MODEL_OPT_LEVERS_OFF); the item-boundary
census (prefetch.py) prints one ITEM line per item. Stand-in runner modules; no GPU, no upstream."""
import hashlib
import json
import os
import sys
import types

import pytest

from opendde_opt import modes, prefetch, ran, registry, report, writer_overlap

TREE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC = os.path.join(TREE, "stock", "src")


# ---------------------------------------------------------------------------------------------------------------- stand-in upstream
class _Dumper:
    """Shaped like runner.dumper.DataDumper @1.1.1 (dump's signature, files under <base>/<name>/seed_<s>/predictions/)."""

    def __init__(self, base_dir, fail_on=None):
        self.base_dir, self.fail_on = base_dir, fail_on
        self.threads = []

    def dump(self, group_name, pdb_id, seed, pred_dict, atom_array, entity_poly_type):
        import threading
        self.threads.append(threading.current_thread().name)
        if pdb_id == self.fail_on:
            raise OSError(f"disk full while writing {pdb_id}")
        d = os.path.join(self.base_dir, group_name, pdb_id, f"seed_{seed}", "predictions")
        os.makedirs(d, exist_ok=True)
        for i, row in enumerate(pred_dict["summary_confidence"]):
            with open(os.path.join(d, f"{pdb_id}_summary_confidence_sample_{i}.json"), "w") as fh:
                json.dump({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in row.items()}, fh, indent=4)
        with open(os.path.join(d, f"{pdb_id}_sample_0.cif"), "w") as fh:
            fh.write(f"data_{pdb_id}\n# atoms={len(atom_array)} entities={sorted(entity_poly_type)}\n")
            fh.write("\n".join(" ".join(f"{x:.3f}" for x in xyz) for xyz in pred_dict["coordinate"].tolist()) + "\n")


class _Atoms(list):
    def copy(self):
        return _Atoms(self)


def _install_fake_runner(monkeypatch, tmp_path, fail_on=None):
    """runner.dumper + runner.inference stand-ins in sys.modules, the lever and the census bound on them (as phase's hook does at import)."""
    dm = types.ModuleType("runner.dumper")
    dm.DataDumper = _Dumper
    monkeypatch.setattr(_Dumper, "dump", _Dumper.__dict__["dump"])            # restored after the test whatever the lever bound
    ri = types.ModuleType("runner.inference")
    reports = []

    class InferenceRunner:
        def __init__(self, out):
            self.error_dir = os.path.join(out, "ERR")
            self.dumper = _Dumper(out, fail_on=fail_on)
            self.closed = 0

        def predict(self, data):
            return {"ok": data["sample_name"]}

        def close(self):
            self.closed += 1

    def _append_error_report(error_dir, filename, message):
        os.makedirs(error_dir, exist_ok=True)
        with open(os.path.join(error_dir, filename), "a") as fh:
            fh.write(message + "\n")
        reports.append(filename)

    ri.InferenceRunner = InferenceRunner
    ri._append_error_report = _append_error_report
    ri.seed_everything = lambda seed, deterministic: None
    ri._next_inference_batch_synchronized = lambda it, *a: next(it)
    ri._prepare_inference_batch = lambda batch: batch
    ri._cleanup_batch_synchronized = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "runner.dumper", dm)
    monkeypatch.setitem(sys.modules, "runner.inference", ri)
    writer_overlap._bind(ri)
    prefetch._bind(ri)
    return ri, reports


def _pred(np, n_atoms=7, n_samples=2, shift=0.0):
    return {"coordinate": np.arange(n_atoms * 3, dtype=np.float32).reshape(n_atoms, 3) + shift,
            "summary_confidence": [{"ranking_score": np.float32(0.5 + i), "plddt": np.linspace(0, 1, 4, dtype=np.float32)} for i in range(n_samples)],
            "full_data": [{"pae": np.ones((3, 3), dtype=np.float32) * i} for i in range(n_samples)]}


def _tree_digest(root):
    out = {}
    for d, _, files in os.walk(root):
        for f in files:
            p = os.path.join(d, f)
            out[os.path.relpath(p, root)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return out


# ---------------------------------------------------------------------------------------------------------------- the lever
def test_dump_runs_on_the_writer_thread_and_the_files_are_the_stock_bytes(monkeypatch, tmp_path, capsys):
    np = pytest.importorskip("numpy")
    ri, _ = _install_fake_runner(monkeypatch, tmp_path)
    assert writer_overlap.STATS["installed"] and writer_overlap._ours(_Dumper.dump) and writer_overlap._ours(ri.InferenceRunner.close)
    lever_out, stock_out = str(tmp_path / "lever"), str(tmp_path / "stock")
    r = ri.InferenceRunner(lever_out)
    items = [("q1", _pred(np)), ("q2", _pred(np, shift=1.0)), ("q3", _pred(np, n_atoms=5, shift=2.0))]
    for name, pd in items:
        assert r.dumper.dump(group_name="", pdb_id=name, seed=101, pred_dict=pd, atom_array=_Atoms(range(7)), entity_poly_type={"1": "polypeptide(L)"}) is None
    st = writer_overlap.kit_stats()
    assert st["items"] == 3 and st["submitted"] == 3 and st["sync"] == 0
    r.close()                                                                   # the drain point: every file on disk after it, the runner's own close ran once
    assert r.closed == 1
    st = writer_overlap.kit_stats()
    assert st["written"] == 3 and st["failed"] == 0 and st["drains"] >= 1 and writer_overlap._ST["writer"].pending() == 0
    assert all(t.startswith("opt-core-writer_overlap") for t in r.dumper.threads), r.dumper.threads   # upstream's dump body ran on the writer thread, never in the loop
    stock = _Dumper(stock_out)                                                  # the same dumps through the unbound method: the reference bytes
    for name, pd in items:
        writer_overlap._ST["orig_dump"](stock, "", name, 101, pd, _Atoms(range(7)), {"1": "polypeptide(L)"})
    assert _tree_digest(lever_out) == _tree_digest(stock_out) and len(_tree_digest(stock_out)) == 9
    out = capsys.readouterr().out
    assert out.count("WRITER item=") == 3 and "WRITER DRAIN at=close" in out and "submitted=3 written=3 failed=0" in out
    assert ran.count("writer_overlap") == 3 and writer_overlap.fallbacks(["writer_overlap"]) == []


def test_a_failed_write_is_printed_reported_counted_and_raised_at_close(monkeypatch, tmp_path, capsys):
    np = pytest.importorskip("numpy")
    ri, reports = _install_fake_runner(monkeypatch, tmp_path, fail_on="bad")
    r = ri.InferenceRunner(str(tmp_path / "o"))
    for name in ("good", "bad", "good2"):
        r.dumper.dump(group_name="", pdb_id=name, seed=7, pred_dict=_pred(np), atom_array=_Atoms(range(7)), entity_poly_type={})
    with pytest.raises(RuntimeError, match=r"1 inference sample\(s\) failed at the output stage \(writer_overlap\)"):
        r.close()
    assert r.closed == 1                                                        # upstream's close ran before the raise
    st = writer_overlap.kit_stats()
    assert st["written"] == 2 and st["failed"] == 1 and st["failures"][0][0] == "bad"
    assert reports == ["bad.txt"] and os.path.exists(os.path.join(str(tmp_path / "o"), "ERR", "bad.txt"))
    assert os.path.exists(os.path.join(str(tmp_path / "o"), "good2", "seed_7", "predictions", "good2_sample_0.cif"))   # the items after the failure were still written
    err = capsys.readouterr().err
    assert "WRITER FAILED item=bad seed=7 OSError: disk full" in err
    fb = writer_overlap.fallbacks(["writer_overlap"])
    assert fb and fb[0].startswith("writer_overlap: 1 write(s) failed (first: bad seed 7: OSError")
    r.close()                                                                   # a second close neither re-raises nor re-reports


def test_a_call_outside_the_pinned_signature_is_written_synchronously_by_name(monkeypatch, tmp_path, capsys):
    np = pytest.importorskip("numpy")
    ri, _ = _install_fake_runner(monkeypatch, tmp_path)
    r = ri.InferenceRunner(str(tmp_path / "o"))
    monkeypatch.setattr(writer_overlap, "DUMP_PARAMS", writer_overlap.DUMP_PARAMS + ("extra",))   # as if the pin's signature had one more parameter
    r.dumper.dump(group_name="", pdb_id="s1", seed=1, pred_dict=_pred(np), atom_array=_Atoms(range(7)), entity_poly_type={})
    assert os.path.exists(os.path.join(str(tmp_path / "o"), "s1", "seed_1", "predictions", "s1_sample_0.cif"))   # written before dump returned
    st = writer_overlap.kit_stats()
    assert st["sync"] == 1 and st["aside"] == {"signature": 1} and st["submitted"] == 0
    assert "aside=signature" in capsys.readouterr().out
    assert any("written synchronously" in f for f in writer_overlap.fallbacks(["writer_overlap"]))


def test_host_tree_takes_every_leaf_off_the_device_in_one_pass():
    torch = pytest.importorskip("torch")
    t = {"coordinate": torch.arange(6.0).reshape(2, 3), "summary_confidence": [{"s": torch.tensor(0.5)}], "full_data": [{"pae": torch.ones(2, 2, dtype=torch.bfloat16)}], "name": "x"}
    out, nbytes, leaves = writer_overlap.host_tree(t)
    assert writer_overlap._cuda_leaves(out) == 0 and out["name"] == "x" and out["full_data"][0]["pae"].dtype == torch.bfloat16
    assert torch.equal(out["coordinate"], t["coordinate"]) and (nbytes, leaves) == ((0, 0) if not torch.cuda.is_available() else (nbytes, leaves))


def test_install_arms_through_phase_and_binds_nothing_without_the_dumper_module(monkeypatch):
    from opendde_opt import phase
    monkeypatch.delitem(sys.modules, "runner.dumper", raising=False)
    fake = types.ModuleType("runner.inference")
    writer_overlap._bind(fake)                                                 # a runner module that never imported runner.dumper: armed, not installed, nothing missing
    assert writer_overlap.STATS["installed"] is False and writer_overlap.STATS["missing"] == []
    assert writer_overlap.fallbacks(["writer_overlap"]) == [] and writer_overlap.fallbacks(["keep_pool"]) == []
    dm = types.ModuleType("runner.dumper"); dm.DataDumper = type("DataDumper", (), {})   # the module is there, the pin's dump site is not: named
    monkeypatch.setitem(sys.modules, "runner.dumper", dm)
    writer_overlap._bind(fake)
    assert writer_overlap.STATS["missing"] == ["runner.dumper.DataDumper.dump"]
    assert writer_overlap.fallbacks(["writer_overlap"])[0].startswith("writer_overlap: runner.dumper.DataDumper.dump not found")
    writer_overlap._reset()
    monkeypatch.setattr(phase, "on_runner_module", lambda cb: "armed")
    assert writer_overlap.install() == "armed" and writer_overlap.STATS["armed"]


# ---------------------------------------------------------------------------------------------------------------- the rows
def test_the_lever_rides_every_single_card_line_and_composes_out_by_name():
    assert "writer_overlap" in registry.LEVERS and registry.LEVERS["writer_overlap"].tier == "exact" and registry.LEVERS["writer_overlap"].kit == registry.HOUSE
    assert registry.PIN_STATUS["writer_overlap"][0] == "tested" and "writer_overlap" in ran.COUNTERS
    assert report.STRATEGY["writer_overlap"] == "F6.output_overlap" and report.CORE_LEVERS["writer_overlap"] == "opt_core.host.outputs"
    for name in ("S1", "LSTAR2A", "BIG_F", modes.BIG_TP_LINE):                   # carried by no line (measured: CHANGES.md) — a registry row, a switch rule, off the TP line by name
        assert "writer_overlap" not in modes.LINES[name].levers, name
    assert "writer_overlap" in modes.BIG_TP_DROP and modes.LEVER_SWITCHES["writer_overlap"] == ()


def test_the_pinned_upstream_still_has_the_sites_the_lever_binds():
    dumper = open(os.path.join(SRC, "runner", "dumper.py")).read()
    inference = open(os.path.join(SRC, "runner", "inference.py")).read()
    import re
    assert re.search(r"^    def dump\(\n\s+self,\n\s+group_name: str,\n\s+pdb_id: str,\n\s+seed: int,\n\s+pred_dict: dict,\n\s+atom_array: AtomArray,\n\s+entity_poly_type", dumper, re.M)
    assert "runner.dumper.dump(" in inference and re.search(r"^    def close\(self\) -> None:", inference, re.M)
    assert re.search(r"runner = InferenceRunner\(configs\)\n    try:\n        infer_predict\(runner, runner\.configs\)\n    finally:\n        runner\.close\(\)", inference)
    assert "def _append_error_report(error_dir: str, filename: str, message: str)" in inference
    for name in [n for names in prefetch.SITES.values() for n in names]:
        assert re.search(rf"^def {name}\(|^from \S+ import {name}$|^\s+{name},$", inference, re.M), name   # the census's loop callables exist on the pin


# ---------------------------------------------------------------------------------------------------------------- the census
def test_the_item_census_prints_one_line_per_item_with_the_loop_parts(monkeypatch, tmp_path, capsys):
    np = pytest.importorskip("numpy")
    ri, _ = _install_fake_runner(monkeypatch, tmp_path)
    r = ri.InferenceRunner(str(tmp_path / "o"))
    it = iter([{"sample_name": "a"}, {"sample_name": "b"}])
    for _ in range(2):
        ri.seed_everything(1, True)
        data = ri._next_inference_batch_synchronized(it)
        ri._prepare_inference_batch(data)
        r.predict(data)
        r.dumper.dump(group_name="", pdb_id=data["sample_name"], seed=1, pred_dict=_pred(np), atom_array=_Atoms(range(7)), entity_poly_type={})
        ri._cleanup_batch_synchronized()
    r.close()
    out = capsys.readouterr().out
    lines = [l for l in out.splitlines() if " ITEM item=" in l]
    assert [l.split("item=")[1].split()[0] for l in lines] == ["a", "b"], out
    for tok in ("seed_s=", "feat_s=", "prep_s=", "predict_s=", "dump_s=", "cleanup_s=", "loop_s=", "outside_s="):
        assert all(tok in l for l in lines)
    assert "ITEMS n=2" in out and prefetch.kit_stats()["lines"] == 2


# ---------------------------------------------------------------------------------------------------------------- json_oneshot
def _fake_file_io(monkeypatch):
    """opendde.utils.file_io stand-in with upstream 1.1.1's save_json / map_values_to_list statements (file_io.py:186-218)."""
    np = pytest.importorskip("numpy")
    fio = types.ModuleType("opendde.utils.file_io")

    def map_values_to_list(data: dict, recursive: bool = True) -> dict:
        for k, v in data.items():
            if hasattr(v, "tolist") and not isinstance(v, (str, bytes)):
                data[k] = v.tolist()
            elif isinstance(v, dict) and recursive:
                data[k] = map_values_to_list(v, recursive)
        return data

    def save_json(data, output_fpath, indent=4):                       # the stock statement: json.dump (pure-Python streaming encoder)
        data_json = data.copy()
        data_json = map_values_to_list(data_json)
        with open(output_fpath, "w") as f:
            if indent is not None:
                json.dump(data_json, f, indent=indent)
            else:
                json.dump(data_json, f)

    fio.map_values_to_list, fio.save_json = map_values_to_list, save_json
    monkeypatch.setitem(sys.modules, "opendde.utils.file_io", fio)
    return fio, np


def _documents(np):
    rng = np.random.default_rng(7)
    full = {"token_pair_pae": np.round(rng.random((37, 37)) * 31.75, 2), "contact_probs": np.round(rng.random((37, 37)), 2), "plddt": np.round(rng.random(211) * 100, 2),
            "nested": {"a": np.arange(6, dtype=np.int64), "b": [1.0, float("nan"), 2.5e-8, 1e22, -0.0]}, "name": "x_\u00e9\u4e2d", "flag": True, "none": None, "i": 3}
    summary = {"ptm": 0.83, "iptm": 0.41, "ranking_score": 0.7231, "chain_ptm": np.round(rng.random(3), 4), "has_clash": False, "n": 37}
    return full, summary


def test_json_oneshot_writes_the_stock_writers_exact_bytes(monkeypatch, tmp_path):
    fio, np = _fake_file_io(monkeypatch)
    dm = types.ModuleType("runner.dumper"); dm.save_json = fio.save_json
    monkeypatch.setitem(sys.modules, "runner.dumper", dm)
    ri = types.ModuleType("runner.inference"); monkeypatch.setitem(sys.modules, "runner.inference", ri)
    writer_overlap.install_json(); writer_overlap._bind_json(ri)
    assert writer_overlap.JSTATS["installed"] and dm.save_json is writer_overlap.save_json_oneshot
    import copy
    for i, (doc, indent) in enumerate([(d, ind) for d in _documents(np) for ind in (None, 4)]):
        a, b = tmp_path / f"stock_{i}.json", tmp_path / f"lever_{i}.json"
        fio.save_json(copy.deepcopy(doc), str(a), indent=indent)      # upstream's json.dump path
        dm.save_json(copy.deepcopy(doc), str(b), indent=indent)       # the lever's json.dumps path
        assert a.read_bytes() == b.read_bytes(), (i, indent)
        assert hashlib.sha256(a.read_bytes()).hexdigest() == hashlib.sha256(b.read_bytes()).hexdigest()
    st = writer_overlap.kit_stats_json()
    assert st["files"] == 4 and st["indent_files"] == 2 and st["mib"] >= 0 and not writer_overlap.fallbacks_json(["json_oneshot"])
    writer_overlap.uninstall_json()
    assert dm.save_json is fio.save_json and not writer_overlap.JSTATS["installed"]


def test_json_oneshot_names_a_moved_site_and_stays_inert_without_the_dumper(monkeypatch):
    ri = types.ModuleType("runner.inference"); monkeypatch.setitem(sys.modules, "runner.inference", ri)
    monkeypatch.delitem(sys.modules, "runner.dumper", raising=False)
    writer_overlap.install_json(); writer_overlap._bind_json(ri)
    assert not writer_overlap.JSTATS["installed"] and writer_overlap.JSTATS["armed"] and not writer_overlap.fallbacks_json(["json_oneshot"])   # no dumper: inert, not a fallback
    dm = types.ModuleType("runner.dumper"); monkeypatch.setitem(sys.modules, "runner.dumper", dm)               # a dumper without the name: the site moved, named
    fio, _ = _fake_file_io(monkeypatch)
    writer_overlap._bind_json(ri)
    fb = writer_overlap.fallbacks_json(["json_oneshot"])
    assert fb and "runner.dumper.save_json not found" in fb[0]


def test_json_oneshot_rows(monkeypatch):
    lv = registry.LEVERS["json_oneshot"]
    assert lv.tier == "exact" and lv.file == "opendde_opt/writer_overlap.py"
    for name in ("S1", "LSTAR2A", "BIG_F"):
        assert "json_oneshot" in modes.LINES[name].levers
        assert "json_oneshot" not in modes.line_without(modes.LINES[name], ("json_oneshot",)).levers
    assert "json_oneshot" in modes.LINES[modes.BIG_TP_LINE].levers and "json_oneshot" not in modes.BIG_TP_DROP   # 0.2.66: every rank's writer binds it under --n_gpu P
    assert "writer_overlap" not in modes.LINES[modes.BIG_TP_LINE].levers and "writer_overlap" in modes.BIG_TP_DROP
    assert modes.LEVER_SWITCHES["json_oneshot"] == () and "json_oneshot" in ran.COUNTERS
    src = open(os.path.join(SRC, "runner", "dumper.py")).read(); fio = open(os.path.join(SRC, "opendde", "utils", "file_io.py")).read()
    assert "from opendde.utils.file_io import save_json" in src and "save_json(" in src
    import re
    assert re.search(r"def save_json\(data, output_fpath, indent=4\):\n    data_json = data.copy\(\)\n    data_json = map_values_to_list\(data_json\)\n    with open\(output_fpath, \"w\"\) as f:\n"
                     r"        if indent is not None:\n            json.dump\(data_json, f, indent=indent\)\n        else:\n            json.dump\(data_json, f\)\n", fio)   # the statement the lever restates


# ---------------------------------------------------------------------------------------------------------------- prefetch (the DataLoader worker)
class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _runner_with_create(monkeypatch, with_replay_sites=True):
    ri = types.ModuleType("runner.inference"); seen = []; seeded = []

    def _create_inference_dataloader_synchronized(configs, inputs, foldcp_config, world_control_group=None):
        seen.append(getattr(configs, "num_workers", None)); return ("loader", configs)
    ri._create_inference_dataloader_synchronized = _create_inference_dataloader_synchronized
    if with_replay_sites:                                            # runner.inference's seed_everything (the loop's RNG reset) and the dataset class the worker calls
        def seed_everything(seed, deterministic):
            seeded.append((seed, deterministic))
        ri.seed_everything = seed_everything
        dl = types.ModuleType("opendde.data.inference.infer_dataloader")

        class InferenceDataset:
            def __getitem__(self, index):
                return {"index": index}
        dl.InferenceDataset = InferenceDataset
        monkeypatch.setitem(sys.modules, "opendde.data.inference.infer_dataloader", dl)
    monkeypatch.setitem(sys.modules, "runner.inference", ri)
    prefetch.install_prefetch(); prefetch._bind_prefetch(ri)
    return ri, seen


def test_prefetch_turns_upstreams_default_loader_into_a_one_worker_loader(monkeypatch, capsys):
    ri, seen = _runner_with_create(monkeypatch)
    cfg = _Cfg(num_workers=0)
    assert ri._create_inference_dataloader_synchronized(cfg, [{}, {}, {}], None) == ("loader", cfg)
    assert seen == [1] and cfg.num_workers == 1
    st = prefetch.kit_stats_prefetch()
    assert st["installed"] and st["calls"] == 1 and st["engaged"] == 1 and st["workers"] == 1 and not prefetch.fallbacks_prefetch(["prefetch"])
    assert "PREFETCH dataloader num_workers=0->1" in capsys.readouterr().out
    assert ran.COUNTERS["prefetch"]() == 1


def test_prefetch_leaves_a_callers_num_workers_and_a_process_group_alone_by_name(monkeypatch, capsys):
    ri, seen = _runner_with_create(monkeypatch)
    ri._create_inference_dataloader_synchronized(_Cfg(num_workers=3), [{}, {}], None)       # the caller's own setting stands
    dist = types.ModuleType("torch.distributed"); dist.is_available = lambda: True; dist.is_initialized = lambda: True; dist.get_world_size = lambda: 2
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    ri._create_inference_dataloader_synchronized(_Cfg(num_workers=0), [{}, {}], None)       # a 2-rank process group: items read in the rank process
    ri._create_inference_dataloader_synchronized(_Cfg(), [{}, {}], None)                    # configs without the field
    monkeypatch.delitem(sys.modules, "torch.distributed", raising=False)
    ri._create_inference_dataloader_synchronized(_Cfg(num_workers=0), [{}], None)            # a one-item query: nothing to featurise ahead
    assert seen == [3, 0, None, 0]
    st = prefetch.kit_stats_prefetch()
    assert st["engaged"] == 0 and st["calls"] == 4 and st["aside"] == {"num_workers=3_set_by_the_caller": 1, "process_group_of_2_ranks": 1, "configs_without_num_workers": 1, "single_item_query": 1}
    assert not prefetch.fallbacks_prefetch(["prefetch"])                              # asides are inert by name, not fallbacks
    assert capsys.readouterr().out.count("PREFETCH aside reason=") == 4


def test_prefetch_replays_the_loops_rng_reset_in_the_worker_before_featurising(monkeypatch):
    """The stock loop featurises inside next() right after seed_everything(seed): under the lever the worker replays that reset before
    dataset[index], so the featuriser's random draws (reference-conformer rigid transforms) read the same RNG state — the same features."""
    np = pytest.importorskip("numpy")
    import random
    ri, _ = _runner_with_create(monkeypatch)
    ri.seed_everything(seed=101, deterministic=True)                  # the loop's reset, recorded by the lever's wrapper
    assert prefetch._SEED == {"seed": 101, "deterministic": True}
    ds = sys.modules["opendde.data.inference.infer_dataloader"].InferenceDataset()
    tud = pytest.importorskip("torch.utils.data")
    monkeypatch.setattr(tud, "get_worker_info", lambda: None)
    monkeypatch.delitem(sys.modules, "opendde.utils.seed", raising=False)   # the fallback path: the three host seeds
    random.seed(7); np.random.seed(7)
    assert ds[3] == {"index": 3}                                     # the parent process (no worker info): nothing replayed
    assert random.random() == random.Random(7).random()
    monkeypatch.setattr(tud, "get_worker_info", lambda: object())     # inside a DataLoader worker
    ds[4]
    a = (random.random(), float(np.random.uniform()))
    random.seed(101); np.random.seed(101)
    assert a == (random.random(), float(np.random.uniform()))         # = the state right after seed_everything(101)


def test_prefetch_steps_aside_when_the_replay_sites_are_absent(monkeypatch, capsys):
    ri, seen = _runner_with_create(monkeypatch, with_replay_sites=False)
    ri._create_inference_dataloader_synchronized(_Cfg(num_workers=0), [{}, {}], None)
    assert seen == [0] and prefetch.kit_stats_prefetch()["engaged"] == 0
    assert "PREFETCH aside reason=rng_replay_unbound" in capsys.readouterr().out or prefetch.fallbacks_prefetch(["prefetch"])


def test_prefetch_names_a_moved_site(monkeypatch):
    ri = types.ModuleType("runner.inference"); monkeypatch.setitem(sys.modules, "runner.inference", ri)
    prefetch.install_prefetch(); prefetch._bind_prefetch(ri)
    fb = prefetch.fallbacks_prefetch(["prefetch"])
    assert fb and "_create_inference_dataloader_synchronized not found" in fb[0]


def test_prefetch_rows_and_the_pinned_site():
    lv = registry.LEVERS["prefetch"]
    assert lv.tier == "exact" and lv.file == "opendde_opt/prefetch.py"
    for name in ("S1", "LSTAR2A", "BIG_F"):
        assert "prefetch" in modes.LINES[name].levers and "prefetch" not in modes.line_without(modes.LINES[name], ("prefetch",)).levers
    assert "prefetch" not in modes.LINES[modes.BIG_TP_LINE].levers and "prefetch" in modes.BIG_TP_DROP and modes.LEVER_SWITCHES["prefetch"] == ()
    import re
    inference = open(os.path.join(SRC, "runner", "inference.py")).read()
    assert re.search(r"^def _create_inference_dataloader_synchronized\(\n    configs: Any,\n", inference, re.M)
    dl = open(os.path.join(SRC, "opendde", "data", "inference", "infer_dataloader.py")).read()
    assert "num_workers=configs.num_workers," in dl
    assert '"num_workers": 0,' in open(os.path.join(SRC, "opendde", "config", "inference_defaults.py")).read()


def test_prefetch_relays_the_workers_featurisation_counters_into_the_parents_books(monkeypatch):
    """drop_bond_mask counts inside featurisation; under the lever that runs in the DataLoader worker (a fork), so the worker adds its
    per-item deltas to shared memory and the parent folds them at next(): ran-or-refuse reads the parent's books as on the stock loader."""
    ri, seen = _runner_with_create(monkeypatch)
    calls = []
    ri._next_inference_batch_synchronized = lambda it, *a: (calls.append(1), next(it))[1]
    prefetch.uninstall_prefetch(); prefetch.install_prefetch(); prefetch._bind_prefetch(ri)      # re-bind with the next() site present
    bm = types.ModuleType("opendde_opt.bondmask"); bm.STATS = {"installed": True, "dropped": 0, "bytes_dropped": 0, "missing": 0}
    monkeypatch.setitem(sys.modules, "opendde_opt.bondmask", bm)
    ri.seed_everything(5, False)
    ri._create_inference_dataloader_synchronized(_Cfg(num_workers=0), [{}, {}], None)                  # engaged: the shared array exists before any worker
    assert prefetch.kit_stats_prefetch()["engaged"] == 1 and prefetch._RELAY["shm"] is not None
    dl = sys.modules["opendde.data.inference.infer_dataloader"]
    orig = prefetch._PST["orig_getitem"]

    def featurise(self, index):                                      # the featuriser under drop_bond_mask: one drop per item
        bm.STATS["dropped"] += 1; bm.STATS["bytes_dropped"] += 800
        return orig(self, index)
    monkeypatch.setitem(prefetch._PST, "orig_getitem", featurise)
    dl.InferenceDataset.__getitem__ = prefetch._getitem_replayed(featurise)
    tud = pytest.importorskip("torch.utils.data")
    monkeypatch.setattr(tud, "get_worker_info", lambda: object())     # "in the worker": two items featurised
    ds = dl.InferenceDataset(); ds[0]; ds[1]
    bm.STATS.update(dropped=0, bytes_dropped=0)                      # the parent's books never saw the worker's increments (a fork's copy did)
    it = iter([("a",), ("b",)])
    ri._next_inference_batch_synchronized(it)                        # the loop's next(): the fold
    assert bm.STATS["dropped"] == 2 and bm.STATS["bytes_dropped"] == 1600 and prefetch.kit_stats_prefetch()["relayed"] >= 2
    ri._next_inference_batch_synchronized(it)                        # nothing new: nothing added twice
    assert bm.STATS["dropped"] == 2


# ------------------------------------------------------------------------------------------------ prefetch × RDKit's conformer generator
_SMILES_JOB = {"name": "aspirin_on_1l2y", "sequences": [{"proteinChain": {"sequence": "NLYIQWLKDGGPSSGRPPPS", "count": 1}},
                                                         {"ligand": {"ligand": "CC(=O)Oc1ccccc1C(=O)O", "count": 1}}]}
_SMILES_JOB2 = {"name": "ibuprofen_on_1l2y", "sequences": [{"proteinChain": {"sequence": "NLYIQWLKDGGPSSGRPPPS", "count": 1}},
                                                            {"ligand": {"ligand": "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "count": 1}}]}
_CCD_JOB = {"name": "atp_on_1l2y", "sequences": [{"proteinChain": {"sequence": "NLYIQWLKDGGPSSGRPPPS", "count": 1}},
                                                  {"ligand": {"ligand": "CCD_ATP", "count": 1}}, {"ion": {"ion": "MG", "count": 2}}]}
_PROT_JOB = {"name": "1l2y", "sequences": [{"proteinChain": {"sequence": "NLYIQWLKDGGPSSGRPPPS", "count": 1}}]}


class _CfgS(_Cfg):
    def __init__(self, num_workers=0, seeds=()):
        super().__init__(num_workers=num_workers); self.seeds = list(seeds)


def test_prefetch_steps_aside_on_two_seed_passes_over_an_rdkit_ligand(monkeypatch, capsys):
    """Two seed passes (``--seeds 101,102``) over a 2-item query whose items carry a SMILES ligand: upstream forks a fresh worker per seed
    pass from a parent whose RDKit generator never advanced, so the second pass would repeat the first pass's conformers where the stock
    loop continues the sequence — the lever steps aside by name, the run complete (inert, never a fallback). The rule is line-independent:
    the unit never reads the mode, and every single-GPU line (exact S1, fast LSTAR2A, the big lines) carries the same unit."""
    for name, line in modes.LINES.items():                            # the same lever unit on every single-GPU line; off the --n_gpu P>1 line by name
        assert ("prefetch" in line.levers) == (name != modes.BIG_TP_LINE), name
    ri, seen = _runner_with_create(monkeypatch)
    ri.seed_everything(101, True)
    cfg = _CfgS(num_workers=0, seeds=["101", "102"])
    assert ri._create_inference_dataloader_synchronized(cfg, [_SMILES_JOB, _SMILES_JOB2], None) == ("loader", cfg)
    assert seen == [0]                                                # upstream's DataLoader as configured: no worker
    st = prefetch.kit_stats_prefetch()
    assert st["engaged"] == 0 and st["aside"] == {"rdkit_rng_multiseed": 1}
    assert prefetch.fallbacks_prefetch(["prefetch"]) == []            # not PARTIAL: refresh files it under levers_inert (LEVER state=off reason=aside:rdkit_rng_multiseed)
    assert "PREFETCH aside reason=rdkit_rng_multiseed" in capsys.readouterr().out
    assert prefetch._seed_passes(cfg, [_SMILES_JOB, _SMILES_JOB2]) == 2 and prefetch._rdkit_ligand_jobs([_SMILES_JOB, _CCD_JOB, _PROT_JOB]) == 1


def test_prefetch_engages_with_one_seed_pass_or_ccd_only_ligands(monkeypatch):
    """One seed pass (any ligands) and CCD-only / ligand-free queries (any number of seeds) keep the worker: one worker featurises the items
    in the loop's order from the parent's generator state, as the stock loop does."""
    ri, seen = _runner_with_create(monkeypatch)
    ri.seed_everything(101, True)
    cases = [(_CfgS(seeds=["101"]), [_SMILES_JOB, _SMILES_JOB2]),                    # one pass, SMILES items
             (_CfgS(seeds=["101", "102", "103"]), [_CCD_JOB, _PROT_JOB]),             # three passes, no RDKit ligand
             (_CfgS(seeds=[]), [dict(_SMILES_JOB, modelSeeds=[7]), dict(_SMILES_JOB2, modelSeeds=[7])]),   # the jobs' own modelSeeds: one pass
             (_CfgS(seeds=["5", "5"]), [_SMILES_JOB, _SMILES_JOB2])]                  # duplicate seeds are one pass (upstream de-duplicates)
    for cfg, items in cases:
        assert ri._create_inference_dataloader_synchronized(cfg, items, None) == ("loader", cfg)
    assert seen == [1, 1, 1, 1]
    st = prefetch.kit_stats_prefetch()
    assert st["engaged"] == 4 and st["aside"] == {}


def test_prefetch_counts_seed_passes_as_upstream_resolves_them_without_drawing(monkeypatch):
    """--seeds for every job, else each job's modelSeeds, else one drawn default per job (runner/inference.py:870-892) — counted, nothing drawn."""
    import random
    state = random.getstate()
    assert prefetch._seed_passes(_CfgS(seeds=["101", "102"]), [_PROT_JOB]) == 2
    assert prefetch._seed_passes(_CfgS(seeds=[]), [dict(_PROT_JOB, modelSeeds=[1, 2]), dict(_CCD_JOB, modelSeeds=[2, 3])]) == 3
    assert prefetch._seed_passes(_CfgS(seeds=[]), [_PROT_JOB, _SMILES_JOB]) == 2          # two jobs without seeds: two drawn defaults = two passes
    assert prefetch._seed_passes(_CfgS(seeds=[]), [dict(_PROT_JOB, modelSeeds=[9]), _SMILES_JOB]) == 2
    assert prefetch._seed_passes(_Cfg(num_workers=0), [_PROT_JOB]) == 1                   # configs without a seeds field: one job, one pass
    assert random.getstate() == state
    assert prefetch._rdkit_ligand_jobs([_CCD_JOB, _PROT_JOB, {"sequences": [{"ligand": {"ligand": "FILE_/x/lig.sdf"}}]}]) == 1   # FILE_ counted, conservatively
    src = open(os.path.join(SRC, "opendde", "data", "inference", "json_parser.py")).read()
    assert 'if ligand_str.startswith("CCD_"):' in src and "atom_info = smiles_to_atom_info(ligand_str)" in src and "AllChem).EmbedMolecule" in src   # the featuriser's own branch the rule restates
    inf = open(os.path.join(SRC, "runner", "inference.py")).read()
    assert 'configured = cli_seeds if cli_seeds else job.get("modelSeeds")' in inf and "schedule.append(seeds or [random.randint(1, 65536)])" in inf
