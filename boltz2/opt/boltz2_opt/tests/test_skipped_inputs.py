"""An input the stock parser skips (boltz process_inputs: 'Failed to process <yaml>. Skipping. Error: <e>.', no record in the manifest) is
named on one `[boltz2-opt] SKIPPED item=<name> reason=...` line, counted failed with that reason, and every other input is predicted —
the kit worker never dies on the short DataLoader (skipped.py; bz_worker_lev*.py process_item / the pipelined schedule; worker.run).
A batch boltz's own predict_step skips on CUDA out of memory (it returns {"exception": True}; the writer writes nothing) is the unit's
`FAILED item=<name> seed=<s> reason=upstream skipped batch: CUDA out of memory (...)` line, exit 1, the other units predicted."""
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

from .. import manifest as mf
from .. import cli, prep, report as rep, skipped, stack, worker, writer
from . import _stubs

TRUNK_SRC = os.path.join(stack.tree_dir(), "opt", "forward", "trunk_levers", "src")
_REAL_RUN = subprocess.run                                          # the off-route tests replace subprocess.run (cli's is the module's); every other call passes through
BASES = ["bz_worker_lev.py", "bz_worker_levf2.py"]


# ---------------------------------------------------------------- the helper module ----------------------------------------------------------------
def test_stock_error_reads_boltz_own_words():
    out = "Checking input data.\nFailed to process /w/in/b.yaml. Skipping. Error: 'GLY1_'.\nsomething else\n"
    err = "Traceback (most recent call last):\n  File \"boltz/main.py\", line 561, in process_input\n    x = chains[name]\nKeyError: 'GLY1_'\n"
    assert skipped.stock_error(out, err, "b") == "KeyError: 'GLY1_'"
    assert skipped.stock_error(out, "", "b") == "'GLY1_'", "no traceback captured: the stdout line's own error text"
    assert skipped.stock_error(out, err, "a") is None and skipped.stock_error("all fine\n", err, "b") is None
    out2 = "Failed to process /run/x/n0800.yaml. Skipping. Error: [Errno 2] No such file or directory: 'n0800/msa/n0800_e0.csv'.\n"   # a relative msa: path the parser could not open
    err2 = "Traceback (most recent call last):\n  File \"main.py\", line 1, in process_input\nFileNotFoundError: [Errno 2] No such file or directory: 'n0800/msa/n0800_e0.csv'\n"
    assert skipped.stock_error(out2, err2, "n0800") == "FileNotFoundError: [Errno 2] No such file or directory: 'n0800/msa/n0800_e0.csv'", "the exception is named"
    assert skipped.reason(skipped.stock_error(out2, err2)) == "stock parser skipped the input (FileNotFoundError: [Errno 2] No such file or directory: 'n0800/msa/n0800_e0.csv')"
    why = skipped.reason("KeyError: 'GLY1_'")
    assert why == "stock parser skipped the input (KeyError: 'GLY1_')"
    assert skipped.line("b", why) == "[boltz2-opt] SKIPPED item=b reason=stock parser skipped the input (KeyError: 'GLY1_')"
    assert skipped.failed_line("b", 0, why) == "[boltz2-opt] FAILED item=b seed=0 reason=" + why
    assert "no record in processed/manifest.json" in skipped.reason(None)
    import re
    assert re.fullmatch(skipped.LINE_RE, skipped.line("b", why)) and skipped.LINE_RE in worker.RELAY_LINES


def test_record_ids_and_results_manifest_path(tmp_path):
    m = tmp_path / "boltz_results_b" / "processed" / "manifest.json"; m.parent.mkdir(parents=True)
    assert skipped.stock_results_manifest(str(tmp_path), "/any/where/b.yaml") == str(m)
    assert skipped.record_ids(str(m)) is None, "absent: the process never reached the parser"
    m.write_text(json.dumps({"records": []}))
    assert skipped.record_ids(str(m)) == []
    m.write_text(json.dumps({"records": [{"id": "b"}]}))
    assert skipped.record_ids(str(m)) == ["b"]
    m.write_text("{not json")
    assert skipped.record_ids(str(m)) is None


def test_next_ready_advances_over_skipped_pairs_in_order():
    asked = []
    skip_at = lambda i: asked.append(i) or i in (1, 2)   # noqa: E731
    assert skipped.next_ready(0, 5, skip_at) == 0 and asked == [0]
    asked.clear(); assert skipped.next_ready(1, 5, skip_at) == 3 and asked == [1, 2, 3]
    asked.clear(); assert skipped.next_ready(4, 5, lambda i: True) == 5
    assert skipped.next_ready(5, 5, lambda i: 1 / 0) == 5, "never asks past the end"


def test_units_close_the_counts_and_name_every_failure():
    items = [{"name": n} for n in "abc"]; have = {("a", 0), ("a", 1), ("c", 0)}
    acct = skipped.units(items, [0, 1], {"b": skipped.reason("KeyError: 'GLY1_'")}, lambda n, s: None if (n, s) in have else f"{n}_model_0.cif")
    assert (acct["expected"], acct["ok"], acct["failed"], acct["status"]) == (6, 3, 3, "incomplete") and acct["ok"] + acct["failed"] == 6
    assert [(u["name"], u["seed"]) for u in acct["failed_units"]] == [("b", 0), ("b", 1), ("c", 1)]
    assert all(u["reason"].startswith("stock parser skipped the input") for u in acct["failed_units"][:2]) and acct["failed_units"][2]["reason"].startswith("outputs short")
    assert acct["skipped"] == [{"name": "b", "reason": skipped.reason("KeyError: 'GLY1_'")}] and acct["all_skipped"] is False
    assert skipped.units(items, [0], {n: "x" for n in "abc"}, lambda n, s: f"{n}_model_0.cif")["all_skipped"] is True
    assert skipped.units(items, [0], {}, lambda n, s: None)["status"] == "complete"
    assert skipped.skipped_from_log({"skipped": [{"name": "b", "reason": "r"}, {"junk": 1}, "x"]}) == {"b": "r"} and skipped.skipped_from_log(None) == {}


def test_units_name_a_batch_boltz_skipped_on_oom_by_the_workers_failed_entry():
    """Reason precedence per failed unit: the input's SKIPPED reason, else the worker's `failed` entry for that very (name, seed) — boltz's own
    batch skip on CUDA out of memory — else outputs short. A unit whose files are there is ok whatever the log says; counts close."""
    items = [{"name": n} for n in "abc"]; have = {("a", 0), ("c", 0), ("c", 1)}
    wl = {"failed": [{"name": "a", "seed": 1, "reason": skipped.oom_reason(66.04)}, {"name": "c", "seed": "1", "reason": "x"}, {"name": "zz"}, "junk"], "skipped": [{"name": "b", "reason": "p"}]}
    failed = skipped.failed_from_log(wl)
    assert failed == {("a", 1): skipped.oom_reason(66.04), ("c", 1): "x"} and skipped.failed_from_log(None) == {} and skipped.failed_from_log({"failed": [{"name": "a", "seed": None}]}) == {}
    acct = skipped.units(items, [0, 1], skipped.skipped_from_log(wl), lambda n, s: None if (n, s) in have else f"{n}_model_0.cif", failed=failed)
    assert [(u["name"], u["seed"], u["reason"]) for u in acct["failed_units"]] == [("a", 1, skipped.oom_reason(66.04)), ("b", 0, "p"), ("b", 1, "p")], "c/1 has its files: ok whatever the entry says"
    assert (acct["expected"], acct["ok"], acct["failed"]) == (6, 3, 3)
    why = skipped.oom_reason(66.04)
    assert why.startswith(skipped.OOM_REASON + " (") and skipped.OOM_REASON == "upstream skipped batch: CUDA out of memory" and "peak allocated 66.0 GiB" in why and "--mode big" in why
    assert skipped.oom_reason().startswith(skipped.OOM_REASON) and "peak allocated" not in skipped.oom_reason()
    assert skipped.upstream_skipped({"upstream_skipped": True, "peak_mem_GB": 66.04}) == why and skipped.upstream_skipped({"upstream_skipped": False}) is None and skipped.upstream_skipped(None) is None
    import re
    assert re.fullmatch(skipped.STOCK_OOM_RE, "| WARNING: ran out of memory, skipping batch") and skipped.STOCK_OOM_RE in worker.RELAY_LINES, "boltz's own words reach the caller's transcript"
    assert re.fullmatch(skipped.STOCK_OOM_RE, "| WARNING: ran out of memory, skipping batch, 3"), "the validation_step spelling too"


# ---------------------------------------------------------------- the kit worker's own code, driven with fakes (no GPU, no boltz) --------------------------------
def _regions(base):
    """The worker base's process_item block, pipelined schedule and non-pipelined loop, verbatim (executed below under stand-ins)."""
    src = open(os.path.join(TRUNK_SRC, base)).read()
    i1 = src.index("PROCESS_INPUTS_KW = dict("); j1 = src.index("# ---------------- RNG state capture")   # the region after its two import / zygote lines (the test injects SK and a PREP stand-in)
    i2 = src.index("\nif args.pipeline:\n"); i3 = src.index("\nfor it in ITEMS:\n")
    assert i1 < j1 < i2 < i3
    return src[i1:j1], src[i2:i3], src[i3:]


class _FakeStock:
    """Stand-ins for what process_item calls: `parse` plays the zygote's child running boltz's check_inputs + process_inputs (prep.parse_one)
    — an input listed in `bad` is skipped exactly as boltz's process_input does it (its words on stdout, the traceback on stderr, no record in
    the manifest), the others get one record whose id is the YAML stem — and hands back the child's captured streams as prep.Result."""

    def __init__(self, bad, affinity=()):
        self.bad = set(bad); self.affinity = set(affinity); self.parses = []

    def parse(self, yaml, out_dir, ccd_path, mol_dir, **kw):                     # prep.Zygote.parse's signature
        assert kw["boltz2"] is True and kw["preprocessing_threads"] == 1 and all(isinstance(x, str) for x in (yaml, out_dir, ccd_path, mol_dir))
        self.parses.append(Path(yaml).stem)
        p = Path(yaml); out_dir = Path(out_dir); (out_dir / "processed" / "structures").mkdir(parents=True, exist_ok=True)
        out = err = ""
        if p.stem in self.bad:
            err = "Traceback (most recent call last):\n  File \"boltz/main.py\", line 561, in process_input\nKeyError: 'GLY1_'\n"
            out = f"Failed to process {p}. Skipping. Error: 'GLY1_'.\n"
            recs = []
        else:
            recs = [{"id": p.stem, "affinity": {"binder": "L"} if p.stem in self.affinity else None}]
        (out_dir / "processed" / "manifest.json").write_text(json.dumps({"records": recs}))
        return prep.Result(0, out, err)

    class Manifest:
        @staticmethod
        def load(path):
            return types.SimpleNamespace(records=[types.SimpleNamespace(id=r["id"], affinity=r.get("affinity")) for r in json.load(open(path))["records"]])

    class BoltzProcessedInput:
        def __init__(self, **kw):
            self.__dict__.update(kw)


def _namespace(tmp_path, bad, names=("a", "b", "c"), seeds=(0, 1), pipeline=1, affinity=(), oom=()):
    ydir = tmp_path / "in"; ydir.mkdir(exist_ok=True); out = tmp_path / "out"; out.mkdir(exist_ok=True); (out / "_kit").mkdir(exist_ok=True)   # KD: the worker makes it at start (bz_worker_lev.py:88)
    items = []
    for n in names:
        y = ydir / f"{n}.yaml"; y.write_text("version: 1\n"); items.append({"name": n, "uid": f"u-{n}", "yaml": str(y), "seeds": list(seeds)})
    fs = _FakeStock(bad, affinity); events = []; calls = {"make_iter": [], "predict": [], "trainer": [], "affinity_leg": [], "affinity_leg_warm": []}
    oom = set(oom); digs = {}                                      # (name, seed) units boltz's predict_step skips on CUDA out of memory: the wrapper's facts say so, nothing is written

    def run_affinity_leg(out_dir, processed, it, s):              # the worker's: None without the property; else upstream's leg wrote affinity_<id>.json beside the structure
        rec = processed.manifest.records[0]
        if not rec.affinity:
            return None
        assert (out_dir / "predictions" / rec.id / f"{rec.id}_model_0.cif").is_file(), "the leg runs after the structure pass, before the outputs move"
        calls["affinity_leg"].append((rec.id, s)); (out_dir / "predictions" / rec.id / f"affinity_{rec.id}.json").write_text("{}")
        return 1.5
    import glob, shutil

    def write_pred(out_dir, name, seed_tag):
        d = out_dir / "predictions" / name; d.mkdir(parents=True, exist_ok=True)
        for f in (f"{name}_model_0.cif", f"pae_{name}_model_0.npz", f"plddt_{name}_model_0.npz", f"confidence_{name}_model_0.json"):
            (d / f).write_text(f"{f} {seed_tag}\n")

    def make_iter(processed, s):                                  # ONE featurized batch per accepted record (a skipped input has none: N-1 batches for N inputs)
        name = processed.manifest.records[0].id; calls["make_iter"].append((name, s))
        return "dm", "dl", iter([{"record": name, "seed": s}]), ("st", name, s)

    def step(out_dir, name, seed, seed_tag):                     # what the real predict_step wrapper leaves in DIGS["last"], and stock's files unless boltz skipped the batch
        assert digs.get("last") == {}, "the loop clears the previous unit's facts before every prediction"
        digs["last"] = {"model_s": 1.0, "peak_mem_GB": 66.04 if (name, seed) in oom else 5.0, "upstream_skipped": (name, seed) in oom, "name": name}
        if (name, seed) not in oom:
            write_pred(out_dir, name, seed_tag)

    def predict_pipelined(dm, it_, st, out_dir, processed):
        batch = next(it_)                                         # a short iterator here would be the old death: StopIteration
        calls["predict"].append((batch["record"], batch["seed"])); step(out_dir, batch["record"], batch["seed"], batch["seed"])

    class _Trainer:
        def __init__(self, out_dir, processed):
            self.out_dir, self.name = out_dir, processed.manifest.records[0].id

        def predict(self, model, datamodule=None, return_predictions=False):
            calls["trainer"].append(self.name); step(self.out_dir, self.name, datamodule.seed, "t")

    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(reset_peak_memory_stats=lambda: None, synchronize=lambda: None), device=lambda s: s)
    cur = {}                                                        # the non-pipelined loop's seed: rng_set(POST_CTOR[s]) runs right before the data module is built

    class _DataModule(fs.BoltzProcessedInput):
        def __init__(self, **kw):
            super().__init__(**kw); self.seed = cur.get("seed")
    ns = dict(os=os, sys=sys, json=json, time=time, glob=glob, shutil=shutil, Path=Path, torch=torch,
              OUT=out, KD=out / "_kit", B={"tag": "pred", "seeds": list(seeds)}, OPTS={}, ITEMS=items, LOG={"events": [], "per_item": [], "env": {}}, DIGS=digs, T0=time.time(),
              ev=lambda name, **kw: events.append((name, kw)), utc=lambda: "2026-01-01T00:00:00Z", args=types.SimpleNamespace(pipeline=pipeline, num_workers=1),
              PREP=fs, SK=skipped, WR=writer, Manifest=fs.Manifest, BoltzProcessedInput=fs.BoltzProcessedInput,   # WR: the kit's writer module itself, not installed here (no BOLTZ_WRITER attach): every WR call is the stock statement
              ccd_path=Path("/cache/ccd.pkl"), mol_dir=Path("/cache/mols"), MODEL=types.SimpleNamespace(to=lambda d: None, eval=lambda: None),
              make_iter=make_iter, predict_pipelined=predict_pipelined, make_trainer=lambda out_dir, processed, **kw: _Trainer(out_dir, processed),
              Boltz2InferenceDataModule=_DataModule, rng_set=lambda st: cur.update(seed=st[1]) if st and st[0] == "post" else None, POST_CTOR={s: ("post", s) for s in seeds}, DEV="cuda:0",
              run_affinity_leg=run_affinity_leg, warm_affinity_leg=lambda processed: calls["affinity_leg_warm"].append(processed.manifest.records[0].id))   # the worker's: AL.warm for an affinity input
    return ns, items, out, events, calls


@pytest.mark.parametrize("base", BASES)
def test_pipelined_worker_skips_a_parser_skipped_input_names_it_and_predicts_the_rest(tmp_path, capsys, base):
    """The pipelined schedule of the REAL worker base (the kit route's --pipeline 1), executed under stand-ins: 3 inputs x 2 seeds,
    the stock parser skips `b` -> the DataLoader stand-in yields batches only for `a` and `c` (N-1) -> no StopIteration, `b` named on one
    SKIPPED line with boltz's own error, recorded once under `skipped`, never given an iterator; `a` and `c` predicted for both seeds and
    moved to by_seed/; the counts close (ok 4 + failed 2 == 6)."""
    r1, r2, _ = _regions(base)
    ns, items, out, events, calls = _namespace(tmp_path, bad={"b"})
    exec(compile(r1, base + ":process_item", "exec"), ns)
    with pytest.raises(SystemExit) as ex:
        exec(compile(r2, base + ":pipelined", "exec"), ns)
    assert ex.value.code == 0
    text = capsys.readouterr()
    assert text.out.count("[boltz2-opt] SKIPPED item=b reason=stock parser skipped the input (KeyError: 'GLY1_')\n") == 1, text.out
    assert "Failed to process" in text.out and "KeyError: 'GLY1_'" in text.err, "stock's own words still reach the transcript"
    log = ns["LOG"]
    assert [(p["name"], p["seed"]) for p in log["per_item"]] == [("a", 0), ("a", 1), ("c", 0), ("c", 1)]
    assert [e["name"] for e in log["skipped"]] == ["b"] and log["skipped"][0]["reason"] == skipped.reason("KeyError: 'GLY1_'") and log["skipped"][0]["uid"] == "u-b"
    assert calls["make_iter"] == [("a", 0), ("a", 1), ("c", 0), ("c", 1)] and calls["predict"] == calls["make_iter"], "no iterator, no predict_step for the skipped input; the others in order"
    assert [e for e in events if e[0] == "item_skipped"] == [("item_skipped", {"item": "b", "reason": skipped.reason("KeyError: 'GLY1_'")})]
    for n in ("a", "c"):
        for s in (0, 1):
            assert (out / "by_seed" / n / f"s{s}" / f"{n}_model_0.cif").read_text() == f"{n}_model_0.cif {s}\n"
    assert not (out / "by_seed" / "b").exists()
    wl = json.load(open(out / "_kit" / "pred_worker_log.json"))
    assert wl["skipped"] == log["skipped"] and len(wl["per_item"]) == 4
    acct = skipped.units(items, [0, 1], skipped.skipped_from_log(wl), lambda n, s: None if (out / "by_seed" / n / f"s{s}" / f"{n}_model_0.cif").is_file() else f"{n}_model_0.cif")
    assert (acct["expected"], acct["ok"], acct["failed"]) == (6, 4, 2) and [(u["name"], u["seed"], u["reason"]) for u in acct["failed_units"]] == [("b", 0, wl["skipped"][0]["reason"]), ("b", 1, wl["skipped"][0]["reason"])]


@pytest.mark.parametrize("base", BASES)
def test_pipelined_worker_with_the_first_and_last_inputs_skipped_and_with_all_skipped(tmp_path, capsys, base):
    r1, r2, _ = _regions(base)
    ns, items, out, events, calls = _namespace(tmp_path, bad={"a", "c"}, seeds=(0,))
    exec(compile(r1, base, "exec"), ns)
    with pytest.raises(SystemExit) as ex:
        exec(compile(r2, base, "exec"), ns)
    assert ex.value.code == 0 and calls["predict"] == [("b", 0)] and [e["name"] for e in ns["LOG"]["skipped"]] == ["a", "c"]
    assert capsys.readouterr().out.count("[boltz2-opt] SKIPPED item=") == 2
    (tmp_path / "all").mkdir(exist_ok=True)
    ns, items, out, events, calls = _namespace(tmp_path / "all", bad={"a", "b", "c"}, seeds=(0,))
    exec(compile(r1, base, "exec"), ns)
    with pytest.raises(SystemExit) as ex:
        exec(compile(r2, base, "exec"), ns)
    assert ex.value.code == 0 and calls["make_iter"] == [] and calls["predict"] == [] and ns["LOG"]["per_item"] == [] and len(ns["LOG"]["skipped"]) == 3
    assert skipped.units(items, [0], skipped.skipped_from_log(ns["LOG"]), lambda n, s: f"{n}_model_0.cif")["all_skipped"] is True


@pytest.mark.parametrize("base", BASES)
def test_pipelined_worker_unchanged_when_nothing_is_skipped(tmp_path, capsys, base):
    """No skip: the schedule is the one it always was — every pair featurized once, in order, the next pair's iterator created before the current predicts."""
    r1, r2, _ = _regions(base)
    ns, items, out, events, calls = _namespace(tmp_path, bad=set())
    exec(compile(r1, base, "exec"), ns)
    with pytest.raises(SystemExit):
        exec(compile(r2, base, "exec"), ns)
    assert calls["make_iter"] == [(n, s) for n in "abc" for s in (0, 1)] == calls["predict"] and "skipped" not in ns["LOG"]
    assert "SKIPPED" not in capsys.readouterr().out


@pytest.mark.parametrize("base", BASES)
def test_non_pipelined_loop_skips_the_same_way(tmp_path, capsys, base):
    r1, _, r3 = _regions(base)
    ns, items, out, events, calls = _namespace(tmp_path, bad={"b"}, pipeline=0)
    exec(compile(r1, base, "exec"), ns); exec(compile(r3, base, "exec"), ns)
    assert calls["trainer"] == ["a", "a", "c", "c"] and [e["name"] for e in ns["LOG"]["skipped"]] == ["b"]
    assert capsys.readouterr().out.count("[boltz2-opt] SKIPPED item=b ") == 1


@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("pipeline", [1, 0])
def test_worker_names_a_batch_boltz_skipped_on_oom_and_predicts_the_rest(tmp_path, capsys, base, pipeline):
    """The REAL worker base's loops under stand-ins: boltz's predict_step skips (b, 1) on CUDA out of memory (the wrapper's facts say
    upstream_skipped) -> one `failed` log entry with the named reason and the peak, an item_failed event, no files for that unit, every other
    unit predicted and moved; the caller's accounting (skipped.units over failed_from_log) names exactly that unit by that reason."""
    r1, r2, r3 = _regions(base)
    ns, items, out, events, calls = _namespace(tmp_path, bad=set(), pipeline=pipeline, oom={("b", 1)})
    exec(compile(r1, base, "exec"), ns)
    if pipeline:
        with pytest.raises(SystemExit) as ex:
            exec(compile(r2, base, "exec"), ns)
        assert ex.value.code == 0
    else:
        exec(compile(r3, base, "exec"), ns)
    log = ns["LOG"]; why = skipped.oom_reason(66.04)
    assert [(e["name"], e["seed"], e["reason"], e["uid"]) for e in log["failed"]] == [("b", 1, why, "u-b")]
    assert [(p["name"], p["seed"], p["upstream_skipped"]) for p in log["per_item"]] == [(n, s, (n, s) == ("b", 1)) for n in "abc" for s in (0, 1)], "the unit stays in per_item with the wrapper's fact; the pass went on in order"
    assert [e for e in events if e[0] == "item_failed"] == [("item_failed", {"item": "b", "seed": 1, "reason": why})]
    assert not (out / "by_seed" / "b" / "s1" / "b_model_0.cif").exists() and (out / "by_seed" / "b" / "s0" / "b_model_0.cif").is_file() and (out / "by_seed" / "c" / "s1" / "c_model_0.cif").is_file()
    wl = json.load(open(out / "_kit" / "pred_worker_log.json"))
    assert wl["failed"] == log["failed"] and "skipped" not in wl
    acct = skipped.units(items, [0, 1], skipped.skipped_from_log(wl), lambda n, s: None if (out / "by_seed" / n / f"s{s}" / f"{n}_model_0.cif").is_file() else f"{n}_model_0.cif", failed=skipped.failed_from_log(wl))
    assert (acct["expected"], acct["ok"], acct["failed"]) == (6, 5, 1) and acct["failed_units"] == [{"name": "b", "seed": 1, "reason": why}]


def test_the_predict_step_wrapper_marks_boltz_own_batch_skip():
    """Source contract (the wrapper needs boltz to run): both bases' predict_step wrapper records `upstream_skipped` from the returned dict's
    `exception` flag — boltz2.py predict_step's out-of-memory catch is the only path that sets it — and both loops clear DIGS before a unit."""
    for base in BASES:
        src = open(os.path.join(TRUNK_SRC, base)).read()
        w = src[src.index("def _ps(self, batch"):src.index("Boltz2.predict_step = _ps")]
        assert 'd["upstream_skipped"] = bool(isinstance(r, dict) and r.get("exception"))' in w and w.index('r = _orig_ps(') < w.index('d["upstream_skipped"]') < w.index('DIGS["last"] = d'), base
        assert src.count('DIGS["last"] = {}') == 2 and src.count("SK.upstream_skipped(DIGS.get(\"last\"))") == 2, base
    stock = open(os.path.join(stack.tree_dir(), "stock", "src", "boltz", "model", "models", "boltz2.py")).read()
    h = stock[stock.index("    def predict_step("):stock.index("    def configure_optimizers(")]
    assert h.count('return {"exception": True}') == 1 and '"out of memory" in str(e)' in h and "| WARNING: ran out of memory, skipping batch" in h, "the pinned stock's catch is the one the reason names"


def test_a_short_iterator_for_an_accepted_record_is_named_not_a_bare_stopiteration():
    for base in BASES:
        src = open(os.path.join(TRUNK_SRC, base)).read()
        i = src.index("def predict_pipelined("); j = src.index("\nif args.pipeline:")
        body = src[i:j]
        assert "except StopIteration:" in body.split("rng_set(st)")[0] and "raise RuntimeError(" in body, base


# ---------------------------------------------------------------- the route: worker.run's accounting around a staged worker --------------------------------
def test_worker_route_names_counts_and_survives_a_parser_skipped_input(tmp_path, monkeypatch, capsys):
    """`pred --mode fast` on a, b, c where the stock parser skips b: the worker's SKIPPED line is relayed on this process's stdout, the unit is
    FAILED with that reason, a and c are complete, the manifest's outputs account closes (3 = 2 ok + 1 failed), the EXIT tally counts it, rc 1."""
    _stubs.gated_ok(monkeypatch); _stubs.install_fake_stage(monkeypatch, skip_items=("b",)); _stubs.run_staged_worker_here(monkeypatch)
    rep.reset_tally()
    yamls = _stubs.write_yamls(str(tmp_path), ("a", "b", "c")); out = str(tmp_path / "o")
    rc = worker.run("fast", yamls, out, [0])
    text = capsys.readouterr().out
    why = skipped.reason("KeyError: 'GLY1_'")
    assert rc == rep.EXIT_FAILED == 1, text
    assert f"[boltz2-opt] SKIPPED item=b reason={why}\n" in text, "the worker's line, relayed verbatim"
    assert f"[boltz2-opt] FAILED item=b seed=0 reason={why}\n" in text
    assert "[boltz2-opt] APPLIED" in text and "NOT ACTIVE" not in text and "KERNELS REQUIRE refused" not in text
    m = mf.LAST
    o = m["report"]["outputs"]
    assert (o["expected"], o["ok"], o["failed"], o["status"]) == (3, 2, 1, "incomplete") and o["failed_units"] == [{"name": "b", "seed": 0, "reason": why}] and o["skipped"] == [{"name": "b", "reason": why}]
    assert m["rc"] == 0 and m["report"]["active"] is True, "the worker itself ran clean; the unit account carries the failure"
    for n in ("a", "c"):
        assert os.path.isfile(os.path.join(out, "by_seed", n, "s0", f"{n}_model_0.cif"))
    assert not os.path.isdir(os.path.join(out, "by_seed", "b"))
    t = rep.tally_snapshot()
    assert (t["predictions"], t["ok"], t["failed"], t["rc"]) == (3, 2, 1, 1)


def test_worker_route_names_a_batch_boltz_skipped_on_oom_and_exits_1(tmp_path, monkeypatch, capsys):
    """`pred --mode fast` on a, b x seeds 0,1 where boltz's own predict_step skips both of b's batches on CUDA out of memory: boltz's WARNING
    line is relayed on this process's stdout, each unit of b is FAILED with the named reason (not `outputs short`), a is complete, the outputs
    account closes (4 = 2 ok + 2 failed), the EXIT tally counts it, rc 1 — the worker itself ran clean (rc 0, ACTIVE, APPLIED)."""
    _stubs.gated_ok(monkeypatch); _stubs.install_fake_stage(monkeypatch, oom_items=("b",)); _stubs.run_staged_worker_here(monkeypatch)
    rep.reset_tally()
    yamls = _stubs.write_yamls(str(tmp_path), ("a", "b")); out = str(tmp_path / "o")
    rc = worker.run("fast", yamls, out, [0, 1])
    text = capsys.readouterr().out
    why = skipped.oom_reason(66.0)
    assert rc == rep.EXIT_FAILED == 1, text
    assert text.count("| WARNING: ran out of memory, skipping batch\n") == 2, "boltz's own words, relayed verbatim once per skipped batch"
    assert f"[boltz2-opt] FAILED item=b seed=0 reason={why}\n" in text and f"[boltz2-opt] FAILED item=b seed=1 reason={why}\n" in text and "outputs short" not in text and "SKIPPED" not in text
    assert "[boltz2-opt] APPLIED" in text and "NOT ACTIVE" not in text and "KERNELS REQUIRE refused" not in text
    m = mf.LAST; o = m["report"]["outputs"]
    assert (o["expected"], o["ok"], o["failed"], o["status"]) == (4, 2, 2, "incomplete") and o["failed_units"] == [{"name": "b", "seed": s, "reason": why} for s in (0, 1)] and o["skipped"] == []
    assert m["rc"] == 0 and m["report"]["active"] is True
    assert os.path.isfile(os.path.join(out, "by_seed", "a", "s1", "a_model_0.cif")) and not os.path.isfile(os.path.join(out, "by_seed", "b", "s0", "b_model_0.cif"))
    t = rep.tally_snapshot(); assert (t["predictions"], t["ok"], t["failed"], t["rc"]) == (4, 2, 2, 1)


def test_worker_route_with_every_input_skipped_exits_1_by_the_skipped_lines_not_5(tmp_path, monkeypatch, capsys):
    """Nothing reached predict_step (every input SKIPPED): the census says NO-STEP; that is 'nothing ran', not an accelerator refusal —
    the SKIPPED / FAILED lines and exit 1 speak (report.kernels_exit all_skipped)."""
    _stubs.gated_ok(monkeypatch); _stubs.install_fake_stage(monkeypatch, skip_items=("a", "b")); _stubs.run_staged_worker_here(monkeypatch)
    rep.reset_tally()
    yamls = _stubs.write_yamls(str(tmp_path), ("a", "b")); out = str(tmp_path / "o")
    rc = worker.run("fast", yamls, out, [0])
    text = capsys.readouterr().out
    assert rc == 1 and text.count("[boltz2-opt] SKIPPED item=") == 2 and text.count("[boltz2-opt] FAILED item=") == 2, text
    assert "KERNELS REQUIRE refused" not in text and "verdict=NO-STEP" in text and "every requested input was SKIPPED" in text
    why = skipped.reason("KeyError: 'GLY1_'")                                                    # the stand-in's parser error, as the real worker names boltz's (skipped.stock_error)
    assert f"[boltz2-opt] NOT ACTIVE: inputs: 2/2 failed to parse in the worker — {why} (first failure: item a); nothing was predicted" in text, \
        "every input failed to parse: the run's reason names the parser's error (0.3.22)"
    o = mf.LAST["report"]["outputs"]
    assert (o["expected"], o["ok"], o["failed"]) == (2, 0, 2)
    assert mf.LAST["report"]["active"] is False and mf.LAST["report"]["reason"].startswith("inputs: 2/2 failed to parse in the worker — stock parser skipped the input (KeyError: 'GLY1_')")


def test_every_input_failing_to_parse_names_the_parser_error_not_a_levers_gate_and_exits_1(tmp_path, monkeypatch, capsys):
    """0.3.22: with every input skipped by the stock parser nothing ran, so a lever's evidence ('… served no call', a scope the stand-in reports
    broken here) is not the run's reason and not its exit: the NOT ACTIVE line names the parser's error of the first input, every unit is
    FAILED by name, exit 1 — as the stock route exits for the same inputs (test_off_route_names_a_parser_skipped_input_and_exits_1_not_5)."""
    _stubs.gated_ok(monkeypatch); _stubs.install_fake_stage(monkeypatch, evidence_ok=False, skip_items=("a", "b")); _stubs.run_staged_worker_here(monkeypatch)
    rep.reset_tally()
    yamls = _stubs.write_yamls(str(tmp_path), ("a", "b")); out = str(tmp_path / "o")
    rc = worker.run("fast", yamls, out, [0, 1])
    text = capsys.readouterr().out
    na = [l for l in text.splitlines() if l.startswith("[boltz2-opt] NOT ACTIVE")]
    assert rc == rep.EXIT_FAILED == 1 and len(na) == 1, text
    assert na[0].startswith("[boltz2-opt] NOT ACTIVE: inputs: 2/2 failed to parse in the worker — stock parser skipped the input (KeyError: 'GLY1_') (first failure: item a)"), na[0]
    assert "served no call" not in na[0] and "scope" not in na[0] and text.count("[boltz2-opt] FAILED item=") == 4 and text.count("[boltz2-opt] SKIPPED item=") == 2
    assert mf.LAST["report"]["active"] is False and mf.LAST["report"]["reason"] == na[0][len("[boltz2-opt] NOT ACTIVE: "):] and rep.tally_snapshot()["rc"] == rep.EXIT_FAILED


def test_worker_route_no_step_with_an_accepted_input_still_refuses_5(tmp_path, monkeypatch, capsys):
    """NO-STEP while an input WAS accepted (outputs written, nothing skipped) is the REQUIRE guard's business exactly as before: exit 5."""
    _stubs.gated_ok(monkeypatch); _stubs.install_fake_stage(monkeypatch, kernels="nostep"); _stubs.run_staged_worker_here(monkeypatch)
    rep.reset_tally()
    yamls = _stubs.write_yamls(str(tmp_path), ("a",)); out = str(tmp_path / "o")
    rc = worker.run("fast", yamls, out, [0])
    text = capsys.readouterr().out
    assert rc == rep.EXIT_KERNELS == 5 and "KERNELS REQUIRE refused: route=fast verdict=NO-STEP" in text, text
    _stubs.install_fake_stage(monkeypatch, kernels="nostep", skip_items=("b",)); rep.reset_tally()
    yamls = _stubs.write_yamls(str(tmp_path / "ab"), ("a", "b")) if (tmp_path / "ab").mkdir() is None else None
    rc = worker.run("fast", yamls, str(tmp_path / "o2"), [0])
    assert rc == 5 and "KERNELS REQUIRE refused" in capsys.readouterr().out, "one input skipped, the other accepted but nothing stepped: still the guard's refusal"


def test_kernels_exit_no_step_rule(capsys):
    nostep = {"verdict": "NO-STEP", "route": "stock", "words": {"cueq_triatt": "engaged:x", "cueq_trimul": "engaged:x"}, "refused": []}
    passed = dict(nostep, verdict="PASS")
    assert rep.kernels_exit("off", 0, [nostep], where="w", all_skipped=True) == 0 and "every requested input was SKIPPED" in capsys.readouterr().out
    assert rep.kernels_exit("off", 0, [nostep], where="w") == rep.EXIT_KERNELS and "KERNELS REQUIRE refused: route=stock verdict=NO-STEP" in capsys.readouterr().out
    assert rep.kernels_exit("fast", 0, [nostep, passed], where="w", all_skipped=True) == rep.EXIT_KERNELS, "a record that is not NO-STEP: the guard's rules exactly as before"
    assert rep.kernels_exit("off", 0, [passed], where="w", all_skipped=True) == 0 and rep.kernels_exit("off", 0, [passed], where="w") == 0
    assert rep.kernels_exit("off", 0, [], where="w", all_skipped=True) == rep.EXIT_NOT_ACTIVE, "no census record is still evidence missing"
    assert rep.kernels_exit("off", 1, [nostep], where="w", all_skipped=True) == 1 and rep.kernels_exit("off", rep.EXIT_KERNELS, [nostep], where="w", all_skipped=True) == rep.EXIT_KERNELS
    capsys.readouterr()


def _off_box(tmp_path, monkeypatch):
    """A box where `pred --mode off` reaches the stock subprocess: a complete (fake) Boltz-2 cache under BOLTZ_CACHE."""
    cache = tmp_path / "cache"; cache.mkdir()
    for f in ("boltz2_conf.ckpt", "boltz2_aff.ckpt", "ccd.pkl", "mols.tar"):
        (cache / f).write_bytes(b"x")
    (cache / "mols").mkdir()
    monkeypatch.setenv("BOLTZ_CACHE", str(cache)); monkeypatch.delenv("MODEL_OPT_WEIGHTS_DIGEST_DIR", raising=False); monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    rep.reset_tally()


def _fake_stock_process(parse_ok, predict_ok, verdict, rc=0):
    """subprocess.run stand-in for the stock caller: writes what `boltz predict <yaml> --out_dir <out>` leaves behind — the processed
    manifest (no record when its parser skipped the input, printing boltz's own words), the prediction files when it predicted, and the
    KERNELS census record the kit's stock_pred writes at exit (verdict NO-STEP when no predict_step ran)."""
    import types as _t

    def run(argv, env=None, timeout=None, **kw):
        if "--kernels-json" not in list(argv):                    # not the stock caller (e.g. stack.gpu_probe's nvidia-smi): the real thing
            return _REAL_RUN(argv, env=env, timeout=timeout, **kw)
        args = argv[argv.index("--") + 1:]; census = argv[argv.index("--kernels-json") + 1]; os.makedirs(os.path.dirname(census), exist_ok=True)   # kernels.emit's own mkdir
        y = args[1]; out = args[args.index("--out_dir") + 1]; stem = os.path.splitext(os.path.basename(y))[0]
        res = os.path.join(out, f"boltz_results_{stem}"); os.makedirs(os.path.join(res, "processed"), exist_ok=True)
        if parse_ok:
            recs = [{"id": stem}]
        else:
            print(f"Failed to process {y}. Skipping. Error: 'GLY1_'."); recs = []
        json.dump({"records": recs}, open(os.path.join(res, "processed", "manifest.json"), "w"))
        if predict_ok:
            d = os.path.join(res, "predictions", stem); os.makedirs(d)
            for f in (f"{stem}_model_0.cif", f"confidence_{stem}_model_0.json"):
                open(os.path.join(d, f), "w").write(f)
        json.dump({"verdict": verdict, "route": "stock", "mode": "off", "settings": "upstream", "words": {"cueq_triatt": "engaged:x", "cueq_trimul": "engaged:x"},
                   "refused": [], "written_unix": time.time()}, open(census, "w"))
        return _t.SimpleNamespace(returncode=rc)
    return run


def test_off_route_names_a_parser_skipped_input_and_exits_1_not_5(tmp_path, monkeypatch, capsys):
    """The stock arm on an input Boltz's parser skips: the process exits 0 having predicted nothing (census NO-STEP). The unit is read from
    upstream's own processed manifest (no record) -> one SKIPPED line, one FAILED line, tally failed, exit 1 — not the REQUIRE guard's 5."""
    _off_box(tmp_path, monkeypatch)
    monkeypatch.setattr(cli.subprocess, "run", _fake_stock_process(parse_ok=False, predict_ok=False, verdict="NO-STEP"))
    y = _stubs.write_yamls(str(tmp_path), ("b",))[0]; out = str(tmp_path / "o")
    rc = cli.main(["pred", "--mode", "off", "--input", y, "--out_dir", out, "--seed", "3"])
    text = capsys.readouterr().out
    why = skipped.reason(None)
    assert rc == 1, text
    assert f"[boltz2-opt] SKIPPED item=b reason={why}\n" in text and f"[boltz2-opt] FAILED item=b seed=3 reason={why}\n" in text and "Failed to process" in text
    assert "KERNELS REQUIRE refused" not in text and "every requested input was SKIPPED" in text
    m = mf.LAST
    o = m["outputs"]
    assert m["rc"] == 1 and (o["expected"], o["ok"], o["failed"], o["status"]) == (1, 0, 1, "incomplete") and o["failed_units"] == [{"name": "b", "seed": 3, "reason": why}] and o["predictions"] == {}
    t = rep.tally_snapshot(); assert (t["predictions"], t["ok"], t["failed"], t["rc"]) == (1, 0, 1, 1)


def test_off_route_no_step_with_the_input_accepted_is_still_the_guards_exit_5(tmp_path, monkeypatch, capsys):
    """The parser accepted the input (its record is in the manifest) yet nothing stepped and nothing was written: not a parser skip — the
    KERNELS guard refuses exactly as before (exit 5), and the unit is named FAILED (outputs short)."""
    _off_box(tmp_path, monkeypatch)
    monkeypatch.setattr(cli.subprocess, "run", _fake_stock_process(parse_ok=True, predict_ok=False, verdict="NO-STEP"))
    y = _stubs.write_yamls(str(tmp_path), ("a",))[0]
    rc = cli.main(["pred", "--mode", "off", "--input", y, "--out_dir", str(tmp_path / "o")])
    text = capsys.readouterr().out
    assert rc == 5 and "KERNELS REQUIRE refused: route=stock verdict=NO-STEP" in text and "[boltz2-opt] FAILED item=a seed=- reason=outputs short" in text and "SKIPPED" not in text, text


def test_off_route_complete_unit_is_ok(tmp_path, monkeypatch, capsys):
    _off_box(tmp_path, monkeypatch)
    monkeypatch.setattr(cli.subprocess, "run", _fake_stock_process(parse_ok=True, predict_ok=True, verdict="PASS"))
    y = _stubs.write_yamls(str(tmp_path), ("a",))[0]; out = str(tmp_path / "o")
    rc = cli.main(["pred", "--mode", "off", "--input", y, "--out_dir", out])
    text = capsys.readouterr().out
    assert rc == 0 and "FAILED" not in text and "SKIPPED" not in text, text
    o = mf.LAST["outputs"]
    assert (o["expected"], o["ok"], o["failed"], o["status"], o["failed_units"]) == (1, 1, 0, "complete", []) and list(o["predictions"]) == ["boltz_results_a/predictions/a"]
    assert not os.path.exists(os.path.join(out, "_kit")), "the stock pass's KERNELS record left with the verb and its kit directory with it"
    monkeypatch.setattr(cli.subprocess, "run", _fake_stock_process(parse_ok=True, predict_ok=False, verdict="PASS", rc=1)); rep.reset_tally()
    rc = cli.main(["pred", "--mode", "off", "--input", y, "--out_dir", str(tmp_path / "o2")])
    assert rc == 1 and "[boltz2-opt] FAILED item=a seed=- reason=the stock process exited 1" in capsys.readouterr().out


def test_stock_unit_account_shapes(tmp_path):
    y = str(tmp_path / "b.yaml"); open(y, "w").write("version: 1\n"); out = str(tmp_path / "o")
    acct = cli.stock_unit_account(out, y, None, 0)                  # nothing on disk: the process never reached the parser -> outputs short, not a skip
    assert acct["failed_units"][0]["reason"].startswith("outputs short") and acct["all_skipped"] is False and acct["skipped"] == []
    os.makedirs(os.path.join(out, "boltz_results_b", "processed")); json.dump({"records": []}, open(skipped.stock_results_manifest(out, y), "w"))
    acct = cli.stock_unit_account(out, y, 7, 0)
    assert acct["all_skipped"] is True and acct["failed_units"] == [{"name": "b", "seed": 7, "reason": skipped.reason(None)}] and (acct["ok"], acct["failed"]) == (0, 1)
    assert cli.stock_unit_account(out, y, 7, 2)["failed_units"][0]["reason"] == "the stock process exited 2", "a failed process is named by its exit code, never called a skip"
    pred = os.path.join(out, "boltz_results_b", "predictions", "b"); os.makedirs(pred); open(os.path.join(pred, "b_model_0.pdb"), "w").write("pdb")
    acct = cli.stock_unit_account(out, y, 7, 0)                     # upstream's `--output_format pdb`: the .pdb is the unit's structure
    assert (acct["ok"], acct["failed"], acct["status"]) == (1, 0, "complete"), acct


@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("pipeline", [1, 0])
def test_the_affinity_leg_runs_after_the_structure_pass_of_an_affinity_input_only(tmp_path, base, pipeline):
    """For an input whose record declares affinity the worker runs upstream's affinity leg right after that (input, seed)'s structure pass and
    before the outputs move: affinity_<id>.json lands in by_seed/<id>/s<seed>/ with the structure files and per_item carries affinity_s; the
    other inputs never reach the leg (affinity_s None)."""
    r1, r2, r3 = _regions(base)
    ns, items, out, events, calls = _namespace(tmp_path, bad=(), pipeline=pipeline, affinity={"b"})
    exec(r1, ns)
    if pipeline:
        with pytest.raises(SystemExit) as ex:
            exec(r2, ns)
        assert ex.value.code == 0
    else:
        exec(r3, ns)
    assert calls["affinity_leg"] == [("b", 0), ("b", 1)]
    assert calls["affinity_leg_warm"] == [it["name"] for it in items], \
        "the loops hand every parsed input to warm_affinity_leg once, right after process_item and ahead of its structure pass (the worker's starts the pass's served interpreter only for an input declaring the property)"
    for s in (0, 1):
        assert (out / "by_seed" / "b" / f"s{s}" / "affinity_b.json").is_file() and not (out / "by_seed" / "a" / f"s{s}" / "affinity_a.json").exists()
    aff_s = {(p["name"], p["seed"]): p["affinity_s"] for p in ns["LOG"]["per_item"]}
    assert aff_s == {("a", 0): None, ("a", 1): None, ("b", 0): 1.5, ("b", 1): 1.5, ("c", 0): None, ("c", 1): None}
    assert all("affinity_b.json" in p["files"] for p in ns["LOG"]["per_item"] if p["name"] == "b")
