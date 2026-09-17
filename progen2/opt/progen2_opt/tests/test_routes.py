"""The exact routes at the command line, both in this process (CPU only: the kits replaced by stand-ins that record what they were asked):
`sample` = ONE load for the call / the job (the job's batch sizes held from the load: (num_samples, max_length) per item), one unit per item,
stdout = the block for a single call, items/<id>/block.txt for a job, the lines in order (ACTIVE, load, ready, item, PEAK), an item that cannot
fit named and the job going on (exit 1), a load that ran out of memory named (exit 1), a lever the load did not install = the refusal by name
(exit 3, nothing generated); `score` = the exit rule (partial activation 3, the outputs kept; a KitRefused escaping is 3 by name). And the
programmatic surface (api.py): the same functions, importable without torch."""
import inspect
import json
import re
import types

import pytest

from progen2_opt import api, cli, generate, report, stack
from progen2_opt import score as _score

PLANNED = ["oneread_mmap", "sampler_exact", "resident_rotary", "static_kv"]


def _active(monkeypatch, tmp_path, route):
    """An active exact report for `small` (the gates replaced: no GPU, no pins on the CPU box) and a run dir; returns the report."""
    rep = {"active": True, "mode": "exact", "variant": "small", "route": route, "kit_line": "sample=serving:pipeline_v0_4", "logged": True,
           "optimizations": {"sample": list(PLANNED), "score": ["rotary_tables"]}, "gpu": {"name": "stub"}, "card_class": "h100",
           "package_home": stack.package_home(), "kit_dirs": {"serving": stack.kit_dir("serving"), "scoring": stack.kit_dir("scoring")}}
    monkeypatch.setattr(stack, "activate", lambda mode, variant=None, **kw: dict(rep))
    monkeypatch.setattr(stack, "pins", lambda: {})
    monkeypatch.setattr(stack, "workdir", lambda variant, create=True: str(tmp_path))
    monkeypatch.setattr(stack, "stock_dir", lambda: str(tmp_path))
    return rep


class _OutOfMemory(RuntimeError):
    def __init__(self, row):
        super().__init__(row["error"]); self.row = row


_OutOfMemory.__name__ = "OutOfMemory"


class _FakeHandle:
    """The kit's Handle as generate.py drives it: hold / unit / held_batches / peak_gib / kit.S.max_slots / levers / load_s."""

    def __init__(self, log, levers, max_length, oom_batches=()):
        self.log, self.levers, self.load_s = log, tuple(levers), 1.5
        self.kit = types.SimpleNamespace(S=types.SimpleNamespace(max_slots=256 if max_length <= 256 else 512))
        self.held, self.oom_batches, self.units = [], set(oom_batches), 0

    def hold(self, shapes):
        self.log.append(("hold", [tuple(s) for s in shapes]))
        for B, L in shapes:
            if B not in self.held and B not in self.oom_batches:
                self.held.append(B)
        return {"allocated": list(self.held), "not_allocated": [], "held": list(self.held)}

    def held_batches(self):
        return sorted(self.held)

    def unit(self, context, max_length, num_samples, rng_seed, t, p, rng_deterministic=True):
        self.log.append(("unit", context, max_length, num_samples, rng_seed, t, p, rng_deterministic))
        if num_samples in self.oom_batches:
            raise _OutOfMemory({"error": f"OUT OF MEMORY at sample N={num_samples} L={max_length}: the static K/V slots for {num_samples} samples do not fit; no fallback applied", "oom": True})
        self.units += 1
        block = f"{context}\n" + "".join(f"\n{i}\n{context}SEQ{rng_seed}\n" for i in range(num_samples)) + "done.\n"
        return {"block": block, "ids": [[1, 2]], "record": {"unit_s": 0.01, "sampler": {"tokens_sha256": "ab" * 32, "shadow_calls": 1, "sampler": "exact"}}}

    def peak_gib(self):
        return 12.345, 13.5

    def close(self):
        self.log.append(("close",)); return {}


def _fake_kit(monkeypatch, log, levers=PLANNED, oom_batches=(), load_raises=None):
    """generate.kit_module / stock_module replaced: the kit's load records its arguments and hands back the stand-in handle."""
    def load(stock, up, workdir, **kw):
        log.append(("load", up, kw))
        if load_raises is not None:
            raise load_raises
        return _FakeHandle(log, levers, kw["max_length"], oom_batches)
    kit = types.SimpleNamespace(load=load, OutOfMemory=_OutOfMemory, AUTO_LEVERS=("oneread_mmap", "sampler_exact"))
    monkeypatch.setattr(generate, "kit_module", lambda: kit)
    monkeypatch.setattr(generate, "stock_module", lambda wd: types.SimpleNamespace(__name__="sample"))
    return kit


def _run(cap, argv):
    rc = cli.main(argv)
    out = cap.readouterr()
    return rc, out.err, out.out


def test_sample_single_call_prints_the_block_and_writes_nothing(tmp_path, monkeypatch, capsys):
    _active(monkeypatch, tmp_path, "sample"); log = []
    _fake_kit(monkeypatch, log)
    monkeypatch.chdir(tmp_path)
    rc, err, out = _run(capsys, ["sample", "--model", "progen2-small", "--context", "1MK", "--num-samples", "2", "--max-length", "64", "--rng-seed", "7", "--t", "0.5", "--fp16", "false", "--rng-deterministic", "false"])
    assert rc == 0, err
    assert out == "1MK\n\n0\n1MKSEQ7\n\n1\n1MKSEQ7\ndone.\n"                                            # stdout = the block, nothing else
    assert [e[0] for e in log] == ["load", "hold", "unit"]                                            # ONE load, the call's own batch size held, ONE unit
    assert log[0][1] == "progen2-small" and log[0][2] == {"fp16": False, "device": "cuda:0", "rng_seed": 42, "rng_deterministic": False, "max_length": 64}   # sample.py's own settings as it types them; the load seeds with the stock default, the unit with --rng-seed
    assert log[1][1] == [(2, 64)] and log[2] == ("unit", "1MK", 64, 2, 7, 0.5, 0.95, False)
    lines = err.splitlines()
    active = f"[progen2-opt] ACTIVE mode=exact variant=small route=sample kit=opt/serving/pipeline_v0_4 on={','.join(PLANNED)}"
    assert active in lines, err
    i = lines.index(active)
    assert lines[i + 1] == "[progen2-opt] load variant=small t=1.50s slots=2 max_length=256" and re.match(r"^\[progen2-opt\] ready variant=small t=[0-9]+\.[0-9]{2}s$", lines[i + 2]), lines
    assert lines[i + 3].startswith("[progen2-opt] item item0000 ") and "FAILED" not in lines[i + 3]
    assert lines[i + 4].startswith("[progen2-opt] PEAK item=item0000 alloc_gib=12.35 reserved_gib=13.50 pid=")
    assert sorted(p.name for p in tmp_path.iterdir()) == [], "a single call writes nothing"


def test_sample_job_is_one_load_with_its_batch_sizes_held_and_a_named_item_failure(tmp_path, monkeypatch, capsys):
    _active(monkeypatch, tmp_path, "sample"); log = []
    _fake_kit(monkeypatch, log, oom_batches={16})
    items = tmp_path / "items.jsonl"
    items.write_text('{"item_id": "a", "context": "1M2", "max_length": 128}\n{"item_id": "b", "context": "1MK2", "num_samples": 16, "rng_seed": 7}\n{"item_id": "c", "context": "1MKV2", "max_length": 512}\n')
    rc, err, out = _run(capsys, ["sample", "--model", "progen2-small", "--input", str(items), "--out_dir", str(tmp_path / "o")])
    assert rc == cli.EXIT_FAIL and out == "", err                                                    # one item short: exit 1; stdout carries nothing for a job
    assert [e[0] for e in log] == ["load", "hold", "unit", "unit", "unit"]                            # ONE load for the job, every item attempted
    assert log[0][2]["max_length"] == 512                                                             # sized for the job's longest request
    assert log[1][1] == [(1, 128), (16, 256), (1, 512)]                                               # the job's shapes, per item, in order (the kit holds their batch sizes)
    o = tmp_path / "o" / "items"
    assert sorted(p.name for p in o.iterdir()) == ["a", "c"] and (o / "a" / "block.txt").read_text() == "1M2\n\n0\n1M2SEQ42\ndone.\n"   # block.txt per completed item, nothing for the failed one, nothing else
    assert sorted(p.name for p in (o / "a").iterdir()) == ["block.txt"]
    lines = err.splitlines()
    failed = [l for l in lines if l.startswith("[progen2-opt] item b ")]
    assert len(failed) == 1 and " FAILED OUT OF MEMORY at sample N=16 L=256: " in failed[0] and "no fallback applied" in failed[0], failed
    assert any(l.startswith("[progen2-opt] load variant=small t=1.50s slots=1 max_length=512") for l in lines), lines
    assert sum(l.startswith("[progen2-opt] PEAK item=") for l in lines) == 2                        # a PEAK line per completed item


def test_sample_refuses_by_name_when_the_load_did_not_install_a_lever(tmp_path, monkeypatch, capsys):
    """A mode is all of its levers: the levers the load reports ARE the composition or nothing is generated — NOT ACTIVE by name, exit 3, no
    ACTIVE / load / ready line, no unit."""
    _active(monkeypatch, tmp_path, "sample"); log = []
    _fake_kit(monkeypatch, log, levers=["oneread_mmap", "sampler_exact"])
    rc, err, out = _run(capsys, ["sample", "--model", "progen2-small"])
    assert rc == cli.EXIT_NOT_ACTIVE and out == "", err
    assert "[progen2-opt] NOT ACTIVE: resident_rotary cannot run: not installed by the load; static_kv cannot run: not installed by the load — nothing generated (mode=exact variant=small route=sample); " + report.REFUSED_ESCAPE in err.splitlines()
    assert "ACTIVE mode=" not in err.replace("NOT ACTIVE", "") and "[progen2-opt] load " not in err and [e[0] for e in log] == ["load", "close"]   # its own load undone


def test_a_load_that_runs_out_of_memory_is_named_and_exits_1(tmp_path, monkeypatch, capsys):
    _active(monkeypatch, tmp_path, "sample"); log = []
    _fake_kit(monkeypatch, log, load_raises=_OutOfMemory({"error": "OUT OF MEMORY at static K/V allocation for sample N=64 L=512: CUDA out of memory; no fallback applied", "oom": True}))
    rc, err, out = _run(capsys, ["sample", "--model", "progen2-small"])
    assert rc == cli.EXIT_FAIL and out == "" and "[progen2-opt] sample FAILED: OutOfMemory: OUT OF MEMORY at static K/V allocation for sample N=64 L=512" in err, err
    _fake_kit(monkeypatch, log, load_raises=KeyError("boom"))                                         # anything else is the traceback: named, then raised (exit 1 by the interpreter)
    with pytest.raises(KeyError):
        cli.main(["sample", "--model", "progen2-small"])
    assert "[progen2-opt] sample FAILED: KeyError: 'boom'" in capsys.readouterr().err


def test_refused_activation_generates_nothing(tmp_path, monkeypatch, capsys):
    log = []
    _fake_kit(monkeypatch, log)
    monkeypatch.setattr(stack, "activate", lambda mode, variant=None, **kw: {"active": False, "mode": "exact", "variant": "small", "reason": "stock files differ from the pinned commit (sample.py: 00000000 != 6451430c)", "logged": True})
    rc, err, out = _run(capsys, ["sample", "--model", "progen2-small", "--mode", "exact"])
    assert rc == cli.EXIT_NOT_ACTIVE and out == "" and log == []
    seen = {}
    monkeypatch.setattr(stack, "activate", lambda mode, variant=None, **kw: (seen.update(kw), {"active": False, "mode": "exact", "variant": "small", "reason": stack.DEVICE_CPU_REFUSAL.format(device="cpu"), "escape": stack.CPU_ESCAPE, "logged": True})[1])
    rc, err, out = _run(capsys, ["sample", "--model", "progen2-small", "--device", "cpu"])
    assert rc == cli.EXIT_NOT_ACTIVE and seen["device"] == "cpu" and seen["outside_defaults"] == [] and log == []   # --device cpu reaches the activation, which refuses it by name: 3, nothing loaded


def test_score_counted_path_contradiction_is_partial_and_kit_refused_is_3_by_name(tmp_path, monkeypatch, capsys):
    """score's exit rule at the CLI: the kit's own evidence contradicting the composition is a PARTIAL activation — 3, the partial-exit line, the
    outputs stay; a KitRefused escaping the verb is 3 by name (main's rule); --fp16 / --rng-deterministic / --sanity reach the route as likelihood.py types them."""
    _active(monkeypatch, tmp_path, "score")
    seen = []

    def fake_run_items(units, variant, rep, emit, **kw):
        seen.append(kw)
        for it in units:
            emit(it, "ll_sum=-1.0\nll_mean=-0.5\n")
        return {"n_items": len(units), "n_complete": len(units), "rc": 0, "partial": ["counted path: eager 0 != forwards 2"], "on": ["rotary_tables"], "notes": []}
    monkeypatch.setattr(_score, "run_items", fake_run_items)
    rc, err, out = _run(capsys, ["score", "--model", "progen2-small", "--mode", "exact", "--context", "1MKV2", "--fp16", "false", "--rng-deterministic", "False", "--sanity", "true"])
    assert rc == cli.EXIT_NOT_ACTIVE and out == "ll_sum=-1.0\nll_mean=-0.5\n"                          # the outputs stay; the exit is 3
    assert seen[-1] == {"seed": 42, "fp16": False, "device": "cuda:0", "rng_deterministic": False, "sanity": True}
    assert "[progen2-opt] NOT ACTIVE: partial activation — counted path: eager 0 != forwards 2; the outputs stay (mode=exact variant=small route=score); exit 3" in err

    class KitRefused(RuntimeError):
        pass
    monkeypatch.setattr(_score, "run_items", lambda *a, **k: (_ for _ in ()).throw(KitRefused("ew levers ['gelu'] cannot run: libew_progen2.so failed to load")))
    rc, err, out = _run(capsys, ["score", "--model", "progen2-small", "--mode", "exact", "--context", "1MKV2"])
    assert rc == cli.EXIT_NOT_ACTIVE and "[progen2-opt] NOT ACTIVE: the kit refused: ew levers ['gelu'] cannot run" in err, err


def test_the_programmatic_surface_is_the_routes_own_functions():
    """api.py: load / sample / score / close over generate.py and score.py — importable without torch (nothing loads at import), the stock's own
    flag names and defaults as keyword arguments."""
    assert api.sample is generate.sample and api.score is _score.score and api.__all__ == ["load", "sample", "score", "close", "Sampled", "Scored"]
    p = inspect.signature(api.sample).parameters
    assert list(p) == ["h", "context", "max_length", "num_samples", "rng_seed", "t", "p", "rng_deterministic"]
    assert (p["context"].default, p["max_length"].default, p["num_samples"].default, p["rng_seed"].default, p["t"].default, p["p"].default, p["rng_deterministic"].default) == ("1", 256, 1, 42, 0.2, 0.95, True)   # sample.py's own defaults
    q = inspect.signature(api.load).parameters
    assert list(q)[:6] == ["model", "route", "fp16", "device", "rng_seed", "rng_deterministic"] and (q["route"].default, q["fp16"].default, q["device"].default) == ("sample", True, "cuda:0")
    g = inspect.signature(generate.load).parameters
    assert "shapes" in g and "max_length" in g and g["rng_seed"].default == 42 and "expect" not in g   # the work in front of the model sizes the load; no announce
    assert list(inspect.signature(_score.score).parameters) == ["h", "context", "item_id"] and "sanity" in inspect.signature(_score.load).parameters
    with pytest.raises(ValueError, match="route 'warm': sample | score"):
        api.load("progen2-small", route="warm")
    r = generate.Sampled("1\n\n0\nMK\ndone.\n", [[3, 4]], {"unit_s": 0.1})
    assert (r.block, r.ids, r.record) == ("1\n\n0\nMK\ndone.\n", [[3, 4]], {"unit_s": 0.1})
    assert generate.OUT_OF_MEMORY == "OutOfMemory" and generate.outside_defaults(True, True, "cuda:0") == [] and generate.outside_defaults(False, False, "cuda:1") == ["fp16=false", "rng_deterministic=false", "device=cuda:1"]
