"""The wrapper's printed `[af3-jax-opt]` lines on the stub box — every `[af3-jax-opt] …` line `pred` prints (ACTIVE, WEIGHTS, BUCKETS, SCRIPT, COMMAND,
LEVER, SERVED, KERNELS, CACHE, TOKAMAX, BIG, PHASE-NOTE, TEMPLATES, DONE …) for exact, fast, big in each region and big --n_gpu 2 —
byte for byte against tests/printed_lines.json after the volatile tokens are masked (temporary paths, digests, seconds, pids, timestamps).
A change to any of these lines is deliberate and re-renders the record on the same stub:
    PRINTED_LINES_WRITE=1 python -m pytest af3_jax_opt/tests/test_printed_lines.py
"""
import inspect
import json
import os
import re

import pytest

from af3_jax_opt import cli, modes, stack

HERE = os.path.dirname(os.path.abspath(__file__))
RECORD = os.path.join(HERE, "printed_lines.json")
PREFIX = "[af3-jax-opt] "

CASES = {                                                                 # case -> (mode, extra argv, classes to warm first, visible GPU count)
    "exact": ("exact", [], ["exact"], 1),
    "fast": ("fast", [], ["fast"], 1),
    "big_region_fast": ("big", [], ["fast", "big"], 1),           # the stub input is small: the size rule resolves region fast (the fast line's program and class)
    "big_region_reach": ("big", ["--n_est", "4000"], ["fast", "big"], 1),
    "big_x2": ("big", ["--n_gpu", "2", "--n_est", "4000"], ["fast", "big"], 2),
}


def _mask(line: str, root: str) -> str:
    """Mask what differs between two runs of the same package: the box's temporary root, the kit's and the core's checkout paths, the start-up hook's
    temporary directory, digests, wall-clock tokens, record timestamps and pids. Versions, counts, lever words and every other token stay literal."""
    kit_root = os.path.dirname(os.path.dirname(stack.opt_home()))         # .../af3_jax (opt_home = .../af3_jax/opt)
    for path, token in ((stack.core_dir(), "<CORE>"), (os.path.realpath(stack.core_dir()), "<CORE>"), (root, "<TMP>"), (os.path.realpath(root), "<TMP>"),
                        (kit_root, "<KIT>"), (os.path.realpath(kit_root), "<KIT>")):
        line = line.replace(path, token)
    line = re.sub(r"\S*/af3_jax_opt_hook_[A-Za-z0-9_]+", "<HOOK>", line)
    line = re.sub(r"\b[0-9a-f]{12,64}\b", "<H>", line)                    # digests (script, cache, weights)
    line = re.sub(r"\d{8}T\d{6}(\.\d+)?Z", "<TS>", line)                 # record timestamps
    line = re.sub(r"\b(wall|total_s|lm_s|trunk_s|sampler_s|conf_s|load_s|seconds)=[0-9.]+s?", lambda m: m.group(1) + "=<S>", line)   # wall-clock tokens
    line = re.sub(r"took [0-9.]+ s(econds)?", "took <S> s", line)
    return _normalise(line)


def _normalise(line: str) -> str:
    """The box-independent form of a masked line (applied to the fresh lines and to the stored record alike, so a record rendered on one box
    compares on any other): the PEAK-NOTE detail is prose about the interpreter the probe rode (its text and truncation point move with that
    path) — the note's scope, status and record path are compared, the prose is not; the core's location is one token wherever it is installed;
    a pid, and any absolute path that is not under the box's root or the kit, are tokens."""
    line = re.sub(r"(PEAK-NOTE .*?detail=)\S+", r"\1<DETAIL>", line)
    line = line.replace("<KIT>/common/opt_core", "<CORE>")
    line = re.sub(r"\bpid[= ]\d+", "pid=<P>", line)
    line = re.sub(r"(?<![\w<>/.\-])/[A-Za-z0-9._@+\-]+(?:/[A-Za-z0-9._@+\-]+)+", "<PATH>", line)   # an absolute path of two or more components left after the root/kit/core tokens
    return line


def _render(box, capsys, monkeypatch, case: str):
    mode, extra, warm, gpus = CASES[case]
    from .conftest import GPU_H100
    monkeypatch.setattr(stack, "gpu_info", lambda index=0: {**GPU_H100, "count": gpus})
    kw = {"executables": False} if "executables" in inspect.signature(box.warm_cache).parameters else {}
    for m in warm:
        box.warm_cache(mode=m, **kw)
    stack._REPORT = None; stack._LAUNCHED.clear()
    inp = box.input_json("rec", seeds=(1,))
    out = os.path.join(box.root, f"out_{case}")
    rc = cli.main(["pred", "--variant", "p2", "--mode", mode, "--json_path", inp, "--output_dir", out, *extra])
    o = capsys.readouterr()
    lines = [_mask(ln.rstrip(), box.root) for ln in (o.err + o.out).splitlines() if ln.startswith(PREFIX)]
    return {"rc": rc, "lines": lines}


@pytest.mark.skipif("big" not in modes.MODES, reason="the memory mode's rows are not in this tree's mode table")
@pytest.mark.parametrize("case", list(CASES))
def test_printed_lines_are_byte_identical(box, capsys, monkeypatch, case):
    got = _render(box, capsys, monkeypatch, case)
    if os.environ.get("PRINTED_LINES_WRITE"):
        rec = json.load(open(RECORD)) if os.path.exists(RECORD) else {}
        rec[case] = got
        with open(RECORD, "w") as f:
            json.dump(rec, f, indent=1, sort_keys=True); f.write("\n")
        return
    rec = json.load(open(RECORD))
    assert case in rec, f"no record for {case}: re-render tests/printed_lines.json"
    want = rec[case]
    assert got["rc"] == want["rc"], (got["rc"], want["rc"], "\n".join(got["lines"][-3:]))
    got_l, want_l = got["lines"], [_normalise(l) for l in want["lines"]]          # the record may have been rendered on another box: compare in the box-independent form
    assert got_l == want_l, "\n".join(f"- {a}\n+ {b}" for a, b in zip(want_l, got_l) if a != b) + f"\n(len {len(want_l)} vs {len(got_l)})"
