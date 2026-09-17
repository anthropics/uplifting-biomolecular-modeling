"""``launch.run_sharded``: rank 0's return value travels BY VALUE — tensors (fp32 / fp16 / bool / int64, non-contiguous views, nested in
dicts / lists / tuples) come back intact run after run, independent of when the worker process exits (no fd-shared storage whose lifetime is
the worker's). 2 gloo processes per run, repeated."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

torch = pytest.importorskip("torch")

REPEATS = int(os.environ.get("ROWPAIR_TEST_LAUNCH_REPEATS", "6"))


def _entry():
    from opt_core.mem.rowpair import dist as DD
    P, r = DD.world()
    x = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    got = DD.all_gather_rows(x[r * 2:(r + 1) * 2].contiguous(), DD.Layout(4, P, r, 2))      # one real collective before returning
    DD.barrier()
    return {"rank": r, "full": got, "view": x.t()[1:], "half": x.half(), "mask": x > 5, "idx": torch.arange(7), "nested": [(x[:1], 3)],
            "big": torch.randn(512, 512)}


def test_hostify_roundtrip_by_value():
    from opt_core.mem.rowpair.launch import _TensorBytes, _hostify, _unhostify
    x = {"a": torch.arange(6, dtype=torch.float16).reshape(2, 3).t(), "b": [torch.ones(3, dtype=torch.bool), 5, "s"]}
    h = _hostify(x)
    assert isinstance(h["a"], _TensorBytes) and isinstance(h["b"][0], _TensorBytes) and h["b"][1:] == [5, "s"]
    y = _unhostify(h)
    assert torch.equal(y["a"], x["a"]) and y["a"].dtype == torch.float16 and torch.equal(y["b"][0], x["b"][0])


def test_mp_rank0_result_tensors_by_value_repeated():
    from opt_core.mem.rowpair import launch
    x = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    for i in range(REPEATS):
        res = launch.run_sharded(2, _entry, cpu_ok=True, backend="gloo", run_timeout_s=300)
        assert res["rank"] == 0, (i, res["rank"])
        assert torch.equal(res["full"], x) and torch.equal(res["view"], x.t()[1:]) and torch.equal(res["half"], x.half())
        assert torch.equal(res["mask"], x > 5) and torch.equal(res["idx"], torch.arange(7)) and torch.equal(res["nested"][0][0], x[:1])
        assert res["nested"][0][1] == 3 and tuple(res["big"].shape) == (512, 512) and res["big"].device.type == "cpu"


def test_schedule_line_carries_every_recorded_word():
    from opt_core.mem.rowpair import evidence as EV
    EV.record_schedule(trimul_RA=16, trimul_RB=32, replicated="m,s,x_atoms")
    s = EV.schedule_line("kit-opt", 1, 4)
    assert s.startswith("[kit-opt] SCHEDULE ") and "rank=1" in s and "P=4" in s and "trimul_RA=16" in s and "replicated=m,s,x_atoms" in s, s
