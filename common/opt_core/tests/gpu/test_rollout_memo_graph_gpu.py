"""opt_core.of3_sampler.rollout_memo under a REAL CUDA graph: a cap-skipped key re-entering the store at the next same-shape rollout must
not move or evict any buffer the captured graph reads (the graph replays the refreshed values at the captured addresses), and a capture met
while the graphs cooperation is not armed raises CaptureError by name."""
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from opt_core.of3_sampler import rollout_memo as RM  # noqa: E402


class _Sampler:
    pass


def _reset():
    RM.STORE.drop_all(); RM.STATE.update(fills=0, hits=0, cap_skips={}, bytes=0, errors={}); RM._GRAPHS["active"] = False; RM._EPOCH["depth"] = 0; RM.set_cap_gb(1.5)


def _capture(fn):
    g = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()                                                     # warm-up outside the capture
    torch.cuda.current_stream().wait_stream(side); torch.cuda.synchronize()
    with torch.cuda.graph(g):
        fn()
    return g


def test_cap_skipped_key_reentering_under_a_live_graph_keeps_every_captured_address():
    _reset(); s = _Sampler(); RM._GRAPHS["active"] = True                                       # the cooperation armed: entries outlive the rollout, refreshed in place
    val = {"n": 0.0}

    def prod(n):
        def f():
            val["n"] += 1.0
            return (torch.full((n,), val["n"], device="cuda"),)
        return f
    try:
        RM.set_cap_gb((2 * 256 * 4 + 64) / 2 ** 30)                                            # A and C fit (256 floats each); B (1M floats) never does
        RM._enter_rollout(s)
        a = RM.memo_call("A", ("A", 256), prod(256))[0]                                          # 1.0
        b = RM.memo_call("B", ("B", 1 << 20), prod(1 << 20))[0]                                  # 2.0, skipped (over cap)
        c = RM.memo_call("C", ("C", 256), prod(256))[0]                                          # 3.0
        assert RM.STORE.get(("B", 1 << 20)) is None and b.numel() == 1 << 20 and RM.STATE["cap_skips"] == {"B": 1}
        out = torch.zeros(256, device="cuda")
        g = _capture(lambda: out.copy_(a + c))                                                   # the "denoiser step": reads A and C at their stored addresses
        g.replay(); torch.cuda.synchronize()
        assert float(out[0]) == 4.0
        pa, pc = a.data_ptr(), c.data_ptr()
        RM._exit_rollout()
        RM._enter_rollout(s)                                                                     # item 2, same shapes: the graph is reused
        a2 = RM.memo_call("A", ("A", 256), prod(256))[0]                                         # 4.0 -> copied into A's storage
        RM.memo_call("B", ("B", 1 << 20), prod(1 << 20))                                          # 5.0, re-enters over cap while C is stale-epoch: sticky skip, NO eviction
        assert ("C", 256) in RM.STORE.ent and RM.STORE.ent[("C", 256)]["t"][0].data_ptr() == pc and RM.STATE["cap_skips"] == {"B": 2}
        c2 = RM.memo_call("C", ("C", 256), prod(256))[0]                                         # 6.0 -> copied into C's storage
        assert a2.data_ptr() == pa and c2.data_ptr() == pc
        g.replay(); torch.cuda.synchronize()
        assert float(out[0]) == 10.0 and float(out[255]) == 10.0                                 # the replay read this rollout's values: 4 + 6
        RM._exit_rollout()
    finally:
        _reset()


def test_a_capture_without_the_cooperation_raises_by_name():
    _reset(); s = _Sampler()
    try:
        RM._enter_rollout(s)
        x = torch.ones(8, device="cuda"); y = torch.empty(8, device="cuda")
        y.copy_(x); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        raised = {}
        try:
            with torch.cuda.graph(g):
                try:
                    RM.memo_call("Q", ("Q", 8), lambda: (x * 2,))
                except RM.CaptureError as e:
                    raised["e"] = e
                y.copy_(x)
        except Exception:                                        # a capture that saw a Python-side refusal may not end cleanly on every driver: the refusal is the point
            pass
        assert "e" in raised and "capture is under way" in str(raised["e"])
        RM._exit_rollout()
    finally:
        _reset(); torch.cuda.synchronize()
