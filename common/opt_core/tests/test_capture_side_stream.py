"""GraphCache warms up and captures on ONE side stream per device. A fresh torch.cuda.Stream per capture left a cuBLAS workspace behind
for every capture (torch caches one per (handle, stream) for the process; 32 MiB on sm_90 by default), so a site that resets per item
grew by that much per item. CUDA-only: skipped by name without a device."""
from __future__ import annotations

import pytest

from opt_core.capture import graphs


def _cuda_present() -> bool:
    try:
        import torch                                       # noqa: PLC0415
        return bool(torch.cuda.is_available())
    except Exception:                                      # noqa: BLE001
        return False


cuda = pytest.mark.skipif(not _cuda_present(), reason="needs a CUDA device")


@cuda
def test_capture_side_stream_is_one_per_device_and_renewed_after_forget():
    import torch                                           # noqa: PLC0415
    dev = torch.device("cuda", torch.cuda.current_device())
    graphs.forget_capture_side_stream(dev)
    s1 = graphs.capture_side_stream(torch, dev)
    assert graphs.capture_side_stream(torch, dev) is s1
    graphs.forget_capture_side_stream(dev)
    assert graphs.capture_side_stream(torch, dev) is not s1


@cuda
def test_capture_reset_cycles_do_not_grow_allocated_memory():
    """stepgraph's pattern: capture a step that runs a cuBLAS GEMM, replay it, reset at the item boundary, repeat."""
    import torch                                           # noqa: PLC0415
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    cache = graphs.GraphCache("test.side_stream", strict=True)

    def step(x, y):
        return x @ y

    def item():
        cache.run(step, a, b)                              # admitting sighting: eager answer, warm-up, capture, verified replay
        cache.run(step, a, b)                              # replay
        cache.reset("item")

    for _ in range(2):                                     # the first workspace and the allocator's pools
        item()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    for _ in range(10):
        item()
    torch.cuda.synchronize()
    grown = torch.cuda.memory_allocated() - base
    assert cache.stats()["captures"] >= 12
    assert grown < (4 << 20), f"{grown / 2 ** 20:.1f} MiB still allocated after 10 capture/reset cycles"
