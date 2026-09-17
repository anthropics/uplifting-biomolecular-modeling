"""The 40b's two-device pipelined forward: consecutive windows of one length run two-in-flight across vortex's layer split (stage 0 = the blocks on the first device, stage 1 = the rest) — scheduling only; every window's arithmetic is the one-at-a-time forward's."""
from __future__ import annotations


def make_pipelined_forward_windows(m):
    """The two-stage pipeline over vortex's layer split (blocks 0-24 on the first device, 25-49 on the second: model.py:682-702) with
    n micro-batches of ONE window each: stage 0 on stream s0 of device 0, stage 1 on stream s1 of device 1, the head (back-copy, final
    norm, unembed) on a third stream s0h of device 0 so the back-copy's barrier never lands on s0 (ATen copy_: a cross-device copy runs
    on the SOURCE device's current stream with a two-way barrier against the destination device's current stream). One host sync per
    call. Returns the per-window head outputs as a LIST (no concatenation): each entry is the (1, L, V) logits tensor of that window on
    device 0 — the tensor the one-at-a-time path produces for the same window: every window is a B == 1 forward through the same block
    calls, so this is scheduling, not arithmetic. The kit's per-device state sees one stream per device.
    ``fwd(ids, n_chunks) -> list[(1, L, V)]``; ``n_chunks == ids.shape[0]`` (chunks of 1)."""
    import torch
    devs = sorted({str(d) for d in m.block_idx_to_device.values()}, key=lambda d: torch.device(d).index)
    assert len(devs) == 2, f"the P1 pipeline is the 2-device split: {devs}"
    stages = [[i for i in range(len(m.blocks)) if str(m.block_idx_to_device[i]) == d] for d in devs]
    assert stages[0] == list(range(0, 25)) and stages[1] == list(range(25, 50)) and str(m.block_idx_to_device[0]) == devs[0], stages
    d0, d1 = torch.device(devs[0]), torch.device(devs[1])
    s0, s1, s0h = torch.cuda.Stream(device=d0), torch.cuda.Stream(device=d1), torch.cuda.Stream(device=d0)

    def fwd(ids, n_chunks):
        chunks = ids.chunk(n_chunks, dim=0); outs = []
        with torch.inference_mode():
            with torch.cuda.stream(s0), torch.cuda.stream(s1):            # current streams for the whole call: s0 on device 0, s1 on device 1
                for c in chunks:
                    with torch.cuda.device(d0):
                        x = m.embedding_layer(c)
                        for bi in stages[0]:
                            x, _ = m.blocks[bi](x, inference_params=None, padding_mask=None)
                    with torch.cuda.device(d1):
                        x = x.to(d1, non_blocking=True)                       # on s0 (source), s1 waits for it
                        for bi in stages[1]:
                            x, _ = m.blocks[bi](x, inference_params=None, padding_mask=None)
                    with torch.cuda.device(d0), torch.cuda.stream(s0h):       # the head: the back-copy's barrier lands on s0h, not s0
                        x = x.to(d0, non_blocking=True)
                        x = m.norm(x); x = m.unembed(x)
                        outs.append(x)
            torch.cuda.synchronize()
        return outs

    info = {"devices": devs, "stages": {d: [st[0], st[-1], len(st)] for d, st in zip(devs, stages)},
            "streams": {"stage0": f"{devs[0]} s0", "stage1": f"{devs[1]} s1", "head": f"{devs[0]} s0h"}, "form": "windows (per-window outputs, no cat)"}
    return fwd, info
