"""GPU numerics of the carried `apb_attn` kernel and the `opt_core.attn.apb_core` entry: agreement with the fp64 statement at or near the
16-bit statement's own error (shared / per-sample / no mask incl. an all-masked row, gate on / off, head dims 24 / 32 / 48 / 64, token counts
that are not tile multiples, a ragged key count; an all-masked sample held to the uniform average), run-to-run determinism, layout `rows` (column slices of one projection output) == layout
`shnd` bitwise, a heads-last bias == the head-major bias bitwise (copied once, counted), padded 16-byte rows == dense rows bitwise,
per-sample bias planes == a loop of single-sample calls bitwise, `out=` strided destinations, and the named refusals with their census."""
import math

import pytest

torch = pytest.importorskip("torch")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")


def _core():
    from opt_core.attn import apb_core
    apb_core.kernel()
    return apb_core


def _make(S, H, N, D, NK=None, mask_kind="shared", gate=True, seed=0, dtype=None, bias_batch=1):
    dtype = dtype or torch.bfloat16
    NK = NK or N
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = "cuda"
    # engine-like strided views: q/k/v are [S, N, H*D] projection outputs viewed [S, N, H, D] and transposed to [S, H, N, D]
    q = (torch.randn(S, N, H * D, generator=g) / math.sqrt(D)).to(dev, dtype).view(S, N, H, D).transpose(1, 2)
    k = torch.randn(S, NK, H * D, generator=g).to(dev, dtype).view(S, NK, H, D).transpose(1, 2)
    v = torch.randn(S, NK, H * D, generator=g).to(dev, dtype).view(S, NK, H, D).transpose(1, 2)
    bias = (torch.randn(bias_batch, H, N, NK, generator=g) * 2.0).to(dev, dtype)
    bias = bias[0] if bias_batch == 1 else bias
    mask = None
    if mask_kind == "shared":
        mask = (torch.rand(1, NK, generator=g) > 0.15).to(dev, torch.float32)
    elif mask_kind == "per_row":
        mask = (torch.rand(S, NK, generator=g) > 0.15).to(dev, torch.float32)
        mask[0] = 0.0                                                     # an all-masked row: the stock statement's uniform average, never NaN
    gt = (torch.randn(S, N, H * D, generator=g) * 2.0).to(dev, dtype) if gate else None
    return q, k, v, bias, mask, gt


def _err(x, ref):
    return float((x.double() - ref).abs().max() / ref.abs().max().clamp_min(1e-30))


def _stmt16(q, k, v, bias, mask, gate):
    """the materialised statement evaluated in the operands' 16-bit dtype (the class the kernel's error is compared with)."""
    K = _core().kernel()
    return K.reference(q, k, v, bias, mask, gate, scale=1.0, dtype=q.dtype)


CASES = [  # S, H, N, D, NK, mask, gate, bias_batch
    (5, 16, 203, 48, None, "shared", True, 1),
    (1, 16, 333, 24, None, "shared", True, 1),
    (3, 4, 130, 32, None, "per_row", False, 1),
    (3, 8, 150, 16, None, "per_row", True, 1),
    (2, 16, 129, 64, None, "per_row", True, 2),
    (2, 16, 129, 64, None, "none", True, 1),
    (5, 16, 61, 48, None, "shared", True, 1),
    (4, 8, 77, 24, None, "shared", True, 4),                              # per-sample bias planes (the confidence head's batched samples)
    (2, 4, 5, 24, 131, "shared", False, 1),                              # ragged: 5 queries over 131 keys
    (1, 2, 1, 24, None, "none", False, 1),
]


@cuda
@pytest.mark.parametrize("S,H,N,D,NK,mask_kind,gate,SB", CASES)
def test_kernel_matches_the_fp64_statement_within_the_16bit_class(S, H, N, D, NK, mask_kind, gate, SB):
    K = _core().kernel()
    q, k, v, bias, mask, gt = _make(S, H, N, D, NK=NK, mask_kind=mask_kind, gate=gate, bias_batch=SB)
    ref = K.reference(q, k, v, bias, mask, gt, scale=1.0)
    out = K.apb_attention(q, k, v, bias, mask=mask, gate=gt, scale=1.0)
    again = K.apb_attention(q, k, v, bias, mask=mask, gate=gt, scale=1.0)
    assert torch.equal(out, again), "run-to-run determinism"
    assert bool(torch.isfinite(out).all())
    rows = slice(None)
    if mask_kind == "per_row":                                             # sample 0 is all-masked: held to the uniform average of v separately
        rows = slice(1, None)
        uni = v[0].double().mean(dim=1).reshape(1, H * D).expand(N, H * D)   # softmax over equal (-1e9 + O(1)) fp32 logits = uniform to 2^-24
        gate0 = torch.sigmoid(gt[0].double()) if gt is not None else 1.0
        assert float((out[0].double() - uni * gate0).abs().max()) <= 2e-2 * float(v.abs().max()), "all-masked row = uniform average"
    e_ours, e_16 = _err(out[rows], ref[rows]), _err(_stmt16(q, k, v, bias, mask, gt)[rows], ref[rows])
    assert e_ours <= max(2.0 * e_16, 1.5e-2), (e_ours, e_16)


@cuda
def test_rows_layout_equals_shnd_layout_bitwise_and_out_destinations():
    C = _core()
    S, H, N, D = 3, 16, 150, 48
    g = torch.Generator(device="cpu").manual_seed(3)
    qkvg = torch.randn(S * N, 4 * H * D, generator=g).to("cuda", torch.bfloat16)       # ONE projection output; q pre-scaled by the caller
    HD = H * D
    bias = torch.randn(H, N, N, generator=g).to("cuda", torch.bfloat16)
    mask = (torch.rand(1, N, generator=g) > 0.2).to("cuda")
    o_rows = C.pair_bias_attention(qkvg[:, :HD], qkvg[:, HD:2 * HD], qkvg[:, 2 * HD:3 * HD], bias, mask, gate=qkvg[:, 3 * HD:], num_samples=S,
                                   num_heads=H, layout="rows", scale=1.0)
    as4 = lambda t: t.reshape(S, N, H, D).transpose(1, 2)  # noqa: E731
    o_shnd = C.pair_bias_attention(as4(qkvg[:, :HD]), as4(qkvg[:, HD:2 * HD]), as4(qkvg[:, 2 * HD:3 * HD]), bias, mask,
                                   gate=qkvg[:, 3 * HD:].reshape(S, N, HD), layout="shnd", scale=1.0)
    assert torch.equal(o_rows.view(S, N, HD), o_shnd)
    dest = torch.zeros(S * N, 2 * HD, device="cuda", dtype=torch.bfloat16)              # a strided 2-D destination (a column slice of a wider block)
    ret = C.pair_bias_attention(qkvg[:, :HD], qkvg[:, HD:2 * HD], qkvg[:, 2 * HD:3 * HD], bias, mask, gate=qkvg[:, 3 * HD:], num_samples=S,
                                num_heads=H, layout="rows", out=dest[:, HD:], scale=1.0)
    assert ret.data_ptr() == dest[:, HD:].data_ptr() and torch.equal(dest[:, HD:], o_rows) and float(dest[:, :HD].abs().max()) == 0.0


@cuda
def test_bias_layouts_serve_the_same_bits_and_are_counted():
    C = _core()
    K = C.kernel()
    S, H, N, D = 4, 16, 131, 24                                                        # N not a multiple of 8: dense rows are unaligned
    q, k, v, bias, mask, gt = _make(S, H, N, D, seed=5)
    ev0 = dict(C.census()["events"])
    o_dense = C.pair_bias_attention(q, k, v, bias, mask, gate=gt, layout="shnd", scale=1.0)
    ev1 = dict(C.census()["events"])
    assert ev1.get("rows_unaligned", 0) == ev0.get("rows_unaligned", 0) + 1
    ld = (N + 7) // 8 * 8
    padded = torch.zeros(H, N, ld, device="cuda", dtype=bias.dtype); padded[..., :N] = bias
    o_pad = C.pair_bias_attention(q, k, v, padded[..., :N], mask, gate=gt, layout="shnd", scale=1.0)
    assert torch.equal(o_pad, o_dense)
    assert C.census()["events"].get("rows_unaligned", 0) == ev1.get("rows_unaligned", 0)          # aligned rows: no event
    heads_last = bias.permute(1, 2, 0).contiguous().permute(2, 0, 1)                   # the stock permute view of an [N, N, H] projection output
    n_copy = C.census()["events"].get("relayout_copy", 0)
    o_hl = C.pair_bias_attention(q, k, v, heads_last[None, None], mask, gate=gt, layout="shnd", scale=1.0)   # leading singleton dims dropped
    assert torch.equal(o_hl, o_dense) and C.census()["events"].get("relayout_copy", 0) == n_copy + 1
    # per-sample planes == a loop of single-sample calls
    planes = torch.randn(S, H, N, N, device="cuda").to(bias.dtype)
    cfg = K.default_config(S, N, D, H)                                                 # one tiling for both arms (the tile size sets the summation order)
    o_all = K.apb_attention(q, k, v, planes, mask=mask, gate=gt, scale=1.0, config=cfg)
    for s_ in range(S):
        o_one = K.apb_attention(q[s_:s_ + 1], k[s_:s_ + 1], v[s_:s_ + 1], planes[s_], mask=mask, gate=gt[s_:s_ + 1], scale=1.0, config=cfg)
        assert torch.equal(o_one[0], o_all[s_]), s_


@cuda
def test_refusals_are_named_before_any_work_and_counted():
    C = _core()
    S, H, N, D = 2, 4, 40, 24
    q, k, v, bias, mask, gt = _make(S, H, N, D, seed=9)
    before = dict(C.census()["refused"]); served0 = C.census()["served"]
    cases = [
        ("dtype:float32", dict(q=q.float(), k=k.float(), v=v.float(), pair_bias=bias, key_mask=mask, layout="shnd")),
        ("shape:mask", dict(q=q, k=k, v=v, pair_bias=bias, key_mask=mask[:, :N - 1], layout="shnd")),
        ("shape:bias", dict(q=q, k=k, v=v, pair_bias=bias[:, :N - 1], key_mask=mask, layout="shnd")),
        ("layout:cols", dict(q=q, k=k, v=v, pair_bias=bias, key_mask=mask, layout="cols")),
        ("nonfinite:inf_or_scale", dict(q=q, k=k, v=v, pair_bias=bias, key_mask=mask, layout="shnd", inf=float("inf"))),
        ("nonfinite:inf_or_scale", dict(q=q, k=k, v=v, pair_bias=bias, key_mask=mask, layout="shnd", scale=float("nan"))),
        ("layout:rows_needs_S_H", dict(q=q.reshape(-1, D), k=k.reshape(-1, D), v=v.reshape(-1, D), pair_bias=bias, key_mask=None, layout="rows")),
        ("head_dim:96", dict(q=torch.zeros(1, 1, 8, 96, device="cuda", dtype=torch.bfloat16), k=torch.zeros(1, 1, 8, 96, device="cuda", dtype=torch.bfloat16),
                             v=torch.zeros(1, 1, 8, 96, device="cuda", dtype=torch.bfloat16), pair_bias=torch.zeros(1, 8, 8, device="cuda"), key_mask=None, layout="shnd")),
        ("shape:gate", dict(q=q.transpose(1, 2).reshape(S * N, H * D), k=k.transpose(1, 2).reshape(S * N, H * D), v=v.transpose(1, 2).reshape(S * N, H * D), pair_bias=bias, key_mask=mask,
                            gate=gt.reshape(S * N, H * D)[: S * N - S], num_samples=S, num_heads=H, layout="rows")),
    ]
    for event, kw in cases:
        with pytest.raises(C.Unsupported) as ei:
            C.pair_bias_attention(kw.pop("q"), kw.pop("k"), kw.pop("v"), kw.pop("pair_bias"), kw.pop("key_mask"), **kw)
        assert ei.value.event == event, (ei.value.event, event)
    ev0 = dict(C.census()["events"])
    with pytest.raises(C.Unsupported):                                                # refused BEFORE any work: no relayout copy for an unserved call
        C.pair_bias_attention(q.float(), k.float(), v.float(), bias.permute(1, 2, 0).contiguous().permute(2, 0, 1), mask, layout="shnd")
    assert dict(C.census()["events"]) == ev0
    after = C.census()
    assert after["served"] == served0
    assert sum(after["refused"].values()) - sum(before.values()) == len(cases) + 1
    assert after["kernel"] == C.kernel().__version__


@cuda
def test_direct_launches_of_the_compiled_handle_equal_the_jit_path_bitwise():
    """After a specialization's first (JITFunction) launch the kernel launches the compiled handle directly; the outputs are the JIT path's bit
    for bit across the classes Triton specializes on (alignment of the bias rows, mask / gate present, shared / per-sample bias, S, D), the
    direct launches are counted, and a shape whose integer classes differ from a cached one takes the JIT path first (no stale binary)."""
    K = _core().kernel()
    shapes = [dict(S=1, H=16, N=203, D=24), dict(S=1, H=16, N=208, D=24), dict(S=5, H=16, N=176, D=48, mask_kind="per_row"),
              dict(S=3, H=16, N=96, D=48, gate=False, mask_kind=None), dict(S=2, H=4, N=64, D=32, bias_batch=2), dict(S=1, H=16, N=400, D=24)]
    K.set_direct_launch(False)
    ref = []
    for i, kw in enumerate(shapes):
        q, k, v, bias, mask, gate = _make(kw["S"], kw["H"], kw["N"], kw["D"], mask_kind=kw.get("mask_kind", "shared"), gate=kw.get("gate", True),
                                          seed=10 + i, bias_batch=kw.get("bias_batch", 1))
        ref.append(((q, k, v, bias, mask, gate), K.apb_attention(q, k, v, bias, mask=mask, gate=gate, scale=1.0).clone()))
    assert K.launch_census()["direct"] == "off:by_request"
    K.set_direct_launch(True)
    c0 = K.launch_census(); assert c0["direct"] == "on"
    for rep in range(3):                                                  # rep 0: first launch per specialization goes through the JIT and caches the handle; reps 1-2: direct
        for (ops, want) in ref:
            q, k, v, bias, mask, gate = ops
            got = K.apb_attention(q, k, v, bias, mask=mask, gate=gate, scale=1.0)
            assert torch.equal(got, want)
    c1 = K.launch_census()
    assert c1["direct"] == "on", c1                                       # this Triton drives the compiled handle (else every call took the JIT path, by name)
    assert c1["direct_launches"] - c0["direct_launches"] >= 2 * len(shapes) and c1["convention"] in ("all", "runtime"), c1
    assert c1["plans"] >= len(shapes), c1                                # one plan per operand metadata (the binaries behind them may be shared)
    # the entry's census carries the launch words
    assert _core().census()["launch"]["direct_launches"] == c1["direct_launches"]
