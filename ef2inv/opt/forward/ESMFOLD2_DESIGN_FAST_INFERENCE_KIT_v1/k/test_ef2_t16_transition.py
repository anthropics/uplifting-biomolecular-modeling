"""Unit tests for ef2_t16_transition — the one-kernel sm_90a pair-transition forward — and its place as K-D3 lean's forward
(ef2_autograd_kernels transition="refround_lean", transition_fwd="t16").  Random-initialised fork modules; a CUDA device of compute capability
9.0 exercises the kernel, any other device exercises the step-aside-by-name path (the same tests, the other branch).

    pytest -q k/test_ef2_t16_transition.py

Classes: the kernel's output is within one bf16 ulp of |out| of K-D3's forward and no farther from an fp32 evaluation than K-D3 is (x 1.25);
repeated launches are BITWISE equal; K-D3 lean's backward behind the kernel's forward is BITWISE equal to K-D3 lean's own (same saved input,
same recompute); the shipped cubin / manifest / source agree byte for byte (ef2_t16_nvjit check without a device)."""
import os, sys, pytest, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
import transformers.models.esmfold2.modeling_esmfold2_common as C
import ef2_autograd_kernels as agk
import ef2_t16_transition as t16
import ef2_t16_nvjit as nvjit

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
AMP = torch.amp.autocast("cuda", dtype=torch.bfloat16)
SM90 = torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (9, 0)


def make_transition(seed=0):
    torch.manual_seed(seed)
    blk = C.PairUpdateBlock(d_pair=256, expansion_ratio=4)
    for m in blk.modules():
        if isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.normal_(m.weight, 1.0, 0.1); torch.nn.init.normal_(m.bias, 0.0, 0.1)
    return blk.to(dev).eval().requires_grad_(False)


def test_carried_cubin_manifest_and_source_agree():
    """No device needed: the manifest entry for the shipped kernel exists, its source key is this tree's .cuh + options, the cubin's sha256 and
    size are the manifest's, the build is the spill-free one the loader accepts (ef2_t16_nvjit.check_shipped raises by name otherwise)."""
    recs = nvjit.check_shipped()
    assert [r["file"] for r in recs] == ["ef2_transition_cute.cubin"] and recs[0]["spill_stores"] == 0 == recs[0]["spill_loads"] and recs[0]["regs_ptxas"] == 168
    man = nvjit.shipped_manifest()
    assert man["arch"] == "sm_90a" and man["cubins"]["ef2_transition_cute"]["lever"] == "t16" and man["cubins"]["ef2_transition_cute"]["canary_digest"]
    assert t16.SMEM_BYTES == 230512 == man["cubins"]["ef2_transition_cute"]["smem"]


@cuda
def test_engage_is_live_on_sm90_and_steps_aside_by_name_elsewhere():
    t16.reset()
    d = t16.engage()
    if SM90:
        assert d["state"] == "on" and t16.live() and d["reason"] is None, d
        k = d["kernel"]
        assert k["cubin_source"] == "shipped" and k["regs"] == 168 and k["spill_bytes"] == 0 and k["arch"] == "sm_90a" and k["canary"].startswith("pass:102:bitwise:"), k
    else:
        assert d["state"] == "stepped_aside" and not t16.live() and d["reason"].startswith("not_sm_90:") and "sm_90a" in d["reason_text"], d   # the loader's own sentence names the arch (measured on an A100: "... is sm_80; the kernel is sm_90a (compute capability 9.0: H100 / H200) ...")
    assert t16.engage()["state"] == d["state"]                       # idempotent


@cuda
def test_forced_step_aside_keeps_kd3_forward():
    """A stack that cannot load the kernel (here: the loader told the device is unfit) is recorded, not raised; K-D3 lean then runs its own
    forward (forward_for -> None, counted not_live) and the node's output equals plain K-D3 lean bitwise."""
    t16.reset()
    orig = nvjit.require_device

    def unfit(*a, **k):
        raise nvjit.NvjitUnavailable("ef2_t16_transition: pretend device is sm_80; the CUDA C++ / CuTe kernels (wgmma / TMA / setmaxnreg) are sm_90a only")
    nvjit.require_device = unfit
    try:
        d = t16.engage()
        assert d["state"] == "stepped_aside" and d["reason"] == "not_sm_90:sm_80" and not t16.live()
        tr = make_transition(5).pair_transition
        x = (torch.randn(1, 24, 24, 256, device=dev) * 3).to(torch.bfloat16)
        outs = []
        for fwd in (None, "t16"):
            cache = {}
            with torch.no_grad(), AMP:
                outs.append(agk.transition_refround(x, tr, cache, lean=True, fwd=fwd))
        assert torch.equal(outs[0], outs[1]) and t16.stats()["served"] == 0 and t16.stats()["fallback"].get("not_live") == 1
    finally:
        nvjit.require_device = orig
        t16.reset()


@pytest.mark.skipif(not SM90, reason="compute capability 9.0 required for the kernel itself")
@pytest.mark.parametrize("N", [40, 67, 128])
def test_kernel_forward_vs_kd3_and_fp32(N):
    """On the fork's Transition with K-D3's cached weights: |t16 - K-D3| <= 1 bf16 ulp of |out| on >= 99.9 % of elements, t16 no farther from the
    fp32 evaluation than K-D3 (x 1.25 slack, max-abs and rms), and 5 launches bitwise equal."""
    t16.reset(); assert t16.engage()["state"] == "on"
    tr = make_transition(N).pair_transition
    x = (torch.randn(1, N, N, 256, device=dev) * 3).to(torch.bfloat16); x2d = x.view(-1, 256)
    cache = {}
    with torch.no_grad(), AMP:
        o_kd3 = agk.transition_refround(x, tr, cache, lean=True).view(-1, 256)
    w = agk._transition_weights3(tr, cache)
    fwd = t16.forward_for(w, C._EPS)
    assert fwd is not None and t16.servable_weights(w)
    o = fwd(x2d)
    for _ in range(4):
        assert torch.equal(fwd(x2d), o)
    ref = t16.reference_fp32(x2d, tr.norm, tr.ffn, master=True)
    scale = ref.abs().max(); ulp = scale * 2.0 ** -8
    frac_off = ((o.float() - o_kd3.float()).abs() > ulp).float().mean().item()
    e_t16 = (o.float() - ref).abs().max().item(); e_kd3 = (o_kd3.float() - ref).abs().max().item()
    r_t16 = ((o.float() - ref).norm() / ref.norm()).item(); r_kd3 = ((o_kd3.float() - ref).norm() / ref.norm()).item()
    assert frac_off <= 1e-3 and e_t16 <= 1.25 * e_kd3 + 1e-6 and r_t16 <= 1.25 * r_kd3 + 1e-9, (N, frac_off, e_t16, e_kd3, r_t16, r_kd3)
    assert w["t16_pack"]["pack"]["W12"] is w["W12"] and w["t16_pack"]["pack"]["W3"] is w["W3"]        # descriptors over K-D3's own bf16 weights: no second copy


@pytest.mark.skipif(not SM90, reason="compute capability 9.0 required for the kernel itself")
def test_kd3_lean_backward_behind_the_kernel_is_bitwise_kd3():
    """K-D3 lean with the t16 forward: dX BITWISE equal to K-D3 lean's own (the backward reads only the saved block input and recomputes LN + W12
    with the same kernels); the forward within 1 ulp (99.9 %); the block-level patched path (agk.enable transition_fwd="t16") engages it and counts."""
    t16.reset()
    blk = make_transition(7); tr = blk.pair_transition
    N = 48; x = (torch.randn(2, N, N, 256, device=dev) * 3).to(torch.bfloat16); gout = torch.randn_like(x)
    res = []
    for fwd in (None, "t16"):
        if fwd:
            assert t16.engage()["state"] == "on"
        xx = x.clone().requires_grad_(True); cache = {}
        with torch.enable_grad(), AMP:
            o = agk.transition_refround(xx, tr, cache, lean=True, fwd=fwd)
        o.backward(gout); res.append((o.detach().float(), xx.grad.detach()))
    assert torch.equal(res[0][1], res[1][1]), (res[0][1] - res[1][1]).abs().max().item()
    scale = res[0][0].abs().max(); ulp = scale * 2.0 ** -8
    assert (((res[0][0] - res[1][0]).abs() > ulp).float().mean().item()) <= 1e-3 and F.cosine_similarity(res[0][0].flatten(), res[1][0].flatten(), dim=0).item() >= 0.999999
    served0 = t16.stats()["served"]
    # block level through the patch: 1 grad-mode call -> 1 kernel launch in the forward (+0 in backward: the recompute is K-D3's LN + GEMM)
    mask = torch.ones(2, N, N, device=dev)
    blk.set_kernel_backend(None); blk.set_chunk_size(None)
    cfg = agk.enable(blk, trimul=None, transition="refround_lean", transition_fwd="t16", checkpoint="none", only_under_grad=False)
    assert agk.describe(cfg) == "trimul=None transition=refround_lean fwd=t16 ckpt=none ln=fp32 eo=bf16"
    zz = x.clone().requires_grad_(True)
    with torch.enable_grad(), AMP:
        ob = blk(zz, pair_attention_mask=mask)
    ob.backward(gout)
    agk.disable(blk)
    assert t16.stats()["served"] == served0 + 1 and not t16.stats()["fallback"]
    try:
        agk.enable(blk, trimul=None, transition="refround", transition_fwd="t16")        # rides on lean only (the non-lean backward reads the saved lin)
    except ValueError:
        pass
    else:
        raise AssertionError("transition_fwd='t16' with transition='refround' must be refused by name")
    finally:
        agk.disable(blk)


@pytest.mark.skipif(not SM90, reason="compute capability 9.0 required for the kernel itself")
def test_kernel_under_cuda_graph_replays_bitwise():
    """Captured in a CUDA graph (the trunk pool's case at <= 256 tokens): replay equals eager bitwise, twice."""
    t16.reset(); assert t16.engage()["state"] == "on"
    tr = make_transition(9).pair_transition
    x = (torch.randn(1, 64, 64, 256, device=dev) * 3).to(torch.bfloat16).view(-1, 256)
    w = agk._transition_weights3(tr, {}); fwd = t16.forward_for(w, C._EPS)
    eager = fwd(x)
    xs = x.clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fwd(xs)                                                     # warm-up on the side stream
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fwd(xs)
    torch.cuda.current_stream().wait_stream(s)
    for _ in range(2):
        xs.copy_(x); g.replay(); torch.cuda.synchronize()
        assert torch.equal(out, eager)
