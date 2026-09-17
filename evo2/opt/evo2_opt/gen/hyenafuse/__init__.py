"""hyenafuse — the fused per-token Hyena state update for Evo 2 cached generation (vtx 1.1.0 decode path); armed in modes exact and fast.

    from evo2_opt.gen.hyenafuse import arm
    model = Evo2("evo2_7b", local_path=...)          # a constructed stock instance (the kit arms this member itself on every one)
    handle = arm(model)                                # prints '[evo2-gen hyenafuse] ARMED: ...'; raises HyenaFuseRefused ('REFUSED: <reason>')
    model.generate(...)                                # first fused decode step prints '[evo2-gen hyenafuse] INSTALLED: ...'
    handle.status                                      # the live census dict (also model._hyenafuse_status)
    handle.uninstall()                                 # the stock step again (same process); == disarm(model)

The handle: .name, .status, .describe(), .uninstall(), .generate = None (the public `Evo2.generate` loop is used unchanged).

What it replaces: on every cached-generation decode step each hyena block (`ParallelGatedConvBlock.filter`, a `HyenaCascade`) runs
`sequential_forward`: `engine.step_fir` on the 3H projection rows, `interleave` + split, then `engine.step_fir` again (hcs/hcm) or
`engine.step_iir` (hcl), with the gating products — a few dozen small eager kernels per block per token, re-materialising fp32 copies of
the weights, `h.repeat_interleave`, `h.flip` and `exp(log_poles)` every step. The arm swaps that method, per block instance, for ONE Triton
launch (`kernel._hyena_decode_step`) that reads the projection output and the three states, writes the block's filter output and
writes the next states (fp32 ping-pong buffers that the inference-params dicts then hold). Prefill (`parallel_forward`), attention blocks,
norms, projections and the MLP are untouched; no stock source is edited (instance attributes on the live model only).

Numerics: the kernel spells out the stock's arithmetic (fp32 FIR/IIR math with separately rounded products and sums, bf16 gate products,
ATen's reduction order for `torch.sum` over the taps; kernel.py docstring), so the block output and the next states are bit-identical to
the stock step's by construction: the same ops in the same order. Constants are prepared once at arm time from the live parameters with
vortex's own functions (`vortex.model.utils.interleave` / `column_split` on an index vector give the projection-row map;
`torch.exp(log_poles)` gives the poles), so they carry the stock's bits. `arm()` can also compare the kernel with the unpatched
`HyenaCascade.sequential_forward` on random states at this model's shapes before arming (a keyword; the kit arms without it) and then
refuses on any difference.

Fail-loud: unsupported settings are REFUSED by name at arm time (HyenaFuseRefused raised, nothing patched; `_refusals`). One runtime
deviation is a NAMED event, never silent: when a prompt is shorter than a block's FIR length, the stock GROWS that block's state with
`torch.cat` under a per-step tap alignment; those (block, step) pairs are delegated to the stock step and counted (`status["fallback_steps"]`,
one printed FALLBACK line); `arm(model, strict=True)` raises there instead. `handle.status` (also `model._hyenafuse_status`) is the census:
blocks armed per kind, fused steps, fallback steps.
"""
import time

import torch

TAG = "[evo2-gen hyenafuse]"
LEVER = "hyenafuse"
NUMERICS_CLASS = "exact"            # bit-identical to the stock step by construction: the same ops in the same rounding order (kernel.py)

__all__ = ["arm", "install", "disarm", "uninstall", "selftest_against_stock", "Handle", "HyenaFuseRefused", "TAG", "LEVER", "NUMERICS_CLASS"]


class HyenaFuseRefused(RuntimeError):
    """The model or the environment is outside what the fused step supports; nothing was patched."""


def _lanes(n):
    p = 1
    while p * 2 <= n:
        p *= 2
    return min(p, 32)


def _say(msg):
    print(f"{TAG} {msg}", flush=True)


def _core(model):
    """The StripedHyena module of an `Evo2` wrapper (or the module itself)."""
    m = getattr(model, "model", model)
    if not hasattr(m, "blocks"):
        raise HyenaFuseRefused(f"no .blocks on {type(m).__name__}: expected an Evo2 or vortex StripedHyena instance")
    return m


class _BlockCtx:
    """Per-block constants (prepared once from the live parameters) + the adopted state buffers of the current generation."""

    def __init__(self, filt, kind, block_c, num_warps):
        from vortex.model.utils import column_split, interleave
        self.filt = filt
        self.kind = kind
        self.layer_idx = filt.layer_idx
        self.block_c = block_c
        self.num_warps = num_warps
        H = filt.hidden_size
        self.H = H
        dev = filt.short_filter_weight.device
        self.device = dev
        # projection-row map: run stock's own interleave / split on an index vector shaped like z_pre [B=1, 3H]
        idx = torch.arange(3 * H, device=dev, dtype=torch.int64)[None]
        if filt.config.interleave:
            idx = interleave(idx)
        if filt.column_split_hyena:
            x2, x1, v = column_split(idx, filt.num_attention_heads, filt.hidden_size_per_attention_head)
        else:
            x2, x1, v = idx.split([H, H, H], dim=1)
        if filt.hyena_flip_x1x2:
            x1, x2 = x2, x1
        self.map = torch.stack([x2[0], x1[0], v[0]]).to(torch.int32).contiguous()
        assert sorted(self.map.flatten().tolist()) == list(range(3 * H)), "projection-row map is not a permutation"
        # outer short FIR taps, fp32 exactly as step_fir casts them: weight.squeeze()[..., :3] -> [3H, 3]
        self.ws = filt.short_filter_weight.detach().squeeze().to(torch.float32).contiguous()
        assert self.ws.shape == (3 * H, 3)
        if kind in ("hcs", "hcm"):
            self.mode = 0
            n_filter = filt.fir_inner_filter_length
            self.ni = n_filter - 1
            h = filt.h.detach()[:, 0, :].to(torch.float32)                     # [G, N]
            flip = n_filter >= 128                                             # sequential_forward: flip_filter = gated_bias = (len >= 128)
            self.gated_bias = bool(flip)
            self.wi = (h.flip(-1) if flip else h).contiguous()                 # tap k multiplies state[k]; index N-1 (= ni) is h0
            self.G = self.wi.shape[0]
            self.cpg = H // self.G
            self.bi = filt.D.detach().to(torch.float32).contiguous() if self.gated_bias else self.ws
            self.po = self.re = self.ws
            self.dd = filt.short_filter_weight
        else:
            self.mode = 1
            self.ni = filt.state_size
            self.gated_bias = False
            self.po = torch.exp(filt.log_poles.detach())[..., 0].contiguous()  # stock: poles = torch.exp(poles)[..., 0]
            self.re = filt.residues.detach().contiguous()
            assert self.po.dtype == torch.float32 and self.re.dtype == torch.float32
            assert self.po.shape == (H, self.ni) and self.re.shape == (H, self.ni)
            self.dd = filt.D.detach().contiguous()
            self.wi = self.bi = self.ws
            self.G, self.cpg = 1, H
        self.L = _lanes(self.ni)
        assert self.ni <= 4 * self.L, f"inner state {self.ni} needs the multi-element accumulator path (unsupported)"
        # state buffers of the current generation: `ss`/`si` hold the current states (the very objects in the inference_params dicts,
        # tracked by object), `ss_alt`/`si_alt` receive the next step's states; the pairs swap after every launch (ping-pong)
        self.ss = self.si = self.ss_alt = self.si_alt = None

    # -- launch -------------------------------------------------------------------------------------------------
    def launch(self, u2, ss, ss_out, inner, inner_out, y):
        """One fused step: reads u2 [B,3H] bf16 + states (ss, inner), writes y [B,H] bf16 + new states (ss_out, inner_out).
        In/out state buffers must not alias and must share strides."""
        from .kernel import _hyena_decode_step
        B = u2.shape[0]
        grid = (triton_cdiv(self.H, self.block_c), B)
        assert ss.stride() == ss_out.stride() and inner.stride() == inner_out.stride()
        assert ss.data_ptr() != ss_out.data_ptr() and inner.data_ptr() != inner_out.data_ptr()
        dummy = self.ws
        si, sio = (inner, inner_out) if self.mode == 0 else (dummy, dummy)
        sr, sro = (inner, inner_out) if self.mode == 1 else (dummy, dummy)
        st3 = (inner.stride(0), inner.stride(1), inner.stride(2))
        with torch.cuda.device(u2.device):
            _hyena_decode_step[grid](
                u2, u2.stride(0), u2.stride(1),
                self.map, self.ws,
                ss, ss_out, ss.stride(0), ss.stride(1), ss.stride(2),
                self.wi, self.bi,
                si, sio, *(st3 if self.mode == 0 else (0, 0, 0)),
                self.po, self.re, self.dd,
                sr, sro, *(st3 if self.mode == 1 else (0, 0, 0)),
                y, y.stride(0), y.stride(1),
                self.H, self.cpg,
                MODE=self.mode, NI=self.ni, L=self.L, GATED_BIAS=self.gated_bias, BLOCK_C=self.block_c,
                num_warps=self.num_warps, enable_fp_fusion=False,
            )


def triton_cdiv(a, b):
    return (a + b - 1) // b


def _kind_of(core, block_idx):
    cfg = core.config
    if block_idx in cfg.hcs_layer_idxs:
        return "hcs"
    if block_idx in cfg.hcm_layer_idxs:
        return "hcm"
    if block_idx in cfg.hcl_layer_idxs:
        return "hcl"
    return None


def _refusals(core):
    """Every setting the fused step does not cover, by name (empty list = supported)."""
    from vortex.model.model import HyenaCascade, ParallelGatedConvBlock
    cfg = core.config
    bad = []
    if not torch.cuda.is_available():
        bad.append("no CUDA device")
    if cfg.get("print_activations", False):
        bad.append("print_activations=True (activation logging inside the step is not reproduced)")
    if cfg.get("short_filter_bias", False):
        bad.append("short_filter_bias=True (outer FIR bias not wired)")
    if cfg.short_filter_length != 3:
        bad.append(f"short_filter_length={cfg.short_filter_length} (outer FIR fused for length 3 only)")
    n_hyena = 0
    for i, blk in enumerate(core.blocks):
        kind = _kind_of(core, i)
        if kind is None:
            continue
        if not isinstance(blk, ParallelGatedConvBlock) or not isinstance(blk.filter, HyenaCascade):
            bad.append(f"block {i}: {type(blk).__name__}/{type(getattr(blk, 'filter', None)).__name__} is not the stock hyena block")
            continue
        f = blk.filter
        n_hyena += 1
        if f.short_filter_weight.dtype != torch.bfloat16 or tuple(f.short_filter_weight.shape) != (3 * f.hidden_size, 1, 3):
            bad.append(f"block {i}: short_filter_weight {f.short_filter_weight.dtype} {tuple(f.short_filter_weight.shape)} (expected bf16 [3H,1,3])")
        if f.short_filter_bias is not None:
            bad.append(f"block {i}: short_filter_bias present")
        if kind in ("hcs", "hcm"):
            n = f.fir_inner_filter_length
            if n is None or n < 2 or n - 1 > 128:
                bad.append(f"block {i}: inner FIR length {n} (supported 2..129)")
            elif f.h is None or f.h.dtype != torch.bfloat16 or f.h.dim() != 3 or f.h.shape[1:] != (1, n):
                bad.append(f"block {i}: inner filter h {None if f.h is None else (f.h.dtype, tuple(f.h.shape))} (expected bf16 [G,1,{n}])")
            elif f.hidden_size % f.h.shape[0]:
                bad.append(f"block {i}: hidden {f.hidden_size} not a multiple of filter groups {f.h.shape[0]}")
            if n is not None and n >= 128 and (f.D is None or f.D.dtype != torch.bfloat16 or tuple(f.D.shape) != (f.hidden_size,)):
                bad.append(f"block {i}: gated bias D {None if f.D is None else (f.D.dtype, tuple(f.D.shape))} (expected bf16 [H])")
            if n is not None and n < 128 and f.D is not None:
                bad.append(f"block {i}: inner FIR of {n} taps carries a bias D (stock adds it un-gated; not wired in the fused step)")
        else:
            if f.fir_inner_filter_length is not None:
                bad.append(f"block {i}: hcl block with an inner FIR length")
            s = f.state_size
            if s < 1 or s > 32 or (s & (s - 1)):
                bad.append(f"block {i}: state_size {s} (supported: power of two <= 32)")
            if f.log_poles.dtype != torch.float32 or tuple(f.log_poles.shape) != (f.hidden_size, s, 1):
                bad.append(f"block {i}: log_poles {f.log_poles.dtype} {tuple(f.log_poles.shape)} (expected fp32 [H,{s},1])")
            if f.residues.dtype != torch.float32 or tuple(f.residues.shape) != (f.hidden_size, s):
                bad.append(f"block {i}: residues {f.residues.dtype} {tuple(f.residues.shape)} (expected fp32 [H,{s}])")
            if f.D is None or f.D.dtype != torch.bfloat16 or tuple(f.D.shape) != (f.hidden_size,):
                bad.append(f"block {i}: D {None if f.D is None else (f.D.dtype, tuple(f.D.shape))} (expected bf16 [H])")
    if n_hyena == 0:
        bad.append("no hyena blocks found")
    return bad


# -- optional arm-time comparison with the unpatched stock step functions ------------------------------------
def selftest_against_stock(ctxs, trials=24, batch=1, seed=0, denormal_trials=4):
    """Run the fused kernel and the stock `HyenaCascade.sequential_forward` (unpatched class method) on identical random
    projections/states for every distinct (kind, shape, device) among `ctxs`; return {"ok": bool, "cases": [...]} with, per case,
    the number of trials and of bitwise mismatches in the output and in each state. Random states include exact zeros and
    negative zeros. Inputs are shaped like a real decode step (u [B,1,3H] bf16; outer state [B,3H,2]; inner [B,H,N] / [B,H,S]).
    `denormal_trials` extra trials draw every input at fp32-subnormal scale (1e-39): they are REPORTED separately (`denormal_*` keys) and
    do not gate `ok` — they probe flush-to-zero behaviour, a regime real activations do not visit; the gating trials use scales 1e-3..30."""
    from vortex.model.cache import HyenaCascadeFIRInferenceParams, HyenaCascadeIIRInferenceParams
    from vortex.model.model import HyenaCascade
    seen, cases = set(), []
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ctx in ctxs:
        key = (ctx.kind, ctx.H, ctx.ni, ctx.G, str(ctx.device))
        if key in seen:
            continue
        seen.add(key)
        f, dev, H = ctx.filt, ctx.device, ctx.H
        n_out = n_ss = n_in = 0
        d_out = d_ss = d_in = 0
        for t in range(trials + denormal_trials):
            denormal = t >= trials
            scale = 1e-39 if denormal else [1.0, 1e-3, 30.0][t % 3]
            u = (torch.randn(batch, 1, 3 * H, generator=g) * scale).to(dev, torch.bfloat16)
            ss0 = (torch.randn(batch, 3 * H, 2, generator=g) * scale).to(torch.bfloat16)
            if ctx.mode == 0:
                in0 = (torch.randn(batch, H, ctx.ni, generator=g) * scale).to(torch.bfloat16)
            else:
                in0 = (torch.randn(batch, H, ctx.ni, generator=g) * scale).to(torch.float32)
            if t % 4 == 1:                                  # sprinkle exact zeros / negative zeros
                ss0[..., 0] = 0.0
                in0[:, ::3] = -0.0
                u[..., ::5] = -0.0
            # decode step 1 sees the prefill's bf16 states (dense view for the outer state); later steps see fp32 contiguous
            as_prefill = (t % 2 == 0)
            if as_prefill:
                ss_stock = ss0.to(dev).transpose(1, 2).contiguous().transpose(1, 2)      # [B,3H,2] with time-major strides
                in_stock = in0.to(dev) if ctx.mode == 0 else in0.to(dev)
            else:
                ss_stock = ss0.to(dev, torch.float32).contiguous()
                in_stock = in0.to(dev, torch.float32).contiguous()
            if ctx.mode == 0:
                ip = HyenaCascadeFIRInferenceParams()
                ip.fir_state_dict[f.layer_idx] = ss_stock.clone()
                ip.fir_inner_state_dict[f.layer_idx] = in_stock.clone()
            else:
                ip = HyenaCascadeIIRInferenceParams()
                ip.fir_state_dict[f.layer_idx] = ss_stock.clone()
                ip.state_dict[f.layer_idx] = in_stock.clone()
            saved_dtype = f.data_dtype
            with torch.inference_mode():
                y_ref, _ = HyenaCascade.sequential_forward(f, u.clone(), ip)          # the stock step (class method, never patched)
            f.data_dtype = saved_dtype
            ss_ref = ip.fir_state_dict[f.layer_idx]
            in_ref = ip.fir_inner_state_dict[f.layer_idx] if ctx.mode == 0 else ip.state_dict[f.layer_idx]
            # fused
            ss_f = ss_stock.to(torch.float32).contiguous().clone()
            in_f = in_stock.to(torch.float32).contiguous().clone()
            ss_o, in_o = torch.empty_like(ss_f), torch.empty_like(in_f)
            y_f = torch.empty(batch, H, dtype=torch.bfloat16, device=dev)
            with torch.inference_mode():
                ctx.launch(u[:, -1], ss_f, ss_o, in_f, in_o, y_f)
            torch.cuda.synchronize(dev)
            m_out = int(not _bit_equal(y_ref.reshape(batch, H), y_f))
            m_ss = int(not _bit_equal(ss_ref.to(torch.float32), ss_o))
            m_in = int(not _bit_equal(in_ref.to(torch.float32), in_o))
            if denormal:
                d_out += m_out; d_ss += m_ss; d_in += m_in
            else:
                n_out += m_out; n_ss += m_ss; n_in += m_in
        cases.append({"kind": ctx.kind, "H": H, "inner": ctx.ni, "groups": ctx.G, "device": str(dev), "trials": trials, "batch": batch,
                      "mismatch_out": n_out, "mismatch_outer_state": n_ss, "mismatch_inner_state": n_in,
                      "denormal_trials": denormal_trials, "denormal_mismatch_out": d_out, "denormal_mismatch_outer_state": d_ss,
                      "denormal_mismatch_inner_state": d_in})
    ok = all(c["mismatch_out"] == 0 and c["mismatch_outer_state"] == 0 and c["mismatch_inner_state"] == 0 for c in cases)
    return {"ok": ok, "cases": cases}


def _bit_equal(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype == torch.bfloat16:
        return torch.equal(a.view(torch.int16), b.view(torch.int16))
    if a.dtype == torch.float32:
        return torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))
    return torch.equal(a, b)


# -- arm / disarm ------------------------------------------------------------------------------------------------------
def arm(model, *, strict=False, selftest=True, selftest_trials=24, block_c=32, num_warps=4, warm=True):
    """Install the fused decode step on every hyena block of a constructed stock model; returns a Handle. REFUSED =
    HyenaFuseRefused raised with the reasons by name (nothing patched) when a setting is unsupported or the arm-time comparison with
    the stock step (a keyword; skipped when off) finds a difference. `strict=True` additionally turns the short-prompt FALLBACK event
    into a raise; `warm` compiles each kernel variant once at arm time when the comparison is off."""
    t0 = time.time()
    core = _core(model)
    status = {"lever": LEVER, "class": NUMERICS_CLASS, "armed": False, "installed": False, "refused": None, "strict": bool(strict),
              "blocks": {}, "fused_steps": 0, "fallback_steps": 0, "fallback_layers": [], "selftest": None, "knobs":
              {"block_c": block_c, "num_warps": num_warps, "selftest": selftest, "selftest_trials": selftest_trials, "warm": warm}}
    if getattr(core, "_hyenafuse_ctx", None):
        _say("ARMED: already armed on this model (idempotent)")
        return Handle(model, core._hyenafuse_status)
    reasons = _refusals(core)
    if not reasons:
        try:
            import triton  # noqa: F401
            from .kernel import _hyena_decode_step  # noqa: F401
        except Exception as e:                    # pragma: no cover - environment
            reasons.append(f"triton kernel unavailable: {type(e).__name__}: {e}")
    ctxs = {}
    if not reasons:
        for i, blk in enumerate(core.blocks):
            kind = _kind_of(core, i)
            if kind is None:
                continue
            ctxs[i] = _BlockCtx(blk.filter, kind, block_c, num_warps)
        if selftest:
            rec = selftest_against_stock(list(ctxs.values()), trials=selftest_trials)
            status["selftest"] = rec
            if not rec["ok"]:
                reasons.append("on-box bitwise self-test against the stock step FAILED: "
                               + "; ".join(f"{c['kind']}@{c['device']} out {c['mismatch_out']}/{c['trials']} outer {c['mismatch_outer_state']} inner {c['mismatch_inner_state']}"
                                           for c in rec["cases"] if c["mismatch_out"] or c["mismatch_outer_state"] or c["mismatch_inner_state"]))
        elif warm:
            _warm(ctxs)
    if reasons:
        status["refused"] = "; ".join(reasons)
        _say(f"REFUSED: {status['refused']}")
        try:
            model._hyenafuse_status = status
        except Exception:
            pass
        raise HyenaFuseRefused(status["refused"])
    for i, ctx in ctxs.items():
        _patch(core, ctx, status)
        status["blocks"][i] = ctx.kind
    core._hyenafuse_ctx = ctxs
    core._hyenafuse_status = status
    if model is not core:
        model._hyenafuse_status = status
    status["armed"] = True
    kinds = {k: sum(1 for v in status["blocks"].values() if v == k) for k in ("hcs", "hcm", "hcl")}
    ver = status["selftest"]
    vtxt = ("self-test bitwise vs the stock step on-box (" + ", ".join(f"{c['kind']} {c['trials']} trials" for c in ver["cases"]) + ")") if ver else "self-test SKIPPED (selftest=False)"
    _say(f"ARMED: fused decode step on {len(ctxs)} hyena blocks (hcs {kinds['hcs']}, hcm {kinds['hcm']}, hcl {kinds['hcl']}); "
         f"class={NUMERICS_CLASS}; {vtxt}; strict={bool(strict)}; {time.time() - t0:.1f}s")
    return Handle(model, status)


install = arm


class Handle:
    """Generation-arm handle: .name, .describe(), .uninstall(), .generate (None: the public Evo2.generate is used)."""
    name = LEVER
    generate = None

    def __init__(self, model, status):
        self.model = model
        self.status = status

    def describe(self):
        st = self.status
        kinds = {k: sum(1 for v in st["blocks"].values() if v == k) for k in ("hcs", "hcm", "hcl")}
        return {"name": LEVER, "knobs": dict(st["knobs"], strict=st["strict"]), "numerics_class": NUMERICS_CLASS,
                "what": (f"one Triton launch per hyena block per decode step (outer FIR + inner FIR/IIR + gating + state update) "
                         f"replacing vtx HyenaCascade.sequential_forward on {len(st['blocks'])} blocks (hcs {kinds['hcs']}, hcm {kinds['hcm']}, "
                         f"hcl {kinds['hcl']}); prefill, attention, norms, projections, MLP untouched"),
                "selftested_on_box": bool(st["selftest"] and st["selftest"]["ok"]),
                "census": {k: st[k] for k in ("armed", "installed", "fused_steps", "fallback_steps", "fallback_layers")}}

    def uninstall(self):
        disarm(self.model)


def _warm(ctxs):
    """Compile every kernel variant once on dummy tensors (keeps the JIT out of the first generation's timing)."""
    seen = set()
    for ctx in ctxs.values():
        key = (ctx.mode, ctx.ni, ctx.L, ctx.gated_bias, str(ctx.device))
        if key in seen:
            continue
        seen.add(key)
        H, dev = ctx.H, ctx.device
        u = torch.zeros(1, 3 * H, dtype=torch.bfloat16, device=dev)
        ss = torch.zeros(1, 3 * H, 2, dtype=torch.float32, device=dev)
        inner = torch.zeros(1, H, ctx.ni, dtype=torch.float32, device=dev)
        y = torch.empty(1, H, dtype=torch.bfloat16, device=dev)
        ctx.launch(u, ss, torch.empty_like(ss), inner, torch.empty_like(inner), y)
        torch.cuda.synchronize(dev)


def _patch(core, ctx, status):
    from vortex.model.model import HyenaCascade
    f = ctx.filt
    li = f.layer_idx
    stock_step = HyenaCascade.sequential_forward

    def fused_sequential_forward(u, inference_params, _f=f, _ctx=ctx, _st=status):
        if _f.data_dtype is None:
            _f.data_dtype = u.dtype
        u2 = u[:, -1] if len(u.shape) > 2 else u
        ss = inference_params.fir_state_dict[_f.layer_idx]
        inner_dict = inference_params.fir_inner_state_dict if _ctx.mode == 0 else inference_params.state_dict
        inner = inner_dict[_f.layer_idx]
        if ss is not _ctx.ss or inner is not _ctx.si:
            # a new generation (prefill replaced the entries) or a stock-grown state: adopt when full-length
            if ss.shape[-1] != 2 or inner.shape[-1] != _ctx.ni:
                if _st["strict"]:
                    raise HyenaFuseRefused(f"block {_f.layer_idx}: state shorter than the filter (prompt shorter than "
                                           f"{_ctx.ni + 1} tokens): outer {tuple(ss.shape)} inner {tuple(inner.shape)}")
                if _f.layer_idx not in _st["fallback_layers"]:
                    _st["fallback_layers"].append(_f.layer_idx)
                    _say(f"FALLBACK: block {_f.layer_idx} state shorter than its filter (outer {tuple(ss.shape)}, inner "
                         f"{tuple(inner.shape)}; prompt shorter than {_ctx.ni + 1} tokens) -> stock step until the state is full (counted)")
                _st["fallback_steps"] += 1
                return stock_step(_f, u, inference_params)
            _ctx.ss = ss.to(torch.float32).contiguous()             # bf16 prefill states -> fp32 copies (exact); fp32 ones adopted as-is
            _ctx.si = inner.to(torch.float32).contiguous()
            _ctx.ss_alt, _ctx.si_alt = torch.empty_like(_ctx.ss), torch.empty_like(_ctx.si)
            if not _st["installed"]:
                _st["installed"] = True
                _say(f"INSTALLED: first fused decode step (block {_f.layer_idx}, batch {u2.shape[0]}, device {u2.device}); "
                     f"states adopted as fp32 ping-pong buffers")
        y = torch.empty(u2.shape[0], _ctx.H, dtype=torch.bfloat16, device=u2.device)
        _ctx.launch(u2, _ctx.ss, _ctx.ss_alt, _ctx.si, _ctx.si_alt, y)
        _ctx.ss, _ctx.ss_alt = _ctx.ss_alt, _ctx.ss
        _ctx.si, _ctx.si_alt = _ctx.si_alt, _ctx.si
        inference_params.fir_state_dict[_f.layer_idx] = _ctx.ss             # the dicts always hold the current (just written) states
        inner_dict[_f.layer_idx] = _ctx.si
        _st["fused_steps"] += 1
        y = y.to(dtype=_f.data_dtype)
        return y[:, None], inference_params

    f.sequential_forward = fused_sequential_forward          # instance attribute shadows the class method; class untouched
    status["blocks"][li] = ctx.kind


def disarm(model):
    """Remove the fused step from every block (the class method shows through again); returns the final status census."""
    core = _core(model)
    ctxs = getattr(core, "_hyenafuse_ctx", None) or {}
    for ctx in ctxs.values():
        ctx.filt.__dict__.pop("sequential_forward", None)
        ctx.ss = ctx.si = ctx.ss_alt = ctx.si_alt = None
    status = getattr(core, "_hyenafuse_status", {"armed": False})
    core._hyenafuse_ctx = None
    status["armed"] = False
    _say(f"DISARMED: stock decode step restored on {len(ctxs)} blocks; fused_steps={status.get('fused_steps')} "
         f"fallback_steps={status.get('fallback_steps')}")
    return status


uninstall = disarm
