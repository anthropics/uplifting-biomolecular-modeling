"""ef2_t16_transition — the pair Transition FORWARD (LN -> W1|W2 -> SwiGLU -> W3 -> + x; d = 256, hidden 1024, bf16) as ONE persistent,
warp-specialised sm_90a CUDA C++ / CuTe kernel, loaded from the cubin this kit carries under k/ef2_t16/sm_90a/ through the CUDA driver API
(ef2_t16_nvjit, cuda-bindings; no compiler at run time). In this kit it is the FORWARD of K-D3 lean (ef2_autograd_kernels
``transition="refround_lean"``, the fast / big composition): the autograd node keeps K-D3's backward unchanged — that backward already recomputes
LN (+ its fp32 statistics) and the W12 GEMM from the block input, so an out-only forward kernel changes no saved tensor it reads.

PROVENANCE: carried from the esmfold2 kit's lever ``t16`` — opt/forward/fast_inference/driver/ef2_transition_cute.py at commit cab5aaaa56b6 (file
sha256 103d5ade8aae): VERSION, BH / NSLOT / SMEM_BYTES, kernel(), CANARY / canary_operands / canary, _check_ptr_arg, pack_transition, _tmx, grid_for,
transition_cute, reference_fp32, reference_recipe are that file's statements with their arithmetic unchanged (its loader module carried as
ef2_t16_nvjit; the kernel source ef2_t16/ef2_transition_cute.cuh, the cubin and its manifest byte-for-byte: ef2_t16/PROVENANCE.md). The part this
kit adds is the engagement surface K-D3 and the composition use: engage() / live() / forward_for() / pack_kd3() / stats() / describe().

Numerics class: TOLERANCE (fast) — fp32 LayerNorm statistics (two-pass), x_hat rounded once to bf16, bf16 operands with fp32 accumulation, h =
bf16(silu(a) * b), out = bf16(fp32(x) + acc); sigmoid = 1 / (1 + 2^(-a log2 e)) with ex2.approx + rcp.approx. Against K-D3's own forward (stock's
rounding points: bf16(silu) before the product, cuBLAS GEMMs) the output differs by at most one bf16 ulp of |out|; both sit at the same distance from
an fp32 evaluation. No TF32, no reduced-precision reductions.

Cards: compute capability 9.0 only (H100 / H200: wgmma / TMA / setmaxnreg). On any other device, on a stack without cuda-bindings, or when the
carried cubin does not pass its manifest checks or its install canary, engage() records the reason and the lever STEPS ASIDE BY NAME: K-D3's own
forward kernels serve (``LEVER name=ef2_t16_transition state=stepped_aside reason=…``) — never silent, never a refusal.

    engage()                      once per process, before any CUDA-graph capture: load the cubin (manifest source key / sha256 / register and
                                  local-memory record checked), run the install canary; -> describe(). Never raises for a device / stack that
                                  cannot serve: the reason is recorded and live() is False.
    live()                        the kernel is loaded and passed its canary in this process.
    pack_kd3(w, eps)              the TMA descriptors over K-D3's cached bf16 weights ``w`` (ef2_autograd_kernels._transition_weights3: W12 [2048, 256],
                                  W3 [256, 1024], LN affine fp32) — no second weight copy; None when the module has not the served shape.
    forward_for(w, eps)           the callable K-D3 lean's forward runs for that weight entry (x2d [M, 256] bf16 -> x + T(x)), or None (counted by
                                  name in stats()) when the lever is not live or the module is not servable.
    stats() / describe()          served / fallthrough counters and the kernel's words (cubin, regs, smem, canary) for the LEVER line.
Grad-mode design use only through K-D3 (ef2_autograd_kernels decides eligibility: CUDA, bf16, grad enabled); the kernel itself is inference
arithmetic (no autograd of its own)."""
import collections
import os

import torch

import ef2_t16_nvjit as ef2_nvjit

NAME = "ef2_t16_transition"
VERSION = "transition_cute.1.0"          # the carried kernel's version word (esmfold2 ef2_transition_cute)
STATS = collections.Counter()
_BF16 = torch.bfloat16
BH, NSLOT = 64, 5                     # kernel constants (hidden units per chunk, weight-ring slots): the shared-memory plan below must match the .cuh
SMEM_BYTES = 128 * 256 * 2 + NSLOT * (BH * 256 * 2) + (4 + 2 * NSLOT) * 8 + 1024   # x tile + ring + mbarriers + 1 KB alignment slack = 230512 B
D_PAIR, HIDDEN = 256, 1024            # the one module shape the kernel serves (ESMFold2's pair Transition: c = 256, expansion 4)
SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ef2_t16")   # the carried kernel source + cubin + manifest (ef2_t16_nvjit.SHIPPED_ROOT)
_STATE = dict(kernel=None, src=None, nsm=None, loadcheck=None, engaged=None, reason=None, reason_text=None, device=None)


# --------------------------------------------------------------------------------------------------------------------------------------------
# kernel load + launch (carried)
# --------------------------------------------------------------------------------------------------------------------------------------------
def _source():
    if _STATE["src"] is None:
        p = os.path.join(SOURCE_DIR, "ef2_transition_cute.cuh")
        with open(p) as f:
            _STATE["src"] = f.read()
    return _STATE["src"]


def nvrtc_sources():
    """The kernel build spec (the manifest's source key is computed from it: source text + compile options; the tag and file stem are the
    esmfold2 kit's so the carried manifest entry matches byte-for-byte)."""
    return [dict(tag=f"ef2_transition_cute {VERSION}", src=_source(), name="ef2_transition_cute.cu", macros=None, opts=(), spill_free=True)]   # a|b chains overlap silu(a): must be spill-free


def kernel():
    """Load the cubin carried under k/ef2_t16/sm_90a through ef2_t16_nvjit (sha256 / source key / device local bytes / register count checked
    against the manifest; no JIT cache, no NVRTC at run time) and run the install canary -> ef2_t16_nvjit.Kernel.  Raises ef2_t16_nvjit.NvjitUnavailable
    by name on any failure (no device of compute capability 9.0, too little opt-in shared memory, a missing or mismatching cubin, a failed canary)."""
    k = _STATE["kernel"]
    if k is None:
        dev = ef2_nvjit.require_device(SMEM_BYTES, who=NAME)
        spec = nvrtc_sources()[0]
        cub = ef2_nvjit.shipped_cubin(spec["src"], spec["name"], macros=spec["macros"], opts=spec["opts"], tag=spec["tag"])   # the shipped binary or NvjitUnavailable by name (no run-time compile)
        k = ef2_nvjit.Kernel(cub, "ef2_transition", smem_bytes=SMEM_BYTES)
        _STATE["kernel"], _STATE["nsm"] = k, dev.multi_processor_count
        STATS["compiles"] += int(cub.source in ("compiled", "memory"))
        STATS["builds"] += 1
        _STATE["loadcheck"] = canary(k, expect=(cub.manifest or {}).get("canary_digest"))   # one small launch vs the fp32 reference: NvjitUnavailable by name off tolerance
    return k


CANARY = dict(rows=(147461, 262147), launches=50, rounds_of=10, seed=20260911, tol=0.0625)   # ragged row counts just above 384^2 and 512^2 pair rows (>= 8 row tiles per
                                                                                                      # CTA: the persistent loop and both accumulator chains many times over); < 1 s on an H100


def canary_operands(M, seed, dev):
    """Fixed synthetic operands for one transition call over M rows (d=256, hidden 1024), from a PRIVATE CPU generator (the process RNG is untouched)."""
    import types
    g = torch.Generator(device="cpu"); g.manual_seed(seed + M)
    H = 1024
    x = torch.randn(M, 256, generator=g).to(dev, _BF16)
    norm = types.SimpleNamespace(weight=(1.0 + 0.1 * torch.randn(256, generator=g)).to(dev), bias=(0.05 * torch.randn(256, generator=g)).to(dev), eps=1e-5)
    ffn = types.SimpleNamespace(w12=types.SimpleNamespace(weight=(torch.randn(2 * H, 256, generator=g) / 16.0).to(dev)),
                                w3=types.SimpleNamespace(weight=(torch.randn(256, H, generator=g) / 32.0).to(dev)), hidden_features=H)
    return x, norm, ffn


def canary(k=None, expect=None):
    """The install canary (before the first fold): at each row count of CANARY["rows"], CANARY["launches"] back-to-back launches of the loaded
    kernel on fixed seeded operands in rounds of CANARY["rounds_of"] distinct output buffers (no host synchronisation inside a round); EVERY
    launch must equal launch 0 bytewise AND launch 0 must match the kernel's rounding-point statement (reference_recipe, fp32 torch) within
    CANARY["tol"] — else ef2_t16_nvjit.NvjitUnavailable BY NAME (the lever cannot serve). Returns dict(word="pass:<n>", passes,
    max_abs_recipe, max_abs_fp32, digest = sha256 over the launch-0 outputs, bitwise = digest == ``expect`` (the manifest's record) | None, rows)."""
    import hashlib
    k = k or kernel()
    dev = torch.device("cuda", torch.cuda.current_device())
    nsm = _STATE["nsm"] or torch.cuda.get_device_properties(dev).multi_processor_count
    h = hashlib.sha256(); passes = 0; worst_r = 0.0; worst_f = 0.0
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
        for M in CANARY["rows"]:
            x, norm, ffn = canary_operands(M, CANARY["seed"], dev)
            pack = pack_transition(norm, ffn)
            grid_x, n_tiles = grid_for(M, nsm)
            tmx = ef2_nvjit.TensorMap(x, dims=[256, M], strides_bytes=[512], box=[64, 64], swizzle=128)

            def launch(out):
                k((grid_x, 1, 1), (384, 1, 1), tmx, pack["tmW12"], pack["tmW3"], x, out, pack["ln_w"], pack["ln_b"], int(M), int(n_tiles), float(pack["eps"]), 1)
            gold = torch.empty_like(x); launch(gold); torch.cuda.synchronize(dev)
            d_recipe = 0.0; d_fp32 = 0.0
            for i in range(0, M, 65536):                                       # the fp32 statements in row chunks (their [rows, 2048] intermediates stay small)
                sl = slice(i, min(M, i + 65536))
                d_recipe = max(d_recipe, (gold[sl].float() - reference_recipe(x[sl], norm, ffn).float()).abs().max().item())
                d_fp32 = max(d_fp32, (gold[sl].float() - reference_fp32(x[sl], norm, ffn, master=False)).abs().max().item())
            if not (d_recipe <= CANARY["tol"]):                                # NaN-safe: a NaN fails
                raise ef2_nvjit.NvjitUnavailable(f"{NAME}: canary FAILED on {torch.cuda.get_device_name(dev)} at {M} rows: max|kernel - rounding-point statement| = "
                                                 f"{d_recipe:.4g} > {CANARY['tol']} (cubin {k.cubin_sha256[:12]}); the lever cannot serve here — stepping aside by name")
            worst_r = max(worst_r, d_recipe); worst_f = max(worst_f, d_fp32)
            R = CANARY["rounds_of"]; bufs = [torch.empty_like(x) for _ in range(R)]; flags = torch.zeros(R, dtype=torch.int64, device=dev); done = 0
            while done < CANARY["launches"]:
                for b in bufs:                                                 # R launches back to back on the current stream: no host sync, nothing else in between
                    launch(b)
                for i, b in enumerate(bufs):
                    flags[i] += (b != gold).any()
                done += R
            torch.cuda.synchronize(dev)
            nbad = int(flags.sum().item())
            if nbad:
                raise ef2_nvjit.NvjitUnavailable(f"{NAME}: canary FAILED on {torch.cuda.get_device_name(dev)} at {M} rows: {nbad} of {done} back-to-back launches differ "
                                                 f"bytewise from launch 0 (cubin {k.cubin_sha256[:12]}); a nondeterministic kernel must not serve — stepping aside by name")
            passes += done + 1
            h.update(gold.cpu().contiguous().view(torch.uint8).numpy().tobytes())
            del bufs, gold, pack, x
    digest = h.hexdigest()
    return dict(word=f"pass:{passes}", passes=passes, max_abs_recipe=worst_r, max_abs_fp32=worst_f, digest=digest, bitwise=(None if not expect else digest == expect),
                rows="/".join(str(m) for m in CANARY["rows"]))


def _check_ptr_arg(name, t, dtype, shape=None):
    """every tensor whose data_ptr reaches the kernel: dtype / device / contiguity asserted by name (a bf16 tensor read as float* is silent garbage)."""
    if not isinstance(t, torch.Tensor) or t.dtype != dtype or not t.is_cuda or not t.is_contiguous() or (shape is not None and tuple(t.shape) != tuple(shape)):
        raise RuntimeError(f"{NAME}: kernel argument {name}: expected contiguous CUDA {dtype} {tuple(shape) if shape is not None else ''}, "
                           f"got {type(t).__name__} {getattr(t, 'dtype', None)} {tuple(getattr(t, 'shape', ()))} cuda={getattr(t, 'is_cuda', None)}")


def pack_transition(norm, ffn):
    """bf16 weight copies in the stock layouts (= the values the stock fused / autocast paths consume) + their TMA descriptors; LN affine fp32.
    Built with autocast disabled and every pointer argument's dtype asserted (the kernel reads ln_w / ln_b as float*, W12 / W3 / x as bf16)."""
    H = ffn.hidden_features
    if H != 1024:
        raise RuntimeError(f"{NAME}: hidden_features {H} (the kernel serves 1024)")
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
        W12 = ffn.w12.weight.detach().to(dtype=_BF16, copy=True).contiguous()      # [2H, 256]: rows [0,H) -> x1 (silu branch), [H,2H) -> x2
        W3 = ffn.w3.weight.detach().to(dtype=_BF16, copy=True).contiguous()        # [256, H]
        ln_w = norm.weight.detach().to(dtype=torch.float32, copy=True).contiguous()
        ln_b = (norm.bias.detach().to(dtype=torch.float32, copy=True) if norm.bias is not None else torch.zeros_like(ln_w)).contiguous()
    return _pack_tensors(W12, W3, ln_w, ln_b, float(norm.eps))


def _pack_tensors(W12, W3, ln_w, ln_b, eps):
    """The kernel's weight operands (contiguous CUDA bf16 W12 [2048, 256] / W3 [256, 1024], fp32 LN affine [256]) + their TMA descriptors."""
    H = HIDDEN
    _check_ptr_arg("W12", W12, _BF16, (2 * H, 256)); _check_ptr_arg("W3", W3, _BF16, (256, H))
    _check_ptr_arg("ln_w", ln_w, torch.float32, (256,)); _check_ptr_arg("ln_b", ln_b, torch.float32, (256,))
    tmW12 = ef2_nvjit.TensorMap(W12, dims=[256, 2 * H], strides_bytes=[512], box=[64, BH], swizzle=128)      # W1_j / W2_j: 4 boxes {64 k, 64 rows}
    tmW3 = ef2_nvjit.TensorMap(W3, dims=[H, 256], strides_bytes=[2 * H], box=[BH, 256], swizzle=128)         # W3_j: one box {64 k, 256 rows}
    return dict(W12=W12, W3=W3, ln_w=ln_w, ln_b=ln_b, tmW12=tmW12, tmW3=tmW3, H=H, eps=float(eps))


_TMX_CACHE = collections.OrderedDict()


def _tmx(x2d):
    """TMA descriptor over an activation buffer, cached by (address, rows).  The cache holds descriptors only — never a reference to the tensor
    (a descriptor is an address + shape; the same key describes any later tensor at that address, and pinning activations would leak GBs)."""
    key = (x2d.data_ptr(), x2d.shape[0])
    tm = _TMX_CACHE.get(key)
    if tm is None:
        tm = ef2_nvjit.TensorMap(x2d, dims=[256, x2d.shape[0]], strides_bytes=[512], box=[64, 64], swizzle=128, keep=False)
        _TMX_CACHE[key] = tm
        while len(_TMX_CACHE) > 256:
            _TMX_CACHE.popitem(last=False)
    else:
        _TMX_CACHE.move_to_end(key)
    return tm


def grid_for(M, nsm):
    n_tiles = (M + 127) // 128
    return min(n_tiles, max(int(nsm), 1)), n_tiles


def transition_cute(x2d, pack, out=None, residual=True):
    """x2d [M, 256] bf16 contiguous (M >= 1) -> out [M, 256] bf16 = x + Transition(x) (residual=True) or Transition(x) (residual=False)."""
    M, K = x2d.shape
    _check_ptr_arg("x", x2d, _BF16, (M, 256))
    k = kernel()
    if out is None:
        out = torch.empty_like(x2d)
    _check_ptr_arg("out", out, _BF16, (M, 256))
    if M < 1 or out.data_ptr() == x2d.data_ptr():
        raise RuntimeError(f"{NAME}: M={M}, in-place={out.data_ptr() == x2d.data_ptr()} (the kernel serves M >= 1 out of place)")
    _check_ptr_arg("ln_w", pack["ln_w"], torch.float32, (256,)); _check_ptr_arg("ln_b", pack["ln_b"], torch.float32, (256,))
    grid_x, n_tiles = grid_for(M, _STATE["nsm"])
    k((grid_x, 1, 1), (384, 1, 1), _tmx(x2d), pack["tmW12"], pack["tmW3"], x2d, out, pack["ln_w"], pack["ln_b"], int(M), int(n_tiles), float(pack["eps"]), (1 if residual else 0))
    STATS["calls"] += 1
    return out


# --------------------------------------------------------------------------------------------------------------------------------------------
# references (the canary and the unit tests) — carried
# --------------------------------------------------------------------------------------------------------------------------------------------
def reference_fp32(x, norm, ffn, master=True):
    """x + w3(silu(w12a(LN(x))) * w12b(LN(x))) in fp32 from the bf16 input; master=True: fp32 master weights, False: bf16-rounded weights."""
    xf = x.float()
    w = (lambda t: t.detach().float()) if master else (lambda t: t.detach().to(_BF16).float())
    xn = torch.nn.functional.layer_norm(xf, (xf.shape[-1],), w(norm.weight), w(norm.bias) if norm.bias is not None else None, norm.eps)
    x12 = xn @ w(ffn.w12.weight).t()
    x1, x2 = x12.split(ffn.hidden_features, dim=-1)
    return xf + (torch.nn.functional.silu(x1) * x2) @ w(ffn.w3.weight).t()


def reference_recipe(x, norm, ffn):
    """torch emulation of the kernel's rounding points (T15 class): x_hat bf16, fp32 GEMMs on bf16 operands, h bf16, out bf16."""
    xf = x.float()
    lw = norm.weight.detach().float(); lb = norm.bias.detach().float() if norm.bias is not None else torch.zeros_like(lw)
    mu = xf.mean(-1, keepdim=True); var = ((xf - mu) ** 2).mean(-1, keepdim=True)
    xh = ((xf - mu) * torch.rsqrt(var + norm.eps) * lw + lb).to(_BF16).float()
    x12 = xh @ ffn.w12.weight.detach().to(_BF16).float().t()
    a, b = x12.split(ffn.hidden_features, dim=-1)
    h = (torch.nn.functional.silu(a) * b).to(_BF16).float()
    return (xf + h @ ffn.w3.weight.detach().to(_BF16).float().t()).to(_BF16)


# --------------------------------------------------------------------------------------------------------------------------------------------
# the engagement surface this kit adds: K-D3 lean's forward (ef2_autograd_kernels) and the composition (fastkit) use it
# --------------------------------------------------------------------------------------------------------------------------------------------
def _reason_word(e) -> str:
    """A blank-free word for the LEVER line from an NvjitUnavailable message (the full text stays in describe()["reason_text"])."""
    t = str(e)
    if " is sm_" in t and "only" in t:
        import re
        m = re.search(r" is (sm_\d+);", t)
        return f"not_sm_90:{m.group(1) if m else 'other'}"
    for needle, word in (("no CUDA device", "no_cuda"), ("dynamic shared memory", "smem"), ("cuda.bindings", "no_cuda_bindings"), ("no shipped cubin", "cubin_absent"),
                         ("file is absent", "cubin_absent"), ("source key", "cubin_source_key"), ("sha256", "cubin_sha256"), ("acceptable build", "cubin_manifest"),
                         ("reports regs=", "cubin_device_record"), ("reports spill_bytes=", "cubin_device_record"), ("local memory", "cubin_local_memory"),
                         ("canary FAILED", "canary_failed"), ("CUDA driver error", "driver_error")):
        if needle in t:
            return word
    return "load_failed"


def engage() -> dict:
    """Once per process, BEFORE any CUDA-graph capture: load the carried cubin and run its canary. A device or stack that cannot serve (not compute
    capability 9.0, no cuda-bindings, a cubin that fails its manifest checks or its canary) is RECORDED, not raised: live() is then False, K-D3's own
    forward serves and describe() names the reason (the lever steps aside by name). Returns describe()."""
    if _STATE["engaged"] is None:
        try:
            if torch.cuda.is_available():
                _STATE["device"] = torch.cuda.get_device_name(torch.cuda.current_device()).replace(" ", "_")
            kernel()
            _STATE["engaged"], _STATE["reason"], _STATE["reason_text"] = True, None, None
        except ef2_nvjit.NvjitUnavailable as e:
            _STATE["engaged"], _STATE["reason"], _STATE["reason_text"] = False, _reason_word(e), str(e)
            _STATE["kernel"] = None
    return describe()


def live() -> bool:
    """The kernel is loaded and passed its canary in this process (engage() said so)."""
    return _STATE["engaged"] is True and _STATE["kernel"] is not None


def servable_weights(w) -> bool:
    """K-D3's weight entry has the one shape the kernel serves: W12 [2048, 256] / W3 [256, 1024] bf16 CUDA contiguous, LN affine fp32 [256]."""
    try:
        W12, W3, lw, lb = w["W12"], w["W3"], w["LN_W32"], w["LN_B32"]
    except (KeyError, TypeError):
        return False
    return (tuple(W12.shape) == (2 * HIDDEN, D_PAIR) and tuple(W3.shape) == (D_PAIR, HIDDEN) and tuple(lw.shape) == (D_PAIR,) and tuple(lb.shape) == (D_PAIR,)
            and W12.dtype == _BF16 and W3.dtype == _BF16 and lw.dtype == torch.float32 and lb.dtype == torch.float32
            and all(t.is_cuda and t.is_contiguous() for t in (W12, W3, lw, lb)))


def pack_kd3(w, eps):
    """The kernel's operand pack over K-D3's cached weights ``w`` (ef2_autograd_kernels._transition_weights3: the bf16 W12 / W3 copies its backward
    reads, the fp32 LN affine) — descriptors only, no second copy; cached in ``w`` under the entry's own version key. None when not servable."""
    return _entry(w, eps)["pack"]


def _entry(w, eps):
    """w["t16_pack"] = dict(ver, pack, fwd): the operand pack and the forward closure over it, re-made when K-D3 re-keys its weights (ver)."""
    ent = w.get("t16_pack")
    if ent is not None and ent.get("ver") == w.get("ver"):
        return ent
    pack = _pack_tensors(w["W12"], w["W3"], w["LN_W32"], w["LN_B32"], float(eps)) if servable_weights(w) else None
    fwd = None
    if pack is not None:
        def fwd(x2d, _pack=pack):                            # one driver-API launch (no torch op but the output allocation: autocast / grad mode do not reach it)
            return transition_cute(x2d, _pack, residual=True)
    ent = dict(ver=w.get("ver"), pack=pack, fwd=fwd)
    w["t16_pack"] = ent
    STATS["packs"] += int(pack is not None)
    return ent


def forward_for(w, eps):
    """-> ``fwd(x2d) -> x + Transition(x)`` ([M, 256] bf16 contiguous in, a new [M, 256] bf16 out) over K-D3's weight entry ``w`` when the lever is
    live and the weights have the served shape; else None, counted by name (``not_live`` / ``shape``) — K-D3's own forward kernels then serve."""
    if not live():
        STATS["fallthrough_not_live"] += 1
        return None
    fwd = _entry(w, eps)["fwd"]
    if fwd is None:
        STATS["fallthrough_shape"] += 1
    return fwd


def kernel_evidence():
    """Blank-free k=v words of the loaded kernel for the LEVER line ({} before kernel()): cubin words + canary=pass:<n>:<bitwise|unrecorded|differs>:<digest12>:max_abs=…"""
    k = _STATE.get("kernel")
    ev = k.evidence() if k is not None else {}
    lc = _STATE.get("loadcheck")
    if lc:
        ev["canary"] = f"{lc['word']}:{'bitwise' if lc.get('bitwise') else ('unrecorded' if lc.get('bitwise') is None else 'differs')}:{lc['digest'][:12]}:max_abs={lc['max_abs_recipe']:.3g}"
    return ev


def stats() -> dict:
    """served = kernel launches from K-D3's forward (+ the canary's are not counted); fallback = the named fall-through counters (non-zero only)."""
    fb = {k[len("fallthrough_"):]: int(v) for k, v in STATS.items() if k.startswith("fallthrough_") and v}
    return dict(served=int(STATS.get("calls", 0)), fallback=fb, packs=int(STATS.get("packs", 0)))


def describe() -> dict:
    """The lever's record: state (``on`` | ``stepped_aside`` | ``off`` = never asked), reason (a word) + reason_text, version, device, the kernel's
    words (cubin / regs / spill_bytes / smem / arch / canary) and the counters."""
    state = "on" if live() else ("stepped_aside" if _STATE["engaged"] is False else "off")
    return dict(name=NAME, state=state, reason=_STATE["reason"], reason_text=_STATE["reason_text"], version=VERSION, device=_STATE["device"],
                kernel=kernel_evidence(), **stats())


def reset() -> None:
    """Forget the process state (tests): the next engage() loads and checks again."""
    _STATE.update(kernel=None, nsm=None, loadcheck=None, engaged=None, reason=None, reason_text=None, device=None)
    _TMX_CACHE.clear(); STATS.clear()
