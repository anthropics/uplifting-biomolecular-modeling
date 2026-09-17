# af3t_msa.py -- fused-kernel adapters for the MSA module of xfold (github.com/Shenggan/xfold @ 22bdeed, PyTorch AlphaFold3), INFERENCE ONLY.
#
#   import af3t_msa as M
#   M.enable(["opm"])      # composable with af3_kernels' levers (class-level patches of different classes / the Evoformer block)
#   ... run xfold as usual (torch.inference_mode + torch.autocast("cuda", torch.bfloat16)) ...
#   M.census()             # {lever: {"served:<key>": n, "fallback:<reason>": n, "kernel_error": n}, "dead": {...}}
#   M.disable()
#
# Lever "opm": xfold.nn.primitives.OuterProductMean on the kit's kernels (third_party/af3t_opm.py): LayerNorm + left|right projections + mask in ONE
# pass over the MSA stream, the outer product as one cuBLAS bf16 GEMM per chunk of OPM_ROWS left tokens straight into the layout the output
# contraction reads (no [N, N, 32*32] permute / copy; the intermediate lives one chunk at a time: OPM_ROWS*N*1024 bf16 elements instead of
# 2 x N*N*1024), the (c, e) -> c_z contraction + b_out + / (eps + norm) in one kernel, and — through xfold.nn.pairformer.EvoformerBlock — the block's
# `pair += outer_product_mean(msa, msa_mask)` folded into that kernel's epilogue (pair rows updated in place; bitwise `pair += update` on the kernel's
# own update).  The mask norm (mask^T mask, exact integer counts in bf16 — bitwise the statement's) is computed once per msa mask tensor, not per call.
# Numerics: the statement's rounding points (fp32 LayerNorm, bf16 GEMM operands / outputs, fp32 accumulation, fp32 epilogue); the fp32 summation order
# inside the LayerNorm statistics, the 64-long projections and the 1024-long output contraction differs -> tolerance-class (fast, big), NOT bitwise;
# deterministic run to run (fixed tiles, no atomics, no autotuning).
# Every route fails safe BY NAME: outside the served cell (cpu, no bf16 autocast, c_m / c / c_z outside the kernels' shapes, autograd) the stock
# forward runs and the reason is counted ("fallback:<reason>", printed once); a kernel error marks the lever dead for the process (stock from then
# on, "kernel_error", census "dead"); an out-of-memory error PROPAGATES (opt_core.oom.is_oom) — it is the caller's failure to report, never a fallback.
import sys, weakref
import torch

from xfold.nn import primitives as XP, pairformer as XPF
from opt_core.oom import is_oom

LEVERS = ("opm",)
_ON = set()
_ORIG = {}
COUNTS = {k: {} for k in LEVERS}
_DEAD = {}
_SEEN = set()
OPM_ROWS = 256                      # left tokens per outer-product chunk: the bf16 intermediate is OPM_ROWS * N * 32*32 elements (436 MiB at 832 tokens, 638 MiB at 1216)
OPM_CELLS = {                       # (BA rows per LayerNorm/projection program, BD right tokens per contraction program, warps) per compute-capability major; measured
    9: dict(BA=64, BD=64, warps=4),  # H100
    8: dict(BA=64, BD=64, warps=4),  # A100
}
_SERVED = dict(c_msa=64, c_outer=32, c_z=128)   # the kernels' shapes (tl.dot tiles: c_m 64 = K, 32 outer channels, 128 = the accumulator width)

try:                                # join the kit's kernel census when af3_kernels is importable (one census dict per process: forward.json's `census`,
    import af3_kernels as _K        # the LEVER lines' served= / fallback= evidence, the FALLBACK line); standalone otherwise
    for _k in LEVERS:
        _K.COUNTS.setdefault(_k, COUNTS[_k]); COUNTS[_k] = _K.COUNTS[_k]
except Exception:                   # pragma: no cover - standalone use
    _K = None


def _log(msg):
    print("[af3t_msa] " + msg, file=sys.stderr, flush=True)


def _count(lever, key):
    d = COUNTS[lever]; d[key] = d.get(key, 0) + 1


def _fallback(lever, reason):
    _count(lever, "fallback:" + reason)
    if (lever, reason) not in _SEEN:
        _SEEN.add((lever, reason)); _log("%s: stock path (%s)" % (lever, reason))


def _kernel_error(lever, e):
    _count(lever, "kernel_error")
    _DEAD[lever] = repr(e)[:300]
    if _K is not None:
        _K._DEAD[lever] = _DEAD[lever]
    _log("%s: KERNEL ERROR -> lever marked dead for this process, stock path from now on: %r" % (lever, e))


def _capturing():
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


def _autocast_bf16():
    return torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.bfloat16


def census():
    out = {k: dict(v) for k, v in COUNTS.items()}
    out["on"] = sorted(_ON); out["dead"] = dict(_DEAD)
    return out


def clear_caches():
    _NORM.clear()


# ----------------------------------------------------------------------------------------------------------------------------- opm
_NORM = {}          # id(mask) -> (weakref(mask), norm16 [N, N] bf16): the mask norm per msa-mask tensor (one per item in the kit's forward; weakref-keyed, nothing kept alive)


def _opm_weights(self):
    ck = getattr(self, "_af3m", None)
    if ck is None:
        with torch.no_grad():
            WT = torch.cat([self.left_projection.weight, self.right_projection.weight], 0).detach().to(torch.bfloat16).t().contiguous()   # [c_m, 2c] = (W_l | W_r)^T, bf16 (autocast's cast of the Linear weights)
            ck = dict(WT=WT, W16=self.output_w.detach().to(torch.bfloat16).reshape(-1, self.output_w.shape[-1]).contiguous(),             # [c*c, c_z] bf16 (autocast's cast of output_w at the einsum)
                      bias=self.output_b.detach().float().contiguous(), lnw=self.layer_norm_input.weight.detach().float().contiguous(),
                      lnb=self.layer_norm_input.bias.detach().float().contiguous(), eps=float(self.layer_norm_input.eps))
        self._af3m = ck
    return ck


def _mask_norm(mask):
    """norm[b, d] = sum_a mask[a, b] mask[a, d] as the statement computes it under autocast (bf16 operands, fp32 accumulate -> exact integer counts ->
    bf16): any bf16 GEMM gives the same bytes.  Memoised per mask tensor (weakref): the MSA module's blocks and the recycling passes share one mask."""
    ent = _NORM.get(id(mask))
    if ent is not None and ent[0]() is mask:
        _count("opm", "norm_memo_hit")
        return ent[1]
    m16 = mask.to(torch.bfloat16)
    norm16 = torch.mm(m16.t(), m16)
    if len(_NORM) > 8: _NORM.clear()
    try:
        _NORM[id(mask)] = (weakref.ref(mask), norm16)
    except TypeError:                                                           # not weak-referenceable: no memo
        pass
    return norm16


def _opm_cell_reason(self, msa, mask, residual):
    if "opm" not in _ON: return "off"
    if not msa.is_cuda: return "cpu"
    if not _autocast_bf16(): return "no-bf16-autocast"
    if torch.is_grad_enabled() and (msa.requires_grad or self.output_w.requires_grad): return "autograd"
    if msa.dim() != 3 or mask.dim() != 2 or tuple(mask.shape) != tuple(msa.shape[:2]): return "shape"
    if self.c_msa != _SERVED["c_msa"] or msa.shape[-1] != self.c_msa: return "c_msa=%d" % msa.shape[-1]
    if self.num_outer_channel != _SERVED["c_outer"] or self.num_output_channel != _SERVED["c_z"]: return "c=%d,c_z=%d" % (self.num_outer_channel, self.num_output_channel)
    if msa.dtype not in (torch.bfloat16, torch.float32): return "dtype=%s" % msa.dtype
    if residual is not None and (residual.dtype != torch.bfloat16 or tuple(residual.shape) != (msa.shape[1], msa.shape[1], self.num_output_channel) or not residual.is_contiguous()):
        return "residual=%s" % (residual.dtype if residual.dtype != torch.bfloat16 else "layout")
    cc = torch.cuda.get_device_capability(msa.device)[0]
    if cc not in OPM_CELLS: return "arch=sm_%d%d" % torch.cuda.get_device_capability(msa.device)
    return None


def opm_kernels(self, msa, mask, residual=None, rows=None):
    """The OuterProductMean statement on the kernels: returns the fp32 update [N, N, c_z] (residual None) or `residual += update` in place (returns residual)."""
    import af3t_opm as KO
    cell = OPM_CELLS[torch.cuda.get_device_capability(msa.device)[0]]
    W = _opm_weights(self)
    S, N, C = msa.shape; CO = self.num_outer_channel; F = self.num_output_channel
    msa = msa.contiguous(); mask_c = mask.contiguous()
    with torch.autocast("cuda", enabled=False):
        LT, R = KO.ln_proj2(msa, W["lnw"], W["lnb"], W["WT"], mask_c, W["eps"], BA=cell["BA"], num_warps=cell["warps"])
        norm16 = _mask_norm(mask_c)
        rows = int(rows or OPM_ROWS); rows = N if rows >= N else rows
        A = LT.view(N * CO, S); B = R.view(S, N * CO)
        T = torch.empty((rows * CO, N * CO), device=msa.device, dtype=torch.bfloat16)          # one chunk of the outer product, reused
        dst = residual if residual is not None else torch.empty((N, N, F), device=msa.device, dtype=torch.float32)
        for b0 in range(0, N, rows):
            nb = min(rows, N - b0)
            Tn = T[: nb * CO]
            torch.mm(A[b0 * CO:(b0 + nb) * CO], B, out=Tn)                                     # T[b, c, d, e] = sum_a l[a, b, c] r[a, d, e]  (bf16, fp32 accumulate)
            KO.opm_out(Tn, W["W16"], W["bias"], norm16, float(self.epsilon), b0, nb, N, dst, residual is not None, BD=cell["BD"], num_warps=cell["warps"])
    return dst


def _opm_forward(self, msa, mask, residual=None):
    """OuterProductMean.forward (+ the optional block residual): the kernels inside the served cell, the stock statement by name outside it."""
    lever = "opm"
    why = _opm_cell_reason(self, msa, mask, residual)
    if why is None and lever not in _DEAD:
        try:
            out = opm_kernels(self, msa, mask, residual)
            _count(lever, "served:%s" % ("block" if residual is not None else "module"))
            return out
        except Exception as e:
            if is_oom(e) or _capturing(): raise                                 # OOM propagates; under a CUDA-graph capture the error is the capture's
            _kernel_error(lever, e)
    elif why not in ("off",):
        _fallback(lever, why if lever not in _DEAD else "dead")
    upd = _ORIG["opm"](self, msa, mask)
    if residual is None:
        return upd
    residual += upd
    return residual


def _evoformer_forward(self, msa, pair, msa_mask, pair_mask):
    """xfold EvoformerBlock.forward with `pair += self.outer_product_mean(msa, msa_mask)` served by the opm kernels' residual epilogue (in place);
    every other statement verbatim (their modules carry af3_kernels' class-level levers as installed)."""
    pair = _opm_forward(self.outer_product_mean, msa, msa_mask, residual=pair)
    msa += self.msa_attention1(msa, msa_mask, pair)
    msa += self.msa_transition(msa)
    pair += self.triangle_multiplication_outgoing(pair, mask=pair_mask)
    pair += self.triangle_multiplication_incoming(pair, mask=pair_mask)
    pair += self.pair_attention1(pair, mask=pair_mask)
    pair += self.pair_attention2(pair, mask=pair_mask)
    pair += self.pair_transition(pair)
    return msa, pair


# ----------------------------------------------------------------------------------------------------------------------------- install
def _install():
    if not _ORIG:
        _ORIG["opm"] = XP.OuterProductMean.forward
        _ORIG["evoformer"] = XPF.EvoformerBlock.forward
    XP.OuterProductMean.forward = _opm_forward if "opm" in _ON else _ORIG["opm"]
    XPF.EvoformerBlock.forward = _evoformer_forward if "opm" in _ON else _ORIG["evoformer"]


def enable(levers=LEVERS):
    """Enable the given levers (list of names, '+'/',' string or 'all'); replaces the current set. Returns the active set."""
    if isinstance(levers, str):
        levers = LEVERS if levers == "all" else [p for p in levers.replace(",", "+").split("+") if p]
    bad = [l for l in levers if l not in LEVERS]
    if bad:
        raise ValueError("unknown levers %s; known: %s" % (bad, LEVERS))
    _ON.clear(); _ON.update(levers)
    _install()
    return sorted(_ON)


def disable():
    _ON.clear()
    if _ORIG:
        _install()
    clear_caches()
    return []


def active():
    return sorted(_ON)
