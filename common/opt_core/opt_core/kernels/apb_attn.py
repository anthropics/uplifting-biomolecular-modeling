"""APB v1.2 — attention with a pair bias shared by a batch of samples (AlphaFold3 Algorithm 24 core; the token diffusion transformer's and
the pairformer single track's AttentionPairBias), one Triton launch for all samples:

    out[s, i, h*D:(h+1)*D] = sigmoid(gate[s, i, h*D:(h+1)*D]) * softmax_j( scale * q[s,h,i] . k[s,h,j] + bias[s|0, h, i, j]
                                                                          + (mask_neg if mask[s|0, j] == 0) ) @ v[s, h, j]

Flash arithmetic: the [S, H, N, N] logits never reach memory; online softmax with fp32 statistics and an fp32 accumulator; q.k and p.v on
the tensor cores in the operands' 16-bit dtype; the approximate exp2 of the GPU's special-function unit (ex2.approx.f32, ~2 ulp)
with log2(e) folded into the scale; the sigmoid gate evaluated in fp32 from the gate
logits and fused into the epilogue; one rounding to the output dtype at the store. Every operand is served IN PLACE whatever its strides
(unit stride on the innermost dim is the only requirement): q/k/v as the transposed views of the projection GEMM outputs, the bias as the
head-major planes a producer or cache holds, the gate and the output as column slices of wider row blocks.

Layout notes (H100): one program = one (query tile, head, sample); the sample index is the fastest grid axis so the S programs that read the
same bias tile are co-resident and share it through L2 (holding the tile in registers across samples loses to register pressure). Strides
are handed to the kernel pre-divided by the largest of 8/4/2/1 that divides all of them (`SDIV` for q/k/v/gate/out, `BDIV` for the bias) and
multiplied back by that constexpr inside: Triton specialises integer arguments on divisibility by 16 only, so a head stride of 24 elements
(D = 24 heads in 48-byte slots) or bias rows of a token count that is not a multiple of 8 would otherwise compile to unvectorised 2-byte loads.
A bias whose rows are 16-byte aligned (row stride a multiple of 8 elements for a 16-bit bias: allocate [.., N, ceil8(N)] and view [..., :N])
is the fast path at every token count. Head dims that are not a power of two are zero-padded in registers (D = 24 -> 32, 48 -> 64).

    apb_attention(q, k, v, bias, mask=None, gate=None, out=None, scale=None, mask_neg=-1e9, config=None, info=None) -> out
      q, k, v  [S, H, N, D] / [S, H, NK, D]   bf16 | fp16, any strides with unit stride on D
      bias     [H, N, NK] or [SB, H, N, NK] with SB in {1, S}   bf16 | fp16 | fp32, any strides (unit key stride = fast path)
      mask     [1, NK] or [S, NK] keep-mask (nonzero = attend) or None; all-masked rows give the stock uniform average, never NaN
      gate     [S, N, H*D] pre-sigmoid logits (any strides, unit column stride) or None
      out      [S, N, H*D] in q's dtype (allocated contiguous unless given; a given `out` may be a strided view with unit column stride)
      scale    multiplies q.k (None = D ** -0.5; pass 1.0 when q arrives pre-scaled)
    Served head dims: 1..64 (zero-padded in registers to 16 / 32 / 64). Raises Unsupported(event, detail) BEFORE any work for inputs outside
    the served domain (`check(...)` is the same validation without the launch, for callers that prepare operands first); `info`, when a
    dict, receives the launch facts (SDIV, BDIV, tile configuration, bias_rows_aligned).
    reference(q, k, v, bias, mask, gate, scale, mask_neg, dtype=float64) is the materialised statement the kernel is held to.
Launch: the validation verdict and the launch plan (tile config, divisors, integer and constexpr arguments) are memoised on the operands'
metadata — shapes, strides, dtypes, devices, 16-byte alignment: a superset of what Triton specializes on —, so a repeated call shape costs
one key lookup; the first launch of a plan goes through Triton's JITFunction (compile or disk cache) and keeps the compiled handle, later
launches drive that handle directly (the same binary with the same arguments: outputs are bitwise the JIT path's). A Triton whose
compiled-kernel launcher takes neither argument convention disables the direct path by name (`launch_census()['direct'] =
'off:<reason>'`) and every call takes the JIT path.
"""
import math
from typing import Dict, Optional

import torch
import triton
import triton.language as tl

__version__ = "1.2"
_LOG2E = 1.4426950408889634
MAX_HEAD_DIM = 64                                              # the served (and tested) head dims: 1..64


class Unsupported(Exception):
    """An input outside the kernel's served domain, refused by name before any work: `.event` is one census word
    (rank / shape:<what> / dtype:<what> / stride:<what> / size:<what> / device:<what> / head_dim:<n>), `.detail` a sentence."""

    def __init__(self, event: str, detail: str = ""):
        super().__init__(f"{event}: {detail}" if detail else event)
        self.event, self.detail = event, detail


@triton.jit
def _online_softmax(s, m_i, l_i):
    """one key tile of the online softmax on the log2-scaled logits s: the tile's probabilities p (relative to the new running max), the new
    running max and sum, and the rescale factor alpha for the accumulator."""
    m_new = tl.maximum(m_i, tl.max(s, 1))
    alpha = tl.math.exp2(m_i - m_new)
    p = tl.math.exp2(s - m_new[:, None])
    l_i = l_i * alpha + tl.sum(p, 1)
    return p, m_new, l_i, alpha


@triton.jit
def _apb_fwd(
    Q, K, V, Bias, Mask, Gate, Out,
    sqs, sqh, sqn,
    sks, skh, skn,
    svs, svh, svn,
    sbs, sbh, sbq, sbk,
    sms, smn,
    sgs, sgn, sgc,
    sos, son, soc,
    S_ROWS, NQ, NK,
    qk_scale2, mneg2,
    HEAD_DIM: tl.constexpr, BD: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HAS_MASK: tl.constexpr, MASK_PER_ROW: tl.constexpr, BIAS_PER_ROW: tl.constexpr, HAS_GATE: tl.constexpr,
    SDIV: tl.constexpr, BDIV: tl.constexpr, LOG2E: tl.constexpr,
):
    # strides arrive pre-divided by SDIV (q/k/v/gate/out) and BDIV (bias); multiplying back by the constexpr lets the compiler prove the
    # alignment of every row segment (see the module docstring)
    sqs = sqs * SDIV
    sqh = sqh * SDIV
    sqn = sqn * SDIV
    sks = sks * SDIV
    skh = skh * SDIV
    skn = skn * SDIV
    svs = svs * SDIV
    svh = svh * SDIV
    svn = svn * SDIV
    sgs = sgs * SDIV
    sgn = sgn * SDIV
    sos = sos * SDIV
    son = son * SDIV
    sbs = sbs * BDIV
    sbh = sbh * BDIV
    sbq = sbq * BDIV
    pid0 = tl.program_id(0)                                    # sample fastest: the programs sharing a bias tile are adjacent
    r = (pid0 % S_ROWS).to(tl.int64)
    pid_q = pid0 // S_ROWS
    h = tl.program_id(1).to(tl.int64)
    offs_m = pid_q * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_mc = tl.minimum(offs_m, NQ - 1)                       # clamped query rows: loads need no row mask, rows >= NQ are never stored (one (sample, head)
    offs_d = tl.arange(0, BD)                                  # slice is int32-addressable — the launcher checks it; the bases below are int64)
    offs_n = tl.arange(0, BLOCK_N)
    d_ok = offs_d < HEAD_DIM
    q_off = offs_mc[:, None] * sqn + offs_d[None, :]
    k_off = offs_n[:, None] * skn + offs_d[None, :]
    v_off = offs_n[:, None] * svn + offs_d[None, :]
    b_off = offs_mc[:, None] * sbq + offs_n[None, :] * sbk
    bias_h = Bias + h * sbh
    if BIAS_PER_ROW:
        bias_h = bias_h + r * sbs
    q_base = Q + r * sqs + h * sqh
    k_base = K + r * sks + h * skh
    v_base = V + r * svs + h * svh
    if BD == HEAD_DIM:
        q = tl.load(q_base + q_off)
    else:
        q = tl.load(q_base + q_off, mask=d_ok[None, :], other=0.0)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BD], dtype=tl.float32)
    if HAS_MASK:
        if MASK_PER_ROW:
            mk_base = Mask + r * sms
        else:
            mk_base = Mask
    hi = (NK // BLOCK_N) * BLOCK_N                             # the full key tiles; then one ragged tail tile when NK % BLOCK_N
    for lo in range(0, hi, BLOCK_N):
        bt = tl.load(bias_h + b_off + lo * sbk).to(tl.float32) * LOG2E
        if HAS_MASK:
            mk = tl.load(mk_base + (lo + offs_n) * smn)
            bt = bt + tl.where(mk != 0, 0.0, mneg2)[None, :]
        if BD == HEAD_DIM:
            kt = tl.load(k_base + k_off + lo * skn)
        else:
            kt = tl.load(k_base + k_off + lo * skn, mask=d_ok[None, :], other=0.0)
        p, m_i, l_i, alpha = _online_softmax(tl.dot(q, tl.trans(kt)) * qk_scale2 + bt, m_i, l_i)
        if BD == HEAD_DIM:
            vt = tl.load(v_base + v_off + lo * svn)
        else:
            vt = tl.load(v_base + v_off + lo * svn, mask=d_ok[None, :], other=0.0)
        acc = tl.dot(p.to(vt.dtype), vt, acc * alpha[:, None])
    if hi < NK:
        lo = hi
        nv = (lo + offs_n) < NK
        bt = tl.load(bias_h + b_off + lo * sbk, mask=nv[None, :], other=0.0).to(tl.float32) * LOG2E
        if HAS_MASK:
            mk = tl.load(mk_base + (lo + offs_n) * smn, mask=nv, other=1)
            bt = bt + tl.where(mk != 0, 0.0, mneg2)[None, :]
        bt = tl.where(nv[None, :], bt, -float('inf'))
        kt = tl.load(k_base + k_off + lo * skn, mask=nv[:, None] & d_ok[None, :], other=0.0)
        p, m_i, l_i, alpha = _online_softmax(tl.dot(q, tl.trans(kt)) * qk_scale2 + bt, m_i, l_i)
        vt = tl.load(v_base + v_off + lo * svn, mask=nv[:, None] & d_ok[None, :], other=0.0)
        acc = tl.dot(p.to(vt.dtype), vt, acc * alpha[:, None])
    o_cols = h * HEAD_DIM + offs_d
    o = acc / l_i[:, None]
    if HAS_GATE:
        g = tl.load(Gate + r * sgs + offs_mc[:, None] * sgn + o_cols[None, :] * sgc, mask=d_ok[None, :], other=0.0).to(tl.float32)
        o = o / (1.0 + tl.math.exp2(-g * LOG2E))
    tl.store(Out + r * sos + offs_m[:, None] * son + o_cols[None, :] * soc, o.to(Out.dtype.element_ty),
             mask=(offs_m < NQ)[:, None] & d_ok[None, :])


def _pow2(x: int, lo: int = 16) -> int:
    p = lo
    while p < x:
        p *= 2
    return p


def _divisor(strides) -> int:
    """Largest of 8 / 4 / 2 / 1 dividing every stride (elements)."""
    for dv in (8, 4, 2):
        if all(int(x) % dv == 0 for x in strides):
            return dv
    return 1


def default_config(S: int, N: int, D: int, H: int = 16) -> Dict[str, int]:
    """Launch configuration as a pure function of the problem (measured on H100 SXM): 4 warps; BLOCK_M 128 / BLOCK_N 64 / 3 stages once the
    grid has about 8 programs per SM (cdiv(N, 128) * S * H >= 8 * 132), else BLOCK_M 64 with BLOCK_N 32 / 2 stages for D <= 32 and BLOCK_N
    64 / 3 stages for 32 < D <= 64 (more, smaller programs fill the machine at the per-call sizes of one structure)."""
    BD = _pow2(D)
    progs128 = -(-int(N) // 128) * int(S) * int(H)
    if progs128 >= 8 * 132:
        return dict(BLOCK_M=128, BLOCK_N=64, num_warps=4, num_stages=3)
    if BD <= 32:
        return dict(BLOCK_M=64, BLOCK_N=32, num_warps=4, num_stages=2)
    if BD <= 64:
        return dict(BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=3)
    return dict(BLOCK_M=64, BLOCK_N=32, num_warps=4, num_stages=2)


def _meta(t: Optional[torch.Tensor]):
    """What the validation, the launch arguments and Triton's specialization read of an operand: shape, strides, dtype, device, 16-byte alignment."""
    return None if t is None else (t.shape, t.stride(), t.dtype, t.device, t.data_ptr() & 15 == 0)


_VERDICTS: Dict[tuple, tuple] = {}                             # operand metadata -> () served | (event, detail) refused
_PLANS: Dict[tuple, dict] = {}                                 # operand metadata + scale / mask_neg / config -> launch plan (+ the compiled handle once launched)
_MEMO_MAX = 512


def check(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, mask: Optional[torch.Tensor] = None,
          gate: Optional[torch.Tensor] = None, out: Optional[torch.Tensor] = None):
    """The served-domain validation of `apb_attention` without the launch: raises Unsupported(event, detail) by name, else returns
    (bias4, mask) — the bias with a leading sample dim and the mask as the kernel reads it (bool -> uint8). The verdict is a pure function
    of the operands' metadata (shapes, strides, dtypes, devices), so it is memoised on it: a repeated call shape is refused or passed by
    name without re-running the checks (every call is still counted by its caller)."""
    key = (_meta(q), _meta(k), _meta(v), _meta(bias), _meta(mask), _meta(gate), _meta(out))
    verdict = _VERDICTS.get(key)
    if verdict is None:
        try:
            _check(q, k, v, bias, mask, gate, out)
            verdict = ()
        except Unsupported as e:
            verdict = (e.event, e.detail)
        if len(_VERDICTS) >= _MEMO_MAX:
            _VERDICTS.clear()
        _VERDICTS[key] = verdict
    if verdict:
        raise Unsupported(*verdict)
    if bias.dim() == 3:
        bias = bias.unsqueeze(0)
    if mask is not None and mask.dtype == torch.bool:
        mask = mask.to(torch.uint8)
    return bias, mask


def _check(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, mask: Optional[torch.Tensor] = None,
           gate: Optional[torch.Tensor] = None, out: Optional[torch.Tensor] = None) -> None:
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise Unsupported("rank", f"q/k/v must be [S,H,N,D], got {tuple(q.shape)} {tuple(k.shape)} {tuple(v.shape)}")
    S, H, N, D = (int(x) for x in q.shape)
    NK = int(k.shape[2])
    if tuple(k.shape) != (S, H, NK, D) or tuple(v.shape) != (S, H, NK, D):
        raise Unsupported("shape:kv", f"k {tuple(k.shape)} v {tuple(v.shape)} vs q {tuple(q.shape)}")
    if q.dtype not in (torch.bfloat16, torch.float16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise Unsupported(f"dtype:{str(q.dtype).replace('torch.', '')}", "q/k/v must be bf16 or fp16 of one dtype (fp32 activations are not served)")
    if not q.is_cuda:
        raise Unsupported("device:not_cuda")
    if D > MAX_HEAD_DIM or D < 1:
        raise Unsupported(f"head_dim:{D}", f"served head dims are 1..{MAX_HEAD_DIM}")
    if N < 1 or NK < 1 or S < 1:
        raise Unsupported("size:empty")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise Unsupported("stride:d_not_unit", "q/k/v need unit stride on the head dim")
    if bias.dim() == 3:
        bias = bias.unsqueeze(0)                                # a view: validated as [1, H, N, NK]
    if bias.dim() != 4 or tuple(bias.shape[1:]) != (H, N, NK) or int(bias.shape[0]) not in (1, S):
        raise Unsupported("shape:bias", f"bias must be [H,N,NK] or [1|S,H,N,NK] = {H, N, NK}, got {tuple(bias.shape)}")
    if bias.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise Unsupported(f"dtype:bias:{str(bias.dtype).replace('torch.', '')}")
    SB = int(bias.shape[0])
    if mask is not None:
        if mask.dim() != 2 or int(mask.shape[1]) != NK or int(mask.shape[0]) not in (1, S):
            raise Unsupported("shape:mask", f"mask must be [1|S, NK], got {tuple(mask.shape)}")
    if gate is not None:
        if gate.dim() != 3 or tuple(gate.shape) != (S, N, H * D):
            raise Unsupported("shape:gate", f"gate must be [S,N,H*D], got {tuple(gate.shape)}")
        if gate.stride(-1) != 1:
            raise Unsupported("stride:gate_col_not_unit")
    if out is not None:
        if tuple(out.shape) != (S, N, H * D) or out.dtype != q.dtype or out.device != q.device:
            raise Unsupported("shape:out", f"out must be [S,N,H*D] in q's dtype on q's device, got {tuple(out.shape)} {out.dtype}")
        if out.stride(-1) != 1:
            raise Unsupported("stride:out_col_not_unit")
    for name_, t_ in (("bias", bias), ("q", q), ("k", k), ("v", v), ("mask", mask), ("gate", gate), ("out", out)):
        if t_ is None:
            continue
        ext = sum((int(d_) - 1) * abs(int(st_)) for d_, st_ in zip(t_.shape[-2:], t_.stride()[-2:]))    # extent of one [rows, cols] slice
        if ext >= 2 ** 31 - 2 ** 24:
            raise Unsupported(f"size:int32_slice:{name_}", "one (sample, head) slice must be int32-addressable (the slice bases are int64)")
        sts = t_.stride()[1:] if (name_ == "bias" and int(t_.shape[0]) == 1) else t_.stride()      # a shared bias's leading stride is never read
        if max((abs(int(st_)) for st_ in sts), default=0) >= 2 ** 31 - 2 ** 24:               # the kernel rebuilds each stride as (stride / divisor) * divisor in int32
            raise Unsupported(f"size:int32_stride:{name_}", "every stride the kernel reads must be below 2**31 (sample / head strides are then multiplied by int64 indices)")
    if N * H * D >= 2 ** 31 - 2 ** 24:
        raise Unsupported("size:int32_slice:out", "one sample's [N, H*D] output block must be int32-addressable")


_LAUNCH: Dict[str, object] = {"direct": "on", "conv": None, "jit": 0, "direct_n": 0}       # conv: the compiled launcher's argument convention ("all" | "runtime"), found at the first direct launch


def launch_census() -> dict:
    """{'direct': 'on'|'off:<reason>', 'convention', 'jit_launches', 'direct_launches', 'plans'} for the entry's census."""
    return {"direct": _LAUNCH["direct"], "convention": _LAUNCH["conv"], "jit_launches": _LAUNCH["jit"], "direct_launches": _LAUNCH["direct_n"],
            "plans": len(_PLANS)}


def set_direct_launch(on: bool) -> None:
    """Tests: force every launch through the JITFunction path (False) or re-enable the compiled-handle path (True)."""
    _LAUNCH["direct"] = "on" if on else "off:by_request"


def _direct(ck, grid, args: tuple, consts: tuple) -> bool:
    """Launch the compiled handle `ck` directly; False (and the direct path off by name) when this Triton's launcher cannot be driven so."""
    conv = _LAUNCH["conv"]
    try:
        if conv is None:                                          # first direct launch in this process: find the launcher's convention (a wrong count raises before launching)
            try:
                ck[grid](*(args + tuple(v for _, v in consts)))
                _LAUNCH["conv"] = "all"
            except TypeError:
                ck[grid](*args)
                _LAUNCH["conv"] = "runtime"
        elif conv == "all":
            ck[grid](*(args + tuple(v for _, v in consts)))
        else:
            ck[grid](*args)
    except (TypeError, AttributeError, ValueError, KeyError, NotImplementedError) as e:   # not drivable this way: named, the JIT path from now on (a CUDA error propagates as from the JIT path)
        _LAUNCH["direct"] = "off:%s" % type(e).__name__
        return False
    _LAUNCH["direct_n"] += 1
    return True


def _plan(q, k, v, bias, mask, gate, out, scale: float, mask_neg: float, config) -> dict:
    """Everything about a launch that is a function of the operands' metadata: tile config, grid, divisors, the integer and constexpr arguments."""
    S, H, N, D = (int(x) for x in q.shape)
    NK = int(k.shape[2]); SB = int(bias.shape[0])
    cfg = dict(default_config(S, N, D, H))
    if config:
        cfg.update({k_: int(v_) for k_, v_ in config.items() if k_ in cfg})
    BM, BN = int(cfg["BLOCK_M"]), int(cfg["BLOCK_N"])
    os0, os1, os2 = (int(x) for x in out.stride()) if out is not None else (N * H * D, H * D, 1)
    SDIV = _divisor([q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(1), v.stride(2),
                     (gate.stride(0) if gate is not None else 0), (gate.stride(1) if gate is not None else 0), os0, os1])
    BDIV = _divisor(([bias.stride(0)] if SB > 1 else []) + [bias.stride(1), bias.stride(2)]) if int(bias.stride(3)) == 1 or NK == 1 else 1
    d = lambda x: int(x) // SDIV  # noqa: E731
    db = lambda x: int(x) // BDIV  # noqa: E731
    ints = (d(q.stride(0)), d(q.stride(1)), d(q.stride(2)),
            d(k.stride(0)), d(k.stride(1)), d(k.stride(2)),
            d(v.stride(0)), d(v.stride(1)), d(v.stride(2)),
            db(bias.stride(0)) if SB > 1 else 0, db(bias.stride(1)), db(bias.stride(2)), int(bias.stride(3)),
            (int(mask.stride(0)) if (mask is not None and mask.shape[0] > 1) else 0), (int(mask.stride(1)) if mask is not None else 0),
            d(gate.stride(0) if gate is not None else 0), d(gate.stride(1) if gate is not None else 0), (int(gate.stride(2)) if gate is not None else 0),
            d(os0), d(os1), os2,
            S, N, NK)
    consts = (("HEAD_DIM", D), ("BD", _pow2(D)), ("BLOCK_M", BM), ("BLOCK_N", BN),
              ("HAS_MASK", mask is not None), ("MASK_PER_ROW", bool(mask is not None and mask.shape[0] > 1 and S > 1)), ("BIAS_PER_ROW", bool(SB > 1 and S > 1)),
              ("HAS_GATE", gate is not None), ("SDIV", SDIV), ("BDIV", BDIV), ("LOG2E", _LOG2E))          # in the kernel signature's order
    info = dict(cfg); info["SDIV"], info["BDIV"] = SDIV, BDIV
    info["bias_rows_aligned"] = (BDIV * bias.element_size()) % 16 == 0 and (int(bias.stride(3)) == 1 or NK == 1)
    return {"out_shape": (S, N, H * D), "grid": (S * triton.cdiv(N, BM), H, 1), "ints": ints, "consts": consts,
            "floats": (float(scale * _LOG2E), float(mask_neg * _LOG2E)), "num_warps": int(cfg["num_warps"]), "num_stages": int(cfg["num_stages"]),
            "info": info, "ck": None}


def apb_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, mask: Optional[torch.Tensor] = None,
                  gate: Optional[torch.Tensor] = None, out: Optional[torch.Tensor] = None, scale: Optional[float] = None,
                  mask_neg: float = -1e9, config: Optional[Dict[str, int]] = None, info: Optional[dict] = None,
                  checked: bool = False) -> torch.Tensor:
    """See the module docstring. Returns `out` [S, N, H*D] in q's dtype. `checked=True` skips the validation for a caller that ran
    `check(...)` on these operands already. The launch plan (tile config, divisors, integer / constexpr arguments) is memoised on the
    operands' metadata; a plan launched once holds the compiled handle and later calls launch it directly."""
    if checked:
        if bias.dim() == 3:
            bias = bias.unsqueeze(0)
        if mask is not None and mask.dtype == torch.bool:
            mask = mask.to(torch.uint8)
    else:
        bias, mask = check(q, k, v, bias, mask, gate, out)
    if scale is None:
        scale = 1.0 / math.sqrt(int(q.shape[3]))
    if not (math.isfinite(float(scale)) and math.isfinite(float(mask_neg))):
        raise Unsupported("nonfinite:scale_or_mask_neg", f"scale and mask_neg must be finite (an all-masked row is the uniform average only for a finite mask_neg), got {scale}, {mask_neg}")
    key = (_meta(q), _meta(k), _meta(v), _meta(bias), _meta(mask), _meta(gate), _meta(out), float(scale), float(mask_neg),
           tuple(sorted(config.items())) if config else None)
    plan = _PLANS.get(key)
    if plan is None:
        plan = _plan(q, k, v, bias, mask, gate, out, scale, mask_neg, config)
        if len(_PLANS) >= _MEMO_MAX:
            _PLANS.clear()
        _PLANS[key] = plan
    if out is None:
        out = torch.empty(plan["out_shape"], device=q.device, dtype=q.dtype)
    if info is not None:
        info.update(plan["info"])
    args = (q, k, v, bias, mask if mask is not None else q, gate if gate is not None else q, out) + plan["ints"] + plan["floats"]
    ck = plan["ck"]
    if ck is not None and _LAUNCH["direct"] == "on" and _direct(ck, plan["grid"], args, plan["consts"]):
        return out
    ck = _apb_fwd[plan["grid"]](*args, **dict(plan["consts"]), num_warps=plan["num_warps"], num_stages=plan["num_stages"])
    _LAUNCH["jit"] += 1
    if ck is not None and hasattr(ck, "__getitem__"):
        plan["ck"] = ck                                           # the compiled handle for this plan's specialization: launched directly from now on
    return out


def reference(q, k, v, bias, mask=None, gate=None, scale=None, mask_neg=-1e9, dtype=torch.float64) -> torch.Tensor:
    """The materialised statement in `dtype` (default fp64), in the stock arithmetic order (q.k * scale + mask bias + pair bias)."""
    S, H, N, D = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    qd, kd, vd, bd = q.to(dtype), k.to(dtype), v.to(dtype), bias.to(dtype)
    if bd.dim() == 3:
        bd = bd[None]
    s = torch.einsum("shid,shjd->shij", qd, kd) * scale
    if mask is not None:
        s = s + (mask_neg * (mask.to(dtype) == 0).to(dtype))[:, None, None, :]
    s = s + bd
    p = torch.softmax(s, dim=-1)
    o = torch.einsum("shij,shjd->sihd", p, vd).reshape(S, N, H * D)
    if gate is not None:
        o = o * torch.sigmoid(gate.to(dtype))
    return o
