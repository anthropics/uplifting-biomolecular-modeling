# fpf_transition_odde: OpenDDE's pair transition as the bf16 composition `sep16` (module docstring below).
"""fpf_transition_odde -- OpenDDE's pair_transition (c_in 384, n 4 -> hidden 1536) served in the trunk arms' numerics class by the
composition ``sep16``: the engine's own statement under bf16 autocast, minus its redundant passes.  A single-engine composition kept in this
kit BY NAME; the core's transition provider (opt_core.kernels.transition, its TRANSITION_CELLS.json) decides per cell whether one of its fused
rows or this composition serves (levers/ARMT/odde_transition_bind).

Statement-by-statement == the stock module under bf16 autocast:
   stock (autocast):  y32 = LN(chunk); a16 = linear_a(bf16(y32)); a16 = silu_(a16) [opmath fp32, round bf16]; b16 = linear_b(bf16(y32)) [y cast AGAIN];
                      b16 *= a16 [fp32 mul, round bf16]; u16 = linear_o(b16); outputs[rows] = u16 (copy); block: z += u16
   sep16:             y16 = bf16(LN(chunk)) ONCE; a16 = y16 @ Wa16^T, b16 = y16 @ Wb16^T (the SAME cuBLAS bf16 GEMM calls: same M, N, K, dtypes, layouts
                      => same kernels => same bits); h16 = one elementwise Triton kernel (silu in fp32 -> round bf16 -> fp32 mul -> round bf16: the two
                      stock rounding points, expf via libdevice == CUDA expf); torch.mm(h16, Wo16^T, out=outputs[rows]) written in place (no copy);
                      residual: the caller adds the update, or transition_into does z2[rows].add_(u16) per chunk.
Bitwise to the engine module at the op on compute capability 9.0 and 8.0 (z bf16 and fp32-under-autocast); the
LayerNorm is the module's own (upstream's fused kernel or torch's, whichever the process runs) -- never re-implemented here.

Provider first: with the kit's transition word live (levers/ARMT/odde_transition_bind: ODDE_TRANSITION=<tier word | row>), every
eligible call asks the core provider's cell decision once per call class; a CARRIED row the cell names is served through the provider face
chunk by chunk, a STOCK-row answer / a refusal by name is this composition's call -- counted on both sides, named on the LEVER line.

API:  apply(module) (idempotent; binds module.forward);  forward(module, x) -> bf16 update (the stock contract under autocast);
      transition_into(module, z) -> z += update in place, chunk-wise;  describe() / STATS: the per-call census (calls, kernel_calls, provider
      calls, stock fallbacks by reason).  Cell = (384, 1536); anything else -> the module's stock forward BY NAME (counted; never silent).
"""
__version__ = "0.2.0"
import os, sys, json, torch, torch.nn.functional as F

STATS = {"calls": 0, "kernel_calls": 0, "prov_calls": 0, "fallback": 0, "fallback_reasons": {}, "chunks": 0, "residual_calls": 0}
_FLAT_CHUNK_ROWS = 262144      # opendde primitives._TRANSITION_FLAT_CHUNK_ROWS (1.0.0 .. 1.1.1): the stock loop's row block, kept so peak memory equals the stock's
CELL = (384, 1536)
VARIANT = "sep16"              # the one composition this module carries, by name (printed on the LEVER line and in describe())
                               # (the LEVER line's variant= field; describe() reports it with the call counters)
try:
    import triton
    import triton.language as tl
    try:
        from triton.language.extra import libdevice as _ld
    except Exception:  # pragma: no cover
        try:
            from triton.language.extra.cuda import libdevice as _ld
        except Exception:
            _ld = None
    _HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None; tl = None; _ld = None; _HAS_TRITON = False


def _bind():
    """The kit's provider binding when its word is live in this process (None otherwise: this composition serves every eligible call)."""
    b = sys.modules.get("odde_transition_bind")
    if b is None:
        try:
            import odde_transition_bind as b  # noqa: F811 -- levers/ARMT is on the path wherever this module is (odde_arm_t put it there)
        except Exception:  # noqa: BLE001
            return None
    return b if b.active() else None


def _note(why):
    STATS["fallback"] += 1; STATS["fallback_reasons"][why] = STATS["fallback_reasons"].get(why, 0) + 1
    if STATS["fallback"] <= 2 or os.environ.get("FPF_VERBOSE"):
        print(f"[fpf_transition_odde] stock fallback: {why}", flush=True)


def _chunks(x2, size):
    if x2.shape[0] > _FLAT_CHUNK_ROWS: return x2.split(_FLAT_CHUNK_ROWS, dim=0)
    if size >= 3200: return torch.chunk(x2, 8, dim=0)
    return (x2,)


if _HAS_TRITON:
    @triton.jit
    def _silu_mul_kernel(A, B, H, n, BLOCK: tl.constexpr, USE_LD: tl.constexpr):
        pid = tl.program_id(0); off = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK); m = off < n
        a16 = tl.load(A + off, mask=m, other=0.0); b16 = tl.load(B + off, mask=m, other=0.0)
        af = a16.to(tl.float32)
        if USE_LD:
            e = _ld.exp(-af)
        else:
            e = tl.exp(-af)
        s16 = (af / (1.0 + e)).to(tl.bfloat16)                      # F.silu(a, inplace=True) on a bf16 tensor: opmath float, ONE rounding to bf16
        h = (b16.to(tl.float32) * s16.to(tl.float32)).to(tl.bfloat16)   # b *= a : float(b)*float(a) -> bf16
        tl.store(H + off, h, mask=m)

def silu_mul16(a16, b16, out=None):
    assert a16.dtype == b16.dtype == torch.bfloat16 and a16.is_contiguous() and b16.is_contiguous() and a16.shape == b16.shape
    h = out if out is not None else torch.empty_like(b16); n = a16.numel(); BLOCK = 2048
    _silu_mul_kernel[(triton.cdiv(n, BLOCK),)](a16, b16, h, n, BLOCK=BLOCK, USE_LD=(_ld is not None), num_warps=4)
    return h

def _w16(module, device):
    c = getattr(module, "_fpf_odde_w16", None)
    wa, wb, wo = module.linear_no_bias_a.weight, module.linear_no_bias_b.weight, module.linear_no_bias.weight
    key = (wa.data_ptr(), wb.data_ptr(), wo.data_ptr(), wa._version, wb._version, wo._version, str(device))
    if c is None or c["key"] != key:
        with torch.no_grad():
            c = {"key": key, "wa": wa.detach().to(device=device, dtype=torch.bfloat16).contiguous(), "wb": wb.detach().to(device=device, dtype=torch.bfloat16).contiguous(),
                 "wo": wo.detach().to(device=device, dtype=torch.bfloat16).contiguous()}      # == autocast's cached bf16 weight casts (same values: RN cast of the fp32 parameter)
        module._fpf_odde_w16 = c
    return c

def _hidden16(module, y16, w):
    a16 = F.linear(y16, w["wa"]); b16 = F.linear(y16, w["wb"])           # identical cuBLAS calls to autocast's linear(bf16(y), bf16(W))
    return silu_mul16(a16, b16, out=b16) if _HAS_TRITON else b16.mul_(F.silu(a16, inplace=True))

def _decide(module, x):
    """(binding, decision) under the kit's live transition word; (None, None) when no word is live."""
    B = _bind()
    if B is None:
        return None, None
    try:
        return B, B.decide(module, x)
    except Exception as e:  # noqa: BLE001 -- the provider's pure selection cannot fail a run: this composition serves, the reason counted
        _note(f"provider-decide:{type(e).__name__}")
        return None, None

def forward_sep16(module, x):
    STATS["calls"] += 1
    ok, why = eligible_sep16(module, x)
    if not ok:
        _note(why); return module._fpf_odde_orig_forward(x) if hasattr(module, "_fpf_odde_orig_forward") else None
    B, dec = _decide(module, x); prov = dec is not None and dec.kind == "provider"; served = None
    w = _w16(module, x.device); other = x.shape[:-1]; x2 = x.reshape(-1, module.c_in)
    out = torch.empty((x2.shape[0], module.c_in), dtype=torch.bfloat16, device=x.device); start = 0
    with torch.autocast("cuda", enabled=False):
        for chunk in _chunks(x2, x.shape[-2]):
            y16 = module.layernorm1(chunk).to(torch.bfloat16)
            if prov:
                try:
                    u16 = B.serve_chunk(module, chunk, y16, dec); rows = u16.shape[0]; out[start:start + rows].copy_(u16)
                    del y16, u16; start += rows; STATS["chunks"] += 1; STATS["prov_calls"] += 1; served = dec.row; continue
                except B.Aside as a:                                       # the rest of the call is this composition's, by the named rule
                    prov = False; dec = dec.aside(a.reason)
            h16 = _hidden16(module, y16, w); rows = h16.shape[0]
            torch.mm(h16, w["wo"].t(), out=out[start:start + rows])          # stock out-GEMM (same M,N,K,layout as F.linear) written in place: no outputs[...] = b copy
            del y16, h16; start += rows; STATS["chunks"] += 1; STATS["kernel_calls"] += 1; served = VARIANT
    if B is not None and dec is not None:
        B.note_call(dec, served if served not in (None, VARIANT) else B.SINGLETON)
    return out.reshape(*other, module.c_in)

def transition_into_sep16(module, z):
    """Block statement z += pair_transition(z) in place: per chunk u16 then z2[rows].add_(u16) (stock adds the whole update after the loop; per-chunk add is the same
    elementwise += on disjoint rows, and LN of later chunks reads rows not yet updated -> identical)."""
    STATS["calls"] += 1; STATS["residual_calls"] += 1
    ok, why = eligible_sep16(module, z)
    if not ok or not z.is_contiguous():
        _note(why or "residual-needs-contiguous-z"); z += (module._fpf_odde_orig_forward(z) if hasattr(module, "_fpf_odde_orig_forward") else module.forward(z)); return z
    B, dec = _decide(module, z); prov = dec is not None and dec.kind == "provider"; served = None
    w = _w16(module, z.device); z2 = z.view(-1, module.c_in); start = 0
    with torch.autocast("cuda", enabled=False):
        for chunk in _chunks(z2, z.shape[-2]):
            y16 = module.layernorm1(chunk).to(torch.bfloat16)
            if prov:
                try:
                    u16 = B.serve_chunk(module, chunk, y16, dec); rows = u16.shape[0]; z2[start:start + rows].add_(u16)
                    del y16, u16; start += rows; STATS["chunks"] += 1; STATS["prov_calls"] += 1; served = dec.row; continue
                except B.Aside as a:
                    prov = False; dec = dec.aside(a.reason)
            h16 = _hidden16(module, y16, w); rows = h16.shape[0]
            u16 = torch.mm(h16, w["wo"].t()); z2[start:start + rows].add_(u16)
            del y16, h16, u16; start += rows; STATS["chunks"] += 1; STATS["kernel_calls"] += 1; served = VARIANT
    if B is not None and dec is not None:
        B.note_call(dec, served if served not in (None, VARIANT) else B.SINGLETON)
    return z

def eligible_sep16(module, x):
    if not (torch.is_tensor(x) and x.is_cuda): return False, "not-cuda"
    if not _HAS_TRITON: return False, "no-triton"
    if module.training: return False, "training"
    if (module.c_in, module.linear_no_bias.weight.shape[1]) != CELL: return False, f"cell-{module.c_in}-{module.linear_no_bias.weight.shape[1]}"
    for lin in (module.linear_no_bias_a, module.linear_no_bias_b, module.linear_no_bias):
        if getattr(lin, "bias", None) is not None or getattr(lin, "precision", None) is not None: return False, "linear-variant"
    if x.dtype == torch.float32:
        if not (torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.bfloat16): return False, "fp32-input-without-bf16-autocast"
    elif x.dtype != torch.bfloat16: return False, f"dtype-{x.dtype}"
    return True, ""

eligible = eligible_sep16
forward, transition_into = forward_sep16, transition_into_sep16

def apply(module):
    """Bind module.forward to this composition (idempotent). odde_arm_t.bind calls this under the kit's transition word."""
    if getattr(module, "_fpf_odde_applied", False): return module
    module._fpf_odde_orig_forward = module.forward
    import types
    module.forward = types.MethodType(lambda self, x: forward(self, x), module)
    module._fpf_odde_applied = True
    return module

def describe():
    B = _bind()
    return {"version": __version__, "cell": CELL, "variant": VARIANT, "stats": json.loads(json.dumps(STATS)), "class": "bf16 autocast statement (fp32 or bf16 z); LayerNorm = the module's own",
            "provider_word": getattr(B, "WORD", None) if B is not None else None}
