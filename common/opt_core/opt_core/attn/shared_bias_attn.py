"""Fused attention with an additive pair bias SHARED across a leading sample axis — served by the carried ``flash_triattn`` kernel.

The op. ``out[g, s, h, i, :] = softmax_j( scale * q[g,s,h,i,:] . k[g,s,h,j,:] + bias[g,0,h,i,j] ) @ v[g,s,h,j,:]``: G bias groups, S samples
that share one bias per group, H heads. This is the triangle-attention core with "rows" read as "samples", so the kernel for it is the
carried Triton kernel ``flash_triattn`` (``opt_core.kernels``; family F1 as a triangle op) reached BY NAME through
``opt_core.kernels.flash_triattn_serve.kernel_module()`` (the kit routes ``flash_triattn`` or the core copy serves) — nothing is copied here.
Where it occurs: every AF3-family diffusion transformer denoises S samples of ONE conditioning, so the token attention's pair bias
``[H, N_tok, N_tok]`` (G = 1) and the sequence-local atom attention's windowed bias ``[n_windows, H, 32, 128]`` (G = n_windows) are shared by
the S samples of a denoiser call; the kernel loads each fp32 bias tile once per ROWS samples and never materialises ``[S, H, Lq, Lk]`` logits.

Contract. ``attention(q, k, v, bias, ...)`` returns ``(out, event)`` — ``out`` ``[G, S, H, Lq, D]`` in ``q``'s dtype (or ``out_dtype``),
``event`` the census word ``served:flash_triattn:<ip>[:pad<d>to<D>][:rows1]``. Every precondition it cannot meet RAISES
:class:`Refused` (``opt_core.attn.sdpa_bias.Refused``: ``.event`` a word, ``.reason`` a sentence) and the kit adapter runs the engine's own
attention and books the word — there is no silent stock path in here. Words: ``torch_missing`` · ``kernel:<kind>`` (the serve layer's probe:
triton_missing / no_cuda_device / cc_lt_8 / kernel_import_failed) · ``triton_lt_3`` (the kernel's ``tl.dot(input_precision=)`` needs triton >= 3
and the compatibility shim is off or could not be installed) · ``rank`` · ``shape:<what>`` · ``dtype:<t>`` · ``head_dim:<d>`` · ``bias_per_sample``
(the bias carries a real sample axis: nothing is shared — pass ``per_sample="rows1"`` to serve it as G*S groups of one row, worded ``:rows1``)
· ``ip:<word>`` (unknown precision word; ``tf32x3`` on triton < 3) · ``cell_uncertified:t<maj>:<dtype>:d<D>[:ip_<w>|:len_lt_<n>|:config]``
(triton 2 or no triton only: the (triton major, dtype, served head dim, precision, lengths, launch configuration) combination is not in
:data:`CERTIFIED_CELLS` — refused BEFORE any compile/launch, because on triton 2.3.1 an untested cell aborts the compiler and with it the process,
uncatchably; on triton >= 3 an uncertified combination is SERVED and named instead: the event carries ``:unverified(t<maj>:<dtype>:d<D>[:…])`` and
ONE stderr line says so the first time — the environment's uncertainty is named, never a reason to disengage) · ``launch:<ExceptionType>`` (the
kernel raised; an out-of-memory is never converted — it propagates).

Arguments (keyword but q, k, v, bias):
    q, k, v          ``[G, S, H, Lq|Lk, D]``; rank-4 ``[S, H, L, D]`` is read as G = 1. Any strides with unit stride on D (else one copy, the kernel's).
    bias             ``[G, 1, H, Lq, Lk]``; rank-4 ``[1, H, Lq, Lk]`` / rank-3 ``[H, Lq, Lk]`` are read as G = 1; a sample axis that is an
                     ``expand()``ed view (stride 0 — how an engine satisfies a `bias.shape[:-2] == q.shape[:-2]` assert without a copy) is shared
                     and read as its row 0. fp32 (cast once if not).
    scale            softmax scale applied to q.k inside the kernel (pass 1.0 when the engine pre-scales q, as Protenix does).
    input_precision  fp32 inputs only: ``tf32`` (tensor-core products, fp32 accumulate — the F4.tf32_matmul class) · ``ieee`` · ``tf32x3``
                     (3-pass compensated; triton >= 3). 16-bit inputs ignore it (bf16/fp16 products, fp32 accumulate).
    pad_head_dim     True: a head dim outside the kernel's set {16, 32, 64, 128} and <= 128 is zero-padded to the next member (q.k and the
                     output are unchanged by zero lanes; one copy of q, k, v and a sliced output — worded ``:pad<d>to<D>``); False: ``head_dim:<d>``.
    per_sample       None: a bias with S > 1 on axis 1 is ``bias_per_sample``; ``"rows1"``: serve it as G*S groups of one row.
    config           the kernel's launch configuration override (``BLOCK_M, BLOCK_N, ROWS, num_warps, num_stages[, ORDER]``); None = its table.
    exact_exp        the kernel's exponent variant (None = its default, exp(s - m) via ex2).
    triton2_shim     True: on triton < 3 install :func:`triton2_dot_shim` (``tl.dot(input_precision=w)`` -> ``allow_tf32 = w != 'ieee'``) before
                     the first launch; the event then carries ``:t2shim``. False: ``triton_lt_3``.

Numerics. Online softmax over key tiles with fp32 statistics and (for fp32 inputs at ``tf32``) TF32 products: NOT bit-exact with an eager
softmax(QK^T + b) V — fast class (tier 2; the engine's envelope decides). No atomics; the launch configuration is a pure function of
(D, Lq, Lk, H, dtype): run-to-run bit-exact at fixed inputs.

Kit adapter (engine glue stays in the kit; a lever built on this call names strategy ``F5.flash_attn_dense``)::

    from opt_core.attn import shared_bias_attn as SBA
    LEDGER = SBA.ledger(expected=("below_gate",))
    def _attention_patched(q, k, v, attn_bias, ...):              # Protenix primitives._attention, q [S,H,L,D] pre-scaled, bias [1,H,L,L]
        try:
            o, ev = SBA.attention(q, k, v, attn_bias, scale=1.0, input_precision="tf32")
        except SBA.Refused as e:
            LEDGER.fallback(e.event); return _attention_stock(q, k, v, attn_bias, ...)
        LEDGER.serve(SBA.shape_key(q)); return o[0]                # G = 1 -> [S,H,L,D]

Python floor: 3.8-compatible, standard library at import; torch / triton / the kernel are imported inside the calls.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .sdpa_bias import Refused
from ..counters import Ledger

STRATEGY = "F5.flash_attn_dense"                # the canonical strategy id (opt_core/STRATEGIES.json) a lever built on this call names
KERNEL = "flash_triattn"                          # the carried kernel serving it (opt_core.kernels.names())
SERVED_HEAD_DIMS = (16, 32, 64, 128)              # the kernel's own set (flash_triattn._SUPPORTED_D); other d <= 128 are padded when pad_head_dim
IP_WORDS = ("tf32", "ieee", "tf32x3")
PER_SAMPLE_WORDS = (None, "rows1")

# TESTED CELLS — the (triton major, input dtype, served head dim) x launch-configuration x precision combinations measured for `attention()`.
# On triton 2 anything else is `Refused('cell_uncertified:<key>')` BEFORE the kernel is reached: on triton 2.3.1 an untested cell (fp32, D=32, the AF3
# atom-attention window shape) aborts the MLIR compiler with SIGABRT — uncatchable, the process dies — so qualification is a hard pre-launch gate there,
# central to every caller, never in a kit adapter.  On triton >= 3 (compile failures are exceptions the kernel's safety net catches) an uncertified
# combination is SERVED and NAMED `unverified(<key>[:<what>])` (opt_core.kernels.cell_words): the event's suffix, ONE stderr line per key per process.
# A cell is added only with its evidence (box id, shapes) in the row's `evidence`.
# key: (triton_major, dtype word, D served by the kernel i.e. after padding). row: ips = precisions served; configs = launch configurations served
# (None = the kernel's own table entry); min_len = smallest Lq and Lk measured; evidence = where it was measured.
CERTIFIED_CELLS: Dict[Tuple[int, str, int], Dict[str, Any]] = {
    (2, "float32", 64): dict(ips=("tf32", "ieee"), min_len=216,
                             configs=(None, (64, 32, 1, 4, 2, 0), (64, 32, 2, 4, 2, 0), (64, 32, 2, 4, 3, 0), (64, 32, 4, 4, 2, 0), (128, 32, 2, 4, 2, 0)),
                             evidence="H100 torch 2.3.1/triton 2.3.1: S{10,40,80} x L{216,416,616} x H16, D48->64 pad; (64|128,64,..) configs fail to compile (caught)"),
    (3, "float32", 64): dict(ips=("tf32", "ieee", "tf32x3"), min_len=216,
                             configs=(None, (64, 32, 1, 4, 2, 0), (64, 32, 2, 4, 2, 0), (64, 32, 2, 4, 3, 0), (64, 32, 4, 4, 2, 0), (64, 64, 2, 4, 2, 0), (128, 32, 2, 4, 2, 0), (128, 64, 2, 8, 2, 0)),
                             evidence="H100 torch 2.7.1/triton 3.3: S{10,40,80} x L{216,416,616} x H16, D48->64 pad"),
    (3, "float32", 32): dict(ips=("tf32", "ieee"), min_len=32,
                             configs=(None, (32, 32, 2, 4, 2, 0), (32, 64, 2, 4, 2, 0), (32, 128, 2, 4, 2, 0), (32, 64, 4, 4, 2, 0)),
                             evidence="H100 torch 2.7.1/triton 3.3: G{54,154} x S{10,40} x H4, Lq 32 / Lk 128 (AF3 atom windows)"),
}
# Rows by compute capability AND triton version — the same convention as the kernel's launch-configuration rows
# (kernels/flash_triattn.py _CONFIG_TABLE_BY_CC): CERTIFIED_CELLS_BY_CC["<cc>|<triton major.minor>"] (exact) else ["<cc>|*"] (every triton on
# that capability) -> {(dtype word, D served): row}. On a device whose capability has a row for the cell, that row LEADS the capability-free
# table above and the word gains ":cc<cc>"; cc 9.0 has no rows here (its words and admissions are unchanged).
#   "8.0|*"  fp32 D 64 on A100: tf32, the measured launch configurations — all SINGLE-STAGE (the only A100 measurement of this cell is the
#            triton-2.3.1 sweep, where every num_stages >= 2 configuration aborts sm_80 lowering: unsupported shared->shared layout conversion,
#            SIGABRT, uncatchable); the tuned row and the triton-2.3 exception row COINCIDE for this cell, hence ONE row and no "8.0|2.3" key.
#            None = the kernel's own pick (its cc 8.0 rows; safe settings after a build failure — kernels/flash_triattn.py safe_state).
CERTIFIED_CELLS_BY_CC: Dict[str, Dict[Tuple[str, int], Dict[str, Any]]] = {
    "8.0|*": {
        ("float32", 64): dict(ips=("tf32",), min_len=216,
                              configs=(None, (64, 64, 1, 4, 1, 0), (64, 32, 2, 4, 1, 0), (64, 64, 2, 4, 1, 0), (32, 64, 1, 4, 1, 0), (64, 32, 1, 4, 1, 0)),
                              evidence="A100-SXM4-80GB (three boxes) torch 2.3.1/triton 2.3.1: S{5,10} x L{384,448,536,616} (+ S{10,40,80} x L{216,416,616}, S{1..16} x L{384..2048}) x H16, D48->64 pad, rel-RMS vs float64 < 1.5e-3, run-to-run bitwise; end-to-end in a sibling kit's --mode fast: served=6400/19200 refused=0"),
    },
}
# KNOWN-FATAL cells (documentation of the hazard; they are simply absent from CERTIFIED_CELLS): (2, "float32", 32) — triton 2.3.1 MLIR abort
# ("Dot's a/b's encoding should be of DotOperandEncodingAttr" -> "unsupported layout conversion UNREACHABLE" -> SIGABRT), H100.


def _config_key(config: Optional[Dict[str, int]]):
    if config is None:
        return None
    return (int(config["BLOCK_M"]), int(config["BLOCK_N"]), int(config["ROWS"]), int(config["num_warps"]), int(config["num_stages"]), int(config.get("ORDER", 0)))


def cell_key(triton_major_: Optional[int], dtype_word: str, d_served: int) -> Tuple[int, str, int]:
    return (int(triton_major_ or 0), str(dtype_word), int(d_served))


_CERT_MEMO: Dict[tuple, Any] = {}
_UNVERIFIED_SAID: set = set()                    # cell words served without a record, said once each (attention())


def certify(triton_major_: Optional[int], dtype_word: str, d_served: int, ip: str, lq: int, lk: int, config: Optional[Dict[str, int]] = None) -> str:
    """The cell word for the event: ``t<maj>:<dtype>:d<D>[:cc<cc>]`` for a tested cell serving this precision, these lengths and this launch
    configuration; ``t<maj>:<dtype>:d<D>[:cc<cc>]:unverified(<what>)`` on triton >= 3 for a combination without a record (``<what>`` = ``cell`` |
    ``ip_<w>`` | ``len_lt_<n>`` | ``config``: SERVED, named); raises ``Refused('cell_uncertified:...')`` for such a combination on triton 2 / no triton
    (an untested cell can abort that compiler, uncatchably). Pure table lookup — launches nothing; memoised per distinct argument tuple (the denoiser
    repeats a handful of shapes thousands of times)."""
    memo_key = (triton_major_, dtype_word, d_served, ip, int(lq), int(lk), _config_key(config))     # one device class per process: the capability is constant under this key
    hit = _CERT_MEMO.get(memo_key)
    if hit is not None:
        if isinstance(hit, Refused):
            raise Refused(hit.event, hit.reason)
        return hit
    try:
        word = _certify(triton_major_, dtype_word, d_served, ip, lq, lk, config)
    except Refused as r:
        if len(_CERT_MEMO) < 4096:
            _CERT_MEMO[memo_key] = r
        raise
    if len(_CERT_MEMO) < 4096:
        _CERT_MEMO[memo_key] = word
    return word


def _current_cc() -> Optional[str]:
    """'M.m' compute capability of the current CUDA device (torch), or None without a CUDA device (the capability-free rows apply)."""
    torch = _torch()
    if not torch.cuda.is_available():
        return None
    mj, mn = torch.cuda.get_device_capability(torch.cuda.current_device())
    return "%d.%d" % (int(mj), int(mn))


def _triton_mm() -> str:
    """major.minor of the importable triton ("" when triton is absent)."""
    from ..kernels import safe_settings
    return safe_settings.triton_mm()


def cc_row(cc: Optional[str], triton_mm: str, dtype_word: str, d_served: int):
    """``(key, row)`` of CERTIFIED_CELLS_BY_CC serving this cell on capability ``cc`` — the core's one resolution (kernels/safe_settings.resolve_cell):
    the exact "<cc>|<mm>" row's cell, else the "<cc>|*" row's, else ``(None, None)``."""
    from ..kernels import safe_settings
    return safe_settings.resolve_cell(CERTIFIED_CELLS_BY_CC, cc, triton_mm, (dtype_word, int(d_served)))


def _certify(triton_major_, dtype_word, d_served, ip, lq, lk, config):
    key = cell_key(triton_major_, dtype_word, d_served)
    word = "t%d:%s:d%d" % key
    cc = _current_cc()
    _, row = cc_row(cc, _triton_mm(), key[1], key[2])                        # a row of THIS compute capability leads the capability-free table
    if row is not None:
        word += ":cc" + cc
    else:
        row = CERTIFIED_CELLS.get(key)
    what, why = None, ""
    if row is None:
        what, why = "cell", f"no certified cell for (triton {key[0]}, {key[1]}, D={key[2]}); certified: {sorted(CERTIFIED_CELLS)}"
    elif ip not in row["ips"]:
        what, why = f"ip_{ip}", f"precision {ip} is not certified on cell {word} (certified: {row['ips']})"
    elif min(int(lq), int(lk)) < int(row["min_len"]):
        what, why = f"len_lt_{row['min_len']}", f"Lq={lq} Lk={lk}: lengths below {row['min_len']} are not certified on cell {word}"
    elif _config_key(config) not in row["configs"]:
        what, why = "config", f"launch configuration {config} is not certified on cell {word}"
    if what is None:
        return word
    if int(key[0]) >= 3:                                                       # triton >= 3: a compile failure is an exception the kernel's safety net catches —
        from ..kernels.cell_words import unverified_word                       # the combination is SERVED and named, never refused for want of a record
        return word + ":" + unverified_word(what)
    raise Refused(f"cell_uncertified:{word}" + ("" if what == "cell" else f":{what}"),
                  why + " — refused before launch: on triton 2 an untested cell can abort the compiler (SIGABRT, uncatchable)")

_SHIM_STATE: Dict[str, Any] = {"installed": False, "word": None}


def _torch():
    try:
        import torch                      # noqa: WPS433 — lazy by contract
    except ImportError as e:             # pragma: no cover
        raise Refused("torch_missing", f"torch is not importable ({e})") from None
    return torch


def triton_major() -> Optional[int]:
    """The importable triton's major version, or None when triton is not importable."""
    try:
        import triton
    except Exception:
        return None
    try:
        return int(str(triton.__version__).split(".")[0])
    except Exception:
        return None


def triton2_dot_shim() -> str:
    """On triton < 3, make ``tl.dot(a, b, acc, input_precision=w)`` mean ``tl.dot(a, b, acc, allow_tf32=(w != 'ieee'))`` (the 2.x spelling of the
    same two product classes; ``tf32x3`` has no 2.x spelling and is refused per call). Idempotent. Returns the word ``native`` (triton >= 3 or a
    2.x whose dot already takes input_precision: nothing installed) or ``t2shim``. Raises :class:`Refused` (``triton_lt_3``) when triton is absent
    or the 2.x builtin cannot be wrapped."""
    if _SHIM_STATE.get("word") in ("t2shim", "native"):                 # decided once per process
        return _SHIM_STATE["word"]
    if _SHIM_STATE["installed"]:
        return _SHIM_STATE["word"]
    try:
        import inspect
        import triton
        import triton.language as tl
        from triton.language import core as tlc
    except Exception as e:
        raise Refused("triton_lt_3", f"triton is not importable ({e!r})") from None
    try:
        params = inspect.signature(tl.dot).parameters
    except (TypeError, ValueError):
        params = {}
    if "input_precision" in params or (triton_major() or 0) >= 3:
        _SHIM_STATE.update(installed=True, word="native")
        return "native"
    if "allow_tf32" not in params or not hasattr(tlc, "builtin"):
        raise Refused("triton_lt_3", f"triton {getattr(triton, '__version__', '?')}: tl.dot takes neither input_precision nor allow_tf32")
    _orig = tl.dot

    @tlc.builtin
    def dot(input, other, acc=None, input_precision=None, allow_tf32=None, max_num_imprecise_acc=None, out_dtype=tl.float32, _builder=None):
        ip = getattr(input_precision, "value", input_precision)
        if ip is not None and str(ip) not in ("tf32", "ieee"):
            raise ValueError(f"tl.dot input_precision={ip!r} has no triton-2 spelling (tf32 | ieee only)")
        if allow_tf32 is None:
            allow_tf32 = (ip is None) or (str(ip) != "ieee")
        else:
            allow_tf32 = getattr(allow_tf32, "value", allow_tf32)
        return _orig(input, other, acc=acc, allow_tf32=bool(allow_tf32), max_num_imprecise_acc=max_num_imprecise_acc, out_dtype=out_dtype, _builder=_builder)

    # triton 2's dependency hasher (runtime/jit.py DependenciesFinder) asserts that every non-JIT callee reached from a kernel lives in a
    # `triton.` module; the wrapper is attributed to the wrapped builtin's module so the kernel's `tl.dot` still hashes as a builtin.
    dot.__module__ = getattr(_orig, "__module__", "triton.language.core")
    dot.__qualname__ = getattr(_orig, "__qualname__", "dot")
    tl.dot = dot
    tlc.dot = dot
    _SHIM_STATE.update(installed=True, word="t2shim")
    return "t2shim"


def kernel():
    """The kernel module of this process (the routed/kit copy or the core copy) via the F1 serve layer; its probe failures are ``kernel:<kind>``."""
    from ..kernels import flash_triattn_serve as F1
    try:
        F1.require()
        return F1.kernel_module()
    except F1.Refusal as r:
        raise Refused(f"kernel:{r.kind}", str(r)) from None


def kernel_impl() -> str:
    """``flash_triattn@<version>`` for the evidence line (the serve layer's word)."""
    from ..kernels import flash_triattn_serve as F1
    return F1.kernel_impl()


def shape_key(q) -> str:
    """``G<g>xS<s>xH<h>xL<lq>xD<d>`` from a canonical (or rank-4) q — the served-shape census key."""
    s = tuple(int(x) for x in q.shape)
    if len(s) == 4:
        s = (1,) + s
    return "G%dxS%dxH%dxL%dxD%d" % (s[0], s[1], s[2], s[3], s[4])


def ledger(*, expected: Tuple[str, ...] = (), min_tokens: Optional[int] = None) -> Ledger:
    """The kit's per-process census for a lever on this call: ``Ledger(STRATEGY, impl=..., expected=...)`` from ``opt_core.counters``."""
    try:
        impl = kernel_impl()
    except Exception:
        impl = KERNEL
    return Ledger(STRATEGY, impl=impl, origin=None, min_tokens=min_tokens, expected=tuple(expected))


def canonical(q, k, v, bias):
    """Views, no copies: rank-4 q/k/v ``[S,H,L,D]`` -> ``[1,S,H,L,D]``; bias ``[H,Lq,Lk]`` / ``[1,H,Lq,Lk]`` / ``[1,1,H,Lq,Lk]`` -> ``[1,1,H,Lq,Lk]``;
    rank-5 inputs pass through. Raises ``Refused('rank' | 'shape:<what>')``."""
    if q.dim() == 4:
        q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
    if q.dim() != 5 or k.dim() != 5 or v.dim() != 5:
        raise Refused("rank", f"q/k/v must be rank 4 [S,H,L,D] or rank 5 [G,S,H,L,D]; got {tuple(q.shape)} {tuple(k.shape)} {tuple(v.shape)}")
    G, S, H, Lq, D = (int(x) for x in q.shape)
    Lk = int(k.shape[3])
    if tuple(k.shape) != (G, S, H, Lk, D) or tuple(v.shape) != (G, S, H, Lk, D):
        raise Refused("shape:kv", f"k/v must be [G,S,H,Lk,D] = [{G},{S},{H},Lk,{D}]; got {tuple(k.shape)} {tuple(v.shape)}")
    while bias.dim() < 5:
        bias = bias.unsqueeze(0)
    if bias.dim() != 5:
        raise Refused("rank", f"bias must be rank 3..5; got {tuple(bias.shape)}")
    if int(bias.shape[1]) > 1 and bias.stride(1) == 0:                      # an expand()ed sample axis (stride 0) IS a shared bias: read row 0 (a view)
        bias = bias[:, :1]
    if int(bias.shape[0]) == 1 and G > 1 and int(bias.shape[1]) == G:      # [1,G,H,Lq,Lk] given for a grouped call: move G to the front
        bias = bias.transpose(0, 1)
    if int(bias.shape[2]) == 1 and H > 1:
        bias = bias.expand(bias.shape[0], bias.shape[1], H, bias.shape[3], bias.shape[4])
    if int(bias.shape[0]) == 1 and G > 1:
        bias = bias.expand(G, bias.shape[1], H, bias.shape[3], bias.shape[4])
    if tuple(bias.shape[2:]) != (H, Lq, Lk) or int(bias.shape[0]) != G:
        raise Refused("shape:bias", f"bias must broadcast to [G,1,H,Lq,Lk] = [{G},1,{H},{Lq},{Lk}]; got {tuple(bias.shape)}")
    return q, k, v, bias


def attention(q, k, v, bias, *, scale: float = 1.0, input_precision: str = "tf32", pad_head_dim: bool = True,
              per_sample: Optional[str] = None, config: Optional[Dict[str, int]] = None, exact_exp: Optional[bool] = None,
              out_dtype=None, triton2_shim: bool = True, allow_uncertified: bool = False):
    """Serve the shared-bias attention through ``flash_triattn`` or raise :class:`Refused` by name (module docstring). Returns ``(out, event)``.
    The tested-cell gate (:data:`CERTIFIED_CELLS`, :func:`certify`) runs BEFORE anything is compiled or launched: on triton >= 3 a combination
    without a record is served and the event carries ``:unverified(<what>)`` (ONE stderr line the first time); on triton 2 it is refused
    (``cell_uncertified:…``) unless ``allow_uncertified=True`` — the qualification probe's own switch (run each new cell in a subprocess: an untested
    cell may abort that compiler); a kit adapter never passes it, and the event then carries ``:UNCERTIFIED-PROBE``."""
    torch = _torch()
    if input_precision not in IP_WORDS:
        raise Refused(f"ip:{input_precision}", f"input_precision must be one of {IP_WORDS}")
    if per_sample not in PER_SAMPLE_WORDS:
        raise Refused(f"per_sample:{per_sample}", f"per_sample must be one of {PER_SAMPLE_WORDS}")
    q, k, v, bias = canonical(q, k, v, bias)
    G, S, H, Lq, D = (int(x) for x in q.shape)
    Lk = int(k.shape[3])
    dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else q.dtype
    if dt not in (torch.float32, torch.bfloat16, torch.float16):
        raise Refused(f"dtype:{str(dt).replace('torch.', '')}", "served dtypes: float32 (at input_precision), bfloat16, float16")
    if Lq < 1 or Lk < 1:
        raise Refused("shape:empty", f"Lq={Lq} Lk={Lk}")
    # head dim: the kernel's set, or zero-pad to the next member (decided here, applied after the gate)
    Dk = D
    if D not in SERVED_HEAD_DIMS:
        nxt = [d for d in SERVED_HEAD_DIMS if d > D]
        if not pad_head_dim or not nxt:
            raise Refused(f"head_dim:{D}", f"head dim {D} is not in the kernel's set {SERVED_HEAD_DIMS}" + ("" if nxt else " and exceeds 128"))
        Dk = nxt[0]
    # THE CERTIFIED-CELL GATE — a pure table lookup on (triton major, dtype, served D, precision, lengths, launch configuration), before anything
    # touches the device, the compiler or the kernel
    ip = input_precision
    dtype_word = str(dt).replace("torch.", "")
    words = []
    try:
        cw = certify(triton_major(), dtype_word, Dk, ip, Lq, Lk, config)
    except Refused:
        if not allow_uncertified:
            raise
        cw = None
        words.append("UNCERTIFIED-PROBE")
    if cw is not None and ":unverified(" in cw:                                 # served without a record (triton >= 3): the event names it; ONE line per cell word
        words.append(cw[cw.index("unverified("):])
        if cw not in _UNVERIFIED_SAID:
            _UNVERIFIED_SAID.add(cw)
            import sys
            print(f"[opt_core/shared_bias_attn] cell={cw}: no certification record for this (triton, dtype, head dim, precision, lengths, launch "
                  f"configuration) combination — served and named, not refused", file=sys.stderr, flush=True)
    if not (q.is_cuda and k.is_cuda and v.is_cuda and bias.is_cuda):
        raise Refused("device:not_cuda", "q, k, v and bias must be CUDA tensors")
    # the shared axis: a bias with a real sample axis shares nothing
    if int(bias.shape[1]) != 1:
        if int(bias.shape[1]) != S:
            raise Refused("shape:bias", f"bias axis 1 must be 1 (shared) or S={S}; got {tuple(bias.shape)}")
        if per_sample != "rows1":
            raise Refused("bias_per_sample", "the bias carries a sample axis (nothing shared across samples); pass per_sample='rows1' to serve it as G*S one-row groups")
        q, k, v = q.reshape(G * S, 1, H, Lq, D), k.reshape(G * S, 1, H, Lk, D), v.reshape(G * S, 1, H, Lk, D)
        bias = bias.reshape(G * S, 1, H, Lq, Lk)
        words.insert(0, "rows1")
    if Dk != D:
        pad = (0, Dk - D)
        q = torch.nn.functional.pad(q, pad); k = torch.nn.functional.pad(k, pad); v = torch.nn.functional.pad(v, pad)
        words.append(f"pad{D}to{Dk}")
    # triton spelling of the product precision
    if dt == torch.float32 or ip != "tf32":
        shim = triton2_dot_shim() if triton2_shim else ("native" if (triton_major() or 0) >= 3 else None)
        if shim is None:
            raise Refused("triton_lt_3", "flash_triattn's tl.dot(input_precision=) needs triton >= 3 (triton2_shim=False)")
        if shim == "t2shim":
            if ip == "tf32x3":
                raise Refused("ip:tf32x3_triton2", "tf32x3 has no triton-2 spelling (tf32 | ieee)")
            words.append("t2shim")
    mod = kernel()
    try:
        out = mod.flash_triangle_attention(q, k, v, bias, mask=None, scale=float(scale), exact_exp=exact_exp, config=config,
                                            out_dtype=out_dtype, input_precision=ip)
    except Exception as e:                                   # a launch/compile failure is a NAMED refusal; out-of-memory propagates
        from ..oom import is_oom
        if is_oom(e):
            raise
        raise Refused(f"launch:{type(e).__name__}", repr(e)[:400]) from None
    if Dk != D:
        out = out[..., :D]
    if "rows1" in words:
        out = out.reshape(G, S, H, Lq, D)
    sw = getattr(mod, "settings_word", None)                                   # the kernel serves its SAFE single-stage settings after a build failure of the
    sw = sw() if callable(sw) else None                                        # picked ones: one more word, settings=safe:build_failed:<exc> (census-able)
    if sw:
        words.append("settings=" + sw)
    event = "served:%s:%s" % (KERNEL, ip if dt == torch.float32 else str(dt).replace("torch.", "")) + "".join(":" + w for w in words)
    return out, event


def reference(q, k, v, bias, *, scale: float = 1.0, dtype=None):
    """Materialised softmax(QK^T*scale + b) V in ``dtype`` (default float64) on the canonical form — the gold reference for numerics rows."""
    torch = _torch()
    q, k, v, bias = canonical(q, k, v, bias)
    dt = torch.float64 if dtype is None else dtype
    qd, kd, vd, bd = q.to(dt), k.to(dt), v.to(dt), bias.to(dt)
    s = torch.matmul(qd, kd.transpose(-1, -2)) * float(scale) + bd
    return torch.matmul(torch.softmax(s, dim=-1), vd)


__all__ = ["STRATEGY", "KERNEL", "SERVED_HEAD_DIMS", "IP_WORDS", "CERTIFIED_CELLS", "Refused", "attention", "reference", "canonical", "certify",
           "cell_key", "shape_key", "ledger", "kernel", "kernel_impl", "triton2_dot_shim", "triton_major"]
