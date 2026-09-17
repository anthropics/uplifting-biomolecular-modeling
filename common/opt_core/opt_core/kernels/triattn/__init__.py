"""Triangle attention: ONE provider over every carried implementation ("row") and the measured cell table (TRIATTN_CELLS.json).

The op (the cuequivariance ``triangle_attention`` calling convention, which every row follows)::

    out[b,n,h,i,:] = softmax_j( scale*q[b,n,h,i,:].k[b,n,h,j,:] + bias[b,0,h,i,j]  (-1e9 where mask[b,n,0,0,j] is False) ) @ v[b,n,h,j,:]
    q/k/v [B,N,H,S,D], bias [B,1,H,S,S] fp32 (shared over the N rows), mask [B,N,1,1,S] bool, out [B,N,H,S,D] in q.dtype.

Starting vs ending node is the caller's transpose: one direction here.  A fully-masked row yields the uniform average of v.

Rows (``ROW_NAMES``; the table's ``rows`` block states each one's envelope, numerics class, backward, capture-safety and named fallback):

    k2b / k2         opt_core.kernels.fpf_triatt_k2b (Triton; cells by compute capability)          fast class, cc >= 8.0
    flash            opt_core.kernels.flash_triattn (Triton; fp32 words tf32 | tf32x3 | ieee)       fast class, cc >= 8.0
    cuda_sm90a       .cuda_sm90a  -- prebuilt sm_90a extension, bf16, D=32; loads only on a stack     fast class, cc 9.0 + a prebuilt ABI
                     with a prebuilt directory, refuses BY NAME elsewhere (fallback row: k2b)
    triattn_native   .triattn_native -- the sealed CUDA package (sm_90a members by sequence range +   fast class, cc 9.0 / 8.0 + a prebuilt ABI
                     an sm_80 member, prebuilt torch extensions); refuses BY NAME without a prebuilt
    exact_headsplit  .headsplit   -- the stock op called once per head on zero-copy head slices above  EXACT by construction, any cc
    triattn_exact    .exact_member -- the carried CUDA member (kernels/triattn_exact), bit-identical to   EXACT (bitwise vs the library op),
                     the library op on its proven cells; nvcc-built at first use; refuses by name else   cc 9.0 / 8.0, bf16 D32, forward
                     n_split tokens (pass-through below); needs the kit's stock callable
    cueq / ds4sci / sdpa          -- named STOCK rows: the kit's own stock callable (``stock=``), or this module's ``sdpa_reference``;
                     ds4sci is flagged capture-UNSAFE (replay-stale inside a CUDA graph: a graphed region must never select it)

Selection (pure; no framework import)::

    sel = select(cc, dtype, head_dim, heads, n_tokens, direction="fwd", *, word, prefer=None, stack=None, config=None)

``word`` is REQUIRED.  A row name serves exactly that row with the row's own launch cell for the device -- what a kit binding that row
today already runs, byte for byte (the default rule: nothing a kit gets today changes unless the kit asks by word).  ``"fast"`` /
``"exact"`` are the OPT-IN tier words: the cell's measured winner of that tier (``prefer=(row, ...)`` narrows the candidates and keeps the
kit's order).  ``"tf32"`` / ``"tf32x3"`` / ``"ieee"`` are the fp32-input precision words (fast tier).  ``"<row>@<setting>"`` names an
opt-in launch setting of the table's ``config_words``.  A row that cannot serve the call raises :class:`Refusal` BY NAME (``.kind``,
``.row``, ``.fallback`` = the row the cell names instead) -- never a silent substitute.  ``sel`` is a :class:`Selection` (row, config,
input_precision, the cell key and its facts: class, exact_vs, capture_safe, measured, x_stock) a kit prints in its own LEVER line
(:func:`describe`).

Serving (imports torch and the selected row at call time only)::

    out = triangle_attention(q, k, v, bias, mask=None, scale=None, *, word, prefer=None, stock=None, config=None)

derives (cc, dtype, head_dim, heads, n_tokens, direction) from the tensors (the autocast dtype under autocast; ``direction="fwdbwd"``
when a gradient is required), selects, and calls the row.  The Triton rows are imported through the core's carried-kernel route (the
same module object, cell-table exports and byte check a kit binding them through ``opt_core.attn.pair_fused`` gets), so ``word="k2b"``
IS today's ``core="k2b"`` call.
``int32_offset:<term>=<value>>=2**31`` -- a Triton row (k2b / k2 / flash) asked to serve operands whose 32-bit IN-ROW offsets would
overflow (a transposed view's position stride times the sequence length; contiguous operands never reach it): refused by name before
the kernel runs, fallback the cell's next non-Triton row (``select(..., position_stride=)`` answers the same question without tensors).

``form=`` (select / triangle_attention): ``bias_only`` | ``keypad`` (one key-padding length per batch element) | ``mask_bias`` (a per-row key
mask): where a cell carries a ``forms`` record measured at that call form, the tier word reads that form's order (contiguous or strided);
cells without one, and calls without the hint, read the cell's order as before.
``exact_vouch_not_recorded:<cc>|torch<v>|cueq<v>`` -- an exact-class row asked (by the EXACT word or by name) on a stack its bitwise vouch was not
measured on (cells / rows carry ``vouched_on``): refused by name, fallback the stock op (exact by definition); fast words are unaffected.
``no_cell:heads=<H>`` / ``no_cell:dtype=fp16`` -- a TIER word at a head count or the fp16 dtype no cell measured: refused by name (fallback the
stock op); the nearest cell is a size neighbour only.  Row words still serve those calls (the row's own launch cell).
``install_failed:<why>`` -- a sealed row whose load / byte gate failed when select() first resolved it in this process (the gate runs at
resolve time = activation, once per process, never inside the first served call): refused by name from then on.
``no_row:head_dim<D>`` -- no kernel row of this family carries the head_dim (a structural fact, not a measurement gap): a tier word serves the
stock op and names it in the selection's reason (`passed over [..., 'no_row:head_dim<D>']`), so a coverage census records a NAMED fallback.
``stack=`` accepts the in-process ABI key (unchanged) or a CELL-STYLE stack word ('H100:2.13.0+cu130/3.7.1/cueq0.11.1', the table's
'<device>:torch<build>/triton<v>/cueq<v>' labels): offline such a word stands for every built prebuilt key of that torch build and the reason says
``prebuilt: any-of [...]``; only a torch build with no built key reports ``no_prebuilt`` (see ``stack_key_candidates``).

``"big"`` -- the memory tier: ``fast``'s resolution minus any row whose RECORDED peak for the cell exceeds the stock op's (the table's
`peak_mib` records; more than max(8 MiB, 2 %)): today no cell splits, so ``big`` == ``fast`` everywhere; a row passed over says
``<row>:peak:<MiB> > <stock> <MiB>`` in the reason.
``inherited_cc:unmeasured`` -- a tier word on a compute capability WITHOUT a measured column (10.0 / 10.3 / 12.0 ...): fast / big serve
the nearest measured column's portable Triton rows (k2b / k2 / flash, source-compiled here; never an sm_90a cubin / extension or another
arch's prebuilt, never the library op while a portable row admits), exact = the library op by name; measured=False, the census says so.
``stepped_aside:error:<ExcType>`` -- an INHERITED portable row failed to build / compile / launch on a part without a column: dead for the
process (census once), the donor cell's next portable row serves, else the library op by name; out-of-memory is re-raised as everywhere.
"""
import json
import sys
import math
import os
import re
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

from ... import cell_census as _CENSUS                  # stdlib-only: the coverage census (one record per decided call class; pure observation)

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS_FILE = os.path.join(HERE, "TRIATTN_CELLS.json")
PREBUILT_DIR = os.path.join(HERE, "cuda_sm90a", "prebuilt")

ROW_NAMES = ("k2b", "k2", "flash", "cuda_sm90a", "triattn_native", "exact_headsplit", "triattn_exact", "cueq", "ds4sci", "sdpa", "stock")
TIER_WORDS = ("fast", "exact", "big")
PEAK_NOISE_MIB = 8.0                            # big's peak rule: a row's recorded peak EXCEEDS the stock row's when above it by more than max(8 MiB, 2 %)
PEAK_NOISE_FRAC = 0.02
FP32_WORDS = ("tf32", "tf32x3", "ieee")
STOCK_ROWS = ("cueq", "ds4sci", "sdpa", "stock")          # "stock" = the kit's own stock callable whatever it is (a cell measured against SDPA names it so)
NEEDS_STOCK = ("cueq", "ds4sci", "stock", "exact_headsplit", "triattn_exact")
TRITON_ROWS = ("k2b", "k2", "flash")          # the carried Triton kernels: 64-bit batch / row / head bases, 32-bit IN-ROW (position) offsets
KERNEL_ROWS = ("k2b", "k2", "flash", "triattn_native", "cuda_sm90a", "triattn_exact")   # the rows that are kernels of this family (the stock rows and the per-head split of the stock op are not): a call none of them admits has no row BY NAME

_ABI_KEY_RE = re.compile(r"^torch\d+\.\d+\.\d+(?:\+[a-z0-9.]+)?-cpython-\d+")          # the in-process stack key: torch build + interpreter ABI (+ '-sm90' for cuda_sm90a)
_TORCH_BUILD_RE = re.compile(r"(?:torch)?(\d+\.\d+\.\d+\+[a-z]+\d+)")                  # the torch build inside a cell-style stack word ('H100:2.13.0+cu130/3.7.1/cueq0.11.1', ...)


def stack_key_candidates(stack: Optional[str], built) -> Optional[List[str]]:
    """The built prebuilt keys a ``stack=`` word stands for.  None: no word (a live process resolves its own key at install).  An ABI key (what
    ``cuda_sm90a.stack_key()`` returns in-process, e.g. 'torch2.13.0+cu130-cpython-311-x86_64-linux-gnu-sm90') matches itself only -- the
    serving path is unchanged.  A CELL-STYLE stack word as the table and kit notes spell it ('H100:2.13.0+cu130/3.7.1/cueq0.11.1',
    'H100-80GB:torch2.13.0+cu130/triton3.7.1/cueq0.11.1/...', the A100 forms) names a torch build but not the interpreter: offline it stands
    for EVERY built key of that torch build (any-of), so a CPU-side resolution query does not report a false no_prebuilt."""
    if stack is None:
        return None
    built = sorted(built)
    if _ABI_KEY_RE.match(stack):
        base = re.sub(r"-sm_?\d+a?$", "", stack)                      # any arch suffix ('-sm90', '-sm80', ...): the package keys carry none, the extension keys carry theirs
        return [k for k in built if k == stack or re.sub(r"-sm_?\d+a?$", "", k) == base]
    m = _TORCH_BUILD_RE.search(stack)
    if m is None:
        return [k for k in built if k == stack]
    pre = f"torch{m.group(1)}-cpython-"
    return [k for k in built if k.startswith(pre)]
INT32_OFFSET_LIMIT = 2 ** 31                  # every in-row element-offset product a Triton row forms stays below this, or the row refuses by name
FWD, FWDBWD = "fwd", "fwdbwd"

_DTYPE_WORDS = {"bf16": "bf16", "bfloat16": "bf16", "torch.bfloat16": "bf16",
                "fp16": "fp16", "float16": "fp16", "half": "fp16", "torch.float16": "fp16",
                "fp32": "fp32", "float32": "fp32", "float": "fp32", "torch.float32": "fp32"}
_DIRECTION_WORDS = {None: FWD, "fwd": FWD, "forward": FWD, "inference": FWD, "start": FWD, "starting": FWD, "end": FWD, "ending": FWD,
                    "outgoing": FWD, "incoming": FWD,
                    "bwd": FWDBWD, "fwdbwd": FWDBWD, "fwd+bwd": FWDBWD, "backward": FWDBWD, "grad": FWDBWD, "train": FWDBWD}


class Refusal(RuntimeError):
    """A row cannot serve this call.  ``kind`` = the refusal word, ``row`` = the row asked for, ``fallback`` = the row the cell names instead
    (what the kit binds when it does not want this refusal to be its hard error).  Never raised after a row started computing."""

    def __init__(self, kind: str, row: str, fallback: Optional[str], detail: str = ""):
        self.kind, self.row, self.fallback, self.detail = kind, row, fallback, detail
        RuntimeError.__init__(self, f"triattn row {row!r} refused: {kind}" + (f" ({detail})" if detail else "")
                              + (f"; the cell's row to bind instead: {fallback!r}" if fallback else ""))


class Selection(NamedTuple):
    row: str                       # the row that serves
    word: str                      # the word the caller passed
    cell: Optional[str]            # the TRIATTN_CELLS key that decided (None: a row word outside every measured cell)
    config: Optional[Dict[str, int]]   # an opt-in launch setting for a Triton row (None: the row's own cell table decides, as today)
    input_precision: Optional[str]     # fp32 inputs on flash: 'tf32' | 'tf32x3' | 'ieee' (None: the row's default)
    cls: str                       # 'fast' | 'exact' | 'stock'
    exact_vs: Optional[str]        # for an exact row: the stock op its output equals bit for bit
    capture_safe: bool             # False: never select inside a CUDA-graph capture (replay-stale)
    backward: bool                 # the row differentiates
    measured: bool                 # the cell was measured at this (cc, dtype, head_dim, heads, size); False = nearest cell (reason says which)
    x_stock: Optional[float]       # stock ms / row ms in that cell (median over measured stacks); None for a stock row or an unmeasured row
    tested: bool                   # exact_headsplit: cc in the row's tested_cc; other rows: True when admitted
    reason: str                    # one line: why this row (for the kit's LEVER line)


# ----------------------------------------------------------------------------------------------------------------- the table
_TABLE: Optional[Dict[str, Any]] = None


def table() -> Dict[str, Any]:
    """TRIATTN_CELLS.json, read once."""
    global _TABLE
    if _TABLE is None:
        with open(CELLS_FILE, encoding="utf-8") as f:
            _TABLE = json.load(f)
    return _TABLE


def rows() -> Dict[str, Dict[str, Any]]:
    return table()["rows"]


def cells() -> Dict[str, Dict[str, Any]]:
    return table()["cells"]


_STACKS_BUILT: Optional[List[str]] = None
_SELECT_MEMO: Dict[tuple, Any] = {}            # select() is pure given the table and the prebuilt trees (immutable in a serving process): memoised per argument tuple


def select_cache_clear() -> None:
    """Forget memoised selections and prebuilt listings (after a prebuilt was added in THIS process; tests)."""
    global _STACKS_BUILT
    _SELECT_MEMO.clear(); _SEL_CACHE.clear(); _STACKS_BUILT = None; _RESOLVE_REFUSED.clear(); _RESOLVED.clear()
    from . import triattn_native as _CC                       # stdlib-only module
    _CC.refresh()


def stacks_built() -> List[str]:
    """The stack keys cuda_sm90a ships a prebuilt extension for (directory names under cuda_sm90a/prebuilt/; read once per process)."""
    global _STACKS_BUILT
    if _STACKS_BUILT is None:
        _STACKS_BUILT = (sorted(d for d in os.listdir(PREBUILT_DIR) if os.path.isfile(os.path.join(PREBUILT_DIR, d, "manifest.json")))
                         if os.path.isdir(PREBUILT_DIR) else [])
    return list(_STACKS_BUILT)


def norm_cc(cc: Union[str, float, Tuple[int, int], List[int]]) -> Tuple[int, int]:
    if isinstance(cc, (tuple, list)):
        return int(cc[0]), int(cc[1])
    s = str(cc).strip()
    if s.startswith("sm_") or s.startswith("sm"):
        s = s[3:] if s.startswith("sm_") else s[2:]
        s = s.rstrip("a")
        return int(s[:-1]), int(s[-1])
    major, _, minor = s.partition(".")
    return int(major), int(minor or 0)


def cc_word(cc) -> str:
    M, m = norm_cc(cc)
    return f"{M}.{m}"


def norm_dtype(dtype) -> str:
    s = str(dtype).strip().lower()
    if s not in _DTYPE_WORDS:
        raise ValueError(f"triattn: dtype {dtype!r} is not one of bf16 / fp16 / fp32")
    return _DTYPE_WORDS[s]


def norm_direction(direction) -> str:
    s = direction if direction is None else str(direction).strip().lower()
    if s not in _DIRECTION_WORDS:
        raise ValueError(f"triattn: direction {direction!r} unknown (fwd | fwdbwd; starting / ending are the caller's transpose = fwd)")
    return _DIRECTION_WORDS[s]


def size_bucket(n_tokens: int, sizes: Sequence[int]) -> Tuple[Optional[int], bool]:
    """(the smallest listed size >= n_tokens, True) or (the largest listed size, False) above every listed size; (None, False) if none."""
    if not sizes:
        return None, False
    for s in sorted(sizes):
        if int(n_tokens) <= s:
            return s, True
    return max(sizes), False


def _cell_dtype(dtype: str) -> str:
    return "bf16" if dtype == "fp16" else dtype           # 16-bit cells were measured in bf16; fp16 runs the same Triton statements


def cell_for(cc, dtype, head_dim: int, heads: int, n_tokens: int, direction=FWD) -> Tuple[Optional[str], Optional[Dict[str, Any]], bool, str]:
    """(key, cell, measured, note) -- the measured cell deciding this call.  measured=False when the size is above every measured size, the
    head count or the 16-bit dtype was not itself measured (note says which); (None, None, False, note) when no cell of this (cc, dtype,
    head_dim) exists at all."""
    ccw, dt, d = cc_word(cc), _cell_dtype(norm_dtype(dtype)), norm_direction(direction)
    prefix = f"{ccw}|{dt}|D{int(head_dim)}|"
    by_heads: Dict[int, List[int]] = {}
    for key in cells():
        if key.startswith(prefix) and key.endswith("|" + FWD):
            _, _, _, hpart, npart, _ = key.split("|")
            by_heads.setdefault(int(hpart[1:]), []).append(int(npart[3:]))
    if not by_heads:
        return None, None, False, f"no measured cell for cc {ccw} {dt} D{head_dim}"
    notes = []
    h = int(heads)
    if h not in by_heads:
        below = [x for x in by_heads if x < h]
        h_use = max(below) if below else min(by_heads)
        notes.append(f"heads {h} not measured: the H{h_use} cell")
    else:
        h_use = h
    size, inside = size_bucket(int(n_tokens), by_heads[h_use])
    if not inside:
        notes.append(f"{n_tokens} tokens is above the largest measured size {size}")
    if norm_dtype(dtype) == "fp16":
        notes.append("fp16 takes the bf16 cell")
    if d == FWDBWD:
        notes.append("a backward is required: only rows with backward serve")
    key = f"{ccw}|{dt}|D{int(head_dim)}|H{h_use}|N<={size}|{FWD}"
    return key, cells()[key], not notes or notes == ["a backward is required: only rows with backward serve"], "; ".join(notes)


# ----------------------------------------------------------------------------------------------------------------- admission (static envelope)
def int32_offset_terms(n_q: int, n_k: int, head_dim: int, *, q_pos_stride: int, k_pos_stride: int, v_pos_stride: int,
                       mask_key_stride: int = 0, bias_q_stride: int = 0, bias_k_stride: int = 0) -> Dict[str, int]:
    """Every element-offset product the carried Triton rows (k2b / k2 / flash) form in 32-bit arithmetic, by name.  Audit of the three
    kernels: the batch, pair-row and head bases are promoted to 64-bit (``b``, ``i``, ``h`` are int64 before they meet a stride); what stays
    32-bit is the offset WITHIN one (batch, row, head) slice -- the position index times the position stride of q / k / v (a whole-row query
    tile ``(n_q-1)*stride+d``, the key-tile base ``start_n*stride`` plus the tile), the key index times the mask's key stride, the bias
    staging pass over the caller's bias strides, and the staged fp32 bias ``n_q * ceil16(n_k)`` (the one product the kernels' own wrapper
    checks).  Contiguous ``[B, N, H, S, D]`` operands have position stride ``D``: no term comes near 2**31 at any size a GPU holds.  A
    transposed VIEW (the ending direction served by strides instead of a copy) has position stride ``N*H*D`` or more, and ``(S-1) * that``
    passes 2**31 in reach of today's crops (H=16, D=64: from 1449 tokens; H=4, D=64: from 2897; H=4, D=32: from 4097) -- past it the kernels
    compute silently wrong values, so the face refuses BY NAME first (``int32_offset``)."""
    skp = -(-int(n_k) // 16) * 16
    return {"q_pos": (int(n_q) - 1) * int(q_pos_stride) + int(head_dim), "k_pos": (int(n_k) - 1) * int(k_pos_stride) + int(head_dim),
            "v_pos": (int(n_k) - 1) * int(v_pos_stride) + int(head_dim), "mask_key": (int(n_k) - 1) * abs(int(mask_key_stride)),
            "bias_src": (int(n_q) - 1) * abs(int(bias_q_stride)) + (int(n_k) - 1) * abs(int(bias_k_stride)), "bias_staged": int(n_q) * skp}


def int32_offsets_ok(terms: Dict[str, int]) -> Tuple[bool, str]:
    """(True, "ok"), or (False, the refusal word ``int32_offset:<term>=<value>[,...]>=2**31``)."""
    over = [f"{k}={v}" for k, v in terms.items() if int(v) >= INT32_OFFSET_LIMIT]
    return (not over, "ok" if not over else "int32_offset:" + ",".join(over) + ">=2**31")


def position_stride_bound(n_tokens: int, head_dim: int) -> int:
    """The largest q / k / v position stride (elements between consecutive sequence positions inside one (batch, row, head) slice) the Triton
    rows serve at ``n_tokens``: ``(n_tokens - 1) * stride + head_dim < 2**31``.  Contiguous operands have stride ``head_dim``."""
    n = max(int(n_tokens) - 1, 1)
    return (INT32_OFFSET_LIMIT - int(head_dim) - 1) // n


def admits(row: str, cc, dtype, head_dim: int, heads: int, n_tokens: int, direction=FWD, *, stack: Optional[str] = None,
           position_stride: Optional[int] = None) -> Tuple[bool, str]:
    """(True, "ok") when ``row``'s static envelope (the table's rows block: cards, ABI, dtypes, head dims, size cap, backward) admits the
    call; else (False, <refusal word>).  Run-time conditions a row checks on the tensors themselves (strides, mask layout, shared-memory
    fit) are that row's own named refusals at call time.  ``position_stride``: the q / k / v position stride the kit will pass (default:
    contiguous, ``head_dim``) -- a Triton row refuses ``int32_offset:...`` when ``(n_tokens-1) * position_stride + head_dim >= 2**31``
    (``triangle_attention`` checks the actual tensors' every 32-bit term regardless)."""
    if row in _RESOLVE_REFUSED:                               # the row's load / byte gate failed when it was resolved in this process
        return False, f"install_failed:{_RESOLVE_REFUSED[row]}"
    if row not in ROW_NAMES:
        return False, f"unknown_row:{row}"
    R = rows()[row]
    M, m = norm_cc(cc)
    dt, d = norm_dtype(dtype), norm_direction(direction)
    if d == FWDBWD and R.get("backward") is False:
        return False, "no_backward"
    if row in STOCK_ROWS:
        return True, "ok"
    if row == "exact_headsplit":
        return (True, "ok") if int(heads) > 1 else (False, "single_head")
    if row == "triattn_exact":
        if f"{M}.{m}" not in R["cc"]:
            return False, f"cc {M}.{m}: served on " + " / ".join(R["cc"]) + " only"
        if dt not in R["dtypes"]:
            return False, f"dtype_{dt}"
        if int(head_dim) not in R["head_dims"]:
            return False, f"head_dim_{head_dim}"
        if n_tokens is not None and int(n_tokens) < int(R.get("min_tokens", 0)):     # below the package's smallest proven S every call would be refused per call:
            return False, f"below_min_tokens_{R.get('min_tokens')}"                  # pass the row over by name here instead (the stock op, no per-call refusal)
        from . import exact_member as _EM                       # stdlib-only module: the library check reads distribution metadata, imports nothing
        return _EM.library_gate(R)
    if row in TRITON_ROWS:
        if (M, m) < (8, 0):
            return False, f"cc<8.0:sm_{M}{m}"
        if dt not in R["dtypes"]:
            return False, f"dtype_{dt}"
        if int(head_dim) not in R["head_dims"]:
            return False, f"head_dim_{head_dim}"
        ps = int(head_dim) if position_stride is None else int(position_stride)
        ok, why = int32_offsets_ok(int32_offset_terms(n_tokens, n_tokens, head_dim, q_pos_stride=ps, k_pos_stride=ps, v_pos_stride=ps))
        if not ok:
            return False, why
        return True, "ok"
    if row == "cuda_sm90a":
        if f"{M}.{m}" not in R["cc"]:
            return False, f"cc {M}.{m}: built for sm_90a only"
        if dt not in R["dtypes"]:
            return False, f"dtype_{dt}"
        if int(head_dim) not in R["head_dims"]:
            return False, f"head_dim_{head_dim}"
        if int(n_tokens) > int(R["max_tokens"]):
            return False, f"tokens>{R['max_tokens']}"
        cands = stack_key_candidates(stack, stacks_built())
        if cands is not None and not cands:
            return False, f"no_prebuilt:{stack}"
        if cands and cands != [stack]:                       # a cell-style stack word (offline query): the built keys of that torch build it stands for
            return True, f"prebuilt: any-of {cands}"
        return True, "ok"
    if row == "triattn_native":
        if f"{M}.{m}" not in R["cc"]:
            return False, f"cc {M}.{m}: measured on " + " / ".join(R["cc"]) + " only"
        from . import triattn_native as _CC                     # stdlib-only module: nothing of the payload is imported here
        strided = position_stride is not None and int(position_stride) != int(head_dim)
        gl = _CC.triton_gluon_available() if (stack is not None or (M, m) != (9, 0)) else None   # off 9.0 the Triton member is the only route: probe its dialect
        if stack is not None and not _ABI_KEY_RE.match(stack):                                  # a cell-style stack word (offline query): any built key of that torch build
            cands = stack_key_candidates(stack, _CC.stacks_built().keys())
            if not cands:
                ok0, why0 = _CC.admits(dt, int(head_dim), int(n_tokens), key=None, strided=strided, triton_gluon=gl, cc=(M, m))   # the route's own verdict first (domain / Triton member)
                return (False, f"no_prebuilt:{_CC.ACTIVE_PKG}@{stack}") if ok0 and (M, m) == (9, 0) and _CC.route_kind(dt, int(head_dim), int(n_tokens), strided).startswith("ext:") else (ok0, why0)
            verdicts = [(k, _CC.admits(dt, int(head_dim), int(n_tokens), key=k, strided=strided, triton_gluon=gl, cc=(M, m))) for k in cands]
            oks = [k for k, (ok_, _w) in verdicts if ok_]
            if oks:
                return True, (f"prebuilt: any-of {oks}" if verdicts[0][1][1].startswith("prebuilt") else verdicts[0][1][1])
            return verdicts[0][1]
        return _CC.admits(dt, int(head_dim), int(n_tokens), key=stack, strided=strided, triton_gluon=gl, cc=(M, m))
    return False, f"unknown_row:{row}"                    # pragma: no cover


def _fallback_for(row: str, cell: Optional[Dict[str, Any]], args, stack, position_stride: Optional[int] = None,
                  exclude: Sequence[str] = ()) -> str:
    """The row the cell names instead of a refusing ``row``: the row's declared fallback if it admits the call, else the first admitted row of
    the cell's fast_order, else the stock op.  ``exclude``: rows sharing the refusing row's condition (never named as the fallback)."""
    cand = []
    fb = rows().get(row, {}).get("fallback")
    if fb:
        cand.append(fb)
    if cell:
        cand += [r.split("@")[0] for r in cell.get("fast_order", []) if r.split("@")[0] != row]
    for r in cand:
        if r in exclude:
            continue
        if r in STOCK_ROWS or admits(r, *args, stack=stack, position_stride=position_stride)[0]:
            return r
    return "cueq"


def _config_word(word: str, cc=None) -> Tuple[str, Optional[Dict[str, int]], str]:
    """'k2b@m128r2' -> ('k2b', {...}, evidence); a bare row -> (row, None, ''); 'triattn_native@<ver>' -> the sealed package while <ver> is the active
    payload, or while the active payload HONOURS <ver> on this device class (triattn_native.honours: generation 11 serves `triattn_native@v10` on cc 9.0 --
    generation 10's routes and outputs, unchanged -- and refuses it by name on cc 8.0, where the two generations run different kernels)."""
    if "@" not in word:
        return word, None, ""
    if word.startswith("triattn_native@"):                       # the sealed package named with its version
        from . import triattn_native as _CC
        ver = word.split("@", 1)[1]
        ok, why = _CC.honours(ver, norm_cc(cc) if cc is not None else None)
        if not ok:
            raise Refusal(f"pkg_version:{why}", "triattn_native", rows().get("triattn_native", {}).get("fallback", "cuda_sm90a"))
        return "triattn_native", None, f"package {ver}" + ("" if str(ver) == str(_CC.ACTIVE_PKG) else f" ({why})")
    cw = table()["words"]["config_words"]
    if word not in cw:
        raise Refusal(f"unknown_config_word:{word}", word.split("@")[0], word.split("@")[0])
    return word.split("@")[0], dict(cw[word]["config"]), cw[word].get("evidence", "")


_RESOLVE_REFUSED: Dict[str, str] = {}      # row -> why its resolve-time install failed in this process (admits refuses it by name from then on)
_RESOLVED: Dict[str, dict] = {}             # row -> the resolve-time install report (seconds, verdict cache)


def _install_at_resolve(row: str) -> Optional[str]:
    """Run a sealed row's load + digest + byte gate ONCE per process when the row is first resolved by select() -- i.e. at the kit's activation
    (LEVER line / describe / warm), before any timed call -- instead of inside the first served call.  Needs a CUDA device in the process
    (torch imported and a device visible); otherwise nothing happens here (table queries, CPU interpreters) and the serving call installs as
    before.  Returns None when installed (or nothing to do), else the failure word (the caller refuses the row by name)."""
    if row in _RESOLVED or row in _RESOLVE_REFUSED or row not in ("triattn_native", "cuda_sm90a", "triattn_exact"):
        return _RESOLVE_REFUSED.get(row)
    torch = sys.modules.get("torch")
    if torch is None or not hasattr(torch, "cuda") or not torch.cuda.is_available():
        return None
    import time as _time
    t0 = _time.perf_counter(); rep: dict = {}
    try:
        if row == "triattn_native":
            from . import triattn_native as _CC
            r = _CC.install(); rep = {"verdict_cache": r.get("verdict_cache"), "install_s": r.get("install_s")}
        elif row == "cuda_sm90a":
            from . import cuda_sm90a as _CS
            r = _CS.install(); rep = {"route": r.get("route")}
        elif row == "triattn_exact":
            from . import exact_member as _EM
            r = _READY["triattn_exact"] = _EM.install(); rep = {k_: r.get(k_) for k_ in ("probe", "route", "backend", "prepare_s", "cache_dir", "proven_cells")}
    except Exception as e:                                     # the row cannot load here: refused by name from now on (is_oom first: memory pressure re-raises)
        from opt_core.oom import is_oom
        if is_oom(e): raise
        return f"{type(e).__name__}:{str(e)[:120]}"
    rep["seconds"] = round(_time.perf_counter() - t0, 3); _RESOLVED[row] = rep
    return None


def select(cc, dtype, head_dim: int, heads: int, n_tokens: int, direction=FWD, *, word: str,
           prefer: Optional[Sequence[str]] = None, stack: Optional[str] = None,
           config: Optional[Dict[str, int]] = None, position_stride: Optional[int] = None, form: Optional[str] = None,
           exact_stack: Optional[str] = None, lib: Optional[str] = None) -> Selection:
    """The row serving this call for ``word`` (see the module docstring).  Raises :class:`Refusal` (by name, with the fallback row) when the
    asked row -- or every candidate of a tier word -- cannot serve.  Memoised per argument tuple (the answer depends only on the table and the
    prebuilt trees, both immutable in a serving process; ``select_cache_clear()`` forgets).  Arguments: see ``_select_uncached``."""
    try:
        key = (tuple(norm_cc(cc)), str(dtype), int(head_dim), int(heads), int(n_tokens), str(direction), word, tuple(prefer) if prefer else None, stack,
               tuple(sorted((config or {}).items())) or None, None if position_stride is None else int(position_stride), form, exact_stack, lib)
    except (TypeError, ValueError, Refusal):
        key = None                                            # odd arguments: answered uncached (the uncached path raises the proper error by name)
    if key is not None:
        hit = _SELECT_MEMO.get(key)
        if hit is not None:
            if isinstance(hit, Selection):
                return hit
            raise Refusal(*hit)
    try:
        sel = _select_uncached(cc, dtype, head_dim, heads, n_tokens, direction, word=word, prefer=prefer, stack=stack, config=config,
                               position_stride=position_stride, form=form, exact_stack=exact_stack, lib=lib)
        failed = _install_at_resolve(sel.row)                     # a sealed row's load + byte gate runs when the row is first RESOLVED (activation), not in the first served call
        if failed:
            _RESOLVE_REFUSED[sel.row] = failed
            base_ = str(word).split("@")[0]
            if base_ in ROW_NAMES:                                # the named row cannot load in this process: refused by name now (the kit binds the fallback at activation)
                raise Refusal(f"install_failed:{failed}", sel.row, _fallback_for(sel.row, cells().get(sel.cell) if sel.cell else None,
                              (cc, dtype, head_dim, heads, n_tokens, direction), stack, position_stride))
            sel = _select_uncached(cc, dtype, head_dim, heads, n_tokens, direction, word=word, prefer=prefer, stack=stack, config=config,
                                   position_stride=position_stride, form=form, exact_stack=exact_stack, lib=lib)   # the tier passes the row over by name (admits reads _RESOLVE_REFUSED)
    except Refusal as r:
        if key is not None and len(_SELECT_MEMO) < 4096:
            _SELECT_MEMO[key] = (r.kind, r.row, r.fallback, r.detail)
        _census(cc, dtype, head_dim, heads, n_tokens, direction, word, stack if exact_stack is None else exact_stack, form, None, r)
        raise
    if key is not None and len(_SELECT_MEMO) < 4096:
        _SELECT_MEMO[key] = sel
    _census(cc, dtype, head_dim, heads, n_tokens, direction, word, stack if exact_stack is None else exact_stack, form, sel, None)   # the census key's stack: the exact word's running stack when named, else the ABI key
    return sel


_PASSED_RE = re.compile(r"passed over \['([^'\]]+?):([^'\]]*)'")     # _select_uncached's reason text for candidates refused by name ahead of the served row


def _census(cc, dtype, head_dim, heads, n_tokens, direction, word, stack, form, sel, refusal) -> None:
    """Classify the decision (cell_hit | inherited | named_fallback | stock | opt_in) from the cell facts and record it in opt_core.cell_census
    (once per memoised argument tuple; the Selection / Refusal was decided first and is untouched)."""
    try:
        ckey, _cell, measured, note = cell_for(cc, dtype, head_dim, heads, n_tokens, direction)
        inside = False
        if ckey:
            try:
                inside = int(n_tokens) <= int(ckey.split("|")[4][3:])
            except (IndexError, ValueError):
                inside = False
        key = dict(cc=cc_word(cc), stack=stack, dtype=norm_dtype(dtype), shape="D%sH%s" % (int(head_dim), int(heads)),
                   bucket=_CENSUS.bucket_word(n_tokens, ckey, inside), form=norm_direction(direction) + (("+" + str(form)) if form else ""), word=word)
        if refusal is not None:
            _CENSUS.record("triattn", key, "named_fallback", "caller:%s" % (refusal.fallback or "-"), cell_id=ckey,
                           refused="%s:%s" % (refusal.row or word, refusal.kind), note="raised")
            return
        base = str(word).split("@")[0]
        if base in ROW_NAMES:
            _CENSUS.record("triattn", key, "opt_in", sel.row, cell_id=ckey, note="" if measured else note)
            return
        m = _PASSED_RE.search(sel.reason or "")
        if INHERITED_CC in (sel.reason or ""):               # a tier word on a cc without a measured column: served from the donor column's portable rows
            _CENSUS.record("triattn", key, "inherited", sel.row, cell_id=sel.cell,
                           note="%s(from %s)%s" % (INHERITED_CC, donor_cc(cc, dtype, head_dim, heads), ";partial_cc" if cc_word(cc) in measured_ccs() else ""))
        elif m is not None:
            _CENSUS.record("triattn", key, "named_fallback", sel.row, cell_id=ckey, refused="%s:%s" % (m.group(1), m.group(2)))
        elif ckey is None:
            _CENSUS.record("triattn", key, "inherited", sel.row, cell_id=None, note="family:none(%s)" % note)
        elif not measured:                                  # the vocabulary of an inherited note: beyond_measured | guard:<dimension> (not a size neighbour)
            why = [("guard:heads(%s)" if p.startswith("heads") else "guard:dtype(%s)" if p.startswith("fp16") else "beyond_measured(%s)"
                    if "above the largest" in p else "%s") % p for p in str(note).split("; ") if p and not p.startswith("a backward")]
            _CENSUS.record("triattn", key, "inherited", sel.row, cell_id=ckey, note=";".join(why))
        elif sel.row in STOCK_ROWS:
            _CENSUS.record("triattn", key, "stock", sel.row, cell_id=ckey)
        else:
            _CENSUS.record("triattn", key, "cell_hit", sel.row, cell_id=ckey)
    except Exception:                                       # the census counts; it never gates or breaks a selection
        return


def vouch_keys(row: str, exact_stack: Optional[str]) -> List[str]:
    """The vouch-key spellings under which ``row`` counts as vouched on ``exact_stack``.  Rows that declare ``ops_builds`` (their equality
    depends on the library's ops build, not only its version) are recorded with the build appended to the library field --
    '<cc>|torch<v>|cueq<v>+<build>[|<device tag>]': the live build when a distribution is visible, else (an offline table query) any build the
    row declares.  Every other row: the key as given."""
    if not exact_stack:
        return []
    builds = rows().get(row, {}).get("ops_builds")
    if not builds:
        return [str(exact_stack)]
    from . import exact_member as _EM                            # stdlib-only: distribution metadata, nothing imported
    found = _EM.ops_distribution()
    live = [found[1]] if found else [str(b) for b in builds]
    parts = str(exact_stack).split("|")
    if len(parts) < 3:
        return [str(exact_stack)]
    return ["|".join(parts[:2] + [parts[2] + "+" + b] + parts[3:]) for b in live]


def vouched(row: str, cell, exact_stack: Optional[str]) -> bool:
    """True when the cell (else the row record) lists a vouch key of ``row`` for ``exact_stack`` (see ``vouch_keys``)."""
    listed = ((cell or {}).get("vouched_on") or {}).get(row) or rows().get(row, {}).get("vouched_on") or []
    return any(k in listed for k in vouch_keys(row, exact_stack))


def exact_floor_for(cell, stack: Optional[str]) -> Optional[dict]:
    """The cell's `exact_floor` record whose torch build matches `stack` (an ABI key 'torch2.7.1+cu128-cpython-...', a vouch key
    '9.0|torch2.7.1+cu128|cueq0.10.0' or a table stack word), else None."""
    ef = (cell or {}).get("exact_floor") or {}
    if not ef or not stack:
        return None
    st = str(stack)
    for build, rec in ef.items():
        b = str(build).replace("torch", "")
        if re.search(r"(?<![0-9.])" + re.escape(b) + r"(?![0-9])", st.replace("torch", "")):
            out = dict(rec); out["stack"] = build
            return out
    return None


def cell_peaks(cell) -> Dict[str, float]:
    """{row: peak MiB} recorded for a cell, row names without a package version ('triattn_native@v11' -> 'triattn_native'): the op-level `peak_mib`
    record (`rows`, or the largest measured size under `at`); empty when the cell carries no per-row peaks that include the stock op."""
    pm = (cell or {}).get("peak_mib") or {}
    rows_ = pm.get("rows")
    if not rows_ and isinstance(pm.get("at"), dict) and pm["at"]:
        size = max(pm["at"], key=lambda s_: int(str(s_).split("/")[0]) if str(s_).split("/")[0].isdigit() else -1)
        rows_ = pm["at"][size]
    out: Dict[str, float] = {}
    for name, v in (rows_ or {}).items():
        if not isinstance(v, (int, float)) or "(" in str(name):          # 'cueq(0.10.0)': another library version's record, not a row
            continue
        base = str(name).split("@")[0].split(":")[0]
        out[base] = min(float(v), out.get(base, float("inf")))
    return out


def peak_exceeds_stock(cell, row: str) -> Optional[str]:
    """big's rule (the table decides): '<row peak> MiB > <ref> <ref peak> MiB' when the cell's recorded peak of `row` exceeds the REFERENCE
    peak -- the larger of the stock row's and the portable Triton row k2b's recorded peaks (the no-regression floor) -- by more than
    max(PEAK_NOISE_MIB, PEAK_NOISE_FRAC x reference); None when it does not or when the row's or the stock op's peak is not recorded."""
    peaks = cell_peaks(cell)
    stock = _stock_of(cell)
    sp = peaks.get(stock, peaks.get("cueq"))
    rp = peaks.get(str(row).split("@")[0])
    if sp is None or rp is None:
        return None
    ref, refname = sp, stock
    kp = peaks.get("k2b")                                    # the no-regression floor: big never steps below the portable Triton row a memory-tier mode already runs --
    if kp is not None and kp > ref:                          # a row is dropped only when it costs more than BOTH the library op and k2b (chunked pair-row calls make the
        ref, refname = kp, "k2b"                             # library op's peak the smaller of the two; square calls the larger -- the reference is whichever is larger)
    if rp > ref + max(PEAK_NOISE_MIB, PEAK_NOISE_FRAC * ref):
        return f"{rp:g} MiB > {refname} {ref:g} MiB"
    return None


def measured_ccs() -> List[str]:
    """The compute capabilities that have a measured column in the table ('8.0', '9.0', ...), ascending."""
    return sorted({k.split("|")[0] for k in cells()}, key=lambda w: tuple(int(x) for x in w.split(".")))


def group_measured(ccw: str, dtype, head_dim: int, heads: int) -> bool:
    """Does the column ``ccw`` hold at least one measured cell of this (dtype, head_dim, heads) key (any size)?"""
    prefix = f"{ccw}|{_cell_dtype(norm_dtype(dtype))}|D{int(head_dim)}|H{int(heads)}|"
    return any(k.startswith(prefix) for k in cells())


def donor_cc(cc, dtype=None, head_dim: Optional[int] = None, heads: Optional[int] = None) -> Optional[str]:
    """The measured column a call inherits from when ITS cc has no cell for the key: among the OTHER columns that measured the (dtype,
    head_dim, heads) key (any column when no key is given), the highest cc not above the device's (10.3 -> 9.0, 8.6 -> 8.0), else the lowest.
    Decided per key, never per cc: a cc owning cells for other keys still inherits for the keys it lacks."""
    M, m = norm_cc(cc); me = cc_word(cc)
    cols = [c for c in measured_ccs() if c != me and (dtype is None or group_measured(c, dtype, head_dim, heads))]
    below = [c for c in cols if tuple(int(x) for x in c.split(".")) <= (M, m)]
    return below[-1] if below else (cols[0] if cols else None)


INHERITED_CC = "inherited_cc:unmeasured"        # the census / reason token of a tier word served on a cc without a measured column
_DEAD_INHERITED: Dict[Tuple[str, str], str] = {}   # (cc word, row) -> exception type name: an INHERITED portable row that failed to build / compile / launch on
                                                 # this device once -- stepped aside for the rest of the process (never retried, never a raw error out of the face)


def inherited_row_dead(cc, row: str) -> Optional[str]:
    """The exception type name that killed an inherited portable row on this cc in this process, or None."""
    return _DEAD_INHERITED.get((cc_word(cc), str(row).split("@")[0]))


def _inherit_unmeasured_cc(base, word, args, stack, position_stride, form, prefer) -> "Selection":
    """Tier words on a cc WITHOUT a measured column: fast / big = the donor column's order (form / layout aware) restricted to the portable
    Triton rows (k2b / k2 / flash: source-compiled on this device, the row's own launch cell -- never the donor's tuned launch cell, never an
    sm_90a cubin / extension or another arch's prebuilt, never the library op while a portable row admits the call); exact = the library op by
    name (an exact-class row is vouched per cc and this cc has no vouch).  measured=False; the reason and the census carry INHERITED_CC."""
    cc, dtype, head_dim, heads, n_tokens, direction = args
    ccw, donor = cc_word(cc), donor_cc(cc, dtype, head_dim, heads)
    dkey, dcell, _dm, dnote = cell_for(donor, dtype, head_dim, heads, n_tokens, direction) if donor else (None, None, False, "")
    partial = ccw in measured_ccs()                              # the cc owns cells for OTHER keys: inheritance is decided per (dtype, head_dim, heads) key, never per cc
    tag = (f"{INHERITED_CC} (cc {ccw} has no measured cell for {_cell_dtype(norm_dtype(dtype))} D{int(head_dim)} H{int(heads)}" + ("; partial_cc: it has cells for other keys" if partial else "; no column at all")
           + f"; from the {donor} column" + (f", cell {dkey}" if dkey else "") + ")")
    if base == "exact":
        stock = _stock_of(dcell) if dcell else "cueq"
        return _selection(stock, word, dkey, dcell, None, None, False, args,
                          f"tier exact; {tag}: no exact-class row vouched on cc {ccw} -> the library op {stock} by name")
    order: List[str] = []
    if dcell:
        order = list(dcell.get("fast_order") or [])
        strided = position_stride not in (None, int(head_dim))
        fr = (dcell.get("forms") or {}).get(form) if form else None
        if isinstance(fr, dict):
            lay = fr.get("strided" if strided else "contiguous") or {}
            order = list(lay.get("fast_order") or order)
        elif strided and isinstance(dcell.get("strided"), dict):
            order = list(dcell["strided"].get("fast_order") or order)
    if prefer:
        order = [str(r) for r in prefer] + order
    cands: List[Tuple[str, Optional[str]]] = []
    skipped: List[str] = []
    for name in order + list(TRITON_ROWS):
        rname = str(name).split("@")[0]; ipx = None
        if ":" in rname:
            rname, ipx = rname.split(":", 1)                  # 'flash:tf32x3' (an fp32 cell's precision word)
            if ipx not in FP32_WORDS:
                ipx = None
        if rname in TRITON_ROWS:
            if (rname, ipx) not in cands:
                cands.append((rname, ipx))
        elif rname not in STOCK_ROWS and rname in ROW_NAMES and rname not in skipped:
            skipped.append(rname)                            # arch-specific rows: never inherited across arch
    passed: List[str] = []
    for rname, ipx in cands:
        dead = _DEAD_INHERITED.get((ccw, rname))
        if dead:
            passed.append(f"{rname}:stepped_aside:error:{dead}"); continue          # failed to build / launch on this device earlier in the process: never retried
        ok, why = admits(rname, *args, stack=stack, position_stride=position_stride)
        if not ok:
            passed.append(f"{rname}:{why}"); continue
        if base == "big" and dcell is not None:
            pk = peak_exceeds_stock(dcell, rname)
            if pk:
                passed.append(f"{rname}:peak:{pk}"); continue
        why_ = (f"tier {base}; {tag}: portable Triton rows only -> {rname} (the row's own launch cell on this device)"
                + (f"; arch-specific rows {skipped} are never inherited across arch" if skipped else "")
                + (f"; passed over {passed}" if passed else "") + (f"; {dnote}" if dnote else ""))
        return _selection(rname, word, dkey, dcell, None, ipx, False, args, why_)
    raise Refusal(f"no_cell:cc={ccw}", base, "cueq", f"{tag}: no portable Triton row admits this call ({passed})")


def _select_uncached(cc, dtype, head_dim: int, heads: int, n_tokens: int, direction=FWD, *, word: str,
           prefer: Optional[Sequence[str]] = None, stack: Optional[str] = None,
           config: Optional[Dict[str, int]] = None, position_stride: Optional[int] = None, form: Optional[str] = None,
           exact_stack: Optional[str] = None, lib: Optional[str] = None) -> Selection:
    """The row serving this call for ``word`` (see the module docstring).  Raises :class:`Refusal` (by name, with the fallback row) when the
    asked row -- or every candidate of a tier word -- cannot serve.  ``stack``: the cuda_sm90a stack key of this process (``cuda_sm90a.
    stack_key()``) so a missing prebuilt is refused here rather than at load; ``config``: an explicit Triton launch setting (passed through);
    ``position_stride``: the q / k / v position stride the kit passes (default contiguous = ``head_dim``) -- the Triton rows refuse
    ``int32_offset:...`` by name past their 32-bit in-row offset bound (``position_stride_bound``), and a tier word moves on to the next row."""
    if not isinstance(word, str) or not word:
        raise TypeError("triattn.select: word= is required (a row name, 'fast' | 'exact', or an fp32 precision word)")
    dt, d = norm_dtype(dtype), norm_direction(direction)
    args = (cc, dtype, head_dim, heads, n_tokens, direction)
    key, cell, measured, note = cell_for(*args)
    ip: Optional[str] = None
    w = word
    if w in FP32_WORDS:
        if dt != "fp32":
            raise Refusal(f"fp32_word_on_{dt}", w, None, "tf32 / tf32x3 / ieee apply to fp32 inputs only")
        ip = w
        fp = (cell or {}).get("fp32_words", {}).get(w, {})
        w = "flash" if w in ("tf32x3", "ieee") else str(fp.get("row", "k2b"))
        if w not in ROW_NAMES:
            w = "k2b"
    base, cfg, cfg_note = _config_word(w, cc) if w not in TIER_WORDS else (w, None, "")
    if config is not None:
        cfg = dict(config)
    if base in ROW_NAMES:
        ok, why = admits(base, *args, stack=stack, position_stride=position_stride)
        if not ok:
            same = TRITON_ROWS if why.startswith("int32_offset") else ()
            raise Refusal(why, base, _fallback_for(base, cell, args, stack, position_stride, exclude=same),
                          "32-bit in-row offsets: pass contiguous [B, N, H, S, D] operands (position stride = head_dim) or bind the fallback" if same else "")
        if exact_stack is not None and rows().get(base, {}).get("class") == "exact":
            if not vouched(base, cell, exact_stack):
                listed = ((cell or {}).get("vouched_on") or {}).get(base) or rows()[base].get("vouched_on") or []
                raise Refusal(f"exact_vouch_not_recorded:{exact_stack}", base, _stock_of(cell), f"vouched on {listed}")
        return _selection(base, word, key, cell, cfg, ip if base == "flash" else None, measured, args,
                          _reason(base, cell, key, note, "named row" + (f"; setting {w}: {cfg_note[:80]}" if cfg_note else "")))
    if base not in TIER_WORDS:
        raise Refusal(f"unknown_word:{word}", str(word), None,
                      f"words: rows {ROW_NAMES}, tiers {TIER_WORDS}, fp32 {FP32_WORDS}, settings {tuple(table()['words']['config_words'])}")
    # ---- an UNMEASURED compute capability (no column of this cc in the table, e.g. 10.0 / 10.3 / 12.0): fast / big INHERIT the nearest
    #      measured column's portable rows (the source-compiled Triton kernels), exact = the library op by name (no exact-class vouch on this cc);
    #      arch-specific rows (sm_90a cubins / extensions, another arch's prebuilts) are never inherited across arch.  Decided PER KEY
    #      (dtype, head_dim, heads): a cc that owns cells for other keys still inherits for the keys it lacks (never stranded to the library op)
    if not group_measured(cc_word(cc), dtype, head_dim, heads) and donor_cc(cc, dtype, head_dim, heads) is not None:
        return _inherit_unmeasured_cc(base, word, args, stack, position_stride, form, prefer)      # per KEY: this cc has no cell for (dtype, head_dim, heads) and another column measured it
    # ---- tier words serve measured cells only: the nearest cell is a SIZE neighbour; another head count or the fp16 dtype is a named refusal
    for part in str(note).split("; "):
        if part.startswith("heads ") and " not measured" in part:
            raise Refusal(f"no_cell:heads={int(heads)}", base, _stock_of(cell),
                          f"tier words serve measured (cc, dtype, head_dim, heads) cells only ({part}); a row word opts in at this head count")
        if part.startswith("fp16 takes"):
            raise Refusal("no_cell:dtype=fp16", base, _stock_of(cell),
                          "tier words serve measured dtypes only (no fp16 cell); a row word opts in for fp16")
    # ---- tier words: the measured winner (OPT-IN)
    no_row = None
    if cell is None and not prefer:                                           # no measured cell of this (cc, dtype, head_dim) at any head count / size
        whys = {r: admits(r, *args, stack=stack, position_stride=position_stride) for r in KERNEL_ROWS}
        admitted = [r for r, (ok_, _w) in whys.items() if ok_]
        if admitted:                                                          # kernel rows could serve it, but nothing was measured: a named refusal, the stock op serves (a row word opts in)
            raise Refusal(f"no_cell:{_cell_dtype(norm_dtype(dtype))}_D{int(head_dim)}", base, "cueq",
                          f"tier words serve measured cells only; rows {admitted} admit this call by name (opt in with the row word)")
        hd = all("head_dim" in _w for ok_, _w in whys.values())              # every kernel row names the head_dim: no row of this family carries it (structural) -- the stock op serves, BY NAME
        no_row = f"no_row:head_dim{int(head_dim)}" if hd else "no_row:" + "|".join(sorted({_w for ok_, _w in whys.values()}))
    if base in ("fast", "big"):
        strided = position_stride is not None and int(position_stride) != int(head_dim) and cell is not None and "strided" in cell
        formrec = (cell or {}).get("forms", {}).get(form) if form else None                     # a call form measured on its own (bias_only / keypad / mask_bias)
        formrec = (formrec or {}).get("strided" if strided else "contiguous") if formrec else None
        cands = list(prefer) if prefer else list((formrec if formrec else cell["strided"] if strided else (cell or {})).get("fast_order", ["k2b", "cueq"]))   # transposed views: the cell's strided order where measured
        libmap = (cell or {}).get("fast_by_lib") or {}
        if lib and not prefer and not formrec and not strided and lib in libmap:      # two measured columns of this cc disagree by STOCK-LIBRARY version: the process's own library decides
            cands = list(libmap[lib].get("fast_order") or cands)
            note = (note + "; " if note else "") + f"fast_by_lib[{lib}] ({libmap[lib].get('column', '?')})"
    else:
        cands = [r for r in (list(prefer) if prefer else [(cell or {}).get("exact", "cueq"), "cueq"])
                 if r in STOCK_ROWS or rows().get(r, {}).get("class") == "exact"]
        floor = exact_floor_for(cell, exact_stack or stack)
        if floor and int(n_tokens) < int(floor["below_tokens"]) and not prefer:     # a per-stack exact floor recorded in the cell: below it the stock op IS the exact tier on that stack
            cands = [floor.get("row") or _stock_of(cell)]
            note = (note + "; " if note else "") + f"exact floor on {floor['stack']} below {floor['below_tokens']} tokens: {cands[0]}"
        if not cands:
            raise Refusal("prefer_has_no_exact_row", "exact", (cell or {}).get("exact", "cueq"), f"prefer={tuple(prefer or ())}")
    refused = []
    for rw in cands:
        r = rw
        if isinstance(rw, str) and "@" in rw:               # a versioned package name in the table (triattn_native@v11 / @v10): the row while that version is active or honoured on this cc
            try:
                r = _config_word(rw, cc)[0]
            except Refusal as e_:
                refused.append(f"{rw}:{e_.kind}")
                continue
        if r not in ROW_NAMES:
            refused.append(f"{r}:unknown_row")
            continue
        ok, why = admits(r, *args, stack=stack, position_stride=position_stride)
        if ok and base == "big" and cell is not None and r not in STOCK_ROWS:      # big = fast minus rows with a MEASURED memory cost above the stock op's (the table's peak records decide)
            over = peak_exceeds_stock(cell, r)
            if over:
                refused.append(f"{r}:peak:{over}")
                continue
        if ok and exact_stack is not None and rows().get(r, {}).get("class") == "exact":
            if not vouched(r, cell, exact_stack):                # an exact vouch holds only on the stack it was measured on: the tier falls to the stock op (exact by definition)
                refused.append(f"{r}:exact_vouch_not_recorded:{exact_stack}")
                continue
        if ok:
            passed = refused + ([no_row] if no_row else [])             # a structural no-row word rides last (the per-row refusals keep their place)
            why_this = f"tier {base}" + (f" among prefer={tuple(prefer)}" if prefer else "") + (f"; passed over {passed}" if passed else "") + (f"; {why}" if str(why).startswith("prebuilt: any-of") else "") + (f"; exact vouch on {exact_stack}" if exact_stack and rows().get(r, {}).get("class") == "exact" else "")
            if cfg is None and cell is not None:                 # a launch cell measured for THIS cell (opt-in: reached through the tier word, never the row word)
                tuned = (cell.get("tuned") or {}).get(r)
                if tuned and int(args[4]) >= int(tuned.get("n_floor", 0)):
                    cfg = dict(tuned["config"]); note = (note + "; " if note else "") + f"tuned launch cell {tuned.get('config_word', '')} (measured for this cell; n_floor {tuned.get('n_floor', 0)})"
            return _selection(r, word, key, cell, cfg, None, measured, args, _reason(r, cell, key, note, why_this))
        refused.append(f"{r}:{why}")
    raise Refusal("no_candidate_admitted", base, "cueq", "; ".join(refused))


def _selection(row, word, key, cell, cfg, ip, measured, args, reason) -> Selection:
    R = rows()[row]
    x = None
    if cell and row not in STOCK_ROWS:
        xr = row if ip in (None, "tf32") else f"flash:{ip}"
        x = cell.get("x_stock", {}).get(xr)
        if x is None and row == "triattn_native":
            from . import triattn_native as _CC
            x = cell.get("x_stock", {}).get(f"triattn_native@{_CC.ACTIVE_PKG}")
            if x is None:                                     # measured on a generation the active payload honours on this cc (v11 = v10's outputs on 9.0): its number stands
                for ver, on in _CC.HONOURED.get(_CC.ACTIVE_PKG, {}).items():
                    if tuple(norm_cc(key.split("|")[0])) in on and f"triattn_native@{ver}" in cell.get("x_stock", {}):
                        x = cell["x_stock"][f"triattn_native@{ver}"]; break
    tested = True
    if row == "exact_headsplit":
        tested = cc_word(args[0]) in R.get("tested_cc", [])
    backward = R.get("backward")
    exact_vs = (str(R.get("exact_vs", "")).split(" ")[0] or None) if R["class"] == "exact" else None
    return Selection(row=row, word=word, cell=key, config=cfg, input_precision=ip, cls=R["class"], exact_vs=exact_vs,
                     capture_safe=bool(R.get("capture_safe", True)),
                     backward=backward is True or (isinstance(backward, str) and row == "exact_headsplit"),
                     measured=bool(measured and cell is not None and (row in STOCK_ROWS or x is not None)),
                     x_stock=x, tested=tested, reason=reason)


def _reason(row, cell, key, note, why) -> str:
    parts = [why]
    if key:
        parts.append(f"cell {key}: fast={cell.get('fast')} exact={cell.get('exact')}")
        for role, text in (cell.get("parity") or {}).items():        # a named winner inside the measurement-noise band says so
            if text.split(" ")[0] == row:
                parts.append(f"parity ({role}): {text}")
    if note:
        parts.append(note)
    return "; ".join(parts)


def describe(sel: Selection) -> str:
    """One line for a kit's LEVER / census line."""
    bits = [f"row={sel.row}", f"word={sel.word}", f"class={sel.cls}"]
    if sel.exact_vs:
        bits.append(f"exact_vs={sel.exact_vs}")
    if sel.input_precision:
        bits.append(f"ip={sel.input_precision}")
    if sel.config:
        bits.append("config=" + ",".join(f"{k}{v}" for k, v in sel.config.items()))
    bits.append(f"cell={sel.cell or '-'}")
    bits.append(f"measured={'yes' if sel.measured else 'no'}")
    if sel.x_stock is not None:
        bits.append(f"x_stock={sel.x_stock}")
    if not sel.capture_safe:
        bits.append("capture_safe=NO")
    if not sel.tested:
        bits.append("tested_cc=NO")
    return " ".join(bits)


def coverage(cc, shapes: Sequence[Tuple[str, int, int]], sizes: Sequence[int] = (400, 800, 1200)) -> Dict[str, str]:
    """{'<cc>|<dtype>|D<d>|H<h>|N<n>': '<cell key> fast=<row> exact=<row>' | 'n/a: <why>'} for a list of (dtype, head_dim, heads) shapes --
    the census a kit (or a test) prints to see every shape it runs land on a measured cell or a named n/a."""
    out = {}
    for dt, D, H in shapes:
        for n in sizes:
            key, cell, measured, note = cell_for(cc, dt, D, H, n)
            tag = f"{cc_word(cc)}|{dt}|D{D}|H{H}|N{n}"
            if cell is None:
                out[tag] = f"n/a: {note}"
            else:
                out[tag] = f"{key} fast={cell['fast']} exact={cell['exact']}" + ("" if measured else f" ({note})")
    return out


# ----------------------------------------------------------------------------------------------------------------- serving (torch at call time)
_HS_WRAPPED: Dict[int, Any] = {}
_READY: Dict[str, Any] = {}
_SEL_CACHE: Dict[tuple, Selection] = {}          # (tensor facts, word, prefer, stack, config) -> Selection: the table walk runs once per distinct call shape


def _tensor_facts(q, k):
    import torch
    dt = q.dtype
    if torch.is_autocast_enabled():
        dt = torch.get_autocast_dtype("cuda")
    dtype = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}.get(dt, str(dt))
    cc = torch.cuda.get_device_capability(q.device) if q.is_cuda else (0, 0)
    heads = int(q.shape[-3]) if q.dim() >= 3 else 1
    direction = FWDBWD if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad) else FWD
    return cc, dtype, int(q.shape[-1]), heads, int(k.shape[-2]), direction


def _pos_stride(x) -> int:
    """Position stride of a q / k / v operand as the Triton rows will see it (they copy an operand whose last stride is not 1 to contiguous)."""
    return int(x.shape[-1]) if x.stride(-1) != 1 else int(x.stride(-2))


def _tensor_offset_terms(q, k, v, bias, mask) -> Dict[str, int]:
    n_q, n_k, d = int(q.shape[-2]), int(k.shape[-2]), int(q.shape[-1])
    mks = int(mask.stride(-1)) if (mask is not None and hasattr(mask, "stride") and mask.dim() >= 1 and mask.shape[-1] == n_k) else 0
    bq = int(bias.stride(-2)) if bias.dim() >= 2 else 0
    bk = int(bias.stride(-1)) if bias.dim() >= 1 else 0
    return int32_offset_terms(n_q, n_k, d, q_pos_stride=_pos_stride(q), k_pos_stride=_pos_stride(k), v_pos_stride=_pos_stride(v),
                              mask_key_stride=mks, bias_q_stride=bq, bias_k_stride=bk)


def _stock_of(cell) -> str:
    """The stock-op row terminating a cell's order (cueq unless the cell was measured against another stock op)."""
    order = (cell or {}).get("fast_order") or []
    return order[-1] if order and order[-1] in STOCK_ROWS else "cueq"


def exact_stack_key(cc=None) -> str:
    """'<cc>|torch<version>|cueq<version>' of this process: the key an exact-class vouch is recorded under (words.exact_vouch)."""
    import sys as _sys
    try:                                                     # a stubbed / absent torch or no CUDA device: DEGRADE BY NAME -- an unknown key matches no vouch, so the
        import torch                                         # exact tier names the library op (never raises out of the select path)
        M, m = norm_cc(cc if cc is not None else tuple(torch.cuda.get_device_capability()))
        tv = str(torch.__version__)
    except (ImportError, AttributeError, RuntimeError, AssertionError, TypeError, ValueError):   # a probe failure (a stub, no device, no torch): the unknown key
        try:
            M, m = norm_cc(cc) if cc is not None else ("?", "?")
        except (AttributeError, TypeError, ValueError):
            M, m = "?", "?"
        tv = "?"
    lib = getattr(_sys.modules.get("cuequivariance_torch"), "__version__", None)
    if lib is None:
        try:
            from importlib import metadata as _md
            lib = _md.version("cuequivariance-torch")
        except (ImportError, LookupError, ValueError):           # library absent: no exact row is vouched against it here
            lib = "none"
    tag = device_tag(f"{M}.{m}") if tv != "?" else ""
    return f"{M}.{m}|torch{tv}|cueq{lib}" + (f"|{tag}" if tag else "")


def stock_lib_key() -> Optional[str]:
    """'cueq<version>' of the stock library importable in this process (the key of a cell's `fast_by_lib`), None when absent."""
    import sys as _sys
    lib = getattr(_sys.modules.get("cuequivariance_torch"), "__version__", None)
    if lib is None:
        try:
            from importlib import metadata as _md
            lib = _md.version("cuequivariance-torch")
        except (ImportError, LookupError, ValueError):
            return None
    return f"cueq{lib}"


def device_tag(ccw: Optional[str] = None, name: Optional[str] = None) -> str:
    """The DEVICE TAG of an exact vouch key (words.exact_vouch.device_tags): '' on the column's reference device (the device the untagged vouches
    were measured on) or without a CUDA device; the table's tag for a listed device ('H200', 'A100-40GB', ...); else the device name sanitized
    (matches no vouch: an exact-class row serves on a device only where it was measured bitwise there)."""
    dt = (table().get("words", {}).get("exact_vouch") or {}).get("device_tags") or {}
    if name is None:
        try:
            import torch
            if not torch.cuda.is_available():
                return ""
            name = str(torch.cuda.get_device_name())
        except (ImportError, AttributeError, RuntimeError, AssertionError, TypeError, ValueError):   # a stub / no device: no tag (the untagged key; CPU-side selection is unchanged)
            return ""
    refs = dt.get("reference_devices") or {}
    if ccw is not None and name in (refs.get(str(ccw)) or []):
        return ""
    if ccw is None and any(name in v for v in refs.values()):
        return ""
    tags = dt.get("tags") or {}
    if name in tags:
        return str(tags[name])
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", name).strip("_")[:40] or "device"


def _route(name: str):
    """A carried Triton kernel through the core's route (the module object, exports and byte check every pair_fused binder gets)."""
    from opt_core.attn.pair_fused import _carried
    return _carried(name)


def warm(rows: Sequence[str] = ("triattn_native",), shapes: Sequence[Tuple[int, int, int]] = (), stock=None) -> Dict[str, Any]:
    """Activation-time hook for kits (call it when the mode engages the rows, before the first model call): per row, load / check / byte-gate
    what the row needs (``triattn_native``: payload import + digests + byte gate or its disk-cached verdict; ``cuda_sm90a``: extension load +
    load-check) and optionally serve one call per (H, S, D) in ``shapes`` so JIT / first-launch
    costs land here.  Returns {row: {"ready": bool, "seconds": float, "report" | "refused": ...}} -- a refused row is reported by name, never
    raised (the kit binds the fallback as it would at the first call)."""
    import time as _time
    out: Dict[str, Any] = {}
    for row in rows:
        t0 = _time.perf_counter(); rec: Dict[str, Any] = {}
        try:
            if row.split("@")[0] == "triattn_native":
                from . import triattn_native as _CC
                rec["report"] = _CC.warm(tuple(shapes))
            elif row == "cuda_sm90a":
                from . import cuda_sm90a as _C
                if "cuda_sm90a" not in _READY:
                    _READY["cuda_sm90a"] = _C.install()
                rec["report"] = {k_: _READY["cuda_sm90a"].get(k_) for k_ in ("route", "prebuilt", "stack_key")}
            elif row == "triattn_exact":
                from . import exact_member as _EM
                if "triattn_exact" not in _READY:
                    _READY["triattn_exact"] = _EM.install()
                rec["report"] = {k_: _READY["triattn_exact"].get(k_) for k_ in ("probe", "route", "backend", "prepare_s", "cache_dir", "proven_cells")}
            elif row in TRITON_ROWS and shapes:
                import torch
                for (H, S, D) in shapes:
                    g = torch.Generator(device="cuda").manual_seed(S)
                    q, k, v = (torch.randn((1, min(S, 64), H, S, D), device="cuda", dtype=torch.bfloat16, generator=g) for _ in range(3))
                    bias = torch.randn((1, 1, H, S, S), device="cuda", dtype=torch.bfloat16, generator=g).float()
                    triangle_attention(q, k, v, bias, None, word=row); torch.cuda.synchronize()
                rec["report"] = {"jit": "warm"}
            else:
                rec["report"] = {"nothing_to_warm": True}
            rec["ready"] = True
        except Refusal as r:
            rec["ready"] = False; rec["refused"] = {"kind": r.kind, "fallback": r.fallback}
        except Exception as e:                                       # a carried package's own load error: reported by name here, raised by name at the call
            from opt_core.oom import is_oom
            if is_oom(e): raise
            rec["ready"] = False; rec["refused"] = {"kind": type(e).__name__, "detail": str(e)[:160]}
        rec["seconds"] = round(_time.perf_counter() - t0, 3); out[row] = rec
    return out


def _launch_triton(row: str, sel: "Selection", q, k, v, bias, mask, sc):
    """One of the portable Triton rows (k2b / k2 / flash) with the selection's launch cell / precision word."""
    kw = {"config": dict(sel.config)} if sel.config else {}
    if row == "k2b":
        return _route("fpf_triatt_k2b").attn_k2b(q, k, v, bias, mask=mask, scale=sc, **kw)
    if row == "k2":
        return _route("fpf_triatt_k2b").attn_k2(q, k, v, bias, mask=mask, scale=sc, **kw)
    if sel.input_precision:
        kw["input_precision"] = sel.input_precision
    return _route("flash_triattn").flash_triangle_attention(q, k, v, bias, mask=mask, scale=sc, **kw)


def _serve_inherited(sel: "Selection", q, k, v, bias, mask, sc, stock, args, word, prefer, stack, config, form):
    """Serve an INHERITED portable row (a part without a measured column) with its first build / compile / launch GUARDED: any failure that is
    not an out-of-memory marks the row dead for this (cc, process) -- census `stepped_aside:error:<ExcType>` once -- and the donor cell's NEXT
    portable row serves, else the library op by name (the kit's `stock` callable when given, else Refusal `inherited_rows_failed` with the
    library op as fallback).  Out-of-memory is re-raised as everywhere; no raw exception leaves the face for an inherited row."""
    from opt_core.oom import is_oom
    cc = args[0]; ccw = cc_word(cc)
    tried: List[str] = []
    cur = sel
    for _ in range(len(TRITON_ROWS) + 1):
        row = cur.row
        try:
            return _launch_triton(row, cur, q, k, v, bias, mask, sc)
        except Refusal:
            raise
        except Exception as e:                                     # is_oom first: memory pressure is the caller's signal, re-raised unchanged
            if is_oom(e):
                raise
            _DEAD_INHERITED[(ccw, row)] = type(e).__name__
            tried.append(f"{row}:{type(e).__name__}")
            _SELECT_MEMO.clear(); _SEL_CACHE.clear()
            try:
                _CENSUS.record("triattn", {"cc": ccw, "stack": stack or "-", "dtype": str(args[1]), "shape": f"D{int(args[2])}H{int(args[3])}", "bucket": f"N={int(args[4])}", "form": str(args[5]), "word": str(word)},
                               "named_fallback", "next_portable_row", cell_id=cur.cell, refused=f"{row}:stepped_aside:error:{type(e).__name__}", note="stepped_aside:error:%s" % type(e).__name__)
            except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):   # the census counts; it never gates a serve
                pass
        try:
            cur = _inherit_unmeasured_cc(str(word).split("@")[0] if str(word).split("@")[0] in TIER_WORDS else "fast", word, args, stack, None, form, prefer)
        except Refusal:
            break
        if cur.row not in TRITON_ROWS or cur.row in [t.split(":")[0] for t in tried]:
            break
    if stock is not None:                                          # every portable row failed here: the library op by name (the kit's stock callable)
        return stock(q, k, v, bias, mask=mask, scale=sc)
    raise Refusal("inherited_rows_failed", sel.row, "cueq", f"the inherited portable rows failed to build / launch on cc {ccw} in this process ({tried}): bind the library op")


def triangle_attention(q, k, v, bias, mask=None, scale=None, *, word: str, prefer: Optional[Sequence[str]] = None,
                       stock=None, config: Optional[Dict[str, int]] = None, selection: Optional[Selection] = None, form: Optional[str] = None):
    """Serve the op through the selected row (module docstring).  ``stock``: the kit's own stock callable (cuequivariance signature) -- required
    by the rows cueq / ds4sci / exact_headsplit.  Returns the output tensor; raises :class:`Refusal` by name before computing when the row
    cannot serve (a kit binds ``.fallback`` or makes it its hard error -- never a silent substitute)."""
    global _WARMED
    if not _WARMED:                                             # first serving call of this face in the process: the stack's heavy libraries imported early + shallow (opt_core.warm)
        _WARMED = True
        from opt_core.warm import auto_warm
        auto_warm("triattn")
    cc, dtype, D, H, S, direction = _tensor_facts(q, k)
    stack = None
    wants_cuda = (word.split("@")[0] in ("cuda_sm90a", "triattn_native")) or (prefer and any(str(p_).split("@")[0] in ("cuda_sm90a", "triattn_native") for p_ in prefer)) or (word in ("fast", "big") and not prefer)
    if wants_cuda and tuple(cc) == (9, 0):
        from . import cuda_sm90a as _C                    # torch is imported already: the module import loads nothing
        stack = _C.stack_key()
    elif wants_cuda and tuple(cc) == (8, 0):              # the sealed package's sm_80 member (generation >= 11): the package's own ABI key names the prebuilt this process needs
        from . import triattn_native as _CCk
        stack = _CCk.stack_key()
    sel = selection
    if sel is None:
        key = (form, tuple(cc), dtype, D, H, S, direction, word, tuple(prefer or ()), stack, tuple(sorted((config or {}).items())))
        sel = _SEL_CACHE.get(key)
        if sel is None:
            xst = exact_stack_key(cc) if (word.split("@")[0] == "exact" or rows().get(word.split("@")[0], {}).get("class") == "exact" or (prefer and any(rows().get(str(p_).split("@")[0], {}).get("class") == "exact" for p_ in prefer))) else None
            sel = _SEL_CACHE[key] = select(cc, dtype, D, H, S, direction, word=word, prefer=prefer, stack=stack, config=config, form=form, exact_stack=xst, lib=stock_lib_key())
    row = sel.row
    sc = (1.0 / math.sqrt(D)) if scale is None else float(scale)
    if row in TRITON_ROWS:                                  # the carried kernels' 32-bit in-row offsets: every term from the actual strides, refused by name past 2**31
        ok, why = int32_offsets_ok(_tensor_offset_terms(q, k, v, bias, mask))
        if not ok:
            args = (cc, dtype, D, H, S, direction)
            raise Refusal(why, row, _fallback_for(row, cells().get(sel.cell) if sel.cell else None, args, stack, max(_pos_stride(q), _pos_stride(k), _pos_stride(v)), exclude=TRITON_ROWS),
                          "32-bit in-row offsets in the Triton rows: pass contiguous [B, N, H, S, D] operands (a transposed view has position "
                          f"stride N*H*D; bound at {S} tokens = {position_stride_bound(S, D)}) or bind the fallback row")
    if row in NEEDS_STOCK and stock is None:
        raise Refusal("needs_stock", row, None, "pass stock=<the kit's stock triangle_attention callable>")
    if row in TRITON_ROWS and INHERITED_CC in (sel.reason or ""):     # an INHERITED portable row on a part without a column: its first build / compile / launch is GUARDED
        return _serve_inherited(sel, q, k, v, bias, mask, sc, stock, (cc, dtype, D, H, S, direction), word, prefer, stack, config, form)
    if row == "k2b":
        kw = {"config": dict(sel.config)} if sel.config else {}
        return _route("fpf_triatt_k2b").attn_k2b(q, k, v, bias, mask=mask, scale=sc, **kw)
    if row == "k2":
        kw = {"config": dict(sel.config)} if sel.config else {}
        return _route("fpf_triatt_k2b").attn_k2(q, k, v, bias, mask=mask, scale=sc, **kw)
    if row == "flash":
        kw = {}
        if sel.config:
            kw["config"] = dict(sel.config)
        if sel.input_precision:
            kw["input_precision"] = sel.input_precision
        return _route("flash_triattn").flash_triangle_attention(q, k, v, bias, mask=mask, scale=sc, **kw)
    if row == "cuda_sm90a":
        from . import cuda_sm90a as C
        if not _READY.get("cuda_sm90a"):
            try:
                _READY["cuda_sm90a"] = C.install()        # load + load-check once per process
            except C.Refused as e:                        # reroute: the named refusal of the extension's own load gate (no CUDA / load-check)
                raise Refusal("refused_at_install", row, "k2b", str(e)) from None
        return C.attn(q, k, v, bias, mask=mask, scale=sc)
    if row == "triattn_native":
        from . import triattn_native as CCm
        try:
            return CCm.triangle_attention(q, k, v, bias, mask, sc)
        except CCm.Unavailable as e:                          # by name: no prebuilt for this stack / digest / byte gate / the package's own typed refusal
            args = (cc, dtype, D, H, S, direction)
            raise Refusal(e.kind, row, _fallback_for(row, cells().get(sel.cell) if sel.cell else None, args, stack), str(e)) from None
    if row == "triattn_exact":
        from . import exact_member as EM
        if not _READY.get("triattn_exact"):
            try:
                _READY["triattn_exact"] = EM.install()        # import + build + self-check probe once per process (resolve time already did it in a serving kit)
            except Exception as e:                            # the member cannot load / build here: rerouted by name to the stock op
                from opt_core.oom import is_oom
                if is_oom(e):
                    raise
                raise Refusal("refused_at_install", row, _fallback_for(row, cells().get(sel.cell) if sel.cell else None, (cc, dtype, D, H, S, direction), stack),
                              f"{type(e).__name__}: {str(e)[:160]}") from None
        return EM.serve(q, k, v, bias, mask, sc, stock)
    if row == "exact_headsplit":
        from . import headsplit as HS
        fn = _HS_WRAPPED.get(id(stock))
        if fn is None:
            pol, _cc = HS._device_policy()
            HS.N_SPLIT = int(pol["n_split"])
            HS._STATE["device"] = pol
            fn = _HS_WRAPPED[id(stock)] = HS._make_split(stock, "tl")
        return fn(q, k, v, bias, mask=mask, scale=sc)
    if row == "sdpa":
        return sdpa_reference(q, k, v, bias, mask, sc)
    return stock(q, k, v, bias, mask=mask, scale=sc)      # cueq / ds4sci / stock: the kit's stock callable


def sdpa_reference(q, k, v, bias, mask=None, scale=None):
    """The sdpa row: torch scaled_dot_product_attention over [B*N, H, S, D] with bias + key mask folded into one additive fp32 tensor."""
    import torch
    import torch.nn.functional as F
    B, N, H, S, D = q.shape
    sc = (1.0 / math.sqrt(D)) if scale is None else float(scale)
    add = bias.to(torch.float32).expand(B, N, H, S, S)
    if mask is not None:
        keep = mask.reshape(B, N, 1, 1, S).to(torch.bool)
        add = add.masked_fill(~keep, -1e9)
    out = F.scaled_dot_product_attention(q.reshape(B * N, H, S, D), k.reshape(B * N, H, S, D), v.reshape(B * N, H, S, D),
                                         attn_mask=add.reshape(B * N, H, S, S).to(q.dtype), scale=sc)
    return out.reshape(B, N, H, S, D)


def reference(q, k, v, bias, mask=None, scale=None, dtype=None, rows_per_step: int = 32):
    """Plain evaluation of the op (fp32 by default) with the fully-masked-row convention (uniform over all keys) -- the numerics yardstick.
    Evaluated ``rows_per_step`` pair rows at a time (the [N,H,S,S] logits of a whole call do not fit a card at the larger sizes)."""
    import torch
    dtype = torch.float32 if dtype is None else dtype
    B, N, H, S, D = q.shape
    sc = (1.0 / math.sqrt(D)) if scale is None else float(scale)
    out = torch.empty(B, N, H, S, D, dtype=dtype, device=q.device)
    bias_ = bias.to(dtype)
    for n0 in range(0, N, int(rows_per_step)):
        n1 = min(N, n0 + int(rows_per_step))
        s = torch.einsum("bnhqd,bnhkd->bnhqk", q[:, n0:n1].to(dtype), k[:, n0:n1].to(dtype)) * sc + bias_
        if mask is not None:
            keep = mask[:, n0:n1].reshape(B, n1 - n0, 1, 1, S).to(torch.bool)
            dead = ~keep.any(dim=-1, keepdim=True)
            s = torch.where(keep | dead, s, torch.full_like(s, float("-inf")))
            s = torch.where(dead.expand_as(s), torch.zeros_like(s), s)          # a fully-masked row: uniform over all keys
        out[:, n0:n1] = torch.einsum("bnhqk,bnhkd->bnhqd", torch.softmax(s, dim=-1), v[:, n0:n1].to(dtype))
    return out

_WARMED = False                                                 # opt_core.warm.auto_warm() ran from this face's first serving call
