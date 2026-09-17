"""JAX-family pair-stack / MSA-stack rows: ONE provider over every carried Pallas (Triton-lowering) and pure-XLA implementation ("row") of
the seven sub-layer ops and their measured cell table (PALLAS_CELLS.json).

Ops (``OPS``) and the call faces beside this table (``opt_core.kernels.pallas.serve``, which imports the framework only when called)::

    attn        attention core: softmax_k(scale*q.k + bias[h,q,k] (shared over the batch rows) + key mask) @ v
                q/k/v [B,S,H,D] ("BSHD", the module layout) or [B,H,S,D] ("BHSD"); bias [H,Sq,Sk]; key_mask [B,Sk] bool  -> serve.attention
    triattn     triangle attention MODULE: LayerNorm -> q/k/v/bias/gate projections -> core -> gate -> output projection; act [N,N,C],
                pair mask [N,N], orientation starting | ending                                                              -> serve.triangle_attention_block
    trimul      triangle multiplication MODULE (LayerNorm -> gated left/right projections * mask -> contraction -> LayerNorm -> gated
                output projection); act [N,N,C], pair mask [N,N], equation outgoing | incoming                              -> serve.triangle_multiplication
    glut        the AF3-form contraction prologue: gated linear unit -> transpose -> mask                                   -> serve.glu_transposed_masked
    transition  LayerNorm -> W1 (+b1) -> relu | swiglu -> W2 (+b2); x [..., C]                                              -> serve.transition
    ln          LayerNorm over the channel axis; x [..., C], scale / offset [C]                                             -> serve.layer_norm
    opm         outer-product mean: (LayerNorm ->) left/right projections * mask -> outer product over the sequence axis ->
                output projection -> / (eps + mask norm); act [S,N,C_m], mask [S,N]                                         -> serve.outer_product_mean

Rows (``ROW_NAMES``; the table's ``rows`` block states each one's module, numerics class, jax lines, cards, dtypes, head dims / channels,
backward, settings, named fallback and refusal words).  Carried sub-packages (byte-identical copies, provenance in each NOTICE):
``cd_trimul/`` (triangle multiplication forward + backward),
``cd_layers/`` (LayerNorm, ReLU transition, outer-product-mean fold), ``rowshared_flash_pallas.py`` (row-shared pair-bias attention forward),
``mlp_transition/`` (fused ReLU / SwiGLU transition forward), ``opm_pallas/`` (outer-product-mean rows).  Rows served from the modules beside
this package: ``fpf_block`` / ``fpf_trimul`` / ``fpf_core`` / ``fpf_transition`` (kernels/fpf_pallas*, through fpf_pallas_serve),
``pallas_attn`` (kernels/pallas_attn, through pallas_attn_serve), ``cd_triatt`` (kernels/pallas_triatt: the triangle-attention
core forward + two-kernel backward), ``glut`` (kernels/pallas_glut), ``triattn_xla`` (kernels/triattn_xla, the
XLA-FFI bridge over pre-compiled kernels: the preferred triangle-attention forward row where the running jax has jax.ffi).  Stock rows name the
framework's own ops: ``xla`` (the program's statement, served by serve.reference_*), ``xla_sdpa`` / ``cudnn`` (jax.nn.dot_product_attention),
``tokamax`` / ``tokamax_glu`` / ``tokamax_glu_T`` (the tokamax library, where importable), ``xla_subbatch4`` (an engine module form, timed
only).

Selection -- pure Python, no framework import::

    sel = select(jax_line, cc, dtype, op, family, n_tokens, direction="fwd", *, word, prefer=None)
    sel.row / sel.arm / sel.config / sel.input_precision / sel.candidates / sel.cell_key / sel.cell / sel.cls / sel.x_stock / sel.note

  jax_line    the running jax version ("0.10.2") or its line ("0.10"); ``jax_line_of(version)``.  A cell dimension: the same kernel
              source lowers and times differently per jax / Triton line (``LINES``).
  cc          compute capability "9.0" | "8.0" | ... (cells measured on 9.0 = H100 and 8.0 = A100-80GB).
  dtype       "bf16" | "f32" (aliases bfloat16 / float32 / fp32): the activations'.
  op, family  ``OPS``; ``families(op)`` lists the measured shape words, ``family(op, **shape)`` builds one from shape facts
              (kind / form / unit, heads, head_dim, c, c_hidden, factor, n_seq, orientation / equation / activation) with the
              n_seq rule (the nearest measured MSA depth at or above the call's, else the deepest).
  n_tokens    the token count: sizes = the smallest measured size at or above it ("N<=" buckets), above the largest = the largest.
  direction   "fwd" (inference) | "fwdbwd" (the call is differentiated): forward-only rows refuse a fwdbwd call BY NAME
              (``no_backward``) and, served through serve.*, raise the same refusal if differentiated anyway.
  word        WHICH row, and it is the switch (every measured winner is OPT-IN; nothing a kit runs today changes underneath it):
                * a row name              -> exactly that row with the row's own defaults = what a kit binding that module today runs;
                * an arm '<row>@<setting>' / '<row>:<f32 word>' -> that row at that measured setting / operand precision (``SETTING_WORDS``,
                                             ``F32_WORDS``);
                * "fast" | "exact" | "big" (``TIER_WORDS``) -> the cell's measured winner of that tier: fast = first arm of the cell's
                                             fast_order the call admits (prefer=(row, ...) narrows to the kit's rows), exact = an arm
                                             bitwise the stock op AND >= x1.00 else 'xla' BY NAME (most Pallas rows are tolerance
                                             class: exact_rule), big = fast's order minus arms with a measured memory cost (big_rule);
                * "tf32" | "tf32x3" | "ieee" -> f32 activations: the cell's fast arm at that operand precision.
              A tier word outside every measured cell names the stock statement ('xla') with a note -- never a guess.
  prefer      restrict tier words to these rows (a kit passes the rows its import can bind), matched in the cell's measured order.

Refusals are BY NAME: ``Refusal`` (RuntimeError) carries ``kind`` (``REFUSAL_KINDS``), ``row`` and ``fallback`` (the row the table names
instead); select() raises the static ones (no_backward, cudnn_shared_bias_grad, launch_fails_sm80, head_dim_not_pow2,
bias_per_window, c_not_served, channels_not_pow2, levers_off, the cell's measured 'refused' words), serve.* the run-time ones
(needs_tokamax, needs_jax_ffi, needs_haiku, the carried modules' own).  serve.* called with a tier word walks the cell's candidates past a
refusing row and records the refusal (serve.COUNTS / serve.LAST_REFUSAL): a row that cannot engage steps aside by name, the next measured
arm serves; ``strict=True`` raises instead.

Default rule (``default_rule``): a row word serves exactly that row at its own defaults; measured winners and tuned launch settings are
reached only through tier words or arm words.  ``MODEL_OPT_LEVERS_OFF`` words ``pallas`` (every non-stock row) and ``pallas:<row>`` switch
rows off: a switched-off row is refused by name (``levers_off``) naming its fallback.

Head-dim rule: attention rows on a head_dim below 16 (the extra-MSA stack's 8) are served ZERO-PADDED to 16 by the face (``PAD_HEAD_DIM_TO``;
an unpadded 8 returns wrong numbers from the Pallas kernels -- measured); head dims that are not powers of two (24, 48) are refused by name.
cuDNN rule: the fused cuDNN attention returns a wrong gradient for a bias shared over the batch rows: rows ``cudnn`` / ``proj+cudnn`` /
``tokamax@cudnn`` are refused by name for direction fwdbwd when the call carries a key mask (``cudnn_shared_bias_grad``: cuDNN's VJP
returns gradients 40-57x the reference scale from fully-masked query rows, measured); ``select(..., key_masked=False)`` / an unmasked
``serve.attention`` call serves them differentiated (cosine >= 0.9999 measured).  A100 f32 triangle multiplication: ``cd_trimul``'s
module tiles run x0.21-0.31 of the stock op at c_hidden >= 128 on cc 8.x (the row word keeps them: default rule); the tuned word
``cd_trimul@t32w8`` runs x2.9-4.2 there (a cell arm behind ``fpf_trimul:tf32``); its f32 backward on the jax 0.5 line fails to launch
there -> refused by name (``launch_fails_sm80``) -> ``fpf_trimul_xlabwd:tf32``.  ``cd_transition`` on cc 8.x at C >= 128 launches with
the face's tile-table entry (``serve.SM80_TRANSITION_TILE``).  Bridge rule (``bridge``): see PALLAS_CELLS.json "bridge".
"""
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ... import cell_census as _CENSUS                # stdlib-only: the coverage census (one record per decided call class; pure observation)

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS_FILE = os.path.join(HERE, "PALLAS_CELLS.json")

with open(CELLS_FILE, encoding="utf-8") as _f:
    TABLE: Dict[str, Any] = json.load(_f)

CELLS: Dict[str, Dict[str, Any]] = TABLE["cells"]
ROWS: Dict[str, Dict[str, Any]] = TABLE["rows"]
BRIDGE: Dict[str, Any] = TABLE["bridge"]
TUNED: Dict[str, Any] = TABLE.get("tuned", {})
OPS = ("attn", "triattn", "trimul", "glut", "transition", "ln", "opm")
CORE_ROWS = ("pallas_attn", "cd_triatt", "rowshared", "fpf_core", "triattn_xla", "cudnn", "tokamax", "xla_sdpa", "xla")
ROW_NAMES = tuple(r for r in ROWS if r != "proj+<core>") + tuple("proj+" + r for r in CORE_ROWS if r not in ("xla", "xla_sdpa", "rowshared"))
STOCK_ROWS = ("xla", "xla_subbatch4", "xla_sdpa", "cudnn", "tokamax", "tokamax_glu", "tokamax_glu_T", "proj+cudnn", "proj+tokamax")
NOT_SERVED = tuple(r for r, d in ROWS.items() if d.get("served") is False)                      # engine module forms timed as arms only
FORWARD_ONLY = tuple(r for r, d in ROWS.items() if d.get("backward") is False) + ("proj+fpf_core", "proj+triattn_xla")
TIER_WORDS = ("fast", "exact", "big")
F32_WORDS = ("tf32", "tf32x3", "ieee")
DEFAULT_PRECISION_ROWS = ("cd_trimul", "cd_triatt", "cd_transition", "cd_ln", "cd_opm")     # f32 products at the library default (TF32 class); no precision word
SETTING_WORDS = tuple(TABLE["words"]["settings"])
DIRECTIONS = ("fwd", "fwdbwd")
DTYPES = {"bf16": "bf16", "bfloat16": "bf16", "f32": "f32", "fp32": "f32", "float32": "f32", "fp16": "fp16", "float16": "fp16", "f16": "fp16"}
LINES = tuple(k for k in TABLE["lines"] if k != "rule")                                           # "0.10", "0.6", "0.5"
PAD_HEAD_DIM_TO = 16
LEVERS_OFF_ENV = "MODEL_OPT_LEVERS_OFF"
LEVERS_OFF_ALL = "pallas"
CC_ENV = "OPT_CORE_PALLAS_CC"                # a compute capability used for selection instead of the default backend's device (tracing off-target)

# refusal kinds (Refusal.kind)
UNKNOWN_WORD = "unknown_word"
UNKNOWN_OP = "unknown_op"
NO_BACKWARD = "no_backward"
CUDNN_SHARED_BIAS_GRAD = "cudnn_shared_bias_grad"
F32_ON_SM80 = "f32_on_sm80"
LAUNCH_FAILS_SM80 = "launch_fails_sm80"
LAUNCH_FAILS = "launch_fails"
COMPILE_HANG = "compile_hang"
HEAD_DIM_NOT_POW2 = "head_dim_not_pow2"
HEAD_DIM_LT_16 = "head_dim_lt_16"
BIAS_PER_WINDOW = "bias_per_window"
WRONG_GRADIENT = "wrong_gradient"                       # a differentiated arm whose gradients were MEASURED outside the op's numerics class on a line (refusal_evidence)
C_NOT_SERVED = "c_not_served"
CHANNELS_NOT_POW2 = "channels_not_pow2"
N_NOT_TILE_MULTIPLE = "n_not_tile_multiple"
NEEDS_TOKAMAX = "needs_tokamax"
NEEDS_JAX_FFI = "needs_jax_ffi"
NEEDS_HAIKU = "needs_haiku"
NOT_SERVED_BY_FACE = "not_served_by_face"
LEVERS_OFF = "levers_off"
PRECISION_REQUIRED = "precision_required"
DTYPE_NOT_SERVED = "dtype_not_served"
ROW_ERROR = "row_error"
REFUSAL_KINDS = (UNKNOWN_WORD, UNKNOWN_OP, NO_BACKWARD, CUDNN_SHARED_BIAS_GRAD, F32_ON_SM80, LAUNCH_FAILS_SM80, LAUNCH_FAILS, COMPILE_HANG,
                 HEAD_DIM_NOT_POW2, HEAD_DIM_LT_16, BIAS_PER_WINDOW, C_NOT_SERVED, CHANNELS_NOT_POW2, N_NOT_TILE_MULTIPLE, NEEDS_TOKAMAX,
                 NEEDS_JAX_FFI, NEEDS_HAIKU, NOT_SERVED_BY_FACE, LEVERS_OFF, PRECISION_REQUIRED, DTYPE_NOT_SERVED, ROW_ERROR)


class Refusal(RuntimeError):
    """A row cannot serve this call: ``kind`` (REFUSAL_KINDS), ``row``, ``fallback`` (the row the table names instead, or None)."""

    def __init__(self, message: str, *, kind: str, row: Optional[str] = None, fallback: Optional[str] = None):
        super().__init__(message)
        self.kind, self.row, self.fallback = kind, row, fallback


class Selection(object):
    """What select() decided: ``row``, ``arm`` (the cell's arm key), ``config`` (launch / setting overrides, {} = the row's defaults),
    ``input_precision`` (f32 word or None), ``candidates`` (arms in serving order for a tier word; [arm] for a row word), ``cell_key`` /
    ``cell`` (None outside every measured cell), ``cls`` ('tol' | 'exact' | 'stock' | 'stock_lib' | None), ``x_stock``, ``word``, ``note``,
    ``pad_head_dim_to`` (16 when the family's head_dim is below 16 and the row is a Pallas attention row, else None)."""

    __slots__ = ("row", "arm", "config", "input_precision", "candidates", "cell_key", "cell", "cls", "x_stock", "word", "note", "pad_head_dim_to",
                 "op", "family", "direction", "jax_line", "cc", "dtype")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))
        if self.config is None:
            self.config = {}
        if self.candidates is None:
            self.candidates = [self.arm or self.row]

    def as_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__ if k != "cell"}

    def __repr__(self):
        return "Selection(%s)" % ", ".join("%s=%r" % (k, getattr(self, k)) for k in ("row", "arm", "config", "input_precision", "cell_key", "cls", "x_stock", "word", "note") if getattr(self, k) not in (None, {}, ""))


# ----------------------------------------------------------------------------------------------------------------- words and environment
def levers_off(env: Optional[Dict[str, str]] = None) -> List[str]:
    """The rows MODEL_OPT_LEVERS_OFF switches off here: word 'pallas' = every non-stock row; 'pallas:<row>' = that row."""
    env = os.environ if env is None else env
    words = [w.strip() for w in re.split(r"[,\s;]+", env.get(LEVERS_OFF_ENV, "") or "") if w.strip()]
    off: List[str] = []
    for w in words:
        if w == LEVERS_OFF_ALL:
            off.extend(r for r in ROW_NAMES if r not in STOCK_ROWS)
        elif w.startswith(LEVERS_OFF_ALL + ":"):
            off.append(w.split(":", 1)[1])
    return sorted(set(off))


def jax_line_of(version: Optional[str]) -> Optional[str]:
    """'0.10.2' -> '0.10'; '0.6.0' -> '0.6'; None -> None.  The line is major.minor; an unlisted line has no measured cell."""
    if version is None:
        return None
    m = re.match(r"^\s*(\d+)\.(\d+)", str(version))
    if not m:
        raise ValueError("jax version %r: expected 'major.minor[.patch]'" % (version,))
    return "%s.%s" % (m.group(1), m.group(2))


def cc_of(cc: Any) -> str:
    """(9, 0) | '9.0' | 9.0 | 'sm_90' -> '9.0'."""
    if isinstance(cc, (tuple, list)):
        return "%d.%d" % (int(cc[0]), int(cc[1]))
    s = str(cc).strip()
    m = re.match(r"^sm_?(\d+)(\d)$", s)
    if m:
        return "%s.%s" % (m.group(1), m.group(2))
    m = re.match(r"^(\d+)\.(\d+)$", s) or re.match(r"^(\d+)(?:\.(\d+))?$", s)
    if not m:
        raise ValueError("compute capability %r: expected '9.0' / (9, 0) / 'sm_90'" % (cc,))
    return "%s.%s" % (m.group(1), m.group(2) or "0")


def dtype_word(dtype: Any) -> str:
    s = getattr(dtype, "name", None) or str(dtype)
    s = s.replace("jnp.", "").replace("<class '", "").replace("'>", "").split(".")[-1].strip().lower()
    if s not in DTYPES:
        raise Refusal("dtype %r: the rows serve bf16 / f32 (fp16 through pallas_attn / stock rows only)" % (dtype,), kind=DTYPE_NOT_SERVED, row=None, fallback="xla")
    return DTYPES[s]


def parse_arm(arm: str) -> Tuple[str, Optional[str], Optional[str]]:
    """'<row>[:<f32 word>][@<setting>]' -> (row, f32 word | None, setting word '<row>@<setting>' | None)."""
    m = re.match(r"^([A-Za-z0-9_+<>]+)(?::([a-z0-9]+))?(?:@([a-z0-9_]+))?$", arm)
    if not m:
        raise Refusal("arm %r: expected '<row>[:<f32 word>][@<setting>]'" % (arm,), kind=UNKNOWN_WORD)
    row, ip, st = m.group(1), m.group(2), m.group(3)
    base = row.split("+", 1)[1] if row.startswith("proj+") else row
    setting = None
    if st is not None:
        setting = "%s@%s" % (base, st)
        if setting not in SETTING_WORDS:
            raise Refusal("setting %r is not a measured setting word (%s)" % (setting, ", ".join(SETTING_WORDS)), kind=UNKNOWN_WORD, row=row)
    if ip is not None and ip not in F32_WORDS:
        raise Refusal("operand-precision word %r: one of %s" % (ip, ", ".join(F32_WORDS)), kind=UNKNOWN_WORD, row=row)
    return row, ip, setting


def setting_config(setting: Optional[str]) -> Dict[str, Any]:
    return dict(TABLE["words"]["settings"][setting]["config"]) if setting else {}


def row_info(row: str) -> Dict[str, Any]:
    if row.startswith("proj+"):
        core = row.split("+", 1)[1]
        d = dict(ROWS["proj+<core>"])
        d.update(core=core, backward=ROWS[core].get("backward") if core in ROWS else None, backward_setting=ROWS[core].get("backward_setting") if core in ROWS else None)
        return d
    if row not in ROWS:
        raise Refusal("row %r: rows are %s" % (row, ", ".join(ROW_NAMES)), kind=UNKNOWN_WORD, row=row)
    return ROWS[row]


def has_backward(row: str, setting: Optional[str] = None) -> bool:
    """Whether the row (at that setting) serves a differentiated call: forward-only rows do not; a row whose table entry names a
    'backward_setting' (mlp_transition@bwd_reference: the XLA reference backward) does at that setting only."""
    info = row_info(row) if (row in ROWS or row.startswith("proj+")) else {}
    bs = info.get("backward_setting")
    if bs and setting == bs:
        return True
    if row in FORWARD_ONLY:
        return False
    return info.get("backward") is not False


# ----------------------------------------------------------------------------------------------------------------- families and cells
def families(op: str) -> List[str]:
    if op not in OPS:
        raise Refusal("op %r: ops are %s" % (op, ", ".join(OPS)), kind=UNKNOWN_OP)
    return list(TABLE["families"].get(op, []))


def attn_family_facts(fam: str) -> Dict[str, Any]:
    """'msarow_s512_h8_d32' -> {kind: msarow, n_seq: 512, heads: 8, head_dim: 32}; 'tri_h4_d32' -> {kind: tri, heads 4, head_dim 32}."""
    m = re.match(r"^(.*)_h(\d+)_d(\d+)$", fam)
    if not m:
        raise Refusal("attention family %r: expected '<kind>[_s<n_seq>]_h<heads>_d<head_dim>'" % (fam,), kind=UNKNOWN_WORD)
    kind, h, d = m.group(1), int(m.group(2)), int(m.group(3))
    n_seq = None
    ms = re.match(r"^(.*)_s(\d+)(slab\d+)?$", kind)
    if ms:
        kind, n_seq = ms.group(1) + (("_" + ms.group(3)) if ms.group(3) else ""), int(ms.group(2))
    return {"kind": kind, "n_seq": n_seq, "heads": h, "head_dim": d}


def family(op: str, fam: Optional[str] = None, **shape) -> str:
    """Build / normalise a family word.  attn: kind (tri | tmpl | msarow | msacol | extramsa_slab512 | af3_single | joltz_single | af3_dit |
    af3_atom_win32x128), heads, head_dim, n_seq (MSA kinds: the nearest measured depth at or above, else the deepest measured);
    triattn: form (af2 | af3 | joltz), unit (pair | tmpl), c, heads, head_dim, orientation; trimul: form, unit, c, c_hidden, equation
    (outgoing | incoming); transition: form, activation (relu | swiglu), c, factor; ln: unit (pair | tmpl | single | atom | dit | msa), c,
    n_seq / t; opm: form, c_m, c, f, n_seq; glut: form, c, out.  A word already in families(op) is returned as is."""
    fams = families(op)
    if fam is not None and not shape:
        if fam in fams:
            return fam
        if op != "attn":
            return fam                                              # unmeasured family word: cell_for finds no cell (tier words name the stock op)
        shape = attn_family_facts(fam)
    if op == "attn":
        kind, h, d, s = shape.get("kind"), int(shape["heads"]), int(shape["head_dim"]), shape.get("n_seq")
        if s is None:
            return "%s_h%d_d%d" % (kind, h, d)
        cands = []
        for f in fams:
            fx = attn_family_facts(f)
            if fx["kind"] == kind and fx["heads"] == h and fx["head_dim"] == d and fx["n_seq"] is not None:
                cands.append(fx["n_seq"])
        if not cands:
            return "%s_s%d_h%d_d%d" % (kind, int(s), h, d)
        at_or_above = sorted(x for x in cands if x >= int(s))
        pick = at_or_above[0] if at_or_above else max(cands)
        base, slab = (kind.split("_slab")[0], kind.split("_slab")[1]) if "_slab" in kind else (kind, None)
        return "%s_s%d%s_h%d_d%d" % (base, pick, ("slab" + slab) if slab else "", h, d)
    if op == "triattn":
        return "%s_%s_c%d_h%d_d%d_%s" % (shape["form"], shape["unit"], int(shape["c"]), int(shape["heads"]), int(shape["head_dim"]), shape["orientation"])
    if op == "trimul":
        eq = {"ikc,jkc->ijc": "outgoing", "kjc,kic->ijc": "incoming"}.get(shape["equation"], shape["equation"])
        return "%s_%s_c%d_ch%d_%s" % (shape["form"], shape["unit"], int(shape["c"]), int(shape["c_hidden"]), eq)
    if op == "transition":
        return "%s_%s_c%d_x%d" % (shape["form"], shape["activation"], int(shape["c"]), int(shape["factor"]))
    if op == "ln":
        w = "%s_c%d" % (shape["unit"], int(shape["c"]))
        if shape.get("n_seq"):
            w += "_s%d" % int(shape["n_seq"])
        if shape.get("t"):
            w += "_t%d" % int(shape["t"])
        return w
    if op == "opm":
        return "%s_opm_cm%d_c%d_f%d_S%d" % (shape["form"], int(shape["c_m"]), int(shape["c"]), int(shape["f"]), int(shape["n_seq"]))
    if op == "glut":
        return "%s_trimul_glu_c%d_out%d" % (shape.get("form", "af3"), int(shape["c"]), int(shape["out"]))
    raise Refusal("op %r" % (op,), kind=UNKNOWN_OP)


def sizes(jax_line: str, cc: str, dtype: str, op: str, fam: str, direction: str = "fwd", measured_only: bool = False) -> List[int]:
    pre = "%s|%s|%s|%s|%s|N<=" % (jax_line, cc, dtype, op, fam)
    out = []
    for k, c in CELLS.items():
        if k.startswith(pre) and k.endswith("|" + direction) and (c.get("measured") or not measured_only):
            out.append(int(k[len(pre):].split("|")[0]))
    return sorted(out)


def cell_for(jax_line: Any, cc: Any, dtype: Any, op: str, fam: str, n_tokens: int, direction: str = "fwd") -> Tuple[Optional[str], Optional[Dict[str, Any]], str]:
    """-> (cell key, cell, note).  Size rule: the smallest MEASURED size at or above n_tokens; above the largest measured size the largest
    (note says so); no measured cell -> (None, None, why)."""
    line = jax_line_of(jax_line) if jax_line is not None and re.match(r"^\d+\.\d+\.", str(jax_line)) else (str(jax_line) if jax_line is not None else None)
    ccw, dt = cc_of(cc), dtype_word(dtype)
    if op not in OPS:
        raise Refusal("op %r: ops are %s" % (op, ", ".join(OPS)), kind=UNKNOWN_OP)
    if direction not in DIRECTIONS:
        raise ValueError("direction %r: one of %s" % (direction, DIRECTIONS))
    if line not in LINES:
        return None, None, "jax line %r has no measured cell (lines %s)" % (line, ", ".join(LINES))
    ns = sizes(line, ccw, dt, op, fam, direction, measured_only=True)
    if not ns:
        alln = sizes(line, ccw, dt, op, fam, direction)
        why = "no measured cell at %s|%s|%s|%s|%s|%s" % (line, ccw, dt, op, fam, direction)
        if alln:
            k = "%s|%s|%s|%s|%s|N<=%d|%s" % (line, ccw, dt, op, fam, alln[-1], direction)
            um = CELLS[k].get("unmeasured") or {}
            why += " (closed: %s)" % ("; ".join(sorted(set(um.values()))) or "not measured")
        return None, None, why
    at_or_above = [n for n in ns if n >= int(n_tokens)]
    n = at_or_above[0] if at_or_above else ns[-1]
    note = "" if at_or_above else "above the largest measured size (%d): the N<=%d cell's rows" % (ns[-1], ns[-1])
    key = "%s|%s|%s|%s|%s|N<=%d|%s" % (line, ccw, dt, op, fam, n, direction)
    return key, CELLS[key], note


def bridge_cell(jax_line: str, cc: str, dtype: str, fam: str) -> Optional[Dict[str, Any]]:
    """The XLA-FFI bridge's own numbers for (line, cc, dtype, head_dim, heads) of a tri_* / tmpl_* attention family, or None."""
    try:
        fx = attn_family_facts(fam)
    except Refusal:
        return None
    if fx["kind"] not in ("tri", "tmpl"):
        return None
    return BRIDGE["cells"].get("%s|%s|%s|D%d|H%d" % (jax_line, cc, dtype, fx["head_dim"], fx["heads"]))


# ----------------------------------------------------------------------------------------------------------------- static refusal rules
def _family_head_dim(op: str, fam: str) -> Optional[int]:
    m = re.search(r"_d(\d+)(?:_|$)", fam)
    return int(m.group(1)) if (m and op in ("attn", "triattn")) else None


def _family_c(fam: str) -> Optional[int]:
    m = re.search(r"(?:^|_)c(\d+)(?:_|$)", fam)
    return int(m.group(1)) if m else None


def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


PALLAS_ATTENTION_ROWS = ("pallas_attn", "cd_triatt", "rowshared", "fpf_core", "fpf_block", "triattn_xla")


def static_refusal(row: str, *, op: str, fam: str, cc: str, dtype: str, direction: str, jax_line: Optional[str], cell: Optional[Dict[str, Any]], arm: str,
                   key_masked: bool = True) -> Optional[Refusal]:
    """The refusals select() can name without the framework: by rule (table 'rows') and by the cell's measured 'refused' words."""
    base = row.split("+", 1)[1] if row.startswith("proj+") else row
    if row in levers_off() or base in levers_off():
        return Refusal("%s: switched off by %s" % (row, LEVERS_OFF_ENV), kind=LEVERS_OFF, row=row, fallback=_fallback(row, direction))
    if base in NOT_SERVED:
        return Refusal("%s: an engine module form timed as an arm; not served by this face" % row, kind=NOT_SERVED_BY_FACE, row=row, fallback="xla")
    if direction == "fwdbwd" and not has_backward(row, parse_arm(arm)[2] if arm else None):
        return Refusal("%s: forward only (no VJP): a differentiated call is refused by name; rows with a backward on this op: %s" % (row, ", ".join(r for r in _rows_for(op) if has_backward(r))),
                       kind=NO_BACKWARD, row=row, fallback=_fallback(row, direction))
    if direction == "fwdbwd" and base == "mlp_transition" and dtype == "bf16" and "bwd_reference" in str(parse_arm(arm)[2] or "") and jax_line_of(jax_line) == "0.6":
        return Refusal("mlp_transition@bwd_reference: its backward in bf16 on the jax 0.6 line returns gradients outside the op's class (measured d/dx cosine 0.58, "
                       "norm x1.12-1.23, d/dW2 cosine -0.08..0.37 vs an f32 reference at c 64 x2 / c 128 x4, 431-800 tokens, cc 9.0 and 8.0; the statement and "
                       "cd_transition read 0.9995 there; f32 and the jax 0.5 line are unaffected) -- refused by name; differentiated bf16 rows on this line: "
                       "cd_transition, xla", kind=WRONG_GRADIENT, row=row, fallback="cd_transition" if activation_of(fam) == "relu" else "xla")
    if base == "cd_transition" and dtype != "bf16":
        return Refusal("cd_transition: bf16 activations only (the module's operand contract: bf16 x bf16 tile products, f32 accumulation); f32: mlp_transition's "
                       "precision words or the statement", kind=DTYPE_NOT_SERVED, row=row, fallback="xla")
    if direction == "fwdbwd" and (base == "cudnn" or arm.endswith("@cudnn")) and key_masked:
        # measured through this face (jax 0.10.2, cc 9.0): with a key mask whose fully-masked query rows are read by the loss, cuDNN's VJP returns
        # gradients 40-57x the reference's scale from those rows (cosine 0.16-0.18 vs the fp32 reference for dq, dk, dv and the shared d(bias);
        # pallas_attn 0.99) and on the jax 0.5 line the shared d(bias) is ZERO whenever a key mask is given; the unmasked form is right on both
        # lines (cosine >= 0.9999, d(bias) rel-RMS 3e-3).  A key mask is a
        # data-dependent hazard the face cannot bound at trace time -> refused by name for differentiated MASKED calls; unmasked calls serve.
        return Refusal("%s: differentiated call with a key mask -- cuDNN's attention VJP drops the shared d(bias) entirely on the jax 0.5 line (d(bias) == 0 "
                       "measured with any key mask) and returns gradients 40-57x the reference scale from fully-masked query rows on the jax 0.10 line "
                       "(cosine 0.16-0.18 measured); refused by name (unmasked differentiated calls and every forward call serve)" % row,
                       kind=CUDNN_SHARED_BIAS_GRAD, row=row, fallback=_fallback(row, direction))
    hd = _family_head_dim(op, fam)
    if hd is not None and base in PALLAS_ATTENTION_ROWS + ("cudnn", "tokamax") and "atom_win" in fam and base != "tokamax":
        return Refusal("%s: a pair bias per window ([B,H,Sq,Sk]) is outside the shared-bias rows" % row, kind=BIAS_PER_WINDOW, row=row, fallback="tokamax" if jax_line == "0.10" else "xla_sdpa")
    if hd is not None and base in PALLAS_ATTENTION_ROWS and not _is_pow2(hd):
        return Refusal("%s: head_dim %d is not a power of two (Triton block shapes); rows: xla_sdpa / tokamax / xla" % (row, hd), kind=HEAD_DIM_NOT_POW2, row=row, fallback="xla_sdpa")
    if base == "cudnn" and dtype == "f32":
        return Refusal("cudnn: fused attention serves fp16 / bf16 only", kind=DTYPE_NOT_SERVED, row=row, fallback="xla_sdpa")
    if base == "fpf_core" and dtype == "f32":
        return Refusal("fpf_core: bf16 only", kind=DTYPE_NOT_SERVED, row=row, fallback="pallas_attn")
    if base == "cd_trimul" and dtype == "f32" and cc.split(".")[0] == "8" and (_family_c_hidden(fam) or 128) >= 128 and direction == "fwdbwd" and jax_line_of(jax_line) == "0.5":
        return Refusal("cd_trimul: f32 backward at c_hidden >= 128 on cc %s, jax 0.5 line: its backward tiles request 262,144 B of shared memory (166,912 available) and fail "
                       "to launch (measured at 400 / 800 / 1200 tokens); refused by name" % cc,
                       kind=LAUNCH_FAILS_SM80, row=row, fallback="fpf_trimul_xlabwd:tf32")
    c = _family_c(fam)
    if op == "transition" and base in ("fpf_transition", "mlp_transition", "cd_transition") and c is not None and (c > 256 or not _is_pow2(c)):
        return Refusal("%s: c %d outside the served channel widths (power of two <= 256); the stock statement by name" % (row, c), kind=C_NOT_SERVED, row=row, fallback="xla")
    if op == "transition" and base == "fpf_transition" and "relu" in fam:
        return Refusal("fpf_transition: swiglu only; relu rows: mlp_transition / cd_transition", kind=C_NOT_SERVED, row=row, fallback="mlp_transition")
    if op == "transition" and base == "cd_transition" and "swiglu" in fam:
        return Refusal("cd_transition: relu only; swiglu rows: fpf_transition / mlp_transition", kind=C_NOT_SERVED, row=row, fallback="mlp_transition")
    if op == "ln" and base == "cd_ln" and c is not None and not (_is_pow2(c) and 32 <= c <= 1024):
        return Refusal("cd_ln: channels %d not a power of two in 32..1024" % c, kind=CHANNELS_NOT_POW2, row=row, fallback="xla")
    if cell is not None:
        why = (cell.get("refused") or {}).get(arm)
        if why:
            kind = why.split(":", 1)[0]
            kind = {"launch_fails": LAUNCH_FAILS, "compile_hang": COMPILE_HANG, "head_dim_lt_16": HEAD_DIM_LT_16}.get(kind, kind if kind in REFUSAL_KINDS else ROW_ERROR)
            if kind == HEAD_DIM_LT_16 and base in ("rowshared", "pallas_attn", "cd_triatt"):
                return None                                                       # the face pads 8 -> 16: served
            return Refusal("%s: measured '%s' on cell %s|...|%s" % (row, why, cell.get("stack"), fam), kind=kind, row=row, fallback=_fallback(row, direction, cell))
    return None


def _rows_for(op: str) -> List[str]:
    out = []
    for r, d in ROWS.items():
        if r == "proj+<core>":
            continue
        ops = d.get("ops", [])
        if any(o.split(" ")[0] == op for o in ops):
            out.append(r)
    if op == "triattn":
        out += ["proj+" + r for r in CORE_ROWS if r not in ("xla", "xla_sdpa", "rowshared")]
    return out


def _fallback(row: str, direction: str, cell: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """The named fallback: the cell's next admissible arm if a cell is known, else the row's declared fallback chain (first with a backward for fwdbwd)."""
    if cell is not None:
        for a in cell.get("fast_order", []):
            r = parse_arm(a)[0]
            if r != row and (direction == "fwd" or has_backward(r)) and r not in NOT_SERVED:
                return a
    seen, r = set(), row
    while True:
        fb = row_info(r).get("fallback") if (r in ROWS or r.startswith("proj+")) else None
        if fb is None or fb in seen:
            return "xla"
        if direction == "fwd" or has_backward(fb):
            return fb
        seen.add(fb)
        r = fb


# ----------------------------------------------------------------------------------------------------------------- select
def select(jax_line: Any, cc: Any, dtype: Any, op: str, fam: str, n_tokens: int, direction: str = "fwd", *, word: str,
           prefer: Optional[Sequence[str]] = None, key_masked: bool = True) -> Selection:
    """What serves the call (see ``_select``); the decision is recorded ONCE per call class in :mod:`opt_core.cell_census` (pure observation:
    the Selection / Refusal is decided first and returned unchanged; the serving walk records a step-aside by name)."""
    try:
        sel = _select(jax_line, cc, dtype, op, fam, n_tokens, direction, word=word, prefer=prefer, key_masked=key_masked)
    except Refusal as e:
        _census(jax_line, cc, dtype, op, fam, n_tokens, direction, word, None, e)
        raise
    _census(jax_line, cc, dtype, op, fam, n_tokens, direction, word, sel, None)
    return sel


_CENSUS_KEYS: Dict[tuple, Dict[str, Any]] = {}          # (cell_key, word, arm) -> the census key select() recorded (the serving walk re-records under it)


def _census(jax_line, cc, dtype, op, fam, n_tokens, direction, word, sel, refusal) -> None:
    """Classify the decision (cell_hit | inherited | named_fallback | stock | opt_in) from the Selection's own facts and record it."""
    try:
        if sel is not None:
            line, ccw, dt, ckey, famw = sel.jax_line, sel.cc, sel.dtype, sel.cell_key, sel.family
        else:
            line = None if jax_line is None else (jax_line_of(jax_line) if re.match(r"^\d+\.\d+\.", str(jax_line) + ".") else str(jax_line))
            ccw, dt, ckey, famw = cc_of(cc if cc is not None else os.environ.get(CC_ENV, "9.0")), str(dtype), None, fam
        inside = False
        if ckey:
            try:
                inside = int(n_tokens) <= int(str(ckey).split("|")[5][3:])
            except (IndexError, ValueError):
                inside = False
        key = dict(cc=ccw, stack="jax%s" % (line if line is not None else "-"), dtype=dt, shape="%s:%s" % (op, famw), bucket=_CENSUS.bucket_word(n_tokens, ckey, inside),
                   form=direction, word=word)
        if refusal is not None:
            _CENSUS.record("pallas", key, "named_fallback", "caller:%s" % (refusal.fallback or "xla"), cell_id=ckey,
                           refused="%s:%s" % (refusal.row or word, refusal.kind), note="raised")
            return
        if len(_CENSUS_KEYS) < 4096:
            _CENSUS_KEYS[(ckey, word, sel.arm)] = key
        if word not in TIER_WORDS and word not in F32_WORDS:
            _CENSUS.record("pallas", key, "opt_in", sel.arm, cell_id=ckey, note="" if ckey else "outside_every_measured_cell")
        elif ckey is None or not (sel.cell or {}).get("measured"):
            if sel.row != "xla" and "bridge cells" in str(sel.note):     # the bridge's own measured cells decided (another table of this tree)
                _CENSUS.record("pallas", key, "cell_hit", sel.arm, cell_id="bridge:%s|%s|%s|%s" % (line, ccw, dt, famw), note="bridge_cells")
            else:
                _CENSUS.record("pallas", key, "inherited", sel.arm, cell_id=None, note="family:none(%s)" % sel.note)
        elif INHERITED_CC_TOKEN in str(sel.note):                                # a cc without a measured column served by the inherited column's rows
            _CENSUS.record("pallas", key, "inherited", sel.arm, cell_id=ckey, note="%s(from %s)" % (INHERITED_CC_TOKEN, str(ckey).split("|")[1] if ckey else "-"))
        elif not inside:
            _CENSUS.record("pallas", key, "inherited", sel.arm, cell_id=ckey, note="beyond_measured")
        elif sel.row in STOCK_ROWS or sel.arm == "xla":
            _CENSUS.record("pallas", key, "stock", sel.arm, cell_id=ckey, note=sel.note if "prefer=" in str(sel.note) else "")
        else:
            _CENSUS.record("pallas", key, "cell_hit", sel.arm, cell_id=ckey)
    except Exception:                                   # the census counts; it never gates or breaks a selection
        return


def census_stepaside(sel: Selection, served_arm: str, refused_row: Optional[str], refused_kind: Optional[str], passed: Sequence[str] = ()) -> None:
    """The serving walk stepped past ``sel``'s first candidate(s) (refused by name with the arrays in hand) and ``served_arm`` engaged: re-record
    the call class as a NAMED_FALLBACK under the key select() recorded."""
    try:
        key = _CENSUS_KEYS.get((sel.cell_key, sel.word, sel.arm))
        if key is None:
            key = dict(cc=sel.cc, stack="jax%s" % (sel.jax_line if sel.jax_line is not None else "-"), dtype=sel.dtype, shape="%s:%s" % (sel.op, sel.family),
                       bucket=_CENSUS.bucket_word(None, sel.cell_key, True), form=sel.direction, word=sel.word)
        _CENSUS.record("pallas", key, "named_fallback", served_arm, cell_id=sel.cell_key, refused="%s:%s" % (refused_row or "-", refused_kind or "-"),
                       note="stepaside(%s)" % "+".join(str(a) for a in passed))
    except Exception:                                   # the census counts; it never gates or breaks a served call
        return


def _select(jax_line: Any, cc: Any, dtype: Any, op: str, fam: str, n_tokens: int, direction: str = "fwd", *, word: str,
            prefer: Optional[Sequence[str]] = None, key_masked: bool = True) -> Selection:
    if op not in OPS:
        raise Refusal("op %r: ops are %s" % (op, ", ".join(OPS)), kind=UNKNOWN_OP)
    if direction not in DIRECTIONS:
        raise ValueError("direction %r: one of %s" % (direction, DIRECTIONS))
    line = None if jax_line is None else (jax_line_of(jax_line) if re.match(r"^\d+\.\d+\.", str(jax_line) + ".") else str(jax_line))
    line = jax_line_of(line) if line is not None else None
    ccw = cc_of(cc if cc is not None else os.environ.get(CC_ENV, "9.0"))
    dt = dtype_word(dtype)
    fam = family(op, fam)
    # PER KEY (unmeasured_cc_rule): the caller's own column first; when it holds no MEASURED cell for this key (line, dtype, op, family, direction) --
    # whether or not it holds cells for other keys -- the nearest arch-compatible measured column that does serves the tier words (portable rows only)
    cc_src = None
    if line is not None:
        key, cell, size_note = cell_for(line, ccw, dt, op, fam, n_tokens, direction)
        if cell is None or not cell.get("measured"):
            for col in inherit_columns(ccw):
                k2, c2, n2 = cell_for(line, col, dt, op, fam, n_tokens, direction)
                if c2 is not None and c2.get("measured"):
                    key, cell, size_note, cc_src = k2, c2, n2, col
                    break
    else:
        key, cell, size_note = None, None, "no jax line given"
    ccl = cc_src or ccw
    if cc_src is not None:
        partial = ccw in measured_ccs()
        size_note = (size_note + "; " if size_note else "") + "%s(from %s)%s: cc %s has no measured cell for this key -- the %s column's source-compiled Pallas / Triton rows serve " \
                    "fast / big (prebuilt per-arch bridge rows and library ops are not inherited); exact = the stock statement (not vouched on this cc)" % (
                        INHERITED_CC_TOKEN, cc_src, " partial_cc" if partial else "", ccw, cc_src)
    common = dict(op=op, family=fam, direction=direction, jax_line=line, cc=ccw, dtype=dt, cell_key=key, cell=cell, word=word)
    hd = _family_head_dim(op, fam)

    def finish(arm: str, candidates: List[str], note: str) -> Selection:
        row, ip, setting = parse_arm(arm)
        cfg = setting_config(setting)
        cfg.update(_tuned_config(row, line, ccl, dt, op, fam, arm))
        if setting is None and (word in TIER_WORDS or word in F32_WORDS):
            tw = tuned_for(row, line, ccl, op, dt, fam)                          # a tier word also applies the launch setting this provider measured
            if tw:                                                                # ahead on (row, line, cc, op); the row word never does (default rule)
                cfg.update(TUNED[tw]["config"])
                note = (note + "; " if note else "") + "tuned: %s (%s)" % (tw, TUNED[tw].get("measured", {}).get("%s|%s" % (line, ccw), ""))
        pad = PAD_HEAD_DIM_TO if (hd is not None and hd < PAD_HEAD_DIM_TO and row.split("+")[-1] != "xla" and op in ("attn", "triattn")) else None
        x = (cell or {}).get("x_stock", {}).get(arm) if cell else None
        cls = (cell or {}).get("cls", {}).get(arm) if cell else None
        return Selection(row=row, arm=arm, config=cfg, input_precision=ip, candidates=candidates, cls=cls, x_stock=x, note=note, pad_head_dim_to=pad, **common)

    if word in TIER_WORDS or word in F32_WORDS:
        if word in F32_WORDS and dt != "f32":
            raise Refusal("word %r selects an f32 operand precision: dtype is %s" % (word, dt), kind=UNKNOWN_WORD)
        if cell is None or not cell.get("measured"):
            # the bridge's own cells may still name its row ahead of the stock statement for a triangle-attention forward (its measured times vs xla)
            brow = _bridge_arm(op, fam, line, ccw, dt, direction) if cc_src is None else None
            if word in ("fast", "big") and brow and (prefer is None or any("triattn_xla" in p for p in prefer)):
                if static_refusal(parse_arm(brow)[0], op=op, fam=fam, cc=ccw, dtype=dt, direction=direction, jax_line=line, cell=None, arm=brow) is None:
                    return finish(brow, [brow, "xla"], "no measured Pallas cell (%s); the bridge's kernels measured ahead of the stock statement here (bridge cells)" % size_note)
            return finish("xla", ["xla"], "no measured cell (%s): the stock statement, by name" % size_note)
        fl = cell.get("n_floor")                                                  # a program floor (TABLE["program_floors"]): below it the floor's arm serves under tier words --
        if fl and int(n_tokens) < int(fl["below"]) and (fl["arm"] == "xla" or word != "exact"):   # the stock statement under every word; a named row under fast / big only (exact keeps the cell's exact arm)
            return finish(fl["arm"], [fl["arm"]] if fl["arm"] == "xla" else [fl["arm"], "xla"],
                          "program floor: below %d tokens %s serves under tier words (%s)" % (int(fl["below"]), fl["arm"], fl.get("source", "")))
        if word == "exact" and cc_src is not None:
            order = ["xla"]                                                       # exact is vouched per stack: never on a cc without a measured column
        elif word == "exact":
            order = [cell["exact"]] + (["xla"] if cell["exact"] != "xla" else [])
        elif word == "big":
            # big = fast's order minus arms with a MEASURED memory cost (TABLE["big_rule"]): an arm whose peak (XLA memory analysis of the jitted arm)
            # exceeds the stock statement's by more than max(8 MiB, 2 %) is passed over, named in the note; per-call deltas below that are not a cost
            order = list(cell["fast_order"])
            costly = memory_cost_arms(cell)
            if costly:
                order = [a for a in order if a not in costly] or ["xla"]
                size_note = (size_note + "; " if size_note else "") + "big passes over a measured memory cost: %s" % ", ".join("%s (+%.0f MiB over the statement)" % (a, costly[a]) for a in order[:0] or costly)
        else:
            order = list(cell["fast_order"])
            if word in F32_WORDS:
                # arms carrying that precision word; the stock rows; and, for 'tf32', rows whose f32 products run at the library default (TF32 class)
                # and expose no precision word (DEFAULT_PRECISION_ROWS)
                order = [a for a in order if (parse_arm(a)[1] in (word, None) and (parse_arm(a)[1] == word or parse_arm(a)[0] in STOCK_ROWS or a == "xla"
                                                                                 or (word == "tf32" and parse_arm(a)[0] in DEFAULT_PRECISION_ROWS)))] or ["xla"]
        # the bridge rule: triangle attention where the bridge's kernels measured ahead -- forward: its forward rows vs the fused Pallas core;
        # differentiated: its differentiable row ('triattn_xla@vjp', by word) vs the best Pallas fwd+bwd row and the stock statement
        brow = _bridge_arm(op, fam, line, ccw, dt, direction) if cc_src is None else None   # the bridge rows are prebuilt per arch: measured cc only
        if word in ("fast", "big") and brow:                                  # the bridge rows hold no per-row S x S intermediates beyond the staged bias
            order = [brow] + order                                              # (an adopting program measured 0.00 GiB peak cost): no memory cost -> big too
        if cc_src is not None and word != "exact":
            order = [a for a in order if portable_arm(a)] or ["xla"]            # unmeasured cc: source-compiled rows only (never a prebuilt of another arch);
            if any(not _library_arm(a) and a != "xla" and parse_arm(a)[0] not in STOCK_ROWS for a in order):   # a library op only when no Triton row remains
                order = [a for a in order if not _library_arm(a)] or ["xla"]
        cands = []
        for a in order:
            r = parse_arm(a)[0]
            if prefer is not None and r not in prefer and a != "xla":
                continue
            if static_refusal(r, op=op, fam=fam, cc=ccw, dtype=dt, direction=direction, jax_line=line, cell=cell, arm=a) is not None:
                continue
            cands.append(a)
        if not cands or cands[-1] != "xla":
            cands.append("xla")
        note = size_note
        if prefer is not None and cands[0] == "xla" and cell["fast"] != "xla":
            note = (note + "; " if note else "") + "none of prefer=%s beat the stock op here (cell fast = %s)" % (list(prefer), cell["fast"])
        return finish(cands[0], cands, note)
    # a row / arm word
    row, ip, setting = parse_arm(word)
    if row not in ROW_NAMES and not (row.startswith("proj+") and op == "triattn"):
        raise Refusal("word %r: rows %s; tiers %s; f32 words %s; settings %s" % (word, ", ".join(ROW_NAMES), ", ".join(TIER_WORDS), ", ".join(F32_WORDS), ", ".join(SETTING_WORDS)),
                      kind=UNKNOWN_WORD, row=row)
    if row not in _rows_for(op) and not row.startswith("proj+"):
        raise Refusal("row %r does not serve op %r (its ops: %s)" % (row, op, ", ".join(row_info(row).get("ops", []))), kind=UNKNOWN_WORD, row=row)
    ref = static_refusal(row, op=op, fam=fam, cc=ccw, dtype=dt, direction=direction, jax_line=line, cell=cell, arm=word, key_masked=key_masked)
    if ref is not None:
        raise ref
    note = size_note if cell is not None else "a row word outside every measured cell: served as named (no measured number here)"
    return finish(word, [word], note)


def memory_cost_arms(cell: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Arms of a measured cell whose peak exceeds the stock statement's by more than max(8 MiB, 2 %) (TABLE["big_rule"]) -> {arm: MiB over}."""
    if not cell or not cell.get("measured"):
        return {}
    pk = cell.get("peak_mib") or {}
    ref = pk.get(cell.get("stock_row", "xla"), pk.get("xla"))
    if ref is None:
        return {}
    thr = max(8.0, 0.02 * float(ref))
    return {a: float(pk[a]) - float(ref) for a in cell.get("fast_order", []) if a in pk and pk[a] is not None and float(pk[a]) - float(ref) > thr}


INHERITED_CC_TOKEN = "inherited_cc:unmeasured"        # census / note token when a cc without a measured column inherits one (unmeasured_cc_rule)
NON_PORTABLE_ROWS = ("native_xla", "triattn_xla")       # prebuilt per-arch bridges (sm_90a CUDA / per-arch cubins): never inherited across compute capabilities
LIBRARY_ROW_MARKS = ("cudnn",)                          # library ops: not inherited for fast / big while an arch-portable Triton row remains in the order


def measured_ccs() -> List[str]:
    """The compute-capability columns this table holds cells for (its stacks' and its cell keys')."""
    cols = {str(v.get("cc")) for v in TABLE.get("stacks", {}).values() if v.get("cc")}
    cols.update(_cell_columns())
    return sorted(cols, key=_cc_num)


_CELL_COLS: Optional[List[str]] = None


def _cell_columns() -> List[str]:
    global _CELL_COLS
    if _CELL_COLS is None or len(TABLE.get("cells", {})) != getattr(_cell_columns, "_n", -1):
        _CELL_COLS = sorted({k.split("|")[1] for k in TABLE.get("cells", {})}, key=_cc_num)
        _cell_columns._n = len(TABLE.get("cells", {}))
    return _CELL_COLS


def _cc_num(ccw: str) -> float:
    try:
        return float(ccw)
    except ValueError:
        return 0.0


def _arch_compatible(ccw: str, col: str) -> bool:
    """Which measured columns may serve cc `ccw`: major >= 9 parts (9.x, 10.x, 12.x) <- columns with major >= 9 at or below the part's own number
    (a newer part runs an older part's source-compiled rows, not the reverse); 8.x <- 8.x columns at or below it; older parts: none."""
    try:
        M, m = (int(v) for v in str(ccw).split("."))
        Mc, mc = (int(v) for v in str(col).split("."))
    except ValueError:
        return False
    if (Mc, mc) > (M, m):
        return False
    if M >= 9:
        return Mc >= 9
    if M == 8:
        return Mc == 8
    return False


def inherit_columns(ccw: str) -> List[str]:
    """Ordered candidate columns for a key the caller's cc has no measured cell for: arch-compatible measured columns other than the caller's own,
    nearest first (10.3 -> 10.0 before 9.0 when both hold cells; 8.9 -> 8.6 before 8.0)."""
    cands = [c for c in measured_ccs() if c != ccw and _arch_compatible(ccw, c)]
    return sorted(cands, key=lambda c: (abs(_cc_num(ccw) - _cc_num(c)), -_cc_num(c)))


def inherit_cc(ccw: str) -> Optional[str]:
    """The nearest arch-compatible measured column other than `ccw` (None when there is none) -- the column a key falls back to when the caller's cc
    holds no measured cell FOR THAT KEY (decided per key in _select, never per cc: a cc owning cells for other keys still inherits for this one)."""
    cols = inherit_columns(ccw)
    return cols[0] if cols else None


def activation_of(fam: str) -> str:
    """relu | swiglu from a transition family word (af2_relu_c64_x2, af3_swiglu_c128_x4, joltz_swiglu_...); '' when not a transition family."""
    m = re.search(r"_(relu|swiglu)_", "_" + str(fam) + "_")
    return m.group(1) if m else ""


def portable_arm(arm: str) -> bool:
    """Source-compiled Pallas / Triton rows and the XLA statements travel across compute capabilities; prebuilt per-arch bridge rows do not."""
    row = parse_arm(arm)[0]
    parts = row.split("+")
    return not any(pt.startswith(NON_PORTABLE_ROWS) for pt in parts)


def _library_arm(arm: str) -> bool:
    """cudnn-backed arms (row cudnn, or a row's @cudnn implementation setting) are library ops."""
    row, _ip, setting = parse_arm(arm)
    return any(m in row or m in str(setting or "") for m in LIBRARY_ROW_MARKS)


def _bridge_arm(op: str, fam: str, line: Optional[str], cc: str, dt: str, direction: str) -> Optional[str]:
    """The bridge arm the 'fast' order is headed by, or None: forward -> 'triattn_xla' ('proj+triattn_xla' at module level) where the bridge
    cell's forward kernels measured ahead of the fused Pallas core; differentiated (head_dim 32) -> 'triattn_xla@vjp' where the bridge cell's
    'fwdbwd' block is preferred (ahead of the stock statement, at or within 3 % of the best Pallas fwd+bwd row)."""
    if op not in ("attn", "triattn"):
        return None
    bfam = fam if op == "attn" else _triattn_core_family(fam)
    if not bfam or not re.match(r"^(tri|tmpl)_h\d+_d\d+$", bfam):
        return None
    b = bridge_cell(line, cc, dt, bfam)
    if not b:
        return None
    if direction == "fwd":
        return ("triattn_xla" if op == "attn" else "proj+triattn_xla") if b.get("ahead_of_pallas") else None
    fb = b.get("fwdbwd")
    if fb and fb.get("preferred") and bfam.endswith("_d32"):
        return "triattn_xla@vjp" if op == "attn" else "proj+triattn_xla@vjp"
    return None


def _family_c_hidden(fam: str) -> Optional[int]:
    m = re.search(r"_ch(\d+)_", fam)
    return int(m.group(1)) if m else None


def _triattn_core_family(fam: str) -> Optional[str]:
    """'af2_pair_c128_h4_d32_starting' -> 'tri_h4_d32'; '..._tmpl_c64_h4_d16_...' -> 'tmpl_h4_d16'."""
    m = re.match(r"^[a-z0-9]+_(pair|tmpl)_c\d+_h(\d+)_d(\d+)_(starting|ending)$", fam)
    if not m:
        return None
    return "%s_h%s_d%s" % ("tri" if m.group(1) == "pair" else "tmpl", m.group(2), m.group(3))


def tuned_for(row: str, line: Optional[str], cc: str, op: str, dtype: Optional[str] = None, fam: Optional[str] = None) -> Optional[str]:
    """The tuned setting word (table 'tuned') measured at >= x1.03 over the row's default launch on (line, cc) for this op (and, where the entry
    names them, this dtype and a family with c_hidden >= its 'min_c_hidden'), or None."""
    base = row.split("+", 1)[1] if row.startswith("proj+") else row
    best, best_x = None, 1.03
    for w, ent in TUNED.items():
        if parse_arm(w)[0] != base or op not in ent.get("ops", [op]):
            continue
        if ent.get("dtypes") and dtype is not None and dtype not in ent["dtypes"]:
            continue
        if ent.get("min_c_hidden") and fam is not None and (_family_c_hidden(fam) or 0) < ent["min_c_hidden"]:
            continue
        x = (ent.get("measured") or {}).get("%s|%s" % (line, cc))
        if isinstance(x, (int, float)) and x >= best_x:
            best, best_x = w, x
    return best


def _tuned_config(row: str, line: Optional[str], cc: str, dtype: str, op: str, fam: str, arm: str) -> Dict[str, Any]:
    """Launch settings this provider measured itself (table 'tuned'): applied only to the arm word that names them ('<row>@<setting>')."""
    out: Dict[str, Any] = {}
    _, _, setting = parse_arm(arm)
    if setting and setting in TUNED:
        ent = TUNED[setting]
        ok_cc = ent.get("cc") in (None, cc) or cc in (ent.get("cc") if isinstance(ent.get("cc"), list) else [ent.get("cc")])
        if ok_cc:
            out.update(ent.get("config", {}))
    return out


# ----------------------------------------------------------------------------------------------------------------- description / coverage
def describe(jax_line: Any, cc: Any, dtype: Any, op: str, fam: str, n_tokens: int, direction: str = "fwd") -> str:
    key, cell, note = cell_for(jax_line, cc, dtype, op, family(op, fam), n_tokens, direction)
    if cell is None:
        return "no cell: " + note
    lines = ["cell %s  (stack %s; stock %s ms%s)" % (key, cell["stack"], cell.get("stock_ms"), "; " + note if note else "")]
    for a in cell["fast_order"]:
        lines.append("  %-34s x%-6s %8s ms  %-9s %s" % (a, cell["x_stock"].get(a), cell["ms"].get(a), cell["cls"].get(a), (cell.get("numerics") or {}).get(a, "")))
    slower = [a for a in cell["ms"] if a not in cell["fast_order"]]
    for a in sorted(slower, key=lambda a: -(cell["x_stock"].get(a) or 0)):
        lines.append("  %-34s x%-6s %8s ms  %-9s (below the stock op)" % (a, cell["x_stock"].get(a), cell["ms"].get(a), cell["cls"].get(a)))
    for a, why in sorted((cell.get("refused") or {}).items()):
        lines.append("  %-34s refused: %s" % (a, why))
    lines.append("  fast=%s exact=%s big=%s%s" % (cell["fast"], cell["exact"], cell["big"], ("  parity: %s" % cell["parity"]) if cell.get("parity") else ""))
    return "\n".join(lines)


def coverage() -> Dict[str, Any]:
    """Counts: cells per (line, cc, op), measured vs closed, winners per row."""
    out: Dict[str, Any] = {"cells": len(CELLS), "measured": 0, "by_line_cc_op": {}, "fast_winners": {}, "exact_named": 0, "bridge_cells": len(BRIDGE.get("cells", {}))}
    for k, c in CELLS.items():
        line, cc, dt, op = k.split("|")[:4]
        g = "%s|%s|%s" % (line, cc, op)
        out["by_line_cc_op"][g] = out["by_line_cc_op"].get(g, 0) + 1
        if c.get("measured"):
            out["measured"] += 1
            r = parse_arm(c["fast"])[0]
            out["fast_winners"][r] = out["fast_winners"].get(r, 0) + 1
            if c["exact"] != "xla":
                out["exact_named"] += 1
    return out


def rows_table() -> List[Dict[str, Any]]:
    return [dict(row=r, **{k: v for k, v in d.items() if k in ("cls", "backward", "jax_lines", "cc", "dtypes", "fallback")}) for r, d in ROWS.items()]
