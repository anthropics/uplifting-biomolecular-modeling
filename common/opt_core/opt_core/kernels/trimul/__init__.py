"""Triangle multiplication: ONE provider over every carried implementation ("row") and the measured cell table (TRIMUL_CELLS.json).

The op (every row serves this boundary unless its ``rows`` entry says otherwise)::

    z [B,N,N,c_z] (or [N,N,c_z]) = the PRE-LayerNorm pair tensor, mask [B,N,N] (or [N,N]) 0/1 (None = all ones)
    x = LN_in(z);  a = mask * sigmoid(x W_ag^T) * (x W_ap^T);  b = mask * sigmoid(x W_bg^T) * (x W_bp^T)          (a, b: [.., N, N, c_hidden])
    X[i,j,:] = sum_k a[i,k,:] b[j,k,:]   (outgoing)   |   sum_k a[k,i,:] b[k,j,:]   (incoming)
    update = sigmoid(x W_og^T) * (LN_out(X) W_o^T);   returns update, or z + update when residual=True

LN_in, the four projections, both gates, the contraction, LN_out and the gated output projection are INSIDE the boundary for every row; the
weights are the ten tensors of ``opt_core.trimul_weights.WEIGHT_KEYS`` (``ln_in_w ln_in_b w_ag w_ap w_bg w_bp ln_out_w ln_out_b w_o w_og``).

Rows (``ROW_NAMES``; the table's ``rows`` block states each one's envelope, numerics class, backward, residual handling and named fallback):

    v4              opt_core.kernels.fpf_trimul_v4 (generic face; Triton K1 | cuBLAS bmm | Triton K3)              fast class, cc >= 8.0, c in {128,256};
                    + (c_z 64, c_hidden 128) bf16 through the package's kernel face with the cell's launch cells
    tmk3_exact      opt_core.kernels.fpf_trimul at its kernel face with the EXACT-mode config table (c_hidden == c_z) EXACT vs the cuequivariance op
    tmk3_fast       the same line with the FAST-mode config table (padded planes, fused LN statistics)               fast class
    tx_sm90a        .tx_sm90a -- prebuilt sm_90a TMA/wgmma kernels, bf16, c 256; loads only on a stack with a      fast class, cc 9.0 + ABI
                    prebuilt binary, refuses BY NAME elsewhere (fallback: v4)
    tx_sm90a_exact  the same package's exact assembly (stock-order LayerNorms, unpadded planes)                       EXACT vs the cuequivariance op
    ef2_fused       .ef2.ef2_trimul -- autograd residual TriMul of the ESM-family pair block (fwd AND bwd, d pair);   fast class; residual REQUIRED;
                    needs the ESM-family fork's vendored GEMM kernels  (import refusal elsewhere; fallback: cueq)
    ef2_cueq_tiles  .ef2.ef2_trimul -- per-capability tile entries for the stock library's GEMM kernels; the op is   EXACT (out and d pair) per pin
                    the stock call itself (fwd AND bwd)
    esm_v5_fwd      .esm_v5 -- the ESM-family engine's forward line (Triton K1 with tensor-descriptor loads | cuBLAS bmm |   fast class, cc >= 8.0 (cells: 9.0),
                    Triton K3; levers incnt + formtab + sigmoid) from the ten canonical tensors through a plain-torch relayout;  bf16, c 128 | 256, forward only,
                    imports torch + triton only (no model package of that stack), so it loads on any stack whose triton has    residual optional; fallback: v4
                    tensor descriptors (>= 3.6 measured) and refuses BY NAME elsewhere
    esm_v61         .esm_v61 -- the same engine's sealed package generation 6.1: the K1 planes and cuBLAS bmm of the line above with   fast class, cc 9.0 ONLY (sm_90a cubin),
                    K3 = one warp-specialized CUDA C++/CuTe kernel shipped as a PREBUILT sm_90a cubin (LN affine folded into the      bf16, c 256, forward only, residual
                    projections) launched through the CUDA driver -- keyed by architecture, not by the framework ABI: the same       True (its epilogue) or False (the same
                    cubin serves every stack whose driver loads it (driver binding: the cuda.bindings wheel or ctypes over libcuda);   cubin over a zero-tile residual map);
                    digests + load check + the package's own vectors as byte gate before the first call; refuses BY NAME elsewhere    fallback: esm_v5_fwd
    native          .native -- the unified triangle-multiplication package `trimul_native` SEALED under native/pkg/<ver>/: fused         fast class (A: the cuequivariance-rounding
                    LN_in/projection/gate/mask prologue and LN_out/projection/gate(+residual) epilogue as arch-keyed CUDA cubins (sm_90a)     class); bf16 | f32z_bf16 z; c_z, c_hidden in
                    loaded through the CUDA driver around a cuBLAS bmm; per-N tile tables + its own measured CELLS.json inside the payload;        the payload's served widths (v1: 64|128|
                    digests + load check + the package's closed-form vectors as byte gate before the first call; keyed by ARCHITECTURE (one     256|384 pairs on cc 9.0); B <= 64; N 16..4096;
                    payload for every torch / CPython ABI: install() checked per stack); refuses BY NAME elsewhere                           forward only; residual optional; fallback v4
    native:f32in    row native's fp32-resident kernels for fp32 / tf32 CALLERS outside autocast (fp32 z read by K1, bf16 operands, fp32 accumulate,   tolerance class vs the fp32 statement
                    fp32 update / residual sum): the fp32-in / bf16-compute form; refuses bf16 callers by name (they use native)                   (bf16 compute); fallback v4
    native_exact    the same payload's EXACT variant (reference-order LayerNorms / contraction: bit-identical to the cuequivariance op, bf16 z)   EXACT vs the cuequivariance op (vouched per stack)
    esm_v5_fwd:f32in  row esm_v5_fwd's line for fp32 / tf32 callers: z cast ONCE to bf16 at the boundary, the bf16 line computes    TOLERANCE class vs the fp32 stock op
    esm_shapes        opt_core.kernels.trimul_esm_shapes bound by name (the ESM-family pair line re-tiled per width with its own measured launch cells:
                      (64,64) (64,128) (128,128) (256,256) (384,384 native, no padding), B 1, residual fused or absent; cc 9.0 descriptor / cc 8.0 and
                      triton-3.3 pointer K1 cells)                                                                  fast class; bf16 | f32z_bf16 z
    esm_shapes:f32in  the same kernels' fp32-in / bf16-compute / fp32-out cells for fp32 | tf32 trunks                            TOLERANCE class (bf16 operands)
                    the update, the update is cast back to z's dtype (residual added in fp32 when asked) -- a named tolerance-class    (numerics stated in the table); cc >= 8.0,
                    lever for fp32-resident engines whose identity band admits a bf16 pair update; never chosen by a tier word         c 128 | 256, forward only; fallback: v4 / cueq
    esm_k1ptr         .esm_k1ptr -- the sub-package's POINTER-load K1 line forced (the same planes byte for byte as its descriptor K1;    fast class; bf16 | f32z_bf16 z; cc >= 8.0,
                      one instruction stream on triton-3.3 and triton-3.7 stacks); c_z == c_hidden in 128 | 256 | 384, B 1        triton >= 3.3; fallback: v4 / cueq
    esm_k1ptr:f32in   the same for fp32 | tf32 trunks (fp32 in / out, bf16 tensor-core operands)                                    TOLERANCE class (bf16 operands)
    of3_form        kernels/trimul/of3_form.py: the OpenFold-family module's INFERENCE statement (LayerNorm / Linear in the module's dtype rules, sigmoid-then-product
                    gates, cuBLAS contraction in upstream's 256-column blocks) issued whole-tensor, LayerNorms on the core's exactln row  EXACT vs that module (bf16 template + trunk
                    classes measured); tolerance vs the cuequivariance op; forward only; any c_z / c_hidden
    af3t_form       kernels/trimul/af3t_form.py: the AF3-family (xfold-form) module statement (fused row LayerNorm, ONE interleaved dual projection, mask-then-gate,
                    one batched contraction, channels-last LayerNorm) issued whole-tensor in measured layouts                            EXACT vs that module (measured
                    classes); tolerance vs the cuequivariance op and vs of3_form; forward only; c_hidden == c_z
    cueq            named STOCK row: the kit's stock callable (``stock=``) or cuequivariance_torch imported here      backward: yes
    torch_math      named STOCK row for images without that library: the module statements in torch ops             backward: yes

Selection (pure; no framework import)::

    sel = select(cc, dtype, c_z, c_hidden, n_tokens, direction="outgoing", *, word, residency=None, backward=False, prefer=None, stack=None)

``word`` is REQUIRED.  A row name serves exactly that row with the row's own launch cell for the device -- what a kit binding that row today
already runs, byte for byte (the default rule: nothing a kit gets today changes unless the kit asks by word).  ``"fast"`` / ``"exact"`` /
``"big"`` are the OPT-IN tier words: the cell's measured winner of that tier (``prefer=(row, ...)`` narrows the tier to the caller's own
carried rows, in its order).  ``residency="fp32"`` with a bf16 compute dtype selects the fp32-resident-z cells (``f32z_bf16``: the trunks that
keep an fp32 pair tensor under bf16 autocast); ``dtype="tf32"`` selects the fp32-with-TF32 cells.  A row outside its envelope raises
``Refusal(kind, row, fallback)`` BY NAME -- never a silent substitute.

Serving (imports torch and the selected row only here)::

    out = triangle_multiplication(z, mask, direction="outgoing", weights=w10, word="fast", prefer=("v4", "tmk3_fast"), residual=False, cache=my_dict)

``cache`` (a dict the caller holds per weight set, e.g. on its module) keeps the row's packed weights and workspaces between calls (keys carry
the weight tensors' identity, so one dict shared across modules re-packs rather than serving another module's pack).  ``stock=`` is the kit's own
stock callable for the ``cueq`` / ``ef2_cueq_tiles`` rows (else ``cuequivariance_torch.triangle_multiplicative_update`` is imported here).
"""
import json
import os
import re
from collections import namedtuple

from ... import cell_census as _CENSUS                            # stdlib-only: the coverage census (one record per decided call class; pure observation)

V4_C64_H128 = (64, 128)          # the one c_hidden != c_z shape v4 is measured at (bf16, cc 9.0 and 8.0, 400-2048 tokens, both directions): served through the carried
                                 # package's KERNEL face with the cell's launch cells -- its generic face admits c_z 128 / 256 only, its kernels are written in separate c_z / c_hidden
ROW_NAMES = ("v4", "tmk3_exact", "tmk3_fast", "tx_sm90a", "tx_sm90a_exact", "ef2_fused", "ef2_cueq_tiles", "esm_v5_fwd", "esm_v61", "native", "native_exact", "native:f32in", "esm_v5_fwd:f32in", "esm_shapes", "esm_shapes:f32in", "esm_k1ptr", "esm_k1ptr:f32in", "of3_form", "af3t_form", "cueq", "torch_math")
TIER_WORDS = ("fast", "exact", "big")
EF2_FUSED_WIDTHS = (128, 256)                                       # row ef2_fused: the widths its fused fwd+bwd kernels are written and measured at (c 64: illegal address >= 1536 tokens; c 384: no compile)
V4_C64_TRITON_MIN = (3, 4)                                          # row v4 at (c_z, c_hidden) = (64, 128): its launch cells are tensor-descriptor cells (tl.make_tensor_descriptor, triton >= 3.4 per the package; 3.3 measured failing)
V4_DESC_TRITON_MIN = (3, 4)                                         # row v4's tensor-descriptor LAUNCH CELLS (K1 impl 'tma' / 'tma2', K3 impl 'tma') are written on tl.make_tensor_descriptor +
V4_DESC_IMPLS = ("tma", "tma2", "desc", "descriptor")               # triton.set_allocator (triton >= 3.4; torch 2.7.1 ships triton 3.3.1, which has neither): a column's descriptor cell is never handed
                                                                    # to a stack whose stated triton cannot build it (v4_table_cell: the v4 package's own cell for its (capability, triton) part
                                                                    # serves there -- its table's '<cc>|3.3' row, pointer kernels -- said in the Selection's reason); the serving call holds the
                                                                    # same line for a stack it could not read (_serve_v4: the package cell by name, never an AttributeError inside the launch)
STOCK_ROWS = ("cueq", "torch_math")
EXACT_ROWS = ("tmk3_exact", "tx_sm90a_exact", "native_exact", "ef2_cueq_tiles", "of3_form", "af3t_form")   # exact vs WHAT is each row's exact_vs: the cuEquivariance op for the first three, the OpenFold-family module statement for of3_form
MODULE_EXACT_ROWS = ("of3_form", "af3t_form")                  # exact-class rows whose statement is an engine MODULE's own arithmetic (not the stock library op): the exact
                                                               # tier's value ONLY under their form key (select(word="exact", form=...)); row words stay the opt-in / measurement surface
FORMS = {"of3_module": "of3_form", "af3t_module": "af3t_form"}  # form word -> the row that IS that statement; cells "<cell key>+<form>" record where it measured bitwise to the module;
                                                               # no form (the default) = the stock library op's exact tier, unchanged; also spelt word="exact+<form>"
FORM_OF_ROW = {v: k for k, v in FORMS.items()}                  # a module-exact row is admitted ONLY where its form's vouch table records the (stack, shape)
PROVE_SUFFIX = ":prove"                                        # word "exact+<form>:prove": shapes outside the vouch table proven PER CALL SHAPE against the caller's module= callable


def vouched(n_tokens, rec):
    """The vouch ENTRY for this exact token count on this form cell's stack record, else None.  `vouched.N` = explicit sizes measured bitwise to
    the module offline (entry {"N": n, "config": <the cell's config>}); `vouched.N_class` = a class (or a list of classes, first match wins)
    {mod, rem, min, max[, except][, config][, measured]} measured DENSELY (every member size in [min, max] compared offline); a class may name
    the layout `config` it was measured with (e.g. every N -> the module's own call sequence; N % 16 == 0 -> the fast layout).  Nothing else is
    admitted: a proof at one shape never licenses another (cuBLAS / the LayerNorm kernels choose code paths per problem shape)."""
    v = (rec or {}).get("vouched") or {}
    n = int(n_tokens)
    if n in set(int(x) for x in v.get("N", ())):
        return {"N": n, "config": v.get("N_config")}
    cs = v.get("N_class") or []
    if isinstance(cs, dict):
        cs = [cs]
    for c in cs:
        if int(c.get("min", 0)) <= n <= int(c.get("max", -1)) and n % int(c.get("mod", 1)) == int(c.get("rem", 0)) and n not in set(int(x) for x in c.get("except", ())):
            return c
    return None


def _vouch_refusal(form, frow, stock_row, ccw, prec, c_z, c_hidden, n_tokens, dirw, stack, why):
    return Refusal("form_vouch_not_recorded(shape=%s|%s|C%d|H%d|N=%d|%s,stack=%s%s)" % (ccw, prec, int(c_z), int(c_hidden), int(n_tokens), dirw, stack, (";" + why) if why else ""),
                   frow, stock_row)


def form_vouch(form, cc, dtype, c_z, c_hidden, n_tokens, direction, *, stack, residency=None, backward=False, tf32=None, has_cueq=None):
    """(fkey, fcell, rec) of the form cell whose stack record vouches this exact (stack, precision, c, H, N, direction) (rec carries
    `vouch_entry`: the matching vouch entry, whose `config` -- when named -- is the layout to serve at this N) -- else raises
    Refusal `form_vouch_not_recorded(shape=..., stack=...)` (kind) toward the stock row: the caller keeps its module for this class."""
    frow = FORMS[form]
    stock_row = "torch_math" if (has_cueq is False or int(c_hidden) != int(c_z)) else "cueq"
    ccw, prec, dirw = cc_word(cc), precision_word(dtype, residency, tf32), direction_word(direction)
    fam = form_family(ccw, prec, int(c_z), int(c_hidden), dirw, "fwdbwd" if backward else "fwd", form)
    for sz in sorted(sz for sz in fam if sz >= int(n_tokens)):                     # the size cells at or above N, smallest first: the first whose record vouches N
        fcell = table()["form_cells"][fam[sz]]
        rec = (fcell.get("stacks") or {}).get(stack) if stack else None
        ent = vouched(n_tokens, rec) if rec is not None else None
        if ent is not None:
            rec = dict(rec); rec["vouch_entry"] = ent                                   # the matching entry (its `config`, when named, is the layout served at this N)
            return fam[sz], fcell, rec
    if not fam:
        why = "no %s cell at this precision / width / direction" % form
    elif not stack or not any(stack in (table()["form_cells"][k].get("stacks") or {}) for k in fam.values()):
        why = "stack not vouched (vouched: %s)" % ",".join(sorted({st for k in fam.values() for st in (table()["form_cells"][k].get("stacks") or {})}))
    elif int(n_tokens) > max(fam):
        why = "N>%d: beyond the form's vouched envelope (its largest size cell) on this stack" % max(fam)
    else:
        why = "N not vouched on this stack"
    raise _vouch_refusal(form, frow, stock_row, ccw, prec, c_z, c_hidden, n_tokens, dirw, stack, why)

OF3_FORM_DTYPES = ("bf16", "f32z_bf16", "fp32", "tf32")        # row of3_form: the module's LayerNorm / Linear dtype rules cover these; fp16 refused by name
BACKWARD_ROWS = ("ef2_fused", "ef2_cueq_tiles", "cueq", "torch_math")
NEEDS_ESM_IMAGE = ("ef2_fused", "ef2_cueq_tiles")                # esm_v5_fwd / esm_v61 are NOT here: they carry their kernels and import no model package
ESM_V5_C = (64, 128, 256)                                        # row esm_v5_fwd: c_hidden == c_z in these widths (256 = the engine's own; 128 = the cofolding trunks')
ESM_V5_N_MIN = 16                                              # the engine's eligibility floor for this line
ESM_V5_TRITON_MIN = (3, 6)                                     # tensor-descriptor loads (tl.make_tensor_descriptor + triton.set_allocator); 3.6 and 3.7 measured
ESM_V61_C = (256,)                                             # row esm_v61: the sealed package's width (c_z == c_hidden == 256)
ESM_V61_CC = "9.0"                                             # its K3 is an sm_90a cubin: capability 9.0 devices only (an architecture fact, not an ABI key)
NATIVE_ROWS = ("native", "native_exact", "native:f32in")                     # the sealed trimul_native payload (kernels/trimul/native/pkg/<ACTIVE_PKG>/): fast | exact variant of ONE face;
NATIVE_WDICT_KEY = "_native_weights"                                        # caller-cache key of the stable weights dict handed to the payload (its per-call fast path keys on id(weights))
NATIVE_OUT_CAST = "native_out_cast"                            # cache key prefix (NATIVE_OUT_CAST, c_z, c_hidden, residual) -> dtype: a cast the row adapter added to honour the out-dtype contract
                                                               # envelope = the payload's build record (arch-keyed cubins) + its vectors' device classes, read by .native.admits (pure)
PRECISIONS = ("bf16", "f32z_bf16", "fp32", "tf32")
WEIGHT_KEYS = ("ln_in_w", "ln_in_b", "w_ag", "w_ap", "w_bg", "w_bp", "ln_out_w", "ln_out_b", "w_o", "w_og")
BIAS_KEYS = ("b_ag", "b_ap", "b_bg", "b_bp", "b_o", "b_og")          # optional projection biases: row v4 and torch_math take them; every other row refuses BY NAME
def _tx_prebuilt_keys():
    """The full ABI keys the carried tx package ships a binary for (prebuilt/manifest.json 'binaries'): torch<version>-<CPython SOABI>-sm90."""
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tx_sm90a", "prebuilt", "manifest.json")) as fh:
            return tuple(sorted(json.load(fh).get("binaries", {})))
    except (OSError, ValueError):
        return ()


TX_PREBUILT_ABIS = _tx_prebuilt_keys()
CELLS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TRIMUL_CELLS.json")

Selection = namedtuple("Selection", "row word cell size_measured stack cls exact_vs backward capture_safe x_stock fallback reason config")


class Refusal(Exception):
    """A row cannot serve this call: ``kind`` is the word for the kit's LEVER line, ``row`` the row that refused, ``fallback`` the row the table
    names instead (the kit decides whether to take it -- nothing is substituted here)."""

    def __init__(self, kind, row=None, fallback=None):
        Exception.__init__(self, "%s%s%s" % (kind, (" [row %s]" % row) if row else "", (" -> fallback row %s" % fallback) if fallback else ""))
        self.kind, self.row, self.fallback = kind, row, fallback


_TABLE = {}


def table():
    """The parsed TRIMUL_CELLS.json (cached)."""
    if "t" not in _TABLE:
        with open(CELLS_PATH, encoding="utf-8") as fh:
            _TABLE["t"] = json.load(fh)
    return _TABLE["t"]


# ------------------------------------------------------------------------------------------------------------------------------ words
def cc_word(cc):
    """'9.0' from (9, 0) | '9.0' | 9.0 | 90."""
    if isinstance(cc, (tuple, list)):
        return "%d.%d" % (int(cc[0]), int(cc[1]))
    s = str(cc)
    if s.isdigit() and len(s) >= 2:
        return "%s.%s" % (s[:-1], s[-1])
    return "%.1f" % float(s)


def precision_word(dtype, residency=None, tf32=None):
    """The table's precision segment from a dtype word / torch dtype (+ residency of z, + TF32 state for fp32)."""
    s = str(dtype).replace("torch.", "").lower()
    s = {"bfloat16": "bf16", "half": "fp16", "float16": "fp16", "float32": "fp32", "float": "fp32", "f32": "fp32"}.get(s, s)
    if s in ("bf16", "fp16"):
        r = str(residency or "").replace("torch.", "").lower()
        if r in ("fp32", "float32", "f32", "float"):
            return "f32z_bf16" if s == "bf16" else "f32z_fp16"
        return s
    if s == "fp32" and tf32:
        return "tf32"
    return s


def direction_word(direction):
    d = str(direction).lower()
    if d in ("outgoing", "out"):
        return "out"
    if d in ("incoming", "in"):
        return "in"
    if d in ("outin", "block", "both"):
        return "outin"
    raise ValueError("direction must be outgoing | incoming (got %r)" % (direction,))


_FAMILIES = {}


def cell_family(cc, prec, c_z, c_hidden, dirw, pas):
    """{size: key} of the measured cells of one (cc, precision, C, H, dir, pass) family (memoised)."""
    fk = (cc, prec, int(c_z), int(c_hidden), dirw, pas)
    out = _FAMILIES.get(fk)
    if out is None:
        pre = "%s|%s|C%d|H%d|" % (cc, prec, int(c_z), int(c_hidden))
        suf = "|%s|%s" % (dirw, pas)
        out = {}
        for k in table()["cells"]:
            if k.startswith(pre) and k.endswith(suf):
                out[int(k.split("|")[4][3:])] = k
        if out:                                                          # memoised when non-empty (a family that gains its first cell later in the process is seen)
            _FAMILIES[fk] = out
    return out


def cell_key(cc, dtype, c_z, c_hidden, n_tokens, direction="outgoing", *, residency=None, backward=False, tf32=None):
    """(key | None, size_measured, note): the cell serving this call.  The smallest measured size >= n_tokens of the family, else the largest
    (note 'beyond_measured'); a per-direction query falls to the 'outin' family (the ESM-form pair block) when only that was measured."""
    ccw, prec, dirw = cc_word(cc), precision_word(dtype, residency, tf32), direction_word(direction)
    pas = "fwdbwd" if backward else "fwd"
    fam = cell_family(ccw, prec, c_z, c_hidden, dirw, pas)
    note = ""
    if not fam and dirw != "outin":
        fam = cell_family(ccw, prec, c_z, c_hidden, "outin", pas)
        note = "outin_family" if fam else ""
    if not fam:
        return None, False, "no_cell:%s|%s|C%s|H%s|%s|%s" % (ccw, prec, c_z, c_hidden, dirw, pas)
    n = int(n_tokens)
    ge = sorted(s for s in fam if s >= n)
    if ge:
        return fam[ge[0]], (ge[0] == n), note
    big = max(fam)
    return fam[big], False, (note + ";" if note else "") + "beyond_measured(N<=%d)" % big


# ------------------------------------------------------------------------------------------------------------- unmeasured capabilities
# A device capability the table has NO column for (10.0 / 10.3 / 12.0 ..., or an 8.x part other than 8.0) INHERITS the nearest measured,
# architecture-compatible capability's cells for the FAST and BIG tiers -- restricted to rows that are portable Triton / CUDA-source rows
# compiled on the device at first use (v4, the tmk3 pair) plus the native payload where ITS build record carries that architecture (its own
# admits says); never an sm_90a cubin / prebuilt row of another architecture (tx_sm90a, esm_v61: their admits refuse by name off 9.0), never
# the library op while such a row admits.  The EXACT tier on an unmeasured capability is the library / stock op BY NAME (no byte vouch exists
# on that capability).  The Selection's reason and the census carry ``inherited_cc:unmeasured(<cc>-><measured cc>)`` + the column read.
BIG_EXCLUDED_ROWS = ("esm_v5_fwd", "esm_v5_fwd:f32in", "esm_v61", "esm_shapes", "esm_shapes:f32in", "esm_k1ptr", "esm_k1ptr:f32in")   # the ESM family: never a big
                                                                          # answer at any N (TRIMUL_CELLS policy.big_excluded; program_evidence.esm_big) -- select() enforces it
WITNESS_BIG_MAX_N = 2048                                                # a binder's first-call WITNESS statement (an N^2-class reference computation) under word=big: skipped BY
WITNESS_SKIP_TOKEN = "witness:skipped_big_large_n"                       # NAME above this many tokens (witness_policy(); the census token a binder prints instead of allocating it)


def witness_policy(word, n_tokens):
    """A binder that witnesses a row's first call against a reference STATEMENT asks this first: under the tier word big above
    WITNESS_BIG_MAX_N tokens the answer is ('skip', WITNESS_SKIP_TOKEN) -- the statement is an N^2-class allocation a memory-bound caller must
    not make at its ceiling; otherwise ('run', '').  The provider itself runs no statement on the tier words."""
    try:
        n = int(n_tokens)
    except (TypeError, ValueError):
        n = 0
    if str(word or "") == "big" and n > WITNESS_BIG_MAX_N:
        return "skip", WITNESS_SKIP_TOKEN
    return "run", ""
INHERIT_PORTABLE_ROWS = ("v4", "tmk3_fast", "tmk3_exact", "native", "native:f32in")


def measured_ccs():
    """The capability words the table has cells for ('8.0', '9.0', ...)."""
    return tuple(sorted({k.split("|")[0] for k in table()["cells"]}, key=lambda w: float(w)))


def _has_family(ccw, fam):
    """True when capability ``ccw`` has >= 1 cell of the family fam = (precision word, c_z, c_hidden, direction word, 'fwd'|'fwdbwd')."""
    prec, C, H, dirw, pas = fam
    return bool(cell_family(ccw, prec, C, H, dirw, pas) or (dirw != "outin" and cell_family(ccw, prec, C, H, "outin", pas)))


def inherited_cc(cc, family=None):
    """The measured capability whose cells an UNMEASURED (capability, family) reads -- decided PER FAMILY KEY: whenever the device's capability has
    no cell FOR THAT FAMILY (whatever it has for other families), the nearest architecture-compatible capability that HAS the family: the same
    major version at or below the device's (8.6 -> 8.0), else the highest measured capability below it (10.3 / 12.0 -> 9.0), else the lowest
    above it.  ``family`` = (precision word, c_z, c_hidden, direction word, 'fwd' | 'fwdbwd'); None = any family (the capability level:
    None when the capability has cells of its own).  None when nothing qualifies."""
    ccw = cc_word(cc)
    try:
        v = float(ccw)
    except ValueError:
        return None
    ms = [m for m in measured_ccs() if m != ccw and (family is None or _has_family(m, family))]
    if family is None and ccw in measured_ccs():
        return None
    same = [m for m in ms if int(float(m)) == int(v) and float(m) <= v]
    if same:
        return max(same, key=float)
    below = [m for m in ms if float(m) < v]
    if below:
        return max(below, key=float)
    above = [m for m in ms if float(m) > v] if any(float(m) <= v for m in measured_ccs()) else []   # upward only for a device at / above the lowest measured capability
    return min(above, key=float) if above else None


_FORM_FAMILIES = {}


def form_family(ccw, prec, c_z, c_hidden, dirw, pas, form):
    """{size: key} of the form cells (TRIMUL_CELLS.json "form_cells": "<cc>|<precision>|C|H|N<=size|<dir>|<pass>+<form>" -- the shapes where
    the form's row measured bitwise to its engine module).  Separate from cell_family and from "cells": a form cell never enters the default
    (no-form) resolution and no consumer of "cells" sees one."""
    k = (ccw, prec, int(c_z), int(c_hidden), dirw, pas, form)
    fam = _FORM_FAMILIES.get(k)
    if fam is None:
        pre, suf = "%s|%s|C%d|H%d|N<=" % (ccw, prec, int(c_z), int(c_hidden)), "|%s|%s+%s" % (dirw, pas, form)
        fam = {}
        for key in table().get("form_cells", {}):
            if key.startswith(pre) and key.endswith(suf):
                try:
                    fam[int(key[len(pre):].split("|", 1)[0])] = key
                except ValueError:
                    continue
        _FORM_FAMILIES[k] = fam
    return fam


# --------------------------------------------------------------------------------------------------------------------------- admission
V4_N_MIN_DEFAULT = 101                                              # row v4's token floor where the table carries no small-N cert for the class (TRIMUL_CELLS rows.v4.admits.n_min):
                                                                    # at and below 100 tokens the stock library op runs its own small-N torch algorithm -- the stock regime by policy


def v4_n_min(cc, prec, c_z, c_hidden):
    """Row v4's token floor for the call class ``<cc>|<precision>|C<c_z>|H<c_hidden>``: ``TRIMUL_CELLS.json rows.v4.admits.n_min_cells[<class>]`` where
    the small-N measurements vouched the row below 101 tokens on the listed H100 columns
    (bitwise-deterministic, CUDA-graph capturable, inside the family's numerics class and faster than the library op at every measured size),
    else the row's ``n_min`` (101).  A DATA answer: the kernel source, its launch cells and every selection at or above 101 tokens are unchanged."""
    adm = (table().get("rows", {}).get("v4", {}) or {}).get("admits", {}) or {}
    cells = adm.get("n_min_cells") or {}
    v = cells.get("%s|%s|C%d|H%d" % (cc_word(cc), prec, int(c_z), int(c_hidden)))
    try:
        return int(v) if v is not None else int(adm.get("n_min", V4_N_MIN_DEFAULT))
    except (TypeError, ValueError):
        return V4_N_MIN_DEFAULT


def admits(row, cc, dtype, c_z, c_hidden, n_tokens, *, residency=None, backward=False, residual=None, batch=1, abi=None, has_cueq=None,
           has_esm_kernels=None, tf32=None, triton=None):
    """Raise ``Refusal`` when ``row`` cannot serve this envelope (static facts only: capability, dtype, widths, size, backward, residual, ABI,
    library presence when the caller states it); return None when it can.  The serving call re-checks with the tensors in hand."""
    if row not in ROW_NAMES:
        raise ValueError("unknown row %r (rows: %s)" % (row, ", ".join(ROW_NAMES)))
    ccw = cc_word(cc)
    prec = precision_word(dtype, residency)
    C, H, N = int(c_z), int(c_hidden), int(n_tokens)
    if backward and row not in BACKWARD_ROWS:
        raise Refusal("no_backward", row, "ef2_fused" if prec == "bf16" else "cueq")
    if row == "v4":
        if float(ccw) < 8.0:
            raise Refusal("cc:%s<8.0" % ccw, row, "cueq")
        if (C, H) == V4_C64_H128:
            if prec != "bf16":
                raise Refusal("dtype:%s!=bf16(c_z=64,c_hidden=128 verified in bf16 only)" % prec, row, "torch_math")
            if triton is not None and _ver(triton) < V4_C64_TRITON_MIN:
                raise Refusal("triton:%s<3.4(c_z=64,c_hidden=128 runs the tensor-descriptor cells: tl.make_tensor_descriptor)" % triton, row, "torch_math")
        elif C not in (128, 256):
            raise Refusal("c_z:%d" % C, row, "cueq")
        elif H not in (128, 256):
            raise Refusal("d_hidden:%d" % H, row, "cueq")
        elif H != C:
            raise Refusal("c_hidden=%d!=c_z=%d(unverified: the launch cells are (128,128) and (256,256); (64,128) is the one verified c_hidden != c_z shape)" % (H, C), row, "torch_math")
        floor = v4_n_min(ccw, prec, C, H)                          # the row's token floor per (cc, precision, c_z, c_hidden) class FROM THE TABLE (rows.v4.admits.n_min_cells:
        if N < floor:                                               # the classes measured below 101); every other class keeps n_min = 101 and the identical word
            raise Refusal("n<%d:below_n_min" % floor, row, "cueq")
        if prec not in PRECISIONS:
            raise Refusal("dtype:%s" % prec, row, "cueq")
    elif row in ("tmk3_exact", "tmk3_fast"):
        if float(ccw) < 8.0:
            raise Refusal("cc:%s<8.0" % ccw, row, "cueq")
        if H != C:
            raise Refusal("c_hidden=%d!=c_z=%d(TM-K3 planes assume c_hidden == c_z; wrong numbers otherwise)" % (H, C), row, "torch_math")
        if C % 64:
            raise Refusal("c:%d_not_multiple_of_64" % C, row, "cueq")
        if prec not in PRECISIONS:
            raise Refusal("dtype:%s" % prec, row, "cueq")
        if row == "tmk3_exact" and has_cueq is False:
            raise Refusal("import:cuequivariance_ops_torch", row, "torch_math")
    elif row in ("tx_sm90a", "tx_sm90a_exact"):
        fb = "v4" if row == "tx_sm90a" else "tmk3_exact"
        if ccw != "9.0":
            raise Refusal("cc:%s!=9.0(sm_90a binary)" % ccw, row, fb)
        if abi is not None and abi not in TX_PREBUILT_ABIS:
            raise Refusal("no_prebuilt:%s" % abi, row, fb)
        if C != 256 or H != 256:
            raise Refusal("c_z=%d,c_hidden=%d!=256" % (C, H), row, fb)
        if prec != "bf16":
            raise Refusal("dtype:%s!=bf16" % prec, row, fb)
        if N < 101:
            raise Refusal("n<101", row, "cueq")
        if int(batch) != 1:
            raise Refusal("batch>1", row, fb)
    elif row == "ef2_fused":
        if residual is False:
            raise Refusal("needs_residual", row, "cueq")
        if prec != "bf16":
            raise Refusal("dtype:%s!=bf16" % prec, row, "cueq")
        if C % 64 or H != C:
            raise Refusal("c:%d/%d" % (C, H), row, "cueq")
        if C not in EF2_FUSED_WIDTHS:                               # the fused kernels' gated-GEMM launch tables are written and proven at K = 128 | 256 only:
            raise Refusal("c_z:%d(proven at 128|256 only: c 64 faults at >= 1536 tokens, c 384 does not compile)" % C, row, "cueq")   # measured
        if has_esm_kernels is False:
            raise Refusal("import:transformers.models.esmfold2", row, "cueq")
    elif row == "ef2_cueq_tiles":
        if has_cueq is False:
            raise Refusal("needs_cuequivariance", row, "torch_math")
        if prec != "bf16":
            raise Refusal("dtype:%s!=bf16" % prec, row, "cueq")
        if H != C:
            raise Refusal("c_hidden=%d!=c_z=%d(the stock library op takes c_hidden == c_z)" % (H, C), row, "torch_math")
        if has_cueq is False:
            raise Refusal("import:cuequivariance_ops_torch", row, "torch_math")
        if has_esm_kernels is False:
            raise Refusal("import:transformers.models.esmfold2", row, "cueq")
    elif row == "esm_v5_fwd":
        if float(ccw) < 8.0:
            raise Refusal("cc:%s<8.0" % ccw, row, "cueq")
        if prec not in ("bf16", "f32z_bf16"):
            raise Refusal("dtype:%s!=bf16" % prec, row, "v4")
        if H != C:
            raise Refusal("c_hidden=%d!=c_z=%d(the carried planes are written for c_hidden == c_z)" % (H, C), row, "torch_math")
        if C not in ESM_V5_C:
            raise Refusal("c_z:%d(not %s)" % (C, "|".join(str(c) for c in ESM_V5_C)), row, "v4" if C in (128, 256) else "cueq")
        if N < ESM_V5_N_MIN:
            raise Refusal("n<%d" % ESM_V5_N_MIN, row, "cueq")
        if triton is not None and _ver(triton) < ESM_V5_TRITON_MIN:
            raise Refusal("triton:%s<3.6(tl.make_tensor_descriptor)" % triton, row, "v4")
        if int(batch) < 1:
            raise Refusal("batch<1", row, "v4")
    elif row == "esm_v61":
        fb5 = "cueq" if float(ccw) < 8.0 else ("esm_v5_fwd" if (C in ESM_V5_C and H == C and prec in ("bf16", "f32z_bf16")) else ("v4" if C in (128, 256) else "cueq"))
        if prec not in ("bf16", "f32z_bf16"):
            raise Refusal("dtype:%s!=bf16" % prec, row, "v4")
        if H != C:
            raise Refusal("c_hidden=%d!=c_z=%d(the carried planes are written for c_hidden == c_z)" % (H, C), row, "torch_math")
        if C not in ESM_V61_C:
            raise Refusal("c_z:%d(not 256)" % C, row, fb5)
        if ccw != ESM_V61_CC:
            raise Refusal("cc:%s!=9.0(sm_90a cubin)" % ccw, row, fb5)
        if N < ESM_V5_N_MIN:
            raise Refusal("n<%d" % ESM_V5_N_MIN, row, "cueq")
        if triton is not None and _ver(triton) < ESM_V5_TRITON_MIN:
            raise Refusal("triton:%s<3.6(tl.make_tensor_descriptor)" % triton, row, "v4")
        if int(batch) < 1:
            raise Refusal("batch<1", row, fb5)
    elif row in NATIVE_ROWS:
        fbn = _native_fallback(row, ccw, prec, C, H)
        if row == "native:f32in":
            if prec not in ("fp32", "tf32"):
                raise Refusal("dtype:%s(fp32 / tf32 callers: bf16-resident and autocast callers use native)" % prec, row, "native")
        elif prec not in ("bf16", "f32z_bf16"):
            raise Refusal("dtype:%s(bf16 | f32z_bf16: the payload computes in bf16; an fp32 / tf32 caller names native:f32in)" % prec, row, fbn)
        from . import native as _NT                                          # stdlib at import; its admits reads the payload's build / vector records (json) through the carried face's pure admits
        wordn = _NT.admits(ccw, "bf16", C, H, N, "outgoing", residency=("fp32" if prec != "bf16" else None), residual=bool(residual),
                           batch=max(1, int(batch)), variant=("exact" if row == "native_exact" else "fast"))
        if wordn is not None:
            raise Refusal(wordn, row, "cueq" if wordn.startswith("shape:n=") else fbn)
    elif row in ("esm_shapes", "esm_shapes:f32in"):
        fbz = "v4" if (C in (128, 256) and H == C) or (C, H) == (64, 128) else "cueq"
        if float(ccw) < 8.0:
            raise Refusal("cc:%s<8.0" % ccw, row, "cueq")
        if row == "esm_shapes" and prec not in ("bf16", "f32z_bf16"):
            raise Refusal("dtype:%s(fp32 | tf32 callers use esm_shapes:f32in)" % prec, row, "esm_shapes:f32in" if prec in ("fp32", "tf32") else "cueq")
        if row == "esm_shapes:f32in" and prec not in ("fp32", "tf32"):
            raise Refusal("dtype:%s(bf16 callers use esm_shapes)" % prec, row, "esm_shapes" if prec in ("bf16", "f32z_bf16") else "cueq")
        try:
            S = __import__("opt_core.kernels.trimul_esm_shapes", fromlist=["supported"])
        except ImportError as e:
            raise Refusal("needs:kernels.trimul_esm_shapes(import:%s)" % getattr(e, "name", "?"), row, fbz if row == "esm_shapes" else "cueq")
        ok, why = S.supported(ccw, "bf16" if row == "esm_shapes" else "fp32", C, H, N, batch=int(batch) if int(batch) >= 1 else 1,
                              descriptor_api=(triton is None or _ver(triton) >= ESM_V5_TRITON_MIN))
        if not ok:
            raise Refusal(why, row, fbz if row == "esm_shapes" else "cueq")
    elif row in ("esm_k1ptr", "esm_k1ptr:f32in"):
        from . import esm_k1ptr as _K1P
        r = _K1P.refusal(row, ccw, prec, C, H, N, batch=batch, triton=triton)
        if r is not None:
            raise Refusal(r[0], row, r[1])
    elif row == "esm_v5_fwd:f32in":
        if float(ccw) < 8.0:
            raise Refusal("cc:%s<8.0" % ccw, row, "cueq")
        if prec in ("bf16", "f32z_bf16"):
            raise Refusal("dtype:%s(bf16 callers use esm_v5_fwd; this row is the fp32-in cast form)" % prec, row, "esm_v5_fwd")
        if prec not in ("fp32", "tf32"):
            raise Refusal("dtype:%s(not fp32|tf32)" % prec, row, "cueq")
        if H != C:
            raise Refusal("c_hidden=%d!=c_z=%d(the carried planes are written for c_hidden == c_z)" % (H, C), row, "torch_math")
        if C not in ESM_V5_C:
            raise Refusal("c_z:%d(not %s)" % (C, "|".join(str(c) for c in ESM_V5_C)), row, "v4" if C in (128, 256) else "cueq")
        if N < 101:
            raise Refusal("n<101(below v4's floor the stock op serves fp32)", row, "cueq")
        if triton is not None and _ver(triton) < ESM_V5_TRITON_MIN:
            raise Refusal("triton:%s<3.6(tl.make_tensor_descriptor)" % triton, row, "v4")
        if int(batch) < 1:
            raise Refusal("batch<1", row, "v4")
    elif row in ("of3_form", "af3t_form"):
        if row == "af3t_form" and H != C:
            raise Refusal("c_hidden=%d!=c_z=%d(the module has one width)" % (H, C), row, "torch_math")
        if float(ccw) < 8.0 and ccw != "0.0":                                    # cuBLAS / ATen statements run anywhere; below sm_80 nothing is measured: named, towards the same statements in torch_math
            raise Refusal("cc:%s<8.0(unmeasured)" % ccw, row, "torch_math")
        if prec not in OF3_FORM_DTYPES:
            raise Refusal("dtype:%s(bf16|f32z_bf16|fp32|tf32)" % prec, row, "torch_math")
        if N < 1 or int(batch) < 1:
            raise Refusal("empty", row, "torch_math")
    elif row == "cueq":
        if H != C:
            raise Refusal("c_hidden=%d!=c_z=%d(the stock library op takes c_hidden == c_z; the engines run their torch path there)" % (H, C), row, "torch_math")
        if has_cueq is False:
            raise Refusal("import:cuequivariance_torch", row, "torch_math")
    return None


def _native_fallback(row, ccw, prec, C, H):
    """The row the table names when a native row refuses: native -> v4 where v4 serves (c 128 | 256, c_hidden == c_z, cc >= 8.0), else the stock op;
    native_exact -> tmk3_exact (the exact class at c_hidden == c_z, c % 64 == 0), else the stock op."""
    try:
        ccf = float(ccw)
    except ValueError:
        ccf = 0.0
    if row == "native_exact":
        return "tmk3_exact" if (H == C and C % 64 == 0 and ccf >= 8.0) else "cueq"
    if row == "native:f32in":
        return "v4" if (C in (128, 256) and H == C and ccf >= 8.0) else ("torch_math" if H != C else "cueq")
    return "v4" if (C in (128, 256) and H == C and ccf >= 8.0 and prec in PRECISIONS) else ("torch_math" if H != C else "cueq")


# ---------------------------------------------------------------------------------------------------------------------------- selection
def _ver(v):
    """(major, minor) from '3.7.0' / '3.6' (unparsable -> (0, 0))."""
    try:
        p = str(v).split("+")[0].split(".")
        return int(p[0]), int(p[1]) if len(p) > 1 else 0
    except ValueError:
        return 0, 0


def _triton_of(stack):
    """'3.7.0' from 'H100:2.12.0+cu130/3.7.0/cueq0.10.0' (None when absent)."""
    try:
        return str(stack).split("/")[1] if stack and "/" in str(stack) else None
    except IndexError:
        return None


def exact_vouched(cell, row, stack):
    """True when ``row``'s exact-class (bitwise) vouch is RECORDED on this very stack for this cell: the cell's class record on that stack starts with
    'bitwise', or cells.<key>.vouched_on[row] lists the stack.  A stack the table never byte-tested the row on (no column of its own, a stack whose
    torch / triton / library versions differ from every column) is NOT vouched: under the tier word ``exact`` the row is refused by name there and the
    word falls to the stock op of the stack (row words are the caller's own statement and are unaffected; fast / big are tolerance classes)."""
    if not cell or not stack:
        return False
    c_ = ((cell.get("class") or {}).get(row) or {}).get(stack)
    return (isinstance(c_, str) and c_.startswith("bitwise")) or stack in (((cell.get("vouched_on") or {}).get(row)) or ())


BATCH_VOUCH_KEY = "vouched_on_batched"                              # cells.<key>.vouched_on_batched[row][stack] = [B, ...] | "any": the leading-batch extents > 1 at which
                                                                    # `row` measured bitwise to the stock op's OWN BATCHED call on that stack (none recorded today)
BATCH_NOTE = "exact_batch_unvouched"                                # the Selection reason / census note word when the batched vouch is absent (the stock op serves BY NAME)


def batch_extent(batch):
    """The leading batch extent a caller states (``batch=``: the product of z's dimensions in front of ``[N, N, c]`` -- 1 for a 3-d z): an int >= 1,
    or None when unstated (a planning call that names no tensor)."""
    if batch is None or isinstance(batch, bool):
        return None
    try:
        b = int(batch)
    except (TypeError, ValueError):
        return None
    return b if b >= 1 else 1


def exact_batch_vouched(cell, row, stack, batch):
    """True when ``row`` may serve a call of leading batch extent ``batch`` under the tier word ``exact`` as far as BATCH is concerned: the extent is
    unstated or 1 (the ordinary vouch, ``exact_vouched``, was taken on ONE pair tensor ``[1, N, N, c]``), or the cell RECORDS the row's bitwise vouch
    against the stock op's own BATCHED call at that extent on this very stack (``cells.<key>.vouched_on_batched[row][stack]`` lists the extent, or is
    the word ``"any"``).  The library op's batched call (``[B, N, N, c]``, B > 1) is NOT B calls of one sample byte for byte (it contracts the batch in
    another order), so a batch-1 vouch never licenses a batched layout: absent the record the exact word is the stock op BY NAME
    (``exact_batch_unvouched(<row>-><stock>:B<b>)``).  No table entry carries the record today -- every batched exact call is the stock op."""
    b = batch_extent(batch)
    if b is None or b <= 1:
        return True
    if not cell or not stack:
        return False
    rec = ((cell.get(BATCH_VOUCH_KEY) or {}).get(row) or {}).get(stack)
    if rec is None:
        return False
    if isinstance(rec, str):
        return rec == "any"
    try:
        return b in set(int(x) for x in rec)
    except (TypeError, ValueError):
        return False


def exact_envelope(cc, dtype, c_z, c_hidden, direction, row, stack=None, *, residency=None, backward=False, tf32=None):
    """(largest bucket N of the family, largest bucket N whose cell RECORDS ``row``'s bitwise vouch on ``stack`` | None): the size envelope inside
    which the exact tier may serve ``row`` on that stack.  Beyond the family's largest bucket no byte test exists at any size >= N on any stack: under
    the tier word ``exact`` an exact-class row is refused by name there (``exact_beyond_vouched_envelope:N>..``) and the word is the stock op --
    fast / big keep serving the nearest cell (tolerance classes).  No key lies below a family's smallest bucket in this grammar (N<=k covers every
    smaller N), so there is no lower edge to guard."""
    ccw, prec, dirw = cc_word(cc), precision_word(dtype, residency, tf32), direction_word(direction)
    pas = "fwdbwd" if backward else "fwd"
    fam = cell_family(ccw, prec, c_z, c_hidden, dirw, pas) or (cell_family(ccw, prec, c_z, c_hidden, "outin", pas) if dirw != "outin" else {})
    if not fam:
        return None, None
    cells = table()["cells"]
    vmax = max((n for n, k in fam.items() if stack and exact_vouched(cells.get(k), row, stack)), default=None)
    return max(fam), vmax


def _stack_parts(st):
    """('H100', '2.12.0', 'cu130', (3, 7, 0), ('cueq', (0, 11, 1)) | ('nocueq', ())) from 'H100:2.12.0+cu130/3.7.0/cueq0.11.1' (None when unparsable)."""
    try:
        card, rest = str(st).split(":", 1)
        torch_w, trit_w, lib_w = rest.split("/")[:3]
        tv, _, cu = torch_w.partition("+")
        nums = lambda v: tuple(int(x) for x in re.findall(r"\d+", v)[:3])
        lib = ("nocueq", ()) if lib_w.startswith("nocueq") else (re.sub(r"[\d.]+$", "", lib_w), nums(lib_w))
        return card, tv, cu, nums(trit_w), lib
    except (ValueError, AttributeError):
        return None


def _vdist(a, b):
    """Distance between two version tuples (major-weighted); a missing side is far."""
    if not a or not b:
        return 10 ** 6
    a, b = (tuple(a) + (0, 0, 0))[:3], (tuple(b) + (0, 0, 0))[:3]
    return abs(a[0] - b[0]) * 10 ** 4 + abs(a[1] - b[1]) * 10 ** 2 + abs(a[2] - b[2])


_CC_REF = {}


def cc_reference_column(cc):
    """The reference column of a cc: the stack word most cells of that cc name as their ref_stack (the cc's reference stack)."""
    ccw = cc_word(cc)
    if ccw not in _CC_REF:
        cnt = {}
        for k, c in table()["cells"].items():
            if k.startswith(ccw + "|") and c.get("ref_stack"):
                cnt[c["ref_stack"]] = cnt.get(c["ref_stack"], 0) + 1
        _CC_REF[ccw] = max(sorted(cnt), key=cnt.get) if cnt else None
    return _CC_REF[ccw]


def _column_rank(cell, st, word):
    """The cell's rows ranked for a tier word's step-aside on column ``st``: the rows TIMED on that column, fastest first (the next word of that
    column); a thin or inherited column (fewer than two fast-tier rows timed on it) ranks by the column it was measured on (its inherited_r*
    record's measured_on, else the cc's reference column, else the cell's); then the cell's fast_order rows timed on the cell's reference column."""
    ms = cell.get("ms") or {}
    timed = {r: v[st] for r, v in ms.items() if st in v}
    if sum(1 for r in timed if r not in STOCK_ROWS) < 2:
        src = None
        for blk in ("inherited_r3", "inherited_r2", "inherited_r1"):
            rec = (cell.get(blk) or {})
            rec = rec.get("%s|%s" % (st, word)) or rec.get("%s|fast" % st) if isinstance(rec, dict) else None
            if isinstance(rec, dict) and rec.get("measured_on"):
                src = rec["measured_on"]
                break
        src = src or _cell_cc_ref(cell) or cell.get("ref_stack")
        timed = {r: v[src] for r, v in ms.items() if src in v} or timed
    ranked = sorted(timed, key=lambda r: (timed[r], r))
    ref = cell.get("ref_stack")
    rest = [r for r in cell.get("fast_order", []) if r not in ranked and ref in ms.get(r, {})]
    return ranked + rest


def _cell_cc_ref(cell):
    """The cc reference column if this cell lists it (else None); the cc is read off the cell's columns' card word."""
    cols = list(cell.get("fast_per_stack") or {})
    for ccw in ("9.0", "8.0"):
        r = cc_reference_column(ccw)
        if r and r in cols:
            card = r.split(":")[0]
            if any(c.split(":")[0] == card for c in cols):
                return r
    return None


def sibling_column(cell, stack):
    """The column of ``cell`` a stack word reads for the FAST / BIG tiers (and a row word's numbers): ``(column, rule)`` with rule
    ``'listed'`` (the stack has its own column), ``'sibling:torch'`` (rule 1: a listed column of the SAME torch version + cuda tag, another
    library / triton minor -- the nearest triton, then the nearest library version), ``'sibling:minor'`` (rule 2: the same torch major.minor --
    the same cuda tag first, then the nearest patch / triton / library), else ``'reference'`` (rule 3: the cell's reference column = the
    cell-level word).  The fast / big rows are tolerance-class and every sealed binary is byte-identical across stacks, so a sibling column's
    word is a safe inheritance; a row that cannot serve THIS process still refuses by name and the word steps aside inside that column.  The
    EXACT tier never inherits: its vouch is stack-specific (``exact_vouched``)."""
    ref = (_cell_cc_ref(cell) or cell.get("ref_stack")) if cell else None      # rule 3: the cc's reference column when the cell lists it, else the cell's own
    cols = list((cell or {}).get("fast_per_stack", {}))
    if not cell:
        return ref, "reference"
    if not stack:
        return cell.get("ref_stack"), "reference"                          # a planning call (no stack stated) reads the cell's reference column as before
    if stack in cols:
        return stack, "listed"
    q = _stack_parts(stack)
    if q is None:
        return ref, "reference"
    qcard, qtv, qcu, qtrit, qlib = q
    parsed = [(c, _stack_parts(c)) for c in cols]
    parsed = [(c, pp) for c, pp in parsed if pp is not None]
    libd = lambda pl: (0 if pl[0] == qlib[0] else 10 ** 5) + (_vdist(pl[1], qlib[1]) if pl[1] and qlib[1] else 0)
    same_torch = [(0 if pp[0] == qcard else 1, _vdist(pp[3], qtrit), libd(pp[4]), c) for c, pp in parsed if pp[1] == qtv and pp[2] == qcu]
    if same_torch:
        return sorted(same_torch)[0][-1], "sibling:torch"
    mm = lambda v: tuple(int(x) for x in re.findall(r"\d+", v)[:2])
    pat = lambda v: (tuple(int(x) for x in re.findall(r"\d+", v)[:3]) + (0, 0, 0))[:3]
    same_minor = [(0 if pp[0] == qcard else 1, 0 if pp[2] == qcu else 1, _vdist(pat(pp[1]), pat(qtv)), _vdist(pp[3], qtrit), libd(pp[4]), c)
                  for c, pp in parsed if mm(pp[1]) == mm(qtv)]
    if same_minor:
        return sorted(same_minor)[0][-1], "sibling:minor"
    return ref, "reference"


def _measured_class(cell, row, st, cls):
    """The class the table MEASURED for this row on this stack ('bitwise...' | 'tol(...)' | 'stock'), else the row's class word."""
    m = (cell or {}).get("class", {}).get(row, {}).get(st) if cell else None
    return m or cls


def _row_facts(row):
    r = table()["rows"].get(row, {})
    return r.get("class"), r.get("exact_vs"), bool(r.get("backward")), bool(r.get("capture_safe", True)), r.get("fallback")


def _cell_config(cell, row, st):
    """The launch cell the table records for ``row`` at this cell: config_by_stack.<row>.<stack> when the sweep ran on that stack, else config.<row>."""
    if not cell:
        return None
    by = ((cell.get("config_by_stack") or {}).get(row) or {})
    if st in by:
        return by[st]
    return (cell.get("config") or {}).get(row)


def v4_desc_cell(cfg):
    """True when a v4 launch cell ``{"k1": {...}, "k3": {...}}`` names a tensor-descriptor kernel (K1 impl ``tma`` / ``tma2``, K3 impl ``tma``:
    written on ``tl.make_tensor_descriptor`` + ``triton.set_allocator``, triton >= 3.4); False for pointer cells (no ``impl`` / ``ptr``) and None."""
    if not isinstance(cfg, dict):
        return False
    return any(isinstance(cfg.get(k), dict) and str(cfg[k].get("impl", "ptr")).lower() in V4_DESC_IMPLS for k in ("k1", "k3"))


def v4_table_cell(cell, st, triton):
    """``(config, note)``: the table's v4 launch cell on column ``st`` FOR A CALLER WHOSE STACK RUNS ``triton`` -- the recorded cell unchanged
    (note ``''``), except a tensor-descriptor cell (:func:`v4_desc_cell`) read by a stack whose stated triton predates the descriptor API
    (``V4_DESC_TRITON_MIN``; torch 2.7.1 images run triton 3.3.1: no ``tl.make_tensor_descriptor``): that cell cannot build there -- it is the
    cell of ANOTHER column (a sibling / reference column measured on triton 3.6 / 3.7) -- so it is not inherited: config ``None`` = the v4
    package's own cell for its (capability, triton) part serves (its table's ``'<cc>|3.3'`` row: pointer kernels, the cell v4 is measured with on
    that stack), note ``v4_cell=package(triton:<v><3.4:tensor_descriptor_cell_of:<st>)``.  ``triton`` None (a planning call that states no stack)
    or unparsable: unchanged (the serving call holds the same line with the triton in hand: ``_serve_v4``)."""
    cfg = _cell_config(cell, "v4", st)
    v = _ver(triton) if triton is not None else (0, 0)
    if v == (0, 0) or v >= V4_DESC_TRITON_MIN or not v4_desc_cell(cfg):
        return cfg, ""
    return None, "v4_cell=package(triton:%s<3.4:tensor_descriptor_cell_of:%s)" % (triton, st)


_SKIP_RE = re.compile(r"skip ([^(;\s]+)\(([^)]*)\)")


def cueq_torch_threshold():
    """The token count AT OR UNDER which cuEquivariance's triangle_multiplicative_update serves by its own torch statement instead of its kernel
    (the library's CUEQ_TRIMUL_FALLBACK_THRESHOLD, default 100) -- read as the library reads it, so a process that moves the threshold moves
    the exact word's stock answer with it."""
    try:
        return int(os.environ.get("CUEQ_TRIMUL_FALLBACK_THRESHOLD", "100"))
    except ValueError:
        return 100


def _exact_stock_row(stock_row, stack, has_cueq, c_z, c_hidden):
    """THIS process's stock row for the exact word: ``torch_math`` where the library cannot serve (c_hidden != c_z, has_cueq False, or a
    stack word naming a library-less stack: ``.../nocueq``), else ``cueq`` -- the op the engine's stock module calls on a library stack.
    (The exact word never serves another column's stock row; see select().)"""
    if int(c_hidden) != int(c_z) or has_cueq is False:
        return "torch_math"
    if has_cueq is None and stack:
        parts = _stack_parts(norm_stack(stack)) if "norm_stack" in globals() else _stack_parts(stack)
        if parts is not None and parts[4] and str(parts[4][0]).startswith("nocueq"):
            return "torch_math"
    return "cueq" if stock_row == "cueq" or has_cueq else stock_row


def select(cc, dtype, c_z, c_hidden, n_tokens, direction="outgoing", *, word, residency=None, backward=False, prefer=None, stack=None,
           tf32=None, has_cueq=None, config=None, abi=None, exclude=None, form=None, batch=None):
    """The row + facts for this call (see ``_select``); the decision is recorded ONCE per call class in :mod:`opt_core.cell_census` (pure
    observation: the Selection / Refusal is decided first and returned unchanged).  A step-aside re-selection (``exclude``) is recorded by the
    serving call, which holds the refusal's name.  ``batch``: the call's leading batch extent (the product of z's dimensions in front of [N, N, c])
    -- read ONLY by the tier word ``exact`` (an exact-class row serves a batched layout, B > 1, only where the cell records its batched
    vouch on this stack: ``exact_batch_vouched``); unstated or 1, and every other word: the decision is unchanged."""
    kw = dict(word=word, residency=residency, backward=backward, prefer=prefer, stack=stack, tf32=tf32, has_cueq=has_cueq, config=config, abi=abi,
              exclude=exclude, form=form, batch=batch)
    try:
        sel = _select(cc, dtype, c_z, c_hidden, n_tokens, direction, **kw)
    except Refusal as e:
        if not exclude:
            _census(cc, dtype, c_z, c_hidden, n_tokens, direction, kw, None, e)
        raise
    if not exclude:
        _census(cc, dtype, c_z, c_hidden, n_tokens, direction, kw, sel, None)
    return sel


def _census_key(cc, dtype, c_z, c_hidden, n_tokens, direction, kw, cell):
    """The census key of a call class + whether n_tokens is inside the measured bucket of ``cell`` (this table's grammar)."""
    word, form = kw.get("word"), kw.get("form")
    if "+" in str(word):
        word, _, wform = str(word).partition("+")
        form = form if form is not None else (wform or None)
    prec = precision_word(dtype, kw.get("residency"), kw.get("tf32"))
    inside = False
    if cell:
        try:
            inside = int(n_tokens) <= int(str(cell).split("|")[4].split("+")[0][3:])
        except (IndexError, ValueError):
            inside = False
    b_ = batch_extent(kw.get("batch")) if (word == "exact" and not form) else None   # a BATCHED exact call class (B > 1) is its own census key (`out.fwd.B5`);
    btag = (".B%d" % b_) if (b_ is not None and b_ > 1) else ""                        # batch 1 / unstated and every other word: the key is unchanged byte for byte
    key = dict(cc=cc_word(cc), stack=kw.get("stack"), dtype=prec, shape="C%sH%s" % (c_z, c_hidden),
               bucket=_CENSUS.bucket_word(n_tokens, cell, inside),
               form="%s.%s%s%s" % (direction_word(direction), "fwdbwd" if kw.get("backward") else "fwd", ("+" + form) if form else "", btag),
               word="%s%s" % (word, ("+" + form) if form else ""))
    return key, word, inside


def _census(cc, dtype, c_z, c_hidden, n_tokens, direction, kw, sel, refusal):
    """Classify the decision (cell_hit | inherited | named_fallback | stock | opt_in) from the Selection's own facts and record it."""
    try:
        cell = sel.cell if sel is not None else None
        bm = re.search(BATCH_NOTE + r"\([^)]*\)", (sel.reason or "") if sel is not None else "")   # a batched exact call the stock op serves BY NAME: its own key
        key, word, inside = _census_key(cc, dtype, c_z, c_hidden, n_tokens, direction, kw if bm else dict(kw, batch=None), cell)   # (`<dir>.fwd.B<b>`) + the note; every other record: the batch-1 key, unchanged
        if refusal is not None:                                           # decided BY NAME against the word: the caller binds the named fallback row
            _CENSUS.record("trimul", key, "named_fallback", "caller:%s" % (refusal.fallback or "-"), cell_id=None,
                           refused="%s:%s" % (refusal.row or word, refusal.kind), note="raised")
            return
        reason, stack = sel.reason or "", kw.get("stack")
        if word in ROW_NAMES:
            _CENSUS.record("trimul", key, "opt_in", sel.row, cell_id=cell, note=reason if cell is None or not inside else "")
        elif cell is None:                                                # no measured cell in this family at all: the named stock row served
            _CENSUS.record("trimul", key, "inherited", sel.row, cell_id=None, note="family:none(%s)" % reason)
        elif "inherited_cc:unmeasured" in reason:                          # an UNMEASURED capability read the nearest measured capability's column
            m = re.search(r"inherited_cc:unmeasured\(([^)]*)\)(?::(\S+))?", reason)
            _CENSUS.record("trimul", key, "inherited", sel.row, cell_id=cell, note="inherited_cc:unmeasured(%s) column=%s" % (m.group(1) if m else "?", (m.group(2) if m and m.group(2) else sel.stack)))
        elif "skip " in reason:                                           # the cell's winner (or a preferred row) refused by name: the next measured row served
            m = _SKIP_RE.search(reason)
            _CENSUS.record("trimul", key, "named_fallback", sel.row, cell_id=cell, refused=("%s:%s" % (m.group(1), m.group(2))) if m else "-:%s" % reason)
        else:
            sib = _CENSUS.sibling_note(sel.stack) if (stack and sel.stack and sel.stack != stack) else ""   # the cell's timing column is a sibling / the reference stack's
                                                                          # (same cc; sibling_column): a CELL_HIT whose note names it -- inherited_stack(measured_on=<column>)
            if not inside or "outin_family" in reason:                    # a TRUE gap: a size neighbour | the out+in block cell read for a per-direction query
                why = "beyond_measured" if not inside else "guard:direction(outin_cell)"
                _CENSUS.record("trimul", key, "inherited", sel.row, cell_id=cell, note=why + ((";" + sib) if sib else ""))
            elif sel.row in STOCK_ROWS:                                   # (+ the batch note when the stock op stands in for an exact-class row at B > 1)
                _CENSUS.record("trimul", key, "stock", sel.row, cell_id=cell, note=";".join(x for x in (sib, bm.group(0) if bm else "") if x))
            else:
                _CENSUS.record("trimul", key, "cell_hit", sel.row, cell_id=cell, note=sib or ("prefer" if "prefer" in reason else ""))
    except Exception:                                                     # the census counts; it never gates or breaks a selection
        return


def _select(cc, dtype, c_z, c_hidden, n_tokens, direction="outgoing", *, word, residency=None, backward=False, prefer=None, stack=None,
            tf32=None, has_cueq=None, config=None, abi=None, exclude=None, form=None, batch=None):
    """The row + facts for this call.  ``word``: a row name (default rule: exactly that row) or a tier word (fast | exact | big: the cell's
    measured winner; on ``stack`` when the table measured that stack, else the cell's reference stack).  ``prefer``: restrict a tier to these
    rows (the first one the cell measured, in the caller's order; when none was measured the cell's named stock row, which is always eligible --
    reason 'prefer(first measured)').  ``abi``: this process's extension ABI key (rows shipping prebuilt binaries are skipped BY NAME when they
    have none for it).  ``exclude``: rows a TIER word must not consider (the serving call lists a row here after it refused with the tensors in
    hand -- 'stepaside'; a row word ignores it).  Pure: no framework import.
    EXACT VOUCH IS STACK-SPECIFIC: under the tier word ``exact`` an exact-class row (EXACT_ROWS) is returned only when ``stack`` (the caller's running
    stack word; the serving call always states it) carries the row's bitwise record in this cell (``exact_vouched``); on any other stack the row is
    skipped by name (reason ``skip <row>(exact_vouch_not_recorded_on:<stack>)``) and the word resolves to the stock op of the stack -- like the ABI gate
    of the prebuilt-binary rows.  ``stack=None`` (a planning call that does not state a stack) reads the reference column as before.
    UNMEASURED CAPABILITY (no column of that cc in the table: 10.0 / 10.3 / 12.0 ..., an 8.x part other than 8.0): fast / big read the
    nearest measured architecture-compatible capability's cells (``inherited_cc``: same major, else the highest measured below) restricted to
    portable rows (INHERIT_PORTABLE_ROWS: v4, the tmk3 pair, the native payload where its build record carries the architecture); the exact
    tier is the stock op by name there; reason / census token ``inherited_cc:unmeasured(<cc>-><measured cc>)`` + the column read.
    EXACT ENVELOPE: beyond the family's largest bucket (note beyond_measured) the exact tier skips every exact-class row by name (reason ``skip
    <row>(exact_beyond_vouched_envelope:N>{largest vouched bucket})``) and resolves to the stock op -- an exact word is vouched where it serves or
    not served; the form-keyed exact refuses likewise (``form_vouch_not_recorded(..): N>..``); fast / big keep the nearest cell (beyond_measured).
    SIBLING COLUMN (fast / big, and a row word's numbers): a stack word the cell does not list reads the NEAREST listed column of the cell
    (``sibling_column``: the same torch version + cuda tag with another library / triton minor, else the same torch major.minor, else the reference
    column); the Selection's ``stack`` is that column and its ``reason`` carries ``column=sibling:<listed stack word>`` (``column=reference:<..>``
    under rule 3) -- one column per stack word, so a tier word's answers over the size buckets are that column's, never a mix of columns.
    """
    if word is None:
        raise ValueError("word is required: a row name (%s) or a tier word (%s)" % (", ".join(ROW_NAMES), ", ".join(TIER_WORDS)))
    prove = False
    if str(word).endswith(PROVE_SUFFIX):                                             # "exact+of3_module:prove": shapes outside the vouch table are proven per call shape (module= callable)
        word, prove = str(word)[: -len(PROVE_SUFFIX)], True
    if "+" in str(word):                                                             # "exact+of3_module": the tier word with its form key in one string (a binding that passes words through)
        word, _, wform = str(word).partition("+")
        form = form if form is not None else (wform or None)
    if form is not None and str(form).endswith(PROVE_SUFFIX):
        form, prove = str(form)[: -len(PROVE_SUFFIX)], True
    if form is not None and form not in FORMS:
        raise ValueError("unknown form %r (forms: %s; no form = the stock library op's statement)" % (form, ", ".join(sorted(FORMS))))
    key, measured, note = cell_key(cc, dtype, c_z, c_hidden, n_tokens, direction, residency=residency, backward=backward, tf32=tf32)
    inh = None                                                                       # an UNMEASURED capability reads the nearest measured capability's cells (fast / big:
    if key is None and str(note).startswith("no_cell:"):                             # its portable rows; exact: the library op by name) -- decided PER FAMILY KEY:
        fam_ = (precision_word(dtype, residency, tf32), int(c_z), int(c_hidden), direction_word(direction), "fwdbwd" if backward else "fwd")
        icc = inherited_cc(cc, family=fam_)                                          # the nearest measured capability that HAS this family, whatever the
        if icc is not None:                                                          # device's capability has for other families (partial_cc)
            k2, _m2, n2 = cell_key(icc, dtype, c_z, c_hidden, n_tokens, direction, residency=residency, backward=backward, tf32=tf32)
            if k2:
                inh = (cc_word(cc), icc)
                key, measured, note = k2, False, "inherited_cc:unmeasured(%s->%s)%s" % (inh[0], inh[1], ",partial_cc" if cc_word(cc) in measured_ccs() else "") + ((";" + n2) if n2 else "")
    cell = table()["cells"].get(key) if key else None
    trit = _triton_of(stack)
    akw = dict(residency=residency, backward=backward, has_cueq=has_cueq, tf32=tf32, triton=trit, abi=abi)
    if form is not None and word == "exact":                                         # the exact tier UNDER A FORM: the engine-module statement's row where the VOUCH TABLE records this
        frow = FORMS[form]                                                           # exact (stack, precision, width, N, direction) -- else refused by name (the caller keeps its module)
        stock_row = "torch_math" if (has_cueq is False or int(c_hidden) != int(c_z)) else "cueq"
        if exclude and frow in tuple(exclude):                                       # the form's row refused with the tensors in hand: no step-aside under a form (the next
            raise Refusal("form:%s(%s refused at serve time; no other row states this module)" % (form, frow), frow, stock_row)   # exact row would be the LIBRARY op's) -- the caller keeps its module
        admits(frow, cc, dtype, c_z, c_hidden, n_tokens, **akw)
        cls, exact_vs, bwd, cap, fb = _row_facts(frow)
        try:
            fkey, fcell, rec = form_vouch(form, cc, dtype, c_z, c_hidden, n_tokens, direction, stack=stack, residency=residency, backward=backward, tf32=tf32, has_cueq=has_cueq)
        except Refusal as e:
            if not prove:
                raise
            # PROVE mode: outside the vouch table the row is served only after THIS call shape is proven bitwise against the caller's module (per exact
            # shape tuple, cached in the caller's cache; never per process) -- see triangle_multiplication(module=)
            ccw, prec, dirw = cc_word(cc), precision_word(dtype, residency, tf32), direction_word(direction)
            pkey = "%s|%s|C%d|H%d|N=%d|%s|%s+%s%s" % (ccw, prec, int(c_z), int(c_hidden), int(n_tokens), dirw, "fwdbwd" if backward else "fwd", form, PROVE_SUFFIX)
            return Selection(frow, "exact", pkey, False, stack, cls, exact_vs, bwd, cap, None, fb, "prove per call shape (%s)" % e.kind, config)
        ent = rec.get("vouch_entry") or {}
        reason = "exact under form %s: vouched on %s (%s)" % (form, stack, "N listed" if "N" in ent else "N class mod %s rem %s%s" % (ent.get("mod"), ent.get("rem"), (" -> config " + str(ent["config"])) if ent.get("config") else ""))
        cfg = config if config is not None else (ent.get("config") or fcell.get("config"))
        return Selection(frow, "exact", fkey, True, stack, rec.get("class") or cls, exact_vs, bwd, cap, rec.get("x_module"), fb, reason, cfg)
    # THE COLUMN this stack reads: its own when listed; else (fast / big / a row word's numbers) the NEAREST listed column of the cell --
    # same torch + cuda tag with another library / triton minor, else the same torch major.minor, else the reference column (sibling_column);
    # the exact tier keeps the reference column for an unlisted stack (its rows serve on vouched stacks only: no inheritance)
    if cell and word == "exact" and not (stack and stack in cell.get("fast_per_stack", {})):
        st, col_rule = cell.get("ref_stack"), "reference"
    else:
        st, col_rule = sibling_column(cell, stack) if cell else (None, "reference")
    col_note = ("column=%s:%s" % ("sibling" if col_rule.startswith("sibling") else "reference", st)) if (cell and stack and col_rule != "listed") else ""
    if inh is not None:
        col_note = "column=inherited_cc:unmeasured(%s->%s):%s" % (inh[0], inh[1], st)
    exclude = tuple(exclude or ())
    if word in ROW_NAMES:
        admits(word, cc, dtype, c_z, c_hidden, n_tokens, **akw)
        if word in MODULE_EXACT_ROWS:                                                # a module-exact row BY NAME is admitted exactly where its form is vouched (same table, same refusal)
            _fk, _fc, _rec = form_vouch(FORM_OF_ROW[word], cc, dtype, c_z, c_hidden, n_tokens, direction, stack=stack, residency=residency, backward=backward, tf32=tf32, has_cueq=has_cueq)
            if config is None:
                config = (_rec.get("vouch_entry") or {}).get("config") or _fc.get("config")   # the layout the vouch names at this N
        cls, exact_vs, bwd, cap, fb = _row_facts(word)
        x = (cell or {}).get("x_stock", {}).get(word, {}).get(st) if cell else None
        v4note = ""
        if config is None:                                                           # the table's launch cell for this row on the column read -- a v4 tensor-descriptor cell is
            config, v4note = v4_table_cell(cell, st, trit) if word == "v4" else (_cell_config(cell, word, st), "")   # never handed to a stack whose triton cannot build it
        fb = "torch_math" if int(c_hidden) != int(c_z) else fb
        rnote = (note or ("row word" if cell else "row word; no cell measured")) + (("; " + v4note) if v4note else "")
        return Selection(word, word, key, measured, st, _measured_class(cell, word, st, cls), exact_vs, bwd, cap, x, fb, (col_note + "; " + rnote) if col_note else rnote, config)
    if word not in TIER_WORDS:
        raise ValueError("unknown word %r (rows: %s; tiers: %s)" % (word, ", ".join(ROW_NAMES), ", ".join(TIER_WORDS)))
    stock_row = "torch_math" if (has_cueq is False or int(c_hidden) != int(c_z)) else "cueq"     # the stock library op takes c_hidden == c_z; the engines run their torch path otherwise
    if cell is None:
        cls, exact_vs, bwd, cap, fb = _row_facts(stock_row)
        return Selection(stock_row, word, None, False, None, cls, exact_vs, bwd, cap, None, fb, "named stock row: %s" % note, config)
    per = cell.get("%s_per_stack" % word, {})
    winner = per.get(st) or cell.get(word) or stock_row                  # THE word of this stack's column (else the cell-level word an unlisted stack inherits)
    if inh is not None and word == "exact":                              # an UNMEASURED capability: no byte vouch exists there -> the library / stock op BY NAME
        cls, exact_vs, bwd, cap, fb = _row_facts(stock_row)
        return Selection(stock_row, word, key, False, st, cls, exact_vs, bwd, cap, None, fb,
                         "exact on an unmeasured capability: the stock op by name (exact_vouch_not_recorded_on_cc:%s); %s; %s" % (inh[0], col_note, note), config)
    if word == "exact":
        order = [winner] + [r for r in STOCK_ROWS if r != winner and st in cell.get("ms", {}).get(r, {})]
        # Under the EXACT word a STOCK-class winner is THIS PROCESS'S stock op -- the library op on a library stack (`cueq`); the module
        # math (`torch_math`) only where the process has no library (has_cueq False / a `nocueq` stack word) or the call is one the library
        # does not take (c_hidden != c_z).  A cell whose columns do not list the running stack reads its REFERENCE column (no exact
        # inheritance, above); when that column is a library-less stack its stock row is `torch_math`, which is NOT the stock op of a
        # library stack and must not stand in for it (the module math differs from the library op in the last bits, and an iterative
        # trunk amplifies that into visibly different coordinates).  Only the winner is rewritten; exact-class rows are untouched (their
        # byte vouch is checked per stack below) and the column's other stock rows keep their place behind it.
        this_stock = _exact_stock_row(stock_row, stack, has_cueq, c_z, c_hidden)
        if (stack or has_cueq) and has_cueq is not False and order[0] in STOCK_ROWS and order[0] != this_stock:   # a planning call (no stack, has_cueq unstated) keeps the column word; has_cueq False is the library-less branch below
            note = (note + "; " if note else "") + "exact_stock_row_of_this_stack(%s->%s)" % (order[0], this_stock)
            order = [this_stock] + [r for r in order if r != this_stock]
            order = [order[0]] + [r for r in order[1:] if r != winner] + ([winner] if winner != this_stock else [])   # the other column's stock row last
        # AT OR UNDER THE LIBRARY'S TORCH-PATH THRESHOLD the exact answer on a library stack is the library op itself.  cuEquivariance's
        # triangle_multiplicative_update serves n_tokens <= CUEQ_TRIMUL_FALLBACK_THRESHOLD (100) by its own torch statement, not its kernel;
        # every exact-class row's byte vouch (`bitwise=cueq`) was recorded against the KERNEL (cells >= 128 tokens), so below the threshold
        # no row is vouched against what stock runs there (a row byte-equal to the library KERNEL, e.g. tmk3_exact on the tf32 column, still
        # differs from the library's torch statement in the last bits, and an fp32 / tf32 caller of <= 100 tokens sees that).  The stock op
        # serves those calls (cueq: the library takes its own below-threshold path = stock's bytes); above the threshold nothing changes.
        if this_stock == "cueq" and not (exclude and "cueq" in tuple(exclude)) and int(n_tokens) <= cueq_torch_threshold() and order[0] != "cueq":   # (a serve-time step-aside that excluded the library op keeps its re-selection)
            note = (note + "; " if note else "") + "exact_below_library_threshold(%s->cueq:N<=%d)" % (order[0], cueq_torch_threshold())
            order = ["cueq"] + [r for r in order if r in STOCK_ROWS and r != "cueq"]
        # THE EXACT VOUCH IS A BATCH-1 VOUCH.  Every exact-class row's byte record (`bitwise=cueq`, vouched_on) was taken on ONE pair tensor
        # [1, N, N, c] against the library op's call on that tensor.  The library's own BATCHED call ([B, N, N, c], B > 1 -- e.g. a confidence
        # head running its pairformer over several diffusion samples as one batch) is not B calls of one sample byte for byte: a row that IS
        # byte-equal to the library per sample (tmk3_exact, batch-invariant) can still differ from the batched library call where the batched
        # sums round differently, and native_exact differs from the library at batch > 1.  A proof at one layout never
        # licenses another: with a leading batch extent > 1 stated (`batch=`: opt_core.trimul.by_word and triangle_multiplication state it from
        # the tensor in hand) an exact-class row serves only where the cell RECORDS its batched vouch on this stack (cells.<key>.
        # vouched_on_batched -- none today); otherwise the stock op of THIS process serves BY NAME (note exact_batch_unvouched(<row>-><stock>:
        # B<b>), census outcome `stock` under its own key `<dir>.fwd.B<b>`).  batch unstated (a planning call) or 1: unchanged byte for byte;
        # fast / big (tolerance classes) and the row words (the caller's own statement): unchanged; the form path (module-exact rows) is not
        # this branch.
        b_ = batch_extent(batch)
        if (b_ is not None and b_ > 1 and has_cueq is not False and order[0] in EXACT_ROWS and not exact_batch_vouched(cell, order[0], stack, b_)
                and (not stack or exact_vouched(cell, order[0], stack)) and "beyond_measured" not in (note or "")):   # (a row this stack's batch-1 vouch does not cover, or a
            note = (note + "; " if note else "") + "%s(%s->%s:B%d)" % (BATCH_NOTE, order[0], this_stock, b_)      #  size beyond the envelope, is refused by ITS OWN name below --
            order = [this_stock] + [r for r in order[1:] if r in STOCK_ROWS and r != this_stock]                 #  that answer and its census token are unchanged)
    else:                                                                 # fast / big: then the next words OF THAT COLUMN -- the rows timed on it, fastest first
        order = [winner] + [r for r in _column_rank(cell, st, word) if r != winner]   # (a thin / inherited column ranks by the column it was measured on)
    if inh is not None:                                                   # fast / big on an UNMEASURED capability: the column's PORTABLE rows only (source-compiled Triton rows,
        port = [r for r in order if r in INHERIT_PORTABLE_ROWS]           # the native payload where its build record carries the architecture -- its admits decides below),
        port += [r for r in cell.get("fast_order", []) if r in INHERIT_PORTABLE_ROWS and r not in port]   # in the column's measured order, then the cell's, then the
        port += [r for r in INHERIT_PORTABLE_ROWS if r not in port]      # family's; the library op only when none admits
        dead = _TABLE.get("dead_rows") or set()                           # a portable row that failed to build / compile / launch on this capability
        order = [r for r in port if (r, inh[0]) not in dead]              # earlier in the process is never retried (_inherited_step_aside)
    if backward:
        order = [r for r in order if r in BACKWARD_ROWS] or [stock_row]
    if word == "big":                                                   # the release policy's big exclusion holds at EVERY N: the ESM family (N^2-scaled workspaces) is never a
        order = [r for r in order if r not in BIG_EXCLUDED_ROWS] or [stock_row]   # big answer -- not as the column word, not on the step-aside walk (native above its shape
                                                                          # ceiling), not on a beyond-measured or inherited bucket; the stock op is the floor
    if has_cueq is False and word == "exact":                            # exact = a statement against the stock library op: without the library the stock module
        order = [r for r in order if r not in ("cueq", "tmk3_exact", "tx_sm90a_exact", "ef2_cueq_tiles", "native_exact")] or ["torch_math"]   # math is the exact row; fast / big
                                                                          # rows that import it refuse by name in the admits pass below (has_cueq threaded)
    why = ""
    if exclude:
        why = "stepaside %s" % "+".join(r for r in exclude)
        order = [r for r in order if r not in exclude] or [r for r in (stock_row, "torch_math") if r not in exclude][:1] or ["torch_math"]
    chosen = None
    if prefer:
        allowed = tuple(prefer) + STOCK_ROWS
        for r in prefer:
            if r in order and (word != "exact" or r in EXACT_ROWS + STOCK_ROWS):
                chosen = r
                break
        if chosen is None:
            for r in order:
                if r in allowed:
                    chosen = r
                    break
        if chosen is None:
            raise Refusal("prefer:%s not measured in cell %s" % ("+".join(prefer), key), None, stock_row)
        why = "prefer" if chosen == prefer[0] else "prefer(first measured)"
    else:
        chosen = order[0]
    for r in [chosen] + [x for x in order if x != chosen]:
        try:
            admits(r, cc, dtype, c_z, c_hidden, n_tokens, **akw)
        except Refusal as e:
            why = (why + ";" if why else "") + "skip %s(%s)" % (r, e.kind)
            if prefer and r in prefer and r == chosen:
                continue
            continue
        if word == "exact" and r in EXACT_ROWS and note and "beyond_measured" in note:                  # BEYOND the family's largest bucket no byte test vouches any
            big, vmax = exact_envelope(cc, dtype, c_z, c_hidden, direction, r, stack, residency=residency, backward=backward, tf32=tf32)   # size >= N: an exact
            why = (why + ";" if why else "") + "skip %s(exact_beyond_vouched_envelope:N>%s)" % (r, vmax if vmax is not None else big)   # word must be vouched
            continue                                                                                    # WHERE it serves -> refused by name, the stock op serves
        if word == "exact" and r in EXACT_ROWS and stack and not exact_vouched(cell, r, stack):
            why = (why + ";" if why else "") + "skip %s(exact_vouch_not_recorded_on:%s)" % (r, stack)   # an exact-class row's bitwise vouch holds ONLY on the
            continue                                                                                    # stacks it was byte-tested on: elsewhere the word falls to the stock op
        cls, exact_vs, bwd, cap, fb = _row_facts(r)
        x = cell.get("x_stock", {}).get(r, {}).get(st)
        reason = ("%s winner on %s" % (word, st)) + ("; " + col_note if col_note else "") + ("; " + why if why else "") + ("; " + note if note else "")
        if config is None:                                                                              # the table's launch cell for the row on the column read -- a v4
            config, v4note = v4_table_cell(cell, st, trit) if r == "v4" else (_cell_config(cell, r, st), "")   # tensor-descriptor cell is never handed to a stack whose
            reason += ("; " + v4note) if v4note else ""                                                  # stated triton cannot build it (the package's own cell serves)
        fb = "torch_math" if int(c_hidden) != int(c_z) else fb
        return Selection(r, word, key, measured, st, _measured_class(cell, r, st, cls), exact_vs, bwd, cap, x, fb, reason, config)
    cls, exact_vs, bwd, cap, fb = _row_facts(stock_row)
    return Selection(stock_row, word, key, measured, st, cls, exact_vs, bwd, cap, None, fb, "no admitted row; named stock row; " + why + (("; " + col_note) if col_note else ""), config)


def describe(sel):
    """One line for a kit's LEVER / census line."""
    return "trimul row=%s word=%s cell=%s%s stack=%s class=%s%s x_stock=%s%s reason=%s" % (
        sel.row, sel.word, sel.cell, "" if sel.size_measured else "(nearest)", sel.stack, sel.cls,
        (" exact_vs=%s" % sel.exact_vs.split(" ")[0]) if sel.exact_vs else "", sel.x_stock, " backward" if sel.backward else "", sel.reason)


def coverage(cc, shapes, *, word="fast", stack=None, has_cueq=None):
    """Census of a kit's shapes against the table: shapes = [(dtype, c_z, c_hidden, n_tokens, direction[, residency[, backward]]), ...] ->
    [{shape, row, cell, measured, x_stock, reason}]."""
    out = []
    for s in shapes:
        dtype, C, H, N, d = s[:5]
        res = s[5] if len(s) > 5 else None
        bwd = bool(s[6]) if len(s) > 6 else False
        try:
            sel = select(cc, dtype, C, H, N, d, word=word, residency=res, backward=bwd, stack=stack, has_cueq=has_cueq)
            out.append({"shape": s, "row": sel.row, "cell": sel.cell, "measured": sel.size_measured, "x_stock": sel.x_stock, "reason": sel.reason})
        except Refusal as e:
            out.append({"shape": s, "row": None, "cell": None, "measured": False, "x_stock": None, "reason": "refused:%s->%s" % (e.kind, e.fallback)})
    return out


def rows_for_kit(carried):
    """The subset of ROW_NAMES a kit can bind given the rows it carries/routes (stock rows always)."""
    return tuple(r for r in ROW_NAMES if r in carried or r in STOCK_ROWS)


# ------------------------------------------------------------------------------------------------------------------------------ serving
_MODS = {}
_TILES_INSTALLED = set()


def _imp(name, refusal_row, fallback):
    """Import ``name`` (a Refusal BY NAME when it cannot be imported).  A module not imported yet is imported through
    :func:`_pystack.import_padded`: third-party module bodies (cuequivariance's import-time autotune table, a Triton front end) never run on the
    interpreter's stack-chunk boundary from here (see ``_pystack``: 1.4 s vs 60 s for the same import in a syscall-heavy runtime)."""
    try:
        return _PS.import_padded(name)
    except ImportError as e:
        raise Refusal("import:%s(%s)" % (name.split(".")[-1] if name.startswith("opt_core") else name, str(e).split("\n")[0][:60].replace(" ", "_")),
                      refusal_row, fallback)


from . import _pystack as _PS                                    # noqa: E402  (padded_call / import_padded: one-off host work off the stack-chunk boundary)

_RESOLVED = {}                                                  # (name, core_name) -> the module resolved once per process (the resolution below stats files: ~0.5 ms a call otherwise)


def _by_name_or_core(name, core_name, row, fallback, needs=()):
    m = _RESOLVED.get((name, core_name))
    if m is not None:
        return m
    m = _RESOLVED[(name, core_name)] = _resolve_by_name_or_core(name, core_name, row, fallback, needs)
    return m


def _resolve_by_name_or_core(name, core_name, row, fallback, needs=()):
    """The module a kit ROUTED under ``name`` to the core copy (``opt_core.kernels.route``: the object it runs today, so one set of compiled
    kernels serves both paths), else the core copy under its own package path.  A kit's private copy of the name on ``sys.path`` (an older or
    diverged line) is never picked up: the rows of this table are the core copies as measured.  ``needs``: attributes the module must have."""
    import sys
    m = sys.modules.get(name)
    if m is not None:
        try:
            from opt_core import kernels as _k
            top = name.split(".")[0]
            core_dir = os.path.abspath(_k.carried_path(top))
            where = os.path.abspath(getattr(m, "__file__", "") or "")
            if where.startswith(core_dir + os.sep) and all(hasattr(m, a) for a in needs):
                return m
        except (ImportError, KeyError, OSError):
            pass
    return _imp(core_name, row, fallback)


_CC = {}


def _device_cc(z):
    """(major, minor) of z's device, memoised per device (torch's query is ~5-10 us a call; the words ask it every call)."""
    if not z.is_cuda:
        return (0, 0)
    k = z.device.index
    cc = _CC.get(k)
    if cc is None:
        import torch
        cc = _CC[k] = tuple(torch.cuda.get_device_capability(z.device))
    return cc


def weights_key(w, names=("w_ap", "w_o")):
    """The identity of a weight set for the rows' pack caches: (data pointer, shape, stride, dtype, device) of the named tensors.  Not ``id()``
    (a module that re-wraps its parameters per call -- ``.detach()``, a slice of a fused projection -- would re-pack every call) and never the
    tensor's version counter (inference tensors have none: reading it raises under ``torch.inference_mode``).  The cache entry holds the keyed
    tensors, so a pointer cannot be reused by another tensor while the entry lives.  Weights are read as constants: a kit that updates them IN
    PLACE clears its cache (or passes a new one) to re-pack."""
    out = []
    for k in names:
        t = w.get(k)
        if t is None:
            out.append((k, None))
        else:
            out.append((k, t.data_ptr(), tuple(t.shape), tuple(t.stride()), str(t.dtype), str(t.device)))
    return tuple(out)


def _packed(cache, tag, w, build, extra=()):
    """cache[(tag, weights_key, *extra)] -> the row's pack, built once by ``build()``; the entry pins the source tensors (see weights_key)."""
    ck = (tag,) + weights_key(w) + tuple(extra)
    hit = cache.get(ck) if cache is not None else None
    if hit is not None:
        return hit[0]
    wp = build()
    if cache is not None:
        cache[ck] = (wp, w.get("w_ap"), w.get("w_o"))
    return wp


def _autocast_dtype(z):
    import torch
    if torch.is_autocast_enabled():
        return torch.get_autocast_dtype(z.device.type) if hasattr(torch, "get_autocast_dtype") else torch.get_autocast_gpu_dtype()
    return None


def compute_input(z, cache=None):
    """The tensor a fused row reads under autocast: ``z`` cast ONCE to the autocast dtype (the stock fused kernel's rounding point); else ``z``.
    With ``cache`` the cast is memoised for this z WITHIN the current face call (one cast per call even when admission and serving both ask, or
    both directions of one 'outin' call); triangle_multiplication releases it on return -- no cast copy (nor the caller's z) is held across calls."""
    dt = _autocast_dtype(z)
    if dt is None or z.dtype == dt:
        return z
    if cache is not None:
        ver = _tensor_version(z)
        hit = cache.get("_z_cast")
        if hit is not None and hit[0] is z and hit[1] == dt and (len(hit) < 4 or hit[3] == ver):   # within the call: same tensor object, dtype AND version
            _CAST_MEMO["hits"] += 1
            return hit[2]
        zc = z.to(dt)
        if "_z_cast" not in cache:
            _CAST_MEMO["live"] += 1
        cache["_z_cast"] = (z, dt, zc, ver)                                # released by triangle_multiplication on return (release_call_scratch)
        _CAST_MEMO["stores"] += 1
        return zc
    return z.to(dt)


def _tensor_version(t):
    """The tensor's in-place version counter (None for inference tensors, which carry none)."""
    try:
        return int(t._version)
    except (RuntimeError, AttributeError):                                 # inference tensors: no version counter
        return None


_CAST_MEMO = {"live": 0, "stores": 0, "hits": 0, "released": 0}          # bookkeeping of the per-call cast memo (entries alive / stored / hit / released)


def release_call_scratch(cache):
    """Drop the per-call scratch a face call left in ``cache`` (the cast memo): nothing cast for one call outlives it."""
    if cache is not None and cache.pop("_z_cast", None) is not None:
        _CAST_MEMO["live"] -= 1
        _CAST_MEMO["released"] += 1


def cast_memo_stats():
    """{'live', 'stores', 'hits', 'released'} of the per-call cast memo."""
    return dict(_CAST_MEMO)


def call_precision(z):
    """(precision word, compute dtype) of a live call: bf16 | f32z_bf16 | fp32 | tf32 (| fp16 words for half)."""
    import torch
    ac = _autocast_dtype(z) if z.is_cuda else None
    if ac is not None and ac != torch.float32:
        w = precision_word(ac, residency=("fp32" if z.dtype == torch.float32 else None))
        return w, ac
    if z.dtype == torch.float32:
        tf = bool(torch.backends.cuda.matmul.allow_tf32) or torch.get_float32_matmul_precision() != "highest"
        return ("tf32" if tf else "fp32"), torch.float32
    return precision_word(z.dtype), z.dtype


_CUEQ_FACTS = {}


def cueq_present():
    """Whether cuequivariance_torch is importable in this process -- WITHOUT importing it (importlib's finder only).  Importing that package
    executes cuequivariance_ops_torch's module bodies (0.11: an ahead-of-time autotune input table of ~1.6 M dicts built at import -- seconds in a
    fresh process, a minute inside a loaded engine process); the words of this table never pay that to answer a question about the stack."""
    if "present" not in _CUEQ_FACTS:
        import sys
        if "cuequivariance_torch" in sys.modules:
            _CUEQ_FACTS["present"] = True
        else:
            import importlib.util
            try:
                _CUEQ_FACTS["present"] = importlib.util.find_spec("cuequivariance_torch") is not None
            except (ImportError, ValueError):
                _CUEQ_FACTS["present"] = False
    return _CUEQ_FACTS["present"]


def cueq_version():
    """cuequivariance_torch's version string ('0.11.1', ...) read from the installed distribution's metadata or the already-imported module --
    never by importing the package here; '?' when neither says."""
    if "version" not in _CUEQ_FACTS:
        import sys
        v = getattr(sys.modules.get("cuequivariance_torch"), "__version__", None)
        if v is None:
            try:
                from importlib import metadata
            except ImportError:                                   # pragma: no cover
                metadata = None
            for dist in (("cuequivariance-torch", "cuequivariance_torch") if metadata is not None else ()):
                try:
                    v = metadata.version(dist)
                    break
                except Exception:                                 # PackageNotFoundError / a broken metadata dir: the next spelling, else '?'
                    continue
        _CUEQ_FACTS["version"] = str(v) if v else "?"
    return _CUEQ_FACTS["version"]


def stack_word(z=None):
    """'<card>:<torch>/<triton>/<cueq version | nocueq>' of this process (the table's stack grammar).  The cuequivariance version is read from
    the distribution metadata (:func:`cueq_version`): asking for the stack word imports no model library."""
    import torch
    try:
        import triton
        tv = triton.__version__
    except ImportError:
        tv = "none"
    cq = "cueq" + cueq_version() if cueq_present() else "nocueq"
    card = "?"
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(z.device if z is not None and z.is_cuda else None)
        card = "H100" if "H100" in name else ("A100" if "A100" in name else name.replace(" ", "_"))
    return "%s:%s/%s/%s" % (card, torch.__version__, tv, cq)


def tx_abi_tag():
    """The tx package's ABI key for this process (its ops.abi_tag), without loading the binary."""
    ops = _imp("opt_core.kernels.trimul.tx_sm90a.ops", "tx_sm90a", "v4")
    return ops.abi_tag()


def _w10(weights, row):
    missing = [k for k in WEIGHT_KEYS if weights.get(k) is None]
    if missing:
        raise Refusal("weights_missing:%s" % ",".join(missing), row, None)
    unknown = [k for k in weights if k not in WEIGHT_KEYS + BIAS_KEYS]
    if unknown:
        raise Refusal("weights_unknown:%s" % ",".join(sorted(unknown)), row, None)
    if row not in ("v4", "torch_math") and any(weights.get(k) is not None for k in BIAS_KEYS):
        raise Refusal("biases_unsupported", row, "v4")
    return weights


def _cueq_fn(stock):
    if stock is not None:
        return stock
    if "cueq" not in _MODS:
        cqt = _imp("cuequivariance_torch", "cueq", "torch_math")
        fn = getattr(cqt, "triangle_multiplicative_update", None)
        if fn is None:
            fn = _imp("cuequivariance_torch.primitives.triangle", "cueq", "torch_math").triangle_multiplicative_update
        _MODS["cueq"] = fn
    return _MODS["cueq"]


def _cueq_call(fn, z, mask, direction, w, eps, cache):
    import torch
    kw = _packed(cache, "cueq_w", w, lambda: dict(norm_in_weight=w["ln_in_w"], norm_in_bias=w["ln_in_b"], p_in_weight=torch.cat([w["w_ap"], w["w_bp"]], 0).contiguous(),
                                                    g_in_weight=torch.cat([w["w_ag"], w["w_bg"]], 0).contiguous(), norm_out_weight=w["ln_out_w"], norm_out_bias=w["ln_out_b"],
                                                    p_out_weight=w["w_o"], g_out_weight=w["w_og"]))
    try:
        return fn(z, direction=direction, mask=mask, eps=eps, **kw)
    except (AssertionError, TypeError, ValueError, NotImplementedError, IndexError) as e:   # the stock library op rejecting an envelope (its own assert at a
        raise Refusal("stock_op_raised:%s(%s)" % (type(e).__name__, str(e).split("\n")[0][:60].strip().replace(" ", "_")),   # width / on a build that does not take
                      "cueq", "torch_math")                                              # it) is a refusal BY NAME -> the module math; never a raise through a word


def tmk3_cfg(Tm, mode, cdt, N, C):
    """The TM-K3 launch cfg for (mode, compute dtype, N, C): the package's own ``_select_cfg`` when the process imported it in that mode, else the
    same selection over ``_build_config_table(mode)`` (identical dict; the process keeps serving its own mode untouched)."""
    import torch
    if getattr(Tm, "_MODE", None) == mode:
        return Tm._select_cfg(cdt, N, C)
    tabs = _MODS.setdefault("tmk3_tables", {})
    tab = tabs.get((id(Tm), mode))
    if tab is None:
        tab = tabs[(id(Tm), mode)] = Tm._build_config_table(mode)
    cls = "bf16" if cdt in (torch.bfloat16, torch.float16) else "fp32"
    sub = tab[cls].get("C%d" % C, tab[cls]["*"])
    cfg = None
    for k, v in sub.items():
        if k == "*":
            cfg = cfg or v
        else:
            lo, hi = [int(x) for x in k.split("-")]
            if lo <= N <= hi:
                cfg = v
    cfg = dict(cfg)
    elem = 2 if cls == "bf16" else 4
    if Tm.smem_bytes(cfg["A"], elem, False) > Tm.SMEM_CAP:
        cfg["A"] = dict(v=9, BM=64, BN=64, BK=32, num_warps=4, num_stages=2)
    if Tm.smem_bytes(cfg["C"], elem, True) > Tm.SMEM_CAP:
        cfg["C"] = dict(v=9, BM=64, BN=64, BK=32, num_warps=4, num_stages=2)
    return Tm._schedule_grid(cfg, N, elem)


def _serve_v4(z, mask, direction, w, residual, eps, cache, pad, config=None):
    G = _by_name_or_core("fpf_trimul_v4.generic", "opt_core.kernels.fpf_trimul_v4.generic", "v4", "cueq", needs=("trimul", "pack_weights", "supported"))
    zin = compute_input(z, cache)
    wp = _packed(cache, "v4_w", w, lambda: G.pack_weights(cache_owner=None, **{k: (v.detach() if hasattr(v, "detach") else v) for k, v in w.items() if k in WEIGHT_KEYS + BIAS_KEYS}),
                 extra=(str(zin.dtype),))
    C, D = int(zin.shape[-1]), int(wp["D"])
    if (C, D) == V4_C64_H128:
        return _serve_v4_kernel_face(zin, mask, direction, wp, residual, eps, cache, pad, config)
    okk = ("v4_ok", str(zin.device), str(zin.dtype), tuple(zin.shape), None if mask is None else (tuple(mask.shape), str(mask.dtype), str(mask.device)))
    ok = cache.get(okk)                                         # the package's admission answer for this (device, dtype, shape, mask form): asked once, not per call
    n_min = _v4_call_floor(z, zin, C, D, cache)                 # the table's token floor for this call class (101 unless the table admits the class lower)
    if ok is None:
        ok = cache[okk] = tuple(_v4_supported(G, zin, mask, wp, n_min))
    if not ok[0]:
        raise Refusal("v4:%s" % str(ok[1]).replace(" ", "_")[:80], "v4", "cueq")
    zs = zin if zin.dim() == 4 else zin[None]
    if int(zs.shape[0]) != 1:                                   # a batch keeps the package's own batch policy (one launch set within its limits, else its named loop)
        try:
            return G.trimul(zin, mask, direction=direction, weights=wp, residual=residual, eps=eps, pad=pad, **({"n_min": n_min} if n_min != V4_N_MIN_DEFAULT else {}))
        except G.TrimulUnsupported as e:
            raise Refusal("v4:%s" % str(getattr(e, "reason", e)).replace(" ", "_")[:80], "v4", "cueq")
    # B = 1: the package's KERNEL face with the package's own cell for this device -- the same kernels and bytes as its generic face, without the
    # generic face's per-call safety net around the launch set (measured +0.9..+1.3 ms per call at c 256 on cc 9.0: +127 % at 400 tokens, +15 % at 1536).
    # The package's once-per-shape warm probe still runs before the first call (a build failure switches the process to its SAFE cell BY NAME).
    import torch
    K = _by_name_or_core("fpf_trimul_v4.kernels", "opt_core.kernels.fpf_trimul_v4.kernels", "v4", "cueq", needs=("trimul_v4_forward", "resolve_cfg"))
    has_bias, f32 = bool(wp.get("has_bias")), zin.dtype == torch.float32
    pk = ("v4_cfg", str(zin.device), C, D, has_bias, f32)
    cfg = cache.get(pk)
    if cfg is None:
        if hasattr(G, "probe"):
            try:
                G.probe(zin.device, C, D, has_bias, f32)
            except G.TrimulUnsupported as e:
                raise Refusal("v4:%s" % str(getattr(e, "reason", e)).replace(" ", "_")[:80], "v4", "cueq")
        CE = _by_name_or_core("fpf_trimul_v4.cells", "opt_core.kernels.fpf_trimul_v4.cells", "v4", "cueq", needs=("cell_for",))
        k1, k3 = K.resolve_cfg(CE.cell_for(zin.device), C, D, has_bias)
        cfg = cache[pk] = {"k1": dict(k1), "k3": dict(k3)}
    if isinstance(config, dict) and "k1" in config and "k3" in config:
        if v4_desc_cell(config) and not getattr(K, "HAS_DESC", True):
            # the table's launch cell for the column this stack read is a TENSOR-DESCRIPTOR cell (K1 tma / tma2, K3 tma) and this triton has no
            # tl.make_tensor_descriptor (torch 2.7.1 stacks: triton 3.3.1): it cannot build here -- launched anyway it would raise AttributeError
            # on every call, counted as a kernel error by the caller's lever.  select() keeps such a cell off a stack whose triton it can read
            # (v4_table_cell); this is the same line for a stack word it could not read / a caller-built Selection: the package's own cell for
            # this (capability, triton) part serves (cfg above: its table's row for this triton, pointer kernels), said once by name.
            _say_once(cache, "v4:tensor_descriptor_cell_unbuildable:%s:%d:%d" % (str(zin.device), C, D),
                      "kernels.trimul: row v4: the table's launch cell %s is a tensor-descriptor cell this triton cannot build (no tl.make_tensor_descriptor); "
                      "serving the v4 package's own cell for this (capability, triton) part: %s" % (json.dumps({k: config[k] for k in ("k1", "k3")}, sort_keys=True), json.dumps(cfg, sort_keys=True)))
        else:
            cfg = {"k1": dict(config["k1"]), "k3": dict(config["k3"])}
    sr = getattr(G, "_STOCK_ROUND", True)
    zc = zs if zs.is_contiguous() else zs.contiguous()
    try:
        out = K.trimul_v4_forward(zc, direction == "outgoing", mask, wp, cfg, eps=eps, residual=residual, stock_round=True if sr is None else bool(sr), pad=pad)
    except AttributeError as e:                                     # an older triton lacking a language feature a cell's kernel names: refused BY NAME (a tier word steps aside to the
        if "triton" not in str(e):                                  # column's next row), never a kernel error through a word -- the (64, 128) kernel face holds the same line; any other
            raise                                                   # AttributeError is a defect and propagates unchanged
        raise Refusal("triton:%s" % str(e).replace(" ", "_")[:80], "v4", "cueq")
    return out if zin.dim() == 4 else out[0]



def _v4_call_floor(z, zin, C, D, cache):
    """Row v4's token floor for this live call's class (v4_n_min), memoised in the call cache: the precision word of the CALL (call_precision on the
    caller's z: bf16 | f32z_bf16 | fp32 | tf32) on this device's capability."""
    k = ("v4_n_min", str(zin.device), str(z.dtype), C, D)
    v = cache.get(k)
    if v is None:
        try:
            prec = call_precision(z)[0]
            v = v4_n_min(cc_word(_device_cc(zin)), prec, C, D)
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):   # a class the words cannot state keeps the package's own floor (named: never an OOM route)
            v = V4_N_MIN_DEFAULT
        cache[k] = v
    return v


def _v4_supported(G, zin, mask, wp, n_min):
    """``G.supported`` with the class floor: the package face takes ``n_min=`` since fpf_trimul_v4 4.4.5; an older face (a kit's own copy by name)
    keeps its N_MIN and answers as before."""
    if n_min != V4_N_MIN_DEFAULT:
        try:
            return G.supported(zin, mask, weights=wp, n_min=n_min)
        except TypeError:                                           # an older package face without the keyword: its own floor stands (refusal by name below it)
            pass
    return G.supported(zin, mask, weights=wp)


def _serve_v4_kernel_face(zin, mask, direction, wp, residual, eps, cache, pad, config):
    """(c_z 64, c_hidden 128): the carried package's kernel face with the cell's launch cells (``config`` {"k1", "k3"}; else the device row's (128,128)
    cells).  The package gates c_z to 128 / 256 at both faces while its kernels take separate c_z and c_hidden; the gate is widened to 64 for the
    duration of this call only and restored, so every other caller of the package in the process keeps the package's own envelope."""
    K = _by_name_or_core("fpf_trimul_v4.kernels", "opt_core.kernels.fpf_trimul_v4.kernels", "v4", "torch_math", needs=("trimul_v4_forward", "resolve_cfg"))
    cfg = config if isinstance(config, dict) and "k1" in config and "k3" in config else cache.get("_v4_c64_cfg")
    if cfg is None:
        CE = _by_name_or_core("fpf_trimul_v4.cells", "opt_core.kernels.fpf_trimul_v4.cells", "v4", "torch_math", needs=("cell_for",))
        k1, k3 = K.resolve_cfg(CE.cell_for(zin.device), 128, 128, False)
        cfg = cache["_v4_c64_cfg"] = {"k1": dict(k1), "k3": dict(k3, BN=min(int(k3.get("BN", 64)), 64))}
    N = int(zin.shape[-2])
    if N < 101:
        raise Refusal("n<101:below_n_min", "v4", "torch_math")
    # this path launches the cell directly (no per-shape warm probe that would switch a pointer cell in BY NAME): a tensor-descriptor cell on a
    # triton without tl.make_tensor_descriptor is refused by name here instead of failing inside the launch
    if not getattr(K, "HAS_DESC", True) and any(str((cfg.get(k) or {}).get("impl", "")).lower() in ("tma", "desc", "descriptor") for k in ("k1", "k3")):
        raise Refusal("triton:no_tensor_descriptor(c_z=64,c_hidden=128 cells are descriptor cells)", "v4", "torch_math")
    m = mask
    if m is not None and m.dim() == 4:
        m = m.reshape(m.shape[0], N, N)
    z3 = zin if zin.is_contiguous() else zin.contiguous()
    gate = K.SUPPORTED_C
    K.SUPPORTED_C = tuple(sorted(set(gate) | {64}))
    try:
        return K.trimul_v4_forward(z3, direction == "outgoing", m, wp, cfg, eps=eps, residual=residual, stock_round=True, pad=pad)
    except AttributeError as e:                                     # an older triton lacking a language feature the cell's kernel names: by name, never a crash through a word
        raise Refusal("triton:%s" % str(e).replace(" ", "_")[:80], "v4", "torch_math")
    finally:
        K.SUPPORTED_C = gate

def _serve_tmk3(row, z, mask, direction, w, residual, eps, cache):
    import torch
    mode = "exact" if row == "tmk3_exact" else "fast"
    K = _by_name_or_core("fpf_trimul.kernels", "opt_core.kernels.fpf_trimul.kernels", row, "cueq", needs=("trimul_forward", "pack_weights"))
    Tm = _by_name_or_core("fpf_trimul.trimul", "opt_core.kernels.fpf_trimul.trimul", row, "cueq", needs=("_build_config_table", "_select_cfg", "_schedule_grid", "smem_bytes"))
    if mode == "exact":
        _imp("cuequivariance_ops_torch", row, "torch_math")            # the exact LayerNorm stages are that library's kernels
    ac = _autocast_dtype(z)
    cdt = ac if ac is not None else z.dtype
    if cdt not in (torch.bfloat16, torch.float16, torch.float32):
        raise Refusal("dtype:%s" % str(cdt).replace("torch.", ""), row, "cueq")
    C = z.shape[-1]
    if C % 64:
        raise Refusal("c:%d_not_multiple_of_64" % C, row, "cueq")
    def _pack():
        def c_(x):
            return x.detach().to(cdt).contiguous()

        def n_(x):
            return x.detach().contiguous()
        return K.pack_weights(n_(w["ln_in_w"]), n_(w["ln_in_b"]), c_(torch.cat([w["w_ap"], w["w_bp"]], 0)), c_(torch.cat([w["w_ag"], w["w_bg"]], 0)),
                              n_(w["ln_out_w"]), n_(w["ln_out_b"]), c_(w["w_o"]), c_(w["w_og"]))
    wp = _packed(cache, "tmk3_w", w, _pack, extra=(str(cdt),))
    zs = z if z.dim() == 4 else z[None]
    ms = None if mask is None else (mask if mask.dim() == 3 else mask[None])
    B = int(zs.shape[0])
    N = int(zs.shape[1])
    cfg = tmk3_cfg(Tm, mode, cdt, N, C)
    if hasattr(K, "launch_grid_y") and K.launch_grid_y(N, C, cfg) > getattr(K, "CUDA_GRID_Y_MAX", 65535):
        raise Refusal("launch_grid_y>65535", row, "cueq")
    contract = cfg.get("contract") or ("cublas" if mode == "exact" else getattr(Tm, "CONTRACT", "cublas"))
    # One [N,N,C] launch set per pair; a batch writes each pair's output INTO its slice of one [B,N,N,C] tensor (no torch.stack copy), B = 1 and a
    # 3-D z return the kernel's own output tensor (a 4-D view of it for a 4-D z): the same bytes as before, one device copy per call fewer.
    if B == 1:
        zb = zs[0] if zs[0].is_contiguous() else zs[0].contiguous()
        mb = None if ms is None else ms[0]
        out = K.trimul_forward(zb, direction, mb, wp, eps=eps, residual=residual, cfg=cfg, contract=contract, cdt=cdt)
        return out[None] if z.dim() == 4 else out
    outz = torch.empty(tuple(zs.shape), dtype=(zs.dtype if residual else cdt), device=zs.device)
    for b in range(B):
        zb = zs[b] if zs[b].is_contiguous() else zs[b].contiguous()
        mb = None if ms is None else ms[min(b, ms.shape[0] - 1)]
        K.trimul_forward(zb, direction, mb, wp, eps=eps, residual=residual, cfg=cfg, contract=contract, cdt=cdt, out=outz[b])
    return outz


def _tx_module(w):
    from types import SimpleNamespace as NS
    P = lambda t: NS(weight=t)                         # noqa: E731
    return NS(layer_norm_in=NS(weight=w["ln_in_w"], bias=w["ln_in_b"]), layer_norm_out=NS(weight=w["ln_out_w"], bias=w["ln_out_b"]),
              linear_a_p=P(w["w_ap"]), linear_a_g=P(w["w_ag"]), linear_b_p=P(w["w_bp"]), linear_b_g=P(w["w_bg"]), linear_z=P(w["w_o"]), linear_g=P(w["w_og"]))


def _serve_tx(row, z, mask, direction, w, residual, eps, cache):
    import torch
    fb = "v4" if row == "tx_sm90a" else "tmk3_exact"
    cc = _device_cc(z)
    if cc != (9, 0):
        raise Refusal("cc:%d.%d!=9.0(sm_90a binary)" % cc, row, fb)
    ops = _imp("opt_core.kernels.trimul.tx_sm90a.ops", row, fb)
    if "tx_loaded" not in _MODS:
        try:
            ops.load()
            _MODS["tx_loaded"] = True
        except (ops.TrimulTxUnavailable, ImportError, OSError) as e:      # no binary for this ABI, or one that does not load in this interpreter
            _MODS["tx_loaded"] = "no_prebuilt:%s(%s:%s)" % (ops.abi_tag(), type(e).__name__, str(getattr(e, "reason", e))[:60].replace(" ", "_"))
    if _MODS["tx_loaded"] is not True:
        raise Refusal(_MODS["tx_loaded"], row, fb)
    zin = compute_input(z, cache)
    zs = zin if zin.dim() == 4 else zin[None]
    B, N, _, C = zs.shape
    if B != 1:
        raise Refusal("batch>1", row, fb)
    if C != 256 or w["w_ap"].shape[0] != 256:
        raise Refusal("c_z=%d,c_hidden=%d!=256" % (C, w["w_ap"].shape[0]), row, fb)
    if zs.dtype != torch.bfloat16:
        raise Refusal("dtype:%s!=bf16" % str(zs.dtype).replace("torch.", ""), row, fb)
    if N < 101:
        raise Refusal("n<101", row, "cueq")
    ck = ("tx_w",) + weights_key(w)
    wp = cache.get(ck)
    if wp is None:
        wp = cache[ck] = ops.pack_weights(_tx_module(w))
    m2 = None
    if mask is not None:
        m2 = (mask if mask.dim() == 2 else mask[0]).to(torch.float32).contiguous()
    z2 = zs[0].contiguous()
    fn = ops.trimul if row == "tx_sm90a" else ops.trimul_exact
    out = fn(z2, direction == "outgoing", m2, wp, residual=residual)
    return out if z.dim() == 3 else out[None]


def ef2_weights(w, direction, cache=None):
    """The ef2 row's weight dict from the ten canonical tensors (== ef2_trimul._weights(engine) for an ESM-family TriangleMultiplicativeUpdate)."""
    import torch
    ck = ("ef2_w", direction) + weights_key(w)
    if cache is not None and ck in cache:
        return cache[ck]
    bf = lambda t: t.detach().to(torch.bfloat16).contiguous()    # noqa: E731
    f32 = lambda t: t.detach().float().contiguous()              # noqa: E731
    ent = dict(ver=None, engine_ref=None, w_val=bf(torch.cat([w["w_ap"], w["w_bp"]], 0)), w_gate=bf(torch.cat([w["w_ag"], w["w_bg"]], 0)),
               g_in=f32(w["ln_in_w"]), b_in=f32(w["ln_in_b"]), g_out=f32(w["ln_out_w"]), b_out=f32(w["ln_out_b"]), w_p=bf(w["w_o"]), w_g=bf(w["w_og"]), flow=direction)
    ent["w_in"] = torch.cat([ent["w_val"], ent["w_gate"]], dim=0).contiguous()
    if cache is not None:
        cache[ck] = ent
    return ent


def _serve_ef2_fused(z, mask, direction, w, residual, eps, cache):
    import torch
    if not residual:
        raise Refusal("needs_residual", "ef2_fused", "cueq")
    E = _imp("opt_core.kernels.trimul.ef2.ef2_trimul", "ef2_fused", "cueq")
    zs = z if z.dim() == 4 else z[None]
    if zs.dtype != torch.bfloat16 or not zs.is_cuda:
        raise Refusal("dtype:%s!=bf16" % str(zs.dtype).replace("torch.", ""), "ef2_fused", "cueq")
    C = zs.shape[-1]
    if C % 64 or w["w_ap"].shape[0] != C:
        raise Refusal("c:%d/%d" % (C, w["w_ap"].shape[0]), "ef2_fused", "cueq")
    m = mask
    if m is not None:
        m = m if m.dim() == 3 else m[None]
        if m.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            m = m.to(torch.float32)
    out = E._TriMulResidual.apply(zs, m, ef2_weights(w, direction, cache))
    return out if z.dim() == 4 else out[0]


def esm_v5_pack(w, cache=None):
    """Row esm_v5_fwd's weight pack from the ten canonical tensors (the engine's relayout, plain torch; cached per weight identity)."""
    ck = ("esm_v5_w",) + weights_key(w)
    if cache is not None and ck in cache:
        return cache[ck]
    R = _imp("opt_core.kernels.trimul.esm_v5.route", "esm_v5_fwd", "v4")
    wp = R.pack({k: w[k] for k in WEIGHT_KEYS})
    if cache is not None:
        cache[ck] = wp
    return wp


def _serve_esm_v5(z, mask, direction, w, residual, eps, cache, config=None):
    """Row esm_v5_fwd: the carried forward line through .esm_v5.route (compute_input -> the row's bf16 rounding point under autocast)."""
    R = _imp("opt_core.kernels.trimul.esm_v5.route", "esm_v5_fwd", "v4")
    zin = compute_input(z, cache)
    try:
        R.kernels()
        wp = esm_v5_pack(w, cache)
        R.check(zin if zin.dim() in (3, 4) else zin, wp)
        cfg = None
        lv = None
        if isinstance(config, dict):
            if "k1" in config and "k3" in config:
                cfg = {"k1": dict(config["k1"]), "k3": dict(config["k3"])}
            lv = config.get("levers")
        if cfg is None:
            ck = ("esm_v5_cfg", str(zin.device))
            cfg = cache.get(ck)
            if cfg is None:
                cfg, key, _kind = R.cell(zin.device)
                cache[ck] = cfg
                cache["esm_v5_cell"] = key
        return R.forward(zin, direction == "outgoing", mask, wp, residual=residual, cfg=cfg, lever_overrides=lv, eps=eps)
    except R.Unavailable as e:
        raise Refusal(e.reason, "esm_v5_fwd", "v4")


def _serve_esm_v5_f32in(z, mask, direction, w, residual, eps, cache, config=None):
    """Row esm_v5_fwd:f32in: an fp32 (or tf32-mode fp32) z is cast ONCE to bf16 at the boundary, row esm_v5_fwd's line computes the update in
    bf16 / fp32-accumulate, the update is cast back to z's dtype; residual=True adds it to the fp32 z in fp32 (no bf16 rounding of z itself).
    A named tolerance-class lever for fp32-resident callers; bf16 callers are refused BY NAME towards esm_v5_fwd."""
    import torch
    if z.dtype == torch.bfloat16:
        raise Refusal("dtype:bf16(bf16 callers use esm_v5_fwd; this row is the fp32-in cast form)", "esm_v5_fwd:f32in", "esm_v5_fwd")
    if z.dtype != torch.float32:
        raise Refusal("dtype:%s(not fp32)" % str(z.dtype).replace("torch.", ""), "esm_v5_fwd:f32in", "cueq")
    try:
        upd = _serve_esm_v5(z.to(torch.bfloat16), mask, direction, w, False, eps, cache, config)
    except Refusal as e:
        raise Refusal(e.kind, "esm_v5_fwd:f32in", "cueq" if e.fallback in NEEDS_ESM_IMAGE or e.fallback == "cueq" else e.fallback)
    upd = upd.to(z.dtype)
    return z + upd if residual else upd


def _serve_esm_shapes(row, z, mask, direction, w, residual, eps, cache, config=None):
    """Rows esm_shapes / esm_shapes:f32in: opt_core.kernels.trimul_esm_shapes bound by name.  esm_shapes reads compute_input(z) (the bf16
    rounding point under autocast, or a bf16 z) and returns bf16; esm_shapes:f32in reads an fp32 z and returns fp32 (bf16 tensor-core operands,
    fp32 statistics / gates / residual: a named tolerance class).  ``config`` = a launch cell overriding the sub-package's table for one caller.
    Its ``Unsupported(word)`` becomes a Refusal BY NAME; kernel errors propagate."""
    import torch
    C = int(z.shape[-1])
    fb = ("v4" if C in (128, 256) else "cueq") if row == "esm_shapes" else "cueq"
    S = _imp("opt_core.kernels.trimul_esm_shapes", row, fb)
    if row == "esm_shapes":
        zin = compute_input(z, cache)
        if zin.dtype != torch.bfloat16:
            raise Refusal("dtype:%s(bf16 z or bf16 autocast; fp32 trunks use esm_shapes:f32in)" % str(zin.dtype).replace("torch.", ""), row, "esm_shapes:f32in" if zin.dtype == torch.float32 else fb)
    else:
        if z.dtype == torch.bfloat16:
            raise Refusal("dtype:bf16(bf16 callers use esm_shapes)", row, "esm_shapes")
        if z.dtype != torch.float32:
            raise Refusal("dtype:%s(not fp32)" % str(z.dtype).replace("torch.", ""), row, "cueq")
        zin = z.contiguous()
    m = None if mask is None else (mask[0] if (mask.dim() == 3 and int(mask.shape[0]) == 1) else mask)
    sub = cache.setdefault("esm_shapes", {}) if cache is not None else None
    try:
        out = S.triangle_multiplication(zin, m, direction=direction, weights=w, residual=residual, cache=sub, eps=eps, cell=config)
    except S.Unsupported as e:
        raise Refusal(e.word, row, fb)
    return out if out.shape == z.shape else out.reshape(z.shape)


def esm_v61_pack(w, cache=None):
    """Row esm_v61's weight pack from the ten canonical tensors (the package's pack_weights; cached per weight identity)."""
    ck = ("esm_v61_w",) + weights_key(w)
    if cache is not None and ck in cache:
        return cache[ck]
    E = _imp("opt_core.kernels.trimul.esm_v61", "esm_v61", "esm_v5_fwd")
    try:
        wp = E.pack({k: w[k] for k in WEIGHT_KEYS})
    except E.Unavailable as e:
        raise Refusal(e.kind, "esm_v61", "esm_v5_fwd")
    if cache is not None:
        cache[ck] = wp
    return wp


def _serve_esm_v61(z, mask, direction, w, residual, eps, cache, config=None):
    """Row esm_v61: the sealed package through .esm_v61 (install once per process: digests, driver binding, cubin load check, byte gate; then
    compute_input -> cached pack -> forward).  ``config`` {'fastsig': bool, 'gate': bool, 'binding': None|'ctypes'|'wheel'} for one caller."""
    E = _imp("opt_core.kernels.trimul.esm_v61", "esm_v61", "esm_v5_fwd")
    cfg = config if isinstance(config, dict) else {}
    zin = compute_input(z, cache)
    try:
        if "esm_v61_report" not in cache:
            cache["esm_v61_report"] = E.install(binding=cfg.get("binding"), gate=bool(cfg.get("gate", True)), device=zin.device)
        wp = esm_v61_pack(w, cache)
        return E.forward(zin, direction == "outgoing", mask, wp, residual=bool(residual), eps=eps, fastsig=bool(cfg.get("fastsig", True)))
    except E.Unavailable as e:
        fb = "esm_v5_fwd"
        if e.kind.startswith(("dtype:", "triton:", "import:triton")):
            fb = "v4"
        elif e.kind.startswith(("n<", "device:")):
            fb = "cueq"
        raise Refusal(e.kind, "esm_v61", fb)


_NATIVE_TOKENS = set()                                                       # (pid, device key) whose NATIVE_STAMP line was printed


def _native_token(T, device=None):
    """Once per (process, device): the row's first-serve facts as ONE stderr line + census context words, so a transcript says whether the payload byte
    gate ran here or a cached stamp honoured it (and how long install took):
    ``[opt_core] NATIVE_STAMP trimul native@<payload> pkg=<vN> cc=<cc> stamp=hit|written|unwritable|disabled:<reason>|not_consulted gate_ran=yes|no
    gate_s=<s> install_s=<s> dir=<env|triattn_root|xdg|home|tmp|->``.  Silent under OPT_CORE_CELL_CENSUS_PRINT=0.  Never raises, never gates."""
    import os as _os, sys as _sys
    try:
        idx = device if (device is None or isinstance(device, int)) else getattr(device, "index", None)
        key = (_os.getpid(), idx)
        if key in _NATIVE_TOKENS:
            return
        rep = T.report(idx)
        if not rep:
            return
        _NATIVE_TOKENS.add(key)
        st = str(rep.get("verdict_stamp", "-")); ran = "yes" if rep.get("gate_ran", True) else "no"
        line = "[opt_core] NATIVE_STAMP trimul native@%s pkg=%s cc=%s stamp=%s gate_ran=%s gate_s=%.2f install_s=%.2f dir=%s" % (
            str(rep.get("payload_version", "?")).split()[-1], rep.get("pkg", "?"), rep.get("cc", rep.get("device_class", "?")), st, ran,
            float(rep.get("gate_s") or 0.0), float(rep.get("install_s") or 0.0), rep.get("verdict_stamp_dir_kind") or "-")
        _CENSUS.set_context(native_stamp=st, native_gate_ran=ran, native_gate_s="%.2f" % float(rep.get("gate_s") or 0.0), native_install_s="%.2f" % float(rep.get("install_s") or 0.0))
        if _os.environ.get("OPT_CORE_CELL_CENSUS_PRINT", "") != "0":
            _sys.stderr.write(line + "\n"); _sys.stderr.flush()
    except Exception:                                                        # noqa: BLE001 -- a report line; it never gates or breaks a served call
        pass


def native_install(device=None, gate=True, binding=None):
    """Rows native / native_exact: check and load the sealed payload once per (process, device) -- digests, driver binding, cubin loads + load check,
    byte gate (``gate``) -- and return its report dict; ``Refusal`` by name (fallback v4) when it cannot serve here.  A kit may call this at start-up
    (outside CUDA-graph capture) so the first served call does no loading."""
    T = _imp("opt_core.kernels.trimul.native", "native", "v4")
    try:
        rep = T.install(device=device, gate=gate, binding=binding)
    except T.Unavailable as e:
        raise Refusal(e.kind, "native", "v4")
    _native_token(T, device)
    return rep


def _serve_native(row, z, mask, direction, w, residual, eps, cache, config=None, word=None):
    """Rows native / native_exact: the sealed trimul_native payload through .native (install once per process: digests, driver binding, cubin loads,
    load check, byte gate; then the payload's serve: its weight pack, launch plans, workspaces and tensor maps live in ``cache``).  z bf16 = the bf16
    form; z fp32 inside a bf16 autocast region = the f32z form (bf16 compute, update / residual sum in fp32).  ``config`` {'gate': bool, 'binding':
    None|'ctypes'|'cuda_bindings', 'incoming_mode' / 'k1_cfg' / 'k3_cfg': launch-plan overrides} for one caller; None = the payload's own table."""
    zs4 = z if z.dim() == 4 else z[None]
    C, H = int(zs4.shape[-1]), int(w["w_ap"].shape[0])
    prec, _cdt = call_precision(zs4)
    fb = _native_fallback(row, cc_word(_device_cc(zs4)), prec, C, H)
    cfg = dict(config) if isinstance(config, dict) else {}
    if row == "native:f32in":
        if prec not in ("fp32", "tf32"):
            raise Refusal("dtype:%s(fp32 / tf32 callers: bf16-resident and autocast callers use native)" % prec, row, "native")
        cfg["f32_resident_ok"] = True                                        # the payload's fp32-resident kernels outside an autocast region: fp32 z in, bf16 operands, fp32 out
    elif prec not in ("bf16", "f32z_bf16"):
        raise Refusal("dtype:%s(bf16 z, or fp32 z inside a bf16 autocast region; an fp32 / tf32 caller names native:f32in)" % prec, row, fb)
    T = _imp("opt_core.kernels.trimul.native", row, fb)
    wt = tuple(w[k] for k in WEIGHT_KEYS)                                    # the payload's per-call fast path keys on id(weights): hand it ONE dict object per weight set (kept in the
    ent = cache.get(NATIVE_WDICT_KEY)                                        # caller's cache, renewed only when a tensor OBJECT changes) -- a dict rebuilt per call made every served call
    if ent is None or len(ent[0]) != len(wt) or any(a is not b for a, b in zip(ent[0], wt)):   # miss it and pay validation + the cell-table scan + the pack key (~0.2-0.4 ms host per call,
        ent = (wt, {k: t for k, t in zip(WEIGHT_KEYS, wt)})                 # measured at c_z 128, N <= 512 on both cards)
        cache[NATIVE_WDICT_KEY] = ent
    try:
        out = T.forward(z, mask, direction=direction, weights=ent[1], residual=bool(residual), cache=cache, eps=eps,
                        variant=("exact" if row == "native_exact" else "fast"), config=cfg, gate=bool(cfg.get("gate", True)), binding=cfg.get("binding"),
                        pack_policy=("lru1" if str(word or "") == "big" else None))   # big: the shared cache keeps at most one layer's packed weights (residency); fast / exact: per layer
    except T.Unavailable as e:
        raise Refusal(e.kind, row, "cueq" if e.kind.startswith(("shape:n=", "weights:", "no_cell:")) else fb)
    _native_token(T, zs4.device.index)
    if prec != "bf16" and out.dtype != zs4.dtype:                            # the face's out-dtype contract for fp32 z (f32z / fp32 / tf32 callers): the caller's z dtype.  Payload 1.0.1
        out = out.to(zs4.dtype)                                              # returns the fp32 update at (128,128) but a bf16 update at (256,256) without residual (fp32 with residual at
        cache[("native_out_cast", C, H, bool(residual))] = str(zs4.dtype).replace("torch.", "")   # both): one cast pass added there, recorded per (width, residual)
    return out                                                               # in the caller's cache (NATIVE_OUT_CAST key) and in TRIMUL_CELLS rows.native.out_dtype; unified upstream later


def ef2_cueq_tiles_install(d_pair, cc=None):
    """Write the per-capability tile entries for d_pair into the stock library's in-process tuning cache (once per (d_pair, cc));
    returns (entries_written, reason) -- reason 'cc_untuned:sm_NN' when the carried table has no entry for this card (library default runs)."""
    E = _imp("opt_core.kernels.trimul.ef2.ef2_trimul", "ef2_cueq_tiles", "cueq")
    import torch
    cc = tuple(cc) if cc is not None else tuple(torch.cuda.get_device_capability())
    key = (int(d_pair), cc)
    if key in _TILES_INSTALLED:
        return _MODS.get(("tiles", key), (0, ""))
    cfg, why = E._cueq_tiles_for(cc)
    if cfg is None:
        res = (0, "%s:sm_%d%d" % (why, cc[0], cc[1]))
    else:
        saved = E._cueq_tiles_install(int(d_pair), cfg)
        _MODS[("tiles_saved", key)] = saved
        res = (len(saved), "")
    _TILES_INSTALLED.add(key)
    _MODS[("tiles", key)] = res
    return res


def ef2_cueq_tiles_restore():
    """Restore every library cache entry this module replaced."""
    E = _imp("opt_core.kernels.trimul.ef2.ef2_trimul", "ef2_cueq_tiles", "cueq")
    for k in [k for k in _MODS if isinstance(k, tuple) and k and k[0] == "tiles_saved"]:
        E._cueq_tiles_restore(_MODS.pop(k))
        _TILES_INSTALLED.discard(k[1])
        _MODS.pop(("tiles", k[1]), None)


def _serve_of3_form(z, mask, direction, w, residual, eps, cache, config=None):
    """Row of3_form: the OpenFold-family module's inference statement issued whole-tensor (kernels/trimul/of3_form.py) -- bitwise to that module
    where TRIMUL_CELLS.json records it (in_form records), tolerance-class to the cuEquivariance op.  Forward only; refusals by name."""
    import torch
    from . import of3_form as OF
    zs4 = z if z.dim() == 4 else z[None]
    m3 = mask if (mask is None or mask.dim() == 3) else mask.reshape(-1, *mask.shape[-2:])
    if m3 is not None and tuple(m3.shape) != tuple(zs4.shape[:3]):
        m3 = torch.broadcast_to(m3, zs4.shape[:3])
    why = OF.supported(zs4, m3, w)
    if why:
        raise Refusal(why.split("(")[0].replace(" ", "_")[:60], "of3_form", "torch_math")
    c = cache.setdefault("of3_form", {})
    if config is not None and str(config) not in OF.CONFIGS:
        raise Refusal("config:%s(not %s)" % (config, "|".join(OF.CONFIGS)), "of3_form", "torch_math")
    out = OF.forward(zs4, m3, direction, w, eps=eps, residual=residual, cache=c, config=(str(config) if config is not None else None))
    cache["of3_form.config"] = str(config) if config is not None else OF.DEFAULT_CONFIG
    cache["of3_form.ln"] = c.get("of3_form.ln")
    return out if z.dim() == 4 else out[0]


def _serve_af3t_form(z, mask, direction, w, residual, eps, cache, config=None):
    """Row af3t_form: the AF3-family module statement issued whole-tensor (kernels/trimul/af3t_form.py) -- bitwise to that module where
    TRIMUL_CELLS.json records it (in_form records); tolerance-class to the cuequivariance op and to of3_form.  Forward only; refusals by name."""
    from . import af3t_form as AF
    m = mask
    if m is not None and m.dim() == z.dim() - 1 and tuple(m.shape) != tuple(z.shape[:-1]):
        import torch
        m = torch.broadcast_to(m, z.shape[:-1])
    why = AF.supported(z, m, w)
    if why:
        raise Refusal(why.split("(")[0].replace(" ", "_")[:60], "af3t_form", "torch_math")
    if config is not None and str(config) not in AF.CONFIGS:
        raise Refusal("config:%s(not %s)" % (config, "|".join(AF.CONFIGS)), "af3t_form", "torch_math")
    c = cache.setdefault("af3t_form", {})
    out = AF.forward(z, m, direction, w, eps=eps, residual=residual, cache=c, config=(str(config) if config is not None else None))
    cache["af3t_form.config"] = str(config) if config is not None else AF.DEFAULT_CONFIG
    return out


def torch_math(z, mask=None, *, direction, weights, eps=1e-5, residual=False):
    """The torch_math row: the module statements in torch ops with the parameters as given (under autocast the GEMMs/einsum run in the autocast
    dtype exactly as an engine's module does).  Differentiable."""
    import torch
    import torch.nn.functional as F
    w = weights
    w = {k: (v.to(dtype=z.dtype) if hasattr(v, "dtype") and v.dtype != z.dtype and v.is_floating_point() else v) for k, v in w.items()} if hasattr(z, "dtype") else w   # weights follow z's dtype (fp32 masters against a bf16 pair tensor are cast here, as the fused rows pack theirs)
    g = w.get
    C = z.shape[-1]
    D = w["w_ap"].shape[0]
    x = F.layer_norm(z, (C,), w["ln_in_w"], w["ln_in_b"], eps)
    m = None if mask is None else mask.unsqueeze(-1).to(x.dtype)

    def proj(wg, bg, wp, bp):
        v = torch.sigmoid(F.linear(x, wg, bg)) * F.linear(x, wp, bp)
        return v if m is None else v * m
    a = proj(w["w_ag"], g("b_ag"), w["w_ap"], g("b_ap"))
    b = proj(w["w_bg"], g("b_bg"), w["w_bp"], g("b_bp"))
    if direction_word(direction) == "out":
        X = torch.einsum("...ikd,...jkd->...ijd", a, b)
    else:
        X = torch.einsum("...kid,...kjd->...ijd", a, b)
    del a, b
    upd = torch.sigmoid(F.linear(x, w["w_og"], g("b_og"))) * F.linear(F.layer_norm(X, (D,), w["ln_out_w"], w["ln_out_b"], eps), w["w_o"], g("b_o"))
    return z + upd if residual else upd


def reference(z, mask=None, *, direction, weights, dtype=None, eps=1e-5, residual=False):
    """Plain evaluation of the op in ``dtype`` (float64 by default) -- the numerics yardstick."""
    import torch
    dtype = torch.float64 if dtype is None else dtype
    w = {k: (v.detach().to(dtype) if hasattr(v, "detach") else v) for k, v in weights.items() if k in WEIGHT_KEYS + BIAS_KEYS and v is not None}
    with torch.autocast(device_type=z.device.type, enabled=False):
        return torch_math(z.detach().to(dtype), None if mask is None else mask.to(dtype), direction=direction, weights=w, eps=eps, residual=residual)


def triangle_multiplication(z, mask=None, *, direction, weights, word=None, selection=None, residual=False, prefer=None, residency=None,
                          stock=None, cache=None, eps=1e-5, config=None, pad=16, contract="bf16", form=None, module=None):
    """Serve the op through the selected row.  ``word`` (row name or tier word) or a ``selection`` from :func:`select` is required.  Returns the
    update (or z + update when ``residual``) in the row's output dtype (the rows follow the stock op: z's dtype, the autocast dtype for the
    fused bf16 rows reading a cast z).  Raises ``Refusal`` BY NAME when the row cannot serve this call; kernel errors propagate."""
    import torch
    cache = {} if cache is None else cache
    release_call_scratch(cache)                                              # a memo left by anything else is never read by this call
    try:
        zs4 = z if z.dim() == 4 else z[None]
        B, N, N2, C = zs4.shape
        if N != N2:
            raise ValueError("z must be [B,N,N,c] or [N,N,c] (got %s)" % (tuple(z.shape),))
        if weights.get("w_ap") is None:
            raise Refusal("weights_missing:w_ap", None, None)
        D = int(weights["w_ap"].shape[0])
        dirw = direction_word(direction)
        dname = "outgoing" if dirw == "out" else "incoming"
        if dirw == "outin":
            raise ValueError("serve one direction per call (outgoing | incoming)")
        if selection is None:
            if word is None:
                raise ValueError("word (row name or tier word) or selection is required")
            prec, _cdt = call_precision(zs4)
            bwd = bool(torch.is_grad_enabled() and (z.requires_grad or any(getattr(t, "requires_grad", False) for t in weights.values())))
            has_cueq = True if stock is not None else None
            st = cache.get("_stack")
            if st is None:
                st = cache["_stack"] = stack_word(zs4)
            skey = ("_sel", _device_cc(zs4), prec, C, D, N, dname, word, residency, bwd, tuple(prefer) if prefer else None, st, has_cueq, form, int(B))   # the leading batch
            selection = cache.get(skey)                                                  # extent is a fact of the call class (the exact word reads it: exact_batch_vouched)
            if selection is None:
                abi = cache.get("_abi")
                if abi is None and word in TIER_WORDS and (C, D) == (256, 256):        # the prebuilt-binary rows are c 256 rows: know this process's key before ranking them
                    try:
                        abi = cache["_abi"] = tx_abi_tag()
                    except (ImportError, OSError, AttributeError, RuntimeError, ValueError):   # an ops module that cannot name its ABI has no binary either
                        abi = cache["_abi"] = "unknown"
                sel_kw = dict(word=word, residency=("fp32" if prec.startswith("f32z_") else residency), backward=bwd, prefer=prefer, stack=st, tf32=(prec == "tf32"),
                              has_cueq=has_cueq, config=config, abi=abi, form=form, batch=int(B))
                sdt = prec.replace("f32z_", "") if prec.startswith("f32z_") else ("fp32" if prec == "tf32" else prec)
                selection = cache[skey] = select(skey[1], sdt, C, D, N, dname, **sel_kw)
                cache[("_selargs",) + skey[1:]] = (skey[1], sdt, C, D, N, dname, sel_kw)
        else:
            skey = None
            if (int(B) > 1 and selection.word == "exact" and selection.row in EXACT_ROWS and selection.row not in MODULE_EXACT_ROWS
                    and "+" not in str(selection.cell or "")):                           # a caller-built exact Selection (decided without the tensor: no batch
                st = cache.get("_stack")                                                 # extent) meets a BATCHED call.  The row's byte vouch is a batch-1 vouch; unless the cell
                if st is None:                                                           # records its batched vouch on this stack the call is re-decided WITH the extent (pure,
                    st = cache["_stack"] = stack_word(zs4)                               # memoised per class): the stock op of this process BY NAME (exact_batch_unvouched) --
                if not exact_batch_vouched(table()["cells"].get(str(selection.cell or "").split("+")[0]), selection.row, st, int(B)):   # no serve-time refusal, no step-aside
                    bkey = ("_selB", _device_cc(zs4), C, D, N, dname, selection.row, selection.stack, st, bool(selection.backward), int(B))
                    resel = cache.get(bkey)
                    if resel is None:
                        prec, _cdt = call_precision(zs4)
                        sdt = prec.replace("f32z_", "") if prec.startswith("f32z_") else ("fp32" if prec == "tf32" else prec)
                        resel = cache[bkey] = select(_device_cc(zs4), sdt, C, D, N, dname, word="exact", residency=("fp32" if prec.startswith("f32z_") else residency),
                                                     backward=bool(selection.backward), stack=st, tf32=(prec == "tf32"), has_cueq=(True if stock is not None else None),
                                                     batch=int(B))
                    selection = resel
        if str(selection.cell or "").endswith(PROVE_SUFFIX):                                # exact+<form>:prove outside the vouch table: proven per call shape against module=
            cache["_last"] = selection
            return _serve_proven(selection, z, zs4, mask, dname, weights, residual, eps, cache, config, pad, contract, stock, module)
        if selection.word not in TIER_WORDS or selection.row in STOCK_ROWS and selection.row == "torch_math":
            cache["_last"] = selection
            return _serve_row(selection, z, zs4, mask, dname, weights, residual, eps, cache, config, pad, contract, stock)
        # A TIER word never surfaces a row's refusal: a row that refuses with the tensors in hand (no prebuilt binary for this interpreter, a
        # library that fails to load, a shape its kernels turn out not to take) steps aside BY NAME -- recorded in cache["_stepaside"], said once on
        # stdout -- and the word serves the next measured row of the cell's order on this stack (select(exclude=...)); the stock rows end the chain.
        tried = []
        while True:
            cache["_last"] = selection
            try:
                return _serve_row(selection, z, zs4, mask, dname, weights, residual, eps, cache, config, pad, contract, stock)
            except Exception as e:                                                   # a Refusal by name, or -- on an INHERITED (unmeasured-capability) row only -- a build /
                from opt_core.oom import is_oom                                          # compile / launch failure (LAUNCH GUARD: _inherited_step_aside); an out-of-memory reaches the caller
                if is_oom(e):
                    raise
                args = cache.get((("_selargs",) + skey[1:])) if skey else None
                if not isinstance(e, Refusal):
                    _inherited_step_aside(selection, e, tried, cache, args[0] if args is not None else _device_cc(zs4))
                else:
                    if selection.row == "torch_math" or len(tried) >= len(ROW_NAMES) or "+" in str(selection.cell or ""):
                        raise                                                        # (a FORM selection never steps aside: the next exact row states the library op, not the module)
                    tried.append(selection.row)
                    cache.setdefault("_stepaside", []).append("%s(%s)" % (selection.row, e.kind))
                    _say_once(cache, "stepaside:%s:%s" % (selection.row, e.kind),
                              "kernels.trimul: tier word %r stepped aside from row %s on this stack (refused by name: %s); serving the next measured row of the cell"
                              % (selection.word, selection.row, e.kind))
                if args is not None:
                    cc_, sdt, C_, D_, N_, dn_, sel_kw = args
                    selection = select(cc_, sdt, C_, D_, N_, dn_, **dict(sel_kw, exclude=tuple(tried)))
                    cache[skey] = selection                                          # later calls of this process go straight to the row that served
                    _census_stepaside(cc_, sdt, C_, D_, N_, dn_, sel_kw, selection, cache.get("_stepaside") or [])
                else:                                                                # a caller-built tier Selection: re-select from its own facts
                    prec, _cdt = call_precision(zs4)
                    sdt = prec.replace("f32z_", "") if prec.startswith("f32z_") else ("fp32" if prec == "tf32" else prec)
                    selection = select(_device_cc(zs4), sdt, C, D, N, dname, word=selection.word, residency=("fp32" if prec.startswith("f32z_") else residency),
                                       backward=selection.backward, stack=cache.get("_stack") or stack_word(zs4), tf32=(prec == "tf32"),
                                       has_cueq=(True if stock is not None else None), exclude=tuple(tried), batch=int(B),
                                       form=((selection.cell or "").partition("+")[2] or None))      # a form Selection ('<cell>+<form>') re-selects under its form (refuses, never steps aside)
                    _census_stepaside(_device_cc(zs4), sdt, C, D, N, dname, dict(word=selection.word, backward=selection.backward, stack=cache.get("_stack"),
                                                                            residency=("fp32" if prec.startswith("f32z_") else residency), tf32=(prec == "tf32")),
                                      selection, cache.get("_stepaside") or [])
    finally:                                                             # the cast memo (compute_input) is scoped to THIS call: never hold a cast copy of the
        release_call_scratch(cache)                                      # caller's z (nor z itself) beyond the call that made it


def _is_oom(exc):
    """True for an out-of-memory error (re-raised as today, never a step-aside)."""
    return type(exc).__name__ in ("OutOfMemoryError",) or "out of memory" in str(exc).lower() or "CUDA_ERROR_OUT_OF_MEMORY" in str(exc)


def _inherited_step_aside(selection, exc, tried, cache, cc):
    """LAUNCH GUARD for an INHERITED (unmeasured-capability) portable row: a row that fails to BUILD / COMPILE / LAUNCH on the new architecture
    (any exception that is not an out-of-memory) is marked unavailable for the process on that capability (never retried: select() drops it from
    every inherited order), recorded as a step-aside ``<row>(stepped_aside:error:<ExcType>)`` (census NAMED_FALLBACK note, said once on stdout),
    and the caller serves the NEXT portable row of the donor cell, else the stock op by name.  Returns True when the loop steps aside; re-raises
    ``exc`` otherwise (a measured selection, a stock row, an out-of-memory error: unchanged behaviour)."""
    if isinstance(exc, Refusal) or "inherited_cc:unmeasured" not in str(getattr(selection, "reason", "") or "") or selection.row in STOCK_ROWS or _is_oom(exc):
        raise exc
    kind = "stepped_aside:error:%s" % type(exc).__name__
    _TABLE.setdefault("dead_rows", set()).add((selection.row, cc_word(cc)))
    if selection.row not in tried:
        tried.append(selection.row)
    cache.setdefault("_stepaside", []).append("%s(%s)" % (selection.row, kind))
    _say_once(cache, "stepaside:%s:%s:%s" % (selection.row, cc_word(cc), kind),
              "kernels.trimul: inherited row %s failed to launch on capability %s (%s: %s); unavailable for this process -- serving the next portable row of the donor cell"
              % (selection.row, cc_word(cc), type(exc).__name__, str(exc).split(chr(10))[0][:120]))
    return True


def _say_once(cache, key, text):
    """Print ``text`` once per process per key (step-asides are said by name, never silently)."""
    said = _TABLE.setdefault("said", set())
    if key in said:
        return
    said.add(key)
    print(text, flush=True)


def _census_stepaside(cc, dtype, c_z, c_hidden, n_tokens, direction, kw, selection, stepaside):
    """The serving call stepped aside from a row that refused with the tensors in hand: a NAMED_FALLBACK record (refused row:reason, the row now serving)."""
    try:
        key, _word, _inside = _census_key(cc, dtype, c_z, c_hidden, n_tokens, direction, kw, selection.cell)
        first = str(stepaside[0]) if stepaside else "-(-)"
        row, _, kind = first.partition("(")
        _CENSUS.record("trimul", key, "named_fallback", selection.row, cell_id=selection.cell, refused="%s:%s" % (row, kind[:-1] if kind.endswith(")") else kind),
                       note="stepaside(%s)" % "+".join(str(x).split("(")[0] for x in stepaside))
    except Exception:                                                     # the census counts; it never gates or breaks a served call
        return


def _serve_proven(selection, z, zs4, mask, dname, weights, residual, eps, cache, config, pad, contract, stock, module):
    """word 'exact+<form>:prove': the form's row for a call shape OUTSIDE the vouch table, served only once THIS exact shape tuple (row, config,
    device, dtypes, c, H, N, direction, mask, autocast, strides, TF32 state) has been proven bitwise against ``module`` -- the caller's own module
    statement as a callable ``module(z, mask) -> the same quantity this call returns`` (the update; z + update when residual) that does not mutate
    z.  The verdict is cached per shape tuple in ``cache`` (never per process, never for another shape); a failed proof raises Refusal
    `form_proof_failed(...)` now and on every later call of that shape (the caller keeps its module); no module= -> Refusal `prove_needs_module`."""
    import torch
    row = selection.row
    if module is None:
        raise Refusal("prove_needs_module(word exact+form:prove requires module=<the caller's module callable>)", row, selection.fallback or "torch_math")
    proofs = cache.setdefault("_form_proofs", {})
    m = mask
    key = (row, selection.config, str(zs4.device), str(zs4.dtype), tuple(zs4.shape[-3:]), int(weights["w_ap"].shape[0]), dname,
           None if m is None else (str(m.dtype), tuple(m.shape)), str(_autocast_dtype(zs4)), tuple(zs4.stride()), bool(residual),
           bool(torch.backends.cuda.matmul.allow_tf32), torch.get_float32_matmul_precision())
    verdict = proofs.get(key)
    if verdict is False:
        raise Refusal("form_proof_failed(shape=%s; proven unequal earlier in this cache)" % (selection.cell,), row, selection.fallback or "torch_math")
    out = _serve_row(selection, z, zs4, mask, dname, weights, residual, eps, cache, config, pad, contract, stock)
    if verdict is True:
        return out
    ref = module(z, mask)
    ok = isinstance(ref, torch.Tensor) and ref.shape == out.shape and ref.dtype == out.dtype and bool(torch.equal(ref, out))
    proofs[key] = ok
    if not ok:
        if not isinstance(ref, torch.Tensor) or ref.shape != out.shape or ref.dtype != out.dtype:
            why = "module returned %s" % (("%s %s" % (tuple(ref.shape), ref.dtype)) if isinstance(ref, torch.Tensor) else type(ref).__name__)
        else:
            why = "max_abs=%.3g" % float((ref.float() - out.float()).abs().max())
        raise Refusal("form_proof_failed(shape=%s;%s)" % (selection.cell, why), row, selection.fallback or "torch_math")
    return out


def _serve_row(selection, z, zs4, mask, dname, weights, residual, eps, cache, config, pad, contract, stock):
    """Serve exactly ``selection.row`` (raises the row's Refusal by name)."""
    row = selection.row
    w = _w10(weights, row)
    C, D = int(zs4.shape[-1]), int(weights["w_ap"].shape[0])
    # FIRST CALL of a (row, device, dtype, widths) class in this cache: the serving branch runs through the large-frame trampoline (_pystack) --
    # the class's one-off host work (a library import, the package's warm probe, Triton compiling the cell's ONE config) never executes on the
    # interpreter's stack-chunk boundary, where a syscall-heavy runtime turns seconds into a minute.  Later calls dispatch directly.
    warm = cache.get("_warm")
    if warm is None:
        warm = cache["_warm"] = set()
    wk = (row, str(zs4.device), str(zs4.dtype), C, D)
    if wk not in warm:
        warm.add(wk)
        stock_preload_once()                                    # the stack's cuequivariance import happens HERE (padded, once per process), not mid-item at a model's lazy import
        return _PS.padded_call(_dispatch, row, z, mask, dname, w, residual, eps, cache, pad, selection, config, contract, stock, zs4, C)
    return _dispatch(row, z, mask, dname, w, residual, eps, cache, pad, selection, config, contract, stock, zs4, C)


def _dispatch(row, z, mask, dname, w, residual, eps, cache, pad, selection, config, contract, stock, zs4, C):
    if row == "v4":
        return _serve_v4(z, mask, dname, w, residual, eps, cache, pad, selection.config)
    if row in ("tmk3_exact", "tmk3_fast"):
        return _serve_tmk3(row, z, mask, dname, w, residual, eps, cache)
    if row in ("tx_sm90a", "tx_sm90a_exact"):
        return _serve_tx(row, z, mask, dname, w, residual, eps, cache)
    if row == "ef2_fused":
        return _serve_ef2_fused(z, mask, dname, w, residual, eps, cache)
    if row == "esm_v5_fwd":
        return _serve_esm_v5(z, mask, dname, w, residual, eps, cache, selection.config if selection.config is not None else config)
    if row == "esm_v61":
        return _serve_esm_v61(z, mask, dname, w, residual, eps, cache, selection.config if selection.config is not None else config)
    if row in NATIVE_ROWS:
        return _serve_native(row, z, mask, dname, w, residual, eps, cache, selection.config if selection.config is not None else config, word=selection.word)
    if row == "esm_v5_fwd:f32in":
        return _serve_esm_v5_f32in(z, mask, dname, w, residual, eps, cache, selection.config if selection.config is not None else config)
    if row in ("esm_shapes", "esm_shapes:f32in"):
        return _serve_esm_shapes(row, z, mask, dname, w, residual, eps, cache, selection.config if selection.config is not None else config)
    if row in ("esm_k1ptr", "esm_k1ptr:f32in"):
        from . import esm_k1ptr as _K1P
        return _K1P.serve(row, z, mask, dname, w, residual, eps, cache, selection.config if selection.config is not None else config, compute_input, Refusal)
    if row == "ef2_cueq_tiles":
        try:
            _imp("cuequivariance_ops_torch", row, "torch_math")             # the tile entries patch this library's tables; without it the row cannot run here
            _imp("cuequivariance_torch", row, "torch_math")
        except Refusal as e:
            raise Refusal("needs_cuequivariance(%s)" % e.kind, row, "torch_math")
        n, why = ef2_cueq_tiles_install(C, _device_cc(zs4))
        cache["ef2_cueq_tiles"] = "entries=%d%s" % (n, (" reason=%s" % why) if why else "")
        out = _cueq_call(_cueq_fn(stock), z, mask, dname, w, eps, cache)
        return z + out if residual else out
    if row == "of3_form":
        return _serve_of3_form(z, mask, dname, w, residual, eps, cache, selection.config if selection.config is not None else config)
    if row == "af3t_form":
        return _serve_af3t_form(z, mask, dname, w, residual, eps, cache, selection.config if selection.config is not None else config)
    if row == "cueq":
        out = _cueq_call(_cueq_fn(stock), z, mask, dname, w, eps, cache)
        return z + out if residual else out
    if row == "torch_math":
        return torch_math(z, mask, direction=dname, weights=w, eps=eps, residual=residual)
    raise ValueError("row %r has no serving branch" % row)


serve = triangle_multiplication

# ------------------------------------------------------------------------------------------------------------------ first-call host cost
PRELOAD_MODULES = {                                              # row -> the modules its FIRST call imports (heavy third-party bodies first)
    "cueq": ("cuequivariance_torch",),
    "ef2_cueq_tiles": ("cuequivariance_ops_torch", "cuequivariance_torch"),
    "tmk3_exact": ("cuequivariance_ops_torch", "opt_core.kernels.fpf_trimul.kernels", "opt_core.kernels.fpf_trimul.trimul"),
    "tmk3_fast": ("opt_core.kernels.fpf_trimul.kernels", "opt_core.kernels.fpf_trimul.trimul"),
    "v4": ("opt_core.kernels.fpf_trimul_v4.generic", "opt_core.kernels.fpf_trimul_v4.kernels", "opt_core.kernels.fpf_trimul_v4.cells"),
    "tx_sm90a": ("opt_core.kernels.trimul.tx_sm90a.ops",),
    "tx_sm90a_exact": ("opt_core.kernels.trimul.tx_sm90a.ops",),
    "esm_v5_fwd": ("opt_core.kernels.trimul.esm_v5.route",),
    "esm_v5_fwd:f32in": ("opt_core.kernels.trimul.esm_v5.route",),
    "esm_v61": ("opt_core.kernels.trimul.esm_v61",),
    "native": ("opt_core.kernels.trimul.native",),
    "native_exact": ("opt_core.kernels.trimul.native",),
    "native:f32in": ("opt_core.kernels.trimul.native",),
    "ef2_fused": ("opt_core.kernels.trimul.esm_v5.route",),
    "esm_shapes": ("opt_core.kernels.trimul_esm_shapes",),
    "esm_shapes:f32in": ("opt_core.kernels.trimul_esm_shapes",),
    "esm_k1ptr": ("opt_core.kernels.trimul_esm_shapes", "opt_core.kernels.trimul.esm_k1ptr"),
    "esm_k1ptr:f32in": ("opt_core.kernels.trimul_esm_shapes", "opt_core.kernels.trimul.esm_k1ptr"),
    "of3_form": ("opt_core.kernels.trimul.of3_form",),
    "torch_math": (),
    "of3_form": ("torch", "opt_core.kernels.trimul.of3_form", "opt_core.kernels.ln"),          # the LayerNorm provider (exactln) at first call
    "af3t_form": ("torch", "triton", "opt_core.kernels.trimul.af3t_form"),                     # three small Triton kernels compile at first call (not importable ahead)
}


def preload(rows=None, *, stock_libraries=False):
    """Import, NOW and from the caller's frame through the large-frame trampoline, the modules the named rows' first call would import
    (``rows``: row names, default every row of PRELOAD_MODULES; ``stock_libraries=True`` adds cuequivariance_torch when it is installed --
    for a kit whose arm still reaches a stock cuequivariance op somewhere, so that library's import-time table is built here, at install,
    not inside the first item).  Returns ``{module: seconds | 'absent' | 'error:<word>'}``; nothing raises.  A kit calls this once from its
    lever-install code; the words never call it (serving imports only what the selected row needs, also through the trampoline)."""
    import time as _time
    names = []
    for r in (ROW_NAMES if rows is None else tuple(rows)):
        for m in PRELOAD_MODULES.get(r, ()):
            if m not in names:
                names.append(m)
    if stock_libraries and cueq_present() and "cuequivariance_torch" not in names:
        names.append("cuequivariance_torch")
    out = {}
    for m in names:
        if m.startswith("cuequivariance") and not cueq_present():
            out[m] = "absent"
            continue
        t0 = _time.perf_counter()
        try:
            _PS.import_padded(m)
            out[m] = round(_time.perf_counter() - t0, 3)
        except ImportError as e:
            out[m] = "absent" if m.split(".")[0] in str(getattr(e, "name", "") or "") else "error:%s" % str(e).split("\n")[0][:60].replace(" ", "_")
        except Exception as e:                                  # a binary that cannot load here etc.: the row refuses BY NAME at its call; preload never raises
            out[m] = "error:%s:%s" % (type(e).__name__, str(e).split("\n")[0][:60].replace(" ", "_"))
    return out


STOCK_PRELOAD_ENV = "OPT_CORE_NO_STOCK_PRELOAD"                      # engineering word (also OPT_CORE_NO_WARM_IMPORTS): set to time the stock library's import yourself
STOCK_PRELOAD = {}                                                # {module: seconds | 'present' | 'absent' | 'skipped:env' | 'error:<word>'}: what stock_preload_once() did in this process


def stock_preload_once():
    """At the FIRST engagement of any row (the face's first call, fpf_trimul_v4.generic's first call): the stack's heavy libraries
    (cuequivariance) imported through the trampoline, once per process -- ``opt_core.warm.auto_warm('trimul')`` (see :mod:`opt_core.warm`:
    every provider face shares that guard; a kit may call ``opt_core.warm_imports()`` at start-up instead).  Returns the warm report
    ``{library: seconds | 'present' | 'absent' | 'skipped:env' | 'error:<word>'}``; never raises.  Kept under this name for existing callers."""
    if STOCK_PRELOAD:
        return STOCK_PRELOAD
    try:
        from opt_core.warm import auto_warm
        STOCK_PRELOAD.update(auto_warm("trimul") or {"cuequivariance_torch": "decided"})
    except Exception as e:                                      # pragma: no cover - the warm module itself cannot fail to import; recorded, never raised
        STOCK_PRELOAD["cuequivariance_torch"] = "error:%s" % type(e).__name__
    if not STOCK_PRELOAD:
        STOCK_PRELOAD["cuequivariance_torch"] = "decided"
    return STOCK_PRELOAD
