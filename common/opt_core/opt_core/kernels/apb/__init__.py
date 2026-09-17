"""Attention with a pair bias: ONE provider over every carried implementation ("row") and the measured cell table (APB_CELLS.json).

The op (three boundaries; every row's ``rows`` entry in the table says which one it serves)::

    core      o[s,n,h,:] = sum_m softmax_m( scale * q[s,n,h,:].k[s,m,h,:] + bias[h,n,m] (+ key mask) ) v[s,m,h,:]   (optionally o *= sigmoid(gate))
              the bias is SHARED by the S samples (diffusion samples, MSA rows) or given per sample; S = 1 is the pairformer's single attention
    producer  bias planes = Linear_{c_z -> H}( LayerNorm(z) ) over the N^2 pair rows, written head-major ([H, N, ld]) for the core to read
    module    the whole AttentionPairBias module: LN(s), q/k/v/g projections, the producer, the core, sigmoid gate, output projection

Rows (``ROW_NAMES``; ``APB_CELLS.json["rows"]`` states each one's boundary, envelope, numerics class, capture safety and named fallback):

    apb_attn        opt_core.kernels.apb_attn through opt_core.attn.apb_core (the core's own Triton kernel; bf16/fp16 activations)      fast, core
    fpf_apb         kernels/apb/fpf_apb   (carried package: apb_views / dit_apb; variant fp16 = the precision cell)                        fast, core
    fpf_atom        kernels/apb/fpf_apb   (atom_triton.atom_apb: 32x128 local windows, keys read in place)                                fast, core (windowed)
    dit_exact       kernels/apb/dit_exact (carried package: prebuilt sm_90 CUDA replica of the fp32 memory-efficient SDPA kernel)           EXACT vs sdpa_upcast, core
    l3a             kernels/apb/l3a       (carried: one launch for S samples sharing one bias; op-for-op the dtk per-sample kernel)           fast, core (same bytes as dtk_loop)
    dtk_loop        opt_core.kernels.dtk_kernels.flash_bias_attn, one launch per sample                                                      fast, core
    sba             opt_core.attn.shared_bias_attn (flash_triattn dense, stride-0 sample axis; variants tf32 | ieee | tf32x3)              fast, core
    ef2_pairbias    kernels/apb/ef2       (carried module of the ESM-family sampler; imports that stack's transformers fork)                fast, core
    composed        ln_proj bias producer (head-major planes) + apb_attn core, torch around them                                             fast, module
    ln_proj         opt_core.kernels.ln_proj.pair_bias                                                                                        fast, producer
    fpf_pf_bias     kernels/apb/fpf_apb   (pf_triton.pf_bias)                                                                                fast, producer
    lnl_ln_linear   opt_core.kernels.lnl_fused.ln_linear (+ head slice, permute view)                                                        fast, producer
    dtk_window      opt_core.kernels.dtk_kernels.window_attn, one launch per sample                                                           fast, core (windowed)
    stock rows      sdpa (variants auto | cudnn | efficient | flash | math), sdpa_upcast, ds4sci (capture-UNSAFE, named), cueq_apb (variant cached_z),
                    torch_module, naive, naive_module, sdpa_gather, torch_ln_linear -- named, nothing carried

Selection.  ``select(cc, dtype, cell, n_tokens, word=...)`` -> ``Selection`` is pure (standard library + the table; no framework import):
    * ``word`` = a row name ("fpf_apb", "sdpa:cudnn", "l3a:c" -- ``row[:variant]``): the DEFAULT RULE is exactly that row (``admits`` may raise
      ``Refusal`` by name; nothing is substituted);
    * ``word`` = a tier word (fast | exact | faithful | big): the OPT-IN path -- the cell's measured winner on the caller's stack when the table
      measured that stack, else the cell's reference stack (``Selection.stack_measured``); ``prefer=`` narrows a tier to the caller's rows;
      ``capture=True`` says the call runs under CUDA-graph capture: capture-unsafe rows are refused by name and never a winner.
    Nothing here changes what an existing caller of opt_core.attn.apb_core / kernels.apb_attn / attn.shared_bias_attn / attn.sdpa_bias /
    kernels.atom_window / kernels.ln_proj executes: those modules are untouched; this face only adds the by-word door and the table.

Serving.  ``pair_bias_attention(q, k, v, bias, ...)`` (core rows), ``atom_attention(...)`` (windowed rows), ``pair_bias_planes(z, ...)`` (producer
rows) import torch / triton inside the call and dispatch to the selected row's module; module-boundary rows are named for a kit's own composition
(the reason string carries the recipe) and are not served by one function here.  ``reference(...)`` is the fp64 yardstick.

Standard library only at import.
"""
import json
import os
import re
from collections import namedtuple

from ... import cell_census as _CENSUS                            # stdlib-only: the coverage census (one record per decided call class; pure observation)

ROW_NAMES = ("apb_attn", "fpf_apb", "fpf_atom", "dit_exact", "l3a", "dtk_loop", "sba", "ef2_pairbias", "composed", "ln_proj", "fpf_pf_bias", "lnl_ln_linear",
             "dtk_window", "sdpa", "sdpa_upcast", "ds4sci", "cueq_apb", "torch_module", "naive", "naive_module", "sdpa_gather", "torch_ln_linear",
             "dit_fast", "torch_dit_rows", "atom_exact")
VARIANTS = {"fpf_apb": ("fp16",), "l3a": ("c",), "sba": ("tf32", "ieee", "tf32x3"), "sdpa": ("auto", "cudnn", "efficient", "flash", "math"), "cueq_apb": ("cached_z",),
            "dit_fast": ("atom",)}
ALIASES = {"ditfast": "dit_fast", "dit_fused": "dit_fast", "atom_fused": "dit_fast:atom", "atomfast": "dit_fast:atom"}   # the producing kits' lever words for the same kernels
DEFAULT_VARIANT = {"sba": "tf32", "sdpa": "auto"}
TIER_WORDS = ("fast", "exact", "faithful", "big")
STOCK_ROWS = ("sdpa", "sdpa_upcast", "ds4sci", "cueq_apb", "torch_module", "naive", "naive_module", "sdpa_gather", "torch_ln_linear", "torch_dit_rows")
EXACT_ROWS = ("dit_exact", "atom_exact")
CAPTURE_UNSAFE_ROWS = ("ds4sci",)
CARRIED_SUBPACKAGES = {"fpf_apb": "fpf_apb", "fpf_atom": "fpf_apb", "fpf_pf_bias": "fpf_apb", "dit_exact": "dit_exact", "l3a": "l3a", "ef2_pairbias": "ef2", "dit_fast": "ditfast",
                       "atom_exact": "atom_exact"}
BOUNDARY = {"apb_attn": "core", "fpf_apb": "core", "dit_exact": "core", "l3a": "core", "dtk_loop": "core", "sba": "core", "ef2_pairbias": "core", "sdpa": "core",
            "sdpa_upcast": "core", "ds4sci": "core", "naive": "core", "fpf_atom": "windowed", "atom_exact": "windowed", "dtk_window": "windowed", "sdpa_gather": "windowed",
            "composed": "module", "cueq_apb": "module", "torch_module": "module", "naive_module": "module",
            "ln_proj": "producer", "fpf_pf_bias": "producer", "lnl_ln_linear": "producer", "torch_ln_linear": "producer",
            "dit_fast": "blockrows", "torch_dit_rows": "blockrows"}
DIT_ROW_OPS = {None: ("adaln", "resgate_adaln", "swiglu", "gate", "resgate"),                       # dit_fast (token stream, c_a rows): ditfast/kernels.py
               "atom": ("adaln2", "resgate_adaln2", "gate2d", "swiglu2d")}                          # dit_fast:atom (atom stream, c_atom rows): ditfast/atom_kernels.py
CELL_WORDS = ("dit_h16d48", "pf_h16d24", "msarow_h8d32", "atom_h4d32w32x128", "mod_pf_c384cz128", "mod_dit_c768cz128",
              "bias_c128h16", "bias_c128h4", "bias_c64h4", "bias_c256h16", "bias_c384h16", "bias_c384h12",
              "ditrows_adaln_c768", "ditrows_resgate_adaln_c768", "ditrows_swiglu_c768", "ditrows_gate_c768", "ditrows_resgate_c768",
              "atomrows_adaln2_c128", "atomrows_resgate_adaln2_c128", "atomrows_gate2d_c128", "atomrows_swiglu2d_c128")
STOCK_OF_BOUNDARY = {"core": "sdpa", "windowed": "sdpa_gather", "module": "torch_module", "producer": "torch_ln_linear", "blockrows": "torch_dit_rows"}
DIT_EXACT_ABIS = ("torch2.13.0-cu130-sm90", "torch2.7.1-cu126-sm90")                # prebuilt/<key>/ directories of the carried package (cpython 3.11 builds, sm_90a)
DIT_EXACT_PYTHON = {"torch2.13.0-cu130-sm90": (3, 11), "torch2.7.1-cu126-sm90": (3, 11)}  # the interpreter each .so was built for (manifest "python"); another minor
                                                                                         # makes dit_exact_abi() append -cp<XY>, refused by name like any other stack
INT32_SAFE = 2 ** 31 - 2 ** 24                                                  # offset products at or above this are refused by name for the rows below
INT32_OFFSET_ROWS = ("fpf_apb", "fpf_atom")                                     # the carried package forms its per-(sample, head) base offsets in int32
                                                                                # (s*stride_s, h*stride_h, h*N*ld); every other served kernel is int64-addressed
CELLS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "APB_CELLS.json")

Selection = namedtuple("Selection", "row variant word cell size_measured stack stack_measured cls exact_vs backward capture_safe x_stock fallback reason config")


class Refusal(Exception):
    """A row cannot serve this call: ``kind`` is the word for the kit's LEVER line, ``row`` the row that refused, ``fallback`` the row the table
    names instead (the kit decides whether to take it -- nothing is substituted here)."""

    def __init__(self, kind, row=None, fallback=None):
        Exception.__init__(self, "%s%s%s" % (kind, (" [row %s]" % row) if row else "", (" -> fallback row %s" % fallback) if fallback else ""))
        self.kind, self.row, self.fallback = kind, row, fallback


_TABLE = {}


def table():
    """The parsed APB_CELLS.json (cached)."""
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


def dtype_word(dtype):
    """'bf16' | 'fp32' | 'fp16' from a word or a torch dtype."""
    s = str(dtype).replace("torch.", "")
    return {"bfloat16": "bf16", "bf16": "bf16", "float32": "fp32", "fp32": "fp32", "float": "fp32", "float16": "fp16", "fp16": "fp16", "half": "fp16"}.get(s, s)


def split_word(word):
    """'sdpa:cudnn' -> ('sdpa', 'cudnn'); 'apb_attn' -> ('apb_attn', None); the producing kits' lever words (ALIASES) resolve to their row."""
    word = ALIASES.get(word, word)
    if word and ":" in word:
        r, v = word.split(":", 1)
        return r, v
    return word, None


def arm_word(row, variant=None):
    return row if not variant else "%s:%s" % (row, variant)


def cell_word(kind, heads=None, head_dim=None, c_z=None, op=None):
    """The table's cell segment: kind = dit | pf | msarow | atom | module_pf | module_dit | bias (+ geometry).  None when no cell family lists the
    geometry (the caller then gets the boundary's stock row by name from ``select``)."""
    if kind == "dit":
        return "dit_h16d48" if (heads, head_dim) in ((16, 48), (None, None)) else None
    if kind == "pf":
        return "pf_h16d24" if (heads, head_dim) in ((16, 24), (None, None)) else None
    if kind == "msarow":
        return "msarow_h8d32" if (heads, head_dim) in ((8, 32), (None, None)) else None
    if kind == "atom":
        return "atom_h4d32w32x128" if (heads, head_dim) in ((4, 32), (None, None)) else None
    if kind == "module_pf":
        return "mod_pf_c384cz128"
    if kind == "module_dit":
        return "mod_dit_c768cz128"
    if kind == "bias":
        w = "bias_c%sh%s" % (c_z, heads)
        return w if w in CELL_WORDS else None
    if kind in ("ditrows", "atomrows"):                                             # cell_word("ditrows", op="adaln", c_z=768) / ("atomrows", op="gate2d", c_z=128)
        w = "%s_%s_c%s" % (kind, op, c_z if c_z is not None else (768 if kind == "ditrows" else 128))
        return w if w in CELL_WORDS else None
    return kind if kind in CELL_WORDS else None


_FAMILIES = {}


def cell_family(cc, dtype, cell, timing="eager"):
    """{(S, size): key} of the measured cells of one (cc, dtype, cell, timing) family (memoised)."""
    fk = (cc_word(cc), dtype_word(dtype), cell, timing)
    if fk not in _FAMILIES:
        fam = {}
        pre = "%s|%s|%s|" % fk[:3]
        for key in table()["cells"]:
            if key.startswith(pre) and key.split("|")[5] == timing:
                parts = key.split("|")
                fam[(int(parts[3][1:]), int(parts[4][3:]))] = key
        _FAMILIES[fk] = fam
    return _FAMILIES[fk]


def cell_key(cc, dtype, cell, n_tokens, samples=1, timing="eager"):
    """(key | None, size_measured, note): the cell serving this call.  Samples: the smallest measured S >= samples (else the largest); size: the
    smallest measured size >= n_tokens of that S (size_measured True), else the largest with size_measured False (beyond_measured).  A timing
    without cells falls to 'eager'.  (The notes 'nearest measured S' / 'timing .. not measured' mark NON-size neighbours: _select serves a tier
    word the boundary's stock row by name there -- the nearest rule is size-only.)"""
    fam = cell_family(cc, dtype, cell, timing)
    note = ""
    if not fam and timing != "eager":
        fam = cell_family(cc, dtype, cell, "eager")
        note = "timing %s not measured (eager cell)" % timing
    if not fam:
        return None, False, "no measured cell for %s|%s|%s" % (cc_word(cc), dtype_word(dtype), cell)
    esses = sorted({s for s, _n in fam})
    S = next((s for s in esses if s >= int(samples)), esses[-1])
    sizes = sorted(n for s, n in fam if s == S)
    n = int(n_tokens)
    snote = "" if S == int(samples) else "nearest measured S %d" % S
    for size in sizes:
        if n <= size:                                                            # inside the measured range: the smallest measured size >= n serves
            return fam[(S, size)], True, "; ".join(w for w in (note, snote) if w)
    return fam[(S, sizes[-1])], False, "; ".join(w for w in (note, snote, "beyond_measured(largest %d)" % sizes[-1]) if w)


def _row_facts(row):
    r = table()["rows"][row]
    return r["class"], r.get("exact_vs"), bool(r.get("backward")), bool(r.get("capture_safe", True)), r.get("fallback")


def int32_products(row, *, samples=1, n_tokens=None, heads=None, head_dim=None, n_atoms=None, n_keys=128, n_queries=32, ld=None):
    """The offset products the int32-addressed rows form (name -> value); empty for int64-addressed rows or unknown sizes."""
    if row not in INT32_OFFSET_ROWS:
        return {}
    H = int(heads or 16); D = int(head_dim or 64); S = int(samples or 1)
    if row == "fpf_atom":
        if n_atoms is None and n_tokens is None:
            return {}
        NA = int(n_atoms if n_atoms is not None else 8 * int(n_tokens))
        nb = -(-NA // int(n_queries))
        return {"S*NA*H*D": S * NA * H * D, "H*blocks*NQ*NK": H * nb * int(n_queries) * int(n_keys)}
    if n_tokens is None:
        return {}
    N = int(n_tokens); L = int(ld if ld is not None else -(-N // 8) * 8)
    return {"S*N*H*D": S * N * H * D, "H*N*ld": H * N * L, "H*N+N": H * N + N}


def admits(row, cc, dtype, cell=None, *, variant=None, head_dim=None, heads=None, c_z=None, capture=False, abi=None, has_cueq=None, has_ds4sci=None,
           has_esm=None, triton=None, key_mask=False, samples=None, n_tokens=None, n_atoms=None):
    """Raise ``Refusal`` when ``row`` cannot serve this envelope (static facts only: capability, dtype, geometry, capture state, ABI, library
    presence, int32 offset bounds).  Passing means the row's own admission (its module's checks) decides the rest at call time."""
    if row in INT32_OFFSET_ROWS:
        for name_, val in int32_products(row, samples=samples or 1, n_tokens=n_tokens, heads=heads, head_dim=head_dim, n_atoms=n_atoms).items():
            if val >= INT32_SAFE:
                raise Refusal("int32_offsets:%s=%d>=2^31-2^24" % (name_, val), row, {"fpf_apb": "apb_attn", "fpf_atom": "dtk_window"}[row])
    if row not in ROW_NAMES:
        raise ValueError("unknown row %r (rows: %s)" % (row, ", ".join(ROW_NAMES)))
    if row == "dit_fast":                                                            # the fused DiT block's row kernels: fp32 | bf16 activations (fp32 statistics inside), any cc triton serves
        if dtype_word(dtype) not in ("fp32", "bf16"):
            raise Refusal("dtype:%s(dit block rows serve fp32 | bf16 activations)" % dtype_word(dtype), row, "torch_dit_rows")
        if cell is not None and not str(cell).startswith(("ditrows_", "atomrows_")):
            raise Refusal("cell:%s(dit_fast serves the ditrows_* / atomrows_* cells)" % cell, row, "torch_dit_rows")
        if cell is not None and ((variant == "atom") != str(cell).startswith("atomrows_")):
            raise Refusal("variant:%s(cell %s is the %s stream)" % (variant or "token", cell, "atom" if str(cell).startswith("atomrows_") else "token"), row, "torch_dit_rows")
    if variant is not None and variant not in VARIANTS.get(row, ()):
        raise Refusal("variant:%s" % variant, row, row)
    ccw, dt = cc_word(cc), dtype_word(dtype)
    fb = _row_facts(row)[4]
    fb = (fb or "").split("|")[0] or None
    if capture and row in CAPTURE_UNSAFE_ROWS:
        raise Refusal("capture_unsafe", row, fb)
    if row == "apb_attn" or row == "composed":
        if dt == "fp32":
            raise Refusal("dtype:float32", row, fb)
        if head_dim is not None and int(head_dim) > 64:
            raise Refusal("head_dim:%s" % head_dim, row, fb)
        if float(ccw) < 8.0:
            raise Refusal("cc:%s" % ccw, row, fb)
        if row == "composed" and c_z is not None and int(c_z) not in (64, 128):
            raise Refusal("ln_proj:width:c%s" % c_z, row, fb)
    elif row == "fpf_apb":
        if head_dim is not None and int(head_dim) not in (24, 32, 48, 64):
            raise Refusal("head_dim:%s" % head_dim, row, fb)
        if variant == "fp16" and dt != "fp32":
            raise Refusal("variant:fp16 needs fp32 inputs", row, fb)
        if key_mask:
            raise Refusal("key_mask (fold -inf into the bias)", row, fb)
        if triton is not None and _ver(triton) < (3, 3) and float(ccw) >= 9.0:
            pass                                                                   # served: the non-TMA cell by name (not a refusal)
    elif row == "fpf_atom":
        if (heads is not None and int(heads) != 4) or (head_dim is not None and int(head_dim) != 32):
            raise Refusal("geometry:H%sD%s" % (heads, head_dim), row, fb)
    elif row == "atom_exact":
        if dt != "fp32":
            raise Refusal("dtype:%s (fp32 statement only)" % dt, row, fb)
        if (heads is not None and int(heads) % 4) or (head_dim is not None and int(head_dim) != 32):
            raise Refusal("geometry:H%sD%s (heads a multiple of 4, head_dim 32)" % (heads, head_dim), row, fb)
    elif row == "dit_exact":
        if ccw != "9.0":
            raise Refusal("cc:%s!=9.0(sm_90 binary)" % ccw, row, fb)
        if dt != "fp32":
            raise Refusal("dtype:%s (fp32 statement only)" % dt, row, fb)
        if head_dim is not None and int(head_dim) != 48:
            raise Refusal("head_dim:%s!=48" % head_dim, row, fb)
        if abi is not None and abi not in DIT_EXACT_ABIS:
            raise Refusal("no_prebuilt_for_stack:%s" % abi, row, fb)
        if key_mask:
            raise Refusal("key_mask (the statement has none: fold into the bias)", row, fb)
    elif row in ("l3a", "dtk_loop"):
        if head_dim is not None and int(head_dim) > 64:
            raise Refusal("head_dim:%s" % head_dim, row, fb)
        if triton is not None and _ver(triton) < (3, 0):
            raise Refusal("triton<3", row, fb)
    elif row == "sba":
        if head_dim is not None and int(head_dim) > 128:
            raise Refusal("head_dim:%s" % head_dim, row, fb)
        if key_mask:
            raise Refusal("key_mask (fold into the bias)", row, fb)
    elif row == "ef2_pairbias":
        if has_esm is False:
            raise Refusal("import:transformers.models.esmfold2", row, fb)
    elif row == "ln_proj":
        if c_z is not None and int(c_z) not in (64, 128):
            raise Refusal("width:c%s" % c_z, row, fb)
        if heads is not None and int(heads) > 64:
            raise Refusal("heads:%s" % heads, row, fb)
    elif row == "fpf_pf_bias":
        if c_z is not None and (int(c_z) & (int(c_z) - 1)):
            raise Refusal("c_pow2:c%s" % c_z, row, fb)
    elif row == "lnl_ln_linear":
        if c_z is not None and int(c_z) == 384:
            raise Refusal("BuildFailed:c384(tl.arange non-power-of-two)", row, fb)
        if c_z is not None and int(c_z) not in (64, 128, 256):
            raise Refusal("width:c%s" % c_z, row, fb)
    elif row in ("fpf_atom", "dtk_window"):
        pass
    elif row == "ds4sci":
        if has_ds4sci is False:
            raise Refusal("import:deepspeed", row, fb)
        if dt == "fp32":
            raise Refusal("dtype:float32", row, fb)
    elif row == "cueq_apb":
        if has_cueq is False:
            raise Refusal("import:cuequivariance_torch", row, fb)
    elif row == "sdpa":
        if variant == "flash":
            raise Refusal("backend:flash(no bias support)", row, "sdpa")
        if variant == "cudnn" and dt == "fp32":
            raise Refusal("backend:cudnn(fp32)", row, "sdpa")
    return True


def _ver(v):
    try:
        parts = str(v).split("+")[0].split(".")
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return (0, 0)


def _triton_of(stack):
    if not stack or "/" not in stack:
        return None
    try:
        return stack.split(":")[1].split("/")[1]
    except IndexError:
        return None


def _measured_class(cell, arm, st, cls):
    if cell and st in cell.get("stacks", {}):
        sc = cell["stacks"][st]
        if split_word(arm)[0] in EXACT_ROWS and arm in sc.get("bitwise_measured", []):
            return "bitwise"
        c = sc.get("class", {}).get(arm)
        if c:
            return c
    return cls


def _stock_for(cell_word_):
    if cell_word_ is None:
        return "sdpa"
    if cell_word_.startswith(("ditrows_", "atomrows_")):
        return "torch_dit_rows"
    if cell_word_.startswith("atom"):
        return "sdpa_gather"
    if cell_word_.startswith("mod_"):
        return "torch_module"
    if cell_word_.startswith("bias_"):
        return "torch_ln_linear"
    return "sdpa"


_SKIP_RE = re.compile(r"skip ([^(;\s]+)\(([^)]*)\)")
_VOUCH_RE = re.compile(r"exact vouch not recorded on (an unnamed stack[^:]*|\S+): (\S+) -> floor (\S+)")   # _select's note when an exact arm is held back to the floor on this stack


def select(cc, dtype, cell, n_tokens, *, word, samples=1, timing="eager", prefer=None, stack=None, capture=False, head_dim=None, heads=None, c_z=None,
           abi=None, has_cueq=None, has_ds4sci=None, has_esm=None, config=None):
    """The row + facts for this call (see ``_select``); the decision is recorded ONCE per call class in :mod:`opt_core.cell_census` (pure
    observation: the Selection / Refusal is decided first and returned unchanged)."""
    kw = dict(word=word, samples=samples, timing=timing, prefer=prefer, stack=stack, capture=capture, head_dim=head_dim, heads=heads, c_z=c_z, abi=abi,
              has_cueq=has_cueq, has_ds4sci=has_ds4sci, has_esm=has_esm, config=config)
    try:
        sel = _select(cc, dtype, cell, n_tokens, **kw)
    except Refusal as e:
        _census(cc, dtype, cell, n_tokens, kw, None, e)
        raise
    _census(cc, dtype, cell, n_tokens, kw, sel, None)
    return sel


def _census_vouch(ckey, sel, served, reason):
    """``<winner>:exact_vouch_not_recorded_on:<stack>`` when _select held the cell's EXACT winner back to the floor on this stack (its note
    ``exact vouch not recorded on <stack>: <dropped arms> -> floor <arm>`` names the winner among the dropped and the floor served), else None
    (a vouched arm served; lower-ranked unvouched arms dropped behind it are not a fallback)."""
    if "exact vouch not recorded on" not in (reason or ""):
        return None
    m = _VOUCH_RE.search(reason)
    if m is None:
        return None
    dropped, floor = m.group(2).split(","), m.group(3)
    tcell = table()["cells"].get(ckey) or {} if ckey else {}
    sc = (tcell.get("stacks") or {}).get(sel.stack) or {}
    winner = (tcell.get("exact_per_stack") or {}).get(sel.stack) or sc.get("exact") or tcell.get("exact") or sc.get("stock_arm")   # as _select ranks the exact tier
    if served != floor or (winner is not None and winner not in dropped):
        return None
    on = "unnamed_stack" if m.group(1).startswith("an unnamed") else m.group(1)
    return "%s:%s_on:%s" % (winner or dropped[0], _CENSUS.EXACT_VOUCH_WORD, on)


def _census(cc, dtype, cell, n_tokens, kw, sel, refusal):
    """Classify the decision (cell_hit | inherited | named_fallback | stock | opt_in) from the Selection's own facts and record it."""
    try:
        ckey = sel.cell if sel is not None else None
        inside = False
        if ckey:
            try:
                inside = int(n_tokens) <= int(str(ckey).split("|")[4][3:])
            except (IndexError, ValueError):
                inside = False
        timing = "graph" if (kw.get("capture") or kw.get("timing") == "graph") else "eager"
        shape = cell if cell else "h%sd%scz%s" % (kw.get("heads"), kw.get("head_dim"), kw.get("c_z"))
        key = dict(cc=cc_word(cc), stack=kw.get("stack"), dtype=dtype_word(dtype), shape=shape, bucket=_CENSUS.bucket_word(n_tokens, ckey, inside),
                   form="%s.S%s" % (timing, kw.get("samples")), word=kw.get("word"))
        word = kw.get("word")
        if refusal is not None:
            _CENSUS.record("apb", key, "named_fallback", "caller:%s" % (refusal.fallback or "-"), cell_id=None,
                           refused="%s:%s" % (refusal.row or word, refusal.kind), note="raised")
            return
        reason, stack, served = sel.reason or "", kw.get("stack"), arm_word(sel.row, sel.variant)
        if reason.startswith(INHERIT_TOKEN):                                     # an unmeasured capability served from the nearest measured column
            _CENSUS.record("apb", key, "inherited", served, cell_id=ckey, note=reason.split(";")[0])
        elif split_word(word)[0] in ROW_NAMES:
            _CENSUS.record("apb", key, "opt_in", served, cell_id=ckey, note=reason if (ckey is None or not inside) else "")
        elif "skip " in reason:
            m = _SKIP_RE.search(reason)
            _CENSUS.record("apb", key, "named_fallback", served, cell_id=ckey, refused=("%s:%s" % (m.group(1), m.group(2))) if m else "-:%s" % reason)
        elif ckey is None:
            _CENSUS.record("apb", key, "inherited", served, cell_id=None, note="family:none(%s)" % reason)
        elif _census_vouch(ckey, sel, served, reason) is not None:           # EXACT is stack-specific: the cell's exact winner held back BY NAME on this stack, the floor serves (never inherited)
            _CENSUS.record("apb", key, "named_fallback", served, cell_id=ckey, refused=_census_vouch(ckey, sel, served, reason))
        else:
            sib = _CENSUS.sibling_note(sel.stack) if (stack and not sel.stack_measured) else ""   # the cell's timing column is a sibling stack's (same cc): a hit, said so
            if not inside or "nearest measured S" in reason or "not measured (eager cell)" in reason:   # a TRUE gap: a size neighbour (| a sample / timing neighbour, were one served)
                why = ("beyond_measured" if not inside else "guard:samples(S%s)" % kw.get("samples") if "nearest measured S" in reason else "guard:timing(graph:eager_cell)")
                _CENSUS.record("apb", key, "inherited", served, cell_id=ckey, note=why + ((";" + sib) if sib else ""))
            elif sel.row in STOCK_ROWS:
                _CENSUS.record("apb", key, "stock", served, cell_id=ckey, note=sib)
            else:
                _CENSUS.record("apb", key, "cell_hit", served, cell_id=ckey, note=sib or ("prefer" if "prefer" in reason else ""))
    except Exception:                                                     # the census counts; it never gates or breaks a selection
        return



# ---------------------------------------------------------------- unmeasured capability: inherit the nearest arch-compatible measured column
INHERIT_TOKEN = "inherited_cc:unmeasured"                       # census / reason token: "<token>(<cc>-><column>)"


def measured_columns(family=None):
    """The capability words holding at least one measured cell ({"9.0", "8.0"} today) -- of the given family (dtype word, cell word, timing)
    when named (any sample count / size bucket), else of any family."""
    if family is None:
        return sorted({k.split("|")[0] for k in table()["cells"]})
    dtw, cellw, timing = family
    out = set()
    for k in table()["cells"]:
        parts = k.split("|")
        if parts[1] == dtw and parts[2] == str(cellw) and parts[5] == timing:
            out.add(parts[0])
    return sorted(out)


_DEAD_ARMS = {}                                                   # capability word -> {arm: "ExcType"}: inherited rows that failed to build / compile / launch on that part (this process)


def dead_arms(cc):
    """Arms an inherited (unmeasured-capability) selection may no longer resolve on ``cc`` in this process (first launch failed)."""
    return dict(_DEAD_ARMS.get(cc_word(cc), {}))


def mark_dead(cc, arm, exc, *, cell=None, dtype=None, n_tokens=None, kw=None):
    """Record that ``arm`` (an inherited portable row) raised ``exc`` at launch on ``cc``: excluded from inherited selections on that part from now
    on; the census hears ``<arm>:stepped_aside:error:<ExcType>`` ONCE."""
    d = _DEAD_ARMS.setdefault(cc_word(cc), {})
    first = arm not in d
    d[arm] = type(exc).__name__
    if first:
        try:
            kw2 = dict(kw or {}); kw2["word"] = "%s>%s" % (kw2.get("word") or "-", arm)          # one census line PER retired arm (the key names it)
            _census(cc, dtype, cell, n_tokens, kw2, None, Refusal("stepped_aside:error:%s" % type(exc).__name__, split_word(arm)[0], None))
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError):   # bookkeeping never gates
            pass


def inherit_column(cc, family=None):
    """The measured column a capability inherits fast / big rows from FOR ONE FAMILY KEY, or None: ``family`` = (dtype word, cell word,
    timing[, pass]) -- the columns considered are those holding at least one cell of that family (any size bucket, any sample count); without ``family``
    any cell counts.  A capability that has the family itself inherits nothing (its own cells decide); otherwise the highest such column at
    or below it (10.0 / 10.3 / 12.0 -> 9.0; 8.6 / 8.9 -> 8.0); below the lowest -> None.  Decided per key: a capability holding cells for
    OTHER keys still inherits for this one (``partial_cc``)."""
    ccw = cc_word(cc)
    cols = measured_columns(family)
    if ccw in cols:
        return None
    try:
        mine = tuple(int(x) for x in ccw.split("."))
    except ValueError:
        return None
    below = [c for c in cols if tuple(int(x) for x in c.split(".")) <= mine]
    return max(below, key=lambda c: tuple(int(x) for x in c.split("."))) if below else None


PORTABLE_ROWS = frozenset(("apb_attn", "fpf_apb", "fpf_atom", "l3a", "dtk_loop", "sba", "ef2_pairbias", "ln_proj", "fpf_pf_bias", "lnl_ln_linear", "dtk_window", "dit_fast", "composed"))
                                                                  # rows whose device code is compiled from source on the running part (Triton): they cross to an
                                                                  # unmeasured capability.  Never across arch: dit_exact / atom_exact (binaries built per arch), ds4sci
                                                                  # (a CUTLASS extension built per arch), cueq_apb (a library); the SDPA / torch statements serve only
                                                                  # when no portable row admits.


def _select(cc, dtype, cell, n_tokens, *, word, **kw):
    """The row + facts for this call (see ``_select_measured``).  Inheritance is decided PER FAMILY KEY: when this capability holds no cell
    of the asked family (dtype form, cell word, timing form, any sample count) -- whether or not it holds cells for other keys -- the tier words
    fast | big | faithful inherit the nearest arch-compatible measured column that does (``inherit_column``): that column's cell decides
    with its source-compiled rows (PORTABLE_ROWS) ranked first, rows built for one arch never crossing, the statement only when no portable
    row admits; reason / census token ``inherited_cc:unmeasured(<cc>-><column>)`` (+ ``partial_cc`` when this capability holds cells for
    other keys).  The exact word there = the statement BY NAME (no byte vouch on this capability).  A capability holding the family decides
    for itself (its buckets, its named fallbacks); row words are the caller's explicit choice and are never redirected."""
    rw = split_word(word)[0] if word is not None else None
    if rw not in TIER_WORDS:
        return _select_measured(cc, dtype, cell, n_tokens, word=word, **kw)
    own = _select_measured(cc, dtype, cell, n_tokens, word=word, **kw)
    if own.cell is not None or "no_family" not in (own.reason or ""):
        return own                                                                   # this capability measured the family: its own decision
    family = (str(dtype).replace("torch.", "").replace("bfloat16", "bf16").replace("float32", "fp32").replace("float16", "fp16"), str(cell), "graph" if (kw.get("capture") or kw.get("timing") == "graph") else "eager")
    col = inherit_column(cc, family)
    if col is None:
        return own
    token = "%s(%s->%s)%s" % (INHERIT_TOKEN, cc_word(cc), col, ";partial_cc" if cc_word(cc) in measured_columns() else "")
    if rw == "exact":
        stock_row = _stock_for(cell)
        cls, exact_vs, bwd, cap, fb = _row_facts(stock_row)
        return Selection(stock_row, DEFAULT_VARIANT.get(stock_row), word, None, False, None, False, cls, exact_vs, bwd, cap, None, fb,
                         "%s; exact on an unmeasured capability = the statement by name (no byte vouch on %s)" % (token, cc_word(cc)), kw.get("config"))
    kw2 = dict(kw); kw2["stack"] = None                                              # the inherited column's reference stack decides (this part has no stack column there)
    sel = _select_measured(col, dtype, cell, n_tokens, word=word, _portable_first=True, _admit_cc=cc, _exclude=tuple(dead_arms(cc)), **kw2)
    if sel.cell is None:
        return own
    return sel._replace(reason=(token + "; " + (sel.reason or ""))[:400])

def _select_measured(cc, dtype, cell, n_tokens, *, word, samples=1, timing="eager", prefer=None, stack=None, capture=False, head_dim=None, heads=None, c_z=None,
            abi=None, has_cueq=None, has_ds4sci=None, has_esm=None, config=None, _portable_first=False, _admit_cc=None, _exclude=()):
    """The row + facts for this call.  ``cell``: a CELL_WORDS entry (see ``cell_word``) or None (no family lists the geometry: the boundary's stock
    row by name for a tier word).  ``word``: a row name (``row[:variant]``; default rule: exactly that row) or a tier word (fast | exact |
    faithful | big: the cell's measured winner on ``stack`` when measured there, else on the cell's reference stack).  ``prefer``: restrict a
    tier to these arms (the first one the cell measured, in the caller's order; else the cell's stock arm -- reason 'prefer(first measured)').
    ``capture``: the call is captured in a CUDA graph (capture-unsafe rows refused / skipped by name; graph-timing cells consulted).  Pure."""
    if word is None:
        raise ValueError("word is required: a row name (%s) or a tier word (%s)" % (", ".join(ROW_NAMES), ", ".join(TIER_WORDS)))
    timing = "graph" if (capture or timing == "graph") else "eager"
    key, measured, note = (cell_key(cc, dtype, cell, n_tokens, samples, timing) if cell else (None, False, "no cell family for this geometry"))
    tcell = table()["cells"].get(key) if key else None
    trit = _triton_of(stack)
    akw = dict(samples=samples, n_tokens=n_tokens, head_dim=head_dim, heads=heads, c_z=c_z, capture=capture, abi=abi, has_cueq=has_cueq, has_ds4sci=has_ds4sci, has_esm=has_esm, triton=trit)
    st_measured = bool(tcell and (stack is None or stack in tcell.get("stacks", {})))      # no stack named = the cell's reference stack (measured)
    st = stack if (tcell and stack in tcell.get("stacks", {})) else (tcell.get("ref_stack") if tcell else None)
    rw, variant = split_word(word)
    if rw in ROW_NAMES:
        if variant is None:
            variant = DEFAULT_VARIANT.get(rw)                                    # a plain `sba` / `sdpa` word IS its default variant: the Selection says which arm its facts are
        admits(rw, cc, dtype, cell, variant=variant, **akw)
        cls, exact_vs, bwd, cap, fb = _row_facts(rw)
        arm = arm_word(rw, variant)
        x = None
        if tcell and st:
            sc = tcell["stacks"][st]
            if sc.get("stock_ms") and sc.get("ms", {}).get(arm):
                x = round(sc["stock_ms"] / sc["ms"][arm], 3)
        return Selection(rw, variant, word, key, measured, st, st_measured, _measured_class(tcell, arm, st, cls), exact_vs, bwd, cap, x, fb,
                         note or ("row word" if tcell else "row word; no cell measured"), config)
    if rw not in TIER_WORDS:
        raise ValueError("unknown word %r (rows: %s; tiers: %s)" % (word, ", ".join(ROW_NAMES), ", ".join(TIER_WORDS)))
    stock_row = _stock_for(cell)
    if tcell is None:                                                            # no cell family for this geometry: the boundary's stock row BY NAME (a named fallback)
        cls, exact_vs, bwd, cap, fb = _row_facts(stock_row)
        return Selection(stock_row, DEFAULT_VARIANT.get(stock_row), word, None, False, None, False, cls, exact_vs, bwd, cap, None, fb,
                         "skip cells(no_family:%s); named stock row: %s" % (cell if cell else "h%sd%s" % (heads, head_dim), note), config)
    # nearest = SIZE only: a neighbour on another dimension (sample count, timing form) never lends its winner to a tier word -- the
    # boundary's stock row serves BY NAME (a named fallback the census records; a measured cell for that key is the way to a kernel here)
    unmeasured_dim = ("samples_unmeasured:S%s;measured:%s" % (samples, "+".join("S%d" % s_ for s_ in sorted({int(k.split("|")[3][1:]) for k in table()["cells"] if k.split("|")[:3] == key.split("|")[:3]})))
                      if "nearest measured S" in (note or "") else
                      "timing_unmeasured:%s" % timing if "not measured (eager cell)" in (note or "") else None)
    if unmeasured_dim:
        cls, exact_vs, bwd, cap, fb = _row_facts(stock_row)
        return Selection(stock_row, DEFAULT_VARIANT.get(stock_row), word, key, measured, st, st_measured, cls, exact_vs, bwd, cap, None, fb,
                         "skip cells(%s); named stock row (the nearest rule is size-only); %s" % (unmeasured_dim.replace(";", ","), note), config)
    sc = tcell["stacks"][st]
    ms = sc.get("ms", {})
    winner = (tcell.get("%s_per_stack" % rw, {}).get(st) if rw in ("fast", "exact", "big") else None) or sc.get(rw) or tcell.get(rw) or sc.get("stock_arm") or stock_row
    excluded = tcell.get("excluded_arms") or {}                                      # arms an engine measured slower in-model in this class: never a tier word's answer
    admissible = [a for a in sorted(ms, key=lambda a: ms[a]) if not a.startswith("x:") and a not in sc.get("outside_band", []) and split_word(a)[0] in ROW_NAMES
                  and a not in excluded]
    if winner in excluded:
        winner = admissible[0] if admissible else stock_row
        note = (note + "; " if note else "") + "excluded_arm -> %s" % winner
    if rw == "exact":
        order = [winner] + [a for a in admissible if a != winner and split_word(a)[0] in STOCK_ROWS]
        vo = tcell.get("vouched_on", {})                                            # EXACT VOUCH IS STACK-SPECIFIC: a non-stock arm serves the exact word only on a stack
        floor = sc.get("stock_arm") or stock_row                                    # where its bitwise identity was measured; elsewhere (or stack unknown) the floor, by name
        kept = [a for a in order if a == floor or (stack is not None and stack in vo.get(a, ()))]
        dropped = [a for a in order if a not in kept]
        if dropped:
            unv_note = "exact vouch not recorded on %s: %s -> floor %s" % (stack or "an unnamed stack (pass stack=)", ",".join(dropped), floor)
            note = (note + "; " if note else "") + unv_note
        order = kept or [floor]
        winner = order[0]
    elif rw == "faithful":
        order = [winner] + [a for a in admissible if a != winner and (sc.get("class", {}).get(a) in ("bitwise", "stock") or (sc.get("relrms", {}).get(a) or 1.0) <= 3e-5)]
    else:
        order = [winner] + [a for a in admissible if a != winner]
    if capture:
        order = [a for a in order if split_word(a)[0] not in CAPTURE_UNSAFE_ROWS] or [stock_row]
    if _exclude:                                                                     # inherited rows that died at launch on the running part: never again in this process
        order = [a for a in order if a not in _exclude] or [stock_row]
    if _portable_first:                                                              # an unmeasured capability inheriting this column: source-compiled rows first,
        port = [a for a in order if split_word(a)[0] in PORTABLE_ROWS]               # rows built for another arch never, the statements only when no portable row admits
        rest = [a for a in order if split_word(a)[0] in STOCK_ROWS and split_word(a)[0] not in ("ds4sci", "cueq_apb")]
        order = port + rest or [stock_row]
    chosen, why = None, ""
    if prefer:
        pref = [p for p in prefer]
        for p in pref:                                                          # an exact arm word first ('l3a' = the plain variant), then any variant of the row
            hit = [a for a in order if a == p or (split_word(p)[1] is None and a == arm_word(p, DEFAULT_VARIANT.get(p)))] or \
                  [a for a in order if split_word(a)[0] == split_word(p)[0] and split_word(p)[1] is None]
            if hit:
                chosen = hit[0]
                break
        if chosen is None:
            for a in order:
                if split_word(a)[0] in STOCK_ROWS:
                    chosen = a
                    break
        if chosen is None:
            raise Refusal("prefer:%s not measured in cell %s" % ("+".join(pref), key), None, stock_row)
        why = "prefer" if (chosen == pref[0] or split_word(chosen)[0] == pref[0]) else "prefer(first measured)"
    else:
        chosen = order[0]
    for a in [chosen] + [x for x in order if x != chosen]:
        r, v = split_word(a)
        if r not in ROW_NAMES:
            continue
        try:
            admits(r, _admit_cc or cc, dtype, cell, variant=v, **akw)
        except Refusal as e:
            why = (why + ";" if why else "") + "skip %s(%s)" % (a, e.kind)
            continue
        cls, exact_vs, bwd, cap, fb = _row_facts(r)
        x = round(sc["stock_ms"] / ms[a], 3) if (sc.get("stock_ms") and ms.get(a)) else None
        reason = ("%s winner on %s" % (rw, st)) + ("" if st_measured or not stack else " (stack not measured: reference stack)") + ("; " + why if why else "") + ("; " + note if note else "")
        if tcell.get("stacks", {}).get(st, {}).get("reference_mismatch") and r not in STOCK_ROWS:
            reason += "; parity owed (reference window rule differs)"
        return Selection(r, v, word, key, measured, st, st_measured, _measured_class(tcell, a, st, cls), exact_vs, bwd, cap, x, fb, reason, config)
    cls, exact_vs, bwd, cap, fb = _row_facts(stock_row)
    return Selection(stock_row, DEFAULT_VARIANT.get(stock_row), word, key, measured, st, st_measured, cls, exact_vs, bwd, cap, None, fb,
                     "no admitted row; named stock row; " + why, config)


def describe(sel):
    """One line for a kit's LEVER / census line."""
    return "apb row=%s%s word=%s cell=%s%s stack=%s%s class=%s%s x_stock=%s%s reason=%s" % (
        sel.row, (":" + sel.variant) if sel.variant else "", sel.word, sel.cell, "" if sel.size_measured else "(nearest)", sel.stack,
        "" if sel.stack_measured else "(ref)", sel.cls, (" exact_vs=%s" % sel.exact_vs) if sel.exact_vs else "", sel.x_stock,
        "" if sel.capture_safe else " CAPTURE-UNSAFE", sel.reason)


def coverage(cc, shapes, *, word="fast", stack=None, capture=False):
    """Census of a kit's shapes against the table: shapes = [(dtype, cell, n_tokens[, samples]), ...] -> [{shape, row, cell, measured, x_stock, reason}]."""
    out = []
    for s in shapes:
        dtype, cell, N = s[:3]
        S = s[3] if len(s) > 3 else 1
        try:
            sel = select(cc, dtype, cell, N, word=word, samples=S, stack=stack, capture=capture)
            out.append({"shape": s, "row": arm_word(sel.row, sel.variant), "cell": sel.cell, "measured": sel.size_measured, "x_stock": sel.x_stock, "reason": sel.reason})
        except Refusal as e:
            out.append({"shape": s, "row": None, "cell": None, "measured": False, "x_stock": None, "reason": "refused:%s->%s" % (e.kind, e.fallback)})
    return out


def rows_for_kit(carried):
    """The subset of ROW_NAMES a kit can bind given the rows it carries/routes (stock rows always)."""
    return tuple(r for r in ROW_NAMES if r in carried or r in STOCK_ROWS)


def capture_unsafe(row):
    """True when the table flags ``row`` capture-unsafe (a graphed region must never select it); the evidence string is rows[row]['capture_evidence']."""
    r, _v = split_word(row)
    return not bool(table()["rows"].get(r, {}).get("capture_safe", True))


# ------------------------------------------------------------------------------------------------------------------------------ serving
def _imp(name, refusal_row, fallback):
    import importlib
    try:
        return importlib.import_module(name)
    except ImportError as e:
        raise Refusal("import:%s(%s)" % (name.split(".")[-1] if name.startswith("opt_core") else name, str(e).split("\n")[0][:60].replace(" ", "_")),
                      refusal_row, fallback)
    except (RuntimeError, OSError) as e:                                        # a carried module whose import probes the device / a binary
        raise Refusal("import:%s(%s:%s)" % (name.split(".")[-1], type(e).__name__, str(e).split("\n")[0][:60].replace(" ", "_")), refusal_row, fallback)


def carried_module(row):
    """The module object serving a carried row (imports torch/triton): fpf_apb -> kernels.apb.fpf_apb.apb_triton, fpf_atom -> .atom_triton,
    fpf_pf_bias -> .pf_triton, dit_exact -> kernels.apb.dit_exact, l3a -> kernels.apb.l3a.fab_batched, ef2_pairbias -> kernels.apb.ef2.ef2_pairbias_attn."""
    name = {"fpf_apb": "opt_core.kernels.apb.fpf_apb.apb_triton", "fpf_atom": "opt_core.kernels.apb.fpf_apb.atom_triton",
            "fpf_pf_bias": "opt_core.kernels.apb.fpf_apb.pf_triton", "dit_exact": "opt_core.kernels.apb.dit_exact",
            "l3a": "opt_core.kernels.apb.l3a.fab_batched", "ef2_pairbias": "opt_core.kernels.apb.ef2.ef2_pairbias_attn",
            "apb_attn": "opt_core.kernels.apb_attn", "dtk_loop": "opt_core.kernels.dtk_kernels", "dtk_window": "opt_core.kernels.dtk_kernels",
            "sba": "opt_core.attn.shared_bias_attn", "ln_proj": "opt_core.kernels.ln_proj", "lnl_ln_linear": "opt_core.kernels.lnl_fused",
            "sdpa": "opt_core.attn.sdpa_bias", "dit_fast": "opt_core.kernels.apb.ditfast.kernels", "dit_fast:atom": "opt_core.kernels.apb.ditfast.atom_kernels",
            "atom_exact": "opt_core.kernels.apb.atom_exact.kernel"}.get(row)
    if name is None:
        raise Refusal("no_module(%s)" % row, row, _row_facts(row)[4] if row in ROW_NAMES else None)
    row = row.split(":")[0]
    return _imp(name, row, (_row_facts(row)[4] or "").split("|")[0] or None)


def dit_exact_abi():
    """'torch<ver>-cu<cuda>-sm<cc>' of this process (the carried package's prebuilt key), with '-cp<major><minor>' appended when the running
    interpreter is not the one that key's binary was built for (a cpython extension module: the full ABI is torch, CUDA, arch AND python);
    None without CUDA torch."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    import sys
    cc = torch.cuda.get_device_capability()
    key = "torch%s-cu%s-sm%d%d" % (torch.__version__.split("+")[0], (torch.version.cuda or "none").replace(".", ""), cc[0], cc[1])
    py = DIT_EXACT_PYTHON.get(key)
    if py is not None and tuple(sys.version_info[:2]) != tuple(py):
        return key + "-cp%d%d" % tuple(sys.version_info[:2])
    return key


def stack_word(device=None):
    """'<card>:torch<torch>/<triton>/<cueq version | nocueq>' of this process (the table's stack grammar)."""
    import torch
    try:
        import triton
        tv = triton.__version__
    except ImportError:
        tv = "none"
    try:
        import cuequivariance_torch as cqt
        cq = "cueq" + str(getattr(cqt, "__version__", "?"))
    except ImportError:
        cq = "nocueq"
    card = "?"
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(device if device is not None else torch.cuda.current_device())
        card = "H100" if "H100" in name else ("A100_40" if ("A100" in name and "40GB" in name) else ("A100" if "A100" in name else name.split(" ")[-1]))
    return "%s:torch%s/%s/%s" % (card, torch.__version__, tv, cq)


def _canon_core(q, k, v, layout):
    """Views only: 'snhd' q/k/v [S,N,H,D] and 'shnd' [S,H,N,D] both returned as (snhd, shnd) pairs."""
    if layout == "snhd":
        return (q, k, v), (q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3))
    if layout == "shnd":
        return (q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)), (q, k, v)
    raise ValueError("layout must be 'snhd' or 'shnd'")


def fold_mask(bias, key_mask, inf=1e9):
    """bias [1|S,H,N,N] + (key_mask [1|S,N] == 0) * -inf  ->  the additive bias the mask-free rows take (fp32)."""
    import torch
    b = bias.float()
    if key_mask is None:
        return b
    m = (1.0 - key_mask.to(torch.float32)) * (-float(inf))
    return b + m[:, None, None, :]


def _stack_for_exact(word, device=None):
    """The running stack word when the EXACT word is asked (its vouch is per stack), else None (tolerance-class words need no stack)."""
    return stack_word(device) if isinstance(word, str) and split_word(word)[0] == "exact" else None


_RELAID = {}                                                       # (row, what) -> count: caller views re-laid contiguous before a row kernel's launch (this process)
_LAST = {}                                                         # the row the current serving call dispatches to (read by the caller-layout guard)


def relaid():
    """{(row, what): n} -- strided caller views this face re-laid before a launch (token relaid:<what> on the Selection's reason)."""
    return dict(_RELAID)


def _relay(t, what, row):
    """A caller's view a row kernel cannot address (the kernels take unit stride on the LAST dim of q / k / v / gate / bias / z): re-laid
    contiguous before the launch -- values identical, transient = numel x element size bytes (a [H, N, N] bf16 plane set: H*N*N*2 B) --
    instead of an assertion inside the carried module.  Chosen for every word (fast / exact-class / big): no row in the table addresses a
    key-strided plane without a copy of its own, and the statement rows materialise the mask as well.  Token ``relaid:<what>`` lands on the
    Selection's reason (census) and one stderr note per (row, what) per process names the bytes."""
    if t is None or not hasattr(t, "stride") or t.dim() == 0 or t.stride(-1) == 1:
        return t, False
    tc = t.contiguous()
    key = (row, what)
    _RELAID[key] = _RELAID.get(key, 0) + 1
    if _RELAID[key] == 1:
        import sys
        sys.stderr.write("[opt_core.kernels.apb] relaid:%s row=%s stride%s -> contiguous (%d B transient; values identical)\n"
                         % (what, row, tuple(t.stride()), tc.numel() * tc.element_size()))
    return tc, True


def _caller_layout_refusals(fn):
    """Never an assertion on a caller's layout / shape / dtype out of a serving call: an ``AssertionError`` raised under the dispatch (the
    carried modules assert their operand contracts) surfaces as this face's ``Refusal('layout:<the assertion>', row, <the row's fallback>)``
    so the binder's next row / the statement serves by name."""
    import functools

    @functools.wraps(fn)
    def guarded(*a, **k):
        try:
            return fn(*a, **k)
        except AssertionError as e:
            sel = k.get("selection")
            row = getattr(sel, "row", None) or _LAST.get(fn.__name__) or "?"
            try:
                fb = _row_facts(row)[4]
            except (KeyError, TypeError):                                 # an unknown row word: the boundary's statement
                fb = "sdpa"
            msg = (str(e).strip() or "assert").replace("\n", " ")[:70].replace(" ", "_")
            raise Refusal("layout:%s" % msg, row, fb)
    return guarded


def pair_bias_attention(q, k, v, bias, key_mask=None, gate=None, *, word=None, selection=None, scale=None, layout="snhd", cell="dit_h16d48", prefer=None,
                        capture=False, out_dtype=None, config=None, cache=None):
    """Serve the core op through the selected row.  q/k/v: [S,N,H,D] ('snhd') or [S,H,N,D] ('shnd'); bias: [1,H,N,N] (shared) or [S,H,N,N];
    key_mask: [1|S, N] (1 = keep) or None; gate: like q or None.  ``word`` (row[:variant] or tier word) or a ``selection`` is required.
    Returns (out in q's layout and dtype (or out_dtype), Selection).  Rows refuse by ``Refusal`` (kind, row, fallback) -- nothing substituted:
    the fused shared-bias rows (``fpf_apb``, ``l3a``) refuse a bias plane set or a key mask given PER SAMPLE (S > 1) by name
    (``bias_per_sample`` / ``key_mask_per_sample`` -> ``dtk_loop``, which serves sample by sample), before any kernel is imported."""
    global _WARMED
    if not _WARMED:                                             # first serving call of this face in the process: the stack's heavy libraries imported early + shallow (opt_core.warm)
        _WARMED = True
        from opt_core.warm import auto_warm
        auto_warm("apb")
    import math
    import torch
    (qs, ks, vs), (qh, kh, vh) = _canon_core(q, k, v, layout)
    S, N, H, D = qs.shape
    cc = torch.cuda.get_device_capability(q.device) if q.is_cuda else (0, 0)
    scale = (1.0 / math.sqrt(D)) if scale is None else float(scale)
    if selection is None:
        selection = select(cc, q.dtype, cell, N, word=word, samples=S, prefer=prefer, capture=capture, head_dim=D, heads=H, stack=_stack_for_exact(word, q.device),
                           abi=dit_exact_abi() if split_word(word)[0] in ("dit_exact", "exact", "faithful") else None, config=config)
        if (selection.reason or "").startswith(INHERIT_TOKEN) and selection.row not in STOCK_ROWS:
            # an INHERITED portable row on a part the table never measured: its first launch here is guarded -- a build / compile / launch error
            # (anything but out-of-memory) retires the arm for this part (census once) and the NEXT portable row of the donor cell serves, the
            # statement last; never a raw exception out of an inherited row
            from opt_core.oom import is_oom
            for _ in range(len(ROW_NAMES) + 2):
                try:
                    return pair_bias_attention(q, k, v, bias, key_mask, gate, word=word, selection=selection, scale=scale, layout=layout, cell=cell, prefer=prefer,
                                               capture=capture, out_dtype=out_dtype, config=config, cache=cache)
                except Refusal:
                    raise
                except Exception as e:                                        # noqa: BLE001
                    if is_oom(e):
                        raise
                    mark_dead(cc, arm_word(selection.row, selection.variant), e, cell=cell, dtype=q.dtype, n_tokens=N,
                              kw=dict(word=word, samples=S, timing="graph" if capture else "eager", stack=None, heads=H, head_dim=D))
                    selection = select(cc, q.dtype, cell, N, word=word, samples=S, prefer=prefer, capture=capture, head_dim=D, heads=H,
                                       stack=_stack_for_exact(word, q.device), config=config)
                    if selection.row in STOCK_ROWS or not (selection.reason or "").startswith(INHERIT_TOKEN):
                        return pair_bias_attention(q, k, v, bias, key_mask, gate, word=word, selection=selection, scale=scale, layout=layout, cell=cell, prefer=prefer,
                                                   capture=capture, out_dtype=out_dtype, config=config, cache=cache)
    row, variant = selection.row, selection.variant
    _LAST["pair_bias_attention"] = row
    if row not in STOCK_ROWS:                                                    # the row kernels take unit stride on the last dim: a caller's permuted view is re-laid, never asserted on
        rl = set()
        bias, r_ = _relay(bias, "bias_key_stride", row); r_ and rl.add("bias_key_stride")
        q, r_ = _relay(q, "qkv_last_stride", row); r_ and rl.add("qkv_last_stride")
        k, r_ = _relay(k, "qkv_last_stride", row); r_ and rl.add("qkv_last_stride")
        v, r_ = _relay(v, "qkv_last_stride", row); r_ and rl.add("qkv_last_stride")
        if gate is not None:
            gate, r_ = _relay(gate, "gate_last_stride", row); r_ and rl.add("gate_last_stride")
        if rl:
            (qs, ks, vs), (qh, kh, vh) = _canon_core(q, k, v, layout)
            selection = selection._replace(reason=((selection.reason or "") + "; relaid:" + ",".join(sorted(rl)))[:240])
    if row in INT32_OFFSET_ROWS:                                                 # a caller-built Selection meets the same bound at launch
        admits(row, cc, q.dtype, samples=S, n_tokens=N, heads=H, head_dim=D)
    if cache is not None:
        cache["_last"] = selection
    od = out_dtype or q.dtype
    if row == "apb_attn":
        core = _imp("opt_core.attn.apb_core", row, "sdpa")
        try:
            o = core.pair_bias_attention(qh, kh, vh, bias, key_mask,                    # [S, H, N, D] views in place (layout 'shnd': any strides, unit D stride)
                                         gate=None if gate is None else _canon_core(gate, gate, gate, layout)[0][0].reshape(S, N, H * D),
                                         num_samples=S, num_heads=H, layout="shnd", scale=scale, config=config)
        except core.Unsupported as e:
            raise Refusal("%s" % getattr(e, "event", str(e)), row, "sdpa")
        o = o.reshape(S, N, H, D)
        return (o if layout == "snhd" else o.permute(0, 2, 1, 3)).to(od), selection
    if row == "fpf_apb":
        if bias.dim() == 4 and bias.shape[0] != 1:                                # one plane set PER SAMPLE ([S, H, N, N], S > 1): this row takes ONE shared set
            raise Refusal("bias_per_sample (fpf_apb takes one shared bias plane set for the S samples)", row, "dtk_loop")
        if key_mask is not None and S > 1 and key_mask.reshape(-1, N).shape[0] != 1:   # a key mask row PER SAMPLE folds into per-sample planes: the same form, named
            raise Refusal("key_mask_per_sample (fpf_apb folds one shared key mask into the shared planes)", row, "dtk_loop")
        PA = carried_module("fpf_apb")
        b = fold_mask(bias, key_mask) if key_mask is not None else bias
        b3 = b[0] if b.dim() == 4 and b.shape[0] == 1 else b
        o = PA.apb_views(qs, ks, vs, b3, None if gate is None else _canon_core(gate, gate, gate, layout)[0][0], scale=scale, out_dtype=od,
                         opd=("fp16" if variant == "fp16" else None), cfg=(config or None))
        return (o if layout == "snhd" else o.permute(0, 2, 1, 3)), selection
    if row == "dit_exact":
        DX = carried_module("dit_exact")
        try:
            mod = DX.load_prebuilt()
        except RuntimeError as e:
            raise Refusal(str(e).replace(" ", "_")[:80], row, "sdpa_upcast")
        b = fold_mask(bias, key_mask) if key_mask is not None else bias.float()
        q32 = (qh.float() * scale)
        k32, v32 = kh.float(), vh.float()
        why = DX._route(q32, k32, v32, b)
        if why is not None:
            raise Refusal("route:%s" % why, row, "sdpa_upcast")
        o = mod.forward(q32, k32, v32, b, 1)
        if gate is not None:
            o = o * torch.sigmoid(_canon_core(gate, gate, gate, layout)[1][0].float())
        return (o.permute(0, 2, 1, 3) if layout == "snhd" else o).to(od), selection
    if row in ("l3a", "dtk_loop"):
        b3 = bias[0] if (bias.dim() == 4 and bias.shape[0] == 1) else bias
        km = None if key_mask is None else key_mask.reshape(-1, N)
        g = None if gate is None else _canon_core(gate, gate, gate, layout)[0][0].reshape(S, N, H * D)     # [S, N, H*D] pre-sigmoid (the kernels' gate layout)
        if row == "l3a":
            L3 = carried_module("l3a")
            if b3.dim() != 3:
                raise Refusal("bias_per_sample (l3a takes one shared bias)", row, "dtk_loop")
            try:
                o = L3.flash_bias_attn_batched(qh, kh, vh, b3, None if km is None else (km[0] if km.shape[0] == 1 else km), g, scale=scale, out_dtype=od,
                                               stage_c=(variant == "c"))
            except L3.KoptAttnUnavailable as e:
                raise Refusal("triton_missing(%s)" % str(e)[:40].replace(" ", "_"), row, "dtk_loop")
            except L3.KoptAttnRefused as e:
                raise Refusal("l3a:%s" % str(e)[:60].replace(" ", "_"), row, "dtk_loop")
        else:
            K = carried_module("dtk_loop")
            outs = []
            for s in range(S):
                bs = b3 if b3.dim() == 3 else b3[s]
                outs.append(K.flash_bias_attn(qh[s], kh[s], vh[s], bs, None if km is None else km[min(s, km.shape[0] - 1)], None if g is None else g[s],
                                              out_dtype=od, scale=scale))
            o = torch.stack(outs, 0)
        o = o.reshape(S, N, H, D)                                               # the kernels write [S, N, H*D] rows
        return (o if layout == "snhd" else o.permute(0, 2, 1, 3)), selection
    if row == "sba":
        SBA = carried_module("sba")
        b = fold_mask(bias, key_mask) if key_mask is not None else bias
        try:
            o, _ev = SBA.attention(qh, kh, vh, b, scale=scale, input_precision=(variant or "tf32"), pad_head_dim=True)
        except Exception as e:                                                  # the module's own Refused type (named); re-raise as this face's word
            from ...oom import is_oom
            if is_oom(e): raise
            if type(e).__name__ in ("Refused", "Refusal", "Unsupported"):
                raise Refusal("sba:%s" % getattr(e, "event", str(e))[:60], row, "sdpa")
            raise
        if o.dim() == 5:                                                        # [G=1, S, H, N, D] -> [S, H, N, D]
            o = o[0]
        if gate is not None:
            o = o * torch.sigmoid(_canon_core(gate, gate, gate, layout)[1][0].to(o.dtype))
        return (o.permute(0, 2, 1, 3) if layout == "snhd" else o).to(od), selection
    if row == "ef2_pairbias":
        E = carried_module("ef2_pairbias")
        b = bias if bias.dim() == 4 else bias[None]
        km = None if key_mask is None else (key_mask.reshape(-1, N).expand(S, N) if key_mask.reshape(-1, N).shape[0] == 1 else key_mask.reshape(S, N))
        o = E.fused_attention_core(qs, ks, vs, b.expand(S, -1, -1, -1) if b.shape[0] == 1 and S > 1 else b,
                                   None if gate is None else _canon_core(gate, gate, gate, layout)[0][0], km, scale)
        return (o if layout == "snhd" else o.permute(0, 2, 1, 3)).to(od), selection
    if row in ("sdpa", "sdpa_upcast"):
        import torch.nn.functional as F
        b = fold_mask(bias, key_mask) if key_mask is not None else bias
        if row == "sdpa_upcast":
            o = F.scaled_dot_product_attention(qh.float() * scale, kh.float(), vh.float(), attn_mask=b.float(), scale=1.0)
        else:
            SB = _imp("opt_core.attn.sdpa_bias", row, None)
            backend = variant or "auto"
            try:
                o, _ev = SB.sdpa_bias(qh, kh, vh, b, None, layout="bhld", bias_layout="bhqk", scale=scale, backend=backend, out_dtype=od)
            except SB.Refused as e:
                raise Refusal("sdpa:%s" % getattr(e, "event", str(e))[:60], row, ("sdpa" if backend != "auto" else "sdpa:math"))
        if gate is not None:
            o = o * torch.sigmoid(_canon_core(gate, gate, gate, layout)[1][0].to(o.dtype))
        return (o.permute(0, 2, 1, 3) if layout == "snhd" else o).to(od), selection
    if row == "naive":
        o = reference(q, k, v, bias, key_mask, gate, scale=scale, layout=layout, dtype=torch.float32)
        return o.to(od), selection
    if row == "ds4sci":
        DS = None
        for modname in ("deepspeed.ops.deepspeed4science", "openfold3.core.kernels.cuda.deepspeed4science"):
            try:
                import importlib
                DS = getattr(importlib.import_module(modname), "DS4Sci_EvoformerAttention")
                break
            except (ImportError, AttributeError):
                continue
        if DS is None:
            raise Refusal("import:deepspeed4science", row, "sdpa")
        if torch.cuda.is_current_stream_capturing():
            raise Refusal("capture_unsafe", row, "sdpa")
        mb = torch.zeros(S, 1, 1, 1, N, dtype=qs.dtype, device=qs.device) if key_mask is None else ((1.0 - key_mask.reshape(-1, N).to(qs.dtype)) * -1e9).reshape(-1, 1, 1, 1, N).expand(S, 1, 1, 1, N)
        pb = bias.to(qs.dtype).reshape(-1, 1, H, N, N).expand(S, 1, H, N, N)
        try:
            o = DS(qs.reshape(S, 1, N, H, D), ks.reshape(S, 1, N, H, D), vs.reshape(S, 1, N, H, D), [mb, pb]).reshape(S, N, H, D)
        except RuntimeError as e:                                                # the extension's JIT build refuses this stack (no CUTLASS checkout / arch): named
            if "JIT" in str(e) or "CUTLASS" in str(e) or "compatible" in str(e):
                raise Refusal("import:deepspeed4science_build(%s)" % str(e)[:60].replace(" ", "_"), row, "sdpa")
            raise
        if gate is not None:
            o = o * torch.sigmoid(_canon_core(gate, gate, gate, layout)[0][0].to(o.dtype))
        return (o if layout == "snhd" else o.permute(0, 2, 1, 3)).to(od), selection
    if BOUNDARY.get(row) == "module":
        raise Refusal("module_boundary:%s (compose: pair_bias_planes(z, word='ln_proj') once per trunk state + pair_bias_attention(word='apb_attn') per call; "
                      "or the library's attention_pair_bias)" % row, row, None)
    raise Refusal("no_core_serving_branch:%s" % row, row, _row_facts(row)[4])


def atom_attention(q, k, v, bias, gate=None, *, word=None, selection=None, n_queries=32, n_keys=128, scale=None, prefer=None, capture=False, out_dtype=None):
    """Serve the WINDOWED op (atom transformer local attention): q/k/v [S, N_atom, H, D]; bias [H, n_windows, n_queries, n_keys] (the windowed pair
    bias, -inf outside the sequence).  Rows: fpf_atom | dtk_window | sdpa_gather.  Returns (out [S, N_atom, H, D], Selection)."""
    import math
    import torch
    S, NA, H, D = q.shape
    cc = torch.cuda.get_device_capability(q.device) if q.is_cuda else (0, 0)
    scale = (1.0 / math.sqrt(D)) if scale is None else float(scale)
    if selection is None:
        selection = select(cc, q.dtype, "atom_h4d32w32x128", max(1, NA // 8), word=word, samples=S, prefer=prefer, capture=capture, head_dim=D, heads=H,
                           stack=_stack_for_exact(word, q.device))
    row = selection.row
    _LAST["atom_attention"] = row
    if row not in STOCK_ROWS:                                                    # unit stride on the last dim for the window kernels: re-laid, never asserted on
        rl = set()
        bias, r_ = _relay(bias, "bias_key_stride", row); r_ and rl.add("bias_key_stride")
        q, r_ = _relay(q, "qkv_last_stride", row); r_ and rl.add("qkv_last_stride")
        k, r_ = _relay(k, "qkv_last_stride", row); r_ and rl.add("qkv_last_stride")
        v, r_ = _relay(v, "qkv_last_stride", row); r_ and rl.add("qkv_last_stride")
        if gate is not None:
            gate, r_ = _relay(gate, "gate_last_stride", row); r_ and rl.add("gate_last_stride")
        if rl:
            selection = selection._replace(reason=((selection.reason or "") + "; relaid:" + ",".join(sorted(rl)))[:240])
    if row in INT32_OFFSET_ROWS:
        admits(row, cc, q.dtype, samples=S, n_atoms=NA, heads=H, head_dim=D)
    od = out_dtype or q.dtype
    if row == "fpf_atom":
        AT = carried_module("fpf_atom")
        o = AT.atom_apb(q, k, v, bias, gate, n_queries=n_queries, n_keys=n_keys, scale=scale, out_dtype=od)
        return o, selection
    if row == "dtk_window":
        K = carried_module("dtk_window")
        if not hasattr(K, "window_attn"):
            raise Refusal("no_window_attn", row, "sdpa_gather")
        outs = [K.window_attn(q[s].permute(1, 0, 2), k[s].permute(1, 0, 2), v[s].permute(1, 0, 2), bias, None, None if gate is None else gate[s].reshape(NA, H * D),
                              scale=scale, NQ=n_queries, NK=n_keys).reshape(NA, H, D) for s in range(S)]
        return torch.stack(outs, 0).to(od), selection
    if row == "sdpa_gather":
        return atom_reference(q, k, v, bias, gate, n_queries=n_queries, n_keys=n_keys, scale=scale, dtype=q.dtype).to(od), selection
    if row == "atom_exact":                                                      # the exact row runs in the trunk's [S, H, NA, D] layout with the scale folded into q
        if int(n_queries) != 32 or int(n_keys) != 128:
            raise Refusal("window:%sx%s!=32x128" % (n_queries, n_keys), row, "sdpa_gather")
        qh = (q * scale).permute(0, 2, 1, 3).contiguous(); kh = k.permute(0, 2, 1, 3).contiguous(); vh = v.permute(0, 2, 1, 3).contiguous()
        o = atom_exact_attention(qh, kh, vh, bias.reshape(1, *bias.shape).float().contiguous(), chunk=256)      # [S, H, NA, D] view of [S, NA, H, D]
        o = o.transpose(-2, -3)
        if gate is not None:
            o = o * torch.sigmoid(gate.reshape(S, NA, H, D).to(o.dtype)) if gate.dtype != torch.bool else o
        return o.to(od), selection
    raise Refusal("no_windowed_serving_branch:%s" % row, row, "sdpa_gather")


_ATOM_EXACT = {"routes": None, "loadcheck": None, "seconds": None, "refused": None}


def atom_exact_ready(device=None):
    """Per process, before the first launch of row ``atom_exact``: determine which cuBLAS numerics the statement's chunked GEMMs use here
    (the carried kernel's route table; ~1 s, torch.matmul on the exact shapes) and check the carried load-check vectors' output digests.
    Returns the carried kernel module; raises Refusal by name (routes_unreproducible | loadcheck:<case>) with fallback sdpa_gather."""
    import hashlib, json as _json, os, time, torch
    K = carried_module("atom_exact")
    if _ATOM_EXACT["refused"]:
        raise Refusal(_ATOM_EXACT["refused"], "atom_exact", "sdpa_gather")
    if _ATOM_EXACT["routes"] is not None:
        return K
    t0 = time.time()
    dev = device if device is not None else torch.device("cuda", torch.cuda.current_device())
    bad = K.determine_routes(dev)                                              # under the process's matmul precision state (the trunks run TF32 matmuls at inference)
    if bad:                                                                     # a reachable chunk batch count whose cuBLAS numerics no variant reproduces: named, never guessed
        _ATOM_EXACT["refused"] = "routes_unreproducible(tf32=%s;%s)" % (bool(torch.backends.cuda.matmul.allow_tf32), K.route_summary().replace(" ", ""))
        raise Refusal(_ATOM_EXACT["refused"], "atom_exact", "sdpa_gather")
    spec = _json.load(open(os.path.join(os.path.dirname(os.path.abspath(K.__file__)), "vectors.json")))
    def _original(*a, **kw):                                                    # the load-check vectors sit inside the kernel's envelope: never reached
        raise Refusal("loadcheck:envelope_miss", "atom_exact", "sdpa_gather")
    wrapped = K.make_local_attention(_original)
    bad = []
    for case in spec["cases"]:                                                  # the producing kit's vectors: [S, NA, H*32] linears (scales .35, 1, 1) viewed [S, H, NA, 32]; bias [1, H, nb, 32, 128] x 2
        g = torch.Generator(device="cpu").manual_seed(int(case["seed"]))
        S_, H_, NA_ = int(case["S"]), int(case["H"]), int(case["NA"])
        nb = (NA_ + K.NQ - 1) // K.NQ
        lin = [torch.randn(S_, NA_, H_ * K.D, generator=g) * sc for sc in (0.35, 1.0, 1.0)]
        q, k, v = [t.to(dev).view(S_, NA_, H_, K.D).transpose(1, 2) for t in lin]
        b = (torch.randn(1, H_, nb, K.NQ, K.NK, generator=g) * 2.0).to(dev)
        o = wrapped(q, k, v, K.NQ, K.NK, attn_bias=None, trunked_attn_bias=b, inf=1e10, use_efficient_implementation=True, inplace_safe=False, chunk_size=256)
        torch.cuda.synchronize()
        dg = hashlib.sha256(o.detach().contiguous().cpu().numpy().tobytes()).hexdigest()
        if dg != case["sha256"] or list(o.shape) != list(case.get("shape", o.shape)):
            bad.append(str(case.get("name", case["seed"])))
    _ATOM_EXACT["loadcheck"] = {"n": len(spec["cases"]), "bad": bad}
    if bad:
        _ATOM_EXACT["refused"] = "loadcheck:%s" % "+".join(bad)
        raise Refusal(_ATOM_EXACT["refused"], "atom_exact", "sdpa_gather")
    _ATOM_EXACT["routes"] = K.route_summary(); _ATOM_EXACT["seconds"] = round(time.time() - t0, 2)
    K.STATE["calls"] = {"kernel": 0, "original": 0}
    return K


def atom_exact_attention(q, k, v, bias_comb, chunk=256):
    """Row ``atom_exact`` in the trunk's own layout: q, k, v [S.., H, NA, 32] fp32 (q already scaled as the statement scales it), bias_comb
    [Sb, H, n_windows, 32, 128] fp32 (pad mask + pair bias combined, Sb in {1, S}); returns [.., H, NA, 32] (a transpose view of a contiguous
    [.., NA, H, 32] buffer) bitwise equal to the windowed statement (dense-trunk rearrangement + chunked math-SDPA) where the process's cuBLAS
    numerics are reproducible; Refusal by name otherwise (fallback sdpa_gather = the statement)."""
    K = atom_exact_ready(q.device)
    try:
        return K.fused_local_attention(q, k, v, bias_comb, chunk)
    except K.Refused as e:
        raise Refusal("atom_exact:%s" % str(e)[:90].replace(" ", "_"), "atom_exact", "sdpa_gather")


# ---------------------------------------------------------------------------------------------------------------- weight-pack cache (opt-in)
PACK_CACHE_MAX = 256                                                            # entries per caller-supplied cache dict (oldest dropped beyond)


def _tensor_identity(t):
    """(id, data_ptr, version, shape, dtype, device) of a tensor -- the identity AND content-version a packed copy was made from; None -> None.
    Inference tensors carry no version counter (-1): an in-place update of such a weight cannot be seen (re-create the cache then)."""
    if t is None:
        return None
    try:
        ver = -1 if t.is_inference() else t._version
    except (AttributeError, RuntimeError):
        ver = -1
    return (id(t), t.data_ptr(), ver, tuple(t.shape), t.dtype, str(t.device))


def pack_cached(cache, tag, tensors, extra, build):
    """The packed operand for (tag, *tensors, extra) from the caller's ``cache`` dict, built by ``build()`` on a miss.  The key is the
    identity + in-place version of EVERY source tensor (never the shape alone) plus ``extra`` (eps, device, widths); the entry keeps
    references to its source tensors so their ids cannot be recycled while it lives; a source updated in place (version bump), re-allocated
    or replaced is a miss and re-packs.  ``cache is None``: build() every call (no caching)."""
    if cache is None:
        return build()
    key = (tag,) + tuple(_tensor_identity(t) for t in tensors) + (extra,)
    ent = cache.get(key)
    if ent is not None and all(a is b for a, b in zip(ent[1], tensors)):
        return ent[0]
    packed = build()
    if len(cache) >= PACK_CACHE_MAX:
        for k in list(cache)[: len(cache) - PACK_CACHE_MAX + 1]:
            cache.pop(k, None)
    cache[key] = (packed, tuple(tensors))
    return packed


def pair_bias_planes(z, ln_weight, ln_bias, weight, *, word=None, selection=None, eps=1e-5, out_layout="hij", out_dtype=None, cache=None, prefer=None):
    """Serve the PRODUCER: bias = Linear_{c_z->H}(LayerNorm(z)) over the pair rows, head-major.  z [B?, I, J, c_z]; weight [H, c_z].
    Rows: ln_proj (planes [B?,H,I,ld][..., :J], ld = J rounded to 8) | fpf_pf_bias ([H,I,J] bf16) | lnl_ln_linear (permute view) | torch_ln_linear.
    ``cache`` (dict) holds the packed weights across calls.  Returns (bias planes, Selection)."""
    global _WARMED
    if not _WARMED:                                             # first serving call of this face in the process: the stack's heavy libraries imported early + shallow (opt_core.warm)
        _WARMED = True
        from opt_core.warm import auto_warm
        auto_warm("apb")
    import torch
    import torch.nn.functional as F
    c = z.shape[-1]
    H = weight.shape[0]
    I = z.shape[-3]
    cc = torch.cuda.get_device_capability(z.device) if z.is_cuda else (0, 0)
    if selection is None:
        selection = select(cc, z.dtype, cell_word("bias", heads=H, c_z=c), I, word=word, prefer=prefer, c_z=c, heads=H, stack=_stack_for_exact(word, z.device))
    row = selection.row
    _LAST["pair_bias_planes"] = row
    if row not in STOCK_ROWS:                                                    # the producers read z rows with unit stride on the channel dim
        z, r_ = _relay(z, "z_last_stride", row)
        if r_:
            selection = selection._replace(reason=((selection.reason or "") + "; relaid:z_last_stride")[:240])
    if row == "ln_proj":
        LP = carried_module("ln_proj")
        try:                                                                     # packed = gamma/beta folded into the projection: keyed on the identity + version
            packed = pack_cached(cache, "ln_proj_packed", (weight, ln_weight, ln_bias), (float(eps), str(z.device)),   # of ALL three tensors (+ eps, device)
                                 lambda: LP.pack_pair_bias_weights(ln_weight, ln_bias, weight, eps, z.device))
        except LP.Unsupported as e:
            raise Refusal("ln_proj:%s" % getattr(e, "reason", str(e))[:60].replace(" ", "_"), row, "torch_ln_linear")
        try:
            out = LP.pair_bias(z, packed, out_layout=("bhij" if out_layout == "hij" else "bijh"), out_dtype=out_dtype or torch.float32)
        except LP.Unsupported as e:
            raise Refusal("ln_proj:%s" % getattr(e, "reason", str(e))[:60].replace(" ", "_"), row, "torch_ln_linear")
        return out, selection
    if row == "fpf_pf_bias":
        PF = carried_module("fpf_pf_bias")
        if z.dim() == 3:
            out = PF.pf_bias(z, ln_weight, ln_bias, weight, eps, out_dtype=out_dtype or torch.bfloat16)
        else:                                                                    # leading batch dims: one pair representation per launch, stacked
            zs = z.reshape((-1,) + tuple(z.shape[-3:]))
            out = torch.stack([PF.pf_bias(zs[i_], ln_weight, ln_bias, weight, eps, out_dtype=out_dtype or torch.bfloat16) for i_ in range(zs.shape[0])]).reshape(tuple(z.shape[:-3]) + (H, I, I))
        return out, selection
    if row == "lnl_ln_linear":
        LNL = carried_module("lnl_ln_linear")
        nout = 16
        while nout < H:
            nout *= 2
        def _pad():
            Wp_ = torch.zeros(nout, c, dtype=torch.bfloat16, device=z.device)
            Wp_[:H] = weight.to(torch.bfloat16)
            return Wp_
        Wp = pack_cached(cache, "lnl_w", (weight,), (nout, str(z.device)), _pad)   # the bf16 padded copy: keyed on the weight's identity + version
        zz = z if z.dim() in (3, 4) else z.reshape((-1,) + tuple(z.shape[-3:]))
        try:
            _y16, y = LNL.ln_linear(zz.contiguous(), ln_weight, ln_bias, Wp, eps=eps)
        except Exception as e:
            from ...oom import is_oom
            if is_oom(e): raise
            if type(e).__name__ in ("Unsupported", "BuildFailed", "Refused"):
                raise Refusal("lnl:%s" % str(e)[:60].replace(" ", "_"), row, "torch_ln_linear")
            raise
        y = y[..., :H]
        out = y.permute(*(list(range(y.dim() - 3)) + [y.dim() - 1, y.dim() - 3, y.dim() - 2])) if out_layout == "hij" else y
        return (out if out_dtype is None else out.to(out_dtype)), selection
    if row == "torch_ln_linear":
        y = F.linear(F.layer_norm(z, (c,), ln_weight.to(z.dtype), ln_bias.to(z.dtype), eps), weight.to(z.dtype))
        out = y.permute(*(list(range(y.dim() - 3)) + [y.dim() - 1, y.dim() - 3, y.dim() - 2])).contiguous() if out_layout == "hij" else y
        return out.to(out_dtype or torch.float32), selection
    raise Refusal("no_producer_serving_branch:%s" % row, row, "torch_ln_linear")


class _TorchDitRows(object):
    """The statements the fused DiT block's row kernels stand for (row ``torch_dit_rows``): the same function names and operand conventions as the
    carried modules (conditioning operands [Ns, C] read with row-modulo: activation row r takes conditioning row r % Ns), fp32 arithmetic (fp64
    in a subclass with WIDE = True: the reference), the result in ``out_dtype``.  Token stream (kernels.py): adaln, gate, swiglu, resgate,
    resgate_adaln; atom stream (atom_kernels.py): adaln2, resgate_adaln2, gate2d, swiglu2d."""
    WIDE = False

    @classmethod
    def _f(cls, t):
        import torch
        return t.to(torch.float64 if cls.WIDE else torch.float32)

    @classmethod
    def _mod(cls, t, M):                                                           # [Ns, C] conditioning -> [M, C] by row-modulo
        Ns = t.shape[0]
        return cls._f(t) if Ns == M else cls._f(t).repeat(M // Ns, 1)

    @classmethod
    def _ln(cls, a, eps):
        import torch
        return torch.nn.functional.layer_norm(cls._f(a), (a.shape[-1],), None, None, eps)

    @classmethod
    def adaln(cls, a, x1, x2, out_dtype, eps=1e-5):
        import torch
        M = a.shape[0]
        return (torch.sigmoid(cls._mod(x1, M)) * cls._ln(a, eps) + cls._mod(x2, M)).to(out_dtype)

    @classmethod
    def gate(cls, o, g, N, out_dtype):
        import torch
        B, H, N_, Cd = o.shape
        return (cls._f(o).permute(0, 2, 1, 3).reshape(B * N_, H * Cd) * torch.sigmoid(cls._f(g))).to(out_dtype)

    gate2d = gate

    @classmethod
    def swiglu(cls, x12, out_dtype):
        import torch
        F_ = x12.shape[1] // 2
        return (torch.nn.functional.silu(cls._f(x12[:, :F_])) * cls._f(x12[:, F_:])).to(out_dtype)

    swiglu2d = swiglu

    @classmethod
    def resgate(cls, gl, x, res, out=None):
        import torch
        y = torch.sigmoid(cls._mod(gl, x.shape[0])) * cls._f(x) + cls._f(res)
        if out is not None:
            out.copy_(y); return out
        return y if cls.WIDE else y.float()

    @classmethod
    def resgate_adaln(cls, gl, x, res, x1, x2, out_dtype, eps=1e-5):
        """In place: res += sigmoid(gl) * x; returns adaln(res, x1, x2)."""
        import torch
        res.copy_(torch.sigmoid(cls._mod(gl, x.shape[0])) * cls._f(x) + cls._f(res))
        return cls.adaln(res, x1, x2, out_dtype, eps)

    @classmethod
    def adaln2(cls, A, sca, sha, sck, shk, out_dtype, eps=1e-5, kv=True):
        """an = sca*LN(A)+sha ; kvn = sck*LN(an)+shk (scales PRE-sigmoided by the caller; the second LN reads an as STORED in out_dtype)."""
        M = A.shape[0]
        an = (cls._mod(sca, M) * cls._ln(A, eps) + cls._mod(sha, M)).to(out_dtype)
        if not kv:
            return an
        kvn = (cls._mod(sck, M) * cls._ln(an, eps) + cls._mod(shk, M)).to(out_dtype)
        return an, kvn

    @classmethod
    def resgate_adaln2(cls, g, x, A, sca=None, sha=None, sck=None, shk=None, out_dtype=None, eps=1e-5, ln=True, kv=True):
        """In place A += g[r%Ns] * x (g PRE-sigmoided); then adaln2 (ln, kv) | its first output only (ln, not kv) | None."""
        A.copy_(cls._mod(g, x.shape[0]) * cls._f(x) + cls._f(A))
        if not ln:
            return None
        return cls.adaln2(A, sca, sha, sck, shk, out_dtype, eps, kv=kv)


torch_dit_rows = _TorchDitRows


def dit_block_rows(*, word="dit_fast", op=None, cc=None, dtype="bf16", n_tokens=None, samples=5, stack=None, capture=False, selection=None):
    """(module-like, Selection): the object whose functions serve the fused DiT block's row ops for this call -- the carried Triton modules for
    row ``dit_fast`` (token stream: adaln / gate / swiglu / resgate / resgate_adaln at c_a rows) and ``dit_fast:atom`` (atom stream: adaln2 /
    resgate_adaln2 / gate2d / swiglu2d at c_atom rows), the torch statements for ``torch_dit_rows``; a tier word (fast | big | exact |
    faithful) resolves per cell: ``op`` names the cell (ditrows_<op>_c768 / atomrows_<op>_c128), the measured winner serves (exact / faithful =
    the statements: the kernels are tolerance-class).  The kits' lever words (ditfast, dit_fused, atom_fused) are aliases of the row words."""
    if selection is None:
        rw, variant = split_word(word)
        if rw in TIER_WORDS:
            if op is None:
                raise ValueError("a tier word needs op= (one of %s / %s)" % (DIT_ROW_OPS[None], DIT_ROW_OPS["atom"]))
            kind = "atomrows" if op in DIT_ROW_OPS["atom"] else "ditrows"
            cellw = cell_word(kind, op=op)
        else:
            cellw = None if op is None else cell_word("atomrows" if op in DIT_ROW_OPS["atom"] else "ditrows", op=op)
        if cc is None:
            import torch
            cc = "%d.%d" % torch.cuda.get_device_capability()
        if stack is None:
            stack = _stack_for_exact(word)                                          # the exact word is vouched per stack: name the running one
        selection = select(cc, dtype, cellw, n_tokens or 800, word=word, samples=samples, stack=stack, capture=capture)
    row, variant = selection.row, selection.variant
    if row == "torch_dit_rows":
        return torch_dit_rows, selection
    if row == "dit_fast":
        return carried_module("dit_fast:atom" if variant == "atom" else "dit_fast"), selection
    raise Refusal("row:%s(not a dit block row)" % row, row, "torch_dit_rows")


def reference(q, k, v, bias, key_mask=None, gate=None, *, scale=None, layout="snhd", dtype=None, inf=1e9):
    """The materialised statement in ``dtype`` (float64 by default), stock arithmetic order (q.k * scale + mask bias + pair bias); returns q's layout."""
    import math
    import torch
    dt = dtype or torch.float64
    (qs, ks, vs), (qh, kh, vh) = _canon_core(q, k, v, layout)
    S, N, H, D = qs.shape
    scale = (1.0 / math.sqrt(D)) if scale is None else float(scale)
    logits = torch.einsum("shnd,shmd->shnm", qh.to(dt) * scale, kh.to(dt))
    if key_mask is not None:
        logits = logits + ((1.0 - key_mask.reshape(-1, N).to(dt)) * (-float(inf)))[:, None, None, :]
    logits = logits + bias.to(dt).reshape(-1, H, N, N)
    o = torch.einsum("shnm,shmd->shnd", torch.softmax(logits, dim=-1), vh.to(dt))
    if gate is not None:
        o = o * torch.sigmoid(_canon_core(gate, gate, gate, layout)[1][0].to(dt))
    return o.permute(0, 2, 1, 3) if layout == "snhd" else o


def atom_reference(q, k, v, bias, gate=None, *, n_queries=32, n_keys=128, scale=None, dtype=None):
    """Windowed statement (the carried atom kernel's own window rule): query block b covers atoms [32b, 32b+32), its keys the 128 atoms centred on the
    block ([32b - 48, 32b + 80) clipped to the sequence with -inf outside); bias [H, n_blocks, 32, 128]."""
    import math
    import torch
    dt = dtype or torch.float64
    S, NA, H, D = q.shape
    scale = (1.0 / math.sqrt(D)) if scale is None else float(scale)
    nb = (NA + n_queries - 1) // n_queries
    out = torch.zeros(S, nb * n_queries, H, D, dtype=dt, device=q.device)
    half = (n_keys - n_queries) // 2
    for b in range(nb):
        q0 = b * n_queries
        k0 = q0 - half
        qi = torch.arange(q0, q0 + n_queries, device=q.device)
        ki = torch.arange(k0, k0 + n_keys, device=q.device)
        qv = qi < NA
        kval = (ki >= 0) & (ki < NA)
        qb = torch.zeros(S, n_queries, H, D, dtype=dt, device=q.device)
        qb[:, qv] = q[:, qi[qv]].to(dt)
        kb = torch.zeros(S, n_keys, H, D, dtype=dt, device=q.device)
        vb = torch.zeros(S, n_keys, H, D, dtype=dt, device=q.device)
        kb[:, kval] = k[:, ki[kval]].to(dt)
        vb[:, kval] = v[:, ki[kval]].to(dt)
        lg = torch.einsum("sqhd,skhd->shqk", qb * scale, kb) + bias[:, b].to(dt)[None]
        lg = lg.masked_fill(~kval[None, None, None, :], float("-inf"))
        ob = torch.einsum("shqk,skhd->sqhd", torch.softmax(lg, -1), vb)
        out[:, q0:q0 + n_queries] = ob
    out = out[:, :NA]
    if gate is not None:
        out = out * torch.sigmoid(gate.to(dt))
    return out

_WARMED = False                                                 # opt_core.warm.auto_warm() ran from this face's first serving call


# ------------------------------------------------------------------------------------------------------------- the ROWS face (rectangular core)
# The core op for a BLOCK of query rows against all N keys — o[s, i, h] for i in this rank's rows [g0, g1) under row sharding
# (opt_core.mem.rowpair.diffusion.dit_block_sharded: the token diffusion transformer's queries = local rows; .transition.apb_local_queries: the
# pairformer's single attention), the pair bias the block's [H, Rq, Nk] ROWS.  Beside pair_bias_attention (square, cell-selected): its own words,
# its own admission (admits_rows / select_rows), its own measured cells (APB_CELLS.json "rows_cells", key grammar ROWS_KEY_GRAMMAR — a section that
# holds MEASURED keys only; absent = no rows cell measured yet: a tier word then serves apb_attn BY NAME, census 'beyond_measured'), the caller's
# statement the fallback (Refusal.fallback None: nothing is substituted here).
ROWS_CELL_WORDS = ("ditrows_h16d48", "pfrows_h16d24")             # token DiT rows (H16 D48, bias shared by S samples) | pairformer single-attention rows (H16 D24, S = 1)
ROWS_KEY_GRAMMAR = "<cc>|<dtype>|<cell>|S<samples>|Q<=<rows>|N<=<keys>|<eager|graph>|fwd"
ROWS_ROWS = ("apb_attn", "sba", "sdpa", "naive")                  # the rows served rectangular (q rows != keys); every other row is refused by name here
ROWS_KERNEL_ROWS = ("apb_attn", "sba", "sdpa")                     # flash rows: no [S, H, q, N] logits transient (a binder may take all local rows per launch)
ROWS_CAST_WORDS = ("big", "fast")                                # the tier words under which fp32 operands are CAST to bf16 for apb_attn (documented policy; a row word never casts)
_ROWS_WS = re.compile(r"\s+")


def _token(text) -> str:
    """A refusal kind / selection reason as ONE ``\S+`` token (a kit's single-token LEVER / census writer refuses blanks by contract): runs of
    whitespace -> ``_``; empty -> ``-``. Log lines may stay readable; every word this face hands to a census passes through here or is built so."""
    t = _ROWS_WS.sub("_", str(text).strip())
    return t or "-"


def _shape_token(t) -> str:
    """``5x16x64x48`` for a tensor's shape (no blanks, unlike ``str(tuple)``); a non-tensor operand: its type name."""
    shape = getattr(t, "shape", None)
    if shape is None:
        return type(t).__name__
    return "x".join(str(int(d)) for d in shape) or "scalar"


def rows_cell_word(kind, heads=None, head_dim=None):
    """The rows cell word of a call class: ``'dit'`` / ``'dit_h16d48'`` (H 16, D 48) -> ``ditrows_h16d48``; ``'pf'`` / ``'pf_h16d24'`` (H 16, D 24) ->
    ``pfrows_h16d24``; a :data:`ROWS_CELL_WORDS` entry passes through; anything else (other geometries included) -> None: no rows cell, the word
    decides by name."""
    k = str(kind or "")
    if k in ROWS_CELL_WORDS:
        return k
    hd = (None if heads is None else int(heads), None if head_dim is None else int(head_dim))
    if k in ("dit", "dit_h16d48", "ditrows") and hd in ((16, 48), (None, None), (16, None), (None, 48)):
        return "ditrows_h16d48"
    if k in ("pf", "pf_h16d24", "pfrows") and hd in ((16, 24), (None, None), (16, None), (None, 24)):
        return "pfrows_h16d24"
    return None


def rows_cells():
    """The measured rows cells (``APB_CELLS.json["rows_cells"]``; ``{}`` until a key is measured — keys are added from measurements only)."""
    return table().get("rows_cells") or {}


def _rows_key_parts(key):
    """``(cc, dtype, cell, S, Q, N, timing)`` of a rows key, or None when it does not parse (:data:`ROWS_KEY_GRAMMAR`)."""
    p = str(key).split("|")
    if len(p) != 8 or not p[3].startswith("S") or not p[4].startswith("Q<=") or not p[5].startswith("N<=") or p[7] != "fwd":
        return None
    try:
        return p[0], p[1], p[2], int(p[3][1:]), int(p[4][3:]), int(p[5][3:]), p[6]
    except ValueError:
        return None


def rows_cell_key(cc, dtype, cell, n_queries, n_keys, *, samples=1, timing="eager"):
    """The measured rows cell serving ``(n_queries, n_keys)``: within the family ``cc|dtype|cell|S<samples>|…|timing|fwd``, the smallest measured
    ``N<=`` bucket >= ``n_keys`` (else the largest, flagged) and inside it the smallest ``Q<=`` bucket >= ``n_queries`` (else the largest, flagged).
    Returns ``(key, size_measured)`` — ``(None, False)`` when the family has no key (or ``n_queries`` / ``n_keys`` is None and the family is empty)."""
    ccw, dt = cc_word(cc), dtype_word(dtype)
    fam = []
    for key in rows_cells():
        parts = _rows_key_parts(key)
        if parts is None:
            continue
        if parts[0] == ccw and parts[1] == dt and parts[2] == cell and parts[3] == int(samples) and parts[6] == timing:
            fam.append((parts[5], parts[4], key))
    if not fam:
        return None, False
    if n_queries is None or n_keys is None:                                          # a static admission: the family exists; the call decides the bucket
        fam.sort()
        return fam[-1][2], False
    ns = sorted({n for n, _q, _k in fam})
    n_in = [n for n in ns if n >= int(n_keys)]
    n_sel, n_ok = (n_in[0], True) if n_in else (ns[-1], False)
    qs = sorted({q for n, q, _k in fam if n == n_sel})
    q_in = [q_ for q_ in qs if q_ >= int(n_queries)]
    q_sel, q_ok = (q_in[0], True) if q_in else (qs[-1], False)
    key = [k for n, q_, k in fam if n == n_sel and q_ == q_sel][0]
    return key, (n_ok and q_ok)


def admits_rows(row, cc, dtype, *, variant=None, head_dim=None, heads=None, samples=None, n_queries=None, n_keys=None, key_mask=False, word=None):
    """Raise :class:`Refusal` when ``row`` cannot serve the ROWS op for this envelope (static facts: the row set, capability, dtype, head dim; the
    kernel's own admission decides strides / int32 bounds at call time, also by name). Returns the admission note: ``''`` or ``'cast:bf16'`` (fp32
    operands under a tier word in :data:`ROWS_CAST_WORDS` on ``apb_attn``: q / k / v are cast to bf16 — 16-bit tensor-core operands, fp32 softmax
    statistics and accumulation inside — the documented cast policy; under the ROW word ``apb_attn`` fp32 is refused by name as everywhere)."""
    if row not in ROWS_ROWS:
        raise Refusal("rows:row:%s(serves:%s)" % (row, ",".join(ROWS_ROWS)), row, None)
    if variant is not None and variant not in VARIANTS.get(row, ()):
        raise Refusal("variant:%s" % variant, row, None)
    ccw, dt = cc_word(cc), dtype_word(dtype)
    tier = split_word(word)[0] if word else None
    note = ""
    if row == "apb_attn":
        if dt == "fp32":
            if tier in ROWS_CAST_WORDS:
                note = "cast:bf16"
            else:
                raise Refusal("dtype:float32", row, None)
        elif dt not in ("bf16", "fp16"):
            raise Refusal("dtype:%s" % dt, row, None)
        if head_dim is not None and not (1 <= int(head_dim) <= 64):
            raise Refusal("head_dim:%s" % head_dim, row, None)
        if float(ccw) < 8.0:
            raise Refusal("cc:%s" % ccw, row, None)
    elif row == "sba":
        if dt not in ("fp32", "bf16", "fp16"):
            raise Refusal("dtype:%s" % dt, row, None)
        if head_dim is not None and int(head_dim) > 128:
            raise Refusal("head_dim:%s" % head_dim, row, None)
        if float(ccw) < 8.0:
            raise Refusal("cc:%s" % ccw, row, None)
    elif row == "sdpa":
        if dt not in ("fp32", "bf16", "fp16"):
            raise Refusal("dtype:%s" % dt, row, None)
        if float(ccw) <= 0.0:
            raise Refusal("device:not_cuda", row, None)
    return note


def select_rows(cc, dtype, cell, n_queries, n_keys, *, word, samples=1, heads=None, head_dim=None, timing="eager", device=None, key_mask=False):
    """The row + facts for a ROWS call. ``word``: a row word (``apb_attn`` | ``sba[:tf32|ieee|tf32x3]`` | ``sdpa[:auto|cudnn|efficient|flash|math]`` |
    ``naive``: that row, admitted or :class:`Refusal`) or a tier word: ``big`` | ``fast`` -> the measured rows cell's winner for ``(n_queries, n_keys)``
    (:func:`rows_cell_key`), else — no rows cell measured for this class yet — ``apb_attn`` BY NAME, reason ``rows:beyond_measured``; ``exact`` |
    ``faithful`` -> :class:`Refusal` ``tier:<word>`` (every rows row is tolerance-class against the fp32 materialised statement: the engine statement is
    the exact path, and it is the caller's).  ``cell``: ``'dit'`` | ``'pf'`` | a :data:`ROWS_CELL_WORDS` entry | None (:func:`rows_cell_word`).  ``cc``
    None: the capability of ``device`` (else of the current CUDA device; ``0.0`` without CUDA).  ``Refusal.fallback`` is None throughout: the rows
    face substitutes nothing — the caller's statement serves a refused call.  One census record per decided call class (provider ``apb``, shape
    ``<rows cell>``, bucket ``Q<=…+N<=…``)."""
    if cc is None:
        cc = _device_cc(device)
    ccw, dt = cc_word(cc), dtype_word(dtype)
    cw = rows_cell_word(cell, heads, head_dim)
    w_row, variant = split_word(word)
    key, measured, reason = None, False, ""
    try:
        if w_row in TIER_WORDS:
            if w_row in ("exact", "faithful"):
                raise Refusal("tier:%s(rows_are_tolerance_class;the_engine_statement_is_the_exact_path)" % w_row, None, None)
            row, variant = "apb_attn", None
            if cw is not None:
                key, measured = rows_cell_key(ccw, dt, cw, n_queries, n_keys, samples=samples, timing=timing)
            if key is not None:
                tcell = rows_cells().get(key) or {}
                win = tcell.get(w_row) or tcell.get("fast") or tcell.get("winner")
                if win:
                    row, variant = split_word(win)
                    reason = "%s_winner:rows_cell%s" % (w_row, "" if measured else ":nearest_bucket:beyond_measured")
                    try:
                        note = admits_rows(row, ccw, dt, variant=variant, head_dim=head_dim, heads=heads, samples=samples, word=w_row)
                    except Refusal as e:                                               # the cell's winner cannot serve this envelope: apb_attn by name, said so
                        reason += ",skip:%s(%s)" % (row, _token(e.kind))
                        row, variant = "apb_attn", None
                        note = admits_rows(row, ccw, dt, head_dim=head_dim, heads=heads, samples=samples, word=w_row)
                else:
                    reason = "rows_cell:%s:no_%s_winner->apb_attn:by_name" % (key, w_row)
                    note = admits_rows(row, ccw, dt, head_dim=head_dim, heads=heads, samples=samples, word=w_row)
            else:
                reason = "rows:beyond_measured(no_rows_cell:%s:S%s:cc%s:%s)->apb_attn:by_name" % (cw or "h%sd%s" % (heads, head_dim), samples, ccw, dt)
                note = admits_rows(row, ccw, dt, head_dim=head_dim, heads=heads, samples=samples, word=w_row)
        elif w_row in ROWS_ROWS:
            row = w_row
            note = admits_rows(row, ccw, dt, variant=variant, head_dim=head_dim, heads=heads, samples=samples, word=w_row)
            reason = "row_word"
            if cw is not None:
                key, measured = rows_cell_key(ccw, dt, cw, n_queries, n_keys, samples=samples, timing=timing)
        elif w_row in ROW_NAMES:
            raise Refusal("rows:row:%s(square_only;serves:%s)" % (w_row, ",".join(ROWS_ROWS)), w_row, None)
        else:
            raise Refusal("rows:word:%s(unknown;rows_words:%s|tier:big,fast)" % (_token(word), ",".join(ROWS_ROWS)), None, None)
    except Refusal as r:
        _rows_census(ccw, dt, cw, heads, head_dim, n_queries, n_keys, samples, timing, word, None, None, r)
        raise
    if note:
        reason = (reason + "," if reason else "") + note
    if variant is None:
        variant = DEFAULT_VARIANT.get(row)
    cls, exact_vs, bwd, cap, _fb = _row_facts(row)
    sel = Selection(row, variant if row in VARIANTS else None, word, key, bool(measured), None, False, cls, exact_vs, bwd, cap, None, None, reason[:240], None)
    _rows_census(ccw, dt, cw, heads, head_dim, n_queries, n_keys, samples, timing, word, sel, key, None)
    return sel


def _device_cc(device=None):
    try:
        import torch
    except ImportError:
        return (0, 0)
    if not torch.cuda.is_available():
        return (0, 0)
    if device is not None and getattr(device, "type", str(device).split(":")[0]) != "cuda":
        return (0, 0)
    return tuple(torch.cuda.get_device_capability(device))


def _rows_census(ccw, dt, cw, heads, head_dim, n_queries, n_keys, samples, timing, word, sel, key, refusal):
    try:
        bucket = "Q=%s+N=%s" % ("-" if n_queries is None else n_queries, "-" if n_keys is None else n_keys)
        if key is not None and sel is not None and sel.size_measured:
            p = _rows_key_parts(key)
            bucket = "Q<=%d+N<=%d" % (p[4], p[5]) if p else bucket
        ckey = dict(cc=ccw, stack=None, dtype=dt, shape=cw or "rows_h%sd%s" % (heads, head_dim), bucket=bucket, form="%s.S%s" % (timing, samples), word=word)
        if refusal is not None:
            _CENSUS.record("apb", ckey, "named_fallback", "caller:-", cell_id=None, refused="%s:%s" % (refusal.row or word, refusal.kind), note="rows:raised")
        elif split_word(word)[0] in ROWS_ROWS:
            _CENSUS.record("apb", ckey, "opt_in", arm_word(sel.row, sel.variant), cell_id=key, note="rows:" + (sel.reason or ""))
        elif key is None or not sel.size_measured:
            _CENSUS.record("apb", ckey, "inherited", arm_word(sel.row, sel.variant), cell_id=key, note="rows:family:%s(%s)" % ("nearest" if key else "none", sel.reason or ""))
        else:
            _CENSUS.record("apb", ckey, "cell_hit", arm_word(sel.row, sel.variant), cell_id=key, note="rows")
    except Exception:                                                                 # noqa: BLE001 — a census never gates a served call
        pass


def reference_rows(q, k, v, bias, key_mask=None, gate=None, *, scale=None, dtype=None, inf=1e9):
    """The materialised ROWS statement in ``dtype`` (float64 by default), stock arithmetic order: ``softmax_j(scale q[s,h,i].k[s,h,j] + (mask_j==0)*-inf
    + bias[s|0,h,i,j]) @ v`` (optionally ``* sigmoid(gate)``).  q ``[S,H,Rq,D]``, k / v ``[S,H,Nk,D]``, bias ``[H,Rq,Nk]`` | ``[1|S,H,Rq,Nk]``, key_mask
    ``[Nk]`` | ``[1|S,Nk]`` | None, gate ``[S,Rq,H*D]`` | None -> ``[S, Rq, H*D]``."""
    import math
    import torch
    dt = dtype or torch.float64
    S, H, R, D = (int(x) for x in q.shape)
    NK = int(k.shape[2])
    scale = (1.0 / math.sqrt(D)) if scale is None else float(scale)
    b = bias.to(dt)
    if b.dim() == 3:
        b = b[None]
    logits = torch.einsum("shid,shjd->shij", q.to(dt) * scale, k.to(dt))
    if key_mask is not None:
        km = key_mask.reshape(-1, NK).to(dt)
        logits = logits + ((1.0 - km) * (-float(inf)))[:, None, None, :]
    logits = logits + b
    o = torch.einsum("shij,shjd->sihd", torch.softmax(logits, dim=-1), v.to(dt)).reshape(S, R, H * D)
    if gate is not None:
        o = o * torch.sigmoid(gate.to(dt).reshape(S, R, H * D))
    return o


def pair_bias_attention_rows(q, k, v, bias, key_mask=None, *, word=None, selection=None, scale=None, gate=None, out=None, out_dtype=None, inf=1e9,
                            config=None, kind="dit", timing="eager"):
    """Serve the ROWS core op — a block of ``Rq`` query rows against all ``Nk`` keys — through the selected row (beside :func:`pair_bias_attention`, whose
    rows take the square ``N x N`` problem):

        q [S, H, Rq, D]   k, v [S, H, Nk, D]   bias [H, Rq, Nk] (shared by the S samples) | [1|S, H, Rq, Nk]   key_mask [Nk] | [1|S, Nk] (nonzero = attend) | None
        gate [S, Rq, H*D] pre-sigmoid | None   out [S, Rq, H*D] | None   ->   (o [S, Rq, H*D] in q's dtype (``out_dtype`` / the cast dtype under a cast note), Selection)

    Any strides with unit stride on ``D`` (q / k / v as the transposed views of the projections) and on ``Nk`` for the bias (the ``[H, q, N]`` rows of a
    ``[H, R, N]`` block are read IN PLACE).  ``word`` (or a ``selection`` from :func:`select_rows`, reused across calls of one class): a row word or a tier
    word — see :func:`select_rows` (``big`` / ``fast``: the measured rows cell's winner, else ``apb_attn`` by name flagged ``beyond_measured``; ``exact`` /
    ``faithful``: refused by name).  Rows: ``apb_attn`` (:mod:`opt_core.kernels.apb_attn` through :mod:`opt_core.attn.apb_core`: 16-bit operands — fp32
    under a tier word is cast to bf16 here, note ``cast:bf16`` — fp32 online-softmax statistics, fp32 accumulation, one output rounding; per-query
    arithmetic independent of ``Rq``: P-invariant), ``sba[:tf32|ieee|tf32x3]`` (:mod:`opt_core.attn.shared_bias_attn`: fp32 operands at the named product
    precision; the key mask folded into the bias), ``sdpa[:…]`` (torch fused SDPA with the bias (+ mask) as ``attn_mask``), ``naive`` (:func:`reference_rows`
    in fp32: the materialised statement).  Every refusal is a :class:`Refusal` ``(kind, row, None)`` BEFORE any work — the caller's statement serves."""
    global _WARMED
    if not _WARMED:
        _WARMED = True
        from opt_core.warm import auto_warm
        auto_warm("apb")
    import math
    import torch
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise Refusal("rows:shape:qkv(want:q=SxHxRqxD,kv=SxHxNkxD;got:q=%s,k=%s,v=%s)" % (_shape_token(q), _shape_token(k), _shape_token(v)), None, None)
    S, H, R, D = (int(x) for x in q.shape)
    NK = int(k.shape[2])
    if tuple(k.shape) != (S, H, NK, D) or tuple(v.shape) != (S, H, NK, D):
        raise Refusal("rows:shape:kv(k=%s,v=%s,q=%s)" % (_shape_token(k), _shape_token(v), _shape_token(q)), None, None)
    b = bias
    while b.dim() > 4 and int(b.shape[0]) == 1:
        b = b[0]
    if b.dim() == 4 and int(b.shape[0]) == 1:
        b = b[0]
    if not ((b.dim() == 3 and tuple(b.shape) == (H, R, NK)) or (b.dim() == 4 and tuple(b.shape) == (S, H, R, NK))):
        raise Refusal("rows:shape:bias(%s;want:HxRqxNk=%dx%dx%d|SxHxRqxNk)" % (_shape_token(bias), H, R, NK), None, None)
    km = key_mask
    if km is not None:
        while km.dim() > 2 and int(km.shape[0]) == 1:
            km = km[0]
        if km.dim() == 1:
            km = km[None, :]
        if km.dim() != 2 or int(km.shape[-1]) != NK or int(km.shape[0]) not in (1, S):
            raise Refusal("rows:shape:key_mask(%s;want:Nk|1xNk|SxNk)" % (_shape_token(key_mask),), None, None)
    if gate is not None and tuple(gate.shape) != (S, R, H * D):
        raise Refusal("rows:shape:gate(%s;want:SxRqxH*D=%dx%dx%d)" % (_shape_token(gate), S, R, H * D), None, None)
    cc = torch.cuda.get_device_capability(q.device) if q.is_cuda else (0, 0)
    scale = (1.0 / math.sqrt(D)) if scale is None else float(scale)
    if selection is None:
        selection = select_rows(cc, q.dtype, kind, R, NK, word=word, samples=S, heads=H, head_dim=D, timing=timing, device=q.device, key_mask=km is not None)
    row, variant = selection.row, selection.variant
    _LAST["pair_bias_attention_rows"] = row
    od = out_dtype or q.dtype
    cast = "cast:bf16" in (selection.reason or "")
    if row == "apb_attn":
        core = _imp("opt_core.attn.apb_core", row, None)
        q_, k_, v_ = q, k, v
        if q.dtype == torch.float32:
            if not cast:
                raise Refusal("dtype:float32", row, None)
            q_, k_, v_ = q.to(torch.bfloat16), (k if k.dtype != torch.float32 else k.to(torch.bfloat16)), (v if v.dtype != torch.float32 else v.to(torch.bfloat16))
        elif k.dtype != q.dtype or v.dtype != q.dtype:
            raise Refusal("dtype:mixed(q=%s,k=%s,v=%s)" % (dtype_word(q.dtype), dtype_word(k.dtype), dtype_word(v.dtype)), row, None)
        o16 = None
        if out is not None and out.dtype == q_.dtype:
            o16 = out
        try:
            o = core.pair_bias_attention(q_, k_, v_, b, km, gate=gate, out=o16, num_samples=S, num_heads=H, layout="shnd", scale=scale, inf=inf, config=config)
        except core.Unsupported as e:
            raise Refusal(_token(getattr(e, "event", str(e))), row, None)
        except Exception as e:                                                        # noqa: BLE001 — a compile / launch failure is a refusal by name; OOM propagates
            from ..oom import is_oom
            if is_oom(e): raise                                                       # noqa: E701
            raise Refusal("launch:%s" % type(e).__name__, row, None)
        o = o.reshape(S, R, H * D)
    elif row == "sba":
        SBA = carried_module("sba")
        bb = fold_mask(b if b.dim() == 4 else b[None], km, inf=inf) if km is not None else b
        try:
            o, _ev = SBA.attention(q, k, v, bb, scale=scale, input_precision=(variant or "tf32"), pad_head_dim=True,
                                   per_sample=("rows1" if (bb.dim() == 4 and int(bb.shape[0]) == S and S > 1) else None))
        except Exception as e:                                                        # noqa: BLE001 — OOM propagates; the row's own refusal / a launch failure is a refusal by name
            from ..oom import is_oom
            if is_oom(e): raise                                                       # noqa: E701
            if type(e).__name__ in ("Refused", "Refusal", "Unsupported"):
                raise Refusal("sba:%s" % _token(str(getattr(e, "event", str(e)))[:60]), row, None)
            raise Refusal("launch:%s" % type(e).__name__, row, None)
        if o.dim() == 5:
            o = o[0]
        o = o.permute(0, 2, 1, 3).reshape(S, R, H * D)                                 # [S, H, Rq, D] -> [S, Rq, H*D]
        if gate is not None:
            o = o * torch.sigmoid(gate.to(o.dtype))
    elif row == "sdpa":
        SB = _imp("opt_core.attn.sdpa_bias", row, None)
        bb = fold_mask(b if b.dim() == 4 else b[None], km, inf=inf) if km is not None else (b if b.dim() == 4 else b[None])
        try:
            o, _ev = SB.sdpa_bias(q, k, v, bb.to(q.dtype) if bb.dtype != q.dtype and q.dtype != torch.float32 else bb, None, layout="bhld", bias_layout="bhqk",
                                  scale=scale, backend=(variant or "auto"), out_dtype=od)
        except SB.Refused as e:
            raise Refusal("sdpa:%s" % _token(str(getattr(e, "event", str(e)))[:60]), row, None)
        o = o.permute(0, 2, 1, 3).reshape(S, R, H * D)
        if gate is not None:
            o = o * torch.sigmoid(gate.to(o.dtype))
    elif row == "naive":
        o = reference_rows(q, k, v, b, km, gate, scale=scale, dtype=torch.float32, inf=inf)
    else:
        raise Refusal("rows:row:%s" % row, row, None)
    if o.dtype != od and not (cast and row == "apb_attn" and out_dtype is None):
        o = o.to(od)
    if out is not None and o is not out:
        out.copy_(o)
        o = out
    _book_rows(S, H, R, NK, D, row)
    return o, selection


_ROWS_SERVED = {}                                                                      # 'row:S<s>H<h>R<r>N<n>D<d>' -> calls (describe_rows)


def _book_rows(S, H, R, NK, D, row):
    key = "%s:S%dH%dR%dN%dD%d" % (row, S, H, R, NK, D)
    _ROWS_SERVED[key] = _ROWS_SERVED.get(key, 0) + 1


def describe_rows():
    """The rows face's served call classes this process (``{'<row>:S..H..R..N..D..': calls}``) and its constants, as data."""
    return {"served": dict(_ROWS_SERVED), "rows": ROWS_ROWS, "kernel_rows": ROWS_KERNEL_ROWS, "cell_words": ROWS_CELL_WORDS, "key_grammar": ROWS_KEY_GRAMMAR,
            "cast_words": ROWS_CAST_WORDS, "measured_keys": sorted(rows_cells().keys())}


# caller-layout guard: assertions under a serving call surface as named refusals (never an assert on a caller's operands)
pair_bias_attention = _caller_layout_refusals(pair_bias_attention)
atom_attention = _caller_layout_refusals(atom_attention)
pair_bias_planes = _caller_layout_refusals(pair_bias_planes)
pair_bias_attention_rows = _caller_layout_refusals(pair_bias_attention_rows)
