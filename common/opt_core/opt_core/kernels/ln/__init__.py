"""LayerNorm family: ONE provider over every carried implementation ("row") and the measured cell table (LN_CELLS.json).

The op::

    forward    y = (x - mean(x)) * rsqrt(var(x) + eps) * gamma + beta over the last dimension of a [rows, C] tensor
               forms: fp32 | bf16 (bf16 x, bf16 params, bf16 out) | bf16w (the autocast form: bf16 x, fp32 gamma/beta, fp32 arithmetic and OUTPUT)
                      | bf16o (the compiled-extension form: bf16 x, fp32 gamma/beta, fp32 statistics, bf16 OUTPUT -- what a trunk's fast_layernorm
                        returns under bf16 autocast; its statement is aten_autocast followed by .to(bfloat16); OPT-IN through out="bf16" / out_dtype)
    backward   dx of the same op for FROZEN gamma/beta given (grad_y, x, mean, rstd), the residual-link gradient optionally folded in (the design
               loop's trunk: five per pair block backward); dgamma/dbeta are not produced
    lnlinear   LN -> a wide Linear (C 384 -> 384 rows; the narrow LN -> pair-bias producers live in opt_core.kernels.apb)

Rows (``ROW_NAMES``; ``LN_CELLS.json["rows"]`` states each one's envelope, numerics class, capture safety and named fallback):

    exactln            kernels/ln/exactln  (carried package: NVRTC replica of ATen's vectorized LayerNorm kernel; variants cuda | widen | triton)   EXACT vs aten
    fastln             kernels/ln/fastln   (carried module: Triton row LayerNorm, fp32 two-pass statistics; variant lp = fastln/lpout.py, house-written:
                       the same kernel with fp32 arithmetic pinned and the OUTPUT dtype named -- the bf16o form's row)                              fast
    ln_rows            opt_core.kernels.ln_proj.layernorm_rows                                                                                     fast
    rfd                opt_core.kernels.rfd_layernorm.triton_layer_norm                                                                              fast (bf16 cells measured bitwise: a parity flag)
    dtk_ln             opt_core.kernels.dtk_kernels.ln_modulate (no scale/shift)                                                                    fast
    ln_proj_ln_linear  opt_core.kernels.ln_proj.ln_linear (the lnlinear cell)                                                                       fast
    ef2_ln_bwd_dx      kernels/ln/ef2      (carried module: dx-only backward, bitwise to the vendored backward kernel it replaces)                   EXACT vs esm_vendored_bwd, backward
    stock rows         aten, aten_autocast (widen form), fast_layernorm (a compiled extension: capture-UNSAFE, named), aten_autograd, aten_autograd_autocast,
                       esm_vendored_bwd, aten_bwd, torch_ln_linear1 -- named, nothing carried

Selection.  ``select(cc, dtype, cell, n_tokens, word=...)`` -> ``Selection`` (pure; standard library + the table):
    * ``word`` = a row name (``row[:variant]``: "exactln", "exactln:widen", "exactln:triton"): the DEFAULT RULE is exactly that row;
    * ``word`` = a tier word (fast | exact | faithful | big): OPT-IN, the cell's measured winner (per stack where measured, else the reference stack);
      ``capture=True``: capture-unsafe rows refused / never winners, graph cells consulted; ``prefer=`` narrows a tier to the caller's rows.
    Nothing here changes what an existing caller of kernels.lnl_fused / ln_proj / rfd_layernorm executes.

Serving.  ``layer_norm(x, normalized_shape, weight, bias, eps, word=...)``, ``layer_norm_backward_dx(...)``, ``ln_linear(...)`` import torch inside the
call.  ``exact_prologue()`` hands a fused Triton cell the exact row's block functions (``mean_rstd_128 / mean_rstd_64 / affine_rows / ln_rows_128 /
ln_rows_64``) so its in-kernel LayerNorm produces ATen's bits (the pair-fused 'aten' arithmetic word).  ``reference(...)`` is the fp64 yardstick.

Standard library only at import.
"""
import json
import os
import re
from collections import namedtuple

from ... import cell_census as _CENSUS                            # stdlib-only: the coverage census (one record per decided call class; pure observation)

ROW_NAMES = ("exactln", "fastln", "fast_layernorm_ext", "ln_rows", "rfd", "dtk_ln", "ln_proj_ln_linear", "ef2_ln_bwd_dx",
             "aten", "aten_autocast", "fast_layernorm", "aten_autograd", "aten_autograd_autocast", "esm_vendored_bwd", "aten_bwd", "torch_ln_linear1")
VARIANTS = {"exactln": ("cuda", "widen", "triton"), "fastln": ("lp",)}
FORM_WORDS = ("fp32", "bf16", "bf16w", "bf16o")
DEFAULT_VARIANT = {}
TIER_WORDS = ("fast", "exact", "faithful", "big")
STOCK_ROWS = ("aten", "aten_autocast", "fast_layernorm", "aten_autograd", "aten_autograd_autocast", "esm_vendored_bwd", "aten_bwd", "torch_ln_linear1")
EXACT_ROWS = ("exactln", "ef2_ln_bwd_dx")
BACKWARD_ROWS = ("ef2_ln_bwd_dx", "aten", "aten_autocast", "aten_autograd", "aten_autograd_autocast", "esm_vendored_bwd", "aten_bwd", "torch_ln_linear1")
CAPTURE_UNSAFE_ROWS = ("fast_layernorm",)
ARM_ALIASES = {"fast_layernorm_ext": "fast_layernorm"}
KIT_SIDE_ARMS = {"fast_layernorm": "fast_layernorm_ext"}                        # measured arms that are an engine's own object, not a row the provider executes: a TIER word
                                                                                # serves them through the carried row named here or skips to the cell's next measured row


def _aten_for(form, pass_="fwd"):
    """The ATen statement row of a form (the last resort when a cell's measured rows all refuse): bf16o / bf16w -> aten_autocast, else aten."""
    if pass_ != "fwd":
        return _stock_for(form, pass_)
    return "aten_autocast" if form in ("bf16o", "bf16w") else "aten"                               # a row whose measured arm carries another name in the cells: fast_layernorm_ext IS the
                                                                                # trunks' extension (same device code, current-stream launches) = the arm fast_layernorm
CARRIED_SUBPACKAGES = {"exactln": "exactln", "fastln": "fastln", "ef2_ln_bwd_dx": "ef2"}
CELL_WORDS = ("pair_c64", "pair_c128", "pair_c256", "pair_c384", "pair_c512", "single_c384", "atom_c128", "dit_c768", "lnlinear_c384",
              "pairbwd_c256_rowmajor", "pairbwd_c256_rowmajor_res", "pairbwd_c256_cmajor")
PASS_WORDS = ("fwd", "bwd", "fwdbwd")
EXACTLN_PROVEN_CC = ("9.0", "8.0")
CELLS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LN_CELLS.json")

Selection = namedtuple("Selection", "row variant word cell size_measured stack stack_measured cls exact_vs backward capture_safe x_stock fallback reason config")


class Refusal(Exception):
    """A row cannot serve this call: ``kind`` is the word for the kit's LEVER line, ``row`` the row that refused, ``fallback`` the row the table
    names instead (the kit decides whether to take it -- nothing is substituted here)."""

    def __init__(self, kind, row=None, fallback=None):
        Exception.__init__(self, "%s%s%s" % (kind, (" [row %s]" % row) if row else "", (" -> fallback row %s" % fallback) if fallback else ""))
        self.kind, self.row, self.fallback = kind, row, fallback


_TABLE = {}


def table():
    """The parsed LN_CELLS.json (cached)."""
    if "t" not in _TABLE:
        with open(CELLS_PATH, encoding="utf-8") as fh:
            _TABLE["t"] = json.load(fh)
    return _TABLE["t"]


def cc_word(cc):
    if isinstance(cc, (tuple, list)):
        return "%d.%d" % (int(cc[0]), int(cc[1]))
    s = str(cc)
    if s.isdigit() and len(s) >= 2:
        return "%s.%s" % (s[:-1], s[-1])
    return "%.1f" % float(s)


def dtype_word(dtype, widen=False, out=None):
    """'fp32' | 'bf16' | 'bf16w' (bf16 x, fp32 parameters, fp32 output: the autocast form) | 'bf16o' (bf16 x, fp32 parameters, bf16 OUTPUT: the
    compiled-extension form; ``widen`` with ``out`` = 'bf16' / torch.bfloat16) from a word or torch dtype."""
    s = str(dtype).replace("torch.", "")
    w = {"bfloat16": "bf16", "bf16": "bf16", "bf16w": "bf16w", "bf16o": "bf16o", "float32": "fp32", "fp32": "fp32", "float": "fp32", "float16": "fp16", "fp16": "fp16"}.get(s, s)
    if w == "bf16" and widen:
        return "bf16o" if str(out).replace("torch.", "") in ("bf16", "bfloat16") else "bf16w"
    return w


def split_word(word):
    if word and ":" in word:
        r, v = word.split(":", 1)
        return r, v
    return word, None


def arm_word(row, variant=None):
    return row if not variant or variant == "cuda" else "%s:%s" % (row, variant)


def cell_word(kind, C, layout=None):
    """The table's cell segment: kind = pair (N^2 rows) | single (N rows) | atom (8N rows) | dit (5N rows) | lnlinear | pairbwd (+ layout rowmajor |
    rowmajor_res | cmajor).  None when the family does not list the width (the caller gets the stock row by name)."""
    if kind == "pairbwd":
        w = "pairbwd_c%d_%s" % (int(C), layout or "rowmajor")
    else:
        w = "%s_c%d" % (kind, int(C))
    return w if w in CELL_WORDS else None


ROWS_OF_KIND = {"pair": lambda n: n * n, "pairbwd": lambda n: n * n, "single": lambda n: n, "lnlinear": lambda n: n, "atom": lambda n: 8 * n, "dit": lambda n: 5 * n}
# rows a cell was MEASURED at, by family kind, from its N (the table's cells carry "rows" too; this is the same arithmetic for cells without it)


class CellWord(str):
    """A cell word (``pair_c128`` ...) that also carries the live call's ROW count (``.rows``): select() then buckets by rows -- the measured
    quantity a LayerNorm's cost depends on -- instead of the caller's nominal n_tokens (which a generic binder cannot always know: an atom
    tensor's second-to-last dim is an atom count, a DiT tensor's is samples x tokens).  Prints and compares as the plain word."""
    __slots__ = ("rows",)

    def __new__(cls, word, rows=None):
        o = str.__new__(cls, word)
        o.rows = None if rows is None else int(rows)
        return o


def cell_rows(key):
    """The row count the cell ``key`` was measured at (its "rows" field, else the family arithmetic on its N)."""
    c = table()["cells"].get(key) or {}
    if "rows" in c:
        return int(c["rows"])
    parts = str(key).split("|")
    kind = parts[2].split("_c")[0]
    return ROWS_OF_KIND.get(kind, lambda n: n)(int(parts[3][3:].split("+")[0]))


_ROW_RANGES = {}


def _row_range(kind, C):
    """(min rows, max rows) over every measured fwd cell of the family kind_c<C> (any cc / form / timing), or None."""
    k = (kind, int(C))
    if k not in _ROW_RANGES:
        w = "%s_c%d" % (kind, int(C))
        rs = [cell_rows(key) for key in table()["cells"] if key.split("|")[2] == w]
        _ROW_RANGES[k] = (min(rs), max(rs)) if rs else None
    return _ROW_RANGES[k]


def cell_for_rows(rows, n_tokens, C):
    """Pick the cell family for a live LayerNorm call from its ROW count (rows = numel / C) and return a :class:`CellWord` carrying the rows:
    among the families that list this width (pair N^2 rows | atom 8N | dit 5N | single N), the one whose MEASURED row range holds ``rows``
    (the smallest measured cell at or above it being nearest), else the family whose range is nearest; None when no family lists the width
    (the caller gets the stock row by name).  ``n_tokens`` is informational (kept for callers that pass a true token count)."""
    r = max(int(rows), 1)
    best, best_score = None, None
    for kind in ("pair", "atom", "dit", "single"):
        w = cell_word(kind, C)
        rng = _row_range(kind, C) if w else None
        if not rng:
            continue
        lo, hi = rng
        if r <= hi:                                                              # inside the measured range (a cell at or above r exists)
            cells = sorted(cell_rows(key) for key in table()["cells"] if key.split("|")[2] == w)
            up = min(c for c in cells if c >= r)
            score = (0, up / float(r))                                          # nearest measured cell above, as a ratio
        else:
            score = (1, r / float(hi))                                          # beyond every cell of the family: how far beyond
        if best_score is None or score < best_score:
            best, best_score = w, score
    return CellWord(best, r) if best else None


_FAMILIES = {}


def cell_family(cc, dtw, cell, timing, pas):
    fk = (cc_word(cc), dtw, cell, timing, pas)
    if fk not in _FAMILIES:
        fam = {}
        pre = "%s|%s|%s|" % fk[:3]
        for key in table()["cells"]:
            if key.startswith(pre):
                parts = key.split("|")
                if parts[4] == timing and parts[5] == pas:
                    fam[int(parts[3][3:])] = key
        _FAMILIES[fk] = fam
    return _FAMILIES[fk]


def cell_key(cc, dtype, cell, n_tokens, *, timing="eager", pass_="fwd", widen=False, out=None, rows=None):
    """(key | None, size_measured, note): smallest measured size >= n_tokens of the (cc, dtype form, cell, timing, pass) family (size_measured
    True), else the largest with size_measured False (beyond_measured); a timing without cells falls to 'eager' (the note marks it: _select
    serves a tier word the statement by name there -- the nearest rule is size-only)."""
    dtw = dtype_word(dtype, widen, out)
    fam = cell_family(cc, dtw, cell, timing, pass_)
    note = ""
    if not fam and timing != "eager":
        fam = cell_family(cc, dtw, cell, "eager", pass_)
        note = "timing %s not measured (eager cell)" % timing
    if not fam:
        return None, False, "no measured cell for %s|%s|%s|%s" % (cc_word(cc), dtw, cell, pass_)
    sizes = sorted(fam)
    if rows is not None:                                                         # bucket by ROWS (the measured quantity): the smallest cell measured at >= rows
        r = int(rows)
        by_rows = sorted((cell_rows(fam[size]), size) for size in sizes)
        for cr, size in by_rows:
            if r <= cr:
                return fam[size], True, note
        return fam[by_rows[-1][1]], False, (note + "; " if note else "") + "beyond_measured(largest %d rows)" % by_rows[-1][0]
    n = int(n_tokens)
    for size in sizes:
        if n <= size:                                                            # inside the measured range: the smallest measured size >= n serves
            return fam[size], True, note
    return fam[sizes[-1]], False, (note + "; " if note else "") + "beyond_measured(largest %d)" % sizes[-1]


def _row_facts(row):
    r = table()["rows"][row]
    return r["class"], r.get("exact_vs"), bool(r.get("backward")), bool(r.get("capture_safe", True)), r.get("fallback")


def _ver(v):
    try:
        parts = str(v).split("+")[0].split(".")
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return (0, 0)


def _ln_ext():
    from opt_core.kernels.ln import ext_loader as S
    return S


def admits(row, cc, dtype, C=None, *, variant=None, widen=False, out=None, pass_="fwd", capture=False, rows=None, aligned=True, affine=True, has_nvrtc=None,
           has_triton=None, has_esm=None, layout="rowmajor", stack=None):
    """Raise ``Refusal`` when ``row`` cannot serve this envelope (static facts: capability, dtype form, width, pass, capture state, libraries)."""
    if row not in ROW_NAMES:
        raise ValueError("unknown row %r (rows: %s)" % (row, ", ".join(ROW_NAMES)))
    if variant is not None and variant not in VARIANTS.get(row, ()):
        raise Refusal("variant:%s" % variant, row, row)
    ccw, dt = cc_word(cc), dtype_word(dtype, widen, out)
    fb = (_row_facts(row)[4] or "").split("|")[0] or None
    if dt == "bf16o" and fb == "aten":
        fb = "aten_autocast"
    if capture and row in CAPTURE_UNSAFE_ROWS:
        raise Refusal("capture_unsafe", row, fb)
    if pass_ == "bwd" and row not in BACKWARD_ROWS:
        raise Refusal("forward_only", row, "esm_vendored_bwd" if has_esm else "aten_bwd")
    if pass_ == "fwdbwd" and row not in ("aten_autograd", "aten_autograd_autocast", "aten", "aten_autocast"):
        raise Refusal("forward_only(no autograd pairing)", row, "aten_autograd")
    if row == "exactln":
        if ccw not in EXACTLN_PROVEN_CC:
            raise Refusal("cc:%s not proven (9.0, 8.0)" % ccw, row, fb)
        if has_nvrtc is False and variant != "triton":                          # neither cuda.bindings nor the ctypes bindings (libcuda + libnvrtc) load
            raise Refusal("import:nvrtc(cuda.bindings|ctypes)", row, fb)
        if C is not None and (int(C) % 4 or int(C) < 4 or int(C) > 1024):
            raise Refusal("width:C%s" % C, row, fb)
        if variant == "triton" and C is not None and int(C) not in (64, 128):
            raise Refusal("width:C%s(triton variant serves 64|128)" % C, row, fb)
        if variant == "triton" and dt != "fp32":
            raise Refusal("dtype:%s(triton variant fp32)" % dt, row, fb)
        if variant == "widen" and dt not in ("bf16w", "bf16o"):
            raise Refusal("dtype:%s(widen = bf16 x, fp32 params)" % dt, row, fb)
        if variant in (None, "cuda") and dt in ("bf16w", "bf16o"):
            raise Refusal("dtype:%s(use variant widen)" % dt, row, "exactln:widen")
        if dt == "fp16":
            raise Refusal("dtype:fp16", row, fb)
        if not aligned:
            raise Refusal("aten_rowwise_path(misaligned rows)", row, fb)
    elif row == "fastln":
        if has_triton is False:
            raise Refusal("triton_missing", row, fb)
        if variant != "lp" and dt == "bf16w":
            raise Refusal("dtype:bf16w(outputs x dtype; variant lp names the output dtype)", row, "fastln:lp")
        if variant != "lp" and dt == "bf16o":
            raise Refusal("dtype:bf16o(statistics in x dtype; use variant lp)", row, "fastln:lp")
    elif row == "fast_layernorm_ext":
        S = _ln_ext()
        fbx = "fastln:lp" if dt == "bf16o" else ("aten_autocast" if dt == "bf16w" else "aten")   # the form's next row: bf16-out -> the Triton bf16-out row; else the statement
        if dt == "bf16w":
            raise Refusal("dtype:bf16w(outputs x dtype: the bf16-out form is bf16o)", row, fbx)
        if not S.landed():
            raise Refusal("not_landed(kernels/ln/rows/fast_layernorm_ext)", row, fbx)
        key = S.key_for_stack(stack) if stack is not None else None
        if stack is not None and key is None:
            raise Refusal("no_prebuilt:%s(keys %s)" % (str(stack).split("/")[0], "+".join(S.keys())), row, fbx)
        ok, why = S.available(key)
        if not ok:
            raise Refusal(why, row, fbx)
    elif row == "ln_rows":
        if C is not None and int(C) not in (64, 128, 256, 384):
            raise Refusal("width:C%s" % C, row, fb)
        if dt in ("bf16w", "bf16o"):
            raise Refusal("dtype:%s" % dt, row, "aten_autocast")
    elif row == "rfd":
        if has_triton is False:
            raise Refusal("triton_missing", row, fb)
        if dt in ("bf16w", "bf16o"):
            raise Refusal("dtype:%s" % dt, row, "aten_autocast")
    elif row == "dtk_ln":
        if has_triton is False:
            raise Refusal("triton_missing", row, fb)
        if dt == "bf16w":
            raise Refusal("dtype:bf16w", row, "aten_autocast")
    elif row == "ln_proj_ln_linear":
        if C is not None and int(C) not in (64, 128, 256, 384):
            raise Refusal("width:C%s" % C, row, fb)
    elif row == "aten":
        if dt in ("bf16w", "bf16o"):                                              # F.layer_norm raises on bf16 x with fp32 parameters: the autocast statement serves the form
            raise Refusal("dtype:%s(F.layer_norm dtype rule)" % dt, row, "aten_autocast")
    elif row == "ef2_ln_bwd_dx":
        if pass_ != "bwd":
            raise Refusal("backward_row", row, "aten")
        if C is not None and int(C) != 256:
            raise Refusal("width:C%s(fallback:vendored)" % C, row, fb)
        if layout == "cmajor":
            raise Refusal("layout:cmajor(left on the vendored kernel)", row, fb)
        if has_triton is False:
            raise Refusal("triton_missing", row, fb)
    elif row == "esm_vendored_bwd":
        if has_esm is False:
            raise Refusal("import:esm_kernels", row, "aten_bwd")
    elif row == "fast_layernorm":
        pass
    return True


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


def _stock_for(dtw, pass_):
    if pass_ == "bwd":
        return "aten_bwd"
    if pass_ == "fwdbwd":
        return "aten_autograd_autocast" if dtw == "bf16w" else "aten_autograd"
    return "aten_autocast" if dtw in ("bf16w", "bf16o") else "aten"


_SKIP_RE = re.compile(r"skip ([^(;\s]+)\(([^)]*)\)")
_VOUCH_RE = re.compile(r"exact vouch not recorded on (an unnamed stack[^:]*|\S+): (\S+) -> floor (\S+)")   # _select's note when an exact arm is held back to the floor on this stack


def tier_floor(word, rows, *, affine=True, timing="eager", form=None):
    """The tier_floors row (LN_CELLS.json) that sends this tier word to the statement below its min_rows, or None.  Row words are never floored."""
    if word not in TIER_WORDS or rows is None:
        return None
    for r in table().get("tier_floors", {}).get("rows", []):
        sc = r.get("scope", {})
        if word not in sc.get("words", TIER_WORDS):
            continue
        if "affine" in sc and bool(sc["affine"]) != bool(affine):
            continue
        if "timing" in sc and sc["timing"] != timing:
            continue
        if "form" in sc and form is not None and sc["form"] != form:
            continue
        if sc.get("row"):                                                        # a floor on one row's exact word is enforced by the cells (recorded only)
            continue
        if int(rows) < int(r["min_rows"]):
            return r
    return None


def select(cc, dtype, cell, n_tokens, *, word, widen=False, out=None, timing="eager", pass_="fwd", prefer=None, stack=None, capture=False, C=None, has_nvrtc=None, rows=None,
           has_triton=None, has_esm=None, aligned=True, config=None, affine=True):
    """The row + facts for this call (see ``_select``); the decision is recorded ONCE per call class in :mod:`opt_core.cell_census` (pure
    observation: the Selection / Refusal is decided first and returned unchanged)."""
    rows = rows if rows is not None else getattr(cell, "rows", None)                 # a CellWord carries the live row count
    fl = tier_floor(word, rows, affine=affine, timing=timing, form=dtype_word(dtype, widen, out)) if word in TIER_WORDS else None
    if fl is not None:                                                               # an engine's in-model floor: the statement row BY NAME below it
        stock_row = _stock_for(dtype_word(dtype, widen, out), pass_)
        cls, exact_vs, bwd, cap, fb = _row_facts(stock_row)
        sel = Selection(stock_row, None, word, None, False, stack, False, cls, exact_vs, bwd, cap, None, fb,
                        "tier_floor:%s(rows %d < %d; %s)" % (fl["id"], int(rows), int(fl["min_rows"]), "no_affine" if not affine else "scope"), config)
        _census(cc, dtype, cell, n_tokens, dict(word=word, widen=widen, out=out, timing=timing, pass_=pass_, stack=stack, C=C, rows=rows), sel, None)
        return sel
    kw = dict(word=word, rows=rows, widen=widen, out=out, timing=timing, pass_=pass_, prefer=prefer, stack=stack, capture=capture, C=C, has_nvrtc=has_nvrtc,
              has_triton=has_triton, has_esm=has_esm, aligned=aligned, config=config)
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
                if kw.get("rows") is not None:                                   # bucketed by rows: inside = the served cell was measured at >= the call's rows
                    inside = int(kw["rows"]) <= cell_rows(ckey)
                else:
                    inside = int(n_tokens) <= int(str(ckey).split("|")[3][3:])
            except (IndexError, ValueError, TypeError):
                inside = False
        timing = "graph" if (kw.get("capture") or kw.get("timing") == "graph") else "eager"
        shape = str(cell) if cell else "c%s" % (kw.get("C"),)
        key = dict(cc=cc_word(cc), stack=kw.get("stack"), dtype=dtype_word(dtype, kw.get("widen"), kw.get("out")), shape=shape,
                   bucket=_CENSUS.bucket_word(n_tokens, ckey, inside), form="%s.%s" % (kw.get("pass_"), timing), word=kw.get("word"))
        word = kw.get("word")
        if refusal is not None:
            _CENSUS.record("ln", key, "named_fallback", "caller:%s" % (refusal.fallback or "-"), cell_id=None,
                           refused="%s:%s" % (refusal.row or word, refusal.kind), note="raised")
            return
        reason, stack, served = sel.reason or "", kw.get("stack"), arm_word(sel.row, sel.variant)
        if reason.startswith(INHERIT_TOKEN):                                     # an unmeasured capability served from the nearest measured column
            _CENSUS.record("ln", key, "inherited", served, cell_id=ckey, note=reason.split(";")[0])
        elif split_word(word)[0] in ROW_NAMES:
            _CENSUS.record("ln", key, "opt_in", served, cell_id=ckey, note=reason if (ckey is None or not inside) else "")
        elif "skip " in reason:
            m = _SKIP_RE.search(reason)
            _CENSUS.record("ln", key, "named_fallback", served, cell_id=ckey, refused=("%s:%s" % (m.group(1), m.group(2))) if m else "-:%s" % reason)
        elif ckey is None:
            _CENSUS.record("ln", key, "inherited", served, cell_id=None, note="family:none(%s)" % reason)
        elif _census_vouch(ckey, sel, served, reason) is not None:           # EXACT is stack-specific: the cell's exact winner held back BY NAME on this stack, the floor serves (never inherited)
            _CENSUS.record("ln", key, "named_fallback", served, cell_id=ckey, refused=_census_vouch(ckey, sel, served, reason))
        else:
            sib = _CENSUS.sibling_note(sel.stack) if (stack and not sel.stack_measured) else ""   # the cell's timing column is a sibling stack's (same cc): a hit, said so
            if not inside or "not measured (eager cell)" in reason:          # a TRUE gap: a size neighbour (| a timing neighbour, were one served)
                why = "beyond_measured" if not inside else "guard:timing(graph:eager_cell)"
                _CENSUS.record("ln", key, "inherited", served, cell_id=ckey, note=why + ((";" + sib) if sib else ""))
            elif sel.row in STOCK_ROWS:
                _CENSUS.record("ln", key, "stock", served, cell_id=ckey, note=sib)
            else:
                _CENSUS.record("ln", key, "cell_hit", served, cell_id=ckey, note=sib or ("prefer" if "prefer" in reason else ""))
    except Exception:                                                     # the census counts; it never gates or breaks a selection
        return



# ---------------------------------------------------------------- unmeasured capability: inherit the nearest arch-compatible measured column
INHERIT_TOKEN = "inherited_cc:unmeasured"                       # census / reason token: "<token>(<cc>-><column>)"


def measured_columns(family=None):
    """The capability words holding at least one measured cell ({"9.0", "8.0"} today) -- of the given family (dtype word, cell word, timing,
    pass) when named, else of any family."""
    if family is None:
        return sorted({k.split("|")[0] for k in table()["cells"]})
    dtw, cellw, timing, pas = family
    out = set()
    for k in table()["cells"]:
        parts = k.split("|")
        if parts[1] == dtw and parts[2] == str(cellw) and parts[4] == timing and parts[5] == pas:
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
    timing[, pass]) -- the columns considered are those holding at least one cell of that family (any size bucket); without ``family``
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


PORTABLE_ROWS = frozenset(("fastln", "exactln", "ln_rows", "rfd", "dtk_ln", "ln_proj_ln_linear", "ef2_ln_bwd_dx"))
                                                                  # rows whose device code is compiled from source on the running part (Triton; NVRTC for exactln):
                                                                  # they cross to an unmeasured capability.  fast_layernorm_ext (prebuilt per arch list) and the
                                                                  # trunks' extension arm never cross; the ATen statements serve only when no portable row admits.


def _select(cc, dtype, cell, n_tokens, *, word, **kw):
    """The row + facts for this call (see ``_select_measured``).  Inheritance is decided PER FAMILY KEY: when this capability holds no cell
    of the asked family (dtype form, cell word, timing form, pass) -- whether or not it holds cells for other keys -- the tier words
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
    family = (dtype_word(dtype, kw.get("widen", False), kw.get("out")), str(cell), "graph" if (kw.get("capture") or kw.get("timing") == "graph") else "eager", kw.get("pass_", "fwd"))
    col = inherit_column(cc, family)
    if col is None:
        return own
    token = "%s(%s->%s)%s" % (INHERIT_TOKEN, cc_word(cc), col, ";partial_cc" if cc_word(cc) in measured_columns() else "")
    if rw == "exact":
        dtw = dtype_word(dtype, kw.get("widen", False), kw.get("out"))
        pass_ = kw.get("pass_", "fwd")
        stock_row = _stock_for(dtw, pass_)
        last = stock_row if stock_row not in KIT_SIDE_ARMS else _aten_for(dtw, pass_)
        cls, exact_vs, bwd, cap, fb = _row_facts(last)
        return Selection(last, None, word, None, False, None, False, cls, exact_vs, bwd, cap, None, fb,
                         "%s; exact on an unmeasured capability = the statement by name (no byte vouch on %s)" % (token, cc_word(cc)), kw.get("config"))
    kw2 = dict(kw); kw2["stack"] = None                                              # the inherited column's reference stack decides (this part has no stack column there)
    sel = _select_measured(col, dtype, cell, n_tokens, word=word, _portable_first=True, _admit_cc=cc, _exclude=tuple(dead_arms(cc)), **kw2)
    if sel.cell is None:
        return own
    return sel._replace(reason=(token + "; " + (sel.reason or ""))[:400])

def _select_measured(cc, dtype, cell, n_tokens, *, word, widen=False, out=None, timing="eager", pass_="fwd", prefer=None, stack=None, capture=False, C=None, has_nvrtc=None, rows=None,
            has_triton=None, has_esm=None, aligned=True, config=None, _portable_first=False, _admit_cc=None, _exclude=()):
    """The row + facts for this call.  ``cell``: a CELL_WORDS entry (``cell_word`` / ``cell_for_rows``) or None (the form's stock row by name for a
    tier word).  ``word``: a row name (default rule: exactly that row) or a tier word (fast | exact | faithful | big).  ``widen`` + ``out='bf16'``:
    the bf16o form (bf16 output; opt-in).  Pure."""
    if word is None:
        raise ValueError("word is required: a row name (%s) or a tier word (%s)" % (", ".join(ROW_NAMES), ", ".join(TIER_WORDS)))
    timing = "graph" if (capture or timing == "graph") else "eager"
    dtw = dtype_word(dtype, widen, out)
    if C is None and cell:
        try:
            C = int(cell.split("_c")[1].split("_")[0])
        except (IndexError, ValueError):
            C = None
    key, measured, note = (cell_key(cc, dtype, cell, n_tokens, timing=timing, pass_=pass_, widen=widen, out=out, rows=rows) if cell else (None, False, "no cell family for this width"))
    if cell and rows is not None and not measured and C is not None:               # bucketed by rows and beyond THIS card's cells of the named family: another
        alt = None                                                               # family of the same width measured at >= rows on this card / form / timing
        for kind in ("pair", "atom", "dit", "single"):                          # serves (a LayerNorm's cost is rows x width; the family word is bookkeeping)
            w2 = cell_word(kind, C)
            if not w2 or w2 == str(cell):
                continue
            k2, m2, n2 = cell_key(cc, dtype, CellWord(w2, rows), n_tokens, timing=timing, pass_=pass_, widen=widen, out=out, rows=rows)
            if k2 and m2 and (alt is None or cell_rows(k2) < cell_rows(alt[0])):
                alt = (k2, w2, n2)
        if alt is not None:
            key, measured = alt[0], True
            note = "family_by_rows:%s (the %s cells of this card stop below %d rows)" % (alt[1], cell, int(rows))
            cell = CellWord(alt[1], rows)
    tcell = table()["cells"].get(key) if key else None
    layout = "cmajor" if (cell or "").endswith("cmajor") else "rowmajor"
    akw = dict(widen=widen, out=out, pass_=pass_, capture=capture, has_nvrtc=has_nvrtc, has_triton=has_triton, has_esm=has_esm, aligned=aligned, layout=layout,
               stack=(None if stack is None else (None if str(stack).startswith("exact:") else stack)))
    st_measured = bool(tcell and (stack is None or stack in tcell.get("stacks", {})))      # no stack named = the cell's reference stack (measured)
    st = stack if (tcell and stack in tcell.get("stacks", {})) else (tcell.get("ref_stack") if tcell else None)
    rw, variant = split_word(word)
    if rw in ROW_NAMES:
        if rw == "exactln" and variant is None and dtw in ("bf16w", "bf16o"):
            variant = "widen"                                                    # the form implies the variant: bf16 x with fp32 parameters = widen
        if rw == "fastln" and variant is None and dtw == "bf16o":
            variant = "lp"                                                       # the bf16-out form of fastln is its lp kernel
        admits(rw, cc, dtype, C, variant=variant, **akw)
        cls, exact_vs, bwd, cap, fb = _row_facts(rw)
        if dtw in ("bf16w", "bf16o") and fb and fb.split("|")[0] == "aten":
            fb = "aten_autocast"                                                 # the widened forms' statement
        arm = arm_word(rw, variant)
        x = None
        if tcell and st:
            sc = tcell["stacks"][st]
            if arm not in sc.get("ms", {}) and rw in ARM_ALIASES:
                arm = ARM_ALIASES[rw]                                            # the row's numbers were measured under its alias arm
            if sc.get("stock_ms") and sc.get("ms", {}).get(arm):
                x = round(sc["stock_ms"] / sc["ms"][arm], 3)
        mcls = cls if arm == ARM_ALIASES.get(rw) else _measured_class(tcell, arm, st, cls)
        if rw in ARM_ALIASES and tcell and st in (tcell.get("vouched_on", {}).get(rw, ()) or ()):
            mcls = "bitwise:%s" % ARM_ALIASES[rw]                                # vouched bitwise against its alias (the trunks' extension) on this stack
        return Selection(rw, variant, word, key, measured, st, st_measured, mcls, exact_vs, bwd, cap, x, fb,
                         note or ("row word" if tcell else "row word; no cell measured"), config)
    if rw not in TIER_WORDS:
        raise ValueError("unknown word %r (rows: %s; tiers: %s)" % (word, ", ".join(ROW_NAMES), ", ".join(TIER_WORDS)))
    stock_row = _stock_for(dtw, pass_)
    if has_esm and pass_ == "bwd":
        stock_row = "esm_vendored_bwd"
    if tcell is None:                                                            # no cell family lists this width / pass: the statement BY NAME (a named fallback)
        cls, exact_vs, bwd, cap, fb = _row_facts(stock_row)
        return Selection(stock_row, None, word, None, False, None, False, cls, exact_vs, bwd, cap, None, fb,
                         "skip cells(no_family:%s); named stock row: %s" % (str(cell) if cell else "c%s" % C, note), config)
    if "not measured (eager cell)" in (note or ""):                                   # nearest = SIZE only: the timing form is another dimension --
        cls, exact_vs, bwd, cap, fb = _row_facts(stock_row)                              # the statement serves BY NAME (a named fallback; a graph cell is the way to a kernel here)
        return Selection(stock_row, None, word, key, measured, st, st_measured, cls, exact_vs, bwd, cap, None, fb,
                         "skip cells(timing_unmeasured:%s); named stock row (the nearest rule is size-only); %s" % (timing, note), config)
    sc = tcell["stacks"][st]
    ms = sc.get("ms", {})
    winner = (tcell.get("%s_per_stack" % rw, {}).get(st) if rw in ("fast", "exact", "big") else None) or sc.get(rw) or tcell.get(rw) or sc.get("stock_arm") or stock_row
    admissible = [a for a in sorted(ms, key=lambda a: ms[a]) if not a.startswith("x:") and a not in sc.get("outside_band", []) and split_word(a)[0] in ROW_NAMES]
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
        rest = [a for a in order if split_word(a)[0] in STOCK_ROWS and split_word(a)[0] not in KIT_SIDE_ARMS]
        order = port + rest or [stock_row if stock_row not in KIT_SIDE_ARMS else _aten_for(dtw, pass_)]
    chosen, why = None, ""
    if prefer:
        pref = list(prefer)
        for p in pref:
            hit = [a for a in order if a == p or a == arm_word(*split_word(p))] or [a for a in order if split_word(a)[0] == split_word(p)[0] and split_word(p)[1] is None]
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
        why = "prefer" if (chosen == pref[0] or split_word(chosen)[0] == split_word(pref[0])[0]) else "prefer(first measured)"
    else:
        chosen = order[0]
    for a in [chosen] + [x for x in order if x != chosen]:
        r, v = split_word(a)
        if r not in ROW_NAMES:
            continue
        if r in KIT_SIDE_ARMS and not prefer:                                       # an arm no provider row executes under its own name (the trunk's extension object):
            r, v = KIT_SIDE_ARMS[r], None                                            # a tier word serves it through its carried row, or SKIPS to the next measured row --
        try:                                                                         # never the arm's name, never the statement ahead of a measured row
            admits(r, _admit_cc or cc, dtype, C, variant=v, **akw)
        except Refusal as e:
            if _admit_cc is not None and r == "exactln" and e.kind.startswith("cc:") and rw != "exact":
                why = (why + ";" if why else "") + "exactln unproven on %s: serves the %s word (fp32 statistics), not the exact word" % (cc_word(_admit_cc), rw)
                unproven_ok = True
            else:
                why = (why + ";" if why else "") + "skip %s(%s)" % (a if r == split_word(a)[0] else "%s>%s" % (a, r), e.kind)
                unproven_ok = False
            if rw == "exact" and a == (sc.get("stock_arm") or stock_row) and split_word(a)[0] in KIT_SIDE_ARMS:
                aten = _aten_for(dtype_word(dtype, widen, out), pass_)          # the exact floor itself cannot run here: the form's ATen statement BY NAME
                cls, exact_vs, bwd, cap, fb = _row_facts(aten)
                return Selection(aten, None, word, key, measured, st, st_measured, cls, exact_vs, bwd, cap, None, fb,
                                 ("exact floor %s unservable here (%s) -> the ATen statement by name; " % (a, e.kind)) + why + ("; " + note if note else ""), config)
            if not unproven_ok:
                continue
        cls, exact_vs, bwd, cap, fb = _row_facts(r)
        x = round(sc["stock_ms"] / ms[a], 3) if (sc.get("stock_ms") and ms.get(a)) else None
        reason = ("%s winner on %s" % (rw, st)) + ("" if st_measured or not stack else " (stack not measured: reference stack)") + ("; " + why if why else "") + ("; " + note if note else "")
        return Selection(r, v, word, key, measured, st, st_measured, _measured_class(tcell, a, st, cls), exact_vs, bwd, cap, x, fb, reason, config)
    last = stock_row if stock_row not in KIT_SIDE_ARMS else _aten_for(dtype_word(dtype, widen, out), pass_)
    cls, exact_vs, bwd, cap, fb = _row_facts(last)
    return Selection(last, None, word, key, measured, st, st_measured, cls, exact_vs, bwd, cap, None, fb, "no admitted row; named stock row; " + why, config)


def describe(sel):
    """One line for a kit's LEVER / census line."""
    return "ln row=%s%s word=%s cell=%s%s stack=%s%s class=%s%s x_stock=%s%s%s reason=%s" % (
        sel.row, (":" + sel.variant) if sel.variant else "", sel.word, sel.cell, "" if sel.size_measured else "(nearest)", sel.stack,
        "" if sel.stack_measured else "(ref)", sel.cls, (" exact_vs=%s" % sel.exact_vs) if sel.exact_vs else "", sel.x_stock,
        " backward" if sel.backward else "", "" if sel.capture_safe else " CAPTURE-UNSAFE", sel.reason)


def coverage(cc, shapes, *, word="fast", stack=None, capture=False):
    """Census of a kit's shapes: shapes = [(dtype, cell, n_tokens[, pass[, widen]]), ...] -> [{shape, row, cell, measured, x_stock, reason}]."""
    out = []
    for s in shapes:
        dtype, cell, N = s[:3]
        pas = s[3] if len(s) > 3 else "fwd"
        widen = bool(s[4]) if len(s) > 4 else False
        outw = s[5] if len(s) > 5 else None
        try:
            sel = select(cc, dtype, cell, N, word=word, pass_=pas, widen=widen, out=outw, stack=stack, capture=capture)
            out.append({"shape": s, "row": arm_word(sel.row, sel.variant), "cell": sel.cell, "measured": sel.size_measured, "x_stock": sel.x_stock, "reason": sel.reason})
        except Refusal as e:
            out.append({"shape": s, "row": None, "cell": None, "measured": False, "x_stock": None, "reason": "refused:%s->%s" % (e.kind, e.fallback)})
    return out


def rows_for_kit(carried):
    return tuple(r for r in ROW_NAMES if r in carried or r in STOCK_ROWS)


def capture_unsafe(row):
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
    except (RuntimeError, OSError) as e:
        raise Refusal("import:%s(%s:%s)" % (name.split(".")[-1], type(e).__name__, str(e).split("\n")[0][:60].replace(" ", "_")), refusal_row, fallback)


def carried_module(row, variant=None):
    """The module serving a row (imports torch / triton / NVRTC bindings): exactln -> kernels.ln.exactln (variant triton -> .triton_ln),
    fastln -> kernels.ln.fastln.fastln (variant lp -> .lpout), ef2_ln_bwd_dx -> kernels.ln.ef2.ef2_fused_ln, ln_rows / ln_proj_ln_linear -> kernels.ln_proj,
    rfd -> kernels.rfd_layernorm, dtk_ln -> kernels.dtk_kernels."""
    name = {"exactln": ("opt_core.kernels.ln.exactln.triton_ln" if variant == "triton" else "opt_core.kernels.ln.exactln"),
            "fastln": ("opt_core.kernels.ln.fastln.lpout" if variant == "lp" else "opt_core.kernels.ln.fastln.fastln"), "ef2_ln_bwd_dx": "opt_core.kernels.ln.ef2.ef2_fused_ln",
            "ln_rows": "opt_core.kernels.ln_proj", "ln_proj_ln_linear": "opt_core.kernels.ln_proj", "rfd": "opt_core.kernels.rfd_layernorm",
            "dtk_ln": "opt_core.kernels.dtk_kernels", "fast_layernorm_ext": "opt_core.kernels.ln.ext_loader"}.get(row)
    if name is None:
        raise Refusal("no_module(%s)" % row, row, _row_facts(row)[4] if row in ROW_NAMES else None)
    return _imp(name, row, (_row_facts(row)[4] or "").split("|")[0] or None)


def exact_prologue():
    """The exact row's Triton block functions for a fused cell's in-kernel LayerNorm (ATen's arithmetic, bit for bit): a module with
    ``mean_rstd_128(x, eps, RCP3)``, ``mean_rstd_64``, ``affine_rows(x, mean, rstd, g, b, AFFINE)``, ``ln_rows_128``, ``ln_rows_64`` and the standalone
    ``layer_norm_triton`` (the same blocks composed as one LayerNorm, for bit-comparison against torch).  Raises Refusal('import:...') where triton is absent."""
    return carried_module("exactln", "triton")


def stack_word(device=None):
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


def _rows_aligned(x, C):
    esz = x.element_size()
    return x.stride(-1) == 1 and (x.data_ptr() % 16 == 0) and ((x.stride(-2) * esz) % (16 if esz == 4 else 8) == 0 if x.dim() >= 2 else True)


def _stack_for_exact(word, device=None):
    """The running stack word when the EXACT word is asked (its vouch is per stack), else None (tolerance-class words need no stack)."""
    return stack_word(device) if isinstance(word, str) and split_word(word)[0] == "exact" else None


def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5, *, word=None, selection=None, n_tokens=None, cell=None, prefer=None, capture=False,
               out_dtype=None, cache=None):
    """Serve the forward through the selected row: F.layer_norm(x, normalized_shape, weight, bias, eps) semantics over the last dim.  The widen form
    (bf16 x with fp32 weight) is detected from the dtypes; ``out_dtype=torch.bfloat16`` on it selects the bf16o form (bf16 output, fp32 statistics:
    the compiled-extension contract) -- opt-in, nothing changes for callers that do not pass it.  Returns (y, Selection); rows refuse by ``Refusal``."""
    global _WARMED
    if not _WARMED:                                             # first serving call of this face in the process: the stack's heavy libraries imported early + shallow (opt_core.warm)
        _WARMED = True
        from opt_core.warm import auto_warm
        auto_warm("ln")
    import torch
    import torch.nn.functional as F
    C = int(normalized_shape[-1]) if isinstance(normalized_shape, (tuple, list)) else int(normalized_shape)
    rows = x.numel() // C
    widen = (x.dtype == torch.bfloat16 and weight is not None and weight.dtype == torch.float32)
    outw = "bf16" if (widen and out_dtype == torch.bfloat16) else None              # the bf16o form: bf16 x, fp32 parameters, bf16 OUT requested
    cc = torch.cuda.get_device_capability(x.device) if x.is_cuda else (0, 0)
    if selection is None:
        n = n_tokens if n_tokens is not None else int(round(rows ** 0.5))
        cw = cell or cell_for_rows(rows, n, C)
        try:
            import triton  # noqa: F401
            has_triton = True
        except ImportError:
            has_triton = False
        selection = select(cc, x.dtype, cw, n, word=word, widen=widen, out=outw, prefer=prefer, capture=capture, C=C, has_triton=has_triton, aligned=_rows_aligned(x, C),
                           stack=_stack_for_exact(word, x.device), rows=rows)      # bucketed by the call's ROWS
        if (selection.reason or "").startswith(INHERIT_TOKEN) and selection.row not in STOCK_ROWS:
            # an INHERITED portable row on a part the table never measured: its first launch here is guarded -- a build / compile / launch error
            # (anything but out-of-memory) retires the arm for this part (census once) and the NEXT portable row of the donor cell serves, the
            # statement last; never a raw exception out of an inherited row
            from opt_core.oom import is_oom
            for _ in range(len(ROW_NAMES) + 2):
                try:
                    return layer_norm(x, normalized_shape, weight, bias, eps, word=word, selection=selection, n_tokens=n_tokens, cell=cell, prefer=prefer,
                                      capture=capture, out_dtype=out_dtype, cache=cache)
                except Refusal:
                    raise
                except Exception as e:                                        # noqa: BLE001
                    if is_oom(e):
                        raise
                    mark_dead(cc, arm_word(selection.row, selection.variant), e, cell=cw, dtype=x.dtype, n_tokens=n,
                              kw=dict(word=word, widen=widen, out=outw, timing="graph" if capture else "eager", pass_="fwd", stack=None, C=C, rows=rows))
                    selection = select(cc, x.dtype, cw, n, word=word, widen=widen, out=outw, prefer=prefer, capture=capture, C=C, has_triton=has_triton,
                                       aligned=_rows_aligned(x, C), stack=_stack_for_exact(word, x.device), rows=rows)
                    if selection.row in STOCK_ROWS or not (selection.reason or "").startswith(INHERIT_TOKEN):
                        return layer_norm(x, normalized_shape, weight, bias, eps, word=word, selection=selection, n_tokens=n_tokens, cell=cell, prefer=prefer,
                                          capture=capture, out_dtype=out_dtype, cache=cache)
    row, variant = selection.row, selection.variant
    if cache is not None:
        cache["_last"] = selection
    if row == "exactln":
        if variant == "triton":
            T = carried_module("exactln", "triton")
            y = T.layer_norm_triton(x.reshape(rows, C), weight, bias, eps, out_dtype=out_dtype or torch.float32)
            return y.reshape(x.shape), selection
        E = carried_module("exactln")
        try:
            y = E.layer_norm(x, (C,), weight, bias, eps, widen=(variant == "widen" or widen), out_dtype=out_dtype)
        except E.Unsupported as e:
            raise Refusal("exactln:%s" % str(e)[:60].replace(" ", "_"), row, "aten_autocast" if widen else "aten")
        except (ImportError, OSError, RuntimeError) as e:                       # NVRTC bindings absent / driver: named, the stock row is the fallback
            raise Refusal("exactln:%s:%s" % (type(e).__name__, str(e)[:50].replace(" ", "_")), row, "aten_autocast" if widen else "aten")
        return y, selection
    if row == "fastln":
        if variant == "lp":
            LP = carried_module("fastln", "lp")
            od = out_dtype or (torch.bfloat16 if outw else (torch.float32 if widen else x.dtype))
            return LP.layer_norm_lp(x, (C,), weight, bias, eps, out_dtype=od), selection
        FL = carried_module("fastln")
        return FL.fast_layer_norm_triton(x, (C,), weight, bias, eps), selection
    if row == "fast_layernorm_ext":
        S = carried_module("fast_layernorm_ext")
        try:
            y = S.layer_norm(x, (C,), weight, bias, eps)
        except S.Unavailable as e:                                               # no prebuilt for this torch key / the binary does not load here: named
            raise Refusal("fast_layernorm_ext:%s" % e.kind[:80], row, "fastln:lp" if outw else ("aten_autocast" if widen else "aten"))
        return (y.to(out_dtype) if (out_dtype is not None and y.dtype != out_dtype) else y), selection
    if row == "ln_rows":
        LP = carried_module("ln_rows")
        try:
            y = LP.layernorm_rows(x.reshape(rows, C), weight, bias, eps)
        except LP.Unsupported as e:
            raise Refusal("ln_proj:%s" % getattr(e, "reason", str(e))[:60].replace(" ", "_"), row, "aten")
        return y.reshape(x.shape), selection
    if row == "rfd":
        R = carried_module("rfd")
        return R.triton_layer_norm(x, (C,), weight, bias, eps), selection
    if row == "dtk_ln":
        D = carried_module("dtk_ln")
        y = D.ln_modulate(x.reshape(rows, C), None, None, weight=weight, bias=bias, eps=eps, out_dtype=out_dtype)
        return y.reshape(x.shape), selection
    if row == "aten":
        return F.layer_norm(x, (C,), weight, bias, eps), selection
    if row == "aten_autocast":
        y = F.layer_norm(x.float(), (C,), None if weight is None else weight.float(), None if bias is None else bias.float(), eps)
        return (y.to(out_dtype) if out_dtype is not None else y), selection
    if row == "fast_layernorm":
        raise Refusal("named_row:fast_layernorm (a trunk's compiled extension; capture-unsafe; not carried)", row, "aten")
    raise Refusal("no_forward_serving_branch:%s" % row, row, "aten")


def layer_norm_backward_dx(grad_y, x, weight, mean, rstd, *, residual_grad=None, word=None, selection=None, n_tokens=None, layout="rowmajor", capture=False,
                           vendored=None):
    """dx of LayerNorm for frozen gamma/beta: grad_y, x [M, D] (row-major views), mean / rstd [M] fp32, residual_grad [M, D] | None folded in.
    Rows: ef2_ln_bwd_dx (D 256) | esm_vendored_bwd (``vendored``: the caller's helper with the (grad_y, x, w, mean, rstd, layout_int, resid, M, D)
    signature) | aten_bwd (autograd through F.layer_norm recomputed).  Returns (dx, Selection)."""
    import torch
    M, D = x.shape[-2], x.shape[-1]
    cc = torch.cuda.get_device_capability(x.device) if x.is_cuda else (0, 0)
    if selection is None:
        n = n_tokens if n_tokens is not None else int(round(M ** 0.5))
        cw = cell_word("pairbwd", D, ("rowmajor_res" if residual_grad is not None else "rowmajor") if layout == "rowmajor" else "cmajor")
        selection = select(cc, x.dtype, cw, n, word=word, pass_="bwd", capture=capture, C=D, has_esm=(vendored is not None), stack=_stack_for_exact(word, x.device))
    row = selection.row
    if row == "ef2_ln_bwd_dx":
        E = carried_module("ef2_ln_bwd_dx")
        dx = E.ln_bwd_dx(grad_y, x, weight, mean, rstd, 1 if layout == "cmajor" else 0, residual_grad, M, D)
        return dx, selection
    if row == "esm_vendored_bwd":
        if vendored is None:
            raise Refusal("vendored helper not given", row, "aten_bwd")
        return vendored(grad_y, x, weight, mean, rstd, 1 if layout == "cmajor" else 0, residual_grad, M, D), selection
    if row == "aten_bwd":
        xx = x.detach().float().requires_grad_(True)
        y = torch.nn.functional.layer_norm(xx, (D,), None if weight is None else weight.float(), None, 1e-5)
        (dx,) = torch.autograd.grad(y, xx, grad_y.float())
        if residual_grad is not None:
            dx = dx + residual_grad.float()
        return dx.to(x.dtype), selection
    raise Refusal("no_backward_serving_branch:%s" % row, row, "aten_bwd")


def ln_linear(x2d, ln_weight, ln_bias, weight, bias=None, eps=1e-5, *, word=None, selection=None, n_tokens=None, act="none", cache=None):
    """LN -> wide Linear (the lnlinear cell): rows ln_proj_ln_linear | torch_ln_linear1.  Returns (y [M, nout], Selection)."""
    import torch
    import torch.nn.functional as F
    M, C = x2d.shape
    cc = torch.cuda.get_device_capability(x2d.device) if x2d.is_cuda else (0, 0)
    if selection is None:
        selection = select(cc, x2d.dtype, cell_word("lnlinear", C), n_tokens if n_tokens is not None else M, word=word, C=C, stack=_stack_for_exact(word, x2d.device))
    row = selection.row
    if row == "ln_proj_ln_linear":
        LP = carried_module("ln_proj_ln_linear")
        from ..apb import pack_cached                                              # one cache rule for both faces: identity + version of every folded tensor
        try:
            packed = pack_cached(cache, "ln_linear_packed", (weight, ln_weight, ln_bias, bias), (float(eps), str(x2d.device)),
                                 lambda: LP.pack_ln_linear_weights(ln_weight, ln_bias, weight, bias, eps, x2d.device))
        except LP.Unsupported as e:
            raise Refusal("ln_proj:%s" % getattr(e, "reason", str(e))[:60].replace(" ", "_"), row, "torch_ln_linear1")
        try:
            return LP.ln_linear(x2d, packed, act=act), selection
        except LP.Unsupported as e:
            raise Refusal("ln_proj:%s" % getattr(e, "reason", str(e))[:60].replace(" ", "_"), row, "torch_ln_linear1")
    if row == "torch_ln_linear1":
        y = F.linear(F.layer_norm(x2d, (C,), ln_weight.to(x2d.dtype), ln_bias.to(x2d.dtype), eps), weight.to(x2d.dtype), None if bias is None else bias.to(x2d.dtype))
        if act == "silu":
            y = F.silu(y)
        elif act == "sigmoid":
            y = torch.sigmoid(y)
        return y, selection
    raise Refusal("no_lnlinear_serving_branch:%s" % row, row, "torch_ln_linear1")


def reference(x, normalized_shape, weight=None, bias=None, eps=1e-5, dtype=None):
    """F.layer_norm evaluated in ``dtype`` (float64 by default) -- the numerics yardstick."""
    import torch
    import torch.nn.functional as F
    dt = dtype or torch.float64
    C = int(normalized_shape[-1]) if isinstance(normalized_shape, (tuple, list)) else int(normalized_shape)
    return F.layer_norm(x.to(dt), (C,), None if weight is None else weight.to(dt), None if bias is None else bias.to(dt), eps)

_WARMED = False                                                 # opt_core.warm.auto_warm() ran from this face's first serving call
