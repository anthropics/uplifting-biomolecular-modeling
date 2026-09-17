"""Transition family: ONE provider over every carried implementation ("row") of the trunk transition and the measured cell table
(TRANSITION_CELLS.json).

The op (the AF3-family SwiGLU transition of the pair / single / template / MSA stacks, and the ESM-family pair transition)::

    forward    y = W_o( silu(W_a n) * (W_b n) )   with n = LayerNorm(x) over the last dimension of a [rows, c] tensor, hidden = factor * c;
               optionally  y = y * mask[:, None]  (the row mask)  and  y = x + y  (the residual, folded in the kernel epilogue)
    backward   dX of the same op for FROZEN weights (the design loop differentiates the trunk; dW / dgamma / dbeta are not produced)

Rows (``ROW_NAMES``; ``TRANSITION_CELLS.json["rows"]`` states each one's envelope, numerics class, capture safety and named fallback):

    v2            kernels/fpf_transition_v2 (carried package, in place): the whole op in ONE Triton kernel, mask + residual folded   fast; fwd
    v1            kernels/fpf_transition (carried package, in place) through attn.pair_fused: LN given (v1) | LN in prologue (v1:lnfused) fast; fwd
    pf            attn.pair_fused.transition(impl='fpf'|'lnl') exactly as the AF3-family kits bind it today (pf:fpf | pf:lnl)      fast; fwd
    lnl           kernels/lnl_fused.fused_transition                                                                               fast; fwd
    af3_fused     .af3.af3_fused -- carried module (c in 64/128/256/384, non-power-of-two c padded)                                  fast; fwd
    flash_sm90a   .flash_sm90a -- carried SEALED package (sm_90a extension per stack, manifest-checked): (c=256, hidden=1024)        exact class where a cell records bitwise, else fast; fwd
    esm_t15       .esm.ef2_pair_v2 -- carried module of the ESM-family kit (imports that stack's fork; refused by name elsewhere)    fast; fwd, residual folded
    esm_t16       .esm.ef2_t16_transition -- carried module + sm_90a cubin (CUDA C++ / CuTe persistent kernel, driver-loaded)       fast; fwd, cc 9.0
    esm_kd3       .esm.ef2_autograd_kernels -- carried module (same import rule): TransitionRefround, variant esm_kd3:lean            fast; fwd AND bwd (dX)
    esm_t16_kd3   K-D3 lean with its forward on esm_t16's kernel (the design kit's own composition, through the carried hook)         fast; fwd AND bwd (dX)
    esm_t15_kd3   K-D3 lean with its forward on esm_t15's kernel (the same hook; the cc 8.0 pairing, t16 being sm_90a only)          fast; fwd AND bwd (dX)
    esm_fused_exact .esm_fused -- this package's one-kernel ESM-family FUSED INFERENCE statement (the fork's tree statistics, bf16 x_hat
                  chain, one-rounding SwiGLU, chained W3 accumulation, round-then-add residual): (c=256, hidden % 64 == 0), cc 9.0       exact class (bitwise vs that statement where a cell records it); fwd
    rowpair       mem.rowpair.transition.transition_rows: the row-block SCHEDULE of a sharded pair layout over any forward row       schedule (the memory tier's lever)
    (v1:liger    = v1 with the Liger-Kernel SiLU*b epilogue -- silu(fp32 a) * fp32 b rounded once -- for engines whose statement is that kernel; cells '<cell>+liger', select(form='liger'))
    torch_swiglu  the statement (cuBLAS; bf16 autocast form for bf16 cells)                                                         STOCK; differentiable
    engine_module the engine's own module call (measured for the record; never constructed here)                                     STOCK
    compile       torch.compile of the statement (stock family; capture_safe=false)                                                  STOCK family

Selection (``select``): the DEFAULT RULE -- a row word serves exactly that row with its own launch cell or raises ``Refusal`` BY NAME carrying the
fallback row (never a silent substitute); the tier words ``fast | faithful | exact | big`` are OPT-IN and resolve to the cell's measured winner
for the caller's stack (EXACT FLOOR x1.00: a row is a cell's exact winner only where measured bitwise AND not slower than the stock arm; else the
stock arm by name).  Nothing here changes what an existing caller of kernels.fpf_transition / kernels.fpf_transition_v2 / kernels.lnl_fused /
attn.pair_fused / mem.rowpair executes: those modules are wrapped in place, unedited.

This file imports nothing but the standard library at import time; torch / triton / the carried modules are imported inside the serving calls.
"""
import json
import os
import re

from ... import cell_census as _CENSUS                    # stdlib-only: the coverage census (one record per decided call class; pure observation)

__all__ = ["pack_residency", "pack_policy", "pack_cache_stats", "pack_cache_clear", "pack_nbytes", "PACK_CACHE_ENV", "PACK_LRU_ENV", "prime", "NEEDS_PRIME", "ROW_NAMES", "TIER_WORDS", "STOCK_ROWS", "EXACT_CLASS_ROWS", "BACKWARD_ROWS", "FORWARD_ONLY_ROWS", "CAPTURE_UNSAFE_ROWS", "SCHEDULE_ROWS",
           "CELL_WORDS", "PASS_WORDS", "Refusal", "Selection", "table", "rows", "describe", "coverage", "exact_floor", "split_word", "cell_word", "cell_key",
           "stack_word", "norm_stack", "select", "admits", "FORMS", "af3_pin", "af3_pin_key", "af3_pinned_buckets", "af3_rows_bucket", "inherit_cc", "measured_ccs", "portable_row", "_measured_v2_substitute", "flash_abi_key", "flash_store_manifest", "cfg_of", "carried_module", "pack", "transition", "transition_autograd", "reference", "rows_for_kit", "Weights"]

ROW_NAMES = ("v2", "v1", "pf", "lnl", "af3_fused", "flash_sm90a", "esm_t15", "esm_t16", "esm_kd3", "esm_t16_kd3", "esm_t15_kd3", "esm_fused_exact",
             "rowpair", "torch_swiglu", "engine_module", "compile")
TIER_WORDS = ("fast", "faithful", "exact", "big")
STOCK_ROWS = ("torch_swiglu", "engine_module", "compile")
EXACT_CLASS_ROWS = ("flash_sm90a", "esm_fused_exact")  # exact CLASS by construction; an exact-tier WINNER only where a cell records bitwise (EXACT FLOOR x1.00)
EXACT_GIVEN_LN_ROWS = ("flash_sm90a", "v1")                  # exact class only given the caller's own LayerNorm output (x_ln); esm_fused_exact carries its statement's LayerNorm
BACKWARD_ROWS = ("esm_kd3", "esm_t16_kd3", "esm_t15_kd3", "torch_swiglu", "engine_module", "compile")
FORWARD_ONLY_ROWS = ("v2", "v1", "pf", "lnl", "af3_fused", "flash_sm90a", "esm_t15", "esm_t16", "esm_fused_exact")
CAPTURE_UNSAFE_ROWS = ("compile",)
WARM_BEFORE_CAPTURE_ROWS = ("af3_fused",)               # autotunes at the first call of a shape bucket: one eager call before capture
SCHEDULE_ROWS = ("rowpair",)
ESM_ROWS = ("esm_t15", "esm_t16", "esm_kd3", "esm_t16_kd3", "esm_t15_kd3")   # the ESM-family rows (c = 256); all but esm_t16 import that stack's transformers fork (refused by name elsewhere)
ESM_FORK_ROWS = ("esm_t15", "esm_kd3", "esm_t16_kd3", "esm_t15_kd3")
T16_CELL = (256, 1024)
ESM_FUSED_C, ESM_FUSED_HIDDEN_STEP = 256, 64                # esm_fused_exact: four resident 64-wide K chunks; the interleaved a|b chunk is 2 x 32 hidden columns
ESM_FUSED_CCS = ("9.0", "8.0")                              # the cards with a measured launch row (bitwise against the statement there), each with its measured epilogue
CELL_WORDS = ("pair_c128_n4", "pair_c256_n4", "pair_c128_n2", "pair_c384_n4", "pair_c256_n2", "single_c384_n4", "single_c768_n4", "single_c384_n2", "rows_c64_n4", "rows_c64_n2", "esmpair_c256_n4",
              "pair_c128_n4+liger", "pair_c128_n2+liger", "rows_c64_n4+liger", "rows_c64_n2+liger", "single_c384_n4+liger",    # '+liger': the same shapes under the Liger SiLU*b form of the statement
              "esmpair_c256_n4+esmfused")                                              # '+esmfused': the ESM-family pair shape under its fork's FUSED INFERENCE statement
FORMS = ("swiglu", "liger", "esmfused")                 # 'esmfused' = the ESM-family fork's fused inference arithmetic end to end (bf16 tree statistics, bf16
                                                        # x_hat chain, silu(fp32 a) * fp32 b rounded once, cuBLAS addmm residual = product rounded then added): its own cells; the statement's SiLU*b arithmetic: 'swiglu' = F.silu(a) * b (bf16 silu output, bf16 product: two roundings);
                                                        # 'liger' = the Liger-Kernel SiLU*b (silu of fp32 a times fp32 b, ONE rounding): a different statement, its own cells
PASS_WORDS = ("fwd", "fwdbwd")
VARIANTS = {"v2": ("fast",), "v1": ("lnfused", "liger"), "pf": ("fpf", "lnl"), "lnl": ("res",), "af3_fused": ("res",), "flash_sm90a": ("kernel_ln",),
            "esm_kd3": ("lean",)}
V2_PACKAGE_CELLS = ((128, 512), (64, 128), (64, 256), (256, 1024), (384, 1536))   # (c, hidden) the carried package launches with its own sm_90 configuration
V2_ROW_LIMIT = 2 ** 31 - 2 ** 12                       # the kernel's row index is int32 (row offsets are int64): more rows are refused by name
AF3_FUSED_C = (64, 128, 256, 384)
V1_C = (64, 128, 256, 384)                              # v1 / pf: widths with a measured pair-fused launch (c 64 is refused further in by the cell table where unmeasured)
LNL_C = (64, 128, 256)                                  # c=384: the tile does not build on the measured stacks (refused by name, measured)
FLASH_CELL = (256, 1024)
ESM_C = 256
_CFG_RE = re.compile(r"^bm(\d+)bh(\d+)w(\d+)s(\d+)il([01])$")
_HERE = os.path.dirname(os.path.abspath(__file__))
_TABLE = {"t": None}


class Refusal(Exception):
    """Raised BY NAME when a word cannot serve the call: ``kind`` (machine word), ``row`` (the word asked), ``fallback`` (the row that serves the
    call by name -- a stock row or another carried row; the caller binds it explicitly, nothing substitutes silently)."""

    def __init__(self, kind, row=None, fallback="torch_swiglu", detail=""):
        super().__init__("%s (row=%s, fallback=%s)%s" % (kind, row, fallback, (": " + detail) if detail else ""))
        self.kind, self.row, self.fallback, self.detail = kind, row, fallback, detail


class Selection(object):
    """What ``select`` resolved: ``row`` + ``variant`` + ``cfg`` (the launch), the ``word`` asked, the ``cell_key`` it was read from (None for a
    plain row word outside the table), ``tier`` (None | fast | faithful | exact | big), the measured ``ms`` / ``x_stock`` / ``cls`` on the
    ``stack`` it was measured on, ``stack_measured`` (False: the reference stack's winner served on another stack), ``beyond_measured`` (n_tokens
    above the largest measured size), ``words`` (named conditions: warm_before_capture, ln_given_for_exact, ...)."""
    __slots__ = ("row", "variant", "cfg", "word", "cell_key", "tier", "ms", "x_stock", "cls", "stack", "stack_measured", "beyond_measured",
                 "fallback", "words", "backward", "capture_safe")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))
        if self.words is None:
            self.words = ()

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}

    def line(self):
        """One ACTIVE-style line for a kit's log."""
        w = self.row + ((":" + self.variant) if self.variant else "") + (("@" + cfg_word(self.cfg)) if (self.cfg and self.row == "v2") else "")
        return "transition:%s word=%s tier=%s cell=%s x_stock=%s cls=%s stack_measured=%s%s" % (
            w, self.word, self.tier, self.cell_key, self.x_stock, self.cls, self.stack_measured, (" " + ",".join(self.words)) if self.words else "")

    def __repr__(self):
        return "Selection(%s)" % self.line()


# --------------------------------------------------------------------------------------------------------------------------------- the table
def table():
    """TRANSITION_CELLS.json parsed once (schema transition_cells/v1)."""
    if _TABLE["t"] is None:
        with open(os.path.join(_HERE, "TRANSITION_CELLS.json")) as f:
            _TABLE["t"] = json.load(f)
    return _TABLE["t"]


def rows():
    return table()["rows"]


def describe():
    t = table()
    return {"schema": t["schema"], "rows": list(ROW_NAMES), "tiers": list(TIER_WORDS), "cells": len(t["cells"]), "stacks": list(t["stacks"]),
            "stock_rows": list(STOCK_ROWS), "backward_rows": list(BACKWARD_ROWS), "capture_unsafe_rows": list(CAPTURE_UNSAFE_ROWS),
            "default_rule": t["default_rule"], "exact_rule": t["exact_rule"]}


def coverage():
    return table()["coverage"]


def exact_floor(key, stack):
    """The smallest n_tokens from which the EXACT tier's kernel arm of cell ``key``'s family is bitwise-vouched on ``stack`` (every vouch measurement at or
    above it on that stack was bitwise equal to the statement), or None (no vouch on that stack).  table()['coverage'][family]['exact_floor'][stack]."""
    cc, dt, cw, nb, timing, direction = key.split("|")
    fam = "%s|%s|%s|%s|%s" % (cc, dt, cw, timing, direction)
    v = (table()["coverage"].get(fam, {}).get("exact_floor") or {}).get(stack)
    return int(v) if v is not None else None


def split_word(word):
    """'v2:fast@bm64bh64w4s2il1' -> ('v2', 'fast', 'bm64bh64w4s2il1'); 'pf:fpf' -> ('pf', 'fpf', None); 'lnl' -> ('lnl', None, None)."""
    w = str(word)
    cfgw = None
    if "@" in w:
        w, cfgw = w.split("@", 1)
    row, var = (w.split(":", 1) + [None])[:2] if ":" in w else (w, None)
    return row, var, cfgw


def cfg_of(cfgword):
    """'bm64bh64w4s2il1' -> the v2 launch dict {'BM': 64, 'BH': 64, 'num_warps': 4, 'num_stages': 2, 'IL': 1}; None -> None."""
    if cfgword is None:
        return None
    m = _CFG_RE.match(str(cfgword))
    if not m:
        raise Refusal("cfg_word:%s" % cfgword, "v2", "torch_swiglu", "v2@bm<BM>bh<BH>w<warps>s<stages>il<0|1>")
    bm, bh, w, s, il = (int(g) for g in m.groups())
    return {"BM": bm, "BH": bh, "num_warps": w, "num_stages": s, "IL": il}


def cfg_word(cfg):
    if not cfg:
        return None
    return "bm%dbh%dw%ds%dil%d" % (cfg["BM"], cfg["BH"], cfg["num_warps"], cfg["num_stages"], cfg.get("IL", 0))


def cell_word(c, hidden, family="pair", form="swiglu"):
    """The table's cell word for (c, hidden) of a row family: 'pair' (N^2 rows) | 'single' (N rows) | 'rows' (N^2 rows of width 64) | 'esmpair'
    (the ESM-family pair transition, residual folded), suffixed '+liger' under form='liger' (the Liger SiLU*b statement) or '+esmfused' under
    form='esmfused' (the ESM-family fork's FUSED INFERENCE statement: bf16 tree statistics, bf16 x_hat chain, one-rounding SwiGLU kernel, cuBLAS
    addmm residual -- the inference engine's exact-tier form); None when the table has no such cell (select then serves row words only)."""
    c, hidden = int(c), int(hidden)
    if form not in FORMS:
        return None
    if form != "swiglu":                                # '<base>+<form>': the same shape under another statement form, where the table has such cells
        base = cell_word(c, hidden, family, "swiglu")
        w = None if base is None else base + "+" + form
        return w if w in CELL_WORDS else None
    if family == "esmpair":
        return "esmpair_c256_n4" if (c, hidden) == (256, 1024) else None
    if family == "single":
        return {(384, 1536): "single_c384_n4", (768, 3072): "single_c768_n4", (384, 768): "single_c384_n2"}.get((c, hidden))
    if family == "rows" or c == 64:
        return {(64, 256): "rows_c64_n4", (64, 128): "rows_c64_n2"}.get((c, hidden))
    w = "pair_c%d_n%d" % (c, hidden // c) if hidden % c == 0 else None
    return w if w in CELL_WORDS else None


def cell_dims(cellw):
    """(c, hidden) of a cell word ('+liger' words share their base word's shape)."""
    base = cellw.split("+")[0]
    fam, cpart, npart = base.split("_")
    c = int(cpart[1:]); n = int(npart[1:])
    return c, c * n


def norm_stack(stack):
    """The table's stack key for a caller's stack word: '<H100|A100>:torch<torch version>/<triton version>'.  Accepts the longer words other providers key on
    ('.../cueq0.10.0' library suffixes are dropped: this family does not depend on them) and card aliases ('A100-80GB', 'A100-SXM4-80GB', 'NVIDIA H100 80GB HBM3',
    'H100-SXM'); anything unparsable is returned unchanged (then it simply matches no measured stack)."""
    if not isinstance(stack, str) or ":" not in stack:
        return stack
    card, _, rest = stack.partition(":")
    cu = card.upper()
    card = ("A100_40" if ("A100_40" in cu or ("A100" in cu and "40GB" in cu)) else "H200" if "H200" in cu else "H100" if "H100" in cu else "A100" if "A100" in cu
            else "cc100" if cu in ("B200", "CC100") or cu.startswith("B200") else "cc103" if cu in ("B300", "CC103") or cu.startswith("B300") else card)   # part-labelled columns stay addressable by an explicit word
    parts = rest.split("/")
    if len(parts) >= 2:
        rest = parts[0] + "/" + parts[1]
    return card + ":" + rest


def stack_word(device=None):
    """'<card>:torch<version>/<triton>' of the running process ('H100' for cc 9.x parts, 'A100' for cc 8.0) -- the table's stack keys."""
    import torch
    try:
        import triton
        tv = triton.__version__
    except ImportError:
        tv = "none"
    cc = torch.cuda.get_device_capability(device) if torch.cuda.is_available() else (0, 0)
    card = {9: "H100", 8: "A100"}.get(cc[0], "cc%d%d" % cc)
    return "%s:torch%s/%s" % (card, torch.__version__, tv)


def _cc_word(cc):
    if isinstance(cc, str):
        return cc
    if isinstance(cc, (tuple, list)):
        return "%d.%d" % (int(cc[0]), int(cc[1]))
    return "%.1f" % float(cc)


NON_PORTABLE_ROWS = {"flash_sm90a": ("9.0",), "esm_t16": ("9.0",), "esm_t16_kd3": ("9.0",)}   # rows whose device code is an arch-specific binary (sm_90a cubin / .so):
INHERIT_MIN_CC = 8.0                                                                                # served on their own capability only, never inherited by another part


def measured_ccs():
    """The capability columns this table carries cells for ('9.0', '8.0', ... as measured)."""
    return tuple(sorted({k.split("|")[0] for k in table()["cells"]}, key=lambda w: float(w)))


def inherit_cc(cc, dtype=None, cellw=None, timing="eager", direction="fwd"):
    """The measured column a capability inherits a family's tolerance-tier words from when it has NO cell for that family key itself (decided per
    (capability, dtype, cell word, timing, direction) -- a capability that owns cells for OTHER keys still inherits the keys it lacks): the highest
    measured capability not above it that carries the family (same or older architecture: its Triton launch words build there); for a part below
    every measured column nothing (and nothing below capability %.1f: no bf16 tensor-core rows there).  With no family named: the nearest column
    by capability alone.  E.g. 10.0 / 10.3 / 12.0 -> 9.0, 8.6 / 8.9 -> 8.0; None when the capability has the key itself.""" % INHERIT_MIN_CC
    ccw = _cc_word(cc)
    try:
        v = float(ccw)
    except (TypeError, ValueError):
        return None
    if v < INHERIT_MIN_CC:
        return None
    if cellw is None:
        cols = list(measured_ccs())
    else:
        cov = table()["coverage"]
        cols = sorted({f.split("|")[0] for f in cov if f.split("|")[1:] == [dtype, cellw, timing, direction]}, key=float)
    if ccw in cols or not cols:
        return None
    below = [m for m in cols if float(m) <= v]
    return max(below, key=float) if below else None


def portable_row(row, cc, *, dtype="bf16", c=None, hidden=None, residual=False):
    """True when the row's device code is source-compiled (Triton) and may serve capability cc by inheritance; arch-specific binaries
    (NON_PORTABLE_ROWS) and af3_fused without a launch pin recorded for cc are not."""
    ccw = _cc_word(cc)
    if row in NON_PORTABLE_ROWS:
        return ccw in NON_PORTABLE_ROWS[row]
    if row == "esm_fused_exact":
        return ccw in ESM_FUSED_CCS
    if row == "af3_fused":
        return bool(c is not None and af3_pinned_buckets(ccw, dtype, c, hidden, residual))
    if row in STOCK_ROWS or row in SCHEDULE_ROWS:
        return False
    return True


_DEAD_INHERITED = set()          # (cc, dtype, c, hidden, arm word): an inherited row that failed to build / launch on this part in this process (never retried)


def _dead_key(cc, dtype, c, hidden, arm):
    return (_cc_word(cc), str(dtype), int(c), int(hidden), str(arm))


def _measured_v2_substitute(sc, arm, *, c, hidden, dtype, cc, direction, residual, mask, rows_count, ln_given):
    """For a measured column off cc 9.0 whose tier word is a bare v2 word (no launch configuration): the column's fastest MEASURED admitted `v2[...]@cfg` word
    (the same numerics variant first, then any), else its fastest other measured admitted kernel row, else None (the caller names the statement)."""
    ms = sc.get("ms") or {}
    row0, var0, _ = split_word(arm)
    outside = set(sc.get("outside_band") or ())
    def ok_(a):
        if a.startswith("x:") or ms.get(a) is None or a in outside:
            return False
        r_, v_, cf_ = split_word(a)
        if r_ in STOCK_ROWS or r_ == "esm_fused_exact":
            return False
        if r_ == "v2" and cf_ is None:
            return False
        good, _w = admits(a, c=c, hidden=hidden, dtype=dtype, cc=cc, direction=direction, residual=residual, mask=mask, rows_count=rows_count, ln_given=ln_given)
        return good
    cands = sorted((a for a in ms if ok_(a)), key=lambda a: ms[a])
    same = [a for a in cands if split_word(a)[0] == "v2" and split_word(a)[1] == var0]
    anyv2 = [a for a in cands if split_word(a)[0] == "v2"]
    for pool in (same, anyv2, cands):
        if pool:
            return pool[0]
    return None


def _inherited_arm(tier_arm, sc, *, cc_real, cc_from, c, hidden, dtype, direction, timing, residual, mask, rows_count, ln_given, family):
    """For a capability served by inheritance: the tier's own arm when its row is portable and admitted, else the fastest portable admitted arm
    of the inherited cell (v2 words without a launch configuration take the cell's fastest measured v2 launch word: the package configures
    cc 9.0 only), else None (the statement)."""
    ms = sc.get("ms") or {}
    outside = set(sc.get("outside_band") or ())
    order = [tier_arm] + [a for _, a in sorted((m_, a_) for a_, m_ in ms.items() if a_ != tier_arm and m_ is not None)]
    for a in order:
        if a is None or a.startswith("x:") or a in outside or (ms.get(a) is None and a != tier_arm):
            continue
        if _dead_key(cc_real, dtype, c, hidden, a) in _DEAD_INHERITED:      # stepped aside earlier in this process (build / launch failure on this part)
            continue
        row, var, cfgw = split_word(a)
        if row in STOCK_ROWS or row in SCHEDULE_ROWS or (timing == "graph" and row in CAPTURE_UNSAFE_ROWS) or (direction == "fwdbwd" and row not in BACKWARD_ROWS):
            continue
        if not portable_row(row, cc_real, dtype=dtype, c=c, hidden=hidden, residual=residual or family == "esmpair"):
            continue
        if row == "v2" and cfgw is None:                            # the package's own configuration is cc 9.0's: name the inherited cell's fastest measured launch word
            v2s = sorted((m_, b_) for b_, m_ in ms.items() if b_.startswith("v2@") and m_ is not None and b_ not in outside)
            if v2s:
                cfgw = split_word(v2s[0][1])[2]
            elif (int(c), int(hidden)) not in V2_PACKAGE_CELLS:
                continue
        cand = _arm_word(row, var, cfgw)
        if _dead_key(cc_real, dtype, c, hidden, cand) in _DEAD_INHERITED:
            continue
        ok, _why = admits(cand, c=c, hidden=hidden, dtype=dtype, cc=cc_from, direction=direction, residual=residual or family == "esmpair", mask=mask,
                          rows_count=rows_count, ln_given=ln_given)
        if ok:
            return cand, a
    return None, None


def cell_key(cc, dtype, cellw, n_tokens, timing="eager", direction="fwd"):
    """-> (key, measured_size, beyond_measured): the smallest measured size >= n_tokens of the family (cc|dtype|cell|timing|direction), else the
    largest measured size with beyond_measured=True; (None, None, False) when the family is not in the table."""
    fam = "%s|%s|%s|%s|%s" % (_cc_word(cc), dtype, cellw, timing, direction)
    cov = table()["coverage"].get(fam)
    if not cov:
        return None, None, False
    sizes = cov["sizes"]
    for s in sizes:
        if int(n_tokens) <= s:
            return "%s|%s|%s|N<=%d|%s|%s" % (_cc_word(cc), dtype, cellw, s, timing, direction), s, False
    s = sizes[-1]
    return "%s|%s|%s|N<=%d|%s|%s" % (_cc_word(cc), dtype, cellw, s, timing, direction), s, True


def admits(word, *, c, hidden, dtype="bf16", cc="9.0", direction="fwd", residual=False, mask=False, rows_count=None, ln_given=False):
    """Static envelope of a row word for a call shape -> (True, '') or (False, '<reason>').  No torch import: device / stack / shared-memory
    conditions are checked where the row is served (they raise Refusal by name there)."""
    row, var, cfgw = split_word(word)
    c, hidden = int(c), int(hidden)
    ccw = _cc_word(cc)
    if row not in ROW_NAMES:
        return False, "unknown_row:%s" % row
    if var is not None and var not in VARIANTS.get(row, ()):
        return False, "unknown_variant:%s:%s" % (row, var)
    if cfgw is not None and row != "v2":
        return False, "cfg_word_only_for_v2"
    if direction not in PASS_WORDS:
        return False, "direction:%s" % direction
    if direction == "fwdbwd" and row not in BACKWARD_ROWS:
        return False, "forward_only:%s" % row
    if row in STOCK_ROWS or row in SCHEDULE_ROWS:
        return True, ""
    if mask and row != "v2":
        return False, "row_mask_folded_by_v2_only"
    if row == "v2":
        if dtype != "bf16":
            return False, "v2:dtype:%s" % dtype
        if cfgw is None and ccw == "9.0" and (c, hidden) not in V2_PACKAGE_CELLS:
            return False, "v2:cell_%dx%d_not_configured" % (c, hidden)
        if cfgw is not None:
            try:
                cfg = cfg_of(cfgw)
            except Refusal as e:
                return False, e.kind
            if hidden % cfg["BH"] != 0:
                return False, "v2:hidden_%d_not_multiple_of_BH_%d" % (hidden, cfg["BH"])
        if rows_count is not None and int(rows_count) >= V2_ROW_LIMIT:
            return False, "v2:rows_%d_int32_row_index" % int(rows_count)
        return True, ""
    if row in ("v1", "pf"):
        if dtype not in ("bf16", "fp32"):
            return False, "%s:dtype:%s" % (row, dtype)
        if dtype == "fp32" and residual:
            return False, "%s:fp32_x_served_residual_false_only" % row
        if hidden % c != 0:
            return False, "%s:hidden_not_factor_of_c" % row
        if c not in V1_C:                                             # the pair-fused kernels' measured widths; wider rows (c 768) run at x0.02-0.03 of the statement
            return False, "%s:c%d_outside_%s" % (row, c, "/".join(map(str, V1_C)))
        return True, ""
    if row == "lnl":
        if c not in LNL_C:
            return False, "lnl:c%d_outside_%s" % (c, "/".join(map(str, LNL_C)))
        if hidden % 64 != 0:
            return False, "lnl:hidden_%d_not_multiple_of_64" % hidden
        if dtype not in ("bf16", "fp32"):
            return False, "lnl:dtype:%s" % dtype
        return True, ""
    if row == "af3_fused":
        if c not in AF3_FUSED_C:
            return False, "af3_fused:c%d_outside_%s" % (c, "/".join(map(str, AF3_FUSED_C)))
        if hidden % 64 != 0:
            return False, "af3_fused:hidden_%d_not_multiple_of_64" % hidden
        if dtype not in ("bf16", "fp32"):
            return False, "af3_fused:dtype:%s" % dtype
        fl = (table().get("row_floors") or {}).get("af3_fused")                 # the row's row-count floors (a kit arch table encoded as data): refused by name below them
        if fl and rows_count is not None:
            if int(rows_count) < int(fl.get("min_rows", 0)):
                return False, fl["words"]["below_min_rows"]
            pad = fl.get("padded") or {}
            if c in (pad.get("c") or ()) and dtype == pad.get("dtype"):
                rule = pad.get(_cc_word(cc))
                if rule == "off":
                    return False, fl["words"]["padded_off"]
                if isinstance(rule, dict) and int(rows_count) < int(rule.get("min_rows", 0)):
                    return False, fl["words"]["padded_below"]
        buckets = af3_pinned_buckets(cc, dtype, c, hidden, residual)
        if not buckets:
            return False, "af3_fused:no_launch_pin:%s|%s|c%d|h%d|res%d" % (_cc_word(cc), dtype, c, hidden, int(bool(residual)))
        if rows_count is not None and af3_rows_bucket(rows_count) not in buckets:
            return False, "af3_fused:no_launch_pin:%s" % af3_pin_key(cc, dtype, rows_count, c, hidden, residual)
        return True, ""
    if row == "flash_sm90a":
        if (c, hidden) != FLASH_CELL:
            return False, "flash_sm90a:cell_%dx%d_not_%dx%d" % (c, hidden, FLASH_CELL[0], FLASH_CELL[1])
        if dtype != "bf16":
            return False, "flash_sm90a:dtype:%s" % dtype
        if ccw != "9.0":
            return False, "flash_sm90a:cc_%s_not_9.0" % ccw
        return True, ""
    if row in ESM_ROWS:
        if c != ESM_C:
            return False, "%s:c%d_not_%d" % (row, c, ESM_C)
        if dtype != "bf16":
            return False, "%s:dtype:%s" % (row, dtype)
        if row != "esm_t16" and not residual:
            return False, "%s:residual_folded_only" % row
        if row == "esm_t15" and hidden % 32 != 0:
            return False, "esm_t15:hidden_%d_not_multiple_of_32" % hidden
        if row in ("esm_t16", "esm_t16_kd3"):
            if (c, hidden) != T16_CELL:
                return False, "%s:cell_%dx%d_not_%dx%d" % (row, c, hidden, T16_CELL[0], T16_CELL[1])
            if ccw != "9.0":
                return False, "%s:cc_%s_not_9.0" % (row, ccw)
        try:
            cc_ok = float(ccw) >= INHERIT_MIN_CC                       # source-compiled rows: any capability from 8.0 (the cubin rows keep their own gate above)
        except ValueError:
            cc_ok = False
        if not cc_ok:
            return False, "%s:cc_%s_below_%.1f" % (row, ccw, INHERIT_MIN_CC)
        return True, ""
    if row == "esm_fused_exact":
        if c != ESM_FUSED_C:
            return False, "esm_fused_exact:c%d_not_%d" % (c, ESM_FUSED_C)
        if hidden <= 0 or hidden % ESM_FUSED_HIDDEN_STEP != 0:
            return False, "esm_fused_exact:hidden_%d_not_multiple_of_%d" % (hidden, ESM_FUSED_HIDDEN_STEP)
        if dtype != "bf16":
            return False, "esm_fused_exact:dtype:%s" % dtype
        if ln_given:
            return False, "esm_fused_exact:x_ln_given_the_row_carries_its_statements_layernorm"
        if ccw not in ESM_FUSED_CCS:
            return False, "esm_fused_exact:cc_%s_unmeasured" % ccw
        return True, ""
    return False, "no_rule:%s" % row


def _short(text, n=72):
    """A refusal kind: the reason's words up to the first path / parenthesis, spaces -> underscores (the Refusal's detail keeps the full text)."""
    t = str(text).split("(")[0].split(" /")[0].split(": /")[0].strip().rstrip(":,;")
    return t[:n].replace(" ", "_")


def _fallback_of(row):
    r = rows().get(row) or {}
    return r.get("fallback") or "torch_swiglu"


def select(word, *, c, hidden, n_tokens=None, dtype="bf16", direction="fwd", timing="eager", family="pair", residual=False, mask=False,
           cc=None, device=None, stack=None, capture=False, rows_count=None, ln_given=False, form="swiglu"):
    """Resolve ``word`` for the call (see ``_select``); the decision is recorded ONCE per call class in :mod:`opt_core.cell_census` (pure
    observation: the Selection / Refusal is decided first and returned unchanged)."""
    kw = dict(c=c, hidden=hidden, n_tokens=n_tokens, dtype=dtype, direction=direction, timing=timing, family=family, residual=residual, mask=mask,
              cc=cc, device=device, stack=stack, capture=capture, rows_count=rows_count, ln_given=ln_given, form=form)
    try:
        sel = _select(word, **kw)
    except Refusal as e:
        _census(word, kw, None, e)
        raise
    _census(word, kw, sel, None)
    return sel


def _census_key(word, kw, sel):
    """The census key dict _census() books a served Selection under (same construction; inherited selections under the caller's own capability)."""
    ckey = sel.cell_key if sel is not None else None
    cc, n = kw.get("cc"), kw.get("n_tokens")
    ccw = _cc_memo(cc, kw.get("device")) if (ckey is None or any(w_.startswith("inherited_from:") for w_ in (sel.words or ()))) else ckey.split("|")[0]
    form = kw.get("form")
    cellw = cell_word(kw.get("c"), kw.get("hidden"), kw.get("family"), form) if form in FORMS else None
    inside = bool(ckey) and not (sel is not None and sel.beyond_measured)
    st = kw.get("stack")
    if st is None and sel is not None and sel.tier is not None and sel.stack_measured is False:
        st = _stack_memo(kw.get("device"))
    return dict(cc=ccw, stack=st, dtype=kw.get("dtype"), shape=cellw or "c%s_h%s_%s" % (kw.get("c"), kw.get("hidden"), kw.get("family")),
                bucket=_CENSUS.bucket_word(n, ckey, inside), form="%s.%s.%s%s" % (kw.get("direction"), kw.get("timing"), form, ".capture" if kw.get("capture") else ""),
                word=word)


def _census(word, kw, sel, refusal):
    """Classify the decision (cell_hit | inherited | named_fallback | stock | opt_in) from the Selection's own facts and record it."""
    try:
        ckey = sel.cell_key if sel is not None else None
        cc, n = kw.get("cc"), kw.get("n_tokens")
        ccw = ckey.split("|")[0] if ckey else _cc_memo(cc, kw.get("device"))   # the cc _select compared (the device's when cc is None; its "9.0" default without CUDA)
        form = kw.get("form")
        cellw = cell_word(kw.get("c"), kw.get("hidden"), kw.get("family"), form) if form in FORMS else None
        inside = bool(ckey) and not (sel is not None and sel.beyond_measured)
        st = kw.get("stack")
        if st is None and sel is not None and sel.tier is not None and sel.stack_measured is False:
            st = _stack_memo(kw.get("device"))                         # what _select compared (None: no torch / no GPU here -- the stack is unknown, not unmeasured)
        if st is None and refusal is not None and str(refusal.kind).startswith("exact_vouch_not_recorded_on_"):
            st = str(refusal.kind)[len("exact_vouch_not_recorded_on_"):] or None   # the stack _select resolved and refused the exact vouch on (named in the refusal)
        key = dict(cc=ccw, stack=st, dtype=kw.get("dtype"), shape=cellw or "c%s_h%s_%s" % (kw.get("c"), kw.get("hidden"), kw.get("family")),
                   bucket=_CENSUS.bucket_word(n, ckey, inside), form="%s.%s.%s%s" % (kw.get("direction"), kw.get("timing"), form, ".capture" if kw.get("capture") else ""),
                   word=word)
        if refusal is not None:
            if str(refusal.kind).startswith("no_cell:"):                 # a tier word with no measured family: the caller binds the named stock row
                _CENSUS.record("transition", key, "inherited", "caller:%s" % (refusal.fallback or "-"), cell_id=None, note="family:none(%s)" % refusal.kind)
            else:
                _CENSUS.record("transition", key, "named_fallback", "caller:%s" % (refusal.fallback or "-"), cell_id=None,
                               refused="%s:%s" % (refusal.row or word, refusal.kind), note="raised")
            return
        served = _arm_word(sel.row, sel.variant, cfg_word(sel.cfg) if (sel.cfg and sel.row == "v2") else None)
        inh = [w for w in (sel.words or ()) if w.startswith("inherited_from:")]
        if inh:                                                    # an unmeasured capability served by inheritance: booked under the part's own capability
            key["cc"] = _cc_memo(cc, kw.get("device"))
            _CENSUS.record("transition", key, "inherited", served, cell_id=ckey, note="inherited_cc:unmeasured;%s" % inh[0].replace("inherited_from:", "from:"))
            return
        if sel.tier is None:
            _CENSUS.record("transition", key, "opt_in", served, cell_id=ckey, note=",".join(sel.words or ()))
        elif ckey is None:
            _CENSUS.record("transition", key, "inherited", served, cell_id=None, note="family:none")
        else:
            sib = _CENSUS.sibling_note(sel.stack) if (sel.stack_measured is False and st is not None) else ""   # the cell's timing column is a sibling stack's (same cc): a hit, said so
            if sel.beyond_measured:                                      # a TRUE gap: a size neighbour
                _CENSUS.record("transition", key, "inherited", served, cell_id=ckey, note="beyond_measured" + ((";" + sib) if sib else ""))
            elif sel.row in STOCK_ROWS:
                _CENSUS.record("transition", key, "stock", served, cell_id=ckey, note=sib)
            else:
                _CENSUS.record("transition", key, "cell_hit", served, cell_id=ckey, note=sib or ",".join(sel.words or ()))
    except Exception:                                              # the census counts; it never gates or breaks a selection
        return


_STACKS_SEEN = {}
_CC_SEEN = {}


def _cc_memo(cc, device):
    """The cc word _select resolves (cc given; else the CUDA device's capability; else "9.0"), once per device word."""
    if cc is not None:
        return _cc_word(cc)
    k = str(device)
    if k not in _CC_SEEN:
        w = "9.0"
        try:
            import torch
            if torch.cuda.is_available():
                w = _cc_word("%d.%d" % torch.cuda.get_device_capability(device))
        except Exception:                                          # no torch / no device: _select's own default
            w = "9.0"
        _CC_SEEN[k] = w
    return _CC_SEEN[k]


def _stack_memo(device):
    """_try_stack(device) once per device word (the census asks per decided call; the stack of a device does not change in a process)."""
    k = str(device)
    if k not in _STACKS_SEEN:
        _STACKS_SEEN[k] = _try_stack(device)
    return _STACKS_SEEN[k]


def _select(word, *, c, hidden, n_tokens=None, dtype="bf16", direction="fwd", timing="eager", family="pair", residual=False, mask=False,
            cc=None, device=None, stack=None, capture=False, rows_count=None, ln_given=False, form="swiglu"):
    """Resolve ``word`` (a row word 'v2' / 'v2:fast' / 'v2@bm64bh64w4s2il1' / 'pf:fpf' / ... or a tier word 'fast' | 'faithful' | 'exact' |
    'big') for the call (c, hidden, dtype, direction, timing, family, residual, mask) on capability ``cc`` ('9.0' | '8.0'; read from
    ``device`` when None and torch sees a GPU) -> ``Selection``; raises ``Refusal`` by name with the fallback row.  ``word=None`` is refused
    ('no_word'): the provider never chooses for a caller that did not ask (a kit keeps its binding until it names a row or a tier word)."""
    if word is None:
        raise Refusal("no_word", None, "torch_swiglu", "name a row word (%s) or a tier word (%s)" % ("|".join(ROW_NAMES), "|".join(TIER_WORDS)))
    if cc is None:
        try:
            import torch
            if torch.cuda.is_available():
                cc = "%d.%d" % torch.cuda.get_device_capability(device)
        except ImportError:
            pass
    ccw = _cc_word(cc) if cc is not None else "9.0"
    if form not in FORMS:
        raise Refusal("unknown_form:%s" % form, word, "torch_swiglu", "form = %s" % " | ".join(FORMS))
    cellw = cell_word(c, hidden, family, form)
    key, size, beyond = (None, None, False)
    inherited_from = None
    if cellw is not None and n_tokens is not None:
        key, size, beyond = cell_key(ccw, dtype, cellw, n_tokens, timing, direction)
        if key is None:                                            # no cell FOR THIS KEY on the part's capability (whether or not it owns cells for other keys): the nearest
            icc = inherit_cc(ccw, dtype, cellw, timing, direction)    # arch-compatible measured column that carries the family
            if icc is not None:
                key, size, beyond = cell_key(icc, dtype, cellw, n_tokens, timing, direction)
                inherited_from = icc if key is not None else None
    tier = word if word in TIER_WORDS else None
    if tier is not None and inherited_from is not None:            # INHERITED tier words: fast | faithful | big = the inherited cell's word when its row is source-compiled
        cell = table()["cells"][key]                               # (else that cell's fastest portable admitted row; never a refusal while a portable row exists);
        sc = cell["stacks"][cell["ref_stack"]]                     # exact = the statement by name (no byte vouch exists on an unmeasured capability)
        floor_row = "engine_module" if family == "esmpair" else "torch_swiglu"
        toks = ("inherited_cc:unmeasured", "inherited_from:%s" % inherited_from)
        arm = sc.get(tier)
        pick, src = (None, None)
        if tier != "exact" and arm is not None:
            pick, src = _inherited_arm(arm, sc, cc_real=ccw, cc_from=inherited_from, c=c, hidden=hidden, dtype=dtype, direction=direction, timing=timing,
                                       residual=residual, mask=mask, rows_count=rows_count, ln_given=ln_given, family=family)
        if pick is None:
            stock_arm = sc.get("stock_arm") or floor_row
            sel = _selection(floor_row, None, None, word, key, tier, sc, stock_arm, cell["ref_stack"], False, beyond, capture, c, hidden, ccw)
            sel.words = tuple(sel.words) + toks + (("exact_unvouched_on_cc_%s" % ccw,) if tier == "exact" else ("no_portable_row",))
            return sel
        row, var, cfgw = split_word(pick)
        sel = _selection(row, var, cfgw, word, key, tier, sc, src, cell["ref_stack"], False, beyond, capture, c, hidden, ccw)
        sel.words = tuple(sel.words) + toks + ((("substituted_for:%s" % arm),) if split_word(pick)[:2] != split_word(arm)[:2] else ())
        return sel
    if tier is not None:
        if key is None:
            raise Refusal("no_cell:%s|%s|%s|%s|%s" % (ccw, dtype, cellw or ("c%d_h%d_%s" % (int(c), int(hidden), family)), timing, direction), word, "torch_swiglu",
                          "tier words resolve measured cells only (n_tokens required); name a row word")
        cell = table()["cells"][key]
        st = norm_stack(stack) if stack is not None else _try_stack(device)
        stack_measured = st in cell["stacks"]
        sc = cell["stacks"][st] if stack_measured else cell["stacks"][cell["ref_stack"]]
        arm = sc.get(tier)
        if arm is None:
            raise Refusal("no_winner:%s:%s" % (tier, key), word, "torch_swiglu")
        if capture and split_word(arm)[0] in CAPTURE_UNSAFE_ROWS:
            raise Refusal("capture_unsafe:%s" % arm, word, "torch_swiglu")
        row, var, cfgw = split_word(arm)
        if row == "v2" and cfgw is None and ccw != "9.0" and tier != "exact":   # a bare v2 word off cc 9.0 would launch the package's 9.0 default tile UNMEASURED on this part:
            pick = _measured_v2_substitute(sc, arm, c=c, hidden=hidden, dtype=dtype, cc=ccw, direction=direction, residual=residual or family == "esmpair",
                                          mask=mask, rows_count=rows_count, ln_given=ln_given)              # the column's fastest MEASURED launch word serves instead
            if pick is None:
                pick = "engine_module" if family == "esmpair" else "torch_swiglu"
            sub_note = ("substituted_for:%s" % arm,) if pick != arm else ()
            arm = pick
            row, var, cfgw = split_word(arm)
        else:
            sub_note = ()
        if tier == "exact" and row not in STOCK_ROWS:              # an EXACT vouch (bitwise vs the statement) holds only on the stack AND sizes it was measured on
            floor_row = "engine_module" if family == "esmpair" else "torch_swiglu"
            if st in (table().get("exact_row_bands") or {}):                      # a stack whose bitwise class depends on the row count's remainder modulo the GEMM library's split (kit-measured)
                band = table()["exact_row_bands"][st]
                if row in band.get("rows", ()) and dtype == band.get("dtype", dtype):
                    rows_here = int(rows_count) if rows_count is not None else (int(n_tokens) ** 2 if (family in ("pair", "rows", "esmpair")) else int(n_tokens))
                    rem = rows_here % int(band["modulus"])
                    if not (int(band["min_remainder"]) <= rem <= int(band["max_remainder"])):
                        raise Refusal("exact_rows_remainder_outside_the_vouched_band_on_%s" % _short(st), word, floor_row,
                                      "%d rows: remainder %d mod %d outside [%d, %d] where %s is bitwise-vouched on this stack; the module floor is the exact tier here" % (
                                          rows_here, rem, int(band["modulus"]), int(band["min_remainder"]), int(band["max_remainder"]), row))
            if st is None:
                raise Refusal("exact_vouch_needs_the_stack", word, floor_row, "pass stack= (or a device) so the vouch can be checked: %s vouched on %s" % (arm, sorted(cell.get("vouched_on", {}))))
            if cell.get("vouched_on", {}).get(st, "").split("@")[0] != arm.split("@")[0]:
                raise Refusal("exact_vouch_not_recorded_on_%s" % _short(st), word, floor_row, "%s is bitwise-vouched on %s only; on this stack the module floor is the exact tier" % (arm, sorted(cell.get("vouched_on", {}))))
            nmin = exact_floor(key, st)
            if nmin is None or int(n_tokens) < nmin:
                raise Refusal("exact_vouch_below_%s_tokens_on_%s" % (nmin, _short(st)), word, floor_row, "%s is bitwise-vouched from %s tokens up on this stack; below it the module floor is the exact tier" % (arm, nmin))
            if rows_count is not None:                                 # the card's cuBLAS piece-remainder rule (attn/pair_fused_cells.json "exact_rows", one statement for every exact
                from ...attn import pair_fused as _PFX                    # construction of the family): outside the served rows the kernel arm cannot equal the statement's GEMM order
                wx = _PFX.exact_rows_word("transition", ccw, int(rows_count))
                if wx is not None:
                    raise Refusal("exact_rows_%s_on_cc_%s" % (wx, ccw), word, floor_row, "%d rows: the stock GEMMs take another summation order there on cc %s (attn.pair_fused exact_rows); the module floor is the exact tier for this call" % (int(rows_count), ccw))
            if capture and row in NEEDS_PRIME and not _row_primed(row) and _capturing():   # the vouched row's ONE-TIME host
                raise Refusal("%s:init_during_capture" % row, word, floor_row,                # initialisation cannot run inside a live CUDA-graph capture and no OTHER row
                              "the exact word's row %s does its one-time process initialisation at first use, which cannot run inside a CUDA-graph capture, "
                              "and no other row carries this stack's byte vouch for cell %s: the module floor is the exact tier for this capture "
                              "(transition.prime() -- or one eager call of this class -- before capturing puts the row inside the graph). "
                              "Before 0.5.212.4 transition() served the cell's fastest OTHER measured row here (a tolerance row's bytes under the exact word: "
                              "esmfold2 full_msa exact, pair_c256_n4 N<=400 inside the MSA-encoder capture -> v2:fast, 21 of 25 identity items off stock)" % (arm, key))
        ok, why = admits(arm, c=c, hidden=hidden, dtype=dtype, cc=ccw, direction=direction, residual=residual or family == "esmpair", mask=mask,
                         rows_count=rows_count, ln_given=ln_given)
        if not ok and not arm.startswith("x:"):
            if mask and row != "v2":                               # the tier winner cannot fold a row mask: name the v2 word or multiply in the caller
                raise Refusal("tier_winner_takes_no_mask:%s" % arm, word, "v2")
            raise Refusal("tier_winner_outside_envelope:%s:%s" % (arm, why), word, _fallback_of(row))
        sel_ = _selection(row, var, cfgw, word, key, tier, sc, arm, st if stack_measured else cell["ref_stack"], stack_measured, beyond, capture, c, hidden, ccw)
        if sub_note:
            sel_.words = tuple(sel_.words or ()) + sub_note
        return sel_
    row, var, cfgw = split_word(word)
    if row not in ROW_NAMES:
        raise Refusal("unknown_row:%s" % row, word, "torch_swiglu")
    if row == "v1" and (var == "liger") != (form == "liger"):          # v1's SiLU*b arithmetic is the statement's: the plain word under the Liger form (or v1:liger under the plain form) is refused by name
        raise Refusal("v1:form_%s_needs_%s" % (form, "v1:liger" if form == "liger" else "v1"), word, "torch_swiglu")
    ok, why = admits(word, c=c, hidden=hidden, dtype=dtype, cc=ccw, direction=direction, residual=residual or family == "esmpair", mask=mask,
                     rows_count=rows_count, ln_given=ln_given)
    if not ok:
        raise Refusal(why, word, _fallback_of(row))
    if capture and row in CAPTURE_UNSAFE_ROWS:
        raise Refusal("capture_unsafe:%s" % row, word, _fallback_of(row))
    sc, st, stack_measured = None, None, None
    if key is not None:
        cell = table()["cells"][key]
        st = stack if stack is not None else _try_stack(device)
        stack_measured = st in cell["stacks"]
        sc = cell["stacks"][st] if stack_measured else cell["stacks"][cell["ref_stack"]]
        if not stack_measured:
            st = cell["ref_stack"]
    if row == "v2" and cfgw is None and ccw != "9.0":              # no package configuration off cc 9.0: the measured sweep's best launch for this cell, named
        best = None
        if sc is not None:
            cands = [(ms, a) for a, ms in sc["ms"].items() if a.startswith("v2@") and (var is None)]
            if cands:
                best = min(cands)[1]
        if best is None:
            raise Refusal("v2:no_measured_launch_on_cc_%s_for_%dx%d" % (ccw, int(c), int(hidden)), word, "torch_swiglu", "name v2@<cfg> or a tier word")
        cfgw = split_word(best)[2]
    return _selection(row, var, cfgw, word, key, None, sc, _arm_word(row, var, cfgw), st, stack_measured, beyond, capture, c, hidden, ccw)


def _arm_word(row, var, cfgw):
    return row + ((":" + var) if var else "") + (("@" + cfgw) if cfgw else "")


def _try_stack(device):
    try:
        return stack_word(device)
    except Exception:                                              # no torch / no GPU: the reference stack's numbers, flagged
        return None


def _selection(row, var, cfgw, word, key, tier, sc, arm, st, stack_measured, beyond, capture, c, hidden, ccw):
    cfg = cfg_of(cfgw) if row == "v2" and cfgw else None      # None on cc 9.0 = the package's own configuration for the cell
    words = []
    if capture and row in WARM_BEFORE_CAPTURE_ROWS:
        words.append("warm_before_capture")
    if beyond:
        words.append("beyond_measured")
    if stack_measured is False:
        words.append("reference_stack_numbers")
    if row in EXACT_GIVEN_LN_ROWS and var is None:
        words.append("exact_class_given_the_callers_ln_output")
    r = rows().get(row, {})
    ms = x = cls = None
    if sc is not None and arm in sc.get("ms", {}):
        ms, x, cls = sc["ms"][arm], sc.get("x_stock", {}).get(arm), sc.get("class", {}).get(arm)
    elif sc is not None and arm in sc.get("refused", {}):
        words.append("measured_refused_on_this_cell:" + str(sc["refused"][arm])[:60].replace(" ", "_"))
    return Selection(row=row, variant=var, cfg=cfg, word=word, cell_key=key, tier=tier, ms=ms, x_stock=x, cls=cls, stack=st,
                     stack_measured=stack_measured, beyond_measured=beyond, fallback=r.get("fallback"), words=tuple(words),
                     backward=row in BACKWARD_ROWS, capture_safe=row not in CAPTURE_UNSAFE_ROWS)


def rows_for_kit(*, c, hidden, dtype="bf16", cc="9.0", direction="fwd", timing="eager", family="pair", n_tokens=(400, 800, 1200)):
    """{n_tokens: {tier: arm}} straight from the table for one cell family -- what an adopter reads before naming a word."""
    out = {}
    cellw = cell_word(c, hidden, family)
    for n in n_tokens:
        key, size, beyond = cell_key(cc, dtype, cellw, n, timing, direction) if cellw else (None, None, False)
        if key is None:
            out[n] = None
            continue
        cell = table()["cells"][key]
        out[n] = {"cell": key, "fast": cell.get("fast"), "fast_x": cell.get("fast_x"), "faithful": cell.get("faithful"), "exact": cell.get("exact"),
                  "big": cell.get("big"), "stock_arm": cell.get("stock_arm"), "lead_uncarried": cell.get("lead_uncarried"), "beyond_measured": beyond}
    return out


# ------------------------------------------------------------------------------------------------------------------------- carried modules
_MODULES = {"v2": ("opt_core.kernels.fpf_transition_v2.kernel",), "v1": ("opt_core.attn.pair_fused",), "pf": ("opt_core.attn.pair_fused",),
            "lnl": ("opt_core.kernels.lnl_fused",), "af3_fused": ("opt_core.kernels.transition.af3.af3_fused",),
            "flash_sm90a": ("opt_core.kernels.transition.flash_sm90a",), "esm_t15": ("opt_core.kernels.transition.esm.ef2_pair_v2",),
            "esm_t16": ("opt_core.kernels.transition.esm.ef2_t16_transition",),
            "esm_kd3": ("opt_core.kernels.transition.esm.ef2_autograd_kernels",),
            "esm_t16_kd3": ("opt_core.kernels.transition.esm.ef2_autograd_kernels", "opt_core.kernels.transition.esm.ef2_t16_transition"),
            "esm_t15_kd3": ("opt_core.kernels.transition.esm.ef2_autograd_kernels", "opt_core.kernels.transition.esm.ef2_pair_v2"),
            "esm_fused_exact": ("opt_core.kernels.transition.esm_fused",),
            "rowpair": ("opt_core.mem.rowpair.transition",)}


def carried_module(row, which=0):
    """Import the module a row serves from (ImportError / a missing stack dependency -> Refusal by name with the row's fallback)."""
    import importlib
    names = _MODULES.get(row)
    if not names:
        raise Refusal("no_module:%s" % row, row, _fallback_of(row))
    name = names[which]
    try:
        if ".esm." in name:                                        # the design kit's modules import one another by top-level name: bind those names to the carried copies first
            importlib.import_module("opt_core.kernels.transition.esm").bind_names()
        return importlib.import_module(name)
    except ImportError as e:
        raise Refusal("import:%s:%s" % (name.rsplit(".", 1)[-1], str(e).split("\n")[0][:80].replace(" ", "_")), row, _fallback_of(row))
    except Exception as e:                                         # a carried module's own import-time probe failing (no GPU, another fork): named, not fatal
        raise Refusal("import:%s:%s:%s" % (name.rsplit(".", 1)[-1], type(e).__name__, str(e)[:60].replace(" ", "_")), row, _fallback_of(row))


# ----------------------------------------------------------------------------------------------------------------------------------- weights
PACK_CACHE_ENV = "OPT_CORE_TRANSITION_PACK_CACHE"       # engineering knob: "layer" | "lru" forces the pack residency policy for every word (kits' A/B)
PACK_LRU_ENV = "OPT_CORE_TRANSITION_PACK_LRU"           # weight sets kept under the lru policy (default 1)
_PACK_LRU = {}                                          # id(Weights) -> (the weight set, {pack name: pack}, [bytes]); insertion order = recency (at most PACK_LRU sets)
_PACK_STATS = {"builds": 0, "evictions": 0, "lru_bytes": 0, "layer_bytes": 0, "layer_builds": 0}


def pack_nbytes(obj, _seen=None):
    """Bytes of every tensor-like member of a pack (duck-typed: numel() * element_size(), or .nbytes), through tuples / lists / dicts / attributes."""
    _seen = _seen if _seen is not None else set()
    if obj is None or id(obj) in _seen or isinstance(obj, (int, float, str, bytes, bool)):
        return 0
    _seen.add(id(obj))
    if hasattr(obj, "numel") and hasattr(obj, "element_size"):
        try:
            return int(obj.numel()) * int(obj.element_size())
        except (TypeError, ValueError, AttributeError):                 # not a tensor after all: counted as 0 (bookkeeping only)
            return 0
    if isinstance(obj, dict):
        return sum(pack_nbytes(v, _seen) for v in obj.values())
    if isinstance(obj, (tuple, list, set, frozenset)):
        return sum(pack_nbytes(v, _seen) for v in obj)
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict) and d:
        return sum(pack_nbytes(v, _seen) for v in d.values())
    if hasattr(obj, "nbytes"):
        try:
            return int(obj.nbytes)
        except (TypeError, ValueError, AttributeError):                 # idem
            return 0
    return 0


def _lru_keep():
    try:
        return max(1, int(os.environ.get(PACK_LRU_ENV, "1")))
    except ValueError:
        return 1


def _lru_drop(key):
    ent = _PACK_LRU.pop(key, None)
    if ent is not None:
        _PACK_STATS["lru_bytes"] -= ent[2][0]
        _PACK_STATS["evictions"] += 1


_RESIDENCY = {"policy": None}                            # the process-level pack residency a kit's MODE asked for (pack_residency()); None = follow the tier word


def pack_residency(policy=False):
    """Get / set the PROCESS-LEVEL pack residency policy: ``pack_residency()`` -> the current setting (None | 'lru' | 'layer'); ``pack_residency('lru')``
    makes every later call of every word pack through the bounded lru store (what a kit's memory-lean MODE asks once at activation when the rows it
    binds are not the big tier word's -- e.g. exact-class rows served inside a big mode), ``pack_residency('layer')`` forces the per-layer cache,
    ``pack_residency(None)`` restores the default (the big tier word -> lru, every other word -> layer).  Precedence per call: the engineering knob
    OPT_CORE_TRANSITION_PACK_CACHE, then this setting, then the word.  Returns the previous setting when setting."""
    if policy is False:
        return _RESIDENCY["policy"]
    if policy not in (None, "lru", "layer"):
        raise ValueError("pack_residency: %r is not None | 'lru' | 'layer'" % (policy,))
    prev, _RESIDENCY["policy"] = _RESIDENCY["policy"], policy
    return prev


def pack_policy(word_or_selection=None):
    """'layer' | 'lru' for a call: OPT_CORE_TRANSITION_PACK_CACHE forces either; else the process-level pack_residency() setting; otherwise the big
    tier word packs through the lru store (resident packs bounded by OPT_CORE_TRANSITION_PACK_LRU weight sets, default 1) and every other word keeps
    its packs with the layer's weight set."""
    forced = os.environ.get(PACK_CACHE_ENV, "").strip().lower()
    if forced in ("layer", "lru"):
        return forced
    if _RESIDENCY["policy"] in ("layer", "lru"):
        return _RESIDENCY["policy"]
    tier = getattr(word_or_selection, "tier", None)
    word = tier if tier is not None else getattr(word_or_selection, "word", None)
    if word is None and isinstance(word_or_selection, str):
        word = word_or_selection
    return "lru" if word == "big" else "layer"


def pack_cache_stats():
    """Bookkeeping of the pack residency: {'lru_entries', 'lru_bytes' (resident in the lru store), 'lru_keep', 'layer_bytes' / 'layer_builds' (built into
    per-layer caches since import; those live and die with the kits' weight sets), 'builds', 'evictions'}."""
    return {"deferred_evictions": int(_PACK_STATS.get("deferred_evictions", 0)), "lru_entries": len(_PACK_LRU), "lru_bytes": int(_PACK_STATS["lru_bytes"]), "lru_keep": _lru_keep(), "layer_bytes": int(_PACK_STATS["layer_bytes"]), "residency": _RESIDENCY["policy"],
            "layer_builds": int(_PACK_STATS["layer_builds"]), "builds": int(_PACK_STATS["builds"]), "evictions": int(_PACK_STATS["evictions"])}


def pack_cache_clear():
    """Drop every lru entry (references only)."""
    for k in list(_PACK_LRU):
        _lru_drop(k)


class Weights(object):
    """Canonical transition weights packed once per module: ``w_a`` / ``w_b`` [hidden, c] (or ``w_ab`` [2*hidden, c] rows a-then-b), ``w_o``
    [c, hidden], LayerNorm ``ln_w`` / ``ln_b`` [c] (None = no affine), ``eps``.  Per-row operand packs are built lazily on first use and cached
    here (bf16 copies; the interleaved layouts the one-kernel rows read)."""

    def __init__(self, w_a, w_b, w_o, ln_w, ln_b, eps, device):
        import torch
        self.c, self.hidden = int(w_o.shape[0]), int(w_o.shape[1])
        if tuple(w_a.shape) != (self.hidden, self.c) or tuple(w_b.shape) != (self.hidden, self.c):
            raise Refusal("weights:shapes:%s:%s:%s" % (tuple(w_a.shape), tuple(w_b.shape), tuple(w_o.shape)), None, "torch_swiglu")
        dev = device if device is not None else w_o.device
        with torch.no_grad():
            self.w_a = w_a.detach().to(dev)
            self.w_b = w_b.detach().to(dev)
            self.w_o = w_o.detach().to(dev)
            self.ln_w = (ln_w.detach().to(dev) if ln_w is not None else torch.ones(self.c, device=dev)).float().contiguous()
            self.ln_b = (ln_b.detach().to(dev) if ln_b is not None else torch.zeros(self.c, device=dev)).float().contiguous()
        self.eps = float(eps)
        self.device = dev
        self._packs = {}
        self._policy = "layer"
        self._inference_at_pack = bool(torch.is_inference_mode_enabled())

    def _op16(self):
        """bf16 operand copies of w_a / w_b / w_o (the tensors themselves when the module already holds bf16), built on first use like every other pack,
        under the inference mode that was active when the weight set was packed (so a first use inside an inference-mode region does not turn a weight
        set packed outside it into inference tensors, and vice versa)."""
        import torch
        def build():
            with torch.inference_mode(self._inference_at_pack), torch.no_grad():
                bf = torch.bfloat16
                return (self.w_a.to(bf).contiguous(), self.w_b.to(bf).contiguous(), self.w_o.to(bf).contiguous())
        return self.pack_for("op16", build)

    wa16 = property(lambda self: self._op16()[0])
    wb16 = property(lambda self: self._op16()[1])
    wo16 = property(lambda self: self._op16()[2])

    def pack_for(self, name, builder):
        """The row's operand pack for THIS weight set.  Policy 'layer' (the exact / fast tiers and row words): built once, kept with this object for its
        lifetime (one pack per layer resident -- speed).  Policy 'lru' (the big tier, or OPT_CORE_TRANSITION_PACK_CACHE=lru): kept in a process-wide
        least-recently-used store keyed by the weight set, OPT_CORE_TRANSITION_PACK_LRU weight sets deep (default 1) -- resident packs never grow with the
        layer count; older layers' packs are dropped (only the cache's references: the module's own parameters are never freed or mutated) and rebuilt
        on their next call."""
        if getattr(self, "_policy", "layer") != "lru":
            p = self._packs.get(name)
            if p is None:
                p = builder()
                self._packs[name] = p
                _PACK_STATS["builds"] += 1
                _PACK_STATS["layer_builds"] += 1
                _PACK_STATS["layer_bytes"] += pack_nbytes(p)
            return p
        key = id(self)
        ent = _PACK_LRU.pop(key, None)                                    # re-inserted below: insertion order is the recency order
        if ent is not None and ent[0] is not self:                        # a recycled id (cannot happen while the store holds the set; defensive)
            _PACK_STATS["lru_bytes"] -= ent[2][0]
            _PACK_STATS["evictions"] += 1
            ent = None
        if ent is None:
            ent = (self, {}, [0])
        _PACK_LRU[key] = ent
        packs = ent[1]
        p = packs.get(name)
        if p is None:
            p = builder()
            packs[name] = p
            nb = pack_nbytes(p)
            ent[2][0] += nb
            _PACK_STATS["builds"] += 1
            _PACK_STATS["lru_bytes"] += nb
        keep = _lru_keep()
        if len(_PACK_LRU) > keep and _capturing():                 # packs a capture is recording must stay resident for the graph's lifetime: eviction is deferred to
            _PACK_STATS["deferred_evictions"] = _PACK_STATS.get("deferred_evictions", 0) + 1   # the next pack made outside a capture (a graph replays the addresses it recorded)
            return p
        while len(_PACK_LRU) > keep:
            old = next(iter(_PACK_LRU))
            if old == key:
                break
            _lru_drop(old)
        return p


def pack(*, w_o, w_a=None, w_b=None, w_ab=None, ln_w=None, ln_b=None, eps=1e-5, device=None):
    """Pack canonical weights (see ``Weights``).  ``w_ab`` [2*hidden, c] = rows a (the silu branch) then b -- the ESM-family fork's w12 and the
    Boltz-family fc1|fc2 concatenation have this layout."""
    if w_ab is not None:
        h = int(w_ab.shape[0]) // 2
        w_a, w_b = w_ab[:h], w_ab[h:]
    if w_a is None or w_b is None:
        raise Refusal("weights:missing_w_a_w_b", None, "torch_swiglu")
    return Weights(w_a, w_b, w_o, ln_w, ln_b, eps, device)


def reference(x, W, residual=False, dtype=None, mask=None):
    """The statement evaluated in ``dtype`` (float64 by default) -- the numerics yardstick."""
    import torch
    import torch.nn.functional as F
    dt = dtype or torch.float64
    xx = x.to(dt)
    n = F.layer_norm(xx, (W.c,), W.ln_w.to(dt), W.ln_b.to(dt), W.eps)
    y = F.linear(F.silu(F.linear(n, W.w_a.to(dt))) * F.linear(n, W.w_b.to(dt)), W.w_o.to(dt))
    if mask is not None:
        y = y * mask.reshape(y.shape[:-1]).to(dt)[..., None]
    return xx + y if residual else y


def _liger_silu_mul():
    """The Liger-Kernel SiLU*b autograd function (LigerSiLUMulFunction.apply) of this process, or None when liger_kernel does not import."""
    try:
        from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    except Exception:                                                  # not installed / no triton: the Liger form is then refused by name (never emulated)
        return None
    return LigerSiLUMulFunction.apply


def _torch_swiglu(x, W, residual, x_ln, mask, autocast, form="swiglu", word="torch_swiglu"):
    """The stock statement: LayerNorm (fp32 params; under bf16 autocast for bf16 rows = the AF3-family trunk form) -> Linear a|b -> silu(a)*b
    (form 'swiglu': F.silu(a) * b; form 'liger': LigerSiLUMulFunction(a, b), the Liger kernel itself -- refused by name when it does not import) ->
    Linear [* mask] [+ x]."""
    import torch
    import torch.nn.functional as F
    bf16 = x.dtype == torch.bfloat16
    wa, wb, wo = (W.wa16, W.wb16, W.wo16) if bf16 else (W.w_a.to(x.dtype), W.w_b.to(x.dtype), W.w_o.to(x.dtype))
    silu_mul = None
    if form == "liger":
        silu_mul = _liger_silu_mul()
        if silu_mul is None:
            raise Refusal("torch_swiglu:form_liger_needs_liger_kernel", word, "engine_module", "the Liger SiLU*b statement is the Liger kernel itself; it does not import here")

    def body():
        n = x_ln if x_ln is not None else F.layer_norm(x, (W.c,), W.ln_w.to(x.dtype) if not bf16 else W.ln_w, W.ln_b.to(x.dtype) if not bf16 else W.ln_b, W.eps)
        if silu_mul is not None:
            y = F.linear(silu_mul(F.linear(n, wa), F.linear(n, wb)), wo)
        else:
            y = F.linear(F.silu(F.linear(n, wa)) * F.linear(n, wb), wo)
        if mask is not None:
            y = y * mask.reshape(y.shape[:-1]).to(y.dtype)[..., None]
        return x + y if residual else y
    if bf16 and autocast:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return body()
    return body()


# ----------------------------------------------------------------------------------------------------------------------------------- serving
def transition(x, W, *, word, residual=False, x_ln=None, mask=None, binary_mask=None, out=None, n_tokens=None, family="pair", timing="eager",
               capture=False, stack=None, form="swiglu", cc=None):
    """Serve the FORWARD of ``word`` (row or tier word) on x [..., c] with packed weights ``W`` -> (y, Selection).  ``residual``: return x + update
    (folded in the kernel where the row folds it); ``x_ln``: the caller's own LayerNorm output (rows v1 / flash_sm90a project it instead of
    normalising: the exact-candidate construction); ``mask``: the row mask (v2 only); ``out``: optional output rows (v2 / flash_sm90a; may be x
    for v2 in place).  Refusal by name (with the fallback row) for anything outside the resolved row's envelope; stock words other than
    torch_swiglu name the caller's own call and are refused here ('callers_own_call')."""
    global _WARMED
    if not _WARMED:                                             # first serving call of this face in the process: the stack's heavy libraries imported early + shallow (opt_core.warm)
        _WARMED = True
        from opt_core.warm import auto_warm
        auto_warm("transition")
    import torch
    c = int(x.shape[-1])
    if c != W.c:
        raise Refusal("c:%d!=%d" % (c, W.c), word, "torch_swiglu")
    dtype = {torch.bfloat16: "bf16", torch.float32: "fp32"}.get(x.dtype, str(x.dtype).replace("torch.", ""))
    M = x.numel() // c if c else 0
    skw = dict(c=c, hidden=W.hidden, n_tokens=n_tokens, dtype=dtype, direction="fwd", timing=timing, family=family, residual=residual,
               mask=mask is not None, device=x.device if x.is_cuda else None, stack=stack, capture=capture, rows_count=M, ln_given=x_ln is not None, form=form,
               cc=cc if cc is not None else (_cc_memo(None, x.device) if x.is_cuda else None))
    sel = select(word, **skw)
    if x.is_cuda and sel.cell_key:
        if not _capturing():
            _auto_prime(x, sel, skw)                               # once per (device, dtype, class): rows with process-level init are engaged OUTSIDE any capture
        elif sel.row in NEEDS_PRIME and not _row_primed(sel.row):  # first use of such a row INSIDE a capture: the cell's next capture-safe measured row serves
            alt = _capture_first_use_substitute(sel, skw)           # this call by name (census token); no host-side initialisation is attempted while capturing;
            if alt is None:                                        # NEVER under the exact tier (no other row carries this stack's byte vouch -> by name)
                _note_refused_in_capture(word, skw, sel)
                raise Refusal("%s:init_during_capture" % sel.row, word, "engine_module" if family == "esmpair" else "torch_swiglu",
                              ("the exact word serves no substitute row (none carries this stack's byte vouch for the cell); " if sel.tier == "exact" else "no capture-safe measured row in the cell; ")
                              + "call transition.prime() (or make one eager call of this class) before capturing")
            _note_substituted(word, skw, sel, _arm_word(alt.row, alt.variant, cfg_word(alt.cfg) if (alt.cfg and alt.row == "v2") else None), "init_during_capture:%s" % sel.row)
            sel = alt
    if not any(w_ == "inherited_cc:unmeasured" for w_ in (sel.words or ())):
        return _dispatch(x, W, sel, word, residual, x_ln, mask, binary_mask, out, form)
    # an INHERITED row (unmeasured capability): its first launch on this part is guarded -- a row that fails to build / compile / launch (anything but an
    # out-of-memory) is marked unavailable for the process (census note stepped_aside:error:<type>, once), the donor cell's next portable row serves, else
    # the statement by name; a raw exception never leaves this call for an inherited row (binders convert Refusal only)
    from opt_core.oom import is_oom
    for _ in range(64):
        if sel.row in STOCK_ROWS or not any(w_ == "inherited_cc:unmeasured" for w_ in (sel.words or ())):
            return _dispatch(x, W, sel, word, residual, x_ln, mask, binary_mask, out, form)
        served = _arm_word(sel.row, sel.variant, cfg_word(sel.cfg) if (sel.cfg and sel.row == "v2") else None)
        try:
            return _dispatch(x, W, sel, word, residual, x_ln, mask, binary_mask, out, form)
        except Refusal as e:
            err = e
        except Exception as e:                                     # noqa: BLE001 -- the guarded first launch of an inherited row
            if is_oom(e):
                raise
            err = e
        src = [w_[len("substituted_for:"):] for w_ in (sel.words or ()) if w_.startswith("substituted_for:")]
        for dead in [served] + src + ([_arm_word(sel.row, sel.variant, None)] if sel.cfg else []):
            _DEAD_INHERITED.add(_dead_key(skw["cc"] if skw.get("cc") is not None else _sel_cc(sel), dtype, c, W.hidden, dead))
        _note_stepped_aside(word, skw, sel, served, err)
        sel = select(word, **skw)
    raise Refusal("inherited_rows_exhausted", word, "torch_swiglu")


def _capture_first_use_substitute(sel, skw):
    """The row that serves a call whose resolved row needs un-primed process initialisation INSIDE a live capture: under a TOLERANCE word (fast |
    faithful | big | a row word) the cell column's fastest other measured, admitted, capture-ready arm (``_capture_safe_substitute``); under the
    EXACT tier word None -- no other row carries this stack's byte vouch for the cell, so the caller refuses BY NAME and the kit binds the module
    statement (stock arithmetic)."""
    if sel is not None and sel.tier == "exact":
        return None
    return _capture_safe_substitute(sel, skw, sel.row)


def _note_refused_in_capture(word, skw, sel):
    """Census bookkeeping for a first-use-inside-capture call refused by name (observation only)."""
    try:
        _CENSUS.record("transition", _census_key(word, skw, sel), "named_fallback", "caller:%s" % ("engine_module" if skw.get("family") == "esmpair" else "torch_swiglu"),
                       cell_id=sel.cell_key, refused="%s:init_during_capture" % sel.row, note="raised" + (":exact_serves_no_substitute" if sel.tier == "exact" else ""))
    except Exception:                                              # noqa: BLE001 -- observation only, never a reroute
        return


def _note_substituted(word, skw, sel, served, why):
    """Census bookkeeping for a call served by a substitute row (observation only: a bookkeeping failure never affects serving)."""
    try:
        _CENSUS.record("transition", _census_key(word, skw, sel), "named_fallback", served, cell_id=sel.cell_key, note="substituted:%s" % why)
    except Exception:                                              # noqa: BLE001 -- observation only, never a reroute
        return


def _note_stepped_aside(word, skw, sel, served, err):
    """Census bookkeeping for an inherited row that stepped aside (observation only: a bookkeeping failure never affects serving)."""
    try:
        _CENSUS.record("transition", _census_key(word, skw, sel), "named_fallback", served, cell_id=sel.cell_key, note="stepped_aside:error:%s" % type(err).__name__)
    except Exception:                                              # noqa: BLE001 -- observation only, never a reroute
        return


def _sel_cc(sel):
    """The capability an inherited Selection was resolved for (its exact_unvouched / census bookkeeping uses the caller's cc, not the donor cell's)."""
    for w_ in (sel.words or ()):
        if w_.startswith("exact_unvouched_on_cc_"):
            return w_[len("exact_unvouched_on_cc_"):]
    return getattr(sel, "cc", None) or "?"


NEEDS_PRIME = ("esm_t16", "esm_t16_kd3", "flash_sm90a")        # rows whose FIRST use in a process does host work that cannot run inside a CUDA-graph capture (a cubin
_PRIMED_ROWS = set()                                            # load + install canary with pageable host->device copies; an extension load + load check); primed
_AUTO_PRIMED = set()                                            # rows / (device, dtype, class) keys already handled this process


def _capturing():
    """True while the current CUDA stream is being captured into a graph (False on a host without CUDA or before CUDA is initialised)."""
    try:
        import torch
        return bool(torch.cuda.is_available() and torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing())
    except (ImportError, RuntimeError, AttributeError):
        return False


def _row_primed(row):
    base = "esm_t16" if row == "esm_t16_kd3" else row
    return base not in ("esm_t16", "flash_sm90a") or base in _PRIMED_ROWS


def prime(device=None, rows=("esm_t16", "flash_sm90a")):
    """Do the one-time process-level initialisation of the rows that have one (``esm_t16``: cubin load + install canary; ``flash_sm90a``: extension
    load + load check) NOW, outside any CUDA-graph capture, so a later first call inside a capture launches only device work.  Kits call this once in
    their warm-up before capturing (the face also primes on its own at the first eager call of a class whose graph cells name such a row); returns
    {row: 'primed' | 'stepped_aside:<reason>' | 'refused:capturing'}.  A row that cannot engage is recorded as primed all the same: its later calls refuse
    by name without any host work."""
    out = {}
    if _capturing():
        return {r: "refused:capturing" for r in rows}
    for row in rows:
        base = "esm_t16" if row == "esm_t16_kd3" else row
        if base in _PRIMED_ROWS:
            out[row] = "primed"; continue
        try:
            if base == "esm_t16":
                _t16_live("prime", "torch_swiglu")
            elif base == "flash_sm90a":
                _flash_ext()
            else:
                out[row] = "no_init"; continue
            out[row] = "primed"
        except Refusal as e:
            out[row] = "stepped_aside:%s" % e.kind
        except (RuntimeError, ImportError, OSError, KeyError, AttributeError) as e:   # the row's own loader said no: its serve path refuses by name later
            out[row] = "stepped_aside:%s" % type(e).__name__
        _PRIMED_ROWS.add(base)
    return out


def _auto_prime(x, sel, skw):
    """At an EAGER (non-capturing) call on a device: prime, once per (device, dtype, class), every row with process-level initialisation that ANY cell of
    this call's class (same capability, dtype, family width; any token count, eager or graph) names as a tier winner -- the rows this caller can meet
    later inside a capture.  Costs one cubin / extension load per process; nothing when the class names no such row."""
    try:
        ck = sel.cell_key or ""
        parts = ck.split("|")
        if len(parts) < 3:
            return
        key = (str(x.device), parts[0], parts[1], parts[2])
        if key in _AUTO_PRIMED:
            return
        _AUTO_PRIMED.add(key)
        want = set()
        pre = "%s|%s|%s|" % (parts[0], parts[1], parts[2])
        for k, cell in table()["cells"].items():
            if not k.startswith(pre):
                continue
            for sc in [cell] + list((cell.get("stacks") or {}).values()):
                for tw in ("fast", "faithful", "big", "exact"):
                    a = sc.get(tw)
                    if a and split_word(a)[0] in NEEDS_PRIME:
                        want.add("esm_t16" if split_word(a)[0] == "esm_t16_kd3" else split_word(a)[0])
        if want:
            prime(device=x.device, rows=tuple(sorted(want)))
    except Refusal:
        return
    except (KeyError, ValueError, TypeError, AttributeError):        # bookkeeping only: never a reason to fail the call
        return


def _capture_safe_substitute(sel, skw, refused_row):
    """Inside a capture, the row the cell named needs un-primed process initialisation: the cell column's fastest OTHER measured, admitted arm that needs
    none (or is primed) serves this call -- a Selection for that arm's row word -- else None (the caller refuses by name and the kit binds the statement)."""
    try:
        cell = table()["cells"].get(sel.cell_key) if sel.cell_key else None
        if cell is None:
            return None
        col = (cell.get("stacks") or {}).get(sel.stack) if sel.stack else None
        ms = dict((col or cell).get("ms") or {})
        order = sorted((v, a) for a, v in ms.items() if v is not None)
        for _v, arm in order:
            row = split_word(arm)[0]
            if row in STOCK_ROWS or row == refused_row or (row in NEEDS_PRIME and not _row_primed(row)):
                continue
            if row == "v2" and split_word(arm)[2] is None and str(sel.cell_key).split("|")[0] != "9.0":
                continue
            ok, _why = admits(arm, **{k: skw[k] for k in ("c", "hidden", "dtype", "cc", "direction", "residual", "mask", "rows_count", "ln_given") if k in skw and skw[k] is not None})
            if not ok:
                continue
            try:
                alt = select(arm, **skw)
            except Refusal:
                continue
            if alt.row in NEEDS_PRIME and not _row_primed(alt.row):
                continue
            return alt
    except (KeyError, ValueError, TypeError, AttributeError):
        return None
    return None


def _dispatch(x, W, sel, word, residual, x_ln, mask, binary_mask, out, form):
    """Serve the resolved row (everything after select()): -> (y, Selection) or Refusal by name."""
    W._policy = pack_policy(sel)
    row, var = sel.row, sel.variant
    if row in NEEDS_PRIME and not _row_primed(row) and x.is_cuda and _capturing():   # the row's one-time host-side initialisation cannot run inside a capture:
        raise Refusal("%s:init_during_capture" % row, word, "torch_swiglu",         # refused BY NAME for this call (transition() substitutes the cell's next
                      "call transition.prime() (or make one eager call of this class) before capturing")   # capture-safe measured row); never a raw error
    lead = tuple(x.shape[:-1])
    if not x.is_cuda and row not in STOCK_ROWS:
        raise Refusal("x_not_cuda", word, "torch_swiglu")
    if row == "torch_swiglu":
        return _torch_swiglu(x, W, residual, x_ln, mask, autocast=True, form=form, word=word), sel
    if row in ("engine_module", "compile"):
        raise Refusal("callers_own_call:%s" % row, word, "torch_swiglu", "the word names the caller's own module / compiled call; nothing is constructed here")
    if row == "rowpair":
        raise Refusal("schedule_row:use_opt_core.mem.rowpair.transition.transition_rows", word, "torch_swiglu")
    if row == "v2":
        return _serve_v2(x, W, sel, residual, mask, binary_mask, out, lead), sel
    if row in ("v1", "pf"):
        return _serve_pf(x, W, sel, residual, x_ln, lead), sel
    if row == "lnl":
        L = carried_module("lnl")
        if var == "res" and not residual:
            residual = True
        try:
            y = L.fused_transition(x, W.ln_w, W.ln_b, W.wa16, W.wb16, W.wo16, eps=W.eps, residual=bool(residual))
        except Exception as e:                                     # noqa: BLE001 -- a tile that fails to BUILD on this stack is refused by name (never a silent safe tile)
            from opt_core.oom import is_oom
            if is_oom(e):
                raise
            from opt_core.kernels import safe_settings as _ss
            if isinstance(e, _ss.catchable()) and _ss.is_build_failure(e):
                raise Refusal("lnl:build_failed:%s" % type(e).__name__, word, "torch_swiglu")
            raise
        return y, sel
    if row == "af3_fused":
        if var == "res" and not residual:
            residual = True
        return _serve_af3(x, W, sel, bool(residual)), sel
    if row == "flash_sm90a":
        return _serve_flash(x, W, sel, residual, x_ln, out, lead), sel
    if row == "esm_t15":
        return _serve_t15(x, W, sel, lead), sel
    if row == "esm_t16":
        return _serve_t16(x, W, sel, residual, out, lead), sel
    if row == "esm_fused_exact":
        return _serve_esm_fused(x, W, sel, residual, out, lead), sel
    if row in ("esm_kd3", "esm_t16_kd3", "esm_t15_kd3"):           # their forward without a graph node (no grad) is the autograd row under no_grad
        with torch.no_grad():
            y, _ = transition_autograd(x, W, word=_arm_word(row, var, None), n_tokens=n_tokens, timing=timing, stack=stack, _sel=sel)
        return y, sel
    raise Refusal("no_serving_branch:%s" % row, word, "torch_swiglu")


AF3_AUTOTUNE_ENV = "OPT_CORE_TRANSITION_AF3_AUTOTUNE"      # =1: the carried module's own autotuned launch (its 6-7 candidate tiles timed at the first call of a rows
                                                        # bucket) -- for measuring pins on a new card only; a run never sets it
_AF3_LAST = {"via": None}


def af3_rows_bucket(m):
    """The carried af3_fused module's rows bucket word (its autotune key MB): 0 (< 65536 rows) | 1 (< 400000) | 2 (>= 400000)."""
    m = int(m)
    return 0 if m < 65536 else (1 if m < 400000 else 2)


def af3_pin(cc, dtype, rows_count, c, hidden, residual):
    """The pinned launch of row af3_fused for (cc, dtype, rows bucket, c, hidden, residual): {'BM','BH','num_warps','num_stages','kernel'} -- the tile the
    carried module's autotuner picked for that key when the cells were measured (table()['launch_pins']['af3_fused']) -- or None (no pin recorded for the key:
    the row refuses that key by name; it never times candidates at run time)."""
    pins = (table().get("launch_pins") or {}).get("af3_fused") or {}
    return pins.get(af3_pin_key(cc, dtype, rows_count, c, hidden, residual))


def af3_pin_key(cc, dtype, rows_count, c, hidden, residual):
    """'<cc>|<dtype>|MB<bucket>|c<c>|h<hidden>|res<0|1>' -- the launch_pins key of row af3_fused."""
    return "%s|%s|MB%d|c%d|h%d|res%d" % (_cc_word(cc), dtype, af3_rows_bucket(rows_count), int(c), int(hidden), int(bool(residual)))


def af3_pinned_buckets(cc, dtype, c, hidden, residual):
    """The rows buckets (0 | 1 | 2) with a recorded launch pin for (cc, dtype, c, hidden, residual)."""
    pins = (table().get("launch_pins") or {}).get("af3_fused") or {}
    pre = "%s|%s|MB" % (_cc_word(cc), dtype); suf = "|c%d|h%d|res%d" % (int(c), int(hidden), int(bool(residual)))
    return sorted(int(k[len(pre)]) for k in pins if k.startswith(pre) and k.endswith(suf))


def _serve_af3(x, W, sel, residual):
    """Row af3_fused: the carried kernel launched with the PINNED tile of the cell table (no candidate timing at run time, ever): a key without a recorded pin
    is refused by name (af3_fused:no_launch_pin:<key>, fallback row); the carried module's autotuned launch runs only under the autotune env word (AF3_AUTOTUNE_ENV, for measuring pins)."""
    import torch
    import triton
    A = carried_module("af3_fused")
    c = W.c; hidden = W.hidden
    xs = x.contiguous().view(-1, c); M = xs.shape[0]
    dtype = {torch.bfloat16: "bf16", torch.float32: "fp32"}.get(x.dtype, str(x.dtype))
    ccd = "%d.%d" % tuple(torch.cuda.get_device_capability(x.device))
    if os.environ.get(AF3_AUTOTUNE_ENV, "") == "1":                                 # the autotune env word ONLY: the carried module's autotuned launch (measuring pins)
        _AF3_LAST["via"] = "autotune"
        return A.fused_transition(x, W.ln_w, W.ln_b, W.wa16, W.wb16, W.wo16, eps=W.eps, residual=bool(residual))
    pin = af3_pin(ccd, dtype, M, c, hidden, residual)
    if pin is None:                                                                # never a live autotune: a key without a recorded launch pin is refused by name
        raise Refusal("af3_fused:no_launch_pin:%s" % af3_pin_key(ccd, dtype, M, c, hidden, residual), sel.word, _fallback_of("af3_fused"))
    if hidden % 64 != 0 or W.wa16.shape != (hidden, c) or W.wo16.shape != (c, hidden):
        raise Refusal("af3_fused:weights_%s_%s" % (tuple(W.wa16.shape), tuple(W.wo16.shape)), sel.word, "torch_swiglu")
    CP = triton.next_power_of_2(c)
    kern = A._ft_pow2 if (CP == c and c <= 128) else A._ft_pad                       # the carried module's own kernel choice; .fn = the kernel under its autotuner
    y = torch.empty((M, c), device=x.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, int(pin["BM"])),)
    kern.fn[grid](xs, y, W.ln_w, W.ln_b, W.wa16, W.wb16, W.wo16, M, af3_rows_bucket(M), W.eps, C=c, CP=CP, HID=hidden, RESIDUAL=bool(residual),
                  BM=int(pin["BM"]), BH=int(pin["BH"]), num_warps=int(pin["num_warps"]), num_stages=int(pin["num_stages"]))
    _AF3_LAST["via"] = "pin:bm%dbh%dw%ds%d" % (int(pin["BM"]), int(pin["BH"]), int(pin["num_warps"]), int(pin["num_stages"]))
    return y.view(x.shape)


def _serve_v2(x, W, sel, residual, mask, binary_mask, out, lead):
    import torch
    V2 = carried_module("v2")
    c = W.c
    x2 = x.view(-1, c) if (out is not None and out is x) else x.reshape(-1, c)
    P = W.pack_for("v2", lambda: V2.pack_weights(W.ln_w, W.ln_b, W.wa16, W.wb16, W.wo16, W.eps, x2.device))
    out2 = out.view(-1, c) if out is not None else torch.empty(x2.shape, dtype=torch.bfloat16, device=x2.device)
    m1 = mask.reshape(-1) if mask is not None else None
    if m1 is not None and m1.numel() != x2.shape[0]:
        raise Refusal("v2:mask_rows_%d_for_%d_rows" % (m1.numel(), x2.shape[0]), sel.word, "torch_swiglu")
    numerics = "fast" if sel.variant == "fast" else "exact"
    try:
        cc_dev = tuple(torch.cuda.get_device_capability(x2.device)) if x2.is_cuda else None
        inherited_ = any(w_ == "inherited_cc:unmeasured" for w_ in (sel.words or ()))
        if sel.cfg is None and cc_dev is not None and cc_dev not in V2.SERVED_CC and not inherited_:
            raise Refusal("v2:no_measured_launch_on_cc_%d.%d_for_%dx%d" % (cc_dev[0], cc_dev[1], int(c), int(W.hidden)), sel.word, "torch_swiglu",
                          "a v2 word without a launch configuration names the package's cc 9.0 tile; on this part only measured launch words (v2@...) serve")
        if sel.cfg is None and cc_dev is not None and cc_dev not in V2.SERVED_CC and (int(c), int(W.hidden)) in V2.CONFIGS:
            _v2_checks(V2, x2, P, m1, out2)                        # an UNMEASURED capability inheriting a package cell: the package's cc 9.0 tile launched explicitly
            V2.launch(x2, P, m1, bool(residual), out2, dict(V2.CONFIGS[(int(c), int(W.hidden))]), binary_mask=binary_mask, numerics=numerics)
        elif sel.cfg is None:                                      # cc 9.0, a package cell: the package's own checks and configuration
            V2.fused_transition(x2, P, mask1d=m1, residual=bool(residual), out2d=out2, binary_mask=binary_mask, numerics=numerics)
        else:                                                      # an explicit launch word (the measured sweep's cells, cc 8.0 included): the package's operand checks minus its capability gate
            _v2_checks(V2, x2, P, m1, out2)
            V2.launch(x2, P, m1, bool(residual), out2, dict(sel.cfg), binary_mask=binary_mask, numerics=numerics)
    except V2.Unsupported as e:
        raise Refusal("v2:%s" % e.reason, sel.word, "torch_swiglu")
    except Exception as e:                                         # noqa: BLE001 -- a launch word whose tile cannot build on this part (shared memory) is refused by name
        from opt_core.oom import is_oom
        if is_oom(e):
            raise
        from opt_core.kernels import safe_settings as _ss
        if isinstance(e, _ss.catchable()) and _ss.is_build_failure(e):
            raise Refusal("v2:build_failed:%s:%s" % (cfg_word(sel.cfg), type(e).__name__), sel.word, "torch_swiglu")
        raise
    return out2.view(lead + (c,)) if out is not None else out2.reshape(lead + (c,))


def _v2_checks(V2, x2d, P, mask1d, out2d):
    """The carried kernel's operand words (dtype, strides, overlap, mask, rows) for an explicit launch on a capability the package itself does not
    configure: the same conditions its _check states, minus served()."""
    import torch
    if x2d.dtype != torch.bfloat16:
        raise V2.Unsupported("x-dtype-%s" % x2d.dtype)
    if x2d.dim() != 2 or x2d.shape[1] != P.c:
        raise V2.Unsupported("x-shape-%s-c%d" % (tuple(x2d.shape), P.c))
    M = x2d.shape[0]
    if M >= V2_ROW_LIMIT:
        raise V2.Unsupported("rows")
    if x2d.stride(1) != 1 or (M > 1 and x2d.stride(0) < P.c):
        raise V2.Unsupported("x-strides-%s" % (tuple(x2d.stride()),))
    if P.wa.device != x2d.device:
        raise V2.Unsupported("packed-on-%s-x-on-%s" % (P.wa.device, x2d.device))
    if out2d.device != x2d.device or out2d.dtype != torch.bfloat16 or tuple(out2d.shape) != tuple(x2d.shape) or out2d.stride(1) != 1 or (M > 1 and out2d.stride(0) < P.c):
        raise V2.Unsupported("out-%s-%s-%s" % (out2d.dtype, tuple(out2d.shape), tuple(out2d.stride())))
    if out2d.data_ptr() != x2d.data_ptr() and _overlap(out2d, x2d):
        raise V2.Unsupported("out-partially-overlaps-x")
    if mask1d is not None:
        if mask1d.device != x2d.device or mask1d.dim() != 1 or mask1d.numel() != M:
            raise V2.Unsupported("mask-shape-%s-M%d" % (tuple(mask1d.shape), M))
        if mask1d.dtype not in V2.MASK_DTYPES:
            raise V2.Unsupported("mask-dtype-%s" % mask1d.dtype)


def _overlap(a, b):
    """True when the byte spans of two 2-D tensors intersect (a partial overlap of out and x is refused; identity = in place is allowed)."""
    if a.numel() == 0 or b.numel() == 0:
        return False

    def span(t):
        lo = t.data_ptr()
        return lo, lo + t.element_size() * (sum((n - 1) * st for n, st in zip(t.shape, t.stride())) + 1)
    a0, a1 = span(a)
    b0, b1 = span(b)
    return not (a1 <= b0 or b1 <= a0)


def _serve_pf(x, W, sel, residual, x_ln, lead):
    import torch
    PF = carried_module("pf")
    T = W.pack_for("pf", lambda: PF.pack_transition_weights(w_out=W.w_o, ln_w=W.ln_w, ln_b=W.ln_b, w_a=W.w_a, w_b=W.w_b, eps=W.eps, device=x.device))
    row, var = sel.row, sel.variant
    if row == "pf":
        impl = var or "fpf"
        kw = dict(impl=impl, ln="fused", residual=bool(residual))
    elif var == "lnfused":
        kw = dict(impl="fpf", ln="fused", residual=bool(residual), variant="v1" if x.dtype == torch.bfloat16 else None)
    else:                                                          # v1: the caller's LayerNorm output is projected (LN_MODE 0)
        if x_ln is None:
            import torch.nn.functional as F
            x_ln = F.layer_norm(x.float(), (W.c,), W.ln_w, W.ln_b, W.eps).to(torch.bfloat16) if x.dtype == torch.bfloat16 else None
            if x_ln is None:
                raise Refusal("v1:ln_given_needs_bf16_rows_or_x_ln", sel.word, "torch_swiglu")
        elif x_ln.dtype != torch.bfloat16:                            # the autocast form hands LayerNorm's fp32 result to a bf16 Linear: the same bf16 rounding, here
            x_ln = x_ln.to(torch.bfloat16)
        kw = dict(impl="fpf", ln="stock", x_ln=x_ln, residual=bool(residual), variant="v1")
        if var == "liger":
            kw["silu"] = "liger"
    try:
        return PF.transition(x, T, **kw)
    except PF.Unsupported as e:
        raise Refusal("%s:%s" % (row, e.reason), sel.word, "torch_swiglu", getattr(e, "words", "") or "")


FLASH_STORE = "flash_prebuilt"          # this package's own prebuilt store for the sealed unit's source, one binary per interpreter ABI key (the sealed
                                        # directory carries the unit's original cp311 binary under its (torch, CUDA) key and stays byte-identical to its kit)
_FLASH_EXT = {}


def flash_abi_key():
    """'torch<torch.__version__>-<CPython SOABI>-sm90' of this process: the key of FLASH_STORE binaries (torch build + interpreter ABI + arch)."""
    import sys
    import sysconfig
    import torch
    soabi = sysconfig.get_config_var("SOABI") or ("cpython-%d%d" % sys.version_info[:2])
    return "torch%s-%s-sm90" % (torch.__version__, soabi)


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def flash_store_manifest():
    """FLASH_STORE/manifest.json parsed ({} when the store is empty): {'module_name', 'sources': {csrc file: sha256}, 'binaries': {abi key: {...}}}."""
    p = os.path.join(_HERE, FLASH_STORE, "manifest.json")
    if not os.path.isfile(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _sealed_prebuilt_refusal(FT):
    """The reason the sealed unit's prebuilt for this stack is refused by its SHA256SUMS (the unit's loader is then not called: the store binary is
    tried, as when the sealed directory has no build for the stack), else None."""
    d = FT.prebuilt_dir()
    man_p = os.path.join(d, "manifest.json")
    if not os.path.isfile(man_p):
        return None
    try:
        with open(man_p, encoding="utf-8") as f:
            so = os.path.join(d, json.load(f)["so"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not os.path.isfile(so):
        return None
    from opt_core.gates import binary_refusal  # noqa: PLC0415  (standard library only; the face imports nothing of the core at module level)
    why = binary_refusal(so)
    return None if why is None else "prebuilt %s refused: %s" % (os.path.basename(so), why)


def _flash_ext():
    """The sealed unit's extension module for this process: the sealed loader first (its own (torch, CUDA) directory and python tag); when it refuses,
    the binary of FLASH_STORE for flash_abi_key() -- built from the SAME csrc (source digests must equal the sealed manifest's), digest-checked, loaded
    under the sealed module name.  Raises RuntimeError('flash_transition: ...') naming both reasons when neither serves."""
    import importlib.machinery
    import importlib.util
    import torch
    FT = carried_module("flash_sm90a")
    _PRIMED_ROWS.add("flash_sm90a")                                # the load (and its check) is attempted now: later calls do device work only or refuse by name
    why_sealed = _sealed_prebuilt_refusal(FT)
    if why_sealed is None:
        try:
            return FT.load()
        except RuntimeError as e:
            why_sealed = str(e).replace("flash_transition: ", "")
    key = flash_abi_key()
    if key in _FLASH_EXT:
        return _FLASH_EXT[key]
    man = flash_store_manifest()
    ent = (man.get("binaries") or {}).get(key)
    if ent is None:
        raise RuntimeError("flash_transition: no prebuilt for %s (sealed loader: %s)" % (key, why_sealed))
    so = os.path.join(_HERE, FLASH_STORE, ent["so"])
    if not os.path.isfile(so):
        raise RuntimeError("flash_transition: store binary missing for %s" % key)
    from opt_core.gates import binary_refusal  # noqa: PLC0415
    why = binary_refusal(so)
    if why:
        raise RuntimeError("flash_transition: store binary for %s refused: %s" % (key, why))
    if ent.get("torch") != torch.__version__ or ent.get("cuda") != torch.version.cuda:
        raise RuntimeError("flash_transition: store binary built for torch %s / CUDA %s, running %s / %s" % (ent.get("torch"), ent.get("cuda"), torch.__version__, torch.version.cuda))
    if _sha256(so) != ent.get("so_sha256"):
        raise RuntimeError("flash_transition: store binary digest differs from the store manifest (%s)" % key)
    sealed = json.load(open(os.path.join(_HERE, "flash_sm90a", "prebuilt", FT.stack_key(), "manifest.json"), encoding="utf-8")) if os.path.isfile(
        os.path.join(_HERE, "flash_sm90a", "prebuilt", FT.stack_key(), "manifest.json")) else {}
    for fn, digest in (man.get("sources") or {}).items():
        src = os.path.join(_HERE, "flash_sm90a", "csrc", fn)
        if not os.path.isfile(src) or _sha256(src) != digest:
            raise RuntimeError("flash_transition: store binary was built from a different csrc/%s than the sealed unit carries" % fn)
        if sealed and (sealed.get("source_sha256") or {}).get(fn) not in (None, digest):
            raise RuntimeError("flash_transition: store source digest of csrc/%s differs from the sealed manifest" % fn)
    name = man.get("module_name") or "flash_transition_sm90a"
    loader = importlib.machinery.ExtensionFileLoader(name, so)
    spec = importlib.util.spec_from_file_location(name, so, loader=loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    for sym in ("flash_transition", "flash_transition_ln", "silu_table", "smem_bytes", "hidden_chunk"):
        if not hasattr(mod, sym):
            raise RuntimeError("flash_transition: store binary lacks symbol %s" % sym)
    _FLASH_EXT[key] = mod
    return mod


def _flash_pack(W, E, device):
    """The kernel's operand pack from the canonical Weights (the sealed unit's layout: W1 rows interleaved a/b per hidden chunk, bf16; W_o bf16;
    LayerNorm affine bf16), built once per (Weights, device) -- no parameter version counters are read (inference-mode tensors carry none)."""
    import torch

    def build():
        bh = int(E.hidden_chunk())
        with torch.no_grad():
            wa16 = W.wa16.to(device); wb16 = W.wb16.to(device)
            nh, c_in = wa16.shape
            if nh % bh:
                raise Refusal("flash_sm90a:hidden_%d_not_multiple_of_%d" % (nh, bh), "flash_sm90a", "v1")
            w1p = torch.stack([wa16.reshape(nh // bh, bh, c_in), wb16.reshape(nh // bh, bh, c_in)], dim=1).reshape(2 * nh, c_in).contiguous()
            wo16 = W.wo16.to(device).contiguous()
            lnw = W.ln_w.to(device=device, dtype=torch.bfloat16).contiguous()
            lnb = W.ln_b.to(device=device, dtype=torch.bfloat16).contiguous()
        return {"w1p": w1p, "wo": wo16, "nh": int(nh), "lnw": lnw, "lnb": lnb, "eps": float(W.eps)}
    return W.pack_for("flash@%s" % device, build)


def _flash_device_ok(E, device):
    """(ok, word): compute capability 9.0 and enough opt-in shared memory per block for the kernel."""
    import torch
    idx = device.index if device.index is not None else torch.cuda.current_device()
    cc = torch.cuda.get_device_capability(idx)
    props = torch.cuda.get_device_properties(idx)
    smem = getattr(props, "shared_memory_per_block_optin", None)
    if cc != (9, 0):
        return False, "cc%d%d-unsupported" % cc
    if smem is not None and smem < int(E.smem_bytes()):
        return False, "smem_%d_lt_%d" % (smem, int(E.smem_bytes()))
    return True, "cc90"


def _serve_flash(x, W, sel, residual, x_ln, out, lead):
    import torch
    import torch.nn.functional as F
    try:
        E = _flash_ext()
    except RuntimeError as e:
        raise Refusal("flash_sm90a:%s" % _short(str(e).replace("flash_transition: ", "")), sel.word, "v1", str(e)[:300])
    if not x.is_cuda:
        raise Refusal("flash_sm90a:x_not_cuda", sel.word, "torch_swiglu")
    ok, why = _flash_device_ok(E, x.device)
    if not ok:
        raise Refusal("flash_sm90a:%s" % str(why)[:80].replace(" ", "_"), sel.word, "torch_swiglu")
    pk = _flash_pack(W, E, x.device)
    c = W.c
    x2 = x.reshape(-1, c)
    M = x2.shape[0]
    if M >= 2 ** 31:
        raise Refusal("flash_sm90a:rows_int32", sel.word, "torch_swiglu")
    out2 = out.reshape(-1, c) if out is not None else torch.empty((M, c), dtype=torch.bfloat16, device=x.device)
    if M == 0:
        return out2.reshape(lead + (c,))
    res = None
    if residual:
        res = x2 if x2.dtype == torch.bfloat16 else x2.to(torch.bfloat16)
        if res.stride(-1) != 1 or res.stride(0) != c:
            res = res.contiguous()
    if sel.variant == "kernel_ln":                                 # LayerNorm in the kernel prologue (the Protenix fast-LN arithmetic): raw bf16 rows in
        xr = x2 if x2.dtype == torch.bfloat16 else x2.to(torch.bfloat16)
        if xr.stride(-1) != 1 or (xr.stride(0) * 2) % 16 or xr.data_ptr() % 16:
            xr = xr.contiguous()
        E.flash_transition_ln(xr, pk["w1p"], pk["wo"], res, out2, pk["lnw"], pk["lnb"], float(W.eps))
        return out2.reshape(lead + (c,))
    if x_ln is not None:
        y = x_ln.reshape(-1, c)
    else:                                                          # the caller's form: LayerNorm with fp32 parameters (autocast semantics), bf16 result
        y = F.layer_norm(x2.float(), (c,), W.ln_w, W.ln_b, W.eps)
    if y.dtype != torch.bfloat16:
        y = y.to(torch.bfloat16)
    if y.dim() != 2 or y.stride(-1) != 1 or (y.stride(0) * 2) % 16 or y.data_ptr() % 16:
        y = y.contiguous()
    E.flash_transition(y, pk["w1p"], pk["wo"], res, out2)
    return out2.reshape(lead + (c,))


def _t15_pack(W, P):
    ns = type("M", (), {})
    norm = ns(); norm.weight = W.ln_w; norm.bias = W.ln_b
    ffn = ns(); ffn.hidden_features = W.hidden
    import torch
    w12 = ns(); w12.weight = torch.cat([W.wa16, W.wb16], 0)
    w3 = ns(); w3.weight = W.wo16
    ffn.w12, ffn.w3 = w12, w3
    return P.pack_transition(norm, ffn)


def _serve_t15(x, W, sel, lead):
    import torch
    P = carried_module("esm_t15")
    if abs(W.eps - 1e-5) > 1e-12:
        raise Refusal("esm_t15:eps_%g_not_1e-5" % W.eps, sel.word, "engine_module")
    try:
        cfg, info = P.select_t15_cfg(x.device)
    except RuntimeError as e:
        raise Refusal("esm_t15:%s" % str(e)[:80].replace(" ", "_"), sel.word, "engine_module")
    pk = W.pack_for("t15", lambda: _t15_pack(W, P))
    x2 = x.reshape(-1, W.c)
    if x2.dtype != torch.bfloat16 or not x2.is_contiguous():
        raise Refusal("esm_t15:rows_bf16_contiguous", sel.word, "engine_module")
    with torch.autocast("cuda", enabled=False):
        out = P.transition_v2(x2, pk, cfg)
    return out.view(lead + (W.c,))


def _t16_live(sel_word, fallback):
    """The carried sm_90a forward's module, engaged once per process (cubin manifest checks + install canary); Refusal by name when it cannot serve."""
    X = carried_module("esm_t16")
    d = X.engage()
    _PRIMED_ROWS.add("esm_t16")                                    # engaged (live or stepped aside): no host-side work remains for later calls
    if not X.live():
        raise Refusal("esm_t16:%s" % str(d.get("reason") or "not_live")[:60].replace(" ", "_"), sel_word, fallback, str(d.get("reason_text") or "")[:120])
    return X


def _serve_t16(x, W, sel, residual, out, lead):
    import torch
    X = _t16_live(sel.word, "torch_swiglu")
    w = _kd3_weights(W, None)
    if not X.servable_weights(w):
        raise Refusal("esm_t16:weights_shape", sel.word, "torch_swiglu")
    pk = W.pack_for("t16", lambda: X._pack_tensors(w["W12"], w["W3"], w["LN_W32"], w["LN_B32"], float(W.eps)))
    x2 = x.reshape(-1, W.c)
    if x2.dtype != torch.bfloat16 or not x2.is_contiguous():
        raise Refusal("esm_t16:rows_bf16_contiguous", sel.word, "torch_swiglu")
    if x2.shape[0] < 1:
        raise Refusal("esm_t16:zero_rows", sel.word, "torch_swiglu")
    o2 = out.reshape(-1, W.c) if out is not None else None
    try:
        y = X.transition_cute(x2, pk, out=o2, residual=bool(residual))
    except RuntimeError as e:
        raise Refusal("esm_t16:%s" % str(e)[:80].replace(" ", "_"), sel.word, "torch_swiglu")
    return y.view(lead + (W.c,))


def _serve_esm_fused(x, W, sel, residual, out, lead):
    """Row esm_fused_exact: the ESM-family fused inference statement in one kernel (this package's .esm_fused): x + T(x) when ``residual`` (the
    statement), T(x) alone otherwise (the same kernel without the residual epilogue); ``out`` may be given (and may be x: in place)."""
    import torch
    E = carried_module("esm_fused_exact")
    x2 = x.reshape(-1, W.c)
    if x2.dtype != torch.bfloat16:
        raise Refusal("esm_fused_exact:x_dtype:%s" % str(x2.dtype).replace("torch.", ""), sel.word, "torch_swiglu")
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    try:
        cfg, _name = E.select_cfg(x.device)
    except RuntimeError as e:
        raise Refusal("esm_fused_exact:%s" % _short(e), sel.word, "torch_swiglu", str(e))
    if not E.admits_shape(W.c, W.hidden, cfg["BH"]):
        raise Refusal("esm_fused_exact:cell_%dx%d_outside_launch_row" % (W.c, W.hidden), sel.word, "torch_swiglu")
    o2 = None
    if out is not None:
        o2 = out.reshape(-1, W.c)
        if o2.dtype != torch.bfloat16 or not o2.is_contiguous() or tuple(o2.shape) != tuple(x2.shape):
            raise Refusal("esm_fused_exact:out_layout", sel.word, "torch_swiglu", "out must be bf16, contiguous, same rows as x")
    pk = W.pack_for("esm_fused", lambda: E.pack_weights(W))
    with torch.autocast("cuda", enabled=False):
        y = E.fused_transition(x2, pk, residual=bool(residual), out=o2, cfg=cfg)
    return out if out is not None else y.view(lead + (W.c,))


def _kd3_weights(W, A, key="kd3"):
    import torch
    return W.pack_for(key, lambda: {"W12": torch.cat([W.wa16, W.wb16], 0).contiguous(), "W3": W.wo16, "LN_W32": W.ln_w, "LN_B32": W.ln_b, "ver": (key,)})


def transition_autograd(x, W, *, word, n_tokens=None, timing="eager", stack=None, _sel=None):
    """Serve a DIFFERENTIABLE residual transition x + T(x) (rows esm_kd3 | esm_kd3:lean | esm_t15_kd3 | torch_swiglu; frozen weights:
    dX only) -> (y, Selection).  Under torch.no_grad the same statements run without a graph node."""
    W._policy = pack_policy(word if _sel is None else _sel)
    import torch
    c = int(x.shape[-1])
    dtype = {torch.bfloat16: "bf16", torch.float32: "fp32"}.get(x.dtype, str(x.dtype).replace("torch.", ""))
    sel = _sel or select(word, c=c, hidden=W.hidden, n_tokens=n_tokens, dtype=dtype, direction="fwdbwd", timing=timing, family="esmpair", residual=True,
                         device=x.device if x.is_cuda else None, stack=stack, rows_count=x.numel() // max(c, 1))
    row, var = sel.row, sel.variant
    if row == "torch_swiglu":
        return _torch_swiglu(x, W, True, None, None, autocast=True), sel
    if row in ("engine_module", "compile"):
        raise Refusal("callers_own_call:%s" % row, word, "torch_swiglu")
    if row not in ("esm_kd3", "esm_t16_kd3", "esm_t15_kd3"):
        raise Refusal("forward_only:%s" % row, word, _fallback_of(row))
    if any(t.requires_grad for t in (W.w_a, W.w_b, W.w_o)):
        raise Refusal("%s:frozen_weights_only" % row, word, "engine_module")
    A = carried_module("esm_kd3")
    if not getattr(A, "_KD3_OK", False):
        raise Refusal("esm_kd3:kernels_unavailable:%s" % str(getattr(A, "_EXP", "?"))[:60].replace(" ", "_"), word, "engine_module")
    if abs(W.eps - float(A.C._EPS)) > 1e-12:
        raise Refusal("%s:eps_%g_not_the_forks_%g" % (row, W.eps, float(A.C._EPS)), word, "engine_module")
    key = row + (":" + var if var else "")
    w = _kd3_weights(W, A, key)                                    # one weight entry per row word: the hook's forward closure and the sm_90a operand pack are cached in it
    if "lean" not in w:
        w["lean"] = (row != "esm_kd3") or (var == "lean")
        w["fwd"] = None
        if row == "esm_t16_kd3":                                   # K-D3 lean's forward on the carried sm_90a kernel: the design kit's own composition (forward_for over this entry)
            X = _t16_live(word, "esm_kd3")
            fwd = X.forward_for(w, float(A.C._EPS))
            if fwd is None:
                w.pop("lean")
                raise Refusal("esm_t16:not_servable:%s" % ",".join(sorted(X.stats().get("fallback", {}))), word, "esm_kd3")
            w["fwd"] = fwd
        elif row == "esm_t15_kd3":                                 # the same hook over the Triton one-kernel forward (its sm80 row serves cc 8.0, where the sm_90a kernel cannot)
            P = carried_module("esm_t15")
            try:
                cfg, info = P.select_t15_cfg(x.device)
            except RuntimeError as e:
                w.pop("lean")
                raise Refusal("esm_t15:%s" % str(e)[:80].replace(" ", "_"), word, "esm_kd3")
            pk = W.pack_for("t15", lambda: _t15_pack(W, P))

            def fwd(x2d, _P=P, _pk=pk, _cfg=cfg):
                with torch.autocast("cuda", enabled=False):
                    return _P.transition_v2(x2d, _pk, _cfg)
            w["fwd"] = fwd
    xb = x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)
    if torch.is_grad_enabled() and x.requires_grad:
        return A.TransitionRefround.apply(xb, w), sel
    with torch.no_grad():
        return A.TransitionRefround.forward(type("ctx", (), {"save_for_backward": lambda *a, **k: None})(), xb, w), sel

_WARMED = False                                                 # opt_core.warm.auto_warm() ran from this face's first serving call
