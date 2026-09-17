"""opt_core.attn.pair_fused — the pair track's fused surround, engine-agnostic: triangle-attention PROLOGUE (LayerNorm -> q|k|v|g|pair-bias
projections in one kernel), attention CORE dispatch (the core's flash_triattn cell, the K2B cell core='k2b', the row kernels.triattn selects for a word core='tier:<word>' -- flash_triattn by name when it refuses --, the block's own per-class choice core='default' = the functions' default (DEFAULT_CORE_TABLE: tier:fast where measured faster, flash_triattn elsewhere), or a vendor kernel kept by name), EPILOGUE (sigmoid gate *
o @ W_o^T [+ residual, transposed scatter for the ending node] in one kernel), and the fused SwiGLU TRANSITION (LN -> W_a|W_b -> silu(a)*b -> W_out
[+ residual]; the n*C hidden never reaches HBM).  Standard library at import; torch / triton are imported inside the calls.

Two implementations, selected BY NAME so a kit that re-points keeps its shipped numerics:
    impl='fpf'  the FlashPairformer Fusion kernels (carried cells fpf_triatt_pro, fpf_triatt_epi, fpf_transition; kernel variants from
                fpf_glue_v2 / fpf_mkpf named per cell row) — the reference recipe;
    impl='lnl'  the lnl_fused kernels (ln_linear prologue + one cuBLAS q|k|v|g GEMM, gate_transpose epilogue + cuBLAS W_o, fused_transition);
    impl='torch' plain torch around the attention core (LN -> one fused q|k|v|g GEMM + bias GEMM; sigmoid-gate product + W_o GEMM), stock
                rounding points, ANY C / H / D the core attention accepts — the named construction for non-power-of-two dims (c_z=384, 12 heads)
                and the A/B baseline; no cell rows (nothing to pin).
Pair-bias rounding (all impls): bias = fp32(bf16 GEMM output of the bf16 LN output) — the autocast-stock rounding (exact-eligible where the
engine's stock projects the bias under autocast; an engine whose stock projects the bias in fp32 would need a variant by name — none does today).

Cells.  A cell is (impl, piece, shape key, compute capability, triton major.minor) -> one pinned launch configuration + its status; the table is
``pair_fused_cells.json`` beside this module (``cells()``).  Resolution (:func:`lookup_cell`, the one place): a ``certified`` row serves (``candidate``
rows too under ``OPT_CORE_PF_ALLOW_CANDIDATE=1``, qualification jobs); a key the table lists as MEASURED OFF — a ``named_off`` entry or a ``candidate`` row
not admitted: shapes the measured compositions run with the lever off — stays on the stock statements BY NAME (its ``no-cell:…`` word qualified
``+off(not-measured)``, ``opt_core.kernels.cell_words``); any OTHER key is UNKNOWN and ENGAGES the lever's SAFE settings for the capability (``opt_core.kernels.safe_settings``;
served ``safe``, ONE line ``safe settings served (no_cell:<key>, …)``) or, for ``impl='lnl'`` (no cell configuration: lnl_fused resolves its own tiles),
its default settings (served ``default``, ONE info line); ``no-cell:...`` remains only where no safe settings admit the shape (a variant without a safe
cell, a shape class the safe rows exclude BY NAME, e.g. ``+no_safe(c_z_above_128_slower_than_stock_on_cc8.0)``).  Every refusal is an :class:`Unsupported`
raised BEFORE any launch, with a short reason token the kit books as ``fallback:<reason>`` on its ledger.  ``lookup_cell`` / ``preflight`` answer the same
question without raising (a :class:`CellDecision`) so a kit decides its named routes ONCE at activation instead of meeting the words mid-forward.

Canonical weights (any engine maps its module's parameters onto these keys once; packed bf16 and cached on the returned object):
    tri-attention  TRIATTN_KEYS      ln_w ln_b [C] (None = no affine) · w_q w_k w_v w_g [H*D, C] (or one fused w_qkvg [4*H*D, C]) · b_g [H*D] (gate bias, optional)
                                     · w_b [H, C] (pair bias) · w_o [C_out, H*D] · b_o [C_out] (optional)
    transition     TRANSITION_KEYS   ln_w ln_b [C] · w_a w_b [n*C, C] (or fused w_ab [2*n*C, C]) · w_out [C, n*C]

Frames.  ``z`` is the pair tensor [N, N, C] (or [B, N, N, C]) in its OWN frame.  ``ending=True`` runs the attention in the transposed frame x = z^T:
the fpf prologue reads z^T by address math (no copy) and the epilogue scatters the update back untransposed; ``bias_frame`` says which frame the
pair bias is projected from — 'x' (the common convention: bias from the transposed, normalised x) or 'z' (bias from norm(z) before
the transpose; == the x-frame bias with its two token axes swapped, served as a strided view).

Usage (a kit's triangle-attention patch; LEVER is the kit's opt_core ledger object for the lever line):

    from opt_core.attn import pair_fused as PF
    W = PF.pack_triattn_weights(ln_w=m.layer_norm.weight, ln_b=m.layer_norm.bias, w_q=m.linear_q.weight, w_k=m.linear_k.weight,
                                w_v=m.linear_v.weight, w_g=m.linear_g.weight, w_b=m.linear.weight, w_o=m.linear_o.weight, n_heads=4, head_dim=32)
    refused = None
    try:
        u = PF.tri_attn_block(z, W, mask=pair_mask, ending=not starting, residual=False)      # update in z's frame
    except PF.Unsupported as e:                     # L = PF.ledger(expected=(...)) once per process; PF.emit_line(L, tag) at exit
        refused = e.reason                          # keep the WORD, never the exception (a bound exception keeps this frame's tensors alive until GC)
    if refused is not None:
        L.fallback(refused); u = stock_forward(z)   # the stock path runs OUTSIDE the except block
    else:
        L.serve(f"c{W.C}h{W.H}d{W.D}")
"""
import importlib
import json
import math
import sys
import os
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

__all__ = ["Unsupported", "TRIATTN_KEYS", "TRANSITION_KEYS", "pack_triattn_weights", "pack_transition_weights", "transition_plan_words", "cells", "cells_sha256",
           "find_cell", "pick_cell", "lookup_cell", "preflight", "CellDecision", "carried_exports", "supported_triattn", "supported_transition", "prologue", "core_attention", "epilogue", "tri_attn_block", "transition", "ln_linear",
           "ledger", "emit_line", "evidence", "evidence_tail", "describe", "allow_candidate", "SERVED_BY_STATUS", "SERVED_VARIANTS", "SERVED_KEYS", "SERVED_CORES", "IMPLS", "CORES", "TIER_CORE_PREFIX", "DEFAULT_CORE", "DEFAULT_CORE_TABLE", "DEFAULT_CORE_OFF_WORDS", "default_core_for", "tier_core_word", "resolve_tier_core",
           "Refused", "SAFE_LEVERS", "SAFE_DIMS", "DEFAULT_IMPLS", "safety_net", "run_cell", "row_key", "shape_dims", "safe_settings_for_shape", "named_off_entry",
           "EXACT_ROWS_PREFIX", "exact_rows_rule", "exact_rows_word", "exact_rows_facts"]

_HERE = os.path.dirname(os.path.abspath(__file__))
CELLS_PATH = os.path.join(_HERE, "pair_fused_cells.json")
ALLOW_CANDIDATE_ENV = "OPT_CORE_PF_ALLOW_CANDIDATE"          # qualification jobs only: serve rows whose status is 'candidate'
IMPLS = ("fpf", "lnl", "torch")                              # 'torch' = plain-torch prologue/epilogue (cuBLAS GEMMs, stock rounding points) around the core: the declared construction for dims the fused kernels do not serve (non-power-of-two C / H*D), and the A/B baseline
CORES = ("flash_triattn", "k2b")                             # named attention cores of the core; a callable keeps a vendor kernel by name
DEFAULT_CORE = "default"                                      # core="default" (the functions' default): the block's OWN choice per call class -- DEFAULT_CORE_TABLE below
                                                              # (tier:fast where the block was measured faster with kernels.triattn's row, flash_triattn elsewhere)
TIER_CORE_PREFIX = "tier:"                                    # core="tier:<word>": the attention core kernels.triattn selects for <word> (a tier word fast | big | exact, or one of its
                                                              # row words) at this call's (cc, dtype, head_dim, heads, keys, mask form) -- OPT-IN; the default core stays flash_triattn
TRIATTN_KEYS = ("ln_w", "ln_b", "w_q", "w_k", "w_v", "w_g", "b_g", "w_b", "w_o", "b_o")
TRANSITION_KEYS = ("ln_w", "ln_b", "w_a", "w_b", "w_out")
DEFAULT_CORE_TABLE: Dict[Tuple[str, str, int, int], int] = {          # (cc, dtype, head_dim, heads) -> keys per row at or above which core="default" serves tier:fast; below it,
    ("8.0", "bf16", 16, 4): 512,                                       # and for every class not listed, the flash_triattn cell (today's core).  Measured, whole block vs the
    ("8.0", "bf16", 32, 4): 512,                                       # flash_triattn core, same process: 8.0 D16 x1.16-1.44 @512 / x1.27-2.75 @1024, D32 x1.09-1.66 / x1.20-2.29;
    ("9.0", "bf16", 32, 4): 512,                                       # 9.0 D32 x1.11-1.29 @512 / x1.29-1.67 @1024, D16 x1.19-1.53 @1024 (9.0 D16 @512 x0.95-1.17 and every
    ("9.0", "bf16", 16, 4): 1024,                                      # class @256 straddle parity: flash_triattn stays)
}
DEFAULT_CORE_OFF_WORDS = ("pair_fused:tier_core", "pair_fused_tier_core")   # MODEL_OPT_LEVERS_OFF words switching the table off (core="default" = flash_triattn everywhere: ablation)
_CARRIED = ("fpf_triatt_pro", "fpf_triatt_epi", "fpf_transition", "fpf_transition_v2", "fpf_glue_v2", "fpf_mkpf", "lnl_fused", "fpf_triatt_k2b", "flash_triattn")


class Unsupported(Exception):
    """A named refusal, raised before any kernel launch. ``reason`` is the short token the kit's ledger carries (``dtype:float32``, ``c:96``,
    ``no-cell:fpf:prologue:256x512:9.0|3.7``, ``mask-shape``, ``import:fpf_triatt_pro`` ...)."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(reason + (": " + detail if detail else ""))
        self.reason = reason
        self.detail = detail


# ------------------------------------------------------------------------------------------------------------------------------- the cell table
_CELLS: Optional[dict] = None


def cells(path: Optional[str] = None) -> dict:
    """The core's cell table (parsed once): {'schema', 'rows': [{impl, piece, key, cc, triton, variant, cfg, status, evidence}, ...]}."""
    global _CELLS
    if path is not None:
        with open(path) as fh:
            return json.load(fh)
    if _CELLS is None:
        with open(CELLS_PATH) as fh:
            _CELLS = json.load(fh)
    return _CELLS


def cells_sha256() -> str:
    import hashlib
    with open(CELLS_PATH, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


_STACK_MEMO: dict = {}                                                   # device word -> (cc, triton): neither changes within a process


def _stack(device) -> Tuple[str, str]:
    """(compute capability 'M.m', triton 'M.m') of the running box (memoised per device: a driver query + a version parse per call otherwise —
    thousands of calls per predicted item on a launch-bound trunk)."""
    import torch
    dkey = ("current", torch.cuda.current_device()) if device is None else str(device)
    got = _STACK_MEMO.get(dkey)
    if got is not None:
        return got
    mj, mn = torch.cuda.get_device_capability(device)
    try:
        import triton
        p = triton.__version__.split(".")
        tt = f"{p[0]}.{p[1]}"
    except Exception:
        tt = "none"
    got = _STACK_MEMO[dkey] = (f"{mj}.{mn}", tt)
    return got


class Refused(RuntimeError):
    """The fail-closed case of a cell lever (opt_core.kernels.safe_settings case iii): even the lever's SAFE settings cannot build / run on
    this box. The message names the lever and the one-flag escape (`--mode off` / the kit's explicit opt-out); a kit turns it into its hard
    error. Never raised for a missing row (that is the safe settings' to serve) and never for anything a running kernel produces."""


SAFE_LEVERS = {("fpf", "transition"): "pair_fused:transition",       # (impl, piece) -> the safe-settings lever name (safe_settings.SAFE_ROWS) whose settings fit this piece's launch
               ("fpf", "prologue"): "pair_fused:prologue", ("fpf", "epilogue"): "pair_fused:epilogue"}
SAFE_VARIANTS = {("fpf", "transition"): ("v1", "v1_fp32x"),          # the kernel variants those settings launch (one launch statement each); other variants have no safe cell
                 ("fpf", "prologue"): ("v3",), ("fpf", "epilogue"): ("v2",)}
SAFE_DIMS = {("fpf", "transition"): ("c_z", "n_hidden"), ("fpf", "prologue"): ("c_z", "H", "D"), ("fpf", "epilogue"): ("c_z", "H", "D")}   # the key's dimension names (SAFE_ROWS `when` bounds)
DEFAULT_IMPLS = ("lnl",)                                                # impls whose rows carry no launch configuration (lnl_fused resolves its own tiles per capability): an UNKNOWN key of theirs
                                                                        # is served the piece's default settings (served_by 'default', ONE info line) — no SAFE row is needed to engage
LNL_NET = "pair_fused:lnl"                                              # the safety net that names the lnl default engagement (SafeNet.inform; the tiles' own net is lnl_fused's)
# The miss policy (ONE place, :func:`lookup_cell`): a servable row serves; a key the table lists as MEASURED OFF (``named_off`` / a ``candidate`` row not
# admitted) is its `no-cell:…` word qualified `+off(not-measured)` — the kit's per-call stock route, counted, exit 0: exactly what those calls run today; any OTHER key is
# UNKNOWN and engages the lever's SAFE settings for the capability (`safe`, ONE line) or its default settings (`default`, impls in DEFAULT_IMPLS);
# `no-cell:…` remains where no safe settings admit the shape (a variant without a safe cell; a shape class the safe rows exclude by name, `+no_safe(<word>)`).
# A hard error (Refused) is only safe_settings case (iii): the lever's kernel cannot BUILD on this box even with its safe settings.
_NETS: Dict[str, Any] = {}


def safety_net(lever: str):
    """This process's safety net of `lever` (opt_core.kernels.safe_settings.SafeNet; its refusal is :class:`Refused`)."""
    net = _NETS.get(lever)
    if net is None:
        from opt_core.kernels import safe_settings as _ss
        net = _NETS[lever] = _ss.SafeNet(lever, refused=Refused)
    return net


def shape_dims(impl: str, piece: str, key: Sequence[int]) -> Dict[str, int]:
    """The key of (impl, piece) by dimension name (SAFE_DIMS): what a safe row's `when` bounds read."""
    return dict(zip(SAFE_DIMS.get((impl, piece), ()), (int(k) for k in key)))


def safe_settings_for_shape(impl: str, piece: str, key: Sequence[int], cc: str) -> Tuple[Optional[dict], Optional[str]]:
    """``(safe row, refusal word)`` of the lever behind (impl, piece) for this key on `cc` (opt_core.kernels.safe_settings.safe_settings_for
    with the key's dimension names): the row that admits the shape, or (None, word) when the capability's safe settings exclude this shape
    class, or (None, None) when the lever has none on the capability."""
    lever = SAFE_LEVERS.get((impl, piece))
    if lever is None:
        return None, None
    from opt_core.kernels import safe_settings as _ss
    return _ss.safe_settings_for(lever, cc, shape_dims(impl, piece, key))


def _safe_row(impl: str, piece: str, key: Sequence[int], cc: str, variant: Optional[str]) -> Tuple[Optional[dict], Optional[str]]:
    """``(row, refusal word)``: the row the lever's SAFE settings make for (impl, piece, key) on capability `cc` — the conservative cell of
    opt_core.kernels.safe_settings.SAFE_ROWS that admits this shape — or (None, word) when the capability has safe settings but none admits this
    shape class (measured slower than the stock statements there), or (None, None) when the piece has no safe settings on the capability or
    `variant` names a kernel variant they do not launch."""
    lever = SAFE_LEVERS.get((impl, piece))
    if lever is None:
        return None, None
    variants = SAFE_VARIANTS[(impl, piece)]
    if variant is not None and variant not in variants:
        return None, None
    srow, refusal = safe_settings_for_shape(impl, piece, key, cc)
    if srow is None:
        return None, refusal
    return ({"id": f"safe:{lever}:{cc}", "impl": impl, "piece": piece, "key": [int(k) for k in key], "cc": cc, "triton": "*",
             "variant": variant if variant is not None else variants[0], "cfg": dict(srow["settings"]), "status": "safe",
             "safe_status": srow.get("status"), "evidence": srow.get("evidence", "")}, None)


def _default_row(impl: str, piece: str, key: Sequence[int], cc: str) -> dict:
    """The row of an impl in DEFAULT_IMPLS for an UNKNOWN key: the piece's default settings (no launch configuration of this module's; the carried
    kernel resolves its own per-capability tiles under its own safety net)."""
    return {"id": f"default:{impl}:{piece}:{cc}", "impl": impl, "piece": piece, "key": [int(k) for k in key], "cc": cc, "triton": "*",
            "variant": piece, "cfg": {}, "status": "default", "evidence": "unknown key: the piece's default settings (P12 default-engage), named once per process"}


def _entry_matches(ent: dict, impl: str, piece: str, key: Sequence[int], cc: str, variant: Optional[str]) -> bool:
    """One ``named_off`` entry against a lookup: every field the entry gives must equal the lookup's (``"*"`` / absent = any; ``key`` a list or ``"*"``;
    an entry that names a ``variant`` matches only a lookup pinned to that variant)."""
    if ent.get("impl", "*") not in ("*", impl) or ent.get("piece", "*") not in ("*", piece) or ent.get("cc", "*") not in ("*", cc):
        return False
    k = ent.get("key", "*")
    if k != "*" and [int(x) for x in k] != [int(x) for x in key]:
        return False
    v = ent.get("variant", "*")
    return v == "*" or (variant is not None and v == variant)


def named_off_entry(impl: str, piece: str, key: Sequence[int], cc: str, variant: Optional[str] = None) -> Optional[dict]:
    """The table's ``named_off`` entry listing (impl, piece, key, cc[, variant]) as MEASURED OFF (first match in file order), else None."""
    for ent in cells().get("named_off", ()):
        if _entry_matches(ent, impl, piece, key, cc, variant):
            return ent
    return None


# ------------------------------------------------------------------------------------------------------------- the exact construction's served rows
# The EXACT construction (impl 'fpf', ln='stock': the caller's own LayerNorm output projected by the fused kernels in ONE summation order) equals
# the stock Linear layers' cuBLAS bf16 GEMMs bit for bit only where cuBLAS keeps that order on the running card. ``cells()["exact_rows"]``
# records, per compute capability, the row counts at which it does not (measured with an engine's det-1 outputs byte-compared against its stock
# run): below ``min_rows`` rows (per piece), and — every call of ``max_rows`` rows and more being worked in pieces of ``piece_rows`` — a remainder
# piece (rows mod piece_rows) outside [min_remainder, piece_rows - top_margin]. Those calls are refused BY NAME before any launch
# (``Unsupported('exact-rows:below_min_rows' | 'exact-rows:above_max_rows')``): the caller's stock statement answers them, which is what keeps an
# exact-class lever bitwise at every input size on that card. ln='fused' (the tolerance-class construction) is never affected; a card without an
# entry has one order at every row count (nothing refused). rows = the call's leading dimensions flattened (a pair call [B, N, N, C] is B*N*N rows).
EXACT_ROWS_PREFIX = "exact-rows:"                                       # Unsupported reason prefix: 'exact-rows:below_min_rows' | 'exact-rows:above_max_rows'
EXACT_ROWS_PIECES = ("transition", "prologue")                          # 'transition' = the fused pair transition; 'prologue' = the fused triangle-attention BLOCK (its prologue
                                                                        # q|k|v|g|bias and epilogue W_o projections share the row count: one decision per block call)


def exact_rows_rule(cc: Optional[str], piece: str) -> Optional[dict]:
    """The served-row rule of the exact construction's ``piece`` ('transition' | 'prologue') on compute capability ``cc`` ('M.m'):
    {min_rows, max_rows, piece_rows, min_remainder, top_margin, below, above, measured_on} or None (no entry for the card: every row count served)."""
    ex = cells().get("exact_rows") or {}
    card = (ex.get("cards") or {}).get(str(cc or ""))
    if not card:
        return None
    p = (card.get("pieces") or {}).get(piece)
    if p is None:
        return None
    words = ex.get("words") or {}
    return {"min_rows": int(p.get("min_rows", 0)), "max_rows": int(card["max_rows"]), "piece_rows": int(card["piece_rows"]),
            "min_remainder": int(card["min_remainder"]), "top_margin": int(card["top_margin"]),
            "below": str(words.get("below", "below_min_rows")), "above": str(words.get("above", "above_max_rows")),
            "measured_on": tuple(card.get("measured_on") or ())}


def exact_rows_word(piece: str, cc: Optional[str], rows: int) -> Optional[str]:
    """The refusal word of an exact-construction call of ``rows`` rows on compute capability ``cc`` — 'below_min_rows' (under the piece's floor),
    'above_max_rows' (at or over max_rows with a remainder piece outside the served band) — or None (served, or the card has no entry)."""
    r = exact_rows_rule(cc, piece)
    if r is None:
        return None
    rows = int(rows)
    if rows < r["min_rows"]:
        return r["below"]
    if rows >= r["max_rows"]:
        rem = rows % r["piece_rows"]
        if not (r["min_remainder"] <= rem <= r["piece_rows"] - r["top_margin"]):
            return r["above"]
    return None


def exact_rows_facts(cc: Optional[str], piece: str) -> Optional[dict]:
    """The rule's numbers for a lever line ({cc, min_rows, max_rows, piece, piece_min, piece_top_margin, below, above}), None without an entry."""
    r = exact_rows_rule(cc, piece)
    if r is None:
        return None
    return {"cc": str(cc), "min_rows": r["min_rows"], "max_rows": r["max_rows"], "piece": r["piece_rows"], "piece_min": r["min_remainder"],
            "piece_top_margin": r["top_margin"], "below": EXACT_ROWS_PREFIX + r["below"], "above": EXACT_ROWS_PREFIX + r["above"]}


def _exact_rows_check(piece: str, t, C: int) -> None:
    """Raise Unsupported by name when the exact construction's call on tensor ``t`` ([..., C]) lies outside the card's served rows."""
    w = exact_rows_word(piece, _stack(t.device)[0], t.numel() // int(C))
    if w is not None:
        r = exact_rows_rule(_stack(t.device)[0], piece) or {}
        raise Unsupported(EXACT_ROWS_PREFIX + w, f"{t.numel() // int(C)} rows on cc {_stack(t.device)[0]}: the stock GEMMs take another summation order there "
                                                 f"(served: {r.get('min_rows')} <= rows < {r.get('max_rows')}, and above it a remainder rows mod {r.get('piece_rows')} within "
                                                 f"[{r.get('min_remainder')}, {r.get('piece_rows', 0) - r.get('top_margin', 0)}]) — the stock statement answers this call")


class CellDecision(NamedTuple):
    """The cell decision for one (impl, piece, key) on one stack. ``row`` is what serves and ``served_by`` says which table key did:
    ``"<cc>|<triton M.m>"`` (the named-exception row), ``"<cc>|*"`` (the capability's row), ``"safe"`` (an UNKNOWN key: the lever's SAFE settings
    serve, opt_core.kernels.safe_settings — ``reason`` = ``no_cell:<key>``), ``"default"`` (an UNKNOWN key of an impl without cell configurations: its
    default settings serve — ``reason`` = ``default:<key>``), or ``""`` with ``row`` None and ``reason`` the word the kit books:
    ``no-cell:<impl>:<piece>:<key>:<cc>|<triton>+off(not-measured)`` (the table lists the key as MEASURED OFF: the stock statements serve it, by name;
    ``off`` is True) or ``no-cell:<impl>:<piece>:<key>:<cc>|<triton>[+no_safe(<why>)]`` (no row and no safe settings admit the shape).  A kit reads it BEFORE any forward (:func:`preflight`): a served decision is an engaged lever
    (``word()`` is its LEVER-line fact); an unserved one is the kit's named per-shape route (or its NOT ACTIVE exit) before anything runs."""
    impl: str
    piece: str
    key: Tuple[int, ...]
    cc: str
    triton: str
    row: Optional[dict]
    reason: str
    served_by: str = ""

    @property
    def served(self) -> bool:
        return self.row is not None

    @property
    def safe(self) -> bool:
        return self.served_by == "safe"

    @property
    def off(self) -> bool:
        """True when the table lists this key as MEASURED OFF (``reason`` carries ``+off(<why>)``)."""
        from opt_core.kernels.cell_words import is_off_word
        return self.row is None and is_off_word(self.reason)

    def word(self, shape: str = "") -> str:
        """The LEVER-line fact: ``served:<variant>`` for a table row; ``safe:no_cell:<shape>`` when the safe settings serve (``shape`` = the
        kit's own naming of the key, default the key itself); ``default:<shape>`` when an impl's default settings serve; ``off:<why>:<shape>`` for a
        key the table lists as measured off; ``refused:<no-cell word>`` when nothing serves."""
        if self.served_by == "safe":
            return "safe:no_cell:" + (shape or "x".join(map(str, self.key)))
        if self.served_by == "default":
            return "default:" + (shape or "x".join(map(str, self.key)))
        if self.row is not None:
            return "served:" + str(self.row.get("variant", "?"))
        if self.off:
            from opt_core.kernels.cell_words import off_why
            return "off:%s:%s" % (off_why(self.reason), shape or "x".join(map(str, self.key)))
        return "refused:" + self.reason


_LOOKUP_MEMO: dict = {}                                                  # the decision per full argument set + table identity (lookup_cell)
_LOOKUP_MEMO_MAX = 4096
LOOKUP_MEMO_STATS = {"hit": 0, "miss": 0, "cleared": 0}


def lookup_memo_clear() -> None:
    """Forget every memoised cell decision — call after EDITING a loaded table in place (a row of ``cells()["rows"]`` / ``["named_off"]`` or an
    entry of ``kernels.safe_settings.SAFE_ROWS`` changed without changing the container); a reloaded table (a new object) and an appended row or
    entry (a new length) miss the memo by themselves."""
    _LOOKUP_MEMO.clear(); LOOKUP_MEMO_STATS["cleared"] += 1


def lookup_cell(impl: str, piece: str, key: Sequence[int], device=None, *, variant: Optional[str] = None,
                stack: Optional[Tuple[str, str]] = None) -> CellDecision:
    """The non-raising form of :func:`find_cell` — the ONE resolution statement: (1) rows of (impl, piece, key, cc[, variant]): exact (cc, triton)
    rows before (cc, '*') rows, first in file order wins; ``certified`` serves, ``candidate`` serves only under OPT_CORE_PF_ALLOW_CANDIDATE=1;
    (2) no servable row and the table lists the key as MEASURED OFF (a ``named_off`` entry, or a ``candidate`` row of the key not admitted) ->
    unserved, ``reason`` = ``no-cell:<impl>:<piece>:<key>:<cc>|<triton>+off(not-measured)`` (``+off(<why>)`` when the entry / row names its own ``off`` word;
    a candidate alone on a key the safe rows exclude by measurement keeps that exclusion's ``+no_safe(<why>)`` word);
    (3) otherwise the key is UNKNOWN: the lever's SAFE settings for the capability (:func:`_safe_row`, served_by ``safe``), or for an impl in
    DEFAULT_IMPLS its default settings (served_by ``default``); (4) else unserved with the ``no-cell:…`` word (no safe settings admit the shape).
    ``stack`` = (cc 'M.m', triton 'M.m') decides for a box other than the running one (a pre-flight on a named part, CPU tests); default: the
    running box (``device``).  Nothing prints and nothing launches here.
    Memoised on the FULL argument set (impl, piece, key, cc, triton, variant, the candidate switch) AND the table's identity (the loaded object,
    its row count, its named_off count): the same question on the same table gets the same answer without re-scanning the rows (the trunk asks
    it thousands of times per predicted item); a reloaded table or a row appended at run time is a different key, an in-place edit of an existing
    row requires :func:`lookup_memo_clear`."""
    cc, tt = stack if stack is not None else _stack(device)
    key_t = tuple(int(k) for k in key)
    table = cells()
    allow_cand = allow_candidate()
    from opt_core.kernels import safe_settings as _ss                    # the safe-settings table is the decision's other input: its identity keys the memo too
    mkey = (impl, piece, key_t, cc, tt, variant, allow_cand, id(table), len(table.get("rows", ())), len(table.get("named_off", ())),
            id(_ss.SAFE_ROWS), len(_ss.SAFE_ROWS))
    got = _LOOKUP_MEMO.get(mkey)
    if got is not None:
        LOOKUP_MEMO_STATS["hit"] += 1
        return got
    LOOKUP_MEMO_STATS["miss"] += 1
    if len(_LOOKUP_MEMO) >= _LOOKUP_MEMO_MAX:
        _LOOKUP_MEMO.clear()
    got = _LOOKUP_MEMO[mkey] = _lookup_cell_scan(impl, piece, key_t, cc, tt, variant, allow_cand)
    return got


INHERIT_TOKEN = "inherited_cc:unmeasured"                               # served_by suffix when an unmeasured capability runs the nearest measured column's row


def measured_columns(impl: Optional[str] = None, piece: Optional[str] = None, key: Optional[Sequence[int]] = None,
                     variant: Optional[str] = None) -> List[str]:
    """Capabilities with at least one CERTIFIED row in the table -- for that (impl, piece, key) when given (any triton line; of that
    ``variant`` when one is pinned, else any variant), else for any shape."""
    kk = [int(x) for x in key] if key is not None else None
    out = set()
    for r in cells()["rows"]:
        if r.get("status") != "certified":
            continue
        if impl is not None and (r["impl"] != impl or r["piece"] != piece or [int(x) for x in r["key"]] != kk):
            continue
        if variant is not None and r.get("variant") != variant:
            continue
        out.add(str(r["cc"]))
    return sorted(out, key=lambda c: tuple(int(x) for x in c.split(".")))


REFERENCE_COLUMNS = ("9.0", "8.0")                                       # capabilities are measured key by key AND variant by variant: they inherit per key only


def inherit_column(cc: str, impl: Optional[str] = None, piece: Optional[str] = None, key: Optional[Sequence[int]] = None,
                   variant: Optional[str] = None) -> Optional[str]:
    """The measured column a capability inherits certified rows from FOR ONE SHAPE KEY (impl, piece, key [, pinned variant]): the highest
    capability at or below it holding a certified row for that key (of that variant when the caller pins one) -- 12.0 -> 10.3's rows where
    they exist for the key, else 10.0 / 9.0 / 8.0; 8.6 / 8.9 -> 8.0 -- or None (this capability has such a row itself / nothing at or below
    it).  Decided per key and per pinned variant: a capability with certified rows for OTHER keys, or for other variants of this key while
    the caller pins one it lacks, still inherits for this one."""
    if variant is not None and str(cc) in REFERENCE_COLUMNS:               # the fully measured columns decide pinned variants by their own table (a variant not certified
        variant = None                                                    # there is a measured absence, not a gap): per-variant inheritance is for the newer parts only
    cols = measured_columns(impl, piece, key, variant)
    if str(cc) in cols:
        return None
    try:
        mine = tuple(int(x) for x in str(cc).split("."))
    except ValueError:
        return None
    below = [c for c in cols if tuple(int(x) for x in c.split(".")) <= mine]
    return below[-1] if below else None


def _lookup_cell_scan(impl: str, piece: str, key_t: Tuple[int, ...], cc: str, tt: str, variant: Optional[str], allow_cand: bool) -> CellDecision:
    """The resolution statement itself (one scan of the table; :func:`lookup_cell` memoises it)."""
    key = list(key_t)
    best = None
    listed_off = None                                                     # a candidate row of this key not admitted: the key is measured off
    for row in cells()["rows"]:
        if row["impl"] != impl or row["piece"] != piece or [int(k) for k in row["key"]] != key or row["cc"] != cc:
            continue
        if row["triton"] not in (tt, "*"):
            continue
        if variant is not None and row.get("variant") != variant:
            continue
        if row["status"] != "certified" and not (row["status"] == "candidate" and allow_cand):
            if row["status"] == "candidate" and listed_off is None:
                listed_off = row
            continue
        if best is None or (row["triton"] == tt and best["triton"] == "*"):
            best = row
            if row["triton"] == tt:
                break
    if best is not None:
        return CellDecision(impl, piece, tuple(key), cc, tt, best, "", f"{cc}|{tt}" if best["triton"] == tt else f"{cc}|*")
    col = inherit_column(cc, impl, piece, key_t, variant)                          # no certified row FOR THIS KEY on this capability (whatever it holds for other keys): the nearest measured
    if col is not None:                                                   # column at or below it lends its certified rows (Triton source, compiled on the running part) --
        from opt_core.kernels.cell_words import NOT_MEASURED as _NM       # never a refusal where that column serves; a MEASURED off entry on this capability outranks it;
        ent0 = named_off_entry(impl, piece, key, cc, variant)             # candidate rows of this capability keep their switch
        if ent0 is None or ent0.get("off", _NM) == _NM:
            d = _lookup_cell_scan(impl, piece, key_t, col, tt, variant, False)
            if d.served:
                return CellDecision(impl, piece, tuple(key), cc, tt, d.row, d.reason, "%s;%s(%s->%s)" % (d.served_by, INHERIT_TOKEN, cc, col))
    no_cell = f"no-cell:{impl}:{piece}:{'x'.join(map(str, key))}:{cc}|{tt}"
    ent = named_off_entry(impl, piece, key, cc, variant)
    safe, no_safe = _safe_row(impl, piece, key, cc, variant)
    if ent is None and listed_off is not None and no_safe:                # an unraced candidate row on a key the safe rows exclude BY MEASUREMENT: the measured
        return CellDecision(impl, piece, tuple(key), cc, tt, None, no_cell + f"+no_safe({no_safe})", "")   # exclusion's word stands (the candidate serves only under its switch)
    if ent is not None or listed_off is not None:
        from opt_core.kernels.cell_words import with_off, NOT_MEASURED
        return CellDecision(impl, piece, tuple(key), cc, tt, None, with_off(no_cell, (ent if ent is not None else listed_off).get("off", NOT_MEASURED)), "")
    if safe is not None:
        return CellDecision(impl, piece, tuple(key), cc, tt, safe, "no_cell:" + "x".join(map(str, key)), "safe")
    if impl in DEFAULT_IMPLS and (variant is None or variant == piece):
        return CellDecision(impl, piece, tuple(key), cc, tt, _default_row(impl, piece, key, cc), "default:" + "x".join(map(str, key)), "default")
    return CellDecision(impl, piece, tuple(key), cc, tt, None, no_cell + (f"+no_safe({no_safe})" if no_safe else ""), "")


def preflight(wanted: Sequence[Tuple[str, str, Sequence[int]]], device=None, *, stack: Optional[Tuple[str, str]] = None) -> List[CellDecision]:
    """One :class:`CellDecision` per wanted (impl, piece, key), in order, for the running box (or ``stack``).  Nothing raises, prints or
    launches: a kit calls this once at activation with the shapes its line runs — every served decision (a row, the safe settings, an impl's
    default settings) is an engaged lever; an unserved one is the kit's named per-shape route (``no-cell:…[+off(…)]``) or its NOT ACTIVE exit,
    decided before any forward."""
    return [lookup_cell(impl, piece, key, device, stack=stack) for impl, piece, key in wanted]


def _table_ccs(impl: str, piece: str) -> List[str]:
    """The capabilities the table carries servable rows of (impl, piece) for — the inform line's `tuned rows exist for cc …` list."""
    return sorted({r["cc"] for r in cells()["rows"] if r["impl"] == impl and r["piece"] == piece and r["status"] == "certified"}, key=lambda c: tuple(int(x) for x in c.split(".")))


def find_cell(impl: str, piece: str, key: Sequence[int], device, *, variant: Optional[str] = None, stack: Optional[Tuple[str, str]] = None) -> dict:
    """The row serving (impl, piece, key) on this box (resolution: :func:`lookup_cell`), or raise Unsupported with the decision's word —
    ``no-cell:…+off(not-measured)`` (the table lists the key as measured off) / ``no-cell:…`` (nothing admits the shape): the caller's per-call
    route to the stock statement (counted on its census, exit 0), never a refusal of the mode.  When the lever's SAFE settings serve an UNKNOWN key the
    safety net engages FOR THAT KEY — ONE line ``[opt_core/<lever>] safe settings served (no_cell:<key>, cc <cc>, triton <mm>)`` the first
    time the key is met (the lever's pinned keys keep their own cells in the same process); when an impl's default settings serve one
    (DEFAULT_IMPLS) ONE info line ``[opt_core/pair_fused:lnl] default settings (no row for
    <piece> <key> on cc <cc>, …); tuned rows exist for cc …`` — and the row is returned."""
    d = lookup_cell(impl, piece, key, device, variant=variant, stack=stack)
    if d.row is None:
        raise Unsupported(d.reason)
    if d.served_by in ("safe", "default"):
        from opt_core.kernels import safe_settings as _ss
        where = _ss.where_word(d.cc, d.triton)
        if d.served_by == "safe":
            safety_net(SAFE_LEVERS[(impl, piece)]).engage(d.reason, where, cell="x".join(map(str, d.key)))
        else:
            safety_net(LNL_NET if impl == "lnl" else f"pair_fused:{impl}").inform("no row for %s %s on cc %s" % (piece, "x".join(map(str, d.key)), d.cc), where, tuned=_table_ccs(impl, piece))
    return d.row


def run_cell(lever: str, cc: str, triton: str, build: Callable[[dict], Any], cfg: dict, *, dims: Optional[Dict[str, int]] = None,
             explicit: bool = False, key: Optional[Sequence[int]] = None):
    """Launch one cell under the lever's safety net (opt_core.kernels.safe_settings.SafeNet.run): ``build(cfg)``; a triton BUILD failure of the
    tuned cell (never an OOM, never a running kernel's error) -> the lever's safe settings for `cc` that admit the call's shape ``dims`` + ONE
    line; the safe settings failing to build -> :class:`Refused` (case 4: the lever cannot run on this box at all; the message names `--mode off`).
    A lever without safe settings on `cc` builds `cfg` as given (any failure propagates); when the safe rows exclude this shape class a tuned cell
    that fails to build is a SHAPE miss — Unsupported('no-cell:<lever>:build_failed:<Exc>+no_safe(<word>)'): the caller's per-call stock route.
    ``key``: the cell's key — the safety net is scoped PER CELL (an unknown key's safe settings never demote a pinned key's ``cfg`` in the
    same process; only a build failure switches the whole process: opt_core.kernels.safe_settings)."""
    from opt_core.kernels import safe_settings as _ss
    srow, no_safe = _ss.safe_settings_for(lever, cc, dims)
    if srow is None and no_safe is None:
        return build(cfg)
    where = _ss.where_word(cc, triton)
    if srow is None:                                                      # safe settings exist on the capability but not for this shape class: a build failure is the refusal
        try:
            return build(cfg)
        except _ss.catchable() as e:
            if explicit or not _ss.is_build_failure(e):
                raise
            raise Unsupported(f"no-cell:{lever}:build_failed:{type(e).__name__}+no_safe({no_safe})", where) from e   # this shape goes to the stock statement per call, by name
    return safety_net(lever).run(build, cfg, (lambda: dict(srow["settings"])), where=where, explicit=explicit,
                                 cell=("x".join(str(int(k)) for k in key) if key is not None else None))


def row_key(row: dict) -> str:
    """The table key a served row came from: ``"<cc>|<triton M.m>"`` (named-exception row), ``"<cc>|*"`` (the capability's row), ``"safe"`` (the lever's
    SAFE settings) or ``"default"`` (an impl's default settings, DEFAULT_IMPLS)."""
    if row.get("status") in ("safe", "default"):
        return str(row["status"])
    return f"{row.get('cc', '?')}|{row.get('triton', '*')}"


def _count_rows(plan: dict) -> None:
    if plan.get("k2b_cells_note"):
        _K2B_NOTE[0] = str(plan["k2b_cells_note"])
    for v in plan.values():
        if isinstance(v, dict) and ("status" in v and "piece" in v):
            SERVED_KEYS.setdefault(f"{v.get('impl', '?')}.{v['piece']}", set()).add(row_key(v))
        if isinstance(v, dict) and v.get("status") in SERVED_BY_STATUS:
            SERVED_BY_STATUS[v["status"]] += 1
            nm = str(v.get("variant", "?"))
            SERVED_VARIANTS[nm] = SERVED_VARIANTS.get(nm, 0) + 1


def allow_candidate() -> bool:
    """True when this process serves candidate (untested) rows — OPT_CORE_PF_ALLOW_CANDIDATE=1, a qualification-job switch; printed on the LEVER line."""
    return os.environ.get(ALLOW_CANDIDATE_ENV, "") == "1"


def pick_cell(impl: str, piece: str, key: Sequence[int], device, variants: Sequence[str]) -> dict:
    """The first servable row over an ORDERED variant preference (e.g. ('prologue_v4', 'v3')): a faster kernel variant whose own precondition
    holds for this call is preferred, the reference variant otherwise; 'no-cell' only when no variant has a row."""
    # NOTE: never keep a caught exception object alive here — an exception bound to a local forms a reference cycle with its own
    # traceback and this frame, whose f_back chain keeps the CALLER frames (tri_attn_block / transition and every kernel intermediate in their
    # locals: q, k, v, g, o, out ...) allocated until the generational GC runs.  Keep the refusal WORDS only and raise a fresh exception.
    last_words: Optional[Tuple[str, str]] = None
    for v in variants:
        try:
            return find_cell(impl, piece, key, device, variant=v)
        except Unsupported as e:
            last_words = (e.reason, e.detail)
    if last_words is not None:
        raise Unsupported(*last_words) from None
    raise Unsupported(f"no-cell:{impl}:{piece}:{'x'.join(map(str, key))}")


# ------------------------------------------------------------------------------------------------------------------------------- carried kernels
_MODS: Dict[str, Any] = {}
_CHECKED: set = set()
_CARRIED_DATA = {"lnl_fused": ("tiles", "lnl_fused.tiles_by_arch.json"), "fpf_triatt_k2b": ("cells", os.path.join("fpf_triatt_k2b", "K2B_CELLS.json"))}
_K2B_TABLE_ID = [""]                                                      # '<basename>@<sha8>' of the K2B table exported (printed on the LEVER line when k2b served)
_K2B_NOTE = [""]                                                          # the carried K2B loader's cells note ('default:no_row': no entry for this capability, its default cell serves) when k2b served
SERVED_BY_STATUS: Dict[str, int] = {"certified": 0, "candidate": 0}      # rows served in this process, by qualification status (the LEVER line prints both)
SERVED_VARIANTS: Dict[str, int] = {}                                    # rows served by kernel variant name (the LEVER line prints variants=<name>:<n>,…)
SERVED_CORES: Dict[str, int] = {}                                       # attention cores served in this process: "flash_triattn" / "k2b" / "callable" / "tier:<word>=<row>" /
                                                                          # "tier:<word>=flash_triattn(refused:<kind>)" -> calls (the LEVER line prints cores=<name>:<n>,...)
_TIER_CORE_MEMO: Dict[tuple, tuple] = {}                                  # (word, cc, dtype, D, H, keys, form, direction) -> ("row", Selection) | ("refused", kind, row, fallback)
SERVED_KEYS: Dict[str, set] = {}                                          # "<impl>.<piece>" -> the table keys that served it in this process ("<cc>|<mm>", "<cc>|*", "safe"): the LEVER line's keys= tail


def carried_exports(names: Optional[Sequence[str]] = None) -> Dict[str, str]:
    """{ENV_VAR: value} the serve layer exports before importing the carried cells that read a data file at import (from each cell's META
    export spec: the lnl tile table, the K2B launch cells, ...) — a kit's activation gate exports these and runs kernels.route_check(name)
    BEFORE the first use.  ``names`` restricts to some carried names (default: every cell with a data export)."""
    from opt_core import kernels as K
    out: Dict[str, str] = {}
    for name, (param, rel) in _CARRIED_DATA.items():
        if names is not None and name not in names:
            continue
        out.update(K.exports(name, **{param: os.path.join(K.KERNELS_DIR, rel)}))
    return out


def _carried(name: str, sub: Optional[str] = None):
    """Import a carried kernel by its routed top-level name (the FPF packages import each other by top-level name, so they are served through
    the core's route, never as opt_core.kernels.<name> submodules) and hold the import to the core copy.  Routes exactly ``name`` and the
    names its META entry declares under ``runtime_imports`` (a route never changes what executes beyond the routed names); exports the data
    files the carried code reads at import (lnl tile table, K2B cells) and validates them with ``kernels.route_check`` before the first import."""
    from opt_core import kernels as K
    modname = name + ("." + sub if sub else "")
    m = _MODS.get(modname)
    if m is not None:
        return m
    if name == "flash_triattn":                                            # ONE module object of the attention core per process: the serve layer's resolution
        from opt_core.kernels import flash_triattn_serve as FS
        err = None
        try:
            m = FS.kernel_module()
        except Exception as e:
            err = repr(e)[:160]
        if err is not None:
            raise Unsupported("import:flash_triattn", err) from None
        _MODS[modname] = m
        return m
    if name in _CARRIED_DATA:                                              # the data file the carried code reads at import: exported through its META export spec
        path = os.path.join(K.KERNELS_DIR, _CARRIED_DATA[name][1])
        for var, val in carried_exports((name,)).items():
            if os.environ.get(var, val) != val:                           # a kit exported another table first: what runs is not the core's cell table
                raise Unsupported(f"export:{var}", os.environ.get(var, ""))
            os.environ[var] = val
        if name == "fpf_triatt_k2b":
            import hashlib
            _K2B_TABLE_ID[0] = os.path.basename(path) + "@" + hashlib.sha256(open(path, "rb").read()).hexdigest()[:8]
    names = [name] + [n for n in K.sums(name).get("runtime_imports", []) if n in K.names()]
    for n in names:
        if n not in K.routed():
            K.route(n)
    if name not in _CHECKED:
        g = K.route_check(name)
        if not g.ok:
            raise Unsupported("route:" + name, str(getattr(g, "reason", ""))[:200])
        _CHECKED.add(name)
    err = None
    try:
        m = importlib.import_module(modname)
    except Exception as e:  # ImportError, triton absent, ...
        err = repr(e)[:160]
    if err is not None:
        raise Unsupported("import:" + name, err) from None
    root = K.carried_path(name)
    here = os.path.abspath(getattr(m, "__file__", "") or "")
    if not (here == os.path.abspath(root) or here.startswith(os.path.abspath(root) + os.sep)):
        raise Unsupported("import:shadowed:" + name, here)          # a kit copy of the same name was imported first: the kit routes or deletes it
    _MODS[modname] = m
    return m


# ------------------------------------------------------------------------------------------------------------------------------------- weights
class TriAttnWeights:
    """Packed triangle-attention weights (bf16) + geometry. Built by :func:`pack_triattn_weights`; holds the fpf cache schema the carried
    prologue/epilogue read (``_fpf_cache``) so the carried kernels run unmodified on it."""
    __slots__ = ("C", "H", "D", "HB", "C_out", "eps", "dev", "gate", "affine", "wqkvg", "bqkvg", "b_g", "wb", "wb_pad", "lnw16", "lnb16", "lnw32", "lnb32",
                 "wo16", "woT16", "b_o", "_fpf_cache", "shapes")


class TransitionWeights:
    __slots__ = ("C", "NH", "n", "eps", "dev", "affine", "wa16", "wb16", "wab16", "wabI16", "wo16", "lnw16", "lnb16", "lnw32", "lnb32", "cache", "v2_packed")


def _split_rows(w, order: Sequence[str], want: Sequence[str], rows: int):
    """Split a fused [len(order)*rows, C] weight into the blocks named in ``want`` order."""
    blocks = {nm: w[i * rows:(i + 1) * rows] for i, nm in enumerate(order)}
    return [blocks[nm] for nm in want]


def pack_triattn_weights(*, w_o, n_heads: int, head_dim: int, ln_w=None, ln_b=None, w_q=None, w_k=None, w_v=None, w_g=None, w_b=None,
                         w_qkvg=None, qkvg_split: Sequence[str] = ("q", "k", "v", "g"), b_g=None, b_o=None, eps: float = 1e-5, device=None) -> TriAttnWeights:
    """Pack once per module (cache the result on the module).  Shapes: w_q/w_k/w_v/w_g [H*D, C] or w_qkvg [4*H*D, C] (row blocks in ``qkvg_split``
    order); b_g [H*D] or None (gate bias: folded into the fused GEMM on lnl|torch, one bf16 add on fpf); w_b [H, C] (pair-bias projection; required);
    w_o [C_out, H*D]; b_o [C_out] or None; ln_w/ln_b [C] or None (no affine).
    ``w_g=None`` with separate q/k/v = a gate-less attention: refused today ('gate:none') rather than silently served."""
    import torch
    H, D = int(n_heads), int(head_dim)
    HD = H * D
    if w_qkvg is not None:
        if tuple(w_qkvg.shape[:1]) != (4 * HD,):
            raise Unsupported("w_qkvg-shape", str(tuple(w_qkvg.shape)))
        w_q, w_k, w_v, w_g = _split_rows(w_qkvg, tuple(qkvg_split), ("q", "k", "v", "g"), HD)
    if w_g is None:
        raise Unsupported("gate:none")
    if w_b is None:
        raise Unsupported("bias:none")
    for nm, t in (("w_q", w_q), ("w_k", w_k), ("w_v", w_v), ("w_g", w_g)):
        if t is None or t.dim() != 2 or t.shape[0] != HD:
            raise Unsupported(nm + "-shape", str(None if t is None else tuple(t.shape)))
    C = int(w_q.shape[1])
    if tuple(w_b.shape) != (H, C):
        raise Unsupported("w_b-shape", str(tuple(w_b.shape)))
    if w_o.dim() != 2 or int(w_o.shape[1]) != HD:
        raise Unsupported("w_o-shape", str(tuple(w_o.shape)))
    dev = device if device is not None else w_o.device
    bf = torch.bfloat16
    W = TriAttnWeights()
    with torch.no_grad():
        W.C, W.H, W.D, W.C_out, W.eps, W.dev, W.gate = C, H, D, int(w_o.shape[0]), float(eps), dev, True
        W.affine = ln_w is not None
        try:
            import triton
            HB = max(16, triton.next_power_of_2(H))
        except Exception:
            HB = max(16, 1 << (H - 1).bit_length())
        W.HB = HB
        W.wqkvg = torch.cat([w_q, w_k, w_v, w_g], 0).detach().to(device=dev, dtype=bf).contiguous()          # [4HD, C]
        wb = torch.zeros((HB, C), dtype=bf, device=dev)
        wb[:H] = w_b.detach().to(device=dev, dtype=bf)
        W.wb_pad = wb                                                                                      # [HB, C] zero-padded (fpf + lnl both want a pow2 >= 16 projection width)
        W.wb = w_b.detach().to(device=dev, dtype=bf).contiguous()
        ones = torch.ones(C, device=dev); zeros = torch.zeros(C, device=dev)
        lw = ln_w.detach().to(dev) if ln_w is not None else ones
        lb = ln_b.detach().to(dev) if ln_b is not None else zeros
        W.lnw16, W.lnb16 = lw.to(bf).contiguous(), lb.to(bf).contiguous()
        W.lnw32, W.lnb32 = lw.float().contiguous(), lb.float().contiguous()
        W.wo16 = w_o.detach().to(device=dev, dtype=bf).contiguous()                                        # [C_out, HD]
        if b_g is not None:                                                                                # gate bias (the engine's gate projection bias): folded into the fused GEMM
            if tuple(b_g.shape) != (HD,):
                raise Unsupported("b_g-shape", str(tuple(b_g.shape)))
            W.b_g = b_g.detach().to(device=dev, dtype=bf).contiguous()
            W.bqkvg = torch.cat([torch.zeros(3 * HD, dtype=bf, device=dev), W.b_g], 0).contiguous()        # [4HD]: zeros for q|k|v, b_g for g
        else:
            W.b_g = None; W.bqkvg = None
        W.woT16 = W.wo16.t().contiguous()                                                                  # [HD, C_out]
        W.b_o = b_o.detach().to(device=dev, dtype=bf).contiguous() if b_o is not None else None
    # the carried fpf_triatt_pro cache schema (prologue.get_cache) + fpf_triatt_epi's flat keys: the kernels read these, unmodified
    W._fpf_cache = {"triatt_pro": dict(kind="triatt_pro", dev=dev, C=C, H=H, D=D, HB=HB, wqkvg=W.wqkvg, wb=W.wb_pad, lnw=W.lnw16, lnb=W.lnb16,
                                       wo=W.wo16, eps=W.eps, wo16=W.wo16),
                    "triatt_epi": dict(dev=dev, wo16=W.wo16, woT16=W.woT16)}
    W.shapes = f"c{C}h{H}d{D}o{W.C_out}"
    return W


def pack_transition_weights(*, w_out, ln_w=None, ln_b=None, w_a=None, w_b=None, w_ab=None, ab_split: Sequence[str] = ("a", "b"),
                            eps: float = 1e-5, device=None) -> TransitionWeights:
    """SwiGLU transition weights: a = LN(x) @ w_a^T, b = LN(x) @ w_b^T ([n*C, C] each, or fused w_ab [2*n*C, C] in ``ab_split`` row-block order);
    out = (silu(a) * b) @ w_out^T, w_out [C, n*C]."""
    import torch
    if w_out.dim() != 2:
        raise Unsupported("w_out-shape", str(tuple(w_out.shape)))
    C, NH = int(w_out.shape[0]), int(w_out.shape[1])
    if w_ab is not None:
        if tuple(w_ab.shape) != (2 * NH, C):
            raise Unsupported("w_ab-shape", str(tuple(w_ab.shape)))
        w_a, w_b = _split_rows(w_ab, tuple(ab_split), ("a", "b"), NH)
    for nm, t in (("w_a", w_a), ("w_b", w_b)):
        if t is None or tuple(t.shape) != (NH, C):
            raise Unsupported(nm + "-shape", str(None if t is None else tuple(t.shape)))
    if NH % C != 0:
        raise Unsupported("hidden", f"n*C hidden expected, got C={C} hidden={NH}")
    dev = device if device is not None else w_out.device
    bf = torch.bfloat16
    T = TransitionWeights()
    with torch.no_grad():
        T.C, T.NH, T.n, T.eps, T.dev, T.affine = C, NH, NH // C, float(eps), dev, ln_w is not None
        T.wa16 = w_a.detach().to(device=dev, dtype=bf).contiguous()
        T.wb16 = w_b.detach().to(device=dev, dtype=bf).contiguous()
        T.wab16 = torch.cat([T.wa16, T.wb16], 0).contiguous()                                              # [2NH, C]   (IL=0 layout)
        T.wabI16 = torch.stack([T.wa16, T.wb16], 1).reshape(2 * NH, C).contiguous()                       # interleaved (IL=1 layout)
        T.wo16 = w_out.detach().to(device=dev, dtype=bf).contiguous()                                      # [C, NH]
        ones = torch.ones(C, device=dev); zeros = torch.zeros(C, device=dev)
        lw = ln_w.detach().to(dev) if ln_w is not None else ones
        lb = ln_b.detach().to(dev) if ln_b is not None else zeros
        T.lnw16, T.lnb16 = lw.to(bf).contiguous(), lb.to(bf).contiguous()
        T.lnw32, T.lnb32 = lw.float().contiguous(), lb.float().contiguous()
    # the carried fpf_transition cache schema (transition._weights)
    T.cache = {"transition_key": ("opt_core.pair_fused",), "wab16": T.wab16, "wabI16": T.wabI16, "wo16": T.wo16, "lnw16": T.lnw16, "lnb16": T.lnb16, "eps": T.eps}
    T.v2_packed = None                                                                                      # the v2 kernel's operand pack, built at the first v2 launch on the call's device
    return T


# --------------------------------------------------------------------------------------------------------------------------------- eligibility
def _check_z(z, W: TriAttnWeights, allow_fp32: bool = False, fused: bool = True):
    """allow_fp32: an fp32 residual stream (trunks that keep an fp32 pair stream under bf16 autocast) is accepted when the caller takes the bf16
    update back (residual=False): with ln='fused' the fpf prologue reads fp32 z directly (LN in-kernel), with ln='stock' z only names the frame."""
    import torch
    if not (torch.is_tensor(z) and z.is_cuda):
        raise Unsupported("device")
    if z.dtype == torch.float32 and not allow_fp32:
        raise Unsupported("dtype:float32", "fp32 z (an fp32 residual stream) is served with residual=False only (the bf16 update is returned), impl fpf|torch")
    if z.dtype not in (torch.bfloat16, torch.float32):
        raise Unsupported("dtype:" + str(z.dtype).replace("torch.", ""))
    if z.dim() not in (3, 4) or z.shape[-2] != z.shape[-3]:
        raise Unsupported("rank", str(tuple(z.shape)))
    if z.shape[-1] != W.C:
        raise Unsupported(f"c:{z.shape[-1]}!={W.C}")
    if z.stride(-1) != 1:
        raise Unsupported("stride", "z last dim must be contiguous")
    C, HD, D = W.C, W.H * W.D, W.D
    if fused and ((C & (C - 1)) or (HD & (HD - 1)) or (D & (D - 1)) or C < 16 or D < 8):
        raise Unsupported(f"pow2:c{C}h{W.H}d{D}")               # the fused kernels' constexpr aranges (C=384 / H*D=384: impl='torch' serves those dims)
    if fused and (W.C_out & (W.C_out - 1)):
        raise Unsupported(f"pow2:c_out{W.C_out}")


def _key_mask(mask, z, ending: bool):
    """-> None or the cuEq/flash_triattn key-mask layout [B, I, 1, 1, J] bool in the ATTENTION frame (a view where possible).
    Accepts: pair mask [N,N] / [B,N,N] (1 = keep; row i's keys j are mask_x[i, j] with x = z or z^T) or the key layout [B,N,1,1,N] itself
    (taken as already in the attention frame)."""
    import torch
    if mask is None:
        return None
    N = z.shape[-2]
    if mask.dim() == 5:
        if tuple(mask.shape[-4:]) != (N, 1, 1, N):
            raise Unsupported("mask-shape", str(tuple(mask.shape)))
        m = mask
    elif mask.dim() in (2, 3):
        if tuple(mask.shape[-2:]) != (N, N):
            raise Unsupported("mask-shape", str(tuple(mask.shape)))
        m = mask if mask.dim() == 3 else mask.unsqueeze(0)
        if ending:
            m = m.transpose(-1, -2)
        m = m[:, :, None, None, :]
    else:
        raise Unsupported("mask-shape", str(tuple(mask.shape)))
    if m.dtype != torch.bool:
        m = m != 0
    return m


def supported_triattn(z, W: TriAttnWeights, mask=None, *, impl: str = "fpf", core: Any = DEFAULT_CORE, ending: bool = False,
                      ln: str = "fused", residual: bool = False, variant=None, engage_cells: bool = False) -> Tuple[bool, str]:
    """(ok, reason) — never raises; the same words tri_attn_block would raise with (incl. ``cell:<id>:stock_block(...)`` from the engage cells:
    default rows always, opt-in rows when ``engage_cells``)."""
    try:
        _plan_triattn(z, W, mask, impl=impl, core=core, ending=ending, ln=ln, residual=residual, variant=variant, engage_cells=engage_cells)
        return True, ""
    except Unsupported as e:
        return False, e.reason


def transition_plan_words(x, T: TransitionWeights, *, impl: str = "fpf", ln: str = "fused", residual: bool = False, variant=None, mask=None,
                          variants=None) -> Tuple[bool, str]:
    """(True, '<variant>') when a row serves the call (the kernel variant that WOULD launch: FOLD_VARIANT folds the row mask and the residual in
    the kernel and may write in place — served only when `variants` names it or `variant` pins it; 'v1' / 'v1_fp32x' / 'transition_v3' return
    the update), else (False, '<reason>').  Nothing launches."""
    try:
        plan = _plan_transition(x, T, impl=impl, ln=ln, residual=residual, variant=variant, mask=mask, variants=variants)
        return True, str(plan["transition"].get("variant") or plan["impl"])
    except Unsupported as e:
        return False, e.reason


def supported_transition(x, T: TransitionWeights, *, impl: str = "fpf", ln: str = "fused", residual: bool = False, variant=None) -> Tuple[bool, str]:
    """(ok, reason).  Reason vocabulary note for the kit: a stock transition that CHUNKS the hidden dimension and accumulates the partial outputs
    in bf16 (as some stock code does at large N) has a different rounding chain than this cell (one fp32 chain over the hidden) — tier-2 there, never exact; a kit
    that needs exactness on those calls routes them to stock by name ('n/a:chunked-order')."""
    try:
        _plan_transition(x, T, impl=impl, ln=ln, residual=residual, variant=variant)
        return True, ""
    except Unsupported as e:
        return False, e.reason


def _variant_pins(variant, pieces):
    """variant=None | '<name>' (pins the first piece) | {piece: '<name>'} -> one pin (or None) per piece."""
    if variant is None:
        return tuple(None for _ in pieces)
    if isinstance(variant, str):
        return tuple(variant if i == 0 else None for i in range(len(pieces)))
    if isinstance(variant, dict):
        bad = [k for k in variant if k not in pieces]
        if bad:
            raise Unsupported("variant:" + ",".join(map(str, bad)))
        return tuple(variant.get(p) for p in pieces)
    raise Unsupported("variant:" + str(variant))


def _pick(impl, piece, key, device, applicable: Sequence[str], pin: Optional[str]) -> dict:
    """The table's preferred APPLICABLE variant (pin=None), or exactly the pinned one: a pin outside the variants applicable to this call, or
    without a servable row, refuses ('no-cell:variant:<name>') — never a substitute."""
    if pin is None:
        return pick_cell(impl, piece, key, device, applicable)
    if pin not in applicable:
        raise Unsupported(f"no-cell:variant:{pin}", f"applicable here: {','.join(applicable)}")
    words = None
    try:
        return find_cell(impl, piece, key, device, variant=pin)
    except Unsupported as e:
        words = e.reason
    raise Unsupported(f"no-cell:variant:{pin}", words) from None


SERVED_PIECES: Dict[str, int] = {}                                    # pieces served through another impl by cell: "epilogue=fpf>lnl(off:slower)@<key>|<cc>" -> count
_PIECE_NOTES: Dict[str, bool] = {}


def _epilogue_statements_piece(err: "Unsupported", W, device, *, pinned: bool) -> dict:
    """The fused epilogue's cell is MEASURED OFF for (C_out, H, D) on this capability (its word carries ``+off(<why>)``): the lnl statements
    piece -- gate_transpose row + one cuBLAS GEMM + add -- serves BY NAME (counted on the LEVER line, one stderr note per key).  Any other
    refusal (unknown key without safe settings, a pinned variant, no lnl cell) re-raises the fused piece's own word."""
    from opt_core.kernels.cell_words import is_off_word, off_why
    if pinned or not is_off_word(err.reason):
        raise err
    try:
        row = find_cell("lnl", "gate_transpose", (W.H * W.D,), device)
    except Unsupported:
        raise err from None
    _carried("lnl_fused")
    tag = "epilogue=fpf>lnl(off:%s)@%dx%dx%d" % (off_why(err.reason) or "not-measured", W.C_out, W.H, W.D)
    SERVED_PIECES[tag] = SERVED_PIECES.get(tag, 0) + 1
    if tag not in _PIECE_NOTES:
        _PIECE_NOTES[tag] = True
        sys.stderr.write("[opt_core/pair_fused] %s: the fused epilogue is measured off here (%s); the statements piece serves by name\n" % (tag, err.reason))
    return dict(row, impl="lnl")


def composition_rows():
    """The composition cells (pair_fused_cells.json "compositions"): pieces the fused block serves through the plain statements BY CELL."""
    return list((_cells_doc().get("compositions") or {}).get("rows") or [])


def composition_for(piece: str, cc, dtype, key, ln: str):
    """The applies=default composition row for (piece, cc, dtype, key, ln form), or None."""
    ccw = cc if isinstance(cc, str) else "%d.%d" % (int(cc[0]), int(cc[1]))
    dt = {"torch.bfloat16": "bf16", "torch.float32": "fp32", "torch.float16": "fp16"}.get(str(dtype), str(dtype).replace("torch.", ""))
    for r in composition_rows():
        if r.get("piece") == piece and r.get("cc") == ccw and r.get("dtype") in (dt, "*") and [int(x) for x in r.get("key", [])] == [int(x) for x in key] \
                and r.get("ln", "*") in (ln, "*") and r.get("applies", "default") == "default":
            return r
    return None


def _prologue_statements_piece(err: "Unsupported", W, z, *, ln: str, pinned: bool) -> dict:
    """The fused prologue has no servable row for (C, H, D) on this capability in this LN form AND a composition cell names the statements
    prologue for it (its fused rows were not selected there while the fused epilogue was): the plain-statements prologue serves BY CELL (counted on the
    LEVER line, one stderr note per key).  A pinned variant or a key without a composition cell re-raises the fused piece's own word."""
    if pinned:
        raise err
    import torch as _t
    if not (_t.is_tensor(z) and z.is_cuda):
        raise err
    cc = _t.cuda.get_device_capability(z.device)
    dt = _t.get_autocast_dtype("cuda") if _t.is_autocast_enabled() else z.dtype
    row = composition_for("prologue", cc, dt, (W.C, W.H, W.D), ln)
    if row is None or row.get("impl") != "torch":
        raise err
    tag = "prologue=fpf>torch(compose:%s)@%dx%dx%d" % (row["id"], W.C, W.H, W.D)
    SERVED_PIECES[tag] = SERVED_PIECES.get(tag, 0) + 1
    if tag not in _PIECE_NOTES:
        _PIECE_NOTES[tag] = True
        sys.stderr.write("[opt_core/pair_fused] %s: the fused prologue rows lost their race in this form here (%s); the statements prologue serves by cell\n" % (tag, err.reason))
    return {"id": "compose:%s" % row["id"], "impl": "torch", "variant": None, "cfg": {}, "triton": "*", "cc": row["cc"], "status": "composition"}


def engage_rows():
    """The engage cells (pair_fused_cells.json "engage"): where the fused tri-attention surround steps aside to the engine's own block by cell."""
    return list((_cells_doc().get("engage") or {}).get("rows") or [])


def engage(cc, dtype, head_dim, heads, keys, *, variant=None, engage_cells=False):
    """(True, '') when no engage cell sends this call class to the engine's own block; else (False, 'cell:<id>:<reason>').  ``applies=default``
    rows bind every caller; ``applies=opt_in`` rows only when ``engage_cells`` is True.  cc '9.0' | (9, 0); dtype word; keys = pair size N."""
    ccw = cc if isinstance(cc, str) else "%d.%d" % (int(cc[0]), int(cc[1]))
    dt = {"torch.bfloat16": "bf16", "torch.float32": "fp32", "torch.float16": "fp16"}.get(str(dtype), str(dtype).replace("torch.", ""))
    for r in engage_rows():
        if str(r.get("cc")) != ccw or str(r.get("dtype")) != dt:
            continue
        if r.get("head_dim") not in ("*", None) and int(r["head_dim"]) != int(head_dim):
            continue
        if r.get("heads") not in ("*", None) and int(r["heads"]) != int(heads):
            continue
        rv = r.get("variant", "*")
        if rv != "*" and str(rv) != str(variant or ""):
            continue
        if r.get("applies") == "opt_in" and not engage_cells:
            continue
        lo, hi = r.get("keys_min_exclusive"), r.get("keys_max_exclusive")
        k = int(keys)
        if (lo is None or k > int(lo)) and (hi is None or k < int(hi)):
            return False, "cell:%s:%s" % (r.get("id"), r.get("reason", "stock_block"))
    return True, ""


def _cells_doc():
    import json as _json
    if "_doc" not in _ENGAGE_MEMO:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pair_fused_cells.json")) as fh:
            _ENGAGE_MEMO["_doc"] = _json.load(fh)
    return _ENGAGE_MEMO["_doc"]


_ENGAGE_MEMO: Dict[str, Any] = {}


def _plan_triattn(z, W, mask, *, impl, core, ending, ln, residual=False, variant=None, engage_cells=False):
    if impl not in IMPLS:
        raise Unsupported("impl:" + str(impl))
    if not _core_ok(core):
        raise Unsupported("core:" + str(core))
    if ln not in ("fused", "stock"):
        raise Unsupported("ln:" + str(ln))
    _check_z(z, W, allow_fp32=(not residual and impl in ("fpf", "torch")), fused=(impl != "torch"))     # fp32 stream: LN in-kernel from fp32 z (fpf) or x_ln given
    _key_mask(mask, z, ending)
    try:                                                                  # engage cells: classes where the surround steps aside to the engine's block BY CELL
        import torch as _t
        if _t.is_tensor(z) and z.is_cuda and z.dim() >= 3:
            _cc = _t.cuda.get_device_capability(z.device)
            _dt = _t.get_autocast_dtype("cuda") if _t.is_autocast_enabled() else z.dtype
            ok_, why_ = engage(_cc, _dt, W.D, W.H, int(z.shape[-2]), variant=variant, engage_cells=engage_cells)
            if not ok_:
                raise Unsupported(why_)
    except Unsupported:
        raise
    except Exception as e:                                                # noqa: BLE001  (facts unreadable: no engage cell applies)
        from opt_core.oom import is_oom
        if is_oom(e): raise
    if impl == "fpf" and ln == "stock":                                       # the exact construction: refused by name at the row counts the card's cuBLAS order differs (exact_rows)
        _exact_rows_check("prologue", z, W.C)
    plan = {"impl": impl}
    pv, ev = _variant_pins(variant, ("prologue", "epilogue"))
    if impl == "torch":
        pass                                                              # plain torch statements: nothing to pin
    elif impl == "fpf":
        import torch
        fp32z = z.dtype == torch.float32
        if ln == "stock":                                                 # the kernel reads x_ln: z's dtype does not reach the arithmetic
            pro_ok = ("prologue_v4", "v3")                                # prologue_v4 == v3 bit-exact on its cells (GLUE test record); x_ln is in the attention frame for either node
        elif fp32z:
            pro_ok = ("v3_fp32z",)                                                             # LN in-kernel from an fp32 pair tensor: its own rows
        else:
            pro_ok = ("mkpf_f1", "v3")                                                         # F1: LN in registers, z / z^T by strides (bf16 z)
        epi_ok = ("epilogue_v3", "v2")
        try:
            plan["prologue"] = _pick("fpf", "prologue", (W.C, W.H, W.D), z.device, pro_ok, pv)
        except Unsupported as e_:                                         # no servable fused prologue row in this form: the statements prologue BY CELL where a composition cell names it
            plan["prologue"] = _prologue_statements_piece(e_, W, z, ln=ln, pinned=pv is not None)
            plan["prologue_impl"] = "torch"
        try:
            plan["epilogue"] = _pick("fpf", "epilogue", (W.C_out, W.H, W.D), z.device, epi_ok, ev)
        except Unsupported as e_:                                         # the fused epilogue is MEASURED OFF on this key / cc: the lnl statements piece by name
            plan["epilogue"] = _epilogue_statements_piece(e_, W, z.device, pinned=ev is not None)
            plan["epilogue_impl"] = "lnl"
        if plan.get("prologue_impl") != "torch":
            _carried("fpf_triatt_pro", "prologue")
        if plan.get("epilogue_impl") != "lnl":
            _carried("fpf_triatt_epi", "epilogue")
        if plan["prologue"]["variant"] == "prologue_v4" or plan["epilogue"]["variant"] == "epilogue_v3":
            _carried("fpf_glue_v2", "kernels")
        if plan["prologue"]["variant"] == "mkpf_f1":
            _carried("fpf_mkpf", "kernels")
    else:
        plan["prologue"] = find_cell("lnl", "ln_linear", (W.C,), z.device)              # lnl tiles come from the carried tiles_by_arch.json
        plan["epilogue"] = find_cell("lnl", "gate_transpose", (W.H * W.D,), z.device)
        _carried("lnl_fused")
    if not callable(core):
        if W.D not in (16, 32, 64, 128):
            raise Unsupported(f"head_dim:{W.D}")
        if core in ("flash_triattn", DEFAULT_CORE) or tier_core_word(core) is not None:   # default / tier cores step aside to flash_triattn by name: the cell must be importable either way
            _carried("flash_triattn")                                     # flash_triattn_serve.kernel_module(): the ONE module object; refuses before any launch
        else:                                                             # 'k2b'
            k2b = _carried("fpf_triatt_k2b")
            plan["k2b_table"] = _K2B_TABLE_ID[0]                          # the exported cells table; a capability it has no entry for is served the package
            note = getattr(k2b.K2B, "cells_note", None)                   # default cell, engaged and named ONCE by the carried loader (cells_note=default:no_row,
            plan["k2b_cells_note"] = (note() if callable(note) else None) or ""   # its `[opt_core/fpf_triatt_k2b] default settings (no row for cc …)` line)
    return plan


FOLD_VARIANT = "v2_fold"                                                   # the transition variant that folds the row mask and the residual in the kernel (opt-in: `variants=`)


def _plan_transition(x, T, *, impl, ln, residual=False, variant=None, mask=None, variants=None):
    import torch
    if impl not in ("fpf", "lnl"):
        raise Unsupported("impl:" + str(impl))
    if ln not in ("fused", "stock"):
        raise Unsupported("ln:" + str(ln))
    if not (torch.is_tensor(x) and x.is_cuda):
        raise Unsupported("device")
    if x.dtype == torch.float32 and not (not residual and impl == "fpf"):
        raise Unsupported("dtype:float32", "fp32 x (an fp32 residual stream) is served with residual=False only (the bf16 update is returned), impl fpf")
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise Unsupported("dtype:" + str(x.dtype).replace("torch.", ""))
    if x.dim() < 2 or x.shape[-1] != T.C:
        raise Unsupported(f"c:{x.shape[-1]}!={T.C}")
    if x.stride(-1) != 1:
        raise Unsupported("stride")
    if impl == "fpf" and ln == "stock":                                       # the exact construction: refused by name at the row counts the card's cuBLAS order differs (exact_rows)
        _exact_rows_check("transition", x, T.C)
    if impl == "fpf":
        (pin,) = _variant_pins(variant, ("transition",))
        if ln == "stock":
            ok = ("transition_v3", "v1")                                  # the kernel reads x_ln: x's dtype does not reach the arithmetic
        elif x.dtype == torch.float32:
            ok = ("v1_fp32x",)                                            # LN in-kernel from an fp32 tensor: its own rows
        else:
            ok = ("v1",)                                                  # bf16 rows, LN in-kernel
            if variants is not None:                                      # the caller's ORDERED preference among this call's variants (FOLD_VARIANT is served only when named here or pinned)
                ok = tuple(v for v in variants if v in (FOLD_VARIANT, "v1")) or ok
            elif pin == FOLD_VARIANT:
                ok = (FOLD_VARIANT,)
        if mask is not None and mask.numel() != x.numel() // T.C:
            raise Unsupported("mask-rows", f"mask {tuple(mask.shape)} for {x.numel() // T.C} rows (one value per row of x)")
        row = _pick("fpf", "transition", (T.C, T.NH), x.device, ok, pin)
        if mask is not None and row["variant"] != FOLD_VARIANT:
            raise Unsupported("mask:variant:" + str(row["variant"]), f"the row mask is folded by the {FOLD_VARIANT} kernel only; other variants take mask=None (the caller multiplies)")
        if row["variant"] == FOLD_VARIANT:
            _carried("fpf_transition_v2", "kernel")
        else:
            _carried("fpf_transition", "transition")
        if row["variant"] == "transition_v3":
            _carried("fpf_glue_v2", "kernels")
        return {"impl": "fpf", "transition": row}
    if mask is not None:
        raise Unsupported("mask:impl:lnl")
    row = find_cell("lnl", "fused_transition", (T.C, T.NH), x.device)
    _carried("lnl_fused")
    return {"impl": "lnl", "transition": row}


# ----------------------------------------------------------------------------------------------------------------------------------- the pieces
class _Owner:
    """Stands in for the engine module the carried fpf kernels take: they read only ``_fpf_cache`` (pre-populated by pack_triattn_weights)."""
    __slots__ = ("_fpf_cache", "__dict__")


def _owner(W: TriAttnWeights, device):
    o = _Owner()
    o._fpf_cache = W._fpf_cache
    if W._fpf_cache["triatt_pro"]["dev"] != device:
        raise Unsupported("device-mismatch", f"weights on {W._fpf_cache['triatt_pro']['dev']}, z on {device}")
    return o


def prologue(z, W: TriAttnWeights, *, ending: bool = False, ln: str = "fused", x_ln=None, impl: str = "fpf", bias_frame: str = "x",
             write_x: bool = False, _plan=None):
    """-> (q, k, v, g, bias): q,k,v [B, I, H, J, D] bf16 (cuEq / flash_triattn layout, I = rows of the attention frame), g [B, I, J, H*D] bf16
    PRE-sigmoid, bias [B, 1, H, I, J] fp32 (bias_frame='z' -> the axes-swapped strided view).  ln='stock': x_ln = the engine's own LayerNorm output
    in the attention frame ([I, J, C] / [B, I, J, C] bf16 contiguous) is projected (the exact-candidate construction); ln='fused': LN in-kernel."""
    if write_x:
        raise Unsupported("write_x", "returning the normalised x is not served (no caller today; file a CORE-ITEM)")
    import torch
    plan = _plan if _plan is not None else _plan_triattn(z, W, None, impl=impl, core="flash_triattn", ending=ending, ln=ln, residual=False)
    if impl == "fpf" and plan.get("prologue_impl") == "torch":
        impl = "torch"                                                    # the block planned the statements prologue (a composition cell: the fused rows were not selected in this form)
    if bias_frame not in ("x", "z"):
        raise Unsupported("bias_frame:" + str(bias_frame))
    z4 = z if z.dim() == 4 else z.unsqueeze(0)
    B = z4.shape[0]
    if ln == "stock":
        if x_ln is None:
            raise Unsupported("ln:stock-needs-x_ln")
        x4 = x_ln if x_ln.dim() == 4 else x_ln.unsqueeze(0)
        if tuple(x4.shape) != (B, z4.shape[1], z4.shape[2], W.C) or x4.dtype != torch.bfloat16 or not x4[0].is_contiguous():
            raise Unsupported("x_ln-shape", str(tuple(x4.shape)))
    else:
        x4 = None
    if impl == "fpf":
        PRO = _carried("fpf_triatt_pro", "prologue")
        row = plan["prologue"]; cfg = dict(row["cfg"]); variant = row["variant"]
        tt = row["triton"] if row.get("triton", "*") != "*" else _stack(z.device)[1]
        owner = _owner(W, z.device)
        cch = W._fpf_cache["triatt_pro"]
        outs = []
        for b in range(B):
            zb = z4[b]; xb = None if x4 is None else x4[b]
            if zb.dtype != torch.bfloat16:                                # fp32 residual stream
                if ln == "stock":
                    zb = xb.transpose(0, 1) if ending else xb                 # the kernel reads x_ln only; z names the frame
                else:                                                     # variant 'v3_fp32z' (the plan picks nothing else for fp32 z + fused LN)
                    base_cfg = {k: cfg[k] for k in ("BI", "BJ", "num_warps", "num_stages") if k in cfg}
                    outs.append(_launch_fpf_prologue_fp32z(PRO, cch, zb, ending=ending, cfg=base_cfg))   # LN in-kernel from fp32 z
                    continue
            if variant == "mkpf_f1":                                      # plan picked it only for ln='fused' on bf16 z
                MK = _carried("fpf_mkpf", "kernels")
                kcfg = {k: cfg[k] for k in ("BI", "BJ", "num_warps", "num_stages", "BN", "NUM_STAGES_W", "WS", "GROUP_I") if k in cfg}
                outs.append(MK.prologue_ln(owner, zb, cch, kcfg, ending=bool(ending), ln_arith=cfg.get("ln_arith", "fused"),
                                           fma_flags=tuple(bool(int(f)) for f in cfg.get("fma_flags", (1, 1, 1)))))
                continue
            if variant == "prologue_v4":                                  # plan picked it only for ln='stock' (x_ln given, attention frame, either node)
                GK = _carried("fpf_glue_v2", "kernels")
                outs.append(GK.prologue_v4(owner, xb, cch, cfg))
            else:                                                         # variant v3: the launch under the lever's safety net (a tuned cell that fails to BUILD -> the safe cell + ONE line)
                base_cfg = {k: cfg[k] for k in ("BI", "BJ", "num_warps", "num_stages") if k in cfg}
                outs.append(run_cell("pair_fused:prologue", row["cc"], tt, lambda c, zb=zb, xb=xb: PRO.triatt_prologue(
                    owner, zb, ending=ending, ln_mode=("stock" if ln == "stock" else "fused"), x_ln=xb, write_x=False, cfg=c),
                    base_cfg, dims=shape_dims("fpf", "prologue", (W.C, W.H, W.D)), key=(W.C, W.H, W.D)))
        def _stk(i):
            return torch.stack([o[i] for o in outs]) if B > 1 else outs[0][i].unsqueeze(0)
        q, k, v, g = _stk(0), _stk(1), _stk(2), _stk(3)
        if W.bqkvg is not None:                                           # gate bias: the fpf prologue kernel has no bias operand -> one bf16 add on g
            g = g + W.b_g                                                 # (tier-2 vs stock's fp32-accumulate-then-round addmm; lnl/torch impls fold it into the GEMM)
        bias = _stk(4).unsqueeze(1)                                       # [B, 1, H, I, J] fp32, projected in the attention (x) frame
        bias_in = "x"
    elif impl == "torch":
        import torch.nn.functional as Fn
        if ln == "stock":
            y16 = x4
        else:
            x4f = (z4.transpose(1, 2) if ending else z4).float()             # attention frame; LN in fp32 (stock: fp32 LN / autocast keeps LN fp32), one bf16 cast
            y16 = Fn.layer_norm(x4f, (W.C,), W.lnw32, W.lnb32, W.eps).to(torch.bfloat16)
        qkvg = Fn.linear(y16, W.wqkvg, W.bqkvg)                                    # [B, I, J, 4HD] one cuBLAS bf16 GEMM (fp32 accumulate, bf16 out = stock Linear under autocast)
        HD = W.H * W.D
        q = qkvg[..., 0:HD].unflatten(-1, (W.H, W.D)).permute(0, 1, 3, 2, 4)
        k = qkvg[..., HD:2 * HD].unflatten(-1, (W.H, W.D)).permute(0, 1, 3, 2, 4)
        v = qkvg[..., 2 * HD:3 * HD].unflatten(-1, (W.H, W.D)).permute(0, 1, 3, 2, 4)
        g = qkvg[..., 3 * HD:4 * HD]
        bias = Fn.linear(y16, W.wb).permute(0, 3, 1, 2).float().unsqueeze(1)              # [B, 1, H, I, J] = fp32(bf16 GEMM output), x frame
        bias_in = "x"
    else:
        RFU = _carried("lnl_fused")
        import torch.nn.functional as Fn
        if ln == "stock":
            y16 = x4                                                      # the engine's LN output, attention frame
            b16 = Fn.linear(y16, W.wb_pad)                                # [B, I, J, HB] bf16 GEMM output (x frame)
            bias_in = "x"
        else:
            # LN + bias projection in one kernel: y16 = LN(z) written in the attention frame (transposed row order for the ending node),
            # b16 = LN(z) @ w_b^T in z's OWN frame (bias_frame 'z' semantics: bias before the transpose)
            y16, b16 = RFU.ln_linear(z4, W.lnw32, W.lnb32, W.wb_pad, eps=W.eps, write_y=True, transpose=bool(ending))
            bias_in = "z"
        qkvg = Fn.linear(y16, W.wqkvg, W.bqkvg)                                    # [B, I, J, 4HD]: one cuBLAS GEMM on the bf16 LN output
        HD = W.H * W.D
        q = qkvg[..., 0:HD].unflatten(-1, (W.H, W.D)).permute(0, 1, 3, 2, 4)          # [B, I, H, J, D] strided views (flash_triattn addresses by stride)
        k = qkvg[..., HD:2 * HD].unflatten(-1, (W.H, W.D)).permute(0, 1, 3, 2, 4)
        v = qkvg[..., 2 * HD:3 * HD].unflatten(-1, (W.H, W.D)).permute(0, 1, 3, 2, 4)
        g = qkvg[..., 3 * HD:4 * HD]                                       # [B, I, J, HD] pre-sigmoid: a column slice of the fused output (row pitch 4HD)
        bias = b16[..., :W.H].permute(0, 3, 1, 2).float().unsqueeze(1)    # [B, 1, H, ., .] fp32 = fp32(bf16 GEMM output), stock rounding
    if ending and bias_in != bias_frame:
        bias = bias.transpose(-1, -2)                                     # LN is per position, so the other frame's bias is the axes-swapped view
    return q, k, v, g, bias


def _launch_fpf_prologue_fp32z(PRO, cch: dict, z, *, ending: bool, cfg: dict):
    """The carried prologue kernel on an fp32 pair tensor (LN_MODE=1: two-pass fp32 LayerNorm in-kernel, bf16 x_ln straight into the projections).
    This is fpf_triatt_pro.prologue.triatt_prologue's launch statement with the same argument list; the carried Python wrapper asserts bf16 z, the
    carried KERNEL loads z through .to(fl32) and is dtype-generic, so the fp32 stream costs no extra pass over z.  Returns (q, k, v, g, bias) in the x frame."""
    import torch, triton
    C, H, D, HB = cch["C"], cch["H"], cch["D"], cch["HB"]
    if z.dim() != 3 or z.shape[-1] != C or z.stride(-1) != 1:
        raise Unsupported("z-layout", f"{tuple(z.shape)} {z.stride()}")
    if ending:
        NI, NJ, s_zi, s_zj = z.shape[1], z.shape[0], z.stride(1), z.stride(0)
    else:
        NI, NJ, s_zi, s_zj = z.shape[0], z.shape[1], z.stride(0), z.stride(1)
    HD = H * D
    BI, BJ, NW, NS = cfg["BI"], cfg["BJ"], cfg["num_warps"], cfg["num_stages"]
    BN = min(128, HD)
    if HD % BN:
        raise Unsupported(f"hd:{HD}")
    dev = z.device
    q = torch.empty((NI, H, NJ, D), dtype=torch.bfloat16, device=dev)
    k = torch.empty((NI, H, NJ, D), dtype=torch.bfloat16, device=dev)
    v = torch.empty((NI, H, NJ, D), dtype=torch.bfloat16, device=dev)
    g = torch.empty((NI, NJ, HD), dtype=torch.bfloat16, device=dev)
    bias = torch.empty((H, NI, NJ), dtype=torch.float32, device=dev)
    grid = (triton.cdiv(NI, BI), triton.cdiv(NJ, BJ))
    PRO._triatt_prologue_kernel[grid](
        z, z, cch["lnw"], cch["lnb"], cch["wqkvg"], cch["wb"],
        q, k, v, g, bias, g,
        NI, NJ, s_zi, s_zj, cch["eps"],
        C=C, H=H, D=D, HB=HB, BI=BI, BJ=BJ, BN=BN,
        LN_MODE=1, WRITE_X=False, FMA_M2=True, FMA_MERGE=True, FMA_AFF=True, DOT_F32=False,
        num_warps=NW, num_stages=NS,
    )
    return q, k, v, g, bias


def tier_core_word(core) -> Optional[str]:
    """The kernels.triattn word of a ``core="tier:<word>"`` argument, else None (named cores / callables)."""
    if isinstance(core, str) and core.startswith(TIER_CORE_PREFIX) and len(core) > len(TIER_CORE_PREFIX):
        return core[len(TIER_CORE_PREFIX):]
    return None


def _core_ok(core) -> bool:
    return callable(core) or core == DEFAULT_CORE or core in CORES or tier_core_word(core) is not None


def default_core_off() -> Optional[str]:
    """The MODEL_OPT_LEVERS_OFF word that switches the default-core table off in this process, else None."""
    words = [w.strip() for w in os.environ.get("MODEL_OPT_LEVERS_OFF", "").replace(",", " ").replace(";", " ").split() if w.strip()]
    for w in DEFAULT_CORE_OFF_WORDS:
        if w in words:
            return w
    return None


def default_core_for(cc, dtype: str, head_dim: int, heads: int, n_keys: int) -> str:
    """The core ``core="default"`` serves at this call class: ``"tier:fast"`` where DEFAULT_CORE_TABLE lists (cc, dtype, head_dim, heads) and
    n_keys reaches its threshold, else ``"flash_triattn"``; ``"flash_triattn"`` everywhere when a DEFAULT_CORE_OFF_WORDS word is in
    MODEL_OPT_LEVERS_OFF.  cc as "8.0" / (8, 0) / 8.0."""
    if default_core_off():
        return "flash_triattn"
    ccw = ("%d.%d" % (int(cc[0]), int(cc[1]))) if isinstance(cc, (tuple, list)) else str(cc)
    thr = DEFAULT_CORE_TABLE.get((ccw, str(dtype), int(head_dim), int(heads)))
    return "tier:fast" if (thr is not None and int(n_keys) >= thr) else "flash_triattn"


def _tier_stack(T, word: str, cc) -> Optional[str]:
    """The prebuilt key kernels.triattn's CUDA rows are checked against for a tier word (the running process's own ABI key), as its serving
    call names it: the sm_90a member's key on cc 9.0, the sealed package's key on cc 8.0; None elsewhere / for words that never reach them."""
    base = str(word).split("@")[0]
    if not (base in ("fast", "big", "cuda_sm90a", "triattn_native")):
        return None
    if tuple(cc) == (9, 0):
        from opt_core.kernels.triattn import cuda_sm90a as _C
        return _C.stack_key()
    if tuple(cc) == (8, 0):
        from opt_core.kernels.triattn import triattn_native as _K
        return _K.stack_key()
    return None


def resolve_tier_core(word: str, cc, dtype, head_dim: int, heads: int, n_keys: int, *, form: str = "bias_only", direction: str = "fwd",
                      stack: Optional[str] = None, exact_stack: Optional[str] = None):
    """kernels.triattn's Selection for ``word`` at this attention call class -- (cc, dtype, head_dim, heads, keys per row, mask form,
    direction) -- exactly as its own serving call resolves it (``stack``: the prebuilt key its CUDA rows are checked against; None = the
    running process's key for the words that reach them).  Raises kernels.triattn.Refusal by name when no row of the word can serve."""
    from opt_core.kernels import triattn as T
    base = str(word).split("@")[0]
    if base == "big" and "big" not in getattr(T, "TIER_WORDS", ()):        # the big word where kernels.triattn records no peaks: big == fast, by name
        word, base = "fast" + str(word)[len("big"):], "fast"
    st = stack if stack is not None else _tier_stack(T, word, cc)
    xst = exact_stack
    if xst is None and (base == "exact" or T.rows().get(base, {}).get("class") == "exact"):
        xst = T.exact_stack_key(cc)
    return T.select(cc, dtype, int(head_dim), int(heads), int(n_keys), direction, word=word, stack=st, form=form, exact_stack=xst)


def _call_class(q, k):
    """-> (cc, dtype word, head_dim, heads, keys per row) of a core call: q [B, I, H, J, D] / k [B, I, H, S, D] in the attention frame."""
    import torch
    cc = tuple(torch.cuda.get_device_capability(q.device)) if q.is_cuda else (0, 0)
    dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else q.dtype
    dtype = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}.get(dt, str(dt))
    return cc, dtype, int(q.shape[-1]), (int(q.shape[-3]) if q.dim() >= 3 else 1), int(k.shape[-2])


def _tier_note(msg: str) -> None:
    print("[opt_core/pair_fused] " + msg, file=sys.stderr, flush=True)


def _tier_core_attention(word: str, q, k, v, bias, mask5, sc, tag: str = ""):
    """core="tier:<word>": the row kernels.triattn selects for <word> serves this core call (its own serving function, given the resolved
    Selection); a Refusal -- at resolution or at the call -- steps aside BY NAME to the flash_triattn cell (today's core), one stderr line per
    call class, counted on the LEVER line (cores=)."""
    import torch
    from opt_core.kernels import triattn as T
    cc, dtype, D, H, S = _call_class(q, k)
    direction = T.FWDBWD if (torch.is_grad_enabled() and (q.requires_grad or k.requires_grad)) else T.FWD
    form = "mask_bias" if mask5 is not None else "bias_only"
    key = (word, cc, dtype, D, H, S, form, direction)
    ent = _TIER_CORE_MEMO.get(key)
    if ent is None:
        try:
            sel = resolve_tier_core(word, cc, dtype, D, H, S, form=form, direction=direction)
            ent = ("row", sel)
            _tier_note("core tier:%s -> %s (cell %s%s; cc %d.%d %s D%d H%d keys %d %s)" % (word, sel.row, sel.cell, "" if sel.measured else " nearest", cc[0], cc[1], dtype, D, H, S, form))
        except T.Refusal as r:
            ent = ("refused", str(r.kind), getattr(r, "row", None), getattr(r, "fallback", None))
            _tier_note("core tier:%s refused by kernels.triattn (%s; row %s) -> flash_triattn by name (cc %d.%d %s D%d H%d keys %d %s)" % (word, ent[1], ent[2], cc[0], cc[1], dtype, D, H, S, form))
        _TIER_CORE_MEMO[key] = ent
    if ent[0] == "row":
        sel = ent[1]
        try:
            o = T.triangle_attention(q, k, v, bias, mask5, sc, word=word, selection=sel, form=form)
            nm = "%stier:%s=%s" % (tag, word, sel.row)
            SERVED_CORES[nm] = SERVED_CORES.get(nm, 0) + 1
            return o
        except T.Refusal as r:                                            # the row cannot serve THIS call (no prebuilt of the running stack, an offset bound ...): by name
            ent = ("refused", str(r.kind), sel.row, getattr(r, "fallback", None))
            _TIER_CORE_MEMO[key] = ent
            _tier_note("core tier:%s: row %s refused at the call (%s) -> flash_triattn by name (cc %d.%d %s D%d H%d keys %d %s)" % (word, sel.row, ent[1], cc[0], cc[1], dtype, D, H, S, form))
    nm = "%stier:%s=flash_triattn(refused:%s)" % (tag, word, ent[1])
    SERVED_CORES[nm] = SERVED_CORES.get(nm, 0) + 1
    return _carried("flash_triattn").flash_triangle_attention(q, k, v, bias, mask=mask5, scale=sc)


def _default_core_attention(q, k, v, bias, mask5, sc):
    """core="default": DEFAULT_CORE_TABLE's choice at this call class -- tier:fast (kernels.triattn's row, flash_triattn by name on its
    refusal) or the flash_triattn cell; counted as cores=default=... on the LEVER line."""
    cc, dtype, D, H, S = _call_class(q, k)
    if default_core_for(cc, dtype, D, H, S) == "tier:fast":
        return _tier_core_attention("fast", q, k, v, bias, mask5, sc, tag="default=")
    SERVED_CORES["default=flash_triattn"] = SERVED_CORES.get("default=flash_triattn", 0) + 1
    return _carried("flash_triattn").flash_triangle_attention(q, k, v, bias, mask=mask5, scale=sc)


def core_attention(q, k, v, bias, mask5=None, *, core: Any = DEFAULT_CORE, scale: Optional[float] = None):
    """o [B, I, H, J, D] = softmax_j(scale q.k + bias (+ -1e9 at masked keys)) v — ``core="default"`` (the default): the block's own choice
    per call class (DEFAULT_CORE_TABLE: tier:fast where the block was measured faster with kernels.triattn's row, the flash_triattn cell
    elsewhere; MODEL_OPT_LEVERS_OFF word pair_fused:tier_core = flash_triattn everywhere); ``"flash_triattn"``: the core's flash_triattn cell;
    ``"k2b"``: the K2B cell; ``"tier:<word>"``: the row kernels.triattn selects for <word> (flash_triattn by name when it refuses); or
    ``core(q,k,v,bias,mask,scale)``.  An explicit core word always wins over the table."""
    D = q.shape[-1]
    sc = (1.0 / math.sqrt(D)) if scale is None else float(scale)
    if callable(core):
        SERVED_CORES["callable"] = SERVED_CORES.get("callable", 0) + 1
        return core(q, k, v, bias, mask5, sc)
    if core == DEFAULT_CORE:                                               # the block's own choice per call class (DEFAULT_CORE_TABLE): tier:fast where measured faster, else flash_triattn
        return _default_core_attention(q, k, v, bias, mask5, sc)
    tw = tier_core_word(core)
    if tw is not None:                                                     # kernels.triattn's row for <word> at this call class, flash_triattn by name on its Refusal
        return _tier_core_attention(tw, q, k, v, bias, mask5, sc)
    if core == "flash_triattn":
        SERVED_CORES["flash_triattn"] = SERVED_CORES.get("flash_triattn", 0) + 1
        return _carried("flash_triattn").flash_triangle_attention(q, k, v, bias, mask=mask5, scale=sc)
    if core == "k2b":                                                      # K2B = flash_triattn v9 + base-2 softmax domain + MAXNREG cap (same call convention)
        SERVED_CORES["k2b"] = SERVED_CORES.get("k2b", 0) + 1
        return _carried("fpf_triatt_k2b").attn_k2b(q, k, v, bias, mask=mask5, scale=sc)
    raise Unsupported("core:" + str(core))


def epilogue(o, g, W: TriAttnWeights, z=None, *, ending: bool = False, residual: bool = False, impl: str = "fpf", o_layout: str = "ihjd",
             out=None, _plan=None):
    """u = bf16(sigmoid(g) * o) @ w_o^T (+ b_o).  residual=False -> the update in the ATTENTION frame's leading dims ([B, I, J, C_out]);
    residual=True -> z (its OWN frame, [B,N,N,C] / [N,N,C]) is updated in place (z[.., j, i] += u[i, j] for the ending node) and returned."""
    import torch
    if impl == "fpf" and _plan is not None and _plan.get("epilogue_impl") == "lnl":
        impl = "lnl"                                                      # the block planned the statements piece (the fused row is measured off here)
    if impl == "fpf":
        if _plan is not None:
            plan = _plan
        else:
            try:
                plan = {"epilogue": pick_cell("fpf", "epilogue", (W.C_out, W.H, W.D), o.device, ("epilogue_v3", "v2") if o_layout == "ihjd" else ("v2",))}
            except Unsupported as e_:                                     # measured off on this key / cc -> the lnl statements piece BY NAME (one note per key)
                _epilogue_statements_piece(e_, W, o.device, pinned=False)
                return epilogue(o, g, W, z, ending=ending, residual=residual, impl="lnl", o_layout=o_layout, out=out)
        row = plan["epilogue"]; cfg = dict(row["cfg"]); variant = row["variant"]
        zz = None
        if residual:
            if z is None:
                raise Unsupported("residual-needs-z")
            zz = z if z.dim() == 4 else z.unsqueeze(0)
        if variant == "epilogue_v3" and o_layout == "ihjd":                # KVER=3 kernel reads the cuEq layout only
            GK = _carried("fpf_glue_v2", "kernels")
            u = GK.epilogue_v3(o, g, W.wo16, zz, ending=ending, residual=residual, out=out, cfg=cfg, woT16=W.woT16)
        else:                                                             # variant v2: the launch under the lever's safety net (a tuned cell that fails to BUILD -> the safe cell + ONE line)
            EPI = _carried("fpf_triatt_epi", "epilogue")
            base = {k: cfg[k] for k in ("KVER", "BI", "BJ", "num_warps", "num_stages", "EXP") if k in cfg}
            tt = row["triton"] if row.get("triton", "*") != "*" else _stack(o.device)[1]
            u = run_cell("pair_fused:epilogue", row["cc"], tt, lambda c: EPI.triatt_epilogue(o, g, W.wo16, zz, ending=ending, residual=residual, out=out,
                                                                                          cfg=c, woT16=W.woT16, o_layout=o_layout),
                         base, dims=shape_dims("fpf", "epilogue", (W.C_out, W.H, W.D)), key=(W.C_out, W.H, W.D))
        if W.b_o is not None:
            if residual:
                zz += W.b_o                                               # bias is frame-free per position
            else:
                u = u + W.b_o
        return z if residual else u
    if impl == "torch":
        import torch.nn.functional as Fn
        o5 = o if o.dim() == 5 else o.unsqueeze(0)
        o_t = o5.permute(0, 1, 3, 2, 4) if o_layout == "ihjd" else o5              # [B, I, J, H, D] view
        g4 = g if g.dim() == 4 else g.unsqueeze(0)
        gs = torch.sigmoid(g4.float()).to(torch.bfloat16)                         # ATen sigmoid under opmath == sigmoid(float).to(bf16)
        gated = (o_t.float() * gs.unflatten(-1, (W.H, W.D)).float()).to(torch.bfloat16).flatten(-2)     # [B, I, J, HD]
        u = Fn.linear(gated, W.wo16, W.b_o)                                       # [B, I, J, C_out] attention frame
        if residual:
            if z is None:
                raise Unsupported("residual-needs-z")
            zz = z if z.dim() == 4 else z.unsqueeze(0)
            zz += (u.transpose(1, 2) if ending else u)
            return z
        return u
    # lnl: gate_transpose (sigmoid(g)*o, transposed row order for the ending node) then one cuBLAS GEMM
    RFU = _carried("lnl_fused")
    import torch.nn.functional as Fn
    o5 = o if o.dim() == 5 else o.unsqueeze(0)
    if o_layout == "ijhd":
        o5 = o5.permute(0, 1, 3, 2, 4)
    o5 = o5.contiguous()
    g4 = g if g.dim() == 4 else g.unsqueeze(0)
    HD = W.H * W.D
    if g4.stride(-1) == 1 and g4.stride(-2) >= HD and g4.dim() == 4:
        # g may be a column slice of the fused qkvg output: gate_transpose reads (g_src, column offset)
        base = g4._base if (g4._base is not None and g4._base.dim() == 4 and g4._base.is_contiguous() and g4._base.shape[:3] == g4.shape[:3]) else None
        if base is not None:
            goff = (g4.data_ptr() - base.data_ptr()) // g4.element_size()
            y = RFU.gate_transpose(o5, base, int(goff), transpose=bool(ending))
        else:
            y = RFU.gate_transpose(o5, g4.contiguous(), 0, transpose=bool(ending))
    else:
        y = RFU.gate_transpose(o5, g4.contiguous(), 0, transpose=bool(ending))
    u = Fn.linear(y, W.wo16, W.b_o)                                       # [B, (J,I | I,J), C_out] — for ending already in z's frame
    if residual:
        if z is None:
            raise Unsupported("residual-needs-z")
        zz = z if z.dim() == 4 else z.unsqueeze(0)
        zz += u
        return z
    if ending:
        return u.transpose(1, 2)                                          # back to the attention frame's [B, I, J, C] (view) — same contract as fpf op mode
    return u


def tri_attn_block(z, W: TriAttnWeights, mask=None, *, ending: bool = False, residual: bool = False, impl: str = "fpf",
                   core: Any = DEFAULT_CORE, ln: str = "fused", x_ln=None, scale: Optional[float] = None, bias_frame: str = "x", variant=None,
                   engage_cells: bool = False):
    """The fused triangle-attention block: prologue -> core -> epilogue.  Returns the update u in z's OWN frame ([.., N, N, C_out]) when
    residual=False (for the ending node u is transposed back as a VIEW — call .contiguous() if the caller needs it), or z updated in place when
    residual=True.  Raises :class:`Unsupported` before any launch."""
    plan = _plan_triattn(z, W, mask, impl=impl, core=core, ending=ending, ln=ln, residual=residual, variant=variant, engage_cells=engage_cells)
    _count_rows(plan)
    m5 = _key_mask(mask, z, ending)
    q, k, v, g, bias = prologue(z, W, ending=ending, ln=ln, x_ln=x_ln, impl=plan.get("prologue_impl", impl), bias_frame=bias_frame, _plan=plan)
    o = core_attention(q, k, v, bias, m5, core=core, scale=scale)
    u = epilogue(o, g, W, z, ending=ending, residual=residual, impl=plan.get("epilogue_impl", impl), o_layout="ihjd", _plan=plan)
    if residual:
        return u
    u = u if z.dim() == 4 else u[0]
    return u.transpose(-2, -3) if ending else u


# ------------------------------------------------------------------------------------------------------------------------------------ transition
def transition(x, T: TransitionWeights, *, residual: bool = False, ln: str = "fused", x_ln=None, impl: str = "fpf", variant=None, out=None,
               mask=None, binary_mask=None, variants=None, silu: str = "swiglu"):
    """y = W_out(silu(W_a n) * W_b n) with n = LN(x) (ln='fused', in-kernel) or n = x_ln (ln='stock': the engine's own LayerNorm output, the
    exact-candidate construction); returns the update, or x + update when residual=True (folded in the kernel epilogue with stock rounding).
    x [..., C] bf16; the n*C hidden never reaches HBM.  mask [...]: the transition's row mask (one value per row of x, any bool / integer /
    float dtype) folded by the v2 kernel — update * mask before the residual add, `binary_mask=True` = every value is 0 or 1 (folded into the
    GEMM input rows), False = general values (epilogue multiply), None = True for bool masks only; a variant other than v2_fold refuses a mask by
    name.  out may be x itself under v2_fold (in place: each program re-reads its rows before its stores).  variants: the caller's ORDERED
    preference among the variants of its call — v2_fold launches only for a caller that names it here (or pins variant='v2_fold'); every other
    caller keeps v1 / v1_fp32x / transition_v3.  silu: 'swiglu' (h = bf16(silu(a)) * b rounded again: the plain statement) | 'liger' (h = silu(fp32 a) *
    fp32 b rounded once: the Liger-Kernel SiLU*b form) -- the v1 launch only; any other variant refuses 'liger' by name."""
    import torch
    if silu not in ("swiglu", "liger"):
        raise Unsupported("silu-form-%s" % silu)
    plan = _plan_transition(x, T, impl=impl, ln=ln, residual=residual, variant=variant, mask=mask, variants=variants)
    if silu == "liger" and not (plan["impl"] == "fpf" and plan["transition"]["variant"] in ("v1", "v1_fp32x")):
        raise Unsupported("silu-liger-needs-v1", str(plan.get("transition", {}).get("variant")))
    _count_rows(plan)
    C = T.C
    lead = tuple(x.shape[:-1])
    x2 = x.view(-1, C) if (out is not None and out is x) else x.reshape(-1, C)   # in place needs the rows as a VIEW (a copy would be written and dropped)
    if plan["impl"] == "fpf" and plan["transition"]["variant"] == "v2_fold":
        row = plan["transition"]; cfg = dict(row["cfg"])
        V2 = _carried("fpf_transition_v2", "kernel")
        P = T.v2_packed
        if P is None or P.wo.device != x2.device:
            P = V2.pack_weights(T.lnw32, T.lnb32, T.wa16, T.wb16, T.wo16, T.eps, x2.device)
            T.v2_packed = P
        out2 = out.view(-1, C) if out is not None else torch.empty(x2.shape, dtype=torch.bfloat16, device=x2.device)
        m1 = mask.reshape(-1) if mask is not None else None             # one value per row (the plan refused any other count by name)
        from opt_core.kernels import safe_settings as _ss
        try:
            V2._check(x2, P, m1, out2)                                    # the kernel's own by-name domain words (strides, overlap, mask dtype, rows)
            V2.launch(x2, P, m1, bool(residual), out2, cfg, binary_mask=binary_mask, numerics="exact")   # the row's own settings; no other variant's safe settings stand in for this kernel
        except V2.Unsupported as e:
            raise Unsupported("v2:" + e.reason) from None
        except _ss.catchable() as e:
            if not _ss.is_build_failure(e):
                raise
            raise Unsupported(f"no-cell:pair_fused:transition:build_failed:{type(e).__name__}", _ss.where_word(row["cc"], _stack(x2.device)[1])) from e   # the caller's per-call route to its own statement, by name
        return out2.view(lead + (C,)) if out is not None else out2.reshape(lead + (C,))
    if ln == "stock":
        if x_ln is None:
            raise Unsupported("ln:stock-needs-x_ln")
        y2 = x_ln.reshape(-1, C)
        if y2.shape != x2.shape or y2.dtype != torch.bfloat16 or y2.stride(-1) != 1:
            raise Unsupported("x_ln-shape", str(tuple(x_ln.shape)))
    if impl == "fpf":
        row = plan["transition"]; cfg = dict(row["cfg"]); variant = row["variant"]
        TR = _carried("fpf_transition", "transition")
        out2 = out.reshape(-1, C) if out is not None else torch.empty(x2.shape, dtype=torch.bfloat16, device=x2.device)
        res2 = x2 if residual else None
        src2 = y2 if ln == "stock" else x2
        if variant == "transition_v3" and ln == "stock":
            GK = _carried("fpf_glue_v2", "kernels")
            GK.transition_v3(src2, T.cache, out2, res2d=res2, cfg=cfg)
        else:                                                             # the v1 launch under the lever's safety net: a tuned cell that fails to BUILD -> the safe cell + ONE line
            run_cell("pair_fused:transition", row["cc"], _stack(x2.device)[1] if row.get("triton", "*") == "*" else row["triton"],
                     lambda c: _launch_fpf_transition(TR, src2, T.cache, out2, res2, ln_mode=(0 if ln == "stock" else 1), cfg=c, silu=(1 if silu == "liger" else 0)), cfg,
                     dims=shape_dims("fpf", "transition", (T.C, T.NH)), key=(T.C, T.NH))
        return out2.reshape(lead + (C,))
    RFU = _carried("lnl_fused")
    if ln == "stock":
        raise Unsupported("ln:stock:lnl")                                 # lnl's kernel always normalises in-kernel
    y = RFU.fused_transition(x, T.lnw32, T.lnb32, T.wa16, T.wb16, T.wo16, eps=T.eps, residual=bool(residual))
    return y


def _launch_fpf_transition(TR, y2d, cache, out2d, res2d, ln_mode: int, cfg: dict, silu: int = 0):
    """fpf_transition.transition._launch with the CELL ROW's configuration (the carried _launch reads only its own pinned table; this is the
    same launch statement with cfg supplied by the core's table)."""
    import triton
    M, C = y2d.shape
    wo16 = cache["wo16"]
    NH = wo16.shape[1]
    BM, BH, IL = cfg["BM"], cfg["BH"], cfg.get("IL", 0)
    if NH % BH != 0:
        raise Unsupported("cfg:BH", f"NH={NH} BH={BH}")
    if not (y2d.stride(1) == 1 and out2d.stride(1) == 1 and (res2d is None or res2d.stride(1) == 1)):
        raise Unsupported("stride")
    wab = cache["wabI16"] if IL == 1 else cache["wab16"]
    CP = triton.next_power_of_2(C)
    if M == 0:
        return out2d
    grid = (triton.cdiv(M, BM),)
    res = res2d if res2d is not None else out2d
    TR._fused_transition_kernel[grid](y2d, wab, wo16, out2d, res, cache["lnw16"], cache["lnb16"], M,
                                      y2d.stride(0), wab.stride(0), wo16.stride(0), out2d.stride(0), res.stride(0), cache["eps"],
                                      C=C, CP=CP, NH=NH, BM=BM, BH=BH, IL=IL, HAS_RES=(res2d is not None), LN_MODE=ln_mode, SILU=silu,
                                      num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    return out2d


def ln_linear(x, ln_w, ln_b, W, *, eps: float = 1e-5, write_y: bool = False, transpose: bool = False, impl: str = "lnl"):
    """(y16 | None, out) = (bf16(LN(x)) [optionally transposed row order], bf16(LN(x) @ W^T)) — LN + a small projection in one kernel (the
    AttentionPairBias / tri-attention bias prologue).  W [NOUT, C] bf16, NOUT a power of two >= 16 (pad); lnl kernel: C in {64, 128}."""
    if impl != "lnl":
        raise Unsupported("impl:" + str(impl))
    import torch
    if not x.is_cuda:
        raise Unsupported("device")
    C = x.shape[-1]
    find_cell("lnl", "ln_linear", (int(C),), x.device)
    RFU = _carried("lnl_fused")
    return RFU.ln_linear(x, ln_w, ln_b, W, eps=eps, write_y=write_y, transpose=transpose)


# -------------------------------------------------------------------------------------------------------------------------------------- evidence
LEVER_NAME = "F3.pair_fused"


def _impl_string() -> str:
    """One short marker per carried kernel this lever uses: ``<name>@<8 hex chars>``, the sha256 of the core's own copy of that
    file RIGHT NOW (computed fresh, held nowhere) -- a run-tracking equality stamp, not a checking manifest."""
    import hashlib

    from opt_core import kernels as K
    main = {"fpf_triatt_pro": "prologue.py", "fpf_triatt_epi": "epilogue.py", "fpf_transition": "transition.py", "fpf_transition_v2": "kernel.py", "fpf_glue_v2": "kernels.py",
            "fpf_mkpf": "kernels.py", "fpf_triatt_k2b": "triatt_k2b.py", "lnl_fused": "lnl_fused.py", "flash_triattn": "flash_triattn.py"}
    parts = []
    for n, f in main.items():
        try:
            root = K.carried_path(n) if K.sums(n)["kind"] == "package" else K.KERNELS_DIR
            with open(os.path.join(root, f), "rb") as fh:
                parts.append(f"{n}@{hashlib.sha256(fh.read()).hexdigest()[:8]}")
        except Exception:
            parts.append(f"{n}@?")
    return "+".join(parts)


def ledger(name: str = LEVER_NAME, *, expected: Sequence[str] = (), min_tokens: int = 0, **kw):
    """The per-process ``opt_core.counters.Ledger`` of this lever (``name=F3.pair_fused``, ``impl`` = the carried cells' shas, ``origin=core``); the
    kit calls ``.serve(shape)`` / ``.fallback(reason)`` (an :class:`Unsupported` ``.reason``) / ``.gate()`` and prints ONE line with :func:`emit_line`."""
    from opt_core.counters import Ledger
    return Ledger(name, impl=_impl_string(), origin="core", min_tokens=int(min_tokens), expected=tuple(expected), **kw)


def _evidence_head() -> dict:
    ev = {"cells": cells_sha256()[:8], "served_certified": SERVED_BY_STATUS["certified"], "served_candidate": SERVED_BY_STATUS["candidate"],
          "allow_candidate": int(allow_candidate()), "variants": ",".join(f"{k}:{v}" for k, v in sorted(SERVED_VARIANTS.items())) or "-",
          "cores": ",".join(f"{k}:{v}" for k, v in sorted(SERVED_CORES.items())) or "-"}
    if SERVED_PIECES:                                                     # pieces served through another impl by cell (the fused piece measured off there)
        ev["pieces"] = ",".join(f"{k}:{v}" for k, v in sorted(SERVED_PIECES.items()))
    if _K2B_TABLE_ID[0]:
        ev["k2b_table"] = _K2B_TABLE_ID[0]
    if _K2B_NOTE[0]:
        ev["k2b_cells_note"] = _K2B_NOTE[0]
    return ev


def evidence_tail() -> dict:
    """The LAST fields of the lever line — the grammar every card's expectations read: ``keys=<impl>.<piece>:<key>[+<key>][,…]`` = the table key
    that served each piece in this process (``<cc>|<triton M.m>`` a named-exception row, ``<cc>|*`` the capability's row, ``safe`` the lever's
    safe settings, ``default`` an impl's default settings; ``-`` before any call), then ``settings=safe:<reason>[;…]`` ONLY when a safety net serves
    in this process (opt_core.kernels.safe_settings: ``no_cell:<key>`` / ``build_failed:<exception class>``), then ``cells_note=<lever>:default:no_row[;…]``
    ONLY when default settings serve an UNKNOWN key (:meth:`SafeNet.inform`)."""
    ev = {"keys": ",".join(f"{p}:{'+'.join(sorted(ks))}" for p, ks in sorted(SERVED_KEYS.items())) or "-"}
    words = [n.word() for _, n in sorted(_NETS.items()) if n.on]
    if words:
        ev["settings"] = ";".join(words)
    notes = [f"{name}:{n.note()}" for name, n in sorted(_NETS.items()) if n.note()]
    if notes:
        ev["cells_note"] = ";".join(notes)
    return ev


def evidence() -> dict:
    """The lever-line evidence fields of this process: the cell table sha, rows served by qualification status, the candidate switch, the
    variants served — then, LAST, the served-keys tail (:func:`evidence_tail`)."""
    ev = _evidence_head()
    ev.update(evidence_tail())
    return ev


def emit_line(L, tag: str, **more) -> str:
    """Print this lever's ONE activation-evidence line (``report.emit(L.line(tag, cells=…, served_certified=…, …, <more>, keys=…[, settings=…]))``):
    the served-keys tail is always the end of the line.  A keyword in ``more`` that the face also produces (e.g. a kit's own ``cores=``) is the
    caller's: the face's value is printed as ``face_<key>=`` instead (:func:`_face_fields`)."""
    from opt_core.report import emit
    head, tail = _face_fields(_evidence_head(), more), _face_fields(evidence_tail(), more)
    return emit(L.line(tag, **head, **more, **tail))


def _face_fields(fields: dict, caller: dict) -> dict:
    """The face's evidence fields next to a caller's own keywords: a key the caller passes too is the CALLER's (its token stays where its
    parsers read it) and the face's value moves to ``face_<key>`` (dropped if the caller names that as well) -- never a duplicate keyword."""
    out = {}
    for k, v in fields.items():
        if k in caller:
            if ("face_" + k) not in caller:
                out["face_" + k] = v
        else:
            out[k] = v
    return out


def describe() -> dict:
    from opt_core import kernels as K
    import opt_core
    out = {"opt_core": opt_core.__version__, "cells_sha256": cells_sha256(), "impls": list(IMPLS), "cores": list(CORES),
           "default_core": {"word": DEFAULT_CORE, "table": {"%s|%s|D%d|H%d" % k_: {"tier:fast_from_keys": v_} for k_, v_ in sorted(DEFAULT_CORE_TABLE.items())},
                            "elsewhere": "flash_triattn", "off_word": default_core_off()}, "carried": {}, **evidence()}
    for n in _CARRIED:
        try:
            out["carried"][n] = {"version": K.sums(n)["version"], "files": K.sums(n)["files"]}
        except Exception as e:
            out["carried"][n] = {"error": repr(e)}
    out["compositions"] = [dict(r) for r in composition_rows()]                 # pieces served through the statements by cell
    return out


# ----------------------------------------------------------------------------------------------- caller-operand guard on the public doors
def _named_assertions(door: str):
    """Never an assertion out of a serving door: an ``AssertionError`` raised under the launch (the carried pieces assert their operand
    contracts -- strides, dtypes, shapes) surfaces as ``Unsupported('assert:<door>', <the assertion>)``, the same named step-aside the planner's
    own checks raise, so the binder's statements block serves.  The planner's named checks on z / mask / x_ln / x precede every launch; this
    guard covers what a caller-built operand (o / g / out on the standalone doors) could still trip."""
    import functools

    def deco(fn):
        @functools.wraps(fn)
        def guarded(*a, **k):
            try:
                return fn(*a, **k)
            except AssertionError as e:
                raise Unsupported("assert:" + door, (str(e).strip() or "assert").replace("\n", " ")[:160]) from None
        return guarded
    return deco


prologue = _named_assertions("prologue")(prologue)
epilogue = _named_assertions("epilogue")(epilogue)
tri_attn_block = _named_assertions("tri_attn_block")(tri_attn_block)
transition = _named_assertions("transition")(transition)
ln_linear = _named_assertions("ln_linear")(ln_linear)
core_attention = _named_assertions("core_attention")(core_attention)
