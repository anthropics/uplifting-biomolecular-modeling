"""L2 provider — the row-block statements of the triangle multiplicative update as FUSED kernels for :func:`trimul_update_`.

:func:`fused_trimul_fns` wraps an engine's :class:`opt_core.mem.rowpair.trimul.TriMulFns` (its torch statements, kept as the named fallback)
and adds the two optional hooks the sharded driver looks for, served by the carried ``fpf_trimul_v4`` kernels (:mod:`opt_core.kernels`):

``proj_into(dst, z_block, mask_block, is_a)``   K1 (``kernels._k1c``, pointer loads): LayerNorm_in (fp32 statistics) -> the a OR b gated
                                                projection (``sigmoid(x W_g^T + b_g) * (x W_p^T + b_p) * mask``) -> bf16, written channel-major
                                                DIRECTLY into ``dst [C_h, rows, N]`` (the resident GEMM-A block of a / the row plane of a b
                                                sub-block) — one launch per row block, no permute / copy. The kernel serves both projections in one
                                                launch at P=1; here it is launched with one projection's weight columns (``D2 = C_h``), a
                                                compile-time specialisation of the same source.
``tile_epilogue(T, z_block, add)``              K3 (``kernels._k3c``): ``x = T[C_h, rows, w]`` (the contraction tile) -> LayerNorm_out ->
                                                ``W_o`` -> gate ``sigmoid(LayerNorm_in(z) W_og^T + b_og)`` recomputed from the ORIGINAL block ->
                                                ``(+ z)`` -> the block, one launch. The kernel addresses a pair tensor whose row stride equals its
                                                width; an epilogue tile is a column window of the shard, so the window is staged contiguous
                                                (one copy in, the launch in place, one copy out — ``2 * rows * w * C_z`` elements per tile).

Served cell — decided per launch from facts that are identical on every rank (device class, dtypes, widths, the cells table, environment):
``z`` bf16 or fp32 (fp32: the kernels' ``IN_F32`` variants — LayerNorm statistics from the fp32 values, bf16 planes and contraction, the
residual added in fp32); ``C_z``, ``C_h`` in {128, 256}; pair size ``N >= min_tokens`` (:data:`DEFAULT_MIN_TOKENS`). Those three are the
lever's documented GATES: a unit outside them returns False and the driver runs the torch statements for it, COUNTED (``dtype:<dt>``,
``c=<Cz>/<Ch>``, ``below_gate`` — the row-block launches do not pay for themselves below the size gate), as does the opt-out
``ROWPAIR_TRIMUL_KERNELS=torch`` (``env_torch``). Launch settings: the ``table.json`` row for this device's ``cc|triton``
(:mod:`fpf_trimul_v4.cells`, the package's one table ``fpf_trimul_v4/table.json`` — the same table and loader its single-GPU line reads,
with ONE extra rule, :func:`pointer_row`: the row block runs the POINTER K1, so when the resolved row is a TMA row and the table carries a
served ``<cc>|*`` pointer row, that row serves — census fact ``cells=<row key>``), or an explicit ``cells=`` mapping; a card WITHOUT a row is
served with the lever's SAFE settings (:mod:`opt_core.kernels.safe_settings`, lever ``pair_fused:trimul``: ONE stderr line, census
``settings=safe:no_cell:rows``, ``cells=safe``), and settings that fail to BUILD switch the process to the safe settings the same way
(``settings=safe:build_failed:<Type>``). Where the lever cannot run at all — no CUDA device, no Triton, the carried kernels not importable, no
row and no safe settings for the card, the safe settings failing to build — the provider raises :class:`RowpairRefused` naming the lever
and the opt-out (``--mode off`` / ``ROWPAIR_TRIMUL_KERNELS=torch``); never a silent plain path. Any other kernel exception propagates.

Widths BESIDE fpf_trimul_v4's — the ROWS UNIT (:data:`ROWS_UNIT` = :mod:`opt_core.kernels.fpf_trimul_rows`, c_z / c_hidden 64 and 384; its
kernels ``_k1r`` / ``_k3r`` are fpf_trimul_v4's K1 / K3 statements reading the channel-last row block and the epilogue's column window IN
PLACE through strides, 384 walked as 3 x 128 channel chunks): a (C_z, C_h) pair outside :data:`SUPPORTED_C` is served by that unit iff its
ROWS-ONLY table (``fpf_trimul_rows/table_rows.json``, keyed ``<cc>|C<c_z>|H<c_h>``; consulted here and nowhere else — no whole-plane
provider reads it) carries a MEASURED key for this card (facts ``unit=fpf_trimul_rows@<ver>`` ``cells=rows:<key>`` ``k1`` ``k1b`` ``k3``
``k1_impl=rows:a|a+b``, schedule ``trimul_kernels=fpf_rows``); an UNMEASURED (``PLACEHOLDER``) key serves only under the engineering switch
``ROWPAIR_TRIMUL_ROWS_UNMEASURED=1``; otherwise the unit answers the SAME decline word those widths answered before it existed
(``c=<Cz>/<Ch>``, the torch statements, counted; fact ``rows_table=<key>:<reason>`` says why). Its K1 also writes a b sub-block in the GEMM-B
layout directly when the key says ``b_direct`` (:meth:`FusedTriMulFns.b_transposed`, asked by the driver; ``_b_slab``'s transpose copy then
disappears); its K3 needs no staging copy. Its safety net is its own (lever word ``pair_fused:trimul_rows``, SAFE cells = the table's ``safe``
member). The same table may name a size floor per (cc, C_z, C_h) — ``min_tokens`` — for ANY width incl. fpf_trimul_v4's (fact
``min_tokens_source=rows_table:<key>``); a kit's explicit ``min_tokens=`` wins over it.

Numerics: the fused single-GPU TriMul's rounding points (LN output bf16; gated projection rounded after gating on fp32 accumulators; bf16
contraction operands with fp32 accumulation; LN_out bf16; gated output bf16; residual in fp32) with the row-sharded schedule's tile-wise GEMM
calls — Tier 2 (tolerance class) against the torch statements and against the single-GPU fused kernel; run-to-run repeatable (fixed cells, no
atomics, no split-K). The GEMM operand / ring dtype of a served shard is bf16 (``operand_dtype``) whatever z's dtype; a declined shard runs the
statements in z's dtype.

Census: :data:`LEDGER` (``name=F2.trimul_rows``; ``served`` = kernel launches by kind, ``fallback_by`` = declines by reason, facts ``cells``
``k1`` ``k3`` ``k1_impl``) -> :func:`emit_line`; schedule fields ``trimul_kernels`` ``trimul_k1_launches`` ``trimul_k3_launches``
(:func:`opt_core.mem.rowpair.evidence.record_schedule`).

Usage (the kit's adapter; ``weights`` in the :data:`opt_core.trimul_weights.WEIGHT_KEYS` vocabulary its single-GPU ``fpf_v4`` provider already uses)::

    from opt_core.mem.rowpair import trimul as RT, trimul_fused as RF
    fns = RF.fused_trimul_fns(weights_of(module), stock_fns)          # stock_fns: the TriMulFns of the module's torch statements
    RT.trimul_update_(fns, z_shard, mask_shard, layout, outgoing=True, inplace_chunk=256)
    ...  RF.emit_line("my_kit")                                        # once per process
"""
from __future__ import annotations

import os
import sys
from typing import Any, Mapping, Optional

from ...counters import Ledger
from ...kernels import fpf_trimul_v4 as UNIT                 # the carried kernel unit's PACKAGE (its name and version; the Triton modules load at the first served call)
from ...kernels import safe_settings as SAFE
from ...trimul_weights import BIAS_KEYS, WEIGHT_KEYS
from . import RowpairRefused
from .evidence import record_schedule
from .trimul import TriMulFns

__all__ = ["LEVER", "KERNEL", "ENV_KERNELS", "KERNEL_WORDS", "DEFAULT_MIN_TOKENS", "SAFE_LEVER", "LEDGER", "FusedTriMulFns", "fused_trimul_fns", "pointer_row",
           "resolve_cells", "emit_line", "describe", "SUPPORTED_C", "ROWS_UNIT", "ROWS_SAFE_LEVER", "rows_widths", "resolve_rows"]

LEVER = "F2.trimul_rows"
KERNEL = UNIT.__name__.rsplit(".", 1)[1]                  # the census word ``impl=<unit>@<version>``
ENV_KERNELS = "ROWPAIR_TRIMUL_KERNELS"                 # fpf_v4 (default) | torch — engineering switch; the kit selects by constructing the provider or not
KERNEL_WORDS = ("fpf_v4", "torch")
DEFAULT_MIN_TOKENS = 2048              # pair size N below which every unit runs the torch statements (reason below_gate): on H100 (bf16, C 128 / 256,
                                       # 2 ranks) the fused row-block path is slower than the statements per call below ~2048 tokens and faster above;
                                       # a (capability, C_z, C_h) entry of the ROWS table (ROWS_UNIT table_rows.json `min_tokens`) replaces it for that
                                       # width on that card when the kit passed no `min_tokens=` of its own (fact min_tokens_source=rows_table:<key>)
SAFE_LEVER = "pair_fused:trimul"                        # the safe-settings row of the carried kernels (:data:`opt_core.kernels.safe_settings.SAFE_ROWS`)
SUPPORTED_C = (128, 256)                               # pair width C_z and hidden width C_h the carried WHOLE-PLANE kernels serve (fpf_trimul_v4 SUPPORTED_C / SUPPORTED_D): these
                                                       # widths keep that unit's kernels + table.json row on the row blocks, unchanged
ROWS_UNIT = KERNEL.replace("_v4", "_rows")             # = the package name of the row-block members BESIDE those widths (opt_core.kernels.fpf_trimul_rows: c_z / c_hidden 64 and 384): served only where
                                                       # its rows-only table carries a MEASURED (cc, C_z, C_h) key — else today's decline word c=<Cz>/<Ch>
ROWS_SAFE_LEVER = "pair_fused:trimul_rows"             # that unit's safety-net lever word (its SAFE cells are its table's `safe` member)
LEDGER = Ledger(LEVER, impl=KERNEL, origin="core")
_PRINTED = set()
_COUNTS = {"k1": 0, "k3": 0}


def _log_once(key: str, msg: str) -> None:
    if key not in _PRINTED:
        _PRINTED.add(key)
        sys.stderr.write("[opt_core.rowpair] trimul_rows %s\n" % msg)
        sys.stderr.flush()


def _kernels():
    """``(kernels module, cells module)`` of the carried kernel unit — its core copy, imported as :mod:`opt_core.kernels` submodules at the
    first served call (Triton and torch load then, not at this module's import). Fact ``kernel_copy=core``."""
    from ...kernels.fpf_trimul_v4 import cells as C, kernels as K
    return K, C


def _rows_face():
    """The row-block members' FACE (:mod:`opt_core.kernels.fpf_trimul_rows`: the rows-only table + its one selection statement; standard library) —
    imported at the first decision that needs it, never at this module's import."""
    from ...kernels import fpf_trimul_rows as R
    return R


def _rows_kernels():
    """The row-block members' Triton module (:mod:`opt_core.kernels.fpf_trimul_rows.kernels`; torch + triton load here, at the first served call)."""
    from ...kernels.fpf_trimul_rows import kernels as RK
    return RK


def rows_widths(C_z: int, C_h: int) -> str:
    """Which unit's kernels a (C_z, C_h) pair is a candidate for on the row blocks: ``"v4"`` (both in :data:`SUPPORTED_C`: fpf_trimul_v4, today's path),
    ``"rows"`` (both in the rows unit's widths and not both v4's: fpf_trimul_rows, subject to its table), ``"none"`` (declined ``c=<Cz>/<Ch>``)."""
    if int(C_z) in SUPPORTED_C and int(C_h) in SUPPORTED_C:
        return "v4"
    W = _rows_face().WIDTHS
    return "rows" if (int(C_z) in W and int(C_h) in W) else "none"


def resolve_rows(cc: str, C_z: int, C_h: int, table: Optional[Mapping] = None, allow_unmeasured: Optional[bool] = None):
    """The rows unit's decision for (cc, C_z, C_h) as ``(selection, k1a, k1b, k3, b_direct)``: ``selection`` is :func:`fpf_trimul_rows.select`'s answer
    (``served``, ``reason`` in served | served:unmeasured | unmeasured | no_row | v4 | width, ``key``); the cells are None unless served. Pure function of the
    table (default: the unit's own ``table_rows.json``) + :data:`fpf_trimul_rows.ENV_UNMEASURED`: identical on every rank."""
    R = _rows_face()
    t = table if table is not None else R.load_table()
    sel = R.select(t, cc, C_z, C_h, allow_unmeasured=allow_unmeasured)
    if not sel.served:
        return sel, None, None, None, False
    k1a, k1b, k3 = R.tiles(sel.cell)
    return sel, k1a, k1b, k3, R.b_direct(sel.cell)


def _cc(device) -> str:
    import torch
    return "%d.%d" % torch.cuda.get_device_capability(device)


def _cell_key(c: dict) -> str:
    return "%s,%s,%s,%s" % (c.get("BM"), c.get("BN"), c.get("num_warps"), c.get("num_stages"))


def pointer_row(table: Mapping, cc: str, exact_key: Optional[str], exact_cfg: Optional[Mapping]):
    """The cells row the row-block K1 (the POINTER kernel) runs — ``(cfg, key)``. Rule: when the resolved row's ``k1.impl`` is ``tma`` and the
    table carries a served ``<cc>|*`` row whose k1 is a pointer cell, that row is preferred (its tiles were tuned for the pointer kernel); else
    the resolved row as is (a tma row's tile numbers under the pointer kernel, ``k1_impl=pointer(tma-row)``). ``served`` = the table's admission
    rule (``opt_core.kernels.fpf_trimul_v4.table.admitted``, the loader's own). Pure function of the table: identical on every rank."""
    if exact_cfg is None:
        return None, None
    if dict(exact_cfg.get("k1", {})).get("impl", "ptr") != "tma":
        return exact_cfg, exact_key
    key = "%s|*" % cc
    row = table.get(key) if isinstance(table, Mapping) else None
    if isinstance(row, Mapping) and "k1" in row and "k3" in row and dict(row["k1"]).get("impl", "ptr") != "tma":
        from opt_core.kernels.fpf_trimul_v4 import table as _v4table           # the table's ONE admission rule (pure)
        if _v4table.admitted(row):
            return _v4table.row_cfg(row), key
    return exact_cfg, exact_key


def _refusal(reason: str, detail: str = "") -> RowpairRefused:
    """The lever's refusal (case: it cannot run in this process). The message names the lever and the opt-out."""
    return RowpairRefused("%s (%s): cannot run in this process — %s%srun the kit with `--mode off`, or opt this lever out with %s=torch"
                          % (LEVER, KERNEL, reason, (detail + "; ") if detail else "", ENV_KERNELS))


def resolve_cells(net: "SAFE.SafeNet", cc: str, explicit: Optional[Mapping], loader_cfg: Optional[Mapping], loader_key: Optional[str],
                  table: Optional[Mapping], triton_mm: Optional[str] = None):
    """``(cfg, row_key)`` the row blocks are served with on capability ``cc``: the explicit ``cells=`` mapping (``explicit``); else the loader's
    row (:func:`pointer_row` applied); else — no row for this card — the lever's SAFE settings (:data:`SAFE_LEVER` in
    :mod:`opt_core.kernels.safe_settings`; ``net`` engaged: its ONE stderr line, census ``settings=safe:no_cell:rows``, row key ``safe``);
    else (no safe settings for ``cc`` either) the lever's refusal. Pure function of its arguments: identical on every rank."""
    if explicit is not None:
        return explicit, "explicit"
    cfg, key = pointer_row(table or {}, cc, loader_key, loader_cfg)
    if cfg is not None:
        return cfg, key
    if net.on:
        return SAFE.safe_row(SAFE_LEVER, cc)["settings"], "safe"
    try:
        settings = SAFE.safe_settings_of(SAFE_LEVER, cc)()
    except LookupError as e:
        raise _refusal("no cells row and no safe settings for cc %s" % cc, str(e)) from None
    net.engage("no_cell:rows", SAFE.where_word(cc, triton_mm))
    return settings, "safe"


class FusedTriMulFns(TriMulFns):
    """A :class:`TriMulFns` carrying ``proj_into`` / ``tile_epilogue`` / ``operand_dtype`` (module docstring). ``proj`` / ``out`` / ``gate`` are
    the wrapped torch statements (the fallback of a declined unit)."""

    def __init__(self, weights: Mapping[str, Any], stock_fns: TriMulFns, *, eps: float = 1e-5, cells: Optional[Mapping] = None,
                 ledger: Optional[Ledger] = None, stock_round: bool = True, min_tokens: Optional[int] = None):
        import torch
        w = {k: v for k, v in dict(weights).items() if v is not None}
        missing = [k for k in WEIGHT_KEYS if k not in w]
        unknown = [k for k in w if k not in WEIGHT_KEYS and k not in BIAS_KEYS]
        if missing or unknown:
            raise ValueError("fused_trimul_fns: weights_missing=%s weights_unknown=%s (vocabulary: opt_core.trimul_weights.WEIGHT_KEYS / BIAS_KEYS)"
                             % ("+".join(missing) or "none", "+".join(sorted(unknown)) or "none"))
        C_h, C_z = int(w["w_ap"].shape[0]), int(w["w_ap"].shape[1])
        TriMulFns.__init__(self, stock_fns.proj, stock_fns.out, stock_fns.gate, C_h)
        if int(stock_fns.C_h) != C_h:
            raise ValueError("fused_trimul_fns: stock_fns.C_h=%d but w_ap is [%d, %d]" % (int(stock_fns.C_h), C_h, C_z))
        self.C_z = C_z
        self.eps = float(eps)
        self.stock_round = bool(stock_round)
        self.min_tokens = int(DEFAULT_MIN_TOKENS if min_tokens is None else min_tokens)
        self._min_tokens_given = min_tokens is not None                     # a kit's explicit floor wins over the rows table's (cc, C_z, C_h) floor
        self._floors = {}                                                   # str(device) -> the size floor in force there (the rows table's entry or the default)
        self._gate_n = None                                                 # the current call's pair size N (latched by operand_dtype)
        self.ledger = ledger if ledger is not None else LEDGER
        self.ledger.set("min_tokens", self.min_tokens)
        self._raw = w
        self._cells_arg = dict(cells) if cells is not None else None
        self._packs = {}                                # str(device) -> packed weights (+ halves)
        self._decision = {}                             # (device, dtype) -> (k1, k3, pack) served | (None, None, reason) declined-and-memoised (the rows unit's table answers)
        self._rows = {}                                 # str(device) -> {"key", "k1b", "b_direct", "reason"} of the rows unit where IT serves this width (absent: fpf_trimul_v4 serves or nobody)
        self.facts = {}
        self.net = SAFE.SafeNet(SAFE_LEVER, refused=RowpairRefused)      # this provider's safety net (safe settings after a build failure / no row; refusal when they fail too)
        self.rows_net = SAFE.SafeNet(ROWS_SAFE_LEVER, refused=RowpairRefused)   # the rows unit's own net (its SAFE cells come from its table; engaged only where that unit serves)
        self._safe_cfgs = {}                            # cc -> (k1, k3) of the safe settings, resolved for this C_z / C_h
        self._rows_safe = {}                            # cc -> (k1, k3) SAFE cells of the rows unit (LookupError memo: None)

    # ---------------------------------------------------------------------------------------------------------------- decision ladder
    def _floor(self, z) -> int:
        """The size floor (``below_gate``) in force for this z: the kit's explicit ``min_tokens=``; else, on a CUDA device, the rows table's ``min_tokens`` of
        (cc, C_z, C_h) when it names one (fact ``min_tokens_source=rows_table:<key>``); else :data:`DEFAULT_MIN_TOKENS`. Memoised per device; CPU tensors
        (the lever cannot run there at all) read the default. Pure function of the table + device: identical on every rank."""
        if self._min_tokens_given or not z.is_cuda:
            return self.min_tokens
        dkey = str(z.device)
        f = self._floors.get(dkey)
        if f is None:
            R = _rows_face()
            cc = _cc(z.device)
            f = int(R.min_tokens(R.load_table(), cc, self.C_z, self.C_h, DEFAULT_MIN_TOKENS))
            self._floors[dkey] = f
            if f != DEFAULT_MIN_TOKENS:
                self.min_tokens = f
                self.ledger.set("min_tokens", f)
                self.ledger.set("min_tokens_source", "rows_table:%s" % R.cell_key(cc, self.C_z, self.C_h))
                self.facts["min_tokens_source"] = "rows_table:%s" % R.cell_key(cc, self.C_z, self.C_h)
        return f

    def _decide(self, z):
        """``(k1 cfg, k3 cfg, pack)`` when this z block is served; ``(None, None, reason)`` for the documented gates — ``env_torch`` (the
        opt-out), ``below_gate`` (size), ``dtype:<dt>`` / ``c=<Cz>/<Ch>`` (dtype / width classes: a width neither unit serves, or one the rows
        unit's table does not carry MEASURED on this card) — whose units run the torch statements, counted; :class:`RowpairRefused` when the
        lever cannot run in this process at all (no CUDA device, no Triton, the carried kernels not importable, no cells row and no safe
        settings for the card, the safe settings failing to build). Pure function of device / dtype / widths / cells / tables / environment:
        identical on every rank."""
        import torch
        if os.environ.get(ENV_KERNELS, "fpf_v4").strip().lower() == "torch":
            return None, None, "env_torch"
        if self._gate_n is not None and self._gate_n < self._floor(z):
            return None, None, "below_gate"
        if not z.is_cuda:                                                   # the lever cannot run here at all: its refusal, never a silent plain path
            raise _refusal("the pair tensor is on %s, not a CUDA device" % z.device.type)
        if z.dtype not in (torch.bfloat16, torch.float32):
            return None, None, "dtype:%s" % str(z.dtype).replace("torch.", "")
        unit = rows_widths(self.C_z, self.C_h)
        if unit == "none":
            return None, None, "c=%d/%d" % (self.C_z, self.C_h)
        key = (str(z.device), str(z.dtype))
        d = self._decision.get(key)
        if d is None:
            d = self._resolve(z) if unit == "v4" else self._resolve_rows(z)
            self._decision[key] = d
        return d

    def _resolve_rows(self, z):
        """The rows unit's decision for this device (widths beside fpf_trimul_v4's): ``(k1a, k3, pack)`` when its table carries a MEASURED (cc, C_z, C_h) key
        (or a PLACEHOLDER one under :data:`fpf_trimul_rows.ENV_UNMEASURED`), else ``(None, None, "c=<Cz>/<Ch>")`` — the word these widths answered before the
        unit existed (fact ``rows_table=<key or ->:<reason>`` says why). Facts when served: ``unit=fpf_trimul_rows@<ver>`` ``cells=rows:<key>`` ``k1`` ``k1b``
        ``k3`` ``k1_impl=rows:a|a+b`` ``kernel_copy=core``; schedule ``trimul_kernels=fpf_rows``."""
        import torch
        try:
            import triton  # noqa: F401
        except Exception as e:  # noqa: BLE001
            raise _refusal("triton is not importable", repr(e)[:120]) from None
        R = _rows_face()
        cc = "%d.%d" % torch.cuda.get_device_capability(z.device)
        sel, k1a, k1b, k3, bdir = resolve_rows(cc, self.C_z, self.C_h)
        word = "%s:%s" % (sel.key or "-", sel.reason)
        self.facts["rows_table"] = word
        self.ledger.set("rows_table", word)
        if not sel.served:
            _log_once("rows-decline:%s" % str(z.device), "DECLINE on %s: C_z=%d C_h=%d — rows table %s (the torch statements serve, counted c=%d/%d)"
                      % (z.device, self.C_z, self.C_h, word, self.C_z, self.C_h))
            return None, None, "c=%d/%d" % (self.C_z, self.C_h)
        try:
            K, _cells = _kernels()                                          # the weight pack is fpf_trimul_v4's (one vocabulary for both units)
            RK = _rows_kernels()
        except Exception as e:  # noqa: BLE001
            raise _refusal("the fpf_trimul_rows / fpf_trimul_v4 kernels are not importable", "%s: %s" % (type(e).__name__, str(e)[:120])) from None
        try:                                                                # the unit's own width rule (channel chunking) before any launch: a width it cannot walk declines by name
            RK.chunking(self.C_z); RK.chunking(self.C_h)
        except RK.RowsUnsupported as e:
            self.facts["rows_table"] = word + ":" + str(e)
            return None, None, "c=%d/%d" % (self.C_z, self.C_h)
        pack = self._pack(z.device, K)
        self._rows[str(z.device)] = {"key": sel.key, "k1b": k1b, "b_direct": bool(bdir), "reason": sel.reason}
        self.facts.update(unit="%s@%s" % (ROWS_UNIT, getattr(R, "__version__", "?")), cells="rows:%s" % sel.key, k1=_cell_key(k1a), k1b=_cell_key(k1b), k3=_cell_key(k3),
                          k1_impl="rows:a+b" if bdir else "rows:a", kernel_copy="core")
        if sel.reason != "served":
            self.facts["rows_admitted"] = sel.reason                        # served:unmeasured — the engineering switch, named
        if self.rows_net.word():
            self.facts["settings"] = self.rows_net.word()
        for k, v in self.facts.items():
            self.ledger.set(k, v)
        self.ledger.impl = "%s@%s" % (ROWS_UNIT, getattr(R, "__version__", "?"))
        record_schedule(trimul_kernels="fpf_rows", trimul_cells_k1=_cell_key(k1a), trimul_cells_k1b=_cell_key(k1b), trimul_cells_k3=_cell_key(k3), trimul_b_direct=bool(bdir))
        _log_once("serve:%s" % str(z.device), "SERVE on %s: unit=%s key=%s (%s) k1=%s k1b=%s k3=%s b_direct=%s C_z=%d C_h=%d z=%s planes=bfloat16 residual_round=%s"
                  % (z.device, ROWS_UNIT, sel.key, sel.reason, _cell_key(k1a), _cell_key(k1b), _cell_key(k3), bool(bdir), self.C_z, self.C_h,
                     str(z.dtype).replace("torch.", ""), "stock" if self.stock_round else "fused"))
        return k1a, k3, pack

    def _resolve(self, z):
        import torch
        try:
            import triton  # noqa: F401
        except Exception as e:  # noqa: BLE001
            raise _refusal("triton is not importable", repr(e)[:120]) from None
        try:
            K, CELLS = _kernels()
        except Exception as e:  # noqa: BLE001
            raise _refusal("the carried fpf_trimul_v4 kernels are not importable", "%s: %s" % (type(e).__name__, str(e)[:120])) from None
        cc = "%d.%d" % torch.cuda.get_device_capability(z.device)
        loader_cfg, loader_key, table = None, None, {}
        if self._cells_arg is None:                                          # the package's one table through the kernels' own loader, exactly as the single-GPU provider
            loader_cfg = CELLS.cell_for(z.device)
            loader_key = (getattr(CELLS, "INFO", {}).get(str(z.device)) or {}).get("key") or "?"
            from opt_core.kernels.fpf_trimul_v4 import table as _v4table
            table = _v4table.load_table(getattr(CELLS, "TABLE_PATH", None))
        cfg, row_key = resolve_cells(self.net, cc, self._cells_arg, loader_cfg, loader_key, table, triton_mm=SAFE.triton_mm())
        pack = self._pack(z.device, K)
        k1, k3 = self._tiles(K, cfg, pack)
        k1_impl = "pointer" if dict(cfg.get("k1", {})).get("impl", "ptr") != "tma" else "pointer(tma-row)"   # rows are served by the pointer-load K1 with the row's tile numbers
        self.facts.update(cells=row_key, k1=_cell_key(k1), k3=_cell_key(k3), k1_impl=k1_impl, kernel_copy="core")
        if self.net.word():
            self.facts["settings"] = self.net.word()
        for k, v in self.facts.items():
            self.ledger.set(k, v)
        record_schedule(trimul_kernels="fpf_v4_rows", trimul_cells_k1=_cell_key(k1), trimul_cells_k3=_cell_key(k3))
        _log_once("serve:%s" % str(z.device), "SERVE on %s: k1=%s (%s) k3=%s C_z=%d C_h=%d z=%s planes=bfloat16 residual_round=%s"
                  % (z.device, _cell_key(k1), k1_impl, _cell_key(k3), self.C_z, self.C_h, str(z.dtype).replace("torch.", ""), "stock" if self.stock_round else "fused"))
        return k1, k3, pack

    def _pack(self, device, K):
        import torch
        key = str(device)
        p = self._packs.get(key)
        if p is not None:
            return p
        raw = {k: (v.detach().to(device) if hasattr(v, "detach") else v) for k, v in self._raw.items()}
        w = K.pack_generic(cdt=torch.bfloat16, **raw)
        D = int(w["D"])
        hb = bool(w["has_bias"])
        halves = {}
        for name, sl in (("a", slice(0, D)), ("b", slice(D, 2 * D))):
            h = {"wgT": w["wgT_in"][:, sl].contiguous(), "wpT": w["wpT_in"][:, sl].contiguous()}
            if hb:
                h["bg"], h["bp"] = w["bg_in"][sl].contiguous(), w["bp_in"][sl].contiguous()
            halves[name] = h
        w["halves"] = halves
        self._packs[key] = w
        return w

    def _tiles(self, K, cfg, pack):
        """``(k1, k3)`` launch settings of a cells row / the safe settings for this C_z, C_h (the row's ``overrides`` applied; ``impl`` dropped)."""
        k1, k3 = K.resolve_cfg(cfg, self.C_z, self.C_h, bool(pack["has_bias"]))
        k1 = dict(k1); k3 = dict(k3)
        for c in (k1, k3):
            c.pop("impl", None)
        return k1, k3

    def _safe_tiles(self, cc: str, K, pack):
        """The lever's SAFE ``(k1, k3)`` for capability ``cc`` (:mod:`opt_core.kernels.safe_settings`); LookupError when it has none there."""
        if cc not in self._safe_cfgs:
            self._safe_cfgs[cc] = self._tiles(K, SAFE.safe_settings_of(SAFE_LEVER, cc)(), pack)
        return self._safe_cfgs[cc]

    def _launch(self, kind: str, launch, tuned: Mapping, cc: str, K, pack) -> None:
        """One K1 / K3 launch under the safety net: ``launch(tuned settings)``; a triton BUILD failure switches this process to the lever's
        safe settings for ``cc`` (ONE stderr line, census ``settings=safe:build_failed:<Type>``) and the launch is redone with them; the safe
        settings failing to build too (or absent for ``cc``) raises the lever's refusal (:class:`RowpairRefused`). Any other exception propagates."""
        idx = 0 if kind == "k1" else 1
        was_on = self.net.on

        def safe():
            try:
                return self._safe_tiles(cc, K, pack)[idx]
            except LookupError as e:
                raise _refusal("the tuned settings failed to build and there are no safe settings for cc %s" % cc, str(e)) from None

        self.net.run(launch, tuned, safe, where=SAFE.where_word(cc, SAFE.triton_mm()))
        if self.net.on and not was_on:                                      # the safe settings serve from here on: say so in the census
            k1s, k3s = self._safe_tiles(cc, K, pack)
            self.facts.update(settings=self.net.word(), k1=_cell_key(k1s), k3=_cell_key(k3s), cells="safe")
            for k, v in self.facts.items():
                self.ledger.set(k, v)
            record_schedule(trimul_cells_k1=_cell_key(k1s), trimul_cells_k3=_cell_key(k3s), trimul_settings=self.net.word())

    def _rows_safe_tiles(self, cc: str):
        """The rows unit's SAFE ``(k1, k3)`` cells for capability ``cc`` (its table's ``safe`` member); LookupError when it has none there."""
        if cc not in self._rows_safe:
            R = _rows_face()
            row = R.safe_cells(R.load_table(), cc)
            self._rows_safe[cc] = None if row is None else tuple({k: int(row[n][k]) for k in R.CELL_KEYS} for n in ("k1", "k3"))
        if self._rows_safe[cc] is None:
            raise LookupError("%s: no safe cells for cc %s in %s" % (ROWS_SAFE_LEVER, cc, ROWS_UNIT))
        return self._rows_safe[cc]

    def _launch_rows(self, kind: str, launch, tuned: Mapping, cc: str) -> None:
        """One rows-unit K1 / K3 launch under ITS safety net (:attr:`rows_net`): as :meth:`_launch`, with the unit's own SAFE cells (its table's ``safe``
        member) after a triton BUILD failure of the tuned cell (ONE stderr line ``[opt_core/pair_fused:trimul_rows] safe settings served (build_failed:…)``,
        census ``settings=safe:build_failed:<Type>``); those failing too / absent for ``cc`` is the lever's refusal. Any other exception propagates."""
        idx = 0 if kind == "k1" else 1
        was_on = self.rows_net.on

        def safe():
            try:
                return self._rows_safe_tiles(cc)[idx]
            except LookupError as e:
                raise _refusal("the tuned %s cells failed to build and there are no safe cells for cc %s" % (ROWS_UNIT, cc), str(e)) from None

        self.rows_net.run(launch, tuned, safe, where=SAFE.where_word(cc, SAFE.triton_mm()))
        if self.rows_net.on and not was_on:
            k1s, k3s = self._rows_safe_tiles(cc)
            self.facts.update(settings=self.rows_net.word(), k1=_cell_key(k1s), k1b=_cell_key(k1s), k3=_cell_key(k3s), cells="rows:safe")
            for k, v in self.facts.items():
                self.ledger.set(k, v)
            record_schedule(trimul_cells_k1=_cell_key(k1s), trimul_cells_k1b=_cell_key(k1s), trimul_cells_k3=_cell_key(k3s), trimul_settings=self.rows_net.word())

    def _rows_cells(self, z, kind: str):
        """The tuned cell the rows unit launches ``kind`` (``k1`` a-layout | ``k1b`` | ``k3``) with on z's device NOW: the safe cell once its net serves
        (process-wide after a build failure), else the table's."""
        cc = _cc(z.device)
        if self.rows_net.wide:
            k1s, k3s = self._rows_safe_tiles(cc)
            return k3s if kind == "k3" else k1s
        if kind == "k1b":
            return self._rows[str(z.device)]["k1b"]
        k1, k3, _pack = self._decision[(str(z.device), str(z.dtype))]
        return k1 if kind == "k1" else k3

    def _decline(self, reason: str) -> bool:
        self.ledger.fallback(reason)
        if "trimul_kernels" not in _sched():
            record_schedule(trimul_kernels="torch")
        record_schedule(trimul_kernels_fallback=reason)
        return False

    # ---------------------------------------------------------------------------------------------------------------- the hooks
    def operand_dtype(self, z_shard):
        """bf16 (the planes / contraction operands the kernels serve) when this shard is served, else None (the statements' own dtype).
        The shard's pair size ``N = z_shard.shape[-2]`` is latched here for the size gate (the driver asks this first, once per call)."""
        import torch
        self._gate_n = int(z_shard.shape[-2])
        k1, _k3, _why = self._decide(z_shard)
        return torch.bfloat16 if k1 is not None else None

    def _served_by_rows(self, z) -> bool:
        """Is this (served) device's width the rows unit's (fpf_trimul_rows) rather than fpf_trimul_v4's?"""
        return str(z.device) in self._rows

    def b_transposed(self, z_shard) -> bool:
        """The driver's optional question before it allocates a b sub-block plane: True = hand ``proj_into`` the ``[C_h, w, N]`` VIEW of a contiguous
        ``[C_h, N, w]`` buffer (the GEMM-B layout) — the rows unit's K1 writes it directly and ``_b_slab``'s transpose copy disappears; False (fpf_trimul_v4
        widths, every declined unit) = today's ``[C_h, w, N]`` plane, byte for byte."""
        k1, _k3, _why = self._decide(z_shard)
        if k1 is None or not self._served_by_rows(z_shard):
            return False
        return bool(self._rows[str(z_shard.device)]["b_direct"])

    def proj_into(self, dst, z_block, mask_block, is_a: bool) -> bool:
        """K1 into ``dst [C_h, rows, N]`` (``dst.stride() == (plane, N, 1)``, bf16 — or, rows unit + :meth:`b_transposed`, the ``[C_h, rows, N]`` view of a
        contiguous ``[C_h, N, rows]`` buffer) from ``z_block [rows, N, C_z]`` and ``mask_block [rows, N]``."""
        import torch
        k1, _k3, pack = self._decide(z_block)
        if k1 is None:
            return self._decline(pack)
        rows, N, C = int(z_block.shape[0]), int(z_block.shape[1]), int(z_block.shape[2])
        if self._served_by_rows(z_block):
            return self._proj_into_rows(dst, z_block, mask_block, is_a, pack, rows, N, C)
        if C != self.C_z or tuple(dst.shape) != (self.C_h, rows, N) or dst.stride(2) != 1 or dst.stride(1) != N or dst.dtype != torch.bfloat16:
            return self._decline("layout")
        if rows == 0:
            return True
        import triton
        K, _ = _kernels()
        zc = z_block if z_block.is_contiguous() else z_block.contiguous()
        hm = mask_block is not None
        m = (mask_block.to(torch.float32).contiguous() if hm else zc)
        h = pack["halves"]["a" if is_a else "b"]
        hb = bool(pack["has_bias"])
        bg = h["bg"] if hb else pack["ln_in_w"]
        bp = h["bp"] if hb else pack["ln_in_w"]

        def launch(c):
            grid = (triton.cdiv(N, int(c["BM"])), rows, 1)
            K._k1c[grid](zc, pack["ln_in_w"], pack["ln_in_b"], h["wgT"], h["wpT"], bg, bp, m, dst,
                         N, N, dst.stride(0), self.eps, rows * N, 0, 0,
                         C=C, D2=self.C_h, BM=int(c["BM"]), BN=min(int(c["BN"]), self.C_h), HAS_MASK=hm, HAS_BIAS=hb,
                         IN_F32=(zc.dtype == torch.float32), num_warps=int(c["num_warps"]), num_stages=int(c["num_stages"]))

        self._launch("k1", launch, k1, _cc(z_block.device), K, pack)
        _COUNTS["k1"] += 1
        self.ledger.serve("k1")
        record_schedule(trimul_k1_launches=_COUNTS["k1"])
        return True

    def _proj_into_rows(self, dst, z_block, mask_block, is_a: bool, pack, rows: int, N: int, C: int) -> bool:
        """K1 of the rows unit (``fpf_trimul_rows.kernels.launch_k1``): z read channel-last IN PLACE (no ``.contiguous()``), the mask in its own float dtype
        (no cast copy; a bool / integer mask is cast once), ``dst`` in a- or b-layout (read off its strides); a layout the kernels do not address declines
        the unit by name (``layout``) before any launch."""
        import torch
        if C != self.C_z:
            return self._decline("layout")
        RK = _rows_kernels()
        try:
            layout = RK.k1_layout(dst, rows, N, self.C_h)
        except RK.RowsUnsupported:
            return self._decline("layout")
        if z_block.dtype not in RK.IO_DTYPES or z_block.stride(2) != 1:
            return self._decline("layout")
        if rows == 0 or N == 0:
            return True
        m = mask_block
        if m is not None and (m.dtype not in RK.MASK_DTYPES or tuple(m.shape) != (rows, N)):
            m = m.to(torch.float32).reshape(rows, N)
        h = pack["halves"]["a" if is_a else "b"]
        hb = bool(pack["has_bias"])
        kind = "k1" if layout == RK.LAYOUT_A else "k1b"
        cell = self._rows_cells(z_block, kind)

        def launch(c):
            RK.launch_k1(z_block, m, dst, ln_w=pack["ln_in_w"], ln_b=pack["ln_in_b"], wgT=h["wgT"], wpT=h["wpT"], bg=h.get("bg"), bp=h.get("bp"),
                         has_bias=hb, cell=c, eps=self.eps)

        self._launch_rows("k1", launch, cell, _cc(z_block.device))
        _COUNTS["k1"] += 1
        self.ledger.serve(kind)
        record_schedule(trimul_k1_launches=_COUNTS["k1"])
        return True

    def _tile_epilogue_rows(self, T, z_block, add: bool, pack, rows: int, w: int, C: int) -> bool:
        """K3 of the rows unit (``fpf_trimul_rows.kernels.launch_k3``) on the column window IN PLACE: no staging copy in, no copy out."""
        import torch
        RK = _rows_kernels()
        if (C != self.C_z or tuple(T.shape) != (self.C_h, rows, w) or T.dtype != torch.bfloat16 or (w > 1 and T.stride(2) != 1)
                or z_block.dtype not in RK.IO_DTYPES or z_block.stride(2) != 1 or (w > 1 and z_block.stride(1) != C)):
            return self._decline("layout")
        if rows == 0 or w == 0:
            return True
        hb = bool(pack["has_bias"])
        cell = self._rows_cells(z_block, "k3")

        def launch(c):                                                      # (a cell BUILD failure raises before the kernel runs: the window is untouched for the retry)
            RK.launch_k3(T, z_block, z_block, ln_out_w=pack["ln_out_w"], ln_out_b=pack["ln_out_b"], ln_in_w=pack["ln_in_w"], ln_in_b=pack["ln_in_b"],
                         wz=pack["wz"], wg_out=pack["wg_out"], bo=pack.get("b_o"), bg=pack.get("b_og"), has_bias=hb, cell=c, residual=bool(add),
                         stock_round=self.stock_round, eps=self.eps)

        self._launch_rows("k3", launch, cell, _cc(z_block.device))
        _COUNTS["k3"] += 1
        self.ledger.serve("k3")
        record_schedule(trimul_k3_launches=_COUNTS["k3"])
        return True

    def tile_epilogue(self, T, z_block, add: bool) -> bool:
        """K3 on the tile: ``z_block [rows, w, C_z] (+)= gate(z_block) * out(T[C_h, rows, w])`` in place (fpf_trimul_v4 widths: column window staged
        contiguous; rows unit: addressed in place)."""
        import torch
        _k1, k3, pack = self._decide(z_block)
        if k3 is None:
            return self._decline(pack)
        rows, w, C = int(z_block.shape[0]), int(z_block.shape[1]), int(z_block.shape[2])
        if self._served_by_rows(z_block):
            return self._tile_epilogue_rows(T, z_block, add, pack, rows, w, C)
        if C != self.C_z or tuple(T.shape) != (self.C_h, rows, w) or T.stride(2) != 1 or T.dtype != torch.bfloat16 or z_block.stride(2) != 1:
            return self._decline("layout")
        if rows == 0 or w == 0:
            return True
        import triton
        K, _ = _kernels()
        zs = z_block if z_block.is_contiguous() else z_block.contiguous()          # the window as a pair tensor whose row stride is its width
        hb = bool(pack["has_bias"])
        bo = pack["b_o"] if hb else pack["ln_in_w"]
        bg = pack["b_og"] if hb else pack["ln_in_w"]

        def launch(c):                                                      # (a settings BUILD failure raises before the kernel runs: the block is untouched for the retry)
            grid = (triton.cdiv(w, int(c["BM"])), rows, 1)
            K._k3c[grid](T, zs, pack["ln_out_w"], pack["ln_out_b"], pack["ln_in_w"], pack["ln_in_b"], pack["wz"], pack["wg_out"], bo, bg, zs,
                         w, T.stride(1), T.stride(0), self.eps, rows * w,
                         C=C, CH=self.C_h, BM=int(c["BM"]), BN=min(int(c["BN"]), C), RESIDUAL=bool(add), STOCK_ROUND=self.stock_round, HAS_BIAS=hb,
                         IN_F32=(zs.dtype == torch.float32), num_warps=int(c["num_warps"]), num_stages=int(c["num_stages"]))

        self._launch("k3", launch, k3, _cc(z_block.device), K, pack)
        if zs.data_ptr() != z_block.data_ptr():
            z_block.copy_(zs)
        _COUNTS["k3"] += 1
        self.ledger.serve("k3")
        record_schedule(trimul_k3_launches=_COUNTS["k3"])
        return True


def _sched() -> dict:
    from .evidence import schedule
    return schedule()


def fused_trimul_fns(weights: Mapping[str, Any], stock_fns: TriMulFns, *, eps: float = 1e-5, cells: Optional[Mapping] = None,
                     ledger: Optional[Ledger] = None, stock_round: bool = True, min_tokens: Optional[int] = None) -> FusedTriMulFns:
    """The fused provider for :func:`opt_core.mem.rowpair.trimul.trimul_update_`. ``weights``: the module's tensors in the
    :data:`opt_core.trimul_weights.WEIGHT_KEYS` (+ ``BIAS_KEYS``) vocabulary; ``stock_fns``: the module's torch :class:`TriMulFns` (the fallback of every
    declined unit and the source of ``C_h``); ``eps``: both LayerNorms' epsilon; ``cells``: an explicit ``{"k1": {...}, "k3": {...}[, "overrides"]}``
    row (default: :func:`fpf_trimul_v4.cells.cell_for`); ``stock_round``: round the gated output to bf16 before the residual add (an engine
    whose stock adds the residual to a bf16 kernel output) or add in fp32 first (``False``); ``min_tokens``: the pair size N below which the
    call runs the torch statements (reason ``below_gate``; default :data:`DEFAULT_MIN_TOKENS`)."""
    return FusedTriMulFns(weights, stock_fns, eps=eps, cells=cells, ledger=ledger, stock_round=stock_round, min_tokens=min_tokens)


def emit_line(tag: str, ledger: Optional[Ledger] = None, **evidence) -> str:
    """Print this lever's ONE activation-evidence line (``[<tag>] LEVER name=F2.trimul_rows …``; :meth:`opt_core.counters.Ledger.line`)."""
    from ...report import emit
    L = ledger if ledger is not None else LEDGER
    if L.impl in (None, KERNEL):
        ver = getattr(UNIT, "__version__", None)
        L.impl = "%s@%s" % (KERNEL, ver) if ver else KERNEL
    return emit(L.line(tag, **evidence))


def describe(ledger: Optional[Ledger] = None) -> dict:
    """Plain data for a manifest: the ledger's fields + the launch counts (+ ``rows_unit``: the rows unit's version and table keys, when importable)."""
    L = ledger if ledger is not None else LEDGER
    d = dict(L.fields())
    d.update(k1_launches=_COUNTS["k1"], k3_launches=_COUNTS["k3"])
    try:
        d["rows_unit"] = _rows_face().describe()
    except Exception as e:  # noqa: BLE001 — a manifest never fails on the optional unit
        d["rows_unit"] = {"error": "%s: %s" % (type(e).__name__, str(e)[:120])}
    return d
