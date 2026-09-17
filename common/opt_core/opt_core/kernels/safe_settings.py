"""opt_core.kernels.safe_settings — the ONE resolution-and-safety-net mechanism of the compute-capability-keyed kernel levers
(flash_triattn / shared_bias_attn / pair_fused cells …).

A lever keeps its launch settings (tiles, warps, stages, cells) in a table keyed ``"<cc>|<triton major.minor>"`` — a NAMED EXCEPTION
for one known environment — and ``"<cc>|*"`` — the capability's default (its tuned settings). Resolution and the safety net, for
every such lever, are:

  (i)   the exact ``<cc>|<mm>`` row, else the ``<cc>|*`` row: :func:`resolve_key` / :func:`resolve_row` / :func:`resolve_cell`
        (a cell = one shape class inside a row; the exact row's cell wins, else the default row's cell); for the device of a call:
        :func:`device_cc` / :func:`row_for_device` / :func:`key_for_device` (per-device memo), :func:`pick_by_block` for a row's
        ``(max block, value)`` ladders; no row / no cell = the
        lever's own capability-free default — its TUNED settings serve everyone (a capability no row names gets them with ONE info line,
        :meth:`SafeNet.inform`: ``[opt_core/<lever>] default settings (no row for cc <cc>, …); tuned rows exist for cc …``, fact
        ``cells_note=default:no_row``; H100: every lever's tables carry its rows — nothing below ever engages there);
  (ii)  the picked settings cannot serve — NO cell for this shape / capability (:meth:`SafeNet.no_cell`) or the settings fail to
        BUILD (a triton compile-time failure caught at the first launch: :meth:`SafeNet.run`, classified by :func:`is_build_failure`;
        never an out-of-memory, never anything a running kernel produces) — → the lever's SAFE settings (a conservative configuration
        valid on any card of the generation: single-stage cells, the smallest tile …, supplied by the lever), ONE line on stderr per
        engagement::

            [opt_core/<lever>] safe settings served (<reason>, cc <cc>, triton <mm>)      reason = no_cell:<shape> | build_failed:<exception class>

        and the census word ``settings=safe:<reason>`` (:meth:`SafeNet.word`) for the lever's activation line — exit 0, the lever
        counts as engaged.  SCOPE: a ``no_cell:<shape>`` engagement holds for THAT cell only — the lever's pinned cells keep their own
        settings in the same process (:meth:`SafeNet.serves_safe`, :attr:`SafeNet.cells`); a ``build_failed`` engagement holds for the
        rest of the process (:attr:`SafeNet.wide`: the stack cannot build the tuned settings, every cell of the lever runs the safe ones);
  (iii) the safe settings cannot build / run either → the lever's own refusal exception (``refused=``), whose message names the
        lever and the one-flag escape (``--mode off`` / the kit's explicit opt-out for the lever); the kit turns it into its hard
        error. This is the only fail-closed case.

Python floor: standard library at import (triton / torch are touched only inside :func:`is_build_failure`, lazily).
"""
from __future__ import annotations

import sys
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union

CcLike = Union[None, str, Tuple[int, int]]


# ------------------------------------------------------------------------------------------------------------------ resolution
def cc_word(cc: CcLike) -> Optional[str]:
    """``"M.m"`` for a ``(major, minor)`` tuple or an ``"M.m"`` string; None for None."""
    if cc is None:
        return None
    if isinstance(cc, str):
        return cc
    return "%d.%d" % (int(cc[0]), int(cc[1]))


def where_word(cc: CcLike, triton_mm: Optional[str]) -> str:
    """``"cc <cc>, triton <mm>"`` — the parenthesised tail of the ONE line (``?`` for an unknown part)."""
    return "cc %s, triton %s" % (cc_word(cc) or "?", triton_mm or "?")


def resolve_key(table: Mapping[str, Any], cc: CcLike, triton_mm: Optional[str]) -> Optional[str]:
    """The key of ``table`` serving (cc, triton major.minor): ``"<cc>|<mm>"`` when that named-exception row exists, else ``"<cc>|*"``
    when the capability has a default row, else None."""
    w = cc_word(cc)
    if w is None:
        return None
    for key in (w + "|" + (triton_mm or ""), w + "|*"):
        if key in table:
            return key
    return None


def resolve_row(table: Mapping[str, Any], cc: CcLike, triton_mm: Optional[str], default: Any = None) -> Any:
    """The row of :func:`resolve_key` (``default`` when no key serves)."""
    key = resolve_key(table, cc, triton_mm)
    return table[key] if key is not None else default


def resolve_cell(table: Mapping[str, Mapping], cc: CcLike, triton_mm: Optional[str], cell) -> Tuple[Optional[str], Any]:
    """``(key, settings)`` of ONE cell: the exact row's cell when that row carries it, else the default row's cell, else
    ``(None, None)`` (a named-exception row need not repeat every cell of the default row)."""
    w = cc_word(cc)
    if w is None:
        return None, None
    for key in (w + "|" + (triton_mm or ""), w + "|*"):
        row = table.get(key)
        if row is not None and cell in row:
            return key, row[cell]
    return None, None


def table_ccs(table: Mapping[str, Any]) -> list:
    """The compute capabilities a "<cc>|…"-keyed (or "<cc>"-keyed) table has rows for, sorted ("8.0", "9.0", …)."""
    return sorted({str(k).split("|")[0] for k in table if str(k) and str(k)[0].isdigit()}, key=lambda w: tuple(int(x) for x in w.split(".") if x.isdigit()))


def triton_mm() -> str:
    """major.minor of the importable triton (``""`` when triton is absent or carries no version)."""
    try:
        import triton
    except ImportError:
        return ""
    return ".".join(str(getattr(triton, "__version__", "")).split("+")[0].split(".")[:2])


def device_cc(device: Any = None) -> Optional[Tuple[int, int]]:
    """(major, minor) of a CUDA device — a ``torch.device``, a device index, a ``"cuda:N"`` string, or None (the current device); None when torch or
    CUDA is absent or the device is not a CUDA device (the capability-free rows of a table serve)."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    if device is None:
        return tuple(torch.cuda.get_device_capability())
    dev = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
    if dev.type != "cuda":
        return None
    return tuple(torch.cuda.get_device_capability(dev))


def _device_slot(device: Any = None) -> tuple:
    """The memo key of a device for :func:`row_for_device`: ``("cuda", index)`` (None / an index-less device = the current device), else ``(type,)``."""
    try:
        import torch
    except ImportError:
        return ("none",)
    if not torch.cuda.is_available():
        return ("none",)
    if device is None:
        return ("cuda", torch.cuda.current_device())
    dev = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
    if dev.type != "cuda":
        return (dev.type,)
    return ("cuda", dev.index if dev.index is not None else torch.cuda.current_device())


def row_for_device(table: Mapping[str, Any], device: Any = None, default: Any = None, cache: Optional[Dict[tuple, Any]] = None) -> Any:
    """The row of a ``"<cc>|…"``-keyed ``table`` serving ``device``'s compute capability under this process's triton (:func:`resolve_row` of
    :func:`device_cc` and :func:`triton_mm`), else ``default`` (no CUDA, a non-CUDA device, a capability without a row). ``cache``: a dict the
    calling module owns, memoising the answer per device slot (the table and triton do not change within a process)."""
    if cache is None:
        return resolve_row(table, device_cc(device), triton_mm(), default=default)
    slot = _device_slot(device)
    if slot not in cache:
        cache[slot] = resolve_row(table, device_cc(device), triton_mm(), default=default)
    return cache[slot]


def key_for_device(table: Mapping[str, Any], device: Any = None) -> Optional[str]:
    """The ``table`` key serving ``device`` (:func:`resolve_key` of :func:`device_cc` and :func:`triton_mm`); None when no row serves it."""
    return resolve_key(table, device_cc(device), triton_mm())


def pick_by_block(pairs, block: int):
    """``pairs`` = ``((max block, value), …)`` ascending: the value of the first pair whose max block >= ``block`` (past the last pair: its value)."""
    for max_block, value in pairs:
        if block <= max_block:
            return value
    return pairs[-1][1]


# ------------------------------------------------------------------------------------------------------------------- safe rows
# The levers' SAFE settings by compute capability ("<cc>" exact, "*" any capability): a conservative configuration VALID (compiles, fits,
# numerics in class) at every shape swept on that capability — slow at wide shapes by construction; it serves only what no tuned row
# serves (case ii above), and the ONE line names it. flash_triattn keeps its safe set beside its tables (kernels/flash_triattn.py
# _SAFE_SINGLE_STAGE: single-stage cells per dtype class and head dim); the pair-stack cells are here (safe_row()).
# A row: {settings, status, evidence[, when][, alternatives][, refusal]} — `when` bounds the shapes the settings serve (`<dim>_max` / `<dim>_min` on
# the lever's dimension names), `alternatives` are further bounded rows tried in order, `refusal` is the word a lever appends to its refusal when it
# has safe settings on the capability but none admits the shape (safe_settings_for); a row without `when` serves every shape (safe_row).
SAFE_ROWS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "pair_fused:transition": {                                   # fpf transition (attn/pair_fused): (BM, BH, num_warps, num_stages, IL)
        "8.0": dict(settings={"BM": 16, "BH": 32, "num_warps": 4, "num_stages": 1, "IL": 0}, status="SAFE",
                    evidence="A100 (cc 8.0), torch 2.10 / triton 3.6: the most conservative of the 14 configurations valid (compiles, fits, fp64 ratio 1.000) at EVERY swept (C, NH) "
                             "128x256 128x512 256x512 256x1024 384x1536; smem 20 KB @C128 .. 82 KB @C384; slow at wide shapes (79 ms vs torch 32 ms @384x1536)"),
        "9.0": dict(settings={"BM": 16, "BH": 32, "num_warps": 4, "num_stages": 1, "IL": 0}, status="SAFE:inferred",
                    evidence="the same cell as one cross-card default (fits both cards); NOT run on H100 (h100-cal-1 covers the tuned rows) — H100's shapes always hit tuned rows, so it never serves there"),
        "10.0": dict(settings={"BM": 16, "BH": 32, "num_warps": 4, "num_stages": 1, "IL": 0}, status="SAFE",
                     evidence="B200 (cc 10.0), torch 2.10 / triton 3.6 (ptxas-blackwell 12.9): the same conservative cell, measured in the 72-cfg sweep at (128, 512) — compiles, smem 20 KB, fp64 ratio 1.000 "
                              "(bitwise == the torch statements under ln=stock), 13.9 ms vs torch 14.2 ms @N 2048 (the tuned row 3.4 ms)"),
        "10.3": dict(settings={"BM": 16, "BH": 32, "num_warps": 4, "num_stages": 1, "IL": 0}, status="SAFE",
                     evidence="B300 (cc 10.3), the same stack and sweep at (128, 512): compiles, smem 20 KB, fp64 ratio 1.000, 13.5 ms vs torch 13.7 ms @N 2048"),
        "*": dict(settings={"BM": 16, "BH": 32, "num_warps": 4, "num_stages": 1, "IL": 0}, status="SAFE:inferred",
                  evidence="every other capability (sm_86 / sm_89 / sm_120 ...): the one cross-card cell of the four certified capabilities — single stage, shared memory 20 KB @C128 .. 82 KB "
                           "@C384, under the 99 KB of the consumer parts; not measured there (a build failure names itself and the lever refuses by name)"),
    },
    "pair_fused:trimul": {                                       # fpf_trimul_v4 K1 / K3 tiles (c_z in {128, 256}, D in {128, 256})
        "8.0": dict(settings={"k1": {"BM": 128, "BN": 64, "num_warps": 8, "num_stages": 2}, "k3": {"BM": 64, "BN": 64, "num_warps": 4, "num_stages": 1}}, status="SAFE",
                    evidence="A100: the cc 8.0 cell — valid at all four (C, D): 92/92 battery on triton 3.7.1 (a100_trimul_v4_evidence.md), 24/24 on triton 3.6.0 (tcell-36-B-1)"),
        "9.0": dict(settings={"k1": {"BM": 128, "BN": 128, "num_warps": 8, "num_stages": 1}, "k3": {"BM": 64, "BN": 64, "num_warps": 4, "num_stages": 1}}, status="SAFE",
                    evidence="the cells table's `9.0|*` pointer-load cell (not the sm_90-only `9.0|3.6` TMA k1); runs on H100 and, measured, on A100"),
        "10.0": dict(settings={"k1": {"BM": 128, "BN": 64, "num_warps": 4, "num_stages": 1}, "k3": {"BM": 64, "BN": 64, "num_warps": 8, "num_stages": 1}}, status="SAFE",
                     evidence="the cells tables' `10.0|*` = `10.3|*` single-stage pointer-load cell: B200 (cc 10.0) torch 2.10 / triton 3.6, valid at all four (C, D) — 24/24 battery "
                              "(tests/test_generic.py B,C: TIER2_SAME_CLASS, r2r bitwise, graph replay); it is also the tuned cell there, so a build failure of it nets to the same refusal"),
        "10.3": dict(settings={"k1": {"BM": 128, "BN": 64, "num_warps": 4, "num_stages": 1}, "k3": {"BM": 64, "BN": 64, "num_warps": 8, "num_stages": 1}}, status="SAFE",
                     evidence="B300 (cc 10.3), same stack: the same cell, 24/24 battery at all four (C, D)"),
        "*": dict(settings={"k1": {"BM": 64, "BN": 32, "num_warps": 4, "num_stages": 1}, "k3": {"BM": 64, "BN": 64, "num_warps": 4, "num_stages": 1}}, status="SAFE:inferred",
                  evidence="every other capability (sm_86 / sm_89 / sm_120 ...): the smallest pointer-K1 tile a certified row uses (BM 64 x BN 32, 4 warps: the H100 (128,128) cell of the "
                           "cells table's 9.0 pointer rows) at ONE stage — z tile 32 KB + gate|proj weight tiles 32 KB at c_z 256, under the 99 KB shared memory of the consumer parts where "
                           "the BM 128 K1 does not fit; K3 64x64 w4 s1 = the certified K3 of cc 8.0 / 9.0; not measured on those cards: fpf_trimul_v4's warm numerics probe holds it to the "
                           "certified class before it serves, and a build failure names itself and the lever refuses by name"),
    },
    "pair_fused:prologue": {                                     # fpf tri-attention prologue, kernel variant v3 (attn/pair_fused): (BI, BJ, num_warps, num_stages); shape dims c_z, H, D
        "8.0": dict(settings={"BI": 8, "BJ": 8, "num_warps": 4, "num_stages": 2}, status="SAFE",
                    evidence="A100 (cc 8.0): valid (the equality class of the 9.0 rows: fp64-error ratio 1.08-1.10, q/k/v/bias bit-exact) at every swept shape — "
                             "(128,4,32) and (64,4,16) full grids, (256,8,32) (256,4,64) (128,8,32) (64,2,32) reduced grids — never on a cliff, faster than the "
                             "torch statements everywhere (x1.1 at c_z 256); NOT safe on sm_80: BI 8/BJ 16/w 4, BI 8/BJ 8/w 8 at c_z 256, BI 16/BJ 32/w 4 (run-to-run determinism broken)"),
        "9.0": dict(settings={"BI": 8, "BJ": 8, "num_warps": 4, "num_stages": 2}, status="SAFE:inferred beyond (128,4,32)",
                    evidence="H100: measured valid at (128,4,32), other shapes inferred; H100's shapes hit tuned rows, so it does not serve there"),
        "10.0": dict(settings={"BI": 8, "BJ": 8, "num_warps": 4, "num_stages": 2}, status="SAFE:inferred beyond (128,4,32) and (64,4,16)",
                     evidence="B200 (cc 10.0), torch 2.10 / triton 3.6: measured valid at (128,4,32) and (64,4,16) in the 54-cfg sweeps (the 9.0 rows' equality class: block fp64 ratio 1.100 / 1.086, "
                              "q/k/v/bias bit-exact under ln=stock), faster than the torch statements; other shapes inferred; NOT buildable on sm_100/sm_103: BI 16 x BJ 32 (tensor memory 768 > 512)"),
        "10.3": dict(settings={"BI": 8, "BJ": 8, "num_warps": 4, "num_stages": 2}, status="SAFE:inferred beyond (128,4,32) and (64,4,16)",
                     evidence="B300 (cc 10.3), same stack and sweeps: measured valid at (128,4,32) and (64,4,16), faster than the torch statements; other shapes inferred"),
        "*": dict(settings={"BI": 8, "BJ": 8, "num_warps": 4, "num_stages": 2}, status="SAFE:inferred",
                  evidence="every other capability (sm_86 / sm_89 / sm_120 ...): the one cell of all four certified capabilities (the smallest tile of the swept space, never on a cliff "
                           "on any measured card); not measured there (a build failure names itself and the lever refuses by name)"),
    },
    "pair_fused:epilogue": {                                     # fpf tri-attention epilogue, kernel variant v2: (KVER, BI, BJ, num_warps, num_stages, EXP); shape dims c_z, H, D
        "8.0": dict(settings={"KVER": 2, "BI": 8, "BJ": 4, "num_warps": 4, "num_stages": 1, "EXP": "libdevice"}, when={"c_z_max": 128, "H_max": 4}, status="SAFE",
                    evidence="A100: valid and faster than the torch statements at (128,4,32) and (64,4,16); the v2 epilogue has an sm_80 slowdown growing with c_z x H",
                    alternatives=[dict(settings={"KVER": 2, "BI": 8, "BJ": 8, "num_warps": 8, "num_stages": 1, "EXP": "libdevice"}, when={"c_z_max": 128, "H_max": 8}, status="SAFE",
                                       evidence="A100 (128,8,32): the one measured configuration that stays fast there")],
                    refusal="c_z_above_128_slower_than_stock_on_cc8.0"),                           # every measured v2 epilogue configuration above c_z 128 on sm_80 is slower than the torch statements: those calls go to stock
        "9.0": dict(settings={"KVER": 2, "BI": 16, "BJ": 8, "num_warps": 4, "num_stages": 1, "EXP": "libdevice"}, status="SAFE",
                    evidence="H100: valid and fast at (128,4,32) and (64,2,32); its num_warps 8 twin at (256,8,32) — no sm_90 slowdown"),
        "10.0": dict(settings={"KVER": 2, "BI": 16, "BJ": 8, "num_warps": 4, "num_stages": 1, "EXP": "libdevice"}, status="SAFE:inferred beyond (128,4,32) and (64,4,16)",
                     evidence="B200 (cc 10.0), torch 2.10 / triton 3.6: the 9.0 safe cell measured valid and faster than the torch statements at (128,4,32) and (64,4,16) in the 36-cfg sweeps; other shapes inferred"),
        "10.3": dict(settings={"KVER": 2, "BI": 16, "BJ": 8, "num_warps": 4, "num_stages": 1, "EXP": "libdevice"}, status="SAFE:inferred beyond (128,4,32) and (64,4,16)",
                     evidence="B300 (cc 10.3), same stack and sweeps: valid and faster than the torch statements at (128,4,32) and (64,4,16); other shapes inferred"),
        "*": dict(settings={"KVER": 2, "BI": 8, "BJ": 4, "num_warps": 4, "num_stages": 1, "EXP": "libdevice"}, status="SAFE:inferred",
                  evidence="every other capability (sm_86 / sm_89 / sm_120 ...): the smallest measured-valid tile of the v2 epilogue (the cc 8.0 cell, valid at (128,4,32) and (64,4,16) "
                           "there), single stage, libdevice exp; no shape bound and no refusal above c_z 128 (that slowdown is measured on sm_80 only: unmeasured elsewhere, so engaged and "
                           "named); a build failure names itself and the lever refuses by name"),
    },
    "lnl:transition": {                                          # lnl_fused._fused_transition_kernel (autotune space _TCFG_ALL): one triton.Config
        "*": dict(settings={"BM": 64, "BH": 64, "num_warps": 4, "num_stages": 1}, status="SAFE:inferred",
                  evidence="the smallest tile of the kernel's own configuration space with ONE stage: its (64, 64, 4 warps, 2 stages) member compiled and ran for all 24 keys of the A100 autotune battery "
                           "(a100_lnl_fused_evidence.md, torch 2.13 / triton 3.7.1) and is in the space the cc 9.0 row was tuned from; one stage needs strictly less shared memory — the single-stage form itself is not measured"),
    },
    "lnl:ln_linear": {                                           # lnl_fused._ln_linear_kernel (autotune space _LCFG_ALL): one triton.Config
        "*": dict(settings={"BM": 64, "num_warps": 4, "num_stages": 1}, status="SAFE:inferred",
                  evidence="the smallest tile of the kernel's own configuration space with ONE stage: its (64, 4 warps, 2 stages) member compiled and ran for all 18 keys of the A100 autotune battery and is in the "
                           "cc 9.0 space; the single-stage form itself is not measured"),
    },
    "k2b": {                                                     # fpf_triatt_k2b.triatt_k2b launch cell (every head dim / dtype): the default tile at 1 row/program, 1 stage, no register cap
        "8.0": dict(settings={"BLOCK_M": 64, "BLOCK_N": 32, "ROWS": 1, "num_warps": 4, "num_stages": 1, "ORDER": 0, "MAXNREG": None}, status="SAFE",
                    evidence="A100-SXM4-80GB, torch 2.13 / triton 3.7.1: compiled, ran, finite in the 380-configuration sweep (k2b_sweep.json M64_N32_R1_w4_s1_o0_rNone, x0.70 of the tuned cell at S=768; a100_k2b_evidence.md)"),
        "*": dict(settings={"BLOCK_M": 64, "BLOCK_N": 32, "ROWS": 1, "num_warps": 4, "num_stages": 1, "ORDER": 0, "MAXNREG": None}, status="SAFE:inferred",
                  evidence="the same cell for every other capability (single stage, no register cap, the tuned cell's tile shape); measured on cc 8.0 only"),
    },
}


def _applies(when: Optional[Mapping[str, Any]], dims: Optional[Mapping[str, int]]) -> bool:
    """Whether a row's ``when`` bounds admit a call's shape ``dims``: keys ``<dim>_max`` / ``<dim>_min`` on the lever's own dimension names
    (a row without ``when`` applies to every shape; a bound on a dimension the caller did not name does not apply)."""
    if not when:
        return True
    if dims is None:
        return False
    for k, v in when.items():
        name, _, kind = str(k).rpartition("_")
        if name not in dims:
            return False
        if kind == "max" and not int(dims[name]) <= int(v):
            return False
        if kind == "min" and not int(dims[name]) >= int(v):
            return False
    return True


def safe_settings_for(lever: str, cc: CcLike, dims: Optional[Mapping[str, int]] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """``(row, refusal)``: the SAFE row of ``lever`` on ``cc`` that applies to a call of shape ``dims`` — the row itself when its ``when``
    bounds admit the shape (or it has none), else the first of its ``alternatives`` whose bounds do — or ``(None, <the row's refusal word>)``
    when the lever HAS safe settings on the capability but none admits this shape (a shape class measured slower than the stock statements
    is not served the safe cell: those calls take the caller's per-call stock route, by that word), or ``(None, None)`` when the lever has no
    safe settings on the capability at all."""
    row = safe_row(lever, cc)
    if row is None:
        return None, None
    for cand in [row] + list(row.get("alternatives") or []):
        if _applies(cand.get("when"), dims):
            return cand, None
    return None, str(row.get("refusal") or "no_safe_settings_for_this_shape")


def safe_row(lever: str, cc: CcLike) -> Optional[Dict[str, Any]]:
    """The SAFE row of ``lever`` for capability ``cc`` (``{"settings", "status", "evidence"}``): the exact "<cc>" entry, else "*", else None
    (the lever has no safe settings for this capability: case iii — its refusal)."""
    rows = SAFE_ROWS.get(str(lever), {})
    w = cc_word(cc)
    for key in ((w,) if w is not None else ()) + ("*",):
        if key in rows:
            return rows[key]
    return None


def safe_settings_of(lever: str, cc: CcLike) -> Callable[[], Any]:
    """A provider callable for :meth:`SafeNet.run` / :meth:`SafeNet.no_cell` / :func:`serve`: the lever's safe SETTINGS for ``cc``; raises
    LookupError when the lever has none there (caught by nobody: the caller decides case iii before building)."""
    def provider():
        row = safe_row(lever, cc)
        if row is None:
            raise LookupError("%s: no safe settings for %s" % (lever, where_word(cc, None).split(",")[0]))
        return row["settings"]
    return provider


# ------------------------------------------------------------------------------------------------------------- build failures
_BUILD_MESSAGES = ("PassManager::run failed", "failed to legalize operation", "ptxas", "Internal Triton PTX codegen error")


def build_error_types() -> tuple:
    """triton's compile-time exception classes importable in this process (CompilationError, OutOfResources, PTXASError …); ``()``
    without triton."""
    out = []
    for modname, names in (("triton.compiler.errors", ("CompilationError", "PTXASError")), ("triton.compiler", ("CompilationError",)),
                           ("triton.runtime.errors", ("OutOfResources", "PTXASError")), ("triton.runtime.autotuner", ("OutOfResources",)),
                           ("triton", ("CompilationError", "OutOfResources"))):
        try:
            mod = __import__(modname, fromlist=list(names))
        except ImportError:
            continue
        for n in names:
            t = getattr(mod, n, None)
            if isinstance(t, type) and issubclass(t, BaseException) and t not in out:
                out.append(t)
    return tuple(out)


def catchable() -> tuple:
    """The exception classes :meth:`SafeNet.run` intercepts for classification: triton's compiler classes + RuntimeError (the words
    of its MLIR / PTX stages) — everything else propagates untouched."""
    return build_error_types() + (RuntimeError,)


def is_build_failure(e: BaseException) -> bool:
    """True for a triton COMPILE-TIME failure of a launch (its compiler exception classes, or a RuntimeError carrying the words its
    MLIR / PTX stages raise); never for an out-of-memory (``opt_core.oom.is_oom``, the core's one classifier) and never for a
    condition a running kernel produces."""
    from ..oom import is_oom
    oom = is_oom(e)
    if oom:
        return False
    types_ = build_error_types()
    if types_ and isinstance(e, types_):
        return True
    return isinstance(e, RuntimeError) and any(m in str(e) for m in _BUILD_MESSAGES)


# ------------------------------------------------------------------------------------------------------------------ safety net
class SafeNet:
    """One lever's safety net in this process: whether its SAFE settings serve, why, the ONE line, the census word.

    ``lever``   the name in the line and the refusal (``flash_triattn``, ``pair_fused:transition`` …)
    ``refused`` the lever's refusal exception class for case (iii) (called as ``refused(message)``); its message names the
                lever and the one-flag escape."""

    def __init__(self, lever: str, refused: Callable[[str], BaseException] = RuntimeError):
        self.lever = str(lever)
        self.refused = refused
        self.state: Dict[str, Any] = {"on": False, "reason": None, "where": None, "note": None, "wide": False, "cells": {}}

    # -- state ---------------------------------------------------------------------------------------------------------------
    @property
    def on(self) -> bool:
        """True once the safe settings serve ANYWHERE in this process (a cell of :attr:`cells`, or process-wide :attr:`wide`) — the census fact."""
        return bool(self.state["on"])

    @property
    def wide(self) -> bool:
        """True once the safe settings serve PROCESS-WIDE — the tuned settings failed to BUILD on this stack (every cell of the lever runs the
        safe settings from then on).  A ``no_cell`` engagement never sets it: that one holds for its own cell (:attr:`cells`)."""
        return bool(self.state["wide"])

    @property
    def cells(self) -> Dict[str, str]:
        """``{cell word: reason}`` — the cells (shape words) whose safe settings serve in this process; pinned cells are never in it."""
        return dict(self.state["cells"])

    def serves_safe(self, cell: Any = None) -> bool:
        """Do the safe settings serve ``cell`` in this process?  Process-wide (:attr:`wide`), or this cell engaged by name (``no_cell``)."""
        return bool(self.state["wide"]) or (cell is not None and str(cell) in self.state["cells"])

    def snapshot(self) -> Dict[str, Any]:
        snap = dict(self.state); snap["cells"] = dict(self.state["cells"])
        return snap

    def word(self) -> Optional[str]:
        """``"safe:<reason>"`` once the safe settings serve (the activation line's ``settings=`` fact), else None."""
        return ("safe:%s" % self.state["reason"]) if self.state["on"] else None

    def note(self) -> Optional[str]:
        """``"default:no_row"`` once :meth:`inform` said the lever's DEFAULT settings serve a capability no row names (the activation line's
        ``cells_note=`` fact), else None."""
        return self.state["note"]

    def reset(self) -> None:
        self.state.update(on=False, reason=None, where=None, note=None, wide=False, cells={})

    def inform(self, what: str, where: str, tuned=()) -> None:
        """The lever's DEFAULT (tuned) settings serve although no row names this capability — not an error and not the safe settings: ONE info
        line ``[opt_core/<lever>] default settings (<what>, <where>); tuned rows exist for cc …``, exit 0, the lever engaged; ``note()`` becomes
        ``"default:no_row"``. (A BUILD failure of those settings is still :meth:`run`'s to catch: then the safe settings serve.)"""
        if self.state["note"] is not None:
            return
        self.state["note"] = "default:no_row"
        tail = ("; tuned rows exist for cc " + ", ".join(str(t) for t in tuned)) if tuned else ""
        print("[opt_core/%s] default settings (%s, %s)%s" % (self.lever, what, where, tail), file=sys.stderr, flush=True)

    def engage(self, reason: str, where: str, cell: Any = None, *, wide: Optional[bool] = None) -> None:
        """The lever's safe settings serve — SCOPED: a ``no_cell:<shape>`` reason (or ``cell`` given) engages them for that ONE cell (the
        lever's pinned cells keep their settings; :meth:`serves_safe`), any other reason (``build_failed:…``, or ``wide=True``) for the whole
        process.  Idempotent per scope; ONE line on stderr per NEW engagement (the first line's reason is the census word :meth:`word`)."""
        reason = str(reason)
        if wide is None:
            wide = cell is None and not reason.startswith("no_cell:")
        if not wide and cell is None:
            cell = reason[len("no_cell:"):] if reason.startswith("no_cell:") else reason
        if wide:
            new = not self.state["wide"]
            self.state["wide"] = True
        else:
            new = str(cell) not in self.state["cells"]
            if new:
                self.state["cells"][str(cell)] = reason
        if not self.state["on"]:
            self.state.update(on=True, reason=reason, where=str(where))
        if new:
            print("[opt_core/%s] safe settings served (%s, %s)" % (self.lever, reason, where), file=sys.stderr, flush=True)

    @staticmethod
    def _settings(safe_settings: Any) -> Any:
        """The safe settings: a value, or a provider callable (called when needed — a lever may derive them from the shape)."""
        return safe_settings() if callable(safe_settings) else safe_settings

    def _refuse(self, reason: str, where: str, e: BaseException) -> BaseException:
        return self.refused("%s: the safe settings cannot build/run either (%s, %s): %s: %s — this lever cannot run in this process; run the kit "
                            "with `--mode off` or its explicit opt-out for this lever" % (self.lever, reason, where, type(e).__name__, str(e)[:200]))

    # -- case (ii) by build failure / case (iii) --------------------------------------------------------------------------------
    def run(self, build: Callable[[Any], Any], settings: Any, safe_settings: Any, *, where: str, explicit: bool = False,
            classify: Callable[[BaseException], bool] = is_build_failure, cell: Any = None) -> Any:
        """``build(settings)``; when that fails to BUILD (``classify``) and the caller did not pin the settings (``explicit`` False):
        engage process-wide (``build_failed:<exception class>``, ONE line) and ``build(safe_settings)``; a build failure of the safe
        settings raises the lever's refusal.  With the safe settings already serving THIS call — process-wide (:attr:`wide`), or ``cell``
        given and engaged by name — ``build(safe_settings)`` runs directly; another cell's ``no_cell`` engagement never demotes these
        ``settings`` (a pinned cell keeps its own).  Every other exception — and any exception under ``explicit`` — propagates unchanged."""
        catch = catchable()
        if not (self.serves_safe(cell) and not explicit):
            try:
                return build(settings)
            except catch as e:
                if explicit or not classify(e):
                    raise
                self.engage("build_failed:%s" % type(e).__name__, where, wide=True)
        try:
            return build(self._settings(safe_settings))
        except catch as e2:
            if not classify(e2):
                raise
            raise self._refuse(self.state["reason"] or ("build_failed:%s" % type(e2).__name__), where, e2) from e2

    # -- case (ii) by a missing cell / case (iii) -------------------------------------------------------------------------------
    def no_cell(self, shape_word: str, build: Callable[[Any], Any], safe_settings: Any, *, where: str,
                classify: Callable[[BaseException], bool] = is_build_failure) -> Any:
        """No cell row for this shape / capability: engage FOR THIS CELL (``no_cell:<shape>``, ONE line the first time this shape word
        engages) and ``build(safe_settings)``; a build failure of the safe settings raises the lever's refusal; every other exception
        propagates.  The lever's pinned cells are untouched (:meth:`run` keeps building their own settings)."""
        self.engage("no_cell:%s" % shape_word, where, cell=str(shape_word))
        try:
            return build(self._settings(safe_settings))
        except catchable() as e:
            if not classify(e):
                raise
            raise self._refuse("no_cell:%s" % shape_word, where, e) from e


# --------------------------------------------------------------------------------------------------------------- one-call form
def serve(net: SafeNet, table: Mapping[str, Mapping], cc: CcLike, triton_mm_: Optional[str], cell, build: Callable[[Any], Any],
          safe_settings: Any, *, shape_word: Optional[str] = None, explicit_settings: Any = None, default_settings: Any = None) -> Tuple[str, Any]:
    """Resolve and run ONE cell of a lever under its safety net — the whole of (i)-(iii) in one call. Returns ``(served, result)`` where
    ``served`` is the row key that served (``"<cc>|<mm>"`` / ``"<cc>|*"``), ``"default"`` (``default_settings``: the lever's capability-free
    table, e.g. its H100 rows), ``"explicit"`` (caller-pinned ``explicit_settings``, never retried) or ``"safe"`` (the safe settings serve:
    :meth:`SafeNet.word` says why).

      * ``explicit_settings`` given → ``build(explicit_settings)`` as pinned;
      * the net already serves the safe settings PROCESS-WIDE (a build failure earlier) → ``build(safe)``;
      * :func:`resolve_cell` finds the cell → :meth:`SafeNet.run` with ITS settings — also after another cell's ``no_cell`` engagement
        (build failure → safe + ONE line; safe failing → the lever's refusal);
      * no cell and ``default_settings`` given → :meth:`SafeNet.run` with them (a capability without rows runs the lever's own defaults);
      * no cell and no default → :meth:`SafeNet.no_cell` (``no_cell:<shape_word>``: safe + ONE line; safe failing → the lever's refusal).

    ``safe_settings``: a value or a provider callable."""
    where = where_word(cc, triton_mm_)
    if explicit_settings is not None:
        return "explicit", net.run(build, explicit_settings, safe_settings, where=where, explicit=True)
    if net.wide:
        return "safe", net.run(build, None, safe_settings, where=where)
    key, settings = resolve_cell(table, cc, triton_mm_, cell)
    if key is None and default_settings is not None:          # no row names this capability: the lever's default (tuned) settings, said ONCE
        if cc is not None and table:
            net.inform("no row for cc %s" % cc_word(cc), where, tuned=table_ccs(table))
        key, settings = "default", default_settings
    if key is None:
        return "safe", net.no_cell(str(shape_word if shape_word is not None else cell), build, safe_settings, where=where)
    result = net.run(build, settings, safe_settings, where=where)
    return ("safe" if net.wide else key), result
