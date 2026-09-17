"""The device loader: which launch cells serve this process's GPU, decided ONCE per device and said in ONE line.

Order: the engineering literal ``FPF_TRIMUL_V4_CFG`` if set; else the package's ONE table (``table.CORE_TABLE_PATH``, ``table.select``: the exact
``'<cc>|<triton M.m>'`` row, then ``'<cc>|*'``; a descriptor cell (K1 ``tma`` / ``tma2``, K3 ``tma``) needs triton's tensor-descriptor API; at the parts the table lists under ``kit_keys`` the
kit's own table — ``FPF_TRIMUL_V4_CELLS`` — serves the part when it has a row for it); else — no admitted row for this capability — the SAFE cell of
``opt_core.kernels.safe_settings`` (``SAFE_ROWS['pair_fused:trimul']``: the capability's row or its any-capability ``*`` row), engaged and named
(``safe settings served (no_cell:<cc>|<mm>, …)``); else None: nothing serves the capability (``generic`` refuses ``no-cell`` by name). The kit table's
own row for the part is named on the line (``kit table … certified row <key>`` | ``no row for this part``): evidence, never the routing decision
outside ``kit_keys``. After the SAFE cell too fails to BUILD in this process the lever is off BY NAME (:func:`lever_off`, ``cell=none: <why>``) and
``generic`` raises its cannot-run refusal for the caller's mode to refuse by name."""
import os, sys, json, torch
from opt_core.kernels import safe_settings as SAFE          # the core's one safety-net mechanism (absolute: this package is served under its routed top-level name too)
from . import kernels as K
from . import table as T

TABLE_PATH = T.CORE_TABLE_PATH                                   # the table this process reads (tests point it elsewhere)
_CFG_OVERRIDE = os.environ.get(T.CFG_ENV)                        # engineering only: JSON literal {"k1": {...}, "k3": {...}}
KIT_TABLE_PATH = os.environ.get(T.KIT_TABLE_ENV) or None         # the kit's own certified table (its evidence; its cells at the kit_keys parts)
_CELL_CACHE = {}
INFO = {}                                                       # str(device) -> {device, cc, triton, cfg, source, key, kind, served_by, kit_table, kit_row, cells_sha}
_OFF = {"why": None}                                            # lever off for the process after the SAFE cell failed to build: 'none: <exception class>'
SAFE_LEVER = "pair_fused:trimul"                                            # this kernel's safe-settings lever name (opt_core.kernels.safe_settings SAFE_ROWS)


def _log(msg):
    sys.stderr.write("[fpf_trimul_v4] %s\n" % msg); sys.stderr.flush()


def cells_sha():
    import hashlib
    try:
        with open(TABLE_PATH, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()[:16]
    except OSError:
        return "missing"


def triton_mm():
    try:
        import triton
        v = triton.__version__.split("+")[0].split(".")
        return "%s.%s" % (v[0], v[1])
    except Exception:
        return "?"


def has_desc() -> bool:
    """Whether this triton builds the descriptor cells — the TMA K1 `_k1t` (device-side tensor descriptors + a settable allocator) and the host-descriptor K1 / K3 of
    kernels.kdesc (impl 'tma2' / K3 'tma'); one API era (triton >= 3.4), one answer."""
    try:
        import triton
    except ImportError:
        return False
    return bool(K.HAS_DESC and hasattr(triton, "set_allocator"))


def _net():
    from .generic import TrimulUnsupported
    return SAFE.SafeNet(SAFE_LEVER, refused=lambda msg: TrimulUnsupported("refused", msg))


class _LazyNet:
    """The process's safety net for this lever, created on first use (generic imports torch)."""
    _n = None
    def __getattr__(self, k):
        if _LazyNet._n is None:
            _LazyNet._n = _net()
        return getattr(_LazyNet._n, k)


NET = _LazyNet()


def safe_cfg(cc):
    """The SAFE cell for capability ``cc`` as a loader cfg, or None (the lever has no safe settings there)."""
    srow = SAFE.safe_row(SAFE_LEVER, cc)
    if srow is None:
        return None
    return {"k1": dict(srow["settings"]["k1"]), "k3": dict(srow["settings"]["k3"]), "overrides": {}}


def lever_off(why):
    """Turn the lever off BY NAME for the rest of the process (the SAFE cell failed to build): every later ``cell_for`` answers None with source
    ``none: <why>``; ``generic`` raises its cannot-run refusal for the caller's mode to refuse by that name; said once."""
    if _OFF["why"] is None:
        _OFF["why"] = str(why)
        _log("cell=none: %s -> the lever cannot run in this process: every later call is refused by that name (cannot_run) for the caller's mode to refuse" % why)
    for key in list(_CELL_CACHE):
        _CELL_CACHE[key] = None
        if key in INFO:
            INFO[key] = dict(INFO[key], cfg=None, source="none: %s" % _OFF["why"], key=None, kind=None, served_by=None)


def off_word():
    """``'none:<exception class>'`` once :func:`lever_off` fired, else None."""
    return None if _OFF["why"] is None else "none:%s" % _OFF["why"].split(":")[0].split(" ")[0]


def _key(device) -> str:
    """The per-device key of INFO / the cell cache: ``str(torch.device)`` with its index (an index-less ``cuda`` means the current device when CUDA is
    up, else index 0). Touches no CUDA state: the loader decides from (cc, triton) alone, so a pre-flight that supplies the capability runs on a host without a GPU."""
    d = torch.device(device)
    if d.type == "cuda" and d.index is None:
        d = torch.device("cuda", torch.cuda.current_device() if torch.cuda.is_initialized() else 0)
    return str(d)


def _device_name(device) -> str:
    """The card's name for the line, or ``?`` on a host where CUDA cannot initialise (the decision needs only the capability)."""
    try:
        return torch.cuda.get_device_name(device)
    except (RuntimeError, AssertionError):
        return "?"


def _kit_table():
    """(the kit's table, note): the JSON at ``FPF_TRIMUL_V4_CELLS``; (None, '') when the process exports none; (None, <named problem>) when unreadable."""
    if not KIT_TABLE_PATH:
        return None, ""
    try:
        return T.load_table(KIT_TABLE_PATH), ""
    except (OSError, ValueError) as e:
        return None, "; kit table %s unreadable (%s: %s): the package table alone serves" % (KIT_TABLE_PATH, type(e).__name__, str(e)[:80])


def cell_for(device):
    """-> the loader cfg ``{"k1": {...}, "k3": {...}, "overrides": {...}}`` serving ``device``, or None. Decided and printed once per device."""
    key = _key(device)
    if key in _CELL_CACHE:
        return _CELL_CACHE[key]
    cc, tmm = "%d.%d" % torch.cuda.get_device_capability(device), triton_mm()
    row_key = kind = served_by = kit_row = None
    if _OFF["why"] is not None:
        cfg, why = None, "none: %s" % _OFF["why"]
    elif _CFG_OVERRIDE:
        cfg, why, served_by = json.loads(_CFG_OVERRIDE), "%s override (ENGINEERING)" % T.CFG_ENV, "engineering"
    else:
        kit_table, kit_note = _kit_table()
        sel = T.select(T.load_table(TABLE_PATH), cc, tmm, kit_table=kit_table, has_desc=has_desc())
        kit_row = sel.kit_row
        skipped = ("; skipped: " + "; ".join(sel.skipped)) if sel.skipped else ""
        if kit_table is not None:
            kit_note = "; kit table %s: %s" % (os.path.basename(os.path.dirname(KIT_TABLE_PATH)) + "/" + os.path.basename(KIT_TABLE_PATH),
                                                ("certified row %s" % kit_row) if kit_row else "no row for this part (uncertified by this kit)")
        if sel.cfg is not None:
            cfg, row_key, kind, served_by = sel.cfg, sel.key, sel.kind, sel.source
            note = "" if kind == "exact" else "; triton %s has no row of its own on cc %s: the capability's any-triton cells serve" % (tmm, cc)
            why = "table.json[%s] (%s match, source=%s, status=%s%s%s)%s" % (row_key, kind, sel.source, (sel.status or "").split(" (")[0].split(":")[0][:40], note, kit_note, skipped)
        else:                                                            # no admitted row for this capability: the lever's SAFE cell, engaged and named ONCE
            cfg = safe_cfg(cc)
            if cfg is not None:
                row_key, kind, served_by = "safe", "safe", "safe"
                srow = SAFE.safe_row(SAFE_LEVER, cc) or {}
                why = "safe settings (opt_core.kernels.safe_settings SAFE_ROWS[%s], status=%s; no table.json row for cc %s%s)%s" % (SAFE_LEVER, srow.get("status", "?"), cc, kit_note, skipped)
                NET.engage("no_cell:%s|%s" % (cc, tmm), SAFE.where_word(cc, tmm))
            else:
                why = "no table.json row and no safe settings for cc %s (triton %s): refused by name (no-cell)%s%s" % (cc, tmm, kit_note, skipped)
    INFO[key] = {"device": _device_name(device), "cc": cc, "triton": tmm, "cfg": cfg, "source": why, "key": row_key, "kind": kind,
                 "served_by": served_by, "kit_table": KIT_TABLE_PATH, "kit_row": kit_row, "cells_sha": cells_sha()}
    _CELL_CACHE[key] = cfg
    _log("cell on %s: %s -> %s" % (key, why, ("SERVE %s%s" % (json.dumps({"k1": cfg["k1"], "k3": cfg["k3"]}), _desc_gate_word(cfg))) if cfg else "STOCK/None for all calls"))
    return cfg


def _desc_gate_word(cfg) -> str:
    """` desc_gate=Np>=<n>` when the row names a host-descriptor cell (its kernels serve plane extents >= kdesc.N_MIN_DESC on bf16 input; smaller calls and fp32
    take the replaced kernels of the same numbers — kernels.LAUNCHES on the exit COUNTS line says which ran), else ``."""
    from . import kdesc as KD
    return " desc_gate=Np>=%d" % KD.N_MIN_DESC if T.needs_desc(cfg) and any(dict(v).get("impl") == "tma2" or (k.startswith("k3") and dict(v).get("impl") == "tma")
                                                                         for k, v in [("k1", cfg.get("k1") or {}), ("k3", cfg.get("k3") or {})] + list((cfg.get("overrides") or {}).items())) else ""


def cell_word(device) -> str:
    """The activation-line fact for ``device``: ``cell=<row key>`` (``:kit-table`` appended when the kit's own row serves a kit_keys part) |
    ``cell=safe(uncertified <cc>|<mm>)`` | ``cell=safe(<build_failed:Exc>)`` | ``cell=none:<why>`` (lever off) | ``cell=no-cell`` | ``cell=?`` (not decided yet)."""
    if _OFF["why"] is not None:
        return "cell=" + off_word()
    info = INFO.get(_key(device))
    if info is None:
        return "cell=?"
    if info.get("served_by") == "safe":
        reason = NET.state.get("reason") or ""
        return "cell=safe(uncertified %s|%s)" % (info.get("cc"), info.get("triton")) if reason.startswith("no_cell:") else "cell=safe(%s)" % reason
    if info.get("cfg") is None:
        return "cell=no-cell"
    by = info.get("served_by") or "core"
    return "cell=%s%s" % (info.get("key"), "" if by == "core" else ":" + by)          # ':kit-table' = the kit's own row serves this kit_keys part


def run(build, cfg, device):
    """``build(cfg)`` under the lever's safety net: a BUILD failure of ``cfg`` (a triton compile / launch-resource error, ``safe_settings.is_build_failure``)
    engages the SAFE cell for the process (ONE line) and builds that; the SAFE cell failing to build — or no SAFE cell on this capability — turns the lever off
    BY NAME (:func:`lever_off`) and raises ``TrimulUnsupported('none', ...)``: the caller's stock TriMul serves. Every other exception propagates unchanged."""
    from .generic import TrimulUnsupported
    key = _key(device)
    cc = (INFO.get(key) or {}).get("cc") or "%d.%d" % torch.cuda.get_device_capability(device)
    provider = lambda: (safe_cfg(cc) or _no_safe(cc))
    try:
        out = NET.run(build, cfg, provider, where=SAFE.where_word(cc, triton_mm()), explicit=bool(_CFG_OVERRIDE))
    except TrimulUnsupported as e:
        if e.reason != "refused":
            raise
        cause = e.__cause__
        lever_off("%s (the SAFE cell failed to build after %s)" % (type(cause).__name__ if cause is not None else "BuildFailed", NET.state.get("reason")))
        raise TrimulUnsupported("none", str(e)) from e
    except _NoSafe as e:
        lever_off("%s (no safe settings for cc %s)" % (NET.state.get("reason") or "build_failed", cc))
        raise TrimulUnsupported("none", str(e)) from e
    if NET.on and (INFO.get(key) or {}).get("served_by") not in (None, "safe"):     # the net switched this process to the SAFE cell during this call: cache it, keep INFO true
        scfg = safe_cfg(cc)
        _CELL_CACHE[key] = scfg
        INFO[key] = dict(INFO[key], cfg=scfg, key="safe", kind="safe", served_by="safe", source="safe settings (%s)" % NET.state.get("reason"))
    return out


class _NoSafe(LookupError):
    pass


def _no_safe(cc):
    raise _NoSafe("%s: no safe settings for cc %s" % (SAFE_LEVER, cc))
