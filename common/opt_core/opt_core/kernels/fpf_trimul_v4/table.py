"""The fpf_trimul_v4 cell table and its ONE selection statement — read by the device loader (``cells.cell_for``), the row-block provider
(``opt_core.mem.rowpair.trimul_fused``) and a kit's pre-flight on a named (cc, triton) part. Pure: no torch, no triton.

The table is the package's own ``table.json`` (:data:`CORE_TABLE_PATH`), the one table. It maps ``'<cc M.m>|<triton M.m>'`` (exact) and
``'<cc M.m>|*'`` (any other triton on that capability) to rows ``{"k1": {...}, "k3": {...}, "status": <word>, "overrides": {...}}``: ``k1``/``k3``
are the (C, D) = (256, 256) no-bias launch cells, ``overrides`` the per-shape cells keyed ``k1_C<C>_D<D>[_bias]`` / ``k3_...`` (:func:`resolve_cfg`).
A part is served by the first of its candidate keys (:func:`candidate_keys`) whose row is admitted (status prefix :data:`SERVED_STATUS_PREFIX`) and
whose cells the stack can build (an ``impl='tma'`` / ``'tma2'`` K1 or ``'tma'`` K3 needs triton's tensor-descriptor API; without it the next candidate serves). A kit exports its own
certified table (``FPF_TRIMUL_V4_CELLS``, :data:`KIT_TABLE_ENV`; the same row format): it names, per part, whether this kit certified the cells it is
served, and at the parts the package table lists under ``kit_keys`` — parts where two sweeps measured different cells — a kit table
that serves the part serves it (``source='kit-table'``); everywhere else the package table's row serves (``source='core'``)."""
import json, os
from typing import Any, Dict, Mapping, NamedTuple, Optional, Tuple

CORE_TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "table.json")   # the ONE table
SERVED_STATUS_PREFIX = "CERTIFIED"                           # the status word (prefix, case-insensitive) of a row the table serves
KIT_TABLE_ENV = "FPF_TRIMUL_V4_CELLS"                        # the kit's own certified table (path): its evidence, and its cells at the kit_keys parts
CFG_ENV = "FPF_TRIMUL_V4_CFG"                                # engineering only: a JSON literal {"k1": {...}, "k3": {...}} served instead of the table
ALLOW_DEFAULT_ENV = "FPF_TRIMUL_V4_ALLOW_DEFAULT_CELLS"      # engineering only: '1' admits rows of any status of a KIT table (a pre-flight's switch; the package table needs none)


class Selection(NamedTuple):
    """What serves a (cc, triton) part: the row ``key`` (``'<cc>|<mm>'`` / ``'<cc>|*'``), ``kind`` (``exact`` | ``generic-triton``), the row
    ``cfg`` (``{"k1", "k3", "overrides"}``), ``source`` (``core`` | ``kit-table``), ``status`` (the row's word), ``skipped`` (candidate rows passed
    over, with the reason) and ``kit_row`` (the kit table's admitted row key for this part, None when it has none or no kit table was given).
    ``key`` None: no admitted row for this part."""
    key: Optional[str]
    kind: Optional[str]
    cfg: Optional[Dict[str, Any]]
    source: Optional[str]
    status: Optional[str]
    skipped: Tuple[str, ...]
    kit_row: Optional[str] = None


def load_table(path: Optional[str] = None) -> Dict[str, Any]:
    """The table at ``path`` (default :data:`CORE_TABLE_PATH`)."""
    with open(path or CORE_TABLE_PATH) as fh:
        return json.load(fh)


def rows(table: Mapping) -> Dict[str, Mapping]:
    """The ``'<cc>|<triton>'`` rows of ``table`` (its ``_doc`` / ``kit_keys`` members are not rows)."""
    return {k: v for k, v in table.items() if isinstance(v, Mapping) and "k1" in v and "k3" in v}


def candidate_keys(cc: str, triton_mm: str) -> Tuple[Tuple[str, str], ...]:
    """The table keys tried for a (cc ``'M.m'``, triton ``'M.m'``) part, in order, with the name of each match:
    the exact ``'<cc>|<triton>'`` row, then the capability's ``'<cc>|*'`` row."""
    return (("%s|%s" % (cc, triton_mm), "exact"), ("%s|*" % cc, "generic-triton"))


def status_word(row: Mapping) -> str:
    """A row's status word (``'DEFAULT'`` when the row names none)."""
    return str(row.get("status", "DEFAULT"))


def admitted(row: Mapping, allow_default_cells: bool = False) -> bool:
    """Whether a table serves ``row``: its status starts with ``SERVED_STATUS_PREFIX`` (case-insensitive), or ``allow_default_cells`` (engineering)."""
    return bool(allow_default_cells) or status_word(row).upper().startswith(SERVED_STATUS_PREFIX)


def allow_default(environ: Optional[Mapping[str, str]] = None) -> bool:
    """The engineering switch ``FPF_TRIMUL_V4_ALLOW_DEFAULT_CELLS=1`` in ``environ`` (default ``os.environ``): a kit pre-flight may admit rows of any status of its own table."""
    return (os.environ if environ is None else environ).get(ALLOW_DEFAULT_ENV, "0") == "1"


def kit_keys(table: Mapping) -> Tuple[str, ...]:
    """The exact ``'<cc>|<mm>'`` parts at which a kit table that serves the part serves it (``table['kit_keys']``)."""
    return tuple(str(k) for k in (table.get("kit_keys") or ()))


def row_cfg(row: Mapping) -> Dict[str, Any]:
    """A row as the loader's cfg ``{"k1", "k3", "overrides"}`` (``_``-prefixed override notes dropped)."""
    return {"k1": dict(row["k1"]), "k3": dict(row["k3"]), "overrides": {k: dict(v) for k, v in (row.get("overrides") or {}).items() if not str(k).startswith("_")}}


def resolve_cfg(cfg: Mapping, C: int, D: int, has_bias: bool) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """cfg -> the (k1, k3) launch cells for this (C, D, bias): the first override among ``k?_C<C>_D<D>_bias`` (bias only), ``k?_C<C>_D<D>``,
    ``k?_C<C>_bias`` (bias only), ``k?_C<C>``; else the cfg's own ``k1`` / ``k3``."""
    k1, k3 = dict(cfg["k1"]), dict(cfg["k3"])
    ov = {k: v for k, v in (cfg.get("overrides") or {}).items() if not str(k).startswith("_")}
    sfx = "_bias" if has_bias else ""
    for kern in ("k1", "k3"):
        for key in ("%s_C%d_D%d%s" % (kern, C, D, sfx), "%s_C%d_D%d" % (kern, C, D), "%s_C%d%s" % (kern, C, sfx), "%s_C%d" % (kern, C)):
            if key in ov:
                if kern == "k1":
                    k1 = dict(ov[key])
                else:
                    k3 = dict(ov[key])
                break
    return k1, k3


INT32_MAX = 2 ** 31 - 1                                     # the descriptor cells' token-row coordinate over a batch's B*N*N rows is int32 (kernels.batch_launch_limit 'tma_rows>int32')
DESC_IMPLS = ("tma", "tma2")                                  # cell impl words that need triton's tensor-descriptor API (K1 'tma' / 'tma2', K3 'tma'); absent / 'ptr' = pointer kernels


def needs_desc(row: Mapping) -> bool:
    """Whether any cell of ``row`` (k1, k3 or an override) names a descriptor impl."""
    cells = [row.get("k1") or {}, row.get("k3") or {}] + [v for k, v in (row.get("overrides") or {}).items() if not str(k).startswith("_")]
    return any(dict(c).get("impl", "ptr") in DESC_IMPLS for c in cells)


def _first(table: Mapping, cc: str, triton_mm: str, has_desc: bool, allow_default_cells: bool = False):
    """(key, kind, row, skipped) of the first admitted, buildable candidate row of (cc, triton) in ``table``; key None when there is none."""
    skipped = []
    for key, kind in candidate_keys(str(cc), str(triton_mm)):
        row = table.get(key) if isinstance(table, Mapping) else None
        if not isinstance(row, Mapping) or "k1" not in row:
            continue
        if not admitted(row, allow_default_cells):
            skipped.append("%s status=%s (not served)" % (key, status_word(row)[:40])); continue
        if needs_desc(row) and not has_desc:
            skipped.append("%s needs the triton tensor-descriptor API" % key); continue
        return key, kind, row, tuple(skipped)
    return None, None, None, tuple(skipped)


def select(table: Mapping, cc: str, triton_mm: str, *, kit_table: Optional[Mapping] = None, has_desc: bool = True) -> Selection:
    """The ONE selection statement: the first admitted candidate row of (cc, triton) in the package ``table`` whose cells this stack can build (a descriptor
    cell, :func:`needs_desc`, needs the tensor-descriptor API, ``has_desc``); at a ``kit_keys`` part a ``kit_table`` row serving the part serves instead (``source='kit-table'``).
    ``kit_row`` names the kit table's own admitted row for the part (its evidence) or None."""
    key, kind, row, skipped = _first(table, cc, triton_mm, has_desc)
    kkey = kkind = krow = None
    if kit_table is not None:
        kkey, kkind, krow, _ = _first(kit_table, cc, triton_mm, has_desc)
    if krow is not None and "%s|%s" % (cc, triton_mm) in kit_keys(table):
        return Selection(kkey, kkind, row_cfg(krow), "kit-table", status_word(krow), skipped, kkey)
    if row is None:
        return Selection(None, None, None, None, None, skipped, kkey)
    return Selection(key, kind, row_cfg(row), "core", status_word(row), skipped, kkey)


def cells_for(table: Mapping, cc: str, triton_mm: str, C: int, D: int, has_bias: bool, *, kit_table: Optional[Mapping] = None,
              has_desc: bool = True) -> Tuple[Selection, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """A pre-flight's whole answer for one part and shape: ``(selection, k1, k3)`` — ``k1``/``k3`` None when no row serves the part."""
    sel = select(table, cc, triton_mm, kit_table=kit_table, has_desc=has_desc)
    if sel.cfg is None:
        return sel, None, None
    k1, k3 = resolve_cfg(sel.cfg, int(C), int(D), bool(has_bias))
    return sel, k1, k3


def table_ccs(table: Mapping) -> Tuple[str, ...]:
    """The capabilities the table has rows for, ascending."""
    return tuple(sorted({k.split("|")[0] for k in rows(table)}, key=float))
