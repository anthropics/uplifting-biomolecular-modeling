"""opt_core.cell_census — the process-wide census of CELL COVERAGE: for every call class a kernel provider face decided, whether a measured cell
decided it, the nearest measured cell was inherited, a row stepped aside by name, or the table named the stock op.  Standard library only.

Every provider face of :mod:`opt_core.kernels` (triattn, triattn_xla, trimul, apb, ln, transition, pallas) resolves a call to a ROW through its
cell table.  At the point where the row is DECIDED the face records ONE fact per distinct key here (pure observation: the selection is made
first and is byte-identical with or without this module; a repeated key is one dict lookup; no framework import, no device synchronisation,
nothing read from a tensor beyond the shape / dtype / device words the face already holds)::

    from opt_core import cell_census
    cell_census.record("trimul", dict(cc="9.0", stack="H100:2.13.0+cu130/3.7.1/cueq0.11.1", dtype="bf16", shape="C128H128", bucket="N=3000",
                                      form="out.fwd", word="fast"), "inherited", "v4", cell_id="9.0|bf16|C128|H128|N<=2048|out|fwd",
                       note="beyond_measured(N<=2048)")

OUTCOMES (the ``outcome`` word; what the face knew when it decided):

    cell_hit        an exactly matching measured cell decided the row (the call's N inside a measured bucket of its own family).  A cell measured
                    on a REFERENCE stack column of the same cc and served on a sibling stack (the caller's own library stack has no column) is a
                    cell hit too, ``note=inherited_stack(measured_on=<reference stack>)`` -- the (cc, dtype, width, form, size) cell exists; only
                    the timing column is the sibling's (tolerance-class words; an EXACT word is served on a vouched stack only -- elsewhere the
                    face refuses the exact row by name = named_fallback ``<row>:exact_vouch_not_recorded_on:<stack>``, never inherited)
    inherited       NO measured cell for this (cc | dtype | shape | N-bucket | direction / form) on ANY stack of the cc: the face served the row +
                    launch cell of the NEAREST measured cell (same cc / dtype / width / form, the closest N-bucket -- the larger when equidistant)
                    -- ``cell_id`` names that cell; ``none`` when the family has no measured cell at all (the face's named default served).  These
                    are the TRUE gaps (``python -m opt_core.cell_census --true-gaps <dump>...`` lists them by name)
    named_fallback  a row refused BY NAME (no prebuilt binary for this interpreter, a library absent from the image, a shape outside the row's
                    envelope, a capture-unsafe row under capture ...) and the face stepped aside to the next measured row or the stock row --
                    ``refused`` = ``<row>:<reason>``; ``served`` = the row that served (``caller:<row>`` when the face raised the refusal and the
                    CALLER binds the named fallback row)
    stock           the tier word resolved to the named STOCK op by the table (a measured decision: the stock op won that cell)
    opt_in          an explicit row word (the default rule: exactly that row; the cell, when one exists, only supplies the launch cell)

TOKENS -- one line per distinct key per process, greppable, grammar pinned by the core's tests (``TOKEN_RE``)::

    UNCOVERED_CELL:<provider>|cc<cc>|<stack>|<dtype>|<shape>|<bucket>|<form> word=<word> served=<row> nearest=<cell id | none>[ note=<note>]
    NAMED_FALLBACK:<provider>|cc<cc>|<stack>|<dtype>|<shape>|<bucket>|<form> word=<word> refused=<row>:<reason> served=<row>[ note=<note>]
    CELL_HIT:<provider>|cc<cc>|<stack>|<dtype>|<shape>|<bucket>|<form> word=<word> served=<row> cell=<cell id>[ note=<note>]
    STOCK_CELL:<provider>|cc<cc>|<stack>|<dtype>|<shape>|<bucket>|<form> word=<word> served=<row> cell=<cell id | none>[ note=<note>]
    OPT_IN:<provider>|cc<cc>|<stack>|<dtype>|<shape>|<bucket>|<form> word=<word> served=<row> cell=<cell id | none>[ note=<note>]

The seven ``|`` segments are the KEY (with ``word``): ``provider`` (triattn | triattn_xla | trimul | apb | ln | transition | pallas), ``cc``
('9.0'), ``stack`` (the face's stack word, ``-`` when the caller named none), ``dtype`` (the table's precision word: bf16 | fp32 | tf32 | f32z_bf16
...), ``shape`` (the width words of that table: ``C128H128``, ``D32H4``, ``pair_c128_n4`` ...), ``bucket`` (``N<=800`` = the measured bucket that
holds the call; ``N=3000`` = the call's own size where no bucket holds it), ``form`` (direction / pass / call form: ``out.fwd``, ``fwdbwd``,
``fwd.eager.swiglu`` ...).  Values never contain spaces or ``|`` (sanitised here); ``cell=`` / ``nearest=`` is the last field before the optional
``note=`` and is the table's own cell key verbatim (it contains ``|``).

NEAREST (the serving policy for a key with no measured cell, one function with hard guards: :func:`nearest`).  A face serves the row +
launch cell of the nearest measured cell ONLY across SIZE: candidates must share the query's card (cc: a cc 9.0 launch cell is never served
on cc 8.0 or the reverse), dtype, numerics class (an exact request inherits only from exact / bitwise / stock-class cells, never from a
tolerance-class cell; an exact vouch is STACK-SPECIFIC: exact inherits only from cells vouched on the running stack key -- elsewhere the face
refuses the exact row by name and the tier falls to its floor row), call form, and shape words (width / heads / head_dim: no inheritance -- the face
refuses the kernel row by name or serves its named default, and the census says ``nearest=none``); among the survivors the covering bucket
(the smallest measured size at or above the call) is a CELL HIT, else the closest size serves (the larger when equidistant) = INHERITED.

EXIT LINES (stderr, the stream of the kit's ACTIVE / LEVER / EXIT lines; registered through :func:`opt_core.report.register_exit_tally` at the
first record, so a forced exit prints them too; ``OPT_CORE_CELL_CENSUS_PRINT=0`` silences them)::

    [opt_core] CELLS pid=<pid> keys=<n> cell_hit=<n> inherited=<n> named_fallback=<n> stock=<n> opt_in=<n> alerts=<n> dump=<path | none>[ context=<k:v,...>]
    UNCOVERED_CELL:...        (every alert token of the process, one per line, in decision order)
    NAMED_FALLBACK:...

ENVIRONMENT::

    OPT_CORE_CELL_CENSUS=/path/file.json   at interpreter exit the JSON dump is written there (a directory, or a value ending in '/', receives
                                           cell_census.<pid>.json -- one file per rank); unset = nothing is written (the registry still counts)
    OPT_CORE_CELL_CENSUS_PRINT=1           each NEW key whose outcome is inherited | named_fallback prints its token line once to stderr at the
                                           moment it is decided; =2 prints every new key's line (cell hits, stock cells and opt-ins too);
                                           =0 prints nothing, at decision time or at exit; unset = the exit lines only

INHERITED NOTES (the ``note=`` word of an UNCOVERED_CELL line says WHICH dimension had no measured cell): ``beyond_measured`` (the call's
size is above every measured bucket: the largest bucket's row serves -- the size neighbour of the NEAREST rule), ``stack(<served stack>)``
(the running library stack has no column in the cell: the reference stack's winner serves), ``guard:<dimension>(...)`` (the face served a
neighbour across a dimension the nearest rule does NOT cross -- heads, samples, timing, direction, dtype, cc: a cell to measure or a
serving to retire, listed for the table's owner), ``family:none(...)`` (no measured cell of this family at all: ``nearest=none``, the face's
named default served).

CONTEXT.  A kit names the run once per process (or per item when the size changes) -- free-form words, carried into every entry recorded after
the call and into the dump's ``context`` block; the summariser's table axes are ``engine``, ``mode``, ``card``, ``size``::

    cell_census.set_context(engine="<kit word>", mode="fast", card="H100", size=800)

READERS.  ``table()`` -> [entry dict, ...] in decision order; ``tokens()`` -> every entry's token line; ``alerts()`` -> the UNCOVERED_CELL /
NAMED_FALLBACK lines only; ``dump('json' | 'tsv')`` -> str; ``reset()`` forgets everything (tests; a face's own selection memo is the face's).
``python -m opt_core.cell_census --summarize a.json b.json [--engine E] [--fmt tsv|json]`` merges dumps into the coverage table
(engine x mode x card x size x provider -> counts per outcome + the alert tokens); ``--tokens`` prints the alert tokens of the dumps.

Never raises into a face: ``record`` swallows its own errors (a census counts, it never gates a served call).
"""
from __future__ import annotations

import atexit
import json
import os
import re
import sys
import threading
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

__all__ = ["ENV_FILE", "ENV_PRINT", "OUTCOMES", "ALERT_OUTCOMES", "TOKEN_WORDS", "KEY_FIELDS", "TOKEN_RE", "EXIT_TAG", "record", "set_context", "context",
           "table", "tokens", "alerts", "dump", "reset", "write", "summarize", "cell_census_dump", "clean", "bucket_word", "nearest", "Nearest",
           "exit_lines", "EXIT_MAX_TOKENS", "INHERITED_SPLIT", "SIBLING_NOTE", "SIBLING_COUNT", "EXACT_VOUCH_WORD", "GAP_KINDS", "sibling_note",
           "normalize_entry", "gap_kind", "true_gaps", "parse_token", "parse_log", "main"]

ENV_FILE = "OPT_CORE_CELL_CENSUS"
ENV_PRINT = "OPT_CORE_CELL_CENSUS_PRINT"
OUTCOMES = ("cell_hit", "inherited", "named_fallback", "stock", "opt_in")
ALERT_OUTCOMES = ("inherited", "named_fallback")                  # the outcomes whose token prints under PRINT=1 (a caller's whitelist names them)
TOKEN_WORDS = {"inherited": "UNCOVERED_CELL", "named_fallback": "NAMED_FALLBACK", "cell_hit": "CELL_HIT", "stock": "STOCK_CELL", "opt_in": "OPT_IN"}
KEY_FIELDS = ("provider", "cc", "stack", "dtype", "shape", "bucket", "form", "word")
CONTEXT_AXES = ("engine", "mode", "card", "size")
INHERITED_SPLIT = ("inherited_size", "inherited_guard", "inherited_family")   # the summariser's split of 'inherited' (the true gaps) by the note's first word
SIBLING_NOTE = "inherited_stack(measured_on=%s)"                  # the note of a cell hit served on a sibling stack of the cell's cc (the timing column is the reference stack's)
SIBLING_COUNT = "sibling_stack"                                   # the summariser's count of such hits per row
EXACT_VOUCH_WORD = "exact_vouch_not_recorded"                     # the faces' refusal word for an exact row on a stack its bitwise vouch was not recorded on
GAP_KINDS = ("size", "guard", "family", "exact_vouch")            # --true-gaps: which dimension has no cell (size neighbour | non-size neighbour | no family) | an exact vouch owed
TOKEN_RE = re.compile(r"^(UNCOVERED_CELL|NAMED_FALLBACK|CELL_HIT|STOCK_CELL|OPT_IN):"
                      r"([a-z0-9_]+)\|cc([^| ]+)\|([^| ]+)\|([^| ]+)\|([^| ]+)\|([^| ]+)\|([^| ]+)"
                      r" word=(\S+)(?: refused=(\S+))? served=(\S+)(?: (?:nearest|cell)=(\S+))?(?: note=(\S+))?$")
SCHEMA = "opt_core.cell_census/1"
EXIT_TAG = "opt_core/cells"                                        # the report.register_exit_tally tag of the exit lines
EXIT_VERB = "CELLS"
EXACT_CLASSES = ("exact", "bitwise", "stock")                      # the numerics-class words an exact request may inherit from (nearest guard)

_LOCK = threading.Lock()
_ENTRIES: Dict[tuple, Dict[str, Any]] = {}
_CONTEXT: Dict[str, Any] = {}
_STATE: Dict[str, Any] = {"t0": None, "atexit": False, "seq": 0, "errors": 0, "written": None}


# ----------------------------------------------------------------------------------------------------------------------------- words
def clean(value: Any, limit: int = 96, keep_bar: bool = False) -> str:
    """One token-safe word: str(value) with whitespace -> '_' and (unless ``keep_bar``) '|' -> '/', capped at ``limit`` characters; None / '' -> '-'.
    Cell ids keep their '|' (they are the table's own keys, quoted verbatim as the token's last field)."""
    if value is None:
        return "-"
    s = str(value).strip()
    if not s:
        return "-"
    s = re.sub(r"\s+", "_", s)
    if not keep_bar:
        s = s.replace("|", "/")
    return s[:limit]


def bucket_word(n: Any, cell_id: Optional[str] = None, inside: bool = True) -> str:
    """The bucket segment: the ``N<=<size>`` part of ``cell_id`` when the call is inside that measured bucket, else ``N=<n>`` (the call's own size)."""
    if inside and cell_id:
        for part in str(cell_id).split("|"):
            if part.startswith("N<="):
                return part.split("+")[0]
    return "N=%s" % ("-" if n is None else clean(n, 24))


def _key_of(provider: str, key: Mapping[str, Any]) -> Dict[str, str]:
    out = {"provider": clean(provider, 32).lower()}
    for f in KEY_FIELDS[1:]:
        out[f] = clean(key.get(f) if isinstance(key, Mapping) else None, 96)
    if out["cc"].startswith("cc"):
        out["cc"] = out["cc"][2:] or "-"
    return out


def _token(e: Mapping[str, Any]) -> str:
    head = "%s:%s|cc%s|%s|%s|%s|%s|%s word=%s" % (TOKEN_WORDS[e["outcome"]], e["provider"], e["cc"], e["stack"], e["dtype"], e["shape"], e["bucket"], e["form"], e["word"])
    if e["outcome"] == "named_fallback":
        line = "%s refused=%s served=%s" % (head, e["refused"] or "-:-", e["served"])
    elif e["outcome"] == "inherited":
        line = "%s served=%s nearest=%s" % (head, e["served"], e["cell"] or "none")
    else:
        line = "%s served=%s cell=%s" % (head, e["served"], e["cell"] or "none")
    if e.get("note") and e["note"] != "-":
        line += " note=%s" % e["note"]
    return line


# --------------------------------------------------------------------------------------------------------------------------- recording
def set_context(**kw: Any) -> Dict[str, Any]:
    """Name the run (``engine=``, ``mode=``, ``card=``, ``size=``, any other word): merged into the current context; entries recorded afterwards
    carry a copy.  ``set_context(size=None)`` removes a word.  Returns the context now in force."""
    with _LOCK:
        for k, v in kw.items():
            if v is None:
                _CONTEXT.pop(k, None)
            else:
                _CONTEXT[str(k)] = v if isinstance(v, (int, float, bool)) else str(v)
        return dict(_CONTEXT)


def context() -> Dict[str, Any]:
    with _LOCK:
        return dict(_CONTEXT)


def _print_level() -> int:
    """2: every new key at decision time; 1: alert keys at decision time; 0: the exit lines only (unset); -1: nothing ('0')."""
    v = os.environ.get(ENV_PRINT, "")
    return 2 if v == "2" else (1 if v == "1" else (-1 if v == "0" else 0))


def _arm_atexit() -> None:
    """At the first record: the exit hook (the JSON dump when OPT_CORE_CELL_CENSUS names a path, then the CELLS line + alert tokens on stderr),
    registered through opt_core.report's exit tally so a forced exit prints it too; plain atexit when report cannot be imported."""
    if _STATE["atexit"]:
        return
    _STATE["atexit"] = True
    if _print_level() < 0:                                        # OPT_CORE_CELL_CENSUS_PRINT=0: no line anywhere; the dump is still written when a path is named
        if os.environ.get(ENV_FILE):
            atexit.register(write)
        return
    try:
        from . import report as _report
        _report.register_exit_tally(EXIT_TAG, _exit_text)
    except Exception:                                             # no package context (a copied module): the interpreter's own hook
        atexit.register(_atexit_print)


def record(provider: str, key: Mapping[str, Any], outcome: str, served_row: Any, cell_id: Optional[str] = None, note: str = "",
           refused: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Record ONE decided call class.  ``key``: dict with cc / stack / dtype / shape / bucket / form / word (missing -> '-'); ``outcome``: OUTCOMES;
    ``served_row``: the row that serves; ``cell_id``: the deciding cell (cell_hit / stock / opt_in) or the NEAREST measured cell whose row was
    inherited (inherited; None = no measured cell in the family); ``refused``: '<row>:<reason>' for named_fallback; ``note``: free words.
    The first record of a key stores the entry (and prints its token under OPT_CORE_CELL_CENSUS_PRINT); the same fact again is one dict lookup
    (counted in ``n``); a DIFFERENT fact for the same key (a row stepped aside after the first decision) replaces it, ``revised`` + 1, and prints
    again.  Returns the entry (a copy) or None when the arguments could not be recorded.  Never raises."""
    try:
        if outcome not in OUTCOMES:
            outcome = "opt_in" if outcome is None else clean(outcome, 24)
            if outcome not in OUTCOMES:
                _STATE["errors"] += 1
                return None
        k = _key_of(provider, key)
        kt = tuple(k[f] for f in KEY_FIELDS)
        served = clean(served_row, 64)
        cell = None if cell_id is None else clean(cell_id, 128, keep_bar=True)
        ref = None if refused is None else clean(refused, 128)
        nt = clean(note, 160) if note else ""
        with _LOCK:
            e = _ENTRIES.get(kt)
            if e is not None and e["outcome"] == outcome and e["served"] == served and e["cell"] == cell and e["refused"] == ref:
                e["n"] += 1
                return dict(e)
            now = time.time()
            if _STATE["t0"] is None:
                _STATE["t0"] = now
            _STATE["seq"] += 1
            if e is None:
                e = dict(k)
                e.update(outcome=outcome, served=served, cell=cell, refused=ref, note=nt, n=1, revised=0, seq=_STATE["seq"],
                         t=round(now - _STATE["t0"], 3), context=dict(_CONTEXT))
                _ENTRIES[kt] = e
            else:
                e.update(outcome=outcome, served=served, cell=cell, refused=ref, note=nt, n=e["n"] + 1, revised=e["revised"] + 1, seq=_STATE["seq"],
                         context=dict(_CONTEXT))
            e["token"] = _token(e)
            out = dict(e)
            _arm_atexit()
        lvl = _print_level()
        if lvl >= 2 or (lvl == 1 and outcome in ALERT_OUTCOMES):
            print(out["token"], file=sys.stderr, flush=True)
        return out
    except Exception:                                            # a census counts; it never gates or breaks a served call
        _STATE["errors"] = int(_STATE.get("errors", 0) or 0) + 1
        return None


# ------------------------------------------------------------------------------------------------------------------------------ nearest
class Nearest(tuple):
    """(cell_id, relation, why): ``relation`` = 'covered' (the smallest measured bucket at or above n holds the call: a cell hit) |
    'nearest_size' (no bucket holds n: the closest measured size of the same family serves -- inherited) | None (no legal cell: ``why`` names
    the guard that emptied the family -- cc | dtype | class | form | shape -- and the face refuses the row by name or serves its named default)."""
    __slots__ = ()

    def __new__(cls, cell_id, relation, why):
        return tuple.__new__(cls, (cell_id, relation, why))

    cell_id = property(lambda self: self[0])
    relation = property(lambda self: self[1])
    why = property(lambda self: self[2])


def _is_exact_class(cls: Any) -> bool:
    return str(cls or "").lower().startswith(EXACT_CLASSES)


def _cls_ok(query_cls: Any, cand_cls: Any) -> bool:
    if not _is_exact_class(query_cls):
        return True                                               # a tolerance-class (fast / big) request may inherit any measured cell of the family
    c = str(cand_cls or "").lower()
    return bool(c) and c.startswith(EXACT_CLASSES)                # exact inherits only from exact / bitwise / stock-class cells


def _vouched_on(c: Mapping[str, Any]) -> List[Any]:
    """The stack keys an exact cell's vouch holds on: ``vouched_on`` (list or one word), else the candidate's one ``stack`` word, else none."""
    v = c.get("vouched_on")
    if v is None:
        v = c.get("stack")
    if v is None:
        return []
    return list(v) if isinstance(v, (list, tuple, set, frozenset)) else [v]


def nearest(query: Mapping[str, Any], candidates: Iterable[Mapping[str, Any]]) -> Nearest:
    """The nearest-cell rule with its hard guards.  ``query``: dict(cc=, dtype=, shape=, form=, cls=, n=); ``candidates``: dicts with the same
    words + ``size`` (the bucket's upper size) + ``id`` (the table's cell key) [+ ``cls``: the numerics class that cell's winner holds].
    Guards, in order (a candidate failing one is never served): same ``cc`` (no cross-card launch cell), same ``dtype``, numerics class (an
    exact ``cls`` query keeps only exact / bitwise / stock-class candidates), VOUCH (an exact ``cls`` query keeps only candidates whose
    ``vouched_on`` stack keys -- else the one ``stack`` word they carry -- include the query's ``stack``: an exact vouch holds on the stack it was
    recorded on and nowhere else; an exact query naming no stack keeps none), same ``form``, same ``shape`` (width / heads / head_dim words).
    Among the survivors: the smallest ``size`` >= n -> ('covered'); else the closest size, the larger when equidistant -> ('nearest_size');
    no survivor -> (None, None, '<guard>:<query word>') naming the FIRST guard that removed the last candidates (``vouch:<stack>``: the face
    refuses the exact row BY NAME there and the tier falls to its floor row)."""
    try:
        n = int(query.get("n"))
    except (TypeError, ValueError):
        return Nearest(None, None, "n:unparsable")
    pool = [c for c in candidates if isinstance(c, Mapping) and c.get("id") is not None]
    if not pool:
        return Nearest(None, None, "family:empty")
    for guard in ("cc", "dtype", "cls", "vouch", "form", "shape"):
        if guard == "cls":
            kept = [c for c in pool if _cls_ok(query.get("cls"), c.get("cls"))]
        elif guard == "vouch":
            if not _is_exact_class(query.get("cls")):
                continue
            st = query.get("stack")
            want = clean(st, 96, keep_bar=True) if st not in (None, "") else None
            kept = [] if want is None else [c for c in pool if want in {clean(v, 96, keep_bar=True) for v in _vouched_on(c)}]
            if not kept:
                return Nearest(None, None, "vouch:%s" % (clean(st, 48) if want is not None else "stack_unknown"))
        else:
            want = clean(query.get(guard), 96, keep_bar=True)
            kept = [c for c in pool if clean(c.get(guard), 96, keep_bar=True) == want]
        if not kept:
            return Nearest(None, None, "%s:%s" % (guard, clean(query.get(guard), 48)))
        pool = kept
    sized = []
    for c in pool:
        try:
            sized.append((int(c.get("size")), c))
        except (TypeError, ValueError):
            continue
    if not sized:
        return Nearest(None, None, "size:none_measured")
    at_or_above = sorted((sz, str(c["id"])) for sz, c in sized if sz >= n)
    if at_or_above:
        return Nearest(at_or_above[0][1], "covered", "bucket:N<=%d" % at_or_above[0][0])
    best = sorted(((abs(sz - n), -sz, str(c["id"])) for sz, c in sized))[0]
    return Nearest(best[2], "nearest_size", "nearest:N<=%d" % -best[1])


# ------------------------------------------------------------------------------------------------------------------------------ readers
def table() -> List[Dict[str, Any]]:
    """Every entry (copies) in decision order."""
    with _LOCK:
        return [dict(e) for e in _ENTRIES.values()]                 # insertion order = the order the keys were first decided


def tokens(outcomes: Optional[Iterable[str]] = None) -> List[str]:
    """The token line of every entry (or of the entries whose outcome is in ``outcomes``), in decision order."""
    keep = None if outcomes is None else set(outcomes)
    return [e["token"] for e in table() if keep is None or e["outcome"] in keep]


def alerts() -> List[str]:
    """The UNCOVERED_CELL and NAMED_FALLBACK lines (what a run's whitelist has to name)."""
    return tokens(ALERT_OUTCOMES)


def reset() -> None:
    """Forget every entry and the context (a face's own selection memo is not touched: clear it through the face)."""
    with _LOCK:
        _ENTRIES.clear()
        _CONTEXT.clear()
        _STATE.update(t0=None, seq=0, written=None)


def _counts(entries: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    c = {o: 0 for o in OUTCOMES}
    for e in entries:
        c[e["outcome"]] = c.get(e["outcome"], 0) + 1
    return c


def dump(fmt: str = "json") -> str:
    """The census as text: 'json' (schema, pid, context, counts, tokens, entries) or 'tsv' (one header line + one line per entry)."""
    rows = table()
    if fmt == "tsv":
        cols = list(KEY_FIELDS) + ["outcome", "served", "cell", "refused", "note", "n", "revised"] + list(CONTEXT_AXES) + ["token"]
        lines = ["\t".join(cols)]
        for e in rows:
            ctx = e.get("context") or {}
            vals = [str(e.get(c, "")) if c not in CONTEXT_AXES else str(ctx.get(c, "")) for c in cols]
            lines.append("\t".join("" if v == "None" else v for v in vals))
        return "\n".join(lines) + "\n"
    if fmt != "json":
        raise ValueError("fmt must be 'json' or 'tsv' (got %r)" % (fmt,))
    doc = {"schema": SCHEMA, "pid": os.getpid(), "context": context(), "counts": _counts(rows), "errors": int(_STATE.get("errors", 0) or 0),
           "tokens": [e["token"] for e in rows], "alerts": [e["token"] for e in rows if e["outcome"] in ALERT_OUTCOMES], "entries": rows}
    return json.dumps(doc, indent=1, sort_keys=False, default=str) + "\n"


def cell_census_dump(fmt: str = "json") -> str:
    """``opt_core.cell_census_dump()`` (the package's lazy export): :func:`dump`."""
    return dump(fmt)


def _target_path(value: str) -> str:
    if value.endswith(("/", os.sep)) or os.path.isdir(value):
        return os.path.join(value, "cell_census.%d.json" % os.getpid())
    return value


def write(path: Optional[str] = None) -> Optional[str]:
    """Write the JSON dump to ``path`` (default: the OPT_CORE_CELL_CENSUS value; a directory receives cell_census.<pid>.json).  Returns the path
    written, None when there is no path.  Never raises (a failed write is reported on stderr)."""
    value = path or os.environ.get(ENV_FILE)
    if not value:
        return None
    try:
        target = _target_path(str(value))
        d = os.path.dirname(os.path.abspath(target))
        if d and not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
        tmp = target + ".tmp%d" % os.getpid()
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(dump("json"))
        os.replace(tmp, target)
        _STATE["written"] = target
        return target
    except Exception as e:                                        # the exit-time writer: report, never raise out of atexit
        print("opt_core.cell_census: could not write %r: %s: %s" % (value, type(e).__name__, e), file=sys.stderr)
        return None


EXIT_MAX_TOKENS = 400                                             # the exit summary prints at most this many alert tokens (a run decides tens; a test session thousands)


def exit_lines(max_tokens: Optional[int] = EXIT_MAX_TOKENS) -> List[str]:
    """The exit summary: the ``[opt_core] CELLS ...`` line, then the alert tokens (UNCOVERED_CELL / NAMED_FALLBACK) in decision order -- at
    most ``max_tokens`` of them, the rest NAMED as one ``[opt_core] CELLS ... more=<n>`` line (the dump holds them all)."""
    rows = table()
    c = _counts(rows)
    ctx = context()
    al = [e["token"] for e in rows if e["outcome"] in ALERT_OUTCOMES]
    head = "[opt_core] %s pid=%d keys=%d %s alerts=%d dump=%s" % (
        EXIT_VERB, os.getpid(), len(rows), " ".join("%s=%d" % (o, c[o]) for o in OUTCOMES), len(al), _STATE.get("written") or "none")
    if ctx:
        head += " context=" + ",".join("%s:%s" % (k, clean(v, 32)) for k, v in ctx.items())
    if max_tokens is not None and len(al) > int(max_tokens):
        rest = len(al) - int(max_tokens)
        al = al[: int(max_tokens)] + ["[opt_core] %s pid=%d more=%d (every token: the dump / opt_core.cell_census.alerts())" % (EXIT_VERB, os.getpid(), rest)]
    return [head] + al


def _exit_text() -> str:
    """The exit hook body: write the dump (when a path is named), then the lines."""
    write()
    return "\n".join(exit_lines())


def _atexit_print() -> None:
    try:
        text = _exit_text()
        if text:
            sys.stderr.write(text + "\n")
            sys.stderr.flush()
    except Exception:                                             # interpreter exit: never raise
        pass


# --------------------------------------------------------------------------------------------------------------------------- summariser
def sibling_note(measured_on: Any, extra: str = "") -> str:
    """The note of a cell hit whose timing column is a sibling stack's: ``inherited_stack(measured_on=<stack>)`` (+ ``;<extra>``)."""
    return (SIBLING_NOTE % clean(measured_on, 96)) + ((";" + extra) if extra else "")


_LEGACY_STACK_RE = re.compile(r"^stack\(([^)]*)\)(.*)$")             # older dumps recorded a sibling-stack hit as inherited with note stack(<ref>)


def normalize_entry(e: Mapping[str, Any]) -> Dict[str, Any]:
    """One dump entry in the current classification: an ``inherited`` entry whose note is the legacy ``stack(<reference stack>)`` (the cell
    exists on a same-cc reference column; only the caller's stack column was missing) becomes ``cell_hit`` with
    ``note=inherited_stack(measured_on=<reference stack>)`` and its token re-rendered; everything else is returned unchanged (a copy)."""
    d = dict(e)
    if d.get("outcome") == "inherited" and d.get("cell"):
        m = _LEGACY_STACK_RE.match(str(d.get("note") or ""))
        if m:
            d["outcome"] = "cell_hit"
            d["note"] = sibling_note(m.group(1), m.group(2).lstrip(";"))
            d["legacy_reclassified"] = "inherited_stack"
            d["token"] = _token(dict({f: d.get(f, "-") for f in KEY_FIELDS}, outcome="cell_hit", served=d.get("served"), cell=d.get("cell"),
                                     refused=None, note=d["note"]))
    return d


def gap_kind(e: Mapping[str, Any]) -> Optional[str]:
    """Which true gap an entry is (None: not a gap): ``family`` (no measured cell of the family on any stack: nearest=none), ``guard`` (a
    non-size neighbour served: note guard:<dim>), ``size`` (a size neighbour served: beyond every measured bucket), ``exact_vouch`` (a named
    fallback whose refusal word is the faces' exact-vouch word: the exact row's bitwise vouch is owed on this stack)."""
    o, note = e.get("outcome"), str(e.get("note") or "")
    if o == "inherited":
        if not e.get("cell") or note.startswith("family:"):
            return "family"
        if note.startswith("guard:"):
            return "guard"
        if _LEGACY_STACK_RE.match(note):
            return None                                           # legacy sibling-stack record: a cell hit, not a gap
        return "size"
    if o == "named_fallback" and EXACT_VOUCH_WORD in str(e.get("refused") or ""):
        return "exact_vouch"
    return None


_OUTCOME_OF_WORD = {v: k for k, v in TOKEN_WORDS.items()}
_TOKEN_ANYWHERE = re.compile(r"(UNCOVERED_CELL|NAMED_FALLBACK|CELL_HIT|STOCK_CELL|OPT_IN):[a-z0-9_]+\|cc")
_CONTEXT_IN_HEAD = re.compile(r"\[opt_core\] %s .*?context=(\S+)" % EXIT_VERB)


def parse_token(line: str) -> Optional[Dict[str, Any]]:
    """The entry a token line states (the inverse of the token grammar), else None.  Anything before the token word on the line (a
    launcher's timestamp column, a rank prefix) is skipped."""
    m0 = _TOKEN_ANYWHERE.search(line or "")
    if m0 is None:
        return None
    m = TOKEN_RE.match(line[m0.start():].rstrip())
    if m is None:
        return None
    tw, provider, cc, stack, dtype, shape, bucket, form, word, refused, served, cell, note = m.groups()
    outcome = _OUTCOME_OF_WORD[tw]
    return dict(provider=provider, cc=cc, stack=stack, dtype=dtype, shape=shape, bucket=bucket, form=form, word=word, outcome=outcome, served=served,
                cell=None if cell in (None, "none") else cell, refused=refused, note=note or "", token=line[m0.start():].rstrip(), n=1, revised=0, context={})


def parse_log(text: str) -> Dict[str, Any]:
    """A census document from a TRANSCRIPT (the exit lines / decision-time lines a run logged on stderr): every token line becomes an entry
    (de-duplicated: ranks print the same key), each carrying the context of the nearest preceding ``[opt_core] CELLS ... context=`` head line."""
    entries: Dict[str, Dict[str, Any]] = {}
    ctx: Dict[str, Any] = {}
    first_ctx: Dict[str, Any] = {}
    for line in (text or "").splitlines():
        h = _CONTEXT_IN_HEAD.search(line)
        if h is not None:
            ctx = {}
            for kv in h.group(1).split(","):
                k, _, v = kv.partition(":")
                if k in CONTEXT_AXES and v:
                    ctx[k] = v
            first_ctx = first_ctx or dict(ctx)
            continue
        e = parse_token(line)
        if e is None:
            continue
        e["context"] = dict(ctx)
        prior = entries.get(e["token"])
        if prior is not None:
            prior["n"] += 1
        else:
            entries[e["token"]] = e
    return {"schema": SCHEMA, "source": "log", "context": first_ctx, "entries": list(entries.values())}


def _path_item(item: Any) -> Tuple[str, Dict[str, Any]]:
    """``paths`` items of :func:`summarize` / :func:`true_gaps`: a path, or ``(path, context)`` naming the engine / mode / card / size of a dump
    or transcript that carries no context of its own (a run that did not call :func:`set_context`)."""
    if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[1], Mapping):
        return str(item[0]), {k: v for k, v in item[1].items() if k in CONTEXT_AXES and v is not None}
    return str(item), {}


def _load(path: str) -> Dict[str, Any]:
    """A dump (JSON: the OPT_CORE_CELL_CENSUS file, or a bare entries list) or a transcript holding token lines (see :func:`parse_log`);
    entries come back in the current classification (:func:`normalize_entry`)."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    try:
        doc = json.loads(text)
    except ValueError:
        doc = parse_log(text)
    if isinstance(doc, list):                                     # a bare entries list is accepted
        doc = {"entries": doc, "context": {}}
    doc["entries"] = [normalize_entry(e) for e in (doc.get("entries") or []) if isinstance(e, Mapping)]
    return doc


def summarize(paths: Iterable[str], engine: Optional[str] = None, overrides: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    """Merge dumps into the coverage table: one row per (engine, mode, card, size, provider) with the count of keys per outcome, the distinct
    served rows, and the alert tokens.  The axes come from each ENTRY's context (else the dump's context); ``engine`` / ``overrides`` FILL IN an
    axis the dump does not name (else '-')."""
    ov = dict(overrides or {})
    if engine is not None:
        ov["engine"] = engine
    acc: Dict[tuple, Dict[str, Any]] = {}
    for item in paths:
        p, pctx = _path_item(item)
        doc = _load(p)
        dctx = dict(pctx)
        dctx.update(doc.get("context") or {})
        for e in doc.get("entries") or []:
            ectx = dict(dctx)
            ectx.update(e.get("context") or {})
            axes = tuple(clean(ectx.get(a) if ectx.get(a) is not None else ov.get(a), 48) for a in CONTEXT_AXES)   # the entry's context, else the fill-in
            k = axes + (e.get("provider", "-"),)
            row = acc.get(k)
            if row is None:
                row = acc[k] = dict(zip(CONTEXT_AXES + ("provider",), k))
                row.update({o: 0 for o in OUTCOMES})
                row.update({o: 0 for o in INHERITED_SPLIT})
                row.update({SIBLING_COUNT: 0})
                row.update(keys=0, served=[], alerts=[], sources=[])
            o = e.get("outcome")
            if o in OUTCOMES:
                row[o] += 1
            if o == "inherited":                                  # which dimension had no cell: size | guard | family (the true gaps)
                row["inherited_" + (gap_kind(e) or "size")] += 1
            elif o in ("cell_hit", "stock") and str(e.get("note") or "").startswith(SIBLING_NOTE.split("%")[0]):
                row[SIBLING_COUNT] += 1                            # a hit whose timing column is a sibling stack's (measured_on=...)
            row["keys"] += 1
            s = e.get("served")
            if s and s not in row["served"]:
                row["served"].append(s)
            if o in ALERT_OUTCOMES:
                tok = e.get("token") or _token(dict({f: e.get(f, "-") for f in KEY_FIELDS}, outcome=o, served=s, cell=e.get("cell"), refused=e.get("refused"), note=""))
                if tok not in row["alerts"]:
                    row["alerts"].append(tok)
            if p not in row["sources"]:
                row["sources"].append(p)
    out = [acc[k] for k in sorted(acc, key=lambda k: tuple(str(x) for x in k))]
    for row in out:
        row["verdict"] = ("named_fallback" if row["named_fallback"] else "inherited" if row["inherited"] else
                          "cell_hit" if row["cell_hit"] else "stock" if row["stock"] else "opt_in" if row["opt_in"] else "-")
    return out


def _summary_text(rows: List[Dict[str, Any]], fmt: str = "tsv") -> str:
    if fmt == "json":
        return json.dumps(rows, indent=1) + "\n"
    cols = list(CONTEXT_AXES) + ["provider", "verdict"] + list(OUTCOMES) + list(INHERITED_SPLIT) + [SIBLING_COUNT, "keys", "served", "alerts"]
    lines = ["\t".join(cols)]
    for r in rows:
        lines.append("\t".join(str(r[c]) if c not in ("served", "alerts") else (",".join(r[c]) if c == "served" else " ;; ".join(r[c])) for c in cols))
    return "\n".join(lines) + "\n"


def true_gaps(paths: Iterable[str], engine: Optional[str] = None, overrides: Optional[Mapping[str, Any]] = None,
              kinds: Iterable[str] = GAP_KINDS) -> List[Dict[str, Any]]:
    """The TRUE no-cell keys of the dumps, by name: one row per distinct census key (provider + the seven segments + word) whose entry is a gap
    (:func:`gap_kind`: ``size`` | ``guard`` | ``family`` -- UNCOVERED_CELL after the sibling-stack hits are set aside -- and ``exact_vouch``, the
    NAMED_FALLBACK entries owing an exact vouch on their stack), with the served row, the nearest cell, the note / refusal, and the union of the
    contexts (engines, modes, cards, sizes) and dumps it was seen in.  Sorted by kind, provider, key."""
    ov = dict(overrides or {})
    if engine is not None:
        ov["engine"] = engine
    want = set(kinds)
    acc: Dict[tuple, Dict[str, Any]] = {}
    for item in paths:
        p, pctx = _path_item(item)
        doc = _load(p)
        dctx = dict(pctx)
        dctx.update(doc.get("context") or {})
        for e in doc["entries"]:
            kind = gap_kind(e)
            if kind is None or kind not in want:
                continue
            k = (kind, e.get("provider", "-")) + tuple(e.get(f, "-") for f in KEY_FIELDS[1:])
            row = acc.get(k)
            if row is None:
                row = acc[k] = dict(kind=kind, provider=e.get("provider", "-"))
                row.update({f: e.get(f, "-") for f in KEY_FIELDS[1:]})
                row.update(served=e.get("served"), nearest=e.get("cell"), note=e.get("note") or "", refused=e.get("refused") or "",
                           token=e.get("token") or "", n=0, dumps=[])
                row.update({a + "s": [] for a in CONTEXT_AXES})
            row["n"] += int(e.get("n") or 1)
            ectx = dict(dctx)
            ectx.update(e.get("context") or {})
            for a in CONTEXT_AXES:
                v = ectx.get(a) if ectx.get(a) is not None else ov.get(a)
                if v is not None and clean(v, 48) not in row[a + "s"]:
                    row[a + "s"].append(clean(v, 48))
            if p not in row["dumps"]:
                row["dumps"].append(p)
    order = {k: i for i, k in enumerate(GAP_KINDS)}
    return [acc[k] for k in sorted(acc, key=lambda k: (order.get(k[0], 99),) + tuple(str(x) for x in k[1:]))]


def _gaps_text(rows: List[Dict[str, Any]], fmt: str = "tsv") -> str:
    if fmt == "json":
        return json.dumps(rows, indent=1) + "\n"
    lines = ["# true no-cell keys: %d (%s)" % (len(rows), ", ".join("%s=%d" % (k, sum(1 for r in rows if r["kind"] == k)) for k in GAP_KINDS) or "none")]
    cols = ["kind", "provider"] + list(KEY_FIELDS[1:]) + ["served", "nearest", "note_or_refused", "engines", "modes", "cards", "sizes", "n", "dumps"]
    lines.append("\t".join(cols))
    for r in rows:
        vals = dict(r, note_or_refused=r["refused"] or r["note"] or "-", nearest=r["nearest"] or "none", dumps=len(r["dumps"]),
                    engines=",".join(r["engines"]) or "-", modes=",".join(r["modes"]) or "-", cards=",".join(r["cards"]) or "-", sizes=",".join(r["sizes"]) or "-")
        lines.append("\t".join(str(vals[c]) for c in cols))
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m opt_core.cell_census", description="Merge cell-census dumps into the coverage table.")
    ap.add_argument("--summarize", nargs="+", metavar="DUMP.json", help="census dumps (OPT_CORE_CELL_CENSUS files) to merge")
    ap.add_argument("--tokens", nargs="+", metavar="DUMP.json", help="print the UNCOVERED_CELL / NAMED_FALLBACK tokens of these dumps")
    ap.add_argument("--true-gaps", nargs="+", metavar="DUMP", dest="true_gaps",
                    help="list the TRUE no-cell keys by name (size / guard / family gaps + exact vouches owed; sibling-stack hits set aside); DUMP = a census "
                         "JSON dump or a run transcript holding the token lines (every reader here accepts both)")
    ap.add_argument("--engine", default=None, help="the engine word for dumps whose context names none (mode / card / size likewise)")
    ap.add_argument("--mode", default=None); ap.add_argument("--card", default=None); ap.add_argument("--size", default=None)
    ap.add_argument("--fmt", default="tsv", choices=("tsv", "json"))
    a = ap.parse_args(argv)
    if not a.summarize and not a.tokens and not a.true_gaps:
        ap.error("give --summarize DUMP... | --tokens DUMP... | --true-gaps DUMP...")
    ov = {k: v for k, v in (("mode", a.mode), ("card", a.card), ("size", a.size)) if v is not None}
    if a.true_gaps:
        sys.stdout.write(_gaps_text(true_gaps(a.true_gaps, engine=a.engine, overrides=ov), a.fmt))
        if not a.summarize and not a.tokens:
            return 0
    if a.tokens:
        seen = []
        for p in a.tokens:
            for t in (_load(p).get("alerts") or [e.get("token") for e in _load(p).get("entries", []) if e.get("outcome") in ALERT_OUTCOMES]):
                if t and t not in seen:
                    seen.append(t)
        sys.stdout.write("\n".join(seen) + ("\n" if seen else ""))
        if not a.summarize:
            return 0
    rows = summarize(a.summarize, engine=a.engine, overrides=ov)
    sys.stdout.write(_summary_text(rows, a.fmt))
    return 0


if __name__ == "__main__":                                        # pragma: no cover
    sys.exit(main())
