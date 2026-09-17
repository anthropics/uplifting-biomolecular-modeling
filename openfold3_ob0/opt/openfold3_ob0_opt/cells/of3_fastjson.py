"""The `fastjson` lever (exact class, in BYTES): the engine's confidence-JSON writer renders the same text through a faster encoder.

The engine's output writer (`<engine>.core.runners.writer`, `OF3OutputWriter.write_confidence_scores`) renders, per diffusion sample, the
aggregated confidences and the FULL confidences (`{"plddt": [N_atom], "pde": [N_tok, N_tok], "pae": [N_tok, N_tok]}`) with
`json.dumps(obj, indent=4, cls=NumpyEncoder)`. With `indent` set, CPython's `json` takes its pure-Python encoder (`json.encoder._make_iterencode`:
the C encoder is used only when `indent is None`), so every one of the ~2·N_tok² + N_atom numbers of every sample passes through Python-level
generator frames on the predicting process's main thread, after the forward: the largest host cost of an item at these sizes, super-linear in N.

This module installs, on the writer module's encoder class (the `cls=` the writer passes), a subclass whose `encode()` produces THE SAME
CHARACTERS by a different route: the layout `indent` asks for (newline + indent per level, `item_separator` / `key_separator` as the encoder
holds them, key order and key coercion as `json` does them, `sort_keys` / `skipkeys` / `ensure_ascii` / `allow_nan` honoured) is built around
rows joined at C level — `str.join` over `map(float.__repr__, row)` — for the 1-D / 2-D numeric arrays, whose Python values are the ENGINE'S OWN
conversion (`self.default(array)`: `tolist()` at 0.4.x, a rounded `tolist()` at 0.5.x), every number formatted by the function `json` formats it
with (`float.__repr__`; `NaN` / `Infinity` / `-Infinity` spelled as `json` spells them, per row, only where the array holds a non-finite value;
`int.__repr__` for integer arrays); every other value (dicts, lists, tuples, str, int, float, bool, None, numpy scalars through `default()`) goes
through a line-by-line port of `_make_iterencode`. A structure the port does not reproduce, or any exception on the fast route, is answered by
the stock encoder for that call — counted (`fallback=<n>`, `fallback_by=`), never silent; `indent=None` calls and plain strings are the stock
encoder's own (already C). At install the patched class encodes a probe structure (nested dicts, numpy scalars, a 2-D float32 block with
non-finite values, an integer array, unicode, empty containers) and is kept only if its text equals the stock class's byte for byte
(else refused by name: `reason=selftest_mismatch`); a switch word it does not know is refused by name too (`reason=bad_word:<VAR>`), never a crash.

Numerics: none — the lever touches no tensor; class EXACT in bytes (every output file identical to the unpatched writer's). It does not move the
model forward; it moves the item's wall time after the forward.
Switches (the kit adapter's `configure(ENV=…, ENV_ROUTE=…)`): `<KIT>_FASTJSON=1` arms it; `<KIT>_FASTJSON_ROUTE=fast|stock` (default `fast`;
`stock` keeps the class installed and answers every call with the stock encoder — the A/B arm, counted `stock=<n>`).
Exit line: `<PREFIX> LEVER name=fastjson state=on route=fast patched=1 served=<calls> fallback=<n> stock=<n> arrays=<n> bytes=<chars> s=<seconds>`.
Engines: the OF3 code family (0.4.x, 0.5.x: the same writer call form); the kits' `cells/fastjson.py` bind it (`M_WRITER`, `ENCODER_ATTR`).
"""
import atexit
import sys
import threading
import time
from json.encoder import INFINITY, encode_basestring, encode_basestring_ascii
from typing import Any, Dict, Optional

CONFIGURABLE = ("PREFIX", "ENV", "ENV_ROUTE", "M_WRITER", "ENCODER_ATTR")
PREFIX = "[of3-opt/fastjson]"
ENV = ""                                   # the arming switch (<KIT>_FASTJSON)
ENV_ROUTE = ""                             # the route knob (<KIT>_FASTJSON_ROUTE)
M_WRITER = ""                              # the engine's writer module (openfold3.core.runners.writer)
ENCODER_ATTR = "NumpyEncoder"              # the encoder class the writer passes as cls=
VALUES = ("1",)
ROUTES = ("fast", "stock")
MARK = "_of3opt_fastjson"


def configure(**kw) -> None:
    for k, v in kw.items():
        if k not in CONFIGURABLE:
            raise KeyError(f"{__name__}.configure: unknown setting {k!r} (known: {', '.join(CONFIGURABLE)})")
        globals()[k] = v


STATE: Dict[str, Any] = {"installed": False, "state": "off", "reason": "", "route": "fast", "patched": 0, "served": 0, "stock": 0, "fallback": 0,
                         "fallback_by": {}, "arrays": 0, "bytes": 0, "seconds": 0.0, "selftest": "-"}
_LOCK = threading.Lock()
_ATEXIT = {"registered": False}


def _log(msg: str) -> None:
    sys.stderr.write(f"{PREFIX} {msg}\n")


def _count(d: dict, k: str, n: int = 1) -> None:
    d[k] = d.get(k, 0) + n


def requested(environ=None) -> bool:
    import os
    environ = os.environ if environ is None else environ
    if not ENV:
        return False
    v = (environ.get(ENV) or "").strip()
    if not v:
        return False
    if v not in VALUES:
        raise ValueError(f"{ENV}={v!r} is not one of {'|'.join(VALUES)}")
    return True


def route(environ=None) -> str:
    import os
    environ = os.environ if environ is None else environ
    v = ((environ.get(ENV_ROUTE) if ENV_ROUTE else "") or "").strip() or "fast"
    if v not in ROUTES:
        raise ValueError(f"{ENV_ROUTE}={v!r} is not one of {'|'.join(ROUTES)}")
    return v


def serving() -> bool:
    return bool(STATE["installed"]) and STATE["state"] == "on"


def census_line() -> str:
    fb = ",".join("%s:%d" % kv for kv in sorted(STATE["fallback_by"].items()))
    return (f"{PREFIX} LEVER name=fastjson state={STATE['state']}" + (f" reason={STATE['reason']}" if STATE["reason"] else "") +
            f" route={STATE['route']} patched={STATE['patched']} served={STATE['served']} fallback={STATE['fallback']} stock={STATE['stock']}"
            f" arrays={STATE['arrays']} bytes={STATE['bytes']} s={STATE['seconds']:.2f}" + (f" fallback_by={fb}" if fb else ""))


class _Generic(Exception):
    """The fast route does not reproduce this call: the stock encoder answers it (counted)."""


def encode_like(enc, o, stats: Optional[dict] = None) -> str:
    """The text `type(enc).__mro__[1].encode(enc, o)` — the stock `json.JSONEncoder.encode` with `enc`'s own settings and `enc.default` —
    produces for an `indent` that is not None: a port of `json.encoder._make_iterencode` (CPython 3.9–3.13) emitting into one list, with
    numpy arrays of 1 or 2 dimensions rendered row by row at C level. Raises `_Generic` (or lets an exception through) for anything it does not
    reproduce; the caller answers those with the stock encoder."""
    indent = enc.indent
    if indent is None:
        raise _Generic("indent_none")
    if not isinstance(indent, str):
        indent = " " * indent
    markers: Optional[dict] = {} if enc.check_circular else None
    _encoder = encode_basestring_ascii if enc.ensure_ascii else encode_basestring
    item_sep, key_sep = enc.item_separator, enc.key_separator
    if not isinstance(item_sep, str) or not isinstance(key_sep, str):
        raise _Generic("separators")
    sort_keys, skipkeys, allow_nan, default = enc.sort_keys, enc.skipkeys, enc.allow_nan, enc.default
    _frepr, _irepr = float.__repr__, int.__repr__
    np = sys.modules.get("numpy")
    ndarray = np.ndarray if np is not None else None
    chunks = []
    emit = chunks.append

    def floatstr(x):
        if x != x:
            text = "NaN"
        elif x == INFINITY:
            text = "Infinity"
        elif x == -INFINITY:
            text = "-Infinity"
        else:
            return _frepr(x)
        if not allow_nan:
            raise _Generic("allow_nan")                       # the stock encoder raises its own ValueError for this call
        return text

    def array_text(a, level):
        """The text `_iterencode_list(default(a), level)` yields for a 1-D / 2-D numeric array, or _Generic."""
        kind = a.dtype.kind
        if a.ndim not in (1, 2) or a.size == 0 or kind not in "fiu":
            raise _Generic("shape_or_kind")
        bad = None
        if kind == "f":
            fin = np.isfinite(a)
            if not bool(fin.all()):
                bad = (~fin) if a.ndim == 1 else (~fin.all(axis=1))
        lst = default(a)                                          # the engine's own conversion of the array (tolist / rounded tolist)
        if not isinstance(lst, list) or len(lst) != a.shape[0]:
            return None, lst
        rep = _frepr if kind == "f" else _irepr
        nl1 = "\n" + indent * (level + 1)
        try:
            if a.ndim == 1:
                body = (item_sep + nl1).join(map(floatstr, lst) if bad is not None and bool(bad.any()) else map(rep, lst))
                return "[" + nl1 + body + "\n" + indent * level + "]", lst
            nl2 = "\n" + indent * (level + 2)
            sep_e = item_sep + nl2
            rows = []
            for i, row in enumerate(lst):
                if not isinstance(row, list) or len(row) != a.shape[1]:
                    return None, lst
                rows.append("[" + nl2 + (sep_e.join(map(floatstr, row)) if bad is not None and bool(bad[i]) else sep_e.join(map(rep, row))) + nl1 + "]")
            return "[" + nl1 + (item_sep + nl1).join(rows) + "\n" + indent * level + "]", lst
        except TypeError:                                         # an element is not the kind's Python type (float.__repr__ / int.__repr__ refuse it): the port renders the list
            return None, lst

    def enc_list(lst, level):
        if not lst:
            emit("[]")
            return
        if markers is not None:
            mid = id(lst)
            if mid in markers:
                raise _Generic("circular")
            markers[mid] = lst
        if indent is not None:
            level += 1
            nl = "\n" + indent * level
            sep = item_sep + nl
            emit("[" + nl)
        else:
            nl = None
            sep = item_sep
            emit("[")
        first = True
        for v in lst:
            if first:
                first = False
            else:
                emit(sep)
            enc_value(v, level)
        if nl is not None:
            level -= 1
            emit("\n" + indent * level)
        emit("]")
        if markers is not None:
            del markers[mid]

    def enc_dict(dct, level):
        if not dct:
            emit("{}")
            return
        if markers is not None:
            mid = id(dct)
            if mid in markers:
                raise _Generic("circular")
            markers[mid] = dct
        emit("{")
        if indent is not None:
            level += 1
            nl = "\n" + indent * level
            isep = item_sep + nl
            emit(nl)
        else:
            nl = None
            isep = item_sep
        first = True
        items = sorted(dct.items()) if sort_keys else dct.items()
        for key, value in items:
            if isinstance(key, str):
                pass
            elif isinstance(key, float):
                key = floatstr(key)
            elif key is True:
                key = "true"
            elif key is False:
                key = "false"
            elif key is None:
                key = "null"
            elif isinstance(key, int):
                key = _irepr(key)
            elif skipkeys:
                continue
            else:
                raise _Generic("key_type")                    # the stock encoder raises its own TypeError for this call
            if first:
                first = False
            else:
                emit(isep)
            emit(_encoder(key))
            emit(key_sep)
            enc_value(value, level)
        if nl is not None:
            level -= 1
            emit("\n" + indent * level)
        emit("}")
        if markers is not None:
            del markers[mid]

    def enc_value(o, level):
        if isinstance(o, str):
            emit(_encoder(o))
        elif o is None:
            emit("null")
        elif o is True:
            emit("true")
        elif o is False:
            emit("false")
        elif isinstance(o, int):
            emit(_irepr(o))
        elif isinstance(o, float):
            emit(floatstr(o))
        elif isinstance(o, (list, tuple)):
            enc_list(o, level)
        elif isinstance(o, dict):
            enc_dict(o, level)
        else:
            if markers is not None:
                mid = id(o)
                if mid in markers:
                    raise _Generic("circular")
                markers[mid] = o
            if ndarray is not None and isinstance(o, ndarray):
                try:
                    text, converted = array_text(o, level)
                except _Generic:
                    text, converted = None, default(o)
                if text is not None:
                    emit(text)
                    if stats is not None:
                        _count(stats, "arrays")
                else:
                    enc_value(converted, level)
            else:
                enc_value(default(o), level)
            if markers is not None:
                del markers[mid]

    enc_value(o, 0)
    return "".join(chunks)


def make_encoder_class(base):
    """A subclass of the writer's encoder class whose encode() takes the fast route for indented calls (route=fast), the stock route for
    everything else; the class carries MARK so a patched module is recognised."""

    class FastJSONEncoder(base):
        def encode(self, o):
            if isinstance(o, str) or self.indent is None:
                return base.encode(self, o)                       # the stock encoder's own fast cases (C encoder / plain string)
            if STATE["route"] != "fast":
                with _LOCK:
                    STATE["stock"] += 1
                return base.encode(self, o)
            t0 = time.perf_counter()
            stats: Dict[str, int] = {}
            try:
                text = encode_like(self, o, stats)
                key = None
            except Exception as e:  # noqa: BLE001 — anything the fast route does not reproduce: the stock encoder answers (and raises what it raises)
                key = (str(e) or "generic") if isinstance(e, _Generic) else type(e).__name__
                text = base.encode(self, o)
            dt = time.perf_counter() - t0
            with _LOCK:
                if key is None:
                    STATE["served"] += 1
                else:
                    STATE["fallback"] += 1
                    _count(STATE["fallback_by"], key)
                STATE["arrays"] += stats.get("arrays", 0)
                STATE["bytes"] += len(text)
                STATE["seconds"] += dt
            return text

    FastJSONEncoder.__name__ = FastJSONEncoder.__qualname__ = f"{base.__name__}_fastjson"
    setattr(FastJSONEncoder, MARK, base)
    return FastJSONEncoder


def selftest(patched, base) -> Optional[str]:
    """None when the patched class renders the probe structure exactly as the base class does (indent=4 and indent=2, sort_keys both ways);
    else a short reason. The probe holds what the writer renders (numpy float32 scalars and 1-D / 2-D float32 blocks, nested dicts keyed by
    chain ids) plus the port's edge cases (non-finite values, integer and boolean arrays, an empty row dimension, unicode, empty containers,
    tuples, bools, None, big and negative ints)."""
    import json
    np = sys.modules.get("numpy")
    if np is None:
        try:
            import numpy as np  # noqa: F811
        except Exception:  # noqa: BLE001
            return "numpy_absent"
    f32 = np.arange(12, dtype=np.float32).reshape(3, 4) / np.float32(7.0)
    f32[1, 2] = np.nan
    f32[2, 0] = np.inf
    f32[2, 3] = -np.inf
    probe = {
        "avg_plddt": np.float32(87.125), "gpde": np.float64(0.4567891234), "ptm": np.array(0.83, dtype=np.float32), "iptm": 0.1,
        "chain_ptm": {"A": np.float32(0.5), "B": np.float32(1e-8)}, "chain_pair_iptm": {"A": {"B": np.float32(1e16)}, "B": {"A": np.float32(-0.0)}},
        "plddt": (np.arange(5, dtype=np.float32) * np.float32(10.03)), "pde": f32, "pae": f32.astype(np.float64) * 1e-7,
        "ints": np.arange(6, dtype=np.int64).reshape(2, 3) - 3, "u8": np.arange(3, dtype=np.uint8), "flags": np.array([True, False]),
        "half": np.array([0.1, 65504.0], dtype=np.float16), "empty_rows": np.zeros((2, 0), dtype=np.float32), "empty": [], "none": None,
        "t": True, "f": False, "big": -123456789012345678901234567890, "s": "ch\u00e2in \u2192 \"q\"\n\t\\", "nested": [[1, 2.5, [None, "x"]], (), {}],
        "d3": np.zeros((1, 2, 2), dtype=np.float32), 7: "int key", 2.5: "float key", None: "none key", True: "bool key",
    }
    keep = {k: (dict(STATE[k]) if isinstance(STATE[k], dict) else STATE[k]) for k in ("served", "stock", "fallback", "fallback_by", "arrays", "bytes", "seconds", "route")}
    STATE["route"] = "fast"
    try:
        for kw in ({"indent": 4}, {"indent": 2, "sort_keys": False, "ensure_ascii": False}, {"indent": "\t"}, {"indent": 4, "separators": (", ", ":  ")}):
            want = json.dumps(probe, cls=base, **kw)
            got = json.dumps(probe, cls=patched, **kw)
            if want != got:
                return "selftest_mismatch"
        if STATE["fallback"] != keep["fallback"] or STATE["served"] != keep["served"] + 4:
            return "selftest_fallback"
        sortable = {"b": 1, "a": {"d": f32, "c": 2}}
        if json.dumps(sortable, cls=base, indent=4, sort_keys=True) != json.dumps(sortable, cls=patched, indent=4, sort_keys=True):
            return "selftest_mismatch_sorted"
    except Exception as e:  # noqa: BLE001
        return f"selftest_error:{type(e).__name__}"
    finally:
        with _LOCK:                                                # the probe's own calls are not the run's census
            STATE.update(keep)
    return None


def patch_writer_module(mod) -> bool:
    """Idempotent: replace `mod.<ENCODER_ATTR>` by the fast subclass when it is the stock class; refuse by name when the attribute is absent
    or the self-test does not hold. Returns True when the module is (now or already) patched."""
    base = getattr(mod, ENCODER_ATTR, None)
    if base is None:
        STATE.update(state="refused", reason=f"no_encoder:{M_WRITER}.{ENCODER_ATTR}")
        _log(f"REFUSED: {M_WRITER}.{ENCODER_ATTR} not found — the engine's writer runs as it is")
        return False
    if getattr(base, MARK, None) is not None:
        STATE.update(patched=1, state="on", reason="")
        return True
    patched = make_encoder_class(base)
    why = selftest(patched, base)
    STATE["selftest"] = "ok" if why is None else why
    if why is not None:
        STATE.update(state="refused", reason=why)
        _log(f"REFUSED: the fast encoder's text differs from {M_WRITER}.{ENCODER_ATTR}'s on the probe structure ({why}) — the engine's writer runs as it is")
        return False
    setattr(mod, ENCODER_ATTR, patched)
    STATE.update(patched=1, state="on", reason="")
    _log(f"installed: {M_WRITER}.{ENCODER_ATTR} -> {patched.__name__} (indented json.dumps calls of the writer rendered row-wise at C level, "
         f"same characters; route={STATE['route']})")
    return True


class _WriterFinder:
    """Meta-path finder that patches the writer module right after its body runs; delegates the find to the other finders."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != M_WRITER:
            return None
        spec = None
        for finder in sys.meta_path:
            if finder is self or type(finder).__name__ == type(self).__name__:
                continue
            try:
                spec = finder.find_spec(fullname, path, target)
            except Exception:  # noqa: BLE001
                spec = None
            if spec is not None:
                break
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec
        orig = spec.loader.exec_module

        def exec_module(module, _orig=orig):
            _orig(module)
            try:
                patch_writer_module(module)
            except Exception as e:  # noqa: BLE001
                STATE.update(state="refused", reason=f"patch_error:{type(e).__name__}")
                _log(f"REFUSED: patching {M_WRITER} raised {type(e).__name__}: {e} — the engine's writer runs as it is")
        spec.loader.exec_module = exec_module
        return spec


FINDER = _WriterFinder()


def install(environ=None) -> dict:
    """Idempotent. Not requested: nothing. Requested: the route knob read, the writer module patched now when imported, else the ONE finder
    armed for its import; the census line registered at exit."""
    import os
    environ = os.environ if environ is None else environ
    if STATE["installed"]:
        return STATE
    try:
        if not requested(environ):
            return STATE
        if not M_WRITER:
            raise RuntimeError(f"{PREFIX} not bound to an engine: the kit adapter must configure(M_WRITER=) before install")
        STATE["route"] = route(environ)
    except ValueError as e:                                            # a switch word the cell does not know: the lever steps aside BY NAME, the engine's writer runs as it is
        STATE.update(installed=True, state="refused", reason="bad_word:" + str(e).split("=", 1)[0])
        _log(f"REFUSED: {e} — the engine's writer runs as it is")
        if not _ATEXIT["registered"]:
            atexit.register(lambda: sys.stderr.write(census_line() + "\n"))
            _ATEXIT["registered"] = True
        return STATE
    STATE.update(installed=True, state="armed", reason="")             # armed: requested, the writer module not met yet; on: its encoder class is ours; refused: named why
    mod = sys.modules.get(M_WRITER)
    if mod is not None:
        patch_writer_module(mod)
    else:
        sys.meta_path[:] = [f for f in sys.meta_path if f is FINDER or type(f).__name__ != type(FINDER).__name__]
        if FINDER not in sys.meta_path:
            sys.meta_path.insert(0, FINDER)
        _log(f"armed: {M_WRITER}.{ENCODER_ATTR} is patched when the engine imports its writer (route={STATE['route']})")
    if not _ATEXIT["registered"]:
        atexit.register(lambda: sys.stderr.write(census_line() + "\n"))
        _ATEXIT["registered"] = True
    return STATE


def uninstall() -> None:
    """Disarm the finder and put the stock class back when the module holds ours (tests; a process keeps one route)."""
    try:
        sys.meta_path.remove(FINDER)
    except ValueError:
        pass
    mod = sys.modules.get(M_WRITER)
    cur = getattr(mod, ENCODER_ATTR, None) if mod is not None else None
    base = getattr(cur, MARK, None) if cur is not None else None
    if base is not None:
        setattr(mod, ENCODER_ATTR, base)
    STATE.update(installed=False, state="off", reason="", patched=0)
