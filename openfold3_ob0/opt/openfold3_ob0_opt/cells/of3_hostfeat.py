"""The `hostfeat` lever (exact class, in BYTES): two host-side featurisation statements of
the engine's data pipeline re-stated in numpy so the per-item featurisation the predicting process waits for (item 1 of every process; the
DataLoader workers hide later items) is shorter, every feature identical.

Parts (`<KIT>_HOSTFEAT_PARTS`, default all the kit binds; each switchable alone):
  * `a3m` — `<engine>.core.data.io.sequence.msa.parse_a3m`: the a3m deletion matrix and the aligned (deletion-free) letter matrix. The engine
    walks every character of every row in Python (`for j in msa_sequence: if j.islower(): …`, then `str.translate`, then `np.empty('<U1')`
    row by row: one interpreter step per character of the alignment); the port computes the same two arrays from the
    byte view of the rows: the non-lowercase positions (ASCII a–z is exactly `str.islower` on ASCII text and exactly what the engine's
    `str.maketrans('', '', ascii_lowercase)` deletes), the deletion count before each = the distance to the previous non-lowercase byte of
    the row, the letters = those bytes as '<U1'. Same dtypes (platform int for the matrix — what `np.array(list of lists of int)` gives —,
    '<U1' C-contiguous letters), same shapes, then the engine's OWN tail verbatim per pinned text (0.4.x: `MsaArray(msa=, deletion_matrix=,
    metadata=)` + `truncate(max_seq_count)`; 0.5.x: `MsaArray.from_parsed(…)` + `truncate(…, inplace=True)`). Steps aside PER CALL to the
    engine's statement (counted) for anything the byte identity does not cover: a non-ASCII row, rows whose aligned lengths differ (the
    engine raises there — it raises the same way, from its own text), an empty alignment.
  * `msaidx` — `<engine>.core.data.resources.residues.map_str_array_to_idx_array` (0.4.x text: one full-array string comparison per alphabet
    letter + `np.isin` over the '<U1' MSA matrix, ~30 passes): a 256-entry lookup table per molecule type, FILLED BY EVALUATING THE ENGINE'S
    OWN FUNCTION on the 256 one-byte characters (so the table is the engine's mapping by construction, gap / unknown handling included), then
    one gather on the low byte of each '<U1' cell — the statement upstream 0.5.0 ships (`_get_residue_idx_lut` + byte view); a matrix holding
    a code point ≥ 256, or not '<U1', goes to the engine's statement (counted). Kits whose engine already ships the table (0.5.x) do not bind
    this part (n/a: upstream).
Why exact: integer / string features only — no floating point is touched; every array the port returns is compared equal, dtype and shape,
to the engine's by the kit tests on random alignments; the model inputs are the same tensors.
Where it runs: the engine imports both modules in the predicting process before the DataLoader forks its workers, so the patched functions
are what the workers run; `MSA_PARSER_REGISTRY['.a3m']` and the by-name importers of the mapping function are re-pointed too. Counters live
in shared memory created before the fork (a forked worker's calls are counted; a spawned one's would not be — the features are the same).
Census (exit): `<PREFIX> LEVER name=hostfeat state=<on|off|refused> parts=<a3m+msaidx|…> a3m_calls=<n> a3m_rows=<n> a3m_fallback=<n>
msaidx_calls=<n> msaidx_cells=<n> msaidx_fallback=<n> aside=<part:reason,…|none>` (a part that stepped aside by name).
Switches (the kit adapter's `configure(ENV=…, ENV_PARTS=…, PARTS=…, DIGESTS=…)`): <KIT>_HOSTFEAT=1 arms; <KIT>_HOSTFEAT_PARTS=a3m,msaidx.
"""
from __future__ import annotations

import atexit
import hashlib
import inspect
import multiprocessing
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

CONFIGURABLE = ("PREFIX", "ENV", "ENV_PARTS", "PARTS", "M_MSA_IO", "M_RESIDUES", "REBIND", "DIGESTS")
PREFIX = "[of3-opt/hostfeat]"
ENV = ""                                     # <KIT>_HOSTFEAT
ENV_PARTS = ""                               # <KIT>_HOSTFEAT_PARTS
PARTS: Tuple[str, ...] = ("a3m", "msaidx")   # the parts this kit binds (its default word)
M_MSA_IO = ""                                # <engine>.core.data.io.sequence.msa
M_RESIDUES = ""                              # <engine>.core.data.resources.residues
REBIND: Dict[str, Tuple[str, ...]] = {}      # part -> modules that import the patched name by name (re-pointed when already imported)
DIGESTS: Dict[str, Dict[str, str]] = {}      # function -> {sha256[:16] of its dedented source: tail variant}; a text not listed refuses that part by name
VALUES = ("1",)
KNOWN_PARTS = ("a3m", "msaidx")
MARK = "__of3opt_hostfeat__"

STATE: Dict[str, Any] = {"installed": False, "state": "off", "reason": "", "parts": (), "refused_parts": {}, "patched": {}}
ORIG: Dict[str, Any] = {}
_N = 8                                        # shared counters: a3m calls, rows, fallbacks, msaidx calls, cells, fallbacks, spare, spare
_SHM = {"arr": None}
_ATEXIT = {"registered": False}


def configure(**kw) -> None:
    for k, v in kw.items():
        if k not in CONFIGURABLE:
            raise TypeError(f"{PREFIX} configure: unknown key {k!r}")
        globals()[k] = v


def _log(msg: str) -> None:
    sys.stderr.write(f"{PREFIX} {msg}\n")


def requested(environ=None) -> bool:
    environ = os.environ if environ is None else environ
    v = (environ.get(ENV) or "").strip() if ENV else ""
    if not v:
        return False
    if v not in VALUES:
        raise ValueError(f"{ENV}={v!r} is not one of {VALUES}")
    return True


def parts(environ=None) -> Tuple[str, ...]:
    environ = os.environ if environ is None else environ
    word = (environ.get(ENV_PARTS) or "").strip() if ENV_PARTS else ""
    if not word:
        return tuple(PARTS)
    out = []
    for p in word.split(","):
        p = p.strip()
        if p not in KNOWN_PARTS or p not in PARTS:
            raise ValueError(f"{ENV_PARTS}={word!r}: part {p!r} is not one of {tuple(PARTS)}")
        if p not in out:
            out.append(p)
    return tuple(out)


def _counters():
    a = _SHM["arr"]
    if a is None:
        try:
            a = multiprocessing.RawArray("q", _N)             # before the DataLoader forks: the workers' increments land here
        except Exception:  # noqa: BLE001
            a = [0] * _N
        _SHM["arr"] = a
    return a


def _bump(i: int, n: int = 1) -> None:
    a = _counters()
    try:
        a[i] += n
    except Exception:  # noqa: BLE001
        pass


def digest(fn) -> str:
    try:
        src = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        return ""
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]


# ─── part a3m ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def a3m_arrays(sequences) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(deletion_matrix, msa) exactly as the engine's loop + translate + _msa_list_to_np + np.array give them, or None when the byte identity
    does not cover the input (non-ASCII text, unequal aligned lengths, an empty alignment) — the caller then runs the engine's statement."""
    n = len(sequences)
    if n == 0:
        return None
    try:
        blob = "".join(sequences).encode("ascii")
    except UnicodeEncodeError:
        return None
    lens = np.fromiter((len(s) for s in sequences), dtype=np.int64, count=n)
    starts = np.zeros(n, dtype=np.int64)
    np.cumsum(lens[:-1], out=starts[1:])
    b = np.frombuffer(blob, dtype=np.uint8)
    keep = np.flatnonzero((b < 97) | (b > 122))               # not ASCII a-z == not str.islower() on ASCII == kept by str.translate(delete a-z)
    if keep.size == 0:
        return None
    row = np.searchsorted(starts, keep, side="right") - 1     # the row each kept byte belongs to
    per_row = np.bincount(row, minlength=n)
    L = int(per_row[0])
    if L == 0 or not np.all(per_row == L):
        return None                                            # ragged / empty aligned rows: the engine's own statements decide (they raise)
    prev = np.empty_like(keep)
    prev[0] = -1
    prev[1:] = keep[:-1]
    dele = keep - np.maximum(prev, starts[row] - 1) - 1        # lowercase run length right before each kept character, within its row
    deletion_matrix = np.ascontiguousarray(dele.reshape(n, L).astype(np.array([0]).dtype, copy=False))
    msa = np.ascontiguousarray(b[keep].astype("<u4").view("<U1").reshape(n, L))
    return deletion_matrix, msa


def make_parse_a3m(mod, orig, variant: str):
    """The engine's parse_a3m with the two arrays from `a3m_arrays`; the tail (MsaArray construction + truncation) is the engine's per text."""
    parse_fasta = mod.parse_fasta

    def parse_a3m(msa_string, max_seq_count=None):
        sequences, metadata = parse_fasta(msa_string)
        arrays = a3m_arrays(sequences)
        if arrays is None:
            _bump(2)
            return orig(msa_string, max_seq_count)
        deletion_matrix, msa = arrays
        _bump(0); _bump(1, len(sequences))
        if variant == "v041":
            parsed_msa = mod.MsaArray(msa=msa, deletion_matrix=deletion_matrix, metadata=metadata)
            if max_seq_count is not None:
                parsed_msa.truncate(max_seq_count)
            return parsed_msa
        parsed_msa = mod.MsaArray.from_parsed(msa=msa, deletion_matrix=deletion_matrix, metadata=metadata)   # v050
        if max_seq_count is not None:
            parsed_msa.truncate(max_seq_count, inplace=True)
        return parsed_msa

    setattr(parse_a3m, MARK, True)
    parse_a3m.__wrapped__ = orig
    parse_a3m.__doc__ = orig.__doc__
    return parse_a3m


# ─── part msaidx ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
_LUT: Dict[Any, Tuple[np.ndarray, np.dtype]] = {}


def make_map_str_array_to_idx_array(orig):
    """A 256-entry table per molecule type filled by the ENGINE'S function on the one-byte characters; one gather on the low byte of '<U1'."""
    u1 = np.dtype("<U1")

    def lut_for(molecule_type):
        hit = _LUT.get(molecule_type)
        if hit is None:
            chars = np.array([chr(i) for i in range(256)], dtype="<U1").reshape(16, 16)
            ref = orig(chars, molecule_type)                   # the engine's own mapping of every byte value (gap, unknown, alphabet)
            hit = (np.ascontiguousarray(ref).reshape(256).copy(), ref.dtype)
            _LUT[molecule_type] = hit
        return hit

    def map_str_array_to_idx_array(msa_array, molecule_type):
        if not isinstance(msa_array, np.ndarray) or msa_array.dtype != u1:
            _bump(5)
            return orig(msa_array, molecule_type)
        arr = np.ascontiguousarray(msa_array)
        codes = arr.view("<u4").reshape(arr.shape)
        if codes.size and int(codes.max()) > 255:
            _bump(5)
            return orig(msa_array, molecule_type)
        try:
            lut, dt = lut_for(molecule_type)
        except Exception:                                      # the engine's function raises for this molecule type (0.4.x: RNA / DNA unknown-letter lookup): its own statement, its own error
            _bump(5)
            return orig(msa_array, molecule_type)
        _bump(3); _bump(4, int(codes.size))
        out = lut[codes]
        return out if out.dtype == dt else out.astype(dt)

    setattr(map_str_array_to_idx_array, MARK, True)
    map_str_array_to_idx_array.__wrapped__ = orig
    map_str_array_to_idx_array.__doc__ = orig.__doc__
    return map_str_array_to_idx_array


# ─── patching ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _refuse_part(part: str, reason: str) -> None:
    STATE["refused_parts"][part] = reason
    _log(f"part {part} steps aside: {reason} — the engine's statement runs as it is")


def _rebind(part: str, name: str, orig, new) -> int:
    n = 0
    for mname in REBIND.get(part, ()):
        m = sys.modules.get(mname)
        if m is not None and getattr(m, name, None) is orig:
            setattr(m, name, new); n += 1
    return n


def patch_msa_io(mod) -> None:
    if "a3m" not in STATE["parts"]:
        return
    orig = getattr(mod, "parse_a3m", None)
    if orig is None or getattr(orig, MARK, False):
        return
    d = digest(orig)
    variant = (DIGESTS.get("parse_a3m") or {}).get(d)
    if variant is None:
        _refuse_part("a3m", f"digest:parse_a3m={d or 'unreadable'}")
        return
    if not hasattr(mod, "parse_fasta") or not hasattr(mod, "MsaArray"):
        _refuse_part("a3m", "module_shape:parse_fasta/MsaArray")
        return
    new = make_parse_a3m(mod, orig, variant)
    ORIG[("a3m", mod.__name__)] = orig
    mod.parse_a3m = new
    reg = getattr(mod, "MSA_PARSER_REGISTRY", None)
    n_reg = 0
    if isinstance(reg, dict):
        for k, v in list(reg.items()):
            if v is orig:
                reg[k] = new; n_reg += 1
    STATE["patched"]["a3m"] = {"variant": variant, "registry": n_reg, "rebound": _rebind("a3m", "parse_a3m", orig, new)}


def patch_residues(mod) -> None:
    if "msaidx" not in STATE["parts"]:
        return
    orig = getattr(mod, "map_str_array_to_idx_array", None)
    if orig is None or getattr(orig, MARK, False):
        return
    d = digest(orig)
    if d not in (DIGESTS.get("map_str_array_to_idx_array") or {}):
        _refuse_part("msaidx", f"digest:map_str_array_to_idx_array={d or 'unreadable'}")
        return
    new = make_map_str_array_to_idx_array(orig)
    ORIG[("msaidx", mod.__name__)] = orig
    mod.map_str_array_to_idx_array = new
    STATE["patched"]["msaidx"] = {"rebound": _rebind("msaidx", "map_str_array_to_idx_array", orig, new)}


def _update_state() -> None:
    live = [p for p in STATE["parts"] if p in STATE["patched"]]
    pending = [p for p in STATE["parts"] if p not in STATE["patched"] and p not in STATE["refused_parts"]]
    if live or pending:
        STATE["state"] = "on"
    elif STATE["refused_parts"]:
        STATE.update(state="refused", reason=";".join(f"{k}={v}" for k, v in STATE["refused_parts"].items()))


TARGETS = {}


class _HostFeatFinder:
    """Meta-path finder patching the two data-pipeline modules right after their bodies run; re-entrant by name (composes with other finders)."""

    def __init__(self):
        self._busy = set()

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in TARGETS or fullname in self._busy:
            return None
        self._busy.add(fullname)
        try:
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
        finally:
            self._busy.discard(fullname)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec
        orig = spec.loader.exec_module

        def exec_module(module, _orig=orig, _name=fullname):
            _orig(module)
            try:
                TARGETS[_name](module)
                _update_state()
            except Exception as e:  # noqa: BLE001
                _refuse_part(_name.rsplit(".", 1)[-1], f"patch_error:{type(e).__name__}")
        spec.loader.exec_module = exec_module
        return spec


FINDER = _HostFeatFinder()


def census_line() -> str:
    a = _counters()
    g = lambda i: int(a[i]) if a is not None else 0  # noqa: E731
    live = [p for p in STATE["parts"] if p in STATE["patched"] or (STATE["state"] == "on" and p not in STATE["refused_parts"])]
    rp = ",".join(f"{k}:{v}" for k, v in STATE["refused_parts"].items()) or "none"      # parts that stepped aside by name (an engine text the port does not re-state)
    reason = f" reason={STATE['reason']}" if STATE["state"] == "refused" else ""
    return (f"{PREFIX} LEVER name=hostfeat state={STATE['state']}{reason} parts={'+'.join(live) or '-'} a3m_calls={g(0)} a3m_rows={g(1)} a3m_fallback={g(2)} "
            f"msaidx_calls={g(3)} msaidx_cells={g(4)} msaidx_fallback={g(5)} aside={rp}")


def serving() -> bool:
    return STATE["state"] == "on"


def install(environ=None) -> dict:
    """Idempotent. Not requested: nothing. Requested: the parts word read, the shared counters created (before any fork), each target module
    patched now when imported, else the ONE finder armed for its import; census at exit."""
    environ = os.environ if environ is None else environ
    if STATE["installed"]:
        return STATE
    try:
        if not requested(environ):
            return STATE
        if not M_MSA_IO or not M_RESIDUES:
            raise RuntimeError(f"{PREFIX} not bound to an engine: the kit adapter must configure(M_MSA_IO=, M_RESIDUES=) before install")
        STATE["parts"] = parts(environ)
    except ValueError as e:
        STATE.update(installed=True, state="refused", reason="bad_word:" + str(e).split("=", 1)[0])
        _log(f"REFUSED: {e} — the engine's featurisation runs as it is")
        if not _ATEXIT["registered"]:
            atexit.register(lambda: sys.stderr.write(census_line() + "\n")); _ATEXIT["registered"] = True
        return STATE
    _counters()
    TARGETS.clear()
    TARGETS[M_MSA_IO] = patch_msa_io
    TARGETS[M_RESIDUES] = patch_residues
    STATE.update(installed=True, state="on", reason="")
    armed = []
    for name, fn in TARGETS.items():
        mod = sys.modules.get(name)
        if mod is not None:
            try:
                fn(mod)
            except Exception as e:  # noqa: BLE001
                _refuse_part(name.rsplit(".", 1)[-1], f"patch_error:{type(e).__name__}")
        else:
            armed.append(name)
    if armed:
        sys.meta_path[:] = [f for f in sys.meta_path if f is FINDER or type(f).__name__ != type(FINDER).__name__]
        if FINDER not in sys.meta_path:
            sys.meta_path.insert(0, FINDER)
    _update_state()
    if not _ATEXIT["registered"]:
        atexit.register(lambda: sys.stderr.write(census_line() + "\n")); _ATEXIT["registered"] = True
    return STATE


def uninstall() -> None:
    try:
        sys.meta_path.remove(FINDER)
    except ValueError:
        pass
    for (part, mname), orig in list(ORIG.items()):
        mod = sys.modules.get(mname)
        name = "parse_a3m" if part == "a3m" else "map_str_array_to_idx_array"
        if mod is not None:
            cur = getattr(mod, name, None)
            setattr(mod, name, orig)
            reg = getattr(mod, "MSA_PARSER_REGISTRY", None)
            if isinstance(reg, dict):
                for k, v in list(reg.items()):
                    if v is cur:
                        reg[k] = orig
            for m2 in REBIND.get(part, ()):
                mm = sys.modules.get(m2)
                if mm is not None and getattr(mm, name, None) is cur:
                    setattr(mm, name, orig)
    ORIG.clear(); _LUT.clear()
    a = _SHM["arr"]
    if a is not None:
        for i in range(_N):
            a[i] = 0
    STATE.update(installed=False, state="off", reason="", parts=(), refused_parts={}, patched={})
