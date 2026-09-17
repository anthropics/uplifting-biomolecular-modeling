"""trimul_native.face -- the provider boundary of the triangle-multiplication kernel package.

The op (identical for every configuration this package serves)::

    z [B,N,N,c_z] or [N,N,c_z] = the PRE-LayerNorm pair tensor;  mask [B,N,N] / [N,N] 0/1 or None (all ones)
    x = LN_in(z);  a = mask * sigmoid(x W_ag^T) * (x W_ap^T);  b = mask * sigmoid(x W_bg^T) * (x W_bp^T)          (a, b: [.., N, N, c_hidden])
    X[i,j,:] = sum_k a[i,k,:] b[j,k,:]  (outgoing)   |   sum_k a[k,i,:] b[k,j,:]  (incoming)
    update = sigmoid(x W_og^T) * (LN_out(X) W_o^T);   serve(...) returns update, or z + update when residual=True

Weights: the ten tensors ``ln_in_w ln_in_b w_ag w_ap w_bg w_bp ln_out_w ln_out_b w_o w_og`` (no biases), eps 1e-5.  Output dtype follows
the caller's op: z's dtype (a bf16 z gives a bf16 update; an fp32-resident z computed under a bf16 autocast region gives the update in fp32
= the "f32z" form).

Structure: prologue (LN_in + the four gated projections + mask -> channel-major bf16 planes) | tensor-core contraction | epilogue (LN_out +
W_o + output gate (+ residual)).  The prologue and epilogue are this package's cubins (``csrc/``, built per architecture by ``build.py``,
recorded in ``build/manifest.json``); they load through the CUDA driver (``launch.py`` over ``_driver.py``) into the framework's primary
context and run on the framework's current stream, allocating their workspaces from the framework's caching allocator -- no host
synchronisation, CUDA-graph capturable after the first (loading) call.

Host-lean serving (engines call this op many times per step at small N, where host microseconds are a measurable share): everything a
repeated shape needs is built once and kept in the caller's ``cache`` dict -- the weight pack (key: identity of the ten tensors), the launch
plan per (device, c_z, c_hidden, N, batch, direction, form, residual) = grid / block / dynamic smem / tile-table entry + prebuilt argument
packs (``launch.ArgPack``) whose pointer slots are patched per call, and encoded tensor maps per (address, sizes, strides, box)
(``launch.tensor_map_cached``).  Per call the host does: dtype/shape checks, ``torch.empty`` for the planes and the output (allocator-cached,
graph-safe), slot patches, three enqueues (prologue cubin, cuBLAS bmm, epilogue cubin).  No Python-side allocation beyond those tensors, no
``.item()`` / synchronisation, no descriptor re-encoding while addresses repeat.

    from trimul_native import face as F
    F.admits(cc=(9,0), dtype="bf16", c_z=128, c_hidden=128, n_tokens=800, direction="outgoing")   # None = served; else the refusal word (pure, no torch)
    rep = F.check()                          # driver binding -> device arch -> manifest-verified cubin loads -> load check -> byte gate; dict of facts
    pack = F.pack_weights(w10, cache)        # once per weight set (cached in ``cache`` when given)
    out = F.serve(z, mask, direction="outgoing", weights=w10, residual=False, cache=cache)

Refusals are ``Refusal`` exceptions raised BY NAME before any launch -- never a silent substitute (``.kind`` is the word, ``.detail`` free text):

    no_cubin:<arch>            the build directory holds no cubin of a needed unit for the device's architecture
    driver_unavailable:<why>   neither libcuda.so.1 nor the cuda.bindings wheel resolves, or torch has no CUDA device
    cc_unsupported:<cc>        the device's compute capability has no architecture in this package (served: 9.0 -> sm_90a; 8.0/8.6/8.7/8.9 -> sm_80)
    loadcheck_failed:<case>    a loaded function's resource fingerprint differs from the manifest, or a packaged test vector does not reproduce
    manifest:<what>            a cubin or carried source file fails its recorded sha256 (or the manifest is absent)
    load:<driver error name>   the driver refused the cubin image (e.g. CUDA_ERROR_UNSUPPORTED_PTX_VERSION / INVALID_IMAGE on an older driver)
    shape:<what> | dtype:<what> | direction:<what> | weights:<what>     the call is outside the op's envelope (stated in ``admits``)
    variant:<what>             the requested variant (exact | fast) is not built for this width / form / architecture
    vectors:<what>             the test-vector set is absent or stale for this build (development trees; a sealed package always carries it)
    not_built:<what>           a unit or feature is not part of this tree yet (development trees only; a sealed package never raises it)

``config`` (serve): None, or a dict of launch-plan overrides for tuning / variant selection: ``{"variant": "fast"|"exact", "incoming_mode":
"kt"|"tn", "k1_cfg": (BI,BJ,NSLOT,SKCH), "k3_cfg": (BI,BJ,NSLOT,NACC)}`` (sm_90a member; absent keys = the member's tile table);
``{"sm80_tiles": (k1_name, k3_name)}`` (sm_80 member).
"""
import collections
import json
import logging
import math
import os
import threading

__all__ = ["Refusal", "admits", "check", "pack_weights", "serve", "describe", "served_table", "units_for_arch", "device_class", "ARCH_OF_CC", "WIDTHS",
           "DTYPE_WORDS", "WEIGHT_KEYS", "VERSION"]

VERSION = "1.2.2"
WEIGHT_KEYS = ("ln_in_w", "ln_in_b", "w_ag", "w_ap", "w_bg", "w_bp", "ln_out_w", "ln_out_b", "w_o", "w_og")
ARCH_OF_CC = {(9, 0): "sm_90a", (8, 0): "sm_80", (8, 6): "sm_80", (8, 7): "sm_80", (8, 9): "sm_80"}    # sm_80 SASS runs on every 8.x part
DEVICE_CLASS = {"sm_90a": "sm90", "sm_80": "sm80"}  # test-vector expectation classes (one set of expected bytes per class)
WIDTHS = (64, 128, 256, 384)                       # c_z and c_hidden, any pairing (tile tables are tuning data keyed by the pair; one code path)
N_MIN, N_MAX = 16, 4096                            # tokens; any value in range incl. ragged / non-multiple sizes (planes are padded internally)
BATCH_MAX = 64
DTYPE_WORDS = {"bf16": "bf16", "bfloat16": "bf16", "torch.bfloat16": "bf16", "fp32": "fp32", "float32": "fp32", "torch.float32": "fp32", "f32": "fp32"}
RESIDENCY_WORDS = (None, "bf16", "fp32")           # fp32 = fp32-resident z under a bf16 autocast region (compute bf16, update returned in fp32)
DIRECTION_WORDS = {"outgoing": "outgoing", "out": "outgoing", "incoming": "incoming", "in": "incoming"}
VARIANT_WORDS = ("fast", "exact")
CALLS_KEY = "trimul_native.calls"                   # the per-call plan memo inside the caller's cache dict (an OrderedDict, LRU-capped)
CALL_MEMO_MAX = 256                                 # distinct (weights object, geometry, mode, config contents) keys kept per cache
EXACT_MIN_N = 101                                   # the exact (reference-order) word is claimed from this many tokens up; below it the reference library
                                                    # itself switches to another arithmetic path, so bit identity is not defined there
PROBE_SELFTEST_SEED = 1234567                     # generator seed of the development probe self-test inputs (compared with a torch expression at run time)
DEV_UNITS = ("probe",)                             # development-only units (route probe); loaded by check(probe=True), never by serve
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BUILD_DIR = os.environ.get("TRIMUL_NATIVE_BUILD_DIR") or os.path.join(ROOT, "build")
_LOCK = threading.Lock()
_STATE = {"checked": {}, "check_runs": 0, "table": None, "table_src": None, "sm90": {}, "gate": {}, "cells": None, "cells_src": None}
CELLS_PATH = os.environ.get("TRIMUL_NATIVE_CELLS") or os.path.join(ROOT, "CELLS.json")     # the measured-cell table shipped with a release
COVER_FACTOR = 1.5                      # a call's N is covered by a measured N-bucket when within this factor of it (log distance)
_NOTICES = {}                           # token -> detail; each token is emitted once per process
_NOTICE_SINKS = []
_LOG = logging.getLogger("trimul_native")
_FORM_WORD = {"b": "bf16", "f": "f32z"}


class Refusal(RuntimeError):
    """Raised BY NAME when this package cannot serve a call as asked.  ``.kind`` is the refusal word (see the module docstring); ``.reason``
    is an alias of ``.kind``; ``.detail`` free text."""

    def __init__(self, kind, detail=""):
        RuntimeError.__init__(self, "trimul_native: %s%s" % (kind, (" (" + detail + ")") if detail else ""))
        self.kind = kind
        self.reason = kind
        self.detail = detail


# ---------------------------------------------------------------------------------------------------------------- what the build serves (pure)
def _read_manifest(build_dir=None):
    p = os.path.join(build_dir or BUILD_DIR, "manifest.json")
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def served_table(build_dir=None):
    """{(arch, c_z, c_hidden, form 'b'|'f'): {"units": [unit, ...], "variants": ["fast"(, "exact")], "kernels": [...]}} from the ``serves``
    records of build/manifest.json (written at build time by asking the op assembly's naming authority which instantiations each width / form
    needs and checking them against the compiled cubin).  sm_90a: one width unit serves a row by itself; sm_80: a row needs its K1 kernel in
    the K1 unit and its K3 kernel in the K3 unit.  Pure (json only); no name patterns live here."""
    bd = os.path.abspath(build_dir or BUILD_DIR)
    with _LOCK:
        if _STATE["table"] is not None and _STATE["table_src"] == bd:
            return _STATE["table"]
    man = _read_manifest(bd) or {}
    table = {}
    sm80 = {}                                                               # (cz, ch, form) -> {"k1": (unit, kernel), "k3": (unit, kernel)}
    for key, ent in (man.get("units") or {}).items():
        unit, arch = ent.get("unit"), ent.get("arch")
        serves = ent.get("serves")
        if ent.get("dev") or not isinstance(serves, list) or not os.path.isfile(os.path.join(bd, arch, unit + ".cubin")):
            continue
        for row in serves:
            try:
                cz, ch, form = int(row["c_z"]), int(row["c_hidden"]), row["form"]
            except (KeyError, TypeError, ValueError):
                continue
            if arch == "sm_90a":
                if row.get("fast"):
                    table[(arch, cz, ch, form)] = {"units": [unit], "variants": ["fast", "exact"] if row.get("exact") else ["fast"], "kernels": {unit: list(row.get("kernels") or [])}}
            elif arch == "sm_80":
                if row.get("present") and row.get("part") in ("k1", "k3"):
                    sm80.setdefault((cz, ch, form), {})[row["part"]] = (unit, row.get("kernel"))
                elif row.get("present") and row.get("part") in ("k1x", "k3x"):
                    sm80.setdefault((cz, ch, form), {})[row["part"]] = (unit, list(row.get("kernels") or []))
    for (cz, ch, form), parts in sm80.items():
        if "k1" in parts and "k3" in parts:
            kern = {}
            for part in ("k1", "k3"):
                kern.setdefault(parts[part][0], []).append(parts[part][1])
            variants = ["fast"]
            if "k1x" in parts and "k3x" in parts:                           # the bit-exact assembly of this pair is compiled in
                variants.append("exact")
                for part in ("k1x", "k3x"):
                    kern.setdefault(parts[part][0], []).extend(parts[part][1])
            table[("sm_80", cz, ch, form)] = {"units": sorted(kern), "variants": variants, "kernels": kern}
    with _LOCK:
        _STATE["table"], _STATE["table_src"] = table, bd
    return table


def units_for_arch(arch, build_dir=None):
    """The compute units a serve on ``arch`` may load (every unit the served table names for that architecture)."""
    us = set()
    for key, v in served_table(build_dir).items():
        if key[0] == arch:
            us.update(v["units"])
    return sorted(us)


def device_class(arch):
    return DEVICE_CLASS.get(arch)


def _dtype_word(dtype):
    return DTYPE_WORDS.get(str(dtype).lower() if not isinstance(dtype, str) else dtype.lower())


def release_archs(build_dir=None):
    """The set of architectures this build / release carries compute units for (development units excluded)."""
    bd = os.path.abspath(build_dir or BUILD_DIR)
    man = _read_manifest(bd) or {}
    return {e.get("arch") for k, e in (man.get("units") or {}).items()
            if e.get("unit") not in DEV_UNITS and e.get("arch") and os.path.isfile(os.path.join(bd, e.get("arch"), str(e.get("unit")) + ".cubin"))}


def admits(cc, dtype, c_z, c_hidden, n_tokens, direction="outgoing", residency=None, residual=False, batch=1, variant="fast", build_dir=None, cells=True):
    """None when the package serves this call, else the refusal word it would raise.  Pure: no framework import, no device query (reads
    build/manifest.json once).

    ``cc`` (major, minor); ``dtype`` the compute dtype word ("bf16"); ``residency`` None | "bf16" | "fp32" (fp32-resident z under bf16
    autocast: bf16 compute, update returned in fp32, residual summed in fp32); ``direction`` outgoing | incoming; ``residual`` bool; ``batch``
    leading batch size (1 for a 3-D z); ``variant`` fast | exact (exact = bit-identical to the reference library; bf16 z, sm_90a).
    ``cells`` True applies the measured-cell policy exactly as ``serve`` does (a served family without a measured cell of this card / form /
    class / width / direction / residual -> its ``no_cell:<key>`` word); False answers for the binaries alone (byte gates, tuning runs)."""
    try:
        cc = (int(cc[0]), int(cc[1]))
    except (TypeError, ValueError, IndexError):
        return "cc_unsupported:%r" % (cc,)
    arch = ARCH_OF_CC.get(cc)
    if arch is None:
        return "cc_unsupported:%d.%d" % cc
    if _dtype_word(dtype) != "bf16":
        return "dtype:%s(compute dtype is bf16; fp32-resident z is residency='fp32')" % (dtype,)
    if residency not in RESIDENCY_WORDS:
        return "dtype:residency=%r" % (residency,)
    try:
        c_z, c_hidden, n_tokens, batch = int(c_z), int(c_hidden), int(n_tokens), int(batch)
    except (TypeError, ValueError):
        return "shape:non-integer"
    if c_z not in WIDTHS:
        return "shape:c_z=%s(not in %s)" % (c_z, list(WIDTHS))
    if c_hidden not in WIDTHS:
        return "shape:c_hidden=%s(not in %s)" % (c_hidden, list(WIDTHS))
    if not (N_MIN <= n_tokens <= N_MAX):
        return "shape:n=%s(outside %d..%d)" % (n_tokens, N_MIN, N_MAX)
    if DIRECTION_WORDS.get(str(direction).lower()) is None:
        return "direction:%s" % (direction,)
    if not (1 <= batch <= BATCH_MAX):
        return "shape:batch=%s(outside 1..%d)" % (batch, BATCH_MAX)
    if variant not in VARIANT_WORDS:
        return "variant:%s(fast|exact)" % (variant,)
    form = "f" if residency == "fp32" else "b"
    table = served_table(build_dir)
    if arch not in release_archs(build_dir):                              # this release carries no binary member for the device's architecture
        return "cc_unsupported:%d.%d(no %s member in this release; members: %s)" % (cc[0], cc[1], arch, ",".join(sorted(release_archs(build_dir))) or "none")
    ent = table.get((arch, c_z, c_hidden, form))
    if ent is None:
        return "no_cubin:%s:z%d_h%d_%s" % (arch, c_z, c_hidden, "f32z" if form == "f" else "bf16")
    if variant == "exact" and c_z != c_hidden:
        return "exact_no_reference:%dx%d(the reference library defines no result at c_hidden != c_z; the fast variant serves this pair)" % (c_z, c_hidden)
    if variant == "exact" and "exact" not in ent["variants"]:
        return "variant:exact(z%d_h%d_%s on %s)" % (c_z, c_hidden, "f32z" if form == "f" else "bf16", arch)
    if variant == "exact" and n_tokens < EXACT_MIN_N:
        return "exact_small_n:%d(the reference library takes a different path below %d tokens; the fast variant serves this size)" % (n_tokens, EXACT_MIN_N)
    if cells:
        cell = resolve_cell(cc, form, c_z, c_hidden, direction, n_tokens, klass=variant, residual=bool(residual))
        if cell.get("refuse"):
            return cell["refuse"]
    return None


def _notice(token, detail=""):
    """Emit an information token once per process: kept in notices(), logged at INFO on the 'trimul_native' logger, passed to every sink
    registered with add_notice_sink().  Never raises."""
    with _LOCK:
        if token in _NOTICES:
            return
        _NOTICES[token] = detail
    line = token + ((" " + detail) if detail else "")
    try:
        _LOG.info(line)
    except Exception:                                                       # noqa: BLE001
        pass
    for fn in list(_NOTICE_SINKS):
        try:
            fn(line)
        except Exception:                                                   # noqa: BLE001
            pass


def notices():
    """The information tokens emitted so far in this process, in order (e.g. 'UNCOVERED_CELL:<key>'), with their detail text."""
    with _LOCK:
        return [(k + ((" " + v) if v else "")) for k, v in _NOTICES.items()]


def add_notice_sink(fn):
    """Register fn(line: str) to receive every information token as it is first emitted (a provider routes these into its status line)."""
    if fn not in _NOTICE_SINKS:
        _NOTICE_SINKS.append(fn)


def _cc_tuple(v):
    if isinstance(v, (tuple, list)) and len(v) == 2:
        return (int(v[0]), int(v[1]))
    t = str(v).strip().lower().replace("sm_", "").replace("sm", "")
    if "." in t:
        a, b = t.split(".", 1)
        return (int(a), int(b[:1] or 0))
    if t.isdigit() and len(t) >= 2:
        return (int(t[:-1]), int(t[-1]))
    return None


def _form_of(row):
    f = str(row.get("form", "")).strip().lower()
    if f:                                                                   # an explicit form word decides: bf16 | f32z (fp32-resident z under bf16 compute)
        return "f" if (f.startswith("f32") or f.startswith("fp32") or "resident" in f) else "b"
    words = " ".join(str(row.get(k, "")) for k in ("dtype", "residency", "statement")).lower()
    if any(w in words for w in ("f32z", "fp32_resident", "fp32-resident", "f32_resident", "resident")) or str(row.get("dtype", "")).lower() in ("fp32", "f32", "float32"):
        return "f"
    return "b"


def _norm_row(raw):
    """One CELLS.json row -> {"cc", "form", "c_z", "c_hidden", "direction", "n", "config", "raw"} or None.  Accepts a flat row or one with a
    nested "key" mapping; width under c_hidden | hidden | c_h; the bucket under N_bucket | n_bucket | N | n | Np; cc as "9.0" | 90 | [9, 0];
    the served configuration under "config" ({"variant", "incoming_mode", "k1_cfg", "k3_cfg", "sm80_tiles"}; absent = the tile table's default)."""
    key = dict(raw)
    if isinstance(raw.get("key"), dict):
        key.update(raw["key"])
    cc = _cc_tuple(key.get("cc", key.get("arch", "")))
    try:
        cz = int(key.get("c_z", key.get("cz")))
        ch = int(key.get("c_hidden", key.get("hidden", key.get("c_h", key.get("ch", cz)))))
    except (TypeError, ValueError):
        return None
    n = None
    for k in ("N_bucket", "n_bucket", "N", "n", "Np", "tokens"):
        v = key.get(k)
        if isinstance(v, (int, float)) and v > 0:
            n = int(v)
            break
        if isinstance(v, str):
            digits = "".join(ch_ if ch_.isdigit() else " " for ch_ in v).split()
            if digits:
                n = int(digits[-1])                                         # "N<=768" -> 768, "512-1023" -> 1023 (upper edge names the bucket)
                break
    if cc is None or n is None:
        return None
    dw = str(key.get("direction", "*")).lower()
    direction = DIRECTION_WORDS.get(dw, "*") if dw not in ("*", "both", "any", "") else "*"
    cfg = raw.get("config") if isinstance(raw.get("config"), dict) else {}
    cfg = {k: (tuple(cfg[k]) if isinstance(cfg[k], list) else cfg[k]) for k in ("variant", "incoming_mode", "k1_cfg", "k3_cfg", "sm80_tiles") if k in cfg}
    words = " ".join(str(key.get(k, "")) for k in ("class", "tier", "variant", "lever", "numerics")).lower() + " " + str(cfg.get("variant", "")).lower()
    klass = "exact" if any(w in words for w in ("exact", "bitwise")) else "fast"
    rv = key.get("residual", "*")
    residual = "*" if rv in ("*", "both", "any", None) else bool(rv)
    return {"cc": cc, "arch": ARCH_OF_CC.get(cc), "form": _form_of(key), "class": klass, "c_z": cz, "c_hidden": ch, "direction": direction,
            "residual": residual, "n": n, "config": cfg, "raw": raw}


def cells(path=None):
    """The measured-cell table (CELLS.json at the package root, or TRIMUL_NATIVE_CELLS): normalised rows, cached per path.  [] when absent."""
    p = os.path.abspath(path or CELLS_PATH)
    with _LOCK:
        if _STATE["cells"] is not None and _STATE["cells_src"] == p:
            return _STATE["cells"]
    rows = []
    try:
        with open(p) as f:
            doc = json.load(f)
        raw_rows = doc if isinstance(doc, list) else (doc.get("rows") or doc.get("cells") or [])
        for raw in raw_rows:
            if isinstance(raw, dict):
                r = _norm_row(raw)
                if r is not None:
                    rows.append(r)
    except (OSError, ValueError):
        rows = []
    with _LOCK:
        _STATE["cells"], _STATE["cells_src"] = rows, p
    return rows


def cell_key(cc, form, c_z, c_hidden, direction, n_bucket, klass="fast", residual="*"):
    """The printed key of a cell: '<cc>|<bf16|f32z>|<fast|exact>|z<c_z>_h<c_hidden>|<direction>|res<0|1|*>|N<bucket>'."""
    return "%d.%d|%s|%s|z%d_h%d|%s|res%s|N%s" % (int(cc[0]), int(cc[1]), _FORM_WORD.get(form, form), klass, int(c_z), int(c_hidden), direction,
                                                 "*" if residual == "*" else int(bool(residual)), n_bucket)


def resolve_cell(cc, form, c_z, c_hidden, direction, n, klass="fast", residual=False, path=None):
    """UNCOVERED-KEY POLICY (pure).  The measured cell for a served call and the configuration to serve.  'Nearest' is taken ONLY within the
    same card (exact cc), form (bf16 | f32z), numerics class (fast | exact -- an exact request never inherits a tolerance-class row nor the
    reverse), width pair (c_z, c_hidden) and call form (direction, residual), and only in SIZE: among those rows the N-bucket nearest to n in
    log distance (ties -> the larger bucket) is the cell.  n within COVER_FACTOR of that bucket = covered (served silently with the row's
    configuration); farther = served with that nearest row's configuration and the token 'UNCOVERED_CELL:<key>' is due; no such row at all
    = {"refuse": "no_cell:<key>"} (the face refuses by name; nothing is inherited from another width, class, card or call form).  A row whose
    direction / residual is recorded as '*' matches both.  No cells table at all (a development tree) = {"table": False}: served with the
    kernel tile table's defaults, notice 'NO_CELLS_TABLE' once.
    Returns {"key", "covered", "cell" (raw row | None), "using" (key of the row used | ""), "config", "token" (None | 'UNCOVERED_CELL:<key>'),
    "refuse" (None | 'no_cell:<key>'), "table" (bool)}."""
    cc = (int(cc[0]), int(cc[1]))
    dname = DIRECTION_WORDS.get(str(direction).lower(), str(direction).lower())
    res = bool(residual)
    key = cell_key(cc, form, c_z, c_hidden, dname, int(n), klass, res)
    table = cells(path)
    if not table:
        return {"key": key, "covered": False, "cell": None, "using": "", "config": {}, "token": None, "refuse": None, "table": False}
    fam = [r for r in table if r["cc"] == cc and r["form"] == form and r["class"] == klass and r["c_z"] == int(c_z) and r["c_hidden"] == int(c_hidden)
           and r["direction"] in (dname, "*") and r["residual"] in (res, "*")]
    if not fam:
        return {"key": key, "covered": False, "cell": None, "using": "", "config": {}, "token": None, "refuse": "no_cell:" + key, "table": True}
    best = min(fam, key=lambda r: (abs(math.log2(max(1, r["n"])) - math.log2(max(1, int(n)))), -r["n"]))
    covered = max(best["n"], int(n)) <= COVER_FACTOR * max(1, min(best["n"], int(n)))
    using = cell_key(cc, form, best["c_z"], best["c_hidden"], dname if best["direction"] == "*" else best["direction"], best["n"], klass,
                     res if best["residual"] == "*" else best["residual"])
    if covered:
        key = using
    return {"key": key, "covered": covered, "cell": best["raw"], "using": using, "config": dict(best["config"]),
            "token": None if covered else "UNCOVERED_CELL:" + key, "refuse": None, "table": True}


def describe(build_dir=None):
    """The envelope, the served table and the build facts as plain data (no framework import)."""
    table = served_table(build_dir)
    out = {"version": VERSION, "archs": sorted(set(ARCH_OF_CC.values())), "cc": sorted("%d.%d" % k for k in ARCH_OF_CC), "widths": list(WIDTHS),
           "n_tokens": [N_MIN, N_MAX], "dtypes": ["bf16"], "residency": ["bf16", "fp32"], "directions": ["outgoing", "incoming"], "residual": [False, True],
           "served": sorted("%s z%d h%d %s %s" % (a, cz, ch, "f32z" if f == "f" else "bf16", "+".join(v["variants"])) for (a, cz, ch, f), v in table.items()),
           "build": None, "vectors": None}
    man = _read_manifest(build_dir)
    if man is None:
        out["build"] = {"error": "manifest absent under %s" % (build_dir or BUILD_DIR)}
    else:
        out["build"] = {k: {"cubin_sha256": v.get("cubin_sha256"), "nvcc": v.get("nvcc"), "n_kernels": len(v.get("kernels") or {}), "dev": bool(v.get("dev"))}
                        for k, v in sorted((man.get("units") or {}).items())}
    tv = os.path.join(ROOT, "testvectors", "manifest.json")
    if os.path.isfile(tv):
        try:
            with open(tv, encoding="utf-8") as f:
                t = json.load(f)
            out["vectors"] = {"version": t.get("version"), "made_utc": t.get("made_utc"), "cases": len(t.get("cases") or []),
                              "classes": sorted({c for case in t.get("cases") or [] for c in (case.get("expected") or {})})}
        except (OSError, ValueError) as e:
            out["vectors"] = {"error": str(e)[:200]}
    d = out
    if isinstance(d, dict):
        cl = cells()
        d["cells"] = {"path": os.path.abspath(CELLS_PATH), "rows": len(cl),
                      "keys": sorted({cell_key(r["cc"], r["form"], r["c_z"], r["c_hidden"], r["direction"], r["n"], r["class"], r["residual"]) for r in cl})[:64]}
        d["uncovered_policy"] = ("nearest in N only, within the same cc / form / class / (c_z,c_hidden) / direction / residual (covered within x%.2g of a "
                                 "measured bucket; farther = served with the nearest bucket's configuration + token UNCOVERED_CELL:<key> once per key); "
                                 "no such row = refusal no_cell:<key>; no table = tile-table defaults + notice NO_CELLS_TABLE") % COVER_FACTOR
        d["notices"] = notices()
    return d


# ---------------------------------------------------------------------------------------------------------------- load + gate
def _load(unit, device=None):
    from . import launch as L
    try:
        return L.load_unit(unit, device=device)
    except L.LoadError as e:
        raise Refusal(e.kind, str(e))


def _arch(device=None):
    from . import launch as L
    try:
        arch, cc = L.arch_of_device(device)
    except L.LoadError as e:
        raise Refusal(e.kind, str(e))
    if arch is None:
        raise Refusal("cc_unsupported:%d.%d" % cc)
    return arch, cc


def check(device=None, probe=False, gate=True, binding=None, cases="gate", verbose=False):
    """Resolve and verify everything a serve needs on ``device`` (default: torch's current device), once per (process, device):
    driver binding (``binding``: None = env TRIMUL_NATIVE_DRIVER / auto, or "ctypes" / "cuda_bindings") -> compute capability -> architecture ->
    every compute unit's cubin loaded after its manifest digest checks (cubin + carried sources) -> load check (registers / local bytes == the
    ptxas record) -> byte gate: the packaged test vectors of the device class are replayed through ``serve`` and compared bitwise
    (``cases``: "gate" = the gate subset, "all" = every case; ``gate=False`` skips the replay for a caller that gates elsewhere).  ``probe=True``
    also loads and runs the development probe unit when the tree carries it.  Returns a dict report; raises ``Refusal`` by name."""
    from . import _driver
    from . import launch as L
    try:
        dev_key = L._device_index(device)
    except Exception:                                                       # noqa: BLE001 -- an unusable device word is reported by _arch() below
        dev_key = str(device)
    memo_key = (dev_key, bool(probe), bool(gate), cases)                # one full check per (process, device, options); later serve() calls cost nothing here
    with _LOCK:
        rep = _STATE["checked"].get(memo_key)
        if rep is None and not gate:                                        # a gated report also satisfies an ungated request
            rep = _STATE["checked"].get((dev_key, bool(probe), True, cases))
        if rep is not None:
            return rep
    try:
        drv = _driver.get(binding)
    except _driver.Unresolvable as e:
        raise Refusal("driver_unavailable:%s" % str(e).replace(" ", "_")[:120])
    arch, cc = _arch(device)
    import torch
    idx = L._device_index(device)
    _STATE["check_runs"] = _STATE.get("check_runs", 0) + 1                  # body executions (a memo hit never reaches here)
    rep = {"version": VERSION, "binding": drv.word, "driver_version": drv.driver_version(), "device": torch.cuda.get_device_name(idx), "cc": "%d.%d" % cc,
           "arch": arch, "device_class": device_class(arch), "torch": torch.__version__, "build_dir": BUILD_DIR, "units": {}, "served": [], "gate": None, "probe": None}
    units = units_for_arch(arch)
    if not units:
        raise Refusal("cc_unsupported:%d.%d" % cc if arch not in release_archs() else "no_cubin:%s" % arch,
                      "this release carries no %s member (members: %s) under %s" % (arch, ",".join(sorted(release_archs())) or "none", BUILD_DIR))
    for unit in units:
        u = _load(unit, idx)
        lc = u.loadcheck()
        bad = [k for k, v in lc.items() if not v["ok"]]
        if bad:
            raise Refusal("loadcheck_failed:%s.%s" % (unit, bad[0]), str(lc[bad[0]]))
        for key, row in served_table().items():                            # the kernels the served table names must resolve in the loaded module
            if key[0] != arch:
                continue
            for kname in (row.get("kernels") or {}).get(unit, []):
                try:
                    u.kernel(kname)
                except L.LoadError as e:
                    raise Refusal("loadcheck_failed:%s.%s" % (unit, kname), str(e))
        regs = [v["regs"] for v in lc.values() if isinstance(v.get("regs"), int)]
        rep["units"][unit] = {"status": "loaded", "cubin_sha256": u.entry.get("cubin_sha256"), "n_kernels": len(lc),
                              "regs": [min(regs), max(regs)] if regs else None, "spilled": [k for k, v in lc.items() if v.get("local_bytes")]} if not verbose else \
                             {"status": "loaded", "cubin_sha256": u.entry.get("cubin_sha256"), "kernels": lc}
    rep["served"] = sorted("z%d_h%d_%s:%s" % (cz, ch, _FORM_WORD[f], "+".join(v["variants"])) for (a, cz, ch, f), v in served_table().items() if a == arch)
    if probe:
        rep["probe"] = _probe_selftest(idx)
    if gate:
        with _LOCK:                                                         # the gate replays vectors through serve(), whose own (ungated) check must hit this memo
            _STATE["checked"][(dev_key, bool(probe), False, cases)] = rep
        rep["gate"] = _byte_gate(idx, arch, cases)
    with _LOCK:
        _STATE["checked"][memo_key] = rep
    return rep


def _probe_selftest(idx):
    """Load the development probe unit and run its kernels against torch expressions (bytes must be identical)."""
    import torch
    from . import launch as L
    u = _load("probe", idx)
    dev = torch.device("cuda", idx)
    g = torch.Generator(device="cpu").manual_seed(PROBE_SELFTEST_SEED)
    rows, cols, scale = 200, 328, 0.71875                                   # ragged against the 64x64 tile on both axes; ld = cols (multiple of 8)
    src = torch.randn(rows, cols, generator=g).to(dev).to(torch.bfloat16)
    want = (src.float() * scale).to(torch.bfloat16)
    out = {}
    grid = ((cols + 63) // 64, (rows + 63) // 64, 1)
    dst = torch.zeros_like(src)
    u.kernel("probe_cp_async").launch(grid, (128, 1, 1), [src, dst, L.i32(rows), L.i32(cols), L.i32(cols), L.f32(scale)])
    torch.cuda.synchronize(dev)
    out["probe_cp_async"] = {"equal": bool(torch.equal(dst, want)), "sha16": _sha16(dst)}
    if u.arch == "sm_90a":
        tm = L.tensor_map(src, box=(64, 64))
        dst2 = torch.zeros_like(src)
        u.kernel("probe_tma").launch(grid, (128, 1, 1), [L.Struct([tm, dst2, L.i32(rows), L.i32(cols), L.i32(cols), L.f32(scale)])])
        torch.cuda.synchronize(dev)
        out["probe_tma"] = {"equal": bool(torch.equal(dst2, want)), "sha16": _sha16(dst2)}
    echo = torch.zeros(8, dtype=torch.int64, device=dev)
    raw = bytes(range(128))
    u.kernel("probe_echo").launch((1, 1, 1), (32, 1, 1), [L.Struct([L.u64(0x1122334455667788), L.i32(-7), L.TensorMap(raw), L.f32(1.5), L.i64(-(2 ** 40)), echo])])
    torch.cuda.synchronize(dev)
    e = [int(v) & (2 ** 64 - 1) for v in echo.tolist()]
    out["probe_echo"] = {"equal": e[0] == 0x1122334455667788 and e[1] == (2 ** 32 - 7) and e[2] == int.from_bytes(raw[:8], "little")
                         and e[3] == int.from_bytes(raw[120:128], "little") and e[4] == 0x3FC00000 and e[5] == (2 ** 64 - 2 ** 40) and e[6] == 256
                         and e[7] == ((64 << 32) | (192 << 16) | 208), "words": ["0x%x" % v for v in e]}
    out["ok"] = all(v.get("equal", True) for v in out.values() if isinstance(v, dict))
    return out


def _sha16(t):
    import hashlib
    import torch
    return hashlib.sha256(t.detach().contiguous().view(-1).view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def _byte_gate(idx, arch, cases="gate"):
    """Replay the packaged test vectors of this device class through ``serve``; any byte difference is ``loadcheck_failed:<case>``, a stale or
    absent vector set is ``vectors:<what>``."""
    from . import vectors as V
    cls = device_class(arch)
    try:
        rep = V.replay(device_index=idx, which=cases, device_class=cls, raise_on_fail=False, quiet=True)
    except V.VectorsError as e:
        raise Refusal("vectors:%s" % e.kind, str(e))
    if rep["failed"]:
        first = rep["failed"][0]
        raise Refusal("loadcheck_failed:%s" % first["id"], first.get("why", ""))
    return {"status": "bitwise", "class": cls, "cases": rep["n"], "passed": rep["passed"], "skipped": rep.get("skipped", []), "seconds": rep.get("seconds")}


# ---------------------------------------------------------------------------------------------------------------- weights
def _wget(weights, k):
    if isinstance(weights, dict):
        return weights.get(k)
    return getattr(weights, k, None)


def pack_weights(weights, cache=None, arch=None, device=None):
    """The kernel-side weight pack of the member serving ``arch`` (default: the current device's) from the ten canonical tensors.  Cached in
    ``cache`` (a dict) under a key made of the tensors' (address, shape, dtype) and the device -- never the tensors' version counters, so the
    pack is usable under ``torch.inference_mode`` -- when given.  A caller that updates weights in place at the same address must drop the
    cache entry.  Raises Refusal("weights:<what>")."""
    import torch
    missing = [k for k in WEIGHT_KEYS if _wget(weights, k) is None]
    if missing:
        raise Refusal("weights:missing:%s" % ",".join(missing))
    if arch is None:
        arch, _ = _arch(device)
    ts = [_wget(weights, k) for k in WEIGHT_KEYS]
    dev = torch.device(device) if device is not None else ts[2].device
    key = ("trimul_native.pack", arch, str(dev)) + tuple((t.data_ptr(), tuple(t.shape), str(t.dtype)) for t in ts)
    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return hit
    c_hidden, c_z = (int(v) for v in _wget(weights, "w_ap").shape)
    shapes = {"ln_in_w": (c_z,), "ln_in_b": (c_z,), "w_ag": (c_hidden, c_z), "w_ap": (c_hidden, c_z), "w_bg": (c_hidden, c_z), "w_bp": (c_hidden, c_z),
              "ln_out_w": (c_hidden,), "ln_out_b": (c_hidden,), "w_o": (c_z, c_hidden), "w_og": (c_z, c_z)}
    for k, shp in shapes.items():
        if tuple(_wget(weights, k).shape) != shp:
            raise Refusal("weights:shape:%s=%s(want %s)" % (k, tuple(_wget(weights, k).shape), shp))
    if c_z not in WIDTHS or c_hidden not in WIDTHS:
        raise Refusal("shape:c_z=%d,c_hidden=%d(not in %s)" % (c_z, c_hidden, list(WIDTHS)))
    w10 = {k: _wget(weights, k).detach() for k in WEIGHT_KEYS}
    with torch.no_grad(), torch.inference_mode(False):
        if arch == "sm_90a":
            from . import ops as O
            pack = O.pack_weights(w10, dev)
        elif arch == "sm_80":
            from . import sm80_ops as S
            pack = S.pack_weights({k: v.to(dev, torch.float32) for k, v in w10.items()}, dev)
            pack["_src"] = id(weights)
        else:
            raise Refusal("cc_unsupported:%s" % arch)
    pack["_arch"] = arch
    if cache is not None:
        cache[key] = pack
    return pack


# ---------------------------------------------------------------------------------------------------------------- serve
def _sm90_kernel():
    """The sm_90a op assembly's kernel module (tile table, lookup(), naming); imported lazily (it imports the framework)."""
    from . import kernel as K
    return K


def _sm90_ops(idx):
    """The sm_90a op assembly with its process-wide kernel table bound to this package's driver route and build directory."""
    with _LOCK:
        O = _STATE["sm90"].get(idx)
    if O is not None:
        return O
    from . import launch as L
    from . import ops as O
    try:
        O.kernels(build_dir=L.DEFAULT_BUILD_DIR)                            # the op assembly loads units through launch.load_unit (digest-verified)
    except TypeError:
        try:
            O.kernels(drv=L.block_driver(device=idx), cubin_dir=os.path.join(L.DEFAULT_BUILD_DIR, "sm_90a"), arch="sm90")   # parameter-block surface
        except TypeError:
            O.kernels()
    with _LOCK:
        _STATE["sm90"][idx] = O
    return O


def _plain_weights(weights, cache):
    """The weights mapping itself when no tensor in it is an inference tensor; else a mapping of ordinary copies made once per (weights address
    set) and kept in ``cache`` -- functional copies of inference tensors are ordinary tensors, so members that consult version counters work
    under torch.inference_mode.  Keyed by data address / dtype / shape like every other weight-derived state of this face."""
    import torch
    keys = [k for k in WEIGHT_KEYS if k in weights] if "WEIGHT_KEYS" in globals() else list(weights.keys())
    tens = {k: weights[k] for k in keys if torch.is_tensor(weights[k])}
    if not any(t.is_inference() for t in tens.values()):
        return weights
    sig = ("trimul_native.plain_weights",) + tuple((k, t.data_ptr(), str(t.dtype), tuple(t.shape)) for k, t in sorted(tens.items()))
    got = cache.get(sig)
    if got is None:
        with torch.inference_mode(False), torch.no_grad():
            got = {k: (t.detach().clone() if t.is_inference() else t) for k, t in tens.items()}
        for k in weights:
            got.setdefault(k, weights[k])
        cache[sig] = got
    return got


def _config_key(config):
    """Hashable key of a config mapping by contents (tensors by address / dtype / shape), so a dict rebuilt per call maps to one memo entry."""
    if not config:
        return ()
    def frz(v):
        if isinstance(v, (str, int, float, bool, type(None))):
            return v
        if isinstance(v, (list, tuple)):
            return tuple(frz(x) for x in v)
        if isinstance(v, dict):
            return tuple(sorted((str(k), frz(x)) for k, x in v.items()))
        if hasattr(v, "data_ptr") and hasattr(v, "shape"):
            return ("tensor", int(v.data_ptr()), str(v.dtype), tuple(v.shape))
        try:
            hash(v)
            return v
        except TypeError:
            return repr(v)
    return tuple(sorted((str(k), frz(v)) for k, v in config.items()))


def serve(z, mask=None, *, direction, weights, residual=False, cache=None, eps=1e-5, config=None, save_intermediates=None):
    """Compute the op on ``z`` ([B,N,N,c_z] or [N,N,c_z], bf16, or fp32 inside a bf16 autocast region = the fp32-resident form) with ``mask``
    ([B,N,N] / [N,N] or None) and the ten ``weights``; returns the update (or z + update when ``residual``) in z's dtype.  ``cache``: a dict
    the caller keeps across calls; it receives the weight pack (``("trimul_native.pack", ...)``, keyed by tensor addresses / shapes / dtypes --
    never version counters), the launch plans and workspaces of the member (per N / width) and the encoded tensor maps (per address and
    geometry), so a repeated shape costs no packing, planning or descriptor encoding.  Safe under ``torch.inference_mode``, ``torch.no_grad``,
    autograd-enabled eager (the forward is not differentiable: outputs never require grad), non-default streams (launches go to the current
    stream) and CUDA-graph capture after the first call on the device (module loads and the one-time layout check synchronise; call ``check``
    or serve once eagerly before capturing).  ``config``: optional launch-plan / variant override (module docstring); None = the packaged table,
    fast variant.  ``save_intermediates``: None (inference; zero cost); the training path is not part of this release.  Raises ``Refusal`` by
    name before any launch when the call is outside the envelope or the package cannot load on this device; never substitutes another
    implementation."""
    import torch
    if save_intermediates is not None:
        raise Refusal("not_built:save_intermediates")
    if cache is None:
        cache = {}
    # ---- per-call fast path: a repeated (weights object, z geometry/dtype, mask geometry, direction, residual, variant, mode) skips validation,
    #      the served-table lookup and the pack key; guarded by the first projection weight's address (an in-place weight swap misses).
    try:
        inference = bool(z.is_inference())
        autocast_on = torch.is_autocast_enabled()
        fkey = ("trimul_native.call", id(weights), z.dtype, z.shape, None if mask is None else (mask.shape, mask.dtype), direction, bool(residual),
                z.device.index, inference, autocast_on, _config_key(config))          # config by CONTENTS: a dict rebuilt per call still hits
        calls = cache.get(CALLS_KEY)
        if calls is None:
            calls = cache[CALLS_KEY] = collections.OrderedDict()
        ent = calls.get(fkey)
        if ent is not None:
            calls.move_to_end(fkey)
        w_ap = _wget(weights, "w_ap")
        if ent is not None and ent["w_ptr"] != w_ap.data_ptr():
            ent = None
    except (AttributeError, TypeError):
        raise Refusal("shape:z=%s(want a CUDA tensor [B,N,N,c_z] or [N,N,c_z])" % (type(z).__name__,))
    if ent is None:
        if not torch.is_tensor(z) or z.dim() not in (3, 4) or z.shape[-2] != z.shape[-3]:
            raise Refusal("shape:z=%s(want [B,N,N,c_z] or [N,N,c_z])" % (tuple(z.shape) if torch.is_tensor(z) else type(z).__name__,))
        if not z.is_cuda:
            raise Refusal("driver_unavailable:z_on_%s" % z.device.type)
        batch = int(z.shape[0]) if z.dim() == 4 else 1
        n, c_z = int(z.shape[-2]), int(z.shape[-1])
        if w_ap is None:
            raise Refusal("weights:missing:w_ap")
        c_hidden = int(w_ap.shape[0])
        if z.dtype == torch.bfloat16:
            residency = "bf16"
        elif z.dtype == torch.float32 and autocast_on and torch.get_autocast_gpu_dtype() == torch.bfloat16:
            residency = "fp32"
        elif z.dtype == torch.float32 and (config or {}).get("f32_resident_ok"):
            residency = "fp32"
        else:
            raise Refusal("dtype:%s(bf16 z, or fp32 z inside a bf16 autocast region)" % (str(z.dtype).replace("torch.", ""),))
        if mask is not None:
            if not torch.is_tensor(mask) or not mask.is_cuda:
                raise Refusal("shape:mask(not a CUDA tensor)")
            ms = tuple(mask.shape)
            if ms != tuple(z.shape[:-1]) and not (z.dim() == 4 and ms in ((1,) + tuple(z.shape[1:-1]), tuple(z.shape[1:-1]))):
                raise Refusal("shape:mask=%s(want %s)" % (ms, tuple(z.shape[:-1])))
        idx = z.device.index if z.device.index is not None else torch.cuda.current_device()
        cc = _STATE.get(("cc", idx))
        if cc is None:
            cc = tuple(torch.cuda.get_device_capability(idx))
            _STATE[("cc", idx)] = cc
        # the measured cell for this key decides the configuration; the caller's config (tuning / tests) overrides it key by key;
        # config {"cells": False} serves without consulting the table (byte gates and tuning runs are table-independent)
        variant = (config or {}).get("variant", "fast")
        if (config or {}).get("cells", True) is False:
            cell = {"config": {}, "token": None, "refuse": None, "table": True, "using": ""}
        else:
            cell = resolve_cell(cc, "f" if residency == "fp32" else "b", c_z, c_hidden, direction, n, klass=variant, residual=bool(residual))
        cfg = dict(cell["config"])
        cfg.update(config or {})
        variant = cfg.get("variant", "fast")
        word = admits(cc, "bf16", c_z, c_hidden, n, direction, residency=residency, residual=residual, batch=batch, variant=variant, cells=False)
        if word is not None:
            raise Refusal(word)
        if cell["refuse"]:                                                  # a cells table exists but holds no row of this (cc, form, class, widths, call form)
            raise Refusal(cell["refuse"], "no measured cell of this card / form / class / width / direction / residual in %s" % CELLS_PATH)
        if not cell["table"]:
            _notice("NO_CELLS_TABLE", "path=%s (served with the kernel tile table's defaults)" % CELLS_PATH)
        arch = ARCH_OF_CC[cc]
        if cell["token"]:                                                   # uncovered size: served with the nearest bucket's configuration; say which
            tiles = ""
            if arch == "sm_90a":
                try:
                    lk = _sm90_kernel().lookup(arch, c_z, c_hidden, "f" if residency == "fp32" else "b", n)
                    if lk is not None:
                        tiles = " tiles=%s|z%d_h%d|%s|N%s%s" % (lk["key"][0], lk["key"][1], lk["key"][2], _FORM_WORD[lk["key"][3]],
                                                             "*" if lk["key"][4] is None else lk["key"][4], "" if lk.get("covered", True) else "(nearest bucket)")
                except Exception:                                           # noqa: BLE001 -- an older op assembly without lookup(): detail omitted
                    pass
            _notice(cell["token"], "nearest=" + cell["using"] + tiles)
        check(idx, gate=cfg.get("gate", True))
        dname = DIRECTION_WORDS[str(direction).lower()]
        # member workspaces live in a per-mode namespace of the caller's cache: buffers created under inference mode are only reused there
        sub = cache.setdefault(("trimul_native.ws", arch, idx, inference), {})
        pack = pack_weights(weights, cache, arch=arch, device=z.device) if arch == "sm_90a" else None
        if arch == "sm_80":
            if cfg.get("sm80_tiles"):
                sub["_sm80_tiles"] = tuple(cfg["sm80_tiles"])
        ent = {"w_ptr": w_ap.data_ptr(), "arch": arch, "idx": idx, "dname": dname, "sub": sub, "pack": pack, "variant": variant,
               "kw90": {k: cfg[k] for k in ("incoming_mode", "k1_cfg", "k3_cfg") if cfg.get(k) is not None},
               "wsrc": weights if isinstance(weights, dict) else {k: _wget(weights, k) for k in WEIGHT_KEYS}}
        calls[fkey] = ent
        while len(calls) > CALL_MEMO_MAX:                                  # bounded: least-recently used call keys leave first
            calls.popitem(last=False)
    arch, idx, dname, sub = ent["arch"], ent["idx"], ent["dname"], ent["sub"]
    # ---- mode normalisation (entered only when needed; each costs microseconds): no autograd graph, no autocast re-casting of our typed
    #      tensors, and inference mode matching the input's kind so cached workspaces are reused only with their own kind
    ctxs = []
    if autocast_on:
        ctxs.append(torch.autocast("cuda", enabled=False))
    inf_on = torch.is_inference_mode_enabled()
    if inference != inf_on:
        ctxs.append(torch.inference_mode(inference))
    elif not inf_on and torch.is_grad_enabled():
        ctxs.append(torch.no_grad())
    for c in ctxs:
        c.__enter__()
    try:
        if arch == "sm_90a":
            O = _STATE["sm90"].get(idx) or _sm90_ops(idx)
            zz = z if z.is_contiguous() else z.contiguous()
            mm = mask
            if mm is not None and mm.dtype != torch.float32:
                mm = mm.to(torch.float32)
            try:
                out = O.trimul(zz, mm, direction=dname, weights=ent["pack"], residual=bool(residual), exact=(ent["variant"] == "exact"), eps=float(eps), cache=sub, **ent["kw90"])
            except O.Unsupported as e:
                raise Refusal("shape:%s" % str(e)[:160])
        elif arch == "sm_80":
            from . import sm80_ops as S
            wsm = ent.get("wsrc_sm80")
            if wsm is None:                                                 # the sm_80 member keys its weight pack on version counters, which inference tensors lack:
                wsm = ent["wsrc_sm80"] = _plain_weights(ent["wsrc"], cache)  # hand it plain aliases (a one-time copy per weights object, keyed by address)
            out = S.serve_sm80(z, mask, direction=dname, weights=wsm, residual=bool(residual), cache=sub, eps=float(eps), exact=(ent["variant"] == "exact"))
        else:
            raise Refusal("cc_unsupported:%s" % arch)
    finally:
        for c in reversed(ctxs):
            c.__exit__(None, None, None)
    return out
