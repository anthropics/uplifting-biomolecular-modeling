"""Box gates: what a kit checks before it applies anything, engine-free.

Contract. A :class:`Gate` is one named check with a verdict, a reason and words; :func:`run_gates` runs every check and records every
verdict (the first refusal is the activation's reason; a later refusal is still recorded — nothing is silent). ``ok`` means the
activation may proceed past the check. A check REFUSES only for what cannot run or must not run: the stock engine's distribution at
another version than the tree pins (:func:`version_gate` ``stock=``), an ``opt_core`` older than the kit's minimum
(:func:`core_pin_check`), missing or altered carried bytes (:func:`verify_sums`). Uncertainty about the ENVIRONMENT — a framework or
toolchain wheel at another patch level, a card no engine was measured on, an unreadable memory size or pin table — passes WITH WORDS
(``Gate.words``: blank-free ``key=value`` tokens in :func:`opt_core.report.word`'s grammar, e.g. ``stack=drift(torch:2.13.0!=2.12.1)``,
``card=uncertified(NVIDIA_L4,22731MiB)``, ``core_pin=unreadable``) that the kit prints on its activation line
(:func:`opt_core.report.with_words` / :func:`opt_core.report.words_of`) and records; the fact each check found stays in ``details``.
The probes are the box as found: :func:`nvidia_smi_probe` (no torch import) and :func:`torch_gpu_probe` (imports torch) return the
same superset ``{"name", "cc", "sm", "memory_mib", "probe"}``; a kit that embeds the probe in its report projects it to its own keys
(``keys=``) so the report's bytes do not move. Three GPU checks, each the transcription of one form of the comparison:
:func:`gpu_name_check` (the card name contains a token), :func:`gpu_cc_check` (the compute capability), and :func:`gpu_class_check`
(name AND memory size); each records ``details["match"]`` (True · False · None = no requirement or nothing probed) and never refuses —
a lever that cannot run on a card says so through :mod:`opt_core.arch` (mechanism floors), not here. :func:`dist_version` reads
distribution metadata without importing the package. :func:`read_sums` / :func:`verify_sums` are ``<sha256>  <relpath>`` manifests with
total accounting (checked, missing, mismatched); :func:`carry_gate` reads the kit's carry manifest at its one place, ``opt/SHA256SUMS``
(entries relative to ``opt/``). :func:`binary_refusal` holds a compiled binary the core ships (extension module, cubin) to the
``SHA256SUMS`` line that lists it — in its own directory, else the nearest directory above it inside the package — before any loader
maps it: unlisted or altered bytes are refused by name, exactly as an absent binary is. :func:`core_pin_check` compares a kit's ``[tool.opt_core]`` pin with the ``opt_core`` actually imported
— the kit's first gate; equality is the git commit the tree is checked out at, not a restated hash, so the pin carries no tree digest.

Three outcomes, never a fourth. A mode is all of its levers on the card's class: (a) a lever that engages under named uncertainty is
ACTIVE and its words print (exit 0); (b) a lever that CANNOT RUN where its mechanism was asked to — a kernel that does not compile or
launch, an unsupported shape or dtype met at run time, a capture that fails, a card below a mechanism's architecture floor — is not a
quiet subset of the mode and not a continuation on the engine's stock code: the lever raises, the exception carries the attribute
``cannot_run = True`` (:func:`cannot_run` marks one, :func:`is_cannot_run` reads it; the core's own refusal classes carry it as a class
attribute), it propagates out of the lever to the kit, and the kit refuses the MODE by name (its NOT ACTIVE line,
``opt_core.report.EXIT_NOT_ACTIVE``); (c) an accounting violation (a census that does not close, an undeclared fallback reason, an
``error:*`` event) fails closed through :func:`opt_core.report.verdict` / :meth:`opt_core.counters.Ledger.gate`.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Optional, Sequence, Tuple


def word(key: str, value=None, *parts) -> str:
    """:func:`opt_core.report.word`, imported on first use: a stock process under the deterministic recipe imports this module
    (``precision.recipe`` -> ``precision.policy`` -> ``gates``) and may hold no core module beyond the recipe's own, so ``opt_core.report``
    is never loaded by importing ``opt_core.gates`` (tests/test_det_path_imports.py)."""
    from .report import word as _word  # noqa: PLC0415
    return _word(key, value, *parts)


# ------------------------------------------------------------------------------------------------------------------ gate records


@dataclass
class Gate:
    """One check: ``ok`` with ``reason`` None, or refused with the reason (a sentence a report can carry verbatim). ``forced`` marks a
    refusal overridden by the kit's force switch (recorded, not silent). ``words`` are the named uncertainty a PASSING check found
    (module contract): the activation proceeds and prints them."""

    name: str
    ok: bool
    reason: Optional[str] = None
    details: dict = field(default_factory=dict)
    forced: bool = False
    words: Tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {"name": self.name, "ok": self.ok, "reason": self.reason, "details": dict(self.details), "forced": self.forced,
                "words": list(self.words)}


def cannot_run(exc: BaseException) -> BaseException:
    """``exc`` marked as a cannot-run event (module contract (b)): ``raise gates.cannot_run(SomeError("<lever>: <why>"))``."""
    try:
        exc.cannot_run = True                      # type: ignore[attr-defined]
    except AttributeError:                         # an exception type without instance attributes: wrap nothing, the caller names it
        pass
    return exc


def is_cannot_run(exc: BaseException) -> bool:
    """True when ``exc`` is a cannot-run event: it, or its class, carries ``cannot_run = True``."""
    return bool(getattr(exc, "cannot_run", False))


def run_gates(checks: Sequence[Callable[[], Gate]]) -> list:
    """Run every check in order and return every gate (a check that raises becomes a refused gate naming the exception)."""
    out = []
    for check in checks:
        try:
            g = check()
        except Exception as e:  # noqa: BLE001
            g = Gate(name=getattr(check, "__name__", "gate"), ok=False, reason=f"{getattr(check, '__name__', 'gate')} raised {e!r}")
        out.append(g)
    return out


def first_refusal(gates: Iterable[Gate]) -> Optional[str]:
    """The reason of the first gate that refused and was not forced, or None."""
    for g in gates:
        if not g.ok and not g.forced:
            return g.reason or f"{g.name} refused"
    return None


def apply_force(gate: Gate, forced: bool) -> Gate:
    """A refused gate under the kit's force switch: ``ok`` stays False, ``forced`` becomes True (the report records the override)."""
    if forced and not gate.ok:
        gate.forced = True
    return gate


# ------------------------------------------------------------------------------------------------------------------ versions and pins


def dist_version(name: str) -> Optional[str]:
    """The installed distribution's version from its metadata (no import of the package), or None when it is not installed."""
    try:
        from importlib import metadata
        return metadata.version(name)
    except Exception:  # noqa: BLE001
        return None


def load_pins(path: str) -> dict:
    """The kit's pin document (JSON) — its own; the core reads it for the kit and interprets nothing."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def version_gate(pins: Mapping[str, str], found: Mapping[str, Optional[str]], *, force: bool = False, name: str = "versions",
                 stock: Optional[Iterable[str]] = None) -> Gate:
    """The pinned distributions against the versions on the box (``found``: the same names -> version, None = not installed). ``stock``
    names the pin entries that are the STOCK ENGINE's distribution(s): one of those at another version (or absent) refuses by name — the
    tree's patches and equality records are of that source. Every other entry (torch, triton, numpy, CUDA wheels …: the
    framework and toolchain) at another version passes with the word ``stack=drift(<dist>:<found>!=<pinned>,…)``. Details carry every name
    with ``pinned``, ``found`` and ``drift``; ``details["stock"]`` lists the stock entries."""
    stock_names = tuple(stock or ())
    details = {k: {"pinned": pins[k], "found": found.get(k), "drift": found.get(k) != pins[k]} for k in pins}
    details = dict(details, stock=list(stock_names))
    bad_stock = [k for k in pins if k in stock_names and found.get(k) != pins[k]]
    drift = [k for k in pins if k not in stock_names and found.get(k) != pins[k]]
    words = (word("stack", "drift", *[f"{k}:{found.get(k)}!={pins[k]}" for k in drift]),) if drift else ()
    if bad_stock:
        reason = "; ".join(f"{k}: pinned {pins[k]}, found {found.get(k)}" for k in bad_stock)
        return apply_force(Gate(name=name, ok=False, reason=f"version mismatch — {reason}", details=details, words=words), force)
    return Gate(name=name, ok=True, details=details, words=words)


# ------------------------------------------------------------------------------------------------------------------ the GPU

PROBE_KEYS = ("name", "cc", "sm", "memory_mib", "probe")


def _project(gpu: dict, keys: Optional[Sequence[str]]) -> dict:
    return gpu if keys is None else {k: gpu.get(k) for k in keys}


def nvidia_smi_probe(timeout_s: float = 20, index: int = 0, *, keys: Optional[Sequence[str]] = None) -> dict:
    """GPU equality without torch: name, compute capability, ``sm<digits>`` and memory (MiB) of GPU ``index`` from nvidia-smi.
    ``probe`` names the source (``nvidia-smi``) or why there is none; the other fields are then None. ``keys`` projects the dict to
    the kit's own keys."""
    empty = {"name": None, "cc": None, "sm": None, "memory_mib": None}
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=timeout_s)
    except Exception as e:  # noqa: BLE001
        return _project(dict(empty, probe=f"nvidia-smi unavailable ({type(e).__name__})"), keys)
    lines = [ln for ln in out.stdout.strip().splitlines() if ln.strip()]
    if out.returncode != 0 or len(lines) <= index:
        return _project(dict(empty, probe=f"nvidia-smi rc={out.returncode}"), keys)
    parts = [x.strip() for x in lines[index].split(",")]
    name = parts[0]
    cc = parts[1] if len(parts) > 1 and parts[1] else None
    mem = None
    if len(parts) > 2:
        try:
            mem = int(float(parts[2]))
        except ValueError:
            mem = None
    return _project({"name": name, "cc": cc, "sm": ("sm" + cc.replace(".", "")) if cc else None, "memory_mib": mem, "probe": "nvidia-smi"}, keys)


def torch_gpu_probe(index: int = 0, *, keys: Optional[Sequence[str]] = None) -> dict:
    """The same shape from torch (imports torch): the device properties of GPU ``index``; memory in MiB from ``total_memory``."""
    import torch
    if not torch.cuda.is_available():
        return _project({"name": None, "cc": None, "sm": None, "memory_mib": None, "probe": "torch: cuda not available"}, keys)
    p = torch.cuda.get_device_properties(index)
    cc = f"{p.major}.{p.minor}"
    return _project({"name": torch.cuda.get_device_name(index), "cc": cc, "sm": f"sm{p.major}{p.minor}",
                     "memory_mib": int(p.total_memory // (1024 * 1024)), "probe": "torch"}, keys)


def _mib(mem) -> str:
    return "memory_unknown" if mem is None else f"{int(mem)}MiB"


def _unprobed(gpu: Mapping) -> str:
    """``card=unprobed(<why>)``: no card name could be read (the probe's reason)."""
    return word("card", "unprobed", gpu.get("probe") or "no_probe")


def gpu_name_check(gpu: Optional[Mapping], contains: Optional[str], *, name: str = "gpu", fold_case: bool = False) -> Gate:
    """The card name contains ``contains`` (substring; case-sensitive unless ``fold_case`` — the kit passes the form of its own comparison;
    None = no requirement). Always ok: another card carries ``card=uncertified(<name>)``, no card read ``card=unprobed(<why>)``;
    ``details["match"]`` is the comparison's result (None without a requirement or a name)."""
    gpu = dict(gpu or {})
    details = {"name": gpu.get("name"), "cc": gpu.get("cc"), "probe": gpu.get("probe"), "required": contains, "fold_case": bool(fold_case), "match": None}
    if not gpu.get("name"):
        return Gate(name=name, ok=True, details=details, words=(_unprobed(gpu),) if contains else ())
    if not contains:
        return Gate(name=name, ok=True, details=details)
    found, want = (gpu["name"].lower(), contains.lower()) if fold_case else (gpu["name"], contains)
    details["match"] = want in found
    words = () if details["match"] else (word("card", "uncertified", gpu["name"]),)
    return Gate(name=name, ok=True, details=details, words=words)


def gpu_cc_check(gpu: Optional[Mapping], cc: Optional[str], *, name: str = "gpu_cc") -> Gate:
    """The compute capability equals ``cc`` (``"9.0"``; None = no requirement). Always ok: another capability carries
    ``card=unlisted-cc(<cc>)`` (``none`` when it could not be read); ``details["match"]`` is the comparison's result."""
    gpu = dict(gpu or {})
    details = {"name": gpu.get("name"), "cc": gpu.get("cc"), "required": cc, "match": None}
    if cc is None:
        return Gate(name=name, ok=True, details=details)
    details["match"] = gpu.get("cc") == cc
    words = () if details["match"] else (word("card", "unlisted-cc", gpu.get("cc")),)
    return Gate(name=name, ok=True, details=details, words=words)


# the GPU classes a kit may name as its target: the card AND its memory size, both measured on a box of that class.
# ``memory_mib`` = the memory size the class is named by; ``memory_mib_members`` (when present) = EVERY memory size the class admits — the A100 class is
# one class by compute capability (8.0) with two parts, 80 GB (SXM4 / PCIe, 81920 MiB) and 40 GB (40960 MiB). A kit's memory-dependent choice
# keys off the PROBED memory (``gpu_memory_member`` / ``arch.device_memory``), never off the class name.
GPU_CLASSES = {
    "H100": {"name_contains": "H100", "memory_mib": 81559, "cc": "9.0"},                                          # NVIDIA H100 80GB HBM3
    "A100": {"name_contains": "A100", "memory_mib": 81920, "cc": "8.0", "memory_mib_members": (81920, 40960)},     # NVIDIA A100: the 80 GB and the 40 GB parts
    "A100_80GB": {"name_contains": "A100", "memory_mib": 81920, "cc": "8.0"},                                     # the 80 GB part only (SXM4 or PCIe)
    "A100_40GB": {"name_contains": "A100", "memory_mib": 40960, "cc": "8.0"},                                     # the 40 GB part only (SXM4 or PCIe)
    "H200": {"name_contains": "H200", "memory_mib": 143771, "cc": "9.0"},                                         # NVIDIA H200 (141 GB; compute capability 9.0 = the H100's key in every cell table — a row here only names the card: no lookup of another class reads it)
}
GPU_MEMORY_TOLERANCE_MIB = 1024


def class_memory_members(cls: Mapping) -> Tuple[int, ...]:
    """The memory sizes (MiB) a GPU_CLASSES row admits: its ``memory_mib_members`` when present, else its one ``memory_mib``."""
    members = cls.get("memory_mib_members")
    return tuple(int(m) for m in members) if members else (int(cls["memory_mib"]),)


def _memory_member(cls: Mapping, mem, tolerance_mib: int) -> Optional[int]:
    """The member of ``cls`` the probed ``mem`` (MiB) is within ``tolerance_mib`` of (the nearest when two qualify), else None."""
    if mem is None:
        return None
    near = [m for m in class_memory_members(cls) if abs(int(mem) - m) <= tolerance_mib]
    return min(near, key=lambda m: abs(int(mem) - m)) if near else None


def gpu_memory_member(gpu: Optional[Mapping], target: str, *, tolerance_mib: int = GPU_MEMORY_TOLERANCE_MIB,
                      classes: Mapping[str, Mapping] = GPU_CLASSES) -> Optional[int]:
    """The memory member (MiB) of class ``target`` the PROBED card is — the key a kit writes its memory-dependent rows against (81920 or 40960
    for A100): None when the class is unknown, no GPU, the name is not the class's, or the probed memory matches no member."""
    gpu = dict(gpu or {})
    cls = classes.get(target)
    if cls is None or not gpu.get("name") or cls["name_contains"] not in gpu["name"]:
        return None
    return _memory_member(cls, gpu.get("memory_mib"), tolerance_mib)


def gpu_class_check(gpu: Optional[Mapping], target: Optional[str], *, tolerance_mib: int = GPU_MEMORY_TOLERANCE_MIB,
                    classes: Mapping[str, Mapping] = GPU_CLASSES, name: str = "gpu_class") -> Gate:
    """The box's GPU against the named class: the name contains the class's token AND the memory size is within ``tolerance_mib`` of one
    of the class's members (a name substring alone accepts another-memory variant of the card). Always ok — a card outside the class is
    where no engine was measured, named, not refused: ``card=uncertified(<name>,<MiB>MiB)`` for another name or another memory size,
    ``card=uncertified(<name>,memory_unknown)`` when the size could not be read, ``card=unprobed(<why>)`` without a card name,
    ``target=unlisted(<target>)`` for a class this table does not know. No target: ok, nothing compared. ``details["match"]`` is the
    comparison's result (None: no target, unknown target, or no card); ``details["memory_member"]`` is the member matched (None otherwise) —
    the key a memory-dependent choice reads (:func:`gpu_memory_member`)."""
    gpu = dict(gpu or {})
    details = {"target": target, "name": gpu.get("name"), "memory_mib": gpu.get("memory_mib"), "cc": gpu.get("cc"), "probe": gpu.get("probe"),
               "match": None, "memory_member": None}
    if not target:
        return Gate(name=name, ok=True, details=details)
    cls = classes.get(target)
    if cls is None:
        return Gate(name=name, ok=True, details=dict(details, known=sorted(classes)), words=(word("target", "unlisted", target),))
    if not gpu.get("name"):
        return Gate(name=name, ok=True, details=details, words=(_unprobed(gpu),))
    mem = gpu.get("memory_mib")
    member = _memory_member(cls, mem, tolerance_mib) if cls["name_contains"] in gpu["name"] else None
    details["match"] = member is not None
    details["memory_member"] = member
    if member is not None:
        return Gate(name=name, ok=True, details=details)
    return Gate(name=name, ok=True, details=details, words=(word("card", "uncertified", gpu["name"], _mib(mem)),))


# ------------------------------------------------------------------------------------------------------------------ bytes: sums and the tree rule

CARRY_MANIFEST = "SHA256SUMS"           # opt/SHA256SUMS: every carried kit file under opt/ with its sha256, paths relative to opt/


def sha256_file(path: str) -> str:
    """sha256 (hex) of one file, read in chunks so a large file never loads whole into memory. Re-exported as
    ``opt_core.extern.digest_file`` for out-of-git artifacts; here for ``verify_sums``/``carry_gate`` and
    ``opt_core.compare``'s output-tree digests (outputs are not in git -- a legitimate, non-manifest use)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_sums(path: str) -> dict:
    """A ``<sha256>  <relpath>`` manifest (sha256sum format; blank lines and ``#`` comments skipped) as {relpath: sha256}."""
    out: dict = {}
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split(None, 1)
            if len(parts) != 2 or len(parts[0]) != 64:
                raise ValueError(f"{path}: not a '<sha256>  <path>' line: {ln!r}")
            out[parts[1].strip()] = parts[0].lower()
    return out


def verify_sums(root: str, sums: Mapping[str, str], subset: Optional[Iterable[str]] = None, *, name: str = "sums") -> Gate:
    """Every listed file (or the ``subset`` of paths) under ``root`` re-hashes to its digest. Details: ``checked``, ``missing``,
    ``mismatched`` (relpath -> {expected, found}; a subset path the manifest does not list is a mismatch with expected None), ``listed``."""
    paths = list(subset) if subset is not None else list(sums)
    missing, mismatched, checked = [], {}, 0
    for rel in paths:
        if rel not in sums:
            mismatched[rel] = {"expected": None, "found": None}
            continue
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            missing.append(rel)
            continue
        found = sha256_file(p)
        checked += 1
        if found != sums[rel]:
            mismatched[rel] = {"expected": sums[rel], "found": found}
    details = {"checked": checked, "missing": missing, "mismatched": mismatched, "listed": len(sums)}
    if missing or mismatched:
        return Gate(name=name, ok=False, reason=f"{len(missing)} missing, {len(mismatched)} mismatched of {len(paths)} listed under {root}", details=details)
    return Gate(name=name, ok=True, details=details)


def carry_manifest_path(opt_dir: str) -> str:
    """``<opt_dir>/SHA256SUMS`` — the one spelling of a kit's carry manifest."""
    return os.path.join(opt_dir, CARRY_MANIFEST)


def carry_gate(opt_dir: str, subset: Optional[Iterable[str]] = None) -> Gate:
    """The carried kit bytes of ``opt_dir`` against ``opt/SHA256SUMS`` (all entries, or ``subset`` — the files a mode needs)."""
    path = carry_manifest_path(opt_dir)
    if not os.path.isfile(path):
        return Gate(name="carry", ok=False, reason=f"carry manifest missing: {path}", details={"path": path})
    g = verify_sums(opt_dir, read_sums(path), subset, name="carry")
    g.details["path"] = path
    return g


# ------------------------------------------------------------------------------------------------------------ shipped binaries

BINARY_SUMS = CARRY_MANIFEST            # the same spelling inside the core: a compiled binary the package ships is a line of a SHA256SUMS in
                                        # its own directory or in a directory above it (the nearest that lists it), relative to that directory
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_SUMS_READ: dict = {}                   # SHA256SUMS path -> {relpath: sha256} ({} when unreadable), read once per process
_BINARY_VERDICTS: dict = {}             # binary path -> None (listed, digest equal) | the refusal reason; hashed once per process
BINARIES_HELD: list = []                # [(path relative to the package, sha256[:16])] of every binary that passed, in load order


def binary_sums_for(path: str) -> Optional[Tuple[str, Optional[str]]]:
    """``(SHA256SUMS path, entry key)`` for the binary at ``path``: the nearest SHA256SUMS — the file's own directory first, then each
    directory above it up to the package directory — that lists the file (the key is the file's path relative to that directory,
    ``/``-separated). When none lists it: ``(nearest SHA256SUMS, None)``, or ``None`` when there is no SHA256SUMS at all. For a path
    outside the package only its own directory is consulted."""
    p = os.path.abspath(path)
    d = os.path.dirname(p)
    inside = (d + os.sep).startswith(_PACKAGE_DIR + os.sep)
    nearest = None
    while True:
        cand = os.path.join(d, BINARY_SUMS)
        if os.path.isfile(cand):
            key = os.path.relpath(p, d).replace(os.sep, "/")
            if key in _sums_read(cand):
                return cand, key
            nearest = nearest or cand
        parent = os.path.dirname(d)
        if not inside or d == _PACKAGE_DIR or parent == d:
            return (nearest, None) if nearest else None
        d = parent


def _sums_read(sums_path: str) -> dict:
    doc = _SUMS_READ.get(sums_path)
    if doc is None:
        try:
            doc = read_sums(sums_path)
        except (OSError, ValueError):
            doc = {}
        _SUMS_READ[sums_path] = doc
    return doc


def binary_refusal(path: str, data: Optional[bytes] = None) -> Optional[str]:
    """``None`` when the compiled binary at ``path`` is a line of its SHA256SUMS (:func:`binary_sums_for`) and its bytes re-hash to that
    line (``data``: the bytes the caller already read, hashed in place of the file). Otherwise the reason, one clause — no SHA256SUMS,
    not listed, unreadable, or the digest differing — after one ``[opt_core] SHA256SUMS: refused <path>: <reason>`` line on stderr; the
    caller then refuses the binary by name exactly as it refuses an absent one. A path outside the package whose directory carries no
    SHA256SUMS is not a shipped binary: ``None``, unchecked. One verdict per path per process."""
    p = os.path.abspath(path)
    if p in _BINARY_VERDICTS:
        return _BINARY_VERDICTS[p]
    inside = p.startswith(_PACKAGE_DIR + os.sep)
    rel = os.path.relpath(p, _PACKAGE_DIR).replace(os.sep, "/") if inside else p
    found = binary_sums_for(p)
    reason: Optional[str] = None
    digest = None
    if found is None:
        if inside:
            reason = f"no {BINARY_SUMS} in or above its directory lists it"
    else:
        sums_path, key = found
        want = _sums_read(sums_path).get(key) if key is not None else None
        srel = os.path.relpath(sums_path, _PACKAGE_DIR).replace(os.sep, "/") if inside else sums_path
        if want is None:
            reason = f"not listed in {srel}"
        else:
            try:
                digest = hashlib.sha256(data).hexdigest() if data is not None else sha256_file(p)
            except OSError as e:
                reason = f"unreadable ({type(e).__name__})"
            else:
                if digest != want:
                    reason = f"sha256 {digest[:16]} != {srel} {want[:16]}"
    _BINARY_VERDICTS[p] = reason
    if reason is None:
        if digest is not None:
            BINARIES_HELD.append((rel, digest[:16]))
    else:
        import sys  # noqa: PLC0415
        print(f"[opt_core] {BINARY_SUMS}: refused {rel}: {reason}", file=sys.stderr, flush=True)
    return reason


# ------------------------------------------------------------------------------------------------------------------ the core pin

PIN_TABLE = "opt_core"          # [tool.opt_core] in a kit's pyproject.toml
PIN_KEYS = ("path", "version")               # version: the MINIMUM core version the kit needs
_PIN_LINE = re.compile(r'^\s*(path|version)\s*=\s*"([^"]*)"')


def version_tuple(text) -> tuple:
    """``"0.5.17.4"`` -> ``(0, 5, 17, 4)``; a non-numeric component counts as -1 (older than any release)."""
    return tuple(int(p) if p.isdigit() else -1 for p in str(text or "").strip().split("."))


def read_pin_table(pyproject_path: str) -> dict:
    """The ``[tool.opt_core]`` table of a pyproject (``tomllib`` when available, else a reader of that one table's string keys)."""
    try:
        import tomllib
    except ModuleNotFoundError:
        tomllib = None
    if tomllib is not None:
        with open(pyproject_path, "rb") as fh:
            return dict(tomllib.load(fh).get("tool", {}).get(PIN_TABLE, {}))
    out, inside = {}, False
    with open(pyproject_path, "r", encoding="utf-8") as fh:
        for ln in fh:
            s = ln.split("#", 1)[0].strip()                 # a trailing comment after a table header is legal TOML
            if s.startswith("["):
                inside = s.replace(" ", "") == f"[tool.{PIN_TABLE}]"
                continue
            m = _PIN_LINE.match(ln) if inside else None
            if m:
                out[m.group(1)] = m.group(2)
    return out


def core_pin(pyproject_path: str) -> dict:
    """The kit's pin: ``path`` (relative to the pyproject; ``abs_path`` resolved) and ``version`` (the minimum core version). Both keys are required."""
    pin = read_pin_table(pyproject_path)
    missing = [k for k in PIN_KEYS if not pin.get(k)]
    if missing:
        raise ValueError(f"{pyproject_path}: [tool.{PIN_TABLE}] lacks {', '.join(missing)}")
    pin["abs_path"] = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(pyproject_path)), pin["path"]))
    return pin


def imported_core() -> dict:
    """The ``opt_core`` this process imports: its version and package directory."""
    import opt_core
    pkg_dir = os.path.dirname(os.path.abspath(opt_core.__file__))
    return {"version": opt_core.__version__, "package_dir": pkg_dir}


def core_pin_check(pyproject_path: str, *, force: bool = False) -> Gate:
    """The kit's pin against the imported core: the imported version must be at least the pinned one. An older core refuses by name
    (a core below the kit's minimum lacks names the kit calls); ``force`` records the override. A pin that cannot be read (no such
    file, no ``[tool.opt_core]`` table, a key missing) passes with the word ``core_pin=unreadable`` and ``details["unreadable"]`` = why:
    the imported core serves unpinned. Details carry both sides."""
    core = imported_core()
    try:
        pin = core_pin(pyproject_path)
    except Exception as e:  # noqa: BLE001
        return Gate(name="opt_core_pin", ok=True, details={"pinned": None, "imported": core, "unreadable": str(e)},
                    words=(word("core_pin", "unreadable"),))
    details = {"pinned": {k: pin[k] for k in PIN_KEYS}, "pinned_abs_path": pin["abs_path"], "imported": core}
    if version_tuple(core["version"]) < version_tuple(pin["version"]):
        g = Gate(name="opt_core_pin", ok=False, details=details,
                 reason=f"opt_core pinned >= v{pin['version']} at {pin['path']}, imported v{core['version']} from {core['package_dir']}")
        return apply_force(g, force)
    return Gate(name="opt_core_pin", ok=True, details=details)
