"""stack.py — the activation: probe, gate, arm, engage, report. What `enable()` / `apply()` / `check()` / `disable()` and the autoload run.

Gates (a refusal by name, `[gpnstar-opt] NOT ACTIVE: <reason>`, exit 3 under the switch): no CUDA device visible; the installed `gpn` is
not the pinned commit — a VCS record off the pin, or installed files that differ from the carried stock archive (an install from that archive,
byte-identical, IS the pinned commit: gpn=<commit8> as for a git install) — the levers patch that source; `transformers` is not upstream's own
pinned version; the tree's stock/PINS.json disagrees with the lever tree's tables. Named, never refused (words on the ACTIVE line): a card of
no tested class, a dependency off its pin, a model outside the pinned table or without digests to compare.

Engage: the levers apply to a model that is ALREADY on the GPU, before its first forward — `apply(model)` for a caller's own model, or the
arm: `activate(arm=True)` wraps the constructors of upstream's three inference wrapper modules (gpn.star.inference: MLMforVEPModel,
MLMforLogitsModel, ModelCenterEmbedding) so that each instance, once built, has its `.model` placed on the GPU and the levers applied there and
then (a wrapper built without a GPU visible engages at its first forward instead, through a pre-hook); ACTIVE + LEVER are printed, and the KV
reporter (a forward hook that prints one KV line per new batch shape from the levers' own route report, and the closing KV line at exit) is
installed. A model on the CPU at that moment, a target row count other than one or a target species other than 0, a weights file off its
digest: NOT ACTIVE and the process ends with exit 3 — stock never runs under the switch. The levers set no numerics flag: the fp32-matmul
TF32 setting the caller runs under (`gpn star … --tf32`, `torch.backends.cuda.matmul`) is followed and named on the ACTIVE line (tf32=on|off).
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import subprocess
import sys
from importlib import metadata
from typing import Dict, List, Optional

from opt_core import report as core

from . import registry, report
from ._names import EXIT_NOT_ACTIVE, LEVERS, MODE, TAG

CARDS = {"NVIDIA H100 80GB HBM3": ("h100", 81559), "NVIDIA H100 PCIe": ("h100", 81559), "NVIDIA A100-SXM4-80GB": ("a100", 81920),
         "NVIDIA A100 80GB PCIe": ("a100", 81920), "NVIDIA A100-SXM4-40GB": ("a100", 40960), "NVIDIA A100-PCIE-40GB": ("a100", 40960)}
CARD_CAPABILITY = {("9.0", 81559): "h100", ("8.0", 81920): "a100", ("8.0", 40960): "a100"}   # (compute capability, memory.total MiB) -> class: the gate is by capability and memory, ±CARD_MIB_TOLERANCE; stock/PINS.json gpus, when readable, is read INSTEAD (pins_card_table): one source
CARD_MIB_TOLERANCE = 0.03
STACK_PINS = ("torch", "transformers", "numpy", "huggingface_hub", "safetensors", "accelerate")   # compared against stock/PINS.json "pins"


class ActivationError(RuntimeError):
    """The mode could not be activated (a gate refused, or the levers refused at engage)."""


def _fresh_state(exit_registered: bool = False) -> Dict:
    return {"report": None, "armed": False, "patches": [], "engaged": [], "handles": [], "kv": {k: 0 for k in report.KV_ROUTES}, "kv_started": {}, "kv_reason": {},
            "kv_first_ms": {}, "kv_last": None, "kvcheck": {"pairs": 0, "rejects": 0}, "seen_shapes": {}, "exit_registered": exit_registered, "compile_line_done": False}


_STATE: Dict = _fresh_state()


# ----------------------------------------------------------------------------------------------------------------- probes
# ----------------------------------------------------------------------------------------------------------------- probes
def probe_gpu_nvidia_smi() -> Optional[dict]:
    """{name, mib, cc, sm, driver} of GPU 0 from nvidia-smi (torch-free); None when there is no device or no tool."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    vis = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if vis in ("-1",) or (vis == "" and "CUDA_VISIBLE_DEVICES" in os.environ):
        return None
    row = [c.strip() for c in out.stdout.strip().splitlines()[0].split(",")]
    if len(row) < 4:
        return None
    name, mib, cc, driver = row[0], row[1], row[2], row[3]
    try:
        mib = int(float(mib))
    except ValueError:
        mib = None
    return {"name": name, "mib": mib, "cc": cc, "sm": "sm" + cc.replace(".", ""), "driver": driver, "source": "nvidia-smi"}


def probe_gpu_torch() -> Optional[dict]:
    """The same record from torch when torch is already imported in this process (never imported here for the probe)."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        p = torch.cuda.get_device_properties(0)
    except Exception:  # noqa: BLE001
        return None
    cc = f"{p.major}.{p.minor}"
    return {"name": p.name, "mib": int(p.total_memory // (1024 * 1024)), "cc": cc, "sm": f"sm{p.major}{p.minor}", "driver": None, "source": "torch"}


def probe_gpu() -> Optional[dict]:
    return probe_gpu_torch() or probe_gpu_nvidia_smi()


def pins_card_table(pins: Optional[dict] = None):
    """({name: class}, {(cc, MiB): class}) from stock/PINS.json gpus ({key: {name, sm: sm_NN, memory_mib, [class], [also: names]}}); the
    built-in tables when the file or the block is absent."""
    try:
        g = (pins if pins is not None else registry.load_pins()).get("gpus") or {}
    except (OSError, ValueError):
        g = {}
    if not g:
        return dict((n, c[0]) for n, c in CARDS.items()), dict(CARD_CAPABILITY)
    names, caps = {}, {}
    for key, ent in g.items():
        if not isinstance(ent, dict) or not ent.get("sm"):
            continue
        cls = ent.get("class") or key
        digits = str(ent["sm"]).replace("sm_", "").replace("sm", "")
        cc = f"{digits[:-1]}.{digits[-1]}" if len(digits) >= 2 else digits
        if ent.get("memory_mib") is not None:
            caps[(cc, int(ent["memory_mib"]))] = cls
        for n in [ent.get("name")] + list(ent.get("also") or []):
            if n:
                names[n] = cls
    return names, caps


def card_gate(gpu: Optional[dict], environ=None, pins: Optional[dict] = None) -> dict:
    """{class, words}: the card's class (h100 | a100 | None) by name, else by (capability, memory), from stock/PINS.json gpus; a card of no
    listed class is a word (card=untested(...)), never a refusal. ``environ`` is unused (kept for the call form)."""
    if not gpu:
        return {"class": None, "words": []}
    words = []
    cls = None
    name, mib, cc = gpu.get("name"), gpu.get("mib"), gpu.get("cc")
    names, caps = pins_card_table(pins)
    if name in names:
        cls = names[name]
    else:
        for (want_cc, want_mib), c in caps.items():
            if cc == want_cc and mib and abs(mib - want_mib) <= CARD_MIB_TOLERANCE * want_mib:
                cls = c
                break
    if cls is None:
        words.append(core.word("card", "untested", report.token(name), f"{mib}MiB", gpu.get("sm")))
    return {"class": cls, "words": words}


def dist_version(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def archive_sources(archive_path: str, package_dir: str = "src/gpn") -> dict:
    """{path relative to site-packages (gpn/...): sha256} of every file under <prefix>/<package_dir> in the carried stock archive (stdlib tarfile)."""
    import tarfile
    out = {}
    with tarfile.open(archive_path, "r:gz") as t:
        for m in t.getmembers():
            if not m.isfile():
                continue
            parts = m.name.split("/", 1)
            if len(parts) < 2 or not parts[1].startswith(package_dir + "/"):
                continue
            top = package_dir.rstrip("/").split("/")[-1]
            out[top + "/" + parts[1][len(package_dir) + 1:]] = hashlib.sha256(t.extractfile(m).read()).hexdigest()
    return out


def stock_archive_path(pins: Optional[dict] = None) -> str:
    """The carried stock archive: stock/<basename of PINS upstream.gpn.archive> in the tree."""
    up = ((pins or {}).get("upstream") or {}).get("gpn") or {}
    name = os.path.basename(up.get("archive") or ((pins or {}).get("archive") or {}).get("file") or "gpn-stock.tar.gz")
    return os.path.join(registry.tree_home(), "stock", name)


def archive_identity(dist, pins: Optional[dict], archive_path: Optional[str] = None) -> dict:
    """Is the installed gpn THE pinned stock by content? {ok, reason, n_files, archive}: ok when the carried archive is present, its sha256 is
    PINS archive.sha256, and every package file in it is present in the installed distribution with identical bytes (the install step's own
    comparison). reason names what could not be shown (archive absent / off its digest: 'unshown:…') or what differs ('differs:…')."""
    pins = pins or {}
    archive = archive_path or stock_archive_path(pins)
    rec = {"ok": False, "reason": None, "n_files": 0, "archive": archive, "modified": [], "missing": []}
    want_sha = (pins.get("archive") or {}).get("sha256")
    if not os.path.isfile(archive):
        rec["reason"] = f"unshown:archive_absent:{os.path.basename(archive)}"
        return rec
    if not want_sha:
        rec["reason"] = "unshown:no_archive_digest_in_pins"
        return rec
    got = sha256_of(archive)
    if got != want_sha:
        rec["reason"] = f"unshown:archive_sha256:{got[:12]}!={want_sha[:12]}"
        return rec
    package_dir = (((pins.get("upstream") or {}).get("gpn") or {}).get("package_dir")) or "src/gpn"
    try:
        sources = archive_sources(archive, package_dir)
    except (OSError, ValueError) as e:  # tarfile errors are OSError / ValueError subclasses or carry them
        rec["reason"] = f"unshown:archive_unreadable:{type(e).__name__}"
        return rec
    except Exception as e:  # noqa: BLE001
        rec["reason"] = f"unshown:archive_unreadable:{type(e).__name__}"
        return rec
    if not sources:
        rec["reason"] = "unshown:no_package_files_in_archive"
        return rec
    for rel, want in sorted(sources.items()):
        p = str(dist.locate_file(rel))
        if not os.path.exists(p):
            rec["missing"].append(rel)
        elif sha256_of(p) != want:
            rec["modified"].append(rel)
        else:
            rec["n_files"] += 1
    if rec["modified"] or rec["missing"]:
        rec["reason"] = "differs:modified=" + ",".join(rec["modified"][:5]) + (";missing=" + ",".join(rec["missing"][:5]) if rec["missing"] else "")
        return rec
    rec["ok"] = True
    return rec


def gpn_install(pins: Optional[dict] = None, archive_path: Optional[str] = None) -> dict:
    """{version, commit, route, detail, archive}: the installed gpn distribution and the commit it stands for. A VCS install names its commit
    in its PEP 610 record (route 'vcs'). Any other install (the kit's own recipe installs the byte-frozen stock archive: a file URL, no
    vcs_info) IS the pinned commit when its installed modules are byte-identical to the carried archive at PINS archive.sha256 — then commit =
    PINS upstream.gpn.commit and route 'archive'. Only when neither can be shown is commit None (route says why); modified files: route 'differs'."""
    rec = {"version": dist_version("gpn"), "commit": None, "route": None, "detail": None, "archive": None}
    try:
        d = metadata.distribution("gpn")
    except metadata.PackageNotFoundError:
        return rec
    if pins is None:
        try:
            pins = registry.load_pins()
        except (OSError, ValueError):
            pins = {}
    raw = None
    try:
        raw = d.read_text("direct_url.json")
    except (OSError, ValueError):
        raw = None
    kind, url = "no_direct_url", None
    if raw:
        try:
            j = json.loads(raw)
        except ValueError:
            j = {}
        url = j.get("url")
        if "vcs_info" in j and (j["vcs_info"] or {}).get("commit_id"):
            rec["commit"] = j["vcs_info"]["commit_id"]
            rec["route"] = "vcs"
            rec["detail"] = url
            return rec
        kind = "archive_url" if "archive_info" in j else ("dir_url" if "dir_info" in j else "unknown_direct_url")
    ident = archive_identity(d, pins, archive_path)
    rec["archive"] = ident
    if ident["ok"]:
        rec["commit"] = (((pins.get("upstream") or {}).get("gpn") or {}).get("commit"))
        rec["route"] = "archive"
        rec["detail"] = f"{kind}; {ident['n_files']} installed files byte-identical to {os.path.basename(ident['archive'])}"
    elif (ident["reason"] or "").startswith("differs:"):
        rec["route"] = "differs"
        rec["detail"] = ident["reason"]
    else:
        rec["route"] = f"{kind}:{ident['reason']}"
        rec["detail"] = url
    return rec


def gpn_gate(g: dict, pins: dict, want_commit: str) -> tuple:
    """(refusals, words) of the gpn install probe against the pin: absent → refused; a commit (vcs or archive-identical) off the pin → refused;
    installed files differing from the stock archive → refused (another stock); no commit shown and nothing to compare → version off the pin
    refused, else the word gpn_commit=unrecorded(<why>) — engaged and named."""
    refusals, words = [], []
    up = ((pins.get("upstream") or {}).get("gpn") or {})
    if g.get("version") is None:
        refusals.append("gpn is not installed in this environment (stock/PINS.json upstream.gpn.install)")
    elif g.get("route") == "differs":
        refusals.append(f"the installed gpn {g['version']} differs from the stock archive ({g.get('detail')}): another stock (stock/PINS.json upstream.gpn)")
    elif g.get("commit") is None:
        if up.get("version") and g["version"] != up["version"]:
            refusals.append(f"the installed gpn is {g['version']}, the pin is {up['version']} @ {want_commit[:8]} (no commit record and no archive identity: {g.get('route')})")
        else:
            words.append(core.word("gpn_commit", "unrecorded", report.token(g.get("route"))))
    elif g["commit"] != want_commit:
        refusals.append(f"the installed gpn is commit {g['commit'][:12]}, the levers are written against {want_commit[:12]} (stock/PINS.json upstream.gpn)")
    return refusals, words


def stack_versions() -> dict:
    v = {n: dist_version(n) for n in STACK_PINS}
    v["gpn"] = dist_version("gpn")
    return v


# ------------------------------------------------------------------------------------------------------------------ gates
def gates(*, need_gpu: bool = True, environ=None) -> dict:
    """{refusals: [reason], words: [word], gpu, card, versions, gpn, pins_ok} — every gate of an activation, none applied."""
    environ = os.environ if environ is None else environ
    refusals: List[str] = []
    words: List[str] = []
    gpu = probe_gpu() if need_gpu else None
    if need_gpu and gpu is None:
        refusals.append("no CUDA device is visible (nvidia-smi / torch report none): the levers run on the GPU")
    try:
        pins = registry.load_pins()
    except (OSError, ValueError) as e:
        pins = {}
        words.append(core.word("pins", "unreadable", type(e).__name__))
    card = card_gate(gpu, environ, pins or None)
    words += card["words"]
    want_commit = registry.gpn_commit()
    g = gpn_install(pins)
    r_gpn, w_gpn = gpn_gate(g, pins, want_commit)
    refusals += r_gpn
    words += w_gpn
    versions = stack_versions()
    want = dict(pins.get("pins") or {})
    refuse_on = pins.get("refuse_on") if isinstance(pins.get("refuse_on"), dict) else {}
    drift = []
    for n in STACK_PINS:
        have, w = versions.get(n), want.get(n)
        if not w:
            continue
        base = (have or "").split("+")[0]
        if have is None or base != str(w).split("+")[0]:
            if n == "transformers" or refuse_on.get(n) == "version":
                refusals.append(f"{n} is {have or 'not installed'}, upstream pins {w} (stock/PINS.json pins.{n})")
            else:
                drift.append(f"{n}:{have or 'absent'}!={w}")
    if drift:
        words.append(core.word("stack", "drift", *drift))
    bad = registry.cross_check(pins) if pins else []
    if bad:
        refusals.append("stock/PINS.json disagrees with the lever tree: " + "; ".join(bad))
    return {"refusals": refusals, "words": words, "gpu": gpu, "card": card["class"], "versions": versions, "gpn": g, "pins_ok": not drift and not bad,
            "gpn_commit8": (g["commit"] or want_commit)[:8] if g["version"] else want_commit[:8]}


def tf32_state() -> Optional[bool]:
    """The process's TF32 setting for fp32 CUDA matmuls, READ (never set): the precision interface first
    (`torch.backends.cuda.matmul.fp32_precision`, else the generic `torch.backends.fp32_precision`: "tf32" | "ieee"; "none" = unset there),
    and only when that names nothing the legacy flag `torch.backends.cuda.matmul.allow_tf32`, inside a guard — once code in the process has
    set the precision interface (transformers' `--tf32` does), touching the legacy flag raises. None when torch is not imported or no
    interface answers. The levers follow this setting; they never change it."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    for owner in (getattr(getattr(torch.backends, "cuda", None), "matmul", None), torch.backends):
        try:
            p = getattr(owner, "fp32_precision", None) if owner is not None else None
        except Exception:  # noqa: BLE001
            p = None
        if p is not None and str(p) in ("tf32", "ieee"):
            return str(p) == "tf32"
    try:
        return bool(torch.backends.cuda.matmul.allow_tf32)
    except Exception:  # noqa: BLE001 -- a process in a mixed-interface state: not known, not guessed
        return None


# -------------------------------------------------------------------------------------------------------------- activation
def status() -> dict:
    rep = _STATE["report"]
    if rep is None:
        return {"active": False, "reason": "enable() has not run in this process"}
    out = dict(rep)
    out["kv"], out["kvcheck"], out["engaged"] = dict(_STATE["kv"]), dict(_STATE["kvcheck"]), [dict((k, v) for k, v in r.items() if k != "model") for r in _STATE["engaged"]]
    return out


def _base_report(g: dict) -> dict:
    return {"active": False, "mode": MODE, "word": MODE, "levers": list(LEVERS), "gpu": g.get("gpu"), "card": g.get("card") or ("untested" if g.get("gpu") else None),
            "mib": (g.get("gpu") or {}).get("mib"), "torch": (g.get("versions") or {}).get("torch"), "transformers": (g.get("versions") or {}).get("transformers"),
            "gpn_commit8": g.get("gpn_commit8"), "words": list(g.get("words") or []), "model_label": None, "model_key": None, "reason": None, "tf32": None}


def activate(*, dry_run: bool = False, strict: bool = False, trigger: Optional[str] = None, arm: bool = True, environ=None) -> dict:
    """Gate and (unless dry_run) arm. Prints ONE line now when there is something to say: NOT ACTIVE, or DRY-RUN; an active arm prints its
    ACTIVE + LEVER lines when the levers engage (the model is not built yet). Idempotent per process: a second call returns the first report."""
    environ = os.environ if environ is None else environ
    prev = _STATE["report"]
    if prev is not None and not dry_run:
        if prev.get("active"):
            if arm and not _STATE["armed"]:
                _arm()
            return prev
        if strict:
            raise ActivationError(prev.get("reason") or "not active")
        return prev
    g = gates(environ=environ)
    rep = _base_report(g)
    rep["trigger"] = trigger
    if dry_run:
        rep["dry_run"] = True
        rep["would_refuse"] = "; ".join(g["refusals"]) or None
        _dry_run_weights(rep)
        report.emit(report.dry_run_line(rep))
        return rep
    if g["refusals"]:
        rep["reason"] = "; ".join(g["refusals"])
        report.emit(report.not_active_line(rep["reason"]))
        _STATE["report"] = rep
        if strict:
            raise ActivationError(rep["reason"])
        return rep
    rep["active"] = True
    _STATE["report"] = rep
    if arm:
        _arm()
    return rep


def _dry_run_weights(rep: dict) -> None:
    """The dry run's weights word: with the primary model's snapshot under HF_HOME and digests pinned for it, hash it; else name why not."""
    key = registry.PRIMARY
    ck = registry.checkpoint(key)
    snap = registry.snapshot_dir(key)
    if not os.path.isdir(snap):
        rep["words"].append(core.word("weights", "unchecked", f"no_snapshot:{key}"))
        return
    bad = weights_mismatch(ck, snap)
    if bad is None:
        rep["words"].append(core.word("weights", "unchecked", f"no_digests:{key}"))
    elif bad:
        rep["would_refuse"] = ((rep.get("would_refuse") + "; ") if rep.get("would_refuse") else "") + "weights off their digests: " + "; ".join(bad)
    else:
        rep["words"].append(core.word("weights", "ok", key))


def sha256_of(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def weights_mismatch(ck: dict, snap: str) -> Optional[List[str]]:
    """None when the checkpoint has no pinned digests; else the list of files absent or off their digest under `snap` ([] = all match)."""
    if not ck.get("files"):
        return None
    bad = []
    for rel, ent in ck["files"].items():
        p = os.path.join(snap, rel)
        if not os.path.isfile(p):
            bad.append(f"{rel}: absent under {snap}")
            continue
        d = sha256_of(p)
        if ent.get("sha256") and d != ent["sha256"]:
            bad.append(f"{rel}: sha256 {d[:12]} != pinned {ent['sha256'][:12]}")
    return bad


def disable() -> dict:
    """Withdraw the kit from the process: restore the three wrapper constructors (or withdraw the hook waiting for their import), disarm an
    autoload finder that has not fired, remove the reporter hooks from every engaged model, forget the activation. A model the levers were
    applied to keeps them — un-patching a live model is not offered; its outputs are the stock's either way. Prints REMOVED."""
    n_arm = 0
    for p in _STATE["patches"]:
        try:
            if p.state == "installed":
                p.restore()
                n_arm += 1
            elif p.disarm():
                n_arm += 1
        except Exception:  # noqa: BLE001
            pass
    auto = sys.modules.get("gpnstar_opt._autoload")
    finder = getattr(auto, "FINDER", None) if auto is not None else None
    if finder is not None and getattr(finder, "armed", False):
        finder.armed = False
        try:
            finder.remove()
        except Exception:  # noqa: BLE001
            pass
        n_arm += 1
    n_hooks = 0
    for h in _STATE["handles"]:
        try:
            h.remove()
            n_hooks += 1
        except Exception:  # noqa: BLE001
            pass
    n_models = len(_STATE["engaged"])
    rec = {"active": False, "arm": n_arm, "hooks": n_hooks, "models": n_models, "levers": "kept"}
    _STATE.update(_fresh_state(_STATE["exit_registered"]))
    report.emit(report.removed_line(n_arm, n_hooks, n_models))
    return rec


# --------------------------------------------------------------------------------------------------------------------- arm
WRAPPERS = ("MLMforVEPModel", "MLMforLogitsModel", "ModelCenterEmbedding")     # gpn.star.inference: the modules the trainer drives
TRIGGER = "gpn.star.inference"


def _arm() -> None:
    """Wrap the three wrapper constructors (now when the module is imported, else at its import) so every instance engages once built."""
    if _STATE["armed"]:
        return
    from opt_core.autoload import patch_attr_at_import
    for cls in WRAPPERS:
        p = patch_attr_at_import(TRIGGER, f"{cls}.__init__", _make_init_wrapper, tag=TAG, name=f"arm:{cls}", exit_not_active=EXIT_NOT_ACTIVE)
        p.install()                                       # a site restored by disable() is patched again
        _STATE["patches"].append(p)
    _STATE["armed"] = True


DEDUP_MIN_TOKENS = 512          # B·L below this runs the unified-K/V route by rule (reason=small_batch on the KV line): the de-dup bookkeeping costs more than its GEMM saves at such sizes


def _make_init_wrapper(original):
    """The upstream wrapper's constructor, followed by the engage: the inner model is placed on the GPU here (the tool places the wrapper
    there before its first forward in any case) and the levers apply NOW — before anything the tool does next (a dtype cast, a
    torch.compile of the wrapper) wraps it. Without a GPU at construction the engage waits for the first forward (pre-hook)."""
    def __init__(self, *a, **kw):
        original(self, *a, **kw)
        if not _STATE["armed"]:                            # disable() ran after this class was wrapped and before the restore reached it: stock
            return
        inner = getattr(self, "model", None)
        engaged = False
        try:
            import torch
            if inner is not None and torch.cuda.is_available():
                try:
                    dev = next(inner.parameters()).device
                except StopIteration:
                    dev = None
                if dev is not None and dev.type != "cuda":
                    inner.to(torch.device("cuda", torch.cuda.current_device()))
                try:
                    apply(inner, rep=_STATE["report"])
                except ActivationError as e:
                    _refuse_in_process(str(e))
                engaged = True
        except ImportError:
            engaged = False
        if not engaged:
            handle = self.register_forward_pre_hook(_engage_hook, with_kwargs=True)
            self._gpnstar_opt_engage = handle
    __init__._opt_core_wrapped = True
    return __init__


def _refuse_in_process(reason: str):
    report.emit(report.not_active_line(reason, f"mode={MODE}"))
    sys.stderr.flush()
    sys.stdout.flush()
    sys.exit(EXIT_NOT_ACTIVE)


def _engage_hook(wrapper, args, kwargs):
    h = getattr(wrapper, "_gpnstar_opt_engage", None)
    if h is not None:
        h.remove()
        wrapper._gpnstar_opt_engage = None
    ts = kwargs.get("target_species")
    if ts is not None:
        try:
            if ts.dim() != 2 or ts.shape[1] != 1 or bool((ts != 0).any().item()):
                vals = sorted({int(v) for v in ts.flatten().tolist()})[:8] if ts.dim() >= 1 else [int(ts)]
                _refuse_in_process(f"target-row contract: the levers serve one target row of species 0 per window; this batch carries "
                                   f"target_species of shape {tuple(ts.shape)} with values {vals}")
        except AttributeError:
            pass
    inner = getattr(wrapper, "model", None)
    if inner is None:
        _refuse_in_process(f"{type(wrapper).__name__} carries no .model to apply the levers to (upstream changed shape)")
    try:
        apply(inner, rep=_STATE["report"])
    except ActivationError as e:
        _refuse_in_process(str(e))
    return None


# ------------------------------------------------------------------------------------------------------------------ engage
def _model_identity(model) -> dict:
    cfg = getattr(model, "config", None)
    name = getattr(cfg, "_name_or_path", None) or getattr(cfg, "name_or_path", None)
    rev = getattr(cfg, "_commit_hash", None)
    key = registry.resolve(name) if name else None
    snap = None
    if name and os.path.isdir(str(name)):
        snap = str(name)
    elif key is not None:
        cand = registry.snapshot_dir(key)
        snap = cand if os.path.isdir(cand) else None
    return {"name": name, "rev": rev, "key": key, "snapshot": snap}


def apply(model, rep: Optional[dict] = None) -> dict:
    """Apply the levers to `model` (a GPNStarForMaskedLM / GPNStarModel ALREADY on the GPU) in place; print ACTIVE + LEVER; install the KV
    reporter. Returns the engaged record. ActivationError names a refusal (CPU-resident model, weights off digest)."""
    import torch
    rep = rep if rep is not None else (_STATE["report"] or {})
    try:
        dev = next(model.parameters()).device
    except StopIteration:
        raise ActivationError("the model has no parameters")
    if dev.type != "cuda":
        raise ActivationError(f"the model is on {dev} when the levers engage: move it to the GPU first (the levers build device tensors at "
                              f"apply time; applying on the CPU and moving afterwards is not supported)")
    ident = _model_identity(model)
    words = list(rep.get("words") or [])
    if ident["key"] is None:
        words.append(core.word("weights", "unpinned", registry.label(None, ident["name"], ident["rev"])))
    else:
        m = registry.models()[ident["key"]]
        if ident["rev"] and ident["rev"] != m["revision"]:
            words.append(core.word("weights", "revision", f"{ident['rev'][:8]}!={m['revision'][:8]}"))
        ck = registry.checkpoint(ident["key"])
        bad = weights_mismatch(ck, ident["snapshot"]) if ident["snapshot"] else None
        if bad:
            raise ActivationError("weights off their digests (stock/PINS.json): " + "; ".join(bad))
        if bad is None:
            words.append(core.word("weights", "unchecked", "no_digests" if ident["snapshot"] else "no_snapshot_dir"))
    from .accel import api as accel_api
    accel_api.make_exact(model, dedup=True)
    tf32 = tf32_state()
    rec = {"model_key": ident["key"], "model_label": registry.label(ident["key"]) if ident["key"] else registry.label(None, ident["name"], ident["rev"]),
           "device": str(dev), "levers": list(LEVERS), "tf32": tf32, "model": model}
    rep = dict(rep, **{"model_label": rec["model_label"], "model_key": rec["model_key"], "words": words, "active": True, "levers": list(LEVERS),
                      "word": MODE, "mode": MODE, "tf32": tf32})
    if rep.get("gpu") is None:
        rep["gpu"] = probe_gpu()
    if rep.get("card") is None and rep.get("gpu"):
        rep["card"] = card_gate(rep["gpu"]).get("class") or "untested"
    rep.setdefault("mib", (rep.get("gpu") or {}).get("mib"))
    if rep.get("torch") is None:
        rep["torch"] = torch.__version__
    if rep.get("transformers") is None:
        rep["transformers"] = dist_version("transformers")
    if rep.get("gpn_commit8") is None:
        rep["gpn_commit8"] = (gpn_install().get("commit") or registry.gpn_commit())[:8]
    _STATE["report"] = rep
    report.emit(report.active_line(rep))
    for ln in report.lever_lines(LEVERS):
        report.emit(ln)
    _STATE["engaged"].append(rec)
    _STATE["handles"].append(model.register_forward_pre_hook(_no_dynamo(_pre_hook), with_kwargs=True))
    _STATE["handles"].append(model.register_forward_hook(_no_dynamo(_kv_hook), with_kwargs=True))
    _memory_guard(model)
    _compile_guard(model)
    if not _STATE["exit_registered"]:
        atexit.register(_kv_exit_line)
        _STATE["exit_registered"] = True
    return {k: v for k, v in rec.items() if k != "model"}


def kv_state(model) -> dict:
    """{route, pairs, rejects} from the levers' own route report (accel.patches.kv_mode_report): route words dedup | unifiedkv | stock."""
    from .accel.patches import kv_mode_report
    r = kv_mode_report(model)
    used = r.get("last_mode_used")
    route = {"dedup": "dedup", "p3b": "unifiedkv", "sdedup": "dedup"}.get(used)
    overrides = r.get("override_per_stock_rows") or {}
    if route is None or used in (None, "None"):
        route = "stock" if any(str(v) == "stock" for v in overrides.values()) else {"dedup": "dedup", "p3b": "unifiedkv"}.get(r.get("configured"), "none")
    rejects = sum(int(v) for v in (r.get("reject_counts") or {}).values())
    return {"route": route, "pairs": len(r.get("validation_log") or []), "rejects": rejects, "raw": r}


def _no_dynamo(fn):
    """The hook as a function torch.compile never traces (torch.compiler.disable): the levers' bookkeeping stays eager."""
    try:
        import torch
        return torch.compiler.disable(fn)
    except (ImportError, AttributeError):
        return fn


def _compile_guard(model) -> None:
    """Keep the patched model out of any dynamo-compiled region: a caller compiled with torch.compile (the tool's --torch-compile wraps the
    upstream wrapper) reaches this forward through a graph break and runs it eager, levers included — no tracing of the patched modules,
    no recompiles per batch content. The rest of the caller compiles as it would."""
    try:
        import torch
        disable = torch.compiler.disable
    except (ImportError, AttributeError):
        return
    fwd = model.forward
    if getattr(fwd, "_gpnstar_opt_guarded", False):
        return
    guarded = disable(fwd)
    try:
        guarded._gpnstar_opt_guarded = True
    except AttributeError:
        pass
    model.forward = guarded


def _memory_guard(model) -> None:
    """Never fail where stock runs: if a forward runs out of device memory while the reduced K/V path holds its extra buffers, the
    levers release them, the batch shape is moved to the stock projections by rule (KV line: route=stock reason=memory) and the SAME
    forward runs once more — its output is the stock computation, exact. A second out-of-memory is the stock model's own and propagates."""
    import torch
    inner = model.forward
    if getattr(inner, "_gpnstar_opt_memory_guard", False):
        return

    def forward(*args, **kwargs):
        try:
            return inner(*args, **kwargs)
        except torch.cuda.OutOfMemoryError:
            from .accel.patches import core_model
            stt = getattr(core_model(model), "_exact_state", None)
            B, L, shape = _batch_shape(args, kwargs)
            if stt is None or B is None or not stt.clade_species:
                raise
            m_stock = B * L * len(stt.clade_species)
            if (stt.memory_fallbacks or {}).get(m_stock, 0) >= 2:
                raise                                                     # already on the stock projections for this shape: this is stock's own limit
        # here the failed attempt's frames (and the tensors they held) are released: move the shape to the stock projections and run again
        from .accel.patches import _memory_fallback
        _memory_fallback(stt, m_stock, where="forward")
        _STATE["kv_reason"][shape] = "memory"
        try:
            return inner(*args, **kwargs)
        except torch.cuda.OutOfMemoryError:
            report.emit(report.error_line(f"out of device memory at batch shape {shape} on the STOCK projections, after the levers released their buffers and "
                                          "stepped aside: this batch does not fit this card for the stock computation either; a smaller batch, or "
                                          "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True set before the process starts, is the remedy"))
            raise

    forward._gpnstar_opt_memory_guard = True
    model.forward = forward


def _entered_through_dynamo(max_depth: int = 120) -> bool:
    """Whether the current call arrived through a torch.compile (dynamo) wrapper: an OptimizedModule / dynamo-context frame is on the
    Python stack above us. (The levers' own torch.compiler.disable wrapper lives in the same file but under DisableContext: not counted.)"""
    import sys as _sys
    f = _sys._getframe(1)
    n = 0
    while f is not None and n < max_depth:
        co = f.f_code
        qual = getattr(co, "co_qualname", co.co_name)
        if co.co_filename.replace("\\", "/").endswith("_dynamo/eval_frame.py") and ("TorchDynamoContext" in qual or "OptimizedModule" in qual or "OptimizeContext" in qual):
            return True
        f = f.f_back
        n += 1
    return False


def _compile_notice() -> None:
    """Print the COMPILE line once, the first time a forward is entered from inside a torch.compile wrapper (never otherwise)."""
    if _STATE.get("compile_line_done"):
        return
    if not _entered_through_dynamo():
        return
    _STATE["compile_line_done"] = True
    report.emit(report.compile_line("compiled" if _compiled_caller() else "eager"))


def _compiled_caller() -> bool:
    """Whether a dynamo-compiled frame has run in this process (the tool's --torch-compile)."""
    try:
        from torch._dynamo.utils import counters
        return int((counters.get("stats") or {}).get("calls_captured", 0)) > 0 or int(sum((counters.get("frames") or {}).values())) > 0
    except Exception:  # noqa: BLE001
        return False


def _batch_shape(args, kwargs):
    ids = kwargs.get("input_ids") if isinstance(kwargs, dict) else None
    if ids is None:
        for a in list(args) + (list(kwargs.values()) if isinstance(kwargs, dict) else []):
            if hasattr(a, "shape") and getattr(a, "dim", lambda: 0)() >= 2:
                ids = a
                break
    if ids is None:
        return None, None, "unknown"
    B, L = int(ids.shape[0]), int(ids.shape[1])
    return B, L, f"{B}x{L}"


def _pre_hook(model, args, kwargs):
    """First forward of a batch shape: check the target-row contract, route a small batch (B·L < DEDUP_MIN_TOKENS) to unified K/V by
    rule through the levers' own per-shape route table, and start the first-forward clock (KV first_ms)."""
    try:
        _compile_notice()
        B, L, shape = _batch_shape(args, kwargs)
        if shape in _STATE["kv_started"]:
            return None
        import time as _time
        ts = kwargs.get("target_species") if isinstance(kwargs, dict) else None
        if ts is not None and hasattr(ts, "dim"):
            if ts.dim() != 2 or ts.shape[1] != 1 or bool((ts != 0).any().item()):
                vals = sorted({int(v) for v in ts.flatten().tolist()})[:8]
                _refuse_in_process(f"target-row contract: the levers serve one target row of species 0 per window; this batch carries "
                                   f"target_species of shape {tuple(ts.shape)} with values {vals}")
        if B is not None and B * L < DEDUP_MIN_TOKENS:
            try:
                from .accel.patches import core_model
                stt = getattr(core_model(model), "_exact_state", None)
            except Exception:  # noqa: BLE001
                stt = None
            if stt is not None and stt.dedup is True and stt.clade_species:
                m_stock = B * L * len(stt.clade_species)
                if stt.mode_override is None:
                    stt.mode_override = {}
                if stt.mode_override.get(m_stock) is None:
                    stt.mode_override[m_stock] = "p3b"
                    _STATE["kv_reason"][shape] = "small_batch"
        try:
            import torch
            torch.cuda.synchronize()
        except Exception:  # noqa: BLE001
            pass
        _STATE["kv_started"][shape] = _time.perf_counter()
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 -- a reporter never breaks the forward
        pass
    return None


def _kv_hook(model, args, kwargs, out):
    """After the first forward of each batch shape: one KV line (route, self-check pairs/rejects, first-forward ms, reason, compile)."""
    try:
        B, L, shape = _batch_shape(args, kwargs)
        if shape in _STATE["seen_shapes"]:
            return None
        import time as _time
        try:
            import torch
            torch.cuda.synchronize()
        except Exception:  # noqa: BLE001
            pass
        t0 = _STATE["kv_started"].get(shape)
        first_ms = None if t0 is None else (_time.perf_counter() - t0) * 1000.0
        ks = kv_state(model)
        _STATE["seen_shapes"][shape] = ks["route"]
        if ks["route"] in _STATE["kv"]:
            _STATE["kv"][ks["route"]] += 1
        _STATE["kvcheck"] = {"pairs": ks["pairs"], "rejects": ks["rejects"]}
        reason = _STATE["kv_reason"].get(shape) or ("kvcheck" if ks["route"] in ("dedup", "unifiedkv", "stock") else "none")
        try:  # a shape the levers moved to the stock projections because the reduced path did not fit in memory says so
            from .accel.patches import core_model
            stt = getattr(core_model(model), "_exact_state", None)
            mf = getattr(stt, "memory_fallbacks", None) or {}
            if B is not None and stt is not None and stt.clade_species and (B * L * len(stt.clade_species)) in mf:
                reason = "memory"
        except Exception:  # noqa: BLE001
            pass
        _STATE["kv_last"] = ks
        _STATE["kv_first_ms"][shape] = first_ms
        report.emit(report.kv_line(ks["route"], ks["pairs"], ks["rejects"], shape, first_ms=first_ms, reason=reason, compile=_compiled_caller()))
    except Exception:  # noqa: BLE001 -- a reporter never breaks the forward
        pass
    return None


def _kv_exit_line():  # noqa: D401
    """At interpreter exit: the closing KV line over every shape seen (shape=all)."""
    try:
        ks = _STATE.get("kv_last") or {"route": None, "pairs": 0, "rejects": 0}
        routes = set(_STATE["seen_shapes"].values())
        route = ks["route"] if len(routes) <= 1 else ("dedup" if "dedup" in routes else ks["route"])
        report.emit(report.kv_line(route, ks["pairs"], ks["rejects"], "all", first_ms=None, reason="none", compile=_compiled_caller()))
    except Exception:  # noqa: BLE001
        pass


def _reset_for_tests() -> None:
    _STATE.update(_fresh_state(_STATE["exit_registered"]))
