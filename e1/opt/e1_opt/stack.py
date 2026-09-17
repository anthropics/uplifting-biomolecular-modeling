"""Activation: the kit tree's paths, the gates, the GPU card class, the stack key, and the one resolver every route uses.

`activate(mode, variant)` resolves the mode by NAME in the mode table (modes.py), gates it, and reports — ONE line (report.py). The
gates REFUSE only what makes the run impossible or not the thing asked for: the kit tree absent or its constants disagreeing with the
mode table, the installed E1 not the pinned stock (`pins.assert_stock_commit()`: a different commit or version is a different "stock"),
the checkpoint file absent from the HF cache (`pins.assert_weights(size)`; its digest is worded against the pin on the weights line —
another digest runs, named), no GPU visible, `det=1` without the recipe's environment. Everything
else about the host is NAMED, never refused: a dependency off its pin (`pins.stack_report()`), an accelerator module the model imports
that does not import here (`stack_imports`: flash_attn, kernels), the hub kernel snapshot off its pinned revision, a GPU of no
tested class or a memory reading off its class's nominal (`card_gate`), a stock install without a commit record — all carried on the
ACTIVE line's trailing `notes="…"` field. The levers engage wherever their mechanism applies. A mode is ALL of its levers on the
card's class (the kit's per-class lever table; a GPU of no tested class takes the lever set of the class the config names): a lever of
that set that cannot run on the host (a compile / launch failure, an unsupported shape or dtype, an accelerator the levers run on absent
or fallen back) makes the MODE refuse by name — `NOT ACTIVE: …`, exit 3 — never a subset of the mode under the mode's name; under mode
off an accelerator upstream itself falls back from keeps its word on the KERNELS line and the stock runs. Nothing is applied here:
`activate(..., arm=True)` (the env / API route, and the command line's child through the E1_OPT autoload) wraps `E1Predictor.__init__`
(and `E1Scorer.__init__` when that module is already imported) so that the kit's `apply(model, size=...)` runs on the model at its first
construction; an apply that cannot run there is the NOT ACTIVE line and exit 3. A refusal raises ActivationError under `strict=True`
after printing `NOT ACTIVE: <reason>`.

The card CLASS (h100 | h200 | a100) keys the kit's per-class constants and lever table: a tested card by name, or the same silicon
under another marketing name by capability (compute capability + memory.total: every A100 — SXM4 or PCIe, 80 GB or 40 GB — is class
a100), read from `nvidia-smi` — torch-free — or from torch when it is already loaded. A card of no tested class runs as
`card=untested`, named.
"""
from __future__ import annotations

import gc
import importlib
import importlib.metadata as _md
import os
import platform
import shutil
import subprocess
import sys
import time
from typing import List, Optional

from . import __version__ as PACKAGE_VERSION
from . import ActivationError, modes, registry, report

from ._names import (ENV, ENV_VARIANT, ENV_DET, ENV_HOME, ENV_TARGET_GPU, MUST_BE_ABSENT_PREFIXES,   # noqa: F401 — the one names module
                     KIT_MODULE_PREFIXES, KIT_DIR_MARKERS, DET_VALUES)
KIT_RELDIR = os.path.join("engines", "e1", "kits", "eager")
KIT_MODULE = "engines.e1.kits.eager"                     # the kit's entry module (its __init__.py holds KIT / MODE, read by AST in modes.py)
KERNEL_CACHE_ENVS = ("KERNELS_CACHE", "HF_KERNELS_CACHE")   # the `kernels` package's cache root (its utils.py: KERNELS_CACHE, the older alias)

# the supported cards: name (the kit's own pin table's spelling; the kit's tests hold it to pins.TRITON_AUTOTUNE_PIN) -> (class, nominal nvidia-smi MiB)
CARDS = {"NVIDIA H100 80GB HBM3": ("h100", 81559), "NVIDIA H200": ("h200", 143771), "NVIDIA A100-SXM4-80GB": ("a100", 81920)}   # H100, H200, A100 80GB (the SXM4 part's name — the pin table's spelling); no other card is a class of its own (CARD_CAPABILITY maps the same silicon under other names)
CARD_CAPABILITY = {("9.0", 81559): "h100", ("9.0", 143771): "h200", ("8.0", 81920): "a100", ("8.0", 40960): "a100"}   # (compute capability, nvidia-smi memory.total MiB) -> class: the gate is by CAPABILITY — a card of a supported class under another marketing name (an 'NVIDIA A100 80GB PCIe': cc 8.0, 81920 MiB) is that class; the A100 40 GB parts ('NVIDIA A100-SXM4-40GB' / 'NVIDIA A100-PCIE-40GB': cc 8.0, 40960 MiB) are class a100 on the same silicon with half the memory (the ACTIVE line's mib= says which; every memory-dependent choice of the kit reads the device, not the class); the names above stay the pin table's spellings
CARD_MIB_TOLERANCE = 0.02                                # the card gate: the name AND the memory within 2 % of the card's nominal nvidia-smi memory.total — the reading varies below nominal by driver (an older driver's ECC carve-out: 81251 for 81920, 40536 for 40960) and by source (torch's total_memory reads under nvidia-smi's figure: about 1.3 % under 40960 on an A100-SXM4-40GB); no other card of a supported compute capability sits within 2 % of a supported size

_REPORT: Optional[dict] = None
_ARMED = {"done": False, "applied": None, "orig": {}}


# ----------------------------------------------------------------------------------------------------------------------- paths
def opt_home() -> str:
    """The e1/opt directory (this package's parent)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tree_home() -> str:
    """The e1/ directory: MODEL_OPT when set, else beside the package."""
    return os.path.abspath(os.environ.get(ENV_HOME) or os.path.dirname(opt_home()))


def forward_root() -> str:
    """The kit tree root (`opt/forward`): the directory that goes on sys.path for `import engines.e1.kits...`."""
    return os.path.join(tree_home(), "opt", "forward")


def kit_dir() -> str:
    return os.path.join(forward_root(), KIT_RELDIR)


def kit_entry() -> str:
    """The kit's entry file (kits/eager/__init__.py): its presence is the kit-tree gate."""
    return os.path.join(kit_dir(), "__init__.py")


def pins_path() -> str:
    return os.path.join(forward_root(), registry.PINS_RELPATH)


def pins_json_path() -> str:
    return os.path.join(tree_home(), "stock", "PINS.json")


def stock_source_path() -> str:
    """The tree's copy of the upstream CLI file (stock/src/E1/tools/score.py)."""
    return os.path.join(tree_home(), "stock", "src", "E1", "tools", "score.py")


def paths() -> dict:
    return {"tree": tree_home(), "opt": opt_home(), "forward_root": forward_root(), "kit_dir": kit_dir(), "kit_entry": kit_entry(),
            "pins": pins_path(), "pins_json": pins_json_path()}


def load_pins():
    return registry.load_pins(forward_root())


def kit_on_sys_path() -> str:
    """Put the kit tree root on sys.path (what the kit's own entry does) so `engines.e1.kits.eager` is importable; returns the root."""
    root = forward_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


# ----------------------------------------------------------------------------------------------------------------------- gates
def kernel_cache_roots() -> List[str]:
    """Where the `kernels` package keeps hub snapshots: its own cache variable(s) first, then the HF hub cache."""
    roots = []
    for k in KERNEL_CACHE_ENVS:
        v = os.environ.get(k)
        if v:
            roots.append(v)
    hf_home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    roots.append(os.path.join(hf_home, "hub"))
    return roots


def kernel_snapshot(pins) -> dict:
    """The pinned hub kernel snapshot on disk: {present, dir, root, layer_norm_py_sha256_ok, searched}; torch-free (a file check)."""
    k = pins.KERNEL
    rel = os.path.join("models--" + k["repo"].replace("/", "--"), "snapshots", k["rev"])
    searched = []
    for root in kernel_cache_roots():
        d = os.path.join(root, rel)
        searched.append(d)
        if os.path.isdir(d):
            ln = os.path.join(d, "build", "torch-universal", "triton_layer_norm", "layer_norm.py")
            ok = None
            if os.path.isfile(ln):
                ok = pins.sha256_file(ln) == k["layer_norm_py_sha256"]
            return {"present": True, "dir": d, "root": root, "layer_norm_py_sha256_ok": ok, "searched": searched, "rev": k["rev"]}
    return {"present": False, "dir": None, "root": None, "layer_norm_py_sha256_ok": None, "searched": searched, "rev": k["rev"]}


def probe_gpu_nvidia_smi() -> Optional[dict]:
    """The first GPU as nvidia-smi reports it: {name, mib, cc, sm, source}; None without nvidia-smi or a GPU."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
    if len(parts) < 3:
        return None
    try:
        mib = int(float(parts[1]))
    except ValueError:
        return None
    cc = parts[2]
    return {"name": parts[0], "mib": mib, "cc": cc, "sm": "sm" + cc.replace(".", ""), "source": "nvidia-smi"}


def probe_gpu_torch(force: bool = False) -> Optional[dict]:
    """The first GPU as torch reports it — only when torch is already imported AND its CUDA context already exists (a device query
    would otherwise create one in this process; the wrapper command's parent never holds a context while its child runs); `force`
    queries regardless (the env route: the process is the one that scores)."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not force and not torch.cuda.is_initialized():
            return None
        if not torch.cuda.is_available():
            return None
        p = torch.cuda.get_device_properties(0)
        cc = f"{p.major}.{p.minor}"
        return {"name": torch.cuda.get_device_name(0), "mib": int(p.total_memory // (1 << 20)), "cc": cc, "sm": "sm" + cc.replace(".", ""),
                "source": "torch"}
    except Exception:  # noqa: BLE001
        return None


def probe_gpu() -> Optional[dict]:
    """nvidia-smi first (torch-free, no CUDA context in this process); torch's view when its context already exists, or as the fallback."""
    return probe_gpu_torch() or probe_gpu_nvidia_smi() or probe_gpu_torch(force=True)


def card_gate(gpu: Optional[dict]) -> dict:
    """{ok, class, notes, reason}: the probed card's CLASS — a tested card by name, or the same silicon under another name by (compute
    capability, memory.total) — with everything else about the card NAMED in `notes` (a card of no tested class, a memory reading off
    the class's nominal, a `--config` whose target class is not this card's): the kit runs there, its levers engaging wherever their mechanism
    applies, the notes worded on the ACTIVE line's `notes=`. The one refusal (`ok` False) is no GPU at all: nothing can run."""
    if not gpu:
        return {"ok": False, "class": None, "notes": [], "reason": "no GPU visible (nvidia-smi absent or no device)"}
    name, mib, cc = gpu.get("name"), gpu.get("mib"), str(gpu.get("cc") or "")
    notes = []
    if name in CARDS:
        cls, want = CARDS[name]
    else:   # by capability: the class whose (compute capability, memory.total) this card has — the same silicon under another marketing name
        hit = next(((c, m) for (c_cc, m), c in CARD_CAPABILITY.items() if c_cc == cc and mib is not None and abs(int(mib) - m) <= CARD_MIB_TOLERANCE * m), None)
        cls, want = hit if hit is not None else (None, None)
        if hit is None:
            notes.append(f"card {name!r} (cc {cc or '?'}, {mib} MiB) is outside the tested classes ({'|'.join(sorted(set(CARD_CAPABILITY.values())))}): the levers engage where their mechanism applies, unmeasured here")
    if want is not None and (mib is None or abs(int(mib) - want) > CARD_MIB_TOLERANCE * want):   # within 2 % of the card's nvidia-smi memory.total (torch's total_memory reads a few hundred MiB lower)
        notes.append(f"card {name!r} reports {mib} MiB, its class {cls} has {want} MiB (±{CARD_MIB_TOLERANCE:.0%})")
    target = (os.environ.get(ENV_TARGET_GPU) or "").strip().lower()
    if target and target != (cls or ""):
        notes.append(f"the config targets {target} ({ENV_TARGET_GPU}), this card is {('class ' + cls) if cls else 'of no tested class'}: the config's constants are used as given")
    return {"ok": True, "class": cls, "notes": notes, "reason": None}


def stack_versions() -> dict:
    """Installed versions from package metadata (no import): the kit's own stack table's packages plus the stock package."""
    out = {}
    for dist, key in (("torch", "torch"), ("triton", "triton"), ("transformers", "transformers"), ("tokenizers", "tokenizers"),
                      ("kernels", "kernels"), ("flash-attn", "flash_attn"), ("E1", "E1")):
        try:
            out[key] = _md.version(dist)
        except _md.PackageNotFoundError:
            out[key] = None
    out["python"] = platform.python_version()
    return out


def stack_imports(pins) -> dict:
    """The stack's accelerator packages judged by IMPORTING the modules the model imports — `flash_attn` (`E1/model/flash_attention.py` L6:
    its `flash_attn_varlen_func` / `flash_attn_func`; the import loads the compiled extension `flash_attn_2_cuda`) and `kernels`
    (`E1/modeling.py` L9, L23: `get_kernel`) — not by their dist-info alone: a distribution at the pinned version whose module does not
    import (an ABI mismatch, a missing extension) is upstream's SILENT fallback route (`E1/model/attention.py` L299-311). NAMED, never
    refused: `notes` lists what does not import or is off its pin; the levers that need the module then cannot run and say so, and what each
    model process actually BOUND is worded at run time by the KERNELS proof (accel.py). Returns {<module facts>, notes: [...]}."""
    got, notes = {}, []
    try:
        import flash_attn                                                                       # noqa: F401 — the import IS the check
        from flash_attn import flash_attn_func, flash_attn_varlen_func                          # noqa: F401 — the two names E1/model/flash_attention.py L6 imports
        got["flash_attn_module"] = str(getattr(flash_attn, "__version__", None))
        got["flash_attn_2_cuda_loaded"] = "flash_attn_2_cuda" in sys.modules
        if got["flash_attn_module"] != pins.STACK["flash_attn"]:
            notes.append(f"flash_attn module {got['flash_attn_module']} (pin {pins.STACK['flash_attn']})")
    except Exception as e:                                                                      # noqa: BLE001
        got["flash_attn_module"] = None
        notes.append(f"flash_attn does not import ({type(e).__name__}: {str(e)[:120]}): upstream takes its varlen-flex fallback, the levers on the flash route cannot engage")
    try:
        import kernels
        getattr(kernels, "get_kernel")
        got["kernels_module"] = str(getattr(kernels, "__version__", None))
    except Exception as e:                                                                      # noqa: BLE001
        got["kernels_module"] = None
        notes.append(f"kernels does not import or has no get_kernel ({type(e).__name__}: {str(e)[:120]}): upstream falls back to torch rms_norm, the RMSNorm pin cannot engage")
    got["notes"] = notes
    return got


def stack_key(gpu: Optional[dict] = None, versions: Optional[dict] = None) -> str:
    """torch<version sans local tag>-cu<CUDA version sans dot>-sm<compute capability digits>, e.g. torch2.8.0-cu128-sm90 — the key a
    persistent JIT cache directory is kept under. The CUDA version comes from torch's local tag (`+cu128`) or, when torch
    is loaded, from torch.version.cuda; unknown parts read `NA`."""
    versions = versions or stack_versions()
    tv = versions.get("torch") or "NA"
    base, _, local = tv.partition("+")
    cu = "NA"
    if local.startswith("cu"):
        cu = local[2:]
    torch = sys.modules.get("torch")
    if torch is not None and getattr(getattr(torch, "version", None), "cuda", None):
        cu = str(torch.version.cuda).replace(".", "")
    sm = (gpu or {}).get("sm") or "smNA"
    return f"torch{base}-cu{cu}-{sm}"


def jit_cache_key() -> str:
    """`stack_key()` for the GPU this box shows (`python -c "from e1_opt.stack import jit_cache_key; print(jit_cache_key())"`)."""
    return stack_key(probe_gpu())


def _gate(name: str, fn, checks: dict, refusals: list, refuse_as: Optional[str] = None) -> None:
    """Run one gate; a PinDrift / OSError / ValueError becomes a named refusal (never a traceback)."""
    try:
        checks[name] = {"ok": True, "detail": fn()}
    except Exception as e:  # noqa: BLE001 — every gate failure is reported by name
        checks[name] = {"ok": False, "detail": f"{type(e).__name__}: {e}"}
        refusals.append(f"{refuse_as or name}: {e}")


def gates(variant: Optional[str], *, need_gpu: bool = True, hash_weights: bool = True) -> dict:
    """Every gate, evaluated (none short-circuits: the report names them all). Returns {checks, refusals, notes, gpu, card, pins, versions}.
    `refusals` are the few reasons nothing can run as asked — the kit's files missing or disagreeing with the mode table, the installed E1 not
    the pinned stock (commit / version: a different stock), the weights file absent, no GPU. `notes` NAME what of this host is off the
    kit's pins — a dependency off its pin, an accelerator module that does not import, the hub kernel snapshot off its pin, a card of
    no tested class, a stock install without a commit record — and never refuse: the ACTIVE line carries them (`notes=`), the
    levers engage wherever their mechanism applies, one that cannot run says so on its own LEVER line."""
    checks, refusals, notes = {}, [], []
    pins = None
    ks = kit_entry()
    if not os.path.isfile(ks):
        checks["kit_tree"] = {"ok": False, "detail": f"kit entry not found at {ks} (set {ENV_HOME} to the e1/ directory)"}
        refusals.append(f"kit missing: {ks}")
    else:
        checks["kit_tree"] = {"ok": True, "detail": ks}
    try:
        pins = load_pins()
        checks["pins_module"] = {"ok": True, "detail": pins_path()}
    except Exception as e:  # noqa: BLE001
        checks["pins_module"] = {"ok": False, "detail": f"{type(e).__name__}: {e}"}
        refusals.append(f"pins module: {e}")
    if os.path.isfile(ks):
        lock = modes.lock_check(kit_dir())
        checks["modes_lock"] = {"ok": not lock, "detail": lock}
        if lock:
            refusals.append("mode table lock: " + "; ".join(lock))
    if pins is not None:
        _gate("stock_commit", pins.assert_stock_commit, checks, refusals, "not the pinned stock")   # a different commit / version refuses (a different stock); no VCS record is named below
        sc = checks["stock_commit"].get("detail")
        if isinstance(sc, dict) and not sc.get("commit_recorded", True):
            notes.append(sc.get("why") or "stock commit not recorded by the installer")
        st = pins.stack_report()                                                                # the dependency stack beside its pins: drift named, never refused
        checks["stack"] = {"ok": not st["drift"], "detail": st}
        notes.extend(f"stack: {d}" for d in st["drift"])
        si = stack_imports(pins)                                                                # the accelerator modules import (not dist-info alone): named
        checks["stack_imports"] = {"ok": not si["notes"], "detail": si}
        notes.extend(si["notes"])
        pj = registry.cross_check(pins, pins_json_path())
        checks["pins_json"] = {"ok": not pj, "detail": pj or pins_json_path()}
        if pj:
            refusals.append("kit files: stock/PINS.json disagrees with the pins module: " + "; ".join(pj))
        kern = kernel_snapshot(pins)
        ok = kern["present"] and kern["layer_norm_py_sha256_ok"] is not False
        checks["kernel"] = {"ok": ok, "detail": kern}
        if not ok:
            notes.append("hub kernel snapshot " + (f"{kern['dir']}: layer_norm.py does not hash to the pin" if kern["present"]
                         else f"of revision {str(kern['rev'])[:12]} not found under {kern['searched']} (upstream fetches it, or falls back to torch rms_norm: the KERNELS line words which)"))
        if variant:
            if hash_weights:
                _gate("weights", lambda: pins.assert_weights(variant), checks, refusals, "weights")
            else:
                snap = registry.snapshot_dir(variant, pins)
                present = os.path.isfile(os.path.join(snap, "model.safetensors"))
                checks["weights"] = {"ok": present, "detail": {"snapshot": snap, "present": present, "sha256": "not hashed"}}
                if not present:
                    refusals.append(f"weights: {variant}: no model.safetensors at {snap}")
    gpu = probe_gpu() if need_gpu else None
    card = card_gate(gpu) if need_gpu else {"ok": True, "class": None, "notes": [], "reason": None}
    checks["card"] = {"ok": card["ok"] and not card["notes"], "detail": gpu if card["ok"] else card["reason"]}
    if need_gpu and not card["ok"]:
        refusals.append(card["reason"])
    notes.extend(card.get("notes") or [])
    return {"checks": checks, "refusals": refusals, "notes": notes, "gpu": gpu, "card": card, "pins": pins, "versions": stack_versions()}


# ------------------------------------------------------------------------------------------------------------------- resolution
def _effective_variant(variant: Optional[str], pins) -> Optional[str]:
    """The variant: the argument, else E1_VARIANT; a set variable that disagrees with the argument is refused (one variant per process)."""
    env_v = (os.environ.get(ENV_VARIANT) or "").strip().lower() or None
    v = registry.check_variant(variant, pins) if variant else None
    if v and env_v and env_v != v:
        raise ValueError(f"variant {v} disagrees with {ENV_VARIANT}={env_v}")
    return v or (registry.check_variant(env_v, pins) if env_v else None)



def _mode_vs_env(mode: str) -> Optional[str]:
    env_m = (os.environ.get(ENV) or "").strip().lower() or None
    if env_m and env_m != mode:
        return f"mode {mode} disagrees with {ENV}={env_m}"
    return None


def det_level(environ=None) -> int:
    """The det switch of the environment (E1_OPT_DET=0|1; the CLI's --det overrides by exporting it) — the ONE reader of E1_OPT_DET: absent or
    empty = 0; any value outside _names.DET_VALUES is refused by name (ActivationError: the NOT ACTIVE line, exit 3), never read as 0."""
    environ = os.environ if environ is None else environ
    v = (environ.get(ENV_DET) or "0").strip()
    if v not in DET_VALUES:
        raise ActivationError(f"{ENV_DET}={v!r}: 0 or 1 only")
    return int(v)


def status() -> dict:
    return dict(_REPORT) if _REPORT else {"active": False, "reason": "enable() has not run in this process"}


def _instances_exist() -> List[str]:
    """Names of the upstream classes that already have live instances (gc scan; only for modules already imported)."""
    found = []
    for modname, cls in (("E1.predictor", "E1Predictor"), ("E1.scorer", "E1Scorer")):
        m = sys.modules.get(modname)
        c = getattr(m, cls, None) if m else None
        if c is None:
            continue
        for obj in gc.get_objects():
            if isinstance(obj, c):
                found.append(cls)
                break
    return found


def activate(mode: Optional[str], variant: Optional[str] = None, *, strict: bool = False, trigger: Optional[str] = None,
             arm: bool = False, dry_run: bool = False, route: Optional[str] = None, hash_weights: bool = True,
             extra: Optional[dict] = None, quiet: bool = False) -> dict:
    """Resolve, gate and report; arm the constructor wrap on request (the env / API route). See the module docstring.
    Returns the activation report and prints its ONE line (unless quiet)."""
    global _REPORT
    if _REPORT is not None and not dry_run:
        same = _REPORT.get("mode") == modes.check_mode(mode) and (variant is None or _REPORT.get("variant") == variant)
        if same:
            return dict(_REPORT)
        rep = dict(_REPORT, active=False, reason=f"already activated in this process as mode={_REPORT.get('mode')} variant={_REPORT.get('variant')}; "
                   f"a different mode or variant is refused")
        if not quiet:
            report.emit(report.not_active_line(rep["reason"]))
        if strict:
            raise ActivationError(rep["reason"])
        return rep
    t0 = time.perf_counter()
    try:
        det_now, det_error = det_level(), None
    except ActivationError as e:                        # E1_OPT_DET outside 0|1: refused by name through the printed line (never silent)
        det_now, det_error = None, str(e)
    rep = {"active": False, "mode": None, "variant": variant, "kit": None, "kit_mode": None, "gpu": None, "stack_key": None,
           "package_version": PACKAGE_VERSION, "reason": None, "route": route or ("env" if trigger else "api"), "trigger": trigger,
           "det": det_now, "dry_run": bool(dry_run), "would_refuse": None, "pins": None, "versions": None, "paths": paths(),
           "e1_opt_env": os.environ.get(ENV), "e1_variant_env": os.environ.get(ENV_VARIANT), "python": platform.python_version()}
    if extra:
        rep.update(extra)

    def finish(reason: Optional[str], active: bool = False) -> dict:
        global _REPORT
        rep["active"] = active
        rep["reason"] = reason
        rep["t_s"] = round(time.perf_counter() - t0, 3)
        if dry_run:
            rep["would_refuse"] = reason if reason and reason != report.OFF_REASON else None
            rep["active"] = False
        else:
            _REPORT = dict(rep)
        if not quiet:
            report.emit(report.activation_line(rep))
        if strict and not dry_run and not active and reason != report.OFF_REASON:
            raise ActivationError(reason)
        return dict(rep)

    if det_error:
        return finish(det_error)
    try:
        m = modes.check_mode(mode)
    except ValueError as e:
        return finish(str(e))
    rep["mode"] = m
    entry = modes.MODE_TABLE[m]
    rep["kit_mode"] = entry.kit_mode
    try:
        rep["kit"] = modes.kit_version(kit_dir())
    except OSError:
        rep["kit"] = None
    dis = _mode_vs_env(m)
    if dis:
        return finish(dis)
    try:
        pins = load_pins()
    except Exception as e:  # noqa: BLE001
        return finish(f"kit missing: {e}")
    try:
        v = _effective_variant(variant, pins)
    except ValueError as e:
        return finish(str(e))
    if v is None:
        return finish(f"no variant: pass variant= ({'|'.join(registry.variants(pins))}) or set {ENV_VARIANT}")
    rep["variant"] = v
    rep["checkpoint"] = registry.checkpoint(v, pins)
    g = gates(v, need_gpu=True, hash_weights=hash_weights)
    rep["pins"] = {k: c for k, c in g["checks"].items()}
    rep["versions"] = g["versions"]
    rep["gpu"] = g["gpu"]
    rep["card"] = g["card"]["class"] or ("untested" if g["gpu"] else None)   # the tested class, or the word: the notes say which card
    rep["mib"] = (g["gpu"] or {}).get("mib")
    rep["sm"] = (g["gpu"] or {}).get("sm")
    rep["stack_key"] = stack_key(g["gpu"], g["versions"])
    rep["pins_ok"] = not g["refusals"] and not [n for n in g["notes"] if n not in (g["card"].get("notes") or [])]   # the pins word: the stock / stack / kernel notes; the card's notes ride notes= alone
    rep["pins_summary"] = "ok" if rep["pins_ok"] else ("drift" if not g["refusals"] else "refused")   # drift: something of this host is off its pin — named on the line (pins=drift, notes=), the run proceeds
    rep["notes"] = list(g["notes"])
    rep["pins"] = {"checks": g["checks"], "refusals": g["refusals"], "notes": g["notes"]}
    wrec = (g["checks"].get("weights") or {}).get("detail")
    if not quiet and isinstance(wrec, dict) and wrec.get("word"):
        report.emit(report.weights_line(wrec, pins), sys.stderr)   # the digest worded against the pin, on stderr: either word runs, the line says which
    if g["refusals"]:
        return finish("; ".join(g["refusals"]))                    # every failed gate by name, in gate order
    if m == "off":
        return finish(report.OFF_REASON, active=False)
    # exact: the kit applies at construction; the wrap is armed here (the command line's child arms it through the E1_OPT autoload)
    if arm and not dry_run:
        if _ARMED["done"]:
            pass
        else:
            try:
                kit_on_sys_path()
                kit = importlib.import_module(KIT_MODULE)
            except Exception as e:  # noqa: BLE001
                return finish(f"kit missing: cannot import the kit package from {forward_root()}: {type(e).__name__}: {e}")
            try:
                applied = bool(getattr(kit, "_state", {}).get("applied"))
            except Exception:  # noqa: BLE001
                applied = False
            if applied:
                return finish("late activation refused: the kit reports itself already applied in this process")
            live = _instances_exist()
            if live:
                return finish(f"late activation refused: {','.join(live)} instance(s) already exist in this process (a model served before activation)")
            try:
                _arm(rep, v)
            except Exception as e:  # noqa: BLE001
                return finish(f"cannot arm the constructor wrap: {type(e).__name__}: {e}")
    return finish(None, active=True)


# ------------------------------------------------------------------------------------------------- the wrap (env / API route)
def _refuse_det_env(missing: list) -> None:
    """E1_OPT_DET=1 with a recipe environment entry absent or at another value: refuse by name (the line, the report, ActivationError)."""
    reason = "det=1 but the recipe's environment is not in force: " + ", ".join(f"{k}={v!r} (want {w!r})" for k, v, w in missing) + \
             " — export them before the interpreter starts (the autoload does when E1_OPT_DET=1 is set at start), or run with E1_OPT_DET=0"
    report.emit(report.not_active_line(reason))
    if _REPORT is not None:
        _REPORT["active"] = False
        _REPORT["reason"] = reason
    raise ActivationError(reason)


def apply_kit_once(model, variant: str, rep: Optional[dict] = None, det: Optional[bool] = None) -> Optional[dict]:
    """The kit's `apply(model, size=variant)`; once per process (one model per process: the kit refuses a second apply itself).
    Returns the kit's record (None when already applied). The kit prints its own KIT / LEVER lines. `det`: None = the E1_OPT_DET switch,
    False = never (True applies the recipe's torch switches before the kit, det.py)."""
    if _ARMED["applied"] is not None:
        return None
    kit_on_sys_path()
    kit = importlib.import_module(KIT_MODULE)
    use_det = bool(det_level() if det is None else det)
    if use_det:
        from . import det as _det
        _pins = load_pins()
        missing = _det.env_missing(_pins, 1)             # the recipe's environment entries (CUBLAS_WORKSPACE_CONFIG, the numpy / OpenBLAS entries) must already
        if missing:                                      # be in force: exported at interpreter start by the autoload (_autoload._export_recipe_env) or by the
            _refuse_det_env(missing)                     # caller — a det=1 run that is not the recipe does not run: NOT ACTIVE naming the variables, exit 3
        _det.apply_torch(_pins)                          # then the torch switches, before the pin / any forward
    rec = kit.apply(model, size=variant, det=use_det)
    _ARMED["applied"] = rec
    applied = {"kit": (rec or {}).get("kit"), "mode": (rec or {}).get("mode"), "size": (rec or {}).get("size"), "levers_on": (rec or {}).get("n_on"), "lines": (rec or {}).get("lines")}
    if rep is not None:
        rep["applied"] = applied
    if _REPORT is not None:
        _REPORT["applied"] = applied
    return rec


def _wrap_init(cls, variant: str, rep: dict, name: str) -> None:
    orig = cls.__init__
    if getattr(orig, "_e1_opt_wrapped", False):
        return

    def __init__(self, model, *args, **kwargs):
        try:
            apply_kit_once(model, variant, rep, det=rep.get("det") if rep.get("child") else None)
        except ActivationError:                          # refused by name (the NOT ACTIVE line is printed): the process stops here, exit 3, never stock
            sys.exit(3)
        except Exception as e:  # noqa: BLE001 — a lever of the mode's set cannot run here: the MODE refuses by name (a mode is all of its levers), never a subset under its name, never stock
            report.emit(report.not_active_line(f"the kit refused at {name} construction: {type(e).__name__}: {e}"))
            sys.exit(3)
        orig(self, model, *args, **kwargs)
    __init__._e1_opt_wrapped = True
    __init__.__wrapped__ = orig
    _ARMED["orig"][name] = (cls, orig)
    cls.__init__ = __init__


def arm(mode: str, variant: str, det: Optional[bool] = None) -> dict:
    """Arm the constructors WITHOUT the gates or the line — for the command line's child (kit_score.py), whose parent ran the gates and
    printed the ACTIVE line: the kit tree on sys.path, E1Predictor / E1Scorer wrapped so the kit applies once at the first construction
    (`det`: the recipe's torch switches before the kit; None = the E1_OPT_DET switch). Returns the child's report dict."""
    kit_on_sys_path()
    rep = {"mode": modes.check_mode(mode), "variant": variant, "kit_mode": modes.entry(mode).kit_mode, "det": det, "child": True}
    if not _ARMED["done"]:
        _arm(rep, variant)
    return rep


def _arm(rep: dict, variant: str) -> None:
    """Wrap E1Predictor.__init__ (the class the kit applies at; E1Scorer builds one) and E1Scorer.__init__ when its module is already
    imported; the first construction applies the kit once. The KERNELS proof (accel.py) is armed first, innermost: it reads what the
    process bound after the kit has applied and the constructor returned and prints `[e1-opt exact] KERNELS route=exact ...` — an
    accelerator that fell back named by its word and the mode's refusal by name (exit 3)."""
    P = importlib.import_module("E1.predictor")
    from . import accel
    accel.install(rep["mode"], rep["mode"], load_pins())   # innermost wrap: fires after the kit-applying wrap below has run
    _wrap_init(P.E1Predictor, variant, rep, "E1Predictor")
    S = sys.modules.get("E1.scorer")
    if S is not None and hasattr(S, "E1Scorer"):
        _wrap_init(S.E1Scorer, variant, rep, "E1Scorer")
    _ARMED["done"] = True
    rep["armed"] = sorted(_ARMED["orig"])


def _reset_for_tests() -> None:
    """Undo the process-global state (tests only): this module's wraps, then the KERNELS proof's (armed inside them)."""
    global _REPORT
    for name, (cls, orig) in list(_ARMED["orig"].items()):
        cls.__init__ = orig
    _ARMED.update({"done": False, "applied": None, "orig": {}})
    _REPORT = None
    from . import accel
    accel._reset_for_tests()
