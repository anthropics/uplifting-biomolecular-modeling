"""Where things are, what is pinned, what the box is, and the activation of a mode.

* paths — the tree (`progen2/`), the kit class dirs under `opt/`, the stock checkout (`stock/src/progen2`, or ``PROGEN2_STOCK_DIR``),
  the weights (``PROGEN2_WEIGHTS``: `<dir>/<upstream name>/{pytorch_model.bin, config.json}`), the stock RUN directory
  (``workdir``: the stock scripts want `./checkpoints/<name>` and `tokenizer.json` in the cwd — a directory of symlinks to the pinned
  checkout plus `checkpoints/<name>` -> the weights; ``PROGEN2_RUN_DIR`` names it, else a per-user temp dir);
* pins — `stock/PINS.json` (the pinned commit, the stock files and their sha256, the pinned stack, the weights' sha256 per size);
  ``pins_gate`` hashes the stock files at the stock dir against the pinned commit's digests (stock/check_pins.py's own check, loaded
  from the tree: THE refusal — a stock file missing or off its digest changes what stock means) and compares the interpreter and the
  three library versions with the pinned stack (a difference is a NOTE on the stack line, never a refusal: the levers were tested on
  the pinned stack, and engage wherever their mechanism applies);
* the box — GPU name, memory and compute capability (torch when loaded, else nvidia-smi; the memory is nvidia-smi's `memory.total`
  whenever nvidia-smi answers for the same card — the figure the class table is written in; torch's `total_memory`, smaller by the
  driver's reservation, only when it does not); ``card_class`` = the card's class by name AND memory within 1 % of one of the class's
  card memories (`h100` / `h200` / `b200` / `a100` — the A100 class is the 80 GB and the 40 GB card), ``wrong_card`` otherwise; ``stack_key``;
* activation — ``activate(mode, variant, route=, device=)``: resolve (modes.resolve), gate (the stock files' digests; the kit dirs
  present — an install error otherwise), note (stack / card / settings outside the tested defaults) and print the activation-time
  line (report.py: the ``stack`` line under exact — the ``ACTIVE`` line follows once the route's composition has its evidence, printed
  by the verb; ``NOT ACTIVE`` for off / a refusal by name — ``--device cpu`` or no CUDA device is one: the levers run on CUDA and a mode
  is all of its levers). Idempotent; a different mode or size in the same process is refused.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as _md
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from typing import Dict, Optional, Tuple

from . import modes, report

ENV_STOCK_DIR, ENV_WEIGHTS, ENV_RUN_DIR, ENV_PYTHON = "PROGEN2_STOCK_DIR", "PROGEN2_WEIGHTS", "PROGEN2_RUN_DIR", "PROGEN2_PYTHON"
DATA_ENV = (ENV_STOCK_DIR, ENV_WEIGHTS, ENV_RUN_DIR, ENV_PYTHON)             # the data-path names: allowed in the stock arm's environment (paths, not switches)
PROCESS_TIMEOUT_S = 6 * 60 * 60                                             # the one wall budget for a stock / kit process the package spawns (an item process, the pin check, a kit test): past it the process is killed and reported
STOCK_SUBDIR = os.path.join("stock", "src", "progen2")
PINS_RELPATH = os.path.join("stock", "PINS.json")
STOCK_FILES = ("sample.py", "likelihood.py", "tokenizer.json", os.path.join("models", "progen", "modeling_progen.py"),
               os.path.join("models", "progen", "configuration_progen.py"), "requirements.txt")
STOCK_LINKS = ("sample.py", "likelihood.py", "tokenizer.json", "models", "requirements.txt", "README.md")   # what the run dir links from the checkout
WEIGHT_FILES = ("pytorch_model.bin", "config.json")
STACK_LIBS = ("torch", "transformers", "tokenizers")

# the known GPU classes: name substring and the card memories of the class (MiB, `nvidia-smi --query-gpu=memory.total`), within 1 %.
# `a100` is ONE class for both A100 cards (80 GB: 81920 MiB; 40 GB: 40960 MiB — SXM4 or PCIe): the same code path; nothing on either route is
# sized by the class name — the memory-dependent choices (the generation kit's static K/V slots per batch size, allocated after a fit check against
# the free memory; the scoring kit's resident budget) read the device's own memory, so the 40 GB card takes fewer samples per call before an
# out-of-memory is named.
GPU_CLASSES: Dict[str, Tuple[str, Tuple[int, ...]]] = {"h100": ("H100", (81559,)), "h200": ("H200", (143771,)), "b200": ("B200", (183359,)), "a100": ("A100", (81920, 40960))}
STACK_ID = "pg2cu128"                                                       # the id of the pinned stack

_STATE: Dict[str, object] = {"report": None}


class ActivationError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------------------------------ paths
def package_home() -> str:
    """`progen2/opt/` — the directory holding the package and the kit class dirs (the package is installed editable from it)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tree_home() -> str:
    return os.path.dirname(package_home())


def python() -> str:
    """The interpreter for every stock / kit process the package spawns: ``$PROGEN2_PYTHON`` when set (STOCK.md 'Variables'), else this one."""
    return os.environ.get(ENV_PYTHON) or sys.executable


def kit_dir(kind: str) -> str:
    rel = {"serving": modes.SERVING_KIT_RELPATH, "forward": modes.FORWARD_ROOT_RELPATH, "scoring": modes.SCORING_KIT_RELPATH, "ew": modes.EW_KIT_RELPATH}[kind]
    return os.path.join(package_home(), rel)


def stock_dir() -> str:
    return os.path.abspath(os.environ.get(ENV_STOCK_DIR) or os.path.join(tree_home(), STOCK_SUBDIR))


def weights_dir() -> Optional[str]:
    w = os.environ.get(ENV_WEIGHTS)
    if w:
        return os.path.abspath(w)
    d = os.path.join(tree_home(), "weights")
    return d if os.path.isdir(d) else None


def pins_path() -> str:
    return os.path.join(tree_home(), PINS_RELPATH)


def pins() -> dict:
    with open(pins_path(), "r", encoding="utf-8") as fh:
        return json.load(fh)


def upstream_name(variant: str) -> str:
    return modes.UPSTREAM_NAME[modes.check_variant(variant)]


def variant_of_upstream(upstream: str) -> str:
    """The variant whose upstream checkpoint name is `upstream` (progen2-small -> small)."""
    for v, up in modes.UPSTREAM_NAME.items():
        if up == upstream:
            return v
    raise ValueError(f"no variant has the upstream name {upstream!r}")


def variant_weights(variant: str) -> Optional[str]:
    w = weights_dir()
    return os.path.join(w, upstream_name(variant)) if w else None


def check_pins_argv(variant: Optional[str] = None) -> list:
    """The args `cmd_check()` sends to `stock/check_pins.py`: always `--stock-dir`; `--weights-dir` + `--size` only when weights
    are staged. The ONE place this argv is built, so check_pins.py's own argparse and this list can never drift apart silently
    (a flag check_pins.py doesn't accept would otherwise make every `progen2-opt check` fail unconditionally)."""
    argv = ["--stock-dir", stock_dir()]
    wdir = weights_dir()
    if wdir and os.path.isdir(wdir):
        argv += ["--weights-dir", wdir]
        v = modes.check_variant(variant)
        if v:
            argv += ["--size", upstream_name(v)]
    return argv


_RUN_ROOTS: Dict[str, str] = {}


def _own_run_root(root: str) -> str:
    """The parent of the default run directories.  Its name is predictable and the temp dir is shared, and the run directory goes first on
    sys.path, so ``root`` serves only when it is this user's own: made here with mode 0700, or already a real directory (not a symbolic link)
    that this user owns and that group and others cannot write.  Anything else at that name is refused by name on stderr and left alone; a
    private directory (tempfile.mkdtemp, mode 0700) serves the process instead, exactly as if ``root`` had been absent.  Decided once per
    process."""
    import stat
    if root in _RUN_ROOTS:
        return _RUN_ROOTS[root]
    why = None
    try:
        os.mkdir(root, 0o700)
    except FileExistsError:
        pass
    except OSError as e:
        why = f"it cannot be made ({e.strerror})"
    if why is None:
        st = os.lstat(root)
        if stat.S_ISLNK(st.st_mode):
            why = "it is a symbolic link; to use it: remove the link"
        elif not stat.S_ISDIR(st.st_mode):
            why = "it is not a directory; to use it: remove it"
        elif hasattr(os, "getuid") and st.st_uid != os.getuid():
            why = f"it belongs to uid {st.st_uid}, not to this user (uid {os.getuid()}); to use it: have its owner remove it"
        elif st.st_mode & 0o022:
            why = f"group or others can write it (mode {st.st_mode & 0o7777:04o}); to use it: chmod go-w"
    use = root
    if why is not None:
        use = tempfile.mkdtemp(prefix=os.path.basename(root) + "-")
        sys.stderr.write(f"[progen2-opt] run directory {root} refused: {why}, or name a directory of your own with {ENV_RUN_DIR}; "
                         f"this process runs from the private directory {use}\n")
        sys.stderr.flush()
    _RUN_ROOTS[root] = use
    return use


def workdir(variant: Optional[str] = None, create: bool = True) -> str:
    """The stock run directory: symlinks to the pinned checkout's entries + `checkpoints/<name>` -> the weights for every size
    present under the weights dir (the stock scripts resolve `./checkpoints/<name>` and `tokenizer.json` against the cwd)."""
    sd = stock_dir()
    w = weights_dir()
    run = os.environ.get(ENV_RUN_DIR)
    if not run:
        tag = hashlib.sha256((sd + "|" + (w or "")).encode()).hexdigest()[:12]
        root = os.path.join(tempfile.gettempdir(), f"progen2-opt-{os.getuid() if hasattr(os, 'getuid') else 'u'}")
        run = os.path.join(_own_run_root(root) if create else root, f"run-{tag}")   # create=False names the directory only: nothing is read from it
    if not create:
        return run
    os.makedirs(os.path.join(run, "checkpoints"), exist_ok=True)
    for name in STOCK_LINKS:
        src, dst = os.path.join(sd, name), os.path.join(run, name)
        if not os.path.exists(src):
            if name in STOCK_FILES:
                raise ActivationError(f"stock checkout at {sd} has no {name}")
            continue
        if os.path.islink(dst) and os.readlink(dst) == src:
            continue
        if os.path.lexists(dst):
            os.unlink(dst)
        os.symlink(src, dst)
    if w:
        for v, up in modes.UPSTREAM_NAME.items():
            src, dst = os.path.join(w, up), os.path.join(run, "checkpoints", up)
            if not os.path.isdir(src):
                continue
            if os.path.islink(dst) and os.readlink(dst) == src:
                continue
            if os.path.lexists(dst):
                os.unlink(dst)
            os.symlink(src, dst)
    if variant:
        ck = os.path.join(run, "checkpoints", upstream_name(variant))
        if not all(os.path.isfile(os.path.join(ck, f)) for f in WEIGHT_FILES):
            if weights_dir() is None:
                raise ActivationError(f"{ENV_WEIGHTS} is not set — see README Variables (the weights for {variant}: ${ENV_WEIGHTS}/{upstream_name(variant)}/ holding {WEIGHT_FILES})")
            raise ActivationError(f"weights for {variant} not found: expected {WEIGHT_FILES} under {variant_weights(variant)} (set {ENV_WEIGHTS})")
    return run


# ------------------------------------------------------------------------------------------------------------------------ pins
def _dist_version(name: str) -> Optional[str]:
    try:
        return _md.version(name)
    except _md.PackageNotFoundError:
        return None


def stack_versions() -> dict:
    out = {"python": platform.python_version()}
    for lib in STACK_LIBS:
        out[lib] = _dist_version(lib)
    return out


def stack_pins(p: dict) -> dict:
    """The four pins the kits assert (stock/PINS.json ``pins``: python at major.minor, torch, transformers, tokenizers)."""
    return {k: v for k, v in (p.get("pins") or {}).items() if k in ("python",) + STACK_LIBS}


def stock_file_pins(p: dict) -> Dict[str, str]:
    """stock/PINS.json ``stock_files.files``: {path relative to the stock dir (the leading checkout dir dropped): the pinned commit's sha256}."""
    digests = (p.get("stock_files") or {}).get("files") or {}
    return {(k[len("progen2/"):] if k.startswith("progen2/") else k): v for k, v in digests.items()}


def check_pins_module():
    """``stock/check_pins.py`` loaded from the tree as a module: its ``check_stock_files`` (presence + the pinned digests, its own
    ``sha256_file``) is the ONE definition of the stock-file check — the install-time checker and the activation share it."""
    spec = importlib.util.spec_from_file_location("check_pins", os.path.join(tree_home(), "stock", "check_pins.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def weight_pins(p: dict, upstream: str) -> Dict[str, str]:
    """stock/PINS.json ``weights.<name>.files.<file>.sha256``."""
    w = (p.get("weights") or {}).get(upstream) or {}
    return {f: d.get("sha256") for f, d in (w.get("files") or {}).items() if isinstance(d, dict)}


STOCK_FILES_REFUSAL = "stock files differ from the pinned commit ({detail}) — this is not the stock the modes are defined against"
STOCK_FILES_ESCAPE = f"exit 3 (every mode refuses it: restore {STOCK_SUBDIR} from the archive beside {PINS_RELPATH}, or point {ENV_STOCK_DIR} at the pinned checkout)"
STACK_NOTE = "stack differs from the pinned one ({detail}): the levers were tested on the pinned stack"


def pins_gate(p: dict) -> Tuple[bool, dict, Optional[str]]:
    """The stock files at the stock dir vs stock/PINS.json ``stock_files.files`` (check_pins.py's own check: present, and at the
    pinned commit's digest) — a file missing or off its digest is THE refusal (``why``: STOCK_FILES_REFUSAL); the interpreter + library
    versions vs the pinned stack (``pins``) — a difference is listed in ``detail["mismatch"]`` for the stack line's note (STACK_NOTE),
    never a refusal. ``detail["hash_ms"]`` = what the hashing cost. Returns (ok, detail, why)."""
    st = stack_pins(p)
    want_files = stock_file_pins(p)
    if not st or not want_files:
        raise ActivationError(f"{pins_path()} carries no `pins` / `stock_files.files` digests: not a stock/PINS.json of this tree")
    have = stack_versions()
    detail = {"have": have, "want": st, "mismatch": [], "files": {}}
    for k, want in st.items():
        if not want:
            continue
        got = have.get(k)
        if k == "python":
            ok = got is not None and ".".join(got.split(".")[:2]) == ".".join(str(want).split(".")[:2])
        else:
            ok = got == want
        if not ok:
            detail["mismatch"].append(f"{k} have {got} want {want}")
    t0 = time.perf_counter()
    bad, fdet = check_pins_module().check_stock_files(p, stock_dir())
    detail["hash_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    got = fdet.get("sha256") or {}
    for rel, want in want_files.items():
        g = got.get(rel)
        detail["files"][rel] = "ok" if g == want else ("missing" if g is None else f"{g[:8]} != {want[:8]}")
    if bad:
        named = [f"{rel}: {v}" for rel, v in detail["files"].items() if v != "ok"] or list(bad)
        return False, detail, STOCK_FILES_REFUSAL.format(detail="; ".join(named))
    return True, detail, None


# ------------------------------------------------------------------------------------------------------------------------ the box
def _nvidia_smi_cards() -> list:
    """[(name, memory.total MiB, compute_cap)] per card nvidia-smi reports, in index order; [] when nvidia-smi does not answer."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
    except Exception:  # noqa: BLE001
        return []
    rows = []
    for l in (out.stdout.splitlines() if out.returncode == 0 else []):
        f = [x.strip() for x in l.split(",")]
        if len(f) >= 3 and f[0]:
            try:
                rows.append((f[0], int(float(f[1])), f[2]))
            except ValueError:
                continue
    return rows


def gpu_info() -> dict:
    """{name, memory_mib, cc, sm, count} of GPU 0 — from torch when it is loaded with CUDA, else nvidia-smi; name None when no GPU.
    ``memory_mib`` is nvidia-smi's `memory.total` of the card of that name whenever nvidia-smi answers — the figure GPU_CLASSES is written
    in — and torch's `total_memory` only when it does not (the driver's reservation is not in torch's figure: 81153 MiB on an
    A100-SXM4-80GB whose memory.total is 81920 MiB, inside the class's 1 % by 52 MiB; a 40 GB A100 read that way would miss its class)."""
    t = sys.modules.get("torch")
    try:
        if t is not None and t.cuda.is_available():
            pr = t.cuda.get_device_properties(0)
            cc = f"{pr.major}.{pr.minor}"
            smi = [m for (n, m, _c) in _nvidia_smi_cards() if n == pr.name]
            mem = smi[0] if smi else int(round(pr.total_memory / (1024 * 1024)))
            return {"name": pr.name, "memory_mib": mem, "cc": cc, "sm": f"sm{pr.major}{pr.minor}", "count": t.cuda.device_count(), "source": "torch"}
    except Exception:  # noqa: BLE001
        pass
    rows = _nvidia_smi_cards()
    if rows:
        name, mem, cc = rows[0]
        return {"name": name, "memory_mib": mem, "cc": cc, "sm": "sm" + cc.replace(".", ""), "count": len(rows), "source": "nvidia-smi"}
    return {"name": None, "memory_mib": None, "cc": None, "sm": None, "count": 0, "source": None}


def card_class(name: Optional[str], memory_mib: Optional[int]) -> str:
    """The class of a card: name substring AND memory within 1 % of one of the class's card memories (the card gate); `wrong_card`
    otherwise, `none` without a GPU."""
    if not name:
        return "none"
    for cls, (sub, mems) in GPU_CLASSES.items():
        if sub in name and memory_mib and any(abs(memory_mib - mem) <= mem * 0.01 for mem in mems):
            return cls
    return "wrong_card"


def stack_key(gpu: Optional[dict] = None) -> str:
    """``torch<version>-cu<cuda>-sm<cc>`` (a JIT-cache-key style stack id; ProGen2 has no Triton, so the key is torch + capability).
    A component this box cannot supply is named (``unknown:<what>``), never a bare ``unknown`` (two different absent components must not
    collide on one key): ``unknown:no_torch`` (torch not importable), ``unknown:no_cuda_suffix`` (a torch build with no ``+cuNNN`` local
    version), ``unknown:no_gpu`` (no compute capability read). None of these three reach an ACTIVE report on a real run: ``pins_gate``
    already refuses before this line when the torch version disagrees with the pinned stack, and ``activate()``'s own no-GPU check
    refuses when the card is absent — both fire whenever a component here would be unavailable. A dry run still prints the named value."""
    v = _dist_version("torch")
    if v is None:
        base, cuda = "unknown:no_torch", "unknown:no_torch"
    else:
        base, _, local = v.partition("+")
        cuda = local[2:] if local.startswith("cu") else "unknown:no_cuda_suffix"
    g = gpu or gpu_info()
    cc = g.get("cc")
    sm = cc.replace(".", "") if cc else "unknown:no_gpu"
    return f"torch{base}-cu{cuda}-sm{sm}"


# ------------------------------------------------------------------------------------------------------------------------ activation
def status() -> dict:
    rep = _STATE.get("report")
    return dict(rep) if rep else {"active": False, "reason": "progen2_opt.enable() has not run in this process"}


CUDA_ONLY = "the levers run on CUDA (fused kernels, device-resident tables, static K/V slots) and a mode is all of its levers"
DEVICE_CPU_REFUSAL = "--device {device}: " + CUDA_ONLY
NO_CUDA_REFUSAL = "no CUDA device on this box: " + CUDA_ONLY
CPU_ESCAPE = "exit 3 (`--mode off --device cpu` runs the stock on the cpu)"
SETTINGS_NOTE = "settings outside the tested defaults: {detail}"


def activate(mode: str, variant: Optional[str] = None, *, strict: bool = False, dry_run: bool = False, route: Optional[str] = None,
             device: Optional[str] = None, outside_defaults: Optional[list] = None) -> dict:
    """Resolve + gate + note + report. ``route`` names what the process will do (`sample` | `score`); ``device`` is the stock's
    own ``--device`` as given (None = its default, the first CUDA card); ``outside_defaults`` lists the stock settings given outside the
    tested defaults (`fp16=false`, …: named on the stack line, the levers engage). Returns the activation report; `strict` raises
    ActivationError instead of returning an inactive report. The activation-time line is printed exactly once here (``rep["logged"]``):
    the ``stack`` line when the levers engage (the verb prints the ``ACTIVE`` line once the composition's evidence is in: report.log_active),
    ``NOT ACTIVE … exit 3`` for a refusal by name — the stock files off the pinned commit, the kit tree not whole, no CUDA device or ``--device
    cpu`` (the levers run on CUDA and a mode is all of its levers: `--mode off` runs the stock on the cpu) — ``NOT ACTIVE`` for off, ``DRY-RUN``
    for `check`."""
    mode = (mode or "").strip().lower()
    prev = _STATE.get("report")
    if prev and not dry_run:
        if prev.get("mode") != mode or (variant and prev.get("variant") not in (None, modes.check_variant(variant))):
            why = f"already activated as mode={prev.get('mode')} variant={prev.get('variant')} in this process; a different mode/variant is refused"
            if strict:
                raise ActivationError(why)
            return dict(prev, refused=why)
        return prev
    rep: dict = {"active": False, "mode": mode, "variant": None, "route": route, "dry_run": dry_run,
                 "package_version": _dist_version("progen2_opt") or _dist_version("progen2-opt"), "stack": stack_versions(), "logged": False}
    try:
        v = modes.check_variant(variant)
        rep["variant"] = v
        rep["upstream_name"] = modes.UPSTREAM_NAME[v] if v else None
        if mode not in modes.MODES:
            raise ActivationError(f"unknown mode {mode!r} (expected {'|'.join(modes.MODES)})")
        g = gpu_info()
        rep["gpu"] = g
        rep["card_class"] = card_class(g.get("name"), g.get("memory_mib"))
        rep["stack_key"] = stack_key(g)
        rep["stack_id"] = STACK_ID
        ok_p, pdet, why_p = pins_gate(pins())                                  # THE refusal, every mode: the stock files off the pinned commit; the stack versions are a note
        rep["pins"] = pdet
        if why_p:
            rep["escape"] = STOCK_FILES_ESCAPE                                 # `--mode off` is no escape from this one: the stock route refuses it too
        if mode == "off":
            rep["kit_line"] = "stock"
            rep["optimizations"] = {"sample": [], "score": []}
            rep["reason"] = "mode off: stock in a clean subprocess"
            rep["active"] = False
            if dry_run:
                rep["would_refuse"] = why_p
            elif why_p:
                raise ActivationError(why_p)
        else:
            res = modes.resolve(mode, v, package_home())
            rep["resolution"] = res
            rep["kit_line"] = modes.describe_line(res, route)
            rep["optimizations"] = res.optimizations
            rep["kit_dirs"] = res.kit_dirs
            rep["package_home"] = package_home()
            rep["notes"] = list(res.notes)
            missing = [k for k, d in res.kit_dirs.items() if not os.path.isdir(d)]
            would = why_p or (f"kit dirs missing: {missing} — the kit tree is not whole (an install error: re-install it)" if missing else None)
            if not would and v is None:
                would = f"a size must be named (--model {'|'.join(modes.MODEL_NAMES)})"
            if pdet["mismatch"]:
                rep["notes"].append(STACK_NOTE.format(detail="; ".join(pdet["mismatch"])))
            if g.get("name") and rep["card_class"] == "wrong_card":
                rep["notes"].append(f"card {g.get('name')} ({g.get('memory_mib')} MiB) is not a house class ({', '.join(GPU_CLASSES)}); the kits' own exact-class checks were run on sm_90 only")
            dev = str(device or "cuda:0")
            if not would and dev.split(":")[0] == "cpu":                       # no lever can run on the cpu: the mode refuses by name (a mode is all of its levers)
                would, rep["escape"] = DEVICE_CPU_REFUSAL.format(device=dev), CPU_ESCAPE
            elif not would and not g.get("name"):
                would, rep["escape"] = NO_CUDA_REFUSAL, CPU_ESCAPE
            if outside_defaults:
                rep["notes"].append(SETTINGS_NOTE.format(detail=", ".join(outside_defaults)))
            if dry_run:
                rep["would_refuse"] = would
                rep["active"] = False
            elif would:
                raise ActivationError(would)
            else:
                rep["active"] = True
                rep["applied"] = "deferred"
    except ActivationError as e:
        rep["active"] = False
        rep["reason"] = rep["refused"] = str(e)                                # refused by name: the verb exits 3, whatever the mode
        report.log_activation(rep)
        rep["logged"] = True
        if not dry_run:
            _STATE["report"] = rep
        if strict:
            raise
        return rep
    report.log_activation(rep)
    rep["logged"] = True
    if not dry_run:
        _STATE["report"] = rep
    return rep
