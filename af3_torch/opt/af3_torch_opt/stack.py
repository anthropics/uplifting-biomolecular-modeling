"""Where the tree, the kit, the two interpreters, the parameters and the caches are; the gates; the activation report.

The model is a Python API inside the kit (``af3_torch_api.build_model`` …) that needs the torch venv, and its featurisation and
output writers are the fork's, in the JAX venv — so the package (on its own interpreter, no torch, no jax) composes three
subprocesses per prediction (cli.py): ``featurise.py`` under ``AF3_TORCH_JAX_PY``, ``forward.py`` under ``AF3_TORCH_PY``,
``postprocess.py`` under ``AF3_TORCH_JAX_PY``. Deployment parameters come from the environment (``configs/<gpu>.env``):

  AF3_TORCH_PY          the torch venv's interpreter (required, no default: REQUIRED below; unset → refused by name)
  AF3_TORCH_JAX_PY      the JAX venv's interpreter — the fork for featurisation / post-processing (required, no default)
  AF3_TORCH_PARAMS_DIR  the converted OpenFold3-preview2 parameters: <dir>/of3_ported_weights.bin.zst (no default: the tree ships no weights)
  AF3_TORCH_CACHE_ROOT  the JIT caches (Triton, Inductor, JAX compilation) and opt_core's byte-gate stamps — <root>/triton, <root>/inductor, <root>/jax, <root>/verdict
  AF3_TORCH_JAX_REPO    the fork checkout whose run_alphafold.py the writers are imported from (required, no default)
  AF3_TORCH_GPU         the target GPU class (reported: gpu_target on the activation report and the ACTIVE line)
  AF3_TORCH_OPT         the mode when --mode is not given (modes.py); AF3_TORCH_OPT_HOME (or MODEL_OPT, run.sh's) / AF3_TORCH_OPT_KIT point at another tree / kit dir

The model processes get a clean environment: every ``AF3_TORCH_OPT*`` variable stripped (the mode, the levers and the DTK switch travel
as argv, printed on the COMMAND line), the cache directories set, for the torch venv the routed kernels' exports (``kernel_exports``:
the kit's fpf_trimul_v4 cell table), and for the JAX venv the fork's XLA variables (stock/PINS.json image.env) plus ``JAX_PLATFORMS=cpu`` (featurisation
and the writers never touch the GPU: the torch process owns it). The gates (``gates``): the core pin, the routed kernels' core copies
and exports, undeclared package variables, the interpreters, the kit dirs and manifest, the parameters.
"""
from __future__ import annotations

import json
import errno
import os
from typing import Dict, List, Optional, Tuple

from opt_core import gates as _cg
from opt_core import home as _ch
from opt_core import kernels as _ck
from opt_core import process as _cp
from opt_core import stock_proof as _csp

from . import digest_memo
from . import modes as _modes
from . import registry as _registry
from .outputs import sha256
from .report import PREFIX, activation_line, emit

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
KIT_RELDIR = os.path.join("opt", "forward", "af3t")       # the model, api and kernel tree
DTK_RELDIR = os.path.join("opt", "forward", "dtk")        # the DTK fused diffusion transformer
PINS_RELPATH = os.path.join("stock", "PINS.json")
PYPROJECT_RELPATH = os.path.join("opt", "pyproject.toml")     # carries the [tool.opt_core] pin
ENV_HOME = "AF3_TORCH_OPT_HOME"
STRIP_PREFIXES = ("AF3_TORCH_OPT",)                       # the package's own switches never reach a model process
DECLARED_ENV = ("AF3_TORCH_OPT", ENV_HOME, "AF3_TORCH_OPT_KIT", "AF3_TORCH_OPT_DTK")   # every name the package reads under its prefix; any other AF3_TORCH_OPT* name is refused
REFUSED_PREFIXES = ("AF3_TORCH_BIG_",)                  # prefixes the package reads NO name under (the big composition is one lever set): any such name is refused by name
DEPLOYMENT = ("AF3_TORCH_PY", "AF3_TORCH_JAX_PY", "AF3_TORCH_JAX_REPO", "AF3_TORCH_STOCK_PY", "AF3_TORCH_PARAMS_DIR", "AF3_TORCH_CACHE_ROOT", "AF3_TORCH_GPU")   # the deployment variables (configs/<gpu>.env); they reach the model processes
REQUIRED = {                                              # the deployment variables with NO default, and what each names: unset → refused BY NAME (gates; configs/<gpu>.env through `exports`); README "Variables"
    "AF3_TORCH_PY": "the torch venv's interpreter (torch + triton at the versions of stock/PINS.json check_packages.torch_python): the model process",
    "AF3_TORCH_JAX_PY": "the JAX venv's interpreter (the sokrypton/alphafold3 fork installed; jax + jaxlib at check_packages.jax_python): featurisation and the output writers",
    "AF3_TORCH_JAX_REPO": "the sokrypton/alphafold3 checkout holding run_alphafold.py (the writers postprocess.py imports)",
}
RUN_ALPHAFOLD = "run_alphafold.py"                        # the fork script the writers come from (postprocess.py load_run_alphafold reads <AF3_TORCH_JAX_REPO>/run_alphafold.py)

class ActivationError(RuntimeError):
    pass


def home() -> str:
    """The tree root (the directory holding run.sh, opt/, stock/): AF3_TORCH_OPT_HOME, else MODEL_OPT (run.sh exports it), else two levels
    above this package (opt_core.home.tree_home)."""
    return _ch.tree_home(__file__, env_home=ENV_HOME, levels=2)


def kit_home() -> str:
    return os.environ.get("AF3_TORCH_OPT_KIT") or os.path.join(home(), KIT_RELDIR)


def dtk_home() -> str:
    return os.environ.get("AF3_TORCH_OPT_DTK") or os.path.join(home(), DTK_RELDIR)


def pins() -> dict:
    with open(os.path.join(home(), PINS_RELPATH), encoding="utf-8") as f:
        return json.load(f)


def checkpoint() -> str:
    """The pinned checkpoint's file name (stock/PINS.json variants.p2.converted.file — the one spelling; stock/check_pins.py reads the same key)."""
    return pins()["variants"]["p2"]["converted"]["file"]
_STATE: Dict[str, Optional[dict]] = {"report": None}


def torch_python() -> Optional[str]:
    return os.environ.get("AF3_TORCH_PY") or None


def jax_python() -> Optional[str]:
    return os.environ.get("AF3_TORCH_JAX_PY") or None


def params_dir() -> Optional[str]:
    return os.environ.get("AF3_TORCH_PARAMS_DIR") or None


# The weights this process runs (warn-and-run): AF3_TORCH_PARAMS_DIR's ONE parameters file — the pinned name when it is there, else the
# directory's single *.bin.zst | *.bin of any name — is THE checkpoint. Its sha256 comes through the kit-local digest memo (digest_memo.py,
# `<AF3_TORCH_CACHE_ROOT>/weights_digests.json`: `check` hashes afresh, `pred` reads a matching entry; the digest alone decides); a digest equal to stock/PINS.json variants.p2.converted.sha256 is the pinned checkpoint
# (`weights=OpenFold3-preview2 sha256=<12> (pinned)` on the ACTIVE / STOCK-CLI line), any other digest RUNS with ONE
# `weights sha256=<12> UNPINNED — …` line (the kit's tests and timings cover the pinned checkpoint only). A directory with several
# parameter files runs the pinned name when present, else the first in name order (xfold's loader takes a directory's first *.bin.zst | *.bin
# too, opt/forward/af3t/af3_torch/xfold/params.py:740-743) — the weights line names it; a missing directory or file stays a refusal by name. The resolution rule has ONE producer,
# stock/check_pins.py (standard library only; loaded here as a module, pins_tool), whose --digest judges by the same words.
UNPINNED_WORDS = "the kit's tests and timings cover the pinned checkpoint only"
_WEIGHTS: Dict[tuple, dict] = {}
_WARNED: set = set()               # digests whose UNPINNED line this process has printed (once per process)
_MEMO_WARNED: set = set()          # memo directories whose UNWRITABLE line this process has printed (once per process)
_TOOL: Dict[str, object] = {}


def pins_tool():
    """stock/check_pins.py loaded as a module (standard library only; the ONE producer of the checkpoint resolution rule and of the --digest verdict)."""
    if "m" not in _TOOL:
        import importlib.util
        spec = importlib.util.spec_from_file_location("af3_torch_check_pins", os.path.join(home(), "stock", "check_pins.py"))
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        _TOOL["m"] = m
    return _TOOL["m"]


def resolve_checkpoint(params_dir_: Optional[str] = None):
    """(path, why): the checkpoint file this process runs (check_pins.resolve_checkpoint: the pinned name when there, else the directory's first
    *.bin.zst | *.bin in name order, as xfold's own loader picks), or None with the refusal sentence — unset / no parameters file."""
    d = params_dir() if params_dir_ is None else params_dir_
    if not d:
        return None, "AF3_TORCH_PARAMS_DIR is not set (the directory holding the checkpoint file — the pinned converted OpenFold3-preview2 parameters or your own *.bin.zst | *.bin; configs/<gpu>.env)"
    ck = checkpoint()
    path, _verdict = pins_tool().resolve_checkpoint(d, ck)
    if path:
        return path, None
    if not os.path.isdir(d):
        return None, f"AF3_TORCH_PARAMS_DIR={d} is not a directory"
    return None, f"AF3_TORCH_PARAMS_DIR={d} holds no parameters file ({' | '.join('*' + x for x in pins_tool().WEIGHT_SUFFIXES)}; the pinned OpenFold3-preview2 checkpoint is {ck})"


def checkpoint_path() -> Optional[str]:
    """The resolved checkpoint FILE handed to the model process (forward.py --params) and to the stock CLI (--model_dir); None when the gate refuses."""
    return resolve_checkpoint()[0]


def weights_memo_dir() -> str:
    """The directory of the on-disk weights digest memo: the cache root (`<AF3_TORCH_CACHE_ROOT>/weights_digests.json`, digest_memo.MEMO_NAME)."""
    return cache_root()


def weights_record(path: str, refresh: bool = False) -> dict:
    """{file, name, sha256, bytes, pinned, variant, cached_utc}: the checkpoint's sha256 against stock/PINS.json variants (pinned = the
    pinned OpenFold3-preview2 bytes). The digest alone decides which checkpoint this is. The digest comes through the kit-local memo (digest_memo:
    `<cache root>/weights_digests.json`, keyed by the file's realpath, size, mtime and inode — those select a memo entry, they never decide
    that): `check` passes refresh=True (hashed afresh, the entry rewritten); `pred` and the stock route read the entry when its key matches
    (cached_utc = when that digest was computed, named on the line) and hash on a miss. Once per (path, size, mtime) within a process."""
    st = os.stat(path); key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    if not refresh and key in _WEIGHTS:
        return _WEIGHTS[key]
    hashed = {}                                                     # the digest this process computed inside the memo call, kept if the memo write fails

    def _hash(p):
        hashed["sha256"] = sha256(p)
        return hashed["sha256"]
    memo_dir = weights_memo_dir(); memo_state = "ok"
    try:
        digest, cached_utc = digest_memo.digest(path, memo_dir, refresh=refresh, hasher=_hash)
    except OSError as e:                                            # the memo directory is unwritable (a read-only cache root): NAMED and hashed afresh — never a refusal
        digest, cached_utc = hashed.get("sha256") or sha256(path), None
        memo_state = f"unwritable:{errno.errorcode.get(e.errno, type(e).__name__)}"
        memo_unwritable(memo_dir, e)
    variant = next((v for v, spec in pins()["variants"].items() if spec["converted"]["sha256"] == digest), None)
    rec = {"file": os.path.abspath(path), "name": pins()["variants"][variant]["name"] if variant else os.path.basename(path), "sha256": digest,
           "bytes": st.st_size, "pinned": variant is not None, "variant": variant, "cached_utc": cached_utc, "memo": memo_state}
    _WEIGHTS[key] = rec
    return rec


def memo_unwritable(memo_dir: str, exc: BaseException) -> None:
    """The ONE line of an unwritable digest memo (a read-only cache root): `weights digest memo UNWRITABLE dir=<dir> reason=<errno>: … — hashed
    afresh, nothing memoised`, once per process and directory; the run proceeds on the digest this process computed."""
    if memo_dir in _MEMO_WARNED:
        return
    _MEMO_WARNED.add(memo_dir)
    emit(f"{PREFIX} weights digest memo UNWRITABLE dir={memo_dir} reason={errno.errorcode.get(getattr(exc, 'errno', None), type(exc).__name__)}: {exc} — hashed afresh, nothing memoised")


def weights_word(rec: dict) -> str:
    """The ACTIVE / STOCK-CLI token value: `OpenFold3-preview2 sha256=<12> (pinned)` or `<file name> sha256=<12> (unpinned)` — the same tokens
    whether the digest was computed or read from the memo (the memo state is the report's `weights_digest` field and the weights line, never this token)."""
    return f"{rec['name']} sha256={rec['sha256'][:12]} ({'pinned' if rec['pinned'] else 'unpinned'})"


def weights_digest_token(rec: dict) -> str:
    """`fresh` (hashed in this process) or `cached@<utc>` (read from the memo; the UTC time that digest was computed) — the activation report's
    `weights_digest` field; never an ACTIVE-line token (that line's grammar is fixed)."""
    return "fresh" if rec.get("cached_utc") is None else f"cached@{rec['cached_utc']}"


def weights_line(rec: dict) -> Optional[str]:
    """The ONE weights line printed before the run: `weights sha256=<12> UNPINNED — …` for any other checkpoint, `weights sha256=<12> pinned`
    for the pinned one when its digest was read from the memo, each ending `(cached digest <utc>)` on a memo hit (digest_memo.word); None for the
    pinned checkpoint hashed in this process."""
    if rec["pinned"]:
        if rec.get("cached_utc") is None:
            return None
        return digest_memo.word(f"{PREFIX} weights sha256={rec['sha256'][:12]} pinned", rec["cached_utc"])
    return digest_memo.word(f"{PREFIX} weights sha256={rec['sha256'][:12]} UNPINNED — {UNPINNED_WORDS}", rec.get("cached_utc"))


def jax_repo() -> Optional[str]:
    """The fork checkout holding run_alphafold.py (the writers postprocess.py imports): AF3_TORCH_JAX_REPO (no default)."""
    return os.environ.get("AF3_TORCH_JAX_REPO") or None


def gpu_target() -> Optional[str]:
    return os.environ.get("AF3_TORCH_GPU") or None


def config_exports() -> str:
    """The shell lines configs/<gpu>.env evaluates (`python -m af3_torch_opt exports`): one presence check per REQUIRED deployment variable
    (the one list). An unset variable prints `af3_torch_opt: <VAR> is not set — <what it names> (README Variables)` and returns 3 from the
    sourced config (run.sh's not-active exit); nothing is defaulted."""
    lines = []
    for var, what in REQUIRED.items():
        lines.append(f'[ -n "${{{var}:-}}" ] || {{ echo "af3_torch_opt: {var} is not set — {what} (README Variables)" >&2; return 3 2>/dev/null || exit 3; }}')
    return "\n".join(lines)


def cache_root() -> str:
    return os.environ.get("AF3_TORCH_CACHE_ROOT") or os.path.join(os.path.expanduser("~"), ".cache", "af3_torch_opt")


def cache_dirs(root: Optional[str] = None) -> Dict[str, str]:
    root = root or cache_root()
    return {"TRITON_CACHE_DIR": os.path.join(root, "triton"), "TORCHINDUCTOR_CACHE_DIR": os.path.join(root, "inductor"),
            "JAX_CACHE_DIR": os.path.join(root, "jax"), VERDICT_ENV: os.path.join(root, "verdict")}


# The pre-filled caches an environment may carry. `run.sh warm` fills the cache root on the box it runs on; an environment built ahead of
# time (the kit's container image) can carry the SAME trees, made by `run.sh warm --mode all` on each supported card, under the model
# interpreter's own prefix: <prefix of AF3_TORCH_PY>/share/af3_torch_cache/<key>/{triton,inductor,jax,verdict}, one <key> per (torch, CUDA, cc) in
# the form torch<version>-cu<cuda digits>-sm<cc digits> (torch2.13.0-cu130-sm90: the torch venv's torch and CUDA from stock/PINS.json, the
# card's compute capability). A pred or warm whose cache root holds no compiled-kernel file yet copies this card's tree in first (seconds,
# local files) and says so on ONE line (`CACHE cache_seed=seeded:<key> files= mb= seconds=`); a root that already holds kernels is left as
# it is (`cache_seed=kept`); the copy's Triton group manifests (`__grp__*.json`, absolute member paths of the root that compiled them) are
# rewritten to name the copies and every member checked present (`groups_rewritten= children_ok=m/t` on the same line) — a copied tree is otherwise
# read as misses; no tree for this card, no readable capability, or an unwritable root is `cache_seed=none:<reason>` — the run
# then compiles what it needs, exactly as on a cold root. Nothing here changes what a mode computes: the caches are the compilers' own
# content-addressed stores (a stale or foreign entry is never selected — it is simply not the key a kernel asks for).
VERDICT_ENV = "OPT_CORE_VERDICT_DIR"                            # opt_core's byte-gate stamp directory (kernels.trimul's native payload writes <dir>/trimul_native/gate-<key>.json once its
                                                                # vectors pass on a machine and skips the ~28 s gate in later processes): a model process gets <cache root>/verdict unless the
                                                                # caller set the variable itself ("0" = no stamp, or a directory of theirs) — so the stamp travels with the other caches
CACHE_SEED_RELDIR = os.path.join("share", "af3_torch_cache")   # under the torch venv's prefix (dirname of dirname of AF3_TORCH_PY, the venv layout <prefix>/bin/python)
CACHE_SEED_KINDS = ("triton", "inductor", "jax", "verdict")     # the sub-caches a key tree may hold (cache_dirs' four leaves); copied leaf for leaf
CACHE_KERNEL_KINDS = ("triton", "inductor")                     # the leaves whose files are compiled kernels: a root holding any is `kept`, never seeded over
GPU_TARGET_CC = {"H100": "90", "H200": "90", "GH200": "90", "A100": "80"}   # AF3_TORCH_GPU class word -> cc digits, read ONLY when nvidia-smi cannot name the card
CACHE_GROUP_PREFIX = "__grp__"                                  # Triton's FileCacheManager group manifest `__grp__<name>.json`: {"child_paths": {<member>: <ABSOLUTE path>}} — the paths
                                                                # name the cache directory of the process that compiled the group (triton/runtime/cache.py put_group: _make_path(member) =
                                                                # <cache dir>/<key>/<member>); get_group keeps only the members whose recorded path EXISTS, so a tree copied to another
                                                                # root reads as an empty group (a miss, recompiled in place) until every manifest names the copies: _relocate_groups below


def cache_seed_key(cc: str) -> str:
    """The key tree name for compute capability ``cc`` ('90' / '9.0'): torch<torch>-cu<cuda>-sm<cc> from stock/PINS.json
    (check_packages.torch_python.torch, cuda) — the form a shared JIT cache keys this stack by."""
    P = pins()
    torch_v = str(P["check_packages"]["torch_python"]["torch"]).split("+")[0]
    cuda = str(P.get("cuda") or "").replace(".", "")
    return f"torch{torch_v}-cu{cuda}-sm{str(cc).replace('.', '')}"


def cache_seed_home() -> Optional[str]:
    """<prefix of AF3_TORCH_PY>/share/af3_torch_cache, or None when AF3_TORCH_PY is unset. The prefix is taken from the variable's own
    spelling (<prefix>/bin/python), never its realpath: a venv's interpreter is a symlink into the base installation."""
    py = torch_python()
    if not py:
        return None
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(py))), CACHE_SEED_RELDIR)


def device_cc(environ: Optional[dict] = None) -> Optional[str]:
    """The compute capability digits of the card a model process launched from here runs on ('90', '80'), WITHOUT importing torch (this
    interpreter has none): ``nvidia-smi --query-gpu=compute_cap`` (the first listed device), else the AF3_TORCH_GPU class word
    (GPU_TARGET_CC), else None."""
    import shutil
    import subprocess
    exe = shutil.which("nvidia-smi")
    if exe:
        try:
            out = subprocess.run([exe, "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True, text=True, timeout=60,
                                 env=(None if environ is None else dict(environ)))
            rows = [r.strip() for r in (out.stdout or "").splitlines() if r.strip()] if out.returncode == 0 else []
        except Exception:                                      # noqa: BLE001 — a hung / broken driver tool: fall through to the class word
            rows = []
        if rows and rows[0].replace(".", "").isdigit():
            return rows[0].replace(".", "")
    word = (gpu_target() or "").upper()
    for name, cc in GPU_TARGET_CC.items():
        if word.startswith(name):
            return cc
    return None


def _count_files(d: str) -> Tuple[int, int]:
    n = b = 0
    for dp, _dns, fns in os.walk(d):
        for fn in fns:
            try:
                b += os.path.getsize(os.path.join(dp, fn)); n += 1
            except OSError:                                    # a file another process is renaming into place: not counted
                pass
    return n, b


def seed_cache_root(root: Optional[str] = None, cc: Optional[str] = None) -> dict:
    """Copy this card's pre-filled cache tree (cache_seed_home()/<key>/…) into the cache root when the root holds no compiled kernel yet.
    Returns the census the CACHE line prints: {state: seeded|kept|none, key, files, mb, seconds, src, root, reason, groups_rewritten,
    children, children_ok, groups_dropped, first_dropped}. The copy is made PORTABLE before the census returns: every Triton group manifest
    in it is rewritten to name the copies (_relocate_groups) — without that a copied tree is read as misses and recompiled in place. Never
    raises: an unwritable root or a copy that fails is `none:<reason>` and the run compiles from cold."""
    import shutil
    import time
    root = os.path.abspath(root or cache_root())                 # the manifests written below name members by absolute path, as Triton's own do
    rec = {"state": "none", "key": None, "files": 0, "mb": 0.0, "seconds": 0.0, "src": None, "root": root, "reason": None,
           "groups_rewritten": 0, "children": 0, "children_ok": 0, "groups_dropped": 0, "first_dropped": None}
    held = sum(_count_files(os.path.join(root, k))[0] for k in CACHE_KERNEL_KINDS)
    if held:
        rec.update(state="kept", files=held); return rec
    home_ = cache_seed_home()
    if not home_ or not os.path.isdir(home_):
        rec["reason"] = "no_seed_dir"; return rec
    cc = cc or device_cc()
    if not cc:
        rec["reason"] = "cc_unknown"; return rec
    key = cache_seed_key(cc); src = os.path.join(home_, key); rec.update(key=key, src=src)
    if not os.path.isdir(src):
        rec["reason"] = f"key_absent:{key}"; return rec
    t0 = time.monotonic(); files = nbytes = 0
    try:
        os.makedirs(root, exist_ok=True)
        for kind in CACHE_SEED_KINDS:
            s = os.path.join(src, kind)
            if not os.path.isdir(s):
                continue
            shutil.copytree(s, os.path.join(root, kind), dirs_exist_ok=True)
            n, b = _count_files(s); files += n; nbytes += b
        groups = _relocate_groups(root)                         # every Triton group manifest of the copy (triton/ AND inductor/) now names the copies, member for member
    except OSError as e:                                        # a read-only or vanished root: named, and the run compiles from cold
        rec.update(reason=f"unwritable:{errno.errorcode.get(e.errno, e.errno)}", files=files, seconds=round(time.monotonic() - t0, 1)); return rec
    rec.update(state="seeded", files=files, mb=round(nbytes / 1e6, 1), seconds=round(time.monotonic() - t0, 1), **groups)
    return rec


def _reroot_by_tail(recorded: str, root: str) -> Optional[str]:
    """<root>/<kind>/<tail> for a recorded absolute path whose parts hold a cache sub-tree segment (triton | inductor | jax): the LAST such
    segment splits it — whatever prefix the compiling process's root had. None when no segment is present."""
    parts = str(recorded).replace("\\", "/").split("/")
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] in CACHE_SEED_KINDS and i + 1 < len(parts):
            return os.path.join(root, parts[i], *parts[i + 1:])
    return None


def _relocate_groups(root: str) -> dict:
    """Rewrite every `__grp__*.json` under <root>/{triton,inductor,jax} so each `child_paths` entry names the member's copy under THIS root, and
    check that each named copy exists. A member is mapped by where Triton keeps it — the manifest's own directory (put_group writes every member
    beside its manifest) — else by the recorded path's tail after its cache sub-tree segment; never by replacing one known prefix (the tree may
    have been filled under any root). Census: groups_rewritten = manifests whose every member resolves (rewritten in place, atomically),
    children_ok / children = members resolved / seen, groups_dropped = manifests with a member that does not resolve (left as they are: Triton
    reads them as a miss and recompiles that kernel; the first is named). Never raises on a corrupt manifest (counted as dropped)."""
    out = {"groups_rewritten": 0, "children": 0, "children_ok": 0, "groups_dropped": 0, "first_dropped": None}
    for kind in CACHE_SEED_KINDS:
        top = os.path.join(root, kind)
        if not os.path.isdir(top):
            continue
        for dp, _dns, fns in os.walk(top):
            for fn in fns:
                if not (fn.startswith(CACHE_GROUP_PREFIX) and fn.endswith(".json")):
                    continue
                gp = os.path.join(dp, fn)
                try:
                    with open(gp, encoding="utf-8") as f:
                        data = json.load(f)
                    cps = data.get("child_paths") if isinstance(data, dict) else None
                except (OSError, ValueError):
                    cps = None
                if not isinstance(cps, dict):
                    out["groups_dropped"] += 1; out["first_dropped"] = out["first_dropped"] or os.path.relpath(gp, root); continue
                new = {}; missing = 0
                for member, recorded in cps.items():
                    out["children"] += 1
                    cand = os.path.join(dp, os.path.basename(str(recorded)))
                    if not os.path.isfile(cand):
                        alt = _reroot_by_tail(str(recorded), root)
                        cand = alt if alt and os.path.isfile(alt) else None
                    if cand is None:
                        missing += 1; new[member] = recorded
                    else:
                        out["children_ok"] += 1; new[member] = cand
                if new != cps:
                    data["child_paths"] = new; tmp = gp + ".seedtmp"
                    with open(tmp, "w", encoding="utf-8") as f:
                        json.dump(data, f)
                    os.replace(tmp, gp)                         # atomic, as Triton's own put(): no reader sees a partial manifest
                if missing:
                    out["groups_dropped"] += 1; out["first_dropped"] = out["first_dropped"] or os.path.relpath(gp, root)
                else:
                    out["groups_rewritten"] += 1
    return out


def cache_seed_line(rec: dict) -> str:
    """The ONE line of seed_cache_root's census: `CACHE cache_seed=seeded:<key> files=<n> mb=<x> seconds=<s> groups_rewritten=<g> children_ok=<m>/<t>
    [groups_dropped=<k>:<first manifest>] from=<dir> root=<root>` (g Triton group manifests rewritten to name the copies, m of t members found present) |
    `CACHE cache_seed=kept files=<n> root=<root>` (the root already holds compiled kernels) | `CACHE cache_seed=none:<reason> root=<root>`."""
    if rec["state"] == "seeded":
        dropped = f" groups_dropped={rec['groups_dropped']}:{rec['first_dropped']}" if rec.get("groups_dropped") else ""   # named once, only when a manifest kept a member that does not resolve
        return (f"{PREFIX} CACHE cache_seed=seeded:{rec['key']} files={rec['files']} mb={rec['mb']} seconds={rec['seconds']} "
                f"groups_rewritten={rec.get('groups_rewritten', 0)} children_ok={rec.get('children_ok', 0)}/{rec.get('children', 0)}{dropped} from={rec['src']} root={rec['root']}")
    if rec["state"] == "kept":
        return f"{PREFIX} CACHE cache_seed=kept files={rec['files']} root={rec['root']}"
    return f"{PREFIX} CACHE cache_seed=none:{rec['reason']} root={rec['root']}"


def forward_dir() -> str:
    """``opt/forward`` of the tree: the root the registry's kit paths are relative to."""
    return os.path.join(home(), "opt", "forward")


def touch_path(rel: str) -> str:
    """Where a registry `touches` entry lives: under opt/forward/, or — a `core:` entry, the module a routed kernel executes from — under
    the shared core's package directory (core_dir()/opt_core/)."""
    if rel.startswith(_registry.CORE):
        return os.path.join(core_dir(), "opt_core", rel[len(_registry.CORE):])
    return os.path.join(forward_dir(), rel)


def kernel_routes() -> List[str]:
    """The kernel names the model process routes to the shared core's carried copies (registry.KERNEL_ROUTES, in order)."""
    return list(_registry.KERNEL_ROUTES)


def kernel_exports() -> Dict[str, str]:
    """The environment the routed kernels read at import (opt_core.kernels.exports, by each sums file's parameter names): this kit's data
    the core copies do not carry — for fpf_trimul_v4 the kit's tuned cell table (FPF_TRIMUL_V4_CELLS=<opt/forward/af3t/…/cells.json>).
    Exported into the torch model process's environment (model_process_env); opt_core.kernels.route_check holds them present there."""
    env: Dict[str, str] = {}
    for name, r in _registry.KERNEL_ROUTES.items():
        env.update(_ck.exports(name, **{param: os.path.join(forward_dir(), rel) for param, rel in r.get("exports", {}).items()}))
    return env


def kernel_route_gate() -> List[str]:
    """Why the routed kernels cannot be served from the core on this box, before any process is launched: the core copy fails its own
    carried-file record (opt_core.kernels' live byte comparison) or a kit file a route exports is missing. The per-process gate (the bytes that
    RESOLVE in the model process == the core copy) is forward.py's route_check."""
    why = []
    for name, r in _registry.KERNEL_ROUTES.items():
        problems = _ck.verify_carry(name)
        if problems:
            why.append(f"routed kernel {name}: the core copy fails its sums file ({'; '.join(problems[:4])})")
        for param, rel in r.get("exports", {}).items():
            p = os.path.join(forward_dir(), rel)
            if not os.path.isfile(p):
                why.append(f"routed kernel {name}: export {param}= names {p}, which is missing")
    return why


def kit_sys_path(kit: Optional[str] = None, dtk: Optional[str] = None) -> List[str]:
    """The kit's own import recipe (af3_torch_api.py docstring: ``<af3t>/af3_torch``, ``<af3t>/kernels``, ``<af3t>/kernels/third_party``)
    plus the DTK dir (dtk_modules.py; its `import dtk_kernels` is the core's routed module)."""
    kit = kit or kit_home()
    return [os.path.join(kit, "af3_torch"), os.path.join(kit, "kernels"), os.path.join(kit, "kernels", "third_party"), dtk or dtk_home()]


def model_process_env(environ: Optional[dict] = None, jax: bool = False) -> dict:
    """The environment of a model process: the caller's minus every AF3_TORCH_OPT* variable, plus the cache directories and, for the torch
    venv, the routed kernels' exports (kernel_exports); for the JAX venv the fork's XLA variables (stock/PINS.json image.env) and JAX_PLATFORMS=cpu."""
    src = os.environ if environ is None else environ
    exports = {**cache_dirs(), **({} if jax else kernel_exports())}
    if VERDICT_ENV in src:
        exports.pop(VERDICT_ENV)                          # the caller's own stamp directory (or "0": no stamp) wins over <cache root>/verdict
    env = _cp.child_env(environ, strip_prefixes=STRIP_PREFIXES, export=exports)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")     # the carried kit dirs stay byte-identical: no __pycache__ under opt/forward
    env.setdefault("PYTHONUNBUFFERED", "1")
    if jax:
        env.update(pins()["image"].get("env", {}))
        env["JAX_PLATFORMS"] = "cpu"
    return env


def proof(env: dict) -> dict:
    """The stock proof of a model process environment: no AF3_TORCH_OPT* variable present (levers travel as argv); the deployment
    variables it does carry, by name."""
    present = sorted(_csp.forbidden(env, STRIP_PREFIXES))
    return {"prefixes": list(STRIP_PREFIXES), "env_present": present, "deployment": sorted(k for k in DEPLOYMENT if k in env), "ok": not present}


def tree_present() -> bool:
    """The files a mode resolution and the pins need: the kit's api and stock/PINS.json (the same checks gates() names)."""
    return os.path.isfile(os.path.join(kit_home(), _modes.API_RELPATH)) and os.path.isfile(os.path.join(home(), PINS_RELPATH))


def undeclared_env(environ=None) -> List[str]:
    """Names under the package prefix that the package does not read (a mistyped switch: AF3_TORCH_OPT_MODE=fast) and names under a prefix
    it reads nothing from (AF3_TORCH_BIG_GRAPH_DROP=0: big is one composition) — a refusal by name, never silently stripped or ignored."""
    src = os.environ if environ is None else environ
    return sorted(k for k in src if (k.startswith(STRIP_PREFIXES) and k not in DECLARED_ENV) or k.startswith(REFUSED_PREFIXES))


def core_dir() -> str:
    """The directory the imported core is loaded from (the parent of the ``opt_core`` package): what a standalone script of this package
    (postprocess.py, on the JAX venv) is told to put on its path for the package's core-routed helpers."""
    return os.path.dirname(_cg.imported_core()["package_dir"])


def padding_gate(policy: Optional[str]) -> Optional[str]:
    """Why the mode's padding policy cannot be served on this box: ``kernel_tile`` needs the shared core's shape_policy module (opt_core
    >= 0.5.1) — a core without it refuses fast / big by name instead of featurising to another row."""
    if policy == "kernel_tile":
        try:
            from opt_core import shape_policy as _sp
        except ImportError:
            import opt_core as _oc
            return f"padding policy kernel_tile needs opt_core.shape_policy (opt_core >= 0.5.1); the pinned core is {getattr(_oc, '__version__', '?')}"
        if not hasattr(_sp, "padded_len"):
            return "padding policy kernel_tile: opt_core.shape_policy has no padded_len"
    return None


def gates(need_params: bool = True) -> List[str]:
    """Why this box cannot run: the kernel routes, the missing interpreters, the parameters, the kit dirs and their manifest. Empty = go.
    (The core pin is judged before this module is imported: ``_core_gate.gate`` at every entry — a process that reached here imports the
    pinned core.)"""
    why = []
    why.extend(kernel_route_gate())
    undeclared = undeclared_env()
    if undeclared:
        why.append(f"undeclared variable(s) under the package prefix: {', '.join(undeclared)} (declared: {', '.join(DECLARED_ENV)})")
    pins_ok = os.path.isfile(os.path.join(home(), PINS_RELPATH))
    for label, py in (("AF3_TORCH_PY", torch_python()), ("AF3_TORCH_JAX_PY", jax_python())):
        if not py:
            why.append(f"{label} is not set — {REQUIRED[label]} (README Variables)")
        elif not (os.path.isfile(py) and os.access(py, os.X_OK)):
            why.append(f"{label}={py} is not an executable interpreter")
    repo = jax_repo()
    if not repo:
        why.append(f"AF3_TORCH_JAX_REPO is not set — {REQUIRED['AF3_TORCH_JAX_REPO']} (README Variables)")
    elif not os.path.isfile(os.path.join(repo, RUN_ALPHAFOLD)):
        why.append(f"AF3_TORCH_JAX_REPO={repo} holds no {RUN_ALPHAFOLD} ({REQUIRED['AF3_TORCH_JAX_REPO']})")
    for label, d in (("kit", kit_home()), ("dtk", dtk_home())):
        if not os.path.isdir(d):
            why.append(f"{label} dir {d} is missing")
    if not os.path.isfile(os.path.join(kit_home(), _modes.API_RELPATH)):
        why.append(f"the kit's api {os.path.join(kit_home(), _modes.API_RELPATH)} is missing (no LEVER_SETS to resolve a mode against)")
    if not os.path.isfile(os.path.join(home(), PINS_RELPATH)):
        why.append(f"{os.path.join(home(), PINS_RELPATH)} is missing (the tree root: AF3_TORCH_OPT_HOME / MODEL_OPT / the package's location)")
    if need_params:
        if not params_dir():
            why.append(resolve_checkpoint()[1])
        elif pins_ok:                                                # the missing pins are named above; the checkpoint's name is theirs to give
            ck_why = resolve_checkpoint()[1]
            if ck_why:
                why.append(ck_why)
    return why


from opt_core.mem import ngpu as _ngpu              # the shared core's ONE producer of the n_gpu words: the `n_gpu=P sharding=…` tokens and the refusal sentences (stdlib-only)

ROWPAIR_MODULE = "opt_core.mem.rowpair"      # the shared core's row-sharded pair stack (launcher + primitives): what n_gpu > 1 runs on (forward.py --n-gpu; rowpair_xfold.py installs it in every rank)
SHARDING_SCHEME = "rowpair"                  # this engine's scheme under n_gpu > 1 (opt_core.mem.ngpu SCHEMES)


def visible_gpus(environ: Optional[dict] = None) -> Optional[int]:
    """How many GPUs a model process launched from here sees, WITHOUT importing torch (this interpreter has none): the non-empty entries
    of CUDA_VISIBLE_DEVICES when it is set, else the device list of ``nvidia-smi --query-gpu=index``; None = unknown (no nvidia-smi)."""
    env = os.environ if environ is None else environ
    cvd = env.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        return len([x for x in cvd.split(",") if x.strip()])
    import shutil
    import subprocess
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True, timeout=60)
    except Exception:                                          # noqa: BLE001 — a hung / broken driver tool: unknown, named on the refusal
        return None
    if out.returncode != 0:
        return None
    return len([l for l in out.stdout.splitlines() if l.strip()])


def n_gpu_gate(mode: str, n_gpu, environ: Optional[dict] = None) -> Tuple[int, Optional[str], Optional[int]]:
    """(P, refusal or None, visible or None) for ``--n_gpu``, in the core's words (opt_core.mem.ngpu). P = 1 always passes (the
    single-GPU bytes; no probe). P > 1 needs mode big (ngpu.refuse_unless_big: REFUSE_MODE), P in modes.N_GPU_SUPPORTED, and P visible
    GPUs (ngpu.refuse_unless_visible on visible_gpus(): `refused: n_gpu=P visible=K`). A refusal is the NOT ACTIVE line's reason (rc 3) —
    never a smaller P."""
    try:
        P = _ngpu.check_n_gpu(n_gpu if not isinstance(n_gpu, str) else n_gpu.strip())
    except ValueError as e:
        return 0, f"refused: {e}", None
    if isinstance(n_gpu, bool):
        return 0, f"refused: n_gpu={n_gpu!r}: a positive integer is required", None
    if P == 1:
        return 1, None, None
    try:
        _ngpu.refuse_unless_big(P, mode)
    except _ngpu.NGpuRefused as e:
        return P, e.reason, None
    if P not in _modes.N_GPU_SUPPORTED:
        return P, f"refused: n_gpu={P} not in the supported set {{{','.join(str(x) for x in _modes.N_GPU_SUPPORTED)}}}", None
    k = visible_gpus(environ)
    try:
        _ngpu.refuse_unless_visible(P, 0 if k is None else k)
    except _ngpu.NGpuRefused as e:
        return P, e.reason + ("" if k is not None else " (unknown: no nvidia-smi and CUDA_VISIBLE_DEVICES unset)") + f" — use --n_gpu <= {0 if k is None else k} on this box, or expose {P} GPUs (CUDA_VISIBLE_DEVICES); a smaller P is never substituted", k
    return P, None, k


def n_gpu_fields(n_gpu: int) -> list:
    """``[("n_gpu", P), ("sharding", "rowpair"|"none")]`` — opt_core.mem.ngpu.active_pairs (scheme rowpair), the one producer of the token text."""
    return _ngpu.active_pairs(int(n_gpu), SHARDING_SCHEME)


def core_version() -> str:
    try:
        from opt_core import __version__ as v
    except Exception:                                          # noqa: BLE001
        v = "unknown"
    return v


def activate(mode: Optional[str] = None, need_params: bool = True, quiet: bool = False, n_gpu=None, refresh_weights: bool = False) -> dict:
    """Resolve the mode on this box and print the ACTIVE / NOT ACTIVE line; the report is returned and kept for ``status()``.
    Never raises on a gate: the gates run first and ``active`` is False with ``reason`` (pred exits 3) — a missing kit, tree or
    parameters never becomes an exception. An unknown or unsupported mode raises UnsupportedMode (the only exception raised here)."""
    mode = mode or _modes.mode_from_env()
    if mode not in _modes.MODES:
        _modes.resolve(mode)                                   # raises UnsupportedMode with the reason, before any file is read
    why = gates(need_params=need_params)
    NG_P, ng_why, visible = n_gpu_gate(mode, _modes.N_GPU_DEFAULT if n_gpu is None else n_gpu)
    if ng_why:
        why.append(ng_why)
    tree_ok = tree_present()                                   # the kit's api and stock/PINS.json are there: a mode resolves and the pins read
    R = _modes.resolve(mode) if tree_ok else {"lever_set": None, "levers": (), "dtk": None, "big": None, "package_levers": (), "padding": None, "levers_off": ()}
    pad_why = padding_gate(R.get("padding"))
    if pad_why:
        why.append(pad_why)
    P = pins() if tree_ok else {"upstream": {}}
    rep = {"active": not why, "mode": mode, "lever_set": R["lever_set"], "levers": list(R["levers"]), "levers_label": _modes.levers_label(R["levers"]),
           "dtk": R["dtk"], "package_levers": list(R.get("package_levers") or ()), "padding": R.get("padding"), "big": R.get("big"),
           "levers_off": list(R.get("levers_off") or ()),           # the levers MODEL_OPT_LEVERS_OFF dropped from the mode's selection for this run (modes.resolve; [] = the mode as it is)
           "kit": kit_home(), "dtk_dir": dtk_home(), "torch_python": torch_python(), "jax_python": jax_python(), "jax_repo": jax_repo(), "params_dir": params_dir(),
           "cache_root": cache_root(), "gpu_target": gpu_target(), "upstream": P["upstream"].get("commit"),
           "n_gpu": NG_P, "sharding": dict(n_gpu_fields(NG_P)).get("sharding") if (NG_P >= 1 and not ng_why) else None, "visible_gpus": visible,
           "package_version": _version(), "reason": "; ".join(why) if why else None, "weights": None, "weights_word": None, "weights_digest": None}
    if not why and need_params:                                # the gates passed: the checkpoint resolves; digest it (check: afresh; pred: through the memo) and name it
        rep["weights"] = weights_record(checkpoint_path(), refresh=refresh_weights); rep["weights_word"] = weights_word(rep["weights"]); rep["weights_digest"] = weights_digest_token(rep["weights"])
    _STATE["report"] = rep
    if not quiet:
        emit(activation_line(rep))
    if rep["weights"]:
        warn_unpinned(rep["weights"])                               # printed under --quiet too, once per process
    return rep


def warn_unpinned(rec: dict) -> None:
    """Emit the ONE weights line (`… UNPINNED — …` for any other checkpoint, `… pinned (cached digest <utc>)` for the pinned one read
    from the memo) — unconditionally (no quiet switch silences it), once per process and digest."""
    wl = weights_line(rec)
    if wl and rec["sha256"] not in _WARNED:
        _WARNED.add(rec["sha256"]); emit(wl)


def check(mode: Optional[str] = None, n_gpu=None) -> dict:
    return activate(mode, quiet=True, n_gpu=n_gpu, refresh_weights=True)   # check: the checkpoint hashed afresh, its memo entry rewritten


def status() -> Optional[dict]:
    return _STATE["report"]


def _version() -> str:
    from . import __version__
    return __version__
