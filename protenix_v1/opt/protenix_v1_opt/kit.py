"""The kit as carried in this tree: where it is, that its bytes are the ones listed, and its own arm grammar.

The kit (PROTENIX_V1_V05_ADDON v1, `opt/forward/v05_addon/`, without the kernel files the shared core carries — `opt_core/kernels` serves
them by name — and with the kit's own code in the lever adapter `ptxfpf/levers_ptx1.py` over the core, five helpers with the out-of-memory
re-raise, and the add-on's layout README) is a set of runtime
monkey-patches over the unmodified pip `protenix==1.1.0`: `ptxfpf/levers_ptx1.py` is its entry (`bind_model`, `apply(arm)`,
`describe`), `lib/`, `lib/kit112_src/` and `ptxfpf/` are the three directories its modules import from, front of sys.path (`KIT_SYS_PATHS`,
in that order). The kit is identified by the git commit carrying it (no in-repo manifest restates it): `check_files()` is the gate every
activation runs — the required entry points present (`REQUIRED_RELPATHS`); a separate test (no gate, since it only matters to the tree
the repo itself carries) keeps the kit tree free of bytecode caches and symlinks.

Nothing here transcribes the kit's grammar: `lever_grammar()` reads the trimul options and the lever names out of `levers_ptx1.apply`
itself (`ast`, no torch), and `parse_arm()` validates an arm string against them exactly as the kit would.
"""
from __future__ import annotations

import ast
import fnmatch
import os
import sys
from typing import List, Optional, Tuple

from opt_core import gates as G
from opt_core import home as H

ENV_KIT = "PROTENIX_V1_OPT_KIT"                                   # an explicit kit directory (else opt/forward/v05_addon beside this package)
ENV_TREE = "MODEL_OPT"                                            # run.sh convention: this model's directory (protenix_v1/)
KIT_DIRNAME = "v05_addon"
KIT_SYS_PATHS = ("lib", os.path.join("lib", "kit112_src"), "ptxfpf")   # the kit's import roots, in sys.path order
LEVERS_RELPATH = os.path.join("ptxfpf", "levers_ptx1.py")          # the lever adapter (protenix 1.x)
LEVERS_MODULE = "levers_ptx1"
TRIATTN_FLOOR_ENV = "PTX_TRIATTN_MIN_TOKENS"                          # the flash triangle-attention size-floor variable: stack.export_triattn_floor exports modes.TRIATTN_FLOOR under this name when a mode declares a floor (None in every shipped mode: nothing is exported, no kit code reads it)
DETPATCH_RELPATHS = (os.path.join("lib", "detpatch", "det_segment_reduce.py"), os.path.join("lib", "detpatch", "scatter_utils.py"))
REQUIRED_RELPATHS = (LEVERS_RELPATH,) + DETPATCH_RELPATHS + ("README.md",)   # the kit files the package reads by path; check_files()'s existence gate


def opt_home() -> str:
    return H.opt_home(__file__, env_tree=ENV_TREE, levels=2)         # <tree_home>/opt (opt_core.home: $MODEL_OPT, else two levels above this package)


def tree_home() -> str:
    return H.tree_home(__file__, env_tree=ENV_TREE, levels=2)        # $MODEL_OPT, else two levels above this package (opt/protenix_v1_opt/ -> protenix_v1/)


def kit_home() -> str:
    env = os.environ.get(ENV_KIT)
    return os.path.abspath(env) if env else os.path.join(opt_home(), "forward", KIT_DIRNAME)


def kit_sys_paths(kit: Optional[str] = None) -> List[str]:
    kit = kit or kit_home()
    return [os.path.join(kit, p) for p in KIT_SYS_PATHS]


sha256_file = G.sha256_file

# The tree rule for counting and staging kit files: every file except __pycache__/ and *.py[cod] (and the other dirs/files a repo never
# carries as kit bytes).
_IGNORED_DIRS = ("__pycache__", "*.egg-info", ".venv", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".ipynb_checkpoints", ".idea", ".vscode", ".git")
_IGNORED_FILES = ("*.py[cod]", ".python-version", ".env", ".env.*", "*.pem", "*.key", ".DS_Store")


def _ignored_dir(name: str) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in _IGNORED_DIRS + _IGNORED_FILES)   # a pattern without a trailing slash matches a directory too (git's rule)


def _ignored_file(name: str) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in _IGNORED_FILES)


def tree_files(root: str) -> list:
    """The relative paths of the tree's files under the rule, sorted."""
    out = []
    for d, dns, fns in os.walk(root):
        dns[:] = sorted(x for x in dns if not _ignored_dir(x))
        out += [os.path.relpath(os.path.join(d, f), root) for f in fns if not _ignored_file(f)]
    return sorted(out)


def autoload_pth_source() -> str:
    """The autoload .pth the wheel ships at its root (opt/<pkg>_autoload.pth — exactly one beside pyproject, the build backend's rule)."""
    import glob as _glob
    pths = sorted(_glob.glob(os.path.join(opt_home(), "*_autoload.pth")))
    if len(pths) != 1:
        raise RuntimeError(f"expected exactly one *_autoload.pth under {opt_home()}, found {pths}")
    return pths[0]


class AutoloadHookError(RuntimeError):
    """The autoload hook is not live in a fresh interpreter — the environment route (`PROTENIX_V1_OPT=<mode> protenix pred`) would run
    stock silently. The message carries the three-way diagnostic: absent / present but not processed / a stale copy."""


HOOK_MODULE = "protenix_v1_opt._autoload"        # what the .pth imports (opt/protenix_v1_opt_autoload.pth); live = in sys.modules at interpreter start


def autoload_hook_probe() -> dict:
    """The site-processed .pth's EFFECT, not its presence: a FRESH interpreter (this one's executable, PROTENIX_V1_OPT removed from its
    environment, no -I: the user site is processed when enabled, exactly as the route's own `python` processes it) reports whether
    HOOK_MODULE is in sys.modules at start, with the site dirs that interpreter processes (its site-packages, its user site and whether
    the user site is enabled) for the diagnostic. {live, site, user_site, user_enabled, prefix}."""
    import json as _json
    import subprocess as _sp
    code = ("import sys, site, json; print(json.dumps({'live': %r in sys.modules, 'site': list(site.getsitepackages()), "
            "'user_site': site.getusersitepackages(), 'user_enabled': bool(site.ENABLE_USER_SITE), 'prefix': sys.prefix}))" % HOOK_MODULE)
    env = {k: v for k, v in os.environ.items() if k != "PROTENIX_V1_OPT"}                    # env -u PROTENIX_V1_OPT python -c …
    r = _sp.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise AutoloadHookError(f"autoload: the fresh-interpreter probe ({sys.executable}) failed rc {r.returncode}: {r.stderr.strip()[-300:]}")
    return _json.loads(r.stdout.strip().splitlines()[-1])


def autoload_hook_diagnostic(probe: dict) -> str:
    """Why the hook is not live, one of three: absent from the searched sites / present but not processed (a copy beside a PYTHONPATH or
    sys.path entry that site.py never processes, the user site when this interpreter has it disabled, or a .pth line that failed at
    site time) / a stale copy (bytes differ from the tree's). The user site counts as a searched site when the interpreter enables it
    (site.ENABLE_USER_SITE: a `pip install --user` / PYTHONUSERBASE install is a live install there)."""
    src = autoload_pth_source(); name = os.path.basename(src); want = open(src, "rb").read()
    user_site = probe.get("user_site"); user_enabled = bool(probe.get("user_enabled"))
    searched = [d for d in list(probe.get("site") or []) if d] + ([user_site] if user_enabled and user_site else [])
    in_site = [os.path.join(d, name) for d in searched if os.path.isfile(os.path.join(d, name))]
    stale = [h for h in in_site if open(h, "rb").read() != want]
    if stale:
        return f"a stale copy: {', '.join(stale)} differs from the tree's {src} — reinstall: `run.sh install`"
    if in_site:
        return (f"present but not processed: {', '.join(in_site)} sits in a site dir yet {HOOK_MODULE} is not in sys.modules at interpreter start "
                f"— its import line failed at site time (site.py prints the error and continues; the package must be importable when site processes it: "
                f"run `python -c pass` and read stderr) — reinstall: `run.sh install`")
    if user_site and not user_enabled and os.path.isfile(os.path.join(user_site, name)):
        return (f"present but not processed: {os.path.join(user_site, name)} is in the user site, which this interpreter has disabled "
                f"(site.ENABLE_USER_SITE is False: a venv without system site, -s, or PYTHONNOUSERSITE) — install into the interpreter's own site-packages: `run.sh install`")
    beside = []
    for d in [x for x in (os.environ.get("PYTHONPATH") or "").split(os.pathsep) if x] + [x for x in sys.path if x]:
        cand = os.path.join(d, name)
        if os.path.isfile(cand) and cand not in beside and d not in searched and os.path.realpath(cand) != os.path.realpath(src):   # the tree's own source is not an install
            beside.append(cand)
    if beside:
        return (f"present but not processed: {', '.join(beside)} lies beside a PYTHONPATH / sys.path entry that site.py never processes (not a site dir) "
                f"— install into site: `run.sh install`")
    return (f"absent from the searched sites ({', '.join(searched) or '-'}{'; user site ' + user_site if user_site else ''}) — install the package into this "
            f"interpreter ({probe.get('prefix') or sys.prefix}): `run.sh install` (pip install -e {os.path.join(tree_home(), '..', 'common', 'opt_core')} -e {opt_home()})")


def autoload_hook_check() -> dict:
    """The autoload-hook gate (the environment route only — run.sh's guard calls it when PROTENIX_V1_OPT names a kit mode and no --mode was given): the hook
    must be LIVE in a fresh interpreter (autoload_hook_probe), else AutoloadHookError with the diagnostic (autoload_hook_diagnostic)."""
    probe = autoload_hook_probe()
    if not probe.get("live"):
        raise AutoloadHookError(f"autoload: {HOOK_MODULE} is not in sys.modules at interpreter start (a fresh {sys.executable}) — the environment route "
                                f"(PROTENIX_V1_OPT=<mode> protenix pred, no --mode) would run stock silently; {autoload_hook_diagnostic(probe)}")
    return probe


def autoload_hook_main() -> None:
    """`python -c "from protenix_v1_opt import kit; kit.autoload_hook_main()"` — run.sh's environment-route guard: the NOT ACTIVE line on
    refusal, exit 3; silent when the hook is live."""
    from . import report as R
    try:
        autoload_hook_check()
    except AutoloadHookError as e:
        R.log(R.not_active_line(str(e)))
        sys.exit(R.EXIT_NOT_ACTIVE)


def stock_url_module() -> str:
    """The installed stock distribution's download-URL module (stock/PINS.json data_urls_module: `protenix/web_service/dependency_url.py`,
    the URL dict the upstream's boot-time download reads), located through the distribution's metadata — never through sys.modules, which a
    test may stub — without executing the package."""
    import importlib.metadata
    from . import stack as S
    st = S.pins()["stock"]
    rel = st["data_urls_module"].split(" ")[0]
    return str(importlib.metadata.distribution(st["package"]).locate_file(rel))


class FrozenWeightsError(RuntimeError):
    """The weights root lacks a file the upstream would otherwise download at boot (checkpoint or data cache) — refused by name before
    any route can reach the download fallback. A PRESENT checkpoint is never refused: it is digested and named pinned or not pinned."""


NOT_PINNED_WORDS = "NOT PINNED — the kit's numerics and speed statements hold for the pinned weights only"
_DIGESTS: dict = {}                                                 # digest_memo.stat_key(path) -> (sha256, cached_utc): the in-process front of the on-disk memo
DIGEST_MEMO_ENV = "TRITON_CACHE_DIR"                                # the kit's cache root (configs/h100.env): the weights digest memo lives there as digest_memo.MEMO_NAME
DIGEST_MEMO_DEFAULT = os.path.join(os.path.expanduser("~"), ".cache", "protenix_v1_opt")   # without a config: a directory of this user's own, never a fixed name in the shared temporary directory
WEIGHTS_MEMO_ENV = "PROTENIX_V1_OPT_WEIGHTS_MEMO"                   # an explicit memo directory (first choice when set); the launching process exports it to its rank processes when the
                                                                    # cache root is unwritable (a read-only cache mount): the private writable directory holding the entry it just computed
_ANNOUNCED: set = set()                                             # digests whose weights line this process already printed


def _stock_option(args, opt):
    """The last `--opt <v>` / `--opt=<v>` value on the stock arguments, else None."""
    val = None
    for j, a in enumerate(args):
        if a == opt and j + 1 < len(args):
            val = args[j + 1]
        elif a.startswith(opt + "="):
            val = a.split("=", 1)[1]
    return val


def stock_model_name(argv=None) -> str:
    """The model the stock loads: `--model_name <n>` on the stock arguments, else the stock default (stock/PINS.json stock.model_name)."""
    from . import stack as S
    args = list(sys.argv[1:] if argv is None else argv)
    return _stock_option(args, "--model_name") or S.pins()["stock"]["model_name"]


def stock_checkpoint(root: str, argv=None) -> str:
    """The checkpoint file the stock arguments resolve to: `<load_checkpoint_dir>/<model_name>.pt` with `--load_checkpoint_dir <d>` when
    given, else `<PROTENIX_ROOT_DIR>/checkpoint` (runner/inference.py:338 `f"{configs.load_checkpoint_dir}/{configs.model_name}.pt"`,
    configs/configs_inference.py:23-29) — for the stock defaults this is `<root>/<stock.checkpoint>`."""
    from . import stack as S
    args = list(sys.argv[1:] if argv is None else argv)
    ckdir = _stock_option(args, "--load_checkpoint_dir") or os.path.join(root, os.path.dirname(S.pins()["stock"]["checkpoint"]))
    return os.path.join(ckdir, f"{stock_model_name(args)}.pt")


def digest_memo_dir() -> str:
    """The directory of the on-disk weights digest memo (`digest_memo.MEMO_NAME` = weights_digests.json): `$PROTENIX_V1_OPT_WEIGHTS_MEMO`
    when set, else the kit's cache root `$TRITON_CACHE_DIR` (configs/h100.env), else ~/.cache/protenix_v1_opt — one file beside the Triton JIT
    cache, shared by the launching process and its rank processes on the box and by every later process that mounts the same cache root."""
    return os.environ.get(WEIGHTS_MEMO_ENV) or os.environ.get(DIGEST_MEMO_ENV) or DIGEST_MEMO_DEFAULT


def _launcher_digest(path: str, refresh: bool) -> Tuple[str, Optional[str]]:
    """digest_memo.digest in the launching process (or any single-process run), hashing at most once. An UNWRITABLE memo directory (a
    read-only cache mount, an uncreatable path) is named on the transcript, never a refusal: the digest just computed stands, the entry is
    written to a process-private writable directory instead, and that directory is exported as $PROTENIX_V1_OPT_WEIGHTS_MEMO so this
    process's rank processes (which never hash) read it there. A failure to READ the checkpoint itself propagates."""
    import tempfile
    from . import digest_memo as DM
    from . import report as R
    memo_dir = digest_memo_dir()
    computed = {}

    def hasher(p):                                                      # the digest is computed once even when the memo store fails after it
        if "sha256" not in computed:
            computed["sha256"] = sha256_file(p)
        return computed["sha256"]
    try:
        return DM.digest(path, memo_dir, refresh=refresh, hasher=hasher)
    except OSError as e:
        if "sha256" not in computed:                                     # raised while reading the checkpoint, not while storing the memo
            raise
        private = tempfile.mkdtemp(prefix="protenix_v1_opt_weights_memo_")
        got = DM.digest(path, private, refresh=True, hasher=hasher)      # the entry lands in the private directory (no second hash: hasher answers from `computed`)
        os.environ[WEIGHTS_MEMO_ENV] = private                          # inherited by this process's rank processes (digest_memo.rank_digest reads it there)
        R.log(f"{R.PREFIX} weights digest memo unwritable at {memo_dir} ({e.__class__.__name__}: {e.strerror or e}) — hashed afresh; "
              f"this run's memo is {private} ({WEIGHTS_MEMO_ENV})")
        return got


def checkpoint_digest(path: str, refresh: bool = False) -> Tuple[str, Optional[str]]:
    """(sha256, cached_utc) of a checkpoint file through the on-disk memo (digest_memo: the digest of the file's full bytes decides;
    realpath/size/mtime/inode select a memo entry, never decide). refresh=True (the `check` verb) hashes afresh and rewrites the
    entry; a rank process of a multi-GPU run never hashes — it reads the entry the launching process wrote (digest_memo.rank_digest, a
    miss raises by name). cached_utc is None when this process (or its launcher chain) computed the digest, else the UTC time the
    memoised digest was computed. The in-process table answers repeat calls without re-hashing."""
    from opt_core.mem.rowpair import launch as L                       # the rank environment (ROWPAIR_WORLD): > 1 only inside a rank process
    from . import digest_memo as DM
    key = DM.stat_key(path)
    if refresh or key not in _DIGESTS:
        _DIGESTS[key] = DM.rank_digest(path, digest_memo_dir()) if L.world_size() > 1 else _launcher_digest(path, refresh)
    return _DIGESTS[key]


def weights_line(w: dict) -> str:
    """`weights=<name> sha256=<12> (pinned)` for the pinned checkpoint, else `weights sha256=<12> NOT PINNED — the kit's numerics and speed
    statements hold for the pinned weights only` (the run proceeds either way); `… (cached digest <utc>)` appended when the digest was
    served from the on-disk memo (digest_memo.word)."""
    from . import digest_memo as DM
    line = f"weights={w['weights']} sha256={w['sha256'][:12]} (pinned)" if w["pinned"] else f"weights sha256={w['sha256'][:12]} {NOT_PINNED_WORDS}"
    return DM.word(line, w.get("cached_utc"))


def frozen_weights_check(root: Optional[str] = None, argv=None, announce: bool = True, refresh: bool = False) -> dict:
    """The weights boot gate (every route, before the upstream runs). The weights root (`stock/PINS.json` stock.root_env = the upstream's
    PROTENIX_ROOT_DIR) must carry the checkpoint the stock arguments resolve to (stock_checkpoint: `--load_checkpoint_dir` / `--model_name`) and every
    data cache the upstream would otherwise DOWNLOAD SILENTLY at boot (runner/inference.py:291-347 `download_inference_cache`; the
    upstream ships no switch that disables the fallback — the only knob is the root itself): a MISSING root, checkpoint or cache is refused
    by name (FrozenWeightsError). A present checkpoint is always accepted: it is digested (checkpoint_digest) against stock.checkpoint_sha256
    through the on-disk digest memo (refresh=True: the `check` verb, hashed afresh) — a match is `weights=<name> sha256=<12> (pinned)`, anything else
    `weights sha256=<12> NOT PINNED — …` printed once per process (announce) and the run proceeds; the record {root, checkpoint, checkpoint_bytes,
    weights, sha256, cached_utc, pinned, caches} lands in the activation's gate details."""
    import importlib.metadata
    import importlib.util
    from . import stack as S                                          # stack.pins: the one reader of stock/PINS.json
    st = S.pins()["stock"]
    root = root if root is not None else os.environ.get(st["root_env"])
    if not root or not os.path.isdir(root):
        raise FrozenWeightsError(f"{st['root_env']}={root!r} is not a directory — the upstream would download the weights at boot (runner/inference.py:291-347); mount the pinned weights")
    name = stock_model_name(argv)
    ckpt = stock_checkpoint(root, argv)                              # <--load_checkpoint_dir or root/checkpoint>/<--model_name or the default>.pt == <root>/<stock.checkpoint> for the stock defaults
    if not os.path.isfile(ckpt):
        raise FrozenWeightsError(f"{ckpt} is absent — the upstream would download the checkpoint at boot (runner/inference.py:338-347); mount the pinned weights")
    have = os.path.getsize(ckpt)
    try:
        url_py = stock_url_module()
    except importlib.metadata.PackageNotFoundError:
        raise FrozenWeightsError(f"the stock package {st['package']} is not installed in this interpreter — nothing to gate; install the pinned stack")
    if not os.path.isfile(url_py):
        raise FrozenWeightsError(f"the installed {st['package']} carries no {url_py} (stock/PINS.json data_urls_module) — not the pinned stock")
    mod_spec = importlib.util.spec_from_file_location("protenix_dependency_url", url_py); mod = importlib.util.module_from_spec(mod_spec); mod_spec.loader.exec_module(mod)
    caches = {}
    for cache_name in st["data_files"]:
        if cache_name not in mod.URL:                                      # the template caches (use_template=true only): not a pinned file
            continue
        path = os.path.join(root, "common", os.path.basename(mod.URL[cache_name]))   # configs/configs_data.py:251-258: <root>/common/<basename>
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            raise FrozenWeightsError(f"{path} ({cache_name}) is absent — the upstream would download it at boot (runner/inference.py:300-316); mount the pinned weights")
        caches[cache_name] = path
    digest, cached_utc = checkpoint_digest(ckpt, refresh=refresh)
    w = {"root": root, "checkpoint": ckpt, "checkpoint_bytes": have, "weights": name, "sha256": digest, "cached_utc": cached_utc,
         "pinned": digest == st["checkpoint_sha256"] and name == st["model_name"], "caches": caches}
    if announce and digest not in _ANNOUNCED:
        from . import report as R
        _ANNOUNCED.add(digest)
        R.log(f"{R.PREFIX} {weights_line(w)}")
    return w


def frozen_weights_main() -> None:
    """`python -c "from protenix_v1_opt import kit; kit.frozen_weights_main()"` — configs/h100.env's boot probe: the gate — the weights
    line (pinned / NOT PINNED, exit 0 either way), or the NOT ACTIVE line on a missing file and exit 3."""
    import sys as _sys
    from . import report as R
    try:
        frozen_weights_check()
    except FrozenWeightsError as e:
        R.log(R.not_active_line(f"frozen weights: {e}"))
        _sys.exit(R.EXIT_NOT_ACTIVE)


def check_files(kit: Optional[str] = None) -> dict:
    """Every required file present (REQUIRED_RELPATHS: the entry point, the detpatch pair, the add-on README, the
    lever files). No manifest is read or written: the kit is identified by the git commit carrying it. Returns
    {"ok", "required", "missing": [...], "files": <live count from tree_files(kit)>}."""
    kit = kit or kit_home()
    missing = [rel for rel in REQUIRED_RELPATHS if not os.path.isfile(os.path.join(kit, rel))]
    return {"ok": not missing, "required": len(REQUIRED_RELPATHS), "missing": missing, "files": len(tree_files(kit))}


def levers_file(kit: Optional[str] = None) -> str:
    return os.path.join(kit or kit_home(), LEVERS_RELPATH)


def _levers_module(kit: Optional[str] = None) -> ast.Module:
    with open(levers_file(kit), "r", encoding="utf-8") as fh:
        return ast.parse(fh.read(), filename=levers_file(kit))


def _apply_function(kit: Optional[str] = None, tree: Optional[ast.Module] = None) -> ast.FunctionDef:
    tree = tree if tree is not None else _levers_module(kit)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "apply":
            return node
    raise RuntimeError(f"{levers_file(kit)}: no top-level `apply`")


def lever_grammar(kit: Optional[str] = None) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """(trimul options, lever names) as `levers_ptx1` spells them: the tuple literal of `apply`'s `parts[0] in (...)` test and the module's
    `LEVER_NAMES = (...)` tuple literal (the one list `apply` validates against). Read from the kit file, never transcribed."""
    tree = _levers_module(kit)
    fn = _apply_function(kit, tree)
    trimul: Optional[Tuple[str, ...]] = None
    levers: Optional[Tuple[str, ...]] = None
    for node in ast.walk(fn):
        if isinstance(node, ast.Compare) and trimul is None and len(node.ops) == 1 and isinstance(node.ops[0], ast.In):
            rhs = node.comparators[0]
            if isinstance(rhs, ast.Tuple) and rhs.elts and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in rhs.elts):
                trimul = tuple(e.value for e in rhs.elts)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "LEVER_NAMES" for t in node.targets) \
                and isinstance(node.value, ast.Tuple) and node.value.elts and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.value.elts):
            levers = tuple(e.value for e in node.value.elts)
    if trimul is None or levers is None:
        raise RuntimeError(f"{levers_file(kit)}: apply() no longer spells its trimul tuple as a literal / the module has no `LEVER_NAMES = (...)` tuple literal")
    return trimul, levers


def parse_arm(arm: str, kit: Optional[str] = None) -> Tuple[str, Tuple[str, ...]]:
    """(trimul, levers) of an arm string under the kit's grammar (`<trimul>[+lever...]`, `stock` alone = the stock path); ValueError otherwise."""
    trimul_opts, lever_opts = lever_grammar(kit)
    parts = [p for p in arm.replace(",", "+").split("+") if p]
    if not parts or parts[0] not in trimul_opts:
        raise ValueError(f"arm {arm!r}: the first element must be one of {trimul_opts}")
    levers = tuple(parts[1:])
    unknown = sorted(set(levers) - set(lever_opts))
    if unknown:
        raise ValueError(f"arm {arm!r}: unknown levers {unknown} (the kit knows {sorted(lever_opts)})")
    if len(set(levers)) != len(levers):
        raise ValueError(f"arm {arm!r}: a lever is repeated")
    return parts[0], levers
