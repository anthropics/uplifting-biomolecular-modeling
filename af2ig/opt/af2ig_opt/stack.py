"""Where things are, the gates, and the activation report.

Locations (every one an environment variable with a tree-relative default): the tree (``AF2IG_OPT_HOME``, default the directory above
``opt/``), the kit (``AF2IG_OPT_KIT``, default ``<tree>/opt/forward/af2ig_kit``), the patched checkout's driver directory (``AF2IG_DIR``,
the kit's own variable: the ``af2_initial_guess/`` directory holding ``predict_pdb.py``; no default — the checkout is built from
``stock/``) and the parameters (``AF2_PARAMS``, the kit's own variable: the directory holding
``params/params_model_1_ptm.npz``; no default — the weights are not redistributed).

Gates, all of them stack facts (``stock/check_pins.py`` is the one implementation): the named pins INSTALLED (``pins_gate``: a required
distribution that is absent refuses — nothing could run; one installed at another version never refuses: the drift is named on the printed
line, ``pins=drift`` / ``note='pins drift …'``, and the run proceeds; the CUDA-only pins required whenever a GPU run is asked for;
``AF2IG_OPT_FORCE=1`` records a refusal and runs), the checkout of the pinned tree
(``checkout_gate``: every relpath of ``af2_initial_guess/`` present — the vendored archive is the git commit's — the
8 patched relpaths byte-for-byte against the kit's own patched copies), the one parameter file present, its sha256 taken and named
pinned or not pinned against the pin (``weights_gate``: a file of other bytes RUNS). The kit's own files are the tree's commit
(``kit_home`` locates the kit by its MANIFEST.json, which the modes read).
``MODEL_OPT_TARGET_GPU`` (the config) against the GPU present is a note in the activation line and the manifest, not a gate: the kit has no
per-GPU table. No gate is derived from a measurement.

``activate(mode, route, dry_run)`` composes the report the command line prints: the mode's flags (modes.resolve) and every gate. A lever of
this kit is a flag the driver parses at its start, so the package's one route is the command line (``route="cli"``); nothing attaches to a
running process.

``applied(report, timers_path)`` is the applied set after a ``pred`` run: every lever of the composition against the driver's own
evidence of it in its ``-timers`` records (the file the package always passes). A lever without evidence is ``partial`` — the run's
exit is 3 unless ``--allow-partial`` is recorded (cli.py); no record at all makes every planned lever partial and names the absence
(``partial_detection``); a lever whose every call another lever of the same mode takes by the mode's definition is ``superseded``
(printed ``state=off reason=superseded_by:<lever>``; big's L11 at compiled lengths from the ``-trimul_chunk`` floor up), never partial.
"""
from __future__ import annotations

import importlib.metadata as _md
import importlib.util
import json
import os
import subprocess
from typing import Dict, List, Optional

from opt_core import home as _core_home

from . import ActivationError, __version__, flash_attn, fused_trimul, modes, opm, registry, triattn

ENV_HOME, ENV_KIT, ENV_FORCE, ENV_TARGET_GPU = "AF2IG_OPT_HOME", "AF2IG_OPT_KIT", "AF2IG_OPT_FORCE", "MODEL_OPT_TARGET_GPU"
ENV_CACHE = "AF2IG_OPT_CACHE_DIR"                                       # the package's cache root (default $XDG_CACHE_HOME/af2ig_opt, else ~/.cache/af2ig_opt): holds the weights digest memo weights_digests.json
ENV_AF2IG_DIR, ENV_AF2_PARAMS = "AF2IG_DIR", "AF2_PARAMS"                 # the kit's own variables (the driver reads them)
PACKAGE_ENV = (modes.ENV, ENV_HOME, ENV_KIT, ENV_FORCE, ENV_CACHE)       # the package's own switches
LEVER_WORDS = (modes.ENV + "_TRIATTN_CORE_DTYPE",)   # the one kernel word a caller may hand the driver child — L19's operand dtype (a precision lever); the 0.7.x length row / named-row words are gone (the provider decides per cell)
                                                                                                                                # cli forwards them past the stock-proof strip for kit modes (lever_words_env) — the stock child never carries an AF2IG_OPT name
DECLARED_ENV = PACKAGE_ENV + (modes.ENV_JIT_ROOT, modes.ENV_TRIMUL, modes.ENV_TMPL, modes.ENV_TIER) + LEVER_WORDS   # every AF2IG_OPT* name the package reads (+ big's trimul opt-in, L18's rows word): any other is a usage error (undeclared_env), so a mistyped switch never runs silently


def lever_words_env(environ: Optional[dict] = None) -> dict:
    """The declared lever words present in ``environ`` (the caller's), to hand to a kit-mode driver child (`cli.child_env` strips every AF2IG_OPT* name for the stock proof)."""
    environ = os.environ if environ is None else environ
    return {k: str(environ[k]) for k in LEVER_WORDS if str(environ.get(k) or "").strip()}



def undeclared_env(environ: Optional[dict] = None) -> List[str]:
    environ = os.environ if environ is None else environ
    return sorted(k for k in environ if k.startswith(modes.ENV) and k not in DECLARED_ENV)
KIT_RELPATH = os.path.join("opt", "forward", "af2ig_kit")
DRIVER = "predict_pdb.py"
UPSTREAM_DISTS = ("jax", "jaxlib", "jax-cuda12-plugin", "dm-haiku", "tensorflow-cpu", "numpy", "biopython")
_PINS_MOD = None


PYPROJECT_RELPATH = os.path.join("opt", "pyproject.toml")                 # the core pin ([tool.opt_core]) lives in the package's own pyproject


def tree_home() -> str:
    """The engine directory: AF2IG_OPT_HOME, else MODEL_OPT (run.sh exports it), else two levels above the package (opt_core.home)."""
    return _core_home.tree_home(__file__, env_home=ENV_HOME, levels=2)


def core_gate() -> dict:
    """The core pin, as the record's ``core=`` field: the facts of THE core pin gate (``af2ig_opt/_core_gate.py``, the tree's kit_template copy,
    run at every entry before any ``opt_core`` import — by the time this runs it has passed; it is called again here only for its returned facts, one
    producer of the 'pinned X, installed Y' fact). Never forced: ``AF2IG_OPT_FORCE`` does not apply to the core pin."""
    import io
    from . import TAG
    from ._core_gate import CoreGateRefused, gate
    try:
        f = gate(os.path.dirname(os.path.abspath(__file__)), tag=TAG, stream=io.StringIO())
    except CoreGateRefused as e:                                     # reachable in-process only (a caller that skipped the entry): the entry's own words
        return {"ok": False, "bad": [e.line.split("NOT ACTIVE: ", 1)[-1]], "forced": False, "detail": {"reason": e.reason}}
    inst = f["installed"]
    return {"ok": True, "bad": [], "forced": False, "detail": {"pinned": {k: f["pinned"][k] for k in ("path", "version")},
                                                          "installed": {"version": inst["version"], "root": inst["root"]}}}


def kit_home() -> str:
    k = os.environ.get(ENV_KIT) or os.path.join(tree_home(), KIT_RELPATH)
    if not os.path.isfile(os.path.join(k, modes.KIT_MANIFEST_RELPATH)):
        raise ActivationError(f"kit not found at {k} (no {modes.KIT_MANIFEST_RELPATH}); set {ENV_KIT} or install the tree editable")
    return os.path.abspath(k)


def af2ig_dir(environ: Optional[dict] = None) -> str:
    environ = os.environ if environ is None else environ
    d = environ.get(ENV_AF2IG_DIR)
    if not d:
        raise ActivationError(f"{ENV_AF2IG_DIR} unset: the patched dl_binder_design/af2_initial_guess directory")
    if not os.path.isfile(os.path.join(d, DRIVER)):
        raise ActivationError(f"{ENV_AF2IG_DIR}={d} has no {DRIVER} (apply the kit patches: README.md)")
    return os.path.abspath(d)


def params_dir(environ: Optional[dict] = None) -> str:
    environ = os.environ if environ is None else environ
    d = environ.get(ENV_AF2_PARAMS)
    if not d:
        raise ActivationError(f"{ENV_AF2_PARAMS} unset: the directory holding params/params_model_1_ptm.npz")
    return os.path.abspath(d)


def pins_module():
    """stock/check_pins.py imported by path (the one implementation of every pin check)."""
    global _PINS_MOD
    if _PINS_MOD is None:
        p = os.path.join(tree_home(), "stock", "check_pins.py")
        spec = importlib.util.spec_from_file_location("af2ig_check_pins", p)
        if spec is None or spec.loader is None:
            raise ActivationError(f"stock/check_pins.py not found at {p}")
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        _PINS_MOD = mod
    return _PINS_MOD


def pins() -> dict:
    return pins_module().read_pins()


def forced(environ: Optional[dict] = None) -> bool:
    environ = os.environ if environ is None else environ
    return (environ.get(ENV_FORCE) or "").strip() == "1"


def pins_gate(gpu: bool = True) -> dict:
    """The named pins against the installed distributions. Refuses (``ok`` False, ``bad``) only a REQUIRED DISTRIBUTION THAT IS ABSENT — nothing
    of the stack could run; a distribution installed at another version than the pin is ``drift``: named on the printed line (``pins=drift``,
    ``note='pins drift …'``) and the run proceeds — uncertainty about the environment is stated, never a reason to refuse."""
    _, detail = pins_module().check(pins(), gpu=gpu)
    absent = [f"{n}: not installed; want {d['want']}" for n, d in detail.items() if d["required"] and d["have"] is None]
    drift = [f"{n}={d['have']} (want {d['want']})" for n, d in detail.items() if d["required"] and d["have"] is not None and not d["pinned"]]
    return {"ok": not absent, "bad": absent, "drift": drift, "forced": bool(absent) and forced(), "gpu_required": gpu, "detail": detail}


def checkout_gate(environ: Optional[dict] = None, kit: Optional[str] = None) -> dict:
    """`kit`, when given (the caller's already-resolved `kit_home()`), is passed through so an overridden AF2IG_OPT_KIT is honored when
    the checkout's patched relpaths are compared to the kit's own patches/patched_files/ copies; omitted, check_checkout falls back to
    the tree-relative kit directory of PINS.json "kit" "dir"."""
    try:
        d = af2ig_dir(environ)
    except ActivationError as e:
        return {"ok": False, "bad": [str(e)], "forced": False, "dir": None, "n": 0}
    bad, n = pins_module().check_checkout(pins(), os.path.dirname(d), kit_dir=kit)
    return {"ok": not bad, "bad": bad, "forced": bool(bad) and forced(environ), "dir": d, "n": n}


def cache_dir(environ: Optional[dict] = None) -> str:
    """The package's cache root: AF2IG_OPT_CACHE_DIR, else $XDG_CACHE_HOME/af2ig_opt, else ~/.cache/af2ig_opt. It holds the weights digest memo
    (``digest_memo.MEMO_NAME`` = weights_digests.json); nothing under it decides a verdict."""
    environ = os.environ if environ is None else environ
    d = environ.get(ENV_CACHE) or os.path.join(environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"), "af2ig_opt")
    return os.path.abspath(d)


def weights_gate(environ: Optional[dict] = None, refresh: bool = False) -> dict:
    """The parameter file, pinned or not, BY DIGEST: sha256 of its full contents against ``stock/PINS.json`` ``weights.sha256`` (``check_pins.weights_verdict``, the one
    producer). The digest comes through the on-disk memo ``digest_memo.digest(path, cache_dir(), refresh)``: an entry keyed by (realpath, size, mtime_ns, inode) —
    those select the entry, they never decide the digest — written only after a full sha256; ``refresh=True`` (the ``check`` verb) hashes afresh and rewrites the entry,
    ``refresh=False`` (``pred``) reuses it and the weights line says ``(cached digest <utc>)``. A memo location that cannot be read or written decides nothing: the file
    is digested afresh and the activation notes it."""
    try:
        d = params_dir(environ)
    except ActivationError as e:
        return {"ok": False, "bad": [str(e)], "forced": False, "dir": None, "file": None}
    from . import digest_memo
    memo_dir = cache_dir(environ); memo_note = []

    def _digest(path):
        try:
            return digest_memo.digest(path, memo_dir, refresh=refresh)
        except OSError as e:                                        # the memo file/dir is unusable (read-only home, full disk): never a verdict input — hash afresh, say so
            memo_note.append(f"weights digest memo at {memo_dir} unusable ({type(e).__name__}: {e}); digested afresh")
            return digest_memo.sha256_file(os.path.realpath(path)), None
    v = pins_module().weights_verdict(pins(), d, digest=_digest)               # a present parameter file is always accepted: pinned (the pin's bytes) or not pinned (named on its own line, the run proceeds); missing = the one refusal
    bad = [v["reason"]] if v["reason"] else []
    return {"ok": not bad, "bad": bad, "forced": bool(bad) and forced(environ), "dir": d, "file": v["file"], "name": v["name"], "sha256": v["sha256"], "pinned": v["pinned"] if v["present"] else None, "pin": v.get("pin"),
            "cached_utc": v.get("cached_utc"), "memo": os.path.join(memo_dir, digest_memo.MEMO_NAME), "refreshed": bool(refresh), "memo_note": "; ".join(memo_note) or None}


def upstream_versions() -> dict:
    out = {}
    for name in UPSTREAM_DISTS:
        try:
            out[name] = _md.version(name)
        except _md.PackageNotFoundError:
            out[name] = None
    return out


def gpu_probe() -> dict:
    """The first card `nvidia-smi` lists (name + memory as the tool prints them, and its compute capability when the tool reports one): the name
    and memory are a report, never a gate — the kit has no card table; the capability keys the shared core's Pallas tile rows (`tiles_gate`)."""
    for query in ("name,memory.total,compute_cap", "name,memory.total"):      # an nvidia-smi without the compute_cap field answers the second query
        try:
            r = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
            line = (r.stdout.strip().splitlines() or [""])[0]
            if r.returncode == 0 and line:
                parts = [p.strip() for p in line.split(",")]
                return {"name": parts[0], "memory": parts[1] if len(parts) > 1 else None, "cc": (parts[2] or None) if len(parts) > 2 else None, "present": True}
        except (OSError, subprocess.SubprocessError):
            pass
    return {"name": None, "memory": None, "cc": None, "present": False}


FUSED_PALLAS_KINDS = (("attn", modes.FLAG_FUSED), ("trimul", modes.FLAG_FTRIMUL))   # the fused Pallas levers (L10 -fused_triattn, L11 -fused_trimul) and the core's row kind each reads


def tiles_gate(res: "modes.Resolution", gpu: dict, environ: Optional[dict] = None) -> dict:
    """The fused Pallas levers of a mode (L10 `-fused_triattn`, L11 `-fused_trimul`) read the shared core's tile rows for THIS card's compute
    capability — `opt_core.kernels.fpf_pallas_serve` TILE_TABLES, the float32 rows af2ig runs (`attn_cfg` / `trimul_cfg`): the card's own rows
    (`own:<cc>`) or its generation's (`safe:<N>.x` — an untested card of a known generation ENGAGES, named). A card the core has no rows for at all
    (`no-tiles:<cc>`), or rows without float32 (`no-f32-tiles:<cc>`), cannot run those levers: they STEP ASIDE by name (activate: `skipped=`, LEVER state=skipped reason=cannot_run:…; before 0.6.0 the MODE refused: `NOT ACTIVE: tiles: … select
    --mode exact`, exit 3) — a mode is all of its levers, never a run under the mode's name with a subset. Met, the gate prints nothing of its own
    (`quiet_ok`: a card with rows prints the words a check without the gate prints); `levers` names the mode's fused levers examined. A mode without
    those levers, or a card whose capability nvidia-smi does not report, is not examined (`checked` empty; the adapters' own setup refuses by name
    inside the driver if the block cannot run there, _fused.refuse)."""
    kinds = [k for k, flag in FUSED_PALLAS_KINDS if flag in res.flags]
    cc = gpu.get("cc")
    if not kinds or not cc:
        return {"ok": True, "bad": [], "forced": False, "quiet_ok": True, "cc": cc, "checked": [], "tiles": None}
    from opt_core.kernels import fpf_pallas_serve as S                  # jax-free at import and for an explicit cc (the core's contract: jax is imported inside its calls)
    bad, tiles = [], None
    try:
        cc_, _, own = S.tables_for(cc)
        tiles = S.tiles_label(cc_, own)
        for kind in kinds:
            (S.attn_cfg if kind == "attn" else S.trimul_cfg)(0, cc=cc, dtype=S.F32_DTYPE)
    except S.Refusal as r:
        bad.append(str(r))
    return {"ok": not bad, "bad": bad, "forced": False, "quiet_ok": True, "cc": cc, "checked": kinds, "tiles": tiles, "levers": [lv for lv in (registry.L10, registry.L11) if lv in res.levers]}   # quiet_ok: named on the line only when unmet (a card with rows prints the words a check without the gate prints)


def producer_gate(res: "modes.Resolution") -> dict:
    """The shared-core modules the memory line's pair-stack lever imports inside the driver (af2ig_opt.pairstack.producers): each importable from
    THIS core, checked by name before the driver starts — a core without one refuses the mode (`producer_missing:<module>`), never a stock-body run."""
    from . import pairstack
    trimul = (registry.TRIMUL in res.levers) and pairstack.parse_trimul(res.flags[res.flags.index(modes.FLAG_TRIMUL) + 1]) or None
    need = pairstack.producers(trimul)
    missing = [n for n in need if not _importable(n)]
    return {"ok": not missing, "bad": [f"producer_missing:{n}" for n in missing], "forced": False, "checked": need}


def _importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def target_gpu_note(gpu: dict, environ: Optional[dict] = None) -> Optional[str]:
    environ = os.environ if environ is None else environ
    target = (environ.get(ENV_TARGET_GPU) or "").strip()
    if not target:
        return None
    name = gpu.get("name") or ""
    if target.lower() not in name.lower():
        return f"{ENV_TARGET_GPU}={target} but the GPU present is {name or 'none'}: the kit has no per-GPU table, nothing refuses"
    return None


def activate(mode: str, *, route: str = "cli", dry_run: bool = False, strict: bool = False, gpu: bool = True, environ: Optional[dict] = None,
             precompile: Optional[int] = None, det: Optional[int] = None) -> dict:
    """The activation report for `mode` on this box: the flags and the gates (active when every gate passes or is forced). dry_run=True marks a
    `check` (nothing runs)."""
    environ = os.environ if environ is None else environ
    if mode not in modes.MODES:
        raise ValueError(f"unknown mode {mode!r} (expected {'|'.join(modes.MODES)})")
    rep: Dict[str, object] = {"active": False, "mode": mode, "route": route, "dry_run": dry_run, "package_version": __version__,
                              "upstream": upstream_versions(), "gpu": gpu_probe(), "notes": [], "target_gpu": environ.get(ENV_TARGET_GPU) or None}
    l19_word, l19_word_off = "", False
    try:
        kit = kit_home()
    except ActivationError as e:
        rep["reason"] = str(e)
        if dry_run:
            rep["would_refuse"] = rep["reason"]
        if strict:
            raise
        return rep
    rep["kit"] = kit
    try:
        l19_word = str(environ.get(registry.L19_ENV) or "").strip().lower()                # L19 : a caller's own core-dtype word wins — fp32 keeps the float32 core (L19 off BY WORD, named), bf16 is the lever's own value
        l19_word_off = bool(l19_word) and triattn.core_dtype({registry.L19_ENV: l19_word}) == "float32"
        res = modes.resolve(mode, kit, precompile=precompile, levers_off=modes.levers_off_from(environ) + ([registry.L19] if l19_word_off else []), trimul=modes.trimul_opt_in(environ) if modes.FOLDED.get(mode, mode) == "big" else None, tmpl=modes.tmpl_rows(modes.FOLDED.get(mode, mode), environ))
    except ValueError as e:
        rep["reason"] = f"mode {mode}: {e}"
        if dry_run:
            rep["would_refuse"] = rep["reason"]
        if strict:
            raise ActivationError(str(rep["reason"]))
        return rep
    if mode != "off":                                              # the kit modes: the ablation switch's census and the deployment lever's placement (ccache), before the planned set is written down
        from . import ccache as _ccache
        rep["levers_off"] = list(res.levers_off)
        rep["compile"] = registry.COMPILE_STATE; rep["no_compile"] = registry.COMPILE_WORD in res.levers_off_ignored   # QoL: compile=stock_jit on the line; `--no-compile` / MODEL_OPT_LEVERS_OFF=compile asked for no compile lever — there is none (inherent stock jit), named, never a refusal
        if res.levers_off_ignored:
            unknown = [l for l in res.levers_off_ignored if l not in registry.LEVERS and l != registry.COMPILE_WORD]   # `compile` is the tree's word, known: no kit compile lever to drop (rep["no_compile"], the line's compile= token)
            other = [l for l in res.levers_off_ignored if l in registry.LEVERS]
            rep["notes"].append(f"{modes.ENV_LEVERS_OFF}: " + "; ".join(x for x in ((f"unknown id(s) {','.join(unknown)} ignored" if unknown else ""),
                                                                              (f"{','.join(other)} not in the composition of {mode} (ignored)" if other else "")) if x))
        from . import det as _det                                     # the numerics recipe the child compiles under names the cache namespace (<root>/<key>/<recipe>/…): what --det puts in the child's XLA_FLAGS
        child_xla = (_det.env(det, dict(environ)).get(_det.ENV_XLA) if det else environ.get(_det.ENV_XLA)) or ""
        cc = _ccache.place(registry.CCACHE in res.levers, environ, rep.get("upstream"), rep.get("gpu"), os.path.join(cache_dir(environ), _ccache.DEFAULT_SUBDIR), det_level=det, child_xla_flags=child_xla)
        rep["recipe"] = cc.get("recipe")
        if cc["state"] != "on" and registry.CCACHE in res.levers:     # a root that cannot be written (or a core without the seam): the lever steps aside by name — out of the set as run, its LEVER line says skipped and why; never a refusal
            res.levers.remove(registry.CCACHE)
        rep["ccache"] = {k: v for k, v in cc.items() if k != "env"}
        rep["cache_env"] = dict(cc["env"]) if cc["state"] == "on" else {}
        rep["skipped"] = {}
        if cc["state"] == "skipped":
            rep["skipped"][registry.CCACHE] = cc.get("reason")
        if cc.get("note"):
            rep["notes"].append(str(cc["note"]))
        if "L13" in res.levers:                                    # the program store's directory beside the jax directory (ccache.programs_dir), substituted for the statement's DIR token; a directory that cannot be written: the lever steps aside by name
            pdir, why = _ccache.programs_dir(cc, os.path.join(cache_dir(environ), _ccache.DEFAULT_SUBDIR))
            i = res.flags.index(modes.FLAG_PROGRAMS)
            if pdir is None:
                del res.flags[i:i + 2]; res.levers.remove("L13"); rep["skipped"]["L13"] = why
            else:
                res.flags[i + 1] = pdir; rep["programs_dir"] = pdir
    rep.update(resolution=res, line_run=res.describe(), levers_planned=list(res.levers), precompile=res.precompile, subbatch=res.subbatch, tmpl_rows=res.tmpl_rows, core_dtype=res.core_dtype, source=res.source)
    gates = {"core": core_gate(), "pins": pins_gate(gpu=gpu), "checkout": checkout_gate(environ, kit=kit), "weights": weights_gate(environ, refresh=dry_run)}   # `check` (dry_run) digests the weights afresh and rewrites the memo entry; `pred` reuses the entry
    if rep["gpu"].get("cc") and any(flag in res.flags for _, flag in FUSED_PALLAS_KINDS):   # fast, big on a card whose compute capability nvidia-smi reports: do the core's tile rows serve the fused Pallas levers here (silent when they do; `tiles=unmet` and the mode refuses by name when they cannot run on this card)
        gates["tiles"] = tiles_gate(res, rep["gpu"], environ)
    rep["folded_into"] = getattr(res, "folded_into", None)        # fast → big (ACTIVE mode=fast→big)
    if modes.FOLDED.get(mode, mode) == "big":                    # the memory line (big): its child environment and the producers its driver hook imports
        from . import big as _big
        rep["memory_env"] = _big.child_env(res.levers)          # empty when MODEL_OPT_LEVERS_OFF drops mem_fraction
        gates["producers"] = producer_gate(res)
    rep["l19_env"] = {registry.L19_ENV: registry.L19_WORD} if registry.L19 in res.levers else {}   # L19 : the bf16 core word for the kit-mode child (cli places it; a caller's word is forwarded after it and wins)
    if l19_word_off and mode in ("fast", "big"):
        rep["l19_word_off"] = True; rep["notes"].append(f"{registry.L19_ENV}={l19_word}: the float32 attention core is kept — L19 off by word")
    rep["gates"] = {k: {kk: vv for kk, vv in v.items() if kk != "detail"} for k, v in gates.items()}
    rep["checkout"] = gates["checkout"].get("dir"); rep["weights"] = gates["weights"].get("file")
    note = target_gpu_note(rep["gpu"], environ)
    if note:
        rep["notes"].append(note)
    drift = gates["pins"].get("drift") or []
    if drift:                                                      # off the pinned stack: named, never refused — the run proceeds on the stack present
        rep["notes"].append("pins drift (stock/PINS.json; runs, named): " + "; ".join(drift[:6]) + (f" (+{len(drift) - 6} more)" if len(drift) > 6 else ""))
    if gates["weights"].get("memo_note"):
        rep["notes"].append(gates["weights"]["memo_note"])
    refusals, asides = [], {}
    for k, g in gates.items():
        if k in ("tiles", "producers") and not g["ok"] and mode != "off":   # levers that cannot run here STEP ASIDE BY NAME (LEVER state=skipped reason=cannot_run:…, `skipped=` on the activation line); the mode runs its remaining levers — never silent, never a refusal
            why = "; ".join(str(x) for x in g["bad"])
            aside = list(g.get("levers") or ([registry.L10, registry.L11] if k == "tiles" else []))
            if k == "producers":
                aside = [registry.TRIMUL] if registry.TRIMUL in res.levers else []
            for lv in aside:
                flag = modes.FLAG_TRIMUL if lv == registry.TRIMUL else registry.LEVERS[lv].flag.split()[0]
                if flag in res.flags:
                    i = res.flags.index(flag); del res.flags[i:i + (2 if lv == registry.TRIMUL else 1)]
                if lv in res.levers:
                    res.levers.remove(lv)
                rep.setdefault("skipped", {})[lv] = f"cannot_run:{why}" + (f" (compute capability {g.get('cc')})" if k == "tiles" else ""); asides[lv] = why
            g["aside"] = aside
            continue
            continue
        if not g["ok"] and not g.get("forced"):
            refusals.append(f"{k}: " + "; ".join(str(x) for x in g["bad"][:3]) + (f" (+{len(g['bad']) - 3} more)" if len(g["bad"]) > 3 else ""))
        elif not g["ok"]:
            rep["notes"].append(f"{k} gate not met, {ENV_FORCE}=1: " + "; ".join(str(x) for x in g["bad"][:3]))
    if registry.L19 in res.levers and registry.L10 not in res.levers and mode != "off":   # L19 is the bridge core's operand word: without L10 (stepped aside or switched off) it cannot engage — it steps aside BY NAME, never partial
        res.levers.remove(registry.L19); rep["l19_env"] = {}
        rep.setdefault("skipped", {})[registry.L19] = "cannot_run:needs_L10 (the triangle-attention bridge is not in the composition as run)"; asides[registry.L19] = "needs L10"
    if asides:                                                     # the composition as run lost the levers that stepped aside: the line and the planned set are rewritten, the step-aside named
        rep.update(line_run=res.describe(), levers_planned=list(res.levers))
        rep["gates"] = {k: {kk: vv for kk, vv in v.items() if kk != "detail"} for k, v in gates.items()}
        rep["notes"].append("stepped aside by name (cannot run here; the mode runs its remaining levers): " + "; ".join(f"{lv}: {w}" for lv, w in asides.items()))
    if refusals:
        rep["would_refuse" if dry_run else "reason"] = "; ".join(refusals)
        if dry_run:
            rep["reason"] = rep["would_refuse"]
        if strict and not dry_run:
            raise ActivationError(str(rep["reason"]))
        return rep
    if mode == "off":
        rep["reason"] = "mode off: stock, nothing applied (the stock line runs through af2ig-opt pred --mode off)"
    else:
        rep["active"] = not dry_run
        rep["applied"] = "argv" if not dry_run else "none"
    return rep


def timer_records(path: str) -> List[dict]:
    """The driver's -timers records (predict_pdb.py:125-137: one JSON object per line, `kind` names it) in file order; [] when the file
    is absent or holds no record."""
    out: List[dict] = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r, dict):
                    out.append(r)
    except OSError:
        return []
    return out


FUSED_CENSUS = {registry.L10: ("fused_triattn", "TriangleAttention", triattn), registry.L11: ("fused_trimul", "TriangleMultiplication", fused_trimul),
                registry.L12: ("opm_reassoc", "OuterProductMean", opm)}   # lever -> (timer record kind, module, adapter): the rebound-module levers evidenced by the adapter's trace-time census (L10, L11 fused Pallas blocks; L12 pure JAX)


def _undeclared(record: dict, expected) -> Dict[str, int]:
    """The events of a kernel lever's census record its mode does not declare — ONE rule for L8, L10 and L11 (the shared core Ledger's fail-closed
    gate read from the record's plain fields): every ``fallback_by`` reason outside the adapter's ``expected`` set, and every ``errors`` entry
    (``error:<Type>``: the kernel raised on a call). Empty = the census closes."""
    out = {str(k): int(v) for k, v in (record.get("fallback_by") or {}).items() if k not in tuple(expected)}
    out.update({f"error:{k}": int(v) for k, v in (record.get("errors") or {}).items()})
    return out


def _flash_expected() -> tuple:
    """L8's declared fallback reasons — the set ``flash_attn.configure`` hands the core Ledger (``flash_attn.EXPECTED_FALLBACKS`` over the serve
    layer's reason words; jax-free). A core whose serve layer is not importable in THIS process leaves the set unreadable: () — every fallback
    reason then counts as undeclared (fail-closed), and the partial reason says so."""
    try:
        return tuple(flash_attn.EXPECTED_FALLBACKS(flash_attn._serve()))
    except RuntimeError:
        return ()


def applied(rep: dict, timers_path: str) -> dict:
    """The applied set of a `pred` run: for every lever of the composition (`levers_planned`) the driver's own evidence in its -timers
    records — `proc_start.argv` the flags it parsed (`-fast` sets L6/L1/U1 at parse time, predict_pdb.py:109-110); a `params_on_device`
    record for L1 (:295-301); a `precompile_done` record for L7 (:757); a `subbatch` record at the planned rows for L9; for the kernel levers (L8,
    L10, L11) the adapter's census record with served >= 1 and NO undeclared event (`_undeclared`: a fallback reason outside the adapter's declared
    set, or any kernel error — one fail-closed rule for the three). Returns the fields the command line prints from:
    `levers_applied` (the evidenced set), `partial` (the levers without evidence), `partial_reasons` (lever -> why), `evidence` (per lever), `superseded` (a lever of
    the set that another lever of the same mode took every call from BY THE MODE'S DEFINITION — L11 under big at compiled lengths >= the
    `-trimul_chunk` floor, where the row-chunked TriangleMultiplication body replaces the fused block: `{lever: "superseded_by:<lever> …"}`; printed
    `state=off reason=…`, never `on`, never partial) and `partial_detection` (the record source); planned = levers_applied + partial + superseded,
    disjoint. A lever that cannot run never reaches here: it stepped aside before the driver started (the tiles gate) or the
    driver refused by name before any design (the adapters' setup, _fused.refuse). With no record at all no lever can be evidenced: every planned lever is `partial` with that
    reason and the absence is named in `partial_detection` — the run exits 3 like any partial (cli.py) unless `--allow-partial` is recorded."""
    planned = list(rep.get("levers_planned") or [])
    records = timer_records(timers_path)
    source = os.path.basename(timers_path)
    if not records:
        return {"levers_applied": [], "partial": planned, "partial_reasons": {lv: f"no driver record in {source} (the lever cannot be evidenced)" for lv in planned},
                "evidence": {}, "superseded": {}, "partial_detection": f"none (no driver record in {source}: the levers cannot be evidenced)"}
    by_kind: Dict[str, List[dict]] = {}
    for r in records:
        by_kind.setdefault(str(r.get("kind")), []).append(r)
    argv = [str(a) for a in ((by_kind.get("proc_start") or [{}])[0].get("argv") or [])]
    designs = by_kind.get("design") or []
    evidence: Dict[str, str] = {}
    partial: List[str] = []
    reasons: Dict[str, str] = {}                                   # lever -> why it is partial (the LEVER line's reason=, the partial line's detail)
    superseded_by: Dict[str, str] = {}                             # lever -> "superseded_by:<lever> …": in the set, every call taken by another lever of the mode by its definition (printed state=off, never partial)
    for lv in planned:
        lever = registry.LEVERS[lv]
        flag = lever.switch
        if lv == "L13" and flag in argv:                       # the program store (af2ig_opt.programs.Store in the driver): the per-signature program records + the store census at exit
            progs = [r for r in (by_kind.get("program") or []) if r.get("source")]
            store = (by_kind.get("program_store") or [{}])[-1]
            if not progs and not store:
                partial.append(lv); reasons[lv] = "no program record (the driver never prepared a program through the store)"; continue
            n_l = sum(1 for r in progs if r.get("source") == "loaded"); n_t = sum(1 for r in progs if r.get("source") == "traced")
            evidence[lv] = (f"argv: {flag}; program records: {len(progs)} signature(s) loaded={n_l} traced={n_t} stored={store.get('stored')} "
                            f"load_failed={store.get('load_failed')} store_failed={store.get('store_failed')} events={','.join(store.get('events') or []) or 'none'} dir={store.get('dir')}")
            continue
        if lv == registry.CCACHE:                              # the deployment lever: placed by the package in the child's environment (ccache.place) — its evidence is the placement and the directory's files before -> after the run
            from . import ccache as _ccache
            cc = rep.get("ccache") or {}
            if cc.get("state") != "on":
                partial.append(lv); reasons[lv] = f"not placed ({cc.get('reason') or 'no placement record'})"; continue
            evidence[lv] = _ccache.evidence(cc); continue
        if lv == registry.L19:                             # L19 : the fused_triattn census of the child — bridge word carries +core_bfloat16 and the bridge served the calls (providers=<row>=<n>)
            cens = by_kind.get("fused_triattn") or []
            last = cens[-1] if cens else {}
            bridge = str(last.get("bridge") or ""); prov = last.get("providers") or {}
            served_bridge = sum(int(v) for k, v in prov.items()) if isinstance(prov, dict) else 0   # the module went to the provider in bf16 — whichever arm its bf16 cell served (the row is the provider's, not this lever's evidence)
            if not cens:
                partial.append(lv); reasons[lv] = "no fused_triattn census record (the bridge adapter never reported)"; continue
            if "+core_bfloat16" not in bridge:
                partial.append(lv); reasons[lv] = f"the attention core stayed float32 (census bridge={bridge or 'absent'})"; continue
            if served_bridge < 1 and int(last.get("served") or 0) >= 1 and prov:
                partial.append(lv); reasons[lv] = f"no TriangleAttention call was served by the provider in bf16 (providers={prov}); the bf16 word had no effect"; continue
            rows_ = ",".join(f"{k.split(':', 1)[1]}={v}" for k, v in sorted(prov.items()) if str(k).startswith("triattn_xla:")) if isinstance(prov, dict) else ""
            evidence[lv] = f"fused_triattn census: core=bfloat16 rows={rows_ or 'replayed'} bridge={bridge}"; continue
        if lv == registry.MEMF:                                # the pool fraction the driver saw (proc_start.env) — an environment lever, not an argv flag
            seen = ((by_kind.get("proc_start") or [{}])[0].get("env") or {}).get(flag)
            from . import big as _big
            if str(seen) != _big.MEM_FRACTION:
                partial.append(lv); reasons[lv] = (f"proc_start.env.{flag}={seen!r}, not {_big.MEM_FRACTION}"); continue
            evidence[lv] = f"proc_start.env: {flag}={seen}"; continue
        preset = lever.in_preset and registry.PRESET_FLAG in argv
        if flag not in argv and not preset:
            partial.append(lv); reasons[lv] = (f"{flag} absent from the driver's argv (proc_start record)"); continue
        how = f"argv: {registry.PRESET_FLAG} (the driver's preset)" if preset and flag not in argv else f"argv: {flag}"
        if lv == registry.L1 and not by_kind.get("params_on_device"):
            partial.append(lv); reasons[lv] = (f"no params_on_device record (the parameters were not put on the device once)"); continue
        if lv == registry.L16:                             # the output writer's census at exit (af2ig_opt.prefetch.OutputWriter.close): jobs / errors / busy, lag and blocked seconds
            ows = by_kind.get("output_writer") or []
            if not ows:
                partial.append(lv); reasons[lv] = "no output_writer record (the writer thread was never built: the loop wrote each design's outputs itself)"; continue
            ow = ows[-1]
            how += ("; output_writer record: " + " ".join(f"{k}={ow.get(k)}" for k in ("impl", "depth", "jobs", "errors", "busy_s", "lag_s", "blocked_s", "max_pending")))
        if lv == registry.L15:                             # the prefetcher's census at exit (af2ig_opt.prefetch.Prefetcher.close): queued / taken / ready / waited, wait and worker seconds, overlap
            pfs = by_kind.get("prefetch") or []
            if not pfs:
                partial.append(lv); reasons[lv] = "no prefetch record (the worker thread was never built: the loop ran stock's serial featurisation)"; continue
            pf = pfs[-1]
            how += ("; prefetch record: " + " ".join(f"{k}={pf.get(k)}" for k in ("impl", "depth", "workers", "offdevice", "queued", "taken", "ready", "waited", "skipped", "errors", "wait_s", "work_s", "overlap_s")))
        if lv == registry.L7 and not by_kind.get("precompile_done"):
            partial.append(lv); reasons[lv] = (f"no precompile_done record (no compilation was triggered before the loop)"); continue
        if lv == registry.L8:                              # the adapter's trace-time census (cumulative; af2ig_opt.flash_attn.census): the last flash_attn record covers the run — served >= 1, every fallback reason declared, no kernel error
            cens = by_kind.get("flash_attn") or []
            last = cens[-1] if cens else {}
            served, fallback = int(last.get("served") or 0), int(last.get("fallback") or 0)
            why = ", ".join(f"{k} x{v}" for k, v in sorted((last.get("fallback_by") or {}).items())) or "none"
            if not cens:
                partial.append(lv); reasons[lv] = (f"no flash_attn census record (the adapter never reported: setup or the model trace did not run with it)"); continue
            if served < 1:
                partial.append(lv); reasons[lv] = (f"the kernel is in no compiled program: 0 Attention calls served, {fallback} left on the stock ops ({why})"); continue
            expected = _flash_expected()
            unexpected = _undeclared(last, expected)
            if unexpected:
                partial.append(lv); reasons[lv] = (f"Attention call(s) left on the stock ops for undeclared reason(s) / kernel error(s) {unexpected} ({served} served"
                                                   + ("" if expected else "; the declared set is unreadable in this process: opt_core.kernels.pallas_attn_serve not importable") + ")"); continue
            how += (f"; flash_attn census ({len(cens)} record(s)): {served} Attention call(s) traced onto the kernel {last.get('impl')} (origin {last.get('origin')}), "
                    f"{fallback} left on the stock ops (reasons: {why}); min_tokens={last.get('min_tokens')}; precision={last.get('precision')}")
        if lv in FUSED_CENSUS:                              # the fused blocks' trace-time census (af2ig_opt.triattn / fused_trimul .census): served >= 1, every fallback reason declared
            kind, module, adapter = FUSED_CENSUS[lv]
            cens = by_kind.get(kind) or []
            last = cens[-1] if cens else {}
            served = int(last.get("served") or 0); fallback = int(last.get("fallback") or 0)
            why = ", ".join(f"{k} x{v}" for k, v in sorted((last.get("fallback_by") or {}).items())) or "none"
            if not cens:
                partial.append(lv); reasons[lv] = (f"no {kind} census record (the adapter never reported: setup or the model trace did not run with it)"); continue
            superseded = None                                   # L11 above the memory line's trimul_chunk floor: every TriangleMultiplication trace runs the row-chunked body (af2ig_opt.pairstack,
            if served < 1 and fallback < 1 and lv == registry.L11:  # by design) and none reaches the fused block — the pairstack census counts those traces; that is the mode's declared interplay: the lever is
                pcs = by_kind.get("pairstack") or []                # `superseded` (printed state=off with the reason), not partial and not evidenced as on
                tc = (pcs[-1].get("trimul_chunk") if pcs else None) or {}
                if int(tc.get("engaged_traces") or 0) >= 1: superseded = tc
            if superseded is not None:
                superseded_by[lv] = (f"superseded_by:{registry.TRIMUL} — 0 {module} calls reached the fused block, {int(superseded.get('engaged_traces') or 0)} trace(s) ran the row-chunked body "
                                     f"(rows={superseded.get('rows')} min_residues={superseded.get('min_residues')}); the block serves compiled lengths below the floor ({kind} census, {len(cens)} record(s))")
                continue
            if served < 1:
                partial.append(lv); reasons[lv] = (f"the block is in no compiled program: 0 {module} calls served, {fallback} left on the stock body ({why})"); continue
            unexpected = _undeclared(last, adapter.EXPECTED_FALLBACKS)
            if unexpected:
                partial.append(lv); reasons[lv] = (f"{module} call(s) left on the stock body for undeclared reason(s) / kernel error(s) {unexpected} ({served} served)"); continue
            how += (f"; {kind} census ({len(cens)} record(s)): {served} {module} call(s) traced onto the block {last.get('impl')} (origin {last.get('origin')}), "
                    f"{fallback} left on the stock body (reasons: {why}); precision={last.get('precision')}" + (f"; tiles={last.get('tiles')}" if last.get("tiles") is not None else ""))
        if lv == registry.L18:                             # L18 : the driver's tmpl_pointwise_sub record — value == this tier's rows (rows=<n> tier=<mode> on the LEVER line)
            recs = by_kind.get("tmpl_pointwise_sub") or []
            want = rep.get("tmpl_rows")
            vals = sorted({int(r.get("value") or 0) for r in recs})
            if not recs:
                partial.append(lv); reasons[lv] = "no tmpl_pointwise_sub record (the driver never applied the template sub-batch)"; continue
            if want and vals != [int(want)]:
                partial.append(lv); reasons[lv] = f"tmpl_pointwise_sub record value(s) {vals}, not the planned {want} rows"; continue
            how += f"; tmpl_pointwise_sub record: rows={vals[0]} tier={rep.get('mode')} (stock {recs[-1].get('stock_value')})"
        if lv == registry.L9:                              # the shared policy's decision record per compiled length (af2ig_opt.subbatch.record): rows requested == rows planned
            recs = by_kind.get("subbatch") or []
            want = int(rep.get("subbatch") or modes.SUBBATCH_ROWS)
            vals = sorted({int(r.get("value") or 0) for r in recs})
            if not recs:
                partial.append(lv); reasons[lv] = (f"no subbatch record (the driver never applied the sub-batch: setup or the model call did not run with it)"); continue
            if vals != [want]:
                partial.append(lv); reasons[lv] = (f"subbatch record value(s) {vals}, not the planned {want} rows"); continue
            how += f"; subbatch record(s): {len(recs)} compiled length(s) at {want} rows per chunk (stock {recs[-1].get('stock_value')}; source {recs[-1].get('source')}; {recs[-1].get('policy')})"
        if lv == registry.TRIMUL:                              # the driver hook's census (af2ig_opt.pairstack.census): the last pairstack record covers the run
            cens = by_kind.get("pairstack") or []
            last = cens[-1] if cens else {}
            if not cens:
                partial.append(lv); reasons[lv] = (f"no pairstack census record (the driver hook never reported: install did not run)"); continue
            tc = last.get("trimul_chunk") or {}
            if not tc:
                partial.append(lv); reasons[lv] = (f"the pairstack census carries no trimul_chunk install"); continue
            eng, dis = int(tc.get("engaged_traces") or 0), int(tc.get("disengaged_traces") or 0)
            how += (f"; pairstack census ({len(cens)} record(s)): rows={tc.get('rows')} min_residues={tc.get('min_residues')} producer={tc.get('producer')} "
                    + (f"engaged_traces={eng} disengaged_traces={dis}" if eng else f"armed, disengaged on every compiled length (traces {dis}: below {tc.get('min_residues')} residues)"))
        if lv == registry.L1:
            how += "; params_on_device record"
        if lv == registry.L7:
            dones = by_kind.get("precompile_done") or []
            if not dones:
                partial.append(lv); reasons[lv] = "no precompile_done record (the precompile pass did not run)"; continue
            shapes = by_kind.get("precompile_shape") or []
            how += (f"; precompile_done record (n_shapes={dones[0].get('n_shapes')} threads={dones[0].get('threads')} dt={dones[0].get('dt')}" + (f" streaming=1 first_alone=1 background={dones[0].get('threads')} requested={dones[0].get('requested')} host_cpus={dones[0].get('host_cpus')} dt_first={dones[0].get('dt_first')}" if dones[0].get('streaming') else "") + "); "
                    f"programs prepared ahead of the loop: {[(r.get('L_compiled'), r.get('source')) for r in shapes]}")
        evidence[lv] = how
    return {"levers_applied": [lv for lv in planned if lv not in partial and lv not in superseded_by], "partial": partial, "partial_reasons": reasons,
            "evidence": evidence, "superseded": superseded_by, "partial_detection": f"{source} (the driver's own records)"}
