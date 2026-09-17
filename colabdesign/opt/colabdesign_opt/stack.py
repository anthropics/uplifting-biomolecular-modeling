"""Activation: the tree's paths, the stock pins, the core pin, the GPU, and the application of a mode through the package's ONE lever installer.

`activate(mode, route)` resolves the mode against the table (modes.resolve), gates — the installed `colabdesign` at the pinned commit (its
dist-info `direct_url.json`, read without importing it: another commit changes what "stock" means), the imported `opt_core` = the core
`opt/pyproject.toml` `[tool.opt_core]` pins (kit-route modes), no late activation — and applies the mode through the package's ONE lever
installer on the in-process routes (`api`, `env`: `levers.install(mode)`). The kit rule: a kit accepts everything stock accepts — a LEVER that
cannot engage here steps aside BY NAME (`LEVER name=<x> state=skipped reason=cannot_run|no_attention_kernel …`, levers.py) and the mode runs
with the rest of its set: the report's `levers` = what is actually on, `skipped` = {lever: reason} (the ACTIVE line's `levers=` /
`skipped=`). What refuses the MODE by name (exit 3) is configuration only: an unknown word, the variable disagreeing with --mode, the
colabdesign commit off its pin, the core pin gate, late activation. On the dry-run routes (`check`, the design parent: nothing is installed)
`skipped` is FORETOLD from the one fact the parent has — no GPU visible (nvidia-smi) means the levers that need the GPU backend
(levers.NEEDS_GPU: the Pallas kernels, the GPU-keyed compile cache) will step aside in the arm, and with them the levers that compose on an
attention kernel (levers.NEEDS_ATTENTION_KERNEL); with a card visible nothing is foretold and the arm's LEVER lines are the record. The rest
of the environment is NAMED, never a reason to refuse or to drop a lever: the card on the ACTIVE line's `gpu=` (any card — one other than the
tested one, or below the kernel's compute-capability floor `gpu.cc_min`, is named, not refused: the levers engage and the card's note says so), a
stack distribution off its pin on `stack_drift=`.
`activate(..., dry_run=True)` resolves, gates and reports without applying anything (`check`), and works without jax or colabdesign installed.

Late activation is refused by name when (1) a lever's marker is already on the upstream objects (each lever module's `installed()`),
put there by another caller, or (2) a `mk_af_model` instance exists (found by a `gc` scan): its executables
were built by the `_prep_model` in place at its `prep_inputs` (prep.py:25-35, :268) and a replacement cannot reach them.
"""
from __future__ import annotations

import gc
import importlib
import importlib.metadata
import json
import os
import sys
from typing import Dict, List, Optional


from opt_core import gates as _core_gates

from . import modes, names

ENV_HOME = "COLABDESIGN_OPT_HOME"                          # the tree root (colabdesign/); else MODEL_OPT (configs/<gpu>.env); else the tree around this package
ENV_MODEL_OPT = "MODEL_OPT"
PACKAGE_ENV = (ENV_HOME, modes.ENV)                             # the package's own variables: never exported into an arm
_REPORT: Dict[str, object] = {"active": False, "reason": "enable() has not run in this process"}
_APPLIED = False


class ActivationError(RuntimeError):
    pass




# ----------------------------------------------------------------------------------------------------------------- paths and pins
def tree_home(environ=None) -> str:
    environ = os.environ if environ is None else environ
    for k in (ENV_HOME, ENV_MODEL_OPT):
        v = environ.get(k)
        if v:
            return os.path.abspath(v)
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))


def pyproject_path() -> str:
    """`opt/pyproject.toml` beside this package: its metadata and its `[tool.opt_core]` pin (the pin belongs to the package's bytes)."""
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "pyproject.toml"))


def core_gate() -> dict:
    """The core pin as the activation record's ``core`` field: the facts of THE pin gate (``colabdesign_opt/_core_gate.py``, the tree's
    kit_template copy, run at every entry before any ``opt_core`` import — by the time this runs it has passed; it is called again here only for
    its returned facts, one producer of the 'pinned X, installed Y' fact)."""
    import io
    from ._core_gate import CoreGateRefused, gate
    try:
        f = gate(os.path.dirname(os.path.abspath(__file__)), tag=names.TAG, stream=io.StringIO())
    except CoreGateRefused as e:                                     # reachable in-process only (a caller that skipped the entry): the entry's own words
        return {"ok": False, "reason": e.line.split("NOT ACTIVE: ", 1)[-1].strip(), "detail": {"reason": e.reason}}
    inst = f["installed"]
    return {"ok": True, "reason": None, "pinned": {k: f["pinned"][k] for k in ("path", "version")},
            "installed": {"version": inst["version"], "root": inst["root"]}}


def pins_path(environ=None) -> str:
    return os.path.join(tree_home(environ), "stock", "PINS.json")


def load_check_pins(environ=None):
    """`stock/check_pins.py` loaded as a module by path (standard library only; never through sys.path): the ONE home of the pin words this package
    reuses — the interpreter rule (python_pin) and the weights words (check_weights / weights_lines / sha256_cached)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("colabdesign_check_pins", os.path.join(tree_home(environ), "stock", "check_pins.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def pins(environ=None) -> dict:
    with open(pins_path(environ), "r", encoding="utf-8") as fh:
        return json.load(fh)


def stock_env_absent(p: dict, arm: str = "kit") -> List[str]:
    """The variable-name prefixes an arm's process must not carry: `must_be_absent_prefixes` (both arms: the kit's and every lever add-on's
    switches) plus, for the stock arm, `stock_arm_absent_prefixes` (jax's persistent-cache variables, which lever compilecache sets INTO the kit arm)."""
    se = p.get("stock_environment") or {}
    out = list(se.get("must_be_absent_prefixes") or [])
    if arm == "stock":
        out += [x for x in (se.get("stock_arm_absent_prefixes") or []) if x not in out]
    return out




# ----------------------------------------------------------------------------------------------------------------- installed stack
dist_version = _core_gates.dist_version                      # an installed distribution's version without importing it (the tree's one reader, opt_core.gates)


def installed_colabdesign() -> dict:
    """The installed colabdesign distribution without importing it: version and the git commit of its `direct_url.json` (a VCS install)."""
    out = {"version": dist_version("colabdesign"), "commit": None, "direct_url": None}
    try:
        dist = importlib.metadata.distribution("colabdesign")
    except importlib.metadata.PackageNotFoundError:
        return out
    try:
        raw = dist.read_text("direct_url.json")
    except Exception:
        raw = None
    if raw:
        try:
            du = json.loads(raw)
            out["direct_url"] = du.get("url")
            out["commit"] = ((du.get("vcs_info") or {}).get("commit_id"))
        except ValueError:
            pass
    return out


def upstream_versions() -> dict:
    cd = installed_colabdesign()
    return {"colabdesign": cd, "jax": dist_version("jax"), "jaxlib": dist_version("jaxlib"), "jax-cuda12-plugin": dist_version("jax-cuda12-plugin"),
            "dm-haiku": dist_version("dm-haiku"), "numpy": dist_version("numpy"), "python": ".".join(str(x) for x in sys.version_info[:3])}


def pin_refusal(p: dict) -> Optional[str]:
    """None when the installed colabdesign is the pinned commit (or, for a non-VCS install, the pinned version); else the named reason."""
    want = p["upstream"]["colabdesign"]
    cd = installed_colabdesign()
    if cd["version"] is None:
        return "colabdesign is not installed (stock/PINS.json upstream.colabdesign.install)"
    if cd["commit"]:
        if cd["commit"] != want["commit"]:
            return f"colabdesign is installed from commit {cd['commit'][:12]}, the pin is {want['commit'][:12]} (stock/PINS.json)"
        return None
    if cd["version"] != want["version"]:
        return f"colabdesign {cd['version']} is installed without a VCS record; the pin is {want['version']} @ {want['commit'][:12]}"
    return None


def stack_drift(p: dict) -> Dict[str, str]:
    """{distribution: "<installed>!=<pinned>"} for every asserted stack pin (stock/PINS.json `pins_asserted` + `python`) whose installed base
    version differs from the pin — NAMED on the ACTIVE line (`stack_drift=`), never a refusal and never a reason to drop a lever: the levers
    engage on the stack that is there; only the colabdesign commit (what "stock" means) is a gate (pin_refusal). {} on the pinned stack."""
    def base(v):
        return None if v is None else str(v).split("+", 1)[0]
    drift: Dict[str, str] = {}
    for name in p.get("pins_asserted") or []:
        if name == "colabdesign":
            continue
        want, have = (p.get("pins") or {}).get(name), dist_version(name)
        if want is not None and base(have) != base(want):
            drift[name] = f"{have or 'absent'}!={want}"
    d = load_check_pins().python_pin(p)                                  # the interpreter: stock/check_pins.py's one rule — any release other than the pin is named here; `run.sh install` refuses another series
    if d["status"] in ("not_pinned", "refused"):
        drift["python"] = f"{d['installed']}!={d['pinned']}"
    return drift


def nvidia_smi_probe() -> dict:
    """{name, memory_mib, cc} of GPU 0 as the tree's one probe reads nvidia-smi (opt_core.gates.nvidia_smi_probe), or {} without one."""
    g = _core_gates.nvidia_smi_probe(keys=("name", "memory_mib", "cc"))
    return {k: v for k, v in g.items() if v is not None} if g.get("name") else {}


def _cc_tuple(v) -> Optional[tuple]:
    """'8.0' -> (8, 0); None for an absent / unparsable value."""
    try:
        return tuple(int(x) for x in str(v).split(".")) if v not in (None, "") else None
    except ValueError:
        return None


def card_check(p: dict, gpu: dict, tolerance: float = 0.01) -> dict:
    """The card against PINS.json `gpu`: {"ok", "expected", "got", "reason"} (+ "note"). `ok` is False, with `reason`, only without a visible GPU:
    the levers that need the GPU backend then step aside by name in the arm (levers.NEEDS_GPU). Every visible card is `ok`; one
    other than the tested one (name; memory within `tolerance`) or below `gpu.cc_min` — the Pallas kernel's compute-capability floor — is
    not the tested card, said in a one-sentence `note` (the key is absent on the tested card): the levers engage on it and the ACTIVE line names it."""
    want = p.get("gpu") or {}
    got = gpu or {}
    if not got.get("name"):
        return {"ok": False, "expected": want, "got": got, "reason": "no GPU visible"}
    cc, cc_min = _cc_tuple(got.get("cc")), _cc_tuple(want.get("cc_min"))
    differs = []
    if want.get("name") and got["name"] != want["name"]:
        differs.append(f"card {got['name']!r} != {want['name']!r}")
    if want.get("memory_mib") and got.get("memory_mib") and abs(got["memory_mib"] - want["memory_mib"]) > tolerance * want["memory_mib"]:
        differs.append(f"memory {got['memory_mib']} MiB outside {tolerance:.0%} of {want['memory_mib']} MiB")
    if cc is not None and cc_min is not None and cc < cc_min:
        differs.append(f"compute capability {got['cc']} < {want['cc_min']}, the Pallas kernel's floor")
    rec = {"ok": True, "expected": want, "got": got, "reason": None}
    if differs:
        rec["note"] = "; ".join(differs) + " — not the tested card"                # named, never refused: the levers engage as on the tested card
    return rec


# ----------------------------------------------------------------------------------------------------------------- late activation
def model_instances() -> int:
    """`mk_af_model` instances alive in this process (0 when colabdesign.af.model is not imported)."""
    mod = sys.modules.get("colabdesign.af.model")
    cls = getattr(mod, "mk_af_model", None) if mod else None
    if cls is None:
        return 0
    return sum(1 for o in gc.get_objects() if isinstance(o, cls))


def kit_markers() -> dict:
    """The levers' markers on the upstream objects, when those modules are imported: {lever: bool} for every registered lever (each module's installed())."""
    from . import levers, registry
    return {l: levers.module_of(l).installed() for l in registry.ORDER}


def late_activation_refusal() -> Optional[str]:
    n = model_instances()
    if n:
        return (f"{n} mk_af_model instance(s) already exist: their executables were built by the _prep_model in place at prep_inputs "
                f"(colabdesign/af/prep.py:25-35, :268); the nosub replacement cannot reach them — enable before mk_afdesign_model()")
    mk = kit_markers()
    on = [k for k, v in mk.items() if v]
    if on and not _APPLIED:
        return f"lever(s) {','.join(on)} are already installed in this process by another caller (markers nosub.MARKER / pallas.MARKER)"
    return None


# ----------------------------------------------------------------------------------------------------------------- activation
def _report(res: modes.Resolved, route: str, settings: Optional[str]) -> dict:
    up = upstream_versions()
    return {"active": False, "mode": res.mode, "route": route, "tier": res.tier, "levers": list(res.levers), "numerics_class": res.numerics_class,
            "arm": res.route, "line": res.line, "settings": settings, "upstream": up,
            "package_version": dist_version("colabdesign_opt") or _pkg_version(), "gpu": {}, "core": None, "pid": os.getpid(),
            **({"base": res.base, "ablated": list(res.ablated)} if res.ablated else {}),
            **({"restored": list(res.restored)} if getattr(res, "restored", ()) else {})}   # the subtractive word only (modes.resolve): a table mode's report is unchanged


def _pkg_version() -> str:
    from . import __version__
    return __version__


def activate(mode: Optional[str], *, route: str = "api", dry_run: bool = False, strict: bool = False,
             settings: Optional[str] = None, environ=None, gpu_probe=None, trigger: Optional[str] = None) -> dict:
    """Resolve, gate and (unless dry_run) install the levers of `mode`; returns the activation report. Idempotent for the in-process routes: a
    repeated call returns the first report; a different mode in the same process is refused."""
    global _APPLIED
    environ = os.environ if environ is None else environ
    gpu_probe = nvidia_smi_probe if gpu_probe is None else gpu_probe
    try:
        res = modes.resolve(mode)
    except ValueError as e:
        rep = {"active": False, "mode": mode, "route": route, "reason": str(e), "tier": None, "levers": None, "line": None, "upstream": upstream_versions(), "gpu": {}}
        if strict:
            raise ActivationError(str(e))
        return rep
    if not dry_run and _REPORT.get("active"):
        if _REPORT.get("mode") == res.mode:
            return dict(_REPORT)
        why = f"{modes.describe(res)} requested but this process already activated mode={_REPORT.get('mode')}"
        if strict:
            raise ActivationError(why)
        return {**_REPORT, "active": False, "reason": why}
    rep = _report(res, route, settings)
    if trigger:
        rep["trigger"] = trigger
    try:
        p = pins(environ)
        rep["gpu"] = gpu_probe() or {}
        rep["card"] = card_check(p, rep["gpu"])                                  # recorded; a card other than the tested one is a note, not a gate (the kit runs on other cards)
        rep["target_gpu"] = environ.get("MODEL_OPT_TARGET_GPU")
        why = pin_refusal(p)
        if why:
            raise ActivationError(why)                                         # another colabdesign commit changes what "stock" means: the one environment refusal
        rep["stack_drift"] = stack_drift(p)                                    # a stack distribution off its pin: NAMED on the ACTIVE line, never a gate
        if res.route == "kit":
            rep["core"] = core_gate()                                          # the core pin's facts (THE gate ran at the entry; called again here only for its record)
            if not rep["core"]["ok"]:
                raise ActivationError(rep["core"]["reason"])
            from . import levers as _levers                                     # no card visible: the levers that need the GPU backend (levers.NEEDS_GPU — the Pallas kernels, the
            rep["skipped"] = _levers.foretell(res.levers, bool(rep["gpu"].get("name")))   # GPU-keyed compile cache) will step aside by name in the arm; foretold here, where nothing is installed
            rep["levers"] = [l for l in res.levers if l not in rep["skipped"]]        # (the dry-run routes' only fact); with a card visible: {} and the arm's LEVER lines are the record
            # any visible card runs every mode: one other than the tested card, or below the kernel's compute-capability floor, is only
            # named (ACTIVE gpu=, the card's note), the levers engage; a kernel that cannot lower there steps aside by name at install (levers.py)
            if not dry_run:
                why = late_activation_refusal()
                if why:
                    raise ActivationError(why)
                rep.update(_apply(res, route, environ))
                _APPLIED = True
        rep["active"] = True
    except ActivationError as e:
        rep["active"] = False
        rep["reason"] = str(e)
        if strict:
            raise
    if not dry_run:
        _REPORT.clear(); _REPORT.update(rep)
    return rep


def _apply(res: modes.Resolved, route: str, environ) -> dict:
    """Install the mode's levers through the ONE installer (levers.install); returns {"levers" (what is on), "levers_installed", "skipped"
    ({lever: reason} — the levers that stepped aside by name: their skipped LEVER line is already printed), "pallas"} from its record. Only
    configuration refuses (levers.LeverError → ActivationError); a lever of the set neither installed nor stepped aside is an installer defect,
    named."""
    from . import levers
    try:
        info = levers.install(res.mode)
    except levers.LeverError as e:
        raise ActivationError(f"mode {res.mode}: {e}") from e
    installed, skipped = list(info["levers_installed"]), dict(info.get("skipped") or {})
    missing = [l for l in res.levers if l not in installed and l not in skipped]
    if missing:
        raise ActivationError(f"mode {res.mode}: lever(s) {','.join(missing)} neither installed nor stepped aside at activation (installed={','.join(installed) or 'none'} skipped={','.join(skipped) or 'none'}) — an installer defect")
    return {"levers": installed, "levers_installed": installed, "skipped": skipped, "pallas": info.get("pallas")}


def status() -> dict:
    return dict(_REPORT)


def reset_for_tests() -> None:
    """Forget this process's activation (tests only): the report, the installer's state, nosub's build records, the Pallas Ledger."""
    global _APPLIED
    _APPLIED = False
    _REPORT.clear(); _REPORT.update({"active": False, "reason": "enable() has not run in this process"})
    from . import kernels, levers, nosub, pallas, compilecache_jax, hoist_prev, launchpad_parcompile, lowercache
    from .kernels import triatt_lever, trimul_fused, layers_opm, layers_ln, proj_attn, layers_transition
    levers.reset_for_tests(); kernels.reset_for_tests(); nosub._BUILDS.clear(); nosub._LINES.clear(); nosub.POLICY.update({'grad': nosub.STOCK, 'fn': nosub.STOCK}); nosub.GATE.reset() if hasattr(nosub.GATE, "reset") else None; pallas.LEDGER = None
    triatt_lever.uninstall(); triatt_lever.LEDGER = None
    layers_opm.uninstall(); layers_opm._CENSUS.clear(); layers_opm._GATED.clear() if hasattr(layers_opm, '_GATED') else None   # the OuterProductMean re-association: stock's class back, census dropped
    layers_ln.uninstall(); [c.clear() for c in layers_ln._CENSUS.values()]
    layers_transition.uninstall(); layers_transition.reset_census()                          # the fused Transition MLP: stock's class back, census dropped
    proj_attn.uninstall(); proj_attn.LEDGER = None                                            # the fused projections: the class below back, census dropped
    trimul_fused.uninstall(); trimul_fused.LEDGER = None                                     # the triangle-multiplication kernels: stock's class back, census dropped
    compilecache_jax.reset_for_tests(); lowercache.reset_for_tests(); launchpad_parcompile.uninstall(); launchpad_parcompile.reset_for_tests(); hoist_prev.uninstall(); hoist_prev._CENSUS.update({k: 0 for k in hoist_prev._CENSUS}); hoist_prev._LINES.clear()   # the exact levers: jax's cache pointed at no directory, ColabDesign's own methods back, censuses zeroed
