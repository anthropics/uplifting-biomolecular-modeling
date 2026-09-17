"""Activation: gates -> patches in the mode row's order -> the ACTIVE line; per-lever LEVER lines + the EXIT tally at interpreter exit.
Family grammar (opt_core.report): ``[atlasfold-opt] ACTIVE mode=<m> n_gpu=<P> gpu=<name(smNN)> levers=<a+b> skipped=<none|lever:reason,...> [ablated=<names>]
stock=atlasfold@<sha7> cueq=<ver> torch=<ver> core=<ver> kit=<ver> det=<0|1> fill=<na|0|1> trigger=<module|cli|api>``; a lever selected and not applied
is named in ``skipped=`` and the process exits 3 unless --allow-partial / AFO_ALLOW_PARTIAL=1 (``PARTIAL allowed: ...``)."""
import atexit
import os
import sys
import time
from typing import Dict, List, Optional

from . import TAG, ENV, STOCK, __version__, ActivationError, _core
from .modes import MODES, MODE_NAMES, levers_of

_REPORT: Optional[dict] = None
_INSTALLED: List = []
_T0 = time.time()


def _emit(line: str) -> str:
    sys.stderr.write(line + "\n"); sys.stderr.flush(); return line


def status() -> dict:
    return dict(_REPORT) if _REPORT else {"active": False, "reason": "enable() has not run"}


def _gpu_probe() -> dict:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"name": "none", "cc": None, "sm": None, "cuda": False}
        p = torch.cuda.get_device_properties(0)
        return {"name": p.name, "cc": f"{p.major}.{p.minor}", "sm": f"sm{p.major}{p.minor}", "memory_mib": p.total_memory // 2**20, "cuda": True,
                "torch": torch.__version__, "n_visible": torch.cuda.device_count()}
    except Exception as e:  # noqa: BLE001
        return {"name": "none", "cc": None, "sm": None, "cuda": False, "error": type(e).__name__}


def _stock_facts() -> dict:
    facts = {"importable": False}
    try:
        import importlib.metadata as md
        try:
            facts["dist_version"] = md.version("atlasfold")
        except md.PackageNotFoundError:
            facts["dist_version"] = None
        import importlib.util as u
        spec = u.find_spec("atlasfold")
        facts["importable"] = spec is not None
        facts["location"] = os.path.dirname(spec.origin) if spec and spec.origin else None
        cue = u.find_spec("cuequivariance_torch")
        facts["cuequivariance_torch"] = md.version("cuequivariance-torch") if cue else None
    except Exception as e:  # noqa: BLE001
        facts["error"] = repr(e)
    return facts


def _fill_word(det_facts: dict) -> str:
    """The det recipe's fill fact for the ACTIVE line: `na` under det 0 (nothing applied), else 0|1 = torch's fill_uninitialized_memory as the recipe left it."""
    if not det_facts.get("det"):
        return "na"
    v = det_facts.get("fill_uninitialized_memory")
    return str(int(v)) if isinstance(v, bool) else str(v)


WARM_LIBRARIES = ("torch", "cuequivariance_ops_torch", "cuequivariance_torch")   # torch FIRST: the cuEquivariance ops wheel resolves the CUDA runtime libraries torch loads (a cu13 ops wheel imported before torch fails to load libnvrtc and the ops stay unavailable for the process); torch is already imported here, named for the order's sake


def warm_imports(prefix: str) -> Dict:
    """opt_core.warm_imports() once at activation (idempotent, never raises; `OPT_CORE_NO_WARM_IMPORTS=1` skips it) and its
    NOTE line: `NOTE warm_imports=<library>:<seconds|present|absent|skipped:env|error:…>,… origin=activate`.  `--mode off` never reaches here (the
    stock route imports its libraries itself)."""
    try:
        import opt_core
        rep = dict(opt_core.warm_imports(libraries=WARM_LIBRARIES, origin="atlasfold_opt.activate"))
        words = ",".join(f"{k}:{v}" for k, v in rep.items()) or "none"
    except Exception as e:  # noqa: BLE001 — a core without warm_imports: named, never fatal (the libraries then load at their first use)
        rep, words = {}, f"unavailable:{type(e).__name__}"
    _emit(f"{prefix} NOTE warm_imports={words} origin=activate")
    return rep


def activate(mode: str, *, strict: bool = False, trigger: str = None, det: int = None, n_gpu: int = 1, allow_partial: bool = None, inputs: dict = None) -> dict:
    global _REPORT
    if _REPORT is not None:                                   # idempotent: the first activation stands
        return dict(_REPORT)
    core = _core.gate()
    from opt_core import report as R
    from opt_core.modes import levers_label
    prefix = R.prefix(TAG)
    if mode not in MODES:
        _emit(R.not_active_line(TAG, f"unknown_mode:{mode} (known: {','.join(MODE_NAMES)})", None) if hasattr(R, "not_active_line")
              else f"{prefix} NOT ACTIVE: reason=unknown_mode:{mode}")
        raise SystemExit(3)
    allow_partial = bool(allow_partial) if allow_partial is not None else (os.environ.get("AFO_ALLOW_PARTIAL") == "1")
    det = int(det if det is not None else os.environ.get("AFO_DET", "0") or 0)
    from . import ablation as A                               # MODEL_OPT_LEVERS_OFF: validated by name against the mode's row BEFORE anything is imported or patched
    ablated = A.requested()
    try:
        row = A.row_without(mode, ablated) if ablated else levers_of(mode)
        why = A.reasons(mode, ablated)                            # requested -> levers_off; their dependency closure -> requires:<lever>(levers_off) (off BY NAME with them, never a partial activation)
    except A.AblationError as e:
        _emit(f"{prefix} NOT ACTIVE: reason=levers_off_refused: {e}")
        raise SystemExit(3)
    ablated = list(why)                                       # the requested names, then the levers that require them (row order)
    rep: Dict = {"active": False, "mode": mode, "kit": "atlasfold", "package_version": __version__, "core_version": core["installed"]["version"],
                 "stock": STOCK, "trigger": trigger or "api", "n_gpu": int(n_gpu), "det": det, "levers_selected": list(row), "levers_ablated": list(ablated), "ablation_reasons": dict(why),
                 "levers_applied": [], "levers_skipped": {}, "t_activate": time.time()}
    if mode == "off":
        rep.update(active=False, reason="mode_off: stock route (pred --mode off runs the stock CLI in a clean subprocess)")
        _REPORT = rep
        return dict(rep)
    # ---- gates
    stock = _stock_facts(); rep["stock_facts"] = stock
    if not stock.get("importable"):
        _emit(f"{prefix} NOT ACTIVE: reason=stock_missing: atlasfold is not importable (pip install -e stock/src or PYTHONPATH=stock/src/src)")
        raise SystemExit(3)
    gpu = _gpu_probe(); rep["gpu"] = gpu
    if not gpu.get("cuda"):
        _emit(f"{prefix} NOT ACTIVE: reason=no_cuda_device (the kit's levers are CUDA kernels; use --mode off on CPU)")
        raise SystemExit(3)
    if int(n_gpu) > 1:
        _emit(f"{prefix} NOT ACTIVE: reason=rowpair_tp_not_in_{__version__}: --n_gpu {n_gpu} (a row-sharded multi-GPU mode returns when it carries levers of its own; run --n_gpu 1)")
        raise SystemExit(3)
    try:                                                      # late activation: refuse once a model instance exists (patched class methods would not reach it consistently)
        from opt_core import instances as I
        chk = I.instance_check("atlasfold.model.model", "AtlasFold")
        if chk.get("n", 0) > 0:
            _emit(f"{prefix} NOT ACTIVE: reason=late_activation: {chk['n']} AtlasFold instance(s) already built (enable() before load_model)")
            raise SystemExit(3)
        I.register_instance_counter("atlasfold.model.model", "AtlasFold")
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        rep["instance_counter"] = f"unavailable:{type(e).__name__}"
    from . import det as D
    rep["det_facts"] = D.apply(det, TAG)
    rep["warm_imports"] = warm_imports(prefix)                # the stack's heavy libraries (cuEquivariance ops) imported ONCE here, before the model is built — not lazily inside item 1's first pair-stack / confidence call
    # ---- levers, in row order
    from .hooks import installers as _installers
    installers = _installers()
    ctx: Dict = {"mode": mode, "det": det, "inputs": dict(inputs or {})}   # inputs: the run's start-up facts from the stock argv (inputs_hint: records / buckets) for process-level lever policies
    rep["inputs"] = dict(inputs or {})
    for lever in row:                                         # the mode's row less MODEL_OPT_LEVERS_OFF (ablated levers are never imported or patched)
        if lever == "lever_report":
            rep["levers_applied"].append(lever); continue
        fn = installers.get(lever)
        if fn is None:
            rep["levers_skipped"][lever] = "no_installer"; continue
        try:
            ins = fn(mode, TAG, ctx)
        except Exception as e:  # noqa: BLE001 — a lever that raises is a NAMED skip, never a traceback into the engine
            rep["levers_skipped"][lever] = f"raised:{type(e).__name__}:{str(e)[:120]}"
            continue
        if ins.applied:
            rep["levers_applied"].append(lever); _INSTALLED.append(ins)
        else:
            rep["levers_skipped"][lever] = ins.reason or "not_applied"
    rep["ctx"] = {k: v for k, v in ctx.items() if isinstance(v, (int, float, str))}
    # ---- verdict on partial activation (family rule)
    partial = bool(rep["levers_skipped"])
    rep["partial"] = partial; rep["allow_partial"] = allow_partial
    label = levers_label(rep["levers_applied"]) if rep["levers_applied"] else "none"
    skipped = ",".join(f"{k}:{v}" for k, v in rep["levers_skipped"].items()) or "none"
    abl = ((A.TOKEN, A.token(ablated)),) if ablated else ()   # `ablated=<names>` right after skipped= — absent when nothing is ablated (the line is then the mode's, byte for byte)
    head = R.kv(("mode", mode), ("n_gpu", int(n_gpu)), ("gpu", R.gpu_label(gpu)), ("levers", label), ("skipped", skipped), *abl,
                ("stock", f"atlasfold@{STOCK['commit'][:7]}"), ("cueq", stock.get("cuequivariance_torch") or "none"), ("torch", gpu.get("torch", "?")),
                ("core", core["installed"]["version"]), ("kit", __version__), ("det", det), ("fill", _fill_word(rep["det_facts"])), ("trigger", trigger or "api"))
    if partial and not allow_partial:
        _emit(f"{prefix} PARTIAL refused: {skipped} (exit 3; --allow-partial / AFO_ALLOW_PARTIAL=1 records and proceeds)")
        _emit(f"{prefix} NOT ACTIVE: reason=partial_activation {head}")
        raise SystemExit(3)
    if partial:
        _emit(f"{prefix} PARTIAL allowed: {skipped} (--allow-partial, recorded)")
    rep["active"] = True
    _emit(f"{prefix} ACTIVE {head}")
    _REPORT = rep
    _register_exit_lines()
    return dict(rep)


def _register_exit_lines() -> None:
    from opt_core import report as R
    for ins in _INSTALLED:
        for i, line_fn in enumerate(ins.lines):
            R.register_exit_tally(f"{TAG}/{ins.lever}/{i}", lambda f=line_fn: f())
    from . import ablation as A                               # one LEVER … state=off reason=levers_off line per ablated lever, beside the kept levers' lines
    for lever, text in zip(_REPORT.get("levers_ablated") or [], A.lever_lines(TAG, _REPORT.get("levers_ablated") or [], _REPORT.get("ablation_reasons"))):
        R.register_exit_tally(f"{TAG}/{lever}/levers_off", lambda t=text: t)

    def exit_line(*_a, **_k):
        gates = []
        for ins in _INSTALLED:
            for g in ins.gates:
                try:
                    gg = g(); gates.append((ins.lever, bool(getattr(gg, "ok", True)), getattr(gg, "reason", None)))
                except Exception as e:  # noqa: BLE001
                    gates.append((ins.lever, False, f"gate_raised:{type(e).__name__}"))
        refused = [f"{n}:{(r or '')[:60]}" for n, ok, r in gates if not ok]
        fields = R.kv(("mode", _REPORT.get("mode")), ("active", int(bool(_REPORT.get("active")))), ("levers", "+".join(_REPORT.get("levers_applied", [])) or "none"),
                      ("gates_refused", ",".join(refused) or "none"), ("wall_s", round(time.time() - _T0, 1)))
        return R.exit_tally_line(TAG, [fields], os.getpid(), "atexit") if hasattr(R, "exit_tally_line") else f"{R.prefix(TAG)} EXIT pid={os.getpid()} source=atexit {fields}"
    R.register_exit_tally(TAG, exit_line)


def gates_verdict() -> dict:
    """After the run: every lever's gate (fail-closed census). {ok, refused:[(lever, reason)]} — cli.cmd_pred folds it into the exit code."""
    out = {"ok": True, "refused": []}
    for ins in _INSTALLED:
        for g in ins.gates:
            try:
                gg = g()
                if not getattr(gg, "ok", True):
                    out["ok"] = False; out["refused"].append((ins.lever, getattr(gg, "reason", None)))
            except Exception as e:  # noqa: BLE001
                out["ok"] = False; out["refused"].append((ins.lever, f"gate_raised:{type(e).__name__}"))
    return out
