"""warm — one design on the example spec that ships in the kit (``opt/forward/fast_inference/tests/specs/pdl1_ref.yaml`` + its PDB;
no network) through `design` in the requested mode, so that the Triton / cuEquivariance JIT compilation lands in the caches the configs
name (TRITON_CACHE_DIR / TORCH_EXTENSIONS_DIR). One design, the two GPU steps ``--steps design inverse_folding``, seed ``SEED``; the
outputs go to a temporary directory that is removed after a PASS. ``off`` warms the same Triton caches for the stock arm (upstream
JIT-compiles the same kernels at its first call)."""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from typing import Optional

from . import design, modes, stack

SPEC_REL = os.path.join("tests", "specs", "pdl1_ref.yaml")
SPEC_DIR_REL = os.path.join("tests", "specs")
N_DESIGNS = 1
SEED = 0
CONFIGURE_ARGS = ("--num_designs", str(N_DESIGNS), "--steps", "design", "inverse_folding")   # upstream's own configure flags: one design, the two GPU steps


def spec_dir() -> str:
    return os.path.join(stack.kit_dir(modes.KIT_PARTNER), SPEC_DIR_REL)


def run(mode: Optional[str], out_dir: Optional[str] = None, echo: bool = True, gpu: Optional[dict] = None) -> dict:
    """One warm design; ``exit`` is design's exit for the run (cli.exit_for: 0 ok, 1 failed, 3 not active — a child refused its activation or a
    lever of the mode could not run: partial), ``status`` names it (PASS / FAIL / NOT ACTIVE); ``opt_manifest.json`` is written in the run dir
    with the partial record."""
    from . import cli as _cli, manifest as _manifest
    mode = modes.check_mode(mode)
    res = modes.resolve(mode, stack.opt_home())
    cache = os.environ.get(stack.ENV_CACHE)
    seed = SEED
    src = spec_dir()
    if not os.path.isfile(os.path.join(src, "pdl1_ref.yaml")):
        return {"status": "FAIL", "rc": 1, "exit": _cli.EXIT_FAIL, "reason": f"vendored spec missing: {os.path.join(src, 'pdl1_ref.yaml')}"}
    tmp = out_dir or tempfile.mkdtemp(prefix="boltzgen_opt_warm_")
    os.makedirs(tmp, exist_ok=True)
    work = os.path.join(tmp, "inputs")
    if os.path.isdir(work):
        shutil.rmtree(work)
    shutil.copytree(src, work)                                             # the spec names its PDB by a relative path, so the whole directory is copied
    run_dir = os.path.join(tmp, "warm")
    t0 = time.time()
    out = {"mode": mode, "out_dir": tmp, "run_dir": run_dir, "spec": os.path.join(work, "pdl1_ref.yaml")}
    try:
        conf = design.configure(out["spec"], run_dir, list(CONFIGURE_ARGS), cache)
        out["configure"] = conf
        if res.active:
            r = design.run_kit(res, run_dir, seed, echo=echo)
        else:
            r = design.run_stock(run_dir, seed, echo=echo)
        r["designs"] = design.account_designs(run_dir, r)               # the DESIGNS line, as `design` prints it
        out["run"] = {k: v for k, v in r.items() if k not in ("inproc_times", "designs")}
        out["designs"] = r["designs"]
        out["rc"] = r["rc"]
        rep = design.process_report(res, r, gpu)
        out["exit"] = _cli.exit_for(r["rc"], rep)
        out["status"] = {_cli.EXIT_OK: "PASS", _cli.EXIT_NOT_ACTIVE: "NOT ACTIVE"}.get(out["exit"], "FAIL")
        out["kernels"] = r.get("kernels")
        out["levers_applied"], out["levers_fallback"], out["fallback_reasons"] = r.get("levers_applied"), r.get("levers_fallback"), r.get("fallback_reasons")
        out["partial"] = stack.partial_levers(rep)
        out["manifest"] = _manifest.write(run_dir, rep, command="warm", exit_code=out["exit"], designs=r["designs"], extra={"seed": seed, "configure": conf, "warm": {"spec": out["spec"], "n_designs": N_DESIGNS}})
    except design.DesignError as e:
        out.update(status="FAIL", rc=1, exit=_cli.EXIT_FAIL, reason=str(e))
    out["wall_s"] = round(time.time() - t0, 1)
    if out.get("status") == "PASS" and out_dir is None:
        shutil.rmtree(tmp, ignore_errors=True)
        out["removed"] = True
    return out


def summary_line(res: dict) -> str:
    return (f"warm mode={res.get('mode')}: {res.get('status')} rc={res.get('rc')} wall={res.get('wall_s')}s levers={','.join(res.get('levers_applied') or []) or 'none'}"
            + (f" partial={','.join(res['partial'])}" if res.get("partial") else "")
            + (f" — {res['reason']}" if res.get("reason") else ""))
