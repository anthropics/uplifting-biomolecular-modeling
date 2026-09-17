"""warm — the checkpoint load: upstream's engine constructed and initialised once (imports, the checkpoint read, the model on the
GPU) in a subprocess of the chosen interpreter, with the mode's delta exported first, so that the first real `design` does not pay
for it blind and the activation / application lines are seen once. The kit's module is imported while the model is built
(`engine.initialize()`), so the `APPLIED` line is read after it. No design is run and nothing is written but the log.

``--mode off`` is the stock arm and is proven stock the way ``design --mode off`` is: the child runs in the pristine interpreter
(RFDIFFUSION3_STOCK_PYTHON, isolated mode ``-I``) with the package switch and every lever name stripped from its environment (stack.strip_env), and before
it imports anything of upstream it runs the stock caller's own proof (stock_design.env_proof: no forbidden name, the finder unarmed,
the rfd3 tree pristine; the upstream pin reported as `PINS` note lines, never refused) — a failed proof is ``NOT STOCK``, exit 3, nothing built. Its PASS needs the ``STOCK`` line and an
``APPLIED none`` line (the kit's module never imported); ``exact`` PASS needs ``APPLIED … RFD3_HOIST=True``. A kit-mode child that
built the engine without that evidence — the module read the switch off, or was never imported — is a PARTIAL activation (``partial``
names the lever, ``partial_reason`` the child's own line): FAIL with ``error=Partial`` and the verb exits 3, unless ``allow_partial``
(``warm --allow-partial`` / RFDIFFUSION3_OPT_ALLOW_PARTIAL=1) records the allowance and the result is the run's own (PASS).

Result: ``{"status": PASS|FAIL, "mode", "activation", "stock_proof", "applied", "exit_code", "wall_s", "init_s", "log", "partial",
"partial_reason", "allow_partial", "error"?, "reason"?}``.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from typing import Optional

from . import modes, stack
from .report import PREFIX, TAG
from .stock_design import PREFIX as STOCK_PREFIX

ACTIVATION_RX = f"{PREFIX} ACTIVE "
APPLIED_RX = f"{PREFIX} APPLIED "
APPLIED_NONE_RX = f"{PREFIX} APPLIED none"
NOT_ACTIVE_RX = f"{PREFIX} NOT ACTIVE"
STOCK_RX = f"{STOCK_PREFIX} STOCK "
NOT_STOCK_RX = f"{STOCK_PREFIX} NOT STOCK"
INIT_RX = f"[{TAG} warm] init "                                       # the child prints it from the same TAG (CHILD below)

CHILD = r'''
import sys, time, os
mode = sys.argv[1]; ckpt = sys.argv[2]
import rfdiffusion3_opt
from rfdiffusion3_opt._autoload import EXIT_NOT_ACTIVE, TAG   # import-free: the child's first opt_core import is inside enable(), after its core pin gate
try:
    rep = rfdiffusion3_opt.enable(mode, strict=True, hook=False)   # the warm verb applies the exit rule from this child's APPLIED line
except rfdiffusion3_opt.ActivationError:
    sys.exit(EXIT_NOT_ACTIVE)            # the NOT ACTIVE line is in the log; nothing of upstream is imported
if mode == "off":                        # the stock arm: the stock caller's own proof, before any upstream import
    from rfdiffusion3_opt import stock_design
    proof = stock_design.env_proof(core_allowed=stock_design.GATE_CORE_MODULES)   # this child gated `off` through the package before the proof
    for note in stock_design.pins_note_lines(proof):
        stock_design.emit(note)              # the upstream pin: reported, never a refusal
    if not proof["ok"]:
        stock_design.emit(stock_design.not_stock_line(proof))
        sys.exit(stock_design.EXIT_NOT_STOCK)
    stock_design.emit(stock_design.stock_line(proof))
t0 = time.time()
from rfd3.engine import RFD3InferenceEngine, RFD3InferenceConfig
from rfdiffusion3_opt import stack
cfg = RFD3InferenceConfig(ckpt_path=ckpt, diffusion_batch_size=1, skip_existing=False, seed=None, verbose=False, compile_model=False)
engine = RFD3InferenceEngine(**cfg)
engine.initialize()                      # the model is built here: rfd3.model.hoist is imported (and reads the switch) during initialize()
try:
    import torch
    torch.cuda.synchronize()
except Exception:
    pass
stack.report_applied()                   # after initialize(): what the kit's module read (off: `APPLIED none`, the module never imported)
print("[%s warm] init %.1f s ckpt=%s" % (TAG, time.time() - t0, engine.ckpt_path), flush=True)
'''


def child_env(mode: str) -> dict:
    """The child's environment: the stock arm sees no package switch and no lever name (as `design --mode off`); the kit arm inherits
    the caller's environment minus the package switch (the child activates explicitly, one route, one activation line) and enable()
    gates what it finds there."""
    env = stack.strip_env() if mode == "off" else dict(os.environ)
    env.pop("RFDIFFUSION3_OPT", None)
    return env


def run(mode: str, ckpt: str, *, python: Optional[str] = None, allow_partial: bool = False) -> dict:
    if not ckpt or not os.path.isfile(ckpt):
        return {"status": "FAIL", "mode": mode, "error": "CheckpointMissing", "reason": f"checkpoint not found: {ckpt!r} (set RFD3_CKPT or pass --ckpt)", "exit_code": None}
    python = python or sys.executable
    log_path = os.path.join(tempfile.mkdtemp(prefix="rfd3_warm_"), "warm.log")
    env = child_env(mode)
    cmd = [python] + (["-I"] if mode == "off" else []) + ["-c", CHILD, mode, os.path.abspath(ckpt)]   # off: isolated, as the stock caller runs
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        proc.wait()                                     # no deadline: a hang is silence plus an idle GPU, judged by the caller, never a kill here
    wall = time.time() - t0
    lines = open(log_path, encoding="utf-8", errors="replace").read().splitlines()
    activation = next((l for l in lines if l.startswith(ACTIVATION_RX) or l.startswith(NOT_ACTIVE_RX)), None)
    stock_proof = next((l for l in lines if l.startswith(STOCK_RX) or l.startswith(NOT_STOCK_RX)), None)
    applied = next((l for l in lines if l.startswith(APPLIED_RX)), None)
    init_line = next((l for l in lines if l.startswith(INIT_RX)), None)
    init_s = float(init_line.split()[3]) if init_line else None
    ran = proc.returncode == 0 and init_line is not None
    partial, partial_reason = [], None
    if mode == "off":
        ok = ran and stock_proof is not None and stock_proof.startswith(STOCK_RX) and applied is not None and applied.startswith(APPLIED_NONE_RX)
    else:
        ok = ran and applied is not None and "RFD3_HOIST=True" in applied
        if ran and not ok:                              # the engine was built, the lever has no evidence of application: partial
            partial = [modes.KIT_SWITCH]                    # the evidence this verb reads is the hoist module's line; the other levers' lines are the design verb's
            partial_reason = (f"{modes.KIT_HOIST_MODULE} was never imported by the engine build (the child's line: {applied})" if applied is None or applied.startswith(APPLIED_NONE_RX)
                              else f"{modes.KIT_HOIST_MODULE} read {modes.KIT_SWITCH} off at its import (the child's line: {applied})")
    res = {"status": "PASS" if ok or (ran and partial and allow_partial) else "FAIL", "mode": mode, "activation": activation, "stock_proof": stock_proof,
           "applied": applied, "exit_code": proc.returncode, "wall_s": round(wall, 1), "init_s": init_s, "log": log_path,
           "command": cmd[:2] + ["<child>", mode, ckpt], "python": python, "partial": partial, "partial_reason": partial_reason,
           "allow_partial": bool(allow_partial)}
    if proc.returncode == 3 and activation and activation.startswith(NOT_ACTIVE_RX) and mode != "off":
        res["error"], res["reason"] = "NotActive", activation[len(PREFIX) + 1:]
    elif proc.returncode == 3 and stock_proof and stock_proof.startswith(NOT_STOCK_RX):
        res["error"], res["reason"] = "NotStock", stock_proof[len(STOCK_PREFIX) + 1:]
    elif proc.returncode != 0:
        res["reason"] = f"the warm-up process exited {proc.returncode} (log: {log_path})"
    elif init_line is None:
        res["reason"] = "no init line: the engine did not initialise"
    elif not ok and mode == "off":
        res["reason"] = ("no STOCK proof line in the log" if not (stock_proof or "").startswith(STOCK_RX)
                         else "the kit module was imported in the stock arm (no `APPLIED none` line): the interpreter is not stock")
    elif partial and not allow_partial:
        res["error"], res["reason"] = "Partial", partial_reason
    elif partial:
        res["reason"] = f"partial activation allowed (--allow-partial, recorded): {partial_reason}"
    return res


def summary_line(res: dict) -> str:
    return (f"{PREFIX} WARM {res.get('status')} mode={res.get('mode')} init={res.get('init_s')}s rc={res.get('exit_code')} wall={res.get('wall_s')}s log={res.get('log')}"
            + (f" error={res['error']}" if res.get("error") else "") + (f" reason={res['reason']}" if res.get("reason") else ""))
