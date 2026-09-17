"""The launcher — one design in a clean subprocess per arm, the environment built here, the design script proving it there.

The arm's environment = the parent's minus every variable with a prefix in stock/PINS.json ``stock_environment.must_be_absent_prefixes``
(``EF2_``, ``EF2INV_``, ``ESMFOLD2_``, the CUDA determinism knobs; the list is the pin's, not this file's), minus the package's own
variables, plus — on a kit arm only — ``EF2_FAST_KIT=<the mode's value>``, the ablation word ``MODEL_OPT_LEVERS_OFF`` when the operator set it
(the kit arm's second allowed variable, resolved by name before the launch: cli.launch_context) and the allocator setting ``KIT_ALLOCATOR``
(unless the operator set it), plus — on a det arm, every mode — the Inductor pin ``DET_PINS``. The data-path variables (``HF_HOME``, ``HF_HUB_OFFLINE``,
``ESMCFOLD_CCD_PATH``, ``TRITON_CACHE_DIR``, ``TORCHINDUCTOR_CACHE_DIR``) pass through. The interpreter runs ``-I`` (isolated: no
``PYTHON*`` variable, no user site) so the path is the venv's plus what the script adds (the kit's ``k/`` by argument on a
kit arm). ``run(timeout_s=…)`` — a caller's watchdog, none from the CLI — kills the process group (exit 124). The launcher then reads back the outputs and writes
``opt_manifest.json`` (manifest.py).
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional

from .modes import JIT_CACHE_VAR, JIT_DIR_VARS, JIT_LOCAL_ROOT_VAR, JIT_SHARED_ROOT_VAR, KIT_SWITCH, Mode, jit_cache_dirs, jit_cache_form, jit_cache_state, kit_paths

PACKAGE_VARS = ("EF2INV_OPT",)
LEVERS_OFF_VAR = "MODEL_OPT_LEVERS_OFF"                     # fastkit.LEVERS_OFF_VAR (the ablation word): carried into a kit arm as given, stripped from the stock arm like every kit variable
DET_PINS = {"TORCHINDUCTOR_SHAPE_PADDING": "0", "TORCHINDUCTOR_DETERMINISTIC": "1"}   # every det arm (--det >= 1, off included): Inductor's matmul shape padding OFF and
                                                         # its deterministic mode ON. The pad_mm choice is otherwise a per-process autotuning decision (and a forced pad is
                                                         # bypassed once dynamo marks the token dimension dynamic), so two det processes could land on different kernels for
                                                         # the same graph; unpadded + deterministic is one structural state every process reaches
KIT_ALLOCATOR = ("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # kit arms only, when the operator has not set it: the caching allocator maps its segments
                                                         # expandably, which removes the fragmentation between the loop's eager activations and the CUDA-graph pools
                                                         # (reserved memory, not arithmetic: the levers' numerics are unchanged; the stock arm's allocator is untouched)
PASS_THROUGH_DATA = ("HF_HOME", "HF_HUB_OFFLINE", "ESMCFOLD_CCD_PATH", "EF2_HF_HOME", "EF2_CCD_PATH", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR")


def arm_env(mode: Mode, prefixes: List[str], base: Optional[Dict[str, str]] = None, det: int = 0) -> Dict[str, str]:
    """The arm's environment (module docstring). ``det >= 1`` = a det arm: its JIT cache is the per-box form (fresh box-local
    ``TRITON_CACHE_DIR`` / ``TORCHINDUCTOR_CACHE_DIR``; modes.jit_cache_dirs) unless the deployment switch ``EF2INV_JIT_CACHE`` in ``base``
    names a form explicitly — then that form is honoured and ``det_cache`` reports it. At ``det 0`` the deployment's dirs pass through."""
    base = dict(os.environ if base is None else base)
    env = {}
    for k, v in base.items():
        if k in PACKAGE_VARS:
            continue
        if k in PASS_THROUGH_DATA:
            env[k] = v
            continue
        if any((k.startswith(p) if p.endswith("_") else k == p) for p in prefixes):
            continue
        env[k] = v
    env.pop(LEVERS_OFF_VAR, None)               # the ablation word is a kit variable whatever its prefix: absent from the stock arm, carried into a kit arm below
    if mode.kit_switch:
        env[KIT_SWITCH] = mode.kit_switch
        if (base.get(LEVERS_OFF_VAR) or "").strip():
            env[LEVERS_OFF_VAR] = base[LEVERS_OFF_VAR].strip()   # the ablation word, resolved by name by the launcher before this arm starts (cli.launch_context) and once more at install in the arm
        env.setdefault(*KIT_ALLOCATOR)          # the operator's own setting, when present in the base, is honoured as given
    env["PYTHONDONTWRITEBYTECODE"] = "1"       # the kit directories carry the kit's bytes only (-I ignores it; the venv's python does not)
    if int(det or 0) >= 1:
        env.update(DET_PINS)
        if det_cache(base, det)["form"] == "per-box":
            env.update(jit_cache_dirs("per-box", local_root=base.get(JIT_LOCAL_ROOT_VAR)))
    return env


def env_words(env: Dict[str, str], det: int) -> str:
    """`` det_pins=<VAR=v,…|none> allocator=<PYTORCH_CUDA_ALLOC_CONF value|unset>`` — the JIT line's words for what arm_env added beyond the caches."""
    pins = ",".join(f"{k}={env.get(k)}" for k in DET_PINS if int(det or 0) >= 1 and k in env) or "none"
    return f" det_pins={pins} allocator={env.get(KIT_ALLOCATOR[0]) or 'unset'}"


TORCH_DEFAULT_INDUCTOR_RE = re.compile(r"^/tmp/torchinductor_[^/]+/?$")     # torch's own default (torch._inductor cache_dir()), which torch WRITES into os.environ when the variable is unset


def launch_base(environ: Optional[Dict[str, str]] = None):
    """The base environment of a launch — ``(base, notes)``. Read it BEFORE anything in this process imports torch: torch's inductor writes
    its ``/tmp/torchinductor_<user>`` default into ``os.environ`` when ``TORCHINDUCTOR_CACHE_DIR`` is unset, and the launch facts import
    torch and the upstream attention modules. The JIT-cache pair is completed by the shared-root rule: when the shared root is named
    (``MODEL_OPT_JIT_ROOT``), exactly one of ``TRITON_CACHE_DIR`` / ``TORCHINDUCTOR_CACHE_DIR`` names a directory under ``<root>/<key>/``
    and the other is unset (torch's ``/tmp/torchinductor_<user>`` default counts as unset), the other becomes its keyed sibling
    (``<root>/<key>/{triton,inductor}``, modes.jit_cache_dirs) — one ``(what, consequence)`` note per completed variable, for the caller's
    NOTE line. Nothing else is touched; a pair that is wholly set, wholly unset, per-box, or read without a shared root is returned as read."""
    base = dict(os.environ if environ is None else environ)
    notes = []
    read = {v: base.get(v) or None for v in JIT_DIR_VARS}
    eff = {v: (None if d is None or (v == "TORCHINDUCTOR_CACHE_DIR" and TORCH_DEFAULT_INDUCTOR_RE.match(d)) else d) for v, d in read.items()}
    have = [v for v, d in eff.items() if d]; lack = [v for v, d in eff.items() if not d]
    shared_root = (base.get(JIT_SHARED_ROOT_VAR) or "").rstrip("/")
    if shared_root and len(have) == 1 and len(lack) == 1:
        sv, uv = have[0], lack[0]; sd = eff[sv]
        root = shared_root + "/"
        if (sd.rstrip("/") + "/").startswith(root):
            key = sd[len(root):].split("/")[0]
            if key:
                sibling = jit_cache_dirs("shared", key=key, shared_root=shared_root)[uv]
                base[uv] = sibling
                was = "unset" if read[uv] is None else f"= {read[uv]} (torch's default, = unset)"
                notes.append((f"{uv} {was}", f"keyed alongside {sv}={sd}: {uv}={sibling}"))
    return base, notes


def arm_jit_view(base: Dict[str, str], env: Dict[str, str]) -> Dict[str, str]:
    """The environment the launch-time JIT check judges: ``base`` (its switch ``EF2INV_JIT_CACHE`` and everything else) with the two cache
    directories THE ARM runs with (``arm_env``'s: the per-box pair of a det arm, else the base's pass-through pair)."""
    view = {k: v for k, v in base.items() if k not in JIT_DIR_VARS}
    view.update({v: env[v] for v in JIT_DIR_VARS if env.get(v)})
    return view


def det_cache(base: Dict[str, str], det: int) -> Dict[str, object]:
    """The JIT-cache form of an arm: at ``det >= 1`` the det-arm default ``per-box`` unless ``EF2INV_JIT_CACHE`` is set in ``base`` (the
    operator's explicit choice, reported as ``explicit``; ``shared`` on a det arm is a NAMED deviation the caller prints); at ``det 0``
    the form the deployment's dirs imply. An unknown / unshipped form raises by name (modes.jit_cache_form)."""
    switch = base.get(JIT_CACHE_VAR)
    if switch is not None:
        return {"form": jit_cache_form(switch), "explicit": True}
    if int(det or 0) >= 1:
        return {"form": "per-box", "explicit": False}
    return {"form": jit_cache_state(base)["form"], "explicit": False}


def argv_for(mode: Mode, model_opt: str, *, cookbook_stock: str, target_name: str, target_sequence: Optional[str] = None, binder_len: Optional[int], seed: int, out: str,
             target_hotspot_ids: Optional[List[str]] = None, det: int = 0, python: Optional[str] = None,
             chunk_size: object = "shipped", kernel_backend: object = "shipped", binder_name: Optional[str] = None,
             require_fast_env: int = 0, batch_size: int = 1, is_antibody: Optional[int] = None, epitope_contact_distance: float = 12.0,
             binder_sequence: Optional[str] = None, use_scaling_critics: int = 0, upstream_fix: Optional[List[str]] = None) -> List[str]:
    kp = kit_paths(model_opt)
    cookbook = cookbook_stock                                                # every mode runs the one stock file; a kit arm carries EF2_FAST_KIT + the design kit's k/ (below)
    binder = ["--binder-len", str(int(binder_len))] if binder_len is not None else ["--binder-name", str(binder_name or "minibinder")]   # a fixed length, else the cookbook's own factory route (settings.STOCK_BINDER_NAME)
    argv = [python or sys.executable, "-I", "-m", "ef2inv_opt.stock_design", "--mode", mode.name, "--cookbook", cookbook,
            "--pins", os.path.join(model_opt, "stock", "PINS.json"), "--target-name", str(target_name),
            *(["--target-sequence", str(target_sequence)] if target_sequence is not None else []),   # the cookbook's main(target_name, target_sequence): a built-in name alone, or a name + its sequence
            *binder, "--seed", str(seed), "--out", out, "--det", str(det)]
    if target_hotspot_ids:                           # the cookbook's design(target_hotspot_ids): stock's name and list form, a token only when given
        argv += ["--target-hotspot-ids", *[str(h) for h in target_hotspot_ids]]
    if mode.is_kit:
        argv += ["--kit-dir", kp["k_dir"]]
    if int(batch_size) != 1:                         # the cookbook's own knobs (settings.STOCK_KNOBS): a token only when the value is not the default, so the flags absent = the same argv
        argv += ["--batch-size", str(int(batch_size))]
    if is_antibody is not None:
        argv += ["--is-antibody", str(int(is_antibody))]
    if float(epitope_contact_distance) != 12.0:
        argv += ["--epitope-contact-distance", str(float(epitope_contact_distance))]
    if binder_sequence is not None:
        argv += ["--binder-sequence", str(binder_sequence)]
    if int(use_scaling_critics):
        argv += ["--use-scaling-critics", "1"]
    if int(require_fast_env or 0):                   # configs/h100.env EF2INV_REQUIRE_FAST_ENV=1, read by the CLI (the variable itself never reaches the arm)
        argv += ["--require-fast-env", "1"]
    if chunk_size != "shipped":                      # upstream's setters ride the argv only when called (the stock arm's `--chunk-size none --kernel-backend cuequivariance`, exact's pins; plain off / fast / big nothing)
        argv += ["--chunk-size", "none" if chunk_size is None else str(chunk_size)]
    if kernel_backend != "shipped":
        argv += ["--kernel-backend", "None" if kernel_backend is None else str(kernel_backend)]
    if upstream_fix:                                 # --upstream-fix: the resolved IDs, a token only when given (the flag absent = the same argv as before, byte for byte)
        argv += ["--upstream-fix", ",".join(upstream_fix)]
    return argv


def run(argv: List[str], env: Dict[str, str], log_path: str, timeout_s: Optional[float] = None) -> dict:
    """Run the arm; stdout+stderr tee'd to ``log_path``. Returns exit code, wall, timed_out."""
    t0 = time.time()
    with open(log_path, "ab") as logf:
        logf.write(("[ef2inv-opt launch] " + " ".join(argv) + "\n").encode())
        p = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            while True:
                line = p.stdout.readline()
                if not line:
                    break
                logf.write(line); logf.flush()
                sys.stderr.write(line.decode(errors="replace"))
                if timeout_s and time.time() - t0 > timeout_s:
                    timed_out = True
                    os.killpg(p.pid, signal.SIGKILL)
                    break
            rc = p.wait()
        finally:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
                rc = p.wait()
    if timed_out:
        rc = 124
    return {"exit_code": rc, "wall_s": round(time.time() - t0, 3), "timed_out": timed_out, "argv": argv}
