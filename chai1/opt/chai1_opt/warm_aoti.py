"""python -m chai1_opt.warm_aoti [--mode fast|big] [--crops 256,384,512,768,1024,1536,2048] [--num-diffn-samples 5] [--force]

Builds, on the GPU present, the ahead-of-time packages of the `compiled` lever's per-step function for the mode's sampler row (``chai1_fastln.aoti``):
one package per crop under ``$MODEL_OPT_JIT_ROOT/<key>/aoti/``. A later ``pred`` of that mode on this card class loads a crop's package
instead of compiling the step through Dynamo at its first fold of the crop (once per crop per process). Needs ``MODEL_OPT_JIT_ROOT`` set,
the kit's image (a C++ compiler and the CUDA development headers, ``CUDA_HOME``) and the weights (``CHAI_DOWNLOADS_DIR``). One export and one compile per crop
(the largest crops take longest); existing packages of the running code are kept unless ``--force``.
The inputs are synthetic tensors of each crop's shapes: nothing is folded. Prints one line per crop and a JSON summary (``--json``)."""
from __future__ import annotations

import argparse, json, os, sys, time
from typing import Sequence


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m chai1_opt.warm_aoti", description="build the sampler step's ahead-of-time packages for this card")
    ap.add_argument("--mode", default=None, help="fast | big (default: the package default mode if it carries the compiled lever, else fast)")
    ap.add_argument("--crops", default="256,384,512,768,1024,1536,2048", help="comma list of model crops (stock's ladder: 256 … 2048)")
    ap.add_argument("--num-diffn-samples", type=int, default=5, help="diffusion samples per fold the packages serve (stock's default 5; another count compiles through Dynamo at run time)")
    ap.add_argument("--unaligned", default=os.environ.get("CHAI1_AOTI_UNALIGNED", ""), help="comma list of step input names (arg:<name> / cache:<key>, or the bare "
                    "name) compiled WITHOUT the 16-byte alignment assumption — for an input a mode hands over as an unaligned view (the fold names it: 'AOTI input realigned: …')")
    ap.add_argument("--force", action="store_true", help="rebuild packages that already exist for the running code")
    ap.add_argument("--json", action="store_true", help="print the summary as JSON")
    a = ap.parse_args(argv)
    from . import jit as _jit, modes, stack
    mode = a.mode or (modes.DEFAULT_MODE if "compiled" in modes.KIT_MODES.get(modes.DEFAULT_MODE, modes.KIT_MODES["fast"]).dstep else "fast")
    if mode not in modes.KIT_MODES: print(f"warm_aoti: '{mode}' is not a kit mode ({', '.join(modes.KIT_MODES)})", file=sys.stderr); return 2
    modes.apply_levers_off()
    km = modes.KIT_MODES[mode]
    dstep = tuple(km.dstep)
    if "compiled" not in dstep: print(f"warm_aoti: mode {mode} does not carry the compiled lever (dstep={','.join(dstep) or 'none'}): nothing to build"); return 0
    if not os.environ.get(_jit.ENV_ROOT): print(f"warm_aoti: {_jit.ENV_ROOT} is unset — the packages need a persistent JIT root (export {_jit.ENV_ROOT}=<dir>)", file=sys.stderr); return 3
    import torch
    if not torch.cuda.is_available(): print("warm_aoti: no GPU in this process — the packages are compiled and tuned on the card they serve", file=sys.stderr); return 4
    jit_rep = _jit.apply(torch=torch)
    optin = modes.with_implied(mode, modes.parse_optin(None))
    stack.apply_optin_pre(optin, torch, export_alloc=False)                      # the row's numerics words (TF32) as pred applies them
    X = stack.dstep_module(); S = stack.eager_module()                           # chai1_fastln.stackx / chai1_eager.stack, imported the way the kit imports them
    import chai1_eager.hoist as H
    from chai1_fastln import aoti as A
    levers = tuple(lv for lv in dstep if lv != "compiled")
    comps = S.Components(device="cuda:0")
    flat = X.new_flat(comps, f"warm_aoti_{mode}")
    pols = X.apply_levers(flat, levers)
    hoister = pols.get("hoister", "base"); r = pols.get("dit_attn")
    lw = A.levers_word_of(hoister=hoister, dit_attn=bool(r is not None and not getattr(r, "aside", None)))
    d = A.package_dir(create=True)
    crops = [int(c) for c in a.crops.split(",") if c.strip()]
    print(f"[chai1-opt] WARM AOTI mode={mode} levers={lw} n_samples={a.num_diffn_samples} crops={','.join(map(str, crops))} dir={d} key_note={jit_rep.get('note', '') if isinstance(jit_rep, dict) else ''}", flush=True)
    rows, t_all = [], time.perf_counter()
    for crop in crops:
        t0 = time.perf_counter(); row = {"crop": crop}
        try:
            hf = H.make(hoister, flat, f"forward_{crop}")
            have = A.find(d, crop, a.num_diffn_samples, lw, A.step_digest(hf))
            if have and not a.force:
                row.update(status="present", package=os.path.basename(have[0])); print(f"[chai1-opt] WARM AOTI crop={crop} present {os.path.basename(have[0])}", flush=True)
            else:
                kw = A.synthetic_inputs(crop, a.num_diffn_samples, device=comps.dev)
                meta = A.build(hf, kw, d, levers_word=lw, log=lambda s: print(s, flush=True), unaligned=[u.strip() for u in str(a.unaligned or "").split(",") if u.strip()])
                row.update(status="built", package=meta["name"] + ".pt2", export_s=meta["export_s"], compile_s=meta["compile_s"], MB=round(meta["package_bytes"] / 1e6, 1), route=meta["export_route"])
            del hf
        except Exception as e:  # noqa: BLE001 — a crop that cannot be built is named; the run-time route for it stays Dynamo
            row.update(status="failed", error=f"{type(e).__name__}: {str(e)[:300]}"); print(f"[chai1-opt] WARM AOTI crop={crop} FAILED {row['error']}", flush=True)
        row["wall_s"] = round(time.perf_counter() - t0, 1); rows.append(row)
        torch.cuda.empty_cache()
    summary = {"mode": mode, "levers": lw, "n_samples": a.num_diffn_samples, "dir": d, "wall_s": round(time.perf_counter() - t_all, 1), "crops": rows}
    print(json.dumps(summary) if a.json else f"[chai1-opt] WARM AOTI done in {summary['wall_s']} s: " + ", ".join(f"{r['crop']}={r['status']}" for r in rows), flush=True)
    return 0 if all(r["status"] in ("built", "present") for r in rows) else 5


def applicable(mode: str, environ=None):
    """(True, None) when ``mode`` carries the compiled lever after MODEL_OPT_LEVERS_OFF and a JIT root is set; else (False, <the reason by name>):
    the warm step is skipped BY NAME, never silently (a mode that never compiles, the lever switched off, no MODEL_OPT_JIT_ROOT to hold packages,
    no CUDA development headers for the launcher's compile — CUDA_HOME/include/cuda.h)."""
    from . import jit, modes
    env = os.environ if environ is None else environ
    modes.apply_levers_off(env)
    km = modes.KIT_MODES.get(mode)
    if km is None or "compiled" not in tuple(getattr(km, "dstep", ()) or ()):
        return False, f"mode {mode} does not carry the compiled step"
    if not env.get(jit.ENV_ROOT):
        return False, f"{jit.ENV_ROOT} is unset (the packages live under $MODEL_OPT_JIT_ROOT/<key>/aoti; configs/<card>.env or the caller sets it)"
    home = env.get("CUDA_HOME") or "/usr/local/cuda"
    if not os.path.isfile(os.path.join(home, "include", "cuda.h")):
        return False, f"no CUDA development headers at {home}/include (the compiled step's launcher links against them; environment/Dockerfile installs cuda-cudart-dev)"
    return True, None


def step(mode: str, *, echo: bool = True, extra: Sequence[str] = ()) -> dict:
    """``run.sh warm``'s second statement: the mode's ahead-of-time step packages for this card in a subprocess (``python -m chai1_opt.warm_aoti
    --mode <mode> --json``), its lines relayed; returns {"status": built|present|skipped|failed, "reason", "wall_s", "crops": [...], "dir"}."""
    import subprocess, time as _time
    ok, why = applicable(mode)
    if not ok:
        return {"status": "skipped", "reason": why, "wall_s": 0.0, "crops": []}
    cmd = [sys.executable, "-m", "chai1_opt.warm_aoti", "--mode", mode, "--json", *extra]
    t0 = _time.time(); last = None; lines = []
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for ln in p.stdout:
        lines.append(ln)
        if ln.startswith("{") and '"crops"' in ln:
            last = ln
        elif echo:
            sys.stderr.write(ln); sys.stderr.flush()
    rc = p.wait()
    res = {"status": "failed", "reason": f"exit {rc}", "wall_s": round(_time.time() - t0, 1), "crops": [], "command": cmd}
    if last:
        try:
            summ = json.loads(last); rows = summ.get("crops") or []
            st = "failed" if (rc != 0 or any(r.get("status") == "failed" for r in rows)) else ("built" if any(r.get("status") == "built" for r in rows) else "present")
            res.update(status=st, reason=None if st != "failed" else ";".join(f"{r['crop']}:{r.get('error')}" for r in rows if r.get("status") == "failed")[:400] or f"exit {rc}",
                       crops=rows, dir=summ.get("dir"), levers=summ.get("levers"), n_samples=summ.get("n_samples"))
        except Exception as e:  # noqa: BLE001
            res["reason"] = f"exit {rc}; summary unreadable: {e}"
    return res


def step_line(res: dict) -> str:
    crops = ",".join(f"{r['crop']}={r['status']}" for r in res.get("crops") or []) or "none"
    return (f"[chai1-opt] WARM AOTI {res.get('status')} crops={crops} wall={res.get('wall_s')}s" + (f" dir={res['dir']}" if res.get("dir") else "") +
            (f" reason={res['reason']}" if res.get("reason") else ""))


if __name__ == "__main__":
    sys.exit(main())
