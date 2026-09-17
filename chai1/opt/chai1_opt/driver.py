"""The kit-driver launcher — the route of a kit mode: ``kit/chai_worker.py`` as shipped, in this process.

    python -m chai1_opt.driver --mode exact|fast|big [fold flags] --uids <spec,...> --seeds 0,1 --out_dir <dir> --tag <tag> --msa_dir <dir> [--det 0|1] [--optin <the mode's implied levers>]
    (--seed / --msa-directory are the same flags in stock's spelling)

What happens, in order (``run``):
  1. the kit's ``kit/`` directory goes first on ``sys.path`` and the kit's own ``chai_proto`` is imported (stdlib at import) — the same
     module object the driver will import a moment later;
  2. the fold settings (stock's run_inference flags, settings.py) are written INTO the kit's ``RUN_KW`` dict (``settings.to_driver_run_kw``):
     the driver reads ``cp.RUN_KW`` at its fold call (the per-seed ``run_folding_on_context`` call in ``chai_worker.py``), so they reach upstream's ``run_folding_on_context`` through the driver's
     own code path. This is the one documented deviation from "the driver as shipped": the driver hard-codes its own fold
     settings and has no flag for them;
  3. two hooks on the kit module, both wrappers that call the kit's function first: ``outputs.install_extra_samples_hook`` (keep every
     diffusion sample's two upstream files, a no-op at one sample) and a wrapper on ``env_report`` that, once the driver has installed its levers and
     applied the recipe (its ``DET_MODE`` statement, before ``env_report``), applies the mode's implied levers (``--optin``, written by the launcher from
     ``modes.KitMode.implied_optin``: precision.apply, gated by modes.optin_refusal before anything runs), installs the mode's eager lever through the eager stack's own ``install``
     (``stack.apply_eager``: on top of the driver's W1 memo, after the recipe and the opt-in flags the stack's graph capture reads) and
     prints the ACTIVE and ``ready`` lines; an eager stack that fails to install, or an opt-in lever that does not show applied, stops the
     process there with its NOT ACTIVE line and exit 3 — the driver never folds on the kit's levers alone under a mode that names more;
  4. the exit tally is registered on the driver's own globals (report.py), the argv the driver expects is set, the working directory
     becomes ``out_dir`` (the driver's relative ``work_chai/`` lands and is removed there), and the driver's bytes execute as
     ``__main__`` — its ``sys.exit`` code is the process's (0 = every seed-fold ok, 2 = a failure; the worker's last statement) unless a
     gate at exit refuses the run (HOIST MISMATCH / PAIRTRACK GATE REFUSED: 1 over a 0; a partial memory item: 3, ``big.exit_gate``).

The deterministic recipe is the driver's own: with ``--det 1`` the launcher sets ``CHAI_DETERMINISTIC=1`` in its environment and the
kit's ``apply_deterministic_mode`` runs inside the driver (det.py). No other environment variable is set.

Mode ``big`` (big.py; exact / fast apply their size-gated memory levers the same way): the memory line is applied through ``opt_core.mem`` before the driver's bytes import torch (a refused lever
is named and nothing folds); the eager line is installed un-graphed and the levers land on the adopted trunk / denoiser instances in
``env_report``; the per-item census and the fail-closed gate run at exit (``big.exit_gate``: a partial item exits 3 unless
``--allow-partial``, recorded; ``--allow-partial`` is also the opt-out that lets a partial ACTIVATION — a lever of the mode the run's own
switches turned off — proceed instead of being refused by name). ``--n_gpu`` (ngpu.py): P ∈ {1}; the report carries it.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from . import __version__
from . import alloc, big, det, hoist, modes, ngpu, outputs, pairtrack, registry, report, settings, stack

T0 = time.time()


def parse(argv):
    ap = argparse.ArgumentParser(prog="python -m chai1_opt.driver", description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", required=True, choices=list(modes.KIT_MODE_NAMES))
    settings.add_fold_arguments(ap)                                           # stock's fold flags, as pred composes them (settings.fold_flags)
    ap.add_argument("--uids", required=True, help="the driver's input specs, comma-joined (inputs.uids_arg)")
    ap.add_argument("--seed", "--seeds", dest="seeds", default=None, help="the seed list every input of this call is folded at (outputs per seed); required — refused by name when absent")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--msa_dir", "--msa-directory", dest="msa_dir", required=True)
    ap.add_argument("--no-compile", action="store_true", help="alias of MODEL_OPT_LEVERS_OFF=compiled (the sanctioned opt-out of the compiled denoiser step)")
    ap.add_argument("--allow-partial", action="store_true", help="the recorded opt-out of the mode's completeness rules: a partial ACTIVATION (a lever of the mode the run's own switches turned off) is refused by name without it and proceeds, named (PARTIAL, one NOTE line), with it; the memory mode's per-item gate at exit likewise (a partial ITEM exits 3 without it)")
    ap.add_argument("--det", type=int, default=det.DEFAULT_LEVEL, choices=list(det.LEVELS))
    ap.add_argument("--optin", default=None, help=f"comma list of the mode's implied levers, written by the launcher (cli.py) from modes.KitMode.implied_optin ({','.join(modes.OPTIN_LEVERS)})")
    ap.add_argument("--n_gpu", default=None, help="GPUs per fold (ngpu.SUPPORTED); refused by name otherwise")
    return ap.parse_args(argv)


def driver_argv(a, driver_path: str) -> list:
    """The argv ``kit/chai_worker.py`` parses: ``<uids> <seeds> <out_root> --tag T --levels L --msa_dir D``."""
    return [driver_path, a.uids, a.seeds, a.out_dir, "--tag", a.tag, "--levels", modes.driver_levels(a.mode), "--msa_dir", a.msa_dir]


def apply_settings(cp, st: dict) -> dict:
    """(2) Write the fold settings INTO the kit's ``RUN_KW`` dict (the same object the driver reads at its fold call) and return the dict
    written (settings.to_driver_run_kw: every fold keyword, the trunk-sample count and the device)."""
    run_kw = settings.to_driver_run_kw(st)
    cp.RUN_KW.clear(); cp.RUN_KW.update(run_kw)
    sys.stderr.write(report.settings_line(report.PREFIX, cp.RUN_KW) + "\n"); sys.stderr.flush()  # the dict the driver folds with, as written (once per pass)
    return run_kw


def run(a) -> int:
    if getattr(a, "no_compile", False):
        modes.no_compile_to_env()                                             # --no-compile == MODEL_OPT_LEVERS_OFF=compiled
    modes.apply_levers_off()                                                  # MODEL_OPT_LEVERS_OFF: the rows reduced by its names before anything reads them; a bad name is modes.refusal's
    km = modes.kit_mode(a.mode)
    optin = modes.with_implied(a.mode, modes.parse_optin(a.optin))   # the row's implied levers first (alloc on every kit row, tf32 on fast / big)
    why = modes.refusal(a.mode) or modes.optin_refusal(a.mode, optin) or ngpu.refusal(a.n_gpu, a.mode)
    if why:
        report.print_activation({"mode": a.mode, "reason": why}); return 3
    n_gpu = ngpu.resolve(a.n_gpu, a.mode)
    st = settings.from_args(a)                                                # the fold settings: stock's defaults unless a flag says otherwise
    r = settings.seed_refusal(a.mode, {"the --uids of this call": settings.parse_seeds(a.seeds)})   # no seed: by name
    if r:
        report.print_activation({"mode": a.mode, "reason": r}); return 3
    dpath = stack.driver_path()                                               # the worker the tree runs (opt/forward/errata_02/chai_worker.py)
    kit_pkg = stack.kit_pkg_dir()                                             # the kit's own modules (kit/), first on sys.path
    if kit_pkg not in sys.path:
        sys.path.insert(0, kit_pkg)
    os.environ["KIT"] = kit_pkg                                               # the amended worker's own switch for the same directory
    from opt_core.oom import is_oom                                           # an out-of-memory error is re-raised first at every handler below, never a refusal
    if km.memory or km.memory_gated:                                          # the memory levers (opt_core.mem) BEFORE the driver's bytes import torch: big's line, or
        try:                                                                  # exact / fast's gated levers (at crop 2048) and fast's msa_rows; a refused lever is named and nothing folds
            big.apply_line(None, out_dir=os.path.join(os.path.abspath(a.out_dir), a.tag), mode=a.mode, allow_partial=bool(a.allow_partial))   # --allow-partial: the apply-time value the exit gate confirms
        except Exception as e:  # noqa: BLE001
            if is_oom(e): raise                                               # noqa: E701
            report.print_activation({"mode": a.mode, "reason": f"big refused — {e}"}); return 3
    import chai_proto as cp                                                   # the kit's own module (kit/chai_proto.py)
    assert os.path.samefile(cp.__file__, os.path.join(kit_pkg, "chai_proto.py")), cp.__file__
    for k, v in det.env_for_driver(a.det).items():
        os.environ[k] = v
    run_kw = apply_settings(cp, st)                                           # (2) the fold settings, through the driver's own path
    outputs.install_extra_samples_hook(cp)                                    # (3)
    rep = stack.base_report(a.mode, trigger="driver", dry_run=False, det=a.det, optin=optin, n_gpu=n_gpu)
    if "alloc" in optin:                                                      # the allocator policy: before this process imports torch (the kit does, below)
        try:
            rep["optin_applied"] = alloc.export()
        except alloc.LeverUnavailable as e:
            rep.update(active=False, reason=str(e)); report.print_activation(rep); return 3
    rep.update(route="driver", run_kw=dict(run_kw), out_dir=os.path.abspath(a.out_dir), tag=a.tag)
    if km.memory or km.memory_gated:
        big.fill(rep)                                                       # the applied record's fields (the ACTIVE line's big= field: big's line or the gated levers)
    g = {"__name__": "__main__", "__file__": dpath, "__builtins__": __builtins__}
    worker_ns = g                                                             # the worker's own namespace (levels, ESM_CACHE, LOAD_LOG …) — env_report below binds its own `g`
    report.register_exit_tally(g, route="driver")                             # (4)
    loop_levers = [n for n in km.levels if n not in registry.ATTR_PROBES]

    orig_env_report = cp.env_report

    def env_report(*args, **kw):                                              # the worker's env_report call — levers and recipe are in place
        r = orig_env_report(*args, **kw)
        try:
            import chai_lab.chai1 as chai1_mod
            import importlib
            esm_mod = importlib.import_module(stack.ESM_MODULE)
            g = stack.gpu_info(use_torch=True)
            import torch
            from . import jit as _jit
            rep["jit"] = _jit.apply(torch=torch)                                       # JIT caches keyed under MODEL_OPT_JIT_ROOT (if set): after torch's import (its version words), before CUDA init and any compile; one NOTE line
            sys.stderr.write(f"{report.PREFIX} NOTE {_jit.note(rep['jit'])}\n")
            rep["optin_applied"] = dict(rep.get("optin_applied") or {}, **stack.apply_optin_pre(optin, torch, export_alloc=False))   # tf32: after the driver's recipe, before the eager stack's graph capture (alloc was exported at start)
            stack.install_msa_form(chai1_mod)                                          # every fold's MSA form = chai-lab's own for the item (the off route's rule), every kit mode
            from . import forward_timer
            forward_timer.install(chai1_mod, report.PREFIX)                            # the FORWARD line around run_folding_on_context (timing only; the off route prints the same grammar)
            rep["esm_memo_scope"] = stack.esm_memo_scope()                            # CHAI1_OPT_ESM_MEMO_SCOPE (validated by name; the CLI's dry run read the same word)
            stack.install_esm_memo_scope(chai1_mod, worker_ns, rep["esm_memo_scope"]) # the W5 memo's scope: global (default) | input (emptied per input's feature context)
            rep["msa_form_rule"] = "chai-lab's own per item: --msa_dir when a chain's .aligned.pqt is there, else none (stock_fold.msa_directory_for)"
            if km.eager:
                stack.apply_eager(km.eager, dstep=km.dstep, pairtrack=km.pairtrack, mode=km.name)   # the kits' own installs, after the driver's levers + recipe (the row as reduced by MODEL_OPT_LEVERS_OFF)
                rep["hoist"] = dict(hoist.STATE)                                      # the adopted denoiser's key and release point
            from . import precision
            rep["numerics"] = precision.numerics(torch)                                # the process numerics signature the recipe and the levers left in force
            c = stack.classify(chai1_mod, esm_mod)
            names = km.lever_names + optin
            rep.update(levers_applied=[n for n in names if n in c["applied"] or n in loop_levers], levers_off=[n for n in c["off"] if n in names],
                       partial=bool(set(names) - set(c["applied"]) - set(loop_levers)), gpu=g,
                       torch_version=stack.torch_version(), active=True, kit_env=r)
            rep["stack_key"] = stack.stack_key(cc=(rep["gpu"] or {}).get("cc"))
            missing = [n for n in ((km.eager,) if km.eager else ()) + km.dstep + km.pairtrack if n not in c["applied"]]
            if missing:
                rep.update(active=False, reason=f"the eager stack lever(s) {','.join(missing)} do not show installed after the install ran")
            missing = [n for n in optin if n not in c["applied"]]
            if missing and rep.get("active"):
                rep.update(active=False, reason=f"the opt-in lever(s) {','.join(missing)} do not show applied after precision.apply ran")
            why_partial = stack.partial_reason(rep, a.allow_partial)          # a partial activation is refused by name unless --allow-partial (then named: PARTIAL field, NOTE line) — both routes' one rule
            if why_partial and rep.get("active"):
                rep.update(active=False, reason=why_partial)
        except (alloc.LeverUnavailable, stack.ActivationError) as e:
            rep.update(active=False, reason=str(e))
        except Exception as e:  # noqa: BLE001
            if is_oom(e): raise                                               # noqa: E701
            rep.update(active=False, reason=f"activation in the driver failed: {e!r}")
        stack._STATE["report"] = rep
        stack.warm_imports(rep)                                                   # the core's early library imports, once, before the first fold
        report.print_activation(rep)
        if not rep.get("active"):
            sys.exit(3)                                                       # EXIT_NOT_ACTIVE: never a fold on a refused activation
        report.print_ready("driver", T0)
        return r
    cp.env_report = env_report

    sys.argv = driver_argv(a, dpath)
    os.makedirs(a.out_dir, exist_ok=True)
    os.chdir(a.out_dir)
    src = open(dpath, "r", encoding="utf-8").read()
    code = compile(src, dpath, "exec")
    rc = 0
    try:
        exec(code, g)  # noqa: S102  the kit driver, byte-for-byte, as __main__
    except SystemExit as e:
        rc = int(e.code or 0) if isinstance(e.code, int) or e.code is None else 1
    finally:
        hs = (stack.eager_stats() or {}).get("hoist") if km.eager else None
        n_items, n_ok = report.items_folded(g), report.items_ok(g)
        why = hoist.verdict(hs, n_items, n_ok)                                  # a completed seed-fold without its own hoist, or one hoisted twice: a refusal by name
        if why:
            sys.stderr.write(f"{report.PREFIX} HOIST MISMATCH {why}\n")
            rc = rc or 1
        unreached = hoist.unreached(hs, n_items, n_ok)                          # folds that failed before their denoiser ran: accounting, named (their SEED FAIL lines carry the cause)
        if unreached:
            sys.stderr.write(f"{report.PREFIX} HOIST NOTE hoist_precomputes={hs.get('n_precompute')}/{n_items}: {unreached} fold call(s) ended before "
                             f"their denoiser hoist (failed folds; {n_ok} completed, each with its own hoist)\n")
        why_pt = pairtrack.verdict()                                            # the trunk levers' fail-closed gate: an unexpected fallback, a kernel error or zero served calls
        if why_pt:
            sys.stderr.write(f"{report.PREFIX} PAIRTRACK GATE REFUSED {why_pt}\n")
            rc = rc or 1
        if km.memory or km.memory_gated:
            v = big.exit_gate(rc, allow_partial=bool(a.allow_partial))          # the per-item census + the fail-closed gate on a partial ITEM (exit 3 unless --allow-partial, recorded)
            rc = int(v.get("exit_code", rc))
            big.fill_exit(stack._STATE.get("report") or rep)                   # the gate census into the activation report too
    return rc


def main(argv=None) -> int:
    a = parse(sys.argv[1:] if argv is None else argv)
    try:                                                                       # a fatal signal inside native code (SIGILL / SIGSEGV / SIGBUS / SIGABRT / SIGFPE: e.g. an AOTI launcher built
        import faulthandler                                                    # for CPU instructions this host lacks) prints the Python stack of every thread to stderr instead of ending
        faulthandler.is_enabled() or faulthandler.enable(file=sys.stderr, all_threads=True)   # the process without a word (inert unless such a signal arrives)
    except Exception:  # noqa: BLE001
        pass
    from . import _core
    try:
        _core.entry_gate()                                                     # statement one: an absent, stale or older core exits 3 by name, nothing folds
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 3
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
