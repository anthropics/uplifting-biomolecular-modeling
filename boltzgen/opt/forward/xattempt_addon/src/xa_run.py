#!/usr/bin/env python
# xa_run.py — the runner every kit mode launches on ONE configured BoltzGen job directory (the output of `boltzgen configure`):
# the in-process pipeline and the CUDA-graph design sampler (bg_inproc / bg_graph_patch, in opt/forward/fast_inference/src) plus this
# directory's exact levers `fastinit` (xa_fastinit) and `hoist` (xa_hoist).
#   usage: python src/xa_run.py <run_dir> <seed> [step1,step2,...]
#   env:   XA_FAST_INIT=0 / XA_HOIST=0 switch levers off; BG_GRAPH=graph|predraw|off selects the sampler (default graph).
# Requires opt/forward/fast_inference/src on PYTHONPATH after this directory: its `sitecustomize` -> bg_hook (seeding fix + timing) runs at interpreter start; bg_inproc.py is this tree's own copy in that directory, found from this file's real location (a runner copied out of the tree: the first PYTHONPATH entry that holds one).
# Step handling: the GPU steps (design, inverse_folding, folding, design_folding, affinity) run in THIS process through bg_inproc.py
# (seeded per step: seed+step_index); the CPU steps (analysis, filtering) run afterwards as stock `python main.py <config>` subprocesses —
# `analysis` uses a multiprocessing 'spawn' pool whose workers re-import the __main__ script, which must not re-run a pipeline
# (bg_inproc.py has no __main__ guard; run through this launcher it is never __main__ of a spawned worker).
import os, sys, subprocess

GPU_STEPS = ("design", "inverse_folding", "folding", "design_folding", "affinity")


def _main():
    here = os.path.dirname(os.path.realpath(__file__))
    os.environ.setdefault("BG_GRAPH", "graph"); os.environ.setdefault("XA_FAST_INIT", "1"); os.environ.setdefault("XA_HOIST", "1")
    if len(sys.argv) < 3:
        sys.exit("usage: xa_run.py <run_dir> <seed> [comma-separated steps]")
    run_dir, seed = sys.argv[1], sys.argv[2]
    import yaml
    steps = [s["name"] for s in yaml.safe_load(open(os.path.join(run_dir, "steps.yaml")))["steps"]]
    want = sys.argv[3].split(",") if len(sys.argv) > 3 else steps
    gpu = [s for s in steps if s in want and s in GPU_STEPS]; cpu = [s for s in steps if s in want and s not in GPU_STEPS]
    # locate bg_inproc.py by path (its module body runs the pipeline at import; never import it): this tree's own copy, reached from this file's real location, before any path entry — a bg_inproc.py in the working directory or on a relative / foreign PYTHONPATH entry never stands in for it
    own = os.path.join(os.path.dirname(os.path.dirname(here)), "fast_inference", "src", "bg_inproc.py")
    found = [c for c in (os.path.join(p, "bg_inproc.py") for p in sys.path if os.path.realpath(p) != here) if os.path.exists(c)]
    kit_inproc = own if os.path.isfile(own) else (found[0] if found else None)
    if kit_inproc == own and found and os.path.realpath(found[0]) != os.path.realpath(own):
        print(f"[xa_run] bg_inproc.py: NOT RUN: {found[0]} (first on sys.path, not this tree's) — this runner runs its own {own}; remove that file or that PYTHONPATH entry to drop this line", file=sys.stderr, flush=True)
    if kit_inproc is None:
        sys.exit("xa_run.py: partner kit module bg_inproc.py not found beside this add-on (../../fast_inference/src) or on PYTHONPATH (export PYTHONPATH=<addon>/src:<partner_kit>/src)")
    if gpu:
        if os.environ.get("XA_FAST_INIT", "1") == "1":
            import xa_fastinit  # noqa: F401
        if os.environ.get("XA_HOIST", "1") == "1":
            import xa_hoist  # noqa: F401
        import atexit
        def _stats():
            for m in ("xa_fastinit", "xa_hoist"):
                if m in sys.modules:
                    print(f"[xa_run] {m} stats:", {k: v for k, v in sys.modules[m].STATS.items() if k not in ("loads",)}, file=sys.stderr, flush=True)
        atexit.register(_stats)
        # execute bg_inproc.py's code here with its argv, WITHOUT making it the __main__ module (see header)
        saved = sys.argv[:]
        sys.argv = [kit_inproc, run_dir, seed, ",".join(gpu)]
        try:
            code = compile(open(kit_inproc).read(), kit_inproc, "exec")
            exec(code, {"__name__": "bg_inproc_exec", "__file__": kit_inproc})
        finally:
            sys.argv = saved
        try:
            import torch, gc; gc.collect(); torch.cuda.empty_cache()
        except Exception:
            pass
    if cpu:
        import boltzgen
        main_py = os.path.join(os.path.dirname(boltzgen.__file__), "resources", "main.py")
        cfgs = {s["name"]: s["config_file"] for s in yaml.safe_load(open(os.path.join(run_dir, "steps.yaml")))["steps"]}
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}   # stock process: no hooks needed for CPU steps
        for i, name in enumerate(cpu):
            env["BOLTZGEN_PIPELINE_STEP"] = name                                # the step's own name in ITS environment, as upstream's `run` starts every step (this process keeps the GPU steps' name)
            rc = subprocess.call([sys.executable, main_py, os.path.join(run_dir, cfgs[name])], env=env)
            print(f"[xa_run] step {name} (stock subprocess) rc={rc}", file=sys.stderr, flush=True)
            if rc != 0:
                sys.exit(rc)


if __name__ == "__main__":
    _main()
