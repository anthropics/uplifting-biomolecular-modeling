"""The stock caller — mode ``off``: the upstream Python API, nothing from the kit on the path.

Upstream ESMFold2 has no command line; this file is the one place that makes the stock call (``--mode off`` execs it in a clean
subprocess: ``python -s -m esmfold2_opt.stock_fold ...``, every kit variable stripped from the environment). It imports
only the upstream packages (``transformers`` fork, ``esm``), torch, numpy and the package's own I/O modules (inputs.py / outputs.py,
which import nothing from the kit); the kit's driver modules are never on ``sys.path`` here, and the process proves it before
importing torch (``env_proof``): no forbidden environment variable, no kit module loaded, no kit directory on the path.

The stock call itself is the block marked ``# --- the stock call`` below: upstream's ``from_pretrained`` at its defaults, the
settings' two model calls (``set_kernel_backend`` / ``set_chunk_size`` — upstream's own API; the library's defaults ``None`` + chunk 64,
the model's own state after ``from_pretrained``; ``--backend fused`` selects upstream's fused backend with chunking off, ``--backend shipped``
makes no call) and ``ESMFold2InputBuilder().fold`` at the settings passed in (settings.py: fold keywords, A3M read form, seeds).
``--det 1`` applies the kit's deterministic recipe, ``--det 2`` level 1 plus the config-level switch (det.py: ``lm_encoder.lm_dropout=0``
on the model config before the weights load) — stock-side switches, never a mode. This is the only stock caller in the tree
(``run.sh pred --mode off`` and ``pred --mode off`` both reach it).

Outputs are the kit driver's file set (outputs.py). This process cannot write them itself — nothing from the kit is on its path — so
it stages every result as upstream produced it (``--stage-dir``; outputs.stage_result) and the ``pred`` process that launched it writes
the file set through the kit's own ``write_outputs``. Run standalone with ``--stage-dir`` and finish with outputs.finalise_staged.
The one ``[esmfold2-opt] ready variant=<v> t=<s>`` line is printed when the weights are on the device, then the one ``SETTINGS`` line: the
backend switch, the model calls made, the model's own switches as loaded and the fast-environment words (attn.py, read on the loaded model —
``atom_attn=flash_attn esmc_mlp=te esmc_attn=sdpa(chain_mask) esmc_rope=flash_attn_triton`` on the pinned image; ``atom_attn=sdpa esmc_mlp=torch
esmc_rope=torch`` on a stack without flash-attn / transformer-engine). Under
``ESMFOLD2_OPT_REQUIRE_FAST_ENV=1`` (configs/h100.env) the caller refuses (exit 3, one ``NOT ACTIVE`` sentence naming the failing words) before
loading the model when a required switch is not set, and again after loading if the bound reading disagrees — never a slow run under the switch.

Run standalone:
    python -s -m esmfold2_opt.stock_fold --variant full_msa --hf-repo biohub/ESMFold2 --input items.json --out_dir out --seeds 0,1 \\
        --settings-json '<Settings.as_dict() JSON>' [--det 0|1|2] [--backend fused|shipped] [--stage-dir DIR] [--env-absent EF2_,...]
"""
import argparse
import json
import os
import sys
from typing import Optional
import time

from . import modes, report


PREFIX = "[esmfold2-opt stock]"
BACKENDS = {"fused": [("set_kernel_backend", "fused"), ("set_chunk_size", None)],   # --backend name -> the model calls it selects: fused = upstream's fused kernel backend with pair-block chunking off;
            "shipped": []}                                                          # shipped = no model call at all: the model exactly as from_pretrained loads it (also bare --mode off)
KIT_MODULE_PREFIXES = ("ef2_", "run_ef2_", "m__tools__")                  # the kit driver's module names
KIT_PATH_MARKER = "ef2_server.py"                                        # a sys.path entry holding the kit server is a kit directory
DEFAULT_ENV_ABSENT = ("EF2_",)                                            # the kit's switch namespace; the package passes PINS.json's full list


def _forbidden(environ, spec) -> list:
    """Variables of `environ` matching `spec` (entries ending in '_' are prefixes, others exact names)."""
    hits = []
    for s in spec:
        for k in environ:
            if (k.startswith(s) if s.endswith("_") else k == s):
                hits.append(k)
    return sorted(set(hits))


def env_proof(env_absent, environ=None, modules=None, path=None) -> dict:
    """The clean-environment proof of this process: forbidden variables absent, no kit module loaded, no kit directory on sys.path,
    no esmfold2_opt activation hook armed. Raises RuntimeError listing every violation."""
    environ = os.environ if environ is None else environ
    modules = sys.modules if modules is None else modules
    path = sys.path if path is None else path
    hits = _forbidden(environ, env_absent)
    kit_mods = sorted(m for m in modules if m.startswith(KIT_MODULE_PREFIXES))
    kit_dirs = sorted(p for p in path if p and os.path.isfile(os.path.join(p, KIT_PATH_MARKER)))
    armed = [type(f).__name__ for f in sys.meta_path if type(f).__module__ == "opt_core.autoload"]
    proof = {"env_absent": list(env_absent), "forbidden_present": hits, "kit_modules_loaded": kit_mods, "kit_dirs_on_path": kit_dirs,
             "autoload_armed": armed, "torch_loaded_before_proof": "torch" in modules, "isolated": bool(sys.flags.isolated),
             "no_user_site": bool(sys.flags.no_user_site),
             "ok": not (hits or kit_mods or kit_dirs or armed)}
    if not proof["ok"]:
        raise RuntimeError(f"{PREFIX} NOT STOCK: forbidden env {hits}, kit modules {kit_mods}, kit dirs {kit_dirs}, hook {armed}")
    return proof


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="python -s -m esmfold2_opt.stock_fold", description="ESMFold2 stock call: upstream API only")
    ap.add_argument("--variant", required=True, choices=list(modes.VARIANTS))
    ap.add_argument("--hf-repo", required=True, help="the variant's checkpoint repository id (stock/PINS.json variants.<variant>.hf_repo)")
    ap.add_argument("--input", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seeds", default=None, help="comma list (per-item 'seeds' keys win; none anywhere = one unseeded fold per input, the library's seed=None)")
    ap.add_argument("--settings-json", required=True, help="Settings.as_dict() as JSON (settings.py)")
    ap.add_argument("--det", type=int, default=0, choices=(0, 1, 2), help="1 = the kit's deterministic recipe, 2 = level 1 + config-level lm_encoder.lm_dropout=0 (det.py)")
    ap.add_argument("--backend", default=None, choices=sorted(BACKENDS), help="the model calls after loading: fused = set_kernel_backend('fused') + set_chunk_size(None); shipped (also the default) = no model call, the model exactly as from_pretrained loads it")
    ap.add_argument("--stage-dir", default=None, help="stage results here for outputs.finalise_staged (the pred process writes the driver's file set); default <out_dir>/staged")
    ap.add_argument("--env-absent", default=",".join(DEFAULT_ENV_ABSENT), help="comma list of forbidden variable prefixes (trailing '_') / names")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args(argv)


def model_calls(settings, backend: str = None) -> list:
    """The upstream model calls ``--backend`` selects (``BACKENDS``: ``fused`` = the fused kernel backend with chunking off; ``shipped``, also
    the default, = NO model call: the model exactly as ``from_pretrained`` loads it)."""
    return list(BACKENDS[backend or "shipped"])


def as_loaded_state(model) -> dict:
    """What the loaded model's own modules say about the two switches, read (never set): the kernel-backend attribute and the chunk sizes,
    the MSA encoder's OuterProductMean modules apart from the pair blocks (``set_chunk_size`` sets both; the library as shipped chunks the
    pair blocks and leaves the outer-product mean unchunked). Values are the distinct ones found; ``"unread"`` when no module carries the
    attribute (a stand-in model)."""
    mods = list(model.modules()) if callable(getattr(model, "modules", None)) else [model]
    found = {"kernel_backend": set(), "chunk": set(), "opm_chunk": set()}
    for m in mods:
        opm = "OuterProduct" in type(m).__name__
        d = getattr(m, "__dict__", {})
        for attr in ("_kernel_backend", "kernel_backend"):
            if attr in d:
                found["kernel_backend"].add(repr(d[attr]))
        for attr in ("_chunk_size", "chunk_size", "chunk"):
            if attr in d:
                found["opm_chunk" if opm else "chunk"].add(repr(d[attr]))
    return {k: ("|".join(sorted(v)) if v else "unread") for k, v in found.items()}


def settings_line(backend: str, calls: list, state: dict, attn_state: dict = None) -> str:
    """The one SETTINGS line of the stock route: the backend switch, the model calls made (``none`` under ``shipped``), the
    model's own switches as loaded/after the calls (:func:`as_loaded_state`) and, with ``attn_state`` (attn.state(model) — upstream's switches and
    the forwards / classes bound on THIS model), the fast-environment words
    ``atom_attn=flash_attn|sdpa atom_forward=stockxN esmc_mlp=te|torch esmc_attn=… esmc_rope=… flash_attn=<v|absent> transformer_engine=… xformers=…``."""
    made = " ".join(f"{n}({v!r})" for n, v in calls) if calls else "none"
    line = (f"SETTINGS backend={backend or 'shipped'} model_calls={made} "
            f"kernel_backend={state['kernel_backend']} chunk={state['chunk']} opm_chunk={state['opm_chunk']}")
    if attn_state is not None:
        from .attn import words as _attn_words                                  # the one formatter of the attention words (every route prints the same words)
        line += " " + _attn_words(attn_state, forward=True)
    return line


def load_model(hf_repo: str, device: str = "cuda", settings=None, det_level: int = 0, backend: str = None):
    """Upstream's model load at its defaults (``load_esmc=True, esmc_precision='bf16'``), moved to `device`, eval mode; with
    `settings`, the two model calls follow (``set_kernel_backend``, ``set_chunk_size`` — the stock arm; under a kit mode the
    kit's own ``configure()`` makes those calls instead; ``backend`` overrides them). ``det_level`` 2 sets the level-2 recipe's
    config-level switch on the model config BEFORE the weights load (det.apply_config)."""
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    from . import det as _det
    kw = {}
    if _det.level(det_level) >= 2:
        from transformers.models.esmfold2.configuration_esmfold2 import ESMFold2Config
        config = ESMFold2Config.from_pretrained(hf_repo)
        _det.apply_config(config, det_level)
        kw["config"] = config
    model = ESMFold2Model.from_pretrained(hf_repo, **kw).to(device).eval()
    for name, value in model_calls(settings, backend):
        getattr(model, name)(value)
    return model


class Outcome:
    """The pass record of a prediction loop: the complexes (input ids, in input order) whose every seed folded and was written or
    staged, the fold in flight, the failure if one, the prediction count. With ``path`` it is written after every change — the stock
    subprocess keeps it under its stage directory (outputs.PROGRESS_NAME) and ``pred`` reads it once the subprocess has exited, so a
    pass that stopped on its third complex still names the two it completed and the one it failed on."""

    def __init__(self, items, path: str = None):
        self.n_items = len(items)
        self.items_complete: list = []
        self.current: dict = None
        self.item_failed: dict = None
        self.n_predictions = 0
        self.path = path
        self.save()

    def start(self, item_id: str, seed=None) -> None:
        self.current = {"id": item_id, "seed": None if seed is None else int(seed)}
        self.save()

    def fold_done(self, k: int) -> None:
        self.n_predictions += int(k)
        self.save()

    def item_done(self, item_id: str) -> None:
        self.items_complete.append(item_id)
        self.current = None
        self.save()

    def fail(self, exc: BaseException) -> None:
        self.item_failed = dict(self.current or {"id": None, "seed": None}, error=f"{type(exc).__name__}: {exc}"[:1000])
        self.save()

    def as_dict(self) -> dict:
        return {"n_items": self.n_items, "items_complete": list(self.items_complete), "current": self.current, "item_failed": self.item_failed,
                "n_predictions": self.n_predictions}

    def save(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.as_dict(), fh)
        os.replace(tmp, self.path)


def read_outcome(path: str):
    """The pass record a stock subprocess left (``Outcome.save``), or None when it never reached its prediction loop."""
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def pass_summary(outcome, exit_code: int) -> dict:
    """The pass fields from a pass record (``Outcome.as_dict()``, the subprocess's file, or None when the loop never ran):
    ``pass_status`` complete | incomplete | failed, ``n_items``, ``n_items_complete``, ``items_complete``, ``item_failed``
    (``{id, seed, error}`` — the recorded failure, or the fold in flight when the process ended without recording one); ``incomplete``
    = some complexes complete, the outputs short of the request (exit 1; ``incomplete <n>/<m>`` on the INCOMPLETE line) — a different
    condition from a partial activation (``partial``: a lever of the mode not applied)."""
    if not outcome:
        return {"pass_status": "complete" if exit_code == 0 else "failed", "n_items": None, "n_items_complete": 0, "items_complete": [], "item_failed": None}
    complete = list(outcome.get("items_complete") or [])
    failed = outcome.get("item_failed")
    n_items = outcome.get("n_items")
    if failed is None and (exit_code != 0 or (n_items is not None and len(complete) < n_items)):
        cur = outcome.get("current") or {"id": None, "seed": None}
        failed = dict(cur, error=f"the process ended (exit code {exit_code}) during this fold" if cur.get("id") else f"the process ended (exit code {exit_code}) before its next fold")
    if failed is None and exit_code == 0:
        status = "complete"
    else:
        status = "incomplete" if complete else "failed"
    return {"pass_status": status, "n_items": n_items, "n_items_complete": len(complete), "items_complete": complete, "item_failed": failed}


def incomplete_of(summary: dict) -> Optional[str]:
    """The verdict's ``incomplete`` word from a pass summary: "<n>/<m>" when the pass is not complete, else None."""
    if summary.get("pass_status") == "complete":
        return None
    return f"{summary.get('n_items_complete') or 0}/{summary.get('n_items') if summary.get('n_items') is not None else '?'}"


def pass_line(summary: dict) -> str:
    """One line naming an incomplete or failed pass (``pass_summary``)."""
    f = summary.get("item_failed") or {}
    where = f"{f.get('id')}" + (f" s{f['seed']}" if f.get("seed") is not None else "") if f.get("id") else "before the first fold"
    return (f"pred {summary['pass_status'].upper()}: {summary['n_items_complete']}/{summary.get('n_items')} complexes complete; "
            f"failed on {where}: {f.get('error')}")


def empty_cuda_cache() -> bool:
    """Return the caching allocator's unused blocks to the device (``torch.cuda.empty_cache``) when torch is loaded and CUDA is
    available; returns whether it ran."""
    torch = sys.modules.get("torch")
    cuda = getattr(torch, "cuda", None) if torch is not None else None
    if cuda is None or not cuda.is_available():
        return False
    cuda.empty_cache()
    return True


ITEM_WALL = ("builder.fold() alone: upstream's fold() call per (input, seed), which featurizes once inside itself; the package's own "
             "prepare_input (the token features the kit's rows need) runs before the window")   # the wall of the `fold <id> s<seed> samples=<k> <wall>s` line


def fold_items(model, builder, items, settings, variant: str, out_dir: str, seeds=None, det: int = 0, log=print, writer=None,
               stage_dir: str = None, device: str = "cuda", outcome: Outcome = None) -> int:
    """The prediction loop shared by every mode: upstream's ``builder.fold`` per (input, seed) at the settings passed in. The results go
    either to the kit's writer (``writer(item, seed, res, depths, feats, wall)`` — a kit mode in the pred process, outputs.write_rows)
    or to the stage directory (the stock subprocess, outputs.stage_result). After each fold is written or staged, its result and
    features are released and the CUDA caching allocator is emptied, so every fold starts from the allocator state of the first one
    (the previous fold's blocks are neither held nor left cached). On a kit route the captured-graph generation is reused across inputs of
    one shape signature and released before an input of another (report.graphgen_decide, one GRAPHGEN line per input). ``outcome`` (Outcome)
    records the pass as it goes. Returns the number of predictions."""
    from esmfold2_opt import inputs, outputs, peakmem, phases
    use_msa = variant == "full_msa"
    kw = settings.fold_kwargs(det=int(det))
    gpu_name = _gpu_name()
    log(phases.boundary_line(phases.install(model)))                       # the per-fold PHASE line's four boundaries on THIS model: upstream's callables, or a kit mode's replacements of them
    graphgen = report.graphgen_install(model)                              # the captured-graph generation decision (reuse | release-then-rebuild, one GRAPHGEN line per input) runs as a forward
                                                                           # pre-hook: after fold() featurised the input, before the network allocates; silent on the stock arm (no kit module)
    n = 0
    try:
        for item in items:
            if outcome is not None:
                outcome.start(item["id"])
            spi = inputs.build_spi(item, use_msa, settings.msa_read_depth, settings.msa_remove_insertions)
            depths = inputs.msa_depths(spi)
            nds = inputs.item_samples(item, kw["num_diffusion_samples"])            # the run's count (inputs.run_samples settled it before the first fold; an input's own key equals it)
            item_seeds = seeds or (inputs.item_seeds(item, settings.seeds) if (settings.seeds or item.get("seeds")) else [None])   # no seed anywhere: upstream's default, one unseeded fold (seed=None)
            if report.cache_scope() == "input":                                 # the default scope: the kit's content-keyed feature / ESM-C caches hold the CURRENT input only — the entries of
                                                                                    # earlier inputs (CUDA feature sets, ESM-C hidden states) are released here; ESMFOLD2_OPT_CACHE_SCOPE=global keeps them
                report.cache_clear()
                report.graphgen_arm(item["id"])                                     # and the kit's captured-graph generation is decided at this input's first model call (graphgen hook, below):
                                                                                    # reused when the input's shape signature is the live generation's, else released BEFORE the network allocates
            for seed in item_seeds:
                seed = None if seed is None else int(seed)                          # upstream's default (unseeded) or the integer seed
                if outcome is not None:
                    outcome.start(item["id"], seed)
                c0 = report.cache_counts()                                          # the kit caches' counters before the window (None on the stock arm: no kit module loaded)
                l0 = report.leverfold_counts()                                      # the lever modules' data-dependent engagement counters before the window (None on the stock arm)
                for note in phases.begin():                                         # the phase accumulators zeroed (outside the window)
                    log(note)
                peakmem.begin()                                                     # the peak-memory counters reset (outside the window)
                t1 = time.perf_counter()
                # --- the timed window (ITEM_WALL): upstream's fold() alone ---------------------------------------------------------
                res = builder.fold(model, spi, seed=seed, complex_id=item["id"], **dict(kw, num_diffusion_samples=nds))
                # ---------------------------------------------------------------------------------------------------------------------
                wall = time.perf_counter() - t1
                pk_alloc, pk_reserved = peakmem.peak()                              # the window's peak, read before anything after the fold allocates
                c1 = report.cache_counts()
                sys.stderr.write(report.cache_line(item["id"], seed, c0, c1) + "\n"); sys.stderr.flush()   # the window's feature/ESM-C cache traffic, one line per fold
                lf = report.leverfold_line(item["id"], seed, l0, report.leverfold_counts())            # the window's data-dependent lever engagements (mh, rg, the roll-out, atom layouts, fall-throughs)
                if lf:
                    sys.stderr.write(lf + "\n"); sys.stderr.flush()
                feats, aux = builder.prepare_input(spi, seed=0, device=getattr(model, "device", device))   # the token features the rows need (seed=0 as the kit passes it) — AFTER
                # the timed window on every route: inside the window each arm pays its own featurisation exactly as upstream's fold() does (fold() prepares its input itself);
                # on a kit arm this call is a feature-cache hit outside the window, counted outside the CACHE line's deltas
                if writer is not None:
                    rows = writer(item, seed, res, depths, feats, wall)
                    k = len(rows)
                else:
                    k = len(outputs.stage_result(stage_dir, item, variant, seed, res, depths, feats, wall, gpu_name))
                n += k
                if outcome is not None:
                    outcome.fold_done(k)
                log(f"fold {item['id']} s{seed} samples={k} {wall:.2f}s")
                phase_line, phase_notes = phases.line(item["id"], seed, wall)  # lm / trunk / sampler / conf inside the same window; total_s = the window's wall
                log(phase_line)
                for note in phase_notes:
                    log(note)
                del res, feats, aux                                        # nothing of this fold is held across the next one
                empty_cuda_cache()
                for pl in peakmem.lines(item["id"], seed, pk_alloc, pk_reserved, peakmem.resident()):   # PEAK (the window's peak) + PEAK-NOTE (seed, what the process carries into the next fold)
                    log(pl)
            if outcome is not None:
                outcome.item_done(item["id"])
    except BaseException as e:
        if outcome is not None:
            outcome.fail(e)
        raise
    finally:
        if graphgen is not None:
            graphgen.remove()
    return n


def _gpu_name() -> str:
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "unknown"


def run(a) -> int:
    t0 = time.perf_counter()
    env_absent = tuple(x for x in a.env_absent.split(",") if x)
    proof = env_proof(env_absent)                                          # before torch: the inherited environment is the one proved
    print(f"{PREFIX} ENV-CLEAN ok: absent={','.join(env_absent)} kit_modules=none kit_dirs=none no_user_site={proof['no_user_site']}", file=sys.stderr, flush=True)
    from esmfold2_opt import det as _det, inputs, outputs, settings as _settings
    st = _settings.from_json(a.settings_json)
    lv = _det.level(a.det)
    if lv:
        _det.apply_env()                                                   # before torch is imported
    import torch  # noqa: F401  (the recipe's torch switches next; level 2's config override is load_model's)
    if lv:
        _det.apply_torch()
    from esmfold2_opt import attn as _attn                                 # the fast-environment reader (upstream's own module switches; nothing of the kit)

    def refuse(why, attn_state):                                           # ESMFOLD2_OPT_REQUIRE_FAST_ENV=1 and an accelerated path not live: refused by name (exit 3), never a slow run under the switch
        print(report.not_active_line(why), file=sys.stderr, flush=True)
        return report.EXIT_NOT_ACTIVE

    pre = _attn.state(load=True)                                           # imports upstream's two modeling modules (load_model imports them next anyway): the switches, before any weights load
    why = _attn.require_refusal(pre)
    if why:
        return refuse(why, pre)
    # --- the stock call: upstream API only (the folds themselves: the marked block in fold_items) ---------------------------------
    from esm.models.esmfold2.processor import ESMFold2InputBuilder
    model = load_model(a.hf_repo, a.device, st, det_level=lv, backend=a.backend)   # upstream's load + the two model calls
    builder = ESMFold2InputBuilder()
    # -------------------------------------------------------------------------------------------------------------------------------
    t_load = time.perf_counter() - t0
    print(report.ready_line(a.variant, t_load), file=sys.stderr, flush=True)   # the one 'model loaded' line (cold start ends here)
    state = as_loaded_state(model)
    attn_state = _attn.state(model)                                        # read again ON THE MODEL: the flag × the forward bound on each SWA3DRoPEAttention, the ESMC classes bound in the LM
    print(f"{PREFIX} {settings_line(a.backend, model_calls(st, a.backend), state, attn_state)}", file=sys.stderr, flush=True)
    why = _attn.require_refusal(attn_state)                                # the bound reading disagrees with the switches (never expected): refused all the same
    if why:
        return refuse(why, attn_state)
    items = inputs.load(a.input)
    seeds_cli = [int(s) for s in a.seeds.split(",")] if a.seeds else None
    stage_dir = a.stage_dir or os.path.join(a.out_dir, outputs.STAGE_DIRNAME)
    outcome = Outcome(items, path=os.path.join(stage_dir, outputs.PROGRESS_NAME))
    n, rc = 0, 0
    try:
        n = fold_items(model, builder, items, st, a.variant, a.out_dir, seeds=seeds_cli, det=lv, stage_dir=stage_dir, device=a.device,
                       log=lambda s: print(f"{PREFIX} {s}", file=sys.stderr, flush=True), outcome=outcome)
    except BaseException:
        rc = 1
        raise
    finally:
        summary = pass_summary(outcome.as_dict(), rc)
        if summary["pass_status"] == "complete":
            print(f"{PREFIX} DONE predictions={n} items={len(items)} out_dir={a.out_dir}", file=sys.stderr, flush=True)
        else:
            print(f"{PREFIX} {pass_line(summary)}", file=sys.stderr, flush=True)
    return 0


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
