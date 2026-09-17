"""The model-process adapter of the row-sharded pair stack: ``--mode big --n_gpu P`` with P > 1 (loaded by file path by the memory mode's
launcher, like inprocess/templates.py; standard library at module level — this interpreter is the fork's, with the shared core importable through
the launcher's core path).

What it does, and only that (the patched module bodies are the shared core's ONE AlphaFold-3 Haiku recipe,
``opt_core.mem.rowpair_jax.alphafold3``; this file writes none):

  1. the mesh — ``opt_core.mem.rowpair_jax.mesh.build(P, platform='gpu')``: the first P visible GPUs, refusals by name (``refused: n_gpu=P
     visible=K``, ``platform=<found> expected=gpu`` when the CUDA plugin did not initialise, a non-local device); the live-pool headroom
     ESTIMATE for NCCL (``mesh.mem_fraction_gate`` with :data:`MEM_FRACTION_CEILINGS`) is a NOTE, never a refusal: over its ceiling the adapter
     prints ``ROWPAIR NOTE mem_headroom: <the estimate's words>; proceeding — may exhaust device memory`` and installs (:func:`headroom`);
  2. the recipe — ``alphafold3.install(rmesh, PatchSet, heads=, conf=, b21=<the kit's pair-conditioning transcription module>, schedule=)``: the
     trunk / model / heads bodies rebound on row blocks, pinned to the alphafold3 sources they transcribe (the recipe refuses other bytes by name);
  3. the driver — the fork's runner script is loaded as a module (its ``ModelRunner`` keeps one device: ``jax.jit(apply, device=)``), its
     ``ModelRunner._jitted_apply`` is rebound to ``haiku.jit_apply(apply, rmesh)`` (replicated in/out shardings; the pair is sharded inside the
     program) and ``run_inference`` places the batch with ``shard.put`` (replicated) instead of ``device_put(device)``; then the script's own
     ``main`` runs under absl with the caller's flags — every other byte of the runner (buckets, seeds, outputs, embeddings, the compilation cache
     flag) is the script's;
  4. the evidence — ``[af3-jax-opt] ROWPAIR installed n_gpu=P sites=<n> heads=<sharded|replicated> conf=<...> diffusion=<b21|absent> schedule=<...>
     library=<sha8>`` at install and, at exit, the family's ONE lever line (``evidence.line``: ``LEVER name=rowpair state=on ... n_gpu=P
     sharding=rowpair devices=... xla_peak_gb_max=... schedule=... kernel=... sites=...``) — or ``ROWPAIR refused: <reason>`` and exit
     :data:`EXIT_ROWPAIR_REFUSED` before anything of the model ran.

P = 1 never reaches this module (the launcher does not load it): the single-device program is byte-identical to the kit without it.
"""
import atexit
import importlib.util
import os
import sys
import time

PREFIX = "[af3-jax-opt]"
TAG = "af3-jax-opt"
LEVER_ID = "ROWPAIR"                                                      # the kit's lever id (modes.ROWPAIR; registry strategy F7.tensor_parallel)
EXIT_ROWPAIR_REFUSED = 5                                                  # the model process's exit code when the mesh / recipe refuses by name (the wrapper names it: rc != 0 → DONE FAILED)
MEM_FRACTION_CEILINGS = {8: 0.90}                                         # an ESTIMATE, printed as a NOTE when exceeded (headroom()), never a refusal — client-memory-fraction ceilings by device count under which NCCL keeps headroom (mesh.mem_fraction_gate refuses above; headroom() words that as the NOTE)
SCRIPT_MODULE = "af3_jax_runner_script"                                   # the name the runner script is loaded under (its `if __name__ == '__main__'` stays false: main runs through absl here)
_STATE = {"installed": False, "rmesh": None, "record": None, "gate": None, "t_install": None, "exit_line_done": False, "schedule": None}
KERNEL_OPS = ("triatt", "trimul")                                           # the two pair ops whose per-shard kernel the exit line names (GridSelfAttention / TriangleMultiplication)
KERNELS = {op: {"kernel": None, "reason": None, "served": 0, "fallback": 0, "stock": 0} for op in KERNEL_OPS}   # the per-shard kernel census (count_call), read by kernel_evidence at exit


def kernel_word(op: str, module) -> str:
    """The per-shard kernel the recipe's sharded body runs for ``op`` on this module instance, in the model's own terms: tri-attention =
    the model's attention core (``global_config.flash_attention_implementation``: triton = the upstream tokamax flash kernel, cudnn, xla)
    behind the recipe's LayerNorm / bias all-gather; triangle multiplication = the upstream tokamax gated-linear-unit kernel (``config.use_glu_kernel``)
    or XLA projections, then the row-sharded XLA einsum (``schedule=`` on the same line)."""
    if op == "triatt":
        return f"af3_attention:{getattr(getattr(module, 'global_config', None), 'flash_attention_implementation', None) or 'unknown'}"
    if op == "trimul":
        return "tokamax_glu+xla_einsum" if getattr(getattr(module, "config", None), "use_glu_kernel", False) else "xla_einsum"
    raise ValueError(f"kernel_word: op must be one of {KERNEL_OPS}, not {op!r}")


def count_call(op: str, module, act, sharded: bool, census=None) -> None:
    """Book ONE traced call of ``op``: inside the sharded region the per-shard kernel decision for the local block ``act`` = ``[n_loc, N, C]``
    (opt_core.mem.rowpair_jax.triatt.kernel_gate — no per-shard fused kernel of this kit declares block constraints, so the gate answers
    ``unconstrained`` and the call is ``served``; a refused block would be ``fallback``, named by the gate's word); outside it the stock body ran
    on the full pair by the recipe's design (``stock``)."""
    c = (census if census is not None else KERNELS)[op]
    if not sharded:
        c["stock"] += 1
        return
    from opt_core.mem.rowpair_jax import triatt as _ta
    ok, why = _ta.kernel_gate(int(act.shape[0]), int(act.shape[1]), None)
    c["kernel"], c["reason"] = kernel_word(op, module), why
    c["served" if ok else "fallback"] += 1


def install_kernel_census(modules, in_sharded_region) -> list:
    """Count the recipe's two pair ops at trace time: a plain outer shim over the (already Haiku-wrapped) bodies the recipe bound on
    ``modules.GridSelfAttention.__call__`` / ``modules.TriangleMultiplication.__call__`` — numerics untouched, one census entry per traced call
    (count_call). Returns the class names shimmed."""
    done = []
    for op, cls_name in (("triatt", "GridSelfAttention"), ("trimul", "TriangleMultiplication")):
        cls = getattr(modules, cls_name, None)
        inner = getattr(cls, "__call__", None) if cls is not None else None
        if inner is None or getattr(inner, "_rowpair_census", False):
            continue

        def shim(self, act, *args, __inner=inner, __op=op, **kwargs):
            count_call(__op, self, act, bool(in_sharded_region()))
            return __inner(self, act, *args, **kwargs)
        shim._rowpair_census = True
        shim.__wrapped__ = inner
        cls.__call__ = shim
        done.append(cls_name)
    return done


def kernel_evidence(census=None) -> dict:
    """The exit line's per-shard kernel fields from the census: ``kernel=triatt:<word>,trimul:<word> kernel_reason=<the gate's word |
    per_shard_fallback:<op>=<n>,…>`` (opt_core.mem.rowpair_jax.triatt.kernel_fields) plus, per op, ``<op>_served= <op>_fallback= <op>_stock=``;
    ``kernel=unrecorded kernel_reason=no_trace`` when no pair op was traced in this process (every bucket came from a serialized executable)."""
    from opt_core.mem.rowpair_jax import triatt as _ta
    k = census if census is not None else KERNELS
    traced = [op for op in KERNEL_OPS if k[op]["served"] or k[op]["fallback"]]
    if not traced:
        fields = {"kernel": "unrecorded", "kernel_reason": "no_trace"}
    else:
        name = ",".join(f"{op}:{k[op]['kernel'] or 'untraced'}" for op in KERNEL_OPS)
        short = [f"{op}={k[op]['fallback']}" for op in KERNEL_OPS if k[op]["fallback"]]
        reasons = sorted({str(k[op]["reason"]) for op in traced if k[op]["reason"]})
        fields = dict(_ta.kernel_fields(name, True, "per_shard_fallback:" + ",".join(short) if short else ("+".join(reasons) or "unconstrained")))
    for op in KERNEL_OPS:
        fields.update({f"{op}_served": k[op]["served"], f"{op}_fallback": k[op]["fallback"], f"{op}_stock": k[op]["stock"]})
    return fields


def _refuse(reason: str) -> None:
    sys.stdout.write(f"{PREFIX} ROWPAIR refused: {str(reason).strip()}\n"); sys.stdout.flush()
    raise SystemExit(EXIT_ROWPAIR_REFUSED)


BUCKET_SITE = "alphafold3.model.pipeline.pipeline.calculate_bucket_size"   # the runner's one padding decision (pipeline.py: WholePdbPipeline.process_item calls it by module name)


def padded_bucket_size(stock_fn, n_gpu: int, on_pad=None):
    """``calculate_bucket_size`` held to the shared recipe's size contract: the stock bucket for the input, rounded UP to the next multiple of
    ``n_gpu`` (``opt_core.mem.rowpair_jax.shard.pad_plan``: every N has a plan, a ladder bin is never refused). The fork's own buckets (32 … 5120)
    are multiples of every P this tree serves, so a bucketed input keeps its stock bucket byte for byte; an input above the largest bucket, which
    the fork pads to exactly its token count, gains at most ``n_gpu - 1`` padding tokens — the runner then pads features AND masks to that length
    itself. ``on_pad(num_tokens, stock_bucket, padded)`` is called when the two differ (the model process prints the ROWPAIR pad line)."""
    from opt_core.mem.rowpair_jax import shard as _shard
    P = int(n_gpu)

    def calculate_bucket_size(num_tokens, buckets):
        stock = int(stock_fn(num_tokens, buckets))
        padded = int(_shard.pad_plan(stock, P)["n_padded"])
        if padded != stock and on_pad is not None:
            on_pad(int(num_tokens), stock, padded)
        return padded

    calculate_bucket_size.__wrapped__ = stock_fn
    calculate_bucket_size.n_gpu = P
    return calculate_bucket_size


def install_bucket_guard(n_gpu: int) -> str:
    """Wrap the runner's bucket decision (:data:`BUCKET_SITE`) with :func:`padded_bucket_size` in this process; idempotent. Prints
    ``ROWPAIR bucket guard=installed site=... n_gpu=P`` once and ``ROWPAIR pad tokens=<n> stock_bucket=<b> padded=<b'> n_gpu=P`` per padded input."""
    import importlib
    modname, attr = BUCKET_SITE.rsplit(".", 1)
    try:
        mod = importlib.import_module(modname)
    except ImportError as e:
        _refuse(f"bucket guard: {modname} not importable ({type(e).__name__}) — the padded size cannot be held to a multiple of n_gpu={n_gpu}")
    cur = getattr(mod, attr, None)
    if cur is None:
        _refuse(f"bucket guard: {BUCKET_SITE} is absent (the runner's padding decision moved)")
    if getattr(cur, "n_gpu", None) == int(n_gpu):
        return "already"
    stock_fn = getattr(cur, "__wrapped__", cur)
    on_pad = lambda n, b, padded: print(f"{PREFIX} ROWPAIR pad tokens={n} stock_bucket={b} padded={padded} n_gpu={int(n_gpu)}", flush=True)
    setattr(mod, attr, padded_bucket_size(stock_fn, n_gpu, on_pad))
    print(f"{PREFIX} ROWPAIR bucket guard=installed site={BUCKET_SITE} n_gpu={int(n_gpu)}", flush=True)
    return "installed"


def headroom(rp_mesh, rmesh, refused_cls) -> dict:
    """The NCCL-headroom ESTIMATE (``rp_mesh.mem_fraction_gate`` with :data:`MEM_FRACTION_CEILINGS`): within its ceiling (or no ceiling for this
    device count) the core's record ``{"mem_fraction_ceiling", "xla_pool_fraction"}`` as is — the install line's two tokens, unchanged; over it
    the estimate is a NOTE decided up front, never a refusal: ``[af3-jax-opt] ROWPAIR NOTE mem_headroom: <the estimate's words>; proceeding — may exhaust device memory``
    is printed once and the record carries the same two tokens plus ``note``. The run then proceeds exactly as within the ceiling; an
    out-of-memory later is the run's own failure (rc 1), named by the wrapper's exit rule."""
    try:
        return dict(rp_mesh.mem_fraction_gate(rmesh, MEM_FRACTION_CEILINGS))
    except refused_cls as e:
        words = str(getattr(e, "reason", None) or e).strip()
        words = words[len("refused:"):].strip() if words.startswith("refused:") else words
        n = int(getattr(rmesh, "n_gpu", 0) or 0)
        keys = sorted(int(k) for k in MEM_FRACTION_CEILINGS if int(k) <= n)
        try:
            frac = rmesh.pool_fraction()
        except Exception:  # noqa: BLE001 — the live pool unreadable: the estimate's words already say so
            frac = None
        rec = {"mem_fraction_ceiling": float(MEM_FRACTION_CEILINGS[keys[-1]]) if keys else "none",
               "xla_pool_fraction": "unavailable" if frac is None else float(f"{frac:.4f}"), "note": words}
        sys.stdout.write(f"{PREFIX} ROWPAIR NOTE mem_headroom: {words}; proceeding — may exhaust device memory\n"); sys.stdout.flush()
        return rec


def install(n_gpu: int, b21=None, schedule: str = "ring", heads: str = "sharded", conf: str = "sharded"):   # schedule: the triangle multiplication's collective plan (opt_core.mem.rowpair_jax.trimul.SCHEDULES: ring | gather)
    """Build the mesh over ``n_gpu`` GPUs and install the shared recipe; print the install line; register the exit line. Returns the recipe's record.
    Refusals (mesh, headroom, pin, hazards) print ``ROWPAIR refused: <reason>`` and exit :data:`EXIT_ROWPAIR_REFUSED`."""
    if _STATE["installed"]:
        return _STATE["record"]
    from opt_core.mem import ngpu
    from opt_core.mem import MemLeverRefused
    try:
        from opt_core.mem.rowpair_jax import alphafold3 as recipe, haiku as rp_hk, mesh as rp_mesh
    except ImportError as e:                                              # a core without the row-sharded pair stack: refused by name, never a traceback
        _refuse(f"producer_missing:{getattr(e, 'name', None) or 'opt_core.mem.rowpair_jax'} — n_gpu > 1 imports opt_core >= 0.4.1 (opt_core.mem.rowpair_jax)")
    P = ngpu.check_n_gpu(n_gpu)
    if P == 1:
        _refuse("n_gpu=1 (the single-device program installs nothing; the launcher never loads this module at P=1)")
    try:
        rmesh = rp_mesh.build(P, platform="gpu")                          # genuine mesh errors (fewer than P visible devices, no GPU platform, a non-local device): refused by name
    except MemLeverRefused as e:
        _refuse(getattr(e, "reason", None) or str(e))
    except (ValueError, RuntimeError) as e:
        _refuse(f"{type(e).__name__}:{str(e).replace(' ', '_')[:300]}")
    gate = headroom(rp_mesh, rmesh, MemLeverRefused)                     # the NCCL-headroom ESTIMATE on the live pool: within → silent; over → ONE NOTE line and the install proceeds (it may exhaust device memory: the user's)
    try:
        patches = rp_hk.PatchSet("rowpair")
        record = recipe.install(rmesh, patches, heads=heads, conf=conf, b21=b21, schedule=schedule)
    except MemLeverRefused as e:                                            # the recipe's own refusals (library bytes not the pinned ones, hazards): the mode cannot meet its statement — refused by name
        _refuse(getattr(e, "reason", None) or str(e))
    except (ValueError, RuntimeError) as e:                                # a usage/plan error of the recipe: named, never a stock run on one device under --n_gpu P
        _refuse(f"{type(e).__name__}:{str(e).replace(' ', '_')[:300]}")
    install_bucket_guard(P)                                              # every N has a plan: the runner's padded length is a multiple of P (shard.pad_plan), never a refusal at the region
    if "TriangleMultiplication.__call__" in (record.get("sites") or []):      # the trunk's pair ops are sharded: count their per-shard kernel per traced call (the exit line names it)
        install_kernel_census(recipe.library()["modules"], recipe.in_sharded_region)
    _STATE.update(installed=True, rmesh=rmesh, record=record, gate=gate, t_install=time.time(), schedule=schedule)
    fields = dict(recipe.install_fields(record))                          # heads= conf= diffusion= n_sites= library= — the recipe's own words (the family line at exit carries the same)
    words = " ".join(f"{k}={str(v).replace(' ', '_')}" for k, v in fields.items())
    print(f"{PREFIX} ROWPAIR installed n_gpu={P} sites={fields.get('n_sites')} {words} schedule={record.get('schedule', schedule)} "
          f"mem_fraction_ceiling={gate.get('mem_fraction_ceiling')} xla_pool_fraction={gate.get('xla_pool_fraction')}", flush=True)
    atexit.register(exit_line)
    return record


def exit_line() -> str:
    """The family's ONE lever line for this process (evidence.line, state=on): mesh facts, XLA env as found, per-device XLA peaks, schedule, per-shard
    kernel, sites. Printed once (atexit or by the launcher)."""
    if _STATE["exit_line_done"] or not _STATE["installed"]:
        return ""
    from opt_core.mem.rowpair_jax import evidence as ev
    rmesh, record = _STATE["rmesh"], _STATE["record"]
    try:
        peaks = ev.device_peaks(rmesh)
    except Exception as e:  # noqa: BLE001 - a backend without stats: the line says unavailable
        from opt_core.oom import is_oom
        if is_oom(e): raise                                               # OOM propagates: no fallback applied (opt_core.oom.is_oom)
        peaks = None
    try:
        kernel = kernel_evidence()                                        # kernel=triatt:<word>,trimul:<word> kernel_reason=… + the per-op served/fallback/stock census
    except Exception as e:  # noqa: BLE001
        from opt_core.oom import is_oom
        if is_oom(e): raise                                               # OOM propagates: no fallback applied (opt_core.oom.is_oom)
        kernel = {"kernel": "unrecorded", "kernel_reason": f"kernel_evidence_error:{type(e).__name__}"}
    _word = lambda v: str(v).replace(" ", "_")                               # lever-line values carry no blanks (opt_core.report.lever_line refuses them)
    sites = [_word(x) for x in (record.get("sites") or [])]
    extra = {k: v for k, v in record.items() if k in ("heads", "conf", "diffusion", "hazards") and v is not None}
    if isinstance(extra.get("hazards"), (list, tuple)):
        extra["hazards"] = ",".join(_word(h) for h in extra["hazards"]) or "none"
    extra = {k: _word(v) for k, v in extra.items()}
    try:
        census = {k: v for k, v in kernel.items() if k not in ("kernel", "kernel_reason")}   # the per-op served/fallback/stock counts: the line's LAST fields
        ln = ev.line(TAG, "on", rmesh.n_gpu, rmesh=rmesh, peaks=peaks, schedule=_STATE["schedule"],
                     sites=sites if sites else "unrecorded", kernel=kernel["kernel"], kernel_reason=kernel["kernel_reason"], **extra, **(_STATE["gate"] or {}), **census)
    except (TypeError, ValueError) as e:                                   # the evidence contract refused the record (a required key absent): named, not swallowed
        ln = f"{PREFIX} LEVER name=rowpair state=skipped reason=evidence_error:{type(e).__name__}:{str(e).replace(' ', '_')[:200]} n_gpu={rmesh.n_gpu} sharding=rowpair"   # never state=on without the family's evidence: the wrapper's exit rule reads this as levers_short
    print(ln, flush=True)
    _STATE["exit_line_done"] = True
    return ln


def patch_runner(mod) -> None:
    """Rebind the runner script's ``ModelRunner`` for the mesh: ``_jitted_apply`` = ``haiku.jit_apply(forward.apply, rmesh)`` (replicated in/out;
    ``--nojit`` keeps its meaning), ``run_inference`` places the featurised batch replicated on the mesh (``shard.put``) — the body is the script's own
    with ``jax.device_put(..., self._device)`` replaced. Refuses by name when the script lacks the two members (a runner this adapter does not know)."""
    import functools
    from opt_core.mem.rowpair_jax import haiku as rp_hk, shard
    rmesh = _STATE["rmesh"]
    MR = getattr(mod, "ModelRunner", None)
    if MR is None or not hasattr(MR, "_jitted_apply") or not hasattr(MR, "run_inference"):
        _refuse(f"runner script {getattr(mod, '__file__', '?')} has no ModelRunner._jitted_apply/run_inference to bind to the mesh")
    hk, model = mod.hk, mod.model

    def _jitted_apply(self):
        @hk.transform
        def forward_fn(batch):
            return model.Model(self._model_config)(batch)
        apply_fn = forward_fn.apply
        if not mod._NOJIT.value:
            apply_fn = rp_hk.jit_apply(apply_fn, rmesh)
        return apply_fn

    MR._jitted_apply = functools.cached_property(_jitted_apply)
    MR._jitted_apply.__set_name__(MR, "_jitted_apply")
    stock_run_inference = MR.run_inference
    jax = mod.jax

    def run_inference(self, featurised_example, rng_key):
        _orig = jax.device_put
        _target = shard.named(rmesh, shard.replicated_spec())              # the mesh, replicated (the regions shard the pair inside the program)
        def _put(x, device=None, *a, **k):                                # the script's one placement call, redirected; the ORIGINAL device_put underneath (shard.put would re-enter this patch)
            return _orig(x, _target)
        jax.device_put = _put
        try:
            return stock_run_inference(self, featurised_example, rng_key)
        finally:
            jax.device_put = _orig

    MR.run_inference = run_inference
    print(f"{PREFIX} ROWPAIR runner bound: ModelRunner._jitted_apply=jit_apply(mesh) run_inference=shard.put(replicated) script={os.path.basename(getattr(mod, '__file__', '?'))}", flush=True)


def run_script(script: str, flags, n_gpu: int, b21=None, **install_kwargs) -> None:
    """The memory mode's model process under n_gpu > 1: install (mesh + recipe), load the runner script as a module, bind its runner to the mesh, run
    its ``main`` under absl with ``[script] + flags``. Never returns normally (absl exits the process; the exit line prints through atexit)."""
    install(n_gpu, b21=b21, **install_kwargs)
    spec = importlib.util.spec_from_file_location(SCRIPT_MODULE, script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[SCRIPT_MODULE] = mod
    sys.argv = [script] + list(flags)
    spec.loader.exec_module(mod)                                          # defines flags, ModelRunner, main; its __main__ guard stays shut
    patch_runner(mod)
    from absl import app
    app.run(mod.main)


def report() -> dict:
    """``{'installed', 'n_gpu', 'record', 'gate'}`` — what this process installed (the launcher's exit lines read it)."""
    rm = _STATE["rmesh"]
    return {"installed": _STATE["installed"], "n_gpu": getattr(rm, "n_gpu", None), "record": _STATE["record"], "gate": _STATE["gate"]}
