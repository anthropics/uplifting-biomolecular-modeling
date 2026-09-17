"""opt_core — the core of the optimization kit: the mechanisms the kit needs that are not specific to the model.

The kit (`<pkg>`) installs this package beside itself and pins it by path + minimum version in its
`pyproject.toml` `[tool.opt_core]`. The core carries no engine knowledge: every engine-specific value (a mode table, a trigger
module, a lever name, a prefix) is data the kit passes in. Modules, by mechanism:

    autoload      the lazy .pth meta-path finder that activates a mode on the first import of a trigger package
    modes         mode tables, env-script transcription, the resolution record, the mode-argument precedence
    gates         GPU probe and class-by-memory, distribution versions, pins, sha256 sums, the core pin gate
    arch          card classes (sm80/sm90/sm100/sm103): the lever-support registry, the box's class and memory readers, the off-by-card words
    jax_arch      the same two readers (class, memory) of a jax device
    home          tree / opt / kit directory resolution and sys.path placement
    warm          warm_imports(): the stack's heavy model libraries imported early and shallow through the large-frame trampoline (_pystack)
    instances     the late-activation instance counter (constructor wrap or self-removing class finder)
    report        exit codes, the exit verdict, the prefix / key=value line primitives, the partial grammar, the exit tally
    strategies    the strategy catalogue (STRATEGIES.json) and its membership check — a development-time table no run path reads
    manifest      opt_manifest.json: build, atomic write, read, the stack and core blocks
    stock_proof   the clean-process proof of a stock run and the stock command contract
    cli           the verb registry and main() of a kit command line
    process       run_logged (wall-clock watchdog over a process group), run_step (one named step of a pass) and child_env
    oom           the one out-of-memory classifier of the core and the kit (standard library only at import)
    upstream_fix  the ``--upstream-fix <ID>`` grammar and registry: confirmed upstream issues a kit fixes on request only
    shape_policy  how many tokens a padding engine compiles for: the stock padding rules by name and the kernels' token multiple
    extern        digests of files the repository does not track (downloaded weights, external archives)
    counters      the per-process call ledger of one lever (served / fallback / error, shapes, the LEVER line, the gate) and census arithmetic
    cell_census   the process-wide cell-coverage census of the kernel provider faces (cell hit | inherited | named fallback | stock; tokens, dump)
    trimul        the triangle-multiplication lever adapter over the carried kernels (decision ladder, weight vocabulary, census)
    trimul_weights  the weight vocabulary of the fused triangle-multiplication providers (the names a kit's weight shim maps its tensors to)
    attn          attention glue: the served-shape size gate, the fused SDPA-with-pair-bias call
    precision     precision policy, cast caches, deterministic scatter, the deterministic recipe applied to torch
    capture       CUDA-graph capture, the graph pool, trajectory-constant hoisting, the XLA compile cache
    mem           the memory mode: the lever registry / record / composition (big), chunked and offloaded pair operations, checkpointing,
                  allocator policy, the peak probe; and the primitives (row chunking, patch sets, pinned host pairs,
                  the graph gate, byte budgets, the n_gpu axis)
    host          host-side levers: process memo, worker pools, pinned output pools, cold-start staging, event records
    host_cache    host-side pipeline primitives: the background output writer, initialisation skipped under a checkpoint load, the resident item loop
    diffusion_loop  diffusion roll-out helpers of a design pipeline (RNG, sync points, guards)
    jax_design    JAX design-engine helpers: persistent compile cache policy, sub-batch policy
    seq           sequence-side helpers: attention backends, resident tables, varlen, numerics and det facades, host env/io, batch plans
    ops           fused operations outside the kernel families (MSA outer-product mean, pair-weighted averaging, msa_fused) with torch fallbacks
    of3_sampler   diffusion-sampler levers of the OF3 code family (AF3-architecture engines): atom windows, hoists, DiT glue, roll-out memo
    of3_trunk     trunk-side levers of the OF3 code family: cast cache, template embedder, trunk graph, APB bias hoist, tuner guard
    tools         importable observe-only tooling shared by kits (graph_audit); nothing imported at package level
    compare       content equality of two output trees: digests, per-run stamped lines dropped, the per-item comparison
    jit_cache     the compiled-kernel cache key of the running stack and the cache-directory rule
    det           the deterministic-recipe shape and its stock-proof carve-out
    testing       recorded fixture lines and recorded fixture manifests: what a kit's adoption tests hold the kit to
    kernels       carried kernels (one copy each, matching the kit byte-for-byte): routed by NAME, held to the core's own copy live before import

The PEP 517 build backend a kit needs (setuptools plus the autoload .pth at the wheel root) is `kit_template/_build_backend.py` beside
this package: pip requires a backend to live inside the kit's own directory, so the kit carries a byte-identical copy that the
core's tests hold to the template.

This module imports nothing (PEP 562): the .pth path of a kit runs through it at interpreter start.
"""
__version__ = "0.5.228.0"


def lazy_getattr(package: str, exports=None):
    """The PEP 562 ``__getattr__`` of a package of this core. ``exports`` (a package's ``LAZY_EXPORTS`` table) maps a public name to the
    sub-module of ``package`` that defines it: the name is bound in the package namespace on first access (later reads never reach
    this function), so importing the package imports none of them. The table is read at access time (the package's own dict, not a
    copy). Any other name is imported as the sub-module ``<package>.<name>`` when there is one; a name that is neither raises
    AttributeError (a sub-module whose own import fails raises its ImportError unchanged)."""
    exports = {} if exports is None else exports

    def __getattr__(name: str):
        import importlib, sys
        sub = exports.get(name)
        if sub is not None:
            value = getattr(importlib.import_module("." + sub, package), name)
            setattr(sys.modules[package], name, value)
            return value
        if name.startswith("__"):
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        try:
            return importlib.import_module("." + name, package)
        except ModuleNotFoundError as e:
            if e.name == package + "." + name:
                raise AttributeError(f"module {package!r} has no attribute {name!r}") from None
            raise

    return __getattr__


LAZY_EXPORTS = {"warm_imports": "warm", "auto_warm": "warm",   # opt_core.warm_imports(): the stack's heavy libraries imported early through the trampoline (warm.py)
                "cell_census_dump": "cell_census"}                # opt_core.cell_census_dump(): the provider faces' cell-coverage census as text (cell_census.py)
__getattr__ = lazy_getattr(__name__, LAZY_EXPORTS)   # every sub-module by name, on first access; the names above from their sub-module
