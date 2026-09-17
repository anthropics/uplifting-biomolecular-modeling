"""P1 — persistent compilation cache + XLA autotune pin for mosaic/JAX design runs.

Usage (must run BEFORE the first jax computation, e.g. at the top of your script):
    from mosaic.fast.repro_cache import enable
    enable("/path/to/compile_cache")           # the same directory for every process that should share executables (per jax version + GPU class)

What it does: sets jax_compilation_cache_dir (+ min compile time 0 so every executable is stored) and appends
--xla_gpu_dump_autotune_results_to / --xla_gpu_load_autotune_results_from to XLA_FLAGS (dump if the file does not exist yet, load otherwise).
Why: (1) the first design in every fresh process otherwise pays the full XLA compilation of the design step and of the refold executable;
(2) every fresh XLA compile re-runs GEMM/conv autotuning and produces a (slightly) different executable, so the same seed gives a different
trajectory in every process. With the pin, trajectories are bitwise identical across processes and hosts of the same GPU class.
Numerics class: exact (arithmetic unchanged; you get the populating process's executable)."""
import os, warnings

def enable(cache_dir: str, autotune_file: str | None = None, verbose: bool = True):
    os.makedirs(cache_dir, exist_ok=True)
    autotune_file = autotune_file or os.path.join(cache_dir, "xla_autotune_results.pb")
    flags = os.environ.get("XLA_FLAGS", "")
    if "autotune_results" not in flags:
        if os.path.exists(autotune_file):
            flags += f" --xla_gpu_load_autotune_results_from={autotune_file}"
        else:
            flags += f" --xla_gpu_dump_autotune_results_to={autotune_file}"
        os.environ["XLA_FLAGS"] = flags.strip()
    try:
        import jax
        if jax.config.jax_compilation_cache_dir not in (None, "", cache_dir):
            warnings.warn(f"jax_compilation_cache_dir already set to {jax.config.jax_compilation_cache_dir}; overriding with {cache_dir}")
        jax.config.update("jax_compilation_cache_dir", cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    except Exception as e:  # jax not importable yet: fall back to env vars read at import
        os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_dir
        os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
    if verbose:
        print(f"[mosaic.fast] compilation cache: {cache_dir}; XLA_FLAGS: {os.environ.get('XLA_FLAGS','')}")
    return {"cache_dir": cache_dir, "autotune_file": autotune_file, "xla_flags": os.environ.get("XLA_FLAGS", "")}

def identity_key(cache_dir: str, autotune_file: str | None = None):
    """sha256 of the autotune results file + listing hash of the cache dir: record this next to your results; equal keys => bitwise-equal trajectories for equal seeds."""
    import hashlib
    autotune_file = autotune_file or os.path.join(cache_dir, "xla_autotune_results.pb")
    key = {"autotune_sha256": None, "cache_listing_sha256": None, "n_cache_entries": 0}
    if os.path.exists(autotune_file):
        key["autotune_sha256"] = hashlib.sha256(open(autotune_file, "rb").read()).hexdigest()
    if os.path.isdir(cache_dir):
        names = sorted(os.listdir(cache_dir)); key["n_cache_entries"] = len(names)
        key["cache_listing_sha256"] = hashlib.sha256("\n".join(names).encode()).hexdigest()
    return key
