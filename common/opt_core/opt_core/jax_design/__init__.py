"""Design-side JAX helpers a kit's thin adapter calls — policies and bookkeeping only; no engine knowledge, no kernel bytes.

    pcc               the persistent compilation cache + XLA autotune pin: the cache key of the running (jax, plugin, GPU type)
                      stack, the environment a mode table exports, the in-process enable for a resident driver, the never-re-apply
                      rule, the equality key of a populated cache (equal keys + equal seeds => equal executables and trajectories),
                      the fields of the kit's activation-evidence line
    subbatch_policy   the attention sub-batch (row-chunk) decision for a traced executable: an explicit request, or 'auto' =
                      unchunked when the adapter's measured forward+backward peak fits the device, else the stock chunk; every
                      outcome is a named source (requested / auto:fits / auto:exceeds / auto:no_device / stock) rendered as one
                      key=value fragment for the kit's activation-evidence line

Pure standard library at module level and at call time: the device size, the measured peak points and any live jax are the ADAPTER's
(it imports jax and passes values or reads ``sys.modules``), so ``import opt_core.jax_design`` never imports jax. Submodules are
imported by name (``from opt_core.jax_design import pcc``). Nothing here names an engine.
"""
