# upstream_issues/ — candidate PRs (mosaic / boltz), texts only

1. `01_boltz_skip_random_init_when_loading_checkpoint.md` — boltz `Boltz2.__init__` spends most of the model-load time (CPU) in `initialize.trunc_normal_init_` (scipy truncnorm) +
   torch init for weights that `load_from_checkpoint(strict=True)` immediately overwrites. Proposal: construct under a skip-init path when loading a checkpoint;
   parameters are identical (sha256 over all leaves). Performance PR; the kit's `mosaic.fast.fastload` is the monkey-patch form of it (lever P2).
2. `02_mosaic_docs_reproducibility_and_compilation_cache.md` — mosaic docs: (a) the notebook seeds x0 with `np.random.randint` and passes `key=None` to
   `simplex_APGM` → runs are not replayable; expose/print keys; (b) Boltz featurization may redraw conformers per process → document frozen features for
   replayable runs; (c) recommend `jax_compilation_cache_dir` + XLA autotune dump/load for multi-process runs (this kit's P1). Documentation PR.
3. `03_mosaic_pyproject_unsatisfiable_jax_cuda_group.md` — at 70fec525 `uv sync --group jax-cuda` fails: `esm` requires torch 2.11 while mosaic pins
   torch==2.7.1; and an unpinned `jax[cuda12]` resolves a newer jax against jax-cuda12-plugin 0.10.2 (silently CPU-only). Proposal: pin
   jax==jaxlib==jax-cuda12-plugin, extra-gate the optional model back-ends. Packaging report with repro log (the kit's environment/requirements.lock is the working set).
No numerical/correctness bug was found in mosaic, joltz or boltz along the Boltz-2 hallucination path, so there is no bug-fix diff here.
