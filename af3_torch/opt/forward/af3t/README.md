# af3t — PyTorch AlphaFold3 forward pass, weight-compatible with the AF3 haiku parameter layout (OF3-ported weights), + kernel stack

Layout
* `af3_torch/xfold/`      — xfold (github.com/Shenggan/xfold @22bdeed, Apache-2.0) + patches: OF3 weight layout flag (`xfold/of3.py`, set `of3.OF3=True`
                            before building), 7 sokrypton-fork code-path mirrors, 2 upstream bug fixes (recycles+1, Gaussian noise), diffusion-head
                            hoist + whole-step CUDA graph (`DiffusionHead.prime_static / forward_graphed`), converter `xfold/params.py::import_jax_weights_(model, params_dir)`.
* `af3_torch/af3_torch_api.py` — DOCUMENTED ENTRY POINTS: build_model(levers='eager'|'fastest' or a tuple of lever names, compile=False), enable_compile(), batch_from_npz()/batch_from_features(), run_target_feat(), run_trunk(),
                            run_diffusion(), denoise(), prime_diffusion(), run_confidence(), run_distogram(), forward(), inference() context. Read its module docstring.
* `kernels/`              — fused-kernel adapters (af3_kernels.py: levers trimul | triattn | transition | apb | …, routed to the shared core's providers
                            under ../../../../common/opt_core/opt_core/kernels/, whose NOTICE files carry their third-party notices; af3t_glu_proj.py;
                            af3t_msa.py + third_party/af3t_opm.py: the MSA module's outer-product-mean kernels; third_party/lnl_fused.py: LayerNorm fused
                            into the consuming linear — both third_party/ files are written for this kit). No per-arch cell table (the providers carry the cells).
* `../../../THIRD_PARTY_NOTICES.md` — third-party notices for this kit (xfold, AlphaFold 3 and the reference fork, the weights' source); licence text in
                            `../../../third_party_licenses/`.
No weights are included anywhere. Weights are read from the params directory at run time.

Minimal use from opt/forward/ (two venvs, ../../README.md "Variables": $AF3_TORCH_JAX_PY = the reference fork's (AlphaFold 3 open code, JAX), $AF3_TORCH_PY = torch 2.13.0+cu130 /
triton 3.7.1; the package's featuriser, ../af3_torch_opt/featurise.py under the JAX venv, writes the batch.npz the api reads; `run.sh pred` composes all of it):
    JAX_PLATFORMS=cpu $AF3_TORCH_JAX_PY ../af3_torch_opt/featurise.py --item x=fold_input.json=WORK --run_data_pipeline 0 --report featurise.json   # -> WORK/seed-<s>/batch.npz per modelSeed
    $AF3_TORCH_PY - <<'PY'
    import sys; sys.path.insert(0, "af3t/af3_torch"); import af3_torch_api as A
    model = A.build_model("<checkpoint file>", levers="fastest"); batch = A.batch_from_npz("WORK/seed-1/batch.npz")
    with A.inference(): out = A.forward(model, batch, seed=1)
    PY
