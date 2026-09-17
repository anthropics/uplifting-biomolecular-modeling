# lib/kit112_src — CUDA-graph and DiT-hoist modules (a kit import root: `protenix_v1_opt.kit.KIT_SYS_PATHS`)

| path | what | imported by |
|---|---|---|
| `infopt_graphs/` | CUDA-graph capture library (`GraphedFunction`, `StaticGraph`) and its Protenix integration (`protenix/`: the graphed diffusion denoiser loop `GraphedDenoiseLoop` — lever `sg`; its host path `sampler_prep` — lever `sampler_prep`; the stream-correct fast-LayerNorm rebuild `fastln_stream`); its own README inside | `ptxfpf/levers_ptx1.py`, `lib/fastln_prebuilt.py` |
| `dit_hoist.py` | lever `hoist`: the diffusion denoiser's step-invariant tensors (token pair-bias chain, conditioning and atom-transformer terms) computed once per sampler call outside the denoiser loop (record / hit / audit modes driven by the graphed loop through its `biascache` attribute) | `ptxfpf/levers_ptx1.py`, `lib/protenix_fpf_ditfast` |
