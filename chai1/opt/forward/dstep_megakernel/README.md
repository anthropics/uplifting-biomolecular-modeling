# chai1_fastln — diffusion-step levers for the `chai1_eager` tier1 line (Chai-1, chai_lab 0.6.1)

Three levers for the hoisted, CUDA-graphed denoiser step, built as a private flat denoiser per lever set (`chai1_fastln/stackx.py`
`build_lever_parts`, `ALL_LEVERS` = `hoist2`, `compiled`, `dit_attn`); nothing else in the pipeline changes (same steps, recycles, schedule, outputs).

* **`hoist2`** — the eager stack's value-taint hoister for the step (`chai1_eager.hoist.HoistedForward2`, `HoistedDiffusionWrapper(hoister="hoist2")`;
  eager README): the step-invariant atom-pair block, pair biases and AdaLN conditioning run once per sample instead of once per denoiser call.
  Bitwise to the base hoister and to the un-hoisted call under the deterministic recipe.
* **`compiled`** (fast / big) — `HoistedDiffusionWrapper.compile = "default"`: the hoisted per-step function through torch.compile / Inductor
  (`chai1_eager/hoist.py` `HoistedForward.compile`), captured by the same CUDA graph: the bias adds / masks on the `[S,16,N,N]` logits, AdaLN / gate /
  SwiGLU chains and strided copies are fused; GEMMs, the fp32 atom attention and LayerNorms are unchanged. Not bitwise (fused reductions reorder fp32
  sums; within the TF32 numerics class). Compile cost lands on the first item per crop; a warm start reuses Inductor's FX-graph / AOT-autograd caches
  under `TORCHINDUCTOR_CACHE_DIR` (chai1_opt/jit.py keys it under `MODEL_OPT_JIT_ROOT`). A compile that cannot engage steps aside by name.
  Ahead of time (`aoti.py`): `python -m chai1_opt.warm_aoti` exports each crop's hoisted step (tensors-only wrapper: device tensors and CPU
  tensors with elements among the step's arguments and cached values are inputs, the rest per-crop constants under a digest) and compiles it with
  AOTInductor into `$MODEL_OPT_JIT_ROOT/<key>/aoti/dstep_c<crop>_s<samples>_<levers>_<digest>.pt2`, weights left out of the package and bound from the
  live module at load (`load_constants(user_managed=True)`; the launcher's constant names are read from Inductor's lowered graph at build time);
  `stackx.build_lever_parts` binds the module on the wrapper and `HoistedDiffusionWrapper._hf` calls `aoti.attach(hf, …)` before compiling: a package
  for (crop, n_samples, levers, digest) serves the compiled step after a load (`run_single_threaded=True`, captured in the step's CUDA graph); none
  there → the Dynamo route, by name on stderr — except on cc 8.0, where the eager hoisted step serves that crop (`no_aoti_packages_cc80`); a package
  whose launcher was built for CPU instructions this host lacks is refused at load (`cpu_isa_mismatch`, `cpu_isa.py`) and the eager hoisted step serves.
* **`dit_attn`** (fast / big) — see below.

## Use (the chai1_opt package does this for every kit mode; `opt/chai1_opt/stack.py` `apply_eager`)

    import chai_lab.chai1 as C1
    import chai1_eager.stack as S, chai1_fastln.stackx as X
    comps = S.Components(device="cuda:0")
    base  = {"tier1": S.build_parts(comps, "tier1")}
    parts = X.build_lever_parts(comps, base, "tier1", ("hoist2",))     # a private flat denoiser on the value-taint hoister (("hoist2", "compiled", "dit_attn"): fast's tuple)
    C1.load_exported = S.make_loader(comps, parts)
    C1.run_inference(fasta_file=..., output_dir=..., num_trunk_recycles=3, num_diffn_timesteps=200, seed=0, ...)

Lower level: `chai1_fastln.dit_attn.patch_flat(flat_diffusion_module)` rebinds `scaled_dot_product_attention` inside that transpiled component's
namespace (an attribute on its shim; other components untouched); `unpatch_flat` restores it. Works graphed and under the deterministic recipe
(run-to-run bitwise).

License: Apache-2.0 (add-on code); derived from the exported TorchScript archives of Chai-1 (chai_lab 0.6.1, Apache-2.0) only through the
transpiler at run time — no weights, no chai-lab source included.

## `dit_attn` (fast / big)
`chai1_fastln.dit_attn`: the denoiser namespace's `scaled_dot_product_attention` becomes a `Router` — the DiT token form (5-D fp32 q/k/v `[1,16,S,N,48]`,
fp32 bias `[1,16,1,N,N]`) is viewed as `[S,16,N,48]` + `[1,16,N,N]` and served by `opt_core.kernels.apb` asked by the MODE's tier word
(`patch_flat(flat, word="fast"|"big")`; `stackx.TIER_WORD`): `_row_for(tokens, samples, D, H)` selects once per (tokens, samples) per process
(`kernels.apb.select(cc, fp32, "dit_h16d48", tokens, word=<tier>, samples=S, timing="graphed", capture=True)`; a trace-time constant inside the
compiled step) and the Router either launches that row inside one custom op (`torch.ops.chai1_fastln.dit_attn`, fake kernel registered: opaque to
Dynamo/Inductor) or — when the row IS the library statement (`STATEMENT_ROWS`: the provider's sdpa rows) or the provider refuses the call class —
keeps the original statement by name (Inductor fuses it). `available(word=)` steps the lever aside by a NAMED word (`no_apb_row_<kind>`) only when the
provider cannot select any row for the cell at install; anything unexpected propagates. `report()` carries the word, the rows served per token count, the
row / refusals for the kit's EXIT tally (`dit_attn=<row>:<served>/<calls>` or `dit_attn=stepped_aside:<reason>`).
