"""chai1_fastln — diffusion-step levers for the chai1_eager tier1 line (Chai-1, chai_lab 0.6.1): a private flat denoiser per lever set
(`stackx.build_lever_parts`; levers `hoist2` | `compiled` | `dit_attn`, `stackx.ALL_LEVERS`), served through the eager stack's own loader.
    import chai1_eager.stack as S, chai1_fastln.stackx as X
    parts = X.build_lever_parts(comps, base, "tier1", ("hoist2",))
`aoti.py` builds and serves the `compiled` lever's step ahead of time (torch.export + AOTInductor packages per crop under
$MODEL_OPT_JIT_ROOT/<key>/aoti, weights bound from the live module at load): `python -m chai1_opt.warm_aoti`; `aoti.attach(hf, wrapper=…)`.
On cc 8.0 a crop without a package runs the eager hoisted step by name instead of compiling; step inputs that arrive as unaligned views are copied
to aligned buffers (cache leaves once per item, arguments per call) and `warm_aoti --unaligned` compiles named inputs without the alignment
assumption; a package whose C++ launcher was compiled for CPU instructions this host lacks (the archive's recorded AOTI_CPU_ISA vs the host's word,
`cpu_isa.py`) is refused by name at load (`cpu_isa_mismatch`) and the eager hoisted step serves. `dit_attn.py`: the diffusion transformer's
pair-biased token attentions through the shared core's `opt_core.kernels.apb` by the mode's tier word.
"""
__version__ = "0.3.13"
