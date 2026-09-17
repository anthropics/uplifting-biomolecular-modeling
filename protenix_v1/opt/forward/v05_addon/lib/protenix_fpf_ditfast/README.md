protenix_fpf_ditfast — the fused diffusion-sampler package behind lever words cond_dedupe / dit_fused / dit_lowp / atom_fused, served to
protenix 1.1.0 through the adapter `ptxfpf/ditfast_ptx1.py`.

Modules: `cond_dedupe` (DiffusionConditioning once per step on the sample-invariant noise level), `dit_fast` (the 24-block token
DiffusionTransformer as one fused forward; `FastTokenStack`), `atom_fast` (both 3-block atom transformers as fused stacks; `FastAtomStack`),
`_plumbing` (LeverRefused, the EXIT census `report()`), `vectors` (byte test vectors: `python -m protenix_fpf_ditfast.vectors`, needs CUDA).
Row kernels: the shared core's `opt_core.kernels.apb.ditfast`; attention: levers ditattn / atomattn's kernels (`opt_core.kernels.apb.fpf_apb`,
imported here under the name `protenix_fpf_apb`, which the adapter registers); hoisted operands: `lib/kit112_src/dit_hoist.py`'s slot protocol.
The installs read env words the adapter sets in-process from the arm words — PTX_DIT_ATTN, PTX_DIT_ATTN_FP16, PTX_DIT_FAST, PTX_DIT_LOWP,
PTX_ATOM_ATTN, PTX_ATOM_FAST — and address DiffusionModule / DiffusionTransformer / AtomTransformer attributes of protenix 1.1.0
(modules/transformer.py, primitives.py, diffusion.py). CELLS.json: the per-card cells.
