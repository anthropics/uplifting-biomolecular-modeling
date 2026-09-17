# upstream/ — patches proposed against the Biohub transformers fork @ ef32577f55da19a4989cd7b22e004dc43a4998cb (transformers 4.57.6, models/esmfold2)

Not applied by the kit: every kit mode obtains U1–U3's and U5's effect in-process on the loaded model (`driver/ef2_opt.py`; U5's by launching the
unpatched kernel in row blocks inside its offset bound), and U4's idea is the padded addressing inside `driver/ef2_w4.py`. U1–U3 and U5 are pure
execution changes with bit-identical outputs (no dtype change, no reordering inside reductions); U4 changes the fp32 accumulation order of one GEMM
(tolerance class) and is guarded by its own switch. U5 is a defect fix: without it the kernel faults on any pair plane past 2**31 - 1 elements.

| patch | file | what | why |
|---|---|---|---|
| U1_swa_sdpa_mask_memoize.diff | modeling_esmfold2_common.py (SWA3DRoPEAttention, SDPA path) | memoize the boolean sliding-window mask (allowed, valid) per (indices tensor, B, N, half_window) instead of rebuilding it in every SWA block at every diffusion step | the mask is identical for all blocks × steps of one sample() call |
| U2_sampler_schedule_host_syncs.diff | modeling_esmfold2_common.py (sample loop) | convert schedule / gammas to Python floats once (`tolist()`) instead of three `.item()` device syncs per diffusion step | the syncs serialize CPU and GPU and prevent launch-ahead |
| U3_confidence_chain_pair_host_syncs.diff | modeling_esmfold2.py (ConfidenceHead pair_chains_iptm) | drop the `if chain_c1.sum() == 0: continue` per-chain host sync | bool() of a CUDA tensor = one sync per chain id; absent chains evaluate to exactly 0.0, equal to the zero initialisation |
| U4_trimul_stage3_row_pitch_alignment.diff | kernels/trimul_with_residual.py (stage 3) | pad the stage-3 einsum operands' row pitch to a multiple of 8 so cuBLAS uses its sm90 kernel when L % 8 != 0 | see U4_trimul_stage3_row_pitch_alignment.md |
| U5_pair_bias_int64_offsets.diff | kernels/fused_attention_pair_bias.py (forward + backward kernels) | form the row offsets of z, the LN statistics and the bias in int64 (`(pid_b.to(tl.int64) * Q + pid_q) * K …`) | the int32 offsets wrap negative once B·Q·K·DIM_Z > 2**31−1 (DIM_Z 256: 2,897 tokens): an illegal memory access in the diffusion transformer's pair bias; see U5_pair_bias_int64_offsets.md |

Also observed on this build (no patch): `ESMFold2Model.apply_torch_compile('fixed_seqlen')` fails for both variants (inductor
`PendingUnbackedSymbolNotFound` from the data-dependent `nonzero` in the atom-attention `to_keys` path inside DiffusionModule / DiffusionTransformer;
compiling only PairUpdateBlock / MSAEncoderBlock works).
