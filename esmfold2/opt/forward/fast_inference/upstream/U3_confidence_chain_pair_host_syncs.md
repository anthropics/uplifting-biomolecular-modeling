# PR: Drop the redundant per-chain host sync in ConfidenceHead pair_chains_iptm

The `if chain_c1.sum() == 0: continue` guard forces a device->host sync per chain right after the sampler; it is also redundant because for an absent chain id the masked numerator is 0 and the denominator is _EPS, so the entry is written as 0 exactly as the pre-initialised tensor. Removing the guard is bit-identical (unit test incl. absent chain ids) and removes n_chains syncs per fold; this was also the line where sticky asynchronous CUDA errors from earlier kernels surfaced, which confused debugging.

Diff: `U3_confidence_chain_pair_host_syncs.diff` (against Biohub/transformers @ ef32577f55da19a4989cd7b22e004dc43a4998cb).
