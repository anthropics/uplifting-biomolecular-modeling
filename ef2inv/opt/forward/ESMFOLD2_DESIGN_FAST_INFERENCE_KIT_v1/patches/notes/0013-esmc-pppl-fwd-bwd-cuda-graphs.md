# ESMC pseudo-perplexity loss: CUDA-graph replay of the 6B LM forward AND backward (`k/ef2_pppl_graph.py`)
**What/why:** the cookbook's `compute_esmc_pseudoperplexity_nll` runs the 80-layer ESMC-6B transformer stack forward+backward
(grad w.r.t. the input embeddings; weights frozen) on LM_MASK_PASSES = 4 masked copies of the binder every design step. This is
launch / host-bound at design binder lengths. `ef2_pppl_graph.enable(esmc_model)` replaces
`esmc.transformer.forward` by a wrapper that, for grad-enabled calls with the cookbook's signature (sequence_id=None, no layer
collection), captures one (fwd, bwd) CUDA-graph pair per input shape (LRU 2) and replays it through a `torch.autograd.Function`;
all other calls (the fold models' own inference-mode use of the shared trunk, the LM-forward graph of `ef2_esmc_graph`) fall through to eager.
Masking RNG, one-hot/straight-through, lm_head, log-softmax and the NLL stay eager.
**Numerics class:** exact — bitwise (replay of the kernels recorded from the eager path). A capture-time replay-vs-eager check (forward
output and input gradient `torch.equal`) is built into the module (`check_replay=True`; on a mismatch the lever disables itself by name).
**Not done (and why):** (i) chunking — with design batch 1 the 4 masked copies are already ONE chunk (LM_LOSS_BATCH_SIZE=128), so
there is nothing to merge; for design batch K>1 the stock code batches 4K rows per chunk already. (ii) 'unmasked-context reuse' —
each masked copy is a full bidirectional forward; there is no shared prefix computation to cache without changing the algorithm.
(iii) bf16-stored LM weights — with the cookbook's REUSE_ESMC=True the LM trunk IS the fold model's bf16 `_esmc`; only the small
lm_head is fp32 (two weight casts per step). (iv) featurisation overlap — step t+1's sequence depends on step t's update, so
prefetch is impossible without changing the algorithm; the featurisation memo (`ef2_loop_prep`) removes the cost whenever the argmax
sequence repeats (most late steps).
