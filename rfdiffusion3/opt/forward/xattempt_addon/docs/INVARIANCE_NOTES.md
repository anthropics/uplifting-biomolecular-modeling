# INVARIANCE_NOTES — why each part of the roll-out cache returns the tensor upstream would have computed
Call structure (`RFD3DiffusionModule.forward`, one call per denoiser evaluation; the sampler's roll-out makes `num_timesteps-1` = 199 calls per design batch and
passes the SAME `initializer_outputs` objects (`Q_L_init, C_L, P_LL, S_I, …`) to every call; inside a call the decoder runs `n_recycle` = 2 times):
| part | expression (upstream code) | free variables | lifetime of the cache | guard |
|---|---|---|---|---|
| pll | `LocalAttentionPairBias`: `b = self.to_b(P_LL)` in the 3 `encoder.atom_transformer` blocks and 3 `decoder` blocks (full-`P_LL` path) | module weights, `P_LL` (initialiser output) | one roll-out (dropped in `finally`) | only if `P_LL` is one of the tensors the current denoiser call registered as roll-out constants (`hoist.mark_const(Q_L_init, C_L, P_LL, S_I)`); the token transformer's blocks call the same line with `Z_II` (rebuilt each call) → never cached |
| downcast | `S_I = self.downcast_c(C_L, S_I, tok_idx)` | weights, `C_L`, `S_I` (initialiser outputs), `f["atom_to_token_map"]` | one roll-out | — |
| valid | `valid = zeros(D,L,L).scatter_(2, indices, True)` in `dense_sdpa_pairbias_attention` | `f["attn_indices"]` (created once per denoiser call by `create_attention_indices(X_noisy_L …)`; the same tensor object is passed to all 9 atom-block invocations of the call) | one denoiser call (`RFD3DiffusionModule.forward` wrapper), keyed on the identity of the caller's indices tensor (pinned) | — |
| where | `bias.to(dtype).masked_fill(~valid, -inf)` → `torch.where(valid, bias.to(dtype), -inf, out=empty(D,H,L,L))` | same inputs; elementwise select; output layout = standard contiguous (as `clone` of the expanded view produces) | n/a (no cache) | — |
| dedup | the masked bias for a given (bias tensor object, valid tensor object) | both pinned in the entry | one denoiser call, FIFO of 3 | — |
CUDA-graph path (`RFD3_CUDAGRAPH`): the captured graph bakes in the addresses of cached tensors, and the graph sampler reuses a captured entry for later design
batches of the same shape after `copy_`ing the new inputs into the entry's static buffers. The roll-out cache therefore lives in the entry (`entry.hoist`), the
thunks that fill it close over the entry's static input buffers, and `hoist_refresh(entry)` recomputes every cached tensor IN PLACE (same storage) whenever
`_get_entry` reuses an entry — before the first replay of the new roll-out. The per-call caches are inside the captured region and replay as recorded. The cache
configuration is folded into the entry signature, so an entry captured under one configuration is never replayed under another.
End to end, the designs equal upstream's bit for bit under deterministic numerics for each part separately and together.
