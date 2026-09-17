# trunk_levers — runtime levers on the Boltz-2 trunk (boltz 2.2.1)

Runtime patches of the Boltz-2 Pairformer / MSA trunk and the persistent worker the kit modes run in; nothing in the installed `boltz` package is
edited. The kit stages these files into its launch directory and sets their switches from its mode table (`opt/boltz2_opt/modes.py`); with no switch
set nothing is patched.

| path | role |
|---|---|
| `boltz_trunk_levers.py` | class-level patches of the `PairformerLayer` / `PairformerNoSeqLayer` forwards (`resid`) and the `AttentionPairBias` forward (`mask2`), with scoped `PairformerModule` / `PairformerNoSeqModule` / `MSAModule` forwards that compute the trivial-mask flag once per call; selected by `BOLTZ_LEVERS` (the modes' row: `resid,mask2`; unset: nothing is patched; an unknown name raises by name); `report()` returns the applied set and the counters the kit reads from the worker log |
| `boltz_flash_triattn_patch.py` | `BOLTZ_TRIATTN=flash`: the flash (fused, online-softmax) triangle-attention kernel (the core's `flash_triattn`) in place of the torch triangle attention at and above `BOLTZ_TRIATTN_MIN_TOKENS` (300) tokens; below it, and on any call the kernel does not take, the stock forward runs and is counted; `BOLTZ_TRIATTN_REPORT=1` prints the per-call census at exit |
| `src/bz_worker_lev.py` | the persistent worker: builds the stock model once, parses every input as one `boltz predict` process per input would (the kit's parsing fork server), replays stock's per-(input, seed) RNG position, predicts every (input, seed) of the batch and writes stock's file set; applies `boltz_trunk_levers` at import |
| `src/bz_worker_levf2.py` | the same worker with the flash triangle-attention patch imported after the trunk levers (the base of `fast` and `big`) |

The kit's rows use `BOLTZ_LEVERS=resid,mask2` with Boltz-2's own fused triangle kernels on (`use_kernels=True`): `resid` folds the eval-mode residual
`z + dropout_mask * f(z)` (mask == 1.0) into `z + f(z)` while keeping the `torch.rand` draw so the CUDA RNG stream is unchanged; `mask2` skips the
`attn + (1 - mask) * -inf` term of the Pairformer sequence attention when the token mask is all ones. Both compute the same arithmetic per output
element as the stock statement. The mask lever is the identity only for an all-ones mask — always true for one unpadded record per worker item —
and the module checks this at run time (one `(mask == 1).all()` per `PairformerModule` / `MSAModule` call), taking the untouched stock path for any
other mask (counted `mask_scope_nontrivial`; a mask it cannot classify is counted `mask_scope_error` and reported by the kit as a fallback). The
module implements no other lever.

Other switches the files read: `BOLTZ_LEVERS_VERBOSE` (1: one `[boltz_trunk_levers]` line at apply), `BOLTZ_TRIATTN_EXACT` / `_LIBDEVICE` /
`_WARM` and the `BOLTZ_TRIATTN_<tile>` overrides of the flash patch (unset by every mode: the kernel's own tile table serves), `BOLTZ_CHUNK` (the
worker's chunking threshold word, `stock` = upstream's).
