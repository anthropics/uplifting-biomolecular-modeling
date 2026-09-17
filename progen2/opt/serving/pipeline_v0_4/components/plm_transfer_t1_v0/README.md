# plm_transfer_t1_v0 — ProGen2 decode-step levers (resident rotary tables + static K/V) on the `sample.py` generation route

The component `progen2_decode.py` installs on the loaded model for the GENERATION workload through the
stock `sample.py` route (`model.generate(input_ids, do_sample=True, temperature, max_length, top_p, num_return_sequences, pad_token_id)`).
The kit patches the STOCK module's class attributes / module globals in-process (`models.progen.modeling_progen`) with the originals kept in
`kit_t1._ORIG`; every decode step runs the stock's eager kernels.

Levels (composable, `kit_t1.install(model, level=...)`):
- `t3`   resident rotary tables: the stock `fixed_pos_embedding` is called ONCE for `max_slots` positions on the device and sliced per call
         (the stock recomputes the sin/cos table on the CPU per layer per forward and copies it to the device). `rotary_table_check`
         compares the cached slices with fresh stock tables at every length 1..L (bit-pattern equality).
- `t3s`  + static KV cache: per-(layer, N) K/V slot buffers in the stock's own key/value dtypes (fp32 K / as-loaded V), written by
         `copy_` at slot t, attention over the views [0, t] (same kernels and shapes as the stock's `torch.cat` cache at every t). A batch
         size N is allocated when the caller asks (`allocate(model, N)`; `progen2_decode.py`: for the batch sizes of the work in front of the
         model and for a unit's on first sight — after its fit check over `kv_bytes_per_sample(model)`), kept until `release(keep)` drops it;
         `install()` allocates nothing but the `batches` it is given (none by default).

Limits: max_length <= max_slots (progen2_decode.py sizes it to the length bucket of the work in front of the model); `--sanity` passes unchanged
(the sanity forward is the stock's path through the same patched attention at batch 1).
