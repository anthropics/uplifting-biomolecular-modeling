# Borzoi kit — what changes vs stock

Stock = Borzoi `borzoi_sad.py` at the pin (STOCK.md). The kit is an entry-script swap: `opt/datapath/pipeline_tf/v17/borzoi_sad.py`
is the stock script with the kit's blocks inserted at fixed anchor lines (rendered by `opt/datapath/pipeline_tf/build.py` from
the pinned bytes; no stock line is edited; each block is bracketed `# ---- kit … ----`), and `v17/kitlib/` holds the levers.
`exact` is all of the levers below; `off` loads none. Lever names are the ones `bash run.sh check` prints. The model's arithmetic
— ops, kernels, precision, ensemble, batch — is stock's in every mode; no kernel of the shared core (`common/opt_core`) is
bound, and `borzoi_opt` depends on nothing outside the standard library.

## exact — outputs identical to stock

- `graph_forward` (`kitlib/forward.py`) — the first forward call of the process is stock's eager Keras call, where cuDNN
  selects its convolution algorithms as in a stock process; the same model is then traced once with
  `tf.function(model, jit_compile=False)` and every later call runs that graph instead of per-op eager dispatch. The
  documented input has one shape, (2, 524288, 4), hence one trace per process; the graph keeps its intermediate buffers on the
  device, so the peak is above the eager call's. Numerics: bitwise (the same ops on the same kernels; dispatch only). Steps
  aside: a call with `head_i` goes to stock's `__call__` (counted as a stock call); without TensorFlow loaded the call stays eager.
- `copy_free_forward` (same module; both forward levers sit behind the package-set switch `KIT_FWD=1`) — stock returns
  `model(x).numpy().astype(dtype)`, a second float32 → float32 host copy of the (2, L, T) prediction; the kit returns the
  `.numpy()` array and converts only when the requested dtype differs. Numerics: bitwise. Steps aside: never.
- `onehot_lut` (`kitlib/onehot.py`) — the one-hot of each window by a (256, 4) lookup table bound in place of
  `baskerville.dna.dna_1hot`'s per-base loop; 0, 1 and 0.25 are exact in float16 and each byte value maps to stock's row.
  Numerics: bitwise. Steps aside: arguments the table cannot reproduce verbatim (`n_sample`, `seq_len` trim/pad with
  `n_uniform=False`) go to the stock function.
- `pipelined_chunked_post` (`kitlib/sad_post.py`) — variant k's host post-processing (untransform → strand collapse →
  statistics → HDF5 write) runs per contiguous column chunk on a thread pool while the main thread proceeds to variant k+1's
  one-hot and forward; results are written from the main thread in variant order and flushed before the percentile pass.
  Numerics: bitwise — untransform and every statistic are elementwise operations or per-column reductions over the same column
  bytes in the same layout, the target length (6144 bins) is a multiple of every SIMD width so chunking moves no element
  between vector and scalar paths, and numpy neither fuses nor reassociates. The chunk width comes from a per-temporary byte
  budget that keeps the float64 temporaries in malloc arenas instead of freshly mapped pages. Steps aside, on one
  `ROUTED post=stock:<reason>` line with the job's own exit code: an option set outside {a targets file with `strand_pair`,
  per-column statistics `SAD SADlog logSAD sqrtSAD SAX D1 logD1 sqrtD1 D2 logD2 sqrtD2 JS logJS`} runs stock's
  post-processing lines for the whole job (`options`); an array outside the shape class (2-D, length a multiple of 16, the
  targets' width) runs them for that variant (`shape`).
- `post_writer` (`kitlib/writer.py`) — stock's `write_snp_len` expressions (`np.power(x, 2)` included), computing only the
  intermediates the requested statistics use. Numerics: bitwise. Steps aside: a statistic outside the set above sends the
  whole call to the stock function.
- `cores_probe` (`kitlib/sad_post.py`) — sizes the thread pool from the cores the process can actually use: a numpy-only
  subprocess probe started before the TensorFlow import and joined at the first post-processing step, checked against
  scheduler affinity and the cgroup quota. Numerics: none (pool size only). Steps aside: a probe that cannot run or reads
  implausibly → the pool is sized from affinity/quota and `borzoi-opt sad` prints `FALLBACK cores_probe=os_count (<reason>) pool=<n>`.
- `kit_stamp` (the entry script) — one `KIT_STAMP {json}` line on stdout at exit (kit name, writer, pool, forward counters)
  and, under `borzoi-opt sad`, the same JSON in the private `KIT_STAMP_DIR`, which the wrapper reads to decide `partial` /
  `ROUTED` / `FALLBACK` and then removes; nothing is written to the output directory. Numerics: none. Steps aside: never.

## Every mode

- No stock exception applies (STOCK.md) and no opt-in upstream fix ships. `--det 1` exports the deterministic recipe
  (STOCK.md) under any mode, `off` included, and changes no lever.
- How a mode reaches the process: `borzoi-opt sad --mode exact` runs `python v17/borzoi_sad.py <stock arguments>` with
  `KIT_FWD=1` and a private `KIT_STAMP_DIR`; `BORZOI_OPT=exact borzoi_sad.py …` reaches the same entry through
  `borzoi_opt_autoload.pth`, which recognises the pinned stock script at interpreter start and re-execs the kit entry with the
  same interpreter flags and arguments (one `SWAP` line); any other program under `BORZOI_OPT=exact` runs as written and
  receives `graph_forward`, `copy_free_forward` and `onehot_lut` when it imports `baskerville` (`borzoi_opt/_autoload.py`; a program
  that predicts through Keras `predict()` or takes gradients does not pass through `SeqNN.__call__` and is not accelerated);
  `--mode off` runs `python -s -m borzoi_opt.stock_sad`, which confirms the process is clean (STOCK.md §How stock is run) and
  then runs the stock script as `__main__`.

## Switches

- `--allow-partial` / `BORZOI_OPT_ALLOW_PARTIAL=1` — a run whose lever did not engage as described (no stamp, the graph not
  armed, a second trace, a stock forward call, a writer other than `exact`) is otherwise `NOT ACTIVE: partial activation — …`,
  exit 3; with the switch it is `PARTIAL allowed: …` and keeps the job's own exit code.
- `--det 0|1` — STOCK.md. No lever-ablation switch ships: the package sets exactly `KIT_FWD=1` and `KIT_STAMP_DIR`, and a
  `KIT_*` name the kit reads, set by the user, is refused before anything runs.
- `configs/h100.env` / `configs/a100.env` / `configs/h200.env` — deployment values only (STOCK.md §Variables); the lever set and
  the code path are the same on every card.
