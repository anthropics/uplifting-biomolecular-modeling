# ChromBPNet kit — what changes vs stock

Stock = ChromBPNet 1.0.1 at the pin (STOCK.md). Under a kit mode the package runs the kit's entry script
`opt/kit_ho/tf/pred_bw_fast.py <the stock arguments>` in place of stock's `predict_to_bigwig.main` — through `run.sh` /
`chrombpnet-opt`, or from the unchanged stock command line through the interpreter-start hook, which acts right after the stock
`chrombpnet` package is imported; `off` loads none of it. The model file, the weights, the 2,114-bp window rule and the arithmetic of
every output stay stock's. A mode is all of its levers on a card or it refuses by name; `fast` includes every `exact` lever. Lever
names are the registry's (`opt/chrombpnet_opt/registry.py`; `check --json` lists them under `levers`); a lever the card's table keeps
off is named after the job as `[chrombpnet-opt] GATED <lever>: <reason>`. The Triton kernels are the kit's own
(`opt/kit/torch/chrombpnet_k1/`); worker pools throughout are sized from the container's cgroup CPU quota and affinity mask.

## exact — outputs identical to stock

Identical to stock run under `--det 1` (TensorFlow's determinism settings), byte for byte for every output file; `det=1
precision=fp32` on the mode line.

- `forward_route` — the `-cm` model re-expressed as batch-invariant Triton kernels under torch (`route=k1`): the dilated-convolution
  stack as im2col dot kernels, the profile head, the global-average-pool + dense counts head; bias, ReLU and residual adds rounded as
  stock rounds them; weights read from `bias_scaled.h5` + `nobias.h5` through vendored bpnet-lite; TensorFlow is not imported on this
  route. Numerics: bitwise — at `ieee` input precision with `BLOCK_K = 16` every output element is one sequential fp32 FMA chain in the
  reduction order TensorFlow 2.8 / cuDNN 8.1's default algorithm uses (sm_90, sm_89). Kernels for sm_90 / sm_89 ship pre-compiled
  (`opt/kit/torch/triton_cache_sm90/`, `triton_cache_sm89_l40s/`); a cache whose recorded kernel-source hash or triton / torch version
  differs from this install is not used and the kernels compile in-process (`K1 cold JIT …` on the line). Steps aside: never on a
  `k1` card (torch / triton not importable, or kernels that fail to compile or launch, refuse the mode by name); on sm_80 (A100), on
  B200 and without a CUDA device there is no bitwise route and `exact` is refused by name (`… no bitwise Triton route on <arch> …`).
- `k1_batch` — the internal batch is `-bs` clamped to 1024 (the kernels index in int32); a larger request is chunked and said on the
  line. Numerics: bitwise (the kernels are batch-invariant).
- `tail` — the remainder batch zero-padded to the batch shape and the padding dropped, so one kernel shape serves the run (H100 /
  H200); stock's tail shape on other classes (`GATED tail`). Numerics: bitwise.
- `warmup` — one forward per shape the job will run, at load time (`k1` and `tf_function` routes); the `keras_predict_fileorder` route
  traces at its first batch as stock does. Numerics: none.
- `native_dilation` — TensorFlow routes only: dilated convolutions through `tf.nn.conv2d(dilations=)` instead of the SpaceToBatch
  form; on for the H100 / H200 / B200 TensorFlow routes, off on A100 / L40S and off under `exact`. Numerics: bitwise.
- `jit_cache` — TensorFlow routes only: `CHROMBPNET_OPT_CACHE_TAR` unpacked add-only into the process's CUDA ComputeCache before
  TensorFlow initialises (TensorFlow 2.8 carries no sm_90 kernels; without it the driver compiles stock's kernels in every process). A
  marker file makes later runs skip it; existing files are never overwritten. Numerics: none.
- `prefetch` — featurisation: the genome read through a memory map and one-hot encoded by an int8 lookup table, the next chunk
  prepared on a thread while the GPU runs the current one; stock's window rule, off-chromosome regions dropped and named. Numerics:
  identical inputs.
- `bigwig_writer` — a numpy re-implementation of libBigWig's writer in stock's block layout and interval rule, in a writer process
  started at script start (before any CUDA context) and fed chunk by chunk, so writing overlaps the forward; zoom levels built
  incrementally. Numerics: file byte-identical.
- `h5_writer` — the predictions `.h5` through stock's h5py call sequence with the `profs` chunks deflated in a thread pool and written
  raw in HDF5's flush order, from `H5_FAST_MIN_N` (20,000) regions up and for a single region; in between, stock's own writer.
  Numerics: file byte-identical.
- `metrics_stage` — with `-bw`: the observed track read by a numpy bigWig reader (one inflate per block), started early so it overlaps
  the forward; the profile JSD vectorised with stock's RNG stream consumed in stock's order; the two PNGs drawn by side interpreters
  spawned at script start; the three finish tasks run concurrently. Numerics: metrics JSON, `.h5` and PNGs byte-identical.
- `preimport` — the finish stage's imports (scipy, matplotlib, stock's metrics module) loaded on a daemon thread during the forward.
- `exit_fast` — after every output is closed, re-opened non-empty and synced and the side processes are joined, `os._exit(0)` skips
  interpreter and CUDA teardown; failure paths exit normally.
- `malloc_env` — glibc `mallopt` tunables (`M_MMAP_MAX`, `M_TRIM_THRESHOLD`, `M_TOP_PAD`) set before the numeric libraries load and
  exported to the side processes. Numerics of these three: none.
- TensorFlow's determinism settings (STOCK.md's `--det 1` block plus `CUBLAS_WORKSPACE_CONFIG=:4096:8`) are composed for the kit's
  process: `det=1 precision=fp32` on the mode line.

## fast — within stock's seed-to-seed variation

Kit-true class: stock's `pred_bw` draws no seeds, so the difference is a precision class, not a spread — `fast` differs from stock
as shipped by TF32 input rounding in the convolutions (stock's own precision class), not bitwise. The default mode. Every `exact`
lever without the determinism settings (`det=0` on the line), and:

- `forward_route` at `precision=tf32` — the same im2col dots at TF32 input precision on the tensor cores (`kernels_tc.py`; one tile
  per architecture from `arch_tiles.json` `tc_tile`: sm_90, sm_80), the two heads on the fp32 kernels. Numerics: TF32 input rounding in
  the convolutions — stock's own precision class as shipped — not bitwise. The tensor-core kernels compile on first use into the
  Triton cache directory, once per machine. On L40S, on B200 and without a CUDA device the route is `keras_predict_fileorder` (stock's
  own `model.predict()` inside the kit's pipeline); an architecture in no table engages `k1` with `not measured on this card` said.

## Every mode

- Stock exceptions: none (STOCK.md). No upstream fix ships: `-d <chr …>` fails in upstream 1.0.1 and the kit modes stop at it before
  predicting (`[pred_bw_fast] REFUSED: -d …`, exit 1).
- Stock arguments on the kit's line: `-cm -cmb -bm -r -g -c -op -os -bs -bw` behave as stock's — the models given run in stock's
  order (`-cmb`, `-cm`, `-bm`) with stock's output suffixes, and each rewrites the `-os` file as in stock; `-cmb` / `-bm` run stock's
  graph on the TensorFlow route in the same process (`TF_FORCE_GPU_ALLOW_GROWTH=true`, so the two frameworks share the card). `-t` is
  accepted and ignored.
- `--items FILE` (`exact` | `fast`): N regions files in one process — model, kernels, writer and PNG processes loaded once,
  `multi=<N>` on the mode line, exit 1 with `incomplete: <ok>/<N>` in `opt_manifest.json` when fewer outputs land than requested;
  `-bm`, `-cmb` and `-d` are refused in this form.
- The route per (architecture, mode) comes from `opt/kit/torch/chrombpnet_k1/arch_tiles.json` (`entries[<arch>].default_route`), the
  class table in `chrombpnet_fastkit/fastdefault.py` as fallback, and is printed as `route=`. A run record showing a forward other
  than that route ends `[chrombpnet-opt] NOT ACTIVE: partial activation — …`, exit 3.

## Switches

None selects a lever: kit-internal names found in the caller's environment are removed and named as `[chrombpnet-opt]
IGNORED names=<N1,N2,…> reason=<…>`; `--det 0|1` is stock-side (`--mode off` only); `--config <card>` sets deployment variables only (STOCK.md §Variables).
