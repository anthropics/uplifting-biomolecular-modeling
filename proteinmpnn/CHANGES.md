# ProteinMPNN kit — what changes vs stock

Stock = ProteinMPNN at the pin (STOCK.md). `off` changes nothing. `exact` is a driver mode: instead of installing modules over stock
functions it runs the kit's worker executable, `opt/forward/mpnn_exact_worker/addon/mpnn_worker2.py`, on the stock arguments with the line
`--mode stream --bb_batch 16 --sort_by_length --x_all --hybrid_gemm`, staged per run as `kit/mpnn_worker2_lowmem.py` (the `lowmem` featuriser
inlined). The worker imports upstream's `protein_mpnn_utils` unmodified from `$MPNN_DIR` and computes in float32 throughout, as stock. A mode
is all of its levers. Lever names are the ones printed in `levers=…` on the run's ACTIVE line (`bb_batch=` is its own field); the mode table
is `opt/proteinmpnn_opt/modes.py`, the registry `opt/proteinmpnn_opt/registry.py`. The kit binds no shared-core kernel: it uses the shared
core (`common/opt_core`) for the mode table, the activation report and the pin gates.

## exact — outputs identical to stock

- `--mode stream` (`stream`) — one resident process designs every backbone of the `parsed.jsonl`; each backbone's decoding-order and sampling
  draws, round after round, are replayed from the one Philox stream a stock invocation consumes across backbones. Numerics: bitwise (the same
  draws at the same stream offsets). Steps aside: never.
- `--bb_batch 16` — 16 backbones per encoder / decoder forward (`design --bb_batch K` sets another width: speed and memory only). A backbone
  shorter than the model's 48 neighbours runs in a batch of its own at the stock shapes, named on one `bb_batch:` line. Numerics: bitwise —
  the batch never enters a backbone's arithmetic (the levers below). Steps aside: never.
- `--sort_by_length` — batches group backbones of similar length. Numerics: bitwise (stream replay keeps each backbone's own offset, so
  order is free). Steps aside: never.
- `--x_all` — the worker's exact set: stands for `--chunk_gemm --cache_enc_ctx --graph_rng --single_graph --fused_draw --analytic_offsets
  --stock_shape_enc` (read from the worker's own help text).
- `--chunk_gemm` — every decode-step GEMM of a batch (the M = B·48 message GEMMs and the M = B position-wise feed-forward / output-head GEMMs,
  B = `--batch_size`) runs per backbone at the stock shape. Numerics: bitwise (a GEMM at the stock shape selects the stock cuBLAS kernel).
- `--stock_shape_enc` — the encoder and the two scoring forwards per backbone on exactly the stock tensors ([B, L] clone sequences,
  unpadded); the encoder output is computed once per backbone and reused by the three passes that need it. Numerics: bitwise (stock
  shapes; the reuse is caching).
- `--cache_enc_ctx` — the decoder's encoder context (`h_EXV_encoder_fw` in upstream's `sample`) assembled per decoded position from `h_V` /
  `h_E` with upstream's own gathers and float32 product, masks indexed directly; the all-positions [N, L, K, 3H] tensors are never built and
  encoder outputs are released as soon as sampling returns. Numerics: bitwise (data movement only).
- `--graph_rng` — the whole autoregressive decode step (3 decoder layers, output head, softmax, the per-backbone draw on registered
  per-backbone Philox generators, the state commits) captured in one CUDA graph and replayed L times per round. Numerics: bitwise (replay
  runs the captured kernels; the RNG stream is consumed as in stock). Steps aside: no CUDA device → the mode refuses at activation.
- `--single_graph` — one graph per decode shape (batch width, `--batch_size`, L_max) and temperature, reused by every later batch of that
  shape; upstream's teacher-forced scoring forward captured per shape likewise and run on its own stream underneath the decode steps.
  Numerics: bitwise. Steps aside: as `--graph_rng`.
- `--fused_draw` — the decode step's per-backbone `torch.multinomial` calls replaced by one Triton kernel per step that computes torch's own
  sampler: Philox4x32-10 at each backbone's generator (seed, offset), curand's uniform, ATen's exponential transform (`__logf` in float32,
  `log` in float64 — upstream's probabilities are float64), IEEE quotient, leftmost argmax; a per-backbone draw counter advances the offset
  by the increment one torch draw consumes; the tile is `--batch_size` rounded up to a power of two with the extra lanes masked. Numerics:
  bitwise by construction, and probed at start-up against `torch.multinomial` on the device in the job's dtype. Steps aside: `fused_draw:
  PROBE FAIL` → the job is refused by name; no CUDA device → refused at activation.
- `--analytic_offsets` — each backbone's RNG stream offset computed from per-call Philox increments instead of replaying every draw.
  Numerics: bitwise (same offsets).
- `--hybrid_gemm` (probe-gated) — the decoder message GEMMs (M = B·48 per backbone) of all backbones of a batch in one call, in groups sized
  so that M stays at or below the card's ceiling in the worker's `HYBRID_GROUP_ROWS` table (sm_80: a 16-backbone batch as two groups of 8;
  sm_90: the whole batch); the M = B GEMMs stay per backbone. Numerics: bitwise only where the batched kernel accumulates identically to the
  per-backbone call, which the worker probes for every (batch size, group, neighbours) decode-step shape of the job on the device before any
  output (`hybrid_gemm: PROBE PASS …`). Steps aside: `PROBE FAIL …` → `hybrid_gemm: REFUSED`, nothing designed, exit 3; `--hybrid_gemm 0`
  leaves it out by name (`opted_out=hybrid_gemm`).
- `lowmem` (`opt/proteinmpnn_opt/lowmem.py`, inlined into the staged worker) — the O(L²) sites of upstream's featuriser — masked CA–CA
  distances before top-k, the 24 atom-pair RBF distances, residue offsets / same-chain flags, the decoding-order `mask_attend` — recomputed
  on [B, L, K] neighbour tensors (K = 48) with the same float32 arithmetic per element. Numerics: bitwise (no random operation added, removed
  or reordered; no reduction changes order); working set linear in L.
- `fast_parse` (`opt/forward/mpnn_pdb_parser/kit/fast_parse.py`) — for a PDB-directory input, the stock parse helper's logic with each file
  read once and its ATOM lines bucketed by chain, started alongside the worker so the worker's input-independent start-up overlaps the
  parse. Numerics: byte-identical `parsed.jsonl`.

Also in the worker, without a switch: up to three batches' decode loops in flight at once, each on its own CUDA stream, decode slot and
generator bank; a batch's native-score forward issued before, and its re-scoring forward after, the next batch's decode loop, underneath the
decode-step replays; files written in batch order as each batch's re-scoring completes; the first batch of a job runs alone (it captures the
graphs). Scheduling only — what runs, on which tensors, with which kernels is unchanged. A decode step is skipped only when all clone
sequences sit on a missing residue (upstream's rule). Outputs follow upstream's own gates: `seqs/<name>.fa` always, `scores/<name>.npz` with
`--save_score 1`, `probs/<name>.npz` with `--save_probs 1` (the arrays are computed either way, as in stock). The `--x_all`-without-
`--chunk_gemm` composition (batched decode-step GEMMs) exists in the worker as a numerics-changing option; no mode selects it.

## Every mode

- No stock exception and no upstream fix ships: `off` is upstream as released, and `exact` refuses by name what its worker does not
  implement — `--ca_only`, `--score_only`, `--conditional_probs_only`, `--unconditional_probs_only`, `--tied_positions_jsonl`,
  `--backbone_noise` above 0 (exit 3, nothing launched; `--mode off` runs them) — as well as a checkout at a commit other than the pin and
  anything a pass cannot run without (the kit's files, the checkout, its weights). What is merely untested — another GPU model, driver or
  torch — is named on the activation lines (`gpu=… match=no`, the `STACK` line) and runs.

## Switches

- `--hybrid_gemm 0` — `exact` without the probe-gated lever, by name. `--bb_batch K` — the batch width. `--allow-partial`
  (`PROTEINMPNN_OPT_ALLOW_PARTIAL=1`) — the one recorded override for levers that cannot run on the machine (`allow_partial=yes
  partial=<levers>`); it does not override a failed probe. `--det 0|1` — accepted and inert (both modes are deterministic at a given `--seed`).
  Variable names and defaults: STOCK.md.
