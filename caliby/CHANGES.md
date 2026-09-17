# Caliby kit — what changes vs stock

Stock = Caliby 0.1 at the pin (STOCK.md). Each lever is one upstream module replaced by the kit's copy of it: all eight copies are served from
`opt/forward/xattempt_addon/fast/` — the `CALIBY_X_*` levers are that directory's own, the `CALIBY_FAST_*` levers are the patches under
`opt/forward/fast_inference/patches/` already folded into the same copies; `stock/` beside `fast/` holds the originals and the two `patches/`
directories the per-lever diffs. The copies are served under the upstream module name by an import hook (`opt/caliby_opt/overlay.py`) inside the one design process after the installed tree has been checked
against the pin; site-packages is never written and `off` installs no hook. Each lever reads its switch at call time; a kit mode exports one
complete switch set (`opt/caliby_opt/modes.py` `ROWS`; `opt/caliby_opt/registry.py` describes each switch) and the switch names below are the ones
the ACTIVE line's `switches=…` and the `LEVER name=…` lines print. One Triton kernel is the kit's own (the low-complexity penalty,
`fast/complexity.py`); from the shared core (`common/opt_core`) the kit uses the activation report and lever-line grammar, the background output
writer, per-sequence RNG positioning and the out-of-memory classifier (an out-of-memory error is re-raised by every lever, never a fallback).

## exact — outputs identical to stock

`ensemble32` (also what `--mode fast --variant ensemble32` runs).
Switch set `CALIBY_FAST_SAMPLER=2 CALIBY_FAST_POTTS_PARAMS=2 CALIBY_X_SPARSE_EXACT=2 CALIBY_X_LCP=1 CALIBY_X_TIED_DET=1 CALIBY_X_ENS_WORKERS=8 CALIBY_X_PP_CACHE=1
CALIBY_X_BG_CIF=fork CALIBY_X_PP_NOSYNC=1 CALIBY_X_PP_FASTPDB=1 CALIBY_X_MULTISEQ=1 CALIBY_X_CIF_WORKERS=8` (`ROWS["ensemble32"]`): the `fast` levers of the next section
(no clean lever — `--clean_workers` N > 1 there is upstream's own joblib-parallel `clean_pdbs`) plus the five below. Numerics: bitwise — the same seed gives the same conformers, sequences, Potts energies and output CIF bytes as `off`. The two
Protpardelle levers load only when `protpardelle` is installed; without it `ensemble32` refuses and the other levers are unaffected.

- `CALIBY_X_TIED_DET` (`atom_mpnn_denoiser.py`) — tied-group aggregation of the Potts field over the conformers by sequential per-member adds on the
  GPU (the CPU `index_add_` order) instead of CUDA's atomic `index_add`: run-to-run deterministic. No effect on `single`.
- `CALIBY_X_ENS_WORKERS` (replaces `caliby/eval/eval_utils/inference_dataloader.py`) — the N conformers of an ensemble featurised on N loader workers
  and collated in index order; `0` = one worker, serially, as stock.
- `CALIBY_X_PP_CACHE` (`caliby/api.py`) — the Protpardelle-1c model kept across `generate_ensembles` inputs (stock reloads the checkpoint per input).
- `CALIBY_X_PP_NOSYNC` (replaces `protpardelle/core/models.py`) — the denoising loop's per-step snapshots stay on the GPU, copied to the host once after it.
- `CALIBY_X_PP_FASTPDB` (replaces `protpardelle/data/pdb_io.py`) — the PDB writer formats from one host copy per tensor instead of per-atom device
  reads, byte-identical text; inputs it does not cover run the stock loop.

## fast — within stock's seed-to-seed variation

`single` (the default mode there); the `exact`-only levers above are not part of it.
Switch set `CALIBY_FAST_SAMPLER=2 CALIBY_FAST_POTTS_PARAMS=2 CALIBY_X_SPARSE_EXACT=2 CALIBY_X_LCP=1 CALIBY_X_CLEAN=0 CALIBY_X_BG_CIF=fork CALIBY_X_MULTISEQ=1
CALIBY_X_CIF_WORKERS=8` (`ROWS["single/serial"]`); `--clean_workers N` with N > 1 exports the same set with `CALIBY_X_CLEAN=loader` (`ROWS["single/loader"]`).
Numerics: the same sequences as stock at the same seed except where a Potts energy that differs from stock's in its last bit flips a sampled
residue — the reason `--mode exact --variant single` is refused by name (exit 2).

- `CALIBY_FAST_SAMPLER` (replaces `caliby/model/seq_denoiser/denoisers/seq_design/potts.py`) — the Potts sweep loop without per-step host syncs
  (`1`), the sweep body captured as one CUDA graph per call (`2`); same ops, same RNG draws; serves upstream's sampling configuration (DLMC proposal,
  no rejection step, no trajectory) on CUDA tensors. Steps aside: any other call runs upstream's loop, `[CALIBY_FAST_SAMPLER] … -> stock sampler for this call`.
- `CALIBY_FAST_POTTS_PARAMS` (replaces `caliby/model/seq_denoiser/denoisers/atom_mpnn_denoiser.py`) — Potts parameters aggregated on the GPU, a dead
  device→host copy removed (`1`); `2` also keeps the decoder's sparse 48-neighbour couplings J for untied positions instead of densifying to N×N.
  Steps aside (declared input gate `symmetry_dense_J`): a batch holding symmetry-tied positions (`--pos_constraint_csv` rows with `symmetry_pos`) takes
  upstream's dense fold of J for that call — stock arithmetic, `off`'s outputs for that step, exit code unchanged — said once per call as `[caliby-opt]
  LEVER name=CALIBY_FAST_POTTS_PARAMS state=skipped reason=symmetry_dense_J impl=atom_mpnn_denoiser origin=kit …` and recorded in `opt_manifest.json` `lever_gates`.
- `CALIBY_X_SPARSE_EXACT` (`potts.py`) — the Potts energy computed on the sparse J with the stock dense energy's arithmetic. Steps aside: a call
  whose edge index it cannot use prints `[CALIBY_X_SPARSE_EXACT] … -> kit path for this call`.
- `CALIBY_X_LCP` (replaces `chroma/layers/complexity.py`) — the low-complexity penalty's entropy lines as one fused Triton kernel, float32, bitwise
  equal to the lines it replaces on compute capability 8.0 and 9.0 (its divisions never enter the subnormal range where `tl.div_rn` and torch's
  float32 division differ); compiled at first call, cached under `TRITON_CACHE_DIR`; launch geometry fixed (`_X_LCP_GEOMETRY`). Alphabets wider than
  32, non-fp32 or CPU tensors run the stock lines. A kernel that cannot build or launch (no triton, no C compiler, a JIT or launch error) raises
  `[CALIBY_X_LCP] fused LCP kernel could not build/launch on this device: <exc>` at the first served call and the process ends NOT ACTIVE.
- `CALIBY_X_MULTISEQ` (`potts.py`, `atom_mpnn_denoiser.py`; `opt/caliby_opt/multiseq.py`) — a batch's `num_seqs_per_pdb` sampler calls issued
  concurrently, one CUDA stream, one CUDA graph and one Philox generator per sequence, each generator positioned where the serial call would draw:
  the serial calls' sequences, energies and CIF bytes. Steps aside: CPU tensors, chromatic proposal or a rejection step → `[CALIBY_X_MULTISEQ] … -> serial fallback`.
- `CALIBY_X_CLEAN` (replaces `caliby/api.py`) — `loader`: with more than one clean worker the stock `clean_pdb` runs per file on forked torch
  DataLoader workers; `0`, or one clean worker, is the stock serial loop. Steps aside: a worker error → `[CALIBY_X_CLEAN=loader] … -> serial fallback`.
- `CALIBY_X_BG_CIF` (replaces `caliby/eval/eval_utils/seq_des_utils.py`; `opt/caliby_opt/bg_writers.py`) — `fork` | `thread` | `0`: each batch's
  CIFs serialised and written by background workers while the GPU samples the next batch (same `to_cif_string`, same bytes); every write joined
  before `run_seq_des` returns; a failed write is `[CALIBY_X_BG_CIF] … failed` on stderr and a non-zero exit, never a rewrite or a drop.
- `CALIBY_X_CIF_WORKERS` (`seq_des_utils.py`) — N background writers: a batch's CIFs split into at most N balanced chunks of ≥ 8 files, at most 2N
  chunks outstanding (the producer blocks rather than buffers); unset or `0` = one writer.

## Every mode

- No stock exceptions and no upstream fixes: `off`, `fast` and `exact` run the three upstream packages as shipped; the model, weights, sweep
  schedule, seeds, batching and output files are upstream's on every mode.
- Lines only this file documents: stock prints `[caliby-opt] STOCK mode=off variant=<variant> tree=stock digest=<16 hex> switches=none`; `check`
  prints `[caliby-opt] CHECK … -> would activate` (exit 0) or `… -> would refuse: <reason>` (exit 3); a GPU other than `MODEL_OPT_TARGET_GPU`, a missing
  compiler or a torch / triton off the pinned stack are `MISMATCH:` / `notes=` on the ACTIVE line, acted on by nothing; every design process also
  prints the `[caliby-opt]` STACK / OUTPUTS_WRITTEN / PEAK / EXIT lines (`opt/caliby_opt/report.py`).
- Refusals (`[caliby-opt] NOT ACTIVE: <reason>`, exit 3): the shared core absent or older than the pin in `opt/pyproject.toml`
  (`reason=core_missing:opt_core` / `core_mismatch` / `core_pin_unreadable`); upstream not at its pin; the installed tree not the pinned upstream
  (`NOT STOCK: …`, naming the reinstall that puts it back); `MODEL_PARAMS_DIR` unset or a weight file missing; no CUDA GPU (kit modes); kit switches
  pre-set that differ from or extend the set (`off`: any kit switch); `enable()` after a replaced module was imported.
- A lever that could not run (the `->` lines above, or the LCP kernel error) ends the design process `[caliby-opt] NOT ACTIVE: <mode> ran short of its
  lever set — <lever>: <record>, …; exit 3` (EXIT line `fallback_lines=` / `lcp_kernel=disabled(…)` / `partial=`; `opt_manifest.json` `levers_fallback`);
  loaded lever modules that are not all the mode's files end it the same way (`tree=mixed`). Outputs short of the request: `incomplete: <n>/<m>`, exit 1.

## Switches

- `--mode` / `CALIBY_OPT`, `--variant` / `CALIBY_VARIANT` — the mode and variant; on your own script `CALIBY_OPT=<mode>` activates the kit at the first
  `import caliby` (or `caliby_opt.enable(...)` before it) and the two rules above end that process with exit 3 from an exit hook; `off` or unset runs stock.
- `--clean_workers N` — N > 1 selects `ROWS["single/loader"]` under `fast` on `single`; elsewhere upstream's own parallel clean.
- `--det 0|1`, `--seed S` — upstream's deterministic recipe and seed, identical on every mode (STOCK.md §How stock is run). No switch runs a mode
  with a subset of its levers; a malformed `CALIBY_*` value is a usage error, never read as off (`opt/caliby_opt/env_words.py`).
