# CALIBY_XATTEMPT_ADDON_v1 — HOWTO

Add-on to **CALIBY_FAST_INFERENCE_KIT_v1** (the partner kit, `../fast_inference/`; credit: its patches 0002-0003 are the base of `fast/potts.py` and
`fast/atom_mpnn_denoiser.py`). Pinned stack: `<release>/caliby/environment/requirements.lock` (Python 3.12, torch 2.6.0+cu124, triton 3.2.0, caliby
@ 41d31560, atomworks-caliby @ ea2c998a, protpardelle-1c @ 7962da09, biotite 1.6.0); weights under `$MODEL_PARAMS_DIR` (`caliby/<ckpt>.ckpt`;
`protpardelle-1c/` for ensemble-conditioned design). Every Caliby sampling default is untouched (500 DLMC sweeps, anneal 1.0 → 0.01, LCP
regularisation, batch_size 4, num_workers 2, nothing omitted); the model, weights, sweeps, seeds, batching and output files are not changed by any lever.
Tier 1 = the same outputs as stock, byte for byte; tier 2 = deterministic, last-bit Potts-energy differences may change an occasional sampled
residue (the partner kit's `CALIBY_FAST_POTTS_PARAMS=2` without `CALIBY_X_SPARSE_EXACT`).

## What the add-on changes (all environment switches read at call time; unset = the partner kit's / stock code path)
| # | file (vs caliby@41d31560 site-packages) | lever | switch | tier |
|---|---|---|---|---|
| X001 | caliby/model/seq_denoiser/denoisers/seq_design/potts.py (= kit 0002 + add-on) | **L-A exact sparse energy**: with `CALIBY_FAST_POTTS_PARAMS=2` (kit: sparse 48-neighbour J, no N x N densification) the per-sweep energy rebuilds the STOCK dense gathered tensor J_ij (B,N,N,C) bit-for-bit from the sparse couplings (zeros + one row write per edge; values `(0.0+J)*((mask_ij>0)*(mask_i[i]*mask_i[j]))` exactly as `_fold_symmetry_pos` (index_add_) and `J*mask_ij` produce them) and reduces it with the same op `J_ij.sum(2)` → U_i, U equal to the dense (stock) path to the last bit. The (B,N,N,22,22) dense J is never built or gathered from. Guard: edge_idx rows distinct + in range (checked once per call on device), else the kit path. | `CALIBY_X_SPARSE_EXACT=2` (1 = fresh zeros buffer per sweep; 2 = persistent buffer, zero_() per sweep) | 1 |
| X004 | chroma/layers/complexity.py | **L-D fused exact LCP entropy**: inside `complexity_lcp` the (B,N,30,20,20) N_ij tensor + `estimate_entropy` (7 elementwise kernels + 2 reductions) are replaced by one Triton kernel that replays the same fp32 op sequence per row with IEEE add/mul/div, libdevice logf (equal to torch.log over every positive finite fp32) and ATen's exact last-dim reduction tree for a 20-wide fp32 row; the launch geometry is 8 rows per program × 2 warps on every card (`_X_LCP_GEOMETRY`, chosen by a launch-geometry sweep on H100 and A100; speed only, never the bits). U_ij is detached in the stock code, so the autograd graph that produces the penalty gradient is unchanged and consumes the same U_ij. | `CALIBY_X_LCP=1` | 1 (on the pinned stack; guarded) |
| X002 | caliby/api.py | **L-B parallel clean without loky**: `clean_pdbs(num_workers>1)` runs the stock `clean_pdb` per file on `num_workers` torch DataLoader worker processes forked from the calling process (the mechanism caliby's own `InferenceDataLoader` uses), results in input order; each worker pays the lazy CCD template load once. Driver: `--clean_workers N`. Any worker error prints `[CALIBY_X_CLEAN=loader] … -> serial fallback` and the stock serial loop runs. **L-G Protpardelle model cache** (same file): `generate_ensembles` memoises the Protpardelle-1c model across input structures instead of re-loading the checkpoint per structure. | `CALIBY_X_CLEAN=loader`; `CALIBY_X_PP_CACHE=1` | 1 (CIF bytes identical except biotite `_entry.date/_entry.time`) |
| X003 | caliby/eval/eval_utils/seq_des_utils.py | **L-C background CIF serialisation**: per batch, the output CIFs are serialised/written by background workers (forked processes, or threads: `caliby_opt.bg_writers` over `opt_core.host.outputs.AsyncWriter`; their count is X011's `CALIBY_X_CIF_WORKERS`, 1 when unset) while the GPU samples the next batch; every write joined before `run_seq_des` returns, a failed write named on stderr and raised (non-zero exit). Same `to_cif_string` on the same objects → same bytes. | `CALIBY_X_BG_CIF=fork` (or `thread`) | 1 |
| X010 | potts.py + atom_mpnn_denoiser.py | **concurrent multi-sequence sampling**: the `num_seqs_per_pdb` sampler calls of a loader batch (the serial loop in `AtomMPNNDenoiser.potts_sample`) issued concurrently: one CUDA stream and one Philox generator per sequence, sweep i of all sequences captured in ONE CUDA graph (S independent chains) replayed 500 times (`sample_potts_multiseq`). Sequence s draws its initial sequence from the default generator positioned where the serial loop's call s starts and its 500 sweeps from a generator of its own positioned right after (`opt_core.diffusion_loop.rng`: the Philox increment of the one draw op is measured on the call's shapes, never assumed); the default generator is left where the serial loop leaves it. Same kernels, same operand shapes, same draws ⇒ same sequences, U and CIF bytes as the serial calls. Rides `CALIBY_FAST_SAMPLER=2`; a call outside the fast sampler's graph path (CPU tensors, chromatic proposal, rejection step) runs the serial calls and prints `[CALIBY_X_MULTISEQ] <reason> -> serial fallback`. Memory: one graph pool with S sweeps' temporaries (with `CALIBY_X_SPARSE_EXACT=2`, S (B,N,N,C) fp32 buffers). | `CALIBY_X_MULTISEQ=1` | 1 |
| X011 | seq_des_utils.py | **parallel CIF writers**: N background CIF writers of the `CALIBY_X_BG_CIF` mode: a batch's output CIFs are split into at most N balanced chunks of ≥ 8 files, one submit per chunk, at most 2N chunks outstanding; N unset/0 = 1 worker. Same `to_cif_string` on the same objects: same bytes. Every write joins before `run_seq_des` returns; a write that fails is `[CALIBY_X_BG_CIF] ... failed` on stderr and the core's RuntimeError (the process exits non-zero) — never a rewrite, never a drop. A malformed value of either switch is a usage error, never silently 0. | `CALIBY_X_CIF_WORKERS=N` | 1 |

X010 / X011 ship in the `fast/` files (no separate patch); the ensemble-only levers are the next section's table.

## Enable
The release package exports these lines (its `modes.py` rows); by hand, in the environment of any caliby process on a tree carrying `fast/`:
```
export MODEL_PARAMS_DIR=/path/to/model_params          # contains caliby/<ckpt>.ckpt
export CALIBY_FAST_SAMPLER=2 CALIBY_FAST_POTTS_PARAMS=2 CALIBY_X_SPARSE_EXACT=2 CALIBY_X_LCP=1 CALIBY_X_CLEAN=0 CALIBY_X_BG_CIF=fork CALIBY_X_MULTISEQ=1 CALIBY_X_CIF_WORKERS=8   # row "X": serial clean
python tests/xcaliby_design.py --inputs in/*.pdb --out_dir out/run1 --num_seqs_per_pdb 4 --seed 11
#   row "XL" = the same line with CALIBY_X_CLEAN=loader and --clean_workers N>1 (the parallel clean; the same output bytes). The clean lever has a fixed
#   cost per worker (process fork + CCD template load) and pays off only with many input structures per call; with a handful of inputs keep one clean worker.
#   (or any caliby entry point: caliby.load_model("soluble_caliby").sample(...), examples/scripts/seq_des*.sh)
```
GPU-only subset (no CPU-side levers): drop `CALIBY_X_CLEAN`, `CALIBY_X_BG_CIF`, `--clean_workers`. With the fused LCP kernel off (pure PyTorch, no
Triton): the same line with `CALIBY_X_LCP=0`. (These are this add-on's switches by hand; the release's modes export whole rows only.)

## Ensemble-conditioned design (32 conformers per structure): levers L-E, L-F, L-G, L-H, L-I
| # | switch | file / patch | what | tier |
|---|---|---|---|---|
| L-E | `CALIBY_X_TIED_DET=1` | atom_mpnn_denoiser.py / X006 | tied-group aggregation of the Potts field h over the conformers of an ensemble done by sequential per-member adds on the GPU (= CPU `index_add_` order) instead of CUDA `index_add` (atomicAdd, run-to-run nondeterministic). **Required for tier 1 in ensemble mode whenever `CALIBY_FAST_POTTS_PARAMS>=1`** (the partner kit's GPU-side aggregation; stock aggregates on the CPU and is deterministic); no effect in single-structure design (groups of size 1). | 1 |
| L-F | `CALIBY_X_ENS_WORKERS=N` | inference_dataloader.py / X007 | featurise the N conformers of an ensemble on N loader workers in parallel and collate per structure in the parent, in index order (stock: one worker featurises all 32 conformers serially while the GPU idles) | 1 |
| L-G | `CALIBY_X_PP_CACHE=1` | api.py / X002 | memoise the Protpardelle-1c model across `generate_ensembles` inputs (stock re-loads the checkpoint for every input structure) | 1 |
| L-H | `CALIBY_X_PP_NOSYNC=1` | protpardelle `core/models.py` / X008 (vs protpardelle-1c@7962da09) | Protpardelle-1c denoise loop: keep the 4 per-step trajectory snapshots on the GPU (clone) and copy them to the host once after the loop, instead of 4 blocking `.cpu()` per step | 1 |
| L-I | `CALIBY_X_PP_FASTPDB=1` | protpardelle `data/pdb_io.py` / X009 | `bb_coords_to_pdb_str`: one host copy per tensor + identical f-string formatting instead of several single-element (CUDA) tensor reads per ATOM line; byte-identical PDB text; uncovered inputs fall back to the stock loop | 1 |

L-H / L-I are served only if `protpardelle` is importable and its two files match protpardelle-1c@7962da09 (the commit caliby pins); otherwise skipped with a message.
Ensemble command: `python tests/xcaliby_design.py --inputs in/*.pdb --out_dir out/e32 --num_seqs_per_pdb 4 --seed 11 --ensemble` with
`CALIBY_FAST_SAMPLER=2 CALIBY_FAST_POTTS_PARAMS=2 CALIBY_X_SPARSE_EXACT=2 CALIBY_X_LCP=1 CALIBY_X_TIED_DET=1 CALIBY_X_ENS_WORKERS=8 CALIBY_X_PP_CACHE=1 CALIBY_X_BG_CIF=fork CALIBY_X_PP_NOSYNC=1 CALIBY_X_PP_FASTPDB=1 CALIBY_X_MULTISEQ=1 CALIBY_X_CIF_WORKERS=8`
(the release's `--mode exact --variant ensemble32`). What is NOT accelerated: the Protpardelle-1c partial-diffusion GPU loop itself (launch-bound).

## Where the levers do NOT apply (fall back automatically to the kit/stock path)
- L-A: symmetry groups present (kit level 2 densifies in that case), CPU tensors, duplicate/out-of-range neighbour indices.
- L-D: alphabet size > 32, non-fp32, CPU → stock lines (routing, silent). A kernel that cannot build or launch (triton missing, no C compiler, a JIT/launch
  error) does NOT fall back: the first served call raises `XLcpKernelError: [CALIBY_X_LCP] fused LCP kernel could not build/launch on this device: <exc> …`,
  the failure is recorded in the module (`_X_LCP_DISABLED_REASON`), and the release's design process ends NOT ACTIVE by name (a mode is all of its
  levers). Tested on torch 2.6.0+cu124 / triton 3.2.0 (H100, A100); on another stack the kernel builds with what is there or stops the same way.
- L-E/L-F/L-G/L-H/L-I: only touch ensemble-conditioned sampling (`ensemble_sample` / `generate_ensembles` / Protpardelle-1c); single-structure runs are unaffected.
  L-H holds the trajectory on the GPU until the batch ends (150 x B x L x 37 x 3 fp32; about 0.6 GB at B=16, L=550).
- L-B: `num_workers=1` / `--clean_workers 1` (caliby's default) = the stock serial path regardless of the switch.

## By-hand install into a source checkout
`patch -p1 < patches/X00{1,2,3,4}-*.vs_upstream.patch` (then X006, X007) on caliby @ 41d31560 and `patches/X008-*`, `X009-*` on protpardelle-1c @ 7962da09 — all
apply cleanly and produce the `fast/` files; X001 / X006 `.vs_upstream` already contain the partner kit's 0002 / 0003, use the `.vs_kit0002` / `.vs_kit0003`
forms on a kit-patched tree. The release package never writes site-packages: it serves `fast/` by import hook over an installed tree proven to be the pin.
