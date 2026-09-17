# mpnn_exact_worker — the exact (bit-identical) executable of the ProteinMPNN kit

One executable that reproduces the pinned upstream command line byte for byte — every sampled sequence (the `.fa` bytes), every per-residue
log-probability and sampling probability, the scores and the chain order — in less time and in memory linear in the number of residues:

| file | reproduces | run by |
|---|---|---|
| `addon/mpnn_worker2.py` | `protein_mpnn_run.py` of dauparas/ProteinMPNN @ `8907e667` (imports its `protein_mpnn_utils` unmodified from `$MPNN_DIR`) | `--mode exact`, variants `soluble` / `vanilla`: staged as `kit/mpnn_worker2.py` beside `../mpnn_pdb_parser/kit/fast_parse.py`; the package inlines its low-memory featuriser (`opt/proteinmpnn_opt/lowmem.py`) and runs the result as `kit/mpnn_worker2_lowmem.py` |

The package (`proteinmpnn_opt`; `run.sh design --mode exact …`) stages `addon/mpnn_worker2.py` as `kit/mpnn_worker2.py` beside `../mpnn_pdb_parser/kit/fast_parse.py` — the `kit/` paths of the block below — and launches it as a subprocess; nothing here is imported in-process.
The command row below is the mode table's source: `opt/proteinmpnn_opt/modes.py` reads it, and the hardware note's `Observed:`
sentence, from this file by line number (`EXACT_ROWS`, `PROBE_ROW`) — keep their positions when editing this file.

## ProteinMPNN / SolubleMPNN — `addon/mpnn_worker2.py`

| lever (flag) | what it does | why the outputs do not move |
|---|---|---|
| `--mode stream` | one persistent process replays the stock command's RNG stream over the whole jsonl: each backbone's Philox generator is positioned at the offset the one-process stock run reaches for it | same seed, same offsets, same draws |
| `--bb_batch K --sort_by_length` | K backbones (length-sorted, padded) decoded per forward, each on its own generator | the batch never enters a backbone's arithmetic (next row) |
| `--x_all` = `--chunk_gemm --cache_enc_ctx --graph_rng --single_graph --fused_draw --analytic_offsets --stock_shape_enc` | every decode-step GEMM and reduction at the stock per-backbone shape (8 clone rows x 48 neighbours); the decoder's encoder context assembled per decoded position from h_V / h_E (never for all positions at once) with its masks indexed directly; the WHOLE autoregressive decode step (3 decoder layers, output head, softmax, the per-backbone `torch.multinomial` draw on registered per-backbone generators, the state commit) captured in ONE CUDA graph per batch and replayed L_max times with a device-side step counter; each backbone's stream offset computed from per-call Philox increments; the encoder and the two scoring forwards per backbone on exactly the stock tensors | same shapes, so the libraries select the same kernels as the stock process on the same machine; graph replay runs the captured kernels; extra draws for already-finished shorter backbones touch only their own generator, which is reset analytically; the rest is data movement |
| `--hybrid_gemm` (probe-gated) | the three message GEMMs of each decoder layer (M = 8*48 rows per backbone) run on all backbones of a batch in one call; the M = 8 GEMMs (position-wise feed-forward, output head) stay per backbone | applied only when an on-device start-up probe of every decode-step cell of the job finds the batched call bit-identical to the per-backbone call (hardware note below); `--hybrid_gemm 0` leaves it out |

Without a CUDA device the worker switches `--graph_rng --single_graph --fused_draw` off itself (the eager path and torch's own draws, same outputs). Gapped backbones: a decode
step is skipped only when all clone rows sit on a missing residue — the stock rule, reproduced.

```bash
# 1) parse PDBs exactly like the stock helper (read-once parser; byte-identical parsed.jsonl)
python kit/fast_parse.py --input_path pdbs/ --output_path parsed.jsonl
# 2) design: protein_mpnn_run.py's arguments for the supported subset (below); outputs in the stock layout (seqs/, scores/, probs/)
python kit/mpnn_worker2.py --jsonl_path parsed.jsonl --chain_id_jsonl assigned.jsonl --out_folder out \
   --path_to_model_weights $MPNN_DIR/soluble_model_weights --model_name v_48_020 \
   --num_seq_per_target 8 --batch_size 8 --sampling_temp 0.1 --seed 37 --mode stream --bb_batch 16 --sort_by_length --x_all
```
Stock options served, as protein_mpnn_run.py reads them: `--seed` (0 draws one per process as upstream draws it), `--num_seq_per_target N --batch_size B`
(N // B batches of B per temperature), `--sampling_temp "T1 T2 …"`, `--omit_AAs`, `--fixed_positions_jsonl`, `--chain_id_jsonl`, `--omit_AA_jsonl`,
`--bias_AA_jsonl`, `--bias_by_res_jsonl`, `--pssm_jsonl` with `--pssm_multi` / `--pssm_threshold` / `--pssm_log_odds_flag` / `--pssm_bias_flag`,
`--backbone_noise 0`, `--save_score` / `--save_probs` (upstream's write gates: 0 = arrays computed, files not written), `--path_to_model_weights` /
`--model_name`. Not served (the package refuses such a pass by name, exit 3, before launching anything; `--mode off` runs it): `--ca_only`, and when
set `--score_only`, `--conditional_probs_only`, `--unconditional_probs_only`, `--tied_positions_jsonl`; `--backbone_noise` above 0.
The worker prints `DEVICE CELL: <gpu> sm_XX torch … cuda …` first, `hybrid_gemm: PROBE PASS|FAIL -> …` once at start-up when `--hybrid_gemm`
is given, and one timing JSON line at the end (also written as `worker_timing.json` beside the outputs).

## Hardware note (exactness across GPU models)
Equality with the stock command's bytes is obtained by executing every GEMM / reduction on tensors of the stock shape (8 clone rows x 48
neighbours per backbone), so the libraries pick the same kernels as in the stock process on the same machine; this holds on any GPU and is
what `--x_all` does. The optional `--hybrid_gemm` additionally batches the three message GEMMs of each decoder layer over backbones; that is
result-neutral only if the kernel cuBLAS selects for the larger M accumulates identically — a property of the full execution cell (GPU model /
sm arch / torch + CUDA / cuBLAS version / dtype / step shapes). When `--hybrid_gemm` is requested the worker therefore probes, once at start-up
and before any output, every (B, K, Kn) decode-step cell the job will use (each decoder layer batched vs. the per-backbone call on random
tensors of exactly those shapes, private RNG) and prints one line `hybrid_gemm: PROBE PASS -> using hybrid ...` or `hybrid_gemm: PROBE FAIL
...` then `hybrid_gemm: REFUSED` (nothing designed, exit 3: `--hybrid_gemm 0` runs the line without the lever); the verdict is recorded in
`worker_timing.json["hybrid_gemm_probe"]` (cell description + per-shape results). On a card with a row ceiling in the worker's
`HYBRID_GROUP_ROWS` (sm_80: 112 rows, i.e. 14 backbones at B = 8, so a 16-backbone batch runs as two groups of 8; larger groups select a
different sgemm kernel there and differ by ~1e-6) the batched GEMM serves the batch in groups under the ceiling and the probe judges those
group cells. The probe is judged for the job's own `--batch_size`; the cells listed here are B = 8.
Observed: H100 80GB HBM3 (sm_90), H200, B200, A100-SXM4-80GB (sm_80): PASS; L40S (sm_89): FAIL -> refused. Stock outputs are per
machine: stock itself is not bit-identical across GPU models or library versions.

## Licences
`addon/mpnn_worker2.py` derives from ProteinMPNN (MIT, Copyright Justas Dauparas) — attributions in `NOTICE`, upstream's licence text in `../mpnn_pdb_parser/licenses/`. No model
weights and no upstream source tree are redistributed here: ProteinMPNN and its repository weights come from the user's checkout of the pinned
commit (`MPNN_DIR`).
