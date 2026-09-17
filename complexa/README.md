# Proteina-Complexa — optimization kit

Drop-in modes that make stock Proteina-Complexa 1.1.0 binder generation (`complexa generate`) faster and lighter on GPU
memory. You call Proteina-Complexa exactly as before; the kit adds a `--mode`:

- `off` — stock Proteina-Complexa, exactly as released ('stock' below always means this unmodified upstream release).
- `exact` — identical outputs, faster.
- `fast` — small, documented numeric differences, faster still. **The default.**
- `big` — lowest GPU memory, for large inputs.

What each optimization changes: `CHANGES.md`. Exact versions, the pinned software stack and all variables: `STOCK.md`.
How the three setup routes (A — Docker, B — Apptainer, C — Python venv) work in general: the top-level `README.md`.

**At a glance** (H100 80 GB vs stock): `exact` identical outputs, faster than stock · `fast` faster still, within stock's seed-to-seed variation · `big` lowest peak GPU memory for large inputs.

## Setup

Pick ONE way to get the pinned stack — stock Proteina-Complexa 1.1.0 and everything it needs (the 'Stack' section of
`STOCK.md` lists every pin): **A — Docker**, **B — Apptainer**, or **C — a Python venv on your own host**.

Every route needs:

- a Linux x86_64 host with NVIDIA driver 560 or newer (data-centre driver: 525 or newer).

Route C additionally needs:

- a released CPython 3.12 with `libxrender1` and `libxext6` installed;
- `LOCAL_CODE_PATH` naming upstream at the pinned commit — the unpacked `stock/` archive or your own clean checkout,
  plain or editable. The pin check refuses another version, another commit or an edited checkout by name (exit 3).

Type the first block from the directory that holds `complexa/` and `common/`:

```bash
# A — Docker (recommended: carries the whole pinned stack, stock and this kit in one image)
docker build -f complexa/environment/Dockerfile -t complexa-kit:dev .
docker run --rm -it --gpus all -w /kit -v /weights:/weights -v $PWD/out:/kit/complexa/out complexa-kit:dev bash   # the steps below run in this shell
# B — Apptainer / Singularity (the whole Setup for B): converts A's image from the local Docker daemon (needs A first; no daemon at run time)
apptainer build complexa-kit.sif complexa/environment/apptainer.def
mkdir -p out $HOME/complexa_weights; kit() { apptainer run --nv --bind $HOME/complexa_weights:/weights/complexa --bind "$PWD/out":/kit/complexa/out complexa-kit.sif "$@"; }   # kit = B's ./run.sh (a shell function)
apptainer exec --bind $HOME/complexa_weights:/weights/complexa complexa-kit.sif python -m complexa_opt.weights /weights/complexa   # /weights/complexa: *.ckpt fetched or checked; read-only ok
export LOCAL_CODE_PATH=/opt/pc CKPT_PATH=/weights/complexa && kit check --config h100   # cards: --config a100|h200 instead, on every kit line
# C — instead of A or B, on your own host (venv): STOCK.md §Stack is the complete recipe
```

Route B (Apptainer) is complete at this point. Its Run lines are the same commands typed as `kit <command> …` with the
input given as an absolute host path, e.g. `kit design --config h100 --mode exact --input "$PWD/pdl1.json" --out out/exact -- $GEN`;
the `LOCAL_CODE_PATH` exported in the block is what the Run block's input file uses. Under **A** you are now in the
container shell, which opens in `/kit`. Under **C** you are in your activated environment inside `complexa/`; skip what
STOCK.md's Stack section already ran. Run:

```bash
[ -f run.sh ] || cd complexa                        # no-op once inside · C via §Stack: skip install too
export LOCAL_CODE_PATH="${LOCAL_CODE_PATH:-$HOME/src/proteina-complexa-916eaaed}"   # C: §Stack's checkout · A: image preset
export CKPT_PATH=/weights/complexa                  # must hold complexa.ckpt + complexa_ae.ckpt
bash run.sh install --weights "$CKPT_PATH"             # --weights: ≈7 GB if absent, else hash-check; read-only ok
bash run.sh check --config h100                        # pins, checkout, weight sizes, GPU, hook; loads no model
```

What the blocks assume:

- **Upstream checkout.** `LOCAL_CODE_PATH` names upstream's checkout at the pin: preset to `/opt/pc` in the image
  (route A keeps the preset; route B exports it in its block), the Stack section's checkout under C.
- **Weights directory.** `CKPT_PATH` (`/weights/complexa` inside the container under A and B; route B binds
  `$HOME/complexa_weights` there) must hold `complexa.ckpt` and `complexa_ae.ckpt`. `install --weights "$CKPT_PATH"`
  (route B: the `python -m complexa_opt.weights` line) fetches them when absent, about 7 GB, and otherwise hash-checks
  them, so a read-only copy works.
- **Outputs under A and B.** The Run examples write to `out/…`, which is the mounted `$PWD/out` on the host.
- **GPU cards.** `--config h100|a100|h200` loads `configs/h100.env` / `a100.env` / `h200.env`, which set
  `MODEL_OPT_TARGET_GPU`, `COMPLEXA_INIT` and upstream's data and tool path variables; `design` and `check` take it.

## Run

```bash
cat > pdl1.json <<EOF                    # upstream's PD-L1 target, one targets_dict entry, binder 80
{"02_PDL1": {"source": "bindcraft_targets", "target_filename": "PD-L1", "target_path": "$LOCAL_CODE_PATH/assets/target_data/bindcraft_targets/PD-L1.pdb",
             "target_input": "A1-115", "hotspot_residues": ["A37", "A39", "A49", "A98"], "binder_length": [80, 80], "pdb_id": null}}
EOF
GEN="++generation.search.algorithm=single-pass ++generation.reward_model=null ++generation.dataloader.dataset.nres.nsamples=32 ++seed=5"  # upstream's overrides: generation stage only, 32 designs
bash run.sh design --config h100 --mode off   --input pdl1.json --out out/off   -- $GEN    # stock, in a clean subprocess
bash run.sh design --config h100 --mode exact --input pdl1.json --out out/exact -- $GEN
bash run.sh design --config h100 --mode fast  --input pdl1.json --out out/fast  -- $GEN    # the default when no mode is named
bash run.sh design --config h100 --mode big --input pdl1.json --out out/big -- $GEN
```

**Options.**

- `--input` takes one target entry in the schema of upstream's `configs/targets/targets_dict.yaml`, as JSON or YAML
  (the block writes upstream's PD-L1 target as `pdl1.json`).
- Everything after `--` goes to `complexa generate` verbatim as Hydra overrides; a later override of the same key wins,
  and overrides may also stand among the kit's own flags.
- `$GEN` keeps the run to the generation stage. Without it, upstream's best-of-n search also scores
  designs with its AF2 reward model, whose parameters `install --weights` does not fetch: upstream's
  `env/download_startup.sh --af2` does, into the checkout's `community_models/ckpts/AF2`; point `AF2_DIR` there
  (STOCK.md, 'Variables').
- Without `run.sh`: `complexa-opt design --mode <mode> …` is the same call (after `source configs/h100.env`);
  `COMPLEXA_OPT=<mode>` names the mode when `--mode` is absent.

**Outputs.**

- Where stock writes them: `<out>/inference/search_binder_local_pipeline_<item>_<run>/job_*/*.pdb` (chain B is the
  binder) and `<out>/logs/`, plus `opt_manifest.json` and `design.log`.
- Give each run a fresh `--out` (or a new `++run_name=`). Repeated into the same `--out`, upstream rewrites the
  same-named files (or stops at its results CSV), no new files appear, and the run ends
  `EXIT … designs_written=0 … incomplete=designs_0/32 exit=1`.

**What a run prints** (on stderr).

- `check` prints one `[complexa-opt] DRY-RUN mode=<m> route=<r> … ok=True` line and exits 0.
- A kit mode prints `[complexa-opt] ACTIVE mode=<m> tier=<exact|tolerance> levers=<…>` once engaged (the kit calls its
  individually switchable optimizations 'levers'), one `LEVER name=… state=…` line per optimization at exit, and ends
  with `[complexa-opt] EXIT … rc=0 designs_written=32 … exit=0`.
- `--mode off` prints the banner `[complexa-opt] NOT ACTIVE: mode off (stock route) (mode=off)`, followed by the same
  `EXIT` line.
- If a mode cannot engage (a pin, a variable, the install, or an optimization that cannot be installed), the command
  prints `[complexa-opt] NOT ACTIVE: <reason>` and exits 3; it never falls back to stock silently.

**Exit codes.**

| code | meaning |
|---|---|
| 0 | finished |
| 1 | the run failed, or fewer designs were written than asked |
| 2 | usage error |
| 3 | not active: the mode was refused |

**First run.** A `design` whose checkpoints are not yet in the host's page cache (the first after boot; the weights
step's hash check also loads them) spends up to about 85 s more reading the 7 GB, so compare modes from the second run
on. At this example's size every mode reports the same `peak_alloc_gib`; `big`'s saving shows on larger targets.

## Modes

- `off` — stock `complexa generate` as released, in a clean subprocess whose environment carries nothing of the kit
  (the install's start-up hook imports nothing without `COMPLEXA_OPT`).
- `exact` — pair featurization without redundant per-step work, and the sampler's schedule read on the host instead of
  per-step device syncs. Outputs identical to `off` at the same seed and batch. Use it when outputs must not move.
- `fast` (default) — `exact` plus all layers' pair-bias LayerNorm and projection fused into one pass per network call,
  and attention through `torch.nn.functional.scaled_dot_product_attention`; fp32 re-association, within stock's
  seed-to-seed variation. Use it for throughput.
- `big` — `exact` plus each layer's pair bias computed over fixed-size row blocks with upstream's own LayerNorm and
  Linear, so the normalized pair tensor is never materialised whole; numerics as `exact` (identical outputs). Use it
  when another mode runs out of memory.

## Notes

- **Where the gain is.** The kit's gain is in the sampling stage. Each `design` call also pays about 30 s of start-up
  and checkpoint load in any mode, so the Run example gains less end to end than its sampling stage does; the gain grows
  with designs per call.
- **Other cards.** `--config a100` / `--config h200` load `configs/a100.env` / `configs/h200.env` (`h100.env` with
  `MODEL_OPT_TARGET_GPU=A100` / `H200`). Same optimizations as on H100 — every optimization of every mode engages on any
  card. The `ACTIVE` line names the card found by compute capability (`gpu_class=H100|A100|other`; an H200, compute
  capability 9.0, reads `H100`). No other card is configured.
- **`LEVER` line counters.** `served=` counts the calls an optimization handled and `fallback_stock=` those that took
  upstream's own path by design (for `pair_bias_*`: attention layers outside the wired sampling transformer, such as the
  decoder's). These are counts, not warnings; `CHANGES.md` lists the cases.
- **Binder length ranges.** A `binder_length` range `[low, high]` pads batches, so upstream's own pair assembly runs and
  the run says so with `LEVER name=pair_assembly state=skipped reason=padded_batches` and
  `EXIT … gated=pair_assembly:padded_batches`; outputs and exit code are unaffected. A fixed length `[L, L]` engages it.
