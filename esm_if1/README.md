# ESM-IF1 — optimization kit

Drop-in modes that make stock ESM-IF1 sequence design (fair-esm 2.0.1 at commit `2b369911`,
`examples/inverse_folding/sample_sequences.py`) faster. You call the script exactly as before; the kit adds a `--mode`:

- `off` — stock ESM-IF1, exactly as released ('stock' below always means this unmodified upstream release).
- `fast` — small, documented numeric differences, faster: seeded, batched sampling from the same distribution as stock,
  not the same draws. **The default.**

What each optimization changes: `CHANGES.md`. Exact versions, the pinned software stack and all variables: `STOCK.md`.
How the three setup routes (A — Docker, B — Apptainer, C — Python venv) work in general: the top-level `README.md`.

**At a glance** (H100 80 GB vs stock): `fast` small documented numeric differences, faster than stock.

## Setup

Pick ONE way to get the pinned stack — fair-esm 2.0.1 at commit `2b369911` and everything it needs (the 'Stack'
section of `STOCK.md` lists every pin): **A — Docker**, **B — Apptainer**, or **C — a Python venv on your own host**.

Every route needs:

- an NVIDIA driver ≥ 530 on the host.

Route C additionally needs:

- Linux x86-64 with a released CPython 3.11; no CUDA toolkit (the wheels carry it);
- ≈3 GB of downloads (≈2–3 min).

Type the first block from the directory that holds `esm_if1/` and `common/`:

```bash
# A — Docker (recommended: one image with the whole pinned stack, stock and this kit); the shell opens in /kit
docker build -f esm_if1/environment/Dockerfile -t esm_if1-kit:dev .
docker run --rm -it --gpus all -v /path/to/weights:/weights -v $PWD/out:/kit/esm_if1/out -w /kit esm_if1-kit:dev bash   # /path/to/weights holds esm_if1/*.pt
# B — Apptainer / Singularity (the whole Setup for B): converts A's image (no Docker daemon at run time); kit() is B's bash run.sh, typed on the host
apptainer build esm_if1-kit.sif esm_if1/environment/apptainer.def
mkdir -p out; kit() { apptainer run --nv --bind /path/to/weights:/weights --bind "$PWD/out":/kit/esm_if1/out esm_if1-kit.sif "$@"; }   # kit = B's ./run.sh · /path/to/weights holds esm_if1/*.pt
export ESM_IF1_WEIGHTS=/weights/esm_if1/esm_if1_gvp4_t16_142M_UR50.pt && kit install --weights /weights/esm_if1   # /weights/esm_if1: the .pt, fetched or checked; read-only ok
kit check --config h100 --mode fast              # Run lines alike: kit design … · cards: --config a100|h200
#     B: input structures outside $HOME, $PWD and /tmp need one more --bind inside kit().
# C — instead of A or B, on your own host (venv): STOCK.md §Stack is the complete recipe
```

Route B (Apptainer) is complete at this point; its Run lines are the same commands typed as `kit <command> …`
(`kit design …`). Under **A** you are now in the container shell, which opens in `/kit`. Under **C** you are in your
activated environment inside `esm_if1/`; STOCK.md's Stack section already ran the install line, so resume at `export`. Run:

```bash
[ -f run.sh ] || cd esm_if1                      # no-op once inside · C via §Stack: resume at export
bash run.sh install --weights /weights/esm_if1      # kit + pin check; --weights DIR: .pt fetched or checked
export ESM_IF1_WEIGHTS=/weights/esm_if1/esm_if1_gvp4_t16_142M_UR50.pt   # optional: the .pt file's path
bash run.sh check --config h100 --mode fast         # versions, the checkpoint's sha256, the GPU; designs nothing
```

What the blocks assume:

- **Weights file.** `install --weights DIR` fetches the 1.7 GB checkpoint into `DIR` when absent and otherwise checks it
  there (`/weights/esm_if1` inside the container under A and B; the host directory you mount at `/weights` holds
  `esm_if1/*.pt`). `ESM_IF1_WEIGHTS` is optional and names the `.pt` file's path; on any route it may name a read-only copy.
- **Outputs under A and B.** The Run examples write to `out/…`, which is the mounted `$PWD/out` on the host.
- **Inputs under B.** Input structures outside `$HOME`, `$PWD` and `/tmp` need one more `--bind` inside `kit()`.
- **Pin check.** `bash run.sh install` accepts only the pinned fair-esm (installed files equal to
  `stock/fair-esm-2b369911.tar.gz`) and names anything else it finds, PyPI's 2.0.0 included.
- **What `check` prints.** The checkpoint's size and digest beside the expected ones, then
  `[esm_if1-opt] DRY-RUN mode=fast … ok=True` (exit 0; exit 3 when fair-esm or torch is missing). It designs nothing.
- **GPU cards.** `--config h100|a100|h200` loads `configs/h100.env`, `configs/a100.env` or `configs/h200.env` on every
  command. The file sets the one writable per-user path, `$HOME/.cache/esm_if1_opt` with `torch_home/` inside, and the
  GPU class `check` compares against. All variables: `STOCK.md`.

## Run

Under B there is no checkout on the host, so extract the example structure from the image instead of typing the first
line below, then type each line as `kit design …`:
`apptainer exec --bind /tmp esm_if1-kit.sif tar -xzf /kit/esm_if1/stock/fair-esm-2b369911.tar.gz -C /tmp examples/inverse_folding/data/5YH2.pdb`

```bash
tar -xzf stock/fair-esm-2b369911.tar.gz -C /tmp examples/inverse_folding/data/5YH2.pdb   # upstream's example structure, from the stock archive
D=/tmp/examples/inverse_folding/data; X=$D/5YH2.pdb
bash run.sh design --config h100 --mode off  $X --chain C --num-samples 8 --outpath out/off/5YH2_C.fasta    # stock, in a clean subprocess
bash run.sh design --config h100 --mode fast $X --chain C --num-samples 8 --outpath out/fast/5YH2_C.fasta   # the default when no mode is named
bash run.sh design --config h100 --mode fast $X --chain C --multichain-backbone --num-samples 8 --outpath out/fast/5YH2_C_complex.fasta
bash run.sh design --config h100 --mode fast --input $D --out out/run --chain C --num-samples 8 --seed 37 --batch_size 64   # a directory of backbones
```

**Options.** `design` takes `sample_sequences.py`'s own arguments unchanged: `PDBFILE` (a `.pdb` or `.cif` file),
`--chain`, `--temperature`, `--num-samples`, `--outpath`, `--multichain-backbone`, `--nogpu`. The kit adds:

- `--mode`;
- `--input <dir|file>` with `--out DIR` — every `*.pdb` / `*.cif` of the directory in name order, one FASTA each at
  `DIR/seqs/<input stem>.fasta`;
- `--seed N` (default 37);
- `--batch_size B` (default 64);
- `--upstream-fix` (see 'Known upstream issues').

`esm_if1-opt design …` is the same command without `run.sh`.

**Outputs.** Beside the sequences (records `>sampled_seq_<i>`, as stock writes them) each run leaves
`opt_manifest.json` and `timing.jsonl`.

**What a run prints** (on stderr).

- `fast` prints `[esm_if1-opt] ACTIVE mode=fast route=kit … batch_size=64 seed=37 …` once it engages.
- `off` prints `[esm_if1-opt] NOT ACTIVE: mode off (stock route) (mode=off)` and runs stock.
- If the mode cannot run here, the command prints `NOT ACTIVE: <reason>` and stops (exit 3); it never falls back to stock.
- Every run ends with `[esm_if1-opt] EXIT … exit=<code>`.

**Exit codes.**

| code | meaning |
|---|---|
| 0 | done |
| 1 | failed, or a structure lacks sequences (`incomplete=k/n`) |
| 2 | usage error, an unknown mode word included |
| 3 | the mode cannot run here (`NOT ACTIVE: <reason>`) |

## Modes

- `off` — stock `sample_sequences.py` as released, one subprocess per structure whose environment is cleared of the
  kit's variables and checked before torch is imported; unseeded. `--seed` and `--batch_size` are reported `NOT APPLIED`.
- `fast` (default) — same distribution as stock, different draws, repeatable at a fixed seed and batch size: one seeded
  process, the model loaded once, `--batch_size` (backbone, sample) rows per forward pass, single- and multichain.

## Known upstream issues

Fixes are opt-in with `--upstream-fix <ID>` on every mode, `off` included; the run's
`[esm_if1-opt] UPSTREAM-FIX <ID> applied (…)` line names the ones applied.

| ID | what it fixes | default |
|---|---|---|
| `ESMIF1-001` | `sample_sequences.py --nogpu` keeps the model on the CPU but samples with `device=cuda` hard-wired (single-chain route) and fails before the first sequence; with the fix `model.sample()` uses the model's own device | off |

## Notes

- **Where the gain is.** The kit's gain is in steady-state design throughput. Every call also pays ≈5 s of start-up and
  checkpoint load in either mode, so the 8-sample example above already ends sooner under `fast`, and many samples of one
  chain far sooner — the gain grows with samples and backbones per call.
- **A100, H200.** `--config a100` / `--config h200` loads that card's file, which sets the GPU class `check` reports
  against and nothing else. No optimization depends on the card: `fast` engages on the GPU torch sees, or on the CPU
  with `--nogpu`. No other card is configured.
- **Out of memory under `fast`.** The batch that does not fit ends the pass with torch's error and
  `EXIT … incomplete=k/n exit=1` (structures finished before it keep their FASTA files). Nothing falls back to a smaller
  batch — lower `--batch_size` and rerun.
- **Expected warning.** Upstream's `UserWarning: Regression weights not found, predicting contacts will not produce correct results.`
  at every model load, under both modes, is expected: the checkpoint has no contact head and nothing here predicts contacts.
- **Determinism.** `--det 1` has nothing to switch on this model and is reported `NOT APPLIED`: `fast` repeats by the
  seed alone. Floating-point values are not bitwise reproducible run to run under either mode; the sampled sequences are.
