# ProGen2 — optimization kit

Drop-in modes that make stock ProGen2 faster — `salesforce/progen` @ `c27a419c`: `sample.py` generation and
`likelihood.py` scoring, all seven released sizes. You call it exactly as before; the kit adds a `--mode`:

- `off` — stock ProGen2, exactly as released ('stock' below always means this unmodified upstream release).
- `exact` — identical outputs, faster. **The default.**

What each mode changes: `CHANGES.md`. Exact versions, the pinned software stack and all variables: `STOCK.md`.
How the three setup routes (A — Docker, B — Apptainer, C — Python venv) work in general: the top-level `README.md`.

**At a glance** (H100 80 GB vs stock): `exact` identical outputs, faster than stock.

## Setup

Pick ONE way to get the pinned stack — Python 3.9, torch 2.8.0+cu128, transformers 4.16.2, tokenizers 0.10.3, with
stock in-tree under `stock/src/progen2` (the 'Stack' section of `STOCK.md` lists every pin): **A — Docker**,
**B — Apptainer**, or **C — a Python venv on your own host**.

Every route needs:

- an NVIDIA driver ≥ 570 and, for `exact`, a GPU of compute capability ≥ 8.0; no CUDA toolkit;
- a weights directory, `PROGEN2_WEIGHTS`: a directory of `<progen2-size>/{pytorch_model.bin, config.json}`, written
  `/path/to/weights` on the host side of the blocks and mounted at `/weights`.

Type the first block from the directory that holds `progen2/`:

```bash
# A — Docker (recommended: the whole pinned stack, stock and this kit in one image)
docker build -f progen2/environment/Dockerfile -t progen2-kit:dev .
docker run --rm -it --gpus all -w /kit -v /path/to/weights:/weights -e PROGEN2_WEIGHTS=/weights -v "$PWD/out":/kit/progen2/out progen2-kit:dev bash
# B — Apptainer (the whole Setup for B; `kit` below: a shell function = B's bash run.sh): build the .sif once where Docker runs (converts A's image); the cluster then needs only Apptainer
apptainer build progen2-kit.sif progen2/environment/apptainer.def
mkdir -p out; kit() { apptainer run --nv --bind /path/to/weights:/weights --bind "$PWD/out":/kit/progen2/out --env PROGEN2_WEIGHTS=/weights progen2-kit.sif "$@"; }   # B's ./run.sh · /weights/<progen2-size>/ = one checkpoint
kit install --weights /weights --model progen2-xlarge   # already-filled read-only /weights: type plain kit install
kit check --model progen2-xlarge                        # Run: kit <verb> … --input "$PWD/items.jsonl"; out/ as is
# C — instead of A or B, on your own host: any released CPython 3.9 (uv shown) — the whole route-C recipe; STOCK.md §Stack = these lines annotated (run one or the other)
command -v uv >/dev/null || { t=$(mktemp) && curl -LsSf -o "$t" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $t" | sha256sum -c - && sh "$t" && rm -f "$t" && . "$HOME/.local/bin/env"; } # uv itself, once per user (skipped when present)
uv venv --seed --managed-python --python 3.9 ~/venvs/progen2 && . ~/venvs/progen2/bin/activate
python -m pip install --no-deps -r progen2/environment/requirements.lock
```

Route B (Apptainer) is complete at this point; its Run lines are the same commands typed as `kit <command> …`.
Under **A** you are now in the container shell, which opens in `/kit`. Under **C** you are in your activated
environment: after the route-C lines above you are still in the directory holding `progen2/`, so type the whole block;
if you followed STOCK.md's Stack section instead, you are inside `progen2/` with `install` already done — continue at
`export`. Run:

```bash
[ -f run.sh ] || cd progen2                                  # no-op once inside · C via §Stack: skip install too
bash run.sh install --weights /weights --model progen2-xlarge   # checkpoints already in DIR (read-only ok): install alone
export PROGEN2_WEIGHTS=/weights                              # required · typed under C only (A: -e on the docker run line)
bash run.sh check --model progen2-xlarge                        # digests ("bytes": hashed, no SHA256SUMS — minutes); dry run
```

What the blocks assume:

- **Weights directory.** `/weights` is the mount point under A and B; under C it is any host directory. It must be
  writable when `install --weights DIR` is to fetch into it (it also writes `SHA256SUMS` there); read-only is fine once
  the checkpoints are in place — then run `bash run.sh install` alone (it skips pip where the kit is already installed, as
  in the A and B images).
- **`PROGEN2_WEIGHTS` under A.** The `docker run` line already sets it with `-e`; the `export` line is typed under C only.
- **Optional variables.** `PROGEN2_PYTHON` names the interpreter when it is not `python` on `PATH`; `PROGEN2_STOCK_DIR`
  names your own checkout of the pinned commit, if you use one instead of `stock/src/progen2`.
- **What the pin check refuses.** A stock file that is missing or modified, and an absent torch / transformers /
  tokenizers; versions off their pins are noted and accepted.
- **Nothing is compiled** at install or run time.
- **What `check` does.** A dry run; it also digest-checks the checkpoints — any checkpoint with no `SHA256SUMS` beside it
  is hashed, which takes minutes for the larger sizes.

## Run

```bash
X=1MGHGVSRPPVVTLRPAVLDDCPVLWRWGLDPDAVKIAMTRYIRYGCLLRLRGDAGVEQ2      # arbitrary example prompt sequence; "1" = start token alone
bash run.sh sample --mode off   --model progen2-xlarge --context 1 --max-length 512 --num-samples 64   # stock, for comparison: one clean sample.py process
bash run.sh sample --mode exact --model progen2-xlarge --context 1 --max-length 512 --num-samples 64   # the default when no mode is named
bash run.sh score  --mode exact --model progen2-xlarge --context $X                                    # stdout: ll_sum=<float> and ll_mean=<float>
printf '{"item_id": "a", "context": "1", "max_length": 512, "num_samples": 64}\n{"item_id": "b", "context": "1M", "rng_seed": 7}\n' > items.jsonl
bash run.sh sample --mode exact --model progen2-xlarge --input items.jsonl --out_dir out/sample         # a list of items through one loaded model
```

**Options.** `sample` passes `sample.py` its own flags and `score` passes `likelihood.py` its own (STOCK.md lists them).
The kit adds `--mode` and `--input FILE --out_dir DIR`.

**Outputs of a single call.** The stock block on stdout, and no files:

- `sample`: the context line, the numbered completions, `done.`;
- `score`: `ll_sum=` / `ll_mean=`.

Stock's loading and timing lines are not reproduced, so compare blocks, not whole stdouts.

**`--input` jobs** (both commands, both modes). `--input FILE` reads JSON lines keyed by the per-item flag names
(`CHANGES.md`, 'Switches' section) and writes `DIR/items/<item_id>/block.txt` per item under `--out_dir DIR`.

**What a run prints** (stderr):

- `check` ends with `[progen2-opt] check_pins rc=0 (…)` and a `[progen2-opt] DRY-RUN mode=exact … on=<levers> … card=…`
  line (`on=` lists the optimizations engaged; the kit calls them 'levers').
- `exact` prints `[progen2-opt] ACTIVE mode=exact variant=<size> route=<sample|score> kit=<dir> on=<levers>` and closes
  with `[progen2-opt] EXIT mode=exact route=… items=<n> kit_modules=…`.
- `off` prints `[progen2-opt] NOT ACTIVE: mode off: stock in a clean subprocess (mode=off variant=<size>)`, runs the
  stock script and exits with its code.
- If a mode cannot engage, the command exits 3 and names the reason on a `[progen2-opt] NOT ACTIVE: …` line; it never
  falls back to stock silently.

**Exit codes.**

| code | meaning |
|---|---|
| 0 | ok |
| 1 | a failed item or run |
| 2 | usage error |
| 3 | not active |

## Modes

- `off` — stock `sample.py` / `likelihood.py` as released, in a clean subprocess with nothing of the kit importable.
- `exact` (default) — in-process, with the model loaded once. For `sample`: one-read weights, resident rotary tables,
  static K/V slots, fused sampler bookkeeping. For `score`: rotary tables, fused elementwise CUDA kernels, one forward
  per direction, host pipelining. Outputs identical to `off`.

## Notes

- **Where the gain is.** The kit's gain is in scoring time per sequence with the model resident. A single `score` call
  also pays the ≈50 s checkpoint load in every mode, so one README example runs no faster end to end — the gain arrives
  with many sequences per process (`--input` lists, the API). `sample` loads in ≈8 s under `exact` (64 × 512 tokens
  peak ≈36.5 GiB).
- **Cards.** Runs alike on A100 (40 or 80 GB), H100, H200 and B200: the card is read at start-up and there is no
  `--config`. Any other CUDA card of compute capability 8.0 or newer runs with a note on the `stack` line; `--device cpu`
  is refused under `exact`.
- **Out of memory** under `exact` generation: the static K/V slots of a `--num-samples` value that cannot fit fail that
  item by name (`OUT OF MEMORY …`, exit 1, no fallback) → use fewer `--num-samples` per call.
- **Context window.** Inputs are bounded by the size's context window (`n_positions`, STOCK.md) in every mode; `score`
  fails a longer sequence by name.
- **Programmatic use**, one size per process: `from progen2_opt.api import load, sample, score, close` —
  `g = load("progen2-xlarge")`, `sample(g, context="1", max_length=512, num_samples=64).block`, `close(g)`;
  `h = load("progen2-base", route="score")`, `score(h, seq).block`.
