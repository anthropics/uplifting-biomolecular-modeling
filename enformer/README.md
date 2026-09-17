# Enformer — optimization kit

A drop-in mode that makes stock `enformer-pytorch` 0.8.12 (`enformer_pytorch.Enformer`, weights
`EleutherAI/enformer-official-rough`) faster. Your own script calls Enformer exactly as before; one environment variable,
`ENFORMER_OPT`, switches the kit on per process (or an `enable()` call in code):

- `off` — stock `enformer-pytorch`, exactly as released ('stock' below always means this unmodified upstream release).
  This is what runs with the variable unset.
- `exact` — identical outputs, faster.

What each optimization changes: `CHANGES.md`. Exact versions, the pinned software stack and all variables: `STOCK.md`.
How the three setup routes (A — Docker, B — Apptainer, C — Python venv) work in general: the top-level `README.md`.

**At a glance** (H100 80 GB vs stock): `exact` identical outputs, faster than stock.

## Setup

Pick ONE way to get the pinned stack — stock `enformer-pytorch` 0.8.12 and everything it needs: Debian 12, Python 3.11,
torch 2.13.0+cu130, CUDA 13.0 as pip wheels (the 'Stack' section of `STOCK.md` lists every pin): **A — Docker**,
**B — Apptainer**, or **C — a Python venv on your own host**.

Every route needs only:

- an NVIDIA driver 580 or newer on the host.

Route C additionally needs:

- `uv` (or any released CPython 3.11).

Type the first block from the directory that holds `enformer/` and `common/`:

```bash
# A — Docker (preferred): the whole pinned stack, stock and this kit in one image
docker build -f enformer/environment/Dockerfile -t enformer-kit:dev .
docker run --rm -it --gpus all -v /data/hf_home:/weights -e HF_HOME=/weights -v $PWD:/work -w /kit enformer-kit:dev bash   # a shell in /kit; your scripts under /work
# B — Apptainer / Singularity (the whole Setup for B): converts A's image — no Docker on the cluster? build the .sif once on any machine with Docker and copy it over
apptainer build enformer-kit.sif enformer/environment/apptainer.def
kit() { apptainer run --nv --bind /data/hf_home:/weights --env HF_HOME=/weights --bind "$PWD":/work enformer-kit.sif "$@"; }   # kit = B's ./run.sh (its verbs only) · your scripts: kitx
kitx() { apptainer exec --nv --bind /data/hf_home:/weights --env HF_HOME=/weights --bind "$PWD":/work enformer-kit.sif "$@"; }   # kitx = same binds via apptainer exec: your own commands
kit check                                                                 # the one verb of use under B (install is baked in)
ENFORMER_OPT=exact kitx python /work/your_script.py                       # this host dir is /work inside; without the variable: stock
#     B writes nothing of its own; `from_pretrained` fills /data/hf_home at first use when the snapshot is not there yet (that bind is written once), afterwards it is only read
# C — instead of A or B, on your own host: fresh venv (released CPython 3.11.12 via uv) + the pinned lock — the whole route-C recipe; STOCK.md §Stack = these lines annotated (run one or the other); an existing environment at the pin also works
command -v uv >/dev/null || { f=$(mktemp) && curl -LsSf -o "$f" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $f" | sha256sum -c - && sh "$f" && rm -f "$f" && . "$HOME/.local/bin/env"; }   # uv 0.12.15 itself, once per user; installer digest-checked
uv venv --seed --managed-python --python 3.11.12 ~/venv-enformer && . ~/venv-enformer/bin/activate
pip install --no-deps -r enformer/environment/requirements.lock           # several GB of wheels (torch, NVIDIA CUDA 13)
```

Route B (Apptainer) is complete at this point: `kit check` is its one `run.sh` command, and your own scripts run through
the `kitx` line shown. Under **A** you are now in the container shell, which opens in `/kit`; the install is already in
the image, so after the `cd` go straight to `check`. Under **C** you are in your activated environment: after the
route-C lines above you are still in the directory holding `enformer/`, so type the whole block; if you followed
STOCK.md's Stack section instead, you are inside `enformer/` with `install` already done — only `check` remains. Run:

```bash
[ -f run.sh ] || cd enformer     # no-op once inside · C via §Stack: skip install too
bash run.sh install                 # stock wheel + kit (editable), then the pin check
bash run.sh check                   # dry run: device, pins, kit files; loads no weights/kernels
```

What the blocks assume:

- **Weights.** No variable is required. Weights come through upstream's own `from_pretrained` into `HF_HOME`, a Hugging
  Face cache root (the blocks mount the host directory `/data/hf_home` at `/weights` and point `HF_HOME` there). The first
  use downloads them; with an existing copy, `export HF_HOME=<dir holding hub/>` first.
- **What `check` prints.** `[enformer-opt] DRY mode=exact gpu=<name>(smNN) kit=v0.2 levers=poscache,fused,graph,xattn build=<build>`,
  exit 0, when `exact` can engage on this machine (for a refusal see Run below).
- **What `install` checks.** It names where `enformer-pytorch` 0.8.12 came from, refuses another version or an editable
  checkout (exit 3), and lists any distribution that differs from the pin without refusing (exit 4; STOCK.md, Stack section).
- **Nothing compiles.** The kit's kernels ship prebuilt under `opt/forward/`; the kit compiles nothing at install or run
  time (torch and the CUDA driver still keep their usual caches, e.g. `/tmp/torchinductor_*`, `~/.nv/ComputeCache`).

## Run

The kit adds no run command: `your_script.py` is your own code.

```bash
export HF_HUB_OFFLINE=1                     # optional, after the first download: stay on that snapshot
python your_script.py                       # off: your own code, stock; under A: /work/your_script.py
ENFORMER_OPT=exact python your_script.py    # exact: same code (B: Setup's kitx line, typed there)
```

```python
import torch                                           # in code instead: import enformer_opt; enformer_opt.enable()
from enformer_pytorch import from_pretrained, str_to_one_hot
model = from_pretrained("EleutherAI/enformer-official-rough").cuda().eval()
seqs = ["ACGT" * 49_152, "TTAC" * 49_152]                # two 196,608-bp windows
with torch.no_grad(): out = model(str_to_one_hot(seqs).cuda())   # {'human': (2, 896, 5313), 'mouse': (2, 896, 1643)}
print(out["human"].shape)                              # torch.Size([2, 896, 5313]) once it ran
```

**Inputs and outputs.** Inputs go in as in stock (one string, a one-hot or an index tensor) and outputs come back from
`model(x)` exactly as stock returns them; no files are written.

**What a run prints** (stderr):

- `off` prints no kit line.
- `exact` prints `[enformer-opt] ACTIVE mode=exact gpu=<name>(smNN) kit=v0.2 levers=poscache,fused,graph,xattn build=<build> applied=deferred`
  once engaged (`levers=` lists the optimizations; the kit calls them 'levers'), then
  `[enformer-opt] APPLIED model#<k> levers=… batch=<B> cost=…s` when a model is fitted at its first forward.
- If `exact` cannot engage (no CUDA device, `enformer-pytorch` not 0.8.12, a kit file missing, no kernel build for the
  device), the `NOT ACTIVE` line names the reason and the process exits 3 under `ENFORMER_OPT=exact`. The kit never
  switches to stock silently.

**From code instead of the variable.**

- `import enformer_opt; enformer_opt.enable()` engages the kit; if `exact` cannot engage, plain `enable()` prints the
  `NOT ACTIVE` line and leaves stock in place, while `enable(strict=True)` raises `enformer_opt.ActivationError`.
- `enformer_opt.apply(model, batch_sizes=(2,))` fits a model up front.
- `enformer_opt.status()` reports the state.
- `enformer_opt.disable()` restores the stock modules.

**Exit codes** (`run.sh`).

| code | meaning |
|---|---|
| 0 | ok |
| 2 | usage error |
| 3 | not active, or pin refused |
| 4 | the stack differs from the pin (informational) |

## Modes

- `off` — `ENFORMER_OPT` unset or `off` and no `enable()`: stock `Enformer.forward`, nothing of the kit hooked or imported.
- `exact` — positional basis cached per layer (`poscache`), fused fp32 elementwise trunk kernels (`fused`), one exact
  kernel chain for the relative-position attention (`xattn`), CUDA-graph trunk replay per batch size (`graph`) — always
  all four. Outputs identical to `off`.

## Notes

- **Where the gain is.** The kit's gain is in forward time per call on a fitted model. Loading the checkpoint (≈4.5 s in
  either mode) and the one graph capture at each new batch size (≈0.7–2.8 s) dominate a script that makes a single call,
  which ends about as fast as stock; the gain accrues over repeated calls.
- **First forward per model.** It loads the extensions, patches the modules and captures one graph for that batch size
  (`APPLIED`). Each further batch size captures once at its first call (`CAPTURED`); a size whose capture does not fit
  free memory runs unreplayed (`UNGRAPHED`).
- **Scope.** Eval-mode models on a CUDA device at the 196,608-bp window. Other lengths and keyword call forms run the
  patched modules without replay (`EAGER`); a training-mode model stays stock (`STOCK`); grad inputs, `model.train()` and
  moving an applied model are refused by name.
- **Cards.** The kit reads the device's compute capability and every optimization engages on each supported build:
  - 9.0 (H100/H200) loads the objects built for 9.0 (`build=sm_90`);
  - 8.x loads the 8.0 build under `opt/forward/classes/a100_80gb/` (`build=class:a100_80gb`);
  - above 9.0 loads the 9.0 PTX (`build=sm_90-ptx`);
  - below 8.0: `NOT ACTIVE`.
- **Other torch builds or capabilities.** A torch build other than 2.13.0+cu130, or a capability other than 9.0 / 8.0, is
  named in a `notes=` clause on the `ACTIVE` line and served.
- **Autocast.** Under `torch.autocast` the kit computes in float32 and says so (`AUTOCAST`); call `enformer_opt.disable()`
  first for mixed-precision or gradient work.
