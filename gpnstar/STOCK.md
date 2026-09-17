# GPN-Star — stock, as pinned

## Pin

Upstream: `gpn` 0.9.0, github.com/songlab-cal/gpn at commit `6f28c81bcbfe7d65cb6d8ece9ce88f87ca583791` (MIT), installed unmodified from the
byte-frozen `git archive` of that commit shipped under `stock/` (`gpn-6f28c81b.tar.gz`, member prefix `gpn-6f28c81b/`, sha256 in
`stock/PINS.json`: the same files, digest for digest, as the archive GitHub serves for the commit), upstream's `LICENSE` as found at its root;
`gpn[inference] @ git+https://github.com/songlab-cal/gpn@6f28c81bcbfe7d65cb6d8ece9ce88f87ca583791` is the byte-equivalent alternative install.
Its `transformers==5.15.0` requirement is upstream's own hard pin. Every pin value, with digests, is in `stock/PINS.json`; `stock/check_pins.py`
reads it. `stock/` is never edited, and the kit patches nothing on disk: the levers attach to the loaded model object at run time.

Weights: the released GPN-Star checkpoints on Hugging Face at immutable revisions. `bash run.sh install --weights DIR [--model KEY|REPO]` stages
one whole snapshot (default: the primary) through `huggingface_hub.snapshot_download(repo, revision=<pinned>)` — the client upstream's loader
uses — into upstream's own cache layout `DIR/hub/models--songlab--<name>/snapshots/<commit>/`, writes the cache's `refs/main` naming that
commit (a download by commit leaves no ref), checks every file PINS.json lists for it
(`weights.<repo>.files`: sha256 and size) and prints `[gpnstar-opt install] OK model=<key> files=<n> snapshot=<dir> …`; `DIR` is then
`HF_HOME`. A file already present is kept and only digest-checked; a file off its digest is named (`FAIL …`) and the step exits 1 with the
file left in place, never deleted. The loader reads `config.json`, `model.safetensors` and, through `config.json` `phylo_dist_path`,
`phylo_dist/pairwise.npy` and `phylo_dist/in_clade.npy`; the snapshot's `calibration_table/*.parquet` is not read by `gpn star vep | logits |
embedding`. `--model-path` takes the snapshot directory (`$HF_HOME/hub/models--songlab--gpn-star-hg38-v100-200m/snapshots/<commit>`): given the repo id instead, upstream asks the
Hub for the phylogenetic-distance files, which fails under `HF_HUB_OFFLINE=1`; `run.sh` injects nothing, and the kit's own loads pass the
pinned revision. All five model cards state `license: mit`. The container
images carry no weights.

| key (`--model`) | repo @ revision | species | native window | snapshot |
|---|---|---|---|---|
| `v100-200m` (primary; the default) | `songlab/gpn-star-hg38-v100-200m` @ `0c949f132d35619a3eb188b402848c998a3313ae` | 100 | 128 bp | 0.8 GB |
| `ce11-25m` | `songlab/gpn-star-ce11-n135-25m` @ `078ad60c23bbaf40fbeae89fb509a21320a71bbb` | 135 | 128 bp | 0.1 GB |
| `m447-200m` | `songlab/gpn-star-hg38-m447-200m` @ `0d8f5164fba8e8108200c160370b86652bfb6244` | 447 | 256 bp | 0.8 GB |
| `p243-200m` | `songlab/gpn-star-hg38-p243-200m` @ `16773016130e826cd3d72e91ed312a3cb27ba2b5` | 243 | 256 bp | 0.8 GB |
| `tair10-25m` | `songlab/gpn-star-tair10-b18-25m` @ `317841ed55164ff4cd3d14e636bcc2d5de23cf49` | 18 | 128 bp | 0.1 GB |

`mm39-85m` (`songlab/gpn-star-mm39-v35-85m`) is known to the package but carries no digests in PINS.json: `--model mm39-85m` stages it,
prints its digests (`DIGEST …` lines) and exits 1, unchecked. `python -I stock/check_pins.py --weights "$HF_HOME" --variant <key>|all` checks
staged files against the pins at any time. Alignments: `gpn star …` reads a local GPN-Star MSA store (`--msa-path`: a directory named by its
species count, target included, holding `all.zarr` — one array per chromosome — or a parent of such directories); it is an input like the
variant table, never fetched by the kit. Upstream publishes the stores as Hugging Face datasets: for the hg38 100-way model
`songlab/multiz100way` (`99.zarr.zip`, a 42 GB download, unpacked as `<root>/100/all.zarr`; dataset card licence MIT), and its model
documentation names the stores for the other checkpoints (the ce11 and tair10 alignment cards carry no licence field). The kit's `examples/` directory holds eight hg38 variants (`variants.parquet`: control rows of upstream's
`songlab/omim_traitgym` table, coordinates and alleles only; dataset card licence MIT) and a sparse cut of that
100-way store covering only their 128-bp windows (`examples/msa/100/all.zarr`, a few megabytes), taken from `songlab/multiz100way` so that a first
run needs no download; `examples/README.md` records how it was cut and `examples/SOURCES.md` records where both come from and under which terms.

## Stack

Debian 12 (`python:3.13.15-slim-bookworm`), Python 3.13.15 (upstream requires ≥ 3.13, < 3.14), the CUDA 13.0 runtime as wheels (cuda-toolkit
13.0.3.0: nvidia-cuda-runtime 13.0.96, nvidia-cublas 13.1.1.3, nvidia-cudnn-cu13 9.20.0.48, nvidia-nccl-cu13 2.29.7; the driver is the host's,
580.65.06 or newer), torch 2.13.0 (PyPI's CUDA 13.0 build), triton 3.7.1, transformers 5.15.0, tokenizers 0.22.2, huggingface_hub 1.31.0
(hf-xet 1.6.0), safetensors 0.8.0, accelerate 1.15.0, numpy 2.5.3, zarr 3.3.0 (numcodecs 0.16.5), cyclopts 4.25.2, jaxtyping 0.3.11, networkx
3.6.1, and upstream's `inference` extra — datasets 5.0.1, pandas 3.0.5, pyarrow 25.0.1, biopython 1.88 — the stack every statement of this kit
holds on (upstream's own `uv.lock` resolves numpy, huggingface_hub and accelerate slightly lower; PINS.json records both). The full list, 86
distributions (`pip freeze --all` of the image's interpreter, pip and setuptools included), is `environment/requirements.lock`;
`environment/Dockerfile` builds it into the image `gpnstar-kit:dev` with stock, the kit and the shared core under `/kit`, and
`environment/apptainer.def` converts that image (`From: gpnstar-kit:dev`; its runscript is `run.sh` from `/kit/gpnstar`). One build serves
every configured card: the kit compiles nothing at build time; at run time the `colattn` lever JIT-compiles two small Triton kernels (Triton ships with
torch; its launcher stub needs the C compiler the image carries) and the `fusedattn` lever compiles two small CUDA kernels once per process through
torch's NVRTC bindings (the `nvidia-cuda-nvrtc` wheel of the pinned stack; no CUDA toolkit); upstream's optional `--torch-compile` compiles with gcc too. Outside the images (README route C), from the directory holding `gpnstar/` and `common/`, on a host with an
NVIDIA driver at 580.65.06 or newer (no CUDA toolkit: the CUDA libraries come as wheels, ≈5 GB installed):

    sudo apt-get update && sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y gcc libc6-dev curl ca-certificates                # bare Ubuntu host (root: no sudo); skip what you have — gcc + libc6-dev are the Dockerfile's one apt line (used only by upstream's --torch-compile); curl + ca-certificates fetch uv below
    command -v uv >/dev/null || { f=$(mktemp) && curl -LsSf -o "$f" https://astral.sh/uv/0.12.15/install.sh && echo "716a1d6844740756c68770fcec2f79c2013fb9b03869a113f61e15f6f482a6a1  $f" | sha256sum -c - && sh "$f" && rm -f "$f" && . "$HOME/.local/bin/env"; } # uv 0.12.15 itself, once per user; installer digest-checked
    uv venv --seed --managed-python --python 3.13 ~/gpnstar-venv && . ~/gpnstar-venv/bin/activate  # uv's own released CPython 3.13, never a system python; --seed puts pip in the venv
    python -m pip install --no-deps -r gpnstar/environment/requirements.lock  # every pin in one pass: torch 2.13.0 (cu130 wheels), transformers 5.15.0, stock gpn from the archive in gpnstar/stock
    cd gpnstar && bash run.sh install --weights /weights/gpnstar/hf_home          # = README step 2's install line — type it once; step 2 continues at its export line (installs the shared core + the kit package editable, runs stock/check_pins.py, stages the checkpoint)

pip builds stock `gpn` from the archive (or the git URL) with hatchling in an isolated build environment fetched from the package index at install time
(network needed; the Dockerfile pins hatchling 1.32.0 for that build); the lock pins runtime packages only. `python -I stock/check_pins.py [--weights DIR [--variant KEY|all]]
[--gpu] [--quiet]` (stdlib only; needs the archive beside it; run by `bash run.sh install`) is the full check of an environment: (1) `gpn` at
the pinned version with installed modules byte-identical to the archive — installed from that archive or from the git URL at the pinned commit,
either passes, and the ACTIVE line reads `gpn=6f28c81b` either way — and `transformers` at upstream's hard pin; anything else is another stock,
refused by name (exit 3); (2) torch, triton, huggingface_hub,
safetensors, accelerate, numpy, zarr and cuDNN against the pins — a package absent or off its pin is refused by name (exit 3); (3) with
`--weights`, every pinned checkpoint file by sha256 and size — a file the loader reads missing or off its digest is refused (exit 3), the
snapshot's other files are named when absent; (4) with `--gpu`, the visible device named as a listed class (`listed (h100|a100|a100_40gb)`)
or unlisted. Exit 0 all pinned · 2 usage · 3 refused. `--skip-package` runs (3)/(4) alone, before the stack is installed. At run time the
kit's own gates refuse fewer things (`gpn` off its commit, `transformers` off its pin, no GPU: NOT ACTIVE, exit 3);
torch, numpy, huggingface_hub, safetensors or accelerate off pin are named on the ACTIVE line (`stack=drift(…)`) and the kit engages. Cards:
PINS.json `gpus` lists the H100 80 GB class and the A100 class (80 GB SXM4 or PCIe, and 40 GB); the ACTIVE line says `card=h100|a100 mib=<MiB>`;
no lever differs by card. Any other card is `card=untested`, engaged, never refused.

## How stock is run

Stock is upstream's own command, run as released with its documented speed flag: `gpn star vep --input-path I --msa-path MSA --window-size W
--model-path M --output-path O --tf32` (or `logits` / `embedding`; upstream's other options as given: `--model-revision`, `--split`, and
transformers' prediction options — `--per-device-eval-batch-size`, default 8; `--dataloader-num-workers`, default 0; `--torch-compile`). The kit is
the same command with `GPNSTAR_OPT=exact` in front; with the variable unset nothing of the kit is imported (the package's `.pth` reads the variable
and stops there). Upstream itself reads only `HF_HOME` / `HF_HUB_CACHE`, `HF_HUB_OFFLINE`, `CUDA_VISIBLE_DEVICES` and `TOKENIZERS_PARALLELISM`.
Stock builds `MLMforVEPModel` around `AutoModelForMaskedLM.from_pretrained(M, revision=R)` (no dtype, attention or device argument) and runs
`Trainer.predict`: fp32 weights and activations, eval mode, `torch.no_grad`, no autocast, `scaled_dot_product_attention` restricted to the math
backend by the model code. Numerics flags are transformers': with the bare flag `--tf32` (`TrainingArguments(tf32=True)`, built before the model)
it turns TF32 tensor-core matmuls on for the process through torch's precision interface (`torch.backends.fp32_precision = "tf32"`; on this
stack the legacy `torch.backends.cuda.matmul.allow_tf32` then no longer answers in that process) — fp32 storage and accumulation, TF32 multiplies
in every cuBLAS GEMM; softmax, LayerNorm, GELU and all reductions stay fp32. Without the flag torch's defaults hold (IEEE fp32 matmuls; the `gpn`
package sets no precision flag). `--torch-compile` compiles the wrapper's forward with TorchInductor, which needs a C compiler at run time (the
image carries gcc for it) and takes tens of seconds once per process, and transformers turns TF32 on with it too. The kit sets none of these and
follows all of them: its statement is bitwise identity with stock, certified for the documented invocation (`--tf32`, as in the README's Run
block); other numeric settings run the same levers without that certification. Upstream's `--bf16-full-eval` does not run on this model at the pinned stack, with or without `--torch-compile`: it raises
`RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16` in `torch.nn.functional.linear`. The input table
gives `chrom, pos, ref, alt` (`pos` one-based); there is no genome FASTA: the reference base is the alignment's target column and a `ref` that
disagrees with it raises. Per variant the window is read from the store on both strands with the variant position masked (index `W/2` forward,
`W/2 − 1` reverse); the score is the mean over strands of logit(alt) − logit(ref) at that position, written to `O` as parquet. On this stack
stock's forward is run-to-run deterministic for a given batch under either numerics setting. When `--output-dir` is not given upstream's runner
hands the trainer a temporary one, so nothing is created in the current directory.

## Stock exceptions

None. The kit runs `gpn` 0.9.0 exactly as released; no upstream issue is patched.

## Variables

| variable | required | default | effect |
|---|---|---|---|
| `HF_HOME` | yes, unless the weights sit in the default cache | `~/.cache/huggingface` | the Hugging Face cache holding the checkpoint snapshot (layout above; `HF_HUB_CACHE` is honoured too); `install --weights DIR` makes DIR one |
| `GPNSTAR_OPT` | no | unset = stock | the switch: `exact` engages the kit at the first `gpn.star.inference` import of the process (upstream's `gpn star …` command, or your program); unset or `off` loads nothing of the kit; any other value is refused there (`NOT ACTIVE: unknown GPNSTAR_OPT=…`, exit 3) |
| `HF_HUB_OFFLINE` | no | unset | upstream's (huggingface_hub's) own: `1` = no Hub access at run time, every load resolves to the staged snapshot; recommended once `install --weights` has run |
| `CUDA_VISIBLE_DEVICES` | no | unset | torch's own: which card the process sees; the kit engages on the device the trainer places the model on |

The kit reads `GPNSTAR_OPT` and `HF_HOME` (the latter only to find the staged snapshot for its digest check) and nothing else; it sets no
variable and no torch flag. Its autoload hook (`opt/gpnstar_opt_autoload.pth`, installed with the package) acts at `import gpn.star.inference`
when `GPNSTAR_OPT=exact` — that is what makes the switch engage from the unchanged `gpn star …` command. JIT caches: the `colattn` Triton kernels honour `TRITON_CACHE_DIR` (Triton's own variable; default `~/.triton`); the `fusedattn` NVRTC kernels are compiled in memory once per process (about a second, nothing written); the kit defines no cache variable of its own.
