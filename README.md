# Inference optimization kits

36 drop-in optimization kits for the inference paths of open protein- and genomics-ML tools: structure prediction and
cofolding, binder and sequence design, protein and genomic language models. One kit per upstream tool.

- Each kit lives in its own directory, next to a copy of the upstream release at a pinned version ('stock'), carried as
  released; the few declared exceptions are listed in `NOTICE` and the kit's `STOCK.md`.
- A kit engages its optimizations under a named **mode** when the tool starts. How you call the upstream tool does not change.
- In 34 kits this happens inside upstream's own Python environment. The two foundry kits (`rfdiffusion3`, `rosettafold3`)
  instead install a small file overlay into a second interpreter and keep a pristine interpreter for `off`.
- This page describes the common case. Where a kit differs — a declared bug-fix patch applied in every mode, no container
  route, no shared core, an extra `run.sh` command — its own `README.md` and `STOCK.md` say so.
- Run a kit from its own `README.md`. Every kit README has the same shape: title block with the At-a-glance line, then
  Setup, Run, Modes, Notes.

Licence: `LICENSE` (Apache License 2.0, for the original code) and `NOTICE` at the top of this tree; each kit's own
`LICENSE` sets `stock/` (the upstream project, under its own licence) apart.

**Status: not maintained and not accepting contributions.** This is a reference release that accompanies the
"How Claude is uplifting biomolecular modeling" blog post. The code is provided as-is at the upstream versions pinned
in each kit's `STOCK.md`; Anthropic does not plan further updates and does not accept pull requests. Issues may be
filed for the record but might not receive a response. Forks are welcome under the terms of `LICENSE`.

## Modes

One vocabulary across all kits. Each kit ships the subset listed in the table below and in its README; any other mode
name is an error.

- `off` — stock: the pinned upstream release exactly as published, with nothing of the kit engaged. In the `run.sh` kits
  this is a clean subprocess; in the library-style kits it is simply your own process with the kit's variable unset.
- `exact` — outputs identical to `off`, faster.
- `fast` — small, documented numeric differences (within the tool's own seed-to-seed variation), faster still. Usually the
  default where a kit ships it.
- `big` — lowest peak GPU memory, for inputs the other modes cannot hold. Where a kit ships it, `--n_gpu P` splits one
  `big` prediction across P GPUs of one host.

Default mode (what runs when no `--mode` is given):

- usually `fast`;
- `exact` in the kits that ship no `fast`, and in `evo2`;
- `off` in `atlasfold`;
- none in `proteinmpnn` — every call names its mode;
- none in the library-style kits (`enformer`, `enformer_deepmind`, `esmc`, `gpnstar`; `flashzoi` too when driven from your
  own script): `<KIT>_OPT=exact`, or an `enable()` call in your script, engages the kit; with the variable unset the process runs stock.

What every kit prints (the `ACTIVE` line):

- A run in a kit mode prints one `[<kit>-opt] ACTIVE mode=…` line (on stderr in most kits) naming what is engaged.
- A mode that cannot engage on this machine prints `[<kit>-opt] NOT ACTIVE: <reason>` and exits 3. A kit never falls back
  to stock silently.

## Layout

Each kit directory has the same layout:

| path | what it is |
|---|---|
| `stock/` | the upstream release at its pinned version (wheel, source archive or commit; weights pinned by digest); carried as released, never edited in place (declared exceptions: `NOTICE`, `STOCK.md`) |
| `opt/` | the kit's own Python package |
| `environment/` | the pinned software stack: `Dockerfile`, `apptainer.def`, `requirements.lock` (most kits) |
| `configs/<card>.env` | per-GPU-card settings (most kits) |
| `run.sh` | the entry point: `install`, `check` and the kit's run commands |
| `README.md` | how to run the kit |
| `STOCK.md` | exact upstream version and pins, the software stack, environment variables |
| `CHANGES.md` | what each optimization changes, by mode |
| `upstream_issues/` | opt-in fixes for upstream defects, off by default (only where a kit has any) |

`common/opt_core/` is the shared runtime nearly every kit imports: mode resolution and the `ACTIVE` line, determinism
helpers, memory and multi-GPU machinery, and the shared GPU kernels (FlashPairformer and the triangle-attention /
triangle-multiplication kernels the structure kits use). `common/opt_core/README.md` describes them.

## Install and run — the pattern

1. Get the kit's pinned stack in ONE of up to three ways (each kit's Setup shows which it offers):
   - **A — Docker**: the kit's Docker image;
   - **B — Apptainer**: an Apptainer image (`.sif`) converted from it;
   - **C — Python venv**: your own environment at the pinned versions, following the 'Stack' section of the kit's `STOCK.md`.
2. From the kit directory: `bash run.sh install` (`--weights /weights/<kit>` where the kit has weights to fetch). This installs
   the kit and the shared core (editable), checks the pins, and fetches the weights through upstream's own downloader,
   digest-checked. Then export the weights variable as the kit's Setup shows.
3. `bash run.sh check` — a dry run: it resolves the mode on this machine and prints what would engage.
4. `bash run.sh <command> --config <card> --mode <mode> …` in most kits; the upstream tool's own arguments pass through
   verbatim. In the library-style kits your own script runs unchanged: `<KIT>_OPT=exact python your_script.py` (or the
   kit's `enable()` call).

### Prebuilt images (planned)

Prebuilt Docker and Apptainer images, with each kit and its dependencies already installed, are planned within a week
of this initial release. Until then, build from each kit's `environment/Dockerfile` or `environment/apptainer.def`, as
described in its README. Model weights are downloaded separately either way.

### How routes A, B and C differ

Under A and B, steps 2–4 run against the image; under C they run in your own environment.

#### A — Docker

- `docker build` makes the image; `docker run --rm -it --gpus all -v … <kit>-kit:dev bash` opens a shell inside it.
- Type steps 2–4 in that shell.

#### B — Apptainer

- Build the `.sif` file once, on any machine that has Docker (the kit's `apptainer.def` converts the image from route A),
  and copy that one file to the cluster. The cluster needs only Apptainer.
- There is no shell step. Each kit README defines a small shell function, `kit()`, wrapping `apptainer run --nv … <kit>-kit.sif`.
  You type `kit install …` (or the kit's weights line) and `kit check …` on the host, and every Run line becomes
  `kit <command> …`. Those `kit` lines are the whole Setup for B; the second Setup block is for A and C only.
- The library-style kits (`esmc`, `enformer`, `enformer_deepmind`) use an `apptainer exec … python your_script.py` line instead.
- Variables you `export` on the host are visible inside the container.
- **Outputs.** The image is read-only and the command starts inside it at `/kit/<kit>`, so nothing is written into the
  image: results go to the host directory you bind over the kit's output path (or to absolute host paths).
- **Inputs.** Give your own input files as absolute host paths.
- **Compile cache.** It goes to `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>` per user and node, unless `MODEL_OPT_JIT_ROOT` names a
  lasting directory. Most B blocks set it to `./jit`; the others say where theirs lives.
- **Weights.** Only `install --weights` writes the weights directory. In most kits both container routes mount one host
  directory at `/weights` (the whole weights root or the kit's sub-directory; a copy kept elsewhere is mounted there too,
  e.g. `--bind /site/af2:/weights/af2_params`), and the exports and commands address files by that container path.
- **Other host directories.** Apptainer binds `$HOME` and the current directory by default; anything else needs
  `--bind <dir>` or `APPTAINER_BIND`.

#### C — Python venv (your own environment)

- Where a kit's Setup inlines the route-C lines, they are the whole recipe. The 'Stack' section of that kit's `STOCK.md`
  is the same commands, annotated — run one or the other, once.
- Then do step 2 from inside the kit directory, skipping `cd` and `install` where the Stack section already ran them
  (continue at the `export` line).

### Compile caches

- A mode's kernels compile on its first run and stay under a cache root: seconds to minutes, once per root.
- The JAX kits that compile per input length or shape (`af2ig`, `colabdesign`, `mosaic`) take minutes for each new
  length or shape; their READMEs say how long.
- An image built with a kit's optional compile-cache tar (its `STOCK.md` says how) carries that cache pre-filled:
  `run.sh` uses it in place or seeds your cache from it on first use and says so (a `jit cache: … (<how>)` line, e.g.
  `seeded from image`). An image built without one starts with an empty cache, and each mode compiles on its first run
  as under route C.
- A pre-filled cache serves any GPU of the same compute capability in the Torch/Triton kits, but only the same GPU model
  in the JAX kits (`af2ig`, `af3_jax`, `colabdesign`, `colabfold`, `mosaic`); there another card — or a GPU configuration
  that differs even under one product name — compiles once, then loads.
- To keep the cache, set `MODEL_OPT_JIT_ROOT` to a writable host directory. Exported on the host it reaches an Apptainer
  container unchanged (bind the directory too if it is outside `$HOME` and the current directory); under Docker pass
  `-e MODEL_OPT_JIT_ROOT=/jit -v <host dir>:/jit`.

### Host prerequisites for route C

The images already carry everything below. On your own machine (Linux x86-64 with an NVIDIA driver):

Files fetched over HTTP (for example a Hugging Face download) lose their executable bits; every command in these READMEs invokes
scripts through `bash`, so nothing depends on them.

- **System packages.** Each kit's `STOCK.md` Stack section opens with the one `apt-get` line a bare Ubuntu host needs,
  where it needs any — the packages the kit's Dockerfile installs (compiler, `git`, the headless X libraries for the
  RDKit / Open Babel kits, …) — and names any CUDA toolkit a source build wants.
- **Python.** The examples use `uv venv --seed --managed-python --python X.Y`, which fetches that CPython with its
  headers. Get uv with the installer line each kit's `STOCK.md` Stack section shows — it downloads uv's installer from
  a version-pinned URL, checks it with `sha256sum -c` and only then runs it (needs `curl` or `wget` and
  `ca-certificates`, absent from bare CUDA images) — or with `pipx install uv && pipx ensurepath` and a new shell. A
  distribution CPython of the kit's minor version works the same way through `python -m venv`, with its `-venv` and
  `-dev` packages (e.g. `python3.11-venv python3.11-dev`). `openfold3` alone wants that standard-library
  `python -m venv` for its pin check (its `STOCK.md` says).
- **libstdc++.** The `exact` modes of `esmc` and `opendde` load a prebuilt kernel linked against a newer libstdc++ than
  Ubuntu 22.04's GCC 12 provides (GCC 14's and GCC 13's); their `STOCK.md` Stack sections give the one-line remedy.
  `common/opt_core/README.md` lists each shared library's glibc and libstdc++ floor.

## Security considerations

These kits are performance tooling for research workloads. They run inside, and with the privileges of, the upstream
model packages they accelerate, and assume the same trust model: trusted inputs, trusted weights, a single-user
machine or container. Before deploying them anywhere else, note:

- **The kits execute upstream code.** Each kit installs and imports an unmodified upstream release (`stock/`, pinned
  by version, commit and digest in `STOCK.md`); track those projects' own advisories.
- **Installing a kit adds an interpreter start-up hook.** `pip install -e opt` places `<kit>_opt_autoload.pth` in that
  environment's `site-packages`, so every Python process started there imports the kit's small `_autoload` module
  before anything else: with no variable under the kit's `<KIT>_OPT` prefix set it returns at once and stock is
  untouched; with a mode named it checks the pinned shared core and installs an import hook that patches the upstream
  package in memory when it is first imported, or ends the process with the kit's `NOT ACTIVE` line. Uninstalling the
  kit removes the file; `af2ig`, `ef2inv`, `esm_if1`, `progen2` and `proteinmpnn` install no such hook.
- **Some kit lines start the model process through `sitecustomize` hooks.** `boltzgen`, `opendde`, `openfold3`,
  `openfold3_ob0` and `protenix_v2` put a hook directory of the kit first on that process's `PYTHONPATH`, and
  `chrombpnet`'s deterministic runs do the same for their helper processes, so the interpreter executes the kit's
  `sitecustomize.py` at start-up; `openfold3`'s and `openfold3_ob0`'s hooks then execute the next hook file in the
  directory an `OPENFOLD3_OPT_*_CHAIN` variable names (the kit points those at its own add-on directories) and
  `chrombpnet`'s runs the next `sitecustomize.py` found on `sys.path`. Whoever controls those variables or
  `PYTHONPATH` decides what these processes execute, as for any Python program.
- **Some levers re-execute upstream source they edit in memory.** In `complexa`, `esmfold2`, `proteinmpnn` and
  `rosettafold3`, and in a few add-on and kernel modules elsewhere in the tree, a lever reads an upstream function's
  source from the installed package with `inspect.getsource`, substitutes anchored statements — refusing when the text
  is not the one it was written against — and executes the result in that module's namespace; what runs is the
  installed upstream file plus the kit's own replacement text, nothing read from anywhere else.
- **`rfdiffusion3` and `rosettafold3` install by overwriting upstream files.** Their install step copies the kit's
  patched versions of a fixed list of `rfd3` / `rf3` modules over the installed ones in the kit interpreter's
  `site-packages` — the originals are backed up beside them, every installed file is compared with the kit's copy by
  SHA-256, and a file found in an unknown state stops the step — while a second, untouched interpreter runs stock; any
  other program using the kit interpreter imports the patched modules too.
- **Model files are deserialized with `torch.load` / pickle.** Loading a checkpoint executes whatever the pickle
  contains; only use weights obtained from the documented upstream locations whose SHA-256 digests match the kit's
  pins.
- **`chrombpnet`'s models are Keras HDF5 files.** Stock ChromBPNet opens them with Keras `load_model`, which rebuilds
  the network from the architecture stored in the file, and a `Lambda` layer in such a file carries Python bytecode
  that runs on load — treat a model file like a pickle and use the pinned files, which the kit digest-checks; the
  kit's PyTorch route reads only the weight arrays from them.
- **Files the kits write and read back are digest-checked.** Compiled-program stores, memo files and any other cache a
  kit writes carry a SHA-256 recorded when the file is written and verified before it is read or unpickled; a file
  without a valid digest is ignored and rebuilt, and a cache directory that other users can write to, or that another
  user owns, is refused by name.
- **Generated kernel source is regenerated, not trusted.** Kernel source a kit writes to disk is compared byte for
  byte with what its generator produces and is never imported as found.
- **Kernels compiled at run time are cached per user.** The directories the kits choose for kernels compiled or
  autotuned on first use — Triton, Inductor, XLA, torch extension builds — lie under the compile-cache root below or
  under a per-user default the kit's `STOCK.md` names (a directory in the user's home, a `-uid<uid>` directory under
  `${TMPDIR:-/tmp}` created with mode 0700, or a freshly made private temporary directory), never a fixed name shared
  between accounts; such a default that another account owns, that is a symbolic link, or that group or others can
  write to is refused by name and that process compiles into a fresh private directory instead. Where a kit sets no
  directory the framework's own default applies as shipped (torch's `/tmp/torchinductor_<user>`, Triton's
  `~/.triton`); point `TORCHINDUCTOR_CACHE_DIR` / `TRITON_CACHE_DIR` at a private directory on a shared host.
- **`chai1` compiles model code shipped inside its weights archive.** The TorchScript class sources are read from the
  digest-pinned archive itself at every load; no file found in a temporary or shared directory is compiled or
  imported.
- **The compile-cache root `run.sh` picks is private.** With no `MODEL_OPT_JIT_ROOT` preset, `run.sh` gives the kits
  that seed or key their compile caches through it the per-user root `${TMPDIR:-/tmp}/model_opt_jit-uid<uid>`, made
  with mode 0700 — the same directory a cache shipped in an image, or the running stack's part of a preset root the
  process cannot write, is seeded into — and uses it only if it is a directory the invoking user owns, not a symbolic
  link and not writable by group or others; anything else is refused by name, nothing is seeded or read there, and the
  run proceeds without that root. A writable root you preset is used as given.
- **Prebuilt GPU binaries and shipped compiler caches are held to digests.** Compiled kernels (`.so`, `.cubin`,
  `.ptx`) ship with their source and build scripts so they can be rebuilt. Every prebuilt binary in the tree is held
  to a SHA-256 recorded in the tree beside it — a `SHA256SUMS` file (for a sealed kernel payload, at the payload's top
  directory, listing the binaries beneath it) or the owning package's manifest — and re-hashed before it is loaded; a
  missing, unlisted or altered file is refused by name and treated as absent, so the kit serves that kernel's fallback
  or, where it has none, reports `NOT ACTIVE`. `chrombpnet`'s shipped Triton caches are re-hashed file by file against
  their `SHA256SUMS` before they are installed into the live cache; one unlisted or differing file leaves the whole
  shipped cache aside by name and the kernels compile afresh. Environment variables that point a loader at a library,
  build or manifest you supply (`TRIATTN_XLA_LAUNCHER`, for one) load what they name, unchecked, by design.
- **Install instructions download installers.** The route-C recipes fetch `uv`'s installer script (`colabdesign`'s
  recipe, the micromamba release binary) from a version-pinned URL, and one container recipe fetches the `uv`
  installer the same way; each download is verified against a recorded SHA-256 before it is run or installed, and the
  recipes skip it when the tool is already on `PATH` — install the tool from your package manager first if you prefer.
- **Build-time downloads are verified.** The container recipes check the SHA-256 of every archive they download while
  building — interpreters, source archives, wheels fetched by URL — and stop the build on a mismatch; packages
  installed from the package index are pinned by exact version in `requirements.lock`, not by hash; `af3_torch`'s fork
  of `alphafold3`, which no index carries, has no installable line in its lock and is installed from the wheel built
  from its pinned source (or the digest-checked copy under `stock/wheels/`), never by name from an index.
- **Helper processes start without a shell.** Where a kit launches worker processes it passes argument lists rather
  than shell command strings and resolves the launcher's path canonically first. The one exception is the shared
  multi-process packing launcher (`common/mps_packing/mps_workers.sh`, which `rfdiffusion1`'s packed runs use), which
  by design runs each worker's command line through `bash -c`; the kit builds that line itself and shell-quotes every
  argument in it. Per-user scratch and helper-script directories follow the same private-directory rule as the compile
  cache.
- **`rfdiffusion1`'s schedule cache is upstream's own pickle.** Upstream RFdiffusion caches its noise schedule as a
  pickle under its checkout and loads it without a digest. When the invoking user cannot write that directory, or
  other users can, the kit passes upstream a per-user directory (`${TMPDIR:-/tmp}/rfdiffusion1_schedules-uid<uid>`)
  held to the private-directory rule above; if that path is refused too, the schedule is recomputed in a private
  temporary directory for the run. A schedule directory you name yourself (`inference.schedule_directory_path=` or the
  route-B bind) is used as given.
- **`af3_torch`'s stock comparison runs an interpreter you name.** `bash run.sh stock` starts the interpreter in
  `AF3_TORCH_STOCK_PY`, which has no default; one whose file, directory or venv belongs to neither the invoking user
  nor root, or that sits in a directory others can write to, is refused by name and never started.
- **Containers run as root.** The Docker and Apptainer recipes do not create an unprivileged user; run them with your
  platform's usual isolation (rootless runtime, no extra capabilities, only the mounts the kit README names).
- **Weights, databases and one helper executable are fetched from upstream URLs** and verified against SHA-256 pins —
  `colabdesign`'s `install` fetches BindCraft's `dssp` executable this way, from the BindCraft repository at the
  pinned commit; a mismatch is a hard refusal. Their licences are set by their providers (see each kit's `STOCK.md`
  and `THIRD_PARTY_NOTICES.md`).
- **Two upstream defaults query a public alignment server.** Stock ColabFold builds an MSA by sending the query
  sequences to `api.colabfold.com` unless the input already carries an alignment or its `--msa-mode single_sequence`
  is given, and OpenFold3's `--use-msa-server` defaults to on; `colabfold` and `openfold3` pass these defaults through
  unchanged, so give precomputed alignments or turn the server off when sequences must not leave the machine.
- **No network listeners.** The kits open no service ports; the only sockets are loopback rendezvous ports for
  single-host tensor-parallel workers.

## Hardware

- Every kit is configured and stated for the NVIDIA H100 80 GB (`configs/h100.env` in most kits) on Linux x86-64.
- The NVIDIA driver must be at the floor the kit's Setup states (525 to 580 depending on the CUDA stack); this is a host
  requirement on every route, containers included.
- A kit that ships `configs/a100.env` (or an H200 / B200 / B300 file) runs on that card as its README's Notes describe.
  The kit reads the card at start-up; an optimization without a kernel table for that card is left out and named on the
  `ACTIVE` line, or the mode exits 3 naming it.
- One pinned software stack per kit, stated in its `STOCK.md`.

## Kits

| kit | upstream (exact pin: the kit's `STOCK.md`) | upstream licence | task | modes shipped |
|---|---|---|---|---|
| [`af2ig`](af2ig/README.md) | AF2 initial guess — nrbennet/dl_binder_design (AlphaFold2 `model_1_ptm`) | MIT; AlphaFold 2 code Apache-2.0 | binder–target interface scoring | off · exact · fast · big |
| [`af3_jax`](af3_jax/README.md) | AlphaFold 3 inference code (JAX fork) on converted OpenFold3-preview2 weights | Apache-2.0 | cofolding | off · exact · fast · big · `--n_gpu P` |
| [`af3_torch`](af3_torch/README.md) | xfold (PyTorch AlphaFold 3 forward pass) on OpenFold3-preview2 weights | Apache-2.0 (note 1) | cofolding | off · exact · fast · big · `--n_gpu P` |
| [`atlasfold`](atlasfold/README.md) | atlasfold 1.0.0 | MIT | structure prediction | off · exact · fast · big |
| [`boltz2`](boltz2/README.md) | Boltz-2 (`boltz` 2.2.1) | MIT | cofolding | off · exact · fast · big · `--n_gpu P` |
| [`boltzgen`](boltzgen/README.md) | BoltzGen 0.3.2 | MIT | binder design | off · exact · fast · big |
| [`borzoi`](borzoi/README.md) | Borzoi (calico, TensorFlow) | Apache-2.0 | variant effect scoring | off · exact |
| [`caliby`](caliby/README.md) | Caliby / SolubleCaliby | Apache-2.0; atomworks-caliby BSD-3-Clause; Protpardelle-1c MIT | sequence design | off · exact · fast |
| [`chai1`](chai1/README.md) | Chai-1 (`chai_lab` 0.6.1) | Apache-2.0 | cofolding | off · exact · fast · big |
| [`chrombpnet`](chrombpnet/README.md) | ChromBPNet 1.0.1 | MIT | chromatin accessibility prediction | off · exact · fast |
| [`colabdesign`](colabdesign/README.md) | BindCraft on ColabDesign 1.1.3 (AlphaFold-Multimer) | MIT (BindCraft); Beerware (ColabDesign) | binder hallucination | off · exact · fast |
| [`colabfold`](colabfold/README.md) | ColabFold 1.6.1 (`colabfold_batch`, AlphaFold2-Multimer v3) | MIT; alphafold-colabfold Apache-2.0 (note 2) | structure prediction from MSAs | off · exact · fast · big · `--n_gpu P` |
| [`complexa`](complexa/README.md) | Proteina-Complexa 1.1.0 | Apache-2.0 | binder generation | off · exact · fast · big |
| [`e1`](e1/README.md) | E1 (Profluent) 150m / 300m / 600m | Profluent-E1 Clickthrough License Agreement (note 3) | protein LM mutant scoring | off · exact |
| [`ef2inv`](ef2inv/README.md) | ESM cookbook binder design through ESMFold2 (`esm` 3.4.0) | MIT (esm); Apache-2.0 (transformers fork) | gradient-based binder design | off · exact · fast · big |
| [`enformer`](enformer/README.md) | `enformer-pytorch` 0.8.12 | MIT | genomic track prediction | off · exact |
| [`enformer_deepmind`](enformer_deepmind/README.md) | Enformer, official TensorFlow release (TF-Hub) | Apache-2.0 | genomic track prediction | off · exact |
| [`esm_if1`](esm_if1/README.md) | ESM-IF1 (`fair-esm` 2.0.1) | MIT | inverse folding | off · fast |
| [`esmc`](esmc/README.md) | ESM C 300m / 600m / 6b (`esm` 3.4.0) | MIT | protein LM inference | off · exact |
| [`esmfold2`](esmfold2/README.md) | ESMFold2 and ESMFold2-Fast | MIT (esm); Apache-2.0 (transformers fork) | structure prediction | off · exact · fast · big · `--n_gpu P` |
| [`evo2`](evo2/README.md) | Evo 2 7b / 40b (`evo2` 0.6.0) | Apache-2.0 | genomic LM scoring and generation | off · exact · fast |
| [`flashzoi`](flashzoi/README.md) | Flashzoi (`borzoi-pytorch` 0.5.1) | Apache-2.0 | genomic track prediction | off · exact |
| [`genie3`](genie3/README.md) | Genie 3 (aqlaboratory) | Apache-2.0 | backbone diffusion, binder design | off · exact · fast |
| [`gpnstar`](gpnstar/README.md) | GPN-Star (`gpn` 0.9.0; `songlab/gpn-star-hg38-v100-200m`) | MIT | variant effect scoring from whole-genome alignments | off · exact |
| [`mosaic`](mosaic/README.md) | mosaic (escalante-bio) driving Boltz-2 through joltz, JAX | MIT | binder hallucination | off · exact · fast · big |
| [`opendde`](opendde/README.md) | OpenDDE 1.1.1 | Apache-2.0 | cofolding | off · exact · fast · big · `--n_gpu P` |
| [`openfold3`](openfold3/README.md) | OpenFold3 0.4.1 (OpenFold3-preview2 weights) | Apache-2.0 | cofolding | off · exact · fast · big · `--n_gpu P` |
| [`openfold3_ob0`](openfold3_ob0/README.md) | OpenFold3 0.5.0 (OpenBind-0) | Apache-2.0 | cofolding | off · exact · fast · big · `--n_gpu P` |
| [`progen2`](progen2/README.md) | ProGen2 (salesforce/progen) | BSD-3-Clause | protein LM sampling and likelihood | off · exact |
| [`proteinmpnn`](proteinmpnn/README.md) | ProteinMPNN (dauparas; `vanilla` and `soluble` weight sets) | MIT | sequence design | off · exact |
| [`protenix_v1`](protenix_v1/README.md) | Protenix 1.1.0 | Apache-2.0 | cofolding | off · exact · fast · big · `--n_gpu P` |
| [`protenix_v2`](protenix_v2/README.md) | Protenix 2.0.0 | Apache-2.0 | cofolding | off · exact · fast · big · `--n_gpu P` |
| [`pxdesign`](pxdesign/README.md) | PXDesign (Protenix 0.5.0+pxd) | Apache-2.0 | binder diffusion | off · exact · fast · big |
| [`rfdiffusion1`](rfdiffusion1/README.md) | RFdiffusion 1.1.0 | BSD-3-Clause | backbone generation | off · exact · fast |
| [`rfdiffusion3`](rfdiffusion3/README.md) | RFdiffusion3 (foundry `rfd3`) | BSD-3-Clause | all-atom backbone generation | off · exact · fast |
| [`rosettafold3`](rosettafold3/README.md) | RoseTTAFold3 (foundry `rf3`) | BSD-3-Clause | cofolding | off · exact · fast · big · `--n_gpu P` |

The licence column gives the licence of the upstream code each kit carries under `stock/` (SPDX identifiers where one
exists). Model weights are not in the tree; they come from their providers under the providers' own terms, which each
kit's `STOCK.md` names, and every further third-party component — including third-party files inside a kit's own
directories — is listed in the kit's `THIRD_PARTY_NOTICES.md` and summarised in `NOTICE`.

- note 1 (af3_torch): 24 upstream files carry stale CC BY-NC-SA header text from an earlier release; upstream's licence
  is Apache-2.0 (see `af3_torch/THIRD_PARTY_NOTICES.md`).
- note 2 (colabfold): each carried wheel includes one LGPL-3.0 data file, OpenStructure's `stereo_chemical_props.txt`
  (see `colabfold/THIRD_PARTY_NOTICES.md`); byte-identical copies of the same file travel inside af2ig's and complexa's
  carried source archives (`NOTICE`).
- note 3 (e1): Profluent-E1 is distributed under a click-through agreement (`e1/stock/src/LICENSE`, with
  `e1/stock/src/ATTRIBUTION`); recipients of the kit receive Profluent-E1 under that agreement. Its Attribution
  Guidelines require attribution to "Profluent-E1" in distributions and documentation and, for redistribution in object
  form, a "Profluent-E1" display each time the program runs; `e1/run.sh` prints that attribution at the start of every
  command.
