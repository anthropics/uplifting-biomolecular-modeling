# common — shared libraries of the optimization kits

Two directories the engine kits in this repository (`<engine>/`, beside `common/` at the top of the tree) depend on; each directory's own `README.md` is its reference.

- `opt_core/` — the `opt_core` Python package (standard library only) that kits import for the mechanisms no single engine owns: the mode vocabulary below, the activation and refusal lines and exit codes, `opt_manifest.json`, the autoload hook behind `<ENGINE>_OPT=<mode>`, GPU and software-stack gates, and shared levers (triangle-multiplication and attention lever adapters, precision policy, CUDA-graph capture, the `big` memory levers and `--n_gpu` sharding, carried Triton and Pallas kernels). `opt_core/kit_template/` holds the two files kits copy byte-for-byte: `_build_backend.py` (beside the kit's `pyproject.toml`; installs the kit's `<pkg>_autoload.pth`, if it has one) and `_core_gate.py` (inside the package of a kit that pins the core).
- `mps_packing/` — the CUDA MPS co-tenancy launcher: `mps_workers.sh` runs K copies of one serving-worker command on one GPU (`K=3 WORKER_GB=<per-worker GB> bash mps_workers.sh '<worker command; {W} = worker index>'`; optional `HEADROOM_GB`, `LOG_DIR`); `rfdiffusion1`'s `run.sh design --pack K` drives it. Scripts, not a package.

## How a kit depends on opt_core

```bash
pip install -e common/opt_core -e <engine>/opt    # from the top of this repository, into the engine's own environment: core first, then the kit, both editable
```

The kit pins the core in its `opt/pyproject.toml`: a `[tool.opt_core]` table with `path = "../../common/opt_core"` and `version = "<minimum core version>"` — a floor, so any installed core at or above it (`opt_core.__version__`) satisfies the pin. Such a kit's entry points call `gate(__file__)` from its `_core_gate.py` copy before importing `opt_core`; an absent or older core prints `[<engine>-opt] NOT ACTIVE: reason=core_missing:opt_core …` or `reason=core_mismatch: …` and exits 3 rather than raising. A kit whose `opt/pyproject.toml` has no `[tool.opt_core]` table does not import the core and installs alone (`pip install -e <engine>/opt`).

## Mode vocabulary (`opt_core.modes`, `opt_core.mem`)

- `off` — stock: no kit variable set, no lever applied. `exact` — outputs identical to `off` under the kit's deterministic recipe. `fast` — the kit's speed levers; outputs within the tolerance the engine's README states. `big` — the memory mode: `fast` (or `exact` where a kit has no `fast`) plus memory levers so larger inputs fit one GPU, plus `--n_gpu P` in kits that can shard one input over P GPUs (default 1; `P > 1` only under `big`). A kit ships a subset of these, always including `off`, and may add modes of its own; a name outside its table is refused by name, never aliased.
- Precedence: `--mode` on the command line, else the kit's `<ENGINE>_OPT` variable, else the kit's default. A run under a kit mode prints one `[<engine>-opt] ACTIVE mode=…` line on stderr and writes `opt_manifest.json` beside its outputs; a mode that cannot engage prints `[<engine>-opt] NOT ACTIVE: <reason>` and exits 3. Exit codes: 0 ok · 1 failed or incomplete · 2 usage · 3 not active (refused, or a lever of the mode did not apply).
