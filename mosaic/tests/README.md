# mosaic — tests

The package's CPU tests (no GPU; expected green on a CPU-only interpreter — inside a GPU environment some cases do not hold by construction, see
`opt/mosaic_opt/tests/__init__.py`): `python -m pytest opt/mosaic_opt/tests`, from the tree, in an environment with `pytest` (the core-gate
route test is pytest-style). They lock the package to the kit — the mode table's rows and the default mode (the literal
`DEFAULT_MODE`), the resolver composing the row's environment and flags (`off` = row A, or D on frozen features), the deterministic recipe, the
registry's names equal to the kit's symbols, the three stock archives at the recipe with the
ProteinMPNN weight sizes of `stock/PINS.json`, the core pin gate through every documented entry on an absent and an older core
(`test_core_gate_routes.py`: `python -m mosaic_opt`, the console script's body, `enable()` / `status()`, `run.sh` with a kit mode, the stock route and
a verb without a mode, `source configs/h100.env`, the real `.pth` from a temporary site dir under a set `MOSAIC_OPT` — one `NOT ACTIVE: reason=…` line
and exit 3 each, never a traceback; `off` / unset and the bare package import stay core-free), the activation refusals by name (no GPU, pins, an
installed `mosaic.fast` that differs from the kit, no cache directory, precision variables, P1 already on, backend initialised, an existing `Boltz2`
instance), the partial in-process report, idempotency, the instance counter on a class with `equinox.Module`'s hashing (and on a real
`equinox.Module` when equinox is installed: `pip install equinox` needs no GPU), the counter registered only after every gate, the `.pth` hook (fires
once after `mosaic`'s body, exit 3 on refusal, the kit driver never levered, nothing heavy at interpreter start), the stock route's stripped
environment, proof and install layout, `design` / `check` / `warm` against a stand-in driver (the row's variables seen by the driver, the driver's
manifest showing P1/P2/P3, a warm shape refused, the exit codes, the exit tally) and the core locks (`test_core_adoption.py`: the `[tool.opt_core]` pin against the tree's core and equal to the installed
core through `mosaic_opt.core_gate()`, the gate as statement one of every entry, a core not at the pin refused by name with status 3, the build
backend and `_core_gate.py` against the installed core's kit templates, the `.pth` equal to the backend's generated text, the one tag spelling, the
hook's import-free copies, the partial-report grammar, the stack key and the P1 variables of the rows against `opt_core.jax_design.pcc`). OOM
propagation through the served in-process entry is in `test_activation_and_launch.py` (a lever stand-in raising torch's / jaxlib's / the host's out-of-memory
comes out of `enable()` unchanged; any other failure stays the NOT ACTIVE report). No design is computed by these tests; `opt_core` is imported from
the install, or from the pinned tree when it is not installed (`tests/__init__.py`). The install step is `test_install_verb.py`: `run.sh install [--weights DIR]`'s argument handling and call order against a stub `python`
(the editable install or its skip, the pin check, the lever files, the weights), `mosaic_opt.leverfiles` against a stand-in installed `mosaic`
package (add-only, idempotent, extra files named and kept) and `mosaic_opt.weights` with the fetch routine injected (the checkpoint's digest
against `stock/PINS.json` "weights", a file off its pin refused and left in place, the pin equal to the fetch tool's own constant).

## The stock route

`mosaic-opt design --mode off --out DIR` runs the kit driver with every lever off (the stock row `A_stock1`: the
driver's default weight load, no P1 environment) through the package's one stock caller, `opt/mosaic_opt/stock_design.py`, in a subprocess whose
environment is stripped of every prefix in `stock/PINS.json` "stock_environment.must_be_absent_prefixes" (`MOSAIC_CACHE_DIR` the one exception); the
caller proves the environment in its own process before the driver imports jax, records the install layout it runs on — the pinned stack without the
kit's lever files (the stock install) or with them (the installed `mosaic.fast` present with every carried kit file; a copy missing kit files or carrying extra ones is
refused) — and prints the `ENV-CLEAN` line (`mosaic.fast=`, `stock_install=`).

Run by hand (the driver `python $MODEL_OPT/opt/mosaic_opt/tools/public_design_run.py ...`), `MOSAIC_OPT` must be unset: the package's
`.pth` hook fires in every python process of the environment that imports `mosaic`, and with `MOSAIC_OPT=exact` it refuses the kit driver (`NOT
ACTIVE`, exit 3) rather than lever a stock arm silently.
