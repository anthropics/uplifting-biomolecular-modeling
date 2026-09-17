# RFdiffusion3 — tests

The package's CPU tests: `opt/rfdiffusion3_opt/tests/`. They need no GPU, no torch and no upstream install; a fake `rfd3` tree and fake
interpreters stand in for the two venvs.

## Package tests (CPU, no torch, no upstream)

`opt/rfdiffusion3_opt/tests/`: `python -m pytest opt/rfdiffusion3_opt/tests` in a venv with the core and the package installed (`pip install -e ../common/opt_core -e opt[test]`;
the `test` extra brings pytest and setuptools — the adoption test loads `opt/_build_backend.py`, and a Python >= 3.12 venv has no setuptools of its own;
the editable install may leave `opt/rfdiffusion3_opt.egg-info/` in the tree — it is git-ignored and tolerated, a `build/` directory is
not) and no torch — the core pin gate on every documented route (`test_core_gate_routes.py`: `python -m rfdiffusion3_opt`, the console
script, `enable()` / `status()`, the public names that load a core-importing module (`MODES`, `stack`, … — the gated set locked against the
sources' import closure), `run.sh` verbs with a kit mode / `--mode off` / no mode / `--config h100`, `source configs/h100.env`, the real
`.pth` at interpreter start under a set `RFDIFFUSION3_OPT` (in a `-S` child through `site.addsitedir`, and in a real venv whose site-packages
carries it, where `RFDIFFUSION3_OPT=<mode> bash run.sh …` / `source configs/h100.env` must still name the core, never "not importable"), the
`install` and `stock_design` child modules — each refused by name with exit 3 and no traceback against an absent core, a stale `0.2.5` core
and a core without `MANIFEST.json`, inert and core-free with the switch unset or `off`), the core
adoption (`test_core_adoption.py`: the gate's facts hold the installed core and fill the report, a wrong pin exits 3 by name,
`_build_backend.py` and `_core_gate.py` byte-identical to the installed core's `kit_template/`, the `.pth` equal to the backend's generated
text), the mode table (exact | fast | off over the core's ModeTable — fast the default —, words that are not modes refused, the mode argument, every `hoist.py:<n>`
pointer the package cites landing on its line; the pin word `attn_pin=1` locked to the routes table's `attn_pin` rows, `test_modes.py`), the deterministic recipe (`test_det.py`: the levels and names, the variables
exported before torch, `apply()` on a stub torch — torch's switches set, upstream's scatter-mean rebound —, the stock child's `--det` applied after its
environment proof and named on the STOCK line, the level carried by the stock command and worded on the activation lines), the KERNELS census (`test_census.py`: every word's grammar against the
consumer's copies, the guard per route on fake engines, the pre-flight atom-attention decision on upstream's rule — printed once per shape with
`pinned=`, refusing a sparse start on the gated routes, PINNED per (D, L, k, H) on the kit routes: the model's later call answers dense from the pin at a
lower free memory where the rule per call says sparse, without evaluating the rule, and makes upstream's path record with `pinned at pre-flight`;
every atom head count pinned; a key the pre-flight did not evaluate and the token transformer's call run the rule; the stock routes install logging
handlers only (no wrap, no shim, no import hook: every word off upstream's own records); upstream's force-sparse switches decide the pre-flight itself; a pinned dense path's out-of-memory error propagates and the
shim's call wrapper carries no handler — and the `design` routes end to end on a fake upstream, the kit process pinned, a mid-run flip invisible to the stock process (no counts there); a missed
expectation is a refusal (exit 5) on the kit routes and one `KERNELS report-only` line with the run's own exit code on the stock routes, the pre-flight dense
bound stopping the kit routes (compiled stock names the path and runs it); upstream's own switches (`RFD3_LOW_MEMORY_MODE`, `RFD3_DENSE_SDPA_ATTENTION`) pass through the stock arm, are named on the
STOCK line (`upstream_env=`), `test_cli_stock.py`, and are refused by name under the kit modes), the kit bytes against the kit's `MANIFEST.json` (the kit directory holds nothing else), the ten-file classification against
the kit's own `install.sh --status` on four fake interpreters built from the bytes the tree carries, activation on fake interpreters
with a fake `nvidia-smi` (the gates, every refusal by name — the GPU gate on a PATH without any `nvidia-smi` — late activation, a second
mode, the environment rules, the shared report keys; the upstream pin and the caller's deterministic recipe reported, never gated — one `PINS` line per
finding —; a classifier-free-guidance, `low_memory_mode=true` or symmetry-sampler design line refused by name (and the same computations met at the engine's initialize: one line, exit 3, no lever evidence after it) — one `NOT ACTIVE` line, nothing exported, exit 3, nothing of the run started), the autoload route (fires before the trigger's body, refusal exits 3, inert
without the switch, a value that is not a mode refused at interpreter start with exit 3, the `APPLIED` hook), the `design` routes end to end
with a stub `rfd3.cli:app` (the stock proof in a real pristine venv, the manifests, the checkpoint digest cache, the exit tally),
the exit rule with the real kit module under a stub torch (`test_exit_rule.py`: full activation exits 0 with the `APPLIED` evidence; the
module reading the switch off is refused at its import — exit 3, the family line, no output — on `design` and on the `.pth` route; a
completed run that never imported it is partial; `--allow-partial` and `RFDIFFUSION3_OPT_ALLOW_PARTIAL=1` record `allow_partial` and
proceed with the run's own code; a failed run keeps its code; the stock route records the flag and strips the switch; `warm` partial /
allowed / full; a classifier-free-guidance line under `exact` declined on both routes — upstream's own run, exit 0, nothing partial — and passed
through under `off`; the family grammar byte-literal; the CFG keys read out of upstream's YAML in the stock archive),
`warm --mode off` proven stock with a stub engine (PASS on the pristine venv with every lever stripped; `NOT STOCK`, exit 3, nothing built
on a patched one), `check` / `warm` exit codes (the stock interpreter's tree the gate, its pin a report), the composed line restating no upstream
default (the YAML in the stock archive), the line's `key=value` tokens as upstream's verbatim (`--ckpt` the one token the kit adds), the removed flags and any `--word` of an upstream setting as usage errors,
the stock archive's sha256 (hashed live) against the pins,
the config and `run.sh` rules (an unknown mode reaches the package and is refused there), the packaging and interpreter start-up budget,
and the merge locks (one mode table — no copy in the autoload finder or `run.sh` — one settings table, one stock caller, no lever switch
in `configs/`, `run.sh` in step with the command list, no stray file or label in the kit directory), and the out-of-memory rule (`test_oom.py`: every broad `except` on a served module — the package's and the add-on's
ten target files — named with its class; the one reroute, the graph sampler's capture fallback, starts with `if is_oom(ex): raise` and imports
`opt_core.oom.is_oom`) — and, with a CPU torch importable (skipped by
name otherwise: the roll-out is tensor code), the roll-out driven on a fake stock sampler with the capture mocked to fail: a
`torch.OutOfMemoryError` propagates uncounted, any other failure completes the design on the eager denoiser with `fallbacks` = 1).

