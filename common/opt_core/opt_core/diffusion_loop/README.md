# `opt_core.diffusion_loop` — sampling-loop mechanisms

How a captured or batched sampling step keeps the stock random stream, the refuse-to-patch gate every loop adapter passes first, and the host-sync census of a step. (Step-invariant memoisation — 'compute once per roll-out, reuse per step' — is `opt_core.capture.hoist`.)
Engine-free (no kit is imported, no engine is named), framework-lazy (torch is imported inside the functions that build or fetch
generators; nothing at module import), values opaque (tensors and equality are the caller's). A kit's adapter — inside the kit — names the
stock function, its guard and the generators; these modules hold the bookkeeping, the refusals and the counters the kit prints in
its ONE activation line per arm per process.

## Modules, by mechanism

| module | mechanism | what it provides |
|---|---|---|
| `opt_core.diffusion_loop.source_guard` | scaffold.gates | `source_guard(fn, expected_sha256=..., contains=...)` → `opt_core.gates.Gate` (the refuse-to-patch rule over a stock function an adapter replaces: unknown upstream text = a recorded refusal, nothing is installed, the stock function stays), `source_sha256(fn)` (the digest an adapter records). Step-invariant / trajectory-constant MEMOISATION is `opt_core.capture.hoist.ConstMemo` (the one const memo of the core; `value(name, key, compute)` is the explicit value-key form a kit uses when each step passes new tensor objects of equal content) |
| `opt_core.diffusion_loop.rng` | loop.rng_discipline | one state API on generator objects (`None` = the device default generator; `torch.cuda.get_rng_state(dev)` is `default_generators[dev].get_state()`); `generator_checkpoint([...])` (warm-up draws undone by restoring every listed generator); the declared capture contract `RNG_NONE` · `RNG_OUTSIDE` · `RNG_INSIDE`, `register_generators(graph, [...])` (explicit for every generator the captured region draws from, default included; without `CUDAGraph.register_generator_state` the default generator is recorded `registered_default_auto` — torch registers it at capture begin — and an explicit non-default generator is the named refusal `RngContractUnsupported`), `capture_contract_gate(contract, modules)` → `Gate` (modules in training mode inside the region are refused unless `RNG_INSIDE`), `offsets_moved([...], fn)` (the check of an rng-none / rng-outside claim); Philox positioning for batched or padded items: `philox_increment(draw, device)` (MEASURED per op and shape, never assumed), `positioned_generator(device, seed, offset)`, `fork_at_offsets(seed, offsets, device)` (one generator per item drawing inside a captured step), `reposition(gen, base, n_calls, increment)` (after a padded replay), `StreamCursors(gen, initial_states)` (K streams over ONE generator by state swapping — items drawn eagerly one at a time, each from the state its own run would start at; no default); `RngLedger` |
| `opt_core.diffusion_loop.sync_census` | loop.sync_census | `sync_census(step_fn)` → `SyncCensus{syncs, sites[file:line, count, text]}` under `torch.cuda.set_sync_debug_mode("warn")` (previous mode restored; foreign warnings re-emitted), `SyncCensus.at_most(n)` → `Gate`. A development aid and a regression guard for sync-free step levers; changes no numerics |
| `opt_core.diffusion_loop.evidence` | scaffold.gates | `census_gate(observed, expected)` → `Gate` (fail-closed: an observed fallback count above the kit's expectation, or an expected count not observed, is a refusal). The activation LINE is the kit's: one line per arm per process from `opt_core.report.prefix` / `kv` over each mechanism's `evidence_fields()` — this package prints nothing |

## Contracts a kit relies on

- **A changed upstream is a refusal, not a patch.** `source_guard` returns a refused `Gate` naming the function and the observed digest; the
  adapter installs nothing and the kit records the gate beside its other gates.
- **RNG-INSIDE capture** = register every generator the captured region draws from BEFORE capture; warm-up draws are undone by restoring
  generator state; each replay then advances seed/offset exactly as the eager call does. Explicit `register_generator_state` is required for
  non-default generators (the method exists from the torch version a donor kit's version gate names — unchecked here: unknown) and
  harmless for the default one (which torch registers itself), so registration is explicit always. **RNG-OUTSIDE** = every draw
  stays outside the graph in stock order (`offsets_moved` proves the region draw-free). A region holding modules in training mode is
  captured only under RNG-INSIDE (`capture_contract_gate`).
- **Per-item streams.** An item padded into a batch, or replayed for more steps than its own run, consumes its own stream as if it had run
  alone: measure the per-call increment, fork or cursor the items, reposition after the padded replay. Increments differ per op
  (`multinomial`, `exponential_`, `randn`) and per operand shape — measure each distinct stock call.
- **One line per arm per process.** `RngLedger` and `SyncCensus` return `evidence_fields()` in a fixed key order; the kit folds them into its
  own line (`opt_core.report.prefix` / `kv`) and gates its fallback counts with `evidence.census_gate` against the counts it expects.

## Tests

`tests/test_diffusion_loop_guard.py`, `tests/test_diffusion_loop_rng.py`, `tests/test_diffusion_loop_sync.py` — CPU, no torch (generators,
graphs and torch's sync-debug surface are stand-ins with the duck-typed methods the modules call).
