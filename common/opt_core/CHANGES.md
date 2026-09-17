# What opt_core changes in an engine process

opt_core patches nothing by being installed or imported. Everything it changes in a running
engine happens when a kit's `enable()` asks for it, and is named in that kit's `CHANGES.md`
lever by lever. The mechanisms a kit can switch on from here:

- `autoload` — a `sys.meta_path` finder (installed from the kit's `.pth` when `<ENGINE>_OPT`
  is set) that runs the kit's `enable()` at the first import of a trigger module, and an
  `atexit` hook that prints the `[opt_core] CELLS` census line.
- `kernels.route` — a finder that serves the carried kernel modules under their bare names,
  so an engine module and a kit import the same single copy.
- kernel providers — replace an engine's op with the selected row for the call class; the
  `exact` tier serves only rows recorded as bitwise identical to the engine's op for that
  class and otherwise leaves the engine's op in place, by name.
  A row's `vouched_on` list names every stack the row equals the stock op on bit for bit, H200
  included (`tests/fixtures/h200_exact_vouch_rows.json` lists those rows).
- `precision` — matmul / reduction precision flags set to the values the kit's mode declares,
  recorded in the run record.
- `mem` (`big`) — chunk sizes, activation offload, checkpoint placement, allocator settings
  and the row-sharded pair stack, each applied by the kit's patch set and recorded in the run record (`mem.record`).
- `capture` — CUDA-graph capture and replay of the loops a kit names (`capture.graphs`,
  `capture.pool`, `capture.hoist`).

`opt_core.__version__` is the interface version kits pin as a floor; git history is the
changelog.
