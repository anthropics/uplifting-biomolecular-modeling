# fast_inference — the driver kit the chai1 modes run on

A persistent-process driver for `chai_lab.chai1.run_inference` (chai_lab 0.6.1). It folds the same inputs with the same model, weights,
MSAs, recycles, diffusion steps, samples and seeds as stock and changes only how the work is scheduled. Users do not call it directly:
`run.sh pred` / `python -m chai1_opt pred --mode exact|fast|big` launch the worker (`../errata_02/chai_worker.py`) with the mode's lever
list, and the chai1_opt package installs the eager trunk and kernel levers on top of it (`opt/chai1_opt/modes.py` is the one mode table).

| lever | what changes in execution (never in the maths) |
|---|---|
| W1 persistent modules | the six exported TorchScript components and the traced ESM2-3B model are loaded once per process instead of once per seed |
| W2 per-input feature context | FASTA parse, reference conformers, MSA load and ESM embedding are computed once per input and reused for every seed (they do not read the seed) |
| W5 cross-input ESM memo | per-sequence ESM embeddings are memoised across the inputs of one process |

Several inputs in one process share the loaded modules (the worker loops them); one input per invocation disables that sharing.
`CHAI_DETERMINISTIC=1` (the kit's `--det 1`) applies the deterministic recipe of `kit/chai_proto.py` `apply_deterministic_mode`
in the worker; `KNOWN_ISSUES.md` lists the upstream reproducibility facts behind it.

## Files

- `../errata_02/chai_worker.py` — the worker: lever install (W1, W5), the per-input loop (W2), one `run_folding_on_context` call per seed.
- `kit/chai_proto.py` — protocol helpers: the deterministic switch, `run_inference` keyword defaults, FASTA / MSA-directory handling, per-seed output capture, the environment report.
- `tests/public_inputs/` — four public inputs (PDB 1BRS barnase–barstar, tiled 1:1 … 4:4; `PACK.json`), the `warm` verb's input.
- `KNOWN_ISSUES.md`, `LICENSE`, `NOTICE`.

No chai-lab source file is modified: the kit imports `chai_lab` and wraps its inference loop in its own process only
(`chai1.load_exported`, `esm.esm_model`, `esm._get_esm_contexts_for_sequences` are replaced in memory by W1 / W5).
`LICENSE` reproduces chai-lab's Apache-2.0 licence and `NOTICE` states the derivation; chai-lab and the Chai-1 weights are Apache-2.0 per upstream; ESM-2 is MIT.
