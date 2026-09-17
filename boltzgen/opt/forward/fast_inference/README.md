# opt/forward/fast_inference

Lever modules imported at interpreter start in the kit modes; nothing in the installed `boltzgen` tree is modified.
`opt/boltzgen_opt/modes.py` composes them; `boltzgen/CHANGES.md` describes each lever.

- `src/sitecustomize.py` → `src/bg_hook.py` — every process of a kit-mode job: the seed recipe's featurizer fix, the `graph_sampler`
  switch (`BG_GRAPH`), per-step timing records appended to `BG_TIMING_FILE` at exit.
- `src/bg_inproc.py` — `inproc`: the GPU steps of one configured job directory in one process, step *i* seeded `seed + i` (run by `xa_run.py`).
- `src/bg_graph_patch.py` — `graph_sampler`: `AtomDiffusion.sample` with the per-step denoiser call captured once per shape and replayed.
- `tests/specs/pdl1_ref.yaml` (+ its target PDB) — the example design spec the kit README's commands and `run.sh warm` use.

Modes: `exact`, `fast`, `big` (`big` exports `BG_GRAPH=off`: the hook and `inproc` only).
