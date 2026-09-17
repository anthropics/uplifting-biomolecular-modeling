# opt/forward/xattempt_addon

The kit modes' runner and two exact levers; nothing in the installed `boltzgen` tree is modified. This `src/` goes first on `PYTHONPATH`,
ahead of `opt/forward/fast_inference/src`; `opt/boltzgen_opt/modes.py` composes the modes, `boltzgen/CHANGES.md` describes each lever.

- `src/xa_run.py <run_dir> <seed> [steps]` — the runner every kit mode launches on one configured job directory (the output of
  `boltzgen configure`): the GPU steps in one process through `bg_inproc.py`, the CPU steps (`analysis`, `filtering`) afterwards as
  stock subprocesses; sets the switch defaults `BG_GRAPH=graph XA_FAST_INIT=1 XA_HOIST=1` and imports the two lever modules below.
- `src/xa_fastinit.py` — `fastinit` (`XA_FAST_INIT=1`): model-construction weight initialisations skipped under the strict checkpoint load.
- `src/xa_hoist.py` — `hoist` (`XA_HOIST=1`): the diffusion transformers' step-invariant attention masks built once per structure.

Modes: `exact` and `fast` use all three; `big` uses the runner and `fastinit` (`XA_HOIST=0`).
