# RFdiffusion-1 kit — what changes vs stock

Stock = RFdiffusion 1.1.0 at the pin (STOCK.md). Each lever is a module the kit installs over one stock function or class at start-up, inside its own
resident driver process, under a kit mode; `off` loads none, and the checkout on disk is never edited. A mode is all of its levers: it engages every one
or refuses by name; `fast` includes `exact`'s levers. Lever names are the ids printed in `levers=` on the run's ACTIVE line and registered in
`opt/rfdiffusion1_opt/registry.py` (mode membership: `opt/rfdiffusion1_opt/modes.py`); the code lives under `opt/forward/` — `fast_inference/drivers/`
and `se3fast_addon/rfd_se3fast/`.
| mode | levers |
|---|---|
| `off` | none — `python -s $RFD_ROOT/scripts/run_inference.py <the typed arguments>` |
| `exact` | U1 + C1 + P + E_einsum + W1 + IO1 (`--fastpath chain_breaks,full_graph,rbf,msa_index --prep 1 --einsum-route 1 --fullgraph 1`, `RFD_PDBIO=1`) |
| `fast` | `exact`'s levers + T2 + K2 + TF32 (`RFD_SE3FAST=t2`, `--triton-ln 1`, `--tf32 1`) |

## exact — outputs identical to stock

The line above runs on `drivers/rfd_bench.py` (plus `--no-traj 0|1` from `inference.write_trajectory`; `--tf32` at the driver default `0`: matmul and cuDNN TF32 off).

- `U1` — resident driver (`drivers/rfd_bench.py`): the model is built and the checkpoint loaded once per call, and every design of the request runs in that
  process; the sampler, the per-design seeding, `inference.cautious` skipping and the writers are upstream's own objects and calls. Numerics: none
  (placement only). Steps aside: never.
- `C1` — per-design constants (`drivers/rfd_fastpath.py`, `--fastpath …`): chain-break masks, the full graph, the RBF table and the MSA index are computed
  once per design shape and reused every step instead of rebuilt per forward. Numerics: bitwise (caching only). Steps aside: never.
- `P` — preprocessing (`drivers/rfd_prep.py`, `--prep 1`): `_preprocess` skips the CPU template `t2d` featurisation whose result the network never reads and
  memoises the per-design constant tensors. Numerics: bitwise. Steps aside: never.
- `E_einsum` — einsum routing (`drivers/rfd_einsum.py`, `--einsum-route 1`): two `opt_einsum` contraction signatures of the attention module run on
  `torch.einsum` (the same GEMMs without the surrounding permute / copy kernels). Numerics: bitwise. Steps aside: the driver compares both routes bitwise
  on the running card at start-up and refuses the lever by name if they differ; the mode is then NOT ACTIVE.
- `W1` — whole-forward CUDA graph (`drivers/rfd_fullgraph.py`, `--fullgraph 1`): embeddings, templates and the 36 iteration blocks are captured as one graph
  per forward phase on the first forward of a design shape and replayed afterwards; a new (contig, length) shape re-captures; the four top-k refinement
  calls stay eager. Numerics: bitwise (same kernels, same order). Steps aside: a failed capture serves that shape with upstream's eager forward, counted, and
  `design` then reports `NOT ACTIVE: partial activation` naming W1 (exit 3, outputs kept); an out-of-memory during capture is raised, not rerouted.
- `IO1` — array-based PDB writers (`opt/rfdiffusion1_opt/pdbio.py`, `RFD_PDBIO=1`): `rfdiffusion.util.writepdb` / `writepdb_multi` re-expressed over numpy
  arrays — the same format string on the same numbers, one host copy per model instead of per-atom tensor indexing. Numerics: bitwise (bytes). Steps aside:
  the first call per argument signature is compared with upstream's writer in the driver process; a difference keeps upstream's bytes, ends the driver process with status 5 (`REFUSED … exit 5` in `run.log`) and `design` reports the pass failed (exit 1).

## fast — within stock's seed-to-seed variation

`exact`'s line plus `RFD_SE3FAST=t2 --triton-ln 1 --tf32 1`.

- `T2` — dense SE(3)-Transformer layer (`se3fast_addon/rfd_se3fast/`, `RFD_SE3FAST=t2`): a destination-major dense formulation of the SE(3) layer with two
  fused Triton kernels (the radial-MLP trunk; the last radial layer × per-edge contraction) in all 40 structure-module calls per step, armed in the driver
  process before the model imports. It allocates dense (L, L, ·) edge tensors, which is why `fast` reaches shorter maximum lengths than `exact`. Numerics:
  fp32 re-association (`tl.dot` at ieee precision), run-to-run deterministic. Kernel: the kit's `rfd_se3fast/kernels.py` (Triton; launch geometry per
  compute capability in `geometry.py` — rows for 9.0 and 8.0, other capabilities take the 9.0 row). Steps aside: Triton absent or the layer not armed on
  the Triton line → the mode refuses by name.
- `K2` — Triton row LayerNorm for `F.layer_norm` in the network (`--triton-ln 1`). Numerics: fused-kernel rounding, last bits. Kernel: shared core
  `opt_core/kernels/rfd_layernorm.py` (Triton), routed to the driver's `import rfd_layernorm` by `driver_run.route_core_kernels`; the kit carries no copy.
  Steps aside: a lost Triton path refuses the mode by name.
- `TF32` — `torch.backends.cuda.matmul.allow_tf32` and `torch.backends.cudnn.allow_tf32` set before the model loads (`--tf32 1`). Numerics: TF32 GEMMs and
  convolutions (10-bit-mantissa inputs, fp32 accumulation); T2's Triton dots stay ieee. Steps aside: never.

## Every mode

- Stock exceptions: none (STOCK.md). `upstream_issues/RFD1-001_first_design_position.md` and `upstream_issues/RFD1-002_igso3_cache_write.md` are write-ups with a proposed upstream change; no mode applies a fix.
- Refusal before anything runs (`[rfdiffusion1-opt] NOT ACTIVE: <reason>`, exit 3): an unknown mode name; the package or the pinned shared core not
  importable (`_core_gate.py`); no checkout (`RFD_ROOT`) or no weights (`WEIGHTS`); a request the kit line cannot serve (STOCK.md "Flags and exit codes");
  `RFDIFFUSION1_OPT=<kit mode>` on upstream's own `scripts/run_inference.py` (refused at the import of `rfdiffusion` through the installed
  `rfdiffusion1_opt_autoload.pth`; `off` is inert there). After the pass, `design` reads the driver log for one evidence line per planned lever
  (`registry.LEVERS[*].evidence`) and reports any lever without its line as `NOT ACTIVE: partial activation — <levers>: <reason>` (exit 3, outputs kept,
  levers named in `opt_manifest.json`). An environment the tables do not name — another card class or compute capability, a Triton cache miss, a stack
  off the pin — is named on the report (`check`, `NOTE` lines, `pinned=False`) and run, never a reason to refuse.
- Hydra's `outputs/<date>/<time>/` log directory is written by `off` only; the kit modes write `run.log`, `opt_manifest.json`, `run_timings.json` beside the designs.
- `warm` runs the bundled example targets in one resident process (model built for the first target, only the target re-initialised for later ones, as
  upstream's `Sampler.initialize` reloads weights only when `inference.ckpt_override_path` changes); its designs are scratch. `design` takes one target per call.
- The drivers' other `RFD_*` environment knobs are not levers of any mode and are stripped from every child process (`stack.DROP_ENV_PREFIXES`), the
  mode's own row excepted.
- What a kit mode cannot serve is listed in `opt/rfdiffusion1_opt/upstream_args.py` (REFUSED_SWITCHES / REFUSED_VALUES / REFUSED_GROUPS) and read as
  upstream reads it: a boolean switch is off only for `false` / `0` / `null` / the empty word, a valued switch for `null` alone.

## Switches

- `--mode` / `RFDIFFUSION1_OPT` name the mode (`design`, `check`, `warm`). There is no per-lever switch: a mode runs with all of its levers or refuses.
- `--det 1` — upstream's `inference.deterministic=True` on the stock line and the kit line alike; `exact --det 1` = `off --det 1` byte for byte (STOCK.md).
- `--pack K` (`design`, `check`; `exact` or `fast`) — the mode's own line, flag for flag, in K resident workers on one GPU under CUDA MPS through
  `common/mps_packing/mps_workers.sh` (`serve.py`); each worker takes a disjoint slice of `inference.num_designs` and has the process history of a
  stock process (its first design is the fresh-process numerics class, later ones the warmed class — `upstream_issues/RFD1-001_first_design_position.md`).
  K is an axis, not a mode, and adds no lever; `--pack` under `--mode off` is refused by name. `MODEL_OPT_PACK_WORKER_GB` sizes the launcher's memory
  estimate; `MODEL_OPT_JIT_ROOT` / `MODEL_OPT_STACK_KEY` / `TRITON_CACHE_DIR` say where `fast`'s compiled kernels are cached (STOCK.md "Variables").
- `--dry-run` (`design`) resolves, gates and prints the plan (`DRY-RUN …`) and runs nothing; `check` does the same without a target and `--json` prints
  its activation report, pin report and mode table as JSON; `warm --out_dir DIR` writes one design per bundled example target and its record into DIR.
