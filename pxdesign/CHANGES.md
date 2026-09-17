# PXDesign kit — what changes vs stock

Stock = PXDesign at commit `f7884413` on Protenix `v0.5.0+pxd` (STOCK.md). Each lever is installed over stock module classes or
runner methods after `InferenceRunner.load_checkpoint` returns, under a kit mode (armed by `run.sh design`, `pxdesign_opt.enable()`
or the installed `.pth` under `PXDESIGN_OPT=<mode>`); `off` loads none and no upstream file is edited. A mode is all of its levers;
each mode below includes the previous mode's levers unless a line says otherwise. The hoist families `h1`–`h5`
(`opt/forward/hoist/pxd_xattempt/hoist.py`) are the names on the `ACTIVE … levers=` / `APPLIED` lines; the package levers
(`opt/pxdesign_opt/`: `sizeceil.py`, `rowpipe.py`, `precision.py`, `sdedup.py`; registry `registry.py`, modes `modes.py`) are the names
on `package_levers=` / `PACKAGE`, and `tf32` / `sdedup` print `LEVER name=…` lines. Everything that depends on the noisy coordinates
or the noise level runs through stock code.

## exact — outputs identical to stock

- `h1` — `DiffusionConditioning` pair path `pair_z = LN+Linear(cat(z_trunk, relpe)) + transition_z1 + transition_z2` and the
  single-path base `Linear(LN(cat(s_trunk, s_inputs)))`, computed once per `sample_diffusion` call with the stock sub-modules and
  replayed at every denoiser step. Numerics: bitwise under `--det 1` (scheduling and caching only). Steps aside: never.
- `h2` — per `DiffusionTransformer` block (16 blocks × 16 heads), the token pair bias `Linear(LN(z_pair))`, cached likewise.
- `h3` — the `AtomAttentionEncoder`'s step-invariant part: `c_l`, `p_lm` (including the small MLP), dense-trunk geometry.
- `h4` — per `AtomTransformer` block (4 encoder + 4 decoder), the local pair bias `Linear(LN(p_lm))` and the AdaLN(s) terms of the
  atom `ConditionedTransitionBlock`s.
- `h5` — the `_local_attention` padding-mask bias (the `attn_bias=None` branch), cached per shape (`PXD_HOIST_MASK`).
`exact` exports `PXD_HOIST_MODE=shape`: the hoisted ops are evaluated on `N_sample`-expanded rows exactly as stock evaluates them, and
row-block 0 is kept alone only when every row-block is `torch.equal` to it (otherwise the full tensor is kept), so every element
equals stock's. Plain PyTorch; nothing compiled; no kernel of the shared core is bound by any mode of this kit.

## fast — within stock's seed-to-seed variation

The default mode. `h1`–`h5` with `PXD_HOIST_MODE=rows` (hoisted ops evaluated once on un-expanded rows and broadcast over `N_sample`; numerics: fp32
re-association, since cuBLAS may pick kernels by `M`), plus:
- `featdiet` — the featurizer's `[N_atom, N_atom]` int64 `bond_mask`, a training-loss input no inference path reads, is dropped in
  `InferenceRunner.predict` before `to_device`. Numerics: exact (host-side only). Steps aside: never.
- `padmask` — the atom transformer's windowed padding mask / bias `[n_trunks, 32, 128]` is built from indices instead of being
  sliced out of a dense `N_atom²` tensor. Numerics: element-identical to stock. Steps aside: never.
- `tf32` — the numerics policy `tf32` of the shared core (`opt_core.precision.policy`: matmul precision `high`, cuDNN TF32 on), set
  after the weights load: every fp32 matmul of the process runs as TF32 tensor-core products with fp32 accumulation. Numerics: TF32
  matmuls. Its `LEVER` line reports the live switches; TF32 not live at exit → `NOT ACTIVE: lever run-time census`, exit 3.
- `sdedup` — the token path's single conditioning `s(t̂)` and the six `c_s→c_token` projections computed from `s` alone are
  evaluated on one sample row and broadcast over `N_sample` (the rows are identical by construction; the shared core's
  `opt_core.capture.hoist.RowDedup` through the hoist's `pxd_xattempt.fuse` hook). Numerics: fp32 re-association. Steps aside: a
  call whose rows differ evaluates the stock rows (counted `fallback_by=nonuniform`); at `N_sample 1` there is one row and nothing to
  fold — `LEVER … dedup=idle`, the mode stays active; a run at `N_sample` > 1 in which it served nothing → `NOT ACTIVE`, exit 3.

## big — lowest peak GPU memory

Everything `fast` does, plus:
- `rowpipe` — the hoist's three pair-plane passes (`h1` pair path, `h2`'s 16 biases, the encoder pair term) are evaluated in balanced
  row slabs (at most `rowpipe.ROWS` token rows per slab and a per-slab GEMM row cap chosen by the device's compute capability,
  `rowpipe.GEMM_ROWS_BY_CC`; the prepare line names both) and written into preallocated outputs; no `[N_tok, N_tok, c]` pair plane
  is ever resident. Numerics: as `fast` (fp32 re-association). Steps aside: never; defined for `PXD_HOIST_MODE=rows` only.

## Every mode

- The stock exception (STOCK.md §Stock exceptions: `LAYERNORM_TYPE` from `--use_fast_ln`) applies identically on every mode, `off`
  included. There are no opt-in upstream fixes.
- Engagement: a lever engages wherever its mechanism applies. A GPU class other than sm80 / sm90, a torch build other than the pin,
  a `MODEL_OPT_TARGET_GPU` that does not match the card, or TF32 override variables in the environment are named on the activation
  line (`notes=…`), never a reason to disengage; lever sets do not change with the card and nothing reads the card's memory size.
- Refusals, by name (`NOT ACTIVE`, exit 3, on every route): a hoist family or package lever that cannot apply to the loaded model
  (`fallbacks=` on the `APPLIED` / `PACKAGE` lines: `lever application incomplete`); a planned lever whose run-time evidence disagrees
  with its plan, or a hook that never ran (`lever run-time census: …` — after the run by `design`, at interpreter exit on the
  `PXDESIGN_OPT` route); no visible GPU; the hoist files absent; a `ProtenixDesign` built before activation; the shared core or the
  installed upstream not the pinned one. `off` is never gated on the upstream pins. Out of memory propagates as raised.
- Run accounting (`design`): the exit code is the shared core's rule (`opt_core.report.verdict`) — the run's own non-zero code stands;
  outputs short of tasks × seeds × `N_sample` → 1 (`INCOMPLETE …`); an exit-census sentence → 3 — and `opt_manifest.json` records it.
  Upstream's resume rule is honoured: (task, seed) directories already holding upstream's success marker are skipped by upstream and
  counted (`DONE … skipped_existing=<k>`); a job whose every pair was already dumped prints `NOTHING RAN …`, makes no mode claim and
  leaves the earlier manifest untouched (`warm` on such an `--out_dir` fails by the same rule).

## Switches

- `--det 0|1` — `1` = the deterministic recipe (`CUBLAS_WORKSPACE_CONFIG=:4096:8`, Protenix's `seed_everything(seed,
  deterministic=True)`) in every mode, `off` included; `0` (default) = upstream's own seeding.
- `PXD_HOIST`, `PXD_HOIST_MODE` (`shape` | `rows`), `PXD_HOIST_MASK` — the hoist's switches (`registry.KIT_SWITCHES`; unset, the hoist's own
  defaults are `1`, `shape`, `1`). Each kit mode exports all three (`exact`: `1` / `shape` / `1`; `fast`, `big`: `1` / `rows` / `1`; the
  `ACTIVE` line's `env=`), `off` sets none; a caller-set one is dropped at activation and named (`notes=`). Deployment variables: STOCK.md.
