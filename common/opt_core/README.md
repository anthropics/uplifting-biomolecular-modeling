# opt_core — the shared core of the optimization kits

`opt_core` is the one package every kit in this tree imports for the machinery that is not specific to an engine:

- mode activation and the refusal contract (a mode either engages or says by name why it cannot);
- the run record (the `ACTIVE`, `LEVER` and `EXIT` lines and `opt_manifest.json`);
- kernel providers and the kernels they serve;
- the `big` memory mode;
- framework glue for PyTorch and JAX.

`opt_core` carries no engine knowledge. A kit passes in its mode table, its trigger modules and the names of its
optimizations ('levers' in the code and in the printed lines), and binds the kernel providers through thin adapters under
its own `cells/` directory.

## Install

    pip install -e common/opt_core -e <engine>/opt

- Standard library only at import time; torch / jax / triton are imported by the code paths that need them, inside the
  engine's own environment.
- A kit pins the core by path and minimum version in its `opt/pyproject.toml` (`[tool.opt_core] path, version`). The
  kit's `_core_gate.py` refuses by name when no importable core meets that floor, and the run record names the version
  and directory of the core the run actually used.

Host floor set by the prebuilt CUDA extensions (they are loaded lazily by the kernel providers):

| component | needs |
|---|---|
| triangle-attention sm_90a extension, the `triattn_native` package members, the DiT pair-bias exact kernel | libstdc++ from GCC 13 (`GLIBCXX_3.4.32`) and glibc 2.32 |
| fused-transition and TriMul sm_90a extensions | `GLIBCXX_3.4.21` and glibc 2.32 |
| XLA triangle-attention libraries | glibc 2.34 |

A process whose libstdc++ or glibc is older cannot load them. The framework environments the kits install carry a
libstdc++ that meets the floor.

## Activation and modes

How a kit switches on without touching the engine's entry points:

- `<ENGINE>_OPT=<mode>` selects a mode.
- The kit's `.pth` file calls `opt_core.autoload`, which installs a meta-path finder; the first import of one of the
  kit's trigger modules runs the kit's `enable()`.

The modes:

- `off` — stock.
- `exact` — outputs bit-identical to stock; a faster kernel is used only where an exact implementation exists for the call.
- `fast` — kernels of the same numerics class as stock.
- `big` — `fast` plus the memory mode; `--n_gpu P` adds the row-sharded pair stack where a kit binds it.

Every optimization has exactly three possible outcomes:

1. it runs;
2. it cannot run here and refuses **by name** — the kit prints `NOT ACTIVE: <lever>: <reason>` and exits 3, or records
   the word and continues, as its mode table says;
3. an accounting check fails closed.

`MODEL_OPT_LEVERS_OFF=<word>[,<word>]` switches the named optimizations or provider rows off for one run (for example
`triattn_xla` or `triattn_xla:triattn_native`).

## Kernel providers

`opt_core.kernels.<family>` is one interface per operation:

- `select(...)` is a pure function of the call class (device class, dtype, shape bucket, form) over the family's
  measured cell table (`*_CELLS.json`). It returns the row to serve, its launch configuration, and the row to fall back to.
- The serving entry launches exactly that row, or raises the row's refusal by name.
- Tier words are `fast`, `exact` and `big`; rows are implementations.
- A call class outside the table serves the stock op and is counted: one `[opt_core] CELLS` line at exit lists
  `UNCOVERED_CELL` / `NAMED_FALLBACK` / `CELL_HIT` tokens (`OPT_CORE_CELL_CENSUS_PRINT` controls the printing).

| family | op | rows (implementation, arch) |
|---|---|---|
| `kernels.triattn` | triangle attention, pair bias + key mask, forward | Triton flash kernels (`k2b`, `k2`, `flash`); the CUDA package `triattn_native` (sm_90a wgmma/TMA members by sequence range + an sm_80 mma.sync member, prebuilt torch extensions); `cuda_sm90a`; the exact rows `triattn_exact` (a CUDA member compiled by nvcc at first use, bit-identical to the library op on its proven cells) and `exact_headsplit`, bitwise identical to the op they stand in for |
| `kernels.triattn_xla` | the same op for JAX programs | one XLA FFI launcher over the CUDA package's sm_90a / sm_80 members and an AOT Triton member; forward, with a `jax.custom_vjp` whose backward is the reference statement |
| `kernels.trimul` | triangle multiplication, LayerNorm-in to gated output | `native` / `native_exact` (arch-keyed CUDA cubins, sm_90a and sm_80, launched through the driver), `tx_sm90a` (TMA + wgmma extension), Triton members (`tmk3_*`, `v4`, blocked-row members), engine-form assemblies |
| `kernels.trimul_xla` | triangle multiplication for JAX | the native cubins behind an XLA FFI launcher |
| `kernels.transition` | SwiGLU transition | `flash_sm90a` (prebuilt extension), a CuTe sm_90a cubin, Triton members |
| `kernels.ln` | LayerNorm (+ linear) | `exactln` (cubins, sm_80 / sm_90, bitwise the stock LayerNorm), `fastln` (Triton) |
| `kernels.apb` | attention with a pair bias shared across a batch | CUDA exact members (prebuilt), Triton members |
| `kernels.pallas*`, `fpf_pallas` | Pallas kernels for the JAX engines | flash attention with row-shared bias, triangle multiplication, transition, outer-product mean |

Kernels without a cell table, and the prebuilt binaries:

- Module kernels without a cell table (`flash_triattn`, `lnl_fused`, `ln_proj`, `gather_attn`, `apb_attn`,
  `atom_window`, `templ_embed`, `dtk_kernels`, `ops.msa_fused`, ...) are routed by name: `kernels.route("<name>")`
  returns the one carried copy.
- Prebuilt binaries are keyed by `torch+CUDA+CPython` (extensions) or by GPU architecture (cubins) and checked against
  their recorded digests at load. A missing key is a refusal by name (`no_prebuilt:<pkg>@<key>`); each binary package
  has a `build_prebuilt.py` for adding a key.
- Nothing is compiled at run time except Triton / Pallas kernels and the NVRTC-compiled CUDA-source members of
  `kernels/ln/exactln`, `kernels/trimul/esm_v61` and `kernels/transition/esm`.
- Third-party code keeps its upstream notice beside it.

## FlashPairformer

FlashPairformer is the name of the pairformer kernel family in this tree. Its headline members are the CUDA-native
triangle-attention and triangle-multiplication packages; its portable members are the Triton and Pallas kernels with
the `fpf_` prefix and the XLA bindings. Kits do not import members directly: a kit binds `kernels.triattn`,
`kernels.trimul` and `kernels.transition` through its `cells/` adapters with a tier word (JAX kits bind the `*_xla` /
Pallas members), and the interface picks the member for the call class.

**Triangle attention** (pair bias + key mask, forward):

- `triattn_native`: prebuilt torch extensions with sm_90a members (warp-specialized wgmma + TMA, chosen by sequence
  range) and an sm_80 member (mma.sync + cp.async), plus Gluon / Triton members in the same package for the ranges they
  serve; bf16 / fp16 operands, fp32 softmax and accumulation.
- `cuda_sm90a`: a single sm_90a extension with the same calling convention.
- Portable members: Triton `k2b` (`fpf_triatt_k2b`) and `flash` (`flash_triattn`); the LayerNorm / projection prologue
  and gating epilogue kernels `fpf_triatt_pro`, `fpf_triatt_epi`, `fpf_glue_v2`, `fpf_mkpf`; Pallas `pallas_triatt` and
  `fpf_pallas`; the XLA binding `triattn_xla`.
- Exact tier: the stock op by name; `triattn_exact` (a CUDA member whose output equals the library op bit for bit on the
  (card, torch, library) stacks the cell table vouches for; the library op by name elsewhere); or `exact_headsplit` where
  it is bitwise the op it replaces.

**Triangle multiplication** (LayerNorm-in, projections and gates, contraction, LayerNorm-out, output projection and gate):

- `trimul_native`: CUDA cubins per architecture (sm_90a, sm_80) launched through the driver — fused prologue, cuBLAS
  batched contraction, fused epilogue; row `native` (bf16 planes, fp32 accumulation), `native_exact` (the reference
  library's LayerNorm and summation order, bitwise) and `native:f32in` (fp32-resident input).
- `tx_sm90a` / `tx_sm90a_exact`: a TMA + wgmma extension.
- Portable members: Triton `tmk3_exact` / `tmk3_fast` (`fpf_trimul`), `v4` (`fpf_trimul_v4`), the row-block members used
  by the row-sharded pair stack (`fpf_trimul_rows`), `trimul_esm_shapes`, and the ESM-family members `esm_v61` (CuTe
  sm_90a cubin), `esm_v5_fwd`, `ef2_fused`; Pallas `fpf_pallas` and `pallas` (`cd_trimul`); the XLA binding `trimul_xla`.

**Transition** (SwiGLU with LayerNorm):

- `flash_sm90a` (prebuilt sm_90a extension); Triton `fpf_transition` / `fpf_transition_v2` and the transition variants in
  `fpf_glue_v2`, `fpf_mkpf` and `lnl_fused`; a CuTe sm_90a cubin and Triton members for the ESM-family layout; Pallas
  `transition_pallas` (`fpf_pallas`) and `mlp_transition` (`pallas`).

The `big` memory machinery (`mem`, `mem.rowpair`) is not a member; it calls these kernels. Members derived from
Protenix code keep their notices (`kernels/fpf_flashpairformer.PROTENIX_LICENSE`).

## Layout

| module | provides |
|---|---|
| `autoload`, `modes`, `gates`, `arch`, `report`, `manifest`, `instances`, `counters` | activation, mode tables, GPU / version / pin gates, the ACTIVE / LEVER / EXIT line grammar, `opt_manifest.json`, instance and call ledgers |
| `kernels`, `trimul`, `attn`, `precision`, `capture`, `ops` | the providers above, lever adapters for triangle multiplication and attention, precision policy, CUDA-graph capture, fused MSA-track ops |
| `mem`, `mem.rowpair`, `mem.rowpair_jax` | the `big` mode: chunking, offload, checkpointing, allocator policy, peak accounting, the row-sharded pair stack (`mem/README.md`, `mem/rowpair/API.md`) |
| `of3_sampler`, `of3_trunk` | levers shared by the AF3-architecture engines (diffusion-sampler hoists, template embedder), bound through `configure(...)` |
| `host`, `host_cache`, `seq`, `jax_design`, `diffusion_loop`, `jit_cache` | process memo and worker pools, resident-weights cache, sequence-engine and JAX design-engine helpers, compile-cache placement |
| `compare`, `cell_census`, `det`, `extern`, `cli`, `tools` | output-tree comparison, the cell census, determinism settings, external-tool wrappers, the `python -m opt_core` commands |

- `kit_template/` holds the two files every kit copies byte for byte (`_core_gate.py`, `_build_backend.py`).
- Tests: `python -m pytest common/opt_core/tests` (CPU; `tests/gpu/` needs a GPU).
