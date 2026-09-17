# `opt_core.mem.rowpair_jax` — API (row-sharded pair stack for `--mode big --n_gpu P` on the JAX cofold engines)

Zero-context: a cofold kit's `big` mode folds bigger inputs (memory levers, fast-class numerics). `--n_gpu P` is its RESOURCE AXIS: P=1 = the
kit's single-device memory line; P>1 adds tensor parallelism by ROW-SHARDING the pair representation `[N, N, C]` over P local devices. This
sub-package holds the engine-free primitives; the kit's adapter (`<pkg>/big.py`) rebinds its own stock classes with them.

## Design (≤40 lines)
1. MESH: `mesh.build(P, platform="gpu")` → ONE 1-D `jax.sharding.Mesh` over the FIRST P `gpu` devices `jax.devices()` returns (bounded by the
   kit's CUDA_VISIBLE_DEVICES); P explicit (`None/auto/all` refused); `n_gpu=1` REFUSED (P=1 builds no mesh — structural); `refused: n_gpu=P
   visible=K` if fewer; `refused: n_gpu=P platform=<found> expected=gpu` if jax fell back to CPU; device kind + sm + pool limit + total read from the
   runtime (no card constant). `opt_core.mem.ngpu` (framework-free, shared with the torch family) is the ONE producer of the axis words:
   `refuse_unless_big` (`refused: n_gpu>1 requires --mode big (sharded reductions are not bitwise)`), `refuse_unless_visible`, `active_fields`.
2. PROGRAM: the kit jits its transformed `apply` on the mesh with REPLICATED in/out shardings (`haiku.jit_apply`, `shard.put`); the pair is born
   sharded inside: `shard.constrain(x, rmesh)` = `with_sharding_constraint(NamedSharding(mesh, P('row', None, None)))` at the pair SITES (prologue
   pair terms, recycle carry via `transition.constrain_carry`, trunk→heads boundary).
3. WHY shard_map, NOT constraints alone: the stock row-chunk loops (`hk.scan` over `dynamic_slice` on the row axis) and the fused custom calls
   (tokamax / Pallas) make the SPMD partitioner REPLICATE (measured: OOM at 74 GiB on 2 cards). So every per-layer
   block runs in ONE `haiku.region(...)` = `shard_map` with `in_specs=(rows, …replicated)`; inside, code sees `[N/P, N, C]` and inserts collectives.
4. PER-OP PLAN (`evidence.SITES` is the table): pair transition, LN/projections/gates, prologue pair terms → row-local (no collective);
   outer-product mean → output rows local, left operand `transition.opm_operands` (dynamic_slice of the replicated MSA projection), MSA replicated;
   triangle multiplication → `trimul.contract(equation, a, b, axis)`: outgoing = all_gather(partner) ; incoming = all_gather(a)+all_to_all(b) so the
   k-contraction is COMPLETE per device (no partial sums; a reduce-scatter form would sum P partials = class `reordered` — not used); schedule
   `ring` = P ppermute steps, transient one block, same class; triangle attention starting node → row-local given `triatt.bias_full` (all_gather of
   the `[N/P,N,H]` bias); ending node → `triatt.enter_transposed` (all_to_all: my column block, swapped = my rows of xᵀ) → the same row attention →
   `exit_transposed`; MSA row-attention / single-attention pair bias → `transition.pair_logits_full` (all_gather); template pair stack = the same
   sub-layers; heads: distogram / distance-error `transition.symmetrize` (all_to_all), masked means `transition.masked_mean` (2 psum), `[N,N]` outputs
   `transition.rows_full` at the writer boundary only; structure module / diffusion attention read pair row i for query i → row-local with the
   attention-output rows all_gathered per block.
5. NUMERICS (per element vs dense): collectives/constraints `moves_bytes` (exact); row-local `row_local` (bit-exact expected; a kernel may tile by
   row count → tested per stack); trimul `complete_contraction` (dense-length sum, GEMM shape M=N/P → library accumulation order may differ →
   STATED tolerance; bit-exact not claimed); psum means `reordered`. ⇒ P>1 is tier-2 by construction; a kit's P>1 result is a BAND vs P=1.
6. PER-SHARD KERNELS: a fused kernel called inside the region sees local shapes; `triatt.kernel_gate(n_loc, n_cols, constraints)` → served or
   `kernel=jnp kernel_reason=<constraint>` on the lever line (a NAMED state under P>1, never silent).
7. HAIKU RECIPE (all four JAX cofold engines are dm-haiku): `haiku.rebind(patches, StockCls, "__call__", body)` re-wraps with Haiku's
   `wrap_method` (name scope kept) and returns the stock body; bodies dispatch on `haiku.in_region()`; the region body runs under
   `haiku.body_frame(rng=None)` (hk.scan inside would leak the rng tracer); methods with their own sub-modules are rebound by name (`~<method>` scope).
8. PROCESS MODEL: single process, P local devices (jax) — the kit launches no workers; users never type a launcher.
9. EVIDENCE: `ngpu.active_fields(P)` (re-exposed as `evidence.active_text` / pairs `evidence.active_fields`) is the ONE producer of
   `n_gpu=P sharding=rowpair` (`n_gpu=1 sharding=none`); `evidence.line(tag, state,
   P, rmesh=, environ=, peaks=)` = `[tag] LEVER name=rowpair state=on impl=opt_core.mem.rowpair_jax@<v> origin=core n_gpu=P sharding=rowpair axis=row
   visible=K platform=gpu devices=d0,d1 device_kind=… sm=… xla_pool_limit=… device_total_bytes=… xla_pool_fraction=… shard_map=… prealloc=… mem_fraction=…
   client_mem_fraction=… allocator=… xla_peak_bytes=d0:…,d1:… xla_peak_gb_max=… peak_scope=xla_allocator schedule=… kernel=… kernel_reason=… sites=…`;
   `state=on` REQUIRES schedule/kernel/kernel_reason/sites (fail-closed). `xla_peak_*` = the XLA allocator's in-use high-water (NOT NCCL buffers /
   CUDA context / non-XLA memory — a ladder's peak-per-card is the device high-water sampled outside the process). The kit touches the XLA memory
   environment (PREALLOCATE / MEM_FRACTION) ONLY under n_gpu>1 — at P=1 it is the tested kit's, byte for byte — and the line records it as found;
   `mesh.mem_fraction_gate(rmesh, {8: <kit's ceiling>})` judges the LIVE pool (`bytes_limit / device total`) and refuses by name when NCCL headroom
   is missing (communicators allocate outside XLA's pool). Adapters put `n_gpu` in their `opt_core.capture.xla_cache` config mapping so per-P
   programs key separately.
10. P=1: `mesh.build(1)` is refused by name — the adapter installs nothing (`state=off reason=n_gpu=1`) → bit-exact to the pre-TP kit by construction
   (no one-code-path option). SIZES: every N has a plan — `shard.pad_plan(N, P)` pads N (pair map AND masks, `shard.pad_pair`) to a multiple of P;
   a ladder bin is never refused; `shard.local_extent` guards the padded N. A hung collective is bounded by the kit's process watchdog
   (`opt_core.process`), not by this package.

## Modules
| module | public surface |
|---|---|
| `mesh` | `build`, `RowMesh(.mesh .axis .n_gpu .devices .visible .facts .describe())`, `explicit_n_gpu`, `device_facts`, `refuse_mode`, `xla_memory_env`, `effective_mem_fraction`, `mem_fraction_gate` |
| `shard` | `rows_spec`, `cols_spec`, `replicated_spec`, `named`, `constrain`, `replicate`, `put`, `shard_map`, `shard_map_flavour`, `next_multiple`, `local_extent`, `axis_index`, `axis_size`, `gather`, `all_to_all`, `a2a_plan`, `a2a_max_elems`, `a2a_fields`, `a2a_record`, `a2a_reset`, `A2A_MAX_ELEMS`, `A2A_MAX_ELEMS_ENV`, `rows_to_cols`, `cols_to_rows`, `transpose_block`, `local_block`, `psum`, `ppermute_shift` |
| `trimul` | `contract(equation, a, b, axis, schedule=DEFAULT, precision=None)` (`DEFAULT = "ring"`: b row-blocks travel once around the ring; `"gather"` all-gathers b — the named alternative), `classify`, `EQUATIONS`, `SCHEDULES`, `DEFAULT`, `transient_bytes` |
| `triatt` | `bias_full`, `enter_transposed`, `exit_transposed`, `mask_rows`, `mask_cols`, `kernel_gate`, `kernel_fields` |
| `transition` | `opm_operands`, `pair_logits_full`, `symmetrize`, `rows_full`, `masked_mean`, `constrain_carry`, `ROW_LOCAL_SITES` |
| `evidence` | `active_fields`, `active_text`, `line`, `device_peaks`, `peak_fields`, `impl_label`, `SITES`, `CLASSES`, `TP_LEVER_ID`, `card_fields(rmesh)` (reads `opt_core.arch.lever_state`; the shared id is declared once by the core, never here)|
| `alphafold` | `library(lever, multimer=True) -> (modules, modules_multimer|None)`, `model_library`, `install(rmesh, modules, multimer=None, *, heads='sharded', conf='sharded', sites=('trunk','model','heads'), schedule=trimul.DEFAULT, lever=, pin=, patches=) -> PatchSet`, `apply_shardings(rmesh, in_tree_shapes, out_tree_shapes)`, `jit_sharded(fn, rmesh, n_args=3)`, `wrap_runner(runner, rmesh, attr='apply', n_args=3)`, `heads_rows`, `heads_rows_conf`, `in_sharded_region`, `in_manual_region`, `uninstall`, `installed`, `describe`, `pad_multiple`, `file_sha`, `check_pin`, `PIN_FILES` / `PIN`, `SITE_GROUPS`, `TRIMUL_DEFAULT` — the AlphaFold-2 Haiku recipe (both AF2 kits' pinned trees): rebinds the pair sub-layers inside row-sharded regions (`trunk`), keeps the pair prologue / recycled pair / template pair / trunk→heads boundary in the row layout (`model`), and runs the heads in one manual region on local pair rows (`heads`: structure-module IPA row-local under `heads='sharded'`, distogram / aligned-error / in-graph confidence on rows under `conf='sharded'`, logits leaving row-sharded); `apply_shardings` / `wrap_runner` place the recycled pair in and the pair-shaped outputs out row-sharded; `describe()` reports the EFFECTIVE words (heads, conf, structure, template, recycle, msa=replicated, masks=replicated, outputs_rows); pinned to the transcribed trees (refused by name on other bytes) |
| `alphafold_heads` | `install_heads(patches, modules, *, lever, rows_active, axis, n_gpu) -> [sites]`, `confidence_module`, `file_sha`, `rows_gate`, `describe_heads`, `HEADS_PIN` (`PIN_MODULES` + `PIN_CONFIDENCE`), `CONFIDENCE_IN_GRAPH`, `HEADS_BODIES`, `HEADS_HAZARDS`, `OUTPUTS_ROWS`, `SITES` — the AF2 pair-reading heads on row blocks: DistogramHead (half-logits Linear on rows + `transition.symmetrize`; logits leave as the row block), PredictedAlignedErrorHead (stock body, row-extent gate), and for a tree that computes confidence IN the program `confidence.compute_predicted_aligned_error` (stock, gated) / `confidence.predicted_tm_score` (per-row sums on rows, `[N]` gather); a tree that computes confidence on the host has nothing rebound there; one producer for both AF2 kits; imports nothing from `alphafold` |
| `alphafold_template` | `install_template(patches, modules, multimer=None, *, lever, rmesh, axis, n_gpu, active, in_region, region=None, pin=None) -> [site names]`, `dgram_rows(positions_rows, positions_all, num_bins, min_bin, max_bin)`, `describe`, `installed`, `forget`, `template_rows`, `tree_facts`, `check_pin`, `TEMPL_PIN`, `TEMPL_BODIES`, `TEMPL_HAZARDS`, `TEMPLATE_KEY_FIELD` — the AF2 template embedding as ONE rows region (monomer `TemplateEmbedding` + `SingleTemplateEmbedding`, `modules_multimer` `TemplateEmbedding` + `SingleTemplateEmbedding`): per-template `[N/P, N, 88]` feature rows (`dgram_rows`, mask products, aatype tiles, unit vectors from the local rows' frames), the nested (already rebound) template pair stack on local rows, the template-pointwise attention on this device's `N/P·N` points; rows out, no gather; installed by `alphafold.install` under `sites∋'model'` (absent module → `template=replicated`, named) |
| `alphafold3` | `install(rmesh, patches, heads=, conf=, b21=, schedule=, sites=, mark=, pin=) -> record`, `install_fields`, `check_pin`, `library_shas`, `PIN`, `BODIES`, `HAZARDS`, `SITES_*` — the AF3 Haiku recipe (trunk + model + heads bodies on row blocks; one producer; the diffusion pair-conditioning transcription `b21` is passed in by the kit) |
| `rowchunk` | `chunked_class(base, rows, ledger=None, shapes=None)`, `check`, `form_of`, `stock_modules`, `transient_bytes`, `LEVER="trimul_chunk"`, `OUTGOING`, `INCOMING` — the AF2-family single-device row-chunked TriangleMultiplication (fused + unfused forms; P=1 lever of the AF2 kits) |
| `haiku` | `PatchSet` (re-export of the family's install record), `wrap_method`, `rebind`, `stock_body`, `body_frame`, `in_region`, `region`, `jit_apply` |

Every module is standard library at import (jax / numpy / haiku inside functions via `_lazy`; a missing one → `MemLeverRefused('rowpair', …)`).
jax ≥ 0.4.30-class `shard_map` (`jax.shard_map` or `jax.experimental.shard_map`). Target stacks: jax 0.5.3 + dm-haiku 0.0.16 (the AF2 kits'
stacks) and jax 0.10.2 + dm-haiku 0.0.16 (the AF3 JAX kit's stack); a stack is called tested by its unit-suite record per
stack and machine, not here.

## ADAPTER GUIDE (kit side; the kit owns this code — the recipes hold only the sharded math)
Common frame for every JAX kit (`<pkg>.big` / its rowpair adapter). P is EXPLICIT (absent `--n_gpu` == 1); nothing below runs at P=1.
```python
from opt_core.mem import MemLeverRefused, ngpu
from opt_core.mem.patchset import PatchSet
from opt_core.mem.rowpair_jax import mesh as rp_mesh, shard as rp_shard, evidence as rp_ev, haiku as rp_hk

P = ngpu.refuse_unless_big(args.n_gpu, mode)          # '--n_gpu>1' under exact/fast: refused by name (rc != 0); absent flag == 1
if P == 1:
    emit(rp_ev.line(TAG, "off", 1, reason="n_gpu=1"))   # nothing installed, XLA env untouched: the kit's single-device program (bitwise vs pre-TP by construction)
else:
    # launcher, BEFORE jax initialises CUDA: set the kit's XLA memory policy for P>1 with ONE variable name (on jaxlib 0.10-class stacks that bake
    # XLA_CLIENT_MEM_FRACTION, also setting XLA_PYTHON_CLIENT_MEM_FRACTION makes the CUDA plugin init raise and jax fall back to CPU — mesh.build refuses that by name)
    RM = rp_mesh.build(P)                                 # platform='gpu': 'refused: n_gpu=P visible=K' / 'refused: n_gpu=P platform=cpu expected=gpu'
    rp_mesh.mem_fraction_gate(RM, KIT_MEM_FRACTION_CEILINGS) # the LIVE pool fraction vs the kit's per-P ceiling (NCCL headroom), e.g. {8: 0.85}
    plan = rp_shard.pad_plan(num_tokens, P)               # pad tokens+masks by plan["pad"] or pick a bucket multiple of P — every ladder bin has a plan, never refuse a bin
    PATCHES = PatchSet("rowpair")
    ... install ONE recipe (below) ...
    params, batch = rp_shard.put(params, RM), rp_shard.put(batch, RM)     # replicated placement over the mesh devices
    out = apply(params, key, batch)                       # the pair is sharded INSIDE by the recipe's regions
    emit(rp_ev.line(TAG, "on", P, rmesh=RM, peaks=rp_ev.device_peaks(RM), schedule="gather", kernel="jnp", kernel_reason=<why>,   # state=on is fail-closed:
                    sites=<record sites>, **rp_ev.card_fields(RM), **<recipe fields>))                                              # rmesh/schedule/kernel/sites required
# ACTIVE / EXIT lines: … + report.kv(*ngpu.active_fields(P))   → 'n_gpu=P sharding=rowpair' (P=1: 'n_gpu=1 sharding=none') — never retyped
```
LAUNCHER RULE (jaxlib 0.10-class stacks): set ONLY `XLA_CLIENT_MEM_FRACTION` for the P>1 memory policy — the stack bakes that name, and ALSO setting
`XLA_PYTHON_CLIENT_MEM_FRACTION` makes the CUDA plugin initialisation raise and jax fall back to CPU silently; `mesh.build(platform='gpu')` then refuses by name
(`refused: n_gpu=P platform=cpu expected=gpu`). On jax 0.5-class stacks `XLA_PYTHON_CLIENT_MEM_FRACTION` is the baked name.

`xla_peak_bytes` on the line is the XLA allocator's in-pool peak per device (`peak_scope=xla_allocator`); a memory-ladder row's peak/card is the device
high-water sampled outside the process (NCCL, the CUDA context and non-XLA allocations are outside the pool).

### AlphaFold 3 (the `af3_jax` kit — `alphafold3.install`)
```python
from opt_core.mem.rowpair_jax import alphafold3 as rp_af3
import af3kit_big_levers as b21                          # the KIT's diffusion pair-conditioning transcription (pair_conditioning_rows, conditioning_factory, REL_FIELDS)
rec = rp_af3.install(RM, PATCHES, heads="sharded", conf="sharded", b21=b21,          # schedule=trimul.DEFAULT ("ring"); "gather" is the named alternative
                     sites=("trunk", "model", "heads"), mark=phase_marker, pin=None)   # pin=None: the built-in PIN (11 stock module digests); other bytes refused by name
# the kit's ModelRunner: hk.transform(forward) as stock; bind the jitted apply to the mesh
runner._model = rp_hk.jit_apply(runner._model_apply, RM, n_args=3)       # or jax.jit(apply, in_shardings=replicated, out_shardings=replicated) in the kit's own words
result = runner._model(rp_shard.put(params, RM), key, rp_shard.put(featurised_example, RM))
emit(rp_ev.line(TAG, "on", P, rmesh=RM, peaks=rp_ev.device_peaks(RM), schedule=rec["schedule"], kernel="jnp", kernel_reason="xla_path_under_rowpair",
                sites=tuple(rec["sites"]), **rp_af3.install_fields(rec), **rp_ev.card_fields(RM)))   # heads=/conf=/diffusion= NAME what is sharded (per-card relief needs the heads)
```
A trunk-only install (`sites=("trunk","model")`) prints `heads=not_installed`: its per-card claim is the trunk's.
Fused pair kernels (the kit's Pallas levers) are single-device: under P>1 the recipe runs the XLA path and the line says `kernel=jnp kernel_reason=…` —
a NAMED lever state, never a silent switch. `PATCHES.restore()` returns every stock attribute.

### AlphaFold 2 (both AF2 kits — `alphafold.install`)
```python
modules, modules_multimer_or_None = rp_af2.library("rowpair", multimer=IS_MULTIMER_TREE)   # refuses by name if the AF2 library is absent
from opt_core.mem.rowpair_jax import alphafold as rp_af2
PATCHES = rp_af2.install(RM, modules, modules_multimer_or_None, schedule="gather", lever="rowpair", patches=PATCHES)   # pin: colabfold 2.3.13 / dl_binder_design cafa3853 trees
N = num_res                                               # any N: the regions zero-pad the residue axes to a multiple of P per call and slice back (describe()['padded_calls'])
# placement — EXACTLY ONE owner per kit:
#   (a) a kit whose runner is alphafold.model.model.RunModel:   rp_af2.wrap_runner(run_model, RM, n_args=3)   → run_model.apply bound to the mesh (haiku.jit_apply);
#       n_args = the number of POSITIONAL arguments of runner.apply: 3 for apply(params, rng, batch); 4 for an initial-guess runner apply(params, rng, batch, initial_guess)
#   (b) a kit with its own placement lever (a device-resident loop that already owns the jit): that lever calls rp_hk.jit_apply(apply, RM, n_args=…)
#       itself when big built a mesh — never both (one placement producer).
out = run_model.apply(rp_shard.put(params, RM), key, rp_shard.put(batch, RM))
emit(rp_ev.line(TAG, "on", P, rmesh=RM, peaks=rp_ev.device_peaks(RM), **rp_af2.describe(), **rp_ev.card_fields(RM)))   # describe() IS the ON vocabulary
rp_af2.uninstall()                                          # at exit / between installs
```
The AF2 kits' single-device `trimul_chunk` lever is `rowchunk.chunked_class` (P=1 memory line; inside a P>1 region the same class chunks the local rows).

### What is sharded and what stays replicated (the per-card floor is a STATED property of each recipe)
| recipe | sharded ÷P (rows of the pair) | replicated on every card (the floor) |
|---|---|---|
| `alphafold3` sites=trunk+model+heads, b21 given, conf=sharded | every PairFormerIteration / EvoformerIteration body (pair stack AND the MSA-module's pair blocks: OPM output, MSA pair-bias read, trimul ×2, grid attention ×2, transition), the template embedder's pairformer blocks, the recycle carry (prev pair), distogram + confidence heads' pair, the diffusion pair conditioning | single `[N,384]`, MSA `[S,N,64]`, atom cross-attention (queries/keys per token window), per-call transients of ONE full plane at each gather site (trimul partner plane `[C,N,N]` bf16, grid-attention bias `[h,N,N]`), params |
| `alphafold3` without b21 (or conf=replicated) | as above minus the diffusion conditioning (minus the confidence head) | + the diffusion conditioning's full pair `[N,N,c_z]` + its f32 LayerNorm/concat transients (+ the confidence pairformer's full pair): the single-card peak of this model lives in these heads, so a trunk-only shard buys wall-clock, not per-card memory — the line's `heads=`/`conf=`/`diffusion=` say which case ran |
| `alphafold` (AF2 monomer/multimer) sites=trunk+model+heads, heads=sharded, conf=sharded | pair `[N,N,c_z]` from the prologue statement that creates it (relative-position / recycled-pair / recycled-position terms, template embedding) through the extra-MSA / Evoformer / template stacks, the recycle carry (in-graph loop or jit boundary) and the heads (distogram, aligned error, structure-module IPA) to the outputs, which leave row-sharded (`[N,N,64]` distogram / aligned-error logits and probabilities, `[N,N]` expected error, the returned pair) | MSA `[S,N,c_m]`, single, per-residue tensors, `[N,N]` no-channel masks, params; transients: `[N/P,N,H]`-class biases gathered per layer (triangle attention starting node, MSA row attention pair bias); no `[N,N,c_t]` template plane under template=rows |
| `alphafold` sites=trunk only (or a module absent from the tree) | the region bodies only | + every consumer of that group, each named in `describe()` (`prologue=replicated`, `template=replicated`, `heads=replicated`, …) — never silent |
The kit's applicability table quotes the MEASURED floor at ≥2 bins (a one-card size and its top P=1 bin) from its own dev check; the words above say which
blocks the number is made of.

### Refusal words when a dependency is absent (entry points raise `opt_core.mem.MemLeverRefused(lever, reason)`, never ImportError)
| entry point | absent | `reason` |
|---|---|---|
| `mesh.build`, `shard.*`, `trimul.contract`, `haiku.*` (any primitive) | jax | `jax import failed (<repr>)` |
| same | numpy | `numpy import failed (<repr>)` |
| `haiku.*`, the recipes | dm-haiku | `haiku import failed (<repr>)` |
| any `shard_map` user | jax older than 0.4.30 | `shard_map unavailable in jax <v> (<repr>); jax >= 0.4.30 is required` |
| `alphafold3.library` / `alphafold3.install` | alphafold3 | `alphafold3.model.network.modules not importable (<msg>)` (the first missing module of `alphafold3.LIBRARY` is named) |
| `alphafold.library` | alphafold (the AF2 library) | `alphafold.model.modules not importable (<msg>)` |
| `alphafold.install` on other bytes | — | `install: <module> sha256 <12 hex> is not a transcribed tree (<pinned labels>)` |
| `alphafold_heads.install_heads` on other bytes | — | `install_heads: <module> sha256 <12 hex> is not a transcribed tree (<pinned labels>)` |
| `alphafold_template.install_template` on other bytes | — | `install_template: <module> sha256 <12 hex> is not a transcribed tree (<pinned labels>)` |
| `alphafold3.install` on other bytes | — | `alphafold3 recipe: the model library is not the transcribed tree (<tree>): <module>:<have>!=<want>,…` |
| `rowchunk.check` / `rowchunk.chunked_class` | alphafold | `alphafold.model.modules not importable (<msg>)` |
| `mesh.build(P)` | fewer devices / wrong platform / P=1 | `refused: n_gpu=P visible=K` · `refused: n_gpu=P platform=<found> expected=<platform>` · `refused: n_gpu=1 installs nothing — …` |
| `ngpu.refuse_unless_big` | `--n_gpu>1` outside big | `refused: n_gpu>1 requires --mode big (sharded reductions are not bitwise)` |
The kit maps a refusal to its usage/refusal exit code and prints the reason; nothing degrades silently.

### Dev-check protocol per change-set (what "green" means for an adapter)
(i) P=1 bit-exact vs the pre-TP kit on 2–3 small equality cases (nothing installed — any diff is a bug); (ii) this package's unit suites in the kit's
environment; (iii) ONE mid-size end-to-end P=2 vs P=1 (~1–2k tokens) inside the engine's fast band;
peak/card at 2–3 mid bins of the engine's memory ladder — pair terms ~1/P for what the line says is sharded.

## Tests
`tests/test_rowpair_jax_logic.py` (skips BY NAME without jax; runs on a CPU machine with `XLA_FLAGS=--xla_force_host_platform_device_count=4` and on a
GPU machine): refusal words, specs, every collective's round trip, `trimul.contract` == dense einsum for 4 equations × 2 schedules × P∈{1,2,4}
(class complete_contraction: f32 at HIGHEST matmul precision rel ≤1e-5, f32 at the backend default precision — tf32-eligible on NVIDIA GPUs — rel ≤2^-10,
bf16 rel ≤2^-7 of max|ref|; measured max-abs-diff, bit-exact-or-not, dtype and precision RECORDED per case), every
moves_bytes primitive BIT-EXACT, `triatt` transposed round trip / bias gather / mask blocks bit-exact, `transition` helpers, evidence grammar (state=on
fail-closed), haiku rebinding + rng-less frame + a two-sub-layer toy model sharded vs dense (tolerance class), init through regions keeps the stock
parameter tree. `tests/test_rowpair_jax_rowchunk.py`: the AF2-family row-chunked TriangleMultiplication == stock (fused + unfused; rows>=N is the
stock body exactly; equation refusal at trace). `tests/test_rowpair_jax_alphafold.py`: the AF2 recipe (install/refusals/restore; sub-layer bodies on row blocks == stock).

## Strategy id
`F7.tensor_parallel` — one canonical id for both frameworks (numerics_changing=true); framework sub-levers are `F7.tp_<name>` (jax: `F7.tp_trimul_ring`
for the ring schedule, `F7.tp_triatt_bias_gather`, `F7.tp_a2a_transpose`; the AF2 single-device row-chunk is the kits' existing `trimul_chunk` lever).
Adapters pass `strategy="F7.tensor_parallel"`.

## The all_to_all element limit (`shard.all_to_all`)

XLA's tiled `lax.all_to_all` on jax 0.5-class GPU runtimes (the AF2-recipe JAX kit's jax 0.5.3 stack) returns `INVALID_ARGUMENT` ('All buffers must have the same
element type and count') once the LOCAL operand holds `>= 2**31` elements — `[2048, 8192, 128]` = 2**31 fails, `[2530, 5060, 128]` passes;
`ppermute` / `all_gather` are unaffected; jax 0.10 stacks (the AF3 recipe's kit) are unaffected. The pair block `[N/P, N, 128]` of the ending-node transposes
(`triatt.enter_transposed` / `exit_transposed`), the incoming multiplication's re-layout (`trimul.contract`) and the half-logit symmetrisation
(`transition.symmetrize`, `[N/P, N, 64]`) reaches it at `N >= 4096·sqrt(P)` (×2 5 793, ×4 8 192, ×8 11 586 tokens). `shard.all_to_all` — the one
binding `rows_to_cols` / `cols_to_rows` / `transpose_block` issue — is the single `lax.all_to_all` call below `shard.A2A_MAX_ELEMS` (2**31; env
`ROWPAIR_JAX_A2A_MAX_ELEMS` lowers it for tests) and, at or above it, `shard.a2a_plan`: consecutive near-equal slices along the FREE dim (the
channel dim — neither split nor concatenated, so the exchange is independent per channel), one all_to_all per slice, `concatenate` along that
dim. The result is the one call's, element for element (class `moves_bytes`); the transient is one extra block while the pieces await the
concatenate. The family line carries `a2a_split=<k>` (`evidence.line`, state=on) when a split was traced in the process; below the limit the
line and the program are unchanged. No kit binds anything: the recipes reach the collectives only through `shard`.

