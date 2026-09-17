# opt_core.mem — the `big` memory mode

`big` is the fourth release mode of a kit, beside `off` / `exact` / `fast`: the mode that folds bigger inputs on the same card.
It composes on the kit's `fast` mode (`exact` where the kit has no fast) and adds **memory levers** — mechanisms that lower the peak
device memory of a pass: host offload of pair tensors, chunked or streamed operations, checkpointing, allocator policy, JAX-side
sub-batching, memory-lean kernels, engine settings. Its guarantee is *runs bigger, within band*: the numerics tier is measured per
lever set with the kit's equality suite, and the record says which levers are exact.

This package is the shared mechanism. It names no engine: a kit's adapter (`<engine>/opt/<pkg>/big.py`) composes the mode into its
table, names its hook points per lever, applies the line, prints the ACTIVE line, writes the manifest block and runs the exit gate.
Everything below is the contract every adapter codes against; each module's docstring is the reference
(`python -c "import opt_core.mem.<module> as m; help(m)"`).

## Layout

```
opt_core/mem/
  __init__.py     apply(levers, ctx) -> AppliedRecord · refuse · census · undo · Refused; re-exports of the API below
  registry.py     Lever · LEVERS · @register · discover; Ctx (hooks, settings); selection (switches, allow_partial); Refusal / refuse / RefusalError; selection
  record.py       AppliedRecord: applied / refused / checked, the mode line, the ACTIVE line, the kit.big manifest block, the census, the exit gate
  compose.py      compose_big(table, base, levers, drop, lines, auto) on opt_core.modes.ModeTable; BigLine; select_line (the
                  kit's line selector and the auto policy record); base_for (fast-else-exact)
  allocator.py    levers expandable_segments (confirmed by read-back) · cache_release; reset_peak / counters; the XLA variables
                  recorded, jax_export (refuses on a pin or an initialised backend)
  offload.py      host offload of named tensors / modules with streamed H2D / D2H (resident and streamed lines under one flag)
  chunk.py        chunked / streamed pair transition, triangle attention, triangle multiplication, confidence head, MSA-row chunking
  ckpt.py         block / cycle checkpointing, diffusion-sample chunking (samples per pass), the seed batching cap
  sample_loop.py  lever sample_loop: the diffusion roll-out + confidence head on `chunk` of the N samples per pass, per-sample outputs
                  assembled on the host in stock order (composes ckpt's stock-order draw discipline; refusals chunk_gt_samples /
                  needs_cross_sample / identical_chunks)
  jax_mem.py      the JAX-side levers
  peak.py         the peak-memory INSTRUMENT (not a lever, not in LEVER_MODULES): one source file, copied into a hook directory at
                  build and imported from there by the model process — never as opt_core.mem.peak; read-only (it samples and writes
                  peak_mem.json beside each pass; it resets nothing)

  primitives.py   the engine-free PRIMITIVES an adapter composes into its own modules (re-exported by opt_core.mem): MemLeverRefused,
                  row_blocks, Ledger, evidence_line, env_int / env_flag, parse_levers
  torch_rowchunk.py  row-chunked evaluation of row-independent pair sub-modules into ONE preallocated output (rowchunk_apply /
                  rowchunk_tensor / rowchunk_concat), the pair-shape guard, FreeList (dead-tensor release), torch_module (lazy torch)
  torch_hostpair.py  PinPool — THE pinned host-buffer pool of the package (budget, host-RAM check, page-lock or refuse by name, or a
                  COUNTED pageable buffer for a pool built pageable=True; offload.PinPool is its `host_park` binding); HostPair (a pinned
                  mirror of a pair tensor served in row / column blocks); ln_rowsplit / LayerNormGuard (layer_norm inputs >= 2**31 elements)
  torch_alloc.py  the PYTORCH_CUDA_ALLOC_CONF writer of a process (export-at-activation, effective-state read-back, the child env row);
                  allocator.py's levers write through it
  graph_gate.py   the token-gated capture decision of a CUDA-graph / static-arena lever (parse_cap, decide); ckpt.py's graph policy decides through it
  budget.py       memory-budget arithmetic: rows_within — THE budget→rows function (refuses below a floor) — nbytes, device_free_bytes,
                  budget_bytes; the device total / free themselves are read through opt_core.arch.device_memory (THE device-memory reader)
  patchset.py     PatchSet: attribute rebinds applied and undone as one unit (framework-free)
  ngpu.py         the n_gpu axis of big: `--n_gpu P` is a RESOURCE axis (P=1 = the single-card levers; P>1 = row-sharded pair-stack
                  tensor parallelism); refusals by name (fewer than P devices visible; P>1 under exact / fast, whose bitwise / band
                  guarantees a re-ordered sharded reduction cannot keep); the ACTIVE / EXIT fields n_gpu=P sharding=rowpair
  rowpair/        the row-sharded pair-stack tensor parallelism P>1 runs on (torch): the pair tensor partitioned by rows across P
                  cards, the kit launching its own workers (users never type torchrun); numerics tier-2 by construction (sharded
                  reductions re-order sums) — its README is the reference for the sharded primitives and their dense-equivalence tests
  _selftest.py    `python -m opt_core.mem._selftest`: every primitive on CPU tensors in seconds (torch optional: the pure-python checks
                  run without it); exit 0 = ok
```

Two layers, one package. The LEVERS (registry / record / compose and the lever modules) are what `--mode big` applies by name;
the PRIMITIVES are the mechanisms below them, importable one at a time by an adapter that installs its own memory line
(`from opt_core.mem import torch_rowchunk as RC`). One producer per fact across both layers: pinned host memory is allocated by
`torch_hostpair.PinPool` only, a row count is sized from bytes by `budget.rows_within` only, the device's bytes are read by
`opt_core.arch.device_memory` only, the allocator variable is written by `torch_alloc` only, the XLA client variables by
`allocator.jax_export` only. A primitive either acts or raises `MemLeverRefused(lever, reason)` (never a silent stock path); its facts
reach the run record through the adapter's ONE activation-evidence line per lever per process (`evidence_line`, the LEVER grammar of
`opt_core.report.lever_line`).

The lever modules register their levers at import; `apply()` imports them (`registry.LEVER_MODULES`) and records which the install
holds, which are absent, and which break without their framework. The LAZY-IMPORT rule: a lever module imports only the standard
library at module level — torch / jax are imported inside the functions that need them — so `discover()` runs on any machine
(`tests/test_mem_assembly.py` enforces it). Tests live in `common/opt_core/tests/test_mem_*.py` (CPU; synthetic modules; torch-free
except where a test names torch and skips without it).

## How an adapter registers `big`

```python
from opt_core import modes, report, manifest
from opt_core import mem

# 1. the mode: big = fast + [levers] - [base levers big leaves out]
TABLE = modes.ModeTable(("off", "exact", "fast"), "fast")                            # the kit's own table
TABLE, LINE = mem.compose_big(TABLE, levers=("expandable_segments", "pair_offload", "chunk_pair_transition"),
                                drop=("graph_sampler",))                             # LINE.base == "fast" (the rule), LINE.levers, LINE.drop

# 2. the hook points: what each lever's docstring names (module references, tensor / attribute names, chunk dimensions)
ctx = mem.Ctx(prefix="ACME", tag="acme-opt", framework="torch",
              hooks={"pair_offload": {"module": model.trunk, "tensor": "z"},
                     "chunk_pair_transition": {"module": model.trunk.transition, "dim": 1}},
              settings={"chunk_pair_transition": {"chunk": 256}},                    # the kit's values for the levers' declared settings (what it read from its own flags)
              graphs=False)                                                          # no CUDA-graph capture in the composed line

# 3. apply: strict — a refused lever raises mem.Refused carrying the record (the kit prints its NOT ACTIVE line and exits 3)
try:
    record = mem.apply(LINE, ctx, line=args.big_line,         # the kit's parsed line selector (None = default/auto), lever switches, opt-out
                       switches=switches, allow_partial=args.allow_partial)
except mem.Refused as e:
    print(e.record.not_active_line("acme-opt"), file=sys.stderr); raise ActivationError(...)
report.log_once(rep, record.active_line("acme-opt"))     # [acme-opt] ACTIVE mode=big:expandable_segments,pair_offload,… base=fast exact=band refused=none off=none on=none allocator=…
record.attach(kit_fields)                                # opt_manifest.json  kit.big = record.manifest_block()

# 4. the census: the kit delimits units; the levers mark themselves from their levered paths
record.unit_begin(item_id); ...; record.unit_end()
verdict = record.exit_gate(rc)                           # exit 3 when a line lever refused or a unit fell short, unless allow_partial (the kit's --allow-partial)
```

`<PREFIX>_OPT_MODE=big` (the kit's own mode variable) selects the mode through the kit's autoload / mode plumbing — which restates
the modes: the kit's `_autoload.py` (`opt_core.autoload.AutoloadSpec.modes`) must list `big` beside its other modes, or the trigger
refuses it as an unknown mode; the adapter's `enable("big")` then runs the steps above. Nothing in the core is imported by a
process whose variable is unset.

## The kit's inputs

| input | meaning |
|---|---|
| `<PREFIX>_OPT_MODE=big` | the kit's mode variable (its own name; `big` is the one spelling in the table, the ACTIVE line, the manifest, the run script) |
| `apply(…, line=<name>)` \| `line="auto"` | the line selector: a line the kit declared (`compose_big(lines={name: {"levers", "tier", "cost_note"}})`; `default` = the `levers` argument with `tier=` / `cost_note=`) or the kit's `auto` policy — every declared line's `{levers, tier, cost_note}` (the declared tier; the speed cost as a ledger row id or `unmeasured`), the chosen line, the reason and the policy's details are recorded (`line_policy`) and the ACTIVE line says `line=<name>[(auto: <reason>)]`; unset = `auto` when the kit declared `auto_default`, else `default` |
| `apply(…, switches={<lever>: off})` | a lever of the line is left out — recorded as `off_by_flag` |
| `apply(…, switches={<lever>: on})` | a registered lever outside the line is added — recorded as `on_by_flag` |
| `Ctx(settings={<lever>: {<setting>: <value>}})` | a lever's declared setting (`Lever.settings`; read through `ctx.setting`, the source recorded; a string value is cast by the lever) |
| `apply(…, allow_partial=True)` \| `exit_gate(rc, allow_partial=True)` | the census opt-out (the kit's parsed `--allow-partial`): a partial run exits 0 with the opt-out recorded in the block |

`<PREFIX>` is the kit's variable stem (`[A-Z][A-Z0-9_]*`, recorded). The kit parses its OWN flags and variables (its `--allow-partial`,
its documented size gates, any lever switch it documents) and passes the values; the core reads no environment variable for the line,
the switches, the settings or the opt-out. A switch value is on (`True`/`1`) or off (`False`/`0`); a switch naming no registered lever and
no line lever, any other value, and an on for a lever the registry does not hold is a refusal by name. A line lever whose module the
install lacks is a refusal by name (never dropped).

## Levers, families, exactness

A lever (`registry.Lever`) is one mechanism with a name, a family, an exactness label with its reason, the preconditions it checks by
name, the settings it reads, and two callables: `applies(ctx)` (every precondition, or a `Refusal` naming the one that fails) and
`apply(ctx)` (installs it, returns an `Applied`: the values in force, the sites patched, its label, an `undo`). A lever module declares
one with the decorator:

```python
from opt_core.mem import registry
from opt_core.mem.registry import Applied, refuse, register

def _applies(ctx):
    ctx.require("pair_offload", "module", "tensor")                                  # HookMissing = a refusal naming hooks.<key>
    return None

@register("pair_offload", family="offload", exact="bitwise", exact_reason="residency only: the bytes moved are the bytes computed",
          applies=_applies, preconditions=("hooks.module", "hooks.tensor", "torch.cuda"), settings=("rows",))
def pair_offload(ctx) -> Applied:
    h = ctx.require("pair_offload", "module", "tensor")
    rows = ctx.setting("pair_offload", "rows", 256, cast=int)                       # ctx.settings["pair_offload"]["rows"], else 256
    ...                                                                              # patch h["module"]; ctx.record.mark(...) from the levered path at run time
    return Applied(lever="pair_offload", settings={"rows": rows}, sites=(f"{type(h['module']).__name__}.forward",), undo=restore)
```

| family | what it names |
|---|---|
| `offload` | host residency of named tensors / modules with streamed transfers (pinned RAM, row / column blocks) |
| `chunk` | a chunked or streamed form of an operation (pair transition, triangle ops, heads, MSA rows, samples per pass) |
| `ckpt` | recomputation in place of storage: block checkpointing, cycle / recycle checkpointing, seed batching caps |
| `allocator` | the device allocator's policy (expandable segments, graph pools, cache release, peak counters) — never numerics |
| `jax` | the JAX-side levers (sub-batching, bucket policy, a flash kernel where one exists) |
| `kernel` | a memory-lean kernel routed in place of a stock op (the carried kernels of `opt_core.kernels`, cited by name) |
| `setting` | a named value of the engine's own configuration that lowers peak (a cap, a dropped unread feature, a guard) |

| label | meaning | proof the lever's tests carry |
|---|---|---|
| `bitwise` | the levered path returns the same bytes as the un-levered path | `torch.equal` on a synthetic module; the equality row of the kit |
| `band` | within the kit's documented tolerance; the reason names the re-ordered reduction | `allclose` at the documented tolerance; the kit's seed-spread band |
| `measured` | the equality suite decides per run | the kit's equality row |

The mode's label is composed from its applied levers' (`measured` if any, else `band` if any, else `bitwise`); the block names the
label per lever beside the declared one. A lever never widens its declared label at apply time (refused: a lever that cannot hold
its label declares the weaker one at registration); it narrows it (a chunk equal to the full dimension is bit-exact) only with
`Applied.narrowed_by` = the equality record id that proves the narrower label — recorded as a named event.

Levers this package registers itself (`allocator.py`): `expandable_segments` (allocator, process scope, bit-exact — `PYTORCH_CUDA_ALLOC_CONF`
`expandable_segments:True` for the kit process; refuses by name on `conf` (a pinned `expandable_segments:False`, a malformed conf),
`graphs` (CUDA-graph capture in the composed line, `ctx.graphs`), `torch` (not importable), `cuda_state` (an initialised CUDA without
the runtime API) and `environ` (a read-only `ctx.environ` on the env path); checked by read-back of
`torch.cuda.memory._snapshot()["allocator_settings"]` — pending until CUDA is up, re-checked at every unit boundary, finalised at the
exit gate) and `cache_release` (allocator, bit-exact — the policy of
`torch.cuda.empty_cache` at the kit's release points: `per_unit` | `per_stage` | `never`; the kit calls `allocator.release(ctx, point)`).
The levers of `offload.py` / `chunk.py` / `ckpt.py` / `jax_mem.py` are tabled in those modules' docstrings; `registry.table()` prints
every registered lever.

## The refusal grammar

A refusal names the lever and the precondition: `<lever>: <precondition>: <reason>` — `pair_offload: hooks.tensor: the kit named no
'tensor' hook …`, `expandable_segments: conf: PYTORCH_CUDA_ALLOC_CONF pins expandable_segments:False …`, `big: switch.x: switch 'x'
names no registered lever and no line lever …`, `pair_offload: registry: line lever 'pair_offload' is not registered …`. Under `strict`
(the default) any refusal raises `mem.Refused` with the record; the kit's NOT ACTIVE line is
`[tag] NOT ACTIVE: big refused — <lever>: <precondition>: <reason>[; …] (mode=big base=<base>)` (`record.not_active_line`).
A refusal is never a silent no-op and never a silent stock fallback: with `strict=False` the record carries the refusals, the mode line
lists only the applied levers, and the exit gate turns the refused line levers into `EXIT_NOT_ACTIVE`.

## The record, the ACTIVE line, the manifest block

`AppliedRecord` holds, in order, the levers applied and refused, every precondition checked, every setting read (value and source),
the kit's lever switches, the allocator settings in force (`found` as the process found them, every variable written, the levers'
values), the lever modules the install holds. The mode line is `big:<lever,lever,…>` (the applied levers in order). The offered
ACTIVE grammar is `[tag] ACTIVE mode=big:<levers> base=<base> exact=<label> refused=<names|none> off=<names|none> on=<names|none>
allocator=<k:v,…|none> [extra…]`; a kit whose ACTIVE bytes the launcher holds composes its own from the record's fields.
`record.attach(kit_fields)` writes `kit.big` into the kit's `opt_manifest.json` through `opt_core.manifest.build(kit=…)`.

## The census (total accounting per unit)

A unit is one item / pass the kit delimits (`unit_begin(id)` / `unit_end()`). Every lever has a scope: `unit` levers mark
themselves on every unit from their levered path (`record.mark(lever)`); `process` levers (an allocator policy, an exported variable)
are marked once on the implicit unit `process` — when applied, or, for a lever with a deferred check (`Applied.verify`), once the
read-back settles: pending checks re-run at every unit boundary and are finalised at the exit gate (a setting never readable by then is
the named fallback). A degraded path is a named event (`record.fallback(lever, reason)` — the
row-level mode field); a site with nothing to do names why (`record.skip(lever, reason)`). `record.census()` compares per unit the
expected lever set (the applied levers of the unit's scope, or the unit's own set when the kit narrowed it) with the observed one: a
lever that fell back or never marked is partial on that unit; a skipped lever is accounted with its reason and is not partial; a lever
that marked without being expected is `unexpected`, and partial when the kit's switches turned it off or the record refused it (a patch that
stayed in force). Marks refuse misuse by name (a unit opened over an open one; a unit-scope mark after the last unit closed).
`record.exit_gate(rc)` is the fail-closed gate through `opt_core.report.verdict`: any refusal or any partial unit turns a successful
exit into `EXIT_NOT_ACTIVE` (3) unless the kit's opt-out — its parsed `--allow-partial` passed as `apply(…, allow_partial=True)` or
`exit_gate(rc, allow_partial=True)` — with the opt-out recorded in the block. A run that
observed no kit unit is partial on every unit-scope lever unless the kit passes `expect_units=False` (a dry run). The partial lines:
`[tag] NOT ACTIVE: big partial — levers=<names> (<lever>: <unit>: <reason>; …); exit 3 (--allow-partial records and proceeds)`
and `[tag] PARTIAL allowed: levers=<names> (…) (--allow-partial, recorded)` (`--allow-partial` = `Ctx.opt_out`, the kit's flag as it spells it).

## Peak memory

The peak is the device peak per pass read by the launcher's memory reader (sampled `nvidia-smi` high-water mark). `peak.py`
is the instrument beside it — read-only: it samples and writes `peak_mem.json` beside each pass and resets nothing; it is not a lever,
not in `LEVER_MODULES`, and the model process imports it from the hook directory, never as `opt_core.mem.peak`. Its loader,
`<hook dir>/peakhook.pth`, is a MEASUREMENT-TIME file: `python -m opt_core.mem.peak install <dir> <python>` writes it into the model
interpreter's site-packages (a local write by the launcher, recorded as `loader: pth` in `peak_mem.json`); it carries no
kit's name, never lives under `<engine>/opt/`, and is installed by no kit build — so it is outside the kit `.pth` rule (a kit's build
backend emits exactly one generated, guarded autoload `.pth` and refuses any other) and outside the install-time namespace census of
`tests/test_selfcontained.py`. The torch counters
`allocator.counters()` (`max_memory_allocated` / `max_memory_reserved` in GiB) and `allocator.reset_peak()` are the adapter's: an
adapter that wants per-pass torch peaks resets at each pass start and reads after the pass. The XLA variables
(`XLA_PYTHON_CLIENT_PREALLOCATE` / `_MEM_FRACTION` / `_ALLOCATOR`, `XLA_FLAGS`) are recorded as found and never overridden: a JAX lever
exports through `allocator.jax_export`, which refuses by name on a conflicting pin and once the XLA backend is initialised.

## The `--n_gpu P` axis and the row-sharded pair stack (`mem.ngpu`, `mem.rowpair`)
`--n_gpu P` is a resource axis of `big` (explicit, default 1). `mem.ngpu` is the framework-free producer of its words: the ACTIVE/EXIT
token `n_gpu=P sharding=<scheme>` (`n_gpu=1 sharding=none`), the refusals (`refused: n_gpu>1 requires --mode big (sharded reductions
are not bit-exact)`; `refused: n_gpu=P visible=K`), and `TP_LEVER` = the canonical strategy id `F7.tensor_parallel` with its `opt_core.arch`
declaration; the torch and JAX stacks both import it. `mem.rowpair` is the torch stack: the kit process launches P workers
(`rowpair.launch.run_sharded`, NCCL, fail-fast teardown, per-rank records), the pair tensor is row-sharded over the block grid
`rowpair.dist.Layout` (bit-preserving gathers / all-to-all transposes / ring), and the cross-row statements — triangle multiplication
(`rowpair.trimul`), triangle attention (`rowpair.triatt`), attention-pair-bias and row-local schedules (`rowpair.transition`), the AF3-family
confidence reducer (`rowpair.confidence`) — run with full reductions per element on one rank (bit-exact vs the dense statement where the
kernel is launch-extent invariant, else inside a stated class: `n_gpu>1` is big-only and tested as a band). At `--n_gpu 1` an
adapter installs nothing and the statements refuse a P=1 layout by name. `mem.rowpair` holds no model-keyed recipe (every engine's
site -> primitive mapping is that kit's adapter); `import opt_core` / `opt_core.mem` / `opt_core.mem.rowpair` pull in no framework or model
library (torch is reached lazily at call time; a missing torch is a refusal by name) — `tests/test_rowpair_logic.py::test_clean_venv_import_without_torch`. Contract: `mem/rowpair/API.md`; install recipe with the
OpenFold3 worked example: `mem/rowpair/ADAPTER_GUIDE.md`; LEVER line: `[tag] LEVER name=F7.tensor_parallel state=on
impl=opt_core.mem.rowpair@<ver> origin=core strategy=F7.tensor_parallel n_gpu=P sharding=rowpair N= P= B= rank= rows=r0:r1 R= Rmax=
peak_alloc_gib_max= peak_alloc_gib_ranks= ranks= rows_tiled= trimul_RA= trimul_RB= tiles_source=budget|fixed …`.
