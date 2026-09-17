# opt_core.mem.rowpair — ADAPTER GUIDE (how a kit installs `--mode big --n_gpu P`)

Audience: a kit author with the kit's tree open and zero context on this package. Read `API.md` first (interface words, the
structural n_gpu=1 rule, the stage table, the module table, the lever / memory / numerics tables). This guide is the recipe: what the
adapter passes at each stage of an AF3-family forward so that the pair tensor is row-sharded from birth to the last head, plus one worked
example (OpenFold3); every other torch engine maps the same STAGES (its file:line list is the kit's TP-MAP) onto the same statements.

## 0. What the adapter is (and is not)
* The adapter is ONE module in the kit (`<pkg>/tp.py` or a section of `<pkg>/big.py`) that (a) adds the `--n_gpu` flag, (b) at P>1
  launches the kit's own predict on P ranks through `launch.run_sharded` (or `launch.run_rank_processes` for a CLI-per-rank engine), and
  (c) INSIDE each rank rebinds the engine's forward sites onto the statements of this package with `opt_core.mem.torch_rowchunk.PatchSet`
  (originals kept), passing the engine's sub-modules as CALLABLES. It never edits stock files and never copies a statement's body: the
  adapter holds engine attribute paths and tensor plumbing only; every row statement, schedule, block loop and collective is imported
  from `opt_core.mem.rowpair`.
* At `--n_gpu 1` the adapter installs NOTHING (no layout, no PatchSet, no environment). Its own test asserts this (`assert PATCHES.names()
  == []` at P=1). Every L2 statement / schedule / driver refuses a P=1 layout by name, so a mistake here is loud.
* The as-run schedule is printed for you: every rank prints `[<ROWPAIR_TAG>] SCHEDULE rank=q P=… <block sizes with sources, sub-block /
  ring / host-gather choices, replicated names>` when its entry returns (`launch.run_sharded` workers); a kit that wants the words on its own
  EXIT line calls `evidence.schedule_fields()` after the forward (a LEVER line printed at install time cannot carry them — the schedule is
  decided inside the forward).
* `proj` returns the pair schedule dtype (z's): its b slabs travel the ring and every rank sizes its receive buffers from `z.dtype`, so an
  autocast projection casts back to `z.dtype` before returning (refused by name otherwise); `out` / `gate` are local epilogue statements and may
  return any dtype torch adds into z. The same rule holds for any producer whose output is communicated (all-to-all windows, `ring_pass` blocks).
* Ring and transpose schedules stage through SCOPED wire buffers (`dist.Staging`): a ring pass holds its own-block copy + two receive
  blocks, a shard transpose its per-peer send / receive blocks, for that pass / call only — freed at its end once every transfer completed, so
  no caller tensor ever goes on the wire and nothing is freed in flight. A block a ring YIELDS (`ring_blocks`, `ring_pass`) is valid until that
  ring's next step — consume it, or copy out the rows you keep, before advancing; never retain the tensor. `staging_peak_mb` on the SCHEDULE /
  MEMSTATS lines is the largest wire set held at once; `ROWPAIR_P2P_SYNC=1` adds a stream synchronize after every point-to-point STEP (diagnostic).
* A hang inside a pair block on real GPUs is almost always an UNMATCHED point-to-point transfer (a rank-dependent argument to a schedule —
  `transpose_band` windows, `ring_pass` shapes, a `Layout` built differently per rank — or the adapter's own `comm().p2p` with a receive missing
  on the peer, or a communicated operand in a rank-dependent dtype): rerun with `ROWPAIR_P2P_CHECK=1` and the offending site refuses by name on
  every rank, with its call stack, instead of hanging.
* Weights are replicated tensors too — guard them once at install: `trunk.guard_replicated_params(module.state_dict(), "<module>")` per
  bound module (one object all-gather; a per-rank random init, a nondeterministic checkpoint load or a rank-local finetune is refused by name
  with the differing tensor names instead of producing silently mixed rows).
* P=1 OPM bound: `msa.opm_rows_budgeted(a, b, Layout(N, 1, 0), outer_fn, rows, C_z=…, out=z, add=True, allow_unsharded=True)` is the
  single-GPU row-blocked OPM (no `[N, N, c·c]` intermediate; the same statement the sharded path runs on its rows); every torch kit's
  big line may bind it as a NAMED lever of its own (default off; its census word `opm_layout=unsharded` on the LEVER line) — that is
  a kit memory lever, not the n_gpu adapter, so the 'installs nothing at n_gpu=1' rule above still holds for the row-sharding PatchSet.
* Kit levers that cannot coexist with sharding are wired OFF BY NAME under `n_gpu>1` in the ACTIVE line (a fused whole-tile
  triangle-multiplication kernel cannot run under the streamed schedule: `fpf_trimul=off reason=conflict:n_gpu`; CUDA-graph capture of a
  region that now contains collectives: `graphs=off reason=conflict:n_gpu`), never silently.
* Engine-named environment variables are the kit's: the adapter maps them onto the `ROWPAIR_*` names of `API.md` (or passes explicit
  arguments) before the trunk runs; no statement of this package reads an engine's variable. A choice between a replicated and a sharded
  form is always an explicit argument the adapter passes (`pwa_rows / msa_transition_rows(m_layout=, shard_tokens=)`, `opm_rows(a_local=)`,
  `run_trunk_sharded(gather=)`, `apb_local_queries(gather=)`, `heads.embed_rows(inplace=)`), never a size threshold.

## 1. The stages (every engine) — what the adapter passes
| stage | engine site | adapter passes | core statement |
|---|---|---|---|
| rank entry | after featurisation | `os.environ["ROWPAIR_CHUNK_ALIGN"] = str(chunk)` (the engine's pinned chunk size); the launch gives every rank process ONE `PYTHONHASHSEED` (the core launcher forms do it and print `[<tag>] RANKENV hashseed=… source=… ranks=P`; a kit with its own launcher layers its ranks' environment on `rankdata.ranks_env` and prints `rankdata.hashseed_word` on its launch line); either featurise on rank 0 and `rankdata.broadcast_features(feats | None, skip_keys=<parked keys>, host_keys=<host keys>)` (data form `rank0_bcast`: store rendezvous so ranks > 0 wait outside any collective, status word — a rank-0 failure refuses on every rank —, meta, tensors, receipt check) or featurise everywhere and `bcast.sync_tensordict_from_rank0(batch)` (`per_rank`), then `rankdata.assert_ranks_agree(rankdata.feature_digest(batch, exclude=<per-rank keys>), mode='refuse'|'census')` (the cross-rank input digest gate: tensors by bytes, non-tensor leaves canonical); optionally `msa_host.park_features(feats, ["msa", "deletion_matrix", …], mode=…, row_dims=…)` | `bcast`, `msa_host` |
| trunk entry (N known) | first statement of the trunk | `lay = dist.ctx(N, chunk)` (aligned policy) or `dist.Layout.auto(N, P, rank, chunk=chunk)` (grid policy) — ONE layout per query; `feats = template.slice_template_inputs_to_rows(feats, lay, PAIR_KEYS)` | `dist`, `template` |
| pair init (z born) | `z = linear(s_i) + linear(s_j) + relpos(...) + linear(token_bonds)` | `init_rows_fn(g0, g1)` = the engine's statement on rows: `trunk.outer_sum_rows(a, b, g0, g1) + linear_relpos(trunk.relpos_onehot_rows(...)) + linear_bonds(trunk.feature_rows(bonds, g0, g1, dev)[..., None])`; `init_like` = a tensor with z's lead dims / dtype / device | `trunk.init_pair_shard` (inside `run_trunk_sharded`) |
| recycling | `z = z_init + linear(LN(z_prev))`; `s = s_init + linear(LN(s_prev))` | `recycle_update_fn = lambda blk: linear_z(ln_z(blk))` (row-local); `single_recycle_fn(s_init, s_prev, cycle)` (replicated) | `trunk.recycle_shard_` |
| template embedder | `z += TemplateEmbedder(feats, z)` | `template_fn(z_loc, cycle)` = `template.template_embed_rows(z_loc, lay, n_templ=, c_t=, unit_rows_fn=<slots, rows → linear(LN(z rows)) + linear(feature rows) built from TemplatePairRows / distogram_rows / pair_mask_rows / same_or_group_rows>, pair_stack_fn=<the per-slot pair stack: `lambda u, mask: pairstack.pair_stack_(blocks, u[0], mask, lay)[None]` — the driver takes the 3-dim `[R, N, c_t]` shard, the template slab is `[1, R, N, c_t]`>, finish_fn=<relu → linear_out (+ LN) on rows>, mask_loc=, slot_groups=template.template_slot_groups(...))`; a templated run whose slots are all dummies: `template.note_all_dummy(census, templated=True)` (the kit decides note vs refuse). REAL templates without dense features (S3): FEATURIZER SIDE (data pipeline / runner, engine-specific) — skip the `[T, N, N]` products (distogram, unit vector, their 2-D masks) and ship the per-token PRECURSORS instead (pseudo-beta coords float64 + mask, backbone frame coords fp32 + mask, the featurizer's asym ids) under one enable predicate for every rank and cycle; MODEL SIDE (core) — `acc = template.real_template_rows(lay, T, pb_coords=, pb_mask=, frames=, frame_mask=, asym_id=, edges=, grp=)` and `unit_rows_fn` reads `acc.rows(key, slots, i0, i1)`; pass `feat_channels=acc.transient_channels` to `template_embed_rows`; print `slot_census(pb_mask, frame_mask, asym_id=).line()` on rank 0. Inter-chain groups: `spec = template.load_interchain_spec()`; `grp, events = template.group_feature(spec, chain_id_per_token, T, N)` (print the events); `template.interchain_words(asym_id, grp, spec)` on the lever line; with `interchain_scope() == 'all'` the engine's model-side chain masks (pairformer template mask) follow `same_or_group_rows` too — that binding is the adapter's | `template` |
| MSA module | `m, z = MSAModule(m, z)` | `msa_fn(z_loc, s_inputs, cycle)` = `msa.msa_module_sharded(m, z_loc, lay, blocks)` with per block `opm=lambda m, z: msa.opm_rows_budgeted(a, b, lay, outer_fn, C_z=, out=z, add=True, …)`, `msa_update=lambda m, z: m + pwa_rows(m, pwa_bias_rows(prep_fn, z, lay), lay, values_fn=, attend_fn=, out_fn=, chunk=)`, `msa_transition=lambda m: msa.msa_transition_rows(fn, m, lay, shard_tokens=<explicit>, m_layout=<explicit>)`, `pair_block=<pairstack.pair_block_ bound to this block's pair sub-modules>`; `m_layout="replicated"` (default) or `"token_sharded"` (m is this rank's token shard `[S, R, c_m]`; pass `a_local=True` to the OPM and `m_layout=` to `pwa_rows` / `msa_transition_rows`) is the adapter's lever, mapped from its own flag; the MSA subsample draw: `msa.draw_replicated(draw_fn)` or the engine's own draw under `trunk.guard_rng_replicated` | `msa`, `transition.opm_rows` |
| pair stack (Pairformer) | `s, z = PairformerStack(s, z)` | `pairstack_fn(z_loc, s, cycle)` = `pairstack.pair_stack_(blocks, z_loc, mask_loc, lay, s=s)` with `blocks[i] = pairstack.bind(trimul_out=TriMulFns(proj, out, gate, C_h), trimul_in=…, triatt_start=TriAttFns(ln, bias, attend, H), triatt_end=…, transition=fn, chunk=chunk, apb=<attn_fn for transition.apb_local_queries>, single_transition=fn, trimul_kw=dict(inplace_chunk=<the engine's in-place tri-mult column chunk — required>))` — the projections / gates / LayerNorms / attention kernel are the engine's modules; for shards of ≥ 2³¹ elements the adapter's `ln` / `proj` callables wrap `shard.ln_rows_guarded` (the streamed schedules call the callables on row blocks and slabs as given); an ending-attention module written as a starting statement on transposed input is passed as `triatt_end` (the driver transposes) | `pairstack`, `trimul`, `triatt`, `transition` |
| trunk exit | `del z_init`; heads follow | `out = trunk.run_trunk_sharded(lay, n_cycles=, …, gather="none")` → `out.z` is the shard `[*, R, N, C]`; `gather="rank0"|"all"` only for an engine stage the kit has not installed on rows (named in the kit's README as its replicated floor) | `trunk` |
| distogram head | `logits = linear(z) + linear(z)ᵀ` | `for i0, i1, L in heads.sym_logit_rows(fn, out.z, lay): write / reduce rows` | `heads` |
| confidence head | `z_conf = z + linear(one_hot(d_ij))`; 4 pair blocks; PAE / PDE / pLDDT | `z_conf = heads.embed_rows(lambda blk, g0, g1: blk + linear(one_hot(dist_rows(x, g0, g1))), out.z, lay)`; its pair stack = `pairstack.pair_stack_`; logits per row block → `confidence.RowBlockReducer(ChainIndex(asym_id, has_frame, is_ligand), lay.r0, lay.r1, device, finish="exact").consume(c0, c1, pae_rows, pde_rows, contact_rows)` then `.finalize(lay.bounds)` → summaries on rank 0 (keys renamed by the adapter; the full f16 PAE / PDE / contact matrices arrive in pinned HOST memory, assembled in column blocks) — or, for an engine whose summary FORMULAS differ, `confidence.ContextReducer.for_layout(contexts, lay)` with the engine's own `finish` callable; where an engine's finishing statistic is ROW-SEPARABLE (per-row means → a max / argmax over rows), reduce per row block inside the head loop and gather `[N]` vectors (`dist.gather_cat_to_rank0`) instead of a context matrix — rank 0 then never holds an `[n_D, n_D]` transient; the trunk shard's placement across the stage is ONE core statement, `plan = heads.ZTrunkPlan(out.z, passes=n)` (n = the sample-loop passes; levers `ROWPAIR_FREE_ZTRUNK` / `ROWPAIR_CONF_PARK_ZTRUNK`, both default 0): per pass `src, inplace = plan.begin(i, lead_out=(k,), out_dtype=<the pair input's dtype>)` → `heads.embed_rows(fn, src, lay, lead_out=(k,), inplace=inplace, out_dtype=…)` → the pass's pair stack / heads → `plan.end(i, device_needed_next=<a statement before the next pass, or after the stage, reads the device shard whole>)`; which mechanism when is decided THERE, never in the adapter: in place iff one sample-equivalent per pass, same dtype, the shard's last use, contiguous torch-owned storage (zero copies); else parked (device storage released BEFORE the `[k, R, N, C]` allocation, rows staged from pinned host, the same storage re-grown only on `device_needed_next`; a park not restored stays live across passes and keeps serving on the last pass); else resident (both levers off, or a lever declined by a NAMED reason); statements that read the device shard whole (distogram column slabs, a diffusion roll-out) run before `begin` when `plan.decide(i, …) != 'resident…'`; `plan.consumed` says the device tensor has stopped holding the rows; the words land in the schedule census (`conf_ztrunk=…`); per-token rows via `confidence.per_row_outputs_to_rank0`; frame / clash statements via `frames.nearest_atoms_rows` / `count_pairs_within` | `heads`, `confidence`, `frames` |
| diffusion | conditioning `z_cond = transitions(linear(LN(cat[z, relpos])))`; per block pair bias; atom attention pair input | `sched = diffusion.DiffusionSchedule.decide(lay, c_z=, c_in=, c_cond=, H=, S=, n_blocks=)`; `z_cond = diffusion.pair_cond_rows(embed_fn, out.z, lay, c_out=, rows=sched.cond_rows, transitions=[…])` once per roll-out (or `plan.source()` after `plan.park_now()` of the confidence stage's `heads.ZTrunkPlan` — the trunk shard parked at ROLL-OUT ENTRY under `ROWPAIR_CONF_PARK_ZTRUNK`, its device storage released for the whole roll-out and the confidence pass that follows: same rows, same output); per denoising step `a = diffusion.diffusion_transformer_sharded(blocks: [DiTBlockFns(norm, kv, attn, update, bias)], a, s, z_cond, lay, schedule=sched, bias_cache=PairBiasCache(sched.bias_cache))`; atom attention: `plan = diffusion.band_plan(q_idx, k_idx, pair_valid, N, max_w=W)`, `band, extras = diffusion.pair_band_rows(proj_fn, z_cond, lay, plan)`, `p_lm = diffusion.band_lookup(band, extras, plan)`; noise: `diffusion.sync_replicated(noise)`; sampler hook sites: `sampler_hook.call_init_noise` / `call_project` | `diffusion`, `sampler_hook` |
| exit | writer | rank 0 writes (`launch.is_output_rank()`); full pair-shaped matrices the writer needs come from the reducer's rank-0 results or `dist.gather_rows_to_rank0_host(x_shard, lay, block_bytes=)` (pinned host tensor on rank 0, column blocks, bounded device transient — the one assembly loop; no kit writes its own); `shard.unshard_rows_to_rank0` only when the writer needs the tensor ON the device | ranks > 0 skip the writer (named in their log) |

Replicated-by-design tensors (`s`, `s_inputs`, `m`, atom tensors, features, noise, coordinates) stay the engine's; they are PROVEN
identical where a divergence would be silent (`trunk.guard_replicated`, `trunk.guard_rng_replicated`, `diffusion.sync_replicated`,
`bcast.assert_replicated`) — a mismatch is a broken run, refused by name with the per-rank checksums.

### 1b. Fused triangle kernels on the row blocks (optional per mode; the torch statements stay the named fallback)
- Triangle multiplication: build the module's torch `TriMulFns` as today, then hand `trimul_update_` `trimul_fused.fused_trimul_fns(weights_of(module),
  stock_fns, eps=<the module's LayerNorm eps>, cells=<the kit's fpf_trimul_v4 cells row for this device>, stock_round=<True when stock adds the
  residual to a bf16 kernel output>, min_tokens=<default 2048: below it the statements run>)` instead — `weights_of` is the kit's existing single-GPU `fpf_v4` weight shim (`opt_core.trimul.WEIGHT_KEYS`).
  Nothing else in the call changes; a declined unit runs the torch statements and is counted; print `trimul_fused.emit_line(tag)` once per process.
- Triangle multiplication above the `2u` device floor (a resident shard plus a whole projected A plane): nothing in the adapter changes — the
  run sets `ROWPAIR_TRIMUL_ROWS_A=<rows>` (or the adapter passes `trimul_update_(..., RA=<rows>)`) and the update runs the A-BLOCK schedule: A resident
  `RA` rows per pass, `ceil(Rmax/RA)` balanced passes, z read-only through the call, the same fused / torch statements per tile on an output row
  block, a host row mirror of u bytes per rank (page-locked chunks from the park pool; `ROWPAIR_PARK_PIN_MAX_GB` / `_STRICT` / `_CHUNK_GIB` govern it),
  one copy back. One pass (the default) is the in-place schedule byte for byte. Append `trimul.describe_ablock()['words']` to the kit's TRIMUL census
  line so the run record says which schedule ran (`ablock=on ablock_ra= ablock_passes= ablock_src= … ablock_host=`); `ROWPAIR_TRIMUL_ABLOCK=0` is the
  by-name opt-out for bisecting.
- Triangle attention: wrap the engine's eager attention core once per module: `core = triatt.attention_core(stock=<engine core>, kernel=<the
  mode's word: flash_triattn | cueq | torch>, min_tokens=<gate>, scale=<None, or 1.0 when q is pre-scaled>, mask_from=<bias0 | a callable over
  the kit's key-mask rows>, serve_dtypes=<add float32 for an fp32 pair stack>, stock_qblock=<the engine's query chunk>)` and pass it as the
  `core_fn` of `attend_query_blocks` (qblock None: the fused kernels hold no logits) or call it inside `TriAttFns.attend`; the gathered bias goes
  in as `triatt.bias_hnn(tb_full, dtype)`; print `triatt.emit_core_line(tag)` once per process. The kit's own switch uses the same value words.
  Planes above the int32 bias-element bound (`H·N·N > 2**31-1`; H=4: N > 23,170) need nothing from the adapter: the core serves them through the
  flash kernel per query block (`ROWPAIR_TRIATT_FLASH_QBLOCK`, default 2048; ≈ 5–7 GB transient per call at N = 60,120 on top of the gathered
  bias; schedule word `triatt_flash_qblock`), and `ROWPAIR_TRIATT_INT32_GUARD=1` is the one-word switch back to the refusal
  (`fallback_by=unsupported:bias_elems>int32`, the engine core serves) should a run need it.
- Both are Tier 2 (tolerance class): a mode that names them is a big-tier mode; the census words above are the mode's activation evidence.
- Outcomes of a constructed provider, one rule for both: the documented GATES (size `below_gate`, dtype / width / head-dim classes
  `dtype:` `c=` `unsupported:<why>`, the opt-outs `env_torch` / `kernel_torch`) run the torch statements COUNTED at exit 0; a card without a
  tuned row runs the lever's SAFE settings (`opt_core.kernels.safe_settings`: one stderr line, census `settings=safe:<reason>`); where the
  lever cannot run at all (no CUDA device, no Triton / kernel import, no row and no safe settings, safe settings failing to build) the call
  raises `RowpairRefused` naming the lever and the opt-out — the kit's existing refusal handling turns it into its non-zero exit; a kit's CPU
  unit tests either stay below the size gate, set `ROWPAIR_TRIMUL_KERNELS=torch` / `kernel="torch"`, or expect the refusal.

## 2. Launcher wiring (kit CLI)
```python
from opt_core.mem import rowpair as RP
from opt_core.mem.rowpair import launch, evidence
P = RP.refuse_unless_big(args.n_gpu or 1, args.mode)          # 'refused: n_gpu>1 requires --mode big (...)' -> print exc.reason, exit 2
RP.refuse_unless_visible(P)                                       # 'refused: n_gpu=P visible=K' (P == 1 is not probed)
ACTIVE = kit_active_line(..., *evidence.fields(P))                # n_gpu=P sharding=rowpair | n_gpu=1 sharding=none
if P == 1:
    result = predict_one(request)                                 # today's bytes
else:
    try:
        result = launch.run_sharded(P, predict_rank, request, mode=args.mode, run_timeout_s=args.timeout)   # predict_rank: module-level
    except launch.RankFailed as exc:
        emit(evidence.rank_failed_line(TAG, exc)); sys.exit(EXIT_FAIL)
EXIT = kit_exit_line(..., *evidence.fields(P), *evidence.peak_fields(result["peaks"]),
                     ranks_ok=f"{sum(r['ok'] for r in launch.last_run())}/{P}" if P > 1 else None)
```
`predict_rank(request)` runs on every rank: it builds the model as the kit does today, installs the PatchSet (P>1 only), runs predict,
prints ONE LEVER line (`evidence.emit_rank0(evidence.lever_line(TAG, "on", P, layout=lay, peaks_gib=evidence.gather_peaks(),
**dict(evidence.rows_census(lay))))` — the schedule census rides on it: every block size with its source, `msa_m=replicated`,
`park_z_init=`, `templ_*`, `diff_*`), and returns host data (dict of floats / paths / CPU tensors) from rank 0.
Engines whose model process must be a fresh interpreter per rank (import-time CUDA init, a CLI that owns argument parsing) use
`launch.run_rank_processes(P, argv_of=lambda r: [...], isolate_devices=True|False, run_timeout_s=...)`; the rank process calls
`dist.init_from_env()` itself at start-up (the namespaced `ROWPAIR_*` environment is already set) and `dist.destroy()` at exit.
Opt-in measurement: `census.mark("trunk_entry" | "after_template" | "after_msa" | "after_pairstack" | "no_gather" | "distogram" |
"confidence" | "diffusion_start" | "diffusion_peak" | "done")` at the stage boundaries (no-ops unless `OPT_CORE_TP_CENSUS=1`).

## 3. Worked example — OpenFold3 (`openfold3_opt`), sites from the kit's TP-MAP
z is `[1, N, N, 128]` fp32 (TF32 matmuls); chunked stock ops use chunk sizes that divide 16 → `ROWPAIR_CHUNK_ALIGN=16`, `lay = dist.ctx(N, 16)`.
| TP-MAP site (stock path under openfold3/) | install |
|---|---|
| featurisation / batch | every rank runs the data pipeline (every rank process started with the launch's one `PYTHONHASHSEED`, `hashseed=` on the launch line); `bcast.sync_tensordict_from_rank0(batch, skip_keys=<per-rank keys>)`, then `rankdata.digest_features` / `ranks_agree` over the shared keys (`[feats] rank r shared <d16> … ranks_identical=`; differing digests refused on every rank, `feats_ranks_differ`); raw MSA + deletion features parked (`msa_host.park_features` — its facts carry per-key `where` words and the parked row count `rows` a placeholder rank reads instead of a broadcast, `ROWPAIR_MSA_HOST=rank0` on 8 ranks); template pair keys sliced to rows at trunk entry |
| pair init + relpos + token bonds (`input_embedders.py`) | `init_rows_fn` from `trunk.outer_sum_rows` / `relpos_onehot_rows` (clip = the config's `max_relative_idx`, chain condition = `same_rows(asym_id)`) / `feature_rows(token_bonds)`; `trunk.run_trunk_sharded(lay, n_cycles=1 + num_recycles, …)` replaces the trunk loop of `model.py` |
| recycling embed (`model.py`: `z = z_init + linear_z(layer_norm_z(z))`) | `recycle_update_fn = lambda blk: linear_z(layer_norm_z(blk))`; `ROWPAIR_PARK_ZINIT=1` above the kit's stated size |
| template embedder (`template_module.py`) | `template.template_embed_rows` with the embedder's `linear_z` / `layer_norm_z` / feature linears in `unit_rows_fn`, its 2-block pair stack through `pairstack.pair_stack_`, `linear_out` (+ v2 LayerNorm) in `finish_fn`; `ROWPAIR_TEMPL_PARK_Z/U` above the kit's stated size |
| MSA module (`msa_module.py`, `outer_product_mean.py`, `msa_pair_weighted_averaging.py`) | `msa.msa_module_sharded` — OPM: `outer_fn(a_rows, b, g0, g1)` = the stock einsum + mask-norm rows + `linear_out`; PWA: `prep_fn` = `linear_z(layer_norm_z(z_rows))` (+ mask bias), `values_fn` = `linear_v/linear_g(layer_norm_m(m chunk))`, `attend_fn` = the stock softmax·V einsum on local query rows, `out_fn` = `linear_o`; MSA subsample under `trunk.guard_rng_replicated`; the block's pair stack = `pairstack.pair_block_` |
| Pairformer 48 blocks (`pairformer.py`, `triangular_multiplicative_update.py`, `triangular_attention.py`, `attention_pair_bias.py`) | `pairstack.pair_stack_` with per block `bind(trimul_out=TriMulFns(proj=<LN_in → sigmoid(linear_*_g)·linear_*_p·mask>, out=<LN_out → linear_z>, gate=<sigmoid(linear_g(LN_in(z)))>, C_h=c_hidden), trimul_in=…, triatt_start=TriAttFns(ln=layer_norm, bias=linear, attend=<the stock mha with the kit's kernel (SDPA / DS4Sci / flash) through triatt.attend_query_blocks>, H=no_heads), triatt_end=<the ending module, a starting statement on transposed input>, transition=pair_transition, chunk=16, apb=<AttentionPairBias body with queries = local rows>, single_transition=single_transition, trimul_kw=dict(inplace_chunk=256))` (256 = the stock `_inference_forward` column chunk) |
| distogram head, confidence head (`prediction_heads.py`) | `heads.sym_logit_rows(distogram_linear, z, lay)`; `heads.ZTrunkPlan(zij_trunk, passes=<sample-loop passes>)` → `heads.embed_rows(src, inplace=…)` for `z_conf` per pass (`OF3TP_FREE_ZTRUNK` / `OF3TP_CONF_PARK_ZTRUNK` → the `ROWPAIR_*` names); the 4 confidence pair blocks through `pairstack.pair_stack_`; PAE / PDE logits per row block into `confidence.RowBlockReducer(..., finish="exact")` over `lay.chunks(confidence.max_row_chunk(N))`; pLDDT / per-token rows via `per_row_outputs_to_rank0`; frame atoms via `frames.nearest_atoms_rows`, clash counts via `frames.count_pairs_within` |
| diffusion module (`diffusion_conditioning.py`, `diffusion_transformer.py`, atom attention encoder/decoder) | `diffusion.DiffusionSchedule.decide` once; `pair_cond_rows` once per roll-out (the kit's conditioning-once lever acts on the shard); `diffusion_transformer_sharded` per step with `DiTBlockFns` built from each block's `AttentionPairBias` (norm / kv / attn / update / bias sub-statements); atom attention pair input through `band_plan` / `pair_band_rows` / `band_lookup` (W = the atom-attention window in tokens); `sync_replicated(noise)`; sampler hook sites wired to `sampler_hook.call_init_noise` / `call_project` |
| writer (`OF3OutputWriter`) | rank 0 only |
OpenFold3-specific: pytorch-lightning must stay in single-device mode — do NOT pass `torchrun_names=True` (its ClusterEnvironment keys off
`LOCAL_RANK`); the DeepSpeed / DS4Sci attention kernel is a per-query-row kernel and composes with `attend_query_blocks` unchanged; the
fused triangle-multiplication kernel lever is off by name under `n_gpu>1`.

## 4. The adapter's dev checks (per change-set; small; scratch, not rows)
1. P=1 bit-exact vs the pre-change kit on 2–3 small equality cases (the structural rule makes this hold by construction; the check proves the
   CLI plumbing did not move a byte).
2. The core's CPU suite (`python -m pytest tests` in `common/opt_core`) green in the kit's environment (CPU; `threaded` + gloo): every row statement vs its dense
   statement at P ∈ {2, 3, 4} (`tests/test_rowpair_*_043.py`, `test_rowpair_e2e_043.py` on the synthetic AF3-family model
   `tests/synthetic_af3.py`). An adapter's own callables are checked the same way: `opt_core.testing.run_ranks(P, fn)` runs `fn(rank, P)`
   on P rank-threads of one process — compare the installed statement with the stock module on a small N, `torch.equal` in fp32.
3. One mid-size end-to-end (a bin that fits one card, 1000–2000 tokens): P=2 vs P=1 inside the engine's fast band; LEVER line shows
   `n_gpu=2 sharding=rowpair rows_tiled=true ranks=2` and the schedule; `launch.last_run()` shows `2/2` ranks ok.
4. Peak memory per card at 2–3 mid sizes for P=1/2/4(/8) with `OPT_CORE_TP_CENSUS=1`: `census.fit_peak_vs_P` gives per-rank peak =
   a(N) + b(N)/P; the adapter's documentation says which replicated terms make up a(N) at each size.

## 4b. Replicated floor — a STATED property of every kit's `big_xP` rows
Whatever stays replicated sets a per-card memory FLOOR that more ranks cannot lower. The kit's README applicability table and its CHANGES
`big_xP` rows state that floor WITH NUMBERS at ≥ 2 bins (a one-card size and the kit's top P=1 bin) and NAME the replicated tensors, e.g.
"a(N) at 10,000 tok = m (1.6 GB) + s / s_inputs + atom tensors + the triangle bias per call (1.6 GB); every pair term ÷P". With every stage
of §1 installed nothing pair-shaped is replicated; what remains replicated in this package by design: `m[S, N, c_m]`, `s`, per-token
features, atom tensors and the diffusion atom attention, the triangle-attention bias `[N, N, H]` per call, the atom-attention band
`[N, 2W+1, c]`, and rank 0's full matrices at the writer (memory model table in `API.md`). A kit that leaves a stage on the replicated
form (`gather=` at trunk exit, an uninstalled head) names that stage in the same rows.

## 5. Failure semantics the adapter inherits (nothing to write)
Any rank raising → all ranks torn down → `RankFailed(event=rank_failed, rank, exitcode, log)` (under `run_rank_processes` the peers of a
rank that exited non-zero first get `fail_grace_s` — default 60 s; the launcher prints `[<tag>] RANKEXIT rank= exitcode= grace_s=` — to
finish; a rank that died on a signal (exit code < 0 or > 128) tears them down at once); a rank hung after
another finished → `rank_timeout` after `finish_grace_s`. A FAILED rank's own group teardown is bounded: `dist.destroy()` on a failure path
(a non-zero registered `dist.exit_code`, `dist.failure()`, an exception propagating through the caller or already reported uncaught) gives
the library `ROWPAIR_ABORT_TIMEOUT_S` (60 s; silent in time; on overrun one `RANKLEAVE` line, the pending EXIT tallies and `os._exit` there and
then); joining the group registers that leave `atexit` and arms the exit guard, which ends a FAILED process still in its exit hooks or threads
`ROWPAIR_EXIT_GRACE_S` (90 s) after its main thread (`RANKSTUCK` line). A healthy rank is never bounded or ended: its leave is the library's
own `destroy_process_group`, which may wait for the peers (rank 0 still writing outputs) — that wait is correct. The adapter's two teardown
obligations: call `dist.exit_code(rc)` the moment its verdict is known (the forced exits use it; unregistered is 70; a non-zero code is
what marks a handled failure as one), and keep nothing that can block on the device or the group unbounded in its own FAILURE epilogue (no
`cuda.synchronize()` / collective after a failure — leave that to `dist.destroy()`); its own `dist.destroy()` call is optional on failure
paths (the registered leave runs regardless) and is the normal leave on healthy ones; the whole call bounded by `run_timeout_s`; NCCL collective timeout `nccl_timeout_s`; a killed
launcher leaves no rank behind (parent-death guards); ranks > 0 write everything (Python and C-level) to `<log_dir>/rank<r>.log`;
CUDA OOM on any rank propagates as that rank's failure with the OOM text in its log (never converted); a replicated tensor that differs
across ranks, an unknown placement word, an uneven case a statement does not support, a templated run with zero real slots (when the kit
asks for refusal) — each refused by name (`RowpairRefused`, message = `exc.reason`).
