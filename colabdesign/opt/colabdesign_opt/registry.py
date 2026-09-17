"""The lever registry — the ONE source of truth for the kit route's levers: per lever, in REGISTRY ORDER (= install order), what it
changes, its numerics class against stock, the module that applies it (this package's adapter over the shared core), the name its LEVER line
carries and the form of that line (evidence.py parses it). A mode's lever set is DERIVED from this table (modes.levers_of): `fast` = every lever here, in this
order; `exact` = the levers of numerics `exact` (the mode exists only while at least one is registered); a mode word's subtractive form
`<mode>-no-<lever>` (names.ABLATION_SEP) runs a mode without named members of its set. Facts only, each with file and line; no lever value lives here.

Numerics words (NUMERICS), per lever against stock at BindCraft's pinned settings:
* `exact`     — stock's bytes by construction: fusion, hoisting, compile / capture levers that keep stock's operations and their association
                (the same design states, trajectory and files bit for bit at the same seed);
* `precision` — the method's mathematics unchanged, floating-point rounding or association changed (bf16 rounding points, an online softmax,
                a re-associated accumulation, a product class below stock's): within the per-step band `fast` is held to against stock's reference
                design states (gradient / logit cosine, sequence argmax agreement, every loss term), never bitwise;
* `approx`    — the mathematics itself approximated, inside the same per-step band (none registered).

Supersession: a lever may declare `supersedes=(<lever>, ...)` — levers it replaces at the same call site in the same tier (a different attention
kernel on the calls another serves). A mode's set is registry order minus every lever a member of the set supersedes; the replaced lever stays
registered and prints `state=off reason=replaced` in that mode's arm. Dropping the superseding lever with the subtractive word RESTORES the lever it
replaced, loudly (`mode_set(candidates)` is computed over the levers that remain: `fast-no-triatt` runs pallas, the ACTIVE line says `ablated=triatt
restored=pallas`, pallas prints its normal line; `fast-no-pallas-no-triatt` runs stock attention). A lever that supersedes another is never itself replaced by a third (no chains).

Lever module protocol (every `module` below; levers.install drives it): `install() -> dict` (idempotent, before any model is built; a lever
that cannot run here raises one of the module's `REFUSALS` — levers.install then steps it aside BY NAME: `state=skipped reason=cannot_run`, the mode runs the rest), `installed() -> bool`, `uninstall()` (restore the
upstream objects it replaced), `off_line(reason) -> str` (its LEVER line with state=off), `evidence() -> dict`; a kernel lever registers its
exit census line with `kernels.register_exit_line` (one printer per process, registry order).
"""
from __future__ import annotations

from typing import Dict, NamedTuple, Tuple

from .names import LEVER_TRIATT, LEVER_OPM_FOLD, LEVER_LN, LEVER_PROJ, LEVER_TRIMUL, LEVER_COMPILECACHE, LEVER_LOWERCACHE, LEVER_HOIST_PREV, LEVER_PARCOMPILE, LEVER_NOSUB, LEVER_NOSUB_FN, LEVER_PALLAS, LEVER_TXLA, LEVER_TRANSITION

NUMERICS = ("exact", "precision", "approx")


class Lever(NamedTuple):
    id: str
    what: str                                             # what the lever changes
    applied_by: str                                       # the adapter (this package) and the shared implementation (opt_core) it imports
    evidence: str                                         # the form of its LEVER line (evidence.py parses it)
    requires: str
    numerics: str                                         # one of NUMERICS: the lever's class against stock (decides its modes: modes.levers_of)
    module: str                                           # the module that applies it (the lever protocol above)
    line_name: str                                        # the name= token of its LEVER line
    supersedes: Tuple[str, ...] = ()                      # levers this one replaces at the same site in the same tier (see Supersession above)


LEVERS: Dict[str, Lever] = {
    LEVER_COMPILECACHE: Lever(
        LEVER_COMPILECACHE,
        "the design step's two XLA executables (ColabDesign's `fn` = jit(_model) and `grad_fn` = jit(value_and_grad(_model)), colabdesign/af/model.py `_get_model`) "
        "persist across processes in jax's own persistent compilation cache, placed and keyed by the shared core (`opt_core.jax_design.pcc`): a process whose "
        "(program, token count, flags, stack) was compiled before on this machine loads the executable instead of compiling it; nothing the step computes changes",
        "opt/colabdesign_opt/compilecache_jax.py (the lever: the directory rule, jax.config + the arm's environment before the first compile, jax's own cache events counted) over `opt_core.jax_design.pcc` (the key and placement)",
        "`[colabdesign-opt] LEVER name=compilecache state=on impl=jax_design.pcc@<core> origin=core dir=<path> dir_source=<default|JAX_COMPILATION_CACHE_DIR> key=<stack key|adopted> autotune=jax_xla_cache requests=<n> hits=<n> misses=<n> compiled_s=<s> saved_s=<s> entries=<a>-><b> reinit=<0|1> source=exit pid=<pid>` at process exit; requests=0 is `missing` (a design process always compiles)",
        "jax with its persistent-cache module (the pinned stack); a cache directory that cannot be created or written steps the lever aside by name (`CompileCacheError`, state=skipped reason=cannot_run: set JAX_COMPILATION_CACHE_DIR or XDG_CACHE_HOME to a writable directory)",
        "exact", "colabdesign_opt.compilecache_jax", LEVER_COMPILECACHE,
    ),
    LEVER_LOWERCACHE: Lever(
        LEVER_LOWERCACHE,
        "the design step's two programs skip jax's trace + lower + cache-key on every model build after the first of a configuration: the COMPILED executable of "
        "`fn` / `grad_fn` is serialized per call signature beside the compile cache and, in a later process or a later trajectory of the same process (BindCraft builds a "
        "fresh model per trajectory, so stock re-traces and re-lowers AlphaFold every trajectory), deserialized and called directly; nothing the step computes changes",
        "opt/colabdesign_opt/lowercache.py (wraps `colabdesign.af.model.mk_af_model._get_model`; key = stack + code content hash + lever set + model configuration + call signature) over `jax.experimental.serialize_executable`",
        "`[colabdesign-opt] LEVER name=lowercache state=on impl=serialize_executable@kit origin=kit numerics=exact dir=<path> calls=<n> memo_hits=<n> loads=<n> stores=<n> traced=<n> fallbacks=<n> load_s=<s> retrace_s=<s> traced_s=<s> store_s=<s> "
        "mb_stored=<n> entries=<at install>-><at exit> relower=<off|same:n|DIFFERENT:n> why=<none|reasons> source=exit pid=<pid>` — calls>0 = applied (a design process always builds a model); "
        "`COLABDESIGN_OPT_LOWERCACHE=relower` lowers + compiles afresh on every load and compares the first call's outputs bit for bit (the key's self-test)",
        "jax with `jax.experimental.serialize_executable` (the pinned stack) and colabdesign importable; a directory that cannot be created or written steps the lever aside by name "
        "(`LowerCacheError`, state=skipped reason=cannot_run); an entry that fails to load or refuses the arguments takes the traced path (fallbacks=<n>), never an error",
        "exact", "colabdesign_opt.lowercache", LEVER_LOWERCACHE,
    ),
    LEVER_PARCOMPILE: Lever(
        LEVER_PARCOMPILE,
        "XLA compiles each design executable's GPU kernels on a thread pool: the LLVM module is split into shards optimised and assembled (LLVM -> PTX -> cubin) in parallel "
        "(`--xla_gpu_enable_llvm_module_compilation_parallelism=true --xla_gpu_force_compilation_parallelism=<threads>`, appended once after upstream's own XLA_FLAGS assignment); "
        "HLO optimisation, buffer assignment and every kernel's code unchanged — cold compile seconds only, nothing once compilecache is warm",
        "opt/colabdesign_opt/launchpad_parcompile.py (install: import colabdesign — its __init__ assigns XLA_FLAGS wholesale — then append the two flags before jax initialises a backend, "
        "stepping aside by name when a backend already exists; threads = min(16, the CPUs this process may run on: affinity, cgroup quota), at least 2)",
        "`[colabdesign-opt] LEVER name=parcompile state=on impl=xla_llvm_module_parallelism@kit origin=kit numerics=exact threads=<n> cpus=<n> flags=<the two flags> xla_flags_kept=<upstream's> source=install pid=<pid>` "
        "once at install (the flags are in force from there; XLA reads them when jax creates its backend)",
        "nothing beyond the kit arm and the CPUs the process may use; a backend that already exists in the process steps the lever aside by name (`ParcompileError`, state=skipped reason=cannot_run)",
        "exact", "colabdesign_opt.launchpad_parcompile", LEVER_PARCOMPILE,
    ),
    LEVER_HOIST_PREV: Lever(
        LEVER_HOIST_PREV,
        "the recycle features never leave the device between design steps: `_af_design._recycle` takes the recycle-0 zeros as device-resident float32 arrays made once "
        "(stock rebuilds float64 host zeros every step) and `_af_design.run`, when one model ran (BindCraft: always), keeps `prev` off the host np.stack/mean path and "
        "puts the device arrays back (colabdesign/af/design.py:80-133, :147-206 re-stated with those two changes); every value stock reads is stock's byte for byte",
        "opt/colabdesign_opt/hoist_prev.py (the lever; kit code, no core part)",
        "`[colabdesign-opt] LEVER name=hoist_prev state=on impl=hoist_prev@kit origin=kit patched=_recycle,run zero_builds=<n> zero_inits=<n> device_prev_steps=<n> multi_model_stock_path=<n> run_calls=<n> source=<install|exit> pid=<pid>` at install and at exit (the last line wins; device_prev_steps=0 at exit is `fallback`)",
        "colabdesign at its pin (the two method bodies are the pinned ones, sha256-locked by the tests)",
        "exact", "colabdesign_opt.hoist_prev", LEVER_HOIST_PREV,
    ),
    LEVER_NOSUB: Lever(
        LEVER_NOSUB,
        "the forward-only and the forward+backward executables get their own config objects and sub-batch decisions: `fn` keeps stock's rule "
        "(chunks of 4 above 384 tokens), `grad_fn` — the design step — is traced unchunked at every size (no memory policy: chunking the remat'd backward pass bounds "
        "nothing this kit runs); at or below 384 tokens both equal stock's and the programs are stock's (the size gate reads `gated`)",
        "`opt/colabdesign_opt/nosub.py` (the `_af_prep._prep_model` replacement) deciding through `opt_core.jax_design.subbatch_policy` "
        "(grad: `fixed(value=None)`, unchunked at every size, source `kit`; fn: `requested=stock`) and counting through `opt_core.attn.size_gate.SizeGate(min_tokens=385)`",
        "`[colabdesign-opt] LEVER name=nosub state=<on|skipped> [reason=gated] impl=subbatch_policy@<core> origin=core tokens=<L> "
        "grad_subbatch=none grad_subbatch_source=kit fn_subbatch=<none|4> fn_subbatch_source=stock gate.nosub=min385 calls= served= gated= fallback=` once per distinct model build (nosub.py)",
        "nothing beyond the kit arm (the decisions consult no device size)",
        "precision", "colabdesign_opt.nosub", LEVER_NOSUB,
    ),
    LEVER_NOSUB_FN: Lever(
        LEVER_NOSUB_FN,
        "the forward-only executable (`fn`, recycle 0, no gradient: it feeds `prev` into the graded step) traced WITHOUT ColabDesign's sub-batch chunking at every size, as `nosub` traces the forward+backward one (stock chunks `fn` by 4 above 384 tokens, colabdesign/af/prep.py:30; at or below it stock's `fn` is unchunked already and the lever reads `skipped reason=gated`)",
        "opt/colabdesign_opt/nosub_fn.py (the lever) over opt/colabdesign_opt/nosub.py's hook and `opt_core.jax_design.subbatch_policy` (the decision, source `kit`)",
        "`[colabdesign-opt] LEVER name=nosub_fn state=<on|skipped> [reason=gated] impl=subbatch_policy@<ver> origin=core tokens=<T> fn_subbatch=none fn_subbatch_source=kit stock_fn_subbatch=<4|none> gate=…` at model build (nosub.py's hook)",
        "nothing beyond the kit arm (the decision consults no device size)",
        "precision", "colabdesign_opt.nosub_fn", LEVER_NOSUB_FN,
    ),
    LEVER_TRIMUL: Lever(
        LEVER_TRIMUL,
        "AF2-Multimer TriangleMultiplication (outgoing `ikc,jkc->ijc` and incoming `kjc,kic->ijc`, the fused-projection form), forward and backward, served by the kit's "
        "fused Pallas/Triton kernels: input LayerNorm + projections + gates as one prologue, one batched GEMM, centre LayerNorm + output projection + gate as one epilogue, "
        "a custom_vjp backward of the same shape; every call of the design model served (a configuration without fused projection weights runs stock's own program, "
        "counted `not_fused` - declared; AF2-Multimer v3 has none); bf16 products with fp32 accumulation like stock, fewer bf16 rounding points",
        "`opt/colabdesign_opt/kernels/trimul_fused.py` (the haiku class rebinding over `colabdesign.af.alphafold.model.modules.TriangleMultiplication`, the Ledger and its "
        "fail-closed gate, the exit line) over the shared core's provider row `cd_trimul` (`opt_core.kernels.pallas`: the kernels, the custom_vjp op, a pure-jnp reference — "
        "this kit's `kernels/trimul_pallas.py` as the core carries it, selected by word through `opt/colabdesign_opt/kernels/provider.py`)",
        "`[colabdesign-opt] LEVER name=trimul_pallas state=on impl=trimul_pallas@<sha8> origin=core served=<n> fallback=<n> fallback_by=<reason:n,...|none> "
        "shapes=<N..xC..xE0|E1:n,...> provider=opt_core.kernels.pallas@<core> row=<row:n> tier=<arm:cells,...> [uncovered=<n>:<family/direction>,...] word=<word> numerics=precision "
        "precision=<bf16|tf32|bf16+tf32> source=exit pid=<pid>` once at exit (served counts TRACED calls: 6 per program, 12 per design process at one token count)",
        "jax >= 0.5 with the GPU backend and its Pallas/Triton lowering (tested 0.6.0); otherwise the lever steps aside by name at install (`trimul_fused.Refusal`, state=skipped reason=cannot_run), never degraded",
        "precision", "colabdesign_opt.kernels.trimul_fused", "trimul_pallas",
    ),
    LEVER_PALLAS: Lever(
        LEVER_PALLAS,
        "flash attention with pair bias and key mask, forward and backward (the tree's ONE AF2 Pallas/Triton kernel `opt_core.kernels.pallas_attn`), "
        "for every eligible `Attention` call of the design model — triangle attention, MSA row and column attention (`all_calls`); the extra-MSA "
        "row attention (8 per head, under the kernel's 16 floor) runs stock's math, counted `head_dim_lt_16` (declared); bf16 rounding points changed, fp32 accumulation, exact online softmax",
        "`opt/colabdesign_opt/pallas.py` over `opt_core.kernels.pallas_attn_serve` (`require`, `ledger(min_tokens=0, expected=(head_dim_lt_16,))`, "
        "`enable([colabdesign.af.alphafold.model.modules], all_calls=True)`: the served call, the eligibility rule, the haiku class rebinding, the Ledger and its fail-closed gate)",
        "`[colabdesign-opt] LEVER name=F1.pallas_attn state=<on|skipped> impl=pallas_attn@<sha8> origin=core served=<n> fallback=<n> fallback_by=<reason:n,…> min_tokens=0 shapes=<B..xH..xS..xD..:n,…> scope=all_calls source=exit pid=<pid>` at process exit (pallas.py)",
        "jax >= 0.5.0 with the GPU backend and its bundled Pallas/Triton lowering (tested 0.5.3 / 0.6.0), compute capability >= 8.0 (the H100 sm_90 and the A100 sm_80 both serve it); otherwise the lever steps aside by name at install (`F1.Refusal`, state=skipped reason=cannot_run), never degraded",
        "precision", "colabdesign_opt.pallas", "F1.pallas_attn",
    ),
    LEVER_TRIATT: Lever(
        LEVER_TRIATT,
        "every `alphafold.model.modules.Attention` call of the design model (triangle start/end, MSA row with pair bias, template point attention; extra-MSA row attention's "
        "head dim 8 padded to 16 inside the kernel) served by the kit's Triton flash-attention kernels written for AF2's shapes: pair bias and mask read in-kernel, "
        "several (batch, head) pairs per program, exact online softmax with fp32 statistics, bf16 products with fp32 accumulation, a fused backward "
        "(dK/dV/d-bias-partials in one kernel, dQ in another); calls with fewer than 16 keys (MSA column attention over 2 sequences) run stock's method "
        "verbatim, counted `below_keys_rule` — replaces `pallas` on the same calls",
        "opt/colabdesign_opt/kernels/triatt_lever.py (the lever: opt_core's Attention interception `pallas_attn_serve` pointed at the op, the Ledger and its fail-closed gate, "
        "the exit line) over the shared core's provider row `cd_triatt` (`opt_core.kernels.pallas_triatt`: the kernels and the custom_vjp op — this kit's `kernels/triatt_attn.py` + "
        "`kernels/attbwd_dkdv.py` as the core carries them, selected by word through `opt/colabdesign_opt/kernels/provider.py`)",
        "`[colabdesign-opt] LEVER name=triatt state=on impl=triatt_attn@<sha8> origin=core served=<n> fallback=<n> fallback_by=<below_keys_rule:n|none> (...) numerics=precision precision=bf16 (...) "
        "provider=opt_core.kernels.pallas@<core> word=<word> row=<row:n> tier=<arm:cells,...> [uncovered=...] source=exit` "
        "once per process at exit (served > 0; a fallback reason outside {below_keys_rule} is a defect the gate names)",
        "jax >= 0.5.0 with the GPU backend and its Pallas/Triton lowering, compute capability >= 8.0; otherwise the lever steps aside by name at install (`F1.Refusal`, state=skipped reason=cannot_run), never degraded",
        "precision", "colabdesign_opt.kernels.triatt_lever", LEVER_TRIATT, supersedes=(LEVER_PALLAS,),
    ),
    LEVER_OPM_FOLD: Lever(
        LEVER_OPM_FOLD,
        "OuterProductMean (`modules.OuterProductMean`; every Evoformer and extra-MSA block) re-associated: T[a,d,c,f] = sum_e right[a,d,e]*W[c,e,f] formed in f32, then ONE f32 GEMM out[b,(d,f)] = left[b,(a,c)] . T[(a,c),(d,f)] with K = S*c — no [N,N,c,c] outer product, no chunk scan, no output transpose; + output_b, / (1e-3 + mask norm) and both masks exactly as stock; backward = XLA autodiff of the same expression; a by-name gate above S = 32 sequences (`gated_s_gt_32`: a pessimisation there; unreachable at the pinned settings, S = 2 / 1)",
        "`opt/colabdesign_opt/kernels/layers_opm.py` (pure JAX, no custom kernel): rebinding `colabdesign.af.alphafold.model.modules.OuterProductMean` to a same-name subclass whose __call__ reads the stock parameters (layer_norm_input, left_projection, right_projection, output_w, output_b) under their own scopes — parameter tree unchanged",
        "`[colabdesign-opt] LEVER name=opm_fold state=on impl=layers_opm@<sha8> origin=core numerics=precision precision=tf32 served=<traced calls> fallback=<n> fallback_by=<gated_s_gt_32:n|none> shapes=<S<s>xN<n>…:n,…> "
        "provider=opt_core.kernels.pallas@<core> word=<word> row=<row:n> tier=<arm:cells,…> [uncovered=…]` at process exit (the re-association served as the shared core's provider row `cd_opm` — "
        "this kit's module as the core carries it, selected by word through kernels/provider.py); served=0 is `missing`; a fallback_by word outside {gated_s_gt_32} is `fallback` (fail-closed)",
        "jax (any backend — plain jnp einsums); steps aside by name only if no backend initialises",
        "precision", "colabdesign_opt.kernels.layers_opm", "opm_fold",
    ),
    LEVER_LN: Lever(
        LEVER_LN,
        "every `common_modules.LayerNorm` call of the design model over the last axis with C a power of two in [32,1024] — Evoformer MSA / pair norms, extra-MSA and template stacks, recycling norms, structure-module pair norm (the 384-channel single-representation norms run stock's method, counted `channels_not_pow2` — declared) — by ONE Pallas/Triton row kernel forward (f32 statistics in registers, one read, one write) and ONE backward (d_x; d_scale / d_offset plain JAX, dead code in the design step); each instance's eps, axis / param_axis, create_scale / create_offset and variance form (use_fast_variance) honoured; installed after `trimul` (the norms `trimul`'s fused module absorbs never reach it)",
        "`opt/colabdesign_opt/kernels/layers_ln.py`: rebinding `colabdesign.af.alphafold.model.common_modules.LayerNorm` to a same-name subclass (haiku wraps the subclass __call__; parameters scale / offset f32 [C] under the instance's own scope — parameter tree unchanged)",
        "`[colabdesign-opt] LEVER name=ln state=on impl=layers_ln@<sha8> origin=core numerics=precision precision=none served=<n> fallback=<n> fallback_by=<channels_not_pow2:n|…> shapes=<C<c>x<bf16|f32>:n,…> "
        "provider=opt_core.kernels.pallas@<core> word=<word> row=<row:n> tier=<arm:cells,…> [uncovered=…]` at process exit (the kernels served as the shared core's provider row `cd_ln` — this kit's "
        "kernels as the core carries them, selected by word through kernels/provider.py); served=0 is `missing`; a fallback_by word outside {channels_not_pow2} is `fallback`",
        "jax >= 0.5 with the GPU backend and its bundled Pallas/Triton lowering (tested 0.6.0, sm_90); otherwise the lever steps aside by name at install (state=skipped reason=cannot_run), never degraded",
        "precision", "colabdesign_opt.kernels.layers_ln", "ln",
    ),
    LEVER_PROJ: Lever(
        LEVER_PROJ,
        "the q, k, v and gating input projections of every kernel-served Attention call as ONE GEMM over the LayerNorm'd activation ([B·S, C] x [C, 2·H·dk+2·H·dv]; two GEMMs q|gate, k|v when stock's sub-batched forward passes q_data/m_data as distinct slices) instead of stock's four contractions each re-reading the activation; q/k/v handed to the active attention op heads-major exactly as the served call hands them (the softmax scale passed to the op exactly once, never applied to q), the gate logits sliced token-major; one dX and one dW GEMM in the backward pass; calls with head dim < 16 (the extra-MSA row attention) go to the class below unchanged, counted `head_dim_lt_16`",
        "`opt/colabdesign_opt/kernels/proj_attn.py`: a subclass of the serve layer's rebound `modules.Attention` under the same class name (haiku module names and the parameter tree unchanged), composed on the serve layer's record (op + ledger) of the ACTIVE attention-kernel lever (`triatt` in fast; `pallas` when `fast-no-triatt` restores it); installed after that lever",
        "`[colabdesign-opt] LEVER name=proj state=on impl=fused_qkvg@kit origin=kit served=<n> fallback=<n> fallback_by=<head_dim_lt_16:n|none> shapes=… numerics=precision precision=bf16 requires=attn_kernel active=<the kernel lever's line name> traced=<n>` at process exit; served=0 is `missing`; a fallback_by word outside {head_dim_lt_16} is `fallback`",
        "an installed attention-kernel lever of this kit in the same run (the serve layer's record for colabdesign's modules package: `triatt`, or `pallas`); without one the lever steps aside by name at install (levers.NEEDS_ATTENTION_KERNEL: state=skipped reason=no_attention_kernel) — so `fast-no-pallas-no-triatt` runs stock's attention with proj (and txla) skipped, named",
        "precision", "colabdesign_opt.kernels.proj_attn", "proj",
    ),
    LEVER_TRANSITION: Lever(
        LEVER_TRANSITION,
        "every `modules.Transition` of the design model (the Evoformer's pair_transition [N,N,128] and msa_transition [S,N,256] per block, the extra-MSA "
        "stack's, the template pair stack's: LayerNorm -> Linear(C->4C) -> relu -> Linear(4C->C)) served by the fused ReLU-transition row the shared "
        "core's JAX-family kernel provider (`opt_core.kernels.pallas`, op `transition`) names for the call's measured cell under the TIER word — the two "
        "GEMMs + bias + relu (and, in the provider's `cd_transition` row, the LayerNorm) as ONE Pallas/Triton kernel forward and ONE backward per call, "
        "the 4C intermediate on-chip per row tile (XLA writes and re-reads the [M, 4C] tensor through HBM in both directions); stock's parameter names, "
        "dtypes and bf16 rounding points; a call whose cell names the stock statement (`xla`) or a row the adapter does not bind runs the stock class BY "
        "NAME (`fallback_by=cell_<row>:n`) — a card whose main cell (pair transition, C=128) names the stock statement still serves the cells that name "
        "a bound row (`admit=cells`); a word the provider refuses or a row that does not import steps the lever aside by name at install (`state=skipped "
        "reason=cannot_run detail=…`) — the kit carries no kernel and no size / card / tile table for this lever",
        "opt/colabdesign_opt/kernels/layers_transition.py (modules.Transition rebound to a same-name subclass: the adapter — class rebinding, structural step-aside words, census; the kernel is the provider's row module, bound through kernels/provider.py)",
        "`[colabdesign-opt] LEVER name=transition state=on impl=<row module>@<sha8> origin=core numerics=precision precision=bf16 variant=<fused|lnkeep> "
        "served=<n> fallback=<n> fallback_by=<cell_xla:n|dtype_not_bf16:n|…|none> shapes=<C..:n,..> admit=<main|cells> provider=opt_core.kernels.pallas@<core> word=fast row=<row:n,…> tier=<arm:cells,…> [uncovered=…] source=exit` at exit",
        "the jax GPU backend with the Pallas/Triton lowering (levers.NEEDS_GPU) and a provider row of a bound form for the model's main cell on this card; without either the lever steps aside by name at install (state=skipped reason=cannot_run) and the mode runs the rest of its set",
        "precision", "colabdesign_opt.kernels.layers_transition", LEVER_TRANSITION,
    ),
    LEVER_TXLA: Lever(
        LEVER_TXLA,
        "the FORWARD-ONLY attention calls of the design model (ColabDesign's `fn`: recycle 0, backprop=False; BindCraft's forward-only predict calls) served by "
        "opt_core's triangle-attention bridge `opt_core.kernels.triattn_xla` (an XLA-FFI custom call over the tree's kernels — on compute capability 9.0 the routed "
        "sm_90a CUDA kernels `triattn_native` / `cuda_sm90a` and the Triton ahead-of-time cubins `k2b_aot`, on 8.x `cuda_80` / `k2b_aot`; the row of every call is the "
        "bridge's OWN measured table: the lever hands it `impl=auto` and names no row and no size): the op lever "
        "`triatt` serves becomes a jax.custom_vjp whose PRIMAL calls the bridge (q scaled and rounded once where triatt rounds it, heads-major = layout BNHSD, "
        "the key mask honoured through lax.cond: mask-free kernel when all ones; a caller under vmap served slice by slice through a custom_vmap face) and whose "
        "forward / backward RULES are triatt's forward and backward — so the graded step (`grad_fn`, hk.remat included) runs triatt exactly as without this lever "
        "and only the forward-only executable changes kernel (served_grad=0); calls below 16 queries / keys or an unpadded head dim below 16 (`small_call`) and calls the "
        "bridge refuses by name at trace time (`refused`) run triatt's op unchanged",
        "opt/colabdesign_opt/txla.py (the lever: the custom_vjp op over triatt's, the serve layer re-served with triatt's ledger / scope / rules and op = the bridged op, "
        "lever proj re-stacked on the re-served class, the Ledger, the exit line) over `opt_core.kernels.triattn_xla` (the bridge: row selection, launch, refusal by name)",
        "`[colabdesign-opt] LEVER name=txla state=on impl=triattn_xla@<core> origin=core served=<n> fallback=<n> "
        "fallback_by=<small_call:n,refused:n|none> shapes=<B..xH..xS..xD..:n,…> rows=<row:n,…|none> served_fn=<n> served_grad=0 vjp_traced=<n> "
        "vmap=<n> refused=<shape:n,…|none> layout=BNHSD impl_word=auto requires=triatt grad=triatt proj_restacked=<0|1> numerics=precision precision=bf16 cc=<cc> tx=<version> "
        "source=exit pid=<pid>` once per process at exit (served = TRACED primal calls the bridge took: 8 per forward-only program; rows = the bridge's own census of kernel binds); "
        "stepped aside at install: levers.install's `state=skipped reason=cannot_run detail=<NeedsTriatt|TxUnavailable>:… source=install`",
        "lever `triatt` serving the modules package (without it — `fast-no-triatt`, pallas restored; no attention-kernel lever — install raises NeedsTriatt and levers.install "
        "steps the lever aside by name) and opt_core >= 0.5.27.0 importable (the bridge; else TxUnavailable, stepped aside the same way); REFUSALS = (NeedsTriatt, TxUnavailable); "
        "a call no row serves on this card / dtype / shape is triatt's by name at trace time, never a refusal",
        "precision", "colabdesign_opt.txla", LEVER_TXLA,
    ),
}

ORDER: Tuple[str, ...] = tuple(LEVERS)                    # registry order = install order = the order of every levers= list the package prints
assert all(l.numerics in NUMERICS and l.module and l.line_name and l.id == k and set(l.supersedes) <= set(LEVERS) - {k} for k, l in LEVERS.items())
assert not any(LEVERS[x].supersedes for l in LEVERS.values() for x in l.supersedes)   # a lever that supersedes another is never itself replaced by a third (no chains: mode_set is one pass)


def replaced_by(lever: str, among: Tuple[str, ...] = None) -> Tuple[str, ...]:
    """The levers of `among` (default: every registered lever) that declare they supersede `lever`, registry order; () when none does."""
    among = tuple(LEVERS) if among is None else among
    return tuple(k for k in among if lever in LEVERS[k].supersedes)


def mode_set(candidates: Tuple[str, ...]) -> Tuple[str, ...]:
    """A mode's lever set from its candidate levers: registry order minus every candidate another candidate supersedes."""
    cands = tuple(k for k in LEVERS if k in candidates)
    return tuple(k for k in cands if not replaced_by(k, cands))


def levers_of_numerics(word: str) -> Tuple[str, ...]:
    """The registered levers of numerics class `word`, in registry order."""
    if word not in NUMERICS:
        raise ValueError(f"numerics word {word!r} is not one of {NUMERICS}")
    return tuple(k for k, l in LEVERS.items() if l.numerics == word)


def line_names() -> Dict[str, str]:
    """{lever id: the name= token of its LEVER line}."""
    return {k: l.line_name for k, l in LEVERS.items()}
