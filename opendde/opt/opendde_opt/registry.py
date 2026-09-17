"""The lever registry: one entry per lever of the kit carried under ``opendde/opt/``, and the kit knobs with their defaults.

A lever is one runtime optimization with its own switch. This table describes levers (kit, class, tier, switch, the kit file that
reads the switch); it never applies one — the kit's own code does (``stack.py`` executes the line's shim). ``modes.LINES`` lists
levers by these names.

Classes follow the lever, directories follow the kit: a kernel or a bind inside the model forward is ``forward`` wherever the
kit lives; ``serving`` is the process model only (the post-runner install hook); ``datapath`` is host-side work outside the model
forward (feature pipeline, allocator, writing).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

CLASSES = ("forward", "datapath", "serving")
TIERS = ("exact", "tolerance", "tier2")
KL = "forward/fast_inference/levers/KIT"            # the kit layer: the cuEquivariance tuning-cache location (no table shipped)
AC = "forward/fast_inference/levers/ACCEL"          # the hook (served levers) + the ACCEL levers
AR = "forward/fast_inference/levers/ARMT"           # the ARM U/Z add-on
DT = "forward/fast_inference/levers/DITFAST/tools"  # DITFAST tools (dit_hoist, dit_align)
SR = "forward/fast_inference/src"                   # fpf_engines (the exact line's op adapter)
XL = "forward/fast_inference/levers/XL"             # the XL add-on: tri_ln (exact)
OF = "forward/fast_inference/levers/OFFLOAD"        # the offload unit: host-resident pair tensors (big)
SM = "forward/fast_inference/levers/SAMPLER"        # the SAMPLER unit: the diffusion-sampler kernel levers


@dataclass(frozen=True)
class Lever:
    name: str
    kit: str            # modes.KIT_DIRS key
    cls: str            # forward | datapath | serving
    tier: str           # exact | tolerance | tier2
    switch: str         # the switch spelling the kit reads (env), or "-" when the lever is a process model with no switch
    file: str           # kit file:line under opendde/opt/ that reads the switch / implements the lever
    routes: str         # "cli": the switch takes effect in the `pred` process (the one route)
    what: str           # one sentence, the kit's own description
    label: str = ""     # the kit's own short label (A1, H2a, E1, ...)



# The upstream accelerator sites (lncensus.KERNEL_SITES) a lever REPLACES with a kit kernel when its line carries it — the KERNELS census
# prints `off-by-route:<levers>` for such a site (the library still serves the lever's below-gate / out-of-scope delegations, counted), and
# `engaged` for a site no lever of the line owns. One row per site; every name is a LEVERS key (tests hold it).
KERNEL_SITE_OWNERS: dict[str, tuple] = {
    "cueq_triatt": ("arm_u", "rowpair_tp",
                    "pair_offload_trunk", "pair_offload_conf", "pair_offload_struct"),
    "cueq_trimul": ("arm_u", "arm_u23", "fpf_trimul_exact", "trimul_core", "trimul_exact", "rowpair_tp",
                    "pair_offload_trunk", "pair_offload_conf", "pair_offload_struct"),
}

# Upstream's own triangle-kernel flag (settings.KERNEL_KNOBS) -> the kit levers of that site a STATED value other than `auto` composes OUT of the line by
# name (opendde_opt/stockknob.py; LEVER state=off reason=aside:stock_knob:<flag>=<value>). The ARM add-on's inseparable attention / TriMul sites read
# the same fact at install (ODDE_STOCK_KNOBS) and stay on the stock op; the row-sharded line's own pair-stack kernels (rowpair_tp, tp_triatt) are not
# upstream's call sites and are unaffected. Every name is a LEVERS key with a composition rule (tests hold it).
STOCK_KNOB_LEVERS: dict[str, tuple] = {
    "triatt_kernel": ("triattn_core", "triattn_exact", "triattn_conf"),
    "trimul_kernel": ("trimul_core", "trimul_exact", "arm_u23", "fpf_trimul_exact"),
}

HOUSE = "house"                                   # a lever of the tree's own package (opendde_opt/), not of a carried kit

LEVERS: dict[str, Lever] = {
    # --- levers/KIT: the kit layer ---------------------------------------------------------------------------------------------------
    "cueq_tuned_cache": Lever("cueq_tuned_cache", "fast_inference", "forward", "exact", "CUEQ_TRITON_CACHE_DIR=$KIT/levers/KIT/cueq_cache_shipped",
                              f"{KL}/cueq_cache_shipped/", "cli",
                              "the cuEquivariance TriMul tuning-cache location exported to the kit layer (no table shipped: the library's own packaged "
                              "per-card tiles serve, the same the stock route reads -- bitwise-equal outputs; a user may add tuned entries there)", "C1"),
    "drop_bond_mask": Lever("drop_bond_mask", HOUSE, "datapath", "exact", "-", "opendde_opt/bondmask.py", "cli",
                            "the stock featurizer's unread [N_atom, N_atom] int64 `bond_mask` feature dropped before the feature dict moves to the GPU "
                            "(8 B x N_atom^2 of device memory for the whole prediction; no consumer at inference) — bitwise by construction, the tree's own lever"),
    "lnstream": Lever("lnstream", HOUSE, "forward", "exact", "-", "opendde_opt/lnstream.py", "cli",
                      "upstream's own fused-LayerNorm kernel source (LAYERNORM_TYPE=fast_layernorm, extension fast_layer_norm_cuda_v2) JIT-built as "
                      "fast_layer_norm_cuda_v2_cs with its 25 stream-less kernel launches put on torch's current stream (device code text identical: "
                      "bitwise by construction) and bound where upstream binds its own — the LayerNorms of a CUDA-graph capture are in the graph; the "
                      "prerequisite of every graph lever, placement-neutral on its own; the tree's own lever"),
    "tmpl_dedup": Lever("tmpl_dedup", HOUSE, "forward", "exact", "-", "opendde_opt/tmpldedup.py", "cli",
                        "TemplateEmbedder.forward embeds each class of bitwise-identical template slots once (the featurizer pads the slots to 4; "
                        "a no-template query carries identical dummy slots) and runs the stock accumulation over all slots verbatim — the c=64 "
                        "2-block template pair stack runs once per distinct slot per recycle instead of once per slot (bitwise by construction; the "
                        "slot classes decided once per item, memoised on the feature tensors); the tree's own lever"),
    "keep_pool": Lever("keep_pool", HOUSE, "datapath", "exact", "-", "opendde_opt/keeppool.py", "cli",
                       "torch.cuda.empty_cache made a counted no-op at the stock model's in-forward call sites (after the trunk, in and after the "
                       "confidence head: opendde/model/opendde.py, modules/confidence.py via utils/torch_utils.cleanup_device_memory) so the blocks a "
                       "phase cached serve the next phase instead of a synchronous cudaFree storm and a pool re-growth; the stock runner's per-item / "
                       "per-seed releases and every kit caller pass through unchanged (allocator policy: the live tensors are the same; on the big lines it "
                       "follows the offload size gate like chunk_lift — bound below it, off with the offload unit and under --n_gpu P>1); the tree's own lever"),
    "stepgraph": Lever("stepgraph", HOUSE, "forward", "exact", "-", "opendde_opt/stepgraph.py", "cli",
                       "the diffusion sampler's denoiser step captured into ONE CUDA graph per sample_diffusion() call (the core's GraphCache: call 1 eager — "
                       "the DITFAST hoist records there — call 2 eager-first + capture + a replay held bit-exact under the deterministic recipe, calls 3..200 "
                       "replayed: copy-in of x_noisy / t_hat, one graph launch), dropped at the call's end so every item re-captures on its own shapes; the stock "
                       "loop, RNG draws and Euler update untouched; size-gated on the query's residue tokens (ODDE_STEPGRAPH_MIN_TOKENS / _MAX_TOKENS), composed "
                       "out of a line running the offload unit, never on --n_gpu P>1, above a card's row (stepgraph.CARD_ROWS: sm_80 caps it at 1,024 tokens, "
                       "card_gate:sm80_replay_not_bit_exact_above:1024) and above the line's sampler-word row (stepgraph.WORD_ROWS: big up to 400 tokens from 0.2.66 — "
                       "measured not ahead at 800 —, 200 at 0.2.65; word_gate:<word>_above:<cap>); inside an admitted sampler call (stepgraph.planning()) the step's tier-word bindings "
                       "choose as under capture — odde_apb_bind the graph-timed provider column, ln_core upstream's LayerNorm — so the eager oracle is the captured "
                       "step's kernels (0.2.66); needs lnstream (a LayerNorm capture canary refuses by name without it); a capture whose "
                       "replay differs from the eager step before any graph output is consumed steps aside by name to the eager sampler (aside:verify_mismatch_...); the tree's own lever"),
    "sampler_amp": Lever("sampler_amp", HOUSE, "forward", "tier2", "-", "opendde_opt/precision.py", "cli",
                         "the diffusion sampler under the runner's ambient bf16 autocast: configs.skip_amp.sample_diffusion=False — the value upstream's "
                         "OPENDDE_FORCE_SAMPLE_DIFFUSION_AMP sets — applied at upstream's own policy site update_inference_configs, per item, for items of at most "
                         "MODEL_OPT_SAMPLER_AMP_MAX_TOKENS tokens (default upstream's own 3840: every item below upstream's AMP regime); the variable itself stays "
                         "absent on every route; composes with stepgraph (the autocast casts are kernels of the captured step); off the big offload line (pair_offload turns the row "
                         "off: without the step graph and with host-streamed pair tensors the casts cost +33 % sampler at 1,400 tokens); tier 2 (bf16 sampler numerics, inside "
                         "stock's seed-to-seed band on the identity canaries). No default line since 0.2.52: on top of the sampler kernels (dit_attn_apb, dit_fused, dit_lowp, "
                         "atom_fused) the word no longer pays — fast with vs without it, H100, 2 warm-ups + 4 timed interleaved: forward 10.91 vs 10.92 s at 400 tokens, "
                         "31.00 vs 30.65 s at 800 (+1.1 %; sampler 3.66 vs 3.60 / 7.09 vs 6.93 s), peak allocated 8.82 = 8.82 / 17.13 vs 18.28 GiB (a 1.15 GiB memory word at 800); "
                         "the multimer canary changes basin with it on (identity note), so it leaves fast and the resident big line and stays selectable by name", "P1"),
    "sched_host": Lever("sched_host", HOUSE, "forward", "exact", "-", "opendde_opt/schedhost.py", "cli",
                        "the diffusion sampler loop's two per-step host round-trips removed: the noise schedule carries a host copy read once per call, so the "
                        "loop's `c_tau > gamma_min` is decided on the host in the schedule's dtype (generator.py:204), and the per-step random rotation drawn "
                        "on the host exactly as stock reaches the device by a non-blocking copy (model/utils.py:69); every device operand, op and order is "
                        "stock's (bitwise by construction); a CPU schedule or CPU coordinates pass through by name; never on --n_gpu P>1; the tree's own lever"),
    "zprep_hoist": Lever("zprep_hoist", HOUSE, "forward", "exact", "-", "opendde_opt/zprephoist.py", "cli",
                         "the denoiser's per-step pair preparation made once per sampler call: f_forward's `permute_final_dims(z, [2,0,1]).contiguous()` "
                         "(diffusion.py:1512, one full fp32 (c,N,N) pair copy on every one of the 200 steps; its source is the conditioning hoist's recorded "
                         "LayerNorm buffer, the same tensor every step) answered from a one-entry memo keyed by the source's identity + version counter; "
                         "bitwise (the recorded copy is the tensor stock computed); the tree's own lever"),
    "structok_sync": Lever("structok_sync", HOUSE, "forward", "exact", "-", "opendde_opt/structoksync.py", "cli",
                           "the structural-token expander's per-role-pair projections (StructuralTokenExpander._pair_project_by_role_full / _chunk / _tile) "
                           "served with ONE host read per call instead of a torch.any read per role pair plus two boolean-mask reads per taken pair "
                           "(structural_tokens.py:311-313, several hundred per item inside the trunk phase): pair ids counted on the device and read once, a "
                           "stable sort enumerating each pair's rows in the mask's own order, the same projection modules on the same rows, index assignment "
                           "as the stock statement; bitwise the stock result; never on --n_gpu P>1; the tree's own lever"),
    "writer_overlap": Lever("writer_overlap", HOUSE, "datapath", "exact", "-", "opendde_opt/writer_overlap.py", "cli",
                            "upstream's result dump (runner/dumper.py DataDumper.dump) run on ONE background writer thread (opt_core.host.outputs.AsyncWriter, "
                            "max 2 items outstanding) over an owned host copy of the prediction taken in the item loop, drained in InferenceRunner.close, a "
                            "failed write printed / reported / raised; byte-identical files by construction. CARRIED BY NO LINE: measured on H100 the thread "
                            "shares the interpreter lock with the next item's featurisation and forward (upstream's dump is pure-Python JSON encoding), so the "
                            "loop gains 0 to 3 % while fwd_s loses 7 % at 800 / 1,200 tokens; json_oneshot removes the cost at its source instead; a writer in "
                            "a child process is the open form of this lever; the tree's own lever"),
    "json_oneshot": Lever("json_oneshot", HOUSE, "datapath", "exact", "-", "opendde_opt/writer_overlap.py", "cli",
                          "upstream's JSON writer (opendde/utils/file_io.py save_json, called by runner/dumper.py for every confidence document: "
                          "`json.dump`, the standard library's pure-Python streaming encoder — nearly the whole result-dump wall for the full_data "
                          "document's ~3 x N_token^2 floats per sample) bound at the name the dump calls (runner.dumper.save_json) to upstream's statement "
                          "with the document encoded once by `json.dumps` (the C encoder for indent=None, the same Python encoder for indent=4) and "
                          "written once: identical bytes (the standard library's dumps == ''.join(iterencode) contract), a fraction of the CPU; wherever "
                          "the dump runs (the loop or writer_overlap's thread); off the --n_gpu P>1 line; the tree's own lever"),
    "prefetch": Lever("prefetch", HOUSE, "datapath", "exact", "-", "opendde_opt/prefetch.py", "cli",
                      "upstream's inference DataLoader built with one worker instead of none (configs.num_workers 0 -> 1 at "
                      "runner.inference._create_inference_dataloader_synchronized; the `opendde pred` CLI has no option for it): the worker process "
                      "featurises item k+1 (JSON entities, CCD / RDKit, tokenisation, MSA features — seconds per item, serial in the stock loop) while "
                      "item k runs, the loop's next() returns at once; the dataset's own __getitem__ output over torch's worker queue, CPU tensors only, "
                      "nothing of the forward changes. Byte identity is kept by replaying the loop's per-item RNG reset (seed_everything) in the worker "
                      "right before each item is featurised (the featuriser draws random numbers: the reference conformers' rigid transform), and the "
                      "featurisation-phase house counters (drop_bond_mask's) tick in the worker and are relayed to the parent's books through shared "
                      "memory at every next(); a one-item query, a caller's own num_workers and a multi-rank process group step aside by name, and so does a run of "
                      "two or more seed passes over a query with a ligand given as SMILES (or FILE_): RDKit embeds such a ligand on its process-global "
                      "generator, which the stock loop advances across seeds and a worker forked afresh per seed would restart (aside: "
                      "rdkit_rng_multiseed, every line — the reason is a sampling one, not a tolerance one); off the "
                      "--n_gpu P>1 line; the tree's own lever"),
    "chunk_lift": Lever("chunk_lift", HOUSE, "forward", "tier2", "-", "opendde_opt/chunklift.py", "cli",
                        "upstream's fixed 450 M-element score-budget clamp on the dynamic attention chunk (opendde.py _bound_pairformer_chunk_size) replaced, "
                        "per item and per resolution site (residue / structural tokens), by upstream's own threshold-table value when the probed device's free "
                        "memory admits the reference statement's [c, H, n, n] fp32 score tensors at that value (else the clamp's value; one CHUNK line per "
                        "decision): fewer, larger chunk_layer launches in the trunk's triangle attention and outer-product mean, the refiner, the sampler's atom "
                        "attention and the confidence head; per element the arithmetic is the un-chunked statement's, yet the outputs are "
                        "byte-identical to --mode off under --det 1 at 800 / 1,000 tokens but NOT at 400 tokens (H100: the output set differs with vs without the lever) — a lifted chunk changes the kernels' reduction order at some sizes, so the lever is tolerance class and rides fast and big only"
                        " (fast, and big below its offload size gate) — the tree's own lever"),
    "rowpair_tp": Lever("rowpair_tp", HOUSE, "forward", "tier2", "- (--n_gpu P: opt_core.mem.rowpair over P rank processes)", "opendde_opt/tp.py; opendde_opt/tp_struct.py; opendde_opt/tp_diffusion.py", "cli",
                        "the row-sharded pair tracks of `--mode big --n_gpu P` (P>1): pair init, the 48-block trunk, the template and MSA-module pair blocks, the "
                        "structural stage (expansion rows from the ring-fetched parent rows, the refiner), the diffusion pair conditioning + transformer biases and "
                        "the distogram / confidence heads all run on this rank's row block of z through the core's row statements — triangle multiplication through "
                        "the ring contraction, triangle attention with the all-gathered bias (ending node through the shard transpose), transitions and "
                        "attention-pair-bias on local rows; nothing N x N x c is whole on a card or on the host; the template pair features are born as this rank's "
                        "[T, R, N, *] rows by the featurizer (tp_feats; a whole [T, N, N, *] tensor is refused by name at the runner's move); m is token-sharded (ROWPAIR_MSA_M_LAYOUT); under P>1 it serves "
                        "what pair_offload_struct / diffz / bigln_guard / free_templ serve on one card (replaced_by=rowpair_tp; ran = presharded_calls and "
                        "zcond_rows and diff_denoise_calls); the kit launches the P ranks itself (one visible card each; TORCH_NCCL_AVOID_RECORD_STREAMS=1: collective operands reusable by the allocator at once; ROWPAIR_TRANSPOSE_INPLACE=1: the shard transposed inside its own storage); refused by name under any other mode and "
                        "with fewer than P cards; tier 2"),

    "alloc_auto": Lever("alloc_auto", HOUSE, "datapath", "exact", "- (PYTORCH_CUDA_ALLOC_CONF via opt_core.mem.torch_alloc)", "opendde_opt/alloc.py", "cli",
                        "the expandable-segments CUDA allocator (`opt_core.mem.torch_alloc`) exported for the kit process of a `pred` call (placement only, "
                        "outputs unchanged by construction); another allocator configuration already present, or CUDA initialised before activation, is a refusal BY NAME"),
    # --- levers/DITFAST/tools ---------------------------------------------------------------------------------------------------
    "dit_hoist": Lever("dit_hoist", "fast_inference", "forward", "exact", "ODDE_ADDON_LEVERS=dit_hoist", f"{DT}/odde_addon.py:319-328", "cli",
                       "every step-invariant tensor of the diffusion denoiser computed once per sample_diffusion call by the stock op on the stock inputs "
                       "and replayed for the remaining steps"),
    "dit_align": Lever("dit_align", "fast_inference", "forward", "exact", "ODDE_ADDON_LEVERS=dit_hoist,dit_align", f"{DT}/odde_addon.py:319-328", "cli",
                       "the hoisted [1,16,N,N] fp32 pair biases live in a row-pitch-multiple-of-8 allocation so SDPA does not re-pad them per call; rides on dit_hoist"),
    # --- levers/ARMT: the ARM U/Z add-on -----------------------------------------------------------------------------------------
    "arm_z": Lever("arm_z", "fast_inference", "forward", "exact", "ODDE_ARM_Z=1", f"{AR}/odde_arm_t/__init__.py:57,685-697", "cli",
                   "arm Z: transpose-free PairformerBlock pair update only (stock attention and TriMul)", "Z"),
    "arm_u": Lever("arm_u", "fast_inference", "forward", "tier2", "ODDE_ARM_U=1", f"{AR}/odde_arm_t/__init__.py:648-666", "cli",
                   "arm U: bf16 FlashPairformer trunk + Z (U as shipped = U3: cast-once tri-attention prologue + fpf_transition_odde)", "U"),
    "arm_u23": Lever("arm_u23", "fast_inference", "forward", "tier2", "ODDE_ARM_U2_TRIMUL=1 / ODDE_ARM_U3=1 / ODDE_U_TRANSITION=1 (under ODDE_ARM_U)",
                     f"{AR}/odde_arm_t/__init__.py:55,58,59,667-684", "cli",
                     "U sub-arms: U2 = the arm's TriMul route: every TriangleMultiplicativeUpdate call the arm admits (design gate, scope, > 100 rows) handed to the core "
                     "TriMul provider by the tier word (lever trimul_core, levers/ARMT/odde_trimul_bind.py) at every row count -- the arm carries no TriMul kernel or tile table of its own "
                     "(a call the provider hands back runs the stock "
                     "TriMul by name, counted); U3 cast-once prologue + sep16 transition (on by default under U)", "U2/U3"),
    # --- levers/ACCEL: the hook (served levers) + the ACCEL_V2 levers -------------------------------------------------------------
    "served_levers_hook": Lever("served_levers_hook", "fast_inference", "serving", "exact", "ODDE_SERVED_LEVERS=1 (+ODDE_ADDON_LEVERS, ODDE_ARM_*)",
                                f"{AC}/sitecustomize.py:65-70; {AC}/odde_served_levers.py:154-158 arm_hook, :38-103 install_on_runner", "cli",
                                "wraps runner.batch_inference.get_default_runner and installs the DITFAST levers then the ARM-T arm on the returned runner's "
                                "model; one ACTIVE line per lever and a SUMMARY line; ODDE_SERVED_LEVERS_STRICT=1 raises on an install failure"),
    "dit_attn_bf16": Lever("dit_attn_bf16", "fast_inference", "forward", "tier2", "ODDE_DIT_ATTN=bf16", f"{AC}/odde_accel_v2.py:61-93", "cli",
                           "the 24 DiffusionTransformer AttentionPairBias cores run bf16 fused SDPA instead of the fp32 upcast", "T2"),
    # --- src: the fpf_engines op adapter (the exact line's two stock TriMul forwards bound in-process; its callables are odde_trimul_bind's) ------------
    "fpf_trimul_exact": Lever("fpf_trimul_exact", "fast_inference", "forward", "exact",
                              "FPF_ENGINE=opendde FPF_OPS=trimul_out,trimul_in FPF_IMPL=...odde_trimul_bind:trimul_out|in (fpf_engines.enable_from_env)",
                              f"{SR}/fpf_engines/__init__.py:234-250,295-318; {AR}/odde_trimul_bind.py:trimul_out", "cli",
                              "TriangleMultiplication{Outgoing,Incoming}.forward bound at class level by the fpf_engines adapter (trunk, MSA, template c_z 64, structure "
                              "refiner, confidence stacks): each call goes to the adapter's FPF_IMPL callables -- the core TriMul provider binding (lever trimul_exact names "
                              "the tier word); FPFFallback = the stock forward by name, counted per reason; enabled in-process by the package through the kit's own enable_from_env()", "S1"),
    # --- levers/XL: the XL add-on (single-card memory levers; the module installs at the model module's import through its shim) ---------
    "xl_tri_ln": Lever("xl_tri_ln", "fast_inference", "forward", "exact", "ODDE_XL=tri_ln", f"{XL}/odde_xl.py:22-85", "cli",
                       "TriangleAttention prologue: the LayerNorm'd full copy of the pair tensor is freed before the stock-chunked attention and the "
                       "LayerNorm recomputed per row-chunk; bias GEMM and q/k/v/o GEMMs keep stock shapes (inert below the stock chunk threshold). Bitwise to stock, but the "
                       "per-chunk LayerNorm costs trunk time — measured +1.0 s (+3 %) at 800 and +2.1 s (+2.4 %) at 1,200 tokens on H100, confidence head +0.1 s — so since "
                       "0.2.48 it rides no default line (a memory lever: a big candidate once the offload lines admit the XL unit); the exact line ran it until 0.2.47"),
    # --- levers/ARMT/odde_trimul_bind.py: the two TriMul routes (ARM U's U2 route; the exact line's FPF adapter) through the core's provider (opt_core.kernels.trimul), one opt-in word per tier ----
    "trimul_core": Lever("trimul_core", "fast_inference", "forward", "tier2", "ODDE_TRIMUL=fast", f"{AR}/odde_trimul_bind.py:serve; {AR}/odde_arm_t/__init__.py:make_trimul_forward", "cli",
                         "ARM U's U2 TriMul calls (every call the route admits: design gate, scope, > 100 rows) served by the core TriMul provider's fast tier (big: its "
                         "big tier) at EVERY row count: per (compute capability, precision, c_z, c_hidden, N bucket, direction) cell the measured winner of the core's table "
                         "(native / native_exact / tmk3_exact / tmk3_fast / esm_shapes ... as the cell says, named per bucket on the LEVER line); a measured stock winner, a "
                         "refusal by name or a row that cannot run is an Aside: that call runs the stock TriMul BY NAME, counted -- the kit re-routes nothing", "U2/core"),
    "trimul_exact": Lever("trimul_exact", "fast_inference", "forward", "exact", "ODDE_TRIMUL=exact (+FPF_IMPL=...odde_trimul_bind:trimul_out|in)", f"{AR}/odde_trimul_bind.py:trimul_out; {SR}/fpf_engines/__init__.py:267-292", "cli",
                          "the exact line's two bound stock TriMul forwards (fpf_engines, class level: trunk, MSA, template c_z 64, structure refiner, confidence stacks) served by the "
                          "core TriMul provider's exact tier at EVERY row count: rows bit-identical to the stock cuEquivariance TriMul on this stack only (native_exact / "
                          "tmk3_exact where the core's table vouches the cell on the running stack); an unvouched cell, a measured stock winner or a refusal by name is an "
                          "Aside = FPFFallback: the adapter runs the stock forward BY NAME, counted per reason on the LEVER line", "S1/core"),
    # --- levers/ARMT/odde_transition_bind.py: the pair-transition sites through the core's provider (opt_core.kernels.transition), one word per tier ----
    "transition_core": Lever("transition_core", "fast_inference", "forward", "tier2", "ODDE_TRANSITION=fast", f"{AR}/odde_transition_bind.py:decide; {AR}/third_party/fpf_transition_odde/__init__.py:forward_sep16", "cli",
                             "the trunk arm's pair_transition modules (c 384, n 4: the pairformer, MSA-module and template pair stacks and, under ODDE_ARM_T_SCOPE, the structural "
                             "refiner and the confidence head's stack) bound to the core transition provider's fast tier: per (compute capability, bf16, pair_c384_n4, N bucket) "
                             "cell a CARRIED row the cell names is served through the provider face; where the cell's measured winner is the statement itself (the stock arm: "
                             "9.0 and 8.0 today) or the word refuses by name above the table's sizes, the call is served by the kit's single-engine composition of that statement, "
                             "sep16 (third_party/fpf_transition_odde: LayerNorm cast once, the two bf16 projections, one fused SiLU*gate kernel with the statement's rounding points, "
                             "the output projection in place; bitwise to the engine module at the op, x1.16-1.34 per call on H100 / A100), named singleton=sep16:<n> on the LEVER line", "U3/core"),
    "transition_exact": Lever("transition_exact", "fast_inference", "forward", "exact", "ODDE_TRANSITION=exact", f"{AR}/odde_transition_bind.py:decide; {AR}/odde_arm_t/__init__.py:bind", "cli",
                              "the exact line's pair_transition modules (arm Z's scope) bound to the core transition provider's exact tier: a row serves only where the cell records it "
                              "bitwise to the statement on this stack; today the statement on both cards -> the kit's composition sep16 by name (bitwise to the engine module under the "
                              "model's bf16 autocast; an fp32 base without autocast keeps the module by name, counted)", "S1/core"),
    # --- opendde_opt/lncore.py: the pair-row LayerNorms through the core's LN provider (opt_core.kernels.ln); levers/ARMT/odde_triattn_bind.py: the arm's triangle-attention
    #     site through the core's provider (opt_core.kernels.triattn) -- one opt-in word per tier each ----
    "ln_core": Lever("ln_core", HOUSE, "forward", "tier2", "ODDE_LN=fast|big", "opendde_opt/lncore.py:make_forward", "cli",
                     "the pair-row LayerNorms (FusedLayerNorm.forward, C = 384 = c_z = c_s, weight and bias present, eager CUDA calls) served by the core LayerNorm provider's "
                     "tier word per (compute capability, dtype, cell family, row bucket) cell: the Triton rows fastln / fastln:lp / ln_rows at pair rows (x1.4-1.65 the "
                     "extension per call on H100, x1.2-1.5 on A100, N 400-1200); a cell whose winner is a stock row, another width (c=64 MSA, c=128 atom, c=768 DiT), "
                     "a weight-only module, a call under CUDA-graph capture (the step graph records the extension: lnstream), autograd, a refusal or a served-row error "
                     "= the extension fast_layer_norm_cuda_v2 by name, counted; tolerance class (max |d| vs the extension 1-2 bf16 ulp / ~1.4e-6 fp32); not on the exact "
                     "line (no provider row is bitwise the extension on this stack)", "LN"),
    "triattn_core": Lever("triattn_core", "fast_inference", "forward", "tier2", "ODDE_TRIATTN=fast|big", f"{AR}/odde_triattn_bind.py:serve; {AR}/odde_arm_t/__init__.py:make_attn", "cli",
                          "the arm's triangle-attention site (every pair stack it binds: trunk, MSA module, structural refiner, confidence head) served through the core's "
                          "provider (opt_core.kernels.triattn) by TIER WORD -- fast on LSTAR2A, big on the memory mode's lines: the word goes straight to the provider's "
                          "select(), which names the row per (compute capability, dtype, head dim, heads, row count, call form) cell of its measured table and carries the "
                          "cell's launch setting; a refusal serves the provider's named fallback row; a cell that names a stock row is the stock op's by name (asides=cell_stock) "
                          "-- the LEVER line names the row that served each cell; the kit carries no attention kernel, cell table or row preference of its own", "U/core"),
    "triattn_exact": Lever("triattn_exact", "fast_inference", "forward", "exact", "ODDE_TRIATTN=exact (+ODDE_ARM_T_SCOPE=all)", f"{AR}/odde_triattn_bind.py:serve; {AR}/odde_arm_t/__init__.py:install", "cli",
                           "the same site under arm Z through the provider's EXACT tier: only a row the provider's cell records as bitwise to the stock op on the running "
                           "card and stack serves (a row that refuses at call time is unavailable BY NAME, its named fallback serving); "
                           "the stock op by the cell's name everywhere else (asides=cell_stock); byte-identical to --mode off under --det 1", "Z/core"),
    "triattn_conf": Lever("triattn_conf", "fast_inference", "forward", "exact", "- (rides ODDE_TRIATTN on every line; ablation: ODDE_TRIATTN_CONF=stock)",
                          f"{AR}/odde_triattn_bind.py:CONF; {AR}/odde_arm_t/__init__.py:_enter_conf; {AR}/odde_arm_t/odde_conf_chunk.py:bind", "cli",
                          "the confidence head's 4-block pair stack served by the same tier word as every other pair stack (its rows and numerics class: triattn_exact's "
                          "bitwise rows on the exact line, triattn_core's on fast / big; upstream runs that head in fp32 below 2560 tokens, so its cells are the fp32 ones); "
                          "on every line since 0.2.57 -- where the provider's cell names the stock op that stack is the stock op's by the cell's name (conf_cells_stock); left "
                          "out by name (MODEL_OPT_LEVERS_OFF=triattn_conf) it keeps the stock op, counted asides=conf_stock; the mark sits on that stack's forward AND on its "
                          "triangle-attention modules (odde_conf_chunk), so where the offload unit's conf stage serves the stack in host-streamed row blocks (big above the "
                          "unit's size gate: block.tri_att_*.mha per block, never the stack's forward) every row block is served by the word under the same rule with that "
                          "block's mask slice and the full triangle bias -- census conf_path=offload_rows_core rows=<row>:<n> blocks=<n>; unbound there, inert by name (conf_path=offload_rows:<n>)"),
    # --- the memory mode's own levers (opendde_opt/big.py over opt_core.mem; tp_kernels.py / tp_struct.py on the row-sharded line): the big lines carry them ------------------------
    "tp_triatt": Lever("tp_triatt", HOUSE, "forward", "tier2", "- (on with line BIG_TP; the core's opt-out ROWPAIR_TRIATT_CORE=torch)", "opendde_opt/tp_kernels.py", "cli",
                       "the row-sharded pair stack's triangle attention on the core's row-block kernel dispatch (opt_core.mem.rowpair.triatt.attention_core "
                       "kernel=flash_triattn, ONE vocabulary with the core): the carried flash triangle-attention kernel per row batch; the engine's torch attention "
                       "statement is the dispatch's named per-call fallback (the core's LEVER name=F1.flash_triattn line: served / fallback / fallback_by; one "
                       "[opendde-opt tp] TRIATT line per rank; the manifest's tp_kernels block); a word the process cannot serve is refused by name at bind. "
                       "Not bitwise vs the torch statement (tier 2, run-to-run repeatable)"),
    "struct_pair_bf16": Lever("struct_pair_bf16", HOUSE, "forward", "tier2", "ODDE_TP_STRUCT_PAIR_DTYPE=bf16|fp32 (BIG_TP exports bf16 under --n_gpu P>1; a caller's fp32 wins, named)",
                              "opendde_opt/tp_struct.py", "cli",
                              "under `--mode big --n_gpu P` the STRUCTURAL pair track runs in bf16: this rank's z_struct row shard is stored bf16 after the expansion and the "
                              "4-block structural refiner (tri-mult, tri-attention, pair transition; the single track's pair bias) runs under bf16 autocast on it — the shard, the "
                              "tri-mult A block and every slab are half their fp32 bytes; the refined shard passes a finite gate (non-finite = refused by name) and every consumer "
                              "(diffusion conditioning rows, DiT pair bias rows) upcasts its rows to fp32 at the read, so the roll-out arithmetic is fp32 as on every line; "
                              "census struct_pair_dtype; tier 2 vs P=1 (equality within the seed band; outside the band the lever is refused by name); nothing at n_gpu=1. "
                              "Capacity: each rank holds the structural shard N_st^2*c_z/P plus one equally sized tri-mult A block while the refiner runs (N_st ~ 1.91 N, measured "
                              "1,956->3,748 and 6,426->12,282; fp32 = 2 x N_st^2 x 1536 B / P: 21.6 GB at 1,956 tokens x2, ~47 GB at 2,894 x2, 57.9 GB at 6,426 x8, on top of "
                              "the residue shard N^2*1536 B/P and the trunk residents -> fp32 fits about 2,500 tokens at P=2 and about 5,000 at P=8 on 80 GB cards; bf16 "
                              "halves both terms: measured bf16 shard 14.5 GB at 6,426 x8)"),
    "no_dit_hoist": Lever("no_dit_hoist", HOUSE, "forward", "exact", "- (size gate MODEL_OPT_BIG_NO_DIT_HOIST_MIN_TOKENS, configs/<gpu>.env: 2565 as shipped; decided by the package from the query)", "opendde_opt/big.py", "cli",
                          "the DITFAST hoist (`dit_hoist` + `dit_align`) left out of a big line: ODDE_ADDON_LEVERS not exported, so the 24 resident [1,16,n_s,n_s] fp32 "
                          "hoisted pair biases of the diffusion transformer (24 x 16 x n_s^2 x 4 B: 39 GB at 2,565 residues) are recomputed per block per step as stock does — "
                          "device memory for wall time; SIZE-GATED: on when the largest input item counts >= the gate in residue tokens, off below (`big.plan`, recorded as "
                          "`kit.big.no_dit_hoist_policy`; its LEVER row `state=off reason=below_gate:<tokens>/<gate>`); exact class (the stock op per step; line BIG_B = fast + this lever is bitwise to fast "
                          "under the recipe); opt_core strategy F7.hoist_off"),
    "sample_chunk": Lever("sample_chunk", HOUSE, "forward", "tier2", "- (samples=1; follows the offload size gate MODEL_OPT_BIG_OFFLOAD_MIN_TOKENS on BIG_F)", "opendde_opt/big.py", "cli",
                          "the diffusion samples run one chunk at a time: `configs.infer_setting.sample_diffusion_chunk_size` (stock 5 = every sample in one batch) set to "
                          "`samples` for each `OpenDDE.run_sample_diffusion_stage` call and restored after it; N_sample <= samples is a named skip; the per-chunk noise "
                          "draws come in a different order than one batch's (tier 2 by construction); on the offload line it follows pair_offload's size gate (off with the "
                          "unit below MODEL_OPT_BIG_OFFLOAD_MIN_TOKENS: fast's one-batch sampler runs; `big.GATE_FOLLOWERS`); opt_core strategy F7.chunked_eval"),
    # --- levers/OFFLOAD: the offload unit (the big mode; the unit's shim installs at interpreter start when ODDE_OFFLOAD is set; SIZE-GATED on
    #     BIG_F: installed when the largest input item counts >= MODEL_OPT_BIG_OFFLOAD_MIN_TOKENS residue tokens (configs/<gpu>.env), below it the
    #     switches are not exported, the unit's directory is off the path and the pair tensors stay resident as on fast — `big.plan`, recorded as
    #     `kit.big.pair_offload_policy`; each row's LEVER line then reads `state=off reason=below_gate:<tokens>/<gate>`) -------
    "pair_offload_struct": Lever("pair_offload_struct", "fast_inference", "forward", "tier2", "ODDE_OFFLOAD=struct", f"{OF}/odde_offload.py:826-914", "cli",
                                 "the structural token expander, refiner pair stack and diffusion pair cache run on a host-resident z_struct "
                                 "streamed in row / column blocks (TriMul hidden channels chunked to fit ODDE_OFFLOAD_MEMFRAC of free GPU memory); "
                                 "a Fold-CP mesh or a distributed pair spec on the stage is refused by name"),
    "pair_offload_trunk": Lever("pair_offload_trunk", "fast_inference", "forward", "tier2", "ODDE_OFFLOAD=trunk", f"{OF}/odde_offload_trunk.py:635-700", "cli",
                                "the trunk's z (init, recycling, templates, MSA module, pairformer stack) host-resident and streamed (stage-method "
                                "patches: OpenDDE.forward stays upstream's, so its TF32 scope, lazy relative-position features and bounded chunk "
                                "size reach the stage through its arguments); relp rows materialised per row block"),
    "pair_offload_conf": Lever("pair_offload_conf", "fast_inference", "forward", "tier2", "ODDE_OFFLOAD=conf", f"{OF}/odde_offload_trunk.py:528-633,676-700", "cli",
                               "the confidence head's pair stack and the distogram contact probabilities host-resident and streamed (per-sample "
                               "outputs retained as upstream's ConfidenceHead does: PAE/PDE logits to the CPU when N_sample > 1)"),
    "diffz": Lever("diffz", "fast_inference", "forward", "tier2", "ODDE_OFFLOAD_DIFFZ=1 (+_PERM, _RELEASE)", f"{OF}/odde_offload.py:707-815", "cli",
                   "diffusion: no per-step pair_z clone, LayerNorm(pair_z) memoised across the 200 steps as one permuted image and the GPU pair_z "
                   "released once it exists; designed bit-preserving by the unit, tested only inside its arms"),
    "bigln_guard": Lever("bigln_guard", "fast_inference", "forward", "exact", "ODDE_OFFLOAD_BIGLN_LIMIT=2147483648 (with diffz)", f"{OF}/odde_offload.py:666-704", "cli",
                         "torch.nn.functional.layer_norm blocked below 2^31 elements per call: torch 2.7.1's CUDA LayerNorm returns wrong rows past "
                         "element index 2^32 (n_struct >= 5,793); inert below the limit (the stock call); blocked == unblocked bitwise"),
    "free_templ": Lever("free_templ", "fast_inference", "forward", "tier2", "ODDE_OFFLOAD_FREE_TEMPL=1", f"{OF}/odde_offload.py:843-850", "cli",
                        "template pair features dropped from the feature dict after the trunk (read by nothing downstream: memory only, "
                        "bit-preserving by construction); tested only inside the unit's arms — no house row isolates it"),
    # --- levers/SAMPLER: the SAMPLER unit (the sampler's two attention sites through the core pair-bias-attention provider BY WORD -- levers/SAMPLER/odde_apb_bind.py:
    #     opt_core.kernels.apb select() per call class, no kernel / cell table / row import in the kit -- + the fused sampler schedules third_party/opendde_fpf_ditfast; routed from the
    #     ACCEL lever's DiT-attention site levers/ACCEL/odde_accel_v2.py install_dit_attn -> levers/SAMPLER/odde_sampler.install after the DITFAST hoist; every install error raises by name)
    "dit_attn_apb": Lever("dit_attn_apb", "fast_inference", "forward", "tier2", "ODDE_DIT_ATTN=fast", f"{SM}/odde_sampler.py:54-60; {SM}/odde_apb_bind.py:install_dit,dit_views,dit_packed", "cli",
                          "the 24 DiffusionTransformer token-attention modules' primitives.Attention.forward replaced per instance (and the fused token stack's attention, dit_fused) by the core "
                          "provider's pair-bias attention opt_core.kernels.apb served BY TIER WORD (fast on the fast line, big on the big lines; a row[:variant] word pins one row): per call "
                          "class (card, dtype, 16 heads x 48, sample count, token bucket, eager | graph) the provider's cell names the fastest measured row -- the fused Triton flash kernel with "
                          "the additive pair bias shared by the diffusion samples in-kernel (fpf_apb; its fp16 / bf16 tensor-core operand forms are the cell's choice), the block-diagonal packed "
                          "kernel (apb_attn) or a stock op by name where that is the measured winner; q|k|v|g consumed as strided views of stock's projections, sigmoid gate fused; replaces the stock "
                          "fp32-upcast SDPA statement and the bf16 recast (dit_attn_bf16) on the fast lines; tolerance class (reduction order, 16-bit operands)", "SP1"),
    "atom_attn_apb": Lever("atom_attn_apb", "fast_inference", "forward", "tier2", "ODDE_ATOM_ATTN=fast", f"{SM}/odde_sampler.py:57-59; {SM}/odde_apb_bind.py:install_atom,atom_views", "cli",
                           "the 3+3 atom-transformer attention modules' primitives.Attention.forward replaced per instance (and the fused atom stacks' attention, atom_fused) by the core provider's "
                           "windowed pair-bias attention opt_core.kernels.apb served BY TIER WORD (cell: 4 heads x 32, 32-query x 128-key windows): ONE Triton launch per call for the local "
                           "windows (fpf_atom: in-kernel window gather replaces primitives._local_attention's pad / unfold / permute / SDPA chain, tf32 operands, sigmoid gate fused; the hoisted "
                           "trunked pair bias consumed in its own strides) or the row the cell names; tolerance class", "SP2"),
    "cond_dedupe": Lever("cond_dedupe", "fast_inference", "forward", "tier2", "ODDE_COND_DEDUPE=1", f"{SM}/odde_sampler.py:68; {SM}/third_party/opendde_fpf_ditfast/cond_dedupe.py:22-37", "cli",
                         "DiffusionConditioning.forward (diffusion.py:1054) computed ONCE per step for the sample-invariant noise level (upstream expands one t_hat over "
                         "N_sample: stride-0 guard per call, the stock path otherwise) and the single conditioning returned as a stride-0 expand over the samples; the "
                         "conditioning GEMMs run with N rows instead of N_sample*N (tolerance class: GEMM M changes); inside a capturing caller (stepgraph: the step's arguments "
                         "cloned contiguous) the deduplicated path is kept once the expanded form was seen, so the captured step computes what the eager step computes", "SP3"),
    "dit_fused": Lever("dit_fused", "fast_inference", "forward", "tier2", "ODDE_DIT_FUSED=1", f"{SM}/odde_sampler.py:70; {SM}/third_party/opendde_fpf_ditfast/dit_fast.py:209-238", "cli",
                       "the sampler's 24-block token DiffusionTransformer as ONE fused forward on packed weights: per block an AdaLN row kernel, one q|k|v|g GEMM, the "
                       "dit_attn_apb kernel (required by name), o GEMM, one [adaLN-zero gate + residual + transition AdaLN] kernel, one a1|a2 GEMM, SwiGLU kernel, b GEMM, "
                       "gate+residual kernel; per-block conditioning and the 24 pair biases (+ OpenDDE's structural extra bias) produced once per item through the DITFAST "
                       "hoist's slots; fp32 residual stream (autocast disabled inside the stack: the house lever sampler_amp governs the sampler around it, not its "
                       "precision plan); tolerance class (GEMM shapes change)", "SP4"),
    "dit_lowp": Lever("dit_lowp", "fast_inference", "forward", "tier2", "ODDE_DIT_LOWP=fp16", f"{SM}/odde_sampler.py:71; {SM}/third_party/opendde_fpf_ditfast/dit_fast.py:219-222", "cli",
                      "precision word of dit_fused: the a-path GEMM operands and the attention operands in fp16 (bf16 selectable) with fp32 accumulation, epilogues and "
                      "residual stream fp32; requires dit_fused (refused by name without it)", "SP4p"),
    "atom_fused": Lever("atom_fused", "fast_inference", "forward", "tier2", "ODDE_ATOM_FUSED=1", f"{SM}/odde_sampler.py:73; {SM}/third_party/opendde_fpf_ditfast/atom_fast.py:178-206", "cli",
                        "both 3-block atom transformers (AtomAttentionEncoder / Decoder .atom_transformer) as fused stacks: entry AdaLN kernel, one q|g and one k|v GEMM per "
                        "block, the atom_attn_apb kernel (required by name), o GEMM, gate+residual+AdaLN kernel, one a1|a2 GEMM, SwiGLU kernel, b GEMM; per-block conditioning "
                        "and local pair biases produced once per item through the hoist's slots; tolerance class", "SP5"),
    "dit_attn_exact": Lever("dit_attn_exact", "fast_inference", "forward", "exact", "ODDE_DIT_ATTN=exact", f"{SM}/odde_sampler.py:54-56; {SM}/odde_apb_bind.py:install_exact", "cli",
                            "primitives._attention's fp32 SDPA statement at the sampler's [S,16,N,48] + [1,16,N,N]-bias shape served through the core provider's EXACT tier (opt_core.kernels.apb "
                            "word exact; stock's decomposition kept: q pre-scaled, fp32 upcast, SDPA scale 1): the provider's bit-exact prebuilt CUDA kernel row (dit_exact, reproducing the "
                            "memory-efficient SDPA kernel's operation order) where the provider VOUCHES it on the running card / stack -- this kit's torch2.7.1-cu126 sm_90 stack -- with a "
                            "two-case torch.equal probe against torch SDPA in every process before the bind; on a card / stack where the provider's exact tier names the stock op (sm_80: no "
                            "kernel row) the lever STEPS ASIDE BY NAME -- LEVER state=off reason=aside:provider_exact_names_stock:<row>@<card>, nothing installed, the stock statement serves; "
                            "other shapes, dtypes and call forms take the stock statement, counted by reason; exact class (bitwise vs stock); ODDE_DIT_ATTN_EXACT=1 is the pre-0.2.57 spelling", "SP6"),
}

# kit knobs: every switch the kit reads that is not a lever, with the kit's default; never exported by a house mode unless in the line
KNOBS: dict[str, tuple[str, str, str]] = {   # name -> (default, file:line, meaning)
    "ODDE_STEPGRAPH_MAX_TOKENS": ("unset (stepgraph.GATE_MAX)", "opendde_opt/stepgraph.py:69", "the sampler step-graph's upper size gate in residue tokens of the query (composed out above it: LEVER row reason above_gate:<t>/<g>)"),
    "ODDE_STEPGRAPH_MIN_TOKENS": ("unset (stepgraph.GATE_MIN)", "opendde_opt/stepgraph.py:69", "the sampler step-graph's lower size gate in residue tokens (composed out below it: below_gate:<t>/<g>)"),
    "ODDE_STEPGRAPH_HEAD": ("1", "opendde_opt/stepgraph.py:348", "eager denoiser calls before the capture (the DITFAST hoist records on call 1)"),
    "ODDE_STEPGRAPH_RTOL": ("1e-4", "opendde_opt/stepgraph.py:484", "without --det: the first replay on new inputs held to an eager re-run within this fraction of max|x| (the eager denoiser is not run-to-run reproducible there); under --det the hold is bit-exact"),
    "ODDE_STEPGRAPH_MEMFRAC": ("0.5", "opendde_opt/stepgraph.py:406", "memory admission: the estimated step arena may take at most this fraction of free device memory, else refused by name"),
    "ODDE_STEPGRAPH_VERBOSE": ("0", "opendde_opt/stepgraph.py:286", "the capture cache's per-key words on stderr"),
    "MODEL_OPT_SAMPLER_AMP_MAX_TOKENS": ("3840 (precision.GATE_MAX = upstream's own AMP size)", "opendde_opt/precision.py", "the sampler-autocast word's upper size gate in tokens (upstream's n_token at its policy site): an item of at most this many tokens runs its diffusion sampler under bf16 autocast (sampler_amp), above it upstream's fp32 sampler stands (LEVER evidence above_gate=); the per-card retune point (as shipped no kit gate: neutral at 1,200 residue tokens with the step graph, wins below); 0 = never engaged"),
    "MODEL_OPT_LEVERS_OFF": ("unset (nothing ablated)", "opendde_opt/ablate.py:23", "ablation: comma-separated levers composed OUT of the resolved line by name (modes.line_without); a name the line does not carry or cannot shed is refused by name (exit 3, the reason on the refusal line)"),
    "ODDE_LNSTREAM_PREBUILT": ("1 (load the shipped binary)", "opendde_opt/lnstream.py:58", "maintainer switch: 0 = never load lnstream's shipped extension binary (prebuilt/<stack key>/); the JIT build serves and the LEVER row says prebuilt=prebuilt_off"),
    "ODDE_LN": ("unset (fast: fast; big: big)", "opendde_opt/lncore.py:38", "the core LayerNorm provider's tier word for the pair-row LayerNorms (fast | big | exact | a provider row name); unset = every LayerNorm on the extension (lever ln_core off)"),
    "ODDE_SERVED_LEVERS_STRICT": ("0 (house arms: 1)", f"{AC}/odde_served_levers.py:58,73,83,126", "install failures raise instead of falling back to stock"),
    "ODDE_SERVED_LEVERS_REPORT": ("unset (the package never sets it: no report file)", f"{AC}/odde_served_levers.py:32", "JSON copy of the ACTIVE/SUMMARY records"),
    "ODDE_ARM_ZT": ("1", f"{AR}/odde_arm_t/__init__.py:56", "transpose-free pair update inside every installed arm"),
    "ODDE_ARM_U3": ("1", f"{AR}/odde_arm_t/__init__.py:58", "U3 prologue under U"),
    "ODDE_U_TRANSITION": ("1", f"{AR}/odde_arm_t/__init__.py:59", "under U3 the pair-transition binding may apply (=0: the module stays under U); the binding itself needs ODDE_TRANSITION"),
    "ODDE_ARM_T_SCOPE": ("trunk", f"{AR}/odde_arm_t/__init__.py:54", "modules the arm binds: trunk | trunk+refiner | trunk+confidence | all (ERRATA_06)"),
    "ODDE_ARM_T_MIN_TOKENS": ("300", f"{AR}/odde_arm_t/__init__.py:50", "below this token count the arm stays stock"),
    "ODDE_ARM_T_TRIMUL": ("fast", f"{AR}/odde_arm_t/__init__.py:52", "the arm's TriMul site (fast = bound: the provider route | stock = left unbound)"),
    "ODDE_SAMPLER_PROBE": ("unset (off)", f"{SM}/odde_sampler.py:150-190", "test hook: one census line per call signature of the sampler sites (shapes / dtypes / strides); never exported by a mode"),
    "ODDE_ADDON_QUIET": ("0", f"{DT}/odde_addon.py:38", "silence the add-on's lines"),
    "FPF_CROSSCHECK": ("0", f"{SR}/fpf_engines/__init__.py:18,317", "the first k calls per op run the stock method beside the kernel and record the delta"),
    "MODEL_OPT_BIG_OFFLOAD_MIN_TOKENS": ("1400", "opendde_opt/big.py", "the size gate of BIG_F's offload unit (pair_offload, and sample_chunk with it) in residue tokens: installed at >= the gate, off below — the line then runs fast's resident lever set (0 = on at every size); a value set in the environment wins (configs/h100.env restates this default); unset, the package sizes it by device 0's total memory (big.gate_default): 1160 under 64 GiB — the 40 GB A100 — 1400 otherwise"),
    "MODEL_OPT_BIG_NO_DIT_HOIST_MIN_TOKENS": ("2565", "opendde_opt/big.py", "the size gate of the big lines' no_dit_hoist lever in residue tokens: on at >= the gate, off below (0 = on at every size); a value set in the environment wins (configs/h100.env restates this default); unset, the package sizes it by device 0's total memory (big.gate_default): 1856 under 64 GiB — the 40 GB A100 — 2565 otherwise"),
    "MODEL_OPT_SMALL_INPUT_FLOOR_TOKENS": ("300", "opendde_opt/smalln.py", "the small-input floor of the exact and fast lines in residue tokens: when every item of a pred call is below it the trunk levers (arm_z | arm_u + arm_u23, fpf_trimul_exact, xl_tri_ln) and fast's dit_attn_bf16 are composed out of the line (LEVER state=off reason=below_gate:<tokens>/<gate>); any item at or above it keeps every lever bound; 0 = bound at every size; 300 = ARM's design gate (ODDE_ARM_T_MIN_TOKENS); configs/<gpu>.env"),
    "ODDE_XL_STRICT": ("1", f"{XL}/sitecustomize.py:39", "an XL install exception raises instead of continuing stock (the house lines export 1)"),
    "ODDE_XL_CROSSCHECK": ("0", f"{XL}/odde_xl.py:40", "tri_ln computes the stock LayerNorm beside the lever on every call and asserts bitwise (test hook)"),
    "ODDE_XL_PROBE": ("0", f"{XL}/sitecustomize.py:9", "per-stage telemetry through odde_xl_probe.py (not carried; absent on every house line)"),
    "ODDE_OFFLOAD_STRICT": ("1", f"{OF}/sitecustomize.py:71-77", "an install failure of the unit ends the interpreter with NOT ACTIVE, exit 3 (0 = one stderr line, the process continues unpatched)"),
    "ODDE_OFFLOAD_ROWS": ("256", f"{OF}/odde_offload.py:34", "row block for projections, transitions, the expander and bias passes"),
    "ODDE_OFFLOAD_CC": ("auto", f"{OF}/odde_offload.py:35", "TriMul hidden-channel chunk (auto = the largest that fits ODDE_OFFLOAD_MEMFRAC)"),
    "ODDE_OFFLOAD_MEMFRAC": ("0.80", f"{OF}/odde_offload.py:36", "fraction of the free GPU memory the streamed TriMul operands may use"),
    "ODDE_OFFLOAD_XBUF": ("host", f"{OF}/odde_offload.py:37", "the chunked-TriMul product buffer: host (pinned) | disk:<dir>"),
    "ODDE_OFFLOAD_PIN": ("1", f"{OF}/odde_offload.py:41", "the offload unit's host buffers (pair tensors, column staging, the buffer pool) in pinned memory; 0 = pageable host buffers. A pinned allocation the driver refuses is an error naming this switch, never an automatic pageable route"),
    "ODDE_OFFLOAD_PITCHED": ("1", f"{OF}/odde_offload.py:259", "column blocks by pitched cudaMemcpy2D"),
    "ODDE_OFFLOAD_DIFFZ_ROWS": ("0", f"{OF}/odde_offload.py:736", "diffz: rows per block of the memoised LayerNorm image (0 = auto)"),
    "ODDE_OFFLOAD_DCHUNK": ("0", f"{OF}/odde_offload.py:61", "retired (upstream bounds the dynamic chunk size): unset or 0 accepted, any other value refused by name"),
    "ODDE_OFFLOAD_FORCE_CHUNK": ("0", f"{OF}/odde_offload.py:62", "retired (upstream clamps a forced chunk size): unset or 0 accepted, any other value refused by name"),
    "ODDE_OFFLOAD_ZINIT": ("recompute", f"{OF}/odde_offload_trunk.py:50", "trunk z init per row block: recompute | host"),
    "ODDE_OFFLOAD_PWA_W": ("full", f"{OF}/odde_offload_trunk.py:53", "MSA pair-weighted averaging weights: full | rows"),
    "ODDE_OFFLOAD_OPM_ROWS": ("0", f"{OF}/odde_offload_trunk.py:55", "outer-product-mean row block (0 = ODDE_OFFLOAD_ROWS)"),
    "ODDE_OFFLOAD_DISTO": ("rows", f"{OF}/odde_offload_trunk.py:58", "distogram contact probabilities: rows (never materialised) | full"),
    "ODDE_OFFLOAD_LAZY_RELP": ("1", f"{OF}/odde_offload_trunk.py:65", "retired (upstream's relp is always lazy; rows materialised per block): unset or 1 accepted, 0 refused by name"),
    "ODDE_OFFLOAD_CYCLE_CKPT": ("1", f"{OF}/odde_offload_trunk.py:60", "per-cycle trunk checkpoint every k cycles WHEN ODDE_OFFLOAD_CKPT_DIR is set (robustness; no house line sets a dir)"),
    "ODDE_OFFLOAD_CKPT_DIR": ("", f"{OF}/odde_offload.py:38", "stage checkpoints (trunk.pt / refiner.pt / trunk_cycle.pt) for resume; absent on every house line"),
    "ODDE_OFFLOAD_RESUME": ("", f"{OF}/odde_offload.py:39", "resume from a stage checkpoint: trunk | refiner; absent on every house line"),
    "ODDE_OFFLOAD_CKPT_MAX_GB": ("80", f"{OF}/odde_offload.py:40", "largest z the trunk checkpoint writes"),
    "ODDE_OFFLOAD_LOG": ("", f"{OF}/odde_offload.py:42", "the unit's per-block log: absent = quiet (as shipped, ERRATA_01); `-` = stdout; a path = that file and stdout"),
    "ODDE_TRAJ_EVERY": ("0", f"{OF}/sitecustomize.py:78-90", "the unit's diffusion-trajectory hook (odde_traj_hook.py, not carried); absent on every house line"),
    "FPF_STRICT": ("0", f"{SR}/fpf_engines/__init__.py:21,314", "a kernel exception raises instead of falling back to stock"),
}


# ------------------------------------------------------------------------------------------------------------ testing per stock pin
# A lever's bytes bind upstream symbols by name (classes re-bound, functions wrapped, runners hooked): a lever is *tested* on an
# opendde version when its binds and its guarantee (tier) were established against that version's symbols. A line whose levers are
# not all tested on the INSTALLED opendde refuses by name — `NOT ACTIVE reason=pending_rebase … levers_untested=<names>`, exit 3
# (stack.pending_rebase_reason; `check` reports it) — never a silent stock run under a kit banner. This dict is the one place that
# fact lives: testing a lever on a pin = adding its name here; `superseded_by_upstream` levers leave their lines instead (LEVERS row).
PIN = "1.1.1"                                     # the tree's stock pin (stock/PINS.json "opendde_version"; tests/test_registry_kits.py locks the two)
STATUSES = ("tested", "pending", "superseded_by_upstream", "retired", "refused")   # refused = not carried on this pin for cause (its reason names the defect and the re-admission test)
UP = "stock/src"                                  # upstream citations below are paths under opendde/stock/src (the pin's source snapshot)
# status of every LEVERS entry on the pin, with its one-line reason (upstream file:line where upstream is the reason):
#   tested               binds established against the pin's symbols (its lines may activate)
#   pending                 not yet rebased/validated on the pin: every line carrying it refuses by name (reason=pending_rebase)
#   superseded_by_upstream  upstream does it itself on this pin: the row stays, no line carries it (modes.LINES), its switch is never exported
#   retired                 conflicts with the pin by construction: no line carries it, its switch is never exported
PIN_STATUS: dict[str, tuple[str, str]] = {
    "cueq_tuned_cache":   ("tested", "not an opendde bind: the cuEquivariance-triton tuning cache location (cuequivariance-ops-torch-cu12 0.10.0 reads CUEQ_TRITON_CACHE_DIR, same key)"),
    "drop_bond_mask":     ("tested", f"the featurizer key and site unchanged ({UP}/opendde/data/core/featurizer.py)"),
    "lnstream":           ("tested", f"binds layer_norm._load_fast_layer_norm_cuda_v2 / fast_layer_norm_cuda_v2 ({UP}/opendde/model/layer_norm/layer_norm.py:42-108,277-283); "
                                      f"the four kernel sources pinned by sha256 ({UP}/opendde/model/layer_norm/kernel/, opendde_opt/lnstream.py PINS)"),
    "ln_core":            ("tested", f"binds FusedLayerNorm.forward ({UP}/opendde/model/layer_norm/layer_norm.py:233-290): the module's normalized_shape / weight / bias / eps attributes and the "
                                      f"forward(input) signature unchanged on the pin; the extension path (FusedLayerNormAffineFunction) is the wrapped original"),
    "sched_host":         ("tested", f"binds InferenceNoiseScheduler.__call__ and the generator module's centre_random_augmentation name ({UP}/opendde/model/generator.py:40-66,10,186-207; "
                                      f"{UP}/opendde/model/utils.py:16-75,90-104)"),
    "zprep_hoist":        ("tested", f"binds the diffusion module's permute_final_dims name ({UP}/opendde/model/modules/diffusion.py:37,1511-1512)"),
    "structok_sync":      ("tested", f"binds StructuralTokenExpander._pair_project_by_role_full / _full_chunk / _full_tile ({UP}/opendde/model/modules/structural_tokens.py:260-353)"),
    "tmpl_dedup":         ("tested", f"binds TemplateEmbedder.forward ({UP}/opendde/model/modules/pairformer.py:2135-2193; single_template_forward :2599-2657; "
                                      f"template_embedder.n_blocks=2 for opendde_v1: {UP}/opendde/config/model_base.py:140-143; dummy slots: {UP}/opendde/data/inference/infer_dataloader.py:269)"),
    "keep_pool":          ("tested", f"the in-forward empty_cache sites unchanged ({UP}/opendde/model/opendde.py:580,1598; {UP}/opendde/model/modules/confidence.py:298,553,813 "
                                      f"through {UP}/opendde/utils/torch_utils.py cleanup_device_memory; runner sites {UP}/runner/inference.py:188,775,1653,1870 pass)"),
    "writer_overlap":     ("tested", f"binds DataDumper.dump ({UP}/runner/dumper.py:93-123; dump_predictions :137-204 reads pred_dict['coordinate'|'summary_confidence'|'full_data'] "
                                      f"through device-independent conversions: dumper.py:34-45,171-177, {UP}/opendde/utils/torch_utils.py:135-148, {UP}/opendde/data/utils.py:528) "
                                      f"and InferenceRunner.close ({UP}/runner/inference.py:1190-1232, run by main() :1900-1911 in `finally`); the call site {UP}/runner/inference.py:1790-1823"),
    "json_oneshot":       ("tested", f"binds the name runner.dumper.save_json ({UP}/runner/dumper.py:18 imports it from {UP}/opendde/utils/file_io.py:211-218; called at "
                                      f"dumper.py:326,332 for the summary_confidence / full_data documents) with opendde.utils.file_io.map_values_to_list ({UP}/opendde/utils/file_io.py:186)"),
    "prefetch":           ("tested", f"binds {UP}/runner/inference.py:291 _create_inference_dataloader_synchronized (configs.num_workers read at "
                                      f"{UP}/opendde/data/inference/infer_dataloader.py:148; upstream default 0 at {UP}/opendde/config/inference_defaults.py:22)"),
    "stepgraph":          ("tested", f"wraps the module-global sampler name the model resolves ({UP}/opendde/model/opendde.py:36-38,1332-1334 -> generator.sample_diffusion) and "
                                      f"routes DiffusionModule.__call__ ({UP}/opendde/model/modules/diffusion.py); the denoiser's keyword signature ({UP}/opendde/model/generator.py:262-278)"),
    "sampler_amp":        ("tested", f"re-binds the module-global policy function runner.inference.update_inference_configs ({UP}/runner/inference.py:1489-1516), which "
                                      f"_prepare_prediction_batch resolves by name at call time ({UP}/runner/inference.py:1545); the model reads configs.skip_amp.sample_diffusion per "
                                      f"sampler call ({UP}/opendde/model/opendde.py:1332)"),
    "chunk_lift":         ("tested", f"binds OpenDDE._resolve_pairformer_chunk_size and OpenDDE._main_inference_loop ({UP}/opendde/model/opendde.py:1796-1817,1838-1932: the two "
                                    "resolutions per item, the table of config/model_base.py:44-51 and the 450e6 clamp of :1819-1836 called, not transcribed); engages where the clamp "
                                    "is below the table and the site's rule lifts it: the residue site only to the table's un-chunked value (663 <= N <= 1,024; above_gate beyond), the structural site to "
                                    "chunked values too (N_st = 663..2560, i.e. N ~ 345..1,340); composed into a pred call only when every item counts <= 1,024 residue tokens (chunklift.plan / compose: "
                                    "above_gate:<tokens>/1024 otherwise, the model class untouched)"),
    "rowpair_tp":         ("tested", f"binds PairformerStack.forward and PairformerBlock.forward_source ({UP}/opendde/model/modules/pairformer.py:325,602-632: the stack reaches blocks "
                                        "through forward_source on this pin)"),
    "alloc_auto":         ("tested", "the allocator setting of the kit process; no upstream bind"),
    "dit_hoist":          ("tested", f"binds the diffusion module INSTANCE's conditioning producers (reached through forward_source unchanged); ERRATA_04: a Fold-CP/atom-window "
                                        f"sample_diffusion call ({UP}/opendde/model/generator.py:96-99) bypasses the hoist by name"),
    "dit_align":          ("tested", "as dit_hoist (the hoist's slot alignment)"),
    "arm_z":              ("tested", f"ERRATA_04: the transpose-free block binds PairformerBlock.forward_source ({UP}/opendde/model/modules/pairformer.py:325-415, the 1.0.0 forward body; "
                                        ":602-632 the stack calls it) — binding .forward alone is bypassed on this pin"),
    "arm_u":              ("tested", f"TriangleAttention / TriangleMultiplication binds bytes-identical to 1.0.0 ({UP}/opendde/model/triangular/triangular.py:477-494) + the bf16 pair-stack bind on PairformerStack.forward (entered per stack on this pin); Z component as arm_z"),
    "arm_u23":            ("tested", "as arm_u"),
    "served_levers_hook": ("tested", f"wraps runner.batch_inference.get_default_runner by module-global name ({UP}/runner/batch_inference.py:365; runner.cli imports runner.batch_inference)"),
    "dit_attn_bf16":      ("tested", "instance-level replacement of the 24 DiT attention modules; counts its bf16 calls"),
    "fpf_trimul_exact":   ("tested", f"binds TriangleMultiplication{{Outgoing,Incoming}}.forward through fpf_engines; the cuEquivariance-path body is bytes-identical to 1.0.0; "
                                        "fused_layer_norm_torch.layer_norm_transpose (cuequivariance-ops-torch 0.10.0) keeps its signature; equality re-taken on the pin"),
    "xl_tri_ln":          ("tested", f"binds TriangleAttention.forward/_chunk ({UP}/opendde/model/triangular/triangular.py:617-701, unchanged); engages from N >= 663 on this pin (chunked trunk attention)"),
    "tp_triatt":          ("tested", "house lever of the row-sharded line: the core's row-block dispatch bound at the kit's own tri-attention seam (opendde_opt/tp.py _triatt_fns); no upstream bind"),
    "trimul_core":        ("tested", "as arm_u23: the same TriangleMultiplicativeUpdate.forward bind (odde_arm_t make_trimul_forward), the provider branch inside its U2 route; no new upstream bind"),
    "transition_core":    ("tested", "an instance-level bind of Transition.forward on the pair_transition modules the arm's scope names (odde_arm_t bind -> third_party/fpf_transition_odde apply; opendde/model/modules/primitives.py Transition: c_in, linear_no_bias_a|b, linear_no_bias, layernorm1 unchanged 1.0.0 .. 1.1.1); the provider's decision is pure (levers/ARMT/odde_transition_bind.py decide); sep16's bitwise claim is tests/test_transition_bind.py (GPU) and the kit's --det 1 identity"),
    "transition_exact":   ("tested", "as transition_core: the same instance bind under arm Z's scope with the provider's exact word"),
    "trimul_exact":       ("tested", "as fpf_trimul_exact: the same two stock forwards bound by fpf_engines, the adapter's FPF_IMPL callables in levers/ARMT/odde_trimul_bind.py; the rows' bitwise claim is the provider's per-call equality (tests/test_trimul_bind.py, GPU) and the kit's --det 1 identity"),
    "triattn_core":       ("tested", "as arm_u: the same attention wrapper (layers.cuequivariance_triangular_attn re-bound), the provider branch inside it; no new upstream bind"),
    "triattn_conf":       ("tested", "as triattn_core / triattn_exact (the same wrapper); the confidence head's pair stack is marked by forward hooks on model.confidence_head.pairformer_stack"),
    "triattn_exact":      ("tested", "as arm_u's attention bind under arm Z (layers.cuequivariance_triangular_attn re-bound); the rows' bitwise claim is the provider's per-call equality (tests/test_triattn_bind.py, GPU) and the kit's --det 1 identity"),
    "struct_pair_bf16":   ("tested", "house lever of the row-sharded line: a dtype of the kit's own structural shard + autocast around the kit-driven refiner; no upstream bind "
                                        "(its acceptance = equality vs P=1 exact within the tier-2 band on the equality panel; outside the band it is refused by name)"),
    "no_dit_hoist":       ("tested", "the DITFAST hoist left out at >= the size gate; house flag, no upstream bind"),
    "sample_chunk":       ("tested", f"sets configs.infer_setting.sample_diffusion_chunk_size, which upstream honours ({UP}/opendde/model/opendde.py:1259)"),
    "pair_offload_struct": ("tested", f"ERRATA_05: the 'struct' stage replacement of OpenDDE.expand_to_structural_tokens accepts the pin's lazy_relp ({UP}/opendde/model/opendde.py:422-431, "
                                         ":1906) and returns the diffusion pair cache with the stock single-device keys; counters calls_struct, calls_diff_cache"),
    "pair_offload_trunk": ("tested", f"ERRATA_05: install_trunk patches the stage methods only (get_pairformer_output, run_confidence_head, compute_distogram_contact_probs); "
                                        f"OpenDDE.forward / _forward_impl are upstream's, so the TF32 scope ({UP}/opendde/model/opendde.py:68-80), the lazy relp and the bounded chunk "
                                        "size reach the offloaded stages through their arguments; counter calls_trunk; the unit's CPU equality test T1 60/60 bitwise on the pin"),
    "pair_offload_conf":  ("tested", f"ERRATA_05: the confidence mirror retains per-sample outputs as ConfidenceHead.forward does on the pin ({UP}/opendde/model/modules/confidence.py: "
                                        "preallocated [N_sample, ...] buffers, PAE/PDE logits to the CPU when N_sample > 1); counters calls_conf, calls_disto"),
    "diffz":              ("tested", f"ERRATA_05: DiffusionConditioning.forward served without the per-step pair_z clone ({UP}/opendde/model/modules/diffusion.py, unchanged name); counter calls_diffz"),
    "bigln_guard":        ("tested", "ERRATA_05: the F.layer_norm guard above 2^31 elements (a torch bind, not an opendde one); counter _BIGLN calls / hits"),
    "free_templ":         ("tested", "ERRATA_05: template pair features released after the trunk inside the offloaded trunk stage; counter calls_free_templ"),
    # levers/SAMPLER: bound against opendde 1.1.1's sampler module tree (names read at install and asserted: diffusion_module.diffusion_transformer.blocks[24]
    # .attention_pair_bias.attention (16x48, gating, SDPA statement primitives.py:268-302), atom_attention_encoder/decoder.atom_transformer (3 blocks, 4x32, 32x128 windows,
    # primitives._local_attention's statement as the windowed kernel assumes it), diffusion_conditioning.forward(t_hat first; returns (s, pair_z)), DiffusionTransformer.forward(+extra_attn_bias))
    "dit_attn_apb":       ("tested", f"binds primitives.Attention.forward per instance on the 24 token modules ({UP}/opendde/model/modules/primitives.py:696-910; transformer.py:1338)"),
    "atom_attn_apb":      ("tested", f"binds primitives.Attention.forward per instance on the 6 atom modules; primitives._local_attention ({UP}/opendde/model/modules/primitives.py:597-695) is the donor's statement byte for byte"),
    "cond_dedupe":        ("tested", f"binds DiffusionConditioning.forward per instance ({UP}/opendde/model/modules/diffusion.py:1054-1121; the sampler's t_hat expand: {UP}/opendde/model/generator.py:221-225)"),
    "dit_fused":          ("tested", f"binds DiffusionTransformer.forward per instance ({UP}/opendde/model/modules/transformer.py:1750-1800); block sub-module names asserted at install"),
    "dit_lowp":           ("tested", "a precision word of dit_fused (opendde_fpf_ditfast dit_fast.py: the a-path operand dtype)"),
    "atom_fused":         ("tested", f"binds AtomTransformer.forward per instance ({UP}/opendde/model/modules/transformer.py:2185-2241); block sub-module names asserted at install"),
    "dit_attn_exact":     ("tested", f"binds the module-global primitives._attention ({UP}/opendde/model/modules/primitives.py:268-302); the provider's exact tier vouches its row per card / stack, in-process bit-equality probe"),
}
# ------------------------------------------------------------------------------------------------------------ engagement per lever
# Ran-or-refuse (ran.py) asks every counted lever for its own counter after the first prediction. A lever is INERT BY DESIGN on a call whose
# facts, known before the run, put it outside its documented domain: the deterministic recipe, the items'
# token counts against the unit's own size gate, the number of rank processes. Then a zero counter is `lever_inert_by_design:<lever>(<reason>)`
# (activation.levers_inert, the run COMPLETE); inside the domain a zero counter is `lever_never_ran:<lever>` (PARTIAL, exit 3). The thresholds
# are READ from the units' own constants (the module the process loaded), never restated here; the one literal (the fpf engine's small-N
# regime) is cited and locked by tests/test_ran.py against the carried bytes.
FPF_TRIMUL_MIN_N = 100                                # levers/ARMT/odde_trimul_bind.py N_MIN - 1: at and below 100 rows the stock module runs cuEquivariance's small-N torch algorithm (Aside n_le_100_stock_small_n_path)


def _unit_const(module: str, attr: str, key: str | None = None):
    """A loaded unit's own constant (``sys.modules[module].attr[key]``); None when the unit is not loaded in this process."""
    import sys as _sys
    m = _sys.modules.get(module)
    if m is None:
        return None
    v = getattr(m, attr, None)
    if key is not None:
        v = (v or {}).get(key) if isinstance(v, dict) else None
    return v



@dataclass(frozen=True)
class Engagement:
    """A counted lever's engagement predicate over the facts of a call (``ran.engagement``): ``min_tokens`` = (reader of the unit's size
    gate, citation): engaged iff an item's token count reaches it; ``multi_gpu`` = engaged only with more than one rank process;
    ``single_gpu`` = the reason more than one rank process makes it inert (it is served by another lever there: ``replaced_by=<lever>: ...``);
    ``featurising_rank`` = the reason a rank other than rank 0 of ``--n_gpu P>1`` makes a data-path lever inert (rank 0 alone featurises there)."""
    min_tokens: tuple | None = None
    multi_gpu: bool = False
    single_gpu: str | None = None
    torch_layernorm: str | None = None                     # the reason the stock's fused LayerNorm (LAYERNORM_TYPE=fast_layernorm) makes it inert: the lever binds torch's F.layer_norm
    fused_layernorm: str | None = None                     # the reason a LayerNorm backend OTHER than the fused extension makes it inert: the lever binds the extension's loader
    featurising_rank: str | None = None                    # the reason a rank process other than rank 0 of --n_gpu P>1 makes a data-path lever inert: rank 0 alone featurises there (tp._rank0_item)
    stock_knob: str | None = None                          # the upstream flag (settings.KERNEL_KNOBS) whose STATED value other than `auto` puts the lever aside by name (stockknob.py)
    stepped_aside: tuple | None = None                     # (reader of the unit's own census, citation): the unit reached its site and handed EVERY call to the stock op by a named rule (the cell's winner is the stock op, a row unavailable on this stack by name) -- inert, the reason = the unit's words


_ARM_GATE = (lambda: _unit_const("odde_arm_t", "CFG", "MIN_TOKENS"), "levers/ARMT/odde_arm_t/__init__.py CFG['MIN_TOKENS'] (ODDE_ARM_T_MIN_TOKENS, 300): below it every ARM call is stock by design")
_FPF_GATE = (lambda: FPF_TRIMUL_MIN_N + 1, "levers/ARMT/odde_trimul_bind.py N_MIN: N <= 100 is the engine's own small-N torch regime (upstream's cuEquivariance path) -- the stock forward by name")
_RANK0_FEATS = ("served_on=rank0: under --n_gpu P rank 0 alone featurises each query item (data_form=rank0_bcast, opendde_opt/tp.py _rank0_item) and the lever runs "
                "there; this rank receives the features rank 0 built, bond_mask already dropped")
_STRUCT_ROWPAIR = ("replaced_by=rowpair_tp: under --n_gpu P the structural stage runs on this rank's structural ROW SHARD (opendde_opt/tp_struct.py: expansion rows from the "
                   "ring-fetched parent rows, the refiner through the core pair-block driver) and the template pair features are this rank's rows — no host pair, nothing to free")
_DIFFZ_ROWPAIR = ("replaced_by=rowpair_tp: under --n_gpu P the row-sharded roll-out (opendde_opt/tp_diffusion.py: z_cond ROWS + per-block pair-bias rows "
                  "computed once per roll-out) serves DiffusionConditioning / the DiffusionTransformer pair path; the unit's diffz hook is never called")
_TRIATTN_TP = ("replaced_by=tp_triatt: under --n_gpu P the row-sharded pair stacks serve triangle attention through the core's row-block dispatch (opendde_opt/tp.py _triatt_fns); "
               "the arm's attention site is this rank's only for a pair track the shard does not own")


def _triattn_aside():
    """odde_triattn_bind's own account when it reached its site and served no call through a provider row: 'aside:<reason:n,...>[;unavailable=<row:kind,...>]'; None otherwise."""
    c = _unit_const("odde_triattn_bind", "COUNTS") or {}
    if not c or int(c.get("calls") or 0) > 0 or int(c.get("stock_calls") or 0) == 0 or c.get("errors"):
        return None
    why = "aside:" + ",".join(f"{k}:{v}" for k, v in sorted((c.get("asides") or {}).items()))
    un = c.get("unavailable") or {}
    return why + (";unavailable=" + ",".join(f"{k}:{v}" for k, v in sorted(un.items())) if un else "")


def _trimul_aside():
    """odde_trimul_bind's own account when it reached its site and served no call through a provider row: 'aside:<reason:n,...>'; None otherwise."""
    c = _unit_const("odde_trimul_bind", "COUNTS") or {}
    if not c or int(c.get("calls") or 0) > 0 or int(c.get("stock_calls") or 0) == 0 or c.get("errors"):
        return None
    return "aside:" + ",".join(f"{k}:{v}" for k, v in sorted((c.get("asides") or {}).items()))


def _lncore_aside():
    """opendde_opt.lncore's own account when it reached its site and served no call through a provider row: 'aside:<reason:n,...>'; None otherwise."""
    import sys as _sys
    m = _sys.modules.get("opendde_opt.lncore")
    return m.aside_word() if m is not None else None


_LNCORE_ASIDE = (_lncore_aside, "opendde_opt/lncore.py forward(): every LayerNorm call of this pass was the extension's by a named rule (the cell's winner is a stock row at these rows, another width / form, capture) -- the extension served, counted")


def _transition_aside():
    """odde_transition_bind's account when its word was live and no pair-transition call was decided under it although the adapter reached its modules: every
    call was the module's own by a named eligibility rule (an fp32 base without bf16 autocast, a cell other than (384, 1536)) -> 'asides=<reason:n,...>'; None otherwise."""
    c = _unit_const("odde_transition_bind", "COUNTS") or {}
    st = (_unit_const("fpf_transition_odde", "STATS") or {})
    if not c or int(c.get("calls") or 0) > 0 or int(st.get("fallback") or 0) == 0:
        return None
    return "asides=" + ",".join(f"{k}:{v}" for k, v in sorted((st.get("fallback_reasons") or {}).items()))


_TRANSITION_ASIDE = (_transition_aside, "levers/ARMT/third_party/fpf_transition_odde eligible_sep16(): every pair-transition call of this pass kept the engine's module by a named rule (fp32 input without bf16 autocast, another cell) -- counted, the run complete")
_TRANSITION_TP = ("under --n_gpu P the row-sharded pair stacks call each block's pair transition on this rank's row block through the core's row-block driver (opendde_opt/tp.py _block_fns); "
                  "the binding serves those calls where the rank's rows are eligible and is this rank's only for them")
_TRIMUL_ASIDE = (_trimul_aside, "levers/ARMT/odde_trimul_bind.py serve(): the provider cell's winner for every call of this pass is the stock op, or the tier's rows refuse on this stack by name -- the route's own construction served, counted")
_TRIMUL_TP = ("replaced_by=rowpair_tp: under --n_gpu P the row-sharded pair stacks serve triangle multiplication through the core's row-block dispatch; "
              "the arm's TriMul site is this rank's only for a pair track the shard does not own")
_TRIATTN_ASIDE = (_triattn_aside, "levers/ARMT/odde_triattn_bind.py serve(): the provider cell's winner for every call of this pass is the stock op, or the tier's rows are unavailable on this stack by name -- the stock op served, counted")
def conf_offload_calls() -> int:
    """The offload unit's own count of confidence passes it served this process (levers/OFFLOAD odde_offload.STATS['calls_conf']: off_confidence_head
    entered); 0 when the unit is absent, off the line, below its size gate, or its conf stage is not among ODDE_OFFLOAD's stages."""
    return int((_unit_const("odde_offload", "STATS") or {}).get("calls_conf") or 0)


def conf_chunk_census() -> dict | None:
    """The confidence stack's per-module mark (levers/ARMT/odde_arm_t/odde_conf_chunk: the hook pair on that stack's triangle-attention modules, which the offload
    unit's row blocks and upstream's chunk loop both pass): its census dict, or None when the unit is not loaded / not bound on a model this process."""
    import sys as _sys
    m = _sys.modules.get("odde_arm_t.odde_conf_chunk") or _sys.modules.get("odde_conf_chunk")   # the unit lives inside the arm package (a bare name from an older tree tolerated)
    if m is None or not int((getattr(m, "COUNTS", None) or {}).get("bound") or 0):
        return None
    try:
        return dict(m.describe())
    except Exception:  # noqa: BLE001
        return dict(getattr(m, "COUNTS", None) or {})


def _triattn_conf_aside():
    """the confidence stack's account: the binding stepped aside for the whole pass (as triattn_*); every marked confidence-stack call went to the stock op
    by the cell's rule ('conf_cells_stock:<n>' -- the module path and the offload unit's row blocks alike: levers/ARMT/odde_arm_t/odde_conf_chunk marks the stack's
    triangle-attention modules per call, so `off_confidence_head` -> `off_pairformer_stack` -> `block.tri_att_*.mha` per host-streamed row block is served
    by the word under the same rule and counted in the same two counters); or the offload unit's conf stage served that stack this pass WITHOUT the
    per-module mark bound ('conf_path=offload_rows:<n>': a tree / model where odde_conf_chunk hooked nothing -- then no call can reach the stack-level mark,
    by construction, and lever pair_offload_conf accounts for the stage); None when a provider row served there, or the mark is bound and read 0 (the
    arm's wrapper never ran on those row blocks: a zero counter is lever_never_ran), or the stack was never reached with the word live."""
    a = _triattn_aside()
    if a is not None:
        return a
    c = _unit_const("odde_arm_t", "COUNTS") or {}
    n, served = int(c.get("att_conf_calls") or 0), int(c.get("att_conf_prov_calls") or 0)
    if n > 0 and served == 0:
        return f"conf_cells_stock:{n}"
    off = conf_offload_calls()
    if n == 0 and off > 0 and conf_chunk_census() is None:                  # the row-block path ran unmarked (the per-module hook not bound here): inert by name
        return f"conf_path=offload_rows:{off} (replaced_by=pair_offload_conf: the offload unit served the confidence head's pair stack in host-streamed row blocks this pass, the per-module mark not bound)"
    return None


_TRIATTN_CONF_ASIDE = (_triattn_conf_aside, "levers/ARMT/odde_arm_t make_attn + levers/ARMT/odde_arm_t/odde_conf_chunk: every triangle-attention call of the confidence head's stack -- its own forward's and the "
                       "offload unit's row blocks' (levers/OFFLOAD odde_offload_trunk.off_confidence_head -> odde_offload.off_pairformer_stack, marked per module) -- went to the stock op by the "
                       "provider cell's rule, the binding stepped aside for the pass, or the row-block path ran with the per-module mark unbound (odde_offload.STATS['calls_conf']) -- inert by design, the run complete")
def _sampler_aside(section: str, calls_key: str = "calls"):
    """stepped_aside reader of a SAMPLER lever: 'asides=<word>:<n>,…' when its unit served NO call through the kernel and stepped aside by name on
    >= 1 call (every call of the run took the module's stock forward: local-attention / no-bias / per-sample-bias call forms); None otherwise.
    The sections are the unit's live census dicts (odde_sampler.SECTIONS, bound into odde_accel_v2.STATS)."""
    def read():
        sec = _unit_const("odde_accel_v2", "STATS", section) or {}
        if (sec.get(calls_key) or 0) > 0 or not (sec.get("asides") or 0):
            return None
        words = {k[len("aside_"):]: v for k, v in sec.items() if str(k).startswith("aside_") and v}
        return "asides=" + ",".join(f"{k}:{v}" for k, v in sorted(words.items()))
    return read


def _u2_admission_aside():
    """stepped_aside reader of arm_u23's TriMul route: 'trimul_route=<reason>:<n>,…' when every admitted TriMul call of the run ran the stock TriMul
    BY NAME for a named route decision (no provider word: lever trimul_core left out; the provider's measured stock winner / refusal by name) or
    by the admission words, no call was served and no kernel exception occurred; None otherwise."""
    m = __import__("sys").modules.get("odde_arm_t")
    if m is None:
        return None
    c = getattr(m, "COUNTS", {}) or {}
    reasons = c.get("trimul_stock_reasons") or {}
    words = tuple(getattr(m, "ADMISSION_TRIMUL_REASONS", ())) + ("no_provider_word",)
    adm = {r: n for r, n in reasons.items() if n and (r in words or str(r).startswith("provider_aside:"))}
    if not adm or reasons.get("exception") or (c.get("trimul_fast_calls") or 0) + (c.get("trimul_bf16_calls") or 0) > 0:
        return None
    return "trimul_route=" + ",".join(f"{k}:{v}" for k, v in sorted(adm.items()))


_SAMPLER_ASIDE_CITE = ("levers/SAMPLER/odde_apb_bind.py _aside / opendde_fpf_ditfast (StepAside, count_aside): a call form the site's rows do not serve "
                       "(local attention, no pair bias, per-sample bias, other bias layouts / shapes) runs the module's stock forward BY NAME, counted per word; 0.2.53")


ENGAGEMENT: dict[str, Engagement] = {
    "arm_z":   Engagement(min_tokens=_ARM_GATE),
    "trimul_core": Engagement(min_tokens=_ARM_GATE, single_gpu=_TRIMUL_TP, stepped_aside=_TRIMUL_ASIDE, stock_knob="trimul_kernel"),
    "trimul_exact": Engagement(min_tokens=_ARM_GATE, single_gpu=_TRIMUL_TP, stepped_aside=_TRIMUL_ASIDE, stock_knob="trimul_kernel"),
    "ln_core": Engagement(stepped_aside=_LNCORE_ASIDE, fused_layernorm="LAYERNORM_TYPE other than fast_layernorm (a caller's own export, named on the ACTIVE line): upstream binds torch's LayerNorm "
                          "class, FusedLayerNorm is never constructed and the lever's wrapper has no call site"),
    "transition_core": Engagement(min_tokens=_ARM_GATE, single_gpu=_TRANSITION_TP, stepped_aside=_TRANSITION_ASIDE),
    "transition_exact": Engagement(min_tokens=_ARM_GATE, stepped_aside=_TRANSITION_ASIDE),
    "triattn_core": Engagement(min_tokens=_ARM_GATE, single_gpu=_TRIATTN_TP, stepped_aside=_TRIATTN_ASIDE, stock_knob="triatt_kernel"),
    "triattn_exact": Engagement(min_tokens=_ARM_GATE, single_gpu=_TRIATTN_TP, stepped_aside=_TRIATTN_ASIDE, stock_knob="triatt_kernel"),
    "triattn_conf": Engagement(min_tokens=_ARM_GATE, single_gpu=_TRIATTN_TP, stepped_aside=_TRIATTN_CONF_ASIDE, stock_knob="triatt_kernel"),
    "lnstream": Engagement(fused_layernorm="LAYERNORM_TYPE other than fast_layernorm (a caller's own export, named on the ACTIVE line): upstream binds torch's LayerNorm "
                                          "and never calls the extension loader the lever wraps — the stock base of this tree is fast_layernorm"),
    "arm_u":   Engagement(min_tokens=_ARM_GATE),
    "arm_u23": Engagement(min_tokens=_ARM_GATE, stock_knob="trimul_kernel", stepped_aside=(_u2_admission_aside, "levers/ARMT/odde_arm_t make_trimul_forward: a U2 TriMul call without the provider word (lever trimul_core left out), or one "
                                                "the core provider hands back (measured stock winner, refusal by name), runs the stock TriMul BY NAME — counted (trimul_route=<reason>:<n>), the run complete; 0.2.57")),
    "fpf_trimul_exact": Engagement(stock_knob="trimul_kernel", min_tokens=_FPF_GATE),
    "drop_bond_mask": Engagement(featurising_rank=_RANK0_FEATS),               # n_gpu>1, rank>0: the featurizer never runs on this rank (rank 0 featurises for every rank)
    "rowpair_tp": Engagement(multi_gpu=True),                                  # n_gpu=1: the adapter installs nothing by design
    "struct_pair_bf16": Engagement(multi_gpu=True),                              # the structural pair shard exists only in a rank process of --n_gpu P>1 (line BIG_TP)
    "tp_triatt": Engagement(multi_gpu=True),                                     # the sharded pair stack's row-block kernel: bound only in a rank process of --n_gpu P>1
    "diffz": Engagement(single_gpu=_DIFFZ_ROWPAIR),                              # n_gpu>1: the row-sharded roll-out serves the diffusion pair path
    "bigln_guard": Engagement(single_gpu=_DIFFZ_ROWPAIR,                         # (rides diffz: the guarded LayerNorm image is diffz's)
                              torch_layernorm="LAYERNORM_TYPE=fast_layernorm: upstream's LayerNorm is the fused extension fast_layer_norm_cuda_v2, which never calls "
                                              "torch.nn.functional.layer_norm — the guard wraps that function, so no LayerNorm of the model reaches it (the stock's precision base)"),
    "pair_offload_struct": Engagement(single_gpu=_STRUCT_ROWPAIR),               # n_gpu>1: the structural stage runs on this rank's structural row shard
    "tmpl_dedup": Engagement(single_gpu="replaced_by=rowpair_tp: under --n_gpu P the template pair blocks run on this rank's ROW SHARD through the core pair-block "
                                        "driver (opendde_opt/tp.py; registry rowpair_tp); TemplateEmbedder.forward — the method the lever wraps — is never entered on a rank "
                                        "process, with or without templates (a rank's LEVER row: state=off with this reason; the x1 lines de-duplicate as before)"),   # n_gpu>1: inert by design, never `lever_never_ran`
    "free_templ": Engagement(single_gpu=_STRUCT_ROWPAIR),                        # (the template pair features are this rank's rows already)
    # levers/SAMPLER: under `--n_gpu P` the row-sharded roll-out (opendde_opt/tp_diffusion.py dit_block_fns / denoise_sharded) drives the token blocks' projections and the
    # module-global primitives._attention with per-rank ROW chunks and its own conditioning: the instance-level forwards these levers replace are never entered there
    "dit_attn_apb":   Engagement(stepped_aside=(_sampler_aside("apb_state_dit"), _SAMPLER_ASIDE_CITE), single_gpu=_DIFFZ_ROWPAIR[0].replace("the unit's diffz hook is never called", "the token modules' instance forwards are never entered")),
    "cond_dedupe":    Engagement(single_gpu=_DIFFZ_ROWPAIR[0].replace("the unit's diffz hook is never called", "DiffusionConditioning.forward is never entered (tp_diffusion._single_cond)")),
    "dit_fused":      Engagement(stepped_aside=(_sampler_aside("dit_fused"), _SAMPLER_ASIDE_CITE), single_gpu=_DIFFZ_ROWPAIR[0].replace("the unit's diffz hook is never called", "DiffusionTransformer.forward is never entered")),
    "dit_lowp":       Engagement(stepped_aside=(_sampler_aside("dit_fused", "lowp_calls"), _SAMPLER_ASIDE_CITE), single_gpu=_DIFFZ_ROWPAIR[0].replace("the unit's diffz hook is never called", "DiffusionTransformer.forward is never entered")),
    "atom_attn_apb": Engagement(stepped_aside=(_sampler_aside("apb_state_atom"), _SAMPLER_ASIDE_CITE)),
    "atom_fused": Engagement(stepped_aside=(_sampler_aside("atom_fused"), _SAMPLER_ASIDE_CITE)),
    "dit_attn_exact": Engagement(stepped_aside=(lambda: (lambda a: f"aside:{a}" if a else None)((_unit_const("odde_apb_bind", "COUNTS", "exact") or {}).get("aside")),
                                                "levers/SAMPLER/odde_apb_bind install_exact(): the core provider's exact tier (opt_core.kernels.apb word exact) names the stock op for the running "
                                                "card / stack (no vouched kernel row) or the in-process probe refused — nothing is installed and the stock fp32 SDPA statement serves (exact by construction); 0.2.47 / 0.2.57")),
}


TESTED_ON: dict[str, frozenset[str]] = {
    "1.0.0": frozenset(LEVERS),                       # every carried lever was built and tested against opendde 1.0.0's symbols
    PIN: frozenset(n for n, (st, _why) in PIN_STATUS.items() if st == "tested"),
}
OFF_LINE = frozenset(n for n, (st, _why) in PIN_STATUS.items() if st in ("superseded_by_upstream", "retired", "refused"))   # levers no line may carry on this pin
REFUSED_ON_PIN = {n: why.split(":", 1)[0] for n, (st, why) in PIN_STATUS.items() if st == "refused"}   # {lever: reason name} — printed on every ACTIVE / PRED line (`refused_on_pin=<lever>:<name>`)
# lines held as a family although each of their levers is tested ({line: reason}); a held line refuses by name like an untested one
# (`pending_rebase ... line_on_hold=<reason>`). No line is held on this pin.
LINES_ON_HOLD: dict[str, str] = {}


def pin_status(lever: str) -> tuple[str, str]:
    """(status, reason) of a lever on the tree's pin (PIN_STATUS)."""
    return PIN_STATUS[lever]


def tested_on(lever: str) -> tuple[str, ...]:
    """The opendde versions a lever is tested on, ascending."""
    return tuple(sorted(v for v, names in TESTED_ON.items() if lever in names))


def untested(levers, version: str | None) -> list[str]:
    """The levers among ``levers`` (a line's composition, in its order) NOT tested on opendde ``version`` — empty = the line may activate."""
    ok = TESTED_ON.get(version or "", frozenset())
    return [n for n in levers if n not in ok]


def render_status_table(pin: str = PIN) -> str:
    """One row per LEVERS entry: its status on `pin` (PIN_STATUS) with the reason, and the versions it is tested on."""
    assert pin == PIN, f"the status table is the tree's pin's ({PIN}), asked {pin}"
    rows = [f"| lever | status on opendde {pin} | why | tested on |", "|---|---|---|---|"]
    for n in LEVERS:
        st, why = PIN_STATUS[n]
        rows.append(f"| `{n}` | {st} | {_cell(why)} | {', '.join(tested_on(n)) or '—'} |")
    return "\n".join(rows) + "\n"


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def validate() -> list[str]:
    """Registry invariants (the tests call this): known classes and tiers, every lever named by a line exists, every lever has a status on
    the pin, no line carries a superseded or retired lever."""
    from . import modes
    bad = []
    for n in LEVERS:
        if n not in PIN_STATUS or PIN_STATUS[n][0] not in STATUSES:
            bad.append(f"{n}: no status on the pin (PIN_STATUS)")
    for n in PIN_STATUS:
        if n not in LEVERS:
            bad.append(f"PIN_STATUS names an unknown lever {n}")
    for ln in modes.LINES.values():
        for n in ln.levers:
            if n in OFF_LINE:
                bad.append(f"line {ln.name} carries {n}, which is {PIN_STATUS[n][0]} on the pin")
    for n, lv in LEVERS.items():
        if lv.cls not in CLASSES:
            bad.append(f"{n}: class {lv.cls}")
        if lv.tier not in TIERS:
            bad.append(f"{n}: tier {lv.tier}")
        if lv.kit not in modes.KIT_DIRS and lv.kit != HOUSE:
            bad.append(f"{n}: kit {lv.kit}")
    for ln in modes.LINES.values():
        for n in ln.levers:
            if n not in LEVERS:
                bad.append(f"line {ln.name}: unknown lever {n}")
    return bad
