"""The lever registry: one entry per lever of the ESMFold2 kit line, keyed by the kit's own flag name.

The kit server (`driver/ef2_server.py`) describes a mode as a tuple ``(base, opt_flags, w4_flags, extra)`` and installs levers in
its ``configure()``; this registry only *describes* each flag — which kit file applies it, its class (the four-class test:
forward / datapath / serving / orchestration), its numerics tier against the kit's own fused line and against library-default stock,
the variants it can act on, and the kit record that proves it was applied in a process (`probe`). No lever value, switch value or
mode composition lives here: modes are read from the server's own table (modes.py) and applied by the server's own `configure()`.

`probe` grammar (read by stack.classify from the kit modules' own state after configure()):
  ("w4", <key>)      ef2_w4.describe()[key] truthy (and the lever not in describe()["device"]["disabled"])
  ("opt", <key>)     the dict ef2_opt.install() returned has a truthy value under key
  ("msa", <key>)     ef2_msa._STATE[key] truthy on a model with an msa_encoder
  ("mk", None)       model.structure_head.diffusion_module._mk_enabled
  ("base", None)     the server's desc string records set_kernel_backend('fused')
  ("atom", <name>)   ef2_atom.groups_on()[name] (ax: the four exact hoists all on; af: fused block + segmented mean on)
  ("feats", None)    ef2_feats.active()
  ("msa2", <name>)   name in ef2_msa_v2.levers_on() (m15 / m16 / m17 / mh) on a model with an msa_encoder
  ("pair", <name>)   ef2_pair_v2.levers_on()[name] (t15 / t15msa / xtr)
  ("cute", <name>)   ef2_transition_cute.levers_on()[name] (t16: the class patches live and the kernel loaded)
  (the trimul group's `tx` is ef2_w4's: probe ("w4", "tx") = ef2_w4.describe()["tx"], on when a provider kernel row serves the pair TriMul under the bound tier word;
   False with describe()["tx_state"]["why"] when the word names the stock op on this stack -- the upstream fused TriMul serves by name, the LEVER line says so)
  ("hoist", <name>)  ef2_hoist.stats()["levers"][name] (trimul / glue / disto)
  ("dit", <name>)    ef2_dit.levers_on()[name] (ro / kd / dit)
  ("ln", <name>)     ef2_xln.levers_on()[name] (xln; False with ef2_xln.refusal() naming the stack word when the shared core's row cannot serve this box)
  ("xte", <name>)    ef2_xte.levers_on()[name] (xte; False with ef2_xte.refusal() naming the class / core / self-check word when the shared core's row cannot serve this box)
  ("rc", <name>)     esmfold2_opt.rowchunk.install.levers_on()[name] — installed after configure() by rowpair.install_rank (n_gpu > 1): stack.classify defers
                     these and stack.settle_route judges them from the installer's record (a member that did not bind is partial: refused by name)

Per-class sets (``Lever.classes`` / ``SUPERSEDED_ON``): a mode is ONE lever set; the compute-capability class of the box decides which of its
transition kernels serve — on 9.0 (H100 / H200) the CuTe kernel t16 (every d=256 Transition + the MSA PairTransition), which is sm_90a code; on every
other class t15 / t15msa (their sm80 / sm90 rows). The pair TriMul (`tx`) is one binding on every class: the shared core's provider decides the row per
capability by its cell table (``STEPS_ASIDE_ON`` is empty). modes.class_drop subtracts the other class's levers before configure() with a reason
token; the LEVER line says ``state=skipped reason=not_for_class:<class>:<token>`` and the DRY-RUN / APPLIED lines carry ``not_for_class=<names>`` —
declared, never silent, never a refusal.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

FIELD_BASE, FIELD_OPT, FIELD_W4, FIELD_MK, FIELD_MSA = "base", "opt", "w4", "mk", "msa"
FIELD_ATOM, FIELD_FEATS, FIELD_MSA2, FIELD_PAIR, FIELD_HOIST, FIELD_DIT = "atom", "feats", "msa2", "pair", "hoist", "dit"   # the fourth field's module groups (ef2_server.GROUPS)
FIELD_TRIMUL = "trimul"                                            # the fourth field's `trimul:` group (lever tx: ef2_w4.enable_tx, the shared core's TriMul provider by tier word)
FIELD_LN = "ln"                                                    # the fourth field's `ln:` group (ef2_xln: the release tree's exactln LayerNorm row at the pair-sized nn.LayerNorm sites)
CLASS_SM90: Tuple[str, ...] = ("9.0",)                             # compute-capability classes of the sm_90a (CuTe / NVRTC) kernels: H100, H200
FIELD_RC = "rc"                                                    # the row-chunking levers of the row-sharded route (esmfold2_opt/rowchunk; installed by rowpair.install_rank at n_gpu > 1): the big set gains them at n_gpu > 1 (modes.route_add), not a server-table field
FIELD_XL = "xl"                                                    # the XL memory add-on's levers (opt/forward/EF2_XL_ADDON_v1/ef2_xl.py): the `big` mode, not a server-table field
ALL_VARIANTS: Tuple[str, ...] = ("fast", "full_msa", "full_nomsa")   # the one variants literal (modes.VARIANTS re-exports it)
FULL_MODEL: Tuple[str, ...] = ("full_msa", "full_nomsa")   # the Full model: the kit applies its MSA-encoder levers to every model with an msa_encoder (ef2_opt.install_msa_fused_trimul; ef2_msa.enable), whether or not the item carries an MSA


@dataclass(frozen=True)
class Lever:
    name: str                      # the kit's flag name as written in the server's MODES table / its EF2_* switches
    field: str                     # which field of the server's mode tuple carries it (FIELD_*)
    kit_file: str                  # kit file that implements it (relative to the kit's driver/)
    what: str                      # one line: what changes
    class_4: str                   # forward | datapath | serving | orchestration
    tier_vs_kit_line: str          # numerics vs the kit's fused line: bitwise | exact | T1 (replay) | T2
    tier_vs_stock: str             # numerics vs library-default stock (every kit mode inherits the fused backend: T2)
    probe: Tuple[str, Optional[str]]
    variants: Tuple[str, ...] = ALL_VARIANTS   # variants on which the lever can act; elsewhere the kit's own guards skip it
    doc: str = ""                  # where the kit documents it (file:line of the kit)
    classes: Optional[Tuple[str, ...]] = None  # compute-capability classes ("9.0", …) whose set carries the lever; None = every class (modes.class_drop subtracts it elsewhere, named)


LEVERS: Dict[str, Lever] = {
    # --- field 1: numerics base (server configure(): set_kernel_backend / set_chunk_size) ------------------------------------------
    "fused": Lever("fused", FIELD_BASE, "ef2_server.py", "set_kernel_backend('fused') + set_chunk_size(None): the vendored bf16 Triton "
                   "kernels with fp32 accumulate replace the reference einsum/Linear path; defines the kit line", "forward", "n/a (defines the kit line)",
                   "T2", ("base", None), doc="CHANGES.md (base)"),
    # --- field 2: ef2_opt.install flags --------------------------------------------------------------------------------------------
    "tg": Lever("tg", FIELD_OPT, "ef2_opt.py", "CUDA-graph pair trunk / coda / confidence trunk (capture + replay)", "serving", "T1 replay", "T2",
                ("opt", "trunk_graphs"), doc="kit README.md (graphs)"),
    "sg": Lever("sg", FIELD_OPT, "ef2_opt.py", "sync-free graphed diffusion sampler (RNG order unchanged)", "serving", "T1 replay", "T2",
                ("opt", "sampler_graphs"), doc="ef2_opt.py:10-14"),
    "eg": Lever("eg", FIELD_OPT, "ef2_opt.py", "CUDA graphs for lm_encoder / msa_encoder", "serving", "T1 replay", "T2", ("opt", "encoder_graphs")),
    "ec": Lever("ec", FIELD_OPT, "ef2_opt.py", "ESMC-6B hidden-state cache across seeds of one design", "datapath", "T1 (cache)", "T2", ("opt", "esmc_cache")),
    "fc": Lever("fc", FIELD_OPT, "ef2_opt.py", "prepare_input feature cache on the builder (seed-blind; a SMILES-ligand input bypasses it per seed — stack.guard_feature_cache)", "datapath", "T1 (cache)", "T2", ("opt", "feature_cache")),
    "pb": Lever("pb", FIELD_OPT, "ef2_opt.py", "per-fold pair-bias cache in the diffusion transformer", "serving", "T1 (cache)", "T2", ("opt", "pair_bias_cache_blocks")),
    "msa": Lever("msa", FIELD_OPT, "ef2_opt.py", "fused TriMul inside the MSA encoder (M1; Tier 2 on both Full variants)", "forward", "T2", "T2", ("opt", "msa_fused_trimul_blocks"),
                 variants=FULL_MODEL, doc="kit README (mode suffix _msa)"),
    # --- field 3: ef2_w4.enable flags -------------------------------------------------------------------------------------------------
    "t1": Lever("t1", FIELD_W4, "ef2_w4.py", "transition no-lin kernel (the small-shared-memory substitute for t6)", "forward", "bitwise", "T2", ("w4", "transition_nolin"),
                doc="W4 README 'device policy'"),
    "t3": Lever("t3", FIELD_W4, "ef2_w4.py", "re-tuned Triton tile table (transition + stage-5 GEMM)", "forward", "bitwise", "T2", ("w4", "tiles")),
    "t5": Lever("t5", FIELD_W4, "ef2_w4.py", "cached weight casts", "forward", "bitwise", "T2", ("w4", "weight_cache")),
    "t6": Lever("t6", FIELD_W4, "ef2_w4.py", "row-block transition kernel", "forward", "bitwise", "T2", ("w4", "transition_rowblock")),
    "t10": Lever("t10", FIELD_W4, "ef2_w4.py", "one-kernel fused pair transition (supersedes t6)", "forward", "T2", "T2", ("w4", "transition_fused")),
    # --- field 4: the vendored levers ('mk' and 'msa:<flags>') --------------------------------------------------------------------------
    "mk": Lever("mk", FIELD_MK, "ef2_mk_sampler.py", "diffusion-sampler conditioning hoisted per fold (same ops; batch-1 only)", "forward", "exact", "T2", ("mk", None),
                doc="W4 README 'enable order'"),
    "t11": Lever("t11", FIELD_MSA, "ef2_msa.py", "MSA transition (one bf16-LN + fused SiLU*mul kernel)", "forward", "T2", "T2", ("msa", "msa_transition"), variants=FULL_MODEL),
    "t12": Lever("t12", FIELD_MSA, "ef2_msa.py", "MSA outer-product mean with a fused epilogue (batch-1 only)", "forward", "bitwise", "T2", ("msa", "opm"), variants=FULL_MODEL),
    "t13": Lever("t13", FIELD_MSA, "ef2_msa.py", "MSA pair-weighted averaging in one Triton kernel (batch-1 only)", "forward", "T2", "T2", ("msa", "pwa"), variants=FULL_MODEL),
    "t14": Lever("t14", FIELD_MSA, "ef2_msa.py", "bf16-input LayerNorm(128) inside t11/t12/t13", "forward", "T2", "T2", ("msa", "ln_bf16"), variants=FULL_MODEL),
    # --- field 2 (ef2_opt.install), the recycle loop ----------------------------------------------------------------------------------
    "ls": Lever("ls", FIELD_OPT, "ef2_opt.py", "the trunk recycle loop re-issued with static loop I/O: graphed regions read and write resident buffers (no per-recycle "
                "clone / copy of the pair state, MSA invariants staged once per fold)", "serving", "bitwise", "T2", ("opt", "loop_static")),
    "rg": Lever("rg", FIELD_OPT, "ef2_opt.py", "ONE CUDA graph per trunk recycle (LM encoder + MSA encoder + injection + folding trunk; the RNG prologue issued eagerly "
                "in upstream's order) for pair planes up to ef2_opt's internal token ceiling; larger inputs take ls / tg (data-dependent, named per fold)",
                "serving", "T1 replay", "T2", ("opt", "recycle_graph"), variants=("fast",)),   # the Fast model only: no lever with a per-call host decision (the Full model's mh) may sit inside the captured body
    # --- field 4: the module groups (atom / fz / msa2 / pair / hoist / dit) ------------------------------------------------------------
    "ax": Lever("ax", FIELD_ATOM, "ef2_atom.py", "the diffusion module's atom encoder / decoder: adaLN factors hoisted per fold, prefix var-len attention I/O, RoPE tables "
                "per fold, alias-not-copy of step-invariant tensors (same kernels on the same operand values)", "forward", "bitwise", "T2", ("atom", "ax")),
    "af": Lever("af", FIELD_ATOM, "ef2_atom.py", "the atom-transformer block fused into three Triton kernels (adaLN + q|k|v|gate GEMM + RoPE / windowed attention / "
                "out-proj + SwiGLU + residual) + the encoder's token mean as a segmented reduction; GEMMs bf16-in / fp32-accumulate", "forward", "T2", "T2", ("atom", "af")),
    "fz": Lever("fz", FIELD_FEATS, "ef2_feats.py", "MSA featurisation vectorised on the host: residue types + deletion counts over one bytes view of the a3m text, and the "
                "taxonomy-paired row table by array operations instead of per-row dicts (identical arrays; the pairing self-checked against upstream once per process)", "datapath", "bitwise", "T2", ("feats", None)),
    "m15": Lever("m15", FIELD_MSA2, "ef2_msa_v2.py", "msa_transition + the block's residual in one Triton kernel (stock rounding points; supersedes t11)", "forward", "T2", "T2",
                 ("msa2", "m15"), variants=FULL_MODEL),
    "m16": Lever("m16", FIELD_MSA2, "ef2_msa_v2.py", "MSA pair-weighted averaging as producer / bias+softmax / bmm / epilogue kernels writing the layouts the bmm reads "
                 "(supersedes t13)", "forward", "T2", "T2", ("msa2", "m16"), variants=FULL_MODEL),
    "m17": Lever("m17", FIELD_MSA2, "ef2_msa_v2.py", "MSA outer-product mean: producer kernel writing the cuBLAS operands + fused projection / normalise / residual epilogue "
                 "(supersedes t12)", "forward", "T2", "T2", ("msa2", "m17"), variants=FULL_MODEL),
    "mh": Lever("mh", FIELD_MSA2, "ef2_msa_v2.py", "the MSA encoder evaluated once per fold and reused across the recycles when no row subsample is drawn (depth <= "
                "msa_max_depth: identical input tensors); engages per fold, named on the fold's lever line", "forward", "bitwise", "T2", ("msa2", "mh"), variants=("full_msa",)),
    "t15": Lever("t15", FIELD_PAIR, "ef2_pair_v2.py", "every FoldingTrunk's d=256 pair Transition (x + T(x)) through the shared core's transition provider by the mode's TIER word "
                 "(fast | big: the provider's measured cell decides the row per card and size; a refused call keeps the statement by name; supersedes t10 where it serves)", "forward", "T2", "T2", ("pair", "t15")),
    "t15msa": Lever("t15msa", FIELD_PAIR, "ef2_pair_v2.py", "the MSA-encoder blocks' PairTransition + residual through the same provider call as t15", "forward", "T2", "T2", ("pair", "t15msa"),
                    variants=FULL_MODEL),
    "xtr": Lever("xtr", FIELD_PAIR, "ef2_pair_v2.py", "the MSA module's PairTransition (pair_transition c=256 / msa_transition c=128, left on the reference LayerNorm -> "
                 "Linear -> SwiGLU -> Linear statements by the other exact levers) through the shared core's transition provider (opt_core.kernels.transition) "
                 "by word with the module's own LayerNorm output given: c=256 flash_sm90a (sealed sm_90a kernel; refused by name without its prebuilt -> v1), "
                 "c=128 v1 (LN-given Triton GEMM chain); bitwise vs the statements at the measured row windows, the statements outside them (counted)",
                 "forward", "bitwise", "T2", ("pair", "xtr"), variants=FULL_MODEL),
    "xte": Lever("xte", FIELD_PAIR, "ef2_xte.py", "the trunk's fused inference Transition (every d=256 / h=1024 C.Transition: folding trunk, lm_encoder, parcae coda, "
                 "confidence trunk; both models) through the shared core's transition provider (opt_core.kernels.transition) row esm_fused_exact by word: "
                 "upstream's LayerNorm-statistics + x_hat + SwiGLU kernel and its residual addmm as ONE kernel with the same rounding points and summation "
                 "order (bitwise vs the statement and vs t6 / t6s / t6i, self-checked at install; the residual in place when t6i is in the set); resolved once "
                 "per class (9.0), steps aside by name on a class / core without the row (t6 / t6s / t6i serve)",
                 "forward", "bitwise", "T2", ("xte", "xte")),
    "t16": Lever("t16", FIELD_PAIR, "ef2_transition_cute.py", "every d=256 / h=1024 Transition (folding trunk, lm_encoder, parcae coda, confidence trunk: x + T(x)) and the "
                 "MSA blocks' PairTransition (T(x); the block adds the residual) through ONE persistent warp-specialised CUDA C++ / CuTe kernel for sm_90a (TMA producer "
                 "warpgroup + two wgmma consumer warpgroups, in-register LayerNorm, fp32 accumulate), a prebuilt sm_90a cubin shipped with the kit, loaded through ef2_nvjit; "
                 "supersedes t15 / t15msa on the 9.0 class; served through the shared core's transition provider face by word (esm_t16 — the same audited object, carried), "
                 "the kit-local object by name when the face refuses", "forward", "T2", "T2", ("cute", "t16"), classes=CLASS_SM90),
    # --- field 4: trimul:tx (ef2_w4.enable_tx: the shared core's TriMul provider by the set's tier word) -------------------------------------------
    "tx": Lever("tx", FIELD_TRIMUL, "ef2_w4.py", "the pair TriMul (both directions of every PairUpdateBlock: LayerNorm, projections + gates + mask, the contraction, LayerNorm + gated "
                "out-projection, residual; bf16, residual in place of the fused call) served by the shared core's TriMul provider (opt_core.kernels.trimul) bound by the MODE's tier "
                "word (`exact` | `fast` | `big`): the provider's measured cell table decides the row per capability, stack, size class and direction on every card, each call "
                "class names its row once on the `tx:` line; under `big` (the graph-free memory mode) the eager calls of classes up to 512 tokens on class 9.x ask the word "
                "with the row preference tx_sm90a first (the cells' winner there, opt_core's native TriMul, is host-bound per call in eager use; LEVER prefer=); a class whose "
                "cell names the stock op (the `exact` word on a stack with no byte-vouched exact-class row) and a call the provider refuses are served by the upstream fused "
                "TriMul BY NAME (printed once, counted)",
                "forward", "T2", "T2", ("w4", "tx")),
    "trimul": Lever("trimul", FIELD_HOIST, "ef2_hoist.py", "the MSA module's reference TriMul contraction re-plumbed: tiled transposes + the identical bmm instead of einsum's "
                    "copies (Full model; the blocks ef2_opt's `msa` lever does not own)", "forward", "bitwise", "T2", ("hoist", "trimul"), variants=FULL_MODEL),
    "glue": Lever("glue", FIELD_HOIST, "ef2_hoist.py", "the TriMul's elementwise glue (mask, sigmoid gates) folded into trimul's transposes (sigmoid checked exhaustively "
                  "against torch's at install)", "forward", "bitwise", "T2", ("hoist", "glue"), variants=FULL_MODEL),
    "disto": Lever("disto", FIELD_HOIST, "ef2_hoist.py", "distogram logits copied device -> pinned host asynchronously right after the head, overlapped with the sampler "
                   "(the memory line's x10 owns this move instead)", "datapath", "bitwise", "T2", ("hoist", "disto")),
    "xln": Lever("xln", FIELD_LN, "ef2_xln.py", "the pair-sized nn.LayerNorm sites (MSA module reference path, injected-pair norm, base_z, confidence / conditioning pair "
                 "norms; >= 4096 rows) served by the release tree's exactln row (opt_core.kernels.ln: ATen's vectorized LayerNorm forward bit for bit, the "
                 "bf16-autocast widen form in one pass); steps aside by name on a stack the row does not serve", "forward", "bitwise", "T2", ("ln", "xln")),
    "ro": Lever("ro", FIELD_DIT, "ef2_dit.py", "the diffusion sampler as a device-resident roll-out: one CUDA graph per step boundary, fold constants staged once, RNG "
                "pre-drawn in upstream's order, an fp32 twin of the cached pair bias, cusolver Kabsch between replays (batch 1, one diffusion sample; other "
                "calls run the previous chain, counted)", "serving", "bitwise", "T2", ("dit", "ro")),
    "kd": Lever("kd", FIELD_DIT, "ef2_dit.py", "the roll-out's Kabsch alignment on the device (sync-free 3x3 SVD kernel instead of cusolver)", "forward", "T2", "T2", ("dit", "kd")),
    "dit": Lever("dit", FIELD_DIT, "ef2_dit.py", "the diffusion-transformer step fused: packed bf16 GEMMs (fp32 accumulate), fused pair-bias flash attention, Triton adaLN / "
                 "gate / SwiGLU kernels, the conditioning transitions in bf16", "forward", "T2", "T2", ("dit", "dit")),
}
ABLATION_KNOBS: Dict[str, Tuple[str, ...]] = {                       # <lever>.<knob> spellings the ablation variable accepts beside lever names (ef2_server.KNOBS; values = the module's own vocabularies)
    "af.gemm": ("fp32", "tf32", "tf32x3", "bf16"),
    "dit.gemm": ("fp32", "tf32", "tf32x3", "bf16x3", "bf16"), "dit.cond": ("fp32", "tf32", "tf32x3", "bf16x3", "bf16"),
    "dit.attn": ("flash", "sdpa"), "dit.attn_precision": ("ieee", "tf32", "tf32x3", "bf16"),
}

XL_FILE = "../EF2_XL_ADDON_v1/ef2_xl.py"                           # relative to the kit's driver/ like every kit_file
XL_LEVERS: Dict[str, Lever] = {                                      # the XL add-on's levers the memory mode composes (modes.XL_FAST_SET / XL_BIG_SET); `cond=chunk` (Tier 2) is never registered
    "x2b": Lever("x2b", FIELD_XL, XL_FILE, "z / relpos / lm_z storage released after their last consumer (FREE; single diffusion sample only)", "forward", "bitwise", "bitwise", ("xl", "free_events")),
    "x3": Lever("x3", FIELD_XL, XL_FILE, "diffusion-conditioning pair path with one [L,L,512] fp32 hidden alive (COND=lean)", "forward", "bitwise", "bitwise", ("xl", "cond_xl_calls")),
    "x6": Lever("x6", FIELD_XL, XL_FILE, "LM->pair projection row-chunked (LMPAIR)", "forward", "bitwise", "bitwise", ("xl", "s2p_xl_calls")),
    "x7": Lever("x7", FIELD_XL, XL_FILE, "relative-position one-hot encoding row-chunked (RELPOS)", "forward", "bitwise", "bitwise", ("xl", "relpos_xl_calls")),
    "x8": Lever("x8", FIELD_XL, XL_FILE, "pair-state init: the same trunc_normal_ RNG calls with in-place blocked selection (INITLEAN)", "forward", "bitwise", "bitwise", ("xl", "init_xl_calls")),
    "x10": Lever("x10", FIELD_XL, XL_FILE, "distogram logits moved to pinned host after the head (DISTOCPU)", "forward", "bitwise", "bitwise", ("xl", "disto_cpu_events")),
    "x4": Lever("x4", FIELD_XL, XL_FILE, "ESMC-6B streamed through the LM pass from the host (one block resident at a time) at inputs >= the line's token threshold (ESMC_OFFLOAD; the big default line: modes.XL_BIG_SET, X4_MIN_TOKENS)", "datapath", "bitwise", "bitwise", ("xl", "esmc_offload_events")),
}
LEVERS.update(XL_LEVERS)

RC_PKG = "esmfold2_opt.rowchunk"                                     # the LEVER line's impl= for the route's levers is the module path (blank-free), like tp's impl=esmfold2_opt.rowpair
RC_LEVERS: Dict[str, Lever] = {                                      # the row-chunking levers of the n_gpu > 1 route (rowpair.FOR_ROUTE): installed AFTER the pair stack is sharded (rowpair.install_rank ->
    #   rowchunk.install), so stack.classify defers them and stack.settle_route judges them from the installer's record; every one is a registry name the
    #   ablation variable can subtract alone (confbf16 / pdeskip / confmem / zbf16 ride on confrows: ablation.DEPENDS)
    "injrows": Lever("injrows", FIELD_RC, "esmfold2_opt.rowpair", "the Parcae recycle inject (a*z + Linear(LayerNorm(x2d))) issued per row block of the rank's pair shard, "
                     "residual in place (rowpair._inject_rows; from EF2_ROWPAIR_ROWCHUNK_MIN_TOKENS tokens, blocks of EF2_ROWPAIR_INJECT_MB)", "forward", "T2", "T2", ("rc", "injrows")),
    "confrows": Lever("confrows", FIELD_RC, RC_PKG + ".confrows", "the sharded confidence statement's LayerNorm -> head -> expectation pairs and its TM reduction issued per "
                      "row block into one preallocated logits tensor (blocks of EF2_ROWPAIR_CONF_MB)", "forward", "T2", "T2", ("rc", "confrows")),
    "confbf16": Lever("confbf16", FIELD_RC, RC_PKG + ".confrows", "the confidence head's pair activations [R,N,c_z] built per row block and STORED bf16 from the prologue on "
                      "(the confidence trunk's residual accumulates in bf16; the heads and the pooling upcast each row block, logits stay fp32); rides on confrows and confmem", "forward", "T2", "T2", ("rc", "confbf16")),
    "pdeskip": Lever("pdeskip", FIELD_RC, RC_PKG + ".confrows", "the PDE head is not evaluated on the row-sharded route: the fold result's pde / pde_logits are None "
                     "(nothing on the route reads or writes them); every kept value unchanged; rides on confrows", "forward", "bitwise", "T2", ("rc", "pdeskip")),
    "confmem": Lever("confmem", FIELD_RC, RC_PKG + ".confrows", "the confidence head's pLDDT / resolved weight gathers in atom blocks, the distogram bins and the fp32 pair rows "
                     "released at their last use (a split on the atom axis: no reduction is split); rides on confrows", "forward", "bitwise", "T2", ("rc", "confmem")),
    "biasfree": Lever("biasfree", FIELD_RC, RC_PKG + ".confmem", "the sampler's per-block pair-bias buffers released through ef2_opt.clear_graphs once "
                      "structure_head.sample() returns (same values, shorter life; the next fold allocates them again)", "serving", "bitwise", "T2", ("rc", "biasfree")),
    "zbf16": Lever("zbf16", FIELD_RC, RC_PKG + ".zbf16rows", "the confidence head reads a bf16 copy of the pair rows built per row band (blocks of EF2_ROWPAIR_ZBF16_MB); "
                   "z itself stays fp32 for the distogram and the sampler; rides on confrows", "forward", "T2", "T2", ("rc", "zbf16")),
}
LEVERS.update(RC_LEVERS)

# The cross-engine strategy id of every lever (the release tree's strategy families: F1 triangle attention · F2 triangle multiplication ·
# F3 graph/jit capture · F4 precision policy · F5 attention backends + MSA/attention kernels · F6 host side (caches, batching, data movement) ·
# F7 memory; LOCAL = engine-specific outside F1-F7). One id per lever: the per-lever LEVER line names it (report.lever_lines), so a
# run's log says which shared strategy was live in the process (ids are the tree's canonical set, opt_core STRATEGIES.json — the package test
# holds every entry against it; LOCAL.esmfold2.<name> where the tree has no id). ESMFold2 has no triangle attention: no lever carries F1.
STRATEGY: Dict[str, str] = {                                                  # canonical ids of common/opt_core/opt_core/STRATEGIES.json, or LOCAL.esmfold2.<name> where the tree has none
    "fused": "F4.autocast_policy",
    "tg": "F3.cuda_graph_trunk", "sg": "F3.cuda_graph_sampler", "eg": "F3.cuda_graph_trunk",
    "ec": "F6.feature_cache", "fc": "F6.feature_cache", "pb": "LOCAL.step_invariant_hoist", "msa": "LOCAL.esmfold2.msa_vendor_fused_trimul",
    "t1": "LOCAL.fused_transition", "t3": "LOCAL.esmfold2.tile_table",
    "t5": "F4.bf16_weight_precast", "t6": "LOCAL.fused_transition",
    "t10": "LOCAL.fused_transition", "mk": "LOCAL.step_invariant_hoist",
    "t11": "F5.fpf_msa_kernels", "t12": "F5.fpf_msa_kernels", "t13": "F5.fpf_msa_kernels", "t14": "LOCAL.layernorm_kernel",
    "ls": "LOCAL.esmfold2.loop_static_io", "rg": "F3.cuda_graph_trunk",
    "ax": "LOCAL.step_invariant_hoist", "af": "LOCAL.esmfold2.atom_block_fused", "fz": "LOCAL.esmfold2.msa_features_vectorised",
    "m15": "F5.fpf_msa_kernels", "m16": "F5.fpf_msa_kernels", "m17": "F5.fpf_msa_kernels", "mh": "LOCAL.step_invariant_hoist",
    "t15": "LOCAL.fused_transition", "t15msa": "LOCAL.fused_transition", "xtr": "LOCAL.fused_transition", "xte": "LOCAL.fused_transition",
    "t16": "LOCAL.fused_transition", "tx": "F2.fpf_trimul_fast",
    "trimul": "LOCAL.esmfold2.msa_trimul_replumb", "glue": "LOCAL.esmfold2.msa_trimul_replumb", "disto": "F6.output_overlap", "xln": "F5.row_layernorm",
    "ro": "F3.cuda_graph_sampler", "kd": "F6.host_sync_elimination", "dit": "LOCAL.dit_fused_kernels",
    "x2b": "F7.chunked_eval", "x3": "F7.chunked_eval",
    "x6": "F7.chunked_eval", "x7": "F7.chunked_eval", "x8": "LOCAL.esmfold2.lean_init", "x10": "LOCAL.esmfold2.disto_to_host",
    "x4": "LOCAL.esmfold2.lm_host_offload",
    "injrows": "F7.chunked_eval", "confrows": "F7.chunked_eval", "confmem": "F7.chunked_eval",
    "confbf16": "LOCAL.esmfold2.bf16_pair_rows", "zbf16": "LOCAL.esmfold2.bf16_pair_rows",
    "pdeskip": "LOCAL.esmfold2.pde_not_computed", "biasfree": "LOCAL.esmfold2.pair_bias_release",
}


SUPERSEDED_ON: Dict[str, Dict[str, str]] = {                      # {lever: {class: the lever that serves instead}}: on that class the set runs the superseding lever when it is in the set
    "t15": {"9.0": "t16"}, "t15msa": {"9.0": "t16"},               # (both are written in the server's table; modes.resolve keeps exactly one per class, named — an ablated t16 brings t15 / t10 back)
    "t10": {"9.0": "t16"},                                         # W4's fused transition: t16 serves every d=256 / h=1024 Transition on 9.0 — t10's kernel is never reached there (t10_calls stays 0)
}


def acts_on(lever: Lever, variant: Optional[str]) -> bool:
    """Whether the kit's own guards let this lever act on `variant` (None: unknown -> assume yes)."""
    return variant is None or variant in lever.variants


STEPS_ASIDE_ON: Dict[str, Dict[str, str]] = {}                  # {lever: {class: reason token}}: on that compute-capability class the lever is slower than the op it replaces, so
                                                                   # the set does not carry it there — it steps aside BY NAME (DRY-RUN / APPLIED not_for_class=<lever>, LEVER line
                                                                   # state=skipped reason=not_for_class:<class>:<token>), never silently. Empty: the pair TriMul's per-card row is the
                                                                   # shared core's cell decision under the tier word (lever tx), not a set subtraction


def class_word(cc: Optional[str]) -> str:
    """'9.0' -> 'sm90' (the LEVER lines' class vocabulary); None -> 'unknown'."""
    return ("sm" + str(cc).replace(".", "")) if cc else "unknown"


def class_excludes(lever: Lever, cc: Optional[str]) -> Optional[str]:
    """``serves_<classes>`` when the lever's kernels are another compute-capability class's (Lever.classes), the STEPS_ASIDE_ON token when the
    lever is measured slower than the op it replaces on class ``cc``, else None; None for an unknown class."""
    if cc and lever.classes is not None and str(cc) not in lever.classes:
        return "serves_" + "+".join(class_word(c) for c in lever.classes)
    if cc:
        return (STEPS_ASIDE_ON.get(lever.name) or {}).get(str(cc))
    return None


def superseded_on(lever: Lever, cc: Optional[str], in_set) -> Optional[str]:
    """``superseded_by_<x>`` when SUPERSEDED_ON names, for class ``cc``, a lever ``x`` that IS in ``in_set`` (the set after every other subtraction), else None."""
    sup = (SUPERSEDED_ON.get(lever.name) or {}).get(str(cc)) if cc else None
    return f"superseded_by_{sup}" if (sup and sup in in_set) else None


def not_on_class(lever: Lever, cc: Optional[str], in_set=()) -> Optional[str]:
    """The reason token why ``lever`` leaves the set on compute-capability class ``cc`` (None = it stays): the class exclusion, else the
    supersession against ``in_set``. An unknown class (no GPU visible, no target configured) subtracts nothing."""
    return class_excludes(lever, cc) or superseded_on(lever, cc, in_set or ())
