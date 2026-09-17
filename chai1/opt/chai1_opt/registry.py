"""The lever registry: one entry per lever the kit composes, keyed by the kit's own name — the driver kit's ``W1``, ``W2``, ``W5``
(``opt/forward/fast_inference``), the eager stack's ``tier1`` (``opt/forward/eager_trunk``, ``chai1_eager.stack.LEVERS``)
and the DSTEP add-on's ``hoist2`` / ``compiled`` / ``dit_attn`` (``opt/forward/dstep_megakernel``, ``chai1_fastln.stackx.ALL_LEVERS``).

It *describes* each lever — which kit file installs it, what it changes, its class under the four-class test (forward / datapath /
serving / orchestration), its numerics tier in the kit's own words (the driver kit's own ``README.md`` lever table
at ``opt/forward/fast_inference/README.md``; the eager stack's own ``README.md``), its switch (the driver's ``--levels`` list;
the eager stack's ``install(levers=...)``; no environment variable)
and the kit record that shows it ran (``probe``). No mode composition lives here: modes are ``modes.KIT_MODES``.

``probe`` grammar (read by stack.classify from the kits' own state):
  ("attr", <module>, <name>, <function name>)   the module attribute holds the kit's function (W1: chai1.load_exported ->
                                                 load_exported_cached, esm.esm_model -> esm_model_resident; W5:
                                                 esm._get_esm_contexts_for_sequences -> _get_ctx_cached). When the eager stack is
                                                 installed, chai1.load_exported holds ITS loader and the driver kit's function is the
                                                 one it wraps (StackHandle.orig): classify looks through the handle.
  ("row", <key>)                                 the per-seed-fold JSON row carries the key (W2: features_reused_from_seed)
  ("eager", <levers>)                            chai1.load_exported is the installed StackHandle's loader built for <levers>
                                                 (stack.py build_parts / EagerTrunkWrapper)
  ("dstep", <lever>)                             the installed handle's denoiser carries the add-on's lever (stackx.py build_lever_parts:
                                                 HoistedDiffusionWrapper.hoister hoist2 / .compile / .dit_attn; stack.dstep_installed_levers)
  ("flag", <torch attribute path>)               torch's own switch reads True (the package's opt-in lever tf32, precision.py:
                                                 torch.backends.cuda.matmul.allow_tf32)
  ("env", <variable>)                            the environment carries exactly the lever's value (alloc, alloc.py: PYTORCH_CUDA_ALLOC_CONF
                                                 = the core policy's configuration)
  ("big", <lever>)                             the memory mode's applied record names the lever (big.py over opt_core.mem: the line's
                                                 apply succeeded for it; its per-item census is the exit gate's)
  ("pairtrack", <lever>)                         pairtrack.applied() names the lever (its hook is installed on the eager trunk's classes / CFG;
                                                 its per-call ledger is the exit gate's)

The package's own opt-in levers (``OPTIN_LEVERS``; modes.OPTIN_LEVERS names the modes each may join) are described here like the kits':
they ride a row only as its implied levers (modes.KitMode.implied_optin).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple



@dataclass(frozen=True)
class Lever:
    name: str
    kit_file: str
    what: str
    cls: str                                   # four-class test
    tier: str                                  # the kit's own numerics reading (README.md lever table)
    default_on: bool                           # the driver's own default --levels
    probe: Tuple
    lines: str = ""
    doc: str = ""


LEVERS: Dict[str, Lever] = {
    "W1": Lever("W1", "kit/chai_worker.py",
                "resident modules: chai1.load_exported memoised (the 6 TorchScript components loaded once per process) and the traced "
                "ESM2-3B kept on the GPU between calls (esm.esm_model replaced)",
                "serving", "never in the maths (README.md lever table); byte-identical under the recipe as part of W1+W2+W5", True,
                ("attr", "chai_lab.chai1", "load_exported", "load_exported_cached"), lines="chai_worker.py: W1 block",
                doc="README.md lever table"),
    "W2": Lever("W2", "kit/chai_worker.py",
                "per-input feature context: make_all_atom_feature_context once per input, the same AllAtomFeatureContext passed to "
                "run_folding_on_context for every seed (built before set_seed in stock; seed-independent)",
                "datapath", "exact by construction — collated tensors bit-identical across seeds (README.md lever table)", True,
                ("row", "features_reused_from_seed"), lines="chai_worker.py: the per-input loop", doc="README.md lever table"),
    "W5": Lever("W5", "kit/chai_worker.py",
                "cross-input ESM memo: per-sequence ESM embeddings memoised across inputs in one process "
                "(esm._get_esm_contexts_for_sequences replaced)",
                "datapath", "the same embedding per sequence, memoised (README.md lever table); the traced module's first-call bits differ from its later "
                "calls under TorchScript profiling (KNOWN_ISSUES.md §2) — the recipe's profiling-mode item removes it", True,
                ("attr", "chai_lab.data.dataset.embeddings.esm", "_get_esm_contexts_for_sequences", "_get_ctx_cached"),
                lines="chai_worker.py: W5 block", doc="README.md lever table"),
}

EAGER_LEVERS: Dict[str, Lever] = {
    "tier1": Lever("tier1", "chai1_eager/stack.py",
                   "the eager stack: trunk.pt re-expressed as eager nn.Modules with copy-free triangle layouts (chai1_eager/trunk.py, "
                   "kernels.py), diffusion_module.pt transpiled op-for-op (ts2eager.py) with its step-invariant part hoisted once per "
                   "sample and the step replayed from a CUDA graph (hoist.py; un-graphed under the deterministic recipe by install's own rule — "
                   "no capture attempted), confidence head and embedders transpiled flat; chai1.load_exported replaced by the stack's loader",
                   "forward", "tier 1 on the pinned stack: the trunk bitwise to the scripted module and the whole model byte-identical to stock "
                   "under the deterministic recipe (eager README.md) — in the exact, fast and big rows; bitwise equality is per "
                   "torch / CUDA stack: on another stack exact runs with the STACK record",
                   False, ("eager", "tier1"),
                   lines="stack.py:install,build_parts; trunk.py; hoist.py; ts2eager.py", doc="eager README.md"),
}
DSTEP_LEVERS: Dict[str, Lever] = {
    "hoist2": Lever("hoist2", "chai1_eager/hoist.py",
                    "the hoisted denoiser step partitions the traced forward by VALUE dependence on (atom_noised_coords, noise_sigma): a statement "
                    "that reads only the metadata of a step-dependent tensor (torch.size / new_empty / prim.device — e.g. the diffusion-sample axis "
                    "length) no longer counts as step-dependent, so the atom-pair update block, the two N=16 atom-pair LayerNorms, the six blocked "
                    "pair-bias einsums + masked_fills and the AdaLN scale/shift/gate projections of the atom transformer run ONCE per sample in the "
                    "precompute (the full traced forward, on the first call's inputs) instead of 398 times per fold; the per-step part keeps 575 of "
                    "the base hoister's 1496 root statements (983 of 1356 kernels at crop 1024). Same ops, same operands, same order inside every "
                    "dependency chain; the hoist cache grows (1.6 -> 5.0 GB at crop 1024, 3.0 -> 8.2 GB at 1536)", "forward",
                    "tier 1: bit-exact (torch.equal to the base hoister's step and to the un-hoisted call under the deterministic recipe; byte-identical "
                    "CIFs / scores vs --mode off end to end under --det 1)", False, ("dstep", "hoist2"),
                    lines="hoist.py:HoistedForward2,partition_value; chai1_eager/stack.py:HoistedDiffusionWrapper(hoister=); stackx.py:build_lever_parts,HOIST_LEVERS",
                    doc="eager README.md; DSTEP README.md"),
    "compiled": Lever("compiled", "chai1_eager/hoist.py",
                      "the hoisted denoiser's PER-STEP function (hoist2's 575 statements) through torch.compile / Inductor, replayed from the same "
                      "CUDA graph: the pointwise / layout glue between the step's GEMMs (bias adds and masks on the [5,16,N,N] logits, AdaLN / gate / "
                      "SwiGLU chains, strided copies) fused into Triton kernels — crop 1024, H100: 983 -> 539 kernels, 55.4 -> 30.5 ms per step "
                      "(-41/-45/-43 %% at crops 512/1024/1536); GEMMs, the fp32 atom attention and LayerNorms unchanged. Compile cost on the first "
                      "calls per crop (cold ~270-290 s; warm = Inductor's on-disk caches under TORCHINDUCTOR_CACHE_DIR, keyed under MODEL_OPT_JIT_ROOT "
                      "by chai1_opt/jit.py); a compile that cannot engage on a box steps aside by name (compile_failed event, eager statements)", "forward",
                      "tier 2: NOT bitwise (fused reductions reorder fp32 sums): per denoiser call max-abs 1.2-1.9 A vs the eager TF32 step, the same size "
                      "as TF32's own deviation from fp32 (2-7 A max, 0.05-0.13 A rms at sigma 16); end to end inside the fast tier's band on the confident canaries",
                      False, ("dstep", "compiled"),
                      lines="hoist.py:HoistedForward.compile; chai1_eager/stack.py:HoistedDiffusionWrapper._run; stackx.py:COMPILE_LEVERS",
                      doc="eager README.md; DSTEP README.md"),
    "dit_attn": Lever("dit_attn", "chai1_fastln/dit_attn.py",
                      "the diffusion transformer's 16 pair-biased token attentions per denoiser step — stock calls scaled_dot_product_attention on 5-D "
                      "fp32 operands ([1,16,S,N,48] with a [1,16,1,N,N] fp32 bias shared by the S samples), which no fused kernel takes, so they run "
                      "SDPA's MATH path (materialised [S,16,N,N] logits: ~36 % of the eager step at crop 1024) — rebound inside the denoiser's transpiled "
                      "namespace to a 4-D view served by opt_core's pair-bias-attention provider (kernels.apb) asked by the MODE'S TIER WORD "
                      "(fast in fast, big in big; exact does not carry it): one selection per token count per process, the row the provider's "
                      "cell table names for (card, fp32, cell dit_h16d48, tokens, S samples, graphed); where that row is the library statement itself "
                      "(the provider's sdpa rows) the statement serves by name (Inductor fuses it: 30.1 vs 36.1 ms per step through the library kernel "
                      "at crop 1024, H100) — the row served per token count and any refusal are recorded; the atom transformer's "
                      "windowed attentions keep their statement; one opaque custom op under `compiled`", "forward",
                      "fast tier: NOT bit-exact (the memory-efficient kernel reorders the math path's softmax reductions; measured inside TF32's own "
                      "deviation: rel-RMS 7.5e-4 vs TF32's 2.5e-3 per step at crop 512)", False, ("dstep", "dit_attn"),
                      lines="dit_attn.py:Router,admits,_row_for_impl,row_decision,_serve4,patch_flat; stackx.py:ATTN_LEVERS,TIER_WORD,build_lever_parts", doc="DSTEP README.md"),
}
OPTIN_LEVERS: Dict[str, Lever] = {
    "tf32": Lever("tf32", "chai1_opt/precision.py",
                  "TF32 tensor-core products for every fp32 GEMM of the process (torch.backends.cuda.matmul.allow_tf32 = True): the transpiled "
                  "denoiser's linears and matmuls (fp32 as traced), the flat embedders' and the confidence head's; the eager trunk's bf16 linears "
                  "are unaffected; applied after the deterministic recipe and before the eager stack installs, so the hoisted step's CUDA graph "
                  "captures TF32 kernels",
                  "forward", "tier 2, inside stock's seed-to-seed band (implied by the fast and big rows); never bitwise to stock, so never under exact (modes.OPTIN_LEVERS)",
                  False, ("flag", "backends.cuda.matmul.allow_tf32"), lines="precision.py:apply,probe"),
    "alloc": Lever("alloc", "chai1_opt/alloc.py",
                   "torch's CUDA caching allocator with expandable segments: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True exported through "
                   "opt_core.mem.torch_alloc before the process's first CUDA allocation (the driver child's environment on the driver route; "
                   "this process before the eager stack installs on the environment route), refused by name when the variable names another "
                   "configuration or CUDA is already initialised; the allocator's own record read back at exit",
                   "serving", "placement only: no kernel input changes — outputs unchanged by construction; unmeasured on this tree — available, "
                   "not default; joins exact and fast",
                   False, ("env", "PYTORCH_CUDA_ALLOC_CONF"), lines="alloc.py:export,env_row"),
}

PAIRTRACK_LEVERS: Dict[str, Lever] = {                             # chai1_opt/pairtrack.py: the trunk levers on the eager line's structured trunk (modes.KitMode.pairtrack)
    "templ_empty": Lever("templ_empty", "chai1_opt/pairtrack.py",
                         "the template embedder returns z when no template slot carries a mask (every route folds templates_path=None): its own "
                         "arithmetic is z + proj_out(relu(sum_t LN(pairformer(z_t)) * 0)) = z, so the 2-block c=64 template pairformer over the "
                         "empty slots is not run; a call WITH templates runs the module, counted fallback:templates_present",
                         "forward", "exact by the module's own algebra (a bias-free projection of relu(0)): the same bits as the eager line", False,
                         ("pairtrack", "templ_empty"), lines="pairtrack.py:_templ_forward"),
    "v4trimul": Lever("v4trimul", "chai1_opt/pairtrack.py",
                      "the trunk's merged triangle multiplication (the pairformer's and the MSA module's c=d=256 pair stack; the template pair stack's "
                      "c=d=64 when templates are present) as two directions served by the shared core's ONE triangle-multiplication provider BY TIER "
                      "WORD (opt_core.trimul.by_word: fast -> `fast`, big -> `big`; the provider's measured cells name the row per card, precision, "
                      "c_z, crop bucket and direction — no kit cell table, no row pinned by name), outgoing with the pair mask, incoming with its "
                      "transpose, the two updates added; a call the provider refuses by name runs the eager module's statement, counted",
                      "forward", "tier 2 (the fused TriMul rows' rounding: different reduction trees from the bf16 merged-projection statement; the provider's "
                      "classes 'fast' / 'big' — no row is byte-vouched against this statement on the kit's stack, so `exact` keeps the statement by name)",
                      False, ("pairtrack", "v4trimul"), lines="pairtrack.py:_trimul_hook,_trimul_weights,_trimul_provider,_trimul_levers,trimul_word"),
    "exactln": Lever("exactln", "chai1_opt/pairtrack.py",
                     "every LayerNorm statement of the structured eager trunk (pairformer, MSA module, recycle projections: LN(x.to(fp32)).to(bf16), "
                     "TriMul's affine-free fp32 output LayerNorms) and the transpiled confidence head's fp32 LayerNorms bound to the shared core's LayerNorm "
                     "provider BY TIER WORD (opt_core.kernels.ln: exact -> 'exact', fast -> 'fast', big -> 'big'): each statement class asks the provider "
                     "once with its own (dtype, cell, width, rows) key and the row its measured cell names serves it — the ATen replica exactln (bitwise; the exact "
                     "word's row where vouched on the stack and at or above the statement's speed), the measured Triton rows under fast / big, or the statement "
                     "itself by name where the stock op is the cell's winner (statement:<row>)",
                     "forward", "exact word: bitwise (the replica of ATen's layer_norm forward arithmetic, bit-compared per class at run time; the statement by name elsewhere); "
                     "fast / big words: tolerance-class (the provider's measured rows, inside the identity band)", False, ("pairtrack", "exactln"), lines="pairtrack.py:install; chai1_exactln/serve.py:ln,bind_flat,cast_cached"),
    "triattn": Lever("triattn", "chai1_opt/triattn_core.py",
                     "the pairformer's, the MSA module's and the template pair stack's triangle attention (both directions, the pair bias with "
                     "the pair mask folded as the statement folds it) evaluated by the shared core's ONE triangle-attention provider "
                     "(opt_core.kernels.triattn) BY THE MODE'S TIER WORD at every crop, head dim and card - no kit cell table, no row pinned: the "
                     "provider's measured cell winner with its launch setting (call form bias_only, strided views of the statement's own projection "
                     "buffer - no copies, the same bias, scale and gating); where the provider names the stock op for a cell the tier1 line's own "
                     "SDPA statement serves BY NAME (fallback:stock_statement, counted); a provider refusal is named and refuses the exit gate",
                     "forward", "tier 2 (bf16 tensor-core products, fp32 accumulation and online softmax: the arithmetic class of the cuDNN / flash "
                     "SDPA backends the statement runs on, not their bits; the core's class 'fast')", False, ("pairtrack", "triattn"),
                     lines="triattn_core.py:make_impl,install; pairtrack.py:install"),
    "msa_pad": Lever("msa_pad", "chai1_opt/pairtrack.py",
                     "the MSA module's own statement on the leading MSA-row slices that carry any mask position instead of all 16384 padded rows: "
                     "the module on the first ceil(S_eff/8192)*8192 rows, its outer-product means on the first ceil(S_eff/4096)*4096 (the export's "
                     "slice grid: every kernel call of a kept slice is the whole statement's call; chai1_eager/msa_kernels.py)",
                     "forward", "exact: the skipped slices are all-masked (signed zeros added to the outer-product mean's running sum, never read "
                     "elsewhere); the first call of every shape class runs whole AND cut and is bit-compared (torch.equal), a differing class is "
                     "served whole and refuses the exit gate by name", False, ("pairtrack", "msa_pad"), lines="pairtrack.py:install; chai1_eager/msa_kernels.py:MsaPad"),
    "transition": Lever("transition", "chai1_opt/pairtrack.py",
                        "the trunk's four transition classes (pairformer pair 256 -> 512, MSA-module pair 256 -> 1024, MSA 64 -> 256, single 384 -> 768) "
                        "bound BY TIER WORD to the shared core's transition provider (opt_core.kernels.transition; chai1_eager/transition_core.py): every "
                        "call asks the mode's tier word (exact | fast | big) at its own cell and serves the row the provider resolves — on exact fed the "
                        "statement's own LayerNorm output; a stock answer, a class without a cell or a refusal by name keep the statement, counted",
                        "forward", "exact where the mode is exact: every served class is bit-compared against the statement at its first call and a differing "
                        "class stays on the statement by name (fallback:class_differs); tolerance-class rows on fast / big (first call per class "
                        "checked finite and inside rel-RMS 5e-2)", False, ("pairtrack", "transition"),
                        lines="pairtrack.py:install; chai1_eager/transition_core.py:TransitionBinding"),
    "trunk_n": Lever("trunk_n", "chai1_opt/trunk_n.py",
                     "the trunk call at the live-token extent: the ten token-indexed inputs of the eager trunk narrowed to N' = ceil64(live tokens) "
                     "(upstream pads to the crop ladder 256..2048: 1200 tokens run as 1536, 800 as 1024), the two trunk representations padded "
                     "back to the crop with zeros for upstream's per-crop embedders / diffusion module / confidence head (they mask the padded positions)",
                     "forward", "tolerance-class (fast / big, never exact): the trunk's reduction extents change with N' (the class of msa_rows); "
                     "no_pad at a crop boundary and layout when the live tokens do not lead, by name", False, ("pairtrack", "trunk_n"),
                     lines="pairtrack.py:install; trunk_n.py:TrunkN"),
}

MEMORY_LEVERS: Dict[str, Lever] = {                                # big.py's memory line (modes.KIT_MODES["big"].memory), mechanisms from opt_core.mem
    "msa_rows": Lever("msa_rows", "chai1_opt/big.py",
                      "the trunk's MSA module on the MSA rows that carry any mask (S_eff) instead of the 16384 rows upstream pads to: "
                      "msa_input_feats / msa_mask sliced at the trunk wrapper's call",
                      "forward", "measured: the dropped rows are all-masked (exact zeros in the outer-product mean, row-independent elsewhere); "
                      "cuBLAS may pick another kernel at the smaller extent — bitwise at small depths, within the fast band at deep MSAs",
                      False, ("big", "msa_rows"), lines="big.py:_slice_msa_rows"),
    "msa_chunk": Lever("msa_chunk", "chai1_opt/big.py",
                       "MSA pair-weighted averaging on blocks of 1024 MSA rows into one output plane (opt_core.mem.chunk.chunk_rows)",
                       "forward", "bitwise: row-local", False, ("big", "msa_chunk"), lines="big.py:_pwa_class"),
    "trunk_chunk": Lever("trunk_chunk", "chai1_opt/big.py",
                         "the pairformer's merged triangle multiplication on 256-row output blocks (opt_core.mem.chunk.triangle_multiplication_chunked) "
                         "and its triangle attention on 256-row query blocks (triangle_attention_chunked), through the eager trunk's CFG plugs, at N >= 1536",
                         "forward", "triangle multiplication bitwise vs the fast line's trimul_bmm; triangle attention NOT bitwise under the recipe "
                         "(SDPA per row block) — fast-class", False, ("big", "trunk_chunk"), lines="big.py:_trimul_chunked,_triattn_chunked"),
    "opm_chunk": Lever("opm_chunk", "chai1_opt/big.py",
                       "the MSA module's outer-product mean on 256-row pair blocks (opt_core.mem.chunk.chunk_rows), at N >= 1536",
                       "forward", "measured: op-level bitwise, in-situ within the fast band", False, ("big", "opm_chunk"),
                       lines="big.py:_opm_class"),
    "nograph": Lever("nograph", "chai1_opt/big.py",
                     "the hoisted denoiser step un-graphed (opt_core.mem.ckpt graph_capture policy): no CUDA-graph pool",
                     "serving", "bitwise: fast's own un-graphed form (its DET form)", False, ("big", "nograph"),
                     lines="big.py:_install_nograph,_gate_nograph,_check_nograph,graphed_override"),
}

DRIVER_LEVERS: Dict[str, Lever] = dict(LEVERS)                    # the driver kit's levers (the W names)

POSTPROC_LEVERS: Dict[str, Lever] = {                              # chai1_opt/postproc.py + featfast.py: the fold's host-side tail and the next item's feature build (modes.KitMode.postproc) — exact-class by construction, every kit row
    "rankcc": Lever("rankcc", "chai1_opt/postproc.py",
                    "upstream's per-sample ranker with its inter-chain clash census counted on the atoms that exist: upstream's own cdist statement (per-pair arithmetic, "
                    "same bits) < clash_threshold on existing distinct atoms, one integer bincount per (chain, chain) instead of two scatter_add_ passes over the padded "
                    "a x a int32 matrix (a = 23 x crop), then upstream's integer / ratio statements verbatim; pTM / ipTM / pLDDT unchanged; the process's first sample is "
                    "bit-compared with upstream's statement (a mismatch refuses the lever by name)",
                    "serving", "exact by construction: integer counts of a bit-identical predicate — the same scores npz bytes (README.md lever table)", True,
                    ("attr", "chai_lab.chai1", "rank", "rank_cc"), lines="postproc.py:clash_scores_cc", doc="README.md lever table"),
    "tailasync": Lever("tailasync", "chai1_opt/postproc.py",
                       "save_to_cif (modelcif, pure Python) and plot_msa (matplotlib) submitted in call order to one writer thread and joined before the fold's "
                       "StructureCandidates exists (a writer's exception re-raises there): sample k's CIF is written while sample k+1 is ranked; same functions, same "
                       "arguments, same bytes",
                       "serving", "never in the maths: the same writers with the same arguments, only when they run moves (README.md lever table)", True,
                       ("attr", "chai_lab.chai1", "save_to_cif", "save_to_cif_async"), lines="postproc.py:_make_writers", doc="README.md lever table"),
    "confmemo": Lever("confmemo", "chai1_opt/featfast.py",
                      "the reference-conformer library loaded once per process: upstream's load_chains_from_raw builds AllAtomResidueTokenizer("
                      "RefConformerGenerator()) on every call and the constructor unpickles the whole CCD conformer library (~5.5 s of the ~6.5 s "
                      "per-item feature build); the lever passes the process's one tokenizer (upstream's own constructors, on first use) whenever the "
                      "caller built none — read-only reference data, the same chains",
                      "serving", "exact by construction — the same objects upstream builds, once (W1's pattern; README.md lever table)", True,
                      ("attr", "chai_lab.chai1", "load_chains_from_raw", "load_chains_from_raw_confmemo"), lines="featfast.py:_make_chains", doc="README.md lever table"),
    "prefetch": Lever("prefetch", "chai1_opt/featfast.py",
                      "the NEXT item's make_all_atom_feature_context (FASTA parse, reference conformers, MSA parquet load + pairing; CPU, no RNG draw) built on one helper "
                      "thread while the current item folds, matched to the worker's later call by FASTA content + bound arguments (a miss computes inline, counted); "
                      "stands aside by name under the MSA / template server modes, with ESM embeddings on, and outside the kit driver",
                      "datapath", "exact by construction — upstream's own feature build on the same bytes and arguments; e2e only, the fold is unchanged (README.md lever table)", True,
                      ("attr", "chai_lab.chai1", "make_all_atom_feature_context", "make_all_atom_feature_context_prefetch"), lines="featfast.py:_make_prefetch", doc="README.md lever table"),
}

LEVERS.update(EAGER_LEVERS)                                       # every lever the kit composes or carries, by the kits' own names
LEVERS.update(DSTEP_LEVERS)
LEVERS.update(OPTIN_LEVERS)                                       # + the package's own opt-in levers
LEVERS.update(PAIRTRACK_LEVERS)                                   # + the trunk levers on the eager line (pairtrack.py)
LEVERS.update(POSTPROC_LEVERS)                                    # + the fold's host-side tail / feature-build levers (postproc.py, featfast.py)
LEVERS.update(MEMORY_LEVERS)                                      # + big's memory line

FAMILY: Dict[str, str] = {"compiled": "F3", "dit_attn": "F5", "hoist2": "LOCAL", "rankcc": "F6", "tailasync": "F6", "confmemo": "F6", "prefetch": "F6",                                         # the strategy-family id of each lever (F1 triatt · F2 trimul · F3 capture · F4 precision · F5 attn · F6 host · F7 memory · LOCAL), named on its LEVER line
    "W1": "F6", "W2": "F6", "W5": "F6",
    "tier1": "F3",
    "msa_rows": "F7", "msa_chunk": "F7", "trunk_chunk": "F7", "opm_chunk": "F7", "nograph": "F7",
    "tf32": "F4", "alloc": "F7",
    "templ_empty": "LOCAL", "v4trimul": "F2", "exactln": "F5", "msa_pad": "LOCAL", "transition": "LOCAL", "trunk_n": "LOCAL",
    "triattn": "F1",
}
assert set(FAMILY) == set(LEVERS), set(FAMILY) ^ set(LEVERS)
ORIGIN: Dict[str, str] = {n: ("core" if n in ("alloc", "msa_chunk", "trunk_chunk", "opm_chunk", "nograph", "v4trimul", "triattn", "exactln") else "kit") for n in LEVERS}   # where the lever's mechanism lives: the shared core's module served through this kit's adapter, or the kit's own files
STRATEGY: Dict[str, str] = {"compiled": "F3.inductor_step", "dit_attn": "F5.apb_dit", "hoist2": "LOCAL.chai1.hoist2", "rankcc": "LOCAL.chai1.rankcc", "tailasync": "F6.output_overlap", "confmemo": "F6.weights_residency_init", "prefetch": "F6.item_ordering_prefetch",                                          # the shared strategy table's canonical id (common/opt_core/opt_core/STRATEGIES.json) or LOCAL.chai1.<name>; printed on the LEVER line
    "W1": "F6.weights_residency_init", "W2": "F6.feature_cache", "W5": "F6.feature_cache", "tier1": "LOCAL.chai1_ts2eager",
    "tf32": "F4.autocast_policy", "alloc": "F7.expandable_segments",
    "msa_rows": "LOCAL.chai1.msa_rows", "msa_chunk": "F7.chunked_eval", "trunk_chunk": "F7.chunked_eval", "opm_chunk": "F7.chunked_eval",
    "nograph": "F7.graph_pool_budget",
    "templ_empty": "LOCAL.deadskip", "v4trimul": "F2.trimul", "exactln": "F5.row_layernorm", "msa_pad": "LOCAL.chai1.msa_pad", "transition": "LOCAL.chai1.transition", "trunk_n": "LOCAL.chai1.trunk_n",
    "triattn": "F1.flash_triatt",
}
assert set(STRATEGY) == set(LEVERS), set(STRATEGY) ^ set(LEVERS)

ATTR_PROBES = {n: lv.probe for n, lv in LEVERS.items() if lv.probe[0] == "attr"}
EAGER_PROBES = {n: lv.probe for n, lv in LEVERS.items() if lv.probe[0] == "eager"}
DSTEP_PROBES = {n: lv.probe for n, lv in LEVERS.items() if lv.probe[0] == "dstep"}
FLAG_PROBES = {n: lv.probe for n, lv in LEVERS.items() if lv.probe[0] == "flag"}
ENV_PROBES = {n: lv.probe for n, lv in LEVERS.items() if lv.probe[0] == "env"}
BIG_PROBES = {n: lv.probe for n, lv in LEVERS.items() if lv.probe[0] == "big"}
PAIRTRACK_PROBES = {n: lv.probe for n, lv in LEVERS.items() if lv.probe[0] == "pairtrack"}
