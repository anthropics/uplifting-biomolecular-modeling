"""The lever registry: one entry per lever of the RoseTTAFold3 kit line, keyed by a house name and carrying the kit's own switch.

This registry only *describes* a lever — its switch (the kit's environment variable), the kit files that implement it, its class
(forward / datapath / serving / memory), its numerics tier against stock, and the kit record that
proves it acted in a process (``probe``). No switch value, mode composition or tree state lives here: modes are rows in modes.py,
tree states are sha sets in tree.py, and the kit's own ``rf3.graph_flags`` reads the switches at import.

``probe`` grammar (read by report.tally from ``rf3.graph_flags`` and, for the FPF levers, from the adapter's own counters after a fold):
  ("describe", <key>, <value>)   graph_flags.describe()[key] == value
  ("stats", <key>)               graph_flags.HOIST_STATS[key] > 0
  ("capture", None)              graph_flags.LAST_CAPTURE carries a capture record (capture_ms or n_replay)
  ("fpf_served", None)           fpf_rf3_adapter.describe()["served"] > 0 (the trimul kernel served at least one call)
  ("fpf_count", <kind>, <key>)   sum of fpf_rf3_adapter.describe_v2()["counts"][kind][k] over keys k starting with <key> > 0
  ("fpf_v2", <section>, <key>)   fpf_rf3_adapter.describe_v2()[section][key] > 0
  ("fpf_v2_sum", <section>, (<key>, …))   the sum of those keys of describe_v2()[section] > 0 (a lever counts as reached when its
                                 own size gate declined the call: the gate is the lever's, recorded in its counters)
  ("mem", <key>)                 mem.report()[key] > 0 (the package's own memory-policy counters, read by report.tally into ``mem``)
  ("dtk", <key>)                 dtk.describe()["census"][key] > 0 (the package's dtk lever: its size gate's census, read by report.tally into ``dtk``)
  ("big", <lever>)             big.state()["census"][lever] closed with units > 0 (the memory levers of the big mode, read by report.tally into ``big``)
  ("rowpair",)                   rowpair.state()["installed"] with n_gpu > 1 (the row-sharded pair stack, read by report.tally into ``rowpair``)
The FPF levers have no environment switch: an arm component of ``modes.KitMode.fpf_arm`` (the adapter's grammar) turns one on, and
``modes.LEVERS_OF_FPF`` maps a component to these names.
"""
from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

from . import _core

CLASS_FORWARD, CLASS_SERVING, CLASS_MEMORY = "forward", "serving", "memory"


@dataclass(frozen=True)
class Lever:
    name: str                      # house name (modes.LEVERS_OF_SWITCH / LEVERS_OF_FPF map a switch / an arm component to these)
    switch: Optional[str]          # the kit's environment variable ("1" turns it on); None for a lever without a switch
    kit: str                       # which carried kit implements it (relative to opt/)
    kit_files: Tuple[str, ...]     # the kit files that implement it (relative to that kit directory)
    what: str
    cls: str                       # forward | datapath | serving | memory (CLASS_*)
    tier: str                      # numerics class against stock under the deterministic recipe
    probe: Tuple                   # see module docstring
    doc: str = ""
    wired: bool = True             # driven by `pred` (True); False = a lever no command of this package drives
    component: Optional[str] = None   # the FPF arm component that turns it on (modes.FPF_COMPONENTS); None for a switch lever
    strategy: str = ""                 # the cross-engine strategy id this lever implements (the release tree's shared vocabulary: F<k>.<name> for a
                                       # family strategy, LOCAL.<name> for an engine-specific one) — printed in the per-lever LEVER line (report.lever_lines)


ADDON = "forward/rf3_xattempt_addon"
PACKAGE = "rosettafold3_opt"                              # a package lever: implemented by this package, not by a carried kit
FPF = "forward/rf3_fpf_trimul_addon"
MKDIT = "forward/rf3_mk_dit_addon"                        # the MK-DiT add-on: the megakernel the package lever mkdit drives (mkdit.ADDON)
_FPF_ADAPTER = "rf3fpf/fpf_rf3_adapter.py"
_FPF_TIER2 = "Tier 2 (same numerics class as stock: bf16 operands, fp32 accumulation / LN statistics; run-to-run bitwise; not bitwise vs stock)"

LEVERS: Dict[str, Lever] = {
    "graph": Lever("graph", "RF3_CUDAGRAPH", ADDON,
                   ("patched/rf3/diffusion_samplers/inference_sampler.py", "patched/rf3/graph_flags.py",
                    "patched/rf3/model/layers/af3_diffusion_transformer.py", "patched/rf3/loss/loss.py"),
                   "pre-drawn RNG + one torch.cuda.CUDAGraph per sampler call (per design and seed), replayed over the diffusion steps",
                   CLASS_FORWARD, "Tier 1 (byte-identical under the deterministic recipe)", ("describe", "RF3_CUDAGRAPH", "1"),
                   doc="graph_flags.py; the [rf3_cudagraph] edits of the patched files"),
    "graph_safe_ops": Lever("graph_safe_ops", None, ADDON, ("patched/rf3/model/layers/af3_diffusion_transformer.py",),
                            "dense/slice rewrites of the eval-branch masked ops of the atom attention encoder and the window compaction "
                            "of the atom transformer (needed for capture); on iff RF3_CUDAGRAPH=1 (graph_flags.GRAPH_SAFE_OPS)",
                            CLASS_FORWARD, "Tier 1 (bitwise-equal rewrites)", ("describe", "RF3_GRAPH_SAFE_OPS", True),
                            doc="graph_flags.py GRAPH_SAFE_OPS"),
    "hoist": Lever("hoist", "RF3_HOIST", ADDON,
                   ("patched/rf3/graph_flags.py", "patched/rf3/diffusion_samplers/inference_sampler.py", "patched/rf3/model/RF3_structure.py",
                    "patched/rf3/model/layers/af3_diffusion_transformer.py"),
                   "the step-invariant sub-graphs of the denoiser (pair conditioning, atom-encoder conditioning, every block's pair bias) "
                   "computed once per roll-out with the unchanged upstream modules and reused for the remaining denoiser calls",
                   CLASS_FORWARD, "Tier 1 (byte-identical under the deterministic recipe)", ("stats", "entries_last"),
                   doc="graph_flags.py:38-41, 56-72; README.md:3"),
    # the FPF add-on: class-level monkey-patches of the trunk applied by its adapter (no file or weight edit; site-packages untouched)
    "fpf_trimul": Lever("fpf_trimul", None, FPF, (_FPF_ADAPTER,),
                        "rf3 TriangleMultiplication.forward served through the shared core's TriMul provider (opt_core.kernels.trimul) under its FAST TIER word "
                        "(arm word fast.fast: the cell winner per shape and card — c_z = d_hidden = 128, the 48 pairformer blocks, the MSA module, the confidence head: "
                        "the generic trimul kernel opt_core.kernels.fpf_trimul_v4 with the core's one launch-cell table) from the provider table's token floor for the call's class "
                        "(kernels.trimul.v4_n_min: 2 tokens on cc 9.0, 101 elsewhere); the template track's (64, 64) modules and calls below that floor stay on the "
                        "stock statement (the xmul lever's site), counted; a row word (fast.v4 / fast.tmk3_fast / fast.esm_v5_fwd ...) serves exactly that row for measurement and steps aside by name",
                        CLASS_FORWARD, _FPF_TIER2, ("fpf_served", None), doc="fpf_rf3_adapter.py _fast_forward / enable", component="fast"),
    "fpf_gflash": Lever("fpf_gflash", None, FPF, (_FPF_ADAPTER,),
                        "triangle attention: fused LN+cast+transpose prologue and one concatenated q|k|v|g GEMM (lnl_fused.ln_linear) around the "
                        "shared core's triangle-attention provider (opt_core.kernels.triattn) -- the attention core is the running device's row "
                        "(the provider's FAST tier word on every device: its cells name the row per card -- the triattn-native / sm_90a CUDA kernels on 9.0, the v11 CUDA kernel or K2B on 8.0, "
                        "its own named fallback elsewhere; an arm sub-word gflash.<word> names one row; a row the provider refuses is bound to the "
                        "cell's fallback BY NAME in the SELECT / CENSUS lines), gate applied in the kernel epilogue",
                        CLASS_FORWARD, _FPF_TIER2, ("fpf_count", "triattn", "served:"), doc="fpf_rf3_adapter.py _tri_forward_v2", component="gflash"),
    "fpf_ttr": Lever("fpf_ttr", None, FPF, (_FPF_ADAPTER,),
                     "the pair / MSA transitions through the shared core's transition provider (opt_core.kernels.transition) under the arm's TIER word fast | big: "
                     "its measured row per width, size and card (the fpf transition v2 kernel at the 128- and 64-wide transitions today), the residual add folded where the res component asks; "
                     "a width whose cell names the statements (the 384-wide single transition) runs the module's own forward BY NAME",
                     CLASS_FORWARD, _FPF_TIER2, ("fpf_count", "transition", "served:"), doc="fpf_rf3_adapter.py _transition_forward_v2", component="ttr"),
    "fpf_apb": Lever("fpf_apb", None, FPF, (_FPF_ADAPTER,),
                     "attention-pair-bias (AttentionPairBiasPairformerDeepspeed): the block's LayerNorm + bias as one Triton kernel, eager otherwise (no compile)",
                     CLASS_FORWARD, _FPF_TIER2, ("fpf_count", "apb", "served:triton"), doc="fpf_rf3_adapter.py _apb_forward_v2 (triton)", component="apb"),
    "fpf_sapb": Lever("fpf_sapb", None, FPF, (_FPF_ADAPTER,),
                      "attention-pair-bias with the stock ops byte-for-byte; the per-call torch.tensor(c).to(device) copy replaced by a cached identical "
                      "tensor so the block is CUDA-graph capturable",
                      CLASS_FORWARD, "Tier 1 (stock ops, bitwise)", ("fpf_count", "apb", "served:safe"),
                      doc="fpf_rf3_adapter.py _apb_forward_v2 (safe)", component="sapb"),
    "fpf_xmul": Lever("fpf_xmul", None, FPF, (_FPF_ADAPTER, "rf3fpf/fpf_rf3_trimul_rows.py"),
                      "the exact tier's triangle multiplication: the stock TriangleMultiplication statement untouched, its cuequivariance call served by the shared "
                      "core's provider (opt_core.kernels.trimul) exact row — tmk3_exact where the core's cell table measured it bitwise == the library op and >= x1.00 "
                      "on this card (c 128 up to its measured size; the c=64 template pair track at every size), else the stock op BY NAME (cueq); arm sub-word xmul.<row>",
                      CLASS_FORWARD, "Tier 1 (the stock op's bits: torch.equal per call on cc 9.0 and 8.0; the stock op itself elsewhere)", ("fpf_v2", "xmul", "served"),
                      doc="fpf_rf3_trimul_rows.py serve_xmul / CuetTrimulProxy", component="xmul"),
    "fpf_xln": Lever("fpf_xln", None, FPF, (_FPF_ADAPTER, "rf3fpf/fpf_rf3_ln_rows.py"),
                     "every nn.LayerNorm call of the model served by the shared core's provider (opt_core.kernels.ln): word exact (default) = the provider's exact "
                     "tier per cell — exactln / exactln:widen where its table measured them bitwise == ATen and >= x1.00 on this card (the c=128 / c=64 / c=256 pair "
                     "tensors' norms in fp32 and in autocast's widen form), ATen BY NAME at the floor cells (N-row norms in eager); fast | big | a row word "
                     "otherwise (arm sub-word xln.<word>); forms the provider does not serve with stock's semantics run the module's own forward by name",
                     CLASS_FORWARD, "Tier 1 (ATen's bits: torch.equal per call on cc 9.0 and 8.0 under word exact; tolerance rows only under xln.fast|big|fastln|ln_rows)",
                     ("fpf_v2", "xln", "served"), doc="fpf_rf3_ln_rows.py _ln_forward", component="xln"),
    "fpf_tg": Lever("fpf_tg", None, FPF, (_FPF_ADAPTER,),
                    "the 48-block pairformer stack captured once per token count (eager warm-up first) and replayed as one torch.cuda.CUDAGraph per "
                    "recycle (Recycler.forward); the kernels replayed are the arm's own, so it inherits the tier of the arm's other components; under the "
                    "token budget by kernel class (tgbudget.py: 1000 tokens on stock kernels, 300 on the FPF-fast kernels — above it the pre-graph forward, a named skip)",
                    CLASS_FORWARD, "Tier 1 on stock kernels (bitwise under the deterministic recipe, the add-on's report §5a); the tier of the arm otherwise",
                    ("fpf_v2", "trunk_graph", "replays"), doc="fpf_rf3_adapter.py _recycler_forward_tg", component="tg"),
    "fpf_xatt": Lever("fpf_xatt", None, FPF, (_FPF_ADAPTER, "rf3fpf/fpf_rf3_triattn.py"),
                      "FPF arm component xatt: the EXACT tier's triangle attention (the stock statement's cuet.triangle_attention call, bias only) through the shared "
                      "core's provider by the word exact — the stock op by name, or exact_headsplit (the stock op called per head, bitwise) where the core's table "
                      "names it (cc 8.0: above 800 tokens, x1.05 / 1.35 at 1200 / 2048); the "
                      "template track (head_dim 64) and sequences at or below cuEquivariance's own reference-path threshold stay on the stock op by name; a row the "
                      "provider refuses is bound to the stock op by name on the census line. Bitwise: composed exact == --mode off on the four det canaries (76/76 files)",
                      CLASS_FORWARD, "Tier 1 (bitwise: the stock op's forward reproduced bit for bit, or the stock op by name)", ("fpf_v2", "xatt", "served"),
                      doc="rf3fpf/fpf_rf3_triattn.py CuetProxy / opt_core.kernels.triattn word=exact", component="xatt", strategy="F1.triattn_exact"),
    "fpf_msa": Lever("fpf_msa", None, FPF, (_FPF_ADAPTER, "rf3fpf/fpf_rf3_msa_rows.py"),
                     "the MSA module's two pair statements on the shared core's fused Triton cells opt_core.ops.msa_opm / msa_pwa (arm component msa[.<word>]): pwa = MSAPairWeightedAverage "
                     "(fused LN -> v|gate prologue, the per-head softmax-weights contraction with the gate and the to_out projection in the epilogue, no [S,I,256] intermediates), "
                     "opm = OuterProductMean_AF3 (the outer-product GEMM over the MSA depth with the 1024 -> 128 projection in the epilogue: the [I,I,1024] outer product never "
                     "reaches HBM); the bare component binds the running card's row (fpf_rf3_msa_rows.CARD_ROWS: 9.0 both cells, 8.0 pwa -- the opm cell's 8.0 tile row is slower "
                     "than cuBLAS and steps aside by name); a call outside the cells' dims / dtype / rank runs the class's own forward, counted fallback:<reason>",
                     CLASS_FORWARD, _FPF_TIER2, ("fpf_v2", "msa", "served"), doc="fpf_rf3_msa_rows.py _opm_forward / _pwa_forward", component="msa"),
    "fpf_smsa": Lever("fpf_smsa", None, FPF, (_FPF_ADAPTER, "rf3fpf/fpf_rf3_msa_rows.py"),
                      "the exact tier's MSA pair-weighted averaging (arm component smsa): RF3's own pair-side statements (norm, bias projection, softmax) and the shared core's "
                      "exact-replica kernels (opt_core.ops.msa_pwa2) on the MSA side (LN -> v|gate, the K = I contraction issued in a cuBLAS summation structure, the output projection), served per "
                      "(I, S) class only after the class's first call compared every candidate structure torch.equal against the stock statements on the real activations and "
                      "locked the equal one; a class without an equal candidate, and every call below the size floor (S < 16 or S*I < 65,536 rows), runs the stock statements BY "
                      "NAME (bitcmp_failed:<I>x<S> / floor)",
                      CLASS_FORWARD, "Tier 1 (the stock statements' bits: torch.equal per (I, S) class at its first call on the running card, the stock statements by name otherwise)",
                      ("fpf_v2", "smsa", "served"), doc="fpf_rf3_msa_rows.py _pwa_exact_forward", component="smsa"),
    "fpf_res": Lever("fpf_res", None, FPF, (_FPF_ADAPTER,),
                     "PairformerBlock.forward with the residual adds of the trimul and transition outputs fused into those kernels' epilogues "
                     "(3 of the 5 [N,N,128] residual passes per block); the same add semantics as torch",
                     CLASS_FORWARD, _FPF_TIER2, ("fpf_v2", "res", "fused"), doc="fpf_rf3_adapter.py _pf_block_forward_res", component="res"),
    "fpf_dattn": Lever("fpf_dattn", None, FPF, (_FPF_ADAPTER,),
                       "the diffusion transformer's attention with pair bias as one scaled_dot_product_attention call (size-gated: stock math below "
                       "400 tokens); AttentionPairBiasDiffusion.forward re-compiled with its attention block swapped",
                       CLASS_FORWARD, _FPF_TIER2, ("fpf_v2_sum", "dattn", ("calls", "fallback")), doc="fpf_rf3_adapter.py _fpf_dattn / enable_dattn", component="dattn"),
    "dtk": Lever("dtk", None, PACKAGE, ("dtk.py",),
                 "the diffusion transformer's pair-biased token attention (AttentionPairBiasDiffusion, every diffusion-transformer block of every denoiser step "
                 "and sample) on the shared core's Triton flash-attention-with-bias kernel (opt_core.kernels.dtk_kernels.flash_bias_attn) above a size gate "
                 "(dtk.MIN_I = 400 tokens); the stock einsum block below it, counted",
                 CLASS_FORWARD, "Tier 2 (online softmax: same class as SDPA, not bitwise with the einsum path)", ("dtk", "served"),
                 doc="dtk.py; the seam: the FPF add-on's dattn source rewrite (rf3fpf/fpf_rf3_adapter.py _DATTN_BLOCK); the big composition (mkdit's place)"),
    "warm": Lever("warm", None, ADDON, ("patched/rf3/graph_flags.py", "patched/rf3/diffusion_samplers/inference_sampler.py"),
                  "the sampler roll-out's eager warm-up once per process: GRAPH_WARMUP (3) steps before the process's first CUDA-graph capture, 1 before "
                  "every later one (graph_flags.warmup_steps / set_levers; the FPF arm's lever sub-step @L1.warm switches it) — the warm-up reads and writes "
                  "only the static clones re-initialised after capture and consumes no draw",
                  CLASS_FORWARD, "Tier 1 (bitwise by construction; exact == off under the deterministic recipe on the canaries)", ("graph_levers", "warm"),
                  doc="graph_flags.py GRAPH_WARM; fpf_rf3_adapter.py LEVER_SUBS"),
    "hostlean": Lever("hostlean", None, PACKAGE, ("hostlean.py",),
                      "validation_step's two symmetry resolutions (rf3.loss.af3_losses Subunit/ResidueSymmetryResolution: Python loops with boolean-index host "
                      "syncs and device-to-host copies, thousands per item) return their input when no configured metric declares a ground-truth input "
                      "(Metric.kwargs_to_compute_args: the inference engine's ptm / iptm / count_clashing_chains read none); a metric that reads ground truth "
                      "keeps them, by name; the training step's loss-path resolutions are untouched",
                      CLASS_FORWARD, "Tier 1 (output-identical by construction: the statements skipped feed nothing a consumer reads)", ("hostlean", "skipped"),
                      doc="hostlean.py"),
    "prefetch": Lever("prefetch", None, PACKAGE, ("prefetch.py", "sidecar.py"),
                      "the persistent featurizer (non-forward): RF3InferenceEngine.run's per-item Transform pipeline (atomworks parsing, MSA loading with its "
                      "per-item multiprocessing pool, templates, reference conformers, atomization: 6-13 s per item at 400-1200 tokens) runs for the NEXT item in "
                      "one helper process forked at BaseInferenceEngine._construct_pipeline (before the trainer, the model and the CUDA context exist) while the "
                      "current item's forward runs. Exact by construction and witnessed at run time: item 0 is featurized in-process AND by the helper and compared "
                      "leaf by leaf (first_input=same, else the lever steps aside by name); every later item is answered only when the helper started it from the "
                      "RNG states (random / numpy / torch CPU) the fold process has at the request — the helper mirrors those states and predicts the forward's "
                      "CPU-generator draws (the sampler's per-step rotation / translation pattern, learnt once: shadow=TxD) — a mismatch re-featurizes in request "
                      "order (respec) or serves the atomworks Compose head and runs the torch-drawing tail in-process (split); n_items=1, start_refused, helper "
                      "errors: the stock statement in-process, by name; n_gpu>1: declined at activation (conflict:n_gpu, like confhoist) pending a measured x2 run. Host memory: one more process holding the pipeline (tree RSS roughly x2, CoW-shared)",
                      "datapath", "Tier 1 (output-identical by construction: the same statements on the same objects from the same RNG state; the first-item "
                      "comparison and the per-item state equality are the run-time witnesses)", ("prefetch", "helper"),
                      doc="prefetch.py"),
    "awrite": Lever("awrite", None, PACKAGE, ("awrite.py", "sidecar.py"),
                    "the asynchronous writer (non-forward): run()'s per-item writers (dump_ranking_scores, dump_top_ranked_outputs, RF3Output.dump per sample: "
                    "the CIF serialisations and JSON dumps, 2-14 s per item at 400-1200 tokens) execute in one side process forked with the featurizer (before "
                    "CUDA) on the item's own RF3Output objects while the fold process moves to the next item; acknowledgements drained per submit, every file "
                    "flushed before run() returns (flush_wait_ms); a failed or unpicklable call re-runs the stock statement in-process with the traceback "
                    "(errors=); a finite-value census of coordinates / summary confidences rides the hand-off (nonfinite=, a report word); start_refused, writer "
                    "gone: the stock writers in-process, by name; n_gpu>1: declined at activation (conflict:n_gpu) pending a measured x2 run",
                    CLASS_SERVING, "Tier 1 (the upstream writer functions on equal objects: the same bytes, written later; flushed before exit)", ("awrite", "files_done"),
                    doc="awrite.py"),
    "xtr": Lever("xtr", None, PACKAGE, ("pf.py",),
                 "the SwiGLU transitions (Transition, class-wide: pair z_transition + single s_transition of every pairformer / MSA-module / confidence "
                 "block) through the shared core's transition provider (opt_core.kernels.transition) under its EXACT TIER word: per call the provider names the row it has "
                 "vouched BITWISE on this stack at this size (v1: the module's own LayerNorm, then one kernel W_out(silu(W_a n)*W_b n) with the hidden on chip) or the statements "
                 "BY NAME (the 384-wide single transition, the c=64 tracks, sizes below its vouch); no exactness table in this kit",
                 CLASS_FORWARD, "Tier 1 candidate (the core's construction is bitwise equal to the unfused bf16 SwiGLU on 9.0)", ("xtr", "served"),
                 doc="pf.py; the exact mode"),
    "mkdit": Lever("mkdit", None, PACKAGE, ("mkdit.py",),
                   "the diffusion module's 24-block token diffusion transformer (DiffusionTransformer, the Beta_II-is-None calls of every denoiser step) on the "
                   "carried MK-DiT add-on's persistent Triton megakernel (forward/rf3_mk_dit_addon/mkdit/mk2.py MK2TokenTransformer: one launch per sample per "
                   "call, the per-block pair bias laid out once per roll-out through the RF3_HOIST cache) above a size gate (mkdit.MIN_I = 400 tokens; "
                   "the stock blocks below it, counted); refuses by name without a visible GPU of compute capability >= 9.0",
                   CLASS_FORWARD, "Tier 2 (bf16 operands / fp32 accumulation like stock, exp2-domain online softmax, one tile configuration per card; deterministic; "
                   "not bitwise with the stock blocks)", ("mkdit", "served"),
                   doc="mkdit.py + the carried kernels forward/rf3_mk_dit_addon/mkdit/{mk2,mkrf3}.py; the fast mode; dtk serves the same blocks' attention in big"),
    "confhoist": Lever("confhoist", None, PACKAGE, ("confhoist.py",),
                       "the confidence head's sample-invariant prologue (ConfidenceHead.forward: the fp32 casts and whole-tensor layer norms of the trunk's "
                       "S / Z / S_inputs, the two S_inputs projections and their outer-sum add into the pair track) computed once per prediction and reused by "
                       "every diffusion sample's confidence call (upstream's statements, executed once; the cache is bounded to the item)",
                       CLASS_FORWARD, "Tier 1 (bitwise: the same kernels on the same operands, once instead of D times)", ("confhoist", "reused"),
                       doc="confhoist.py; refuses by name when ConfidenceHead.forward's source is not the pin's"),
    "confln": Lever("confln", None, PACKAGE, ("confhoist.py",),
                    "the confidence head's three whole-tensor layer norms (F.layer_norm over every element of S / Z / S_inputs, no affine: one row of the "
                    "row-wise kernel, a single thread block) computed as (x - mean) * rsqrt(var + eps) from torch.var_mean's grid-wide reduction inside "
                    "confhoist's prologue (same formula, fp32, another summation order)",
                    CLASS_FORWARD, "Tier 2 (same function, reordered fp32 reduction; judged by the identity form)", ("confln", "calls"),
                    doc="confhoist.py (_whole_layer_norm); needs confhoist (refused by name without it)"),
    "mem": Lever("mem", None, PACKAGE, ("mem.py",),
                 "the caching allocator's free blocks released at the kit's roll-out seams (rf3.graph_flags.hoist_begin / hoist_end): the trunk's "
                 "blocks (main stream), the graph warm-up's (side stream) and the deleted graph's pool cannot serve each other, so reserved "
                 "memory otherwise grows to the card before the allocator frees its cache; plus the allocator's expandable segments and reclaim "
                 "threshold, and the add-on's trunk-graph cache budget on tg arms (mem.POLICY, every kit row)",
                 CLASS_MEMORY, "not a numerics lever (allocator only: no tensor, kernel or draw changes; the row stays bitwise)", ("mem", "releases"),
                 doc="mem.py (the memory policy of the kit rows); the seams: opt/forward/rf3_xattempt_addon/patched/rf3/graph_flags.py hoist_begin / hoist_end"),
    "big_atom_pair_local": Lever("big_atom_pair_local", "ROSETTAFOLD3_BIG_ATOM_PAIR_LOCAL", PACKAGE, ("big.py", "levers.py"),
                                   "the atom-pair conditioning P_LL built in the atom transformer's window form ([nq, 32, 128, c] in place of [L, L, c]: the "
                                   "dense tensor is every arm's first OOM at the 2500 rung); big mode only", CLASS_MEMORY,
                                   "measured (pointwise per pair)", ("big", "atom_pair_local"), doc="big.py; levers.py"),
    "big_opm_chunk": Lever("big_opm_chunk", "ROSETTAFOLD3_BIG_OPM_CHUNK", PACKAGE, ("big.py", "levers.py"),
                             "the MSA outer product mean over row blocks (opt_core.mem.chunk.chunk_rows); big mode only", CLASS_MEMORY,
                             "measured (per-pair arithmetic)", ("big", "opm_chunk"), doc="big.py; levers.py"),
    "big_cond_chunk": Lever("big_cond_chunk", "ROSETTAFOLD3_BIG_COND_CHUNK", PACKAGE, ("big.py", "levers.py"),
                              "the diffusion pair conditioning over row blocks (opt_core.mem.chunk.chunk_rows); big mode only", CLASS_MEMORY,
                              "measured (per-pair arithmetic)", ("big", "cond_chunk"), doc="big.py; levers.py"),
    "big_confidence_offload": Lever("big_confidence_offload", "ROSETTAFOLD3_BIG_CONFIDENCE_OFFLOAD", PACKAGE, ("big.py", "levers.py"),
                                      "pae/pde logits per sample to the host (opt_core.mem.offload PinPool staging); the consumers per sample on the device; big mode only",
                                      CLASS_MEMORY, "measured (copies + the stock's per-sample statements; expected bitwise)", ("big", "confidence_offload"),
                                      doc="big.py; levers.py"),
    "big_transition_chunk": Lever("big_transition_chunk", "ROSETTAFOLD3_BIG_TRANSITION_CHUNK", PACKAGE, ("big.py", "levers.py"),
                                    "the stock Transition (pair / MSA / single) in row blocks of the leading dim (opt_core.mem.chunk.chunk_rows): the [.., N, 4c] hidden never whole; "
                                    "big (the fused ttr disengaged by lever property; refused by name under an arm that keeps it)", CLASS_MEMORY,
                                    "measured (per element of the leading dims the statement is stock's; the linears' GEMM at M = rows x J)", ("big", "transition_chunk"), doc="big.py; levers.py"),
    "big_feature_park": Lever("big_feature_park", "ROSETTAFOLD3_BIG_FEATURE_PARK", PACKAGE, ("big.py", "levers.py"),
                                "the trunk-only input feature tensors (msa_stack and its per-recycle slice, the template conditioning, the stock-cast fp32 originals the engine "
                                "keeps referencing) parked on pinned host between their uses and after the trunk (opt_core.mem.offload.HostPark under the ONE pinned budget "
                                "shared with confidence_offload, B24); big", CLASS_MEMORY,
                                "measured (copies only: a parked tensor returns byte-identical; the cast is the stock's own op one statement earlier)", ("big", "feature_park"), doc="big.py; levers.py"),
    "big_triatt_chunk": Lever("big_triatt_chunk", "ROSETTAFOLD3_BIG_TRIATT_CHUNK", PACKAGE, ("big.py", "levers.py"),
                                "the trunk's triangle attention in query-row blocks (opt_core.mem.chunk.triangle_attention_chunked on the stock cuEquivariance site: the "
                                "full bias once, q/k/v/gate per block); big (the fused gflash disengaged by lever property; refused by name under an arm that keeps it)", CLASS_MEMORY,
                                "measured (per query row the statement is stock's; the projections' GEMM at M = rows x N)", ("big", "triatt_chunk"), doc="big.py; levers.py"),
    "rowpair": Lever("rowpair", None, PACKAGE, ("rowpair.py", "fold.py", "stack.py"),
                     "the row-sharded pair stack over --n_gpu P GPUs (opt_core.mem.rowpair: row Layout, ring/gathered trimul contracts, bias-gathered triangle attention, "
                     "row-local transition / OPM / template rows, rank-0 writer); big mode only, P > 1 (P = 1 constructs nothing); pred starts the rank processes",
                     CLASS_MEMORY, "fast-class (sharded reductions reorder sums: within the fast band, never bitwise; refused by name under exact / fast)", ("rowpair",),
                     doc="rowpair.py; README.md (--n_gpu P)"),
}
STRATEGY_OF: Dict[str, str] = {           # house lever name -> the release tree's canonical cross-engine strategy id (report.lever_lines prints it as strategy=;
                                          # a prerequisite lever carries the id of the strategy it serves: graph_safe_ops the sampler graph's, fpf_sapb the trunk graph's)
    "graph": "F3.cuda_graph_sampler", "graph_safe_ops": "F3.cuda_graph_sampler", "hoist": "LOCAL.step_invariant_hoist",
    "fpf_trimul": "F2.fpf_trimul_fast", "fpf_gflash": "F1.triattn_rows", "fpf_ttr": "LOCAL.fused_transition", "fpf_apb": "LOCAL.layernorm_kernel",
    "fpf_sapb": "F3.cuda_graph_trunk", "fpf_tg": "F3.cuda_graph_trunk", "fpf_res": "F5.kernel_glue_gates", "fpf_xatt": "F1.triattn_exact", "fpf_msa": "F5.fpf_msa_kernels", "fpf_smsa": "F5.fpf_msa_kernels", "fpf_xmul": "F2.trimul", "fpf_xln": "LOCAL.layernorm_kernel",
    "fpf_dattn": "F5.flash_attn_dense", "dtk": "F5.flash_attn_dense", "xtr": "LOCAL.fused_transition", "mkdit": "LOCAL.dit_fused_kernels", "mem": "F7.expandable_segments",
    "confhoist": "LOCAL.step_invariant_hoist",
    "warm": "LOCAL.sampler_warm_once", "hostlean": "LOCAL.host_sync_removal", "prefetch": "F6.item_ordering_prefetch", "awrite": "F6.output_overlap",
    "confln": "LOCAL.layernorm_kernel",
    "big_atom_pair_local": "F7.chunked_eval", "big_opm_chunk": "F7.chunked_eval", "big_cond_chunk": "F7.chunked_eval",
    "big_transition_chunk": "F7.chunked_eval", "big_triatt_chunk": "F7.chunked_eval",
    "big_confidence_offload": "F7.pair_offload", "big_feature_park": "F7.pair_offload",
    "rowpair": _core.load("mem.ngpu").TP_LEVER,            # the canonical id of the row-sharded pair stack (F7.tensor_parallel), spelled by the core
}
IMPL_OF: Dict[str, Tuple[str, str]] = {          # house lever name -> (impl, origin) for the LEVER line: the implementing kit file or core kernel module, kit | core
    "graph": ("patched/rf3/diffusion_samplers/inference_sampler.py", "kit"), "graph_safe_ops": ("patched/rf3/graph_flags.py", "kit"),
    "hoist": ("patched/rf3/model/layers/af3_diffusion_transformer.py", "kit"),
    "fpf_trimul": ("opt_core.kernels.fpf_trimul_v4", "core"), "fpf_gflash": ("opt_core.kernels.triattn", "core"), "fpf_ttr": ("opt_core.kernels.transition", "core"),
    "fpf_apb": ("opt_core.kernels.lnl_fused", "core"), "fpf_sapb": ("rf3fpf/fpf_rf3_adapter.py", "kit"),
    "fpf_tg": ("rf3fpf/fpf_rf3_adapter.py", "kit"), "fpf_res": ("opt_core.kernels.transition+fpf_trimul_v4", "core"), "fpf_dattn": ("rf3fpf/fpf_rf3_adapter.py", "kit"),
    "fpf_xmul": ("opt_core.kernels.trimul", "core"), "fpf_xln": ("opt_core.kernels.ln", "core"), "fpf_xatt": ("opt_core.kernels.triattn", "core"), "fpf_msa": ("rf3fpf/fpf_rf3_msa_rows.py", "kit"), "fpf_smsa": ("rf3fpf/fpf_rf3_msa_rows.py", "kit"),
    "dtk": ("opt_core.kernels.dtk_kernels", "core"), "xtr": ("opt_core.kernels.transition", "core"), "mkdit": ("forward/rf3_mk_dit_addon/mkdit/mk2.py", "kit"), "mem": ("opt_core.mem.torch_alloc+mem.py", "core"),
    "confhoist": ("confhoist.py", "kit"),
    "warm": ("rf3_xattempt_addon/patched/rf3/graph_flags.py", "kit"),
    "hostlean": ("hostlean.py", "kit"), "prefetch": ("prefetch.py", "kit"), "awrite": ("awrite.py", "kit"),
    "confln": ("confhoist.py", "kit"),
    "big_atom_pair_local": ("levers.py", "kit"), "big_opm_chunk": ("opt_core.mem.chunk+levers.py", "core"), "big_cond_chunk": ("opt_core.mem.chunk+levers.py", "core"),
    "big_transition_chunk": ("opt_core.mem.chunk+levers.py", "core"), "big_triatt_chunk": ("opt_core.mem.chunk+levers.py", "core"),
    "big_confidence_offload": ("opt_core.mem.offload+levers.py", "core"), "big_feature_park": ("opt_core.mem.offload+levers.py", "core"),
    "rowpair": ("opt_core.mem.rowpair+rowpair.py", "core"),
}
LEVERS = {n: replace(lv, strategy=STRATEGY_OF[n]) for n, lv in LEVERS.items()}
FPF_LEVERS: Dict[str, Lever] = {n: lv for n, lv in LEVERS.items() if lv.kit == FPF}


def switches() -> Tuple[str, ...]:
    return tuple(lv.switch for lv in LEVERS.values() if lv.switch)
