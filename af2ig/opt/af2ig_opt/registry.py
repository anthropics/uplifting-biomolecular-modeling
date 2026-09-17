"""The lever registry: one entry per lever of the kit, keyed by the kit's own id (CHANGES.md, the driver's argparse help).

The kit describes its levers in CHANGES.md and implements every one of them as an opt-in flag of its PyRosetta-free driver
`af2_initial_guess/predict_pdb.py` (patch 00; the later patches of the series add their flags to it, 05 also the flash-attention core of the
vendored AlphaFold's Attention). This registry only *describes* each lever — the flag, which kit file
applies it, its class (forward / datapath / serving / orchestration), its numerics tier against the stock line, where the kit documents
it. No lever value or mode composition lives here: the composition is read from the kit's own statement (modes.py) and the flags are
parsed by the kit's own driver.
"""
from dataclasses import dataclass
from typing import Dict

L1, L6, L7, L8, L9, L10, L11, L12, L15, L16, L18, L19, U1 = "L1", "L6", "L7", "L8", "L9", "L10", "L11", "L12", "L15", "L16", "L18", "L19", "U1"
L19_ENV, L19_WORD = "AF2IG_OPT_TRIATTN_CORE_DTYPE", "bf16"                     # L19 : the bridge attention core's OPERAND dtype word placed in the kit-mode child (fast, big); af2ig_opt.triattn reads it
OPM = L12                                                                      # the re-associated OuterProductMean (af2ig_opt.opm); one alias so the id is renamed in one place
CCACHE = "ccache"                                                               # the deployment lever of every kit mode: JAX's persistent compilation cache placed in the driver child's environment (af2ig_opt.ccache); never on the stock line
LEVERS_OFF_ENV = "MODEL_OPT_LEVERS_OFF"
COMPILE_WORD = "compile"                                                        # the tree's `compile` ablation word (MODEL_OPT_LEVERS_OFF=compile; `--no-compile` is its alias): af2ig has NO kit compile lever — JAX's jit per distinct length is stock behaviour the kit inherits untouched (L7 / ccache / L13 only shorten that wall) — so the word drops nothing and the activation line says so: compile=stock_jit
COMPILE_STATE = "stock_jit"                                         # the tree's uniform ablation switch: comma-separated lever ids dropped from the mode's composition (modes.resolve levers_off); unknown ids are named and ignored
TRIMUL, MEMF = "trimul_chunk", "mem_fraction"                                  # the memory line (big): installed by the driver hook of patch 06 (af2ig_opt.pairstack) / the child environment (mem_fraction)
DRIVER = "patches/patched_files/af2_initial_guess/predict_pdb.py"     # the kit's copy of the driver (the same bytes at $AF2IG_DIR/predict_pdb.py)
PUBLIC_INPUTS = "tests/inputs/pdbs"
PRESET_FLAG = "-fast"                                                 # the driver's preset: -host_outputs -device_params -sort_by_length (predict_pdb.py:106,109-110)


@dataclass(frozen=True)
class Lever:
    name: str                      # the kit's own id
    flag: str                      # the driver's flag (or the environment variable)
    what: str                      # one line: what changes
    class_4: str                   # forward | datapath | serving | orchestration
    tier_vs_stock: str             # numerics vs the stock line
    kit_file: str                  # kit file that implements it (relative to the kit root) with lines
    doc: str = ""                  # where the kit documents it
    in_preset: bool = False        # part of the driver's -fast preset
    strategy: str = ""             # the tree's canonical strategy id (opt_core/STRATEGIES.json) or LOCAL.af2ig.<name> (this engine only) — strategy= on the LEVER evidence line
    switch: str = ""               # the one token that engages it: the driver's argv flag (`-subbatch`) or the environment variable's name — what the evidence reads (stack.applied) and the LEVER line prints (flag=)


LEVERS: Dict[str, Lever] = {
    L6: Lever(L6, "-host_outputs", "the model's result tree copied to host NumPy in one jax.device_get before post-processing, instead of "
                  "hundreds of lazy device->host transfers from numpy/scipy (5-20 s per design)", "datapath", "bitwise (the same float32 values)",
              DRIVER + ":103,533-535", "CHANGES.md L6", True, "F6.host_sync_elimination", "-host_outputs"),
    L1: Lever(L1, "-device_params", "the parameter tree jax.device_put once at start-up instead of an implicit host->device copy of ~370 MB per call",
              "datapath", "bitwise", DRIVER + ":99,307-313", "CHANGES.md L1", True, "F6.weights_residency_init", "-device_params"),
    U1: Lever(U1, "-sort_by_length", "inputs processed in (residue count, name) order instead of file-name order; nothing else changes",
              "orchestration", "bitwise (scheduling only)", DRIVER + ":96,592-599", "CHANGES.md U1", True, "F6.item_ordering_prefetch", "-sort_by_length"),
    L7: Lever(L7, "-precompile N", "one input per distinct (padded) length featurised and the jitted model called for all of them from N host threads "
                  "before the loop; the loop then runs the executables jax already cached (the same executables the loop would compile serially)",
              "orchestration", "bitwise (same executables)", DRIVER + ":104,731-778", "CHANGES.md L7", strategy="LOCAL.af2ig.precompile", switch="-precompile"),
    L8: Lever(L8, "-flash_attn", "the attention core (logits + mask/pair bias, softmax, weighted sum) of every eligible alphafold Attention call computed by the kit adapter "
                  "af2ig_opt.flash_attn over the tree's shared Pallas flash-attention kernel (opt_core.kernels.pallas_attn) instead of the three XLA ops; ineligible calls "
                  "(non-GPU backend, unequal or < 16 head widths, a mask bias not [b,1,1,k], keys below the size gate) run the stock ops and are counted",
              "forward", "Tier 2: online-softmax re-association, TF32 float32 products (f32_precision=tf32; not bitwise); the pair-biased calls only; composed on the fast line", DRIVER + ":105,112-119,300-302,552-554,772-773,827-830 + patches/05_flash_attention.diff + opt/af2ig_opt/flash_attn.py", "CHANGES.md L8", strategy="F1.flash_triatt", switch="-flash_attn"),
    L9: Lever(L9, "-subbatch N", "the inference sub-batch of every Attention module (triangle start/end, MSA row/column, template) and every Transition "
                  "(alphafold mapping.inference_subbatch; stock 4 rows per chunk, sized for 16 GB cards) set to N rows: the same per-row math in N-row chunks, "
                  "N/rows loop iterations of large GEMMs instead of many small ones; decided and recorded through the tree's shared sub-batch policy",
               "forward", "Tier-2: other GEMM shapes, other accumulation order (not bitwise); the same per-row math", DRIVER + " -subbatch + opt/af2ig_opt/subbatch.py over opt_core.jax_design.subbatch_policy", strategy="F7.chunked_eval", switch="-subbatch"),
    L18: Lever(L18, "-tmpl_pointwise_sub N", "the row sub-batch of the template point-wise attention (alphafold TemplateEmbedding, template.subbatch_size; stock 128 rows of the "
                    "N*N-row attention over the templates) set to N rows: the same per-row softmax/GEMM statements traced at another batch width — fewer, larger kernel launches; "
                    "XLA may pick other GEMM algorithms at the new width (tier 2: worst Ca-RMSD 0.035 A on the identity canaries); the rows are a per-tier parameter "
                    "(fast 8192; big by its measured memory cost, modes.TMPL_ROWS) named on the LEVER line (rows=<n> tier=<mode>)", "forward",
                    "tier 2 (tolerance: identity form Ca-RMSD; deterministic run-to-run under --det 1)", "opt/forward/af2ig_kit/patches/15_tmpl_pointwise_sub.diff (predict_pdb.py -tmpl_pointwise_sub)",
                    "CHANGES.md L18", strategy="F7.chunked_eval", switch="-tmpl_pointwise_sub"),
    L19: Lever(L19, "env AF2IG_OPT_TRIATTN_CORE_DTYPE=bf16", "bfloat16 OPERANDS (q, k, v) for the triangle-attention bridge core of L10 (af2ig_opt.triattn: the pair bias, softmax and "
                    "accumulation stay float32; the output is cast back to the activations' float32): the core's bf16 row order then serves each cell (its per-cell choice, "
                    "never pinned by word here); an environment lever of the kit-mode child (no argv token), evidenced by the fused_triattn census (bridge=…+core_bfloat16, "
                    "providers=<row>=<calls>); a caller's own AF2IG_OPT_TRIATTN_CORE_DTYPE=fp32 keeps the float32 core (L19 off by word, named)", "forward",
                    "tier 2 (bf16 operand rounding inside the attention core; identity form Ca-RMSD; deterministic run-to-run under --det 1)", "opt/af2ig_opt/triattn.py core_dtype (the word) + stack.activate / cli (placement in the child)",
                    "CHANGES.md L19", strategy="LOCAL.af2ig.triattn_core_bf16", switch="AF2IG_OPT_TRIATTN_CORE_DTYPE"),
    L10: Lever(L10, "-fused_triattn", "every TriangleAttention module (starting / ending node; the 48 Evoformer blocks and the template pair stack) computed by the tree's fused "
                   "triangle-attention block (LayerNorm + pair bias + q|k|v|gate projections + flash core + gating + output projection in one Pallas program per call, float32 with "
                   "TF32 products; refused calls run the stock body, counted by reason)",
                "forward", "Tier-2: online-softmax re-association and fused projections, TF32 float32 products (not bitwise); deterministic", "opt/af2ig_opt/triattn.py over opt_core.kernels.fpf_pallas_serve.tri_attn_block (" + DRIVER + " -fused_triattn installs it)", strategy="F1.flash_triatt", switch="-fused_triattn"),
    L11: Lever(L11, "-fused_trimul", "every TriangleMultiplication module (outgoing / incoming; the 48 Evoformer blocks and the template pair stack) computed by the tree's fused "
                   "triangle-multiplication block (input LayerNorm + left|right projections, gates and mask in a Pallas prologue, one batched TF32 GEMM for the contraction, centre LayerNorm + "
                   "output projection + gate in a Pallas epilogue; float32; refused calls run the stock body, counted by reason)",
                "forward", "Tier-2: fused LayerNorm/projection arithmetic, TF32 float32 products (not bitwise); deterministic", "opt/af2ig_opt/fused_trimul.py over opt_core.kernels.fpf_pallas_serve.trimul_block (" + DRIVER + " -fused_trimul installs it)", strategy="F2.fpf_trimul_fast", switch="-fused_trimul"),
    L12: Lever(L12, "-opm_reassoc", "every OuterProductMean module (the 48 Evoformer blocks and the 4 extra-MSA blocks) computed re-associated: the output weight folded into the "
                   "right projection ([S, 32, N, c_z] float32), then ONE GEMM over (MSA row, channel) for the [N, N, c_z] update — no [rows, N, 32x32] outer-product intermediate "
                   "and ~6.6x fewer FLOPs at this model's 5-row MSA; mask, normalisation and epsilon as stock; pure JAX, rebound onto the stock class in the driver process (an input the "
                   "body does not expect runs the stock body, counted by reason)",
                "forward", "Tier-2: re-associated float32 sums, XLA default-precision (TF32-class) products as stock's einsums (not bitwise); deterministic", "opt/af2ig_opt/opm.py (" + DRIVER + " -opm_reassoc installs it; patches/12_opm_reassoc.diff)", "CHANGES.md L12", strategy="LOCAL.af2ig.opm_reassoc", switch="-opm_reassoc"),
    TRIMUL: Lever(TRIMUL, "-trimul_chunk <rows>:<min residues>", "TriangleMultiplication (outgoing / incoming: Evoformer, extra-MSA and template pair stacks) evaluated in row chunks of the pair "
                  "representation once the compiled length reaches <min residues>: the [N, N, c] f32 projection/einsum intermediates become [rows, N, c]; the body is the tree's "
                  "row-chunk producer (opt_core.mem.rowpair_jax.rowchunk), rebound onto the stock class in the driver process (af2ig_opt.pairstack)", "forward",
                  "TIER-2 (within the stock seed band on every key; the same sums per element, XLA free to re-associate)",
                  DRIVER + " (patch 06: -trimul_chunk) + opt/af2ig_opt/pairstack.py", strategy="F7.chunked_eval", switch="-trimul_chunk"),
    MEMF: Lever(MEMF, "env XLA_PYTHON_CLIENT_MEM_FRACTION=0.95", "the XLA client's memory pool at 95 % of the card instead of the default 75 % (allocator reservation, set in the driver's "
                "environment by the memory mode; the driver records it in proc_start.env)", "serving", "bitwise (no numerics: a pool size)",
                DRIVER + ":720-724 (the record) + opt/af2ig_opt/big.py (the value)", strategy="F7.jax_memory_flags", switch="XLA_PYTHON_CLIENT_MEM_FRACTION"),
    "L13": Lever("L13", "-program_cache", "every model program the process compiles (one per distinct residue count and lever set) is serialized under a directory keyed by the stack "
                 "(<jit root>/<stack key>/programs; af2ig_opt.programs: the key names the code, the levers, the stack, the device and the XLA flags) and LOADED by later processes instead of "
                 "traced, lowered and compiled again — the warm first design of a length runs at the steady model time plus the load; a loaded program's trace-time kernel census "
                 "travels with it and is replayed, named; a file that does not load is named and the program is traced", "serving",
                 "bitwise (no numerics: the executable is the one this stack compiled from this code)", "predict_pdb.py AF2_runner.model_program + opt/af2ig_opt/programs.py (patch 13)",
                 "CHANGES.md L13", strategy="F3.jit_cache_keyed", switch="-program_cache"),
    L15: Lever(L15, "-prefetch N", "the inputs of the next N designs read and featurised (PDB parse, template / single-sequence MSA features, chain-break detection, the TensorFlow feature "
                 "pipeline: one host core) on ONE background worker thread while the current design runs on the GPU, taken by the loop at their turn; the same featurisation, one at a time, in the "
                 "loop's order — only when it runs moves; a featurisation error surfaces at the design's turn as before; at most N featurised inputs queued on the host", "orchestration",
              "bitwise (scheduling only: which host thread featurises, and when)", DRIVER + " -prefetch + opt/af2ig_opt/prefetch.py (patch 14)", "CHANGES.md L15", strategy="F6.item_ordering_prefetch", switch="-prefetch"),
    L16: Lever(L16, "-overlap_output N", "each design's output step (confidence metrics, RMSDs, the PDB file, the score line, then the checkpoint line) run on ONE background writer thread, "
                 "in the loop's order, while the next design's forward runs; at most N outputs wait behind the writer; the step's code and inputs unchanged, a failing step reported "
                 "and checkpointed as the loop does; host memory: up to N+1 designs' outputs alive", "orchestration",
              "bitwise (scheduling only: which host thread writes, and when)", DRIVER + " -overlap_output + opt/af2ig_opt/prefetch.py OutputWriter (patch 14)", "CHANGES.md L16", strategy="F6.item_ordering_prefetch", switch="-overlap_output"),
    CCACHE: Lever(CCACHE, "env JAX_COMPILATION_CACHE_DIR=<root>/<stack key>/jax", "JAX's persistent compilation cache placed in the driver's environment under a root keyed by the stack "
                  "(AF2IG_OPT_JIT_ROOT, else MODEL_OPT_JIT_ROOT, else <the package cache root>/jit; a JAX_COMPILATION_CACHE_DIR the caller set is kept) with the thresholds that store every "
                  "compilation: a later process on the same stack LOADS the executables an earlier one compiled — one per distinct residue count, tens of seconds each — instead of compiling them; "
                  "every kit mode carries it, the stock line never (a must-be-absent name of the stock rule); a root that cannot be written steps aside by name", "serving",
                  "bitwise (no numerics: a loaded executable is the one that was compiled)", "opt/af2ig_opt/ccache.py over opt_core.capture.xla_cache (cli.py places it in the driver child's environment)",
                  "CHANGES.md ccache", strategy="F3.jit_cache_keyed", switch="JAX_COMPILATION_CACHE_DIR"),
}
MEMORY_LEVERS = (TRIMUL, MEMF)                                       # the memory line's lever ids (big): MEMORY_DEFAULT composed, MEMORY_OPT_IN by word
MEMORY_DEFAULT = (MEMF,)                                             # what big composes — the XLA pool fraction (the reach lever)
MEMORY_OPT_IN = (TRIMUL,)                                            # composed only when AF2IG_OPT_TRIMUL_CHUNK opts it in (measured memory + speed cost at every size)
PROGRAM_LEVERS = ("L7", "L13")                                         # the program line of every kit mode: prepared ahead of the loop (L7, AOT on host threads) and kept across processes (L13)
PIPELINE_LEVERS = (L15, L16)                                          # the host pipeline of every kit mode — the next designs' inputs featurised ahead of their turn on a worker thread (L15) and each design's output step on a writer thread behind the loop (L16); exact-class
DEPLOYMENT_LEVERS = (CCACHE,)                                        # placed in every kit mode (exact, fast, big): no numerics, no argv token — an environment lever of the driver child


def _check_switches() -> None:
    """Every lever names its switch explicitly (one source: no token is parsed out of the descriptive `flag` text)."""
    missing = [k for k, lv in LEVERS.items() if not lv.switch or any(c.isspace() for c in lv.switch)]
    if missing:
        raise RuntimeError(f"registry: lever(s) without a one-token switch: {missing}")


_check_switches()
