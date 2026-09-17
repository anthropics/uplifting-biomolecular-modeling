"""The lever registry: one entry per lever a mode of this kit plans (modes.py) — the resident driver's (opt/forward/fast_inference/driver/
g3fast.py: L1 persistent process, L2 sync-free step + value-identical forward patches, L4 CUDA-graph replay), the capture-cell levers over it
(opt/forward/g3cap/g3cap.py: L8 hoist, L9 in-request graph reuse), and the batched capture line's own (opt/genie3_opt/g3batch.py: L11
upstream's batch semantics at B, L17 the lean pair stack, L18 wide capture, L19 the allocator setting, L12 the pair transition chunked per design, L13 TF32 matmuls, L16 the inductor-compiled core, L7 the fused TriangleMultiplication kernel through
opt/genie3_opt/trimul.py). The two carried modules are libraries the batched capture line imports unchanged; that line is the one all three
modes run, and every lever here is in at least one mode.

A lever is described here — which kit file implements it, the switch the line uses for it (a driver flag written exactly as the line
writes it), its class (forward / orchestration), its numerics tier against stock, and the evidence the line leaves when it has applied the
lever in a process (a line the driver prints, or a field of the timings JSON the driver writes with `--timings`; design.py reads both back) —
and composed elsewhere: modes are lists of these ids in modes.py, the only place a mode names its switches. Nothing here applies anything;
the kit's own driver does.

Switch grammar: ("flag", <driver flag>, <value>) is passed on the driver's command line (a value of None = a bare flag);
("default", None) names a lever the line applies unconditionally (no flag); ("driver", <path>) names the resident driver itself.
Evidence grammar: ("log", <regex>) = a line of the driver's stdout; ("timings", <key>) = a key of the timings JSON present and truthy;
("judged", <name>) = a predicate of design.JUDGES over (log lines, timings record).
A lever that cannot serve a given request declines it by name: ``Lever.declined = ("log+timings", <regex over this pass's driver log>,
<key path into this pass's timings record>)`` — both present = declined (design.read_evidence), either absent = missing (the pass is partial).
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

KIT = "fast_inference"                                    # opt/forward/<kit>
G3CAP = "g3cap"                                           # opt/forward/g3cap: the capture-cell levers (L8, L9) over the resident driver
KIT_DIRS = {KIT: "opt/forward/fast_inference", G3CAP: "opt/forward/g3cap"}
DRIVER = "driver/g3fast.py"                                # the kit's resident driver (KIT)
G3CAP_DRIVER = "g3cap.py"                                  # the capture-cell module (G3CAP): install_hoist / feat_signature / GraphCache / rebind_graph, imported by the batched capture line after the resident driver
G3LEAN = "g3lean.py"                                       # the capture line's memory levers beside G3CAP_DRIVER (L17 lean pair stack, L18 wide capture), imported by the batched capture line
PACKAGE = "genie3_opt"                                     # opt/genie3_opt: a driver-chain entry that names the package's own file rather than a kit's (stack.chain_file: the launchers); never a kit directory (the stock proof lists kit directories only)
G3BATCH = "g3batch.py"                                     # the batched capture driver (PACKAGE): upstream's batch semantics at B (L11) under L2/L4/L8/L9, importing DRIVER and G3CAP_DRIVER unchanged; L12/L13 by their own flags; the modes' line (modes.py)


@dataclass(frozen=True)
class Lever:
    id: str
    name: str
    cls: str                                   # forward | serving | orchestration
    tier: Optional[int]                        # 1 = byte-identical claim, 2 = tolerance (the kit's own distributional standard), 3 = numerics-changing, None = no numerics
    switch: Tuple                              # ("flag", flag, value) | ("default", None) | ("driver", path)
    file: str                                  # kit file (line) implementing it
    evidence: Optional[Tuple] = None           # ("log", regex) | ("timings", key) | ("judged", name: design.JUDGES — a predicate over the driver's record, e.g. the numerics readback); None = the driver itself
    doc: str = ""
    declined: Optional[Tuple] = None           # ("log+timings", regex, (key, …)): a lever that cannot serve THIS request says so twice — its line in this pass's log AND the reason at this key path of this pass's timings record (L7 under the kernel's token floor: KERNELS trimul=declined:… + timings.trimul.declined); both present = `declined` (by that reason), not missing; either absent = missing


L7_STANDING = ("class 3 (tolerance tier), measured on this engine; mode fast's TriangleMultiplication provider on every card with a served row in opt/genie3_opt/fpf_cells.json "
               "(cc 9.0: H100 / H200; cc 8.0: A100) — never on the exact line. Calls under the kernel's own floor "
               "(101 tokens: fpf_trimul_v4.generic.N_MIN) run the module's forward, counted `below_min_tokens` (an EXPECTED reason: the census word reads `partial:…`); a request "
               "whose every call is under the floor declines the lever by name (`declined:below_min_tokens`, the LEVER line `state=skipped`) — never a refusal; any other "
               "fallback reason or a kernel error refuses the lever's gate (`fallback:…`: no evidence, the pass is partial).")


LEVERS: Dict[str, Lever] = {
    "L1": Lever("L1", "persistent process: model, sampler and featuriser built once, many designs per process", "orchestration", None,
                ("driver", DRIVER), "driver/g3fast.py (build_everything: the stock path's set-up, once per process)", ("timings", "model_setup_s"),
                "one python process per `genie3 generate` on the stock line (a fresh start-up each time); the driver's set-up equals the stock path's"),
    "L2": Lever("L2", "sync-free DDIM step + the value-identical forward patches (device-side index tensors, vectorised Frenet frames, static cond-group count)", "forward", 1,
                ("default", None), "driver/g3fast.py (StepTables / ddim_math), driver/g3fast_patches.py (apply, called by the line with verbose=False — no log line; the timings key `patches` is its record)", ("timings", "patches"),
                "applied by the line unconditionally (no flag): identical values, identical kernels for the arithmetic"),
    "L4": Lever("L4", "CUDA-graph replay of the denoiser core (pair_transform_net + sequence_net + structure_net), one capture per batch shape, replayed for the 100 steps", "forward", 1,
                ("flag", "--cuda-graphs", None), "driver/g3fast.py (GraphedDenoiser)", ("timings", "graph_capture_s"),
                "the embedders stay eager (rot_to_quat calls cuSOLVER eigh); one capture per distinct batch shape per process, kept and replayed under L9"),
    "L8": Lever("L8", "loop-invariant hoist in the pair featuriser: the relative-position encoding, the four pair masks and the conditional-template term (an N²-row rot_to_quat eigen-solve, three Linear layers, five concatenations) computed once per design at sampling entry instead of once per DDIM step — none depends on the step", "forward", 1,
                ("flag", "--hoist", None), "g3cap.py (install_hoist: V1PairFeatureNet.forward rebound; the per-batch cache is filled on the first step and cleared per batch)", ("judged", "hoist"),
                "class 1: identical kernels on identical values (the hoisted terms are step-invariant; torch.cat of the cached and per-step parts is an exact copy) — byte-identical PDBs to stock; the batched driver writes the timings key `hoist` from its own lever block (g3batch.py)"),
    "L9": Lever("L9", "in-request CUDA-graph reuse: captured denoiser graphs kept alive keyed by feature-shape signature (up to R of them); a later design of the same shapes replays the kept graph with its features copied into the graph's static tensors instead of re-capturing (the resident driver captures once per design at batch 1)", "forward", 1,
                ("flag", "--reuse-graphs", "16"), "g3cap.py (feat_signature, GraphCache, rebind_graph; the batched capture line acquires / captures / rebinds per batch shape)", ("timings", "graph_cache"),
                "class 1 (the same graph replays the same kernels on the new design's values); pays where a request repeats shapes — every design of a fixed-length unconditional request, none of a binder request whose lengths all differ; the cache lives inside one process and is freed at exit (nothing is cached across requests); memory: up to R private graph pools stay allocated"),
    "L7": Lever("L7", "fused TriangleMultiplication kernel: the ten TriangleMultiplicativeUpdate modules served by the shared core's fpf_trimul_v4 (generic entry, native leading batch) through opt_core.trimul's ladder — provider fpf_v4, strategy word F2.fpf_trimul_fast (bf16 tensor-core GEMM operands, fp32 accumulate / LayerNorm / gating); unsupported calls run the module's own forward counted by reason, an unexpected reason refuses the lever's gate", "forward", 3,
                ("flag", "--trimul", "fpf"), "opt/genie3_opt/trimul.py (patch site: stock src/genie3/generation/model/module/triangular_multiplicative_update.py:116-146; the census word: opt/genie3_opt/kernels.py); opt/genie3_opt/g3batch.py (the switch, applied before the graph capture)",
                ("log", r"\[genie3-opt\] KERNELS .*\btrimul=(?:engaged|partial):fpf_trimul_v4@\S*\[served=[1-9]"),
                L7_STANDING, declined=("log+timings", r"\[genie3-opt\] KERNELS .*\btrimul=declined:below_min_tokens\b", ("trimul", "declined"))),
    "L11": Lever("L11", "upstream's own batch semantics at batch size B: the request's designs B per denoiser call in dataset order, the initial noise and each step's noise one draw over the batch tensor, the timestep vector per step, the PDB files per batch — what `genie3 generate` computes with `generation.dataset.batch_size: B`", "forward", 1,
                 ("flag", "--batch-size", "1"), "opt/genie3_opt/g3batch.py (featurize_batches: data_module.py GenieTestSampler's index blocks through the dataset's own __getitem__; step_vectors: sampler.py:109; the sampling loop: sampler.py:98-110 / ddim.py _step over g3fast.StepTables / ddim_math / GraphedDenoiser; write_batch: postprocess per batch as runner.py on_test_batch_end)", ("judged", "batches"),
                 "class 1 at every B: byte-identical PDB files vs eager deterministic stock at the SAME batch size (GEMM shapes are stock's own at that B, the noise draws stock's own); B is `design --batch_size B`, else the request's own generation.dataset.batch_size, else upstream's default 1 — the kit changes no workload key; at B = 1 it is the per-design stream of the capture-cell line"),
    "L12": Lever("L12", "the pair transition chunked per design: upstream runs PairTransition's two-Linear MLP through chunk_layer in 4-row chunks at inference (transition.py:37,102-110 — B·N sequential chunk calls per layer per step); the switch sets chunk_size = the pair tensor's row count N on the five modules per batch shape, so each call runs B chunks of one design's N×N×c slab — the large-GEMM path of the whole-tensor form with one design's 4·c hidden intermediate live instead of the batch's (708 tokens × batch 8: 4 GB instead of 32 GB)", "forward", 2,
                 ("flag", "--pt-chunk", "design"), "opt/genie3_opt/g3batch.py (chunk_pair_transition_per_design, set per batch shape before the capture)", ("timings", "pt_chunk"),
                 "class 2: identical arithmetic per element, but cuBLAS selects another GEMM kernel for the larger row count — not byte-identical to stock at every shape; inside fast only, never on the exact line"),
    "L13": Lever("L13", "TF32 tensor-core matmuls for the whole process (torch.backends.cuda.matmul.allow_tf32, cudnn.allow_tf32, float32 matmul precision `high`); strategy word F4.tf32_matmul (the shared core's STRATEGIES.json canonical id)", "forward", 3,
                 ("flag", "--tf32", None), "opt/genie3_opt/g3batch.py (main: opt_core.precision.policy apply(FP32_TF32); the live readback recorded as timings `numerics_readback` / `global_numerics_state`)", ("judged", "tf32"),
                 "class 3, the tolerance tier's precision lever: admissible inside `fast` only, admitted distributionally (tolerance tier) and always reported beside the same line without it; never on the exact line; stock stays numerics-as-shipped (fp32)"),
    "L16": Lever("L16", "compiled denoiser core: torch.compile (inductor, default mode, static shapes) of the pair-transform / sequence / structure nets — the elementwise, LayerNorm, mask, bias and residual chains between the GEMMs fused into generated Triton kernels — with the COMPILED callable recorded by the kit's own CUDA graph (L4) once per batch shape; the TriangleMultiplicativeUpdate forward (L7's adapter) stays out of dynamo and inside the graph; strategy word F5.inductor_core", "forward", 3,
                 ("flag", "--compile", None), "driver/g3fast.py (GraphedDenoiser._compiled_core: compile during the eager warm-up, capture under the fail_on_recompile stance)", ("timings", "compile"),
                 "class 2-3: inductor re-associates fp32 reductions and fuses producer chains — a few last bits per call differ from the eager core (measured: x0-hat per call within 0.07 Å of the eager fast core); inside fast only, never on the exact line; one compile per captured batch shape per process (tens of seconds cold, seconds with a warm inductor cache under $MODEL_OPT_JIT_ROOT/inductor)"),
    "L17": Lever("L17", "lean pair stack: LatentTransformer / LatentTransformerBlock forward with ownership passing (a block's input pair tensor is freed the moment the block rebinds it — upstream keeps one [B, N+g, N+g, c_p] tensor alive through every block for nothing) and, on the stock TriangleMultiplication provider (mode exact), the module's forward with in-place gating and the bmm operands materialised before the un-permuted projections are freed; the same Linear / bmm / LayerNorm calls on the same values in the same order — reference lifetimes only", "forward", 1,
                 ("flag", "--lean-pair", None), "opt/forward/g3cap/g3lean.py (install_lt_release, install_trimul_lean)", ("timings", "lean_pair"),
                 "class 1: byte-identical (ops, inputs and order unchanged; measured by the exact line's PDB digests); one pair tensor fewer at the core's activation peak, which is what sizes the CUDA graph's private pool"),
    "L18": Lever("L18", "wide capture: the hoisted pair featuriser's per-step tail (zi_i + zj_j + relative-position term + template term + conditional term, masked) computed INSIDE the captured region one batch element at a time straight into the pre-padded pair tensor the transform consumes by ownership, and the hoist's loop-invariant terms held as graph-owned buffers refilled in place when a kept graph is re-pointed at a new batch; only the noisy frames, the single features and the pairwise quaternions (the host-synchronising eigen-solve) stay eager", "forward", 1,
                 ("flag", "--wide-capture", None), "driver/g3fast.py (GraphedDenoiser wide) + opt/forward/g3cap/g3lean.py (wide_static_terms, wide_tail) + g3cap.py (rebind_graph)", ("timings", "wide_capture"),
                 "class 1: byte-identical (row-independent ops on per-element slices; measured); the eager pre-part's pair-sized transients and the hoist's whole-batch fill leave the allocator's default pool, the tail's one pair-sized allocation IS the transformer's input"),
    "L19": Lever("L19", "allocator: the CUDA caching allocator with expandable segments (PYTORCH_CUDA_ALLOC_CONF expandable_segments:True, set in the driver process before CUDA initialises — less fragmentation beside the graph pools) and the capacity probe run once per feature-shape signature per process instead of once per batch (its answer is a function of the signature; the per-batch probe's transient is what sets the pass peak once the pools are lean)", "serving", None,
                 ("flag", "--alloc", "expandable"), "opt/genie3_opt/g3batch.py (main: allocator setting; the capacity section: probe_seen)", ("timings", "alloc"),
                 "no numerics (allocator mapping and probe bookkeeping); the CAPACITY line still prints per batch, from the shape's one probe"),
}

# lines the driver prints that make a pass invalid whatever else it printed (design.py: forbidden lines -> the pass is `failed`)
FORBIDDEN_LINES: Tuple[str, ...] = (r"^Traceback \(most recent call last\)", r"AssertionError", r"\[D59\] RNG stream bookkeeping diverged", r"CUDA-graph replay \(mode differs from capture\)")


def flags(ids) -> List[str]:
    """The driver flags of these levers in the given order (the kit's own grammar; a ("default", ...) or ("driver", ...) lever adds none)."""
    out: List[str] = []
    for i in ids:
        sw = LEVERS[i].switch
        if sw[0] == "flag":
            out.append(sw[1])
            if sw[2] is not None:
                out.append(sw[2])
    return out


def evidence(ids) -> Dict[str, Tuple]:
    """{lever id: evidence spec} for the levers that leave one."""
    return {i: LEVERS[i].evidence for i in ids if LEVERS[i].evidence is not None}
