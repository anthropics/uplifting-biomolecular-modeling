"""The lever registry: one entry per lever, keyed by the lever's name (modes.KIT_MODES composes them into modes).

This registry only *describes* each lever — which file applies it, its switch, its class (forward / datapath / serving /
orchestration), its numerics tier, the form it acts in (in-process import, the runner process) and the module record that shows it
was applied in a process (`probe`). No lever value, switch value or mode composition lives here: modes are modes.py, switch values
are read from the runner (modes.kit_defaults), and application is the lever modules' own code.

`probe` grammar (read by stack.classify from the lever modules' own state after activation):
  ("stats", <module>, <key>)   sys.modules[<module>].STATS[<key>] truthy (xa_fastinit / xa_hoist / sz_levers / fl_levers / hl_levers keep a STATS dict)
  ("mode", <module>, <key>)    sys.modules[<module>].STATS[<key>] == the mode's exported value of the lever's own switch
                               (bg_graph_patch.STATS["mode"] vs BG_GRAPH; fl_levers.STATS["attn_backend"] vs FL_ATTN_BACKEND)
  ("acc", <module>, <key>)     sys.modules[<module>]._acc[<key>] truthy (bg_hook keeps its records in `_acc`)
  None                         a process form: the runner, shown by its own stderr lines (stack.RUNNER_LINES)

`fell_back` grammar (read by stack.runtime_fallbacks at the end of an activated process: the module records that show a lever STOPPED
serving during the run — the in-process reading of the same events the launching verb reads off the child's lines, stack.RUNNER_DISABLED_LINES):
  ("set", <module>, <key>)     sys.modules[<module>].STATS[<key>] truthy: the lever stopped serving and the value is its own reason
                               (xa_hoist disabled_reason; xa_fastinit fallback_reason — a load whose stock initialisers were replayed;
                               bg_graph_patch capture_error; fl_levers attn_/dit_disabled_reason)
  ("gate", <module>, <prefix>) sys.modules[<module>].gate() — the module's own problem sentences (its GATE-FAIL words) that start with
                               <prefix>; "" = every sentence no other lever of the same module claims by its prefix (fl_levers names
                               cond_dedup's problems by dedup site — `s_path: …`, `cond_dedup: …` — so cond_dedup takes what `attn` /
                               `dit` do not claim, `attn_mask: …` going to attn_bf16; hl_levers has one lever: async_writer takes all)
`serves`: (<module>, <function>) — ``function(task)`` returns the reason the lever cannot serve upstream's model step ``task`` (a
boltzgen Predict about to ``run``), None when it can — a sentence that stands alone on the refusal line; read by stack.step_refusals
before the step's model loads (xa_fastinit.unserved_reason: a DataLoader in the step's own process).
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

FORM_INPROC, FORM_RUNNER = "inproc", "runner"


@dataclass(frozen=True)
class Lever:
    name: str
    kit: str                      # modes.KIT_XATTEMPT | KIT_PARTNER | KIT_SIZE_LEVERS | KIT_FAST_LEVERS | KIT_HOST_LEVERS
    file: str                     # the kit file that applies it (relative to the kit directory)
    switch: Optional[str]         # the kit's own switch, None when the lever is the form itself
    what: str
    klass: str                    # forward | datapath | serving | orchestration
    tier: str                     # "exact" (bit-identical to seeded stock) for every lever of `exact` / `big`; "2" (no
                                   # bitwise claim) for the `fast` mode's own levers (cond_dedup, attn_bf16, attn_cudnn, dit_fused)
    form: str                     # FORM_*
    probe: Optional[Tuple[str, str, str]]
    doc: str = ""
    fell_back: Tuple[Tuple[str, str, str], ...] = ()
    serves: Optional[Tuple[str, str]] = None


LEVERS: Dict[str, Lever] = {
    # --- forward/fast_inference ----------------------------------------------------------------------------------------------------
    "inproc": Lever("inproc", "forward/fast_inference", "src/bg_inproc.py", None,
                    "the GPU steps of one configured job run in one process (no per-step interpreter start, one model build per step)",
                    "orchestration", "exact", FORM_RUNNER, None, doc="bg_inproc.py l.1-9; xa_run.py l.41-47"),
    "graph_sampler": Lever("graph_sampler", "forward/fast_inference", "src/bg_graph_patch.py", "BG_GRAPH",
                           "CUDA-graph replay of the design-step diffusion sampler (`graph`); design step only",
                           "forward", "exact", FORM_INPROC, ("mode", "bg_graph_patch", "mode"), doc="bg_graph_patch.py l.21 (STATS), l.299-313 (apply); bg_hook.py l.46-48 (applied at import when BG_GRAPH != off)",
                           fell_back=(("set", "bg_graph_patch", "capture_error"),)),
    # --- forward/xattempt_addon ----------------------------------------------------------------------------------------------------
    "fastinit": Lever("fastinit", "forward/xattempt_addon", "src/xa_fastinit.py", "XA_FAST_INIT",
                      "L1: checkpoint-loaded parameters skip their random initialiser (strict load overwrites every one)",
                      "serving", "exact", FORM_INPROC, ("stats", "xa_fastinit", "enabled"), doc="xa_fastinit.py l.2-14 (what), l.26 (STATS); cold start: model construction at checkpoint load, not the forward pass",
                      fell_back=(("set", "xa_fastinit", "fallback_reason"),), serves=("xa_fastinit", "unserved_reason")),
    "hoist": Lever("hoist", "forward/xattempt_addon", "src/xa_hoist.py", "XA_HOIST",
                   "L4: the diffusion transformer's step-invariant attention masks — the 24 token layers' and the atom encoder's / decoder's six windowed ones — built once per structure",
                   "forward", "exact", FORM_INPROC, ("stats", "xa_hoist", "enabled"), doc="xa_hoist.py l.2-12 (what), l.26-27 (STATS), l.39 (disabled_reason)",
                   fell_back=(("set", "xa_hoist", "disabled_reason"),)),
    # --- forward/size_levers — big mode only -------------------------------------------------------------------------------------
    "td_chunk": Lever("td_chunk", "forward/size_levers", "src/sz_levers.py", "SZ_TD_CHUNK",
                      "TokenDistanceModule feature construction row-chunked: the same per-pair math, only GEMM shapes change",
                      "forward", "exact", FORM_INPROC, ("stats", "sz_levers", "td_enabled"), doc="src/sz_levers.py (STATS td_enabled/td_calls/td_rows/td_site)"),
    # --- forward/fast_levers — fast mode only --------------------------------------------------------------------------------------
    "cond_dedup": Lever("cond_dedup", "forward/fast_levers", "src/fl_levers.py", "FL_COND_DEDUP",
                        "the sampler's multiplicity-invariant conditioning (SingleConditioning; each token layer's AdaLN s-path and output gates) computed on one row per step and broadcast over the B designs of a diffusion batch (opt_core.capture.hoist.RowDedup, strategy F7.row_dedup); GEMM row count changes",
                        "forward", "2", FORM_INPROC, ("stats", "fl_levers", "dedup_enabled"), doc="src/fl_levers.py (STATS dedup_enabled/dedup_sites; per-site RowDedup census in the exit stats line)",
                        fell_back=(("gate", "fl_levers", ""),)),
    "attn_bf16": Lever("attn_bf16", "forward/fast_levers", "src/fl_levers.py", "FL_ATTN_BF16",
                       "the token transformer's pair-biased self-attention (and the atom transformers' windowed 32 x 128 attention) as one fused bfloat16 call through opt_core.attn.sdpa_bias (backend efficient, strategy F5.flash_attn_dense; every eligible call accounted by an opt_core.attn.size_gate.SizeGate) inside the fp32 sampler; the fp32 attn_mask converted on one slice (RowDedup) and broadcast as the bias, output back to fp32",
                       "forward", "2", FORM_INPROC, ("stats", "fl_levers", "attn_enabled"), doc="src/fl_levers.py (STATS attn_enabled/attn_site/attn_calls/attn_eligible/attn_event/passthrough_by; the size gate's census in the exit stats line)",
                       fell_back=(("set", "fl_levers", "attn_disabled_reason"), ("gate", "fl_levers", "attn"))),
    "attn_cudnn": Lever("attn_cudnn", "forward/fast_levers", "src/fl_levers.py", "FL_ATTN_BACKEND",
                        "attn_bf16's call pinned to torch's cuDNN fused attention (opt_core.attn.sdpa_bias backend pin `cudnn`: reads the stride-0 bf16 bias like the memory-efficient kernel, faster at every sampler shape; a pin the stack cannot honour is the core's refusal by name and attn_bf16's DISABLED line, never another kernel unannounced)",
                        "forward", "2", FORM_INPROC, ("mode", "fl_levers", "attn_backend"), doc="src/fl_levers.py (STATS attn_backend; the attn_bf16 installed / evidence lines carry backend=<pin>)",
                        fell_back=(("set", "fl_levers", "attn_disabled_reason"),)),
    "dit_fused": Lever("dit_fused", "forward/fast_levers", "src/fl_levers.py", "FL_DIT_FUSED",
                       "the 24 token DiffusionTransformer layers' fp32 elementwise statements around the GEMMs (AdaLN, the sigmoid output gates with the residual adds, the transition's SwiGLU with its two input GEMMs run as one) as the shared core's fused Triton row kernels `dtk_kernels` (opt_core.kernels.route, sha-held; strategy LOCAL.dit_fused_kernels), fed the cond_dedup sites' one-row conditioning blocks; accounted per layer call by a SizeGate",
                       "forward", "2", FORM_INPROC, ("stats", "fl_levers", "dit_enabled"), doc="src/fl_levers.py (STATS dit_enabled/dit_site/dit_layers_served/dit_eligible/dit_passthrough_by; the size gate's census in the exit stats line)",
                       fell_back=(("set", "fl_levers", "dit_disabled_reason"), ("gate", "fl_levers", "dit"))),
    # --- host/writer_levers — every kit mode ---------------------------------------------------------------------------------------
    "async_writer": Lever("async_writer", "host/writer_levers", "src/hl_levers.py", "HL_ASYNC_WRITER",
                          "upstream's DesignWriter.write_on_batch_end run unchanged on host copies of each unit's tensors (opt_core.host.outputs.to_host, one pass) in background worker processes of opt_core.host.outputs.AsyncWriter (strategy F6.output_overlap; bounded queue, drained at the predict loop's end, tally at exit, a failed write fails the run): writing unit i overlaps computing unit i+1; the same files, names and bytes",
                          "orchestration", "exact", FORM_INPROC, ("stats", "hl_levers", "writer_enabled"), doc="src/hl_levers.py (STATS writer_enabled/mode/workers/batches/designs; the core's TALLY line and the CENSUS word async_writer[W=…,submitted=…,written=…,failed=…] at exit)",
                          fell_back=(("gate", "hl_levers", ""),)),
}
