"""The lever registry: one entry per lever the kits ship for RFdiffusion-1 (the base kit RFDIFFUSION1_FAST_INFERENCE_KIT and the
dense SE(3)-layer add-on RFD_SE3FAST_ADDON), keyed by the id the kits' own documents use.

A lever is described here — which kit file implements it, the switch the kit's own line uses for it (a driver flag, written
exactly as the kit writes it), its class (forward / datapath / orchestration), its numerics tier against stock, and the
line the kit prints when it has applied the lever in a process (`evidence`, read from the driver log by design.py) — and composed
elsewhere: modes are lists of these ids in modes.py, the only place a mode names its switches. Nothing here applies anything;
the kits' own drivers do.

Switch grammar: ("flag", <driver flag>, <value>) is passed on the driver's command line; ("driver", <path>) names the resident
driver itself; ("env", <NAME>, <value>) is exported to the driver process by the mode's row (modes.Mode.env, stack.driver_environment)
— the SE(3) add-on's own switch RFD_SE3FAST is the one lever switched that way, and driver_run.py acts on it in the driver process
(the add-on's documented in-driver route: rfd_se3fast.apply() before the model imports). The drivers' other RFD_* knobs are not
levers of a mode: stack.DROP_ENV_PREFIXES keeps them out of every child process, the row's own names excepted.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

KIT_BASE, KIT_SE3 = "fast_inference", "se3fast_addon"      # opt/forward/<kit>
PKG = "rfdiffusion1_opt"                                 # levers this package implements itself (kit_file relative to opt/rfdiffusion1_opt/), armed in the driver process by driver_run.py
KIT_DIRS = {KIT_BASE: "opt/forward/fast_inference", KIT_SE3: "opt/forward/se3fast_addon"}
SE3FAST_ENV, SE3FAST_MODE = "RFD_SE3FAST", "t2"                # the SE(3) add-on's own switch and the value of its Triton line (rfd_se3fast/__init__.py)
BASE_DRIVER = "drivers/rfd_bench.py"                    # the base kit's resident driver (KIT_BASE)
FORWARD, DATAPATH, ORCHESTRATION = "forward", "datapath", "orchestration"


@dataclass(frozen=True)
class Lever:
    id: str
    kit: str                                   # KIT_BASE | KIT_SE3
    kit_file: str                              # file inside the kit directory that implements it
    switch: Tuple[str, ...]                    # ("flag", flag, value) | ("driver", relpath) | ("env", NAME, value)
    class_4: str                               # forward | datapath | orchestration
    tier: Optional[int]                        # 1 = byte-equal to stock under the deterministic recipe; 2 = tolerance; None = no numerics
    what: str
    doc: str                                   # where the kit states it (file:line inside the kit directory)
    evidence: Optional[str] = None             # regex of the line the kit prints when the lever is applied in a process
    evidence_doc: str = ""                     # file:line of that print


LEVERS: Dict[str, Lever] = {
    # --- base kit RFDIFFUSION1_FAST_INFERENCE_KIT (opt/forward/fast_inference) ---------------------------------------------------------
    "U1": Lever("U1", KIT_BASE, "drivers/rfd_bench.py", ("driver", BASE_DRIVER), ORCHESTRATION, None,
                "resident process: the model loaded once, many designs per process; the same sampler, seeding and writers as scripts/run_inference.py",
                "CHANGES.md 'Levers'; drivers/rfd_bench.py:3-15"),
    "C1": Lever("C1", KIT_BASE, "drivers/rfd_fastpath.py", ("flag", "--fastpath", "chain_breaks,full_graph,rbf,msa_index"), FORWARD, 1,
                "per-design constants memoised (chain breaks, full graph, rbf, msa index)", "CHANGES.md 'Levers'; drivers/rfd_bench.py:33,92-93",
                evidence=r"^fastpath levers: ", evidence_doc="drivers/rfd_bench.py:93"),
    "P": Lever("P", KIT_BASE, "drivers/rfd_prep.py", ("flag", "--prep", "1"), FORWARD, 1,
               "fast _preprocess: the dead CPU template t2d skipped, per-design constants memoised", "CHANGES.md 'Levers'; drivers/rfd_bench.py:37,129-131",
               evidence=r"^prep lever: active = True", evidence_doc="drivers/rfd_bench.py:131"),
    "E_einsum": Lever("E_einsum", KIT_BASE, "drivers/rfd_einsum.py", ("flag", "--einsum-route", "1"), FORWARD, 1,
                      "two opt_einsum signatures routed to torch.einsum, checked byte-equal at start-up", "CHANGES.md 'Levers'; drivers/rfd_bench.py:36,120-122",
                      evidence=r"^einsum lever E: ", evidence_doc="drivers/rfd_bench.py:122"),
    "W1": Lever("W1", KIT_BASE, "drivers/rfd_fullgraph.py", ("flag", "--fullgraph", "1"), FORWARD, 1,
                "one CUDA graph for embeddings + templates + the 36 blocks (the low-memory alternative to C3)", "drivers/rfd_bench.py:34,123-128",
                evidence=r"^fullgraph mode: applied = True", evidence_doc="drivers/rfd_bench.py:128"),
    "K2": Lever("K2", KIT_BASE, "drivers/rfd_bench.py", ("flag", "--triton-ln", "1"), FORWARD, 2,
                "Triton row LayerNorm replaces F.layer_norm (last-bit numerics change): the shared core's carried kernel opt_core/kernels/rfd_layernorm.py, "
                "routed to the driver's `import rfd_layernorm` by driver_run.route_core_kernels", "CHANGES.md 'Levers'; drivers/rfd_bench.py:35,132-134",
                evidence=r"^triton LN lever \(Tier 2\): active = True", evidence_doc="drivers/rfd_bench.py:134"),
    "TF32": Lever("TF32", KIT_BASE, "drivers/rfd_bench.py", ("flag", "--tf32", "1"), FORWARD, 2,
                  "TF32 tensor-core math for the fp32 GEMMs and convolutions: the driver sets torch.backends.cuda.matmul.allow_tf32 and "
                  "torch.backends.cudnn.allow_tf32 from its own switch before the model loads (cuBLAS / cuDNN round GEMM inputs to 10 mantissa bits, "
                  "accumulate in fp32; the SE(3) add-on's Triton dots stay at input_precision=ieee); per-seed trajectories differ from stock, run-to-run deterministic",
                  "CHANGES.md 'Levers'; drivers/rfd_bench.py:31,45-46,87",
                  evidence=r'^\s*"tf32": true,?\s*$', evidence_doc="drivers/rfd_bench.py:62-69 (the driver's run manifest, printed at start-up one key per line)"),
    "IO1": Lever("IO1", PKG, "pdbio.py", ("env", "RFD_PDBIO", "1"), DATAPATH, 1,
                 "the PDB writers of rfdiffusion.util (writepdb: the design's backbone file; writepdb_multi: the two 50-model trajectory files) re-expressed over "
                 "numpy arrays — the same format string on the same numbers, one host copy per model in place of per-atom tensor indexing; byte-equal files "
                 "(the first call per argument signature is compared with upstream's own writer in the driver process: a difference keeps upstream's bytes and exits 5)",
                 "opt/rfdiffusion1_opt/pdbio.py; driver_run.py (armed before the driver's `from rfdiffusion.util import writepdb_multi, writepdb`, drivers/rfd_bench.py:81)",
                 evidence=r"^pdb writer lever IO1: armed", evidence_doc="opt/rfdiffusion1_opt/pdbio.py arm()"),
    # --- SE(3) add-on RFD_SE3FAST_ADDON v0.3.0 (opt/forward/se3fast_addon): sits on the base kit's line -------------------------------
    "T2": Lever("T2", KIT_SE3, "rfd_se3fast/__init__.py", ("env", SE3FAST_ENV, SE3FAST_MODE), FORWARD, 2,
                "dense destination-major SE(3)-Transformer layer with two fused Triton kernels (radial-MLP trunk; last radial layer x per-edge "
                "contraction) in all 40 Str2Str calls per step, fp32, TF32 off, tl.dot ieee: re-associated reductions — not byte-equal to stock, "
                "run-to-run deterministic; dense (L, L, .) edge tensors (~2.5 GB per call at L >= 700)",
                "CHANGES.md 'Levers'; rfd_se3fast/__init__.py:32-88",
                evidence=r"^\[rfd_se3fast\] v0\.3\.0 applied: mode=t2 scope=all se3_fn=se3_dense_forward_triton triton=\d", evidence_doc="rfd_se3fast/__init__.py:85-87"),
}

# lines that, in a driver log, mean a lever did not run as the mode names it (design.py reports them by name)
FORBIDDEN_LINES = (r"REFUSED", r"^Traceback \(most recent call last\)", r"AssertionError")


def flags(ids) -> list:
    """The driver flags of these levers, in the order given (each ("flag", f, v) -> [f, v])."""
    out = []
    for i in ids:
        s = LEVERS[i].switch
        if s[0] == "flag":
            out += [s[1], s[2]]
    return out


def env(ids) -> Dict[str, str]:
    """The environment switches of these levers ({NAME: value}): the SE(3) add-on's RFD_SE3FAST for T2; empty for every other lever."""
    return {LEVERS[i].switch[1]: LEVERS[i].switch[2] for i in ids if LEVERS[i].switch[0] == "env"}
