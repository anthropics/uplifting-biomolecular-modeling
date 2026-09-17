"""The lever registry: one entry per switch of the two kits, keyed by the kit's own switch name.

This registry only *describes* a lever — which kit and patch install it, the upstream file it replaces (stock/src path), its class
(forward / datapath), the variants it acts on, and where its numerics statement lives. The per-value meaning of every switch is the
kits' own text (``opt/forward/fast_inference/HOWTO.md``, ``opt/forward/xattempt_addon/HOWTO.md``); no switch value, mode composition or
lever value lives here — rows are read from the kit files by modes.py and applied by the kits' own installed code.

``probe`` grammar — where the kit's own code leaves the record of a lever that fell back to its stock line at call time (read by
report.py: the ``module_attr`` probe through ``lcp_state()``, the ``stderr`` probes as ``FALLBACK_MARKERS``, counted on this process's
stderr; ``activate.completion`` turns them into ``levers_fallback`` / ``partial``):
  ("module_attr", <module>, <attr>)   the attribute's value in the loaded kit module (None when the module is not loaded)
  ("stderr", <prefix>)                the kit reports this lever's fallback on stderr with a line carrying <prefix>
  None                                the lever leaves no readable record beyond the installed file digest
The stderr prefixes are the kit's own printed lines (opt/forward/xattempt_addon/fast, and the package's word modules ``multiseq`` /
``fast_sampler`` that print for ``potts.py``); tests/test_activation_rules.py pins them to the kit sources.

``gates`` — the lever's declared input gates: input classes its mechanism is not defined for, on which the call takes upstream's own
lines for that step with the SAME outputs (stock arithmetic; only the step's speed-up is idle). A gate is not a fallback: it never counts
toward ``partial`` and the run stays the mode's run (exit code unchanged); it is said once per call as the core's per-lever line
``[caliby-opt] LEVER name=<switch> state=skipped reason=<gate> impl=<module> origin=kit <evidence>`` (``report.note_gate``, which refuses a
reason this table does not declare) and recorded in ``opt_manifest.json`` ``lever_gates``. CHANGES.md states every gate by name.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from . import stack

CLASS_FORWARD, CLASS_DATAPATH = "forward", "datapath"
GATE_SYMMETRY_DENSE_J = "symmetry_dense_J"      # CALIBY_FAST_POTTS_PARAMS level 2: symmetry-tied positions take upstream's dense fold of J (potts_params.keep_sparse)


@dataclass(frozen=True)
class Lever:
    name: str                          # the kit's switch name (CALIBY_FAST_* / CALIBY_X_*)
    kit: str                           # opt/forward/<kit>
    patch: str                         # the kit's patch id that installs it
    label: str                         # the kit's own short label (P1/P2 levels, L-A..L-I)
    file: str                          # the replaced upstream file (stock/src path) — the installed copy is the kit's fast/<name>
    cls: str                           # forward | datapath
    variants: Tuple[str, ...]          # variants the lever acts on
    probe: Optional[tuple] = None
    note: str = ""
    gates: Tuple[str, ...] = ()        # declared input gates (reason words of the LEVER state=skipped line); see the module text


ALL = ("single", "ensemble32")
ENS = ("ensemble32",)

LEVERS: Dict[str, Lever] = {
    "CALIBY_FAST_SAMPLER": Lever("CALIBY_FAST_SAMPLER", stack.KIT_PARTNER, "0002", "fast sampler (levels 1, 2)",
                                 "caliby/model/seq_denoiser/denoisers/seq_design/potts.py", CLASS_FORWARD, ALL, ("stderr", "[CALIBY_FAST_SAMPLER]"),
                                 "level 2 = host-sync-free sweep loop + one CUDA graph per call (fast_inference/HOWTO.md); serves upstream's sampling configuration "
                                 "(proposal dlmc, no rejection step, no trajectory, a differentiable penalty) on CUDA tensors (caliby_opt.fast_sampler: the one predicate); "
                                 "any other call takes upstream's sweep loop and prints `[CALIBY_FAST_SAMPLER] ... -> stock sampler for this call`"),
    "CALIBY_FAST_POTTS_PARAMS": Lever("CALIBY_FAST_POTTS_PARAMS", stack.KIT_PARTNER, "0003", "GPU-side Potts parameters (levels P1, P2)",
                                      "caliby/model/seq_denoiser/denoisers/atom_mpnn_denoiser.py", CLASS_FORWARD, ALL, None,
                                      "level 2 keeps the sparse 48-neighbour J (fast_inference/HOWTO.md); its energies are bit-identical only with CALIBY_X_SPARSE_EXACT>=1; "
                                      "gate symmetry_dense_J: a batch with symmetry-tied positions (upstream's constraint CSV `symmetry_pos`) takes upstream's dense fold of J "
                                      "(stock arithmetic, the same outputs; the sparse-J step and CALIBY_X_SPARSE_EXACT idle for that call), said as the LEVER state=skipped line",
                                      (GATE_SYMMETRY_DENSE_J,)),
    "CALIBY_X_SPARSE_EXACT": Lever("CALIBY_X_SPARSE_EXACT", stack.KIT_ADDON, "X001", "L-A exact sparse energy",
                                   "caliby/model/seq_denoiser/denoisers/seq_design/potts.py", CLASS_FORWARD, ALL, ("stderr", "[CALIBY_X_SPARSE_EXACT]"),
                                   "requires CALIBY_FAST_POTTS_PARAMS=2 and CALIBY_FAST_SAMPLER>=1 (xattempt_addon/HOWTO.md); "
                                   "a call whose edge index the kit cannot use takes the kit path and prints `[CALIBY_X_SPARSE_EXACT] ... -> kit path for this call`"),
    "CALIBY_X_LCP": Lever("CALIBY_X_LCP", stack.KIT_ADDON, "X004", "L-D fused LCP kernel (Triton)",
                          "chroma/layers/complexity.py", CLASS_FORWARD, ALL,
                          ("module_attr", "chroma.layers.complexity", "_X_LCP_DISABLED_REASON"),
                          "Triton JIT at first call (a C compiler on PATH); a kernel that cannot build or launch raises `[CALIBY_X_LCP] fused LCP kernel could not "
                          "build/launch ...` at the first served call and the mode refuses by name (never a subset under its name); the failure is recorded in the module (the EXIT line's lcp_kernel=disabled(...))"),
    "CALIBY_X_CLEAN": Lever("CALIBY_X_CLEAN", stack.KIT_ADDON, "X002", "L-B parallel structure cleaning",
                            "caliby/api.py", CLASS_DATAPATH, ALL, ("stderr", "[CALIBY_X_CLEAN="),
                            "values 0 | loader: the stock clean_pdb() per file on the clean workers (torch DataLoader processes forked from the calling process); inert at one clean worker"),
    "CALIBY_X_BG_CIF": Lever("CALIBY_X_BG_CIF", stack.KIT_ADDON, "X003", "L-C background CIF writer",
                             "caliby/eval/eval_utils/seq_des_utils.py", CLASS_DATAPATH, ALL, None,
                             "fork | thread | 0: each batch's CIFs written by background workers of that mode (caliby_opt.bg_writers over opt_core.host.outputs.AsyncWriter), "
                             "joined before run_seq_des returns; a write that fails is the `[CALIBY_X_BG_CIF] ... failed` line and the core's RuntimeError (non-zero exit), never a fallback"),
    "CALIBY_X_CIF_WORKERS": Lever("CALIBY_X_CIF_WORKERS", stack.KIT_ADDON, "X011", "L-C parallel background CIF writers",
                                  "caliby/eval/eval_utils/seq_des_utils.py", CLASS_DATAPATH, ALL, None,
                                  "N workers of the CALIBY_X_BG_CIF mode: a batch's CIFs split into at most N chunks of >= 8 files, at most 2N outstanding; unset/0 = 1 worker"),
    "CALIBY_X_MULTISEQ": Lever("CALIBY_X_MULTISEQ", stack.KIT_ADDON, "X010", "concurrent multi-sequence sampling",
                               "caliby/model/seq_denoiser/denoisers/seq_design/potts.py", CLASS_FORWARD, ALL, ("stderr", "[CALIBY_X_MULTISEQ]"),
                               "the num_seqs_per_pdb sampler calls of a batch issued concurrently (one stream + Philox generator per sequence, all sequences' sweep in one CUDA graph, "
                               "generators positioned by opt_core.diffusion_loop.rng where the serial loop's calls start); rides CALIBY_FAST_SAMPLER=2; "
                               "a call outside the fast sampler's graph path runs the serial calls and prints `[CALIBY_X_MULTISEQ] ... -> serial fallback`; "
                               "call site caliby/model/seq_denoiser/denoisers/atom_mpnn_denoiser.py potts_sample"),
    "CALIBY_X_TIED_DET": Lever("CALIBY_X_TIED_DET", stack.KIT_ADDON, "X006", "L-E deterministic tied aggregation",
                               "caliby/model/seq_denoiser/denoisers/atom_mpnn_denoiser.py", CLASS_FORWARD, ENS, None,
                               "ensemble mode only; no effect in single-structure design (xattempt_addon/HOWTO.md)"),
    "CALIBY_X_ENS_WORKERS": Lever("CALIBY_X_ENS_WORKERS", stack.KIT_ADDON, "X007", "L-F parallel conformer featurisation",
                                  "caliby/eval/eval_utils/inference_dataloader.py", CLASS_DATAPATH, ENS, None, "N loader workers; 0 = stock"),
    "CALIBY_X_PP_CACHE": Lever("CALIBY_X_PP_CACHE", stack.KIT_ADDON, "X002", "L-G Protpardelle-1c model cache",
                               "caliby/api.py", CLASS_FORWARD, ENS, None, "memoises the conformer generator across inputs"),
    "CALIBY_X_PP_NOSYNC": Lever("CALIBY_X_PP_NOSYNC", stack.KIT_ADDON, "X008", "L-H Protpardelle-1c no-sync denoise loop",
                                "protpardelle/core/models.py", CLASS_FORWARD, ENS, None, "installed only when protpardelle is importable at its pin"),
    "CALIBY_X_PP_FASTPDB": Lever("CALIBY_X_PP_FASTPDB", stack.KIT_ADDON, "X009", "L-I Protpardelle-1c fast PDB writer",
                                 "protpardelle/data/pdb_io.py", CLASS_DATAPATH, ENS, None, "installed only when protpardelle is importable at its pin"),
}


def for_variant(variant: str) -> Dict[str, Lever]:
    return {k: v for k, v in LEVERS.items() if variant in v.variants}


def classes(names) -> Dict[str, str]:
    return {n: LEVERS[n].cls for n in names if n in LEVERS}


def stderr_probes() -> Dict[str, str]:
    """switch -> the prefix of the kit's own stderr line reporting that lever's fallback (the ``("stderr", <prefix>)`` probes)."""
    return {n: lv.probe[1] for n, lv in LEVERS.items() if lv.probe and lv.probe[0] == "stderr"}


def declared_gates() -> Dict[str, Tuple[str, ...]]:
    """switch -> its declared input gates (the reason words a ``LEVER state=skipped`` line may carry for it); levers without gates are absent."""
    return {n: lv.gates for n, lv in LEVERS.items() if lv.gates}
