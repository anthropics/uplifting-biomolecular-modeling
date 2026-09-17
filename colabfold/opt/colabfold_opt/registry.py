"""The lever registry: one entry per switch the carried kit reads, keyed by the kit's own environment-variable name.

It *describes* each lever — which kit file installs it, what it changes, its class (forward / datapath / serving /
orchestration), its numerics class, its switch and the state that shows it ran (``probe``). No mode composition lives here: the modes and the levers each applies are ``modes.TABLE``
(``resolve()``); the levers with ``in_mode=False`` are reachable by the kit's own switch names and in no mode.

``probe`` grammar (read by ``applied()`` from the kit's own state, after the kit's ``enable()``):
  ("state", <key>)                                   ``af2_pallas_attn._STATE[<key>]`` is true (``:16``)
  ("class_attr", <module>, <class>, <attribute>)     the patched class carries the marker (``Attention._pallas_patched``, ``:96``)
  ("env", <variable>, <value>)                       the switch is a value in the environment the kit reads at the call (the backward-only
                                                     and ``_ALL`` switches: the core kernel's ``af2_flash_pallas.make_flash_attention``, ``af2_pallas_attn.py:77``)
  ("module_state", <module>, <key>)                  ``sys.modules[<module>]._STATE[<key>]`` is true (the package's own lever, device_resident.py)
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from . import modes as _modes

SWITCH = f"{_modes.KIT_SWITCH}={_modes.KIT_SWITCH_ON} in the environment before the kit module is imported (af2_pallas_attn.py:124-125), or enable() (:69)"


@dataclass(frozen=True)
class Lever:
    name: str
    kit_file: str
    what: str
    cls: str                                   # forward | datapath | serving | orchestration
    tier: str                                  # the kit's own numerics reading
    in_mode: bool                              # applied by a mode (modes.resolve) or reachable by the kit's own switch name only
    probe: Tuple
    lines: str = ""
    doc: str = ""
    deployment: bool = False                   # placed by every mode of the kit when the deployment names it (no numerics; not in the ACTIVE line's levers)
    impl: str = "af2_pallas_flash"             # the LEVER line's impl= slot: the module that implements the lever
    origin: str = "kit"                        # the LEVER line's origin= slot: kit (this tree) | core (opt_core)
    strategy: str = ""                         # the LEVER line's strategy= slot: a canonical id of opt_core/STRATEGIES.json ("" = none named)


LEVERS: Dict[str, Lever] = {
    _modes.HOST_LEVER: Lever(_modes.HOST_LEVER, "colabfold_opt/device_resident.py",
                             "alphafold.model.model.RunModel rebound to a subclass whose jitted apply keeps host NumPy parameters, features "
                             "and recycling inputs resident on the device across the Python recycle loop (model.py:164-192), casts the outputs "
                             "to float16 on the device, hands the loop only the scalars it reads per recycle (the stop rules' ranking_confidence / "
                             "tol, the logged means) and fetches the output tree once after the last recycle, where stock uploads every leaf and "
                             "pulls the whole tree back at every recycle and casts on the host", "datapath",
                             "placement only: the same executables on the same arguments, IEEE round-to-nearest casts on both routes — "
                             "byte-identical outputs expected, measured per mode rather than constructed", True,
                             ("module_state", _modes.HOST_LEVER_MODULE, "enabled"), lines="device_resident.py", doc="CHANGES.md §The levers",
                             impl=_modes.HOST_LEVER_MODULE, origin="kit", strategy="F6.host_sync_elimination"),
    _modes.SUBBATCH_LEVER: Lever(_modes.SUBBATCH_LEVER, "colabfold_opt/subbatch.py",
                                 "alphafold.model.config.model_config wrapped so every model configuration colabfold builds carries "
                                 "global_config.subbatch_size = the value opt_core.jax_design.subbatch_policy decides for the run's largest input "
                                 "against the visible device (128 when the chunk-128 program fits, else stock's 4, named): every inference_subbatch "
                                 "scan — triangle attention starting/ending node, MSA row / column / global-column attention, every Transition — "
                                 "runs 128 rows per launch instead of 4", "forward",
                                 "the same contractions over the same reduction lengths in a different XLA program (GEMM tilings chosen per shape): "
                                 "Tier 2, inside stock's seed band; byte equality with stock is measured per mode, not constructed", True,
                                 ("module_state", _modes.SUBBATCH_LEVER_MODULE, "enabled"), lines="subbatch.py", doc="CHANGES.md §The levers",
                                 impl="opt_core.jax_design.subbatch_policy", origin="core", strategy="F7.jax_subbatch"),
    _modes.TRIMUL_LEVER: Lever(_modes.TRIMUL_LEVER, "colabfold_opt/trimul_pallas.py",
                               "alphafold.model.modules.TriangleMultiplication rebound to a subclass whose call reads the module's own parameters and runs "
                               "the shared core's JAX-family provider face by the mode's TIER word (opt_core.kernels.pallas.serve.triangle_multiplication, "
                               "word=fast | big: the row the provider measured fastest / lowest-peak for the call's cell — compute capability x dtype x "
                               "family x N — on both measured cards today its carried fused Pallas triangle multiplication: prologue kernel, ONE cuBLAS batched "
                               "GEMM on channel-major planes, epilogue kernel); every call site (Evoformer, extra-MSA stack, template pair stack), outgoing and "
                               "incoming; a call the face does not serve, or whose cell names the stock statement, runs the stock body, counted under its reason", "forward",
                               "bf16 operands, f32 accumulation, LayerNorm statistics and sigmoids in f32, in-register intermediates never rounded: fewer rounding "
                               "points than the stock bf16 body (closer to an f32 reference than stock's own): Tier 2, inside stock's seed band; deterministic run to run", True,
                               ("module_state", _modes.TRIMUL_LEVER_MODULE, "enabled"), lines="trimul_pallas.py; opt_core/kernels/pallas/serve.py", doc="CHANGES.md §The levers",
                               impl="opt_core.kernels.pallas.serve:triangle_multiplication", origin="core", strategy="F2.trimul"),
    _modes.LEVER: Lever(_modes.LEVER, "af2_pallas_flash/af2_pallas_attn.py",
                        "alphafold.model.modules.Attention rebound to a subclass whose __call__, for calls carrying a pair bias — triangle attention "
                        "starting/ending node (Evoformer, extra-MSA stack, template pair stack) and MSA-row attention —, runs the module's projections heads-major, "
                        "ONE call of the shared core provider's attention face by the mode's tier word (opt_core.kernels.pallas.serve.attention, word fast | big, "
                        "the site's family; the provider's measured cell names the kernel per card, size and word: its pre-compiled triangle-attention rows, its Pallas "
                        "flash rows, a library row or the stock statement by name — triattn_xla.bind puts that op in the adapter's FLASH_OP) and the stock gating / "
                        "output projection; projections, gating and the output projection untouched; stock math on CPU or key_dim != value_dim",
                        "forward", "class 3, Tier 2: bf16 attention arithmetic re-associated (fp32 softmax statistics and accumulation in every served row), outputs "
                        "differ from stock at the bf16 rounding level (README.md:14); deterministic run to run (README.md:16)", True,
                        ("state", "enabled"), lines="af2_pallas_attn.py:58-66,69-103,124-125; triattn_xla.py (bind, provider_op); opt_core/kernels/pallas", doc="README.md:14,16,23-25",
                        impl="opt_core.kernels.pallas", origin="core", strategy="F1.flash_triatt"),
    "AF_PALLAS_ATTN_ALL": Lever("AF_PALLAS_ATTN_ALL", "af2_pallas_flash/af2_pallas_attn.py",
                                "widens the patch to the calls without a pair bias (MSA-column and template attention)",
                                "forward", "the kit's default 0 on every row of its record; not measured on this route", False,
                                ("env", "AF_PALLAS_ATTN_ALL", "1"), lines="af2_pallas_attn.py:77", doc="README.md:26"),
    "AF_PALLAS_ATTN_PRECISE_BWD": Lever("AF_PALLAS_ATTN_PRECISE_BWD", "opt_core/kernels/pallas_attn/af2_flash_pallas.py",
                                        "backward pass only (the custom VJP): the precise dK/dV path", "forward",
                                        "backward only — no effect on a forward-only scoring run", False,
                                        ("env", "AF_PALLAS_ATTN_PRECISE_BWD", "1"), lines="af2_flash_pallas.py:350"),
    "AF_PALLAS_ATTN_DBIAS_F32": Lever("AF_PALLAS_ATTN_DBIAS_F32", "opt_core/kernels/pallas_attn/af2_flash_pallas.py",
                                      "backward pass only: the pair-bias gradient accumulated in f32", "forward",
                                      "backward only — no effect on a forward-only scoring run", False,
                                      ("env", "AF_PALLAS_ATTN_DBIAS_F32", "1"), lines="af2_flash_pallas.py:367"),
    "AF_PALLAS_ATTN_BWD_BATCH_CHUNK": Lever("AF_PALLAS_ATTN_BWD_BATCH_CHUNK", "opt_core/kernels/pallas_attn/af2_flash_pallas.py",
                                            "backward pass only: the batch chunk of the dK/dV kernel (scratch memory bound; same math)", "forward",
                                            "backward only — no effect on a forward-only scoring run (README.md:27)", False,
                                            ("env", "AF_PALLAS_ATTN_BWD_BATCH_CHUNK", "<n>"), lines="af2_flash_pallas.py:380", doc="README.md:27"),
}

LEVERS[_modes.MSA_LEVER] = Lever(_modes.MSA_LEVER, "colabfold_opt/msa_attn.py",
                                 "alphafold.model.modules.Attention rebound (after AF_PALLAS_ATTN's own rebinding, wrapping it) to a subclass that sends the two "
                                 "call classes the pair-bias lever leaves on XLA — extra-MSA row attention (8 channels per head, with the pair bias) and every "
                                 "bias-free call the shared core's JAX-family provider has a measured family for (MSA column attention) — through the provider's "
                                 "attention face by the mode's tier word (opt_core.kernels.pallas serve.attention: the measured cell names the row — a Pallas "
                                 "flash row, cuDNN's fused attention or the XLA statement — per card, dtype and size; at the column site every row but the one "
                                 "MSA_COL_CUDNN binds); a call of neither class, or one the provider has no family for, proceeds to the wrapped class, counted "
                                 "under its reason", "forward",
                                 "the provider's tolerance class at the two sites (bf16 operands, fp32 accumulation, exact softmax): Tier 2, inside stock's seed "
                                 "band; deterministic run to run", True,
                                 ("module_state", _modes.MSA_LEVER_MODULE, "enabled"), lines="msa_attn.py", doc="CHANGES.md §The levers",
                                 impl="opt_core.kernels.pallas:attention", origin="core", strategy="F5.flash_attn_dense")
LEVERS[_modes.TRIATTN_LEVER] = Lever(_modes.TRIATTN_LEVER, "colabfold_opt/triattn_xla.py",
                                     "the shared core provider's pre-compiled triangle-attention row (opt_core.kernels.triattn_xla: the sealed CUDA kernels on compute "
                                     "capability 9.0 / 8.0, the ahead-of-time compiled K2B cubins, one XLA-FFI launch per sub-batch chunk on heads-major operands, the pair "
                                     "mask rows, no logits materialised) INSIDE AF_PALLAS_ATTN's binding: applied, the provider's order for the tier word reaches the row where "
                                     "its measured cell names it (per card, family and size); not applied, the binding narrows the word to the provider's other rows; a call the "
                                     "row does not take (the cell names another row, the bridge refuses at the call) is served by the provider's next row, counted under its reason", "forward",
                                     "bf16 tensor-core products, fp32 softmax, bias and accumulation, in the class of the Pallas flash rows (rel-rms vs an fp32 reference 2-5e-3); "
                                     "a fully-masked (padded) row attends uniformly, as under stock: Tier 2, inside stock's seed band; deterministic run to run", True,
                                     ("module_state", _modes.TRIATTN_LEVER_MODULE, "enabled"), lines="triattn_xla.py", doc="CHANGES.md §The levers",
                                     impl="opt_core.kernels.triattn_xla", origin="core", strategy="F1.flash_triatt")
LEVERS[_modes.COL_LEVER] = Lever(_modes.COL_LEVER, "colabfold_opt/msa_col_cudnn.py",
                                 "alphafold.model.modules.Attention rebound (after PALLAS_MSA's rebinding, wrapping it) to a subclass that sends MSA column attention "
                                 "(no pair bias, 8 heads x 32 channels, keys = the MSA depth) through the shared core's JAX-family provider by the mode's tier word "
                                 "with every row admitted (opt_core.kernels.pallas serve.attention kind msacol: on compute capability 9.0 its measured cell names "
                                 "the cudnn row — the library's fused attention in the key-lengths form, no dense mask operand — elsewhere the row measured first "
                                 "there); the pair-biased sites pass through, template attention and calls the provider refuses proceed to the wrapped class, "
                                 "counted under their reason", "forward",
                                 "the provider's tolerance class (cuDNN / Pallas flash attention: bf16 operands, f32 accumulation and softmax statistics): Tier 2, "
                                 "inside stock's seed band; deterministic run to run", True,
                                 ("module_state", _modes.COL_LEVER_MODULE, "enabled"), lines="msa_col_cudnn.py", doc="CHANGES.md §The levers",
                                 impl="opt_core.kernels.pallas:attention", origin="core", strategy="F5.sdpa_cudnn")
LEVERS[_modes.TEMPL_LEVER] = Lever(_modes.TEMPL_LEVER, "colabfold_opt/templ_dedup.py",
                                   "alphafold.model.modules_multimer.TemplateEmbedding rebound to a subclass whose scan over the template rows compares each row "
                                   "with its predecessor on the device (aatype, atom positions, atom mask) and reuses the predecessor's SingleTemplateEmbedding output "
                                   "for an equal row instead of recomputing it (hk.cond: only the taken branch runs) — colabfold's default run carries four identical "
                                   "mock template rows (colabfold/batch.py mk_mock_template, padded to MAX_TEMPLATES=4), so one row is embedded and three reuse it, "
                                   "every recycle; rows that differ (real templates) are each embedded as stock does, counted per predict on the host "
                                   "(model.RunModel.predict rebound to a counting subclass: templates / embedded / reused / unique_hist); the monomer route and a "
                                   "template-less model configuration and stock's --use-dropout are named step-asides (fallback_by=monomer_route|template_disabled|dropout_enabled)", "forward",
                                   "stock's sum in stock's order over stock's operands (the reused embedding IS the value the previous row produced): byte-identical "
                                   "to the mode without it on 5 of 6 measured inputs, rounding-level different on one (the cond's compute branch compiles apart from "
                                   "the plain scan body) — Tier 2 with fast, inside stock's seed band", True,
                                   ("module_state", _modes.TEMPL_LEVER_MODULE, "enabled"), lines="templ_dedup.py", doc="CHANGES.md §The levers",
                                   impl=_modes.TEMPL_LEVER_MODULE, origin="kit", strategy="LOCAL.colabfold.template_dedup")
LEVERS[_modes.TRANSITION_LEVER] = Lever(_modes.TRANSITION_LEVER, "colabfold_opt/transition.py",
                                        "AlphaFold's Transition module (LayerNorm -> Linear -> ReLU -> Linear; the Evoformer's msa / pair transitions, the extra-MSA "
                                        "stack's, the template pair stack's) bound to the shared core's JAX-family provider face opt_core.kernels.pallas.serve.transition "
                                        "by the mode's tier word (fast | big): the provider serves the row it measured fastest per call class (mlp_transition / "
                                        "cd_transition fused kernels, or the stock statement — then stock's own body runs, counted under fallback_by=cell_stock / no_cell)",
                                        "forward", "tolerance class (the provider's tol rows: bf16 operands, f32 accumulation, another summation order than the stock GEMMs') "
                                        "— Tier 2 with fast; f32 activations resolve to the stock statement", True,
                                        ("module_state", _modes.TRANSITION_LEVER_MODULE, "enabled"), lines="transition.py", doc="CHANGES.md §The levers",
                                        impl="opt_core.kernels.pallas.serve:transition", origin="core", strategy="LOCAL.fused_transition")
LEVERS[_modes.TP_LEVER] = Lever(_modes.TP_LEVER, "colabfold_opt/big.py",
                                "big --n_gpu P>1: AlphaFold-Multimer end to end with the pair representation [N, N, C] split by rows over P local GPUs — "
                                "the prologue's pair terms born row-sharded, the Evoformer / extra-MSA / template pair stacks as shard_map regions running the stock "
                                "block bodies on this device's row block, the recycle carry row-sharded across the jit boundary, the structure module's pair reads and "
                                "the distogram / aligned-error heads on rows (opt_core.mem.rowpair_jax.alphafold, site groups trunk+model+heads, with the collectives of "
                                "opt_core.mem.rowpair_jax); MSA / single / parameters replicated (the stated floor); DEVICE_RESIDENT binds the jitted apply through the "
                                "recipe's leaf shardings; the kit pads N to a multiple of P (monomer pad_len; multimer feature dict at RunModel.predict)", "forward",
                                "fast class: complete contractions per element on one device (no cross-device partial sums), byte movement elsewhere; "
                                "not bitwise against P=1 (XLA fuses the local blocks differently); P=1 installs nothing", True,
                                ("module_state", _modes.TP_LEVER_MODULE, "enabled"), lines="big.py", doc="CHANGES.md §Modes",
                                impl="opt_core.mem.rowpair_jax", origin="core", strategy="F7.tensor_parallel")
LEVERS["XLA_CACHE"] = Lever("XLA_CACHE", "colabfold_opt/xla_cache.py",
                            "JAX's persistent compilation cache placed under <COLABFOLD_OPT_JIT_ROOT>/<stack key>/<recipe>/jax through "
                            "opt_core.capture.xla_cache (recipe = det | default from the caller's XLA_FLAGS; kept when JAX_COMPILATION_CACHE_DIR is already set; "
                            "skipped by name without a root): a process loads the executables an earlier process on the same stack and recipe compiled; "
                            "under det XLA's autotune results are neither loaded from nor stored in that root (JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none)", "serving",
                            "none: a cached executable is the compiled one", False,
                            ("module_state", "colabfold_opt.xla_cache", "enabled"), lines="xla_cache.py", doc="CHANGES.md §The levers", deployment=True,
                            impl="opt_core.capture.xla_cache", origin="core", strategy="F3.jit_cache_keyed")
IN_MODE: Tuple[str, ...] = tuple(n for n, lv in LEVERS.items() if lv.in_mode)
DEPLOYMENT: Tuple[str, ...] = tuple(n for n, lv in LEVERS.items() if lv.deployment)
NOT_WIRED: Tuple[str, ...] = tuple(n for n, lv in LEVERS.items() if not lv.in_mode and not lv.deployment)
REFUSED_SWITCHES: Dict[str, str] = {                                                            # a kit switch set in the caller's environment that the activation refuses by name (stack.gate_switches)
    "AF_PALLAS_ATTN_ALL": "MSA-column and template attention through the kernel are outside the measured band: the same kernel measured "
                          "far outside band at key length <= 5 in a short-key probe (mean pLDDT delta 5.1) and template "
                          "attention here has key length = n_templates <= 4 - refused until the kernel refuses short key lengths itself",
}
# No card name and no memory size is assumed anywhere in the package — a GPU visible at all is the one device gate (stack.gate_gpu); its
# compute capability is a note on the line (stack.gpu_note). The markers each lever's patch leaves on the class it rebinds:
PATCH_MARKER = ("class_attr", "alphafold.model.modules", "Attention", "_pallas_patched")      # af2_pallas_attn.py:96
HOST_MARKER = ("class_attr", "alphafold.model.model", "RunModel", "_device_resident")           # device_resident.MARKER
SUBBATCH_MARKER = ("class_attr", "alphafold.model.config", "model_config", "_colabfold_opt_subbatch")   # subbatch.MARKER on the wrapped model_config
TRIMUL_MARKER = ("class_attr", "alphafold.model.modules", "TriangleMultiplication", "_trimul_pallas")     # trimul_pallas.MARKER on the rebound class
MSA_MARKER = ("class_attr", "alphafold.model.modules", "Attention", "_pallas_msa")                        # msa_attn.MARKER on the rebound class
TRIATTN_MARKER = ("class_attr", "alphafold.model.modules", "Attention", "_pallas_word")                   # triattn_xla.MARKER: the binding's site-naming class the bridge row rides (triattn_xla.bind)
COL_MARKER = ("class_attr", "alphafold.model.modules", "Attention", "_msa_col_cudnn")                     # msa_col_cudnn.MARKER on the rebound class
TEMPL_MARKER = ("class_attr", "alphafold.model.modules_multimer", "TemplateEmbedding", "_templ_dedup")     # templ_dedup.MARKER on the rebound class
TRANSITION_MARKER = ("class_attr", "alphafold.model.modules", "Transition", "_transition_served")             # transition.MARKER on the rebound class
TP_MARKER = ("module_state", _modes.TP_LEVER_MODULE, "enabled")                                # big.py _STATE: the axis installed (P>1: the rebound classes are named on its LEVER line, sites=…)
MARKERS = {_modes.LEVER: PATCH_MARKER, _modes.HOST_LEVER: HOST_MARKER, _modes.SUBBATCH_LEVER: SUBBATCH_MARKER, _modes.TRIMUL_LEVER: TRIMUL_MARKER,
           _modes.MSA_LEVER: MSA_MARKER, _modes.TRIATTN_LEVER: TRIATTN_MARKER, _modes.COL_LEVER: COL_MARKER,
           _modes.TEMPL_LEVER: TEMPL_MARKER, _modes.TRANSITION_LEVER: TRANSITION_MARKER, _modes.TP_LEVER: TP_MARKER}   # lever -> the marker its patch leaves

# The kernel's per-call fallback (af2_pallas_attn.py:58-65: an enabled call whose fused path returns None takes the stock math and is
# counted in _STATE["fallbacks"]) has two documented classes on this route; every other fallback is beyond the class. `calls` and
# `fallbacks` count Attention call sites as haiku traces them (the counters sit in the traced __call__), so their ratio is a property of
# the model's module graph, not of the run's length: per model trace 9 Attention call sites are traced (main Evoformer row / column /
# triangle start / end, extra-MSA row / triangle start / end, template triangle start / end) and 2 take the stock path.
# Under fast / big PALLAS_MSA (msa_attn.py) serves both classes upstream of the carried kernel — the census then reads fallbacks=0 on the
# multimer route; the carried kernel's remaining fallback is a bias-free call the provider has no family for (template attention on the
# monomer route: PALLAS_MSA's `no_cell_family`, then the adapter's own stock path under AF_PALLAS_ATTN_ALL=0).
FALLBACK_CLASSES: Tuple[str, ...] = (
    "attention without a pair bias (MSA-column and template attention) under AF_PALLAS_ATTN_ALL=0 (af2_pallas_attn.py:23-24)",
    "extra-MSA-stack MSA row attention with pair bias, 8 channels per head < 16 (extra_msa_channel 64 over 8 heads, config.py:538,483; af2_pallas_attn.py:29)",
)
FALLBACK_MAX_SHARE = 0.30            # fallbacks / (calls + fallbacks) at exit; the documented classes give <= 2/7 = 0.286 per trace
STEP_ASIDE_RULES: Tuple[str, ...] = ("below_keys_rule", "below_size_rule", "cell", "monomer_route", "template_disabled", "dropout_enabled", "no_cell_family", "stock_by_name", "cell_stock", "no_cell")   # cell: the provider's cell named another row for every call of a lever's class (TRIATTN_XLA); no_cell_family: no provider family for a call class (PALLAS_MSA / MSA_COL_CUDNN); stock_by_name: the tier word named the STOCK statement for the call's cell (TRIMUL_PALLAS); cell_stock / no_cell: TRANSITION — the tier word names the stock statement by cell / no cell; dropout_enabled: TEMPL_DEDUP under stock's --use-dropout
                                     # (below_keys_rule / below_size_rule: the shared core serve layer's words, opt_core.kernels.pallas_attn_serve BELOW_KEYS_RULE / BELOW_SIZE_RULE). A lever whose
                                     # EVERY call stepped aside by these rules (calls=0, fallbacks>=1, all so attributed) ran the
                                     # stock operation by design at that size: not a partial activation (manifest.stepped_aside_rule). calls=0 with no attributed fallback stays PARTIAL.


def _probe(p: Tuple, kit_module) -> Optional[bool]:
    kind = p[0]
    if kind == "state":
        st = getattr(kit_module, "_STATE", None) if kit_module is not None else None
        return bool(st.get(p[1])) if isinstance(st, dict) else None
    if kind == "class_attr":
        m = sys.modules.get(p[1])
        if m is None:
            return None
        return bool(getattr(getattr(m, p[2], None), p[3], False))
    if kind == "env":
        return os.environ.get(p[1], "0") not in ("", "0")
    if kind == "module_state":
        m = sys.modules.get(p[1])
        st = getattr(m, "_STATE", None) if m is not None else None
        return bool(st.get(p[2])) if isinstance(st, dict) else None
    raise ValueError(f"unknown probe {p!r}")


def applied(kit_module=None) -> Dict[str, Optional[bool]]:
    """Every lever's probe against the process: True applied, False not, None undecidable (module absent)."""
    if kit_module is None:
        kit_module = sys.modules.get(_modes.KIT_MODULE)
    return {n: _probe(lv.probe, kit_module) for n, lv in LEVERS.items()}


def applied_in_mode(kit_module=None) -> List[str]:
    """The mode levers (IN_MODE) whose probe is true — the report's `levers_applied`."""
    a = applied(kit_module)
    return [n for n in IN_MODE if a.get(n)]


def patch_marker_present() -> Optional[bool]:
    """The kit's own marker on the rebound Attention class (af2_pallas_attn.py:96), when the module is imported."""
    return _probe(PATCH_MARKER, None)


def marker_present(lever: str) -> Optional[bool]:
    """The marker `lever`'s patch leaves on the class it rebinds (MARKERS), when that module is imported; None otherwise."""
    return _probe(MARKERS[lever], None)
