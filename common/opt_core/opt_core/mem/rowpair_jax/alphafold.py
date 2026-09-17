"""The AlphaFold-2 recipe of the row-sharded pair stack (dm-haiku, ``alphafold.model.modules``): ``--n_gpu P``, P > 1 — ONE producer for
two pinned library trees (the multimer-era 2.3.x library with fused ``TriangleMultiplication`` projections and ``modules_multimer``; the earlier monomer
library whose pair classes use ``hk.LayerNorm`` and the unfused projections).

This module names no engine: it receives the STOCK module objects — ``alphafold.model.modules`` and, when the caller runs the multimer
embedder, ``alphafold.model.modules_multimer`` — plus a ``RowMesh`` of :mod:`mesh`, and rebinds the pair sub-layer classes to bodies that run
the STOCK arithmetic on this device's row block ``[N/P, N, C]`` with the collectives of this package inserted (:mod:`shard` / :mod:`trimul` /
:mod:`triatt` / :mod:`transition` are the one producer of every collective, :mod:`haiku` of the Haiku facts; this module owns only
AlphaFold-2's call structure and parameter names).

The plan (rows = dim 0 of the pair; MSA, single and parameters replicated; masks enter whole and are sliced per device):

| class (``modules.py``) | inside a region | collective | numerics class |
|---|---|---|---|
| ``EvoformerIteration.__call__`` (main and extra-MSA stacks) | THE REGION: ``shard_map`` over (msa: replicated, pair: rows, msa mask: replicated, pair mask: replicated, key: replicated) → (msa, pair rows); the stock body runs inside on the local blocks with the pair mask sliced to my rows; the pair carry is pinned to the row layout on entry and exit | ``haiku.region``, ``shard.constrain`` | moves_bytes |
| ``TemplateEmbeddingIteration.__call__`` (``modules_multimer``) / ``TemplatePairStack.__call__`` (monomer template stack) | a region over (act: rows, pair mask: replicated, key: replicated) → act rows | same | moves_bytes |
| ``TriangleMultiplication`` (fused or unfused projections) | layer norm, projections and gates on my rows (row-local); the triangle contraction complete over ``k`` for my output rows | ``trimul.contract`` | complete_contraction |
| ``TriangleAttention`` per_row (starting node) | my rows are the attention batch; the non-batched pair bias needs every row → gathered | ``triatt.bias_full`` | row_local |
| ``TriangleAttention`` per_column (ending node) | my COLUMN block becomes my row block of the transpose (one all_to_all in, one out); the mask columns from the gathered mask | ``triatt.enter_transposed`` / ``exit_transposed`` / ``mask_cols``, ``shard.gather`` | row_local |
| ``OuterProductMean`` | the left operand and the normaliser mask sliced to my residues → exactly my output rows; the right operand whole | ``transition.opm_operands`` | row_local |
| ``MSARowAttentionWithPairBias`` | the pair bias of my rows gathered to all rows; the MSA attention itself replicated | ``transition.pair_logits_full`` | row_local |
| pair ``Transition``, MSA column attention, MSA transition | stock unchanged (row-local / replicated) | none | row_local |

Beyond the trunk regions the plan has two more site groups (``install(sites=("trunk", "model", "heads"))``; each presumes the previous):

| group ``model`` | what stays ROW-SHARDED | mechanism | class |
|---|---|---|---|
| ``EmbeddingsAndEvoformer.__call__`` (both module files) | the recycled pair entering (``prev_pair``), the pair leaving (trunk → heads boundary) | ``shard.constrain`` around the stock body | moves_bytes |
| the pair prologue: ``prev_pos_linear`` (recycled-position distogram ``[N, N, 15]`` → Linear), ``prev_pair_norm`` (LayerNorm of the recycled pair), ``pair_activiations`` / ``position_activations`` (relative-position one-hot → Linear) | operand and output of each pinned module pinned to rows — the producers between them (left/right single broadcast sum, the one-hot / distogram compares) are partitioned by propagation from both sides | ``hk.intercept_methods`` keyed on the pinned module names under scope ``evoformer`` | moves_bytes |
| ``hk.while_loop`` / ``hk.fori_loop`` / ``hk.scan`` of the two module files (the in-graph recycle loop incl. its initial zeros, the ensemble loops) | every ``[N, N, c]`` carry leaf on the initial value, the body result, the loop result | the modules' ``hk`` name proxied | moves_bytes |
| the template embedding (feature stack, per-template sum, pointwise attention) | the template pair never handed back gathered (``TemplatePairStack`` returns rows) | :data:`TEMPLATE_MODULE` ``install_template`` (absent → ``template=replicated``, named) | row_local |
| placement (:func:`apply_shardings` / :func:`wrap_runner`) | the recycled pair ENTERS the jit row-sharded; the returned pair / recycled pair / distogram logits / aligned-error logits, probabilities and expected error LEAVE row-sharded (the host assembles them: the writer boundary) | ``jax.jit(in_shardings=, out_shardings=)`` from :data:`ROW_LEAVES_IN` / :data:`ROW_LEAVES_OUT` | moves_bytes |

| group ``heads`` | | | |
|---|---|---|---|
| ``AlphaFoldIteration.__call__`` (both module files) | ONE manual region after the trunk: the pair enters as the row block; the stock heads loop runs inside; ``heads=sharded``: the structure module's InvariantPointAttention reads my pair rows (bodies below the structure banner); ``conf=sharded``: DistogramHead / PredictedAlignedErrorHead read my rows (bodies: :data:`HEADS_MODULE`) and their logits leave row-sharded; a word reading ``replicated`` gathers the pair ONCE inside the region (named) | ``shard.shard_map`` + :func:`haiku.body_frame` with the structure key | moves_bytes |
| ``confidence.get_confidence_metrics`` (the trees that compute pLDDT / PAE / pTM / ipTM in the program) | a second region over the aligned-error logits rows; ``[N, N, bins]`` probabilities and ``[N, N]`` expected error leave row-sharded, scalars replicated | the modules' ``confidence`` name proxied; row body :data:`HEADS_MODULE` | moves_bytes |

Replicated by design and NAMED on the evidence (:func:`describe`): the MSA representation ``[S, N, c_m]`` (linear in N: ``msa=replicated``), the
single representation and every per-residue tensor, the ``[N, N]`` no-channel pair masks (``masks=replicated``), the structure module's affine /
point path (per residue). Not bit-exact against P = 1 (complete contractions per element, no partial sums across devices; XLA fuses the local blocks
differently): a P>1 run is tested as a BAND against the kit's P=1 run. P = 1 installs nothing: the caller gates it and never calls
:func:`install`. Under the model group an N that does not divide by P is refused by name at the model boundary (the kit pads N to a multiple of P);
the trunk group alone pads each region call.

PIN. :func:`install` hashes ``modules.__file__`` (and ``modules_multimer.__file__``) and REFUSES BY NAME unless the sha256 is one of :data:`PIN`
(the two pinned stock trees); a kit on another tree passes ``pin=`` only together with its own review. HAZARDS named (:data:`HAZARDS`): the
column attention slices its mask from the gathered whole mask (no symmetry assumed); region bodies run under :func:`haiku.body_frame` (rng-less at
apply — the stock blocks receive their ``safe_key`` explicitly, drawn OUTSIDE the region at the stock's position); dropout is the stock's
(``is_training`` / ``deterministic``) on local blocks — at inference it is off.

The unfused ``TriangleMultiplication`` branch transcribes ``modules.py`` ``_triangle_multiplication`` (2.3.x) = the earlier library's
``TriangleMultiplication.__call__`` body (``layer_norm_input``, ``left_/right_projection``, ``left_/right_gate``, ``center_layer_norm``,
``output_projection``, ``gating_linear``); :data:`BODIES` cites the stock statements of every rebound body. Standard library at import; jax / haiku
inside the bodies.
"""
from __future__ import annotations

import contextlib as _contextlib
import hashlib
import threading as _threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .. import MemLeverRefused
from . import LEVER as FAMILY_LEVER
from . import haiku as _hk
from . import shard as _shard
from . import transition as _transition
from . import triatt as _triatt
from . import trimul as _trimul

TREE_CF = "alphafold-colabfold 2.3.13"                                     # the multimer-era 2.3.x library (PyPI alphafold-colabfold 2.3.13)
TREE_DL = "dl_binder_design cafa3853"                                        # the monomer library of dl_binder_design's af2_initial_guess (+ its overlay where it changes bytes)
PIN_FILES: Dict[str, Dict[str, str]] = {                                 # per stock file kind: sha256 → label — the ONE table of the AlphaFold-2 family (siblings derive their views)
    "modules": {
        "fec9f1210f865dbdd049c50359baed97146a97b1306dc37d335b22f79c261ea0": TREE_CF + " alphafold/model/modules.py",
        "f4a8229bd130d1376d5e61540e47217012ed60a1460cb8f19d09f28dc38f94c4": TREE_DL + " alphafold/model/modules.py",
        "5d554d86835d486b06fab8a01fbeb509de1037652425ac6d2e45778af632360d": TREE_DL + " alphafold/model/modules.py + a kit patch inside Attention.__call__ only (no rebound class touched)",
    },
    "modules_multimer": {"d07d0a4e00bbaa1f4c2d080507d6908ef6d5a96f4cfa6edfe9c673e1b75d4345": TREE_CF + " alphafold/model/modules_multimer.py"},
    "folding": {
        "85295f2ca8fa9b7ff7a369e645ee4e948e33132f246d509cf002dbf9ff75c1ac": TREE_CF + " alphafold/model/folding.py",
        "63e943be2e57b5da47fe0ee13a41fbd21f497fa2bdc9e7b67941b14df648a06e": TREE_DL + " alphafold/model/folding.py",
    },
    "folding_multimer": {"6eb571fba2812c54ae65c465bc1c1f8bdb92131e2a635dabd7834cefe6f028d5": TREE_CF + " alphafold/model/folding_multimer.py"},
    "confidence": {"36b4b6df974d2186b24b2ae82e53395ce4a7c98cf46991cf8c8a0287c2ba75d5": TREE_CF + " alphafold/common/confidence.py (run IN the program: jax.numpy under use_jnp=True)"},
}
PIN: Dict[str, str] = {sha: label for group in PIN_FILES.values() for sha, label in group.items()}   # every pinned file → its label (install refuses a sha that is not a key)
PIN_TREE: Dict[str, str] = {sha: (TREE_CF if label.startswith(TREE_CF) else TREE_DL) for sha, label in PIN.items()}   # sha → which library it is
HAZARDS = ("col_attention_mask_from_gathered_mask(no_symmetry_assumed)", "rngless_body_frame(apply;safe_key_explicit)", "dropout=stock(is_training)",
           "module_state_trace_flags(thread_local)", "prologue_interceptor(module_names_pinned)", "hk_name_proxy(while_loop,fori_loop,scan;carry_by_name)",
           "structure_key_drawn_outside_region(stock_position)", "multimer_structure_outside_region_when_replicated")
SOURCE = "colabfold_opt/rowpair_af2.py@58e6ca7e6a83"
CF, DL, MM = "alphafold-colabfold 2.3.13 modules.py", "dl_binder_design cafa3853 modules.py", "alphafold-colabfold 2.3.13 modules_multimer.py"
BODIES = {"EvoformerIteration.__call__": ("moves_bytes", SOURCE), "TemplateEmbeddingIteration.__call__": ("moves_bytes", SOURCE),
          "TemplatePairStack.__call__": ("moves_bytes", "this module (same region form as the template iteration)"),
          "TriangleMultiplication.__call__": ("complete_contraction", SOURCE + " (fused); modules.py _triangle_multiplication (unfused)"),
          "TriangleAttention.__call__": ("row_local", SOURCE), "OuterProductMean.__call__": ("row_local", SOURCE),
          "MSARowAttentionWithPairBias.__call__": ("row_local", SOURCE),
          # ---- model group
          "EmbeddingsAndEvoformer.__call__": ("moves_bytes", f"stock body; prev_pair in / pair out constrained ({CF}:1871-2095, {DL}:1701-1928, {MM}:628-775)"),
          "prologue(prev_pos_linear,prev_pair_norm,pair_activiations,position_activations)": ("moves_bytes", f"{CF}:1911,1925,1940; {DL}:1739,1758,1775; {MM}:561,620,635 (operand and output constrained to rows)"),
          "hk.while_loop/hk.scan(carry prev_pair,pair)": ("moves_bytes", f"{DL}:388-391 (recycle), {CF}:298-301 / {DL}:191-195 (ensemble), {MM}:334-335 (ensemble scan)"),
          # ---- heads group
"AlphaFoldIteration.__call__(heads region)": ("moves_bytes", f"{CF}:236-368 = {DL}:143-268; {MM}:295-403 — the heads loop verbatim inside ONE region over (pair rows; msa/single/batch replicated); a consumer whose word is replicated reads the pair gathered once inside"),
          "confidence.get_confidence_metrics(region)": ("moves_bytes", f"{CF}:466-476, {MM}:467-478 call alphafold/common/confidence.py:172-213 in the program: a region over the aligned-error logits rows; the row body is HEADS_MODULE's")}

HEADS_MODULE = "opt_core.mem.rowpair_jax.alphafold_heads"                    # the distogram / aligned-error / confidence ROW BODIES (install_heads); this module owns the regions
TEMPLATE_MODULE = "opt_core.mem.rowpair_jax.alphafold_template"              # the template embedding ROW BODIES (install_template): feature stack, per-template sum, pointwise attention
PAIR_HEADS = ("distogram", "predicted_aligned_error")                        # heads that read representations['pair'] under conf=; 'structure_module' under heads=
TEMPLATE_ITERATION = "TemplateEmbeddingIteration"                                                            # alphafold.model.modules_multimer 2.3.x
TEMPLATE_PAIR_STACK = "TemplatePairStack"                                                                     # the monomer template pair stack (both trees)
SITE_GROUPS = ("trunk", "model", "heads")                                    # install(sites=…): each group presumes the previous
TRIMUL_DEFAULT = _trimul.DEFAULT                                             # the P>1 triangle-multiplication schedule = the primitive's schedule ("ring"; "gather" is the named alternative)
HEAD_WORDS = ("sharded", "replicated")                                       # install(heads=…, conf=…)
EVOFORMER_SCOPE = "evoformer"                                                # EmbeddingsAndEvoformer(name='evoformer'): cf modules.py:1866, dl :1696, mm :486
PROLOGUE_PAIR_MODULES = ("prev_pos_linear", "prev_pair_norm", "pair_activiations", "position_activations")   # the prologue's [N, N, c] producers by module name (BODIES 'prologue')
PROLOGUE_SINGLE_MODULES = ("left_single", "right_single")                   # the pair's creating statement pair = left_single(x)[:, None] + right_single(x)[None]: cf :1893-1898, dl :1722-1728, mm :651-657
PAIR_CARRY_KEYS = ("prev_pair", "pair")                                      # the loop carries' pair leaves: get_prev (cf :406-412, dl :302-309, mm :412-418) / the ensemble representations
KNOWN_CARRY_KEYS = ("prev_pos", "prev_msa_first_row", "msa", "msa_first_row", "single", "structure_module")   # the carries' per-residue / MSA leaves (left in place whatever their shape — S == N included)
ROW_LEAVES_IN = (("prev", "prev_pair"), ("prev_pair",))                      # apply's batch leaves that ENTER row-sharded (path suffixes)
ROW_LEAVES_OUT = (("pair",), ("prev", "prev_pair"), ("distogram", "logits"), ("predicted_aligned_error", "logits"),
                  ("predicted_aligned_error",), ("aligned_confidence_probs",), ("pae_matrix_with_logits", "logits"))   # output leaves that LEAVE row-sharded when [N, N, …]-shaped

_PLAN_EMPTY: Dict[str, Any] = {"rmesh": None, "axis": None, "n_gpu": 1, "modules": None, "multimer": None, "patches": None, "stock": {},
                               "schedule": "gather", "lever": FAMILY_LEVER, "regions_built": 0, "padded_calls": 0, "shas": {},
                               "sites": (), "model": False, "heads_on": False, "heads": "not_installed", "conf": "not_installed", "structure": "not_installed",
                               "structure_reason": "none", "conf_reason": "none", "template": "replicated", "template_reason": "none", "library": {}, "outputs_rows": [],
                               "inputs_rows": [], "runner_wrapped": "none", "heads_sites": [], "template_sites": [], "structure_sites": []}
_TREE: Dict[str, Any] = {"pae_asym_id": False}                             # facts of the pinned AlphaFoldIteration source (:func:`_tree_facts`)
_PLAN: Dict[str, Any] = dict(_PLAN_EMPTY)


def installed() -> bool:
    return _PLAN["patches"] is not None


# ----------------------------------------------------------------------------------------------------------------- trace state
class _TraceState(_threading.local):
    """The recipe's trace-time flags (HAZARD ``module_state_trace_flags``) read by the rebound bodies while ONE thread traces the model: ``heads`` — the depth of
    heads regions entered (:func:`_manual_body`; the trunk / template regions are :func:`haiku.region`, whose depth :func:`haiku.in_region` reports for every
    region of this package), and the heads switches ``heads_rows`` (the structure module holds the LOCAL pair rows — the InvariantPointAttention bodies below the
    structure banner read it) / ``heads_rows_conf`` (the distogram / aligned-error heads and the in-graph confidence hold local rows — :data:`HEADS_MODULE` reads
    it). Thread-local: jax traces a jitted function on the calling thread; a second thread sees its own state."""
    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.heads = 0
        self.heads_rows, self.heads_rows_conf = False, False


_S = _TraceState()


def in_manual_region() -> bool:
    """True inside ANY shard_map region of this package (trunk, template, heads: :func:`haiku.in_region` — every region body runs under :func:`haiku.body_frame`):
    code here sees local blocks, places no ``with_sharding_constraint`` and builds no nested region; a rebound entry point reached here runs its stock body on
    the blocks it is given."""
    return _hk.in_region()


def in_sharded_region() -> bool:
    """True inside a TRUNK region (Evoformer block / template pair stack / a region built by :func:`_region`) and not inside the heads region: the pair
    sub-layer bodies run on this device's row block."""
    return _hk.in_region() and int(_S.heads) == 0


def heads_rows() -> bool:
    """True while the structure module holds the LOCAL pair rows ``[N/P, N, c_z]`` (heads=sharded, inside the heads region)."""
    return bool(_S.heads_rows)


def heads_rows_conf() -> bool:
    """True while the distogram / predicted-aligned-error heads or the in-graph confidence hold the local pair / logits rows (conf=sharded, inside their regions)."""
    return bool(_S.heads_rows_conf)


@_contextlib.contextmanager
def _manual_body(rng: Any, lever: str):
    """The heads regions' body: a Haiku frame carrying ``rng`` (None: rng-less) with the heads depth raised."""
    _S.heads += 1
    try:
        with _hk.body_frame(rng, lever):
            yield
    finally:
        _S.heads -= 1


def _region(body, rmesh, in_specs, out_specs):
    """:func:`haiku.region` of ``body`` on the plan's lever (one more region closure built): the trunk regions and, through ``install_template(region=…)``, the
    template regions — inside, :func:`in_sharded_region` / :func:`in_manual_region` are true."""
    _PLAN["regions_built"] += 1
    return _hk.region(body, rmesh, in_specs=in_specs, out_specs=out_specs, lever=_PLAN["lever"])


def _site_word(name: str) -> str:
    """``Owner.attr`` from a :class:`PatchSet` name (``package.module.Class.method`` → ``Class.method``; ``package.module.attr`` → ``module.attr``)."""
    head, attr = name.rsplit(".", 1) if "." in name else (name, "")
    return f"{head.rsplit('.', 1)[-1]}.{attr}" if attr else head


def describe() -> Dict[str, Any]:
    """The installed plan's evidence in the family line's ON vocabulary (:data:`evidence.ON_REQUIRED`), ready for
    ``evidence.line(tag, "on", P, rmesh=…, peaks=…, **describe())``: ``sites`` (the rebound ``Class.method`` names), ``schedule`` (the trimul collective
    schedule), ``kernel=jnp`` + ``kernel_reason=xla_path_under_rowpair`` (the sub-layers run the library's XLA ops on the row block; a kit that runs a fused
    per-shard kernel overrides both words), ``regions_built`` (region closures constructed — one per traced call site), ``padded_calls`` (trunk region calls
    whose N was not a multiple of P and were zero-padded: with the trunk group ALONE each region pads its call and un-pads its result; under the model / heads
    groups the whole program holds the pair in rows, so an N that does not divide by P is refused by name at the model boundary and the kit pads the input —
    ``padded_calls`` then stays 0), and the EFFECTIVE states of the model / heads groups: ``heads`` (sharded: a pair consumer of the
    heads holds local rows; replicated: the heads gather the pair; not_installed), ``conf`` (distogram / aligned-error / in-graph confidence), ``structure``
    (the structure module's InvariantPointAttention: sharded = row-local attention over my query rows; replicated = one all_gather of the pair inside its
    region) + ``structure_reason``, ``recycle`` (rows: the recycled pair crosses the loop / jit boundary row-sharded), ``template`` (rows: the template pair
    never comes back replicated) + ``template_reason``, ``msa=replicated`` (the stated floor: ``[S, N, c_m]``, linear in N), ``masks=replicated`` (the
    ``[N, N]`` no-channel pair masks ``mask_2d`` / ``template_mask_2d`` enter the regions whole and are sliced per device), ``outputs_rows`` (the output leaves the last placement
    assigned the row layout — they leave the program row-sharded; the host assembles them), ``library`` (module shas), ``hazards``."""
    p = _PLAN["patches"]
    model = bool(_PLAN["model"])
    lib = _PLAN.get("library") or {}
    heads_words: Dict[str, Any] = {"conf_sites": ",".join(_PLAN.get("heads_sites") or []) or "none",
                                   "confidence": ("in_graph" if lib.get("confidence") is not None else "host") if _PLAN["heads_on"] else "not_installed",
                                   "structure_sites": ",".join(_PLAN.get("structure_sites") or []) or "none", "template_sites": ",".join(_PLAN.get("template_sites") or []) or "none"}
    if _PLAN["conf"] == "sharded":                                          # the heads module's own words (conf_sites, confidence, its output leaves) when it is installed
        mod = _optional_module(HEADS_MODULE)
        described = getattr(mod, "describe_heads", None)
        if described is not None:
            for k, v in dict(described() or {}).items():
                if k == "outputs_rows":
                    continue                                                # outputs_rows below is the placement's record (apply_shardings): what LEFT the jit row-sharded
                heads_words[k] = ",".join(v) if isinstance(v, (list, tuple)) else v
    return {"sites": ",".join(sorted(set(_site_word(n) for n in p.names()))) if p else "none",
            "schedule": _PLAN["schedule"], "kernel": "jnp", "kernel_reason": "xla_path_under_rowpair",
            "regions_built": _PLAN["regions_built"], "padded_calls": _PLAN.get("padded_calls", 0), "msa": "replicated",
            "heads": _PLAN["heads"], "conf": _PLAN["conf"], "conf_reason": _PLAN.get("conf_reason", "none"), "structure": _PLAN["structure"],
            "structure_reason": _PLAN["structure_reason"],
            "recycle": "rows" if model else "replicated", "template": _PLAN.get("template", "replicated"), "template_reason": _PLAN.get("template_reason", "none"),
            "masks": "replicated",
            "outputs_rows": ",".join(_PLAN["outputs_rows"]) or "none", "inputs_rows": ",".join(_PLAN["inputs_rows"]) or "none",
            "library": ",".join(f"{k.rsplit('.', 1)[-1]}:{v[:8]}" for k, v in sorted(_PLAN["shas"].items())) or "none", "hazards": ",".join(HAZARDS), **heads_words}


LIBRARY = ("alphafold.model.modules", "alphafold.model.modules_multimer")


def library(lever: str = FAMILY_LEVER, multimer: bool = True):
    """``(modules, modules_multimer | None)`` of the AlphaFold-2 model library in this process — the kit's import through ONE door that refuses
    BY NAME (``MemLeverRefused(lever, 'alphafold.model.modules not importable (…)')``) instead of an ImportError; ``multimer=False`` skips the second."""
    import importlib  # noqa: PLC0415
    out = []
    for name in LIBRARY[: 2 if multimer else 1]:
        try:
            out.append(importlib.import_module(name))
        except ImportError as e:
            raise MemLeverRefused(lever, f"{name} not importable ({e})") from None
    return (out[0], out[1] if multimer else None)


def file_sha(module: Any, lever: str = FAMILY_LEVER) -> str:
    """sha256 of a module's source file (the pin unit)."""
    path = getattr(module, "__file__", None)
    if not path:
        raise MemLeverRefused(lever, f"{getattr(module, '__name__', module)!s} has no __file__ to pin")
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def check_pin(modules: Any, multimer: Any = None, pin: Optional[Dict[str, str]] = None, lever: str = FAMILY_LEVER, extra: Sequence[Any] = ()) -> Dict[str, str]:
    """Refuse BY NAME unless ``modules`` (and ``multimer``, and every module of ``extra``) hash to a key of ``pin`` (default :data:`PIN`). Returns ``{name: sha}``."""
    want = PIN if pin is None else pin
    out = {}
    for m in (modules, multimer, *extra):
        if m is None:
            continue
        sha = file_sha(m, lever)
        if sha not in want:
            raise MemLeverRefused(lever, f"install: {getattr(m, '__name__', m)!s} sha256 {sha[:12]} is not a transcribed tree ({'; '.join(sorted(set(want.values())))})")
        out[str(getattr(m, "__name__", m))] = sha
    return out


def model_library(modules: Any, multimer: Any = None, lever: str = FAMILY_LEVER) -> Dict[str, Any]:
    """The modules the model / heads groups rebind besides ``modules``: ``folding`` (``modules.folding`` — the structure module of the SAME package),
    ``folding_multimer`` (``multimer.folding_multimer`` when ``multimer`` is given), ``confidence`` (``modules.confidence`` when the tree computes the
    confidence metrics IN the program — a tree without it computes them on the host from the returned logits: absent, not refused)."""
    lib: Dict[str, Any] = {"folding": getattr(modules, "folding", None), "folding_multimer": None, "confidence": getattr(modules, "confidence", None),
                           "confidence_multimer": None}
    if multimer is not None:
        lib["folding_multimer"] = getattr(multimer, "folding_multimer", None)
        lib["confidence_multimer"] = getattr(multimer, "confidence", None)
    return lib


def install(rmesh: Any, modules: Any, multimer: Any = None, *, heads: str = "sharded", conf: str = "sharded", sites: Sequence[str] = SITE_GROUPS,
            schedule: str = TRIMUL_DEFAULT, lever: Optional[str] = None, pin: Optional[Dict[str, str]] = None, patches: Any = None, _test_single_device_mesh: bool = False):
    """Rebind the plan on ``rmesh`` (``rmesh.n_gpu`` > 1) into a :class:`PatchSet` (returned; ``uninstall()`` / ``patches.restore()`` restore every stock
    attribute; ``patches``: install INTO the adapter's own set). ``sites`` names the groups: ``trunk`` = the pair sub-layer classes of ``modules`` (and
    ``multimer.TemplateEmbeddingIteration``; ``modules.TemplatePairStack``) inside row-sharded regions; ``model`` = the pair prologue, the recycled pair and
    the template pair kept in the row layout (``EmbeddingsAndEvoformer.__call__``, the prologue producers, ``hk.while_loop`` / ``hk.scan`` carries,
    ``TemplateEmbedding.__call__``) and the placement words of :func:`apply_shardings`; ``heads`` = the pair consumers of the heads in regions on local rows
    (``DistogramHead``, ``PredictedAlignedErrorHead``, the in-graph ``confidence`` metrics, ``folding.StructureModule``). ``heads=`` / ``conf=`` choose
    sharded | replicated for the structure module / the confidence heads (replicated: the region gathers the pair once — NAMED on :func:`describe`).
    Refused by name: an unpinned tree (:func:`check_pin` over every rebound module; ``pin`` overrides for a reviewed tree)."""
    lv = str(lever or FAMILY_LEVER)
    n = int(getattr(rmesh, "n_gpu", 0) or 0)
    sites = tuple(str(s) for s in (sites or ()))
    want_model, want_heads = "model" in sites, "heads" in sites
    lib: Dict[str, Any] = {}
    extra: List[Any] = []
    if want_model:
        lib = model_library(modules, multimer, lv)
        extra = [lib["folding"], lib["folding_multimer"], lib["confidence"]]
    shas = check_pin(modules, multimer, pin, lv, extra=extra)
    patches = patches if patches is not None else _hk.PatchSet(lv)
    stock: Dict[str, Any] = {}
    structure, structure_reason, conf_state, conf_reason = "not_installed", "none", "not_installed", "none"
    template_state, template_reason = "replicated", ("trunk_only" if not want_model else "none")
    heads_sites: List[str] = []
    template_sites: List[str] = []
    structure_sites: List[str] = []
    try:
        stock["EvoformerIteration"] = _hk.rebind(patches, modules.EvoformerIteration, "__call__", _evoformer_iteration_call, lv)
        stock["TriangleMultiplication"] = _hk.rebind(patches, modules.TriangleMultiplication, "__call__", _triangle_multiplication_call, lv)
        stock["TriangleAttention"] = _hk.rebind(patches, modules.TriangleAttention, "__call__", _triangle_attention_call, lv)
        stock["OuterProductMean"] = _hk.rebind(patches, modules.OuterProductMean, "__call__", _outer_product_mean_call, lv)
        stock["MSARowAttentionWithPairBias"] = _hk.rebind(patches, modules.MSARowAttentionWithPairBias, "__call__", _msa_row_attention_call, lv)
        if multimer is not None:
            stock[TEMPLATE_ITERATION] = _hk.rebind(patches, getattr(multimer, TEMPLATE_ITERATION), "__call__", _template_iteration_call, lv)
        if hasattr(modules, TEMPLATE_PAIR_STACK):
            stock[TEMPLATE_PAIR_STACK] = _hk.rebind(patches, getattr(modules, TEMPLATE_PAIR_STACK), "__call__", _template_pair_stack_call, lv)
        if want_model:                                                      # ---- the model group: prologue, recycle carry, template pair, trunk→heads boundary in rows
            stock["EmbeddingsAndEvoformer"] = _hk.rebind(patches, modules.EmbeddingsAndEvoformer, "__call__", _make_embeddings_call("EmbeddingsAndEvoformer"), lv)
            patches.replace(modules, "hk", _HKProxy(modules.hk))
            if multimer is not None:
                stock["multimer.EmbeddingsAndEvoformer"] = _hk.rebind(patches, multimer.EmbeddingsAndEvoformer, "__call__", _make_embeddings_call("multimer.EmbeddingsAndEvoformer"), lv)
                patches.replace(multimer, "hk", _HKProxy(multimer.hk))
            templ_mod = _optional_module(TEMPLATE_MODULE)                  # the template embedding on rows (feature stack, per-template sum, pointwise attention)
            if templ_mod is not None:
                hook = {"_test_single_device_mesh": True} if _test_single_device_mesh else {}   # the unit tests' 1-device fidelity install reaches the sibling with the same hook
                template_sites = list(templ_mod.install_template(patches, modules, multimer, lever=lv, rmesh=rmesh, axis=str(rmesh.axis), n_gpu=n, active=_model_here,
                                                                 in_region=in_manual_region, region=_region, **hook) or [])
                template_state = "rows"
            else:
                template_state, template_reason = "replicated", "alphafold_template_absent"   # NAMED on describe(): the template pair stack hands its rows back gathered
        if want_heads:                                                      # ---- the heads group: ONE region after the trunk; the pair consumers of the heads on local rows
            _TREE.update(_tree_facts(modules, multimer))
            stock["AlphaFoldIteration"] = _hk.rebind(patches, modules.AlphaFoldIteration, "__call__", _alphafold_iteration_call, lv)
            if multimer is not None:
                stock["multimer.AlphaFoldIteration"] = _hk.rebind(patches, multimer.AlphaFoldIteration, "__call__", _alphafold_iteration_multimer_call, lv)
            if lib.get("confidence") is not None:                           # the tree computes the confidence metrics in the program: a second region on the logits rows
                patches.replace(modules, "confidence", _ConfidenceProxy(lib["confidence"]))
            if lib.get("confidence_multimer") is not None:
                patches.replace(multimer, "confidence", _ConfidenceProxy(lib["confidence_multimer"]))
            if conf == "sharded":
                heads_mod = _optional_module(HEADS_MODULE)
                if heads_mod is not None:
                    heads_sites = list(heads_mod.install_heads(patches, modules, lever=lv, rows_active=heads_rows_conf, axis=str(rmesh.axis), n_gpu=n) or [])
                    conf_state = "sharded"
                else:
                    conf_state, conf_reason = "replicated", "alphafold_heads_absent"   # NAMED on describe(): the heads read the pair gathered once inside the region
            else:
                conf_state, conf_reason = "replicated", "requested"
            if heads == "sharded":
                installer = globals().get("_install_structure")            # the InvariantPointAttention row bodies (below the structure banner)
                if installer is not None:
                    structure_sites = list(installer(patches, {"folding": lib["folding"], "folding_multimer": lib.get("folding_multimer")}, lv) or [])
                    structure = "sharded"
                else:
                    structure, structure_reason = "replicated", "ipa_row_bodies_absent"   # NAMED on describe(): the structure module reads the pair gathered once inside the region
            else:
                structure, structure_reason = "replicated", "requested"
    except Exception:
        patches.restore()
        for mod_name in (TEMPLATE_MODULE, HEADS_MODULE):
            mod = _optional_module(mod_name)
            forget = getattr(mod, "forget", None) if mod is not None else None
            if callable(forget):
                forget()
        raise
    if want_heads and structure == "replicated" and multimer is not None:
        structure_reason = f"{structure_reason}+multimer_stock_outside_region"   # the multimer structure module then runs before the heads region (see _run_heads_region)
    heads_eff = ("sharded" if "sharded" in (structure, conf_state) else "replicated") if want_heads else "not_installed"
    _S.reset()
    _PLAN.update(rmesh=rmesh, axis=str(rmesh.axis), n_gpu=n, modules=modules, multimer=multimer, patches=patches, stock=stock,
                 schedule=str(schedule), lever=lv, regions_built=0, shas=shas, padded_calls=0, sites=sites, model=want_model, heads_on=want_heads,
                 heads=heads_eff, conf=conf_state, conf_reason=conf_reason, structure=structure, structure_reason=structure_reason,
                 template=template_state, template_reason=template_reason, library=lib, outputs_rows=[], inputs_rows=[], runner_wrapped="none",
                 heads_sites=heads_sites, template_sites=template_sites, structure_sites=structure_sites)
    return patches


def uninstall() -> List[str]:
    """Restore every stock attribute; returns the restored ``Owner.name`` list (empty when nothing was installed). The sibling modules' own installed-state
    (:data:`TEMPLATE_MODULE` / :data:`HEADS_MODULE` ``forget()``, when they define it) is cleared with it, so install → uninstall → install cycles work."""
    p = _PLAN["patches"]
    names = p.restore() if p is not None else []
    for mod_name in (TEMPLATE_MODULE, HEADS_MODULE):
        mod = _optional_module(mod_name)
        forget = getattr(mod, "forget", None) if mod is not None else None
        if callable(forget):
            forget()
    _PLAN.clear()
    _PLAN.update(dict(_PLAN_EMPTY))
    _PLAN["stock"], _PLAN["library"], _PLAN["outputs_rows"], _PLAN["inputs_rows"] = {}, {}, [], []
    _S.reset()
    return names


def pad_multiple() -> int:
    """The row count must divide by this (``shard.local_extent`` refuses otherwise): P when installed, else 1."""
    return int(_PLAN["n_gpu"]) if installed() else 1


# ----------------------------------------------------------------------------------------------------------------- placement
def _path_keys(path) -> Tuple[str, ...]:
    """The dict-key words of a jax key path (sequence / attribute entries contribute nothing)."""
    out = []
    for entry in path:
        key = getattr(entry, "key", None)
        if key is not None and not hasattr(entry, "idx"):
            out.append(str(key))
    return tuple(out)


def _row_leaf(keys: Tuple[str, ...], shape: Sequence[int], table: Sequence[Tuple[str, ...]], n_gpu: int, lever: str) -> bool:
    """True when a leaf whose dict-key path ends with an entry of ``table`` is pair-shaped ``[N, N, …]``; such a leaf whose N does not divide by P is
    refused by name (the kit pads N to a multiple of P)."""
    if not any(len(keys) >= len(t) and tuple(keys[len(keys) - len(t):]) == tuple(t) for t in table):
        return False
    if len(shape) < 2 or int(shape[0]) != int(shape[1]):
        return False
    try:
        _shard.local_extent(int(shape[0]), int(n_gpu), lever)
    except MemLeverRefused as e:
        raise MemLeverRefused(lever, f"{e.reason}; a pair-shaped leaf {'.'.join(keys)} of the apply enters / leaves row-sharded: pad the input to a multiple of n_gpu={int(n_gpu)} "
                                     f"(the kit's padding path or length-bucket ladder)") from None
    return True


def apply_shardings(rmesh: Any, in_tree_shapes: Any, out_tree_shapes: Any, lever: Optional[str] = None) -> Tuple[Any, Any]:
    """``(in_shardings, out_shardings)`` for ``jax.jit`` of the model's ``apply`` on the mesh — pytrees matching ``in_tree_shapes`` / ``out_tree_shapes``
    (trees of objects with ``.shape``: ``jax.eval_shape`` of the apply gives the output tree; ``jax.ShapeDtypeStruct`` / the arrays themselves the
    inputs) whose leaves are the ROW layout ``P(axis, None, …)`` for the pair-shaped leaves named in :data:`ROW_LEAVES_IN` / :data:`ROW_LEAVES_OUT`
    (the recycled pair entering, the returned pair / recycled pair / distogram logits / aligned-error logits, probabilities and expected error leaving)
    and REPLICATED for every other leaf. A kit that owns its jit passes these to ``jax.jit(apply, in_shardings=…, out_shardings=…)``; :func:`wrap_runner`
    does it for a runner. The chosen leaves are recorded (``describe()['inputs_rows'|'outputs_rows']``). A named leaf whose N does not divide by P is
    refused by name. Without the model group installed every leaf is replicated (the trunk regions shard the pair inside the program only)."""
    from ._lazy import jax as _lazy_jax  # noqa: PLC0415
    lv = str(lever or _PLAN["lever"])
    jax = _lazy_jax(lv)
    axis = str(getattr(rmesh, "axis", None) or _PLAN["axis"])
    n_gpu = int(getattr(rmesh, "n_gpu", 0) or _PLAN["n_gpu"])
    rep = _shard.named(rmesh, _shard.replicated_spec())
    model = installed() and bool(_PLAN["model"])
    chosen: Dict[str, List[str]] = {"in": [], "out": []}

    def build(tree, table, which):
        def leaf(path, x):
            shape = getattr(x, "shape", None)
            if not model or shape is None:
                return rep
            keys = _path_keys(path)
            if _row_leaf(keys, tuple(shape), table, n_gpu, lv):
                chosen[which].append(".".join(keys[-2:]) if len(keys) >= 2 else ".".join(keys))
                return _shard.named(rmesh, _shard.rows_spec(axis, len(shape), 0))
            return rep
        return jax.tree_util.tree_map_with_path(leaf, tree)

    ins = build(in_tree_shapes, ROW_LEAVES_IN, "in")
    outs = build(out_tree_shapes, ROW_LEAVES_OUT, "out")
    if installed():
        _PLAN["inputs_rows"] = sorted(set(chosen["in"]))
        _PLAN["outputs_rows"] = sorted(set(chosen["out"]))
    return ins, outs


class _RowJit:
    """The callable :func:`jit_sharded` returns: ``fn`` jitted on the mesh with the shardings of :func:`apply_shardings`, built and CACHED per
    (argument treedef, leaf shapes / dtypes) on first call (the output tree comes from ``jax.eval_shape``). Positional arguments beyond ``n_args`` and
    keyword arguments enter replicated."""
    def __init__(self, fn: Callable, rmesh: Any, n_args: int, lever: str, jit_kwargs: Dict[str, Any]):
        self._fn, self._rmesh, self._n_args, self._lever, self._jit_kwargs = fn, rmesh, int(n_args), lever, dict(jit_kwargs)
        self._cache: Dict[Any, Callable] = {}
        self.__wrapped__ = fn

    def _key(self, jax, args, kwargs):
        def sig(x):
            return (tuple(getattr(x, "shape", ()) or ()), str(getattr(x, "dtype", type(x).__name__)))
        return (jax.tree_util.tree_structure((args, kwargs)), tuple(sig(x) for x in jax.tree_util.tree_leaves((args, kwargs))))

    def __call__(self, *args, **kwargs):
        from ._lazy import jax as _lazy_jax  # noqa: PLC0415
        jax = _lazy_jax(self._lever)
        key = self._key(jax, args, kwargs)
        jitted = self._cache.get(key)
        if jitted is None:
            rep = _shard.named(self._rmesh, _shard.replicated_spec())
            out_shapes = jax.eval_shape(self._fn, *args, **kwargs)
            head = tuple(args[: self._n_args])
            ins_head, outs = apply_shardings(self._rmesh, head, out_shapes, self._lever)
            ins = tuple(ins_head) + tuple(jax.tree_util.tree_map(lambda _: rep, a) for a in args[self._n_args:])
            jitted = (jax.jit(self._fn, in_shardings=ins, out_shardings=outs, **self._jit_kwargs), ins)   # pending core hunk: haiku.jit_apply(in_shardings=, out_shardings=)
            self._cache[key] = jitted
        fn, ins = jitted
        args = tuple(jax.device_put(a, s) for a, s in zip(args, ins))          # host arrays upload straight into their layout (the recycled pair never lands whole on one device); a committed array in another layout is resharded, not refused
        return fn(*args, **kwargs)


def jit_sharded(fn: Callable, rmesh: Any, n_args: int = 3, lever: Optional[str] = None, **jit_kwargs) -> Callable:
    """``fn`` (the transformed model's ``apply``: ``(params, rng, batch[, …])``) as a callable that, per new (input treedef, leaf shapes/dtypes), runs
    ``jax.eval_shape`` → :func:`apply_shardings` → ``jax.jit(fn, in_shardings=…, out_shardings=…)`` and caches the compiled callable: the ONE producer of the
    leaf selection + jit for a kit that owns its jit (call it instead of :func:`haiku.jit_apply` when a mesh is set) and for :func:`wrap_runner`. The first
    ``n_args`` positional arguments get the computed shardings; further positional and keyword arguments enter replicated."""
    lv = str(lever or _PLAN["lever"])
    return _RowJit(fn, rmesh, int(n_args), lv, jit_kwargs)


def wrap_runner(runner: Any, rmesh: Any, attr: str = "apply", n_args: int = 3, lever: Optional[str] = None, **jit_kwargs) -> Any:
    """Bind the model runner's jitted apply to the mesh: ``runner.<attr>`` (``alphafold.model.model.RunModel.apply`` = ``jax.jit(hk.transform(f).apply)``,
    signature ``(params, rng, batch[, …])``) becomes :func:`jit_sharded` of it — input / output shardings from :func:`apply_shardings`: the recycled pair enters and
    the pair-shaped outputs leave ROW-SHARDED under the model group (the host's ``np.asarray`` assembles them: the writer boundary), everything else
    replicated; the pair is sharded inside by the regions and the constraints. The caller places ``params`` / the batch with :func:`shard.put` (replicated).
    Returns the runner."""
    lv = str(lever or _PLAN["lever"])
    fn = getattr(runner, attr, None)
    setattr(runner, attr, jit_sharded(fn, rmesh, n_args=n_args, lever=lv, **jit_kwargs))
    _PLAN["runner_wrapped"] = f"{type(runner).__name__}.{attr}"
    return runner


# ----------------------------------------------------------------------------------------------------------------- the model group
def _rows_extent(n_rows: int) -> int:
    """``N / P`` at the model boundary (:func:`shard.local_extent`); an N that does not divide by P is refused by name WITH the remedy: under the model / heads
    groups the whole program holds the pair in the row layout, so the kit pads the INPUT to a multiple of P (its padding path or its length-bucket ladder) —
    the trunk group alone pads each region call instead."""
    lv, p = str(_PLAN["lever"] or FAMILY_LEVER), int(_PLAN["n_gpu"])
    try:
        return _shard.local_extent(int(n_rows), p, lv)
    except MemLeverRefused as e:
        raise MemLeverRefused(lv, f"{e.reason}; sites=model/heads keep the pair row-sharded through the whole program: pad the input to a multiple of n_gpu={p} "
                                  f"(the kit's padding path or length-bucket ladder)") from None


def _model_here() -> bool:
    return installed() and bool(_PLAN["model"]) and not in_manual_region()


def _heads_here() -> bool:
    return installed() and bool(_PLAN["heads_on"]) and not in_manual_region()


def _optional_module(name: str):
    """A sibling module of this package by dotted name, or None when this tree of the core lacks it — the caller then records the NAMED replicated state
    (``conf_reason=alphafold_heads_absent`` / ``template_reason=alphafold_template_absent`` on :func:`describe`)."""
    import importlib  # noqa: PLC0415
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def _tree_facts(modules: Any, multimer: Any = None) -> Dict[str, Any]:
    """Facts of the pinned ``AlphaFoldIteration.__call__`` source the heads-region transcription branches on: ``pae_asym_id`` — the aligned-error output
    carries ``batch['asym_id']`` when the batch has it (alphafold-colabfold 2.3.13 modules.py:359-360; absent in the dl_binder_design tree)."""
    import inspect  # noqa: PLC0415
    src = inspect.getsource(modules.AlphaFoldIteration.__call__)
    return {"pae_asym_id": "ret[name]['asym_id'] = batch['asym_id']" in src}


def _haiku(lever: str):
    try:
        import haiku as hk  # noqa: PLC0415 — the model's own dependency, present wherever these classes are traced
    except ImportError as e:
        raise MemLeverRefused(lever, f"haiku not importable ({e})") from None
    return hk


def _intercept_methods(hk: Any):
    """``hk.intercept_methods`` (``hk.experimental.intercept_methods`` where the release keeps it there)."""
    fn = getattr(hk, "intercept_methods", None)
    return fn if fn is not None else hk.experimental.intercept_methods


def _prologue_interceptor(next_f, args, kwargs, context):
    """Inside ``EmbeddingsAndEvoformer.__call__`` (scope :data:`EVOFORMER_SCOPE`) the ``__call__`` of the modules named in :data:`PROLOGUE_PAIR_MODULES` —
    the recycled-position distogram Linear, the recycled-pair LayerNorm, the relative-position Linear — take their ``[N, N, c]`` operand and give their
    ``[N, N, c_z]`` output pinned to the ROW layout (``with_sharding_constraint``), and the two single projections of :data:`PROLOGUE_SINGLE_MODULES` give
    ``left_single`` ``[N, c_z]`` pinned to rows / ``right_single`` replicated, so the broadcast sum ``left[:, None] + right[None]`` that CREATES the pair is
    born in the row layout: every ``[N, N, c]`` statement of the prologue is placed, none is left to propagation. Every other call passes through untouched."""
    if not _model_here() or getattr(context, "method_name", "__call__") != "__call__":
        return next_f(*args, **kwargs)
    parts = str(getattr(getattr(context, "module", None), "module_name", "") or "").split("/")
    name = parts[-1] if len(parts) >= 2 and EVOFORMER_SCOPE in parts[:-1] else None
    rmesh = _PLAN["rmesh"]
    if name in PROLOGUE_SINGLE_MODULES:                                       # the broadcast-sum site: left rows ⊕ right whole → the pair is BORN in the row layout
        out = next_f(*args, **kwargs)
        if getattr(out, "ndim", 0) != 2:
            return out
        return _shard.constrain(out, rmesh) if name == PROLOGUE_SINGLE_MODULES[0] else _shard.replicate(out, rmesh)
    if name not in PROLOGUE_PAIR_MODULES:
        return next_f(*args, **kwargs)
    if args and getattr(args[0], "ndim", 0) == 3:
        args = (_shard.constrain(args[0], rmesh),) + tuple(args[1:])
    out = next_f(*args, **kwargs)
    return _shard.constrain(out, rmesh) if getattr(out, "ndim", 0) == 3 else out


def _make_embeddings_call(key: str):
    """``EmbeddingsAndEvoformer.__call__`` of ``modules`` / ``modules_multimer`` (BODIES): the STOCK body with the recycled pair entering in the row layout,
    the prologue producers pinned to rows (:func:`_prologue_interceptor`) and the pair leaving in rows (the trunk → heads boundary)."""
    def _embeddings_call(self, batch, is_training, safe_key=None):
        stock = _PLAN["stock"][key]
        if not _model_here():
            return stock(self, batch, is_training, safe_key=safe_key)
        rmesh, lv = _PLAN["rmesh"], _PLAN["lever"]
        hk = _haiku(lv)
        batch = dict(batch)
        if "prev_pair" in batch:
            _rows_extent(int(batch["prev_pair"].shape[0]))                           # N % P != 0 is refused by name (with the remedy) under the model group
            batch["prev_pair"] = _shard.constrain(batch["prev_pair"], rmesh)
        with _intercept_methods(hk)(_prologue_interceptor):
            out = stock(self, batch, is_training, safe_key=safe_key)
        out = dict(out)
        out["pair"] = _shard.constrain(out["pair"], rmesh)
        return out
    _embeddings_call.__name__ = "_embeddings_call"
    return _embeddings_call


def _pair_shaped(x: Any) -> bool:
    """A ``[N, N, c]``-looking array leaf: 3-D with equal leading extents (used only to REFUSE unknown names, never to select)."""
    shape = getattr(x, "shape", None)
    return shape is not None and len(shape) == 3 and int(shape[0]) == int(shape[1])


def _constrain_loop_carry(carry: Any, path: str = "") -> Any:
    """A loop carry with its pair leaves pinned to the row layout BY NAME: dict entries named in :data:`PAIR_CARRY_KEYS` are constrained
    (``with_sharding_constraint``, inside jit, outside shard_map); entries named in :data:`KNOWN_CARRY_KEYS` and non-array leaves are left alone whatever their
    shape; an array entry under an UNKNOWN name that looks pair-shaped (``[a, a, c]``) is refused by name at trace — a tree with a new pair carry is reviewed into
    the table, never row-constrained or replicated silently. Dicts / tuples / lists are walked."""
    rmesh, lv = _PLAN["rmesh"], str(_PLAN["lever"] or FAMILY_LEVER)
    if isinstance(carry, dict):
        out = {}
        for k, v in carry.items():
            if isinstance(v, (dict, tuple, list)):
                out[k] = _constrain_loop_carry(v, f"{path}/{k}")
            elif k in PAIR_CARRY_KEYS and getattr(v, "ndim", 0) == 3:
                out[k] = _shard.constrain(v, rmesh)
            elif k not in KNOWN_CARRY_KEYS and k not in PAIR_CARRY_KEYS and _pair_shaped(v):
                raise MemLeverRefused(lv, f"loop carry leaf {path}/{k} shape {tuple(int(s) for s in v.shape)} looks pair-shaped but is not a name of the plan "
                                          f"(pair: {','.join(PAIR_CARRY_KEYS)}; known non-pair: {','.join(KNOWN_CARRY_KEYS)}) — review it into the table")
            else:
                out[k] = v
        return out
    if isinstance(carry, (tuple, list)):
        seq = [_constrain_loop_carry(c, f"{path}[{i}]") if isinstance(c, (dict, tuple, list)) else c for i, c in enumerate(carry)]
        return tuple(seq) if isinstance(carry, tuple) else seq
    return carry


class _HKProxy:
    """The ``hk`` name of ``modules`` / ``modules_multimer``: every attribute is haiku's; ``while_loop`` (the monomer tree's in-graph recycle loop — the
    initial ``prev_pair`` zeros are born constrained — and its ensemble loop, in the program even at zero trips), ``fori_loop`` and ``scan`` (the multimer
    ensemble loop) pin the pair leaves of the carry (:func:`_constrain_loop_carry`: by name) to the row layout on the initial value, the body's result and the
    loop's result — the recycled pair never crosses a loop boundary whole (BODIES ``hk.while_loop/hk.scan``)."""
    def __init__(self, hk_mod: Any):
        self._hk = hk_mod

    def __getattr__(self, k):
        return getattr(self._hk, k)

    def while_loop(self, cond_fun, body_fun, init_val):
        if not _model_here():
            return self._hk.while_loop(cond_fun, body_fun, init_val)
        cc = _constrain_loop_carry
        return cc(self._hk.while_loop(cond_fun, lambda x: cc(body_fun(x)), cc(init_val)))

    def fori_loop(self, lower, upper, body_fun, init_val):
        if not _model_here():
            return self._hk.fori_loop(lower, upper, body_fun, init_val)
        cc = _constrain_loop_carry
        return cc(self._hk.fori_loop(lower, upper, lambda i, x: cc(body_fun(i, x)), cc(init_val)))

    def scan(self, f, init, xs, *args, **kwargs):
        if not _model_here():
            return self._hk.scan(f, init, xs, *args, **kwargs)
        cc = _constrain_loop_carry

        def g(carry, x):
            carry, y = f(carry, x)
            return cc(carry), y
        carry, ys = self._hk.scan(g, cc(init), xs, *args, **kwargs)
        return cc(carry), ys


# ----------------------------------------------------------------------------------------------------------------- the heads region
def _run_heads_region(module_self, heads: Dict[str, Any], representations: Dict[str, Any], batch: Dict[str, Any], is_training: bool, order: str):
    """The heads of ``AlphaFoldIteration`` in ONE manual region after the trunk: ``representations['pair']`` enters as this device's ROW BLOCK ``[N/P, N, c_z]``
    (msa / single / the batch replicated), the region's Haiku frame carries the structure module's key drawn at the stock position, the flags of the trace state
    are raised (``heads_rows``: structure=sharded; ``heads_rows_conf``: conf=sharded) so the rebound head bodies (:data:`HEADS_MODULE`, the structure banner) run
    on rows, and a consumer whose word reads ``replicated`` is handed the pair gathered ONCE inside the region (:func:`transition.rows_full`). The stock heads loop
    (``order``: ``monomer`` = alphafold-colabfold 2.3.13 modules.py:328-368 = dl_binder_design modules.py:235-268; ``multimer`` = modules_multimer.py:356-403) runs
    inside verbatim. Row-sharded outputs LEAVE the region row-sharded (``representations['pair']``; the distogram / aligned-error logits under conf=sharded);
    everything else leaves replicated. structure=replicated on the multimer tree: the stock ``folding_multimer`` structure module runs BEFORE the region on the
    program's arrays (``structure_reason`` carries ``multimer_stock_outside_region``) — its body holds a ``zeros_like`` of a parameter a manual region cannot trace."""
    M, rmesh, axis, lv = _PLAN["modules"], _PLAN["rmesh"], _PLAN["axis"], _PLAN["lever"]
    hk = _haiku(lv)
    structure_rows, conf_rows = _PLAN["structure"] == "sharded", _PLAN["conf"] == "sharded"
    representations = dict(representations)
    n = int(representations["pair"].shape[0])
    _rows_extent(n)
    representations["pair"] = _shard.constrain(representations["pair"], rmesh)   # the trunk → heads boundary
    import jax.numpy as jnp  # noqa: PLC0415
    runs_structure = "structure_module" in heads and (order != "multimer" or ("entity_id" in batch and "all_atom_positions" in batch))
    sm_outside = None
    if runs_structure and order == "multimer" and not structure_rows:          # structure=replicated on the multimer tree: the STOCK ``folding_multimer`` structure module runs HERE, before the
        _, fold_module = heads["structure_module"]                            # region, on the program's arrays (its stock body holds ``jnp.zeros_like(<parameter>)``, folding_multimer.py:265, which a
        sm_outside = fold_module(representations, batch, is_training)         # manual region cannot trace); it draws its own key at the stock position (modules_multimer.py:359-362 run it first)
    draws_key = runs_structure and sm_outside is None
    heads_key = hk.next_rng_key() if draws_key else jnp.zeros((2,), jnp.uint32)   # = StructureModule.__call__'s draw (folding.py:482-483; folding_multimer.py:577-578): same position in the outer sequence; no draw → an rng-less frame (the placeholder is never read)
    sm_arg = sm_outside if sm_outside is not None else jnp.zeros((), jnp.float32)   # the outside structure output enters the region replicated (placeholder otherwise, never read)
    row_outputs = [name for name in PAIR_HEADS if name in heads and conf_rows]   # head outputs whose 'logits' leave row-sharded (fixed before the trace: the out_specs)

    def pick_pair(name, pair_rows, pair_full):
        rows_ok = structure_rows if name == "structure_module" else conf_rows if name in PAIR_HEADS else True
        return pair_rows if rows_ok else pair_full

    def heads_body(reps_, batch_, key_, sm_):
        with _manual_body(key_ if draws_key else None, lv):
            _S.heads_rows, _S.heads_rows_conf = structure_rows, conf_rows
            try:
                reps = dict(reps_)
                pair_rows = reps["pair"]
                need_full = ("structure_module" in heads and not structure_rows and sm_outside is None) or (any(h in heads for h in PAIR_HEADS) and not conf_rows)
                pair_full = _transition.rows_full(pair_rows, axis) if need_full else None   # conf|structure=replicated: ONE all_gather of the f32 pair inside the region (named)
                ret: Dict[str, Any] = {}
                if order == "multimer":                                        # modules_multimer.py:356-403
                    structure_module_output = dict(sm_) if sm_outside is not None else None
                    if structure_module_output is None and "structure_module" in heads and "entity_id" in batch_ and "all_atom_positions" in batch_:
                        _, fold_module = heads["structure_module"]
                        reps["pair"] = pick_pair("structure_module", pair_rows, pair_full)
                        structure_module_output = fold_module(reps, batch_, is_training)
                    ret["representations"] = reps
                    for name, (head_config, module) in heads.items():
                        if name == "structure_module" and structure_module_output is not None:
                            ret[name] = structure_module_output
                            reps["structure_module"] = structure_module_output.pop("act")
                        elif name in {"predicted_lddt", "predicted_aligned_error", "experimentally_resolved"}:
                            continue
                        else:
                            reps["pair"] = pick_pair(name, pair_rows, pair_full)
                            ret[name] = module(reps, batch_, is_training)
                    if module_self.config.heads.get("predicted_lddt.weight", 0.0):
                        name = "predicted_lddt"
                        head_config, module = heads[name]
                        ret[name] = module(reps, batch_, is_training)
                    if module_self.config.heads.experimentally_resolved.weight:
                        name = "experimentally_resolved"
                        head_config, module = heads[name]
                        ret[name] = module(reps, batch_, is_training)
                    if module_self.config.heads.get("predicted_aligned_error.weight", 0.0):
                        name = "predicted_aligned_error"
                        head_config, module = heads[name]
                        reps["pair"] = pick_pair(name, pair_rows, pair_full)
                        ret[name] = module(reps, batch_, is_training)
                        ret[name]["asym_id"] = batch_["asym_id"]
                else:                                                          # modules.py (cf :328-368 = dl :235-268), compute_loss=False
                    ret["representations"] = reps
                    for name, (head_config, module) in heads.items():
                        if name in ("predicted_lddt", "predicted_aligned_error"):
                            continue
                        reps["pair"] = pick_pair(name, pair_rows, pair_full)
                        ret[name] = module(reps, batch_, is_training)
                        if "representations" in ret[name]:
                            reps.update(ret[name].pop("representations"))
                    if module_self.config.heads.get("predicted_lddt.weight", 0.0):
                        name = "predicted_lddt"
                        head_config, module = heads[name]
                        ret[name] = module(reps, batch_, is_training)
                    if ("predicted_aligned_error" in module_self.config.heads and module_self.config.heads.get("predicted_aligned_error.weight", 0.0)):
                        name = "predicted_aligned_error"
                        head_config, module = heads[name]
                        reps["pair"] = pick_pair(name, pair_rows, pair_full)
                        ret[name] = module(reps, batch_, is_training)
                        if _TREE["pae_asym_id"] and "asym_id" in batch_:
                            ret[name]["asym_id"] = batch_["asym_id"]
                reps["pair"] = pair_rows                                       # the returned pair representation: my rows
                rows_out = tuple(ret[name].pop("logits") for name in row_outputs)
                return ret, reps.pop("pair"), rows_out
            finally:
                _S.heads_rows, _S.heads_rows_conf = False, False

    rows, rep = _shard.rows_spec(axis, 3, 0), _shard.replicated_spec()
    reps_in = dict(representations)
    in_specs = ({k: (rows if k == "pair" else rep) for k in reps_in}, rep, rep, rep)
    out_specs = (rep, rows, tuple(rows for _ in row_outputs))
    _PLAN["regions_built"] += 1
    ret, pair_out, rows_out = _shard.shard_map(heads_body, rmesh, in_specs, out_specs)(reps_in, batch, heads_key, sm_arg)
    ret["representations"]["pair"] = _shard.constrain(pair_out, rmesh)
    for name, logits in zip(row_outputs, rows_out):
        ret[name]["logits"] = _shard.constrain(logits, rmesh)
    return ret


def _alphafold_iteration_call(self, ensembled_batch, non_ensembled_batch, is_training, compute_loss=False, ensemble_representations=False,
                              return_representations=False):
    """Monomer ``AlphaFoldIteration.__call__`` (alphafold-colabfold 2.3.13 modules.py:236-368 = dl_binder_design modules.py:143-268) verbatim up to the heads;
    the heads run in :func:`_run_heads_region`. ``compute_loss=True`` (training losses) and Haiku initialisation keep the stock body."""
    stock = _PLAN["stock"]["AlphaFoldIteration"]
    lv = _PLAN["lever"]
    hk = _haiku(lv)
    if not _heads_here() or hk.running_init():
        return stock(self, ensembled_batch, non_ensembled_batch, is_training, compute_loss=compute_loss,
                     ensemble_representations=ensemble_representations, return_representations=return_representations)
    if compute_loss:
        raise MemLeverRefused(lv, "heads region: compute_loss=True is not a path of the row-shard plan (inference only)")
    M = _PLAN["modules"]
    F = _PLAN["library"]["folding"]
    import functools  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    num_ensemble = jnp.asarray(ensembled_batch['seq_length'].shape[0])
    if not ensemble_representations:
        assert ensembled_batch['seq_length'].shape[0] == 1

    def slice_batch(i):
        b = {k: v[i] for k, v in ensembled_batch.items()}
        b.update(non_ensembled_batch)
        return b

    evoformer_module = M.EmbeddingsAndEvoformer(self.config.embeddings_and_evoformer, self.global_config)
    batch0 = slice_batch(0)
    representations = evoformer_module(batch0, is_training)
    msa_representation = representations['msa']
    del representations['msa']
    if ensemble_representations:
        def body(x):
            """Add one element to the representations ensemble."""
            i, current_representations = x
            feats = slice_batch(i)
            representations_update = evoformer_module(feats, is_training)
            new_representations = {}
            for k in current_representations:
                new_representations[k] = (current_representations[k] + representations_update[k])
            return i + 1, new_representations

        _, representations = M.hk.while_loop(lambda x: x[0] < num_ensemble, body, (1, representations))   # M.hk: the carry's pair stays in rows (_HKProxy)
        for k in representations:
            if k != 'msa':
                representations[k] /= num_ensemble.astype(representations[k].dtype)
    representations['msa'] = msa_representation
    batch = batch0
    heads = {}
    for head_name, head_config in sorted(self.config.heads.items()):
        if not head_config.weight:
            continue
        head_factory = {'masked_msa': M.MaskedMsaHead, 'distogram': M.DistogramHead, 'structure_module': functools.partial(F.StructureModule, compute_loss=compute_loss),
                        'predicted_lddt': M.PredictedLDDTHead, 'predicted_aligned_error': M.PredictedAlignedErrorHead,
                        'experimentally_resolved': M.ExperimentallyResolvedHead}[head_name]
        heads[head_name] = (head_config, head_factory(head_config, self.global_config))
    return _run_heads_region(self, heads, representations, batch, is_training, "monomer")


def _alphafold_iteration_multimer_call(self, batch, is_training, return_representations=False, safe_key=None):
    """``modules_multimer.AlphaFoldIteration.__call__`` (alphafold-colabfold 2.3.13 modules_multimer.py:295-403) verbatim up to the heads (the ensemble scan's
    carry keeps the pair in rows: :class:`_HKProxy`); the heads run in :func:`_run_heads_region`. Haiku initialisation keeps the stock body."""
    stock = _PLAN["stock"]["multimer.AlphaFoldIteration"]
    lv = _PLAN["lever"]
    hk = _haiku(lv)
    if not _heads_here() or hk.running_init():
        return stock(self, batch, is_training, return_representations=return_representations, safe_key=safe_key)
    M, M2 = _PLAN["modules"], _PLAN["multimer"]
    FM = _PLAN["library"]["folding_multimer"]
    import jax.numpy as jnp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    if is_training:
        num_ensemble = np.asarray(self.config.num_ensemble_train)
    else:
        num_ensemble = np.asarray(self.config.num_ensemble_eval)
    embedding_module = M2.EmbeddingsAndEvoformer(self.config.embeddings_and_evoformer, self.global_config)
    repr_shape = hk.eval_shape(lambda: embedding_module(batch, is_training))
    representations = {k: jnp.zeros(v.shape, v.dtype) for (k, v) in repr_shape.items()}

    def ensemble_body(x, unused_y):
        """Add into representations ensemble."""
        del unused_y
        representations, safe_key = x
        safe_key, safe_subkey = safe_key.split()
        representations_update = embedding_module(batch, is_training, safe_key=safe_subkey)
        for k in representations:
            if k not in {'msa', 'true_msa', 'bert_mask'}:
                representations[k] += representations_update[k] * (1. / num_ensemble).astype(representations[k].dtype)
            else:
                representations[k] = representations_update[k]
        return (representations, safe_key), None

    (representations, _), _ = M2.hk.scan(ensemble_body, (representations, safe_key), None, length=num_ensemble)   # M2.hk: the carry's pair stays in rows
    self.representations = representations
    self.batch = batch
    self.heads = {}
    for head_name, head_config in sorted(self.config.heads.items()):
        if not head_config.weight:
            continue
        head_factory = {'masked_msa': M.MaskedMsaHead, 'distogram': M.DistogramHead, 'structure_module': FM.StructureModule,
                        'predicted_aligned_error': M.PredictedAlignedErrorHead, 'predicted_lddt': M.PredictedLDDTHead,
                        'experimentally_resolved': M.ExperimentallyResolvedHead}[head_name]
        self.heads[head_name] = (head_config, head_factory(head_config, self.global_config))
    return _run_heads_region(self, self.heads, representations, batch, is_training, "multimer")


class _ConfidenceProxy:
    """The ``confidence`` name of ``modules`` / ``modules_multimer`` (``alphafold.common.confidence``, whose ``get_confidence_metrics`` the trees' ``AlphaFold.__call__``
    runs IN the program on the head outputs: alphafold-colabfold 2.3.13 modules.py:466-476, modules_multimer.py:467-478): every attribute is the module's;
    ``get_confidence_metrics`` under conf=sharded is a region over the aligned-error logits ROWS (pLDDT logits, breaks, mask, asym_id replicated) inside which the
    module's function runs with ``heads_rows_conf()`` raised (its row body: :data:`HEADS_MODULE`) — the ``[N, N, bins]`` probabilities and the ``[N, N]``
    expected error LEAVE row-sharded, the scalars replicated. ``keep_pae``'s aliasing of the head output happens outside the region as stock (confidence.py:180-181).
    conf=replicated: the module's stock function on the program's arrays (named)."""
    def __init__(self, conf_mod: Any):
        self._c = conf_mod

    def __getattr__(self, k):
        return getattr(self._c, k)

    def get_confidence_metrics(self, prediction_result, mask, rank_by="plddt", use_jnp=False, keep_pae=False):
        C = self._c
        if not (_heads_here() and use_jnp and _PLAN["conf"] == "sharded"):
            return C.get_confidence_metrics(prediction_result, mask, rank_by=rank_by, use_jnp=use_jnp, keep_pae=keep_pae)
        rmesh, axis, lv = _PLAN["rmesh"], _PLAN["axis"], _PLAN["lever"]
        rows3, rows2, rep = _shard.rows_spec(axis, 3, 0), _shard.rows_spec(axis, 2, 0), _shard.replicated_spec()
        has_pae = "predicted_aligned_error" in prediction_result
        sub = {"predicted_lddt": {"logits": prediction_result["predicted_lddt"]["logits"]}}
        specs: Dict[str, Any] = {"predicted_lddt": {"logits": rep}}
        out_specs: Dict[str, Any] = {"plddt": rep, "mean_plddt": rep, "ranking_confidence": rep}
        if has_pae:
            if keep_pae:
                prediction_result["pae_matrix_with_logits"] = prediction_result["predicted_aligned_error"]   # confidence.py:180-181
            pae = prediction_result["predicted_aligned_error"]
            _rows_extent(int(pae["logits"].shape[0]))
            sub["predicted_aligned_error"] = {"logits": _shard.constrain(pae["logits"], rmesh), "breaks": pae["breaks"]}
            specs["predicted_aligned_error"] = {"logits": rows3, "breaks": rep}
            out_specs.update({"aligned_confidence_probs": rows3, "predicted_aligned_error": rows2, "max_predicted_aligned_error": rep, "ptm": rep})
            if "asym_id" in pae:
                sub["predicted_aligned_error"]["asym_id"] = pae["asym_id"]
                specs["predicted_aligned_error"]["asym_id"] = rep
                out_specs["iptm"] = rep

        def body(sub_, mask_):
            with _manual_body(None, lv):
                _S.heads_rows_conf = True
                try:
                    return C.get_confidence_metrics(sub_, mask_, rank_by=rank_by, use_jnp=True, keep_pae=False)
                finally:
                    _S.heads_rows_conf = False

        _PLAN["regions_built"] += 1
        out = _shard.shard_map(body, rmesh, (specs, rep), out_specs)(sub, mask)
        if has_pae:
            out["aligned_confidence_probs"] = _shard.constrain(out["aligned_confidence_probs"], rmesh)
            out["predicted_aligned_error"] = _shard.constrain(out["predicted_aligned_error"], rmesh)
        return out


# ----------------------------------------------------------------------------------------------------------------- the two regions
def _sharded_here() -> bool:
    return installed() and in_sharded_region()


def _evoformer_iteration_call(self, activations, masks, is_training=True, safe_key=None):
    """``EvoformerIteration.__call__``: outside a region and with the plan installed → run the STOCK body inside one region over this block;
    inside a region (never nested by the stock trees) or without the plan → the stock body as is."""
    stock = _PLAN["stock"]["EvoformerIteration"]
    if not installed() or in_manual_region():
        return stock(self, activations, masks, is_training=is_training, safe_key=safe_key)
    M, rmesh, axis = _PLAN["modules"], _PLAN["rmesh"], _PLAN["axis"]
    if safe_key is None:                                                     # the stock default draws hk.next_rng_key() — outside the region, as stock
        safe_key = M.prng.SafeKey(_hk_next_rng_key(M))
    msa, pair = activations["msa"], activations["pair"]
    msa_mask, pair_mask = masks["msa"], masks["pair"]
    n, d = _residue_pad(pair)                                                # N % P != 0: pad the residue axes with masked zeros for this region, slice after
    msa, msa_mask = _pad(msa, d, (1,)), _pad(msa_mask, d, (1,))
    pair, pair_mask = _pad(pair, d, (0, 1)), _pad(pair_mask, d, (0, 1))
    rows, rep = _shard.rows_spec(axis, 3, 0), _shard.replicated_spec()
    pair = _shard.constrain(pair, rmesh)                                     # the carry enters in the row layout

    def body(msa_b, pair_b, msa_mask_b, pair_mask_full, key_b):
        pair_mask_loc = _triatt.mask_rows(pair_mask_full, axis, int(pair_b.shape[0]))
        out = stock(self, {"msa": msa_b, "pair": pair_b}, {"msa": msa_mask_b, "pair": pair_mask_loc},
                    is_training=is_training, safe_key=M.prng.SafeKey(key_b))
        return out["msa"], out["pair"]

    region = _region(body, rmesh, in_specs=(rep, rows, rep, rep, rep), out_specs=(rep, rows))
    msa_out, pair_out = region(msa, pair, msa_mask, pair_mask, safe_key.get())
    if d:
        _PLAN["padded_calls"] += 1
        return {"msa": msa_out[:, :n], "pair": pair_out[:n, :n]}
    return {"msa": msa_out, "pair": _shard.constrain(pair_out, rmesh)}


def _template_iteration_call(self, act, pair_mask, is_training=True, safe_key=None):
    """``modules_multimer.TemplateEmbeddingIteration.__call__`` — the template pair stack's block as a region over (act rows, mask, key)."""
    stock = _PLAN["stock"][TEMPLATE_ITERATION]
    if not installed() or in_manual_region():
        return stock(self, act, pair_mask, is_training=is_training, safe_key=safe_key)
    M, rmesh, axis = _PLAN["modules"], _PLAN["rmesh"], _PLAN["axis"]
    if safe_key is None:
        safe_key = M.prng.SafeKey(_hk_next_rng_key(M))
    n, d = _residue_pad(act)
    act, pair_mask = _pad(act, d, (0, 1)), _pad(pair_mask, d, (0, 1))
    rows, rep = _shard.rows_spec(axis, 3, 0), _shard.replicated_spec()
    act = _shard.constrain(act, rmesh)

    def body(act_b, pair_mask_full, key_b):
        pair_mask_loc = _triatt.mask_rows(pair_mask_full, axis, int(act_b.shape[0]))
        return stock(self, act_b, pair_mask_loc, is_training=is_training, safe_key=M.prng.SafeKey(key_b))

    region = _region(body, rmesh, in_specs=(rows, rep, rep), out_specs=rows)
    out = region(act, pair_mask, safe_key.get())
    if d:
        _PLAN["padded_calls"] += 1
        return out[:n, :n]
    return _shard.constrain(out, rmesh)


def _template_pair_stack_call(self, pair_act, pair_mask, is_training, safe_key=None):
    """``modules.TemplatePairStack.__call__`` (the monomer template pair stack, ``layer_stack`` over its iteration inside) as a region over (act rows,
    mask, key) — the same form as the multimer template iteration; its sub-layers are the patched classes."""
    stock = _PLAN["stock"][TEMPLATE_PAIR_STACK]
    if not installed() or in_manual_region():
        return stock(self, pair_act, pair_mask, is_training, safe_key=safe_key)
    M, rmesh, axis = _PLAN["modules"], _PLAN["rmesh"], _PLAN["axis"]
    if safe_key is None:
        safe_key = M.prng.SafeKey(_hk_next_rng_key(M))
    n, d = _residue_pad(pair_act)
    pair_act, pair_mask = _pad(pair_act, d, (0, 1)), _pad(pair_mask, d, (0, 1))
    rows, rep = _shard.rows_spec(axis, 3, 0), _shard.replicated_spec()
    pair_act = _shard.constrain(pair_act, rmesh)

    def body(act_b, pair_mask_full, key_b):
        pair_mask_loc = _triatt.mask_rows(pair_mask_full, axis, int(act_b.shape[0]))
        return stock(self, act_b, pair_mask_loc, is_training, safe_key=M.prng.SafeKey(key_b))

    region = _region(body, rmesh, in_specs=(rows, rep, rep), out_specs=rows)
    out = region(pair_act, pair_mask, safe_key.get())
    if d:
        _PLAN["padded_calls"] += 1
        out = out[:n, :n]
    if _PLAN.get("template") == "rows":                                      # template=rows: the template embedding's consumers run on rows (TEMPLATE_MODULE) — no hand-back
        return out if d else _shard.constrain(out, rmesh)
    # template=replicated (trunk group alone): the consumer is the stock template-pointwise attention, an inference_subbatch scan with dynamic_slice
    # over the flattened N*N axis that must run the single-device program (a scan over a row-sharded operand reshards inside every iteration).
    # One all_gather of [N, N, c_t] per template per recycle.
    return _shard.replicate(out, rmesh)


def _residue_pad(pair_like):
    """``(N, d)``: the residue count of this region call and the zero rows/columns that make it a multiple of P (``d = (-N) mod P``). Padded positions
    carry mask 0: the stock bodies multiply the triangle-multiplication operands by the mask (padded k add exact zeros), give padded keys the
    ``1e9·(mask−1)`` attention bias (softmax weight 0), and normalise the outer-product mean per (i, j) — the real positions' values are the unpadded
    computation's up to GEMM tiling (the P>1 tolerance class); padded rows are finite and sliced off. P=1 never reaches here."""
    n = int(pair_like.shape[0])
    return n, (-n) % int(_PLAN["n_gpu"])


def _pad(x, d: int, axes):
    """``x`` zero-padded by ``d`` at the END of each of ``axes`` (no-op for d == 0)."""
    if not d:
        return x
    from ._lazy import jnp as _lazy_jnp  # noqa: PLC0415
    jnp = _lazy_jnp(_PLAN["lever"] or FAMILY_LEVER)
    widths = [(0, 0)] * x.ndim
    for a in axes:
        widths[a] = (0, d)
    return jnp.pad(x, widths)


def _site_norm(M, name):
    """The pair sub-layers' LayerNorm EXACTLY as each pinned tree spells it at these sites (TriangleAttention ``query_norm``, MSARowAttentionWithPairBias
    ``query_norm``/``feat_2d_norm``, OuterProductMean ``layer_norm_input``, unfused TriangleMultiplication ``layer_norm_input``/``center_layer_norm``):
    ``common_modules.LayerNorm(axis=[-1], create_scale=True, create_offset=True, name=…)`` (2.3.x: two-pass variance, ``use_fast_variance=False``) or, in the
    tree without ``common_modules.LayerNorm``, ``hk.LayerNorm(axis=[-1], create_scale=True, create_offset=True, name=…)``."""
    LN = getattr(M.common_modules, "LayerNorm", None)
    if LN is None:
        import haiku as hk  # noqa: PLC0415
        LN = hk.LayerNorm
    return LN(axis=[-1], create_scale=True, create_offset=True, name=name)


def _fused_norm(M, name):
    """The FUSED TriangleMultiplication's norms only (``left_norm_input``, ``center_norm``): the tree's ``modules._layer_norm(axis=-1, name=…)`` (one-pass
    ``use_fast_variance=True``) — a tree with the fused body always defines it."""
    return M._layer_norm(axis=-1, name=name)


def _hk_next_rng_key(M):
    import haiku as hk  # noqa: PLC0415 — the model's own dependency, present wherever these classes are traced
    return hk.next_rng_key()


# ----------------------------------------------------------------------------------------------------------------- the sub-layer bodies
def _triangle_multiplication_call(self, left_act, left_mask, is_training=True):
    """``TriangleMultiplication.__call__`` on my row block: the stock fused body (``modules.py`` ``_fused_triangle_multiplication``: ``left_norm_input``,
    ``projection``/``gate`` of width ``2*c_i``, ``center_norm``) or the stock unfused body (``_triangle_multiplication`` / the earlier library's
    ``__call__``: ``layer_norm_input``, ``left_/right_projection``, ``left_/right_gate``, ``center_layer_norm``) by ``config.fuse_projection_weights``
    (absent = unfused) — the same parameter names — with the contraction served by ``trimul.contract``, complete over ``k`` for my output rows."""
    stock = _PLAN["stock"]["TriangleMultiplication"]
    if not _sharded_here():
        return stock(self, left_act, left_mask, is_training)
    del is_training
    M, axis = _PLAN["modules"], _PLAN["axis"]
    c, gc = self.config, self.global_config
    import jax  # noqa: PLC0415
    mask = left_mask[..., None]
    if bool(getattr(c, "fuse_projection_weights", False)):
        left_act = _fused_norm(M, "left_norm_input")(left_act)
        projection = M.common_modules.Linear(2 * c.num_intermediate_channel, name="projection")
        proj_act = mask * projection(left_act)
        gate_values = M.common_modules.Linear(2 * c.num_intermediate_channel, name="gate", bias_init=1., initializer=M.utils.final_init(gc))(left_act)
        proj_act *= jax.nn.sigmoid(gate_values)
        left_proj_act = proj_act[:, :, :c.num_intermediate_channel]
        right_proj_act = proj_act[:, :, c.num_intermediate_channel:]
        act = _trimul.contract(c.equation, left_proj_act, right_proj_act, axis, schedule=_PLAN["schedule"], lever=_PLAN["lever"])
        act = _fused_norm(M, "center_norm")(act)
    else:
        act = _site_norm(M, "layer_norm_input")(left_act)
        input_act = act
        left_proj_act = mask * M.common_modules.Linear(c.num_intermediate_channel, name="left_projection")(act)
        right_proj_act = mask * M.common_modules.Linear(c.num_intermediate_channel, name="right_projection")(act)
        left_gate_values = jax.nn.sigmoid(M.common_modules.Linear(c.num_intermediate_channel, bias_init=1., initializer=M.utils.final_init(gc), name="left_gate")(act))
        right_gate_values = jax.nn.sigmoid(M.common_modules.Linear(c.num_intermediate_channel, bias_init=1., initializer=M.utils.final_init(gc), name="right_gate")(act))
        left_proj_act *= left_gate_values
        right_proj_act *= right_gate_values
        act = _trimul.contract(c.equation, left_proj_act, right_proj_act, axis, schedule=_PLAN["schedule"], lever=_PLAN["lever"])
        act = _site_norm(M, "center_layer_norm")(act)
        left_act = input_act
    output_channel = int(left_act.shape[-1])
    act = M.common_modules.Linear(output_channel, initializer=M.utils.final_init(gc), name="output_projection")(act)
    gate_values = M.common_modules.Linear(output_channel, bias_init=1., initializer=M.utils.final_init(gc), name="gating_linear")(left_act)
    act *= jax.nn.sigmoid(gate_values)
    return act


def _triangle_attention_call(self, pair_act, pair_mask, is_training=False):
    """``TriangleAttention.__call__`` on my row block (``modules.py:1011-1059``): per_row — my rows are the batch, the non-batched bias gathered
    over rows; per_column — the block transpose in, the same attention, the block transpose out; the mask columns from the gathered mask."""
    stock = _PLAN["stock"]["TriangleAttention"]
    if not _sharded_here():
        return stock(self, pair_act, pair_mask, is_training)
    M, axis = _PLAN["modules"], _PLAN["axis"]
    import haiku as hk  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    c = self.config
    assert len(pair_act.shape) == 3 and len(pair_mask.shape) == 2 and c.orientation in ("per_row", "per_column")
    n_loc = int(pair_act.shape[0])
    if c.orientation == "per_column":
        pair_act = _triatt.enter_transposed(pair_act, axis)                             # my rows of pairᵀ = my columns of pair
        pair_mask = _triatt.mask_cols(_shard.gather(pair_mask, axis, 0), axis, n_loc)   # my columns of the whole mask, as rows
    bias = (1e9 * (pair_mask - 1.))[:, None, None, :]
    pair_act = _site_norm(M, "query_norm")(pair_act)
    init_factor = 1. / jnp.sqrt(int(pair_act.shape[-1]))
    weights = hk.get_parameter("feat_2d_weights", shape=(pair_act.shape[-1], c.num_head), dtype=pair_act.dtype,
                               init=hk.initializers.RandomNormal(stddev=init_factor))
    nonbatched_bias = _triatt.bias_full(jnp.einsum("qkc,ch->hqk", pair_act, weights), axis, row_dim=1)   # [H, N/P, N] → [H, N, N]
    attn_mod = M.Attention(c, self.global_config, pair_act.shape[-1])
    pair_act = M.mapping.inference_subbatch(attn_mod, self.global_config.subbatch_size, batched_args=[pair_act, pair_act, bias],
                                            nonbatched_args=[nonbatched_bias], low_memory=not is_training)
    if c.orientation == "per_column":
        pair_act = _triatt.exit_transposed(pair_act, axis)
    return pair_act


def _outer_product_mean_call(self, act, mask, is_training=True):
    """``OuterProductMean.__call__`` (``modules.py:1604-1672``) producing my output rows: the left projection and the normaliser's left mask are
    sliced to my residues; the right side stays whole (the MSA is replicated). The stock chunk loop runs over my rows only."""
    stock = _PLAN["stock"]["OuterProductMean"]
    if not _sharded_here():
        return stock(self, act, mask, is_training=is_training)
    M, axis = _PLAN["modules"], _PLAN["axis"]
    import haiku as hk  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    gc, c = self.global_config, self.config
    n_loc = _shard.local_extent(int(act.shape[1]), int(_PLAN["n_gpu"]), _PLAN["lever"])
    mask = mask[..., None]
    act = _site_norm(M, "layer_norm_input")(act)
    left_act = mask * M.common_modules.Linear(c.num_outer_channel, initializer="linear", name="left_projection")(act)
    right_act = mask * M.common_modules.Linear(c.num_outer_channel, initializer="linear", name="right_projection")(act)
    left_act, mask_left = _transition.opm_operands(left_act, mask, axis, n_loc, res_dim=1)              # [S, N/P, c], [S, N/P, 1]
    init_w = hk.initializers.Constant(0.0) if gc.zero_init else hk.initializers.VarianceScaling(scale=2., mode="fan_in")
    output_w = hk.get_parameter("output_w", shape=(c.num_outer_channel, c.num_outer_channel, self.num_output_channel), dtype=act.dtype, init=init_w)
    output_b = hk.get_parameter("output_b", shape=(self.num_output_channel,), dtype=act.dtype, init=hk.initializers.Constant(0.0))

    def compute_chunk(left_act):
        left_act = jnp.transpose(left_act, [0, 2, 1])
        act = jnp.einsum("acb,ade->dceb", left_act, right_act)
        act = jnp.einsum("dceb,cef->dbf", act, output_w) + output_b
        return jnp.transpose(act, [1, 0, 2])

    act = M.mapping.inference_subbatch(compute_chunk, c.chunk_size, batched_args=[left_act], nonbatched_args=[], low_memory=True,
                                       input_subbatch_dim=1, output_subbatch_dim=0)
    epsilon = 1e-3
    norm = jnp.einsum("abc,adc->bdc", mask_left, mask)                                                  # [N/P, N, 1]: my rows of the stock normaliser
    act /= epsilon + norm
    return act


def _msa_row_attention_call(self, msa_act, msa_mask, pair_act, is_training=False):
    """``MSARowAttentionWithPairBias.__call__`` (``modules.py:830-883``) with the pair bias of my rows gathered to every row; the MSA attention
    runs replicated (every device holds the MSA)."""
    stock = _PLAN["stock"]["MSARowAttentionWithPairBias"]
    if not _sharded_here():
        return stock(self, msa_act, msa_mask, pair_act, is_training=is_training)
    M, axis = _PLAN["modules"], _PLAN["axis"]
    import haiku as hk  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    c = self.config
    assert len(msa_act.shape) == 3 and len(msa_mask.shape) == 2 and c.orientation == "per_row"
    bias = (1e9 * (msa_mask - 1.))[:, None, None, :]
    msa_act = _site_norm(M, "query_norm")(msa_act)
    pair_act = _site_norm(M, "feat_2d_norm")(pair_act)
    init_factor = 1. / jnp.sqrt(int(pair_act.shape[-1]))
    weights = hk.get_parameter("feat_2d_weights", shape=(pair_act.shape[-1], c.num_head), dtype=msa_act.dtype,
                               init=hk.initializers.RandomNormal(stddev=init_factor))
    nonbatched_bias = _transition.pair_logits_full(jnp.einsum("qkc,ch->hqk", pair_act, weights), axis, row_dim=1)   # [H, N/P, N] → [H, N, N]
    attn_mod = M.Attention(c, self.global_config, msa_act.shape[-1])
    msa_act = M.mapping.inference_subbatch(attn_mod, self.global_config.subbatch_size, batched_args=[msa_act, msa_act, bias],
                                           nonbatched_args=[nonbatched_bias], low_memory=not is_training)
    return msa_act


# ---- structure module (J1-SM) ----
# Everything below this banner is the StructureModule's share of the heads region: the pair rows reach the structure module as this device's ROW
# BLOCK ``[N/P, N, c_z]`` and never come back whole. Names read from above the banner: ``heads_rows()`` (the heads-region flag of the trace state),
# ``_PLAN["axis"]`` / ``_PLAN["n_gpu"]`` / ``_PLAN["lever"]``, ``_hk`` / ``_shard`` / ``MemLeverRefused``.
#
# The structure module's pair consumers, complete (both pinned ``folding.py`` trees and ``folding_multimer.py``; every other statement of
# ``generate_affines`` / ``generate_monomer_rigids`` / ``FoldIteration`` / ``StructureModule`` / ``MultiRigidSidechain`` / ``QuatRigid`` / ``PointProjection``
# is per-residue on ``[N, c]`` activations, affines and 1-D batch fields):
#   * ``pair_layer_norm`` — ``common_modules.LayerNorm`` / ``hk.LayerNorm`` over the channel axis of ``representations['pair']`` (folding.py:436-441 =
#     dl_binder_design folding.py:438-443; folding_multimer.py:526-531): per (i, j) — runs on the row block as is (class row_local, no collective);
#   * ``FoldIteration`` hands the normed block to ``InvariantPointAttention`` as ``inputs_2d`` (folding.py:325-329 / :327-331; folding_multimer.py:421-425) —
#     a pass-through (``hk.scan`` / the Python layer loop close over it);
#   * ``InvariantPointAttention.__call__`` READS it twice: ``attention_2d = Linear(num_head)(inputs_2d)`` — the 2-D logits bias (folding.py:209-215 /
#     :211-217; folding_multimer.py:308-310) — and ``result_attention_over_2d = einsum('hij, ijc->ihc' | 'ijh, ijc->ihc', attn, inputs_2d)`` (folding.py:265 /
#     :267; folding_multimer.py:361). Both are ROW-LOCAL in the query residue ``i`` given every target ``j``; so are the module's own ``[N, N, ·]``-class
#     transients built from 1-D inputs (the point-distance term ``[H, N, N, P_qk]`` folding.py:195-201 / folding_multimer.py:285-287, the scalar logits and
#     ``attn`` ``[H, N, N]``, ``mask_2d`` ``[N, N]``, the point-value product ``[H, N, N, P_v]`` folding.py:232-234 / folding_multimer.py:334-335): with the
#     queries restricted to my rows each is ``[·, N/P, N, ·]`` here.
# Hence ONE rebound site per folding module (:data:`SM_BODIES`): the IPA body with queries = my rows, keys / values / target points over all residues
# (the 1-D activations and the frames are replicated and small), the softmax complete over ``j`` per row (no partial sums), and the concatenated
# ``final_act`` rows all-gathered ONCE to ``[N, F]`` before ``output_projection`` — so ``FoldIteration``'s residual, transition, affine / rigid update and
# side chains stay the replicated stock program on the full ``[N]``. The two monomer trees' ``InvariantPointAttention.__call__`` bodies are statement-
# identical (dl_binder_design cafa3853 = alphafold-colabfold 2.3.13 shifted by +2 lines); ``folding_multimer`` has its own (``Rigid3Array`` frames,
# ``PointProjection`` sub-modules, logits laid out ``[q, k, h]``). Frames are sliced to my rows with the stock's own field maps
# (``QuatAffine.apply_tensor_fn``; ``jax.tree_util.tree_map`` over the ``Rigid3Array`` pytree) — no rotation is recomputed.

SM_PIN: Dict[str, str] = {**PIN_FILES["folding"], **PIN_FILES["folding_multimer"]}   # the folding files the bodies below transcribe (a view of PIN)
SM_LIBRARY = ("alphafold.model.folding", "alphafold.model.folding_multimer")   # imported from the same package as ``modules``; the second only with ``modules_multimer``
SM_SITE = "InvariantPointAttention.__call__"
SM_BODIES = {                                                            # site → (numerics class vs the stock body on the same rows, the stock lines transcribed)
    "folding." + SM_SITE: ("row_local", "folding.py:72-278 (alphafold-colabfold 2.3.13) = folding.py:74-280 (dl_binder_design cafa3853); final_act rows all_gather (moves_bytes) before output_projection"),
    "folding_multimer." + SM_SITE: ("row_local", "folding_multimer.py:221-371 (alphafold-colabfold 2.3.13); final_act rows all_gather (moves_bytes) before output_projection"),
}
SM_REQUIRED = {"affine": ("common_modules", "squared_difference"),          # what each rows body reads off its folding module, by frame flavour
               "rigid": ("common_modules", "geometry", "PointProjection")}
SM_HAZARDS = ("ipa_queries=my_rows(keys_values_targets=all)", "final_act_gathered_before_output_projection", "frames_sliced_by_field(no_recompute)")


def _install_structure(patches: Any, folding_modules: Dict[str, Any], lever: str) -> List[str]:
    """Rebind ``InvariantPointAttention.__call__`` of every given folding module (``{label: module}``; ``None`` values are skipped — a monomer-only tree has
    no ``folding_multimer``) INTO ``patches`` through :func:`haiku.rebind`, choosing the rows body by the stock signature's frame word: ``affine`` (the
    ``QuatAffine`` IPA of ``folding``) or ``rigid`` (the ``Rigid3Array`` IPA of ``folding_multimer``). The rebound method IS the stock body unless
    ``heads_rows()`` — replicated call sites keep the stock program bit for bit. Returns the installed site names (``<module>.InvariantPointAttention.__call__``)."""
    import inspect  # noqa: PLC0415
    sites: List[str] = []
    for label in sorted(folding_modules):
        F = folding_modules[label]
        if F is None:
            continue
        cls = getattr(F, "InvariantPointAttention", None)
        stock = _hk.stock_body(cls, "__call__")
        words = tuple(inspect.signature(stock).parameters)
        flavour = words[-1] if words[:-1] == ("self", "inputs_1d", "inputs_2d", "mask") and words[-1] in SM_REQUIRED else None
        sha = file_sha(F, lever)
        if sha not in PIN:                                                  # the bytes the rows body transcribes (PIN: the folding entries)
            raise MemLeverRefused(lever, f"install: {getattr(F, '__name__', F)!s} sha256 {sha[:12]} is not a transcribed tree ({'; '.join(sorted(SM_PIN.values()))})")
        make = _ipa_call_affine if flavour == "affine" else _ipa_call_rigid
        _hk.rebind(patches, cls, "__call__", make(stock, F), lever)
        sites.append(f"{str(getattr(F, '__name__', label)).rsplit('.', 1)[-1]}.{SM_SITE}")
    return sites


def _ipa_call_affine(stock, F):
    """``folding.InvariantPointAttention.__call__``: the stock body, or — inside the heads region — :func:`_ipa_rows_affine` on the pair row block."""
    def __call__(self, inputs_1d, inputs_2d, mask, affine):
        if not heads_rows():  # noqa: F821 — the heads-region flag of the trace state (defined with it, above the banner)
            return stock(self, inputs_1d, inputs_2d, mask, affine)
        return _ipa_rows_affine(self, inputs_1d, inputs_2d, mask, affine, F)
    return __call__


def _ipa_call_rigid(stock, F):
    """``folding_multimer.InvariantPointAttention.__call__``: the stock body, or — inside the heads region — :func:`_ipa_rows_rigid` on the pair row block."""
    def __call__(self, inputs_1d, inputs_2d, mask, rigid):
        if not heads_rows():  # noqa: F821
            return stock(self, inputs_1d, inputs_2d, mask, rigid)
        return _ipa_rows_rigid(self, inputs_1d, inputs_2d, mask, rigid, F)
    return __call__


def _sm_rows(inputs_1d, inputs_2d):
    """``(n_loc, start)``: the extent of this device's query rows (= the pair row block's, ``inputs_2d`` ``[N/P, N, c]``) and their global offset
    ``axis_index * n_loc``. A block that does not tile the ``N`` residues of ``inputs_1d`` under ``n_gpu`` is refused by name (the pair reaching the
    structure module is then not the row block)."""
    lever = str(_PLAN["lever"] or FAMILY_LEVER)
    n, n_loc, P = int(inputs_1d.shape[0]), int(inputs_2d.shape[0]), int(_PLAN["n_gpu"])
    if n_loc * P != n or int(inputs_2d.shape[1]) != n:
        raise MemLeverRefused(lever, f"structure module rows: inputs_2d {tuple(inputs_2d.shape)} is not this device's row block of the [{n}, {n}, c] pair under n_gpu={P}")
    return n_loc, _shard.axis_index(_PLAN["axis"]) * n_loc


def _ipa_rows_affine(self, inputs_1d, inputs_2d, mask, affine, F):
    """``folding.InvariantPointAttention.__call__`` (folding.py:99-278 alphafold-colabfold 2.3.13; :101-280 dl_binder_design cafa3853 — the same statements)
    with the QUERY residues restricted to my rows: ``inputs_1d`` ``[N, C]``, ``mask`` ``[N, 1]`` and ``affine`` (``QuatAffine`` over ``[N]``) replicated;
    ``inputs_2d`` = my pair rows ``[N/P, N, C']``. Parameter names and statement order are the stock's; ``num_residues`` is the target count, ``n_loc`` the
    query count where the stock has one ``num_residues`` for both."""
    import haiku as hk  # noqa: PLC0415 — the model's own dependencies, present wherever these classes are traced
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    common_modules, squared_difference = F.common_modules, F.squared_difference
    axis = _PLAN["axis"]
    num_residues, _ = inputs_1d.shape                                        # :99 — the TARGET residues (all)
    n_loc, start = _sm_rows(inputs_1d, inputs_2d)                            # my QUERY residues
    rows = lambda x: jax.lax.dynamic_slice_in_dim(x, start, n_loc, axis=0)   # noqa: E731
    inputs_1d_rows = rows(inputs_1d)                                         # [N/P, C]
    mask_rows = rows(mask)                                                   # [N/P, 1]
    affine_rows = affine.apply_tensor_fn(rows)                               # quaternion / translation / rotation fields of my rows (quat_affine.py apply_tensor_fn; nothing recomputed)

    num_head = self.config.num_head                                          # :102-111
    num_scalar_qk = self.config.num_scalar_qk
    num_point_qk = self.config.num_point_qk
    num_scalar_v = self.config.num_scalar_v
    num_point_v = self.config.num_point_v
    num_output = self.config.num_channel
    assert num_scalar_qk > 0
    assert num_point_qk > 0
    assert num_point_v > 0

    q_scalar = common_modules.Linear(num_head * num_scalar_qk, name='q_scalar')(inputs_1d_rows)          # :115-119 — my rows
    q_scalar = jnp.reshape(q_scalar, [n_loc, num_head, num_scalar_qk])

    kv_scalar = common_modules.Linear(num_head * (num_scalar_v + num_scalar_qk), name='kv_scalar')(inputs_1d)   # :123-129 — all residues
    kv_scalar = jnp.reshape(kv_scalar, [num_residues, num_head, num_scalar_v + num_scalar_qk])
    k_scalar, v_scalar = jnp.split(kv_scalar, [num_scalar_qk], axis=-1)

    q_point_local = common_modules.Linear(num_head * 3 * num_point_qk, name='q_point_local')(inputs_1d_rows)     # :135-144 — my rows, my frames
    q_point_local = jnp.split(q_point_local, 3, axis=-1)
    q_point_global = affine_rows.apply_to_point(q_point_local, extra_dims=1)
    q_point = [jnp.reshape(x, [n_loc, num_head, num_point_qk]) for x in q_point_global]

    kv_point_local = common_modules.Linear(num_head * 3 * (num_point_qk + num_point_v), name='kv_point_local')(inputs_1d)   # :151-166 — all residues, all frames
    kv_point_local = jnp.split(kv_point_local, 3, axis=-1)
    kv_point_global = affine.apply_to_point(kv_point_local, extra_dims=1)
    kv_point_global = [jnp.reshape(x, [num_residues, num_head, (num_point_qk + num_point_v)]) for x in kv_point_global]
    k_point, v_point = list(zip(*[jnp.split(x, [num_point_qk, ], axis=-1) for x in kv_point_global]))

    scalar_variance = max(num_scalar_qk, 1) * 1.                             # :171-189
    point_variance = max(num_point_qk, 1) * 9. / 2
    num_logit_terms = 3
    scalar_weights = np.sqrt(1.0 / (num_logit_terms * scalar_variance))
    point_weights = np.sqrt(1.0 / (num_logit_terms * point_variance))
    attention_2d_weights = np.sqrt(1.0 / (num_logit_terms))
    trainable_point_weights = jax.nn.softplus(hk.get_parameter('trainable_point_weights', shape=[num_head], init=hk.initializers.Constant(np.log(np.exp(1.) - 1.))))
    point_weights *= jnp.expand_dims(trainable_point_weights, axis=1)

    v_point = [jnp.swapaxes(x, -2, -3) for x in v_point]                     # :191-207 — [H, N, ·] targets; [H, N/P, ·] queries
    q_point = [jnp.swapaxes(x, -2, -3) for x in q_point]
    k_point = [jnp.swapaxes(x, -2, -3) for x in k_point]
    dist2 = [squared_difference(qx[:, :, None, :], kx[:, None, :, :]) for qx, kx in zip(q_point, k_point)]   # [H, N/P, N, P_qk]
    dist2 = sum(dist2)
    attn_qk_point = -0.5 * jnp.sum(point_weights[:, None, None, :] * dist2, axis=-1)

    v = jnp.swapaxes(v_scalar, -2, -3)
    q = jnp.swapaxes(scalar_weights * q_scalar, -2, -3)
    k = jnp.swapaxes(k_scalar, -2, -3)
    attn_qk_scalar = jnp.matmul(q, jnp.swapaxes(k, -2, -1))                  # [H, N/P, N]
    attn_logits = attn_qk_scalar + attn_qk_point

    attention_2d = common_modules.Linear(num_head, name='attention_2d')(inputs_2d)   # :209-215 — the pair read #1: MY rows [N/P, N, H]
    attention_2d = jnp.transpose(attention_2d, [2, 0, 1])
    attention_2d = attention_2d_weights * attention_2d
    attn_logits += attention_2d

    mask_2d = mask_rows * jnp.swapaxes(mask, -1, -2)                         # :217-218 — [N/P, N]
    attn_logits -= 1e5 * (1. - mask_2d)

    attn = jax.nn.softmax(attn_logits)                                       # :221 — complete over every target j per row
    result_scalar = jnp.matmul(attn, v)                                      # :224
    result_point_global = [jnp.sum(attn[:, :, :, None] * vx[:, None, :, :], axis=-2) for vx in v_point]   # :232-234
    result_scalar = jnp.swapaxes(result_scalar, -2, -3)                      # :237-240
    result_point_global = [jnp.swapaxes(x, -2, -3) for x in result_point_global]

    output_features = []                                                     # :244-259 — [N/P, ·] each
    result_scalar = jnp.reshape(result_scalar, [n_loc, num_head * num_scalar_v])
    output_features.append(result_scalar)
    result_point_global = [jnp.reshape(r, [n_loc, num_head * num_point_v]) for r in result_point_global]
    result_point_local = affine_rows.invert_point(result_point_global, extra_dims=1)   # my rows' frames
    output_features.extend(result_point_local)
    output_features.append(jnp.sqrt(self._dist_epsilon + jnp.square(result_point_local[0]) + jnp.square(result_point_local[1]) + jnp.square(result_point_local[2])))

    result_attention_over_2d = jnp.einsum('hij, ijc->ihc', attn, inputs_2d)  # :265-269 — the pair read #2: MY rows
    num_out = num_head * result_attention_over_2d.shape[-1]
    output_features.append(jnp.reshape(result_attention_over_2d, [n_loc, num_out]))

    final_init = 'zeros' if self._zero_initialize_last else 'linear'         # :271-278
    final_act = jnp.concatenate(output_features, axis=-1)                    # [N/P, F]
    final_act = _shard.gather(final_act, axis, dim=0)                        # THE collective of the site: [N, F] — everything downstream is the replicated stock program
    return common_modules.Linear(num_output, initializer=final_init, name='output_projection')(final_act)


def _ipa_rows_rigid(self, inputs_1d, inputs_2d, mask, rigid, F):
    """``folding_multimer.InvariantPointAttention.__call__`` (folding_multimer.py:255-371 alphafold-colabfold 2.3.13) with the QUERY residues restricted to my
    rows: ``inputs_1d`` ``[N, C]``, ``mask`` ``[N, 1]`` and ``rigid`` (``Rigid3Array`` over ``[N]``) replicated; ``inputs_2d`` = my pair rows ``[N/P, N, C']``.
    The logits are the stock's ``[q, k, h]`` layout with ``q`` = my rows; ``num_query_residues`` is ``n_loc``."""
    import haiku as hk  # noqa: PLC0415
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    common_modules, geometry, PointProjection = F.common_modules, F.geometry, F.PointProjection
    axis = _PLAN["axis"]
    n_loc, start = _sm_rows(inputs_1d, inputs_2d)
    rows = lambda x: jax.lax.dynamic_slice_in_dim(x, start, n_loc, axis=0)   # noqa: E731
    inputs_1d_rows = rows(inputs_1d)                                         # [N/P, C]
    mask_rows = rows(mask)                                                   # [N/P, 1]
    rigid_rows = jax.tree_util.tree_map(rows, rigid)                         # the Rigid3Array pytree's fields of my rows

    num_head = self.config.num_head                                          # :255-257
    attn_logits = 0.

    num_point_qk = self.config.num_point_qk                                  # :259-274
    point_variance = max(num_point_qk, 1) * 9. / 2
    point_weights = np.sqrt(1.0 / point_variance)
    softplus = lambda x: jnp.logaddexp(x, 0.)                                # noqa: E731 — :265 spells the zero jnp.zeros_like(x); x is the PARAMETER here, and a
    # *_like of a value captured from OUTSIDE the manual region inherits its auto-mesh sharding, which jax (0.5) refuses inside shard_map (full_like →
    # broadcast_in_dim); the weak scalar zero is the same elementwise logaddexp (bit-exact — the 1-device-mesh case of the unit tests asserts it)
    raw_point_weights = hk.get_parameter('trainable_point_weights', shape=[num_head], init=hk.initializers.Constant(np.log(np.exp(1.) - 1.)))
    trainable_point_weights = softplus(raw_point_weights)
    point_weights *= trainable_point_weights
    q_point = PointProjection([num_head, num_point_qk], self.global_config, name='q_point_projection')(inputs_1d_rows, rigid_rows)   # :275-278 — my rows, my frames
    k_point = PointProjection([num_head, num_point_qk], self.global_config, name='k_point_projection')(inputs_1d, rigid)             # :280-283 — all residues

    dist2 = geometry.square_euclidean_distance(q_point[:, None, :, :], k_point[None, :, :, :], epsilon=0.)   # :285-288 — [N/P, N, H, P_qk]
    attn_qk_point = -0.5 * jnp.sum(point_weights[:, None] * dist2, axis=-1)
    attn_logits += attn_qk_point

    num_scalar_qk = self.config.num_scalar_qk                                # :290-306
    scalar_variance = max(num_scalar_qk, 1) * 1.
    scalar_weights = np.sqrt(1.0 / scalar_variance)
    q_scalar = common_modules.Linear([num_head, num_scalar_qk], use_bias=False, name='q_scalar_projection')(inputs_1d_rows)   # my rows
    k_scalar = common_modules.Linear([num_head, num_scalar_qk], use_bias=False, name='k_scalar_projection')(inputs_1d)        # all residues
    q_scalar *= scalar_weights
    attn_logits += jnp.einsum('qhc,khc->qkh', q_scalar, k_scalar)            # [N/P, N, H]

    attention_2d = common_modules.Linear(num_head, name='attention_2d')(inputs_2d)   # :308-310 — the pair read #1: MY rows [N/P, N, H]
    attn_logits += attention_2d

    mask_2d = mask_rows * jnp.swapaxes(mask, -1, -2)                         # :312-313 — [N/P, N]
    attn_logits -= 1e5 * (1. - mask_2d[..., None])

    attn_logits *= np.sqrt(1. / 3)                                           # :315-316
    attn = jax.nn.softmax(attn_logits, axis=-2)                              # complete over every target k per row

    num_scalar_v = self.config.num_scalar_v                                  # :318-326
    v_scalar = common_modules.Linear([num_head, num_scalar_v], use_bias=False, name='v_scalar_projection')(inputs_1d)          # all residues
    result_scalar = jnp.einsum('qkh, khc->qhc', attn, v_scalar)

    num_point_v = self.config.num_point_v                                    # :328-335
    v_point = PointProjection([num_head, num_point_v], self.global_config, name='v_point_projection')(inputs_1d, rigid)         # all residues
    result_point_global = jax.tree_util.tree_map(lambda x: jnp.sum(attn[..., None] * x, axis=-3), v_point[None])

    output_features = []                                                     # :339-355 — [N/P, ·] each
    num_query_residues, _ = inputs_1d_rows.shape
    flat_shape = [num_query_residues, -1]
    result_scalar = jnp.reshape(result_scalar, flat_shape)
    output_features.append(result_scalar)
    result_point_global = jax.tree_util.tree_map(lambda r: jnp.reshape(r, flat_shape), result_point_global)
    result_point_local = rigid_rows[..., None].apply_inverse_to_point(result_point_global)   # my rows' frames
    output_features.extend([result_point_local.x, result_point_local.y, result_point_local.z])
    point_norms = result_point_local.norm(self._dist_epsilon)
    output_features.append(point_norms)

    result_attention_over_2d = jnp.einsum('ijh, ijc->ihc', attn, inputs_2d)  # :361-362 — the pair read #2: MY rows
    output_features.append(jnp.reshape(result_attention_over_2d, flat_shape))

    final_init = 'zeros' if self._zero_initialize_last else 'linear'         # :364-371
    final_act = jnp.concatenate(output_features, axis=-1)                    # [N/P, F]
    final_act = _shard.gather(final_act, axis, dim=0)                        # THE collective of the site: [N, F]
    return common_modules.Linear(self.config.num_channel, initializer=final_init, name='output_projection')(final_act)
