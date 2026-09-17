"""Modes: what a package mode exports, on which tree state, and which FPF arm it applies.

The patched files (``opt/forward/rf3_xattempt_addon/``) have no mode table and no hook: their levers are read from the environment
when ``rf3.graph_flags`` is imported (``patched/rf3/graph_flags.py``) — ``RF3_CUDAGRAPH=1 RF3_HOIST=1``. The FPF adapter
(``opt/forward/rf3_fpf_trimul_addon/``) is a runtime re-binding of the trunk applied in-process, one call per process,
``apply_arm("<arm>")`` (``rf3fpf/fpf_rf3_adapter.py``); the ``fast`` mode applies ``FPF_ARM``, the ``exact`` mode ``FPF_ARM_TG_SAPB``.
A package mode is therefore an environment
row exported at the process boundary on an interpreter whose tree is in the ``patched`` state (tree.py), plus the FPF arm the
adapter applies once the upstream model package is imported (stack.py) — never a code path of this package.
This module is the ONE table of modes; ``configs/*.env`` carry deployment parameters only.

Package modes (``KIT_MODES``; ``MODES`` / ``DEFAULT_MODE`` are the one list and the one default every command reads; ``off`` = stock = upstream as
shipped: default ≡ stock, no non-shipped setting):
  fast    the ``exact`` row's switches plus the FPF add-on's fast arm (``FPF_ARM``: the TriMul provider's fast tier, fused triangle
          attention, fused Triton pair transition, the Triton attention-pair-bias kernel (``apb``), fused residual adds, the MSA
          module's fused pair statements (``msa``), the pairformer-stack CUDA graph within tgbudget.py's token budget for these
          kernels (``tg``), the template-track TriMul on the provider's exact row (``xmul.eager``) and every LayerNorm on the shared
          core's LN provider (``xln``); ``@L1.warm`` = the kit levers on, the same state the row exports, with the roll-out warm-up)
          plus the package levers of ``KIT_MODES["fast"].kit_levers``: ``mkdit`` (mkdit.py: the diffusion module's token transformer on
          the carried MK-DiT megakernel), ``confhoist`` / ``confln`` (confhoist.py), ``hostlean``, ``prefetch``, ``awrite``; in the
          big composition ``dtk`` (dtk.py: the same blocks' token-level pair-biased attention on the shared core's flash kernel)
          takes mkdit's place. Tier 2: same numerics class as stock, not bitwise (the add-on's README; dtk: SDPA's class).
  exact   ``RF3_CUDAGRAPH=1 RF3_HOIST=1``: the sampler CUDA graph (the patched files' ``[rf3_cudagraph]`` edits) and the step-invariant
          conditioning cache of the add-on, both on (the kit's capture-safe op rewrites follow the graph mode: ``graph_flags.GRAPH_SAFE_OPS``);
          plus the add-on's graph arm over STOCK kernels (``FPF_ARM_TG_SAPB``: ``tg+sapb`` = the pairformer stack replayed under
          one CUDA graph per token count within the token budget of tgbudget.py, the attention-pair-bias in its graph-capturable
          stock form; ``xatt`` / ``xmul.eager`` / ``xln`` / ``smsa`` = the exact-tier triangle attention, TriMul, LayerNorm and MSA
          pair-weighted-averaging rows of the shared core's providers, each bitwise to the stock op or the stock op by name)
          plus the package levers ``xtr`` (pf.py: the SwiGLU transitions on the shared core's fused pair-track transition in its
          exact construction, served cells only, named routes otherwise), ``confhoist``, ``hostlean``, ``prefetch``, ``awrite``;
          bitwise equal to stock when both run a deterministic ``scatter_mean`` (stock's own uses CUDA atomics).
  big   the memory mode, named by its guarantee (folds bigger inputs; fast-class numerics, never bitwise): the ``fast`` row by
          reference + the memory levers of ``big.LEVERS`` composed through ``opt_core.mem`` (``big_mode``; the FPF arm is fast's
          minus the components a memory lever disengages by property, ``big.DISENGAGED`` — the fused attention / transition
          kernels and the trunk graph; the package levers fast's with ``big.REPLACED_KIT`` applied, minus ``big.DISENGAGED_KIT`` — ``dtk`` in mkdit's place), under the memory
          policy of the kit rows.
  off     stock: the upstream CLI on the pristine interpreter in a clean subprocess (stock_fold.py); no lever variable set.
The default mode is ``fast`` (``DEFAULT_MODE``: ``--mode`` omitted); ``exact`` is the mode that reproduces stock's outputs. ``fast``'s
per-shape costs (kernel compilation, graph capture) are paid on the first item of a shape and amortised over later items or a warm
cache (README.md: for a single cold item use ``--mode off|exact``); the trunk graph of ``exact`` likewise pays from the second
same-shape item of a process.
With ``ROSETTAFOLD3_OPT`` unset the hook installs nothing and the package exports nothing (_autoload.py) — ``off`` from this tree's
side; what then runs is what the interpreter carries (stock on the pristine one; on the patched one the kit's own defaults,
``graph_flags.CUDAGRAPH_MODE``, and no FPF arm: the adapter is never imported unless a mode names an arm).

The kit files read no ``RF3_*`` name besides the two switches (the capture-safe op rewrites follow the graph mode and the capture warm-up
count is fixed in ``graph_flags.py``); the ``replay`` value of ``RF3_CUDAGRAPH`` is no mode's (the row-sharded line
sets it per rank at run time, ``graph_flags.set_mode``). The FPF adapter's one knob (``FPF_KNOBS``: ``FPF_RF3_TG_MAX``, the trunk-graph cache size) likewise stays at
the adapter's own default and is never exported by a mode (the memory policy, mem.py, writes it on rows whose arm carries ``tg``).

``flag_table()`` reads every ``os.environ.get("<PREFIX>...", default)`` of a carried file by AST (no torch, no import), so the switch
and knob defaults above are checked against the kit files (``graph_flags.py``, ``fpf_rf3_adapter.py``), not remembered (``tests/test_modes.py``).
``fpf_components()`` is the one parser of an arm string (the adapter's grammar, ``fpf_rf3_adapter.py apply_arm``): the components
and the ``@L`` lever state, which must agree with the row's switches (``resolve`` refuses a disagreeing pair by name).
"""
from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

ENV = "ROSETTAFOLD3_OPT"                                  # the package's mode variable (outside the kit's RF3_* namespace)
MODES: Tuple[str, ...] = ("off", "exact", "fast", "big")
DEFAULT_MODE = "fast"                                     # the package default (--mode omitted); `exact` reproduces stock's outputs (bitwise); README.md names the single-cold-item
                                                          # trade-off (--mode off|exact)
KIT_RELDIR = os.path.join("forward", "rf3_xattempt_addon")   # the add-on, relative to opt/
FPF_RELDIR = os.path.join("forward", "rf3_fpf_trimul_addon")   # the FPF add-on, relative to opt/
GRAPH_FLAGS_RELPATH = os.path.join("patched", "rf3", "graph_flags.py")   # the add-on's switch table
FPF_ADAPTER_RELPATH = os.path.join("rf3fpf", "fpf_rf3_adapter.py")   # the FPF add-on's adapter (its arm grammar and knob table)
FPF_SYS_PATH = ("rf3fpf",)                               # the add-on directory a row with an arm imports from (its adapter); its three kernel modules
                                                          # are opt_core.kernels' (stack.KERNELS, routed by name) — the kit carries no copy of them
FPF_RUNTIME_MEMBERS = (                                   # the add-on files a row with an arm reads at run time (stack.fpf_present asserts them)
    "rf3fpf/fpf_rf3_adapter.py",)                         # (the TriMul launch cells are the shared core's one table, opt_core/kernels/fpf_trimul_v4/table.json — no kit copy)

SWITCHES: Tuple[str, ...] = ("RF3_CUDAGRAPH", "RF3_HOIST")            # the lever switches a mode exports
FPF_KNOBS: Dict[str, str] = {"FPF_RF3_TG_MAX": "6"}                  # the adapter's trunk-graph cache size (graphs held per process); mem.py's policy writes it, no mode does
FPF_ENV_PREFIXES: Tuple[str, ...] = ("FPF_RF3_", "FPF_TRIMUL_")      # the add-on's namespaces; a mode exports none of them except the routed fast row's
                                                                     # absent from the stock arm
FPF_ARM = "fast.fast+gflash+ttr+apb.fast+res+tg+xmul.eager+xln+msa@L1.warm"      # the fast modes' arm: the TriMul provider's FAST tier word (fast.fast: opt_core.kernels.trimul's cell per shape and card;
#   a row sub-word fast.<row> pins one row and is never a mode's) + the fused triangle-attention (gflash), transition (ttr) and
#   attention-pair-bias (apb.fast) kernels with fused residual adds (res) + the MSA module's fused pair statements (msa) + the
#   pairformer-stack CUDA graph within tgbudget.py's budget (tg) + the template-track TriMul on the provider's exact row in eager
#   (xmul.eager) + every LayerNorm on the shared core's LN provider (xln); @L1.warm = the kit levers on, with the roll-out warm-up
FPF_ARM_TG_SAPB = "tg+sapb+xatt+xmul.eager+xln+smsa@L1.warm"                  # the exact mode's arm: the pairformer-stack CUDA graph on stock kernels (tg+sapb) + the exact-tier triangle attention (xatt), TriMul (xmul.eager), LayerNorm (xln) and MSA pair-weighted averaging (smsa) rows — each bitwise to stock or the stock op by name
# the adapter's arm grammar (fpf_rf3_adapter.py apply_arm): trimul choice, then independent components
FPF_TRIMUL = ("stock", "fast")
FPF_COMPONENTS = ("gflash", "ttr", "apb", "sapb", "tg", "dattn", "res", "xmul", "xln", "xatt", "msa", "smsa")
FPF_SUBWORDS = {"fast": ("v4", "tmk3_fast", "tmk3_exact", "tx_sm90a", "esm_v5_fwd", "esm_v61", "fast", "big"),   # <head|component>.<word>: the ROW the trimul head word / a component serves — each tuple IS
                "xmul": ("exact", "eager", "tmk3_exact", "esm_v5_fwd", "esm_v61", "cueq"),                             # that component's COMPONENT_WORDS in its rows module (FPF_SUBWORD_MODULES; tests/test_modes reads
                "gflash": ("card", "flash", "k2b", "k2", "cuda_sm90a", "triattn_native", "fast"), "xatt": ("exact", "exact_headsplit", "cueq"), # them back by AST)
                "apb": ("stmt", "sdpa", "fpf_apb", "apb_attn", "fast", "big"),
                "xln": ("exact", "fast", "big", "exactln", "exactln:triton", "exactln:widen", "fastln", "ln_rows", "aten"),
                "msa": ("card", "pwa", "opm", "both")}
FPF_SUBWORD_MODULES = {"fast": "rf3fpf/fpf_rf3_trimul_rows.py", "xmul": "rf3fpf/fpf_rf3_trimul_rows.py", "gflash": "rf3fpf/fpf_rf3_triattn.py", "xatt": "rf3fpf/fpf_rf3_triattn.py",
                       "apb": "rf3fpf/fpf_rf3_apb_rows.py", "xln": "rf3fpf/fpf_rf3_ln_rows.py", "msa": "rf3fpf/fpf_rf3_msa_rows.py"}   # where each component's words live (relative to the FPF add-on home)
FPF_LEVER_STATE = {"L1": True}                              # `@L1` = set_kit_levers(): RF3_CUDAGRAPH + RF3_HOIST (+ GRAPH_SAFE_OPS) on
FPF_LEVER_SUBS: Tuple[str, ...] = ("warm",)                  # the lever step's sub-steps (adapter LEVER_SUBS): `@L1.warm` = the roll-out's eager warm-up once per process (graph_flags.set_levers)
LEVERS_OF_SUB = {"warm": ("warm",)}                          # registry names per lever sub-step (registry.py)
SUB_LEVERS: Tuple[str, ...] = ("warm",)


@dataclass(frozen=True)
class KitMode:
    name: str
    switches: Dict[str, str]          # the environment row exported at the process boundary ({} for off)
    tree_state: str                   # tree.py state the interpreter must be in
    doc: str
    fpf_arm: Optional[str] = None     # the FPF add-on arm applied in-process (adapter grammar), None = the adapter is never imported
    kit_levers: Tuple[str, ...] = ()  # the package's own in-process levers the row turns on after the arm (registry names of kit rosettafold3_opt, e.g. dtk)


TP_ENV: Dict[str, str] = {"ROWPAIR_PARK_ZINIT": "1", "ROWPAIR_MSA_HOST": "rank0", "ROWPAIR_FREE_ZTRUNK": "1", "ROWPAIR_CONF_PARK_ZTRUNK": "1", "ROWPAIR_TRANSPOSE_INPLACE": "1",
                           "ROWPAIR_DIFF_WORK_GB": "2",                                       # the row-sharded line's memory levers at `--n_gpu P > 1` (the names opt_core.mem.rowpair reads,
                           "ROWPAIR_RANK_THREADS": "auto",        # API.md "Levers read from the environment"); placement only: values bitwise, the device high-water lower;
                           "ROWPAIR_HOST_SLAB": "lease",          # exported to the rank processes by fold.kit_env (the rank's census prints the value in force):
                           "ROWPAIR_DIFF_BIAS": "ln_proj"}        # ROWPAIR_DIFF_BIAS ln_proj: the DiT blocks' pair-bias rows written LN + projection -> head-major [H, rows, N] in one pass (kernels.ln_proj; bf16 dot, fp32 accumulate; census dit_bias=ln_proj:<n>);
                                                                  # ROWPAIR_PARK_ZINIT: the z_init pair shard parked on pinned host between recycles (trunk.ShardPark; census park_z_init);
                                                                  # ROWPAIR_MSA_HOST rank0: the raw MSA stack [n_recycle, S, I, c] kept on rank 0's pinned host, the cycle's rows [S, I, c] moved per recycle and
                                                                  #   broadcast to ranks > 0 (census msa_host=rank0, msa_host_rows);
                                                                  # ROWPAIR_FREE_ZTRUNK / ROWPAIR_CONF_PARK_ZTRUNK: the trunk shard under the confidence head embedded in place (fp32 shard, last use) or parked
                                                                  #   (host copy, device storage released before the per-sample fp32 pair input) per pass — heads.ZTrunkPlan, census conf_ztrunk;
                                                                  # ROWPAIR_TRANSPOSE_INPLACE: every pair block's ending-orientation transpose swaps blocks inside the shard's own storage (ring.transpose_shard_inplace_
                                                                  #   via pairstack.pair_block_, census pairstack_transpose=inplace: no second shard-sized buffer in any pair block);
                                                                  # ROWPAIR_DIFF_WORK_GB: the roll-out's row-block transient budget, 2 GB per rank (the DiT pair-bias cache is the kit's fit decision, rowpair.TP_DIT_BIAS_CACHE);
                                                                  # ROWPAIR_RANK_THREADS auto: per-rank CPU threads = cores / ranks (census rank_threads);
                                                                  # ROWPAIR_HOST_SLAB lease: the u-sized pinned host copies (z_init park, z_trunk park, tri-mult row mirror) share ONE leased slab per rank (tp_conf retires
                                                                  #   the z_trunk park at the last pass's embed).


def tp_env() -> Dict[str, str]:
    """The row-sharded line's lever row (``TP_ENV``), as exported to the rank processes."""
    return dict(TP_ENV)


def reach_env(n_gpu: int = 1) -> Dict[str, str]:
    """The environment a pass owes its GPU count: ``tp_env()`` at ``n_gpu > 1`` (the row-sharded line's levers), else nothing."""
    return tp_env() if int(n_gpu) > 1 else {}


KIT_MODES: Dict[str, KitMode] = {
    "fast": KitMode("fast", {"RF3_CUDAGRAPH": "1", "RF3_HOIST": "1"}, "patched",
                    "the patched files' RF3_CUDAGRAPH=1 RF3_HOIST=1 + the FPF adapter's component stack with the compile-free attention-pair-bias kernel (apb) "
                    "(the trunk kernels, without the pairformer-stack CUDA graph) + the package's mkdit lever (the diffusion module's 24-block token "
                    "transformer on the carried MK-DiT megakernel, one launch per sample per denoiser step), confhoist (the confidence head's sample-invariant "
                    "prologue computed once per item) and confln (that prologue's whole-tensor layer norms by a grid-wide var_mean)", fpf_arm=FPF_ARM, kit_levers=("mkdit", "confhoist", "confln", "hostlean", "prefetch", "awrite")),
    "exact": KitMode("exact", {"RF3_CUDAGRAPH": "1", "RF3_HOIST": "1"}, "patched",
                     "the patched files' RF3_CUDAGRAPH=1 RF3_HOIST=1 + the FPF adapter's graph arm on STOCK kernels (tg+sapb: the 48-block pairformer stack replayed "
                     "under one CUDA graph per token count, the attention-pair-bias in its graph-capturable stock form) + the package's xtr lever "
                     "(the SwiGLU transitions on the shared core's fused pair-track transition in its exact construction) and confhoist lever (the confidence "
                     "head's sample-invariant prologue computed once per item: upstream's statements, executed once) — bitwise to stock under "
                     "the deterministic recipe", fpf_arm=FPF_ARM_TG_SAPB, kit_levers=("xtr", "confhoist", "hostlean", "prefetch", "awrite")),
    "off": KitMode("off", {}, "stock", "the upstream CLI on the pristine interpreter (stock_fold.py)"),
}


def big_mode(base: str) -> KitMode:
    """The big row, PARAMETRIC on the base: the named base's row switches by reference + the base's FPF arm with the components
    the memory levers disengage by lever property removed (``big.fpf_arm_for``: the fused triangle attention / transition kernels whose
    sites triatt_chunk / transition_chunk take, the trunk CUDA graph as a memory holder), minus the CUDA-graph switches
    (``big.DISENGAGED_SWITCHES``) + the base's package levers with ``big.REPLACED_KIT`` applied, minus ``big.DISENGAGED_KIT``.
    ``big.BASE`` names the base (``fast``); ``big_mode("exact")`` is the same composition on the exact row, never a second code path."""
    if base not in ("fast", "exact"):
        raise ValueError(f"big composes on the fast or exact row, not {base!r}")
    from . import big as _big
    km = KIT_MODES[base]
    arm, gone = _big.fpf_arm_for(km.fpf_arm)
    switches = dict(km.switches)
    switches.update({sw: "0" for sw in _big.DISENGAGED_SWITCHES})                              # the memory row runs no CUDA graph (big.DISENGAGED_SWITCHES): RF3_CUDAGRAPH exported 0, RF3_HOIST as the base
    kit_levers = tuple(lv for lv in (_big.REPLACED_KIT.get(x, x) for x in km.kit_levers) if lv not in _big.DISENGAGED_KIT)   # the base's package levers, a memory holder replaced
                                                                                                 # by the lever of its site that holds none (mkdit -> dtk), minus the ones a memory lever disengages by property
    return KitMode("big", switches, km.tree_state, f"the {base} row by reference minus its CUDA graphs ({','.join(f'{k}=0' for k in _big.DISENGAGED_SWITCHES)}) + the memory levers "
                   f"(big.LEVERS); FPF arm {arm!r} (the base's {km.fpf_arm!r} minus {sorted(gone) or 'nothing'}: disengaged by lever property, the lever step with the graph); "
                   f"package levers {kit_levers}",
                   fpf_arm=arm, kit_levers=kit_levers)


BIG_SWITCHES: Dict[str, str] = {**KIT_MODES["exact"].switches, "RF3_CUDAGRAPH": "0"}   # the memory row runs no CUDA graph (big.DISENGAGED_SWITCHES; big_mode derives the same row from the base)
KIT_MODES["big"] = KitMode("big", BIG_SWITCHES, "patched",
                             "the fast row + the memory levers (big.LEVERS) — the row switches are the bases' minus the CUDA graph (RF3_CUDAGRAPH=0 RF3_HOIST=1: the memory row runs no CUDA graph); the FPF arm is "
                             "fast's minus the components the memory levers disengage by property (kit_mode('big') = big_mode(big.BASE) resolves it; the record cites the APPLIED arm)", fpf_arm=None)
assert KIT_MODES["fast"].switches == KIT_MODES["exact"].switches, "the big row assumes fast and exact share the switch row (modes.py KIT_MODES)"
KIT_LEVERS_ON_FPF_SEAM: Tuple[str, ...] = ("dtk",)  # the kit levers that install through the FPF add-on's source seam (dtk: the adapter's _DATTN_BLOCK) and so need an arm
KIT_LEVERS: Tuple[str, ...] = ("dtk", "xtr", "mkdit", "confhoist", "confln", "hostlean", "prefetch", "awrite")  # the package's in-process levers a row may name in kit_levers (registry kit = rosettafold3_opt; mem is a policy, not a row lever)
DIT_LEVERS: Tuple[str, ...] = ("mkdit", "dtk")           # the diffusion-transformer levers: mkdit replaces the token transformer's blocks whole, dtk (and the add-on's dattn component) the
                                                          # attention inside them — a row names at most one (test_modes locks it; mkdit.enable refuses by name otherwise)
LEVERS_OF_SWITCH = {"RF3_CUDAGRAPH": ("graph", "graph_safe_ops"), "RF3_HOIST": ("hoist",)}   # registry names per switch (registry.py)
# registry names per FPF arm component (registry.py); the trimul choice `fast` is a component too, `stock` trimul names no house lever
LEVERS_OF_FPF = {"fast": ("fpf_trimul",), "gflash": ("fpf_gflash",), "ttr": ("fpf_ttr",), "apb": ("fpf_apb",), "sapb": ("fpf_sapb",),
                 "tg": ("fpf_tg",), "res": ("fpf_res",), "dattn": ("fpf_dattn",), "xmul": ("fpf_xmul",), "xln": ("fpf_xln",), "xatt": ("fpf_xatt",), "msa": ("fpf_msa",), "smsa": ("fpf_smsa",)}
FPF_LEVERS: Tuple[str, ...] = ("fpf_trimul", "fpf_gflash", "fpf_ttr", "fpf_apb", "fpf_sapb", "fpf_tg", "fpf_res", "fpf_dattn", "fpf_xmul", "fpf_xln", "fpf_xatt", "fpf_msa", "fpf_smsa")


# rf3's two cuEquivariance call sites (triangle attention, triangle multiplication) and, per op, the FPF arm components that replace
# that op's call sites — the memory levers read it to leave a site alone when the arm owns it (levers._arm_engages); a row whose arm
# names such a component legitimately never calls the op in a fold process.
CUEQ_OPS: Tuple[str, ...] = ("cueq_triattn", "cueq_trimul")
FPF_TAKES_CUEQ: Dict[str, Tuple[str, ...]] = {"cueq_triattn": ("gflash", "xatt"),   # xatt: the exact tier's statement served by the provider's exact row (the stock op by name for the template track and S <= its threshold); the fused triangle-attention component (registry fpf_gflash)
                                              "cueq_trimul": ("fast", "xmul")}      # the adapter's trimul choice other than `stock` (registry fpf_trimul)


@dataclass
class Resolution:
    mode: str                          # what was asked
    row: str                           # the resolved row name (== mode for a house mode)
    is_house_mode: bool
    switches: Dict[str, str]
    tree_state: str
    levers: List[str]                  # registry names the row turns on
    levers_off: List[str]              # registry names the row leaves at stock
    notes: List[str] = field(default_factory=list)
    fpf_arm: Optional[str] = None      # the FPF arm the row applies in-process (None: the adapter is never imported)
    fpf_components: List[str] = field(default_factory=list)   # the arm's components (adapter grammar), lever suffix stripped
    fpf_lever_state: Optional[bool] = None   # the arm's @L step (True = @L1), None when the arm names none
    fpf_knobs: Dict[str, str] = field(default_factory=dict)   # the adapter's own knob defaults (from the adapter file when available)
    kit_levers: List[str] = field(default_factory=list)      # the package's own in-process levers the row names (KIT_LEVERS)
    withheld: List[str] = field(default_factory=list)        # registry names MODEL_OPT_LEVERS_OFF removed from this row (leversoff.py; [] = the mode as it is)
    absent: List[str] = field(default_factory=list)          # MODEL_OPT_LEVERS_OFF words this kit has no lever for as a class (leversoff.ABSENT: compile) — accepted, reported <name>=none

    @property
    def interpreter(self) -> str:
        return "stock" if self.row == "off" else "opt"


def fpf_components(arm: str) -> Tuple[List[str], Optional[bool]]:
    """The adapter's grammar (``apply_arm``): ``<trimul>[+component...][@L1]`` -> (components in order, the lever state or None).
    A component outside the adapter's vocabulary is refused by name (the adapter would raise the same)."""
    body, _, lv = arm.partition("@")
    lever = None
    if lv:
        step, *subs = lv.split(".")
        if step not in FPF_LEVER_STATE:
            raise ValueError(f"FPF arm {arm!r}: lever suffix must be one of {tuple(FPF_LEVER_STATE)} (+ sub-steps {FPF_LEVER_SUBS})")
        bad_subs = [s for s in subs if s not in FPF_LEVER_SUBS]
        if bad_subs:
            raise ValueError(f"FPF arm {arm!r}: unknown lever sub-step(s) {bad_subs} (adapter grammar: @L1[.{'][.'.join(FPF_LEVER_SUBS)}])")
        lever = FPF_LEVER_STATE[step]
    parts = [p for p in body.split("+") if p]
    if not parts:
        parts = ["stock"]                                  # the adapter's own default when the body is empty (apply_arm)
    bad = [p for p in parts if p.partition(".")[0] not in FPF_TRIMUL + FPF_COMPONENTS
           or (p.partition(".")[1] and p.partition(".")[2] not in FPF_SUBWORDS.get(p.partition(".")[0], ()))]
    if bad:
        raise ValueError(f"FPF arm {arm!r}: unknown component(s) {bad} (adapter grammar: {FPF_TRIMUL + FPF_COMPONENTS})")
    return parts, lever


def fpf_lever_subs(arm: Optional[str]) -> List[str]:
    """The lever step's sub-steps an arm names, in order (``…@L1.warm`` -> ['warm']; none without a lever step)."""
    if not arm or "@" not in arm:
        return []
    fpf_components(arm)                                    # validates step and sub-steps by name
    return arm.partition("@")[2].split(".")[1:]


def fpf_levers_of(arm: Optional[str]) -> List[str]:
    """Registry names the arm turns on, in arm order (``fast`` trimul = the FPF-FAST kernel; ``stock``/``exact`` trimul name none), then
    the lever sub-steps' (``@L1.warm`` -> ``warm``)."""
    if not arm:
        return []
    parts, _ = fpf_components(arm)
    out: List[str] = []
    for p in parts:
        for name in LEVERS_OF_FPF.get(p.partition(".")[0], ()):
            if name not in out:
                out.append(name)
    for s in fpf_lever_subs(arm):
        for name in LEVERS_OF_SUB.get(s, ()):
            if name not in out:
                out.append(name)
    return out


def check_mode(mode: Optional[str]) -> str:
    m = (mode or "").strip().lower() or DEFAULT_MODE
    if m not in MODES:
        raise ValueError(f"unknown mode {mode!r} (expected {'|'.join(MODES)})")
    return m


def kit_mode(name: str) -> KitMode:
    if name == "big":
        from . import big as _big                       # the base is big.BASE (fast; parametric)
        return big_mode(_big.BASE)
    if name in KIT_MODES:
        return KIT_MODES[name]
    raise ValueError(f"unknown mode {name!r} (modes: {'|'.join(MODES)})")


def levers_of(switches: Dict[str, str]) -> Tuple[List[str], List[str]]:
    on, off = [], []
    for sw in SWITCHES:
        names = LEVERS_OF_SWITCH[sw]
        (on if switches.get(sw) == "1" else off).extend(names)
    return on, off


def resolve(mode: str, kit_home: Optional[str] = None, fpf_home: Optional[str] = None) -> Resolution:
    """The one resolver: a package mode -> its environment row, tree state, levers and FPF arm."""
    km = kit_mode(mode)
    on, off = levers_of(km.switches)
    notes: List[str] = []
    if kit_home:
        gf = os.path.join(kit_home, GRAPH_FLAGS_RELPATH)
        if os.path.exists(gf):
            table = flag_table(gf)
            missing = [s for s in SWITCHES if s not in table]
            if missing:
                notes.append(f"switches {missing} are not read by {GRAPH_FLAGS_RELPATH} (the kit file in the tree differs from the shipped kit)")
        else:
            notes.append(f"{GRAPH_FLAGS_RELPATH} not found under {kit_home}: the switches cannot be checked against the kit file")
    components: List[str] = []
    fpf_knobs: Dict[str, str] = {}
    lever_state: Optional[bool] = None
    if km.fpf_arm:
        components, lever_state = fpf_components(km.fpf_arm)
        lever = lever_state
        want = all(km.switches.get(sw) == "1" for sw in SWITCHES)
        if lever is not None and lever != want:
            raise ValueError(f"row {km.name!r}: FPF arm {km.fpf_arm!r} sets the kit levers {'on' if lever else 'off'} but the row exports "
                             f"{describe_line_of(km.switches)} (the adapter's set_kit_levers would contradict the exported row)")
        on = on + [lv for lv in fpf_levers_of(km.fpf_arm) if lv not in on]
        fpf_knobs = dict(FPF_KNOBS)
        if fpf_home:
            ad = os.path.join(fpf_home, FPF_ADAPTER_RELPATH)
            if os.path.exists(ad):
                table = flag_table(ad)
                fpf_knobs = {k: table.get(k, v) for k, v in FPF_KNOBS.items()}
                gone = [k for k in FPF_KNOBS if k not in table]
                if gone:
                    notes.append(f"FPF knobs {gone} are not read by {FPF_ADAPTER_RELPATH} (the adapter in the tree differs from the shipped add-on)")
            else:
                notes.append(f"{FPF_ADAPTER_RELPATH} not found under {fpf_home}: FPF knob defaults are the module's literals")
    for lv in km.kit_levers:
        if lv not in KIT_LEVERS:
            raise ValueError(f"row {km.name!r}: kit lever {lv!r} is not one of {KIT_LEVERS}")
        if lv in KIT_LEVERS_ON_FPF_SEAM and not km.fpf_arm:
            raise ValueError(f"row {km.name!r}: kit lever {lv!r} installs through the FPF add-on's seam and needs an arm")
        if lv == "dtk" and "dattn" in components:
            raise ValueError(f"row {km.name!r}: dtk and the arm's dattn replace the same attention block — name one of them")
    on = on + [lv for lv in km.kit_levers if lv not in on]
    off = off + [lv for lv in FPF_LEVERS + SUB_LEVERS + KIT_LEVERS if lv not in on]      # every FPF / sub-step / kit lever the row leaves at stock
    res = Resolution(mode=mode, row=km.name, is_house_mode=km.name in KIT_MODES, switches=dict(km.switches), tree_state=km.tree_state,
                     levers=on, levers_off=off, notes=notes, fpf_arm=km.fpf_arm, fpf_components=components, fpf_knobs=fpf_knobs,
                     fpf_lever_state=lever_state, kit_levers=list(km.kit_levers))
    from . import leversoff as _leversoff                                     # MODEL_OPT_LEVERS_OFF: the named levers leave the row here, the one resolver — unset: res unchanged
    return _leversoff.withhold(res)


def describe_line_of(switches: Dict[str, str]) -> str:
    return ",".join(f"{k}={v}" for k, v in switches.items()) or "none"


def describe_line(res: Resolution) -> str:
    """The row's spelling for the activation line: ``RF3_CUDAGRAPH=1,RF3_HOIST=1`` (``none`` for off); the FPF arm is printed beside it."""
    return describe_line_of(res.switches)


# ------------------------------------------------------------------------------------------------ readers of the kit's own files
def flag_table(graph_flags_path: str) -> Dict[str, str]:
    """Every ``os.environ.get("<NAME>", "<default>")`` of a kit file (graph_flags.py, fpf_rf3_adapter.py), by AST: {name: default}
    (no import, no torch)."""
    tree = ast.parse(open(graph_flags_path, "r", encoding="utf-8").read(), filename=graph_flags_path)
    out: Dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr == "get" and isinstance(f.value, ast.Attribute) and f.value.attr == "environ" \
                and isinstance(f.value.value, ast.Name) and f.value.value.id == "os" and node.args:
            name = node.args[0]
            if isinstance(name, ast.Constant) and isinstance(name.value, str):
                default = node.args[1].value if len(node.args) > 1 and isinstance(node.args[1], ast.Constant) else None
                out[name.value] = default
    return out


def jit_cache_key(torch_version: Optional[str], cc: Optional[str]) -> Optional[str]:
    """``torch<ver>-sm<cc>``: the key configs/*.env derive for the Triton cache (the box's own torch + capability), composed by the shared
    core's ``opt_core.jit_cache`` (shape ``torch-sm``); None when either fact is unknown (the config then refuses to guess a cache)."""
    if not torch_version or not cc:
        return None
    from . import _core
    return _core.load("jit_cache").compose("torch-sm", version=torch_version, cc=cc)
