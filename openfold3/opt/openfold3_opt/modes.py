"""Modes and the environment line each one materialises — the one mode table of this tree.

The add-ons as shipped have no switch file: each is an import hook (`sitecustomize.py` on PYTHONPATH) whose levers are selected by
environment variables, and each README states its activation line. This module holds those lines as data, quoted from the kit files
named beside them, and nothing else in the tree writes a lever switch (`configs/*.env` carry deployment parameters only). `resolve()`
turns a mode into the exports, the hook directories in the add-ons' documented order and the lever names; `stack.activate()` applies it
through the add-ons' own hook files. `tests/test_modes.py` covers the table's shape, the resolver's exports/conflicts/refusals, and the
tier and lever-composition invariants that hold across every line.

Modes (``MODES``; the package default ``DEFAULT_MODE`` = fast):
  off     stock: nothing exported, no hook directory on the path (`pred --mode off` is the stock caller in a clean subprocess, stock_pred.py).
  exact Tier 1 (byte-identical to stock under the deterministic recipe, det.py: both arms on ``STOCK_DET_YAML`` — stock's cuEquivariance
          triangle kernels on, the DS4Sci evoformer attention off); one line (``EXACT_LINES``):
            cueq      stock's own cuEquivariance triangle kernels (the stock configuration: ``STOCK_YAML``, ``STOCK_DET_YAML`` under --det,
                      cli.row_yaml) plus the levers that are bitwise on top of them: fast init (`OF3_FAST_INIT=1`,
                      fast_inference/README.md), the trunk-kernels add-on's two exact levers `OF3T_TEMPL_DISTINCT=1 OF3T_PAIRCACHE=1`
                      (trunk_kernels/README.md: `of3t_hook` first with `OF3T_KIT_LEVERS=<kit>/of3_levers`), the exact-class cells
                      (atom_hoist, castcache, apb_hoist, post_release, transition_exact: the pair-stack SwiGLU transitions after their own
                      LayerNorm as one kernel with the engine's arithmetic; trimul_exact: the pair stacks' triangle multiplication on the core's
                      trimul provider on word exact, bit-proven per call class; trimul_form: its module-statement classes (the template pair
                      stack) on the provider's exact tier under the form of3_module (row of3_form), bit-proven per class and token count; triatt_exact: the library's triangle-attention call served by the
                      core's provider on word exact, bit-proven per call class) and the CUDA-graphed diffusion sampler `OF3_CUDA_GRAPHS=1
                      OF3_GRAPHS_STRICT=1` up to the line's own 512-token cap (Line.graphs_max_tokens: launch-bound below, GPU-bound above —
                      eager there and without a token count); `OF3T_TRIATT` / `OF3T_TRIMUL` / `OF3T_APB` unset. Inside the graphed rollout the
                      diffusion module's attention runs the cuEquivariance / torch path (DS4Sci launches on the legacy default stream and
                      cannot be captured); under --det 1 the line never called DS4Sci: bitwise vs stock, graphed or eager.
  fast    Tier 2 (the tolerance tier; not bitwise): the kit line plus the trunk-kernels add-on's default set — its two exact levers and its
          composition with the kit: `of3t_hook` first, `OF3T_KIT_LEVERS=<kit>/of3_levers`; the triangle attention and the triangle
          multiplication are the pair cells' — the core's providers by the line's tier word — and, beneath them, stock's own statement
          with the line's runner configuration) + `OF3_GRAPHS_STRICT=1`:
          every graphed line of this tree is
          strict, so a capture failure fails the run instead of producing an eager row under a graphed label; the cells hook's levers
          (``LINES[("fast", None)]``) and torch's expandable-segments allocator (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
          lever `alloc_expandable`: the confidence head's headroom after the item-boundary release at large token counts).

  big   Tier 2 (the memory tier: a lever taken for its ceiling, the speed cost reported beside): ONE lever set per
          GPU count (``BIG_LINES``, selected by `--n_gpu` alone: P = 1 runs `resident`, P in 2|4|8 runs `tp`). The multi-GPU line:
            tp        the row-sharded pair representation across P GPUs of one node: opt_core.mem.rowpair's row statements bound to OpenFold3's
                      modules by opt/openfold3_opt/tp_rowpair (entry hook tp_rowpair/hook/sitecustomize.py; one `pred` rank process per GPU spawned by
                      tp.py; an NCCL group from OF3TP_RANK/OF3TP_WORLD/OF3TP_ADDR/OF3TP_PORT) with the fast-inference kit's fast init chained
                      after the hook (OF3_FAST_INIT=1, fast_inference/README.md) and the allocator `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
                      CUDA graphs and the trunk-kernels swaps stay off: the sharded pair stack replaces the ops they act on
                      (CHANGES.md `tp` contract: CUDA graphs off; DS4Sci/cuEq kernels off in reference AND candidate). The world size is the run's, not the line's:
                      `--n_gpu P` / ``OPENFOLD3_OPT_N_GPU`` (``LINE_PARAMS[("big", "tp")]``: P in 2|4|8), and
                      the CLI spawns the ranks (tp.py); OF3TP_WORLD/OF3TP_RANK/OF3TP_ADDR/OF3TP_PORT/OF3TP_CHUNK are the run's variables
                      (``RUN_VARS``), set by the launcher, never exported by the line.
          The single-GPU line `resident` (``LINES``) is the offload port's O1 line — the pair-track statements in row blocks with the pair
          representation resident on the device — composed UNDER the fast line's kernel and sampler cells: big = fast minus only the levers
          with a device-memory cost (the CUDA-graphed sampler, the pair cache, `atom_hoist`, `dit_glue`, `castcache`,
          `trunk_graph`; CHANGES.md), plus the memory levers. Hook chain confhead > of3o > cells > of3t_hook > of3_levers:
          the port's units install beneath the cells and remain their named fallbacks. It carries the confidence-phase levers (``CONF_LEVERS``)
          from ``CONF_MIN_TOKENS`` polymer tokens up (``conf_gate``, opt_core.attn.size_gate words): the row-block confidence head
          (`confhead`, opt/openfold3_opt/confhead.py, hook opt/openfold3_opt/hooks/confhead first in the chain) over the port's chunked scorer
          (`conf_chunked`). Below ``CONF_MIN_TOKENS`` the resolver drops the hook, the switches (``CONF_ENVS_BELOW``) and the levers: the line
          then runs the port's stock confidence path, byte for byte the line without them.

"""
import importlib.metadata
import os
import shutil
import subprocess
from dataclasses import dataclass, field, replace as _dc_replace
from typing import Dict, List, Mapping, Optional, Tuple

MODES: Tuple[str, ...] = ("off", "exact", "fast", "big")
# `exact` is the composition bitwise vs stock under the det recipe: stock's own cuEquivariance triangle kernels plus the levers that are bitwise on
# top of them; the recipe turns the DS4Sci evoformer attention off on both arms (that kernel is bf16 and not run-to-run deterministic), so equality is
# anchored on the cuEquivariance kernels plus upstream's own PyTorch attention where they decline (STOCK_DET_YAML; the stock arm runs it too).
# The determinism row of every mode: the recipe the CLI applies to the process (`det.apply_env(level, graphed)`; `--det 0|1`, and `--det 2`
# where the caller names it), whether the mode's lines capture CUDA graphs (the graphed flag is per line: `graphed(mode, line)` — the recipe
# re-asserts OF3_GRAPHS_STRICT on graphed lines only), the seed source, and the det recipe as shipped against stock.
DETERMINISM: Dict[str, Dict[str, str]] = {
    "off":   {"recipe": "det.apply_env(level) on the stock child (no switch of the kit)", "graphs": "none", "seed": "upstream's (the runner yaml / --num-model-seeds)", "equality": "stock vs stock: --det 1 on the det configuration (STOCK_DET_YAML: the cuEquivariance triangle kernels on, the DS4Sci attention off); --det 2 adds the CPU-thread pin"},
    "exact": {"recipe": "det.apply_env(level, graphed=graphed_call(...)) — the graphed sampler at or below the line's 512-token cap, eager above and without a token count", "graphs": "diffusion graphs (C3) up to 512 tokens: the graphed rollout runs the diffusion module's attention on the cuEquivariance / torch path instead of DS4Sci (legacy-default-stream kernel) — under --det 1 the line never called DS4Sci, so the graphed line is bitwise the eager one", "seed": "upstream's", "equality": "bitwise vs stock under --det 1: both arms on STOCK_DET_YAML (structures and confidences, sha256)"},
    "fast":  {"recipe": "det.apply_env(level, graphed=True)", "graphs": "diffusion graphs (C3)", "seed": "upstream's", "equality": "band (seed-spread rule); Tier 2"},
    "big": {"recipe": "det.apply_env(level, graphed=graphed_call(...)) — resident: eager from the port's item gate up (the graph switches unset), the fast line's graphed sampler below it (OF3_GRAPHS_STRICT re-asserted there); tp: eager", "graphs": "resident below the port's item gate: the fast line's (diffusion sampler + pairformer trunk captured); none from the gate up and on tp", "seed": "upstream's", "equality": "resident: the fast line's kernel and sampler cells under the port's row-blocked statements — band (seed-spread rule), run-to-run bitwise under --det 1; Tier 2; tp: one pred process per GPU (2|4|8) in an NCCL group, eager; run-to-run bitwise at a fixed world size under --det 1, NOT 1-GPU-exact (rank 0 vs off/exact 0/45 at 5 samples, 0/9 at 1 sample) = band (seed-spread rule at P=2, n_anchors 8)"},
}
DEFAULT_MODE = "fast"                      # the house default: fast wherever a fast mode ships
EXACT_LINES: Tuple[str, ...] = ("cueq",)
DEFAULT_EXACT_LINE = "cueq"
BIG_LINES: Tuple[str, ...] = ("resident", "tp")                                    # the memory mode's lines, ONE per GPU count: the pair representation resident on one GPU
                                                                                    # (O1; `--n_gpu 1`) / row-sharded across 2|4|8 GPUs (tp; `--n_gpu P`) — `line_arg`: no other selector
DEFAULT_BIG_LINE = "resident"
@dataclass(frozen=True)
class LineParam:
    """A line's parameter (a line that takes a value): its name, CLI flag, variable, allowed values and default (None = required)."""
    name: str
    flag: str
    env: str
    values: Tuple[str, ...]
    default: Optional[str] = None


# Lines reserved by name for a branch that has not landed (check_line refuses them with the reason); none at present.
# A line's parameter (a line that takes a value). `--n_gpu P` / OPENFOLD3_OPT_N_GPU=P is the memory mode's resource axis (opt_core.mem.ngpu:
# explicit, default 1, never auto-detected): P = 1 is every route's single-GPU run (ACTIVE/EXIT `n_gpu=1 sharding=none`); P in 2|4|8 under
# `big` runs the `tp` line — the pair representation row-sharded over P GPUs of the box, one rank process each (ACTIVE/EXIT `n_gpu=P
# sharding=rowpair`); P > 1 under any other mode or line is refused by name (ngpu.REFUSE_MODE / foreign_params), and so is a P the line does
# not run (check_tp_world) or a box with fewer visible GPUs than P (tp.launch, ngpu.refuse_unless_visible) — never a silent single-GPU run.
N_GPU_VALUES: Tuple[str, ...] = ("1", "2", "4", "8")            # the P set this kit ships (the runner renders big_xP arms for P > 1 in it)
TP_LINES: Tuple[str, ...] = ("tp",)                                               # the multi-GPU lines (one process per GPU; the row-sharded pair representation)
LINE_PARAMS: Dict[Tuple[str, Optional[str]], LineParam] = {
    ("big", "tp"): LineParam(name="n_gpu", flag="--n_gpu", env="OPENFOLD3_OPT_N_GPU", values=("2", "4", "8"), default="1"),
}
FAST_PRECISION = "bf16"                                                            # the fast mode's precision: upstream's own `pl_trainer_args.precision: bf16-mixed` through the runner yaml
FAST_BF16_YAML = os.path.join("opt", "openfold3_opt", "fast_bf16_predict.yml")   #  FAST_BF16_YAML (stock_predict.yml's model_update + that block; trunk + heads under bf16 autocast, the sampler under the rollout_bf16 lever)
MODE_LINES: Dict[str, Tuple[str, ...]] = {"exact": EXACT_LINES, "big": BIG_LINES}      # the modes whose table entries are keyed by a line name (line_arg selects it; no user-facing selector)
TIER = {"off": "stock", "exact": "exact", "fast": "tolerance"}
# the big lines' tiers are PER LINE: `exact` only for a line whose outputs equal `exact` byte for byte under the deterministic recipe,
# else `tolerance` ("pending" is refused as a tier word)
# big is tier 2 by definition: the resident line's units being bitwise on their own does not make the mode exact class
LINE_TIER: Dict[Tuple[str, Optional[str]], str] = {("big", "resident"): "tolerance", ("big", "tp"): "tolerance"}

# The size gate of the CUDA-graphed diffusion step (every line that carries `cuda_graphs`: fast) — AVAILABLE,
# NOT THE DEFAULT. The decision and its words are the tree's shared ones (opt_core.mem.graph_gate: parse_cap / decide → GateDecision): the cap
# comes from OPENFOLD3_OPT_GRAPHS_MAX_TOKENS / `--graphs-max-tokens` — unset or `always` = no cap (the tested lines as they are), a
# positive integer N = capture only for a query of at most N polymer tokens (inputs.polymer_tokens: residues of the protein/RNA/DNA chains ×
# their copies; ligand atoms uncounted — a named lower bound), `0`/`off` = never capture. `pred` counts the query's tokens; where the
# decision is eager/off the graph switches are left unset (the sampler runs stock's own loop: the line minus the two graph levers) and the
# ACTIVE line, the activation report and the `cuda_graphs` LEVER line carry the decision (`graph=capture|eager:n_tok>cap|off`,
# `gate=<reason>`). The trade: the graphed step is a large win on small inputs and about neutral in steady time at mid sizes, where
# its pools raise the device peak; a cap gives that peak back for a little time. `warm` and the env route carry no
# token count: the line as written (named `gate=n_tok_unknown` when a cap is set).
GRAPHS_LEVERS: Tuple[str, ...] = ("cuda_graphs", "graphs_strict")
GRAPHS_ENVS: Tuple[str, ...] = ("OF3_CUDA_GRAPHS", "OF3_GRAPHS_STRICT")
ENV_GRAPHS_MAX_TOKENS = "OPENFOLD3_OPT_GRAPHS_MAX_TOKENS"
GRAPHS_MAX_TOKENS: Optional[int] = None                                   # no cap unless one is given (available, not the default)
ENV_GRAPHS_KEEP = "OPENFOLD3_OPT_GRAPHS_KEEP"                              # the graphed step's keep-across-items knob, read live by of3_graphs (KEEP_ENV): unset / 1 = a captured
                                                                          # generation is REPLAYED by every later predict item of the process whose capture key is equal on every
                                                                          # tensor the step reads (same token/atom bucket, sample count, numerics mode; entries the step never reads —
                                                                          # the MSA / template stacks — may differ), after the per-item checks (storage unmoved, gather tables refreshed
                                                                          # in place, static buffers at their captured addresses); 0 = drop and re-capture at every item whose feature
                                                                          # signature differs anywhere. Not a lever and not in the line spelling: a caller's
                                                                          # knob like ENV_GRAPHS_MAX_TOKENS; the cuda_graphs LEVER line names it (keep=on|off kept=<n>). Bitwise either way.
GRAPH_GATE_NAME = "graph"                                                 # the fragment's name on the evidence lines (opt_core.mem.graph_gate)

# The size gate of the confidence-phase levers of the single-GPU big line (`confhead` + `conf_chunked`;
# `conf_chunked`): ENGAGED FROM ``CONF_MIN_TOKENS`` POLYMER TOKENS UP, absent below. The predicate and its words are the tree's
# shared ones (opt_core.attn.size_gate.SizeGate(min_tokens=…).decide → `served` | `gated:lt_min`); a call without a token count (`warm`,
# the env route) runs the line as written, named `n_tok_unknown`. Below 2048 tokens the full-size logits fit and the line is the port's O1 line with
# the stock confidence path, so a resident call under 2048 tokens is byte-identical with or without these levers in the tree. Below
# the gate `resolve()` removes the `confhead` hook from the chain, rewrites the switches to ``CONF_ENVS_BELOW`` (dropping the rest of
# ``CONF_ENVS``) and the levers from the request; the ACTIVE line and the levers' LEVER lines carry the decision
# (`conf=served|gated:lt_min|n_tok_unknown min_tokens=2048`).
CONF_LEVERS: Tuple[str, ...] = ("confhead", "conf_chunked")
CONF_HOOK = "confhead"                                                    # the package hook the gate adds/removes at the front of the chain (PACKAGE_HOOKS)
CONF_ENVS: Tuple[str, ...] = ("OPENFOLD3_OPT_CONFHEAD", "OF3O_CONF_MODE", "OF3O_CONF_TM_BACKEND")   # the switches the gate governs
CONF_ENVS_BELOW: Mapping[str, str] = {"OF3O_CONF_MODE": "stock"}   # their values below the gate where the line carries them (the port's stock scorer)
CONF_MIN_TOKENS = 2048
CONF_GATE_NAME = "conf"                                                   # the fragment's name on the evidence lines
ENV_CONFHEAD = "OPENFOLD3_OPT_CONFHEAD"                                   # the confhead hook's switch (hooks/confhead/sitecustomize.py)


def graphs_cap(environ: Optional[Mapping[str, str]] = None, default: Optional[int] = GRAPHS_MAX_TOKENS) -> Optional[int]:
    """The size gate's cap (opt_core.mem.graph_gate.parse_cap of OPENFOLD3_OPT_GRAPHS_MAX_TOKENS): None = no cap (`always`; unset on a line without
    its own cap), 0 = never capture (`0`, `off`), N > 0 = capture at or below N polymer tokens; unset = `default` (the line's own cap,
    Line.graphs_max_tokens, where it has one); any other value refused by name."""
    from opt_core.mem.graph_gate import parse_cap
    environ = os.environ if environ is None else environ
    raw = environ.get(ENV_GRAPHS_MAX_TOKENS)
    try:
        return parse_cap(raw, default)
    except ValueError as e:
        raise ValueError(f"{ENV_GRAPHS_MAX_TOKENS}={raw!r}: {e}") from None


def graphs_gate(levers, n_tokens: Optional[int], cap: Optional[int]):
    """The size gate's decision for a line (opt_core.mem.graph_gate.decide → GateDecision) — None when the line captures no graphs, or when no
    token count is known while a cap would need one (the line as written; resolve names it `gate=n_tok_unknown`)."""
    if "cuda_graphs" not in levers:
        return None
    if n_tokens is None and cap not in (None, 0):
        return None
    from opt_core.mem.graph_gate import decide
    return decide(0 if n_tokens is None else int(n_tokens), cap, name=GRAPH_GATE_NAME)


def conf_gate(line: "Line", n_tokens: Optional[int]) -> Optional[dict]:
    """The confidence levers' size-gate decision for a line — None when the line carries none of CONF_LEVERS; else {served, reason, fragment,
    min_tokens}: `served` from CONF_MIN_TOKENS polymer tokens up (opt_core.attn.size_gate.SizeGate(min_tokens).decide: its event word is the
    reason, `served` | `gated:lt_min`), and served-as-written with the reason `n_tok_unknown` when no token count is known."""
    if not any(l in CONF_LEVERS for l in line.levers):
        return None
    from opt_core.attn.size_gate import SizeGate
    gate = SizeGate(CONF_GATE_NAME, min_tokens=CONF_MIN_TOKENS, source="modes.CONF_MIN_TOKENS")
    if n_tokens is None:
        served, reason = True, "n_tok_unknown"
    else:
        d = gate.decide(int(n_tokens))
        served, reason = d.served, d.event
    return {"served": served, "reason": reason, "fragment": f"{CONF_GATE_NAME}={reason} min_tokens={CONF_MIN_TOKENS}", "min_tokens": CONF_MIN_TOKENS}


# ----------------------------------------------------------------------------------------------------------------- the offload port's item gate
# (big/resident). Below OF3O_MIN_TOKENS polymer tokens the port's O1 units (OF3O_UNIT_LEVERS) step aside BY NAME and the line resolves to the fast
# line AS COMPOSED (of3o_aside: the fast line's hooks, switches, cells, CUDA-graph levers and runner yaml; no offload hook, no OF3O_* word; the runner
# yaml is the fast line's (FAST_BF16_YAML: no attention chunking); the graphs are memory-neutral below the gate by the gate's own definition — the
# below-gate peak is the fast line's, which fits the card there). The gate is decided PER ITEM on the item's own polymer token count: `pred`
# runs a query set whose items fall on both sides of it as one pass per side (of3o_item_groups, cli.run_item_groups); within a pass the gates read the
# pass's largest member. At or above the gate — and whenever no token count is known
# (an unreadable query, `check`, `warm`: fail-safe for memory) — the port engages as written. The gate exists because the memory line must never be
# slower than stock at the sizes stock fits: its default per card (OF3O_GATE_BY_CARD, keyed by MODEL_OPT_TARGET_GPU as
# configs/<card>.env export it) is one token above the largest input size at which the fast line's peak leaves >= 15 % of the card free;
# OF3O_MIN_TOKENS=<int> | none (always engaged) is the caller's knob.
OF3O_GATE_ENV = "OF3O_MIN_TOKENS"
OF3O_GATE_NAME = "of3o_gate"                                                # the fragment's name on the ACTIVE / DRY-RUN line and the units' LEVER lines (gate=…)
OF3O_HOOK = "offload"
OF3O_GATE_BY_CARD: Mapping[str, int] = {"H100": 1401, "A100": 1401}       # engaged from this many polymer tokens up; below, the units step aside
OF3O_GATE_DEFAULT = 1401                                                    # no MODEL_OPT_TARGET_GPU named: the H100 value
OF3O_UNIT_LEVERS: Tuple[str, ...] = ("trimul_hostsnap", "triatt_lean", "trans_inplace", "cond_once", "input_rows", "recycle_rows", "templ_host",
                                     "conf_chunked", "bigln_guard", "host_pool", "chunk_pin")   # registry rows of impl `offload` the hook installs (alloc_expandable is an allocator word both compositions keep)


def of3o_min_tokens(environ: Optional[Mapping[str, str]] = None) -> Tuple[Optional[int], str]:
    """(gate, source): the caller's OF3O_MIN_TOKENS (an int >= 0, or `none` = always engaged) -> (n|None, "env"); else the card table's entry
    for MODEL_OPT_TARGET_GPU -> (n, "card:<gpu>"); else (OF3O_GATE_DEFAULT, "default"). A malformed value raises ValueError (named)."""
    environ = os.environ if environ is None else environ
    raw = (environ.get(OF3O_GATE_ENV) or "").strip()
    if raw:
        if raw.lower() == "none":
            return None, "env"
        try:
            n = int(raw)
        except ValueError:
            raise ValueError(f"{OF3O_GATE_ENV}={raw!r}: expected an integer token count or `none`") from None
        if n < 0:
            raise ValueError(f"{OF3O_GATE_ENV}={raw!r}: expected >= 0")
        return n, "env"
    gpu = (environ.get("MODEL_OPT_TARGET_GPU") or "").strip().upper()
    if gpu in OF3O_GATE_BY_CARD:
        return OF3O_GATE_BY_CARD[gpu], f"card:{gpu}"
    return OF3O_GATE_DEFAULT, "default"


def of3o_gate(line: "Line", n_tokens: Optional[int], environ: Optional[Mapping[str, str]] = None) -> Optional[dict]:
    """The port's item-gate decision for a line — None when the line does not carry the offload hook; else {engaged, reason, fragment, min_tokens}:
    reason `engaged` (n_tokens >= gate), `aside:lt_min` (below: the units step aside), `n_tok_unknown` (no count: engaged), `no_gate` (knob `none`)."""
    if OF3O_HOOK not in line.hooks:
        return None
    gate, source = of3o_min_tokens(environ)
    if gate is None:
        engaged, reason = True, "no_gate"
    elif n_tokens is None:
        engaged, reason = True, "n_tok_unknown"
    else:
        engaged = int(n_tokens) >= gate
        reason = "engaged" if engaged else "aside:lt_min"
    frag = f"{OF3O_GATE_NAME}={reason} min_tokens={'none' if gate is None else gate}({source})"
    return {"engaged": engaged, "reason": reason, "fragment": frag, "min_tokens": gate, "source": source}




def of3o_aside(line: "Line") -> "Line":
    """The memory line below the port's item gate: the FAST line as composed — its hooks, switches, cells, CUDA-graph levers (cuda_graphs, graphs_strict,
    trunk_graph) and runner yaml (LINES[("fast", None)]; the fast line names no runner yaml of its own: ASIDE_YAML = the bf16 member cli.member_yaml composes
    for it), under the resident line's source words; no `offload` hook, no OF3O_* export. The graphs are memory-neutral below the gate by the gate's definition
    (the below-gate peak IS the fast line's, per card); the port's units, and the cells the memory line drops by their memory
    cost, are moot below it."""
    fast = LINES[("fast", None)]
    return _dc_replace(fast, runner_yaml=fast.runner_yaml or ASIDE_YAML, source=line.source + " | below the of3o gate: the fast line as composed, its CUDA graphs included (modes.of3o_aside)")


def effective_line(mode: str, el: Optional[str], environ: Optional[Mapping[str, str]] = None, n_tokens: Optional[int] = None) -> "Line":
    """LINES[(mode, el)] as a call of `n_tokens` polymer tokens runs it: the port's units aside below the of3o gate (of3o_aside), else as written."""
    ln = LINES[(mode, el)]
    if n_tokens is not None and int(n_tokens) <= 0:
        n_tokens = None
    g = of3o_gate(ln, n_tokens, environ)
    return of3o_aside(ln) if g is not None and not g["engaged"] else ln


# --------------------------------------------------------------------------------------------- the resident line's REACH gate (big/resident)
# From REACH_GATE_BY_CARD[card] polymer tokens up, the trunk's full-N^2 pair kernels of the fast composition (REACH_LEVERS: the tri-mult /
# tri-attention / pair-transition cells and their provider rows) step aside BY NAME on the ('big', 'resident') line and the offload port's own
# row-block statements (trimul_hostsnap, triatt_lean, trans_inplace -- installed beneath the cells on this line at every size) serve the pair
# stack: at the gate the cells' full-size operands put the trunk-phase peak at the edge of an 80 GB card while the row-block statements
# need about a third of it and run the trunk several times slower -- reach over speed, from the gate up only. Keyed by card exactly like OF3O_GATE_BY_CARD
# (MODEL_OPT_TARGET_GPU as configs/<gpu>.env exports it, else REACH_GATE_DEFAULT); the caller's word REACH_GATE_ENV = <int> | none overrides
# it (none = never aside). Below the gate, whenever no token count is known, below the port's item gate and on every other line the resolution
# is the line exactly as written (no fragment, no word): the gate only ever REMOVES levers at or above its threshold. Decided per `pred` pass on
# the pass's polymer token count (the item gate's passes read their largest member), after MODEL_OPT_LEVERS_OFF.
REACH_LEVERS: Tuple[str, ...] = ("trimul_v4", "trimul_provider", "triatt_block", "triatt_provider", "pair_transition")
REACH_LINES: Tuple[Tuple[str, Optional[str]], ...] = (("big", "resident"),)
REACH_GATE_ENV = "OPENFOLD3_OPT_REACH_GATE"
REACH_GATE_NAME = "reach"                                                     # the fragment's name on the ACTIVE / DRY-RUN lines (reach=aside:n_tok>=<gate> min_tokens=<gate>(<source>)) and the levers' gate= word
REACH_GATE_REASON = "reach_gate"                                              # the LEVER lines' reason word for the levers the gate set aside (report.lever_lines)
REACH_GATE_BY_CARD: Mapping[str, int] = {"H100": 4500, "A100": 4500}         # aside from this many polymer tokens up (80 GB-class cards)
REACH_GATE_DEFAULT = 4500                                                     # no MODEL_OPT_TARGET_GPU named: the 80 GB value


def reach_min_tokens(environ: Optional[Mapping[str, str]] = None) -> Tuple[Optional[int], str]:
    """(gate, source): the caller's REACH_GATE_ENV (an int >= 0, or `none` = never aside) -> (n|None, "env"); else the card table's entry for
    MODEL_OPT_TARGET_GPU -> (n, "card:<gpu>"); else (REACH_GATE_DEFAULT, "default"). A malformed value raises ValueError (named)."""
    environ = os.environ if environ is None else environ
    raw = (environ.get(REACH_GATE_ENV) or "").strip()
    if raw:
        if raw.lower() == "none":
            return None, "env"
        try:
            n = int(raw)
        except ValueError:
            raise ValueError(f"{REACH_GATE_ENV}={raw!r}: expected an integer token count or `none`") from None
        if n < 0:
            raise ValueError(f"{REACH_GATE_ENV}={raw!r}: expected >= 0")
        return n, "env"
    gpu = (environ.get("MODEL_OPT_TARGET_GPU") or "").strip().upper()
    if gpu in REACH_GATE_BY_CARD:
        return REACH_GATE_BY_CARD[gpu], f"card:{gpu}"
    return REACH_GATE_DEFAULT, "default"


def reach_gate(mode: str, el: Optional[str], levers, n_tokens: Optional[int], environ: Optional[Mapping[str, str]] = None) -> Optional[dict]:
    """The reach gate's decision for a call -- None unless (mode, el) is a REACH_LINES line that still carries one of REACH_LEVERS, a polymer
    token count is known, the gate is not `none`, and n_tokens >= the gate; else {"aside": True, "reason": "aside:n_tok>=<gate>", "fragment":
    "reach=aside:n_tok>=<gate> min_tokens=<gate>(<source>)", "min_tokens": gate, "source": source, "off": (the REACH_LEVERS in `levers`, in
    their order)}. None means: the line as written (nothing printed, nothing dropped)."""
    if (mode, el) not in REACH_LINES:
        return None
    off = tuple(n for n in levers if n in REACH_LEVERS)
    if not off or n_tokens is None or int(n_tokens) <= 0:
        return None
    gate, source = reach_min_tokens(environ)
    if gate is None or int(n_tokens) < gate:
        return None
    reason = f"aside:n_tok>={gate}"
    return {"aside": True, "reason": reason, "fragment": f"{REACH_GATE_NAME}={reason} min_tokens={gate}({source})", "min_tokens": gate, "source": source, "off": off}


def apply_reach_gate(res: "Resolution", environ: Mapping[str, str], engaged: bool = True) -> None:
    """Apply the reach gate to a Resolution (after apply_levers_off, with res.n_tokens set): at or above the gate on the engaged resident line the
    REACH_LEVERS it still carries leave `levers`, their switches leave `exports` and join `unsets` exactly as MODEL_OPT_LEVERS_OFF=<those names>
    would (the pair-cell family OPENFOLD3*_OPT_PAIR* with the last cell, the provider words with their levers), a hook nothing arms any more
    leaves `hooks`; `reach_off`, the fragment, the reason and one note record it. A malformed REACH_GATE_ENV is a conflict (named). No-op
    everywhere else."""
    if not engaged:
        return
    try:
        rg = reach_gate(res.mode, res.line, res.levers, res.n_tokens, environ)
    except ValueError as e:
        res.conflicts.append(str(e))
        return
    if rg is None:
        return
    off = list(rg["off"])
    drop: List[str] = []
    for n in off:
        if n not in PAIR_CELL_LEVERS:
            drop += [k for k in LEVER_SWITCHES.get(n, ()) if k not in drop]
    if any(n in PAIR_CELL_LEVERS for n in off):
        left = [n for n in PAIR_CELL_LEVERS if n in res.levers and n not in off]
        if left and PAIR_ENVS[0] in res.exports:
            res.exports[PAIR_ENVS[0]] = ":".join(left)
        else:
            drop += [k for k in PAIR_ENVS if k not in drop]
    for k in drop:
        res.exports.pop(k, None)
        if k not in res.unsets and k not in LEVERS_OFF_KEPT_SETTABLE:
            res.unsets.append(k)
    res.levers = [l for l in res.levers if l not in off]
    res.hooks = [h for h in res.hooks if h not in HOOK_ARMING or any(k in res.exports for k in HOOK_ARMING[h])]
    res.reach_gate, res.reach_reason, res.reach_min_tokens, res.reach_off = rg["fragment"], rg["reason"], rg["min_tokens"], off
    res.notes.append(f"reach gate: {','.join(off)} aside from {rg['min_tokens']} polymer tokens ({rg['source']}) -- the offload port's row-block pair statements serve the trunk")


ITEM_GATE_NAME = "ITEM GATE"                                                # cli: the summary line of a query set that straddles the port's item gate (item_gate_line, cli.run_item_groups)
OF3O_ASIDE_WORD = "fast"                                                        # the below-gate line's word on that line (of3o_aside: the fast line as composed, its CUDA graphs included)


def of3o_item_groups(line: "Line", tokens_each: Mapping[str, Optional[int]], environ: Optional[Mapping[str, str]] = None) -> Optional[dict]:
    """The port's item gate decided PER ITEM of a query set — None when `line` does not carry the offload hook (nothing to decide); else
    {"below": names, "engaged": names, "tokens": {name: n}, "min_tokens": gate | None, "source": word}: every item is judged by of3o_gate on its OWN
    polymer token count, i.e. exactly as a query set holding that item alone resolves (effective_line) — below the gate the line of of3o_aside, at or
    above it the port; a count of 0 / None is unknown and engages (the memory fail-safe). The cli runs a set whose items fall on both sides of the gate
    as one `pred` pass per side (cli.run_item_groups); a set on one side runs as one pass, as written (within a pass the gates read the pass's largest
    member, which is then on the same side as every member)."""
    if OF3O_HOOK not in line.hooks:
        return None
    gate, source = of3o_min_tokens(environ)
    below, engaged, tokens = [], [], {}
    for name, n in tokens_each.items():
        n = int(n) if n is not None else 0
        tokens[str(name)] = n
        g = of3o_gate(line, n if n > 0 else None, environ)
        (engaged if g["engaged"] else below).append(str(name))
    return {"below": tuple(below), "engaged": tuple(engaged), "tokens": tokens, "min_tokens": gate, "source": source}


def item_gate_line(groups: Mapping, engaged_line: str = "resident") -> str:
    """The cli's one summary line for a query set that straddles the port's item gate and runs as two passes (cli.run_item_groups):
    `ITEM GATE: below=<n> (<tokens,…>) line=<OF3O_ASIDE_WORD> | engaged=<n> (<tokens,…>) line=<engaged line> gate=<n|none> source=<card:X|env|default>`."""
    t = groups.get("tokens") or {}

    def toks(names):
        return ",".join(str(t.get(n, "?")) for n in names) or "-"
    gate = groups.get("min_tokens")
    return (f"{ITEM_GATE_NAME}: below={len(groups['below'])} ({toks(groups['below'])}) line={OF3O_ASIDE_WORD} | engaged={len(groups['engaged'])} "
            f"({toks(groups['engaged'])}) line={engaged_line} gate={'none' if gate is None else gate} source={groups.get('source')}")


def tier(mode: str, line: Optional[str] = None) -> str:
    """The numerics tier of a mode, per line where the lines are tiered separately (LINE_TIER)."""
    if (mode, line) in LINE_TIER:
        return LINE_TIER[(mode, line)]
    return TIER[mode]
# the tp run's own variables: set per rank by the launcher (tp.py), read by the rank's hook and the core's process group (tp_rowpair/hook/sitecustomize.py, opt_core.mem.rowpair.dist); never a line's export
RUN_VARS: Tuple[str, ...] = ("OF3TP_WORLD", "OF3TP_RANK", "OF3TP_ADDR", "OF3TP_PORT", "OF3TP_CHUNK")

# add-on directories (relative to the openfold3/ tree) and the hook directory inside each: the directory that holds the add-on's sitecustomize.py
KITS: Dict[str, str] = {
    "fast_inference": os.path.join("opt", "forward", "fast_inference"),        # the fast-inference add-on (fast init, the CUDA-graphed sampler)
    "trunk_kernels": os.path.join("opt", "forward", "trunk_kernels"),          # the trunk-kernels add-on
    "offload": os.path.join("opt", "forward", "offload"),                      # the offload port (the `big` mode; offload/README.md)
}
HOOK_DIR: Dict[str, str] = {"fast_inference": "of3_levers", "trunk_kernels": "of3t_hook", "offload": "of3o"}
PORTS: Tuple[str, ...] = ("offload",)           # add-on table entries that are PORTS (merged lever by lever into this tree), not carried whole
HOOK_FILE = "sitecustomize.py"
STOCK_YAML = os.path.join("opt", "openfold3_opt", "stock_cueq_on_predict.yml")        # the stock configuration (stock/PINS.json "stock_config"): the predict preset as shipped (DS4Sci attention on) + the cuEquivariance triangle kernels on — `pred --mode off`, the base of `exact`, what every arm is timed against
STOCK_DET_YAML = os.path.join("opt", "openfold3_opt", "stock_cueq_on_det_predict.yml")   # stock under the det recipe (--det 1|2): STOCK_YAML with the DS4Sci attention off — the equality runs' stock arm and `exact --det`; cli.row_yaml selects it by the det level
KERNELS_OFF_YAML = os.path.join(KITS["fast_inference"], "config", "stock_predict.yml")   # the kernels-off configuration (DS4Sci off, cuEquivariance off: upstream's PyTorch pair kernels): the base of `fast` (FAST_BF16_YAML = it + bf16) and of the `big` lines, whose own cells and offload wrappers replace or stream those modules
SHIPPED_YAML = os.path.join("opt", "openfold3_opt", "shipped_predict.yml")          # upstream's shipped configuration exactly (no runner yaml: DS4Sci attention on, cuEquivariance off) — a reference file; no line runs it
KIT_LEVERS_ENV = "OF3T_KIT_LEVERS"                                                   # trunk_kernels/of3t_hook/sitecustomize.py:12-18: the chained kit hook
OFFLOAD_KIT_LEVERS_ENV = "OF3O_KIT_LEVERS"                                           # offload/of3o/sitecustomize.py: the same chain for the big lines
CONFHEAD_CHAIN_ENV = "OPENFOLD3_OPT_CONFHEAD_CHAIN"                                # hooks/confhead/sitecustomize.py: the chained hook directory (the offload port's)
PAIRFUSED_CHAIN_ENV = "OPENFOLD3_OPT_PAIR_CHAIN"                                    # hooks/cells/sitecustomize.py: the chained hook directory (the trunk-kernels add-on's of3t_hook)
KIT_LEVERS_ENVS = (KIT_LEVERS_ENV, OFFLOAD_KIT_LEVERS_ENV, CONFHEAD_CHAIN_ENV, PAIRFUSED_CHAIN_ENV)   # every chain variable (excluded from the line spelling: the chain is the hook order)
CONFHEAD_HOOK_DIR = os.path.join("opt", "openfold3_opt", "hooks", "confhead")        # the package's own hook directory: the row-block confidence head (confhead.py)
CELLS_HOOK_DIR = os.path.join("opt", "openfold3_opt", "hooks", "cells")              # the package's own hook directory: the cell levers (cells/pairfused.py, rollout.py, dit_attn.py, dit_glue.py, apb_trunk.py, token_agg.py, atom_window.py, atom_hoist.py, templ_embed.py)
DIT_ENVS: Tuple[str, ...] = ("OPENFOLD3_OPT_DIT", "OPENFOLD3_OPT_DIT_MIN_TOKENS", "OPENFOLD3_OPT_DIT_CORE")   # dit_attn.py's switches (the fast line exports the first)
APB_WORD_ENVS: Tuple[str, ...] = ("OPENFOLD3_OPT_APB_TIER", "OPENFOLD3_OPT_APB_WORD")             # cells/apb_word.py: the pair-bias attention family's provider word (the fast / big lines export the tier; the second is the caller's knob: any provider word)
DIT_GLUE_ENVS: Tuple[str, ...] = ("OPENFOLD3_OPT_DIT_GLUE", "OPENFOLD3_OPT_DIT_GLUE_MIN_TOKENS", "OPENFOLD3_OPT_DIT_GLUE_CORE", "OPENFOLD3_OPT_DIT_GLUE_ROWS")   # dit_glue.py's switches (the fast line exports the first)
APB_TRUNK_ENVS: Tuple[str, ...] = ("OPENFOLD3_OPT_APB_TRUNK", "OPENFOLD3_OPT_APB_TRUNK_MIN_TOKENS", "OPENFOLD3_OPT_APB_TRUNK_HIGH_PRECISION", "OPENFOLD3_OPT_APB_TRUNK_SCOPE", "OPENFOLD3_OPT_APB_TRUNK_PRODUCER")                 # apb_trunk.py's switches (the fast line exports the first)
TOKEN_AGG_ENVS: Tuple[str, ...] = ("OPENFOLD3_OPT_TOKEN_AGG",)                                              # token_agg.py's switch (the fast line exports it)
ATOM_WINDOW_ENVS: Tuple[str, ...] = ("OPENFOLD3_OPT_ATOM_WINDOW", "OPENFOLD3_OPT_ATOM_WINDOW_PRECISION", "OPENFOLD3_OPT_ATOM_WINDOW_INV")   # atom_window.py's switches (the fast line exports the first)
TEMPL_EMBED_ENVS: Tuple[str, ...] = ("OPENFOLD3_OPT_TEMPL_EMBED",)                                          # templ_embed.py's switch (the fast line exports it)
ATOM_HOIST_ENVS: Tuple[str, ...] = ("OPENFOLD3_OPT_ATOM_HOIST", "OPENFOLD3_OPT_ATOM_HOIST_MAX_GB", "OPENFOLD3_OPT_ATOM_HOIST_INV")   # atom_hoist.py's switches (the exact and fast lines export the first)
CASTCACHE_ENVS: Tuple[str, ...] = ("OPENFOLD3_OPT_CASTCACHE",)                                              # castcache.py's switch (the exact and fast lines export it)
EXACTLN_ENVS: Tuple[str, ...] = ("OPENFOLD3_OPT_EXACTLN", "OPENFOLD3_OPT_EXACTLN_WORD")                    # exactln.py's switch (the exact line exports it: the LayerNorm primitive on the core's provider by the word exact; _WORD is the caller's knob)
LN_PROVIDER_ENVS: Tuple[str, ...] = ("OPENFOLD3_OPT_LN_PROVIDER", "OPENFOLD3_OPT_LN_TIER", "OPENFOLD3_OPT_LN_WORD")   # ln_provider.py's switches (the fast and big/resident lines export the first two: the same binding asked with the line's tier word; _WORD is the caller's knob)
APB_HOIST_ENVS: Tuple[str, ...] = ("OPENFOLD3_OPT_APB_HOIST",)                                              # apb_hoist.py's switch (the exact and fast lines export it)
TRUNK_GRAPH_ENVS: Tuple[str, ...] = ("OPENFOLD3_OPT_TRUNK_GRAPH", "OPENFOLD3_OPT_TRUNK_GRAPH_NMAX")          # trunk_graph.py's switches (the fast line exports the first)
ASIDE_YAML = FAST_BF16_YAML                                                 # the fast line names no runner yaml of its own (cli.member_yaml composes the bf16 member): named here for the resident line below the port's item gate (of3o_aside)
POST_RELEASE_ENVS: Tuple[str, ...] = ("OPENFOLD3_OPT_POST_RELEASE", "OPENFOLD3_OPT_POST_RELEASE_NTOK", "OPENFOLD3_OPT_POST_RELEASE_MIN_FREE_GB")   # post_release.py's switches (the exact and fast lines export the first)
TUNER_GUARD_ENVS: Tuple[str, ...] = ("OPENFOLD3_OPT_TUNER_GUARD",)                                          # tuner_guard.py's switch (the fast line exports it)
POSTFWD_MEM_ENVS: Tuple[str, ...] = ("OPENFOLD3_OPT_POSTFWD_MEM", "OPENFOLD3_OPT_POSTFWD_MEM_MIB")             # postfwd_mem.py's switches (the exact, fast and big/resident lines export the first; _MIB: the block budget, 0 = the engine's statements)
SYNC_HOIST_ENVS: Tuple[str, ...] = ("OPENFOLD3_OPT_SYNC_HOIST", "OPENFOLD3_OPT_SYNC_HOIST_PARTS")            # sync_hoist.py's switches (the exact and fast lines export the first; _PARTS is the caller's: all | none | a part list)
LOADER_WORKERS_ENVS: Tuple[str, ...] = ("OPENFOLD3_OPT_LOADER_WORKERS", "OPENFOLD3_OPT_LOADER_WORKERS_CAP")    # loader_workers.py's switches (the exact, fast and big/resident lines export the first; _CAP=none: the engine's worker count)
TRIATT_EXACT_ENVS: Tuple[str, ...] = ("OPENFOLD3_OPT_TRIATT_EXACT",)                                          # triatt_exact.py's switch (the exact line exports it: the library's triangle-attention call on the core's provider, word exact)
TRIMUL_EXACT_ENVS: Tuple[str, ...] = ("OPENFOLD3_OPT_TRIMUL_EXACT", "OPENFOLD3_OPT_TRIMUL_EXACT_FORM")       # trimul_exact.py's switch (the exact line exports it: the pair stacks' triangle multiplication on the core's ONE trimul provider, word exact, bit-proven per class) + trimul_form's (cells/trimul_form.py: the module-statement classes on the provider's exact tier under the form of3_module (row of3_form); the exact line exports both, the arming switch first)
TRANSITION_EXACT_ENVS: Tuple[str, ...] = ("OPENFOLD3_OPT_TRANSITION_EXACT",)                                  # transition_exact.py's switch (the exact line exports it: the pair-stack SwiGLU transitions after their own LayerNorm as one kernel with the engine's arithmetic, the core's transition provider by the tier word exact under this engine's form)
TRIMUL_PROVIDER_ENVS: Tuple[str, ...] = ("OPENFOLD3_OPT_TRIMUL_PROVIDER", "OPENFOLD3_OPT_TRIMUL_WORD", "OPENFOLD3_OPT_TRIMUL_TIER")             # of3_trimul.py's switches (the fast line exports the first: the trimul_v4 cell's row per call class from the core's ONE trimul provider in this kit's order; _WORD is the caller's knob: fast | any provider word)
FASTJSON_ENVS: Tuple[str, ...] = ("OPENFOLD3_OPT_FASTJSON", "OPENFOLD3_OPT_FASTJSON_ROUTE")               # fastjson.py's switches (every kit line exports the first; the second is the caller's A/B knob: fast|stock)
WRITER_OVERLAP_ENVS: Tuple[str, ...] = ("OPENFOLD3_OPT_WRITER_OVERLAP", "OPENFOLD3_OPT_WRITER_OVERLAP_ROUTE")   # writer_overlap.py's switches (the exact, fast and big/resident lines export the first; the second is the route knob: process|thread|sync)
HOSTFEAT_ENVS: Tuple[str, ...] = ("OPENFOLD3_OPT_HOSTFEAT", "OPENFOLD3_OPT_HOSTFEAT_PARTS")   # hostfeat.py's switches (every kit line but off exports the first; the second names the parts: a3m,msaidx)
CKPT_MMAP_ENVS: Tuple[str, ...] = ("OPENFOLD3_OPT_CKPT_MMAP",)   # ckpt_mmap.py's switch (every kit line but off exports it)
ROLLOUT_ENVS: Tuple[str, ...] = ("OPENFOLD3_OPT_ROLLOUT",)                                                  # rollout.py's switch (the fast line exports it at its default precision)
PAIR_ENVS: Tuple[str, ...] = ("OPENFOLD3_OPT_PAIR", "OPENFOLD3_OPT_PAIR_IMPL", "OPENFOLD3_OPT_PAIR_CORE", "OPENFOLD3_OPT_PAIR_LN", "OPENFOLD3_OPT_PAIR_STRICT")   # pairfused.py's switches (the fast line exports them; every other line requires them unset)
CELL_FAMILIES: Tuple[Tuple[str, ...], ...] = (PAIR_ENVS, ROLLOUT_ENVS, DIT_ENVS, DIT_GLUE_ENVS, APB_TRUNK_ENVS, TOKEN_AGG_ENVS, ATOM_WINDOW_ENVS, ATOM_HOIST_ENVS, TEMPL_EMBED_ENVS,
                                                CASTCACHE_ENVS, EXACTLN_ENVS, LN_PROVIDER_ENVS, APB_HOIST_ENVS, TRUNK_GRAPH_ENVS, POST_RELEASE_ENVS, TUNER_GUARD_ENVS, FASTJSON_ENVS, WRITER_OVERLAP_ENVS, HOSTFEAT_ENVS, CKPT_MMAP_ENVS,
                                                POSTFWD_MEM_ENVS, LOADER_WORKERS_ENVS, SYNC_HOIST_ENVS, TRIATT_EXACT_ENVS, TRIMUL_EXACT_ENVS, TRANSITION_EXACT_ENVS, TRIMUL_PROVIDER_ENVS, APB_WORD_ENVS)   # every cell lever's switch family, arming switch first: a family
# preset in the caller's environment while the line does not export its arming switch is a lever the line does not carry — refused by name in resolve() (the cells
# hook installs any cell whose own switch is set, so a stray switch must never reach a line that mounts the hook)
ACTIVATION_CELLS: Tuple[str, ...] = ("fastjson", "writer_overlap", "hostfeat", "ckpt_mmap")      # cells the PACKAGE installs at activation on every kit line (stack.activate -> cells.<name>.install()), not a hook: engine plumbing every
# line shares (the output writer); their switches are read by opt/openfold3_opt/cells/<name>.py whatever hooks the line mounts (tests/test_modes.py reads them there)
ACTIVATION_FAMILIES: Tuple[Tuple[str, ...], ...] = (FASTJSON_ENVS, WRITER_OVERLAP_ENVS, HOSTFEAT_ENVS, CKPT_MMAP_ENVS)   # their switch families: cell families for the conflict rule, but they arm no hook (HOOK_ARMING: the cells hook leaves the chain when only these are set)
TP_ROWPAIR_HOOK_DIR = os.path.join("opt", "openfold3_opt", "tp_rowpair", "hook")    # the package's tensor-parallel hook: the opt_core rowpair adapter (tp_rowpair/__init__.py)
PACKAGE_HOOKS: Dict[str, str] = {"confhead": CONFHEAD_HOOK_DIR, "tp_rowpair": TP_ROWPAIR_HOOK_DIR, "cells": CELLS_HOOK_DIR}                      # hooks the package itself ships (not carried add-ons), relative to the tree
CHAIN_ENV: Dict[str, str] = {"confhead": CONFHEAD_CHAIN_ENV, "cells": PAIRFUSED_CHAIN_ENV, "offload": OFFLOAD_KIT_LEVERS_ENV, "trunk_kernels": KIT_LEVERS_ENV}   # the hooks whose sitecustomize executes a chained hook directory named by a variable -> that variable
OFFLOAD_SHIPPED_C16_YAML = os.path.join(KITS["offload"], "config", "shipped_predict_c16.yml")   # the big lines on the SHIPPED attention kernel (DS4Sci on) with the same pinned plan: the memory rows against the shipped stock
BIG_BF16_C16_YAML = os.path.join("opt", "openfold3_opt", "big_bf16_c16_predict.yml")   # the big lines' runner yaml (resident, tp: Line.runner_yaml): OFFLOAD_C16_YAML's model_update + pl_trainer_args.precision bf16-mixed
OFFLOAD_C16_YAML = os.path.join(KITS["offload"], "config", "stock_predict_c16.yml")   # the big equality runs' configuration on BOTH arms: KERNELS_OFF_YAML + chunk_size 16, tune_chunk_size false                                                   # trunk_kernels/of3t_hook/sitecustomize.py:12-18: the chained kit hook


@dataclass(frozen=True)
class Line:
    """One activation line: the hook directories in the order they go on PYTHONPATH (first = the hook file that is executed and chains
    the rest; carried add-ons, the add-on table, or the package's own hooks, PACKAGE_HOOKS), the exported switches, the switches that must be unset, the kit whose `of3_levers` OF3T_KIT_LEVERS names, the lever names
    (registry.LEVERS), where the line was read
    of the kit's own (relative to the openfold3/ tree; the same directory on the served route)."""
    hooks: Tuple[str, ...]
    env: Mapping[str, str]
    unset: Tuple[str, ...]
    kit_levers: Optional[str]
    levers: Tuple[str, ...]
    source: str
    kit_levers_env: str = KIT_LEVERS_ENV                        # the variable the entry hook reads for the chained of3_levers directory
    runner_yaml: Optional[str] = None                 # the runner yaml the line runs under (relative to the openfold3/ tree); None = the mode's configuration (cli.row_yaml)
    stock_kernels: bool = False                       # the line's base is the stock configuration (stock's cuEquivariance + DS4Sci kernels on the runner yaml): a caller's yaml under it is
                                                      # NOT pinned kernels-off (runner_yaml.line_pins); its graphed sampler turns DS4Sci off inside the rollout itself
    graphs_max_tokens: Optional[int] = None           # the line's OWN default cap of the graphed sampler (graphs_cap's default when OPENFOLD3_OPT_GRAPHS_MAX_TOKENS is unset):
                                                      # None = the line captures at every size; N = the graphed step pays only up to N tokens on this line (exact: 512) —
                                                      # such a line runs EAGER when no token count is known (fail-closed, named gate=n_tok_unknown)
                                                      # templates: every line behaves as stock 0.4.1 — a templated query is accepted, parsed, and not consumed at
                                                      # inference (templ_guard names it); no line has a template switch


ALLOC_POLICY = "expandable"                                            # the allocator policy of the lines that carry lever `alloc_expandable` (opt_core.mem.torch_alloc.POLICIES): torch's
ALLOC_ENV: Dict[str, str] = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}   # expandable segments; its ONE spelling here = torch_alloc.env_row(ALLOC_POLICY) (resolve asserts it and composes a
                                                                       # caller's other allocator keys + the graphed-line declaration through torch_alloc.export)
_TP_ENV: Dict[str, str] = {"OF3TP_TRIATT_QBLOCK": "256", "OF3TP_TRIATT_ROWBLOCK": "256", "OF3TP_APB_QBLOCK": "256", "OF3TP_APB_ROWBLOCK": "256", "OF3TP_TRIMUL_SUB": "128", "OF3_FAST_INIT": "1", **ALLOC_ENV}
"""The multi-GPU lines' switches (TP_LINES): the row-sharded pair representation's levers as shipped."""
NOISE_SYNC_SWITCH = "OF3TP_DIFF_NOISE_SYNC"
"""The replicated-tensor sync policy word of the `tp` line, set per call by determinism level (replicated_sync_policy; core name
ROWPAIR_DIFF_NOISE_SYNC): `bcast` = rank 0 authoritative at every sync point (the per-denoiser-call diffusion state, every noise draw), `guard` =
strict (a replicated tensor that differs across ranks fails the run by name)."""


def replicated_sync_policy(line: str, det_level: int, environ: Optional[dict] = None) -> str:
    """Set the multi-GPU line's replicated-tensor sync policy for this rank BY DETERMINISM LEVEL and return its census words: det 0 -> `bcast`
    (non-deterministic kernels leave replicated-by-recompute tensors only tier-2 identical across ranks, so rank 0's tensor is adopted at every
    sync point), det >= 1 -> `guard` (bitwise kernels: a mismatch is a real defect and fails by name)."""
    environ = os.environ if environ is None else environ
    v, why = ("guard", f"det {det_level}: strict, a replicated tensor that differs across ranks fails by name") if det_level >= 1 else \
             ("bcast", f"det {det_level}: rank 0 authoritative at every sync point")
    environ[NOISE_SYNC_SWITCH] = v                            # the adapter maps it to the core's name at install (tp_rowpair/env.py ENV_MAP), which runs after this
    return f"noise_sync={v} ({why}) diff_noise={'guard' if v == 'guard' else 'bcast_rank0_state'} sampler_exit=rank0_coordinates(diff_rank_spread_A recorded, refused above 1 A)"
_TP_UNSET: Tuple[str, ...] = ("OF3_CUDA_GRAPHS", "OF3_GRAPHS_STRICT", "OF3T_KIT_LEVERS", "OF3T_TEMPL_DISTINCT", "OF3T_PAIRCACHE", "OF3T_TRIMUL", "OF3T_TRIATT",
               "OF3T_APB", "OF3TP_TEMPL_FIX", "OF3TP_TEMPL_INTERCHAIN")

LINES: Dict[Tuple[str, Optional[str]], Line] = {
    ("exact", "cueq"): Line(
        hooks=("cells", "trunk_kernels", "fast_inference"),
        env={"OF3_FAST_INIT": "1", "OF3_CUDA_GRAPHS": "1", "OF3_GRAPHS_STRICT": "1", "OF3T_TEMPL_DISTINCT": "1", "OF3T_PAIRCACHE": "1", "OPENFOLD3_OPT_ATOM_HOIST": "1",
             "OPENFOLD3_OPT_CASTCACHE": "1", "OPENFOLD3_OPT_EXACTLN": "1", "OPENFOLD3_OPT_APB_HOIST": "1", "OPENFOLD3_OPT_POST_RELEASE": "1", "OPENFOLD3_OPT_SYNC_HOIST": "1", "OPENFOLD3_OPT_FASTJSON": "1",
             "OPENFOLD3_OPT_POSTFWD_MEM": "1", "OPENFOLD3_OPT_LOADER_WORKERS": "1", "OPENFOLD3_OPT_WRITER_OVERLAP": "1", "OPENFOLD3_OPT_HOSTFEAT": "1", "OPENFOLD3_OPT_CKPT_MMAP": "1",    # the runner-side cells: confidence scoring after the forward per (sample, row block), the predict loader's workers capped at the item count (cells/postfwd_mem.py, loader_workers.py)
             "OPENFOLD3_OPT_TRIATT_EXACT": "1", "OPENFOLD3_OPT_TRIMUL_EXACT": "1",
             "OPENFOLD3_OPT_TRANSITION_EXACT": "1"},       # OPENFOLD3_OPT_TRIMUL_EXACT_FORM: trimul_exact's module-statement classes (the template pair stack's c_z 64 TriMul) on the trimul provider's exact tier under the form of3_module (row of3_form), bit-proven per class and token count (cells/trimul_form.py); the library's triangle-attention call / the pair stacks' triangle multiplication on the core's providers, word exact: H100 the bit-exact sm_90a kernel (proven per call class against the library), elsewhere the library by name (cells/triatt_exact.py)
        unset=("OF3T_TRIATT", "OF3T_TRIMUL", "OF3T_APB"),
        kit_levers="fast_inference",
        levers=("fast_init", "cuda_graphs", "graphs_strict", "templ_distinct", "paircache", "atom_hoist", "castcache", "exactln", "apb_hoist", "triatt_exact", "trimul_exact", "transition_exact", "post_release", "sync_hoist", "postfwd_mem", "loader_workers", "fastjson", "writer_overlap", "hostfeat", "ckpt_mmap"),
        stock_kernels=True, graphs_max_tokens=512,
        source="trunk_kernels/README.md (OF3T_TEMPL_DISTINCT, OF3T_PAIRCACHE), :80 (with the kit: OF3T_KIT_LEVERS, OF3_FAST_INIT=1); "
               "fast_inference/README.md (OF3_CUDA_GRAPHS=1 + OF3_GRAPHS_STRICT=1: the graphed sampler — on this line up to 512 tokens (Line.graphs_max_tokens: the fp32 "
               "step is launch-bound below it and GPU-bound above, where the eager DS4Sci sampler is faster); the graphed rollout runs the diffusion module's attention on "
               "the cuEquivariance / torch path instead of DS4Sci (legacy-default-stream kernel, uncapturable) — under --det 1 the line never called DS4Sci: bitwise); "
               "opt/openfold3_opt/cells/{atom_hoist,castcache,exactln,apb_hoist,post_release}.py (the exact-class cells: the atom path's per-rollout memo, memoised bf16 weight "
               "casts, the LayerNorm primitive on the core's exactln row, the memoised key-mask bias, the item-boundary memory release at >= 2400 tokens — each bitwise); "
               "the base is the stock configuration (cli.row_yaml: STOCK_YAML, STOCK_DET_YAML under --det) — stock's own cuEquivariance triangle kernels in the trunk"),
    ("fast", None): Line(
        hooks=("cells", "trunk_kernels", "fast_inference"),
        env={"OF3_FAST_INIT": "1", "OF3_CUDA_GRAPHS": "1", "OF3_GRAPHS_STRICT": "1", "OF3T_TEMPL_DISTINCT": "1", "OF3T_PAIRCACHE": "1",
             "OPENFOLD3_OPT_PAIR": "trimul_v4:triatt_block:pair_transition", "OPENFOLD3_OPT_PAIR_IMPL": "fpf", "OPENFOLD3_OPT_PAIR_CORE": "provider",   # the pair-track cells (bf16-stream rows of the core's cell table); impl fpf = the core's fused prologue/epilogue cells; core provider = the block's attention core per call shape from the core's triangle-attention provider (lever triatt_provider: the line's tier word fast — the cell table's row per GPU, head dim, key count and call form)
             "OPENFOLD3_OPT_PAIR_STRICT": "1", "OPENFOLD3_OPT_TRIMUL_PROVIDER": "1", "OPENFOLD3_OPT_TRIMUL_TIER": "fast",                                        # trimul_provider: the trimul_v4 cell's row per call class from the core's ONE trimul provider (of3_trimul.PREFER; the kit row v4 = the cell's own statement)
             "OPENFOLD3_OPT_ROLLOUT": "bf16", "OPENFOLD3_OPT_DIT": "flash_bias_attn", "OPENFOLD3_OPT_APB_TIER": "fast",
             "OPENFOLD3_OPT_DIT_GLUE": "1", "OPENFOLD3_OPT_TOKEN_AGG": "seg_reduce", "OPENFOLD3_OPT_ATOM_WINDOW": "1", "OPENFOLD3_OPT_ATOM_HOIST": "1",   # the token DiT block schedule, the atom -> token aggregation kernel, the fused atom-attention windows, the atom path's per-rollout memo (cells/dit_glue.py, token_agg.py, atom_window.py, atom_hoist.py)
             "OPENFOLD3_OPT_APB_TRUNK": "1",                                                                       # the pairformer single track's attention with pair bias (cells/apb_trunk.py)
             "OPENFOLD3_OPT_TEMPL_EMBED": "1",                                                                     # the template embedder's fused feature embedding + mean/relu/linear_t tail (cells/templ_embed.py)
             "OPENFOLD3_OPT_CASTCACHE": "1", "OPENFOLD3_OPT_LN_PROVIDER": "1", "OPENFOLD3_OPT_LN_TIER": "fast", "OPENFOLD3_OPT_APB_HOIST": "1", "OPENFOLD3_OPT_TRUNK_GRAPH": "1", "OPENFOLD3_OPT_POST_RELEASE": "1", "OPENFOLD3_OPT_TUNER_GUARD": "1", "OPENFOLD3_OPT_SYNC_HOIST": "1", "OPENFOLD3_OPT_FASTJSON": "1",
             "OPENFOLD3_OPT_POSTFWD_MEM": "1", "OPENFOLD3_OPT_LOADER_WORKERS": "1", "OPENFOLD3_OPT_WRITER_OVERLAP": "1", "OPENFOLD3_OPT_HOSTFEAT": "1", "OPENFOLD3_OPT_CKPT_MMAP": "1",                          # the runner-side cells: confidence scoring after the forward per (sample, row block), the predict loader's workers capped at the item count
                                                                                                                   # the host-side scheduling cells: memoised bf16 weight casts, memoised key-mask bias, the trunk stack captured into a CUDA graph, the item-boundary release at >= 2400 tokens
             **ALLOC_ENV},                                          # the expandable-segments allocator: after the item-boundary release the confidence head's per-sample outputs need unfragmented headroom for one large allocation
        unset=("OF3T_APB", "OF3T_TRIATT", "OPENFOLD3_OPT_PAIR_LN"),   # OPENFOLD3_OPT_PAIR_LN: the cells' LayerNorm placement stays the module default (fused)
        kit_levers="fast_inference",
        levers=("fast_init", "cuda_graphs", "graphs_strict", "templ_distinct", "paircache",
                "trimul_v4", "trimul_provider", "triatt_block", "triatt_provider", "pair_transition", "rollout_bf16", "dit_attn", "dit_glue", "token_agg", "atom_window", "atom_hoist", "apb_trunk", "templ_embed",
                "castcache", "ln_provider", "apb_hoist", "trunk_graph", "post_release", "tuner_guard", "sync_hoist", "alloc_expandable", "postfwd_mem", "loader_workers", "fastjson", "writer_overlap", "hostfeat", "ckpt_mmap"),
        source="trunk_kernels/README.md (of3t_hook on PYTHONPATH, the default set), :80 (with the kit: OF3T_KIT_LEVERS, OF3_CUDA_GRAPHS=1 OF3_FAST_INIT=1), :7; "
               "fast_inference/README.md (OF3_GRAPHS_STRICT=1: every graphed line of this tree); "
               "PYTHONPATH=trunk_kernels/of3t_hook:fast_inference/of3_levers, OF3T_KIT_LEVERS=fast_inference/of3_levers"),
    ("big", "tp"): Line(
        hooks=("cells", "tp_rowpair", "fast_inference"),                   # hooks/cells first: the predict loader's worker cap in every rank (loader_workers; the row-sharded line scores confidence itself — postfwd_mem's engine functions are not on its path)
        env=dict(_TP_ENV, OPENFOLD3_OPT_FASTJSON="1", OPENFOLD3_OPT_LOADER_WORKERS="1", OPENFOLD3_OPT_HOSTFEAT="1", OPENFOLD3_OPT_CKPT_MMAP="1"),
        unset=_TP_UNSET,
        kit_levers=None,
        levers=("fast_init", "tp_shard_s", "sample_loop", "tp_triatt", "tp_trimul", "loader_workers", "fastjson", "hostfeat", "ckpt_mmap"),
        runner_yaml=BIG_BF16_C16_YAML, source="opt/openfold3_opt/tp_rowpair/__init__.py (the adapter: OpenFold3's modules bound as callables onto opt_core.mem.rowpair's row statements — the pair "
               "representation born row-sharded, template / MSA / Pairformer / confidence pair blocks through the core's one pair-block driver, distogram and confidence "
               "heads row-blocked with the exact-finish reducer and the output matrices assembled on rank 0's host, diffusion conditioning rows + local-query "
               "DiffusionTransformer); opt/openfold3_opt/tp_rowpair/hook/sitecustomize.py (the entry hook: the deterministic recipe, installs the adapter on import of "
               "the runner when OF3TP_WORLD > 1, chain-loads the fast-inference kit's of3_levers, OF3_FAST_INIT=1 per fast_inference/README.md); the env "
               "= the row-block sizes (TRIATT/APB QBLOCK+ROWBLOCK 256 and TRIMUL_SUB 128 — attention and triangle transients bounded in row blocks; tp_rowpair/env.py "
               "maps each OF3TP_* size onto the core's ROWPAIR_* lever, opt_core/mem/rowpair/API.md) over the line's fixed placements (the [4,N,N,*] dummy template "
               "features never materialised; shards parked or streamed between phases; the core's fused triangle-attention and triangle-multiplication kernels on the "
               "row blocks — opt_core.mem.rowpair triatt.attention_core flash_triattn / trimul_fused fpf_v4, the torch statements their named fallbacks; raw MSA and bond "
               "features host-resident; tp_rowpair/env.py LINE_CONSTANTS); the launcher, chunk plan, census and refusals: tp.py. CONFLICTS (why the other add-ons' levers are unset, "
               "not composed): the sharded pair stack replaces the trunk's pair-stack, MSA, template and confidence modules — the trunk-kernels add-on's pair cache and "
               "template de-duplication (OF3T_PAIRCACHE, OF3T_TEMPL_DISTINCT), its triangle kernels (OF3T_TRIMUL/TRIATT/APB) "
               "patch those replaced modules and would never run or would run on unsharded rows; the CUDA-graphed diffusion step (OF3_CUDA_GRAPHS) captures a sampler "
               "the sharded rollout replaces."),
    ("off", None): Line(hooks=(), env={}, unset=(), kit_levers=None, levers=(), source="stock: no switch, no hook directory"),
    # the memory mode, one GPU: the offload port (opt/forward/offload; the switches as its README's O1 line states them)
    # UNDER the fast line's kernel and sampler cells — big = fast minus ONLY the levers with a device-memory cost, plus the memory
    # levers (the tier rule). Carried from `fast`, each memory-neutral or better across sizes on this chain (CHANGES.md,
    # section `big`): the trunk-kernels routes `OF3T_TEMPL_DISTINCT=1
    # the pair-track cells `trimul_v4:triatt_block:pair_transition` on the core's providers by the tier word `big` (`trimul_provider`,
    # `triatt_provider`), `apb_trunk`, `templ_embed`, `apb_hoist`, `post_release`
    # (inert below its 2400-token gate; a release above it), `tuner_guard`, `sync_hoist`, the sampler cells `rollout_bf16` `dit_attn` `token_agg`
    # `atom_window`, the runner-side cells and the allocator policy. ABSENT FOR THEIR DEVICE-MEMORY COST on this chain: `cuda_graphs` +
    # `graphs_strict` (the graph pools), `paircache`, `atom_hoist`, `dit_glue` (its block tables), `castcache`, `trunk_graph` (its pool, AND structurally foreign:
    # the port's host-snapshot unit synchronises inside the captured region — cudaErrorStreamCaptureInvalidated); the graph family and the pair cache are required unset, an absent cell's switch preset by a caller is refused by name.
    # the LayerNorm binding rides this line as `ln_provider` with the word `big` (the provider's lowest-peak row inside each cell's band; memory-neutral by the table's own peak rule).
    # this line's eager sampler its served per-step classes (bf16 / fp32 C 128 pair norms) cost more than ATen per call on small inputs
    # and much less on large ones — below the gate every call keeps the statement
    # (`kept_by=gated:lt_min`); it binds outermost over the port's LN-SAFE guard (OF3O_LNSAFE, the same primitive rebound; `over=_ln_safe_forward.forward`)
    # and never hands the replica an operand of >= 2^31 elements (`kept_by=ge2p31`: the guard's domain).
    # THE PORT'S ITEM GATE (of3o_gate, OF3O_MIN_TOKENS): below it every O1 unit of this line steps aside by name and the call runs the fast line's
    # composition minus the graph family and the levers absent for memory (of3o_aside) — the memory line is never slower than stock where stock fits.
    ("big", "resident"): Line(
        hooks=("confhead", "offload", "cells", "trunk_kernels", "fast_inference"),
        env={"OPENFOLD3_OPT_CONFHEAD": "1", "OF3_FAST_INIT": "1", "OF3O_LAYER": "1", "OF3O_ROWS": "128", "OF3O_COND_ROWS": "128", "OF3O_INPUT_ROWS": "128", "OF3O_EMBED_ROWS": "1",
             "OF3O_TEMPL_ROWS": "128", "OF3O_TEMPL_EMBED_ROWS": "128", "OF3O_RECYCLE_ROWS": "128", "OF3O_CONF_MODE": "chunked", "OF3O_CONF_TM_BACKEND": "blockreduce",
             "OF3O_PIN_BUDGET_GB": "600", "OF3O_PIN_POLICY": "census", "OF3O_LNSAFE": "1", "OF3O_TEMPL_FIX": "0", "OPENFOLD3_OPT_FASTJSON": "1",
             "OPENFOLD3_OPT_POSTFWD_MEM": "1", "OPENFOLD3_OPT_LOADER_WORKERS": "1", "OPENFOLD3_OPT_WRITER_OVERLAP": "1", "OPENFOLD3_OPT_HOSTFEAT": "1", "OPENFOLD3_OPT_CKPT_MMAP": "1",                                                       # the runner-side cells (confidence scoring after the forward per (sample, row block) below the confidence gate — above it the port's chunked scorer serves — and the predict loader's workers capped at the item count)
             "OF3T_TEMPL_DISTINCT": "1",                         # the trunk-kernels add-on's exact lever (trunk_kernels/README.md)
             "OPENFOLD3_OPT_PAIR": "trimul_v4:triatt_block:pair_transition", "OPENFOLD3_OPT_PAIR_IMPL": "fpf", "OPENFOLD3_OPT_PAIR_CORE": "provider:big", "OPENFOLD3_OPT_PAIR_STRICT": "1", "OPENFOLD3_OPT_TRIMUL_PROVIDER": "1", "OPENFOLD3_OPT_TRIMUL_TIER": "big",   # the fast line's pair cells minus trimul_v4 (absent by number: the cuEquivariance TriMul route + the port's host-snapshot unit serve)
             "OPENFOLD3_OPT_APB_TRUNK": "1", "OPENFOLD3_OPT_TEMPL_EMBED": "1", "OPENFOLD3_OPT_LN_PROVIDER": "1", "OPENFOLD3_OPT_LN_TIER": "big",                  # the LayerNorm primitive on the core's provider by the word big (the lowest-peak row inside each cell's band)
             "OPENFOLD3_OPT_APB_HOIST": "1", "OPENFOLD3_OPT_POST_RELEASE": "1", "OPENFOLD3_OPT_TUNER_GUARD": "1", "OPENFOLD3_OPT_SYNC_HOIST": "1",
             "OPENFOLD3_OPT_ROLLOUT": "bf16", "OPENFOLD3_OPT_DIT": "flash_bias_attn", "OPENFOLD3_OPT_APB_TIER": "big", "OPENFOLD3_OPT_TOKEN_AGG": "seg_reduce", "OPENFOLD3_OPT_ATOM_WINDOW": "1",   # the sampler cells minus dit_glue and atom_hoist (absent by number)
             **ALLOC_ENV},
        unset=("OF3_CUDA_GRAPHS", "OF3_GRAPHS_STRICT", "OF3T_PAIRCACHE", "OF3O_CHUNK", "OF3T_APB", "OF3T_TRIATT", "OPENFOLD3_OPT_PAIR_LN"),   # the graph family and the pair cache stay off this line (their memory cost); an absent cell's switch preset by a caller (trunk_graph, dit_glue, atom_hoist, castcache, trimul_v4's list) is refused by name — the cells' family gate
        kit_levers=None,                                                                                                              # every hook chains the next (CHAIN_ENV): confhead > of3o > cells > of3t_hook > of3_levers
        levers=("fast_init", "trimul_hostsnap", "triatt_lean", "trans_inplace", "cond_once", "input_rows", "recycle_rows", "templ_host", "conf_chunked", "confhead", "bigln_guard", "host_pool",
                "templ_distinct", "trimul_v4", "trimul_provider", "triatt_block", "triatt_provider", "pair_transition", "rollout_bf16", "dit_attn", "token_agg", "atom_window",
                "apb_trunk", "templ_embed", "ln_provider", "apb_hoist", "post_release", "tuner_guard", "sync_hoist", "alloc_expandable", "postfwd_mem", "loader_workers", "fastjson", "writer_overlap", "hostfeat", "ckpt_mmap"),
        runner_yaml=BIG_BF16_C16_YAML, source="offload/README.md §Lines (O1: env.sh o1 of the add-on as shipped — OF3O_LAYER=1 and the row switches; the port sets the pin policy strict, the template fix off) "
               "and §Composition seams (the O1 units under another hook chain: tri_att_end_lean reads the trunk-kernels dispatcher); the confidence phase over the port's chunked scorer with the TM backend named "
               "(OF3O_CONF_MODE=chunked, OF3O_CONF_TM_BACKEND=blockreduce: offload/of3o/of3_offload.py `wrap_confidence_scores_cpu` -> of3o_confidence.get_confidence_scores_chunked); of3_offload.py `apply_core`, `apply_runner`; "
               "opt/openfold3_opt/confhead.py + hooks/confhead/sitecustomize.py (OPENFOLD3_OPT_CONFHEAD=1: the row-block head, first in the chain, OPENFOLD3_OPT_CONFHEAD_CHAIN -> of3o); "
               "trunk_kernels/README.md (the default set) and the fast line's cells (opt/openfold3_opt/cells/*.py, hooks/cells) minus the levers with a measured memory cost (CHANGES.md: the per-lever memory table); "
               "fast_inference/README.md (OF3_FAST_INIT=1)",
        kit_levers_env=OFFLOAD_KIT_LEVERS_ENV),
}


# Switch families the units as shipped read (OF3_, OF3T_, OF3FPF_, OF3FLASHPF_, OF3TP_; OF3O_: the offload add-on's, opt/forward/offload) and one
# reserved family no shipped unit reads (BFTP_); `unset` above and stock/PINS.json
# "must_be_absent_prefixes" are drawn from them.
SWITCH_PREFIXES: Tuple[str, ...] = ("OF3_", "OF3T_", "OF3FPF_", "OF3FLASHPF_", "OF3TP_", "BFTP_", "OF3O_")
# Names in those families that are not levers: the deterministic recipe (det.py: level 1's switch) and the extension build cache directory (configs/<gpu>.env).
NOT_LEVERS: Tuple[str, ...] = ("OF3_DETERMINISTIC", "OF3O_MIN_TOKENS") + RUN_VARS   # + the offload port's item-gate knob (of3o_min_tokens): a caller's to set under big
# The offload port's UNIT switches (apply_core, each O1 install its own unit, default on; not in any line — a composition on another hook
# chain turns single units off): OF3O_TRIMUL OF3O_TRIATT OF3O_TRANS OF3O_COND OF3O_INPUT OF3O_TEMPL OF3O_RUN_TRUNK OF3O_FORWARD.
OFFLOAD_UNIT_SWITCHES: Tuple[str, ...] = ("OF3O_TRIMUL", "OF3O_TRIATT", "OF3O_TRANS", "OF3O_COND", "OF3O_INPUT", "OF3O_TEMPL", "OF3O_RUN_TRUNK", "OF3O_FORWARD")

# ---------------------------------------------------------------------------------------- MODEL_OPT_LEVERS_OFF: the ablation door ----
ENV_LEVERS_OFF = "MODEL_OPT_LEVERS_OFF"
"""One word across the model-opt kits: a comma list of lever names to leave OFF for one run — an ablation of the selected mode's line, never a
shipping configuration (no configs/<gpu>.env sets it). Resolved once, in ``resolve()`` (``apply_levers_off``): every named lever the line carries
is removed from the line BEFORE anything is exported — its switches (``LEVER_SWITCHES``) are not set and are required unset (a caller cannot
re-arm it behind the mode: the existing conflict rule), the pair cells are dropped from ``OPENFOLD3_OPT_PAIR``'s value, a hook directory that
only those switches arm leaves the chain (``HOOK_ARMING``: the next hook leads), the lever leaves ``levers_requested``; the ACTIVE / DRY-RUN /
exit lines carry ``levers_off=<names>`` and each such lever's LEVER line reads ``state=off reason=levers_off``. A name the line does not
carry is refused by name (NOT ACTIVE, exit 3) listing the line's levers; a lever that cannot leave on its own is refused by name with the
reason (``LEVERS_OFF_STUCK``); a lever another one rides on takes it along, named (``LEVERS_OFF_TAKES``); ``compile`` is a known no-op word on
every line (``LEVERS_OFF_NOOP``: the spelling behind ``--no-compile``; no mode of this kit compiles, compile=none); ``--mode off`` ignores the variable
(a note). Unset or empty = the line as written, byte for byte (tests/test_lines_unchanged.py, test_modes.py)."""
LEVERS_OFF_TAKES: Dict[str, Tuple[str, ...]] = {"cuda_graphs": ("graphs_strict",), "conf_chunked": ("confhead",), "triatt_block": ("triatt_provider",), "trimul_v4": ("trimul_provider",), "trimul_exact": ("trimul_form",)}
"""Switching the key off takes the named levers with it: graphs_strict is the capture's failure policy (nothing to be strict about without the
capture); the row-block confidence head consumes the chunked scorer's row blocks; the providers' row choice rides on the cell it serves (triatt_block / trimul_v4)."""
LEVERS_OFF_STUCK: Dict[str, str] = {
    "graphs_strict": "it is the capture-failure policy of cuda_graphs and the deterministic recipe re-asserts it on every graphed call (det.GRAPHED_EXTRA) — "
                     "switch cuda_graphs off instead (it takes graphs_strict with it)",
    **{n: "the offload port's O1 units default ON inside the port (OFFLOAD_UNIT_SWITCHES) and its row sizes are the line's constants — the big/resident line is "
          "applied as composed (its cells, the trunk-kernels routes, fast_init, conf_chunked, confhead and alloc_expandable are its switchable levers)"
       for n in ("trimul_hostsnap", "triatt_lean", "trans_inplace", "cond_once", "input_rows", "recycle_rows", "templ_host", "bigln_guard", "host_pool")},
    **{n: "it has no switch of its own: on with the big/tp line (the row-sharded pair stack is applied as composed; the core's documented opt-outs "
          "ROWPAIR_TRIATT_CORE=torch / ROWPAIR_TRIMUL_KERNELS=torch are the tp line's words for its two kernel levers)"
       for n in ("tp_shard_s", "sample_loop", "tp_triatt", "tp_trimul")},
}
"""lever -> why this door cannot switch it off on its own (refused by name with this reason)."""
PAIR_CELL_LEVERS: Tuple[str, ...] = ("trimul_v4", "triatt_block", "pair_transition")   # spelled as members of OPENFOLD3_OPT_PAIR's value (cells/pairfused.py ENV_LEVERS: names joined by ':'); the last one to leave takes the PAIR_ENVS family
LEVER_SWITCHES: Dict[str, Tuple[str, ...]] = {                                          # lever -> the line exports that are its alone: dropped with it and required unset (the allocator variable is dropped, never required unset: a caller's own allocator keys stay theirs)
    "fast_init": ("OF3_FAST_INIT",), "cuda_graphs": GRAPHS_ENVS, "graphs_strict": ("OF3_GRAPHS_STRICT",),
    "templ_distinct": ("OF3T_TEMPL_DISTINCT",), "paircache": ("OF3T_PAIRCACHE",), "trimul_cueq": ("OF3T_TRIMUL",),
    "rollout_bf16": ROLLOUT_ENVS, "dit_attn": DIT_ENVS, "dit_glue": DIT_GLUE_ENVS, "apb_trunk": APB_TRUNK_ENVS, "token_agg": TOKEN_AGG_ENVS, "atom_window": ATOM_WINDOW_ENVS,
    "atom_hoist": ATOM_HOIST_ENVS, "templ_embed": TEMPL_EMBED_ENVS, "castcache": CASTCACHE_ENVS, "exactln": EXACTLN_ENVS, "ln_provider": LN_PROVIDER_ENVS, "apb_hoist": APB_HOIST_ENVS, "trunk_graph": TRUNK_GRAPH_ENVS,
    "post_release": POST_RELEASE_ENVS, "tuner_guard": TUNER_GUARD_ENVS, "sync_hoist": SYNC_HOIST_ENVS, "fastjson": FASTJSON_ENVS, "writer_overlap": WRITER_OVERLAP_ENVS, "hostfeat": HOSTFEAT_ENVS, "ckpt_mmap": CKPT_MMAP_ENVS, "postfwd_mem": POSTFWD_MEM_ENVS, "loader_workers": LOADER_WORKERS_ENVS, "alloc_expandable": tuple(ALLOC_ENV),
    "triatt_exact": TRIATT_EXACT_ENVS, "trimul_exact": TRIMUL_EXACT_ENVS, "trimul_form": TRIMUL_EXACT_ENVS[1:], "transition_exact": TRANSITION_EXACT_ENVS, "triatt_provider": ("OPENFOLD3_OPT_PAIR_CORE",), "trimul_provider": (TRIMUL_PROVIDER_ENVS[0], TRIMUL_PROVIDER_ENVS[2]),           # triatt_provider: the fused block's core word alone (left off, pair_fused's default flash core serves the block — the line before it)
    "confhead": (ENV_CONFHEAD,), "conf_chunked": ("OF3O_CONF_TM_BACKEND",),              # conf_chunked: and OF3O_CONF_MODE rewritten to the port's stock scorer (CONF_ENVS_BELOW), as below the confidence gate
}
LEVERS_OFF_KEPT_SETTABLE: Tuple[str, ...] = tuple(ALLOC_ENV)                            # dropped from the exports when their lever leaves, but not required unset
COMPILE_STATE = "none"
"""The `compile=` word of the ACTIVE / DRY-RUN lines (report.compile_field), one across the model-opt kits (`on | off:user | stepped_aside:<reason> |
none`): `none` on every mode of this kit — stock `run_openfold predict` calls no torch.compile / TorchScript on its inference path and no lever of
this kit adds one (exact never would), so there is no compile behaviour to inherit, switch or step aside."""
LEVERS_OFF_NOOP: Dict[str, str] = {
    "compile": f"nothing to leave off: no mode of this kit compiles anything (stock compiles nothing, the modes inherit that; compile={COMPILE_STATE})",
}
"""Names the door accepts on any line as KNOWN NO-OPS — named in a note, never a conflict: `compile` is the word behind `--no-compile`, the
model-opt kits' one opt-out of a kit-added compile lever (an alias of MODEL_OPT_LEVERS_OFF=compile, cli.apply_no_compile_flag); this kit
carries no such lever, so the word changes nothing (the line stays byte for byte) and the ACTIVE line says compile=none with or without it."""
HOOK_ARMING: Dict[str, Tuple[str, ...]] = {"fast_inference": ("OF3_FAST_INIT", "OF3_CUDA_GRAPHS"), "cells": tuple(fam[0] for fam in CELL_FAMILIES if fam not in ACTIVATION_FAMILIES), "confhead": (ENV_CONFHEAD,)}
"""The hooks that install their finder only when one of these exports is set (of3_levers/sitecustomize.py, hooks/cells/sitecustomize.py `_ON`,
hooks/confhead): left with none of them, the hook leaves the chain and the next hook leads (hooks.missing would otherwise name it). The
trunk-kernels hook (the per-item model timer), the offload port's and the tp line's install unconditionally and never leave."""


def levers_off_names(environ: Optional[Mapping[str, str]] = None) -> Tuple[str, ...]:
    """``MODEL_OPT_LEVERS_OFF`` as names: comma list, blanks dropped, order kept, duplicates dropped; () when unset or empty."""
    environ = os.environ if environ is None else environ
    out: List[str] = []
    for w in (environ.get(ENV_LEVERS_OFF) or "").split(","):
        w = w.strip()
        if w and w not in out:
            out.append(w)
    return tuple(out)


def levers_off_ignored(environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """The note of `--mode off` when the variable is set (stock has no lever to leave off), None when unset."""
    environ = os.environ if environ is None else environ
    if not levers_off_names(environ):
        return None
    return f"{ENV_LEVERS_OFF}={environ.get(ENV_LEVERS_OFF)!r} ignored: mode off runs stock (no lever to leave off)"


def apply_levers_off(res: "Resolution", ln: "Line", environ: Mapping[str, str]) -> None:
    """Apply ``MODEL_OPT_LEVERS_OFF`` to a Resolution fresh from its line (exports = the line's switches, levers = the line's list, hooks = the
    line's chain): the named levers leave `levers`, their switches leave `exports` and join `unsets` (LEVERS_OFF_KEPT_SETTABLE excepted), a pair
    cell leaves OPENFOLD3_OPT_PAIR's value (the last one takes the family), conf_chunked leaves the port on its stock scorer, a hook nothing arms
    any more leaves `hooks`; `levers_off` and one note record it. A name outside the line or a stuck lever becomes a conflict (the caller's NOT
    ACTIVE, exit 3) and nothing is rewritten. No-op when the variable is unset/empty; a note only under mode off."""
    names = levers_off_names(environ)
    if not names:
        return
    raw = environ.get(ENV_LEVERS_OFF)
    if res.mode == "off":
        res.notes.append(levers_off_ignored(environ))
        return
    noop = [n for n in names if n in LEVERS_OFF_NOOP]
    if noop:                                                                                # known no-op words (compile): named, nothing rewritten, never a conflict
        res.notes.extend(f"{ENV_LEVERS_OFF}: {n} — {LEVERS_OFF_NOOP[n]}" for n in noop)
        names = tuple(n for n in names if n not in LEVERS_OFF_NOOP)
        if not names:
            return
    sel = f"{res.mode}{'/' + res.line if res.line else ''}"
    line_levers = tuple(ln.levers)
    unknown = [n for n in names if n not in line_levers]
    if unknown:
        res.conflicts.append(f"{ENV_LEVERS_OFF}={raw!r} names {','.join(unknown)}: not a lever of the {sel} line (its levers: {','.join(line_levers)})")
        return
    stuck = [n for n in names if n in LEVERS_OFF_STUCK]
    if stuck:
        res.conflicts.extend(f"{ENV_LEVERS_OFF}={raw!r}: lever {n} of the {sel} line cannot be switched off on its own — {LEVERS_OFF_STUCK[n]}" for n in stuck)
        return
    asked = set(names)
    for n in names:
        asked.update(t for t in LEVERS_OFF_TAKES.get(n, ()) if t in line_levers)
    off = [n for n in line_levers if n in asked]                                            # the line's order
    drop: List[str] = []
    for n in off:
        if n not in PAIR_CELL_LEVERS:
            drop += [k for k in LEVER_SWITCHES.get(n, ()) if k not in drop]
    if any(n in PAIR_CELL_LEVERS for n in off):
        left = [n for n in PAIR_CELL_LEVERS if n in line_levers and n not in off]
        if left and PAIR_ENVS[0] in res.exports:
            res.exports[PAIR_ENVS[0]] = ":".join(left)                                      # cells/pairfused.py reads the names joined by ':'
        else:
            drop += [k for k in PAIR_ENVS if k not in drop]
    if "conf_chunked" in off:
        for k, v in CONF_ENVS_BELOW.items():
            if k in res.exports:
                res.exports[k] = v                                                          # the port's stock scorer, as below the confidence gate
    for k in drop:
        res.exports.pop(k, None)
        if k not in res.unsets and k not in LEVERS_OFF_KEPT_SETTABLE:
            res.unsets.append(k)
    res.levers = [l for l in res.levers if l not in off]
    res.hooks = [h for h in res.hooks if h not in HOOK_ARMING or any(k in res.exports for k in HOOK_ARMING[h])]
    res.levers_off = list(off)
    taken = [n for n in off if n not in names]
    res.notes.append(f"{ENV_LEVERS_OFF}: {','.join(off)} left off" + (f" ({','.join(taken)} taken along)" if taken else "")
                     + f" — an ablation of the {sel} line, not a shipping configuration")

# Switches and add-ons no house mode carries (what the add-ons document as opt-in, rejected, unshipped or uncarried); data for `check`.


@dataclass
class Resolution:
    mode: str
    line: Optional[str]                                   # the exact line name, None for fast/off
    tier: str
    hooks: List[str] = field(default_factory=list)        # kit keys in PYTHONPATH order
    hook_dirs: List[str] = field(default_factory=list)    # absolute hook directories in the same order
    entry_hook: Optional[str] = None                      # the sitecustomize.py that is executed (the first hook); it chains the rest
    exports: Dict[str, str] = field(default_factory=dict) # every variable the mode sets (incl. OF3T_KIT_LEVERS)
    unsets: List[str] = field(default_factory=list)       # variables the mode requires absent
    levers: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)    # caller-preset switches that contradict the line (activation refuses)
    notes: List[str] = field(default_factory=list)
    source: str = ""
    size_gate: Optional[str] = None                       # the graph gate's fragment for this call (opt_core.mem.graph_gate: graph=capture|eager:<reason>|off), None when the line has no graphs
    gate_reason: Optional[str] = None                     # the gate's reason word (within_cap|no_cap|n_tok>cap|off|n_tok_unknown) — printed once as gate=<reason> on the cuda_graphs LEVER line
    precision: Optional[str] = None                       # the fast mode's precision word (FAST_PRECISION); under big the line's runner-yaml precision (bf16-mixed); None otherwise
    conf_gate: Optional[str] = None                       # the confidence levers' size-gate fragment (conf=served|gated:lt_min|n_tok_unknown min_tokens=N), None when the line carries none of CONF_LEVERS
    conf_reason: Optional[str] = None                     # its decision word alone (served | gated:lt_min | n_tok_unknown) — printed as gate=<word> on the CONF_LEVERS' LEVER lines
    n_tokens: Optional[int] = None                        # the query's polymer token count the gate read (inputs.polymer_tokens), None when unknown
    levers_off: List[str] = field(default_factory=list)   # the line's levers MODEL_OPT_LEVERS_OFF left off for this call (apply_levers_off), in the line's order; [] = the line as written
    reach_gate: Optional[str] = None                      # the reach gate's fragment (reach=aside:n_tok>=<gate> min_tokens=<gate>(<source>)) when it set levers aside on this call, else None (modes.reach_gate)
    reach_reason: Optional[str] = None                    # its decision word alone (aside:n_tok>=<gate>) -- printed as gate=<word> on the REACH_LEVERS' LEVER lines
    reach_min_tokens: Optional[int] = None                # the gate the decision read (REACH_GATE_BY_CARD / REACH_GATE_ENV), None when the gate did not apply
    reach_off: List[str] = field(default_factory=list)    # the REACH_LEVERS the gate set aside for this call, in the line's order; [] = the line as written
    of3o_gate: Optional[str] = None                       # the offload port's item-gate fragment (of3o_gate=engaged|aside:lt_min|n_tok_unknown|no_gate min_tokens=N(source)), None when the line has no port
    of3o_reason: Optional[str] = None                     # its decision word alone — printed as gate=<word> on the OF3O_UNIT_LEVERS' LEVER lines
    of3o_min_tokens: Optional[int] = None                 # the gate value the decision used (None = `none` / no port)


def unknown_mode_message(mode) -> str:
    """The sentence of an unknown mode name (the one spelling: the table's `unknown_message` and the empty-name refusal)."""
    return f"unknown mode {mode!r}; expected one of {'|'.join(MODES)}"


def table():
    """The mode table as opt_core.modes.ModeTable (MODES, DEFAULT_MODE; the unknown-mode sentence is this package's)."""
    from opt_core.modes import ModeTable
    return ModeTable(modes=MODES, default=DEFAULT_MODE, unknown_message=unknown_mode_message)


def check_mode(mode: str) -> str:
    """The normalised mode name; ValueError (opt_core.modes.ModeError) on a name outside MODES — an empty name is refused too (the default
    is the CLI's, cli.resolve_mode)."""
    if not (mode or "").strip():
        raise ValueError(unknown_mode_message(mode))
    return table().check(mode)


def stock_kernel_line(ln: Line) -> bool:
    """A line whose base is the stock configuration (stock's cuEquivariance triangle kernels and DS4Sci attention on the runner yaml): the exact
    lines. Their graphed sampler runs over those kernels (DS4Sci off inside the rollout only), so a caller's yaml under them is not pinned
    kernels-off (runner_yaml.line_pins)."""
    return bool(ln.stock_kernels)


def line_graphed(ln: Line) -> bool:
    """Whether a line's switches capture the diffusion sampler in CUDA graphs (OF3_CUDA_GRAPHS=1). The graphed step cannot capture upstream's
    DS4Sci evoformer attention (a legacy-default-stream kernel; the rollout runs the cuEquivariance / torch path instead, named); a graphed line
    over the kernels-off base (fast) composes a caller's runner yaml with the alternative kernel flags pinned off, the override named
    (runner_yaml.line_pins), the exact line (stock_kernel_line) keeps stock's kernels."""
    return ln.env.get("OF3_CUDA_GRAPHS") == "1"


def graphed(mode: str, line: Optional[str] = None) -> bool:
    """Whether the line AS WRITTEN captures CUDA graphs (line_graphed): the det recipe re-asserts OF3_GRAPHS_STRICT=1 on graphed lines only
    (det.GRAPHED_EXTRA); the big lines as written run eager and declare OF3_GRAPHS_STRICT unset — below the port's item gate the resident line runs
    the fast line, graphs included: graphed_call (per call, by token count) says so."""
    mode = check_mode(mode)
    if mode == "off":
        return False
    return line_graphed(LINES[(mode, line)])


def graphed_call(mode: str, line: Optional[str] = None, n_tokens: Optional[int] = None, environ: Optional[Mapping[str, str]] = None) -> bool:
    """Whether a call under (mode, line) with this token count captures CUDA graphs: the line carries them and the size gate (graphs_gate:
    opt_core.mem.graph_gate) does not decide eager/off. The det recipe's OF3_GRAPHS_STRICT export follows this."""
    if n_tokens is not None and int(n_tokens) <= 0:
        n_tokens = None                                            # no polymer tokens counted: unknown (resolve does the same)
    ln = effective_line(check_mode(mode), line, environ, n_tokens)                            # the line as this call runs it: (big, resident) below the port's item gate = the fast line, graphs included (of3o_aside)
    if "cuda_graphs" not in ln.levers or "cuda_graphs" in levers_off_names(environ):        # MODEL_OPT_LEVERS_OFF=cuda_graphs: the call runs the sampler eager (apply_levers_off)
        return False
    cap = graphs_cap(environ, line_graphs_cap(ln))
    d = graphs_gate(ln.levers, n_tokens, cap)
    if d is None:
        return not (ln.graphs_max_tokens is not None and n_tokens is None and cap not in (None, 0))   # a line capped by default runs eager without a token count
    return bool(d.capture)


def line_graphs_cap(ln: "Line") -> Optional[int]:
    """The cap graphs_cap falls back to for this line when OPENFOLD3_OPT_GRAPHS_MAX_TOKENS is unset: the line's own (Line.graphs_max_tokens) or the
    tree-wide GRAPHS_MAX_TOKENS (no cap)."""
    return ln.graphs_max_tokens if ln.graphs_max_tokens is not None else GRAPHS_MAX_TOKENS


def line_arg(mode: str, environ: Optional[Mapping[str, str]] = None, n_gpu=None) -> Optional[str]:
    """The line a mode runs — never a caller's choice: `exact` has one line (``DEFAULT_EXACT_LINE``); under `big` the GPU count selects it
    (`--n_gpu 1` / unset = `resident`, `--n_gpu P>1` or ``OPENFOLD3_OPT_N_GPU`` = `tp`); None for a mode without lines."""
    mode = check_mode(mode)
    if mode not in MODE_LINES:
        return None
    if mode == "exact":
        return DEFAULT_EXACT_LINE
    return "tp" if n_gpu_of(n_gpu, environ, strict=False) > 1 else DEFAULT_BIG_LINE


def n_gpu_of(n_gpu=None, environ: Optional[Mapping[str, str]] = None, strict: bool = True) -> int:
    """P of this call: the `--n_gpu` argument, else OPENFOLD3_OPT_N_GPU, else 1 — a positive integer (opt_core.mem.ngpu.check_n_gpu) in
    N_GPU_VALUES, refused otherwise by name (ValueError). `strict=False` returns 1 for an unreadable value (the refusal is the caller's)."""
    from opt_core.mem import ngpu
    environ = os.environ if environ is None else environ
    prm = LINE_PARAMS[("big", "tp")]
    raw = n_gpu if n_gpu not in (None, "") else ((environ.get(prm.env) or "").strip() or prm.default)
    try:
        p = ngpu.check_n_gpu(raw)
    except ValueError as e:
        if not strict:
            return 1
        raise ValueError(f"{prm.flag} / {prm.env}: {e}") from None
    if str(p) not in N_GPU_VALUES:
        if not strict:
            return p
        raise ValueError(f"refused: n_gpu={p} is not a GPU count this kit runs ({prm.flag} P / {prm.env}=P, P in {'|'.join(N_GPU_VALUES)}; "
                         f"P > 1 = the big/tp line's row-sharded pair stack)")
    return p


def check_n_gpu_route(mode: str, line: Optional[str], n_gpu=None, environ: Optional[Mapping[str, str]] = None) -> int:
    """The n_gpu rules of a (mode, line) selection, refused by name: P > 1 needs `--mode big` (opt_core.mem.ngpu.refuse_unless_big: the
    exact sentence; sharded reductions reorder sums, so the run is fast-class, never bitwise) and runs the tp line only; the tp line needs
    P > 1. Returns P."""
    from opt_core.mem import ngpu
    p = n_gpu_of(n_gpu, environ)
    try:
        ngpu.refuse_unless_big(p, check_mode(mode))
    except ngpu.NGpuRefused as e:
        raise ValueError(str(e.reason)) from None
    if p > 1 and line not in TP_LINES:
        raise ValueError(f"refused: n_gpu={p} runs the big/tp line (the row-sharded pair stack); the selected line is big/{line}, a one-GPU line")
    if p == 1 and check_mode(mode) == "big" and line in TP_LINES:
        raise ValueError(f"refused: the big/{line} line runs on --n_gpu P GPUs, P in {'|'.join(LINE_PARAMS[('big', line)].values)} (got n_gpu=1)")
    return p


def tp_settable() -> set:
    """The row-sharded line's caller-settable names: the `OF3TP_*` sizes and budgets its statements read (tp_rowpair/env.py ENV_MAP / KIT_SWITCHES)."""
    from .tp_rowpair import env as _tpenv
    return set(_tpenv.ENV_MAP) | set(getattr(_tpenv, "KIT_SWITCHES", ()))


def foreign_params(mode: str, line: Optional[str], args: Optional[Mapping[str, object]] = None, environ: Optional[Mapping[str, str]] = None) -> List[str]:
    """A line's parameter set while the selection is NOT its owning (mode, line) — another line of the mode, another mode, `off`: named
    refusals (one per parameter), never a silent ignore — a caller who asks for `--n_gpu 4` under exact or the resident line asked for 4
    GPUs and would get one. `args` = the CLI's values by parameter name (--n_gpu → "n_gpu"); `environ` = the variables (OPENFOLD3_OPT_N_GPU).
    `--n_gpu 1` is every route's and never foreign. The owning line's own check (check_tp_world) covers the parameter's value. The env/.pth route with no mode or `off` selected is the hook's
    gate (_autoload.LINE_PARAMS: the same refusal before anything is imported)."""
    environ = os.environ if environ is None else environ
    selected = f"{mode}/{line}" if line else mode
    own = {prm.name for (m, owner), prm in LINE_PARAMS.items() if (m, owner) == (mode, line)}     # parameters the selected line owns (n_gpu is every multi-GPU line's)
    out = []
    named = set()                                                                          # one refusal per parameter NAME (n_gpu is every multi-GPU line's parameter)
    for (m, owner), prm in LINE_PARAMS.items():
        if (m, owner) == (mode, line) or prm.name in own or prm.name in named:
            continue
        named.add(prm.name)
        via = []
        one_gpu = (lambda v: prm.name == "n_gpu" and str(v).strip() == "1")               # `--n_gpu 1` / OPENFOLD3_OPT_N_GPU=1 is every route's, never foreign
        if args is not None and args.get(prm.name) not in (None, "") and not one_gpu(args.get(prm.name)):
            via.append(f"{prm.flag} {args.get(prm.name)}")
        if (environ.get(prm.env) or "").strip() and not one_gpu(environ.get(prm.env)):
            via.append(f"{prm.env}={environ.get(prm.env)}")
        if via:
            if prm.name == "n_gpu":                                                  # the memory mode's multi-GPU axis: opt_core.mem.ngpu's sentence under another mode, the line rule under big
                from opt_core.mem import ngpu
                out.append(ngpu.REFUSE_MODE + f" ({' / '.join(via)}; selected: {selected})" if mode != ngpu.MEMORY_MODE else
                           f"refused: {' / '.join(via)} runs the big/tp line (the row-sharded pair stack); the selected line is {selected}, a one-GPU line")
                continue
            out.append(f"{' / '.join(via)} is the {m}/{owner} line's parameter; the selected mode/line is {selected} — select --mode {m} or unset the parameter")
    return out


REQUIRED_PRODUCERS = ("opt_core.mem.ngpu", "opt_core.arch")   # core modules this package imports beyond the 0.3.x surface (the `--n_gpu` axis' token/refusal producer, the GPU-class registry): core >= 0.4.0
PRODUCER_MISSING = "producer_missing"                          # the NOT ACTIVE reason word when one is absent from the installed core (an older core beside this package): refused by name, never a traceback or a silent one-GPU run (_autoload.core_refusal)


def core_refusal() -> Optional[str]:
    """The NOT ACTIVE reason for an absent or older core (None when the core carries this package's producers): `_autoload.core_refusal`,
    the one probe every entry route runs before anything of the core beyond `opt_core` itself is imported."""
    from ._autoload import core_refusal as _probe
    return _probe()


def rank_world(environ: Optional[Mapping[str, str]] = None) -> int:
    """OF3TP_WORLD of a big/tp rank process as an int (1 when absent or unreadable: the refusal of a non-rank is resolve's)."""
    environ = os.environ if environ is None else environ
    w = (environ.get("OF3TP_WORLD") or "").strip()
    return int(w) if w.isdigit() and int(w) >= 1 else 1


def check_tp_world(n_gpu=None, environ: Optional[Mapping[str, str]] = None) -> int:
    """The tp line's world size P (LINE_PARAMS[("big", "tp")]): the `--n_gpu` argument, else its variable; one of the line's values (2|4|8),
    refused otherwise by name (never a silent single-GPU run under a multi-GPU selection)."""
    environ = os.environ if environ is None else environ
    prm = LINE_PARAMS[("big", "tp")]
    raw = environ.get(prm.env) if n_gpu in (None, "") else n_gpu
    try:
        n = int(str(raw).strip()) if raw not in (None, "") else None
    except ValueError:
        n = None
    if n is None or str(n) not in prm.values:
        raise ValueError(f"refused: the big/tp line runs on --n_gpu P GPUs, P in {'|'.join(prm.values)}: {prm.flag} P / {prm.env}=P (got {raw!r})")
    return n



def line_of(mode: str, environ: Optional[Mapping[str, str]] = None, n_gpu=None) -> Tuple[Optional[str], Line]:
    mode = check_mode(mode)
    if mode in MODE_LINES:
        el = line_arg(mode, environ, n_gpu)                                        # under big the GPU count selects the line (`--n_gpu` / OPENFOLD3_OPT_N_GPU)
        return el, LINES[(mode, el)]
    return None, LINES[(mode, None)]


def hook_dir(home: str, kit: str) -> str:
    """The add-on's hook directory (HOOK_DIR; a package hook: PACKAGE_HOOKS)."""
    if kit in PACKAGE_HOOKS:                                                                # the package's own hook (one directory on every route)
        return os.path.join(home, PACKAGE_HOOKS[kit])
    return os.path.join(home, KITS[kit], HOOK_DIR[kit])


def hook_dirs_of(home: str, kit: str) -> List[str]:
    """The directories an add-on's hook may be executed from: its hook directory (one per add-on; hooks.installed)."""
    return [os.path.abspath(hook_dir(home, kit))]


def resolve(mode: str, home: str, environ: Optional[Mapping[str, str]] = None, n_tokens: Optional[int] = None, n_gpu=None) -> Resolution:
    """The one resolver: mode -> Resolution against the tree at `home` (the openfold3/ directory) and the caller's environment.
    Exports are the line's switches plus the chain variables (CHAIN_ENV: the absolute hook directory each chaining hook executes next —
    OF3T_KIT_LEVERS, OF3O_KIT_LEVERS, OPENFOLD3_OPT_CONFHEAD_CHAIN), after the two size gates (graphs_gate, conf_gate). A caller-preset
    switch that contradicts the line (a different value, or a switch the line requires unset) is a conflict, never silently overridden."""
    environ = os.environ if environ is None else environ
    el, ln = line_of(mode, environ, n_gpu)
    og = of3o_gate(ln, None if (n_tokens is not None and int(n_tokens) <= 0) else n_tokens, environ)   # the offload port's item gate (big/resident): below OF3O_MIN_TOKENS the units step aside —
    if og is not None and not og["engaged"]:                                                              #  the line resolves to the fast line's composition minus the graph family (of3o_aside)
        ln = of3o_aside(ln)
    res = Resolution(mode=check_mode(mode), line=el, tier=tier(check_mode(mode), el), hooks=list(ln.hooks), levers=list(ln.levers), source=ln.source)
    if og is not None:
        res.of3o_gate, res.of3o_reason, res.of3o_min_tokens = og["fragment"], og["reason"], og["min_tokens"]
    res.exports = dict(ln.env)
    res.unsets = list(ln.unset)
    apply_levers_off(res, ln, environ)                                                        # MODEL_OPT_LEVERS_OFF (the ablation door): the named levers leave the line before anything is exported
    if n_tokens is not None and int(n_tokens) <= 0:                                         # a query set without polymer tokens (ligand-only) counts as unknown: the graph gate reads
        n_tokens = None                                                                       # polymer tokens (ligand atoms are tokens the count does not see), so 0 is no size at all
    res.n_tokens = n_tokens
    apply_reach_gate(res, environ, engaged=(og is None or og["engaged"]))                    # the resident line's reach gate (REACH_GATE_BY_CARD): the trunk's full-N^2 pair cells aside by name from the gate up
    g = conf_gate(ln, n_tokens) if any(l in CONF_LEVERS for l in res.levers) else None                                                               # the confidence levers' size gate (opt_core.attn.size_gate): below CONF_MIN_TOKENS the hook, the switches and the levers are the line's without them
    if g is not None:
        res.conf_gate, res.conf_reason = g["fragment"], g["reason"]
        if not g["served"]:
            res.hooks = [h for h in res.hooks if h != CONF_HOOK]
            res.levers = [l for l in res.levers if l not in CONF_LEVERS]
            for k in CONF_ENVS:
                if k in res.exports:
                    if k in CONF_ENVS_BELOW:
                        res.exports[k] = CONF_ENVS_BELOW[k]
                    else:
                        del res.exports[k]
                        res.unsets.append(k)
    res.hook_dirs = [hook_dir(home, k) for k in res.hooks]
    res.entry_hook = os.path.join(res.hook_dirs[0], HOOK_FILE) if res.hook_dirs else None
    if ln.kit_levers and ln.kit_levers in res.hooks:
        res.exports[ln.kit_levers_env] = hook_dir(home, ln.kit_levers)
    for a, b in zip(res.hooks, res.hooks[1:]):                                                 # every chaining hook names the next hook directory in its own variable (confhead -> OPENFOLD3_OPT_CONFHEAD_CHAIN,
        if a in CHAIN_ENV and CHAIN_ENV[a] not in res.exports:                                #  offload -> OF3O_KIT_LEVERS, trunk_kernels -> OF3T_KIT_LEVERS); a line's kit_levers export above takes precedence
            res.exports[CHAIN_ENV[a]] = hook_dir(home, b)
    if res.mode == "fast":
        res.precision = FAST_PRECISION                                                         # the ACTIVE line's precision= word; the CLI composes the bf16 runner yaml (cli.row_yaml)
    if res.mode == "big" and ln.runner_yaml:                                                # the big lines run under their own runner yaml (Line.runner_yaml): its Lightning precision,
        from .stock_pred import effective_precision                                            #  read from the file (the one reader), is the ACTIVE line's precision= word
        res.precision = effective_precision(os.path.join(home, ln.runner_yaml)).split(" ")[0]
    try:
        cap = graphs_cap(environ, line_graphs_cap(ln))
    except ValueError as e:
        res.conflicts.append(str(e)); cap = line_graphs_cap(ln)
    d = graphs_gate(res.levers, n_tokens, cap)                                                # the CUDA-graph size gate (opt_core.mem.graph_gate): eager/off → the graph switches are not exported

    def _eager():
        for k in GRAPHS_ENVS:
            res.exports.pop(k, None)
        res.unsets += [k for k in GRAPHS_ENVS if k not in res.unsets]
        res.levers = [l for l in res.levers if l not in GRAPHS_LEVERS]
    if d is not None:
        res.size_gate, res.gate_reason = d.fragment, d.reason
        if not d.capture:
            _eager()
    elif "cuda_graphs" in res.levers:                                                         # a cap is set but no token count is known (unreadable query, warm):
        if ln.graphs_max_tokens is not None:                                                  #  a line capped BY DEFAULT (exact) runs eager — fail-closed, named;
            res.size_gate, res.gate_reason = f"{GRAPH_GATE_NAME}=eager:n_tok_unknown", "n_tok_unknown"
            _eager()
        else:                                                                                 #  a line graphed at every size by default runs as written, named
            res.size_gate, res.gate_reason = f"{GRAPH_GATE_NAME}=capture", "n_tok_unknown"
    alloc_env = None
    if ALLOC_ENV.keys() <= res.exports.keys():                                                # the allocator policy (lever alloc_expandable): composed through the core's torch_alloc —
        from opt_core.mem import torch_alloc as _ta                                           # its one spelling, the caller's other allocator keys kept, the graphed-line composition
        alloc_env = _ta.ENV                                                                   # declared on purpose (torch keeps graph private pools off expandable segments: the policy
        assert ALLOC_ENV == _ta.env_row(ALLOC_POLICY), (ALLOC_ENV, _ta.env_row(ALLOC_POLICY))   # serves the non-graph heap — the confidence head's late allocations after post_release)
        cur = environ.get(_ta.ENV)
        try:
            have = _ta.parse_conf(cur)
            pre = have.get("expandable_segments")
            if pre is not None and pre.strip().lower() != "true":
                raise _ta.MemLeverRefused("alloc_expandable", f"{_ta.ENV}={cur!r} preset pins expandable_segments:{pre}, the {res.mode}{'/' + el if el else ''} line sets it True (unset it)")
            stage: Dict[str, str] = {}
            _ta.export(ALLOC_POLICY, environ=stage, lever="alloc_expandable", graphs_on="cuda_graphs" in res.levers, allow_with_graphs=True, cuda_initialized=False)
            have.update(_ta.parse_conf(stage[_ta.ENV]))
            res.exports[_ta.ENV] = _ta.format_conf(have)
        except (_ta.MemLeverRefused, ValueError) as e:
            res.conflicts.append(str(e))
    for k, v in res.exports.items():
        cur = environ.get(k)
        if cur is not None and cur != v and k != alloc_env:
            res.conflicts.append(f"{k}={cur!r} preset, the {res.mode}{'/' + el if el else ''} line sets {v!r}")
    for k in res.unsets:
        if environ.get(k) not in (None, ""):
            res.conflicts.append(f"{k}={environ.get(k)!r} preset, the {res.mode}{'/' + el if el else ''} line requires it unset")
    if res.mode != "off":                                                                  # the modes set their own switches: a preset add-on variable the line does not set is refused by name
        settable = set(NOT_LEVERS) | (tp_settable() if res.line in TP_LINES else set())       # (the deterministic recipe's switch, the run variables, and on the row-sharded line its documented OF3TP_* sizes)
        for k in environ:
            if k.startswith(SWITCH_PREFIXES) and k not in res.exports and k not in res.unsets and k not in settable and (environ.get(k) or "").strip():
                res.conflicts.append(f"{k}={environ.get(k)!r} preset: not a switch of the {res.mode}{'/' + el if el else ''} line (the mode sets its own switches; unset it)")
        for fam in CELL_FAMILIES:                                                          # the package's own cell levers arm on their first switch: preset while the line does not export it =
            if fam[0] in res.exports:                                                       # a lever this line does not carry (the line that carries it owns the family: its parameters — size gates,
                continue                                                                    # caps, core pins — are then the caller's to set)
            for k in fam:
                if (environ.get(k) or "").strip() and k not in res.unsets:
                    res.conflicts.append(f"{k}={environ.get(k)!r} preset: the {res.mode}{'/' + el if el else ''} line does not carry this cell lever "
                                         "(the modes set their own lever switches; unset it)")
    res.conflicts.extend(foreign_params(res.mode, res.line, environ=environ))              # another mode's parameter set (OPENFOLD3_OPT_N_GPU > 1 under exact/fast; the fast precision under another mode): NOT ACTIVE, by name
    if res.mode == "big" and res.line in TP_LINES:
        if not (environ.get("OF3TP_RANK") or "").strip() or not (environ.get("OF3TP_WORLD") or "").strip():
            res.conflicts.append("the big/tp line runs one process per GPU: this process is not a rank (OF3TP_RANK/OF3TP_WORLD absent) — "
                                 f"run `openfold3_opt pred --mode big --n_gpu P` (P in {'|'.join(LINE_PARAMS[('big', 'tp')].values)}), which spawns the ranks; the env/.pth route cannot")
        else:
            try:
                check_tp_world(environ.get("OF3TP_WORLD"), environ)
            except ValueError as e:
                res.conflicts.append(str(e))
    return res


def describe_line(res: Resolution) -> str:
    """The mode's spelling for the activation line: `<line>: K=V ... @ hook1>hook2` (`off` for stock)."""
    if res.mode == "off":
        return "off"
    kv = " ".join(f"{k}={v}" for k, v in res.exports.items() if k not in KIT_LEVERS_ENVS)
    return f"{res.line + ': ' if res.line else ''}{kv} @ {'>'.join(d + '(' + h + ')' for h, d in zip(res.hooks, hook_spellings(res)))}"


def hook_spellings(res: Resolution) -> List[str]:
    """Each hook directory of the resolution relative to its kit directory (`of3_levers`, `of3t_hook`), or relative to the add-ons' parent
    directory when it lies elsewhere under it, or relative to the tree for a package hook (`opt/openfold3_opt/hooks/confhead`) — the one spelling the activation line, the report's `hooks_spelling`
    and `describe_line` share."""
    out = []
    for h, d in zip(res.hooks, res.hook_dirs):
        if h in PACKAGE_HOOKS:                                                                  # a package hook: its directory relative to the tree (opt/openfold3_opt/hooks/<hook>)
            out.append(PACKAGE_HOOKS[h].replace(os.sep, "/"))
            continue
        kit_dir, dd = KITS[h].replace(os.sep, "/"), d.replace(os.sep, "/")
        parent = kit_dir.rsplit("/", 1)[0]                                                      # opt/forward
        i = dd.rfind("/" + kit_dir + "/")
        j = dd.rfind("/" + parent + "/")
        out.append(dd[i + len(kit_dir) + 2:] if i >= 0 else (dd[j + len(parent) + 2:] if j >= 0 else os.path.basename(dd)))
    return out


# ------------------------------------------------------------------------------------------------------- box probes (no torch) ----
def openfold3_version() -> Optional[str]:
    try:
        return importlib.metadata.version("openfold3")                    # metadata only, no openfold3 import
    except importlib.metadata.PackageNotFoundError:
        return None


def torch_version() -> Optional[str]:
    try:
        return importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return None


def triton_version() -> Optional[str]:
    try:
        return importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        return None


def nvidia_smi_query(fields: str = "name,compute_cap,memory.total", path_env: Optional[str] = None) -> Optional[List[str]]:
    """First GPU's `nvidia-smi --query-gpu=<fields>` row (no CUDA context)."""
    exe = shutil.which("nvidia-smi", path=path_env)
    if not exe:
        return None
    try:
        r = subprocess.run([exe, f"--query-gpu={fields}", "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
    except Exception:
        return None
    lines = (r.stdout or "").strip().splitlines()
    return [x.strip() for x in lines[0].split(",")] if r.returncode == 0 and lines else None


def nvidia_smi_compute_cap(path_env: Optional[str] = None) -> Optional[str]:
    row = nvidia_smi_query("compute_cap", path_env)
    return row[0] if row else None


def jit_cache_key(version: Optional[str] = None, cuda: Optional[str] = None, cc: Optional[str] = None) -> str:
    """The compiled-kernel cache key of the running stack, `torch<ver>-cu<cuda>-sm<cc>` (opt_core.jit_cache.key: the metadata version without
    its local tag, cuda from the local tag `cu<digits>` else torch.version.cuda, cc from nvidia-smi). A part that cannot be resolved on this host
    (no torch metadata, no GPU) makes the whole key the word `unknown` — configs/h100.env then keys the JIT caches under `$MODEL_OPT_JIT_ROOT/unknown/` when that root is set, a
    directory no GPU pinned stack shares — and the unresolvable part is named on stderr."""
    import sys
    from opt_core import jit_cache
    try:
        return jit_cache.key(version, cuda, cc)
    except jit_cache.StackKeyUnknown as e:                                  # the core names the unresolvable part; the kit's word for the whole key is `unknown`
        sys.stderr.write(f"[openfold3-opt] stack key unknown: {e}\n")
        return "unknown"
