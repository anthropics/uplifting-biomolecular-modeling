"""Modes and variants, and how a package mode resolves to a kit server mode.

The kit server's ``MODES`` table (``driver/ef2_server.py``) is the switch table: every entry is a tuple
``(base, opt_flags, w4_flags?, extra?)`` that the server's own ``configure(model, name, builder)`` applies in its fixed order
(ef2_mk_sampler -> ef2_msa -> ef2_w4 -> ef2_opt.install). This module never transcribes a tuple: ``server_table()`` reads the
table out of the server file itself (``ast.literal_eval`` of the dict literals, so no torch is needed), and ``resolve()`` maps
a package mode to a server mode NAME plus the kit's own two override switches for the vendored levers (``EF2_MK`` / ``EF2_MSA``,
which the server reads in ``configure()`` for any mode). The split of a tuple into flags below mirrors ``configure()``'s parsing
for the dry-run report only; application is always the server's.

Package modes (``KIT_MODES``, the one place to change what a mode means; ``MODES`` / ``DEFAULT_MODE`` are the one list and the one
default every command reads):
  exact   bitwise-identical to the library's own fused backend under the det recipe — the fastest bitwise configuration of this model
          (the reference einsum path is the library default; the fused backend is the library's own, selected by two model calls): server
          mode ``opt7x`` — the fused base, the exact ``ef2_opt`` levers and the bitwise W4 set — with no override switch, the same line
          on every variant.
  fast    server mode ``opt14_msa`` — tier 2 — plus the XL add-on's storage levers that are bitwise and speed-neutral on the fused kernels
          (``FAST_LINE``: ``XL_FAST_SET``); the package default (``DEFAULT_MODE``).
  big   the memory mode, DERIVED from ``FAST_LINE`` (``BIG_FAST`` = fast's server mode, fast's XL set and fast's drops, plus only the
          memory words: no CUDA-graph capture (``ENV_GRAPH_CAPTURE=0``), x4, the confidence head once per structure sample
          (``ENV_CONF_PER_SAMPLE=1``), and the levers with a memory cost left off,
          ``BIG_DROP_MEMORY``) — fast is a subset of big BY CONSTRUCTION (tests/test_fast_subset_big.py); the only ``--n_gpu P`` mode.
  off     stock: the upstream API in a clean subprocess (stock_fold.py); nothing from the kit on the path.
The user-facing tuple is exactly ``off / exact / fast / big`` (``MODES``), each named by its guarantee; ``pred`` / ``check`` / ``warm`` take
the package modes only.

Variants (``VARIANTS``): ``fast`` (ESMFold2-Fast, single sequence on every chain), ``full_msa`` (ESMFold2; every chain with the MSA
its input names, single sequence where it names none — upstream's rule), ``full_nomsa`` (ESMFold2, single sequence on every
chain). ``VARIANT_USES_MSA`` states which variant reads MSAs. The kit server knows two model names (``fast`` / ``full``);
``SERVER_VARIANT`` maps the package's axis onto them and the checkpoint per variant comes from ``stock/PINS.json``. One variant per
process: the W4 levers patch module-level entry points process-wide and the kit never calls ``disable()`` between variants.
"""
import ast
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .registry import ALL_VARIANTS, LEVERS, FIELD_BASE, FIELD_OPT, FIELD_W4, FIELD_MK, FIELD_MSA, acts_on

VARIANTS: Tuple[str, ...] = ALL_VARIANTS                                                     # the one literal lives in registry.py
SERVER_VARIANT: Dict[str, str] = {"fast": "fast", "full_msa": "full", "full_nomsa": "full"}     # the kit server's --variants names
VARIANT_USES_MSA: Dict[str, bool] = {"fast": False, "full_msa": True, "full_nomsa": False}

ENV_MK, ENV_MSA = "EF2_MK", "EF2_MSA"                             # the server's own override switches (configure() reads EF2_MK / EF2_MSA for any mode)
ENV_GRAPH_BUDGET = "EF2_GRAPH_BUDGET_TOKENS"                     # ef2_opt's per-shape graph budget (CFG.graph_budget_tokens; 0 = every shape captured)
ENV_GRAPH_BUDGET_TRUNK = "EF2_GRAPH_BUDGET_TOKENS_TRUNK"         # ef2_opt's per-SITE budgets (CFG.graph_budget_tokens_trunk / _encoder / _sampler): -1 = the site inherits the per-shape
ENV_GRAPH_BUDGET_ENCODER = "EF2_GRAPH_BUDGET_TOKENS_ENCODER"     #   budget above, 0 = no cap at this site, N = the site's own cap (trunk = the pair trunks and the recycle graph, encoder = the
ENV_GRAPH_BUDGET_SAMPLER = "EF2_GRAPH_BUDGET_TOKENS_SAMPLER"     #   lm / msa encoder graphs, sampler = the diffusion sampler's step graphs and ef2_dit's roll-out)
ENV_GRAPH_LRU_SAMPLER = "EF2_GRAPH_LRU_SAMPLER"                  # ef2_opt's sampler-graph budget per graph generation (CFG.lru_sampler; 8 = the driver default)
ENV_RECYCLE_GRAPH_MAX = "EF2_RECYCLE_GRAPH_MAX_TOKENS"            # ef2_opt's ceiling of the whole-recycle graph (lever rg; CFG.recycle_graph_max_tokens): pair planes above it take ls / tg
ENV_CONF_PER_SAMPLE = "EF2_CONF_PER_SAMPLE"                      # ef2_conf: "1" = the confidence head runs once per structure sample (the memory mode's word; big.conf_export)
ENV_GRAPH_CAPTURE = "EF2_GRAPH_CAPTURE"                          # ef2_opt's capture POLICY (CFG.graph_capture): "0" = no CUDA graph at any site — every graph optimization (tg / sg / eg / rg,
                                                                 #   ef2_dit's roll-out) runs its eager forward: the same kernels in the same order, no private pool, no static clones — whatever
                                                                 #   the budgets say; "1" (ef2_opt's default) = the budgets decide. The memory line's word (BIG_FAST)
GRAPH_SITE_BUDGETS = (ENV_GRAPH_BUDGET_TRUNK, ENV_GRAPH_BUDGET_ENCODER, ENV_GRAPH_BUDGET_SAMPLER)
GRAPH_ENV_DEFAULTS = {ENV_GRAPH_BUDGET: 0, ENV_GRAPH_LRU_SAMPLER: 8,   # ef2_opt's own import-time defaults for the graph switches (locked against the driver source by test)
                      ENV_GRAPH_BUDGET_TRUNK: -1, ENV_GRAPH_BUDGET_ENCODER: -1, ENV_GRAPH_BUDGET_SAMPLER: -1, ENV_RECYCLE_GRAPH_MAX: 512, ENV_GRAPH_CAPTURE: 1}
PACKAGE_GRAPH_DEFAULTS = {ENV_GRAPH_LRU_SAMPLER: 2}               # the package's defaults for every kit mode, exported at activation when neither the mode nor the caller sets the switch
                                                                  # (stack.package_graph_exports; the caller's own value wins): the sampler step-graph budget per generation — inputs rarely
                                                                  # repeat an atom count, so step graphs beyond the live one are captured and never replayed; 2 keeps the replay of an
                                                                  # immediate repeat (report.graphgen_decide)
PACKAGE_GRAPH_BUDGET_SAMPLER = {ENV_GRAPH_BUDGET_SAMPLER: 1536}   # the sampler SITE under exact / fast (both models) when the caller sets no budget of its own: the diffusion sampler's step
                                                                  # graphs / ef2_dit's roll-out are captured up to 1536 tokens and run the same statements eagerly
                                                                  # above — a captured roll-out holds address-stable copies of the fold's pair rows and conditioning cache (growing with
                                                                  # L^2), an eager one holds references only; big captures nothing (ENV_GRAPH_CAPTURE=0: no site budget applies there)
PACKAGE_GRAPH_BUDGET_FULL = {ENV_GRAPH_BUDGET: 800}               # the package's default per-shape graph budget for the FULL model under exact / fast (FULL_BUDGET_MODES, FULL_BUDGET_SERVER_VARIANT),
FULL_BUDGET_MODES: Tuple[str, ...] = ("exact", "fast")           # exported at activation when neither the mode nor the caller sets EF2_GRAPH_BUDGET_TOKENS (stack.package_graph_defaults): shapes up to
FULL_BUDGET_SERVER_VARIANT = "full"                               # 800 tokens are captured exactly as with no budget; above it the trunk / encoder graph levers run the same kernels eagerly (ef2_opt's
                                                                  # named `graph budget … -> eager` event) and ef2_w4's first-call probe is skipped — the Full model's largest shapes do not hold a
                                                                  # trunk graph pool beside their activations (the same budget as the Fast model's sites: above it the graphs'
                                                                  # launch savings are negligible against the fold while their private pools would set the peak reserved).
                                                                  # --mode off and big (no CUDA graph is captured there: ENV_GRAPH_CAPTURE=0) are untouched; 0 = every shape.
PACKAGE_GRAPH_BUDGET_FASTMODEL = {ENV_GRAPH_BUDGET_TRUNK: 800, ENV_GRAPH_BUDGET_ENCODER: 800}   # the package's per-site defaults for the FAST model (FASTMODEL_BUDGET_SERVER_VARIANT) under exact /
FASTMODEL_BUDGET_SERVER_VARIANT = "fast"                          # fast: the pair trunks / encoders are captured up to 800 tokens and run the same kernels eagerly above (the graphed regions'
                                                                  # private pools are what sets the peak there; the launch savings the graphs buy are a small share of a large fold); exported
                                                                  # when neither the mode nor the caller sets the site's switch or the per-shape budget
ENV_W4_IDPROBE = "EF2_W4_IDENTITY_PROBE"                         # ef2_w4's first-call probe switch (runs the stock op beside the patched one — tx / T10 — once per name: O(N²·C) reference buffers); "0" = off — the memory
                                                                 # mode sets it off on every line and at every --n_gpu: a diagnostic never costs reach; exact / fast at P=1 leave the add-on's default
SWITCHES = (ENV_MK, ENV_MSA)


@dataclass(frozen=True)
class KitMode:
    server_mode: str                                             # a NAME in the server's MODES table
    overrides: Dict[str, str] = field(default_factory=dict)      # the kit's own override switches that complete the line (EF2_MK / EF2_MSA)
    what: str = ""
    xl: bool = False                                             # the XL memory add-on's EXACT lever set installed on the model before configure() (big.xl_install)
    backend: Optional[str] = None                                # a `--backend` name (stock_fold.BACKENDS): the two upstream model calls the package makes on the loaded model
                                                                 # before the add-on installs (stock_fold.model_calls; stack.activate); None = no such call — every line in
                                                                 # KIT_MODES leaves it None (the kit server's base makes the model calls)
    blocked: Optional[str] = None                                # a named numerics finding: the mode is in the table (the hook declares it) but refuses by name in resolve()
    xl_set: Tuple[str, ...] = ()                                 # the XL add-on levers this line turns on (registry names; big.xl_knobs maps them to ef2_xl.apply keywords)
    xl_knobs: Dict[str, object] = field(default_factory=dict)    # `ef2_xl.apply` keywords this line fixes over the set's mapping (e.g. own=False: the fast line keeps x2's LOOPFREE
                                                                 # without OWN — the kit's ef2_opt trunk wrapper re-passes the pair, so the OWN list handoff would pop twice)
    line: Optional[str] = None                                   # big's composition key (`fast` = the mode itself); None elsewhere
    drop: Tuple[str, ...] = ()                                   # levers of the server mode's set this line leaves OFF (registry names; configure(off=)): a lever one of the line's own
                                                                 # replaces (FAST_DROP: x10 owns the distogram move) or that pins memory on the memory line (BIG_DROP_MEMORY)
    alloc_strict: bool = False                                   # the allocator policy (expandable segments) is exported before any CUDA work on every kit line; True = the memory
                                                                 # mode REFUSES by name a process that cannot take it (big.alloc_export), False = the graph lines keep the caller's
                                                                 # allocator and say so (stack.graph_lines_alloc_export: a note, never a refusal)
    conf_per_sample: bool = False                                # the confidence head runs once per structure sample when num_diffusion_samples > 1 (driver/ef2_conf.py): the
                                                                 # memory mode only — the batched statement holds [S, N, N, 256] fp32 pair tensors (19 GiB at 2000 tokens x 5)


XL_FAST_SET: Tuple[str, ...] = ("x2b", "x3", "x6", "x7", "x8", "x10")   # the XL add-on's STORAGE levers, on the fast line and therefore on big: bitwise on the fused kernels (registry class
                                                                            # bitwise: --det 1 cif + npz identical with / without the set on both models), speed-neutral, and a lower peak
                                                                            # allocated growing with L^2 (x3 is the largest single saving). The add-on's chunked TriMul is
                                                                            # inert under the fused kernels and its MSA pair-weighted-averaging lever conflicts with the kit's fused one, so neither
                                                                            # is a lever of this kit; x2b (FREE, one diffusion sample only: the add-on's own scope rule, named on its LEVER line at
                                                                            # samples > 1) is bitwise-neutral beside the kit's ESMC cache. The add-on's recycle-loop re-issue (LOOPFREE / OWN) is
                                                                            # not a lever of this kit: ef2_opt's static loop `ls` performs that release in its own frame in every mode
                                                                            # (loop_static_released), so the class re-issue would never run on the line (XL_FAST_KNOBS keeps OWN off)
XL_FAST_KNOBS: Dict[str, object] = {"own": False}
FAST_DROP: Tuple[str, ...] = ("disto",)                                    # off on the fast line, one owner per path: the XL lever x10 owns the distogram-to-host move (`disto` and x10 are both
                                                                            # bitwise and time-neutral; x10 also lowers the peak); exact keeps `disto` (no XL lever there)
X4_MIN_TOKENS = 1500                                                        # x4 (ESMC-6B streamed through the LM pass from the host) engages at inputs >= this many tokens: below it the LM's
                                                                            # ~15 GB of weights never decides the peak; bitwise vs the line without it at --det 1 (cif + pae
                                                                            # identical) — it lowers the peak by the LM's resident weights at the cost of the host-to-device stream
BIG_MEMORY_XL: Tuple[str, ...] = ("x4",)                                # the XL levers ONLY the memory line carries: x4 adds LM-pass time per fold where it engages
                                                                            # (>= X4_MIN_TOKENS) for the LM's weight memory — a memory lever with a speed cost, so not on fast
XL_BIG_SET: Tuple[str, ...] = XL_FAST_SET + BIG_MEMORY_XL              # big's XL set = fast's + the memory-only levers (by construction)
XL_BIG_KNOBS: Dict[str, object] = dict(XL_FAST_KNOBS, esmc_min_tok=X4_MIN_TOKENS)
BIG_DROP_MEMORY: Tuple[str, ...] = ("mh",)                              # levers of the fast line the memory line leaves OFF because they hold memory: `mh` (the MSA encoder's output
                                                                            # held across the recycles when depth <= msa_max_depth) pins a buffer growing with L^2 for a fold-time saving
                                                                            # WHEN it engages; inert (mh_hits = 0) on inputs whose MSA is deeper than msa_max_depth — the line that exists to
                                                                            # fit large inputs does not pin it (big = fast minus only the levers with a memory cost)
BIG_DROP: Tuple[str, ...] = FAST_DROP + BIG_DROP_MEMORY                # everything big leaves off the server mode's set: fast's own drop + the memory drops

BIG_GRAPH_CAPTURE = "0"                                         # the memory mode captures NO CUDA graph (the mode rule: CUDA graphs are allowed under exact / fast, never on the memory
                                                                  #   line): every graph optimization of the fast line stays installed and runs its eager forward — the same kernels in
                                                                  #   the same order (capture vs eager is bitwise: --det 1 cif + npz identical), no private pool, no address-stable
                                                                  #   clones of a fold's constants (the graph pools are what the memory line gives up); the cost is host launch
                                                                  #   latency where the fold is launch-bound (small inputs), nothing above the graph budgets where the fast line
                                                                  #   already runs eagerly
FAST_LINE = KitMode("opt14_msa", {},
                    "the tolerance-class lever set: the fused base, the CUDA-graph and cache levers, the W4, pair-transition, MSA-module, atom-path and diffusion-step kernels on top of the "
                    "exact line's hoists, plus the XL add-on's storage levers that are bitwise and speed-neutral on the fused kernels (x2b, x3, x6, x7, x8, x10: -1.9 ... -4.3 GiB peak at 800-1200 "
                    "tokens; x10 owns the distogram-to-host move, so `disto` is off here); tier 2 (within the stock's seed band)",
                    xl=True, xl_set=XL_FAST_SET, xl_knobs=XL_FAST_KNOBS, drop=FAST_DROP)
BIG_FAST = KitMode(FAST_LINE.server_mode, dict(FAST_LINE.overrides, **{ENV_GRAPH_CAPTURE: BIG_GRAPH_CAPTURE, ENV_W4_IDPROBE: "0"}),
                     "the memory mode = the fast line under the memory words (PROTOCOL: big composes on top of fast; fast is a subset of big by construction): `fast`'s server mode, lever set and "
                     "XL storage levers with NO CUDA graph captured (EF2_GRAPH_CAPTURE=0: the trunk / encoder / recycle / sampler graph levers and the roll-out run their eager forwards — same "
                     "kernels, no pools, no static clones), the W4 identity probe off, without `mh` (its held encoder output is memory: BIG_DROP_MEMORY), "
                     f"+ x4 (ESMC-6B streamed through the LM pass from the host, inputs >= {X4_MIN_TOKENS} tokens); the kit server's own model calls (the fused base); tier 2 (within the stock's seed band); "
                     "the only mode that accepts --n_gpu P",
                     xl=True, xl_set=FAST_LINE.xl_set + BIG_MEMORY_XL, xl_knobs=dict(FAST_LINE.xl_knobs, esmc_min_tok=X4_MIN_TOKENS), line="fast",
                     drop=FAST_LINE.drop + BIG_DROP_MEMORY, alloc_strict=True, conf_per_sample=True)


KIT_MODES: Dict[str, KitMode] = {
    "exact": KitMode("opt7x", {}, "bitwise-identical to the library's fused backend under the det recipe — the fastest bitwise configuration (the fused base, the graph / static-loop / cache levers, the bitwise W4 set, the atom-path hoists, the exact transition kernels, the MSA-module hoists, the sampler roll-out), no override switch, the same line on every variant"),
    "fast": FAST_LINE,
    "big": BIG_FAST,                                          # derived from FAST_LINE above: big = fast - CUDA-graph capture + x4 - BIG_DROP_MEMORY, nothing else
}
MODES: Tuple[str, ...] = ("exact", "fast", "big", "off")

# big's composition table: the mode itself (KIT_MODES["big"] = BIG_FAST, key `fast`, the token the ACTIVE line prints as line=fast); one lever set per mode.
KIT_LINES: Dict[str, Dict[str, KitMode]] = {
    "big": {
        "fast": KIT_MODES["big"],
    },
}
DEFAULT_LINE: Dict[str, str] = {"big": "fast"}
assert all(KIT_LINES[m][DEFAULT_LINE[m]] is KIT_MODES[m] for m in KIT_LINES)


DEFAULT_MODE = "fast"                                            # the package default: fast wherever a fast mode ships

SERVER_RELPATH = os.path.join("driver", "ef2_server.py")
DRIVER_RELPATH = os.path.join("driver", "run_ef2_om.py")


# --------------------------------------------------------------------------------------------------------------- the JIT cache key
def jit_cache_key(version: Optional[str] = None, cuda: Optional[str] = None, cc: Optional[str] = None, strict: bool = True) -> str:
    """The JIT cache key, ``torch<version>-cu<cuda>-sm<cc>``: the torch version without its local tag, the CUDA version without the dot,
    the device's compute capability digits (e.g. ``torch2.13.0-cu130-sm90``) — the release tree's one resolver (opt_core.jit_cache.key),
    which never imports torch: ``version`` defaults to the installed torch's distribution metadata, then ``torch/version.py``; ``cuda`` to the
    version's local tag (``+cu130``), then ``torch/version.py``'s ``cuda``, then the nvidia-cuda-runtime wheel; ``cc`` to nvidia-smi. A part
    that cannot be established raises (``StackKeyUnknown``): ``configs/h100.env``, which keys ``TRITON_CACHE_DIR`` by this value (one cache
    per stack), then refuses by name. ``strict=False`` renders such a part ``unknown`` — for display only (the activation report's field)."""
    from opt_core.jit_cache import key
    return key(version, cuda, cc, strict=strict)


# ----------------------------------------------------------------------------------------------------------------- reading the server
def _dict_literal(node) -> Optional[dict]:
    try:
        v = ast.literal_eval(node)
    except Exception:
        return None
    return v if isinstance(v, dict) else None


def server_table(server_path: str) -> Dict[str, tuple]:
    """The server's MODES table, read from the file: the ``MODES = {...}`` literal plus every ``MODES.update({...})`` in source
    order. Raises ValueError when the file carries no such table."""
    with open(server_path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=server_path)
    table: Dict[str, tuple] = {}
    found = False
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MODES" for t in node.targets):
            d = _dict_literal(node.value)
            if d is not None:
                table.update({k: tuple(v) for k, v in d.items()}); found = True
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            f = node.value.func
            if isinstance(f, ast.Attribute) and f.attr == "update" and isinstance(f.value, ast.Name) and f.value.id == "MODES" and node.value.args:
                d = _dict_literal(node.value.args[0])
                if d is not None:
                    table.update({k: tuple(v) for k, v in d.items()}); found = True
    if not found:
        raise ValueError(f"no MODES table found in {server_path}")
    return table


# ----------------------------------------------------------------------------------------------------------------- composition
def _split(spec: Optional[str], sep: str) -> List[str]:
    return [x for x in (spec or "").replace("+", sep).split(sep) if x] if spec else []


GROUPS: Tuple[str, ...] = ("trimul", "atom", "fz", "msa2", "pair", "hoist", "ln", "dit", "mk", "msa")   # the fourth field's groups (== ef2_server.GROUPS; locked by test)
GROUP_INSTALL_ORDER: Tuple[str, ...] = ("mk", "msa", "w4", "trimul", "atom", "fz", "opt", "msa2", "pair", "hoist", "ln", "dit")   # configure()'s install order after the base (levers_of lists names in it)


def parse_extra(extra: Optional[str]) -> Dict[str, List[str]]:
    """The fourth field -> {group: [flags]} (configure()'s own parsing: ';'-separated ``<group>:<flags>`` or a bare group; flags on ',' / '+').
    Raises ValueError naming an unknown group."""
    out: Dict[str, List[str]] = {}
    for tok in [t.strip() for t in str(extra or "").split(";") if t.strip()]:
        name, _, flags = tok.partition(":")
        if name not in GROUPS:
            raise ValueError(f"server MODES: unknown group {name!r} in {extra!r} (known: {', '.join(GROUPS)})")
        out[name] = _split(flags, ",") if flags else []
    return out


def composition(entry: tuple, overrides: Optional[Dict[str, str]] = None) -> dict:
    """A server mode tuple -> ``{"base", "opt": [...], "w4": [...], "mk": bool, "msa": [...], "groups": {group: [flags]}}`` (configure()'s own
    parsing: field 2 split on '+', field 3 on ',', field 4 on ';' into groups; ``overrides`` are the server's EF2_MK / EF2_MSA semantics: a set
    value wins)."""
    overrides = overrides or {}
    base = entry[0]
    opt = _split(entry[1] if len(entry) > 1 else None, "+")
    w4 = _split(entry[2] if len(entry) > 2 else None, ",")
    groups = parse_extra(entry[3] if len(entry) > 3 else "")
    mk = "mk" in groups
    msa = ",".join(groups.get("msa", []))
    if ENV_MK in overrides:
        mk = overrides[ENV_MK] == "1"
    if ENV_MSA in overrides:
        msa = overrides[ENV_MSA]
    msa_flags = [] if (not msa or msa.lower() in ("0", "off", "none")) else _split(msa, ",")
    budget = int(overrides[ENV_GRAPH_BUDGET]) if ENV_GRAPH_BUDGET in overrides else 0
    return {"base": base, "opt": opt, "w4": w4, "mk": mk, "msa": msa_flags, "has_w4_field": len(entry) > 2,
            "groups": {g: list(v) for g, v in groups.items() if g not in ("mk", "msa")},
            "graph_budget_tokens": budget,                              # > 0: the trunk / encoder graph levers run eagerly above this token count (ef2_opt; the sampler site follows its own switch)
            "graph_capture": str(overrides.get(ENV_GRAPH_CAPTURE, "1")) != "0"}   # False: no CUDA graph is captured at any site (the memory line); the graph levers stay in the set and run eagerly


def levers_of(comp: dict) -> List[str]:
    """Registry names of the levers a composition names, in the server's install order (base, mk, msa, w4, trimul, atom, fz, opt, msa2, pair, hoist, dit)."""
    names: List[str] = []
    if comp["base"] == "fused":
        names.append("fused")
    groups = comp.get("groups") or {}
    for g in GROUP_INSTALL_ORDER:
        if g == "mk":
            names += ["mk"] if comp["mk"] else []
        elif g == "msa":
            names += list(comp["msa"])
        elif g == "w4":
            names += list(comp["w4"])
        elif g == "opt":
            names += list(comp["opt"])
        elif g == "fz":
            names += ["fz"] if "fz" in groups else []
        elif g in groups:
            names += list(groups[g])
    return names


# ----------------------------------------------------------------------------------------------------------------- resolution
@dataclass
class Resolution:
    mode: str
    variant: Optional[str]
    server_mode: str
    entry: tuple
    overrides: Dict[str, str]
    composition: dict
    levers: List[str]                         # every lever the line names (registry names; unknown flags kept verbatim)
    levers_for_variant: List[str]             # the subset the kit's own guards let act on this variant
    levers_not_for_variant: List[str]         # e.g. the MSA-encoder levers on fast (no msa_encoder: the kit skips them itself)
    levers_unknown: List[str]                 # flags in the table the registry does not describe (reported, never dropped)
    server_variant: Optional[str]
    notes: List[str] = field(default_factory=list)
    line: Optional[str] = None                # big's composition key resolved (KIT_LINES); None for the other modes
    xl_set: Tuple[str, ...] = ()              # the XL levers the line turns on (big.xl_knobs)
    xl_knobs: Dict[str, object] = field(default_factory=dict)   # the line's fixed `ef2_xl.apply` keywords (KitMode.xl_knobs)
    drop: Tuple[str, ...] = ()                # the line's own subtractions from the server mode's set (KitMode.drop; not in `levers`)
    ablate: Tuple[str, ...] = ()              # levers the ablation variable subtracted (ablation.py; not in `levers`; named ablate= on the ACTIVE / APPLIED lines)
    knobs: Dict[str, str] = field(default_factory=dict)         # the ablation variable's <lever>.<knob>=<value> sub-choices (configure(knobs=))
    ablate_tokens: Tuple[str, ...] = ()       # the variable's tokens as given (the ACTIVE line's ablate= word)
    for_route: Tuple[str, ...] = ()           # the row-chunking levers the row-sharded route ADDS to the set at n_gpu > 1 (route_add; in `levers`, installed by rowpair.install_rank)
    not_for_route: Dict[str, str] = field(default_factory=dict)   # {lever: reason token}: the set's levers the row-sharded route leaves off at n_gpu > 1 (route_drop; not in `levers`)
    n_gpu: int = 1
    not_for_class: Dict[str, str] = field(default_factory=dict)   # {lever: reason token}: the set's levers another compute-capability class serves (class_drop; not in `levers`)
    cc: Optional[str] = None                  # the compute-capability class the set was resolved for ("9.0", "8.0", …; None = unknown: nothing subtracted by class)

    @property
    def levers_off(self) -> List[str]:
        """What configure(off=) receives: the line's drops, the ablated levers, the route's and the class's subtractions and the set's levers the registry keeps off this variant."""
        return sorted(set(self.drop) | set(self.ablate) | set(self.not_for_route) | set(self.not_for_class) | set(self.levers_not_for_variant))


def kit_mode(mode: str, line: Optional[str] = None) -> KitMode:
    if mode not in KIT_MODES:
        raise ValueError(f"unknown kit mode {mode!r}; expected one of {sorted(KIT_MODES)} (or 'off')")
    if line is None:
        return KIT_MODES[mode]
    if mode not in KIT_LINES or line not in KIT_LINES[mode]:
        raise ValueError(f"mode {mode!r} has no composition {line!r}" + (f"; expected one of {sorted(KIT_LINES[mode])}" if mode in KIT_LINES else ""))
    return KIT_LINES[mode][line]


def check_variant(variant: Optional[str]) -> Optional[str]:
    if variant is None:
        return None
    v = variant.strip().lower()
    if v not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    return v


def route_drop(n_gpu: int) -> Dict[str, str]:
    """The levers of a set the row-sharded route (``--n_gpu`` > 1) does not carry, {lever: reason token} — the multi-GPU line's own table
    (``tp.not_for_route``); nothing at n_gpu 1 (every in-process route)."""
    if int(n_gpu or 1) <= 1:
        return {}
    from . import tp
    return {str(k): str(v) for k, v in dict(tp.not_for_route(int(n_gpu))).items()}


def route_add(n_gpu: int) -> List[str]:
    """The levers the row-sharded route (``--n_gpu`` > 1) ADDS to a set, in install order — the multi-GPU line's own table
    (``tp.for_route``: the row-chunking levers of esmfold2_opt.rowchunk); nothing at n_gpu 1."""
    if int(n_gpu or 1) <= 1:
        return []
    from . import tp
    return [str(n) for n in tp.for_route(int(n_gpu))]


def class_drop(cc: Optional[str], line_set: List[str]) -> Dict[str, str]:
    """The levers of a set another compute-capability class serves, {lever: reason token} — the registry's per-class words
    (``Lever.classes``: ``serves_<classes>``; ``SUPERSEDED_ON``: ``superseded_by_<lever>`` when that lever is in the set); nothing when the
    class is unknown (no GPU visible and no target configured: the plan names every lever of the table). resolve() applies the two words in
    two phases (the class exclusion before the ablation variable is read, the supersession after it, against the levers still in the set)."""
    if not cc:
        return {}
    from .registry import not_on_class
    out: Dict[str, str] = {}
    for n in line_set:
        if n in LEVERS:
            why = not_on_class(LEVERS[n], cc, in_set=line_set)
            if why:
                out[n] = why
    return out


def target_cc(target_gpu: Optional[str]) -> Optional[str]:
    """The compute-capability class a configured target GPU word names (configs/*.env MODEL_OPT_TARGET_GPU: H100 / H200 -> 9.0, A100 -> 8.0,
    B200 -> 10.0, B300 -> 10.3, L40S / RTX 6000 Ada -> 8.9); None for an unknown word. The probe's own class wins when a GPU is visible."""
    t = (target_gpu or "").strip().lower()
    for words, cc in ((("h100", "h200", "gh200"), "9.0"), (("a100", "a800"), "8.0"), (("b200", "gb200"), "10.0"), (("b300", "gb300"), "10.3"),
                      (("l40", "rtx 6000 ada", "l4"), "8.9"), (("a10", "a40", "rtx a6000"), "8.6")):
        if any(t.startswith(w) or t == w for w in words):
            return cc
    return None


def resolve(mode: str, variant: Optional[str], kit_home: str, table: Optional[Dict[str, tuple]] = None, line: Optional[str] = None,
            ablate: Optional[str] = None, n_gpu: int = 1, cc: Optional[str] = None) -> Resolution:
    """Resolve a package mode to its server mode by NAME in the server's own table (read from the kit at `kit_home`). The one
    resolver; ``line`` = big's internal composition key (None = the mode itself); ``ablate`` = the ablation variable's text (ablation.parse:
    named levers leave the set before anything is installed; an unknown or inconsistent token raises by name); ``n_gpu`` > 1 = the row-sharded
    route, whose own table (route_drop) names the levers it does not carry — they leave configure()'s set like the line's drops, named
    ``not_for_route`` on the kit's lines; ``cc`` = the box's compute-capability class ("9.0" …): the levers another class serves leave the
    set the same way (class_drop), named ``not_for_class``."""
    from .ablation import parse as _parse_ablation
    table = table if table is not None else server_table(os.path.join(kit_home, SERVER_RELPATH))
    km = kit_mode(mode, line)
    variant = check_variant(variant)
    if km.blocked:
        raise ValueError(f"mode {mode!r} is BLOCKED ({km.blocked}); it is not runnable in this tree")
    if km.server_mode not in table:
        raise ValueError(f"mode {mode!r} -> server mode {km.server_mode!r} is not in the server's MODES table ({len(table)} entries): "
                         f"the kit in the tree does not carry it")
    entry = table[km.server_mode]
    comp = composition(entry, km.overrides)
    line_set = [n for n in levers_of(comp) if n not in km.drop] + (list(km.xl_set) if km.xl else [])
    off_route = {k: v for k, v in route_drop(n_gpu).items() if k in line_set}   # the row-sharded route's own subtractions (n_gpu > 1), by name with their reason
    line_set = [n for n in line_set if n not in off_route]
    on_route = tuple(n for n in route_add(n_gpu) if n not in line_set)          # the row-sharded route's own additions (n_gpu > 1): the row-chunking levers, registry names the ablation variable can subtract
    line_set = line_set + list(on_route)
    from .registry import class_excludes, superseded_on
    off_class = {n: why for n in line_set if n in LEVERS and (why := class_excludes(LEVERS[n], cc))}   # phase 1: the levers whose kernels are another class's (t16 off 9.0) are not in
    line_set = [n for n in line_set if n not in off_class]                                            # this box's set at all (naming one in the ablation variable is an unknown token)
    ab = _parse_ablation(ablate, line_set, mode)                              # raises AblationError (a ValueError) naming an unknown / inconsistent token
    levers = [n for n in line_set if n not in ab.levers]
    superseded = {n: why for n in levers if n in LEVERS and (why := superseded_on(LEVERS[n], cc, levers))}   # phase 2: the levers a lever STILL in the set serves instead on this class
    off_class.update(superseded)                                                                            # (t15 / t15msa / t10 by t16 on 9.0; an ablated t16 leaves them in)
    levers = [n for n in levers if n not in superseded]
    known = [n for n in levers if n in LEVERS]
    unknown = [n for n in levers if n not in LEVERS]
    for_variant = [n for n in known if acts_on(LEVERS[n], variant)]
    not_for_variant = [n for n in known if not acts_on(LEVERS[n], variant)]
    notes = []
    if ab.tokens:
        notes.append(f"ablation: {','.join(ab.tokens)} (levers off: {','.join(ab.levers) or 'none'}; knobs: {','.join(f'{k}={v}' for k, v in ab.knobs.items()) or 'none'})")
    xl_set = tuple(n for n in km.xl_set if n not in ab.levers) if km.xl else ()   # the XL set MINUS the ablated levers: what big.xl_install turns on (an ablated XL lever is off in the add-on too, not only on its LEVER line)
    comp = dict(comp, xl=bool(km.xl), alloc_strict=bool(km.alloc_strict), conf_per_sample=bool(getattr(km, "conf_per_sample", False)))   # alloc_strict: the memory mode refuses a process that cannot take the allocator policy; the graph lines note it (stack)
    return Resolution(mode=mode, variant=variant, server_mode=km.server_mode, entry=entry, overrides=dict(km.overrides), composition=comp,
                      levers=levers, levers_for_variant=for_variant, levers_not_for_variant=not_for_variant, levers_unknown=unknown,
                      server_variant=SERVER_VARIANT.get(variant) if variant else None, notes=notes,
                      line=km.line, xl_set=xl_set, xl_knobs=dict(km.xl_knobs) if km.xl else {}, for_route=on_route, not_for_route=off_route, n_gpu=int(n_gpu or 1),
                      drop=tuple(km.drop), ablate=tuple(ab.levers), knobs=dict(ab.knobs), ablate_tokens=tuple(ab.tokens), not_for_class=off_class, cc=(str(cc) if cc else None))


def describe_line(res: Resolution) -> str:
    """The mode's server spelling for the activation line: the server mode name, plus ``+EF2_MK=<v>,EF2_MSA=<v>`` for a mode that
    carries override switches (``opt7x`` and ``opt14_msa`` carry none)."""
    ov = ",".join(f"{k}={v}" for k, v in res.overrides.items())
    return res.server_mode + (f"+{ov}" if ov else "")
