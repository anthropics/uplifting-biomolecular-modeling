"""The mode table — the one place a mode is defined — and how a mode resolves to a kit lever set and the DTK swap.

A mode names one of the kit's own lever sets (``LEVER_SETS`` in ``opt/forward/af3t/af3_torch/af3_torch_api.py``, read here by
``kit_lever_sets`` from the file, never transcribed) and whether the DTK FusedDiT add-on (``opt/forward/dtk``) replaces the diffusion
token transformer:

  off   = xfold as shipped: the kit's ``eager`` set (no lever: the patched xfold module tree, fp32 weights under bf16 autocast), no DTK,
          xfold's fastnn Triton kernels ON (``FASTNN_MODES``: the stock CLI's ``--fastnn`` default) and NO token padding
          (``MODE_PADDING['off']`` = none: each input at its own token count, as the stock CLI featurises). ``--nofastnn`` runs the eager
          port instead (xfold's torch ops for layer norm / attention / gated linear unit).
  fast  = the kit's ``fastest`` set (bf16 weights, fused trimul / tri-attention / transition kernels, SDPA attention-pair-bias,
          hoisted + whole-step-graphed diffusion, torch.compile of the step glue) + DTK — Tier 2 against ``off`` (fp64 error
          ratio at or below stock, bitwise run-to-run — registry.LEVERS numerics per lever).
  big = ``fastest`` recomposed for memory headroom (``big.py``, the levers on ``opt_core.mem``): NO CUDA graph at any size
          (``graph_drop``: ``stepgraph`` dropped from the build, the step eager on its ``hoist`` — ``big.GRAPH_DROP_MIN_TOKENS = 0``),
          the diffusion statics released before the heads, the recycle's prev embeddings consumed once, the allocator's
          expandable segments + DTK — ``fast``'s numerics class per remaining lever; the mode for inputs past ``fast``'s ceiling
          (slower than ``fast`` where fast fits and the sampler step is launch-bound; the same speed at sizes where it is not).
  exact = ``off``'s base (the stock kernels, no padding: ``FASTNN_MODES`` / ``MODE_PADDING``) plus the levers of ``fastest`` whose outputs equal it byte for byte
          on H100 (``registry.EXACT``: the whole-step-graphed sampler with its hoist) and the package levers ``template_dedupe``, ``dev_scalars``, ``tri_layout``, ``ln_rows``, ``attn_layout``, ``gate_fuse`` and ``castcache``; no DTK.
          Its outputs are ``off``'s byte for byte, sooner.

Every mode but ``off`` also carries PACKAGE levers (``MODE_PACKAGE_LEVERS``; ``registry.PACKAGE_LEVERS``), composed in the model process on
top of the kit's build: ``template_dedupe`` — the template embedder evaluates each distinct template slot once (bitwise by construction;
``template_dedupe.py``); ``dev_scalars`` — the trunk pass's constant scalars (the norm's epsilon clip, the bond contact matrix's ones /
zero) resident on the device instead of copied from the host per use (bitwise by construction; ``dev_scalars.py``); ``tri_layout`` — the stock triangle
multiplication's operand and product layout copies made by a tiled transpose kernel instead of torch's generic strided copy (the same bytes in the
same layouts: bitwise by construction; ``tri_layout.py``); ``ln_rows`` — xfold's fastnn Triton LayerNorm kernel run over 8 rows per program instead of
one (the same per-row code: bitwise by construction; ``ln_rows.py``); ``attn_layout`` — the triangle attention's head-major operands and
token-major output written by a row-regroup kernel instead of torch's strided copy (same bytes: bitwise by construction; ``attn_layout.py``); ``gate_fuse`` — those two forwards' mask / sigmoid-gate statements as one
kernel with the stock rounding points (``gate_fuse.py``); ``castcache`` (exact) — each fp32 Linear's bf16 autocast weight cast made once
per process instead of once per call (the same cast: bitwise by construction; under fast / big ``bf16w`` leaves autocast nothing to cast;
``castcache.py``); ``canonical_noise`` (fast, big) — the
sampler's shape-dependent random draws are made at stock's bucket length and sliced (``canonical_noise.py``), so the kernel_tile padding
keeps stock's noise for the seed.

Token padding (``MODE_PADDING``; ``cli.bucket_row``): ``off`` / ``exact`` do not pad (each input at its own token count — what xfold's CLI
featurises to); ``fast`` / ``big`` pad every input, at any length, to the next multiple of the shared core's kernel tile
(``opt_core.shape_policy.padded_len(n, 'kernel_tile')``, the one constant every padding kit of the line shares: 400 -> 448, 1000 -> 1024,
1400 -> 1408, 2500 -> 2560), with ``canonical_noise`` drawing the sampler's noise at the input's own token count so the real atoms' noise is stock's.

``--n_gpu P`` is a RESOURCE AXIS of ``big`` (``N_GPU_SUPPORTED`` = 1, 2, 4, 8): P = 1 is the single-GPU composition above, byte for byte.
P > 1 row-shards the pair representation over P GPUs of the box END TO END (``rowpair_xfold.py`` on the shared core's ``opt_core.mem.rowpair``,
``forward.py main_sharded``: the pair tensor is born as each rank's row shard in the input embedder and stays a shard through recycling, the
template embedder, the MSA module, the Pairformer, the distogram and confidence heads and the diffusion conditioning / transformer —
nothing ``N x N x c`` is ever whole on a rank; the single / MSA / atom tensors are replicated by design). Tier 2 like every ``big`` arm.
P is explicit (default 1, never auto-detected); a P outside the set, fewer visible GPUs than P, or P > 1 under ``off`` / ``fast`` is refused
by name (``stack.n_gpu_gate``; the NOT ACTIVE line, rc 3).

the ACTIVE line carries the set and swap that ran;
the mode table itself never changes. The package default (``--mode`` omitted, ``AF3_TORCH_OPT`` unset) is ``fast`` (the rule: ../README.md); ``off`` is selected by name.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Tuple

MODES: Tuple[str, ...] = ("off", "exact", "fast", "big")
# Mode names this package REFUSES BY NAME, with the reason every entry route prints (``--mode``, ``AF3_TORCH_OPT``, the autoload hook):
# none — the table is the mechanism (cli.refused_mode_line, _autoload) for a standard mode name whose guarantee this engine could not meet.
REFUSED_MODES: Dict[str, str] = {}
DEFAULT_MODE = "fast"     # always fast: the steady-state choice once the caches are warm — a single cold item is faster under --mode off (no compile)
ENV_MODE = "AF3_TORCH_OPT"

N_GPU_DEFAULT = 1
N_GPU_SUPPORTED: Tuple[int, ...] = (1, 2, 4, 8)   # the P set ``--mode big --n_gpu P`` does not refuse (every rank holds 1/P of the pair rows end to end; P = 4 / 8 complete a larger input than one GPU on the measured ladder)
N_GPU_MODES: Tuple[str, ...] = ("big",)          # n_gpu > 1 is an axis of the memory mode alone (opt_core.mem.rowpair MEMORY_MODE)

MODE_SETS: Dict[str, str] = {"off": "eager", "exact": "fastest", "fast": "fastest", "big": "fastest"}   # mode -> the kit lever set it runs (a key of the kit's LEVER_SETS); exact keeps only the bitwise levers (MODE_FILTER), fast runs it whole, big recomposes it (big.build_levers)
MODE_DTK: Dict[str, bool] = {"off": False, "exact": False, "fast": True, "big": True}               # mode -> DTK FusedDiT swapped into the diffusion token transformer (tier 2: never under exact)

MODE_FILTER: Dict[str, str] = {"exact": "EXACT"}                                                                   # mode -> the registry tuple of kit levers the mode KEEPS from its kit set: the ones byte-equal to the stock-kernels base (registry.EXACT)
FASTNN_MODES = ("off", "exact")                                                                                     # the modes whose model process runs xfold's shipped fastnn Triton kernels (the stock CLI's `--fastnn`, default on): off = xfold as shipped; exact = the same base, its statement made against it. fast / big: the kit's kernel levers serve those modules (layer norm / attention / gated linear unit) — fastnn is not passed to their model process
MODE_PADDING: Dict[str, str] = {"off": "none", "exact": "none", "fast": "kernel_tile", "big": "kernel_tile"}       # mode -> the token-padding policy the featuriser pads to (cli.bucket_row): none on the stock-numerics modes (xfold as shipped: each input at its own token count); the shared core's kernel tile (opt_core.shape_policy, ONE constant for every padding kit) on fast / big
MODE_PACKAGE_LEVERS: Dict[str, Tuple[str, ...]] = {"off": (), "exact": ("template_dedupe", "dev_scalars", "tri_layout", "ln_rows", "attn_layout", "gate_fuse", "castcache", "prefetch", "write_behind", "autotune_cache", "feat_par"), "fast": ("template_dedupe", "dev_scalars", "tri_layout", "ln_rows", "attn_layout", "gate_fuse", "canonical_noise", "prefetch", "write_behind", "autotune_cache", "feat_par"), "big": ("template_dedupe", "dev_scalars", "tri_layout", "ln_rows", "attn_layout", "gate_fuse", "canonical_noise", "prefetch", "write_behind", "autotune_cache", "feat_par")}   # mode -> the package levers composed on the kit's build (registry.PACKAGE_LEVERS; canonical_noise rides the kernel_tile padding)
API_RELPATH = os.path.join("af3_torch", "af3_torch_api.py")          # the kit file that defines LEVER_SETS (relative to the kit dir)
LEVER_SETS_NAME = "LEVER_SETS"
ENV_LEVERS_OFF = "MODEL_OPT_LEVERS_OFF"                                 # the ablation switch (the line's one convention): a comma list of lever names ONE run of a mode drops from its resolved selection — kit levers of the mode's tuple, package levers, `dtk`, big's memory levers (levers_off / resolve); a name that is no lever of this kit or not in the mode's selection is refused BY NAME (LeversOffRefused: usage, rc 2); the ACTIVE line names what was dropped (`levers_off=`), each dropped lever's LEVER line reads `state=off reason=levers_off`; unset or blank = the mode as it is, no token anywhere


from opt_core.modes import UnsupportedMode, levers_label, literal_assignment  # noqa: E402,F401 — the core's: a mode this model does not have (``exact``) or an unknown one; the ACTIVE line's levers= label


class LeversOffRefused(UnsupportedMode):
    """A ``MODEL_OPT_LEVERS_OFF`` request this kit refuses by name (an UnsupportedMode: every entry route's usage refusal, rc 2): a name that
    is no lever of this kit, a lever the mode's selection does not carry, or any name under ``off`` (which applies no lever)."""


def levers_off(environ: Optional[dict] = None) -> Tuple[str, ...]:
    """The lever names ``MODEL_OPT_LEVERS_OFF`` lists, in order — comma-separated, whitespace stripped, duplicates folded; () when unset or blank."""
    raw = (os.environ if environ is None else environ).get(ENV_LEVERS_OFF) or ""
    out: list = []
    for w in raw.split(","):
        w = w.strip()
        if w and w not in out:
            out.append(w)
    return tuple(out)


def kit_lever_names(sets: Optional[Dict[str, Tuple[str, ...]]] = None) -> Tuple[str, ...]:
    """Every lever name this kit has, in a stable order: the kit's LEVER_SETS members, the registry's levers (``hoist``, ``dtk``, the package
    levers included) and big's memory levers — the names ``MODEL_OPT_LEVERS_OFF`` may carry (anything else is refused by name)."""
    from . import big as _big, registry as _registry
    sets = kit_lever_sets() if sets is None else sets
    out: list = []
    for name in [l for v in sets.values() for l in v] + list(_registry.LEVERS) + list(_big.LEVER_ORDER):
        if name not in out:
            out.append(name)
    return tuple(out)


def kit_lever_sets(kit: Optional[str] = None) -> Dict[str, Tuple[str, ...]]:
    """The kit's own ``LEVER_SETS`` (name -> lever names), parsed from ``af3_torch_api.py`` without importing it (no torch on this
    interpreter: opt_core.modes.literal_assignment). A changed shape (the assignment missing, not a literal, not name -> lever names) is a
    named failure, never a fallback."""
    from .stack import kit_home
    path = os.path.join(kit or kit_home(), API_RELPATH)
    try:
        sets = literal_assignment(path, LEVER_SETS_NAME)
    except ValueError as e:
        raise RuntimeError(f"{e} (the kit's api changed shape)")
    if not isinstance(sets, dict) or not all(isinstance(k, str) and isinstance(v, (tuple, list)) for k, v in sets.items()):
        raise RuntimeError(f"{path}: {LEVER_SETS_NAME} is not a dict of name -> tuple of lever names (the kit's api changed shape)")
    return {k: tuple(v) for k, v in sets.items()}


def resolve(mode: str, kit: Optional[str] = None, environ: Optional[dict] = None) -> dict:
    """{'mode', 'lever_set', 'levers', 'dtk', 'big', 'package_levers', 'padding', 'fastnn', 'levers_off'} for a package mode
    (``big``: the memory composition, big.selection — every memory lever in force). ``MODEL_OPT_LEVERS_OFF`` in ``environ`` (the
    process's when None) drops the named levers from the selection after it is composed (``levers_off``: the names dropped, () when none)."""
    if mode in REFUSED_MODES:                                   # a standard mode name this engine refuses by name, with its reason
        raise UnsupportedMode(REFUSED_MODES[mode])
    if mode not in MODES:
        raise UnsupportedMode(f"unknown mode {mode!r}; modes are {' | '.join(MODES)}")
    sets = kit_lever_sets(kit)
    set_name = MODE_SETS[mode]
    if set_name not in sets:
        raise RuntimeError(f"the kit's LEVER_SETS has no {set_name!r} set (keys {sorted(sets)}); the mode table names it")
    lever_set, lever_names = set_name, sets[set_name]
    if mode in MODE_FILTER:                                     # exact: the kit set reduced to the levers byte-equal to the stock-kernels base (registry.EXACT), in the set's order
        from . import registry as _registry
        keep = getattr(_registry, MODE_FILTER[mode])
        unknown = [l for l in keep if l not in sets[set_name]]
        if unknown:
            raise RuntimeError(f"mode {mode!r}: registry.{MODE_FILTER[mode]} names {unknown}, absent from the kit's {set_name!r} set {sets[set_name]} (the kit's api changed shape)")
        lever_set, lever_names = mode, tuple(l for l in lever_names if l in keep)
    sel = None
    if mode == "big":                                        # the memory line: the base set recomposed + the memory levers in force (big.selection)
        from . import big as _big
        present, why = _big.mem_present()
        if not present:
            raise UnsupportedMode(why)
        sel = _big.selection()
        lever_set, lever_names = "big", _big.build_levers(lever_names, sel["levers"])
    pkg = MODE_PACKAGE_LEVERS[mode]
    if MODE_PADDING[mode] == "kernel_tile" and "canonical_noise" not in pkg:
        raise RuntimeError(f"mode {mode!r}: padding kernel_tile without the canonical_noise package lever (MODE_PACKAGE_LEVERS) — the sampler's noise would depend on the padded length")
    dtk = MODE_DTK[mode]
    off = levers_off(environ)
    if off:                                                     # the ablation switch: ONE run of the mode without the named levers of its selection (ENV_LEVERS_OFF); validated by name first
        from . import big as _big
        known = kit_lever_names(sets)
        selection = list(lever_names) + list(pkg) + (["dtk"] if dtk else []) + (list(sel["levers"]) if sel else [])
        unknown = [n for n in off if n not in known]
        if unknown:
            raise LeversOffRefused(f"{ENV_LEVERS_OFF}={','.join(off)}: {', '.join(unknown)} — not a lever of this kit (its levers: {', '.join(known)})")
        absent = [n for n in off if n not in selection]
        if absent:
            hint = ""
            if "hoist" in absent and "stepgraph" in selection:
                hint = " — the hoist rides stepgraph in this mode: name stepgraph"
            elif "stepgraph" in absent and sel is not None and "hoist" in selection:
                hint = " — big drops stepgraph for its hoist (graph_drop): name hoist"
            raise LeversOffRefused(f"{ENV_LEVERS_OFF}={','.join(off)}: {', '.join(absent)} — not in mode {mode!r}'s selection "
                                   f"({levers_label(selection) if selection else 'no lever: off applies none'}){hint}")
        if sel is not None:                                     # a memory lever dropped: the kit tuple is recomposed from the base without it (graph_drop off keeps stepgraph)
            gone = [l for l in sel["levers"] if l in off]
            sel = dict(sel, levers=[l for l in sel["levers"] if l not in off])
            if gone:
                sel["disabled"] = gone
            lever_names = _big.build_levers(tuple(l for l in sets[set_name] if l not in off), sel["levers"])
        lever_names = tuple(l for l in lever_names if l not in off)
        pkg = tuple(l for l in pkg if l not in off)
        dtk = dtk and "dtk" not in off
    return {"mode": mode, "lever_set": lever_set, "levers": tuple(lever_names), "dtk": dtk, "big": sel, "package_levers": tuple(pkg), "padding": MODE_PADDING[mode], "fastnn": mode in FASTNN_MODES,
            "levers_off": tuple(off)}
def mode_from_env(environ: Optional[dict] = None) -> str:
    env = os.environ if environ is None else environ
    return (env.get(ENV_MODE) or "").strip() or DEFAULT_MODE


# The stock CLI's own protocol knobs, passed through under their upstream names and defaults (xfold @ the pin, ``run_alphafold.py``):
# ``--num_diffusion_samples`` (``flags.DEFINE_integer('num_diffusion_samples', 5, …)``, run_alphafold.py:198-202; ``tests/test_stock_route.py``
# reads the 5 from the pinned archive) and ``--fastnn`` / ``--nofastnn`` (``flags.DEFINE_bool('fastnn', True, …)``, run_alphafold.py:98-102).
# ``--num_recycles`` / ``--diffusion_steps`` are the model constructor's (``xfold/alphafold3.py`` ``AlphaFold3(num_recycles=10, …,
# diffusion_steps=200)``): absent = the constructor's own defaults; upstream's CLI has no flag for them.
STOCK_NUM_DIFFUSION_SAMPLES = 5
UPSTREAM_SAMPLES = STOCK_NUM_DIFFUSION_SAMPLES            # the name the chain-contract tests read


def fastnn_refusal(mode: str, fastnn: Optional[bool]) -> Optional[str]:
    """The NOT ACTIVE reason for a ``--fastnn`` / ``--nofastnn`` value this mode cannot serve (None = served; None value = the mode's own:
    on under ``FASTNN_MODES``, the kit's kernels elsewhere). ``exact`` is stated against xfold's shipped fastnn kernels, so ``--nofastnn`` is
    ``off``'s alone; ``fast`` / ``big`` replace those modules with the kit's kernel levers, so an explicit ``--fastnn`` is refused there."""
    if fastnn is False and mode == "exact":
        return "--nofastnn is served under --mode off only, not exact (exact = xfold's shipped fastnn kernels plus the levers byte-equal to them; the eager port is off's --nofastnn)"
    if fastnn is True and mode not in FASTNN_MODES:
        return f"--fastnn is served under --mode {' | '.join(FASTNN_MODES)} only, not {mode} (the kit's kernel levers serve xfold's fastnn modules under {mode}; drop the flag, or use --mode off / exact for the stock kernels)"
    return None
