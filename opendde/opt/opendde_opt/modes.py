"""Modes, kit lines, and the environment each one materialises — the one mode table of this tree.

Every optimization of OpenDDE in this tree is a runtime add-on switched by environment variables and loaded by one kit shim (the
kit's ``levers/ACCEL/sitecustomize.py`` listed FIRST on ``PYTHONPATH``, or the same file executed by ``stack.py`` when the package
activates a mode in-process). A *line* is a named, complete composition: the kit directories in ``PYTHONPATH`` order, the switches
exported, the switches that must be absent, and — for the FPF TriMul lines — the kit's own enable call. Package modes map to lines:

  off     stock — no switch exported, no kit directory on the path (``stock_pred.py`` proves it in a clean subprocess)
  exact line ``S1``: the kit's base line ``S`` = ``LEV_ENV`` (CHANGES.md "Levers") — the served-levers hook,
          the cuEquivariance tuning-cache location, the DITFAST levers ``dit_hoist,dit_align`` and ARM ``Z`` installed after the
          runner exists — plus FPF TriMul EXACT enabled in-process by the kit's own ``fpf_engines.enable_from_env()`` at the
          exported contract, plus the house ``ODDE_SERVED_LEVERS_STRICT=1``. Under the deterministic recipe (``det.py``) the line is
          byte-identical to stock — `bf16_fastln`, the base every mode runs — file for file (the kit's test T3b is the same check).
  fast    line ``LSTAR2A``: the kit's base line ``S`` + ARM ``U`` + the SAMPLER unit's sampler stack with the arm's partner
          bf16 C=384 TriMul cell (``ODDE_ARM_U2_TRIMUL=1``, lever ``arm_u23``) and the arm's binds extended to the structural refiner and
          the confidence head's pairformer (``ODDE_ARM_T_SCOPE=all``); tier 2, numerics-changing by construction.
  big   line ``BIG_F``: ``fast`` without ``alloc_auto`` + the memory levers of the adapter
          (``big.py`` over ``opt_core.mem``): ``pair_offload`` (the offload unit: pair tensors in pinned host RAM; size-gated, with
          ``sample_chunk``, and turning fast's ``chunk_lift`` and ``keep_pool`` off — below its gate the line is fast's resident lever set), ``no_dit_hoist``
          (size-gated), ``sample_chunk``; the
          expandable-segments allocator; tier 2. At ``--n_gpu P>1`` the mode is line ``BIG_TP``:
          the same lever set with every pair track row-sharded over the P rank processes (``tp.py``) in place of the offload unit.

A mode is one line: the four lines above are the whole table. Nothing in this file is a lever value the kit does not state itself: every export is the kit's
own README spelling, cited per lever in ``registry.py``; ``configs/<gpu>.env`` carries deployment
parameters only. One composition is size-gated on the two single-card lines: below the small-input floor (``smalln.py``,
``MODEL_OPT_SMALL_INPUT_FLOOR_TOKENS`` = ARM's 300-token design gate as shipped) on every item of a ``pred`` call, the trunk levers
(``smalln.FLOOR_LEVERS``: ``arm_z`` | ``arm_u`` + ``arm_u23``, ``fpf_trimul_exact``, and the triangle-attention / TriMul / LayerNorm /
transition provider words) are composed out of the line
(``line_without``) — the trunk kernels are stock there by the units' own gates and the binds can only cost host time; below the floor ``exact``
and ``fast`` are then one trunk composition (the hook, the tile cache, the DITFAST hoist); the sampler-autocast word ``sampler_amp`` rides no default line (speed ~0 on top of the sampler kernels; registry row).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from . import registry
from . import MODES                               # the house mode names, spelled once in the package root (with the env route's names)

# kit directories under opendde/opt/ — the one place they are spelled
KIT_DIRS = {
    "fast_inference": "forward/fast_inference",   # the kit: levers/{ACCEL,ARMT,DITFAST,KIT,LNSTREAM,OFFLOAD,SAMPLER,XL}, src/
}                                                 # the XL and offload units live beside the kit's own levers (levers/XL, levers/OFFLOAD): one key
                                                  # (fast_inference) covers every line path (the cross-check builder's kit_dir_of rule)
# the kit's directories in path order (`PYTHONPATH=levers/ACCEL:levers/ARMT:levers/DITFAST/tools:levers/KIT:src`)
ACCEL = "fast_inference/levers/ACCEL"             # the hook (sitecustomize.py -> odde_served_levers.arm_hook) + the ACCEL_V2 levers
ARMT = "fast_inference/levers/ARMT"               # the ARM U/Z add-on (K2B kernels third party; the TriMul / tri-attention provider bindings)
DITFAST_TOOLS = "fast_inference/levers/DITFAST/tools"   # odde_addon.py (dit_hoist, dit_align)
KIT_LAYER = "fast_inference/levers/KIT"           # the kit layer: the cuEquivariance tuning-cache location, no table shipped (cueq_cache_shipped/
SRC = "fast_inference/src"                        # fpf_engines (the exact line's op adapter)
KIT_PATH_ORDER = (ACCEL, ARMT, DITFAST_TOOLS, KIT_LAYER, SRC)
XL = "fast_inference/levers/XL"                   # OPENDDE_XL_ADDON (lever tri_ln, exact): odde_xl.py +
                                                  # its shim, LAST on the path of the lines that carry the XL levers: ACCEL's shim chains
XL_PATH_ORDER = KIT_PATH_ORDER + (XL,)            # to it (the XL hook installs at the model module's import); every shim chains forward from its own
                                                  # path position under one re-entry guard: each runs once per process, in path order
OFFLOAD = "fast_inference/levers/OFFLOAD"         # the offload unit (host-resident pair tensors streamed in row/column blocks): SECOND on the
                                                  # big path — ACCEL's shim chains to it (its shim imports the model,
OFFLOAD_PATH_ORDER = (ACCEL, OFFLOAD, ARMT, DITFAST_TOOLS, KIT_LAYER, SRC)   # installs, then chains forward to the next shim on the path)
SAMPLER = "fast_inference/levers/SAMPLER"         # the SAMPLER unit (the diffusion-sampler kernel levers): no shim — the ACCEL lever routes
                                                  # to it from install_dit_attn (levers/ACCEL/odde_accel_v2.py) and resolves the sibling directory itself
HOOK = ACCEL                                      # every kit line's hook: its sitecustomize.py FIRST on the path; `$KIT` in an export = the kit root (KIT_DIRS["fast_inference"])
SHIM = "sitecustomize.py"                         # the hook's shim file name

ARM_SWITCHES = ("ODDE_ARM_Z", "ODDE_ARM_U", "ODDE_ARM_U2_TRIMUL", "ODDE_ARM_U3", "ODDE_ARM_T_TRIMUL", "ODDE_ARM_T_SCOPE",
                "ODDE_TRIATTN",                     # levers/ARMT/odde_triattn_bind.py: the arm's attention site through the core provider by tier word (fast | exact | big | <row>)
                "ODDE_TRIATTN_CONF",                # ablation of lever triattn_conf only (MODEL_OPT_LEVERS_OFF=triattn_conf -> =stock); no line exports it (the confidence head binds the tier word on every line)
                "ODDE_TRIMUL",                      # levers/ARMT/odde_trimul_bind.py: the two TriMul routes through the core provider (fast | exact | big | <row>)
                "ODDE_TRANSITION")                  # levers/ARMT/odde_transition_bind.py: the pair-transition sites through the core provider (fast | exact | big | <row>)
FPF_SWITCHES = ("FPF_ENGINE", "FPF_OPS", "FPF_IMPL", "FPF_CROSSCHECK", "FPF_CROSSCHECK_REF", "FPF_STRICT", "FPF_DIMS")   # the fpf_engines adapter's switches (the kit
                                                       # carries no TriMul kernel copy: the former FPF_TRIMUL_* knobs of src/fpf_trimul are no kit switch)
TEST_HOOKS = ("ODDE_SERVED_DETERMINISTIC", "ODDE_SERVED_KEEP_RAW", "DIT_HOIST_CROSSCHECK", "ODDE_HOST_CROSSCHECK",
              "ODDE_D59_TRUNK_HASH", "ODDE_D59_OBS_OUT",
              "ODDE_SAMPLER_PROBE")                                                              # the SAMPLER unit's shape census                                         # never exported by a mode; det.py owns the first
OTHER_LINE_SWITCHES = ("ODDE_DIT_ATTN", "ODDE_QUEUE", "ODDE_SUBPROCESS", "ODDE_SHARD",
                       "ODDE_LN")                        # opendde_opt/lncore.py: the pair-row LayerNorms through the core LN provider by tier word (fast | big); absent on the exact line (no bitwise row: the extension by name)
SAMPLER_SWITCHES = ("ODDE_ATOM_ATTN", "ODDE_COND_DEDUPE", "ODDE_DIT_FUSED", "ODDE_DIT_LOWP", "ODDE_ATOM_FUSED",   # the SAMPLER unit's (+ ODDE_DIT_ATTN=<word> above); the last two are retired spellings
                    "ODDE_DIT_ATTN_EXACT", "ODDE_DIT_ATTN_FP16")     # (superseded by ODDE_DIT_ATTN=exact / by the provider cell's own fp16-operand choice): exported by no line, unset by every line
SUPERSEDED_SWITCHES = ()                          # switches of levers upstream supersedes on the pin (registry.PIN_STATUS): in every line's must-unset pool, exported by none
OFFLOAD_SWITCHES = ("ODDE_OFFLOAD", "ODDE_OFFLOAD_STRICT", "ODDE_OFFLOAD_ROWS", "ODDE_OFFLOAD_CC", "ODDE_OFFLOAD_MEMFRAC", "ODDE_OFFLOAD_XBUF", "ODDE_OFFLOAD_PIN",
                    "ODDE_OFFLOAD_PITCHED", "ODDE_OFFLOAD_DIFFZ", "ODDE_OFFLOAD_DIFFZ_PERM", "ODDE_OFFLOAD_DIFFZ_RELEASE", "ODDE_OFFLOAD_DIFFZ_ROWS",
                    "ODDE_OFFLOAD_DCHUNK", "ODDE_OFFLOAD_FORCE_CHUNK", "ODDE_OFFLOAD_FREE_TEMPL", "ODDE_OFFLOAD_BIGLN_LIMIT", "ODDE_OFFLOAD_ZINIT",
                    "ODDE_OFFLOAD_PWA_W", "ODDE_OFFLOAD_OPM_ROWS", "ODDE_OFFLOAD_DISTO", "ODDE_OFFLOAD_LAZY_RELP", "ODDE_OFFLOAD_CYCLE_CKPT",
                    "ODDE_OFFLOAD_CKPT_DIR", "ODDE_OFFLOAD_RESUME", "ODDE_OFFLOAD_CKPT_MAX_GB", "ODDE_OFFLOAD_LOG", "ODDE_TRAJ_EVERY")   # the offload unit's (+ its traj hook's)
                                                       # switches: the big offload line exports the _OFFLOAD_EXPORTS subset, the rest stay in every line's must-unset pool
XL_SWITCHES = ("ODDE_XL", "ODDE_XL_STRICT", "ODDE_XL_CROSSCHECK", "ODDE_XL_PROBE")          # the XL unit's switches; the first two exported, the rest absent
ALLOCATOR = "PYTORCH_CUDA_ALLOC_CONF"             # absent for every line but one: the kit runs on the default allocator; the big line exports its
                                                  # own (Line.allocator, applied by the package for the kit process only; the stock route forbids it)
STRICT = {"ODDE_SERVED_LEVERS_STRICT": "1"}       # every house arm: an install failure raises instead of falling back to stock

# the served-levers hook's environment (`$KIT` = the kit root) + the house strict switch
_S_EXPORTS = {
    "ODDE_SERVED_LEVERS": "1",
    "CUEQ_TRITON_CACHE_DIR": "$KIT/levers/KIT/cueq_cache_shipped",
    "ODDE_ADDON_LEVERS": "dit_hoist,dit_align",
    "ODDE_ARM_Z": "1",
    **STRICT,
}
UPSTREAM_EXPOSED_UNUSED = ("OPENDDE_FORCE_SAMPLE_DIFFUSION_AMP", "OPENDDE_FORCE_CONFIDENCE_AMP")   # upstream's exposed, deliberately unused switches (runner/inference.py:1511-1514:
                                                # force bf16 autocast in the diffusion sampler / confidence head below upstream's own 3840 / 2560-token precision policy) — absent on
                                                # EVERY route: every line unsets them here; the stock caller strips and proves them absent (stock/PINS.json must_be_absent_prefixes)
                                                # — the configs value the first one sets (skip_amp.sample_diffusion=False) is what the house lever sampler_amp
                                                # applies on fast / big at upstream's policy site WITHOUT the variable (opendde_opt/precision.py)
_ALL_SWITCHES = tuple(dict.fromkeys(tuple(_S_EXPORTS) + UPSTREAM_EXPOSED_UNUSED + ARM_SWITCHES + OTHER_LINE_SWITCHES + SUPERSEDED_SWITCHES + FPF_SWITCHES + XL_SWITCHES
                                    + OFFLOAD_SWITCHES + TEST_HOOKS + (ALLOCATOR, "CUEQ_TRITON_TUNING")
                                    + SAMPLER_SWITCHES))   # every switch a line may export or must unset
_S_UNSET = tuple(k for k in _ALL_SWITCHES if k not in _S_EXPORTS)
# the offload unit at its shipped defaults, every knob declared (levers/OFFLOAD/odde_offload.py:32-59, odde_offload_trunk.py:40-52): all three
# stages host-resident; no checkpoint dir (the per-cycle checkpoint/resume is robustness, off the memory line), no log file, no trajectory hook;
# strict: an install failure of the unit ends the interpreter with NOT ACTIVE, exit 3 (never stock stages under an offload banner);
# ODDE_OFFLOAD_DCHUNK / ODDE_OFFLOAD_FORCE_CHUNK (upstream bounds the chunk size) and ODDE_OFFLOAD_LAZY_RELP (upstream's relp is always lazy)
# are retired in the unit and stay in the must-unset pool
_OFFLOAD_EXPORTS = {"ODDE_OFFLOAD": "all", "ODDE_OFFLOAD_STRICT": "1", "ODDE_OFFLOAD_ROWS": "256", "ODDE_OFFLOAD_CC": "auto", "ODDE_OFFLOAD_MEMFRAC": "0.80", "ODDE_OFFLOAD_XBUF": "host",
                    "ODDE_OFFLOAD_PIN": "1", "ODDE_OFFLOAD_PITCHED": "1", "ODDE_OFFLOAD_DIFFZ": "1", "ODDE_OFFLOAD_DIFFZ_PERM": "1", "ODDE_OFFLOAD_DIFFZ_RELEASE": "1",
                    "ODDE_OFFLOAD_DIFFZ_ROWS": "0", "ODDE_OFFLOAD_FREE_TEMPL": "1",
                    "ODDE_OFFLOAD_BIGLN_LIMIT": str(1 << 31), "ODDE_OFFLOAD_ZINIT": "recompute", "ODDE_OFFLOAD_PWA_W": "full", "ODDE_OFFLOAD_OPM_ROWS": "0",
                    "ODDE_OFFLOAD_DISTO": "rows", "ODDE_OFFLOAD_CYCLE_CKPT": "1"}
_OFFLOAD_LEVERS = ("pair_offload_struct", "pair_offload_trunk", "pair_offload_conf", "diffz", "bigln_guard", "free_templ")
EXPANDABLE = "expandable_segments:True"           # the allocator of the big line (set by the package's activation for the kit process only; the stock route forbids it)
# the XL unit's exact-class lever (levers/XL/odde_xl.py: tri_ln = the LN'd pair copy freed before chunked triangle attention; bitwise = stock,
# memory only); strict: an install exception raises
_XL_EXACT = {"ODDE_XL": "tri_ln", "ODDE_XL_STRICT": "1"}
_XL_LEVERS = ("xl_tri_ln",)
# The exact line's TriMul: the fpf_engines op adapter (src/fpf_engines: binds the two stock TriMul forwards in-process, FPFFallback = the stock forward
# by name, counted) whose callables are the core-provider binding's (levers/ARMT/odde_trimul_bind.py) -- lever fpf_trimul_exact = the adapter
# enabled at its environment contract; lever trimul_exact = the provider's exact tier word. The kit carries no TriMul kernels of its own.
_FPF_EXACT = {
    "FPF_ENGINE": "opendde",
    "FPF_OPS": "trimul_out,trimul_in",
    "FPF_IMPL": "trimul_out=odde_trimul_bind:trimul_out,trimul_in=odde_trimul_bind:trimul_in",
}
_TRANSITION_EXACT = {"ODDE_TRANSITION": "exact"}                             # lever transition_exact: arm Z applies the pair-transition binding under the provider's exact word (levers/ARMT/odde_transition_bind.py)
_TRIMUL_EXACT = {"ODDE_TRIMUL": "exact"}                                    # lever trimul_exact: the word the binding's callables serve by; without it they raise FPFFallback (stock by name)


def _with(exports: dict, add: dict) -> dict:
    return {**exports, **add}


def _unset_but(*keep: str) -> tuple:
    return tuple(k for k in _S_UNSET if k not in keep)


def _unset_for(exports: dict) -> tuple:
    """Every switch of the pool a line does not export."""
    return tuple(k for k in _ALL_SWITCHES if k not in exports)


@dataclass(frozen=True)
class Line:
    name: str
    hook: str | None                  # kit sub-path whose directory is FIRST on the path (its shim loads the levers); None = no hook
    path_order: tuple                 # directories on the path in order, as kit sub-paths spelled on a KIT_DIRS key
    exports: dict                     # switch -> value; `$KIT` expands to the kit root's absolute path (KIT_DIRS["fast_inference"])
    unset: tuple                      # switches that must be absent for the line
    levers: tuple                     # registry.LEVERS names the line turns on
    tier: str                         # "exact" | "tolerance" | "tier2"
    note: str = ""
    fpf: str | None = None            # "exact" | "fast": the package calls the kit's `fpf_engines.enable_from_env()` after the shim (in-process routes only)
    allocator: str | None = None      # PYTORCH_CUDA_ALLOC_CONF for the kit process (None = the default allocator, the switch absent)


_S_LEVERS = ("served_levers_hook", "cueq_tuned_cache", "dit_hoist", "dit_align", "arm_z",
             "drop_bond_mask",                     # the tree's own lever (opendde_opt/bondmask.py): every line carries it
             "lnstream",                           # the tree's own lever (opendde_opt/lnstream.py): upstream's fused LayerNorm on the current stream, every line (placement-neutral, bitwise)
             "tmpl_dedup", "keep_pool",            # the tree's own levers (opendde_opt/tmpldedup.py: identical template slots embedded once; keeppool.py: the in-forward empty_cache sites kept — on the big lines below the offload size gate only, BIG_KIT_ROWS_OFF / BIG_TP_DROP)
             "sched_host",                         # the tree's own lever (opendde_opt/schedhost.py): the sampler loop's two per-step host round-trips answered on the host / made non-blocking
             "structok_sync",                      # the tree's own lever (opendde_opt/structoksync.py): the structural-token expander's role-pair projections with one host read per call
             "json_oneshot",                       # the tree's own lever (opendde_opt/writer_overlap.py): upstream's save_json encodes each confidence document once with json.dumps (the C encoder) and writes once — identical bytes
             "prefetch",                           # the tree's own lever (opendde_opt/prefetch.py): upstream's DataLoader with one worker (num_workers 0 -> 1) — the next item featurised while this one runs
             "zprep_hoist")                        # the tree's own lever (opendde_opt/zprephoist.py): the denoiser's per-step (N,N,c)->(c,N,N) pair copy made once per sampler call (memo on the hoisted source)
_U_LEVERS = tuple(x for x in _S_LEVERS if x != "arm_z") + ("arm_u",)
# ---- levers/SAMPLER: the sampler levers. `_SP_FAST_*` = the fast-tier sampler stack (the two attention sites through the core pair-bias-attention
# provider by its TIER WORD — levers/SAMPLER/odde_apb_bind.py, opt_core.kernels.apb TIER_WORDS; the big lines say `big` (big_line) — + the fused
# conditioning / token / atom schedules); `_SP_EXACT_*` = the same token site's statement through the provider's exact tier (ODDE_DIT_ATTN=exact).
SAMPLER_FAST_WORD, SAMPLER_BIG_WORD, SAMPLER_EXACT_WORD = "fast", "big", "exact"
_SP_FAST_EXPORTS = {"ODDE_DIT_ATTN": SAMPLER_FAST_WORD, "ODDE_ATOM_ATTN": SAMPLER_FAST_WORD, "ODDE_COND_DEDUPE": "1", "ODDE_DIT_FUSED": "1", "ODDE_DIT_LOWP": "fp16", "ODDE_ATOM_FUSED": "1"}
_SP_FAST_LEVERS = ("dit_attn_apb", "atom_attn_apb", "cond_dedupe", "dit_fused", "dit_lowp", "atom_fused")
_SP_EXACT_EXPORTS = {"ODDE_DIT_ATTN": SAMPLER_EXACT_WORD}
_SP_EXACT_LEVERS = ("dit_attn_exact",)
SAMPLER_LEVERS = _SP_FAST_LEVERS + _SP_EXACT_LEVERS          # every lever of the unit (registry rows; each has a LEVER_SWITCHES ablation rule)

_TRIATTN_EXACT = {"ODDE_TRIATTN": "exact", "ODDE_ARM_T_SCOPE": "all"}   # lever triattn_exact (+ triattn_conf): arm Z's attention site through the core provider's exact tier on every pair
                  # stack the arm binds, the confidence head's included (the tier word goes straight to the provider's select(); a cell whose exact row is the stock op is served by the stock op by the cell's name)
LINES: dict[str, Line] = {
    "S1": Line("S1", HOOK, KIT_PATH_ORDER, _with(_S_EXPORTS, {**_FPF_EXACT, **_TRIATTN_EXACT, **_SP_EXACT_EXPORTS, **_TRIMUL_EXACT, **_TRANSITION_EXACT}), _unset_but(*_FPF_EXACT, *_TRIATTN_EXACT, *_SP_EXACT_EXPORTS, *_TRIMUL_EXACT, *_TRANSITION_EXACT),
               _S_LEVERS + ("fpf_trimul_exact", "alloc_auto") + ("stepgraph", "triattn_exact", "triattn_conf") + _SP_EXACT_LEVERS + ("trimul_exact", "transition_exact"), "exact",   # no xl_tri_ln: measured trunk cost (registry row);   # no chunk_lift: not bitwise at every size (registry row)
               "S + the core TriMul provider's exact tier (rows bit-identical to the stock cuEquivariance TriMul on this stack, at every row count; "
               "bitwise = stock under the kit DET recipe, CHANGES.md) through the fpf_engines "
               "adapter enabled in-process by the package (the kit's `enable_from_env()` at its environment contract) + the expandable-segments "
               "allocator for the kit process of a `pred` call (alloc_auto; placement only) + the SAMPLER unit's bit-exact replacement of the sampler's fp32 "
               "SDPA statement through the core provider's exact tier (dit_attn_exact, ODDE_DIT_ATTN=exact; from 0.2.40; bound at every size)", fpf="exact"),
    "LSTAR2A": Line("LSTAR2A", HOOK, KIT_PATH_ORDER, _with(_S_EXPORTS, {"ODDE_ARM_U": "1", "ODDE_ARM_U2_TRIMUL": "1", "ODDE_ARM_T_SCOPE": "all", "ODDE_TRIATTN": "fast", **_SP_FAST_EXPORTS, "ODDE_TRIMUL": "fast", "ODDE_LN": "fast", "ODDE_TRANSITION": "fast"}),
                    _unset_but("ODDE_ARM_U", "ODDE_ARM_U2_TRIMUL", "ODDE_ARM_T_SCOPE", "ODDE_TRIATTN", *_SP_FAST_EXPORTS, "ODDE_TRIMUL", "ODDE_LN", "ODDE_TRANSITION"), _U_LEVERS + ("alloc_auto", "arm_u23", "chunk_lift", "stepgraph", "triattn_core", "triattn_conf") + _SP_FAST_LEVERS + ("trimul_core", "ln_core", "transition_core"), "tier2",   # no sampler_amp: speed ~0 on top of the sampler stack (registry row);   # no XL unit: under scope=all the arm binds every TriangleAttention module, so the unit's tri_ln prologue has no call site
                    "S + ARM U + bf16 DiT attention with the partner bf16 C=384 TriMul cell inside ARM U's region (the arm's U2 lever `arm_u23`) and the arm's binds — bf16 region, "
                    "transpose-free block, cast-once prologue, sep16 transition, K2B attention, TriMul — extended to the structural refiner and the confidence "
                    "head's pairformer (ODDE_ARM_T_SCOPE=all, ERRATA_06); tier 2. From 0.2.40 the sampler runs the SAMPLER unit's stack in place of the bf16 DiT-attention "
                    "recast: dit_attn_apb, atom_attn_apb (both through the core provider's fast tier by word from 0.2.57), cond_dedupe, dit_fused (+dit_lowp=fp16), atom_fused — bound at every size (not floor levers)"),
}
# ---- the small-input floor (opt/opendde_opt/smalln.py): the trunk levers of the single-card exact and fast lines, and fast's bf16 DiT attention,
# composed OUT of the line for a `pred` call whose every item counts fewer residue tokens than MODEL_OPT_SMALL_INPUT_FLOOR_TOKENS — not exported,
# not installed — because below ARM's own 300-token design gate the trunk kernels are stock anyway and the binds can only cost host time, and the
# DiT attention's bf16 casts cost more than bf16 saves there (CHANGES.md "Modes")
SMALL_LINES = ("S1", "LSTAR2A", "BIG_F")         # the lines the floor composes: exact, fast and the single-card big line (below its offload gate it is fast's resident
                                                    # set, so a small query composes the trunk levers out exactly like fast; BIG_TP keeps its own size gates)
LEVER_SWITCHES: dict[str, tuple] = {               # floor lever -> the switches that bind it on those lines (composed out = absent from the exports, in the unset pool)
    "arm_z": ("ODDE_ARM_Z",),
    "arm_u": ("ODDE_ARM_U", "ODDE_ARM_T_SCOPE", "ODDE_ARM_Z"), "arm_u23": ("ODDE_ARM_U2_TRIMUL",),   # ARM U carries Z: the arm out = every ODDE_ARM_* switch out, the unit never imported
    "trimul_core": ("ODDE_TRIMUL",),                                                     # the TriMul provider word out = the arm's TriMul route runs the stock forward by name
    "transition_core": ("ODDE_TRANSITION",), "transition_exact": ("ODDE_TRANSITION",),   # the transition word out = the pair transition stays the engine's module (the binding and its singleton not applied)
    "trimul_exact": ("ODDE_TRIMUL",),                                                    # the word out: the adapter's callables (odde_trimul_bind:trimul_out|in) raise FPFFallback = the stock forward by name
    "ln_core": ("ODDE_LN",),                                                             # the LayerNorm provider word out = every FusedLayerNorm call on the extension (opendde_opt/lncore.py never installed)
    "triattn_core": ("ODDE_TRIATTN",),                                                   # the provider word out = the arm's attention site keeps the stock op (attention=stock)
    "triattn_exact": ("ODDE_TRIATTN", "ODDE_ARM_T_SCOPE"),                              # (the exact word carries the scope switch on the exact line)
    "triattn_conf": (),                                                                  # no switch of its own: left out by a value (LEVER_OFF_EXPORTS: ODDE_TRIATTN_CONF=stock)
    "fpf_trimul_exact": tuple(_FPF_EXACT),                                               # + the in-process enable (Line.fpf) off
    "xl_tri_ln": tuple(_XL_EXACT),                                                       # + the XL unit's directory off the path
    "dit_attn_bf16": ("ODDE_DIT_ATTN",),
    "chunk_lift": (),                                                                    # the house chunk lever: no switch (its ceiling composes it out: chunklift.compose)
    "lnstream": (),                                                                      # the house LayerNorm-stream lever: no switch (composed out by name: MODEL_OPT_LEVERS_OFF, ablate.py)
    "tmpl_dedup": (), "keep_pool": (),                                                   # the house template de-dup / keep-pool levers: no switch (installed from stack._apply; ablated by name)
    "structok_sync": (),                                                                 # the house structural-token sync lever: no switch (composed out by name: MODEL_OPT_LEVERS_OFF)
    "zprep_hoist": (),                                                                   # the house pair-prep hoist: no switch (composed out by name: MODEL_OPT_LEVERS_OFF)
    "sched_host": (),                                                                    # the house sampler host-sync lever: no switch (composed out by name: MODEL_OPT_LEVERS_OFF)
    "sampler_amp": (),                                                                   # the house sampler-autocast word: no switch (off the lever list = not installed; left out by name: MODEL_OPT_LEVERS_OFF=sampler_amp)
    "writer_overlap": (),                                                                # the house result-writer lever (carried by no line: measured, CHANGES.md): no switch
    "json_oneshot": (),                                                                  # the house JSON-writer lever: no switch (composed out by name: MODEL_OPT_LEVERS_OFF)
    "prefetch": (),                                                                      # the house featurisation-prefetch lever: no switch (composed out by name: MODEL_OPT_LEVERS_OFF)
    "stepgraph": (),                                                                     # the house sampler step-graph lever: no switch (its size gate / the offload line compose it out: stepgraph.compose)
    # ablation by name (MODEL_OPT_LEVERS_OFF, ablate.py): every other lever's rule, so any single lever can be left out of a line
    "drop_bond_mask": (), "alloc_auto": (),                                              # house levers: off the lever list = not installed / not exported
    "cueq_tuned_cache": ("CUEQ_TRITON_CACHE_DIR",),                                      # the library's default cache location instead of the kit layer's
    "served_levers_hook": ("ODDE_SERVED_LEVERS", "ODDE_SERVED_LEVERS_STRICT", "ODDE_ADDON_LEVERS"),   # the installer: cascades every lever it installs (LEVER_DEPENDENTS)
    "dit_hoist": (), "dit_align": (),                                                    # one value switch (LEVER_VALUE_SWITCHES: ODDE_ADDON_LEVERS=<list>)
    "struct_pair_bf16": (), "tp_triatt": (),                                             # BIG_TP: turned off by an export value (LEVER_OFF_EXPORTS), not by absence
    "no_dit_hoist": (), "sample_chunk": (),                                              # big memory levers: the big line rebuilt without them (big_line)
    "pair_offload_trunk": (), "pair_offload_struct": (), "pair_offload_conf": (),        # the offload unit's stages: ODDE_OFFLOAD=<remaining stages>; none left = the unit off the line
    "free_templ": (), "diffz": (),                                                       # the unit's sub-levers: an export value turns them off (LEVER_OFF_EXPORTS)
    # levers/SAMPLER (each individually ablatable; a precision word / fused stack leaves with the lever it requires: LEVER_DEPENDENTS)
    "dit_attn_apb": ("ODDE_DIT_ATTN",), "atom_attn_apb": ("ODDE_ATOM_ATTN",), "cond_dedupe": ("ODDE_COND_DEDUPE",),
    "dit_fused": ("ODDE_DIT_FUSED",), "dit_lowp": ("ODDE_DIT_LOWP",), "atom_fused": ("ODDE_ATOM_FUSED",), "dit_attn_exact": ("ODDE_DIT_ATTN",),   # one switch, the word picks the site (exact | fast | big | <row>)
}
LEVER_VALUE_SWITCHES: dict[str, tuple] = {         # lever -> (switch, token): the token removed from the switch's comma list (the switch unexported when the list empties)
    "dit_hoist": ("ODDE_ADDON_LEVERS", "dit_hoist"), "dit_align": ("ODDE_ADDON_LEVERS", "dit_align"),
}
LEVER_OFF_EXPORTS: dict[str, dict] = {             # lever -> the exports that turn it OFF by value (levers a line turns on by a value, not by presence)
    "struct_pair_bf16": {"ODDE_TP_STRUCT_PAIR_DTYPE": "fp32"},
    "triattn_conf": {"ODDE_TRIATTN_CONF": "stock"},    # read by the binding only while ODDE_TRIATTN is exported (LEVER_OFF_EXPORTS_WITH)
    "tp_triatt": {"ROWPAIR_TRIATT_CORE": "torch"},
    "free_templ": {"ODDE_OFFLOAD_FREE_TEMPL": "0"},
    "diffz": {"ODDE_OFFLOAD_DIFFZ": "0"},
}
LEVER_OFF_EXPORTS_WITH: dict[str, str] = {"triattn_conf": "ODDE_TRIATTN"}   # lever -> the export its off-value rides on: absent that export (the parent left out, the floor), the off-value is not exported either
LEVER_DEPENDENTS: dict[str, tuple] = {             # lever -> the levers that go with it when it is left out (installed by it / a property of it)
    "served_levers_hook": ("dit_hoist", "dit_align", "arm_z", "arm_u", "arm_u23", "dit_attn_bf16") + SAMPLER_LEVERS,       # + the SAMPLER unit's levers (installed from the hook's DiT-attention route)
    "dit_hoist": ("dit_align",),                                                         # align is a property of the hoisted storage (the addon re-adds the hoist for align alone)
    "arm_u": ("arm_u23", "triattn_core", "triattn_conf", "trimul_core", "transition_core"),             # the provider bindings ride inside the arm's attention wrapper / U2 route / module bind
    "arm_z": ("triattn_exact", "triattn_conf", "transition_exact"),
    "triattn_core": ("triattn_conf",), "triattn_exact": ("triattn_conf",),                # the confidence-stack sub-lever rides the word
    "arm_u23": ("trimul_core",),                                                         # the TriMul provider binding rides inside the U2 route
    "fpf_trimul_exact": ("trimul_exact",),                                               # ... and inside the exact line's FPF adapter
    "diffz": ("bigln_guard",),
    "dit_attn_apb": ("dit_fused", "dit_lowp"), "atom_attn_apb": ("atom_fused",), "dit_fused": ("dit_lowp",),   # SAMPLER: the fused token stack needs the provider-served token site; the fused atom stacks the atom site; lowp qualifies dit_fused
    "lnstream": ("stepgraph",),                                                          # the step graph records upstream's LayerNorm launches only on the current stream: without lnstream it cannot capture (shed by name with it)
}
LEVER_NOT_SHEDDABLE: dict[str, str] = {           # levers without an individual rule, with the reason (ablate.py names it)
    "rowpair_tp": "it is the --n_gpu P>1 line itself (run --n_gpu 1 instead)",
    "bigln_guard": "a correctness guard of the offload unit's diffz path (torch 2.7.1 LayerNorm 2^32-element limit), left out together with diffz",
}
_OFFLOAD_STAGE_OF_ROW = {"pair_offload_trunk": "trunk", "pair_offload_struct": "struct", "pair_offload_conf": "conf"}


def line_without(line: Line, levers: tuple) -> Line:
    """The line ``line`` with the floor levers ``levers`` composed OUT: their switches (LEVER_SWITCHES) absent from the exports and in the
    must-unset pool, their rows off the lever list, the FPF in-process enable off with ``fpf_trimul_exact``, the XL unit's directory off the
    path with ``xl_tri_ln``. Levers the line does not carry are ignored; an empty drop returns the line itself (the static row); a lever
    without a rule here is refused (ValueError), never silently kept."""
    drop = [x for x in levers if x in line.levers]
    if not drop:
        return line
    fixed = [x for x in drop if x in LEVER_NOT_SHEDDABLE]                  # the NAMED levers need a rule (or their reason); the levers that go with them need none
    if fixed:
        raise ValueError(f"line {line.name}: {', '.join(f'{x} cannot be left out alone: {LEVER_NOT_SHEDDABLE[x]}' for x in fixed)}")
    unknown = [x for x in drop if x not in LEVER_SWITCHES]
    if unknown:
        raise ValueError(f"line {line.name}: no composition rule for lever(s) {unknown} (modes.LEVER_SWITCHES)")
    for x in list(drop):                                                   # a lever left out takes the levers that go with it (LEVER_DEPENDENTS), transitively
        stack = list(LEVER_DEPENDENTS.get(x, ()))
        while stack:
            d = stack.pop()
            if d in line.levers and d not in drop:
                drop.append(d)
                stack.extend(LEVER_DEPENDENTS.get(d, ()))
    mem = [x for x in drop if line.name in BIG_LINES and x in BIG_MEM_LEVERS.get(line.name, ())]    # big memory levers: the line rebuilt without them
    stage_rows = [x for x in drop if x in _OFFLOAD_STAGE_OF_ROW and "ODDE_OFFLOAD" in line.exports]
    if mem or stage_rows:
        on = [m for m in BIG_MEM_LEVERS[line.name] if m not in mem and _mem_lever_on(line, m)]
        stages = [s for s in (line.exports.get("ODDE_OFFLOAD") or "").replace("all", "trunk,struct,conf").split(",") if s]
        stages = [s for s in stages if s not in {_OFFLOAD_STAGE_OF_ROW[r] for r in stage_rows}]
        if stage_rows and not stages:
            on = [m for m in on if m != "pair_offload"]                    # every stage left out: the unit off the line
        base = big_line(line.name, tuple(on), note=line.note)
        if "pair_offload" in on and stage_rows:
            ex = dict(base.exports); ex["ODDE_OFFLOAD"] = ",".join(stages)
            staged = {r for st in stages for r in _OFFLOAD_STAGE_ROWS.get(st, ())} | {"diffz", "bigln_guard"}
            lv = tuple(x for x in base.levers if x not in _OFFLOAD_LEVERS or x in staged)
            base = Line(base.name, base.hook, base.path_order, ex, _unset_for(ex), lv, base.tier, base.note, fpf=base.fpf, allocator=base.allocator)
        rest = tuple(x for x in drop if x not in mem and x not in stage_rows)
        out = line_without(base, rest) if rest else base
        return out if out.levers != line.levers or out.exports != line.exports else line
    keep = tuple(x for x in line.levers if x not in drop)
    exports = {k: v for k, v in line.exports.items() if not any(k in LEVER_SWITCHES.get(x, ()) for x in drop)}
    for x in drop:                                                         # value switches: the lever's token out of the list; an empty list unexports the switch
        sw, tok = LEVER_VALUE_SWITCHES.get(x, (None, None))
        if sw and sw in exports:
            toks = [t for t in exports[sw].split(",") if t and t != tok]
            if toks:
                exports[sw] = ",".join(toks)
            else:
                del exports[sw]
        for k, v in LEVER_OFF_EXPORTS.get(x, {}).items():                  # levers turned off by a value
            exports[k] = v
    for x, need in LEVER_OFF_EXPORTS_WITH.items():                         # ... a value that only means something beside its parent's export: dropped with the parent
        if need not in exports:
            for k in LEVER_OFF_EXPORTS.get(x, {}):
                exports.pop(k, None)
    fpf = line.fpf if "fpf_trimul_exact" not in drop else None
    path = line.path_order if "xl_tri_ln" not in drop else tuple(d for d in line.path_order if d != XL)
    return Line(line.name, line.hook, path, exports, _unset_for(exports), keep, line.tier, line.note, fpf=fpf, allocator=line.allocator)


def _mem_lever_on(line: Line, m: str) -> bool:
    """Is the big memory lever ``m`` in force on this (possibly composed) big line? Read from the line itself: the offload unit by its switch,
    the hoist's removal by the hoist rows' absence, the others by the registry rows they turn on."""
    if m == "pair_offload":
        return bool(line.exports.get("ODDE_OFFLOAD"))
    if m == "no_dit_hoist":
        return "dit_hoist" not in line.levers
    rows = BIG_KIT_ROWS.get(m, ())
    return any(r in line.levers for r in rows) if rows else False


# ---- the big mode's lines: composed on `fast` by the adapter over opt_core.mem (opt/opendde_opt/big.py) ---------------------------
BIG_BASE_MODE = "fast"                           # the mode big composes on (big = fast's lever set + memory levers)
BIG_BASE = "LSTAR2A"                              # = MODE_LINES[BIG_BASE_MODE] (asserted below the mode table)
BIG_LINES = ("BIG_F", "BIG_TP")                # the mode's line and its `--n_gpu P>1` line
BIG_OFFLOAD_LINES = ("BIG_F",)                 # the lines with the offload unit on their path (OFFLOAD_PATH_ORDER while pair_offload is among the line's levers — below its size gate the path is fast's KIT_PATH_ORDER; the XL unit is not: the offload unit carries its own lazy relp)
BIG_OFFLOAD_LINE = "BIG_F"
BIG_TP_LINE = "BIG_TP"                         # `--mode big --n_gpu P>1` (opendde_opt/tp.py + tp_struct.py + tp_diffusion.py): every pair track row-sharded over the ranks; neither the offload nor the XL unit on the path
TP_EXPORTS = {"TORCH_NCCL_AVOID_RECORD_STREAMS": "1", "ROWPAIR_TRANSPOSE_INPLACE": "1", "ODDE_TP_STRUCT_PAIR_DTYPE": "bf16", "ROWPAIR_PARK_ZINIT": "1", "ROWPAIR_PARK_ZRES": "1", "ODDE_TP_STRUCT_TRIMUL_RB": "128", "ROWPAIR_NCCL_TIMEOUT_S": "600", "ROWPAIR_RANK_THREADS": "auto", "ROWPAIR_TRIATT_STAGE": "once"}   # + struct_pair_bf16: the structural pair shard and its refiner pair stack in bf16 (a caller's ODDE_TP_STRUCT_PAIR_DTYPE=fp32 wins, named); + the ending-orientation transpose swapping blocks inside the shard's own storage (no second shard-sized buffer); the row-sharded line's rank processes: collective operands are not record_stream'd on the NCCL streams (the core's
                                                   # collectives are blocking, operand lifetime is the caller's), so the caching allocator reuses their blocks at once
                                                   # instead of mapping new memory behind still-"active" blocks (reserved tracks allocated; the driver view scales with 1/P)
TP_STRUCT_BF16_LEVER = "struct_pair_bf16"                # BIG_TP: z_struct shard + refiner pair stack in bf16 (ODDE_TP_STRUCT_PAIR_DTYPE=bf16|fp32)
TP_LEVER = "rowpair_tp"                            # the registry row of the row-sharded pair stack (tp.LEVER is this name)
TP_KERNEL_LEVERS = ("tp_triatt",)                  # BIG_TP: the sharded pair stack's row-block kernel (opendde_opt/tp_kernels.py: flash tri-attention on the core's dispatch)
BIG_MEM_LEVERS: dict[str, tuple] = {             # line -> its memory levers (opt_core.mem registry names, registered by big.py), in apply order; the static row = every lever on (big.plan gates pair_offload + sample_chunk and no_dit_hoist by input size)
    "BIG_F": ("pair_offload", "no_dit_hoist", "sample_chunk"),
    "BIG_TP": ("no_dit_hoist", "sample_chunk"),
}
BIG_OFFLOAD_STAGES = {"BIG_F": "all"}           # line -> the offload unit's stages (`ODDE_OFFLOAD`); `BIG_TP` carries no offload stage: under `--n_gpu P>1` every pair
                                                   # track (trunk, template, MSA module, structural stage, diffusion conditioning, heads) is row-sharded across the ranks (rowpair_tp)
_OFFLOAD_STAGE_ROWS = {"struct": ("pair_offload_struct", "free_templ"), "trunk": ("pair_offload_trunk",), "conf": ("pair_offload_conf",)}   # stage -> its registry rows (diffz,
                                                   # bigln_guard come with any stage)
BIG_DROP = ("alloc_auto", "zprep_hoist",                      # fast's levers off every big line at EVERY size: a big line's allocator is its fixed field; zprep_hoist's persistent
              "dit_fused", "dit_lowp")                           # (c,N,N) pair copy and the fused token stack + its precision word (the stack's resident packed fp16 weight copy) are resident-memory
                                                                 # costs — off every big line by name with their switches (ODDE_DIT_FUSED / ODDE_DIT_LOWP unset); fast keeps them; the sampler's other
                                                                 # levers stay. The sampler step graph `stepgraph` is NOT in this tuple: it rides the resident set below the offload size gate like chunk_lift
                                                                 # and keep_pool — the graph replays the line's own fp32 unfused step up to the big word's row (stepgraph.WORD_ROWS); inside a graphed sampler
                                                                 # call the step's tier-word bindings choose as under capture (stepgraph.planning(): the pair-bias attention's graph-timed provider column and
                                                                 # upstream's LayerNorm); above that row the replayed fp32 rows are slower than the eager fp16-operand rows and the graph pool costs resident
                                                                 # memory, so the eager sampler serves there — and it leaves with the offload unit (BIG_KIT_ROWS_OFF["pair_offload"]: the unit streams the pair
                                                                 # tensors the step reads) and off the row-sharded line (BIG_TP_DROP); its private graph pool is the memory it costs below the gate
                                                                 # (stepgraph.py: transient inside one sampler call, bounded by the call's memory admission ODDE_STEPGRAPH_MEMFRAC)
BIG_TP_DROP = ("chunk_lift", "stepgraph", "keep_pool", "sched_host", "structok_sync", "tmpl_dedup", "zprep_hoist",  # fast's levers off the row-sharded line: its ranks run the core's pair stack, upstream's chunk clamp kept; the rank processes' sampler is not graphed
                 "writer_overlap",                  # writer_overlap: carried by no line (its measured cost); json_oneshot STAYS on the row-sharded line: every rank's writer
                                                    # binds it (rank 0 writes the run's documents incl. the [N, N] full_data maps, ranks >= 1 their own under .rowpair/rank<r>/) — identical
                                                    # bytes by construction (the C encoder over the same document), the pure-Python encoder's per-item seconds off rank 0's tail
                 "prefetch")                        # prefetch: the row-sharded ranks read every item in the rank process itself (tp.py: a DataLoader worker is refused there by name)
BIG_KIT_ROWS: dict[str, tuple] = {               # memory lever -> the registry rows it turns on (registry.LEVERS names)
    "pair_offload": _OFFLOAD_LEVERS, "no_dit_hoist": ("no_dit_hoist",), "sample_chunk": ("sample_chunk",)}
BIG_KIT_ROWS_OFF: dict[str, tuple] = {"no_dit_hoist": ("dit_hoist", "dit_align"),   # memory lever -> the registry rows it turns off (the offload unit keeps upstream's
                                          "pair_offload": ("chunk_lift", "stepgraph", "keep_pool", "tmpl_dedup")} # chunk clamp / keep-pool: chunk_lift, keep_pool ride the line below the offload size gate only; the sampler step graph too — the unit streams the pair tensors the step reads)
_BIG_NOTES = {
    "BIG_F": "the big mode: fast (LSTAR2A: ARM U with the bf16 TriMul and the scope binds, bf16 DiT attention) without alloc_auto + the offload unit at its "
               "shipped defaults from the offload size gate (pair_offload: trunk z, structural z and the confidence pair stack in pinned host RAM, streamed through the GPU "
               "in 256-row / column blocks, TriMul chunked to fit; diffz, free_templ, the torch-2.7.1 LayerNorm 2^32 guard) + the DITFAST hoist left out at >= its size "
               "gate (no_dit_hoist) + the diffusion samples one chunk at a time with the offload unit (sample_chunk), upstream's chunk clamp and in-forward releases kept with it (chunk_lift, keep_pool off); below the offload "
               "size gate the line is fast's resident lever set, chunk_lift and keep_pool included; expandable-segments allocator; tier 2; host RAM sized per input",
    "BIG_TP": "the big mode at `--n_gpu P>1`: fast without alloc_auto, chunk_lift, keep_pool, the XL unit and the offload unit + the row-sharded pair "
                "tracks over the P rank processes (rowpair_tp: pair init, trunk, template, MSA-module, structural-stage, diffusion-conditioning and "
                "confidence pair statements on this rank's row block of z through opt_core.mem.rowpair; the levers the offload unit serves on one card — "
                "structural z placement, diffz, the LayerNorm guard, free_templ — are served by the row shard) + no_dit_hoist at >= the size gate + "
                "sample_chunk; expandable-segments allocator; tier 2 vs P=1 (tiled contractions)",
}


BIG_TRIMUL_WORD = "big"                        # the TriMul provider's tier word on every big line (levers/ARMT/odde_trimul_bind.py; opt_core.kernels.trimul TIER_WORDS)
BIG_TRIATTN_WORD = "big"                       # the triangle-attention provider's tier word on every big line (ODDE_TRIATTN=big; levers/ARMT/odde_triattn_bind.py; opt_core.kernels.triattn TIER_WORDS)
BIG_TRANSITION_WORD = "big"                    # the transition provider's tier word on every big line (levers/ARMT/odde_transition_bind.py; opt_core.kernels.transition TIER_WORDS)
BIG_LN_WORD = "big"                            # the LayerNorm provider's tier word on every big line (opendde_opt/lncore.py; opt_core.kernels.ln TIER_WORDS)
BIG_WORD_SWITCHES = {"ODDE_TRIMUL": BIG_TRIMUL_WORD, "ODDE_DIT_ATTN": SAMPLER_BIG_WORD, "ODDE_ATOM_ATTN": SAMPLER_BIG_WORD,   # provider-bound switches the big lines export with the provider's OWN big word
                       "ODDE_LN": BIG_LN_WORD, "ODDE_TRANSITION": BIG_TRANSITION_WORD, "ODDE_TRIATTN": BIG_TRIATTN_WORD}                       # (TriMul, the sampler's two attention sites, the pair-row LayerNorms, the pair transition, triangle attention)


def big_line(name: str, mem_levers: tuple, settings: dict | None = None, note: str = "") -> Line:
    """The big line ``name`` composed from its memory levers (``big.compose`` calls this at the flags in force; the static rows below
    are the levers' defaults): the base line's exports without the XL unit's, plus each lever's own switches; the base line's registry levers minus BIG_DROP, minus the rows a lever turns off, plus the rows it
    turns on; the hook first on the path; the expandable-segments allocator. ``settings`` = {(memory lever, setting): value} for a
    lever's declared settings (the shipped rows pass none: every setting at its default)."""
    settings = settings or {}
    base = LINES[BIG_BASE]
    exports = dict(base.exports)
    for _sw, _w in BIG_WORD_SWITCHES.items():                                  # the memory mode binds the providers by THEIR OWN tier word: the core's big cells (a row
        if _sw in exports:                                                       # measured for peak memory as well as time)
            exports[_sw] = _w
    levers = [x for x in base.levers if x not in BIG_DROP]
    exports = {k: v for k, v in exports.items() if k not in _XL_EXACT              # no XL unit on the offload line and on the row-sharded line;
               and not any(k in LEVER_SWITCHES.get(x, ()) for x in BIG_DROP)}  # a dropped lever's own switches leave with it (unset by _unset_for)
    levers = [x for x in levers if x not in _XL_LEVERS]
    path = OFFLOAD_PATH_ORDER if (name in BIG_OFFLOAD_LINES and "pair_offload" in mem_levers) else KIT_PATH_ORDER   # the unit's directory on the path only with its lever (big.plan: off below the offload size gate)
    if "no_dit_hoist" in mem_levers:
        exports.pop("ODDE_ADDON_LEVERS", None)                                   # the DITFAST addon's lever switch not exported (and unset if present)
    if name == BIG_TP_LINE:
        levers = [x for x in levers if x not in BIG_TP_DROP]
        exports.update(TP_EXPORTS)
        levers = levers + [TP_STRUCT_BF16_LEVER]                                  # the structural pair stack in bf16 under n_gpu>1 (registry struct_pair_bf16; tp_struct.py)
    stages = BIG_OFFLOAD_STAGES.get(name, "all")
    if "pair_offload" in mem_levers:
        exports.update(_OFFLOAD_EXPORTS)
        exports["ODDE_OFFLOAD"] = stages
        for switch, s in (("ODDE_OFFLOAD_ROWS", "rows"), ("ODDE_OFFLOAD_CC", "cc"), ("ODDE_OFFLOAD_MEMFRAC", "memfrac"), ("ODDE_OFFLOAD_PIN", "pin")):
            if settings.get(("pair_offload", s)):
                exports[switch] = str(settings[("pair_offload", s)])
        if exports.get("ODDE_OFFLOAD_PIN") not in ("0", "1"):                     # the setting's whole domain, refused by name outside it
            raise ValueError(f"refused: pair_offload pin={exports.get('ODDE_OFFLOAD_PIN')!r}; one of 1 (pinned host buffers; a refused pinned allocation is an error), 0 (pageable host buffers) is required")
    for m in mem_levers:
        levers = [x for x in levers if x not in BIG_KIT_ROWS_OFF.get(m, ())]
    for m in mem_levers:
        levers += [x for x in BIG_KIT_ROWS.get(m, ()) if x not in levers]
    if "pair_offload" in mem_levers and stages != "all":                         # the unit's rows of the stages this line runs (diffz / bigln_guard with any stage)
        staged = {r for st in stages.split(",") for r in _OFFLOAD_STAGE_ROWS.get(st, ())} | {"diffz", "bigln_guard"}
        levers = [x for x in levers if x not in _OFFLOAD_LEVERS or x in staged]
    if name == BIG_TP_LINE:
        levers.append(TP_LEVER)
        levers += list(TP_KERNEL_LEVERS)
    return Line(name, HOOK, path, exports, _unset_for(exports), tuple(levers), "tier2", note or _BIG_NOTES[name], allocator=EXPANDABLE)


LINES.update({n: big_line(n, BIG_MEM_LEVERS[n]) for n in BIG_LINES})

MODE_LINES: dict[str, str | None] = {"off": None, "exact": "S1", "fast": "LSTAR2A", "big": "BIG_F"}   # mode -> line name (the kit owner's rows)
DETERMINISM: dict[str, str] = {                    # the determinism statement of every mode
    "off": "the stock's own `--det` (torch deterministic algorithms; no kit switch)",
    "exact": "bitwise vs stock UNDER the deterministic recipe (`det.py`, `--det 1`); at shipped defaults (`--det 0`): band — the composition replaces the stock's kernels (FPF TriMul fixed-order TF32, the shipped cuEq tile cache, ARM Z), a different deterministic order than stock's own run-to-run variation",
    "fast": "the deterministic recipe under `--det 1` (the bf16/K2B kernels deterministic per call); `--det 0` = production numerics",
    "big": "the deterministic recipe under `--det 1`: the offload's host copies are exact and ordered (kit-vs-kit `--det 1` bitwise on line BIG: a12, a17) and no_dit_hoist runs the stock op per step (line BIG_B bitwise = fast under the recipe); sample_chunk's per-chunk noise draws and the fast base's bf16 kernels are tier 2 at any `--det`",
}
assert MODE_LINES[BIG_BASE_MODE] == BIG_BASE          # big composes on the fast mode's line
DEFAULT_MODE = "fast"                                                              # the package default; the env switch unset = off


class OpenModeError(ValueError):
    """A selection refused by name (the base of big.BigRefusal and smalln.SmallFloorRefusal): reported NOT ACTIVE, never resolved to a candidate silently."""


@dataclass
class Resolution:
    mode: str | None                  # the house mode, or None when a line was asked for by name
    line: Line | None                 # None for `off`
    tree: str                         # opendde/ directory
    exports: dict = field(default_factory=dict)     # switch -> value with `$KIT` expanded
    unset: tuple = ()
    sys_path: list = field(default_factory=list)    # absolute kit directories in path order (the line's PYTHONPATH)
    hook_dir: str | None = None
    shim: str | None = None           # the hook's sitecustomize.py (absolute)
    levers: tuple = ()
    notes: list = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.mode or (self.line.name if self.line else "off")

    def pythonpath(self) -> str:
        return os.pathsep.join(self.sys_path)


def check_mode(mode: str) -> str:
    """A mode name of this package (case-folded); an unknown name raises ValueError."""
    m = (mode or "").strip().lower()
    if m not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {', '.join(MODES)}")
    return m


def kit_dir(tree: str, key: str) -> str:
    """Absolute directory for a KIT_DIRS key or a sub-path spelled on a KIT_DIRS key (``fast_inference/levers/KIT``, ``fast_inference/src``)."""
    head, _, rest = key.partition("/")
    if head not in KIT_DIRS:
        raise KeyError(f"unknown kit key {key!r}")
    return os.path.join(tree, "opt", KIT_DIRS[head], *([rest] if rest else []))


def kit_layer(tree: str) -> str:
    return kit_dir(tree, KIT_LAYER)


def is_mode_name(mode_or_line: str) -> bool:
    """A selection names a MODE when it is a house mode name case-insensitively and not a kit line by its exact (upper-case) name:
    ``big`` is the mode, ``BIG`` the kit line — the one rule for every reader of a selection (line_of, resolve, the activation report)."""
    key = (mode_or_line or "").strip()
    return key not in LINES and key.lower() in MODES


def line_of(mode_or_line: str) -> Line | None:
    """The Line for a house mode (``off`` -> None) or a line by its own name; raises ValueError for an unknown name."""
    key = (mode_or_line or "").strip()
    if key in LINES:                                   # a kit line by its exact name (`BIG` the line; `big` the mode)
        return LINES[key]
    if key.lower() in MODES:
        ln = MODE_LINES[key.lower()]
        return LINES[ln] if ln else None
    if key.upper() in LINES:
        return LINES[key.upper()]
    raise ValueError(f"unknown mode or line {mode_or_line!r}; modes: {', '.join(MODES)}; lines: {', '.join(LINES)}")


def resolve(mode_or_line: str, tree: str, environ: dict | None = None) -> Resolution:
    """Materialise a mode or a named line against the tree: exports with ``$KIT`` expanded, the path order as absolute directories,
    the hook's shim. Raises OpenModeError for a selection that the size gates refuse by name and ValueError for an unknown name; never touches ``os.environ``."""
    env = os.environ if environ is None else environ
    key = (mode_or_line or "").strip()
    is_mode = is_mode_name(key)
    line = line_of(key)
    if line is not None and line.name in BIG_LINES:          # the memory mode's lines are composed at the package's size-gate switch (big.compose); a caller's OPENDDE_BIG_* variable refuses by name
        from . import big as _big
        line = _big.compose(line.name, env)
    if line is not None and line.name in SMALL_LINES and not (set(line.levers) & set(_OFFLOAD_LEVERS)):   # exact / fast / big below its offload gate are composed at the small-input floor (smalln.compose): every item of the
        from . import smalln as _smalln                          # call below MODEL_OPT_SMALL_INPUT_FLOOR_TOKENS -> the floor's levers out; a malformed variable refuses by name
        line = _smalln.compose(line, env)
    if line is not None and "chunk_lift" in line.levers:       # the chunk lever's ceiling (chunklift.compose): composed in only when every item of the call counts
        from . import chunklift as _chunklift                    # <= the table's un-chunked gate; above it / mixed / no query read -> composed out (reason above_gate:<t>/<g>)
        line = _chunklift.compose(line)
    if line is not None and "stepgraph" in line.levers:        # the sampler step-graph's size gate and offload rule (stepgraph.compose): composed in only when every item counts
        from . import stepgraph as _stepgraph                    # within [ODDE_STEPGRAPH_MIN_TOKENS, ODDE_STEPGRAPH_MAX_TOKENS] and the line runs no offload stage
        line = _stepgraph.compose(line)
    from . import stockknob as _stockknob                        # upstream's --triatt_kernel / --trimul_kernel stated (other than auto): the kit levers of that site composed
    line = _stockknob.compose(line, env)                         # out by name (registry.STOCK_KNOB_LEVERS; LEVER state=off reason=aside:stock_knob:<flag>=<value>); else the line itself
    from . import ablate as _ablate                              # MODEL_OPT_LEVERS_OFF: the named levers composed out of this call's line (line_without); an unknown name,
    line = _ablate.compose(line, env)                            # a lever the line does not carry or cannot shed, or any name under `off` refuses by name; unset: the line itself
    res = Resolution(mode=key.lower() if is_mode else None, line=line, tree=tree)
    if line is None:
        res.notes.append("stock: no switch exported, no kit directory on the path")
        return res
    if line.name in BIG_LINES and line is not LINES[line.name]:
        res.notes.append(f"big: {line.note}")
    kr = kit_dir(tree, "fast_inference")
    res.exports = {k: v.replace("$KIT", kr) for k, v in line.exports.items()}
    res.unset = tuple(k for k in line.unset if not (line.allocator and k == ALLOCATOR))
    res.sys_path = [kit_dir(tree, k) for k in line.path_order]
    if line.hook:
        res.hook_dir = kit_dir(tree, line.hook)
        res.shim = os.path.join(res.hook_dir, SHIM)
    elif line.path_order:
        res.hook_dir = res.sys_path[0]
        res.shim = os.path.join(res.hook_dir, SHIM)
    res.levers = tuple(line.levers)
    for k in res.unset:
        if env.get(k) not in (None, ""):
            res.notes.append(f"{k}={env.get(k)!r} is set in the environment and must be absent for line {line.name}: the package unsets it")
    if line.allocator:
        res.exports[ALLOCATOR] = line.allocator     # the line's own allocator for the kit process (never on the stock route)
    elif env.get(ALLOCATOR):
        res.notes.append(f"{ALLOCATOR}={env.get(ALLOCATOR)!r} is set and line {line.name} runs on the default allocator: the package unsets it")
    return res


def describe_line(res: Resolution) -> str:
    """The activation line's spelling of a resolution: line name + hook + exports (paths shortened to the kit key)."""
    if res.line is None:
        return "off"
    parts = []
    for k, v in res.exports.items():
        if v.startswith(res.tree):
            v = "$MODEL_OPT" + v[len(res.tree):]
        parts.append(f"{k}={v}")
    hook = f"hook={os.path.basename(res.line.hook)}" if res.line.hook else "hook=none"
    return f"{res.line.name}({hook}; {' '.join(parts)})"


def jit_cache_key() -> str:
    """torch<version sans local tag>-cu<CUDA version sans dot>-sm<compute capability digits>, e.g. torch2.7.1-cu126-sm90 — the key the JIT
    cache volume is laid out by (configs/<gpu>.env MODEL_OPT_STACK_KEY): the core's one rule (opt_core.jit_cache.key; torch's distribution
    metadata and nvidia-smi, nothing imported)."""
    import sys as _sys
    from opt_core import jit_cache
    try:
        return jit_cache.key()
    except jit_cache.StackKeyUnknown as e:                   # a GPU-less / metadata-less interpreter (a CPU `check` box): the display form, its unknown
        k = jit_cache.key(strict=False)                       # part spelled `unknown` INSIDE the key, named on stderr — never a silent whole-key fallback
        _sys.stderr.write(f"[opendde-opt] stack key {k!r} has an unknown part on this interpreter ({e}); the JIT cache is addressed by the full key on a GPU box\n")
        return k


# ---------------------------------------------------------------------------------------------------------------- the KERNELS census's expectations
_KNOB_SITES = {"triatt_kernel": "cueq_triatt", "trimul_kernel": "cueq_trimul"}   # == settings.KERNEL_KNOBS (upstream's flag -> the census site it selects; tests hold the equality)
RUN_FACTS = ("ODDE_STOCK_KNOBS",)                        # per-run facts the PACKAGE writes into the kit process after activation (stockknob.export) — never a line's export, never
                                                         # unset by a line, never a user's switch: the stated stock knobs, read by the ARM add-on at install and by the rank processes
KERNEL_SITES = ("cueq_triatt", "cueq_trimul")            # == lncensus.KERNEL_SITES (lncensus imports nothing of this table; cli / stock_pred hand it these rows)


def route_word(line, mode: str | None, n_gpu: int = 1) -> str:
    """The `route=` token of the KERNELS line: stock (the stock caller) | exact | fast | big | line-<NAME>, with `_x<P>` appended for a P-GPU pass."""
    if line is None:
        base = "stock"
    else:
        base = mode if mode in ("exact", "fast", "big") else f"line-{line.name}"
    return base + (f"_x{int(n_gpu)}" if int(n_gpu or 1) > 1 else "")


def kernel_expectations(line, *, ln_requested: bool, n_gpu: int = 1, knobs: dict | None = None) -> dict:
    """The expected KERNELS words of a route, from THIS table (the single source of truth the REQUIRE guard reads): accelerator ->
    {kind: engaged | off-by-route, reason, resolved (the triangle-kernel value upstream must resolve to), need_stack (the cuEquivariance stack
    must import and its kernels run on the device)}. `line` = the resolved Line (None = the stock caller); `ln_requested` = the environment asks for
    LAYERNORM_TYPE=fast_layernorm (settings.ln_requested); `n_gpu` = the pass's GPU count (the row-sharded line's ranks; the stock caller runs at 1);
    `knobs` = the triangle-kernel flags the caller STATED with a value other than `auto` (settings.stock_knobs): that site's word is the caller's —
    kind ``user``, ``user:<flag>=<value>``, nothing required of the cuEquivariance stack and no resolved value expected (never a refusal), on every route."""
    exp = {"fast_layernorm": ({"kind": "engaged", "reason": None} if ln_requested else
                              {"kind": "off-by-route", "reason": "LAYERNORM_TYPE_unset(upstream_default_torch:layers.py:25-30)"})}
    for site in KERNEL_SITES:
        owners = [lv for lv in (line.levers if line is not None else ()) if lv in registry.KERNEL_SITE_OWNERS[site]]
        exp[site] = {"kind": "off-by-route" if owners else "engaged", "reason": "+".join(owners) if owners else None,
                     "resolved": "cuequivariance", "need_stack": True}
        flag = next((f for f, st in _KNOB_SITES.items() if st == site), None)
        if flag and (knobs or {}).get(flag):                                    # upstream's own flag stated for this site: as the caller says, by name
            exp[site] = {"kind": "user", "reason": f"{flag}={knobs[flag]}", "resolved": None, "need_stack": False}
    return exp
