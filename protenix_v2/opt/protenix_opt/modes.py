"""Modes and the environment each one materialises.

env.sh is the one switch table: `resolve()` sources the kit's `opt/forward/flashpairformer/env.sh` (ARM=E for "exact", ARM=T for
"fast") in a bash subprocess on top of the caller's environment and takes the delta — the same variables, values, defaults and unsets
the kit's own route produces, including env.sh's own probes (nvidia-smi compute capability; the prebuilt fast-LN for the
installed torch, selected from the kit's own third_party/fastln_prebuilt*/ directories). Two things are set here and nowhere
else: the kit README's per-(cc | triton) row switches around env.sh, and lazy init. PYTHONPATH is not exported: `stack.kit_sys_path()`
carries env.sh's entries into sys.path.
"""
import importlib.metadata
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

from . import _core  # noqa: F401
from opt_core import jit_cache as _jit_cache

MODES: Dict[str, List[str]] = {
    "exact": [                                    # E* (env.sh ARM=E + the E* row switches of the kit README) + DEADSKIP + lazy init
        "layernorm_fast", "cueq_tuned_tiles", "template_dedupe", "t1_fused_transition", "nomask", "blk2_block_path", "pwa_zcache",
        "blk2_chunked_exact", "deadskip", "lever_report", "trimul_core_exact", "stackgraph", "sampler_graph", "sampler_graph_cache_policy", "sampler_prep", "sampler_reach",
        "sampler_fuse", "fastln_prebuilt", "xl_policy", "pad8", "glue_v2", "mk_pf", "lazy_init", "keep_pool", "summary_hostidx", "dit_attn_exact", "triatt_exact", "atom_attn_exact", "transition_core_exact", "triatt_prologue_cuda",
        "sampler_admit", "pred_release",
        "templ_trimul_tmk3",        # the template embedder's c = 64 pair stack through the shared core (src/ptx_c64_routes.py)
    ],
    "fast": [                                     # T* (env.sh ARM=T: K2B attention, smalln routing + the T* row switches) + lazy init
        "layernorm_fast", "cueq_tuned_tiles", "template_dedupe", "t1_fused_transition", "nomask", "blk2_block_path", "pwa_zcache",
        "blk2_chunked_k2b", "k2b_flash_triattention", "trimul_core", "smalln_size_gate", "deadskip", "lever_report",
        "stackgraph", "sampler_graph", "sampler_graph_cache_policy", "sampler_prep", "sampler_reach", "sampler_fuse", "fastln_prebuilt", "xl_policy", "glue_v2", "mk_pf", "lazy_init", "keep_pool", "summary_hostidx",
        "dit_attn", "dit_attn_fp16", "atom_attn", "pf_attn", "opm_fused", "pwa_fused", "cond_dedupe", "dit_fused", "dit_lowp", "atom_fused", "atom_attn_exact", "dit_attn_exact", "transition_core", "triatt_prologue_cuda",                                                                  # the sampler attention levers (T* rows of cc 9.0: README_ROWS; dit_attn_fp16 = the precision lever on dit_attn)
        "triattn_native",                                                                                           # the block core's Tier-2 attention through opt_core.kernels.triattn by the tier word fast, every card with cells (README_ROWS pre-export the tier word PTX_T_ATT=fast on cc 9.0 and cc 8.0; big exports PTX_T_ATT=big in its place, readme_row)
        "sampler_admit", "pred_release",
        "templ_trimul_esm", "templ_triatt_core",         # the template embedder's c = 64 pair stack through the shared core (src/ptx_c64_routes.py)
        "ln_core",                                       # the model's standalone LayerNorms through the shared core's LN provider by tier word (PACKAGE_POST PTX_LN_TIER=fast; src/protenix_ptx_ln_core.py)
    ],
    "big": [                                    # fast's row without BIG_DROPPED + the runner-hook and memory levers (big_levers)
        "layernorm_fast", "cueq_tuned_tiles", "template_dedupe", "t1_fused_transition", "nomask", "blk2_block_path", "pwa_zcache",
        "blk2_chunked_k2b", "k2b_flash_triattention", "trimul_core", "smalln_size_gate", "deadskip", "lever_report",
        "sampler_graph", "sampler_graph_cache_policy", "sampler_prep", "sampler_reach", "sampler_fuse", "fastln_prebuilt", "xl_policy", "glue_v2", "mk_pf", "lazy_init", "summary_hostidx",   # without BIG_DROPPED (no graph pools, no PAD8, no keep_pool), then
        "dit_attn", "dit_attn_fp16", "atom_attn", "pf_attn", "opm_fused", "pwa_fused", "cond_dedupe", "dit_fused", "dit_lowp", "atom_fused", "atom_attn_exact", "dit_attn_exact", "transition_core", "triatt_prologue_cuda", "triattn_native",
        "sampler_admit", "pred_release",
        "templ_trimul_esm", "templ_triatt_core",         # the template embedder's c = 64 pair stack through the shared core (src/ptx_c64_routes.py)
        "ln_core",                                       # as fast, word big (PACKAGE_POST PTX_LN_TIER=big)
        "guard_lift",                                                                                              # the package's own runner-hook lever, then
        "drop_bond_mask", "cond_chunk", "apb_bias_chunk", "cache_release", "relp_lazy", "msa_zfree", "diffcache_free",     # MEM_LEVERS (big.py), then
    ],                                            # literal is validated == big_levers() (tests/test_big.py)
    "off": [],                                    # stock protenix: nothing set, nothing applied
}
MODE_NAMES: Tuple[str, ...] = ("off", "exact", "fast", "big")   # the modes this kit ships, the stock word first — ONE literal line (tools outside this tree parse it); == set(MODES) (tests/test_big.py)
# big (memory) = the base arm's (fast's) lever set without BIG_DROPPED, plus the package's runner-hook lever and every memory
# lever (`big_levers()`). What the row changes against its base, by the constant that does it:
#   BIG_PRE (exported before env.sh; a caller's own value wins): PTX_TRANSITION=big (lever transition_core's tier word);
#     PTX_FPF_CHUNK_TOK=auto (the XL lean tri-attention statement above env.sh's own device-memory-keyed token threshold, the same
#     `auto` exact / fast use, stated here so the row reads whole; up to the threshold the block core's own attention kernel serves the
#     item whole, above it the lean statement runs the same core per stock row-chunk); PTX_BLK_GRAPH=0 (no trunk stack-graph pool);
#     PTX_SAMPLER_GRAPH_MAXTOK=1 (the graph route's token cap below every input: no sampler CUDA graph; `sampler_admit` (admit=item)
#     drives the DiT hoist eagerly per item, and the graphed-sampler modules stay installed for the hoist, its host path and reach).
#   BIG_DROPPED: the base's levers those switches would have applied, and keep_pool, are not in the row (pad8, stackgraph,
#     keep_pool); every other lever of the base is carried (the fused DiT token stack dit_fused + dit_lowp and pf_attn among them).
#   BIG_POST_DROPPED: the README row's PTX_E_PAD8 exports are not made (the unpadded stock cuEquivariance call at every N).
#   BIG_POST (after env.sh): the stock n_token guard is lifted (guard_lift, the package's own runner hook) and the kept allocator
#     pool is released (PTX_KEEP_POOL=0: stock's empty_cache() releases in the confidence head stay).
#   MEM_LEVERS: the memory levers of big.py (on the shared core's registry opt_core.mem) are added.
BIG_PRE: Dict[str, str] = {"PTX_TRANSITION": "big", "PTX_FPF_CHUNK_TOK": "auto", "PTX_BLK_GRAPH": "0", "PTX_SAMPLER_GRAPH_MAXTOK": "1"}   # see the block above; set before env.sh (a caller value wins)
BIG_DROPPED: Tuple[str, ...] = ("pad8", "stackgraph", "keep_pool")   # the base's levers the row does not apply: pad8 and stackgraph (their switches are not made / turned off: BIG_POST_DROPPED, BIG_PRE) and keep_pool (the memory-first line releases on purpose: BIG_POST sets PTX_KEEP_POOL=0)
BIG_POST_DROPPED: Tuple[str, ...] = ("PTX_E_PAD8", "PTX_E_PAD8_MIN_TOKENS")                                 # the README row's PAD8 exports, not made
BIG_POST: Dict[str, str] = {"PTX_GUARD_LIFT": "1", "PTX_KEEP_POOL": "0"}   # after env.sh: the stock n_token guard lifted (registry `guard_lift`; the package's own runner hook) — big only, never exported by another mode; the kept pool released.
BIG_BASE = "fast"                                    # the arm big composes on
MEM_LEVERS: Tuple[str, ...] = ("drop_bond_mask", "cond_chunk", "apb_bias_chunk", "cache_release", "relp_lazy", "msa_zfree", "diffcache_free")   # big.py's line on the core registry (== big.LINE), in order


RUNNER_LEVERS: Tuple[str, ...] = ("guard_lift",)                   # the package's runner-hook lever(s), in row order
def big_levers() -> List[str]:
    """The big row: the base's (fast's) levers without BIG_DROPPED, then the package's runner-hook lever, then MEM_LEVERS."""
    return [n for n in MODES[BIG_BASE] if n not in BIG_DROPPED] + list(RUNNER_LEVERS) + list(MEM_LEVERS)
ARM: Dict[str, Optional[str]] = {"exact": "E", "fast": "T", "big": "T", "off": None}   # big: its base's arm (fast)

# The kit README's per-(cc | triton) composition table, transcribed as data — opt/forward/flashpairformer/README.md 'Compositions per (cc | triton)' (rows) and
# its 'Order of env operations' line (`export <the row's pre switches>` -> `ARM=<E|T> source env.sh` -> `export PTX_E_PAD8=1 PTX_GLUE_V2=1
# [PTX_MK_PF=...] [PTX_E_PAD8_MIN_TOKENS=512]`). "pre" is exported before env.sh is sourced (env.sh reads it), "post" after it. A value
# the caller already has for any of these variables is honoured (env.sh is sourced on top of the caller's environment; the rows fill only
# what the caller left unset). tests/test_modes_match_env_sh.py parses the shipped README and locks every row against resolve().
README_ROWS: Dict[str, Dict[str, Dict[str, Dict[str, str]]]] = {
    "9.0|3.3": {                                                                     # (H100, torch 2.7.1 / triton 3.3.1)
        "exact": {"pre": {}, "post": {"PTX_E_PAD8": "1", "PTX_GLUE_V2": "1", "PTX_E_PAD8_MIN_TOKENS": "512", "PTX_DIT_ATTN_EXACT": "1", "PTX_TRIATT_EXACT": "1", "PTX_ATOM_ATTN_EXACT": "1", "PTX_TRIATT_PROCUDA": "1"}},
        "fast": {"pre": {"PTX_T_ATT": "fast"}, "post": {"PTX_GLUE_V2": "1", "PTX_DIT_ATTN": "1", "PTX_DIT_ATTN_FP16": "1", "PTX_ATOM_ATTN": "1", "PTX_PF_ATTN": "1", "PTX_OPM_FUSED": "1", "PTX_PWA_FUSED": "1", "PTX_COND_DEDUPE": "1", "PTX_DIT_FAST": "1", "PTX_DIT_LOWP": "fp16", "PTX_ATOM_FAST": "1", "PTX_TRIATT_PROCUDA": "1"}},
    },
    "9.0|3.7": {                                                                     # (H100, torch 2.13 / triton 3.7.1): as above + PTX_MK_PF=F1
        "exact": {"pre": {}, "post": {"PTX_E_PAD8": "1", "PTX_GLUE_V2": "1", "PTX_E_PAD8_MIN_TOKENS": "512", "PTX_MK_PF": "F1", "PTX_DIT_ATTN_EXACT": "1", "PTX_TRIATT_EXACT": "1", "PTX_ATOM_ATTN_EXACT": "1", "PTX_TRIATT_PROCUDA": "1"}},
        "fast": {"pre": {"PTX_T_ATT": "fast"}, "post": {"PTX_GLUE_V2": "1", "PTX_MK_PF": "F1", "PTX_DIT_ATTN": "1", "PTX_DIT_ATTN_FP16": "1", "PTX_ATOM_ATTN": "1", "PTX_PF_ATTN": "1", "PTX_OPM_FUSED": "1", "PTX_PWA_FUSED": "1", "PTX_COND_DEDUPE": "1", "PTX_DIT_FAST": "1", "PTX_DIT_LOWP": "fp16", "PTX_ATOM_FAST": "1", "PTX_TRIATT_PROCUDA": "1"}},
    },
    "8.0|3.7": {                                                                     # (A100, torch 2.13 / triton 3.7.1): the block core's fused tri-attention statement engages on sm80 from CELLS.json blk2_triatt_min_tokens up, with the core packages' own 8.0|3.7 GLUE_V2 / MK-PF F1 cells (entry sm80_t37); below the floor the stock statement runs by name. PAD8 is not set on this card
        "exact": {"pre": {}, "post": {"PTX_GLUE_V2": "1", "PTX_MK_PF": "F1", "PTX_TRIATT_EXACT": "1"}},
        "fast": {"pre": {"PTX_T_ATT": "fast"}, "post": {"PTX_GLUE_V2": "1", "PTX_MK_PF": "F1", "PTX_DIT_ATTN": "1", "PTX_DIT_ATTN_FP16": "1", "PTX_ATOM_ATTN": "1"}},   # T: the shared core's tri-attention provider by the tier word fast inside the statement from the floor up (PTX_T_ATT=fast; opt_core row triattn_native's sm_80 member) and the sampler attention levers dit_attn / dit_attn_fp16 / atom_attn through the kit's binding of the core's attention-with-pair-bias rows (protenix_opt/apb_core.py; big through its base)
    },
    "10.0|3.7": {                                                                    # (B200): PAD8 off on cc >= 10 by design
        "exact": {"pre": {}, "post": {"PTX_GLUE_V2": "1", "PTX_MK_PF": "F1,F3"}},
        "fast": {"pre": {}, "post": {"PTX_GLUE_V2": "1", "PTX_MK_PF": "F1,F3"}},
    },
    "10.3|3.7": {                                                                    # (B300, torch 2.13 / triton 3.7.1): PAD8 off on cc >= 10 by design; GLUE_V2 / MK_PF have no 10.3 cells (they refuse by name if set)
        "exact": {"pre": {}, "post": {}},
        "fast": {"pre": {}, "post": {}},
    },
    # broad rows: a triton not listed above takes its cc's row below (= the newest tested triton's composition on that cc); every
    # switch keeps its own kernel-key gate, so a kernel without cells for the running triton refuses by name rather than run untested
    "9.0|*": {
        "exact": {"pre": {}, "post": {"PTX_E_PAD8": "1", "PTX_GLUE_V2": "1", "PTX_E_PAD8_MIN_TOKENS": "512", "PTX_MK_PF": "F1", "PTX_DIT_ATTN_EXACT": "1", "PTX_TRIATT_EXACT": "1", "PTX_ATOM_ATTN_EXACT": "1", "PTX_TRIATT_PROCUDA": "1"}},
        "fast": {"pre": {"PTX_T_ATT": "fast"}, "post": {"PTX_GLUE_V2": "1", "PTX_MK_PF": "F1", "PTX_DIT_ATTN": "1", "PTX_DIT_ATTN_FP16": "1", "PTX_ATOM_ATTN": "1", "PTX_PF_ATTN": "1", "PTX_OPM_FUSED": "1", "PTX_PWA_FUSED": "1", "PTX_COND_DEDUPE": "1", "PTX_DIT_FAST": "1", "PTX_DIT_LOWP": "fp16", "PTX_ATOM_FAST": "1", "PTX_TRIATT_PROCUDA": "1"}},
    },
    "8.0|*": {
        "exact": {"pre": {}, "post": {"PTX_GLUE_V2": "1", "PTX_MK_PF": "F1", "PTX_TRIATT_EXACT": "1"}},
        "fast": {"pre": {"PTX_T_ATT": "fast"}, "post": {"PTX_GLUE_V2": "1", "PTX_MK_PF": "F1", "PTX_DIT_ATTN": "1", "PTX_DIT_ATTN_FP16": "1", "PTX_ATOM_ATTN": "1"}},
    },
    "10.0|*": {
        "exact": {"pre": {}, "post": {"PTX_GLUE_V2": "1", "PTX_MK_PF": "F1,F3"}},
        "fast": {"pre": {}, "post": {"PTX_GLUE_V2": "1", "PTX_MK_PF": "F1,F3"}},
    },
    "10.3|*": {
        "exact": {"pre": {}, "post": {}},
        "fast": {"pre": {}, "post": {}},
    },
}
OTHER_ROW: Dict[str, Dict[str, Dict[str, str]]] = {"exact": {"pre": {}, "post": {}}, "fast": {"pre": {}, "post": {}}}   # any other card: env.sh's own ARM E / ARM T defaults
OTHER_KEY = "other"


def row_key(key: Optional[str]) -> str:
    """The README row that applies to a kernel key: the exact `<cc>|<triton>` row, else the cc's broad `<cc>|*` row, else "other"."""
    if key in README_ROWS:
        return key
    broad = f"{key.split('|')[0]}|*" if key and "|" in key else None
    return broad if broad in README_ROWS else OTHER_KEY


def readme_row(key: Optional[str], mode: str, base: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    """The README row's pre/post exports for a kernel key and mode ("other" for any key not in the table; nothing for "off"); big's is
    its base's (`base`, default BIG_BASE) with BIG_PRE before env.sh and the PAD8 post exports not made."""
    if mode == "off":
        return {"pre": {}, "post": {}}
    if mode == "big":                                        # the base's row on this key: BIG_PRE before env.sh, the PAD8 post exports not made
        row = README_ROWS.get(row_key(key), OTHER_ROW)[base or BIG_BASE]
        tier = {"PTX_T_ATT": "big"} if row["pre"].get("PTX_T_ATT") == "fast" else {}   # the block core's tri-attention names ITS OWN tier word to the shared core's provider wherever the base row binds the provider by tier word (lever triattn_native; a card without provider cells keeps env.sh's default there)
        return {"pre": {**row["pre"], **BIG_PRE, **tier}, "post": {**{k: v for k, v in row["post"].items() if k not in BIG_POST_DROPPED}, **BIG_POST}}
    return README_ROWS.get(row_key(key), OTHER_ROW)[mode]
LAZY_INIT_DEFAULT = "1"                                                     # honours a pre-set 0
# The graphed diffusion sampler's token cap (registry `sampler_graph`; env.sh's own default is `PTX_SAMPLER_GRAPH_MAXTOK=${PTX_SAMPLER_GRAPH_MAXTOK:-995}`):
# exported BEFORE env.sh for the modes whose row composes the sampler graph (exact, fast; big pre-sets the cap to 1 — BIG_PRE: no graph route there), so items up to
# this many tokens run the captured denoiser step + the DiT hoist instead of the stock sampler. The graph's private pool costs device
# memory, so the larger cap (SAMPLER_GRAPH_MAXTOK) applies on a device with at least SAMPLER_GRAPH_MEM_MIB of total memory (nvidia-smi
# memory.total) or when no device is probed; a smaller device keeps env.sh's own default (SAMPLER_GRAPH_MAXTOK_SMALL), above which the
# stock sampler runs — `sampler_graph_maxtok()`, the value resolve() exports; the unit's `[fpf_clisampler] SAMPLER:on(…)` line prints it as max_tokens= and
# the lever's record carries it (stack.GATE_FIELDS). A caller's own value wins
# (PTX_SAMPLER_GRAPH_MAXTOK=995 restores the kit's own default everywhere; =0 with PTX_SAMPLER_GRAPH=0 disables the graph).
SAMPLER_GRAPH_MAXTOK = "1536"
SAMPLER_GRAPH_MAXTOK_SMALL = "995"                                           # env.sh's own default, kept on a device below SAMPLER_GRAPH_MEM_MIB
SAMPLER_GRAPH_MEM_MIB = 64 * 1024                                            # total device memory at and above which SAMPLER_GRAPH_MAXTOK holds


def sampler_graph_maxtok(memory_mib: Optional[int]) -> str:
    """The sampler-graph token cap for a device with `memory_mib` MiB of memory (None: no device probed -> SAMPLER_GRAPH_MAXTOK)."""
    return SAMPLER_GRAPH_MAXTOK if memory_mib is None or memory_mib >= SAMPLER_GRAPH_MEM_MIB else SAMPLER_GRAPH_MAXTOK_SMALL


# The exact TriMul's structural launch edge, kept as named constants: its projection stages A and C launch on a CUDA grid whose axis 1 holds
# N·⌈N/BM⌉ programs and gridDim.y is at most 65535, so with the (bf16, c_z = 256) tile BM = 128 the construction launches iff N ≤ 2849
# (TRIMUL_EXACT_NMAX, as a string: the form an FPF_TRIMUL_EXACT_NMAX export takes). resolve() does not export it: PACKAGE_PRE below
# carries only the sampler-graph cap.
CUDA_GRID_Y_MAX = 65535                                                      # CUDA gridDim.y limit
TRIMUL_EXACT_TILE_BM = 128                                                   # BM of the exact TriMul's stages A and C for the (bf16, c_z = 256) tile


def grid_y_edge(bm: int, grid_y_max: int = CUDA_GRID_Y_MAX) -> int:
    """Largest N with N·⌈N/bm⌉ ≤ grid_y_max: the exact TriMul's launch edge on CUDA grid axis 1."""
    n = int((grid_y_max * bm) ** 0.5) + bm
    while n * -(-n // bm) > grid_y_max:
        n -= 1
    return n


TRIMUL_EXACT_NMAX = str(grid_y_edge(TRIMUL_EXACT_TILE_BM))                   # 2849 at BM = 128
PACKAGE_PRE: Dict[str, Dict[str, str]] = {"exact": {"PTX_SAMPLER_GRAPH_MAXTOK": SAMPLER_GRAPH_MAXTOK},
                                           "fast": {"PTX_SAMPLER_GRAPH_MAXTOK": SAMPLER_GRAPH_MAXTOK}}   # set before env.sh (a caller value wins); big takes its base's entry (BIG_PRE on top)
# The package's own post exports of the exact and fast rows, made after env.sh with the README row's post exports, and big's through its
# base: the diffusion transformer's fused elementwise kernels (registry `sampler_fuse`, opt/protenix_opt/sampler_fuse.py) — on in every kit
# mode (the sampler graph of exact / fast and the eager sampler of big alike). A caller's own value wins (PTX_SAMPLER_FUSE=0 leaves the lever off: a recorded opt-out, `off_by_flag` in the report, not a fallback).
PACKAGE_POST: Dict[str, Dict[str, str]] = {                                                                    # + pred_release (every mode) and the sampler admission words of the speed rows
    "exact": {"PTX_SAMPLER_FUSE": "1", "PTX_PRED_RELEASE": "1", "PTX_SAMPLER_ADMIT": "memory", "PTX_SAMPLER_RELEASE": "reach", "PTX_SAMPLER_HOIST_EAGER": "1"},   # (exact/fast: admit against 0.85 x the
    "fast": {"PTX_SAMPLER_FUSE": "1", "PTX_PRED_RELEASE": "1", "PTX_SAMPLER_ADMIT": "memory", "PTX_SAMPLER_RELEASE": "reach", "PTX_SAMPLER_HOIST_EAGER": "1", "PTX_LN_TIER": "fast"},    # device — the speed rows spend free memory); PTX_LN_TIER: lever ln_core's tier word (exact binds none: the library op by name)
    "big": {"PTX_SAMPLER_FUSE": "1", "PTX_PRED_RELEASE": "1", "PTX_SAMPLER_ADMIT": "item", "PTX_SAMPLER_RELEASE": "item", "PTX_SAMPLER_HOIST_EAGER": "1", "PTX_LN_TIER": "big"},     # big: admit against the item's own
                                                                                                                                            # peak (never raised for speed)
}
LAZY_INIT_VALUES = ("0", "1")                                               # any other PTX_LAZY_INIT value is refused (ValueError)
BOOKKEEPING = ("PWD", "OLDPWD", "SHLVL", "_", "PYTHONPATH")                 # shell bookkeeping; PYTHONPATH goes to sys.path, not os.environ


@dataclass
class Resolution:
    mode: str
    exports: Dict[str, str]                                  # variables env.sh set or changed (+ the extras)
    unsets: List[str]                                        # variables env.sh removed (its `unset` line: PTX_BLK_ATT FPF_OPS FPF_TRIMUL_CONTRACT; FPF_OPS returns in both arms)
    pythonpath: List[str] = field(default_factory=list)      # the PYTHONPATH env.sh leaves, in its order: its own entries and the caller's where env.sh places them
    pythonpath_caller: List[str] = field(default_factory=list)  # the caller's own PYTHONPATH entries (a subset of `pythonpath`)
    extras: Dict[str, str] = field(default_factory=dict)     # the subset of exports set here, after env.sh (README row "post" + lazy init)
    pre_exports: Dict[str, str] = field(default_factory=dict)  # README row "pre" exports set here before env.sh (the T* rows' PTX_T_ATT tier word)
    kernel_key: Optional[str] = None                         # "<cc>|<triton major.minor>" when both are known
    row_key: str = OTHER_KEY                                 # the README row that applied (a key of README_ROWS, or "other")
    row: Dict[str, Dict[str, str]] = field(default_factory=lambda: {"pre": {}, "post": {}})
    kit_spec: str = ""                                       # env.sh's KIT_SPEC line (stdout of the sourced script)
    notes: List[str] = field(default_factory=list)
    base: Optional[str] = None                               # big: the arm composed on (BIG_BASE); None for the other modes

    def final(self, environ: Mapping[str, str]) -> Dict[str, str]:
        out = {k: v for k, v in environ.items() if k not in self.unsets}
        out.update(self.exports)
        return out


# ------------------------------------------------------------------------------------------------------------- probes ----
def jit_cache_key(version: Optional[str] = None, cuda: Optional[str] = None, cc: Optional[str] = None) -> str:
    """The JIT cache key, `torch<version>-cu<cuda>-sm<cc>` (opt_core.jit_cache.key): the torch version without its local tag, the CUDA
    version without the dot, the device's compute capability digits (e.g. torch2.13.0-cu130-sm90), resolved without importing torch:
    `version` defaults to the installed torch's metadata version, then torch/version.py; `cuda` to the version's local tag when it reads
    `cu<digits>` (the pinned wheels), then torch/version.py's `cuda`; `cc` to nvidia-smi compute_cap (the probe env.sh itself makes,
    `nvidia_smi_compute_cap`). A part that cannot be established raises opt_core.jit_cache.StackKeyUnknown naming it — never a silent
    `unknown` in a cache path; configs/h100.env exports the key as MODEL_OPT_STACK_KEY and names the word `unknown` itself when
    this raises (a cold, unshared cache directory, printed)."""
    return _jit_cache.key(version, cuda, cc if cc is not None else nvidia_smi_compute_cap())   # cc: the nvidia-smi probe env.sh itself makes (compute_cap alone); None -> the core's probe, then StackKeyUnknown


def triton_version() -> Optional[str]:
    try:
        return importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        return None


def nvidia_smi_compute_cap(path_env: Optional[str] = None) -> Optional[str]:
    """First GPU's compute capability from `nvidia-smi --query-gpu=compute_cap` (the probe env.sh itself makes; no CUDA context)."""
    exe = shutil.which("nvidia-smi", path=path_env)
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
    except Exception:
        return None
    line = (r.stdout or "").strip().splitlines()
    return line[0].strip() if r.returncode == 0 and line else None


def nvidia_smi_memory_mib(path_env: Optional[str] = None) -> Optional[int]:
    """First GPU's total memory in MiB from `nvidia-smi --query-gpu=memory.total` (no CUDA context); None when absent or unreadable."""
    exe = shutil.which("nvidia-smi", path=path_env)
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "--query-gpu=memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        line = (r.stdout or "").strip().splitlines()
        return int(line[0].strip()) if r.returncode == 0 and line else None
    except Exception:
        return None


def kernel_key(compute_cap: Optional[str], triton: Optional[str]) -> Optional[str]:
    if not compute_cap or not triton:
        return None
    return f"{compute_cap}|{'.'.join(triton.split('.')[:2])}"


# -------------------------------------------------------------------------------------------------------- env.sh source ----
_SCRIPT = 'env -0 > "$PROTENIX_OPT_OUTB"; ARM=$PROTENIX_OPT_ARM source "$PROTENIX_OPT_ENVSH" > "$PROTENIX_OPT_KITSPEC" 2>&1; env -0 > "$PROTENIX_OPT_OUTA"'
_OWN = ("PROTENIX_OPT_OUTB", "PROTENIX_OPT_OUTA", "PROTENIX_OPT_ARM", "PROTENIX_OPT_ENVSH", "PROTENIX_OPT_KITSPEC")


def source_env_sh(fpf_home: str, arm: str, environ: Mapping[str, str]) -> Tuple[Dict[str, str], Dict[str, str], str]:
    """Run `ARM=<arm> source <fpf_home>/env.sh` in bash on top of `environ`; return (before, after, kit_spec) environments."""
    envsh = os.path.join(fpf_home, "env.sh")
    if not os.path.isfile(envsh):
        raise FileNotFoundError(envsh)
    bash = shutil.which("bash", path=environ.get("PATH"))
    if not bash:
        raise RuntimeError("bash not found on PATH (env.sh must be sourced by bash)")
    with tempfile.TemporaryDirectory(prefix="protenix_opt_envsh_") as d:
        env = {k: v for k, v in environ.items() if k not in _OWN}
        env["FPF_HOME"] = fpf_home
        env.update({"PROTENIX_OPT_OUTB": os.path.join(d, "b"), "PROTENIX_OPT_OUTA": os.path.join(d, "a"), "PROTENIX_OPT_ARM": arm,
                    "PROTENIX_OPT_ENVSH": envsh, "PROTENIX_OPT_KITSPEC": os.path.join(d, "kit_spec")})
        r = subprocess.run([bash, "-c", _SCRIPT], env=env, capture_output=True, text=True)
        spec = open(env["PROTENIX_OPT_KITSPEC"], errors="replace").read() if os.path.exists(env["PROTENIX_OPT_KITSPEC"]) else ""
        if r.returncode != 0 or not os.path.exists(env["PROTENIX_OPT_OUTA"]):
            raise RuntimeError(f"sourcing {envsh} (ARM={arm}) failed rc={r.returncode}: {(r.stderr or spec)[-800:]}")
        parse = lambda f: dict(x.split("=", 1) for x in open(f, errors="replace").read().split("\0") if "=" in x)
        before, after = parse(env["PROTENIX_OPT_OUTB"]), parse(env["PROTENIX_OPT_OUTA"])
    for k in _OWN:
        before.pop(k, None); after.pop(k, None)
    return before, after, spec.strip()


def resolve(mode: str, environ: Optional[Mapping[str, str]], fpf_home: str, *, compute_cap: Optional[str] = None,
            triton: Optional[str] = None, memory_mib: Optional[int] = None, probe_gpu: bool = True, skip_pre=(), pre_override=None) -> Resolution:
    """The environment a mode materialises on top of `environ`: the kit README row for this box's kernel key (pre exports) and the
    package's own pre exports (PACKAGE_PRE: the sampler-graph token cap, sized to the device's memory by sampler_graph_maxtok()), env.sh's own delta (sourced: its switch families, its compute-capability branch, its
    prebuilt fast-LN selection for the installed torch), the row's post exports, the package's own per-mode post exports (PACKAGE_POST:
    the diffusion transformer's fused kernels) and lazy init — in the README's order of env operations.

    `compute_cap`/`triton`/`memory_mib` override the device probes (tests). `pre_override`: pre words to export in their place ({switch: word}: the replaced lever's word an ablation restores). `skip_pre`: README-row / package pre exports NOT to set (the
    ablation of a lever whose switch env.sh reads: env.sh is sourced without it, so its own default word stands). "off" resolves to nothing."""
    environ = dict(os.environ if environ is None else environ)
    res = Resolution(mode=mode, exports={}, unsets=[])
    if mode == "off":
        return res
    base = BIG_BASE if mode == "big" else None             # big: the arm and the README row are the base's
    arm = ARM[base] if base else ARM[mode]
    res.base = base
    cc = compute_cap if compute_cap is not None else (nvidia_smi_compute_cap(environ.get("PATH")) if probe_gpu else None)
    mem = memory_mib if memory_mib is not None else (nvidia_smi_memory_mib(environ.get("PATH")) if probe_gpu else None)
    res.kernel_key = kernel_key(cc, triton if triton is not None else triton_version())
    res.row_key = row_key(res.kernel_key)
    res.row = readme_row(res.kernel_key, mode, base)
    if skip_pre or pre_override:                              # an ablation of a lever env.sh selects by word (ablation.pre_switches / restored_words): the row as this
        res.row = {"pre": {**{k: v for k, v in res.row["pre"].items() if k not in set(skip_pre)}, **dict(pre_override or {})},   # process sources it — the ablated word gone,
                   "post": dict(res.row["post"])}                                                                              # the replaced lever's word restored in its place
    pre = environ.copy()
    pkg = {k: (sampler_graph_maxtok(mem) if k == "PTX_SAMPLER_GRAPH_MAXTOK" else v) for k, v in PACKAGE_PRE.get(base or mode, {}).items()}   # the sampler-graph cap sized to this device's memory
    for k, v in {**pkg, **res.row["pre"]}.items():   # README 'Order of env operations': the row's pre exports before env.sh (env.sh reads them), then the package's own pre exports (PACKAGE_PRE); a caller value wins
        if k not in environ and (k not in set(skip_pre) or k in (pre_override or {})):   # a caller value wins; a skipped word stays out unless the ablation restored another word for it
            pre[k] = v; res.pre_exports[k] = v
    before, after, res.kit_spec = source_env_sh(fpf_home, arm, pre)
    for k, v in after.items():
        if k in BOOKKEEPING:
            continue
        if before.get(k) != v:
            res.exports[k] = v
    res.unsets = [k for k in before if k not in after and k not in BOOKKEEPING]
    res.pythonpath_caller = [p for p in environ.get("PYTHONPATH", "").split(":") if p]
    res.pythonpath = []
    for p in after.get("PYTHONPATH", "").split(":"):                                # env.sh's order, the caller's entries where it puts them
        if p and p not in res.pythonpath:
            res.pythonpath.append(p)
    for k, v in res.pre_exports.items():
        res.exports[k] = after.get(k, v)                                            # set here, before env.sh: part of the delta
    # after env.sh (one place): the README row's post exports and the package's own per-mode post exports (a caller value wins), and lazy init for both modes
    for k, v in {**res.row["post"], **PACKAGE_POST.get(mode, PACKAGE_POST.get(base or mode, {}))}.items():   # a row of the mode itself first (big's words), else its base's
        if k not in environ:
            res.extras[k] = v
    if res.row_key == OTHER_KEY:
        res.notes.append(f"kernel key {res.kernel_key or 'unknown'}: no kit README row (the compositions table's row 'other': ARM {arm} defaults, no extra switch)")
    lz = environ.get("PTX_LAZY_INIT")
    if lz is not None and lz not in LAZY_INIT_VALUES:
        raise ValueError(f"PTX_LAZY_INIT={lz!r} is not one of {'|'.join(LAZY_INIT_VALUES)}")
    res.extras["PTX_LAZY_INIT"] = lz if lz is not None else LAZY_INIT_DEFAULT
    for k, v in res.extras.items():
        if environ.get(k) != v:
            res.exports[k] = v
    return res
