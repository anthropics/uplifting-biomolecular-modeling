"""The `big` memory mode of protenix_v1: ONE composition — the kit's `fast` base plus the memory levers below — over the memory-lever
library of the core (opt_core.mem: registry / record / compose / chunk / ckpt / allocator / offload). Guarantee: folds bigger inputs on the
same card, fast-class numerics (tier 2). `--n_gpu P` is the mode's resource axis (ngpu.py); this module is the P=1 lever set.

Composition (LINE, in order — ONE lever set: no lever is switchable by the caller; the two documented size statements below are the only
`PROTENIX_V1_BIG_*` names read, any other is refused by name at interpreter start (_autoload) and at composition (compose); a
precondition that does not hold REFUSES BY NAME, exit 3, never a silent no-op):

  lever                  where it lives          class      what it does
  expandable_segments    opt_core.mem.allocator  bitwise    PYTORCH_CUDA_ALLOC_CONF expandable_segments:True, confirmed by read-back at every unit boundary; engages
                                                            at every size (no CUDA-graph pool exists under big: the sampler graphs `sg` are off)
  cache_release          opt_core.mem.allocator  bitwise    torch.cuda.empty_cache() once per unit, at its end — policy per_unit; placement only (one release per unit,
                                                            not at the recycle seams or the confidence head's stage boundary: extra releases cost wall-clock and raise
                                                            the device peak under the expandable-segments allocator). MEMORY-GATED
  chunk_pair             this module (setting)   bitwise    the upstream's own chunked pair path (modes.MEMORY_PRESET: infer_setting.chunk_size=128, dynamic_chunk_size=
                                                            False) on the runner's configs — the report's `memory` row. MEMORY-GATED: at or below the gate the configs are
                                                            untouched (the upstream's own dynamic chunk ladder, stock's setting: no chunking up to 1024 tokens)
  drop_bond_mask         this module (setting)   bitwise    the dead int64 [N,N] `bond_mask` feature dropped before to_device (the upstream builds and ships it, no
                                                            module reads it: 8 B/token²)
  relp_lean              this module (chunk)     bitwise    RelativePositionEncoding.generate_relp in row blocks: the same fp32 0/1 relative-position features written
                                                            block by block into the [N,N,139] plane — the int64 [N,N,·] comparison / one-hot intermediates exist per block
  recycle_carry          opt_core.mem.ckpt       bitwise    z_init (fp32 [N,N,128], read once per recycle) parked on pinned host between its uses (RecycleCarry:
                                                            hold / get / seam; the seam no longer releases the caching allocator — flush off); s_init stays on the device. MEMORY-GATED
  diffusion_cond_chunk   this module (chunk)     measured   DiffusionConditioning.prepare_cache (cat[z_trunk, relpe] → LayerNorm(261) → Linear(261→128) → 2 transitions,
                                                            once per item) in row blocks (opt_core.mem.chunk.chunk_rows, per-call label band = the fp32 site): the fp32
                                                            [N,N,261] concat and its LayerNorm copy exist for one block. MEMORY-GATED
  conf_head_chunk        this module (chunk)     measured   ConfidenceHead.memory_efficient_forward with the PAE/PDE tail (LayerNorm + Linear) in row blocks
                                                            (opt_core.mem.chunk.chunk_rows); the logits on the CPU above 2000 tokens as the stock does. MEMORY-GATED
  msa_zfree              this module (release)   bitwise    each MSA block's pair INPUT released (storage resize to 0: free_storage) once the
                                                            outer-product-mean residual has produced the block's new z — before the block's MSA stack and pair stack
                                                            run; the tensor is read by no later statement (MSAModule.forward / checkpoint_blocks / get_pairformer_output
                                                            rebind z) yet stays referenced by those frames, so only an explicit release returns it to the allocator.
                                                            MSABlock.forward wrapped (the stock body called, not re-stated) with per-call pre-hooks on its msa_stack / pair_stack
  diffcache_free         this module (release)   bitwise    the diffusion roll-out's shared caches (DiffusionConditioning.prepare_cache's pair_z [N,N,c_z] fp32 and
                                                            AtomAttentionEncoder.prepare_cache's p_lm / c_l; protenix.py _main_inference_loop `cache`) released at the
                                                            confidence head's entry: sampling has ended, the confidence head reads z_trunk, nothing reads the caches again
                                                            (the pinned protenix.py). The two prepare_cache and Protenix.run_confidence_head wrapped at class level
  trimul_torch           this module (setting)   measured   configs.triangle_multiplicative='torch' (the upstream's in-place row-chunked TriangleMultiplication) —
                                                            SIZE-GATED: engaged only above TRIMUL_GATE_TOKENS
The two release levers are size-blind (free: a storage release costs no wall-clock and changes no arithmetic) and, like the four levers the
row-sharded path supersedes, not applied under `--n_gpu P>1` (ROWPAIR_SITELESS, named replaced_by_rowpair by gates_for: tp.py owns the MSA
module, prepare_cache and the confidence head there; their LEVER rows read `state=off reason=property_gate gate=n_gpu<=1`).
MEMORY-GATED = engaged when the run's largest item has N_token > MEMORY_GATE_TOKENS (0: every sized input); an unsized input (no token
estimate) holds them by name and PROTENIX_V1_BIG_SIZE_N_TOKEN states the size yourself; the five arm together. A gated lever's LEVER line
below its gate reads
`state=skipped reason=below_gate … gate=n_token<=<threshold> n=<sized N>`; the BIG line names it under property_off.

Base levers off under big at every size (compose drops BASE_LEVERS_OFF from the `fast` arm; recorded on the BIG line as property_off and
in the activation report under big; `sampler_prep` rides `sg`; `dit_fused` + `dit_lowp` hold packed weight copies through the item):
  keep_pool  the fast base's allocator policy (the stock in-forward empty_cache sites skipped): the cached pool it keeps through the
         confidence head — and, above 2000 tokens, across the per-sample pae/pde offload releases — is reserved memory the line returns
  hoist  the fast base's DiT pair-bias hoist: static buffers (∝ N_token² × N_sample) held through the item and carried into the confidence
         head (the peak stage of a prediction at the sizes fast serves)
  sg     the sampler CUDA graphs: a captured graph's private pool is held to process exit and carried into the confidence head the same way
         with no graph pool the expandable-segments allocator engages
The size gates (two thresholds — the memory gate MEMORY_GATE_TOKENS and the TriMul gate TRIMUL_GATE_TOKENS — sized once per process on the
run's largest item: `estimate_tokens` on the command line's `--input`, or `PROTENIX_V1_BIG_SIZE_N_TOKEN=<n>` stated by the caller;
recorded the same way):
An item whose featurized N_token falls on the other side of a gate than the estimate decided keeps the composition the process was sized
for and says so: `NOTE size gate re-decided at featurization: unit <u> N_token=<n> …: <lever> on|off …` on the transcript and a
`size_gate_crossings` record on the census (report.big.gate); the run proceeds — the estimate counts polymer residues; ligand / ion entries
are named as unsized.

Seams (the ONE attach point per stage; stack.py calls them, nothing else does):
  compose(environ, argv)      the sized composition → modes.resolve("big") builds the run's Resolution from it (arm, levers, memory preset)
  arm(res)                    activation, before the stock runner exists: opt_core.mem.apply over LINE (flags, settings, the allocator lever's
                              export while CUDA is uninitialised, recycle_carry's host park); a refusal is an ActivationError by name
  apply(runner, res)          after the stock runner is built (the runner __init__ wrap): chunk_pair / trimul_torch on runner.configs, the
                              patched stock methods installed (relp_lean, recycle_carry's trunk, diffusion_cond_chunk, conf_head_chunk — each
                              stock file pinned by sha256, stock/PINS.json model_files), the per-item census hooks on the instance's predict;
                              returns the `memory` record {applied, replaced} the MEMORY line prints; logs the BIG line. The stock
                              attention package (cuequivariance_torch) is imported here, ahead of the trunk (preload_stock_attention: the
                              upstream's lazy import inside the first item's pair stack takes far longer there than at process start)
  before_predict / after_predict   one census unit per `InferenceRunner.predict` item: drop_bond_mask acts, the settings are re-read on the
                              item (mark or named fallback), the size gates are checked against the featurized N_token, cache_release at unit
                              end (when it armed: opt_core.mem.allocator.release is a no-op for a lever not applied)
  exit_join(partial, reason, evidence, allow_partial, run_ok)   the exit rule's big part (report.verdict): base levers gated for the whole
                              run at this size are disengaged by property; the census gate (opt_core.mem.record.exit_gate) joins the partial state
  lever_rows(mode) / exit_lines()   the LEVER lines of the memory levers and the census EXIT line (report.log_exit_lines)
"""
from __future__ import annotations

import functools
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

from opt_core import gates as CORE_GATES
from opt_core import report as CORE_R
from opt_core.oom import is_oom                                  # the one out-of-memory classifier: the teardown handler re-raises an OOM
from opt_core.mem import allocator as ALLOC
from opt_core.mem import apply as mem_apply, Refused
from opt_core.mem import chunk as CHUNK
from opt_core.mem import ckpt as CKPT
from opt_core.mem import compose as COMPOSE
from opt_core.mem.registry import Applied, Ctx, LEVERS as REGISTRY, RefusalError, refuse, register
from opt_core import modes as CORE_MODES

from . import modes as M
from .report import PREFIX, TAG

MODE = "big"
FLAG_PREFIX = "PROTENIX_V1"                                # the kit's variable stem (recorded in the core's block); PROTENIX_V1_BIG_* below
POLICY_LEVERS = ("size",)                                    # the line's policy lever: registered so its settings (n_token, memory_gate_tokens) are recorded like every other setting; it applies nothing
N_TOKEN_ENV = "PROTENIX_V1_BIG_SIZE_N_TOKEN"              # settings["size"]["n_token"]: the caller's own statement of the run's largest token count (sizes the gates; the estimate otherwise)
BASE_MODE = "fast"                                          # the mode big composes on (opt_core.mem.compose base rule fast-else-exact; this kit ships fast)

# The line, in application order: the allocator levers first (the export precedes CUDA), then the settings, then the patched statements. Two size gates decide which levers arm at this run's size
# (gates_for, sized once per process): the memory gate (MEMORY_GATED_LEVERS above MEMORY_GATE_TOKENS) and the TriMul gate (trimul_torch
# above TRIMUL_GATE_TOKENS); SIZE_GATED_LEVERS = every gated lever, in LINE order.
LINE: Tuple[str, ...] = ("expandable_segments", "cache_release", "chunk_pair", "drop_bond_mask", "relp_lean", "recycle_carry",
                         "diffusion_cond_chunk", "conf_head_chunk", "msa_zfree", "diffcache_free", "trimul_torch")
ROWPAIR_SITELESS: Tuple[str, ...] = ("msa_zfree", "diffcache_free")   # the release levers' sites (MSA blocks; prepare_cache + confidence head) belong to the row-sharded statements under `--n_gpu P>1` (tp.py): left out of every rank's composition, named replaced_by_rowpair (gates_for) — ngpu.SUPERSEDED_LEVERS and the ×P line untouched
MEMORY_GATE_TOKENS = 0                                       # settings["size"]["memory_gate_tokens"]: 0 = the five memory-cost levers below engage at every sized input; an unsized input (no token estimate) still holds them by name; PROTENIX_V1_BIG_SIZE_N_TOKEN states the size yourself
MEMORY_GATED_LEVERS: Tuple[str, ...] = ("cache_release", "chunk_pair", "recycle_carry", "diffusion_cond_chunk", "conf_head_chunk")   # gated TOGETHER: below the gate each costs wall-clock and frees nothing the card feels
SIZE_GATED_LEVERS: Tuple[str, ...] = MEMORY_GATED_LEVERS + ("trimul_torch",)
KIT_LEVERS: Tuple[str, ...] = ("chunk_pair", "trimul_torch", "drop_bond_mask", "relp_lean", "diffusion_cond_chunk", "conf_head_chunk", "msa_zfree", "diffcache_free")   # registered here
CORE_LEVERS: Tuple[str, ...] = ("expandable_segments", "cache_release", "recycle_carry")                                          # registered by opt_core.mem

BASE_LEVERS_OFF: Tuple[str, ...] = ("hoist", "sg", "keep_pool", "sampler_prep", "dit_fused", "dit_lowp")                      # the fast base's levers the mode drops from its arm at every size: the DiT pair-bias hoist, the sampler CUDA graphs (sampler_prep rides them) and keep_pool hold buffers / pools through the item into the confidence head; dit_fused + dit_lowp: below
# `dit_fused` + `dit_lowp` (the fused 24-block DiT stack and its fp16 word): OFF the memory mode's arm — the stack's packed weight copies raise the
# allocated peak at every size; `cond_dedupe` and `atom_fused` stay (no resident cost). `tricuda` (the triangle-attention core riding gflash: the
# provider's row for this card, by name) stays ON: its peak allocated and reserved match the flash core's in this mode.
TRIMUL_GATE_ENV, TRIMUL_GATE_TOKENS = "PROTENIX_V1_BIG_TRIMUL_TORCH_TOKENS", 2048  # settings["trimul_torch"]["tokens"]: the one size gate
GATE_SETTINGS = {TRIMUL_GATE_ENV: ("trimul_torch", "tokens", 0), N_TOKEN_ENV: ("size", "n_token", 1)}   # the ONLY PROTENIX_V1_BIG_* names read: name -> (lever, setting, lowest value); read here (line_settings), passed to the core as ctx.settings (== _autoload.BIG_DECLARED, test_merge_locks)
BIG_PREFIX = FLAG_PREFIX + "_BIG_"                       # every other name under it is refused by name (refuse_undeclared)
POLYMER_KEYS = ("proteinChain", "dnaSequence", "rnaSequence")

# Settings of the line (ctx.settings, recorded per lever): constants, except the two size statements line_settings reads from the caller.
SETTINGS: Dict[str, Dict[str, object]] = {
    "cache_release": {"policy": "per_unit"},                     # one torch.cuda.empty_cache() per unit at its end; releases at the per-stage seams (recycle seams + the confidence head's stage boundary) cost wall-clock and raise the device peak under the expandable-segments allocator, so the policy is per_unit
    "chunk_pair": {"chunk": int(M.MEMORY_PRESET["infer_setting.chunk_size"])},
    "relp_lean": {"rows": 512},
    "recycle_carry": {"park": "host", "pin_max_gb": 24.0, "flush": False},   # flush off: the recycle seam does not release the caching allocator (as cache_release's per_unit policy: a seam release is slower with no lower peak)
    "diffusion_cond_chunk": {"rows": 256},
    "conf_head_chunk": {"rows": 256},
    "msa_zfree": {},
    "diffcache_free": {},
    "trimul_torch": {"tokens": TRIMUL_GATE_TOKENS},
    "size": {"n_token": None, "memory_gate_tokens": MEMORY_GATE_TOKENS},
}

# The applicability table of the memory levers (CHANGES.md lists them; report.lever_line reads strategy / impl / origin from it).
TABLE: Dict[str, Dict[str, str]] = {
    "expandable_segments": {"adopted": "yes", "class": "bitwise", "strategy": "F7.expandable_segments", "impl": "opt_core.mem.allocator", "origin": "core",
                            "evidence": "lean rows 1000-3025 (2AH1 948: 10.36 vs stock 13.02 GB; 3P8C 3025: 33.41 vs 71.04 GB); engages once no graph pool remains"},
    "cache_release": {"adopted": "yes (per unit)", "class": "bitwise", "strategy": "F7.cache_release", "impl": "opt_core.mem.allocator", "origin": "core",
                      "evidence": "per-stage seams vs one release per unit at 1200 / 1340 / 2000 tokens (kit notes B5): the seams cost wall-clock and RAISE the device peak under expandable segments — per_unit kept, per_stage dropped; per_unit unmeasured as a standalone row"},
    "chunk_pair": {"adopted": "yes", "class": "bitwise", "strategy": "F7.chunked_eval", "impl": "infer_setting (modes.MEMORY_PRESET)", "origin": "kit",
                   "evidence": "every big run; equal to stock bit for bit under the recipe (4O9A 1592 / 8G5T 2048 real-MSA, s356)"},
    "drop_bond_mask": {"adopted": "yes", "class": "bitwise", "strategy": "LOCAL.protenix_v1.drop_bond_mask", "impl": "big.py (input_feature_dict)", "origin": "kit",
                       "evidence": "composition runs 1000-3025; bit-equal by construction (no module reads bond_mask)"},
    "relp_lean": {"adopted": "yes", "class": "bitwise", "strategy": "F7.chunked_eval", "impl": "big.py (RelativePositionEncoding.generate_relp)", "origin": "kit",
                  "evidence": "composition runs 1000-3025; bit-equal (integer one-hot: exactly one product per term)"},
    "recycle_carry": {"adopted": "yes", "class": "bitwise", "strategy": "F7.recycle_carry", "impl": "opt_core.mem.ckpt (RecycleCarry) + big.py (Protenix.get_pairformer_output)", "origin": "core",
                      "evidence": "7EBY 4944: 84.94 GB completes (stock / exact / fast OOM at 4076+); the lever that moves the ceiling 4076 -> 4944"},
    "diffusion_cond_chunk": {"adopted": "yes", "class": "measured", "strategy": "F7.chunked_eval", "impl": "big.py (DiffusionConditioning.forward) over opt_core.mem.chunk.chunk_rows", "origin": "kit",
                             "evidence": "composition rows 2000-4944 (8G5T 2048: 19.41 vs stock 33.99 GB; words in band)"},
    "conf_head_chunk": {"adopted": "yes", "class": "measured", "strategy": "F7.chunked_eval", "impl": "big.py (ConfidenceHead.memory_efficient_forward) over opt_core.mem.chunk.chunk_rows", "origin": "kit",
                        "evidence": "composition rows 2000-4944 (same rows as diffusion_cond_chunk)"},
    "msa_zfree": {"adopted": "yes", "class": "bitwise", "strategy": "F7.chunked_eval", "impl": "big.py (MSABlock.forward; storage resize to 0, big.free_storage)", "origin": "kit",
                  "evidence": "det-1 byte-identity vs the line without it at 1200 and 1340 tokens and the peak before/after at 1200 / 2000 tokens (the kit notes)"},
    "diffcache_free": {"adopted": "yes", "class": "bitwise", "strategy": "F7.chunked_eval", "impl": "big.py (Protenix.run_confidence_head; storage resize to 0, big.free_storage)", "origin": "kit",
                       "evidence": "det-1 byte-identity vs the line without it at 1200 and 1340 tokens and the peak before/after at 1200 / 2000 tokens (the kit notes)"},
    "trimul_torch": {"adopted": "yes (size-gated above 2048 tokens)", "class": "measured", "strategy": "F7.chunked_eval", "impl": "configs.triangle_multiplicative (the upstream's torch path)", "origin": "kit",
                     "evidence": "n/a above stock's ceiling (the reach above 4k: 7EBY 4944 with recycle_carry); gated off at <= 2048, where the fast base owns the site"},
}

_STATE: Dict[str, Any] = {"composition": None, "record": None, "ctx": None, "unit": None, "installed": {}, "undo": [], "gate": None,
                          "excused": {}, "units_seen": 0, "memory": None}


class BigError(RuntimeError):
    """A refusal of the mode by name (stack turns it into NOT ACTIVE, exit 3)."""


# ============================================================================================================== sizing and the gates
def refuse_undeclared(environ=None) -> None:
    """Any PROTENIX_V1_BIG_* name other than the two size statements (GATE_SETTINGS) is refused by name: the line's levers are one set,
    not switchable by the caller (the .pth hook refuses the same names at interpreter start: _autoload.install)."""
    environ = os.environ if environ is None else environ
    unknown = sorted(k for k in environ if k.startswith(BIG_PREFIX) and k not in GATE_SETTINGS)
    if unknown:
        raise BigError(f"undeclared {','.join(unknown)}: the line's levers are one set, not switchable; the size statements it reads are {', '.join(sorted(GATE_SETTINGS))}")


def line_settings(environ=None) -> Dict[str, Dict[str, object]]:
    """The line's settings as the core records them (ctx.settings): SETTINGS, with the caller's two size statements read HERE — the one
    reader of PROTENIX_V1_BIG_SIZE_N_TOKEN (a positive integer) and PROTENIX_V1_BIG_TRIMUL_TORCH_TOKENS (a non-negative integer);
    a malformed value is a BigError by name."""
    environ = os.environ if environ is None else environ
    settings = {k: dict(v) for k, v in SETTINGS.items()}
    for name, (lever, key, lowest) in GATE_SETTINGS.items():
        raw = environ.get(name)
        if raw is not None and str(raw).strip() != "":
            settings[lever][key] = _token_count(name, lowest)(raw)
    return settings


def _token_count(name: str, lowest: int):
    def cast(raw):
        try:
            v = int(str(raw).strip())
        except ValueError:
            raise BigError(f"{name}={raw!r}: an integer token count") from None
        if v < lowest:
            raise BigError(f"{name}={raw!r}: a {'positive' if lowest == 1 else 'non-negative'} token count")
        return v
    return cast


def gate_tokens(env: str, environ=None) -> int:
    """A gate threshold: the caller's statement (line_settings) or the line's default (SETTINGS, the one statement of it)."""
    lever, key, _ = GATE_SETTINGS[env]
    v = line_settings(environ)[lever][key]
    return int(SETTINGS[lever][key]) if v is None else int(v)


def memory_gate_tokens() -> int:
    """The memory gate's threshold: the line's one statement of it (SETTINGS["size"]["memory_gate_tokens"]; no caller name reads it — the size
    statement PROTENIX_V1_BIG_SIZE_N_TOKEN decides which side of it a run is on)."""
    return int(SETTINGS["size"]["memory_gate_tokens"])


def gates_for(n_token: Optional[int], environ=None) -> dict:
    """The lever-property gates at a token count (None = unsized: the below-gate composition, recorded as such): {drop: the base levers off under
    the mode (BASE_LEVERS_OFF, every size), levers_off: memory levers not applied at this size (in LINE order), size_gated_off: those of them a
    size gate leaves out, off_by_property: {name: reason}, gates: {lever: threshold} for every size-gated lever, memory_gate_tokens,
    trimul_gate_tokens, sized, rowpair_world}."""
    tg = gate_tokens(TRIMUL_GATE_ENV, environ)
    mg = memory_gate_tokens()
    n = -1 if n_token is None else int(n_token)
    nword = n if n >= 0 else "unsized"
    drop: Tuple[str, ...] = BASE_LEVERS_OFF
    off: Dict[str, str] = {
        "hoist": "off under big at every size: the DiT pair-bias hoist's static buffers (N_token² × N_sample) are held through the item into the confidence head, the peak stage",
        "sg": "off under big at every size: a captured sampler graph's private pool is held to process exit and carried into the confidence head; with no graph pool the expandable-segments allocator engages",
        "keep_pool": "off under big at every size: the line keeps the stock in-forward cache releases (the confidence head's empty_cache sites return the sampler's blocks before the head; above 2000 tokens the per-sample releases of the pae/pde offload path) — the reserved pool the lever keeps is the memory the line exists to give back",
    }
    gated: set = set()
    if n <= mg:                                                        # the memory gate: at or below it the upstream's own path holds nothing these levers free (no chunking up to 1024 tokens in its
        gated |= set(MEMORY_GATED_LEVERS)                              # own ladder) and each costs wall-clock (host park + seam flushes, row-blocked conditioning / confidence, allocator flush per item)
        for lv in MEMORY_GATED_LEVERS:
            off[lv] = (f"not applied at N_token {nword} <= {mg} (the memory gate): "
                       + ("the upstream's own dynamic chunk ladder is stock's setting at this size" if lv == "chunk_pair" else
                          "the per-unit cache release — gated with the memory levers (measured together)" if lv == "cache_release" else
                          "nothing to free below the upstream's own chunking threshold; the lever costs wall-clock"))
    if n <= tg:
        gated.add("trimul_torch"); off["trimul_torch"] = f"not applied at N_token {nword} <= {tg} ({TRIMUL_GATE_ENV}): the fast TriMul cell owns the site at or below the gate"
    else:
        off["fast"] = f"disengaged by property at N_token {n} > {tg}: trimul_torch takes the TriMul site (the upstream's torch path; the kit records stock:path)"
        off["gflash"] = f"disengaged by property at N_token {n} > {tg}: the fused triangle attention is gated by the kit above {tg} pair rows under the chunked path (stock:chunk)"
    P = rowpair_world(environ)
    if P > 1:                                                          # the row-sharded line (ngpu.py / rowpair.py / tp.py): the base arm's kernel and graph levers own NO
        for lv, why in ROWPAIR_REPLACES.items():                       # site under the sharded statements — replaced by rowpair BY NAME (excused from `served 0`, never partial)
            off[lv] = f"replaced_by_rowpair at n_gpu {P}: {why}"
        for lv in ROWPAIR_SITELESS:                                    # the release levers' sites are the sharded statements' own (tp.py): not applied in any rank, by name
            gated.add(lv); off[lv] = f"replaced_by_rowpair at n_gpu {P}: the row-sharded statements own the site (tp.py: the MSA module, prepare_cache, the confidence head)"
    levers_off: Tuple[str, ...] = tuple(lv for lv in LINE if lv in gated)               # the line's levers NOT applied at this size / world (compose leaves them out)
    thresholds = {lv: (tg if lv == "trimul_torch" else mg) for lv in SIZE_GATED_LEVERS}
    return {"drop": drop, "levers_off": levers_off, "size_gated_off": tuple(lv for lv in levers_off if lv in SIZE_GATED_LEVERS), "off_by_property": off,
            "gates": thresholds, "memory_gate_tokens": mg, "trimul_gate_tokens": tg, "sized": n_token is not None, "rowpair_world": P}


ROWPAIR_REPLACES = {                                                   # base-arm levers whose sites the row-sharded statements replace under `--n_gpu P>1` (report.verdict reads
    "fast": "the triangle multiplicative update runs as the core's row-sharded statement over the module's own projections (opt_core.mem.rowpair.trimul); "
            "the fused TriMul cell needs the whole pair tensor on one device",                              # off_by_property: `<lever> served 0 calls` is then excused by name)
    "gflash": "the triangle attention runs per rank on its own query rows through the module's attention with the run's --triatt_kernel "
              "(opt_core.mem.rowpair.triatt); the module-level fused site is never entered",
    "ttr": "the pair transitions run as row-local statements (opt_core.mem.rowpair.transition); the module-level fused transition site is never entered",
    "summary_hostidx": "the summary confidences are assembled on rank 0 from the sharded confidence head's reducers (tp.compute_full_data_and_summary_tp); "
                       "the stock per-chain statement the lever restates is never entered (the lever steps aside by name at install)",
    "ditattn": "the DiffusionTransformer runs as the row-sharded statement (tp.DiTBlockFns: local query rows against all rows, the bias rows of the rank); "
               "the sample-batched self-attention call the fused kernel serves is never made",
    "ditattnfp16": "rides ditattn, which the row-sharded DiffusionTransformer statement replaces",
    "atomattn": "the atom transformer runs under the row-sharded sampler (tp) with the rank's p_lm band; the module-level local-attention call the kernel serves is the single-device statement's",
    "dit_fused": "the DiffusionTransformer runs as the row-sharded statement (tp.DiTBlockFns over the blocks' own modules); the single-device fused token stack "
                 "(diffusion_transformer.forward) is never entered (tp.TP_DROPPED)",
    "dit_lowp": "rides dit_fused, which the row-sharded DiffusionTransformer statement replaces",
    "cond_dedupe": "the diffusion conditioning runs under the row-sharded sampler (tp): the dedupe wrapper serves only where that statement enters DiffusionConditioning.forward "
                   "with the sample-expanded noise level; a rank statement that computes the conditioning itself never enters it",
    "atom_fused": "the atom transformers run under the row-sharded sampler (tp) with the rank's p_lm band: the fused atom stacks serve only where that statement enters "
                  "atom_transformer.forward; a rank statement that runs the blocks itself never enters them",
    "pfattn": "the pairformer single attention runs per rank on its local query rows over the module's own projections (tp.apb_fn); the module-level standard_multihead_attention the fused kernels serve is never entered",
    "opm_fused": "the MSA module runs as the row-sharded statement (tp.msa_module_tp: the outer product mean per row budget over the module's own projections); OuterProductMean.forward is never entered",
    "pwa_fused": "the MSA module runs as the row-sharded statement (tp.msa_module_tp: the pair-weighted averaging over the module's own projections on the rank's rows); MSAPairWeightedAveraging.forward is never entered",
}


def rowpair_world(environ=None) -> int:
    """P of this process: the rank environment the core launcher sets in every worker of a `--n_gpu P>1` run (opt_core.mem.rowpair.launch
    ROWPAIR_WORLD); 1 in the launching process and in every single-GPU run."""
    environ = os.environ if environ is None else environ
    try:
        return max(1, int(str(environ.get("ROWPAIR_WORLD", "1")).strip() or "1"))
    except ValueError:
        return 1


def estimate_tokens(input_json_path: str) -> Tuple[int, dict]:
    """The token count of a protenix input JSON from its polymer sequences (residues × count), per item and the largest; a non-polymer entry
    (ligand, ion) contributes no tokens and is NAMED under `unsized` (the featurizer's atom-level token count is not reproduced here — the
    per-item size check at predict names any gate it crosses)."""
    with open(input_json_path, encoding="utf-8") as fh:
        doc = json.load(fh)
    items = doc if isinstance(doc, list) else [doc]
    largest, per_item, unsized = 0, {}, {}
    for idx, it in enumerate(items):
        n, other = 0, []
        for seq in (it.get("sequences") or []):
            for key, val in seq.items():
                if key in POLYMER_KEYS and isinstance(val, dict):
                    n += len(str(val.get("sequence", ""))) * int(val.get("count", 1) or 1)
                else:
                    other.append(str(key))
        name = str(it.get("name", idx))
        per_item[name] = n
        if other:
            unsized[name] = other
        largest = max(largest, n)
    if not per_item:
        raise BigError(f"{input_json_path}: no input items to size")
    return largest, {"per_item": per_item, "unsized": unsized}


def input_from_argv(argv: Optional[List[str]] = None) -> Optional[str]:
    """The stock CLI's input argument (`--input <file|dir>`, `--input=<...>`) on the process's command line, or None. The runner carries
    `input_json_path` per item only after it is built; the composition is sized before it exists, from the command line."""
    argv = list(sys.argv if argv is None else argv)
    for i, a in enumerate(argv):
        if a == "--input" and i + 1 < len(argv):
            return argv[i + 1]
        if isinstance(a, str) and a.startswith("--input="):
            return a.split("=", 1)[1]
    return None


def input_files(path: str) -> List[str]:
    """The JSON files an input argument names: the file, or every *.json of a directory (the stock CLI's batch form), sorted; [] when none exists."""
    if os.path.isdir(path):
        return sorted(os.path.join(path, f) for f in os.listdir(path) if f.endswith(".json"))
    return [path] if (path.endswith(".json") and os.path.isfile(path)) else []


def size_of_run(environ=None, argv=None) -> dict:
    """{n_token: int|None, source: environment|estimate|unsized, input, per_item, unsized, reason}: the run's largest item — the caller's
    PROTENIX_V1_BIG_SIZE_N_TOKEN when set (a positive integer, else BigError), else the polymer estimate of the command line's --input, else unsized."""
    environ = os.environ if environ is None else environ
    n = line_settings(environ)["size"]["n_token"]
    if n is not None:
        return {"n_token": int(n), "source": "environment", "input": None, "per_item": None, "unsized": None, "reason": f"{N_TOKEN_ENV}={n}"}
    path = input_from_argv(argv)
    files = input_files(path) if path else []
    if not files:
        return {"n_token": None, "source": "unsized", "input": path, "per_item": None, "unsized": None,
                "reason": "no input JSON on the command line to size (the below-gate composition; every item is size-checked at predict)"}
    n, per_item, unsized = 0, {}, {}
    for f in files:
        nf, d = estimate_tokens(f)
        key = (lambda k: f"{os.path.basename(f)}:{k}") if len(files) > 1 else (lambda k: k)
        n = max(n, nf); per_item.update({key(k): v for k, v in d["per_item"].items()}); unsized.update({key(k): v for k, v in d["unsized"].items()})
    return {"n_token": n, "source": "estimate", "input": path, "per_item": per_item, "unsized": unsized or None,
            "reason": f"polymer residues of the largest item of {path}" + (f" (non-polymer entries unsized: {sorted(unsized)})" if unsized else "")}


def compose(environ=None, argv=None) -> dict:
    """The composition of this process (module contract; cached — one per process): {line: big, arm: the kit arm string, base_arm,
    levers: the memory levers to apply in order, drop, gates, size, graphs}. modes.resolve("big") builds the Resolution from it."""
    environ_given = environ is not None or argv is not None
    if not environ_given and _STATE["composition"] is not None:
        return _STATE["composition"]
    environ = os.environ if environ is None else environ
    refuse_undeclared(environ)
    size = size_of_run(environ, argv)
    gates = gates_for(size["n_token"], environ)
    base_arm = M.KIT_MODES[BASE_MODE].arm
    drop = gates["drop"]
    parts = base_arm.split("+")
    arm = "+".join(p for p in parts if p not in drop)
    levers = tuple(x for x in LINE if x not in gates["levers_off"])
    comp = {"line": MODE, "arm": arm, "base_arm": base_arm, "base": BASE_MODE, "levers": levers, "drop": tuple(drop),
            "gates": gates, "size": size}
    if not environ_given:
        _STATE["composition"] = comp
    return comp


def reset() -> None:
    """Forget the cached composition and every installed piece (tests; a process composes once)."""
    undo()
    _STATE.update(composition=None)


# ============================================================================================================== the kit-side levers
def _none(ctx: Ctx):
    return None


def _configs(ctx: Ctx, lever: str):
    runner = ctx.hook(lever, "runner")
    configs = getattr(runner, "configs", None)
    if configs is None:
        raise RefusalError(refuse(lever, "runner.configs", "the stock runner carries no `configs`"))
    return runner, configs


def _cfg_get(configs, dotted: str):
    node = configs
    for part in dotted.split("."):
        node = getattr(node, part, None) if not isinstance(node, dict) else node.get(part)
        if node is None:
            return None
    return node


def _cfg_set(configs, dotted: str, value, lever: str):
    """Set an EXISTING dotted field of the runner's configs in place; a field the configs do not carry refuses by name. Returns the value before."""
    head, _, leaf = dotted.rpartition(".")
    node = _cfg_get(configs, head) if head else configs
    if node is None or (_cfg_get(node, leaf) is None and not (hasattr(node, leaf) or (isinstance(node, dict) and leaf in node))):
        raise RefusalError(refuse(lever, f"configs.{dotted}", f"the runner's configs carry no `{dotted}` (a setting of this lever)"))
    before = _cfg_get(node, leaf)
    if isinstance(node, dict):
        node[leaf] = value
    else:
        setattr(node, leaf, value)
    return before


def _pins() -> dict:
    from . import stack
    return stack.pins()


STOCK_FILE_MODULES = ("protenix.model.modules.embedders", "protenix.model.modules.diffusion", "protenix.model.modules.confidence", "protenix.model.protenix")


def stock_file_sha256() -> Dict[str, str]:
    """The stock model files the patched statements are pinned to: stock/PINS.json "stock".model_files (the kit's ONE pin table, read through
    stack.pins()) — module import path -> sha256 of the installed file (the members of the carried protenix 1.1.0 wheel)."""
    table = dict((_pins().get("stock") or {}).get("model_files") or {})
    table.pop("rule", None)
    if not table:
        raise RefusalError(refuse(MODE, "PINS.model_files", "stock/PINS.json carries no stock.model_files: the big levers' stock pins"))
    return table


def _module_pinned(lever: str, modname: str):
    """Import a stock module and check its file against stock/PINS.json stock.model_files (sha256): a lever that replaces a method body
    copies stock statements, so the file it copies from must be the pinned one — a different file refuses the lever by name."""
    import importlib
    want = stock_file_sha256().get(modname)
    if not want:
        raise RefusalError(refuse(lever, "PINS.model_files", f"stock/PINS.json stock.model_files has no sha256 for {modname}"))
    mod = importlib.import_module(modname)
    got = CORE_GATES.sha256_file(mod.__file__)
    if got != want:
        raise RefusalError(refuse(lever, f"sha256.{modname}", f"{mod.__file__} sha256 {got[:12]}… != the pinned {want[:12]}… (stock/PINS.json stock.model_files): the lever copies statements of the pinned file only"))
    return mod


def _register_policy_levers() -> None:
    """Declare the line's policy lever (size): a registry entry whose only role is to record its setting (n_token, the caller's size
    statement or None) like every other setting; it applies nothing (an Applied with the setting recorded). Idempotent."""
    if all(REGISTRY.get(n) is not None for n in POLICY_LEVERS):
        return
    for name, keys, what in (("size", ("n_token", "memory_gate_tokens"), "the caller's statement of the run's largest token count (else the --input estimate) and the memory gate it is held against"),):
        def _apply(ctx, _name=name, _keys=keys) -> Applied:
            return Applied(lever=_name, settings={k: ctx.setting(_name, k, SETTINGS[_name][k]) for k in _keys}, sites=())
        register(name, family="setting", exact="bitwise", replace=True, strategy=f"LOCAL.protenix_v1.{name}",
                 exact_reason="a sizing policy of the line: selects which levers arm, changes no arithmetic", applies=_none,
                 description=what, preconditions=(), settings=tuple(keys))(_apply)


def _register_kit_levers() -> None:
    """Declare this engine's levers in the core's registry (idempotent: a name already registered by this module is kept)."""
    _register_policy_levers()
    if all(REGISTRY.get(n) is not None for n in KIT_LEVERS):
        return

    @register("chunk_pair", family="setting", exact="bitwise", replace=True, strategy=TABLE["chunk_pair"]["strategy"],
              exact_reason="the upstream's own chunked pair path at every size: a reduction chunked over rows, the same arithmetic (equal to stock bit for bit under the recipe: 4O9A 1592, 8G5T 2048, s356)",
              applies=_none, description="infer_setting.chunk_size=<chunk>, dynamic_chunk_size=False on the stock runner's configs (modes.MEMORY_PRESET)",
              preconditions=("runner.configs",), settings=("chunk",))
    def chunk_pair(ctx: Ctx) -> Applied:
        chunk = ctx.setting("chunk_pair", "chunk", int(M.MEMORY_PRESET["infer_setting.chunk_size"]), cast=int)
        if chunk < 1:
            raise RefusalError(refuse("chunk_pair", "chunk", f"chunk={chunk}: a positive row count"))
        preset = dict(M.MEMORY_PRESET, **{"infer_setting.chunk_size": chunk})
        return Applied(lever="chunk_pair", settings=preset, sites=("InferenceRunner.configs.infer_setting",))

    @register("trimul_torch", family="setting", exact="measured", replace=True, strategy=TABLE["trimul_torch"]["strategy"],
              exact_reason="the upstream's own TriangleMultiplication torch path (triangular.py:519-525 _inference_forward: in-place, row-chunked by inplace_chunk_size 256) in place of the cuEquivariance fused kernel (triangular.py:494-517) — a different implementation of the same block (the path stock itself selects on compute-capability 7.x GPUs, runner/inference.py:564); measured against stock",
              applies=_none, description="configs.triangle_multiplicative='torch' on the stock runner's configs (the model reads self.configs.triangle_multiplicative at every pairformer block; protenix.py:252-294): the trunk's whole-plane TriMul transients (~5 bf16 [N,N,128] planes per block at the cuEq site) become row blocks",
              preconditions=("runner.configs",), settings=("tokens",))
    def trimul_torch(ctx: Ctx) -> Applied:
        return Applied(lever="trimul_torch", settings={"triangle_multiplicative": "torch"}, sites=("InferenceRunner.configs.triangle_multiplicative", "Protenix.configs (the same ConfigDict, protenix.py:98)"))

    @register("msa_zfree", family="allocator", exact="bitwise", replace=True, strategy=TABLE["msa_zfree"]["strategy"],
              exact_reason=MSA_ZFREE_REASON, applies=_none,
              description="each MSA block's pair input released (storage resize to 0) once the outer-product-mean residual produced the block's new z; the stock body is called, not re-stated",
              preconditions=("import:protenix.model.modules.pairformer (MSABlock binds outer_product_mean_msa, pair_stack)",), settings=())
    def msa_zfree(ctx: Ctx) -> Applied:
        return Applied(lever="msa_zfree", settings={}, sites=("MSABlock.forward", "MSAModule.forward"))

    @register("diffcache_free", family="allocator", exact="bitwise", replace=True, strategy=TABLE["diffcache_free"]["strategy"],
              exact_reason=DIFFCACHE_FREE_REASON, applies=_none,
              description="the diffusion roll-out's shared caches (pair_z, p_lm, c_l of protenix.py _main_inference_loop) released at the confidence head's entry",
              preconditions=("sha256:protenix.model.protenix",), settings=())
    def diffcache_free(ctx: Ctx) -> Applied:
        _module_pinned("diffcache_free", "protenix.model.protenix")
        return Applied(lever="diffcache_free", settings={}, sites=("DiffusionConditioning.prepare_cache", "AtomAttentionEncoder.prepare_cache", "Protenix.run_confidence_head"))

    @register("drop_bond_mask", family="setting", exact="bitwise", replace=True, strategy=TABLE["drop_bond_mask"]["strategy"],
              exact_reason="bond_mask is read by the training loss only (protenix/model/loss.py:1696,1704); no inference path reads it",
              applies=_none, description="the INT64 [N_atom, N_atom] bond_mask dropped from the item before to_device (runner/inference.py:226)",
              preconditions=("runner.predict",), settings=())
    def drop_bond_mask(ctx: Ctx) -> Applied:
        return Applied(lever="drop_bond_mask", settings={"key": "bond_mask"}, sites=("InferenceRunner.predict",))

    @register("relp_lean", family="chunk", exact="bitwise", replace=True, strategy=TABLE["relp_lean"]["strategy"],
              exact_reason="the same fp32 0/1 relative-position features written row-block by row-block; no INT64 one-hot intermediates",
              applies=_none, description="RelativePositionEncoding.generate_relp (embedders.py:150-201) in row blocks of <rows> tokens",
              preconditions=("sha256:protenix.model.modules.embedders",), settings=("rows",))
    def relp_lean(ctx: Ctx) -> Applied:
        rows = ctx.setting("relp_lean", "rows", 512, cast=int)
        if rows < 1:
            raise RefusalError(refuse("relp_lean", "rows", f"rows={rows}: a positive row count"))
        _module_pinned("relp_lean", "protenix.model.modules.embedders")
        return Applied(lever="relp_lean", settings={"rows": rows}, sites=("RelativePositionEncoding.generate_relp",))

    @register("diffusion_cond_chunk", family="chunk", exact="measured", replace=True, strategy=TABLE["diffusion_cond_chunk"]["strategy"],
              exact_reason="position-wise ops (LayerNorm, Linear, Transition) computed in row blocks: the GEMM's M changes, decided by measurement against stock",
              applies=_none, description="DiffusionConditioning.prepare_cache (diffusion.py:86-107) in row blocks of <rows> tokens into one [N,N,c_z] output",
              preconditions=("sha256:protenix.model.modules.diffusion",), settings=("rows",))
    def diffusion_cond_chunk(ctx: Ctx) -> Applied:
        rows = ctx.setting("diffusion_cond_chunk", "rows", 256, cast=int)
        if rows < 1:
            raise RefusalError(refuse("diffusion_cond_chunk", "rows", f"rows={rows}: a positive row count"))
        _module_pinned("diffusion_cond_chunk", "protenix.model.modules.diffusion")
        return Applied(lever="diffusion_cond_chunk", settings={"rows": rows}, sites=("DiffusionConditioning.prepare_cache",))

    @register("conf_head_chunk", family="chunk", exact="measured", replace=True, strategy=TABLE["conf_head_chunk"]["strategy"],
              exact_reason="the PAE/PDE tail (LayerNorm + Linear, position-wise) in row blocks: the GEMM's M changes, decided by measurement against stock",
              applies=_none, description="ConfidenceHead.memory_efficient_forward (confidence.py:258-348) with the PAE/PDE tail in row blocks of <rows> tokens; logits to the CPU above 2000 tokens as the stock does",
              preconditions=("sha256:protenix.model.modules.confidence",), settings=("rows",))
    def conf_head_chunk(ctx: Ctx) -> Applied:
        rows = ctx.setting("conf_head_chunk", "rows", 256, cast=int)
        if rows < 1:
            raise RefusalError(refuse("conf_head_chunk", "rows", f"rows={rows}: a positive row count"))
        _module_pinned("conf_head_chunk", "protenix.model.modules.confidence")
        return Applied(lever="conf_head_chunk", settings={"rows": rows}, sites=("ConfidenceHead.memory_efficient_forward",))


# ============================================================================================================== census helpers
def _rec():
    return _STATE.get("record")


def _event(kind: str, lever: str, text: str = "") -> None:
    """One census event on the open unit (mark / fallback / skip). The record refuses a unit-scope event with no unit open (a statement
    reached outside `predict`): noted on the record by name, never raised into the fold."""
    rec = _rec()
    if rec is None or (kind != "fallback" and lever not in rec.levers):
        return
    try:
        if kind == "mark":
            rec.mark(lever, unit=_STATE.get("unit"), detail=text)
        elif kind == "fallback":
            rec.fallback(lever, text, unit=_STATE.get("unit"))
        else:
            rec.skip(lever, text, unit=_STATE.get("unit"))
    except ValueError as e:
        rec.note(f"{lever}: {kind} outside a unit ({text}): {e}")


def _mark(lever: str, detail: str = "") -> None:
    _event("mark", lever, detail)


def _fallback(lever: str, reason: str) -> None:
    _event("fallback", lever, reason)


def _skip(lever: str, reason: str) -> None:
    _event("skip", lever, reason)


def _stage_release() -> bool:
    """The confidence head's stage boundary (the upstream's own torch.cuda.empty_cache() site): the cache_release lever's call when applied
    (recorded), else the stock statement itself."""
    ctx = _STATE.get("ctx")
    rec = _rec()
    if ctx is not None and rec is not None and "cache_release" in rec.levers:
        return ALLOC.release(ctx, "stage", unit=_STATE.get("unit"))
    import torch
    torch.cuda.empty_cache()
    return True


# ============================================================================================================== the patched stock statements
# the chunk lever's label is FIXED at install, never decided per call — `band`, the label of the lever's fp32 site (the runner keeps
# the diffusion conditioning in fp32 up to 3840 tokens, runner/inference.py update_inference_configs skip_amp.sample_diffusion; a bf16
# autocast site above is row-local and bit-equal, which the record notes as information).
EXACT_DIFFUSION_COND_CHUNK = "band"
EXACT_DIFFUSION_COND_CHUNK_REASON = "fp32 GEMM site (≤ 3840 tokens: the runner's skip_amp.sample_diffusion keeps prepare_cache in fp32): launch-extent-dependent reduction order (the chunk module's evidence); the label is fixed at install (B20)"


MSA_ZFREE_REASON = ("a storage release of each MSA block's pair input after its last read (the outer-product-mean residual add makes the block's new z; "
                    "MSAModule.forward / checkpoint_blocks / get_pairformer_output rebind z and never read the input again) — arithmetic untouched")
DIFFCACHE_FREE_REASON = ("a storage release of the diffusion roll-out's shared caches (pair_z, p_lm, c_l) at the confidence head's entry: sampling has ended, "
                         "the confidence head reads z_trunk and the features, nothing reads the caches again (protenix.py _main_inference_loop, pinned) — arithmetic untouched")


def free_storage(t) -> int:
    """The release levers' one primitive: resize a dead tensor's storage to 0 bytes in place (the tensor object and every frame's reference to
    it stay valid; the caching allocator gets the block back now instead of at the last reference's death). Returns the bytes released — 0 for
    None, a non-tensor, or an already-empty storage (idempotent)."""
    try:
        import torch
    except Exception:                                                           # noqa: BLE001
        return 0
    if not torch.is_tensor(t):
        return 0
    st = t.untyped_storage()
    nb = int(st.nbytes())
    if nb > 0:
        st.resize_(0)
    return nb


def _free_dead(lever: str, t) -> int:
    """Release one dead tensor's storage (free_storage); bytes released (0: nothing was held)."""
    return free_storage(t)


def _install_msa_zfree() -> Callable[[], None]:
    """MSABlock.forward (protenix.model.modules.pairformer) wrapped at class level — the stock body is CALLED, never re-stated: the wrapper
    remembers the block's pair input and registers, for the duration of the call, forward pre-hooks on the block's own msa_stack / pair_stack;
    both run with the block's NEW z (the outer-product-mean residual's result), so the first of them to fire releases the remembered input
    when its storage is not the new z's (free_storage). One mark per item (blocks and bytes released); an
    autograd-enabled call, a residual that returned the input's storage, and an item whose MSA blocks never ran (MSAModule.forward returns
    before its blocks without MSA features) are named skips."""
    import importlib
    import torch
    mod = importlib.import_module("protenix.model.modules.pairformer")
    cls, mcls = mod.MSABlock, mod.MSAModule
    stock, mstock = cls.forward, mcls.forward
    for name in ("outer_product_mean_msa", "pair_stack"):
        if name not in cls.__init__.__code__.co_names:
            raise RefusalError(refuse("msa_zfree", "protenix.model.modules.pairformer", f"MSABlock no longer binds {name!r}: the release points are the block's msa_stack / pair_stack calls after the outer-product-mean residual"))
    tally = {"unit": object(), "bytes": 0, "blocks": 0, "calls": 0}

    def _unit_sync():
        u = _STATE.get("unit")
        if tally["unit"] != u:
            tally.update(unit=u, bytes=0, blocks=0, calls=0)

    def forward(self, m, z, *args, _stock=stock, **kwargs):
        _unit_sync(); tally["calls"] += 1
        if not torch.is_tensor(z) or torch.is_grad_enabled():
            _skip("msa_zfree", "autograd enabled or no pair tensor: nothing released")
            return _stock(self, m, z, *args, **kwargs)
        pending = [z]

        def release(_mod, a, k):                                          # fires at the block's msa_stack(m, z_new) / pair_stack(z=z_new) call: the input is dead by then
            if not pending:
                return None
            z_in = pending.pop()
            z_new = k.get("z") if "z" in k else next((t for t in a if torch.is_tensor(t) and t.shape == z_in.shape), None)
            if torch.is_tensor(z_new) and z_new.untyped_storage().data_ptr() == z_in.untyped_storage().data_ptr():
                _skip("msa_zfree", "the residual add returned the input's storage: nothing to release")
                return None
            nb = _free_dead("msa_zfree", z_in)
            tally["bytes"] += nb; tally["blocks"] += 1
            _mark("msa_zfree", f"blocks={tally['blocks']} released_bytes={tally['bytes']}")
            return None
        handles = [self.pair_stack.register_forward_pre_hook(release, with_kwargs=True)]
        ms = getattr(self, "msa_stack", None)
        if ms is not None and not getattr(self, "is_last_block", False):
            handles.append(ms.register_forward_pre_hook(release, with_kwargs=True))   # the earlier point on every block but the last (which has no MSA stack call)
        try:
            return _stock(self, m, z, *args, **kwargs)
        finally:
            for h in handles:
                h.remove()
            pending.clear()

    def module_forward(self, *a, _stock=mstock, **k):                    # an item whose MSA blocks never ran: a named skip, never an absent lever
        out = _stock(self, *a, **k)
        _unit_sync()
        if tally["calls"] == 0:
            _skip("msa_zfree", "the MSA blocks did not run on this item (no MSA features)")
        return out

    forward.__wrapped__ = stock; module_forward.__wrapped__ = mstock
    cls.forward = forward; mcls.forward = module_forward

    def undo():
        setattr(cls, "forward", stock); setattr(mcls, "forward", mstock)
    return undo


def _install_diffcache_free() -> Callable[[], None]:
    """DiffusionConditioning.prepare_cache and AtomAttentionEncoder.prepare_cache wrapped at class level (whatever owns them when this lever
    installs — the stock body or diffusion_cond_chunk's row-blocked one — is CALLED): their outputs, the roll-out's shared caches
    (protenix.py _main_inference_loop `cache["pair_z"]`, `cache["p_lm/c_l"]`), are remembered by weak reference (a per-step call with the
    shared-variable cache off dies by refcount as before); Protenix.run_confidence_head wrapped: at its entry the caches still alive are
    released (free_storage) — one mark per item with the bytes, or a named skip when nothing was alive."""
    import importlib
    import weakref
    import torch
    _module_pinned("diffcache_free", "protenix.model.protenix")              # the dead-tensor claim is reviewed against the pinned _main_inference_loop (a different file refuses by name)
    PX = importlib.import_module("protenix.model.protenix")
    DM = importlib.import_module("protenix.model.modules.diffusion")
    TF = importlib.import_module("protenix.model.modules.transformer")
    ccls, ecls, pcls = DM.DiffusionConditioning, TF.AtomAttentionEncoder, PX.Protenix
    c_prev, e_prev, p_prev = ccls.prepare_cache, ecls.prepare_cache, pcls.run_confidence_head
    held: List[Any] = []

    def remember(out):
        for t in (out if isinstance(out, (tuple, list)) else (out,)):
            if torch.is_tensor(t) and t.is_cuda and not torch.is_grad_enabled():
                held.append(weakref.ref(t))
        return out

    def cond_prepare_cache(self, *a, _prev=c_prev, **k):
        return remember(_prev(self, *a, **k))

    def enc_prepare_cache(self, *a, _prev=e_prev, **k):
        return remember(_prev(self, *a, **k))

    def run_confidence_head(self, *a, _prev=p_prev, **k):
        alive = [t for t in (r() for r in held) if t is not None]
        held.clear()
        nb = sum(_free_dead("diffcache_free", t) for t in alive)
        del alive
        if nb == 0:
            _skip("diffcache_free", "no diffusion cache alive at the confidence head (shared-variable cache off, or already released)")
        else:
            _mark("diffcache_free", f"released_bytes={nb}")
        return _prev(self, *a, **k)

    cond_prepare_cache.__wrapped__ = c_prev; enc_prepare_cache.__wrapped__ = e_prev; run_confidence_head.__wrapped__ = p_prev
    cond_prepare_cache._big = getattr(c_prev, "_big", False)             # a marker an owner of prepare_cache may read for idempotence is carried
    ccls.prepare_cache = cond_prepare_cache; ecls.prepare_cache = enc_prepare_cache; pcls.run_confidence_head = run_confidence_head

    def undo():
        setattr(ccls, "prepare_cache", c_prev); setattr(ecls, "prepare_cache", e_prev); setattr(pcls, "run_confidence_head", p_prev); held.clear()
    return undo


def _install_relp_lean(rows: int) -> Callable[[], None]:
    import torch
    import torch.nn.functional as F
    mod = _module_pinned("relp_lean", "protenix.model.modules.embedders")
    cls = mod.RelativePositionEncoding
    stock = cls.generate_relp

    def generate_relp(self, input_feature_dict, _rows=rows):
        """embedders.py:150-201 in row blocks (_relp.relp_rows_into: the same expression per block of query tokens), written straight into the fp32 output."""
        from . import _relp as RL
        with torch.no_grad():
            asym_id = input_feature_dict["asym_id"]
            n = asym_id.shape[-1]
            out = torch.empty(asym_id.shape[:-1] + (n, n, RL.relp_width(self)), dtype=torch.float32, device=asym_id.device)
            for i in range(0, n, _rows):
                j = min(i + _rows, n)
                RL.relp_rows_into(self, input_feature_dict, i, j, out[..., i:j, :, :])
            input_feature_dict["relp"] = out
        _mark("relp_lean", f"rows={_rows} n={n}")
        return input_feature_dict

    generate_relp.__wrapped__ = stock
    cls.generate_relp = generate_relp
    return lambda: setattr(cls, "generate_relp", stock)


def _install_recycle_carry(park: str) -> Callable[[], None]:
    """Protenix.get_pairformer_output (protenix.py:170-303 of the pinned wheel, stock/PINS.json model_files) transcribed line for line with
    the core's RecycleCarry (opt_core.mem.ckpt.carry_of: the recycle_carry lever's carry) at the seams: z_init is HELD (parked on host when park='host'; the device buffer dropped) right after it is built,
    brought back as a fresh device copy at each cycle start for `z = z_init + ...` and dropped with the expression, and the allocator is
    flushed at every recycle seam; released after the loop. Every other statement is the upstream's own."""
    import torch
    import torch.nn.functional as F
    from opt_core.mem import ckpt as CK
    mod = _module_pinned("recycle_carry", "protenix.model.protenix")
    cls = mod.Protenix
    stock = cls.get_pairformer_output

    def get_pairformer_output(self, input_feature_dict, N_cycle, inplace_safe=False, chunk_size=None, mc_dropout=False, mc_dropout_rate=0.4, _park=park):
        if self.train_confidence_only:
            self.input_embedder.eval(); self.template_embedder.eval(); self.msa_module.eval(); self.pairformer_stack.eval()
        s_inputs = self.input_embedder(input_feature_dict, inplace_safe=False, chunk_size=chunk_size)
        z_constraint = None
        if "constraint_feature" in input_feature_dict:
            z_constraint = self.constraint_embedder(input_feature_dict["constraint_feature"])
        s_init = self.linear_no_bias_sinit(s_inputs)
        z_init = (self.linear_no_bias_zinit1(s_init)[..., None, :] + self.linear_no_bias_zinit2(s_init)[..., None, :, :])
        if inplace_safe:
            z_init += self.relative_position_encoding(input_feature_dict["relp"])
            z_init += self.linear_no_bias_token_bond(input_feature_dict["token_bonds"].unsqueeze(dim=-1))
            if z_constraint is not None:
                z_init += z_constraint
        else:
            z_init = z_init + self.relative_position_encoding(input_feature_dict["relp"])
            z_init = z_init + self.linear_no_bias_token_bond(input_feature_dict["token_bonds"].unsqueeze(dim=-1))
            if z_constraint is not None:
                z_init = z_init + z_constraint
        z = torch.zeros_like(z_init)
        s = torch.zeros_like(s_init)
        carry = CK.carry_of(_STATE["ctx"])                            # the core lever's carry (opt_core.mem.ckpt recycle_carry: the host park opened at activation, pin_max_gb its budget)
        carry.hold(z_init=z_init)                                    # the seam: z_init leaves the device (park='host') for the trunk body; the parked buffer is re-used item after item
        del z_init
        for cycle_no in range(N_cycle):
            with torch.set_grad_enabled(self.training and (not self.train_confidence_only) and cycle_no == (N_cycle - 1)):
                z_init_dev = carry.get("z_init")                     # a fresh device copy, alive for this expression only
                if mc_dropout:
                    z = z_init_dev + F.dropout(self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z)), p=self.configs.mc_dropout_rate)
                else:
                    z = z_init_dev + self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z))
                del z_init_dev
                if inplace_safe:
                    if self.template_embedder.n_blocks > 0:
                        z += self.template_embedder(input_feature_dict, z, triangle_multiplicative=self.configs.triangle_multiplicative, triangle_attention=self.configs.triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size)
                    z = self.msa_module(input_feature_dict, z, s_inputs, pair_mask=None, triangle_multiplicative=self.configs.triangle_multiplicative, triangle_attention=self.configs.triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size)
                else:
                    if self.template_embedder.n_blocks > 0:
                        z = z + self.template_embedder(input_feature_dict, z, triangle_multiplicative=self.configs.triangle_multiplicative, triangle_attention=self.configs.triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size)
                    z = self.msa_module(input_feature_dict, z, s_inputs, pair_mask=None, triangle_multiplicative=self.configs.triangle_multiplicative, triangle_attention=self.configs.triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size)
                s = s_init + self.linear_no_bias_s(self.layernorm_s(s))
                s, z = self.pairformer_stack(s, z, pair_mask=None, triangle_multiplicative=self.configs.triangle_multiplicative, triangle_attention=self.configs.triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size)
                carry.seam(cycle_no)                                 # the boundary: the cycle's transients are dead; the caching allocator flushed
        _mark("recycle_carry", f"park={_park} cycles={N_cycle} held={carry.record['held'].get('z_init', {}).get('bytes')}")   # the park stays open for the next item (the core releases it with the process: opt_core.mem.undo)
        if self.train_confidence_only:
            self.input_embedder.train(); self.template_embedder.train(); self.msa_module.train(); self.pairformer_stack.train()
        return s_inputs, s, z

    get_pairformer_output.__wrapped__ = stock
    cls.get_pairformer_output = get_pairformer_output
    return lambda: setattr(cls, "get_pairformer_output", stock)


def _install_diffusion_cond_chunk(rows: int) -> Callable[[], None]:
    """prepare_cache (diffusion.py:86-107) through the core's row chunker (opt_core.mem.chunk.chunk_rows): the stock's own submodules per
    block of query rows into one [N,N,c_z] output. The declaration passed to the chunker is the lever's fixed label (EXACT_DIFFUSION_COND_CHUNK):
    `band` — the fp32 GEMM site's class (the site is fp32 up to 3840 tokens, the runner's skip_amp.sample_diffusion; autocast bf16
    above, row-local and bit-equal, noted on the record as information, never as the label)."""
    import torch
    from opt_core.mem import chunk as CHUNK
    mod = _module_pinned("diffusion_cond_chunk", "protenix.model.modules.diffusion")
    cls = mod.DiffusionConditioning
    stock = cls.prepare_cache
    for name in ("relpe", "layernorm_z", "linear_no_bias_z", "transition_z1", "transition_z2"):
        if name not in cls.__init__.__code__.co_names:
            raise RefusalError(refuse("diffusion_cond_chunk", "protenix.model.modules.diffusion", f"DiffusionConditioning no longer binds {name!r}: the carried patch does not fit"))

    def prepare_cache(self, relp_feature, z_trunk, inplace_safe=False, _rows=rows):
        def block(z_blk, i0, i1):
            p = torch.cat([z_blk, self.relpe(relp_feature[..., i0:i1, :, :])], dim=-1)
            p = self.linear_no_bias_z(self.layernorm_z(p))
            p += self.transition_z1(p)
            p += self.transition_z2(p)
            return p
        fp32_site = z_trunk.dtype == torch.float32 and not CHUNK.autocast_active(z_trunk.device.type)
        if not fp32_site and not _STATE.get("diffusion_cond_chunk_bf16_seen"):
            _STATE["diffusion_cond_chunk_bf16_seen"] = True             # information for the record (the site's dtype at this call), never the label
        return CHUNK.chunk_rows(block, z_trunk, -3, _rows, exact=EXACT_DIFFUSION_COND_CHUNK, reason=EXACT_DIFFUSION_COND_CHUNK_REASON, record=_rec(), with_offsets=True,
                                lever="diffusion_cond_chunk", site="DiffusionConditioning.prepare_cache")   # the record's per-call sink (AppliedRecord.call): the unit marked ran, the label counted per site

    prepare_cache.__wrapped__ = stock
    cls.prepare_cache = prepare_cache
    return lambda: setattr(cls, "prepare_cache", stock)


def _install_conf_head_chunk(rows: int) -> Callable[[], None]:
    """memory_efficient_forward (confidence.py:258-348) carried with its PAE/PDE tail (:317-331) through the core's row chunker: one
    block of query rows upcast, the two heads on it, the logits assembled into [N,N,64] outputs — on the CPU above 2000 tokens (where
    the stock moves them, :229-233). An fp32 GEMM site: declared `band` to the chunker (the chunk module's evidence)."""
    import torch
    from opt_core.mem import chunk as CHUNK
    mod = _module_pinned("conf_head_chunk", "protenix.model.modules.confidence")
    cls = mod.ConfidenceHead
    stock = cls.memory_efficient_forward
    one_hot, broadcast_token_to_atom = mod.one_hot, mod.broadcast_token_to_atom
    reason = "fp32 GEMM site (the heads run with autocast off): launch-extent-dependent reduction order (the chunk module's evidence)"

    def memory_efficient_forward(self, input_feature_dict, s_trunk, z_pair, pair_mask, x_pred_rep_coords, triangle_multiplicative="torch",
                                 triangle_attention="torch", inplace_safe=False, chunk_size=None, _rows=rows):
        with torch.amp.autocast("cuda", enabled=False):
            x_pred_rep_coords = x_pred_rep_coords.to(torch.float32)
            distance_pred = torch.cdist(x_pred_rep_coords, x_pred_rep_coords)
        if inplace_safe:
            z_pair += self.linear_no_bias_d(one_hot(x=distance_pred, lower_bins=self.lower_bins, upper_bins=self.upper_bins))
            z_pair += self.linear_no_bias_d_wo_onehot(distance_pred.unsqueeze(dim=-1))
        else:
            z_pair = z_pair + self.linear_no_bias_d(one_hot(x=distance_pred, lower_bins=self.lower_bins, upper_bins=self.upper_bins))
            z_pair = z_pair + self.linear_no_bias_d_wo_onehot(distance_pred.unsqueeze(dim=-1))
        s_single, z_pair = self.pairformer_stack(s_trunk, z_pair, pair_mask, triangle_multiplicative=triangle_multiplicative,
                                                 triangle_attention=triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size)
        s_single = s_single.to(torch.float32)
        atom_to_token_idx = input_feature_dict["atom_to_token_idx"]
        atom_to_tokatom_idx = input_feature_dict["atom_to_tokatom_idx"]
        n = z_pair.shape[-2]
        big = (not self.training) and n > 2000                      # the stock's CPU residency rule for the logits (confidence.py:229-233)
        with torch.amp.autocast("cuda", enabled=False):
            def pae_rows(z_blk, i0, i1):
                return self.linear_no_bias_pae(self.pae_ln(z_blk.to(torch.float32)))
            def pde_rows(z_blk, i0, i1):
                return self.linear_no_bias_pde(self.pde_ln(z_blk.to(torch.float32) + z_pair[..., :, i0:i1, :].transpose(-2, -3).to(torch.float32)))
            outs = {}
            for name, fn, lin in (("pae", pae_rows, self.linear_no_bias_pae), ("pde", pde_rows, self.linear_no_bias_pde)):
                out = torch.empty(z_pair.shape[:-1] + (lin.weight.shape[0],), dtype=torch.float32, device="cpu") if big else None
                outs[name] = CHUNK.chunk_rows(fn, z_pair, -3, _rows, exact="band", reason=reason, record=_rec(), out=out, with_offsets=True,
                                              lever="conf_head_chunk", site=f"ConfidenceHead.{name}")
            pae_pred, pde_pred = outs["pae"], outs["pde"]
            a = broadcast_token_to_atom(x_token=s_single, atom_to_token_idx=atom_to_token_idx)
            plddt_pred = torch.einsum("...nc,ncb->...nb", self.plddt_ln(a), self.plddt_weight[atom_to_tokatom_idx])
            resolved_pred = torch.einsum("...nc,ncb->...nb", self.resolved_ln(a), self.resolved_weight[atom_to_tokatom_idx])
        if big:
            _stage_release()                                            # the stock's OWN release at its own site (confidence.py:347): the cache_release lever's recorded call (opt_core.mem.allocator.release, point stage), the stock torch.cuda.empty_cache() without the lever
        return plddt_pred, pae_pred, pde_pred, resolved_pred

    memory_efficient_forward.__wrapped__ = stock
    cls.memory_efficient_forward = memory_efficient_forward
    return lambda: setattr(cls, "memory_efficient_forward", stock)


# ============================================================================================================== activation: arm / apply
def _base_table():
    """This kit's modes as an opt_core.modes.ModeTable (big composes on it: compose_big names the base and declares the line)."""
    return CORE_MODES.ModeTable(modes=tuple(m for m in M.MODES if m != MODE), default=M.DEFAULT_MODE)


def route_opt_out(rep: Optional[dict]) -> Optional[str]:
    """How the PARTIAL lines name the census opt-out on this route: the environment route (a .pth trigger in the report) names its word
    `PROTENIX_V1_OPT_ALLOW_PARTIAL=1`; the verbs keep the core's default (`--allow-partial`) — None."""
    from . import stack as S
    return f"{S.ENV_ALLOW_PARTIAL}=1" if (rep or {}).get("trigger") else None


def rowpair_switches(P: int) -> Dict[str, bool]:
    """The ×P line's selection at n_gpu P>1: the four single-GPU levers whose sites the row-sharded path replaces (ngpu.SUPERSEDED_LEVERS)
    OFF, by name — stated in-process in every rank (the LEVER line reads `state=off reason=flag`); {} at P=1."""
    from . import ngpu as NGPU
    return {lv: False for lv in NGPU.SUPERSEDED_LEVERS} if int(P) > 1 else {}


def line_tier(levers) -> str:
    """The declared tier of the line AS COMPOSED at this size (opt_core.mem.record.compose_exact over the TABLE classes of the levers that arm):
    `bitwise` at or below the memory gate (only the free levers arm), `measured` above it — so the core's declared-vs-composed check has
    nothing to note."""
    from opt_core.mem import record as REC
    return REC.compose_exact(TABLE[lv]["class"] for lv in levers if lv in TABLE)


def arm(res: M.Resolution, environ=None, *, allow_partial: bool = False, opt_out: Optional[str] = None,
        switches: Optional[Dict[str, bool]] = None) -> dict:
    """The activation-time part (module contract): the composition's levers through opt_core.mem.apply with the line's EXPLICIT selection
    (the levers of LINE; at n_gpu P>1 the superseded four off by name: rowpair_switches; `switches` = further named on/off a caller in this
    process states — the CPU tests turn the CUDA levers off), the settings (line_settings: constants + the two size statements), the census
    opt-out (`allow_partial`: the verb's `--allow-partial` or the environment route's word, which `opt_out` names on the PARTIAL lines;
    None = the flag), the allocator lever exported (before CUDA is initialised when called from stack.activate) and recycle_carry's host park
    opened. Returns the report's `big` fields; BigError names a refusal. Idempotent within a process."""
    if _STATE.get("record") is not None:
        return fields()
    comp = compose() if environ is None else compose(environ)
    _register_kit_levers()
    environ = os.environ if environ is None else environ
    hooks = {"recycle_carry": {"device": "cuda", "carried": ["z_init"]}}
    settings = line_settings(environ)
    sel = dict(rowpair_switches(comp["gates"]["rowpair_world"]))
    from . import ablation as A                                              # MODEL_OPT_LEVERS_OFF (modes.resolve put the names on res.ablated): the named memory levers OFF in the line's own selection, by name
    sel.update(A.switches(getattr(res, "ablated", ()) or (), LINE))
    sel.update(switches or {})
    sel = dict(sorted(sel.items()))                                          # named by lever, in name order (the record's off / on lists and their LEVER lines follow it)
    ctx = Ctx(prefix=FLAG_PREFIX, tag=TAG, framework="torch", hooks=hooks, settings=settings, environ=environ, graphs=False,
              extra={"line": comp["line"], "arm": comp["arm"], "n_token": comp["size"]["n_token"], "n_token_source": comp["size"]["source"],
                     "gates": {k: v for k, v in comp["gates"].items() if k != "off_by_property"}, "off_by_property": dict(comp["gates"]["off_by_property"])},
              **({"opt_out": opt_out} if opt_out else {}))
    _, bline = COMPOSE.compose_big(_base_table(), base=BASE_MODE, levers=comp["levers"], drop=comp["drop"], tier=line_tier(comp["levers"]),
                                     cost_note="at or below the memory gate the fast base's arithmetic less sg / hoist (the free levers only); above it slower than fast where fast fits (ladder_table v11: 1H6K 2565 483.0 s vs fast 398.8 s)")
    try:
        rec = mem_apply(bline, ctx, base=None, strict=True, switches=sel or None, allow_partial=bool(allow_partial))
    except Refused as e:
        rec = e.record
        names = "; ".join(f"{r.lever}: {r.precondition}: {r.reason}" for r in rec.refused)
        raise BigError(f"big refused — {names}") from None
    _STATE.update(record=rec, ctx=ctx)
    return fields()


STOCK_ATTENTION_PACKAGE = "cuequivariance_torch"                    # the upstream's attention / TriMul kernel package (configs triangle_attention=cuequivariance): imported lazily by the stock at its first call


def preload_stock_attention() -> dict:
    """Import the stock attention core's package when the line is applied (a CUDA process only), so that its module-level tables
    (cuequivariance_ops_torch builds large ahead-of-time autotune key lists at import) are built BEFORE the trunk: imported lazily
    inside the first item's pair stack — the upstream's own first `cuequivariance_triangular_attn` call, which the fused block's stock-gated
    calls reach on the unchunked path — the same import takes far longer there than at process start. The import is the upstream's own dependency and happens in
    every run anyway; this only moves it ahead of the forward. Never a refusal: a CPU process or a missing package records why and the stock
    imports as before."""
    import importlib
    import time
    try:
        import torch
        if not torch.cuda.is_available():
            return {"package": STOCK_ATTENTION_PACKAGE, "preloaded": False, "reason": "no CUDA device (nothing to preload for)"}
    except Exception as e:                                                      # noqa: BLE001
        return {"package": STOCK_ATTENTION_PACKAGE, "preloaded": False, "reason": f"torch unavailable: {type(e).__name__}"}
    t0 = time.time()
    try:
        importlib.import_module(STOCK_ATTENTION_PACKAGE)
    except Exception as e:                                                      # noqa: BLE001 — the stock imports it itself at its first call and says so then
        if is_oom(e): raise
        return {"package": STOCK_ATTENTION_PACKAGE, "preloaded": False, "reason": f"{type(e).__name__}: {e}"}
    return {"package": STOCK_ATTENTION_PACKAGE, "preloaded": True, "seconds": round(time.time() - t0, 1)}


def apply(runner, res: M.Resolution) -> dict:
    """After the stock runner is built (stack's runner __init__ wrap; module contract): the settings on runner.configs (chunk_pair — the
    upstream's own fields set in place, a missing field refused by name; trimul_torch on the model's configs too), the patched stock methods
    installed, the census hooks attached to this instance's predict. Returns {applied: {dotted: value}, replaced: {dotted: before}} — the record
    the MEMORY line prints (chunk_pair's preset; trimul_torch's setting when engaged)."""
    if _STATE.get("record") is None:
        try:
            from . import stack as S                                       # the environment route without the activation seam: armed here (CUDA initialised: the allocator lever's runtime-API path, recorded)
            rep = S.status()
            arm(res, allow_partial=bool(rep.get("allow_partial")), opt_out=route_opt_out(rep))
        except BigError as e:
            from .stack import ActivationError
            raise ActivationError(str(e))
    rec, ctx = _STATE["record"], _STATE["ctx"]
    from .stack import ActivationError
    configs = getattr(runner, "configs", None)
    if configs is None:
        raise ActivationError("big: the stock runner carries no `configs` (chunk_pair sets infer_setting fields on it)")
    _STATE["stock_attention_preload"] = preload_stock_attention()             # the stock attention core's package imported now, not inside the first item's pair stack (see preload_stock_attention)
    applied, replaced, installed, undo = {}, {}, {}, []
    try:
        for a in rec.applied:
            s = dict(a.settings)
            if a.lever == "chunk_pair":
                preset = {k: v for k, v in s.items() if k.startswith("infer_setting.")}
                for key, value in preset.items():
                    replaced[key] = _cfg_set(configs, key, value, "chunk_pair"); applied[key] = value
                installed[a.lever] = preset
            elif a.lever == "trimul_torch":
                replaced["triangle_multiplicative"] = _cfg_set(configs, "triangle_multiplicative", "torch", "trimul_torch"); applied["triangle_multiplicative"] = "torch"
                mcfg = getattr(getattr(runner, "model", None), "configs", None)
                if mcfg is not None and mcfg is not configs:
                    _cfg_set(mcfg, "triangle_multiplicative", "torch", "trimul_torch")
                installed[a.lever] = {"triangle_multiplicative": "torch"}
            elif a.lever == "drop_bond_mask":
                installed[a.lever] = {}
            elif a.lever == "relp_lean":
                undo.append(_install_relp_lean(int(s["rows"]))); installed[a.lever] = s
            elif a.lever == "recycle_carry":
                _module_pinned("recycle_carry", "protenix.model.protenix")
                undo.append(_install_recycle_carry(str(s["park"]))); installed[a.lever] = {k: s[k] for k in ("park", "flush", "pin_max_gb") if k in s}
            elif a.lever == "diffusion_cond_chunk":
                undo.append(_install_diffusion_cond_chunk(int(s["rows"]))); installed[a.lever] = s
            elif a.lever == "conf_head_chunk":
                undo.append(_install_conf_head_chunk(int(s["rows"]))); installed[a.lever] = s
            elif a.lever == "msa_zfree":
                undo.append(_install_msa_zfree()); installed[a.lever] = {}
            elif a.lever == "diffcache_free":
                undo.append(_install_diffcache_free()); installed[a.lever] = {}
            else:
                installed[a.lever] = s                                     # the core's process-scope levers (expandable_segments, cache_release): applied at arm
    except RefusalError as e:
        for fn in reversed(undo):
            fn()
        raise ActivationError(f"big refused — {e.refusal.lever}: {e.refusal.precondition}: {e.refusal.reason}")
    _STATE.update(installed=installed, undo=undo, memory={"applied": applied, "replaced": replaced})
    _attach_predict(runner)
    CORE_R.emit(line())
    return {"applied": applied, "replaced": replaced}


def _attach_predict(runner) -> None:
    """The census unit per item: this instance's `predict` opens the unit before the class's predict (the stock body, stack's item record)
    runs and closes it after — the one attach point of the per-item levers."""
    bound = runner.predict

    @functools.wraps(getattr(bound, "__func__", bound))
    def predict(data, *a, **kw):
        unit = before_predict(runner, data)
        try:
            return bound(data, *a, **kw)
        finally:
            after_predict(unit)

    predict.__wrapped__ = bound
    runner.predict = predict


def before_predict(runner, data) -> Optional[str]:
    """The unit opens (the item's sample_name); drop_bond_mask acts; chunk_pair / trimul_torch are re-read on the item's configs (mark, or a
    named fallback); expandable_segments is re-read by the record; the size gate is checked against the featurized N_token."""
    rec = _rec()
    if rec is None:
        return None
    name = data.get("sample_name") if isinstance(data, dict) else None
    unit = str(name if name is not None else f"item{_STATE['units_seen']}")
    if unit in rec.units:
        unit = f"{unit}#{_STATE['units_seen']}"
    _STATE["units_seen"] += 1
    rec.unit_begin(unit)
    _STATE["unit"] = unit
    inst = _STATE.get("installed") or {}
    feats = data.get("input_feature_dict") if isinstance(data, dict) else None
    if "drop_bond_mask" in inst:
        if isinstance(feats, dict) and "bond_mask" in feats:
            t = feats.pop("bond_mask")
            _mark("drop_bond_mask", f"dropped {tuple(getattr(t, 'shape', ()))} {getattr(t, 'dtype', '')}")
        else:
            _skip("drop_bond_mask", "the item carries no bond_mask")
    configs = getattr(runner, "configs", None)
    for lever in ("chunk_pair", "trimul_torch"):
        if lever in inst:
            cfg = (getattr(getattr(runner, "model", None), "configs", None) or configs) if lever == "trimul_torch" else configs
            want = dict(inst[lever]); have = {k: _cfg_get(cfg, k) for k in want}
            if have == want:
                _mark(lever, ",".join(f"{k}={v}" for k, v in have.items()))
            else:
                _fallback(lever, f"configs read {have} on the item, the lever set {want}")
    _size_check(data, feats)
    return unit


def _size_check(data, feats) -> None:
    """The estimate that sized the gates against the item's featurized N_token: an item on the other side of a gate than the composition
    decided is NAMED — one `NOTE size gate re-decided at featurization: …` line stating each crossed lever's kept state, a `size_gate` note and a
    `size_gate_crossings` record on the census — and the run proceeds on the composition the process was sized for (levers are sized once per
    process; nothing re-composes, nothing exits)."""
    rec = _rec()
    comp = _STATE.get("composition") or {}
    n = None
    if isinstance(data, dict):
        n = data.get("N_token")
        if n is None and isinstance(feats, dict) and "residue_index" in feats:
            n = getattr(feats["residue_index"], "shape", [None])[-1]
    try:
        n = int(n) if n is not None else None
    except (TypeError, ValueError):
        n = None
    if rec is None or not comp:
        return
    if n is None:                                                            # named, never silent: the gate could not read this item's size
        rec.note(f"size_gate: unit {_STATE.get('unit')}: the item's N_token could not be read (no data['N_token'], no residue_index) — the size gate did not check this unit")
    est = comp["size"]["n_token"]
    g = comp["gates"]
    crossed: List[str] = []
    kept: List[str] = []
    if n is not None:
        e = -1 if est is None else est
        for gname, thr, members in (("memory", g.get("memory_gate_tokens", MEMORY_GATE_TOKENS), MEMORY_GATED_LEVERS), ("trimul_torch", g["trimul_gate_tokens"], ("trimul_torch",))):
            side_est, side_run = e > thr, n > thr
            if side_est != side_run:
                crossed.append(f"{gname} gate {thr}: sized {est if est is not None else 'unsized'} ({comp['size']['source']}), the item has N_token={n}")
                kept += [f"{lv} {'on' if lv in rec.levers else 'off'}" for lv in members]   # the state the process was sized with (the record's lever set: the gate's side, less any lever off by flag), kept for this item
    entry = {"unit": _STATE.get("unit"), "n_token": n, "estimate": est, "crossed": crossed, "kept": kept}
    rec.extra.setdefault("size_checks", []).append(entry)
    if crossed:
        rec.note(f"size_gate: {'; '.join(crossed)}")
        _STATE.setdefault("size_gate_crossings", []).append(entry)
        from . import report as R
        R.log(R.note_line(f"size gate re-decided at featurization: unit {_STATE.get('unit')} N_token={n} (sized {est if est is not None else 'unsized'}, {comp['size']['source']}): "
                          f"{', '.join(kept)} — the composition the process was sized for is kept for this item; the run proceeds"))


def after_predict(unit: Optional[str]) -> None:
    rec = _rec()
    if rec is None or unit is None:
        return
    ctx = _STATE.get("ctx")
    if ctx is not None:
        ALLOC.release(ctx, "unit", unit=unit)
    rec.unit_end(unit)
    _STATE["unit"] = None


# ============================================================================================================== the record, the lines, the exit
def fields() -> Optional[dict]:
    """The report's `big` block (JSON-safe; None outside the mode): line, arm, base, tokens (estimate + source), gates, property_off, the
    levers applied / refused / off / on, the composed label, the allocator settings, the installed settings, the size checks, the census gate."""
    rec, comp = _rec(), _STATE.get("composition")
    if rec is None or comp is None:
        return None
    block = rec.manifest_block()
    return {"line": comp["line"], "arm": comp["arm"], "base": comp["base"], "base_arm": comp["base_arm"],
            "n_token_estimate": comp["size"]["n_token"], "n_token_source": comp["size"]["source"], "input": comp["size"]["input"],
            "per_item": comp["size"]["per_item"], "unsized": comp["size"]["unsized"],
            "gates": {k: v for k, v in comp["gates"].items() if k != "off_by_property"}, "off_by_property": dict(comp["gates"]["off_by_property"]),
            "stock_attention_preload": _STATE.get("stock_attention_preload"),
            "levers": list(rec.levers), "refused": [r.as_dict() for r in rec.refused], "off_by_flag": list(rec.off_by_flag), "on_by_flag": list(rec.on_by_flag),
            "exact": rec.exact, "exact_per_lever": rec.exact_per_lever, "allocator": dict(rec.allocator_settings or {}),
            "installed": dict(_STATE.get("installed") or {}),
            "size_checks": list(rec.extra.get("size_checks") or []), "size_gate_crossings": list(_STATE.get("size_gate_crossings") or []),
            "excused_by_property": dict(_STATE.get("excused") or {}), "gate": _STATE.get("gate"), "record": block}


def line() -> str:
    """`[protenix-v1-opt] BIG line=big tokens=<n|-> source=<estimate|environment|unsized> base=fast arm=<arm> levers=<a,b,…|->
    exact=<label> refused=<names|none> off=<names|none> on=<names|none> property_off=<names|none> allocator=<k:v,…|none>` — once per process
    when the levers are on the runner (the fields of opt_core.mem's AppliedRecord; the words are opt_core.report's kv grammar)."""
    f = fields() or {}
    names = lambda xs: [x if isinstance(x, str) else x.get("lever", "?") for x in (xs or [])] or None
    return CORE_R.prefix(TAG) + " BIG " + CORE_R.kv(
        ("line", f.get("line")), ("tokens", f.get("n_token_estimate") if f.get("n_token_estimate") is not None else "-"), ("source", f.get("n_token_source")),
        ("base", f.get("base")), ("arm", f.get("arm")), ("levers", f.get("levers") or "-"), ("exact", f.get("exact")),
        ("refused", names(f.get("refused"))), ("off", f.get("off_by_flag") or None), ("on", f.get("on_by_flag") or None),
        ("property_off", sorted(f.get("off_by_property") or {}) or None), ("allocator", f.get("allocator") or None))


def exit_join(partial: List[str], reason: Optional[str], evidence: Dict[str, dict], allow_partial: bool, run_ok: bool) -> Tuple[List[str], Optional[str]]:
    """The exit rule's big part (report.verdict calls it; a no-op outside the mode). (1) A base lever that served no call, took no fallback
    and is off by property at this run's size (gates_for: `fast` / `gflash` above the TriMul gate) is DISENGAGED, not partial — recorded under
    excused_by_property with its counters. (2) The census gate of the memory levers (opt_core.mem.record.exit_gate: every unit's expected lever
    set observed, every refusal) joins the partial state: a partial unit names `<lever>@<unit>`; size-gate crossings are recorded beside it
    (`size_gate_crossings`, each with the kept lever states) and never join it. The census
    opt-out is the kit's `--allow-partial` (passed to the core's gate as the one source): it records and proceeds."""
    rec, comp = _rec(), _STATE.get("composition")
    if rec is None or comp is None:
        return partial, reason
    off = comp["gates"]["off_by_property"]
    excused = {}
    kept = []
    for lv in partial:
        e = evidence.get(lv) or {}
        if lv in off and not e.get("fallback") and not e.get("served"):
            excused[lv] = {"why": off[lv], "gated": e.get("gated") or {}}
        else:
            kept.append(lv)
    if excused:
        _STATE["excused"] = excused
        reasons = [r for r in (reason or "").split("; ") if r and not any(r.startswith(f"{lv} served 0") for lv in excused)]
        reason = "; ".join(reasons) or None
    partial = kept
    rc_in = CORE_R.EXIT_OK if run_ok else CORE_R.EXIT_FAIL
    gate = rec.exit_gate(rc_in, expect_units=bool(run_ok), allow_partial=bool(allow_partial))              # the opt-out: the verb's --allow-partial or the environment route's word (the value arm() passed; stated again here)
    cen = gate.get("census") or {}
    crossings = list(_STATE.get("size_gate_crossings") or [])
    names: List[str] = []
    for lv in gate.get("partial") or []:
        units = [u for u, d in (cen.get("units") or {}).items() if lv in (d.get("partial") or [])]
        names += [f"{lv}@{u}" for u in units] or [f"{lv}:{'refused' if lv in rec.refused_names else 'unobserved'}"]
    gate = dict(gate, size_gate_crossings=crossings, excused_by_property=dict(excused))
    _STATE["gate"] = _json_safe(dict(gate, names=names))
    if names:
        partial = partial + [n for n in names if n not in partial]
        why = [f"big {lv}: {r}" for lv, r in (gate.get("reasons") or {}).items()]
        reason = "; ".join(x for x in ([reason] if reason else []) + why) or None
    return partial, reason


def _json_safe(obj):
    return json.loads(json.dumps(obj, default=str))


def lever_rows(mode: Optional[str], ablated=()) -> List[str]:
    """The LEVER lines of the memory levers, in the core's ONE grammar for them (opt_core.mem.record.AppliedRecord.lever_lines: `name=<lever>
    state=on|skipped|off … strategy=<canonical id>`; report.lever_lines prints them before the `memory` row, which is chunk_pair's settings as
    applied): every applied lever `on` with its label, scope, settings, sites and deferred-check verdict; every refused lever `skipped` with its
    precondition; every lever off by flag or dropped `off`; then — composed here from the same primitive (opt_core.report.lever_line) — the
    size-gated levers below their gate at this run's size (`skipped reason=below_gate … gate=n_token<=<threshold> n=<sized N|unsized>`: in the
    mode's set, not applied at this size — chunk_pair included, whose evidence above the gate is the report's `memory` row) and any other lever
    off by property (`off reason=property_gate gate=…`). `ablated` = the memory levers MODEL_OPT_LEVERS_OFF withheld
    (ablation.py; off in the line's selection): their row reads `off reason=ablated` in place of the record's `flag`.
    Outside the mode: nothing (the `memory` row states `off reason=not_in_mode:<mode>` for the family)."""
    rec, comp = _rec(), _STATE.get("composition")
    if mode != MODE:
        return []
    from . import ablation as A
    abl = {n for n in (ablated or ()) if n in TABLE}
    out: List[str] = []
    gated_off = set((comp or {}).get("gates", {}).get("size_gated_off") or ())           # the size-gated levers below their gate at this run's size (gates_for)
    named = set() if "chunk_pair" in gated_off else {"chunk_pair"}                     # chunk_pair applied (or withheld by flag): its evidence is the report's `memory` row; below its gate its row is composed here like every gated lever's
    if rec is not None:
        skip = {"chunk_pair"} | set(rec.drop) | abl                                    # the dropped BASE levers (hoist, sg) are the kit's own rows (report.lever_lines): one line per lever; an ablated lever's row is composed below
        out += [l for l in rec.lever_lines(TAG) if l.split(" name=", 1)[1].split(" ", 1)[0] not in skip]
        named |= set(rec.levers) | set(rec.refused_names) | set(rec.off_by_flag)
    for lever, row in TABLE.items():
        impl = row["impl"].split(" ", 1)[0]
        if lever in abl and lever != "chunk_pair":                                     # withheld by MODEL_OPT_LEVERS_OFF (chunk_pair's row is the report's `memory` row)
            out.append(CORE_R.lever_line(TAG, lever, "off", reason=A.REASON, impl=impl, origin=row["origin"], strategy=row["strategy"]))
            continue
        if lever in named or lever in abl:
            continue
        if rec is None or comp is None:
            out.append(CORE_R.lever_line(TAG, lever, "off", reason="not_armed", impl=impl, origin=row["origin"], strategy=row["strategy"]))
        elif lever in gated_off:                                                        # in the mode's set, not applied at this size: the core grammar's `skipped` with the gate word and the sized N (the kit's words for its base levers' size gates, report.lever_lines: `state=skipped reason=below_gate … gate=<word> n=<n>`)
            out.append(CORE_R.lever_line(TAG, lever, "skipped", ("gate", _gate_word(lever, comp)), ("n", _sized_word(comp)), reason="below_gate", impl=impl, origin=row["origin"], strategy=row["strategy"]))
        elif lever in comp["gates"]["off_by_property"]:
            out.append(CORE_R.lever_line(TAG, lever, "off", ("gate", _gate_word(lever, comp)), reason="property_gate", impl=impl, origin=row["origin"], strategy=row["strategy"]))
        else:
            out.append(CORE_R.lever_line(TAG, lever, "off", reason="not_applied", impl=impl, origin=row["origin"], strategy=row["strategy"]))
    return out


def _gate_word(lever: str, comp: dict) -> str:
    g = comp["gates"]
    thr = (g.get("gates") or {}).get(lever)
    if thr is not None:
        return f"n_token<={thr}"
    if lever == "trimul_torch":
        return f"n_token<={g['trimul_gate_tokens']}"
    if lever in ROWPAIR_SITELESS:
        return "n_gpu<=1"
    if lever == "expandable_segments":
        return "graphs_on"
    return "gate"


def _sized_word(comp: dict):
    n = ((comp or {}).get("size") or {}).get("n_token")
    return n if n is not None else "unsized"


def exit_lines() -> List[str]:
    """The census EXIT line of the mode (opt_core.mem.record.exit_line on the gate exit_join computed) — [] outside the mode or before the gate."""
    rec, gate = _rec(), _STATE.get("gate")
    if rec is None or gate is None:
        return []
    text = rec.exit_line(TAG, gate)
    return [text] if text else []


def report(rep: dict) -> Optional[dict]:
    """The `memory` record as the activation report holds it (None on a run without the memory mode) — the activation report's memory and the
    MEMORY line read it."""
    if (rep or {}).get("mode") != MODE:
        return None
    return rep.get("memory")


def attach_name() -> str:
    from . import stack                      # the wrap's names live with the wrap (stack.py); read at call time, not at import
    return f"{stack.RUNNER_MODULE}.{stack.RUNNER_CLASS}.__init__"


def undo() -> List[str]:
    """Restore the patched stock methods and the core's levers (tests)."""
    done = []
    for fn in reversed(_STATE.get("undo") or []):
        fn(); done.append("patch")
    rec = _rec()
    if rec is not None:
        from opt_core.mem import undo as mem_undo
        try:
            done += mem_undo(rec)
        except Exception as e:                                                  # noqa: BLE001
            if is_oom(e): raise                                   # an out-of-memory is the caller's to see: never rerouted (opt_core.oom)
            pass
    _STATE.update(record=None, ctx=None, unit=None, installed={}, undo=[], gate=None, excused={}, units_seen=0, memory=None)
    _STATE.pop("size_gate_crossings", None); _STATE.pop("stock_attention_preload", None)
    return done
