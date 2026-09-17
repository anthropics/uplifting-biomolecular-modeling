"""The design kit at the cookbook level — installed on the STOCK cookbook module from outside, after its import and before any model is
built, on a kit arm (``exact | fast | big``); ``off`` installs nothing and the stock module runs exactly as shipped.

The design kit's lever modules (``opt/forward/<design kit>/k/ef2_*.py``: kernels, CUDA graphs, the memory-planned checkpoint policy, the
loop-level re-plumbing, the state guard) attach to the fork's model objects and to the cookbook module's own functions. What ties them to
the cookbook's loop is here, as hooks on the imported module ``BD`` (``stock/src/cookbook/tutorials/binder_design.py``) — no cookbook
file is edited or copied:

  at install (``install``, before any model exists)
  1. ``BD.prepare_esmfold2_tensors``  the featurisation cache (exact): protein-only featurisation is a pure function of (sequences,
                                      msa=None), so the per-step CPU featurisation is memoised per designed sequence; the stock function
                                      computes every miss.
  2. ``BD.ESMFold2Design._enable_fast_kit(self, n_tokens)``  the kit-enable step (``enable`` below).
  3. ``BD.ESMFold2Design.design``     computes the complex size the loop will build and runs (2) before the stock ``design`` (a no-op when
                                      the design script enabled the kit for that size already); one state-guard check after it returns.
  at enable (``enable``, models loaded, complex size known; ONE composition per ``EF2_FAST_KIT`` value, ``COMPOSITION``)
  4. ``BD.fold_and_get_distogram`` / ``BD.compute_esmc_pseudoperplexity_nll`` / ``BD.build_gradient_mask``  the loop-level exact levers,
                                      innermost first: ``ef2_loop_prep`` (the fold's entry re-plumbed: one host transfer, the ESMC-6B feature
                                      pass launched before the CPU featurisation, hidden states memoised, the target's featurisation
                                      spliced), ``ef2_loop_pppl`` (the pseudo-perplexity body without host syncs), ``ef2_esmc_overlap`` (the
                                      pseudo-perplexity term launched on a side stream while the fold runs), then the state guard's per-step
                                      check as the outermost kit wrapper (every design fold after the first checks the global numeric state
                                      the previous step left). The design script wraps its fold observer over all of it afterwards.
  then, in this order (a helper rebinding made after a CUDA-graph capture would be inert inside the captured region, so every rebinding
  precedes every capture): the state guard's first snapshot; the process-wide levers (every mode: ``ef2_trimul`` cueq_tiles — tuned tiles
  for the stock cuEquivariance kernels, written into the library's in-process table, so every cuEquivariance triangle multiplication of the
  process runs them: ``exact``'s inversion models (the mode table pinned that backend; the grad-mode triangle multiplication per mode is
  COMPOSITION's ``trimul`` word: cueq_tiles | bmm2 = agk's K-A2 | fused = ef2_trimul's) and, on every kit arm, the hero critics' folds
  (written through every fold model on the backend, one write per pair width: ``exact`` the first inversion model and the critics,
  ``fast``/``big`` the critics — never an inversion model there, whose handle is the fused kernel's); ``fast``/``big``: ``ef2_fused_ln``; every mode: ``ef2_esmc_rope`` on each
  distinct ESMC trunk); per inversion model: the pair-stack chunk (``fast``/``big``: the fork's 64 — read on the no-grad confidence path
  only, the grad-mode kernels never chunk), ``ef2_autograd_kernels`` (the patched trunk forward that carries the checkpoint policy;
  ``fast``/``big`` also its K-D3 transition), ``fast``/``big``: ``ef2_trimul`` fused, skip-unused-confidence, ``ef2_lazy_structure``,
  ``fast``/``big``: ``ef2_bf16_confidence``; on every fold model (inversion models and hero critics): ``ef2_pairbias_attn`` hoist and
  ``ef2_sampler_graph``; then the memory plan ``ef2_bwd_ckpt`` (the checkpoint policy chosen from the memory free at that moment plus what
  the captures below will still take), the trunk fwd+bwd CUDA-graph pool ``ef2_stepgraph`` when the complex is at most ``POOL_MAX_TOKENS``
  tokens (every mode; the launch-bound regime, where replay pays — above it the pool's resident activations cost more recompute
  than the launches save: it steps aside by name), the ESMC-6B forward graph, over it the target-chain hoist ``ef2_esmc_hoist`` (the feature pass computed for the
  binder's rows only, the target's hidden states kept: a wrapper OVER the graph's — under a replay it would never run) and the
  pseudo-perplexity fwd+bwd graph (every mode). Prints THE activation line
  ``fast kit: <switch>, checkpoint policy <ck> for <n> tokens, trunk graph pool <w>, esmc hoist <on|stepped_aside(cc_unproven:sm_NN)|ablated>[,
  ablated <names>]`` and one planner line per inversion model (evidence.fastkit_lines; the LEVER census after the loop is evidence.collect_after's).

Ablation (``MODEL_OPT_LEVERS_OFF=<comma list of LEVER census names>``, resolved ONCE per process: ``resolve_levers_off`` in the launcher for the
ACTIVE line's ``ablated=`` word and the typo guard, and at ``install`` in the arm, whose environment carries the word as the kit arm's second
allowed variable): a named lever is not installed — it says so by name on the kit's line and its LEVER line reads ``state=ablated`` — and a
lever that only serves an ablated one goes with it, by name (``REQUIRES``). A name the arm does not compose is refused by name before anything
launches; the modes are unchanged (an ablated run is not the mode: the ACTIVE / EVIDENCE lines say ``ablated=``).

Module state lives on ``BD`` under the names the design kit uses (``FAST_KIT``, ``_D59_GUARD``, ``_FEATURE_CACHE``, ``_LEVERS_OFF``,
``_KIT_ASIDE``; ``app._fast_kit_tokens``), read back by evidence.py (F7, F10, the census) and by nothing else. ``EF2_FAST_KIT_D59`` /
``EF2_FAST_KIT_PPPL`` (evidence F6) are read with the kit's defaults; the launcher strips them from every arm and the environment proof
proves them absent."""
from __future__ import annotations

import functools
import hashlib
import inspect
import os
from typing import List, Optional

BIG_CHUNK_SIZE = 64                       # the fork's own default pair-stack chunk (triangle multiplication + pair transition), set on fast / big's inversion models (the no-grad confidence path reads it; lower peak memory at large complexes)
POOL_MAX_TOKENS = 256                       # the trunk fwd+bwd graph pool (ef2_stepgraph) is installed for complexes of at most this many tokens on a composition that carries it (exact, fast), stepped aside by name above; big carries none
TRUNK_POOL_SLOTS = 2                        # one captured segment per grad-enabled trunk pass of a design step (run_step: num_loops=1 -> 2 passes)
TRUNK_PASSES = 2                            # grad-enabled trunk passes per design step, the memory planner's unit
FEATURE_CACHE_MAX = 4096                    # entries; cleared whole when exceeded (one trajectory designs far fewer distinct sequences per process)
D59_VAR, PPPL_VAR = "EF2_FAST_KIT_D59", "EF2_FAST_KIT_PPPL"   # evidence F6: stripped and proven absent in every arm; defaults "assert" / "1"
HOOKS = ("prepare_esmfold2_tensors", "ESMFold2Design._enable_fast_kit", "ESMFold2Design.design",                       # at install
         "fold_and_get_distogram", "compute_esmc_pseudoperplexity_nll", "build_gradient_mask")                           # at enable (loop-level levers + the guard)
NAME = "design_kit"                         # the record's name on the PATCH line / run.json patches / opt_manifest.json deviations
SOURCE = "opt/ef2inv_opt/fastkit.py: the design kit installed on the stock cookbook module (featurisation cache, kit-enable step, loop-level levers, guard checked at every design step)"
KIT_SWITCHES = ("exact", "agk3", "big")   # the EF2_FAST_KIT values the design kit accepts = the modes' switches (modes.MODES[*].kit_switch: exact -> exact, fast -> agk3, big -> big; the kit's line prints the word)
CHUNK_PINNED = "pinned"                     # exact: the pair-stack chunk is the mode table's model switch (stock_design.apply_model_switches), not the kit's call
LEVERS_OFF_VAR = "MODEL_OPT_LEVERS_OFF"        # the ablation word: a comma list of LEVER census names (levers_of), resolved once per process (resolve_levers_off)

# The ONE composition per EF2_FAST_KIT value (modes.py renders the lever lists from these words; evidence.py checks the arm against them):
#   trimul           the grad-mode triangle multiplication: "cueq_tiles" = the stock cuEquivariance kernels with ef2_trimul's tile table (exact);
#                    "fused" = ef2_trimul's fused fwd + frozen-weight bwd (fast class); "bmm2" = agk's K-A2 fwd+bwd kernels (fast class; the
#                    alternative no mode selects: one word here selects it). The tile table itself is process-wide on EVERY
#                    kit arm (the hero critics fold on the cuequivariance backend): `cueq_tiles` is its own LEVER census name
#   transition       the grad-mode pair transition: None = stock's compiled module; "refround_lean" = agk's K-D3 (fast class; brings ef2_fused_ln, and — where K-D3's own forward serves — its W12 projection + SwiGLU as one kernel on a card with an entry, ef2_kd3_gemmswiglu: sm_80)
#   transition_fwd   the FORWARD kernel K-D3 lean runs: None = K-D3's own; "t16" = the carried one-kernel sm_90a transition (k/ef2_t16_transition.py,
#                    fast class, compute capability 9.0: engaged once per process before any capture; on any other card / stack it steps aside BY
#                    NAME and K-D3's forward serves — its LEVER line says which). LEVER census name ef2_t16_transition (the ablation word switches it).
#   chunk            the pair-stack chunk the kit sets on the inversion models (CHUNK_PINNED: left to the mode table's switch; fast / big: the
#                    fork's 64 — the fused kernel and K-D3 never read it under grad, so it acts on the no-grad confidence path only: less memory
#                    at the confidence steps of large complexes, the same digits)
#   ckpt             the memory plan's rule (ef2_bwd_ckpt.plan, CKPT_RULES): "budget" = FILL the card — keep the activations of as many pair
#                    blocks as the budget holds, checkpoint the rest (exact, fast); "floor" = the plan PINNED to its floor — policy `block`, every
#                    pair block of every grad pass checkpointed, kept 0, the plan's reason word `memory_floor` (big: stock's activation
#                    footprint; the estimate and the budget are still computed and printed). A planned choice either way (evidence F5 refuses a
#                    hand-forced policy, and refuses a plan whose rule is not the composition's)
#   pool             the trunk fwd+bwd CUDA-graph pool for complexes <= POOL_MAX_TOKENS, stepped aside by name above (exact, fast: below the gate
#                    replay pays and the resident activations are small; above it peak memory decides); False = no pool at any size (big: the pool's
#                    private memory holds the trunk's activations resident, CHANGES.md §big)
#   mode_off         the LEVER census names of the levers the composition switches OFF BY NAME for the memory each keeps resident (MODE_OFF_LEVERS:
#                    the CUDA-graph levers — private pools / static tensors resident — and the fold-pLM overlap — the pLM term's forward+backward working
#                    set alive through the trunk's backward instead of after it; CHANGES.md §big): () on exact / fast,
#                    big's tuple. enable() never calls them, the census prints `LEVER name=<lever> state=off … reason=mode:<mode>` for each, the
#                    ablation word refuses them as not composed
#   bf16_confidence  the confidence head under bf16 autocast on the inversion models' confidence steps (fast class)
#   nosave_fwd       the no-save first pass of the checkpointed pair blocks on the opt_core provider's forward-only triangle-multiplication row,
#                    by class word (ef2_trimul_nosave: "fast" = rows tx_sm90a > esm_v61 > esm_v5_fwd > v4 asked BY NAME, the first the stack serves; its LEVER line also prints what the provider's tier word resolves to here, tier_fast=<row>; None = torch's checkpoint, both
#                    passes on the grad-path kernels — the exact arm keeps the compiled block intact)
CKPT_RULES = ("budget", "floor")                  # COMPOSITION's ckpt word (ef2_bwd_ckpt.plan: rule budget | floor -> its record's reason word budget | memory_floor)
MODE_OFF_LEVERS = ("ef2_esmc_graph", "ef2_esmc_hoist", "ef2_pppl_graph", "ef2_sampler_graph", "ef2_pairbias_attn",   # the levers a composition may switch off by name (mode_off): the CUDA-graph levers and the pair-bias hoist (a private pool /
                   "ef2_esmc_overlap", "ef2_loop_pppl", "ef2_loop_prep", "ef2_lazy_structure", "ef2_fused_ln", "ef2_esmc_rope")   # static tensors resident each), the fold-pLM overlap (two working sets alive at once) and the loop / small levers; a set closed under REQUIRES' dependants (mode_off checks it). The kernels, the chunk, the plan and the trunk pool are composition words of their own, never named here
COMPOSITION = {
    "exact": {"trimul": "cueq_tiles", "transition": None, "transition_fwd": None, "chunk": CHUNK_PINNED, "ckpt": "budget", "pool": True, "mode_off": (), "bf16_confidence": False, "nosave_fwd": None},
    "agk3":  {"trimul": "fused", "transition": "refround_lean", "transition_fwd": "t16", "chunk": BIG_CHUNK_SIZE, "ckpt": "budget", "pool": True, "mode_off": (), "bf16_confidence": True, "nosave_fwd": "fast"},   # fast (modes.MODES["fast"].kit_switch = agk3)
}
BIG_MODE_OFF: tuple = ("ef2_esmc_overlap", "ef2_pairbias_attn", "ef2_sampler_graph", "ef2_esmc_graph", "ef2_esmc_hoist", "ef2_pppl_graph")   # the levers big switches off by name for the memory each keeps resident, in enable order (CHANGES.md §big)
COMPOSITION["big"] = {**COMPOSITION["agk3"], "ckpt": "floor", "pool": False, "mode_off": BIG_MODE_OFF}   # big = fast MINUS its memory-resident levers: the plan's fill (pinned to the floor instead), the trunk graph pool (none at any size), the levers named in mode_off
TRANSITION_FWD_KERNELS = (None, "t16")           # COMPOSITION's transition_fwd word: None = K-D3's own forward kernels; "t16" = the carried one-kernel sm_90a transition (k/ef2_t16_transition.py; LEVER census name ef2_t16_transition)
KD3_GEMMSWIGLU = "ef2_kd3_gemmswiglu"             # K-D3 lean's OWN forward (transition refround_lean), where it serves — no live out-only kernel on this card — runs its W12 projection + SwiGLU as one kernel on a card with an entry (k/ef2_kd3_gemmswiglu.py, sm_80); a LEVER census name of fast / big; never asked where t16 serves (sm_90)
TRIMUL_KERNELS = ("cueq_tiles", "bmm2", "fused")
# Ablation: a lever that only serves another goes with it, by name (the kit's own dependency rules: the sync-free pseudo-perplexity body reads
# the re-plumbed fold's memo, the overlap launches that body, the frozen-LayerNorm backward serves K-D3 only, and on fast / big the tile
# table reaches the process through a hero critic on the cuequivariance backend — the critic switches put it there).
REQUIRES = {"ef2_loop_pppl": ("ef2_loop_prep",), "ef2_esmc_overlap": ("ef2_loop_pppl",), "ef2_fused_ln": ("agk_transition",), "ef2_t16_transition": ("agk_transition",), "ef2_kd3_gemmswiglu": ("agk_transition",), "cueq_tiles": ("critic_switches",)}
CRITIC_LEVER = "critic_switches"            # the hero critics' two switches (modes.Mode.critic_chunk_size / critic_kernel_backend, applied by stock_design): a LEVER census name the kit-enable step does not install itself
COMPILE_LEVER = "compile"                   # the name of a KIT-ADDED torch.compile lever in the ablation word (`design --no-compile` is its alias). This kit adds none — the only
                                            # torch.compile of any mode is stock's own (the cookbook's COMPILE: the inversion models' MSA encoder and pair-update blocks), inherited
                                            # untouched — so the name is ACCEPTED and worded (compile_word), never refused, and switches nothing off


def compile_word(asked: bool = False) -> str:
    """The ``compile=`` word of the ACTIVE line: ``stock`` (stock's own torch.compile, inherited; no kit-added compile lever exists) | ``stock:not_kit_added``
    when the opt-out was asked (``--no-compile`` / the ablation word naming ``compile``): accepted, nothing to switch off."""
    return "stock:not_kit_added" if asked else "stock"


class LeversOffError(ValueError):
    """``MODEL_OPT_LEVERS_OFF`` names a lever that is not a LEVER census name, or one the arm does not compose (refused by name, exit 3)."""


def levers_of(mode_or_kit) -> List[str]:
    """The LEVER census names the arm composes, in enable order — the ablation word's vocabulary for that arm. ``mode_or_kit``: a modes.Mode
    (its ``kit_switch`` + critic-scope switches) or a bare EF2_FAST_KIT value (then the critic switches, the mode table's, are not among them)."""
    kit = getattr(mode_or_kit, "kit_switch", mode_or_kit)
    if kit not in COMPOSITION:
        return []
    c = COMPOSITION[kit]
    names = ["ef2_loop_prep", "ef2_loop_pppl", "ef2_esmc_overlap"]
    names += ["ef2_fused_ln"] if c["transition"] else []
    names += ["cueq_tiles", "ef2_esmc_rope"]
    names += ["chunk"] if c["chunk"] != CHUNK_PINNED else []
    names += ["agk_transition"] if c["transition"] else []
    names += ["ef2_t16_transition"] if c.get("transition_fwd") else []
    names += [KD3_GEMMSWIGLU] if c["transition"] == "refround_lean" else []
    names += ["trimul"] if c["trimul"] in ("fused", "bmm2") else []
    names += ["ef2_trimul_nosave"] if c["nosave_fwd"] else []
    names += ["skip_unused_confidence", "ef2_lazy_structure"]
    names += ["ef2_bf16_confidence"] if c["bf16_confidence"] else []
    names += ["ef2_pairbias_attn", "ef2_sampler_graph", "ef2_bwd_ckpt"]
    names += ["ef2_stepgraph"] if c["pool"] else []
    names += ["ef2_esmc_graph", "ef2_esmc_hoist", "ef2_pppl_graph", "featurisation_cache"]
    if any(getattr(mode_or_kit, a, "shipped") != "shipped" for a in ("critic_chunk_size", "critic_kernel_backend")):
        names += [CRITIC_LEVER]
    return [n for n in names if n not in c["mode_off"]]                      # a lever the composition switches off by name is not among its levers (big: mode_off) — the census still prints its line, state=off reason=mode:<mode>


def mode_off(kit: str) -> tuple:
    """The levers the composition switches OFF BY NAME for the memory each keeps resident (COMPOSITION mode_off, a subset of MODE_OFF_LEVERS;
    empty on exact / fast)."""
    names = tuple(COMPOSITION[kit]["mode_off"]) if kit in COMPOSITION else ()
    unknown = [n for n in names if n not in MODE_OFF_LEVERS]
    if unknown:
        raise ValueError(f"fastkit.COMPOSITION[{kit!r}]['mode_off'] names {unknown}, not levers a composition may switch off ({MODE_OFF_LEVERS})")
    orphans = [n for n, needs in REQUIRES.items() if n not in names and any(d in names for d in needs)]
    if orphans:                                                              # a lever whose prerequisite the composition switches off must be switched off with it (REQUIRES: overlap <- loop_pppl <- loop_prep)
        raise ValueError(f"fastkit.COMPOSITION[{kit!r}]['mode_off'] switches off a prerequisite of {orphans} but not {orphans} (REQUIRES {REQUIRES})")
    return names


def resolve_levers_off(value: Optional[str], mode_or_kit) -> dict:
    """``MODEL_OPT_LEVERS_OFF`` resolved once for the arm: ``{"asked": [names as given], "off": [the ablated set in enable order: the asked names plus
    every lever that only serves one of them (REQUIRES)], "chained": {name: the ablated lever it serves}}``. An empty / absent word ablates nothing.
    A name that is not a LEVER census name of any arm (a typo) or one this arm does not compose raises LeversOffError naming it and the arm's levers.
    ``compile`` (COMPILE_LEVER) is accepted on every arm and switches nothing off (``"compile": compile_word(True)`` in the result; no kit-added compile lever exists)."""
    asked = [w.strip() for w in (value or "").split(",") if w.strip()]
    composed = levers_of(mode_or_kit)
    out = {"asked": asked, "off": [], "chained": {}}
    if COMPILE_LEVER in asked:
        asked[:] = [n for n in asked if n != COMPILE_LEVER]; out["compile"] = compile_word(True)
    if not asked:
        return out
    vocabulary = set().union(*(levers_of(k) for k in KIT_SWITCHES)) | {CRITIC_LEVER}
    kit = getattr(mode_or_kit, "kit_switch", mode_or_kit)
    unknown = [n for n in asked if n not in vocabulary]
    if unknown:
        raise LeversOffError(f"{LEVERS_OFF_VAR}: unknown lever name(s) {','.join(unknown)} — the names are the LEVER census names; EF2_FAST_KIT={kit} composes {','.join(composed)}")
    foreign = [n for n in asked if n not in composed]
    if foreign:
        raise LeversOffError(f"{LEVERS_OFF_VAR}: EF2_FAST_KIT={kit} does not compose {','.join(foreign)}; its levers are {','.join(composed)}")
    off = set(asked); grew = True
    while grew:                                                              # the closure over REQUIRES (a chain: overlap <- loop_pppl <- loop_prep)
        grew = False
        for name, needs in REQUIRES.items():
            if name in composed and name not in off and any(d in off for d in needs):
                off.add(name); out["chained"][name] = next(d for d in needs if d in off); grew = True
    out["off"] = [n for n in composed if n in off]
    return out


CUEQ_BACKEND = "cuequivariance"               # the fork's backend word the cuEquivariance tile table serves (modes.STOCK_KERNEL_BACKEND)


def user_switches_of(BD) -> dict:
    """The user's pair-stack switches over the mode's value, as install() recorded them on the module (``{}`` when none / before install)."""
    return dict(getattr(BD, "_USER_SWITCHES", None) or {})


def _switch_word(v, key: str = "chunk_size") -> str:
    """The record spelling of a user switch value (settings.user_switch_word: chunk none|N, backend fused|cuequivariance|None)."""
    return ("none" if key == "chunk_size" else "None") if v is None else str(v)


def ablated(BD, name: str) -> bool:
    """True when the arm's ablation word switched ``name`` off (``BD._LEVERS_OFF``, set by install)."""
    return name in (getattr(BD, "_LEVERS_OFF", None) or ())


def _arguments(sig: inspect.Signature, args: tuple, kw: dict) -> dict:
    """The call's arguments by parameter name, defaults applied, a ``**kwargs`` catch-all flattened into the map."""
    b = sig.bind(*args, **kw); b.apply_defaults()
    out = dict(b.arguments)
    for prm in sig.parameters.values():
        if prm.kind is inspect.Parameter.VAR_KEYWORD:
            out.update(out.pop(prm.name, {}) or {})
    return out


def is_design_fold(kw: dict) -> bool:
    """The loop's per-step fold (``run_step``: num_loops=1, no confidence) as opposed to a hero-critic fold (confidence, num_loops=3) —
    the rule stock_design's fold observer classifies calls by."""
    return not (bool(kw.get("calculate_confidence", False)) and kw.get("num_loops", 0) == 3)


def complex_tokens(BD, *, target_name: str, target_sequence: Optional[str], binder_name: Optional[str], binder_sequence: Optional[str], seed: int) -> int:
    """The complex size ``design_binder`` builds: the target's residues (a built-in name's sequence, else the given one; chain breaks ``|``
    dropped) + the binder's (the given sequence's length, else the cookbook's own seed-determined prompt draw for ``binder_name``)."""
    tseq = target_sequence if target_sequence is not None else BD.TARGET_SEQUENCES[target_name]
    blen = len(binder_sequence) if binder_sequence is not None else len(BD.BINDER_PROMPT_FACTORIES[binder_name].sample(seed=seed))   # the same deterministic draw design_binder makes
    return len(tseq.replace("|", "")) + blen


def fold_models(app, labelled: bool = False) -> List[object]:
    """Every distinct ESMFold2 model the loaded app holds (the inversion models AND the critics): the values of each dict attribute whose
    members carry the stock API's ``set_kernel_backend`` / ``set_chunk_size`` — the rule stock_design.model_handles labels them by.
    ``labelled``: ``(<attr>.<key>, model)`` pairs instead (the census words)."""
    out, seen = [], set()
    for attr in sorted(vars(app)) if app is not None else []:
        d = getattr(app, attr)
        if not isinstance(d, dict) or not d or not all(hasattr(m, "set_kernel_backend") and hasattr(m, "set_chunk_size") for m in d.values()):
            continue
        for name in sorted(d):
            if id(d[name]) not in seen:
                seen.add(id(d[name])); out.append((f"{attr}.{name}", d[name]) if labelled else d[name])
    return out


def esmc_trunks(app) -> List[object]:
    """Every distinct ESMC trunk the app holds: each fold model's ``_esmc`` and the masked-LM the loop scores with (``app.esmc_model``)."""
    out, seen = [], set()
    for holder in fold_models(app) + [getattr(app, "esmc_model", None)]:
        e = getattr(holder, "_esmc", None) or getattr(holder, "esmc", None)
        if e is not None and id(e) not in seen:
            seen.add(id(e)); out.append(holder)
    return out


def trunk_pool_wanted(kit: str, n_tokens: int) -> bool:
    """The size rule of the trunk graph pool: the mode carries it AND the complex is at most POOL_MAX_TOKENS tokens."""
    return bool(COMPOSITION[kit]["pool"]) and int(n_tokens) <= POOL_MAX_TOKENS


def pool_word(kit: str, n_tokens: int, is_ablated: bool = False) -> str:
    """The kit line's trunk-graph-pool word: ``2 slots`` | ``none above 256 tokens`` (a composition that carries the pool, by the size rule) |
    ``none`` (a composition that carries no pool: big) | ``none (ablated)``."""
    if is_ablated and COMPOSITION[kit]["pool"]:
        return "none (ablated)"
    if trunk_pool_wanted(kit, n_tokens):
        return f"{TRUNK_POOL_SLOTS} slots"
    return "none" if not COMPOSITION[kit]["pool"] else f"none above {POOL_MAX_TOKENS} tokens"


def ckpt_floor(kit: str) -> bool:
    """True when the composition pins the memory plan to its floor (ckpt word ``floor``: big) — ef2_bwd_ckpt.enable(floor=True)."""
    rule = COMPOSITION[kit]["ckpt"]
    if rule not in CKPT_RULES:
        raise ValueError(f"fastkit.COMPOSITION[{kit!r}]['ckpt'] = {rule!r} is not one of {CKPT_RULES}")
    return rule == "floor"


def _enable_cueq_tiles(self, models: List[object], *, BD, kit: str) -> None:
    """The process-wide tile table of the stock cuEquivariance triangle-multiplication kernels (``ef2_trimul`` cueq_tiles: entries in the
    library's in-process table, read per call by every model of the process; the entries are keyed by the model's pair width), installed through
    every fold model that runs that backend and holds no other handle: ``exact`` — the first inversion model (the mode table pinned the backend; a
    table that cannot be written raises, the arm did not enable) and the hero critics; ``fast`` / ``big`` — the hero critics (the mode table's
    critic scope put them on the backend; an inversion model's handle is the fused kernel's, so the table never goes there). A backend the
    operator overrode leaves no fold model on cuequivariance: the lever steps aside BY NAME (``BD._KIT_ASIDE``; the census words it, evidence decides)."""
    import ef2_trimul
    holders = []
    ub = user_switches_of(BD).get("kernel_backend", "shipped")
    if ub != "shipped" and ub != CUEQ_BACKEND:                               # the user's --kernel-backend put EVERY model on another backend (stock's setter at load): no triangle multiplication of this process runs the cuEquivariance kernels, the table has nothing to serve — aside by name, the arm runs
        BD._KIT_ASIDE["cueq_tiles"] = f"user_kernel_backend:{_switch_word(ub, 'kernel_backend')}"
        BD.logger.info(f"fast kit: cueq_tiles steps aside — the user's --kernel-backend {_switch_word(ub, 'kernel_backend')} is every model's (no model runs the cuequivariance backend)")
        return
    if COMPOSITION[kit]["trimul"] == "cueq_tiles":
        if models[0].__dict__.get("_ef2_trimul_handle") is None:
            ef2_trimul.enable(models[0], variant="cueq_tiles")              # raises by name: the mode's own backend is cuequivariance (applied at load), so a table that cannot be written means the arm did not enable
        holders.append(models[0])
    reason = "no hero critic model is loaded"
    for m in (m for m in fold_models(self) if all(m is not im for im in models)):   # every hero critic on the backend: the entries are keyed by the model's pair width, so each distinct width is written once
        h = m.__dict__.get("_ef2_trimul_handle")
        if h is not None:
            holders += [m] if getattr(h, "variant", None) == "cueq_tiles" else []
            continue
        try:
            ef2_trimul.enable(m, variant="cueq_tiles"); holders.append(m)
        except RuntimeError as e:                                             # ef2_trimul's own words: no PairUpdateBlock of this model runs the cuequivariance backend / the library is not importable
            reason = str(e)
    if not holders:
        BD._KIT_ASIDE["cueq_tiles"] = reason
        BD.logger.info(f"fast kit: cueq_tiles steps aside — {reason}")


def _install_fold_guard(BD) -> None:
    """Hook 4's outermost kit wrapper: the state guard checked at every design fold after the first (idempotent per module object)."""
    if getattr(BD.fold_and_get_distogram, "_ef2_fold_guard", False):
        return
    inner = BD.fold_and_get_distogram

    @functools.wraps(inner)
    def fold_and_get_distogram(model, *args, **kw):
        if BD._D59_GUARD is not None and is_design_fold(kw):
            if BD._design_folds > 0:
                BD._D59_GUARD.check(f"end of design step {BD._design_folds - 1}")
            BD._design_folds += 1
        return inner(model, *args, **kw)
    fold_and_get_distogram._ef2_fold_guard = True
    BD.fold_and_get_distogram = fold_and_get_distogram


def _lm_pools_resident(model, esmc_model) -> bool:
    """True when the ESMC-forward and pseudo-perplexity graphs are already captured (their pools exist on the device now)."""
    import ef2_pppl_graph
    eeg = getattr(getattr(model, "_esmc", None), "_eeg", None)
    st = ef2_pppl_graph.stats(esmc_model)
    return bool(eeg is not None and getattr(eeg, "entries", None)) and bool(st and st.get("entries"))


def _hoist_resident(model) -> bool:
    """True when the target-chain hoist already holds its hidden states / reduced graph on the device (a second enable in one process)."""
    import ef2_esmc_hoist
    st = ef2_esmc_hoist.stats(model)
    return bool(st and st.get("entries"))


def hoist_word() -> str:
    """The target-chain hoist's word for the activation line: `on` on a compute capability its reduced pass is proven bitwise on
    (ef2_esmc_hoist.PROVEN_CC), else `stepped_aside(cc_unproven:sm_NN)` — the lever does not install there and says so before the loop."""
    import ef2_esmc_hoist
    proven, sm = ef2_esmc_hoist.cc_route()
    return "on" if proven or sm is None else f"stepped_aside(cc_unproven:{sm})"


def _pool_extra_bytes(n_models: int, n_tokens: int, kernels: str) -> float:
    """Resident memory the trunk graph pools add beyond what the planner's model counts: every captured slot keeps its pass's working set
    (the planner counts one transient), i.e. (models x passes - 1) further transients (ef2_bwd_ckpt's own constants)."""
    import ef2_bwd_ckpt as bc
    transient = bc.TRANSIENT_KEPT_EQUIV * bc.KEPT_BYTES_PER_POS[kernels] * float(n_tokens) ** 2 + bc.TRANSIENT_CONST_BYTES
    return float(n_models * TRUNK_PASSES - 1) * transient


def enable(self, n_tokens: int, *, BD, kit: str) -> None:
    """``ESMFold2Design._enable_fast_kit``: the design kit's ONE composition for ``kit`` on the loaded models for a complex of ``n_tokens``
    tokens, in the order the module docstring states. Idempotent: a second call for the same size re-installs nothing (one named line); a new
    size in the same process first returns the previous size's shape-bound graph pools and memoised hidden states, then re-plans."""
    import ef2_autograd_kernels as agk, ef2_state_guard, ef2_trimul, ef2_fused_ln, ef2_bwd_ckpt, ef2_stepgraph, ef2_esmc_graph, ef2_pppl_graph, ef2_kd3_gemmswiglu, \
        ef2_esmc_rope, ef2_esmc_overlap, ef2_loop_prep, ef2_loop_pppl, ef2_lazy_structure, ef2_pairbias_attn, ef2_sampler_graph, ef2_bf16_confidence, ef2_esmc_hoist, \
        ef2_t16_transition, ef2_trimul_nosave
    comp = COMPOSITION[kit]
    off = tuple(getattr(BD, "_LEVERS_OFF", None) or ())                       # the ablation word, resolved once at install (levers_of names, closure over REQUIRES)
    moff = mode_off(kit)                                                     # the levers the composition itself switches off by name (big's mode_off): never called below, worded on the kit line

    def on(name: str) -> bool:
        return name not in off and name not in moff
    prev = getattr(self, "_fast_kit_tokens", None)
    if prev == n_tokens:
        BD.logger.info(f"fast kit: {kit} already enabled for {n_tokens} tokens (nothing re-installed)")
        return
    models = list(self.inversion_models.values())
    if prev is not None:                              # a new complex size: the previous size's graph pools and memos are returned before re-planning
        for m in models:
            ef2_stepgraph.disable(m)
        ef2_esmc_graph.release(models[0]); ef2_esmc_hoist.release(models[0]); ef2_pppl_graph.release(self.esmc_model); ef2_loop_prep.release(BD)
    self._fast_kit_tokens = n_tokens
    if BD._D59_GUARD is None:                         # the state guard: snapshot process-global numeric state BEFORE the kit touches anything
        BD._D59_GUARD = ef2_state_guard.StateGuard(mode=os.environ.get(D59_VAR, "assert"), label="before kit enable (models loaded)")
    # hook 4: the loop-level exact levers on the cookbook module, innermost first, the guard's per-step check outermost (all idempotent)
    if on("ef2_loop_prep"):
        ef2_loop_prep.enable(BD)
    if on("ef2_loop_pppl"):
        ef2_loop_pppl.enable(BD)
    if on("ef2_esmc_overlap"):
        ef2_esmc_overlap.enable(BD, esmc_model=self.esmc_model)
    _install_fold_guard(BD)
    # process-wide levers, before any CUDA-graph capture
    transition = comp["transition"] if on("agk_transition") else None
    trimul = comp["trimul"] if on("trimul") else None   # `trimul` (a census name on fast / big only) is the fast-class kernel; ablated: the loader's backend under grad, as the cookbook runs it
    if transition and on("ef2_fused_ln"):
        ef2_fused_ln.enable()                         # agk's frozen-LayerNorm backward helper (row-major operands: tensor-equal to the vendored kernel)
    transition_fwd = comp["transition_fwd"] if (transition and on("ef2_t16_transition")) else None   # K-D3's forward kernel word; ablated (or agk_transition ablated: REQUIRES): K-D3's own forward
    if transition_fwd == "t16":
        t16 = ef2_t16_transition.engage()             # the carried sm_90a transition forward: cubin loaded + canary once per process, before any capture; a card / stack it cannot serve is recorded (never raised), K-D3's forward serves
        if t16["state"] != "on":
            BD._KIT_ASIDE["ef2_t16_transition"] = f"{t16['reason']}"
        BD.logger.info(f"fast kit: ef2_t16_transition {t16['state']}" + (f" ({t16['reason']}: {t16['reason_text']}) — K-D3's own forward serves the pair transition" if t16["state"] != "on"
                                                                        else f" (cubin {t16['kernel'].get('cubin')}, canary {t16['kernel'].get('canary')})"))
    if transition == "refround_lean" and on(KD3_GEMMSWIGLU):        # K-D3 lean's OWN forward — where it serves — with the W12 projection + SwiGLU as one kernel: process-wide hook in agk, bound once before any capture on a card with an entry (sm_80); a card without one is recorded (never raised); never asked where the t16 kernel runs that forward (sm_90)
        if transition_fwd == "t16" and ef2_t16_transition.live():
            BD._KIT_ASIDE[KD3_GEMMSWIGLU] = "t16_serves"
            BD.logger.info(f"fast kit: {KD3_GEMMSWIGLU} not asked on this card — K-D3 lean's forward runs on the t16 kernel")
        else:
            gsw = ef2_kd3_gemmswiglu.engage()
            if gsw["state"] != "on":
                BD._KIT_ASIDE[KD3_GEMMSWIGLU] = f"{gsw['reason']}"
            BD.logger.info(f"fast kit: {KD3_GEMMSWIGLU} {gsw['state']}" + (f" ({gsw['reason']}: {gsw['reason_text']}) — K-D3's cuBLAS W12 projection + SwiGLU kernel serve" if gsw["state"] != "on"
                                                                          else f" (tile {gsw['tile']} on {gsw['cc']}) — K-D3 lean's W12 projection + SwiGLU in one kernel"))
    if on("cueq_tiles"):
        _enable_cueq_tiles(self, models, BD=BD, kit=kit)   # the tile table of the stock cuEquivariance kernels, process-wide: exact's inversion models, every arm's hero critics
    if on("ef2_esmc_rope"):
        for holder in esmc_trunks(self):
            ef2_esmc_rope.enable(holder)              # the exact fused rotary on every distinct ESMC trunk (the fold's feature pass and the pseudo-perplexity passes)
    # per inversion model: chunk, the patched trunk forward (checkpoint policy carrier) + kernels, the usage-level exact levers
    user_chunk = "chunk_size" in user_switches_of(BD)                       # the user's --chunk-size is on every model already (stock's setter at load): the composition's chunk yields to it BY NAME
    if user_chunk and comp["chunk"] != CHUNK_PINNED and on("chunk"):
        BD._KIT_ASIDE["chunk"] = f"user_chunk_size:{_switch_word(user_switches_of(BD)['chunk_size'])}"
        BD.logger.info(f"fast kit: chunk {comp['chunk']} yields to the user's --chunk-size {_switch_word(user_switches_of(BD)['chunk_size'])} (every model's, set at load)")
    for m in models:
        if comp["chunk"] != CHUNK_PINNED and not user_chunk:
            m.set_chunk_size(comp["chunk"] if on("chunk") else None)   # fast / big: the fork's 64 (read on the no-grad confidence path only); ablated: the pair stack unchunked; a user --chunk-size: no call (above)
        agk.enable(m, trimul=("bmm2" if trimul == "bmm2" else None), transition=transition, checkpoint="block", transition_fwd=transition_fwd)   # the patched trunk (policy carrier) + the agk kernels asked (K-D3's forward on the t16 kernel where live)
        if trimul == "fused" and m.__dict__.get("_ef2_trimul_handle") is None:
            ef2_trimul.enable(m, variant="fused")     # the fused triangle multiplication fwd + frozen-weight bwd on the block's own submodules (agk trimul=None calls them)
        if comp["nosave_fwd"] and on("ef2_trimul_nosave") and m.__dict__.get("_ef2_nosave_handle") is None:
            BD.logger.info(ef2_trimul_nosave.describe(ef2_trimul_nosave.enable(m, word=comp["nosave_fwd"], n_tokens=n_tokens)))   # fast class: the checkpointed blocks' no-save first pass on the provider's forward-only row (outermost on the tri-mul modules, over ef2_trimul's); steps aside by name (core absent / sm_80 / no row)
        if on("skip_unused_confidence"):
            agk.enable_skip_unused_confidence(m)      # exact: upstream ignores calculate_confidence=False
        if on("ef2_lazy_structure"):
            ef2_lazy_structure.enable(m)              # exact: the unused one-step structure sample of a design fold is deferred until read
        if comp["bf16_confidence"] and on("ef2_bf16_confidence"):
            ef2_bf16_confidence.enable(m)             # fast class: the confidence head under bf16 autocast on confidence steps
    for m in fold_models(self):                       # the inversion models AND the hero critics share the diffusion sampler's code path
        if on("ef2_pairbias_attn"):
            ef2_pairbias_attn.enable(m, variant="hoist")   # exact: the pair bias projected once per sample instead of once per denoise step
        if on("ef2_sampler_graph"):
            ef2_sampler_graph.enable(m)               # exact: one denoise step captured as a CUDA graph, replayed per step
    # the memory plan, then the captures it budgets for
    pool = trunk_pool_wanted(kit, n_tokens) and on("ef2_stepgraph")
    lm_graphs = on("ef2_esmc_graph") or (on("ef2_pppl_graph") and os.environ.get(PPPL_VAR, "1") == "1")
    kern = ef2_bwd_ckpt.kernels_of(models[0])
    extra = (ef2_bwd_ckpt.LM_GRAPH_POOL_BYTES if lm_graphs and not _lm_pools_resident(models[0], self.esmc_model) else 0.0) \
        + (_pool_extra_bytes(len(models), n_tokens, kern) if pool else 0.0) \
        + (ef2_esmc_hoist.POOL_BYTES if on("ef2_esmc_hoist") and ef2_esmc_hoist.cc_route()[0] and not _hoist_resident(models[0]) else 0.0) \
        + ef2_trimul_nosave.extra_bytes(models[0], n_tokens)   # the no-save checkpoint holds one recomputed block output through its backward (0 when the lever is off / stepped aside); nothing reserved on a card where the hoist steps aside, or when it is ablated
    if on("ef2_bwd_ckpt"):
        for m in models:
            ef2_bwd_ckpt.enable(m, n_tokens, num_passes=TRUNK_PASSES, copies=(len(models) if pool else 1), extra_reserved_bytes=extra, floor=ckpt_floor(kit), verbose=False)   # floor: the composition's rule (big pins the plan to policy block, reason memory_floor; exact / fast fill the budget)
            BD.logger.info(ef2_bwd_ckpt.describe(m))
    else:
        BD.logger.info("fast kit: ef2_bwd_ckpt ablated — no memory plan; the trunk keeps agk's carried policy `block` (every pair block checkpointed: the leanest, the most recompute)")
    if pool:
        for m in models:
            ef2_stepgraph.enable(m, n_slots=TRUNK_POOL_SLOTS)   # captured lazily at the first design step, one segment per trunk pass; the policy above is what it captures
    if on("ef2_esmc_graph"):
        ef2_esmc_graph.enable(models[0])              # the shared ESMC-6B forward (idempotent)
    if on("ef2_esmc_hoist"):
        ef2_esmc_hoist.enable(models[0])              # exact: the target chain's hidden states hoisted — the feature pass for the binder's rows only; OVER the graph wrapper (idempotent); on a compute capability outside its PROVEN_CC it steps aside by name (nothing installed)
    if os.environ.get(PPPL_VAR, "1") == "1" and on("ef2_pppl_graph"):
        ef2_pppl_graph.enable(self.esmc_model)        # the pseudo-perplexity fwd+bwd through ESMC-6B (idempotent)
    ck = getattr(models[0], "_bwd_ckpt", None)
    ck = ck.get("policy", "block") if ck else "block"
    BD.logger.info(f"fast kit: {kit}, checkpoint policy {ck} for {n_tokens} tokens, trunk graph pool {pool_word(kit, n_tokens, 'ef2_stepgraph' in off)}, esmc hoist {hoist_word() if on('ef2_esmc_hoist') else ('off' if 'ef2_esmc_hoist' in moff else 'ablated')}"
                   + (f", off by mode {','.join(moff)}" if moff else "") + (f", ablated {','.join(off)}" if off else ""))


def install(BD, kit, environ=None, user_switches=None) -> dict:
    """Install hooks 1-3 on the imported stock cookbook module for the arm ``kit`` — a modes.Mode (its ``kit_switch``; its critic switches make
    ``critic_switches`` one of the arm's census names) or a bare EF2_FAST_KIT value (a kit arm; never called on ``off``). The ablation word is
    resolved ONCE here from ``environ`` (default os.environ; LeversOffError by name on a word the arm cannot honour) and kept on the module as
    ``BD._LEVERS_OFF``; the pair-stack switches THE USER set over the mode's value (settings.user_switches: ``{"chunk_size": none|N,
    "kernel_backend": …}``, applied to every model at load by the design script) are kept as ``BD._USER_SWITCHES`` — the enable step reads them
    and a lever whose contract needs another value steps aside BY NAME (``BD._KIT_ASIDE``), never a refusal. Idempotent per module object;
    returns the record the arm prints and stores (name, applied, kit, hooks, levers_off, user_switches, the enable step's code digest)."""
    mode, kit = (kit, kit.kit_switch) if hasattr(kit, "kit_switch") else (kit, kit)
    if kit not in KIT_SWITCHES:
        raise ValueError(f"fastkit.install: kit {kit!r} is not one of {KIT_SWITCHES}")
    levers_off = resolve_levers_off((os.environ if environ is None else environ).get(LEVERS_OFF_VAR), mode)
    user_switches = {k: v for k, v in dict(user_switches or {}).items() if k in ("chunk_size", "kernel_backend")}
    rec = {"name": NAME, "applied": True, "kit": kit, "hooks": list(HOOKS), "module": BD.__name__, "source": SOURCE, "levers_off": list(levers_off["off"]),
           "user_switches": dict(user_switches), "function_sha256": hashlib.sha256(enable.__code__.co_code).hexdigest()}
    if getattr(BD, "FAST_KIT", None) is not None:                           # installed already on this module object
        if BD.FAST_KIT != kit:
            raise RuntimeError(f"fastkit.install: the module carries kit {BD.FAST_KIT!r}, asked {kit!r}")
        return rec
    BD.FAST_KIT = kit; BD._D59_GUARD = None; BD._FEATURE_CACHE = {}; BD._design_folds = 0
    BD._LEVERS_OFF = tuple(levers_off["off"]); BD._KIT_ASIDE = {}          # the ablated census names (enable order); levers that stepped aside by name at enable {name: reason}
    BD._USER_SWITCHES = user_switches                                       # the user's pair-stack switches over the mode's value (every model carries them already): read at enable
    cls = BD.ESMFold2Design

    # (1) the featurisation cache around the stock function
    orig_prep = BD.prepare_esmfold2_tensors
    prep_sig = inspect.signature(orig_prep)

    @functools.wraps(orig_prep)
    def prepare_esmfold2_tensors(*args, **kw):
        p = _arguments(prep_sig, args, kw)
        inp, max_atoms = p["input"], p.get("max_atoms")
        cacheable = all(getattr(p, "msa", None) is None and not getattr(p, "modifications", None) and isinstance(getattr(p, "sequence", None), str) for p in inp.sequences)
        if not cacheable:
            return orig_prep(*args, **kw)
        key = (tuple((str(p.id), type(p).__name__, p.sequence) for p in inp.sequences), max_atoms)
        hit = BD._FEATURE_CACHE.get(key)
        if hit is None:
            hit = orig_prep(*args, **kw)
            if len(BD._FEATURE_CACHE) > FEATURE_CACHE_MAX:
                BD._FEATURE_CACHE.clear()
            BD._FEATURE_CACHE[key] = hit
        return {k: (v.clone() if BD.torch.is_tensor(v) else v) for k, v in hit.items()}
    if "featurisation_cache" not in BD._LEVERS_OFF:
        BD.prepare_esmfold2_tensors = prepare_esmfold2_tensors
    else:
        BD._FEATURE_CACHE = None                                              # ablated: the stock function computes every call (the census reads state=ablated)

    # (2) the kit-enable step as the app class's method (the design script calls it before its fold observer; the design wrapper's call is then the named no-op)
    def _enable_fast_kit(self, n_tokens: int) -> None:
        enable(self, n_tokens, BD=BD, kit=kit)
    cls._enable_fast_kit = _enable_fast_kit

    # (3) design(): enable for the complex size before the stock loop; one guard check after it
    orig_design = cls.design
    design_sig = inspect.signature(orig_design)

    @functools.wraps(orig_design)
    def design(self, *args, **kw):
        p = _arguments(design_sig, (self, *args), kw)
        self._enable_fast_kit(complex_tokens(BD, target_name=p["target_name"], target_sequence=p.get("target_sequence"), binder_name=p.get("binder_name"),
                                             binder_sequence=p.get("binder_sequence"), seed=p["seed"]))
        BD._design_folds = 0
        out = orig_design(self, *args, **kw)
        if BD._D59_GUARD is not None:
            BD._D59_GUARD.check(f"end of design step {max(BD._design_folds - 1, 0)}")
        return out
    cls.design = design
    return rec


def record_off() -> dict:
    """The record's shape on the stock arm (nothing installed) — opt_manifest.json deviations list the same keys on every arm."""
    return {"name": NAME, "applied": False, "kit": None, "hooks": [], "module": None, "source": SOURCE, "levers_off": [], "user_switches": {}}
