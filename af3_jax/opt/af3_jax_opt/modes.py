"""The mode table — the one place a mode is defined (``MODES``) — and how a mode resolves to a script, a launcher, a flag row and a lever
environment. Switch values are read out of the kit files themselves (the base kit's rows file, the FlashPairformer package's switches);
the one table held here is the Pallas add-on's Tier-1 lever set (``PALLAS_LEVERS``). ``off`` = the stock
``run_alphafold.py``, no launcher; ``exact`` = the base kit's script plus the Pallas add-on's Tier-1 levers, bitwise vs stock inside one
autotune class; ``fast`` (``DEFAULT_MODE``) = the same script through the FlashPairformer add-on plus the diffusion hoist, not bitwise;
``big`` = the memory mode, folding inputs the other modes cannot hold on one card, or sharding across ``--n_gpu`` cards.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Sequence

MODES = ("off", "exact", "fast", "big")
DEFAULT_MODE = "fast"                                                  # the package default (README §Modes); the env switch unset = off
STOCK_SCRIPT, FAST_SCRIPT = "run_alphafold.py", "run_alphafold_fast.py"
PATCHED_SCRIPT_RELPATH = os.path.join("patches", "patched_files", "run_alphafold.py")   # the base kit's file that becomes run_alphafold_fast.py
ROWS_RELPATH = "rows.sh"                                               # the base kit's COMMON= / FAST= rows, read as text (kit_rows)
FPF_LAUNCHER = "run_alphafold_flashpairformer.py"                     # the FlashPairformer add-on's launcher (fpf_launch.py runs it: import the package, then the script)
FPF_PACKAGE_INIT = os.path.join("af3_flashpairformer", "__init__.py")
LEVERS_LAUNCHER = "levers_launch.py"                                   # the tree's launcher beside this module: apply_levers() + the script
FPF_TREE_LAUNCHER = "fpf_launch.py"                                    # the tree's launcher beside this module: the add-on's launcher + the SERVED line at exit
BIG_LAUNCHER = "big_launch.py"                                     # the tree's launcher beside this module: the base's installation + the memory levers (big.py)
OFF_ARG = "--off"                                                       # the memory launcher's own word after its fixed prefix: the memory levers a --n_gpu P > 1 composition switches off (with_n_gpu)
BIG = "big"                                                        # the memory mode (big.py): ONE composition on the fast base, parametric on the base
BIG_BASE = "fast"                                                    # big = the fast MODE + the kit's memory levers; a big on "exact" = the same composition on that base (one line in BIG_LINES, never a second code path)
BIG_DISENGAGED = {                                                   # region reach, ALWAYS: fast levers disengaged BY LEVER PROPERTY where a memory lever owns the site or the lever is a memory holder
    "FPF_TRIMUL": "site_owned (trimul_chunk takes TriangleMultiplication; the fused kernel forgone measured 0.00 GiB and -5.9 / -13.7 s per input at 2,048 / 3,072 padded tokens on one H100)",
    "FPF_HOIST": "memory holder (24 x [16, N, N] persistent pair logits: +2.88 GiB at 2,048, +6.36 GiB at 3,072, about +11.4 GiB at 4,096 padded tokens, measured) + site_owned (logits_shard takes the transformer's pair path)",
    "HOIST_LOGITS": "dependant of the memory holder (indexes FPF_HOIST's resident pair logits)",
    "ATOM_COND_HOIST": "dependant of the memory holder (its precompute rides on FPF_HOIST's step) + site_owned (samples_per_pass re-implements the sampler)",
    "TRIMUL_CD": "site_owned (trimul_chunk takes TriangleMultiplication: the provider's row serves whole square planes, the chunked class owns the site)"}

BIG_SERVE_EDGE = 5120                                                  # padded tokens: the largest pair size the fused pair kernels serve (FPF_TRIATT / TTR: N % KERNEL_TILE == 0 and <= 5120); above it,
BIG_DISENGAGED_AT_EDGE = {                                            # with the size unreadable, or under --n_gpu P (the row-sharded transcription calls the module's own attention / transition per row block) these leave too:
    "FPF_TRIATT": "serve edge (the fused kernel serves N % KERNEL_TILE == 0 and <= 5120 on one GPU; the fork's attention with TRIATT_XLA's core beneath) — kept below the edge: 0.00 GiB, -18.4 / -43.1 s per input at 2,048 / 3,072",
    "TTR": "serve edge (tile multiple, <= 5120, one GPU) — kept below the edge beside transition_shard, whose row shards call the fused block: 0.00 GiB, -1.5 / -3.6 s per input at 2,048 / 3,072",
    "COND_SHARE": "the row-sharded line composes SAMPLES_PER_PASS, whose chunked sampler re-implements diffusion_head.sample (site owned); past the serve edge / size unknown: kept off (unmeasured there)"}
BIG_FUSED_CC_FLOOR = 9.0                                              # compute capability floor for the fused pair kernels INSIDE the memory line: below it (cc 8.0 cards) they are disengaged as well —
BIG_DISENGAGED_BELOW_CC = {                                           # at 2,048 padded tokens on a cc 8.0 80 GB card the memory line's allocator peak is 12.94 GiB with FPF_TRIATT + TTR, 11.11 GiB without
    "FPF_TRIATT": "memory holder on this card class (cc < 9.0: the fused pair kernels' A100 tiles hold +1.83 GiB allocator peak at 2,048 padded tokens, measured together with TTR; exact's ATTNCFG serves the site) — kept on cc >= 9.0: 0.00 GiB there",
    "TTR": "memory holder on this card class (cc < 9.0, measured together with FPF_TRIATT: +1.83 GiB at 2,048; transition_shard's row shards call the module's own transition) — kept on cc >= 9.0"}
BIG_FALLBACK = {"GLUT": "FPF_TRIMUL", "ATTNCFG": "FPF_TRIATT"}         # exact's bitwise Pallas levers at the sites the disengaged fast levers vacate (levers_launch.install; the fork's route beneath): each joins only when the fast lever at its site is disengaged
BIG_LINES = {BIG: BIG_BASE}                                      # line -> the base mode composed on: the mode's own composition only (big_levers.LINE_LEVERS names the levers per line)
BIG_N_STAR = 1408                                                     # the memory mode's size boundary (padded tokens), every card: at or below it `big` runs the fast line's program (the memory
                                                                        # levers cannot lower a peak that is already under 9 GiB there: 8.35 GiB allocator bytes at 1,408); above it the memory
                                                                        # line runs, whose allocator peak stays below the fast line's program at every size (2,048 padded tokens: memory line
                                                                        # 10.5 GiB where the fast line's program holds 15.2 GiB with FPF_HOIST's resident pair logits; 3,072: 20 GiB class vs 32.3 GiB) —
                                                                        # the boundary is a memory promise, not fast's single-card reach (fast itself runs to 3,072 on an 80 GB card)
BIG_N_STAR_BY_CARD_MIB = ((75_000, 1408), (0, 1408))                  # (memory.total MiB floor, N*): one boundary for the 80 GB and the 40 GB card alike (the memory line's peak at 2,048, 10.5 GiB, is inside
                                                                        # every supported card); kept as a table so a card-specific boundary remains one row; no GPU visible when the region is decided: BIG_N_STAR
BIG_REGIONS = ("fast", "reach")                                       # big's two regions, decided UP FRONT by the wrapper from the input's token estimate (big_region): at or below
                                                                        # BIG_N_STAR on one device the memory mode IS the fast line lever for lever (region fast); above it, under --n_gpu > 1, or
                                                                        # when the size cannot be read, the memory levers' program (region reach: base_levers_of + the memory levers)
BIG_NOT_COMPOSED = {                                                  # registered memory levers the one-GPU line does NOT compose (named on the launcher's --off word), each with the measurement that keeps it off:
    "SAMPLES_PER_PASS": "0.00 GiB at 2,048 and 4,096 padded tokens for +45 % time (one diffusion sample per pass); 0.00 GiB on the row-sharded line (--n_gpu 2, 2,048 padded tokens) too: off on every line (0.3.38)",
    "LOGITS_SHARD": "0.00 GiB at 2,048 and 4,096 padded tokens for +4.6 % time (per-block pair logits in row shards)"}
                                                                        # re-enable either by one KIT_MODES line if a measurement above the serve edge (5,120) shows a benefit
BIG_NGPU_LEVERS: List[str] = []                                       # memory levers composed under --n_gpu P > 1 ONLY, ahead of the line's levers: none — SAMPLES_PER_PASS does not pay there either:
                                                                        # on 2 devices at 2,048 padded tokens (5 samples, 10 recycles) the row-sharded program with the stock vmapped sampler holds the
                                                                        # same allocator peak (9.00 GiB per device either way) and runs faster per input, so the P > 1 line leaves it off by name like the one-GPU line
BIG_MEMORY_LEVERS_LABEL = "transition_shard,cond_shard,trimul_chunk"   # the memory levers named on the region NOTE (big_levers.LINE_LEVERS['big'], in line order)


def card_memory_mib() -> Optional[int]:
    """The first visible GPU's memory.total in MiB (nvidia-smi, stack.gpu_info), or None (no GPU / no nvidia-smi: a CPU check)."""
    try:
        from .stack import gpu_info
        g = gpu_info()
    except Exception:                                                     # noqa: BLE001 — the region decision never raises on the probe; None = the reference card's value, named in the rule words
        return None
    return int(g["memory_mib"]) if g and g.get("memory_mib") else None


def card_compute_cap() -> Optional[float]:
    """The first visible GPU's compute capability (nvidia-smi, stack.gpu_info), or None (no GPU / no nvidia-smi)."""
    try:
        from .stack import gpu_info
        g = gpu_info()
    except Exception:                                                     # noqa: BLE001 — as card_memory_mib: the composition never raises on the probe
        return None
    return float(g["compute_cap"]) if g and g.get("compute_cap") else None


def n_star_for_card(memory_mib: Optional[int]) -> int:
    """The memory mode's size boundary for a card of this memory (BIG_N_STAR_BY_CARD_MIB); None (no GPU visible) -> BIG_N_STAR."""
    if memory_mib is None:
        return BIG_N_STAR
    for floor, n_star in BIG_N_STAR_BY_CARD_MIB:
        if int(memory_mib) >= floor:
            return n_star
    return BIG_N_STAR_BY_CARD_MIB[-1][1]


def big_region(n_est: Optional[int], n_gpu: int = 1, memory_mib: Optional[int] = -1, cc: Optional[float] = -1.0) -> dict:
    """The memory mode's region for an input, decided up front: {'region': 'fast'|'reach', 'n_est', 'n_padded', 'n_star', 'rule' (the words of
    the decision)} by the size rule: --n_gpu > 1 → reach
    (the row-sharded stack is the memory levers' program); size unknown → reach (named); the estimate padded to the kernel tile <= BIG_N_STAR
    → fast (the fast line's program); else reach (the memory line). A lever inactive by this rule is decided here, never a fallback at run time."""
    mem = card_memory_mib() if memory_mib == -1 else memory_mib             # -1 = probe the visible card (nvidia-smi); None = no GPU visible: the 80 GB reference value, said in the rule words
    cc = (card_compute_cap() if memory_mib == -1 else None) if cc == -1.0 else cc   # the card's compute capability (nvidia-smi; -1 = probe with the memory), None when no GPU is visible / not probed: the cc >= 9.0 reference composition
    n_star = int(n_star_for_card(mem))
    card = ("no GPU visible: the 80 GB reference" if mem is None else f"{round(int(mem) / 1024)} GB card")
    n_padded = None
    if n_est is not None:
        tile = kernel_tile()
        n_padded = -(-int(n_est) // tile) * tile
    base = {"n_est": None if n_est is None else int(n_est), "n_padded": n_padded, "n_star": n_star, "n_gpu": int(n_gpu), "card_mib": mem, "cc": cc}
    if int(n_gpu) > 1:
        return {**base, "region": "reach", "rule": f"n_gpu={int(n_gpu)} > 1: the row-sharded pair stack is the memory levers' program"}
    if n_est is None:
        return {**base, "region": "reach", "rule": "input size not readable up front (no fold-input JSON of the alphafold3 dialect): the memory levers' program"}
    if n_padded <= n_star:
        return {**base, "region": "fast", "rule": f"n_est={int(n_est)} pads to {n_padded} <= n_star={n_star} (the memory line's size boundary, {card})"}
    edge = (("the fused pair kernels off on this card class (cc=%.1f < %s: memory holders here)" % (float(cc), BIG_FUSED_CC_FLOOR)) if (cc is not None and float(cc) < BIG_FUSED_CC_FLOOR) else
            "the fused pair kernels kept (at or below their serve edge %d)" % BIG_SERVE_EDGE) if n_padded <= BIG_SERVE_EDGE else "past the fused pair kernels' serve edge %d (the fork's classes beneath)" % BIG_SERVE_EDGE
    return {**base, "region": "reach", "rule": f"n_est={int(n_est)} pads to {n_padded} > n_star={n_star} (above the memory line's size boundary, {card}); {edge}"}


def effective_mode(mode: str, region: Optional[str] = None) -> str:
    """The mode whose program runs: big in region fast IS the fast line (its launcher, levers, flags, environment, cache class and evidence
    rules — one composition, never a second table); every other (mode, region) is the mode itself."""
    return BIG_LINES[BIG] if (mode == BIG and region == "fast") else mode


def region_note(reg: dict) -> str:
    """The ONE line the wrapper prints after ACTIVE under big: which lever set the region decided inactive, and why (the rule's words)."""
    if reg["region"] == "fast":
        return (f"BIG NOTE region=fast n_est={reg['n_est']} n_star={reg['n_star']}: memory levers {BIG_MEMORY_LEVERS_LABEL} inactive by size "
                f"({reg['rule']}); the fast line's levers engaged lever for lever")
    dis = big_disengaged(reg)
    off = ",".join(sorted(dis))
    kept = [lv for lv in ("FPF_TRIATT", "TTR") if lv not in dis]
    cc = reg.get("cc")
    card_rule = (f"; FPF_TRIATT+TTR off by the card rule (cc={float(cc):.1f} < {BIG_FUSED_CC_FLOOR}: +1.83 GiB allocator peak at 2,048 padded tokens with them, measured)"
                 if (cc is not None and float(cc) < BIG_FUSED_CC_FLOOR and int(reg.get("n_gpu") or 1) == 1) else "")
    return (f"BIG NOTE region=reach n_est={'unknown' if reg['n_est'] is None else reg['n_est']} n_star={reg['n_star']}: memory levers engaged "
            f"({reg['rule']}); the fast line's {off} inactive by property (a memory lever owns the site, the lever is a memory holder, or the size is past "
            f"the fused pair kernels' serve edge {BIG_SERVE_EDGE} / row-sharded)" + (f"; {'+'.join(kept)} kept (0.00 GiB measured, faster)" if kept else "") + card_rule)


BIG_UNSIZED = "big: the program depends on the input size — pass --input_dir or --n_est N"   # the refusal of a big line with nothing to size (effective_line require_size; warm's rule)
N_EST_FLAG = "--n_est"                                                    # the caller's own token estimate (pred / check / warm): wins over the input's


class BigUnsized(ValueError):
    """``--mode big`` with neither an input to size nor ``--n_est`` where the verb must know the line up front
    (warm: the class it builds is the class pred reads for that size — a class warmed blind is one pred never reads). Usage class (cli: rc 2)."""


def effective_line(mode: str, input_path: Optional[str] = None, n_est: Optional[int] = None, n_gpu=1,
                   require_size: bool = False, record: Optional[dict] = None) -> dict:
    """THE resolver of the line a (mode, input) runs — the one place pred, check, warm and stack.activate decide the memory mode's region, the
    mode whose program runs and the cache class, so that the class ``warm`` builds IS the class ``pred`` reads for the same input:
    ``{'mode', 'mode_effective' (effective_mode), 'region' ('fast'|'reach' under big, else None), 'record' (big: big_region's record +
    'estimate', the ACTIVE line's region tokens; else None), 'cache_class' (cache_class_of)}``. Every other mode is itself
    (region None, the table's class). Under big the size is, in this order: ``record`` (a record this function already resolved — activate's
    path: nothing re-read), ``n_est`` (the caller's ``--n_est``: wins over the input), ``input_path`` (a fold input or a directory of them, read
    by pred's estimator inputs.token_estimate — the one reader), else unknown.
    ``require_size``: a big line with nothing to size (no record, no --n_est, no input, n_gpu <= 1) raises BigUnsized
    (BIG_UNSIZED) instead of taking the size-unknown rule (reach) — warm's rule; check's dry run and the Python API name the rule instead."""
    if mode not in KIT_MODES:
        raise UnsupportedMode(f"unknown mode {mode!r}; modes are {' | '.join(MODES)}")
    if mode != BIG:
        return {"mode": mode, "mode_effective": mode, "region": None, "record": None, "cache_class": KIT_MODES[mode]["cache_class"]}
    try:
        from opt_core.mem.ngpu import check_n_gpu                         # the resource axis' one validator: a non-integer / non-positive --n_gpu is a usage error in its words
        n_gpu = check_n_gpu(n_gpu)
    except ImportError:                                                   # an older core (no opt_core.mem.ngpu): stack.activate's producer gate names it (NOT ACTIVE), nothing resolves past it
        pass
    if record is not None:
        rec = dict(record)
    else:
        if n_est is not None:
            est = {"n_est": int(n_est), "files": 0, "per_file": {}, "unreadable": [], "given_by": N_EST_FLAG}
        elif input_path:
            from .inputs import token_estimate
            est = token_estimate(input_path)                              # pred's estimator: polymer residues + ligand atoms per copy, the largest input of a directory; unreadable -> None (named)
        else:
            sized_otherwise = isinstance(n_gpu, int) and n_gpu > 1                # n_gpu > 1 is the reach region by the resource rule
            if require_size and not sized_otherwise:
                raise BigUnsized(BIG_UNSIZED)
            est = {"n_est": None, "files": 0, "per_file": {}, "unreadable": ["no input given"]}
        rec = {**big_region(est["n_est"], n_gpu), "estimate": est}
    region = rec["region"]
    if region not in BIG_REGIONS:
        raise ValueError(f"unknown region {region!r}; regions: {' | '.join(BIG_REGIONS)}")
    try:
        cache_class = cache_class_of(BIG, region=region)
    except UnsupportedMode as e:                                         # the memory mode's own gate (big.gate: the core's lever registry / a lever unregistered): stack.activate refuses by
        cache_class, rec = None, {**rec, "cache_class_refused": str(e)}   # the same gate BY NAME (NOT ACTIVE, rc 3) — the class is undecidable, never guessed
    return {"mode": BIG, "mode_effective": effective_mode(BIG, region), "region": region, "record": rec, "cache_class": cache_class}


ROW_NAMES = ("COMMON", "FAST")
KIT_MODES = {   # last key: the add-on directories (the add-on table in stack) a mode runs, gated by carry.carry_check (presence) at activation
    "off": {"script": STOCK_SCRIPT, "levers": [], "launcher": None, "cache_class": "", "kits": ()},
    "exact": {"script": FAST_SCRIPT, "levers": ["FIX1", "GLUT", "ATTNCFG", "L1", "WRITER"], "launcher": "levers", "cache_class": "",
              "kits": ("fast_inference", "pallas")},
    "fast": {"script": FAST_SCRIPT, "levers": ["FIX1", "FPF_TRIMUL", "FPF_TRIATT", "FPF_HOIST", "TTR", "DATTN", "TRIATT_XLA", "SAMPLER_BF16", "ATOM_ATTN", "TRIMUL_CD", "HOIST_LOGITS", "COND_SHARE", "ATOM_COND_HOIST", "LNP", "L1", "WRITER"], "launcher": "fpf",
             "cache_class": "__fast", "kits": ("fast_inference", "fpf")},
    # big: the memory levers (big.py / big_levers.py, the ids of registry.LEVERS) composed on the fast base (BIG_BASE; the fast levers of
    # BIG_DISENGAGED off by their own property, BIG_FALLBACK at the vacated sites, the base's row lever kept); the levers in force and the
    # cache class resolve per flag set (levers_of; cache_class_of: `__big-big-<sha8 of the lever set in force>` — an executable serialized
    # under one lever set is never loaded by another, the kit's executable key carries no lever); the last key = every add-on directory the composition runs
    "big": {"script": FAST_SCRIPT, "levers": ["TRANSITION_SHARD", "COND_SHARD", "TRIMUL_CHUNK"], "launcher": "big", "cache_class": "__big",
              "kits": ("fast_inference", "pallas", "fpf")},
}


def big_disengaged(size: Optional[dict] = None) -> Dict[str, str]:
    """The fast levers region reach disengages for an input, lever -> why: BIG_DISENGAGED always; BIG_DISENGAGED_AT_EDGE too when the
    padded size is past BIG_SERVE_EDGE, unreadable, or the run is row-sharded (--n_gpu > 1); BIG_DISENGAGED_BELOW_CC (FPF_TRIATT, TTR) when the
    card's compute capability is below BIG_FUSED_CC_FLOOR. ``size`` = the region record (big_region:
    n_padded, n_gpu, cc); None = size unknown (the conservative set: every fused pair lever off)."""
    rec = size or {}
    n_padded, n_gpu = rec.get("n_padded"), int(rec.get("n_gpu") or 1)
    out = dict(BIG_DISENGAGED)
    why = ("n_gpu=%d > 1 (row blocks call the module's own attention / transition)" % n_gpu if n_gpu > 1 else
           "size not readable up front" if n_padded is None else
           "n_padded=%d > serve edge %d" % (int(n_padded), BIG_SERVE_EDGE) if int(n_padded) > BIG_SERVE_EDGE else None)
    if why:
        out.update({lv: f"{words} [{why}]" for lv, words in BIG_DISENGAGED_AT_EDGE.items()})
    cc = rec.get("cc")
    if cc is not None and float(cc) < BIG_FUSED_CC_FLOOR:               # the card class below the fused kernels' floor (cc 8.0): the fused pair kernels are memory holders there (measured) — off by name
        out.update({lv: f"{words} [cc={float(cc):.1f} < {BIG_FUSED_CC_FLOOR}]" for lv, words in BIG_DISENGAGED_BELOW_CC.items() if lv not in out})
    return out


def base_levers_of(line: str = BIG, disengaged: Optional[Dict[str, str]] = None) -> List[str]:
    """The base's graph/in-script lever ids a big line keeps: the base mode's levers minus the disengaged set (big_disengaged: by lever
    property and the input's size) plus the BIG_FALLBACK levers whose site a disengaged lever vacates (exact's bitwise kernels); the base's
    row levers are composed by resolve() after the memory levers. ``disengaged`` None = size unknown (big_disengaged(None))."""
    dis = big_disengaged(None) if disengaged is None else disengaged
    base = [lv for lv in KIT_MODES[BIG_LINES[line]]["levers"] if lv not in ROW_LEVER_FLAGS]
    return [lv for lv in base if lv not in dis] + [fb for fb, vacated_by in BIG_FALLBACK.items() if vacated_by in dis]


def levers_of(mode: str, disengaged: Optional[Dict[str, str]] = None) -> List[str]:
    """The lever ids of a mode's composition (registry.LEVERS keys): a kit mode's table row; big = the base levers it keeps
    (base_levers_of, for the input's disengaged set) + the memory levers of its line, in order (big.selection) + the base's row levers."""
    if mode != BIG:
        return list(KIT_MODES[mode]["levers"])
    from . import big as _b
    why = _b.gate()                                                      # the core's lever registry absent or a lever unregistered: refused BY NAME, never composed silently
    if why:
        raise UnsupportedMode(why)
    rows = [lv for lv in KIT_MODES[BIG_LINES[BIG]]["levers"] if lv in ROW_LEVER_FLAGS]
    return base_levers_of(BIG, disengaged) + [lv.upper() for lv in _b.selection()["levers"] if lv.upper() not in BIG_NOT_COMPOSED] + rows   # the registered line minus the levers the one-GPU line does not compose (with_n_gpu adds BIG_NGPU_LEVERS back for P > 1)


PALLAS_LEVER_VARS = {"GLUT": "AF3P_GLU_T", "ATTNCFG": "AF3P_ATTN_CFG"}   # the Pallas add-on's lever ids -> their PALLAS_LEVERS variable (with_n_gpu / with_levers_off remove a lever by removing its variable)
FPF_KERNEL_VAR, FPF_HOIST_VAR, FPF_HOIST_OFF = "AF3_FLASHPAIRFORMER", "AF3_DIFFUSION_HOIST", "0"   # the FlashPairformer add-on's two words (fpf_env reads the package's defaults; fpf_env_disengaged their off values)


def lever_switch_vars() -> Dict[str, str]:
    """lever id -> the ONE model-process variable that switches it, for the levers that have one of their own (GLUT, ATTNCFG: the Pallas add-on's;
    TTR, DATTN: the tree's TREE_LEVER_ENV). The FPF levers share the add-on's two words (fpf_kernel_switch, FPF_HOIST_VAR), L1 rides its row
    flags, the memory levers the launcher's --off word, FIX1 and ROWPAIR have no switch (ablation.SWITCHLESS)."""
    return {**PALLAS_LEVER_VARS, **{lv: next(iter(env)) for lv, env in TREE_LEVER_ENV.items()}}


def fpf_kernel_switch(levers: Sequence[str]) -> str:
    """The FlashPairformer add-on's kernel word for a composition: ``both`` | ``trimul`` | ``triatt`` | ``off`` by which of FPF_TRIMUL / FPF_TRIATT it
    names (af3_flashpairformer.patch MODES; ``both`` is the package default fpf_env reads for the full ``fast`` row)."""
    t, a = "FPF_TRIMUL" in levers, "FPF_TRIATT" in levers
    return "both" if (t and a) else "trimul" if t else "triatt" if a else "off"


def with_levers_off(res: dict, names: Sequence[str], n_gpu: int = 1) -> dict:
    """The composition ``res`` (``resolve``'s, after ``with_n_gpu``) WITHOUT the levers ``names`` lists — the ablation switch
    ``MODEL_OPT_LEVERS_OFF`` (ablation.py states the semantics). Each lever leaves through the switch the kit already composes for it: its
    variable leaves ``env`` (lever_switch_vars), the FlashPairformer word is rewritten to what remains (fpf_kernel_switch) and the hoist's to
    its off value, L1's flags leave ``flags``, a memory lever joins the launcher's ``--off`` word; ``levers`` loses the names, ``ablated``
    records them (request order); the cache class, script and launcher file stay the mode's. ``names`` empty: ``res`` unchanged (no
    ``ablated`` key). A name the composition cannot ablate raises ablation.AblationError (validated here, by name)."""
    from . import ablation as _abl
    names = _abl.validate(res["mode"], list(names), res.get("levers") or [], n_gpu=n_gpu)
    if not names:
        return res
    out = dict(res)
    gone = set(names)
    levers = [l for l in res["levers"] if l not in gone]
    env = dict(res.get("env") or {})
    switch = lever_switch_vars()
    for l in names:
        if l in switch:
            env.pop(switch[l], None)
    if gone & set(SERVED_KERNELS.values()):
        env[FPF_KERNEL_VAR] = fpf_kernel_switch(levers)
    if "FPF_HOIST" in gone:
        env[FPF_HOIST_VAR] = FPF_HOIST_OFF
    row_gone = [l for l in names if l in ROW_LEVER_FLAGS]
    flags = list(res.get("flags") or [])
    if row_gone:
        drop = {f for l in row_gone for f in ROW_LEVER_FLAGS[l]}
        flags = [t for t in flags if t.split("=", 1)[0] not in drop]
        out["row_levers"] = [l for l in res.get("row_levers") or [] if l not in gone]
        if not out["row_levers"]:
            out["row"] = ""
    mem_gone = [l.lower() for l in names if l in KIT_MODES[BIG]["levers"]]   # the memory levers (the memory line's own set), as the launcher's lowercase words
    launcher = list(res.get("launcher") or [])
    if mem_gone and launcher:
        if OFF_ARG in launcher:                                         # --n_gpu P already names the superseded set: one word, extended
            i = launcher.index(OFF_ARG)
            launcher[i + 1] = ",".join(launcher[i + 1].split(",") + mem_gone)
        else:
            launcher += [OFF_ARG, ",".join(mem_gone)]
    out.update(levers=levers, env=env, flags=flags, launcher=launcher or res.get("launcher"), ablated=list(names))
    return out


def ablated_per_lever(names: Sequence[str]) -> dict:
    """{lever: {'name', 'state': 'off', 'reason': 'ablated'}} for the levers MODEL_OPT_LEVERS_OFF removed — their LEVER lines (report.lever_lines renders
    them after the composition's own: ``state=off reason=ablated``)."""
    from . import ablation as _abl, registry
    return {l: {"name": (registry.LEVERS.get(l) or {}).get("strategy") or l, "state": "off", "reason": _abl.REASON} for l in names}


def cache_class_of(mode: str, region: Optional[str] = None) -> str:
    """The cache class suffix of a mode: the table's; big = ``__big`` + ``-big-<sha8>`` of its lever set (big.cache_suffix);
    big in region fast = the fast line's class (the same program: a cache warmed for fast serves it)."""
    mode = effective_mode(mode, region)
    if mode != BIG:
        return KIT_MODES[mode]["cache_class"]
    from . import big as _b
    why = _b.gate()
    if why:
        raise UnsupportedMode(why)
    return KIT_MODES[mode]["cache_class"] + _b.cache_suffix()
KIT_ROW = "FAST"                                                        # the kit's row whose flags the row levers select from (ROW_LEVER_FLAGS)
ROW_LEVER_FLAGS = {                                                     # row lever -> the flag NAMES it owns in the FAST row (values stay in the row); every FAST-row flag belongs to exactly one
    "L1": ("--featurisation_workers", "--featurisation_prefetch"),
    "WRITER": ("--output_writer",),
}
def exact_only_flags(kit: Optional[str] = None) -> List[str]:
    """The flag names of the kit's FAST row (the fast-inference levers' switches), read from the row: refused on the stock route."""
    return sorted({t.split("=", 1)[0] for t in split_row(kit_rows(kit)["FAST"]) if t.startswith("--")})
COMPOSED_KEYS = ("--cache_dir", "--input_dir", "--json_path", "--output_dir")   # the COMMON row's flags the command layer composes (cli.py)
# COMMON="..." / FAST="..." — the value is the quoted text
_ROW_RX = re.compile(r'^(COMMON|FAST)="(.*)"\s*$')
PALLAS_LEVERS = {"AF3P_GLU_T": "1", "AF3P_ATTN_CFG": "64,64,4,3"}         # the Pallas add-on's Tier-1 (bitwise vs stock) lever set, as its apply_levers() reads it from the
                                                                        # model process environment (pallas_addon/patches/af3_pallas_levers.py): L-GLUT on; L-ATTNCFG = the pair
                                                                        # flash-attention tile pin block_q,block_k,num_warps,num_stages — the module's only two switches.
_FPF_MODE_RX = re.compile(r'_os\.environ\.get\("(AF3_FLASHPAIRFORMER)",\s*"(\w+)"\)')
_FPF_HOIST_RX = re.compile(r'_os\.environ\.get\("(AF3_DIFFUSION_HOIST)",\s*"\w+"\)[^\n]*\bin \(([^)]*)\)')


from opt_core.modes import UnsupportedMode  # noqa: E402  the core's: a ValueError (ModeError) the kit raises for an unknown mode / composition
def kit_rows(kit: Optional[str] = None) -> Dict[str, str]:
    """{'COMMON': ..., 'FAST': ...} as written in the base kit's rows file (ROWS_RELPATH; shell variables unexpanded)."""
    from .stack import kit_home
    path = os.path.join(kit or kit_home(), ROWS_RELPATH)
    rows: Dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for ln in f:
            m = _ROW_RX.match(ln)
            if m and m.group(1) not in rows:
                rows[m.group(1)] = m.group(2)
    missing = [r for r in ROW_NAMES if r not in rows]
    if missing:
        raise RuntimeError(f"{path}: rows {missing} not found (the kit's rows file changed shape)")
    return rows


def common_flags(kit: Optional[str] = None) -> List[str]:
    """The COMMON row's flags every arm shares, minus the ones cli.py composes (COMPOSED_KEYS) and the ``$MODEL`` placeholder."""
    return [t for t in kit_rows(kit)["COMMON"].split() if not t.startswith("$") and t.split("=", 1)[0] not in COMPOSED_KEYS]


def pallas_levers() -> Dict[str, str]:
    """The Pallas add-on's Tier-1 lever set as {variable: value} (a copy of PALLAS_LEVERS: callers may update it)."""
    return dict(PALLAS_LEVERS)


def fpf_env(kit: Optional[str] = None) -> Dict[str, str]:
    """The FlashPairformer add-on's two switches, read from its package: the kernel switch at the package default (both kernels), the hoist
    switch at the first value its package accepts as on."""
    from .stack import kit_home
    path = os.path.join(kit or kit_home("fpf"), FPF_PACKAGE_INIT)
    with open(path, encoding="utf-8") as f:
        src = f.read()
    m1, m2 = _FPF_MODE_RX.search(src), _FPF_HOIST_RX.search(src)
    if not m1 or not m2:
        raise RuntimeError(f"{path}: the add-on's switches were not found (the package changed shape)")
    on_values = [v.strip().strip('"') for v in m2.group(2).split(",")]
    return {m1.group(1): m1.group(2), m2.group(1): on_values[0]}


def split_row(row: str, cache_dir: Optional[str] = None) -> List[str]:
    """A row into argv tokens, `$CACHE` -> cache_dir (left as written when cache_dir is None)."""
    if cache_dir is not None:
        row = row.replace("$CACHE", cache_dir)
    return row.split()


def launcher(kind: Optional[str]) -> List[str]:
    """The argv prefix in front of the script: [] (off), [levers_launch.py, <pallas patches dir>] (exact), [fpf_launch.py, run_alphafold_flashpairformer.py]
    (fast) or [big_launch.py, <core dir>, <pallas patches dir>, <run_alphafold_flashpairformer.py>, --line, big, --base, fast] (big);
    the big prefix ends with ``--off <the not-composed levers>`` (BIG_NOT_COMPOSED); with_n_gpu / with_levers_off rewrite that one word."""
    from .stack import kit_home, opt_home
    if kind is None:
        return []
    if kind == "levers":
        return [os.path.join(opt_home(), "af3_jax_opt", LEVERS_LAUNCHER), os.path.join(kit_home("pallas"), "patches")]
    if kind == "fpf":
        return [os.path.join(opt_home(), "af3_jax_opt", FPF_TREE_LAUNCHER), os.path.join(kit_home("fpf"), FPF_LAUNCHER)]
    if kind == "big":                                                     # + --line <line> --base <mode>: the launcher installs the base by line and records both on its ACTIVE line
        from .stack import core_dir
        return [os.path.join(opt_home(), "af3_jax_opt", BIG_LAUNCHER), core_dir(), os.path.join(kit_home("pallas"), "patches"),
                os.path.join(kit_home("fpf"), FPF_LAUNCHER), "--line", BIG, "--base", BIG_LINES[BIG],
                OFF_ARG, ",".join(lv.lower() for lv in BIG_NOT_COMPOSED)]   # the registered levers the one-GPU line does not compose, off BY NAME on the launcher's ACTIVE line (with_n_gpu rewrites the word for P > 1)
    raise UnsupportedMode(f"unknown launcher kind {kind!r}")


TREE_LEVER_ENV = {"COND_SHARE": {"AF3_JAX_COND_SHARE": "1"}, "ATOM_COND_HOIST": {"AF3_JAX_ATOM_COND_HOIST": "1"}, "DATTN": {"AF3_JAX_DATTN": "1"}, "TTR": {"AF3_JAX_TTR": "fast"}, "TRIATT_XLA": {"AF3_JAX_TRIATT_XLA": "fast"}, "SAMPLER_BF16": {"AF3_JAX_SAMPLER_BF16": "1"}, "ATOM_ATTN": {"AF3_JAX_ATOM_ATTN": "1"}, "TRIMUL_CD": {"AF3_JAX_TRIMUL_CD": "fast"}, "LNP": {"AF3_JAX_LNP": "fast"}, "HOIST_LOGITS": {"AF3_JAX_HOIST_LOGITS": "1"}}   # IN INSTALL ORDER (fpf_launch.TREE_LEVERS states why)   # the tree's own in-process levers: switch variables under the package's prefix (inprocess/dattn.py, inprocess/ttr.py ENV_SWITCH)
TIER_WORD_SWITCHES = {"DATTN": "AF3_JAX_DATTN", "TTR": "AF3_JAX_TTR", "TRIATT_XLA": "AF3_JAX_TRIATT_XLA", "TRIMUL_CD": "AF3_JAX_TRIMUL_CD", "LNP": "AF3_JAX_LNP"}      # tree levers bound through an opt_core provider face by the provider's TIER word (exact | fast | big; opt_core.kernels.pallas TIER_WORDS):
                                                                        # the switch VALUE the model process reads IS the composition's word — `1` in this table = the module's DEFAULT_WORD (fast); lever_env / tier_word_env write the mode's
                                                                        # own word (fast under fast, big under big in both regions); the provider resolves the measured row per call per card from it (inprocess/ttr.py, inprocess/lnp.py word())
INPROCESS_MODULES = {"DATTN": "inprocess/dattn.py", "TTR": "inprocess/ttr.py", "TRIATT_XLA": "inprocess/triatt_xla.py", "SAMPLER_BF16": "inprocess/sampler_bf16.py", "ATOM_ATTN": "inprocess/atom_attn.py", "TRIMUL_CD": "inprocess/trimul_cd.py", "LNP": "inprocess/lnp.py", "HOIST_LOGITS": "inprocess/hoist_logits.py", "COND_SHARE": "inprocess/cond_share.py", "ATOM_COND_HOIST": "inprocess/atom_cond_hoist.py",   # the tree modules a lever loads into the model process
                     **{lv: "big_levers.py" for lv in list(KIT_MODES[BIG]["levers"]) + list(BIG_NOT_COMPOSED) + BIG_NGPU_LEVERS}}
GRAPH_KINDS = ("kernel", "schedule")                                    # registry kinds whose levers change the compiled graph (executables are indexed under them, det.py)



def mode_env(mode: str) -> Dict[str, str]:
    """The add-on rows a mode sets in the model process (the mode table owns them; the caller's copies are stripped first, stack.model_process_env).
    The composition's full variable set — these plus the switches of the tree's own in-process levers it names — is lever_env()."""
    if mode == "exact":
        return pallas_levers()
    if mode == "fast":
        return fpf_env()
    if mode == BIG:                                                       # the base's switches with the fast levers disengaged by property, exact's LEVERS row at the vacated sites: size unknown here —
        return big_env(levers_of(BIG))                                  # the conservative composition (lever_env(BIG, levers) is the composition-exact form resolve() uses)
    return {}


def big_env(levers: Sequence[str]) -> Dict[str, str]:
    """Region reach's add-on rows FOR A COMPOSITION: fast's FlashPairformer words set to what the composition keeps (fpf_kernel_switch:
    ``triatt`` when FPF_TRIATT is kept below the serve edge, ``off`` past it / row-sharded; the hoist always ``0`` — FPF_HOIST is the memory
    holder) and exact's LEVERS row for the fallback levers the composition names (GLUT at trimul_chunk's site; ATTNCFG only where FPF_TRIATT
    is disengaged). The keys are fast's and exact's own, never a second spelling."""
    env = dict(fpf_env())
    env[FPF_KERNEL_VAR] = fpf_kernel_switch(levers)
    env[FPF_HOIST_VAR] = "1" if "FPF_HOIST" in levers else FPF_HOIST_OFF
    pal = pallas_levers()
    for lv, var in PALLAS_LEVER_VARS.items():
        if lv not in levers:
            pal.pop(var, None)
    env.update(pal)
    return env


def fpf_env_disengaged() -> Dict[str, str]:
    """fast's FlashPairformer switches at their OFF values (the add-on imports and installs nothing; the hoist stays off): the size-unknown /
    past-the-edge / row-sharded reach composition — the keys are fast's own (fpf_env), never a second spelling."""
    return {k: ("off" if k == "AF3_FLASHPAIRFORMER" else "0") for k in fpf_env()}


def tier_word_env(mode: str, levers: Sequence[str], env: Dict[str, str]) -> Dict[str, str]:
    """``env`` with the switch of every TIER_WORD_SWITCHES lever the composition names set to the composition's tier word — the mode's name
    (exact | fast | big): the word the lever hands the shared core's provider, which resolves the measured row per call from it."""
    out = dict(env)
    if mode != BIG:                                                     # fast keeps the table's values (`fast`; DATTN's `1` = its default word fast); exact names none of these levers
        return out
    for lever, var in TIER_WORD_SWITCHES.items():
        if lever in levers and var in out:
            out[var] = BIG                                              # big's own composition (region reach, and under --n_gpu P where the lever survives) names `big`; at or below N* big IS the fast line (resolve: region fast), word fast
    return out


def lever_env(mode: str, levers: Sequence[str]) -> Dict[str, str]:
    """The model-process variables of a composition: mode_env(mode) (big: big_env(levers), composition-exact) + TREE_LEVER_ENV of every lever in
    it, the TIER_WORD_SWITCHES levers' switches carrying the composition's tier word (tier_word_env)."""
    env = dict(big_env(levers) if mode == BIG else mode_env(mode))
    for lever in levers:
        env.update(TREE_LEVER_ENV.get(lever, {}))
    return tier_word_env(mode, levers, env)


TIER_WORD = {"fast": "fast", BIG: "big"}                            # the provider tier word a kit mode names (opt_core.kernels.pallas TIER_WORDS); exact names no provider row at these sites (the provider's exact word there is the stock statement, by name)
TIER_WORD_VARS = tuple(TIER_WORD_SWITCHES.values())                     # the switches that CARRY the word ("1" = the module's default word fast, else the word verbatim)


def with_tier_word(env: Dict[str, str], mode: str) -> Dict[str, str]:
    """``env`` with every TIER_WORD_SWITCHES variable present carrying ``mode``'s tier word (the pair stack's, the transition's and the LayerNorm
    lever's provider switches alike) — one rule for every provider-bound lever: the word is the composition's mode (big's memory line: big; the fast line, also when big runs it at or below N*: fast)."""
    if mode != BIG:
        return dict(env)
    return {k: (BIG if k in TIER_WORD_VARS else v) for k, v in env.items()}


def row_lever_flags(levers: List[str], cache_dir: Optional[str] = None, kit: Optional[str] = None) -> List[str]:
    """The FAST row's tokens (as written, in row order, ``$CACHE`` -> cache_dir) whose flag name belongs to one of ``levers``
    (ROW_LEVER_FLAGS). Raises when the row carries a flag no row lever owns or a listed row lever finds none of its flags in the row —
    the row and the map agree by construction or the kit's rows file changed shape."""
    row = split_row(kit_rows(kit)[KIT_ROW], cache_dir)
    owner = {flag: lever for lever, flags in ROW_LEVER_FLAGS.items() for flag in flags}
    unowned = [t for t in row if t.startswith("--") and t.split("=", 1)[0] not in owner]
    if unowned:
        raise RuntimeError(f"{KIT_ROW} row flags {unowned} belong to no row lever (ROW_LEVER_FLAGS)")
    wanted = [l for l in levers if l in ROW_LEVER_FLAGS]
    for lever in wanted:
        if not any(t.split("=", 1)[0] in ROW_LEVER_FLAGS[lever] for t in row):
            raise RuntimeError(f"row lever {lever}: none of its flags {ROW_LEVER_FLAGS[lever]} is in the {KIT_ROW} row")
    return [t for t in row if t.startswith("--") and owner[t.split("=", 1)[0]] in wanted]


PAD_POLICY = {"off": "kernel_tile", "exact": "kernel_tile", "fast": "kernel_tile", BIG: "kernel_tile"}   # ONE padding in every mode, the stock route included (the stock route runs upstream with --buckets=<the kernel_tile list>; a caller's own --buckets later on the line wins): every mode computes the same padded size for an input, so `exact` == `off` compares like for like and the fused pair kernels of `fast` / `big` are served at every padded size (pad_buckets)
PAD_POLICIES = ("af3_buckets", "af3_buckets_served", "kernel_tile")     # af3_buckets_served = the fork's own list without the buckets below the kernel tile (an input of <= 64 tokens pads to the tile and is served; every larger input pads exactly as the fork pads it)


def kernel_tile() -> int:
    """The shared core's kernel tile (``opt_core.shape_policy.KERNEL_TILE``): the padding multiple of the fast line, the same constant in every
    kit that pads; the fused pair kernels serve exactly the token counts that are multiples of it."""
    from opt_core.shape_policy import KERNEL_TILE
    return int(KERNEL_TILE)


def pad_buckets(policy: str, tree: Optional[str] = None) -> List[int]:
    """The ``--buckets`` list a padding policy writes ([] = nothing written: the fork's default list as shipped). ``kernel_tile``: every
    multiple of the core's KERNEL_TILE up to the fork's largest default bucket (settings.upstream_buckets: ... 5120) — an input pads to the next
    multiple of the tile (tile 64: 400 -> 448, 600 -> 640, 1400 -> 1408; the fork's own list pads 400 -> 512, 600 -> 768 and 1400 -> 1536, and every
    padded token is computed in every layer), so every padded size is one the fused kernels serve; above the largest bucket the fork's own rule
    stands (a bucket of exactly the input's size). ``af3_buckets_served``: the fork's list without the buckets that are not multiples of the
    tile (32, 64). NOTE the fork draws the diffusion noise at the padded shape (``jax.random.normal`` values depend on the shape): a padded
    length other than the fork's default bucket is an implicit re-seed of the sampler for that input (fast-class numerics either way)."""
    if policy not in PAD_POLICIES:
        raise ValueError(f"unknown padding policy {policy!r}; policies: {' | '.join(PAD_POLICIES)}")
    if policy == "af3_buckets":
        return []
    from . import settings as _settings
    tile, stock = kernel_tile(), _settings.upstream_buckets(tree)
    if max(stock) < tile:
        raise RuntimeError(f"the fork's largest --buckets default {max(stock)} is below the core's kernel tile {tile}")
    if policy == "af3_buckets_served":
        return [b for b in stock if b % tile == 0]
    return list(range(tile, max(stock) + 1, tile))


def pad_flags(mode: str, tree: Optional[str] = None) -> List[str]:
    """``--buckets=<list>`` for the mode's PAD_POLICY (pad_buckets), or nothing. One compiled executable per bucket met, cached in the mode's
    cache class like every other; a caller's own ``--buckets`` later on the line wins by absl order (the fork's list as shipped, stated
    verbatim, is ``--buckets=`` + settings.upstream_buckets)."""
    kept = pad_buckets(PAD_POLICY[mode], tree)
    if not kept:
        return []
    from . import settings as _settings
    return [f"--{_settings.BUCKETS_FLAG_NAME}={','.join(str(b) for b in kept)}"]


def resolve(mode: str, cache_dir: Optional[str] = None, kit: Optional[str] = None, region: Optional[str] = None, size: Optional[dict] = None) -> dict:
    """{'mode', 'script', 'row' (the FAST row as written when any row lever runs, else ''), 'row_levers' (the row levers whose flags
    are taken from it), 'flags' (their tokens, $CACHE substituted, then the mode's padding flag: pad_flags), 'levers' (the composition: the mode's row —
    graph and in-script levers first, then the row levers), 'launcher' (argv prefix), 'env' (lever_env of the composition), 'cache_class',
    'base' (big: the mode composed on, else None), 'disengaged' (big reach: big_disengaged(size), lever -> why; else {}), 'region'
    (big: 'fast'|'reach', else None)}. ``region='fast'`` under big: the fast line's own resolution, lever for lever (effective_mode), labelled
    mode big. ``size`` = the region record (big_region / effective_line: n_padded, n_gpu) — region reach's composition is decided from it
    (the fused pair kernels kept at or below BIG_SERVE_EDGE on one GPU); None = size unknown, the conservative composition."""
    if mode not in KIT_MODES:
        raise UnsupportedMode(f"unknown mode {mode!r}; modes are {' | '.join(MODES)}")
    if mode == BIG and region == "fast":
        out = resolve(effective_mode(mode, region), cache_dir, kit)
        return {**out, "mode": BIG, "base": BIG_LINES[BIG], "disengaged": {}, "region": "fast"}   # the fast line lever for lever — its executables (cache class __fast, warmed once for both) and therefore its provider word (fast) on every face; region reach names big (lever_env)
    if region not in (None, "reach") and mode == BIG:
        raise ValueError(f"unknown region {region!r}; regions: {' | '.join(BIG_REGIONS)}")
    spec = KIT_MODES[mode]
    dis = big_disengaged(size) if mode == BIG else {}
    comp = levers_of(mode, dis if mode == BIG else None)               # the mode's composition (big: base levers kept for this size + its memory levers + the base's row levers)
    levers = [l for l in comp if l not in ROW_LEVER_FLAGS] + [l for l in comp if l in ROW_LEVER_FLAGS]   # graph + in-script levers, then the row levers
    row_levers = [l for l in levers if l in ROW_LEVER_FLAGS]
    return {"mode": mode, "script": spec["script"], "row": kit_rows(kit)[KIT_ROW] if row_levers else "", "row_levers": row_levers,
            "flags": (row_lever_flags(levers, cache_dir, kit) if row_levers else []) + pad_flags(mode), "levers": levers, "launcher": launcher(spec["launcher"]),
            "env": lever_env(mode, levers), "cache_class": cache_class_of(mode),
            "base": BIG_LINES[BIG] if mode == BIG else None, "disengaged": dis,
            "region": ("reach" if mode == BIG else None)}   # big: the base composed on and the fast levers off by property (the BIG NOTE line carries the reasons)


def exact_only(argv: List[str]) -> List[str]:
    """The kit-row flags present in argv (refused on the stock route)."""
    names = set(exact_only_flags())
    return [a for a in argv if a.split("=", 1)[0] in names]


# the evidence a lever process prints (what `pred`/`warm` require of the model process's transcript — one line per lever per process):
#   exact — the tree's launcher: `[af3-jax-opt] LEVERS active=<name>+<name> ...`, one add-on lever per AF3P_* variable of the LEVERS row
#   fast  — the FlashPairformer package's own install line (af3_flashpairformer/patch.py `log.warning`):
#           `af3_flashpairformer: mode=<mode> (TriangleMultiplication=<class>, GridSelfAttention=<class>), jax <ver> backend <b>`
#           and the tree launcher's line at exit (fpf_launch.py): `[af3-jax-opt] SERVED trimul fused=<n> fallback=<n> triatt fused=<n> fallback=<n>
#           fallback_shapes=<...|none> hoist=<traced sample calls>|off|uncounted|probe_error:<Exc> tiles=own:<cc>|fallback:<cc>->9.0 cc=<x.y>` — counts are TRACE-time call sites (one per traced
#           site, not per executed call); tiles = the add-on's tile table for this GPU (its own, or its fallback to the sm_90 table: a named degraded configuration);
#           `dattn=<traced calls>|off dattn_sites=<site:n,...|none>` = the tree's DATTN lever (inprocess/dattn.py report: calls served through the shared core's
#           provider opt_core.kernels.pallas serve.attention by the mode's tier word) — plus the lever's own census line at exit,
#           `[af3-jax-opt] DATTN word=<tier word> served=<n> rows=<provider arm:n,...|none> aside=<reason:n,...|none> uncovered=<family:n,...|none>` (dattn.census_line;
#           the LEVER line quotes it, the kernels gate reads the served rows from it);
#           `ttr=<sites a provider kernel row served>|off ttr_routed=<sites the provider's stock statement served BY NAME: C > 256 (the single transitions), a family without a measured cell> ttr_fallback=<shape:reason:n,...|none>`
#           = the tree's TTR lever (inprocess/ttr.py report: the shared core's transition face by tier word; a fallback is a transition site the provider did not serve — named, never silent);
#           `sbf16=<traced cast sites>|off sbf16_sites=<site:n,...|none>` = the tree's SAMPLER_BF16 lever (inprocess/sampler_bf16.py report);
#           `txla=<served traced calls>|off txla_rows=<row:n,...|none> txla_aside=<reason:n,...|none>` = the tree's TRIATT_XLA lever (inprocess/triatt_xla.py report: the core's
#           triangle-attention kernels served / the calls they stepped aside from, by name);
#           `atomattn=<traced calls>|off atomattn_sites=<site:n,...|none> atomattn_fallback=<reason:n,...|none>` = the tree's ATOM_ATTN lever (inprocess/atom_attn.py report;
#           a fallback is a cross-attention call whose shapes the fused kernel does not tile — named, the stock lines served it);
#           `hlog=<step traces>|off|skipped hlog_dtype=<bfloat16|none|<skip reason>>` = the tree's HOIST_LOGITS lever (inprocess/hoist_logits.py report;
#           skipped = the hoist it indexes is ablated or held: the lever steps aside by name);
#           `cnoise=off cnoise_rule=none` = constant words (no lever draws the sampler's noise other than the fork's own)
#           `tcd=<served traced calls>|off tcd_rows=<arm:n,...|none> tcd_aside=<reason:n,...|none>` = the tree's TRIMUL_CD lever (inprocess/trimul_cd.py report: the core's
#           JAX-family provider row that computed the triangle-multiplication module at each traced call site / the calls it stepped aside from, by name);
#           + the LNP lever's own line after it (LNP_LINE_RX: `LNP served=<n>|off word= rows= units= routed= aside= uncovered=`, inprocess/lnp.py line())
#   big — the base's own evidence with the fast kernels DISENGAGED (exact's LEVERS line naming the fallback levers; the SERVED line, when printed,
#           with nothing fused) AND the launcher's two lines (big.py ACTIVE_RX / CENSUS_RX, the grammar of opt_core.mem.record):
#           `[af3-jax-opt] ACTIVE mode=big:<lever,...> base=<mode> drop=<..> exact=<label> refused=<..|none> off=<..|none> on=<..|none> allocator=<..|none> line=big`
#           before the script runs (every lever applied or refused BY NAME) and, at exit, `[af3-jax-opt] BIG census units=<n> applied=<..>
#           refused=<..> traced=<lever=n,...> verdict=ok|partial` (the launcher's fail-closed gate)
#   row levers — the kit script's own lines (fast_inference/patches/patched_files/run_alphafold.py): L1 `Featurisation prefetch enabled: <n> worker
#           process(es), <k> item(s) ahead.`; WRITER `Output writer enabled: result extraction and output writing run on one writer thread behind the next fold job.`
LEVERS_LINE_RX = re.compile(r"\[af3-jax-opt\] LEVERS active=(\S+)")
FPF_LINE_RX = re.compile(r"af3_flashpairformer: mode=(\w+) \(TriangleMultiplication=(\w+), GridSelfAttention=(\w+)\)")
SERVED_LINE_RX = re.compile(r"\[af3-jax-opt\] SERVED trimul fused=(\d+) fallback=(\d+) triatt fused=(\d+) fallback=(\d+) fallback_shapes=(\S+)(?: hoist=([\w:.]+))?(?: tiles=(\S+) cc=(\S+))?(?: dattn=(\w+) dattn_sites=(\S+))?(?: ttr=(\w+) ttr_routed=(\d+) ttr_fallback=(\S+))?(?: sbf16=(\w+) sbf16_sites=(\S+))?(?: txla=(\w+) txla_rows=(\S+) txla_aside=(\S+))?(?: atomattn=(\w+) atomattn_sites=(\S+) atomattn_fallback=(\S+))?(?: hlog=(\w+) hlog_dtype=(\S+))?(?: cshare=(\w+))?(?: achoist=(\w+) achoist_sites=(\S+) achoist_aside=(\S+))?(?: cnoise=(\w+) cnoise_rule=(\S+))?(?: tcd=(\w+) tcd_rows=(\S+) tcd_aside=(\S+)(?: tcd_word=(\S+) tcd_uncovered=(\S+))?)?(?: held=(\S+))?")
SERVED_KERNELS = {"trimul": "FPF_TRIMUL", "triatt": "FPF_TRIATT"}       # the add-on's kernel names on the SERVED line -> the lever ids
# The LNP lever's own census line (inprocess/lnp.py line(), printed by the tree's launcher at exit after SERVED): `[af3-jax-opt] LNP served=<traced calls the
# provider served>|off word=<provider word: fast | big | a row> rows=<row:n,...|none> units=<unit:n,...|none> routed=<reason:n,...|none: calls kept on the stock
# class BY RULE> aside=<refusal:n,...|none: provider refusals by name> uncovered=<family:n,...|none: served calls whose family has no measured cell>`
LNP_LINE_RX = re.compile(r"\[af3-jax-opt\] LNP served=(\w+) word=(\S+) rows=(\S+) units=(\S+) routed=(\S+) aside=(\S+) uncovered=(\S+)")
ROW_LEVER_RX = {                                                        # row lever -> the kit script's line that proves it ran in this process
    "L1": re.compile(r"Featurisation prefetch enabled: (\d+) worker"),
    "WRITER": re.compile(r"Output writer enabled: (result extraction)"),
}
# The template census guard's lines (inprocess/templates.py, installed by the launchers of every mode of the kit): `TEMPLATES guard=installed|failed:<..>`
# once per model process, `TEMPLATES chain=<id> declared= kept= real=<r>/<max> dropped=<slot:reason,...|none>` per chain that declared templates,
# `TEMPLATES census declared= kept= real= chains_templated= chains_all_dummy=[ form=row_born] verdict=ok|untemplated|all_dummy` per featurised input
# (templates_census reads them: monitoring only, recorded on the DONE line).
TEMPLATES_LINE_RX = re.compile(r"\[af3-jax-opt\] TEMPLATES (guard=(?P<guard>\S+)|chain=(?P<chain>\S+) declared=(?P<cdecl>\d+) kept=(?P<ckept>\d+) real=(?P<creal>\d+)/(?P<cmax>\d+) dropped=(?P<dropped>\S+)"
                               r"|census declared=(?P<decl>\d+) kept=(?P<kept>\d+) real=(?P<real>\d+) chains_templated=(?P<ct>\d+) chains_all_dummy=(?P<cz>\d+)(?: form=(?P<form>row_born))? verdict=(?P<verdict>\w+))")
BIG_LINE_RX = re.compile(r"\[af3-jax-opt\] (ACTIVE mode=big:|BIG )")               # the big launcher's ACTIVE / census / record lines (parsed by big.py)
from .inprocess import dattn as _dattn_lines  # noqa: E402  DATTN's census-line grammar and switch words (a stdlib-only module: the model process loads the same file by path)
DATTN_LINE_RX = _dattn_lines.LINE_RX                                     # the DATTN lever's own census line at exit: `[af3-jax-opt] DATTN word= served= rows= aside= uncovered=`
EVIDENCE_RXS = (LEVERS_LINE_RX, FPF_LINE_RX, SERVED_LINE_RX, LNP_LINE_RX, TEMPLATES_LINE_RX, BIG_LINE_RX, DATTN_LINE_RX) + tuple(ROW_LEVER_RX.values())


def templates_census(lines: List[str], declared_in_input: int, mode: str) -> dict:
    """The template census of one wrapper run from the model process's TEMPLATES lines (monitoring only — the model process runs the input
    exactly as the stock script does): ``guard`` (installed | failed:<..> | absent), ``censuses`` (one per featurised input), totals
    ``declared``/``real``, ``chains`` (per-chain lines), ``verdict``: ``untemplated`` (the input declares no template), ``ok``, ``all_dummy``
    (a chain's declared templates are all dummies — counted, the run proceeds), ``uncensused`` (mode off = the stock script: no census hook)
    or ``census_missing`` (the input declares templates and no census line came back), and the DONE-line token ``token``."""
    guard, censuses, chains = "absent", [], []
    for ln in lines:
        m = TEMPLATES_LINE_RX.search(ln)
        if not m:
            continue
        if m.group("guard"):
            guard = m.group("guard")
        elif m.group("chain"):
            chains.append({"chain": m.group("chain"), "declared": int(m.group("cdecl")), "kept": int(m.group("ckept")), "real": int(m.group("creal")),
                           "max_templates": int(m.group("cmax")), "dropped": [] if m.group("dropped") == "none" else m.group("dropped").split(",")})
        elif m.group("verdict"):
            censuses.append({"declared": int(m.group("decl")), "kept": int(m.group("kept")), "real": int(m.group("real")),
                             "chains_templated": int(m.group("ct")), "chains_all_dummy": int(m.group("cz")), "form": m.group("form"), "verdict": m.group("verdict")})
    declared, real = sum(c["declared"] for c in censuses), sum(c["real"] for c in censuses)
    if mode == "off":
        verdict = "uncensused" if declared_in_input else "untemplated"
    elif any(c["verdict"] == "all_dummy" for c in censuses):
        verdict = "all_dummy"
    elif declared_in_input and not censuses:
        verdict = "census_missing"
    elif not declared_in_input and not declared:
        verdict = "untemplated"
    else:
        verdict = "ok"
    token = {"untemplated": "none", "uncensused": f"uncensused:{declared_in_input}", "census_missing": f"census_missing:{declared_in_input}",
             "all_dummy": f"all_dummy:{real}/{declared or declared_in_input}", "ok": f"{real}/{declared}"}[verdict]
    return {"guard": guard, "censuses": censuses, "chains": chains, "declared_in_input": int(declared_in_input),
            "declared": declared, "real": real, "verdict": verdict, "token": token}


REFUSAL_RX = re.compile(r"^\[af3-jax-opt\] NOT ACTIVE: (FPF_TRI(?:MUL|ATT)=\S.*)$")   # the tree launcher's refusal of a kernel that cannot install on this GPU (fpf_launch.refusal_line)


def refusal_reason(lines: List[str]) -> Optional[str]:
    """The reason text of the model process's own NOT ACTIVE line for a lever that could not engage (fpf_launch.refuse_held), else None."""
    for ln in lines:
        mm = REFUSAL_RX.search(ln.strip())
        if mm:
            return mm.group(1)
    return None


def is_evidence_line(ln: str) -> bool:
    """Whether a transcript line is one of the lever-evidence lines (what pred/warm collect for lever_evidence), or the launcher's refusal line."""
    return any(rx.search(ln) for rx in EVIDENCE_RXS) or bool(REFUSAL_RX.search(ln.strip()))


def _row_lever_evidence(levers: List[str], lines: List[str]) -> dict:
    """The row levers of ``levers``: each proven by its own line (ROW_LEVER_RX)."""
    short, reported = [], []
    for lever in (l for l in levers if l in ROW_LEVER_RX):
        hits = [ln.strip() for ln in lines if ROW_LEVER_RX[lever].search(ln)]
        reported += hits[:1]
        if not hits:
            short.append(f"{lever} printed no '{ROW_LEVER_RX[lever].pattern.split('(')[0].strip()}' line")
    return {"short": short, "reported": reported}




def lnp_evidence(lines: List[str]) -> dict:
    """The LNP lever's fields from its own line (LNP_LINE_RX; the last one printed): lnp (served traced calls | off | absent), lnp_word, lnp_rows,
    lnp_units, lnp_routed, lnp_aside, lnp_uncovered — 'absent' / 'none' when no line was printed."""
    ms = [m for ln in lines for m in [LNP_LINE_RX.search(ln)] if m]
    if not ms:
        return {"lnp": "absent", "lnp_word": "none", "lnp_rows": "none", "lnp_units": "none", "lnp_routed": "none", "lnp_aside": "none", "lnp_uncovered": "none"}
    m = ms[-1]
    return {"lnp": m.group(1), "lnp_word": m.group(2), "lnp_rows": m.group(3), "lnp_units": m.group(4), "lnp_routed": m.group(5), "lnp_aside": m.group(6), "lnp_uncovered": m.group(7)}


def per_lever(mode: str, levers: List[str], ev: dict, lines: List[str]) -> dict:
    """{lever_id: {'name' (the shared strategy id, else the lever id), 'state' ('on' = its evidence is in this process's transcript;
    'skipped' = in the composition, evidence absent — 'reason' names why), 'evidence'}} for every lever of the composition — the census the
    LEVER lines print (report.lever_lines; one line per lever per process). FIX1 lives in the kit script itself: its evidence is that the
    kit script ran (the SCRIPT line's digest, cli.ensure_fast_script), stated as ``evidence=in-script``."""
    from . import registry
    spec, out = KIT_MODES[mode], {}
    lv = [m.group(1) for ln in lines for m in [LEVERS_LINE_RX.search(ln)] if m]
    active = ([] if lv[-1] == "none" else lv[-1].split("+")) if lv else []
    served, hoist = ev.get("served") or {}, str(ev.get("hoist", "absent"))
    for lever in levers:
        name = (registry.LEVERS.get(lever) or {}).get("strategy") or lever
        st, reason, evidence, fields = "skipped", None, None, {}
        if lever == "FIX1":
            st, evidence = ("on", "in-script") if spec["script"] == FAST_SCRIPT else ("skipped", None)
            reason = None if st == "on" else "the stock script carries no FIX1"
        elif lever in ("GLUT", "ATTNCFG"):
            st = "on" if f"L-{lever}" in active else "skipped"
            evidence, reason = (f"LEVERS active={'+'.join(active) or 'none'}", None) if st == "on" else (None, f"not on the LEVERS line (active={'+'.join(active) or 'none'})")
        elif lever in SERVED_KERNELS.values():                           # kernel levers: the core's served= / fallback= keys (trace-time call sites)
            c = served.get(lever)
            if c:
                fields = {"served": c["fused"], "fallback": c["fallback"]}
            if lever in (ev.get("held") or {}):                             # the core refused this part: the class stayed stock, by name
                reason = ev["held"][lever]
            elif c and c["fused"] and not c["fallback"]:
                st = "on"
            else:
                reason = "no SERVED line" if not c else f"fused={c['fused']} fallback={c['fallback']}"
        elif lever == "FPF_HOIST":
            if hoist.isdigit() and int(hoist) > 0:
                st, evidence = "on", f"hoist={hoist}"
            else:
                reason = f"hoist={hoist}"
        elif lever == "ATOM_ATTN":
            aa = str(ev.get("atomattn", "absent"))
            if aa.isdigit() and int(aa) > 0 and ev.get("atomattn_fallback", "none") == "none":
                st, evidence = "on", f"atomattn={aa} sites={ev.get('atomattn_sites', 'none')}"
            else:
                reason = f"atomattn={aa} fallback={ev.get('atomattn_fallback', 'none')}"
        elif lever == "SAMPLER_BF16":
            sbf16 = str(ev.get("sbf16", "absent"))
            if sbf16.isdigit() and int(sbf16) > 0:
                st, evidence = "on", f"sbf16={sbf16} sites={ev.get('sbf16_sites', 'none')}"
            else:
                reason = f"sbf16={sbf16}"
        elif lever == "TRIATT_XLA":
            txla = str(ev.get("txla", "absent"))
            if txla.isdigit() and int(txla) > 0:
                st, evidence = "on", f"txla={txla} rows={ev.get('txla_rows', 'none')} aside={ev.get('txla_aside', 'none')}"
            else:
                reason = f"txla={txla} aside={ev.get('txla_aside', 'none')}"
        elif lever == "TRIMUL_CD":
            tcd = str(ev.get("tcd", "absent"))
            if tcd.isdigit() and int(tcd) > 0:
                st, evidence = "on", f"tcd={tcd} word={ev.get('tcd_word', 'none')} rows={ev.get('tcd_rows', 'none')} aside={ev.get('tcd_aside', 'none')} uncovered={ev.get('tcd_uncovered', 'none')}"   # the word named (the mode's tier word, TIER_WORD) and the rows the provider actually served
            else:
                reason = f"tcd={tcd} aside={ev.get('tcd_aside', 'none')}"
        elif lever == "LNP":
            lnp = str(ev.get("lnp", "absent"))
            if lnp.isdigit() and int(lnp) > 0 and ev.get("lnp_aside", "none") == "none":
                st, evidence = "on", f"lnp={lnp} word={ev.get('lnp_word', 'none')} rows={ev.get('lnp_rows', 'none')} units={ev.get('lnp_units', 'none')} routed={ev.get('lnp_routed', 'none')} uncovered={ev.get('lnp_uncovered', 'none')}"   # the word named (tier word fast | big) and the rows the provider actually served, per unit
            else:
                reason = f"lnp={lnp} aside={ev.get('lnp_aside', 'none')}"
        elif lever == "HOIST_LOGITS":
            hlog = str(ev.get("hlog", "absent"))
            if hlog.isdigit() and int(hlog) > 0:
                st, evidence = "on", f"hlog={hlog} dtype={ev.get('hlog_dtype', 'none')}"
            elif hlog == "skipped":
                st, reason = "skipped", f"the hoist it indexes is not in force ({ev.get('hlog_dtype', 'none')})"
            else:
                reason = f"hlog={hlog}"
        elif lever == "COND_SHARE":
            cs = str(ev.get("cshare", "absent"))
            if cs.isdigit() and int(cs) > 0:
                st, evidence = "on", f"cshare={cs}"
            else:
                reason = f"cshare={cs}"
        elif lever == "ATOM_COND_HOIST":
            ach = str(ev.get("achoist", "absent"))
            if ach.isdigit() and int(ach) > 0 and ev.get("achoist_aside", "none") == "none":
                st, evidence = "on", f"achoist={ach} sites={ev.get('achoist_sites', 'none')}"
            elif ach == "skipped":
                st, reason = "skipped", "FPF_HOIST's step is not in force (the atom conditioning hoist rides on it)"
            else:
                reason = f"achoist={ach} aside={ev.get('achoist_aside', 'none')}"
        elif lever == "TTR":
            ttr = str(ev.get("ttr", "absent"))
            if ttr.isdigit() and int(ttr) > 0 and ev.get("ttr_fallback", "none") == "none":
                st, evidence = "on", f"ttr={ttr} routed={ev.get('ttr_routed', '0')}"
            else:
                reason = f"ttr={ttr} fallback={ev.get('ttr_fallback', 'none')}"
        elif lever == "DATTN":                                            # served through the shared core's provider by tier word: the SERVED line's traced count + the lever's own census line (inprocess/dattn.py census_line: word, the provider arms that served, step-asides and uncovered families by name)
            dattn = str(ev.get("dattn", "absent"))
            _dattn = _dattn_lines
            cen = _dattn.scan(lines)
            cwords = f" word={cen['word']} rows={_dattn._fmt(cen['rows'])} aside={_dattn._fmt(cen['aside'])} uncovered={_dattn._fmt(cen['uncovered'])}" if cen else ""   # no census line (a transcript that ends before the exit hook): the SERVED count alone, as before
            if dattn.isdigit() and int(dattn) > 0:
                st, evidence = "on", f"dattn={dattn} sites={ev.get('dattn_sites', 'none')}{cwords}"
            else:
                reason = f"dattn={dattn}{cwords}"
        elif lever in KIT_MODES[BIG]["levers"] or lever in BIG_NGPU_LEVERS:   # memory levers (+ any BIG_NGPU_LEVERS with_n_gpu composes under P > 1): applied by name on the launcher's ACTIVE line, nothing refused, the census at exit ok; traced = the patched site's trace count
            b = ev.get("big") or {}
            act, cen = b.get("active") or {}, b.get("census") or {}
            lv_ = lever.lower()
            traced = (cen.get("traced") or {}).get(lv_)
            if lv_ in (act.get("levers") or ()) and lv_ not in (cen.get("refused") or ()) and (not cen or lv_ in (cen.get("applied") or ())):
                if cen and cen.get("verdict") == "ok":
                    st, evidence = "on", f"applied traced={traced}" if traced is not None else "applied"
                else:
                    reason = "no BIG census line at exit" if not cen else f"census verdict={cen.get('verdict')}"
            else:
                reason = "not on the big ACTIVE line" if not act else (f"refused ({act.get('refused')})" if lv_ in str(act.get("refused")) else "not applied")
        elif lever in ROW_LEVER_RX:
            hits = [ln.strip() for ln in lines if ROW_LEVER_RX[lever].search(ln)]
            if hits:
                st, evidence = "on", "line"
            else:
                reason = "no line"
        elif lever == ROWPAIR:                                            # the row-sharded pair stack (n_gpu > 1): the adapter's install line + the family's lever line state=on
            rp = ev.get("rowpair") or {}
            if rp.get("ok"):
                st, evidence = "on", f"installed n_gpu={rp['installed']} {rp.get('installed_fields') or ''}".strip() + (f" notes={len(rp['notes'])}" if rp.get("notes") else "")
            else:
                reason = rp.get("reason") or "no ROWPAIR evidence parsed"
        else:
            reason = "no evidence rule for this lever"
        out[lever] = {"name": name, "state": st, "reason": reason, "evidence": evidence, **({"fields": fields} if fields else {})}
    return out


def _lever_evidence(mode: str, lines: List[str], levers: List[str], n_gpu: int = 1) -> dict:
    """{'ok', 'reported': [...], 'reason', ...} for the lever lines a mode's model process printed (``levers``: the composition that ran,
    default the mode's row).
    exact needs the launcher's line naming as many levers as the LEVERS row sets variables; fast needs the add-on's install line at the mode
    the tree sets AND the tree launcher's SERVED line with every kernel served on this input (fused >= 1, fallback == 0; 'served' and
    'fallback_shapes' returned) and the hoist traced (hoist >= 1) — a kernel installed but not served (the add-on serves N % KERNEL_TILE == 0 only: an
    exact-size input above the fork's largest bucket falls back call by call) is a named reason, the exit rule's levers_short (``served_via=traced``).
    big needs the launcher's ACTIVE line naming every lever of the selection with nothing refused, its census
    line at exit with verdict=ok (big.lever_evidence), the base's evidence for the levers it keeps (exact's LEVERS line, the row levers) and
    a SERVED line, when printed, with nothing fused (the fast kernels are disengaged: one that served is named). Every row lever in the
    composition needs its own line (_row_lever_evidence); off needs none."""
    spec = KIT_MODES[mode]
    if spec["launcher"] == "big":                                          # the launcher's ACTIVE + census lines (every lever of the selection applied, nothing refused, verdict ok) AND the base's own evidence
        from . import big as _b
        memory = set(KIT_MODES[BIG]["levers"]) | (set(BIG_NGPU_LEVERS) if int(n_gpu) > 1 else set())   # under --n_gpu P > 1 with_n_gpu composes BIG_NGPU_LEVERS AHEAD of the line's memory levers — they are memory levers of THIS composition, expected on the launcher's ACTIVE line (not handed to the base's evidence); P = 1: the one-GPU line's set, unchanged
        own = _b.lever_evidence(lines, expected=[l for l in levers if l in memory])   # the composition's memory levers (flags and --n_gpu already applied by resolve) are the expectation
        base = _lever_evidence("exact", lines, [l for l in levers if l not in memory])   # the fallback kernels' LEVERS line (exact's evidence) + the row levers kept
        served = [m for ln in lines for m in [SERVED_LINE_RX.search(ln)] if m]
        fused = [f"{SERVED_KERNELS[k]} served fused={n}" for k, n in (("trimul", served[-1].group(1)), ("triatt", served[-1].group(3))) if int(n) and SERVED_KERNELS[k] not in levers] if served else []
        if fused:                                                         # a DISENGAGED fast kernel that served = a defect, named (the SERVED line carries nothing fused for a kernel the composition disengaged)
            base = {**base, "ok": False, "reported": base["reported"] + [served[-1].group(0)],
                    "reason": "; ".join(x for x in ([base["reason"]] if base["reason"] else []) + [f"disengaged fast kernels served: {', '.join(fused)}"])}
        extra = {}
        soft: List[str] = []                                              # DECLARED step-asides of kept fused kernels (the stock class ran at shapes the line names): levers_short WORDS, never an exit code (the fast line's rule)
        if "FPF_TRIATT" in levers:                                        # kept in this region: the add-on's fused triangle attention must have engaged; calls that fell back at declared shapes are named (softened), a kernel that never ran and named nothing is levers_short
            tf, tb = (int(served[-1].group(3)), int(served[-1].group(4))) if served else (0, 0)
            extra = {**extra, "served": {"FPF_TRIATT": {"fused": tf, "fallback": tb}}}
            if not tf and not tb:
                base = {**base, "ok": False, "reason": "; ".join(x for x in ([base["reason"]] if base["reason"] else []) + [f"FPF_TRIATT not served: fused={tf} fallback={tb}"])}
            elif tb:
                soft.append(f"FPF_TRIATT " + (f"aside:pad{kernel_tile()}:{tb}" if not tf else f"not served: fused={tf} fallback={tb}") + f" at {served[-1].group(5)}")
        if "TTR" in levers:                                               # kept beside transition_shard: traced fused sites > 0; a fallback the line declares by shape is a named step-aside (softened), not traced at all is levers_short
            ttr, ttr_routed, ttr_fb = ((served[-1].group(11) or "absent"), (served[-1].group(12) or "0"), (served[-1].group(13) or "none")) if served else ("absent", "0", "none")
            extra = {**extra, "ttr": ttr, "ttr_routed": ttr_routed, "ttr_fallback": ttr_fb}
            if not (ttr.isdigit() and int(ttr) > 0):
                base = {**base, "ok": False, "reason": "; ".join(x for x in ([base["reason"]] if base["reason"] else []) + [f"TTR not served: ttr={ttr} fallback={ttr_fb}"])}
            elif ttr_fb != "none":
                soft.append(f"TTR aside at {ttr_fb}")
        if "DATTN" in levers:                                             # inherited from the fast base at one GPU: the SERVED line's traced count is its evidence; absent / off / 0 = levers_short by name
            dattn, sites = ((served[-1].group(9) or "absent"), (served[-1].group(10) or "none")) if served else ("absent", "none")
            extra = {**extra, "dattn": dattn, "dattn_sites": sites}          # merged: the kept fused levers' fields above (FPF_TRIATT served=, TTR ttr=) stay on the record
            if not (dattn.isdigit() and int(dattn) > 0):
                base = {**base, "ok": False, "reason": "; ".join(x for x in ([base["reason"]] if base["reason"] else []) + [f"DATTN not traced: dattn={dattn}"])}
        if "SAMPLER_BF16" in levers:                                      # inherited from the fast base: the SERVED line's traced cast count is its evidence; absent / off / 0 = levers_short by name
            sbf16, sbf16_sites = ((served[-1].group(14) or "absent"), (served[-1].group(15) or "none")) if served else ("absent", "none")
            extra = {**extra, "sbf16": sbf16, "sbf16_sites": sbf16_sites}
            if not (sbf16.isdigit() and int(sbf16) > 0):
                base = {**base, "ok": False, "reason": "; ".join(x for x in ([base["reason"]] if base["reason"] else []) + [f"SAMPLER_BF16 not traced: sbf16={sbf16}"])}
        if "TRIATT_XLA" in levers:                                        # inherited from the fast base (variant b: the module's core on this line): served calls at trace time; absent / off / 0 = levers_short by name
            txla, txla_rows, txla_aside = ((served[-1].group(16) or "absent"), (served[-1].group(17) or "none"), (served[-1].group(18) or "none")) if served else ("absent", "none", "none")
            extra = {**extra, "txla": txla, "txla_rows": txla_rows, "txla_aside": txla_aside}
            if not (txla.isdigit() and int(txla) > 0):
                base = {**base, "ok": False, "reason": "; ".join(x for x in ([base["reason"]] if base["reason"] else []) + [f"TRIATT_XLA not served: txla={txla} aside={txla_aside}"])}
        if "ATOM_ATTN" in levers:                                         # inherited from the fast base: the SERVED line's traced call count is its evidence; absent / off / 0 or a named fallback = levers_short by name
            aa, aa_sites, aa_fb = ((served[-1].group(19) or "absent"), (served[-1].group(20) or "none"), (served[-1].group(21) or "none")) if served else ("absent", "none", "none")
            extra = {**extra, "atomattn": aa, "atomattn_sites": aa_sites, "atomattn_fallback": aa_fb}
            if not (aa.isdigit() and int(aa) > 0) or aa_fb != "none":
                base = {**base, "ok": False, "reason": "; ".join(x for x in ([base["reason"]] if base["reason"] else []) + [f"ATOM_ATTN not served: atomattn={aa} fallback={aa_fb}"])}
        if "COND_SHARE" in levers:                                        # inherited from the fast base in region reach on one GPU (no chunked sampler composed there): sample calls traced through the shared-conditioning body; absent / off / 0 = levers_short by name
            cshare = (served[-1].group(24) or "absent") if served else "absent"
            extra = {**extra, "cshare": cshare}
            if not (cshare.isdigit() and int(cshare) > 0):
                base = {**base, "ok": False, "reason": "; ".join(x for x in ([base["reason"]] if base["reason"] else []) + [f"COND_SHARE not traced: cshare={cshare}"])}
        if "HOIST_LOGITS" in levers:                                      # inherited from the fast base: step traces are its evidence; `skipped` (the hoist is off/held here) is the lever stepping aside by name, not a shortfall
            hlog, hlog_dtype = ((served[-1].group(22) or "absent"), (served[-1].group(23) or "none")) if served else ("absent", "none")
            extra = {**extra, "hlog": hlog, "hlog_dtype": hlog_dtype}
            if not ((hlog.isdigit() and int(hlog) > 0) or hlog == "skipped"):
                base = {**base, "ok": False, "reason": "; ".join(x for x in ([base["reason"]] if base["reason"] else []) + [f"HOIST_LOGITS not traced: hlog={hlog}"])}
        if "LNP" in levers:                                               # inherited from the fast base: its own LNP line (served calls at trace time through the provider); absent / off / 0 or a refusal by name = levers_short
            lf = lnp_evidence(lines)
            extra = {**extra, **lf}
            if not (lf["lnp"].isdigit() and int(lf["lnp"]) > 0) or lf["lnp_aside"] != "none":
                base = {**base, "ok": False, "reason": "; ".join(x for x in ([base["reason"]] if base["reason"] else []) + [f"LNP not served: lnp={lf['lnp']} aside={lf['lnp_aside']}"])}
        ng = n_gpu_evidence(lines, int(n_gpu), ROWPAIR in levers)         # FAIL-CLOSED: the n_gpu the model process reports must equal the request (a dropped axis never passes)
        rp = rowpair_evidence(lines, int(n_gpu)) if ROWPAIR in levers else None   # --n_gpu P > 1: the row-sharded pair stack installed and evidenced by name, else levers_short
        oks = [own["ok"], base["ok"], ng["ok"]] + ([rp["ok"]] if rp is not None else [])
        return {"ok": all(oks) and not soft, "reported": own["reported"] + base["reported"], "softened": bool(soft) and all(oks), "fell_back": soft,
                "reason": "; ".join(x for x in (ng["reason"], own["reason"], base["reason"], (rp or {}).get("reason"), *soft) if x) or None,
                "big": {"active": own["active"], "census": own["census"]}, "base": {k: base[k] for k in ("ok", "reason")}, "n_gpu": ng, **extra, **({"rowpair": rp} if rp is not None else {})}
    row = _row_lever_evidence(levers, lines)
    if spec["launcher"] == "levers":
        seen = [m.group(1) for ln in lines for m in [LEVERS_LINE_RX.search(ln)] if m]
        want = len(pallas_levers()) - len([l for l in ("GLUT", "ATTNCFG") if levers and l not in levers])   # the row the launcher sets, minus the Pallas levers this composition supersedes by name (--n_gpu P > 1: GLUT → one fewer)
        if not seen:
            reason = "no LEVERS line: the launcher did not run"
        else:
            active = [] if seen[-1] == "none" else seen[-1].split("+")
            reason = None if len(active) == want else f"levers active={seen[-1]}: {len(active)} of the {want} the LEVERS row sets"
        reasons = ([reason] if reason else []) + row["short"]
        return {"ok": not reasons, "reported": seen + row["reported"], "reason": "; ".join(reasons) or None}
    if spec["launcher"] == "fpf":
        seen = [m.group(0) for ln in lines for m in [FPF_LINE_RX.search(ln)] if m]
        wanted = {k for k, lv in SERVED_KERNELS.items() if not levers or lv in levers}   # the add-on kernels this composition names: both for the mode proper, fewer under MODEL_OPT_LEVERS_OFF (with_levers_off)
        want = fpf_env()[FPF_KERNEL_VAR] if len(wanted) == len(SERVED_KERNELS) else fpf_kernel_switch(levers)
        ev: dict = {"ok": False, "reported": seen + row["reported"], "reason": None}
        if not seen and want != "off":
            return {**ev, "reason": "no af3_flashpairformer install line: the add-on did not install"}
        got = FPF_LINE_RX.search(seen[-1]).group(1) if seen else "off"      # AF3_FLASHPAIRFORMER=off (both kernels ablated): the add-on installs no kernel and prints no install line
        if got != want:
            return {**ev, "reason": f"af3_flashpairformer mode={got}, the tree sets {want}"}
        served = [m for ln in lines for m in [SERVED_LINE_RX.search(ln)] if m]
        if not served:
            return {**ev, "reason": "no SERVED line: the tree's launcher did not report the add-on's served set"}
        m = served[-1]
        counts = {"trimul": {"fused": int(m.group(1)), "fallback": int(m.group(2))}, "triatt": {"fused": int(m.group(3)), "fallback": int(m.group(4))}}
        shapes = [] if m.group(5) == "none" else m.group(5).split(",")
        hoist = m.group(6) or "absent"                                    # traced sample calls, 'off' (not installed), 'uncounted', 'probe_error:<Exc>' (the launcher's probe failed) or absent (a launcher without the token)
        tiles, cc = m.group(7) or "absent", m.group(8) or "absent"        # the add-on's tile table for this GPU: 'own', or its fallback (a degraded configuration, named)
        dattn, dattn_sites = m.group(9) or "absent", m.group(10) or "none"
        ttr, ttr_routed, ttr_fallback = m.group(11) or "absent", m.group(12) or "0", m.group(13) or "none"
        sbf16, sbf16_sites = m.group(14) or "absent", m.group(15) or "none"        # the tree's SAMPLER_BF16 lever: traced cast sites inside the sampler (inprocess/sampler_bf16.py report)
        txla, txla_rows, txla_aside = m.group(16) or "absent", m.group(17) or "none", m.group(18) or "none"   # the tree's TRIATT_XLA lever: the core's triangle-attention kernels served at trace time / stepped aside from, by name (inprocess/triatt_xla.py report)
        atomattn, atomattn_sites, atomattn_fallback = m.group(19) or "absent", m.group(20) or "none", m.group(21) or "none"   # the tree's ATOM_ATTN lever: traced cross-attention calls through the fused kernel (inprocess/atom_attn.py report)
        hlog, hlog_dtype = m.group(22) or "absent", m.group(23) or "none"            # the tree's HOIST_LOGITS lever: step traces | off | skipped (the hoist is not in force) and the resident dtype (inprocess/hoist_logits.py report)
        cshare = m.group(24) or "absent"                                    # the tree's COND_SHARE lever: sample calls traced through the shared-conditioning body (inprocess/cond_share.py report)
        achoist, achoist_sites, achoist_aside = m.group(25) or "absent", m.group(26) or "none", m.group(27) or "none"   # the tree's ATOM_COND_HOIST lever: sample calls whose atom conditioning was hoisted, enc:/dec:/passed: call counts, step-asides by name (inprocess/atom_cond_hoist.py report)
        cnoise, cnoise_rule = m.group(28) or "absent", m.group(29) or "none"      # constant words on the line (cnoise=off cnoise_rule=none): no lever draws the sampler's noise otherwise
        tcd, tcd_rows, tcd_aside, tcd_word, tcd_uncov = m.group(30) or "absent", m.group(31) or "none", m.group(32) or "none", m.group(33) or "none", m.group(34) or "none"   # the tree's TRIMUL_CD lever: the provider row's served triangle-multiplication calls at trace time / stepped aside from, by name (inprocess/trimul_cd.py report)
        held = {SERVED_KERNELS.get(k, k): w for k, w in (x.split(":", 1) for x in m.group(35).split(",") if ":" in x)} if m.group(35) else {}   # lever -> fallback:no_tiles_cc<NN>(<kind>)
        lf = lnp_evidence(lines)                                          # the tree's LNP lever: its own line (inprocess/lnp.py line())
        degraded = [] if tiles == "own" or tiles.startswith("own:") or tiles.startswith("safe:") or held else [f"FPF tile table not this GPU's own: tiles={tiles} cc={cc}"]   # own:<cc> | safe:<gen> (the core's generation rows) engage; a held kernel names the refusal itself
        fields = {"served": {SERVED_KERNELS[k]: c for k, c in counts.items()}, "fallback_shapes": shapes, "hoist": hoist, "tiles": tiles, "cc": cc,
                  "dattn": dattn, "dattn_sites": dattn_sites, "ttr": ttr, "ttr_routed": ttr_routed, "ttr_fallback": ttr_fallback, "sbf16": sbf16, "sbf16_sites": sbf16_sites, "txla": txla, "txla_rows": txla_rows, "txla_aside": txla_aside, "atomattn": atomattn, "atomattn_sites": atomattn_sites, "atomattn_fallback": atomattn_fallback, "hlog": hlog, "hlog_dtype": hlog_dtype, "cshare": cshare, "achoist": achoist, "achoist_sites": achoist_sites, "achoist_aside": achoist_aside, "cnoise": cnoise, "cnoise_rule": cnoise_rule, "tcd": tcd, "tcd_rows": tcd_rows, "tcd_aside": tcd_aside, "tcd_word": tcd_word, "tcd_uncovered": tcd_uncov, **lf,
                  **({"held": held} if held else {})}
        declared = {k: [s for s in shapes if s.startswith(k + ":")] for k in counts}   # the fallback shapes the launcher DECLARED per kernel (fallback_shapes=<k>:<shape>,...): a fallback with its shape named is a step-aside by name
        unserved = [f"{SERVED_KERNELS[k]} held: {held[SERVED_KERNELS[k]]} (no tile table for cc={cc}: the stock class served every call)" if SERVED_KERNELS[k] in held else
                    f"{SERVED_KERNELS[k]} not served: fused={c['fused']} fallback={c['fallback']}" + (f" at {','.join(declared[k])}" if c["fallback"] else "")
                    for k, c in counts.items() if k in wanted and (SERVED_KERNELS[k] in held or (not c["fused"] and not (c["fallback"] and declared[k])))]   # never engaged (fused=0 with no declared fallback: the kernel did not run and did not say why) or held: the exit rule's levers_short (a kernel the composition does not name is not judged)
        fell_back = [f"{SERVED_KERNELS[k]} " + (f"aside:pad{kernel_tile()}:{c['fallback']}" if not c["fused"] else f"not served: fused={c['fused']} fallback={c['fallback']}") + f" at {','.join(declared[k])}"
                     for k, c in counts.items() if k in wanted and c["fallback"] and declared[k] and SERVED_KERNELS[k] not in held]   # the stock class ran at DECLARED shapes — some call sites (fused>0) or, for an input whose padded size is not a multiple of the kernel tile (a caller's own --buckets value, an exact-size input above the largest bucket), every call site (fused=0: aside:pad<tile>:<n>) — named (levers_short words), never an exit code: the run completed on the stock class by name
        if "FPF_HOIST" in levers and not (hoist.isdigit() and int(hoist) > 0):
            unserved.append(f"FPF_HOIST not traced: hoist={hoist}")
        if "DATTN" in levers and not (dattn.isdigit() and int(dattn) > 0):
            unserved.append(f"DATTN not traced: dattn={dattn}")
        if "TTR" in levers and not (ttr.isdigit() and int(ttr) > 0):
            unserved.append(f"TTR not traced: ttr={ttr}")
        if "SAMPLER_BF16" in levers and not (sbf16.isdigit() and int(sbf16) > 0):
            unserved.append(f"SAMPLER_BF16 not traced: sbf16={sbf16}")
        if "TRIATT_XLA" in levers and not (txla.isdigit() and int(txla) > 0):   # installed and nothing served (every call stepped aside, or the module never bound): levers_short by name, the asides worded
            unserved.append(f"TRIATT_XLA not served: txla={txla} aside={txla_aside}")
        if "ATOM_ATTN" in levers and not (atomattn.isdigit() and int(atomattn) > 0):
            unserved.append(f"ATOM_ATTN not traced: atomattn={atomattn}")
        if "ATOM_ATTN" in levers and atomattn_fallback != "none":          # a cross-attention call the fused kernel did not tile: named, the run is levers_short (never a silent stock path)
            unserved.append(f"ATOM_ATTN not served at {atomattn_fallback}")
        if "HOIST_LOGITS" in levers and not ((hlog.isdigit() and int(hlog) > 0) or hlog == "skipped"):
            unserved.append(f"HOIST_LOGITS not traced: hlog={hlog}")
        if "COND_SHARE" in levers and not (cshare.isdigit() and int(cshare) > 0):   # the shared-conditioning sampler body traced, or levers_short by name
            unserved.append(f"COND_SHARE not traced: cshare={cshare}")
        if "ATOM_COND_HOIST" in levers and not ((achoist.isdigit() and int(achoist) > 0 and achoist_aside == "none") or achoist == "skipped"):   # hoisted sample calls > 0 with no step-aside; `skipped` = FPF_HOIST's step not in force (the lever steps aside by name with the hoist)
            unserved.append(f"ATOM_COND_HOIST not applied: achoist={achoist} aside={achoist_aside}")
        if "TRIMUL_CD" in levers and not (tcd.isdigit() and int(tcd) > 0):   # installed and nothing served (every call stepped aside, or the module never bound): levers_short by name, the asides worded
            unserved.append(f"TRIMUL_CD not served: tcd={tcd} aside={tcd_aside}")
        if "LNP" in levers and not (lf["lnp"].isdigit() and int(lf["lnp"]) > 0):   # installed and nothing served (or no LNP line): levers_short by name
            unserved.append(f"LNP not served: lnp={lf['lnp']} aside={lf['lnp_aside']}")
        if "LNP" in levers and lf["lnp_aside"] != "none":                 # a provider refusal by name at a routed unit (levers_off, a stack without the row): named, the run is levers_short (never a silent stock path)
            unserved.append(f"LNP refused at {lf['lnp_aside']}")
        if "TTR" in levers and ttr_fallback != "none":                    # a pair-transition site the fused kernel did not serve, DECLARED by shape on the line: named — a step-aside by name (the stock transition ran there), never an exit code
            fell_back.append(f"TTR aside at {ttr_fallback}")
        reasons = unserved + degraded + row["short"]
        return {**ev, **fields, "ok": not reasons and not fell_back, "reported": ev["reported"] + [m.group(0)], "reason": "; ".join(reasons + fell_back) or None,
                "served_via": "traced", "softened": bool(fell_back) and not reasons, "fell_back": fell_back}   # softened: the ONLY shortfall is kernels that engaged and fell back at some call sites
    return {"ok": not row["short"], "reported": row["reported"], "reason": "; ".join(row["short"]) or None}


def lever_evidence(mode: str, lines: List[str], levers: Optional[List[str]] = None, n_gpu: int = 1,
                   region: Optional[str] = None) -> dict:
    """_lever_evidence plus the per-lever census (``per_lever``: per_lever()) — the gate's verdict and, lever by lever, on / skipped with
    the reason; one LEVER line each is printed from it (report.lever_lines). big in region fast is judged by the fast line's rules."""
    mode = effective_mode(mode, region)
    levers = levers_of(mode) if levers is None else list(levers)
    ev = _lever_evidence(mode, lines, levers, n_gpu)
    return {**ev, "per_lever": per_lever(mode, levers, ev, lines)}


# ---- the resource axis of the memory mode: --n_gpu P (opt_core.mem.ngpu is the one producer of the words; the row-sharded pair stack is
# opt_core.mem.rowpair_jax's AF3 recipe, driven in the model process by inprocess/rowpair.py under the memory mode's launcher)
ENV_N_GPU = "AF3_JAX_N_GPU"                                             # the axis variable of the model process (big_launch.ENV_N_GPU spells the same; the wrapper sets it from --n_gpu)
ROWPAIR = "ROWPAIR"                                                     # lever id of the row-sharded pair stack (registry: strategy F7.tensor_parallel)
ROWPAIR_SUPERSEDES = ("TRIMUL_CHUNK", "COND_SHARD", "LOGITS_SHARD", "GLUT", "DATTN", "LNP")   # (LNP: the recipe transcribes the pair / MSA / template stacks per row block on its mesh; the LayerNorm provider binding is a one-device lever, not composed under the mesh — off by its switch there)   # levers whose sites the recipe owns under n_gpu > 1 (off by name there, reason=rowpair_owns_site): the single-device memory levers at the triangle-multiplication / pair-conditioning / pair-logits sites (COND_SHARD's transcription is the recipe's b21) and GLUT (the recipe's triangle-multiplication body calls the library's gated linear unit on row blocks; the FlashPairformer kernels serve square pair maps only and are outside the memory mode at every P)
INPROCESS_MODULES[ROWPAIR] = "inprocess/rowpair.py"


def rowpair_mem_fraction_ceilings() -> Dict[int, float]:
    """The row-sharded adapter's XLA-pool ceilings by device count, READ from ``inprocess/rowpair.py`` ``MEM_FRACTION_CEILINGS`` (the one table;
    that module runs in the model process, so the wrapper reads its literal, never a copy): ``{8: 0.9}`` — NCCL's communicators allocate outside
    XLA's pool, so at P devices the pool fraction must leave them room."""
    import ast
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "inprocess", "rowpair.py"), encoding="utf-8").read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "MEM_FRACTION_CEILINGS" for t in node.targets):
            return {int(k): float(v) for k, v in ast.literal_eval(node.value).items()}
    raise UnsupportedMode("inprocess/rowpair.py carries no MEM_FRACTION_CEILINGS table")


def rowpair_mem_fraction_ceiling(n_gpu: int) -> Optional[float]:
    """The pool ceiling that applies at ``n_gpu`` devices: the table's entry for the largest device count <= n_gpu (inprocess/rowpair.headroom's
    rule), None below the table's first count (P = 2, 4: no ceiling)."""
    table = rowpair_mem_fraction_ceilings()
    keys = sorted(k for k in table if k <= int(n_gpu))
    return table[keys[-1]] if keys else None


def with_n_gpu(res: dict, n_gpu: int) -> dict:
    """The composition ``res`` (resolve's) on ``n_gpu`` devices: P = 1 → unchanged plus ``n_gpu: 1`` (byte-identical program: nothing of the
    row-sharded stack is imported); P > 1 → accepted only for the memory mode (opt_core.mem.ngpu.refuse_unless_big: any other mode raises
    NGpuRefused with its sentence), the levers of ROWPAIR_SUPERSEDES leave the composition by name (``superseded``: the memory levers among
    them named on the launcher's ``--off <lever,…>`` argument, a tree lever's / GLUT's switch removed from the model-process environment) and
    ROWPAIR joins it; the model process
    reads the axis from ``AF3_JAX_N_GPU`` (ENV_N_GPU) in its environment; the cache class is the line's own (cache_class_of: one class per
    lever line, whatever P — the compiled programs inside it are keyed by device kind and shape)."""
    from opt_core.mem import ngpu
    P = ngpu.refuse_unless_big(ngpu.check_n_gpu(n_gpu), res["mode"])
    out = dict(res, n_gpu=P, sharding=ngpu.sharding_value(P), superseded=[])
    if P == 1:
        return out
    if not res["launcher"]:
        raise UnsupportedMode(f"n_gpu={P}: mode {res['mode']} has no launcher to drive the row-sharded pair stack")
    out["superseded"] = [l for l in res["levers"] if l in ROWPAIR_SUPERSEDES]
    row = [l for l in res["levers"] if l in ROW_LEVER_FLAGS]
    kept = [l for l in res["levers"] if l not in ROWPAIR_SUPERSEDES and l not in ROW_LEVER_FLAGS]
    mem_first = next((i for i, l in enumerate(kept) if l in KIT_MODES[BIG]["levers"]), len(kept))
    ngpu_lv = [l for l in BIG_NGPU_LEVERS if l not in kept] if res["mode"] == BIG else []
    out["levers"] = kept[:mem_first] + ngpu_lv + kept[mem_first:] + [ROWPAIR] + row   # the row-sharded line composes BIG_NGPU_LEVERS (none at present) ahead of the line's memory levers, then ROWPAIR and its row levers
    env = dict(res["env"])
    env[ENV_N_GPU] = str(P)                                               # the launcher's reading of the axis (big_launch.n_gpu_requested): > 1 → inprocess/rowpair.py drives the script on the mesh
    mem_off = []
    for lv in out["superseded"]:
        if lv in TREE_LEVER_ENV:                                          # a tree lever leaves by its own switch (inprocess/<lever>.py ENV_SWITCH): absent → never installed in the ranks
            for k in TREE_LEVER_ENV[lv]:
                env.pop(k, None)
        elif lv == "GLUT":
            env.pop("AF3P_GLU_T", None)                                   # the Pallas add-on installs GLUT from this switch (levers_launch): absent → the library class stays for the recipe to rebind
        else:
            mem_off.append(lv.lower())                                    # a memory lever: named on the launcher's --off argument (big_launch.parse_argv), off by name on its ACTIVE line
    out["env"] = env
    off_word = list(dict.fromkeys([lv.lower() for lv in BIG_NOT_COMPOSED if lv not in ngpu_lv] + mem_off))   # ONE --off word: the not-composed levers this line still leaves off + the memory levers the recipe supersedes
    lau = list(res["launcher"])
    if OFF_ARG in lau:
        i = lau.index(OFF_ARG); del lau[i:i + 2]
    out["launcher"] = lau + ([OFF_ARG, ",".join(off_word)] if off_word else [])
    return out

# ---- the shared core's producers this package imports at activation: named before anything resolves, never a traceback
BASE_PRODUCERS = ("opt_core.gates", "opt_core.report", "opt_core.manifest", "opt_core.autoload", "opt_core.mem.ngpu")   # imported by every mode's activation
CORE_FLOOR = "0.4.1"                                                    # the first core version carrying every producer of this package (opt_core.mem.ngpu, the memory mode's opt_core.mem modules, opt_core.mem.rowpair_jax)


def required_producers() -> tuple:
    """The ONE producer table of this package: the base producers above + the memory mode's core modules (big.CORE_MODULES, the same tuple its own gate reads)."""
    from . import big as _b
    return tuple(dict.fromkeys(BASE_PRODUCERS + tuple(getattr(_b, "CORE_MODULES", ()))))



def missing_producers(names: Optional[Sequence[str]] = None) -> List[str]:
    """The producers of ``names`` (default: required_producers()) this interpreter cannot find (importlib's find_spec, parents first; nothing of the model is imported)."""
    import importlib.util
    out = []
    for n in (required_producers() if names is None else names):
        try:
            if importlib.util.find_spec(n) is None:
                out.append(n)
        except (ImportError, ValueError):                                 # a missing parent package raises instead of returning None
            out.append(n)
    return out


def producer_missing_reason(missing: Sequence[str]) -> str:
    return (f"producer_missing:{','.join(missing)} — this package imports opt_core >= {CORE_FLOOR} "
            f"(the pin: [tool.opt_core] in opt/pyproject.toml); install the pinned core beside the kit (pip install -e common/opt_core)")

ROWPAIR_INSTALLED_RX = re.compile(r"\[af3-jax-opt\] ROWPAIR installed n_gpu=(\d+) ?(.*)$")     # the adapter's install line (inprocess/rowpair.py)
ROWPAIR_REFUSED_RX = re.compile(r"\[af3-jax-opt\] ROWPAIR refused: (.*)$")                      # its refusal (exit 5)
ROWPAIR_NOTE_RX = re.compile(r"\[af3-jax-opt\] ROWPAIR NOTE (\w+): (.*)$")                      # its named notes (a memory ESTIMATE exceeded: the run proceeded) — recorded, never a reason
ROWPAIR_FAMILY_RX = re.compile(r"\[af3-jax-opt\] LEVER name=rowpair state=(\w+) ?(.*)$")         # the family's ONE lever line at exit (opt_core.mem.rowpair_jax.evidence.line)
ROWPAIR_FALLBACK_RX = re.compile(r"\b(triatt|trimul)_fallback=(\d+)")                    # the family line's per-shard kernel census (inprocess/rowpair.kernel_evidence)


def rowpair_evidence(lines: List[str], n_gpu: int) -> dict:
    """The row-sharded pair stack's evidence in a model process's transcript under ``n_gpu`` > 1: the adapter's install line, no refusal, and the
    family's lever line with ``state=on`` (it carries the mesh facts, the XLA peaks, schedule / kernel / sites). ``ok`` False names what is missing."""
    inst = [m for ln in lines for m in [ROWPAIR_INSTALLED_RX.search(ln)] if m]
    ref = [m.group(1).strip() for ln in lines for m in [ROWPAIR_REFUSED_RX.search(ln)] if m]
    fam = [m for ln in lines for m in [ROWPAIR_FAMILY_RX.search(ln)] if m]
    notes = [f"{m.group(1)}: {m.group(2).strip()}" for ln in lines for m in [ROWPAIR_NOTE_RX.search(ln)] if m]
    out = {"installed": int(inst[-1].group(1)) if inst else None, "installed_fields": inst[-1].group(2).strip() if inst else None,
           "refused": ref[-1] if ref else None, "family_state": fam[-1].group(1) if fam else None, "family_fields": fam[-1].group(2).strip() if fam else None, "notes": notes}
    if ref:
        reason = f"ROWPAIR refused: {ref[-1]}"
    elif not inst:
        reason = "ROWPAIR printed no 'ROWPAIR installed' line (the adapter did not install)"
    elif out["installed"] != int(n_gpu):
        reason = f"ROWPAIR installed n_gpu={out['installed']}, the wrapper asked {n_gpu}"
    elif not fam:
        reason = "ROWPAIR: no family LEVER name=rowpair line at exit (the process died before its evidence)"
    elif out["family_state"] != "on":
        reason = f"ROWPAIR family line state={out['family_state']} ({(out['family_fields'] or '')[:160]})"
    else:
        reason = None
    ff = out["family_fields"] or ""
    fb = {m.group(1): int(m.group(2)) for m in ROWPAIR_FALLBACK_RX.finditer(ff)}   # the per-shard kernel census: a shard the named kernel did not serve is a partial activation, by name
    out["kernel"] = (re.search(r"\bkernel=(\S+)", ff) or [None, None])[1]
    out["per_shard_fallback"] = {k: v for k, v in fb.items() if v}
    if reason is None and out["per_shard_fallback"]:
        reason = "ROWPAIR per-shard fallback: " + " ".join(f"{k}_fallback={v}" for k, v in sorted(out["per_shard_fallback"].items())) + f" (kernel={out['kernel']})"
    return {**out, "ok": reason is None, "reason": reason}

LAUNCHER_NGPU_RX = re.compile(r"\[af3-jax-opt\] ACTIVE mode=big:\S* .*?\bn_gpu=(\d+)")          # the memory mode LAUNCHER's ACTIVE line (big_launch.py: `mode=big:<levers> ... n_gpu=P`) reports the axis it read — not the wrapper's own ACTIVE line


def n_gpu_evidence(lines: List[str], requested: int, rowpair_expected: bool) -> dict:
    """FAIL-CLOSED account of the resource axis in a model process's transcript: the n_gpu the launcher REPORTS (its ACTIVE line) must equal the
    wrapper's request, and the row-sharded stack's install line must be present exactly when P > 1 — `n_gpu_mismatch requested=P active=Q`
    otherwise (a dropped or invented axis is never a pass)."""
    act = [int(m.group(1)) for ln in lines for m in [LAUNCHER_NGPU_RX.search(ln)] if m]
    inst = [int(m.group(1)) for ln in lines for m in [ROWPAIR_INSTALLED_RX.search(ln)] if m]
    active = act[-1] if act else None
    if active is None:
        reason = f"n_gpu_mismatch requested={requested} active=unreported (the launcher's ACTIVE line carries no n_gpu)"
    elif active != int(requested):
        reason = f"n_gpu_mismatch requested={requested} active={active}"
    elif int(requested) == 1 and inst:
        reason = f"n_gpu_mismatch requested=1 active=1 but ROWPAIR installed n_gpu={inst[-1]}"
    elif int(requested) > 1 and not rowpair_expected:
        reason = f"n_gpu_mismatch requested={requested}: the composition carries no ROWPAIR"
    else:
        reason = None
    return {"requested": int(requested), "active": active, "installed": inst[-1] if inst else None, "ok": reason is None, "reason": reason}

EVIDENCE_RXS = tuple(EVIDENCE_RXS) + (ROWPAIR_INSTALLED_RX, ROWPAIR_REFUSED_RX, ROWPAIR_FAMILY_RX, re.compile(r"\[af3-jax-opt\] ROWPAIR runner bound:"))   # the row-sharded stack's lines are evidence lines (pred/warm collect them)
