"""The mode table, and how a package mode resolves to the kit driver's own flags.

The kit states its compositions machine-readably in ``MANIFEST.json`` ``levers`` — ``exact_default_stack = "-fast (= -host_outputs
-device_params -sort_by_length) + -precompile N + -program_cache DIR + -prefetch N + -overlap_output N"``, ``tier2_opt_in``, ``tier2_env``,
``memory_line``, ``memory_opt_in``, ``deployment``. This module never transcribes a flag:
``kit_levers()`` reads that block and ``resolve()`` returns the flags to append to the driver's line — the kit's own tokens with the lever
values substituted (the thread count of ``-precompile``, ``PRECOMPILE_THREADS``; the sub-batch rows, ``SUBBATCH_ROWS``; the memory
line's chunk rule, ``TRIMUL_ROWS`` / ``TRIMUL_MIN_RESIDUES``).

Package modes (``KIT_MODES``, the one place to change what a mode means; ``MODES`` / ``DEFAULT_MODE`` are the one list and the one
default every command reads):
  exact   ``-fast`` (= L6 + L1 + U1, the driver's preset) + ``-precompile N`` (L7: the programs of the run's distinct lengths prepared
          ahead of the loop by jax's AOT path on N host threads, N = PRECOMPILE_THREADS or ``--precompile N``; no forward is run) +
          ``-program_cache <dir>`` (L13: the programs kept across processes). Its promise is the stock line's bytes under the deterministic recipe (det.py).
  fast    exact + ``-subbatch 128 -flash_attn -fused_triattn -fused_trimul -opm_reassoc`` (L9, L8, L10, L11, L12) + ``-tmpl_pointwise_sub 8192``
          (L18) + the bf16 TriangleAttention word (L19, environment): Tier-2, not bitwise; the default word; ``--precompile [N]`` as for exact.
  big   fast's composition without L18 + the memory line: the XLA pool fraction 0.95 in the driver child's variables (big.py); one GPU. The
          row-chunked TriangleMultiplication ``-trimul_chunk <rows>:<min residues>`` is an OPT-IN of big (``AF2IG_OPT_TRIMUL_CHUNK``), composed by no
          shipped mode: it costs memory and time at every size.
  off     stock: the same driver with no lever flag, through the stock caller in a clean subprocess (stock_cli.py).
Every kit mode also carries the deployment lever ``ccache`` (MANIFEST.json ``levers.deployment`` = ``JAX_COMPILATION_CACHE_DIR``; ccache.py places
JAX's persistent compilation cache in the driver child's environment — no argv token, no numerics; the stock line never carries it).
``MODEL_OPT_LEVERS_OFF=<ids>`` (the tree's uniform ablation switch, ``levers_off``) drops the named levers from the composition: their argv tokens
leave the line (a lever of the ``-fast`` preset dropped spells the remaining preset levers by their own flags), environment levers are not placed;
ids unknown or outside the mode are returned as ignored and named on the activation line — never an error, never a partial run.
"""
from __future__ import annotations

import json
import os
import re
import shlex
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import registry

ENV = "AF2IG_OPT"
ENV_JIT_ROOT, ENV_JIT_ROOT_COMMON = ENV + "_JIT_ROOT", "MODEL_OPT_JIT_ROOT"   # the ccache lever's root: the kit's own word (configs/<card>.env gives it a default), then the tree's cross-kit word; ccache.py says which one placed the cache
ENV_TIER = ENV + "_TIER"                                          # the provider's TIER WORD placed in the kit-mode driver child (fast | big; exact binds no kernel): the three kernel adapters hand it to opt_core's provider, which serves the fastest measured row per cell
F32_PRODUCTS = "tf32"                                             # the float32 product class of the tolerance tiers (fast / big), passed ONCE from here as the provider's input_precision word — no per-adapter precision constants
TIER_WORDS = ("fast", "big")
ENV_LEVERS_OFF = registry.LEVERS_OFF_ENV                          # MODEL_OPT_LEVERS_OFF=<ids>: the ablation switch (resolve levers_off)
DEPLOYMENT_SWITCH = "JAX_COMPILATION_CACHE_DIR"                   # MANIFEST.json levers.deployment: the one environment name the ccache lever sets in the driver child
FLAG_PREFETCH, PREFETCH_DEPTH = "-prefetch", 2                     # L15, every kit mode: the inputs of the next PREFETCH_DEPTH designs featurised ahead of their turn on one worker thread (exact-class; host RAM: <= depth featurised inputs queued)
FLAG_OVERLAP, OVERLAP_DEPTH = "-overlap_output", 1                  # L16, every kit mode: each design's output step on one writer thread behind the loop (exact-class; host RAM: <= depth+1 outputs alive)
FLAG_PROGRAMS, PROGRAMS_DIR_TOKEN = "-program_cache", "DIR"       # L13: the flag and the placeholder the statement carries; stack.activate substitutes the directory (ccache.programs_dir)
PRECOMPILE_THREADS = 6                                         # N of -precompile N under a bare --precompile (~3 GB host RAM per compiling thread)
SUBBATCH_ROWS = 128                                             # L9: rows per inference sub-batch chunk on the fast line (-subbatch; stock 4): the largest per-chunk intermediate, the attention
                                                                #     logits [rows, 4, N, N] f32, is 2.0 GB at N=1000 and 4.4 GB at N=1472 — finite by construction (never unchunked)
MODES: Tuple[str, ...] = ("off", "exact", "fast", "big")
DEFAULT_MODE = "fast"                                            # the default word
FOLDED: Dict[str, str] = {}                               # {word: mode} makes `--mode <word>` resolve by name to <mode>'s line (ACTIVE says mode=<word>→<mode>);
                                                                 # empty: every mode word is its own line (L18's rows differ between fast and big)
KIT_MANIFEST_RELPATH = "MANIFEST.json"
DRIVER_RELPATH = os.path.join("af2_initial_guess", "predict_pdb.py")   # under the checkout ($AF2IG_DIR's parent)
FLAG_TMPL = "-tmpl_pointwise_sub"                                # L18: the template point-wise attention sub-batch rows (patch 15)
TMPL_ROWS: Dict[str, Optional[int]] = {"fast": 8192, "big": None}   # L18 rows per tier: fast 8192 (t_model -5.1/-4.4/-4.6 % at 400/800/1200, peak +0/+0.30/+0.68 GiB); big: None = stock 128 (lever absent) unless a peak-neutral setting is measured; AF2IG_OPT_TMPL_POINTWISE_SUB=<rows>|off overrides per process
ENV_TMPL = ENV + "_TMPL_POINTWISE_SUB"
FLAG_PRECOMPILE, FLAG_SUBBATCH, FLAG_FLASH, FLAG_FUSED, FLAG_FTRIMUL, FLAG_OPM, FLAG_PRESET = "-precompile", "-subbatch", "-flash_attn", "-fused_triattn", "-fused_trimul", "-opm_reassoc", registry.PRESET_FLAG
PRESET_LEVERS = (registry.L6, registry.L1, registry.U1)       # what -fast expands to (predict_pdb.py:109-110), in the driver's order


@dataclass(frozen=True)
class KitMode:
    name: str
    what: str
    determinism: str = ""                                     # the mode's determinism statement: what its outputs promise against the stock line, and under which switch


KIT_MODES: Dict[str, KitMode] = {
    "exact": KitMode("exact", "-fast alone (the driver's preset: L6 + L1 + U1); the MANIFEST's -precompile N is the opt-in --precompile [N]",
                     "bitwise with the stock line under --det 1 (XLA_FLAGS=--xla_gpu_autotune_level=0 on both arms); at default numerics stock does not reproduce itself"),
    "fast": KitMode("fast", "exact + -subbatch 128 (L9) + -flash_attn (L8) + -fused_triattn (L10) + -fused_trimul (L11) + -opm_reassoc (L12) + -tmpl_pointwise_sub 8192 (L18) + the bf16 attention-core word (L19, 0.7.6; environment); since 0.7.4 its own line again (un-folded: L18's rows differ per tier); one GPU.",
                    "Tier 2: padded attention and LayerNorm reductions, a re-associated outer-product mean, the template point-wise attention at another batch width — not bitwise; the seed-spread band is its word"),
    "big": KitMode("big", "the memory mode: fast's composition + XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 in the driver child (the reach lever); the row-chunked TriangleMultiplication is an opt-in (AF2IG_OPT_TRIMUL_CHUNK) no shipped mode composes since 0.7.1",
                     "fast's band (the composition it builds on)"),
}
STOCK_DETERMINISM = "the stock arm: reproduces itself only under --det 1"
DETERMINISM: Dict[str, str] = {"off": STOCK_DETERMINISM, **{m: km.determinism for m, km in KIT_MODES.items()}}   # one row per mode of MODES 


@dataclass
class Resolution:
    mode: str
    flags: List[str]                                          # the lever flags appended to the driver's line (none for off)
    levers: List[str] = field(default_factory=list)           # registry ids the flags engage
    precompile: Optional[int] = None
    subbatch: Optional[int] = None                            # L9: rows per inference sub-batch chunk (fast line), None = stock
    source: str = ""                                          # the kit statement the flags come from
    levers_off: List[str] = field(default_factory=list)       # MODEL_OPT_LEVERS_OFF ids dropped from this mode's composition (in the composition's order)
    levers_off_ignored: List[str] = field(default_factory=list)   # MODEL_OPT_LEVERS_OFF ids that are not levers of this mode (unknown, or another mode's): named, ignored
    tmpl_rows: Optional[int] = None                            # L18's rows on this line (None = lever absent: stock 128)
    core_dtype: Optional[str] = None                           # L19: 'bf16' when the line carries L19 (the child's AF2IG_OPT_TRIATTN_CORE_DTYPE), else None (float32 core)
    folded_into: Optional[str] = None                          # the mode whose line this word resolved to (FOLDED: fast → big); None = its own line

    def describe(self) -> str:
        return " ".join(shlex.quote(a) for a in self.flags) or "none"


def kit_levers(kit_home: str) -> dict:
    """The kit's own composition block: MANIFEST.json ``levers`` (exact_default_stack, tier2_opt_in, memory_line)."""
    with open(os.path.join(kit_home, KIT_MANIFEST_RELPATH), "r", encoding="utf-8") as fh:
        lv = json.load(fh)["levers"]
    for k in ("exact_default_stack", "tier2_opt_in"):
        if k not in lv:
            raise ValueError(f"{KIT_MANIFEST_RELPATH}: levers.{k} missing")
    return lv


def parse_exact_stack(text: str) -> dict:
    """``"-fast (= -host_outputs -device_params -sort_by_length) + -precompile N + -program_cache DIR"`` ->
    {preset: "-fast", preset_expands_to: [...], extra: ["-precompile", "N"] (the first extra), extras: [["-precompile", "N"], ["-program_cache", "DIR"]]}."""
    m = re.match(r"^\s*(\S+)\s*\(=\s*([^)]*)\)\s*((?:\+\s*\S+\s+\S+\s*)*)$", text)
    if not m:
        raise ValueError(f"MANIFEST levers.exact_default_stack unreadable: {text!r}")
    extras = [e.split() for e in re.findall(r"\+\s*(\S+\s+\S+)", m.group(3) or "")]
    return {"preset": m.group(1), "preset_expands_to": m.group(2).split(), "extra": extras[0] if extras else [], "extras": extras}
def levers_off_from(environ: Optional[dict] = None) -> List[str]:
    """``MODEL_OPT_LEVERS_OFF``: the comma- or blank-separated lever ids to drop from the mode's composition (registry ids: L8, ccache, mem_fraction …); [] when unset."""
    environ = os.environ if environ is None else environ
    out: List[str] = []
    for t in re.split(r"[,\s]+", environ.get(ENV_LEVERS_OFF) or ""):
        if t and t not in out:
            out.append(t)
    return out


def precompile_threads() -> Tuple[int, str]:
    """N of -precompile N under a bare --precompile: PRECOMPILE_THREADS (an explicit --precompile N names its own)."""
    return PRECOMPILE_THREADS, "modes.PRECOMPILE_THREADS"


FLAG_TRIMUL = "-trimul_chunk"                                      # the memory line's driver flag (patch 06)
TRIMUL_ROWS = 256                                                 # rows of the pair representation per TriangleMultiplication chunk (the memory mode's tested value)
TRIMUL_MIN_RESIDUES = 1473                                        # engage at compiled lengths >= this — above the speed ladder's range (200..1472): those lengths keep the stock TriangleMultiplication body
ENV_TRIMUL = ENV + "_TRIMUL_CHUNK"                                   # the opt-in word of the row-chunked TriangleMultiplication under big: `on` | `1` = the table value (TRIMUL_ROWS:TRIMUL_MIN_RESIDUES), `<rows>:<min residues>` = that, unset | `off` | `0` = not composed
ENV_MEM_FRACTION, MEM_FRACTION = "XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95"   # the XLA client's pool under the memory mode (stock default 0.75), set in the driver child's environment
def trimul_flag_value() -> str:
    return f"{TRIMUL_ROWS}:{TRIMUL_MIN_RESIDUES}"


def tmpl_rows(mode: str, environ=None) -> Optional[int]:
    """L18's rows for `mode` (fast | big): AF2IG_OPT_TMPL_POINTWISE_SUB=<rows> overrides the tier's table value (TMPL_ROWS), `off`/`0` removes the
    lever, unset = the table; other modes carry no L18. ValueError names a bad word."""
    if mode not in ("fast", "big"):
        return None
    environ = os.environ if environ is None else environ
    v = (environ.get(ENV_TMPL) or "").strip().lower()
    if v in ("", "table", "default"):
        return TMPL_ROWS.get(mode)
    if v in ("off", "0", "stock"):
        return None
    if v.isdigit() and int(v) > 0:
        return int(v)
    raise ValueError(f"{ENV_TMPL}={environ.get(ENV_TMPL)!r}: expected <rows> | off")


def trimul_opt_in(environ=None) -> Optional[str]:
    """The -trimul_chunk value big composes when AF2IG_OPT_TRIMUL_CHUNK opts the lever in: `on`/`1`/`table` = the table value, `<rows>:<min residues>` = that
    (two positive integers), unset/`off`/`0`/empty = None (not composed). A malformed value raises ValueError (a usage error, named)."""
    environ = os.environ if environ is None else environ
    v = (environ.get(ENV_TRIMUL) or "").strip().lower()
    if v in ("", "off", "0", "no", "false"):
        return None
    if v in ("on", "1", "yes", "true", "table"):
        return trimul_flag_value()
    m = re.fullmatch(r"(\d+):(\d+)", v)
    if not m or int(m.group(1)) <= 0 or int(m.group(2)) <= 0:
        raise ValueError(f"{ENV_TRIMUL}={environ.get(ENV_TRIMUL)!r}: expected on | off | <rows>:<min residues>")
    return f"{int(m.group(1))}:{int(m.group(2))}"


def resolve(mode: str, kit_home: str, precompile: Optional[int] = None, levers_off=(), trimul: Optional[str] = None, tmpl: Optional[int] = -1) -> Resolution:
    """The driver flags for `mode`, composed from the kit's own statement with the lever values substituted (the thread
    count of `-precompile`, L7, composed on every kit mode: `precompile` = N from `--precompile N`; None or 0 = the default N, PRECOMPILE_THREADS;
    the fast line = exact + the sub-batch raise (L9) + the flash-attention core (L8) + the fused triangle-attention block (L10) + the fused triangle-multiplication block (L11) + the re-associated outer-product mean (L12): fast, and big, which builds on it;
    the deployment lever ccache on every kit mode), minus the levers `levers_off` names (MODEL_OPT_LEVERS_OFF, `levers_off_from`): their argv tokens are dropped —
    a lever of the `-fast` preset dropped spells the remaining preset levers by their own flags, in the driver's order — and environment levers (mem_fraction, ccache)
    are not placed; ids that are not levers of this mode are returned in `levers_off_ignored`. `trimul` (big only): the opt-in
    `-trimul_chunk <rows>:<min residues>` value (`trimul_opt_in`) — None = not composed (the shipped big)."""
    if mode == "off":
        if precompile is not None:
            raise ValueError("--precompile is a lever of the kit modes (exact | fast), not of stock")
        return Resolution("off", [], [], source="stock: the driver with no lever flag (STOCK.md stock line)")
    if mode not in KIT_MODES:
        raise ValueError(f"unknown mode {mode!r} (expected {'|'.join(MODES)})")
    word, mode = mode, FOLDED.get(mode, mode)               # `fast` resolves by name to the big line (one composed line); the Resolution keeps the word given and names the fold
    lv = kit_levers(kit_home)
    ex = parse_exact_stack(lv["exact_default_stack"])
    segs: List[Tuple[Tuple[str, ...], List[str]]] = [(tuple(PRESET_LEVERS), [ex["preset"]])]   # (levers, argv tokens) in the driver's order: the preset token carries three levers
    if [e[0] for e in ex["extras"]] != [FLAG_PRECOMPILE, FLAG_PROGRAMS, FLAG_PREFETCH, FLAG_OVERLAP]:
        raise ValueError(f"MANIFEST levers.exact_default_stack does not read '+ {FLAG_PRECOMPILE} N + {FLAG_PROGRAMS} {PROGRAMS_DIR_TOKEN} + {FLAG_PREFETCH} N + {FLAG_OVERLAP} N': {ex['extras']}")
    n = precompile if (precompile is not None and precompile > 0) else precompile_threads()[0]   # L7, every kit mode: the programs of the run's distinct lengths prepared ahead of the loop on N host threads (AOT: no forward is run); --precompile N names N
    segs.append(((registry.L7,), [FLAG_PRECOMPILE, str(n)]))
    padded = mode in ("fast", "big")                                # the fast line: fast's tier-2 levers, and big's on them
    if padded:
        tier2 = list(lv["tier2_opt_in"])
        for want in (FLAG_SUBBATCH, FLAG_FLASH, FLAG_FUSED, FLAG_FTRIMUL, FLAG_OPM, FLAG_TMPL):
            if want not in tier2:
                raise ValueError(f"MANIFEST levers.tier2_opt_in does not list {want}: {tier2}")
        segs += [((registry.L9,), [FLAG_SUBBATCH, str(SUBBATCH_ROWS)]),      # L9: the inference sub-batch raise
                 ((registry.L8,), [FLAG_FLASH]),                             # L8: the attention core of the pair-biased Attention calls on the shared Pallas flash kernel (TF32-class, as the stock einsums)
                 ((registry.L10,), [FLAG_FUSED]),                            # L10: TriangleAttention start/end on the shared fused block (with it, L8's kernel serves the MSA row attention)
                 ((registry.L11,), [FLAG_FTRIMUL]),                           # L11: TriangleMultiplication out/in on the shared fused block
                 ((registry.L12,), [FLAG_OPM])]                              # L12: OuterProductMean re-associated (pure JAX; lowers the per-call working set, so big keeps it)
        rows = TMPL_ROWS.get(mode) if tmpl == -1 else tmpl           # L18: the template point-wise attention rows of THIS tier (fast 8192; big none, unless the env word names rows); None = stock 128, lever absent
        if rows:
            segs.append(((registry.L18,), [FLAG_TMPL, str(int(rows))]))
        segs.append(((registry.L19,), []))                 # L19 , fast and big: the bridge core's bf16 operand word in the child environment (registry.L19_ENV=L19_WORD) — no argv token; peak equal or lower
    if mode == "big":                                     # the memory line: the XLA pool fraction in the child environment (big.child_env); the row-chunked TriangleMultiplication only when opted in (0.7.1: a measured memory and speed COST at every size — MANIFEST levers.memory_opt_in)
        memory = list(lv.get("memory_line") or []); opt_in = list(lv.get("memory_opt_in") or [])
        if memory != [ENV_MEM_FRACTION] or opt_in != [FLAG_TRIMUL]:
            raise ValueError(f"MANIFEST levers.memory_line / memory_opt_in do not read [{ENV_MEM_FRACTION}] / [{FLAG_TRIMUL}]: {memory} / {opt_in}")
        if trimul:
            segs.append(((registry.TRIMUL,), [FLAG_TRIMUL, trimul]))
        segs.append(((registry.MEMF,), []))                 # mem_fraction: an environment lever, no argv token
    segs.append((("L13",), [FLAG_PROGRAMS, PROGRAMS_DIR_TOKEN]))   # L13, every kit mode: the serialized-program store; the directory is substituted at activation (ccache.programs_dir) or the lever steps aside by name
    segs.append(((registry.L15,), [FLAG_PREFETCH, str(PREFETCH_DEPTH)]))   # L15, every kit mode (exact-class): the next PREFETCH_DEPTH inputs featurised ahead of their turn on one worker thread; host RAM only, so big keeps it
    segs.append(((registry.L16,), [FLAG_OVERLAP, str(OVERLAP_DEPTH)]))     # L16, every kit mode (exact-class): each design's output step on one writer thread behind the loop; host RAM only, so big keeps it
    deployment = list(lv.get("deployment") or [])           # the deployment lever of every kit mode (ccache): an environment lever of the driver child (ccache.place), no argv token
    if deployment != [DEPLOYMENT_SWITCH]:
        raise ValueError(f"MANIFEST levers.deployment does not read [{DEPLOYMENT_SWITCH}]: {deployment}")
    segs.append(((registry.CCACHE,), []))
    composition = [l for lvs, _ in segs for l in lvs]
    off = [l for l in levers_off if l in composition]
    ignored = [l for l in levers_off if l not in composition]
    flags: List[str] = []
    levers: List[str] = []
    for lvs, toks in segs:
        keep = [l for l in lvs if l not in off]
        if not keep:
            continue
        if len(keep) < len(lvs):                            # part of the -fast preset dropped: the remaining preset levers by their own flags, in the driver's order
            toks = [registry.LEVERS[l].flag for l in keep]
        flags += toks
        levers += keep
    n_run = n if registry.L7 in levers else 0
    res = Resolution(word, flags, levers, precompile=n_run, subbatch=SUBBATCH_ROWS if registry.L9 in levers else None, tmpl_rows=(int(flags[flags.index(FLAG_TMPL) + 1]) if FLAG_TMPL in flags else None), core_dtype=(registry.L19_WORD if registry.L19 in levers else None),
                      source=f"{KIT_MANIFEST_RELPATH} levers.exact_default_stack (-precompile {n_run}, -program_cache <jit root>/<stack key>/programs, -prefetch {PREFETCH_DEPTH}, -overlap_output {OVERLAP_DEPTH})" + (f" + levers.tier2_opt_in {FLAG_SUBBATCH} {FLAG_FLASH} {FLAG_FUSED} {FLAG_FTRIMUL} {FLAG_OPM}" if padded else "")
                      + ((" + levers.memory_line" + (f" + levers.memory_opt_in {FLAG_TRIMUL} {trimul} ({ENV_TRIMUL})" if trimul else "")) if mode == "big" else "") + " + levers.deployment"
                      + (f" minus {ENV_LEVERS_OFF}={','.join(off)}" if off else "") + (f" (the word {word} folded into {mode}, 0.7.2)" if word != mode else ""),
                      levers_off=off, levers_off_ignored=ignored)
    res.folded_into = mode if word != mode else None             # a folded word: ACTIVE prints mode=<word>→<mode>
    return res



def tier_word(environ=None) -> str:
    """The provider tier word of this process (``AF2IG_OPT_TIER``: fast | big); unset or unknown = ``fast`` (a driver run by hand with a kernel flag)."""
    environ = os.environ if environ is None else environ
    w = str(environ.get(ENV_TIER, "") or "").strip().lower()
    return w if w in TIER_WORDS else "fast"
