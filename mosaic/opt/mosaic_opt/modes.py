"""Modes, and how a package mode resolves to the kit's activation row.

An activation row is one arm line of the kit's design procedure, ``run_arm TAG "ENV ASSIGNMENTS" ARGS...``: the environment exported before
the driver starts and the driver flags that follow `--tag TAG --out OUTDIR`, run from the kit's `tools/` (driver.py). ``ROWS`` carries the
rows as data, verbatim; ``resolve()`` turns a package mode into the environment and the driver flags of its row by *parsing the row*
(``parse_row``: shell assignments before the driver arguments, the `$C1` / `$F1` / `$FSHA` placeholders substituted); no environment
variable name, flag or value is transcribed anywhere else. `MOSAIC_CACHE_DIR` is exported for every arm, and `PYTHONUNBUFFERED=1` under `--det 1` (det.py); the JAX allocator is
the library's default on every arm.

Words and modes (``WORDS`` — the kit's vocabulary in preference order: the shared core's `opt_core.modes` OFF / EXACT / FAST plus `big`,
the memory tier; ``MODES`` — the words SERVED, i.e. those with a row; ``NOT_WIRED`` — the tier words named and refused):
  fast    the fast-class per-step levers (registry ``klass`` fast: same error class as stock at fixed
          states, run-to-run bitwise in one process, never claimed bitwise against stock) — row `T_fast` (``KIT_MODES["fast"]["row"]``).
  big   the memory tier (registry ``klass`` big: makes a token count fit the card; numerics fast-class) — row `U_big`.
  exact   P1 + P2 + P3 — the warm row `C_p1warm_p2`: the persistent compilation cache + autotune load (P1, environment), the fast
          weight load (P2, `--weights fastinit`) and the frozen features (P3, `--features-in NPZ --features-sha SHA`); populated once per
          (image, GPU type, shape) by the populate row `B_p1populate_p2` (`warm`). Same numerics as stock; the design of the populating
          process is reproduced bitwise in every later process of the same (image, GPU type).
  off     stock: the driver with every lever off in a clean subprocess (stock_design.py): the stock arm `A_stock1` (in-process
          featurization), or `D_stock2` when the caller passes frozen features.
A tier word whose row composes no per-step lever is not in MODES and is REFUSED BY NAME (``unknown_mode_message``).

`jit_cache_key()` names the (stack, GPU type) the P1 files belong to: `MODEL_OPT_STACK_KEY` in `configs/<gpu>.env`; the P1 files live under
`$MOSAIC_OPT_CACHE_ROOT/<shape>/`, one root per key (per image + GPU type; the key names jax,
jaxlib, the CUDA plugin and the GPU product).
"""
import os
import shlex
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .registry import LEVERS, WIRED

WORDS: Tuple[str, ...] = ("fast", "exact", "big", "off")     # the tier vocabulary of the kit, in preference order (`opt_core.modes` OFF / EXACT / FAST + `big`); MODES (below) = the words with a row
TIER_WORDS: Tuple[str, ...] = ("fast", "big")                 # the words whose row exists only when a per-step lever is wired into them
DEFAULT_MODE = "fast"    # a value, never a rule computed from the mode table: the package default is fast wherever one ships
DRIVER_RELPATH = os.path.join("tools", "public_design_run.py")
AUTOTUNE_FILENAME = "xla_autotune_results.pb"                   # the row's file name (`$C1/xla_autotune_results.pb`) = opt_core.jax_design.pcc's, copied import-free (a stock process imports this module); locked by tests/test_core_adoption.py

# The activation rows, verbatim `run_arm` lines. A mode names a row by its tag.
ROWS: Dict[str, dict] = {
    "B_p1populate_p2": {"phase": "populate", "text":
        'run_arm B_p1populate_p2 "JAX_COMPILATION_CACHE_DIR=$C1 JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0 XLA_FLAGS=--xla_gpu_dump_autotune_results_to=$C1/xla_autotune_results.pb" --seed 0 --weights fastinit --features-in "$F1" --features-sha "$FSHA"'},
    "C_p1warm_p2": {"phase": "warm", "text":
        'run_arm C_p1warm_p2 "JAX_COMPILATION_CACHE_DIR=$C1 JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0 XLA_FLAGS=--xla_gpu_load_autotune_results_from=$C1/xla_autotune_results.pb" --seed 0 --weights fastinit --features-in "$F1" --features-sha "$FSHA"'},
    "A_stock1": {"phase": "stock", "text": 'run_arm A_stock1 "" --seed 0 --features-out "$F1"'},
    "D_stock2": {"phase": "stock", "text": 'run_arm D_stock2 "" --seed 0 --features-in "$F1" --features-sha "$FSHA"'},
    "T_fast": {"phase": "cache", "text":                        # the fast tier's row: P1 transparent (the compilation cache of the shape, no autotune pin: `$C1` = <root>/<shape>/xla_cache_fast, dropped BY NAME when MOSAIC_OPT_CACHE_ROOT is unset), P2, then its per-step levers by the ONE flag; the featurization in process (as A_stock1)
        'run_arm T_fast "JAX_COMPILATION_CACHE_DIR=$C1 JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0" --seed 0 --weights fastinit --levers fast'},
    "U_big": {"phase": "cache", "text":                       # the big tier's row: the fast row's environment and P2 (`$C1` = <root>/<shape>/xla_cache_big), the fast tier's levers then P5 memlevers by the ONE flag, featurizing in process like A
        'run_arm U_big "JAX_COMPILATION_CACHE_DIR=$C1 JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0" --seed 0 --weights fastinit --levers big'},
}
KIT_MODES: Dict[str, dict] = {                                  # word -> (levers, the row, the populate row it needs); a tier word's row is None until a per-step lever is stitched into it
    "fast": {"levers": ("P1", "P2", "E1", "P6", "K1", "E10", "F6", "F8", "F9", "P7"), "row": "T_fast", "specs": {"K1": "fast", "P7": "pf+msa", "F8": "fast"}, "populate": None, "klass": "fast",  # the fast tier: P1 transparent + P2 (the exact row's warm-free levers: TIER_QOL), then E1 dead-template skip, P6 precision (diffusion=high), K1 triangle attention through the shared core's JAX-family provider at the row's tier word (`fast`: the fastest measured row per call), E10 sampler layer unroll, F6 channel-major triangle multiplication (cmajor), F8 triangle multiplication through the shared core's JAX-family provider at the row's tier word (bfloat16 and float32 calls, forward-only and differentiated), P7 the trunk pairformer's pair track on bf16 operands (pinned `pf+msa`); featurizes in process like row A
             "promise": "same error class as stock at fixed states; run-to-run bitwise in one process"},
    "exact": {"levers": ("P1", "P2", "P3"), "row": "C_p1warm_p2", "specs": {}, "populate": "B_p1populate_p2", "klass": "exact",
              "promise": "bitwise: the design of the populating process reproduced in every later process of the same (image, GPU type)"},
    "big": {"levers": ("P1", "P2", "E1", "P6", "K1", "E10", "F6", "F8", "F9", "P5", "P7"), "row": "U_big", "specs": {"K1": "big", "P5": "pf8+sub", "P7": "pf+msa", "F8": "big"}, "populate": None, "klass": "big",   # the memory tier: the fast tier's levers at the fast tier's settings, then P5 memlevers pinned at `pf8+sub` (the lever's default setting; the memory schedule at every input size); featurizes in process like row A
              "promise": "a token count that does not fit the card under stock fits; numerics fast-class unless shown bitwise"},
    "off": {"levers": (), "row": "A_stock1", "specs": {}, "populate": None, "features_row": "D_stock2", "klass": None,   # D: the stock arm on frozen features
            "promise": "stock"},
}
MODES: Tuple[str, ...] = tuple(w for w in WORDS if KIT_MODES[w]["row"] is not None)   # the SERVED mode words: a tier word is a mode only once a per-step lever is wired into its row (levers.MODES is this tuple)
NOT_WIRED: Tuple[str, ...] = tuple(w for w in TIER_WORDS if w not in MODES)           # the tier words the kit names and refuses (CHANGES.md "What is not wired")
assert set(WORDS) == set(KIT_MODES) and DEFAULT_MODE in MODES and "off" in MODES
assert set(l for km in KIT_MODES.values() for l in km["levers"]) == set(WIRED)          # the modes compose exactly the wired levers (registry.py `wired`)
assert all(bool([l for l in KIT_MODES[w]["levers"] if LEVERS[l].route == "install"]) == (w in MODES) for w in TIER_WORDS)   # a tier row exists iff a per-step lever is wired into it
assert all(set(KIT_MODES[w].get("specs", {})) <= {l for l in KIT_MODES[w]["levers"] if LEVERS[l].route == "install"} and all(isinstance(v, str) and v for v in KIT_MODES[w].get("specs", {}).values()) for w in KIT_MODES)   # a pinned setting names a per-step lever of its own row, as a non-empty spec word
TIER_QOL: Tuple[str, ...] = tuple(l for l in KIT_MODES["exact"]["levers"]                # ("P1", "P2"): the exact row's qol levers a tier row carries too (exact ⊂ fast ⊂ big) — every one that needs
                             if LEVERS[l].klass == "qol" and "$F1" not in (LEVERS[l].flag or ""))   # no warm step; P3 reads the frozen features `warm` writes (`$F1`) and stays exact-only
BIG_EXCLUDES: Tuple[str, ...] = ()                            # levers of the fast row the big row leaves out, each stated in CHANGES.md (none on this version)
BIG_SPECS_DIFFER: Tuple[str, ...] = ()                         # fast-row levers the big row runs at another setting, each stated in CHANGES.md (none on this version)
TIER_FOLLOWING: Tuple[str, ...] = ("K1", "F8")                    # levers whose pinned spec IS the row's tier word (the shared core's provider picks the implementation per call by that word): fast pins `fast`, big `big` — K1 triangle attention, F8 triangle multiplication
assert all(LEVERS[l].route == "install" or l in TIER_QOL for w in TIER_WORDS for l in KIT_MODES[w]["levers"])   # a tier row composes per-step levers plus the warm-free qol levers only (P3 never inside a tier row)
assert all(set(TIER_QOL) <= set(KIT_MODES[w]["levers"]) for w in TIER_WORDS if w in MODES)      # exact ⊂ fast|big: every warm-free lever of the exact row rides every tier row
assert "fast" not in MODES or "big" not in MODES or set(KIT_MODES["fast"]["levers"]) - set(BIG_EXCLUDES) <= set(KIT_MODES["big"]["levers"])   # fast ⊂ big but for the named, measured exclusions
assert "fast" not in MODES or "big" not in MODES or all(KIT_MODES["big"]["specs"].get(l) == v for l, v in KIT_MODES["fast"]["specs"].items() if l not in BIG_EXCLUDES + BIG_SPECS_DIFFER + TIER_FOLLOWING)   # a fast lever rides the big row at the fast row's pinned setting but for the named differences
assert all(KIT_MODES[w]["specs"].get(l) == KIT_MODES[w]["klass"] for w in TIER_WORDS if w in MODES for l in TIER_FOLLOWING if l in KIT_MODES[w]["levers"])   # a tier-following lever's pinned spec is the row's own tier word, stated per row


def not_wired_reason(word: str) -> str:
    """The refusal for a tier word with no row (cli / hook / enable print it after `unknown mode`)."""
    return (f"{word!r} is a tier word of this kit line with no {KIT_MODES[word]['klass']}-class lever wired into it "
            f"(mosaic/CHANGES.md \"What is not wired\"); the modes served: {', '.join(MODES)}")


def unknown_mode_message(value: str) -> str:
    v = (value or "").strip().lower()
    if v in NOT_WIRED:
        return f"unknown mode {value!r}: {not_wired_reason(v)}"
    return f"unknown mode {value!r}; expected one of {MODES}"


def install_levers_of(mode: str) -> Tuple[str, ...]:
    """The mode's per-step levers (registry route `install`) in row order: what `levers.install(mode)` applies in-process."""
    return tuple(l for l in KIT_MODES[mode]["levers"] if LEVERS[l].route == "install")


def levers_words(mode: str) -> str:
    """The mode's levers as CHANGES.md spells them: ids joined by ` + `, a pinned setting after its id as `` (`SPEC`)``."""
    sp = KIT_MODES[mode].get("specs", {})
    return " + ".join(l + (f" (`{sp[l]}`)" if l in sp else "") for l in KIT_MODES[mode]["levers"])


def specs_of(mode: str) -> dict:
    """{lever id: SPEC} — the settings the mode PINS for its per-step levers (`KIT_MODES[mode]["specs"]`, stated explicitly per tier, never
    inferred): `levers.install(mode)` configures those levers with that SPEC and every other lever of the row with None (= the module's own
    default setting). Empty for a row whose levers all run at their default setting."""
    return dict(KIT_MODES[mode].get("specs", {}))


P1_FORMS = ("pinned", "transparent")   # how a row carries P1 (p1_form)


def p1_form(mode: str) -> Optional[str]:
    """How the mode's row carries P1 — read from the table, never from the word: ``pinned`` = the row has a populate row (exact: the
    compilation cache AND the autotune results dumped once per shape by `warm`, loaded by every later process — the cross-process bitwise
    mechanism; a shape that is not warm is warmed first, a process without `MOSAIC_OPT_CACHE_ROOT` is refused by name); ``transparent`` = P1
    in the row and no populate row (fast, big: the compilation cache alone, filled by whichever process compiles first and read by every
    later one — no autotune pin, no warm step, no bitwise promise across processes beyond the tier's own; when `MOSAIC_OPT_CACHE_ROOT` is
    unset the lever steps aside BY NAME and the row runs without it); ``None`` = no P1 in the row (off)."""
    km = KIT_MODES[mode]
    if not any(LEVERS[l].route == "env" and l == "P1" for l in km["levers"]):
        return None
    return "pinned" if km.get("populate") else "transparent"


# ----------------------------------------------------------------------------------------------------------------- the row grammar
def parse_row(text: str) -> dict:
    """`run_arm TAG "ENV" ARGS...` -> {"tag", "env": {NAME: value-with-placeholders}, "args": [...]} (shlex; the ENV field is one
    quoted word of `NAME=value` assignments, `run_arm` exports it with `eval "export $envs"`)."""
    words = shlex.split(text)
    if len(words) < 3 or words[0] != "run_arm":
        raise ValueError(f"not a run_arm line: {text!r}")
    tag, envs, args = words[1], words[2], words[3:]
    env = {}
    for a in envs.split():
        k, _, v = a.partition("=")
        env[k] = v
    return {"tag": tag, "env": env, "args": args}


def substitute(value: str, cache_dir: Optional[str] = None, features: Optional[str] = None, features_sha: Optional[str] = None) -> str:
    """The rows' placeholders: `$C1` (the P1 directory), `$F1` (the frozen features npz), `$FSHA` (its sha256)."""
    out = value
    for ph, val in (("$C1", cache_dir), ("$F1", features), ("$FSHA", features_sha)):
        if ph in out:
            if val is None:
                raise ValueError(f"row needs {ph} but none was given")
            out = out.replace(ph, str(val))
    return out


@dataclass(frozen=True)
class Resolution:
    mode: str
    row: str                                         # the tag of the row
    levers: Tuple[str, ...]                          # the levers the row switches on
    env: Dict[str, str]                              # the row's environment (placeholders substituted)
    flags: List[str]                                 # the row's driver flags after `--seed <s>` (placeholders substituted)
    populate_row: Optional[str] = None               # the row that must have run once for this (image, GPU type, shape)
    notes: Tuple[str, ...] = field(default_factory=tuple)
    levers_off: Tuple[str, ...] = field(default_factory=tuple)   # levers of the row switched off BY NAME for this process (`MODEL_OPT_LEVERS_OFF`: levers.levers_off) — their environment / flags dropped from env / flags, the mode keeps its word
    aside: Dict[str, str] = field(default_factory=dict)          # {lever: reason} — levers of the row that step aside by name in this process (P1 transparent without MOSAIC_OPT_CACHE_ROOT)


def drop_flag(args: List[str], flag_text: str) -> List[str]:
    """`args` without the token run `flag_text` spells (a flag-route lever's registry ``flag``, placeholders unsubstituted: P2's
    `--weights fastinit`, P3's `--features-in "$F1" --features-sha "$FSHA"`); `args` unchanged when the run is absent."""
    toks = shlex.split(flag_text)
    n = len(toks)
    for i in range(len(args) - n + 1):
        if args[i:i + n] == toks:
            return args[:i] + args[i + n:]
    return list(args)


DRIVER_ONLY = "driver_only"                                       # the aside WORD of a call-site lever (registry route "flag": P2 `--weights fastinit`, P3 `--features-in …`) on the in-process
                                                                  # route — a replacement only the kit driver makes at its own call site, so `MOSAIC_OPT=<mode>` / `enable()` run the row's other
                                                                  # levers and name these on the ACTIVE line (`row=<tag>[…]-aside[P2:driver_only]`); never a partial activation, never an exit
ASIDE_WORDS = {DRIVER_ONLY: "a call-site replacement only the kit driver applies (`mosaic-opt design`); this route runs the row's other levers"}   # aside reasons that are words
                                                                  # ride the row token as `<lever>:<word>`; a free-text reason (P1 without a cache root, a cold shape) rides `p1=none(…)` and the notes


def resolve(mode: str, cache_dir: Optional[str] = None, features: Optional[str] = None, features_sha: Optional[str] = None,
            phase: str = "warm", levers_off: Tuple[str, ...] = (), p1_aside: Optional[str] = None, aside: Optional[Dict[str, str]] = None) -> Resolution:
    """A package mode -> its row, expanded for this process: the environment `run_arm` would export and the driver flags
    that follow `--seed`. `phase="populate"` selects the populate row of the mode (the first process of a shape). Mode `off` resolves to
    the stock row `A_stock1` (no environment, in-process featurization; `--features-out` is the caller's choice and is dropped here), or,
    when frozen features are given, to the stock row on them, `D_stock2` (`--features-in NPZ --features-sha SHA`).
    `levers_off` (registry ids, validated by the caller: `levers.levers_off()` reads `MODEL_OPT_LEVERS_OFF`) switches levers of the row off
    BY NAME for this process: an env-route lever's variables (registry ``env``) leave the environment, a flag-route lever's flag text leaves
    the flags, a per-step lever is the installer's to skip (the driver reads the same variable) — the mode keeps its word, ``levers`` names
    what stays on, ``levers_off`` what was switched off. `p1_aside` = the reason P1 steps aside in this process (the transparent form with
    no cache root): P1's variables dropped, ``aside["P1"]`` = the reason; refused for a pinned row (its P1 is the mode's promise). `aside` =
    {lever: reason} for any other lever of the row that steps aside by name in this process (the in-process route's call-site levers:
    ``{"P2": DRIVER_ONLY}``): its flag text leaves the flags, ``levers`` no longer names it, ``aside`` does."""
    mode = (mode or "").strip().lower()
    if mode not in MODES:
        raise ValueError(unknown_mode_message(mode))                                       # a tier word without a row is named as such, never resolved to a stock row
    km = KIT_MODES[mode]
    unknown = [l for l in levers_off if l not in LEVERS]
    if unknown:
        raise ValueError(f"levers_off names no lever of this kit: {','.join(unknown)} (the registry's ids: {','.join(LEVERS)})")
    tag = km["populate"] if (phase == "populate" and km["populate"]) else km["row"]
    if features and km.get("features_row"):
        tag = km["features_row"]
    row = ROWS[tag]
    parsed = parse_row(row["text"])
    off = tuple(l for l in km["levers"] if l in levers_off)                                # in row order
    aside = {l: str(why) for l, why in {**({"P1": p1_aside} if p1_aside else {}), **(aside or {})}.items() if l in km["levers"] and l not in off}   # levers of the row that step aside BY NAME in this process: P1 transparent without a root; P1 + P3 of the pinned row on a shape not yet warm (the design compiles and featurizes as stock does)
    gone = set(off) | set(aside)
    drop_env = {k for l in gone for k in LEVERS[l].env}
    env = {k: substitute(v, cache_dir, features, features_sha) for k, v in parsed["env"].items() if k not in drop_env}
    args = parsed["args"]
    for l in gone:
        if LEVERS[l].route == "flag" and LEVERS[l].flag:
            args = drop_flag(args, LEVERS[l].flag)
    # the row's own `--seed 0` is an example item, not a lever: the caller's seed replaces it; `--features-out` (stock arm A) likewise
    flags: List[str] = []
    skip = 0
    for i, a in enumerate(args):
        if skip:
            skip -= 1
            continue
        if a in ("--seed", "--features-out"):
            skip = 1
            continue
        flags.append(substitute(a, cache_dir, features, features_sha))
    notes: Tuple[str, ...] = ()
    if km["populate"] and phase == "warm" and "P1" not in gone:
        notes = (f"loads the autotune results dumped by the populate row ({km['populate']}) for this shape",)
    if aside:
        notes = notes + tuple(f"{l} steps aside: {ASIDE_WORDS.get(why, why)}" for l, why in aside.items())
    if off:
        notes = notes + (f"levers switched off by name (MODEL_OPT_LEVERS_OFF): {','.join(off)}",)
    return Resolution(mode=mode, row=tag, levers=tuple(l for l in km["levers"] if l not in gone), env=env, flags=flags,
                      populate_row=km["populate"] if mode != "off" else None, notes=notes, levers_off=off, aside=aside)


def describe_line(res: Resolution) -> str:
    """`<row>[<levers>|stock][-off[<ids>]][-aside[<id>[:<word>],…]]` — the row token of the ACTIVE / DRY-RUN line: a lever that steps aside for a
    reason that is a WORD (ASIDE_WORDS: `driver_only`) carries it (`P2:driver_only`); a free-text reason stays off the token (`-aside[P1,P3]`)."""
    aside = ",".join(l + (f":{why}" if why in ASIDE_WORDS else "") for l, why in res.aside.items())
    return (res.row + (f"[{','.join(res.levers)}]" if res.levers else "[stock]") + (f"-off[{','.join(res.levers_off)}]" if res.levers_off else "")
            + (f"-aside[{aside}]" if res.aside else ""))


for _tag, _row in ROWS.items():                                   # every environment variable a row assigns is owned by an env-route lever of a mode that names the row (registry ``env``): the resolver can drop a lever's variables by name, and nothing unowned rides a row
    _owners = {k for w, km in KIT_MODES.items() if _tag in (km["row"], km.get("populate"), km.get("features_row")) for l in km["levers"] for k in LEVERS[l].env}
    assert set(parse_row(_row["text"])["env"]) <= _owners, (_tag, sorted(set(parse_row(_row["text"])["env"]) - _owners))
for _w in MODES:                                                  # a P1 row's environment names the P1 directory ($C1); a pinned one names the autotune file in it too, a transparent one does not (no autotune flag: pcc autotune=off)
    _f = p1_form(_w); _vals = list(parse_row(ROWS[KIT_MODES[_w]["row"]]["text"])["env"].values())
    assert (_f is None) == (not _vals) and (_f is None or any("$C1" in v for v in _vals)) and (_f == "pinned") == any(AUTOTUNE_FILENAME in v for v in _vals), (_w, _f, _vals)


# ----------------------------------------------------------------------------------------------------------------- the stack key
def gpu_slug(name: Optional[str]) -> str:
    from opt_core.jax_design import pcc
    return pcc.gpu_slug(name or "unknown-gpu")


def jit_cache_key(jax_version: Optional[str] = None, jaxlib_version: Optional[str] = None, plugin_version: Optional[str] = None,
                  gpu_name: Optional[str] = None) -> str:
    """`jax<v>-jaxlib<v>-cuda12plugin<v>-<gpu product name slug>`: the (stack, GPU type) the P1 files belong to — the shared JAX
    persistent-cache key (`opt_core.jax_design.pcc.key`), e.g. `jax0.10.2-jaxlib0.10.2-cuda12plugin0.10.2-nvidia-h100-80gb-hbm3`. Defaults read
    the installed distributions (`jax`, `jaxlib`, `jax-cuda12-plugin`) and nvidia-smi's product name; `configs/<gpu>.env` sets MODEL_OPT_STACK_KEY
    from it. Raises `pcc.PccError` (a RuntimeError) naming the missing part — no jax, no jaxlib, no CUDA plugin, no GPU visible (a
    wrong-but-plausible key is a silent P1-cache collision, not a valid one)."""
    from opt_core.jax_design import pcc
    return pcc.key(jax_version, jaxlib_version, plugin_version, gpu_name)
