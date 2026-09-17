"""Modes and variants, and how a package mode resolves to the kit's own command line.

The kit is a driver: an optimized mode runs a different executable on the stock arguments — ``kit/mpnn_worker2.py`` — and the
switches are that executable's flags. The documented line
is a row of the kit's README (``opt/forward/mpnn_exact_worker/README.md``); this module READS that row out of the carried README
(``kit_line``), strips the arguments that are stock options or inputs (``SETTINGS_FLAGS``, supplied per run by settings.py / inputs.py) and keeps the
lever flags. Nothing is transcribed: the ``--x_all`` expansion comes from the worker's own ``--x_all`` help text and its assignment line
(``x_all_expansion`` / ``x_all_assignment``), the CPU rule from the worker's own line (``cpu_disabled_levers``), the per-device
probe statement from the README's hardware note (``probe_kit_observed``).

Package modes (``KIT_MODES``: the one table; ``MODES`` / ``VARIANTS`` / ``DEFAULT_VARIANT`` / ``DEFAULT_MODE`` are what every command
reads). ``DEFAULT_MODE`` is ``None``, the same for every variant: the family default is ``fast`` and this engine ships no fast tier, so its
commands have no default — a command that names no mode (no ``--mode``, no ``PROTEINMPNN_OPT``) is a usage error whose one line names the modes
served (``no_mode_message``, exit 2); a pass names ``--mode exact`` or ``--mode off``. The environment switch unset means nothing activates
(the stock command line is stock).
  off     stock — the upstream command line in a clean subprocess (stock_run.py); nothing from the kit on the path.
  exact   the kit's bit-identical line: ``kit/mpnn_worker2.py <stock args> --mode stream --bb_batch 16 --sort_by_length
          --x_all`` (README row ``EXACT_ROW``) plus the worker's probe-gated ``--hybrid_gemm`` (the table's ``probe_gated``: the
          decoder message GEMMs of all backbones of a batch in one call; the worker probes every decode-step cell on the device before any
          output and applies it only when bit-identical — on a probe FAIL the worker refuses the job by name, the line being all of its levers), executed as ``kit/mpnn_worker2_lowmem.py`` — the staged worker with the low-memory
          featuriser and decoding-order mask of ``lowmem.py`` (the table's ``transform``, applied per run by ``stage.stage_lowmem``: a
          working set linear in the number of residues, the same bytes out).
``fast`` is not a mode of this engine: proteinmpnn ships no fast tier (no tolerance-tier line exists for either executable).
``check_mode`` refuses the name, as it refuses every standard mode name of ``opt_core.modes.STANDARD_MODES`` the table lacks, with ``UnsupportedMode`` — ``proteinmpnn ships no fast tier: select --mode exact``, exit 3 on
the command line — before anything resolves; an unknown name is a usage error (exit 2). Shipping a tier is a change to ``MODES`` / ``KIT_MODES`` only.
"""
from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from opt_core import modes as core_modes

MODES: Tuple[str, ...] = ("exact", "off")
VARIANTS: Tuple[str, ...] = ("soluble", "vanilla")                  # the two ProteinMPNN weight sets: one program, one mode table
DEFAULT_VARIANT = "vanilla"                                     # upstream's own default weight set: protein_mpnn_run.py loads vanilla_model_weights unless
                                                                # --use_soluble_model / --path_to_model_weights say otherwise (protein_mpnn_run.py:44-49)
DEFAULT_MODE: Optional[str] = None   # a value, never a rule computed from the mode table: the family default is fast, a tier this engine does not ship, so no mode is the default here — a command names one (check_mode: no_mode_message, a usage error); shipping a fast tier is a change to MODES / KIT_MODES and this value only
KIT_MODE = "exact"          # the kit mode that ships: the pointer of the fast refusal (unsupported_message)

WORKER_DIR = "mpnn_exact_worker"                                  # under kit_home (opt/forward)
PARSER_DIR = "mpnn_pdb_parser"
README = "README.md"                                          # the worker directory's README: the mode rows
WORKER = os.path.join("addon", "mpnn_worker2.py")             # the exact executable (staged as kit/mpnn_worker2.py)

# README rows (1-based line ranges in opt/forward/mpnn_exact_worker/README.md; continuation lines end with a backslash)
EXACT_ROW: Tuple[int, int] = (30, 32)                         # the worker command line of mode exact
PROBE_ROW: Tuple[int, int] = (56, 57)                         # the hardware note's "Observed: ..." sentence

LOWMEM = "lowmem"                                             # the package transform of the exact executable (lowmem.py, stage.stage_lowmem) and the lever name it reports

# The one table: a row per mode of MODES (every variant: the variants are weight sets). route: 'stock' | 'worker'; 'transform': the package transform of
# the row's executable (LOWMEM: kit/mpnn_worker2_lowmem.py), absent = the row's executable as carried; 'probe_gated': worker flags appended
# to the row's lever flags whose application the worker decides by its own on-device probe (PROBE_GATED_FLAGS; each has one opt-out on the
# design command line, OPT_OUTS: `--hybrid_gemm 0` leaves it out of the set by name).
OPT_OUTS: Dict[str, str] = {"hybrid_gemm": "--hybrid_gemm"}           # lever name -> its opt-out on the design command line (`<flag> 0` leaves the probe-gated lever out of the set by name; 1, the default, requests it)
PROBE_GATED_FLAGS: Tuple[str, ...] = ("--hybrid_gemm",)               # the worker's probe-gated batched message GEMMs (its --hybrid_gemm; README hardware note PROBE_ROW)
KIT_MODES: Dict[str, dict] = {
    "off": {"route": "stock"},
    "exact": {"route": "worker", "row": EXACT_ROW, "transform": LOWMEM, "probe_gated": PROBE_GATED_FLAGS},
}

# Arguments of the README rows that are stock options or inputs (supplied per run), not levers.
SETTINGS_FLAGS = {"--jsonl_path", "--chain_id_jsonl", "--out_folder", "--path_to_model_weights", "--model_name", "--num_seq_per_target",
                  "--batch_size", "--sampling_temp", "--seed"}


class ModeError(ValueError):
    """A mode / variant that does not resolve: an unknown name (a usage error)."""


class UnsupportedMode(ModeError):
    """A standard mode name (``opt_core.modes.STANDARD_MODES``) this engine does not ship — ``fast``: refused by name with the pointer to the mode that
    ships (``unsupported_message``; exit 3 on the command line, where an unknown name is exit 2)."""


def unsupported_message(mode: str) -> str:
    """``proteinmpnn ships no <mode> tier: select --mode <KIT_MODE>`` — the one text of the refusal (the command line, ``enable()``, the docs' sentence)."""
    return f"proteinmpnn ships no {mode} tier: select --mode {KIT_MODE}"


def no_mode_message() -> str:
    """``no mode named: --mode off|exact (or PROTEINMPNN_OPT) — proteinmpnn ships no fast tier, so it has no default mode`` — the usage error of a command without a mode."""
    return f"no mode named: --mode {'|'.join(sorted(MODES, key=lambda m: m != 'off'))} (or PROTEINMPNN_OPT) — proteinmpnn ships no fast tier, so it has no default mode"


@dataclass
class Resolution:
    mode: str
    variant: str
    route: str                                   # stock | worker
    executable: Optional[str] = None             # the kit executable named by the README row (relative, as the row writes it)
    flags: List[str] = field(default_factory=list)          # the lever flags of the row, settings stripped (tokens)
    levers: List[str] = field(default_factory=list)         # the levers those flags name, --x_all expanded (names without dashes)
    bb_batch: Optional[int] = None
    hybrid_gemm: bool = False                    # the probe-gated lever is requested (the worker applies it iff its on-device probe passes)
    probe_gated: List[str] = field(default_factory=list)     # the requested lever flags whose application the worker's probe decides (names without dashes in `levers`)
    transform: Optional[str] = None              # LOWMEM (exact): the executable is the transformed staged worker (stage.stage_lowmem)
    opted_out: List[str] = field(default_factory=list)       # probe-gated levers the command line left out by name (OPT_OUTS: `--hybrid_gemm 0`)
    row: Optional[Tuple[int, int]] = None
    line: str = ""                               # the README row, joined


# ----------------------------------------------------------------------------------------------------------------- reading the kit
MODE_TABLE = core_modes.ModeTable(modes=MODES, default=KIT_MODE, unknown_message=lambda m: f"unknown mode {str(m)!r}: choose one of {'|'.join(MODES)}")   # the core's table over the names that ship; a name never reaches it empty (check_mode refuses an empty name first), so its default is never consulted


def check_mode(mode: Optional[str]) -> str:
    """The mode name: the core's ModeTable over this kit's names (opt_core.modes). No name is a usage error (ModeError, ``no_mode_message``: this
    engine has no default mode); a standard mode name the table lacks (``fast``) raises UnsupportedMode with the pointer; any other unknown name raises
    ModeError with the kit's usage text."""
    if mode is None or not str(mode).strip():
        raise ModeError(no_mode_message())
    try:
        return MODE_TABLE.check(mode)
    except core_modes.ModeError as e:
        name = str(mode).strip().lower()
        if name in core_modes.STANDARD_MODES:
            raise UnsupportedMode(unsupported_message(name)) from None
        raise ModeError(str(e)) from None


def check_variant(variant: Optional[str]) -> str:
    variant = DEFAULT_VARIANT if variant in (None, "") else str(variant)
    if variant not in VARIANTS:
        raise ModeError(f"unknown variant {variant!r}: choose one of {'|'.join(VARIANTS)}")
    return variant


def worker_home(kit_home: str) -> str:
    return os.path.join(kit_home, WORKER_DIR)


def readme_lines(kit_home: str) -> List[str]:
    with open(os.path.join(worker_home(kit_home), README), encoding="utf-8") as fh:
        return fh.read().split("\n")


def readme_row(kit_home: str, row: Tuple[int, int]) -> str:
    """Lines ``row`` (1-based, inclusive) of the README joined into one command line (backslash continuations removed)."""
    lines = readme_lines(kit_home)[row[0] - 1:row[1]]
    parts = []
    for ln in lines:
        ln = ln.strip()
        if ln.endswith("\\"):
            ln = ln[:-1].rstrip()
        parts.append(ln)
    return " ".join(parts)


def parse_command(line: str) -> Tuple[str, List[Tuple[str, Optional[str]]]]:
    """``python <exe> --a 1 --b`` -> (exe, [(--a, 1), (--b, None)]). A flag's value is the following token when it is not a flag."""
    toks = shlex.split(line)
    if toks and toks[0] == "python":
        toks = toks[1:]
    if not toks:
        raise ModeError(f"README row is not a command: {line!r}")
    exe, args = toks[0], []
    i = 1
    while i < len(toks):
        t = toks[i]
        if not t.startswith("--"):
            raise ModeError(f"unexpected token {t!r} in README row {line!r}")
        val = None
        if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
            val = toks[i + 1]
            i += 1
        args.append((t, val))
        i += 1
    return exe, args


def lever_flags(args: List[Tuple[str, Optional[str]]]) -> List[str]:
    out = []
    for flag, val in args:
        if flag in SETTINGS_FLAGS:
            continue
        out.append(flag)
        if val is not None:
            out.append(val)
    return out


def _worker_source(kit_home: str) -> str:
    with open(os.path.join(worker_home(kit_home), WORKER), encoding="utf-8") as fh:
        return fh.read()


def x_all_expansion(kit_home: str) -> List[str]:
    """The flags ``--x_all`` stands for, from the worker's own ``--x_all`` help text ("recommended exact set = --a --b ...")."""
    src = _worker_source(kit_home)
    m = re.search(r'add_argument\("--x_all".*?help="[^"]*?recommended exact set = ((?:--\w+\s*)+)', src, re.S)
    if not m:
        raise ModeError("the worker's --x_all help text does not name its expansion (addon/mpnn_worker2.py)")
    return m.group(1).split()


def x_all_assignment(kit_home: str) -> List[str]:
    """The flags the worker sets when ``--x_all`` is given (its ``if args.x_all: args.a = args.b = ... = True`` line)."""
    src = _worker_source(kit_home)
    m = re.search(r"if args\.x_all:\s*((?:args\.\w+\s*=\s*)+)True", src)
    if not m:
        raise ModeError("the worker's --x_all assignment line was not found (addon/mpnn_worker2.py)")
    return ["--" + n for n in re.findall(r"args\.(\w+)\s*=", m.group(1))]


def worker_flag(kit_home: str, flag: str) -> str:
    """``flag`` if the worker's own argparse declares it (``p.add_argument("<flag>", ...)`` in addon/mpnn_worker2.py), else ModeError."""
    if not re.search(r"add_argument\(\s*[\"']" + re.escape(flag) + r"[\"']", _worker_source(kit_home)):
        raise ModeError(f"the worker declares no {flag} switch (addon/mpnn_worker2.py)")
    return flag


def cpu_disabled_levers(kit_home: str) -> List[str]:
    """The levers the worker switches off without a CUDA device (its ``if not torch.cuda.is_available(): args.a = ... = False`` line)."""
    src = _worker_source(kit_home)
    m = re.search(r"if not torch\.cuda\.is_available\(\):\s*((?:args\.\w+\s*=\s*)+)False", src)
    if not m:
        raise ModeError("the worker's CPU rule was not found (addon/mpnn_worker2.py)")
    return ["--" + n for n in re.findall(r"args\.(\w+)\s*=", m.group(1))]


def probe_kit_observed(kit_home: str) -> Dict[str, List[str]]:
    """The kit's own per-device statement on the optional ``--hybrid_gemm`` probe (README hardware note, "Observed: ... PASS; ...: FAIL").
    Returns ``{"PASS": [...], "FAIL": [...]}`` with the device names as the README writes them."""
    text = " ".join(l.strip() for l in readme_lines(kit_home)[PROBE_ROW[0] - 1:PROBE_ROW[1]])
    m = re.search(r"Observed:\s*(.*?)(?:\.\s+Stock outputs|$)", text)
    out: Dict[str, List[str]] = {"PASS": [], "FAIL": []}
    if not m:
        return out
    for part in m.group(1).split(";"):
        mm = re.match(r"\s*(.+?):\s*(PASS|FAIL)", part)
        if mm:
            out[mm.group(2)] += [n.strip() for n in mm.group(1).split(",")]
    return out


def probe_verdict_kit_observed(kit_home: str, sm: Optional[str]) -> Optional[str]:
    """PASS/FAIL the README records for a compute capability written ``sm_90`` (None when the kit names no observation for it)."""
    if not sm:
        return None
    for verdict, names in probe_kit_observed(kit_home).items():
        if any(sm in n for n in names):
            return verdict
    return None


def expanded_flags(flags: List[str], kit_home: str) -> List[str]:
    """The row's flag tokens with ``--x_all`` replaced by the flags it stands for (the worker's own help text); values kept in place."""
    out: List[str] = []
    for f in flags:
        out += x_all_expansion(kit_home) if f == "--x_all" else [f]
    return out


def levers_of(flags: List[str], kit_home: str) -> List[str]:
    """Lever names for the row's flags: ``--x_all`` expanded through the worker's help text, values kept for parameters."""
    names: List[str] = []
    i = 0
    while i < len(flags):
        f = flags[i]
        val = flags[i + 1] if i + 1 < len(flags) and not flags[i + 1].startswith("--") else None
        if f == "--x_all":
            names += [x.lstrip("-") for x in x_all_expansion(kit_home)]
        elif f == "--bb_batch":
            pass                                              # reported as its own field
        elif f == "--mode":
            names.append(str(val))                            # stream: the replay of the stock RNG stream
        elif val is not None:
            names.append(f"{f.lstrip('-')}{val}")             # a valued flag: <name><value>
        else:
            names.append(f.lstrip("-"))
        i += 2 if val is not None else 1
    return names


# ----------------------------------------------------------------------------------------------------------------- resolution
def resolve(mode: Optional[str], variant: Optional[str], kit_home: str, opt_out: Optional[List[str]] = None,
            bb_batch: Optional[int] = None) -> Resolution:
    """A package mode for a variant -> the kit's documented line (read from the README); a name outside the table is check_mode's refusal
    (ModeError). ``opt_out``: probe-gated lever names the command line leaves out of the set (OPT_OUTS keys; another name is a ModeError).
    ``bb_batch``: the caller's backbones-per-batch for the worker line (its ``--bb_batch``; None = the README row's own value) — how many
    backbones share one padded forward pass: speed and GPU memory, never the outputs (every backbone keeps its stock-shaped encoder call, its
    per-backbone GEMMs and its own RNG stream offset). The stock route has no batch of backbones: naming one there is a ModeError."""
    mode, variant = check_mode(mode), check_variant(variant)
    ent = KIT_MODES[mode]
    out = [str(n) for n in (opt_out or [])]
    bad = [n for n in out if n not in OPT_OUTS]
    if bad:
        raise ModeError(f"no opt-out exists for {', '.join(bad)}: the levers with an opt-out are {', '.join(f'{n} ({f} 0)' for n, f in OPT_OUTS.items())}")
    if bb_batch is not None and (isinstance(bb_batch, bool) or not isinstance(bb_batch, int) or bb_batch < 1):
        raise ModeError(f"--bb_batch {bb_batch!r}: a whole number of backbones per batch, 1 or more")
    if ent["route"] == "stock":
        if bb_batch is not None:
            raise ModeError(f"--bb_batch {bb_batch}: mode {mode} runs the stock command line, one backbone per forward pass; the backbone batch "
                            f"is the kit line's (--mode {'|'.join(m for m in MODES if KIT_MODES[m]['route'] != 'stock')})")
        return Resolution(mode, variant, "stock")
    line = readme_row(kit_home, ent["row"])
    exe, args = parse_command(line)
    flags = lever_flags(args)
    if bb_batch is not None:                                  # the caller's batch replaces the row's value (the flag is the worker's own; ModeError when the row has none)
        if "--bb_batch" not in flags or flags.index("--bb_batch") + 1 >= len(flags):
            raise ModeError(f"--bb_batch {bb_batch}: the {mode} line ({line}) carries no --bb_batch value to set")
        flags = list(flags); flags[flags.index("--bb_batch") + 1] = str(int(bb_batch))
    gated_all = [f for f in ent.get("probe_gated", ()) if f not in flags]
    gated = [f for f in gated_all if f.lstrip("-") not in out]
    for f in gated:
        worker_flag(kit_home, f)                              # the worker declares the flag, or ModeError (never a guessed switch)
    flags = flags + gated
    res = Resolution(mode, variant, ent["route"], executable=exe, flags=flags, levers=levers_of(flags, kit_home), row=ent["row"], line=line,
                     transform=ent.get("transform"), probe_gated=list(gated), opted_out=[f.lstrip("-") for f in gated_all if f.lstrip("-") in out])
    if res.transform:
        res.levers.append(res.transform)                      # reported and evidenced like a flag lever (the worker record's "lowmem" field)
    for i, f in enumerate(flags):
        if f == "--bb_batch":
            res.bb_batch = int(flags[i + 1])
        elif f == "--hybrid_gemm":
            res.hybrid_gemm = True
    return res

def describe_line(res: Resolution) -> str:
    if res.route == "stock":
        return "stock: protein_mpnn_run.py"
    line = f"{res.executable} <stock args> {' '.join(res.flags)}"
    if res.transform == LOWMEM:
        line += " (run as kit/mpnn_worker2_lowmem.py: + the low-memory featuriser and decoding-order mask, lowmem.py)"
    return line

