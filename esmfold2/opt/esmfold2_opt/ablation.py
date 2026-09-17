"""The package's one declared ablation variable, ``ESMFOLD2_OPT_ABLATE`` — a way to turn individual optimizations of a mode OFF (or pick a
declared precision sub-choice) for a diagnostic run, never part of a mode's plain line:

    ESMFOLD2_OPT_ABLATE=<token>[,<token>...]      token = <lever>                  a registry lever name in the resolved mode's set: subtracted from the
                                                                                    set BEFORE anything is installed (the mode installs the rest; the
                                                                                    all-or-refuse verdict treats it as not in the set; its LEVER line
                                                                                    reads state=off reason=ablated)
                                                           <lever>.<knob>=<value>   a declared sub-choice of a lever that stays in the set
                                                                                    (registry.ABLATION_KNOBS: af.gemm, dit.gemm, dit.cond, dit.attn,
                                                                                    dit.attn_precision; values = the module's own vocabulary)

The ACTIVE and APPLIED lines carry ``ablate=<tokens>`` whenever the variable is set (and only then). Every token is checked by name against the
registry and the mode's set before activation: an unknown lever, a lever not in this mode's set, an unknown knob or value, a knob of an ablated
lever, or a subtraction that strands a dependent lever (DEPENDS: e.g. ``ro`` off while ``dit`` stays) is an ``AblationError`` — the mode refuses
by name (NOT ACTIVE), it never runs a guessed set. ``parse`` reads no environment: stack passes the variables' text (``env_text``) to modes.resolve.

``MODEL_OPT_LEVERS_OFF`` (``ENV_LEVERS_OFF``) is the release tree's uniform spelling of the same request and an ALIAS here: its words that name a
lever of THIS kit (a registry name, or ``<lever>.<knob>=<value>`` on one) are ESMFOLD2_OPT_ABLATE tokens with the same checks. The shared core's
kernel packages read that same variable for their own words (``trimul_tx``, ``pallas:<row>``, ...), so a word that names no lever of this kit is
THEIRS: left to them and named once on the ACTIVE line's notes (``foreign_words``) — never an error here and never guessed onto a kit lever.
``ESMFOLD2_OPT_ABLATE`` itself stays strict (every token must be this kit's).
"""
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .registry import ABLATION_KNOBS, LEVERS

ENV_ABLATE = "ESMFOLD2_OPT_ABLATE"
ENV_LEVERS_OFF = "MODEL_OPT_LEVERS_OFF"          # the release tree's uniform ablation word — an alias of ENV_ABLATE for this kit's levers (env_text); the core's packages read it too
ENV_WORDS = (ENV_ABLATE, ENV_LEVERS_OFF)         # every environment name this module reads (stack / the tests hold the list)
DEPENDS: Dict[str, Tuple[str, ...]] = {          # lever -> the levers it rides on (configure() installs the dependent only with its base): ablating a base requires ablating its dependents
    "ls": ("tg",), "rg": ("ls", "tg"),           # the static loop I/O lives inside the trunk-graph wrappers; the recycle graph inside the static loop
    "kd": ("ro",), "dit": ("ro",),               # the device Kabsch and the fused step ride on the roll-out
    "glue": ("trimul",),                         # the TriMul glue is folded into trimul's transposes
    "af": ("ax",),                               # the fused atom block reads the exact hoists' static tables
    "confbf16": ("confrows", "confmem"), "pdeskip": ("confrows",), "confmem": ("confrows",), "zbf16": ("confrows",),   # the row-chunking flags ride on the row-blocked confidence
    #   statement (n_gpu > 1); confbf16 also on confmem: its bf16 pair reaches the row-attention pooling, which only confmem's per-block statement upcasts
}


class AblationError(ValueError):
    """An ablation token the package cannot honour, by name."""


def _split(text) -> List[str]:
    return [t for t in re.split(r"[,\s]+", str(text or "").strip()) if t]


def kit_word(token: str, levers=None) -> bool:
    """True when an ablation token addresses THIS kit: ``<lever>`` or ``<lever>.<knob>=<value>`` with ``<lever>`` a registry name."""
    levers = LEVERS if levers is None else levers
    return token.split("=", 1)[0].split(".", 1)[0] in levers


def env_text(environ=None, levers=None) -> Optional[str]:
    """The ablation text for modes.resolve from the environment: ``ESMFOLD2_OPT_ABLATE`` verbatim (strict: parse checks every token) plus the
    words of ``MODEL_OPT_LEVERS_OFF`` that address this kit (kit_word), a word given in both variables once. None when ESMFOLD2_OPT_ABLATE is
    unset and MODEL_OPT_LEVERS_OFF names no lever of this kit (its other words are the core packages': foreign_words)."""
    environ = os.environ if environ is None else environ
    own = environ.get(ENV_ABLATE)
    alias = [t for t in _split(environ.get(ENV_LEVERS_OFF)) if kit_word(t, levers)]
    if own is None and not alias:
        return None
    return ",".join(dict.fromkeys(_split(own) + alias))


def foreign_words(environ=None, levers=None) -> List[str]:
    """``MODEL_OPT_LEVERS_OFF`` words that name no lever of this kit (they address the shared core's kernel packages, which read the same
    variable): left to them; stack names them once on the ACTIVE line's notes."""
    environ = os.environ if environ is None else environ
    return [t for t in _split(environ.get(ENV_LEVERS_OFF)) if not kit_word(t, levers)]


COMPILE_WORD = "compile"                         # the release tree's sanctioned compile opt-out word (MODEL_OPT_LEVERS_OFF=compile = --no-compile elsewhere): stock ESMFold2 never calls
                                                 # torch.compile and this kit adds no compile step, so the word names nothing here — accepted BY NAME as n/a (the ACTIVE line's
                                                 # compile=n/a, report.COMPILE_WORD), never an error, nothing turned off


def foreign_note(environ=None, levers=None) -> Optional[str]:
    """The one note naming foreign_words(), or None: the compile word as n/a (COMPILE_WORD), every other word as the core packages'."""
    words = foreign_words(environ, levers)
    if not words:
        return None
    parts = []
    if COMPILE_WORD in words:
        parts.append(f"{ENV_LEVERS_OFF}={COMPILE_WORD}: n/a — neither stock ESMFold2 nor this kit compiles anything (compile=n/a), nothing to turn off")
    rest = [w for w in words if w != COMPILE_WORD]
    if rest:
        parts.append(f"{ENV_LEVERS_OFF} words that name no lever of this kit are left to the core's kernel packages: {','.join(rest)}")
    return "; ".join(parts)


@dataclass(frozen=True)
class Ablation:
    tokens: Tuple[str, ...] = ()                 # the variable's tokens, in the order given (the ACTIVE line's ablate= word)
    levers: Tuple[str, ...] = ()                 # lever names subtracted from the set
    knobs: Dict[str, str] = field(default_factory=dict)   # {"<lever>.<knob>": value}

    @property
    def active(self) -> bool:
        return bool(self.tokens)

    def word(self) -> str:
        """``ablate=<tokens>`` (comma-joined, blank-free) or '' when nothing is ablated."""
        return ("ablate=" + ",".join(self.tokens)) if self.tokens else ""


def tokens_of(text) -> Tuple[str, ...]:
    """Split the variable's text on ',' / whitespace; empty / unset / 'none' / 'off' -> ()."""
    if text is None:
        return ()
    s = str(text).strip()
    if not s or s.lower() in ("none", "off", "0"):
        return ()
    return tuple(t for t in s.replace(" ", ",").split(",") if t)


def parse(text, mode_set: Iterable[str], mode: str = "?") -> Ablation:
    """The variable's text against the mode's lever set (registry names, install order irrelevant). Returns the Ablation; raises AblationError
    naming the first token that is unknown, not in the set, or inconsistent."""
    toks = tokens_of(text)
    if not toks:
        return Ablation()
    in_set = list(mode_set)
    levers, knobs = [], {}
    for tok in toks:
        if "=" in tok:
            key, _, value = tok.partition("=")
            if key not in ABLATION_KNOBS:
                raise AblationError(f"{ENV_ABLATE}: unknown knob {key!r} in token {tok!r} (declared knobs: {', '.join(sorted(ABLATION_KNOBS))})")
            if value not in ABLATION_KNOBS[key]:
                raise AblationError(f"{ENV_ABLATE}: {key}={value!r} is not a declared value (one of {', '.join(ABLATION_KNOBS[key])})")
            lever = key.split(".", 1)[0]
            if lever not in in_set:
                raise AblationError(f"{ENV_ABLATE}: knob {key} names lever {lever!r}, which is not in mode {mode}'s set ({','.join(in_set)})")
            if key in knobs and knobs[key] != value:
                raise AblationError(f"{ENV_ABLATE}: knob {key} given twice ({knobs[key]} and {value})")
            knobs[key] = value
            continue
        if "." in tok:
            raise AblationError(f"{ENV_ABLATE}: token {tok!r}: a knob needs a value (<lever>.<knob>=<value>)")
        if tok not in LEVERS:
            raise AblationError(f"{ENV_ABLATE}: unknown lever {tok!r} (the registry's names: {', '.join(sorted(LEVERS))})")
        if tok not in in_set:
            raise AblationError(f"{ENV_ABLATE}: lever {tok!r} is not in mode {mode}'s set ({','.join(in_set)}); nothing to ablate")
        if tok == "fused":
            raise AblationError(f"{ENV_ABLATE}: the numerics base 'fused' defines the kit line; it is not an ablatable lever (use --mode off --backend for the stock backends)")
        if tok not in levers:
            levers.append(tok)
    for k in knobs:
        lever = k.split(".", 1)[0]
        if lever in levers:
            raise AblationError(f"{ENV_ABLATE}: knob {k} set on lever {lever!r}, which the same variable ablates")
    remaining = [n for n in in_set if n not in levers]
    for dep, bases in DEPENDS.items():
        if dep in remaining:
            missing = [b for b in bases if b in levers]
            if missing:
                raise AblationError(f"{ENV_ABLATE}: lever {dep!r} rides on {', '.join(missing)}: ablate {dep!r} too (its module installs it only together with {', '.join(bases)})")
    return Ablation(tokens=toks, levers=tuple(levers), knobs=dict(knobs))
