"""Upstream's own arguments, read the way upstream reads them.

``scripts/run_inference.py`` is a Hydra application: positional ``KEY=VALUE`` overrides over ``config/inference/<name>.yaml`` (``base``
unless ``--config-name`` names another) plus Hydra's own flags. Every mode of this package takes exactly that argument list. ``--mode off``
hands it to upstream's command line unchanged (design.py, stock_cli.py). A kit mode (design.py) reads from it the one target a ``run_inference.py`` invocation
designs — input structure, contigs, hotspots, design count and start number, output prefix, checkpoint directory — and composes every
other typed override onto the resident driver's per-design configuration verbatim (driver_run.py ``--compose``: plain, ``+KEY``, ``++KEY`` and
``~KEY`` forms alike), where upstream's own sampler, denoiser, potential and contig code read it: guiding potentials, the noise schedules,
partial diffusion (``diffuser.partial_T``), ``contigmap.length``, ``inference.align_motif`` / ``final_step`` and the rest of
``config/inference/base.yaml`` are served by every mode.

What a kit mode cannot serve it refuses by name before anything runs (``refused()``: design.py prints ``[rfdiffusion1-opt] NOT ACTIVE: mode=<m>
cannot serve …`` and exits 3; ``--mode off`` runs every such request on upstream's command line): the features of ``REFUSED_SWITCHES`` /
``REFUSED_VALUES`` / ``REFUSED_GROUPS`` below — each names the mechanism — plus a primary config other than ``base``
(``--config-name symmetry``), Hydra's application flags (``--cfg``, ``--multirun``, …: they act in ``hydra.main()``, which the resident driver
does not run) and a token that is neither an override nor a flag.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

CONFIG_NAME_FLAGS = ("--config-name", "-cn")                    # Hydra's flag naming the primary config (run_inference.py docstring: `--config-name symmetry`)
# Hydra's own command-line flags (hydra 1.3 `python app.py --hydra-help`): name -> takes a value token. `--help` stays this package's.
HYDRA_FLAGS = {"--config-name": True, "-cn": True, "--config-path": True, "-cp": True, "--config-dir": True, "-cd": True,
               "--cfg": True, "-c": True, "--package": True, "-p": True, "--experimental-rerun": True,
               "--resolve": False, "--run": False, "-r": False, "--multirun": False, "-m": False, "--shell-completion": False, "-sc": False,
               "--hydra-help": False, "--info": False, "-i": False}
INFO_WORDS = ("all", "config", "defaults", "defaults-tree", "plugins", "searchpath")   # `--info`'s optional value
STOCK_CONFIG = "base"                                           # config/inference/base.yaml: the configuration the resident driver composes

# What a kit mode refuses by name (`refused()`); every other key is served — composed verbatim onto the resident driver's configuration.
# key -> (feature, mechanism); refused when the typed value switches the feature on (a true boolean / a non-null value).
REFUSED_SWITCHES = {
    "inference.symmetry": ("symmetric oligomers", "the resident driver restates SelfConditioning.sample_step without upstream's symmetry branches and composes config/inference/base.yaml, not symmetry.yaml"),
    "inference.cyclic": ("cyclic peptides", "upstream's per-call cyclic index loops in the positional embedding (torch.unique, mask writes) cannot run inside lever W1's captured forward graph"),
    "inference.empty_cache_per_design": ("emptying the CUDA cache per design", "it would evict the resident allocator and graph pools levers U1 and W1 keep across designs"),
    "logging.inputs": ("the model-input pickler", "it wraps the model's forward, the attribute lever W1 replaces with the captured graph"),
    "scaffoldguided.scaffoldguided": ("fold-conditioned design", "unproven on the kit line: its checkpoints (Complex_Fold_base_ckpt.pt, InpaintSeq_Fold_ckpt.pt) are outside the kit's pinned weights"),
    "contigmap.inpaint_seq": ("sequence inpainting", "unproven on the kit line: upstream selects InpaintSeq_ckpt.pt for it, outside the kit's pinned weights"),
    "contigmap.provide_seq": ("sequence inpainting", "unproven on the kit line: upstream selects InpaintSeq_ckpt.pt for it, outside the kit's pinned weights"),
    "contigmap.inpaint_str": ("structure inpainting", "unproven on the kit line: upstream selects InpaintSeq_ckpt.pt for it, outside the kit's pinned weights"),
}
# How upstream reads a switch, so `switched_on()` reads it the same way and fails closed: a boolean switch is tested for truth (`if conf.inference.cyclic:`),
# so only OmegaConf's falsy parses — false / 0 / null / the empty word — are off and every other spelling (True, 1, yes, Y, T, 2, …) is on;
# a valued switch is tested `is not None`, so only null is off.
BOOLEAN_SWITCHES = ("inference.cyclic", "inference.empty_cache_per_design", "logging.inputs", "scaffoldguided.scaffoldguided")
BOOL_OFF_WORDS = ("false", "0", "null", "")
NULL_WORDS = ("null",)
# key -> (served values, mechanism for any other value)
REFUSED_VALUES = {
    "inference.model_runner": (("SelfConditioning",), "the resident driver restates SelfConditioning.sample_step (`default` is upstream's legacy loop without the self-conditioning template; ScaffoldedSampler is fold-conditioned design)"),
}
# key group (prefix) -> (feature, mechanism); SERVED_IN_GROUPS are the exceptions
REFUSED_GROUPS = {
    "model.": ("the network's architecture keys", "upstream re-applies a typed model/diffuser/preprocess key over the checkpoint's value from hydra.main()'s override record, which the resident driver's hydra.compose() does not create — and levers W1, T2 and P are written for the trained architecture"),
    "preprocess.": ("the network's input-feature keys", "upstream re-applies a typed model/diffuser/preprocess key over the checkpoint's value from hydra.main()'s override record, which the resident driver's hydra.compose() does not create — and lever P restates _preprocess for the trained features"),
    "diffuser.": ("the trained noise schedule", "upstream re-applies a typed model/diffuser/preprocess key over the checkpoint's value from hydra.main()'s override record, which the resident driver's hydra.compose() does not create (diffuser.partial_T, absent from the checkpoint, is served)"),
    "hydra.": ("Hydra's run settings", "they configure hydra.main()'s run directory and logging; the resident driver composes through hydra.compose()"),
}
SERVED_IN_GROUPS = ("diffuser.partial_T",)
# base.yaml's own values for the target keys the kit reads (stock/src/config/inference/base.yaml:3-8).
TARGET_DEFAULTS = {"inference.num_designs": 10, "inference.design_startnum": 0, "inference.output_prefix": "samples/design"}
TRUE_WORDS, FALSE_WORDS = ("true", "1", "yes", "on"), ("false", "0", "no", "off", "null", "none", "")
_OV = re.compile(r"^(\+\+|\+|~)?([A-Za-z_][\w.\-/@]*)(?:=(.*))?$", re.S)


@dataclass
class UpstreamArgs:
    """The argument list as typed (``argv``, order kept), split into Hydra flags and ``KEY=VALUE`` overrides."""
    argv: List[str]
    overrides: List[Tuple[str, str, str]] = field(default_factory=list)   # (prefix '' | '+' | '++' | '~', key, value) in typed order
    flags: List[str] = field(default_factory=list)                       # Hydra's flags, a value token folded in as `flag value`
    config_name: Optional[str] = None
    stray: List[str] = field(default_factory=list)                       # tokens that are neither (upstream's parser reports them)

    def typed(self) -> Dict[str, str]:
        """{key: value} of the plain and ``++`` (force) overrides — the value a key of base.yaml ends with (last one wins, as Hydra applies them)."""
        return {k: v for p, k, v in self.overrides if p in ("", "++")}

    def compose_tokens(self) -> List[str]:
        """The override tokens in typed order (Hydra flags and stray words left out): what the resident driver composes."""
        return [t for kind, toks in tokens(self.argv) if kind == "override" for t in toks]

    def value(self, key: str, default=None):
        return self.typed().get(key, default)


def tokens(argv: List[str]) -> List[Tuple[str, List[str]]]:
    """The one tokenizer of upstream's argument grammar: [(kind, tokens)] in typed order, kind = "flag" (a Hydra flag and its value token,
    `--flag value` | `--flag=value` | `--info [word]`), "override" (KEY=VALUE, +KEY=…, ++KEY=…, ~KEY[=…]) or "stray"."""
    out, i = [], 0
    while i < len(argv):
        t = argv[i]
        name = t.split("=", 1)[0]
        if t.startswith("-"):
            takes = HYDRA_FLAGS.get(name, False) and "=" not in t and i + 1 < len(argv)
            optional = name in ("--info", "-i") and i + 1 < len(argv) and argv[i + 1] in INFO_WORDS
            if takes or optional:
                out.append(("flag", [t, argv[i + 1]]))
                i += 2
                continue
            out.append(("flag" if name in HYDRA_FLAGS else "stray", [t]))
        else:
            m = _OV.match(t)
            out.append(("override", [t]) if m and (m.group(3) is not None or m.group(1) == "~") else ("stray", [t]))
        i += 1
    return out


def parse(argv: List[str]) -> UpstreamArgs:
    ua = UpstreamArgs(argv=list(argv))
    for kind, toks in tokens(argv):
        if kind == "flag":
            ua.flags.append(" ".join(toks))
            name = toks[0].split("=", 1)[0]
            if name in CONFIG_NAME_FLAGS:
                ua.config_name = toks[1] if len(toks) > 1 else (toks[0].split("=", 1)[1] if "=" in toks[0] else None)
        elif kind == "override":
            m = _OV.match(toks[0])
            ua.overrides.append((m.group(1) or "", m.group(2), m.group(3)))
        else:
            ua.stray.extend(toks)
    return ua


def split_hydra_flags(argv: List[str]) -> Tuple[List[str], List[str]]:
    """(Hydra's own flags with their value tokens, in typed order; every other token in typed order) — cli.main lifts the flags out
    before the kit's own argument parser sees the line."""
    flags, rest = [], []
    for kind, toks in tokens(argv):
        (flags if kind == "flag" else rest).extend(toks)
    return flags, rest


def as_bool(v, default: bool) -> bool:
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in TRUE_WORDS:
        return True
    if s in FALSE_WORDS:
        return False
    return default


def as_int(v, default: int) -> int:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return default


def switched_on(key: str, value: Optional[str]) -> bool:
    """Whether a typed value switches `key` on, read as upstream reads it and failing closed (BOOLEAN_SWITCHES / BOOL_OFF_WORDS / NULL_WORDS):
    any spelling upstream would take as on counts as on."""
    s = "" if value is None else str(value).strip().lower()
    return s not in (BOOL_OFF_WORDS if key in BOOLEAN_SWITCHES else NULL_WORDS)


def flag_parts(flag: str) -> Tuple[str, Optional[str]]:
    """("--config-name symmetry" | "-cn=symmetry" | "--resolve") -> (name, its own value or None) for a flag as `parse()` records it."""
    name, sep, value = flag.partition(" ")
    if sep:
        return name, value
    name, sep, value = flag.partition("=")
    return name, (value if sep else None)


def is_override(token: str) -> bool:
    """A Hydra override token (KEY=VALUE, +KEY=…, ++KEY=…, ~KEY[=…]) — the grammar of `tokens()`, for callers sorting a mixed argument list."""
    return not token.startswith("-") and tokens([token])[0][0] == "override"


def refused(ua: UpstreamArgs) -> List[str]:
    """What a kit mode cannot serve in this request, one reason per token (empty = the kit line serves it whole): `<feature> [<token>]: <mechanism>`.
    design.refuse_unserved prints them on the NOT ACTIVE line and exits 3 before anything runs; `--mode off` is the route that runs them."""
    why = []
    for f in ua.flags:
        name, value = flag_parts(f)                                            # each flag answers for its own value
        if name in CONFIG_NAME_FLAGS:
            if value is None:                                                  # the flag without the name it takes: nothing to compose (upstream's own parser refuses it too)
                why.append(f"Hydra flag [{f}] without its value (usage: {name} <config>): the resident driver composes config/inference/{STOCK_CONFIG}.yaml")
            elif value != STOCK_CONFIG:                                        # naming base.yaml itself is served (the resident driver composes it)
                why.append(f"primary config {value}.yaml [{f}]: the resident driver composes config/inference/{STOCK_CONFIG}.yaml (symmetry.yaml is symmetric oligomer design)")
        else:
            why.append(f"Hydra flag [{f}]: it acts in hydra.main(), which the resident driver does not run (the configuration is composed through hydra.compose())")
    seen = set()
    for p, k, v in ua.overrides:
        r = _refused_key(p, k, v)
        if r and r not in seen:
            seen.add(r)
            why.append(r)
    if ua.stray:
        why.append(f"argument{'s' if len(ua.stray) > 1 else ''} {ua.stray!r}: not a Hydra override (KEY=VALUE) or flag")
    return why


def _refused_key(prefix: str, key: str, value: Optional[str]) -> Optional[str]:
    token = f"{prefix}{key}" + (f"={value}" if value is not None else "")
    if prefix == "~":                                                          # deleting a key switches nothing on
        return None
    if key in REFUSED_SWITCHES:
        if switched_on(key, value):                                            # read as upstream reads it, failing closed (a spelling upstream takes as on is refused, never served)
            feature, why = REFUSED_SWITCHES[key]
            return f"{feature} [{token}]: {why}"
        return None
    if key in REFUSED_VALUES:
        served, why = REFUSED_VALUES[key]
        return None if str(value).strip() in served else f"{key}={value} [{token}]: {why}"
    if key in SERVED_IN_GROUPS:
        return None
    for group, (feature, why) in REFUSED_GROUPS.items():
        if key.startswith(group):
            return f"{feature} [{token}]: {why}"
    return None


def existing_indices(prefix: str) -> List[int]:
    """Design indices already on disk at `prefix` — upstream's own scan (run_inference.py:58-69: `<prefix>*.pdb`, the trailing `_<i>`)."""
    out = []
    for e in glob.glob(prefix + "*.pdb"):
        m = re.match(r".*_(\d+)\.pdb$", e)
        if m:
            out.append(int(m.group(1)))
    return sorted(set(out))


@dataclass
class Target:
    """The one target of a run_inference.py invocation, in the resident driver's case grammar."""
    prefix: str                     # absolute output prefix: designs are <prefix>_<i>.pdb / .trb, trajectories <dirname>/traj/<basename>_<i>_*.pdb
    pdb: str                        # inference.input_pdb as typed ('null' = upstream's own default input)
    contigs: str
    hotspots: str
    startnum: int                   # first design index after upstream's -1 rule (run_inference.py:57-69)
    num_designs: int
    ckpt_override: Optional[str]

    @property
    def name(self) -> str:
        return os.path.basename(self.prefix) or "design"


def target(ua: UpstreamArgs, cwd: Optional[str] = None) -> Target:
    """Read the target from the typed overrides with upstream's defaults and upstream's start-number rule."""
    cwd = cwd or os.getcwd()
    prefix = ua.value("inference.output_prefix") or TARGET_DEFAULTS["inference.output_prefix"]
    prefix = prefix if os.path.isabs(prefix) else os.path.join(cwd, prefix)
    n = as_int(ua.value("inference.num_designs"), TARGET_DEFAULTS["inference.num_designs"])
    start = as_int(ua.value("inference.design_startnum"), TARGET_DEFAULTS["inference.design_startnum"])
    if start == -1:
        on_disk = existing_indices(prefix)
        start = (max(on_disk) + 1) if on_disk else 0
    return Target(prefix=prefix, pdb=ua.value("inference.input_pdb") or "null", contigs=ua.value("contigmap.contigs") or "null",
                  hotspots=ua.value("ppi.hotspot_res") or "null", startnum=start, num_designs=n,
                  ckpt_override=ua.value("inference.ckpt_override_path"))
