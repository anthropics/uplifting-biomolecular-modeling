"""The mode table — the one place a mode is defined (`KIT_MODES`), and how a mode resolves to the kit's own switches.

Package modes (`MODES`, an `opt_core.modes.ModeTable`; `DEFAULT_MODE` is the one default every command reads):
  exact   the tier-1 line, selected by name: `PXD_HOIST=1 PXD_HOIST_MODE=shape PXD_HOIST_MASK=1` (every value the
          kit's own code applies when the switch is unset, `registry.KIT_SWITCHES`), installed at the kit's activation point
          after checkpoint load by `pxd_xattempt.hoist.install(runner.model)`.
          Its claim: byte-identical to stock under the deterministic recipe (det.py).
          `PXDESIGN_OPT` unset is `off` (_autoload.py installs nothing).
  fast    the package default (`DEFAULT_MODE = FAST_MODE`) — speed inside the engine's tier-2 band: exact's lever set with `PXD_HOIST_MODE=rows` plus the package levers
          `featdiet`, `padmask` (exact by construction), `tf32` (precision.py: the numerics policy — the process's fp32 matmuls as TF32
          tensor-core products, fp32 accumulate; `opt_core.precision.policy`), `sdedup` (sdedup.py: the token path's single conditioning
          and its per-block projections on ONE sample row, broadcast over N_sample; `opt_core.capture.hoist.RowDedup` through the hoist kit's
          own hook `pxd_xattempt.fuse`). Tolerance class (tier 2): not byte-identical to stock; deterministic run to run under the recipe.
  big   the memory-reach line, never the default: exact's lever set with `PXD_HOIST_MODE=rows` — the hoisted LayerNorm/Linear
          evaluated once on the un-expanded [N_tok^2, c] rows and broadcast instead of on the N_sample-expanded rows (`hoist.py:437`:
          no [N_sample, N_tok, N_tok, c] fp32 transient at prepare) — plus the package's three size levers (`registry.PACKAGE_LEVERS`;
          `sizeceil.py`, `rowpipe.py`): `featdiet` (the [N_atom, N_atom] int64 `bond_mask` never copied to the GPU), `padmask` (the atom
          transformer's windowed padding mask / bias built from indices, never sliced out of a dense [N_atom, N_atom] tensor) and
          `rowpipe` (the hoist prepare's three pair-plane passes evaluated in balanced row slabs (rowpipe.slab_rows) with no resident pair_z: no
          [N_tok, N_tok, c] fp32 plane is materialised), and `fast`'s two tolerance levers `tf32` and `sdedup` (the slabbed passes are
          fp32 GEMMs: TF32 is where the big mode's time goes at large N_tok). NOT byte-identical to stock — `fast`'s numerics class (the `rows` evaluation, the row slabs,
          TF32, the dedup; featdiet and padmask are exact by construction), tier 2. OOM propagates: no fallback is applied at or above its ceiling.
  off     stock: the upstream CLI (`pxdesign infer`) or, under the deterministic recipe, the in-process upstream route, in a clean
          subprocess with nothing from the kit on the path (stock_infer.py).
There is no variant axis (one checkpoint, `stock/PINS.json` "variants"); `--variant` is not accepted.
`--mode` / `PXDESIGN_OPT` accept exactly `MODES`; any other name is refused by name with one sentence (`unknown_message`, the table's
own: the CLI prints it with exit status 2, `enable()` raises it, the env route's finder refuses at the trigger with exit status 3).
The switch values a mode exports are the kit's own defaults, read from `registry.KIT_SWITCHES` (one copy; the tests lock that column
to `hoist.py`); this table names which switches each mode exports, which hoist levers it plans and which package levers it selects.
`jit_cache_key()` is the running stack's key for the JIT cache (`opt_core.jit_cache.key`; `configs/h100.env` exports it as
`MODEL_OPT_STACK_KEY`; a part that cannot be established is refused by name, never rendered into a cache directory).

`resolve(mode)` returns the Resolution every command uses (design, check, warm): the levers the mode plans, the switch values
it exports, and the switches it drops from the caller's environment so that this table stays the source of truth.
`kit_switch_defaults(kit_home)` reads the defaults out of the kit file itself (`os.environ.get("<switch>", "<default>")` literals of
hoist.py, no torch needed): the tests lock `KIT_MODES["exact"].env` to them, so the table never drifts from the kit.
"""
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from opt_core.modes import ModeError, ModeTable

from .registry import KIT_RELPATH, KIT_SWITCHES, LEVER_FILE, LEVERS, PACKAGE_LEVERS

ENV = "PXDESIGN_OPT"                                     # the mode variable (read by _autoload.py, the CLI and run.sh)
MODES: Tuple[str, ...] = ("off", "exact", "fast", "big")   # every name --mode / PXDESIGN_OPT accepts (the house table off|exact|fast|big)
FAST_MODE: Optional[str] = "fast"                        # the tolerance-class speed line (see the module docstring)
DEFAULT_MODE: str = FAST_MODE                            # the default is `fast` (tier 2); `exact` (tier 1) and `big` are selected by name; PXDESIGN_OPT unset = off


def unknown_message(value) -> str:
    """The one sentence for a name outside the table (the CLI's usage error, enable()'s ValueError, run.sh's refusal carry these bytes)."""
    return f"unknown mode {value!r}; expected one of {'|'.join(MODES)}"


TABLE = ModeTable(modes=MODES, default=DEFAULT_MODE, unknown_message=unknown_message)
SWITCHES: Tuple[str, ...] = tuple(KIT_SWITCHES)                     # every switch the package drops from the caller's environment
EXACT_SWITCHES: Tuple[str, ...] = ("PXD_HOIST", "PXD_HOIST_MODE", "PXD_HOIST_MASK")   # the switches every mode of the kit exports
HOIST_LEVERS: Tuple[str, ...] = tuple(LEVERS)            # h1..h5: every mode of the kit plans the whole hoist
SIZE_LEVERS: Tuple[str, ...] = ("featdiet", "padmask", "rowpipe")   # the package's size levers (sizeceil.py, rowpipe.py); `big` selects all three, `fast` the first two
FAST_LEVERS: Tuple[str, ...] = ("featdiet", "padmask", "tf32", "sdedup")   # `fast`: the two exact-by-construction size levers + the numerics policy (precision.py) + the single-conditioning row dedup (sdedup.py)
BIG_LEVERS: Tuple[str, ...] = SIZE_LEVERS + ("tf32", "sdedup")           # `big`: the three size levers + `fast`'s two tolerance levers (the slabbed pair-plane GEMMs under TF32, the single-conditioning dedup)
ROWS_VALUE = "rows"                                       # PXD_HOIST_MODE's other value (hoist.py:437, registry.KIT_SWITCHES): `big` exports it


@dataclass(frozen=True)
class ModeSpec:
    name: str
    levers: Tuple[str, ...]                               # registry.LEVERS names the mode applies (the hoist families)
    env: Dict[str, str]                                   # the kit switches the mode exports (values = the kit's own, or the named other value)
    promise: str
    package_levers: Tuple[str, ...] = ()                  # registry.PACKAGE_LEVERS names the mode selects (applied by stack.py around the runner)
    tier: str = "1"                                       # "1" = byte-identical to stock under the deterministic recipe; "2" = tolerance class, within the engine's stated numeric bound

    @property
    def all_levers(self) -> Tuple[str, ...]:
        return tuple(self.levers) + tuple(self.package_levers)


_EXACT_ENV = {k: KIT_SWITCHES[k][0] for k in EXACT_SWITCHES}          # the kit's own defaults (registry.KIT_SWITCHES, locked to hoist.py by the tests)

KIT_MODES: Dict[str, ModeSpec] = {
    "exact": ModeSpec("exact", HOIST_LEVERS, dict(_EXACT_ENV),
                      "byte-identical to stock under the deterministic recipe; the kit's default line, eager",
                      package_levers=(), tier="1"),
    "big": ModeSpec("big", HOIST_LEVERS, dict(_EXACT_ENV, PXD_HOIST_MODE=ROWS_VALUE),
                      "memory reach: exact's levers with the hoisted LayerNorm/Linear on un-expanded rows (hoist.py:437) plus the size levers featdiet + padmask "
                      "(sizeceil.py) + rowpipe (rowpipe.py: the prepare's pair-plane passes in row slabs, no resident pair_z) and fast's tolerance levers tf32 + sdedup "
                      "— 2788 (`exact`) → 5988 tokens (`big`) on one 80 GB H100 at N_sample 5; "
                      "NOT byte-identical to stock — fast's numerics class, tier 2; never the default; OOM propagates, no fallback applied",
                      package_levers=BIG_LEVERS, tier="2"),
    "fast": ModeSpec("fast", HOIST_LEVERS, dict(_EXACT_ENV, PXD_HOIST_MODE=ROWS_VALUE),
                     "speed inside the engine's tier-2 band: exact's levers with the hoisted LayerNorm/Linear on un-expanded rows (hoist.py:437), the size levers featdiet + padmask, "
                     "the numerics policy tf32 (precision.py: the process's fp32 matmuls as TF32 tensor-core products, fp32 accumulate) and sdedup (sdedup.py: the token path's "
                     "single conditioning and its six per-block projections on one sample row, broadcast over N_sample); NOT byte-identical to stock — tolerance class, tier 2; "
                     "the package default; OOM propagates, no fallback applied",
                     package_levers=FAST_LEVERS, tier="2"),
    "off": ModeSpec("off", (), {}, "stock — nothing from the kit on the path, no switch set", tier="stock"),
}



@dataclass
class Resolution:
    mode: str
    levers: Tuple[str, ...]
    env: Dict[str, str]
    package_levers: Tuple[str, ...] = ()
    tier: str = "1"
    dropped: List[str] = field(default_factory=list)      # caller-set switches the activation removes
    notes: List[str] = field(default_factory=list)

    @property
    def levers_planned(self) -> List[str]:
        return list(self.levers)                           # the hoist families (the ACTIVE line's levers=); package levers are named beside them (package_levers=)

    @property
    def is_stock(self) -> bool:
        return self.mode == "off"


def check_mode(mode: Optional[str]) -> str:
    """The table's own check; an absent value is refused here (the CLI resolves the default before it asks)."""
    if mode is None or not str(mode).strip():
        raise ValueError(unknown_message(mode))
    try:
        return TABLE.check(mode)
    except ModeError as e:
        raise ValueError(str(e)) from None


def resolve(mode: Optional[str], environ=None) -> Resolution:
    """Resolve a package mode to its lever set and switch values; never applies anything."""
    m = check_mode(mode)
    spec = KIT_MODES[m]
    environ = os.environ if environ is None else environ
    present = [k for k in SWITCHES if k in environ]
    notes = []
    if present:
        notes.append(f"caller environment sets {present}: activation drops them (the mode table decides)")
    return Resolution(m, spec.levers, dict(spec.env), package_levers=tuple(spec.package_levers), tier=spec.tier, dropped=present, notes=notes)


_GET_RX = re.compile(r"""os\.environ\.get\(\s*"(?P<name>PXD_HOIST[A-Z_]*)"\s*,\s*"(?P<default>[^"]*)"\s*\)""")


def kit_switch_defaults(kit_home: str) -> Dict[str, str]:
    """The kit's own defaults, read out of hoist.py (`os.environ.get("<switch>", "<default>")` literals)."""
    path = os.path.join(kit_home, LEVER_FILE)
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    out: Dict[str, str] = {}
    for m in _GET_RX.finditer(src):
        out.setdefault(m.group("name"), m.group("default"))
    return out


def lever_table() -> List[dict]:
    """The per-lever rows (name, what, class, tier, switch, code) for check --json, manifests and docs; data, not prose."""
    rows = [{"lever": n, "what": lv.what, "class": lv.cls, "tier": lv.tier, "switch": lv.switch, "code": lv.code, "prepare": lv.prepare} for n, lv in LEVERS.items()]
    rows += [{"lever": n, "what": lv.what, "class": lv.cls, "tier": lv.tier, "switch": lv.switch, "code": lv.code, "prepare": lv.prepare,
              "modes": [m for m, s in KIT_MODES.items() if n in s.package_levers]} for n, lv in PACKAGE_LEVERS.items()]
    return rows


def mode_table() -> List[dict]:
    """The per-mode rows (name, tier, levers, env, default) for check --json and manifests."""
    return [{"mode": m, "tier": s.tier, "levers": list(s.all_levers), "env": dict(s.env),
             "default": m == DEFAULT_MODE, "promise": s.promise} for m, s in KIT_MODES.items()]


def kit_home(tree_home: str) -> str:
    return os.path.join(tree_home, KIT_RELPATH)


def jit_cache_key(version: Optional[str] = None, cuda: Optional[str] = None, cc: Optional[str] = None, *, strict: bool = True) -> str:
    """The JIT cache key, ``torch<version>-cu<cuda>-sm<cc>`` — the core's rule (`opt_core.jit_cache.key`): the torch version without its
    local tag, the CUDA version without the dot, the device's compute capability digits (e.g. ``torch2.3.1-cu121-sm90``). A part that is
    neither passed nor derivable on this box raises ``opt_core.jit_cache.StackKeyUnknown`` naming it — ``configs/h100.env``, which exports the
    key as ``MODEL_OPT_STACK_KEY`` (a pre-set value is kept) and keys ``TORCH_EXTENSIONS_DIR`` (torch's JIT-extension root; Protenix's LayerNorm extension builds
    inside its own package, not there) by it, refuses then (rc 3: set the key explicitly). ``strict=False`` renders such a part ``unknown``: a display-only key (the manifest's
    box-as-found block), never a cache directory."""
    from opt_core.jit_cache import key
    return key(version, cuda, cc, strict=strict)


__all__ = ["ENV", "MODES", "DEFAULT_MODE", "FAST_MODE", "TABLE", "EXACT_SWITCHES", "HOIST_LEVERS", "SIZE_LEVERS", "FAST_LEVERS", "BIG_LEVERS", "ROWS_VALUE", "KIT_MODES", "ModeSpec", "Resolution",
           "resolve", "check_mode", "unknown_message", "kit_switch_defaults", "lever_table", "mode_table", "kit_home", "jit_cache_key", "SWITCHES"]
