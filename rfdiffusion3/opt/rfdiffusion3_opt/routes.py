"""The routes table — the routes a pass runs under, the accelerator words each route expects, and the reader that maps
(mode, design line) to a route. Standard library only: this module loads in the pristine interpreter (the stock child, an outside
runner's stock arm) where the shared core is not necessarily importable; `modes` serves the same names for the kit routes.

  stock    `--mode off` + upstream's `compile_model=true`: upstream's fastest documented environment, pristine interpreter
  default  `--mode off`: upstream exactly as shipped, pristine interpreter
  exact    `--mode exact`: the kit's exact class (compile_model=true refused: COMPILE_UNDER_EXACT)
  fast     `--mode fast`: the kit's tolerance class (compile_model=true refused: COMPILE_UNDER_FAST)

Every row names the expected word KIND of the KERNELS census per accelerator — compile, atom_attn, the two dead surfaces, and `rmsnorm`
(apex's FusedRMSNorm engaged on every route: the pinned stack carries apex, STOCK.md "Pinned stack") — (census.py reads it, restating nothing) and whether the
pre-flight atom-attention decision gates the pass (`preflight_gate`: the KIT routes exact and fast, whose captured / compiled dense step
cannot serve upstream's sparse path — the pass is refused there by name; the stock routes take whatever path upstream's own memory rule
picks, dense or sparse, and the line NAMES it), and whether that decision is authoritative for the process (`attn_pin`: on the kit routes the
pre-flight evaluation of upstream's rule is pinned per attention shape key and every later query of the decision function answers from the
pin; the stock routes keep upstream's per-call evaluation). The stock routes carry no `atom_attn` expectation and `default` no `rmsnorm`
expectation: stock is upstream through its own surface — which attention path it picks, and which RMSNorm class its environment binds
(apex when the interpreter carries it, torch's otherwise), are facts the KERNELS line reports, never misses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

KERNEL_ACCELERATORS: Tuple[str, ...] = ("compile", "atom_attn", "cueq", "ds4sci", "rmsnorm")   # the KERNELS line's words, in order; census.ACCELERATORS restates it import-free (locked by test_census), every ROUTES row names all five
_NA_UPSTREAM = {"cueq": "n/a-upstream", "ds4sci": "n/a-upstream"}                # cueq / ds4sci: dead code at the pin (attention.py:282; pairformer_layers.py:38) — the same word on every route
_NA = {**_NA_UPSTREAM, "rmsnorm": "engaged"}        # cueq / ds4sci: dead code at the pin (attention.py:282; pairformer_layers.py:38); rmsnorm: apex's FusedRMSNorm bound as every RMSNorm of the network (layer_utils.py:13-23) — the pinned stack carries apex, every route expects it engaged


@dataclass(frozen=True)
class Route:
    name: str
    mode: str                                        # off | exact | fast
    compile: bool                                    # the line asks upstream's compile_model
    expected: Dict[str, str]                         # accelerator -> expected word KIND (engaged | off-by-route | absent | fallback | n/a-upstream); every row names all of KERNEL_ACCELERATORS
    preflight_gate: bool = False                     # the pre-flight atom-attention decision refuses the pass when sparse (kernels ATTN_PREFLIGHT): the routes whose attention calls the interpreter does not execute one by one
    attn_pin: bool = False                           # the pre-flight atom-attention decision is AUTHORITATIVE on this route: pinned per attention shape key for the process, every later query of upstream's decision function answers from the pin (kernels ATTN_PREFLIGHT pinned=1; the kit routes — stock is upstream's own per-call / trace-time rule); no switch restores per-call evaluation on a kit route
    enforce: bool = True                             # an accelerator word that misses its expected kind REFUSES the pass (the kit routes: a lever the mode planned must engage) | is REPORTED on a `KERNELS report-only` line and the pass proceeds (the stock routes: upstream completes such a pass itself — apex absent, dynamo declining a frame; the pre-flight dense bound, `preflight_gate`, is the one stop the stock route keeps)
    doc: str = ""


ROUTES: Dict[str, Route] = {
    "stock": Route("stock", "off", True, {"compile": "engaged", **_NA}, enforce=False,
                   doc="upstream `rfd3 design … compile_model=true` in the pristine interpreter: upstream's fastest documented environment (rfdiffusion3.yaml:70-71, whose comment documents the steady-state roll-out saving; engine.py:226-235)"),
    "default": Route("default", "off", False, {"compile": "off-by-route", **_NA_UPSTREAM}, enforce=False,
                     doc="upstream `rfd3 design …` exactly as shipped (compile_model: False, rfdiffusion3.yaml:71) in the pristine interpreter"),
    "exact": Route("exact", "exact", False, {"compile": "off-by-route", "atom_attn": "engaged", **_NA}, preflight_gate=True, attn_pin=True,
                   doc="KIT_MODES exact (hoist + fused transition + graph sampler: the eager step's kernels captured and replayed, so its attention calls run outside the interpreter after the capture); "
                       "bit-identical designs vs EAGER upstream under the deterministic recipe — compile_model=true is refused under exact (COMPILE_UNDER_EXACT)"),
    "fast": Route("fast", "fast", False, {"compile": "engaged", "atom_attn": "engaged", **_NA}, preflight_gate=True, attn_pin=True,
                  doc="KIT_MODES fast (graph sampler + hoist + fused transition + RFD3_COMPILE + RFD3_TOKEN_SDPA + RFD3_GATHER_ATTN): expects compile engaged in the KIT form (`engaged:kit:…`) and atom_attn engaged in the gather form (`engaged:gather_attn(…)[served=<n>,stock=0]`; `partial:gather_attn(…)` — some atom calls refused by name — is refused like a fallback) — "
                      "torch.compile of upstream's four targets applied by the kit at the first roll-out, its cache accessors as dynamo boundaries, the compiled step inside the captured graph"),
}
COMPILE_UNDER_EXACT = ("compile_model=true under exact: torch.compile (inductor) reorders floating-point reductions, and exact's class is bit-identical designs "
                       "against EAGER upstream — `--mode off` runs this line as stock (compiled upstream, route stock)")
COMPILE_UNDER_FAST = ("compile_model=true under fast: fast compiles upstream's four targets itself (RFD3_COMPILE=1, applied at the first roll-out with the kit's cache accessors and "
                      "the shared core's fused-transition kernel as dynamo boundaries); upstream's own switch wraps the same modules at initialize() without those boundaries — inductor "
                      "cannot re-emit the core's Triton kernel (`NameError: _ld is not defined`) and the pass would die in its first roll-out — `--mode off` runs this line as stock "
                      "(compiled upstream, route stock); `--mode fast` without the token is the kit's compiled route")
COMPILE_REFUSED = {"exact": COMPILE_UNDER_EXACT, "fast": COMPILE_UNDER_FAST}      # kit mode -> the NOT ACTIVE reason for upstream's compile switch on its line


def route_of(mode: str, tokens=None) -> Route:
    """The route a pass runs under: its mode plus whether its design line asks upstream's compile switch (settings.compile_requested over the
    line's key=value tokens). Raises ValueError, by name, for a combination that is not a route (compile under exact | fast)."""
    from .settings import compile_requested
    want = compile_requested(tokens or [])
    for r in ROUTES.values():
        if r.mode == mode and r.compile == want:
            return r
    if want and mode in COMPILE_REFUSED:
        raise ValueError(COMPILE_REFUSED[mode])
    raise ValueError(f"mode {mode!r} with compile_model={'true' if want else 'false'} is not a route ({'|'.join(ROUTES)})")


def route_refusal(tokens, mode: str) -> Optional[str]:
    """The NOT ACTIVE reason when `mode` has no route for this line (compile under exact | fast), else None."""
    try:
        route_of(mode, tokens)
    except ValueError as e:
        return str(e)
    return None
