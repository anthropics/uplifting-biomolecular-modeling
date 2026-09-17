"""The small-input floor of the single-card exact and fast lines (``S1``, ``LSTAR2A``): size-gated composition at activation.

Below ARM's own 300-token design gate every triangle-attention / TriMul call the arm binds is handed back to the stock kernel
(``design_below_gate_300``), the FPF TriMul adapter falls back below its crossover, the XL prologue returns the stock body un-chunked —
the trunk levers do no GPU work there, and their binds, adapters and wrappers cost host time per call in a regime that is launch-bound
already (a 200-token item is slower under the bound line than under stock); fast's bf16 DiT attention costs more in casts than bf16 saves
at that size. So when EVERY item of a ``pred`` call counts fewer residue tokens than the floor, the levers of ``FLOOR_LEVERS`` are composed
OUT of the line before anything is exported or installed
(``modes.line_without``): their switches absent, their units never bound, their LEVER rows ``state=off reason=below_gate:<tokens>/<gate>``
(``<tokens>`` = the largest item's residue tokens, ``<gate>`` = the floor), the ACTIVE and PRED lines carrying ``floor=below_gate:<tokens>/<gate>``.

The rule (:func:`policy`), one parameter — ``MODEL_OPT_SMALL_INPUT_FLOOR_TOKENS`` (configs/<gpu>.env; default ``FLOOR_DEFAULT`` = ARM's
design gate, so nothing changes at or above it):
  * every item below the floor            -> the levers composed out (``below``);
  * any item at or above the floor         -> the full composition, every lever bound: a query mixing sizes keeps the line whole and ARM's
                                             per-design gate serves the small items stock kernels, as it always has (``mixed`` / ``above``);
  * no query read (``check``, the env route) -> the static line (``unknown``): the DRY RUN line is the line as shipped;
  * ``0``                                  -> the floor off: bound at every size;
  * anything but a non-negative integer    -> refused by name (``SmallFloorRefusal``), nothing applied.
Residue tokens are the kit's one size unit (``big.count_tokens``: polymer residues x count; ligand atoms are the model's tokens, not the
gate's — an item whose residues are below the floor runs the stock kernels even when its ligands lift the model's own count past ARM's gate).

Contract with the package: ``plan(res, query_path)`` once in ``cli.pred`` before activation (kept in this process, like ``big.plan``);
``compose(line)`` from ``modes.resolve``; ``gated_off(line)`` / ``word()`` for the report (``stack.activate``: ``levers_gated_off``,
``small_input_floor``); ``_reset()`` for the CPU tests.
"""
from __future__ import annotations

import os
from typing import Optional

from . import modes
from .big import BELOW_GATE, count_tokens        # the size-gate vocabulary (`reason=below_gate:<tokens>/<gate>`) and the kit's one residue-token counter

GATE = "MODEL_OPT_SMALL_INPUT_FLOOR_TOKENS"        # configs/<gpu>.env; registry.KNOBS documents it
FLOOR_DEFAULT = "300"                              # = ARM's design gate (levers/ARMT odde_arm_t CFG MIN_TOKENS, ODDE_ARM_T_MIN_TOKENS): at or above it nothing changes
FLOOR_LEVERS = ("arm_z", "arm_u", "arm_u23", "fpf_trimul_exact", "triattn_core", "triattn_exact", "triattn_conf", "trimul_core", "trimul_exact", "ln_core", "transition_core", "transition_exact")   # composed out below the floor (CHANGES.md "Modes"): the trunk levers (ARM's binds, the FPF TriMul adapter, the triangle / LayerNorm / transition provider bindings); the DITFAST hoist (dit_hoist, dit_align) and the SAMPLER unit's sampler levers stay bound at every size
_ST = {"plan": None}


class SmallFloorRefusal(modes.OpenModeError):
    """The floor's variable malformed: refused by name (NOT ACTIVE), nothing applied."""


def _reset() -> None:
    _ST["plan"] = None


def gate_value(environ: Optional[dict] = None) -> int:
    """The floor in residue tokens from ``environ`` (default ``FLOOR_DEFAULT``); a value that is not a non-negative integer is refused by name."""
    env = os.environ if environ is None else environ
    raw = str(env.get(GATE) or FLOOR_DEFAULT).strip()
    if not raw.isdigit():
        raise SmallFloorRefusal(f"{GATE}={raw!r}: a non-negative integer (residue tokens; 0 = the levers bound at every size) is required")
    return int(raw)


def policy(tokens: Optional[dict], environ: Optional[dict] = None) -> dict:
    """The floor decision: ``{"gate", "max_tokens", "n_items", "below", "case", "reason"}`` over ``tokens`` = ``big.count_tokens(jobs)``
    (``{item: {"residue_tokens", "ligands_uncounted"}}``) or None when no query was read. ``below`` is True only when the gate is > 0 and
    every item counts fewer residue tokens than it (``case`` = below | above | mixed | off | unknown)."""
    gate = gate_value(environ)
    if tokens is None:
        return {"gate": gate, "max_tokens": None, "n_items": 0, "below": False, "case": "unknown",
                "reason": "no query read before activation: the static line (every lever bound)"}
    counts = [int(t["residue_tokens"]) for t in tokens.values()]
    mx, n = max(counts, default=0), len(counts)
    lig = sum(int(t.get("ligands_uncounted") or 0) for t in tokens.values())
    lig_note = f"; {lig} ligand/ion entit{'y' if lig == 1 else 'ies'} uncounted" if lig else ""
    if gate == 0:
        return {"gate": 0, "max_tokens": mx, "n_items": n, "below": False, "case": "off", "reason": f"{GATE}=0: the floor off (every lever bound at every size)"}
    if n and mx < gate:
        return {"gate": gate, "max_tokens": mx, "n_items": n, "below": True, "case": "below",
                "reason": f"every item below the floor (largest {mx} residue tokens < {gate}{lig_note}): {','.join(FLOOR_LEVERS)} composed out"}
    n_below = sum(1 for c in counts if c < gate)
    case = "mixed" if n_below else "above"
    return {"gate": gate, "max_tokens": mx, "n_items": n, "below": False, "case": case,
            "reason": (f"{n_below} of {n} items below the floor, the largest {mx} residue tokens >= {gate}{lig_note}: every lever bound (ARM's per-design gate serves the small items)"
                       if n_below else f"every item at or above the floor (largest {mx} residue tokens >= {gate}{lig_note}): every lever bound")}


def plan(res: modes.Resolution, query_path: Optional[str], environ: Optional[dict] = None) -> dict:
    """The pre-activation hook (``cli.pred``): the floor decision for a line the floor composes (``modes.SMALL_LINES``) from the query's
    residue tokens, kept in this process for :func:`compose`; ``{}`` for every other line (big: its own gates) and for ``off``."""
    if res.line is None or res.line.name not in modes.SMALL_LINES:
        _ST["plan"] = None
        return {}
    tokens = None
    if query_path:
        from . import inputs as _inputs
        tokens = count_tokens(_inputs.load_query(query_path))
    pol = policy(tokens, environ)
    pol["tokens"] = tokens
    _ST["plan"] = pol
    return pol


def planned() -> Optional[dict]:
    """This process's floor decision (None until :func:`plan` ran for a composed line)."""
    return dict(_ST["plan"]) if _ST["plan"] is not None else None


def word() -> Optional[str]:
    """``below_gate:<tokens>/<gate>`` when the plan composed the levers out, else None (the ACTIVE / PRED lines' ``floor=`` token, the LEVER reason)."""
    pol = _ST["plan"]
    if pol is None or not pol.get("below"):
        return None
    return f"{BELOW_GATE}:{pol['max_tokens']}/{pol['gate']}"


def compose(line: modes.Line, environ: Optional[dict] = None) -> modes.Line:
    """``line`` as the plan composes it: without ``FLOOR_LEVERS`` when every item is below the floor, else the line itself. The variable is
    read (and a malformed value refused) on every resolution, query or none, so ``check`` refuses a bad config before any ``pred``."""
    gate_value(environ)
    pol = _ST["plan"]
    if pol is None or not pol.get("below") or line.name not in modes.SMALL_LINES:
        return line
    return modes.line_without(line, FLOOR_LEVERS)


def gated_off(line: Optional[modes.Line]) -> dict:
    """``{lever: "below_gate:<tokens>/<gate>"}`` for the levers the floor composed out of ``line`` (the static row's levers absent from it); {} otherwise."""
    w = word()
    if line is None or w is None or line.name not in modes.SMALL_LINES:
        return {}
    static = modes.LINES[line.name].levers
    return {x: w for x in static if x in FLOOR_LEVERS and x not in line.levers}
