"""The recipe accessor — the design recipe as the kit driver runs it, obtained by IMPORTING the kit's own recipe module
(`opt/mosaic_opt/tools/recipe.py`, the one file the driver `tools/public_design_run.py` builds its design through: model load,
featurization with the target's staged MSA or single-sequence, the loss composition, x0, the two `simplex_APGM` stages, the refold). Nothing
of the recipe is typed here: this package's routes (and any caller that builds a design step) read the literals and call the functions of
that module, exactly as the stock caller does.

    from mosaic_opt import recipe
    R = recipe.recipe(binder_length=80)                                   # the public target, single-sequence
    R = recipe.recipe(target_fasta="t.fasta", msa="t.a3m", binder_length=120)
    R["module"]     # tools/recipe.py, imported: .load_model .featurize .build_loss .x0 .run_stage .eval_step … (jax / mosaic load inside them)
    R["target"], R["tokens"], R["stages"], R["phases"], R["kind_of_phase"], R["refold"], R["loss_terms"], R["boltz2_loss"], R["flags"]

Importing this module or calling `recipe()` loads no jax and no upstream package (CPU, no GPU): the kit module defers those imports to
its functions. `module()` finds the kit directory the way the package does everywhere (stack-free: modes.DRIVER_RELPATH under the kit home).
"""
from __future__ import annotations

import importlib.util
import os
import sys
from typing import Optional

KIT_RECIPE_RELPATH = os.path.join("tools", "recipe.py")     # under the kit home (opt/mosaic_opt/)
_MODULES = {}


def kit_home_default() -> str:
    """The kit home: MOSAIC_OPT_KIT when set, else this package's own directory (stack.kit_home's rule without importing stack, which
    pulls the shared core)."""
    env = os.environ.get("MOSAIC_OPT_KIT")
    if env:
        return os.path.abspath(env)
    return os.path.dirname(os.path.abspath(__file__))


def module(kit_home: Optional[str] = None):
    """The kit's tools/recipe.py imported by path (cached per kit home) — the same module object the driver imports as `recipe`."""
    home = os.path.abspath(kit_home or kit_home_default())
    if home in _MODULES:
        return _MODULES[home]
    path = os.path.join(home, KIT_RECIPE_RELPATH)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path}: the kit's recipe module is not under kit home {home} (MOSAIC_OPT_KIT / the package directory opt/mosaic_opt)")
    tools = os.path.dirname(path)
    if tools not in sys.path:
        sys.path.insert(0, tools)                                # the module imports the public target's constants from fetch_public_inputs beside it, as the driver does
    spec = importlib.util.spec_from_file_location("mosaic_kit_recipe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _MODULES[home] = mod
    return mod


def recipe(kit_home: Optional[str] = None, *, binder_length: int, target_fasta: Optional[str] = None, target_seq: Optional[str] = None,
           target_name: Optional[str] = None, target_copies: int = 1, msa: Optional[str] = None, seed: int = 0,
           steps1: Optional[int] = None, steps2: Optional[int] = None, first_record: bool = False, epitope: Optional[str] = None) -> dict:
    """The recipe as data + the kit module that runs it. `target_*`: a FASTA (its one record; record 1 of a multi-record file with `first_record`,
    refused by name without it) or a sequence, else the public target; `msa`: the target chain's staged alignment (use_msa semantics; a chain
    asking for an MSA without one is refused by the module by name); `epitope`: the `--epitope` positions (1-based along the target, every copy;
    "epitope" = the module's epitope record, None when unset). "shape" / "flags": the inputs.Shape of that target (inputs.shape_of) as its record
    and as the driver's flags — the one composer the command line uses too."""
    from .inputs import shape_of                                           # inputs imports this module lazily (shape_from_args), so no cycle
    M = module(kit_home)
    T = M.target(fasta=target_fasta, sequence=target_seq, name=target_name, copies=target_copies, msa=msa, first_record=first_record)
    E = M.epitope(epitope, T) if epitope is not None else None
    S = shape_of(T, binder_length, target_copies, E)
    st = M.stages(steps1, steps2)
    return {"module": M, "kit_home": os.path.abspath(kit_home or kit_home_default()), "recipe_path": os.path.join(os.path.abspath(kit_home or kit_home_default()), KIT_RECIPE_RELPATH),
            "target": T, "binder_length": int(binder_length), "target_copies": int(target_copies), "tokens": M.tokens(T, binder_length), "seed": int(seed),
            "loss_terms": [list(t) for t in M.LOSS_TERMS], "mpnn": {"weights": M.MPNN_WEIGHTS, "temp": M.MPNN_TEMP}, "boltz2_loss": dict(M.BOLTZ2_LOSS),
            "x0": {"form": "softmax(scale * gumbel(key(seed), (L, 20)))", "scale": M.X0_GUMBEL_SCALE}, "stages": st, "phases": list(M.PHASES), "kind_of_phase": dict(M.KIND_OF_PHASE),
            "stage_call": M.STAGE_CALL, "step_kind": M.STEP_KIND, "refold": dict(M.REFOLD), "refold_key": M.REFOLD_KEY, "epitope": E, "shape": S.record(),
            "flags": S.flags()}
