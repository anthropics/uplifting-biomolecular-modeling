"""The ablation switch: ``MODEL_OPT_LEVERS_OFF=<LEVER>[,<LEVER>…]`` runs a mode WITHOUT the named levers of its composition — one A/B
attribution run, never one of the kit's modes.

Semantics, the same on every route the wrapper activates (``run.sh pred|check|warm``, ``python -m af3_jax_opt``):

* names are lever ids of ``registry.LEVERS`` (the ACTIVE line's ``levers=`` words), comma-separated, blanks ignored, matched without
  regard to case, duplicates folded; unset or blank = no ablation and every printed line is the mode's, byte for byte (no ``ablated=``
  token anywhere);
* every name must be a lever of the composition that would run — ``modes.resolve`` + ``modes.with_n_gpu``: the mode's row, ``big``'s per
  region, the superseded set gone and ROWPAIR present under ``--n_gpu P`` — and have a switch the wrapper composes: an unknown name, a
  lever outside the composition, a switchless lever (FIX1 lives in the kit script itself; ROWPAIR is the ``--n_gpu`` axis) and any name
  under ``--mode off`` are refused BY NAME (``NOT ACTIVE … reason=MODEL_OPT_LEVERS_OFF refused — …``, exit 3); nothing runs under a name
  it does not have;
* applied to the resolved composition (``modes.with_levers_off``) through the switches the kit already composes, nothing new in the model
  process: the lever leaves ``levers``; GLUT / ATTNCFG / TTR / DATTN / TRIATT_XLA / SAMPLER_BF16 / ATOM_ATTN / HOIST_LOGITS / COND_SHARE / ATOM_COND_HOIST lose their variable (``AF3P_GLU_T`` / ``AF3P_ATTN_CFG`` / ``AF3_JAX_TTR`` /
  ``AF3_JAX_DATTN`` / ``AF3_JAX_TRIATT_XLA`` / ``AF3_JAX_SAMPLER_BF16`` / ``AF3_JAX_ATOM_ATTN`` / ``AF3_JAX_HOIST_LOGITS`` / ``AF3_JAX_COND_SHARE`` / ``AF3_JAX_ATOM_COND_HOIST``); FPF_TRIMUL / FPF_TRIATT rewrite the FlashPairformer add-on's word (``AF3_FLASHPAIRFORMER=both|trimul|triatt|off`` by what
  remains) and FPF_HOIST its own (``AF3_DIFFUSION_HOIST=0``); L1's flags leave the command line; a memory lever joins the memory launcher's
  ``--off`` word (as ``--n_gpu P`` names the superseded set). The cache class stays the mode's — the rule ``--n_gpu P`` already follows
  (one class per lever line, whatever is switched off inside it): the JAX compilation cache is keyed by program and XLA's autotune results
  by fusion, so the ablated program compiles beside the mode's and the two share every kernel choice they have in common — the A/B differs
  by the lever alone, and the deterministic recipe (``off`` on the class ``warm`` built) holds for an ablated composition as for the mode;
* the ACTIVE line gains ``ablated=<names>`` after ``levers=`` (request order), each ablated lever's LEVER line reads ``state=off
  reason=ablated``, the COMMAND line shows the environment and flags that ran, and the levers kept are judged all-or-refuse exactly as
  the mode's are (``modes.lever_evidence`` over the reduced composition: a kept lever that did not engage is still ``levers_short``, exit 3);
* ROW words of the support library ride in the same variable: ``triattn_xla:<row>`` (``opt_core.kernels.triattn_xla``'s own switch words, e.g.
  ``triattn_xla:cuda_sm90a``) is not a kit lever — it switches ONE row of the TRIATT_XLA lever's provider off inside the model process (which
  inherits the variable; the library's next row in its ``CELLS.json`` order serves and the SERVED line's ``txla_rows=`` names it). Refused BY
  NAME: under ``--mode off``, the bare package word ``triattn_xla`` (every row: remove the lever with ``TRIATT_XLA``), a composition without
  TRIATT_XLA or with it ablated, a row the pinned core does not have. Named on the ACTIVE / DONE lines as ``rows_off=<words>`` after ``ablated=``.
"""
from __future__ import annotations

import os
from typing import List, Mapping, Optional, Sequence

ENV = "MODEL_OPT_LEVERS_OFF"
REASON = "ablated"                                    # the LEVER line's reason= for an ablated lever (report.lever_lines)
TOKEN = "ablated"                                     # `` ablated=<A,B>`` on the ACTIVE line (report.activation_line)
SWITCHLESS = {                                        # levers of a composition MODEL_OPT_LEVERS_OFF cannot remove, and why (the refusal's words)
    "FIX1": "FIX1 is the kit script itself (run_alphafold_fast.py: the featurisation cache and the one-shot autotune attempt) and has no switch — --mode off runs the stock script without it",
    "ROWPAIR": "ROWPAIR is the --n_gpu axis, not a switch — --n_gpu 1 runs without it",
}


ROW_WORD_LEVER = "TRIATT_XLA"                            # the kit lever served by the support library package whose ROW switch shares this variable:
ROW_WORD_PREFIX = "triattn_xla"                          #   opt_core.kernels.triattn_xla.LEVER — `triattn_xla:<row>` switches ONE of its rows off inside the lever (the next row in the
ROWS_OFF_TOKEN = "rows_off"                              #   library's CELLS.json order serves; the SERVED line's txla_rows= names what served); `` rows_off=<words>`` on the ACTIVE / DONE lines


def is_row_word(tok: str) -> bool:
    """A MODEL_OPT_LEVERS_OFF token addressed to the support library's triangle-attention rows (``triattn_xla`` / ``triattn_xla:<row>``), not a kit lever."""
    t = (tok or "").strip().lower()
    return t == ROW_WORD_PREFIX or t.startswith(ROW_WORD_PREFIX + ":")


def row_words(environ: Optional[Mapping[str, str]] = None) -> List[str]:
    """The library row words ``MODEL_OPT_LEVERS_OFF`` carries (``triattn_xla:<row>``), request order, duplicates folded, as spelled; [] when none.
    They are not removed from the variable: the model process inherits it (stack.model_process_env) and the library reads its own words there."""
    env = os.environ if environ is None else environ
    out: List[str] = []
    for tok in (env.get(ENV) or "").split(","):
        tok = tok.strip()                                # spelling kept: the library matches its words case-sensitively (validate_row_words refuses another spelling by name)
        if tok and is_row_word(tok) and tok not in out:
            out.append(tok)
    return out


def core_rows() -> List[str]:
    """The row names of the pinned core's ``opt_core.kernels.triattn_xla`` (its ROWS words; the package imports no framework at import), [] if absent."""
    try:
        from opt_core.kernels import triattn_xla as _tx
        return list(getattr(_tx, "ROWS", ()))
    except Exception:  # noqa: BLE001 — a core without the package: no row can be named (refused below by name)
        return []


def validate_row_words(mode: str, words: Sequence[str], composition: Sequence[str], kit_names: Sequence[str] = ()) -> List[str]:
    """The library row words checked BY NAME against the composition that would run: mode off carries no lever; the bare package word names every
    row (the lever's own name removes the lever); the lever must be in the composition and not itself ablated; the row must be one of the core's."""
    words = [w for w in words if w]
    if not words:
        return []
    shown = f"{ENV}={','.join(words)}"
    if mode == "off":
        raise AblationError(f"{shown}: mode off applies no lever, there is nothing to switch (unset {ENV} for the stock route)")
    spelled = [w for w in words if w != w.lower()]
    if spelled:
        raise AblationError(f"{shown}: {', '.join(spelled)}: the support library matches its row words case-sensitively — spell it {ROW_WORD_PREFIX}:<row> in lower case")
    bare = [w for w in words if w == ROW_WORD_PREFIX]
    if bare:
        raise AblationError(f"{shown}: {ROW_WORD_PREFIX} names every row of the support library's triangle attention — remove the lever with {ROW_WORD_LEVER} instead; one row is {ROW_WORD_PREFIX}:<row>")
    if ROW_WORD_LEVER not in list(composition):
        raise AblationError(f"{shown}: {ROW_WORD_LEVER} is not in the composition mode {mode} runs here ({'+'.join(composition) or 'none'}) — its row words have nothing to switch")
    if ROW_WORD_LEVER in list(kit_names):
        raise AblationError(f"{shown}: {ROW_WORD_LEVER} is ablated in this run — its row words have nothing to switch")
    rows = core_rows()
    unknown = [w for w in words if w.split(":", 1)[1] not in rows]
    if unknown:
        raise AblationError(f"{shown}: {', '.join(unknown)}: not a row of opt_core.kernels.triattn_xla (rows: {', '.join(rows) or 'none — the pinned core has no such package'})")
    return list(words)


class AblationError(ValueError):
    """MODEL_OPT_LEVERS_OFF names something the composition cannot ablate — refused by name (stack.activate: NOT ACTIVE, exit 3)."""


def _canon(name: str) -> str:
    """A requested name as the registry spells it (case-insensitive match on registry.LEVERS), else the name as given (refused later by name)."""
    from .registry import LEVERS
    by_upper = {k.upper(): k for k in LEVERS}
    return by_upper.get(name.strip().upper(), name.strip())


def requested(environ: Optional[Mapping[str, str]] = None) -> List[str]:
    """The lever ids ``MODEL_OPT_LEVERS_OFF`` asks to remove, in request order, duplicates folded, spelled as the registry spells them; [] when unset / blank."""
    env = os.environ if environ is None else environ
    out: List[str] = []
    for tok in (env.get(ENV) or "").split(","):
        tok = tok.strip()
        if not tok or is_row_word(tok):                  # the library's row words are not kit levers (row_words / validate_row_words)
            continue
        name = _canon(tok)
        if name not in out:
            out.append(name)
    return out


def validate(mode: str, names: Sequence[str], composition: Sequence[str], n_gpu: int = 1, environ: Optional[Mapping[str, str]] = None) -> List[str]:
    """``names`` checked against the composition that would run (the lever ids ``modes.resolve`` / ``with_n_gpu`` produce for ``mode``);
    returns them (request order) or raises AblationError with the refusal's words."""
    from .registry import LEVERS
    names = [n for n in names if n]
    validate_row_words(mode, row_words(environ), composition, names)   # the library's row words (triattn_xla:<row>): refused by name or passed through to the model process
    if not names:
        return []
    shown = f"{ENV}={','.join(names)}"
    if mode == "off":
        raise AblationError(f"{shown}: mode off applies no lever, there is nothing to ablate (unset {ENV} for the stock route)")
    unknown = [n for n in names if n not in LEVERS]
    if unknown:
        raise AblationError(f"{shown}: {', '.join(unknown)}: not a lever of this kit (registry levers: {', '.join(LEVERS)})")
    comp = list(composition)
    outside = [n for n in names if n not in comp]
    if outside:
        axis = f" --n_gpu {n_gpu}" if int(n_gpu or 1) > 1 else ""
        raise AblationError(f"{shown}: {', '.join(outside)}: not in the composition mode {mode}{axis} runs here ({'+'.join(comp) or 'none'})")
    fixed = [n for n in names if n in SWITCHLESS]
    if fixed:
        raise AblationError(f"{shown}: " + "; ".join(SWITCHLESS[n] for n in fixed))
    return list(names)


def token(names: Sequence[str], environ: Optional[Mapping[str, str]] = None) -> str:
    """`` ablated=<A,B>`` (kit levers removed) then `` rows_off=<triattn_xla:row,…>`` (library rows switched off) for the ACTIVE / DONE lines; ``""`` when neither."""
    rw = row_words(environ)
    return (f" {TOKEN}={','.join(names)}" if names else "") + (f" {ROWS_OFF_TOKEN}={','.join(rw)}" if rw else "")
