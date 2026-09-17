"""The ablation switch: ``MODEL_OPT_LEVERS_OFF=<lever>[,<lever>…]`` runs a mode WITHOUT the named levers of its set — one A/B attribution
run, never one of the kit's modes. The variable name and its grammar are the shared core's (``opt_core``);
the mechanism is this kit's own: the activation
(stack.activate) resolves the mode's lever tuple (modes.TABLE) and removes the named levers from it BEFORE anything is exported, placed or
installed — a removed lever's module is never imported, its class never rebound, its switch never exported.

Semantics (the same on every route the package activates — ``run.sh pred|check|warm``, ``python -m colabfold_opt``, the drop-in ``.pth``
environment route, the in-process ``enable()``; the model process `pred` launches inherits the variable):

* names are registry lever names (``registry.LEVERS``), comma-separated, whitespace ignored, duplicates folded, matched as spelled (upper
  case, the names the LEVER lines print); unset or blank = no ablation and the run is byte for byte the mode's (no ``ablated=`` token anywhere);
* removable: every lever of the mode's set as THIS process applies it (``modes.MODES[mode]``; under ``big`` at ``--n_gpu 1`` that is fast's
  set — ``ROWPAIR`` installs nothing there — and at ``--n_gpu P`` > 1 it is the set minus the one-device pair levers ``TRIMUL_PALLAS`` / ``TEMPL_DEDUP`` / ``TRANSITION``,
  whose sites the row-sharded pair stack owns: modes.ONE_DEVICE_PAIR_LEVERS), and the deployment levers placed in each of the kit's modes (``registry.DEPLOYMENT``: ``XLA_CACHE`` — no numerics, so removing one
  changes compile seconds only; this kit places them from its activation, so they are removable like any lever. ``XLA_CACHE`` ablated = the kit places no compile cache; a caller's own ``JAX_COMPILATION_CACHE_DIR`` is jax's to honour, as under stock);
* refused BY NAME (``[colabfold-opt] NOT ACTIVE: MODEL_OPT_LEVERS_OFF refused — …; exit 3``; nothing runs under a name it does not have):
  an unknown name; a lever outside the mode's set; ``ROWPAIR`` (at ``--n_gpu P`` > 1 it is the axis itself — ``--n_gpu 1`` is the unsharded
  arm —, at ``--n_gpu 1`` it is not applied); ``TRIMUL_PALLAS`` / ``TEMPL_DEDUP`` / ``TRANSITION`` at ``--n_gpu P`` > 1 (not applied there); a list that removes every lever of the
  mode (that is ``--mode off``, not an ablation); and any name under ``--mode off`` (stock applies no lever: nothing to ablate);
* the ACTIVE / DRY-RUN lines gain ``ablated=<names>`` (request order) right after ``levers=…`` 's fields, each ablated lever's LEVER line at
  exit reads ``state=off reason=ablated``, the kept levers engage all-or-refuse exactly as the mode does, and the exit census (partial
  activation, fallback share) expects only the kept levers — an ablated lever is never a `partial activation`;
* ``AF_PALLAS_ATTN`` ablated: its switch ``AF_PALLAS_ATTN=1`` leaves the mode's exports (``kit_env``) and is unset in the process if a caller
  had set it, the carried adapter is never imported and the pair-biased attention sites run stock attention; ``TRIATTN_XLA`` (the provider's
  pre-compiled triangle-attention row inside that binding) is ablated with it, by name (``ablated=AF_PALLAS_ATTN,TRIATTN_XLA``);
  ``TRIATTN_XLA`` ablated alone: the binding narrows the tier word to the provider's other rows (the provider's own word ``pallas:triattn_xla``
  in the same variable does the same inside the core; ``pallas:<row>`` switches any one provider row off, validated here against the provider's
  row names and named on the line); ``PALLAS_MSA`` kept meanwhile wraps the STOCK ``Attention`` class (its two call classes on
  the flash kernel, every other attention call stock). ``PALLAS_MSA`` ablated with ``AF_PALLAS_ATTN`` kept: MSA-column and extra-MSA row
  attention take the carried adapter's documented fallback path (its ``fallbacks`` counter, inside registry.FALLBACK_CLASSES' share). The two
  are independent levers over one class; either, both or neither may be removed.
"""
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import os

from . import modes as _modes
from .registry import DEPLOYMENT, LEVERS

ENV = "MODEL_OPT_LEVERS_OFF"
ROW_WORD_LEVERS = {"triattn_xla": _modes.TRIATTN_LEVER}   # the shared core's own words in the same variable: `triattn_xla:<row>` switches ONE row of the
ROW_WORD_ROWS = {"triattn_xla": ("triattn_native", "cuda_sm90a", "cuda_80", "k2b_aot")}  # triangle-attention bridge off inside the core (opt_core.kernels.triattn_xla levers_off) — accepted
                                                      # when TRIATTN_XLA is applied and kept, left in the environment for the core, named on the line
# ---- the shared core's JAX-family provider's own words (opt_core.kernels.pallas MODEL_OPT_LEVERS_OFF `pallas:<row>`): ONE provider row switched off inside the core;
#      accepted when a lever bound to that provider is applied and kept (the tier word then serves the next row of the provider's order or the stock statement), left in the variable
#      for the core, named on the line. The rows are the provider's own table (read when a request names the head; an absent core names none).
PROVIDER = "opt_core.kernels.pallas"
PROVIDER_HEAD = "pallas"
PROVIDER_LEVERS: Tuple[str, ...] = (_modes.TRIMUL_LEVER, _modes.LEVER, _modes.MSA_LEVER, _modes.COL_LEVER, _modes.TRANSITION_LEVER)   # the kit levers that bind the provider by tier word (trimul_pallas.py, the attention adapter + triattn_xla.py, msa_attn.py, msa_col_cudnn.py, transition.py)
ROW_WORD_LEVERS[PROVIDER_HEAD] = PROVIDER_LEVERS


def provider_rows() -> Tuple[str, ...]:
    """The provider's row names (opt_core.kernels.pallas ROWS), () when the shared core is not importable here."""
    try:
        import importlib
        return tuple(str(r) for r in importlib.import_module(PROVIDER).ROWS)
    except Exception:  # noqa: BLE001 — an absent / older core: no row word can be honoured, the request is refused by name
        return ()


def rows_of(head: str) -> Tuple[str, ...]:
    """The row names a head accepts: the table above, or the provider's own rows for its head."""
    if head in ROW_WORD_ROWS:
        return tuple(ROW_WORD_ROWS[head])
    return provider_rows() if head == PROVIDER_HEAD else ()


def owners_of(head: str) -> Tuple[str, ...]:
    """The kit lever(s) a head's row words address (one name or several)."""
    o = ROW_WORD_LEVERS.get(head)
    return () if o is None else ((o,) if isinstance(o, str) else tuple(o))
# ---- end: provider words


REASON = "ablated"                    # the LEVER line's reason for an ablated lever: `LEVER name=<lever> state=off reason=ablated`
TOKEN = "ablated"                     # the ACTIVE / DRY-RUN line token: `ablated=<a>,<b>`


class AblationError(ValueError):
    """MODEL_OPT_LEVERS_OFF names something this run cannot ablate — refused by name (NOT ACTIVE, exit 3)."""


def requested(environ: Optional[Mapping[str, str]] = None) -> List[str]:
    """The lever names in MODEL_OPT_LEVERS_OFF, in order, duplicates folded; [] when unset or blank."""
    env = os.environ if environ is None else environ
    out: List[str] = []
    for tok in (env.get(ENV) or "").split(","):
        tok = tok.strip()
        if tok and tok not in out:
            out.append(tok)
    return out


def validate(mode: str, names: Sequence[str], levers: Sequence[str], n_gpu: int = 1) -> List[str]:
    """Check ``names`` against ``mode`` as this process applies it and return them (a list, request order). ``levers``: the mode's lever
    tuple AFTER the ``--n_gpu`` axis settled it (stack.activate: ``ROWPAIR`` dropped at P = 1, the one-device pair levers modes.ONE_DEVICE_PAIR_LEVERS dropped at P > 1). Raises
    AblationError (every problem named at once) for an unknown name, a lever outside the mode, the axis lever, a lever the axis dropped,
    a request that empties the mode, or any name under mode ``off``."""
    names = list(names)
    if not names:
        return names
    if mode == "off":
        raise AblationError(f"{ENV}={','.join(names)}: mode off applies no lever, there is nothing to ablate (unset {ENV} for the stock route)")
    table = tuple(_modes.TABLE[mode][0])                                # the mode's set as the table names it (big's includes ROWPAIR)
    applied = tuple(levers)                                             # … as this process applies it at this n_gpu
    removable = tuple(l for l in applied if l != _modes.TP_LEVER) + tuple(DEPLOYMENT)
    problems: List[str] = []
    rows = [n for n in names if row_word(n) is not None]                # the core's row words (triattn_xla:<row>): validated here, applied by the core
    for n in rows:
        head, _, row = n.partition(":")
        owners = owners_of(head); owner = "|".join(owners)             # (provider words: one head, several kit levers — ablation.PROVIDER_LEVERS)
        if not row:
            problems.append(f"{n}: names every row of {owner} — ablate the kit lever ({ENV}={owner})")
        elif row not in rows_of(head):
            problems.append(f"{n}: not a row of {owner} (rows: {', '.join(rows_of(head)) or 'none — the shared core is not importable'})")
        elif not any(o in applied and o not in names for o in owners):
            problems.append(f"{n}: {owner} is not applied by this run (mode {mode}" + (", ablated" if any(o in names for o in owners) else "") + "), nothing to switch a row of")
    names_all, names = names, [n for n in names if n not in rows]
    removable = removable + tuple(rows)
    unknown = [n for n in names if n not in LEVERS]
    if unknown:
        problems.append(f"{', '.join(unknown)}: not a lever of this kit (registry levers: {', '.join(LEVERS)})")
    axis = [n for n in names if n == _modes.TP_LEVER and n in table]
    if axis:
        problems.append(f"{_modes.TP_LEVER}: the --n_gpu axis of mode {mode}, not removable by {ENV} "
                        + (f"(at --n_gpu {int(n_gpu)} it IS the arm; --n_gpu 1 runs the unsharded lever set)" if int(n_gpu) > 1 else
                           "(at --n_gpu 1 it installs nothing: the process's lever set is fast's)"))
    dropped = [n for n in names if n in table and n not in applied and n != _modes.TP_LEVER]
    if dropped:
        problems.append(f"{', '.join(dropped)}: not applied by mode {mode} at --n_gpu {int(n_gpu)} ({_modes.TP_LEVER} owns triangle multiplication "
                        f"at --n_gpu P > 1), nothing to ablate")
    outside = [n for n in names if n in LEVERS and n not in table and n not in DEPLOYMENT]
    if outside:
        problems.append(f"{', '.join(outside)}: not in mode {mode}'s lever set ({','.join(table)}; deployment levers {','.join(DEPLOYMENT)})")
    if _modes.LEVER in names and _modes.TRIATTN_LEVER in applied and _modes.TRIATTN_LEVER not in names:   # the bridge row rides AF_PALLAS_ATTN's binding: without the binding no site reaches it —
        names.append(_modes.TRIATTN_LEVER)                                                                  # ablating the binding ablates the row with it, BY NAME (ablated=AF_PALLAS_ATTN,TRIATTN_XLA on the line)
        names_all = list(names_all) + [_modes.TRIATTN_LEVER]
    kept = [l for l in applied if l not in names]
    if not problems and not kept:
        problems.append(f"the request removes every lever of mode {mode} ({','.join(applied)}): that is --mode off, not an ablation")
    if problems:
        raise AblationError(f"{ENV} refused — " + "; ".join(problems))
    names = names_all
    assert all(n in removable for n in names), (names, removable)       # every accepted name is one this process would have applied or placed (or a row word the core applies)
    return names


def _provider_rows() -> Tuple[str, ...]:
    """The provider's row names for `pallas:<row>` (opt_core.kernels.pallas ROW_NAMES: a standard-library import, no jax); () when absent."""
    try:
        import importlib
        return tuple(getattr(importlib.import_module("opt_core.kernels.pallas"), "ROW_NAMES", ()))
    except Exception:  # noqa: BLE001
        return ()


def row_word(name: str) -> Optional[str]:
    """The kit lever(s) a core row word addresses (`triattn_xla:<row>` -> TRIATTN_XLA; `pallas:<row>` -> the PROVIDER_LEVERS joined by `|`), None for anything
    else (kit lever names are upper case)."""
    head = str(name).partition(":")[0]
    o = ROW_WORD_LEVERS.get(head)
    return "|".join(o) if isinstance(o, tuple) else o


def apply(res: Dict, names: Sequence[str]) -> Dict:
    """The mode's resolution (modes.resolve()'s dict, axis-settled) minus the ablated levers: ``levers`` without them, ``kit_env`` without an
    ablated lever's switch (``AF_PALLAS_ATTN=1`` leaves the exports when ``AF_PALLAS_ATTN`` is ablated). Deployment names are not in
    ``levers``; the activation reads them from the report's ``levers_ablated`` when it places the deployment levers."""
    names = set(names)
    kit_env = {k: v for k, v in dict(res.get("kit_env") or {}).items() if not (_modes.LEVER in names and k == _modes.KIT_SWITCH)}
    return dict(res, levers=tuple(l for l in res["levers"] if l not in names), kit_env=kit_env)


def switches(names: Sequence[str]) -> Tuple[str, ...]:
    """The environment switches the ablated levers read — unset in the process at activation so a caller's export cannot re-engage one
    (only the carried adapter has a switch: ``AF_PALLAS_ATTN``)."""
    return (_modes.KIT_SWITCH,) if _modes.LEVER in set(names) else ()


def token(names: Sequence[str]) -> str:
    """`` ablated=<a>,<b>`` for the ACTIVE / DRY-RUN line; the empty string when nothing is ablated (the line reads exactly as before)."""
    names = [n for n in names or [] if n]
    return f" {TOKEN}={','.join(names)}" if names else ""
