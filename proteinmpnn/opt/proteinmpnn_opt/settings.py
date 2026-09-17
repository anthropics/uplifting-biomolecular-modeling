"""The stock command line's own options, passed through verbatim, and their map onto the kit executables.

One design pass takes ``--mode`` and, beyond the package's inputs and outputs, only options of the stock command line itself, spelled and
defaulted as upstream defines them: base variants protein_mpnn_run.py (stock/src/protein_mpnn_run.py:421-466: ``--seed`` 0 = a random seed per
process, ``--num_seq_per_target`` 1, ``--batch_size`` 1, ``--sampling_temp`` "0.1", ``--backbone_noise`` 0.0, ``--save_score`` 0, ``--save_probs`` 0,
``--omit_AAs`` X, ``--model_name`` v_48_020, ...). ``stock_table`` reads each option and its default from
the carried source (the one table; nothing is restated here), ``parse`` accepts a pass's tokens against it — an option upstream does not define,
a kit lever (registry.LEVERS: modes are lists of levers, modes.py) or an option the package supplies itself (``MANAGED_FLAGS``: the inputs, the
outputs, the two chain-assignment files) is refused by name — and both routes receive the accepted tokens: the stock command line verbatim
(``stock_argv``: an option not given is upstream's default by construction), the kit executable through ``kit_argv``.

Kit route, base variants (addon/mpnn_worker2.py): the worker takes the stock options its argparse defines (``worker_options``, read from the
carried worker: the seed — 0 draws one per process as upstream draws it —, the sequence counts and temperatures, ``--omit_AAs``, the jsonl
dictionaries for omitted / biased residues and PSSMs with their ``--pssm_*`` settings, ``--save_score`` / ``--save_probs``, ``--model_name``) and is
handed upstream's default for each one it defines differently when the pass does not give it (``WORKER_KNOBS``), so a pass means the same values on
both routes. ``--backbone_noise 0.0`` / ``--use_soluble_model`` are the worker's built-in semantics or rendered by the kit (``WORKER_IMPLIED``) and
are dropped from its argv, as is ``--suppress_print`` (upstream's console verbosity: nothing the kit route computes or writes, ``WORKER_INERT``);
``--pdb_path_chains`` is applied by the kit when it prepares a single-PDB input's chain assignment (``KIT_HANDLED``, kit_run.py). What mode exact
cannot serve is refused by name before any process starts (``worker_refuses``: the option and the mechanism — ``REFUSED_BY_WORKER``: the CA-only
model, the scoring-only and probability-only passes, tied decoding; ``--backbone_noise`` other than 0.0; ``--num_seq_per_target`` below
``--batch_size``), exit 3: ``--mode off`` runs the stock command line with those options. Upstream's own weights selectors pass through on both
routes (``BASE_WEIGHTS_FLAGS``): the kit renders its variant's weights directory only when a pass leaves them unset.
"""
from __future__ import annotations

import os
import re
from typing import Dict, FrozenSet, List, Optional, Tuple

from . import registry, stack
from .modes import WORKER

# The design pass's own options (its input, its output folder, its chain / position assignment files): ``design`` parses them itself, once,
# under upstream's names (cli.cmd_design) — met here, among the pass-through options, they were given a second time: refused by name.
MANAGED_FLAGS: Dict[str, str] = {
    "--jsonl_path": "the design pass's input, given once (--jsonl_path | --pdb_path | --input)",
    "--pdb_path": "the design pass's input, given once (--jsonl_path | --pdb_path | --input)",
    "--out_folder": "the design pass's output folder, given once (--out_folder | --out)",
    "--chain_id_jsonl": "the design option --chain_id_jsonl, given once", "--fixed_positions_jsonl": "the design option --fixed_positions_jsonl, given once",
}
# Upstream's own weights selectors, passed through verbatim; when a pass gives one the kit renders none of its own (``weights_given``):
# ``--path_to_model_weights`` / ``--use_soluble_model`` (protein_mpnn_run.py:35-49; else the variant's ``$MPNN_DIR/<variant>_model_weights``).
BASE_WEIGHTS_FLAGS = ("--path_to_model_weights", "--use_soluble_model")
# Options of the stock CLI that select another pass or model when SET (upstream's `if args.score_only:` …; ``--ca_only`` is a switch): mode exact refuses
# them by name at any value but upstream's default (``worker_refuses``; exit 3 before any process, ``--mode off`` runs them) — each changes the computation
# inside the region the kit worker re-states; the value is the mechanism, printed on the refusal line. At upstream's default they mean the plain design pass.
REFUSED_BY_WORKER: Dict[str, str] = {
    "--ca_only": "the CA-only model (its CA_ProteinFeatures featuriser and ca_model_weights) is outside the encoder the kit worker re-states",
    "--score_only": "a scoring-only pass (score .npz files, no design) is another program path than the design pass the kit worker re-states: no sampling loop for its levers",
    "--conditional_probs_only": "a conditional-probabilities pass (model.conditional_probs: its own masked forwards, no design) is another program path than the design pass the kit worker re-states",
    "--unconditional_probs_only": "an unconditional-probabilities pass (model.unconditional_probs: one order-free forward, no design) is another program path than the design pass the kit worker re-states",
    "--tied_positions_jsonl": "tied decoding (model.tied_sample: one decode step per tied group over the group's summed logits, one draw shared by its positions) is another decode loop than the per-position loop the kit worker's step kernels and graphs re-state",
}
BACKBONE_NOISE_REASON = ("backbone noise above 0 draws torch.randn_like noise inside the featuriser at each of protein_mpnn_run.py's three forwards per round: RNG draws inside "
                         "the encoder, which the kit worker computes once per backbone without them")
# Stock options that are the worker's built-in semantics (no flag of its own): dropped from its argv (--backbone_noise above 0 is refused by name, BACKBONE_NOISE_REASON;
# 0 or less adds no noise in upstream's featuriser, `if self.augment_eps > 0`).
WORKER_IMPLIED = {"--backbone_noise": "0.0", "--use_soluble_model": None}
# Stock options that change nothing the design pass computes or writes: accepted with any value, dropped from the worker's argv — upstream's console verbosity,
# and the two options only a pass mode exact refuses reads (``--path_to_fasta``: --score_only's; ``--conditional_probs_only_backbone``: --conditional_probs_only's).
WORKER_INERT = ("--suppress_print", "--path_to_fasta", "--conditional_probs_only_backbone")
# Stock options the kit applies itself when it prepares the worker's inputs (kit_run.run), not options of the worker's argv.
KIT_HANDLED = ("--pdb_path_chains",)                                            # the designed chains of a single-PDB input: its chain assignment, built as protein_mpnn_run.py builds it
# Stock options the worker defines with a default of its own: handed upstream's default explicitly when the pass does not give them.
WORKER_KNOBS = ("--num_seq_per_target", "--batch_size", "--sampling_temp", "--seed", "--omit_AAs", "--model_name")


class SettingsError(ValueError):
    """A token that is not a stock option, one the package supplies itself, or a lever (exit 2, as upstream's argparse refuses an unknown option)."""


_ADD_ARGUMENT = re.compile(r'add_argument\(\s*"(--\w+)"')
_DEFAULT = re.compile(r"""default\s*=\s*(?P<v>"[^"]*"|'[^']*'|[\w.+-]+)""")


def _argparse_table(path: str) -> Dict[str, Tuple[Optional[str], int]]:
    """``{--option: (default as the source writes it | None, line of the option name)}`` for every ``add_argument("--option", ...)`` call
    of ``path`` (one line or several: the call's text runs to the next ``add_argument(``)."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    table: Dict[str, Tuple[Optional[str], int]] = {}
    ms = list(_ADD_ARGUMENT.finditer(text))
    for k, m in enumerate(ms):
        body = text[m.end(): ms[k + 1].start() if k + 1 < len(ms) else len(text)][:600]
        d = _DEFAULT.search(body)
        v = d.group("v").strip("\"'") if d else ("False" if "store_true" in body else None)
        table.setdefault(m.group(1), (v, text.count("\n", 0, m.start(1)) + 1))
    return table


def stock_source(variant: str) -> str:
    """The carried stock command line: stock/src/protein_mpnn_run.py (every variant: the variants are weight sets of one program)."""
    return os.path.join(stack.stock_dir(), "src", "protein_mpnn_run.py")


def stock_table(variant: str) -> Dict[str, Tuple[Optional[str], int]]:
    """The stock command line's options with their defaults and definition lines, read from the carried source."""
    return _argparse_table(stock_source(variant))


def stock_options(variant: str) -> FrozenSet[str]:
    return frozenset(stock_table(variant))


def stock_default(variant: str, flag: str) -> Optional[str]:
    """Upstream's default for ``flag`` as its source writes it (``"0"`` for --seed, ``"0.1"`` for --sampling_temp, ...)."""
    return stock_table(variant)[flag][0]


def worker_options() -> FrozenSet[str]:
    """The options the kit worker's argparse takes, read from the carried addon/mpnn_worker2.py."""
    return frozenset(_argparse_table(os.path.join(stack.worker_dir(), WORKER)))


def parse(tokens: Optional[List[str]], variant: str) -> List[Tuple[str, Optional[str]]]:
    """The pass-through tokens of a pass as ``[(--option, value | None)]`` in the order given (``--option=value`` split; a flag without a
    value carries None). SettingsError, naming the token, for anything that is not an option of the stock command line, for a kit lever and
    for an option the package supplies itself."""
    toks = [str(t) for t in (tokens or [])]
    table = stock_table(variant)
    src = os.path.relpath(stock_source(variant), stack.tree_home())
    pairs: List[Tuple[str, Optional[str]]] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if not t.startswith("--"):
            raise SettingsError(f"{t!r} is not an option: the design pass takes --mode, its inputs/outputs and the stock command line's own "
                                f"options ({src}), `--option value`")
        name, eq, inline = t.partition("=")
        if name in registry.LEVERS:
            raise SettingsError(f"{name} is a kit lever ({registry.LEVERS[name]['name']}): modes are lists of levers (modes.py) and no lever is "
                                "accepted as an option; choose a mode")
        if name in MANAGED_FLAGS:
            raise SettingsError(f"{name} is {MANAGED_FLAGS[name]}")
        if name not in table:
            raise SettingsError(f"{name} is not an option of the stock command line ({src} argparse): nothing else reaches a design process")
        if eq:
            value: Optional[str] = inline
        elif i + 1 < len(toks) and not toks[i + 1].startswith("--"):
            value = toks[i + 1]; i += 1
        else:
            value = None
        pairs.append((name, value))
        i += 1
    return pairs


def given(pairs: List[Tuple[str, Optional[str]]]) -> Dict[str, Optional[str]]:
    """``{--option: value}`` of the tokens given (the last occurrence wins, as argparse has it)."""
    return {f: v for f, v in pairs}


def effective(pairs: List[Tuple[str, Optional[str]]], variant: str, flags) -> Dict[str, Optional[str]]:
    """``{--option: value}`` for ``flags``: the value given, else upstream's default from the carried source."""
    g = given(pairs)
    return {f: (g[f] if f in g else stock_default(variant, f)) for f in flags}


def worker_refuses(pairs: List[Tuple[str, Optional[str]]], variant: str) -> List[Tuple[str, str]]:
    """Kit route: what mode exact cannot serve in this pass, by name — ``[(<option>[=<value>], <the mechanism>)]``, empty when the worker serves the pass.
    ``design`` refuses such a pass before any process starts (cli.run_design: one NOT ACTIVE line naming each option with its mechanism, exit 3;
    ``--mode off`` runs the stock command line with those options). The reasons: an option of ``REFUSED_BY_WORKER`` at a value other than upstream's
    default (the default = the plain design pass: served); ``--backbone_noise`` above 0 (``BACKBONE_NOISE_REASON``); ``--num_seq_per_target`` below
    ``--batch_size`` (no batch: nothing would be designed); a stock option none of this module's tables places (refused rather than dropped). Everything
    else the stock command line takes the kit route serves: on the worker's argv (``kit_argv``), applied by the kit (``KIT_HANDLED``) or inert (``WORKER_INERT``)."""
    worker = worker_options()
    reasons: List[Tuple[str, str]] = []

    def refuse(token: str, why: str) -> None:
        if token not in [t for t, _ in reasons]:
            reasons.append((token, why))

    for f, v in pairs:
        token = f if v is None else f"{f}={str(v).replace(' ', ',')}"          # one token per option (no blanks)
        if f in REFUSED_BY_WORKER:
            if v is None or not _is_default(variant, f, v):                     # a switch given, or a value other than upstream's default: the other pass / model is asked for
                refuse(token, REFUSED_BY_WORKER[f])
        elif f not in worker and f not in WORKER_IMPLIED and f not in WORKER_INERT and f not in KIT_HANDLED and f not in BASE_WEIGHTS_FLAGS:
            refuse(token, "a stock option the kit route has no place for (neither the worker's, nor applied by the kit, nor inert)")
    e = effective(pairs, variant, ("--seed", "--num_seq_per_target", "--batch_size", "--sampling_temp", "--backbone_noise"))
    for f, conv in (("--seed", int), ("--num_seq_per_target", int), ("--batch_size", int), ("--backbone_noise", float)):
        try:
            conv(e[f])
        except (TypeError, ValueError):
            raise SettingsError(f"{f} {e[f]!r}: invalid {conv.__name__} value (protein_mpnn_run.py argparse: type={conv.__name__})") from None
    try:
        [float(item) for item in str(e["--sampling_temp"]).split()]
    except ValueError:
        raise SettingsError(f"--sampling_temp {e['--sampling_temp']!r}: not blank-separated floats (protein_mpnn_run.py: [float(item) for item in sampling_temp.split()])") from None
    if float(e["--backbone_noise"]) > 0.0:                                     # upstream's featuriser adds noise only `if self.augment_eps > 0`
        refuse(f"--backbone_noise={e['--backbone_noise']}", BACKBONE_NOISE_REASON)
    if int(e["--batch_size"]) < 1 or int(e["--num_seq_per_target"]) // int(e["--batch_size"]) < 1:
        refuse(f"--num_seq_per_target={e['--num_seq_per_target']}<--batch_size={e['--batch_size']}",
               "num_seq_per_target // batch_size = 0 batches: nothing would be designed (protein_mpnn_run.py writes an empty .fa then)")
    return reasons


def _is_default(variant: str, flag: str, value: str) -> bool:
    """True when ``value`` is upstream's default for ``flag`` as protein_mpnn_run.py would read it (its int options compared as ints, the rest as text)."""
    default = stock_default(variant, flag)
    try:
        return int(str(value)) == int(str(default))
    except (TypeError, ValueError):
        return str(value) == str(default if default is not None else "")


def designs_per_target(pairs: List[Tuple[str, Optional[str]]], variant: str) -> Dict[str, int]:
    """The sequences one pass designs per backbone and temperature, by protein_mpnn_run.py's rule — whole batches only: ``num_seq_per_target // batch_size``
    batches of ``batch_size`` — as ``{"requested": n, "batch_size": b, "produced": (n // b) * b}`` (both routes; cli prints the NOTE line when they differ)."""
    e = effective(pairs, variant, ("--num_seq_per_target", "--batch_size"))
    n, b = int(e["--num_seq_per_target"]), int(e["--batch_size"])
    return {"requested": n, "batch_size": b, "produced": (n // b) * b if b > 0 else 0}


def refusal_reason(mode: str, reasons: List[Tuple[str, str]]) -> str:
    """The NOT ACTIVE line's reason for a pass mode ``mode`` cannot serve: every refused option named with its mechanism, the exit code and the stock path."""
    named = "; ".join(f"{token} ({why})" for token, why in reasons)
    return f"cannot serve {named} under mode {mode} — refused by name, nothing launched (exit 3): --mode off runs the stock command line with these options"


def pdb_path_chains(pairs: List[Tuple[str, Optional[str]]]) -> Optional[List[str]]:
    """The designed chains of a single-PDB input as protein_mpnn_run.py reads ``--pdb_path_chains`` (blank-separated letters; not given or empty: every chain)."""
    v = given(pairs).get("--pdb_path_chains")
    return [str(item) for item in str(v).split()] if v else None


def weights_given(pairs: List[Tuple[str, Optional[str]]]) -> bool:
    """Base variants: the pass names its weights by upstream's own selectors (``--path_to_model_weights`` / ``--use_soluble_model``) — the kit renders
    no ``--path_to_model_weights`` of its own then."""
    g = given(pairs)
    return any(f in g for f in BASE_WEIGHTS_FLAGS)


def base_weights_dir(pairs: List[Tuple[str, Optional[str]]], variant: str, mpnn_dir: str) -> str:
    """The weights directory a base-variant pass loads: upstream's own rule when the pass gives a selector (protein_mpnn_run.py:35-49:
    a non-empty ``--path_to_model_weights`` as given; else ``<checkout>/soluble_model_weights`` for ``--use_soluble_model``, else
    ``<checkout>/vanilla_model_weights`` when ``--path_to_model_weights`` is given empty), else the variant's ``<checkout>/<variant>_model_weights``
    (what the kit renders as ``--path_to_model_weights`` for a pass that gives no selector)."""
    g = given(pairs)
    if g.get("--path_to_model_weights"):
        return str(g["--path_to_model_weights"])
    if "--use_soluble_model" in g:
        return os.path.join(mpnn_dir, "soluble_model_weights")
    if "--path_to_model_weights" in g:                                         # given, empty: upstream's own default directory
        return os.path.join(mpnn_dir, "vanilla_model_weights")
    return os.path.join(mpnn_dir, f"{variant}_model_weights")


def weights_path(pairs: List[Tuple[str, Optional[str]]], variant: str, mpnn_dir: Optional[str]) -> Optional[str]:
    """The one weights file the pass loads, by the same rule that renders its argv — ``<base_weights_dir>/<model_name>.pt`` (protein_mpnn_run.py:57);
    None when no rule names a file (MPNN_DIR unset with no ``--path_to_model_weights`` given: the pin check says so by name). The activation's pin check
    digests and names THIS file (stack.check_pins ``weights_file``)."""
    g = given(pairs)
    if not mpnn_dir and not g.get("--path_to_model_weights"):
        return None
    return os.path.join(base_weights_dir(pairs, variant, mpnn_dir or ""), f"{model_name(pairs, variant)}.pt")


def _render(pairs: List[Tuple[str, Optional[str]]]) -> List[str]:
    out: List[str] = []
    for f, v in pairs:
        out.append(f)
        if v is not None:
            out.append(str(v))
    return out


def stock_argv(pairs: List[Tuple[str, Optional[str]]]) -> List[str]:
    """The accepted tokens for the stock command line: verbatim, in the order given."""
    return _render(pairs)


def kit_argv(pairs: List[Tuple[str, Optional[str]]], variant: str) -> List[str]:
    """The accepted tokens for the kit executable (a pass ``worker_refuses`` found servable): the options the worker's argparse takes, verbatim — the
    built-in ones (``WORKER_IMPLIED``), the inert ones (``WORKER_INERT``), the ones the kit applies itself (``KIT_HANDLED``) and the pass selectors left at
    upstream's default (``REFUSED_BY_WORKER``) dropped — then upstream's default for every ``WORKER_KNOBS`` option the pass did not give."""
    g = given(pairs)
    worker = worker_options()
    out = _render([(f, v) for f, v in pairs if f in worker and f != "--path_to_model_weights"])   # the worker's own options only (implied / inert / kit-applied / at-default pass selectors dropped); the weights directory is rendered by kit_run (base_weights_dir)
    for f in WORKER_KNOBS:
        if f not in g:
            out += [f, str(stock_default(variant, f))]
    return out


def model_name(pairs: List[Tuple[str, Optional[str]]], variant: str) -> Optional[str]:
    """The weights file name a pass loads: ``--model_name`` as given, else upstream's default (v_48_020)."""
    return effective(pairs, variant, ("--model_name",))["--model_name"]
