"""The sampling settings of a fold are ``rf3 fold``'s own: hydra ``key=value`` overrides, passed through verbatim.

The upstream CLI (``stock/src/models/rf3/src/rf3/cli.py:44-65``) takes every argument as a hydra override; its sampling keys and
defaults are ``models/rf3/configs/inference_engine/rf3.yaml:12-21``: ``n_recycles=10 diffusion_batch_size=5 num_steps=50
early_stopping_plddt_threshold=0.5 seed=null`` (``stock/PINS.json`` "settings"."cli_defaults" is the pin's copy, the values
``describe()`` reports when a key is not overridden). ``pred`` states NOTHING of its own: with no override ``rf3 fold`` runs at those
defaults plus the three inputs of a run — ``inputs=``, ``out_dir=``, ``ckpt_path=`` (from ``--input`` / ``--out_dir`` / ``--ckpt``)
— and ``seed=<s>`` per run (from ``--seeds``); every ``key=value`` token given after the options is appended verbatim, in order, for
every mode alike (the kit arms hand the same argv to the patched interpreter). The four keys the kit itself writes are refused by
name as overrides. MSA featurization (``n_msa``, ``max_msa_sequences``, dense pairing) is the checkpoint's own validation transform
and is not a CLI key.
"""
from typing import Dict, List, Optional, Sequence

KEYS = ("n_recycles", "diffusion_batch_size", "num_steps", "early_stopping_plddt_threshold")   # the sampling keys the effective table reports
RESERVED = {"inputs": "--input", "out_dir": "--out_dir", "ckpt_path": "--ckpt"}   # written by the kit per run; refused as overrides (seed= is rf3 fold's own and passes; --seeds with seed= is refused in overrides())


def check(tokens: Optional[Sequence[str]]) -> List[str]:
    """The caller's ``key=value`` tokens, validated in form only (hydra judges the keys): each has a non-empty key before ``=``;
    the kit's own three keys are refused by name."""
    if isinstance(tokens, (str, bytes)):
        raise ValueError(f"overrides must be a sequence of key=value tokens, not one string: {tokens!r}")
    out: List[str] = []
    for t in tokens or ():
        t = str(t)
        key = t.split("=", 1)[0].lstrip("+~")
        if "=" not in t or not key:
            raise ValueError(f"not a key=value override for rf3 fold: {t!r} (the input file is --input; sampling keys e.g. {' '.join(f'{k}=…' for k in KEYS)})")
        if key in RESERVED:
            raise ValueError(f"override {t!r} refused: {key}= is written by the kit from {RESERVED[key]}")
        out.append(t)
    return out


def hydra_value(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def parse_value(s: str):
    """A hydra scalar as Python: null | true | false | int | float | the string itself."""
    t = s.strip()
    if t.lower() in ("null", "~", "none"):
        return None
    if t.lower() in ("true", "false"):
        return t.lower() == "true"
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        return t


def overrides(tokens: Optional[Sequence[str]] = None, seed: Optional[int] = None) -> List[str]:
    """The ``key=value`` argv tail of one ``rf3 fold``: the caller's tokens verbatim, in order, then ``seed=<s>`` when given."""
    out = check(tokens)
    if seed is not None:
        refuse_double_seed(out)
        out.append(f"seed={int(seed)}")
    return out


def seed_tokens(tokens: Optional[Sequence[str]]) -> List[str]:
    """The caller's ``seed=`` tokens (rf3 fold's own override), in order."""
    return [str(t) for t in tokens or () if str(t).split("=", 1)[0].lstrip("+~") == "seed"]


def refuse_double_seed(tokens: Optional[Sequence[str]]):
    """``--seeds`` together with a ``seed=`` token names the seed twice: refused by name (either alone is fine)."""
    given = seed_tokens(tokens)
    if given:
        raise ValueError(f"override {given[0]!r} together with --seeds: give the seed once (seed=<s> is rf3 fold's own override; --seeds S[,S...] runs one fold per seed)")


def upstream_defaults() -> Dict[str, object]:
    """The upstream CLI's sampling defaults (``stock/PINS.json`` "settings"."cli_defaults", the pin's copy of ``rf3.yaml:12-21``), by KEYS."""
    from . import stack
    cli = stack.pins()["settings"]["cli_defaults"]
    return {k: cli[k] for k in KEYS}


def describe(tokens: Optional[Sequence[str]] = None) -> Dict[str, object]:
    """The effective sampling settings of a run: upstream's defaults overlaid with the caller's overrides of KEYS, plus the overrides verbatim."""
    toks = check(tokens)
    eff: Dict[str, object] = dict(upstream_defaults())
    for t in toks:
        k, v = t.split("=", 1)
        k = k.lstrip("+~")
        if k in eff:
            eff[k] = parse_value(v)
    return {**eff, "overrides": list(toks)}
