"""Fold settings — the upstream API's own knobs, passed through by name (`pred` flags; ``resolve()`` is the one resolver).

Every knob is a keyword of the pinned library, spelled as the library spells it, defaulting to the library's own default:
  ``ESMFold2InputBuilder.fold`` (stock/src/esm/models/esmfold2/processor.py:327): ``--num_loops`` (20), ``--num_sampling_steps`` (200),
  ``--num_diffusion_samples`` (1), ``--msa_max_depth`` (1024, the fold-level subsample, resampled every loop), ``--lm_dropout`` (0.3),
  ``--msa_column_mask_rate`` (0.1), and its optional sampler / LM-mask overrides ``--noise_scale --step_scale --max_inference_sigma
  --lm_mask_pct`` (None: the checkpoint's own) and ``--early_exit`` (False) — forwarded to ``fold()`` only when given (SAMPLER_OVERRIDES),
  on every mode; ``fold()``'s ``seed=None`` is unseeded, so ``--seeds`` (or an input's own ``seeds`` key) is required — the package never invents a seed.
  ``MSA.from_a3m`` (esm.utils.msa): ``--remove_insertions`` (False: insertions kept), ``--max_sequences`` (None: the whole file).
  The model's two levers (``set_kernel_backend``, ``set_chunk_size``) stay at what ``from_pretrained`` leaves (the reference kernel path,
  pair-block chunk 64) unless ``pred --mode off --backend fused|shipped`` says otherwise (stock_fold.BACKENDS); the kit modes run on the
  kit server's fused base (their composition: modes.py / stack.py).
The defaults are read from the live signatures when the upstream package is importable, else from ``stock/PINS.json`` ``library_defaults``
(tests/test_registry_settings_det.py holds the two equal field by field). ``num_sampling_steps`` is the length of the noise schedule BEFORE
upstream truncates it at ``max_inference_sigma=256``: the number of denoising steps actually executed is smaller. The package records the
executed count beside ``num_sampling_steps`` in every manifest from the table in ``stock/PINS.json`` (``library_defaults.executed_steps``:
100 -> 68, 68 -> 46, 200 -> 134, 14 -> 10, the checkpoint's schedule); other values are reported as None.
``--det`` (det.py) sets ``lm_dropout=0`` and ``msa_column_mask_rate=0`` on top of the flags; ``fold_kwargs(det=True)`` applies it.
``--seeds`` wins over an input's own ``seeds`` key (cli._seeds_gate); the diffusion-sample count is ONE per run (inputs.run_samples):
``--num_diffusion_samples`` when given — an input naming another count is refused before anything loads — else the inputs' common
``num_diffusion_samples`` key, else the library's default; the `settings` line prints the count that runs.
Nothing here is a lever.
"""
from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from . import det as _det, modes

FOLD_KWARGS = ("num_loops", "num_sampling_steps", "num_diffusion_samples", "msa_max_depth", "lm_dropout", "msa_column_mask_rate")
DRIVER_RELPATH = modes.DRIVER_RELPATH                                # the kit's own driver: its fold-call constants
SERVER_RELPATH = modes.SERVER_RELPATH                                # the kit server: the fused base of every kit mode
KIT_CONSTANTS = ("NUM_LOOPS", "NUM_STEPS", "NSAMP", "MSA_MAX")

SAMPLER_OVERRIDES = ("noise_scale", "step_scale", "max_inference_sigma", "lm_mask_pct", "early_exit")   # fold()'s optional overrides: forwarded only when given (None / False = the checkpoint's own)
MSA_READ_KWARGS = ("remove_insertions", "max_sequences")                # MSA.from_a3m's
STOCK_FLAGS = FOLD_KWARGS + SAMPLER_OVERRIDES + MSA_READ_KWARGS            # every pass-through flag, by the library's own name
_FLAG_TYPES = {"num_loops": int, "num_sampling_steps": int, "num_diffusion_samples": int, "msa_max_depth": int,
               "lm_dropout": float, "msa_column_mask_rate": float, "max_sequences": int,
               "noise_scale": float, "step_scale": float, "max_inference_sigma": float, "lm_mask_pct": float}


def _bool_word(text: str) -> bool:
    t = str(text).strip().lower()
    if t in ("1", "true", "yes"):
        return True
    if t in ("0", "false", "no"):
        return False
    raise ValueError(f"expected true|false, got {text!r}")


def add_stock_flags(parser) -> None:
    """The upstream API's knobs on a command's parser, by the library's names; default None = the library's own default (resolve())."""
    g = parser.add_argument_group("fold settings (the upstream API's keywords; default: the library's own defaults)")
    for k in FOLD_KWARGS:
        g.add_argument(f"--{k}", type=_FLAG_TYPES[k], default=None, help=f"ESMFold2InputBuilder.fold {k}= (library default when absent)")
    for k in SAMPLER_OVERRIDES[:-1]:
        g.add_argument(f"--{k}", type=float, default=None, help=f"ESMFold2InputBuilder.fold {k}= (optional override; the checkpoint's own when absent; --mode off only)")
    g.add_argument("--early_exit", type=_bool_word, default=None, metavar="true|false", help="ESMFold2InputBuilder.fold early_exit= (library default False; --mode off only)")
    g.add_argument("--remove_insertions", type=_bool_word, default=None, metavar="true|false", help="MSA.from_a3m remove_insertions= (library default False: insertions kept)")
    g.add_argument("--max_sequences", type=int, default=None, help="MSA.from_a3m max_sequences= (library default None: the whole A3M)")


def flags_given(args) -> Dict[str, object]:
    """The stock flags the caller actually passed ({name: value}), in STOCK_FLAGS order."""
    return {k: getattr(args, k) for k in STOCK_FLAGS if getattr(args, k, None) is not None}


def executed_steps_table(pins: Optional[dict] = None) -> Dict[int, int]:
    """Denoising steps EXECUTED per ``num_sampling_steps`` (the schedule length before upstream's truncation at ``max_inference_sigma``),
    read from ``stock/PINS.json`` ``library_defaults.executed_steps`` — the one place the table lives. Values for other N are reported
    as None."""
    p = pins if pins is not None else _pins_or_none()
    tab = ((p or {}).get("library_defaults") or {}).get("executed_steps") or {}
    return {int(k): int(v) for k, v in tab.items() if k != "note"}


def _pins_or_none() -> Optional[dict]:
    try:
        from . import stack
        return stack.pins() if os.path.isfile(stack.pins_path()) else None
    except Exception:  # noqa: BLE001
        return None


class _ExecutedSteps:
    """``EXECUTED_STEPS.get(N)`` -> the executed count for N from PINS.json (read on use; None when unknown)."""
    def get(self, n, default=None):
        return executed_steps_table().get(int(n), default)

    def table(self, pins: Optional[dict] = None) -> Dict[int, int]:
        return executed_steps_table(pins)


EXECUTED_STEPS = _ExecutedSteps()


@dataclass
class Settings:
    num_loops: int
    num_sampling_steps: int
    num_diffusion_samples: int
    msa_max_depth: Optional[int]
    lm_dropout: Optional[float]
    msa_column_mask_rate: Optional[float]
    kernel_backend: Optional[str]
    chunk_size: Optional[int]
    msa_remove_insertions: bool
    msa_read_depth: Optional[int]                 # MSA.from_a3m max_sequences: None = the whole file
    seeds: Optional[List[int]] = None             # None = unseeded (the library default): the caller must pass --seeds
    source: str = ""
    executed_steps: Optional[int] = None          # denoising steps actually run for num_sampling_steps (EXECUTED_STEPS; None = not recorded)
    fold_overrides: Optional[Dict[str, object]] = None   # fold()'s SAMPLER_OVERRIDES the caller gave ({name: value}); None = none given: nothing forwarded

    def fold_kwargs(self, det: int = 0) -> dict:
        kw = {"num_loops": self.num_loops, "num_sampling_steps": self.num_sampling_steps,
              "num_diffusion_samples": self.num_diffusion_samples, "msa_max_depth": self.msa_max_depth}
        if self.lm_dropout is not None:
            kw["lm_dropout"] = self.lm_dropout
        if self.msa_column_mask_rate is not None:
            kw["msa_column_mask_rate"] = self.msa_column_mask_rate
        for k in SAMPLER_OVERRIDES:
            if (self.fold_overrides or {}).get(k) is not None:
                kw[k] = self.fold_overrides[k]
        if det:
            kw.update(_det.fold_overrides(int(det)))
        return kw

    def msa_read_kwargs(self) -> dict:
        return {"remove_insertions": self.msa_remove_insertions, "max_sequences": self.msa_read_depth}

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------------------------------------------- library defaults
def library_fold_defaults(pins: Optional[dict] = None) -> Optional[dict]:
    """``ESMFold2InputBuilder.fold`` keyword defaults: from the live signature when esm is importable, else stock/PINS.json."""
    try:
        import inspect
        from esm.models.esmfold2 import ESMFold2InputBuilder  # type: ignore
        sig = inspect.signature(ESMFold2InputBuilder.fold)
        return {k: p.default for k, p in sig.parameters.items() if p.default is not inspect.Parameter.empty}
    except Exception:
        return dict(pins["library_defaults"]["fold"]) if pins else None


def library_msa_read_defaults(pins: Optional[dict] = None) -> Optional[dict]:
    """``MSA.from_a3m`` keyword defaults (``remove_insertions``, ``max_sequences``): live signature, else stock/PINS.json."""
    try:
        import inspect
        from esm.utils.msa import MSA  # type: ignore
        sig = inspect.signature(MSA.from_a3m)
        return {k: sig.parameters[k].default for k in ("remove_insertions", "max_sequences")}
    except Exception:
        return dict(pins["library_defaults"]["msa_read"]) if pins else None


def library_model_defaults(pins: Optional[dict] = None) -> Optional[dict]:
    """The model's own defaults for the two calls (``kernel_backend``, ``chunk_size``): stock/PINS.json ``library_defaults.model``
    (the record of the pinned fork; the model class sets them in ``__init__``, which needs weights to introspect)."""
    if not pins:
        return None
    m = pins["library_defaults"]["model"]
    return {"kernel_backend": m["kernel_backend"], "chunk_size": m["chunk_size"]}


def upstream(pins: Optional[dict] = None) -> Settings:
    fd, mr, md = library_fold_defaults(pins), library_msa_read_defaults(pins), library_model_defaults(pins)
    if fd is None or mr is None or md is None:
        raise ValueError("fold settings: the library defaults are needed (esm importable, or stock/PINS.json library_defaults)")
    live = library_fold_defaults(None) is not None
    src = ("ESMFold2InputBuilder.fold / MSA.from_a3m signatures + stock/PINS.json library_defaults.model" if live
           else "stock/PINS.json library_defaults")
    return Settings(fd["num_loops"], fd["num_sampling_steps"], fd["num_diffusion_samples"], fd["msa_max_depth"],
                    fd.get("lm_dropout"), fd.get("msa_column_mask_rate"), md["kernel_backend"], md["chunk_size"],
                    bool(mr["remove_insertions"]), mr["max_sequences"], seeds=None, source=src,
                    executed_steps=EXECUTED_STEPS.get(fd["num_sampling_steps"]))


# ------------------------------------------------------------------------------------------------------------------- kit driver
def kit_constants(driver_path: str) -> Dict[str, int]:
    """``NUM_LOOPS, NUM_STEPS, NSAMP, MSA_MAX = ...`` read from the kit driver's source (ast; no import, no torch)."""
    tree = ast.parse(open(driver_path, encoding="utf-8").read(), filename=driver_path)
    out: Dict[str, int] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Tuple) and isinstance(node.value, ast.Tuple):
                for n, v in zip(tgt.elts, node.value.elts):
                    if isinstance(n, ast.Name) and n.id in KIT_CONSTANTS:
                        out[n.id] = ast.literal_eval(v)
            elif isinstance(tgt, ast.Name) and tgt.id in KIT_CONSTANTS:
                out[tgt.id] = ast.literal_eval(node.value)
    missing = [k for k in KIT_CONSTANTS if k not in out]
    if missing:
        raise ValueError(f"kit driver {driver_path}: constants {missing} not found")
    return out


_FUSED_BASE = re.compile(r'if\s+base\s*==\s*"fused"\s*:\s*\n\s*model\.set_kernel_backend\((?P<backend>[^)]*)\);\s*model\.set_chunk_size\((?P<chunk>[^)]*)\)')


def kit_model_calls(server_path: str) -> dict:
    """The server's two model calls for its ``fused`` base (the base every kit mode in modes.KIT_MODES runs on), read from
    ``configure()``'s source: ``{"kernel_backend": ..., "chunk_size": ...}``."""
    src = open(server_path, encoding="utf-8").read()
    m = _FUSED_BASE.search(src)
    if not m:
        raise ValueError(f"kit server {server_path}: configure()'s fused base (set_kernel_backend / set_chunk_size) not found")
    return {"kernel_backend": ast.literal_eval(m.group("backend")), "chunk_size": ast.literal_eval(m.group("chunk"))}


# ------------------------------------------------------------------------------------------------------------------------- load
def resolve(args=None, pins: Optional[dict] = None, **flags) -> Settings:
    """The settings a command runs at: the library's own defaults (``upstream(pins)``) with every stock flag the caller passed laid
    over them by name (``args``: a parsed namespace carrying the STOCK_FLAGS attributes, and/or keyword ``flags``; None = not given).
    ``seeds`` stays None: ``--seeds`` / the inputs' own keys are the callers' (cli._seeds_gate)."""
    st = upstream(pins)
    given = dict(flags_given(args) if args is not None else {})
    given.update({k: v for k, v in flags.items() if v is not None})
    unknown = [k for k in given if k not in STOCK_FLAGS]
    if unknown:
        raise ValueError(f"unknown fold settings {unknown}; the flags are {STOCK_FLAGS}")
    for k, v in given.items():
        if k == "remove_insertions":
            st.msa_remove_insertions = bool(v)
        elif k == "max_sequences":
            st.msa_read_depth = int(v)
        elif k in SAMPLER_OVERRIDES:
            st.fold_overrides = dict(st.fold_overrides or {}); st.fold_overrides[k] = bool(v) if k == "early_exit" else float(v)
        else:
            setattr(st, k, _FLAG_TYPES[k](v))
    if given:
        st.source += " + flags: " + " ".join(f"--{k} {given[k]}" for k in STOCK_FLAGS if k in given)
    st.executed_steps = None if (st.fold_overrides or {}).get("max_inference_sigma") is not None else EXECUTED_STEPS.get(st.num_sampling_steps)   # another sigma cut = another executed count: not recorded
    return st


def from_json(s: str) -> Settings:
    """Inverse of ``Settings.as_dict()`` (the stock subprocess receives its settings this way)."""
    return Settings(**json.loads(s))
