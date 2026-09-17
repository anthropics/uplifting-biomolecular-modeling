"""Fold settings — stock's own ``run_inference`` knobs, passed through verbatim.

The knobs are the keyword arguments of ``chai_lab.chai1.run_inference`` that govern a fold (``FOLD_KEYS``: ESM embeddings on/off, the MSA /
template servers, MSA recycling subsample, trunk recycles, diffusion timesteps, diffusion samples, trunk samples, low_memory). Stock's own
command line is typer over that very function (``stock/src/chai_lab/main.py:38`` ``app.command("fold")(run_inference)``), so stock's flags ARE
the keyword names with dashes — ``--num-trunk-recycles``, ``--use-esm-embeddings / --no-use-esm-embeddings``, … (``STOCK_FLAGS``) — and their
defaults are the signature's (``stock/src/chai_lab/chai1.py:487-502``: use_esm_embeddings=True, use_msa_server=False,
use_templates_server=False, recycle_msa_subsample=0, num_trunk_recycles=3, num_diffn_timesteps=200, num_diffn_samples=5, num_trunk_samples=1,
low_memory=True). This package exposes exactly those flags with exactly those defaults on ``pred`` and ``check`` (``add_fold_arguments``);
a flag not given is stock's default, read from the pinned source (``library_defaults``, AST — the upstream is never imported here; the pin
check proves the installed file is the pinned one), so an invocation without fold flags folds at stock's defaults on every mode.

``from_args`` resolves a parsed namespace into the settings record every route consumes: ``{"fold": {FOLD_KEYS: value}, "non_default":
{the knobs given a value other than stock's default}, "flags": [those knobs as stock's flags, FOLD_KEYS order]}``. The off route hands
``fold`` to ``run_inference`` (stock_fold.py, whose pass-through gate re-reads the INSTALLED signature and refuses unless the keywords differ
from it in exactly ``non_default``); a kit mode writes it into the driver's ``RUN_KW`` (``to_driver_run_kw``; driver.py) — the kit driver's
feature-context call, its loop over trunk samples (run_inference's own) and its fold calls read every keyword (and ``--device``) from RUN_KW. Seeds are input plumbing, not a fold setting: ``--seed <int>`` (stock's flag;
``--seeds 0,1`` is the list form), else the item's own list (``seeds_for``); outputs are written per seed (``<key>/seed_<s>/``), so a kit
mode without one is refused by name (``seed_refusal``) and ``--mode off`` without one is stock's own unseeded call — ``run_inference(seed=None)``
seeds nothing (chai1.py:570-572 ``if seed is not None: set_seed([seed])``) — written under ``<key>/seed_none/``. ``--device`` is stock's
(chai1.py:501, default None = cuda:0 at :510): passed through on every mode — ``off`` hands it to run_inference, the kit driver reads it from its
``RUN_KW`` where run_inference derives ``torch_device``.
"""
from __future__ import annotations

import ast
import inspect
import json
import os
import sys
from typing import Dict, List, Optional


FOLD_KEYS = ("use_esm_embeddings", "use_msa_server", "use_templates_server", "recycle_msa_subsample", "num_trunk_recycles",
             "num_diffn_timesteps", "num_diffn_samples", "num_trunk_samples", "low_memory", "msa_server_url", "constraint_path", "template_hits_path")
BOOL_KEYS = ("use_esm_embeddings", "use_msa_server", "use_templates_server", "low_memory")
STR_KEYS = ("msa_server_url",)                                              # a string valued flag (stock: str)
PATH_KEYS = ("constraint_path", "template_hits_path")                      # path valued flags (stock: Path | None); made absolute when given (the driver runs in the output directory)
STOCK_FLAGS: Dict[str, str] = {k: "--" + k.replace("_", "-") for k in FOLD_KEYS}   # stock's own flag per keyword (typer over run_inference: main.py:38); a bool has the `--no-` twin
DRIVER_RUN_KW_KEYS = ("use_esm_embeddings", "use_msa_server", "msa_server_url", "constraint_path", "use_templates_server", "template_hits_path",
                      "recycle_msa_subsample", "num_trunk_recycles", "num_diffn_timesteps", "num_diffn_samples", "num_trunk_samples", "device",
                      "low_memory")                                          # the keys of the kit's RUN_KW (chai_proto.py RUN_KW): what its feature-context call, trunk loop and fold calls read


def _pins() -> dict:
    from . import stack
    p = os.path.join(stack.tree_home(), "stock", "PINS.json")
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def library_defaults(chai1_path: Optional[str] = None) -> Dict[str, object]:
    """``run_inference``'s keyword defaults for FOLD_KEYS — stock's defaults, the defaults of every fold flag — parsed from the pinned source
    file (``stock/src/chai_lab/chai1.py``; ``chai1_path`` overrides), else stock/PINS.json ``library_defaults``. Never imports the upstream:
    the CLI and the launcher stay torch-free until the driver's own import order, and the pin check (stock/check_pins.py) is what proves the
    installed ``chai1.py`` has the pinned bytes — ``installed_signature_defaults`` reads the live signature for ``check --json``."""
    if chai1_path is None:
        from . import stack
        chai1_path = os.path.join(stack.tree_home(), "stock", "src", "chai_lab", "chai1.py")
    if os.path.isfile(chai1_path):
        tree = ast.parse(open(chai1_path, "r", encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "run_inference":
                args = node.args
                kw = list(args.kwonlyargs); defaults = list(args.kw_defaults)
                out = {}
                for a, d in zip(kw, defaults):
                    if a.arg in FOLD_KEYS and d is not None:
                        out[a.arg] = ast.literal_eval(d)
                missing = [k for k in FOLD_KEYS if k not in out]
                if missing:
                    raise ValueError(f"run_inference in {chai1_path} lacks defaults for {missing}")
                return out
    return dict(_pins()["library_defaults"])


def installed_signature_defaults() -> Optional[Dict[str, object]]:
    """The live ``run_inference`` signature's defaults for FOLD_KEYS when ``chai_lab.chai1`` is ALREADY imported in this process
    (``check --json`` cross-check); None otherwise — this function never imports the upstream."""
    mod = sys.modules.get("chai_lab.chai1")
    fn = getattr(mod, "run_inference", None)
    if fn is None:
        return None
    try:
        sig = inspect.signature(fn)
        return {k: sig.parameters[k].default for k in FOLD_KEYS}
    except (TypeError, ValueError, KeyError):
        return None


def _flag_bool(text: str) -> bool:
    t = str(text).strip().lower()
    if t in ("1", "true", "yes", "on"):
        return True
    if t in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"not a boolean: {text!r}")


def add_fold_arguments(ap) -> None:
    """Stock's fold flags on an argparse parser, stock's spelling and stock's defaults: ``--num-trunk-recycles N`` … for the integers,
    ``--use-esm-embeddings / --no-use-esm-embeddings`` … for the booleans (typer's twin form; ``--low-memory false`` is read too). Every
    destination defaults to None = not given = stock's default (``from_args``)."""
    d = library_defaults()
    g = ap.add_argument_group("fold settings (stock's run_inference keywords, stock's defaults; README Settings)")
    for k in FOLD_KEYS:
        flag = STOCK_FLAGS[k]
        if k in BOOL_KEYS:
            g.add_argument(flag, dest=k, nargs="?", const=True, default=None, type=_flag_bool, metavar="BOOL", help=f"{k} (stock default {d[k]})")
            g.add_argument("--no-" + flag[2:], dest=k, action="store_false", default=None, help=f"{k}=False")
        elif k in STR_KEYS:
            g.add_argument(flag, dest=k, type=str, default=None, metavar="URL", help=f"{k} (stock default {d[k]})")
        elif k in PATH_KEYS:
            g.add_argument(flag, dest=k, type=os.path.abspath, default=None, metavar="PATH", help=f"{k} (stock default {d[k]})")
        else:
            g.add_argument(flag, dest=k, type=int, default=None, metavar="N", help=f"{k} (stock default {d[k]})")
    g.add_argument("--device", dest="device", default=None, metavar="DEV", help="run_inference device (stock default None = cuda:0), every mode")


def fold_flags(values: Dict[str, object]) -> List[str]:
    """``values`` (keyword -> value) as stock's flags, FOLD_KEYS order: a boolean as ``--<k>`` / ``--no-<k>``, an integer as ``--<k> N``, a
    string / path as ``--<k> <value>``."""
    out: List[str] = []
    for k in FOLD_KEYS:
        if k not in values:
            continue
        v = values[k]
        if k in BOOL_KEYS:
            out.append(STOCK_FLAGS[k] if v else "--no-" + STOCK_FLAGS[k][2:])
        elif k in STR_KEYS or k in PATH_KEYS:
            out += [STOCK_FLAGS[k], str(v)]
        else:
            out += [STOCK_FLAGS[k], str(int(v))]
    return out


def from_values(given: Dict[str, object], device: Optional[str] = None) -> dict:
    """The settings record from the knobs GIVEN (keyword -> value; absent = stock's default): ``{"fold", "non_default", "flags", "device"}``
    (``device``: stock's ``--device`` as given, None = not given = stock's default, cuda:0)."""
    unknown = [k for k in given if k not in FOLD_KEYS]
    if unknown:
        raise KeyError(f"not run_inference fold keywords: {unknown} (known: {', '.join(FOLD_KEYS)})")
    d = library_defaults()
    fold = dict(d)
    for k, v in given.items():
        fold[k] = (bool(v) if k in BOOL_KEYS else (str(v) if k in STR_KEYS or k in PATH_KEYS else int(v)))
    non_default = {k: fold[k] for k in FOLD_KEYS if fold[k] != d[k]}
    return {"fold": fold, "non_default": non_default, "flags": fold_flags(non_default), "device": (str(device) if device is not None else None)}


def from_args(ns) -> dict:
    """The settings record from a namespace ``add_fold_arguments`` parsed (a knob left None was not given)."""
    return from_values({k: getattr(ns, k) for k in FOLD_KEYS if getattr(ns, k, None) is not None}, device=getattr(ns, "device", None))


def label(st: dict) -> str:
    """One whitespace-free token naming the settings: ``stock`` at stock's defaults, else the non-default knobs ``k=v,k=v`` (FOLD_KEYS order)."""
    nd = st.get("non_default") or {}
    return "stock" if not nd else ",".join(f"{k}={nd[k]}" for k in FOLD_KEYS if k in nd)


def seeds_for(cli_seeds: Optional[List[int]], item_seeds: Optional[List[int]]) -> Optional[List[int]]:
    """Seed precedence: --seed / --seeds, then the item's own list; None = no seed named (off: stock's unseeded call; a kit mode: seed_refusal)."""
    for s in (cli_seeds, item_seeds):
        if s:
            return [int(x) for x in s]
    return None


def seed_refusal(mode: str, seeds_by_item: Dict[str, Optional[List[int]]]) -> Optional[str]:
    """Why a kit mode cannot fold these items (None = it can; ``off`` needs no seed): the kit writes outputs per seed, ``<key>/seed_<s>/``, and
    its driver folds a seed list — an item with no seed named (--seed / --seeds / the items file) is refused by name, never given one."""
    if mode == "off":
        return None
    missing = [k for k, v in seeds_by_item.items() if not v]
    if not missing:
        return None
    return (f"kit modes need --seed <int> (or --seeds 0,1 / the items' own seeds): outputs are written per seed, <key>/seed_<s>/ — no seed named for "
            f"{', '.join(missing[:5])}{' …' if len(missing) > 5 else ''} (--mode off without a seed is stock's unseeded call)")


def to_driver_run_kw(st: dict) -> Dict[str, object]:
    """The RUN_KW-shaped dict for the launcher (driver.py): the fold keywords in the kit's own key set plus the device (``--device``, else
    cuda:0 — stock's default)."""
    f = st["fold"]
    out = {k: f[k] for k in DRIVER_RUN_KW_KEYS if k in f}
    out["device"] = st.get("device") or "cuda:0"
    return out


def parse_seeds(text: Optional[str]) -> Optional[List[int]]:
    if text is None or str(text).strip() == "":
        return None
    return [int(x) for x in str(text).replace(";", ",").split(",") if x.strip() != ""]
