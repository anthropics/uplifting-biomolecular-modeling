"""The det recipe (``--det 0|1``) — the kit's deterministic segment-mean scatter plus the attention element, applied identically on
every mode that is given ``--det 1``, ``off`` included.

The recipe (``FUNCTION_TEXT``, applied by ``design --det 1``): ``transformers.models.esmfold2.modeling_esmfold2_common.
scatter_atom_to_token`` replaced by an exact segment mean (one-hot einsum + count), rebound in every function of ``modeling_esmfold2_common``
and ``modeling_esmfold2_experimental`` whose globals hold the original. Stock's own scatter uses ``scatter_reduce`` atomics and is not
run-to-run reproducible, so the bitwise claim of ``exact`` holds under this recipe on both arms, in ONE process and between
two processes that land in the same autotune class. The recipe does not pin that class: kernel configurations are chosen by timing at first
use (live autotune), so two det processes can differ beyond bitwise — stock vs stock included — while repeats inside one process are bitwise.
Det arms run on a per-box JIT cache (launch.arm_env at ``det >= 1``; configs/h100.env ``EF2INV_JIT_CACHE``): no other box's choices
are read or written; the choice itself stays the process's.

The Inductor pins (level 1, every arm, launch.DET_PINS): ``TORCHINDUCTOR_SHAPE_PADDING=0`` and ``TORCHINDUCTOR_DETERMINISTIC=1`` in the arm's
environment. Inductor decides whether to pad a matmul's operands by timing both forms in the compiling process (and a forced pad is bypassed once
dynamo marks the token dimension dynamic); two det processes compiling from empty caches can decide differently and then differ beyond bitwise
although each is deterministic. Padding off + Inductor's deterministic mode is one structural state every det process reaches; kernel arithmetic
elsewhere is untouched. Nothing is pinned at det 0.

The attention element (``det_attn``, level 1 only): flash-attn's ``flash_attn_func`` / ``flash_attn_varlen_func`` accumulate their backward
with atomics unless called with ``deterministic=True`` (their forward is deterministic). ``flash_det`` (the default recipe) rebinds the
flash-attn callables the two dispatching upstream modules hold (``modeling_esmfold2_common``: ``flash_attn_func``, ``flash_attn_varlen_func``;
``modeling_esmc``: ``flash_attn_func``, ``flash_attn_varlen_qkvpacked_func``) to ``functools.partial(fn, deterministic=True)`` and rebinds
``modeling_esmc._xformers_available`` False, so ESM-C's attention (the pseudo-perplexity forward + backward) leaves xformers — whose flash
backward xformers calls with ``deterministic=False``, hard-coded — for the module's own ``flash_attn_func`` branch (l.638-641), now
deterministic at any sequence length: the same flash-attn kernels as a run without ``--det``, with the deterministic backward (the proof line names it:
``esmc_attn=flash_attn det_attn=flash_det exempt=esmc_attn=flash_attn``; a run without ``--det`` keeps xformers). On a box without flash-attn the
rebinding is a no-op and the words say ``sdpa`` (attention.py). A det arm's
ESM-C forward therefore runs flash-attn through the module's direct call instead of through xformers' binding of the same flash-attn build:
`--det 1` designs are compared with other `--det 1` designs (``exact --det 1`` equals ``off --det 1`` bit for bit), never with a run made without it.

Nothing else: no ``torch.use_deterministic_algorithms``, no cuBLAS workspace variable (the cookbook's cloud app class has a ``deterministic``
parameter the local route never reaches, stock file l.1429/1433-1434). Level 0 = off.
"""
import functools
from typing import Dict, Optional

LEVELS = (0, 1)
DET_ATTN = "flash_det"                             # the attention element's word on the ATTN line (det_attn=flash_det); repeated stock runs and exact are bitwise under it
LEVEL_WHAT = {0: "off", 1: "the kit's deterministic segment-mean scatter_atom_to_token in both esmfold2 model modules (det.FUNCTION_TEXT) "
                            "+ the attention element det_attn (flash_det: flash-attn callables rebound with deterministic=True and ESM-C attention off xformers onto that deterministic flash_attn_func) "
                            "+ Inductor pinned in the arm's environment (launch.DET_PINS: TORCHINDUCTOR_SHAPE_PADDING=0, TORCHINDUCTOR_DETERMINISTIC=1), so that every det process compiles the same unpadded, deterministic kernels"}
FLASH_CALLABLES = {"transformers.models.esmfold2.modeling_esmfold2_common": ("flash_attn_func", "flash_attn_varlen_func"),
                   "transformers.models.esmc.modeling_esmc": ("flash_attn_func", "flash_attn_varlen_qkvpacked_func")}
FLASH_FLAGS = {"transformers.models.esmfold2.modeling_esmfold2_common": "FLASH_ATTN_AVAILABLE", "transformers.models.esmc.modeling_esmc": "_flash_attn_available"}
XFORMERS_FLAG = ("transformers.models.esmc.modeling_esmc", "_xformers_available")   # the element rebinds it False: xformers' own flash backward (fa2B, deterministic=False hard-coded in xformers) is bypassed under the recipe
# the recipe's function, one statement per line; make_det_scatter builds it against the running torch module
FUNCTION_TEXT = '''def det_scatter_atom_to_token(atom_features, atom_to_token_idx, n_tokens, atom_mask=None):
    oh = torch.nn.functional.one_hot(atom_to_token_idx.long(), n_tokens).to(atom_features.dtype)
    if atom_mask is not None: oh = oh * atom_mask.unsqueeze(-1).to(oh.dtype)
    summed = torch.einsum("...at,...ad->...td", oh, atom_features); cnt = oh.sum(-2).clamp(min=1).unsqueeze(-1)
    return summed / cnt
'''


def level(value) -> int:
    lv = int(value or 0)
    if lv not in LEVELS:
        raise ValueError(f"--det must be one of {LEVELS} (got {value!r}): 0 = off, 1 = the kit's deterministic scatter")
    return lv


def make_det_scatter(torch):
    """The recipe's function, built against the given torch module (so the text above and this body stay one thing)."""
    ns: Dict[str, object] = {"torch": torch}
    exec(FUNCTION_TEXT, ns)
    return ns["det_scatter_atom_to_token"]


def apply_attn(modules: Optional[Dict[str, object]] = None) -> dict:
    """The attention element on the two dispatching modules (``modules``: {name: module} injection for the CPU tests, else imported).
    Returns ``{"det_attn": DET_ATTN, "rebound": [module.name, ...], "flags": {module.FLAG: value}}``."""
    import importlib
    rebound, flags = [], {}
    for modname, names in FLASH_CALLABLES.items():
        mod = modules.get(modname) if modules is not None else importlib.import_module(modname)
        if mod is None:
            continue
        flag = FLASH_FLAGS[modname]
        if (modname, XFORMERS_FLAG[1]) == XFORMERS_FLAG and getattr(mod, XFORMERS_FLAG[1], False):
            setattr(mod, XFORMERS_FLAG[1], False); rebound.append(f"{modname}.{XFORMERS_FLAG[1]}=False")     # ESM-C attention leaves xformers for the module's flash_attn_func branch (deterministic=True below)
        if modname == XFORMERS_FLAG[0]:
            flags[f"{modname}.{XFORMERS_FLAG[1]}"] = bool(getattr(mod, XFORMERS_FLAG[1], False))
        for n in names:
            fn = getattr(mod, n, None)
            if fn is None or (isinstance(fn, functools.partial) and fn.keywords.get("deterministic") is True):
                continue
            setattr(mod, n, functools.partial(fn, deterministic=True)); rebound.append(f"{modname}.{n}")
        flags[f"{modname}.{flag}"] = bool(getattr(mod, flag, False))
    return {"det_attn": DET_ATTN, "rebound": rebound, "flags": flags}


def apply(lv: int) -> dict:
    """Level 1: rebind ``scatter_atom_to_token`` to the recipe's segment mean in both model modules, then the attention element. Returns
    what was rebound (``{}`` at level 0)."""
    if level(lv) < 1:
        return {}
    import torch
    from transformers.models.esmfold2 import modeling_esmfold2_common as C, modeling_esmfold2_experimental as E
    orig = C.scatter_atom_to_token
    det = make_det_scatter(torch)
    C.scatter_atom_to_token = det
    rebound = []
    for mod in (C, E):
        for name in dir(mod):
            o = getattr(mod, name)
            if getattr(o, "__module__", "") and hasattr(o, "__globals__") and o.__globals__.get("scatter_atom_to_token") is orig:
                o.__globals__["scatter_atom_to_token"] = det
                rebound.append(f"{mod.__name__}.{name}")
    attn = apply_attn()
    return {"scatter_atom_to_token": "segment-mean (det.FUNCTION_TEXT)", "rebound_in": rebound,
            "det_attn": attn["det_attn"], "det_attn_rebound": attn["rebound"], "det_attn_flags": attn["flags"]}

