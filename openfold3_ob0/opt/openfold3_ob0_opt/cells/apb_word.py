"""The pair-bias attention family's provider word (not a lever: the binding the family's levers share).

The DiT attention levers (`dit_attn`, `dit_glue`) and the trunk's attention with pair bias (`apb_trunk`) resolve their attention core through
the tree (`opt_core.of3_sampler.dit_rows.resolve_apb_core`). With a word bound here that core is the tree's ONE pair-bias attention provider
(`opt_core.kernels.apb`) asked by the word: a TIER word — `fast` | `big`, the line's, exported by modes.py as OPENFOLD3_OB0_OPT_APB_TIER — serves the
provider's row per call class (GPU capability, dtype, heads x head_dim, samples, token bucket, eager | CUDA-graph timing) from its
cell table; the caller's knob OPENFOLD3_OB0_OPT_APB_WORD names any provider word instead (a tier word, or a row word `row[:variant]` for an ablation by
name, e.g. `apb_attn`, `fpf_apb`, `sdpa:cudnn`, `dtk_loop`). No word bound (neither variable set) = the tree's direct entry
`opt_core.attn.apb_core`. The word is bound once per process, before the first forward (each lever's install calls `bind`); the
levers' LEVER lines carry the served rows in their `core_census={...}` field (`word`, `rows`, `cells`, `timing`, `fallback`, `refused`) and the
core prints its own `[opt_core] CELLS …` coverage line at exit."""
from __future__ import annotations

import os
import sys
from typing import Optional

from opt_core.of3_sampler import dit_rows as DR

PREFIX = "[openfold3_ob0-opt/apb_word]"
ENV_TIER = "OPENFOLD3_OB0_OPT_APB_TIER"                                  # the line's tier word (modes.py: fast on the fast line, big on the big lines)
ENV_WORD = "OPENFOLD3_OB0_OPT_APB_WORD"                                  # the caller's knob: any provider word (tier or row[:variant]); wins over the tier
TIER_VALUES = ("fast", "big")
STATE = {"word": None, "source": "none", "bound": False, "error": None}


def word(environ=None) -> Optional[str]:
    """The word this process asks: OPENFOLD3_OB0_OPT_APB_WORD when set, else OPENFOLD3_OB0_OPT_APB_TIER, else None (validated by `requested`)."""
    environ = os.environ if environ is None else environ
    w = (environ.get(ENV_WORD) or "").strip().lower()
    if w:
        return w
    t = (environ.get(ENV_TIER) or "").strip().lower()
    return t or None


def requested(environ=None) -> bool:
    """True when a word is set; raises ValueError BY NAME on a tier value outside fast|big or a word the provider does not know."""
    environ = os.environ if environ is None else environ
    t = (environ.get(ENV_TIER) or "").strip().lower()
    if t and t not in TIER_VALUES:
        raise ValueError(f"{ENV_TIER}={t!r} is not one of {'|'.join(TIER_VALUES)}")
    w = (environ.get(ENV_WORD) or "").strip().lower()
    if w:
        try:
            (DR.provider_word if hasattr(DR, "provider_word") else str)(w)
        except ValueError as e:
            raise ValueError(f"{ENV_WORD}={w!r}: {e}") from None
    return bool(t or w)


def bind(environ=None) -> Optional[str]:
    """Bind the process's provider word (idempotent); None when no word is set or the tree predates the word door (named on stderr once)."""
    environ = os.environ if environ is None else environ
    if STATE["bound"]:
        return STATE["word"]
    STATE["bound"] = True
    if not requested(environ):
        return None
    w = word(environ)
    src = "caller" if (environ.get(ENV_WORD) or "").strip() else "tier"
    if not hasattr(DR, "set_apb_word"):
        STATE.update(error="core_without_word_door")
        sys.stderr.write(f"{PREFIX} the tree's dit_rows has no set_apb_word (an older opt_core): the direct pair-bias entry serves; word {w!r} not bound\n")
        return None
    DR.set_apb_word(w, source=src)
    STATE.update(word=w, source=src)
    sys.stderr.write(f"{PREFIX} bound: the pair-bias attention family asks the core's provider by word={w} (source={src})\n")
    return w


def census_word() -> str:
    return f"word={STATE['word'] or 'none'} word_source={STATE['source']}"
