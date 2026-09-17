"""Modes, variants and the one resolver every command calls.

Package modes (``KIT_MODES``, the one place to change what a mode means; ``MODES`` / ``DEFAULT_MODE`` are the one list and the one
default every command reads):
  exact   the kits' exact lever set for the variant (``registry.LEVERS``; every lever tier ``exact``): the pipeline kit (datapath) on
          every size; on the loaded model the fused eager forward (every size; its composed kernels per shape) in every call shape —
          batch 1 and batched alike, no graph capture, nothing that waits for a shape to recur.
  off     nothing applied: upstream runs untouched. "Stock", the reference every lever is exact against, is the fastest correct form
          reachable through upstream's own switches with nothing of this package imported, each NAMED by ``load_call``: ``device=cuda`` and, per call shape, the attention class of ``STOCK_USE_FLASH_ATTN`` — the shipped varlen
          flash-attention class (``use_flash_attn=True``) in both regimes: upstream's other class (``use_flash_attn=False``) costs less
          wall per call at batch 1 but its outputs equal the shipped class's only at short lengths, so it is not a correct stock.
The kit modes load through ``KIT_CALL`` (the shipped flash-attention class the kits' records name) in both regimes. The KERNELS line of
each run states what the built model engaged, the regime, and — were the table to name the other class for a regime — the rule that chose it.
The package default is ``exact``. There is no fast mode for this model (no tier-2 composition).

Accelerators (``ACCELERATORS``, the words of the KERNELS line in order; ``kernels.py`` reads them off the built model): every mode expects
the pinned stack's set — the version ``stock/PINS.json`` "stacks" pins for the stack whose ``role`` is the pinned stack — engaged in
the built model. A run whose built model engages less than its expectation names every differing accelerator (the KERNELS SHORT line)
and proceeds on what the model engages — an environment short of the pins is stated, never refused.

Variants (``VARIANTS``) are the three checkpoint sizes, one per process; the ``regime`` is the call shape a command serves, derived
from the batch size: ``b1`` = the SDK's one-protein call, ``batched`` = the caller's own padded batched forward (B > 1 sequences
per call).
"""
from __future__ import annotations

from typing import Optional

from . import registry

MODES = ("off", "exact")
DEFAULT_MODE = "exact"
STOCK_MODES = ("off",)                               # the stock-class mode: nothing applied, upstream untouched; no lever, nothing of the kits imported
VARIANTS = ("300m", "600m", "6b")
REGIMES = ("b1", "batched")
DEFAULT_REGIME = "b1"

# mode -> the lever names (registry.LEVERS) in apply order
KIT_MODES = {
    "off": (),
    "exact": ("pipe", "fused"),
}

# mode -> the route word of the KERNELS line (`route=`): the stock route, the kits' exact line
ROUTE_WORDS = {"off": "stock", "exact": "exact"}

# The stock route's attention class per regime — the ONE place it is chosen: regime -> `use_flash_attn` (esm/models/esmc/compatibility.py:158,
# upstream's own keyword; True -> attn_implementation='flash_attention_2', the varlen class EsmcFlashMultiHeadAttention, compatibility.py:169 /
# model.py:286-291; False -> 'sdpa', EsmcMultiHeadAttention, whose cascade runs flash-attention's dense kernel on an unmasked batch,
# layers.py:378-388). The shipped class in both regimes: the other class costs less wall per `logits` call at batch 1 (no per-block host
# read-back of the batch's longest length, layers.py:525; no un/pad, model.py:462/493-495) but under a deterministic recipe its logits equal
# the shipped class's byte for byte only at short lengths (and at 6B); at 300M / 600M beyond that its top tokens differ at a rate far above
# reduction-order noise — not the same model semantics, so not stock. A regime whose entry is False is
# named on the KERNELS line by STOCK_RULE_NAMES (kernels.py: the word `off-by-rule:<name>(…)`, expected and read alike).
STOCK_USE_FLASH_ATTN = {"b1": True, "batched": True}
STOCK_RULE_NAMES = {"b1": "stock-b1", "batched": "stock-batched"}
# the kit modes' call in both regimes: the shipped flash-attention class (the class the kits' records name; `device` default = CUDA when present)
KIT_CALL = {"device": "cuda", "use_flash_attn": True}


def load_call(mode: str, regime: Optional[str] = None) -> dict:
    """The keyword arguments of ``ESMC.from_pretrained`` for ``mode`` in ``regime`` (None = the default regime): ``off`` -> `device=cuda` + the
    stock table's `use_flash_attn` for the regime (``STOCK_USE_FLASH_ATTN``: the shipped class in both) — upstream's own defaults, named
    (`device` None = CUDA when present, model.py:246-248; `use_flash_attn` True); a kit mode -> ``KIT_CALL``. "cuda" is resolved to the torch
    device by the driver."""
    m, rg = check_mode(mode), check_regime(regime)
    if m == "off":
        return {"device": "cuda", "use_flash_attn": STOCK_USE_FLASH_ATTN[rg]}
    return dict(KIT_CALL)


def call_rule(mode: str, regime: Optional[str] = None) -> Optional[str]:
    """The name of the rule that turned flash-attention's varlen class OFF in ``load_call(mode, regime)`` (``STOCK_RULE_NAMES[regime]`` when
    the stock table names the other class for that regime), or None when the call requests the shipped class (every mode and regime as the
    table stands). The KERNELS line's flash_attn word carries it."""
    m, rg = check_mode(mode), check_regime(regime)
    return STOCK_RULE_NAMES[rg] if (m == "off" and STOCK_USE_FLASH_ATTN[rg] is False) else None

# The accelerators upstream engages on the SDK route, in KERNELS-line order: name -> the `stock/PINS.json` "stacks.<stack>" key that pins
# the version it engages at (None: pinned absent on every stack, "pins.xformers"), the implementation word of its engaged form, and the
# implementation upstream falls back to when it does not engage (esm/models/esmc/kernels.py:64-90 — the conditions of upstream's own
# import-time warnings). kernels.py derives the expected word of each from this table and the stack's pins; nothing else names them.
ACCELERATORS = {
    "flash_attn": {"pin": "flash_attn", "dist": "flash_attn", "engaged": "flash_attn", "fallback": "sdpa",
                   "site": "model.py:286-291 _use_flash_attn -> layers.py:489 EsmcFlashMultiHeadAttention, :540 flash_attn_varlen_qkvpacked_func; model.py:453-495 bert_padding unpad/pad"},
    "rotary": {"pin": "flash_attn", "dist": "flash_attn", "engaged": "flash_attn.ops.triton.rotary", "fallback": "torch",
               "site": "kernels.py:55-61 FLASH_ATTN_ROTARY_INSTALLED -> layers.py:196-228 EsmcTritonRotaryEmbedding (flash class), :186-189 (sdpa class)"},
    "te": {"pin": "transformer_engine", "dist": "transformer_engine", "engaged": "transformer_engine", "fallback": "torch",
           "site": "model.py:298 use_te -> layers.py:331-332 te.LayerNormLinear, :338-339 te.Linear, :279-288 te.LayerNormMLP"},
    "xformers": {"pin": None, "dist": "xformers", "engaged": "xformers", "fallback": None,
                 "site": "kernels.py:35-41 XFORMERS_INSTALLED -> layers.py:368-376 (the sdpa-class cascade only; unreachable on the flash class; pinned absent: stock/check_pins.py refuses it present)"},
}


def pinned_stack(p: Optional[dict] = None) -> str:
    """The name of the pinned stack in ``stock/PINS.json`` "stacks" (the one whose ``role`` says so)."""
    if p is None:
        from . import stack
        p = stack.pins()
    names = [n for n, s in (p.get("stacks") or {}).items() if str(s.get("role", "")).startswith("pinned stack")]
    if len(names) != 1:
        raise ResolveError(f"stock/PINS.json names {len(names)} pinned stacks ({names}); exactly one expected")
    return names[0]


# the variant -> the SDK model name (esm/utils/constants/models.py) and the HF repo (esm/models/esmc/config.py): ONE home, stock/PINS.json
# "variants" (read here once; the driver and the weights check read these views)
def _variant_table(field: str) -> dict:
    from . import stack
    return {v: d[field] for v, d in stack.pins()["variants"].items()}


class _Lazy(dict):
    """A read-only view of stock/PINS.json "variants" (loaded on first use; the tree is located at that moment, not at import)."""
    def __init__(self, field):
        super().__init__(); self._field = field; self._loaded = False
    def _load(self):
        if not self._loaded:
            super().update(_variant_table(self._field)); self._loaded = True
    def __getitem__(self, k):
        self._load(); return super().__getitem__(k)
    def __contains__(self, k):
        self._load(); return super().__contains__(k)
    def items(self):
        self._load(); return super().items()
    def keys(self):
        self._load(); return super().keys()


VARIANT_MODEL_NAME = _Lazy("model_name")
VARIANT_HF_REPO = _Lazy("hf_repo")


class ResolveError(ValueError):
    pass


def check_mode(mode: str) -> str:
    m = (mode or "").strip().lower()
    if m not in MODES:
        raise ResolveError(f"mode {mode!r} is not one of {'|'.join(MODES)}")
    return m


def check_variant(variant) -> str:
    """``300m`` | ``600m`` | ``6b``; the SDK's own model names (``esmc_300m`` | ``esmc_600m`` | ``esmc_6b``, stock/PINS.json variants[].model_name)
    name the same sizes."""
    v = (variant or "").strip().lower()
    if v not in VARIANTS:
        v = next((k for k in VARIANTS if VARIANT_MODEL_NAME[k] == v), v)
    if v not in VARIANTS:
        raise ResolveError(f"variant {variant!r} is not one of {'|'.join(VARIANTS)} (or the SDK model names {'|'.join(VARIANT_MODEL_NAME[k] for k in VARIANTS)}) (--variant)")
    return v


def check_regime(regime) -> str:
    r = (regime or DEFAULT_REGIME).strip().lower()
    if r not in REGIMES:
        raise ResolveError(f"regime {regime!r} is not one of {'|'.join(REGIMES)}")
    return r


def regime_of_batch(batch: int) -> str:
    """The call shape of a batch size: ``b1`` for one item per forward, ``batched`` for a padded batch of N > 1."""
    return "batched" if int(batch) > 1 else DEFAULT_REGIME


def resolve(mode: str, variant: str, regime: str = DEFAULT_REGIME) -> dict:
    """The one resolution: (mode, variant, regime) -> {"mode", "variant", "regime", "levers": (names,), "levers_out": {name: why}}.
    ``levers`` is what the activation applies, in order; ``levers_out`` names every lever of the mode that does not serve this
    (variant, regime) and why (the kit's own gate, quoted from the registry) — nothing is dropped silently."""
    m, v, r = check_mode(mode), check_variant(variant), check_regime(regime)
    names = KIT_MODES[m]
    on = registry.levers_for(v, r, names)
    out = {}
    for n in names:
        if n in on:
            continue
        lv = registry.LEVERS[n]
        if v not in lv["variants"]:
            out[n] = f"serves {'/'.join(lv['variants'])} only ({lv['route_condition']})"
        else:
            out[n] = f"serves the {lv['regime']} regime only (this call shape: {r})"
    return {"mode": m, "variant": v, "regime": r, "levers": tuple(on), "levers_out": out}


def describe(mode: str, variant: Optional[str] = None, regime: Optional[str] = None) -> str:
    """One sentence per (mode, variant, regime): the levers in and the levers out with their reasons."""
    m = check_mode(mode)
    if m == "off":
        return ("stock (no kit on the path: upstream's own ESMC.from_pretrained, device=cuda, use_flash_attn="
                + "|".join(f"{STOCK_USE_FLASH_ATTN[r]}@{r}" for r in REGIMES) + ")")
    if variant is None:
        return "the kits' exact lever set: " + ", ".join(f"{n} ({registry.LEVERS[n]['class']}, {registry.LEVERS[n]['dir']})" for n in KIT_MODES[m])
    r = resolve(m, variant, regime)
    ins = ", ".join(f"{n} ({registry.LEVERS[n]['class']}, {registry.LEVERS[n]['dir']})" for n in r["levers"])
    outs = "; ".join(f"{n}: {why}" for n, why in r["levers_out"].items())
    return f"exact variant={r['variant']} regime={r['regime']}: levers {ins}" + (f" | out: {outs}" if outs else "")
