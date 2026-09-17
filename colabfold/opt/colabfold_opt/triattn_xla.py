"""Pair-biased attention through the shared core's provider, by tier word — and the lever ``TRIATTN_XLA`` (the provider's pre-compiled
triangle-attention rows inside that binding).

The sites: AlphaFold's ``Attention`` calls that carry a non-batched (pair) bias — triangle attention starting / ending node (Evoformer,
extra-MSA stack, template pair stack) and MSA row attention with pair bias. ``AF_PALLAS_ATTN`` (the carried adapter
``opt/forward/af2_pallas_flash/af2_pallas_flash/af2_pallas_attn.py``) rebinds ``alphafold.model.modules.Attention`` and computes, for those
calls, the module's own q / k / v projections heads-major (``[b, h, s, c]``), ONE fused attention call (its ``FLASH_OP`` slot) and the stock
gating / output projection. This module supplies that slot: ``bind(kit, word)`` puts ``provider_op(word)`` into ``FLASH_OP[0]`` — one call of
``opt_core.kernels.pallas.serve.attention(q, k, v, bias, key_mask, scale, word=<tier word>, kind=<site>, n_seq=<rows>, layout="BHSD",
direction="fwd")`` per site, heads-major operands as the adapter emits them (the provider's fused rows — the pre-compiled triangle-attention
kernels ``triattn_xla``, the Pallas flash rows ``pallas_attn`` / ``cd_triatt`` / ``rowshared`` — take that layout without a transpose) — and
rebinds ``Attention`` once more to a thin subclass that names the SITE of each call to the op (``triangle_start`` … ``msa_row``,
``template_`` / ``extra_`` prefixed; the provider's family word: kind ``tri`` | ``tmpl`` | ``msarow`` + heads, channels per head, rows).
Which kernel serves a call is the provider's cell for (jax line, compute capability, dtype, family, size, tier word) — this
package carries no row pin, no size floor and no launch-setting table; a launch setting the provider's table carries for a row (``pallas_attn@s3``
on compute capability 9.0) rides the tier word. ``--mode fast`` binds the word ``fast``, ``--mode big`` the word ``big``; ``exact`` does
not bind these sites (stock attention: the provider's exact arm for these cells is the stock statement by name).

``TRIATTN_LEVER`` = the provider's row ``triattn_xla`` (``opt_core.kernels.triattn_xla``: the sealed CUDA kernels on compute capability
9.0 / 8.0, the ahead-of-time compiled K2B cubins; forward only — colabfold never differentiates) INSIDE the binding: ``enable()`` checks the
bridge loads on this machine (``require()``: importable, its XLA-FFI launcher for this jaxlib, the GPU backend — a refusal BY NAME, the activation
is then NOT ACTIVE) and lets the provider's order reach the row; with the lever out of the set (``MODEL_OPT_LEVERS_OFF=TRIATTN_XLA``) the op
narrows the tier word to the provider's other rows (``prefer=``), the provider's own word ``pallas:triattn_xla`` in the same variable does the
same inside the core. Its LEVER line: ``calls`` = sites the bridge row served, ``fallbacks`` + ``fallback_by`` = sites another row served
(``cell:<n>`` — the provider's cell named another row; ``refused(<kind>):<n>`` — the bridge refused at the call and the provider's next row
served), ``rows`` (``cuda_sm90a:n,…`` — the bridge's own served census), ``sites``, ``shapes`` (``NxSxHxD:n``), ``word``, ``cc``, ``core``
(the bridge's version). The binding's census rides AF_PALLAS_ATTN's LEVER line (``bind_state()``: ``word``, ``served=<row>:<n>``,
``sites=<site>:<row>:<n>``, ``cells=<cell key>:<n>``, ``provider``).

Numerics: every fused row is tolerance class (bf16 tensor-core products, fp32 softmax statistics and accumulation; rel-rms vs an fp32
reference 2-5e-3 in the provider's cells) — Tier 2, inside stock's seed band; deterministic run to run; rows are not bitwise with one another.
Under ``big`` at ``--n_gpu P`` > 1 the row-sharded pair stack runs the pair sites inside its shard_map regions through the Attention class this
lever is bound over: the bridge serves there with the one-GPU shapes (a 128-row chunk of this GPU's rows x S = N keys behind the gathered bias);
the lever stays on there (modes.ONE_DEVICE_PAIR_LEVERS does not list it).
"""
from __future__ import annotations

import importlib
import sys
import threading
from contextlib import contextmanager
from typing import Any, Dict, Optional, Sequence, Tuple

NAME = "TRIATTN_XLA"
STRATEGY = "F1.flash_triatt"
MODULES = "alphafold.model.modules"
CLASS = "Attention"
MARKER = "_pallas_word"                              # the site-naming subclass bind() installs over the carried adapter's class
SERVE = "opt_core.kernels.triattn_xla"              # the bridge package (the provider's row `triattn_xla`); require() / its served-row census
PROVIDER = "opt_core.kernels.pallas"                # the JAX-family provider: rows, tier words, cells
PROVIDER_SERVE = "opt_core.kernels.pallas.serve"    # its call face: attention(q, k, v, bias, key_mask, scale, *, word, kind, n_seq, layout, direction, prefer)
ROW = "triattn_xla"                                  # the provider's row this lever stands for
LAYOUT = "BHSD"                                      # [b, h, s, c]: what the carried adapter's projections emit — the provider's fused rows take it without a transpose
N_GPU_REASON = "n_gpu>1"                             # the LEVER line's reason word for a lever dropped at P > 1; not read for this lever, which modes.ONE_DEVICE_LEVERS does not list (it stays on)
CELL_RULE = "cell"                                   # registry.STEP_ASIDE_RULES: the provider's cell named another row for the call (by its table, by name)
BRIDGE_MIN: Tuple[int, ...] = (1, 1, 0)              # opt_core.kernels.triattn_xla >= 1.1.0: the face carries the vmap rule and the row switches
CORE_WORD = "triattn_xla"                            # the bridge's own MODEL_OPT_LEVERS_OFF word: `triattn_xla:<row>` switches one of ITS rows off inside the core
CORE_ROWS: Tuple[str, ...] = ("triattn_native", "cuda_sm90a", "cuda_80", "k2b_aot")   # the bridge's rows (restated: ablation.py validates `triattn_xla:<row>` without importing jax)
PROVIDER_WORD = "pallas"                             # the provider's own MODEL_OPT_LEVERS_OFF word: `pallas:<row>` switches one provider row off inside the core
SITES = {"triangle_attention_starting_node": "triangle_start", "triangle_attention_ending_node": "triangle_end",
         "msa_row_attention_with_pair_bias": "msa_row"}
KIND_BY_SITE = {"msa_row": "msarow"}                 # the provider's attention kind per site word: MSA row attention carries its row count (n_seq); template_* -> tmpl; else tri

_STATE: Dict[str, Any] = {"enabled": False, "calls": 0, "fallbacks": 0, "fallback_by": "none", "rows": "none", "sites": "none", "shapes": "none",
                          "word": None, "cc": None, "core": None}
_COUNTS: Dict[str, Dict] = {"fallback_by": {}, "sites": {}, "shapes": {}}
_BIND: Dict[str, Any] = {"bound": False, "word": None, "calls": 0, "served": "none", "sites": "none", "cells": "none", "provider": None}
_BCOUNTS: Dict[str, Dict] = {"served": {}, "sites": {}, "cells": {}}
_ORIG: Dict[str, Any] = {"cls": None, "modules": None, "op": None, "kit": None}
_CTX = threading.local()


class Refusal(RuntimeError):
    """The lever cannot be applied in this process (the bridge's launcher does not load on this jaxlib, no GPU backend): raised by
    ``enable()`` BY NAME; the activation is then NOT ACTIVE (a mode is all of its levers)."""

    def __init__(self, kind: str, detail: str = ""):
        self.kind = kind
        super().__init__(f"{NAME}: {kind}" + (f" — {detail}" if detail else ""))


def _render(d: Dict) -> str:
    return ",".join(f"{k}:{v}" for k, v in sorted(d.items(), key=lambda kv: str(kv[0]))) or "none"


def _bump(counts: Dict, state: Dict, key: str, val: str) -> None:
    d = counts[key]
    d[val] = d.get(val, 0) + 1
    state[key] = _render(d)


def compute_capability() -> Optional[str]:
    """The first jax GPU device's compute capability ("9.0", "8.0", …), None when jax or a GPU device is not there."""
    try:
        import jax
        for d in jax.devices():
            cc = getattr(d, "compute_capability", None)
            if cc:
                return str(cc)
    except Exception:  # noqa: BLE001 - a label; the caller names the absence
        return None
    return None


def site_of(module) -> str:
    """The call site's word from the haiku module path of the ``Attention`` instance (``…/triangle_attention_starting_node/attention``)."""
    try:
        path = str(getattr(module, "module_name", "") or "")
    except Exception:  # noqa: BLE001 - outside a transform the property may raise; the label degrades, nothing else
        path = ""
    word = "other"
    for key, w in SITES.items():
        if key in path:
            word = w
            break
    if "template" in path:
        return "template_" + word
    if "extra_msa" in path:
        return "extra_" + word
    return word


def kind_of(site: str) -> str:
    """The provider's attention kind for a site word: ``msarow`` (MSA row attention: the family carries the row count), ``tmpl`` (the
    template pair stack's heads), else ``tri``."""
    base = site.split("_", 1)[1] if site.startswith(("template_", "extra_")) else site
    if site.startswith("template_"):
        return "tmpl"
    return KIND_BY_SITE.get(base, "tri")


@contextmanager
def site_context(site: str):
    prev = getattr(_CTX, "site", None)
    _CTX.site = site
    try:
        yield
    finally:
        _CTX.site = prev


def current_site() -> str:
    return getattr(_CTX, "site", None) or "other"


def rows_without_bridge() -> Tuple[str, ...]:
    """The provider's rows minus this lever's: the ``prefer=`` the op hands the tier word when the bridge lever is not applied."""
    P = importlib.import_module(PROVIDER)
    return tuple(r for r in getattr(P, "ROW_NAMES", ()) if r != ROW and not str(r).startswith("proj+"))


def provider_rows() -> Tuple[str, ...]:
    """The provider's row names (``pallas:<row>`` words ablation.py accepts); () when the provider is not importable here."""
    try:
        return tuple(getattr(importlib.import_module(PROVIDER), "ROW_NAMES", ()))
    except Exception:  # noqa: BLE001
        return ()


def _bridge_served() -> Dict[str, int]:
    """The bridge's own served-row census (its report()), {} when it is not imported."""
    m = sys.modules.get(SERVE)
    if m is None:
        return {}
    try:
        rep = m.report()
    except Exception:  # noqa: BLE001
        return {}
    served = rep.get("served") if isinstance(rep, dict) else None
    return {str(k): int(v) for k, v in served.items()} if isinstance(served, dict) else {}


def provider_op(word: str):
    """The carried adapter's ``FLASH_OP``: ``op(q, k, v, bias, key_mask, scale)`` on heads-major operands (q / k / v ``[b, h, s, c]``, the pair
    bias ``[h, q, k]`` in the activations' dtype, the key mask ``[b, k]``) = ONE ``serve.attention(…, word=<word>, kind=<site's>, n_seq=b,
    layout="BHSD", direction="fwd")``; the bridge lever out of the set narrows the word to the provider's other rows. Counts the site, the row
    that served (the provider's report) and the cell on both census dicts."""
    def op(q, k, v, bias, key_mask, scale):
        S_ = importlib.import_module(PROVIDER_SERVE)
        site = current_site()
        kind = kind_of(site)
        B, H, S, D = int(q.shape[0]), int(q.shape[1]), int(q.shape[2]), int(q.shape[3])
        prefer = None if _STATE["enabled"] else rows_without_bridge()
        before = dict(getattr(S_, "COUNTS", {}) or {})
        bridge_before = _bridge_served()
        kw = dict(word=word, kind=kind, layout=LAYOUT, direction="fwd")
        if kind == "msarow":
            kw["n_seq"] = B
        if prefer is not None:
            kw["prefer"] = prefer
        sel = None
        try:
            fam = S_.family("attn", kind=kind, heads=H, head_dim=D, n_seq=(B if kind == "msarow" else None)) if hasattr(S_, "family") else None
            if fam is not None and hasattr(S_, "resolve"):
                sel = S_.resolve("attn", fam, q.dtype, S, word=word, direction="fwd", prefer=prefer, key_masked=key_mask is not None)
        except Exception:  # noqa: BLE001 - the selection is evidence for the census; the call below resolves again by itself
            sel = None
        out = S_.attention(q, k, v, bias, key_mask, scale, **kw, **({"selection": sel} if sel is not None else {}))
        row = _served_row(getattr(S_, "COUNTS", {}) or {}, before, sel)
        _record(site, row, sel, B, S, H, D, bridge_before)
        return out
    op.word = word                                   # type: ignore[attr-defined]
    op.colabfold_opt = NAME                          # type: ignore[attr-defined]
    return op


def _served_row(after: Dict, before: Dict, sel) -> str:
    """The arm the provider served for the call just made: the `served:…` counter that moved, else the selection's arm."""
    for key, n in after.items():
        if str(key).startswith("served") and int(n) > int(before.get(key, 0)):
            return str(key).split(":")[-1]
    return str(getattr(sel, "arm", None) or getattr(sel, "row", None) or "unknown")


def _record(site: str, row: str, sel, B: int, S: int, H: int, D: int, bridge_before: Dict[str, int]) -> None:
    base_row = row.split("@")[0].split(":")[0]
    _BIND["calls"] += 1
    _bump(_BCOUNTS, _BIND, "served", row)
    _bump(_BCOUNTS, _BIND, "sites", f"{site}:{base_row}")
    _bump(_BCOUNTS, _BIND, "cells", str(getattr(sel, "cell_key", None) or "none"))
    shape = f"{B}x{S}x{H}x{D}"
    if base_row == ROW:
        _STATE["calls"] += 1
        _bump(_COUNTS, _STATE, "sites", site)
        _bump(_COUNTS, _STATE, "shapes", shape)
        now = _bridge_served()
        moved = [r for r, n in now.items() if n > bridge_before.get(r, 0)] or ["served"]
        d = _COUNTS.setdefault("rows", {})
        d[moved[0]] = d.get(moved[0], 0) + 1
        _STATE["rows"] = _render(d)
    elif _STATE["enabled"]:
        _STATE["fallbacks"] += 1
        cands = [str(c).split("@")[0].split(":")[0] for c in (getattr(sel, "candidates", None) or [])]
        reason = "refused" if (cands and cands[0] == ROW) else CELL_RULE      # the bridge headed the order and did not serve: it refused at the call, the next row served
        _bump(_COUNTS, _STATE, "fallback_by", reason)


def bind_state() -> Dict[str, Any]:
    """The binding's census for AF_PALLAS_ATTN's LEVER line: word, calls through the provider, served rows, sites, cells, provider version."""
    return {"word": _BIND["word"], "provider_calls": _BIND["calls"], "served": _BIND["served"], "sites": _BIND["sites"], "cells": _BIND["cells"],
            "provider": _BIND["provider"]}


def _build(wrapped_cls):
    class Attention(wrapped_cls):                                # defined in a class body so haiku's metaclass wraps __call__ (the module's own name / parameter scope)
        def __call__(self, q_data, m_data, bias, nonbatched_bias=None):
            if nonbatched_bias is None:
                return super().__call__(q_data, m_data, bias, nonbatched_bias)
            with site_context(site_of(self)):                   # the site's word for the provider op below (the carried adapter's FLASH_OP)
                return super().__call__(q_data, m_data, bias, nonbatched_bias)

    setattr(Attention, MARKER, True)
    Attention._pallas_word_wrapped = wrapped_cls
    Attention.__qualname__ = Attention.__name__ = CLASS
    Attention.__module__ = wrapped_cls.__module__
    return Attention


def bind(kit, word: str, modules=None) -> str:
    """AF_PALLAS_ATTN's binding: the carried adapter's ``FLASH_OP[0]`` = ``provider_op(word)`` and ``modules.Attention`` rebound (over the
    adapter's class) to the site-naming subclass. Idempotent. Returns the word (the activation report's ``attn_word``)."""
    op = provider_op(word)
    if kit is not None and hasattr(kit, "FLASH_OP"):
        if _ORIG["op"] is None:
            _ORIG["op"] = kit.FLASH_OP[0]
        kit.FLASH_OP[0] = op
        _ORIG["kit"] = kit
    m = modules if modules is not None else importlib.import_module(MODULES)
    cur = getattr(m, CLASS)
    if not getattr(cur, MARKER, False):
        setattr(m, CLASS, _build(cur))
        _ORIG.update(cls=cur, modules=m)
    try:
        pv = getattr(importlib.import_module("opt_core"), "__version__", None)
    except Exception:  # noqa: BLE001
        pv = None
    _BIND.update(bound=True, word=word, provider=pv)
    _STATE["word"] = word
    return word


def require() -> Dict[str, Any]:
    """The lever's floor: the bridge importable from the shared core, its launcher loadable on this jaxlib, the GPU backend. Raises
    ``Refusal`` (kind = core_missing [absent or older than BRIDGE_MIN] | no_launcher | not_gpu_backend) otherwise; returns {"cc", "core", "jax"}."""
    try:
        TX = importlib.import_module(SERVE)
    except ImportError as e:
        raise Refusal("core_missing", f"{SERVE} is not importable ({e})") from None
    have = tuple(int(x) for x in str(getattr(TX, "__version__", "0.0.0")).split(".")[:3] if x.isdigit())
    if have < BRIDGE_MIN:
        raise Refusal("core_missing", f"{SERVE} {getattr(TX, '__version__', '?')} is older than {'.'.join(map(str, BRIDGE_MIN))} (the face's vmap rule)")
    try:
        import jax
    except ImportError as e:
        raise Refusal("no_jax", str(e)) from None
    if jax.default_backend() != "gpu":
        raise Refusal("not_gpu_backend", f"the default jax backend is {jax.default_backend()!r}, the kernels need the GPU backend")
    try:
        importlib.import_module(SERVE + "._launch").load()          # the XLA-FFI launcher for this jaxlib's FFI API version: refused by name when there is no build
    except Exception as e:  # noqa: BLE001 - the bridge's own words
        raise Refusal("no_launcher", str(e)) from None
    return {"cc": compute_capability(), "core": getattr(TX, "__version__", None), "jax": jax.__version__}


def enable(modules=None) -> Dict[str, Any]:
    """Let the provider's order reach the bridge row at the bound sites (idempotent). Requires ``require()`` to pass on this machine: a refusal
    raises by name and the activation is NOT ACTIVE."""
    facts = require()
    _STATE.update(cc=facts.get("cc"), core=facts.get("core"), enabled=True)
    if _BIND["word"] is not None:
        _STATE["word"] = _BIND["word"]
    return dict(_STATE)


def disable() -> None:
    _STATE["enabled"] = False


def unbind() -> None:
    m, cls = _ORIG["modules"], _ORIG["cls"]
    if m is not None and cls is not None:
        setattr(m, CLASS, cls)
    kit = _ORIG["kit"]
    if kit is not None and _ORIG["op"] is not None and hasattr(kit, "FLASH_OP"):
        kit.FLASH_OP[0] = _ORIG["op"]
    _ORIG.update(cls=None, modules=None, op=None, kit=None)
    _BIND.update(bound=False)


def marker_present(modules=None) -> Optional[bool]:
    """True when the bridge lever is applied in this process (it rebinds no class of its own: the binding's subclass is AF_PALLAS_ATTN's)."""
    return bool(_STATE["enabled"])


def bound(modules=None) -> Optional[bool]:
    m = modules if modules is not None else sys.modules.get(MODULES)
    if m is None:
        return None
    return bool(getattr(getattr(m, CLASS, None), MARKER, False))


def reset_for_tests() -> None:
    disable()
    unbind()
    _STATE.update(enabled=False, calls=0, fallbacks=0, fallback_by="none", rows="none", sites="none", shapes="none", word=None, cc=None, core=None)
    _COUNTS.update(fallback_by={}, sites={}, shapes={})
    _COUNTS.pop("rows", None)
    _BIND.update(bound=False, word=None, calls=0, served="none", sites="none", cells="none", provider=None)
    _BCOUNTS.update(served={}, sites={}, cells={})
    _CTX.site = None
