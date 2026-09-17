"""The attention words — which attention kernels the upstream modules dispatch in this process, read from what is bound at run time.

Two upstream modules decide, at import, between flash-attn and their fallbacks; no kit lever touches either:

  fold_atom_attn  ``transformers.models.esmfold2.modeling_esmfold2_common``: ``FLASH_ATTN_AVAILABLE`` (l.25-41, ``from flash_attn import
                  flash_attn_func, flash_attn_varlen_func``) → ``SWA3DRoPEAttention.forward`` (l.578-637) runs ``flash_attn_varlen_func`` /
                  ``flash_attn_func`` (sliding window) in every fold of every arm — the input embedder's atom encoder and the structure module's
                  atom encoder / decoder — else ``F.scaled_dot_product_attention`` over a dense window mask. Word: ``flash_attn | sdpa``.
  esmc_attn       ``transformers.models.esmc.modeling_esmc``: ``_scaled_dot_product_attention`` (l.607-661) of the design loop's ESMC-6B (the
                  pseudo-perplexity term, fwd+bwd) and the fold's shared LM: xformers ``memory_efficient_attention`` when installed (l.77-87,
                  629-637), else ``flash_attn_func`` (l.638-641, bf16/fp16), else ``F.scaled_dot_product_attention``; a call carrying a chain
                  ``sequence_id`` runs SDPA with its mask in every case. The whole-module flash class ``_FlashMultiHeadAttention`` is bound only
                  when a model was loaded with ``attn_implementation="flash_attention_2"`` (``_use_flash_attn``, l.1027-1028) — read off the
                  live modules when models are passed. Word: ``xformers | flash_attn | sdpa``.
  esmc_rope       the same module's ``_flash_attn_rotary_available`` (l.56-62): ``RotaryEmbedding.forward`` (l.424-429) applies RoPE with
                  flash-attn's Triton kernel (``flash_attn.ops.triton.rotary.apply_rotary``, a raw launcher outside autograd: the rotated q/k
                  carry no grad_fn) or with the module's torch arithmetic. Word: ``flash_triton | torch``: ``flash_triton`` as imported on a box with
                  flash-attn; ``torch`` when the arm applies upstream fix EF2INV-0003 (``--upstream-fix EF2INV-0003``, patches.pin_esmc_rope).
  esmc_mlp        the same module's ``_te_available`` (l.64-75, ``import transformer_engine.pytorch as te``) decides at model BUILD time whether
                  the ESM-C blocks hold Transformer Engine's fused ``te.LayerNormMLP`` / ``te.LayerNormLinear`` / ``te.Linear`` (l.543-577) or the
                  module's ``_PyTorchLayerNormMLP`` / ``_PyTorchLayerNormLinear`` / ``nn.Linear`` — read off the live models' submodule classes
                  (the flag alone when no model is passed). Word: ``te | torch``.
  det_attn        the det recipe's attention element (det.py), present at ``--det >= 1``: ``flash_det`` — under it ``esmc_attn`` reads
                  ``flash_attn`` (xformers bypassed for the deterministic flash_attn_func).

``read`` is the one reader (the design script for every arm, ``check``); ``words`` the one formatter of the proof segment
``esmc_mlp=… esmc_attn=… esmc_rope=… fold_atom_attn=…[ det_attn=…]``; ``require_problems`` the fail-loud guard behind
``EF2INV_REQUIRE_FAST_ENV=1`` (configs/h100.env; ``guard_ruling``, every mode alike): a box on which any word is not the pinned stack's accelerated value
is worded on ONE info line naming the word and the packages that bind it, and the run proceeds with the mode's levers engaged — the speed
figures apply to the pinned stack only, never a reason to refuse; ``esmc_rope=torch`` is exempt exactly when the arm applied upstream fix
EF2INV-0003 (``--upstream-fix EF2INV-0003``: the differentiable RoPE, patches.pin_esmc_rope), an exemption the proof line names.
"""
from __future__ import annotations

import contextlib
import functools
import importlib
import sys
from typing import Dict, Iterable, Optional

ESMC_MODULE = "transformers.models.esmc.modeling_esmc"
FOLD_MODULE = "transformers.models.esmfold2.modeling_esmfold2_common"
FLASH_DIST = "flash-attn"
TE_DIST = "transformer_engine"
REQUIRE_VAR = "EF2INV_REQUIRE_FAST_ENV"            # configs/h100.env; read by the CLI, reaches the arm as --require-fast-env 1 (never as a variable)
WORD_KEYS = ("esmc_mlp", "esmc_attn", "esmc_rope", "fold_atom_attn")
FAST_ENV = {"esmc_mlp": "te", "esmc_attn": "xformers", "esmc_rope": "flash_triton", "fold_atom_attn": "flash_attn"}   # the pinned stack's values (STOCK.md §Stack)
TE_CLASSES = ("LayerNormMLP", "LayerNormLinear", "Linear")                 # transformer_engine.pytorch classes the fork builds ESM-C blocks from (l.543-577)
TORCH_CLASSES = ("_PyTorchLayerNormMLP", "_PyTorchLayerNormLinear")       # the fork's pure-torch fallbacks (nn.Linear for the out-projection)


def _module(name: str, modules: Optional[Dict[str, object]] = None):
    if modules is not None:
        return modules.get(name)
    try:
        with contextlib.redirect_stdout(sys.stderr):   # the fork prints import-time notices on stdout (auto_docstring); `check --json` stays one JSON document
            return importlib.import_module(name)
    except Exception:                                # noqa: BLE001 — an absent stack reads as words 'unread', never a crash of the reader
        return None


def dist_version(name: str = FLASH_DIST) -> Optional[str]:
    try:
        from importlib import metadata
        return metadata.version(name)
    except Exception:                                # noqa: BLE001
        return None


def xformers_facts() -> Optional[dict]:
    """xformers' version and which of its fmha forward ops are usable here (``memory_efficient_attention`` takes the first available in its
    dispatch order: the flash op — whose NAME carries the flash-attn build it binds — before cutlass). None without xformers."""
    v = dist_version("xformers")
    if not v:
        return None
    ops = {}
    try:
        with contextlib.redirect_stdout(sys.stderr):
            from xformers.ops import fmha
        for name in ("flash", "cutlass"):
            op = getattr(getattr(fmha, name, None), "FwOp", None)
            if op is None:
                continue
            try:
                ok = bool(op.is_available())
            except Exception as e:                   # noqa: BLE001
                ok = f"unknown ({type(e).__name__})"
            ops[str(getattr(op, "NAME", name))] = ok
    except Exception as e:                           # noqa: BLE001
        ops["error"] = repr(e)[:200]
    return {"version": v, "fw_ops": ops}


def te_version() -> Optional[str]:
    """transformer_engine's version: the distribution (``transformer_engine`` / ``transformer-engine``), else the module attribute."""
    for n in (TE_DIST, "transformer-engine", "transformer_engine_torch"):
        v = dist_version(n)
        if v:
            return v
    return None


def callable_word(fn) -> Optional[str]:
    """What a bound attention callable is: None | '<module>.<qualname>' | the same + '(deterministic=True)' for the det recipe's partial."""
    if fn is None:
        return None
    det = isinstance(fn, functools.partial) and fn.keywords.get("deterministic") is True
    f = fn.func if isinstance(fn, functools.partial) else fn
    return f"{getattr(f, '__module__', '?')}.{getattr(f, '__qualname__', getattr(f, '__name__', '?'))}" + ("(deterministic=True)" if det else "")


def read(esmc_models: Iterable[object] = (), det: Optional[dict] = None, modules: Optional[Dict[str, object]] = None, flash_version: Optional[str] = None,
         te_ver: Optional[str] = None) -> dict:
    """The dispatch facts of this process. ``esmc_models``: live ESMC modules (the loop's ``esmc_model``, the fold's ``_esmc``) whose bound
    attention classes are counted; ``det``: the det record (det.apply) for the ``det_attn`` word; ``modules`` / ``flash_version``: injection
    for the CPU tests (fake upstream modules), else the imported modules and pip metadata."""
    E = _module(ESMC_MODULE, modules); C = _module(FOLD_MODULE, modules)
    rec: dict = {"flash_attn_version": flash_version if modules is not None else dist_version(),
                 "transformer_engine_version": te_ver if modules is not None else te_version(),
                 "xformers": xformers_facts() if modules is None else None,
                 "modules_read": {"esmc": ESMC_MODULE if E is not None else None, "fold": FOLD_MODULE if C is not None else None}}
    fold = {"FLASH_ATTN_AVAILABLE": bool(getattr(C, "FLASH_ATTN_AVAILABLE", False)) if C is not None else None,
            "flash_attn_func": callable_word(getattr(C, "flash_attn_func", None)) if C is not None else None,
            "flash_attn_varlen_func": callable_word(getattr(C, "flash_attn_varlen_func", None)) if C is not None else None}
    esmc = {"_xformers_available": bool(getattr(E, "_xformers_available", False)) if E is not None else None,
            "_flash_attn_available": bool(getattr(E, "_flash_attn_available", False)) if E is not None else None,
            "_flash_attn_rotary_available": bool(getattr(E, "_flash_attn_rotary_available", False)) if E is not None else None,
            "_te_available": bool(getattr(E, "_te_available", False)) if E is not None else None,
            "flash_attn_func": callable_word(getattr(E, "flash_attn_func", None)) if E is not None else None,
            "apply_triton_rotary": callable_word(getattr(E, "apply_triton_rotary", None)) if E is not None else None}
    flash_cls = getattr(E, "_FlashMultiHeadAttention", None) if E is not None else None
    te = getattr(E, "te", None) if E is not None else None
    te_cls = tuple(c for c in (getattr(te, n, None) for n in TE_CLASSES) if isinstance(c, type)) if te is not None else ()
    torch_cls = tuple(c for c in (getattr(E, n, None) for n in TORCH_CLASSES) if isinstance(c, type)) if E is not None else ()
    n_models = n_flash_mha = n_te = n_torch = 0; use_flags = []
    for m in esmc_models or ():
        n_models += 1
        use_flags.append(getattr(m, "_use_flash_attn", None))
        for sub in (m.modules() if hasattr(m, "modules") else []):
            if flash_cls is not None and isinstance(sub, flash_cls):
                n_flash_mha += 1
            if te_cls and isinstance(sub, te_cls):
                n_te += 1
            elif torch_cls and isinstance(sub, torch_cls):
                n_torch += 1
    esmc.update({"models_read": n_models, "use_flash_attn": use_flags, "flash_mha_modules": n_flash_mha, "te_modules": n_te, "torch_fallback_modules": n_torch})
    if C is None:
        fold_word = "unread"
    else:
        fold_word = "flash_attn" if (fold["FLASH_ATTN_AVAILABLE"] and getattr(C, "flash_attn_func", None) is not None) else "sdpa"
    if E is None:
        attn_word = rope_word = mlp_word = "unread"
    else:
        if n_models:                                  # live models: what the blocks HOLD (te.* vs the torch fallbacks), never the flag alone
            mlp_word = "te" if (n_te and not n_torch) else ("torch" if not n_te else "mixed")
        else:
            mlp_word = "te" if (esmc["_te_available"] and te_cls) else "torch"
        if n_flash_mha or any(bool(f) for f in use_flags):
            attn_word = "flash_attn"
        elif esmc["_xformers_available"]:
            attn_word = "xformers"
        elif esmc["_flash_attn_available"] and getattr(E, "flash_attn_func", None) is not None:
            attn_word = "flash_attn"
        else:
            attn_word = "sdpa"
        rope_word = "flash_triton" if esmc["_flash_attn_rotary_available"] else "torch"
    words = {"esmc_mlp": mlp_word, "esmc_attn": attn_word, "esmc_rope": rope_word, "fold_atom_attn": fold_word}
    if det and det.get("det_attn"):
        words["det_attn"] = det["det_attn"]
    rec.update({"fold": fold, "esmc": esmc, "words": words})
    return rec


def words(rec: dict) -> str:
    """``esmc_mlp=W esmc_attn=X esmc_rope=Y fold_atom_attn=Z[ det_attn=V]`` — the proof / check segment, one grammar for every route."""
    w = rec.get("words") or {}
    s = " ".join(f"{k}={w.get(k, 'unread')}" for k in WORD_KEYS)
    return s + (f" det_attn={w['det_attn']}" if w.get("det_attn") else "")




def guard_ruling(problems: list, mode_name: str) -> Optional[str]:
    """The fast-environment guard: the ONE info line (``report.fast_env_line(mode, problems)``) on any sentence, every mode alike — it names
    the words that are not the pinned stack's and the packages that bind them; the mode's levers engage unchanged and the run proceeds (a box
    that lacks an accelerated path is worded, never refused; the speed figures apply to the pinned stack only). ``None`` without a sentence."""
    if not problems:
        return None
    from . import report as R
    return R.fast_env_line(mode_name, problems)


def require_problems(rec: dict, image: Optional[str] = None, base_image: Optional[str] = None, rope_pin: Optional[str] = None) -> list:
    """The sentences of ``EF2INV_REQUIRE_FAST_ENV=1``: empty when flash-attn and transformer_engine are installed AND every word is
    its ``FAST_ENV`` value; else ONE sentence naming each differing word and the packages of the pinned stack that bind it (worded on the
    guard's info line; the run proceeds). ``image`` / ``base_image`` are accepted for the callers' positional form and never worded.
    ``rope_pin='torch'``: the arm applied upstream fix EF2INV-0003, so ``esmc_rope=torch`` is exempt by name (``None``: no exemption — the
    word must be the stack's ``flash_triton``); a det arm's ``det_attn`` element (in ``rec['words']``)
    exempts the attention word it moves by design (``flash_det``: ``esmc_attn=flash_attn``). Exemptions are recorded in ``rec['require_exempt']`` and printed on the ATTN line."""
    why = []
    if not rec.get("flash_attn_version"):
        why.append("flash_attn is not installed")
    if not rec.get("transformer_engine_version"):
        why.append("transformer_engine is not installed")
    w = rec.get("words") or {}
    exempt = []
    for k in WORD_KEYS:
        got, want = w.get(k, "unread"), FAST_ENV[k]
        if got == want:
            continue
        if k == "esmc_rope" and rope_pin == "torch" and got == "torch":
            exempt.append("esmc_rope=torch (upstream fix EF2INV-0003: patches.pin_esmc_rope, the differentiable RoPE)")
            continue
        det_attn = w.get("det_attn")                 # the det recipe's attention element (det.py) moves ESM-C attention off xformers by design: named, not refused
        if det_attn == "flash_det" and k == "esmc_attn" and got == "flash_attn":
            exempt.append("esmc_attn=flash_attn (det_attn=flash_det: xformers bypassed under the det recipe for flash_attn_func(deterministic=True))")
            continue
        flag = {"esmc_mlp": "_te_available", "esmc_attn": "_xformers_available", "esmc_rope": "_flash_attn_rotary_available"}.get(k)
        src = f"{FOLD_MODULE}.FLASH_ATTN_AVAILABLE={(rec.get('fold') or {}).get('FLASH_ATTN_AVAILABLE')}" if k == "fold_atom_attn" else f"{ESMC_MODULE}.{flag}={(rec.get('esmc') or {}).get(flag)}"
        why.append(f"{k}={got} (expected {want}; {src})")
    rec["require_exempt"] = exempt
    if not why:
        return []
    return [f"fast environment ({REQUIRE_VAR}=1) NOT present: {' and '.join(why)} — this box does not carry the pinned stack "
            f"(install flash_attn / transformer_engine / xformers as STOCK.md §Pinned stack for its speed figures); the levers engage on what is bound"]
