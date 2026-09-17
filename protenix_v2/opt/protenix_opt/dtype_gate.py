"""The per-call dtype gate in front of the kit's bf16-only fused kernels, and the tally the verdict reads.

Stock accepts ``protenix pred --dtype fp32|fp16|bf16`` (``runner/batch_inference.py``: the trunk runs under ``torch.autocast(dtype)``; ``fp32``
disables autocast, so every trunk statement runs on fp32 operands; the diffusion sampler runs fp32 under every ``--dtype`` — stock's
``skip_amp.sample_diffusion``). The kit's fused MSA-module kernels (``opm_fused``, ``pwa_fused``: third_party/protenix_fpf_msa) and the Pairformer
pair-bias attention (``pf_attn``: ``apb_core``) reproduce the arithmetic of stock's bf16-autocast Linears (bf16 operands, fp32 statistics and
accumulation): that is their cell, and the packages assert it (``msa_triton``: ``outer.dtype == torch.bfloat16``). A kit accepts what
stock accepts, so a call OUTSIDE a live bf16 autocast region is not refused and not silently re-cast: it steps aside BY NAME, per call, to the
module's own statement (the class's method as this mode has it: stock's, or an exact-class lever's class-level forward such as ``pwa_zcache``),
counted under the word that says why — ``dtype_fp32`` (no autocast: ``--dtype fp32``) or ``dtype_fp16`` (fp16 autocast: ``--dtype fp16``).
The exact composition's ``atom_attn_exact`` (``apb_atom_exact``) has the same shape of rule for a different reason: its kernel serves
only the chunk batch counts whose cuBLAS numerics it determined at install; a call it cannot serve raises the package's ``Refused`` BY NAME, which
the kit's binding (``sampler_levers``) catches PER CALL and answers with the stock statement, counted under the refusal's word
(``cublas_route_full_chunk`` / ``cublas_route_remainder``) — under ``--dtype fp32`` the input embedder's atom encoder call enters the kernel's dtype
envelope with an un-chunked batch count the install never probes.

The gate is kit glue only: the sealed packages are untouched; the kit re-wraps the callable a package's ``install`` put on the module (an
instance-level ``forward`` / ``standard_multihead_attention``) or on the module global (``primitives._local_attention``) from the outside.

Tally (one per gated lever, ``new_tally()``): ``calls`` = calls that reached the site, ``served`` = calls the lever's own callable answered
(for atom_attn_exact: kernel or the package's own counted original path), ``aside`` = ``{word: n}``; ``calls == served + sum(aside)``.
The LEVER line carries ``aside=<word>:<n>[,<word>:<n>]`` only when a call stepped aside (a bf16 run's lines are unchanged); the verdict
(``stack.served_census``) reads the tally: every call accounted for (served, or a declared aside) is the lever working as designed — it stays
``applied``, ``partial`` stays false —; calls that reached the site with none served and some unaccounted are a fallback by name (exit 3).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

WORD_FP32 = "dtype_fp32"          # no autocast region live at the call (stock `--dtype fp32`, or a module stock runs with autocast disabled)
WORD_FP16 = "dtype_fp16"          # an fp16 autocast region (stock `--dtype fp16`)
REFUSED = "refused"               # an atom_attn_exact Refused whose text names neither chunk form (never expected; still counted, never raised)


def new_tally() -> Dict[str, Any]:
    return {"calls": 0, "served": 0, "aside": {}}


def device_type(args, kwargs, default: str = "cuda") -> str:
    """The device type of the call's first tensor-like argument (``.device.type``); ``default`` when none carries one."""
    for x in tuple(args) + tuple(kwargs.values()):
        t = getattr(getattr(x, "device", None), "type", None)
        if isinstance(t, str):
            return t
    return default


def autocast_word(torch, device: str = "cuda") -> Optional[str]:
    """None when a bf16 autocast region is live for ``device`` (the fused kernels' cell: stock's Linears take bf16 operands there); else the
    aside word — ``dtype_fp16`` under an fp16 autocast region, ``dtype_fp32`` otherwise. Pure function of the autocast state (never of a
    tensor's dtype: under bf16 autocast stock's MSA / pair tensors are a mix of fp32 and bf16 and the kernels serve exactly that)."""
    try:
        on = torch.is_autocast_enabled(device)
    except TypeError:                                   # torch < 2.4: no device argument (the bare call is the CUDA state)
        on = torch.is_autocast_enabled() if device == "cuda" else torch.is_autocast_cpu_enabled()
    if not on:
        return WORD_FP32
    try:
        dt = torch.get_autocast_dtype(device)
    except (AttributeError, TypeError):
        dt = torch.get_autocast_gpu_dtype() if device == "cuda" else torch.get_autocast_cpu_dtype()
    if dt == torch.bfloat16:
        return None
    return WORD_FP16 if dt == torch.float16 else WORD_FP32


def count_aside(tally: Dict[str, Any], word: str) -> None:
    tally["aside"][word] = int(tally["aside"].get(word, 0)) + 1


def gate_method(module, attr: str, kernel_fn: Callable, tally: Dict[str, Any], torch, lever: str) -> Callable:
    """The instance-level replacement for ``module.<attr>`` (the package's ``kernel_fn`` is what its install put there): under a bf16 autocast
    region the package's callable, counted ``served``; otherwise the class's ``attr`` bound to the module — the statement the module runs
    without this lever, resolved at the call (a class-level lever of the mode keeps its place) —, counted under the aside word. Carries the
    package's ownership marks (``_fpf_msa`` / ``_fpf_apb``, ``__wrapped__``) so the other levers' "owned by" reads are unchanged."""

    def gated(*a, **kw):
        tally["calls"] += 1
        word = autocast_word(torch, device_type(a, kw))
        if word is None:
            out = kernel_fn(*a, **kw)
            tally["served"] += 1
            return out
        count_aside(tally, word)
        return getattr(type(module), attr)(module, *a, **kw)

    for mark in ("_fpf_msa", "_fpf_apb", "__wrapped__"):
        if hasattr(kernel_fn, mark):
            setattr(gated, mark, getattr(kernel_fn, mark))
    gated.__name__ = getattr(kernel_fn, "__name__", None) or getattr(getattr(kernel_fn, "func", None), "__name__", attr)
    gated._ptx_dtype_gate = lever
    gated._ptx_kernel_fn = kernel_fn
    return gated


def refused_word(exc: BaseException) -> str:
    """The aside word for an atom_attn_exact ``Refused``: which chunk form's cuBLAS numerics the install did not determine (whitespace-free,
    digits-free so one word covers every batch count)."""
    s = str(exc)
    if "remainder chunk" in s:
        return "cublas_route_remainder"
    if "full chunks" in s:
        return "cublas_route_full_chunk"
    return REFUSED


def gate_refusable(fused: Callable, original: Callable, refused: Tuple[type, ...], tally: Dict[str, Any],
                   on_aside: Optional[Callable[[str], None]] = None) -> Callable:
    """The replacement for a module-global site whose kernel callable raises ``refused`` BY NAME on a call outside its envelope
    (atom_attn_exact's ``primitives._local_attention``): the call is answered by ``original`` (the stock statement the package's install
    displaced), counted under ``refused_word``; ``on_aside(word)`` lets the binding keep the package's own census truthful."""

    def gated(*a, **kw):
        tally["calls"] += 1
        try:
            out = fused(*a, **kw)
        except refused as e:
            word = refused_word(e)
            count_aside(tally, word)
            if on_aside is not None:
                on_aside(word)
            return original(*a, **kw)
        tally["served"] += 1
        return out

    for mark in ("_atom_attn_exact",):
        if hasattr(fused, mark):
            setattr(gated, mark, getattr(fused, mark))
    gated.__name__ = getattr(fused, "__name__", "gated")
    gated.__wrapped__ = original
    gated._ptx_dtype_gate = "atom_attn_exact"
    gated._ptx_kernel_fn = fused
    return gated


def aside_token(tally: Optional[Dict[str, Any]]) -> Optional[str]:
    """``<word>:<n>[,<word>:<n>]`` (sorted) when a call stepped aside, else None (the LEVER line then carries no ``aside`` pair)."""
    aside = (tally or {}).get("aside") or {}
    if not aside:
        return None
    return ",".join(f"{w}:{int(n)}" for w, n in sorted(aside.items()))


def census(tally: Optional[Dict[str, Any]]) -> Tuple[Optional[bool], str]:
    """The verdict on one gate tally, in ``stack.served_census``'s form ``(ok, detail)``: None = no call reached the site (nothing to judge);
    True = the lever served calls, or every call stepped aside by name (declared: by design); False = calls reached the site, none was served
    and some are unaccounted for (an undeclared served-0: a fallback by name)."""
    t = tally or {}
    calls, served = int(t.get("calls") or 0), int(t.get("served") or 0)
    aside = {str(k): int(v) for k, v in (t.get("aside") or {}).items()}
    n_aside = sum(aside.values())
    detail = f"calls={calls} served={served} aside={aside_token(t) or 'none'}"
    if calls <= 0:
        return None, "no call reached the site: " + detail
    if served > 0:
        return True, detail
    if n_aside >= calls:
        return True, "every call stepped aside by name: " + detail
    return False, f"served 0 of {calls} calls, {calls - n_aside} not accounted for by a declared aside: " + detail
