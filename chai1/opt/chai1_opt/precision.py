"""The package's opt-in precision lever (modes.OPTIN_LEVERS; F4 precision policy): ``tf32``.

``tf32`` applies the shared core's numerics policy ``Policy(matmul="high")`` (``opt_core.precision.policy``: torch's fp32 matmul
precision word, which sets ``torch.backends.cuda.matmul.allow_tf32 = True``) for the process: every fp32 GEMM cuBLAS runs — the transpiled denoiser's
(its linears and matmuls are fp32 as traced), the flat embedders' and the confidence head's — uses TF32 tensor-core products (10-bit
mantissa inputs, fp32 accumulate) instead of fp32 SGEMM. The eager trunk's linears run in bf16 as traced and are unaffected; cuDNN's own
TF32 switch (convolutions) is left alone — the model has none. The lever is applied once, after the deterministic recipe and before the
eager stack installs (stack.activate / driver.run), so the hoisted denoiser step is captured into its CUDA graph with TF32 kernels; it is
probed from torch's own flag (the registry's ``("flag", <path>)`` probe), never asserted. Numerics: TF32 changes the rounding of every fp32
product, so the lever rides ``fast`` and ``big`` (modes.OptIn.modes; implied by those rows) as tier 2 — inside stock's seed-to-seed band —
and never ``exact``.
"""
from __future__ import annotations

from typing import Dict, Tuple

from . import modes, registry


def _flag(torch, name: str):
    """``(owner object, attribute)`` of the torch switch the lever's registry probe names (``("flag", "backends.cuda.matmul.allow_tf32")``)."""
    if name not in modes.OPTIN_LEVERS or name not in registry.FLAG_PROBES:
        raise KeyError(f"{name!r} is not an opt-in lever ({','.join(modes.OPTIN_LEVERS)})")
    path = registry.FLAG_PROBES[name][1].split(".")
    obj = torch
    for p in path[:-1]:
        obj = getattr(obj, p)
    return obj, path[-1]


POLICIES = {                                                     # lever -> the shared core's numerics Policy it applies (opt_core.precision.policy)
    "tf32": dict(name="tf32", matmul="high", note="TF32 tensor-core products for the process's fp32 GEMMs; cuDNN TF32 untouched (no convolutions)"),
}


def apply(names: Tuple[str, ...], torch) -> Dict[str, object]:
    """Apply the requested flag levers in order through the shared core's policy (``opt_core.precision.policy.apply``: ``tf32`` =
    ``Policy(matmul="high")``, torch's fp32 matmul precision word, which drives ``torch.backends.cuda.matmul.allow_tf32``); returns
    ``{name: the flag read back after the write, name_policy: the core's activation record}``. modes.optin_refusal gates the names
    before this runs; an unknown name here raises KeyError."""
    out: Dict[str, object] = {}
    if not names:
        return out
    from ._core import ensure_importable
    ensure_importable()
    from opt_core.precision.policy import Policy, apply as apply_policy
    for n in names:
        obj, attr = _flag(torch, n)
        if n not in POLICIES:
            raise KeyError(f"{n!r} has no precision policy ({','.join(POLICIES)})")
        out[f"{n}_policy"] = apply_policy(Policy(**POLICIES[n]), torch)
        out[n] = getattr(obj, attr)
    return out


def numerics(torch) -> Dict[str, object]:
    """The process numerics signature (``opt_core.precision.policy.numerics_signature``: fp32 matmul precision, TF32 flags, cuDNN
    deterministic / benchmark, deterministic algorithms, autocast state) as a dict — recorded once per activation on the report and
    printed as the ``NUMERICS`` line; ``{"error": ...}`` when torch cannot be read (named, never absent)."""
    try:
        from ._core import ensure_importable
        ensure_importable()
        from opt_core.precision.policy import numerics_signature
        return dict(numerics_signature(torch))
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def probe(name: str, torch) -> bool:
    """True iff torch's own state shows the lever applied (its flag reads True); a torch without the switch shows nothing applied."""
    try:
        obj, attr = _flag(torch, name)
        return bool(getattr(obj, attr))
    except AttributeError:
        return False


def applied(torch) -> Tuple[str, ...]:
    """Every opt-in lever torch's state shows applied in this process."""
    return tuple(n for n in modes.OPTIN_LEVERS if n in registry.FLAG_PROBES and probe(n, torch))
