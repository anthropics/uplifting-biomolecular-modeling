"""Restore the truncated-normal form the sharded pair init requires.

``opt_core.mem.rowpair.rng.trunc_normal_rows`` carries a rows replica of only the WHOLE-TENSOR
REJECTION form of truncated-normal init, and refuses BY NAME on a torch whose
``nn.init.trunc_normal_`` is the inverse-CDF (``uniform_`` + ``erfinv_``) form:

    RowpairRefused: trunc_normal_rows: this torch's nn.init.trunc_normal_ is the 'inverse_cdf' form

The kit pins torch 2.13.0, which is the rejection form, so this never fires there.  On a newer
torch the P>1 route cannot start AT ALL -- not a memory problem, the run simply refuses.  This
restores the pinned form, which is also what makes the two forms' RNG streams comparable.

"""
import math

_STATE = {"applied": False}


def needed() -> bool:
    """True when this torch's trunc_normal_ is NOT the rejection form rowpair requires."""
    try:
        import torch.nn.init as _I
    except Exception:                                    # noqa: BLE001
        return False
    src = ""
    try:
        import inspect
        src = inspect.getsource(_I._no_grad_trunc_normal_)
    except Exception:                                    # noqa: BLE001
        return False
    # the inverse-CDF form draws uniformly and inverts; the rejection form draws normal_ and resamples
    return ("erfinv_" in src) or ("uniform_" in src and "normal_" not in src)


def apply(force: bool = False) -> dict:
    """Patch ``torch.nn.init._no_grad_trunc_normal_`` process-wide when (and only when) this torch carries the inverse-CDF form
    (``needed()``); returns ``{"applied": True, ...}`` when the shim is in force (now or from an earlier call), ``{"skipped": why}``
    otherwise. ``rowpair.install_rank`` names the outcome on its LEVER line (trunc_normal=native|shim)."""
    if _STATE["applied"]:
        return {"applied": True, "already": True, "torch": _STATE.get("torch")}
    if not (force or needed()):
        return {"skipped": "this torch already carries the rejection form"}

    import torch
    import torch.nn.init as _I

    def _no_grad_trunc_normal_(tensor, mean, std, a, b, generator=None):
        def norm_cdf(x):
            return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
        p = norm_cdf((b - mean) / std) - norm_cdf((a - mean) / std)
        if not (p > 0.3):
            raise RuntimeError("tn_shim: acceptance mass <= 0.3 (log-pdf branch not implemented)")
        with torch.no_grad():
            lo = tensor.new_tensor(a).item()
            hi = tensor.new_tensor(b).item()
            tensor.normal_(mean, std, generator=generator)
            while True:
                mask = (tensor < lo) | (tensor > hi)
                if not mask.any():
                    break
                cand = torch.empty_like(tensor).normal_(mean, std, generator=generator)
                tensor.copy_(torch.where(mask, cand, tensor))
                del cand
            del mask
            return tensor

    _I._no_grad_trunc_normal_ = _no_grad_trunc_normal_
    _STATE["applied"] = True; _STATE["torch"] = torch.__version__
    return {"applied": True, "torch": torch.__version__}
