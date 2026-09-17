"""esmc_opt — explicit interface to the ESM C inference optimizations; a drop-in beside the user's own ESM C code.

    import esmc_opt
    report = esmc_opt.enable("exact")                        # or "off"; idempotent, once per process. variant="6b" names the size; unnamed,
    esmc_opt.status()                                        # it is derived from the model loaded at ESMC.from_pretrained. status() = the last report

or, without code changes, `ESMC_OPT=exact python my_script.py`: the package's .pth installs a lazy import hook that activates the mode
the first time the upstream package (`esm`; `_autoload.TRIGGERS`) is imported. Nothing is imported at interpreter start beyond this
module; torch and the upstream package load only when a mode is activated.

ESM C has no command line upstream: the model is a Python API (`esm.models.esmc.ESMC.from_pretrained` + `encode` + `logits`), and the
kits patch a LOADED model (`apply(client.model)`). Activation is therefore in two steps: `enable()` resolves and gates the mode (applies
the datapath lever and arms a wrapper on `ESMC.from_pretrained`), and the model-side lever is applied to each client by the kit's own
`apply()` when the client is built, at `ESMC.from_pretrained`. A lever of the mode NOT APPLIED when the client is built
(`levers_fallback=`: its kit cannot run here) is a refusal by name — a mode is all of its levers, so `ESMC.from_pretrained` prints
`NOT ACTIVE: partial activation — <levers: reasons> …` and raises `stack.PartialActivation` (an `ActivationError`). Environment drift
(versions off pin, an unrecorded shape, a cache miss) is never a refusal: it is named on the lines and the levers engage.

Modes (`modes.MODES`): "exact" = the kits' exact lever set for the variant (byte-identical outputs to stock), "off" = nothing applied
(upstream runs untouched). The package default is "exact"; there is no fast mode for this model. Every activation proves the
accelerators the BUILT model engages (`kernels.py`: the KERNELS line, read off the bound objects; a shortfall against the pinned stack is
named on the KERNELS SHORT line and the run proceeds on what the model engages). Variants (`modes.VARIANTS`): "300m", "600m", "6b" —
several models may live in one process, one lever set per process. Regimes (`modes.REGIMES`): "b1" (the SDK's one-protein call) and
"batched" (the caller's own padded batched forward); `registry.LEVERS` describes each lever, the kits' own code applies them (stack.py).

The contract: `enable(mode, variant=..., regime=...)` returns the activation report (a dict; `active` says whether the levers are on,
`reason` says why not), `status()` returns the last report. Late activation: `enable()` may run before or after the upstream package is
imported, but is refused by name once an `ESMC` client exists in the process (a client built before activation would run unpatched;
instances are counted by a constructor wrap on the client class from activation on, `stack.instance_check`) or once a kit's own module
reports a lever applied (`stack.kit_levers_applied`); repeated calls return the first report, and a different mode, variant or regime
in the same process is refused. `check` (the one CLI verb) is the same resolution as a dry run: it gates the mode on this box and
applies nothing.
"""
__version__ = "0.4.0"
__all__ = ["enable", "status", "MODES", "VARIANTS", "REGIMES", "ActivationError", "registry", "modes", "stack", "kernels"]


class ActivationError(RuntimeError):
    """A requested mode could not be activated (pins, no GPU, kit missing, variant conflict, or a lever failed to apply)."""


def enable(mode: str, variant: str = None, regime: str = None, *, strict: bool = False, trigger: str = None) -> dict:
    """Activate `mode` ("exact" | "off") for `variant` ("300m" | "600m" | "6b"; None = derived from the model loaded at
    `ESMC.from_pretrained`) and `regime` ("b1" | "batched"; default "b1") in this process; returns the activation report
    {"active", "mode", "variant", "regime", "levers", "levers_out", "levers_applied", "levers_fallback", "gpu": {"name", "sm", "mem_mib"},
    "upstream": {esm, torch, flash_attn, transformer_engine, triton}, "package_version", "reason" (when inactive)}.
    `strict=True` raises ActivationError instead of returning an inactive report."""
    from . import stack
    try:
        return stack.activate(mode, variant, regime, strict=strict, trigger=trigger or "explicit")
    except stack.ActivationError as e:
        raise ActivationError(str(e)) from None


def status() -> dict:
    """The last activation report, or {"active": False, "reason": ...} before any activation."""
    from . import stack
    return stack.status()


def __getattr__(name):
    if name in ("MODES", "VARIANTS", "REGIMES"):
        from . import modes
        return getattr(modes, name)
    if name in ("registry", "modes", "stack", "manifest", "det", "driver", "outputs", "inputs", "report", "kernels"):
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(name)
