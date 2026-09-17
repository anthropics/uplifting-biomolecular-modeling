"""af3_torch_opt — explicit interface to the PyTorch AlphaFold 3 port (xfold @ 22bdeed + OpenFold3-layout patches) running the converted
OpenFold3-preview2 parameters, with the kit's fused kernels and the DTK FusedDiT add-on.

    import af3_torch_opt
    report = af3_torch_opt.enable("fast")        # or "off"; resolves + gates on this box ("fast" is the package default; "off", the stock path, by name)
    af3_torch_opt.status()                       # the last activation report
    af3_torch_opt.check("off")                   # the dry run

or, without code: `af3-torch-opt pred --mode fast --json_path in.json --output_dir out` (== `python -m af3_torch_opt ...`,
== `run.sh pred --config h100 ...`), or `AF3_TORCH_OPT=fast` in the environment of that command.

The model is a Python API inside the kit (`opt/forward/af3t/af3_torch/af3_torch_api.py`) that needs the image's torch venv; its
featurisation and its output writers are the fork's (`sokrypton/alphafold3`, the image's JAX venv). The package therefore runs on its own
interpreter and composes three model processes per prediction (cli.py: featurise → forward → postprocess); it never imports torch, jax or
the fork, and nothing runs at interpreter start (`_autoload.py` applies nothing). `AF3_TORCH_OPT` in the environment is honoured by
the wrapper command only — under it, a direct import of the kit's api is refused (`NOT ACTIVE`, exit 3: that process cannot apply the
mode and would run stock under the variable), as is a value that is not a mode, and any undeclared `AF3_TORCH_OPT_*` name (`stack.gates`).

Modes (`modes.MODES`): "off" = the kit's eager set, no DTK (the port's own baseline),
"exact" = the stock-kernels base (xfold's shipped fastnn kernels) + the levers byte-equal to it (`registry.EXACT`: bf16 weights, the
whole-step-graphed sampler) + `template_dedupe` — `off`'s outputs byte for byte (xfold's shipped fastnn kernels); "fast" = the kit's `fastest` set + DTK
FusedDiT (tier 2), "big" = the memory line on the fast base. One variant: the OpenFold3-preview2 parameters (`p2`: the public checkpoint converted by the reference fork's converter, stock/PINS.json variants).

The contract: `enable(mode, n_gpu=None)` returns the activation report (`active`, `mode`, `lever_set`, `levers`, `dtk`, `n_gpu`, `padding`,
the interpreters, `params_dir`, `cache_root`, `image`, `upstream`, `package_version`, `reason` when inactive);
`status()` the last report; `check()` the dry run; `cli.last_run()` the last pred's record (run_record.py). Every route runs
the pin gate first (`_core_gate.gate`, the core's kit template carried byte-for-byte: the importable `opt_core` must be the one
`opt/pyproject.toml [tool.opt_core]` pins — `NOT ACTIVE: reason=core_missing:opt_core` / `reason=core_mismatch: …`, exit 3) and then the
producers gate (`_producers.refuse_if_missing`: `reason=producer_missing:<modules>`, exit 3), before any `opt_core` import.
"""
__version__ = "0.2.16"

VARIANTS = ("p2",)
_LAZY = {"enable": ("stack", "activate"), "activate": ("stack", "activate"), "check": ("stack", "check"), "status": ("stack", "status"),
         "ActivationError": ("stack", "ActivationError"), "MODES": ("modes", "MODES"), "DEFAULT_MODE": ("modes", "DEFAULT_MODE"), "UnsupportedMode": ("modes", "UnsupportedMode")}

__all__ = ["enable", "activate", "check", "status", "ActivationError", "UnsupportedMode", "MODES", "DEFAULT_MODE", "VARIANTS", "__version__"]


def __getattr__(name):          # PEP 562: the interface resolves on first use, so the .pth hook's `import af3_torch_opt._autoload` imports nothing of the
    if name in _LAZY:           # shared core at interpreter start — a core missing from the interpreter is the hook's named NOT ACTIVE, never site's traceback
        import importlib
        from ._core_gate import gate
        gate(__file__, tag="af3-torch-opt")   # the in-process route (enable / activate / check / status / MODES ...): THE pin gate first — an opt_core absent or not
        from ._producers import refuse_if_missing   # the pinned one is the NOT ACTIVE line and SystemExit(3) BEFORE any opt_core submodule import; then the
        refuse_if_missing(argv=())            # producers gate (producer_missing: the finer words)
        mod, attr = _LAZY[name]
        value = getattr(importlib.import_module(f"{__name__}.{mod}"), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
