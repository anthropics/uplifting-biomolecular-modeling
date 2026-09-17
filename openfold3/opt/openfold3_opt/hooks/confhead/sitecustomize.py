# The row-block confidence head's import hook (the `big` mode's `confhead` lever; the resident line's ENTRY hook: first on PYTHONPATH,
# executed by openfold3_opt's activation or, on the PYTHONPATH route, by the interpreter at start-up). After
# `openfold3.core.model.heads.prediction_heads` executes -> openfold3_opt.confhead.install() when OPENFOLD3_OPT_CONFHEAD=1 (the lever's
# switch; the resolver exports it at and above modes.CONF_MIN_TOKENS tokens and leaves it out below — the hook then installs nothing).
# OPENFOLD3_OPT_CONFHEAD_CHAIN=<dir> names the next hook directory (the offload port's of3o/) whose sitecustomize.py is executed next — the
# chain confhead > offload > fast_inference, each link exported by the package's resolver (modes.CHAIN_ENV). A failed hook raises (always strict).
import importlib.abc
import importlib.util
import os
import sys

_TARGET = "openfold3.core.model.heads.prediction_heads"
_ON = os.environ.get("OPENFOLD3_OPT_CONFHEAD", "").strip() == "1"


class _PostImportFinder(importlib.abc.MetaPathFinder):
    _target = _TARGET                     # read by openfold3_opt.hooks.installed()
    _busy = set()

    def find_spec(self, name, path=None, target=None):
        if name != _TARGET or name in self._busy:
            return None
        self._busy.add(name)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            self._busy.discard(name)
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module, _orig=orig_exec):
            _orig(module)
            from openfold3_opt import confhead
            confhead.install()

        spec.loader.exec_module = exec_module
        return spec


if _ON:
    sys.meta_path.insert(0, _PostImportFinder())

_chain = os.environ.get("OPENFOLD3_OPT_CONFHEAD_CHAIN", "")
if _chain:
    _hook = os.path.join(_chain, "sitecustomize.py")
    if not os.path.isfile(_hook):
        raise RuntimeError(f"[openfold3-opt/confhead] OPENFOLD3_OPT_CONFHEAD_CHAIN={_chain!r}: no sitecustomize.py there")
    if _chain not in sys.path:
        sys.path.insert(1, _chain)
    exec(compile(open(_hook).read(), _hook, "exec"), {"__name__": "openfold3_opt_hook_offload", "__file__": _hook})