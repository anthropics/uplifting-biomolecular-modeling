# The offload add-on's import hook (the `big` mode's entry hook on one GPU; executed by openfold3_ob0_opt's activation or, on the
# PYTHONPATH route, by the interpreter at start-up). After `openfold3.projects.of3_all_atom.model` executes -> of3_offload.apply_core();
# after `...runner` -> apply_runner(). A failed hook raises (always strict). OF3O_KIT_LEVERS=<dir> names the fast_inference add-on's
# of3_levers directory whose sitecustomize.py is executed next (the line's other levers: fast_init), exported by the package's resolver.
import importlib.abc
import importlib.util
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))

_HOOKS = {
    "openfold3.projects.of3_all_atom.model": "apply_core",
    "openfold3.projects.of3_all_atom.runner": "apply_runner",
}


class _PostImportFinder(importlib.abc.MetaPathFinder):
    _targets = frozenset(_HOOKS)
    _busy = set()

    def find_spec(self, name, path=None, target=None):
        if name not in _HOOKS or name in self._busy:
            return None
        self._busy.add(name)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            self._busy.discard(name)
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module, _orig=orig_exec, _name=name):
            _orig(module)
            if _here not in sys.path:
                sys.path.insert(0, _here)
            import of3_offload
            getattr(of3_offload, _HOOKS[_name])()

        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _PostImportFinder())

_chain = os.environ.get("OF3O_KIT_LEVERS", "")
if _chain:
    _hook = os.path.join(_chain, "sitecustomize.py")
    if not os.path.isfile(_hook):
        raise RuntimeError(f"[of3o] OF3O_KIT_LEVERS={_chain!r}: no sitecustomize.py there")
    if _chain not in sys.path:
        sys.path.insert(1, _chain)
    exec(compile(open(_hook).read(), _hook, "exec"), {"__name__": "openfold3_ob0_opt_hook_fast_inference", "__file__": _hook})
