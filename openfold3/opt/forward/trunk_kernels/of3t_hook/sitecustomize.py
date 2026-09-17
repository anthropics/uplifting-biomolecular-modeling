# Part of OF3_TRUNK_KERNELS_ADDON (inference-speed add-on for OpenFold3); attributions: NOTICE.
"""of3t import hook: chains the fast-inference add-on's of3_levers sitecustomize (fast init / CUDA graphs / deterministic) if present on OF3T_KIT_LEVERS, then installs
of3t_levers (OF3T_* env) after openfold3.projects.of3_all_atom.model is imported."""
import os, sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

# 1) chain kit levers (their sitecustomize is shadowed by ours when both dirs are on PYTHONPATH)
_kit = os.environ.get("OF3T_KIT_LEVERS")
if _kit and os.path.isdir(_kit):
    if _kit not in sys.path:
        sys.path.insert(1, _kit)
    _ks = os.path.join(_kit, "sitecustomize.py")
    if os.path.exists(_ks):
        exec(compile(open(_ks).read(), _ks, "exec"), {"__name__": "of3_kit_sitecustomize", "__file__": _ks})

_WANT_LEVERS = any(os.environ.get(k) for k in ("OF3T_TRIATT", "OF3T_APB", "OF3T_TRIMUL", "OF3T_TEMPL_DISTINCT"))
_WANT_PAIRCACHE = os.environ.get("OF3T_PAIRCACHE") == "1"


def _install_model_timer(module):
    """print 'model forward <s>s (N tokens)' per predicted item to stderr (one cuda sync per item; numerically inert)"""
    import time
    try:
        import torch
        OF = module.OpenFold3; _orig = OF.forward
        def forward(self, batch):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            out = _orig(self, batch)
            torch.cuda.synchronize()
            try:
                n = int(batch["token_mask"].shape[-1])
            except Exception:
                n = -1
            sys.stderr.write(f"[of3t] model forward {time.perf_counter() - t0:.2f}s ({n} tokens)\n"); sys.stderr.flush()
            return out
        OF.forward = forward
    except Exception as e:
        sys.stderr.write(f"[of3t] model timer not installed: {e}\n")


class _Finder:
    _target = "openfold3.projects.of3_all_atom.model"
    _done = False
    def find_spec(self, name, path=None, target=None):
        if name != self._target or self._done:
            return None
        type(self)._done = True
        for finder in sys.meta_path:
            if finder is self:
                continue
            try:
                spec = finder.find_spec(name, path, target)
            except Exception:
                continue
            if spec is None or spec.loader is None:
                continue
            orig_exec = spec.loader.exec_module
            def exec_module(module, _orig=orig_exec):
                _orig(module)
                try:
                    if _WANT_LEVERS:
                        import of3t_levers; of3t_levers.install()
                    if _WANT_PAIRCACHE:
                        import of3t_paircache; of3t_paircache.install()
                    _install_model_timer(module)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    sys.stderr.write(f"[of3t] install failed: {e}\n")
                    raise
            spec.loader.exec_module = exec_module
            return spec
        return None


sys.meta_path.insert(0, _Finder())                 # the per-item model timer always; the levers and the pair cache when their switches are set
