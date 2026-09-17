"""Import hook that applies OF3 levers selected by env vars.
  OF3_FAST_INIT=1      -> of3_fastinit.enable() as soon as openfold3.core.model.primitives.linear is imported (exact)
  OF3_CUDA_GRAPHS=1    -> of3_graphs.enable() when openfold3.core.model.structure.diffusion_module is imported
"""
import os, sys, importlib

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

if os.environ.get("OF3_DETERMINISTIC") in ("1", "warn"):
    # deterministic REFERENCE configuration (testing aid, not a speed lever): cuBLAS workspace + torch deterministic algorithms
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
        torch.use_deterministic_algorithms(True, warn_only=(os.environ.get("OF3_DETERMINISTIC") == "warn"))
        sys.stderr.write(f"[of3_levers] deterministic algorithms ON (warn_only={os.environ.get('OF3_DETERMINISTIC') == 'warn'})\n")
    except Exception as e:
        sys.stderr.write(f"[of3_levers] deterministic setup failed: {e}\n")

class _LeverFinder:
    _targets = {"openfold3.core.model.primitives.linear", "openfold3.core.model.structure.diffusion_module"}
    _done = set()
    def find_spec(self, name, path=None, target=None):
        if name in self._targets and name not in self._done:
            self._done.add(name)
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
                def exec_module(module, _orig=orig_exec, _name=name):
                    _orig(module)
                    if _name == "openfold3.core.model.primitives.linear" and os.environ.get("OF3_FAST_INIT") == "1":
                        import of3_fastinit
                        of3_fastinit.enable()
                        sys.stderr.write("[of3_levers] fast-init enabled (weight init skipped; checkpoint load overwrites all params)\n")
                    if _name == "openfold3.core.model.structure.diffusion_module" and os.environ.get("OF3_CUDA_GRAPHS") == "1":
                        import of3_graphs
                        of3_graphs.enable()
                spec.loader.exec_module = exec_module
                return spec
        return None

if os.environ.get("OF3_FAST_INIT") == "1" or os.environ.get("OF3_CUDA_GRAPHS") == "1":
    sys.meta_path.insert(0, _LeverFinder())
