"""Permissive stub so that the ESM cookbook binder_design.py can be imported where `modal` is not installed (local path never uses it)."""
class _Dummy:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k):
        if len(a) == 1 and not k and (callable(a[0]) or isinstance(a[0], type)):
            return a[0]                      # decorator use: @app.cls(...)(cls) / @modal.enter()(fn) -> unchanged object
        return _Dummy()
    def __getattr__(self, name): return _Dummy()
    def __getitem__(self, k): return _Dummy()
    def __iter__(self): return iter([])
    def __bool__(self): return False
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __repr__(self): return "<modal stub>"
def __getattr__(name):
    return _Dummy()
