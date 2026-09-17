"""Test support: the CPU stand-in of `opt_core.capture.graphs.GraphCache` for opendde_opt/stepgraph.py's unit tests (no CUDA): 'capture' =
static copies of the arguments + the call on them; 'replay' = copy-in + the same call on the static copies; `poison()` = the probe on the
stand-in. Same control flow and the same aliasing discipline as the graph path (the callable only ever sees the static buffers). The tests
install it with `monkeypatch.setattr(stepgraph, "_make_cache", ...)`."""
from typing import Any, Dict



class EagerTestCache:
    """The GraphCache surface this module uses, without CUDA (unit tests): 'capture' = static copies of the arguments + the call on them; 'replay' =
    copy-in + the same call on the static copies. Same control flow and the same aliasing discipline (the callable only sees the static buffers)."""

    def __init__(self, name, **kw):
        self.name, self.kw = name, kw
        self.stats_ = {"captures": 0, "replays": 0, "capture_s": 0.0, "verify_pass": 0, "failures": 0, "pool_bytes": 0}
        self._entries: Dict[Any, Any] = {}
        self.disabled_ = None

    def run(self, fn, *args):
        if self.disabled_ is not None:
            return fn(*args)
        key = tuple((tuple(a.shape), str(a.dtype)) for a in args)
        ent = self._entries.get(key)
        if ent is None:
            ref = fn(*args)                                                    # eager-first: the admitting call's answer
            static = [a.clone() for a in args]
            for _ in range(int(self.kw.get("warmup", 1))):
                fn(*static)
            out = fn(*static)                                              # the 'capture'
            if not (out.shape == ref.shape and bool((out == ref).all())):
                self.disabled_ = "hold"
                self.stats_["failures"] += 1
                return ref
            self._entries[key] = (static, fn, out)
            self.stats_["captures"] += 1
            self.stats_["verify_pass"] += 1
            return ref
        static, f0, _out = ent
        for s, a in zip(static, args):
            s.copy_(a)
        self.stats_["replays"] += 1
        out = f0(*static)                                                 # the 'replay' (the graph's kernels again)
        self._entries[key] = (static, f0, out)
        return out.clone()

    def poison(self, key=None):
        """The probe on the stand-in: NaN into the static copies, the call re-run, restore."""
        import torch
        if not self._entries:
            return None
        static, f0, _ = next(iter(self._entries.values()))
        saved = [s.clone() for s in static]
        static[0].fill_(float("nan"))
        out = f0(*static)
        moved = bool(torch.isnan(out).any())
        for s, v in zip(static, saved):
            s.copy_(v)
        return moved

    def disable(self, reason):
        self.disabled_ = str(reason)

    @property
    def disabled(self):
        return self.disabled_

    def reset(self, reason="reset"):
        self._entries.clear()

    def stats(self):
        d = dict(self.stats_)
        d.update(disabled=self.disabled_, per_key=[], keys=len(self._entries))
        return d

