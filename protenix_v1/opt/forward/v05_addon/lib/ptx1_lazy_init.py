"""ptx1_lazy_init — the `lazy_init` lever: skip the dead random initialisation of the model's parameters while the stock runner constructs
`Protenix(configs)`; the strict checkpoint load that follows overwrites every parameter and persistent buffer, so the random values are never
read (protenix 1.1.0: runner.inference.InferenceRunner.init_model() then .load_checkpoint(), load_state_dict(strict=True)).

No environment switch of its own: the arm word `lazy_init` installs it (protenix_v1_opt.stack, before the stock constructor runs — the one
lever that acts ahead of the model); the census is `report()` (read by levers_ptx1.describe); the equality check is a mode of `install()`:
`recheck` constructs the model twice in the process (stock init, then
lazy), loads the checkpoint into each exactly as the runner does, and compares sha256 of every state_dict tensor plus every non-persistent
buffer — N/N identical is the claim (EXACT class: the loaded model is the same bytes; nothing at predict time changes).

What is skipped: torch.nn.init's random initialisers (trunc_normal_, normal_, xavier_*, kaiming_*, uniform_) and protenix's own helpers
(trunc_normal_init_ — scipy truncnorm on the CPU, the expensive one — lecun/he/glorot/normal_init_, ipa_point_weights_init_) wherever a
protenix module imported them; zeros_/ones_/constant_ and the zero/one/gating/final helpers stay stock (cheap, and they may write tensors the
checkpoint does not cover). The patches hold ONLY while init_model() runs and are restored in a `finally`.
"""
import contextlib
import hashlib
import sys
import time

_STATE = {"installed": False, "mode": None, "runner": None, "lazy_construct_s": None, "patched": None, "stock_construct_s": None,
          "recheck": None, "error": None, "constructs": 0}
RANDOM_INITS = ("trunc_normal_", "normal_", "xavier_uniform_", "xavier_normal_", "kaiming_uniform_", "kaiming_normal_", "uniform_")
HELPER_INITS = ("trunc_normal_init_", "lecun_normal_init_", "he_normal_init_", "glorot_uniform_init_", "normal_init_", "ipa_point_weights_init_")
MODULE_PREFIXES = ("protenix", "openfold")


def _noop_ret(t, *a, **k):
    return t


def _noop_none(*a, **k):
    return None


def _collect_targets():
    """(module, attr, replacement) for every random init function reachable: torch.nn.init's and protenix's helpers at their import sites."""
    import torch.nn as nn
    targets = [(nn.init, name, _noop_ret) for name in RANDOM_INITS if hasattr(nn.init, name)]
    for modname, mod in list(sys.modules.items()):
        if mod is None or not modname.startswith(MODULE_PREFIXES):
            continue
        for a in HELPER_INITS:
            f = getattr(mod, a, None)
            if f is not None and callable(f):
                targets.append((mod, a, _noop_none))
    return targets


@contextlib.contextmanager
def skip_random_init():
    """The random initialisers no-op'd for the duration; yields the patched names; always restores."""
    targets = _collect_targets(); saved = []
    for mod, a, rep in targets:
        saved.append((mod, a, getattr(mod, a))); setattr(mod, a, rep)
    try:
        yield [f"{m.__name__}.{a}" for m, a, _ in targets]
    finally:
        for mod, a, orig in saved:
            setattr(mod, a, orig)


def state_hashes(model):
    """{name: sha256|shape|dtype} of every state_dict tensor and every non-persistent buffer of `model` (host copies)."""
    import torch
    def h(t):
        t = t.detach().to("cpu").contiguous()
        raw = t.numpy().tobytes() if t.dtype == torch.bool else t.view(torch.uint8).numpy().tobytes()
        return hashlib.sha256(raw).hexdigest() + f"|{tuple(t.shape)}|{t.dtype}"
    sd = model.state_dict(keep_vars=False)
    out = {k: h(v) for k, v in sd.items()}
    for k, v in model.named_buffers():
        if k not in sd:
            out["<nonpersistent>" + k] = h(v)
    return out


def install(runner_cls, mode="1", log=None):
    """Patch `runner_cls.init_model` (and `.load_checkpoint` under mode 'recheck') BEFORE the runner is constructed. mode: '1' = lazy
    construction; 'recheck' = stock construction + checkpoint load + hashes, then lazy construction, then (at the runner's own checkpoint load)
    the N/N comparison. One install per process; returns the state."""
    if _STATE["installed"]:
        return dict(_STATE)
    if mode not in ("1", "recheck"):
        raise ValueError("ptx1_lazy_init.install: mode %r (want '1' or 'recheck')" % (mode,))
    import protenix.model.protenix              # noqa: F401 — the model tree imported, so the helpers' import sites exist to patch
    import protenix.model.modules.primitives    # noqa: F401
    import protenix.model.triangular.layers     # noqa: F401
    say = log or (lambda s: print(s, flush=True))
    orig_init_model = runner_cls.init_model

    def init_model(self, *a, **kw):
        if mode == "recheck" and _STATE["stock_construct_s"] is None:
            t = time.time(); orig_init_model(self, *a, **kw); _STATE["stock_construct_s"] = round(time.time() - t, 2)
            t = time.time(); orig_load(self); ck = round(time.time() - t, 2)
            _STATE["recheck"] = {"stock_hashes": state_hashes(self.model), "stock_ckpt_s": ck}
            del self.model
            try:
                import torch; torch.cuda.empty_cache()
            except Exception:                      # noqa: BLE001
                pass
        t = time.time()
        with skip_random_init() as patched:
            orig_init_model(self, *a, **kw)
        _STATE.update(lazy_construct_s=round(time.time() - t, 2), patched=list(patched), constructs=_STATE["constructs"] + 1)
        say("[ptx1_lazy_init] model constructed with random init skipped in %.1fs (%d init functions no-op'd; the strict checkpoint load follows)"
            % (_STATE["lazy_construct_s"], len(patched)))
    init_model.__wrapped__ = orig_init_model
    runner_cls.init_model = init_model
    orig_load = runner_cls.load_checkpoint
    if mode == "recheck":
        def load_checkpoint(self, *a, **kw):
            orig_load(self, *a, **kw)
            rc = _STATE.get("recheck")
            if rc and "stock_hashes" in rc and "verdict" not in rc and _STATE["lazy_construct_s"] is not None:
                lazy = state_hashes(self.model); stock = rc.pop("stock_hashes")
                keys = sorted(set(stock) | set(lazy)); neq = [k for k in keys if stock.get(k) != lazy.get(k)]
                rc.update(n_tensors=len(keys), n_equal=len(keys) - len(neq), unequal=neq[:50], verdict="IDENTICAL" if not neq else "DIFFERENT")
                say("[ptx1_lazy_init] RECHECK after the checkpoint load: %d/%d tensors sha256-identical (parameters + buffers + non-persistent buffers) -> %s"
                    % (rc["n_equal"], rc["n_tensors"], rc["verdict"]))
        load_checkpoint.__wrapped__ = orig_load
        runner_cls.load_checkpoint = load_checkpoint
    _STATE.update(installed=True, mode=mode, runner=f"{runner_cls.__module__}.{runner_cls.__name__}")
    return dict(_STATE)


def report():
    """The census: {installed, mode, runner, lazy_construct_s, patched: count, patched_names, stock_construct_s, recheck: {...}|None, error, constructs}."""
    out = {k: v for k, v in _STATE.items() if k != "patched"}
    out["patched"] = len(_STATE["patched"] or ()); out["patched_names"] = list(_STATE["patched"] or ())
    if isinstance(out.get("recheck"), dict):
        out["recheck"] = {k: v for k, v in out["recheck"].items() if k != "stock_hashes"}
    return out
