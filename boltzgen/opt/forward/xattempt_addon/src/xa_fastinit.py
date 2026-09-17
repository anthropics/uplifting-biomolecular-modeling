
# xa_fastinit.py — `fastinit` for BoltzGen 0.3.2 (exact; removes a fixed cost per pipeline step / per process).
#
# What stock does: `Predict.run()` builds the model with `Boltz.load_from_checkpoint(ckpt, strict=True, map_location="cpu")`.
# Construction runs every layer's random initialiser first — in particular `trunc_normal_init_` (scipy `truncnorm.rvs`, CPU, one
# call per Linear in the pairformer / diffusion transformer / atom encoders; boltzgen/model/layers/initialize.py and the copy in
# model/layers/triangular_attention/primitives.py) — and only then overwrites EVERY parameter from the checkpoint's state_dict
# (strict=True => a parameter missing from the checkpoint raises, so no initial value can survive into inference).
# The cost that matters is the design-step
# model's construction; the inverse-folding model is small.
#
# What this lever does: while a strict=True `load_from_checkpoint` is executing, `trunc_normal_init_` returns immediately (the tensor
# keeps the values nn.Linear's own constructor gave it, which are then overwritten by the checkpoint exactly as before).
# Nothing reachable from inference changes: same parameters (bitwise), same kernels.
#
# RNG contract (why outputs stay byte-identical): trunc_normal_init_ draws from scipy/numpy's GLOBAL RandomState. In the stock pipeline
# the only later consumers of that global stream are (a) DataLoader worker seeding — pl.seed_everything(workers=True) derives worker
# seeds from the base seed + worker id + rank, NOT from the global stream — and (b) the featurizer's `np.random.default_rng(None)`,
# which the seeding fix (bg_hook.py) seeds from the seeded numpy stream's state INSIDE the DataLoader worker process (fresh seeded state)
# when num_workers > 0 (upstream's default is 1). With the DataLoader in the step's own process — num_workers == 0, or the step's `debug`
# flag (upstream's Predict.run then sets num_workers 0 itself) — the featurizer would draw from the main-process stream after
# construction, so the lever cannot serve such a step: `unserved_reason(task)` says why, and the boltzgen_opt package refuses the step
# by name before its model loads (its step gate is the one reader; this module does not decide a process's exit).
# A load whose checkpoint does not cover every skipped initialiser replays the stock initialisers in order (COVERAGE INCOMPLETE /
# coverage check failed): correct but not the lever — STATS["fallback_replayed"] / ["fallback_reason"] record it by name.
# torch's RNG is not touched by trunc_normal_init_ at all (nn.Linear's kaiming init runs in both cases).
#
# Env: XA_FAST_INIT=1 (default on when this module is imported) / 0 = off. Prints one line per load with the number of skipped inits.
import os, sys, time
STATS = {"enabled": os.environ.get("XA_FAST_INIT", "1") == "1", "active": False, "skipped": 0, "numel": 0, "loads": [],
         "fallback_replayed": 0, "fallback_reason": None}   # fallback_*: stock initialisers replayed for a load the checkpoint did not provably cover (the lever did not serve that load)


def _log(msg):
    print("[xa_fastinit] " + msg, file=sys.stderr, flush=True)


RNG_SHIFT = "the featurizer's RNG stream would shift from stock's (fast-init skips the initialisers' draws on numpy's global stream, which an in-process DataLoader then reads)"


def unserved_reason(task):
    """Why fast-init cannot serve upstream's model step `task` (a boltzgen Predict about to run), None when it can: a step whose DataLoader
    runs in the step's own process — its `debug` flag (upstream's Predict.run sets num_workers 0 itself), num_workers
    0, or a worker count that cannot be read — draws the featurizer's random numbers from the main-process numpy stream, whose position
    fast-init changes: the RNG contract above. The one reader is the caller that refuses the step before its model loads
    (boltzgen_opt's step gate, stack.step_refusals); the sentence stands alone on its NOT ACTIVE line."""
    if not STATS.get("enabled"):
        return None
    try:
        if getattr(task, "debug", False):
            return "in-process DataLoader (debug=true: upstream sets num_workers 0 inside Predict.run): " + RNG_SHIFT
        data = getattr(task, "data", None)
        nw = getattr(data, "num_workers", None)
        if nw is None and hasattr(data, "cfg"):                            # FromGeneratedDataModule (inverse folding / folding steps) keeps it in .cfg
            nw = getattr(data.cfg, "num_workers", None)
        if nw is None or int(nw) == 0:
            return "in-process DataLoader (num_workers=%s): %s" % (nw, RNG_SHIFT)
        return None
    except Exception as e:
        return "DataLoader worker count unreadable (%r): %s" % (e, RNG_SHIFT)


def _replayed(n, reason):
    """Record a load the lever did not serve: `n` stock initialisers replayed in order, and why."""
    STATS["fallback_replayed"] = int(STATS.get("fallback_replayed") or 0) + int(n)
    STATS["fallback_reason"] = reason


if STATS["enabled"]:
    try:
        import boltzgen.model.layers.initialize as _I
        import boltzgen.model.layers.triangular_attention.primitives as _P
        import pytorch_lightning as pl
        from boltzgen.model.models.boltz import Boltz

        import torch
        _DEFERRED = []   # (orig_fn, weights_tensor, args, kwargs) in stock call order, for the coverage fallback

        def _mk(orig):
            def trunc_normal_init_(weights, *a, **kw):
                if STATS["active"]:
                    STATS["skipped"] += 1; STATS["numel"] += int(weights.numel())
                    _DEFERRED.append((orig, weights, a, kw))
                    return None
                return orig(weights, *a, **kw)
            trunc_normal_init_._xa_orig = orig
            return trunc_normal_init_
        _I.trunc_normal_init_ = _mk(_I.trunc_normal_init_)
        _P.trunc_normal_init_ = _mk(_P.trunc_normal_init_)

        _desc = None
        for klass in Boltz.__mro__:
            if "load_from_checkpoint" in klass.__dict__:
                _desc = klass.__dict__["load_from_checkpoint"]; break

        # Coverage check: the skip is only sound if EVERY tensor whose init we skipped is overwritten by the checkpoint.
        # We check it inside Boltz.on_load_checkpoint (runs after construction, before load_state_dict, with the final key-renamed
        # state_dict): every model parameter AND buffer key must be present in checkpoint['state_dict'] with an equal shape. If any key is
        # missing/mismatched, we replay ALL deferred stock inits in their original order (same scipy global-RNG consumption as stock =>
        # identical values), i.e. the lever turns itself into a no-op for that load, and says so.
        _orig_olc = Boltz.on_load_checkpoint
        def on_load_checkpoint(self, checkpoint):
            r = _orig_olc(self, checkpoint)
            if STATS["active"]:
                try:
                    sd = checkpoint["state_dict"]
                    missing = []; mismatched = []
                    for name, t in list(self.named_parameters()) + list(self.named_buffers()):
                        if name not in sd:
                            missing.append(name)
                        elif tuple(sd[name].shape) != tuple(t.shape):
                            mismatched.append(name)
                    # buffers that are legitimately absent from checkpoints (non-persistent) are not created by trunc_normal_init_,
                    # but we stay strict: any absent PARAMETER => fallback; absent buffers are only reported.
                    param_names = {n for n, _ in self.named_parameters()}
                    missing_params = [m for m in missing if m in param_names]
                    STATS["coverage"] = {"n_params": len(param_names), "n_state_dict": len(sd), "missing_params": missing_params[:20],
                                         "n_missing_params": len(missing_params), "n_missing_buffers": len(missing) - len(missing_params),
                                         "mismatched": mismatched[:20], "n_deferred": len(_DEFERRED)}
                    if missing_params or mismatched:
                        n = 0
                        for orig, w, a, kw in _DEFERRED:
                            orig(w, *a, **kw); n += 1
                        _replayed(n, "COVERAGE INCOMPLETE (%d missing params, %d shape mismatches): replayed %d stock inits in order; the lever is a no-op for this load"
                                  % (len(missing_params), len(mismatched), n))
                        _log("COVERAGE INCOMPLETE (%d missing params, %d shape mismatches) -> replayed %d stock inits in order; lever is a no-op for this load"
                             % (len(missing_params), len(mismatched), n))
                    else:
                        _log("coverage OK: all %d parameters (+%d persistent buffers) present in checkpoint state_dict with equal shapes; %d trunc_normal inits skipped"
                             % (len(param_names), len(list(self.named_buffers())) - (len(missing) - len(missing_params)), len(_DEFERRED)))
                except Exception as e:
                    # cannot prove coverage -> replay (safe)
                    n = 0
                    for orig, w, a, kw in _DEFERRED:
                        orig(w, *a, **kw); n += 1
                    _replayed(n, "coverage check failed (%r): replayed %d stock inits" % (e, n))
                    _log("coverage check failed (%r) -> replayed %d stock inits" % (e, n))
                finally:
                    _DEFERRED.clear()
            return r
        Boltz.on_load_checkpoint = on_load_checkpoint

        def _load_from_checkpoint(cls, *a, **kw):
            strict = kw.get("strict", None)
            use = STATS["enabled"] and strict is True
            STATS["active"] = use; _DEFERRED.clear()
            n0, t0 = STATS["skipped"], time.time()
            try:
                model = _desc.__get__(None, cls)(*a, **kw)
            finally:
                STATS["active"] = False
                if _DEFERRED:   # on_load_checkpoint never ran (unexpected path) -> replay to be safe
                    for orig, w, aa, kk in _DEFERRED: orig(w, *aa, **kk)
                    _replayed(len(_DEFERRED), "deferred inits replayed outside on_load_checkpoint (%d)" % len(_DEFERRED))
                    _log("deferred inits replayed outside on_load_checkpoint (%d)" % len(_DEFERRED)); _DEFERRED.clear()
            rec = {"strict": strict, "fast": use, "skipped_inits": STATS["skipped"] - n0, "load_s": round(time.time() - t0, 3), "coverage": STATS.get("coverage")}
            STATS["loads"].append(rec)
            _log("load_from_checkpoint strict=%s fast=%s skipped_trunc_normal_inits=%d load_s=%.2f" % (strict, use, rec["skipped_inits"], rec["load_s"]))
            return model
        Boltz.load_from_checkpoint = classmethod(_load_from_checkpoint)
    except Exception as e:
        _log("not installed: %r" % e); STATS["enabled"] = False
