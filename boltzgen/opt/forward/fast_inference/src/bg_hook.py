
# bg_hook.py — imported via sitecustomize in every python process (incl. DataLoader workers) of a kit-mode BoltzGen run.
# (1) seeding fix: the featurizer's `np.random.default_rng(None)` derives its seed from the seeded global numpy RNG's state
#     (stock draws an OS-entropy seed there -> ref_pos differs run to run even under pl.seed_everything). Function-preserving.
# (2) graph sampler switch: BG_GRAPH=graph|predraw applies bg_graph_patch (design step only, gated at call time).
# (3) timing (CUDA-synced) per pipeline step + per design, checkpoint-load breakdown, cuEquivariance call counters — appended as one JSON line per process to BG_TIMING_FILE at exit (nothing is written when it is unset).
import os, time, json, atexit, threading
_f = os.environ.get("BG_TIMING_FILE")
_lock = threading.Lock(); _acc = {}; _records = []; _cueq = {"triangle_multiplicative_update": 0, "triangle_attention": 0}
def _step(): return os.environ.get("BOLTZGEN_PIPELINE_STEP", "")
def _rec(k, dt):
    with _lock:
        a = _acc.setdefault(k, [0.0, 0]); a[0] += dt; a[1] += 1
def _flush():
    if not _f: return
    extra = {}
    try:
        import bg_graph_patch as g; extra = {k: v for k, v in g.STATS.items() if k != "last"}; extra["last"] = g.STATS.get("last")
    except Exception: pass
    with open(_f, "a") as fh:
        fh.write(json.dumps({"pid": os.getpid(), "step": _step(), "seed": None, "t_end": time.time(), "acc": _acc, "records": _records,
                             "cueq_calls": _cueq, "graph_stats": extra}) + "\n")
try:
    import numpy as np, random as _random, torch
    _orig_default_rng = np.random.default_rng
    _seed_calls = {}
    def _seeded_default_rng(seed=None, *a, **kw):
        if seed is None:
            # The seed is derived from the seeded global numpy stream's STATE, never by a draw from it -- a draw here would advance the
            # stream the data modules read next (the binder-length `np.random.randint` draw in boltzgen/data/parse/schema.py), so designs
            # r>=1 would differ from stock. Recipe: MT19937 key + position, hashed with the call's ordinal -- deterministic under the
            # seeded stream, distinct per call, the stream untouched.
            # The ordinal counts the seedless calls of THIS process under THIS stream key (keyed by (pid, key)): a forked DataLoader
            # worker -- where upstream's seeded featurizer runs -- counts from its own first call, and a re-seed of the stream (the seed
            # line of the next pipeline step in this same process; torch's and Lightning's per-worker seeding) opens a fresh count.
            # Seedless calls made before any seed line (library imports below construct generators of their own) and the parent's
            # count inherited across the fork therefore never shift the seeds a step's workers derive: those are a function of the
            # step's seed, the worker and the order of its items alone. `--mode off` installs the same derivation in its stock child
            # (boltzgen_opt.stock_design.seed_default_rng), so every route hands the featurizer the same generators at one --seed.
            import hashlib
            _st = np.random.get_state()
            _key = np.asarray(_st[1], dtype=np.uint32).tobytes()
            _k = (os.getpid(), hashlib.sha256(_key).digest()[:8])
            _seed_calls[_k] = _seed_calls.get(_k, 0) + 1
            seed = int.from_bytes(hashlib.sha256(_key + int(_st[2]).to_bytes(4, "little")
                                                 + _seed_calls[_k].to_bytes(8, "little")).digest()[:4], "little") & (2**31 - 1)
        return _orig_default_rng(seed, *a, **kw)
    np.random.default_rng = _seeded_default_rng
    _rec("seed_fix_active", 1.0)
    if os.environ.get("BG_GRAPH", "off") != "off":
        import bg_graph_patch
        bg_graph_patch.apply(os.environ["BG_GRAPH"])
    try:
        import cuequivariance_torch.primitives.triangle as _tri
        for _nm in list(_cueq.keys()):
            if hasattr(_tri, _nm):
                def _mk(orig, nm):
                    def w(*a, **kw):
                        _cueq[nm] += 1; return orig(*a, **kw)
                    return w
                setattr(_tri, _nm, _mk(getattr(_tri, _nm), _nm))
        _rec("cueq_import_ok", 1.0)
    except Exception as e:
        _rec("cueq_import_failed:" + repr(e)[:120], 0.0)
    def _sync():
        try:
            if torch.cuda.is_available(): torch.cuda.synchronize()
        except Exception: pass
    def _wrap(obj, name, key):
        fn = getattr(obj, name)
        def w(*a, **kw):
            _sync(); t0 = time.perf_counter()
            try: return fn(*a, **kw)
            finally: _sync(); _rec(key, time.perf_counter() - t0)
        setattr(obj, name, w)
    _last_sample = {"t": None}
    import pytorch_lightning as pl
    from boltzgen.model.models import boltz as _bz
    # checkpoint-load breakdown: torch.load, load_state_dict and load_from_checkpoint are timed (construction time is derived, see NOTE)
    _orig_tload = torch.load
    def _tload(*a, **kw):
        t0 = time.perf_counter()
        try: return _orig_tload(*a, **kw)
        finally: _rec(_step() + ".torch.load", time.perf_counter() - t0)
    torch.load = _tload
    # NOTE: Boltz.__init__ is deliberately NOT wrapped: pytorch_lightning's _load_state filters checkpoint hyper-parameters by
    # inspect.getfullargspec(cls.__init__); a (*a, **kw) wrapper disables that filter (-> TypeError: unexpected kwarg 'chain_sampling_args').
    # Construction time is derived as load_from_checkpoint - torch.load - load_state_dict.
    _orig_lsd = _bz.Boltz.load_state_dict
    def _lsd(self, *a, **kw):
        t0 = time.perf_counter()
        try: return _orig_lsd(self, *a, **kw)
        finally: _rec(_step() + ".load_state_dict", time.perf_counter() - t0)
    _bz.Boltz.load_state_dict = _lsd
    _orig_load = _bz.Boltz.load_from_checkpoint
    def _load(*a, **kw):
        t0 = time.perf_counter()
        try: return _orig_load(*a, **kw)
        finally: _rec(_step() + ".load_from_checkpoint", time.perf_counter() - t0); _rec(_step() + ".t_model_loaded_epoch", time.time())
    _bz.Boltz.load_from_checkpoint = staticmethod(_load)
    _orig_predict = pl.Trainer.predict
    def _predict(self, model=None, *a, **kw):
        st = _step(); m = model
        if m is not None:
            for attr, key in [("pairformer_module", "trunk.pairformer"), ("diffusion_conditioning", "diffusion.conditioning"), ("confidence_module", "confidence")]:
                sub = getattr(m, attr, None)
                if sub is not None and hasattr(sub, "forward"): _wrap(sub, "forward", st + "." + key)
            sm = getattr(m, "structure_module", None)
            if sm is not None and hasattr(sm, "sample"):
                fn = sm.sample
                def w(*aa, **kk):
                    _sync(); t0 = time.perf_counter()
                    try: return fn(*aa, **kk)
                    finally:
                        _sync(); dt = time.perf_counter() - t0; _rec(st + ".diffusion.sample", dt); _last_sample["t"] = dt
                sm.sample = w
            if hasattr(m, "load_checkpoint_weights"): _wrap(m, "load_checkpoint_weights", st + ".checkpoint_switch")
            _wrap(m, "forward", st + ".model.forward_total")
        _rec(st + ".t_predict_start_epoch", time.time()); _sync(); t0 = time.perf_counter()
        try: return _orig_predict(self, model, *a, **kw)
        finally:
            _sync(); _rec(st + ".trainer.predict_total", time.perf_counter() - t0); _rec(st + ".t_predict_end_epoch", time.time())
            try:
                _rec(st + ".gpu.max_memory_allocated_GB", torch.cuda.max_memory_allocated() / 1e9); _rec(st + ".gpu.max_memory_reserved_GB", torch.cuda.max_memory_reserved() / 1e9)
            except Exception: pass
    pl.Trainer.predict = _predict
    _orig_ps = _bz.Boltz.predict_step
    _n = {"i": 0}
    def _ps(self, batch, *a, **kw):
        i = _n["i"]; _n["i"] += 1; _last_sample["t"] = None; st = _step()
        _sync(); t0 = time.perf_counter()
        out = _orig_ps(self, batch, *a, **kw)
        _sync(); dt = time.perf_counter() - t0
        try:
            gl = None
            try:
                import bg_graph_patch as g; gl = g.STATS.get("last")
            except Exception: pass
            _records.append({"step": st, "i": i, "id": str(batch["id"][0]) if "id" in batch else None,
                             "dsi": int(batch["data_sample_idx"][0]) if "data_sample_idx" in batch else None,
                             "n_tokens": int(batch["token_pad_mask"].sum()) if "token_pad_mask" in batch else None,
                             "n_design": int(batch["design_mask"].sum()) if "design_mask" in batch else None,
                             "n_atoms": int(batch["atom_pad_mask"].sum()) if "atom_pad_mask" in batch else None,
                             "t_step": dt, "t_sample": _last_sample["t"],
                             "step_scale": getattr(self, "current_step_scale", None), "noise_scale": getattr(self, "current_noise_scale", None),
                             "ckpt_idx": getattr(self, "current_checkpoint_index", None), "max_mem_GB": torch.cuda.max_memory_allocated() / 1e9,
                             "t_end_epoch": time.time(), "graph_last": gl})
        except Exception as e:
            _records.append({"step": st, "i": i, "err": repr(e)[:200], "t_step": dt})
        return out
    _bz.Boltz.predict_step = _ps
    _rec("t_hook_import_epoch", time.time())
except Exception as e:
    _rec("hook_error:" + repr(e)[:200], 0.0)
atexit.register(_flush)
