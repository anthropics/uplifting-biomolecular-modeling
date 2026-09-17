"""
P2 `fastload` — Boltz2 weight load for mosaic without the torch checkpoint read + tensor copy in every process.

save_fast(model, dir): (1) eqx.tree_serialise_leaves of the joltz module; (2) the checkpoint's hyper_parameters (tiny) pickled, so a later process can
                       build the torch module SKELETON (random init, no state_dict load) with identical structure, convert it with joltz.from_torch
                       (structure only matters), and then overwrite every array leaf from the .eqx file.
load_fast(dir):        -> (mosaic Boltz2 instance, info). The parameter fingerprint (`fingerprint()`) must equal the torch-loaded model's; the caller checks.
"""
import time, pickle, os, hashlib
from pathlib import Path
from dataclasses import asdict
import numpy as np, jax, jax.numpy as jnp, equinox as eqx

def _paths(d):
    d = Path(d); return d / "joltz2_leaves.eqx", d / "boltz2_hparams.pkl"

def save_fast(model, d, checkpoint_path=None):
    import torch
    from mosaic.cache import resolve_cache
    leaves_p, hp_p = _paths(d); Path(d).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    eqx.tree_serialise_leaves(str(leaves_p) + ".tmp", model.model); os.replace(str(leaves_p) + ".tmp", leaves_p)
    out = {"t_serialise_s": time.time() - t0, "bytes": leaves_p.stat().st_size}
    ckpt = resolve_cache(checkpoint_path, "boltz", "boltz2_conf.ckpt")
    t0 = time.time()
    # hyper_parameters only (torch.load still reads the zip index; use weights_only=False since hparams contain plain dicts)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    hp = dict(ck["hyper_parameters"]); del ck
    with open(str(hp_p) + ".tmp", "wb") as f: pickle.dump(hp, f, protocol=5)
    os.replace(str(hp_p) + ".tmp", hp_p); out["t_hparams_s"] = time.time() - t0; out["hparams_keys"] = sorted(hp.keys())[:60]
    return out

import contextlib
@contextlib.contextmanager
def no_random_init():
    """boltz's module constructors spend most of the construction time in initialize.trunc_normal_init_ (scipy truncnorm) + torch init fills for weights that are then
    overwritten by the checkpoint (stock path) or by the .eqx leaves (fast path). Make those fills no-ops for the duration of construction.
    Exactness: every parameter/buffer is subsequently overwritten (stock: strict=True state_dict load; fast: all-leaves deserialise); `fingerprint()` hashes every leaf for the caller's check."""
    import torch.nn.init as init
    patched = {}
    try:
        import boltz.model.layers.initialize as binit
        for name in [n for n in dir(binit) if n.endswith("_init_")]:
            patched[(binit, name)] = getattr(binit, name); setattr(binit, name, (lambda w, *a, **k: w))
    except Exception:
        pass
    for name in ["kaiming_uniform_", "kaiming_normal_", "xavier_uniform_", "xavier_normal_", "uniform_", "normal_", "trunc_normal_", "orthogonal_"]:
        if hasattr(init, name): patched[(init, name)] = getattr(init, name); setattr(init, name, (lambda t, *a, **k: t))
    # modules that imported the init functions by name (from boltz...initialize import lecun_normal_init_ etc.)
    import sys
    byname = {}
    for modname, mod in list(sys.modules.items()):
        if modname.startswith("boltz.") and mod is not None:
            for name in [n for n in dir(mod) if n.endswith("_init_")]:
                f = getattr(mod, name)
                if callable(f): byname[(mod, name)] = f; setattr(mod, name, (lambda w, *a, **k: w))
    try:
        yield
    finally:
        for (mod, name), f in list(patched.items()) + list(byname.items()): setattr(mod, name, f)

def load_stock_fast_init(checkpoint_path=None):
    """STOCK load path (torch ckpt -> joltz.from_torch, identical to mosaic.losses.boltz2.load_boltz2) but with random init disabled during construction."""
    import mosaic.losses.boltz2 as mlb2
    from mosaic.models.boltz2 import Boltz2
    # import boltz submodules first so no_random_init can patch by-name imports
    import boltz.model.models.boltz2  # noqa
    with no_random_init():
        jm = mlb2.load_boltz2(checkpoint_path) if checkpoint_path else mlb2.load_boltz2()
    m = Boltz2.__new__(Boltz2); object.__setattr__(m, "model", jm); return m

def build_skeleton(hp):
    """Construct boltz.model.models.boltz2.Boltz2 with the checkpoint's hyper-parameters (+ the same overrides mosaic passes) WITHOUT loading weights, then joltz.from_torch."""
    import torch, joltz
    import boltz.main as boltz_main
    import boltz.model.models.boltz2  # noqa: ensure submodules imported before patching
    from boltz.model.models.boltz2 import Boltz2 as TorchBoltz2
    kw = dict(hp)
    kw.update(dict(predict_args={"recycling_steps": 0, "sampling_steps": 25, "diffusion_samples": 1},
                   diffusion_process_args=asdict(boltz_main.Boltz2DiffusionParams()),
                   msa_args=asdict(boltz_main.MSAModuleArgs(subsample_msa=True, num_subsampled_msa=1024, use_paired_feature=True)),
                   pairformer_args=asdict(boltz_main.PairformerArgsV2())))
    # mosaic.losses.boltz2.load_boltz2 passes exactly these overrides to load_from_checkpoint (see source); everything else comes from hparams
    import inspect
    sig = inspect.signature(TorchBoltz2.__init__).parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.values())
    dropped = [k for k in list(kw) if (k not in sig and not accepts_var_kw)]
    for k in dropped: kw.pop(k)          # lightning's load_from_checkpoint does the same filtering (_sanitize/ignored hparams)
    build_skeleton.dropped_hparams = dropped
    with torch.no_grad(), no_random_init():
        tm = TorchBoltz2(**kw)
    tm.eval()
    return joltz.from_torch(tm)

def load_fast(d):
    from mosaic.models.boltz2 import Boltz2
    leaves_p, hp_p = _paths(d)
    info = {}; t0 = time.time()
    hp = pickle.load(open(hp_p, "rb")); like = build_skeleton(hp); info["t_skeleton_s"] = time.time() - t0
    t1 = time.time(); jm = eqx.tree_deserialise_leaves(str(leaves_p), like); info["t_deserialise_s"] = time.time() - t1
    m = Boltz2.__new__(Boltz2); object.__setattr__(m, "model", jm)
    info["t_total_s"] = time.time() - t0; info["leaves_sha256_16"] = None
    return m, info

def fingerprint(model):
    leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(eqx.filter(model.model, eqx.is_array))]
    h = hashlib.sha256()
    for l in leaves: h.update(np.ascontiguousarray(l).tobytes())
    return {"n_leaves": len(leaves), "n_params": int(sum(l.size for l in leaves)), "abs_sum": float(sum(np.abs(l.astype(np.float64)).sum() for l in leaves)), "all_leaves_sha256": h.hexdigest()}
