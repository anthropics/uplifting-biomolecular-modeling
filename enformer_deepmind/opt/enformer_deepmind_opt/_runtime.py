"""enformer_deepmind_opt._runtime — the whole runtime: resolve the kit on this machine, hook the two entry points the Enformer SavedModel is
loaded through (``tensorflow_hub.load`` and ``tf.saved_model.load``), rewrite each loaded Enformer's prediction graph at its first prediction
(or eagerly through ``apply``), print the lines, undo on ``disable``.

Lines (stderr, one each, ``[enformer-deepmind-opt]`` first):
  ACTIVE     mode=exact gpu=<name>(smNN) kit=<version> levers=<...> build=<...> applied=deferred [notes=…]
  DRY        the same fields from ``check`` — nothing applied
  NOT ACTIVE mode=exact reason=<why>   — nothing applied, stock untouched
  APPLIED    model#<k> levers=… cost=<s>s (<what was rewritten>)
  STOCK      model#<k> <why this SavedModel runs its stock graph>   — said once per model
  STOCK-CALL model#<k> <why this kind of call runs the stock function> — a call under a gradient tape or inside a tf.function; said once per model
  REMOVED    model#<k>
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time

PREFIX = "[enformer-deepmind-opt]"
MODE = "exact"
OPT_HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))                  # enformer_deepmind/opt (the package is installed editable from it)
KIT_ROOT = os.path.dirname(OPT_HOME)                                                     # enformer_deepmind/
PINS_PATH = os.path.join(KIT_ROOT, "stock", "PINS.json")
TESTED_TF = "2.17.1"                                                 # the TensorFlow whose GPU arithmetic the ops reproduce; edm_ops.so loads only under the version it was built against (ops.load), so another version reaches this note only after a rebuild — then it is named, not refused
TESTED_CARDS = {(9, 0): "H100 / H200", (8, 0): "A100"}              # the devices the op library carries native code for; any other device the library can serve is named as untested
INPUT_LENGTH = 393_216                                               # the SavedModel's input window (its graph crops to the central 196,608 bp)
HEADS = ("human", "mouse")


class ActivationError(RuntimeError):
    """enable(strict=True) / the ENFORMER_DEEPMIND_OPT route: the kit cannot engage on this machine (the NOT ACTIVE line carries the reason)."""


_LOCK = threading.RLock()
_REPORT: dict | None = None
_HOOKS = {"installed": False, "hub_load": None, "saved_model_load": None, "tf_load_alias": None}
_MODELS: dict = {}                      # id(model object) -> record
_COUNTER = {"n": 0}
_ENABLING = False                       # True while enable() imports tensorflow itself (the autoload finder stands down)


# ----------------------------------------------------------------------------------------------------------------- lines
def log(line: str) -> str:
    print(line, file=sys.stderr, flush=True)
    return line


def line_of(rep: dict) -> str:
    if not rep.get("active"):
        return f"{PREFIX} NOT ACTIVE mode={MODE} reason={rep.get('reason') or 'unknown'}"
    g = rep.get("gpu") or {}
    gpu = f"{g['name']}(sm{g['sm'][0]}{g['sm'][1]})" if g.get("name") else "none"
    core = f"mode={MODE} gpu={gpu} kit={rep['kit_version']} levers={','.join(rep['levers'])} build={rep['build']}"
    head, tail = ("DRY", "") if rep.get("dry_run") else ("ACTIVE", " applied=deferred")
    notes = f" notes={'; '.join(rep['notes'])}" if rep.get("notes") else ""
    return f"{PREFIX} {head} {core}{tail}{notes}"


# ----------------------------------------------------------------------------------------------------------------- resolution
def _device() -> dict:
    """Device 0 by TensorFlow (importing tensorflow allocates nothing on the GPU): name and capability, or the reason there is none."""
    try:
        import tensorflow as tf
    except Exception as e:                                             # noqa: BLE001
        return {"name": None, "reason": f"tensorflow is not importable ({type(e).__name__}: {e})"}
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return {"name": None, "reason": "no CUDA device visible (tf.config.list_physical_devices('GPU') is empty)", "tensorflow": tf.__version__}
    det = tf.config.experimental.get_device_details(gpus[0])
    cc = det.get("compute_capability") or (0, 0)
    return {"name": det.get("device_name") or gpus[0].name, "sm": (int(cc[0]), int(cc[1])), "tensorflow": tf.__version__, "n_gpus": len(gpus)}


def _versions() -> dict:
    from importlib import metadata
    out = {}
    for dist in ("tensorflow", "tensorflow-hub", "numpy"):
        try:
            out[dist] = metadata.version(dist)
        except Exception:                                              # noqa: BLE001
            out[dist] = None
    return out


def pins() -> dict:
    with open(PINS_PATH, encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(dry_run: bool) -> dict:
    """Everything enable() decides before touching a model: device, kit files, the levers' requirements, stack notes. Never raises."""
    from . import _levers, __version__
    rep = {"mode": MODE, "active": False, "dry_run": dry_run, "reason": None, "notes": [], "gpu": None, "kit_version": __version__,
           "levers": list(_levers.LEVERS), "build": None, "stack": None, "package": OPT_HOME}
    dev = _device()
    rep["gpu"] = dev
    if not dev.get("name"):
        rep["reason"] = dev["reason"]
        return rep
    sm = tuple(dev["sm"])
    if not os.path.isfile(PINS_PATH):
        rep["reason"] = f"kit files missing: {PINS_PATH} (install the kit editable from its release directory: bash run.sh install)"
        return rep
    try:
        rep["build"] = _levers.build_label(sm)                        # the op library's build serving this device (ops.serves: native code, or its PTX); raises by name when the library is missing, is not the recorded build, does not load under this TensorFlow, or has no code for the device
    except Exception as e:                                             # noqa: BLE001
        rep["reason"] = f"{type(e).__name__}: {e}"
        return rep
    if sm not in TESTED_CARDS:
        rep["notes"].append(f"untested device — capability {sm[0]}.{sm[1]} is outside the tested set ({', '.join(f'{a}.{b} {n}' for (a, b), n in TESTED_CARDS.items())})")
    st = _versions()
    rep["stack"] = st
    if st.get("tensorflow") != TESTED_TF:
        rep["notes"].append(f"tensorflow {st.get('tensorflow')} differs from the tested {TESTED_TF} (the levers engage; byte-identity to stock was established on that build)")
    want_hub = pins().get("pins", {}).get("tensorflow-hub")
    if want_hub and st.get("tensorflow-hub") not in (None, want_hub):
        rep["notes"].append(f"tensorflow-hub {st['tensorflow-hub']} differs from the tested {want_hub}")
    rep["active"] = True
    return rep


# ----------------------------------------------------------------------------------------------------------------- public API
def check() -> dict:
    """The dry run: resolve and print the DRY / NOT ACTIVE line; apply nothing."""
    rep = resolve(dry_run=True)
    log(line_of(rep))
    return rep


def status() -> dict:
    rep = dict(_REPORT or {"mode": MODE, "active": False, "reason": "enable() has not been called"})
    rep["models"] = [{"index": r["index"], "applied": r["fn"] is not None, "stock_reason": r["stock_reason"], "calls": r["calls"], "stock_calls": r["stock_calls"]}
                     for r in _MODELS.values()]
    return rep


def enable(*, strict: bool = False, trigger: str | None = None) -> dict:
    """Engage the kit for this process: every Enformer SavedModel loaded through ``tensorflow_hub.load`` / ``tf.saved_model.load`` after this
    call predicts through the rewritten graph. Idempotent. Returns the activation report; with ``strict`` a refusal raises ActivationError
    after the NOT ACTIVE line."""
    global _REPORT, _ENABLING
    with _LOCK:
        if _REPORT is not None and _REPORT.get("active") and not _REPORT.get("dry_run"):
            return _REPORT
        _ENABLING = True
        try:
            rep = resolve(dry_run=False)
            rep["trigger"] = trigger or "enable()"
            if rep["active"]:
                try:
                    _install_hooks()
                except Exception as e:                                 # noqa: BLE001
                    rep.update(active=False, reason=f"cannot install the hooks ({type(e).__name__}: {e})")
        finally:
            _ENABLING = False
        _REPORT = rep
        log(line_of(rep))
        if not rep["active"] and strict:
            raise ActivationError(rep["reason"])
        return rep


def apply(obj, batch_sizes=()) -> dict:
    """Rewrite ``obj``'s prediction graph now (``obj`` = what ``hub.load`` / ``tf.saved_model.load`` returned, or its ``.model``) and run it
    once per size in ``batch_sizes`` on a host array (the documented kind of argument) so TensorFlow's own first-execution costs at those
    shapes and for that call path (graph optimization, cuDNN algorithm selection, allocation) are paid here rather than at the first
    prediction. Requires enable(); an object not loaded through the hooked entry points is hooked here."""
    if _REPORT is None or not _REPORT.get("active") or _REPORT.get("dry_run"):
        raise ActivationError("enformer_deepmind_opt.apply(): call enable() first (the kit is not active in this process)")
    import numpy as np
    with _LOCK:
        model = getattr(obj, "model", obj)
        rec = _MODELS.get(id(model)) or _hook_model(model)
        if rec is None:
            raise ActivationError("enformer_deepmind_opt.apply(): not the Enformer SavedModel (model.predict_on_batch taking one float32 (None, 393216, 4) tensor)")
        if rec["fn"] is None and rec["stock_reason"] is None:
            _build(rec)
        if rec["stock_reason"]:
            raise ActivationError(f"model#{rec['index']}: {rec['stock_reason']}")
        t0 = time.perf_counter()
        for b in batch_sizes:
            out = rec["fn"](np.zeros((int(b), INPUT_LENGTH, 4), np.float32))          # the documented kind of argument: a float32 array on the host
            _ = [v.shape for v in out.values()]
            rec["warmed"].add(int(b))
        _sync()
        return {"index": rec["index"], "levers": rec["levers"], "apply_cost_s": rec["cost_s"], "warm_cost_s": round(time.perf_counter() - t0, 2), "batch_sizes": sorted(rec["warmed"])}


def disable() -> dict:
    """Undo everything: each model gets its stock prediction function back and the library entry points are restored."""
    global _REPORT
    with _LOCK:
        removed = []
        for rec in list(_MODELS.values()):
            m = rec["model"]
            try:
                m.predict_on_batch = rec["stock_fn"]
            except Exception:                                          # noqa: BLE001
                pass
            removed.append(rec["index"])
            log(f"{PREFIX} REMOVED model#{rec['index']}")
        _MODELS.clear()
        _remove_hooks()
        _REPORT = None
        return {"removed": removed}


# ----------------------------------------------------------------------------------------------------------------- hooks
def _install_hooks() -> None:
    if _HOOKS["installed"]:
        return
    import tensorflow as tf
    _HOOKS["saved_model_load"] = tf.saved_model.load

    def load(export_dir, *a, **kw):
        return _after_load(_HOOKS["saved_model_load"](export_dir, *a, **kw), export_dir)
    load.__wrapped__ = _HOOKS["saved_model_load"]
    tf.saved_model.load = load
    try:
        import tensorflow_hub as hub
    except Exception:                                                  # noqa: BLE001  (tensorflow-hub absent: tf.saved_model.load is still served)
        hub = None
    if hub is not None:
        _HOOKS["hub_load"] = hub.load

        def hub_load(handle, *a, **kw):
            obj = _HOOKS["hub_load"](handle, *a, **kw)
            try:
                path = hub.resolve(handle)
            except Exception:                                          # noqa: BLE001
                path = None
            return _after_load(obj, path)
        hub_load.__wrapped__ = _HOOKS["hub_load"]
        hub.load = hub_load
    _HOOKS["installed"] = True


def _remove_hooks() -> None:
    if not _HOOKS["installed"]:
        return
    import tensorflow as tf
    tf.saved_model.load = _HOOKS["saved_model_load"]
    hub = sys.modules.get("tensorflow_hub")
    if hub is not None and _HOOKS["hub_load"] is not None:
        hub.load = _HOOKS["hub_load"]
    _HOOKS.update(installed=False, hub_load=None, saved_model_load=None)


def is_enformer(model) -> bool:
    """The Enformer SavedModel's network object: ``predict_on_batch`` with one concrete function taking one float32 (None, 393216, 4) tensor
    and returning the two heads."""
    import tensorflow as tf
    fn = getattr(model, "predict_on_batch", None)
    try:
        cfs = list(fn.concrete_functions)
    except Exception:                                                  # noqa: BLE001
        return False
    if len(cfs) != 1:
        return False
    args, kwargs = cfs[0].structured_input_signature
    specs = list(args) + list(kwargs.values())
    outs = cfs[0].structured_outputs
    return (len(specs) == 1 and isinstance(specs[0], tf.TensorSpec) and specs[0].dtype == tf.float32 and list(specs[0].shape)[1:] == [INPUT_LENGTH, 4]
            and isinstance(outs, dict) and set(outs) == set(HEADS))


def _after_load(obj, export_dir):
    """Something was loaded through a hooked entry point: the Enformer SavedModel gets its ``model.predict_on_batch`` served by the kit (the
    graph is rewritten at the first prediction); anything else is returned untouched."""
    model = getattr(obj, "model", None)
    if model is not None and id(model) not in _MODELS and is_enformer(model):
        with _LOCK:
            rec = _hook_model(model)
            if rec is not None:
                rec["export_dir"] = export_dir
    return obj


def _hook_model(model):
    if not is_enformer(model):
        return None
    _COUNTER["n"] += 1
    rec = {"index": _COUNTER["n"], "model": model, "stock_fn": model.predict_on_batch, "fn": None, "levers": [], "cost_s": None, "stats": None,
           "stock_reason": None, "said_stock": False, "said_stock_call": False, "calls": 0, "stock_calls": 0, "warmed": set(), "export_dir": None}
    _MODELS[id(model)] = rec

    def predict_on_batch(x, _rec=rec):
        return _predict(_rec, x)
    predict_on_batch.__wrapped__ = rec["stock_fn"]
    predict_on_batch.concrete_functions = rec["stock_fn"].concrete_functions          # what identifies the model stays visible
    model.predict_on_batch = predict_on_batch
    return rec


# ----------------------------------------------------------------------------------------------------------------- application
def _build(rec: dict) -> None:
    """Rewrite the model's prediction graph with every lever; on a graph that is not the released Enformer's, record why and serve stock."""
    from . import _graph, _levers, ops
    t0 = time.perf_counter()
    try:
        ops.load()
        cf = rec["stock_fn"].concrete_functions[0]
        fn, stats = _graph.rewrite(cf, {name: _levers.REWRITES[name] for name in _levers.LEVERS})
    except Exception as e:                                             # noqa: BLE001
        rec["stock_reason"] = f"the prediction graph is not the released Enformer's ({type(e).__name__}: {e}); this model runs its stock graph"
        if not rec["said_stock"]:
            rec["said_stock"] = True
            log(f"{PREFIX} STOCK model#{rec['index']} {rec['stock_reason']}")
        return
    rec.update(fn=fn, stats=stats, levers=list(_levers.LEVERS), cost_s=round(time.perf_counter() - t0, 2))
    note = _weights_note(rec)
    what = "; ".join(f"{k}: {v}" for k, v in stats.items())
    log(f"{PREFIX} APPLIED model#{rec['index']} levers={','.join(rec['levers'])} cost={rec['cost_s']}s (prediction graph rewritten — {what}){note}")


def _weights_note(rec: dict) -> str:
    """' weights=<name>' when the SavedModel's files are the released model's (stock/PINS.json digests); a differing or unknown SavedModel is
    served and named as untested."""
    d = rec.get("export_dir")
    try:
        want = pins()["weights"]["tfhub:deepmind/enformer/1"]["files"]
        if not d or not os.path.isdir(d):
            return " weights=unverified (the SavedModel's directory is not known to the kit: served, untested)"
        for rel, meta in want.items():
            p = os.path.join(d, rel)
            if not os.path.isfile(p) or sha256_file(p) != meta["sha256"]:
                return f" weights=untested (the SavedModel at {d} is not the released deepmind/enformer/1 by digest: served)"
        return " weights=deepmind/enformer/1"
    except Exception as e:                                             # noqa: BLE001
        return f" weights=unverified ({type(e).__name__})"


def _predict(rec: dict, x):
    """``model.predict_on_batch(x)``: a float32 (B, 393216, 4) batch in a plain (eager, no gradient tape) call runs the rewritten graph;
    anything else the stock function decides — a call under a ``tf.GradientTape`` or inside a ``tf.function`` runs stock, said once per model
    (the rewritten graph is a forward pass: its ops carry no gradients and it is built for eager calls)."""
    if not _conforms(x):
        rec["stock_calls"] += 1
        return rec["stock_fn"](x)
    why = _differentiating_or_tracing()
    if why is not None:
        rec["stock_calls"] += 1
        if not rec.get("said_stock_call"):
            rec["said_stock_call"] = True
            log(f"{PREFIX} STOCK-CALL model#{rec['index']} {why}: this call and every such call runs the stock function (the rewritten graph serves plain forward calls)")
        return rec["stock_fn"](x)
    with _LOCK:
        if rec["fn"] is None and rec["stock_reason"] is None:
            _build(rec)
    if rec["stock_reason"] is not None:
        rec["stock_calls"] += 1
        return rec["stock_fn"](x)
    rec["calls"] += 1
    return rec["fn"](x)



def _differentiating_or_tracing():
    """Why the current call cannot take the rewritten graph — a gradient tape is recording, or TensorFlow is tracing a ``tf.function`` —
    or None for a plain eager call."""
    import tensorflow as tf
    if not tf.executing_eagerly():
        return "called while TensorFlow traces a tf.function"
    try:
        from tensorflow.python.eager import record                    # the tape stack of this thread (TensorFlow 2.13+)
        recording = record.could_possibly_record()
    except ImportError:
        from tensorflow.python.eager import tape as record            # its name before that
        recording = record.could_possibly_record()
    return "a tf.GradientTape is recording" if recording else None

def _conforms(x) -> bool:
    try:
        shape = tuple(int(d) for d in x.shape)
        name = getattr(getattr(x, "dtype", None), "name", str(getattr(x, "dtype", "")))
    except Exception:                                                  # noqa: BLE001
        return False
    return len(shape) == 3 and shape[0] >= 1 and shape[1] == INPUT_LENGTH and shape[2] == 4 and name == "float32"


def _sync():
    try:
        import tensorflow as tf
        tf.test.experimental.sync_devices()
    except Exception:                                                  # noqa: BLE001
        pass
