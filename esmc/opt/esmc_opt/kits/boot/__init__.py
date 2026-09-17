"""ESMC kit ``boot`` — U1 (boot): the device-side weight loader, exact by construction.

The user's time on the ESMC route goes to boot before it goes to the forward: ``from_pretrained`` materialises every
checkpoint tensor through page faults on a file mapping plus one pageable host-to-device copy per tensor. This kit
replaces ONLY that materialisation — the file's bytes are read in large chunks into pinned host memory and copied
asynchronously into device tensors (``loader.py``) — and leaves every other step of the stock load path to the stock's
own code: key adaptation, layout normalisation (``published_to_native`` — on device tensors a ``torch.cat`` is a copy),
``load_state_dict(assign=True)``, the strictness accounting and its error text, buffer materialisation, ``eval``.

A second lever rides the same scope: ``skip_meta_init`` — the stock constructors initialise every parameter on the META device
(accelerate's ``init_empty_weights`` / the fork's meta build) before the checkpoint overwrites all of them; on meta the write has
no value, but torch routes it through ``torch._refs`` whose first call imports ``torch._dynamo``. The patch returns META tensors
untouched (counted) and passes real tensors to the stock writer (counted; zero are expected on the served route).

One route, one mechanism, applied by a SCOPED patch of the one module attribute the route reads its tensors through
(restored in ``finally``; never a file edit; counted):

    from_pretrained(name, **kw)      ``esm.models.hub.read_safetensors_dir`` -> loader.load_files (cast to ``kw['dtype']`` per tensor) on the stock's default
                                     device (``torch.get_default_device()`` inside the stock's ``with torch.device(device)``)
                                     for a CUDA target

A CPU target, an absent checkpoint, a non-``pt`` framework: the stock function runs (counted as a passthrough — the kit
never falls back silently). A layout the reader does not vouch for is refused (``loader.LayoutRefused``).

Pins (``PINS``): the stock package versions + source commits and the sha256 of the source text of the stock functions
whose contract the kit relies on; ``apply()`` refuses on any drift (zero rows, never a guess).
"""
from __future__ import annotations

import contextlib
import hashlib
import inspect
import os
import time

from . import loader

KIT = "boot"
VERSION = "v0.2"
LEVERS = ("U1_device_loader", "U1_skip_meta_init")
KIT_ENV = "ESMC_KIT"                          # the shared kits package's env name (esmc_opt.kits.KIT_ENV); set by apply()
ROUTES = ("esm",)

#: the stock functions this kit's contract depends on: sha256 of ``inspect.getsource`` of each (decorators included)
PINS = {
    "esm": {
        "version": "3.4.0",
        "commit": "43ccece2ad485f27db46afdb67da2a9601e8f106",
        "functions": {
            "esm.models.hub:HubPreTrainedModel._load_pretrained": "e3dd0059a4f635a0b3b6c560ccbc68bd8541bca61e267b82fdba271e00c523f0",
            "esm.models.hub:read_safetensors_dir": "66f80da6344b4e652fd7b83708824c1cfcb718b1085244ff51c82668348816b3",
            "esm.models.esmc.model:EsmcPreTrainedModel.from_pretrained": "7d1a249e7ebd60b867363a9d08e97a27d2b28aa18208c666872686ca32fcc2e9",
            "esm.models.esmc.compatibility:ESMC.from_pretrained": "8f62141103ba7c07c450bfe726d31439c4de79ef3f76547039e7309882525c38",   # the SDK's dtype rule this kit repeats (sdk_dtype): bf16 whenever the model leaves the CPU
        },
    },
}

COUNTERS = {"esm_device_loads": 0, "hf_device_opens": 0, "tensors_staged": 0, "bytes_staged": 0, "files_staged": 0,
            "stock_passthroughs": 0, "slices_served": 0, "meta_inits_skipped": 0, "real_inits_passed": 0}
_state = {"applied": None, "last_stats": None, "last_paths": None}


class KitRefused(RuntimeError):
    """The kit refuses to run: a pin drifted, an unexpected route, an untested access pattern."""


# ---------------------------------------------------------------------------------------------- pins
def source_sha256(obj) -> str:
    return hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()


def _resolve(spec: str):
    import importlib
    mod, _, qual = spec.partition(":")
    obj = importlib.import_module(mod)
    for part in qual.split("."):
        obj = getattr(obj, part)
    return obj


def _dist_record(dist_name: str) -> dict:
    """{version, commit, url} of an installed distribution from its metadata + PEP 610 direct_url.json (bytes, not memory)."""
    import importlib.metadata as md
    import json
    d = md.distribution(dist_name)
    rec = {"version": d.version, "commit": None, "url": None}
    try:
        raw = d.read_text("direct_url.json")
    except Exception:                                                  # noqa: BLE001 — absent for index installs
        raw = None
    if raw:
        du = json.loads(raw)
        rec["url"] = du.get("url")
        rec["commit"] = (du.get("vcs_info") or {}).get("commit_id")
    return rec


def assert_pins(route: str) -> dict:
    """Read the installed stock bytes and refuse on any drift from PINS[route]. Returns the record (what was read)."""
    if route not in ROUTES:
        raise KitRefused(f"route {route!r} is not one of {ROUTES}")
    pin = PINS[route]
    dist = _dist_record("esm")
    rec = {"route": route, "dist": dist, "functions": {}}
    drift = []
    if dist["version"] != pin["version"]:
        drift.append(f"version {dist['version']} != pinned {pin['version']}")
    if pin["commit"] and dist["commit"] != pin["commit"]:
        drift.append(f"source commit {dist['commit']} != pinned {pin['commit']}")
    for spec, want in pin["functions"].items():
        got = source_sha256(_resolve(spec))
        rec["functions"][spec] = got
        if got != want:
            drift.append(f"{spec} source sha256 {got[:16]}… != pinned {want[:16]}…")
    if drift:
        raise KitRefused(f"{KIT}: stock pin drift on route {route!r}: " + "; ".join(drift))
    return rec


def apply(route: str = "esm") -> dict:
    """Install the kit for this process on ``route``: pins checked from bytes, the kit env set, the record returned."""
    t0 = time.perf_counter()
    rec = assert_pins(route)
    os.environ[KIT_ENV] = KIT
    _state["applied"] = route
    return {"kit": KIT, "version": VERSION, "levers": list(LEVERS), "route": route, "pins": rec,
            "loader": {"chunk_bytes": loader.CHUNK_BYTES, "n_slots": loader.N_SLOTS, "n_readers": loader.N_READERS},
            "apply_s": round(time.perf_counter() - t0, 4)}


# ---------------------------------------------------------------------------------------------- counters
def counters_snapshot() -> dict:
    return dict(COUNTERS)


def counters_delta(before: dict) -> dict:
    return {k: COUNTERS[k] - before.get(k, 0) for k in COUNTERS}


def _stage(paths, device, out_dtype=None) -> dict:
    st = {}
    out = loader.load_files(paths, device, out_dtype=out_dtype, stats=st)
    COUNTERS["tensors_staged"] += st["n_tensors"]
    COUNTERS["bytes_staged"] += st["n_bytes"]
    COUNTERS["files_staged"] += st["n_files"]
    _state["last_stats"] = st
    _state["last_paths"] = list(paths)
    return out


def last_stats() -> dict | None:
    return _state["last_stats"]


@contextlib.contextmanager
def _patched(module, name: str, replacement):
    original = getattr(module, name)
    setattr(module, name, replacement)
    try:
        yield original
    finally:
        setattr(module, name, original)


# ---------------------------------------------------------------------------------------------- U1_skip_meta_init
#: torch.nn.init writers the stock module constructors call at build time (nn.Linear: kaiming_uniform_ + uniform_; nn.Embedding:
#: normal_; nn.LayerNorm: ones_ + zeros_; …). On a META tensor the write has no value to produce — but torch routes it through
#: torch._refs, whose first call imports torch._dynamo (~1 s). The patch returns META tensors untouched; real tensors take the
#: stock function (a parameter the checkpoint does not cover is still initialised exactly as stock).
INIT_WRITERS = ("uniform_", "normal_", "trunc_normal_", "constant_", "ones_", "zeros_", "eye_", "dirac_", "xavier_uniform_",
                "xavier_normal_", "kaiming_uniform_", "kaiming_normal_", "orthogonal_", "sparse_")


@contextlib.contextmanager
def skip_meta_init():
    """Scoped: every torch.nn.init writer skips META tensors (counted) and passes real tensors to the stock function (counted)."""
    import torch.nn.init as init
    originals = {name: getattr(init, name) for name in INIT_WRITERS if hasattr(init, name)}

    def make(name, original):
        def skipping(tensor, *args, **kw):
            if getattr(tensor, "is_meta", False):
                COUNTERS["meta_inits_skipped"] += 1
                return tensor
            COUNTERS["real_inits_passed"] += 1
            return original(tensor, *args, **kw)
        skipping.__name__ = name
        skipping.__wrapped__ = original
        return skipping

    for name, original in originals.items():
        setattr(init, name, make(name, original))
    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(init, name, original)


# ---------------------------------------------------------------------------------------------- the esm route
def sdk_dtype(cls, kw: dict, dev):
    """The dtype the stock load will cast the model to after reading the checkpoint (``HubPreTrainedModel._load_pretrained``:
    ``model.to(dtype)``): the caller's own ``dtype=`` (``EsmcForMaskedLM.from_pretrained``), else — for the SDK wrapper
    ``ESMC.from_pretrained``, which takes no dtype and passes ``dtype=torch.bfloat16 if device.type != "cpu" else None`` (compatibility.py,
    pinned in PINS) — bf16 off the CPU. None = no cast: the tensors arrive in the file's dtype."""
    import torch
    if "dtype" in kw:
        return kw["dtype"]
    try:
        from esm.models.esmc.compatibility import ESMC as _SDK
    except Exception:  # noqa: BLE001
        return None
    if isinstance(cls, type) and issubclass(cls, _SDK):
        return torch.bfloat16 if dev.type != "cpu" else None
    return None


def from_pretrained(name_or_path: str, *, cls=None, **kw):
    """``EsmcForMaskedLM.from_pretrained(name_or_path, **kw)`` (or ``cls``'s) with the checkpoint read on the device, its floating
    tensors delivered in the dtype the stock load casts to next (sdk_dtype) — so the device never holds the fp32 checkpoint beside
    the bf16 model."""
    import torch
    import esm.models.hub as hub
    if cls is None:
        from esm.models.esmc.model import EsmcForMaskedLM as cls
    if _state["applied"] != "esm":
        raise KitRefused(f"{KIT}: apply('esm') before from_pretrained (applied: {_state['applied']!r})")
    stock_read = hub.read_safetensors_dir

    def device_read(directory):
        paths = loader.checkpoint_files(str(directory))
        dev = torch.get_default_device()                              # the stock's ``with torch.device(device)``
        if not paths or dev.type != "cuda":
            COUNTERS["stock_passthroughs"] += 1
            return stock_read(directory)
        COUNTERS["esm_device_loads"] += 1
        return _stage(paths, dev, out_dtype=sdk_dtype(cls, kw, dev))     # floating tensors arrive in the dtype the stock casts to next: its model.to(dtype) finds nothing to convert, and nothing fp32 is held meanwhile

    with _patched(hub, "read_safetensors_dir", device_read), skip_meta_init():
        return cls.from_pretrained(name_or_path, **kw)
