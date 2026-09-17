"""Kit ``v0_ew`` for ProGen2: exact fused elementwise CUDA kernels for the stock block (salesforce/progen @ c27a419c, transformers
4.16.2, torch 2.8.0+cu128), each reproducing the stock's per-op rounding chain bit for bit and capturable inside a CUDA graph
(no host syncs, no allocations beyond the output tensors, launches on the current stream).

Levers (``LEVERS``; patches.py): gelu (8 kernels -> 1), rotary (the qkv -> q/k/v section: ~21 kernels + the per-call CPU
tables and H2D -> hoisted tables + 1 kernel), residual (2 -> 1), glue (3 -> 1), ln (the vectorized layer-norm replica, the
building block for an add+LN or LN+GEMV fusion). Binary: libew_progen2.so built by build.py (BUILD.json pins).

Engagement (``apply``): the levers engage wherever their mechanism applies — every one of them, or none: what is UNTESTED about the
box is NAMED in the record's ``notes`` and engaged on (the running torch / transformers off the versions the kernels were tested on;
the installed transformers' `gelu_new` off the bytes the gelu kernel was tested against; a card without a prebuilt kernel image, served
by the driver's JIT of the PTX image: SASS sm_80 serves cc 8.x, sm_90 9.0, sm_100 10.x, PTX compute_90 any newer card); what CANNOT run
raises KitRefused naming the levers (``ew:gelu, ew:rotary, … cannot run: <reason>``: the library missing, not loading, binding another
CUDA runtime major or another launcher table, a card older than every image, no CUDA device) — the caller's mode refuses by name.
"""
from __future__ import annotations

import hashlib
import os

from .ext import KitRefused
from .patches import (CTR, EXACT_LEVERS, apply as _apply_patches, assert_counts, counters_snapshot, expected_counts, stock_source_pins,
                      unapply)

KIT = "v0_ew"
LEVERS = tuple(EXACT_LEVERS)
STOCK_COMMIT = "c27a419c234a0997923761e1fe7daffcebf0eaf5"
STOCK_ACTIVATIONS_SHA256 = "6065ad152b42cf790f120037a2be94765b8514a27c6dd4de581f162c308b3825"
STACK = {"torch": "2.8.0+cu128", "transformers": "4.16.2", "python": "3.9.23"}
_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULES = ("__init__.py", "patches.py", "ext.py", "kernels.cu")
_state = {"applied": False, "size": None, "record": None, "hook": None, "levers": None, "model_id": None}


def running_stack() -> tuple[dict, list]:
    """The running torch / transformers versions and, when they differ from the stack the kernels were tested on (STACK), the note
    that says so — an uncertainty named, never a refusal."""
    import torch
    import transformers
    got = {"torch": torch.__version__, "transformers": transformers.__version__}
    diff = [f"{k} {v} (tested on {STACK[k]})" for k, v in got.items() if v != STACK[k]]
    return got, ([f"kit v0_ew: the running stack differs from the tested one — {'; '.join(diff)}"] if diff else [])


def stock_bytes() -> tuple[dict, list]:
    """The installed transformers activations.py against the pinned bytes: its `gelu_new` is the function the gelu kernel was tested
    against bit for bit; other bytes there are an UNTESTED library version — named as a note, the lever engages (the other levers replace
    this repo's own vendored modeling_progen.py, which the caller's activation hashes against the pinned commit). Returns (record, notes)."""
    from models.progen import modeling_progen as mp
    from transformers import activations
    assert mp.__file__   # importable, on sys.path; its bytes are the stock's, hashed by the caller's activation
    got = hashlib.sha256(open(activations.__file__, "rb").read()).hexdigest()
    notes = []
    if got != STOCK_ACTIVATIONS_SHA256:
        notes.append(f"kit v0_ew: transformers/activations.py sha256 {got[:8]} != the tested {STOCK_ACTIVATIONS_SHA256[:8]}: the gelu kernel was tested against other bytes of gelu_new (engaged)")
    return {"activations.py": got}, notes


def cannot_run(reason: str, levers=None) -> KitRefused:
    """The refusal of this kit's levers BY NAME: `ew:gelu, ew:rotary, … cannot run: <reason>` (a mode is all of its levers — the caller
    refuses the mode, exit 3; it never runs under the mode's name without them)."""
    names = ", ".join(f"ew:{l}" for l in (levers or LEVERS))
    return KitRefused(f"kit v0_ew: {names} cannot run: {reason}")


def kernel_image(build: dict, device) -> str | None:
    """Which image of the prebuilt library runs on the model's card (ext.kernel_image on its compute capability): a SASS image of the
    card's major (cc 8.x / 9.0 / 10.x: nothing to say), the PTX image JIT-compiled by the driver for a card without one (returned as the
    note that names it), or none — a card older than every image: the levers cannot run there (raises cannot_run)."""
    import torch
    from . import ext
    cc = tuple(torch.cuda.get_device_capability(device))
    kind, arch = ext.kernel_image(build, cc)
    if kind is None:
        raise cannot_run(arch)
    if kind == "ptx":
        return f"ew kernels JIT-compiled from PTX (compute_{arch[0]}{arch[1]}) on sm_{cc[0]}{cc[1]}: no prebuilt image for this card in {ext.BINARY}"
    if kind == "unknown":
        return f"{ext.BINARY}'s BUILD.json names no kernel images: whether one runs on sm_{cc[0]}{cc[1]} shows at the first launch"
    return None


def counters_delta(prev: dict) -> dict:
    now = counters_snapshot()
    return {k: now.get(k, 0) - prev.get(k, 0) for k in set(now) | set(prev)}


def expected_per_forward(model=None, autocast: bool = False) -> dict:
    """What ONE forward must count under the applied levers (shape-independent: one kernel call per patched site)."""
    return expected_counts(model, _state["levers"] if _state["levers"] is not None else LEVERS, autocast=autocast)


def assert_counted_path(delta: dict, n_forwards: int, autocast: bool = False) -> None:
    """Counts == n_forwards x the expectation for every patched path, no shadow mismatch, forwards == n_forwards."""
    if not _state["applied"]:
        raise KitRefused("kit v0_ew not applied")
    if n_forwards < 1 or delta.get("forwards", 0) != n_forwards:
        raise KitRefused(f"forward count {delta.get('forwards')} != {n_forwards}")
    assert_counts(delta, expected_per_forward(autocast=autocast), n_forwards)


def _count_forward(module, args, kwargs):
    CTR["forwards"] += 1


def apply(model, *, size: str, levers=LEVERS, shadow: bool = False, ln_variant: int | None = None,
          require_gpu: bool = True) -> dict:
    """The running stack and the installed gelu_new bytes NAMED when they are not the tested ones (``notes``: untested, engaged), the
    prebuilt library loaded and its image for the card resolved (a PTX JIT named as a note), then every lever bound on the stock objects
    -> record. What CANNOT run raises KitRefused by lever name (cannot_run: the library missing / not loading / binding another CUDA
    runtime major / another launcher table, no kernel image for the card) — the caller's mode refuses; nothing is applied in part. Apply
    this kit BEFORE anything else patches the same classes (it refuses when the stock functions it captures are already replaced)."""
    import torch
    from . import ext
    if _state["applied"]:
        raise KitRefused("kit v0_ew already applied in this process")
    if require_gpu and not torch.cuda.is_available():
        raise cannot_run("no CUDA device (the kernels are CUDA kernels)", levers)
    stack, notes = running_stack()
    stock, notes_b = stock_bytes()
    notes += notes_b
    src = stock_source_pins()
    try:
        ext.load()
    except KitRefused as e:                                            # missing / another CUDA runtime major / another launcher table
        raise cannot_run(str(e), levers) from None
    except OSError as e:                                               # bytes that do not load on this box
        raise cannot_run(f"{ext.BINARY} does not load ({type(e).__name__}: {e}) — {ext.REBUILD}", levers) from None
    build = ext.record()["build"]
    dev = model.transformer.wte.weight.device
    if dev.type == "cuda":
        note = kernel_image(build, dev)                                # raises cannot_run for a card older than every image
        if note:
            notes.append(note)
    if ln_variant is None:
        ln_variant = int(build.get("ln_variant", 0))
    rec_p = _apply_patches(model, levers=tuple(levers), shadow=shadow, ln_variant=ln_variant)
    _state["hook"] = model.register_forward_pre_hook(_count_forward, with_kwargs=True)
    CTR.clear()
    _state.update(applied=True, size=size, levers=tuple(levers), model_id=id(model))
    rec = {"kit": KIT, "levers": list(levers), "size": size, "stack": stack, "notes": notes,
           "stock_bytes": stock, "stock_commit": STOCK_COMMIT, "stock_source_sha256": src, "library": {k: v for k, v in ext.record().items() if k != "build"},
           "build": build, **rec_p}
    _state["record"] = rec
    return rec


def stamp() -> dict:
    """The fail-closed stamp every row carries: refuses unless the kit is applied in this process."""
    if not _state["applied"]:
        raise KitRefused("stamp: kit v0_ew is not in force in this process")
    import torch
    from . import ext
    from . import patches
    return {"kit": KIT, "levers": list(_state["levers"]), "size": _state["size"],
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "counters": counters_snapshot(), "ln_variant": _state["record"]["ln_variant"], "shadow": _state["record"]["shadow"],
            "rotary_tables": patches.table_sha(), "aten_cpu_capability": torch.backends.cpu.get_cpu_capability()}


def release(model) -> None:
    """Undo the patches and the forward hook (a caller may toggle lever sets in one process)."""
    if _state["hook"] is not None:
        _state["hook"].remove()
    unapply()
    _state.update(applied=False, size=None, record=None, hook=None, levers=None, model_id=None)
