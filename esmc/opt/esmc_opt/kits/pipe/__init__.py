"""ESMC kit ``pipe`` — the PIPELINE levers (everything around the forward) of the SDK route
``ESMC.from_pretrained`` → ``encode`` → ``logits``. Nothing here runs a forward, writes a file or touches an output tensor.

  boot     esmc_opt.kits.boot BY IMPORT: the device-side weight loader + the meta-init skip (its own pins + counters),
           applied at the load.
  tok      The exact character-table tokeniser behind the route's one call form ``tokenizer([seq], return_tensors="pt",
           padding=True)`` (esm compatibility.py ``_tokenize``): a 256-entry table over the single-character vocab + cls/eos — the
           stock's int64 ids; every other call form / character goes to the stock call, counted.

apply("sdk", levers | "auto", weights_dir=None) -> record (pins from bytes; refuse on drift). levers="auto" = the kit's own table
(ROUTE_DEFAULTS / resolve_levers). Both levers patch at the load: the one swap of the documented lines is the from_pretrained call,
served by ``from_pretrained_sdk``. apply() may run before or after the upstream import — nothing here depends on import order.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import sys
import time

KIT = "pipe"
VERSION = "v0.16"
LEVERS = ("boot", "tok")
ROUTES = ("sdk",)
KIT_ENV = "ESMC_KIT"
BOOT_KIT = "esmc_opt.kits.boot"          # the loader's home (composed by import, never copied)

_HERE = os.path.dirname(os.path.abspath(__file__))
_PINS_FILE = os.path.join(_HERE, "pins.json")
#: versions + sha256 of the pinned files on the image: pins.json (written once by `build` on the image; frozen with the kit). The tok
#: lever's sites: the tokenizer module (its class + SEQUENCE_VOCAB) and compatibility.py (the route's one tokenizer call form).
PINS_TEMPLATE = {"torch": {"version": None},
                 "esm": {"version": None, "models/esmc/tokenizer.py": None, "models/esmc/compatibility.py": None}}
PINS = json.load(open(_PINS_FILE)) if os.path.exists(_PINS_FILE) else PINS_TEMPLATE

COUNTERS = {"tok_fast": 0, "tok_fallback": 0, "tok_patched": 0}
_state = {"applied": None, "levers": (), "boot": None, "boot_record": None, "tok": None}


class KitRefused(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------- pins
def _dist_file(dist: str, rel: str) -> str:
    mod = sys.modules.get(dist)
    if mod is not None and getattr(mod, "__file__", None):
        base = os.path.dirname(mod.__file__)
    else:
        spec = importlib.util.find_spec(dist)
        if spec is None or not spec.submodule_search_locations:
            raise KitRefused(f"{KIT}: distribution {dist!r} not importable")
        base = list(spec.submodule_search_locations)[0]
    return os.path.join(base, rel)


def _sha_file(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _version(dist: str) -> str:
    from importlib.metadata import version
    return version(dist)


def read_pins() -> dict:
    """The live values of every pin (versions + file shas) — what `build` freezes and `apply` compares."""
    out = {}
    for dist, spec in PINS.items():
        rec = {"version": _version(dist)}
        for rel in spec:
            if rel == "version":
                continue
            rec[rel] = _sha_file(_dist_file(dist, rel))
        out[dist] = rec
    return out


def assert_pins(levers) -> dict:
    live = read_pins()
    bad = []
    for dist, spec in PINS.items():
        for rel, want in spec.items():
            if want is None:
                bad.append(f"{dist}:{rel} UNPINNED (run build on the image)")
            elif live[dist][rel] != want:
                bad.append(f"{dist}:{rel} {live[dist][rel][:12]} != pinned {str(want)[:12]}")
    if bad:
        raise KitRefused(f"{KIT}: refused — pin drift: {bad}")
    return live


def build(out_dir: str | None = None) -> dict:
    """Write-once: pins.json (the live versions + file shas of this image) into the kit dir (or out_dir) — the frozen pins."""
    live = read_pins()
    rec = {k: {kk: live[k][kk] for kk in spec} for k, spec in PINS_TEMPLATE.items()}
    path = os.path.join(out_dir or _HERE, "pins.json")
    if os.path.exists(path):
        raise FileExistsError(path)
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
    return {"pins": rec, "path": path}


# ---------------------------------------------------------------------------------------------- the lever table
BOOT_FLOOR_BYTES = 8 << 30                      # the checkpoint size at or above which the device-side loader alone pays for the kit route (the package's size gate reads it)
ROUTE_DEFAULTS = {
    "sdk": {"boot": "on at every size",
            "tok": "on"},
}


def checkpoint_bytes(weights_dir) -> int:
    """The checkpoint's bytes on disk (*.safetensors / *.bin / *.pt under the dir) — os.stat only, no torch, before the route import."""
    if not weights_dir or not os.path.isdir(weights_dir):
        return 0
    n = 0
    for root, _, files in os.walk(weights_dir):
        for f in files:
            if f.endswith((".safetensors", ".bin", ".pt", ".pth")):
                try:
                    n += os.stat(os.path.join(root, f)).st_size
                except OSError:
                    pass
    return n


def resolve_levers(route: str, weights_dir=None) -> tuple:
    """The kit's own table -> (levers, {checkpoint_bytes, reasons}) for the route."""
    if route not in ROUTES:
        raise KitRefused(f"{KIT}: route {route!r} not in {ROUTES}")
    nbytes = checkpoint_bytes(weights_dir)
    out = list(LEVERS)
    why = {x: ROUTE_DEFAULTS[route][x] for x in out}
    return tuple(out), {"checkpoint_bytes": nbytes, "reasons": why}


def apply(route: str, levers="auto", weights_dir=None) -> dict:
    t0 = time.perf_counter()
    if route not in ROUTES:
        raise KitRefused(f"{KIT}: route {route!r} not in {ROUTES}")
    auto = None
    if levers == "auto" or (isinstance(levers, (tuple, list)) and tuple(levers) == ("auto",)):
        levers, auto = resolve_levers(route, weights_dir)
    levers = tuple(levers)
    bad = [x for x in levers if x not in LEVERS]
    if bad:
        raise KitRefused(f"{KIT}: unknown levers {bad}")
    if _state["applied"]:
        raise KitRefused(f"{KIT}: already applied ({_state['applied']})")
    pins = assert_pins(levers)
    rec = {"kit": KIT, "version": VERSION, "route": route, "levers": list(levers), "pins": pins,
           "auto": auto}                                                  # the table's verdict + reasons when levers='auto'; None when explicit
    if "boot" in levers:
        rec["boot"] = "deferred to the load call (the boot kit's apply reads its pins from torch + the route's loader module: counted in from_pretrained, never in the import terms)"
    os.environ[KIT_ENV] = KIT
    _state["applied"], _state["levers"] = route, levers
    rec["apply_s"] = round(time.perf_counter() - t0, 4)
    return rec


def serves_load(levers) -> bool:
    """The kit's load wrapper serves the from_pretrained call when a lever lives there: boot (the loader) and tok (the tokenizer patch
    is installed inside the wrapper, on the class that exists only after the load)."""
    return bool({"boot", "tok"} & set(levers))


def counters() -> dict:
    out = {"pipe": dict(COUNTERS)}
    if _state["boot"] is not None:
        out["boot"] = _state["boot"].counters_snapshot()
        out["boot_stats"] = _state["boot"].last_stats()
        out["boot_record"] = _state["boot_record"]
    if _state["tok"] is not None:
        out["tok"] = _state["tok"]
    return out


# ---------------------------------------------------------------------------------------------- the load
def _boot(route: str):
    """The boot kit applied once, at the load (its pins read torch + the route's loader module — on the load clock, not the import clock)."""
    if _state["boot"] is None:
        t0 = time.perf_counter()
        boot = importlib.import_module(BOOT_KIT)
        _state["boot_record"] = boot.apply("esm")
        _state["boot_record"]["apply_s_at_load"] = round(time.perf_counter() - t0, 4)
        _state["boot"] = boot
    return _state["boot"]


def from_pretrained_sdk(name_or_path: str, *, device, **kw):
    """``ESMC.from_pretrained(name_or_path, device=device, **kw)`` served by the kit's levers."""
    _check("sdk")
    from esm.models.esmc import ESMC
    levers = _state["levers"]
    if "boot" in levers:
        _boot("sdk")
        client = _state["boot"].from_pretrained(name_or_path, cls=ESMC, device=device, **kw)
    else:
        client = ESMC.from_pretrained(name_or_path, device=device, **kw)
    if "tok" in levers:
        _install_tok("sdk")
    return client


def _check(route: str) -> None:
    if _state["applied"] != route:
        raise KitRefused(f"{KIT}: apply({route!r}) before the load (applied: {_state['applied']!r})")


# ---------------------------------------------------------------------------------------------- tok (U2): the exact character-table tokeniser
#: the route tokenises ONE sequence per unit through the call tokenizer([seq], return_tensors="pt", padding=True)
#: (esm compatibility.py:230 _tokenize) — on a CHARACTER-LEVEL vocab (SEQUENCE_VOCAB: 33 tokens, every residue one
#: character, BPE with no merges, TemplateProcessing <cls> $A <eos>). The fast path = a 256-entry table over the single-character vocab
#: entries + the instance's cls/eos ids: input_ids int64 (1, L+2), attention_mask int64 ones — the stock's bytes (witness: the ids sha per
#: unit in the cert pass). Anything else (other kwargs, a batch, a non-ASCII / non-vocab character, special-token strings, lowercase) goes
#: to the stock call unchanged and is COUNTED (tok_fallback). Patched per tokenizer CLASS after the load (the class exists only then).
_TOK_SITES = {"sdk": ("esm.models.esmc.tokenizer", "EsmcTokenizer")}


def _install_tok(route: str) -> dict:
    import numpy as np
    import torch
    from transformers import BatchEncoding
    modname, clsname = _TOK_SITES[route]
    mod = importlib.import_module(modname)
    cls = getattr(mod, clsname)
    vocab = list(mod.SEQUENCE_VOCAB)
    table = np.full(256, -1, dtype=np.int64)
    for i, t in enumerate(vocab):
        if len(t) == 1:
            table[ord(t)] = i
    stock_call = cls.__call__
    rec = {"module": modname, "class": clsname, "n_vocab": len(vocab), "single_char_entries": int((table >= 0).sum()),
           "vocab_sha256": hashlib.sha256("\n".join(vocab).encode()).hexdigest(), "stock_call": f"{stock_call.__module__}.{stock_call.__qualname__}"}

    def fast_call(self, text=None, *args, **kw):
        if (args or set(kw) != {"return_tensors", "padding"} or kw["return_tensors"] != "pt" or kw["padding"] is not True
                or not isinstance(text, list) or len(text) != 1 or not isinstance(text[0], str) or not text[0].isascii()):
            COUNTERS["tok_fallback"] += 1
            return stock_call(self, text, *args, **kw)
        s = text[0]
        ids = table[np.frombuffer(s.encode("ascii"), dtype=np.uint8)] if s else np.empty(0, dtype=np.int64)
        if ids.size and (ids < 0).any():
            COUNTERS["tok_fallback"] += 1
            return stock_call(self, text, *args, **kw)
        out = np.empty(ids.size + 2, dtype=np.int64)
        out[0] = self.cls_token_id
        out[1:-1] = ids
        out[-1] = self.eos_token_id
        t = torch.from_numpy(out).unsqueeze(0); COUNTERS["tok_fast"] += 1
        return BatchEncoding({"input_ids": t, "attention_mask": torch.ones_like(t)})

    cls.__call__ = fast_call
    COUNTERS["tok_patched"] += 1
    _state["tok"] = rec
    return rec
