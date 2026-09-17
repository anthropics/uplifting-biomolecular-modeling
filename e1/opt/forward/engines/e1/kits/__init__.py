"""E1 kits — the lever modules (``engines.e1.kits.<kit>``), one surface for all of them.

KIT SURFACE (every kit module exposes exactly these; ``assert_kit_surface`` checks it):
    KIT                       the kit's name (its key in ``KITS``)
    LEVERS                    tuple of lever ids the kit applies (each bit-identical to the stock where engaged)
    TESTED_SHAPES             tuple of (B, L) shapes the kit declares (pins.PROBE_SHAPES or a subset)
    kit_files()               {relative path: present} for the kit's source files — presence, not bytes
    apply(model, *, size, num_warps=None, ...) -> record   patches in force + counters reset
    counters_snapshot() / counters_delta(prev) / expected_per_forward(model, B, L)
    stamp() -> dict           the fail-closed run stamp (kit, kit_files, W, rmsnorm path, counters); raises unless the kit is in force
    KitRefused                the exception every refusal raises
"""
from __future__ import annotations

import importlib

KIT_SURFACE = ("KIT", "LEVERS", "TESTED_SHAPES", "kit_files", "apply", "counters_snapshot", "counters_delta",
               "expected_per_forward", "stamp", "KitRefused")

#: kit name -> module
KITS = {"v0": "engines.e1.kits.v0",                 # P1 pre-cast + P2 unpad-once + P3 block-mask cache
        "ew1": "engines.e1.kits.ew1",               # fused elementwise with the stock's rounding points, one rotary table pair per module
        "v1": "engines.e1.kits.v1",                 # the EXACT composition ew1 -> attn -> P1 under the one-owner order (checks its components present)
        "attn": "engines.e1.kits.attn",             # varlen with shape-derived cu_seqlens + the index-only flex mask
        "v1.2": "engines.e1.kits.v1_2",             # kits.v1's composition with the class-keyed A3 table (A3 off by name on a100/b200/l40s)
        "ew1_i64": "engines.e1.kits.ew1_i64",       # the int64-offset ew1 kernels: a component dir (composed by the eager kit), never applied alone
        "multiseq": "engines.e1.kits.multiseq",     # multi-sequence / padded rows: the document block mask from block statistics + the pinned flex callable on the fallback route
        "kvcache": "engines.e1.kits.kvcache",       # prefilled-cache forwards (retrieval-augmented scoring): cache views, no re-append, packed global attention written once
        "tokenmemo": "engines.e1.kits.tokenmemo",   # host side: a context tokenised once per job, not once per row
        "v2.0": "engines.e1.kits.eager"}            # the kit applied on every forward: kits.v1_2's composition + ew1_i64 / the index guard + the any-length flex route + multiseq / kvcache / tokenmemo



def assert_kit_surface(mod) -> None:
    missing = [a for a in KIT_SURFACE if not hasattr(mod, a)]
    if missing:
        raise RuntimeError(f"kit {getattr(mod, '__name__', mod)!r} lacks the surface attributes {missing}")


def load_kit(name: str):
    if name not in KITS:
        raise KeyError(f"unknown kit {name!r}; known: {sorted(KITS)}")
    mod = importlib.import_module(KITS[name])
    assert_kit_surface(mod)
    if mod.KIT != name:
        raise RuntimeError(f"kit module {KITS[name]} declares KIT={mod.KIT!r} != {name!r}")
    return mod
