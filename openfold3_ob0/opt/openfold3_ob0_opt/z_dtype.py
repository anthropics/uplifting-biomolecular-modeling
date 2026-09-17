"""The `z_dtype` lever: the dtype of the trunk's hand-off to the roll-out — the three representations `run_trunk` returns (`s_input`,
`s`, `z`; the pair representation `z` is the one that matters for memory) reach the diffusion sampler and the confidence phase in the
trunk's own bf16, as OpenFold3 0.4.1 hands them (projects/of3_all_atom/model.py 0.4.1 `run_trunk`: `return s_input, s, z`), instead of
upstream 0.5.0's fp32 copies (`return s_input.float(), s.float(), z.float()`, model.py `run_trunk`).

What upstream 0.5.0's `.float()` costs: after the trunk the roll-out holds an fp32 pair representation (N² · 128 · 4 B) where 0.4.1 holds bf16
(half of it), and every consumer downstream reads that fp32 `z`: the diffusion
conditioning (the sampler runs under `torch.amp.autocast(..., dtype=torch.float32)` in both versions — with a bf16 `z` its statements up-cast
per op, with an fp32 `z` they read it as is), the confidence phase's `z` embedding and Pairformer, the distogram head. On the row-sharded line
(`big --n_gpu P`) the same holds per rank for the z shard (rows/P), and the diffusion stage's pair-conditioning caches follow the hand-off
dtype — that stage sets the device peak of large inputs.

The lever's statements, by route:
  * `fast` / `exact` (upstream's model class): `OpenFold3.run_trunk` is wrapped at import (opt_core.autoload.patch_attr_at_import, fail-closed
    by name); under `bf16` its three outputs are handed on as bfloat16 (`.to(torch.bfloat16)` right after upstream's `.float()`: a transient
    fp32 copy at the hand-off, then 0.4.1's dtypes; bf16 → fp32 → bf16 is exact, so the values are the trunk's own).
  * `big` resident (opt/forward/offload/of3o/of3_offload.py `run_trunk`) and `big --n_gpu P` (tp_rowpair/trunk.py): the kit's own trunk
    statements end in `handoff(s_input, s, z)` — under `bf16` 0.4.1's `return s_input, s, z` (no copy at all), under `fp32` 0.5.0's `.float()`.
Under `fp32` (upstream) nothing is patched and `handoff` is upstream's statement. The confidence phase's `cast_dtype = si_trunk.dtype`
(model.py `_rollout`) then follows the hand-off as in 0.4.1: `z_dtype=bf16` with `conf_dtype=fp32` is refused by name (modes.resolve), the
two upstream words go together (`--z-dtype fp32 --conf-dtype fp32` = upstream 0.5.0's roll-out dtypes).

Numerics class: tier 2 — the sampler reads `z`, so COORDINATES MOVE (unlike conf_dtype), within the run-to-run floor and the
sample-to-sample spread.

Words: OPENFOLD3_OB0_OPT_Z_DTYPE = bf16 | fp32 — the mode's line sets it (`bf16` on fast and both big compositions, `fp32` = upstream on
exact; stock under off); a value preset in the caller's environment, or `pred --z-dtype`, wins (`source=env`); OPENFOLD3_OB0_OPT_Z_DTYPE_SOURCE
= mode | env. ACTIVE line `z_dtype=<word> source=<mode|env>`; exit census `[openfold3_ob0-opt/z_dtype] LEVER name=z_dtype state=on|off
dtype=<word> source=<word> handoffs=<n> in=<dtype seen> out=<dtype handed>`. `fallbacks()` = {}.
"""
from __future__ import annotations

import atexit
import os
import sys
import threading
from typing import Optional, Tuple

from . import conf_dtype as _CD                                # one vocabulary for the kit's dtype knobs: WORDS / SOURCES / the word check

TAG = "openfold3_ob0-opt/z_dtype"
PREFIX = f"[{TAG}]"
ENV = "OPENFOLD3_OB0_OPT_Z_DTYPE"                             # bf16 | fp32
ENV_SOURCE = "OPENFOLD3_OB0_OPT_Z_DTYPE_SOURCE"               # mode | env
WORDS: Tuple[str, ...] = _CD.WORDS
SOURCES: Tuple[str, ...] = _CD.SOURCES
UPSTREAM = "fp32"                                            # upstream 0.5.0's hand-off (the word of a process without the lever)
OF3_041 = "bf16"                                             # OpenFold3 0.4.1's hand-off: the trunk's own dtype under bf16-mixed
TARGET = "openfold3.projects.of3_all_atom.model"             # OpenFold3.run_trunk: the seam on upstream's model class (registry T_MODEL)
ATTR = "OpenFold3.run_trunk"

STATE: dict = {"installed": False, "word": None, "source": None, "handoffs": 0, "in": None, "out": None, "errors": []}
_LOCK = threading.Lock()
PATCH = None


def check_word(value: str) -> str:
    """`bf16` | `fp32`, else ValueError naming THIS variable."""
    w = (value or "").strip().lower()
    if w not in WORDS:
        raise ValueError(f"{ENV}={value!r}: expected one of {'|'.join(WORDS)}")
    return w


def setting(environ=None) -> Tuple[Optional[str], Optional[str]]:
    """(word, source) this process runs with: word None when the variable is unset (upstream's fp32, no lever)."""
    environ = os.environ if environ is None else environ
    raw = environ.get(ENV)
    if raw in (None, ""):
        return None, None
    src = (environ.get(ENV_SOURCE) or "env").strip().lower()
    return check_word(raw), (src if src in SOURCES else "env")


def word() -> str:
    """The word in force in this process (install() recorded it; before install, read the environment; unset = upstream)."""
    w = STATE["word"]
    if w is None:
        w = setting()[0]
    return w or UPSTREAM


def fragment(word_: Optional[str], source: Optional[str]) -> Optional[str]:
    """The ACTIVE line's words, `z_dtype=<word> source=<mode|env>` (None when unset)."""
    return None if word_ is None else f"z_dtype={word_} source={source}"


def _dt(t) -> str:
    return str(getattr(t, "dtype", None)).rsplit(".", 1)[-1]


def handoff(s_input, s, z):
    """The trunk's hand-off statement for the kit's own trunks (the resident port's run_trunk, the row-sharded trunk): under `bf16`
    OpenFold3 0.4.1's `return s_input, s, z` (the trunk's tensors themselves, no copy); under `fp32` upstream 0.5.0's
    `return s_input.float(), s.float(), z.float()`. Counted; the dtypes seen and handed are recorded for the census."""
    w = word()
    out = (s_input, s, z) if w == "bf16" else (s_input.float(), s.float(), z.float())
    with _LOCK:
        STATE["handoffs"] += 1; STATE["in"] = _dt(z); STATE["out"] = _dt(out[2])
    return out


def _make_run_trunk(orig):
    """The wrapper factory the core calls when upstream's model module is imported (this process runs the model): announce once here, not
    at arm time — upstream's data-loader workers activate the package too and never import the model."""
    import torch
    sys.stderr.write(f"{PREFIX} installed: {ATTR} hands s_input, s, z to the roll-out in bfloat16 (OpenFold3 0.4.1's hand-off; "
                     f"dtype=bf16 source={STATE['source']}); the kit's own trunks hand off uncast\n")

    def run_trunk(self, *args, **kwargs):
        s_input, s, z = orig(self, *args, **kwargs)                         # upstream's statement, `.float()` included
        out = (s_input.to(torch.bfloat16), s.to(torch.bfloat16), z.to(torch.bfloat16))   # 0.4.1's dtypes; bf16 -> fp32 -> bf16 is exact
        with _LOCK:
            STATE["handoffs"] += 1; STATE["in"] = _dt(z); STATE["out"] = _dt(out[2])
        return out
    run_trunk._of3ob0_z_dtype = True                      # stack's probe reads it
    run_trunk.__wrapped__ = orig
    return run_trunk


def install(environ=None) -> dict:
    """Idempotent. Reads the words; under `bf16` patches OpenFold3.run_trunk now or at the module's import (fail-closed by name); under
    `fp32` / unset records the word and patches nothing (the kit's own trunks read `word()` at their hand-off). Returns STATE."""
    global PATCH
    w, source = setting(environ)
    with _LOCK:
        STATE["word"], STATE["source"] = w or UPSTREAM, source or "-"
    if w == "bf16" and not STATE["installed"]:
        from opt_core import autoload as _core_autoload
        prior = _core_autoload._PATCHES.get((TARGET, ATTR))
        if prior is not None and prior.state == "installed" and sys.modules.get(TARGET) is None:   # the patched module left sys.modules (a test re-importing stock): that patch is history, start over
            _core_autoload._PATCHES.pop((TARGET, ATTR), None)
        PATCH = _core_autoload.patch_attr_at_import(TARGET, ATTR, _make_run_trunk, tag=TAG, name="z_dtype")
        with _LOCK:
            STATE["installed"] = True
        atexit.register(lambda: sys.stderr.write(census_line() + "\n"))
    return STATE


def uninstall() -> None:
    """Disarm / restore the patch (tests)."""
    global PATCH
    if PATCH is not None:
        try:
            if PATCH.state == "installed" and sys.modules.get(TARGET) is not None:
                PATCH.restore()                                                   # the core puts the original attribute back
            else:
                PATCH.disarm()
            PATCH.state = "disarmed"
        finally:
            from opt_core import autoload as _core_autoload
            _core_autoload._PATCHES.pop((TARGET, ATTR), None)
            PATCH = None
    with _LOCK:
        STATE.update(installed=False, word=None, source=None, handoffs=0, **{"in": None, "out": None}, errors=[])


def census_line() -> str:
    return (f"{PREFIX} LEVER name=z_dtype state={'on' if STATE['installed'] else 'off'} dtype={STATE['word'] or UPSTREAM} "
            f"source={STATE['source'] or '-'} handoffs={STATE['handoffs']} in={STATE['in'] or '-'} out={STATE['out'] or '-'}")


def fallbacks() -> dict:
    """Nothing falls back by design: the lever is a dtype decision."""
    return {}
