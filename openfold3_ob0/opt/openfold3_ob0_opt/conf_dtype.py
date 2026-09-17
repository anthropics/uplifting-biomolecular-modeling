"""The `conf_dtype` lever: the dtype the confidence phase runs under — the auxiliary heads (the 4-block confidence Pairformer, the
PAE / PDE / pLDDT / experimentally-resolved heads and the distogram head) under `torch.autocast("cuda", dtype=torch.bfloat16)`, as OpenFold3
0.4.1 runs them (the trunk's bf16 carried into the heads), instead of upstream 0.5.0's inference fp32.

Upstream 0.5.0 makes the confidence phase fp32 twice over: `run_trunk` returns `s_input.float(), s.float(), z.float()` (projects/of3_all_atom/
model.py `run_trunk`), so `_rollout`'s `cast_dtype = torch.float32 if self.training else si_trunk.dtype` (model.py, the `# Compute confidence
logits` block) is fp32 at inference, and `AuxiliaryHeadsAllAtom.forward` (core/model/heads/head_modules.py) takes `pairformer_dtype=torch.float32`,
under which `PairformerEmbedding` runs its Pairformer (core/model/heads/prediction_heads.py `torch.amp.autocast(..., dtype=pairformer_dtype)`);
`_rollout` never passes another value. Under `bf16` this lever wraps that ONE forward: it runs inside `torch.autocast("cuda", dtype=torch.bfloat16)`
with `pairformer_dtype=torch.bfloat16` unless the caller passed one (counted `explicit=<n>`, honoured). Nothing else moves: the trunk outputs stay
fp32 (`.float()` untouched — they feed the diffusion conditioning), the diffusion sampler keeps its own dtype (upstream's fp32 roll-out, or the
`rollout_bf16` lever's), and the confidence phase runs AFTER the sampler on its finished coordinates — no head output feeds back into
`atom_positions_predicted` (model.py `_rollout`: `sample_diffusion(...)` → `output["atom_positions_predicted"]`, then `output.update(self.aux_heads(...))`;
the runner's sample ranking only SELECTS among the written samples). Numerics class: tier 2 for the confidence numbers (pLDDT / PAE / PDE / pTM /
ipTM / ranking score), coordinates bitwise those of the fp32 heads. The row-sharded line (`big --n_gpu P`) reads the same word for its own
statement of the phase (tp_rowpair/model.py: `cast_dtype`, `aux_heads_rows(pairformer_dtype=…)`).

Words: OPENFOLD3_OB0_OPT_CONF_DTYPE = bf16 | fp32 — the mode's line sets it (`bf16` on fast and on both big compositions, `fp32` = upstream on
exact; stock under off) and a value preset in the caller's environment, or `pred --conf-dtype`, wins over the line's (never a conflict: it is a
knob); OPENFOLD3_OB0_OPT_CONF_DTYPE_SOURCE = mode | env says which one decided (modes.resolve writes it). The ACTIVE line carries
`conf_dtype=<word> source=<mode|env>`; at exit the census `[openfold3_ob0-opt/conf_dtype] LEVER name=conf_dtype state=on|off dtype=<word>
source=<word> calls=<n> explicit=<n>`. Under `bf16` the row-sharded line's kernel census loses its fp32 gate words (`TRIATT … fallback_by=
unsupported:dtype_float32`, `TRIMUL … fallback_by=layout`): the confidence pair blocks reach the fused kernels in bf16 like the trunk's.
`fallbacks()` = {} (nothing degrades by design).
"""
from __future__ import annotations

import atexit
import os
import sys
import threading
from typing import Optional, Tuple

TAG = "openfold3_ob0-opt/conf_dtype"
PREFIX = f"[{TAG}]"
ENV = "OPENFOLD3_OB0_OPT_CONF_DTYPE"                          # bf16 | fp32
ENV_SOURCE = "OPENFOLD3_OB0_OPT_CONF_DTYPE_SOURCE"            # mode | env (modes.resolve writes it; absent = env when ENV is set by hand)
WORDS: Tuple[str, ...] = ("bf16", "fp32")
SOURCES: Tuple[str, ...] = ("mode", "env")
UPSTREAM = "fp32"                                            # upstream 0.5.0's inference dtype of the phase (the word of a process without the lever)
TARGET = "openfold3.core.model.heads.head_modules"           # AuxiliaryHeadsAllAtom.forward: the ONE seam (registry T_HEAD_MODULES)
ATTR = "AuxiliaryHeadsAllAtom.forward"

STATE: dict = {"installed": False, "word": None, "source": None, "calls": 0, "explicit": 0, "errors": []}
_LOCK = threading.Lock()
PATCH = None                                                 # the opt_core.autoload.AttrPatch once install() armed it


def check_word(value: str) -> str:
    """`bf16` | `fp32`, else ValueError naming the variable."""
    w = (value or "").strip().lower()
    if w not in WORDS:
        raise ValueError(f"{ENV}={value!r}: expected one of {'|'.join(WORDS)}")
    return w


def setting(environ=None) -> Tuple[Optional[str], Optional[str]]:
    """(word, source) this process runs with: word None when the variable is unset (upstream's fp32, no lever), source `mode` | `env`."""
    environ = os.environ if environ is None else environ
    raw = environ.get(ENV)
    if raw in (None, ""):
        return None, None
    src = (environ.get(ENV_SOURCE) or "env").strip().lower()
    return check_word(raw), (src if src in SOURCES else "env")


def torch_dtype(word: Optional[str] = None):
    """The torch dtype of a word (None / unset → upstream's float32)."""
    import torch
    w = STATE["word"] if word is None else word
    return torch.bfloat16 if w == "bf16" else torch.float32


def head_dtypes(default_cast):
    """(cast_dtype, pairformer_dtype) for the row-sharded line's statement of the phase (tp_rowpair/model.py): under `bf16` both bf16, else
    upstream's (`default_cast` = si_trunk.dtype, pairformer fp32). Counts the call like the wrapped forward does."""
    import torch
    with _LOCK:
        STATE["calls"] += 1
    if STATE["word"] == "bf16":
        return torch.bfloat16, torch.bfloat16
    return default_cast, torch.float32


def fragment(word: Optional[str], source: Optional[str]) -> Optional[str]:
    """The ACTIVE line's words, `conf_dtype=<word> source=<mode|env>` (None when unset)."""
    return None if word is None else f"conf_dtype={word} source={source}"


def _make_forward(orig):
    """The wrapper factory the core calls when the heads module is imported (this process runs the model): announce once here, not at
    arm time — upstream's data-loader workers activate the package too and never import the model."""
    import torch
    sys.stderr.write(f"{PREFIX} installed: {ATTR} under torch.autocast(cuda, bfloat16) with pairformer_dtype=bfloat16 "
                     f"(dtype=bf16 source={STATE['source']})\n")

    def forward(self, *args, **kwargs):
        with _LOCK:
            STATE["calls"] += 1
            if kwargs.get("pairformer_dtype") is not None:
                STATE["explicit"] += 1
        if kwargs.get("pairformer_dtype") is None:
            kwargs["pairformer_dtype"] = torch.bfloat16
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return orig(self, *args, **kwargs)
    forward._of3ob0_conf_dtype = True                     # stack's probe reads it
    forward.__wrapped__ = orig
    return forward


def install(environ=None) -> dict:
    """Idempotent. Reads the words; under `bf16` patches AuxiliaryHeadsAllAtom.forward now or at the module's import
    (opt_core.autoload.patch_attr_at_import — fail-closed by name); under `fp32` / unset records the word and patches nothing. Returns STATE."""
    global PATCH
    word, source = setting(environ)
    with _LOCK:
        STATE["word"], STATE["source"] = word or UPSTREAM, source or "-"
    if word == "bf16" and not STATE["installed"]:
        from opt_core import autoload as _core_autoload
        PATCH = _core_autoload.patch_attr_at_import(TARGET, ATTR, _make_forward, tag=TAG, name="conf_dtype")
        with _LOCK:
            STATE["installed"] = True
        atexit.register(lambda: sys.stderr.write(census_line() + "\n"))
    return STATE


def uninstall() -> None:
    """Disarm / restore the patch (tests)."""
    global PATCH
    if PATCH is not None:
        try:
            if PATCH.state == "installed" and PATCH.original is not None:
                mod = sys.modules.get(TARGET)
                owner = getattr(mod, ATTR.split(".")[0], None) if mod is not None else None
                if owner is not None:
                    setattr(owner, ATTR.split(".")[1], PATCH.original)
            PATCH.state = "disarmed"
        finally:
            from opt_core import autoload as _core_autoload
            _core_autoload._PATCHES.pop((TARGET, ATTR), None)
            PATCH = None
    with _LOCK:
        STATE.update(installed=False, word=None, source=None, calls=0, explicit=0, errors=[])


def census_line() -> str:
    return (f"{PREFIX} LEVER name=conf_dtype state={'on' if STATE['installed'] else 'off'} dtype={STATE['word'] or UPSTREAM} "
            f"source={STATE['source'] or '-'} calls={STATE['calls']} explicit={STATE['explicit']}")


def fallbacks() -> dict:
    """Nothing falls back by design: the lever is a dtype decision."""
    return {}
