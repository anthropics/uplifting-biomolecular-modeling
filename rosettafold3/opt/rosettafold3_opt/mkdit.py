"""The ``mkdit`` lever: the diffusion module's 24-block TOKEN diffusion transformer
(``rf3.model.layers.af3_diffusion_transformer.DiffusionTransformer``, the calls with ``Beta_II is None``: AdaLN → pair-biased attention →
gated residual → AdaLN → SwiGLU transition → gated residual, per block) served by the carried MK-DiT add-on's persistent Triton megakernel
(``opt/forward/rf3_mk_dit_addon/mkdit/mk2.py`` ``MK2TokenTransformer``: ONE launch per sample per denoiser call, grid = the card's SM count,
per-row-tile dependency counters, no grid barrier; plus three S-side cuBLAS/LN launches per sample) instead of the ~1800 launches of the
stock blocks. The megakernel is single-sample: a call with a leading diffusion batch ``D`` (upstream ``diffusion_batch_size=5``) runs ``D``
launches, one per sample, on the current stream in a fixed order. The per-block pair bias (``to_b(ln_0(Z_II))`` of every block, exactly the
stock sub-graph under the caller's autocast) is laid out once per roll-out as ``[24, H, I, I]`` bf16 through ``rf3.graph_flags.hoist_get``
(the ``RF3_HOIST=1`` per-roll-out cache the kit rows export) — that tensor is the lever's memory cost: ``24·16·I²·2`` bytes (0.77 GB at
1000 tokens) plus the kernel's activation buffers; the memory mode disengages the lever (``big.REPLACED_KIT``).

Numerics class: tier 2 (bf16 operands / fp32 accumulation like stock, exp2-domain online softmax, one fixed tile configuration per card;
deterministic run to run; not bitwise with the stock blocks). It is a lever of the ``fast`` mode and of the rows that name it
(modes.KIT_MODES ``kit_levers``); ``dtk`` and the FPF add-on's ``dattn`` serve the attention INSIDE the blocks this lever replaces
whole: a row names ``mkdit`` or one of them, never both (refused by name).

Install seam: ``DiffusionTransformer.forward`` is replaced class-wide by :func:`forward` after the FPF arm is applied (``stack.fpf_apply``);
the two atom transformers (the calls with ``Beta_II`` given: windowed attention) keep the class's own forward — not this lever's site,
counted ``atom_calls``. Per token call a size gate (``opt_core.attn.size_gate``; ``MIN_I`` = 400 tokens)
decides on I: below it the stock blocks run, counted ``gated``; at or above it the call is served. Inside
the sampler CUDA graph the Python side of a call runs at warm-up and capture only (the census counts those; replays repeat the captured
launches); the kernel's Triton JIT compiles once per process at the first served call (10-90 s, inside the first item).

Fail-loud: :func:`enable` refuses by name (``MkditRefused``) when the interpreter's ``rf3`` is not the patched tree (no ``rf3.graph_flags``),
when no CUDA GPU is visible (nvidia-smi),
when ``RF3_HOIST`` is off (the pair-bias layout would be rebuilt at every denoiser call), when ``dtk`` or the add-on's ``dattn`` is installed,
when a carried kernel file is absent, or when Triton / the kernel module do not import.
The tile configuration is per architecture (``CONFIGS``, keyed by compute capability: a card takes the row of the highest key at or below
its own — sm_90 for H100 / B200, sm_80 for A100). A card below the lowest key (8.0) has no configuration: :func:`enable` raises
``MkditRefused`` naming it and the mode refuses by name (the NOT ACTIVE line) — a mode is all of its levers, never a subset. A served call the megakernel
cannot take raises ``MkditRefused`` out of the forward — an input that is not on a CUDA device, a rank other than ``[I, C]`` / ``[D, I, C]``,
a pair bias with its own batch dimension: there is no silent stock branch at any batch size.

Evidence: :func:`describe` (read by report.tally into ``mkdit``) — on, impl (where the kernel executes from), the kernel files' sha256, the
tile configuration, min_tokens, the census (calls = served + gated; samples, hoists, atom_calls, shapes), ``ok`` and ``reason``: a seam no
token call reached, or a recorded error, fails the run by name (fold.lever_failures).
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
from typing import Optional

from . import _core

MIN_I = 400                                       # the size gate's lower bound in tokens (I); below it the stock blocks run, counted `gated`
ADDON = "forward/rf3_mk_dit_addon"              # the carried add-on, relative to opt/ (registry.MKDIT)
KERNEL_DIR = "mkdit"                            # the add-on's kernel directory: mk2.py (the megakernel) imports mkrf3.py (weight packing, hoist, S-side ops) by name
KERNEL_FILES = ("mkdit/mk2.py", "mkdit/mkrf3.py")
KERNEL = "mk2"                                  # the module that carries MK2TokenTransformer
CONFIGS = {                                     # one tile configuration per architecture (the kernel's numerics are configuration-specific); a card takes the row of the
    (9, 0): {"BMR": 64, "BN": 128, "BK": 64, "BNK": 64, "num_warps": 8, "num_stages": 3, "bms": 16},   # highest key at or below its compute capability. sm_90: H100 (B200, sm_100, takes this row)
    (8, 0): {"BMR": 64, "BN": 128, "BK": 64, "BNK": 64, "num_warps": 8, "num_stages": 3, "bms": 16},   # sm_80: A100 — the same tiles fit its 163 KB of shared memory per block
}
MIN_CAPABILITY = min(CONFIGS)                   # (8, 0): below it no configuration exists and enable() refuses by name
CONFIG = CONFIGS[(9, 0)]                        # the sm_90 configuration (the reference row)
ADAPTER = "fpf_rf3_adapter"                     # the FPF add-on's module (its DATTN record says whether dattn is installed)

STATE = {"on": False, "impl": None, "home": None, "files_sha256": None, "error": None, "gpu": None, "reason": None, "config": None,
         "samples": 0, "hoists": 0, "atom_calls": 0, "objs": 0, "shapes": {}}
GATE = None                                     # opt_core.attn.size_gate.SizeGate, built by enable()
_ORIG = {}                                      # {"forward": the class's own DiffusionTransformer.forward}
_OBJS = {}                                      # id(DiffusionTransformer instance) -> MK2TokenTransformer


class MkditRefused(RuntimeError):
    """The lever cannot apply (named precondition) — at enable: stack.fpf_apply turns it into the NOT ACTIVE line; inside a served call: the
    fold dies by name (no stock branch)."""


def make_gate(min_tokens: int = MIN_I):
    """The size gate on I (``opt_core.attn.size_gate``): calls below ``min_tokens`` run the stock blocks, counted ``gated``."""
    sg = _core.load("attn.size_gate")
    return sg.SizeGate(name="mkdit", min_tokens=min_tokens)


def addon_home(opt_root: str) -> str:
    return os.path.join(opt_root, ADDON)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_files(opt_root: str) -> dict:
    """The carried kernel files are present: {relative path under the add-on: sha256} (the hashes are reporting evidence — what this
    process actually loaded — not a check against a stored manifest). Raises :class:`MkditRefused` naming
    an absent file."""
    home = addon_home(opt_root)
    out = {}
    for rel in KERNEL_FILES:
        p = os.path.join(home, rel)
        if not os.path.isfile(p):
            raise MkditRefused(f"mkdit: carried kernel file {p} is absent")
        out[rel] = _sha256(p)
    return out


def _graph_flags():
    """``rf3.graph_flags`` of the patched tree, or MkditRefused (the pristine tree carries no hoist cache)."""
    try:
        import rf3.graph_flags as GF                      # noqa: N812
    except ImportError as e:
        raise MkditRefused(f"mkdit: rf3.graph_flags does not import ({type(e).__name__}: {e}): the interpreter's rf3 is not the patched tree") from e
    return GF


def _kernel_module(home: str):
    """Import the carried megakernel module from the add-on's kernel directory (mk2 imports mkrf3 by name from the same directory)."""
    d = os.path.join(home, KERNEL_DIR)
    if d not in sys.path:
        sys.path.insert(0, d)
    try:
        import triton  # noqa: F401
    except ImportError as e:
        raise MkditRefused(f"mkdit: triton does not import ({e}): the megakernel is a Triton kernel") from e
    try:
        import importlib
        mod = importlib.import_module(KERNEL)
    except ImportError as e:
        raise MkditRefused(f"mkdit: {KERNEL} does not import from {d} ({type(e).__name__}: {e})") from e
    impl = os.path.realpath(getattr(mod, "__file__", "") or "")
    if not impl.startswith(os.path.realpath(d) + os.sep):
        raise MkditRefused(f"mkdit: {KERNEL} executes from {impl}, not the carried copy under {d}")
    if not callable(getattr(mod, "MK2TokenTransformer", None)):
        raise MkditRefused(f"mkdit: {impl} exposes no MK2TokenTransformer")
    return mod


def config_for(cc) -> Optional[dict]:
    """The tile configuration for a card of compute capability ``cc`` (a tuple): the ``CONFIGS`` row of the highest key at or below it; None below 8.0."""
    keys = [k for k in CONFIGS if cc is not None and tuple(cc) >= k]
    return dict(CONFIGS[max(keys)]) if keys else None


def _new_mk(tok):
    """One MK2TokenTransformer per DiffusionTransformer instance (weights packed once; the SM count read from the input's card; the tile
    configuration selected at enable for the visible card)."""
    mod = sys.modules.get(KERNEL) or _kernel_module(STATE["home"])
    return mod.MK2TokenTransformer(tok, **(STATE.get("config") or CONFIG))


def _cc_tuple(cc):
    """nvidia-smi's compute capability word (``"9.0"`` / ``9.0`` / ``(9, 0)``) as a tuple, or None when unreadable."""
    if isinstance(cc, (tuple, list)) and len(cc) >= 2:
        return (int(cc[0]), int(cc[1]))
    m = re.match(r"^\s*(\d+)\.(\d+)\s*$", str(cc)) if cc is not None else None
    return (int(m.group(1)), int(m.group(2))) if m else None


def _check_device(t) -> None:
    import torch
    if not t.is_cuda:
        raise MkditRefused(f"mkdit: the token transformer's input is on {t.device}: the megakernel runs on a CUDA device only (--mode exact / --mode off elsewhere)")
    cap = torch.cuda.get_device_capability(t.device)
    if tuple(cap) < MIN_CAPABILITY:
        raise MkditRefused(f"mkdit: compute capability {cap[0]}.{cap[1]} < {MIN_CAPABILITY[0]}.{MIN_CAPABILITY[1]}: the megakernel has no tile configuration "
                           f"for this card (CONFIGS: sm_80 and up; --mode exact serves this card)")


def forward(self, A_I, S_I, Z_II, Beta_II):
    """``DiffusionTransformer.forward``, class-wide. Atom transformers (``Beta_II`` given): the class's own forward. Token transformer: below the
    size gate the class's own forward (counted ``gated``); else one megakernel launch per leading-batch sample, the pair bias hoisted once per
    roll-out — ``A_I`` ``[I, C]`` or ``[D, I, C]``, ``S_I`` ``[I, Cs]`` / ``[1, I, Cs]`` / ``[D, I, Cs]``, ``Z_II`` shared by the batch."""
    if Beta_II is not None:                                   # the atom transformers (windowed attention): not this lever's site, by design
        STATE["atom_calls"] += 1
        return _ORIG["forward"](self, A_I, S_I, Z_II, Beta_II)
    if A_I.dim() == 2:
        A3 = A_I.unsqueeze(0)
    elif A_I.dim() == 3:
        A3 = A_I
    else:
        STATE["error"] = f"A_I rank {A_I.dim()} shape {tuple(A_I.shape)}: expected [I, C] or [D, I, C]"
        raise MkditRefused("mkdit: " + STATE["error"])
    D, I, _ = (int(s) for s in A3.shape)
    if not GATE.decide(I).served:                              # below MIN_I: the stock blocks, counted `gated`
        return _ORIG["forward"](self, A_I, S_I, Z_II, Beta_II)
    try:                                                       # no handler: a raise inside a served call propagates (the fold dies by name); `finally` records it for the exit tally
        _check_device(A_I)
        if S_I.dim() == 2:
            S3 = S_I.unsqueeze(0).expand(D, -1, -1)
        elif S_I.dim() == 3 and int(S_I.shape[0]) in (1, D):
            S3 = S_I if int(S_I.shape[0]) == D else S_I.expand(D, -1, -1)
        else:
            raise MkditRefused(f"mkdit: S_I shape {tuple(S_I.shape)} beside A_I {tuple(A_I.shape)}: expected [I, Cs], [1, I, Cs] or [D, I, Cs]")
        import torch
        GF = sys.modules.get("rf3.graph_flags") or _graph_flags()
        mk = _OBJS.get(id(self))
        if mk is None:
            mk = _new_mk(self)
            _OBJS[id(self)] = mk
            STATE["objs"] += 1

        def _hoist():                                          # to_b(ln_0(Z_II)) of every block -> [24, H, I, I] bf16; a pair bias with its own batch raises here (the kernel's assert), never a silent branch
            STATE["hoists"] += 1
            return mk.hoist(Z_II)
        Bt = GF.hoist_get((id(self), "mkdit_pair_bias"), _hoist)
        mk.Bt = Bt
        mk.I = int(Bt.shape[-1])
        if mk.I != I:
            raise MkditRefused(f"mkdit: the roll-out's hoisted pair bias is for I={mk.I}, this call has I={I}")
        outs = [mk.forward(A3[d], S3[d]) for d in range(D)]     # one launch per sample, same stream, fixed order (deterministic)
        out = torch.stack(outs, 0) if A_I.dim() == 3 else outs[0]
    finally:
        exc = sys.exc_info()[1]
        if exc is not None:
            STATE["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    STATE["samples"] += D
    key = f"D{D}_I{I}"
    STATE["shapes"][key] = STATE["shapes"].get(key, 0) + 1
    return out


def enable(opt_root: Optional[str] = None, gpu: Optional[dict] = None) -> dict:
    """Install the lever in this process (once): a visible GPU of compute capability >= 8.0 (nvidia-smi, no CUDA initialisation), the patched
    tree with ``RF3_HOIST`` on, no ``dtk`` / ``dattn`` installed, the carried kernel files present, Triton and the kernel
    module importable. Returns :func:`describe`. Raises :class:`MkditRefused` naming the failed precondition. Touches no CUDA state (the first
    served call packs the weights and compiles). ``gpu``: the probe record ``{name, cc, ...}`` (``stack.gpu_info()`` when None)."""
    global GATE
    if STATE["on"]:
        return describe()
    from . import dtk as _dtk
    from . import stack as _stack
    root = opt_root or _stack.opt_root()
    g = gpu if gpu is not None else _stack.gpu_info()
    if not g or not g.get("name"):
        raise MkditRefused("mkdit: no CUDA GPU visible (nvidia-smi): the megakernel needs one (--mode off runs the stock CLI)")
    cc = _cc_tuple(g.get("cc"))
    STATE["gpu"] = {"name": g.get("name"), "cc": g.get("cc")}
    cfg = config_for(cc)
    if cfg is None:                                           # no tile configuration below MIN_CAPABILITY: the mode refuses by name (a mode is all of its levers)
        raise MkditRefused(f"mkdit: tile configurations exist for compute capability {', '.join(f'{a}.{b}' for a, b in sorted(CONFIGS))} and up; "
                           f"GPU {g.get('name')!r} is compute capability {g.get('cc')!r}: the megakernel cannot run here (--mode exact serves this card)")
    STATE["config"] = cfg                                     # this card's row (sm_90 on H100 / B200, sm_80 on A100): the MKDIT line's config= words
    GF = _graph_flags()
    if not getattr(GF, "HOIST", False):
        raise MkditRefused("mkdit: RF3_HOIST is off in this process: the per-block pair bias [24, H, I, I] rides rf3.graph_flags' per-roll-out cache "
                           "(RF3_HOIST=1, every row naming mkdit exports it); without it the layout would be rebuilt at every denoiser call")
    if _dtk.STATE.get("on"):
        raise MkditRefused("mkdit: dtk is installed in this process: mkdit replaces the diffusion-transformer blocks whose attention dtk serves — a row names one of them")
    adp = sys.modules.get(ADAPTER)
    if adp is not None and (getattr(adp, "DATTN", None) or {}).get("on"):
        raise MkditRefused("mkdit: the FPF add-on's dattn is on in this process: it serves the attention inside the blocks mkdit replaces — a row names one of them")
    GATE = make_gate()
    STATE["home"] = addon_home(root)
    STATE["files_sha256"] = check_files(root)
    mod = _kernel_module(STATE["home"])
    STATE["impl"] = os.path.realpath(mod.__file__)
    try:
        import rf3.model.layers.af3_diffusion_transformer as DT   # noqa: N812
    except ImportError as e:
        raise MkditRefused(f"mkdit: rf3.model.layers.af3_diffusion_transformer does not import ({type(e).__name__}: {e})") from e
    cls = DT.DiffusionTransformer
    _ORIG.setdefault("forward", cls.forward)
    cls.forward = forward
    STATE["on"] = True
    return describe()


def census() -> Optional[dict]:
    if GATE is None:
        return None
    c = dict(GATE.census())
    c.update({"samples": STATE["samples"], "hoists": STATE["hoists"], "atom_calls": STATE["atom_calls"], "objs": STATE["objs"], "shapes": dict(STATE["shapes"])})
    return c


def problems() -> list:
    """The fail-closed gate as sentences (empty = clean): a recorded error; a seam that no token call reached while the lever was on."""
    if not STATE["on"] or GATE is None:
        return []
    out = list(GATE.problems(allow_fallback=False))
    if STATE["error"]:
        out.append(f"mkdit error: {STATE['error']}")
    if GATE.census()["calls"] == 0:
        out.append("no DiffusionTransformer token call reached the mkdit seam in this process (the forward is installed but never ran)")
    return out


def describe() -> dict:
    """The exit tally's ``mkdit`` block: on, impl, files_sha256, config, min_tokens, the census, ok + reason."""
    bad = problems()
    return {"on": STATE["on"], "impl": STATE["impl"], "origin": "kit", "files_sha256": dict(STATE["files_sha256"] or {}), "config": dict(STATE.get("config") or CONFIG), "gpu": STATE.get("gpu"),
            "min_tokens": (GATE.min_tokens if GATE is not None else None), "census": census(),
            "ok": STATE["on"] and not bad, "reason": ("; ".join(bad) if bad else (None if STATE["on"] else (STATE.get("reason") or "not installed")))}


def lever_evidence() -> list:
    """(key, value) pairs for the LEVER line after name/strategy/state: impl, origin, the kernel files' short shas, then the gate's own pairs."""
    shas = ",".join(f"{k.rsplit('/', 1)[-1]}:{v[:12]}" for k, v in sorted((STATE["files_sha256"] or {}).items()))
    pairs = [("impl", STATE["impl"]), ("origin", "kit"), ("files", shas or None), ("samples", STATE["samples"]), ("hoists", STATE["hoists"])]
    if GATE is not None:
        pairs += GATE.lever_evidence()
    return pairs
