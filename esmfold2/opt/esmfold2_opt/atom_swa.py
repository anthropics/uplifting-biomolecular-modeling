"""Banded evaluation of ESMFold2's sliding-window atom attention — the atom-count reach lever of ``pred --mode big --n_gpu P``
(sub-lever ``atom_swa`` of the kit's row-sharded line, :mod:`esmfold2_opt.rowpair`).

WHAT IT REPLACES. ``SWA3DRoPEAttention.forward`` (``modeling_esmfold2_common.py`` @ef32577f: class :547, forward :561-635) has three
branches, selected by the module constant ``FLASH_ATTN_AVAILABLE`` (:25-41; the package's reader is :mod:`esmfold2_opt.attn`). On the pinned
image (stock/PINS.json "image": flash-attn installed) every atom transformer of the model — the input embedder's ``atom_attention_encoder`` (:1972),
the structure module's ``atom_encoder`` (:1461) and ``atom_decoder`` (:1476), 3 ``SWAAtomBlock`` each — runs ``flash_attn_varlen_func``
(:578-604: the window inside the kernel, O(N) memory, no mask tensor) and this lever is NOT installed (gate ``flash_attn_present``, path
``stock_flash``). On a stack without flash-attn they run the DENSE fallback (:613-631), which is what this lever replaces. With ``valid[B, N]``
the atom mask (B = batch × diffusion samples, N = the padded atom count):

    rank    = cumsum(valid, 1) - 1                                        (:620)  int64 [B, N]
    within  = |rank[:, :, None] - rank[:, None, :]| <= half_window        (:621)  int64 [B, N, N] difference, its int64 abs, then bool
    allowed = within & valid[:, None, :] & valid[:, :, None]              (:622)  bool  [B, N, N] (+ 2 bool temporaries)
    allowed |= eye(N)                                                     (:623)  bool  [N, N]
    out = scaled_dot_product_attention(q, k, v, attn_mask=allowed[:, None], scale)     (:624-630)  over ALL N keys
    out = out * valid[..., None, None]                                    (:631)

MEMORY of the dense statement: a transient of 17·B·N² bytes at :621 (8 + 8 + 1 per element: :func:`dense_transient_bytes`), ≈ 3·B·N²
bytes of boolean masks alive into the SDPA call, plus the SDPA backend's own additive copy of the mask (bool → 0 / −inf in q.dtype:
2·B·N² bytes in bf16 on the memory-efficient backend; [B, H, N, N] weights on the math backend). It is REPLICATED on every rank whatever P
is (atom tensors are not sharded): 44k atoms (≈ 5.8k tokens) → 33 GB at :621, 59k atoms (≈ 7.8k tokens) → 59 GB — the reach ceiling of
the row-sharded line, independent of the pair shards.

WHAT RUNS INSTEAD. The SAME statements evaluated per query block ``[i0, i1)`` of ``q_block`` rows against only the key band the window
can reach: ``rank`` is
non-decreasing, so every key a query of the block may attend has ``rank`` in ``[rank[i0] − hw, rank[i1 − 1] + hw]``, i.e. index in
``[searchsorted(rank, lo, left), searchsorted(rank, hi, right))``; the block's own indices (the ``| eye`` diagonal) are inside; the band is
widened to multiples of ``key_align`` so that both band edges are key-tile edges of the SDPA kernel (the dense call's tiles start at
key 0). Inside the band the mask is :621-623 verbatim on the block; every key outside it is masked in the dense statement (its softmax
term is exp(−inf) = 0). ATTENDED SET: identical to the dense statement for every (b, query, key) — :func:`assert_band_covers` proves it
per block (package test), it is not assumed. NUMERICS: per-element arithmetic identical; the softmax reduction runs over the band instead
of N keys whose extra terms are exact zeros, so the result is bitwise the dense one whenever the kernel meets the same tile edges:
``key_align`` = 512 is a multiple of torch-CPU flash attention's 512-key tile (``torch.equal`` in fp32 and bf16 in the package test, every
case) and of the CUDA memory-efficient kernel's 64/128-key tiles; with edges off the tile the difference is the
online-softmax rescaling order (fp32 <= 5e-7, bf16 <= 1 ulp at the output scale, package test). On the card the kernel torch selects is
inside the big line's tier-2 word (not claimed bitwise here). MEMORY per block: 17·B·q_block·W bytes (:func:`banded_transient_bytes`) +
the SDPA call on [q_block × W], W ≤ q_block + 2·half_window + (invalid atoms inside that rank range) + 2·(key_align − 1)
(:func:`band_width_bound`; <= 3198 at the defaults, ≈ 0.11 GB per block at B = 1) — no tensor with N² elements is allocated (the package
test's numel guard).

REPLICATED BY DESIGN (named ``atoms`` in :data:`REPLICATED` and the census fields): atom tensors (q, k, v, the per-atom conditioning) and
this attention are window-local per atom and run WHOLE on every rank; no pair rows are read (ESMFold2's atom attention has no pair bias:
``DiffusionModule.forward`` :1509-1615 passes no ``z`` to the atom encoder / decoder), so nothing here communicates and no block size is
rank-derived (``q_block`` / ``key_align`` are explicit P-independent values, printed in the census).

INSTALL RULE (:func:`install`; the gate table of the module is :data:`GATES`). ``n_gpu > 1``: the banded form is installed unconditionally
— the class attribute ``SWA3DRoPEAttention.forward``, i.e. every instance of the process; it is the row-sharded line's one atom-window
implementation (no dense path, no switch); a stock forward whose source digest is not :data:`FORWARD_SOURCE_SHA256` is refused by name
(``opt_core.diffusion_loop.source_guard``), nothing patched. ``n_gpu = 1``: not installed (the single-GPU line is stock's own path). flash-attn importable in the model module
(the pinned image): not installed at any ``n_gpu``, named ``atom_swa=stock_flash`` (the stock flash branches :578-612 never build the dense
mask: ``flash_attn_varlen_func`` / ``flash_attn_func`` with ``window_size`` evaluate the window inside the kernel, O(N) memory) — the banded
form is the reach lever of stacks without flash-attn.
``EF2_ROWPAIR_ATOM_QBLOCK`` (default 2048) and ``EF2_ROWPAIR_ATOM_KALIGN`` (default 512: a multiple of the CUDA memory-efficient kernel's
128-key tile and of the torch-CPU kernel's 512-key tile, see NUMERICS) are explicit, P-independent block sizes; ``half_window`` is the
model's (64), never a knob. The banded forward reads its block plan on the host (one [B, N] int64 device→host copy per call): it is for
lines that capture no CUDA graph over the atom transformers (the memory mode's lines capture none).

INSTANCE FORWARDS (:func:`install`, gate ``instance_forward``). ``nn.Module.__call__`` resolves ``self.forward``: an attribute in an
INSTANCE's ``__dict__`` is served before the class attribute, so a class patch alone is dead code on every instance that carries one. The
kit's fast line installs exactly that — ``ef2_opt.install_swa_mask_cache`` (the U1 lever, on under every server mode that captures sampler
graphs or memoizes the SWA mask) rebinds ``m.forward = MethodType(_swa_forward_cached, m)`` on every ``SWA3DRoPEAttention`` and marks the
module ``_ef2opt_swa``; that forward builds and keeps the dense ``[B, N, N]`` masks. :func:`install` therefore patches the class AND rebinds
the ``forward`` of every instance so marked to the banded forward (both through the lever's :class:`PatchSet` records; :func:`uninstall`
serves U1's bound method and the stock class attribute again). An instance ``forward`` this module does not know (no ``_ef2opt_swa`` mark)
is refused BY NAME, nothing patched. What runs is reported per module, not per intent: ``atom_forward=`` in :func:`census_fields` is the
forward ``__call__`` resolves on each instance after install (:func:`forward_resolution`), and ``atom_rebound=`` counts the instance rebinds.
"""
from __future__ import annotations

import os
import sys
import types
from typing import Dict, List, Mapping, Optional, Tuple

from . import attn as _attn                                               # the package's one attention-path reader (forward_resolution, the words)
from opt_core.diffusion_loop.source_guard import source_guard         # the refuse-to-patch rule over the stock forward this module mirrors
from opt_core.mem.patchset import PatchSet                            # the family's install record (class-attribute patches, restore())
from opt_core.mem.rowpair import RowpairRefused                       # every refusal of this module is BY NAME through the family's class

LEVER = "atom_swa"                                                     # the sub-lever's name in the kit's row-sharded line (RowpairRefused.lever, PatchSet.lever)
CLASS_NAME = "SWA3DRoPEAttention"                                      # modeling_esmfold2_common.py:547 — the one class whose `forward` is replaced
ENV_QBLOCK = "EF2_ROWPAIR_ATOM_QBLOCK"                                 # query rows per block (positive int; default Q_BLOCK)
ENV_KALIGN = "EF2_ROWPAIR_ATOM_KALIGN"                                 # key-band alignment (positive int; default KEY_ALIGN)
Q_BLOCK = 2048                                                         # query rows per block: 17·B·Q_BLOCK·W bytes of mask transient per block (≈ 0.11 GB at B = 1)
KEY_ALIGN = 512                                                        # band edges on multiples of 512 keys: a multiple of the CUDA memory-efficient kernel's 64/128-key tiles and of torch-CPU flash attention's 512-key tile (band edges = the dense call's tile edges: torch.equal in the package test)
U1_MARK = "_ef2opt_swa"                                                # the attribute ef2_opt.install_swa_mask_cache sets on every instance whose `forward` it rebinds (driver/ef2_opt.py)
U1_FORWARD = "_swa_forward_cached"                                     # that instance forward's function name (reported by forward_resolution)
REPLICATED = ("atoms",)                                                # computed whole on every rank (module docstring, REPLICATED BY DESIGN)
FORWARD_SOURCE_SHA256 = ("66dd9ef1e6cf899e34f6e1abe1a56d91467e0ad37320823868ad0460376dd727",)   # opt_core.diffusion_loop.source_guard.source_sha256(SWA3DRoPEAttention.forward) @ef32577f (:561-635): the text _forward_banded mirrors
EXACT = ("attended_set=dense (proven per block); out-of-band softmax terms are exact zeros; bitwise iff both band edges are key-tile edges of the SDPA kernel "
         "(key_align a multiple of the tile: torch-CPU flash 512 -> torch.equal fp32+bf16 in the package test; edges off the tile: fp32 <= 5e-7, bf16 <= 1 ulp "
         "at the output scale); the CUDA kernel's tile on the kit's torch is inside the big line's tier-2 word, not claimed")
GATES = (                                                              # (gate, where it is decided, values, value in force under `--mode big --n_gpu P>1`, what the other value runs)
    ("atom_count", "resolve (none exists)", "no atom-count threshold: the path never depends on N", "banded at every N", "-"),
    ("flash_attn_present", "install (once, from the model module's FLASH_ATTN_AVAILABLE)", "True on the pinned image (flash-attn installed, stock/PINS.json 'image'); False on a stack without flash-attn", "False -> banded installed; True -> not installed, named atom_swa=stock_flash", "the stock flash branches (common.py:578-612: window inside the kernel, no dense mask, O(N) memory)"),
    ("valid_from_indices", "the forward, per call (common.py:614-619 verbatim)", "indices present -> valid mask; else all-valid", "unchanged (the stock semantics, not a size gate)", "-"),
    ("instance_forward", f"install (every {CLASS_NAME} instance's __dict__)", f"none | marked {U1_MARK} (the fast line's U1 forward) | foreign", f"none -> class patch serves it; {U1_MARK} -> the instance forward is rebound to the banded forward too; foreign REFUSED BY NAME", "an instance forward left in place shadows the class patch: U1's dense memoized [B,N,N] masks would run under the word banded"),
)

PATCHES = PatchSet(LEVER)                                              # SWA3DRoPEAttention.forward when installed; empty otherwise
INSTANCE_PATCHES = PatchSet(LEVER + ":instances")                      # the `forward` of every U1-marked instance when installed (restore() serves U1's bound method again)
STATE: dict = {"installed": False, "path": "dense_stock"}             # the install record (census_fields / the kit's manifest block read it)
_MODEL_MODULE = {"cmn": None}                                          # the module that defines the patched class (qk_norm, apply_rotary_emb_3d), bound at install


# ----------------------------------------------------------------------------------------------------------------- memory formulas
def dense_transient_bytes(n_atoms: int, b: int = 1) -> int:
    """Bytes transiently alive at common.py:621 of the dense statement: the int64 difference [B, N, N], its int64 ``abs`` and the bool
    comparison result (8 + 8 + 1 per element). The masks of :622-623 and the SDPA backend's additive copy come after and are smaller."""
    return 17 * int(b) * int(n_atoms) * int(n_atoms)


def banded_transient_bytes(q_block: int, band: int, b: int = 1) -> int:
    """The same statement's transient on one [q_block × band] block of the banded form."""
    return 17 * int(b) * int(q_block) * int(band)


def band_width_bound(q_block: int, half_window: int, key_align: int, pad: int = 0) -> int:
    """Upper bound of the key band W of one query block: the block's own keys, ``half_window`` reachable ranks on each side, ``pad``
    invalid (masked) atoms interleaved with that rank range (trailing padding counts for the blocks that reach it), and the alignment
    slack of at most ``key_align − 1`` on each end."""
    q_block, hw, ka, pad = int(q_block), int(half_window), int(key_align), int(pad)
    return q_block + 2 * hw + pad + 2 * (ka - 1)


# ----------------------------------------------------------------------------------------------------------------- the band (engine-free: tensors + ints)
def band_bounds(rank, i0: int, i1: int, hw: int, N: int, key_align: int) -> Tuple[int, int]:
    """Key index range ``[klo, khi)`` containing every key allowed for queries ``[i0, i1)`` under the window mask: ``rank`` =
    ``cumsum(valid, 1) - 1`` [B, N] (non-decreasing per row), so the allowed ranks ``[rank[i0] − hw, rank[i1 − 1] + hw]`` map to one index
    interval by binary search; the block's own indices are included (the diagonal term); both ends are rounded out to ``key_align``.
    Pass a CPU tensor to keep the searches off the device."""
    import torch
    B = rank.shape[0]
    lo = rank[:, i0].min() - hw
    hi = rank[:, i1 - 1].max() + hw
    klo, khi = N, 0
    for b in range(B):
        rb = rank[b].contiguous()
        klo = min(klo, int(torch.searchsorted(rb, lo.reshape(1), right=False).item()))
        khi = max(khi, int(torch.searchsorted(rb, hi.reshape(1), right=True).item()))
    klo = min(klo, i0)
    khi = max(khi, i1)
    klo = (klo // key_align) * key_align
    khi = min(N, -(-khi // key_align) * key_align)
    return klo, khi


def band_plan(rank, hw: int, q_block: int, key_align: int) -> List[Tuple[int, int, int, int]]:
    """``[(i0, i1, klo, khi), ...]`` for every query block of ``q_block`` rows over ``range(N)`` (:func:`band_bounds` per block). The
    searches run on a host copy of ``rank`` ([B, N] int64, one transfer) so a forward issues no per-block device synchronisation."""
    rank_h = rank.detach().to("cpu")
    N = int(rank_h.shape[1])
    plan = []
    for i0 in range(0, N, int(q_block)):
        i1 = min(i0 + int(q_block), N)
        klo, khi = band_bounds(rank_h, i0, i1, int(hw), N, int(key_align))
        plan.append((i0, i1, klo, khi))
    return plan


def allowed_block(rank, valid, i0: int, i1: int, klo: int, khi: int, hw: int):
    """common.py:621-623 restricted to the ``[i0:i1) × [klo:khi)`` block: bool [B, i1−i0, khi−klo]."""
    import torch
    rq = rank[:, i0:i1]
    rk = rank[:, klo:khi]
    within = (rq.unsqueeze(2) - rk.unsqueeze(1)).abs() <= hw
    allowed = within & valid[:, klo:khi].unsqueeze(1) & valid[:, i0:i1].unsqueeze(2)
    qi = torch.arange(i0, i1, device=valid.device)
    kj = torch.arange(klo, khi, device=valid.device)
    allowed |= (qi.unsqueeze(1) == kj.unsqueeze(0))
    return allowed


def banded_window_attention(q, k, v, valid, hw: int, scale: float, *, q_block: Optional[int] = None, key_align: Optional[int] = None):
    """``q, k, v`` [B, N, H, hd] (dtype as the stock forward cast them), ``valid`` [B, N] bool. Returns what
    ``F.scaled_dot_product_attention(qᵀ, kᵀ, vᵀ, attn_mask=allowed[:, None], scale).transpose(1, 2)`` returns in the stock fallback
    (common.py:624-630, BEFORE the ``* valid`` multiply of :631), computed block by block over the key band (the block bounds come
    from :func:`band_plan`)."""
    import torch
    import torch.nn.functional as F
    q_block = int(q_block or STATE.get("q_block") or Q_BLOCK)
    key_align = int(key_align or STATE.get("key_align") or KEY_ALIGN)
    rank = torch.cumsum(valid, dim=1) - 1                                          # common.py:620
    out = torch.empty_like(q)
    for i0, i1, klo, khi in band_plan(rank, hw, q_block, key_align):
        allowed = allowed_block(rank, valid, i0, i1, klo, khi, hw)
        o = F.scaled_dot_product_attention(
            q[:, i0:i1].transpose(1, 2),
            k[:, klo:khi].transpose(1, 2),
            v[:, klo:khi].transpose(1, 2),
            attn_mask=allowed.unsqueeze(1),
            scale=scale,
        ).transpose(1, 2)
        out[:, i0:i1] = o
        del allowed, o
    return out


def assert_band_covers(valid, hw: int, q_block: int = Q_BLOCK, key_align: int = KEY_ALIGN) -> int:
    """The attended-set proof on a case small enough to hold the dense mask: builds common.py:620-623 whole, and for every
    query block asserts (a) no allowed (query, key) lies outside the block's band and (b) :func:`allowed_block` equals the dense mask's
    block. Returns the largest band width seen. AssertionError names the failing block."""
    import torch
    B, N = valid.shape
    rank = torch.cumsum(valid, dim=1) - 1
    within = (rank.unsqueeze(2) - rank.unsqueeze(1)).abs() <= hw
    allowed = within & valid.unsqueeze(1) & valid.unsqueeze(2)
    allowed |= torch.eye(N, dtype=torch.bool, device=valid.device)
    maxw = 0
    for i0, i1, klo, khi in band_plan(rank, hw, q_block, key_align):
        blk = allowed[:, i0:i1]
        outside = blk.clone()
        outside[:, :, klo:khi] = False
        assert not bool(outside.any()), f"allowed key outside band for query block [{i0},{i1}): band [{klo},{khi})"
        inside = allowed_block(rank, valid, i0, i1, klo, khi, hw)
        assert torch.equal(inside, blk[:, :, klo:khi]), f"banded mask block != dense mask block for query block [{i0},{i1})"
        maxw = max(maxw, khi - klo)
    return maxw


# ----------------------------------------------------------------------------------------------------------------- the patched forward
def _forward_banded(self, x, attention_params: tuple):
    """``SWA3DRoPEAttention.forward`` (common.py:561-635) with the dense-mask fallback (:613-631) evaluated on key bands; every other
    statement verbatim. The flash-attn branches (:578-612) are not reproduced: :func:`install` patches only when the model module has
    no flash-attn."""
    import torch
    CMN = _MODEL_MODULE["cmn"] or sys.modules[type(self).__module__]
    B, N = x.shape[:2]
    cos, sin = attention_params[0], attention_params[1]

    x_input = x
    qkv = self.Wqkv(x)
    qkv = qkv.view(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 1, 3, 4)
    q, k, v = qkv.unbind(0)
    q, k = CMN.qk_norm(q), CMN.qk_norm(k)

    q = CMN.apply_rotary_emb_3d(q, cos, sin)
    k = CMN.apply_rotary_emb_3d(k, cos, sin)

    input_dtype = q.dtype
    if q.dtype not in (torch.float16, torch.bfloat16):
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()

    if len(attention_params) > 2:                                                    # common.py:614-619
        valid = torch.zeros(B * N, dtype=torch.bool, device=q.device)
        valid[attention_params[2]] = True
        valid = valid.view(B, N)
    else:
        valid = torch.ones(B, N, dtype=torch.bool, device=q.device)
    out = banded_window_attention(q, k, v, valid, self.half_window, self.scale)      # :620-630 on key bands
    out = out * valid.unsqueeze(-1).unsqueeze(-1)                                    # :631

    out = out.to(input_dtype).reshape(B, N, -1)                                      # :633-635
    out = out * torch.sigmoid(self.gate_proj(x_input))
    return self.out_proj(out)


# ----------------------------------------------------------------------------------------------------------------- install / census
def _env_int(environ: Mapping, name: str, default: int) -> int:
    raw = str(environ.get(name, "") or "").strip()
    if not raw:
        return int(default)
    try:
        val = int(raw)
    except ValueError:
        raise RowpairRefused(f"{name}={raw!r}: a positive integer is required", LEVER) from None
    if val < 1:
        raise RowpairRefused(f"{name}={raw!r}: a positive integer is required", LEVER)
    return val


def resolve(n_gpu: int) -> str:
    """``banded`` under ``n_gpu > 1`` (the row-sharded line's one atom-window implementation: every rank evaluates the window banded, never the
    dense [B, N_atom, N_atom] mask); ``dense`` at ``n_gpu = 1`` (= nothing installed: the single-GPU line is stock's own path)."""
    p = int(n_gpu)
    if p < 1:
        raise RowpairRefused(f"n_gpu={n_gpu!r}: a positive integer is required", LEVER)
    return "banded" if p > 1 else "dense"


def settings(environ: Optional[Mapping] = None, q_block: Optional[int] = None, key_align: Optional[int] = None) -> dict:
    """The block sizes in force: explicit arguments, else ``EF2_ROWPAIR_ATOM_QBLOCK`` / ``EF2_ROWPAIR_ATOM_KALIGN``, else :data:`Q_BLOCK`
    / :data:`KEY_ALIGN`. P-independent by construction (nothing here is derived from a rank's free memory)."""
    environ = os.environ if environ is None else environ
    return {"q_block": int(q_block) if q_block else _env_int(environ, ENV_QBLOCK, Q_BLOCK),
            "key_align": int(key_align) if key_align else _env_int(environ, ENV_KALIGN, KEY_ALIGN)}


def _swa_class(model):
    classes = {type(m) for m in model.modules() if type(m).__name__ == CLASS_NAME}
    if not classes:
        raise RowpairRefused(f"no {CLASS_NAME} module under {type(model).__name__}: the atom attention surface this lever patches is absent", LEVER)
    if len(classes) > 1:
        raise RowpairRefused(f"{len(classes)} distinct classes named {CLASS_NAME} under the model ({sorted(c.__module__ for c in classes)}): one model module expected", LEVER)
    cls = classes.pop()
    cmn = sys.modules.get(cls.__module__)
    if cmn is None:
        raise RowpairRefused(f"{cls.__module__} is not in sys.modules", LEVER)
    for name in ("qk_norm", "apply_rotary_emb_3d", "FLASH_ATTN_AVAILABLE"):
        if not hasattr(cmn, name):
            raise RowpairRefused(f"{cmn.__name__} has no attribute {name!r}: the atom attention surface moved", LEVER)
    probe = next(m for m in model.modules() if isinstance(m, cls))
    for name in ("Wqkv", "gate_proj", "out_proj", "half_window", "scale", "n_heads", "head_dim"):
        if not hasattr(probe, name):
            raise RowpairRefused(f"{CLASS_NAME} has no attribute {name!r}: the atom attention surface moved", LEVER)
    return cls, cmn


_forward_name = _attn._forward_name                                        # the function name behind a bound method / plain function stored as an instance ``forward``


def instance_forwards(instances) -> Tuple[list, list]:
    """Split the instances that carry ``forward`` in their own ``__dict__`` into (marked ``U1_MARK``, foreign). Instances without one are in
    neither list: the class attribute is what ``__call__`` resolves on them."""
    marked, foreign = [], []
    for m in instances:
        if "forward" in vars(m):
            (marked if getattr(m, U1_MARK, False) else foreign).append(m)
    return marked, foreign


def forward_resolution(instances) -> Dict[str, int]:
    """What ``nn.Module.__call__`` resolves ``self.forward`` to on each instance, counted by word: ``banded`` (this module's forward, as the
    class attribute or an instance rebinding), ``stock`` (the class's own stock forward), ``instance:<name>`` (an instance attribute that is
    not the banded forward — e.g. ``instance:_swa_forward_cached`` = U1's dense memoized masks), ``class:<name>`` (another class patch).
    The census itself is the package's one reader (attn.forward_resolution) with this module's word for its forward."""
    return _attn.forward_resolution(instances, {_forward_banded: _attn.BANDED_WORD})


def install(model, n_gpu: int, *, environ: Optional[Mapping] = None, q_block: Optional[int] = None, key_align: Optional[int] = None) -> dict:
    """Install the banded forward on ``SWA3DRoPEAttention`` (the class found under ``model``; every instance of the process) per the
    INSTALL RULE, and return the install record (also kept in :data:`STATE`): ``installed``, ``path`` (``banded`` | ``dense_stock`` |
    ``stock_flash``), ``q_block``, ``key_align``, ``half_window`` (the model's values), ``modules`` (instances found), ``sites``,
    ``rebound`` (instances whose U1 forward was rebound), ``forward_resolved`` (:func:`forward_resolution` after install), ``replicated``,
    ``exact``. Refusals are :class:`opt_core.mem.rowpair.RowpairRefused` with ``lever = "atom_swa"``."""
    environ = os.environ if environ is None else environ
    path = resolve(n_gpu)
    st = settings(environ, q_block, key_align)
    rec = {"installed": False, "path": "dense_stock", "n_gpu": int(n_gpu), **st, "half_window": [], "modules": 0, "sites": [],
           "replicated": list(REPLICATED), "exact": EXACT, "env": {k: environ.get(k) for k in (ENV_QBLOCK, ENV_KALIGN) if environ.get(k)}}
    if path == "dense":                                                            # n_gpu = 1: nothing is touched
        STATE.clear(); STATE.update(rec)
        return dict(rec)
    cls, cmn = _swa_class(model)
    inst = [m for m in model.modules() if isinstance(m, cls)]
    rec["half_window"] = sorted({int(m.half_window) for m in inst})
    rec["modules"] = len(inst)
    if bool(getattr(cmn, "FLASH_ATTN_AVAILABLE")):                                 # gate flash_attn_present: resolved once, named
        rec["path"] = "stock_flash"
        STATE.clear(); STATE.update(rec)
        return dict(rec)
    if len(PATCHES) or len(INSTANCE_PATCHES):
        raise RowpairRefused(f"already installed ({PATCHES.names()} + {len(INSTANCE_PATCHES)} instance forwards): uninstall() first", LEVER)
    gate = source_guard(cls.forward, expected_sha256=FORWARD_SOURCE_SHA256, name=f"source_guard:{CLASS_NAME}.forward")
    if not gate.ok:                                                                # a fork that moved the stock forward: refused by name, nothing patched
        raise RowpairRefused(str(gate.reason), LEVER)
    marked, foreign = instance_forwards(inst)                                      # gate instance_forward: an instance attribute shadows the class patch
    if foreign:
        names = sorted({_forward_name(vars(m)["forward"]) for m in foreign})
        raise RowpairRefused(f"{len(foreign)} of {len(inst)} {CLASS_NAME} instances carry an instance-level forward this lever does not know "
                             f"({names}; no {U1_MARK} mark): it would shadow the banded class forward — refusing to patch, nothing installed", LEVER)
    STATE.clear(); STATE.update(rec)                                               # q_block / key_align are read by the forward from STATE
    _MODEL_MODULE["cmn"] = cmn
    PATCHES.replace(cls, "forward", _forward_banded)
    for m in marked:                                                               # the fast line's U1 forward on this instance: rebound to the banded forward (restore() serves U1's again)
        INSTANCE_PATCHES.replace(m, "forward", types.MethodType(_forward_banded, m))
    resolved = forward_resolution(inst)
    STATE.update(installed=True, path="banded", sites=PATCHES.names() + ([f"{CLASS_NAME}().forward x{len(marked)}"] if marked else []),
                 rebound=len(marked), forward_resolved=resolved)
    if set(resolved) != {"banded"}:                                                # by construction unreachable; checked, not assumed — everything restored before the refusal
        uninstall()
        raise RowpairRefused(f"after install {CLASS_NAME} instances resolve forward to {resolved}, not banded on every module", LEVER)
    return dict(STATE)


def uninstall() -> List[str]:
    """Restore every rebound instance ``forward`` (U1's bound methods are served again) and the stock class attribute; returns the restored
    site names."""
    names = [f"{CLASS_NAME}().forward" for _ in INSTANCE_PATCHES.restore()] + PATCHES.restore()   # instance restores first (U1's bound methods back), then the class attribute
    STATE.clear(); STATE.update({"installed": False, "path": "dense_stock"})
    _MODEL_MODULE["cmn"] = None
    return names


def census_fields(rec: Optional[Mapping] = None) -> dict:
    """The sub-lever's words for the line's LEVER / census line: ``atom_swa=<path> atom_qblock= atom_kalign= atom_window= atom_modules=
    atom_forward=<resolved forward per instance, counted> atom_rebound=<U1 instance forwards rebound>``."""
    rec = STATE if rec is None else rec
    hw = rec.get("half_window") or []
    resolved = rec.get("forward_resolved") or {}
    return {"atom_swa": rec.get("path", "dense_stock"), "atom_qblock": rec.get("q_block", Q_BLOCK), "atom_kalign": rec.get("key_align", KEY_ALIGN),
            "atom_window": ",".join(str(2 * int(h)) for h in hw) or "-", "atom_modules": int(rec.get("modules", 0) or 0),
            "atom_forward": ",".join(f"{k}x{v}" for k, v in resolved.items()) or "-",      # what __call__ resolves per instance after install (banded xN when live)
            "atom_rebound": int(rec.get("rebound", 0) or 0)}


__all__ = ["LEVER", "ENV_QBLOCK", "ENV_KALIGN", "Q_BLOCK", "KEY_ALIGN", "REPLICATED", "EXACT", "GATES", "FORWARD_SOURCE_SHA256", "PATCHES", "INSTANCE_PATCHES", "STATE", "U1_MARK", "U1_FORWARD",
           "dense_transient_bytes", "banded_transient_bytes", "band_width_bound", "band_bounds", "band_plan", "allowed_block",
           "banded_window_attention", "assert_band_covers", "resolve", "settings", "instance_forwards", "forward_resolution", "install", "uninstall", "census_fields"]
