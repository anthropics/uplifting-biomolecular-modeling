"""
g3fast_patches — value-identical, host-sync-free replacements for five constructs in Genie 3's denoiser forward.

Stock forward performs, per denoiser call: CPU index-tensor construction + implicit H2D copies
(`batched_gather`, `sinusoidal_encoding`, `quat_to_rot`), and two device->host syncs
(`torch.sum(token_mask[i]).long()` used as a Python slice bound in `_compute_frenet_frames`;
`range(1, 1 + batch["cond_group"].max())` in the pair embedder).  These make the forward un-capturable by
CUDA graphs and add host round trips.  Each replacement below produces bit-identical tensors for every valid
token (eager patched vs eager stock: max|delta| == 0), so applying them is
numerics class 1.  `apply()` monkeypatches the loaded genie3 modules in place, in memory; the checkout's source
files are never edited.
"""
from __future__ import annotations

import inspect
import textwrap

import torch
import torch.nn.functional as F


def _batched_gather(data: torch.Tensor, inds: torch.Tensor, dim: int = 0, no_batch_dims: int = 0) -> torch.Tensor:
    # stock builds `torch.arange(s)` on the CPU and lets advanced indexing copy it to the GPU (H2D per call)
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s, device=inds.device)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)
    remaining_dims = [slice(None) for _ in range(len(data.shape) - no_batch_dims)]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[tuple(ranges)]


def _sinusoidal_encoding(v: torch.Tensor, N: int, D: int) -> torch.Tensor:
    import math
    k = torch.arange(1, D + 1, device=v.device)                  # stock: torch.arange(1, D + 1).to(v.device)
    sin_div_term = N ** (2 * k / D)
    sin_div_term = sin_div_term.view(*((1,) * len(v.shape) + (len(sin_div_term),)))
    sin_enc = torch.sin(v.unsqueeze(-1) * math.pi / sin_div_term)
    cos_div_term = N ** (2 * (k - 1) / D)
    cos_div_term = cos_div_term.view(*((1,) * len(v.shape) + (len(cos_div_term),)))
    cos_enc = torch.cos(v.unsqueeze(-1) * math.pi / cos_div_term)
    enc = torch.zeros_like(sin_enc)
    enc[..., 0::2] = cos_enc[..., 0::2]
    enc[..., 1::2] = sin_enc[..., 1::2]
    return enc


_QTR_CACHE = {}


def _quat_to_rot(quat: torch.Tensor) -> torch.Tensor:
    from genie3.generation.utils import affine_utils as au
    key = (quat.device, quat.dtype)
    m = _QTR_CACHE.get(key)
    if m is None:
        m = au._qtr_mat.to(device=quat.device)                    # stock: .to(quat.device) on every call (H2D)
        _QTR_CACHE[key] = m
    quat = quat[..., None] * quat[..., None, :]
    shaped_qtr_mat = m.view((1,) * len(quat.shape[:-2]) + (4, 4, 3, 3))
    quat = quat[..., None, None] * shaped_qtr_mat
    return torch.sum(quat, dim=(-3, -4))


def _compute_frenet_frames(token_mask, trans_mask, rot_mask, trans_atom_positions, rot_catom_positions,
                           rot_latom_positions, rot_ratom_positions, legacy: bool = False):
    """Stock computes rotations for tokens [:l] with l = int(token_mask[i].sum()) (a device->host sync and a
    Python loop over the batch) and leaves zeros for tokens >= l.  Valid tokens are always a prefix (padding is
    appended), so computing all tokens elementwise and zeroing tokens >= l gives identical values: the per-token
    math (cross products, F.normalize with eps clamp) is independent across tokens; NaN cannot arise (0/eps=0)."""
    from genie3.generation.utils.geo_utils import _compute_frenet_rotations
    from genie3.generation.utils.affine_utils import T
    n_token = token_mask.shape[1]
    l = torch.sum(token_mask, dim=1).long()                        # [B] on device
    prefix = (torch.arange(n_token, device=token_mask.device)[None, :] < l[:, None]).to(rot_catom_positions.dtype)
    rots = _compute_frenet_rotations(catom=rot_catom_positions, latom=rot_latom_positions,
                                      ratom=rot_ratom_positions, legacy=legacy)
    rots = rots * prefix[..., None, None]
    trans = trans_atom_positions * trans_mask[..., None]
    rots = rots * rot_mask[..., None, None]
    return T(rots, trans)


def _cond_group_max(batch):
    v = batch.get("_g3fast_cond_group_max", None)
    if v is not None:
        return v                                                   # python int precomputed once per batch
    return batch["cond_group"].max()                               # stock behaviour (device->host sync)


def apply(verbose: bool = True):
    """Monkeypatch the loaded genie3 modules. Idempotent."""
    from genie3.generation.utils import tensor_utils, geo_utils, encode_utils, affine_utils
    from genie3.generation.model.embedder.single import v1 as single_v1
    from genie3.generation.model.embedder.pair import v1 as pair_v1
    from genie3.generation.model.module import backbone_update

    tensor_utils.batched_gather = _batched_gather
    geo_utils.batched_gather = _batched_gather
    encode_utils.sinusoidal_encoding = _sinusoidal_encoding
    single_v1.sinusoidal_encoding = _sinusoidal_encoding
    affine_utils.quat_to_rot = _quat_to_rot
    backbone_update.quat_to_rot = _quat_to_rot
    geo_utils._compute_frenet_frames = _compute_frenet_frames

    # pair embedder: replace the host-synchronising `.max()` loop bound; everything else verbatim
    cls = pair_v1.V1PairFeatureNet
    if not getattr(cls, "_g3fast_patched", False):
        src = textwrap.dedent(inspect.getsource(cls.forward))
        needle = 'batch["cond_group"].max()'
        assert needle in src, "V1PairFeatureNet.forward changed upstream; update g3fast_patches"
        src = src.replace(needle, "_cond_group_max(batch)")
        ns = dict(pair_v1.__dict__)
        ns["_cond_group_max"] = _cond_group_max
        exec(compile(src, "<g3fast_patched_V1PairFeatureNet.forward>", "exec"), ns)
        cls.forward = ns["forward"]
        cls._g3fast_patched = True
    if verbose:
        print("[g3fast_patches] applied: batched_gather, sinusoidal_encoding, quat_to_rot, _compute_frenet_frames, pair cond_group.max")


def add_static_ints(batch: dict) -> dict:
    """Precompute the python ints the patched forward can use instead of device->host syncs (once per batch)."""
    batch["_g3fast_cond_group_max"] = int(batch["cond_group"].max().item())
    return batch
