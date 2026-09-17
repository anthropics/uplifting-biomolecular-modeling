import math
from functools import partial

import numpy as np
import torch
from rfd3.model import cudagraph_sampler as _cg
import torch.nn as nn
from torch.nn.functional import silu

from foundry.training.checkpoint import activation_checkpointing
from foundry.utils.ddp import RankedLogger

ranked_logger = RankedLogger(__name__, rank_zero_only=True)
try:
    from apex.normalization.fused_layer_norm import FusedRMSNorm

    ranked_logger.info("Fused RMSNorm enabled!")
    RMSNorm_ = FusedRMSNorm
except (ImportError, ModuleNotFoundError):
    ranked_logger.warning(
        "Using nn.RMSNorm instead of apex.normalization.fused_layer_norm.FusedRMSNorm."
        "Ensure you're using the correct apptainer"
    )
    RMSNorm_ = nn.RMSNorm


# Allow bias=False to be passed for RMSNorm
def RMSNorm(*args, **kwargs):
    if "bias" in kwargs:
        kwargs.pop("bias")
    return RMSNorm_(*args, **kwargs)


SWAP_LAYER_NORM_FOR_RMS_NORM = True
RMSNorm = RMSNorm if SWAP_LAYER_NORM_FOR_RMS_NORM else nn.LayerNorm
linearNoBias = partial(torch.nn.Linear, bias=False)


class EmbeddingLayer(nn.Linear):
    """
    Specialized linear layer for correct weight initialization for embedding layers.

    Embedding layers are functionally a multiplication of an N channel input by an NxC weight matrix to produce an
    embedding of length C. However, we compute the components separately with a ModuleDict, then sum at the end, for
    embedding reusability and interoperability purposes.

    This layer uses Xavier initialization as described in [1]_.

    References
    ----------
    .. [1] Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty
           of training deep feedforward neural networks." (2010)
           http://proceedings.mlr.press/v9/glorot10a.html
    """

    def __init__(
        self,
        this_in_features,
        total_embedding_features,
        out_features,
        device=None,
        dtype=None,
    ):
        self.total_embedding_features = total_embedding_features
        self.out_features = out_features
        super().__init__(
            this_in_features, out_features, bias=False, device=device, dtype=dtype
        )
        self.reset_parameters()

    def reset_parameters(self, **kwargs):
        super().reset_parameters()
        a = math.sqrt(6.0 / float(self.total_embedding_features + self.out_features))
        nn.init._no_grad_uniform_(self.weight, -a, a)


def collapse(x, L):
    return x.reshape((L, x.numel() // L))


class MultiDimLinear(nn.Linear):
    def __init__(self, in_features, out_shape, norm=False, **kwargs):
        self.out_shape = out_shape
        out_features = np.prod(out_shape)
        super().__init__(in_features, out_features, **kwargs)
        if norm:
            self.ln = RMSNorm((out_features,))
            self.use_ln = True
        else:
            self.use_ln = False
        self.reset_parameters()

    def reset_parameters(self, **kwargs) -> None:
        super().reset_parameters()
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        out = super().forward(x)
        if self.use_ln:
            out = self.ln(out)
        return out.reshape(x.shape[:-1] + self.out_shape)


class LinearBiasInit(nn.Linear):
    def __init__(self, *args, biasinit, **kwargs):
        assert biasinit == -2.0  # Sanity check
        self.biasinit = biasinit
        super().__init__(*args, **kwargs)

    def reset_parameters(self) -> None:
        super().reset_parameters()
        self.bias.data.fill_(self.biasinit)


class Transition(nn.Module):
    def __init__(self, n, c):
        super().__init__()
        self.layer_norm_1 = RMSNorm(c)
        self.linear_1 = linearNoBias(c, n * c)
        self.linear_2 = linearNoBias(c, n * c)
        self.linear_3 = linearNoBias(n * c, c)

    def forward(self, X):
        # [RFD3_CUDAGRAPH] checkpoint wrapper bypassed inside the captured step (memory-only device)
        if _cg.graph_mode():
            return self._forward_impl(X)
        return self._forward_ckpt(X)

    def _forward_impl(
        self,
        X,
    ):
        X = self.layer_norm_1(X)
        A = self.linear_1(X)
        B = self.linear_2(X)
        X = self.linear_3(silu(A) * B)
        return X

    @activation_checkpointing
    def _forward_ckpt(self, X):
        return self._forward_impl(X)


# [RFD3_FZT] fused SwiGLU transition. Read once, here, at import (as rfd3/model/hoist.py reads RFD3_HOIST): RFD3_FZT=1 routes
# Transition._forward_impl through opt_core's fused pair-track transition cell (opt_core.attn.pair_fused.transition, impl 'fpf',
# ln='stock'): the module's own norm runs unchanged; its output, cast to bf16 exactly as autocast casts it at linear_1 / linear_2,
# is the kernel's normalised input; ONE Triton kernel then computes bf16(fp32-acc(y @ W1^T)), bf16(fp32-acc(y @ W2^T)),
# h = bf16(silu(a) * b) in fp32 opmath, out = bf16(fp32-acc over ascending hidden chunks of h @ W3^T) — the stock rounding points,
# the n*c hidden never written to memory. Served: the (c, n*c) cells of FZT_CELLS on CUDA tensors under bf16 autocast in eval mode
# (the diffusion module's inference context) at a row count inside the running card's range (FZT_ROWS_BY_CC below: every row count
# on 9.0); every other call runs the stock body above and is counted by reason (FZT_STATS).
# A cell of FZT_CELLS the core's table (opt_core/attn/pair_fused_cells.json, keyed by compute capability and Triton line) does not
# serve on this stack is a NAMED fallback, never a silent one and never fatal: the core's refusal words (pair_fused.Unsupported
# 'no-cell:fpf:transition:<C>x<NH>:<cc>|<triton>') are recorded once per cell in FZT_UNSERVED and logged, that cell's calls run the
# stock body counted under cells_stock['no-cell:<C>x<NH>'], and the package names it on its FALLBACK line and treats the activation as
# partial (rfdiffusion3_opt.stack.report_fzt). Every other refusal of a served call (an uncovered dtype / layout) or a launch error
# propagates. Default off = the code above, byte for byte in effect.
import os as _os

FZT = _os.environ.get("RFD3_FZT", "0") == "1"
FZT_CELLS = ((128, 256), (128, 512), (256, 512), (256, 1024))
# [RFD3_FZT] served row range by compute capability. The kernel's summation order is the one the stock Linear layers' cuBLAS bf16 GEMMs
# use at every row count on compute capability 9.0 — no 9.0 row: every row count of a served cell is fused there, as before. On 8.0
# (A100) the pinned stack's cuBLAS takes other orders in row-count bands below these floors (measured per cell on torch 2.13.0+cu130:
# 128x256 at 641-3264, 5631-5856 and 12544-13359 rows; 128x512 at 193-3136; 256x512 up to 1472; 256x1024 up to 1600) and from
# 2^20-16 rows up, where it splits the row dimension — so a call whose row count (the product of the leading dims: L*L and B*L*L for
# the pair track) lies outside its cell's (min_rows, max_rows) runs the stock body BY NAME, counted cells_stock['rows<MIN:<cell>'] /
# cells_stock['rows>MAX:<cell>']: a stock route by design (exact == off bitwise at every size on that card), not a fallback and not a
# partial activation; fzt_describe() names the card's ranges (rows_cc=, rows_served=) when the running device has a row here. The
# capability is read from the tensor's device (torch.cuda.get_device_capability), never from the configuration's target-GPU word.
FZT_ROWS_BY_CC = {"8.0": {(128, 256): (16384, 1048544), (128, 512): (4096, 1048544), (256, 512): (4096, 1048544), (256, 1024): (4096, 1048544)}}
FZT_STATS = {"RFD3_FZT": FZT, "installs": 0, "fused_calls": 0, "stock_calls": 0, "cells_fused": {}, "cells_stock": {}, "modules_packed": 0}
FZT_UNSERVED = {}                      # "<C>x<NH>" -> the core's refusal words for a cell of FZT_CELLS its table does not serve on this stack (named once, then stock)
_FZT_WEIGHTS = {}
_FZT_CC = {}                           # CUDA device index -> "M.m", read once per device (a host query, made at the eager warm-up's first call)
_FZT_PF = None
_transition_forward_stock = Transition._forward_impl


def _fzt_count(kind, cell):
    FZT_STATS[kind + "_calls"] += 1
    d = FZT_STATS["cells_" + kind]
    d[cell] = d.get(cell, 0) + 1


def _fzt_cc(device):
    """The compute capability "M.m" of the CUDA `device`, memoised by index."""
    i = device.index if device.index is not None else torch.cuda.current_device()
    cc = _FZT_CC.get(i)
    if cc is None:
        cc = _FZT_CC[i] = "%d.%d" % tuple(torch.cuda.get_device_capability(i))
    return cc


def _fzt_rows_word(cell, rows, cc):
    """None when `rows` rows of `cell` are served on compute capability `cc` (no row range on that card, or inside it); else the census
    word of the stock route by name: 'rows<MIN' / 'rows>MAX'."""
    rng = FZT_ROWS_BY_CC.get(cc, {}).get(cell)
    if rng is None:
        return None
    if rows < rng[0]:
        return f"rows<{rng[0]}"
    if rows > rng[1]:
        return f"rows>{rng[1]}"
    return None


def _fzt_rows_fields():
    """The row-range evidence of this process: {} when no device seen so far has a row in FZT_ROWS_BY_CC (every row count served), else
    {'rows_cc': 'M.m', 'rows_served': '<C>x<NH>:<min>-<max>,…'} for the first such capability."""
    for cc in sorted(set(_FZT_CC.values())):
        table = FZT_ROWS_BY_CC.get(cc)
        if table:
            return {"rows_cc": cc, "rows_served": ",".join(f"{c}x{h}:{table[(c, h)][0]}-{table[(c, h)][1]}" for c, h in FZT_CELLS if (c, h) in table)}
    return {}


def _fzt_autocast_bf16():
    try:
        on = torch.is_autocast_enabled("cuda")
    except TypeError:
        on = torch.is_autocast_enabled()
    if not on:
        return False
    dt = torch.get_autocast_dtype("cuda") if hasattr(torch, "get_autocast_dtype") else torch.get_autocast_gpu_dtype()
    return dt == torch.bfloat16


def _fzt_weights(mod, device):
    T = _FZT_WEIGHTS.get(id(mod))
    if T is None or T.dev != device:
        eps = getattr(mod.layer_norm_1, "eps", None)
        T = _FZT_PF.pack_transition_weights(w_out=mod.linear_3.weight, w_a=mod.linear_1.weight, w_b=mod.linear_2.weight,
                                           eps=float(eps if eps is not None else torch.finfo(torch.float32).eps), device=device)
        _FZT_WEIGHTS[id(mod)] = T
        FZT_STATS["modules_packed"] += 1
    return T


def _fzt_unserved(key, words):
    """Record, once per cell, that the core serves no fused transition cell for `key` on this stack; later calls of the cell take the
    stock body before any packing or planning."""
    if key not in FZT_UNSERVED:
        FZT_UNSERVED[key] = str(words)
        ranked_logger.warning("[RFD3_FZT] no fused transition cell for %s on this stack (%s): its calls run the stock transition, counted as cells_stock['no-cell:%s']", key, words, key)


def _transition_forward_fzt(self, X):
    C = int(X.shape[-1])
    cell = (C, int(self.linear_1.weight.shape[0]))
    key = f"{cell[0]}x{cell[1]}"
    if cell not in FZT_CELLS:
        _fzt_count("stock", "cell:" + key); return _transition_forward_stock(self, X)
    if key in FZT_UNSERVED:
        _fzt_count("stock", "no-cell:" + key); return _transition_forward_stock(self, X)
    if not X.is_cuda:
        _fzt_count("stock", "not-cuda:" + key); return _transition_forward_stock(self, X)
    if self.training:
        _fzt_count("stock", "training:" + key); return _transition_forward_stock(self, X)
    if X.dtype not in (torch.float32, torch.bfloat16):
        _fzt_count("stock", f"dtype-{X.dtype}:" + key); return _transition_forward_stock(self, X)
    if not _fzt_autocast_bf16():
        _fzt_count("stock", "no-bf16-autocast:" + key); return _transition_forward_stock(self, X)
    rows_word = _fzt_rows_word(cell, X.numel() // C, _fzt_cc(X.device))   # [RFD3_FZT] the card's served row range (FZT_ROWS_BY_CC): outside it the stock body, by name
    if rows_word is not None:
        _fzt_count("stock", rows_word + ":" + key); return _transition_forward_stock(self, X)
    T = _fzt_weights(self, X.device)
    Y = self.layer_norm_1(X)
    Y = Y if Y.dtype == torch.bfloat16 else Y.to(torch.bfloat16)
    Y2 = Y.reshape(-1, C)
    if Y2.stride(-1) != 1 or (Y2.shape[0] > 1 and Y2.stride(0) != C):
        Y2 = Y2.contiguous()
    words = None                                                  # the refusal WORDS only: an exception kept alive pins this frame's tensors (opt_core pair_fused.pick_cell)
    try:
        out = _FZT_PF.transition(Y2, T, residual=False, ln="stock", x_ln=Y2, impl="fpf")
    except _FZT_PF.Unsupported as _e:
        words = str(_e.reason)
        if not words.startswith("no-cell:"):                     # any other refusal of a served call propagates
            raise
    if words is not None:
        _fzt_unserved(key, words)
        _fzt_count("stock", "no-cell:" + key); return _transition_forward_stock(self, X)
    _fzt_count("fused", key)
    return out.reshape(tuple(X.shape[:-1]) + (C,))


def fzt_describe():
    """What this module read and what the route has done so far (the package's APPLIED / EXIT lines read it)."""
    d = dict(FZT_STATS)
    d["cells"] = ",".join(f"{c}x{h}" for c, h in FZT_CELLS)
    d.update(_fzt_rows_fields())                                  # rows_cc= rows_served= on a card with a row range (8.0); nothing on 9.0
    d["cells_unserved"] = dict(FZT_UNSERVED)
    if _FZT_PF is not None:
        d["opt_core_cells_sha256"] = _FZT_PF.cells_sha256()
    return d


if FZT:
    try:
        from opt_core.attn import pair_fused as _FZT_PF
    except ImportError as _e:
        raise ImportError("[RFD3_FZT] RFD3_FZT=1 needs opt_core importable in this interpreter (pip install -e common/opt_core): "
                          f"{type(_e).__name__}: {_e}") from _e
    from rfd3.model.hoist import nodynamo as _nodynamo  # [RFD3_COMPILE] inductor cannot re-emit the core's Triton kernel: the fused call is a dynamo boundary

    Transition._forward_impl = _nodynamo(_transition_forward_fzt)
    FZT_STATS["installs"] += 1
    ranked_logger.info("[RFD3_FZT] Transition._forward_impl routed through opt_core pair_fused transition (cells %s)", FZT_CELLS)



# [RFD3_INIT_CHUNK] row-blocked pair embedding of the token initialiser (layers/blocks.py SinusoidalDistEmbed, whose forward calls in here).
# Read once, here, at import, as RFD3_FZT above is: RFD3_INIT_CHUNK=1 bounds the fp32 intermediates of the two all-atom-pair embeddings init_atoms computes once per
# design batch (motif positions, reference positions) to INIT_CHUNK_PAIRS pairs per block instead of L_atom**2 at once.
INIT_CHUNK = _os.environ.get("RFD3_INIT_CHUNK", "0") == "1"
INIT_CHUNK_PAIRS = 1 << 20          # rows * L atom pairs per block: 1 Mi pairs -> the [rows, L, 64] fp32 tensor is 256 MiB (the whole fp32 family < 1 GiB)
INIT_CHUNK_STATS = {"RFD3_INIT_CHUNK": INIT_CHUNK, "calls": 0, "chunked_calls": 0, "blocks": 0, "max_atoms": 0}


def init_chunk_rows(L):
    """Rows per block for an L-atom structure: the largest count with rows * L <= INIT_CHUNK_PAIRS (at least 1); >= L means one block."""
    return max(1, INIT_CHUNK_PAIRS // max(1, int(L)))


def _sinusoidal_dist_embed_rows(mod, pos_q, pos, valid_mask_q, freq):
    """SinusoidalDistEmbed.forward's statements for the query rows pos_q against every atom of pos (the stock body, row-sliced)."""
    D_qL = pos_q.unsqueeze(-2) - pos.unsqueeze(-3)  # [rows, L, 3] or [B, rows, L, 3]
    dist_matrix = torch.linalg.norm(D_qL, dim=-1)
    angles = dist_matrix.unsqueeze(-1) * freq
    sin_embed = torch.sin(angles)
    cos_embed = torch.cos(angles)
    sincos_embed = torch.cat([sin_embed, cos_embed], dim=-1)
    P_qL = mod.output_proj(sincos_embed)
    P_qL = P_qL * valid_mask_q
    P_qL = P_qL + mod.process_valid_mask(valid_mask_q.to(P_qL.dtype)) * valid_mask_q
    return P_qL


def init_chunk_note(L):
    """One SinusoidalDistEmbed call under the switch (row-blocked or stock's single block): the tally the kit's APPLIED line reads."""
    INIT_CHUNK_STATS["calls"] += 1
    INIT_CHUNK_STATS["max_atoms"] = max(INIT_CHUNK_STATS["max_atoms"], int(L))


def sinusoidal_dist_embed_chunked(mod, pos, valid_mask):
    L = pos.shape[-2]
    rows = init_chunk_rows(L)
    half_dim = mod.n_freqs
    freq = torch.exp(
        -math.log(10000.0)
        * torch.arange(0, half_dim, dtype=torch.float32)
        / half_dim
    ).to(pos.device)  # [n_freqs] — the stock expression
    init_chunk_note(L); INIT_CHUNK_STATS["chunked_calls"] += 1
    P_LL = None
    for i0 in range(0, L, rows):
        P_qL = _sinusoidal_dist_embed_rows(mod, pos[..., i0:i0 + rows, :], pos, valid_mask[..., i0:i0 + rows, :, :], freq)
        if P_LL is None:
            P_LL = P_qL.new_empty(P_qL.shape[:-3] + (L,) + P_qL.shape[-2:])
        P_LL[..., i0:i0 + rows, :, :] = P_qL
        INIT_CHUNK_STATS["blocks"] += 1
        del P_qL
    return P_LL


def init_chunk_describe():
    """The switch and its process tally (read by the kit's report line): calls served in blocks, blocks run, the largest atom count seen."""
    return dict(INIT_CHUNK_STATS, pairs_per_block=INIT_CHUNK_PAIRS)


class AdaLN(nn.Module):
    def __init__(self, c_a, c_s, n=2):
        super().__init__()
        self.ln_a = RMSNorm(normalized_shape=(c_a,), elementwise_affine=False)
        self.ln_s = RMSNorm(normalized_shape=(c_s,), bias=False)
        self.to_gain = nn.Sequential(
            nn.Linear(c_s, c_a),
            nn.Sigmoid(),
        )
        self.to_bias = linearNoBias(c_s, c_a)

    def forward(
        self,
        Ai,  # [B, I, C_a]
        Si,  # [B, I, C_s]
    ):
        """
        Output:
            [B, I, C_a]
        """
        Ai = self.ln_a(Ai)
        Si = self.ln_s(Si)
        return self.to_gain(Si) * Ai + self.to_bias(Si)


def create_batch_dimension_if_not_present(batched_n_dim):
    """
    Decorator for adapting a function which expects batched arguments with ndim `batched_n_dim` also
    accept unbatched arguments.
    """

    def wrap(f):
        def _wrap(arg):
            inserted_batch_dim = False
            if arg.ndim == batched_n_dim - 1:
                arg = arg[None]
                inserted_batch_dim = True
            elif arg.ndim == batched_n_dim:
                pass
            else:
                raise Exception(
                    f"arg must have {batched_n_dim - 1} or {batched_n_dim} dimensions, got shape {arg.shape=}"
                )
            o = f(arg)

            if inserted_batch_dim:
                assert o.shape[0] == 1, f"{o.shape=}[0] != 1"
                return o[0]
            return o

        return _wrap

    return wrap


def unpack_args_for_checkpointing(arg_names):
    def wrap(f):
        def _wrap(*args):
            f = args[0]
            return f(**dict(zip(arg_names, args)))

        return _wrap

    return wrap
