"""Lever pae_stream_m (AtlasFold-M twin of pae_stream): the stock ConfidenceHead_Multimer PRE-ALLOCATES two fp32 sample stacks before its
per-sample loop — pae_logits AND pde_logits [B, N, L, L, 64] (confidence_head.py L488-491; 11.25 GiB each at N=5, L=3,072) — and
model_multimer.inference (L393-431) then soft-maxes the whole PAE stack once more and reduces it with compute_pae/ptm/iptm/chain_tm_scores
_from_probs and compute_pde.  Every one of those reductions is per sample.  This lever runs the stock per-sample body unchanged and reduces
each sample's PAE/PDE logits IMMEDIATELY with the STOCK functions (same softmax, same bin centers, same masks, same asym_id), keeping only what
model_multimer.inference returns anyway: pae/pde planes [B,N,L,L], ptm/iptm [B,N], chain_ptm [B,N,C], interface_iptm [B,N,C,C].  The later stock
calls receive the reduced values through a small carrier (torch.softmax on it is answered through __torch_function__) and return them as they
are.  Chain / chain-pair scores are evaluated on blocks of the STOCK [N, Lc, Lc, bins] shape (this sample in its own slot) so every
reduction sees the stock geometry and element offsets (CUDA row reductions split their vectorized loop by the row's base alignment, which for
unpadded chain lengths depends on the sample slot); the full-plane reductions keep stock geometry because the head runs at the padded bucket
length.  One sample's [L,L,64] logits+probs alive at a time instead of 2N+1 stacks.  Same functions per sample -> identical values (CPU unit test:
torch.equal).  plddt / experimentally_resolved logits stay stock (small).
Site: atlasfold.model.network.confidence_head:ConfidenceHead_Multimer.forward + atlasfold.model.utils.confidence_metrics:compute_pde/
compute_pae_from_probs/compute_ptm_from_probs/compute_iptm_from_probs/compute_chain_tm_scores_from_probs."""
from __future__ import annotations

import torch

from opt_core.counters import Ledger

from . import Installed

NAME = "LOCAL.atlasfold.pae_stream_m"


class StreamedM:
    """Carrier for the already-reduced per-sample confidence values of AtlasFold-M (what model_multimer.inference computes from the stacks)."""
    __slots__ = ("pae", "pde", "ptm", "iptm", "chain_ptm", "interface_iptm", "shape", "ndim", "is_probs")

    def __init__(self, pae, pde, ptm, iptm, chain_ptm, interface_iptm, logits_shape):
        self.pae, self.pde, self.ptm, self.iptm, self.chain_ptm, self.interface_iptm = pae, pde, ptm, iptm, chain_ptm, interface_iptm
        self.shape = tuple(logits_shape); self.ndim = len(self.shape); self.is_probs = False

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):        # model_multimer.inference: pae_probs = torch.softmax(pae_logits, dim=-1)
        if func in (torch.softmax, torch.nn.functional.softmax) and args and isinstance(args[0], StreamedM):
            args[0].is_probs = True
            return args[0]
        return NotImplemented

    def squeeze(self, dim=None):
        return self


def _chain_tm_scores_slot(stock, probs_i, i: int, N: int, bin_centers, asym_id, mask):
    """compute_chain_tm_scores_from_probs (confidence_metrics.py L243-366) for ONE sample, evaluated on blocks of the STOCK shape:
    each chain / chain-pair block is an [N, Lc, Lc, bins] tensor whose slot i holds this sample (other slots zero) and row i of the stock
    function's result is kept.  Same shapes and element offsets as the stock call on the N-stack -> the same CUDA reduction geometry per
    element (row base alignment included), at the cost of N-fold work on the small chain blocks only."""
    B = probs_i.shape[0]
    valid_mask = mask[:, 0].bool()
    chain_ids_by_batch = [torch.unique(asym_id[b][valid_mask[b]], sorted=True) for b in range(B)]
    max_chains = max((len(c) for c in chain_ids_by_batch), default=0)
    chain_ptm = torch.full((B, 1, max_chains), torch.nan, dtype=probs_i.dtype, device=probs_i.device)
    interface_iptm = torch.full((B, 1, max_chains, max_chains), torch.nan, dtype=probs_i.dtype, device=probs_i.device)
    for b, chain_ids in enumerate(chain_ids_by_batch):
        chain_indices = [torch.nonzero(valid_mask[b] & (asym_id[b] == cid), as_tuple=False).squeeze(-1) for cid in chain_ids]
        p1 = probs_i[b, 0]                                                     # [L, L, bins]
        for ci, idx in enumerate(chain_indices):
            blk = torch.zeros((N, len(idx), len(idx), p1.shape[-1]), dtype=p1.dtype, device=p1.device)
            blk[i] = p1[idx][:, idx]
            cm = torch.ones((N, len(idx)), dtype=torch.bool, device=p1.device)
            chain_ptm[b, 0, ci] = stock["compute_ptm_from_probs"](blk, bin_centers, cm)[i]
            del blk
        for ci, idx_i in enumerate(chain_indices):
            for cj in range(ci + 1, len(chain_indices)):
                pair = torch.cat((idx_i, chain_indices[cj]))
                blk = torch.zeros((N, len(pair), len(pair), p1.shape[-1]), dtype=p1.dtype, device=p1.device)
                blk[i] = p1[pair][:, pair]
                pm = torch.ones((N, len(pair)), dtype=torch.bool, device=p1.device)
                pa = torch.cat((torch.zeros(len(idx_i), dtype=asym_id.dtype, device=p1.device), torch.ones(len(chain_indices[cj]), dtype=asym_id.dtype, device=p1.device)))
                score = stock["compute_iptm_from_probs"](blk, bin_centers, pa, pm)[i]
                interface_iptm[b, 0, ci, cj] = score; interface_iptm[b, 0, cj, ci] = score
                del blk
    return chain_ptm, interface_iptm


def install(mode: str, tag: str, settings: dict):
    try:
        from atlasfold.model.network import confidence_head as CH
        from atlasfold.model.utils import confidence_metrics as CM
        from atlasfold.utils.torch_utils import get_context_dtype
    except Exception as e:  # noqa: BLE001
        return Installed("pae_stream_m", False, reason=f"import:{type(e).__name__}")
    if not hasattr(CH, "ConfidenceHead_Multimer"):
        return Installed("pae_stream_m", False, reason="no_ConfidenceHead_Multimer")
    ledger = Ledger(NAME, impl="per-sample(compute_pde,compute_*_from_probs)", origin="kit")
    stock_forward = CH.ConfidenceHead_Multimer.forward
    stock = {n: getattr(CM, n) for n in ("compute_pde", "compute_pae_from_probs", "compute_ptm_from_probs", "compute_iptm_from_probs", "compute_chain_tm_scores_from_probs")}

    def forward(self, batch, s, z, x_pred, kernel_backend: str = "torch"):
        # == stock ConfidenceHead_Multimer.forward (confidence_head.py L436-514) with the N-stacks of pae/pde logits replaced by per-sample reduction
        s, z, x_pred = map(lambda x: x.detach(), (s, z, x_pred))
        device = s.device
        dtype = get_context_dtype(device.type)
        s, z = s.to(dtype, copy=True), z.to(dtype, copy=True)
        s = self.proj_s(s)
        aa_emb = self.embed_aa(batch["aatype"])
        z = z + aa_emb[:, :, None, :] + aa_emb[:, None, :, :]
        mask = batch["seq_mask"]
        distogram_boundaries = torch.linspace(self.min_dist, self.max_dist, self.num_bins - 1, device=device)

        def compute_confidences_single(s, z, mask, coords):               # verbatim stock inner body
            s = s.clone()
            cbeta_idx = batch["pseudo_beta"]
            distogram = CH.get_distogram(coords, cbeta_idx, distogram_boundaries)
            z = z + self.linear_distogram(distogram.to(z.dtype))
            s, z = self.stack(s, z, mask, kernel_backend=kernel_backend)
            s, z = s.float(), z.float()
            logits = {}
            with torch.autocast(device.type, enabled=False):
                logits["plddt"] = self.plddt_head(s)
                logits["experimentally_resolved"] = self.experimentally_resolved_head(s)
                logits["pae"] = self.pae_head(z)
                logits["pde"] = self.pde_head(z)
            return logits

        B, N, L, _, _ = x_pred.shape
        plddt_logits = torch.zeros(B, N, L, self.num_plddt_bins, device=device)
        exp_resolved_logits = torch.zeros(B, N, L, 37, device=device)
        pae_bin_centers = CH.get_bin_centers(0.0, self.max_pae_error, self.num_pae_bins, device=device)
        pde_bin_centers = CH.get_bin_centers(0.0, self.max_pde_error, self.num_pde_bins, device=device)
        m1 = mask.unsqueeze(1)                                               # [B, 1, L] == model_multimer.inference's mask
        red = {"pae": [], "pde": [], "ptm": [], "iptm": [], "chain_ptm": [], "interface_iptm": []}
        for i in range(N):
            logits = compute_confidences_single(s, z, mask, x_pred[:, i])
            plddt_logits[:, i] = logits["plddt"]
            exp_resolved_logits[:, i] = logits["experimentally_resolved"]
            with torch.autocast(device.type, enabled=False):              # == model_multimer.inference L393-425, one sample
                pae_i = logits.pop("pae").unsqueeze(1)                       # [B, 1, L, L, bins] fp32
                pde_i = logits.pop("pde").unsqueeze(1)
                red["pde"].append(stock["compute_pde"](pde_i, pde_bin_centers, m1)); del pde_i
                probs = torch.softmax(pae_i, dim=-1); del pae_i
                red["pae"].append(stock["compute_pae_from_probs"](probs, pae_bin_centers, m1))
                red["ptm"].append(stock["compute_ptm_from_probs"](probs, pae_bin_centers, m1))
                red["iptm"].append(stock["compute_iptm_from_probs"](probs, pae_bin_centers, batch["asym_id"], m1))
                cp, ii = _chain_tm_scores_slot(stock, probs, i, N, pae_bin_centers, batch["asym_id"], m1)
                red["chain_ptm"].append(cp); red["interface_iptm"].append(ii); del probs
        ledger.serve("N%dxL%d" % (N, L))
        carrier = StreamedM(*(torch.cat(red[k], dim=1) for k in ("pae", "pde", "ptm", "iptm", "chain_ptm", "interface_iptm")),
                            logits_shape=(B, N, L, L, self.num_pae_bins))
        out = {}
        out["experimentally_resolved"] = {"logits": exp_resolved_logits}
        out["plddt"] = {"logits": plddt_logits, "bin_centers": CH.get_bin_centers(0.0, 1.0, self.num_plddt_bins, device=device)}
        out["pae"] = {"logits": carrier, "bin_centers": pae_bin_centers}
        out["pde"] = {"logits": carrier, "bin_centers": pde_bin_centers}
        return out

    def compute_pde(logits, bin_centers, mask, *a, **k):
        if isinstance(logits, StreamedM):
            return logits.pde
        return stock["compute_pde"](logits, bin_centers, mask, *a, **k)

    def compute_pae_from_probs(probs, bin_centers, mask, *a, **k):
        if isinstance(probs, StreamedM):
            return probs.pae
        return stock["compute_pae_from_probs"](probs, bin_centers, mask, *a, **k)

    def compute_ptm_from_probs(probs, bin_centers, mask, *a, **k):
        if isinstance(probs, StreamedM):
            return probs.ptm
        return stock["compute_ptm_from_probs"](probs, bin_centers, mask, *a, **k)

    def compute_iptm_from_probs(probs, bin_centers, asym_id, mask, *a, **k):
        if isinstance(probs, StreamedM):
            return probs.iptm
        return stock["compute_iptm_from_probs"](probs, bin_centers, asym_id, mask, *a, **k)

    def compute_chain_tm_scores_from_probs(probs, bin_centers, asym_id, mask, *a, **k):
        if isinstance(probs, StreamedM):
            return probs.chain_ptm, probs.interface_iptm
        return stock["compute_chain_tm_scores_from_probs"](probs, bin_centers, asym_id, mask, *a, **k)

    forward.__wrapped_stock__ = stock_forward
    forward.__qualname__ = "ConfidenceHead_Multimer.forward[atlasfold_opt:pae_stream_m]"
    CH.ConfidenceHead_Multimer.forward = forward
    for n, f in (("compute_pde", compute_pde), ("compute_pae_from_probs", compute_pae_from_probs), ("compute_ptm_from_probs", compute_ptm_from_probs),
                 ("compute_iptm_from_probs", compute_iptm_from_probs), ("compute_chain_tm_scores_from_probs", compute_chain_tm_scores_from_probs)):
        f.__wrapped_stock__ = stock[n]; setattr(CM, n, f)

    def _restore():
        CH.ConfidenceHead_Multimer.forward = stock_forward
        for n, f in stock.items():
            setattr(CM, n, f)
    return Installed("pae_stream_m", True, lines=[lambda: ledger.line(tag)], gates=[ledger.gate], facts={"scope": "multimer-only", "restore": _restore})
