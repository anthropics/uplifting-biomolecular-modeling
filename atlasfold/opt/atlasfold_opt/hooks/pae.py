"""Lever pae_stream (class exact): the stock monomer ConfidenceHead materializes the PAE logits of all N samples at once as fp32 [B, N, L, L, 64]
(confidence_head.py:313-320) and model.predict then soft-maxes that twice more (compute_pae, compute_ptm: confidence_metrics.py:31-116) —
three N·L²·64·4-byte tensors (5.4 GB each at N=5, L=2,048).  This lever runs the stock per-sample body unchanged and reduces each sample's
logits to its expected-PAE plane [B, 1, L, L] and pTM [B, 1] immediately with the STOCK compute_pae / compute_ptm on the same mask, so only one
sample's logits are alive at a time.  model.predict's later compute_pae / compute_ptm calls receive the reduced values through a small carrier
object and return them as they are.  Same functions, same per-row softmax, same per-sample reductions -> identical values (the kit's unit test
checks torch.equal on CPU).  Site: atlasfold.model.network.confidence_head:ConfidenceHead_Monomer.forward +
atlasfold.model.utils.confidence_metrics:compute_pae/compute_ptm.  MONOMER-ONLY: AtlasFold-M's ConfidenceHead_Multimer feeds compute_*_from_probs (pae/ptm/iptm/chain-pair TM over all samples' probabilities), a different reduction; the lever leaves it unchanged and its line then reads state=skipped reason=n/a:monomer_only(...) instead of no_calls."""
from __future__ import annotations

import torch

from opt_core.counters import Ledger

from . import Installed

NAME = "LOCAL.atlasfold.pae_stream"


class StreamedPAE:
    """Carrier for the already-reduced PAE plane and pTM of N samples (what compute_pae / compute_ptm would have returned)."""
    __slots__ = ("pae", "ptm", "mask_shape", "ndim", "shape")

    def __init__(self, pae: torch.Tensor, ptm: torch.Tensor, mask_shape):
        self.pae, self.ptm, self.mask_shape = pae, ptm, tuple(mask_shape)
        self.ndim = pae.dim() + 1                     # the logits rank model.predict's shape checks would have seen
        self.shape = tuple(pae.shape) + (0,)

    def squeeze(self, dim):                           # model.predict squeezes every output when the input was unbatched; never reached for 'pae' logits, kept for safety
        return self


def install(mode: str, tag: str, settings: dict):
    from atlasfold.model.network import confidence_head as CH
    from atlasfold.model.utils import confidence_metrics as CM
    from atlasfold.model.network.confidence_head import get_bin_centers, get_distogram
    from atlasfold.utils.torch_utils import get_context_dtype

    ledger = Ledger(NAME, impl="per-sample(compute_pae,compute_ptm)", origin="kit")
    stock_forward = CH.ConfidenceHead_Monomer.forward
    stock_compute_pae, stock_compute_ptm = CM.compute_pae, CM.compute_ptm

    def forward(self, batch, s, z, x_pred, kernel_backend="torch"):
        B, N, L, _, _ = x_pred.shape
        device = s.device
        dtype = get_context_dtype(device.type)
        s, z = s.to(dtype), z.to(dtype)
        aa_emb = self.embed_aa(batch["aatype"])
        z = z + aa_emb[:, :, None, :] + aa_emb[:, None, :, :]
        mask = batch["seq_mask"]
        distogram_boundaries = torch.linspace(self.min_dist, self.max_dist, self.num_bins - 1, device=device)
        pae_bin_centers = get_bin_centers(0.0, self.max_pae_error, self.num_pae_bins, device=device)
        mask_pred = mask.unsqueeze(1)                                   # == model.predict's batch["seq_mask"].unsqueeze(1)

        def single(coords):                                             # == stock compute_confidences_single (confidence_head.py:277-304)
            s1 = s.clone()
            distogram = get_distogram(coords, batch["pseudo_beta"], distogram_boundaries)
            z1 = z + self.linear_distogram(distogram.to(z.dtype))
            _s, _ = self.single_stack(s1, z1.clone(), mask, kernel_backend)
            _, _z = self.pair_stack(None, z1.clone(), mask, kernel_backend)
            s2, z2 = _s.float(), _z.float()
            del _s, _z
            with torch.autocast(device.type, enabled=False):
                return {"plddt": self.plddt_head(s2), "experimentally_resolved": self.experimentally_resolved_head(s2), "pae": self.pae_head(z2)}

        plddt_logits = torch.zeros(B, N, L, self.num_plddt_bins, device=device)
        exp_resolved_logits = torch.zeros(B, N, L, 37, device=device)
        pae = torch.zeros(B, N, L, L, device=device, dtype=torch.float32)
        ptm = torch.zeros(B, N, device=device, dtype=torch.float32)
        for i in range(N):
            logits = single(x_pred[:, i])
            if N == 1:                                                   # stock keeps the un-zero-initialised tensors when N == 1 (unsqueeze path)
                plddt_logits, exp_resolved_logits = logits["plddt"].unsqueeze(1), logits["experimentally_resolved"].unsqueeze(1)
            else:
                plddt_logits[:, i] = logits["plddt"]; exp_resolved_logits[:, i] = logits["experimentally_resolved"]
            lg = logits.pop("pae").unsqueeze(1)                          # [B, 1, L, L, bins] fp32 — one sample alive at a time
            with torch.autocast(device.type, enabled=False):
                pae[:, i:i + 1] = stock_compute_pae(lg, pae_bin_centers, mask_pred)
                ptm[:, i:i + 1] = stock_compute_ptm(lg, pae_bin_centers, mask_pred)
            del lg, logits
            ledger.serve("N%dxL%d" % (N, L))
        out = {"experimentally_resolved": {"logits": exp_resolved_logits},
               "plddt": {"logits": plddt_logits, "bin_centers": get_bin_centers(0.0, 1.0, self.num_plddt_bins, device=device)},
               "pae": {"logits": StreamedPAE(pae, ptm, mask_pred.shape), "bin_centers": pae_bin_centers}}
        return out

    def compute_pae(logits, bin_centers, mask, *a, **k):
        if isinstance(logits, StreamedPAE):
            if tuple(mask.shape) != logits.mask_shape:
                raise ValueError(f"Mask shape {tuple(mask.shape)} does not match the streamed PAE mask {logits.mask_shape}.")
            return logits.pae
        return stock_compute_pae(logits, bin_centers, mask, *a, **k)

    def compute_ptm(logits, bin_centers, mask, *a, **k):
        if isinstance(logits, StreamedPAE):
            if tuple(mask.shape) != logits.mask_shape:
                raise ValueError(f"Mask shape {tuple(mask.shape)} does not match the streamed PAE mask {logits.mask_shape}.")
            return logits.ptm
        return stock_compute_ptm(logits, bin_centers, mask, *a, **k)

    forward.__wrapped_stock__ = stock_forward
    CH.ConfidenceHead_Monomer.forward = forward
    CM.compute_pae, CM.compute_ptm = compute_pae, compute_ptm
    multimer_calls = [0]
    stock_multimer = getattr(getattr(CH, "ConfidenceHead_Multimer", None), "forward", None)
    if stock_multimer is not None:                                   # monomer-only lever: the multimer head runs unchanged; only counted for the line
        def forward_multimer(self, *a, **k):
            multimer_calls[0] += 1
            return stock_multimer(self, *a, **k)
        forward_multimer.__wrapped_stock__ = stock_multimer
        forward_multimer.__qualname__ = "ConfidenceHead_Multimer.forward[atlasfold_opt:pae_stream:passthrough]"
        CH.ConfidenceHead_Multimer.forward = forward_multimer

    def line():
        f = ledger.fields() if hasattr(ledger, "fields") else {}
        if multimer_calls[0] and not (f.get("served") or 0):
            return ledger.line(tag, state="skipped", reason=f"n/a:monomer_only(ConfidenceHead_Multimer_unchanged_x{multimer_calls[0]})")
        return ledger.line(tag)

    def _restore():
        CH.ConfidenceHead_Monomer.forward = stock_forward
        CM.compute_pae, CM.compute_ptm = stock_compute_pae, stock_compute_ptm
        if stock_multimer is not None:
            CH.ConfidenceHead_Multimer.forward = stock_multimer
    return Installed("pae_stream", True, lines=[line], gates=[ledger.gate], facts={"scope": "monomer-only", "multimer_calls": multimer_calls})
