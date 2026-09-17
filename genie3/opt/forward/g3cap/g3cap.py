"""
g3cap — the capture-cell levers of the Genie 3 kit, carried as a LIBRARY over the resident driver (g3fast.py):
the batched capture line opt/genie3_opt/g3batch.py imports this module unchanged, after g3fast, for two levers
(both numerics class 1 = bit-identical). It has no command line of its own.

  --hoist            loop-invariant hoist inside V1PairFeatureNet.forward (install_hoist): the relative-position
                     encoding, the four pair masks and the whole conditional-structure template term
                     linear_cond_template(cat[encode_positions(cond_fi.trans), encode_orientations(cond_fi.rots) ...])
                     depend only on the (fixed) batch features, not on the noisy coordinates x_t or t, yet stock (and
                     the resident driver, whose graph covers only the transformer core) recompute them at every one of
                     the 100 DDIM steps - including an N^2 pairwise rot_to_quat -> cuSOLVER eigh (chunks of 31291 4x4
                     matrices, one device->host `info` check = sync per chunk).  Cached once per batch (opt-in per batch
                     dict: batch["_g3cap_hoist"] = True; the cache is batch["_g3cap_static"]); the op sequence on zij is
                     unchanged (zij = zi+zj; zij += R; zij += TEMPLATE*mask; zij += C; return zij*mask) => identical
                     kernels on identical values => bit-identical.
  --reuse-graphs R   keep up to R captured graphs alive keyed by feature-shape signature (feat_signature / GraphCache)
                     and reuse them for later batches of identical shape (rebind_graph: the new features copied into
                     the graph's static tensors, the hoist cache dropped) instead of re-capturing per batch (capture
                     0.7-2.6 s per shape; nothing is cached ACROSS requests - the cache lives inside one process /
                     request only and is freed at exit).
"""
from __future__ import annotations
import collections
from typing import Dict
import torch


# ----------------------------------------------------------------------------------------------------------------
# lever H: loop-invariant hoist in the pair feature net (opt-in per batch dict: batch["_g3cap_hoist"] = True)
# ----------------------------------------------------------------------------------------------------------------
def install_hoist():
    from genie3.generation.model.embedder.pair import v1 as pair_v1
    cls = pair_v1.V1PairFeatureNet
    if getattr(cls, "_g3cap_hoist_installed", False):
        return
    orig_forward = cls.forward                          # (kit-patched) stock forward stays the fallback
    ccsf = pair_v1.compute_conditional_structure_frames  # same symbol the stock forward uses

    def _static_terms(self, batch):
        token_pair_mask = batch["token_mask"][..., None] * batch["token_mask"][..., None, :]
        token_struct_pair_mask = batch["token_struct_mask"][..., None] * batch["token_struct_mask"][..., None, :]
        token_struct_frame_pair_mask = (batch["token_struct_frame_mask"][..., None]
                                        * batch["token_struct_frame_mask"][..., None, :])
        cond_struct_group_pair_mask = torch.zeros_like(token_pair_mask)
        n_cond_group = batch.get("_g3fast_cond_group_max", None)
        if n_cond_group is None:
            n_cond_group = batch["cond_group"].max()
        for group_id in range(1, 1 + n_cond_group):
            cond_struct_group_mask = (batch["cond_group"] == group_id) * batch["cond_struct_mask"]
            cond_struct_group_pair_mask += (cond_struct_group_mask[..., None] * cond_struct_group_mask[..., None, :])
        cond_struct_group_frame_pair_mask = (batch["cond_struct_frame_mask"][..., None]
                                             * batch["cond_struct_frame_mask"][..., None, :]) * cond_struct_group_pair_mask
        R = self.relpos_embedder(asym_id=batch["asym_id"], sym_id=batch["sym_id"], entity_id=batch["entity_id"],
                                 token_index=batch["token_index"], residue_index=batch["residue_index"])
        cond_fi = ccsf(batch)
        C = (self.linear_cond_template(torch.cat([
                self._encode_positions(coords=cond_fi.trans, dist_bin_min=self.cond_template_dist_bin_min,
                                       dist_bin_max=self.cond_template_dist_bin_max,
                                       dist_no_bins=self.cond_template_dist_no_bins),
                (self._encode_orientations(cond_fi.rots) * cond_struct_group_frame_pair_mask[..., None]),
                cond_struct_group_pair_mask[..., None],
                cond_struct_group_frame_pair_mask[..., None],
            ], axis=-1)) * cond_struct_group_pair_mask[..., None])
        return {"tpm": token_pair_mask, "tspm": token_struct_pair_mask, "tsfpm": token_struct_frame_pair_mask,
                "csgpm": cond_struct_group_pair_mask, "csgfpm": cond_struct_group_frame_pair_mask, "R": R, "C": C}

    def forward(self, batch, fi, si):
        if not batch.get("_g3cap_hoist", False):
            return orig_forward(self, batch, fi, si)
        st = batch.get("_g3cap_static", None)
        if st is None:
            st = _static_terms(self, batch)
            batch["_g3cap_static"] = st
        zi = self.linear_zi(si)
        zj = self.linear_zj(si)
        zij = zi[:, :, None, :] + zj[:, None, :, :]
        zij += st["R"]
        zij += (self.linear_template(torch.cat([
                    self._encode_positions(coords=fi.trans, dist_bin_min=self.template_dist_bin_min,
                                           dist_bin_max=self.template_dist_bin_max,
                                           dist_no_bins=self.template_dist_no_bins),
                    (self._encode_orientations(fi.rots) * st["tsfpm"][..., None]),
                    st["tspm"][..., None],
                    st["tsfpm"][..., None],
                    st["csgpm"][..., None],
                    st["csgfpm"][..., None],
                ], axis=-1)) * st["tspm"][..., None])
        zij += st["C"]
        return zij * st["tpm"][..., None]

    cls.forward = forward
    cls._g3cap_hoist_installed = True


# ----------------------------------------------------------------------------------------------------------------
# graph cache (lever R): signature of everything a captured GraphedDenoiser bakes in
# ----------------------------------------------------------------------------------------------------------------
def feat_signature(bd: Dict[str, torch.Tensor]):
    sig = []
    for k in sorted(bd.keys()):
        v = bd[k]
        if isinstance(v, torch.Tensor):
            sig.append((k, tuple(v.shape), str(v.dtype)))
        elif k == "_g3fast_cond_group_max":
            sig.append((k, int(v)))
    return tuple(sig)


class GraphCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.store: "collections.OrderedDict[tuple, list]" = collections.OrderedDict()
        self.n = 0
        self.hits = 0; self.misses = 0

    def take(self, sig):
        lst = self.store.get(sig)
        if lst:
            gd = lst.pop()
            self.n -= 1
            if not lst:
                del self.store[sig]
            self.hits += 1
            return gd
        self.misses += 1
        return None

    def give(self, sig, gd):
        if self.capacity <= 0:
            del gd
            return
        self.store.setdefault(sig, []).append(gd); self.store.move_to_end(sig); self.n += 1
        while self.n > self.capacity:
            k0 = next(iter(self.store)); lst = self.store[k0]; old = lst.pop(0); self.n -= 1
            del old
            if not lst:
                del self.store[k0]

    def clear(self):
        self.store.clear(); self.n = 0


def rebind_graph(gd, bd: Dict[str, torch.Tensor]):
    """Point a cached GraphedDenoiser at a new design of identical signature: copy the new feature values into the
    captured (static) feature tensors; drop the hoist cache so the loop-invariant terms are recomputed."""
    for k, v in bd.items():
        if isinstance(v, torch.Tensor):
            gd.batch[k].copy_(v)
    gd.batch.pop("_g3cap_static", None)
    if getattr(gd, "wide", False):                      # L18: the graph-owned static-term buffers are refilled in place at the graph's next call
        gd._st_dirty = True
    if "_g3cap_hoist" in bd:
        gd.batch["_g3cap_hoist"] = bd["_g3cap_hoist"]
