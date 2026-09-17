"""g3lean — memory levers of the capture line: the same statements upstream runs, with reference LIFETIMES shortened and the
hoisted pair featuriser's per-step tail computed inside the captured region one batch element at a time. Nothing here changes an op, its
inputs or their order on the exact line; what changes is which tensors are still referenced when the core's activation peak is reached, and
therefore how large the CUDA graph's private pool (sized by ONE eager core forward at capture) and the allocator's default pool become.

  install_lt_release()      L17  LatentTransformer / LatentTransformerBlock forward with OWNERSHIP PASSING: the (si, zij, mask) triple travels
                                 into each block in a list the block empties, so a block's input pair tensor is freed the moment `p = p + …`
                                 rebinds it (upstream keeps one pair-sized tensor [B, N+g, N+g, c_p] alive through every block for nothing).
                                 The net forward also accepts the pair tensor PRE-PADDED (wide capture hands it over with the global-token
                                 border already in place) and then skips F.pad. The activation-checkpointing branch is upstream's own.
  install_trimul_lean()     L17  the STOCK TriangleMultiplicativeUpdate forward (the exact line's provider; mode fast's fused kernel replaces
                                 the forward and is untouched): sigmoid gates and mask products in place on fresh Linear outputs, the two bmm
                                 operands materialised contiguous explicitly (the copy torch.matmul makes of the permuted views) so the
                                 un-permuted projections are freed BEFORE the bmm, the LayerNorm'd input freed before the output gate. Same
                                 Linear / bmm / LayerNorm calls on the same values in the same order — byte-identical by measurement.
  wide_static_terms(), wide_tail()
                            L18  WIDE capture: the hoist's loop-invariant terms (g3cap._static_terms: five pair masks, relative-position term R,
                                 conditional-template term C) written into GRAPH-OWNED buffers — fresh at capture, refilled in place when a kept
                                 graph is re-pointed at a new batch — and the per-step tail of the pair featuriser (zi_i + zj_j + R + template
                                 term + C, masked) computed INSIDE the captured region, one batch element at a time, straight into the interior
                                 of the [B, N+g, N+g, c_p] tensor the pair transform would otherwise build with F.pad. Only the noisy frames, the
                                 single features and the pairwise quaternions (rot_to_quat's host-synchronising eigen-solve) stay eager. The
                                 fill's temporaries are 1/B of a pair tensor; the tail's one pair-sized allocation IS the transformer's input.
Row-independent ops on row slices: the same values per element (B=1 GEMM shapes; byte-identity to the whole-batch form is measured by the
PDB digests of the exact line, not assumed)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

LT_MARK = "_g3lean_release"
TM_MARK = "_g3lean_trimul"


def install_lt_release() -> int:
    """L17 (both kit modes). Returns the number of forwards replaced (2) — 0 when already installed."""
    from genie3.generation.model.latent import transformer as LT
    Blk, Net = LT.LatentTransformerBlock, LT.LatentTransformer
    if getattr(Net, LT_MARK, False):
        return 0
    orig_blk, orig_net = Blk.forward, Net.forward

    def blk_forward(self, inputs):
        if not isinstance(inputs, list):
            return orig_blk(self, inputs)
        s, p, token_mask = inputs
        inputs.clear(); del inputs                                        # the caller's box no longer references the input pair tensor
        token_pair_mask = token_mask.unsqueeze(1) * token_mask.unsqueeze(2)
        s = s + self.ipa(s, p, token_mask)
        s = self.ipa_dropout(s)
        s = self.ipa_layer_norm(s)
        s = self.ipa_transition(s)
        s = s * token_mask[..., None]
        pi = self.linear_pi(s)
        pj = self.linear_pj(s)
        p = p + self.linear_p(pi[:, :, None, :] + pj[:, None, :, :])    # rebinding p frees the block's input pair tensor here (its last reference)
        p = p + self.dropout_row_layer(self.tri_mul_out(p, token_pair_mask))
        p = p + self.dropout_row_layer(self.tri_mul_in(p, token_pair_mask))
        p = p + self.pair_transition(p, token_pair_mask)
        p = p * token_pair_mask.unsqueeze(-1)
        return (s, p, token_mask)

    def net_forward(self, si, zij, mask, return_global_tokens=False):
        prepadded = False
        if isinstance(zij, list):                                         # wide capture hands the pair tensor over in a box it no longer references; [t, True] = already padded
            _box = zij
            prepadded = len(_box) > 1 and bool(_box[1])
            zij = _box[0]; _box.clear(); del _box
        if self.enable_activation_checkpointing:                          # upstream's own path (checkpoint takes tensor arguments)
            return orig_net(self, si, zij, mask, return_global_tokens=return_global_tokens)
        if self.n_global_token > 0:
            batch_size, c_s = si.shape[0], si.shape[-1]
            si = torch.concat([torch.zeros((batch_size, self.n_global_token, c_s), dtype=si.dtype, device=si.device), si], dim=-2)
            if not prepadded:
                zij = F.pad(zij, (0, 0, self.n_global_token, 0, self.n_global_token, 0, 0, 0), value=0)
            mask = torch.concat([torch.ones((batch_size, self.n_global_token), dtype=mask.dtype, device=mask.device), mask], dim=-1)
        mods = list(self.net)
        for i in range(0, len(mods), self.chunk_size):
            ms = mods[i: i + self.chunk_size]
            box = [si, zij, mask]
            del si, zij, mask
            for m in ms:
                out = m(box)                                              # the block empties `box`; its result is the next block's box
                box = list(out); del out
            si, zij, mask = box
            del box
        if self.n_global_token > 0:
            gi = si[:, : self.n_global_token]
            si = si[:, self.n_global_token:]
            zij = zij[:, self.n_global_token:, self.n_global_token:]
        if return_global_tokens:
            return gi, si, zij
        else:
            return si, zij

    Blk.forward = blk_forward
    Net.forward = net_forward
    setattr(Net, LT_MARK, True)
    return 2


def install_trimul_lean() -> int:
    """L17 (the exact line's stock provider; mode fast's fused-kernel adapter replaces the forward and is not touched). Returns 1, or 0 when
    already installed."""
    from genie3.generation.model.module import triangular_multiplicative_update as TM
    from genie3.generation.utils.tensor_utils import permute_final_dims
    cls = TM.TriangleMultiplicativeUpdate
    if getattr(cls, TM_MARK, False):
        return 0

    def forward(self, z, mask=None):
        if mask is None:
            mask = z.new_ones(z.shape[:-1], requires_grad=False)
        mask = mask.unsqueeze(-1)
        z = self.layer_norm_in(z)
        ga = self.linear_a_g(z); ga = torch.sigmoid_(ga)                 # gate in place on the Linear's fresh output (same elementwise values as sigmoid(.))
        a = self.linear_a_p(z); a.mul_(ga); del ga                       # a = linear_a_p(z) * sigmoid(linear_a_g(z))
        a.mul_(mask)                                                      # a = a * mask
        A = permute_final_dims(a, 2, 0, 1) if self._outgoing else permute_final_dims(a, 2, 1, 0)   # [*, C, N_i, N_k]
        lead = a.shape[:-3]; C = a.shape[-1]
        del a
        A = A.reshape(-1, A.shape[-2], A.shape[-1])                       # the contiguous copy torch.matmul's batched path makes of the permuted view, made here so `a` is freed before the bmm
        gb_ = self.linear_b_g(z); gb_ = torch.sigmoid_(gb_)
        b = self.linear_b_p(z); b.mul_(gb_); del gb_
        b.mul_(mask)
        Bm = permute_final_dims(b, 2, 1, 0) if self._outgoing else permute_final_dims(b, 2, 0, 1)  # [*, C, N_k, N_j]
        del b
        Bm = Bm.reshape(-1, Bm.shape[-2], Bm.shape[-1])
        p = torch.matmul(A, Bm)                                           # 3-D contiguous operands -> at::bmm, the call matmul's batched path issues on its own copies
        del A, Bm
        p = p.view(*lead, C, p.shape[-2], p.shape[-1])
        x = permute_final_dims(p, 1, 2, 0)                                # [*, N_i, N_j, C] view, as stock hands it to layer_norm_out
        del p
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        g = self.linear_g(z); del z
        g = torch.sigmoid_(g)
        x.mul_(g)                                                         # z = x * g
        return x

    cls._g3lean_orig_forward = cls.forward
    cls.forward = forward
    setattr(cls, TM_MARK, True)
    return 1


def lt_release_installed() -> bool:
    from genie3.generation.model.latent import transformer as LT
    return bool(getattr(LT.LatentTransformer, LT_MARK, False))


# ------------------------------------------------------------------------------------------------------------------------------ wide capture (L18)
def wide_static_terms(pfn, batch, out=None):
    """The hoist's loop-invariant terms of a batch (the statements of g3cap._static_terms) written into graph-owned buffers — fresh ones at
    capture (out=None) or the kept graph's (`out`) when it is re-pointed at a new batch. The five pair masks are computed whole ([B, N, N]);
    the two pair-sized terms R (relative-position embedding) and C (conditional template) are computed ONE BATCH ELEMENT AT A TIME straight
    into their [B, N, N, c_p] buffers, so the fill's temporaries are 1/B of a pair tensor instead of ~3 pair tensors."""
    from genie3.generation.model.embedder.pair import v1 as pair_v1
    ccsf = pair_v1.compute_conditional_structure_frames
    self = pfn
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
    masks = {"tpm": token_pair_mask, "tspm": token_struct_pair_mask, "tsfpm": token_struct_frame_pair_mask,
             "csgpm": cond_struct_group_pair_mask, "csgfpm": cond_struct_group_frame_pair_mask}
    del token_pair_mask, token_struct_pair_mask, token_struct_frame_pair_mask, cond_struct_group_pair_mask, cond_struct_group_frame_pair_mask
    if out is None:
        out = dict(masks)
    else:
        for k, v in masks.items():
            out[k].copy_(v)
    masks = {k: out[k] for k in masks}
    cond_fi = ccsf(batch)
    B = int(batch["token_mask"].shape[0])

    def R_of(sl):
        return self.relpos_embedder(asym_id=batch["asym_id"][sl], sym_id=batch["sym_id"][sl], entity_id=batch["entity_id"][sl],
                                    token_index=batch["token_index"][sl], residue_index=batch["residue_index"][sl])

    def Clin_of(sl):
        return self.linear_cond_template(torch.cat([
            self._encode_positions(coords=cond_fi.trans[sl], dist_bin_min=self.cond_template_dist_bin_min,
                                   dist_bin_max=self.cond_template_dist_bin_max,
                                   dist_no_bins=self.cond_template_dist_no_bins),
            (self._encode_orientations(cond_fi.rots[sl]) * masks["csgfpm"][sl][..., None]),
            masks["csgpm"][sl][..., None],
            masks["csgfpm"][sl][..., None],
        ], axis=-1))

    for bi in range(B):
        sl = slice(bi, bi + 1)
        Rb = R_of(sl)
        if "R" not in out:
            out["R"] = torch.empty((B,) + tuple(Rb.shape[1:]), dtype=Rb.dtype, device=Rb.device)
        out["R"][sl].copy_(Rb); del Rb
        lin = Clin_of(sl)
        if "C" not in out:
            out["C"] = torch.empty((B,) + tuple(lin.shape[1:]), dtype=lin.dtype, device=lin.device)
        torch.mul(lin, masks["csgpm"][sl][..., None], out=out["C"][sl]); del lin
    st = dict(masks); st["R"] = out["R"]; st["C"] = out["C"]
    return st


def wide_tail(pfn, st, si, trans, q, n_pad: int):
    """The per-step half of the hoisted pair featuriser, INSIDE the captured region, one batch element at a time straight into the interior
    of the buffer the pair transform would otherwise build with F.pad (n_pad global-token rows/columns of zeros in front): the values of
    g3cap's hoisted forward (`zij = zi_i + zj_j; += R; += linear_template(cat(...)) * tspm; += C; * tpm`) written through `out=` / in-place
    forms into a [B, N+n_pad, N+n_pad, c_p] tensor whose border stays zero."""
    self = pfn
    zi = self.linear_zi(si)
    zj = self.linear_zj(si)
    B, N, C = int(zi.shape[0]), int(zi.shape[1]), int(zi.shape[2])
    g = int(n_pad)
    buf = torch.zeros((B, N + g, N + g, C), dtype=zi.dtype, device=zi.device)

    def T_of(sl):
        return self.linear_template(torch.cat([
                self._encode_positions(coords=trans[sl], dist_bin_min=self.template_dist_bin_min,
                                       dist_bin_max=self.template_dist_bin_max, dist_no_bins=self.template_dist_no_bins),
                (q[sl] * st["tsfpm"][sl][..., None]),
                st["tspm"][sl][..., None],
                st["tsfpm"][sl][..., None],
                st["csgpm"][sl][..., None],
                st["csgfpm"][sl][..., None],
            ], axis=-1))

    for b in range(B):
        v = buf[b, g:, g:, :]                                             # [N, N, C] interior view of element b
        torch.add(zi[b][:, None, :], zj[b][None, :, :], out=v)           # zij = zi_i + zj_j
        v += st["R"][b]
        T = T_of(slice(b, b + 1))[0]
        v += T * st["tspm"][b][..., None]
        del T
        v += st["C"][b]
        v *= st["tpm"][b][..., None]
    return buf
