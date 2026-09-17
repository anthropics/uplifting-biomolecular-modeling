# rfd_se3fast/dense_torch.py
# Dense (all-pairs, destination-major) re-formulation of RFdiffusion v1's Str2Str -> SE3Transformer call.
# This is (a) the executable specification / fallback for the Triton kernels in kernels.py and (b) by itself the
# "t2-torch" path: no DGL graph, no per-edge gathers of index lists, masked dense softmax / pooling, static shapes
# (so the data-dependent top-k refinement calls become CUDA-graph capturable).
#
# Fidelity class: TIER-2 (re-associated fp32 reductions: softmax and neighbour sums are dense last-axis reductions
# instead of DGL CSR segment reductions; everything else calls the STOCK sub-modules on re-ordered edge tensors).
# Precision policy equal to stock (fp32; TF32 stays off).
#
# Layout: X[d, s, ...] is the edge s -> d (source s, destination d).  Diagonal (s == d) is present in memory and
# masked out (stock graphs have no self-edges).  For stock edge i -> j: edge feature = pair[b, i, j], rel_pos =
# xyz[j] - xyz[i]; attention softmax and the final pooling reduce over incoming edges of each destination j.
#
# VENDORED_FROM: control flow of Str2Str.forward is transcribed from RosettaCommons/RFdiffusion rfdiffusion/Track_module.py
# @ 86507b6538f51fce57b5a72477165f03999ed7ae (BSD-3-Clause); SE(3) layer semantics from NVIDIA SE3Transformer (MIT) as
# vendored in the same repository (env/SE3Transformer).  See LICENSES.md and NOTICE beside this package.
from typing import Dict, Optional
import math
import torch
from torch import Tensor

import rfdiffusion.Track_module as TM                      # get_seqsep / rbf looked up at CALL time (kit fastpath may patch them)
from se3_transformer.model.basis import get_basis, update_basis_with_fused

STATS = dict(dense_full_calls=0, dense_topk_calls=0, fallback_calls=0)


def topk_mask_dst_major(xyz_ca: Tensor, idx: Tensor, top_k: int = 64, kmin: int = 32, eps: float = 1e-6) -> Tensor:
    """Boolean (L, L) mask M[d, s] = stock make_topk_graph edge s -> d exists.  Same arithmetic as
    rfdiffusion.util_module.make_topk_graph (B == 1), without torch.where / DGL (static shapes, no host sync)."""
    B, L = xyz_ca.shape[:2]
    device = xyz_ca.device
    D = torch.cdist(xyz_ca, xyz_ca) + torch.eye(L, device=device).unsqueeze(0) * 999.9      # (B, L, L)
    sep = idx[:, None, :] - idx[:, :, None]
    sep = sep.abs() + torch.eye(L, device=device).unsqueeze(0) * 999.9
    D = D + sep * eps
    D_neigh, E_idx = torch.topk(D, min(top_k, L), largest=False)
    topk_matrix = torch.zeros((B, L, L), device=device)
    topk_matrix.scatter_(2, E_idx, 1.0)
    cond = torch.logical_or(topk_matrix > 0.0, sep < kmin)                                   # cond[b, i, j]: edge i -> j
    return cond[0].transpose(0, 1).contiguous()                                              # M[d = j, s = i]


def full_mask(L: int, device) -> Tensor:
    return ~torch.eye(L, dtype=torch.bool, device=device)


def dense_edge_inputs(pair: Tensor, xyz: Tensor):
    """pair: (1, L, L, Ce) post-embedding pair features; xyz: (1, L, 3, 3) backbone (N, CA, C).
    Returns ED (L*L, Ce) with ED[d*L+s] = pair[0, s, d];  rel (L*L, 3) = ca[d] - ca[s];  l1 (L, 3, 3)."""
    L = pair.shape[1]
    ED = pair[0].transpose(0, 1).reshape(L * L, -1)                     # contiguous copy in (d, s) order
    ca = xyz[0, :, 1, :]                                                # (L, 3)
    rel = (ca[:, None, :] - ca[None, :, :]).reshape(L * L, 3)           # rel[d, s] = xyz[dst] - xyz[src]  (stock: xyz[b,j]-xyz[b,i])
    l1 = (xyz - xyz[:, :, 1, :].unsqueeze(2)).reshape(L, -1, 3)         # stock l1_feats (B*L, 3, 3)
    return ED, rel, l1


def se3_dense_forward(se3, node0: Tensor, node1: Tensor, ED: Tensor, rel: Tensor, mask: Tensor) -> Dict[str, Tensor]:
    """Dense evaluation of se3_transformer.model.SE3Transformer `se3` (num_layers=1, max_degree=1 as in RFdiffusion)
    using the STOCK sub-modules on destination-major dense edge tensors.
      node0 (L, C0, 1), node1 (L, 3, 3), ED (L*L, Ce), rel (L*L, 3), mask (L, L) bool [d, s]."""
    L = node0.shape[0]
    E = L * L
    gm = se3.graph_modules
    ab, norm, fc = gm[0], gm[1], gm[2]
    # ---- bases and invariant edge features (stock functions; rows in (d, s) order)
    basis = get_basis(rel, max_degree=se3.max_degree, compute_gradients=False,
                      use_pad_trick=se3.tensor_cores and not se3.low_memory, amp=torch.is_autocast_enabled())
    basis = update_basis_with_fused(basis, se3.max_degree, use_pad_trick=se3.tensor_cores and not se3.low_memory,
                                    fully_fused=se3.tensor_cores and not se3.low_memory)
    r = rel.norm(dim=-1, keepdim=True)                                    # (E, 1)
    inv = torch.cat([ED, r], dim=1)                                       # (E, Ce+1) == stock invariant_edge_feats
    maskf = mask.to(node0.dtype)                                          # (L, L)
    # ---- attention block: keys/values from source-node features (broadcast over destinations)
    kv_conv = ab.to_key_value
    f0 = node0.unsqueeze(0).expand(L, L, *node0.shape[1:]).reshape(E, *node0.shape[1:])       # feat[d, s] = node0[s]
    f1 = node1.unsqueeze(0).expand(L, L, *node1.shape[1:]).reshape(E, *node1.shape[1:])
    kv = kv_conv.conv_in['0'](f0, inv, basis['in0_fused']) + kv_conv.conv_in['1'](f1, inv, basis['in1_fused'])   # (E, 16, 4)
    key, value = ab._get_key_value_from_fused(kv)                          # value, key = chunk(kv, 2, dim=-2)
    query = ab.to_query({'0': node0, '1': node1})
    att = ab.attention
    H = att.num_heads
    keyh = key.reshape(E, H, -1)                                           # (E, H, 8)
    q = torch.cat([query[str(d)] for d in att.key_fiber.degrees], dim=-1)
    q = q.reshape(L, H, -1)                                                # (L, H, 8)
    logits = (keyh.view(L, L, H, -1) * q.view(L, 1, H, -1)).sum(-1)        # (L_d, L_s, H): e_dot_v with the DESTINATION's query
    logits = logits / math.sqrt(att.key_fiber.num_features)
    logits = logits.masked_fill(~mask.unsqueeze(-1), float('-inf'))
    w = torch.softmax(logits, dim=1)                                       # over sources = incoming edges of d
    v = value.view(L, L, H, -1, value.shape[-1])                           # (L, L, H, C/H, 4)
    z = torch.einsum('dsh,dshck->dhck', w, v)                              # copy_e_sum(w * v)
    z = z.reshape(L, -1, value.shape[-1])                                  # merge heads -> (L, 8, 4)
    zs = z.split([2 * d + 1 for d in att.value_fiber.degrees], dim=-1)
    z = {str(d): t for d, t in zip(att.value_fiber.degrees, zs)}
    z_cat = {k: torch.cat([z[k], {'0': node0, '1': node1}[k]], dim=1) for k in z}     # aggregate_residual(node_features, z, 'cat')
    x = ab.project(z_cat)                                                  # {'0': (L, 32, 1), '1': (L, 32, 3)}
    # ---- norm
    x = norm(x)
    # ---- final ConvSE3 (fused per output degree) + self-interaction + masked pooling over incoming edges
    xf = torch.cat([x['0'], x['1']], dim=-1)                               # (L, 32, 4)
    fs = xf.unsqueeze(0).expand(L, L, *xf.shape[1:]).reshape(E, *xf.shape[1:])          # per edge: source features
    out = {}
    deg = maskf.sum(1)                                                     # in-degree per destination (L,)
    for d_out in fc.fiber_out.degrees:
        o = fc.conv_out[str(d_out)](fs, inv, basis[f'out{d_out}_fused'])  # (E, C_out, 2*d_out+1)
        o = (o.view(L, L, *o.shape[1:]) * maskf.view(L, L, 1, 1)).sum(1)   # (L, C_out, 2d+1)   copy_e_sum
        if fc.self_interaction and str(d_out) in fc.to_kernel_self:
            # stock adds kernel_self @ x[dst] to EVERY edge before pooling  ->  in-degree * (K @ x[d])
            o = o + deg.view(L, 1, 1) * (fc.to_kernel_self[str(d_out)] @ x[str(d_out)])
        out[str(d_out)] = o
    return out


_PACK = {}


def _packed_radial_params(se3):
    """Stack the parameters of the 4 RadialProfile MLPs of one SE(3) layer (kv conv_in['0'], kv conv_in['1'],
    final conv_out['0'], conv_out['1']) for the fused trunk kernel; cached per module (inference: params frozen)."""
    key = id(se3)
    p = _PACK.get(key)
    if p is not None:
        return p
    ab, fc = se3.graph_modules[0], se3.graph_modules[2]
    convs = [ab.to_key_value.conv_in['0'], ab.to_key_value.conv_in['1'], fc.conv_out['0'], fc.conv_out['1']]
    nets = [c.radial_func.net for c in convs]        # Linear, LN, ReLU, Linear, LN, ReLU, Linear(no bias)
    st = lambda i, a: torch.stack([getattr(n[i], a).detach() for n in nets]).contiguous().float()
    p = dict(W1=st(0, 'weight'), B1=st(0, 'bias'), G1=st(1, 'weight'), BE1=st(1, 'bias'),
             W2=st(3, 'weight'), B2=st(3, 'bias'), G2=st(4, 'weight'), BE2=st(4, 'bias'),
             W3=[n[6].weight.detach().contiguous().float() for n in nets],
             eps=float(nets[0][1].eps),
             meta=[(c.channels_in, c.channels_out, c.freq_sum) for c in convs])
    assert all(abs(n[1].eps - p['eps']) < 1e-12 and abs(n[4].eps - p['eps']) < 1e-12 for n in nets)
    _PACK[key] = p
    return p


def se3_dense_forward_triton(se3, node0: Tensor, node1: Tensor, ED: Tensor, rel: Tensor, mask: Tensor) -> Dict[str, Tensor]:
    """Same contract as se3_dense_forward; radial MLP trunks (x4) in ONE Triton kernel and each VersatileConvSE3's
    'radial_weights @ tmp' in the fused radial_conv kernel.  fp32 throughout."""
    from . import kernels as KR
    L = node0.shape[0]
    E = L * L
    gm = se3.graph_modules
    ab, norm, fc = gm[0], gm[1], gm[2]
    P = _packed_radial_params(se3)
    basis = get_basis(rel, max_degree=se3.max_degree, compute_gradients=False,
                      use_pad_trick=se3.tensor_cores and not se3.low_memory, amp=torch.is_autocast_enabled())
    basis = update_basis_with_fused(basis, se3.max_degree, use_pad_trick=se3.tensor_cores and not se3.low_memory,
                                    fully_fused=se3.tensor_cores and not se3.low_memory)
    r = rel.norm(dim=-1, keepdim=True)
    inv = torch.cat([ED, r], dim=1).contiguous()
    hid = KR.radial_trunk(inv, P['W1'], P['B1'], P['G1'], P['BE1'], P['W2'], P['B2'], P['G2'], P['BE2'], eps=P['eps'])   # (4, E, 32)
    maskf = mask.to(node0.dtype)

    def conv(g, features, basis_t, fuse_full):
        """stock VersatileConvSE3.forward with radial_func(inv) @ tmp replaced by the fused kernel."""
        c_in, c_out, J = P['meta'][g]
        in_dim = features.shape[2]
        out_dim = basis_t.shape[-1]
        if not fuse_full:
            out_dim += out_dim % 2 - 1
        tmp = (features @ basis_t.view(E, in_dim, -1)).view(E, -1, basis_t.shape[-1]).contiguous()   # (E, c_in*J, K)
        out = KR.radial_conv(hid[g], P['W3'][g], tmp, CO=c_out, J=c_in * J, K=basis_t.shape[-1])
        return out[:, :, :out_dim]

    f0 = node0.unsqueeze(0).expand(L, L, *node0.shape[1:]).reshape(E, *node0.shape[1:])
    f1 = node1.unsqueeze(0).expand(L, L, *node1.shape[1:]).reshape(E, *node1.shape[1:])
    kv = conv(0, f0, basis['in0_fused'], True) + conv(1, f1, basis['in1_fused'], True)                 # (E, 16, 4)
    key, value = ab._get_key_value_from_fused(kv)
    query = ab.to_query({'0': node0, '1': node1})
    att = ab.attention
    H = att.num_heads
    keyh = key.reshape(E, H, -1)
    q = torch.cat([query[str(d)] for d in att.key_fiber.degrees], dim=-1).reshape(L, H, -1)
    logits = (keyh.view(L, L, H, -1) * q.view(L, 1, H, -1)).sum(-1) / math.sqrt(att.key_fiber.num_features)
    logits = logits.masked_fill(~mask.unsqueeze(-1), float('-inf'))
    w = torch.softmax(logits, dim=1)
    v = value.view(L, L, H, -1, value.shape[-1])
    z = torch.einsum('dsh,dshck->dhck', w, v).reshape(L, -1, value.shape[-1])
    zs = z.split([2 * d + 1 for d in att.value_fiber.degrees], dim=-1)
    z = {str(d): t for d, t in zip(att.value_fiber.degrees, zs)}
    nf = {'0': node0, '1': node1}
    z_cat = {k: torch.cat([z[k], nf[k]], dim=1) for k in z}
    x = ab.project(z_cat)
    x = norm(x)
    xf = torch.cat([x['0'], x['1']], dim=-1)
    fs = xf.unsqueeze(0).expand(L, L, *xf.shape[1:]).reshape(E, *xf.shape[1:])
    out = {}
    deg = maskf.sum(1)
    for gi, d_out in enumerate(fc.fiber_out.degrees):
        o = conv(2 + gi, fs, basis[f'out{d_out}_fused'], False)
        o = (o.reshape(L, L, *o.shape[1:]) * maskf.view(L, L, 1, 1)).sum(1)
        if fc.self_interaction and str(d_out) in fc.to_kernel_self:
            o = o + deg.view(L, 1, 1) * (fc.to_kernel_self[str(d_out)] @ x[str(d_out)])
        out[str(d_out)] = o
    return out


def str2str_forward_dense(self, msa, pair, R_in, T_in, xyz, state, idx, motif_mask, top_k=64, eps=1e-5, cyclic_reses=None,
                          se3_fn=None):
    """Replacement for rfdiffusion.Track_module.Str2Str.forward (B == 1).  Everything except the graph/SE(3) call is the
    stock code; TM.get_seqsep / TM.rbf are resolved at call time so the kit's fastpath memoisation stays in effect."""
    B, N, L = msa.shape[:3]
    if motif_mask is None:
        motif_mask = torch.zeros(L).bool()
    node = self.norm_msa(msa[:, 0])
    pair = self.norm_pair(pair)
    state = self.norm_state(state)
    node = torch.cat((node, state), dim=-1)
    node = self.norm_node(self.embed_x(node))
    pair = self.norm_edge1(self.embed_e1(pair))
    neighbor = TM.get_seqsep(idx, cyclic_reses)
    rbf_feat = TM.rbf(torch.cdist(xyz[:, :, 1], xyz[:, :, 1]))
    pair = torch.cat((pair, rbf_feat, neighbor), dim=-1)
    pair = self.norm_edge2(self.embed_e2(pair))                            # (B, L, L, Ce)
    # ---- dense SE(3)
    if top_k != 0:
        mask = topk_mask_dst_major(xyz[:, :, 1, :], idx, top_k=top_k)
        STATS['dense_topk_calls'] += 1
    else:
        mask = full_mask(L, xyz.device)
        STATS['dense_full_calls'] += 1
    ED, rel, l1 = dense_edge_inputs(pair, xyz)
    node0 = node.reshape(B * L, -1, 1)
    fn = se3_fn or se3_dense_forward
    shift = fn(self.se3.se3, node0, l1, ED, rel, mask)
    # ---- stock epilogue
    state = shift['0'].reshape(B, L, -1)
    offset = shift['1'].reshape(B, L, 2, 3)
    offset[:, motif_mask, ...] = 0
    delTi = offset[:, :, 0, :] / 10.0
    R = offset[:, :, 1, :] / 100.0
    Qnorm = torch.sqrt(1 + torch.sum(R * R, dim=-1))
    qA, qB, qC, qD = 1 / Qnorm, R[:, :, 0] / Qnorm, R[:, :, 1] / Qnorm, R[:, :, 2] / Qnorm
    delRi = torch.zeros((B, L, 3, 3), device=xyz.device)
    delRi[:, :, 0, 0] = qA * qA + qB * qB - qC * qC - qD * qD
    delRi[:, :, 0, 1] = 2 * qB * qC - 2 * qA * qD
    delRi[:, :, 0, 2] = 2 * qB * qD + 2 * qA * qC
    delRi[:, :, 1, 0] = 2 * qB * qC + 2 * qA * qD
    delRi[:, :, 1, 1] = qA * qA - qB * qB + qC * qC - qD * qD
    delRi[:, :, 1, 2] = 2 * qC * qD - 2 * qA * qB
    delRi[:, :, 2, 0] = 2 * qB * qD - 2 * qA * qC
    delRi[:, :, 2, 1] = 2 * qC * qD + 2 * qA * qB
    delRi[:, :, 2, 2] = qA * qA - qB * qB - qC * qC + qD * qD
    Ri = torch.einsum('bnij,bnjk->bnik', delRi, R_in)
    Ti = delTi + T_in
    alpha = self.sc_predictor(msa[:, 0], state)
    return Ri, Ti, state, alpha
