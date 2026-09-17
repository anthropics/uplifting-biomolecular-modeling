"""The low-memory rewrites of the base variants' exact line: the O(L^2) sites of upstream's ``protein_mpnn_utils.py`` and of the kit
worker recomputed on [B, L, K] tensors (K = the 48 neighbours the model reads) with the same float32 arithmetic per element, so every
output byte is the ``exact`` line's and the working set is linear in the number of residues L.

The dense sites and their replacements (``install``):
  ProteinFeatures._dist      [B,L,L] masked CA-CA distances (3-4 live copies) before top-k  -> row slabs of at most CHUNK_ELEMS elements,
                             top-k per slab, concatenated (the slab rows are whole rows: the same values enter the same top-k)
  ProteinFeatures._get_rbf   [B,L,L] distances for each of the 24 atom pairs, then gathered at the neighbours  -> the neighbour coordinates
                             gathered first ([B,L,K,3]), the same sqrt(sum((a-b)**2) + 1e-6) per element
  ProteinFeatures.forward    [B,L,L] residue offsets and same-chain flags, then gathered  -> gathered per neighbour (integers)
  ProteinMPNN.forward/.sample and the worker's forward_scores/sample (``mask_attend``): one_hot(decoding_order) [B,L,L] x2,
                             einsum('ij,biq,bjp->bqp') with the [L,L] lower triangle, gather  -> step[b,i] = the decode step of residue i;
                             mask_attend[b,i,k] = 1.0 iff step[E_idx[b,i,k]] < step[b,i] (an integer comparison; the einsum's 0/1 sums are exact)
No random operation is added, removed or reordered (the RNG stream is the exact line's); no reduction changes its order. ``ProteinMPNN.tied_sample``
keeps its dense site (not reached by the exact route: tied positions run on the stock command line).

This file is also the text ``stage.stage_lowmem`` inlines into the staged worker (``kit/mpnn_worker2_lowmem.py``): it has no ``__future__``
import and imports torch inside its functions, so the package imports it without torch (the CPU tests) and the worker runs it as its own code.
"""
import inspect
import re
import textwrap

CHUNK_ELEMS = 2.0e8          # elements of one [B, rows, L] float32 distance slab (0.8 GB); rows = CHUNK_ELEMS // (B*L), at least 1

# upstream's three statements of the dense decoding-order mask (protein_mpnn_utils.py ProteinMPNN.forward / .sample @ the stock pin, and the
# worker's forward_scores), and the worker's own spelling in its sample(); each is replaced by MASK_CALL at the statement's indentation.
DENSE_MASK_UPSTREAM = (
    "permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()",
    "order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)",
    "mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)",
)
DENSE_MASK_WORKER = (
    "permutation_matrix_reverse = F.one_hot(decoding_order, num_classes=mask_size).float()",
    'order_mask_backward = torch.einsum("ij, biq, bjp->bqp", (1-torch.triu(torch.ones(mask_size, mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)',
    "mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)",
)
MASK_CALL = "mask_attend = _lm_mask_attend(decoding_order, E_idx)"
SITES = ("ProteinFeatures._dist", "ProteinFeatures._get_rbf", "ProteinFeatures.forward", "ProteinMPNN.forward", "ProteinMPNN.sample")


def mask_attend(decoding_order, E_idx):
    """[B,L,K,1] float32: 1.0 where neighbour E_idx[b,i,k] is decoded before residue i — the value the dense form gathers at j = E_idx[b,i,k]
    from order_mask_backward[b,i,j] = 1.0 iff step[j] < step[i] (step = the inverse permutation of decoding_order)."""
    import torch
    B, L = decoding_order.shape
    step = torch.empty_like(decoding_order)
    step.scatter_(1, decoding_order, torch.arange(L, device=decoding_order.device, dtype=decoding_order.dtype).unsqueeze(0).expand(B, -1))
    nb_step = torch.gather(step, 1, E_idx.reshape(B, -1)).view_as(E_idx)
    return (nb_step < step.unsqueeze(-1)).float().unsqueeze(-1)


def replace_dense_mask(src: str, statements=DENSE_MASK_UPSTREAM, expect: int = 1) -> str:
    """``src`` with the dense-mask block — the three ``statements`` on consecutive lines at one indentation — replaced by MASK_CALL at that
    indentation; the block must occur exactly ``expect`` times, else ValueError (the bytes changed: nothing is guessed)."""
    a, b, c = (re.escape(x) for x in statements)
    block = re.compile(r"(?m)^(?P<i>[ \t]*)" + a + r"[ \t]*\n(?P=i)" + b + r"[ \t]*\n(?P=i)" + c + r"[ \t]*\n")
    n = len(block.findall(src))
    if n != expect:
        raise ValueError(f"dense decoding-order mask block ({statements[0][:40]}...) found {n} time(s), expected {expect}")
    return block.sub(lambda m: m.group("i") + MASK_CALL + "\n", src)


def install(U, chunk_elems: float = CHUNK_ELEMS) -> list:
    """Rebind the dense sites of the imported ``protein_mpnn_utils`` module ``U`` to the low-memory forms; returns SITES.
    ProteinMPNN.forward/.sample are recompiled from their own source with the three mask statements replaced (``replace_dense_mask``),
    in U's namespace, so every other statement is upstream's byte for byte."""
    import numpy as np
    import torch
    gather_nodes = U.gather_nodes

    def _dist(self, X, mask, eps=1E-6):
        B, L = mask.shape[0], mask.shape[1]
        k = int(np.minimum(self.top_k, X.shape[1]))
        rows = max(1, int(chunk_elems // max(1, B * L)))
        D_out, E_out = [], []
        for i0 in range(0, L, rows):
            i1 = min(L, i0 + rows)
            mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask[:, i0:i1], 2)          # [b,i,j] = mask[b,j] * mask[b,i]
            dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X[:, i0:i1], 2)                       # [b,i,j,:] = X[b,j] - X[b,i]
            D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
            D_max, _ = torch.max(D, -1, keepdim=True)
            D_adjust = D + (1. - mask_2D) * D_max
            d, e = torch.topk(D_adjust, k, dim=-1, largest=False)
            D_out.append(d); E_out.append(e)
            del mask_2D, dX, D, D_max, D_adjust
        return torch.cat(D_out, 1), torch.cat(E_out, 1)

    def _get_rbf(self, A, B, E_idx):
        B_nb = gather_nodes(B, E_idx)                                                           # [B,L,K,3]: B[b, E_idx[b,i,k]]
        D_A_B_neighbors = torch.sqrt(torch.sum((A[:, :, None, :] - B_nb)**2, -1) + 1e-6)
        return self._rbf(D_A_B_neighbors)

    def forward(self, X, mask, residue_idx, chain_labels):
        if self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
        b = X[:, :, 1, :] - X[:, :, 0, :]
        c = X[:, :, 2, :] - X[:, :, 1, :]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + X[:, :, 1, :]
        Ca = X[:, :, 1, :]; N = X[:, :, 0, :]; C = X[:, :, 2, :]; O = X[:, :, 3, :]
        D_neighbors, E_idx = self._dist(Ca, mask)
        RBF_all = [self._rbf(D_neighbors)]
        for P, Q in ((N, N), (C, C), (O, O), (Cb, Cb), (Ca, N), (Ca, C), (Ca, O), (Ca, Cb), (N, C), (N, O), (N, Cb), (Cb, C), (Cb, O), (O, C),
                     (N, Ca), (C, Ca), (O, Ca), (Cb, Ca), (C, N), (O, N), (Cb, N), (C, Cb), (O, Cb), (C, O)):
            RBF_all.append(self._get_rbf(P, Q, E_idx))
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)
        offset = residue_idx[:, :, None] - gather_nodes(residue_idx[:, :, None], E_idx)[:, :, :, 0]
        E_chains = ((chain_labels[:, :, None] - gather_nodes(chain_labels[:, :, None], E_idx)[:, :, :, 0]) == 0).long()
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx

    U.ProteinFeatures._dist = _dist
    U.ProteinFeatures._get_rbf = _get_rbf
    U.ProteinFeatures.forward = forward
    for name in ("forward", "sample"):
        src = replace_dense_mask(textwrap.dedent(inspect.getsource(getattr(U.ProteinMPNN, name))))
        ns = dict(U.__dict__); ns["_lm_mask_attend"] = mask_attend
        exec(compile(src, f"<proteinmpnn_opt.lowmem:ProteinMPNN.{name}>", "exec"), ns)
        setattr(U.ProteinMPNN, name, ns[name])
    return list(SITES)
