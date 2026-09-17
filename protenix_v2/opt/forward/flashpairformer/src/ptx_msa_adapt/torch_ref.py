"""ptx_msa_adapt.torch_ref — pure-torch implementations of the canonical API (ptx_msa_adapt.api) that replicate the STOCK
Protenix v2.0.0 forward op-for-op.  Two flavours per op:
  *_stockpath : identical op sequence to the stock module (same LayerNorm callable passed in, same F.linear /
                einsum / softmax calls, same in-place pattern) -> expected BITWISE vs stock.  This is what the
                adapter uses as its default backend, so `fpf.enable(ops=ptx_msa_adapt.ops_for_fpf())` with no external
                kernel plugged in must reproduce stock bit for bit (adapter op test record).
  *_math      : the same math written plainly (fp32 LayerNorm via F.layer_norm etc.) for readability / CPU tests;
                NOT bitwise (different LN kernel), same error class.
`ln` arguments are callables x -> LayerNorm(x) so the stock FusedLayerNorm module object can be passed straight in.
"""
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------------------------- OPM
def opm_stockpath(m, ln, w_a, w_b, w_out, b_out, *, eps_opm=1e-3, inplace_safe=True):
    """m [S,N,c_m] (bf16 under autocast) -> z_upd [N,N,c_z].  Mirrors OuterProductMean._forward with mask=None, chunk_size=None."""
    mask = m.new_ones(m.shape[:-1])            # [S,N]
    x = ln(m)                                  # [S,N,c_m]
    mask = mask.unsqueeze(-1)                  # [S,N,1]
    a = F.linear(x, w_a) ; a = a * mask        # the module's bias-free Linear == F.linear(x, W, None)
    b = F.linear(x, w_b) ; b = b * mask
    del x
    a = a.transpose(-2, -3)                    # [N,S,c]
    b = b.transpose(-2, -3)
    outer = torch.einsum("...bac,...dae->...bdce", a, b)          # [N,N,c,c]
    outer = outer.reshape(outer.shape[:-2] + (-1,))               # [N,N,c*c]
    outer = F.linear(outer, w_out, b_out)                         # [N,N,c_z]
    norm = torch.einsum("...abc,...adc->...bdc", mask, mask)      # [N,N,1]
    norm = norm + eps_opm
    if inplace_safe:
        outer /= norm
    else:
        outer = outer / norm
    return outer


def opm_math(m, w_ln, b_ln, w_a, w_b, w_out, b_out, *, eps_ln=1e-5, eps_opm=1e-3):
    """Plain statement of the OPM math (fp32 accumulate). m [S,N,c_m] any float dtype; returns m.dtype."""
    S = m.shape[-3]
    x = F.layer_norm(m.float(), (m.shape[-1],), w_ln.float(), b_ln.float(), eps_ln)
    a = x @ w_a.float().t()                    # [S,N,c]
    b = x @ w_b.float().t()
    # outer[i,j,c1,c2] = sum_s a[s,i,c1] b[s,j,c2]  == (a as [N*c, S]) @ (b as [S, N*c]) reshaped
    N, c = a.shape[-2], a.shape[-1]
    A = a.permute(1, 2, 0).reshape(N * c, S)   # [(i,c1), s]
    B = b.permute(0, 1, 2).reshape(S, N * c)   # [s, (j,c2)]
    O = (A @ B).reshape(N, c, N, c).permute(0, 2, 1, 3).reshape(N, N, c * c)
    z = O @ w_out.float().t() + b_out.float()
    z = z / (S + eps_opm)
    return z.to(m.dtype)


# ----------------------------------------------------------------------------------------------- PWA
def pwa_stockpath(m, z, ln_m, w_v, ln_z, w_z, w_g, w_o, *, n_heads, c):
    """m [S,N,c_m], z [N,N,c_z] -> m_upd [S,N,c_m].  Mirrors MSAPairWeightedAveraging.forward."""
    m = ln_m(m)
    v = F.linear(m, w_v)
    v = v.reshape(*v.shape[:-1], n_heads, c)
    b = F.linear(ln_z(z), w_z)                                    # [N,N,H]
    g = torch.sigmoid(F.linear(m, w_g))
    g = g.reshape(*g.shape[:-1], n_heads, c)
    w = torch.nn.functional.softmax(b, dim=-2)                    # nn.Softmax(dim=-2) == F.softmax(dim=-2) (autocast: fp32 out)
    wv = torch.einsum("...ijh,...mjhc->...mihc", w, v)
    o = g * wv
    o = o.reshape(*o.shape[:-2], n_heads * c)
    return F.linear(o, w_o)


def pwa_math(m, z, w_ln_m, b_ln_m, w_v, w_ln_z, b_ln_z, w_z, w_g, w_o, *, n_heads, c, eps=1e-5):
    S, N, cm = m.shape
    x = F.layer_norm(m.float(), (cm,), w_ln_m.float(), b_ln_m.float(), eps)
    v = (x @ w_v.float().t()).reshape(S, N, n_heads, c)
    zb = F.layer_norm(z.float(), (z.shape[-1],), w_ln_z.float(), b_ln_z.float(), eps) @ w_z.float().t()   # [N,N,H]
    w = torch.softmax(zb, dim=-2)                                                                            # over j
    g = torch.sigmoid(x @ w_g.float().t()).reshape(S, N, n_heads, c)
    # wv[s,i,h,:] = sum_j w[i,j,h] v[s,j,h,:]
    wv = torch.einsum("ijh,sjhc->sihc", w, v)
    o = (g * wv).reshape(S, N, n_heads * c)
    return (o @ w_o.float().t()).to(m.dtype)


# ----------------------------------------------------------------------------------------------- Transition (eval branch)
def transition_stockpath(x, ln, w_a, w_b, w_out, *, c_in):
    """Mirrors primitives.Transition.forward eval branch (chunk rule on x.shape[-2])."""
    other_dims = x.shape[:-1]
    dim_size = x.shape[-1]
    size = x.shape[-2]
    x = x.reshape(-1, dim_size)
    chunk_num = 1 if size < 3200 else 8
    chunks = torch.chunk(x, chunk_num, dim=-2)
    outputs = torch.empty((x.shape[0], c_in), dtype=x.dtype, device=x.device)
    start = 0
    for chunk in chunks:
        y = ln(chunk)
        a = F.linear(y, w_a)
        a = F.silu(a, True)
        b = F.linear(y, w_b)
        del y
        b *= a
        del a
        b = F.linear(b, w_out)
        outputs[start: start + b.shape[0]] = b
        start += b.shape[0]
    return outputs.reshape(*other_dims, c_in)


def transition_math(x, w_ln, b_ln, w_a, w_b, w_out, *, eps=1e-5):
    y = F.layer_norm(x.float(), (x.shape[-1],), w_ln.float(), b_ln.float(), eps)
    return ((F.silu(y @ w_a.float().t()) * (y @ w_b.float().t())) @ w_out.float().t()).to(x.dtype)
