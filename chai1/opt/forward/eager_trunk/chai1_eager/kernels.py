"""Alternative implementations of trunk sub-ops for the structured eager trunk (plugged via chai1_eager.trunk.CFG).

trimul_bmm — same math as TriangleMultiplication.forward, but laid out so that no permute/clone copies are needed:
  * projections are produced channel-major ([4D, N*N] = W @ LN(z)^T, a cuBLAS 'TN' GEMM on the untouched row-major LN output),
  * gating/masking happen in that layout, the two triangle contractions are strided-batched matmuls over the D channel planes
    (outgoing: A_d @ B_d^T, incoming: A_d^T @ B_d — transposes are cuBLAS flags, not copies),
  * the only layout change left is folded into the fp32 up-cast that the no-affine LayerNorm needs anyway.
  torch.einsum in the export materialises 3 permuted copies per direction; this layout makes none of them."""
import torch
import torch.nn.functional as F
from .trunk import BF, F32, bfw, ln_bf16, ln_f32

__all__ = ["trimul_bmm"]


def trimul_bmm(mod, z, mask):
    B, N, _, C = z.shape
    D = mod.d
    zn = ln_bf16(z, C, mod.layernorm_z_in.weight, mod.layernorm_z_in.bias)   # [B,N,N,C] bf16: LN(z.to(f32)).to(bf16) (same kernel pair as stock, or the replica serving it fused)
    zf = zn.reshape(B, N * N, C)
    Wp, Wg = bfw(mod.merged_linear_p.weight), bfw(mod.merged_linear_g.weight)
    pT = torch.matmul(Wp, zf.transpose(1, 2))                     # [B, 4D, NN]  channel-major a1|b1|a2|b2
    gT = torch.sigmoid(torch.matmul(Wg[: 4 * D], zf.transpose(1, 2)))   # [B, 4D, NN]
    out_gate = torch.sigmoid(F.linear(zn, Wg[4 * D:]))            # [B,N,N,C] row-major (as stock) for the final gating
    abT = torch.mul(pT, gT).reshape(B, 4 * D, N, N)
    del pT, gT, zf, zn
    # the traced masks, applied IN PLACE on the two halves of abT (a1|b1 by mask[i,j], a2|b2 by mask[j,i]): the same zeros at the same positions
    # as the four out-of-place masked_fill copies of the D-planes, without the four [B,D,N,N] allocations and their copy passes (bitwise: the
    # matmuls below read identical operands; the planes are contiguous views of abT)
    abT[:, :2 * D].masked_fill_(torch.bitwise_not(mask).reshape(B, 1, N, N), 0)
    abT[:, 2 * D:].masked_fill_(torch.bitwise_not(mask.transpose(1, 2)).reshape(B, 1, N, N), 0)
    a1, b1, a2, b2 = abT[:, 0 * D:1 * D], abT[:, 1 * D:2 * D], abT[:, 2 * D:3 * D], abT[:, 3 * D:4 * D]
    del abT
    x1 = torch.matmul(a1, b1.transpose(-1, -2))                    # [B,D,N,N]  x1[d,i,j] = sum_k a1[d,i,k] b1[d,j,k]
    del a1, b1
    x2 = torch.matmul(a2.transpose(-1, -2), b2)                    # [B,D,N,N]  x2[d,i,j] = sum_k a2[d,k,i] b2[d,k,j]
    del a2, b2
    x1 = x1.permute(0, 2, 3, 1).to(F32, memory_format=torch.contiguous_format)   # layout change folded into the fp32 up-cast
    x2 = x2.permute(0, 2, 3, 1).to(F32, memory_format=torch.contiguous_format)
    x = torch.add(ln_f32(x1, D), ln_f32(x2, D))
    del x1, x2
    x = F.linear(x.to(BF), bfw(mod.linear_z_out.weight))
    return torch.mul(x, out_gate)
