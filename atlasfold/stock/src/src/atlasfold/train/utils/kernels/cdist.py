import torch
import triton
import triton.language as tl


def cdist(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute pairwise distances safely using Triton."""
    assert x.shape[:-2] == y.shape[:-2], "Batch dimensions must match"
    return CdistTriton.apply(x, y, eps)


# ---------------------------------------------------------------------------
# 1. Triton Forward Kernel
# ---------------------------------------------------------------------------
@triton.jit
def _cdist_fwd_kernel(
    x_ptr,
    y_ptr,
    dist_ptr,
    stride_xb,
    stride_xm,
    stride_xd,
    stride_yb,
    stride_yn,
    stride_yd,
    stride_db,
    stride_dm,
    stride_dn,
    M,
    N,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b_idx = tl.program_id(0)
    m_idx = tl.program_id(1)
    n_idx = tl.program_id(2)

    offs_m = m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = n_idx * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # Calculate base pointers for the current batch
    x_base = x_ptr + b_idx * stride_xb
    y_base = y_ptr + b_idx * stride_yb
    dist_base = dist_ptr + b_idx * stride_db

    # Load coordinates for x (shape: [BLOCK_M])
    x_x = tl.load(x_base + offs_m * stride_xm + 0 * stride_xd, mask=mask_m, other=0.0)
    x_y = tl.load(x_base + offs_m * stride_xm + 1 * stride_xd, mask=mask_m, other=0.0)
    x_z = tl.load(x_base + offs_m * stride_xm + 2 * stride_xd, mask=mask_m, other=0.0)

    # Load coordinates for y (shape: [BLOCK_N])
    y_x = tl.load(y_base + offs_n * stride_yn + 0 * stride_yd, mask=mask_n, other=0.0)
    y_y = tl.load(y_base + offs_n * stride_yn + 1 * stride_yd, mask=mask_n, other=0.0)
    y_z = tl.load(y_base + offs_n * stride_yn + 2 * stride_yd, mask=mask_n, other=0.0)

    # Compute differences via broadcasting in SRAM
    dx = x_x[:, None] - y_x[None, :]
    dy = x_y[:, None] - y_y[None, :]
    dz = x_z[:, None] - y_z[None, :]

    # Calculate safe distance
    sq_dist = dx * dx + dy * dy + dz * dz
    dist = tl.sqrt(sq_dist + eps)

    # Store results to HBM
    offs_d = offs_m[:, None] * stride_dm + offs_n[None, :] * stride_dn
    mask_d = mask_m[:, None] & mask_n[None, :]
    tl.store(dist_base + offs_d, dist, mask=mask_d)


# ---------------------------------------------------------------------------
# 2. Triton Backward Kernels (X and Y separately to avoid atomic operations)
# ---------------------------------------------------------------------------
@triton.jit
def _cdist_bwd_x_kernel(
    x_ptr,
    y_ptr,
    dist_ptr,
    grad_out_ptr,
    grad_x_ptr,
    stride_xb,
    stride_xm,
    stride_xd,
    stride_yb,
    stride_yn,
    stride_yd,
    stride_db,
    stride_dm,
    stride_dn,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b_idx = tl.program_id(0)
    m_idx = tl.program_id(1)

    offs_m = m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    x_base = x_ptr + b_idx * stride_xb
    y_base = y_ptr + b_idx * stride_yb
    dist_base = dist_ptr + b_idx * stride_db
    grad_out_base = grad_out_ptr + b_idx * stride_db
    grad_x_base = grad_x_ptr + b_idx * stride_xb

    x_x = tl.load(x_base + offs_m * stride_xm + 0 * stride_xd, mask=mask_m, other=0.0)
    x_y = tl.load(x_base + offs_m * stride_xm + 1 * stride_xd, mask=mask_m, other=0.0)
    x_z = tl.load(x_base + offs_m * stride_xm + 2 * stride_xd, mask=mask_m, other=0.0)

    grad_xx_acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    grad_xy_acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    grad_xz_acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    for n in range(0, N, BLOCK_N):
        offs_n = n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        y_x = tl.load(y_base + offs_n * stride_yn + 0 * stride_yd, mask=mask_n, other=0.0)
        y_y = tl.load(y_base + offs_n * stride_yn + 1 * stride_yd, mask=mask_n, other=0.0)
        y_z = tl.load(y_base + offs_n * stride_yn + 2 * stride_yd, mask=mask_n, other=0.0)

        offs_d = offs_m[:, None] * stride_dm + offs_n[None, :] * stride_dn
        mask_d = mask_m[:, None] & mask_n[None, :]

        # other=1.0 avoids division by zero for out-of-bounds elements
        dist = tl.load(dist_base + offs_d, mask=mask_d, other=1.0)
        grad_out = tl.load(grad_out_base + offs_d, mask=mask_d, other=0.0)

        dx = x_x[:, None] - y_x[None, :]
        dy = x_y[:, None] - y_y[None, :]
        dz = x_z[:, None] - y_z[None, :]

        grad_factor = grad_out / dist
        grad_xx_acc += tl.sum(grad_factor * dx, axis=1)
        grad_xy_acc += tl.sum(grad_factor * dy, axis=1)
        grad_xz_acc += tl.sum(grad_factor * dz, axis=1)

    tl.store(grad_x_base + offs_m * stride_xm + 0 * stride_xd, grad_xx_acc, mask=mask_m)
    tl.store(grad_x_base + offs_m * stride_xm + 1 * stride_xd, grad_xy_acc, mask=mask_m)
    tl.store(grad_x_base + offs_m * stride_xm + 2 * stride_xd, grad_xz_acc, mask=mask_m)


@triton.jit
def _cdist_bwd_y_kernel(
    x_ptr,
    y_ptr,
    dist_ptr,
    grad_out_ptr,
    grad_y_ptr,
    stride_xb,
    stride_xm,
    stride_xd,
    stride_yb,
    stride_yn,
    stride_yd,
    stride_db,
    stride_dm,
    stride_dn,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b_idx = tl.program_id(0)
    n_idx = tl.program_id(1)

    offs_n = n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    x_base = x_ptr + b_idx * stride_xb
    y_base = y_ptr + b_idx * stride_yb
    dist_base = dist_ptr + b_idx * stride_db
    grad_out_base = grad_out_ptr + b_idx * stride_db
    grad_y_base = grad_y_ptr + b_idx * stride_yb

    y_x = tl.load(y_base + offs_n * stride_yn + 0 * stride_yd, mask=mask_n, other=0.0)
    y_y = tl.load(y_base + offs_n * stride_yn + 1 * stride_yd, mask=mask_n, other=0.0)
    y_z = tl.load(y_base + offs_n * stride_yn + 2 * stride_yd, mask=mask_n, other=0.0)

    grad_yx_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    grad_yy_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    grad_yz_acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    for m in range(0, M, BLOCK_M):
        offs_m = m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        x_x = tl.load(x_base + offs_m * stride_xm + 0 * stride_xd, mask=mask_m, other=0.0)
        x_y = tl.load(x_base + offs_m * stride_xm + 1 * stride_xd, mask=mask_m, other=0.0)
        x_z = tl.load(x_base + offs_m * stride_xm + 2 * stride_xd, mask=mask_m, other=0.0)

        offs_d = offs_m[:, None] * stride_dm + offs_n[None, :] * stride_dn
        mask_d = mask_m[:, None] & mask_n[None, :]

        dist = tl.load(dist_base + offs_d, mask=mask_d, other=1.0)
        grad_out = tl.load(grad_out_base + offs_d, mask=mask_d, other=0.0)

        # Derivative for y is (y - x) / dist
        dx = y_x[None, :] - x_x[:, None]
        dy = y_y[None, :] - x_y[:, None]
        dz = y_z[None, :] - x_z[:, None]

        grad_factor = grad_out / dist
        grad_yx_acc += tl.sum(grad_factor * dx, axis=0)
        grad_yy_acc += tl.sum(grad_factor * dy, axis=0)
        grad_yz_acc += tl.sum(grad_factor * dz, axis=0)

    tl.store(grad_y_base + offs_n * stride_yn + 0 * stride_yd, grad_yx_acc, mask=mask_n)
    tl.store(grad_y_base + offs_n * stride_yn + 1 * stride_yd, grad_yy_acc, mask=mask_n)
    tl.store(grad_y_base + offs_n * stride_yn + 2 * stride_yd, grad_yz_acc, mask=mask_n)


# ---------------------------------------------------------------------------
# 3. PyTorch Autograd Wrapper
# ---------------------------------------------------------------------------
class CdistTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, eps=1e-8):
        x = x.contiguous()
        y = y.contiguous()

        x_shape = x.shape
        y_shape = y.shape
        batch_shape = x_shape[:-2]

        # Flatten any batch dimensions to 3D: [B, M/N, 3]
        x_flat = x.view(-1, x_shape[-2], 3)
        y_flat = y.view(-1, y_shape[-2], 3)

        B, M, _ = x_flat.shape
        _, N, _ = y_flat.shape

        dist_flat = torch.empty((B, M, N), device=x.device, dtype=x.dtype)

        grid = lambda meta: (  # noqa
            B,
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

        _cdist_fwd_kernel[grid](
            x_flat,
            y_flat,
            dist_flat,
            x_flat.stride(0),
            x_flat.stride(1),
            x_flat.stride(2),
            y_flat.stride(0),
            y_flat.stride(1),
            y_flat.stride(2),
            dist_flat.stride(0),
            dist_flat.stride(1),
            dist_flat.stride(2),
            M,
            N,
            eps,
            BLOCK_M=64,
            BLOCK_N=64,
        )

        ctx.save_for_backward(x_flat, y_flat, dist_flat)
        ctx.x_shape = x_shape
        ctx.y_shape = y_shape

        out_shape = batch_shape + (x_shape[-2], y_shape[-2])
        return dist_flat.view(out_shape)

    @staticmethod
    def backward(ctx, grad_out):
        x_flat, y_flat, dist_flat = ctx.saved_tensors
        grad_out = grad_out.contiguous().view_as(dist_flat)

        B, M, _ = x_flat.shape
        _, N, _ = y_flat.shape

        grad_x = torch.empty_like(x_flat)
        grad_y = torch.empty_like(y_flat)

        grid_x = lambda meta: (B, triton.cdiv(M, meta["BLOCK_M"]))  # noqa
        _cdist_bwd_x_kernel[grid_x](
            x_flat,
            y_flat,
            dist_flat,
            grad_out,
            grad_x,
            x_flat.stride(0),
            x_flat.stride(1),
            x_flat.stride(2),
            y_flat.stride(0),
            y_flat.stride(1),
            y_flat.stride(2),
            dist_flat.stride(0),
            dist_flat.stride(1),
            dist_flat.stride(2),
            M,
            N,
            BLOCK_M=64,
            BLOCK_N=64,
        )

        grid_y = lambda meta: (B, triton.cdiv(N, meta["BLOCK_N"]))  # noqa
        _cdist_bwd_y_kernel[grid_y](
            x_flat,
            y_flat,
            dist_flat,
            grad_out,
            grad_y,
            x_flat.stride(0),
            x_flat.stride(1),
            x_flat.stride(2),
            y_flat.stride(0),
            y_flat.stride(1),
            y_flat.stride(2),
            dist_flat.stride(0),
            dist_flat.stride(1),
            dist_flat.stride(2),
            M,
            N,
            BLOCK_M=64,
            BLOCK_N=64,
        )

        return grad_x.view(ctx.x_shape), grad_y.view(ctx.y_shape), None
