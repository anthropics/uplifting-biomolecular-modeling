"""pallas_glut.py — the L-GLUT kernel of the kit's Pallas levers module (tokamax 0.0.12, jax 0.10.2):
the stock `tokamax.gated_linear_unit -> jnp.transpose(.,(2,0,1)) -> *= mask` as ONE Pallas-Triton kernel that stores its output transposed
and masked. Same tile config as stock (resolved by the same tokamax policy), same rounding points (f32 acc -> bf16 -> f32 -> act -> bf16 store),
mask multiply in the store dtype exactly as stock. Bit-exact on the kernel (all 54 tile configs x 3 sizes). The kernel and its wrapper below are
lines 28-74 of that file, bytes unchanged; the Haiku TriangleMultiplication subclass and apply_levers() that USE it stay in the kit.
Importing this module imports tokamax (its tile-config policy and pallas_call wrapper are what make the kernel bit-exact to the stock call).
"""
import functools
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu
import tokamax
from tokamax._src.ops.gated_linear_unit import pallas_triton as _gt
from tokamax._src.pallas import block as _block, grid as _grid

# ----------------------------------------------------------------------------- L-GLUT kernel
def _glu_T_kernel(x_ref, w_ref, mask_ref, out_ref, *, block_m, block_n, block_k, activation, precision):
    def body(i, acc):
        k_span = _block.ds(i, block_k)
        x = x_ref.at[:, k_span].load(bounds_check=(False, True))
        w = w_ref.at[k_span, 0].load(bounds_check=(True, False))
        v = w_ref.at[k_span, 1].load(bounds_check=(True, False))
        acc[0] += pl.dot(x, w.astype(x.dtype), precision=precision)
        acc[1] += pl.dot(x, v.astype(x.dtype), precision=precision)
        return acc
    num_iters = pl.cdiv(x_ref.shape[-1], block_k)
    acc0 = jnp.zeros((block_m, block_n), jnp.float32); acc1 = jnp.zeros((block_m, block_n), jnp.float32)
    gates, proj = jax.lax.fori_loop(0, num_iters, body, init_val=[acc0, acc1])
    gates = gates.astype(x_ref.dtype); proj = proj.astype(x_ref.dtype)          # stock rounding point 1 (f32 acc -> x dtype)
    proj = proj.astype(jnp.float32); gates = gates.astype(jnp.float32)
    out = proj * (gates if activation is None else activation(gates))           # stock f32 epilogue
    out = out.astype(out_ref.dtype)                                             # stock rounding point 2 (store dtype)
    if mask_ref is not None:
        out = out * mask_ref.load()[:, None]                                    # stock: `projection *= mask` in the act dtype
    out_ref.store(out.T)                                                        # transposed store == stock jnp.transpose(., (2, 0, 1))


def glu_transposed_masked(x, weights, mask2d, *, activation, config=None):
    """== jnp.transpose(tokamax.gated_linear_unit(x, weights, activation=activation), (2, 0, 1)) * mask2d[None]
    x [A, B, K], weights [K, 2, N] (fused: [:,0]=gate, [:,1]=projection), mask2d [A, B] -> out [N, A, B]."""
    lead = x.shape[:-1]
    x2 = jax.lax.collapse(x, 0, -1); m, k = x2.shape; n = weights.shape[-1]
    if config is None:
        # exactly the config the stock tokamax call would resolve (autotuning cache overlay -> cache -> heuristics)
        config = _gt.PallasTritonGatedLinearUnit().bind(x, weights, activation=activation).default_config
    bm, bn, bk = config.block_m, config.block_n, config.block_k
    get_pids = functools.partial(_grid.get_cheapest_grid_pids, grid_m=pl.cdiv(m, bm), grid_n=pl.cdiv(n, bn),
                                 block_m_cost=bm * k * jnp.dtype(x.dtype).itemsize, block_n_cost=bn * k * jnp.dtype(weights.dtype).itemsize * 2)
    wim = lambda fn: lambda pid: fn(*get_pids(pid))
    kern = functools.partial(_glu_T_kernel, block_m=bm, block_n=bn, block_k=bk, activation=activation, precision=None)
    in_specs = [pl.BlockSpec((bm, k), wim(lambda i, j: (i, 0))), pl.BlockSpec((k, 2, bn), wim(lambda i, j: (0, 0, j)))]
    ins = [x2, weights]
    if mask2d is not None:
        mask1 = jax.lax.collapse(mask2d, 0, mask2d.ndim).astype(x.dtype)
        in_specs.append(pl.BlockSpec((bm,), wim(lambda i, j: (i,)))); ins.append(mask1)
    else:
        in_specs.append(None); ins.append(None)
    name = "pallas_glu_T_" + getattr(activation, "__name__", "act") + ("_masked" if mask2d is not None else "")
    out = _block.pallas_call(kern, name=name, grid=(pl.cdiv(m, bm) * pl.cdiv(n, bn),), out_shape=jax.ShapeDtypeStruct((n, m), x.dtype),
                             in_specs=tuple(in_specs), out_specs=pl.BlockSpec((bn, bm), wim(lambda i, j: (j, i))), filter_specs=True,
                             compiler_params=plgpu.CompilerParams(num_warps=config.num_warps, num_stages=config.num_stages))(*ins)
    return out.reshape((n,) + tuple(lead))
