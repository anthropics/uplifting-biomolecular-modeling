"""af3_pallas_levers.py — the `exact` mode's two Pallas levers for sokrypton/alphafold3 @ bc32b22f (tokamax 0.0.12, jax 0.10.2), installed into
the model process by af3_jax_opt/levers_launch.py (and by big_launch.py at the sites the memory mode vacates) as run-time class rebindings; the
stock tree is never modified. The launcher prints this module's sha256 on its LEVERS line.

L-GLUT  (env AF3P_GLU_T=1)      TriangleMultiplication: the stock `tokamax.gated_linear_unit -> jnp.transpose(.,(2,0,1)) -> *= mask`
                                 becomes ONE Pallas-Triton kernel that stores its output transposed and masked, served through the
                                 shared core's JAX-family provider (opt_core.kernels.pallas serve.glu_transposed_masked) by the ROW word
                                 `glut` (GLUT_WORD): the provider's tier words resolve the GLU family at c 128 only (cells
                                 af3_trimul_glu_c128_out256: exact = glut on cc 9.0 and 8.0); the template pair stack's c 64 GLU
                                 (af3_trimul_glu_c64_out128) has no measured cell, where a tier word names the stock XLA statement —
                                 not the bitwise kernel — so the exact tier keeps the kernel BY NAME until that cell is measured. Same
                                 tile config as stock (resolved by the same tokamax policy), same rounding points (f32 acc -> bf16 ->
                                 f32 -> act -> bf16 store), mask multiply in the store dtype exactly as stock. Bitwise at the kernel
                                 level (all 54 tile configs x 3 sizes) and end to end. A call the provider refuses by name
                                 (MODEL_OPT_LEVERS_OFF=pallas[:glut]) runs the stock class body, counted (ASIDE).
L-ATTNCFG (env AF3P_ATTN_CFG=bq,bk,warps,stages)  pins the Pallas-Triton flash-attention tile config for every
                                 tokamax.dot_product_attention call (= GridSelfAttention / pair attention only in AF3). The bitwise
                                 class: block_k=64 with num_warps in {4,8} (any block_q, any num_stages); the stock heuristic is
                                 64,64,4,2. The mode table pins 64,64,4,3.
The two switches are set by the kit's mode table only; a caller's own AF3P_* variables never reach the model process.
"""
import os
import jax, jax.numpy as jnp
import tokamax
from tokamax._src.ops.attention import pallas_triton as _pt

LEVERS = {}

# ----------------------------------------------------------------------------- L-GLUT kernel
# The kernel is the shared core's carried copy (opt_core/kernels/pallas_glut.py), reached through the core's JAX-family provider face
# opt_core.kernels.pallas.serve.glu_transposed_masked(x, weight, mask_t, activation=, word=GLUT_WORD); the launcher puts the pinned core on
# sys.path before this module is imported. The Haiku class below and apply_levers() are this add-on's.
try:
    from opt_core.kernels.pallas import Refusal as _Refusal, family as _family
    from opt_core.kernels.pallas import serve as _PS
except ImportError as _e:
    raise ImportError(f"af3_pallas_levers: the core's JAX-family provider (opt_core.kernels.pallas) is not importable ({_e}); put the pinned core directory "
                      f"(<tree>/common/opt_core, the kit's [tool.opt_core] path) on sys.path before importing this module") from _e
GLUT_WORD = "glut"      # the provider's ROW word for the bitwise GLU kernel: served by name at every triangle-multiplication site (pair c 128, template c 64); see the module doc for why not the tier word
GLUT_FORM = "af3"
ASIDE = {}              # reason -> calls that ran the stock class body instead (the provider refused the row by name before any parameter was read)


def glu_transposed_masked(x, weights, mask2d, *, activation, config=None):
    """The provider's GLU face by the row word GLUT_WORD in the carried kernel's signature: ``== transpose(tokamax.gated_linear_unit(x, weights,
    activation), (2,0,1)) * mask2d[None]`` (mask2d None: the transpose alone), bitwise. The memory mode's chunked triangle multiplication composes
    on this name (af3_jax_opt/big_levers.py TRIMUL_CHUNK ``glut_attr``)."""
    return _PS.glu_transposed_masked(x, weights, mask2d, activation=activation, word=GLUT_WORD, form=GLUT_FORM, strict=True, config=config)


def glut_selection(act):
    """The provider's decision for this call's GLU (the projection | gate pair: out = 2C), taken BEFORE any parameter is read; None = refused by
    name (counted in ASIDE): the caller runs the stock class body."""
    c = act.shape[-1]
    try:
        return _PS.resolve("glut", _family("glut", form=GLUT_FORM, c=int(c), out=2 * int(c)), act.dtype, int(act.shape[0]), word=GLUT_WORD, direction="fwd")
    except _Refusal as e:
        key = f"{getattr(e, 'row', None) or GLUT_WORD}:{getattr(e, 'kind', type(e).__name__)}"
        ASIDE[key] = ASIDE.get(key, 0) + 1
        return None


def make_patched_trimul_class(modules):
    """Subclass of the stock TriangleMultiplication whose __call__ is the stock body with the use_glu_kernel branch replaced by
    glu_transposed_masked (everything else verbatim). Defined in a class body so haiku's ModuleMetaclass wraps __call__ (name scope)."""
    class TriangleMultiplicationGLUT(modules.TriangleMultiplication):
        def __call__(self, act, mask):
            from alphafold3.model.components import haiku_modules as hm
            sel = glut_selection(act) if self.config.use_glu_kernel else None   # the provider's row for this call, decided before any parameter is read
            if self.config.use_glu_kernel and sel is None:                       # refused by name: the stock class body (its own parameters, its own kernel)
                return super().__call__(act, mask)
            mask_b = mask[None, ...]
            num_channels = act.shape[-1]
            equation = {'ikc,jkc->ijc': 'cik,cjk->cij', 'kjc,kic->ijc': 'ckj,cki->cij'}[self.config.equation]
            act = hm.LayerNorm(name='left_norm_input')(act)
            input_act = act
            if self.config.use_glu_kernel:
                weights_projection, _ = hm.haiku_linear_get_params(act, num_output=num_channels * 2, name='projection')
                weights_gate, _ = hm.haiku_linear_get_params(act, num_output=num_channels * 2, initializer=self.global_config.final_init, name='gate')
                weights_glu = jnp.stack([weights_gate, weights_projection], axis=1)
                if mask.dtype == act.dtype:
                    projection = _PS.glu_transposed_masked(act, weights_glu, mask, activation=jax.nn.sigmoid, word=GLUT_WORD, form=GLUT_FORM, selection=sel, strict=True)   # == transpose(glu) * mask[None] (same dtype)
                else:  # stock promotes on `projection *= mask`; keep that exact op outside the kernel
                    projection = _PS.glu_transposed_masked(act, weights_glu, None, activation=jax.nn.sigmoid, word=GLUT_WORD, form=GLUT_FORM, selection=sel, strict=True)   # == transpose(glu)
                    projection *= mask_b
            else:
                projection = hm.Linear(num_channels * 2, name='projection')(act)
                projection = jnp.transpose(projection, (2, 0, 1))
                projection *= mask_b
                gate = hm.Linear(num_channels * 2, name='gate', bias_init=1.0, initializer=self.global_config.final_init)(act)
                gate = jnp.transpose(gate, (2, 0, 1))
                projection *= jax.nn.sigmoid(gate)
            projection = projection.reshape(num_channels, 2, *projection.shape[1:])
            a, b = jnp.split(projection, 2, axis=1)
            a, b = jnp.squeeze(a, axis=1), jnp.squeeze(b, axis=1)
            act = jnp.einsum(equation, a, b)
            act = hm.LayerNorm(name='center_norm', axis=0, param_axis=0)(act)
            act = jnp.transpose(act, (1, 2, 0))
            act = hm.Linear(num_channels, initializer=self.global_config.final_init, name='output_projection')(act)
            gate_out = hm.Linear(num_channels, name='gating_linear', bias_init=1.0, initializer=self.global_config.final_init)(input_act)
            act *= jax.nn.sigmoid(gate_out)
            return act
    TriangleMultiplicationGLUT.__name__ = "TriangleMultiplication"; TriangleMultiplicationGLUT.__qualname__ = "TriangleMultiplication"
    return TriangleMultiplicationGLUT


def apply_levers():
    """Reads the two AF3P_* switches, installs the class rebindings, returns the dict of active levers (printed by the launcher)."""
    from alphafold3.model.network import modules
    if os.environ.get("AF3P_GLU_T", "0") == "1":
        modules.TriangleMultiplication = make_patched_trimul_class(modules)   # call sites look the name up in modules' globals
        LEVERS["L-GLUT"] = "TriangleMultiplication: fused transposed+masked Pallas-Triton GLU (stock tile config)"
    cfg = os.environ.get("AF3P_ATTN_CFG", "")
    if cfg:
        bq, bk, nw, ns = (int(v) for v in cfg.split(","))
        impl = _pt.PallasTritonFlashAttention(config=_pt.Config(block_q=bq, block_k=bk, num_warps=nw, num_stages=ns))
        orig = tokamax.dot_product_attention
        def pinned_dpa(*a, **kw):
            if kw.get("implementation", "triton") == "triton":
                kw["implementation"] = impl
            return orig(*a, **kw)
        class _TokamaxProxy:  # only modules.py (GridSelfAttention = pair attention) sees the pinned kernel; other tokamax users untouched
            dot_product_attention = staticmethod(pinned_dpa)
            def __getattr__(self, k): return getattr(tokamax, k)
        modules.tokamax = _TokamaxProxy()
        LEVERS["L-ATTNCFG"] = f"pair attention Pallas-Triton config pinned to block_q={bq}, block_k={bk}, num_warps={nw}, num_stages={ns} (stock heuristic 64,64,4,2)"
    return LEVERS
