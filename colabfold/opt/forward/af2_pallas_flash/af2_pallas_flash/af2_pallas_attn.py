"""af2_pallas_attn.py — drop-in flash attention (fwd+bwd Pallas) for the alphafold(-colabfold) AF2 & AF-Multimer haiku modules.
Enable with env AF_PALLAS_ATTN=1 (or call enable()) BEFORE the model is jitted. Replaces the core of `Attention.__call__`
(logits einsum + bias + softmax + weighted sum) with af2_flash_pallas.flash_attention for calls that carry a nonbatched (pair) bias —
i.e. TriangleAttention starting/ending node and MSARowAttentionWithPairBias — and (optionally, AF_PALLAS_ATTN_ALL=1) also for MSAColumnAttention /
template attention (no pair bias). Projections, gating and output projection are untouched (same params, same einsums). Precision: the kernel takes
q,k,v,bias in the activations' dtype (bf16 under global_config.bfloat16=True, the models' precision policy) and accumulates in fp32; softmax is exact (online).
Falls back to the stock math when shapes are unsupported (key_dim != value_dim) or when running on CPU."""
import os, functools
import jax, jax.numpy as jnp
import haiku as hk
import af2_flash_pallas as K
# The differentiable op used by the patched Attention; a 1-element list so tests can swap it (e.g. K.make_flash_attention(precise_bwd=True)).
# Default: built at import from env (AF_PALLAS_ATTN_PRECISE_BWD / AF_PALLAS_ATTN_DBIAS_F32 / AF_PALLAS_ATTN_BWD_BATCH_CHUNK, see af2_flash_pallas.make_flash_attention).
FLASH_OP = [K.make_flash_attention()]

_STATE = {"enabled": False, "calls": 0, "fallbacks": 0}


def flash_attention_call(self, q_data, m_data, bias, nonbatched_bias=None, all_calls=True):
    """Functional form of the fused path for an AF2 `Attention` haiku module instance `self` (must be called inside the module's
    haiku context, i.e. from its __call__). Returns None when the call is ineligible (non-GPU backend, head_dim < 16 or
    key_dim != value_dim, non-AF2 bias form, or no pair bias while all_calls=False) so the caller can fall back to the stock math."""
    if jax.default_backend() != "gpu" or (nonbatched_bias is None and not all_calls):
        return None
    key_dim = self.config.get('key_dim', int(q_data.shape[-1]))
    value_dim = self.config.get('value_dim', int(m_data.shape[-1]))
    num_head = self.config.num_head
    key_dim = key_dim // num_head; value_dim = value_dim // num_head
    if key_dim != value_dim or key_dim < 16 or bias.ndim != 4 or bias.shape[1] != 1 or bias.shape[2] != 1:   # Triton dot needs dims >= 16 (template-stack triangle attention has 8/head) -> stock path
        return None
    glorot_uniform = lambda: hk.initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform')
    q_weights = hk.get_parameter('query_w', shape=(q_data.shape[-1], num_head, key_dim), dtype=q_data.dtype, init=glorot_uniform())
    k_weights = hk.get_parameter('key_w', shape=(m_data.shape[-1], num_head, key_dim), dtype=q_data.dtype, init=glorot_uniform())
    v_weights = hk.get_parameter('value_w', shape=(m_data.shape[-1], num_head, value_dim), dtype=q_data.dtype, init=glorot_uniform())
    # heads-major projections (same contraction as stock 'bqa,ahc->bqhc', different output layout; the 1/sqrt(d) scale is applied inside the kernel in fp32)
    q = jnp.einsum('bqa,ahc->bhqc', q_data, q_weights)
    k = jnp.einsum('bka,ahc->bhkc', m_data, k_weights)
    v = jnp.einsum('bka,ahc->bhkc', m_data, v_weights)
    kmask = bias[:, 0, 0, :] > -1e8          # stock: bias = 1e9 * (mask - 1)
    sq, sk = q.shape[2], k.shape[2]
    nb = jnp.zeros((num_head, sq, sk), q.dtype) if nonbatched_bias is None else nonbatched_bias.astype(q.dtype)
    weighted_avg = FLASH_OP[0](q, k, v, nb, kmask, float(key_dim ** (-0.5)))     # [b,h,q,c]
    weighted_avg = jnp.swapaxes(weighted_avg, 1, 2)                                   # [b,q,h,c]
    if self.global_config.zero_init:
        init = hk.initializers.Constant(0.0)
    else:
        init = glorot_uniform()
    if self.config.gating:
        gating_weights = hk.get_parameter('gating_w', shape=(q_data.shape[-1], num_head, value_dim), dtype=q_data.dtype, init=hk.initializers.Constant(0.0))
        gating_bias = hk.get_parameter('gating_b', shape=(num_head, value_dim), dtype=q_data.dtype, init=hk.initializers.Constant(1.0))
        gate_values = jnp.einsum('bqc, chv->bqhv', q_data, gating_weights) + gating_bias
        weighted_avg *= jax.nn.sigmoid(gate_values)
    o_weights = hk.get_parameter('output_w', shape=(num_head, value_dim, self.output_dim), dtype=q_data.dtype, init=init)
    o_bias = hk.get_parameter('output_b', shape=(self.output_dim,), dtype=q_data.dtype, init=hk.initializers.Constant(0.0))
    return jnp.einsum('bqhc,hco->bqo', weighted_avg, o_weights) + o_bias


def _make_call(orig_call, all_calls):
    def __call__(self, q_data, m_data, bias, nonbatched_bias=None):
        out = flash_attention_call(self, q_data, m_data, bias, nonbatched_bias, all_calls=all_calls) if _STATE["enabled"] else None
        if out is None:
            if _STATE["enabled"]: _STATE["fallbacks"] += 1
            return orig_call(self, q_data, m_data, bias, nonbatched_bias)
        _STATE["calls"] += 1
        return out
    return __call__


def enable(modules_list=None, all_calls=None):
    """Route AF2 Attention through the Pallas flash kernel. Patches the module objects in `modules_list`
    (default: alphafold.model.modules if importable).
    Implementation note (haiku): the replacement class is created through haiku's metaclass with the SAME class
    name 'Attention', so the auto-derived module name stays 'attention' and stock parameter trees apply unchanged;
    the module-global name `Attention` is rebound so TriangleAttention / MSARowAttentionWithPairBias /
    MSAColumnAttention pick it up at call time. Call BEFORE the model function is jitted/traced."""
    if all_calls is None:
        all_calls = os.environ.get("AF_PALLAS_ATTN_ALL", "0") == "1"
    mods = []
    if modules_list is None:
        for name in ["alphafold.model.modules"]:
            try:
                mods.append(__import__(name, fromlist=["x"]))
            except Exception:
                pass
    else:
        mods = list(modules_list)
    for m in mods:
        A = m.Attention
        if getattr(A, "_pallas_patched", False):
            continue
        _patched_call = _make_call(A.__call__, all_calls)
        # The method must be defined INSIDE a class body so haiku's module metaclass wraps it (enters the module's name/parameter
        # scope); assigning __call__ after class creation bypasses the wrapping and hk.get_parameter would run in the CALLER's scope.
        # The class keeps the name 'Attention' so haiku's auto-derived module name ('attention') and hence the parameter tree are unchanged.
        class Attention(A):
            _pallas_patched = True
            _stock_cls = A
            def __call__(self, q_data, m_data, bias, nonbatched_bias=None):
                return _patched_call(self, q_data, m_data, bias, nonbatched_bias)
        Attention.__qualname__ = "Attention"; Attention.__module__ = A.__module__
        m.Attention = Attention
    _STATE["enabled"] = True
    return mods


def disable(modules_list=None):
    """Restore the stock Attention class (for A/B tests in one process; re-jit afterwards)."""
    mods = []
    if modules_list is None:
        for name in ["alphafold.model.modules"]:
            try:
                mods.append(__import__(name, fromlist=["x"]))
            except Exception:
                pass
    else:
        mods = list(modules_list)
    for m in mods:
        A = m.Attention
        if getattr(A, "_pallas_patched", False):
            m.Attention = A._stock_cls
    _STATE["enabled"] = False


if os.environ.get("AF_PALLAS_ATTN", "0") == "1":
    enable()
