"""ATOM_ATTN — the diffusion sampler's atom cross-attention core (AtomCrossAttEncoder / Decoder: diffusion_transformer.CrossAttTransformer, 3
blocks each per denoising step; queries in subsets of 32 atoms attending the 128-atom key window around them, 4 heads x 32) through this
kit's fused windowed-attention Pallas kernel (inprocess/af3_jax_atom_attn_pallas.py, Triton backend) instead of the stock
einsum -> +bias +pair_logits -> softmax -> einsum with [S, H, 32, 128] f32 logits materialised twice per call
(alphafold3/model/network/diffusion_transformer.py ``cross_attention``). Shared strategy id ``F5.flash_attn_windowed``; the kernel is this
kit's own (KERNEL_FILE, loaded by file path).

What changes: only the attention core of ``diffusion_transformer.cross_attention`` — the adaptive layer norms of x_q / x_k, the q/k/v/gate
projections (names and shapes, hence the parameters read), the sigmoid gate and the adaptive-zero output are the stock lines, resolved
through the module at call time (so SAMPLER_BF16's rebinding of adaptive_layernorm / adaptive_zero_init composes: under it q/k/v are bf16, the
kernel's operands). Inside the kernel, per (sample, subset, head): f32 logits of the 32 x 128 window at the f32 dot's default precision
(word qk=f32; XLA's stock einsum is an f32 GEMM at default precision), + stock's mask term 1e9*(mask_q-1)(mask_k-1) + the block's pair logits,
softmax with f32 statistics over the whole window (one tile: no online re-association), weights cast to the activation dtype as stock
casts them, P.V accumulated in f32, output in the activation dtype. Query subsets whose 32 atoms are all padding are skipped (zeros
written): their rows never reach an observable output (encoder/decoder multiply by queries_mask; key windows hold real atoms only).
Numerics: NOT bitwise with stock (summation order, exp; padded-subset rows by design) — a fast-class (tolerance) lever.
Every call site is served because CrossAttTransformer.block names ``cross_attention`` as a module global: the diffusion head's atom encoder
(3 blocks) and decoder (3 blocks) per denoising step x 200 steps (x num_samples through hk.vmap), and the evoformer's per-recycle atom
encoder (token_atoms_act=None; same shapes). A call whose shapes the kernel does not tile (eligible() names the reason) takes the stock
lines — counted in report()["fallback"] by reason, printed on the SERVED line, never silent.

Switch: ``AF3_JAX_ATOM_ATTN=1`` in the model process's environment (set by the mode table for a composition that names ATOM_ATTN;
modes.TREE_LEVER_ENV; declared in stack.LEVER_SWITCH_ENV). Evidence: ``report()`` -> {"installed", "traced", "sites", "fallback", "cfg"} —
the tree's launcher prints ``atomattn=<traced calls>|off atomattn_sites=<site:n,...|none> atomattn_fallback=<reason:n,...|none>`` on its
SERVED line. Ablation: ``MODEL_OPT_LEVERS_OFF=ATOM_ATTN`` removes the variable (modes.with_levers_off); nothing is rebound.
"""
import importlib.util
import os
import sys

ENV_SWITCH = "AF3_JAX_ATOM_ATTN"
REBINDS = ("alphafold3.model.network.diffusion_transformer:cross_attention",)   # what install() rebinds (module:attribute) — the windowed atom cross-attention function; the launcher's INSTALL_ORDER table (fpf_launch.py) and tests/test_install_order.py read it
KERNEL_FILE = "af3_jax_atom_attn_pallas.py"                           # beside this module (inprocess/), loaded by file path (_load_kernel)
_STATE = {"installed": False, "traced": 0, "sites": {}, "fallback": {}, "stock": None, "kernel": None}


def wanted(environ=os.environ) -> bool:
    return environ.get(ENV_SWITCH, "") == "1"


def _load_kernel():
    if _STATE["kernel"] is None:
        name = "af3_jax_opt.inprocess.af3_jax_atom_attn_pallas"
        mod = sys.modules.get(name)
        if mod is None:
            spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(os.path.abspath(__file__)), KERNEL_FILE))
            mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
        _STATE["kernel"] = mod
    return _STATE["kernel"]


def _site(name) -> str:
    n = str(name)
    if "encoder" in n:
        return "evoformer_encoder" if n.startswith("evoformer") else "diffusion_encoder"
    if "decoder" in n:
        return "diffusion_decoder"
    return "other"


def _make_cross_attention(DT, kmod):
    """The stock ``cross_attention`` with its einsum/softmax core replaced by the fused kernel; every other line is the stock's."""
    import jax
    import jax.numpy as jnp
    hm = DT.hm
    stock = DT.cross_attention

    def cross_attention(x_q, x_k, mask_q, mask_k, config, global_config, pair_logits=None, single_cond_q=None, single_cond_k=None, name=""):
        assert len(mask_q.shape) == len(x_q.shape) - 1, f"{mask_q.shape}, {x_q.shape}"
        assert len(mask_k.shape) == len(x_k.shape) - 1, f"{mask_k.shape}, {x_k.shape}"
        assert config.key_dim % config.num_head == 0
        assert config.value_dim % config.num_head == 0
        key_dim = config.key_dim // config.num_head
        value_dim = config.value_dim // config.num_head
        H = config.num_head
        q_shape = tuple(x_q.shape[:-1]) + (H, key_dim); k_shape = tuple(x_k.shape[:-1]) + (H, key_dim); v_shape = tuple(x_k.shape[:-1]) + (H, value_dim)
        why = kmod.eligible(q_shape, k_shape, v_shape, None if pair_logits is None else tuple(pair_logits.shape))
        if why:                                                          # a shape the kernel does not tile: the stock lines, counted by reason (SERVED atomattn_fallback=)
            _STATE["fallback"][why] = _STATE["fallback"].get(why, 0) + 1
            return stock(x_q, x_k, mask_q, mask_k, config, global_config, pair_logits=pair_logits, single_cond_q=single_cond_q,
                         single_cond_k=single_cond_k, name=name)
        _STATE["traced"] += 1
        site = _site(name)
        _STATE["sites"][site] = _STATE["sites"].get(site, 0) + 1

        x_q = DT.adaptive_layernorm(x_q, single_cond_q, name=f"{name}q")
        x_k = DT.adaptive_layernorm(x_k, single_cond_k, name=f"{name}k")

        q = hm.Linear((H, key_dim), use_bias=True, name=f"{name}q_projection")(x_q)
        k = hm.Linear((H, key_dim), use_bias=False, name=f"{name}k_projection")(x_k)
        v = hm.Linear((H, value_dim), use_bias=False, name=f"{name}v_projection")(x_k)

        # core: q [S, 32, H, d], k / v [S, 128, H, d] in the activation dtype; the kernel forms the f32 logits (q * d**-0.5 . k), adds
        # stock's mask term and the block's pair logits [S, H, 32, 128], softmax over the 128 keys, weights in x_q's dtype, P.V in f32.
        weighted_avg = kmod.atom_attn(q, k, v.astype(q.dtype), mask_q, mask_k, pair_logits, scale=float(key_dim) ** -0.5)
        weighted_avg = jnp.asarray(weighted_avg, dtype=x_q.dtype)       # [S, 32, H * value_dim] (stock: the weights' dtype = x_q.dtype)

        gate_logits = hm.Linear(H * value_dim, bias_init=1.0, initializer="zeros", name=f"{name}gating_query")(x_q)
        weighted_avg *= jax.nn.sigmoid(gate_logits)

        return DT.adaptive_zero_init(weighted_avg, x_q.shape[-1], single_cond_q, global_config, name)

    return cross_attention


def install() -> bool:
    """Rebind ``diffusion_transformer.cross_attention`` (idempotent). Returns whether the lever is installed in this process."""
    if _STATE["installed"]:
        return True
    from alphafold3.model.network import diffusion_transformer as DT
    kmod = _load_kernel()
    _STATE["stock"] = DT.cross_attention
    DT.cross_attention = _make_cross_attention(DT, kmod)
    _STATE["installed"] = True
    return True


def uninstall() -> None:
    if _STATE["installed"]:
        from alphafold3.model.network import diffusion_transformer as DT
        DT.cross_attention = _STATE["stock"]
        _STATE["installed"] = False


def report() -> dict:
    k = _STATE["kernel"]
    return {"installed": _STATE["installed"], "traced": _STATE["traced"], "sites": dict(_STATE["sites"]), "fallback": dict(_STATE["fallback"]),
            "cfg": dict(k.DEFAULT_CFG) if k else {}}
