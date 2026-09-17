"""ptx_msa_adapt — FlashPairformer MSA-module op ADAPTERS for Protenix v2 (fpf registry names
`outer_product_mean`, `msa_pair_weighted_avg`, `transition` (msa transition_m instances only)).

Purpose: the MSA kernels themselves live elsewhere (opt_core.ops.msa_fused);
this package is the Protenix-side glue through which such kernels drop in:

  1. CANONICAL FUNCTIONAL API (ptx_msa_adapt.api): one plain function per op with explicit weight tensors,
     dtypes and layouts documented — this is the signature a kernel implements:
        opm_fn(m, w_ln, b_ln, w_a, w_b, w_out, b_out, *, eps_ln, eps_opm) -> z_update [N,N,c_z]
        pwa_fn(m, z, w_ln_m, b_ln_m, w_v, w_ln_z, b_ln_z, w_z, w_g, w_o, *, n_heads, c, eps) -> m_update [S,N,c_m]
        transition_fn(x, w_ln, b_ln, w_a, w_b, w_out, *, eps) -> update [..., c]
  2. STOCK-PATH REFERENCE implementations of that API in pure torch (ptx_msa_adapt.torch_ref) that replicate the
     stock module forward op-for-op (same LayerNorm kernel, same F.linear/einsum/softmax calls in the
     same order) => expected BITWISE vs stock; used to test the adapter itself adds zero numerics change.
  3. STOCK-SIGNATURE WRAPPERS (ptx_msa_adapt.adapters) = what fpf.enable(ops={...}) receives:
        fn(module, *stock_args, **stock_kwargs) -> stock return value
     They read weights from the stock nn.Module instance (cached once on module._fpf_cache, never mutating
     parameters), route to a pluggable backend (set_backend("opm", callable) or PTX_MSA_OPM=pkg.mod:fn env),
     and fall back to the stock forward outside the supported envelope (training mode, fp16 autocast branch,
     masks given, chunked OPM above the dynamic chunk threshold) so semantics are unchanged everywhere.

Env hooks (read at first call):  PTX_MSA_OPM=pkg.mod:fn  PTX_MSA_PWA=pkg.mod:fn  PTX_MSA_TRANS=pkg.mod:fn
In-model:  FPF_OPS="outer_product_mean=ptx_msa_adapt.adapters:opm,msa_pair_weighted_avg=ptx_msa_adapt.adapters:pwa,transition=ptx_msa_adapt.adapters:transition_msa"
"""
from .adapters import opm, pwa, transition_msa, set_backend, get_backend, backends_report, ops_for_fpf  # noqa: F401
__version__ = "0.1"
