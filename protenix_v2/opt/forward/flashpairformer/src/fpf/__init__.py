"""fpf — FlashPairformer shared interface: the op registry that routes stock module methods to provider callables.

REGISTRY (op names are the contract; signatures = the STOCK Protenix v2.0.0 module methods they replace; shapes for the protenix-v2 checkpoint,
bf16 autocast, cuequivariance trimul + triangle attention, LAYERNORM_TYPE=fast_layernorm, inplace_safe=True, chunk_size=None (<=~1500 tok), pair_mask=None):

  name                    replaces (monkeypatch target)                                          call signature (eval, no grad)                                   dims (protenix-v2; RESOLVED AT RUNTIME by fpf.dims(model))
  ----------------------  ---------------------------------------------------------------------  ----------------------------------------------------------------  -----------------------------------------------------------
  trimul_out / trimul_in  protenix.model.triangular.triangular.TriangleMultiplicationOutgoing /  forward(z[N,N,c_z] bf16, mask=None, inplace_safe=True,           pairformer: c_z=256, c_hidden=256 (hidden_scale_up); msa pair stack: same; template stack: c=64
                          ...Incoming .forward                                                     _add_with_inplace=True, triangle_multiplicative='cuequivariance') -> z_new [N,N,c_z] (= z + update; stock returns z+z_in)
  triatt_start/triatt_end protenix.model.triangular.triangular.TriangleAttention .forward         forward(x[N,N,c_z] bf16, mask=None, chunk_size=None, triangle_attention='cuequivariance', inplace_safe=True) -> update [N,N,c_z] (caller does z += )
                          (TriangleAttentionEndingNode subclass: starting=False; input is the caller-transposed z)                                              pairformer: H = c_z//32 = 8, D = 32 (hidden_scale_up: no_heads_pair = c_z // c_hidden_pair_att); template (c=64): H=2? -> RESOLVE via fpf.dims
  pair_transition         protenix.model.modules.primitives.Transition .forward (instances named *.pair_transition, msa *.transition_m, template)  forward(x[..., c]) -> update (caller adds)      pair: c=256,n=4 (hidden 1024); msa transition_m: c_m=64? (protenix-v2 msa c_m=128) ; template c=64 n=2 -> RESOLVE
  single_attention_pb     protenix.model.modules.transformer.AttentionPairBias .forward (pairformer, has_s=False)  forward(a[N,c_s], s=None, z[N,N,c_z]) -> a_update        c_s=384, n_heads=16, z->bias: LN(c_z, no offset) + Linear(c_z->16)
  single_transition       primitives.Transition (c_in=c_s=384, n=4)
  outer_product_mean      protenix.model.triangular.layers.OuterProductMean .forward(m[n_msa,N,c_m], mask=None, chunk_size, inplace_safe) -> z_update [N,N,c_z]   c_m -> c_hidden=32 (x c_hidden) -> c_z
  msa_pair_weighted_avg   protenix.model.modules.pairformer.MSAPairWeightedAveraging .forward(m, z) -> m_update                                       c_m, c=32? heads=8 -> RESOLVE
  z_transpose             pairformer.py `z.transpose(-2,-3).contiguous()` x2 per block (the ZT lever's tiled-transpose kernel, exact)
  layernorm_z             every LayerNorm(c_z) on the pair tensor (tri-att x2, transition x1, trimul in/out x2 each inside cuEq, APB layernorm_z x1) = 8 LN passes over z per Pairformer block (stock)

STOCK PASS COUNT over z per Pairformer block (inplace path; 'pass' = full read or write of [N,N,c_z]):  trimul x2: clone(r+w) + LN-in(r+w ch-major) + proj GEMM(r) ... + out(w) + add(2r+w) ~= 9 each;
tri-att x2: LN(r+w) + bias-linear(r) + qkvg GEMMs(r x4 stock / x1 with T2g) + o(w) + gate-mul(2r+w) + out-proj(r+w) + residual(2r+w) ~= 12 each; transposes 2x(r+w); transition: LN(r+w)+GEMM a,b (2r, 8w hidden) + silu*mul (16r,8w hidden-width) + GEMM(8r + w) + residual(2r+w); APB: LN(z)(r+w)+linear(r).
=> ~60 z-sized HBM passes per block (stock), vs a Level-3 floor of ~14 (each of the 5 sub-modules: read z once, write z once, + tri-att q/k/v/o traffic).

ADAPTER:  fpf.enable(ops={"pair_transition": callable, ...}) monkeypatches the stock classes so that `callable(module, *args, **kw)` is invoked instead of the stock forward
(module = the stock nn.Module instance: read its weights from it; cache your packed weights on the instance, e.g. module._fpf_cache). fpf.disable() restores. Env hook:
FPF_OPS="pair_transition=mypkg.mod:fn,triatt=..." (import path : attribute).
"""
import os, importlib, torch
_TARGETS = {
    "trimul_out": ("protenix.model.triangular.triangular", "TriangleMultiplicationOutgoing", "forward"),
    "trimul_in": ("protenix.model.triangular.triangular", "TriangleMultiplicationIncoming", "forward"),
    "triatt": ("protenix.model.triangular.triangular", "TriangleAttention", "forward"),            # both starting/ending (self.starting tells which)
    "transition": ("protenix.model.modules.primitives", "Transition", "forward"),                  # pair, single, msa, template transitions (dispatch on self.c_in if you only cover some)
    "single_attention_pb": ("protenix.model.modules.transformer", "AttentionPairBias", "forward"),
    "outer_product_mean": ("protenix.model.triangular.layers", "OuterProductMean", "forward"),
    "msa_pair_weighted_avg": ("protenix.model.modules.pairformer", "MSAPairWeightedAveraging", "forward"),
    "pairformer_block": ("protenix.model.modules.pairformer", "PairformerBlock", "forward"),         # Level-3 whole-block replacement
}
_ORIG = {}
STATS = {}
def targets(): return dict(_TARGETS)
def enable(ops):
    """ops: {registry_name: callable(module, *args, **kwargs)}.  The callable MUST accept exactly the stock signature (see registry) and return the stock return value."""
    for name, fn in ops.items():
        modname, clsname, attr = _TARGETS[name]
        cls = getattr(importlib.import_module(modname), clsname)
        if (name, attr) not in _ORIG: _ORIG[(name, attr)] = (cls, getattr(cls, attr))
        orig = _ORIG[(name, attr)][1]
        def make(fn, orig, name):
            def f(self, *a, **kw):
                STATS[name] = STATS.get(name, 0) + 1
                return fn(self, *a, **kw)
            f._fpf_orig = orig
            return f
        setattr(cls, attr, make(fn, orig, name))
    return report()
def disable(names=None):
    for (name, attr), (cls, orig) in list(_ORIG.items()):
        if names is None or name in names:
            setattr(cls, attr, orig); del _ORIG[(name, attr)]
def original(name):
    """the stock (unpatched) method for fallback inside your callable: fpf.original('transition')(module, x)"""
    modname, clsname, attr = _TARGETS[name]
    if (name, attr) in _ORIG: return _ORIG[(name, attr)][1]
    return getattr(getattr(importlib.import_module(modname), clsname), attr)
def report(): return {"enabled": sorted(n for n, _ in _ORIG), "calls": dict(STATS)}
def enable_from_env():
    spec = os.environ.get("FPF_OPS", "").strip()
    if not spec: return None
    ops = {}
    for item in spec.split(","):
        name, target = item.split("="); modpath, attr = target.split(":")
        ops[name.strip()] = getattr(importlib.import_module(modpath.strip()), attr.strip())
    return enable(ops)
def dims(model):
    """Resolve every registry dim from a built Protenix model (authoritative; do not hardcode)."""
    import protenix.model.modules.pairformer as PF
    out = {}
    pb = model.pairformer_stack.blocks[0]
    out["pairformer"] = {"n_blocks": len(model.pairformer_stack.blocks), "c_z": pb.tri_mul_out.c_z, "trimul_c_hidden": pb.tri_mul_out.c_hidden,
                         "triatt_heads": pb.tri_att_start.no_heads, "triatt_c_hidden": pb.tri_att_start.c_hidden, "triatt_c_in": pb.tri_att_start.c_in,
                         "pair_transition": {"c_in": pb.pair_transition.c_in, "n": pb.pair_transition.n},
                         "single": None if pb.c_s <= 0 else {"c_s": pb.c_s, "apb_heads": pb.attention_pair_bias.n_heads, "single_transition": {"c_in": pb.single_transition.c_in, "n": pb.single_transition.n}}}
    mm = model.msa_module; mb = mm.blocks[0]
    out["msa_module"] = {"n_blocks": mm.n_blocks, "c_m": mm.c_m, "opm_c_hidden": mb.outer_product_mean_msa.c_hidden, "opm_c_z": mb.outer_product_mean_msa.c_z,
                         "pwa": None if mb.is_last_block else {"c": mb.msa_stack.msa_pair_weighted_averaging.c, "heads": mb.msa_stack.msa_pair_weighted_averaging.n_heads, "c_z": mb.msa_stack.msa_pair_weighted_averaging.c_z},
                         "transition_m": None if mb.is_last_block else {"c_in": mb.msa_stack.transition_m.c_in, "n": mb.msa_stack.transition_m.n},
                         "pair_stack": {"c_z": mb.pair_stack.tri_mul_out.c_z, "trimul_c_hidden": mb.pair_stack.tri_mul_out.c_hidden, "triatt_heads": mb.pair_stack.tri_att_start.no_heads, "triatt_c_hidden": mb.pair_stack.tri_att_start.c_hidden},
                         "msa_configs": dict(getattr(mm, "msa_configs", {}))}
    te = model.template_embedder; tb = te.pairformer_stack.blocks[0] if te.n_blocks > 0 else None
    out["template_embedder"] = {"n_blocks": te.n_blocks, "c": te.c, "c_z": te.c_z,
                                "pair_stack": None if tb is None else {"c_z": tb.tri_mul_out.c_z, "trimul_c_hidden": tb.tri_mul_out.c_hidden, "triatt_heads": tb.tri_att_start.no_heads, "triatt_c_hidden": tb.tri_att_start.c_hidden, "transition": {"c_in": tb.pair_transition.c_in, "n": tb.pair_transition.n}}}
    out["N_cycle"] = None
    return out
