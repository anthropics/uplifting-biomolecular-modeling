"""ptx_fpf_v02.py — FlashPairformer EXACT-arm additions (Protenix 2.0.0).  All env-gated, monkeypatch-only,
applied by sitecustomize AFTER ptx_trunk2_levers.apply_from_env() (so the BLK2 block path is already installed on PairformerBlock.forward).
Every lever keeps the stock module objects/weights; each has a counter in report().

PTX_V02_MSABLK=1   (lever 1; a NO-OP on the pinned stack — OPM returns bf16 under bf16 autocast, so the MSA pair stack already takes BLK2; kept as a diagnostic/robustness path only) MSA-module pair stack (4 PairformerBlock(c_z=256, c_s=0) x 10 recycles = 40 block calls/prediction) onto the BLK2 block path.
                   Diagnosis: the BLK2 guard `z.dtype != torch.bfloat16` trips — inside MSABlock, `z = z + self.outer_product_mean_msa(...)` promotes z to
                   fp32 for the whole MSA module when OPM returns fp32 (an OPM that runs its einsum/linear_out outside bf16), so pair_stack receives an
                   fp32 z (pair_mask None, inplace True, chunk None are all fine <=1024 tok).  Guard census (PTX_V02_CENSUS=1) prints the tripping term per call class.
                   Exact handling: the stock fp32-z block is  z += bf16_update  four times (tmu_out, tmu_in via cuEq wrapper add; tri_att start/end; transition):
                   every sub-module casts z->bf16 at its first autocast op (LN / linear), so the UPDATE tensors are pure functions of bf16(z) EXCEPT that the residual
                   accumulates in fp32.  => we run the BLK2 statements on zb = z.to(bf16) for the *producers* and apply every residual into the fp32 master z with the
                   stock op (`z += u` / `z = z + u`), re-deriving zb = z.to(bf16) before the next statement (one fused cast kernel; replaces the cast every stock
                   sub-module does internally anyway).  Statement list and order == stock inplace branch.  If OPM returns bf16 on some stack (z stays bf16) the
                   BLK2 path already takes it and this lever is a no-op (counter msablk_bf16_direct).
PTX_V02_TMPL=1     (lever 2a) template pair stack (c=64, hidden_scale_up -> H=2,D=32; 2 blocks x 1 distinct template (kit L2) x 10 cycles = 20 block calls) onto a
                   c=64 block path: stock cuEq TriMul x2 -> [stock LN -> stock q/k/v/g/bias GEMMs -> cuEq attention(mask = stock all-ones->None only if NOMASK cell
                   checked) -> stock sigmoid/o*g/linear_o] with the two z.transpose(-2,-3).contiguous() passes REPLACED by strided views (tri_att_end reads z^T
                   through strides; its LN makes the contiguous copy exactly like stock's LN on the transposed-contiguous tensor: same rows, same kernel) and the
                   residual added in x-frame.  No Triton kernel at c=64 (prologue/epilogue/transition cells for (64,64)/(64,128) are NOT in the checked tables and
                   stay off) => arithmetic is op-for-op stock; only copies removed.  pair_mask is all-ones here (TemplateEmbedder builds it) -> handled: mask given
                   => stock TriangleAttention.forward with mask (we do not drop a real mask).
PTX_V02_CONF=1     (lever 2b; WITHDRAWN: the confidence pairformer runs under bf16 autocast in protenix-v2 (skip_amp.confidence_head=False) and already takes BLK2; enabling this would DOWNGRADE it to stock kernels — do not set) confidence-head 4-block pairformer (c_z=256, c_s=384, runs once per sample under autocast DISABLED => fp32 z, fp32 GEMMs):
                   fp32 => none of the bf16-tested kernels apply (different arithmetic).  What is exact in fp32: removing the two transpose().contiguous()
                   passes per block (pure data movement) exactly as in TMPL.  => same strided block path, dtype-agnostic, stock kernels.  Counter conf_blk_calls.
PTX_V02_GLUE=1     (lever 4) trunk glue, exact by construction:
                     (a) recycle prologue `z = z_init + linear_no_bias_z_cycle(layernorm_z_cycle(z))`: unchanged ops; we only pre-allocate nothing — SKIPPED (no exact gain found)
                     (b) MSAModule: `torch.cuda.empty_cache()` is NOT called in v2.0.0 trunk (only confidence head >2000 tok) — nothing to remove.
                     (c) PairformerStack._prep_blocks builds 48 functools.partial per call (host only, ~30 us) — negligible, untouched.
                   => GLUE reduces to (d): OPM fp32->bf16 cast hoist is NOT exact (changes where rounding happens) — REJECTED by rule.  The lever therefore only
                   installs counters (glue_census) so the report shows the host-side overhead per recycle; no numerics path is changed.
PTX_V02_GRAPH=1    (lever 3) per-recycle CUDA-graph replay of the 48-block Pairformer stack — see graph_trunk.py (separate module; needs the stream-correct fast-LN
                   prebuilt .so (third_party/fastln_prebuilt*); bitwise-vs-eager is asserted per capture in-process and the lever REFUSES (eager fallback, counter) on mismatch).
"""
import os, sys, math, json, time
import torch
import torch.nn.functional as F

STATS = {"applied": [], "msablk_calls": 0, "msablk_bf16_direct": 0, "msablk_fallback": 0, "msablk_fallback_reasons": {},
         "tmpl_blk_calls": 0, "tmpl_fallback": 0, "tmpl_fallback_reasons": {}, "conf_blk_calls": 0, "conf_fallback": 0, "conf_fallback_reasons": {}, "census": {}}
_CENSUS = os.environ.get("PTX_V02_CENSUS", "0") == "1"


def report():
    r = json.loads(json.dumps(STATS, default=str))
    try:
        g = sys.modules.get("fpf_stackgraph.stackgraph") or sys.modules.get("graph_trunk")
        if g is not None:
            r["graph"] = g.report()
    except Exception as e:
        r["graph_report_err"] = repr(e)
    return r


def _why(d, key):
    d[key] = d.get(key, 0) + 1


def _census(name, **kw):
    if not _CENSUS:
        return
    c = STATS["census"].setdefault(name, {"n": 0, "first": None})
    c["n"] += 1
    if c["first"] is None:
        c["first"] = {k: (str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v) for k, v in kw.items()}


# ============================================================================================ shared: strided (transpose-free) stock block, dtype-agnostic
def _triatt_stock_strided(module, z, ending, mask, triangle_attention, inplace_safe, chunk_size):
    """== stock:  zt = z.transpose(-2,-3).contiguous(); zt += tri_att_end(zt, mask=mask^T); z = zt.transpose(-2,-3).contiguous()
    computed as   z += tri_att_end(z^T-view)^T-view   (TriangleAttention.forward itself transposes its input view for the ending node and applies LN,
    which materialises a contiguous copy row-wise exactly like LN on the contiguous transposed tensor; the GEMMs/attention then see identical operands).
    For the starting node this is literally the stock statement.  Residual add: stock adds in the transposed frame (zt[j,i,:] += u[j,i,:]); we add the same
    fp values in the x-frame (z[i,j,:] += uT[i,j,:]) — elementwise identical adds, so bitwise identical."""
    if not ending:
        return module(z, mask=mask, triangle_attention=triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size)
    zt = z.transpose(-2, -3)                                   # view, no copy
    mt = mask.transpose(-1, -2) if mask is not None else None  # stock passes pair_mask.transpose(-1,-2) to tri_att_end
    u = module(zt, mask=mt, triangle_attention=triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size)   # [.., J, I, C] update in zt-frame
    return u.transpose(-2, -3)                                 # view back to x-frame


def _block_strided(self, s, z, pair_mask, triangle_multiplicative, triangle_attention, inplace_safe, chunk_size, counter):
    """PairformerBlock inplace branch with the two transpose().contiguous() passes removed; every other statement is the stock call.  dtype-agnostic."""
    z = self.tri_mul_out(z, mask=pair_mask, inplace_safe=inplace_safe, _add_with_inplace=True, triangle_multiplicative=triangle_multiplicative)
    z = self.tri_mul_in(z, mask=pair_mask, inplace_safe=inplace_safe, _add_with_inplace=True, triangle_multiplicative=triangle_multiplicative)
    z += self.tri_att_start(z, mask=pair_mask, triangle_attention=triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size)
    z += _triatt_stock_strided(self.tri_att_end, z, True, pair_mask, triangle_attention, inplace_safe, chunk_size)
    z += self.pair_transition(z)
    STATS[counter] += 1
    if self.c_s > 0:
        s = s + self.attention_pair_bias(a=s, s=None, z=z)
        s = s + self.single_transition(s)
    return s, z


# ============================================================================================ lever 1: MSA-module pair stack (fp32 master z) onto BLK2 kernels
def _install_msablk():
    import protenix.model.modules.pairformer as PF
    LEV = sys.modules.get("ptx_trunk2_levers")
    blk2_forward = PF.PairformerBlock.forward            # whatever ptx_trunk2_levers installed (BLK2 blk_forward) — for bf16 z it is the tested path
    stock_forward = getattr(LEV, "_BLK_ORIG", None) if LEV else None
    try:
        from fpf_triatt_epi.epilogue import triatt_epilogue as _epi_k
        from fpf_triatt_pro.prologue import triatt_prologue as _pro, get_cache as _gc
        from fpf_transition.transition import fn as _tr_fn
    except Exception as e:
        STATS["applied"].append(f"MSABLK:unavailable({e!r})"); return
    if LEV is None or not hasattr(LEV, "_triatt_block_pro_epi"):
        STATS["applied"].append("MSABLK:unavailable(ptx_trunk2_levers BLK2 not applied)"); return

    def msa_pair_stack_forward(blk, s, z, pair_mask=None, triangle_multiplicative="torch", triangle_attention="torch", inplace_safe=False, chunk_size=None):
        # only the MSA-module pair_stack instances are routed here (installed per-instance below); everything must match the BLK2 cell except dtype
        _census("msablk_entry", dtype=z.dtype, shape=tuple(z.shape), inplace=inplace_safe, mask=pair_mask is not None, chunk=chunk_size, ta=triangle_attention,
                autocast=torch.is_autocast_enabled(), contiguous=z.is_contiguous())
        if z.dtype == torch.bfloat16:
            STATS["msablk_bf16_direct"] += 1
            return blk2_forward(blk, s, z, pair_mask=pair_mask, triangle_multiplicative=triangle_multiplicative, triangle_attention=triangle_attention,
                                inplace_safe=inplace_safe, chunk_size=chunk_size)
        why = None
        if not inplace_safe: why = "not_inplace"
        elif triangle_attention != "cuequivariance": why = "ta_" + str(triangle_attention)
        elif pair_mask is not None: why = "mask"
        elif chunk_size is not None: why = "chunk"
        elif z.dtype != torch.float32: why = "dtype_" + str(z.dtype)
        elif z.dim() != 3 or z.shape[-1] != 256 or z.shape[-2] <= 16: why = "shape"
        elif not torch.is_autocast_enabled() or torch.get_autocast_gpu_dtype() != torch.bfloat16: why = "no_bf16_autocast"
        elif blk.training or blk.c_s != 0: why = "train_or_cs"
        elif not z.is_contiguous(): why = "noncontig"
        if why is not None:
            STATS["msablk_fallback"] += 1; _why(STATS["msablk_fallback_reasons"], why)
            return blk2_forward(blk, s, z, pair_mask=pair_mask, triangle_multiplicative=triangle_multiplicative, triangle_attention=triangle_attention,
                                inplace_safe=inplace_safe, chunk_size=chunk_size)
        z = _msa_block_fp32(blk, z, triangle_multiplicative, triangle_attention, _epi_k, _pro, _gc, _tr_fn, LEV)
        STATS["msablk_calls"] += 1
        return s, z

    # install per instance (MSAModule.blocks[i].pair_stack) at model-build time: wrap MSABlock.__init__? The model may already be built when we run (no: sitecustomize
    # runs at pairformer import, before model construction).  Route by a class-level forward that checks an instance flag set lazily from MSABlock.forward.
    _orig_msablock_forward = PF.MSABlock.forward
    def msablock_forward(self, m, z, pair_mask, triangle_multiplicative="torch", triangle_attention="torch", inplace_safe=False, chunk_size=None):
        ps = self.pair_stack
        if not getattr(ps, "_fpf_v02_msa", False):
            import types
            ps.forward = types.MethodType(msa_pair_stack_forward, ps); ps._fpf_v02_msa = True
        return _orig_msablock_forward(self, m, z, pair_mask, triangle_multiplicative=triangle_multiplicative, triangle_attention=triangle_attention,
                                      inplace_safe=inplace_safe, chunk_size=chunk_size)
    PF.MSABlock.forward = msablock_forward
    STATS["applied"].append("MSABLK:1(fp32-master z; BLK2 producers on bf16(z); stock residual adds)")


def _msa_block_fp32(blk, z, triangle_multiplicative, triangle_attention, _epi_k, _pro, _gc, _tr_fn, LEV):
    """Stock inplace branch on an fp32 z under bf16 autocast, statement by statement:
         z = tri_mul_out(z, _add_with_inplace=True)  -> cuEq wrapper: computes update from bf16(z) [autocast casts inside], then z += update (fp32 += bf16)
         z = tri_mul_in(...)                          -> same
         z += tri_att_start(z)                        -> LN(z fp32 -> fast_layernorm returns? see NOTE) ... update bf16; fp32 += bf16
         z = z.T.contiguous(); z += tri_att_end(z); z = z.T.contiguous()
         z += pair_transition(z)
       NOTE on exactness of the producer inputs: Protenix LayerNorm (fast_layernorm) on an fp32 input under autocast computes LN in fp32 and returns fp32; the
       following Linear (autocast) casts LN-out to bf16.  On a bf16 input it computes from bf16 and returns bf16.  These are NOT the same numbers
       (LN(fp32 z) rounded once vs LN(bf16(z))).  Therefore the producers must consume the fp32 z exactly like stock: we call the STOCK sub-modules for the
       LN+producer part whenever the input is fp32 (tri_att: stock module.layer_norm(x) on the fp32 view -> fp32 x_ln -> .to(bf16) == what autocast feeds the
       stock q/k/v/g/bias Linears) and only then hand bf16 x_ln to the tested prologue (ln_mode='stock' takes x_ln as given).  TriMul: stock cuEq module call
       (unchanged statement).  Transition: Fusion fn() does its own LN dispatch: for fp32 input it falls back to the stock body (dtype guard) — so we call the
       stock pair_transition and add.  => every producer sees bit-identical operands to stock; every residual add is the stock fp32 += bf16 op."""
    import protenix.model.triangular.layers as TL
    # 1-2: TriMul (stock statements, identical)
    z = blk.tri_mul_out(z, mask=None, inplace_safe=True, _add_with_inplace=True, triangle_multiplicative=triangle_multiplicative)
    z = blk.tri_mul_in(z, mask=None, inplace_safe=True, _add_with_inplace=True, triangle_multiplicative=triangle_multiplicative)
    # 3-4: triangle attention start / end without transposes: stock LN on the (strided) fp32 view -> cast bf16 (== autocast cast at linear_q input) -> tested
    #      prologue (GEMMs from given x_ln) -> cuEq attention (mask None == NOMASK cell, tested) -> tested epilogue in OP mode (out = sigmoid(g)*o @ Wo^T, bf16)
    #      -> stock residual: z(fp32) += out (x-frame; for the ending node out is produced in zt-frame and added through a transposed view: same elementwise adds)
    for module, ending in ((blk.tri_att_start, False), (blk.tri_att_end, True)):
        if getattr(module, "_deadskip", False):
            continue                                                        # stock adds exactly 0
        x = z.transpose(-2, -3) if ending else z                            # fp32 view
        x_ln = module.layer_norm(x)                                         # stock LN call on fp32 (returns fp32, contiguous in x-frame of this node)
        x_ln16 = x_ln.to(torch.bfloat16)                                    # == autocast's cast of the Linear input (single rounding of the same fp32 values)
        q, k, v, g, bias = _pro(module, x_ln16, ending=False, ln_mode="stock", x_ln=x_ln16)
        scale = 1.0 / math.sqrt(module.mha.c_hidden)
        import ptx_trunk2_levers as _LEVg                                            # sm100f exactness guard (shared predicate; module already imported by sitecustomize)
        o = TL.cuequivariance_triangular_attn(q, k, v, bias.unsqueeze(0), (_LEVg._stock_true_mask(q, q.shape[-4], k.shape[-2]) if _LEVg._cueq_sm100f_exposed(k.shape[-2]) else None), scale)
        o = o[0] if isinstance(o, (tuple, list)) else o
        if o.dim() == 5: o = o[0]
        cch = getattr(module, "_fpf_cache", None)
        if cch is None: cch = {}; module._fpf_cache = cch
        ns = cch.setdefault("trunk2", {})
        wo16 = ns.get("wo16")
        if wo16 is None or wo16.device != z.device:
            wo16 = module.mha.linear_o.weight.detach().to(torch.bfloat16).contiguous(); ns["wo16"] = wo16
        upd = _epi_k(o, g, wo16, None, ending=False, residual=False)        # [1, I', J', C] bf16 in this node's own frame (op mode == stock mha output incl. linear_o)
        upd = upd[0] if upd.dim() == 4 else upd
        if ending:
            z += upd.transpose(-2, -3)                                      # stock: zt += upd  (same elementwise fp32+bf16 adds, x-frame indexing)
        else:
            z += upd
    # 5: transition: stock module call on fp32 z (Fusion/T1 dtype guards route fp32 input to the stock two-GEMM body) + stock residual
    if not getattr(blk.pair_transition, "_deadskip", False):
        z += blk.pair_transition(z)
    return z


# ============================================================================================ lever 2a/2b: template (c=64) and confidence-head (fp32) stacks: strided stock block
def _install_strided(kind):
    """kind = 'TMPL' (TemplateEmbedder.pairformer_stack blocks, c=64) or 'CONF' (ConfidenceHead.pairformer_stack blocks, fp32).  Installs per-instance forwards
    lazily (first call of the owning module) so the trunk's 48 blocks and the MSA pair stack keep their BLK2 forward."""
    import types
    import protenix.model.modules.pairformer as PF
    counter = "tmpl_blk_calls" if kind == "TMPL" else "conf_blk_calls"
    fb, fbr = ("tmpl_fallback", "tmpl_fallback_reasons") if kind == "TMPL" else ("conf_fallback", "conf_fallback_reasons")
    LEV = sys.modules.get("ptx_trunk2_levers")

    def strided_forward(blk, s, z, pair_mask=None, triangle_multiplicative="torch", triangle_attention="torch", inplace_safe=False, chunk_size=None):
        cls_forward = PF.PairformerBlock.forward                    # class-level (BLK2 guard -> stock for these cells); used as reference / fallback
        _census(kind.lower() + "_entry", dtype=z.dtype, shape=tuple(z.shape), inplace=inplace_safe, mask=pair_mask is not None, chunk=chunk_size,
                ta=triangle_attention, autocast=torch.is_autocast_enabled(), contiguous=z.is_contiguous(), c_s=blk.c_s)
        why = None
        if not inplace_safe: why = "not_inplace"
        elif chunk_size is not None: why = "chunk"
        elif blk.training or torch.is_grad_enabled(): why = "train/grad"
        elif z.dim() < 3: why = "shape"
        if why is not None:
            STATS[fb] += 1; _why(STATS[fbr], why)
            return cls_forward(blk, s, z, pair_mask=pair_mask, triangle_multiplicative=triangle_multiplicative, triangle_attention=triangle_attention,
                               inplace_safe=inplace_safe, chunk_size=chunk_size)
        return _block_strided(blk, s, z, pair_mask, triangle_multiplicative, triangle_attention, inplace_safe, chunk_size, counter)

    def _bind(stack):
        for b in stack.blocks:
            if not getattr(b, "_fpf_v02_strided", False):
                b.forward = types.MethodType(strided_forward, b); b._fpf_v02_strided = kind

    if kind == "TMPL":
        _orig = PF.TemplateEmbedder.single_template_forward
        def single_template_forward(self, *a, **kw):
            _bind(self.pairformer_stack)
            return _orig(self, *a, **kw)
        PF.TemplateEmbedder.single_template_forward = single_template_forward
    else:
        import protenix.model.modules.confidence as CF
        _orig = CF.ConfidenceHead.memory_efficient_forward
        def memory_efficient_forward(self, *a, **kw):
            _bind(self.pairformer_stack)
            return _orig(self, *a, **kw)
        CF.ConfidenceHead.memory_efficient_forward = memory_efficient_forward
    STATS["applied"].append(f"{kind}:strided(stock kernels, no transposes)")


# ============================================================================================ apply
def apply_from_env():
    if STATS["applied"]:
        return report()                       # idempotent
    if os.environ.get("PTX_V02_MSABLK", "") == "1":
        _install_msablk()
    if os.environ.get("PTX_V02_TMPL", "") == "1":
        _install_strided("TMPL")
    if os.environ.get("PTX_V02_CONF", "") == "1":
        _install_strided("CONF")
    if os.environ.get("PTX_V02_GRAPH", "") == "1" or os.environ.get("PTX_BLK_GRAPH", "") == "1":
        # lever 3 (converged with SmallN-megablock's layer G): CUDA-graph replay of the 48-block stack across recycles/seeds/designs (fpf_stackgraph); needs the
        # stream-correct prebuilt fast-LN: INFOPT_FASTLN_PREBUILT (dir with manifest.json + .so) — default = $FPF_HOME/third_party/fastln_prebuilt if shipped
        try:
            if not os.environ.get("INFOPT_FASTLN_PREBUILT"):
                cand = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "fastln_prebuilt")
                if os.path.isdir(cand): os.environ["INFOPT_FASTLN_PREBUILT"] = cand
            import fpf_stackgraph
            STATS["applied"].append("GRAPH:" + str(fpf_stackgraph.install()))
        except Exception as e:
            STATS["applied"].append(f"GRAPH:unavailable({e!r})")
    if _CENSUS:
        STATS["applied"].append("CENSUS")
    return report()
