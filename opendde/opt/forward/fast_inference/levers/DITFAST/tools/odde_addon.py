"""odde_addon.py -- the DITFAST add-on: EXACT diffusion-sampler levers for OpenDDE 1.x (dit_hoist, dit_align).
All levers are default-OFF monkeypatches (nothing under site-packages is edited); install(model, levers) / remove(model).

Levers (diffusion-sampler levers; not trunk-block kernels):

  dit_hoist   DiT conditioning hoist: every STEP-INVARIANT tensor of the diffusion
              denoiser is computed by the STOCK op on the STOCK inputs at the first denoiser call of a sample_diffusion() call ('record'),
              copied into a fixed-address static buffer, and served from that buffer on denoiser calls 2..200 ('hit').  Hoisted:
                tok.z      DiffusionModule.normalize(z_pair.float()) -> permute(2,0,1).contiguous()      (efficient-fusion path)   1/step
                tok.bias.i token AttentionPairBias i (24): conv2d(z, W_z*ln_w) (+ structural_pair_attn_bias)  [1,16,N,N] fp32   24/step
                cond.s     DiffusionConditioning: linear_no_bias_s(layernorm_s(cat[s_trunk, s_inputs]))   [N,384]              1/step
                enc.cl_s   AtomAttentionEncoder: linear_no_bias_s(layernorm_s(s_trunk)) (token->atom broadcast source)          1/step
                enc.p_lm   AtomAttentionEncoder._add_atom_single_context_and_mlp(p_lm, c_l_q, c_l_k) -> p_lm [.., nb, 32, 128, 16]  1/step
                atom.*     6 atom-transformer blocks (3 enc + 3 dec): AdaLN(s=c_l) affine pair, local pair bias from p, output gate,
                           transition AdaLN pair + gate                                                                     6x6/step
              Exactness: each hoisted op is a pure function of (weights, s_trunk, s_inputs, pair_z, c_l, p_lm, features) that are
              constant across the 200 steps of one sample_diffusion call, evaluated by the same kernel on the same shapes as stock
              -> bitwise the value stock recomputes every step.  DIT_HOIST_CROSSCHECK=step,step,... re-runs the stock producer at those
              denoiser calls and torch.equal-compares with the buffer (mismatch raises; never a silent fallback).
  dit_align   8-aligned pair-bias storage (SDPA alignment hazard): the hoisted tok.bias buffers are allocated with a row pitch rounded
              up to a multiple of 8 elements and SDPA receives the [..., :N] view, so torch's mem-efficient backend does not re-pad+copy
              the [1,16,N,N] fp32 bias on each of the 24x200 attention calls when N % 8 != 0.  Values and logical shape identical
              (bitwise); only strides differ.  Requires dit_hoist (it is where the bias is materialised once).

Env knobs: DIT_HOIST_CROSSCHECK (default ""), DIT_HOIST_MAX_ENTRIES (default 3), DIT_HOIST_PARTS (tok,cond,enc,atom), ODDE_ADDON_QUIET.
"""
from __future__ import annotations
from opt_core.oom import is_oom                      # an out-of-memory error is re-raised before any reroute below (the core's one classifier)
import os, sys, time, json, collections
from typing import Any, Dict, Optional
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)

def _log(*a):
    if os.environ.get("ODDE_ADDON_QUIET", "0") != "1": print("[odde_addon]", *a, flush=True)

STATS: Dict[str, Any] = {"records": 0, "hits": 0, "crosschecks": 0, "crosscheck_checks": 0, "crosscheck_fail": 0, "bypass": 0, "sampler_calls": 0,
                         "denoiser_calls": 0, "entries_created": 0, "evictions": 0, "released": 0, "aligned_slots": 0, "events": []}

# =====================================================================================================================================
# DiT conditioning hoist
# =====================================================================================================================================
class _Slot:
    __slots__ = ("name", "buf", "n_rec")
    def __init__(self, name): self.name = name; self.buf = None; self.n_rec = 0


class DitHoist:
    PARTS_ALL = ("tok", "cond", "enc", "atom")

    def __init__(self, align: bool = False):
        self.mode = "off"; self.cur = None; self.entries = collections.OrderedDict(); self.installed = False
        self.parts = tuple(p for p in os.environ.get("DIT_HOIST_PARTS", ",".join(self.PARTS_ALL)).split(",") if p)
        self.crosscheck_steps = tuple(int(x) for x in os.environ.get("DIT_HOIST_CROSSCHECK", "").split(",") if x.strip())
        self.max_entries = max(1, int(os.environ.get("DIT_HOIST_MAX_ENTRIES", "3")))
        self.align = bool(align)
        self.step = -1; self._orig = {}; self.model = None
        self.report = {"parts": self.parts, "crosscheck_steps": self.crosscheck_steps, "align": self.align}

    # ---------------------------------------------------------------- protocol
    def _slot(self, name):
        s = self.cur["slots"].get(name)
        if s is None: s = _Slot(name); self.cur["slots"][name] = s
        return s

    def cached(self, name, producer, active=True, align_last=False):
        if self.mode == "off" or self.cur is None or not active:
            STATS["bypass"] += 1; return producer()
        slot = self._slot(name)
        if self.mode == "hit":
            if slot.buf is None: raise RuntimeError(f"dit_hoist: 'hit' before 'record' for {name}")
            STATS["hits"] += 1; return slot.buf
        out = producer()
        if not torch.is_tensor(out): STATS["bypass"] += 1; return out
        if self.mode == "record":
            if slot.buf is None or slot.buf.shape != out.shape or slot.buf.dtype != out.dtype or slot.buf.device != out.device:
                # LAYOUT RULE (exactness): the buffer must present the consumer with EXACTLY the strides the stock producer's output has --
                # a value-identical tensor in a different memory layout can take a different kernel/pad path downstream (observed: the
                # non-fusion token pair bias reaches SDPA permuted (channels-last-like); re-laying it out standard-contiguous changed SDPA's
                # result bits under the deterministic recipe).  So: (a) standard-contiguous producer output with an unaligned last dim ->
                # 8-pitched allocation whose [..., :N] view has the SAME logical layout torch's own pad+slice would produce (dit_align);
                # (b) anything else -> torch.empty_strided with the producer's exact strides (dense non-overlapping) or a contiguous fallback
                #     only when the producer output itself is contiguous.
                if align_last and self.align and out.dim() >= 2 and out.shape[-1] % 8 != 0 and out.is_contiguous():
                    N = out.shape[-1]; pitch = (N + 7) // 8 * 8
                    base = torch.zeros(*out.shape[:-1], pitch, dtype=out.dtype, device=out.device)   # zero padding, as torch's own pad
                    slot.buf = base[..., :N]; STATS["aligned_slots"] += 1
                else:
                    slot.buf = torch.empty_like(out)          # preserve_format: same strides as `out` for dense non-overlapping tensors
                    if slot.buf.stride() != out.stride():
                        try:
                            slot.buf = torch.empty_strided(out.shape, out.stride(), dtype=out.dtype, device=out.device)
                        except Exception as _e:
                            if is_oom(_e): raise
                            pass
                    if slot.buf.stride() != out.stride():
                        STATS["events"].append({"stride_mismatch_slot": name, "out": list(out.stride()), "buf": list(slot.buf.stride())})
            slot.buf.copy_(out); slot.n_rec += 1; STATS["records"] += 1
            return slot.buf          # hand on the BUFFER (same values; aligned strides when align_last)
        if self.mode == "cross-check":
            STATS["crosschecks"] += 1; STATS["crosscheck_checks"] += 1
            if slot.buf is None or not torch.equal(slot.buf, out):
                STATS["crosscheck_fail"] += 1
                raise RuntimeError(f"dit_hoist CROSSCHECK FAIL: {name} at denoiser call {self.step} differs from the recorded buffer")
            return out
        raise AssertionError(self.mode)

    def _bind(self, key):
        ent = self.entries.pop(key, None)
        if ent is None: ent = {"slots": {}, "t": time.time()}; STATS["entries_created"] += 1
        self.entries[key] = ent
        while len(self.entries) > self.max_entries:
            k_old = next(iter(self.entries))
            if k_old == key: break
            old = self.entries.pop(k_old)
            for s in old["slots"].values(): s.buf = None
            old["slots"].clear(); STATS["evictions"] += 1
        self.cur = ent

    def _release(self, key):
        """Drop an entry and free its buffers: the sampler records and reads them within ONE sampler call, so nothing is held across items."""
        ent = self.entries.pop(key, None)
        if ent is None: return
        for s in ent["slots"].values(): s.buf = None
        ent["slots"].clear(); STATS["released"] += 1

    def static_bytes(self):
        n = 0
        for e in self.entries.values():
            for s in e["slots"].values():
                if s.buf is not None:
                    b = s.buf._base if s.buf._base is not None else s.buf
                    n += b.numel() * b.element_size()
        return n

    # ---------------------------------------------------------------- install
    def install(self, model):
        from opendde.model.modules import transformer as TR, diffusion as DF, primitives as PR
        import opendde.model.opendde as OM
        self.model = model; dm = model.diffusion_module; H = self
        assert isinstance(dm, DF.DiffusionModule)
        tok_blocks = list(dm.diffusion_transformer.blocks)
        atom_blocks = [("enc", j, b) for j, b in enumerate(dm.atom_attention_encoder.atom_transformer.diffusion_transformer.blocks)] + \
                      [("dec", j, b) for j, b in enumerate(dm.atom_attention_decoder.atom_transformer.diffusion_transformer.blocks)]
        self.report["n_tok_blocks"] = len(tok_blocks); self.report["n_atom_blocks"] = len(atom_blocks)

        # (0) sampler wrapper: one entry per shape signature; denoiser-call counter drives record/hit/cross-check.
        stock_sd = OM.sample_diffusion            # the stock sampler, by the module-global name the model resolves
        self._orig["OM.sample_diffusion"] = stock_sd
        def sample_diffusion_hoisted(*args, **kw):
            denoise_net = kw.get("denoise_net", args[0] if args else None)
            if args or denoise_net is not dm or kw.get("pair_z") is None or kw.get("p_lm") is None or kw.get("c_l") is None \
                    or kw.get("pair_z_spec") is not None or kw.get("z_trunk") is not None or torch.is_grad_enabled() \
                    or kw.get("atom_window_spec") is not None or kw.get("foldcp_attention_bias") is not None or kw.get("foldcp_group") is not None:
                if kw.get("atom_window_spec") is not None or kw.get("foldcp_attention_bias") is not None or kw.get("foldcp_group") is not None:
                    STATS["bypass_foldcp_args"] = STATS.get("bypass_foldcp_args", 0) + 1   # opendde 1.1.1's Fold-CP / atom-window sampler arguments (generator.py:96-99) set: the hoist stands aside by name
                STATS["bypass"] += 1; H.mode = "off"; return stock_sd(*args, **kw)      # cache disabled / fold-CP / grad / positional call / 1.1.1 Fold-CP args: stock
            ifd = kw["input_feature_dict"]; s_inputs = kw["s_inputs"]; N_sample = int(kw.get("N_sample", 1))
            dcs = kw.get("diffusion_chunk_size"); chunk_n = N_sample if dcs is None else min(N_sample, max(1, int(dcs)))
            key = (int(s_inputs.shape[-2]), int(ifd["atom_to_token_idx"].shape[-1]), tuple(kw["pair_z"].shape), tuple(kw["p_lm"].shape), str(s_inputs.dtype),
                   chunk_n, bool(kw.get("inplace_safe", False)), bool(kw.get("enable_efficient_fusion", False)), kw.get("attn_chunk_size"))
            H._bind(key); STATS["sampler_calls"] += 1
            try:
                H.mode = "record"; H.step = -1                # the stock eager loop: denoiser call 0 records, 1.. hit (cross-check at DIT_HOIST_CROSSCHECK)
                return stock_sd(**kw)
            finally:
                H._release(key)                              # the hoisted buffers serve this sampler call only
                H.mode = "off"; H.cur = None
        OM.sample_diffusion = sample_diffusion_hoisted

        # denoiser call counter (DiffusionModule.forward is what the samplers call per step)
        orig_dm_forward = dm.forward
        self._orig["dm.forward"] = orig_dm_forward
        def dm_forward_counted(*a, **k):
            if H.mode != "off" and H.step < 10 ** 6:
                H.step += 1; STATS["denoiser_calls"] += 1
                if H.step == 0: H.mode = "record"
                elif H.step in H.crosscheck_steps: H.mode = "cross-check"
                else: H.mode = "hit"
            elif H.mode != "off":
                STATS["denoiser_calls"] += 1
            return orig_dm_forward(*a, **k)
        dm.forward = dm_forward_counted

        # (1) tok.z : DiffusionModule.normalize output (+ the permute/contiguous that follows it in f_forward is re-done by stock code on
        #     the cached tensor: permute of a cached contiguous tensor... to hoist the .contiguous() copy too we cache at the level of the
        #     token DiffusionTransformer's z argument: wrap dm.diffusion_transformer.forward to swap z for the cached contiguous copy.
        if "tok" in self.parts:
            norm = dm.normalize; orig_norm = norm.forward; self._orig["normalize.forward"] = (norm, orig_norm)
            def norm_fwd(x):
                return H.cached("tok.znorm", lambda: orig_norm(x))
            norm.forward = norm_fwd
            dt = dm.diffusion_transformer; orig_dt = dt.forward; self._orig["dt.forward"] = (dt, orig_dt)
            def dt_fwd(a, s, z, **k):
                # stock f_forward did: z = normalize(...) [cached above]; z = permute_final_dims(z,[2,0,1]).contiguous()  <- a full copy per step.
                # cache that copy: producer returns the tensor stock just built (z); hit returns the recorded buffer (same values).
                z2 = H.cached("tok.zperm", lambda: z)
                return orig_dt(a, s, z2, **k)
            dt.forward = dt_fwd
            # tok.bias.i : per token block, the pair bias (conv2d in efficient-fusion mode, LN+linear otherwise) + extra bias, aligned
            for i, blk in enumerate(tok_blocks):
                apb = blk.attention_pair_bias; orig_sma = apb.standard_multihead_attention; self._orig[f"tok{i}.sma"] = (apb, orig_sma)
                def make(apb, orig_sma, i):
                    def sma(q, kv, z, extra_attn_bias=None, inplace_safe=False, enable_efficient_fusion=False):
                        if H.mode == "off" or apb._foldcp_diffusion_bias_row_chunk_size() > 0:
                            return orig_sma(q, kv, z, extra_attn_bias=extra_attn_bias, inplace_safe=inplace_safe, enable_efficient_fusion=enable_efficient_fusion)
                        def producer():
                            # verbatim stock bias construction (transformer.py AttentionPairBias.standard_multihead_attention)
                            if enable_efficient_fusion:
                                w = (apb.linear_nobias_z.weight * apb.layernorm_z.weight[None, :])[:, :, None, None]
                                bias = F.conv2d(z, w)
                            else:
                                bias = apb.linear_nobias_z(apb.layernorm_z(z)); bias = TR.permute_final_dims(bias, [2, 0, 1])
                            eb = extra_attn_bias
                            if eb is not None:
                                while len(eb.shape) < len(bias.shape) - 1: eb = eb.unsqueeze(dim=0)
                                if len(eb.shape) == len(bias.shape) - 1: eb = eb.unsqueeze(dim=-3)
                                bias = bias + eb.to(dtype=bias.dtype, device=bias.device)
                            return apb._align_bias_to_query(bias, q, n_pair_dims=2)
                        bias = H.cached(f"tok.bias.{i}", producer, align_last=True)
                        return apb.attention(q_x=q, kv_x=kv, attn_bias=bias, inplace_safe=inplace_safe)
                    return sma
                apb.standard_multihead_attention = make(apb, orig_sma, i)

        # (2) cond.s : DiffusionConditioning single path before the noise embedding
        if "cond" in self.parts:
            dc = dm.diffusion_conditioning; orig_lns = dc.linear_no_bias_s.forward; self._orig["cond.lin_s"] = (dc.linear_no_bias_s, orig_lns)
            # layernorm_s(cat) -> linear_no_bias_s : cache the linear output; the LN + cat still run in record/cross-check via the producer chain,
            # so wrap at linear_no_bias_s and make layernorm_s a pass-through in hit mode by caching ITS output too.
            orig_ln = dc.layernorm_s.forward; self._orig["cond.ln_s"] = (dc.layernorm_s, orig_ln)
            dc.layernorm_s.forward = lambda x: H.cached("cond.ln_s", lambda: orig_ln(x))
            dc.linear_no_bias_s.forward = lambda x: H.cached("cond.s", lambda: orig_lns(x))

        # (3) enc.cl_s and enc.p_lm
        if "enc" in self.parts:
            enc = dm.atom_attention_encoder
            orig_els = enc.linear_no_bias_s.forward; orig_eln = enc.layernorm_s.forward
            self._orig["enc.lin_s"] = (enc.linear_no_bias_s, orig_els); self._orig["enc.ln_s"] = (enc.layernorm_s, orig_eln)
            enc.layernorm_s.forward = lambda x: H.cached("enc.ln_s", lambda: orig_eln(x))
            enc.linear_no_bias_s.forward = lambda x: H.cached("enc.cl_s", lambda: orig_els(x))
            orig_mlp = enc._add_atom_single_context_and_mlp; self._orig["enc.mlp"] = (enc, orig_mlp)
            def mlp(p_lm, c_l_q, c_l_k, inplace_safe=False):
                # stock writes into p_lm (a clone made by the encoder when inplace_safe, else the cached p_lm from prepare_cache... in the
                # stock non-inplace path p_lm[...] = ... ALSO writes into the passed tensor).  The producer runs stock on the tensor it was
                # given; 'hit' returns the recorded result buffer and leaves the incoming p_lm untouched.
                return H.cached("enc.p_lm", lambda: orig_mlp(p_lm, c_l_q, c_l_k, inplace_safe=inplace_safe))
            enc._add_atom_single_context_and_mlp = mlp
            # rearrange_qk_to_dense_trunk(c_l, c_l) in the encoder is also step-invariant but cheap; left stock.

        # (4) atom blocks: AdaLN affine (s = c_l is step-invariant), local pair bias from p (step-invariant), gates
        if "atom" in self.parts:
            for where, j, blk in atom_blocks:
                apb = blk.attention_pair_bias; ctb = blk.conditioned_transition_block; tag = f"atom.{where}{j}"
                for nm, aln in (("ln_a", apb.layernorm_a),) + ((("ln_kv", apb.layernorm_kv),) if getattr(apb, "cross_attention_mode", False) and hasattr(apb, "layernorm_kv") else ()):
                    if not isinstance(aln, PR.AdaptiveLayerNorm): continue
                    o_ls, o_lns, o_lnS = aln.linear_s.forward, aln.linear_nobias_s.forward, aln.layernorm_s.forward
                    self._orig[f"{tag}.{nm}"] = (aln, o_ls, o_lns, o_lnS)
                    def mk(aln, o_ls, o_lns, o_lnS, key):
                        aln.layernorm_s.forward = lambda s: H.cached(key + ".lns", lambda: o_lnS(s))
                        aln.linear_s.forward = lambda s: H.cached(key + ".lin_s", lambda: o_ls(s))
                        aln.linear_nobias_s.forward = lambda s: H.cached(key + ".lin_nb_s", lambda: o_lns(s))
                    mk(aln, o_ls, o_lns, o_lnS, f"{tag}.{nm}")
                # output gate sigmoid(linear_a_last(s)): cache the linear output (sigmoid stays per step: cheap elementwise; keeps op order stock)
                if getattr(apb, "has_s", False):
                    o = apb.linear_a_last.forward; self._orig[f"{tag}.gate"] = (apb.linear_a_last, o)
                    apb.linear_a_last.forward = (lambda o, key: (lambda s: H.cached(key, lambda: o(s))))(o, f"{tag}.gate")
                # local pair bias: linear_nobias_z(layernorm_z(p)) inside local_multihead_attention
                o_lz, o_lnz = apb.linear_nobias_z.forward, apb.layernorm_z.forward; self._orig[f"{tag}.z"] = (apb, o_lz, o_lnz)
                apb.layernorm_z.forward = (lambda o, key: (lambda p: H.cached(key, lambda: o(p))))(o_lnz, f"{tag}.lnz")
                apb.linear_nobias_z.forward = (lambda o, key: (lambda p: H.cached(key, lambda: o(p))))(o_lz, f"{tag}.linz")
                # transition: adaln affine + gate
                aln = ctb.adaln
                o_ls, o_lns, o_lnS = aln.linear_s.forward, aln.linear_nobias_s.forward, aln.layernorm_s.forward
                self._orig[f"{tag}.ctb.adaln"] = (aln, o_ls, o_lns, o_lnS)
                aln.layernorm_s.forward = (lambda o, key: (lambda s: H.cached(key, lambda: o(s))))(o_lnS, f"{tag}.ctb.lns")
                aln.linear_s.forward = (lambda o, key: (lambda s: H.cached(key, lambda: o(s))))(o_ls, f"{tag}.ctb.lin_s")
                aln.linear_nobias_s.forward = (lambda o, key: (lambda s: H.cached(key, lambda: o(s))))(o_lns, f"{tag}.ctb.lin_nb_s")
                o = ctb.linear_s.forward; self._orig[f"{tag}.ctb.gate"] = (ctb.linear_s, o)
                ctb.linear_s.forward = (lambda o, key: (lambda s: H.cached(key, lambda: o(s))))(o, f"{tag}.ctb.gate")
        self.installed = True
        _log("dit_hoist installed:", json.dumps(self.report))
        return self.report

    def remove(self):
        if not self.installed: return
        import opendde.model.opendde as OM
        dm = self.model.diffusion_module
        OM.sample_diffusion = self._orig.pop("OM.sample_diffusion")
        dm.forward = self._orig.pop("dm.forward")
        # instance-level forward overrides: delete the instance attribute to fall back to the class method, or restore saved bound method
        for k, v in list(self._orig.items()):
            if k == "normalize.forward": v[0].forward = v[1]
            elif k == "dt.forward": v[0].forward = v[1]
            elif k.endswith(".sma"): v[0].standard_multihead_attention = v[1]
            elif k in ("cond.lin_s", "cond.ln_s", "enc.lin_s", "enc.ln_s"): v[0].forward = v[1]
            elif k == "enc.mlp": v[0]._add_atom_single_context_and_mlp = v[1]
            elif k.endswith(".ln_a") or k.endswith(".ln_kv") or k.endswith(".ctb.adaln"):
                aln, o_ls, o_lns, o_lnS = v; aln.linear_s.forward = o_ls; aln.linear_nobias_s.forward = o_lns; aln.layernorm_s.forward = o_lnS
            elif k.endswith(".gate") or k.endswith(".ctb.gate"): v[0].forward = v[1]
            elif k.endswith(".z"):
                apb, o_lz, o_lnz = v; apb.linear_nobias_z.forward = o_lz; apb.layernorm_z.forward = o_lnz
            self._orig.pop(k, None)
        # drop instance attrs that equal the class function (clean state): bound methods were saved, so re-assigning them is exact
        for e in self.entries.values():
            for s in e["slots"].values(): s.buf = None
        self.entries.clear(); self.cur = None; self.mode = "off"; self.installed = False
        _log("dit_hoist removed")


# =====================================================================================================================================
# registry
# =====================================================================================================================================
_ACTIVE: Dict[str, Any] = {}


def install(model, levers, ckpt_path=None):
    """levers: list of names in {dit_hoist, dit_align}. Returns an info dict per lever."""
    info = {}
    levers = list(levers)
    if "dit_align" in levers and "dit_hoist" not in levers:
        levers.insert(0, "dit_hoist")     # align is a property of the hoisted storage
    if "dit_hoist" in levers:
        H = DitHoist(align=("dit_align" in levers)); info["dit_hoist"] = H.install(model); _ACTIVE["dit_hoist"] = H
        info["dit_align"] = H.align
    return info


def remove(model):
    if "dit_hoist" in _ACTIVE:
        _ACTIVE.pop("dit_hoist").remove()


def stats():
    out = dict(STATS); out["events"] = out["events"][-20:]
    H = _ACTIVE.get("dit_hoist")
    if H is not None:
        out["static_bytes"] = H.static_bytes(); out["n_entries"] = len(H.entries)
        out["slots_per_entry"] = [len(e["slots"]) for e in H.entries.values()]
    return out
