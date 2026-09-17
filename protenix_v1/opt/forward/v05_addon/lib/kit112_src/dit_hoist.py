"""dit_hoist.py — lever `hoist`: EVERY step-invariant tensor of the Protenix diffusion denoiser hoisted out of the 200-step sampler loop
(computed once per sample_diffusion chunk with the stock ops, served from static buffers to the remaining steps):

  tok.znorm      DiffusionModule.normalize(z_pair.float()) + permute(2,0,1).contiguous()           1/step   [C=256,N,N] fp32
  tok.bias[i]    token AttentionPairBias i: conv2d(z, W_z*ln_w) (efficient fusion) -> [1,16,N,N]    24/step
  cond.single    DiffusionConditioning: linear_no_bias_s(layernorm_s(cat[s_trunk, s_inputs]))     1/step   [N,384]
  enc.cl_s       AtomAttentionEncoder: linear_no_bias_s(layernorm_s(s_trunk)) (token->atom add)   1/step   [1,N,128]
  enc.p_lm       AtomAttentionEncoder: p_lm + lin_cl(relu(c_l_q)) + lin_cm(relu(c_l_k)) + small_mlp  1/step [1,nb,32,128,16]
  atom{enc,dec}.block[j] (3+3 blocks, conditioning s == c_l and pair p == p_lm are step-invariant):
      apb.layernorm_a  : sigmoid(linear_s(layernorm_s(s))) , linear_nobias_s(layernorm_s(s))     [1,N_atom,128] x2
      apb.layernorm_kv : same two tensors of the key/value AdaLN (cross_attention_mode)            [1,N_atom,128] x2
      apb.locbias      : permute(linear_nobias_z(layernorm_z(p)), [3,0,1,2])                     [1,4,nb,32,128]
      apb.gate         : sigmoid(linear_a_last(s))                                                [1,N_atom,128]
      ctb.adaln        : sigmoid(adaln.linear_s(adaln.layernorm_s(s))), adaln.linear_nobias_s(..) [1,N_atom,128] x2
      ctb.gate         : sigmoid(linear_s(s))                                                     [1,N_atom,128]
  = 27 + 6*8 = 75 cached tensors per prediction (fp32/bf16 as stock produces them; the tok.* buffers grow with N_token^2).

NOT hoisted (depend on t_hat via the Fourier noise embedding -> s_single): cond noise path + transitions, token-block AdaLN/gates/
transitions, dm.layernorm_s/linear_no_bias_s, everything downstream of x_noisy.

Mechanism: mode 'record' = the STOCK ops run on the stock inputs in the stock shapes at the
first step of every sample_diffusion chunk and their outputs are copied into static buffers (fixed addresses -> CUDA-graph safe);
mode 'hit' = steps 2..200 return the buffers without launching the producer kernels; mode 'audit' = stock ops run from scratch and
every output is torch.equal-compared with its buffer (mismatch raises; no silent fallback).  Nothing is re-implemented: in record
and check the tensor handed on is the one the stock code computed.  Exactness argument: each hoisted op is a pure function of
(weights, s_trunk, s_inputs, z/pair_z, c_l, p_lm, features) which are constant across the steps of one sample_diffusion call, and
is evaluated by the same kernel on the same shapes as stock -> bit for bit the value stock computes at every step (checkable in-run at audit steps).

Eager use (stock sampler):  H = install(model); the module wraps Protenix.sample_diffusion so that every call does
    record on denoiser call 1, hit on calls 2..N (per chunk signature), audit at the configured audit steps (none by default)
Graphed use (infopt_graphs.protenix.graphed.GraphedDenoiseLoop): H.bind(ent) per graph entry; set_mode('record') for the eager step-0
    pass, 'hit' inside capture/replay, 'audit' for diagnostic eager steps — the loop drives H through its `biascache` attribute
    (bind / set_mode / poison / unpoison / summary), so `sampler.biascache = H` composes.
All four parts (tok, cond, enc, atom) are hoisted; aligned bias rows and the glue caches are on (Hoist.__init__).
"""
from __future__ import annotations
from opt_core.oom import is_oom          # rerouting handlers re-raise an out-of-memory first
import inspect, os, time, hashlib
from typing import Any, Dict, Optional
import torch
import torch.nn.functional as F

STATS = {"records": 0, "hits": 0, "audits": 0, "audit_checks": 0, "audit_fail": 0, "bypass": 0, "sampler_calls": 0, "denoiser_calls": 0, "poison": [], "events": []}


class Slot:
    """One hoisted tensor: static buffer + producer closure protocol."""
    __slots__ = ("name", "buf", "n_rec")
    def __init__(self, name):
        self.name = name; self.buf = None; self.n_rec = 0


class DitHoist:
    PARTS_ALL = ("tok", "cond", "enc", "atom")

    def __init__(self):
        self.mode = "off"                     # off | record | hit | audit
        self.cur: Optional[Dict[str, Any]] = None   # bound entry: {"slots": {name: Slot}}
        self.entries: Dict[Any, Dict[str, Any]] = {}
        self.installed = False
        self.parts = tuple(self.PARTS_ALL)
        self.audit_steps = ()
        self.report: Dict[str, Any] = {"parts": self.parts, "audit_steps_env": self.audit_steps}
        self.in_sampler = False
        self.step = -1
        self._x_shape = None
        self.n_token_blocks = 0
        self.max_entries = 2
        self.release_on_exit = False
        self.align = True                     # aligned row storage for tok.bias (exact; see cached())
        self.glue = True                      # G1 pair_z clone + G2 local padding-bias cache (exact; see install())
        self.report["align"] = self.align; self.report["glue"] = self.glue
        self._padbias = {}; self._glue_on = True; self._g1_ok = True; self._g1_version = {}                                            # G2 cache: (N, n_q, n_k, inf, dtype, device) -> trunked bias view

    # ------------------------------------------------------------------ entry binding (graph entry or eager signature)
    # MEMORY: the static buffers scale with N_token^2 (tok.*) and N_atom (enc / atom parts), x chunk N_sample.
    # Eager mode keeps at most max_entries (2 = the two chunk signatures of one sampler call) shape-keyed
    # entries in an LRU and frees the evicted entry's buffers immediately (release_on_exit would additionally drop every buffer
    # when the sampler call returns; off).  Graphed mode: the buffers live inside the
    # graph entry dict (ent["dit_hoist"]) and are freed when that entry is evicted — they add to, and must be budgeted
    # with, the graph entry's pool.
    def bind(self, ent: Dict[str, Any]):
        self.cur = ent.setdefault("dit_hoist", {"slots": {}, "allocated_at": time.time()})

    def _bind_eager(self, key):
        ent = self.entries.pop(key, None)
        if ent is None:
            ent = {}
        self.entries[key] = ent                      # move to MRU position
        while len(self.entries) > self.max_entries:  # evict LRU (never the one just inserted)
            k_old = next(iter(self.entries))
            if k_old == key: break
            self.release(self.entries.pop(k_old)); STATS["evictions"] = STATS.get("evictions", 0) + 1
        self.bind(ent)

    def release(self, ent: Optional[Dict[str, Any]] = None):
        """Free the static buffers of one entry (default: all entries)."""
        ents = [ent] if ent is not None else list(self.entries.values())
        for e in ents:
            dh = e.get("dit_hoist") if isinstance(e, dict) else None
            if not dh: continue
            for slot in dh["slots"].values():
                slot.buf = None
            dh["slots"].clear()
        if ent is None:
            self.entries.clear(); self.cur = None

    def static_bytes(self) -> int:
        n = 0
        for e in self.entries.values():
            dh = e.get("dit_hoist") or {}
            n += sum(sl.buf.numel() * sl.buf.element_size() for sl in dh.get("slots", {}).values() if sl.buf is not None)
        return n

    def set_mode(self, mode: str):
        assert mode in ("off", "record", "hit", "audit"), mode
        self.mode = mode

    def _slot(self, name) -> Slot:
        s = self.cur["slots"].get(name)
        if s is None:
            s = Slot(name); self.cur["slots"][name] = s
        return s

    def cached(self, name: str, producer, active: bool = True):
        """The core protocol. producer() runs the STOCK computation and returns its tensor."""
        if self.mode == "off" or self.cur is None or not active:
            STATS["bypass"] += 1
            return producer()
        slot = self._slot(name)
        if self.mode == "hit":
            if slot.buf is None:
                raise RuntimeError(f"dit_hoist: 'hit' before 'record' for {name}")
            STATS["hits"] += 1
            return slot.buf
        out = producer()
        if not torch.is_tensor(out) or not out.is_cuda and torch.cuda.is_available():
            # CPU tensors (unit tests) are still cached; non-tensors bypass
            if not torch.is_tensor(out):
                return out
        if self.mode == "record":
            STATS["records"] += 1
            if name == "glue.pair_z_clone": self._g1_version.clear()
            if slot.buf is None or slot.buf.shape != out.shape or slot.buf.dtype != out.dtype or slot.buf.device != out.device:
                if self.align and name.startswith("tok.bias") and out.dim() >= 2 and out.shape[-1] % 8 != 0:
                    # ALIGNED STORAGE for the token pair biases (exact). PyTorch's memory-efficient SDPA (the kernel Protenix's fp32 token
                    # attention dispatches to) requires every leading stride of the [.., H, N, N] mask to be a multiple of 8 ELEMENTS
                    # (aten/native/transformers/attention.cpp: aligned_tensor<8> / pad_bias<8>); when N_token % 8 != 0 it pads the fp32 bias on
                    # EVERY call (at::pad = fill + full copy, then slice) — 24x per diffusion step. Storing the
                    # hoisted bias with a row pitch rounded up to 8 and returning the [..., :N] view hands the kernel exactly the strides and
                    # values PyTorch's own pad+slice produces, so outputs are bit-equal and the per-call pad disappears.
                    padded = list(out.shape); padded[-1] = (padded[-1] + 7) // 8 * 8
                    slot.buf = torch.empty(padded, dtype=out.dtype, device=out.device)[..., : out.shape[-1]]
                    STATS["aligned_slots"] = STATS.get("aligned_slots", 0) + 1
                else:
                    slot.buf = torch.empty_like(out, memory_format=torch.contiguous_format) if out.is_contiguous() else torch.empty_strided(out.shape, out.stride(), dtype=out.dtype, device=out.device)
            slot.buf.copy_(out); self._g1_resync(name, slot)
            slot.n_rec += 1
            return slot.buf
        # audit
        STATS["audits"] += 1; STATS["audit_checks"] += 1
        ok = slot.buf is not None and slot.buf.shape == out.shape and bool(torch.equal(out, slot.buf))
        self.report.setdefault("audit", []).append((name, self.step, ok))
        if not ok:
            STATS["audit_fail"] += 1
            md = "n/a" if slot.buf is None or slot.buf.shape != out.shape else float((out.float() - slot.buf.float()).abs().max())
            raise RuntimeError(f"dit_hoist AUDIT FAILED: {name} at step {self.step} differs from the static buffer (max|d|={md})")
        return slot.buf

    # ------------------------------------------------------------------ poison protocol (graph reads the buffers?)
    def poison(self, scale: float = 8.0):
        backup = {}
        for name, slot in self.cur["slots"].items():
            if slot.buf is None: continue
            backup[name] = slot.buf.clone()
            gen = torch.Generator(device=slot.buf.device); gen.manual_seed(20260823)
            slot.buf.copy_((torch.randn(slot.buf.shape, device=slot.buf.device, dtype=torch.float32, generator=gen) * scale).to(slot.buf.dtype))
            self._g1_resync(name, slot)      # the hoist's OWN poison write is legitimate: re-base the version guard
        return backup

    def unpoison(self, backup):
        for name, t in backup.items():
            self.cur["slots"][name].buf.copy_(t)
            self._g1_resync(name, self.cur["slots"][name])

    def _g1_resync(self, name, slot):
        """After an in-place write performed BY THE HOIST ITSELF (record copy_, poison, unpoison) re-base the G1 read-only guard's
        reference version; foreign in-place writes (anything else) still trip it."""
        if name == "glue.pair_z_clone" and slot.buf is not None and id(slot.buf) in self._g1_version:
            try: self._g1_version[id(slot.buf)] = slot.buf._version
            except Exception: pass

    def summary(self):
        n_slots = len(self.cur["slots"]) if self.cur else 0
        mb = sum(s.buf.numel() * s.buf.element_size() for s in (self.cur["slots"].values() if self.cur else []) if s.buf is not None) / 2**20
        v = self.report.get("audit", [])
        return dict(STATS, n_slots=n_slots, static_MB=round(mb, 1), static_MB_all_entries=round(self.static_bytes() / 2**20, 1), parts=self.parts, audit_checks=len(v), audit_all_ok=all(x[2] for x in v) if v else None,
                    entries=len(self.entries), max_entries=self.max_entries)

    # ------------------------------------------------------------------ installation
    def install(self, model, sampler=None, wrap_eager_sampler: bool = True):
        from protenix.model.modules import diffusion as D
        from protenix.model.modules import transformer as T
        from protenix.model.modules import primitives as P
        from protenix.model.utils import permute_final_dims
        if self.installed:
            if sampler is not None: sampler.biascache = self
            return self
        H = self
        dms = [m for m in model.modules() if isinstance(m, D.DiffusionModule)]
        assert len(dms) == 1, f"expected one DiffusionModule, found {len(dms)}"
        dm = dms[0]
        # ---- source guards: the call sites this lever relies on (refuse on drift)
        src_ff = inspect.getsource(D.DiffusionModule.f_forward)
        assert src_ff.count("self.normalize(") == 1 and "permute_final_dims(z, [2, 0, 1]).contiguous()" in src_ff, "DiffusionModule.f_forward normalize/permute site changed: refusing"
        src_sma = inspect.getsource(T.AttentionPairBias.standard_multihead_attention)
        assert "bias = F.conv2d(z, weight)" in src_sma and "self.attention(q_x=q, kv_x=kv, attn_bias=bias, inplace_safe=inplace_safe)" in src_sma, "standard_multihead_attention changed: refusing"
        src_lma = inspect.getsource(T.AttentionPairBias.local_multihead_attention)
        assert "self.linear_nobias_z(\n            self.layernorm_z(z)\n        )" in src_lma or "self.linear_nobias_z(self.layernorm_z(z))" in src_lma.replace("\n", " ").replace("  ", " "), "local_multihead_attention bias site changed: refusing"
        src_apbf = inspect.getsource(T.AttentionPairBias.forward)
        assert "torch.sigmoid(self.linear_a_last(s))" in src_apbf, "AttentionPairBias.forward gate site changed: refusing"
        src_ada = inspect.getsource(P.AdaptiveLayerNorm.forward)
        assert "a = torch.sigmoid(self.linear_s(s)) * a + self.linear_nobias_s(s)" in src_ada, "AdaptiveLayerNorm.forward changed: refusing"
        src_ctb = inspect.getsource(T.ConditionedTransitionBlock.forward)
        assert "a = torch.sigmoid(self.linear_s(s)) * self.linear_nobias_b(b)" in src_ctb, "ConditionedTransitionBlock.forward changed: refusing"
        src_cond = inspect.getsource(D.DiffusionConditioning.forward)
        assert "self.linear_no_bias_s(\n                self.layernorm_s(single_s)" in src_cond or "self.linear_no_bias_s(self.layernorm_s(single_s))" in " ".join(src_cond.split()), "DiffusionConditioning.forward single site changed: refusing"
        src_enc = inspect.getsource(T.AtomAttentionEncoder.forward)
        assert "x_token=self.linear_no_bias_s(self.layernorm_s(s))" in src_enc and "p_lm = p_lm + self.small_mlp(p_lm)" in src_enc, "AtomAttentionEncoder.forward sites changed: refusing"
        self.report["source_sha"] = {k: hashlib.sha256(v.encode()).hexdigest()[:12] for k, v in
                                     dict(f_forward=src_ff, sma=src_sma, lma=src_lma, apb_forward=src_apbf, adaln=src_ada, ctb=src_ctb, cond=src_cond, enc=src_enc).items()}

        parts = set(self.parts)
        # ================= tok: normalize(z) + 24 conv2d biases
        if "tok" in parts:
            ln = dm.normalize
            o_norm = ln.forward
            def norm_fwd(z, _o=o_norm):
                # stock: z = self.normalize(z_pair.float()); z = permute_final_dims(z,[2,0,1]).contiguous()
                # cache the permuted-contiguous tensor and return its logical [...,N,N,C] view so the call-site permute+contiguous is a free view.
                def producer():
                    out = _o(z)
                    d = out.dim()
                    return out.permute(*range(d - 3), d - 1, d - 3, d - 2).contiguous()
                permc = H.cached("tok.znorm", producer)
                d = permc.dim()
                return permc.permute(*range(d - 3), d - 2, d - 1, d - 3)
            ln.forward = norm_fwd
            for i, b in enumerate(dm.diffusion_transformer.blocks):
                apb = b.attention_pair_bias
                def sma(q, kv, z, inplace_safe=False, enable_efficient_fusion=False, _apb=apb, _i=i):
                    def producer():
                        if enable_efficient_fusion:
                            weight = (_apb.linear_nobias_z.weight * _apb.layernorm_z.weight[None, :])[:, :, None, None]
                            return F.conv2d(z, weight)
                        bias = _apb.linear_nobias_z(_apb.layernorm_z(z))
                        return permute_final_dims(bias, [2, 0, 1])
                    bias = H.cached(f"tok.bias{_i}", producer)
                    return _apb.attention(q_x=q, kv_x=kv, attn_bias=bias, inplace_safe=inplace_safe)
                apb.standard_multihead_attention = sma
        # ================= cond: single base
        if "cond" in parts:
            cond = dm.diffusion_conditioning
            o_lns = cond.layernorm_s.forward; o_lin = cond.linear_no_bias_s.forward
            # cache the composite linear_no_bias_s(layernorm_s(x)): wrap linear_no_bias_s; layernorm_s returns input marker in hit mode
            class _Skip:  # sentinel carrying nothing (hit mode: the LN is not evaluated)
                pass
            def lns_fwd(x, _o=o_lns):
                if H.mode == "hit" and H.cur is not None and "cond.single" in H.cur["slots"]:
                    return _Skip
                return _o(x)
            def lin_fwd(x, _o=o_lin):
                return H.cached("cond.single", (lambda: _o(x)) if x is not _Skip else (lambda: (_ for _ in ()).throw(RuntimeError("cond.single producer called in hit"))))
            cond.layernorm_s.forward = lns_fwd; cond.linear_no_bias_s.forward = lin_fwd
        # ================= enc: token->atom single add, p_lm update chain
        enc = dm.atom_attention_encoder
        dec = dm.atom_attention_decoder
        if "enc" in parts:
            o_elns = enc.layernorm_s.forward; o_elin = enc.linear_no_bias_s.forward
            class _Skip2: pass
            def elns_fwd(x, _o=o_elns):
                if H.mode == "hit" and H.cur is not None and "enc.cl_s" in H.cur["slots"]:
                    return _Skip2
                return _o(x)
            def elin_fwd(x, _o=o_elin):
                return H.cached("enc.cl_s", (lambda: _o(x)) if x is not _Skip2 else (lambda: (_ for _ in ()).throw(RuntimeError("enc.cl_s producer called in hit"))))
            enc.layernorm_s.forward = elns_fwd; enc.linear_no_bias_s.forward = elin_fwd
            # p_lm chain: p_lm = p_lm + lin_cl(relu(c_l_q)) + lin_cm(relu(c_l_k)); p_lm = p_lm + small_mlp(p_lm)   (inplace_safe=False path)
            # or        p_lm = p_lm + lin_cl(...); p_lm += lin_cm(...); p_lm += small_mlp(p_lm)                      (inplace_safe=True path)
            # Both end with small_mlp(p_lm) being the LAST producer; the three addends are cached individually (each a pure function of c_l / p_lm):
            o_cl = enc.linear_no_bias_cl.forward; o_cm = enc.linear_no_bias_cm.forward; o_mlp = enc.small_mlp.forward
            enc.linear_no_bias_cl.forward = lambda x, _o=o_cl: H.cached("enc.p_cl", lambda: _o(x))
            enc.linear_no_bias_cm.forward = lambda x, _o=o_cm: H.cached("enc.p_cm", lambda: _o(x))
            enc.small_mlp.forward = lambda x, _o=o_mlp: H.cached("enc.p_mlp", lambda: _o(x))
            # NOTE: relu(c_l_q/c_l_k) and the adds still execute per step (cheap elementwise); the GEMMs/MLP do not.
        # ================= atom blocks (encoder + decoder): AdaLN s-terms, kv AdaLN s-terms, local pair bias, gates, CTB s-terms
        if "atom" in parts:
            def hook_atom_block(b, pfx):
                apb = b.attention_pair_bias
                for lnname in ("layernorm_a", "layernorm_kv"):
                    ada = getattr(apb, lnname, None)
                    if ada is None: continue
                    def ada_fwd(a, s, _ada=ada, _n=f"{pfx}.{lnname}"):
                        # stock: a = LN_a(a); s = LN_s(s); a = sigmoid(linear_s(s)) * a + linear_nobias_s(s)
                        a = _ada.layernorm_a(a)
                        memo = {}
                        def s_norm():
                            if "s" not in memo: memo["s"] = _ada.layernorm_s(s)
                            return memo["s"]
                        g = H.cached(_n + ".sig", lambda: torch.sigmoid(_ada.linear_s(s_norm())))
                        b_ = H.cached(_n + ".lin", lambda: _ada.linear_nobias_s(s_norm()))
                        return g * a + b_
                    ada.forward = ada_fwd
                # local pair bias: wrap local_multihead_attention's bias producer via layernorm_z/linear_nobias_z pair
                o_lz = apb.layernorm_z.forward; o_lnz = apb.linear_nobias_z.forward
                class _SkipZ: pass
                def lz_fwd(z, _o=o_lz, _n=f"{pfx}.locbias"):
                    if H.mode == "hit" and H.cur is not None and _n in H.cur["slots"]:
                        return _SkipZ
                    return _o(z)
                def lnz_fwd(x, _o=o_lnz, _n=f"{pfx}.locbias"):
                    # returns [..., nb, nq, nk, H]; stock then permutes (a view) -> cache pre-permute tensor
                    return H.cached(_n, (lambda: _o(x)) if x is not _SkipZ else (lambda: (_ for _ in ()).throw(RuntimeError("locbias producer in hit"))))
                apb.layernorm_z.forward = lz_fwd; apb.linear_nobias_z.forward = lnz_fwd
                # output gate sigmoid(linear_a_last(s)): cache the linear output (sigmoid stays per-step: 1 elementwise) — keeps stock op order
                o_gl = apb.linear_a_last.forward
                apb.linear_a_last.forward = lambda s, _o=o_gl, _n=f"{pfx}.gate_lin": H.cached(_n, lambda: _o(s))
                ctb = b.conditioned_transition_block
                ada = ctb.adaln
                def cada_fwd(a, s, _ada=ada, _n=f"{pfx}.ctb.adaln"):
                    a = _ada.layernorm_a(a)
                    memo = {}
                    def s_norm():
                        if "s" not in memo: memo["s"] = _ada.layernorm_s(s)
                        return memo["s"]
                    g = H.cached(_n + ".sig", lambda: torch.sigmoid(_ada.linear_s(s_norm())))
                    b_ = H.cached(_n + ".lin", lambda: _ada.linear_nobias_s(s_norm()))
                    return g * a + b_
                ada.forward = cada_fwd
                o_cl = ctb.linear_s.forward
                ctb.linear_s.forward = lambda s, _o=o_cl, _n=f"{pfx}.ctb.gate_lin": H.cached(_n, lambda: _o(s))
            for j, b in enumerate(enc.atom_transformer.diffusion_transformer.blocks):
                hook_atom_block(b, f"atomenc{j}")
            for j, b in enumerate(dec.atom_transformer.diffusion_transformer.blocks):
                hook_atom_block(b, f"atomdec{j}")

        # ================= glue (exact, step-invariant by construction; Hoist.glue)
        if self.glue:
            import protenix.model.modules.primitives as PR
            # --- G1: DiffusionConditioning.forward clones the cached pair_z under inplace_safe on EVERY denoiser call (256*N*N fp32).
            #         Nothing downstream writes into the returned z (token blocks read it through normalize()/conv2d, the atom encoder through
            #         layernorm_z; no in-place op targets it), so the per-call clone is served from ONE static slot: record = stock `pair_z.clone()`
            #         copied into the slot, hit = the slot. To stop the stock forward from cloning again it is called with inplace_safe=False; the
            #         only other use of the flag inside this forward is `single_s += transition(single_s)` vs `single_s = single_s + transition(...)`,
            #         which produce bit-equal values (same elementwise add kernel on the same operands; checked under the deterministic
            #         recipe) — the flag changes aliasing, not arithmetic. Guarded by source inspection; refuses to install if the forward changed.
            cond = dm.diffusion_conditioning
            src_cf = inspect.getsource(type(cond).forward)
            ok_src = ("pair_z_clone = pair_z.clone()" in src_cf and "single_s += self.transition_s1(single_s)" in src_cf
                      and "single_s = single_s + self.transition_s1(single_s)" in src_cf and src_cf.count("inplace_safe") <= 8)
            assert ok_src, "DiffusionConditioning.forward changed: refusing G1"
            H.report.setdefault("source_sha", {})["cond_forward"] = hashlib.sha256(src_cf.encode()).hexdigest()[:12]
            o_condf = cond.forward
            def cond_forward(t_hat_noise_level, relp_feature, s_inputs, s_trunk, z_trunk, pair_z=None, inplace_safe=False, use_conditioning=True, _o=o_condf):
                if H._glue_on and inplace_safe and pair_z is not None and H.mode != "off" and H.cur is not None and H._g1_ok:
                    pz = H.cached("glue.pair_z_clone", lambda: pair_z.clone().requires_grad_(False))
                    # ENFORCED read-only guarantee: the served slot must not have been written since it was recorded. torch bumps
                    # Tensor._version on every in-place op (also through views); inference-mode tensors have no counter -> then nothing can be
                    # checked: fall back to the stock clone (H._g1_ok = False for the rest of the process).
                    try:
                        ver = pz._version
                    except Exception as e:
                        if is_oom(e): raise                                   # an out-of-memory is the caller's to see: never rerouted (opt_core.oom)
                        H._g1_ok = False; STATS["glue_clone_uncheckable"] = STATS.get("glue_clone_uncheckable", 0) + 1
                        return _o(t_hat_noise_level=t_hat_noise_level, relp_feature=relp_feature, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk,
                                  pair_z=pair_z, inplace_safe=inplace_safe, use_conditioning=use_conditioning)
                    rec = H._g1_version.get(id(pz))
                    if rec is None:
                        H._g1_version[id(pz)] = ver
                    elif ver != rec:
                        STATS["audit_fail"] += 1
                        raise RuntimeError(f"dit_hoist G1: the served pair_z slot was modified in place (version {rec} -> {ver}); refusing to continue")
                    STATS["glue_clone_served"] = STATS.get("glue_clone_served", 0) + 1
                    return _o(t_hat_noise_level=t_hat_noise_level, relp_feature=relp_feature, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk,
                              pair_z=pz, inplace_safe=False, use_conditioning=use_conditioning)
                return _o(t_hat_noise_level=t_hat_noise_level, relp_feature=relp_feature, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk,
                          pair_z=pair_z, inplace_safe=inplace_safe, use_conditioning=use_conditioning)
            cond.forward = cond_forward
            H.report["glue_G1"] = True
            # --- G2: rearrange_to_dense_trunk(attn_bias=None) rebuilds the local-attention padding bias (zeros [n+q_pad, n+pad_l+pad_r] fp32,
            #         three -inf fills, reshape/permute/reshape copy, unfold view) on every atom-attention call: a pure function of
            #         (n, n_queries, n_keys, inf, dtype, device). Cache the resulting trunked VIEW per key (read-only downstream: `+ trunked_attn_bias`
            #         is out-of-place) -> identical values and strides every call.
            src_r2 = inspect.getsource(PR.rearrange_to_dense_trunk)
            h_r2 = hashlib.sha256(src_r2.encode()).hexdigest()[:12]
            H.report.setdefault("source_sha", {})["r2dt"] = h_r2
            assert "concat_split_data = optimized_concat_split(attn_bias, n_queries)" in src_r2 and "attn_bias[..., n::, :] = -inf" in src_r2, "rearrange_to_dense_trunk changed: refusing G2"
            o_r2dt = PR.rearrange_to_dense_trunk
            def r2dt(q, k, v, n_queries, n_keys, attn_bias=None, inf=1e10, _o=o_r2dt):
                if attn_bias is not None or H.mode == "off" or not H._glue_on:
                    return _o(q, k, v, n_queries, n_keys, attn_bias=attn_bias, inf=inf)
                key = (int(q.shape[-2]), len(q.shape), int(n_queries), int(n_keys), float(inf), q.dtype, str(q.device))
                ent = H._padbias.get(key)
                if ent is None:
                    q_t, k_t, v_t, bias_t, q_pad = _o(q, k, v, n_queries, n_keys, attn_bias=None, inf=inf)
                    if len(H._padbias) >= 8: H._padbias.pop(next(iter(H._padbias)))
                    H._padbias[key] = (bias_t, q_pad)
                    STATS["glue_padbias_builds"] = STATS.get("glue_padbias_builds", 0) + 1
                    return q_t, k_t, v_t, bias_t, q_pad
                bias_t, q_pad = ent
                q_t, kv_t, _pi = PR.rearrange_qk_to_dense_trunk(q=q, k=[k, v], dim_q=-2, dim_k=[-2, -2], n_queries=n_queries, n_keys=n_keys, compute_mask=False)
                assert _pi["q_pad"] == q_pad
                STATS["glue_padbias_hits"] = STATS.get("glue_padbias_hits", 0) + 1
                if H.mode == "audit":
                    ref = _o(q, k, v, n_queries, n_keys, attn_bias=None, inf=inf)[3]
                    ok = ref.shape == bias_t.shape and bool(torch.equal(ref, bias_t)); STATS["audit_checks"] += 1
                    H.report.setdefault("audit", []).append(("glue.padbias", H.step, ok))
                    if not ok: STATS["audit_fail"] += 1; raise RuntimeError("dit_hoist AUDIT FAILED: glue.padbias")
                return q_t, kv_t[0], kv_t[1], bias_t, q_pad
            PR.rearrange_to_dense_trunk = r2dt
            # _local_attention resolved the module-global name at call time -> patched symbol is picked up (same module namespace).
            H.report["glue_G2"] = True
        # ================= denoiser-call counter + eager sampler driver
        o_dmf = dm.forward
        def dm_forward(*a, **k):
            if H.in_sampler:
                H.step += 1; STATS["denoiser_calls"] += 1
                x_noisy = k.get("x_noisy", a[0] if a else None)
                sig = (tuple(x_noisy.shape), str(x_noisy.dtype)) if torch.is_tensor(x_noisy) else None
                if H._eager_driving:
                    if sig != H._x_shape:          # new chunk / new N_sample signature within this sampler call -> re-record
                        H._x_shape = sig; H.step = 0
                        H._bind_eager(("eager", sig))
                        H.set_mode("record")
                    elif H.step in H.audit_steps:
                        H.set_mode("audit")
                    else:
                        H.set_mode("hit")
            out = o_dmf(*a, **k)
            if H._g1_version and H.cur is not None:
                sl = H.cur["slots"].get("glue.pair_z_clone")
                if sl is not None and sl.buf is not None:
                    rec = H._g1_version.get(id(sl.buf))
                    if rec is not None and sl.buf._version != rec:
                        STATS["audit_fail"] += 1
                        raise RuntimeError("dit_hoist G1: pair_z slot modified in place during the denoiser call; refusing")
            return out
        dm.forward = dm_forward
        self._eager_driving = False
        if wrap_eager_sampler:
            import protenix.model.protenix as PM
            o_sd = PM.Protenix.sample_diffusion
            def sd(model_self, *a, **k):
                STATS["sampler_calls"] += 1
                H.in_sampler = True; H.step = -1; H._x_shape = None; H._eager_driving = (sampler is None)
                try:
                    return o_sd(model_self, *a, **k)
                finally:
                    H.in_sampler = False; H._eager_driving = False
                    if sampler is None:
                        H.set_mode("off")
                        if H.release_on_exit: H.release()
            PM.Protenix.sample_diffusion = sd
        if sampler is not None:
            sampler.biascache = self      # graphed.py drives bind/set_mode('record'|'hit'|'audit')/poison through this attribute
        self.installed = True
        self.report["installed"] = True
        self.report["n_tok_blocks"] = len(dm.diffusion_transformer.blocks); self.n_token_blocks = len(dm.diffusion_transformer.blocks)
        self.report["n_atom_blocks"] = (len(enc.atom_transformer.diffusion_transformer.blocks), len(dec.atom_transformer.diffusion_transformer.blocks))
        return self


HOIST = DitHoist()

def install(model, sampler=None, wrap_eager_sampler=True) -> DitHoist:
    return HOIST.install(model, sampler=sampler, wrap_eager_sampler=wrap_eager_sampler)
