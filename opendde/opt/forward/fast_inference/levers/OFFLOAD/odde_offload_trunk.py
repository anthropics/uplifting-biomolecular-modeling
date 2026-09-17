# SPDX-License-Identifier: Apache-2.0
# odde_offload_trunk.py -- ODDE_OFFLOAD trunk / MSA-module / confidence-head / distogram host-offload for OpenDDE 1.1.1 (the pinned
# wheel, stock/PINS.json), built on the primitives of odde_offload.py (HostPair, off_trimul, off_triatt, off_transition, off_apb,
# off_pairformer_block/stack, off_structural_expander, off_diffusion_prepare_pair_cache); the lazy relp rows per row block and the
# chunked LayerNorm prologue are the same devices levers/XL uses.
#
# What is mirrored from where (OpenDDE 1.1.1 wheel; bodies copied verbatim, only tensor residency changed):
#   A  off_get_pairformer_output   <- opendde/model/opendde.py  OpenDDE.get_pairformer_output           (L952-1062)
#      off_zinit_rows              <- the z_init expression of the same function (L987-1003), per row block, relp rows lazy
#   B  off_msa_module              <- opendde/model/modules/pairformer.py MSAModule.forward              (L2008-2073)
#      off_msa_block               <- MSABlock.forward (L1649-1728; order: msa_stack -> z += OPM(m) -> pair_stack(c_s=0))
#      off_msa_stack               <- MSAStack.inference_forward (L896-953)
#      off_msa_pwa                 <- MSAPairWeightedAveraging.forward (L799-842)
#      off_opm_add                 <- opendde/model/triangular/layers.py OuterProductMean._forward + chunk_layer rows
#   C  off_confidence_head         <- opendde/model/modules/confidence.py ConfidenceHead.forward         (L201-486)
#      off_confidence_mef          <- ConfidenceHead.memory_efficient_forward (L649-814)
#      off_contact_probs           <- opendde/model/modules/head.py DistogramHead.forward + sample_confidence.compute_contact_prob
#   D  install_trunk(stages)       <- called by odde_offload.install(); patches OpenDDE.{get_pairformer_output,
#                                     run_confidence_head, compute_distogram_contact_probs} (+ a materialising guard on
#                                     expand_to_structural_tokens when the 'struct' stage is not offloaded).  OpenDDE.forward
#                                     is upstream's own (`_tf32_runtime_scope` + `_forward_impl`: relp always lazy, the N^2-bounded
#                                     dynamic chunk at the trunk and at the 2N structural-refiner call site, multi-seed CPU retention):
#                                     the unit hooks stage methods only, so those policies apply to an offloaded run unchanged.
# Numerics: fp32, stock weights, stock sub-module code on row blocks; no algorithmic substitution.  Row-blocked LayerNorm/GEMM
# launches differ in SHAPE from the stock full-tensor launches -> tier 2 (bit-exactness to stock is not claimed),
# exactly like odde_offload.py.  RNG: the only RNG consumer in these stages is the stock MSA row sub-sampling
# (opendde/model/msa_sampling.py: torch.randperm on the msa tensor's device, global generator, once per recycle) which is CALLED
# UNCHANGED (MSAModule._prepare_msa_sample) -> identical draw sequence under the DET recipe.  This module consumes no RNG.
# OPM schedule under the deterministic recipe: with torch.use_deterministic_algorithms(True) and no grad, the stock
# MSABlock.forward computes the outer-product-mean pair update with `_deterministic_outer_product_mean_full_update` (a 2x2
# row/column tile schedule, pairformer.py L1702-1714); the mirror (off_msa_block -> off_opm_add) runs the chunked
# `OuterProductMean._forward` schedule at every setting, row blocks aligned to the stock chunk grid.  Same operands and formula,
# a different fp32 summation order under DET -> the trunk stage is tier 2 against a DET stock run by construction (without
# deterministic algorithms stock takes the same chunked schedule and the mirror is bitwise; the unit's CPU equality test runs
# that way).
import os, time, math
from typing import Optional, Any
import torch
import torch.nn.functional as F

import odde_offload as OO
from opendde_opt.tp import upstream_stage_cast            # rows read inside upstream's autocast-disabled stages get the decorator's fp32 cast (bf16 base); a no-op on fp32 rows
from odde_offload import HostPair, blocks, CFG, STATS, log, gpu_mem_str, host_rss_gib

__version__ = "0.2"

OPT = {
    # z_init residency: 'recompute' rebuilds the (cheap, deterministic) z_init row block from s_init every recycle -> ONE host
    # pair tensor; 'host' keeps z_init as a second HostPair exactly as the stock keeps a second GPU tensor.  Bitwise identical.
    "zinit": os.environ.get("ODDE_OFFLOAD_ZINIT", "recompute"),
    # MSAPairWeightedAveraging softmax weights: 'full' holds b/w = [N, N, n_heads] on the GPU (stock softmax + einsum shapes,
    # N^2*8*4 B); 'rows' streams row blocks (softmax over j is row-local).
    "pwa_w": os.environ.get("ODDE_OFFLOAD_PWA_W", "full"),
    # OuterProductMean row sub-chunk when the caller's chunk_size is None (stock would run one [N,N,c,c] einsum): rows per einsum.
    "opm_rows": int(os.environ.get("ODDE_OFFLOAD_OPM_ROWS", "0") or 0),
    # distogram contact probs: 'rows' never materialises [N,N,no_bins] (linear on row block + linear on column block);
    # 'full' builds the stock [N,N,no_bins] logits tensor on the GPU from row blocks and runs the stock symmetrisation/softmax.
    "disto": os.environ.get("ODDE_OFFLOAD_DISTO", "rows"),
    # per-recycle trunk checkpoint (s, z, rng) into CFG['ckpt_dir'] every k cycles (0 = off); size-guarded by CFG['ckpt_max_gb']
    "cycle_ckpt": int(os.environ.get("ODDE_OFFLOAD_CYCLE_CKPT", "1") or 0),   # save trunk_cycle.pt (s, z, RNG, cycle index) every k cycles when ODDE_OFFLOAD_CKPT_DIR is set
}
# ODDE_OFFLOAD_LAZY_RELP is RETIRED: OpenDDE 1.1.1 always carries input_feature_dict['relp'] as a LazyRelativePositionEncodingFeatures
# (opendde.py `_forward_impl` -> generate_relp(lazy=True)); off_zinit_rows materialises relp row blocks from it.  "1" (the behaviour
# upstream has) or unset is accepted; "0" (an eager [N,N,139] one-hot) has no implementation and refuses by name.
OO.refuse_retired("ODDE_OFFLOAD_LAZY_RELP", accepted=("", "1"), reason="retired:relp_always_lazy_upstream")
STATS.setdefault("h2h_bytes", 0)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def hostpair_like(hp: HostPair) -> HostPair:
    return HostPair(hp.n, hp.c, dtype=hp.dtype, device=hp.device, disk_dir=(CFG["xbuf"][5:] if hp.kind == "disk" else ""))


def hostpair_clone(hp: HostPair) -> HostPair:
    """host-side copy (== z.clone() of the stock code)."""
    out = hostpair_like(hp)
    out.t.copy_(hp.t)
    STATS["h2h_bytes"] += hp.t.numel() * hp.t.element_size()
    return out


def ensure_hostpair(z, rows=None) -> HostPair:
    if isinstance(z, HostPair):
        return z
    return HostPair.from_tensor(z, rows=rows or CFG["rows"])


def _relp_rows(relp, r0, r1):
    """rows r0:r1 of the relative-position one-hot features ([r1-r0, N, 139]); lazy (upstream
    LazyRelativePositionEncodingFeatures.materialize) or eager tensor rows."""
    if hasattr(relp, "materialize"):
        return relp.materialize(slice(r0, r1), slice(None))
    return relp[..., r0:r1, :, :]


# ==========================================================================================================================
# A. trunk: z_init rows, recycling, get_pairformer_output
# ==========================================================================================================================
@torch.no_grad()
def off_zinit_rows(model, input_feature_dict, z1, z2, r0, r1, inplace_safe):
    """rows r0:r1 of z_init, verbatim from OpenDDE.get_pairformer_output:
         z_init = linear_no_bias_zinit1(s_init)[..., None, :] + linear_no_bias_zinit2(s_init)[..., None, :, :]
         z_init += relative_position_encoding(relp);  z_init += linear_no_bias_token_bond(token_bonds.unsqueeze(-1))
    with z1 = linear_no_bias_zinit1(s_init), z2 = linear_no_bias_zinit2(s_init) precomputed ([N, c_z] each)."""
    z_init = z1[..., r0:r1, None, :] + z2[..., None, :, :]
    rel = _relp_rows(input_feature_dict["relp"], r0, r1)
    tb = input_feature_dict["token_bonds"][..., r0:r1, :]
    if inplace_safe:
        z_init += model.relative_position_encoding(rel)
        z_init += model.linear_no_bias_token_bond(tb.unsqueeze(dim=-1))
    else:
        z_init = z_init + model.relative_position_encoding(rel)
        z_init = z_init + model.linear_no_bias_token_bond(tb.unsqueeze(dim=-1))
    return z_init


@torch.no_grad()
def off_build_z_init(model, input_feature_dict, s_init, inplace_safe=True, rows=None) -> HostPair:
    """z_init as a HostPair (OPT['zinit'] == 'host')."""
    rows = rows or CFG["rows"]
    n = s_init.shape[-2]
    z1 = model.linear_no_bias_zinit1(s_init); z2 = model.linear_no_bias_zinit2(s_init)
    hp = HostPair(n, z1.shape[-1], dtype=z1.dtype, device=s_init.device)
    for r0, r1 in blocks(n, rows):
        hp.put_rows(r0, r1, off_zinit_rows(model, input_feature_dict, z1, z2, r0, r1, inplace_safe))
    return hp


@torch.no_grad()
def off_recycle_z(model, input_feature_dict, s_init, hp: HostPair, zinit_hp: Optional[HostPair], first_cycle: bool, inplace_safe: bool,
                  template_zero_add: bool, rows=None):
    """z = z_init + linear_no_bias_z_cycle(layernorm_z_cycle(z))   [+ 0 from a template embedder that returns 0], row-blocked
    IN PLACE on the host z.  On the first cycle the stock z is torch.zeros_like(z_init): zero rows are generated on the GPU."""
    rows = rows or CFG["rows"]
    n = hp.n
    z1 = z2 = None
    if zinit_hp is None:
        z1 = model.linear_no_bias_zinit1(s_init); z2 = model.linear_no_bias_zinit2(s_init)
    for r0, r1 in blocks(n, rows):
        z_init = zinit_hp.rows(r0, r1) if zinit_hp is not None else off_zinit_rows(model, input_feature_dict, z1, z2, r0, r1, inplace_safe)
        if first_cycle:
            z = torch.zeros_like(z_init)
        else:
            z = hp.rows(r0, r1)
        z = z_init + model.linear_no_bias_z_cycle(model.layernorm_z_cycle(z))
        if template_zero_add:               # stock: z += self.template_embedder(...) which returned the python int 0
            if inplace_safe:
                z += 0
            else:
                z = z + 0
        hp.put_rows(r0, r1, z)
        del z, z_init


@torch.no_grad()
def off_template_add(te, input_feature_dict, hp: HostPair, triangle_attention="torch", triangle_multiplicative="torch", inplace_safe=True, chunk_size=None, rows=None):
    """z += TemplateEmbedder.forward(input_feature_dict, z, ...)  (pairformer.py L1585-1612 + single_template_forward L1891-1943) with z host-resident.
    Per template: v = linear_no_bias_z(layernorm_z(z)) + linear_no_bias_a(at) is built ROW-BLOCKED into a GPU
    [N, N, c=64] tensor (`at` rows assembled exactly as stock: dgram, pseudo-beta mask, restype_j, restype_i, unit vector, backbone mask, all times
    multichain_mask * pair_mask); the c=64 template PairformerStack and layernorm_v run STOCK on the GPU ([N,N,64] = 1/6 of a trunk pair tensor);
    u = sum_t / (1e-7 + T); finally z rows += linear_no_bias_u(relu(u rows)) into the host tensor."""
    from opendde.model.utils import expand_at_dim
    try:
        from opendde.data.constants import STD_RESIDUES_WITH_GAP
    except Exception:
        from opendde.model.modules.pairformer import STD_RESIDUES_WITH_GAP
    rows = rows or CFG["rows"]
    if "template_aatype" not in input_feature_dict or te.n_blocks < 1:
        return
    n = hp.n; dev = hp.device
    asym_id = input_feature_dict["asym_id"]
    multichain_mask = (asym_id[:, None] == asym_id[None, :]).to(hp.dtype)
    num_templates = input_feature_dict["template_aatype"].shape[0]
    pair_mask = torch.ones((n, n), dtype=hp.dtype, device=dev)
    n_restype = len(STD_RESIDUES_WITH_GAP)
    u = 0
    for template_id in range(num_templates):
        dgram_t = input_feature_dict["template_distogram"][template_id]
        pbm_t = input_feature_dict["template_pseudo_beta_mask"][template_id]
        aatype = F.one_hot(input_feature_dict["template_aatype"][template_id], num_classes=n_restype)
        uv_t = input_feature_dict["template_unit_vector"][template_id]
        bbm_t = input_feature_dict["template_backbone_frame_mask"][template_id]
        v = torch.empty((n, n, te.c), dtype=hp.dtype, device=dev)
        for r0, r1 in blocks(n, rows):
            mm = multichain_mask[r0:r1]; pm = pair_mask[r0:r1]
            to_concat = [dgram_t[r0:r1] * mm[..., None] * pm[..., None],
                         (pbm_t[r0:r1] * mm * pm).unsqueeze(-1),
                         expand_at_dim(aatype, dim=-3, n=r1 - r0),
                         expand_at_dim(aatype[r0:r1], dim=-2, n=n),
                         uv_t[r0:r1] * mm[..., None] * pm[..., None],
                         (bbm_t[r0:r1] * mm * pm).unsqueeze(-1)]
            at = torch.concat(to_concat, dim=-1)
            v[r0:r1] = te.linear_no_bias_z(te.layernorm_z(hp.rows(r0, r1))) + te.linear_no_bias_a(at)
            del at, to_concat
        _, v = te.pairformer_stack(s=None, z=v, pair_mask=pair_mask, triangle_multiplicative=triangle_multiplicative, triangle_attention=triangle_attention,
                                   inplace_safe=inplace_safe, chunk_size=chunk_size)
        v = te.layernorm_v(v)
        u = u + v
        del v
    u = u / (1e-7 + num_templates)
    for r0, r1 in blocks(n, rows):
        upd = te.linear_no_bias_u(te.relu(u[r0:r1]))
        z = hp.rows(r0, r1)
        z += upd                      # stock: z += self.template_embedder(...) (inplace_safe branch)
        hp.put_rows(r0, r1, z); del z, upd
    del u
    log(f"template embedder: T={num_templates} n={n} c={te.c} (c=64 pair stack on GPU, z update row-blocked) {gpu_mem_str()}")


def _template_contributes(model, input_feature_dict) -> bool:
    """TemplateEmbedder.forward (pairformer.py L1563+) returns the int 0 when 'template_aatype' is absent or n_blocks < 1."""
    te = model.template_embedder
    return te.n_blocks > 0 and ("template_aatype" in input_feature_dict)


def _rng_state():
    import random as _random
    st = {"cpu": torch.get_rng_state(), "py": _random.getstate()}
    try:
        import numpy as _np; st["numpy"] = _np.random.get_state()
    except Exception:
        pass
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _rng_restore(st):
    import random as _random
    torch.set_rng_state(st["cpu"])
    if "py" in st: _random.setstate(st["py"])
    if "numpy" in st:
        try:
            import numpy as _np; _np.random.set_state(st["numpy"])
        except Exception:
            pass
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


@torch.no_grad()
def off_get_pairformer_output(model, input_feature_dict, N_cycle, inplace_safe=False, chunk_size=None):
    """== OpenDDE.get_pairformer_output (opendde.py L952-1062) with z_init / z host-resident.  Returns (s_inputs, s, z: HostPair)."""
    self = model
    OO.count("trunk")                                                     # RAN-OR-REFUSE: pair_offload_trunk engaged (one per forward)
    # Line 1-5
    s_inputs = self.input_embedder(input_feature_dict, inplace_safe=False, chunk_size=chunk_size)  # [..., N_token, 449]
    s_init = self.linear_no_bias_sinit(s_inputs)  # [..., N_token, c_s]
    foldcp_result = self._maybe_get_pairformer_output_foldcp_local(
        input_feature_dict=input_feature_dict, N_cycle=N_cycle, s_inputs=s_inputs, s_init=s_init, inplace_safe=inplace_safe, chunk_size=chunk_size)
    if foldcp_result is not None:
        return foldcp_result
    assert s_init.dim() == 2, f"offload trunk expects unbatched inference tensors, got s_init {tuple(s_init.shape)}"
    n = s_init.shape[-2]
    template_real = _template_contributes(self, input_feature_dict)      # stock: z += TemplateEmbedder(...) (a real [N,N,c_z] update, even for dummy templates)
    template_zero_add = (self.template_embedder.n_blocks > 0) and not template_real   # stock then executes `z += 0` / `z = z + 0`
    t0 = time.time()
    zinit_hp = off_build_z_init(self, input_feature_dict, s_init, inplace_safe) if OPT["zinit"] == "host" else None
    # Line 6
    hp = HostPair(n, self.c_z, dtype=s_init.dtype, device=s_init.device)     # z (zeros never materialised: see off_recycle_z)
    s = torch.zeros_like(s_init)
    log(f"trunk: N={n} c_z={self.c_z} N_cycle={N_cycle} chunk_size={chunk_size} zinit={OPT['zinit']} pwa_w={OPT['pwa_w']} msa_depth={self.msa_module.msa_depth} "
        f"template_zero_add={template_zero_add} {gpu_mem_str()} ({time.time()-t0:.1f}s)")

    # per-cycle resume (ODDE_OFFLOAD_RESUME=cycle): continue after the last checkpointed cycle with the RNG stream restored
    start_cycle = 0
    if CFG["resume"] == "cycle" and CFG["ckpt_dir"] and os.path.exists(os.path.join(CFG["ckpt_dir"], "trunk_cycle.pt")) and (ck := OO.ckpt_load("trunk_cycle")) is not None:
        start_cycle = int(ck["cycle_done"])
        hp.t.copy_(ck["z"]); s = ck["s"].to(s_init.device); _rng_restore(ck["rng"])
        log(f"RESUMED trunk after cycle {start_cycle}/{N_cycle} from trunk_cycle.pt (RNG cpu/cuda/numpy/python restored) -> row label TIER-2-BY-RESUME unless a DET resumed-vs-uninterrupted equality has passed for this kit version")
    # Line 7-13 recycling
    for cycle_no in range(start_cycle, N_cycle):
        tc = time.time()
        with torch.set_grad_enabled(False):
            off_recycle_z(self, input_feature_dict, s_init, hp, zinit_hp, first_cycle=(cycle_no == 0), inplace_safe=inplace_safe, template_zero_add=template_zero_add)
            if template_real:
                assert inplace_safe, "stock only applies the template embedder on the inplace_safe (inference) path"
                off_template_add(self.template_embedder, input_feature_dict, hp, triangle_multiplicative=self.configs.triangle_multiplicative,
                                 triangle_attention=self.configs.triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size)
            _sync(); t_rec = time.time() - tc
            tm = time.time()
            off_msa_module(self.msa_module, input_feature_dict, hp, s_inputs, pair_mask=None,
                           triangle_multiplicative=self.configs.triangle_multiplicative, triangle_attention=self.configs.triangle_attention,
                           inplace_safe=inplace_safe, chunk_size=chunk_size, tag=f"msa c{cycle_no+1}")
            _sync(); t_msa = time.time() - tm
            s = s_init + self.linear_no_bias_s(self.layernorm_s(s))
            s = OO.off_pairformer_stack(self.pairformer_stack, s, hp,
                                        triangle_multiplicative=self.configs.triangle_multiplicative, triangle_attention=self.configs.triangle_attention,
                                        inplace_safe=inplace_safe, chunk_size=chunk_size, tag=f"trunk c{cycle_no+1}/{N_cycle}", every=8)
        _sync()
        log(f"trunk cycle {cycle_no+1}/{N_cycle} done: recycle={t_rec:.1f}s msa={t_msa:.1f}s total={time.time()-tc:.1f}s {gpu_mem_str()} rss={host_rss_gib()[0]:.1f}GiB")
        if OPT["cycle_ckpt"] and CFG["ckpt_dir"] and (cycle_no + 1) % OPT["cycle_ckpt"] == 0 and cycle_no + 1 < N_cycle \
                and hp.t.numel() * 4 / 2**30 <= CFG["ckpt_max_gb"]:
            OO.ckpt_save("trunk_cycle", {"cycle_done": cycle_no + 1, "s": s.cpu(), "z": hp.t, "rng": _rng_state(), "s_inputs": s_inputs.cpu()})
    if zinit_hp is not None:
        zinit_hp.free()
    return s_inputs, s, hp


# ==========================================================================================================================
# B. MSA module
# ==========================================================================================================================
@torch.no_grad()
def _pwa_weights_full(pwa, hp: HostPair, rows):
    """b = linear_no_bias_z(layernorm_z(z)) assembled on the GPU from row blocks ([N, N, n_heads]); w = softmax over dim=-2 (stock)."""
    n = hp.n
    b = None
    for r0, r1 in blocks(n, rows):
        br = pwa.linear_no_bias_z(pwa.layernorm_z(hp.rows(r0, r1)))
        if b is None:
            b = torch.empty((n, n, br.shape[-1]), dtype=br.dtype, device=br.device)
        b[r0:r1] = br; del br
    w = pwa.softmax_w(b)  # [...,n_token, n_token, n_heads]
    del b
    return w


@torch.no_grad()
def off_msa_pwa(pwa, m, hp: HostPair, cache: dict, rows=None):
    """== MSAPairWeightedAveraging.forward(m, z) (pairformer.py L555-599) with z host-resident.  The pair logits/softmax weights
    depend on z only and are cached in `cache` across the msa-row chunks of one MSAStack call (stock recomputes identical values)."""
    rows = rows or CFG["rows"]
    # Input projections
    m = pwa.layernorm_m(m)  # [...,n_msa_sampled, n_token, c_m]
    v = pwa.linear_no_bias_mv(m)  # [...,n_msa_sampled, n_token, n_heads * c]
    v = v.reshape(*v.shape[:-1], pwa.n_heads, pwa.c)  # [...,n_msa_sampled, n_token, n_heads, c]
    g = torch.sigmoid(pwa.linear_no_bias_mg(m))  # [...,n_msa_sampled, n_token, n_heads * c]
    g = g.reshape(*g.shape[:-1], pwa.n_heads, pwa.c)  # [...,n_msa_sampled, n_token, n_heads, c]
    if OPT["pwa_w"] == "full":
        if cache.get("w") is None:
            cache["w"] = _pwa_weights_full(pwa, hp, rows)
        w = cache["w"]
        wv = torch.einsum("...ijh,...mjhc->...mihc", w, v)  # [...,n_msa_sampled,n_token,n_heads,c]
    else:   # row-streamed: softmax over j (dim=-2) is local to the row block
        n = hp.n
        wv = torch.empty((*v.shape[:-3], n, pwa.n_heads, pwa.c), dtype=v.dtype, device=v.device)
        for r0, r1 in blocks(n, rows):
            b = pwa.linear_no_bias_z(pwa.layernorm_z(hp.rows(r0, r1)))
            w = pwa.softmax_w(b)
            wv[..., r0:r1, :, :] = torch.einsum("...ijh,...mjhc->...mihc", w, v)
            del b, w
    o = g * wv
    o = o.reshape(*o.shape[:-2], pwa.n_heads * pwa.c)  # [...,n_msa_sampled, n_token, n_heads * c]
    m = pwa.linear_no_bias_out(o)  # [...,n_msa_sampled, n_token, c_m]
    del v, g, wv, o
    return m


@torch.no_grad()
def off_msa_stack(stack, m, hp: HostPair):
    """== MSAStack.forward -> inference_forward(m, z, msa_chunk_size) (pairformer.py L626-698), z host-resident."""
    chunk_size = stack.msa_chunk_size
    num_msa = m.shape[-3]
    if chunk_size is None:
        chunk_size = max(num_msa, 1)
    no_chunks = num_msa // chunk_size + (num_msa % chunk_size != 0)
    cache = {}
    for i in range(no_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, num_msa)
        # Use inplace to save memory
        msa_update = off_msa_pwa(stack.msa_pair_weighted_averaging, m[start:end, :, :], hp, cache)
        m[start:end, :, :] += msa_update
        m[start:end, :, :] += stack.transition_m(m[start:end, :, :])
    cache.clear()
    return m


@torch.no_grad()
def off_opm_add(opm, m, hp: HostPair, inplace_safe: bool, chunk_size: Optional[int], mask=None, rows=None):
    """z = z + OuterProductMean(m)  (layers.py OuterProductMean._forward L535-585; MSABlock passes no mask), z host-resident.
    Row blocks of the output are what the stock chunk_layer(partial(_opm, b=b), {'a': a}, chunk_size, no_batch_dims=1) computes;
    the einsum sub-chunks are aligned to the stock chunk grid (multiples of chunk_size) inside every host row block."""
    from opendde.model.utils import is_fp16_enabled
    rows = rows or CFG["rows"]
    if is_fp16_enabled():
        m = m.float()
    if mask is None:
        mask = m.new_ones(m.shape[:-1])
    # [*, N_seq, N_res, C_m]
    ln = opm.layer_norm(m)
    # [*, N_seq, N_res, C]
    mask = mask.unsqueeze(-1)
    a = opm.linear_1(ln)
    a = a * mask
    b = opm.linear_2(ln)
    b = b * mask
    del ln
    a = a.transpose(-2, -3)
    b = b.transpose(-2, -3)
    n = hp.n
    sub = chunk_size if chunk_size is not None else (OPT["opm_rows"] or rows)
    t0 = time.time()
    for r0, r1 in blocks(n, rows):
        if chunk_size is None and sub >= (r1 - r0):
            outer = opm._opm(a[..., r0:r1, :, :], b)
        else:
            outer = None
            q = r0
            while q < r1:
                q1 = min(((q // sub) + 1) * sub, r1)          # stock chunk boundaries are multiples of chunk_size
                oc = opm._opm(a[..., q:q1, :, :], b)
                if outer is None:
                    outer = oc.new_zeros((*oc.shape[:-3], r1 - r0, *oc.shape[-2:]))
                outer[..., q - r0:q1 - r0, :, :] = oc
                del oc; q = q1
        # [*, rows, N_res, 1]
        norm = torch.einsum("...abc,...adc->...bdc", mask[..., :, r0:r1, :], mask)
        norm = norm + opm.eps
        if inplace_safe:
            outer /= norm
        else:
            outer = outer / norm
        z = hp.rows(r0, r1)
        z = z + outer                       # stock (both branches): z = z + self.outer_product_mean_msa(...)
        hp.put_rows(r0, r1, z)
        del z, outer, norm
    del a, b
    _sync()
    log(f"opm n={n} n_msa={m.shape[-3]} rows={rows} sub={sub} {time.time()-t0:.1f}s")


@torch.no_grad()
def off_msa_block(block, m, hp: HostPair, pair_mask=None, triangle_multiplicative="torch", triangle_attention="torch", inplace_safe=False, chunk_size=None):
    """== MSABlock.forward (pairformer.py L1110-1181): upstream updates the MSA first, then writes the refreshed MSA state back to z,
    then runs the pair stack (a PairformerBlock with c_s=0 -> odde_offload.off_pairformer_block)."""
    assert pair_mask is None, "offload: pair_mask must be None (OpenDDE inference passes None)"
    m = off_msa_stack(block.msa_stack, m, hp)
    off_opm_add(block.outer_product_mean_msa, m, hp, inplace_safe=inplace_safe, chunk_size=chunk_size)
    OO.off_pairformer_block(block.pair_stack, None, hp, triangle_multiplicative, triangle_attention, inplace_safe, chunk_size)
    if block.is_last_block:
        return None
    return m


@torch.no_grad()
def off_msa_module(msa_module, input_feature_dict, hp: HostPair, s_inputs, pair_mask=None, triangle_multiplicative="torch", triangle_attention="torch",
                   inplace_safe=False, chunk_size=None, tag="msa"):
    """== MSAModule.forward (pairformer.py L1436-1503) with z host-resident (updated in place); returns hp."""
    self = msa_module
    t0 = time.time()
    msa_sample = self._prepare_msa_sample(input_feature_dict=input_feature_dict, s_inputs=s_inputs, z_token_dim=hp.n)   # stock (RNG: torch.randperm)
    if msa_sample is None:
        return hp
    assert self._maybe_foldcp_mesh() is None, "offload: fold-CP distributed mode is not supported"
    n_msa = int(msa_sample.shape[-3])
    # == checkpoint_blocks(self._prep_blocks(...), args=(msa_sample, z), blocks_per_ckpt=None)
    for bi, block in enumerate(self.blocks):
        tb = time.time()
        msa_sample = off_msa_block(block, msa_sample, hp, pair_mask=pair_mask, triangle_multiplicative=triangle_multiplicative,
                                   triangle_attention=triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size)
        _sync()
        log(f"{tag} block {bi+1}/{len(self.blocks)} n_msa={n_msa} {time.time()-tb:.1f}s {gpu_mem_str()} rss={host_rss_gib()[0]:.1f}GiB")
    log(f"{tag} module done {time.time()-t0:.1f}s")
    return hp


# ==========================================================================================================================
# C. confidence head + distogram/contact probabilities
# ==========================================================================================================================
@torch.no_grad()
def off_confidence_mef(head, input_feature_dict, s_trunk, hp: HostPair, pair_mask, x_pred_rep_coords, triangle_multiplicative="torch",
                       triangle_attention="torch", inplace_safe=False, chunk_size=None, compute_plddt=True, compute_pae=True, compute_pde=True,
                       compute_resolved=True, rows=None):
    """== ConfidenceHead.memory_efficient_forward (confidence.py L649-814) with z_pair host-resident (modified in place)."""
    from opendde.model.utils import broadcast_token_to_atom, one_hot
    self = head
    rows = rows or CFG["rows"]
    assert pair_mask is None, "offload: pair_mask must be None (OpenDDE inference passes None)"
    n = hp.n
    # Embed pair distances of representative atoms:
    with torch.amp.autocast("cuda", enabled=False):
        x_pred_rep_coords = x_pred_rep_coords.to(torch.float32)
        distance_pred = torch.cdist(x_pred_rep_coords, x_pred_rep_coords)  # [..., N_tokens, N_tokens]
    for r0, r1 in blocks(n, rows):
        z_pair = hp.rows(r0, r1)
        d = distance_pred[..., r0:r1, :]
        if inplace_safe:
            z_pair += self.linear_no_bias_d(one_hot(x=d, lower_bins=self.lower_bins, upper_bins=self.upper_bins).to(dtype=self.linear_no_bias_d.weight.dtype))
            z_pair += self.linear_no_bias_d_wo_onehot(d.unsqueeze(dim=-1).to(dtype=self.linear_no_bias_d_wo_onehot.weight.dtype))
        else:
            z_pair = z_pair + self.linear_no_bias_d(one_hot(x=d, lower_bins=self.lower_bins, upper_bins=self.upper_bins).to(dtype=self.linear_no_bias_d.weight.dtype))
            z_pair = z_pair + self.linear_no_bias_d_wo_onehot(d.unsqueeze(dim=-1).to(dtype=self.linear_no_bias_d_wo_onehot.weight.dtype))
        hp.put_rows(r0, r1, z_pair); del z_pair, d
    del distance_pred
    # Line 4
    s_single = OO.off_pairformer_stack(self.pairformer_stack, s_trunk, hp, triangle_multiplicative=triangle_multiplicative, triangle_attention=triangle_attention,
                                       inplace_safe=inplace_safe, chunk_size=chunk_size, extra_attn_bias=input_feature_dict.get("structural_pair_attn_bias", None),
                                       tag="confidence")
    if compute_plddt or compute_resolved:
        s_single = s_single.to(torch.float32)
    atom_to_token_idx = input_feature_dict["atom_to_token_idx"]
    atom_to_tokatom_idx = input_feature_dict["atom_to_tokatom_idx"]
    with torch.amp.autocast("cuda", enabled=False):
        pae_pred = pde_pred = None
        if compute_pae or compute_pde:
            for r0, r1 in blocks(n, rows):
                z_pair = hp.rows(r0, r1).to(torch.float32)
                if compute_pae:
                    p = self.linear_no_bias_pae(self.pae_ln(z_pair))
                    if pae_pred is None:
                        pae_pred = torch.empty((*p.shape[:-3], n, n, p.shape[-1]), dtype=p.dtype, device=p.device)
                    pae_pred[..., r0:r1, :, :] = p; del p
                if compute_pde:
                    zt = hp.rows_T(r0, r1).to(torch.float32)             # rows r0:r1 of z_pair.transpose(-2, -3)
                    p = self.linear_no_bias_pde(self.pde_ln(z_pair + zt))
                    if pde_pred is None:
                        pde_pred = torch.empty((*p.shape[:-3], n, n, p.shape[-1]), dtype=p.dtype, device=p.device)
                    pde_pred[..., r0:r1, :, :] = p; del p, zt
                del z_pair
        if compute_plddt or compute_resolved:
            a = broadcast_token_to_atom(x_token=s_single, atom_to_token_idx=atom_to_token_idx)
            plddt_pred = (torch.einsum("...nc,ncb->...nb", self.plddt_ln(a), self.plddt_weight[atom_to_tokatom_idx]) if compute_plddt else None)
            resolved_pred = (torch.einsum("...nc,ncb->...nb", self.resolved_ln(a), self.resolved_weight[atom_to_tokatom_idx]) if compute_resolved else None)
        else:
            plddt_pred = None
            resolved_pred = None
    if n > 2000 and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return plddt_pred, pae_pred, pde_pred, resolved_pred


@torch.no_grad()
def off_confidence_head(head, input_feature_dict, s_inputs, s_trunk, z_trunk, pair_mask, x_pred_coords, z_trunk_spec=None, triangle_multiplicative="torch",
                        triangle_attention="torch", inplace_safe=False, chunk_size=None, compute_plddt=True, compute_pae=True, compute_pde=True,
                        compute_resolved=True, rows=None, consume_z=False):
    """== ConfidenceHead.forward (confidence.py L201-486) with z_trunk a HostPair (or a tensor, converted).  Every sample runs on a
    fresh host copy of (z_init + z_trunk) (== the stock per-sample clone); with consume_z=True the last sample reuses the trunk
    HostPair itself (the caller guarantees nobody reads z_trunk afterwards).  Sample outputs are retained exactly as the stock does:
    preallocated [N_sample, ...] buffers filled per sample, PAE/PDE logits on the CPU when N_sample > 1 and no grad is recorded
    (the stock `offload_pair_outputs` policy), pLDDT/resolved on the compute device."""
    self = head
    rows = rows or CFG["rows"]
    OO.count("conf")                                                       # RAN-OR-REFUSE: pair_offload_conf engaged (one per confidence stage)
    assert z_trunk_spec is None and self._maybe_create_foldcp_mesh() is None, "offload: fold-CP distributed mode is not supported"
    s_inputs = s_inputs.detach()
    s_trunk = s_trunk.detach()
    hp0 = ensure_hostpair(z_trunk, rows)
    own_hp0 = consume_z or (hp0 is not z_trunk)          # we may overwrite / reuse hp0
    s_trunk = self.input_strunk_ln(torch.clamp(s_trunk, min=-512, max=512))
    x_rep_atom_mask = self._select_distogram_rep_atom_mask(input_feature_dict=input_feature_dict, n_token=s_trunk.shape[-2])
    x_pred_rep_coords = x_pred_coords[..., x_rep_atom_mask, :]
    N_sample = x_pred_rep_coords.size(-3)
    n = hp0.n
    # z_init = linear_no_bias_s1(s_inputs)[..., None, :, :] + linear_no_bias_s2(s_inputs)[..., None, :];  z_trunk = z_init + z_trunk
    s1 = self.linear_no_bias_s1(s_inputs); s2 = self.linear_no_bias_s2(s_inputs)
    zt = hp0 if own_hp0 else hostpair_like(hp0)
    for r0, r1 in blocks(n, rows):
        z_init = s1[..., None, :, :] + s2[..., r0:r1, None, :]
        zt.put_rows(r0, r1, z_init + hp0.rows(r0, r1)); del z_init
    del s1, s2
    plddt_preds = pae_preds = pde_preds = resolved_preds = None
    offload_pair_outputs = N_sample > 1 and not torch.is_grad_enabled()   # stock: completed PAE/PDE samples are retained on the CPU

    def _allocate_sample_output(prediction, sample_dim):                   # == the stock closure: [.., N_sample, ..] on the prediction's device
        if prediction is None:
            return None
        output_shape = list(prediction.shape)
        output_shape.insert(sample_dim if sample_dim >= 0 else prediction.ndim + 1 + sample_dim, N_sample)
        return prediction.new_empty(output_shape)

    t0 = time.time()
    for i in range(N_sample):
        last = i == N_sample - 1
        z_pair = zt if last else hostpair_clone(zt)      # stock: z_trunk.clone() per sample; zt is ours -> the last sample consumes it
        plddt_pred, pae_pred, pde_pred, resolved_pred = off_confidence_mef(
            self, input_feature_dict, s_trunk.clone() if inplace_safe else s_trunk, z_pair, pair_mask, x_pred_rep_coords[..., i, :, :],
            triangle_multiplicative=triangle_multiplicative, triangle_attention=triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size,
            compute_plddt=compute_plddt, compute_pae=compute_pae, compute_pde=compute_pde, compute_resolved=compute_resolved, rows=rows)
        if z_pair is not zt:
            z_pair.free()
        if offload_pair_outputs:
            if pae_pred is not None: pae_pred = pae_pred.to(device="cpu")
            if pde_pred is not None: pde_pred = pde_pred.to(device="cpu")
        if i == 0:
            plddt_preds, pae_preds, pde_preds, resolved_preds = (_allocate_sample_output(plddt_pred, -3), _allocate_sample_output(pae_pred, -4),
                                                              _allocate_sample_output(pde_pred, -4), _allocate_sample_output(resolved_pred, -3))
        for output, prediction, sample_dim in ((plddt_preds, plddt_pred, -3), (pae_preds, pae_pred, -4), (pde_preds, pde_pred, -4), (resolved_preds, resolved_pred, -3)):
            if output is None or prediction is None:
                continue
            output.select(sample_dim if sample_dim >= 0 else output.ndim + sample_dim, i).copy_(prediction)
        del plddt_pred, pae_pred, pde_pred, resolved_pred
        _sync()
        log(f"confidence sample {i+1}/{N_sample} {time.time()-t0:.1f}s {gpu_mem_str()} rss={host_rss_gib()[0]:.1f}GiB pair_logits={'cpu' if offload_pair_outputs else 'device'}")
    zt.free()
    return plddt_preds, pae_preds, pde_preds, resolved_preds                # [.., N_sample, N_atom, plddt_bins], [.., N_sample, N, N, pae_bins], ...


@torch.no_grad()
def off_distogram_logits(distogram_head, hp: HostPair, rows=None):
    """== DistogramHead.forward(z) (head.py): logits = linear(z); logits = logits + logits.transpose(-2, -3)  -> [N, N, no_bins] on GPU."""
    rows = rows or CFG["rows"]
    n = hp.n
    logits = None
    for r0, r1 in blocks(n, rows):
        l = distogram_head.linear(upstream_stage_cast(hp.rows(r0, r1)))      # the stage runs autocast-disabled with fp32 arguments (opendde.py:1490): the host rows likewise
        if logits is None:
            logits = torch.empty((n, n, l.shape[-1]), dtype=l.dtype, device=l.device)
        logits[r0:r1] = l; del l
    logits = logits + logits.transpose(-2, -3)
    return logits


@torch.no_grad()
def off_contact_probs(distogram_head, hp: HostPair, min_bin, max_bin, no_bins, thres=8.0, rows=None):
    """== sample_confidence.compute_contact_prob(DistogramHead(z), ...) -> [N, N].  OPT['disto']=='rows': per row block
    logits_rows = linear(z[I,:]) + linear(z[:,I])^T (never an [N,N,no_bins] tensor); 'full': stock-shaped logits on the GPU."""
    from opendde.model import sample_confidence as SC
    rows = rows or CFG["rows"]
    n, dev = hp.n, hp.device
    OO.count("disto")                                                      # RAN-OR-REFUSE: the distogram half of pair_offload_conf engaged
    if OPT["disto"] == "full":
        return SC.compute_contact_prob(distogram_logits=off_distogram_logits(distogram_head, hp, rows), min_bin=min_bin, max_bin=max_bin, no_bins=no_bins, thres=thres)
    contact = None
    for r0, r1 in blocks(n, rows):
        lg = distogram_head.linear(upstream_stage_cast(hp.rows(r0, r1)))     # the stage runs autocast-disabled with fp32 arguments (opendde.py:1490): the host rows likewise
        lg = lg + distogram_head.linear(upstream_stage_cast(hp.rows_T(r0, r1)))            # rows r0:r1 of logits + logits.transpose(-2, -3)
        c = SC.compute_contact_prob(distogram_logits=lg, min_bin=min_bin, max_bin=max_bin, no_bins=no_bins, thres=thres)
        if contact is None:
            contact = torch.empty((n, n), dtype=c.dtype, device=c.device)
        contact[r0:r1] = c; del lg, c
    return contact


# ==========================================================================================================================
# D. install
# ==========================================================================================================================
_TRUNK_INSTALLED = False


def install_trunk(stages):
    """Patch OpenDDE 1.1.1: 'trunk' -> get_pairformer_output host-resident; 'conf' (or a HostPair pair_z produced by 'trunk') ->
    confidence head + distogram contact probabilities host-resident.  OpenDDE.forward / _forward_impl / _main_inference_loop stay
    upstream's: the TF32 scope, the always-lazy relp and the N^2-bounded chunk sizes reach the offloaded stages through their
    arguments."""
    global _TRUNK_INSTALLED
    if _TRUNK_INSTALLED:
        return
    from opendde.model import opendde as M
    OpenDDE = M.OpenDDE
    stages = set(stages)
    log(f"install_trunk v{__version__}: stages={sorted(stages)} opt={OPT}")

    if "trunk" in stages:
        def get_pairformer_output(self, input_feature_dict, N_cycle, inplace_safe=False, chunk_size=None):
            return off_get_pairformer_output(self, input_feature_dict, N_cycle, inplace_safe=inplace_safe, chunk_size=chunk_size)
        OpenDDE.get_pairformer_output = get_pairformer_output

        if "struct" not in stages:
            # trunk-only offload: the stock structural expander / diffusion need a GPU tensor -> materialise (must fit).
            def _materialise(z):
                t = getattr(z, "_materialised", None)
                if t is None:
                    OO.fallback("struct stage not offloaded: residue pair tensor materialised on the GPU"); log("WARNING: 'struct' stage not offloaded -> materialising the residue pair tensor on the GPU for the stock downstream stages")
                    t = z.to_tensor(); z._materialised = t
                return t
            stock_expand = OpenDDE.expand_to_structural_tokens
            def expand_to_structural_tokens(self, input_feature_dict, s_inputs, s, z, inplace_safe=False, chunk_size=None, lazy_relp=False):
                if isinstance(z, HostPair):
                    z = _materialise(z)
                return stock_expand(self, input_feature_dict, s_inputs, s, z, inplace_safe=inplace_safe, chunk_size=chunk_size, lazy_relp=lazy_relp)
            OpenDDE.expand_to_structural_tokens = expand_to_structural_tokens
            stock_select = OpenDDE.select_pair_output_branch
            def select_pair_output_branch(self, *a, **kw):
                fd, si, s, z = stock_select(self, *a, **kw)
                if isinstance(z, HostPair) and "conf" not in stages:
                    z = _materialise(z)
                return fd, si, s, z
            OpenDDE.select_pair_output_branch = select_pair_output_branch

    # confidence head: HostPair input -> always offloaded; tensor input -> offloaded iff 'conf' in stages
    stock_run_conf = OpenDDE.run_confidence_head
    def run_confidence_head(self, *args, **kwargs):
        z_trunk = kwargs.get("z_trunk")
        if isinstance(z_trunk, HostPair) or ("conf" in stages and torch.is_tensor(z_trunk)):
            kw = dict(kwargs)
            log(f"confidence head offload: z_trunk {'HostPair' if isinstance(z_trunk, HostPair) else 'tensor->HostPair'} n={z_trunk.n if isinstance(z_trunk, HostPair) else z_trunk.shape[-2]}")
            # _main_inference_loop reads pair_z in the distogram stage BEFORE this stage and only `del`s it afterwards (opendde.py
            # L2025-2055) -> the trunk HostPair may be consumed by the last sample
            kw["consume_z"] = isinstance(z_trunk, HostPair)
            from opendde.utils.torch_utils import autocasting_disable_decorator      # same AMP policy as the stock run_confidence_head
            fn = lambda *a, **k: off_confidence_head(self.confidence_head, *a, **k)
            return autocasting_disable_decorator(self.configs.skip_amp.confidence_head)(fn)(*args, **kw)
        return stock_run_conf(self, *args, **kwargs)
    OpenDDE.run_confidence_head = run_confidence_head

    stock_cdcp = OpenDDE.compute_distogram_contact_probs
    def compute_distogram_contact_probs(self, pair_z, pair_z_spec=None):
        if isinstance(pair_z, HostPair):
            from opendde.model import sample_confidence as SC
            t0 = time.time()
            out = off_contact_probs(self.distogram_head, pair_z, **SC.get_bin_params(self.configs.confidence.distogram))
            _sync(); log(f"distogram contact probs (host pair, mode={OPT['disto']}) n={pair_z.n} {time.time()-t0:.1f}s")
            return out
        return stock_cdcp(self, pair_z, pair_z_spec=pair_z_spec)
    OpenDDE.compute_distogram_contact_probs = compute_distogram_contact_probs
    _TRUNK_INSTALLED = True


# consumers of the residue-level pair_z in OpenDDE._main_inference_loop (opendde.py L1838-2071) and what happens with a HostPair:
#   select_pair_output_branch            -> pass-through (no tensor ops)                                   [ok as is]
#   expand_to_structural_tokens(z=..., lazy_relp=True)
#                                        -> odde_offload 'struct' stage (off_structural_expander accepts HostPair z_res) or the
#                                           materialising guard above when 'struct' is not offloaded      [patched]
#   prepare_diffusion_cache_for_sampling / run_sample_diffusion_stage(z=structural z)
#                                        -> odde_offload 'struct' stage                                    [odde_offload]
#   run_distogram_contact_stage(pair_z)  -> compute_distogram_contact_probs -> off_contact_probs          [patched]
#   run_confidence_head_stage(z_trunk)   -> run_confidence_head -> off_confidence_head                   [patched]
#   `del pair_z` after the confidence stage; run_post_confidence_outputs_stage takes no pair tensor      [ok as is]
