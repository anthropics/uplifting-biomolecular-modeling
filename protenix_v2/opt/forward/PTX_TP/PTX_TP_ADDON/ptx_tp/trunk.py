# Portions derived from Protenix v2.0.0 (https://github.com/bytedance/Protenix), Copyright 2024 ByteDance and/or its affiliates,
# used under the Apache License, Version 2.0.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ptx_tp.trunk -- TP re-orchestration of the STOCK Protenix-v2 inference forward (protenix 2.0.0 model/protenix.py 170-304 and 470-655).

The pair representation is ROW-SHARDED end-to-end: rank r holds z[r0:r1, :, :] ([R_r, N, c_z], bf16 under autocast) on the P-invariant
Layout grid (ptx_tp.dist.Layout, B=128). Every N^2 statement of the stock forward is routed through the FIXED seam names:
    ptx_tp.<impl>.pairformer.tp_pairformer_stack     msa.tp_msa_module          diffusion.tp_prepare_cache / tp_sample_diffusion
    ptx_tp.<impl>.confidence.tp_confidence_head      summary.tp_full_data_and_summary
where <impl> is the seam's module (ptx_tp.impl(seam): ptx_tp.<seam>) and the ROW-LOCAL trunk statements are written here (z_init rows,
recycling rows, distogram/contact rows).
Autocast policy: the diffusion seams are called INSIDE autocasting_disable_decorator(skip_amp.sample_diffusion) and the confidence seam inside
autocasting_disable_decorator(skip_amp.confidence_head), exactly where stock applies them (tensor args arrive cast as stock's callee sees them).
RNG: every rank executes the same RNG-consuming statements as single-card stock (MSA sampling inside MSAModule, mc-dropout on the full
tensor, diffusion noise), from RNG states synchronised to rank 0 by runner_hooks -> replicated tensors are identical on all ranks
(asserted under the deterministic recipe, PTX_DET=1)."""
from __future__ import annotations

import os
import re
import time
from typing import Any, Optional

import torch
import torch.nn.functional as F

from protenix.model import sample_confidence
from protenix.utils.torch_utils import autocasting_disable_decorator

import ptx_tp
from ptx_tp import dist as D
from ptx_tp.mirror import PhaseLog

_PL: Optional[PhaseLog] = None


def phase(name, **kw):
    global _PL
    if _PL is None:
        _PL = PhaseLog.from_env()
    return _PL.phase(name, **kw)


def dbg_checksum(t, name):
    if ptx_tp.det_recipe() and torch.is_tensor(t):
        cs = D.allreduce_checksum(t, name)
        ptx_tp.log(f"checksum {name} {tuple(t.shape)} {t.dtype} = {cs}")


def triatt_chunk_override(chunk_size, N_token):
    """PTX_TP_TRIATT_CHUNK=<int> replaces stock's dynamic tri-attention query chunk (32 above 2,560 tokens ...)
    for EVERY pair stack the TP trunk drives (trunk pairformer, MSA pair stack, template stack, confidence pairformer) -- it is the `chunk_size`
    argument threaded through tp_pairformer_stack / tp_triatt. Unset = stock dynamic chunk."""
    v = os.environ.get("PTX_TP_TRIATT_CHUNK", "").strip()
    if not v:
        return chunk_size
    new = int(v)
    if int(os.environ.get("RANK", "0")) == 0:
        ptx_tp.log(f"PTX_TP_TRIATT_CHUNK: tri-attention chunk_size {chunk_size} -> {new} (N_token={N_token})")
    try:
        ptx_tp.runmeta_update(triatt_chunk=new)
    except Exception:
        pass
    return new


def debug_block_subops(blk, s_in, z_in, layout, kw):
    """On the first bad block: run the stock sub-ops one at a time on full tensors vs the TP sub-ops on shards (rank-0 print; debugging aid)."""
    try:
        from ptx_tp import trimul as TM, triatt as TA, rowlocal as RL
    except Exception as e:
        print(f"[PF-COMPARE] subops unavailable: {e!r}", flush=True)
        return
    try:
        with torch.no_grad():
            z = z_in.clone()
            # stock statements of PairformerBlock.forward (inference, inplace_safe path uses the same math): tri_mul_out -> tri_mul_in -> tri_att_start -> tri_att_end -> transition
            ref = {}
            zz = z.clone()
            zz = zz + blk.dropout_row(blk.tri_mul_out(zz, mask=None, inplace_safe=False, _add_with_inplace=False)) if hasattr(blk, "dropout_row") else zz + blk.tri_mul_out(zz, mask=None, inplace_safe=False)
            ref["tri_mul_out"] = zz.clone()
            sh = D.Layout(layout.N, layout.P, layout.rank)
            zs = z[layout.r0:layout.r1].clone().contiguous()
            out = TM.tp_trimul(blk.tri_mul_out, zs, sh, outgoing=True, with_add=True)
            full = D.all_gather_rows(out, sh)
            print(f"[PF-COMPARE]   subop tri_mul_out(+add): bitwise={torch.equal(full, ref['tri_mul_out'])} max|d|={(full.float()-ref['tri_mul_out'].float()).abs().max().item():.3e}", flush=True)
    except Exception as e:
        print(f"[PF-COMPARE]   subop compare failed: {e!r}", flush=True)


def mem_line(tag: str, layout=None) -> None:
    """EVERY rank prints its own torch allocated / reserved / max_allocated GiB (NO collective: safe to call from rank-asymmetric code paths).
    Deliberately no collective: this is called from rank-asymmetric sites, where an all_gather would deadlock."""
    if not torch.cuda.is_available():
        return
    r = layout.rank if layout is not None else int(os.environ.get("RANK", "0"))
    a, rsv, mx = torch.cuda.memory_allocated() / 2**30, torch.cuda.memory_reserved() / 2**30, torch.cuda.max_memory_allocated() / 2**30
    if r == 0 or os.environ.get("PTX_TP_MEM_ALL_RANKS", "1") == "1":
        print(f"[ptx_tp mem r{r}] {tag}: allocated {a:.2f} GiB | reserved {rsv:.2f} | max_allocated {mx:.2f}", flush=True)
    try:
        if r == 0:
            ptx_tp.runmeta_update(**{f"mem_{re.sub(r'[^a-zA-Z0-9]+', '_', tag)[:40]}_GiB_r0": round(a, 2)})
    except Exception:
        pass
def zinit_rows_block(model, a_rows: torch.Tensor, b_all: torch.Tensor, feats: dict, rpe, c0: int, c1: int, zc_rows_blk=None) -> torch.Tensor:
    """z_init rows [c0:c1] (GLOBAL row indices): zinit1(s_init)[c0:c1,None,:] + zinit2(s_init)[None,:,:] + relp rows + token_bond rows (+ constraint rows).
    Same statements and order as zinit_rows / stock, restricted to one row block (every op is row-local => bitwise identical)."""
    zb = a_rows[c0:c1, None, :] + b_all[None, :, :]
    zb += rpe.linear_no_bias(relp_block(feats, rpe, c0, c1))
    zb += model.linear_no_bias_token_bond(feats["token_bonds"][c0:c1].to(zb.device, non_blocking=True).unsqueeze(dim=-1))   # token_bonds may live on the host
    if zc_rows_blk is not None:
        zb += zc_rows_blk
    return zb


def recycle_inplace_blocks(model, z: torch.Tensor, z_init, s_init: torch.Tensor, feats: dict, layout: D.Layout, *, mc_dropout: bool, cycle_no: int,
                           zc_rows=None, row_block: int = 128) -> torch.Tensor:
    """Cycle-start statement z = z_init + [dropout](linear_no_bias_z_cycle(layernorm_z_cycle(z))) executed PER 128-ROW BLOCK, IN PLACE into the z shard
    (peak transient = a few [row_block, N, c_z] blocks instead of 2-3 whole shards). z_init rows come from the stored shard (z_init tensor) or are
    recomputed per block (z_init is None; PTX_TP_ZINIT_RECOMPUTE). MC-dropout: draw-ON at N <= PTX_TP_DROPOUT_FULL_MAX keeps the exact full-tensor
    Philox statement (gather; small N only); above it the P-invariant block-seeded mask is generated per block (a different mask than stock's by construction; logged)."""
    N, p = layout.N, float(model.configs.mc_dropout_rate)
    R = z.shape[0]
    full_mask = mc_dropout and N <= int(os.environ.get("PTX_TP_DROPOUT_FULL_MAX", "6000"))
    if full_mask:                                   # exact stock statement (small N): needs the whole projection at once
        u = recycle_rows(model, z)
        u = mc_dropout_rows(model, u, layout, cycle_no)
    rpe = model.relative_position_encoding
    a_rows = b_all = None
    if z_init is None:
        a_rows = model.linear_no_bias_zinit1(s_init)          # [N, c_z] (tiny)
        b_all = model.linear_no_bias_zinit2(s_init)
    if mc_dropout and not full_mask and layout.rank == 0 and cycle_no == 0:
        ptx_tp.log(f"TIER-2 NOTICE: mc_dropout at N={N} uses the P-invariant block-seeded mask (not torch's full-tensor Philox mask)")
    base = int(torch.initial_seed()) % (2 ** 62)
    for i0 in range(0, R, row_block):
        i1 = min(R, i0 + row_block)
        c0, c1 = layout.r0 + i0, layout.r0 + i1
        if full_mask:
            ub = u[i0:i1]
        else:
            ub = model.linear_no_bias_z_cycle(model.layernorm_z_cycle(z[i0:i1]))
            if mc_dropout:
                g = torch.Generator(device=z.device)
                g.manual_seed(base + 1_000_003 * (cycle_no + 1) + (c0 // 128))
                keep = torch.rand((i1 - i0,) + tuple(ub.shape[1:]), generator=g, device=z.device, dtype=torch.float32) >= p
                ub = (ub.float() * keep * (1.0 / (1.0 - p))).to(ub.dtype)
        if z_init is None:
            zb = zinit_rows_block(model, a_rows, b_all, feats, rpe, c0, c1, None if zc_rows is None else zc_rows[i0:i1])
        else:
            zb = z_init[i0:i1]
        z[i0:i1] = zb + ub
        del ub, zb
    if full_mask:
        del u
    return z


_MSA_S_LOG = []


def install_msa_sample_logger():
    """Log the MSA sample size S per MSAModule call (stock samples S rows without replacement per cycle; RNG-consuming)."""
    import protenix.model.modules.pairformer as PFM
    if getattr(PFM, "_ptx_tp_s_logger", False):
        return
    orig = PFM.sample_msa_feature_dict_random_without_replacement

    def logged(feat_dict, dim_dict, *a, **k):
        out = orig(feat_dict, dim_dict, *a, **k)
        try:
            S = int(out["msa"].shape[-2 if out["msa"].dim() >= 2 else 0]) if "msa" in out else -1
            n_in = int(feat_dict["msa"].shape[0]) if "msa" in feat_dict else -1
        except Exception:
            S, n_in = -2, -2
        _MSA_S_LOG.append(S)
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[ptx_tp msa] sample call #{len(_MSA_S_LOG)}: N_msa_in={n_in} -> S={S} (cutoff={k.get('cutoff', a[0] if a else '?')})", flush=True)
        return out

    PFM.sample_msa_feature_dict_random_without_replacement = logged
    PFM._ptx_tp_s_logger = True


def _release_cached(layout, tag: str) -> None:
    """At N >= PTX_TP_GC_ABOVE_N: collect dead reference cycles + return cached allocator blocks; then print the per-rank memory line."""
    if torch.cuda.is_available() and layout is not None and layout.N >= int(os.environ.get("PTX_TP_GC_ABOVE_N", "8000")):
        import gc
        gc.collect()
        try:
            from ptx_tp.pairformer import release_device_caches      # optional: the pairformer seam's device caches
            release_device_caches()
        except Exception:
            pass
        torch.cuda.empty_cache()
    mem_line(tag, layout)


def _templ_park_active(z, layout) -> bool:
    v = os.environ.get("PTX_TP_TEMPL_PARK_Z", "auto").strip().lower()
    if v in ("0", "off", "no"):
        return False
    if v in ("1", "on", "host", "yes"):
        return True
    gb = z.numel() * z.element_size() / 1e9 if torch.is_tensor(z) else 0.0
    return gb > float(os.environ.get("PTX_TP_TEMPL_PARK_ABOVE_GB", "8"))


class _env_swap:
    def __init__(self, **kv):
        self.kv = {k: v for k, v in kv.items() if v is not None}
        self.old = {}

    def __enter__(self):
        for k, v in self.kv.items():
            self.old[k] = os.environ.get(k)
            os.environ[k] = str(v)

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _template_forward_blocked_ln(H2T, te, tid, feats, w, zdt, layout, kw):
    """ptx_tp.template's _tp_single_template_forward with the final `v = te.layernorm_v(v)` applied PER 128-ROW BLOCK in place
    (hazard: F.layer_norm on >= 2^31 elements returns wrong tail rows, and a whole [R, N, 64] shard exceeds that at large N).
    Per-row arithmetic identical."""
    R, N, r0 = layout.R, layout.N, layout.r0
    from ptx_tp.msa import tp_pair_stack_block as _psb
    v = None
    for i0 in range(0, max(R, 1), 128):
        i1 = min(i0 + 128, R)
        if R == 0:
            i0 = i1 = 0
        at = H2T._template_pair_features_rows(te, tid, feats, r0 + i0, r0 + i1, zdt, N)
        blk = w[i0:i1] + te.linear_no_bias_a(at)
        if v is None:
            v = blk.new_empty((R,) + tuple(blk.shape[1:]))
        v[i0:i1] = blk
        del at, blk
        if R == 0:
            break
    for blk_mod in te.pairformer_stack.blocks:
        v = _psb(blk_mod, v, layout, triangle_multiplicative=kw.get("triangle_multiplicative"), triangle_attention=kw.get("triangle_attention"),
                 inplace_safe=kw.get("inplace_safe", True), chunk_size=kw.get("chunk_size"))
    for i0 in range(0, R, 128):
        i1 = min(i0 + 128, R)
        v[i0:i1] = te.layernorm_v(v[i0:i1])
    return v


def template_update_parked(model, feats, z, layout, kw, msa_mod):
    """PTX_TP_TEMPL_PARK_Z=auto|1|0: the TemplateEmbedder update with the z shard PARKED ON HOST while the c=64 template pair stacks run.
    Same statements as ptx_tp.template's tp_template_embedder: w = linear_z(LN(z rows)) per 128-row block; per template v-stack via
    _template_forward_blocked_ln; mean; then z rows are restored from host and `z[b] = z_host[b] + linear_u(relu(u[b]))` per 128-row block
    (tp_template_embedder applies linear_u on the whole [R,N,64] shard; the per-block form is the same bf16 GEMM class it uses for linear_z per block).
    Peak: w + u + v (3 x [R,N,64]) + the c=64 pair-stack transients; the 256-wide shard is on the host meanwhile (pageable; PTX_TP_TEMPL_PARK_PIN=1 pins).
    c=64 TriMul tiles: PTX_TP_TRIMUL_ROWS_A_C64 (default 512 at N >= 15,000) is swapped into PTX_TP_TRIMUL_ROWS_A around the stack calls."""
    from ptx_tp import template as H2T
    te = model.template_embedder
    if hasattr(H2T, "template_embedder_is_active") and not H2T.template_embedder_is_active(te, feats):
        return z
    z = ensure_owned(z, "template park")
    R, N = layout.R, layout.N
    dev, zdt = z.device, z.dtype
    nbytes = z.numel() * z.element_size()
    t0 = time.time()
    blocks = [(i0, min(i0 + 128, R)) for i0 in range(0, R, 128)] or [(0, 0)]
    w = torch.cat([te.linear_no_bias_z(te.layernorm_z(z[i0:i1])) for (i0, i1) in blocks], dim=0)
    host = torch.empty(z.shape, dtype=zdt, device="cpu", pin_memory=os.environ.get("PTX_TP_TEMPL_PARK_PIN", "0") == "1")
    for (i0, i1) in blocks:
        host[i0:i1].copy_(z[i0:i1])
    torch.cuda.synchronize()
    z.untyped_storage().resize_(0)
    torch.cuda.empty_cache()
    mem_line(f"template: z parked on host in {time.time() - t0:.1f}s (w built)", layout)
    T = int(feats["template_aatype"].shape[0])
    rows_c64 = os.environ.get("PTX_TP_TRIMUL_ROWS_A_C64", "512" if N >= 15000 else None)
    u = None
    with _env_swap(PTX_TP_TRIMUL_ROWS_A=rows_c64):
        for tid in range(T):
            v = _template_forward_blocked_ln(H2T, te, tid, feats, w, zdt, layout, kw)
            if u is None:
                u = v
            else:
                u += v           # in place (same elementwise bf16 add as `u + v`; saves one [R,N,64] transient)
                del v
            mem_line(f"template {tid + 1}/{T} done", layout)
    del w
    u = u / (1e-7 + T)
    mem_line("template: stacks done (u64 resident, z still parked)", layout)
    t1 = time.time()
    z.untyped_storage().resize_(nbytes)
    for (i0, i1) in blocks:
        zb = host[i0:i1].to(dev, non_blocking=False)
        z[i0:i1] = zb + te.linear_no_bias_u(te.relu(u[i0:i1]))
        del zb
    del host, u
    torch.cuda.synchronize()
    mem_line(f"template: z restored + update added in {time.time() - t1:.1f}s", layout)
    return z


def ensure_owned(z: torch.Tensor, where: str) -> torch.Tensor:
    """The pairformer seam's end-of-stack storage release needs the z shard to OWN its storage (contiguous, offset 0, storage bytes == tensor bytes);
    otherwise re-materialise it once (logged)."""
    if not torch.is_tensor(z):
        return z
    ok = z.is_contiguous() and z.storage_offset() == 0 and z.untyped_storage().nbytes() == z.numel() * z.element_size()
    if ok:
        return z
    ptx_tp.log(f"shard-ownership: z at '{where}' is a view (offset={z.storage_offset()}, storage={z.untyped_storage().nbytes()/2**30:.2f} GiB vs "
               f"tensor={z.numel()*z.element_size()/2**30:.2f} GiB) -> cloning into an owning buffer")
    out = torch.empty_like(z, memory_format=torch.contiguous_format)
    out.copy_(z)
    return out


def make_layout(N: int) -> D.Layout:
    P, rank = D.world()
    B = int(os.environ.get("PTX_TP_B", "128"))
    L = D.Layout(N, P, rank, B=B)
    if not L.replicated:
        empties = [r for r in range(P) if D.Layout(N, P, r, B=B).R == 0]
        if empties and rank == 0:
            ptx_tp.log(f"WARNING layout B={B}: ranks {empties} own ZERO rows at N={N}, P={P} (trunk is R==0-safe; diffusion/confidence tails may not be) "
                       f"-- launch via ptx_tp.launch (auto-B) or set PTX_TP_B to one of 64/32/16")
        if B != 128 and rank == 0:
            ptx_tp.log(f"layout block B={B} (P-invariant grid)")
            try:
                ptx_tp.runmeta_update(layout_B=B)
            except Exception:
                pass
    return L


# ----------------------------------------------------------------------------------------------------------------- row-local statements
def relp_rows(rpe, feats: dict, r0: int, r1: int) -> torch.Tensor:
    """Rows [r0,r1) of RelativePositionEncoding.generate_relp's dense one-hot 'relp' [r1-r0, N, 4*r_max+2*s_max+7] fp32, computed from the
    per-token index features only (same formula as embedders.py 148-201; never materialises [N,N,*])."""
    r_max, s_max = rpe.r_max, rpe.s_max
    asym_id, residue_index, entity_id, token_index, sym_id = (feats["asym_id"], feats["residue_index"], feats["entity_id"], feats["token_index"], feats["sym_id"])
    ai, ri, ei, ti, si = asym_id[r0:r1, None], residue_index[r0:r1, None], entity_id[r0:r1, None], token_index[r0:r1, None], sym_id[r0:r1, None]
    b_same_chain = (ai == asym_id[None, :]).long()
    b_same_residue = (ri == residue_index[None, :]).long()
    b_same_entity = (ei == entity_id[None, :]).long()
    d_residue = torch.clip(input=ri - residue_index[None, :] + r_max, min=0, max=2 * r_max) * b_same_chain + (1 - b_same_chain) * (2 * r_max + 1)
    a_rel_pos = F.one_hot(d_residue, 2 * (r_max + 1))
    d_token = torch.clip(input=ti - token_index[None, :] + r_max, min=0, max=2 * r_max) * b_same_chain * b_same_residue + \
        (1 - b_same_chain * b_same_residue) * (2 * r_max + 1)
    a_rel_token = F.one_hot(d_token, 2 * (r_max + 1))
    d_chain = torch.clip(input=si - sym_id[None, :] + s_max, min=0, max=2 * s_max) * b_same_entity + (1 - b_same_entity) * (2 * s_max + 1)
    a_rel_chain = F.one_hot(d_chain, 2 * (s_max + 1))
    return torch.cat([a_rel_pos, a_rel_token, b_same_entity[..., None], a_rel_chain], dim=-1).float()


class TPLazyRelp:
    """Fallback stand-in for input_feature_dict['relp'] (same .rows(i0, i1) API as the single-card lazy-relp lever's LazyRelp; global row indices): holds the five [N]
    index vectors; rows are rebuilt with the stock generate_relp statements restricted to rows i0:i1 (integer ops + one_hot: exact per element)."""

    def __init__(self, d: dict, r_max: int, s_max: int):
        self.d = {k: d[k] for k in ("asym_id", "residue_index", "entity_id", "token_index", "sym_id")}
        self.r_max, self.s_max = r_max, s_max
        n = self.d["asym_id"].shape[-1]
        self.shape = tuple(self.d["asym_id"].shape[:-1]) + (n, n, 4 * r_max + 2 * s_max + 7)
        self.device, self.dtype = self.d["asym_id"].device, torch.float32

    def dim(self):
        return len(self.shape)

    def size(self, i=None):
        return self.shape if i is None else self.shape[i]

    def rows(self, i0: int, i1: int) -> torch.Tensor:
        class _R:  # tiny shim so relp_rows can read r_max/s_max
            pass
        r = _R(); r.r_max, r.s_max = self.r_max, self.s_max
        with torch.no_grad():
            return relp_rows(r, self.d, i0, i1)

    def full(self) -> torch.Tensor:
        return self.rows(0, self.shape[-2])

    def to(self, *a, **k):
        return self

    def untyped_storage(self):      # lets PTX_XL_FREE=relp style hooks call .untyped_storage().resize_(0) harmlessly
        class _S:
            def resize_(self, n): return self
            def size(self): return 0
            def nbytes(self): return 0
        return _S()

    def __repr__(self):
        return f"TPLazyRelp(shape={self.shape})"


def relp_block(feats: dict, rpe, c0: int, c1: int) -> torch.Tensor:
    """relp rows c0:c1 (GLOBAL) from whatever input_feature_dict['relp'] is: dense stock tensor (slice), a LazyRelp / TPLazyRelp (.rows), or absent (generate)."""
    relp = feats.get("relp")
    if relp is None:
        return relp_rows(rpe, feats, c0, c1)
    if hasattr(relp, "rows"):
        return relp.rows(c0, c1)
    return relp[..., c0:c1, :, :]


def mc_dropout_rows(model, u_rows: torch.Tensor, layout: D.Layout, cycle_no: int) -> torch.Tensor:
    """Stock applies F.dropout(p=mc_dropout_rate) to the FULL [N,N,c_z] recycling projection for a data-dependent ~40 % of (item, seed) pairs
    (random.random() < mc_dropout_apply_rate, drawn once per predict). torch's Philox dropout mask depends on the tensor's numel/launch geometry, so a
    shard-local F.dropout can never reproduce it. Two modes:
      N <= PTX_TP_DROPOUT_FULL_MAX (default 6000): gather u, F.dropout on the full tensor (the stock statement; same RNG consumption on every rank), re-slice
                                                   => EXACT vs stock; costs one transient [N,N,c_z].
      else: P-invariant block-seeded mask (private torch.Generator per 128-row global block seeded from (initial_seed, cycle, block)); the global RNG
            stream is untouched so ranks stay in sync => a different mask than single-card stock BY CONSTRUCTION (logged)."""
    N, p = layout.N, float(model.configs.mc_dropout_rate)
    if N <= int(os.environ.get("PTX_TP_DROPOUT_FULL_MAX", "6000")):
        full = D.all_gather_rows(u_rows.contiguous(), layout)
        full = F.dropout(full, p=p)
        out = full[layout.r0:layout.r1].contiguous()
        del full
        return out
    if layout.rank == 0:
        ptx_tp.log(f"TIER-2 NOTICE: mc_dropout at N={N} uses the P-invariant block-seeded mask (not torch's full-tensor Philox mask)")
    out = torch.empty_like(u_rows)
    base = int(torch.initial_seed()) % (2 ** 62)
    for c0, c1 in layout.chunks(128):
        g = torch.Generator(device=u_rows.device)
        g.manual_seed(base + 1_000_003 * (cycle_no + 1) + (c0 // 128))
        keep = torch.rand((c1 - c0,) + tuple(u_rows.shape[1:]), generator=g, device=u_rows.device, dtype=torch.float32) >= p
        out[c0 - layout.r0:c1 - layout.r0] = (u_rows[c0 - layout.r0:c1 - layout.r0].float() * keep * (1.0 / (1.0 - p))).to(u_rows.dtype)
    return out


def zinit_rows(model, s_init: torch.Tensor, feats: dict, layout: D.Layout, z_constraint_rows=None, row_block: int = 128) -> torch.Tensor:
    """z_init[r0:r1] = zinit1(s_init)[I,None,:] + zinit2(s_init)[None,:,:] + relpos_enc(relp rows I) + token_bond rows (+ constraint rows).
    Same statement order as stock (inplace_safe branch, protenix.py 209-219); relp rows generated per row_block to bound the fp32 one-hot transient."""
    r0, r1 = layout.r0, layout.r1
    z = model.linear_no_bias_zinit1(s_init)[r0:r1, None, :] + model.linear_no_bias_zinit2(s_init)[None, :, :]
    rpe = model.relative_position_encoding
    for c0 in range(r0, r1, row_block):                       # relp rows per block: the fp32 one-hot transient is [row_block, N, 139] at most
        c1 = min(r1, c0 + row_block)
        z[c0 - r0:c1 - r0] += rpe.linear_no_bias(relp_block(feats, rpe, c0, c1))
    z += model.linear_no_bias_token_bond(feats["token_bonds"][r0:r1].to(z.device, non_blocking=True).unsqueeze(dim=-1))
    if z_constraint_rows is not None:
        z += z_constraint_rows
    return z


def recycle_rows(model, z_sh: torch.Tensor) -> torch.Tensor:
    """linear_no_bias_z_cycle(layernorm_z_cycle(z rows)) -- row-local (LN over c_z, Linear over c_z)."""
    return model.linear_no_bias_z_cycle(model.layernorm_z_cycle(z_sh))


def contact_probs_rows(model, z_sh: torch.Tensor, layout: D.Layout) -> torch.Tensor:
    """Rows of compute_contact_prob(distogram_head(z)): logits rows = linear(z rows) + transpose_shards(linear(z rows)) (DistogramHead symmetrises),
    softmax/expectation in fp32 as stock (autocast disabled). Returns [R, N] fp32."""
    lz = model.distogram_head.linear(z_sh)                    # [R, N, no_bins] under autocast (as stock)
    lzT = D.transpose_shards(lz.contiguous(), layout)
    logits = lz + lzT
    return autocasting_disable_decorator(True)(sample_confidence.compute_contact_prob)(
        distogram_logits=logits, **sample_confidence.get_bin_params(model.configs.loss.distogram))


def _rows(z, layout):
    return z[layout.r0:layout.r1].contiguous()


# ----------------------------------------------------------------------------------------------------------------- trunk
def get_pairformer_output_tp(model, input_feature_dict: dict, N_cycle: int, inplace_safe: bool = True, chunk_size: Optional[int] = None,
                             mc_dropout: bool = False, layout: Optional[D.Layout] = None, tri_chunk="same"):
    """TP version of Protenix.get_pairformer_output (inference). Returns (s_inputs, s, z_shard)."""
    try:                                                     # real template rows hook; idempotent, import-cycle free here
        from ptx_tp.template_real import install_rows_hook, REAL_MARKER
        install_rows_hook(log=lambda msg: ptx_tp.log(msg))
    except Exception as _e:
        if "template_real_rows" in input_feature_dict:
            raise RuntimeError(f"real template features present but the rows hook failed to install: {_e!r}")
        ptx_tp.log(f"template_real rows hook not installed: {_e!r}")
    N = input_feature_dict["residue_index"].shape[-1]
    layout = layout or make_layout(N)
    feats = input_feature_dict
    phase("trunk_start", N=N, P=layout.P, R=layout.R, trunk_impl="tp", seams=ptx_tp.ledger(), reset_peak=True)

    s_inputs = model.input_embedder(feats, inplace_safe=False, chunk_size=chunk_size)          # replicated (per-token statement)
    dbg_checksum(s_inputs, "s_inputs")
    if "constraint_feature" not in feats and "constraint_feature_dropped" in feats:
        ce = model.constraint_embedder
        enabled = [n for n in ("pocket_embedder_config", "contact_embedder_config", "contact_atom_embedder_config", "substructure_embedder_config")
                   if getattr(ce, n, {}).get("enable", False)]
        if enabled:
            raise RuntimeError(f"constraint features were dropped (PTX_TP_DROP_CONSTRAINT=1) but this model ENABLES {enabled}: rerun with PTX_TP_DROP_CONSTRAINT=0")
    z_constraint = model.constraint_embedder(feats["constraint_feature"]) if "constraint_feature" in feats else None
    s_init = model.linear_no_bias_sinit(s_inputs)
    zc_rows = None if z_constraint is None else _rows(z_constraint, layout)
    shard_gb = layout.R * N * model.c_z * 2 / 1e9
    zr_env = os.environ.get("PTX_TP_ZINIT_RECOMPUTE", "auto").strip().lower()
    recompute = zr_env == "1" or (zr_env == "auto" and shard_gb > float(os.environ.get("PTX_TP_ZINIT_RECOMPUTE_ABOVE_GB", "8")))
    if layout.rank == 0:
        ptx_tp.log(f"trunk: z shard {layout.R}x{N}x{model.c_z} bf16 = {shard_gb:.1f} GB/rank; z_init {'RECOMPUTED per cycle per 128-row block' if recompute else 'stored (one extra shard)'}; cycle-start statement runs per 128-row block in place")

    def build_zinit():
        return zinit_rows(model, s_init, feats, layout, zc_rows)

    z_init = None if recompute else build_zinit()
    z = torch.zeros((layout.R, N, model.c_z), dtype=(z_init.dtype if z_init is not None else s_init.dtype), device=s_init.device) \
        if z_init is None else torch.zeros_like(z_init)
    s = torch.zeros_like(s_init)
    mem_line("after z / z_init shard allocation", layout)
    msa_mod = ptx_tp.impl("msa")
    pf_mod = ptx_tp.impl("pairformer")
    kw = dict(triangle_multiplicative=model.configs.triangle_multiplicative, triangle_attention=model.configs.triangle_attention,
              inplace_safe=inplace_safe, chunk_size=(chunk_size if tri_chunk == "same" else tri_chunk))
    for cycle_no in range(N_cycle):
        t0 = time.time()
        z = recycle_inplace_blocks(model, z, z_init, s_init, feats, layout, mc_dropout=mc_dropout, cycle_no=cycle_no, zc_rows=zc_rows)
        if cycle_no == 0:
            mem_line(f"after cycle-start statement (cycle {cycle_no})", layout)
        if model.template_embedder.n_blocks > 0:          # protenix-v2 runs its TemplateEmbedder every cycle, also on the featurizer's dummy templates
            mem_line(f"cycle {cycle_no}: before template", layout)
            if _templ_park_active(z, layout):
                z = template_update_parked(model, feats, z, layout, kw, msa_mod)
            else:
                u_t = msa_mod.tp_template_embedder(model.template_embedder, feats, z, layout, **kw)   # TemplateEmbedder.forward on the row shard
                if torch.is_tensor(u_t) or u_t != 0:
                    z += u_t
                del u_t
            _release_cached(layout, f"cycle {cycle_no}: after template (gc + empty_cache)")
        if os.environ.get("PTX_TP_LOG_MSA_S", "1") == "1":
            install_msa_sample_logger()
        with _env_swap(PTX_TP_TRIMUL_ROWS_A=os.environ.get("PTX_TP_TRIMUL_ROWS_A_MSA")):     # optional smaller TriMul a-tiles for the MSA block's pair stack
            z = msa_mod.tp_msa_module(model.msa_module, feats, z, s_inputs, layout, **kw)
        _release_cached(layout, f"cycle {cycle_no}: after msa module (gc + empty_cache)")
        z = ensure_owned(z, f"after msa module (cycle {cycle_no})")
        s = s_init + model.linear_no_bias_s(model.layernorm_s(s))
        s, z = pf_mod.tp_pairformer_stack(model.pairformer_stack, s, z, layout, **kw)
        dbg_checksum(s, f"s_cycle{cycle_no}")
        phase("trunk_cycle_end", cycle=cycle_no, s=round(time.time() - t0, 2))
        mem_line(f"after trunk cycle {cycle_no} ({time.time() - t0:.0f}s)", layout)
    return s_inputs, s, z, layout


def main_inference_loop_tp(model, input_feature_dict: dict, N_cycle: int, mode: str = "inference", inplace_safe: bool = True,
                           chunk_size: Optional[int] = 4, mc_dropout: bool = False):
    """TP version of Protenix._main_inference_loop (inference / no labels). Returns (pred_dict, log_dict, time_tracker) like stock."""
    step_st = time.time()
    feats = input_feature_dict
    N_token = feats["residue_index"].shape[-1]
    if hasattr(model.configs.infer_setting, "dynamic_chunk_size") and model.configs.infer_setting.dynamic_chunk_size:
        chunk_size = model._get_dynamic_chunk_size(N_token)
    tri_chunk = triatt_chunk_override(chunk_size, N_token)     # pair stacks + confidence pairformer only; the input embedder keeps stock's chunk
    layout = make_layout(N_token)
    log_dict, pred_dict, time_tracker = {}, {}, {}
    N_sample = model.configs.sample_diffusion["N_sample"]
    N_step = model.configs.sample_diffusion["N_step"]
    skip_amp_diff = model.configs.skip_amp.sample_diffusion

    s_inputs, s, z, layout = get_pairformer_output_tp(model, feats, N_cycle=N_cycle, inplace_safe=inplace_safe, chunk_size=chunk_size,
                                                      mc_dropout=mc_dropout, layout=layout, tri_chunk=tri_chunk)
    for key in [k for k in feats.keys() if "template_" in k or k in ["msa", "has_deletion", "deletion_value", "profile", "deletion_mean"]]:
        del feats[key]
    if os.environ.get("PTX_TP_KEEP_TOKEN_BONDS", "0") != "1" and int(N_token) >= 8000 and "token_bonds" in feats:
        del feats["token_bonds"]                                  # [N,N] fp32 (3.9 GB @31k), only consumed by the trunk z_init rows
    step_trunk = time.time()
    time_tracker["pairformer"] = step_trunk - step_st
    mem_line("trunk_final (msa/template feats dropped)", layout)
    phase("trunk_end", s=round(step_trunk - step_st, 2))

    # ---- diffusion: pass skip_amp, the seam applies stock's autocast/cast policy itself.
    dif = ptx_tp.impl("diffusion")
    noise_schedule = model.inference_noise_scheduler(N_step=N_step, device=s_inputs.device, dtype=s_inputs.dtype)
    phase("diffusion_start", reset_peak=False)
    pair_z = dif.tp_prepare_cache(model.diffusion_module.diffusion_conditioning, feats, z, layout, skip_amp=skip_amp_diff, inplace_safe=inplace_safe) \
        if model.enable_diffusion_shared_vars_cache else None
    mem_line("after prepare_cache (pair_z shard built)", layout)
    coords = dif.tp_sample_diffusion(model, feats, s_inputs, s, None, pair_z, layout, N_sample, N_step,
                                     skip_amp=skip_amp_diff, inplace_safe=inplace_safe, noise_schedule=noise_schedule)
    del pair_z
    dbg_checksum(coords, "coordinate")
    pred_dict["coordinate"] = coords
    try:
        dif_free = getattr(dif, "free_caches", None)          # optional seam hook (per-block pair-bias cache etc.)
        if callable(dif_free):
            dif_free()
    except Exception as e:
        ptx_tp.log(f"diffusion free_caches failed: {e!r}")
    torch.cuda.empty_cache()
    mem_line("confidence entry (pair_z, sampler caches freed)", layout)
    step_diffusion = time.time()
    time_tracker["diffusion"] = step_diffusion - step_trunk
    phase("diffusion_end", s=round(step_diffusion - step_trunk, 2))

    # ---- distogram contact probabilities + confidence head + summaries: contact ROWS [R,N] fp32 from the z shard (never [N,N,64]
    #      logits at once); the PAE / PDE logits are reduced in row blocks (RowBlockReducer, finish 'exact'; the full_data O(N^2) members
    #      are not handed to the JSON dumper), the summary finishes on rank 0.
    conf_mod, summ = ptx_tp.impl("confidence"), ptx_tp.impl("summary")
    bins = sample_confidence.get_bin_params(model.configs.loss.distogram)
    phase("confidence_start")
    # NOT wrapped in autocasting_disable_decorator -- stock evaluates distogram_head(z) in the AMBIENT autocast (bf16) and only
    # compute_contact_prob runs amp-off on fp32-cast logits; the seam reproduces exactly that split internally. Wrapping the whole call
    # would run the distogram linear in fp32 on an fp32-cast z => contact_probs / gpde would differ from stock.
    contact_rows = conf_mod.tp_distogram_contact_rows(model.distogram_head, z, layout, **bins)
    reduce_mode = "block"
    ckw = dict(contact_rows=contact_rows, consume_z_trunk=os.environ.get("PTX_TP_CONF_CONSUME_Z", "1") == "1", row_chunk=int(os.environ.get("PTX_TP_CONF_ROW_CHUNK", "64")))
    head_out = autocasting_disable_decorator(model.configs.skip_amp.confidence_head)(conf_mod.tp_confidence_head)(
        model.confidence_head, feats, s_inputs, s, z, None, coords, layout,
        triangle_multiplicative=model.configs.triangle_multiplicative, triangle_attention=model.configs.triangle_attention,
        inplace_safe=inplace_safe, chunk_size=tri_chunk, **ckw)
    for k in ("plddt", "pae", "pde", "resolved"):
        if isinstance(head_out, dict) and torch.is_tensor(head_out.get(k)):
            pred_dict[k] = head_out[k]
    step_confidence = time.time()
    time_tracker["confidence"] = step_confidence - step_diffusion
    time_tracker["model_forward"] = time.time() - step_st
    pred_dict["summary_confidence"], pred_dict["full_data"] = summ.tp_full_data_and_summary(
        model.configs, head_out, layout=layout, token_asym_id=feats["asym_id"], token_has_frame=feats["has_frame"],
        atom_coordinate=coords, atom_to_token_idx=feats["atom_to_token_idx"], atom_is_polymer=1 - feats["is_ligand"],
        N_recycle=N_cycle, contact_rows=contact_rows, contact_probs_full=None, return_full_data=True,
        interested_atom_mask=None, mol_id=None, elements_one_hot=None)
    rec = phase("confidence_end", s=round(step_confidence - step_diffusion, 2), total_s=round(time.time() - step_st, 2), reduce_mode=reduce_mode)
    try:   # 'per-rank peak / single-card peak' line (single-card peaks may be supplied as PTX_TP_REF_PEAK_GIB="<N_token>:<GiB>,..."; else 'n/a')
        peaks = [None] * layout.P if D.is_dist() else [rec["cuda"].get("max_alloc_GiB")]
        if D.is_dist():
            torch.distributed.all_gather_object(peaks, rec["cuda"].get("max_alloc_GiB"))
        ref = dict(x.split(":") for x in os.environ.get("PTX_TP_REF_PEAK_GIB", "").split(",") if ":" in x)
        ref_peak = ref.get(str(int(N_token)))
        if layout.rank == 0:
            line = (f"per-rank peak alloc GiB = {peaks} (max {max(p for p in peaks if p is not None):.2f}) / single-card peak = {ref_peak or 'n/a'} GiB  "
                    f"[N_token={N_token}, P={layout.P}, seams: {ptx_tp.ledger()}; tiling: {os.environ.get('PTX_TP_TILING_DESC', 'module defaults')}]")
            print(line, flush=True)
            ptx_tp.runmeta_update(per_rank_peak_alloc_GiB=peaks, single_card_peak_GiB=ref_peak, tiling=os.environ.get("PTX_TP_TILING_DESC", "module defaults"), wall_model_forward_s=round(time.time() - step_st, 2),
                                  t_trunk_s=round(time_tracker["pairformer"], 2), t_diffusion_s=round(time_tracker["diffusion"], 2), t_confidence_s=round(time_tracker["confidence"], 2))
    except Exception as e:
        ptx_tp.log(f"peak report failed: {e!r}")
    if layout.rank == 0:
        try:
            ptx_tp.runmeta_update(confidence_reduce_mode=reduce_mode, seams_ledger=ptx_tp.ledger())
        except Exception:
            pass
    return pred_dict, log_dict, time_tracker
