# ef2_xl.py — ESMFOLD2-XL add-on: memory levers for long sequences / large complexes on one GPU.
# Applies at runtime to the Biohub `transformers` ESMFold2 implementation.
# No source file, weight, or fold-call argument is modified. Every lever is default OFF and enabled by apply(...) keywords. Levers:
#   X2  EF2_XL_LOOPFREE=1    _run_one_loop re-issued with `del` of dead pair temporaries before the trunk call; lm_z pre-cast to the
#                            trunk dtype once when per-loop LM dropout is off (arithmetic-free). The class method serves the STOCK loop; a
#                            kit that re-issues the loop on the instance (ef2_opt's loop_static "G4", whole-recycle graph "G5") owns that
#                            frame and performs the same release there itself (its counters loop_static_released / recycle_graph_loops are
#                            the memory line's evidence for those folds; loop_calls counts the folds this re-issue ran).
#   X2b EF2_XL_FREE=z,relpos,lmz  ConfidenceHead: release storage of z / relpos / token-bond encodings after their last consumer; del z_base;
#                            'lmz' releases the fp32 lm_z once the bf16 copy exists (X2 must be on).
#   X3  EF2_XL_COND=lean     DiffusionConditioning pair path: LN(cat[z,relpos]) built row-wise into one buffer + ONE z_proj GEMM
#                            (identical GEMM shape to stock) + row-chunked bf16 transitions. Two routes to the same statements: the class
#                            forward (the stock sampler chain) and the PAIR_PATH_HOOK attribute (`_ef2_pair_path`) a re-issued sampler step
#                            calls for its once-per-fold z branch (ef2_dit's fused step); cond_calls / cond_xl_calls count both, cond_hook_calls
#                            the hook route.
#   X4  EF2_XL_ESMC_OFFLOAD=1  ESMC-6B streamed through the LM pass from the host at large inputs: engine "pinned" = pinned masters +
#                            double-buffered prefetch on a copy stream (apply(esmc_stream="pinned"|"legacy") selects the engine; no environment switch).
#   OWN EF2_XL_OWN=1        pass pair-tensor ownership through FoldingTrunk (input freed after the first residual) - arithmetic-free
#   X2c (with OWN)           per-loop LM-dropout copy cast to the trunk dtype immediately (fp32 copy freed) - arithmetic-free
#   X6  EF2_XL_LMPAIR=1     LanguageModelShim SingleToPair MLP + LayerNorm row-chunked (bf16 GEMMs M-chunked)
#   X7  EF2_XL_RELPOS=1     relative-position pair encoding row-chunked (one-hot features per row block; bf16 GEMM M-chunked)
#   X10 EF2_XL_DISTOCPU=1   distogram logits moved to host right after the head (output only) - arithmetic-free
#   X8  EF2_XL_INITLEAN=1   _init_pair_state: torch trunc_normal_ rejection sampler re-issued with identical full-size RNG calls but chunked in-place
#                            mask/select (no full-size where/mask temporaries) - same values AND same generator state afterwards
#   X2f (with LOOPFREE)      previous-iteration msa_pair released before the MSA encoder/trunk run again (Full model) - arithmetic-free
#   X2d (with OWN)           loop inject statement row-chunked: LN -> cast -> F.linear(b_mat) into one buffer; dead temporaries dropped first
# Ledger: stderr line "[EF2XL] applied ..." at apply; "[EF2XL] stats {...}" at exit.
from __future__ import annotations
import os, sys, json, atexit, inspect, textwrap, hashlib, time
import torch
import torch.nn.functional as F
from torch import Tensor

__version__ = "1.5.0"
_STATS = {"loop_calls": 0, "cond_calls": 0, "cond_xl_calls": 0, "cond_hook_calls": 0, "free_bytes": 0, "free_events": 0,
          "esmc_offload_events": 0, "esmc_reload_events": 0, "lm_streamed_passes": 0, "lm_streamed_blocks": 0, "lm_stream_h2d_bytes": 0, "verify_checks": 0, "verify_mismatch": 0,
          "s2p_calls": 0, "s2p_xl_calls": 0, "relpos_calls": 0, "relpos_xl_calls": 0, "owned_blocks": 0, "disto_cpu_events": 0,
          "init_xl_calls": 0, "init_rounds": 0, "inject_xl_calls": 0}
_APPLIED = {}
_ORIG = {}
model_ref = {"m": None}
_CFG = {"rows_target": 262144, "probe": 0, "free": set(), "cond": "", "esmc_min_tok": 0, "loopfree": False,
        "own": False, "lmpair": False, "relpos": False, "initlean": False}


def _log(msg):
    print(f"[EF2XL] {msg}", file=sys.stderr, flush=True)


def _mods():
    import transformers.models.esmfold2.modeling_esmfold2_common as CMN
    import transformers.models.esmfold2.modeling_esmfold2 as MOD
    return CMN, MOD


def _rows_for(L: int) -> int:
    r = max(1, _CFG["rows_target"] // max(1, L))
    return max(16, min(L, r))


# --------------------------------------------------------------------------------------------------------------- source patcher
def _shift4(block: str) -> str:
    return "".join((l[4:] if l.startswith("    ") else l) for l in block.splitlines(keepends=True))


def _patch_method(owner, name, edits, tag, extra_globals=None):
    """Re-issue owner.name from its own source with anchor-checked insertions/replacements. Every anchor must occur exactly once."""
    fn = getattr(owner, name)
    src = textwrap.dedent(inspect.getsource(fn))
    sha = hashlib.sha256(src.encode()).hexdigest()[:12]
    for anchor, replacement in edits:
        # edits are written with in-class indentation (method body at 8 spaces); the dedented source has the body at 4.
        anchor = _shift4(anchor); replacement = _shift4(replacement)
        n = src.count(anchor)
        if n != 1:
            raise RuntimeError(f"[EF2XL] patch {tag}: anchor occurs {n} times (need 1): {anchor[:70]!r}; source sha {sha}")
        src = src.replace(anchor, replacement)
    mod = sys.modules[owner.__module__]
    g = dict(mod.__dict__)
    g.update({"_EF2XL_STATS": _STATS, "_ef2xl_free": _free_storage, "_EF2XL_CFG": _CFG, "torch": torch, "F": F,
              "_ef2xl_inject_linear": _inject_linear_xl})
    if extra_globals:
        g.update(extra_globals)
    loc = {}
    exec(compile(src, f"<ef2xl:{tag}:{sha}>", "exec"), g, loc)
    new_fn = loc[name]
    _ORIG[f"{tag}"] = fn
    setattr(owner, name, new_fn)
    return sha


def _free_storage(t, label):
    if t is None or not isinstance(t, torch.Tensor):
        return
    try:
        nb = t.untyped_storage().nbytes()
        if nb > 0:
            t.untyped_storage().resize_(0)
            _STATS["free_bytes"] += nb
            _STATS["free_events"] += 1
    except Exception as e:  # pragma: no cover
        print(f"[EF2XL] free({label}) failed: {e}", file=sys.stderr)


# X2: _run_one_loop with dels (statement order unchanged)
_LOOP_EDITS = [
    ("        for _ in range(total_steps):\n",
     "        _EF2XL_STATS['loop_calls'] += 1\n"
     "        if _EF2XL_CFG['loopfree'] and (not _per_loop_lm_dropout) and lm_z is not None and self.lm_encoder is not None and lm_z.dtype != z_init.dtype:\n"
     "            _lmz32 = lm_z\n"
     "            lm_z = lm_z.to(z_init.dtype)\n"
     "            if 'lmz' in _EF2XL_CFG['free']:\n"
     "                _ef2xl_free(_lmz32, 'lm_z_fp32')\n"
     "            del _lmz32\n"
     "        for _ef2xl_step in range(total_steps):\n"),
    ("            z = self.folding_trunk(z, pair_attention_mask=pair_mask)\n",
     "            if _EF2XL_CFG['loopfree']:\n"
     "                del lm_z_i, refined_lm_z, z_inject_pair, injected_pair\n"
     "            if _EF2XL_CFG['own']:\n"
     "                _zh = [z]\n"
     "                del z\n"
     "                z = self.folding_trunk(_zh, pair_attention_mask=pair_mask)\n"
     "                del _zh\n"
     "            else:\n"
     "                z = self.folding_trunk(z, pair_attention_mask=pair_mask)\n"),
    ("            injected_pair = self.parcae_input_norm(z_inject_pair)\n"
     "            z = a * z + F.linear(injected_pair.to(z.dtype), b_mat)\n",
     "            if _EF2XL_CFG['own']:\n"
     "                refined_lm_z = None\n"
     "                _lin = _ef2xl_inject_linear(self.parcae_input_norm, z_inject_pair, b_mat, z.dtype)\n"
     "                injected_pair = None\n"
     "                if z_inject_pair is not z_init:\n"
     "                    z_inject_pair = None\n"
     "                z = a * z + _lin\n"
     "                del _lin\n"
     "            else:\n"
     "                injected_pair = self.parcae_input_norm(z_inject_pair)\n"
     "                z = a * z + F.linear(injected_pair.to(z.dtype), b_mat)\n"),
    ("                z_inject_pair = (\n"
     "                    msa_pair\n"
     "                    if self.config.msa_encoder_overwrite\n"
     "                    else (z_inject_pair + msa_pair)\n"
     "                )\n",
     "                z_inject_pair = (\n"
     "                    msa_pair\n"
     "                    if self.config.msa_encoder_overwrite\n"
     "                    else (z_inject_pair + msa_pair)\n"
     "                )\n"
     "                if _EF2XL_CFG['loopfree']:\n"
     "                    msa_pair = None\n"),
    ("                refined_lm_z = self.lm_encoder(\n"
     "                    lm_z_i.to(z_init.dtype), pair_attention_mask=pair_mask\n"
     "                )\n",
     "                if _EF2XL_CFG['own']:\n"
     "                    _lh = [lm_z_i.to(z_init.dtype)]\n"
     "                    if lm_z_i is not lm_z:\n"
     "                        lm_z_i = None\n"
     "                    refined_lm_z = self.lm_encoder(_lh, pair_attention_mask=pair_mask)\n"
     "                    del _lh\n"
     "                else:\n"
     "                    refined_lm_z = self.lm_encoder(\n"
     "                        lm_z_i.to(z_init.dtype), pair_attention_mask=pair_mask\n"
     "                    )\n"),
]


# OWN: FoldingTrunk.forward accepts a 1-element list (takes ownership) and runs blocks so the block input is freed at the first residual
_FT_EDITS = [
    ("        orig_dtype = pair.dtype\n",
     "        if isinstance(pair, list):\n"
     "            _h0 = pair\n"
     "            pair = _h0.pop()\n"
     "            del _h0\n"
     "        orig_dtype = pair.dtype\n"),
    ("                pair = fn(pair)\n",
     "                if _EF2XL_CFG['own']:\n"
     "                    _h = [pair]\n"
     "                    del pair\n"
     "                    pair = _ef2xl_owned_block(block, _h, pair_attention_mask)\n"
     "                    del _h\n"
     "                else:\n"
     "                    pair = fn(pair)\n"),
]


def _owned_block(block, holder, pair_attention_mask):
    """PairUpdateBlock.forward statements, with the input taken from `holder` so the pre-residual pair is freed as soon as it is dead."""
    _STATS["owned_blocks"] += 1
    pair = holder.pop()
    if block._can_use_fused_trimul_with_residual(pair):
        pair = block._fused_trimul_with_residual(pair, "outgoing", pair_attention_mask)
        pair = block._fused_trimul_with_residual(pair, "incoming", pair_attention_mask)
    else:
        pair = block.row_drop(pair, block.tri_mul_out(pair, mask=pair_attention_mask))
        pair = block.row_drop(pair, block.tri_mul_in(pair, mask=pair_attention_mask))
    pair = block.pair_transition(pair)
    return pair


# X2b: ConfidenceHead.forward frees
_CONF_EDITS = [
    ("        z_base = self.z_norm(z)\n",
     "        z_base = self.z_norm(z)\n"
     "        if 'z' in _EF2XL_CFG['free']:\n"
     "            _ef2xl_free(z, 'z')\n"),
    ("            z_base = z_base + relative_position_encoding\n",
     "            z_base = z_base + relative_position_encoding\n"
     "            if 'relpos' in _EF2XL_CFG['free']:\n"
     "                _ef2xl_free(relative_position_encoding, 'relpos')\n"),
    ("            z_base = z_base + token_bonds_encoding\n",
     "            z_base = z_base + token_bonds_encoding\n"
     "            if 'relpos' in _EF2XL_CFG['free']:\n"
     "                _ef2xl_free(token_bonds_encoding, 'bonds')\n"),
    ("        pair = pair + self.dist_bin_pairwise_embed(distogram_bins)\n",
     "        pair = pair + self.dist_bin_pairwise_embed(distogram_bins)\n"
     "        del z_base\n"),
]

# X3: DiffusionConditioning pair path
_COND_EDITS = [
    ("            z_rel = relative_position_encoding.to(dtype=torch.float32)\n"
     "            z = torch.cat([z_trunk.to(dtype=torch.float32), z_rel], dim=-1)\n"
     "            z = self.z_proj(self.z_input_norm(z))\n"
     "            with torch.autocast(device_type=\"cuda\", dtype=torch.bfloat16):\n"
     "                for block in self.z_transitions:\n"
     "                    z = z + block(z)\n",
     "            z = _ef2xl_cond_pair(self, z_trunk, relative_position_encoding)\n"),
]


def _cond_pair_xl(self, z_trunk: Tensor, relative_position_encoding: Tensor) -> Tensor:
    _STATS["cond_calls"] += 1
    mode = _CFG["cond"]
    B, L, L2, cz = z_trunk.shape
    if (not z_trunk.is_cuda) or B != 1 or mode != "lean":
        z_rel = relative_position_encoding.to(dtype=torch.float32)
        z = torch.cat([z_trunk.to(dtype=torch.float32), z_rel], dim=-1)
        z = self.z_proj(self.z_input_norm(z))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for block in self.z_transitions:
                z = z + block(z)
        return z
    _STATS["cond_xl_calls"] += 1
    R = _rows_for(L)
    dev = z_trunk.device
    # LN(cat) row-wise into ONE fp32 buffer, then a single z_proj GEMM with the stock shape (exact by construction).
    zin = torch.empty((B, L, L2, 2 * cz), dtype=torch.float32, device=dev)
    for s in range(0, L, R):
        e = min(s + R, L)
        zr = torch.cat([z_trunk[:, s:e].to(dtype=torch.float32), relative_position_encoding[:, s:e].to(dtype=torch.float32)], dim=-1)
        zin[:, s:e] = self.z_input_norm(zr)
        del zr
    z = self.z_proj(zin)
    del zin
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for block in self.z_transitions:
            znew = torch.empty_like(z)
            for s in range(0, L, R):
                e = min(s + R, L)
                zs = z[:, s:e]
                znew[:, s:e] = zs + block(zs)
            z = znew
            del znew
    return z


PAIR_PATH_HOOK = "_ef2_pair_path"        # the attribute X3 sets on DiffusionConditioning for a re-issued sampler step's once-per-fold z branch (ef2_dit reads
                                         # the same name): a plain function on the class -> a bound method `cd._ef2_pair_path(z_trunk, relative_position_encoding)`


def _cond_pair_hook(self, z_trunk: Tensor, relative_position_encoding: Tensor) -> Tensor:
    """X3 by the hook route: the same statements as the class forward's pair path (_cond_pair_xl), counted as a hook call."""
    _STATS["cond_hook_calls"] += 1
    return _cond_pair_xl(self, z_trunk, relative_position_encoding)


# X6: LanguageModelShim SingleToPair + LayerNorm row-chunked
_LMS_EDITS = [("        lm_z = self.base_z_mlp(lm_z)  # [B, L, L, d_z]\n", "        lm_z = _ef2xl_s2p(self.base_z_mlp, lm_z)\n")]


def _s2p_xl(mlp, x):
    _STATS["s2p_calls"] += 1
    if (not _CFG["lmpair"]) or (not x.is_cuda) or x.dim() != 3:
        return mlp(x)
    _STATS["s2p_xl_calls"] += 1
    s2p, ln = mlp[0], mlp[1]
    xd = s2p.downproject(x)
    B, L, dz = xd.shape
    R = _rows_for(L); out = None
    for s in range(0, L, R):
        e = min(s + R, L)
        xr = xd[:, s:e]
        cat = torch.cat([(xr.unsqueeze(2) * xd.unsqueeze(1)), (xr.unsqueeze(2) - xd.unsqueeze(1))], dim=3)
        y = ln(s2p.output_mlp(cat))
        if out is None:
            out = torch.empty((B, L, L, y.shape[-1]), dtype=y.dtype, device=y.device)
        out[:, s:e] = y
        del cat, y, xr
    return out


# X7: relative-position pair encoding row-chunked (statements of ResIdxAsymIdSymIdEntityIdEncoding.forward with the row side sliced)
def _relpos_xl(self, residue_index, asym_id, sym_id, entity_id, token_index):
    _STATS["relpos_calls"] += 1
    if (not _CFG["relpos"]) or (not residue_index.is_cuda):
        return _ORIG["relpos_forward"](self, residue_index, asym_id, sym_id, entity_id, token_index)
    _STATS["relpos_xl_calls"] += 1
    B, L = residue_index.shape
    R = _rows_for(L); out = None
    nr, nc = self.n_relative_residx_bins, self.n_relative_chain_bins
    for s in range(0, L, R):
        e = min(s + R, L)
        bij_same_chain = asym_id[:, s:e].unsqueeze(2) == asym_id.unsqueeze(1)
        bij_same_residue = residue_index[:, s:e].unsqueeze(2) == residue_index.unsqueeze(1)
        bij_same_entity = entity_id[:, s:e].unsqueeze(2) == entity_id.unsqueeze(1)
        dij_residue = residue_index[:, s:e].unsqueeze(2) - residue_index.unsqueeze(1)
        dij_residue = torch.clip(dij_residue + nr, 0, 2 * nr)
        dij_residue = torch.where(bij_same_chain, dij_residue, 2 * nr + 1)
        aij_rel_pos = F.one_hot(dij_residue, 2 * nr + 2)
        dij_token = torch.clip(token_index[:, s:e].unsqueeze(2) - token_index.unsqueeze(1) + nr, 0, 2 * nr)
        dij_token = torch.where(bij_same_chain & bij_same_residue, dij_token, 2 * nr + 1)
        aij_rel_token = F.one_hot(dij_token, 2 * nr + 2)
        dij_chain = torch.clip(sym_id[:, s:e].unsqueeze(2) - sym_id.unsqueeze(1) + nc, 0, 2 * nc)
        dij_chain = torch.where(bij_same_chain, 2 * nc + 1, dij_chain)
        aij_rel_chain = F.one_hot(dij_chain, 2 * nc + 2)
        feats = torch.cat([aij_rel_pos.float(), aij_rel_token.float(), bij_same_entity.float().unsqueeze(-1), aij_rel_chain.float()], dim=-1)
        y = self.embed(feats)
        if out is None:
            out = torch.empty((B, L, L, y.shape[-1]), dtype=y.dtype, device=y.device)
        out[:, s:e] = y
        del feats, y, aij_rel_pos, aij_rel_token, aij_rel_chain, dij_residue, dij_token, dij_chain, bij_same_chain, bij_same_residue, bij_same_entity
    return out


# X8: exact, memory-lean re-issue of torch.nn.init.trunc_normal_ (torch 2.13 rejection branch, p > 0.3) for _init_pair_state
def _trunc_normal_lean_(tensor, mean, std, a, b):
    import math as _m
    def norm_cdf(x): return (1.0 + _m.erf(x / _m.sqrt(2.0))) / 2.0
    p = norm_cdf((b - mean) / std) - norm_cdf((a - mean) / std)
    if p <= 0.3 or tensor.dim() < 2:
        return torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)
    lo = tensor.new_tensor(a, device="cpu").item(); hi = tensor.new_tensor(b, device="cpu").item()
    with torch.no_grad():
        result = tensor.normal_(mean, std)                       # same first RNG call as stock (in place)
        L = result.shape[1]; R = _rows_for(L)
        while True:
            anybad = False
            for s_ in range(0, L, R):                             # stock: mask = (result < lo) | (result > hi); mask.any()
                blk = result[:, s_:s_ + R]
                if bool(((blk < lo) | (blk > hi)).any()):
                    anybad = True; break
            if not anybad:
                break
            _STATS["init_rounds"] += 1
            cand = torch.empty_like(result).normal_(mean, std)   # same full-size RNG call as stock -> same stream, same generator state
            for s_ in range(0, L, R):                             # stock: result = torch.where(mask, cand, result)  (elementwise select)
                blk = result[:, s_:s_ + R]; cb = cand[:, s_:s_ + R]
                m = (blk < lo) | (blk > hi)
                blk.copy_(torch.where(m, cb, blk))
                del m, cb
            del cand
    return tensor


def _init_pair_state_xl(self, ref):
    if (not _CFG["initlean"]) or (not ref.is_cuda):
        return _ORIG["init_pair_state"](self, ref)
    import math as _m
    _STATS["init_xl_calls"] += 1
    std = _m.sqrt(2.0 / (5.0 * ref.shape[-1]))
    state = torch.empty_like(ref, dtype=torch.float32)
    _trunc_normal_lean_(state, 0.0, std, -3 * std, 3 * std)
    return state.to(dtype=ref.dtype)


# X2d: loop inject statement, row-chunked: F.linear(parcae_input_norm(z_inject).to(dt), b_mat) written into one [B,L,L,d] buffer of dtype dt
def _inject_linear_xl(norm, z_inject_pair, b_mat, dt):
    _STATS["inject_xl_calls"] += 1
    B, L = z_inject_pair.shape[0], z_inject_pair.shape[1]
    R = _rows_for(L); out = None
    for s_ in range(0, L, R):
        y = F.linear(norm(z_inject_pair[:, s_:s_ + R]).to(dt), b_mat)
        if out is None:
            out = torch.empty((B, L) + tuple(y.shape[2:]), dtype=y.dtype, device=y.device)
        out[:, s_:s_ + R] = y
        del y
    return out


# X10: distogram logits to host (output only; decode() calls .cpu() on it anyway)
def _install_disto_cpu(model):
    if getattr(model, "_ef2xl_disto_hook", None) is not None:
        return
    def hook(mod, args, out):
        if isinstance(out, torch.Tensor) and out.is_cuda and out.dim() == 4:
            _STATS["disto_cpu_events"] += 1
            host_t = torch.empty(out.shape, dtype=out.dtype, device="cpu", pin_memory=True)
            host_t.copy_(out); torch.cuda.synchronize()
            return host_t
        return out
    model._ef2xl_disto_hook = model.distogram_head.register_forward_hook(hook)


# --------------------------------------------------------------------------------------------------------------- X4 ESMC offload
def _install_esmc_offload_legacy(model, min_tok: int):
    """X4 engine "legacy": for an LM pass of at least ``min_tok`` tokens the language model's transformer blocks are STREAMED through the pass from the host — each
    block's weights are moved to the device right before its forward and back to the host right after (forward hooks), so one block is
    resident at a time instead of the whole 6B model; the token embedding and the final norm stay on the device. Below ``min_tok`` (and for
    an LM without a block list) the LM runs resident, as loaded, and is moved back whole if an earlier large input had sent its blocks to
    the host. Same weights, same kernels, same call: only where the weights rest changes. One ``esmc_offload_events`` per streamed pass
    (the gate's account of x4 by size), ``lm_streamed_blocks`` per block moved in."""
    MODcls = type(model)
    if "esmc_clhs" in _ORIG:
        return
    orig = MODcls._compute_lm_hidden_states
    _ORIG["esmc_clhs"] = orig

    def _blocks(lm):
        bl = getattr(getattr(lm, "transformer", None), "blocks", None)
        return bl if isinstance(bl, torch.nn.ModuleList) and len(bl) > 0 else None

    def _hook(lm, dev):
        if getattr(lm, "_ef2xl_stream_hooks", None):
            return
        def pre(m, args):                                                    # host -> device, ordered on the current stream before the block's kernels
            with torch.inference_mode(mode=False), torch.no_grad():
                m.to(dev)
            _STATS["lm_streamed_blocks"] += 1
        def post(m, args, out):                                              # device -> host once the block's forward has been issued
            with torch.inference_mode(mode=False), torch.no_grad():
                m.to("cpu")
        lm._ef2xl_stream_hooks = [h for blk in _blocks(lm) for h in (blk.register_forward_pre_hook(pre), blk.register_forward_hook(post))]

    def _unhook(lm):
        for h in getattr(lm, "_ef2xl_stream_hooks", None) or []:
            h.remove()
        lm._ef2xl_stream_hooks = None

    def clhs(self, input_ids, *a, **k):
        lm = self._esmc
        if lm is None:
            return orig(self, input_ids, *a, **k)
        dev = input_ids.device
        ntok = int(input_ids.shape[-1]) if hasattr(input_ids, "shape") else 0
        blocks = _blocks(lm)
        where = getattr(lm, "_ef2xl_where", "cuda")
        if ntok < min_tok or blocks is None or dev.type != "cuda":           # below x4's size: the resident LM, as loaded
            if where != "cuda":
                t0 = time.time(); _unhook(lm); lm.to(dev); torch.cuda.synchronize(); lm._ef2xl_where = "cuda"
                _STATS["esmc_reload_events"] += 1
                _log(f"LM -> cuda ({time.time()-t0:.1f}s)")
            return orig(self, input_ids, *a, **k)
        if where == "cuda":                                                   # the first large input: the block weights move to the host once
            t0 = time.time(); blocks.to("cpu"); torch.cuda.synchronize(); torch.cuda.empty_cache(); lm._ef2xl_where = "host"
            _log(f"LM blocks -> cpu ({ntok} tok, {time.time()-t0:.1f}s)")
        _hook(lm, dev)
        t0 = time.time()
        out = orig(self, input_ids, *a, **k)                                  # the pass: every block streamed in and out by the hooks
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        _STATS["esmc_offload_events"] += 1; _STATS["lm_streamed_passes"] += 1
        _log(f"LM blocks streamed through the pass ({ntok} tok, {len(blocks)} blocks, {time.time()-t0:.1f}s)")
        return out
    MODcls._compute_lm_hidden_states = clhs




# X4 engine "pinned": the blocks rest in PINNED host memory and are streamed with a double-buffered prefetch on a side copy stream.
X4_ENGINES = ("pinned", "legacy")
_X4_STATE = {"streamer": None, "engine": None}


def _x4_engine(requested=None) -> str:
    """The x4 engine in force: the ``apply(esmc_stream=...)`` keyword (an ablation handle for the kit's own tests; no environment switch),
    default "pinned" — pinned masters + prefetch is how x4 works."""
    v = "pinned" if requested in (None, "") else str(requested).strip().lower()
    if v not in X4_ENGINES:
        raise ValueError(f"x4 engine {v!r}: expected one of {X4_ENGINES}")
    return v


class _PinnedBlockStreamer:
    """X4 engine "pinned": the LM's transformer blocks rest in page-locked host memory — ONE flat pinned buffer per block, every parameter /
    buffer of the block a view into it at a 256-byte-aligned offset (strides preserved) — and are streamed through the pass with a
    double-buffered prefetch: while block k computes on the current stream, block k+1's flat buffer is copied host->device by ONE
    cudaMemcpyAsync on a side copy stream into the other of two resident device slots (per block layout); the compute stream waits on the
    block's `ready` event before its first kernel, the copy stream waits on the `done` event of the slot's previous occupant before it
    overwrites the slot. No host synchronisation inside the pass, no device->host traffic (inference: the weights never change, the pinned
    copies are the masters for the life of the process). Same weights, same kernels, same stream order => the LM's arithmetic is untouched.
    Memory: pinned host = the blocks' weights once (ESMC-6B bf16: ~11.4 GiB); device = 2 slots of one block (~2 x 145 MiB) instead of the
    whole model."""
    ALIGN = 256
    NSLOT = 2

    def __init__(self, lm, blocks, dev):
        self.lm, self.blocks, self.dev = lm, blocks, torch.device(dev)
        self.n = len(blocks)
        self.entries = [None] * self.n        # per block: [(tensor, offset, nbytes, shape, stride)]
        self.host = [None] * self.n           # per block: flat pinned uint8 master
        self.sig = [None] * self.n            # per block: layout signature
        self.slots = {}                       # signature -> [flat device uint8] * NSLOT
        self.slot_rr = {}                     # signature -> round-robin counter (this pass)
        self.slot_done = {}                   # (signature, slot) -> event: the previous occupant's forward has been issued past
        self.ready = [None] * self.n          # per block (this pass): (event, slot index)
        self.cur = {}                         # block -> slot index in use (this pass)
        self.copy_stream = None
        self.hooks = None
        self.where = "cuda"                   # where the block weights rest: "cuda" (as loaded / reloaded) | "host" (pinned masters)
        self.pinned_bytes = 0
        self.t_pin_s = None

    # -- layout ---------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _leaves(block):
        seen, out = set(), []
        for mod in block.modules():
            for name, t in list(mod._parameters.items()) + list(mod._buffers.items()):
                if t is None or id(t) in seen:
                    continue
                seen.add(id(t)); out.append(t)
        return out

    @staticmethod
    def _dense(t):
        try:
            return t.is_contiguous() or t.contiguous().numel() * t.element_size() == t.untyped_storage().nbytes()
        except Exception:
            return t.is_contiguous()

    def _view(self, flat, o, nb, dtype, shape, stride):
        v = flat[o:o + nb].view(dtype)
        return v.as_strided(shape, stride) if len(shape) else v.reshape(())

    def pin(self):
        """Once per process (first input at/above the threshold): lay every block out in one pinned host buffer, copy the weights device->host
        into it, and re-point the block's tensors at the pinned views (the device copies go back to the allocator)."""
        t0 = time.time()
        with torch.inference_mode(mode=False), torch.no_grad():
            for bi, blk in enumerate(self.blocks):
                lay, off = [], 0
                for t in self._leaves(blk):
                    nb = t.numel() * t.element_size()
                    stride = tuple(t.stride()) if (t.is_contiguous() or self._dense(t)) else None
                    lay.append([t, off, nb, tuple(t.shape), stride])
                    off = -(-(off + max(nb, 1)) // self.ALIGN) * self.ALIGN
                host = torch.empty(max(off, self.ALIGN), dtype=torch.uint8, pin_memory=True)
                for e in lay:
                    t, o, nb, shape, stride = e
                    if stride is None:                                   # a non-dense view as a parameter: keep its values, contiguous layout
                        stride = tuple(t.contiguous().stride()); e[4] = stride
                    hv = self._view(host, o, nb, t.dtype, shape, stride)
                    hv.copy_(t.data)
                    t.data = hv
                self.entries[bi] = [tuple(e) for e in lay]; self.host[bi] = host
                self.sig[bi] = tuple((nb, str(t.dtype), shape, stride) for (t, o, nb, shape, stride) in self.entries[bi])
                self.pinned_bytes += host.numel()
            torch.cuda.synchronize(); torch.cuda.empty_cache()
        self.where = "host"; self.t_pin_s = time.time() - t0
        _STATS["lm_pinned_gib"] = round(self.pinned_bytes / 2 ** 30, 2)

    def to_host_views(self, which=None):
        """Point the blocks' tensors at their pinned masters (host); device slot views / reloaded device copies are dropped."""
        with torch.inference_mode(mode=False), torch.no_grad():
            for bi in (range(self.n) if which is None else which):
                for (t, o, nb, shape, stride) in self.entries[bi]:
                    if t.data.device.type != "cpu" or t.data.data_ptr() != self.host[bi].data_ptr() + o:
                        t.data = self._view(self.host[bi], o, nb, t.dtype, shape, stride)
        self.where = "host"

    def reload_to_device(self):
        """A later input below the threshold: the resident LM again — every block copied from its pinned master to its own device tensors
        (the masters stay pinned for the next large input: no second device->host pass, ever)."""
        self.unhook(); self.to_host_views()
        with torch.inference_mode(mode=False), torch.no_grad():
            self.blocks.to(self.dev, non_blocking=True)
            torch.cuda.synchronize()
        self.where = "cuda"

    def offload(self):
        """A large input after a reload: drop the device copies, rest on the pinned masters again (no copy needed)."""
        self.to_host_views()
        torch.cuda.synchronize(); torch.cuda.empty_cache()

    # -- the pass ----------------------------------------------------------------------------------------------------------------------------
    def _slots_for(self, sig, nbytes):
        sl = self.slots.get(sig)
        if sl is None:
            with torch.inference_mode(mode=False), torch.no_grad():
                sl = [torch.empty(nbytes, dtype=torch.uint8, device=self.dev) for _ in range(self.NSLOT)]
            self.slots[sig] = sl
        return sl

    def begin_pass(self):
        if self.copy_stream is None:
            self.copy_stream = torch.cuda.Stream(device=self.dev)
        self.slot_rr.clear(); self.slot_done.clear(); self.cur.clear(); self.ready = [None] * self.n
        ev = torch.cuda.Event(); ev.record(torch.cuda.current_stream(self.dev))
        self.copy_stream.wait_event(ev)                                   # the slots' last readers (the previous pass) are behind this point
        self._issue(0)

    def _issue(self, k):
        if k >= self.n or self.ready[k] is not None:
            return
        sig, host = self.sig[k], self.host[k]
        slots = self._slots_for(sig, host.numel())
        si = self.slot_rr.get(sig, 0) % self.NSLOT; self.slot_rr[sig] = self.slot_rr.get(sig, 0) + 1
        prev = self.slot_done.get((sig, si))
        with torch.cuda.stream(self.copy_stream):
            if prev is not None:
                self.copy_stream.wait_event(prev)                         # the slot's previous occupant has been consumed
            slots[si].copy_(host, non_blocking=True)                      # ONE pinned host->device memcpy for the whole block
            ev = torch.cuda.Event(); ev.record(self.copy_stream)
        self.ready[k] = (ev, si)
        _STATS["lm_streamed_blocks"] += 1; _STATS["lm_stream_h2d_bytes"] = int(_STATS.get("lm_stream_h2d_bytes", 0)) + int(host.numel())

    def pre(self, k):
        self._issue(k)                                                    # normally already in flight (prefetched by pre(k-1))
        ev, si = self.ready[k]
        torch.cuda.current_stream(self.dev).wait_event(ev)
        flat = self.slots[self.sig[k]][si]
        with torch.inference_mode(mode=False), torch.no_grad():
            for (t, o, nb, shape, stride) in self.entries[k]:
                t.data = self._view(flat, o, nb, t.dtype, shape, stride)
        self.cur[k] = si
        self._issue(k + 1)                                                # prefetch: overlaps block k's compute (waits only on block k-1's done)

    def post(self, k):
        ev = torch.cuda.Event(); ev.record(torch.cuda.current_stream(self.dev))
        self.slot_done[(self.sig[k], self.cur[k])] = ev
        self.to_host_views((k,))

    def hook(self):
        if self.hooks:
            return
        hs = []
        for k, blk in enumerate(self.blocks):
            hs.append(blk.register_forward_pre_hook(lambda m, a, _k=k: self.pre(_k)))
            hs.append(blk.register_forward_hook(lambda m, a, o, _k=k: self.post(_k)))
        self.hooks = hs

    def unhook(self):
        for h in self.hooks or []:
            h.remove()
        self.hooks = None


def _install_esmc_offload(model, min_tok: int, engine: str = "pinned"):
    """X4: for an LM pass of at least ``min_tok`` tokens the language model's transformer blocks are STREAMED through the pass from the host;
    below it (and for an LM without a block list) the LM runs resident, as loaded, and is moved back whole if an earlier large input had sent
    its blocks to the host. ``engine``: "pinned" (pinned masters + double-buffered prefetch on a copy stream, _PinnedBlockStreamer) or
    "legacy" (pageable ``module.to()`` per block in forward hooks, weights copied back to the host after every block). Same weights, same
    kernels, same call under both: only where the weights rest and how they travel changes. If the environment refuses page-locked memory the
    pinned engine names it and streams from pageable memory (the legacy engine) — x4 stays engaged. Counters: one ``esmc_offload_events`` /
    ``lm_streamed_passes`` per streamed pass (the gate's account of x4 by size), ``lm_streamed_blocks`` per block moved in, ``lm_pinned_gib``,
    ``lm_stream_h2d_bytes``, ``lm_stream_last_s`` / ``lm_stream_last_gbps`` (pinned engine)."""
    engine = (engine or "pinned").strip().lower()
    if engine not in X4_ENGINES:
        raise ValueError(f"x4 engine {engine!r}: expected one of {X4_ENGINES}")
    _X4_STATE["engine"] = engine; _STATS["esmc_stream_engine"] = engine
    if engine == "legacy":
        return _install_esmc_offload_legacy(model, min_tok)
    MODcls = type(model)
    if "esmc_clhs" in _ORIG:
        return
    orig = MODcls._compute_lm_hidden_states
    _ORIG["esmc_clhs"] = orig

    def _blocks(lm):
        bl = getattr(getattr(lm, "transformer", None), "blocks", None)
        return bl if isinstance(bl, torch.nn.ModuleList) and len(bl) > 0 else None

    def clhs(self, input_ids, *a, **k):
        lm = self._esmc
        if lm is None:
            return orig(self, input_ids, *a, **k)
        dev = input_ids.device
        ntok = int(input_ids.shape[-1]) if hasattr(input_ids, "shape") else 0
        blocks = _blocks(lm)
        st = _X4_STATE["streamer"]
        if ntok < min_tok or blocks is None or dev.type != "cuda":           # below x4's size: the resident LM, as loaded
            if st is not None and st.where != "cuda":
                t0 = time.time(); st.reload_to_device(); _STATS["esmc_reload_events"] += 1
                _log(f"LM -> cuda from the pinned masters ({time.time()-t0:.1f}s)")
            return orig(self, input_ids, *a, **k)
        if st is None:                                                        # the first large input of the process: pin once
            st = _PinnedBlockStreamer(lm, blocks, dev)
            try:
                st.pin()
            except RuntimeError as e:                                         # page-locked memory refused by the environment (limits / no pinned pool):
                _log(f"x4: page-locked host memory refused ({e!r}) — the LM blocks stream from PAGEABLE host memory in this process "
                     f"(v1.3 engine: per-block module.to() copies, no prefetch); x4 engaged, environment case 'pageable'")
                _STATS["esmc_stream_engine"] = "pageable(pin_refused)"; _STATS["esmc_pin_refused"] = repr(e)[:200]
                MODcls._compute_lm_hidden_states = _ORIG.pop("esmc_clhs")
                _install_esmc_offload_legacy(model, min_tok)
                return MODcls._compute_lm_hidden_states(self, input_ids, *a, **k)
            _X4_STATE["streamer"] = st
            _log(f"LM blocks -> pinned host ({ntok} tok, {st.n} blocks, {st.pinned_bytes/2**30:.2f} GiB pinned in {st.t_pin_s:.1f}s)")
        elif st.where == "cuda":                                              # large again after a reload: rest on the masters, drop the device copies
            t0 = time.time(); st.offload(); _log(f"LM blocks -> host (pinned masters, {time.time()-t0:.2f}s)")
        st.hook()
        t0 = time.time(); h2d0 = int(_STATS.get("lm_stream_h2d_bytes", 0))
        try:
            st.begin_pass()
            out = orig(self, input_ids, *a, **k)                              # the pass: every block prefetched in by the hooks, none copied back
            torch.cuda.synchronize()
        finally:
            st.to_host_views()                                                # the blocks rest on their masters between passes, whatever happened
            torch.cuda.empty_cache()                                          # the pass's cached segments go back before the trunk allocates (reserved high-water)
        dt = time.time() - t0; gb = (int(_STATS.get("lm_stream_h2d_bytes", 0)) - h2d0) / 1e9
        _STATS["esmc_offload_events"] += 1; _STATS["lm_streamed_passes"] += 1
        _STATS["lm_stream_last_s"] = round(dt, 3); _STATS["lm_stream_last_gbps"] = round(gb / dt, 1) if dt > 0 else None
        _log(f"LM blocks streamed through the pass (pinned, prefetched: {ntok} tok, {st.n} blocks, {gb:.2f} GB host->device in {dt:.2f}s = {gb/max(dt,1e-9):.1f} GB/s)")
        return out
    MODcls._compute_lm_hidden_states = clhs


# --------------------------------------------------------------------------------------------------------------- apply / unapply
def apply(model=None, *, loopfree=False, free="", cond="", esmc_offload=False, esmc_min_tok=0,
          rows_target=262144, probe=0, own=False, lmpair=False, relpos=False, distocpu=False, initlean=False, esmc_stream=None):
    """Install levers (process-global class patches; esmc offload needs the model instance). Returns the ledger dict."""
    CMN, MOD = _mods()
    if model is not None: model_ref["m"] = model
    _CFG["rows_target"] = int(rows_target); _CFG["probe"] = int(probe)
    _CFG["free"] = set(x for x in str(free).replace("+", ",").split(",") if x)
    _CFG["cond"] = cond or ""
    _CFG["esmc_min_tok"] = int(esmc_min_tok); _CFG["loopfree"] = bool(loopfree)
    _CFG["own"] = bool(own); _CFG["lmpair"] = bool(lmpair); _CFG["relpos"] = bool(relpos); _CFG["initlean"] = bool(initlean)
    led = {"version": __version__, "loopfree": bool(loopfree), "free": sorted(_CFG["free"]),
           "cond": _CFG["cond"], "esmc_offload": bool(esmc_offload), "esmc_min_tok": int(esmc_min_tok), "esmc_stream": _x4_engine(esmc_stream),
           "rows_target": _CFG["rows_target"], "probe": _CFG["probe"], "own": bool(own), "lmpair": bool(lmpair),
           "relpos": bool(relpos), "distocpu": bool(distocpu), "initlean": bool(initlean), "patch_shas": {}}
    if (own or loopfree) and "ft" not in _ORIG and own:
        led["patch_shas"]["FoldingTrunk.forward"] = _patch_method(CMN.FoldingTrunk, "forward", _FT_EDITS, "ft",
                                                                   extra_globals={"_ef2xl_owned_block": _owned_block})
    if lmpair and "lms" not in _ORIG:
        led["patch_shas"]["LanguageModelShim.forward"] = _patch_method(CMN.LanguageModelShim, "forward", _LMS_EDITS, "lms",
                                                                        extra_globals={"_ef2xl_s2p": _s2p_xl})
    if relpos and "relpos_forward" not in _ORIG:
        _ORIG["relpos_forward"] = CMN.ResIdxAsymIdSymIdEntityIdEncoding.forward
        CMN.ResIdxAsymIdSymIdEntityIdEncoding.forward = _relpos_xl
    if distocpu:
        if model is None:
            raise ValueError("distocpu needs the model instance")
        _install_disto_cpu(model)
    if initlean and "init_pair_state" not in _ORIG:
        _ORIG["init_pair_state"] = MOD.ESMFold2Model._init_pair_state
        MOD.ESMFold2Model._init_pair_state = _init_pair_state_xl
    if (loopfree or own) and "loop" not in _ORIG:
        led["patch_shas"]["_run_one_loop"] = _patch_method(MOD.ESMFold2Model, "_run_one_loop", _LOOP_EDITS, "loop")
        _APPLIED["loopfree"] = True
    if (_CFG["free"] & {"z", "relpos"}) and "conf" not in _ORIG:
        led["patch_shas"]["ConfidenceHead.forward"] = _patch_method(MOD.ConfidenceHead, "forward", _CONF_EDITS, "conf")
        _APPLIED["free"] = True
    if _CFG["cond"] and "cond" not in _ORIG:
        led["patch_shas"]["DiffusionConditioning.forward"] = _patch_method(
            CMN.DiffusionConditioning, "forward", _COND_EDITS, "cond", extra_globals={"_ef2xl_cond_pair": _cond_pair_xl})
        _ORIG["cond_hook"] = getattr(CMN.DiffusionConditioning, PAIR_PATH_HOOK, None)   # the second route: a re-issued sampler step's z branch calls this hook
        setattr(CMN.DiffusionConditioning, PAIR_PATH_HOOK, _cond_pair_hook)
        _APPLIED["cond"] = True
    if esmc_offload:
        if model is None:
            raise ValueError("esmc_offload needs the model instance")
        _install_esmc_offload(model, int(esmc_min_tok), engine=_x4_engine(esmc_stream))
        _APPLIED["esmc_offload"] = True
    _log("applied " + json.dumps(led, sort_keys=True))
    return led


def unapply():
    """Restore every patched callable (esmc offload wrapper included)."""
    CMN, MOD = _mods()
    if "loop" in _ORIG:
        MOD.ESMFold2Model._run_one_loop = _ORIG.pop("loop")
    if "conf" in _ORIG:
        MOD.ConfidenceHead.forward = _ORIG.pop("conf")
    if "cond" in _ORIG:
        CMN.DiffusionConditioning.forward = _ORIG.pop("cond")
    if "cond_hook" in _ORIG:
        prev = _ORIG.pop("cond_hook")
        if prev is None:
            try:
                delattr(CMN.DiffusionConditioning, PAIR_PATH_HOOK)
            except AttributeError:
                pass
        else:
            setattr(CMN.DiffusionConditioning, PAIR_PATH_HOOK, prev)
    if "esmc_clhs" in _ORIG:
        MOD.ESMFold2Model._compute_lm_hidden_states = _ORIG.pop("esmc_clhs")
    st = _X4_STATE.get("streamer")
    if st is not None:                                                     # the LM back on the device, whole, from its pinned masters
        try:
            if st.where != "cuda":
                st.reload_to_device()
            st.unhook()
        except Exception as e:  # noqa: BLE001
            _log(f"x4 unapply: reload failed ({e!r})")
        _X4_STATE["streamer"] = None
    if "ft" in _ORIG:
        CMN.FoldingTrunk.forward = _ORIG.pop("ft")
    if "lms" in _ORIG:
        CMN.LanguageModelShim.forward = _ORIG.pop("lms")
    if "relpos_forward" in _ORIG:
        CMN.ResIdxAsymIdSymIdEntityIdEncoding.forward = _ORIG.pop("relpos_forward")
    if "init_pair_state" in _ORIG:
        MOD.ESMFold2Model._init_pair_state = _ORIG.pop("init_pair_state")
    if model_ref.get("m") is not None and getattr(model_ref["m"], "_ef2xl_disto_hook", None) is not None:
        model_ref["m"]._ef2xl_disto_hook.remove(); model_ref["m"]._ef2xl_disto_hook = None
    _APPLIED.clear(); _CFG["free"] = set(); _CFG["cond"] = ""; _CFG["loopfree"] = False
    _CFG["own"] = False; _CFG["lmpair"] = False; _CFG["relpos"] = False; _CFG["initlean"] = False
    _log("unapplied (stock callables restored)")


def stats():
    return dict(_STATS)


atexit.register(lambda: _log("stats " + json.dumps(_STATS, sort_keys=True)))
