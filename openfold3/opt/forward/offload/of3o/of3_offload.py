# The offload add-on's hook module: single-GPU large-N inference units for OpenFold3 0.4.1 (the `big` mode's row-blocked
# statements, the pinned host pool, the LayerNorm guard, the chunk-size plan, the runner wrap and the logging-only diagnostics;
# apply_core / apply_runner install them).
# Monkeypatch-only. Nothing numeric is patched unless OF3O_LAYER is set to 1 (0 = instrumentation + chunk logging only; 2 is refused by name).
# Design rule: every GEMM keeps the operand shapes/strides of stock 0.4.1; only tensor residency (GPU vs pinned host)
# and the lifetime of O(N^2) transients change.  README.md (beside of3o/) lists the units and their switches.
import json
import os
import sys
import threading
import time

import torch

_T0 = time.time()
LAYER = int(os.environ.get("OF3O_LAYER", "0") or 0)
_LOG_LOCK = threading.Lock()
STATE = {"query_id": None, "seed": None, "n_tok": None, "phase": None, "block_times": [], "tuned": {},
         "fallbacks": {}, "pageable": {}, "conf": None, "lnsafe_scoped": 0}
# FAIL-CLOSED: every branch that hands a call to the saved stock body, and every host buffer left pageable,
# is a NAMED EVENT counted in STATE["fallbacks"] / STATE["pageable"]
# and read by openfold3_opt (stack.levers_record -> the exit tally's offload_* counters); a `big` row with any count > 0
# is PARTIAL (rc 3, by name: a mode is all of its levers). A host buffer over OF3O_PIN_BUDGET_GB is pageable with a NOTE line and a count (OF3O_PIN_POLICY=census, the
# default; `strict` refuses it by name) — never PARTIAL; a pinned allocation the host refuses raises by name.


def fallback(site, **kw):
    """A stock-body fallback under the add-on's name: counted per site and logged; never silent."""
    STATE["fallbacks"][site] = STATE["fallbacks"].get(site, 0) + 1
    log("fallback", site=site, n=STATE["fallbacks"][site], **kw)


def census():
    """The add-on's record for the package: what was applied, every fallback and pageable event, the confidence path as run."""
    if STATE.get("conf") and STATE["conf"].get("path") in ("chunked", "auto"):
        import of3o_confidence as _OC                                               # the chunked scorer ran: its census is REQUIRED (an unreadable record raises, never reads empty)
        conf_census = dict(_OC.CENSUS)
    else:
        _OC = sys.modules.get("of3o_confidence")
        conf_census = dict(getattr(_OC, "CENSUS", {})) if _OC is not None else {}
    return {"applied": _APPLIED, "applied_runner": _APPLIED_RUNNER, "layer": LAYER, "fallbacks": dict(STATE["fallbacks"]), "pageable": dict(STATE["pageable"]),
            "pinned_total_gb": round(_PIN_BYTES["pinned"] / 2**30, 3), "pinned_peak_gb": round(_PIN_BYTES.get("peak", 0) / 2**30, 3), "pinned_allocs": _PIN_BYTES.get("allocs", 0),
            "forward_prior": STATE.get("forward_prior"), "forward_superseded": STATE.get("forward_superseded"), "triatt_end_dispatch": STATE.get("triatt_end_dispatch"),
            "conf_head_stock_blocks": STATE.get("conf_head_stock_blocks", 0), "conf": STATE.get("conf"), "conf_census": conf_census, "tuned": dict(STATE["tuned"]),
            "lnsafe_calls": STATE.get("lnsafe_calls", 0), "lnsafe_scoped_modules": STATE.get("lnsafe_scoped", 0), "templ_fix_calls": STATE.get("templ_fix_calls", 0)}


def log(event, **kw):
    rec = {"t": round(time.time() - _T0, 3), "event": event}
    rec.update(kw)
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        rec["alloc_gb"] = round(torch.cuda.memory_allocated() / 2**30, 3)
        rec["max_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        rec["reserved_gb"] = round(torch.cuda.memory_reserved() / 2**30, 3)
    line = json.dumps(rec, default=str)
    with _LOG_LOCK:
        sys.stderr.write("[of3o] " + line + "\n")
        sys.stderr.flush()

PIN_POLICIES = ("census", "strict")          # OF3O_PIN_POLICY: `census` (the default) pages a host buffer over OF3O_PIN_BUDGET_GB with a NOTE line and a count; `strict` refuses it by name


def pin_policy(environ=None) -> str:
    """The pinned-pool policy word of this process (OF3O_PIN_POLICY; default `census`; an unknown word is refused by name)."""
    environ = os.environ if environ is None else environ
    word = (environ.get("OF3O_PIN_POLICY") or "census").strip() or "census"
    if word not in PIN_POLICIES:
        raise ValueError(f"[of3o] OF3O_PIN_POLICY={word!r}: one of {PIN_POLICIES}")
    return word


def note(msg: str) -> None:
    """One human NOTE line on stderr (a property decided up front that the run proceeds with; the JSON event log carries the same fact for readers)."""
    with _LOG_LOCK:
        sys.stderr.write("[of3o] " + msg + "\n")
        sys.stderr.flush()



def env_int(name, default):
    v = os.environ.get(name, "")
    return int(v) if v.strip() else default




def chunk_for(name, default):
    """Fixed chunk size for a stack in layers where the tuners are disabled: OF3O_CHUNK = int | JSON {tuner_name: chunk, "default": chunk}."""
    v = os.environ.get("OF3O_CHUNK", "").strip()
    if not v:
        return default
    if v.startswith("{"):
        plan = json.loads(v)
        r = plan.get(name, plan.get("default", default))
        return default if r is None else int(r)
    return int(v)


# ----------------------------------------------------------------------------------------------- pinned host helpers
_PIN_POOL = {}
_PIN_BYTES = {"pinned": 0}


def pinned_like(t, tag):
    """Reusable host buffer with t's shape/dtype (one per tag).  Pinned while the running total stays under OF3O_PIN_BUDGET_GB (default 600 GiB);
    beyond it pageable with a NOTE line and a count (OF3O_PIN_POLICY=census, the default) or refused by name (`strict`); a pinned allocation the host
    refuses (the locked-memory limit) raises by name — the budget is the up-front knob, nothing pages around a failure at run time."""
    key = (tag, tuple(t.shape), t.dtype)
    buf = _PIN_POOL.get(key)
    if buf is None:
        free_pinned(tag)
        nbytes = t.numel() * t.element_size()
        budget = float(os.environ.get("OF3O_PIN_BUDGET_GB", "600")) * 2**30
        want_pin = (_PIN_BYTES["pinned"] + nbytes) <= budget
        t0 = time.time()
        pinned = False
        policy = pin_policy()
        if want_pin:
            try:
                buf = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
                pinned = True
            except RuntimeError as e:  # the locked-memory limit (or the host) refused the pinned allocation: named, never paged around at run time
                raise RuntimeError(f"[of3o] pinned host buffer {tag} ({nbytes / 2**30:.2f} GB): pinned allocation failed (pinned so far "
                                   f"{_PIN_BYTES['pinned'] / 2**30:.2f} GB): {str(e)[:160]} — OF3O_PIN_BUDGET_GB=<gb> below this total pages the buffers beyond it up front") from e
        if buf is None:                                                            # over OF3O_PIN_BUDGET_GB: decided up front by the budget arithmetic
            if policy == "strict":
                raise RuntimeError(f"[of3o] pinned host buffer {tag} ({nbytes / 2**30:.2f} GB): over OF3O_PIN_BUDGET_GB (pinned so far {_PIN_BYTES['pinned'] / 2**30:.2f} GB); "
                                   f"OF3O_PIN_POLICY=strict refuses a pageable buffer (OF3O_PIN_POLICY=census, the default, pages and counts it)")
            STATE["pageable"][tag] = STATE["pageable"].get(tag, 0) + 1
            note(f"NOTE pinned host pool over OF3O_PIN_BUDGET_GB ({budget / 2**30:.0f} GB; pinned so far {_PIN_BYTES['pinned'] / 2**30:.2f} GB): buffer {tag} "
                 f"({nbytes / 2**30:.2f} GB) is pageable — slower host<->device copies, same numbers; counted (pageable x{STATE['pageable'][tag]}); OF3O_PIN_BUDGET_GB raises the budget, OF3O_PIN_POLICY=strict refuses instead")
            log("pageable", tag=tag, reason="over OF3O_PIN_BUDGET_GB", gb=round(nbytes / 2**30, 2), n=STATE["pageable"][tag])
            buf = torch.empty(t.shape, dtype=t.dtype)
        if pinned:
            _PIN_BYTES["pinned"] += nbytes
            _PIN_BYTES["peak"] = max(_PIN_BYTES.get("peak", 0), _PIN_BYTES["pinned"]); _PIN_BYTES["allocs"] = _PIN_BYTES.get("allocs", 0) + 1   # the pool's high-water record (read by the package after the buffers are freed)
        buf._of3o_pinned_bytes = nbytes if pinned else 0
        log("host_alloc", tag=tag, gb=round(nbytes / 2**30, 2), pinned=pinned, pinned_total_gb=round(_PIN_BYTES["pinned"] / 2**30, 2),
            s=round(time.time() - t0, 1))
        _PIN_POOL[key] = buf
    return buf


def to_host(t, tag):
    h = pinned_like(t, tag)
    h.copy_(t, non_blocking=False)
    return h


def free_pinned(tag=None):
    for k in [k for k in _PIN_POOL if tag is None or k[0] == tag]:
        _PIN_BYTES["pinned"] -= getattr(_PIN_POOL[k], "_of3o_pinned_bytes", 0)
        del _PIN_POOL[k]


# ----------------------------------------------------------------------------------------------- chunk-size tuner pin
def patch_tuner():
    from openfold3.core.utils import chunk_utils as CU

    orig = CU.ChunkSizeTuner.tune_chunk_size
    plan_env = os.environ.get("OF3O_CHUNK", "").strip()

    def plan_for(name):
        if not plan_env:
            return None
        if plan_env.startswith("{"):
            plan = json.loads(plan_env)
            v = plan.get(name, plan.get("default"))
            return None if v is None else int(v)
        return int(plan_env)

    def tune_chunk_size(self, representative_fn, args, min_chunk_size, max_chunk_size=CU.DEFAULT_MAX_CHUNK_SIZE):
        name = getattr(self, "_of3o_name", "tuner%x" % id(self))
        v = plan_for(name)
        shapes = [tuple(a.shape) for a in args if isinstance(a, torch.Tensor)]
        if v is None:
            t0 = time.time()
            v = orig(self, representative_fn, args, min_chunk_size, max_chunk_size)
            log("chunk_tuned", tuner=name, chunk=v, min=min_chunk_size, max=max_chunk_size, shapes=shapes, s=round(time.time() - t0, 2))
        else:
            v = max(int(v), int(min_chunk_size))
            self.cached_chunk_size = v
            self.cached_arg_data = None
            log("chunk_pinned", tuner=name, chunk=v, min=min_chunk_size, max=max_chunk_size, shapes=shapes)
        STATE["tuned"][name] = v
        return v

    CU.ChunkSizeTuner.tune_chunk_size = tune_chunk_size


def name_tuners(model):
    for n, m in model.named_modules():
        t = getattr(m, "chunk_size_tuner", None)
        if t is not None and not hasattr(t, "_of3o_name"):
            t._of3o_name = n or "root"


# ----------------------------------------------------------------------------------------------- O-TRIMUL
def trimul_inference_forward(self, z, mask=None, inplace_chunk_size=None, with_add=True, use_triton_triangle_kernels=False):
    """Stock TriangleMultiplicativeUpdate._inference_forward with the z-cache replaced by a pinned-host snapshot of the
    ORIGINAL rows (outgoing) — all projections / einsums / LN / linears are the stock statements with stock chunk boundaries."""
    from openfold3.core.utils.tensor_utils import permute_final_dims

    assert not use_triton_triangle_kernels, "of3o: triton tri-mul path not supported"
    assert inplace_chunk_size is not None
    if mask is None:
        mask = z.new_ones(z.shape[:-1])
    mask = mask.unsqueeze(-1)

    def compute_projection_helper(pair, mask, a=True):
        if a:
            linear_g, linear_p = self.linear_a_g, self.linear_a_p
        else:
            linear_g, linear_p = self.linear_b_g, self.linear_b_p
        pair = self.layer_norm_in(pair)
        p = linear_g(pair)
        p.sigmoid_()
        p *= linear_p(pair)
        p *= mask
        p = permute_final_dims(p, (2, 0, 1))
        return p

    def compute_projection(pair, mask, a=True, chunked=True):
        need_transpose = self._outgoing ^ a
        if not chunked:
            p = compute_projection_helper(pair, mask, a)
            if need_transpose:
                p = p.transpose(-1, -2)
        else:
            linear_g = self.linear_a_g if a else self.linear_b_g
            c = linear_g.weight.shape[-2]
            out_shape = pair.shape[:-3] + (c,) + pair.shape[-3:-1]
            p = pair.new_zeros(out_shape)
            for i in range(0, pair.shape[-3], inplace_chunk_size):
                pair_chunk = compute_projection_helper(pair[..., i: i + inplace_chunk_size, :, :], mask[..., i: i + inplace_chunk_size, :, :], a)
                if need_transpose:
                    pair_chunk = pair_chunk.transpose(-1, -2)
                    p[..., i: i + inplace_chunk_size] = pair_chunk
                else:
                    p[..., i: i + inplace_chunk_size, :] = pair_chunk
                del pair_chunk
        return p

    a = compute_projection(z, mask, True, chunked=True)

    n = a.shape[-1]
    half_n = n // 2 + n % 2
    row_dim, col_dim = -3, -2
    b_chunk_dim = row_dim if self._outgoing else col_dim

    def slice_tensor(t, start, end, dim):
        s = [slice(None) for _ in t.shape]
        s[dim] = slice(start, end)
        return t[tuple(s)]

    snap = None
    if b_chunk_dim == row_dim:
        # original rows are needed after their left parts have been overwritten: read them from a host snapshot
        snap = to_host(z, "trimul_snap")

    # identical chunk boundaries to stock (one contracted chunk before half_n)
    i_range = list(range(0, half_n, inplace_chunk_size))
    initial_offsets = [i_2 - i_1 for i_1, i_2 in zip(i_range, i_range[1:] + [half_n], strict=True)]
    after_half = list(range(half_n, n, inplace_chunk_size))
    after_half_offsets = [inplace_chunk_size for _ in after_half]
    for i, offset in zip(i_range + after_half, initial_offsets + after_half_offsets, strict=False):
        if b_chunk_dim == row_dim:
            z_chunk_b = slice_tensor(snap, i, i + offset, row_dim).to(z.device, non_blocking=False)
        else:
            z_chunk_b = slice_tensor(z, i, i + offset, col_dim)
        mask_chunk = slice_tensor(mask, i, i + offset, b_chunk_dim)
        b_chunk = compute_projection(z_chunk_b, mask_chunk, a=False, chunked=False)
        del z_chunk_b
        x_chunk = torch.einsum("...ij,...jk->...ik", a, b_chunk)
        x_chunk = permute_final_dims(x_chunk, (1, 2, 0))
        x_chunk = self.layer_norm_out(x_chunk)
        x_chunk = self.linear_z(x_chunk)
        z_chunk_g = slice_tensor(z, i, i + offset, col_dim)
        g_chunk = self.linear_g(self.layer_norm_in(z_chunk_g))
        g_chunk.sigmoid_()
        del z_chunk_g
        x_chunk *= g_chunk
        z_slicer = [slice(None) for _ in z.shape]
        z_slicer[col_dim] = slice(i, i + offset)
        if with_add:
            z[tuple(z_slicer)] += x_chunk
        else:
            z[tuple(z_slicer)] = x_chunk
        del x_chunk, g_chunk, b_chunk
    del a
    return z


# ----------------------------------------------------------------------------------------------- O-TRIATT (ending node) + O-TRANS
def tri_att_end_lean(tri_att, z, pair_mask, attn_chunk_size, use_deepspeed_evo_attention, use_cueq_triangle_kernels,
                     use_triton_triangle_kernels, use_lma):
    """Ending-node triangle attention on z WITHOUT the two transpose().contiguous() copies of stock PairBlock.
    x := LN(z^T) built block-wise at transposed positions (identical values/layout to stock's LN of the contiguous z^T),
    bias GEMM on the full x exactly as stock, MHA through the stock chunk_layer with _out=x, then z += x^T (elementwise)."""
    from openfold3.core.utils.tensor_utils import permute_final_dims

    disp = sys.modules.get("of3t_levers")                                      # NEED 2: an installed trunk-kernels dispatcher (OF3T_TRIATT) decides the backend here too —
    if disp is not None and os.environ.get("OF3T_TRIATT") and hasattr(disp, "_backend_kwargs"):   # this path bypasses TriangleAttention.forward, where that hook routes
        bk = disp._backend_kwargs(os.environ["OF3T_TRIATT"]) or {}
        use_deepspeed_evo_attention = bk.get("use_deepspeed_evo_attention", use_deepspeed_evo_attention)
        use_cueq_triangle_kernels = bk.get("use_cueq_triangle_kernels", use_cueq_triangle_kernels)
        use_triton_triangle_kernels = bk.get("use_triton_triangle_kernels", use_triton_triangle_kernels)
        STATE["triatt_end_dispatch"] = os.environ["OF3T_TRIATT"]

    N = z.shape[-2]
    x = torch.empty_like(z)  # holds LN(z^T): x[..., j, i, :] = LN(z)[..., i, j, :]
    cb = 256                                                                   # LayerNorm row block
    for j0 in range(0, N, cb):
        j1 = min(N, j0 + cb)
        blk = tri_att.layer_norm(z[..., :, j0:j1, :])          # [*, N, jb, C] (LN is per position)
        x[..., j0:j1, :, :] = blk.transpose(-2, -3)
        del blk
    if pair_mask is None:
        mask = x.new_ones(x.shape[:-1])
    else:
        mask = pair_mask.transpose(-1, -2)                    # what stock passes to tri_att_end
    mask_bias = (tri_att.inf * (mask - 1))[..., :, None, None, :]
    triangle_bias = permute_final_dims(tri_att.linear_z(x), (2, 0, 1))
    triangle_bias = triangle_bias.unsqueeze(-4)
    biases = [mask_bias, triangle_bias]
    if attn_chunk_size is not None:
        x = tri_att._chunk(x, biases, attn_chunk_size, use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                           use_lma=use_lma, use_cueq_triangle_kernels=use_cueq_triangle_kernels,
                           use_triton_triangle_kernels=use_triton_triangle_kernels, inplace_safe=True)
    else:
        x = tri_att.mha(q_x=x, kv_x=x, biases=biases, use_deepspeed_evo_attention=use_deepspeed_evo_attention, use_lma=use_lma,
                        use_cueq_triangle_kernels=use_cueq_triangle_kernels, use_triton_triangle_kernels=use_triton_triangle_kernels)
    del biases, triangle_bias, mask_bias
    z.add_(x.transpose(-2, -3))
    del x
    return z


def pairblock_tri_att_start_end(self, z, _attn_chunk_size, pair_mask, use_deepspeed_evo_attention, use_cueq_triangle_kernels,
                                 use_triton_triangle_kernels=False, use_lma=False, inplace_safe=False):
    from openfold3.core.utils.tensor_utils import add
    if not inplace_safe:
        fallback("PairBlock.tri_att_start_end", inplace_safe=bool(inplace_safe))
        return _ORIG["PairBlock.tri_att_start_end"](self, z, _attn_chunk_size, pair_mask, use_deepspeed_evo_attention,
                                                    use_cueq_triangle_kernels, use_triton_triangle_kernels, use_lma, inplace_safe)
    z = add(z, self.ps_dropout_row_layer(self.tri_att_start(z, mask=pair_mask, chunk_size=_attn_chunk_size,
                                                             use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                                                             use_cueq_triangle_kernels=use_cueq_triangle_kernels,
                                                             use_triton_triangle_kernels=use_triton_triangle_kernels,
                                                             use_lma=use_lma, inplace_safe=inplace_safe)), inplace=True)
    z = tri_att_end_lean(self.tri_att_end, z, pair_mask, _attn_chunk_size, use_deepspeed_evo_attention, use_cueq_triangle_kernels,
                         use_triton_triangle_kernels, use_lma)
    return z


def pairblock_forward(self, z, pair_mask, chunk_size=None, use_deepspeed_evo_attention=False, use_cueq_triangle_kernels=False,
                      use_triton_triangle_kernels=False, use_lma=False, inplace_safe=False, _mask_trans=True, _attn_chunk_size=None):
    if chunk_size is None and inplace_safe and STATE.get("conf_head_unchunked"):
        STATE["conf_head_stock_blocks"] = STATE.get("conf_head_stock_blocks", 0) + 1     # the confidence head's pair blocks under kernels: stock's unchunked path by stock's own rule (named, counted; census conf_head_stock_blocks)
        if STATE["conf_head_stock_blocks"] == 1:
            log("conf_head_unchunked", note="stock runs the confidence head's pair stack unchunked under kernels with several samples (prediction_heads.py); the O1 pair-block lever covers the trunk")
        return _ORIG["PairBlock.forward"](self, z, pair_mask, chunk_size, use_deepspeed_evo_attention, use_cueq_triangle_kernels,
                                          use_triton_triangle_kernels, use_lma, inplace_safe, _mask_trans, _attn_chunk_size)
    if not inplace_safe or chunk_size is None:
        fallback("PairBlock.forward", inplace_safe=bool(inplace_safe), chunk_size=chunk_size)
        return _ORIG["PairBlock.forward"](self, z, pair_mask, chunk_size, use_deepspeed_evo_attention, use_cueq_triangle_kernels,
                                          use_triton_triangle_kernels, use_lma, inplace_safe, _mask_trans, _attn_chunk_size)
    pair_trans_mask = pair_mask if _mask_trans else None
    if _attn_chunk_size is None:
        _attn_chunk_size = chunk_size
    z = self.tri_mul_out_in(z=z, pair_mask=pair_mask, inplace_safe=inplace_safe, use_cueq_triangle_kernels=use_cueq_triangle_kernels,
                            use_triton_triangle_kernels=use_triton_triangle_kernels)
    z = self.tri_att_start_end(z=z, _attn_chunk_size=_attn_chunk_size, pair_mask=pair_mask,
                               use_deepspeed_evo_attention=use_deepspeed_evo_attention, use_cueq_triangle_kernels=use_cueq_triangle_kernels,
                               use_triton_triangle_kernels=use_triton_triangle_kernels, use_lma=use_lma, inplace_safe=inplace_safe)
    # O-TRANS: z[rows] += transition(z[rows]) with the stock chunk (identical per-chunk GEMMs; no [N,N,C] output buffer)
    tr = self.pair_transition
    N, C = z.shape[-2], z.shape[-1]
    zf = z.view(-1, N, C)                       # flat rows, same flattening as chunk_layer (no_batch_dims = ndim-2)
    if pair_trans_mask is None:
        mf = None
    else:
        m = pair_trans_mask.unsqueeze(-1)
        m = m.expand(z.shape[:-1] + (1,))
        mf = m.reshape(-1, N, 1)
    for i in range(0, zf.shape[0], chunk_size):
        xm = mf[i: i + chunk_size] if mf is not None else zf.new_ones(zf[i: i + chunk_size].shape[:-1] + (1,))
        zf[i: i + chunk_size] += tr._transition(x=zf[i: i + chunk_size], mask=xm)
    return z


# ----------------------------------------------------------------------------------------------- relpos rows (exact integer path)
def relpos_rows(batch, max_relative_idx, max_relative_chain, r0, r1):
    """rows r0:r1 of openfold3.core.utils.relpos.relpos_complex(batch) — same integer ops, restricted query index."""
    from openfold3.core.utils.tensor_utils import binned_one_hot
    res_idx, asym_id, entity_id = batch["residue_index"], batch["asym_id"], batch["entity_id"]
    same_chain = asym_id[..., r0:r1, None] == asym_id[..., None, :]
    same_res = res_idx[..., r0:r1, None] == res_idx[..., None, :]
    same_entity = entity_id[..., r0:r1, None] == entity_id[..., None, :]

    def relpos(pos, condition, rel_clip_idx):
        offset = pos[..., r0:r1, None] - pos[..., None, :]
        clipped_offset = torch.clamp(offset + rel_clip_idx, min=0, max=2 * rel_clip_idx)
        final_offset = torch.where(condition, clipped_offset, (2 * rel_clip_idx + 1) * torch.ones_like(clipped_offset))
        boundaries = torch.arange(start=0, end=2 * rel_clip_idx + 2, device=final_offset.device)
        return binned_one_hot(final_offset, boundaries)

    rel_pos = relpos(res_idx, same_chain, max_relative_idx)
    rel_token = relpos(batch["token_index"], same_chain & same_res, max_relative_idx)
    rel_chain = relpos(batch["sym_id"], same_entity, max_relative_chain)
    same_entity = same_entity[..., None].to(dtype=rel_pos.dtype)
    return torch.cat([rel_pos, rel_token, same_entity, rel_chain], dim=-1)


# ----------------------------------------------------------------------------------------------- O-INPUT
def input_embedder_forward(self, batch, inplace_safe=False, use_high_precision_attention=False):
    from openfold3.core.utils.relpos import relpos_complex
    rows = env_int("OF3O_INPUT_ROWS", 0)
    with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
        a, _, _, _ = self.atom_attn_enc(batch=batch, use_high_precision_attention=use_high_precision_attention)
    a = a.to(dtype=self.linear_s.weight.dtype)
    s_input = torch.cat([a, batch["restype"], batch["profile"], batch["deletion_mean"].unsqueeze(-1)], dim=-1)
    s = self.linear_s(s_input)
    s_input_emb_i = self.linear_z_i(s_input)
    s_input_emb_j = self.linear_z_j(s_input)
    N = s.shape[-2]
    if rows <= 0 or not inplace_safe:
        token_bonds_emb = self.linear_token_bonds(batch["token_bonds"].unsqueeze(-1).to(dtype=s.dtype))
        z = s_input_emb_i[..., None, :] + s_input_emb_j[..., None, :, :]
        relpos_feats = relpos_complex(batch=batch, max_relative_idx=self.max_relative_idx, max_relative_chain=self.max_relative_chain).to(dtype=z.dtype)
        z += self.linear_relpos(relpos_feats)
        del relpos_feats
        z += token_bonds_emb
        return s_input, s, z
    z = torch.empty(s.shape[:-2] + (N, N, self.linear_relpos.weight.shape[0]), dtype=s.dtype, device=s.device)
    tb = batch["token_bonds"]
    for r0 in range(0, N, rows):
        r1 = min(N, r0 + rows)
        zr = s_input_emb_i[..., r0:r1, None, :] + s_input_emb_j[..., None, :, :]
        rp = relpos_rows(batch, self.max_relative_idx, self.max_relative_chain, r0, r1).to(dtype=s.dtype)
        zr += self.linear_relpos(rp)
        del rp
        zr += self.linear_token_bonds(tb[..., r0:r1, :].unsqueeze(-1).to(dtype=s.dtype))
        z[..., r0:r1, :, :] = zr
        del zr
    return s_input, s, z


# ----------------------------------------------------------------------------------------------- O-COND (pair path once per rollout)
COND = {"zij": None, "chunk": None}
HANDOFF = {}
AUX_HEADS_IMPL = {"fn": None}


def cond_pair_path(dc, batch, zij_trunk, chunk_size):
    """DiffusionConditioning pair path (relpos -> cat -> LN -> linear_z -> 2 transitions), t-independent.
    OF3O_COND_ROWS=0: stock statements on full tensors; >0: evaluated per block of token rows (GEMM M changes for linear_z)."""
    from openfold3.core.utils.relpos import relpos_complex
    rows = env_int("OF3O_COND_ROWS", 0)
    token_mask = batch["token_mask"]
    pair_token_mask = token_mask.unsqueeze(-1) * token_mask.unsqueeze(-2)
    N = zij_trunk.shape[-2]
    hz = getattr(zij_trunk, "_of3o", None)
    dev = token_mask.device
    if hz is not None and rows <= 0:
        rows = env_int("OF3O_ROWS", 128)
    if rows <= 0:
        relpos_zij = relpos_complex(batch=batch, max_relative_idx=dc.max_relative_idx, max_relative_chain=dc.max_relative_chain).to(dtype=zij_trunk.dtype)
        zij = torch.cat([zij_trunk, relpos_zij], dim=-1)
        del relpos_zij
        zij = dc.linear_z(dc.layer_norm_z(zij))
    else:
        zij = torch.empty(zij_trunk.shape[:-1] + (dc.linear_z.weight.shape[0],), dtype=zij_trunk.dtype, device=dev)
        for r0 in range(0, N, rows):
            r1 = min(N, r0 + rows)
            rp = relpos_rows(batch, dc.max_relative_idx, dc.max_relative_chain, r0, r1).to(dtype=zij_trunk.dtype)
            if hz is not None:
                zrows = hz.rows(r0, r1).view(zij_trunk.shape[:-3] + (r1 - r0, N, zij_trunk.shape[-1]))
            else:
                zrows = zij_trunk[..., r0:r1, :, :]
                if not zrows.is_cuda:
                    zrows = zrows.to(dev)
            cat = torch.cat([zrows, rp], dim=-1)
            del zrows
            del rp
            zij[..., r0:r1, :, :] = dc.linear_z(dc.layer_norm_z(cat))
            del cat
    # pair transitions, in place per row block with the (pinned/tuned) chunk: zij[rows] += l(zij[rows])
    C = zij.shape[-1]
    zf = zij.view(-1, N, C)
    mf = pair_token_mask.unsqueeze(-1).expand(zij.shape[:-1] + (1,)).reshape(-1, N, 1)
    cs = chunk_size if chunk_size is not None else zf.shape[0]
    for l in dc.transition_z:
        for i in range(0, zf.shape[0], cs):
            zf[i: i + cs] += l._transition(x=zf[i: i + cs], mask=mf[i: i + cs])
    return zij


def diffusion_conditioning_forward(self, batch, t, si_input, si_trunk, zij_trunk, use_conditioning, chunk_size=None):
    """Pair path from the per-rollout cache (COND), single path per step exactly as stock."""
    token_mask = batch["token_mask"]
    if COND["zij"] is None:
        fallback("DiffusionConditioning.forward", cond_cache="empty")
        return _ORIG["DiffusionConditioning.forward"](self, batch, t, si_input, si_trunk, zij_trunk, use_conditioning, chunk_size)
    if not use_conditioning:
        si_trunk = si_trunk * 0
    zij = COND["zij"]
    si = torch.cat([si_trunk, si_input], dim=-1)
    si = self.linear_s(self.layer_norm_s(si))
    n = 0.25 * torch.log(t / self.sigma_data)
    n = self.fourier_emb(n.unsqueeze(-1))
    si = si + self.linear_n(self.layer_norm_n(n)).unsqueeze(-2)
    cs = COND.get("chunk", chunk_size)
    for l in self.transition_s:
        si = si + l(si, mask=token_mask, chunk_size=cs)
    return si, zij


# ----------------------------------------------------------------------------------------------- O-RECYCLE / trunk
def run_trunk(self, batch, num_cycles, inplace_safe=False):
    from openfold3.core.utils.tensor_utils import add
    if not inplace_safe:
        fallback("OpenFold3.run_trunk", inplace_safe=bool(inplace_safe))
        return _ORIG["OpenFold3.run_trunk"](self, batch, num_cycles, inplace_safe)
    mode_mem_settings = self._get_mode_mem_settings()
    name_tuners(self)
    STATE["phase"] = "trunk"
    STATE["n_tok"] = int(batch["token_mask"].shape[-1])
    log("phase_start", phase="trunk", n_tok=STATE["n_tok"], n_atom=int(batch["atom_mask"].shape[-1]) if "atom_mask" in batch else None, num_cycles=num_cycles)
    torch.cuda.reset_peak_memory_stats()

    s_input, s_init, z_init = self.input_embedder(batch=batch, inplace_safe=inplace_safe, use_high_precision_attention=True)
    log("input_embedder_done")
    z_init_h = to_host(z_init, "z_init")
    zshape, zdtype, dev = z_init.shape, z_init.dtype, z_init.device
    del z_init
    s = torch.zeros_like(s_init)
    z = torch.zeros(zshape, dtype=zdtype, device=dev)
    token_mask = batch["token_mask"]
    pair_mask = token_mask[..., None] * token_mask[..., None, :]
    rows = env_int("OF3O_ROWS", 256)

    for cycle_no in range(num_cycles):
        tc = time.time()
        with torch.no_grad():
            # z = z_init + linear_z(LN(z))  — full-size GEMM as stock; z_init streamed from pinned host per row block
            rrows = env_int("OF3O_RECYCLE_ROWS", 0)
            if rrows > 0:
                # row-chunked recycle (peak = z + one row block instead of 2 z-eq): LN per position, linear_z per row block
                # (GEMM M = rows*N, the statement O2 uses; unit-tested bitwise vs the full GEMM), written back in place.
                N = z.shape[-3]
                for r0 in range(0, N, rrows):
                    r1 = min(N, r0 + rrows)
                    blk = self.linear_z(self.layer_norm_z(z[..., r0:r1, :, :]))
                    blk += z_init_h[..., r0:r1, :, :].to(dev, non_blocking=False)
                    z[..., r0:r1, :, :] = blk
                    del blk
            else:
                t = self.layer_norm_z(z)
                del z
                u = self.linear_z(t)
                del t
                N = u.shape[-2]
                for r0 in range(0, N, rows):
                    r1 = min(N, r0 + rows)
                    u[..., r0:r1, :, :] += z_init_h[..., r0:r1, :, :].to(dev, non_blocking=False)
                z = u
                del u
            t_kw = dict(batch=batch, z=z, pair_mask=pair_mask, chunk_size=mode_mem_settings.chunk_size, _mask_trans=True,
                        use_deepspeed_evo_attention=mode_mem_settings.use_deepspeed_evo_attention,
                        use_triton_triangle_kernels=mode_mem_settings.use_triton_triangle_kernels,
                        use_cueq_triangle_kernels=mode_mem_settings.use_cueq_triangle_kernels,
                        use_lma=mode_mem_settings.use_lma, inplace_safe=inplace_safe)
            if env_int("OF3O_TEMPL", 1):
                t_out = template_embedder_forward(self.template_embedder, _of3o_accum=True, **t_kw)   # adds into z in place
                assert t_out is None
            else:
                z = add(z, self.template_embedder(**t_kw), inplace=inplace_safe)
            m, msa_mask = self.msa_module_embedder(batch=batch, s_input=s_input)
            swiglu_token_cutoff = mode_mem_settings.msa_module.swiglu_chunk_token_cutoff
            transition_ckpt_chunk_size = (mode_mem_settings.msa_module.swiglu_seq_chunk_size
                                          if swiglu_token_cutoff is None or swiglu_token_cutoff > m.shape[-2] else None)
            offload_msa_module = self._do_inference_offload(seq_len=batch["token_mask"].shape[-1], module_name="msa_module")
            if offload_msa_module:
                input_tensors = [m, z]
                del m, z
                z = self.msa_module.forward_offload(input_tensors, msa_mask=msa_mask.to(dtype=input_tensors[0].dtype),
                                                    pair_mask=pair_mask.to(dtype=input_tensors[1].dtype), chunk_size=mode_mem_settings.chunk_size,
                                                    transition_ckpt_chunk_size=transition_ckpt_chunk_size,
                                                    use_deepspeed_evo_attention=mode_mem_settings.use_deepspeed_evo_attention,
                                                    use_triton_triangle_kernels=mode_mem_settings.use_triton_triangle_kernels,
                                                    use_cueq_triangle_kernels=mode_mem_settings.use_cueq_triangle_kernels,
                                                    use_lma=mode_mem_settings.use_lma, _mask_trans=True)
                del input_tensors, msa_mask
            else:
                z = self.msa_module(m, z, msa_mask=msa_mask.to(dtype=m.dtype), pair_mask=pair_mask.to(dtype=z.dtype),
                                    chunk_size=mode_mem_settings.chunk_size, transition_ckpt_chunk_size=transition_ckpt_chunk_size,
                                    use_deepspeed_evo_attention=mode_mem_settings.use_deepspeed_evo_attention,
                                    use_triton_triangle_kernels=mode_mem_settings.use_triton_triangle_kernels,
                                    use_cueq_triangle_kernels=mode_mem_settings.use_cueq_triangle_kernels,
                                    use_lma=mode_mem_settings.use_lma, inplace_safe=inplace_safe, _mask_trans=True)
                del m, msa_mask
            log("msa_module_done", cycle=cycle_no)
            s = s_init + self.linear_s(self.layer_norm_s(s))
            s, z = self.pairformer_stack(s=s, z=z, single_mask=token_mask.to(dtype=z.dtype), pair_mask=pair_mask.to(dtype=s.dtype),
                                       chunk_size=mode_mem_settings.chunk_size,
                                       use_deepspeed_evo_attention=mode_mem_settings.use_deepspeed_evo_attention,
                                       use_triton_triangle_kernels=mode_mem_settings.use_triton_triangle_kernels,
                                       use_cueq_triangle_kernels=mode_mem_settings.use_cueq_triangle_kernels,
                                       use_lma=mode_mem_settings.use_lma, inplace_safe=inplace_safe, _mask_trans=True)
        log("trunk_cycle_done", cycle=cycle_no, s=round(time.time() - tc, 1))
    del z_init_h
    free_pinned("z_init")
    free_pinned("trimul_snap")
    log("phase_end", phase="trunk", peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
    return s_input, s, z


# ----------------------------------------------------------------------------------------------- forward / rollout / heads orchestration
def model_forward(self, batch):
    from openfold3.core.utils.tensor_utils import tensor_tree_map
    inplace_safe = not (self.training or torch.is_grad_enabled())
    if not inplace_safe or "ground_truth" in batch:
        fallback("OpenFold3.forward", inplace_safe=bool(inplace_safe), ground_truth="ground_truth" in batch)
        return _ORIG["OpenFold3.forward"](self, batch)
    _lnsafe_scope(self)
    num_cycles = self.shared.num_recycles + 1
    output = {"recycles": self.shared.num_recycles}
    if env_int("OF3O_TEMPL_HOST", 1):
        template_feats_to_host(batch)
    si_input, si_trunk, zij_trunk = self.run_trunk(batch=batch, num_cycles=num_cycles, inplace_safe=inplace_safe)
    si_input = si_input.unsqueeze(1)
    si_trunk = si_trunk.unsqueeze(1)
    hz_attr = getattr(zij_trunk, "_of3o", None)
    zij_trunk = zij_trunk.unsqueeze(1)
    if hz_attr is not None:
        zij_trunk._of3o = hz_attr
    ref_space_uid_to_perm = batch.pop("ref_space_uid_to_perm", None)
    batch = tensor_tree_map(lambda t: t.unsqueeze(1), batch)
    batch["ref_space_uid_to_perm"] = ref_space_uid_to_perm
    holder = [zij_trunk]
    del zij_trunk
    output.update(rollout(self, batch, si_input, si_trunk, holder))
    if self.settings.clear_cache_between_steps:
        torch.cuda.empty_cache()
    return batch, output


def rollout(self, batch, si_input, si_trunk, holder):
    from openfold3.projects.of3_all_atom.model import create_noise_schedule
    mode_mem_settings = self._get_mode_mem_settings()
    offload_confidence_heads = self._do_inference_offload(seq_len=batch["token_mask"].shape[-1], module_name="confidence_heads")
    no_rollout_steps = self.shared.diffusion.no_full_rollout_steps
    no_rollout_samples = self.shared.diffusion.no_full_rollout_samples
    STATE["phase"] = "sampler"
    torch.cuda.reset_peak_memory_stats()
    log("phase_start", phase="sampler", steps=no_rollout_steps, samples=no_rollout_samples)
    t0 = time.time()
    dc = self.diffusion_module.diffusion_conditioning
    name_tuners(self)
    zij_trunk = holder.pop()
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float32):
        chunk = mode_mem_settings.chunk_size
        hz_trunk = getattr(zij_trunk, "_of3o", None)
        if dc.chunk_size_tuner is None and os.environ.get("OF3O_CHUNK", "").strip():
            chunk = max(int(chunk or 1), chunk_for("model.diffusion_module.diffusion_conditioning", 16))
        if env_int("OF3O_COND", 1) and chunk is not None and dc.chunk_size_tuner is not None:
            plan_env = os.environ.get("OF3O_CHUNK", "").strip()
            if plan_env:
                chunk = dc.chunk_size_tuner.tune_chunk_size(representative_fn=None, args=(si_trunk, zij_trunk), min_chunk_size=chunk, max_chunk_size=2048)
            else:
                # stock tunes with dc._forward on clones of the embedded (si, zij); reproduce that trial on same-shaped probes
                si_probe = torch.cat([si_trunk, si_input], dim=-1)
                si_probe = dc.linear_s(dc.layer_norm_s(si_probe))
                z_probe = torch.zeros(zij_trunk.shape[:-1] + (dc.c_z,), dtype=zij_trunk.dtype, device=zij_trunk.device)
                chunk = dc.chunk_size_tuner.tune_chunk_size(representative_fn=dc._forward, args=(si_probe, z_probe, batch["token_mask"]),
                                                            min_chunk_size=chunk, max_chunk_size=2048)
                del si_probe, z_probe
        if env_int("OF3O_COND", 1):
            COND["chunk"] = chunk
            zij = cond_pair_path(dc, batch, zij_trunk, chunk)
            log("cond_pair_cached", chunk=chunk, rows=env_int("OF3O_COND_ROWS", 0))
            zshape = zij_trunk.shape
            if hz_trunk is not None:
                zij_trunk_h = hz_trunk.h.view(zshape)                       # already host-resident (layer 2)
            else:
                zij_trunk_h = to_host(zij_trunk, "zij_trunk")
            del zij_trunk
            COND["zij"] = zij
            del zij
            z_arg = torch.zeros(zshape[:-3] + (1, 1, zshape[-1]), dtype=si_trunk.dtype, device=si_trunk.device)
        else:
            zij_trunk_h = None
            z_arg = zij_trunk
            del zij_trunk
        noise_schedule = create_noise_schedule(no_rollout_steps=no_rollout_steps, **self.config.architecture.noise_schedule,
                                               dtype=si_input.dtype, device=si_input.device)
        try:
            atom_positions_predicted = self.sample_diffusion(batch=batch, si_input=si_input, si_trunk=si_trunk, zij_trunk=z_arg,
                                                             noise_schedule=noise_schedule, no_rollout_samples=no_rollout_samples,
                                                             use_conditioning=True, chunk_size=mode_mem_settings.chunk_size,
                                                             use_deepspeed_evo_attention=mode_mem_settings.use_deepspeed_evo_attention,
                                                             use_triton_triangle_kernels=mode_mem_settings.use_triton_triangle_kernels,
                                                             use_cueq_triangle_kernels=mode_mem_settings.use_cueq_triangle_kernels,
                                                             use_lma=mode_mem_settings.use_lma, _mask_trans=True)
        finally:
            COND["zij"] = None
        if zij_trunk_h is None:
            zij_trunk_h = to_host(z_arg, "zij_trunk")
        del z_arg
        self.clear_autocast_cache()
    log("phase_end", phase="sampler", s=round(time.time() - t0, 1), peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
    STATE["phase"] = "confidence"
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    log("phase_start", phase="confidence")
    output = {"si_trunk": si_trunk, "zij_trunk": zij_trunk_h, "atom_positions_predicted": atom_positions_predicted}
    with torch.amp.autocast(device_type="cuda", dtype=si_trunk.dtype):
        if AUX_HEADS_IMPL["fn"] is not None:
            kw = dict(use_deepspeed_evo_attention=mode_mem_settings.use_deepspeed_evo_attention,
                      use_triton_triangle_kernels=mode_mem_settings.use_triton_triangle_kernels,
                      use_cueq_triangle_kernels=mode_mem_settings.use_cueq_triangle_kernels, use_lma=mode_mem_settings.use_lma)
            output.update(AUX_HEADS_IMPL["fn"](self.aux_heads, batch=batch, si_input=si_input, output=output, chunk_size=mode_mem_settings.chunk_size,
                                             kw=kw, inplace_safe=True, offload_inference=offload_confidence_heads))
        else:
            output.update(aux_heads_forward(self.aux_heads, batch=batch, si_input=si_input, output=output, chunk_size=mode_mem_settings.chunk_size,
                                            use_deepspeed_evo_attention=mode_mem_settings.use_deepspeed_evo_attention,
                                            use_triton_triangle_kernels=mode_mem_settings.use_triton_triangle_kernels,
                                            use_cueq_triangle_kernels=mode_mem_settings.use_cueq_triangle_kernels,
                                            use_lma=mode_mem_settings.use_lma, inplace_safe=True, offload_inference=offload_confidence_heads))
    log("phase_end", phase="confidence_heads", s=round(time.time() - t0, 1), peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
    free_pinned("zij_trunk")
    return output


def aux_heads_forward(heads, batch, si_input, output, chunk_size, use_deepspeed_evo_attention, use_cueq_triangle_kernels,
                      use_triton_triangle_kernels, use_lma, inplace_safe, offload_inference):
    """AuxiliaryHeadsAllAtom.forward with zij_trunk arriving from pinned host (the H2D copy plays the role of stock's .clone())
    and no second GPU copy of zij held while the head pairformer runs."""
    from openfold3.core.utils.atomize_utils import broadcast_token_feat_to_atoms, get_token_representative_atoms
    aux_out = {}
    out_dtype = output["atom_positions_predicted"].dtype
    si = output["si_trunk"]
    dev = si.device
    zij = output["zij_trunk"].to(dev, non_blocking=False)          # == stock's zij.detach().clone() values
    atom_positions_predicted = output["atom_positions_predicted"].to(dtype=si.dtype)
    dist_dev = "cpu" if offload_inference else dev
    aux_out["distogram_logits"] = heads.distogram(z=zij).to(dist_dev)
    si_input = si_input.detach().clone()
    si = si.detach().clone()
    atom_positions_predicted = atom_positions_predicted.detach().clone()
    token_mask = batch["token_mask"]
    pair_mask = token_mask[..., None] * token_mask[..., None, :]
    repr_x_pred, repr_x_mask = get_token_representative_atoms(batch=batch, x=atom_positions_predicted, atom_mask=batch["atom_mask"])
    num_samples = repr_x_pred.shape[-3]
    apply_per_sample = (not torch.is_grad_enabled() and num_samples > 1 and heads.per_sample_token_cutoff is not None
                        and repr_x_pred.shape[-2] > heads.per_sample_token_cutoff)
    pe = heads.pairformer_embedding
    if apply_per_sample:
        si, zij = pe(si_input=si_input, si=si, zij=zij, x_pred=repr_x_pred, single_mask=repr_x_mask, pair_mask=pair_mask, chunk_size=chunk_size,
                     use_deepspeed_evo_attention=use_deepspeed_evo_attention, use_cueq_triangle_kernels=use_cueq_triangle_kernels,
                     use_triton_triangle_kernels=use_triton_triangle_kernels, use_lma=use_lma, inplace_safe=inplace_safe,
                     offload_inference=offload_inference, _mask_trans=True, apply_per_sample=apply_per_sample)
    else:
        HANDOFF["zij"] = zij
        del zij
        si, zij = pairformer_emb_lean(pe, si_input=si_input, si=si, x_pred=repr_x_pred, single_mask=repr_x_mask, pair_mask=pair_mask,
                                      chunk_size=chunk_size, use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                                      use_cueq_triangle_kernels=use_cueq_triangle_kernels, use_triton_triangle_kernels=use_triton_triangle_kernels,
                                      use_lma=use_lma, inplace_safe=inplace_safe)
    max_atom_per_token_mask = broadcast_token_feat_to_atoms(token_mask=token_mask, num_atoms_per_token=batch["num_atoms_per_token"],
                                                           token_feat=token_mask, max_num_atoms_per_token=heads.max_atoms_per_token)
    max_atom_per_token_mask = max_atom_per_token_mask.expand((*atom_positions_predicted.shape[:-2], -1))
    si = si.to(device=dev)
    aux_out["plddt_logits"] = heads.plddt(s=si, max_atom_per_token_mask=max_atom_per_token_mask)
    aux_out["experimentally_resolved_logits"] = heads.experimentally_resolved(si, max_atom_per_token_mask)
    zij = zij.to(device=dev)
    pde_logits = heads.pde(zij, apply_per_sample=apply_per_sample)
    if heads.config.pae.enabled:
        offload_device = "cpu" if offload_inference else dev
        pde_logits = pde_logits.to(device=offload_device)
        aux_out["pae_logits"] = heads.pae(zij, apply_per_sample=apply_per_sample)
    del zij
    aux_out["pde_logits"] = pde_logits.to(device=dev)
    aux_out["distogram_logits"] = aux_out["distogram_logits"].to(device=dev)
    aux_out = {k: v.to(dtype=out_dtype) for k, v in aux_out.items()}
    return aux_out


def embed_zij_rows(pe, si_input, zij, x_pred):
    """PairformerEmbedding.embed_zij evaluated IN PLACE per block of token rows: zij[r] = ((zij[r] + li) + lj) + linear_distance(onehot(d_r)).
    Same elementwise add order as stock; linear_distance acts on a one-hot (single non-zero term per output element)."""
    rows = env_int("OF3O_ROWS", 256)
    orig_dtype = zij.dtype
    with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
        li = pe.linear_i(si_input.unsqueeze(-2))          # [*, N, 1, C]
        lj = pe.linear_j(si_input.unsqueeze(-3))          # [*, 1, N, C]
        bins = torch.linspace(pe.min_bin, pe.max_bin, pe.no_bin, device=zij.device, dtype=zij.dtype)
        squared_bins = bins**2
        upper = torch.cat([squared_bins[1:], squared_bins.new_tensor([pe.inf])], dim=-1)
        N = zij.shape[-2]
        if rows <= 0:
            rows = N
        for r0 in range(0, N, rows):
            r1 = min(N, r0 + rows)
            zr = zij[..., r0:r1, :, :]
            zr += li[..., r0:r1, :, :]
            zr += lj
        target = tuple(torch.broadcast_shapes(tuple(zij.shape), (*x_pred.shape[:-2], *zij.shape[-3:])))
        if tuple(zij.shape) != target:
            # stock's `zij + linear_distance(dij)` broadcasts zij ([*, 1, N, N, C]) over x_pred's sample axis (below the per-sample token cutoff
            # every sample shares one zij): materialise that broadcast once, then add the per-sample distance term in place per row block (the
            # same elementwise adds as stock's; the port's modification — the add-on as shipped ran one diffusion sample per query)
            zij = zij.expand(*target).clone()
        for r0 in range(0, N, rows):
            r1 = min(N, r0 + rows)
            zr = zij[..., r0:r1, :, :]
            dij = torch.sum((x_pred[..., r0:r1, None, :] - x_pred[..., None, :, :]) ** 2, dim=-1, keepdims=True)
            dij = ((dij > squared_bins) * (dij < upper)).type(x_pred.dtype)
            zr += pe.linear_distance(dij)
            del dij
    return zij.to(dtype=orig_dtype)


def pairformer_emb_lean(pe, si_input, si, x_pred, single_mask, pair_mask, chunk_size, use_deepspeed_evo_attention,
                        use_cueq_triangle_kernels, use_triton_triangle_kernels, use_lma, inplace_safe):
    """PairformerEmbedding.pairformer_emb (stock statements) with embed_zij done in place on the H2D copy taken from HANDOFF,
    so only ONE [N,N,C] tensor is resident while the 4-block head pairformer runs."""
    zij = HANDOFF.pop("zij")
    if env_int("OF3O_EMBED_ROWS", 1):
        zij = embed_zij_rows(pe, si_input, zij, x_pred)
    else:
        zij = pe.embed_zij(si_input=si_input, zij=zij, x_pred=x_pred)
    batch_dims = x_pred.shape[:-2]

    def reshape_inputs(x, feat_dims):
        x = x.expand(*(batch_dims + feat_dims))
        return x.reshape(-1, *feat_dims)

    def reshape_outputs(x, feat_dims):
        return x.reshape(*batch_dims, *feat_dims)

    si = reshape_inputs(si, si.shape[-2:]).clone()
    zij = reshape_inputs(zij, zij.shape[-3:])
    single_mask = reshape_inputs(single_mask, single_mask.shape[-1:])
    pair_mask = reshape_inputs(pair_mask, pair_mask.shape[-2:])
    use_kernels = use_deepspeed_evo_attention or use_cueq_triangle_kernels or use_triton_triangle_kernels
    if use_kernels and si.shape[0] > 1:
        chunk_size = None                                   # stock's own rule (prediction_heads.py: the DS kernel with chunk tuning and several samples)
    STATE["conf_head_unchunked"] = chunk_size is None       # the port: the head's pair blocks then run stock's unchunked path by stock's rule — counted, not a fallback
    try:
        si, zij = pe.pairformer_stack(si, zij, single_mask, pair_mask, chunk_size=chunk_size, use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                                     use_cueq_triangle_kernels=use_cueq_triangle_kernels, use_triton_triangle_kernels=use_triton_triangle_kernels,
                                     use_lma=use_lma, inplace_safe=inplace_safe, _mask_trans=True)
    finally:
        STATE["conf_head_unchunked"] = False
    si = reshape_outputs(si, si.shape[-2:])
    zij = reshape_outputs(zij, zij.shape[-3:])
    return si, zij


# ----------------------------------------------------------------------------------------------- confidence scoring on CPU (fallback)
def wrap_confidence_scores_cpu(cls):
    """Confidence scoring modes (OF3O_CONF_MODE): 'auto' (default) = stock below OF3O_CONF_MIN tokens (default 3000), else the
    row-chunked scorer (of3o_confidence, OF3 adapter on BLOCKREDUCE v0, of3o_blockreduce.py; logits may be host-resident);
    'chunked' = always the row-chunked scorer; 'cpu' = stock functions on CPU copies (values differ from GPU at rounding level);
    'stock' = untouched."""
    orig = cls._compute_confidence_scores

    def _per_sample(self, outputs):
        num_samples = self.config.architecture.shared.diffusion.no_full_rollout_samples
        num_atoms = outputs["atom_positions_predicted"].shape[-2]
        return (num_samples > 1 and self.config.settings.memory.eval.per_sample_atom_cutoff is not None
                and num_atoms > self.config.settings.memory.eval.per_sample_atom_cutoff)

    def scores(self, batch, outputs):
        n = int(batch["token_mask"].shape[-1])
        mode = os.environ.get("OF3O_CONF_MODE", "auto").strip() or "auto"
        thr = env_int("OF3O_CONF_MIN", 3000)
        logits_on_cpu = any(isinstance(outputs.get(k), torch.Tensor) and not outputs[k].is_cuda for k in ("pae_logits", "pde_logits", "distogram_logits"))
        STATE["conf"] = {"mode": mode, "n_tok": n, "min": thr, "path": "stock" if (mode == "stock" or (mode == "auto" and n < thr and not logits_on_cpu)) else mode if mode in ("cpu", "chunked") else "chunked"}
        if mode == "stock" or (mode == "auto" and n < thr and not logits_on_cpu):
            return orig(self, batch, outputs)
        t0 = time.time()
        dev = outputs["atom_positions_predicted"].device
        if mode == "cpu":
            cpu_out = {k: (v.to("cpu") if isinstance(v, torch.Tensor) else v) for k, v in outputs.items()}
            cpu_batch = {k: (v.to("cpu") if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            for k in ("pae_logits", "pde_logits", "distogram_logits"):
                if k in outputs and isinstance(outputs[k], torch.Tensor) and outputs[k].is_cuda:
                    outputs[k] = cpu_out[k]
            torch.cuda.empty_cache()
            r = orig(self, cpu_batch, cpu_out)
            r = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in r.items()}
            log("confidence_scores", mode="cpu", n_tok=n, s=round(time.time() - t0, 1))
            return r
        import of3o_confidence as OC
        r = OC.get_confidence_scores_chunked(batch=batch, outputs=outputs, config=self.config, compute_per_sample=_per_sample(self, outputs),
                                             rows=env_int("OF3O_CONF_ROWS", 128), logits_device=dev,
                                             tm_backend=os.environ.get("OF3O_CONF_TM_BACKEND", "blockreduce").strip() or "blockreduce",   # explicit: BLOCKREDUCE absent -> ImportError, never native
                                             tm_finish="exact", gpde_mode="exact")
        log("confidence_scores", mode="chunked", n_tok=n, s=round(time.time() - t0, 1), blockreduce=bool(getattr(OC, "HAVE_BLOCKREDUCE", False)))
        return r

    cls._compute_confidence_scores = scores


# ----------------------------------------------------------------------------------------------- O-TEMPL
TEMPLATE_KEYS = ("template_restype", "template_pseudo_beta_mask", "template_backbone_frame_mask", "template_distogram", "template_unit_vector")


def template_feats_to_host(batch):
    """Move the [*, N_templ, N, N, *] template features off the GPU for the whole prediction (only the template
    embedder reads them; O-TEMPL brings one template slice back at a time)."""
    moved = {}
    for k in TEMPLATE_KEYS:
        v = batch.get(k)
        if isinstance(v, torch.Tensor) and v.is_cuda:
            batch[k] = v.to("cpu")
            moved[k] = tuple(v.shape)
    if moved:
        torch.cuda.empty_cache()
        log("template_feats_to_host", shapes=moved)


def _template_slice(batch, i, dev):
    out = {"asym_id": batch["asym_id"]}
    for k in TEMPLATE_KEYS:
        v = batch[k]
        out[k] = v[:, i: i + 1].to(dev, non_blocking=False)   # template dim is dim 1 for [B, N_templ, ...]
    return out




def embed_feats_rows(tpe, feats, r0, r1, dev):
    """TemplatePairEmbedder._embed_feats for token rows r0:r1 of ONE template slice (feats on host or device; same statement
    order as stock; the eight small linears see M = rows*N instead of N*N -> U3-tested)."""
    dtype = feats["template_unit_vector"].dtype
    asym = feats["asym_id"].to(dev)
    g = lambda k: feats[k].to(dev, non_blocking=True)  # noqa: E731  ([*, T, N] token-level features are small)
    mpm = (asym[..., r0:r1, None] == asym[..., None, :])[..., None, :, :, None]                 # [*, 1, R, N, 1]
    tpb = g("template_pseudo_beta_mask")
    pbm = (tpb[..., r0:r1, None] * tpb[..., None, :])[..., None] * mpm
    dgram = feats["template_distogram"][..., r0:r1, :, :].to(dev, non_blocking=True)
    tbf = g("template_backbone_frame_mask")
    bfm = (tbf[..., r0:r1, None] * tbf[..., None, :])[..., None] * mpm
    uv = feats["template_unit_vector"][..., r0:r1, :, :].to(dev, non_blocking=True)
    x, y, z = uv.unbind(dim=-1)
    rt = g("template_restype")
    n_token = rt.shape[-2]
    ti = rt[..., r0:r1, None, :].expand(*rt.shape[:-2], r1 - r0, n_token, -1)
    tj = rt[..., None, :, :].expand(*rt.shape[:-2], r1 - r0, -1, -1)
    a = tpe.dgram_linear(dgram)
    a = a + tpe.pseudo_beta_mask_linear(pbm)
    a = a + tpe.aatype_linear_1(ti.to(dtype=dtype))
    a = a + tpe.aatype_linear_2(tj.to(dtype=dtype))
    a = a + tpe.x_linear(x[..., None])
    a = a + tpe.y_linear(y[..., None])
    a = a + tpe.z_linear(z[..., None])
    a = a + tpe.backbone_mask_linear(bfm)
    return a


def build_v(tpe, u, feats, dev):
    """v = u[..., None, :, :, :] + _embed_feats(template slice): full (stock statements) or per row block (OF3O_TEMPL_EMBED_ROWS)."""
    erows = env_int("OF3O_TEMPL_EMBED_ROWS", env_int("OF3O_ROWS", 128))
    if erows <= 0:
        feats_dev = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in feats.items()}
        a = tpe._embed_feats(batch=feats_dev)
        del feats_dev
        v = u[..., None, :, :, :] + a
        return v
    N = u.shape[-2]
    v = torch.empty(u.shape[:-3] + (1,) + u.shape[-3:], dtype=u.dtype, device=dev)
    for r0 in range(0, N, erows):
        r1 = min(N, r0 + erows)
        a = embed_feats_rows(tpe, feats, r0, r1, dev)
        v[..., :, r0:r1, :, :] = u[..., None, r0:r1, :, :] + a
        del a
    return v


def template_embedder_forward(self, batch, z, pair_mask, chunk_size=None, _mask_trans=True, use_deepspeed_evo_attention=False,
                              use_cueq_triangle_kernels=False, use_triton_triangle_kernels=False, use_lma=False, inplace_safe=False,
                              _of3o_accum=False):
    """TemplateEmbedderAllAtom.forward one template at a time: v_i = linear_z(LN(z)) + embed_feats(template i); pair stack + LN
    per template exactly as stock does per template inside each block; per-template outputs staged on the host; the template
    mean uses torch.sum over the stacked templates per row block (same reduction op as stock's torch.sum(dim=-4)).
    OF3O_TEMPL_ROWS>0: u = linear_z(LN(z)) and the final linear_t are evaluated per row block (GEMM M change: U3-tested).
    _of3o_accum=True (used by the add-on's run_trunk): the result is added into z in place and None is returned."""
    if not inplace_safe or LAYER < 1:
        fallback("TemplateEmbedderAllAtom.forward", inplace_safe=bool(inplace_safe), layer=LAYER)
        return _ORIG["TemplateEmbedderAllAtom.forward"](self, batch, z, pair_mask, chunk_size, _mask_trans, use_deepspeed_evo_attention,
                                                        use_cueq_triangle_kernels, use_triton_triangle_kernels, use_lma, inplace_safe)
    tpe, tps = self.template_pair_embedder, self.template_pair_stack
    dev = z.device
    n_templ = batch["template_restype"].shape[1]
    N = z.shape[-2]
    rows = env_int("OF3O_ROWS", 256) or N
    trows = env_int("OF3O_TEMPL_ROWS", 0)
    t0 = time.time()
    c_t = tpe.linear_z.weight.shape[0]
    if trows:
        u = torch.empty(z.shape[:-1] + (c_t,), device=dev, dtype=z.dtype)
        for r0 in range(0, N, trows):
            r1 = min(N, r0 + trows)
            u[..., r0:r1, :, :] = tpe.linear_z(tpe.layer_norm_z(z[..., r0:r1, :, :]))
    else:
        u = tpe.linear_z(tpe.layer_norm_z(z))                 # [*, N, N, C_t] — full GEMM as stock
    pm = pair_mask[..., None, :, :].to(dtype=z.dtype)         # [*, 1, N, N]
    # distinct template slices (the 4 dummy slots of a template-free query are bitwise identical -> computed once)
    slices = [_template_slice(batch, i, "cpu") for i in range(n_templ)]
    rep = []
    for i in range(n_templ):
        r_i = i
        if env_int("OF3O_TEMPL_DEDUP", 1):
            for j in range(i):
                if rep[j] == j and all(torch.equal(slices[i][k], slices[j][k]) for k in TEMPLATE_KEYS):
                    r_i = j
                    break
        rep.append(r_i)
    distinct = [i for i in range(n_templ) if rep[i] == i]
    staged_by = {}
    for i in distinct:
        v = build_v(tpe, u, slices[i], dev)                    # u + embedded template features (full or per row block)
        if i == distinct[-1]:
            del u
        v = tps(v, pm, chunk_size=chunk_size, use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cueq_triangle_kernels=use_cueq_triangle_kernels, use_triton_triangle_kernels=use_triton_triangle_kernels,
                use_lma=use_lma, inplace_safe=inplace_safe, _mask_trans=_mask_trans)   # blocks + final LN, one template
        staged_by[i] = to_host(v, "templ_%d" % i)
        del v
    staged = [staged_by[rep[i]] for i in range(n_templ)]
    torch.cuda.empty_cache()
    out = None
    if not (trows and _of3o_accum):
        T = torch.empty(z.shape[:-1] + (staged[0].shape[-1],), device=dev, dtype=z.dtype)
    for r0 in range(0, N, rows):
        r1 = min(N, r0 + rows)
        stk = torch.cat([h[..., :, r0:r1, :, :].to(dev) for h in staged], dim=-4)   # [*, n_templ, rows, N, C_t]
        Tr = torch.sum(stk, dim=-4) / n_templ
        del stk
        if trows and _of3o_accum:
            Tr = torch.nn.functional.relu(Tr)
            z[..., r0:r1, :, :] += self.linear_t(Tr)
        else:
            T[..., r0:r1, :, :] = Tr
        del Tr
    if not (trows and _of3o_accum):
        T = torch.nn.functional.relu(T)
        if trows:
            out = torch.empty_like(z)
            for r0 in range(0, N, trows):
                r1 = min(N, r0 + trows)
                out[..., r0:r1, :, :] = self.linear_t(T[..., r0:r1, :, :])
        else:
            out = self.linear_t(T)                                 # full GEMM as stock
        del T
        if _of3o_accum:
            z += out
            out = None
    for i in distinct:
        free_pinned("templ_%d" % i)
    log("template_embedder_done", n_templ=int(n_templ), distinct=len(distinct), rows=trows, accum=bool(_of3o_accum), s=round(time.time() - t0, 1))
    return out


# ----------------------------------------------------------------------------------------------- progress / abort probe
def wrap_pairformer_block():
    from openfold3.core.model.latent import pairformer as PF
    orig = PF.PairFormerBlock.forward
    every = 8                                                                  # the pairformer-block log cadence (and the first 3 blocks)
    counter = {"n": 0}

    def fwd(self, *a, **k):
        t0 = time.time()
        out = orig(self, *a, **k)
        if not torch.is_grad_enabled():
            torch.cuda.synchronize()
            dt = time.time() - t0
            counter["n"] += 1
            STATE["block_times"].append(round(dt, 3))
            if counter["n"] % every == 0 or counter["n"] <= 3:
                log("pairformer_block", n=counter["n"], s=round(dt, 3), phase=STATE["phase"])
        return out

    PF.PairFormerBlock.forward = fwd


# ----------------------------------------------------------------------------------------------- NUMX diagnostics (logging only)
def _chunk_stats(t, rows=128):
    """fp64 mean/std/absmax + nan/inf counts of a tensor, evaluated per leading-row chunk (never a >= 2^31-element kernel)."""
    dim = next((i for i, n_ in enumerate(t.shape[:-1]) if n_ > 1), 0)
    n = t.shape[dim]
    s1 = 0.0; s2 = 0.0; amax = 0.0; nnan = 0; ninf = 0; per_chunk = []; first_nan = None
    for i in range(0, n, rows):
        c = t.narrow(dim, i, min(rows, n - i))
        cf = c.float() if c.dtype != torch.float32 else c
        nan_c = int(torch.isnan(cf).sum().item()); inf_c = int(torch.isinf(cf).sum().item())
        fin = torch.nan_to_num(cf, nan=0.0, posinf=0.0, neginf=0.0) if (nan_c or inf_c) else cf
        s1 += float(torch.sum(fin, dtype=torch.float64).item()); s2 += float(torch.sum(fin * fin, dtype=torch.float64).item())
        am = float(fin.abs().amax().item()); amax = max(amax, am); per_chunk.append(round(am, 4))
        nnan += nan_c; ninf += inf_c
        if (nan_c or inf_c) and first_nan is None:
            first_nan = i
    numel = t.numel(); mean = s1 / numel; var = max(s2 / numel - mean * mean, 0.0)
    return {"mean": round(mean, 6), "std": round(var ** 0.5, 6), "absmax": round(amax, 4), "n_nan": nnan, "n_inf": ninf, "numel": int(numel),
            "shape": [int(d) for d in t.shape], "first_bad_row": first_nan, "row_chunk_absmax": per_chunk}


def _find_pair_single(objs):
    z = s = None
    for o in objs:
        if isinstance(o, torch.Tensor) and o.is_cuda and o.dim() >= 3 and o.shape[-2] == o.shape[-3] and o.shape[-2] >= 64:
            if z is None or o.numel() > z.numel():
                z = o
    for o in objs:
        if isinstance(o, torch.Tensor) and o.is_cuda and o is not z and o.dim() >= 2 and z is not None and o.shape[-2] == z.shape[-2] and o.shape[-1] != z.shape[-2]:
            if s is None or o.numel() > s.numel():
                s = o
    return z, s


def wrap_predict_step():
    from openfold3.projects.of3_all_atom import runner as R
    cls = None
    cands = [getattr(R, n) for n in dir(R)]
    cands = [o for o in cands if isinstance(o, type) and getattr(o, "__module__", "") == R.__name__
             and "predict_step" in o.__dict__ and "_compute_confidence_scores" in o.__dict__]
    if cands:
        cls = cands[0]
    log("runner_class", cls=None if cls is None else cls.__name__)
    if cls is None:
        log("warn", msg="runner class with predict_step not found; ckpt naming disabled")
        return
    orig = cls.predict_step
    orig_scores = cls._compute_confidence_scores

    def predict_step(self, batch, batch_idx):
        try:
            q = batch.get("query_id")
            STATE["query_id"] = "_".join(q) if isinstance(q, (list, tuple)) else str(q)
            sd = batch.get("seed")
            STATE["seed"] = int(sd.flatten()[0].item()) if isinstance(sd, torch.Tensor) else (sd[0] if isinstance(sd, (list, tuple)) else sd)
        except Exception as e:  # noqa
            log("warn", msg="query/seed capture failed: %s" % e)
        STATE["block_times"] = []
        try:
            name_tuners(self)
        except Exception as e:  # noqa
            log("warn", msg="tuner naming failed: %s" % e)
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        log("predict_start", query=STATE["query_id"], seed=STATE["seed"])
        out = orig(self, batch, batch_idx)
        log("ln_tally", lnsafe_calls=STATE.get("lnsafe_calls", 0), lnsafe_sizes=STATE.get("lnsafe_sizes", {}), ln_big=STATE.get("ln_big", {}),
            ln_ge2p32_calls=STATE.get("ln_ge2p32_calls", 0), guard_on=bool(env_int("OF3O_LNSAFE", 1)))
        log("predict_end", query=STATE["query_id"], seed=STATE["seed"], s=round(time.time() - t0, 1), ok=out is not None,
            tuned=STATE["tuned"], n_tok=STATE["n_tok"])
        return out

    def scores(self, batch, outputs):
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        r = orig_scores(self, batch, outputs)
        log("phase_end", phase="confidence_scores", s=round(time.time() - t0, 1), peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        return r

    cls.predict_step = predict_step
    cls._compute_confidence_scores = scores

    # keep the (dummy) template pair features on the host: stock moves [*, 4, N, N, ~44] fp32 to the GPU with the batch
    # although only the template embedder reads them (O-TEMPL slices them per template).
    if LAYER >= 1 and env_int("OF3O_TEMPL_HOST", 1) and hasattr(cls, "transfer_batch_to_device"):
        orig_tb = cls.transfer_batch_to_device

        def transfer_batch_to_device(self, batch, device, dataloader_idx):
            held = {}
            if isinstance(batch, dict):
                for k in TEMPLATE_KEYS:
                    if k in batch and isinstance(batch[k], torch.Tensor):
                        held[k] = batch.pop(k)
            out = orig_tb(self, batch, device, dataloader_idx)
            if held:
                for k, v in held.items():
                    out[k] = v.pin_memory() if not v.is_pinned() else v
                log("template_feats_kept_on_host", shapes={k: tuple(v.shape) for k, v in held.items()},
                    gb=round(sum(v.numel() * v.element_size() for v in held.values()) / 2**30, 2))
            return out

        cls.transfer_batch_to_device = transfer_batch_to_device
    if LAYER >= 1:
        wrap_confidence_scores_cpu(cls)




# ----------------------------------------------------------------------------------------------- stock-arm phase timers (LAYER 0; no numeric change)
def wrap_stock_phases(M):
    orig_rt, orig_ro = M.OpenFold3.run_trunk, M.OpenFold3._rollout

    def run_trunk_timed(self, batch, num_cycles, inplace_safe=False):
        STATE["phase"] = "trunk"
        STATE["n_tok"] = int(batch["token_mask"].shape[-1])
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        log("phase_start", phase="trunk", n_tok=STATE["n_tok"], n_atom=int(batch["atom_mask"].shape[-1]) if "atom_mask" in batch else None, num_cycles=num_cycles)
        out = orig_rt(self, batch, num_cycles, inplace_safe=inplace_safe)
        torch.cuda.synchronize()
        log("phase_end", phase="trunk", s=round(time.time() - t0, 1), peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        return out

    def rollout_timed(self, *a, **k):
        STATE["phase"] = "rollout"
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        out = orig_ro(self, *a, **k)
        torch.cuda.synchronize()
        log("phase_end", phase="rollout(diffusion+heads)", s=round(time.time() - t0, 1), peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        return out

    M.OpenFold3.run_trunk = run_trunk_timed
    M.OpenFold3._rollout = rollout_timed



# ----------------------------------------------------------------------------------------------- LN-SAFE (kernel bug guard)
# torch (2.10.0+cu128 in the OF3 image) nn.functional.layer_norm returns WRONG values for inputs with numel > 2^32
# (rows beyond element 2^32 are garbage; checked on B200: N=5,900 x 5,900 x 128 -> 160,712,704 wrong elements, first bad
# row 2^25; N=4,600 exact).  Every LayerNorm over the full pair tensor [N, N, 128] is affected for N >= 5,793 tokens.
# Guard: evaluate LayerNorm per leading-dim chunk when numel >= 2^31 (LN is per position -> bitwise identical to a correct
# full-tensor kernel; unit test U_lnsafe compares against chunked evaluation at small size).  Always on (OF3O_LNSAFE=0 disables).
_LN_LIMIT = 2 ** 31


def _ln_tally(x, guard_on):
    """Count LayerNorm calls over big tensors (NUMX diag; logging only)."""
    key = "x".join(str(int(d)) for d in x.shape)
    tal = STATE.setdefault("ln_big", {})
    tal[key] = tal.get(key, 0) + 1
    if x.numel() >= 2 ** 32:
        STATE["ln_ge2p32_calls"] = STATE.get("ln_ge2p32_calls", 0) + 1
    if tal[key] == 1:
        log("ln_big_first", shape=list(x.shape), numel=int(x.numel()), ge_2p32=bool(x.numel() >= 2 ** 32), guard_on=bool(guard_on), dtype=str(x.dtype))


def _lnsafe_scope(model):
    """Mark every LayerNorm module of this model as in scope for the guard (the wrapped forward is a pass-through for any other instance:
    the guard reaches no LayerNorm outside this model's modules)."""
    n = 0
    for m in model.modules():
        if isinstance(m, torch.nn.LayerNorm) or type(m).__name__ == "LayerNorm":
            if not getattr(m, "_of3o_lnsafe_scope", False):
                m._of3o_lnsafe_scope = True
                n += 1
    if n:
        STATE["lnsafe_scoped"] = STATE.get("lnsafe_scoped", 0) + n
        log("lnsafe_scoped", modules=n)


def _ln_safe_forward(orig_forward):
    def forward(self, x):
        if not getattr(self, "_of3o_lnsafe_scope", False):
            return orig_forward(self, x)
        if x.numel() >= _LN_LIMIT:
            try:
                _ln_tally(x, env_int("OF3O_LNSAFE", 1))
            except Exception as e:  # noqa
                fallback("ln_tally", error=repr(e)[:200])                        # the LNSAFE evidence is incomplete: a named event (PARTIAL), never a warn alone
        if x.numel() < _LN_LIMIT or x.dim() < 2 or not env_int("OF3O_LNSAFE", 1):
            return orig_forward(self, x)
        dim = next((i for i, n_ in enumerate(x.shape[:-1]) if n_ > 1), None)
        if dim is None:
            return orig_forward(self, x)
        n = x.shape[dim]
        per = max(1, int((_LN_LIMIT // 8) * n // x.numel()))      # <= 2^28 elements (1 GiB fp32) per chunk
        out = torch.empty_like(x)
        for i in range(0, n, per):
            k = min(per, n - i)
            out.narrow(dim, i, k).copy_(orig_forward(self, x.narrow(dim, i, k)))
        STATE["lnsafe_calls"] = STATE.get("lnsafe_calls", 0) + 1
        sz = STATE.setdefault("lnsafe_sizes", {}); key = "x".join(str(int(d)) for d in x.shape); sz[key] = sz.get(key, 0) + 1
        if sz[key] == 1:
            log("lnsafe_first", shape=list(x.shape), numel=int(x.numel()), chunks=int((n + per - 1) // per), per=int(per))
        return out
    return forward


def apply_lnsafe():
    try:
        from openfold3.core.model.primitives import normalization as NZ
        if not getattr(NZ.LayerNorm, "_of3o_lnsafe", False):
            NZ.LayerNorm.forward = _ln_safe_forward(NZ.LayerNorm.forward)
            NZ.LayerNorm._of3o_lnsafe = True
    except Exception as e:  # noqa
        log("lnsafe_patch_failed", target="openfold3 LayerNorm", err=repr(e)[:200])
        raise RuntimeError("[of3o] LNSAFE could not patch openfold3's LayerNorm: %r" % (e,))
    if not getattr(torch.nn.LayerNorm, "_of3o_lnsafe", False):
        torch.nn.LayerNorm.forward = _ln_safe_forward(torch.nn.LayerNorm.forward)
        torch.nn.LayerNorm._of3o_lnsafe = True
    log("applied_lnsafe", limit=_LN_LIMIT)




# ----------------------------------------------------------------------------------------------- TEMPL-FIX (stock 0.4.1 inference bug)
# OpenFold3 0.4.1's InferenceDataset calls process_template_structures_of3(..., template_cache_directory=None, ...) and
# sample_templates() returns {} whenever template_cache_directory is None (the inference branch that reads
# chain_data["cache_entry_file_path"] sits inside `if k > 0 and template_cache_directory is not None`), so
# `run_openfold predict --use-templates true` silently featurises DUMMY templates for every chain.  Fix (data plumbing only,
# no arithmetic): when template_cache_directory is None and the chain carries cache_entry_file_path, call the stock function
# with a placeholder directory so its own inference branch is taken.  OF3O_TEMPL_FIX=0 disables.  Logged per call.
def apply_templ_fix():
    if not env_int("OF3O_TEMPL_FIX", 1):
        return
    try:
        from pathlib import Path
        from openfold3.core.data.primitives.structure import template as ST
        from openfold3.core.data.pipelines.sample_processing import template as SPT
        orig = ST.sample_templates

        def sample_templates_fixed(assembly_data, template_cache_directory, n_templates, take_top_k, chain_id,
                                   template_structure_array_directory, template_file_format, use_roda_monomer_format=False, **kw):
            tcd = template_cache_directory
            cd = assembly_data.get(chain_id, {}) if isinstance(assembly_data, dict) else {}
            if tcd is None and isinstance(cd, dict) and cd.get("cache_entry_file_path") and "alignment_representative_id" not in cd:
                tcd = Path("/nonexistent_of3o_placeholder")
            out = orig(assembly_data=assembly_data, template_cache_directory=tcd, n_templates=n_templates, take_top_k=take_top_k,
                       chain_id=chain_id, template_structure_array_directory=template_structure_array_directory,
                       template_file_format=template_file_format, use_roda_monomer_format=use_roda_monomer_format, **kw)
            STATE["templ_fix_calls"] = STATE.get("templ_fix_calls", 0) + 1
            STATE["templ_fix_sampled"] = STATE.get("templ_fix_sampled", 0) + len(out)
            if STATE["templ_fix_calls"] <= 3 or len(out) == 0:
                log("templ_fix_sample", chain=str(chain_id), n_sampled=len(out), ids=[str(k) for k in list(out)[:4]],
                    cache_dir_was_none=template_cache_directory is None)
            return out

        ST.sample_templates = sample_templates_fixed
        if getattr(SPT, "sample_templates", None) is orig:
            SPT.sample_templates = sample_templates_fixed
        log("applied_templ_fix")
    except Exception as e:  # noqa
        log("templ_fix_patch_failed", err=repr(e)[:300])
        raise RuntimeError("[of3o] OF3O_TEMPL_FIX=1 was requested and the patch failed: %r" % (e,))


# ----------------------------------------------------------------------------------------------- install
_ORIG = {}
_APPLIED = False


def apply_core():
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True
    from openfold3.core.model.latent import base_blocks as BB
    from openfold3.core.model.layers import triangular_multiplicative_update as TM
    from openfold3.core.model.layers import diffusion_conditioning as DCm
    from openfold3.core.model.feature_embedders import input_embedders as IE
    from openfold3.projects.of3_all_atom import model as M

    apply_lnsafe()
    apply_templ_fix()
    patch_tuner()
    wrap_pairformer_block()
    _ORIG["PairBlock.forward"] = BB.PairBlock.forward
    _ORIG["PairBlock.tri_att_start_end"] = BB.PairBlock.tri_att_start_end
    _ORIG["TMU._inference_forward"] = TM.TriangleMultiplicativeUpdate._inference_forward
    _ORIG["DiffusionConditioning.forward"] = DCm.DiffusionConditioning.forward
    _ORIG["InputEmbedderAllAtom.forward"] = IE.InputEmbedderAllAtom.forward
    from openfold3.core.model.latent import template_module as TPL
    _ORIG["TemplateEmbedderAllAtom.forward"] = TPL.TemplateEmbedderAllAtom.forward
    _ORIG["OpenFold3.run_trunk"] = M.OpenFold3.run_trunk
    _ORIG["OpenFold3.forward"] = M.OpenFold3.forward
    if LAYER == 0:
        wrap_stock_phases(M)
    if LAYER >= 1:
        if env_int("OF3O_TRIMUL", 1):
            TM.TriangleMultiplicativeUpdate._inference_forward = trimul_inference_forward
        if env_int("OF3O_TRIATT", 1):
            BB.PairBlock.tri_att_start_end = pairblock_tri_att_start_end
        if env_int("OF3O_TRANS", 1):
            BB.PairBlock.forward = pairblock_forward
        if env_int("OF3O_COND", 1):
            DCm.DiffusionConditioning.forward = diffusion_conditioning_forward
        if env_int("OF3O_INPUT", 1):                                     # the port: each O1 install is a switched unit (composable on another hook chain)
            IE.InputEmbedderAllAtom.forward = input_embedder_forward
        if env_int("OF3O_TEMPL", 1):
            TPL.TemplateEmbedderAllAtom.forward = template_embedder_forward
        if env_int("OF3O_RUN_TRUNK", 1):
            M.OpenFold3.run_trunk = run_trunk
        if env_int("OF3O_FORWARD", 1):
            prior = M.OpenFold3.forward                                        # NEED 3: the binding this install supersedes, recorded by name
            STATE["forward_prior"] = f"{getattr(prior, '__module__', '?')}.{getattr(prior, '__qualname__', getattr(prior, '__name__', '?'))}"
            if not getattr(prior, "__module__", "").startswith("openfold3."):
                STATE["forward_superseded"] = STATE["forward_prior"]
                log("forward_superseded", prior=STATE["forward_prior"], note="a hook's OpenFold3.forward wrapper installed before the offload hook is superseded by model_forward (the offload hook fires last on the model module)")
            M.OpenFold3.forward = model_forward
    if LAYER >= 2:
        raise RuntimeError("OF3O_LAYER=2 (the pair representation streamed through host rows) is not part of this tree; the big mode runs OF3O_LAYER=1")
    log("applied_core", layer=LAYER, env={k: v for k, v in os.environ.items()
                                          if k.startswith("OF3O_") or k in ("PYTORCH_CUDA_ALLOC_CONF", "OF3_DETERMINISTIC", "CUBLAS_WORKSPACE_CONFIG")})


_APPLIED_RUNNER = False


def apply_runner():
    global _APPLIED_RUNNER
    if _APPLIED_RUNNER:
        return
    _APPLIED_RUNNER = True
    wrap_predict_step()
    log("applied_runner", layer=LAYER)
