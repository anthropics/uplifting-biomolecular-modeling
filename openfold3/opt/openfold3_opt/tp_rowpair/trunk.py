"""OpenFold3.run_trunk (``openfold3/projects/of3_all_atom/model.py:172-316``) on row shards. The recycling loop with the pair representation
born sharded is the core's (``opt_core.mem.rowpair.trunk.run_trunk_sharded``); this module supplies OpenFold3's statements as its callables:
the z_init ROW statement (``init_rows_fn``: outer sum of ``linear_z_i / linear_z_j(s_input)`` + ``linear_relpos`` of the relpos one-hot rows +
``linear_token_bonds`` of the ``token_bonds`` rows, ``input_embedders.py:151-166`` — the pair representation is born here as
``z_init[:, r0:r1]`` only, never ``[N, N, 128]``), the recycling update ``linear_z(layer_norm_z(.))`` (``model.py:232``), the single
recycle ``s_init + linear_s(layer_norm_s(s))`` (``:298``), and per cycle the template embedder (``template``), the MSA embedder + MSA module
(``msa``; ``m`` replicated — the core proves the RNG stream identical across ranks before the draw) and the Pairformer (``pairstack``).

OpenFold3-specific and therefore here: the replicated single-representation statements of ``InputEmbedderAllAtom.forward`` (atom attention
encoder, ``linear_s``), ``relpos_complex``'s composition (``core/utils/relpos.py``: residue offset | token offset | same-entity | chain
offset, conditions same-chain / same-chain-and-residue / same-entity, ``binned_one_hot``) over the core's row primitives, the attribute
paths, the mode-memory settings object, and the refusals (training, ``offload_inference.msa_module``, fused attention / triangle kernels on
shards — all pinned off by the launcher's yaml; refused by name if a yaml re-enables them).
"""
from __future__ import annotations

import time

import torch

from . import core as C


class TrunkRefused(RuntimeError):
    pass


KERNEL_FLAGS = ("use_deepspeed_evo_attention", "use_cueq_triangle_kernels", "use_triton_triangle_kernels", "use_lma")


# ----------------------------------------------------------------------------------------------------------------- input embedder
def input_embedder_single(emb, batch: dict, use_high_precision_attention: bool = True):
    """The replicated part of ``InputEmbedderAllAtom.forward`` (``input_embedders.py:105-150``): ``(s_input, s, s_input_emb_i, s_input_emb_j)``
    — OpenFold3's statements, unchanged shapes (every rank runs them)."""
    with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
        a, _, _, _ = emb.atom_attn_enc(batch=batch, use_high_precision_attention=use_high_precision_attention)
    a = a.to(dtype=emb.linear_s.weight.dtype)
    s_input = torch.cat([a, batch["restype"], batch["profile"], batch["deletion_mean"].unsqueeze(-1)], dim=-1)
    s = emb.linear_s(s_input)
    return s_input, s, emb.linear_z_i(s_input), emb.linear_z_j(s_input)


def _binned_one_hot(d, n):
    """OpenFold3's ``binned_one_hot(offset, arange(n))`` (``core/utils/tensor_utils.py``) on the core's int64 bins — values 0/1 as ``one_hot``."""
    from openfold3.core.utils.tensor_utils import binned_one_hot
    return binned_one_hot(d, torch.arange(start=0, end=n, device=d.device))


def relpos_rows(batch: dict, g0: int, g1: int, max_relative_idx: int, max_relative_chain: int, dtype=torch.float32):
    """``relpos_complex(batch, max_relative_idx, max_relative_chain)[..., g0:g1, :, :]`` (``core/utils/relpos.py``): rows ``[g0, g1)`` of the
    ``[*, N, N, 2 (2 r + 2) + 1 + (2 s + 2)]`` one-hot features, composed from the core's row primitives (only the row slab is ever built)."""
    T = C.seam("trunk")
    same_chain = T.same_rows(batch["asym_id"], g0, g1)
    same_res = T.same_rows(batch["residue_index"], g0, g1)
    same_entity = T.same_rows(batch["entity_id"], g0, g1)
    rel_pos = T.relpos_onehot_rows(batch["residue_index"], g0, g1, max_relative_idx, condition=same_chain, dtype=dtype, one_hot=_binned_one_hot)
    rel_token = T.relpos_onehot_rows(batch["token_index"], g0, g1, max_relative_idx, condition=same_chain & same_res, dtype=dtype, one_hot=_binned_one_hot)
    rel_chain = T.relpos_onehot_rows(batch["sym_id"], g0, g1, max_relative_chain, condition=same_entity, dtype=dtype, one_hot=_binned_one_hot)
    return torch.cat([rel_pos, rel_token, same_entity[..., None].to(dtype=dtype), rel_chain], dim=-1)


def relpos_bins(max_relative_idx: int, max_relative_chain: int) -> int:
    return 2 * (2 * int(max_relative_idx) + 2) + 1 + (2 * int(max_relative_chain) + 2)


def init_rows_fn_of(emb, batch: dict, emb_i, emb_j, s_dtype, inplace_safe: bool = False):
    """``init_rows_fn(g0, g1)`` -> ``z_init[..., g0:g1, :, :]``: OpenFold3's per-element statements (outer sum, + relpos rows, + token-bond rows)."""
    from openfold3.core.utils.tensor_utils import add
    T = C.seam("trunk")

    def init_rows_fn(g0: int, g1: int):
        tb_rows = T.feature_rows(batch["token_bonds"], g0, g1, device=emb_i.device, row_dim=-2)     # msa.BONDS_ON_HOST: the [N, N] feature is host-resident; the row slab moves
        token_bonds_emb = emb.linear_token_bonds(tb_rows.unsqueeze(-1).to(dtype=s_dtype))
        del tb_rows
        z = T.outer_sum_rows(emb_i, emb_j, g0, g1)                                                  # input_embedders.py:156
        relpos_emb = emb.linear_relpos(relpos_rows(batch, g0, g1, emb.max_relative_idx, emb.max_relative_chain, dtype=z.dtype))   # :158-163
        z = add(z, relpos_emb, inplace=inplace_safe)                                                # :164
        z = add(z, token_bonds_emb, inplace=inplace_safe)                                           # :166
        return z

    return init_rows_fn


# ----------------------------------------------------------------------------------------------------------------- trunk
def refuse_settings(model, mode_mem_settings, N: int) -> None:
    if model.training:
        raise TrunkRefused("refused: the tp line is inference-only (model.training is True)")
    from openfold3.projects.of3_all_atom.model import OffloadModules
    if model._do_inference_offload(seq_len=N, module_name=OffloadModules.MSA_MODULE.value):
        raise TrunkRefused("refused: settings.memory.eval.offload_inference.msa_module=true under the tp line (the launcher's yaml pins it false)")
    on = [f for f in KERNEL_FLAGS if bool(getattr(mode_mem_settings, f, False))]
    if on:
        raise TrunkRefused(f"refused: settings.memory.eval.{'/'.join(on)}=true under the tp line (row shards run the torch attention / tri-mul statements; the launcher's yaml pins them false)")


def trunk_rng_guard() -> str:
    """The trunk's per-cycle RNG-stream proof (``run_trunk_sharded(guard=)``) follows the replicated-draw policy (``msa.sync_policy``): ``guard`` ->
    ``"rng"`` (every rank draws the MSA subsample itself, so the generators must be in lockstep and are proven so before each draw); ``bcast`` ->
    ``"off"`` (rank 0 draws and broadcasts, its generator alone advances by design, and the VALUE is proven replicated instead — the core prints the
    guard as off)."""
    from . import msa as MSA
    return "rng" if MSA.sync_policy() == "guard" else "off"


def run_trunk_rows(model, batch: dict, num_cycles: int, inplace_safe: bool = True, lay=None, log=None):
    """``OpenFold3.run_trunk`` (inference) with z row-sharded from birth -> ``(s_input, s, z_loc, lay)``; ``z_loc`` is this rank's shard
    ``[1, n_loc, N, 128]`` (no gather: every consumer downstream is row-local)."""
    from . import msa as MSA, template as TEMPL, pairstack as PS
    T = C.seam("trunk")
    comm = C.comm()
    log = log or comm.log
    mode_mem_settings = model._get_mode_mem_settings()
    N = int(batch["token_mask"].shape[-1])
    refuse_settings(model, mode_mem_settings, N)
    lay = C.layout_of(N) if lay is None else lay
    if int(lay.N) != N:
        raise TrunkRefused(f"refused: layout N={lay.N} != batch N={N}")
    chunk_size = mode_mem_settings.chunk_size
    kernel_kw = {f: False for f in KERNEL_FLAGS}

    C.mark("trunk_entry")
    MSA.place_batch_features(batch)                        # the host-resident MSA / bond features (msa.MSA_HOST / BONDS_ON_HOST; idempotent)
    MSA.log_batch_residency(batch, "run_trunk entry")
    emb = model.input_embedder
    s_input, s_init, emb_i, emb_j = input_embedder_single(emb, batch, use_high_precision_attention=True)
    token_mask = batch["token_mask"]
    pair_mask_loc = T.pair_mask_rows(token_mask, lay.r0, lay.r1)

    def template_fn(z_loc, cycle):                          # model.py:234  z = add(z, template_embedder(batch, z, pair_mask, ...), inplace)
        z_loc = TEMPL.template_embedder_add_rows_(model.template_embedder, batch, z_loc, pair_mask_loc, lay, census_tag=f"cycle{cycle}.template",
                                                 chunk_size=chunk_size, _mask_trans=True, inplace_safe=inplace_safe, **kernel_kw)
        C.mark("after_template")
        return z_loc

    def msa_fn(z_loc, cycle):                               # model.py:251-282  m, msa_mask = msa_module_embedder(batch, s_input); z = msa_module(m, z, ...)
        m, msa_mask = model.msa_module_embedder(batch=batch, s_input=s_input)
        MSA.log_batch_residency(batch, f"after msa_module_embedder (cycle {cycle}; m={tuple(m.shape)})")
        if comm.verbose:
            comm.checksum(m, f"cycle{cycle}.m")
        cutoff = mode_mem_settings.msa_module.swiglu_chunk_token_cutoff
        transition_ckpt_chunk_size = mode_mem_settings.msa_module.swiglu_seq_chunk_size if cutoff is None or cutoff > m.shape[-2] else None
        z_loc = MSA.msa_module_rows(model.msa_module, m, z_loc, msa_mask=msa_mask.to(dtype=m.dtype), pair_mask_loc=pair_mask_loc.to(dtype=z_loc.dtype), lay=lay,
                                    chunk_size=chunk_size, transition_ckpt_chunk_size=transition_ckpt_chunk_size, inplace_safe=inplace_safe, _mask_trans=True, **kernel_kw)
        del m, msa_mask
        C.mark("after_msa")
        return z_loc

    def single_recycle_fn(s, cycle):                        # model.py:298  s = s_init + linear_s(layer_norm_s(s))
        return s_init + model.linear_s(model.layer_norm_s(s))

    def pairstack_fn(s, z_loc, cycle):                      # model.py:300-312  s, z = pairformer_stack(s, z, single_mask, pair_mask, ...)
        s, z_loc = PS.pairformer_stack_rows(model.pairformer_stack, s, z_loc, single_mask=token_mask.to(dtype=z_loc.dtype), pair_mask_loc=pair_mask_loc.to(dtype=s.dtype),
                                          lay=lay, chunk_size=chunk_size, inplace_safe=inplace_safe, _mask_trans=True, **kernel_kw)
        C.mark("after_pairstack")
        if comm.verbose:
            comm.checksum(s, f"cycle{cycle}.s")
        return s, z_loc

    def on_cycle_start(cycle, is_final):                    # model.py:220-223
        if is_final:
            model.clear_autocast_cache()

    t0 = time.time()
    out = T.run_trunk_sharded(lay, n_cycles=int(num_cycles),
                              init_rows_fn=init_rows_fn_of(emb, batch, emb_i, emb_j, s_init.dtype, inplace_safe=inplace_safe), init_like=emb_i,
                              init_channels=3 * int(emb_i.shape[-1]) + relpos_bins(emb.max_relative_idx, emb.max_relative_chain),
                              recycle_update_fn=lambda z_rows: model.linear_z(model.layer_norm_z(z_rows)),          # model.py:232
                              s_init=s_init, single_recycle_fn=single_recycle_fn, template_fn=template_fn, msa_fn=msa_fn, pairstack_fn=pairstack_fn,
                              s_input=s_input, gather="none", on_cycle_start=on_cycle_start,
                              guard=trunk_rng_guard(),                                                 # the per-cycle RNG-stream proof: on under `guard`, off (printed) under `bcast`
                              census_fn=lambda tag: MSA.log_live_tensors(tag), log=log)
    del emb_i, emb_j
    C.mark("no_gather")                                     # z leaves the trunk as the shard: no all-gather of the pair representation anywhere
    log(f"trunk done N={N} P={lay.P} rows {lay.r0}:{lay.r1} in {time.time() - t0:.1f}s")   # the launcher's census reads `trunk done N=… rows a:b`
    return s_input, out.s, out.z, lay
