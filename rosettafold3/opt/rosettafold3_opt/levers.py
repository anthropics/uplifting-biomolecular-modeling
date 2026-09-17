"""The big levers of rosettafold3, registered with ``opt_core.mem.registry`` at import (``big.apply`` imports this module at the
trigger). Each lever: ``applies(ctx)`` names the precondition that fails (a class the upstream did not define, a hoist that is off, a
setting outside its range), ``apply(ctx)`` patches the sites and returns the ``Applied`` with the values in force and the sites by
name. Levered paths mark the open unit (``ctx.record.mark``); a site that runs the stock path instead records a named ``fallback``.

The patched sites and what stays the stock's (every value the levers compute is the stock's arithmetic on a subset of its elements;
the stock functions are called for everything else):

  atom_pair_local
    AttentionPairBiasDiffusion.atom_attention      derived from the kit's own method source: the one statement that gathers the local
                                                   windows out of the dense pair (``B_local = GF.hoist_get(..., lambda: self.to_b(
                                                   self.ln_0(Z_II[:, indicesQ[:, :, None], indicesK[:, None, :]])))``) becomes the same
                                                   ``to_b(ln_0(...))`` on the window form when the pair arrives windowed (dim 4) and the
                                                   stock gather (a named fallback) when it arrives dense. Every other byte of the
                                                   method is the kit's (``_derive``: the statement must match verbatim, else refused).
    AtomAttentionEncoderDiffusion._forward_hoisted derived the same way: the one statement ``C_L0, C_L, P_LL = GF.hoist_get((id(self),
                                                   "prefix"), _prefix)`` hands the prefix to ``prefix_windowed`` (the kit's ``_prefix``
                                                   arithmetic on the window index set; the kit's ``_prefix`` is the named fallback).
    AtomAttentionEncoderPairformer.forward         replaced (``pairformer_forward_windowed``): the stock method's arithmetic with the
                                                   pair built on the window index set.
  opm_chunk        OuterProductMean_AF3.forward     replaced: the stock method over row blocks of ``l``.
  cond_chunk       DiffusionConditioning.forward_hoisted   derived: the statement ``Z_II = GF.hoist_get((id(self), "Z_II"), _pair)`` hands the
                                                   pair conditioning to ``pair_chunked`` (the kit's ``_pair`` over row blocks).
  confidence_offload
    ConfidenceHead.forward                         wrapped: ``pae_logits`` / ``pde_logits`` copied to pinned host memory on return.
    rf3.utils.predicted_error.compile_af3_style_confidence_outputs   wrapped: one sample's logits back on the device, the stock function
                                                   on that sample (``batch_idx=0``), the full-batch ``plddt`` the stock returns rebuilt
                                                   from the stock's own ``unbin_logits`` expression.
    rf3.metrics.predicted_error.compute_ptm        wrapped: per sample on the device from the host copy (the stock function each time).

Window geometry is the atom transformer's own (``atom_attention(qbatch=32, kbatch=128)``): ``indicesQ[i, j] = 32 i + j``,
``indicesK[i, k] = 32 i - 48 + k``, clamped to ``[0, L-1]`` (the clamped entries are masked by ``-1e9`` in the attention and their
values never reach a softmax weight; they are computed on the clamped atom as the stock gather does).
"""
from __future__ import annotations

from dataclasses import replace as _dc_replace
import inspect
import sys
import textwrap
from typing import Optional

from . import _core
from . import modes as _modes

registry = _core.load("mem.registry")
register, refuse, Applied, RefusalError = registry.register, registry.refuse, registry.Applied, registry.RefusalError
_offload = _core.load("mem.offload")                        # the ONE offload budget (Settings / PinPool): confidence_offload's pinned staging (stdlib import)

def _park_settings(ctx, **overrides):
    """``opt_core.mem.offload.Settings`` through the core (``Settings.from_ctx`` reads the kit's ``ctx.settings["host_park"]``; nothing from the
    environment), with this tree's stated values (``PIN_MAX_GB_DEFAULT`` / ``MIN_TOKENS_DEFAULT``: the pinned budget and the
    engagement threshold of ``confidence_offload`` / ``feature_park``) supplied as the kit settings, and a site's own override (``cols``) last."""
    kit = {"pin_max_gb": float(PIN_MAX_GB_DEFAULT), "min_tokens": int(MIN_TOKENS_DEFAULT)}
    kit.update(ctx.settings.get(_offload.LEVER) or {})
    ctx.settings[_offload.LEVER] = kit
    s = _offload.Settings.from_ctx(ctx)
    return _dc_replace(s, **overrides) if overrides else s



def _chunker():
    """``opt_core.mem.chunk`` — the ONE chunker (``chunk_rows``): opm_chunk / cond_chunk run their row blocks through it (imports torch)."""
    return _core.load("mem.chunk")


CHUNK_ENTRIES_KEPT = 64                                     # the chunker's per-call entries kept verbatim (the count and summary keep the rest)

QBATCH, KBATCH = 32, 128                                    # the atom transformer's window geometry (atom_attention defaults)
OPM_ROWS_RANGE = (16, 65536)
COND_ROWS_RANGE = (16, 65536)
# the chunker's per-call label + reason (stated once; every call cites it): the lever's own label stays `measured` — narrowed by a deterministic comparison
OPM_BAND_REASON = ("per (l, m) pair (the s-reduction is inside one element), but the einsum's GEMM at M = rows x c_outer may select another reduction kernel "
                   "than at M = L x c_outer (ULP-level on CPU at rows=1): measured, pending a deterministic comparison")
TRANS_BAND_REASON = ("per element of the leading dims the transition is the stock statement; the linears' GEMM at M = rows x J may select another kernel than at "
                     "M = I x J: measured, pending a deterministic comparison")
COND_BAND_REASON = ("concat, LayerNorm over channels, linears and transitions are per pair; the linears' GEMM at M = rows x I may select another kernel than at "
                    "M = I x I: measured, pending a deterministic comparison")

L_ATOM_PAIR, L_OPM, L_COND, L_CONF = "atom_pair_local", "opm_chunk", "cond_chunk", "confidence_offload"
UNIT_STEM = "fold"

STATE: dict = {"ctx": None, "orig": {}, "chunk_entries": [], "chunk_calls": 0, "offload": None}


def chunk_record(lever: str, site: str, exact: str, reason: str, **details) -> dict:
    """The chunker's per-call record sink (``opt_core.mem.chunk``'s ``record`` contract: ``(lever, site, exact, reason, **details)``):
    every call counted, the last CHUNK_ENTRIES_KEPT entries kept (``big.state`` summarises them with the core's ``chunk.summary``)."""
    entry = {"lever": lever, "site": site, "exact": exact, "reason": reason}
    entry.update(details)
    STATE["chunk_calls"] += 1
    STATE["chunk_entries"].append(entry)
    del STATE["chunk_entries"][:-CHUNK_ENTRIES_KEPT]
    return entry


# ------------------------------------------------------------------------------------------------------------ helpers
def _torch():
    t = sys.modules.get("torch")
    if t is None:
        raise RefusalError(refuse("big", "torch", "torch is not imported in this process"))
    return t


def _rec():
    ctx = STATE["ctx"]
    return ctx.record if ctx is not None else None


def mark(lever: str, detail: Optional[str] = None) -> None:
    r = _rec()
    if r is not None:
        r.mark(lever, detail=detail)


def fallback(lever: str, reason: str) -> None:
    r = _rec()
    if r is not None:
        r.fallback(lever, reason)


def windows(L: int, device, qbatch: int = QBATCH, kbatch: int = KBATCH):
    """``(indicesQ [nq, qbatch], indicesK [nq, kbatch])`` — the atom transformer's own index arithmetic (clamped)."""
    torch = _torch()
    nqbatch = (L + qbatch - 1) // qbatch
    Cs = torch.arange(nqbatch, device=device) * qbatch + qbatch // 2
    patchq = torch.arange(qbatch, device=device) - qbatch // 2
    patchk = torch.arange(kbatch, device=device) - kbatch // 2
    indicesQ = torch.clamp(Cs[:, None] + patchq[None, :], 0, L - 1)
    indicesK = torch.clamp(Cs[:, None] + patchk[None, :], 0, L - 1)
    return indicesQ, indicesK


DERIVED_ATTR = "__big_derived__"                           # set on every function _derive produces: {"cls", "method", "old", "new"}


def _derive(cls, method: str, old: str, new: str, module) -> object:
    """A function derived from the kit's / the upstream's own method source with ONE statement replaced verbatim (``old`` -> ``new``);
    executed in the defining module's namespace. The statement must occur exactly once, else a refusal by name (the tree's bytes are
    not the ones this lever was written against)."""
    fn = getattr(cls, method)
    if getattr(fn, DERIVED_ATTR, None) is not None:            # derived here already (its code has no source file): a second derivation is refused by name
        d = getattr(fn, DERIVED_ATTR)
        raise RefusalError(refuse("big", f"source.{cls.__name__}.{method}", f"{cls.__name__}.{method} is already the function this kit derived in this process "
                                  f"(statement {d['old'].strip()!r} -> {d['new'].strip()!r}); the levers apply once per process"))
    try:
        src = inspect.getsource(fn)                            # at the class's indentation: the statements below match at theirs
    except OSError as e:                                       # no source behind the function (not this tree's file-backed class body): refused by name
        raise RefusalError(refuse("big", f"source.{cls.__name__}.{method}", f"the source of {cls.__name__}.{method} cannot be read ({e}); "
                                  f"this lever derives its function from the class's source text")) from e
    n = src.count(old)
    if n != 1:
        raise RefusalError(refuse("big", f"source.{cls.__name__}.{method}", (f"the statement this lever replaces is ABSENT from {cls.__name__}.{method} "
                                  f"(the upstream bytes differ from the shipped kit): {old.strip()!r}") if n == 0 else
                                  f"the statement this lever replaces occurs {n} times in {cls.__name__}.{method} (expected once): {old.strip()!r}"))
    src = textwrap.dedent(src.replace(old, new))
    ns = dict(vars(module))
    exec(compile(src, f"<big:{cls.__name__}.{method}>", "exec"), ns)
    derived = ns[method]
    setattr(derived, DERIVED_ATTR, {"cls": cls.__name__, "method": method, "old": old, "new": new})
    return derived


# ============================================================================================================ atom_pair_local
def _atom_pair_applies(ctx):
    h = ctx.require(L_ATOM_PAIR, "encoder_diffusion", "encoder_pairformer", "attention", "graph_flags")
    gf = h["graph_flags"]
    if not getattr(gf, "HOIST", False):
        return refuse(L_ATOM_PAIR, "hoist", "RF3_HOIST is off in this process: the hoisted prefix (the patched site) never runs")
    if not hasattr(h["encoder_diffusion"], "_forward_hoisted"):
        return refuse(L_ATOM_PAIR, "kit_bytes", "AtomAttentionEncoderDiffusion has no _forward_hoisted: not the kit's patched tree")
    if not hasattr(h["attention"], "atom_attention"):
        return refuse(L_ATOM_PAIR, "upstream_bytes", "AttentionPairBiasDiffusion has no atom_attention")
    return None


_OLD_BLOCAL = ("        B_local = GF.hoist_get(\n            (id(self), \"B_local\"),\n"
               "            lambda: self.to_b(self.ln_0(Z_II[:, indicesQ[:, :, None], indicesK[:, None, :]])),\n        )\n")
_NEW_BLOCAL = ("        B_local = _BIG_B_LOCAL(self, Z_II, indicesQ, indicesK, nqbatch, qbatch, kbatch)\n")
_OLD_PREFIX = "        C_L0, C_L, P_LL = GF.hoist_get((id(self), \"prefix\"), _prefix)\n"
_NEW_PREFIX = "        C_L0, C_L, P_LL = GF.hoist_get((id(self), \"prefix\"), lambda: _BIG_PREFIX(self, f, S_trunk_I, Z_II, tok_idx, L, I, _prefix))\n"


def b_local(self, Z_II, indicesQ, indicesK, nqbatch, qbatch, kbatch):
    """The window pair bias, after the method's ``Z_II = Z_II[None]``: the window form (``[1, nq, qbatch, kbatch, c]``) feeds ``to_b(ln_0(...))``
    directly; a dense pair (``[1, L, L, c]``) takes the stock gather through a named fallback."""
    GF = sys.modules["rf3.graph_flags"]
    if Z_II.dim() == 5 and tuple(Z_II.shape[1:4]) == (nqbatch, qbatch, kbatch):          # Z_II[None] happened above: [1, nq, q, k, c]
        mark(L_ATOM_PAIR, "atom_attention:window")
        return GF.hoist_get((id(self), "B_local"), lambda: self.to_b(self.ln_0(Z_II)))
    if Z_II.dim() == 4:
        fallback(L_ATOM_PAIR, f"atom_attention: dense pair {tuple(Z_II.shape)} reached the window site (stock gather ran)")
        return GF.hoist_get((id(self), "B_local"), lambda: self.to_b(self.ln_0(Z_II[:, indicesQ[:, :, None], indicesK[:, None, :]])))
    raise RuntimeError(f"big {L_ATOM_PAIR}: atom_attention got a pair of shape {tuple(Z_II.shape)} (neither dense [1, L, L, c] nor "
                       f"window [1, {nqbatch}, {qbatch}, {kbatch}, c])")


def pair_windows_dense(process_z, Z_II, tq, tk):
    """The token-pair window term of the atom encoder's pair: ``process_z(Z)[tq, tk]`` -> ``[nq, q, k, c_atompair]`` for the window token
    indices ``tq [nq, q]`` / ``tk [nq, k]`` of a dense pair ``Z [I, I, c_z]``."""
    return process_z(Z_II)[tq[:, :, None], tk[:, None, :], :]


PAIR_WINDOWS = pair_windows_dense      # the window term's producer: the dense statement above; ``rowpair`` rebinds it under --n_gpu>1 (the
                                       # banded form over the row-sharded conditioned pair, opt_core.mem.rowpair.diffusion.pair_band_rows)


def prefix_windowed(self, f, S_trunk_I, Z_II, tok_idx, L, I, stock_prefix):
    """The kit's ``_prefix`` (``AtomAttentionEncoderDiffusion._forward_hoisted``) with ``P_LL`` on the window index set: ``C_L0``,
    ``C_L`` as the stock, ``P`` of shape ``[nq, qbatch, kbatch, c_atompair]``. Layouts the kit's prefix does not produce (a batched
    trunk, the old broadcast) take the stock prefix through a named fallback."""
    torch = _torch()
    DT = sys.modules["rf3.model.layers.af3_diffusion_transformer"]
    GF = sys.modules["rf3.graph_flags"]
    ref, uid = f["ref_pos"], f["ref_space_uid"]
    if ref.dim() != 2 or uid.dim() != 1 or Z_II.dim() != 3 or S_trunk_I.dim() != 2 or getattr(self, "broadcast_trunk_feats_on_1dim_old", False):
        fallback(L_ATOM_PAIR, f"diffusion prefix: layout ref_pos{tuple(ref.shape)} Z{tuple(Z_II.shape)} S{tuple(S_trunk_I.shape)} "
                              f"old_broadcast={getattr(self, 'broadcast_trunk_feats_on_1dim_old', None)} is not the windowed form's (dense prefix ran)")
        return stock_prefix()
    iQ, iK = windows(L, ref.device)
    C_L0 = self.process_input_features(torch.cat(tuple(DT.collapse(f[name], L) for name in self.atom_1d_features), dim=-1))
    if self.use_atom_level_embedding:
        assert "atom_level_embedding" in f
        C_L0 = C_L0 + self.process_atom_level_embedding(f["atom_level_embedding"])
    D_w = ref[iQ][:, :, None, :] - ref[iK][:, None, :, :]                                  # [nq, q, k, 3]  (stock: D_LL[l, m] = ref[l] - ref[m])
    V_w = (uid[iQ][:, :, None] == uid[iK][:, None, :]).unsqueeze(-1)                         # [nq, q, k, 1]
    P = self.process_d(D_w) * V_w
    if GF.GRAPH_SAFE_OPS:
        Vf = V_w[..., 0:1]
        if self.use_inv_dist_squared:
            inv = 1 / (1 + torch.sum(D_w * D_w, dim=-1, keepdim=True))
        else:
            inv = 1 / (1 + torch.linalg.norm(D_w, dim=-1, keepdim=True))
        upd = self.process_inverse_dist(inv)
        P = torch.where(Vf, P + upd, P)
        vm = self.process_valid_mask(Vf.to(P.dtype))
        P = torch.where(Vf, P + vm, P)
    else:
        m = V_w[..., 0]
        if self.use_inv_dist_squared:
            P[m] += self.process_inverse_dist(1 / (1 + torch.sum(D_w[m] * D_w[m], dim=-1, keepdim=True)))
        else:
            P[m] += self.process_inverse_dist(1 / (1 + torch.linalg.norm(D_w[m], dim=-1, keepdim=True)))
        P[m] += self.process_valid_mask(V_w[m].to(P.dtype))
    S_trunk_embed_L = self.process_s_trunk(S_trunk_I)[..., tok_idx, :]
    C_L = C_L0 + S_trunk_embed_L
    tq, tk = tok_idx[iQ], tok_idx[iK]
    P = P + PAIR_WINDOWS(self.process_z, Z_II, tq, tk)                                        # [nq, q, k, c]  (the stock: gathered [tok_l, tok_m])
    sl, sm = self.process_single_l(C_L), self.process_single_m(C_L)
    P = P + (sl[iQ][:, :, None, :] + sm[iK][:, None, :, :])
    P = P + self.pair_mlp(P)
    mark(L_ATOM_PAIR, "diffusion_prefix:window")
    return C_L0, C_L, P


def pairformer_forward_windowed(self, f, R_L, S_trunk_I, Z_II):
    """``AtomAttentionEncoderPairformer.forward`` (the trunk's input embedder) with the pair on the window index set; the stock's
    arithmetic otherwise (its ``embed_features`` body, the same statements in the same order)."""
    torch = _torch()
    PL = sys.modules["rf3.model.layers.pairformer_layers"]
    assert R_L is None
    assert S_trunk_I is None
    assert Z_II is None
    tok_idx = f["atom_to_token_map"]
    L = len(tok_idx)
    I = tok_idx.max() + 1
    f["ref_atom_name_chars"] = f["ref_atom_name_chars"].reshape(L, -1)
    C_L = self.process_input_features(torch.cat(tuple(PL.collapse(f[feature_name], L) for feature_name in self.atom_1d_features), dim=-1))
    if self.use_atom_level_embedding:
        assert "atom_level_embedding" in f
        C_L = C_L + self.process_atom_level_embedding(f["atom_level_embedding"])
    ref, uid = f["ref_pos"], f["ref_space_uid"]
    if ref.dim() != 2 or uid.dim() != 1:
        fallback(L_ATOM_PAIR, f"pairformer encoder: layout ref_pos{tuple(ref.shape)} uid{tuple(uid.shape)} is not the windowed form's (stock dense forward ran)")
        return STATE["orig"]["pairformer_forward"](self, f, R_L, S_trunk_I, Z_II)
    iQ, iK = windows(L, ref.device)
    D_w = ref[iQ][:, :, None, :] - ref[iK][:, None, :, :]
    V_w = (uid[iQ][:, :, None] == uid[iK][:, None, :]).unsqueeze(-1)

    @PL.activation_checkpointing
    def embed_features(C_L, D_w, V_w):
        P = self.process_d(D_w) * V_w
        if self.use_inv_dist_squared:
            P += self.process_inverse_dist(1 / (1 + torch.sum(D_w * D_w, dim=-1, keepdim=True))) * V_w
        else:
            P = P + self.process_inverse_dist(1 / (1 + torch.linalg.norm(D_w, dim=-1, keepdim=True))) * V_w
        P = P + self.process_valid_mask(V_w.to(P.dtype)) * V_w
        Q_L = C_L
        sl, sm = self.process_single_l(C_L), self.process_single_m(C_L)
        P = P + (sl[iQ][:, :, None, :] + sm[iK][:, None, :, :])
        P = P + self.pair_mlp(P)
        Q_L = self.atom_transformer(Q_L, C_L, P)
        A_I_shape = Q_L.shape[:-2] + (I, self.c_token)
        processed_Q_L = self.process_q(Q_L)
        processed_Q_L = processed_Q_L.to(Q_L.dtype)
        A_I = PL.scatter_mean(torch.zeros(A_I_shape, device=Q_L.device, dtype=Q_L.dtype), -2, f["atom_to_token_map"].long(), processed_Q_L)
        return A_I, Q_L, C_L, P

    mark(L_ATOM_PAIR, "pairformer_encoder:window")
    return embed_features(C_L, D_w, V_w)


@register(L_ATOM_PAIR, family="chunk", exact="measured",
          exact_reason="measured, pending a deterministic comparison: the pair conditioning is pointwise per (l, m) pair: the same values on the window index set; the GEMM kernel a "
                       "different M selects for the channel linears is the open question a deterministic comparison decides",
          applies=_atom_pair_applies, description="the atom-pair conditioning P_LL built in the atom transformer's window form (never dense [L, L, c])",
          preconditions=("hooks.encoder_diffusion", "hooks.encoder_pairformer", "hooks.attention", "hooks.graph_flags", "hoist", "kit_bytes", "upstream_bytes",
                         "source.AttentionPairBiasDiffusion.atom_attention", "source.AtomAttentionEncoderDiffusion._forward_hoisted"),
          settings=())
def atom_pair_local(ctx) -> Applied:
    h = ctx.require(L_ATOM_PAIR, "encoder_diffusion", "encoder_pairformer", "attention")
    DT = sys.modules["rf3.model.layers.af3_diffusion_transformer"]
    att, enc_d, enc_p = h["attention"], h["encoder_diffusion"], h["encoder_pairformer"]
    DT._BIG_B_LOCAL = b_local
    DT._BIG_PREFIX = prefix_windowed
    new_att = _derive(att, "atom_attention", _OLD_BLOCAL, _NEW_BLOCAL, DT)
    new_enc = _derive(enc_d, "_forward_hoisted", _OLD_PREFIX, _NEW_PREFIX, DT)
    orig = {"atom_attention": att.atom_attention, "_forward_hoisted": enc_d._forward_hoisted, "pairformer_forward": enc_p.forward}
    STATE["orig"].update(orig)
    att.atom_attention = new_att
    enc_d._forward_hoisted = new_enc
    enc_p.forward = pairformer_forward_windowed

    def undo():
        att.atom_attention = orig["atom_attention"]
        enc_d._forward_hoisted = orig["_forward_hoisted"]
        enc_p.forward = orig["pairformer_forward"]

    return Applied(lever=L_ATOM_PAIR, settings={"qbatch": QBATCH, "kbatch": KBATCH, "form": "[nq, qbatch, kbatch, c_atompair]"},
                   sites=("AttentionPairBiasDiffusion.atom_attention", "AtomAttentionEncoderDiffusion._forward_hoisted", "AtomAttentionEncoderPairformer.forward"),
                   undo=undo)


# ============================================================================================================ opm_chunk
def _rows(ctx, lever, default, rng):
    rows = ctx.setting(lever, "rows", default, cast=int)
    if not (rng[0] <= rows <= rng[1]):
        raise RefusalError(refuse(lever, "rows", f"rows={rows} outside {rng}"))
    return rows


def _opm_applies(ctx):
    h = ctx.require(L_OPM, "module")
    if not hasattr(h["module"], "forward"):
        return refuse(L_OPM, "upstream_bytes", "OuterProductMean_AF3 has no forward")
    _rows(ctx, L_OPM, 256, OPM_ROWS_RANGE)
    return None


def opm_forward_chunked(self, msa):
    """``OuterProductMean_AF3.forward`` with the einsum + ``proj_out`` over row blocks of ``l`` through ``opt_core.mem.chunk.chunk_rows``
    (the stock's statements per block)."""
    torch = _torch()
    rows = STATE["opm_rows"]
    B, N, L = msa.shape[:3]
    msa = self.norm(msa)
    left = self.proj_left(msa)
    right = self.proj_right(msa)
    right = right / float(N)

    def block(lb):
        return self.proj_out(torch.einsum("bsli,bsmj->blmij", lb, right).reshape(B, lb.shape[2], L, -1))

    # left [B, s, L, i] -> the block on axis 2; the block's output [B, r, L, c] carries the rows on axis 1: chunk_rows wants the same axis
    # index on input and output, so the input is presented rows-first ([B, L, s, i] view; the einsum reads it back as [B, s, r, i]).
    out = _chunker().chunk_rows(lambda lb: block(lb.transpose(1, 2)), left.transpose(1, 2), 1, rows, exact="band", reason=OPM_BAND_REASON,
                            record=chunk_record, lever=L_OPM, site="outer_product_mean")
    mark(L_OPM, f"rows={rows}")
    return out


@register(L_OPM, family="chunk", exact="measured",
          exact_reason="measured, pending a deterministic comparison: the outer product mean is independent per (l, m) pair; row blocks change the GEMM's M, not its per-element reduction",
          applies=_opm_applies, description="OuterProductMean_AF3 over row blocks (the [N, N, c_outer^2] einsum output never whole)",
          preconditions=("hooks.module", "upstream_bytes", "rows"), settings=("rows",))
def opm_chunk(ctx) -> Applied:
    OP = sys.modules["rf3.model.layers.outer_product"]
    cls = ctx.require(L_OPM, "module")["module"]
    rows = _rows(ctx, L_OPM, 256, OPM_ROWS_RANGE)
    STATE["opm_rows"] = rows
    orig = cls.forward
    STATE["orig"]["opm_forward"] = orig
    deco = getattr(OP, "activation_checkpointing", None)
    cls.forward = deco(opm_forward_chunked) if deco is not None else opm_forward_chunked

    def undo():
        cls.forward = orig

    return Applied(lever=L_OPM, settings={"rows": rows}, sites=("OuterProductMean_AF3.forward",), undo=undo)


# ============================================================================================================ cond_chunk
_OLD_PAIR = "        Z_II = GF.hoist_get((id(self), \"Z_II\"), _pair)\n"
_NEW_PAIR = "        Z_II = GF.hoist_get((id(self), \"Z_II\"), lambda: _BIG_PAIR(self, f, Z_trunk_II, _pair))\n"


def _cond_applies(ctx):
    h = ctx.require(L_COND, "module", "graph_flags")
    if not getattr(h["graph_flags"], "HOIST", False):
        return refuse(L_COND, "hoist", "RF3_HOIST is off in this process: forward_hoisted (the patched site) never runs")
    if not hasattr(h["module"], "forward_hoisted"):
        return refuse(L_COND, "kit_bytes", "DiffusionConditioning has no forward_hoisted: not the kit's patched tree")
    _rows(ctx, L_COND, 512, COND_ROWS_RANGE)
    return None


def pair_chunked(self, f, Z_trunk_II, stock_pair):
    """The kit's ``_pair`` (``cat[Z_trunk.float(), relpos(f)] -> to_zii -> transition_1[0], [1]``) over row blocks of the first pair
    dimension through ``opt_core.mem.chunk.chunk_rows`` (the stock's statements per block). A batched trunk takes the stock pair (named)."""
    torch = _torch()
    rows = STATE["cond_rows"]
    if Z_trunk_II.dim() != 3:
        fallback(L_COND, f"pair conditioning: Z_trunk{tuple(Z_trunk_II.shape)} is not [I, I, c] (stock whole-pair path ran)")
        return stock_pair()
    relpos = self.relative_position_encoding(f)

    def block(zb, i0, i1):
        z = torch.cat([zb.float(), relpos[i0:i1]], dim=-1)
        z = self.to_zii(z)
        for b in range(2):
            z = z + self.transition_1[b](z)
        return z

    out = _chunker().chunk_rows(block, Z_trunk_II, 0, rows, exact="band", with_offsets=True, reason=COND_BAND_REASON,
                            record=chunk_record, lever=L_COND, site="diffusion_pair_conditioning")
    mark(L_COND, f"rows={rows}")
    return out


@register(L_COND, family="chunk", exact="measured",
          exact_reason="measured, pending a deterministic comparison: the pair conditioning (concat, LayerNorm over channels, linears, transitions) is pointwise per pair; row blocks change the GEMM's M only",
          applies=_cond_applies, description="DiffusionConditioning pair path over row blocks (the [N, N, 2 c_z] fp32 transients never whole)",
          preconditions=("hooks.module", "hooks.graph_flags", "hoist", "kit_bytes", "rows", "source.DiffusionConditioning.forward_hoisted"), settings=("rows",))
def cond_chunk(ctx) -> Applied:
    RS = sys.modules["rf3.model.RF3_structure"]
    cls = ctx.require(L_COND, "module")["module"]
    rows = _rows(ctx, L_COND, 512, COND_ROWS_RANGE)
    STATE["cond_rows"] = rows
    RS._BIG_PAIR = pair_chunked
    new = _derive(cls, "forward_hoisted", _OLD_PAIR, _NEW_PAIR, RS)
    orig = cls.forward_hoisted
    STATE["orig"]["forward_hoisted"] = orig
    cls.forward_hoisted = new

    def undo():
        cls.forward_hoisted = orig

    return Applied(lever=L_COND, settings={"rows": rows}, sites=("DiffusionConditioning.forward_hoisted",), undo=undo)


# ============================================================================================================ confidence_offload
PIN_MAX_GB_DEFAULT = 48.0                                   # ONE pinned budget per process, shared by confidence_offload's two staging buffers (2 x N^2 x 64 x 4 B each) and feature_park's parked features (msa_stack fp32 + bf16, the template conditioning)
MIN_TOKENS_DEFAULT = 1000                                   # below this the logits are small (< 0.5 GB per sample) and stay on the device (recorded: below_gate)


def _conf_applies(ctx):
    h = ctx.require(L_CONF, "model", "head", "predicted_error", "metrics")
    for name, mod, attr in (("predicted_error", h["predicted_error"], "compile_af3_style_confidence_outputs"), ("metrics", h["metrics"], "compute_ptm")):
        if not hasattr(mod, attr):
            return refuse(L_CONF, "upstream_bytes", f"rf3 {name} module has no {attr}")
    if not hasattr(h["head"], "forward"):
        return refuse(L_CONF, "upstream_bytes", "ConfidenceHead has no forward")
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return refuse(L_CONF, "cuda", "no CUDA device: host offload has nothing to offload from")
    try:
        _park_settings(ctx)
    except _offload.OffloadRefusal as e:
        return refuse(L_CONF, "settings", str(e))
    return None


class _Staging:
    """The lever's host side: one pinned staging buffer per (key, shape, dtype) from the library's budgeted pool (a refusal by name
    above the budget — never a pageable staging buffer), the device -> pinned -> pageable copy of one sample's logits, the per-sample
    device fetch with a one-sample cache per key, and the counters the record carries."""

    def __init__(self, settings):
        self.s = settings
        self.pool = _offload.PinPool(settings)
        self.staging = {}
        self.cache = {}
        self.n_offloaded = 0
        self.n_fetched = 0
        self.n_compile = 0
        self.n_ptm = 0
        self.n_compile_device = 0                                # consumer calls that found the logits on the device below the token gate (the declared policy)
        self.unit_offloaded = False                              # THIS unit's head decision (set per confidence-head call): the consumers run in the engine's post-processing and
                                                                 # metrics AFTER the unit closed, so they read the last unit's decision — never the process-cumulative n_offloaded
        self.n_device_after_offload = 0                          # consumer calls that found device logits although the last unit's samples WERE offloaded: a process-scope named event
        self.bytes_d2h = 0
        self.bytes_h2d = 0
        self.below_gate = 0

    def offload(self, key, t):
        torch = _torch()
        sig = (key, tuple(t.shape), t.dtype)
        buf = self.staging.get(sig)
        if buf is None:
            for k in [k for k in self.staging if k[0] == key]:
                self.pool.release(self.staging.pop(k))
            buf = self.pool.alloc(tuple(t.shape), t.dtype, tag=f"{L_CONF}:{key}")
            self.staging[sig] = buf
        buf.copy_(t, non_blocking=False)
        self.bytes_d2h += t.numel() * t.element_size()
        self.n_offloaded += 1
        self.cache.pop(key, None)
        return buf.clone()                                       # the pageable copy the stock's torch.cat stacks (the staging buffer is reused)

    def fetch(self, key, stack, i, dev):
        c = self.cache.get(key)
        if c is not None and c[0] is stack and c[1] == i:
            return c[2]
        g = stack[i:i + 1].to(dev, non_blocking=False)
        self.bytes_h2d += g.numel() * g.element_size()
        self.n_fetched += 1
        self.cache[key] = (stack, i, g)
        return g

    def counters(self):
        return {"n_offloaded": self.n_offloaded, "n_fetched": self.n_fetched, "n_compile": self.n_compile, "n_ptm": self.n_ptm, "n_compile_device": self.n_compile_device,
                "n_device_after_offload": self.n_device_after_offload, "below_gate": self.below_gate,
                "bytes_d2h": self.bytes_d2h, "bytes_h2d": self.bytes_h2d, "pinned_peak_gb": round(self.pool.bytes_peak / 2**30, 3),
                "pinned_now_gb": round(self.pool.bytes_now / 2**30, 3), "settings": self.s.as_dict()}


def head_forward_offload(self, *a, **k):
    out = STATE["orig"]["head_forward"](self, *a, **k)
    st = STATE["offload"]
    moved = []
    for key in ("pae_logits", "pde_logits"):
        t = out.get(key) if isinstance(out, dict) else None
        if t is None or t.device.type != "cuda":
            continue
        n_tokens = int(t.shape[-2])
        if n_tokens < st.s.min_tokens:
            st.below_gate += 1
            st.unit_offloaded = False
            mark(L_CONF, f"head:below_gate:N={n_tokens}")
            return out
        try:
            out[key] = st.offload(key, t)
        except _offload.OffloadRefusal as e:
            raise RuntimeError(f"big {L_CONF}: {e}") from e     # the budget refusal is the lever's own: never a pageable staging copy
        moved.append(key)
    st.unit_offloaded = bool(moved)
    if moved:
        mark(L_CONF, "head:" + ",".join(moved))
    else:
        fallback(L_CONF, "confidence head returned no device pae/pde logits to offload")
    return out


def device_logits_seen(st, consumer: str) -> None:
    """A consumer (``compile`` = the engine's post-processing, ``compute_ptm`` = its metrics) found the logits on the DEVICE. Both run after the
    fold unit closed (``model.forward`` delimits the unit), so the event is attributed by the LAST unit's own head decision, ``unit_offloaded``,
    never by the process-cumulative ``n_offloaded`` (an above-gate item followed by a below-gate item in one process is the ordinary case):
    below the gate — the declared policy — a counter; device logits although that unit's samples WERE offloaded — a named PROCESS-scope event
    (counter ``n_device_after_offload`` on the lever's exit evidence plus a record note), never a unit-scope event outside an open unit."""
    if st is None or not st.unit_offloaded:
        if st is not None:
            st.n_compile_device += 1
        return
    st.n_device_after_offload += 1
    r = _rec()
    if r is not None:
        r.note(f"{L_CONF}: {consumer}: pae logits arrived on the device although the last unit's samples were offloaded (stock whole-batch call ran) — n_device_after_offload={st.n_device_after_offload}")


def compile_offload(plddt_logits, pae_logits, pde_logits, *a, batch_idx=0, **k):
    """``compile_af3_style_confidence_outputs`` on one sample from the host copy (the stock function, ``batch_idx=0`` on the slice);
    ``plddt`` in the result is the stock's full-batch expression."""
    orig = STATE["orig"]["compile"]
    if pae_logits.device.type != "cpu":                          # the logits stayed on the device: below the token gate (the DECLARED policy, a counter) — the stock whole-batch call runs
        device_logits_seen(STATE["offload"], "compile")
        return orig(plddt_logits, pae_logits, pde_logits, *a, batch_idx=batch_idx, **k)
    PE = sys.modules["rf3.utils.predicted_error"]
    st = STATE["offload"]
    dev = plddt_logits.device
    res = orig(plddt_logits[batch_idx:batch_idx + 1], st.fetch("pae_logits", pae_logits, batch_idx, dev), st.fetch("pde_logits", pde_logits, batch_idx, dev),
               *a, batch_idx=0, **k)
    cfg = k["confidence_loss_cfg"] if "confidence_loss_cfg" in k else a[3]
    res["plddt"] = PE.unbin_logits(plddt_logits.reshape(-1, plddt_logits.shape[1], PE.NHEAVY, cfg.plddt.n_bins).permute(0, 3, 1, 2).float(),
                                   cfg.plddt.max_value, cfg.plddt.n_bins)
    st.n_compile += 1                                            # outside the fold unit (the engine's post-processing): a counter, never a unit mark
    return res


def ptm_offload(pae, to_calculate=None, *a, **k):
    """``compute_ptm`` per sample on the device from the host copy (the stock function each time; ``[D]`` assembled)."""
    torch = _torch()
    orig = STATE["orig"]["compute_ptm"]
    st = STATE["offload"]
    if pae.device.type != "cpu":                                 # below the token gate: the declared policy (a counter) — the stock whole-batch call runs
        device_logits_seen(st, "compute_ptm")
        return orig(pae, to_calculate, *a, **k)
    dev = torch.device("cuda", torch.cuda.current_device())
    tc = to_calculate.to(dev) if hasattr(to_calculate, "to") else to_calculate
    parts = [orig(st.fetch("pae_logits", pae, i, dev), tc, *a, **k) for i in range(pae.shape[0])]
    st.n_ptm += 1                                                # outside the fold unit: a counter (the head's mark inside the unit is the census event)
    return torch.cat(parts, dim=0)


@register(L_CONF, family="offload", exact="measured",
          exact_reason="measured, pending a deterministic comparison: host copies and the stock's own per-sample computations (expected bitwise; the det-recipe outputs decide)",
          applies=_conf_applies, description="pae/pde logits per sample to the host (pinned staging under the library's budget); the consumers run per sample on the device",
          preconditions=("hooks.model", "hooks.head", "hooks.predicted_error", "hooks.metrics", "upstream_bytes", "cuda", "settings"), settings=())
def confidence_offload(ctx) -> Applied:
    h = ctx.require(L_CONF, "head", "predicted_error", "metrics")
    head, PE, ME = h["head"], h["predicted_error"], h["metrics"]
    engine = ctx.hook(L_CONF, "engine")
    settings = _park_settings(ctx)
    STATE["offload"] = _Staging(settings)
    orig = {"head_forward": head.forward, "compile": PE.compile_af3_style_confidence_outputs, "compute_ptm": ME.compute_ptm}
    STATE["orig"].update(orig)
    head.forward = head_forward_offload
    PE.compile_af3_style_confidence_outputs = compile_offload
    ME.compute_ptm = ptm_offload
    sites = ["ConfidenceHead.forward", "rf3.utils.predicted_error.compile_af3_style_confidence_outputs", "rf3.metrics.predicted_error.compute_ptm"]
    rebound = False
    if engine is not None and getattr(engine, "compile_af3_style_confidence_outputs", None) is orig["compile"]:
        engine.compile_af3_style_confidence_outputs = compile_offload
        rebound = True
        sites.append("rf3.inference_engines.rf3.compile_af3_style_confidence_outputs")

    def undo():
        head.forward = orig["head_forward"]
        PE.compile_af3_style_confidence_outputs = orig["compile"]
        ME.compute_ptm = orig["compute_ptm"]
        if rebound:
            engine.compile_af3_style_confidence_outputs = orig["compile"]

    return Applied(lever=L_CONF, settings={"pin_max_gb": settings.pin_max_gb, "min_tokens": settings.min_tokens, "host_reserve_gb": settings.host_reserve_gb,
                                           "staging": "pinned (opt_core.mem.offload.PinPool)", "engine_rebound": rebound}, sites=tuple(sites), undo=undo)


def offload_counters() -> Optional[dict]:
    st = STATE["offload"]
    return st.counters() if st is not None else None


# ============================================================================================================ triatt_chunk
L_TRI = "triatt_chunk"
TRI_ROWS_RANGE = (64, 8192)                                 # query rows per block (default 512: at 4076 tokens one block holds ~3.5 GB of q/k/v/gate/out instead of ~35 GB whole)


def _arm_engages(ctx, tokens):
    """The component of the APPLIED FPF arm (``ctx.extra["fpf_arm"]``, the adapter's grammar ``<trimul>[+c...][@L]``) that engages a site,
    or ``None``: the base's arm minus big.DISENGAGED leaves the stock forward at the chunked levers' sites."""
    arm = (ctx.extra or {}).get("fpf_arm") or ""
    parts = [p.partition(".")[0] for p in arm.partition("@")[0].split("+")]   # a sub-word (fast.tmk3_fast, gflash.k2b) engages the component's site
    return next((p for p in parts if p in tokens), None)


TRI_GATE_RANGE = (0, 65536)                                # tokens: under an arm that engages the fused triangle attention (gflash) the kernel holds the site for a pair of at
                                                            # most `gate` tokens and the row-block statement takes it above (big's size gate; 0 = the fused kernel never holds the site)


def _tri_gate(ctx):
    gate = ctx.setting(L_TRI, "gate", 0, cast=int)
    if not (TRI_GATE_RANGE[0] <= int(gate) <= TRI_GATE_RANGE[1]):
        raise RefusalError(refuse(L_TRI, "gate", f"gate={gate} outside {TRI_GATE_RANGE}"))
    return int(gate)


def _tri_applies(ctx):
    """The stock triangle-attention site (``rf3.model.layers.attention.TriangleAttention``) on the EXACT base. With the FPF arm's fused
    triangle attention engaged (``gflash``) the lever is the SIZE GATE (setting ``gate``, tokens): a pair of at most ``gate`` tokens is the
    kernel's (the class forward found at install — the add-on's), a larger pair takes the row-block statement (the kernel steps aside by
    name: its q|k|v|g projections are whole [N, N, ·] tensors, and its launch fails from ~2,500–3,000 tokens); without a gate such an arm
    runs stock / the row-block statement at every size (gate 0: the fused kernel never holds the site). Without gflash in the arm the gate is moot (the row-block
    statement above ``rows``, stock's whole forward below)."""
    h = ctx.require(L_TRI, "module")
    owner = _arm_engages(ctx, _modes.FPF_TAKES_CUEQ["cueq_triattn"])   # the arm components that take the triangle-attention site (modes.py, the one table)
    _tri_gate(ctx)                                           # range-checked; 0 under a gflash arm = stock / row-block at every size (the fused kernel never holds the site)
    for name in ("forward", "_forward_cuequivariance"):
        if not callable(getattr(h["module"], name, None)):
            return refuse(L_TRI, "upstream_bytes", f"TriangleAttention has no {name}")
    _rows(ctx, L_TRI, 512, TRI_ROWS_RANGE)
    return None


def _tri_stock_forward(orig):
    """Stock's ``TriangleAttention.forward`` for the calls the gate sends past the kernel: the FPF add-on keeps the class's own forward in its
    ``_ORIG`` table when it binds the fused one; without the add-on the class forward found at install IS stock's."""
    adp = sys.modules.get("fpf_rf3_adapter")
    fn = (getattr(adp, "_ORIG", None) or {}).get("tri_forward") if adp is not None else None
    return fn or orig


def _tri_attention_rows(self, x_rows, bias_full, starting):
    """The stock ``_forward_cuequivariance`` statement on a block of LayerNorm'd query rows ``[B, R, J, C]`` with the FULL bias: the same casts,
    projections and rearranges as stock (attention.py); for the ending node the core assembled the bias on the transposed pair, so it is
    transposed back to stock's ``bias[i, k]`` orientation before the ``b i j h -> b 1 h i j`` rearrange."""
    torch = _torch()
    from einops import rearrange
    A = sys.modules["rf3.model.layers.attention"]
    bias = bias_full if starting else bias_full.transpose(-2, -3)
    if torch.is_autocast_enabled():
        dtype = torch.get_autocast_dtype(x_rows.device.type)
        x_rows = x_rows.to(dtype=dtype); bias = bias.to(dtype=dtype)
    gate = torch.sigmoid(self.to_g(x_rows))
    query = rearrange(self.to_q(x_rows), "b i j (h d) -> b i h j d", h=self.h)
    key = rearrange(self.to_k(x_rows), "b i k (h d) -> b i h k d", h=self.h)
    value = rearrange(self.to_v(x_rows), "b i k (h d) -> b i h k d", h=self.h)
    bias_cueq = rearrange(bias, "b i j h -> b 1 h i j")
    out = A.cuet.triangle_attention(query, key, value, bias=bias_cueq, scale=self.scaling)
    out = rearrange(out, "b i h j d -> b i j (h d)")
    return gate * out


def tri_forward_chunked(self, pair):
    """``TriangleAttention.forward`` in query-row blocks (``STATE['tri_rows']``): the core's two-pass ``triangle_attention_chunked`` with the
    stock module's own LayerNorm, bias projection and cuEquivariance statement as its parts; ``to_out`` on the assembled result as stock.
    Stock runs (named) for a pair at or below the block size, a non-4-D pair, or the vanilla route (no cuEquivariance / no CUDA)."""
    A = sys.modules["rf3.model.layers.attention"]
    orig = STATE["orig"]["tri_forward"]
    rows = STATE["tri_rows"]
    gate = int(STATE.get("tri_gate") or 0)                                # tokens the arm's fused kernel may hold (big: 0 = never); moot without gflash in the arm
    owner = bool(STATE.get("tri_owner"))                                 # the applied arm engages gflash: the class forward found at install is the add-on's fused one
    if owner and gate and pair.dim() == 4 and pair.shape[-3] <= gate:
        mark(L_TRI, f"gflash:N<={gate}")                                  # at or below the gate: the arm's fused triangle attention (fpf_gflash) holds the site
        return orig(self, pair)
    if owner:
        mark(L_TRI, f"gate:N={pair.shape[-3] if pair.dim() == 4 else 'na'}>{gate}:gflash-steps-aside")   # above the gate (every size at gate 0), by name: stock's parts below
        orig = _tri_stock_forward(orig)
    cueq = bool(self.use_cuequivariance and A.SHOULD_USE_CUEQUIVARIANCE)   # stock's own route predicate (attention.py forward)
    if pair.dim() != 4 or not cueq:
        fallback(L_TRI, f"triangle attention: pair{tuple(pair.shape)} cuequivariance={cueq} (stock forward ran: the vanilla route)")
        return orig(self, pair)
    if pair.shape[-3] <= rows:
        mark(L_TRI, f"whole:N={pair.shape[-3]}<=rows")
        return orig(self, pair)
    starting = bool(self.start_node)
    ch = _chunker()
    parts = ch.TriAttnParts(layer_norm=self.norm, bias=self.to_b, attention=lambda xr, bias_full, mr: _tri_attention_rows(self, xr, bias_full, starting))
    out = ch.triangle_attention_chunked(pair, None, parts, chunk=rows, starting=starting, record=chunk_record, lever=L_TRI)
    mark(L_TRI, f"{'start' if starting else 'end'}:rows={rows}")
    return self.to_out(out)


@register(L_TRI, family="chunk", exact="measured",
          exact_reason="measured, pending a deterministic comparison: per query row the statement is stock's (the softmax is whole over the key axis; the bias is full); the projections' GEMM runs at M = rows x N",
          applies=_tri_applies, description="the trunk's triangle attention in query-row blocks (opt_core.mem.chunk.triangle_attention_chunked on the stock cuEquivariance site; the lean line); under an arm with the fused kernel (gflash) the size gate: the kernel at or below `gate` tokens, the row-block statement above",
          preconditions=("hooks.module", "site_owned", "upstream_bytes", "rows", "gate"), settings=("rows", "gate"))
def _tri_apply(ctx):
    cls = ctx.hooks[L_TRI]["module"]
    STATE["tri_rows"] = _rows(ctx, L_TRI, 512, TRI_ROWS_RANGE)
    owner = _arm_engages(ctx, _modes.FPF_TAKES_CUEQ["cueq_triattn"])
    STATE["tri_gate"] = _tri_gate(ctx) if owner else 0                    # the gate applies only when the arm's kernel holds the site below it; otherwise the current forward is unchanged
    STATE["tri_owner"] = bool(_arm_engages(ctx, _modes.FPF_TAKES_CUEQ["cueq_triattn"]))   # the applied arm engages the fused triangle attention
    STATE["orig"]["tri_forward"] = cls.forward
    cls.forward = tri_forward_chunked

    def undo():
        cls.forward = STATE["orig"]["tri_forward"]
    settings = {"rows": STATE["tri_rows"]}
    if owner:                                                              # the arm engages the fused kernel: report the gate policy (0 = the kernel never holds the site)
        settings.update(gate=STATE["tri_gate"], below_gate=owner)
    return Applied(lever=L_TRI, settings=settings, sites=("TriangleAttention.forward",), undo=undo)


# ============================================================================================================ transition_chunk
L_TRANS = "transition_chunk"
TRANS_ROWS_RANGE = (16, 65536)                              # pair rows per block (default 256: at 4076 tokens one block's [256, N, 4c] hidden is ~2 GB instead of 34 GB whole)


def _trans_applies(ctx):
    """The stock pair transition (``rf3.model.layers.layer_utils.Transition``) on the EXACT base: under the FPF arm the site is the add-on's
    fused Triton transition (``ttr``) engaged the site is the add-on's, refused by name; big's arm disengages it (site_owned by property)."""
    h = ctx.require(L_TRANS, "module")
    owner = _arm_engages(ctx, ("ttr",))
    if owner:
        return refuse(L_TRANS, "site_owned", f"the applied FPF arm {ctx.extra['fpf_arm']!r} engages the transition site ({owner}): big's arm disengages it by lever property (big.DISENGAGED) — an arm that keeps it refuses here")
    for name in ("forward", "layer_norm_1", "linear_1", "linear_2", "linear_3"):
        if not hasattr(h["module"], name) and name == "forward":
            return refuse(L_TRANS, "upstream_bytes", f"Transition has no {name}")
    _rows(ctx, L_TRANS, 256, TRANS_ROWS_RANGE)
    return None


@register(L_TRANS, family="chunk", exact="measured",
          exact_reason="measured, pending a deterministic comparison: LayerNorm over channels, two linears, SiLU-gate and a linear are per element of the leading dims; row blocks change the GEMMs' M only",
          applies=_trans_applies, description="the stock Transition (pair / MSA / single) in row blocks of the leading dim (opt_core.mem.chunk.chunk_rows on the stock site; the lean line)",
          preconditions=("hooks.module", "site_owned", "upstream_bytes", "rows"), settings=("rows",))
def _trans_apply(ctx):
    cls = ctx.hooks[L_TRANS]["module"]
    STATE["trans_rows"] = _rows(ctx, L_TRANS, 256, TRANS_ROWS_RANGE)
    STATE["orig"]["trans_forward"] = cls.forward
    cls.forward = transition_forward_chunked

    def undo():
        cls.forward = STATE["orig"]["trans_forward"]
    return Applied(lever=L_TRANS, settings={"rows": STATE["trans_rows"]}, sites=("Transition.forward",), undo=undo)


def transition_forward_chunked(self, X):
    """``Transition.forward`` (stock ``layer_utils.py``) in row blocks along dim 0 of a ``[I, J, c]`` pair (the leading dim; batched inputs
    ``[B, I, J, c]`` chunk along dim 1); below the block size or on a 2-d input the stock statement runs whole (named)."""
    orig = STATE["orig"]["trans_forward"]
    rows = STATE["trans_rows"]
    if X.dim() < 3:
        mark(L_TRANS, f"whole:dim={X.dim()}")
        return orig(self, X)
    dim = 1 if X.dim() == 4 else 0
    if X.shape[dim] <= rows:
        mark(L_TRANS, f"whole:n={X.shape[dim]}<=rows")
        return orig(self, X)

    def block(xb):
        return orig(self, xb)

    out = _chunker().chunk_rows(block, X, dim, rows, exact="band", reason=TRANS_BAND_REASON, record=chunk_record, lever=L_TRANS, site="transition")
    mark(L_TRANS, f"rows={rows}")
    return out


# ============================================================================================================ feature_park
L_FP = "feature_park"
FP_CAST_KEYS = ("msa_stack", "profile", "template_distogram", "template_restype", "template_unit_vector")   # the feature tensors RF3.forward casts to the autocast dtype (RF3.py:154-164 / :394-404): the cast copy is the working tensor, the fp32 original stays referenced by the engine's batch (the L4000 memdiag's batch_to holders)
FP_STEP_KEYS = ("distogram_condition",)                      # read inside `recycle` only (the template embedder, pairformer_layers.py:772-776): parked between recycles and after the trunk
FP_POST_KEYS = FP_CAST_KEYS + ("msa", "has_distogram_condition", "distogram_condition_noise_scale")   # trunk-only: parked after the trunk for the diffusion + confidence phases (`msa` = the per-recycle slice view of msa_stack, RF3.py:264 — its storage is freed only once the view is parked too)
FP_MIN_BYTES = 64 << 20                                     # tensors below this stay on the device (a park costs two copies and a pinned buffer)


def _fp_applies(ctx):
    h = ctx.require(L_FP, "model", "trunk")
    for cls, meth in ((h["model"], "forward"), (h["trunk"], "trunk_forward_with_recycling")):
        if not callable(getattr(cls, meth, None)):
            return refuse(L_FP, "upstream_bytes", f"{cls.__name__} has no {meth}")
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return refuse(L_FP, "cuda", "no CUDA device: host parking has nothing to park from")
    try:
        _park_settings(ctx, cols="rowloop")
    except _offload.OffloadRefusal as e:
        return refuse(L_FP, "settings", str(e))
    return None


@register(L_FP, family="offload", exact="measured",
          exact_reason="copies only: a parked tensor returns byte-identical and the cast is the stock's own op (RF3.forward), done one statement earlier; measured until the det-recipe record",
          applies=_fp_applies, description="the trunk-only input feature tensors (msa_stack and its per-recycle slice, the template conditioning, the stock-cast originals the engine keeps referencing) parked on pinned host between their uses and after the trunk (opt_core.mem.offload.HostPark under the ONE pinned budget shared with confidence_offload)",
          preconditions=("hooks.model", "hooks.trunk", "upstream_bytes", "cuda", "settings"), settings=("pin_max_gb",))
def _fp_apply(ctx):
    model_cls, trunk_cls = ctx.hooks[L_FP]["model"], ctx.hooks[L_FP]["trunk"]
    staging = STATE.get("offload")
    settings = staging.s if staging is not None else _park_settings(ctx, cols="rowloop")
    STATE["fp_settings"] = settings
    STATE["fp_pool"] = staging.pool if staging is not None else _offload.PinPool(settings)   # ONE pinned budget per process — confidence_offload's staging and this park draw on the same PinPool; either refuses by name above it
    STATE["fp"] = None
    STATE["fp_counters"] = {"items": 0, "orig_parked": 0, "orig_dropped": 0, "step_parks": 0, "post_parks": 0, "unparks": 0, "refused": 0, "h2d_gb": 0.0, "d2h_gb": 0.0, "steps": 0}
    STATE["orig"]["fp_forward"] = model_cls.forward
    STATE["orig"]["fp_trunk"] = trunk_cls.trunk_forward_with_recycling
    model_cls.forward = fp_forward
    trunk_cls.trunk_forward_with_recycling = fp_trunk

    def undo():
        model_cls.forward = STATE["orig"]["fp_forward"]
        trunk_cls.trunk_forward_with_recycling = STATE["orig"]["fp_trunk"]
    return Applied(lever=L_FP, settings={"pin_max_gb": settings.pin_max_gb, "host_reserve_gb": settings.host_reserve_gb, "min_bytes": FP_MIN_BYTES,
                                         "budget": "the PinPool shared with confidence_offload" if staging is not None else "own PinPool (confidence_offload not applied)",
                                         "keys": {"cast": FP_CAST_KEYS, "step": FP_STEP_KEYS, "post": FP_POST_KEYS}},
                   sites=(f"{model_cls.__name__}.forward", f"{trunk_cls.__name__}.trunk_forward_with_recycling"), undo=undo)


def _fp_device_type():
    return STATE["fp_settings"].device.split(":")[0]


def _fp_candidate(t):
    torch = _torch()
    return torch.is_tensor(t) and t.device.type == _fp_device_type() and t.numel() * t.element_size() >= FP_MIN_BYTES


def _fp_park(park, name, t, kind):
    """Park ``t`` under ``name`` and re-point the tensor OBJECT at the pinned buffer (``t.data = host``: every holder of the object — the
    engine's batch, the features dict, a slice view's base — now sees the host copy and the device bytes return to the allocator);
    a refusal (budget, host RAM, non-contiguous) is named, the tensor stays on the device."""
    c = STATE["fp_counters"]
    try:
        ht = park.park(name, t)
    except _offload.OffloadRefusal as e:
        c["refused"] += 1
        fallback(L_FP, f"park {name} ({kind}): {e.name}: {e.reason} (the tensor stayed on the device)")
        return False
    t.data = ht.host
    c[kind] += 1
    return True


def _fp_unpark(park, name, t):
    """The parked ``name`` back on the device (H2D on the compute stream), the object re-pointed at the device copy."""
    t.data = park.unpark(name)
    STATE["fp_counters"]["unparks"] += 1


def fp_forward(self, input, *a, **k):
    """RF3WithConfidence.forward under the lever: the stock's autocast cast of the feature tensors done first (the same ``.to`` op —
    the stock statement then finds the cast tensor and returns it unchanged), each fp32 original the engine still references parked
    (an unreferenced one is simply dropped); the trunk wrapper parks the per-recycle tensors between steps and every trunk-only tensor
    after the trunk; the park is closed (every host buffer released) when the item returns."""
    torch = _torch()
    orig = STATE["orig"]["fp_forward"]
    f = input.get("f") if isinstance(input, dict) else None
    if not isinstance(f, dict) or STATE["fp"] is not None:
        fallback(L_FP, "forward: no input['f'] dict or a nested forward (stock forward ran)")
        return orig(self, input, *a, **k)
    park = _offload.HostPark(STATE["fp_settings"], tag=L_FP)
    park.pool = STATE["fp_pool"]
    try:
        park.check()
    except _offload.OffloadRefusal as e:
        fallback(L_FP, f"HostPark.check: {e.name}: {e.reason} (stock forward ran)")
        return orig(self, input, *a, **k)
    st = STATE["fp"] = {"park": park, "steps": 0}
    c = STATE["fp_counters"]
    c["items"] += 1
    if c["items"] == 1:                                              # the feature census of the first item: every device tensor of input["f"] at or above FP_MIN_BYTES by key (the record names the holders a future key list parks)
        c["feature_census"] = {k: [list(v.shape), str(v.dtype).replace("torch.", ""), round(v.numel() * v.element_size() / 1e9, 3)] for k, v in f.items() if _fp_candidate(v)}
    try:
        dev_type = next(self.parameters()).device.type
        if torch.is_autocast_enabled(dev_type):                      # the stock's check (RF3.forward) is the device's autocast state: "cuda" on the box; the CPU test line passes "cpu"
            dt = torch.get_autocast_dtype(dev_type)
            for x in FP_CAST_KEYS:
                t = f.get(x)
                if _fp_candidate(t) and t.dtype != dt:
                    f[x] = t.to(dt)                                 # the stock's own cast (RF3.forward), one statement earlier
                    if sys.getrefcount(t) > 2:                      # the dict entry is gone: any further reference is the engine's (the batch it keeps)
                        _fp_park(park, f"{x}.orig", t, "orig_parked")
                    else:
                        c["orig_dropped"] += 1
                    del t
        out = orig(self, input, *a, **k)
        mark(L_FP, f"steps={st['steps']} step_parks={c['step_parks']} post_parks={c['post_parks']} orig_parked={c['orig_parked']} orig_dropped={c['orig_dropped']} "
                   f"d2h_gb={round(park.d2h_bytes / 1e9, 2)} h2d_gb={round(park.h2d_bytes / 1e9, 2)}")   # the unit's mark: the lever acted on this item (a refusal on the way is a named fallback beside it)
        return out
    finally:
        c["steps"] += st["steps"]
        c["h2d_gb"] = round(c["h2d_gb"] + park.h2d_bytes / 1e9, 3)
        c["d2h_gb"] = round(c["d2h_gb"] + park.d2h_bytes / 1e9, 3)
        from opt_core.oom import is_oom
        try:
            park.close()
        except Exception as e:                                       # noqa: BLE001 — recorded, never masks the item's own exception; out-of-memory propagates
            if is_oom(e): raise
            STATE["fp_counters"]["close_error"] = repr(e)[:160]
        STATE["fp"] = None


def fp_trunk(self, f, *args, **kwargs):
    """RF3.trunk_forward_with_recycling under the lever: a generator around the stock generator — before each step the parked
    per-recycle tensors return to the device, after each step they are parked again; after the last step every trunk-only tensor
    (the cast msa_stack, its slice view `msa`, the template conditioning) is parked for the diffusion and confidence phases."""
    orig = STATE["orig"]["fp_trunk"]
    st = STATE["fp"]
    if st is None:                                                   # no item context (a direct trunk call): stock
        yield from orig(self, f, *args, **kwargs)
        return
    park = st["park"]
    step_keys = [x for x in FP_STEP_KEYS if _fp_candidate(f.get(x))]
    gen = orig(self, f, *args, **kwargs)
    while True:
        for x in step_keys:
            if x in park.tensors:
                _fp_unpark(park, x, f[x])
        try:
            out = next(gen)
        except StopIteration:
            break
        st["steps"] += 1
        for x in step_keys:
            _fp_park(park, x, f[x], "step_parks")
        yield out
    for x in FP_STEP_KEYS + FP_POST_KEYS:                             # the step keys too: the unpark before the exhausting next() brought them back (one extra H2D per item)
        t = f.get(x)
        if _fp_candidate(t) and x not in park.tensors:
            _fp_park(park, x, t, "post_parks")


def feature_park_counters() -> Optional[dict]:
    return dict(STATE["fp_counters"]) if STATE.get("fp_counters") else None


# ============================================================================================================ units
def install_units(ctx) -> None:
    """One census unit per model forward: ``RF3WithConfidence.forward`` opens ``fold<n>`` and closes it on return (any exit)."""
    from . import big as _big
    STATE["ctx"] = ctx
    model = ctx.hook(L_CONF, "model")
    if model is None:
        m = sys.modules.get("rf3.model.RF3")
        model = getattr(m, "RF3WithConfidence", None)
    if model is None:
        ctx.record.note("units: rf3.model.RF3.RF3WithConfidence not found — every event lands on the implicit unit `process`")
        return
    orig = model.forward
    STATE["orig"]["model_forward"] = orig

    def forward(self, *a, **k):
        _big.STATE["units"] += 1
        unit = f"{UNIT_STEM}{_big.STATE['units']}"
        if _big.STATE["units"] == 1:
            late_bind(ctx)
        ctx.record.unit_begin(unit)
        try:
            return orig(self, *a, **k)
        finally:
            ctx.record.unit_end(unit)

    model.forward = forward


def chunk_summary() -> dict:
    """The chunker's record folded (``opt_core.mem.chunk.summary`` over the kept entries) + the call count."""
    ent = STATE["chunk_entries"]
    if not ent:
        return {"calls": STATE["chunk_calls"], "kept": 0, "summary": {}}
    return {"calls": STATE["chunk_calls"], "kept": len(ent), "summary": _chunker().summary(ent), "last": ent[-1]}


def late_bind(ctx) -> None:
    """At the first unit (every upstream import complete): the engine module's own name for ``compile_af3_style_confidence_outputs``
    (bound by ``from … import`` at its import) re-pointed to the offload wrapper when it still names the stock function; recorded."""
    if "compile" not in STATE["orig"]:
        return
    engine = sys.modules.get("rf3.inference_engines.rf3")
    if engine is None:
        ctx.record.note("confidence_offload: rf3.inference_engines.rf3 not imported at the first unit (no engine name to re-point)")
        return
    cur = getattr(engine, "compile_af3_style_confidence_outputs", None)
    if cur is STATE["orig"]["compile"]:
        engine.compile_af3_style_confidence_outputs = compile_offload
        ctx.record.note("confidence_offload: rf3.inference_engines.rf3.compile_af3_style_confidence_outputs re-pointed to the offload wrapper at the first unit")
    elif cur is compile_offload:
        ctx.record.note("confidence_offload: the engine's compile_af3_style_confidence_outputs already names the offload wrapper")
    else:
        ctx.record.note(f"confidence_offload: the engine names {cur!r} for compile_af3_style_confidence_outputs (neither stock nor the wrapper): per-sample compile will not run through the engine")
