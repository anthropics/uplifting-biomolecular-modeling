"""Row-block reductions for co-folding confidence summaries (BLOCKREDUCE).

Tensors in / tensors out; no model-package imports.  OpenFold3's key names,
dict ordering and JSON writing live in the adaptor (of3o_confidence.py).

Problem.  With the pair representation row-sharded over P ranks, the PAE / PDE logits
[N, N, b] (b = 64) must never be materialised for large N.  Every summary statistic
OpenFold3 derives from them is, however, a function of

    E[i, j]    = sum_b softmax(logits[i, j, :])[b] * center[b]          (expected PAE / PDE)
    TM_d[i, j] = sum_b softmax(pae_logits[i, j, :])[b] * w_d[b],         w_d[b] = 1 / (1 + (center[b] / d0(N_d))^2)

where d0 is the TM-score normalisation and N_d the size of the token subset ("context") the
statistic is evaluated on.  The contexts such a summary enumerates are

    FULL          N_d = N            -> ptm  (mean_j TM, max over rows with a frame)
                                     -> iptm (mean over j in a different chain, max over framed rows)
    CHAIN a       N_d = |a|          -> chain_ptm[a]
    PAIR (a, b)   N_d = |a| + |b|    -> chain_pair_iptm[a, b]  (a < b; symmetric fill)
                                        -> chain_iptm[a] (mean over pairs touching a whose first chain has a frame)
                                        -> chain_pair_iptm_global (ligand-aware mix of chain_iptm)
    gpde family   sum_ij pde_ij * contact_ij / sum_ij contact_ij over (all | chain a x chain a | chain a x chain b)

`RowBlockReducer.consume(c0, c1, pae_logits_rows, pde_logits_rows, contact_rows)` is called by
the owning rank for consecutive row blocks [c0, c1) of its shard; only [c, N, b] tensors exist
at any time.  `RowBlockReducer.finalize(bounds)` (collective; every rank calls it with the
list of per-rank (q0, q1) row bounds) returns the finished statistics on rank 0.

Two finishing regimes (both produce the same keys):

  finish="exact"   keep the fp32 row pieces TM_d[i, cols(d)] (about 3N floats per row in
                   total), gather each context's [N_d, N_d] matrix to rank 0 (one context at a
                   time, freed immediately) and apply the single-device post-contraction
                   statements verbatim on tensors of the SAME shape the single-device code
                   uses -> the reductions run in the same order -> bitwise-identical results
                   whenever the per-element steps (softmax over b, (p*w).sum(-1), p @ centers)
                   are shape-invariant.  Rank-0 transient:
                   max_d N_d^2 fp32 (<= 4 N^2 bytes) + the [N, N] fp32 expected-PDE and
                   contact matrices for the gpde family.
  finish="rowsum"  reduce every piece to per-row sums on the owning rank and gather only
                   O(N * n_ctx) scalars + [C, C] tables; finish with sum / count.  Summation
                   order differs from the single-device code -> last-bit differences
                   (tolerance-class numerics), minimal traffic / memory.

int32 rule: callers must choose the row block c such that c * N * b < 2**31 (asserted).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

FULL, CHAIN, PAIR = "full", "chain", "pair"


# ----------------------------------------------------------------------------- TM helpers
def tm_bin_centers(min_bin: float, max_bin: float, no_bins: int) -> torch.Tensor:
    """Bin centres exactly as the model computes them (CPU fp32 linspace + half width)."""
    bin_width = (max_bin - min_bin) / no_bins
    boundaries = torch.linspace(start=min_bin, end=max_bin - bin_width, steps=no_bins)
    return boundaries + 0.5 * bin_width


def tm_d0(N: int) -> float:
    """TM-score normalisation constant d0(N) = 1.24 (max(N,19) - 15)^(1/3) - 1.8."""
    return 1.24 * (max(N, 19) - 15) ** (1 / 3) - 1.8


def tm_bin_weight(N_d: int, centers_cpu: torch.Tensor, device) -> torch.Tensor:
    """w_d[b] = 1 / (1 + (center_b / d0(N_d))^2), computed on CPU fp32 then moved (the model's order)."""
    return (1 / (1 + (centers_cpu / tm_d0(N_d)) ** 2)).to(device)


def expected_value_rows(logits_rows: torch.Tensor, centers_dev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """softmax over bins -> (prob [1,c,N,b], prob @ centers [1,c,N]); fp32; leading batch dim kept
    so the statements are literally the single-device ones."""
    prob = torch.nn.functional.softmax(logits_rows.unsqueeze(0), dim=-1)
    return prob, prob @ centers_dev


# ----------------------------------------------------------------------------- chains / contexts
@dataclass
class Context:
    kind: str
    a: int = -1
    b: int = -1
    mask: Optional[torch.Tensor] = None   # bool [N] (None for FULL)
    N_d: int = 0
    w: Optional[torch.Tensor] = None      # [b] on device


class ChainIndex:
    """Chain bookkeeping shared by all reductions (remap to contiguous ids as the model does)."""

    def __init__(self, asym_id: torch.Tensor, has_frame: torch.Tensor,
                 token_is_ligand: Optional[torch.Tensor] = None):
        asym_raw = asym_id.long()
        uniq = torch.unique(asym_raw)
        if len(uniq) != int(asym_raw.max().item()) + 1:
            remap = {old.item(): new for new, old in enumerate(uniq)}
            asym = torch.tensor([remap[x.item()] for x in asym_raw], dtype=torch.long, device=asym_raw.device)
        else:
            asym = asym_raw
        self.N = int(asym.shape[0])
        self.asym_raw = asym_raw
        self.asym = asym
        self.C = int(len(torch.unique(asym)))
        self.masks = [asym == a for a in range(self.C)]
        self.sizes = [int(m.sum().item()) for m in self.masks]
        self.has_frame = has_frame.bool()
        self.chain_has_frame = [bool((self.masks[a] & self.has_frame).any().item()) for a in range(self.C)]
        if token_is_ligand is None:
            token_is_ligand = torch.zeros_like(asym, dtype=torch.bool)
        self.token_is_ligand = token_is_ligand.bool()
        self.chain_is_ligand = [
            bool((self.token_is_ligand[self.masks[a]].sum() >= self.masks[a].sum() // 2).item())
            for a in range(self.C)
        ]

    def contexts(self, centers_cpu: torch.Tensor, device) -> List[Context]:
        ctxs = [Context(FULL, N_d=self.N, w=tm_bin_weight(self.N, centers_cpu, device))]
        for a in range(self.C):
            ctxs.append(Context(CHAIN, a=a, mask=self.masks[a], N_d=self.sizes[a],
                                w=tm_bin_weight(self.sizes[a], centers_cpu, device)))
        for a in range(self.C):
            for b in range(a + 1, self.C):
                n = self.sizes[a] + self.sizes[b]
                ctxs.append(Context(PAIR, a=a, b=b, mask=self.masks[a] | self.masks[b], N_d=n,
                                    w=tm_bin_weight(n, centers_cpu, device)))
        return ctxs


# ----------------------------------------------------------------------------- P2P varlen gather
def _world() -> Tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    return 1, 0


def gather_cat_to_rank0(piece: torch.Tensor, counts: Sequence[int]) -> Optional[torch.Tensor]:
    """Rank q contributes `piece` ([counts[q], ...]); rank 0 receives the rank-ordered concat."""
    P, rank = _world()
    piece = piece.contiguous()
    assert piece.shape[0] == counts[rank], (tuple(piece.shape), list(counts), rank)
    if P == 1:
        return piece
    if rank == 0:
        full = piece.new_empty((int(sum(counts)),) + tuple(piece.shape[1:]))
        o = 0
        for q in range(P):
            n = int(counts[q])
            if n > 0:
                if q == 0:
                    full[o:o + n] = piece
                else:
                    dist.recv(full[o:o + n], src=q)
            o += n
        return full
    if counts[rank] > 0:
        dist.send(piece, dst=0)
    return None


# ----------------------------------------------------------------------------- reducer
class RowBlockReducer:
    """Per-sample accumulator of row-block statistics on the rank owning global rows [r0, r1)."""

    def __init__(self, chains: ChainIndex, r0: int, r1: int, device, *,
                 pae_bins=(0.0, 32.0, 64), pde_bins=(0.0, 32.0, 64),
                 finish: str = "exact", eps: float = 1e-8,
                 keep_value_rows: bool = True):
        assert finish in ("exact", "rowsum"), finish
        self.chains, self.r0, self.r1, self.R = chains, int(r0), int(r1), int(r1 - r0)
        self.N, self.finish, self.eps = chains.N, finish, eps
        self.device = torch.device(device)
        self.pae_bins, self.pde_bins = tuple(pae_bins), tuple(pde_bins)
        self.centers_pae_cpu = tm_bin_centers(*self.pae_bins)
        self.centers_pde_cpu = tm_bin_centers(*self.pde_bins)
        self.centers_pae = self.centers_pae_cpu.to(self.device)
        self.centers_pde = self.centers_pde_cpu.to(self.device)
        self.ctxs = chains.contexts(self.centers_pae_cpu, self.device)
        self.keep_value_rows = keep_value_rows
        N, R = self.N, self.R
        if keep_value_rows:
            self.pae_rows_f16 = torch.empty((R, N), dtype=torch.float16, device=self.device)
            self.pde_rows_f16 = torch.empty((R, N), dtype=torch.float16, device=self.device)
            self.contact_rows_f16 = torch.empty((R, N), dtype=torch.float16, device=self.device)
        if finish == "exact":
            self.pde_rows_f32 = torch.empty((R, N), dtype=torch.float32, device=self.device)
            self.contact_rows_f32 = torch.empty((R, N), dtype=torch.float32, device=self.device)
            self.pieces: Dict[int, List[torch.Tensor]] = {k: [] for k in range(len(self.ctxs))}
        else:
            C = chains.C
            self.rs_num: Dict[int, List[torch.Tensor]] = {k: [] for k in range(len(self.ctxs))}
            self.rs_full_mean_num: List[torch.Tensor] = []     # sum_j TM_N[i, j]            (ptm)
            self.gp_num = torch.zeros((R, C), dtype=torch.float32, device=self.device)  # sum_{j in x} pde*contact
            self.gp_den = torch.zeros((R, C), dtype=torch.float32, device=self.device)  # sum_{j in x} contact
        self._next = self.r0

    # ------------------------------------------------------------------ consume one row block
    @torch.no_grad()
    def consume(self, c0: int, c1: int, pae_logits_rows: torch.Tensor, pde_logits_rows: torch.Tensor,
                contact_rows: torch.Tensor) -> None:
        """pae/pde logits rows [c, N, b] (fp32), contact rows [c, N] (fp32) for global rows [c0, c1)."""
        assert c0 == self._next and self.r0 <= c0 < c1 <= self.r1, (c0, c1, self._next, self.r0, self.r1)
        c = c1 - c0
        assert pae_logits_rows.shape[0] == c and pae_logits_rows.shape[1] == self.N
        assert pae_logits_rows.numel() < 2 ** 31, "int32 rule: row block too large"
        lo, hi = c0 - self.r0, c1 - self.r0
        dev_type = "cuda" if pae_logits_rows.is_cuda else "cpu"
        with torch.autocast(device_type=dev_type, enabled=False):
            pae_logits_rows = pae_logits_rows.to(torch.float32)
            pde_logits_rows = pde_logits_rows.to(torch.float32)
            contact_rows = contact_rows.to(torch.float32)
            # expected values (the model's statement: prob = softmax(logits); score = prob @ centers)
            pde_prob, pde_val = expected_value_rows(pde_logits_rows, self.centers_pde)   # [1,c,N,b], [1,c,N]
            del pde_prob
            pae_prob, pae_val = expected_value_rows(pae_logits_rows, self.centers_pae)
            if self.keep_value_rows:
                self.pae_rows_f16[lo:hi] = pae_val[0].to(torch.float16)
                self.pde_rows_f16[lo:hi] = pde_val[0].to(torch.float16)
                self.contact_rows_f16[lo:hi] = contact_rows.to(torch.float16)
            if self.finish == "exact":
                self.pde_rows_f32[lo:hi] = pde_val[0]
                self.contact_rows_f32[lo:hi] = contact_rows
            else:
                pc = pde_val[0] * contact_rows                                  # [c, N]
                for x in range(self.chains.C):
                    m = self.chains.masks[x]
                    self.gp_num[lo:hi, x] = pc[:, m].sum(dim=-1)
                    self.gp_den[lo:hi, x] = contact_rows[:, m].sum(dim=-1)
            # TM contexts
            for k, ctx in enumerate(self.ctxs):
                if ctx.kind == FULL:
                    sub = pae_prob                                              # [1,c,N,b]
                    rows_asym = self.chains.asym_raw[c0:c1]
                    cols_asym = self.chains.asym_raw
                else:
                    rsel = ctx.mask[c0:c1]
                    if not bool(rsel.any()):
                        continue
                    sub = pae_prob[:, rsel][:, :, ctx.mask]                     # [1,n,N_d,b]
                    rows_asym = self.chains.asym[c0:c1][rsel]
                    cols_asym = self.chains.asym[ctx.mask]
                T = (sub * ctx.w).sum(dim=-1)                                   # [1,n,N_d]  (the model's statement)
                if self.finish == "exact":
                    self.pieces[k].append(T[0])
                else:
                    if ctx.kind == FULL:
                        self.rs_full_mean_num.append(T[0].sum(dim=-1))
                    if ctx.kind == CHAIN:
                        self.rs_num[k].append(T[0].sum(dim=-1))
                    else:  # FULL (iptm) and PAIR: different-chain columns only
                        is_diff = cols_asym[None, :] != rows_asym[:, None]      # [n, N_d]
                        self.rs_num[k].append((T[0] * is_diff).sum(dim=-1))
                del sub, T
            del pae_prob, pae_val, pde_val
        self._next = c1

    # ------------------------------------------------------------------ helpers
    def _counts(self, ctx: Context, bounds: Sequence[Tuple[int, int]]) -> List[int]:
        if ctx.kind == FULL:
            return [q1 - q0 for (q0, q1) in bounds]
        return [int(ctx.mask[q0:q1].sum().item()) for (q0, q1) in bounds]

    def _cat_or_empty(self, lst: List[torch.Tensor], tail_shape: Tuple[int, ...]) -> torch.Tensor:
        if len(lst):
            return torch.cat(lst, dim=0)
        return torch.empty((0,) + tuple(tail_shape), dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------ single-device finishing statements
    def _finish_ptm_T(self, T: torch.Tensor, has_frame: torch.Tensor) -> torch.Tensor:
        """T [1, N_d, N_d] -> ptm  (T.mean(-1)[..., has_frame].max(-1))"""
        if int(has_frame.sum()) == 0:
            return torch.zeros(size=T.shape[:-2], device=T.device)
        return T.mean(dim=-1)[..., has_frame].max(dim=-1).values

    def _finish_iptm_T(self, T: torch.Tensor, has_frame: torch.Tensor, asym: torch.Tensor) -> torch.Tensor:
        if int(has_frame.sum()) == 0:
            return torch.zeros(size=T.shape[:-2], device=T.device)
        is_diff_chain = asym[None, :] != asym[:, None]
        iptm = (T * is_diff_chain).sum(dim=-1) / (self.eps + is_diff_chain.sum(dim=-1))
        return iptm[..., has_frame].max(dim=-1).values

    def _finish_rowsum(self, num: torch.Tensor, den, has_frame: torch.Tensor) -> torch.Tensor:
        """num [N_d] per-row sums, den scalar or [N_d] -> max over framed rows of num/den (the rowsum regime's order)."""
        if int(has_frame.sum()) == 0 or num.numel() == 0:
            return torch.zeros((1,), device=self.device)
        v = (num / den).unsqueeze(0)
        return v[..., has_frame].max(dim=-1).values

    # ------------------------------------------------------------------ finalize (collective)
    @torch.no_grad()
    def finalize(self, bounds: Sequence[Tuple[int, int]], *, collect_full: bool = True) -> Optional[dict]:
        """Collective over all ranks (each holding the reducer of its own shard, same sample).

        bounds: per-rank (q0, q1) global row ranges in rank order (contiguous, ascending).
        Returns on rank 0 a dict with batch-dim-1 tensors:
          ptm [1], iptm [1], chain_ptm [1,C], chain_iptm [1,C], chain_pair_iptm [1,C,C],
          chain_pair_iptm_global [1,C,C], gpde [1], chain_gpde [1,C], chain_pair_gpde [1,C,C],
          and (exact) token_pair_pde_f32 [1,N,N], contact_probs_f32 [N,N] on device,
          and (collect_full) token_pair_pae_f16 / token_pair_pde_f16 / contact_probs_f16 [N,N] on CPU.
        Other ranks return None.
        """
        assert self._next == self.r1, f"rows [{self._next},{self.r1}) were never consumed"
        P, rank = _world()
        ch, dev = self.chains, self.device
        C = ch.C
        is0 = rank == 0
        out = {}
        batch = (1,)
        chain_pair_iptm = torch.zeros(size=batch + (C, C)).to(dev)
        chain_ptm = torch.zeros(size=batch + (C,)).to(dev)
        full_counts = [q1 - q0 for (q0, q1) in bounds]

        # ---- TM family, one context at a time
        for k, ctx in enumerate(self.ctxs):
            counts = self._counts(ctx, bounds)
            if ctx.kind == FULL:
                hf, asym_sub = ch.has_frame, ch.asym_raw
            else:
                hf, asym_sub = ch.has_frame[ctx.mask], ch.asym[ctx.mask]
            if self.finish == "exact":
                piece = self._cat_or_empty(self.pieces[k], (ctx.N_d,))
                Tm = gather_cat_to_rank0(piece, counts)
                self.pieces[k] = []          # free on every rank
                del piece
                if is0:
                    Tm = Tm.unsqueeze(0)     # [1, N_d, N_d]  == single-device token_token_ptm
                    if ctx.kind == FULL:
                        out["ptm"] = self._finish_ptm_T(Tm, hf)
                        out["iptm"] = self._finish_iptm_T(Tm, hf, asym_sub)
                    elif ctx.kind == CHAIN:
                        chain_ptm[:, ctx.a] = self._finish_ptm_T(Tm, hf)
                    else:
                        chain_pair_iptm[:, ctx.a, ctx.b] = self._finish_iptm_T(Tm, hf, asym_sub)
                del Tm
            else:
                num = gather_cat_to_rank0(self._cat_or_empty(self.rs_num[k], ()), counts)
                if ctx.kind == FULL:
                    mnum = gather_cat_to_rank0(self._cat_or_empty(self.rs_full_mean_num, ()), counts)
                if is0:
                    if ctx.kind == FULL:
                        out["ptm"] = self._finish_rowsum(mnum, float(ctx.N_d), hf)
                        sizes = torch.tensor(ch.sizes, device=dev)
                        nd = ctx.N_d - sizes[ch.asym]                       # different-chain column count per row
                        out["iptm"] = self._finish_rowsum(num, self.eps + nd, hf)
                    elif ctx.kind == CHAIN:
                        chain_ptm[:, ctx.a] = self._finish_rowsum(num, float(ctx.N_d), hf)
                    else:
                        sizes = torch.tensor(ch.sizes, device=dev)
                        nd = ctx.N_d - sizes[asym_sub]
                        chain_pair_iptm[:, ctx.a, ctx.b] = self._finish_rowsum(num, self.eps + nd, hf)
        if is0:
            # symmetric fill, chain_iptm, chain_pair_iptm_global: single-device loops verbatim
            for a1 in range(C):
                for a2 in range(C):
                    if a1 > a2:
                        chain_pair_iptm[:, a1, a2] = chain_pair_iptm[:, a2, a1]
            chain_iptm = torch.zeros(size=batch + (C,)).to(dev)
            for aid in range(C):
                pairs = [(i, j) for i in range(C) for j in range(C)
                         if (i == aid or j == aid) and (i != j) and ch.chain_has_frame[i]]
                vals = [chain_pair_iptm[:, i, j] for (i, j) in pairs]
                if len(vals) > 0:
                    chain_iptm[:, aid] = torch.stack(vals, dim=-1).mean(dim=-1)
            chain_pair_iptm_global = torch.zeros(size=batch + (C, C)).to(dev)
            for a1 in range(C):
                for a2 in range(C):
                    if a1 == a2:
                        continue
                    if ch.chain_is_ligand[a1]:
                        chain_pair_iptm_global[:, a1, a2] = chain_iptm[:, a1]
                    elif ch.chain_is_ligand[a2]:
                        chain_pair_iptm_global[:, a1, a2] = chain_iptm[:, a2]
                    else:
                        chain_pair_iptm_global[:, a1, a2] = (chain_iptm[:, a1] + chain_iptm[:, a2]) * 0.5
            out.update({"chain_ptm": chain_ptm, "chain_iptm": chain_iptm,
                        "chain_pair_iptm": chain_pair_iptm, "chain_pair_iptm_global": chain_pair_iptm_global})

        # ---- gpde family
        if self.finish == "exact":
            pde_full = gather_cat_to_rank0(self.pde_rows_f32, full_counts)
            contact_full = gather_cat_to_rank0(self.contact_rows_f32, full_counts)
            if is0:
                pde_full = pde_full.unsqueeze(0)                                 # [1, N, N]
                out["token_pair_pde_f32"] = pde_full
                out["contact_probs_f32"] = contact_full                          # [N, N]
                out.update(gpde_from_full(pde_full, contact_full, ch, self.eps))
        else:
            num = gather_cat_to_rank0(self.gp_num, full_counts)                  # [N, C]
            den = gather_cat_to_rank0(self.gp_den, full_counts)
            if is0:
                out["gpde"] = (num.sum() / den.sum()).reshape(1)
                chain_gpde = torch.zeros(size=batch + (C,), device=dev)
                chain_pair_gpde = torch.zeros(size=batch + (C, C), device=dev)
                for a in range(C):
                    ra = ch.masks[a]
                    chain_gpde[:, a] = num[ra, a].sum() / (den[ra, a].sum() + self.eps)
                    for b in range(C):
                        if a == b:
                            continue
                        if b < a:
                            chain_pair_gpde[:, a, b] = chain_pair_gpde[:, b, a]
                            continue
                        chain_pair_gpde[:, a, b] = num[ra, b].sum() / (den[ra, b].sum() + self.eps)
                out["chain_gpde"], out["chain_pair_gpde"] = chain_gpde, chain_pair_gpde

        # ---- full matrices for the full_data replacement (fp16, CPU on rank 0)
        if collect_full and self.keep_value_rows:
            for name, rows in (("token_pair_pae_f16", self.pae_rows_f16), ("token_pair_pde_f16", self.pde_rows_f16),
                               ("contact_probs_f16", self.contact_rows_f16)):
                g = gather_cat_to_rank0(rows, full_counts)
                if is0:
                    out[name] = g.cpu()
                del g
        return out if is0 else None


# ----------------------------------------------------------------------------- gpde on full matrices
def gpde_from_full(token_pair_pde: torch.Tensor, contact_probs: torch.Tensor, ch: ChainIndex, eps: float = 1e-8) -> dict:
    """Single-device gpde statements on full matrices: token_pair_pde [1,N,N], contact_probs [N,N]."""
    out = {"gpde": (token_pair_pde * contact_probs).sum(dim=[-1, -2]) / contact_probs.sum(dim=[-1, -2])}
    C = ch.C
    batch_shape = token_pair_pde.shape[:-2]
    device = token_pair_pde.device

    def _cal(m1, m2):
        mc = contact_probs[..., m1, :][..., m2]
        mp = token_pair_pde[..., m1, :][..., m2]
        return (mp * mc).sum(dim=(-1, -2)) / (mc.sum(dim=(-1, -2)) + eps)

    chain_gpde = torch.zeros(size=batch_shape + (C,), device=device)
    for a in range(C):
        chain_gpde[..., a] = _cal(ch.masks[a], ch.masks[a])
    chain_pair_gpde = torch.zeros(size=batch_shape + (C, C), device=device)
    for a in range(C):
        for b in range(C):
            if a == b:
                continue
            if b < a:
                chain_pair_gpde[..., a, b] = chain_pair_gpde[..., b, a]
                continue
            chain_pair_gpde[..., a, b] = _cal(ch.masks[a], ch.masks[b])
    out["chain_gpde"], out["chain_pair_gpde"] = chain_gpde, chain_pair_gpde
    return out


def max_row_chunk(N: int, bins: int = 64, cap: int = 128) -> int:
    """Largest row block <= cap honouring the int32 rule (c * N * bins < 2**31) on a 128-grid divisor."""
    c = min(cap, max(1, (2 ** 31 - 1) // (N * bins)))
    for d in (128, 64, 32, 16, 8, 4, 2, 1):
        if d <= c:
            return d
    return 1
