"""The fold's host-side tail levers (registry: ``rankcc``, ``tailasync``; modes.KitMode.postproc) — what upstream's ``run_folding_on_context``
does AFTER the confidence head, on CPU tensors it has already moved off the GPU: per diffusion sample ``rank()`` (pTM / ipTM / pLDDT summaries
and the inter-chain clash census), ``save_to_cif`` (modelcif), ``np.savez``; once per fold ``plot_msa`` (matplotlib). Both levers are
install-style rebindings of the module attributes upstream's fold body calls (``chai_lab.chai1.rank``, ``.save_to_cif``, ``.plot_msa``,
``.StructureCandidates``); ``run_folding_on_context`` itself is untouched and every file it writes has the same bytes.

``rankcc``   upstream's ranker with its clash census counted on the atoms that exist. ``chai_lab.ranking.clashes.get_scores`` builds, per
             sample, the padded a x a distance matrix (a = 23 * crop: 23,552 atoms at crop 1024), an a x a validity mask, its
             int32 copy and an a x a int64 index, then two ``scatter_add_`` passes over them (single-threaded on CPU) to count clashing atom
             pairs per chain pair. The lever computes the SAME predicate — upstream's own ``cdist`` statement (``torch.cdist(…,
             compute_mode='donot_use_mm_for_euclid_dist')``: each pair's distance is computed from its own two coordinate rows, so the rows
             that exist give the same bits whether or not the padded rows ride along) ``< clash_threshold`` on existing, distinct atoms — and
             counts the clashing pairs per (chain, chain) with one integer ``bincount``: identical integers, then upstream's own integer /
             ratio statements verbatim (same dtypes). pTM / ipTM / pLDDT are upstream's calls unchanged. Exact by construction (integer counts
             of a bit-identical predicate); the FIRST sample of a process is also ranked by upstream's statement and the four clash outputs
             bit-compared (``torch.equal``) — a mismatch refuses the lever by name for the rest of the process (``fallback_by=witness``) and
             upstream's result is used; leading batch dims other than ``[1]`` or CUDA operands take upstream's statement by name (``batch`` /
             ``cuda``), counted.
``tailasync`` the writers off the critical path: ``save_to_cif`` (pure-Python modelcif, CPU-bound) and ``plot_msa`` are
             submitted, in call order, to ONE writer thread and return at once, so sample k's CIF is written while sample k+1 is ranked (torch's
             CPU kernels release the GIL); the fold's ``StructureCandidates`` is constructed only after every pending write has finished
             (``__post_init__`` joins the writer; a writer's exception is re-raised there, as the synchronous call would have raised it). Same
             functions, same arguments -> the same bytes; only WHEN they run moves.

LEVER-line evidence (``evidence``) / EXIT tally (``tally_fields``): ``rankcc_served= rankcc_fallback= [rankcc_fallback_by=…] rankcc_witness=
pass|fail|none``; ``tailasync_cif= tailasync_plot= tailasync_join_s=``.
"""
from __future__ import annotations

import sys
import threading
import time
from typing import Dict, Optional, Tuple

LEVERS = ("rankcc", "tailasync")
RANK_NAME = "rank_cc"                      # __name__ of the installed ranker (registry probe: ("attr", "chai_lab.chai1", "rank", RANK_NAME))
CIF_NAME = "save_to_cif_async"             # (registry probe: ("attr", "chai_lab.chai1", "save_to_cif", CIF_NAME))
EXPECTED_FALLBACKS = {"rankcc": ("batch", "cuda"), "tailasync": ()}
_STATE: Dict[str, object] = {"installed": (), "orig": {}, "rank": None, "tail": None}


class LeverUnavailable(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------------------------------------- rankcc
class _RankState:
    def __init__(self):
        self.served = 0; self.fallback_by: Dict[str, int] = {}; self.witness = "none"; self.refused = None; self.seconds = 0.0

    def fallback(self, why):
        self.fallback_by[why] = self.fallback_by.get(why, 0) + 1


def clash_scores_cc(atom_coords, atom_mask, atom_asym_id, atom_entity_type, clash_threshold=1.1, max_clashes=100, max_clash_ratio=0.5):
    """``chai_lab.ranking.clashes.get_scores`` with the clash census counted on existing atoms (see the module doc): the same ClashScores, the
    same integers and dtypes. Leading batch dims must be ``[1]`` and operands on CPU (the fold's own call); anything else -> ValueError (the
    caller takes upstream's statement by name)."""
    import torch
    from einops import reduce
    import chai_lab.ranking.clashes as clashes
    from chai_lab.utils.tensor_utils import cdist
    assert atom_asym_id.dtype in (torch.int32, torch.int64)
    atom_asym_id = (atom_asym_id - 1).to(torch.int64)
    assert torch.amin(atom_asym_id) >= 0
    n_chains = atom_asym_id.amax().add(1).item()
    assert isinstance(n_chains, int)
    *b, a = atom_mask.shape
    if list(b) != [1]:
        raise ValueError("batch")
    if atom_coords.is_cuda or atom_mask.is_cuda:
        raise ValueError("cuda")
    idx = atom_mask[0].nonzero(as_tuple=False).squeeze(-1)                      # the atoms that exist, ascending
    xr = atom_coords[0].index_select(0, idx).unsqueeze(0)                       # [1, n, 3]
    d = cdist(xr)[0]                                                             # upstream's statement: torch.cdist(x, x, donot_use_mm) — per-pair arithmetic, the padded rows' absence changes no bit
    close = d < clash_threshold                                                  # upstream: valid_mask & ~eye & (pairwise_dists < clash_threshold)
    close.fill_diagonal_(False)
    rows, cols = close.nonzero(as_tuple=True)
    ci = atom_asym_id[0].index_select(0, idx)
    ids = ci.index_select(0, rows) * n_chains + ci.index_select(0, cols)
    cc = torch.bincount(ids, minlength=n_chains * n_chains).view(1, n_chains, n_chains)
    clashes_chain_chain = cc.to(torch.int32)                                     # == upstream's two scatter_add_ passes over the a x a int32 matrix (ordered pairs per chain pair)
    total_clashes = reduce(clashes_chain_chain, "... i j -> ...", "sum") // 2   # upstream's statements from here on, verbatim
    clashes_chain_chain = clashes_chain_chain // (1 + torch.diag(clashes_chain_chain.new_ones(n_chains)))
    non_diag = 1 - torch.diag(clashes_chain_chain.new_ones(n_chains))
    inter_chain_chain = non_diag * clashes_chain_chain
    inter_chain_clashes = reduce(inter_chain_chain, "... i j -> ... ", "sum") // 2
    return clashes.ClashScores(
        total_clashes=total_clashes,
        total_inter_chain_clashes=inter_chain_clashes,
        chain_chain_clashes=clashes_chain_chain,
        has_inter_chain_clashes=clashes.has_inter_chain_clashes(
            atom_mask=atom_mask, atom_asym_id=atom_asym_id, atom_entity_type=atom_entity_type,
            per_chain_pair_clashes=inter_chain_chain, max_clashes=max_clashes, max_clash_ratio=max_clash_ratio),
    )


def _clash_equal(a, b) -> bool:
    import torch
    for f in ("total_clashes", "total_inter_chain_clashes", "chain_chain_clashes", "has_inter_chain_clashes"):
        x, y = getattr(a, f), getattr(b, f)
        if x.dtype != y.dtype or x.shape != y.shape or not torch.equal(x, y):
            return False
    return True


def _make_rank(st: _RankState):
    def rank_cc(atom_coords, atom_mask, atom_token_index, token_exists_mask, token_asym_id, token_entity_type, token_valid_frames_mask,
                lddt_logits, lddt_bin_centers, pae_logits, pae_bin_centers, clash_threshold: float = 1.1, max_clashes: int = 100,
                max_clash_ratio: float = 0.5):
        """chai_lab.ranking.rank.rank — statement for statement, the clash census through clash_scores_cc (rankcc)."""
        import torch
        import chai_lab.ranking.clashes as clashes
        import chai_lab.ranking.plddt as plddt
        import chai_lab.ranking.ptm as ptm
        import chai_lab.ranking.utils as rank_utils
        from chai_lab.ranking.rank import SampleRanking
        upstream_clashes = clashes.get_scores
        t0 = time.perf_counter()
        ptm_scores = ptm.get_scores(pae_logits=pae_logits, token_exists_mask=token_exists_mask, valid_frames_mask=token_valid_frames_mask,
                                    bin_centers=pae_bin_centers, token_asym_id=token_asym_id)
        atom_asym_id = torch.gather(token_asym_id, dim=-1, index=atom_token_index.long())
        ckw = dict(atom_coords=atom_coords, atom_mask=atom_mask, atom_asym_id=atom_asym_id,
                   atom_entity_type=torch.gather(token_entity_type, dim=-1, index=atom_token_index.long()),
                   max_clashes=max_clashes, max_clash_ratio=max_clash_ratio, clash_threshold=clash_threshold)
        clash_scores = None
        if st.refused is None:
            try:
                clash_scores = clash_scores_cc(**ckw)
            except ValueError as e:                                              # batch / cuda: upstream's statement by name
                st.fallback(str(e) if str(e) in EXPECTED_FALLBACKS["rankcc"] else "error:" + type(e).__name__)
            if clash_scores is not None and st.witness == "none":              # first served sample of the process: bit-compare against upstream's statement
                ref = upstream_clashes(**ckw)
                if _clash_equal(clash_scores, ref):
                    st.witness = "pass"
                else:
                    st.witness = "fail"; st.refused = "witness"
                    sys.stderr.write("[chai1-opt] LEVER rankcc REFUSED BY NAME: the clash census on existing atoms did not bit-match upstream's statement on "
                                     "this process's first sample; upstream's ranker serves the rest of the process (fallback_by=witness)\n")
                    clash_scores = ref
        if clash_scores is None:
            if st.refused is not None:
                st.fallback(st.refused)
            clash_scores = upstream_clashes(**ckw)
        else:
            if st.refused is None:
                st.served += 1
        plddt_scores = plddt.get_scores(lddt_logits=lddt_logits, atom_mask=atom_mask, bin_centers=lddt_bin_centers, atom_asym_id=atom_asym_id)
        aggregate_score = (0.2 * ptm_scores.complex_ptm + 0.8 * ptm_scores.interface_ptm - 100 * clash_scores.has_inter_chain_clashes.float())
        _, asyms = rank_utils.get_chain_masks_and_asyms(asym_id=token_asym_id, mask=token_exists_mask)
        st.seconds += time.perf_counter() - t0
        return SampleRanking(asym_ids=asyms, aggregate_score=aggregate_score, ptm_scores=ptm_scores, clash_scores=clash_scores, plddt_scores=plddt_scores)

    rank_cc.__name__ = RANK_NAME
    rank_cc.chai1_opt_lever = "rankcc"
    return rank_cc


# --------------------------------------------------------------------------------------------------------------------------- tailasync
class _TailState:
    def __init__(self):
        from concurrent.futures import ThreadPoolExecutor                           # at install time only: importing concurrent.futures registers a threading exit hook, which the interpreter refuses once shutdown began
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chai1_tailasync")
        self.pending = []; self.lock = threading.Lock()
        self.cif = 0; self.plot = 0; self.join_s = 0.0; self.joins = 0; self.errors = 0

    def submit(self, fn, *a, **kw):
        fut = self.pool.submit(fn, *a, **kw)
        with self.lock:
            self.pending.append(fut)
        return fut

    def join(self):
        with self.lock:
            futs, self.pending = self.pending, []
        if not futs:
            return
        t0 = time.perf_counter(); first = None
        for f in futs:
            try:
                f.result()
            except BaseException as e:  # noqa: BLE001 — re-raised below, as the synchronous statement would have raised it
                self.errors += 1
                first = first or e
        self.join_s += time.perf_counter() - t0; self.joins += 1
        if first is not None:
            raise first


def _make_writers(ts: _TailState, orig_cif, orig_plot):
    def save_to_cif_async(coords, output_batch, write_path, asym_entity_names, bfactors=None):
        """chai_lab.data.io.cif_utils.save_to_cif on the fold's writer thread (tailasync); joined before the fold's StructureCandidates exists."""
        ts.cif += 1
        ts.submit(orig_cif, coords=coords, output_batch=output_batch, write_path=write_path, asym_entity_names=asym_entity_names, bfactors=bfactors)

    def plot_msa_async(input_tokens, msa_tokens, out_fname, *a, **kw):
        """chai_lab.utils.plot.plot_msa on the fold's writer thread (tailasync); returns the path as upstream does."""
        ts.plot += 1
        ts.submit(orig_plot, input_tokens, msa_tokens, out_fname, *a, **kw)
        return out_fname

    save_to_cif_async.__name__ = CIF_NAME; save_to_cif_async.chai1_opt_lever = "tailasync"
    plot_msa_async.__name__ = "plot_msa_async"; plot_msa_async.chai1_opt_lever = "tailasync"
    return save_to_cif_async, plot_msa_async


def _joined_candidates(C1, ts: _TailState):
    Base = C1.StructureCandidates

    class StructureCandidates(Base):                                             # the fold's return value: every pending write has landed before it exists
        __doc__ = Base.__doc__

        def __post_init__(self):
            ts.join()
            sup = getattr(super(), "__post_init__", None)
            if sup is not None:
                sup()

    StructureCandidates.__module__ = Base.__module__
    StructureCandidates.chai1_opt_lever = "tailasync"
    return StructureCandidates


# ----------------------------------------------------------------------------------------------------------------------------- install
def install(levers: Tuple[str, ...], chai1_mod=None) -> dict:
    """Rebind the fold body's module attributes for ``levers`` (idempotent per lever; unknown names -> LeverUnavailable). Returns report facts."""
    unknown = [l for l in levers if l not in LEVERS]
    if unknown:
        raise LeverUnavailable(f"unknown postproc levers {unknown}; known: {LEVERS}")
    if not levers:
        return {"postproc_levers": list(_STATE["installed"])}
    import importlib
    C1 = chai1_mod or importlib.import_module("chai_lab.chai1")
    facts = {}
    if "rankcc" in levers and "rankcc" not in _STATE["installed"]:
        st = _RankState()
        _STATE["orig"]["rank"] = getattr(C1, "rank", None)
        C1.rank = _make_rank(st)
        _STATE["rank"] = st
    if "tailasync" in levers and "tailasync" not in _STATE["installed"]:
        ts = _TailState()
        _STATE["orig"].update(save_to_cif=getattr(C1, "save_to_cif", None), plot_msa=getattr(C1, "plot_msa", None), StructureCandidates=getattr(C1, "StructureCandidates", None))
        C1.save_to_cif, C1.plot_msa = _make_writers(ts, _STATE["orig"]["save_to_cif"], _STATE["orig"]["plot_msa"])
        if _STATE["orig"]["StructureCandidates"] is not None:                    # the join point: the fold's return value (a module without it has no fold body to serve)
            C1.StructureCandidates = _joined_candidates(C1, ts)
        _STATE["tail"] = ts
        import atexit
        try:
            atexit.register(_drain)                                                  # the writer drained at exit (install time; the fold's own join point is StructureCandidates)
        except RuntimeError:                                                         # registered during shutdown: nothing to arm — the join point drains inline
            pass
    _STATE["installed"] = tuple(l for l in LEVERS if l in levers or l in _STATE["installed"])
    facts["postproc_levers"] = list(_STATE["installed"])
    return facts


def _drain():
    ts = _STATE["tail"]
    if ts is not None:
        try:
            ts.join()
        except BaseException:  # noqa: BLE001 — at interpreter exit; the fold that owned the write has already returned or raised
            pass


def applied() -> Tuple[str, ...]:
    return tuple(_STATE["installed"])


def census() -> dict:
    out = {}
    st = _STATE["rank"]
    if st is not None:
        out["rankcc"] = {"served": st.served, "fallback": sum(st.fallback_by.values()), "fallback_by": dict(sorted(st.fallback_by.items())),
                         "witness": st.witness, "seconds": round(st.seconds, 3)}
    ts = _STATE["tail"]
    if ts is not None:
        out["tailasync"] = {"cif": ts.cif, "plot": ts.plot, "join_s": round(ts.join_s, 3), "joins": ts.joins, "errors": ts.errors}
    return out


def evidence(name: str) -> dict:
    """The LEVER-line evidence fields of one postproc lever (empty when it is not installed)."""
    if name not in _STATE["installed"]:
        return {}
    return census().get(name, {})


def gate() -> Tuple[bool, str]:
    """Fail-closed on rankcc: no unexpected fallback and no failed witness. tailasync has nothing to refuse (a writer error re-raises in the fold)."""
    bad = []
    c = census()
    r = c.get("rankcc")
    if r is not None:
        unexpected = [k for k in r["fallback_by"] if k not in EXPECTED_FALLBACKS["rankcc"]]
        if unexpected:
            bad.append(f"rankcc:fallback_by={','.join(unexpected)}")
        if r["witness"] == "fail":
            bad.append("rankcc:witness_failed")
    return (not bad), ";".join(bad)


def verdict() -> Optional[str]:
    if not _STATE["installed"]:
        return None
    ok, why = gate()
    return None if ok else why


def tally_fields() -> list:
    c = census(); f = []
    if "rankcc" in c:
        r = c["rankcc"]
        f.append(f"rankcc_served={r['served']} rankcc_fallback={r['fallback']} rankcc_witness={r['witness']}" +
                 (f" rankcc_fallback_by={','.join(f'{k}:{n}' for k, n in r['fallback_by'].items())}" if r["fallback_by"] else ""))
    if "tailasync" in c:
        t = c["tailasync"]
        f.append(f"tailasync_cif={t['cif']} tailasync_plot={t['plot']} tailasync_join_s={t['join_s']}")
    return f


def reset_for_tests(chai1_mod=None) -> None:
    C1 = chai1_mod if chai1_mod is not None else sys.modules.get("chai_lab.chai1")
    if C1 is not None:
        for attr, fn in _STATE["orig"].items():
            setattr(C1, attr, fn)
    ts = _STATE["tail"]
    if ts is not None:
        ts.pool.shutdown(wait=True)
    _STATE.update(installed=(), orig={}, rank=None, tail=None)
