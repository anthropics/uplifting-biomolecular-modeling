"""protenix_ptx_summary_host (module; lever `summary_hostidx`). Numerics class: EXACT-BITWISE.

Stock (pinned Protenix 2.0.0) `protenix.model.sample_confidence.compute_full_data_and_summary` runs `_compute_full_data_and_summary` once per
diffusion sample (5x per item) and, inside, `calculate_chain_based_{gpde,ptm,plddt}` + `calculate_clash` walk every chain and chain PAIR with
host round-trips: `torch.unique(asym_id)` (sync), `aid.item()` per chain, boolean-mask indexing `x[..., mask, :]` (= nonzero -> sync) per chain /
pair / call, `if has_frame.sum() == 0` (sync), `per_bin_weight.to(device)` and `torch.zeros(...).to(device)` (pageable H2D = sync) per call,
`torch.where(clash)` + `.shape[0]` and `torch.sum(mask).item()` per chain pair in the clash check — a host-synchronisation count per item
that grows with chains^2.

The lever computes the chain BOOKKEEPING on the host from ONE device->host copy of (asym_id, has_frame) and ONE of (atom_to_token_idx,
atom_is_polymer) per item — chain / chain-pair token and atom index lists (ascending = the order boolean masking yields), frame positions,
chain_has_frame, chain_is_ligand, chain types, atom counts — uploads all index lists as ONE int64 blob and all pTM bin-weight rows as ONE fp32 blob,
and then issues, per sample, THE SAME VALUE-PRODUCING GPU OPS AS STOCK in the same order on the same operands: boolean-mask gathers become
integer-index gathers of the identical elements in the identical order (bitwise-equal tensors), every reduction (sum / mean / max / matmul / cdist)
is the stock call on a bitwise-equal input, control flow that stock derives from device scalars (has_frame.sum()==0, chain_is_ligand, chain types,
clash counts) is derived from the host copies of the same integers. The one restated decision is the AF3 clash flag: stock takes
`total = len(torch.where(dist < thr))` to the host and evaluates `total > 100 or total / min(Ni, Nj) > 0.5` in Python; here `total = (dist < thr).sum()`
stays on the device and the flag is `(total > 100) | (2 * total > min(Ni, Nj))` — an exact integer identity of the same predicate (for integers
t, m < 2^52: t/m > 0.5 in float64 <=> 2t > m), so the bool is identical. Bin centres / weights are computed on the CPU by the stock functions
(get_bin_centers, calculate_normalization) exactly as stock does and only then copied, so the fp32 operands are bitwise those stock uploads.

Scope: the inference call (interested_atom_mask is None, mol_id / elements_one_hot None — `Protenix.main_inference_loop` mode="inference"). Any
other call shape (training / eval extras: pb_ranking_score, vdw clash) is delegated to the stock function unchanged and counted (`delegated`).
Import name: protenix_ptx_summary_host (kit rule: new top-level modules are protenix_-prefixed).

Switch: PTX_SUMMARY_HOST=1 (env.sh, ARM=E and ARM=T; big inherits fast). install() is idempotent; report() -> counters; atexit SUMMARY line.
"""
from __future__ import annotations

import atexit
import json
import os
import sys
from typing import Any, Optional

import numpy as np
import torch

ENV = "PTX_SUMMARY_HOST"
TARGET = "protenix.model.sample_confidence"
_ST = {"installed": False, "orig": None, "calls": 0, "samples": 0, "delegated": 0, "chains_max": 0, "pairs": 0, "h2d": 0, "d2h": 0, "errors": 0}
_DEV_CONST: dict = {}          # (device, min_bin, max_bin, no_bins) -> bin_centers on device (stock get_bin_centers().to(device), cached: same bits)


def enabled(environ=None) -> bool:
    v = (environ if environ is not None else os.environ).get(ENV, "")
    if v in ("", "0"):
        return False
    if v != "1":
        raise ValueError(f"{ENV}={v!r}: 1 (host-indexed summary confidence) or unset/0")
    return True


# ------------------------------------------------------------------------------------------------------------------ host index
class HostIndex:
    """Per-item chain bookkeeping from one host copy of the token rows and one of the atom rows (2 D2H) + 1 H2D index blob."""

    def __init__(self, sc, token_asym_id, token_has_frame, atom_to_token_idx, atom_is_polymer, bins_pae):
        dev = token_asym_id.device
        tok = torch.stack([token_asym_id.reshape(-1).long(), token_has_frame.reshape(-1).long()]).cpu().numpy(); _ST["d2h"] += 1
        atm = torch.stack([atom_to_token_idx.reshape(-1).long(), atom_is_polymer.reshape(-1).long()]).cpu().numpy(); _ST["d2h"] += 1
        asym, hasf = tok[0], tok[1] != 0
        a2t, is_poly = atm[0], atm[1]
        self.n_token, self.n_atom = int(asym.shape[0]), int(a2t.shape[0])
        uniq = np.unique(asym)                                   # sorted, = torch.unique
        if len(uniq) != int(asym.max()) + 1:                     # stock: remap gaps to 0..K-1 (order-preserving)
            asym = np.searchsorted(uniq, asym).astype(np.int64)      # == {old: new for new, old in enumerate(uniq)}[x]
        K = self.K = int(len(uniq))
        atom_asym = asym[a2t]
        atom_is_lig = 1 - is_poly
        tok_is_lig = np.zeros(self.n_token, dtype=np.int64); np.add.at(tok_is_lig, a2t, atom_is_lig); tok_is_lig = tok_is_lig > 0
        # atom_type = 1*is_ligand + 2*is_protein(=is_polymer) + 3*0 + 4*0  (calculate_clash passes dna/rna dummies)
        atom_type = (1 * atom_is_lig + 2 * is_poly).astype(np.int64)
        self.tok_idx, self.frame_pos, self.n_frame, self.chain_has_frame, self.chain_is_ligand = [], [], [], [], []
        self.atom_idx, self.n_atom_c, self.chain_types = [], [], []
        pieces, off = [], 0

        def push(arr):
            nonlocal off
            arr = np.ascontiguousarray(arr, dtype=np.int64); pieces.append(arr); s = (off, off + arr.shape[0]); off += arr.shape[0]; return s
        self._tok_sl, self._frame_sl, self._atom_sl = [], [], []
        for c in range(K):
            m = asym == c
            ti = np.nonzero(m)[0]
            self._tok_sl.append(push(ti)); self.tok_idx.append(ti)
            hf_c = hasf[ti]
            fp = np.nonzero(hf_c)[0]
            self._frame_sl.append(push(fp)); self.n_frame.append(int(fp.shape[0]))
            self.chain_has_frame.append(bool(hf_c.any()))
            self.chain_is_ligand.append(bool(tok_is_lig[ti].sum() >= (int(m.sum()) // 2)))
            am = atom_asym == c
            ai = np.nonzero(am)[0]
            self._atom_sl.append(push(ai)); self.n_atom_c.append(int(ai.shape[0]))
            types = np.unique(atom_type[am])
            assert len(types) == 1                               # stock: assert len(atom_type_i.unique()) == 1
            t = int(types[0]) if len(types) else 0
            if t == 0:
                sc_logger_warning("Unknown asym_id type: not in ligand / protein / dna / rna")
            self.chain_types.append(ID2TYPE()[t])
        # global frame positions (top-level ptm / iptm: `[..., has_frame]`)
        self._gframe_sl = push(np.nonzero(hasf)[0]); self.n_frame_all = int(hasf.sum())
        # chain pairs a<b: token union (ascending), frame positions within the union, atom union
        self._pair_tok_sl, self._pair_frame_sl, self._pair_atom_sl, self.pair_nframe = {}, {}, {}, {}
        for a in range(K):
            for b in range(a + 1, K):
                pm = (asym == a) | (asym == b)
                pi = np.nonzero(pm)[0]
                self._pair_tok_sl[(a, b)] = push(pi)
                pf = np.nonzero(hasf[pi])[0]
                self._pair_frame_sl[(a, b)] = push(pf); self.pair_nframe[(a, b)] = int(pf.shape[0])
                pam = (atom_asym == a) | (atom_asym == b)
                self._pair_atom_sl[(a, b)] = push(np.nonzero(pam)[0])
        blob = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int64)
        self.blob = torch.from_numpy(blob).to(dev); _ST["h2d"] += 1
        # pTM per-bin weights for every distinct N_d used (stock: computed on CPU per call, then .to(device)): one fp32 blob
        nds = sorted({self.n_token} | {len(t) for t in self.tok_idx} | {self._pair_tok_sl[k][1] - self._pair_tok_sl[k][0] for k in self._pair_tok_sl})
        rows = []
        for nd in nds:
            ptm_norm = sc.calculate_normalization(nd)
            bin_center = sc.get_bin_centers(bins_pae["min_bin"], bins_pae["max_bin"], bins_pae["no_bins"])
            rows.append(1 / (1 + (bin_center / ptm_norm) ** 2))          # the stock CPU expression, bit for bit
        self.wblob = torch.stack(rows).to(dev); _ST["h2d"] += 1
        self._wrow = {nd: i for i, nd in enumerate(nds)}
        _ST["chains_max"] = max(_ST["chains_max"], K); _ST["pairs"] += len(self._pair_tok_sl)

    def sl(self, s):
        return self.blob[s[0]:s[1]]

    def w(self, nd):
        return self.wblob[self._wrow[nd]]


def ID2TYPE():
    from protenix.metrics.clash import ID2TYPE as _T
    return _T


def sc_logger_warning(msg):
    try:
        from protenix.metrics.clash import logger as _lg
        _lg.warning(msg)
    except Exception:
        pass


def _bin_centers_dev(sc, dev, p):
    key = (str(dev), p["min_bin"], p["max_bin"], p["no_bins"])
    t = _DEV_CONST.get(key)
    if t is None:
        t = sc.get_bin_centers(p["min_bin"], p["max_bin"], p["no_bins"]).to(dev); _ST["h2d"] += 1
        _DEV_CONST[key] = t
    return t


def _logits_to_score(sc, logits, p, return_prob=False):
    """stock logits_to_score with the bin-centre vector cached on the device (same CPU-computed bits)."""
    prob = sc.logits_to_prob(logits, dim=-1)
    score = prob @ _bin_centers_dev(sc, logits.device, p)
    return (score, prob) if return_prob else score


# ------------------------------------------------------------------------------------------------------------------ value path (stock ops, host control flow)
def _ptm(hx: HostIndex, pae_prob, tok_sl=None, frame_sl=None, n_frame=None, n_d=None):
    """calculate_ptm: token_mask None -> (global frames); else the chain's token slice."""
    if tok_sl is not None:
        idx = hx.sl(tok_sl)
        pae_prob = pae_prob[..., idx, :, :][..., :, idx, :]
    if n_frame == 0:
        return torch.zeros(size=pae_prob.shape[:-3], device=pae_prob.device)
    per_bin_weight = hx.w(n_d)
    token_token_ptm = (pae_prob * per_bin_weight).sum(dim=-1)
    return token_token_ptm.mean(dim=-1)[..., hx.sl(frame_sl)].max(dim=-1).values


def _iptm(hx: HostIndex, pae_prob, asym_id, tok_sl=None, frame_sl=None, n_frame=None, n_d=None, eps: float = 1e-8):
    """calculate_iptm: token_mask None -> global; else the pair's token slice (asym_id gathered with the same indices)."""
    if tok_sl is not None:
        idx = hx.sl(tok_sl)
        pae_prob = pae_prob[..., idx, :, :][..., :, idx, :]
        asym_id = asym_id[idx]
    if n_frame == 0:
        return torch.zeros(size=pae_prob.shape[:-3], device=pae_prob.device)
    per_bin_weight = hx.w(n_d)
    token_token_ptm = (pae_prob * per_bin_weight).sum(dim=-1)
    is_diff_chain = asym_id[None, :] != asym_id[:, None]
    iptm = (token_token_ptm * is_diff_chain).sum(dim=-1) / (eps + is_diff_chain.sum(dim=-1))
    return iptm[..., hx.sl(frame_sl)].max(dim=-1).values


def _chain_based_gpde(hx: HostIndex, token_pair_pde, contact_probs, eps: float = 1e-8):
    batch_shape = token_pair_pde.shape[:-2]; device = token_pair_pde.device; K = hx.K

    def _cal_gpde(i1, i2):
        masked_contact_probs = contact_probs[..., i1, :][..., i2]
        masked_pde = token_pair_pde[..., i1, :][..., i2]
        return (masked_pde * masked_contact_probs).sum(dim=(-1, -2)) / (masked_contact_probs.sum(dim=(-1, -2)) + eps)
    chain_gpde = torch.zeros(size=batch_shape + (K,), device=device)
    for aid in range(K):
        i = hx.sl(hx._tok_sl[aid]); chain_gpde[..., aid] = _cal_gpde(i, i)
    chain_pair_gpde = torch.zeros(size=batch_shape + (K, K), device=device)
    for a1 in range(K):
        for a2 in range(K):
            if a1 == a2:
                continue
            if a2 < a1:
                chain_pair_gpde[..., a1, a2] = chain_pair_gpde[..., a2, a1]; continue
            chain_pair_gpde[..., a1, a2] = _cal_gpde(hx.sl(hx._tok_sl[a1]), hx.sl(hx._tok_sl[a2]))
    return {"chain_gpde": chain_gpde, "chain_pair_gpde": chain_pair_gpde}


def _chain_based_ptm(hx: HostIndex, pae_prob, asym_id):
    batch_shape = pae_prob.shape[:-3]; dev = pae_prob.device; K = hx.K
    chain_pair_iptm = torch.zeros(size=batch_shape + (K, K), device=dev)          # stock: zeros(...).to(device) — same fp32 zeros, no H2D
    for a1 in range(K):
        for a2 in range(K):
            if a1 == a2:
                continue
            if a1 > a2:
                chain_pair_iptm[:, a1, a2] = chain_pair_iptm[:, a2, a1]; continue
            ts = hx._pair_tok_sl[(a1, a2)]
            chain_pair_iptm[:, a1, a2] = _iptm(hx, pae_prob, asym_id, tok_sl=ts, frame_sl=hx._pair_frame_sl[(a1, a2)],
                                               n_frame=hx.pair_nframe[(a1, a2)], n_d=ts[1] - ts[0])
    chain_ptm = torch.zeros(size=batch_shape + (K,), device=dev)
    for aid in range(K):
        ts = hx._tok_sl[aid]
        chain_ptm[:, aid] = _ptm(hx, pae_prob, tok_sl=ts, frame_sl=hx._frame_sl[aid], n_frame=hx.n_frame[aid], n_d=ts[1] - ts[0])
    chain_iptm = torch.zeros(size=batch_shape + (K,), device=dev)
    for aid in range(K):
        pairs = [(i, j) for i in range(K) for j in range(K) if (i == aid or j == aid) and (i != j) and hx.chain_has_frame[i]]
        vals = [chain_pair_iptm[:, i, j] for (i, j) in pairs]
        if len(vals) > 0:
            chain_iptm[:, aid] = torch.stack(vals, dim=-1).mean(dim=-1)
    chain_pair_iptm_global = torch.zeros(size=batch_shape + (K, K), device=dev)
    for a1 in range(K):
        for a2 in range(K):
            if a1 == a2:
                continue
            if hx.chain_is_ligand[a1]:
                chain_pair_iptm_global[:, a1, a2] = chain_iptm[:, a1]
            elif hx.chain_is_ligand[a2]:
                chain_pair_iptm_global[:, a1, a2] = chain_iptm[:, a2]
            else:
                chain_pair_iptm_global[:, a1, a2] = (chain_iptm[:, a1] + chain_iptm[:, a2]) * 0.5
    return {"chain_ptm": chain_ptm, "chain_iptm": chain_iptm, "chain_pair_iptm": chain_pair_iptm, "chain_pair_iptm_global": chain_pair_iptm_global}


def _chain_based_plddt(hx: HostIndex, atom_plddt):
    batch_shape = atom_plddt.shape[:-1]; dev = atom_plddt.device; K = hx.K
    chain_plddt = torch.zeros(size=batch_shape + (K,), device=dev)
    for aid in range(K):
        chain_plddt[:, aid] = atom_plddt[:, hx.sl(hx._atom_sl[aid])].mean(-1)
    chain_pair_plddt = torch.zeros(size=batch_shape + (K, K), device=dev)
    for a1 in range(K):
        for a2 in range(K):
            if a1 == a2:
                continue
            key = (a1, a2) if a1 < a2 else (a2, a1)                # stock recomputes (a2,a1) from the same mask: identical op on identical input
            chain_pair_plddt[:, a1, a2] = atom_plddt[:, hx.sl(hx._pair_atom_sl[key])].mean(-1)
    return {"chain_plddt": chain_plddt, "chain_pair_plddt": chain_pair_plddt}


def _clash(hx: HostIndex, pred_coordinate, threshold: float):
    """calculate_clash -> Clash(compute_vdw_clash=False) af3 path: [N_sample] bool (N_sample == 1 per call, as stock)."""
    N_sample = pred_coordinate.shape[0]; dev = pred_coordinate.device; K = hx.K
    has_af3_clash_flag = torch.zeros(N_sample, K, K, device=dev, dtype=torch.bool)
    for s in range(N_sample):
        for i in range(K):
            if hx.chain_types[i] == "UNK":
                continue
            N_i = hx.n_atom_c[i]
            for j in range(i + 1, K):
                if hx.chain_types[j] == "UNK":
                    continue
                if hx.chain_types[i] == "lig" or hx.chain_types[j] == "lig":       # AF3 clash only considers polymer chains
                    continue
                N_j = hx.n_atom_c[j]
                chain_1_coords = pred_coordinate[s, :, :][hx.sl(hx._atom_sl[i]), :]
                chain_2_coords = pred_coordinate[s, :, :][hx.sl(hx._atom_sl[j]), :]
                pred_dist = torch.cdist(chain_1_coords, chain_2_coords)
                clash_per_atom_pair = pred_dist < threshold
                total_clash = clash_per_atom_pair.sum()                            # == af3_clash_pairs.shape[0], kept on the device
                flag = (total_clash > 100) | (2 * total_clash > min(N_i, N_j))       # == total > 100 or total / min(Ni,Nj) > 0.5 (exact)
                has_af3_clash_flag[s, i, j] = flag
                has_af3_clash_flag[s, j, i] = has_af3_clash_flag[s, i, j]
    return has_af3_clash_flag.reshape(N_sample, -1).max(dim=-1)[0]


def _one_sample(sc, configs, hx: HostIndex, pae_logits, plddt_logits, pde_logits, contact_probs, token_asym_id, token_has_frame,
                atom_coordinate, atom_to_token_idx, atom_is_polymer, N_recycle, return_full_data):
    p_plddt, p_pde, p_pae = (sc.get_bin_params(configs.loss.plddt), sc.get_bin_params(configs.loss.pde), sc.get_bin_params(configs.loss.pae))
    full_data = {}
    full_data["atom_plddt"] = _logits_to_score(sc, plddt_logits, p_plddt)
    pde_logits = pde_logits.to(plddt_logits.device)
    full_data["token_pair_pde"] = _logits_to_score(sc, pde_logits, p_pde)
    del pde_logits
    full_data["contact_probs"] = contact_probs.clone()
    pae_logits = pae_logits.to(plddt_logits.device)
    full_data["token_pair_pae"], pae_prob = _logits_to_score(sc, pae_logits, p_pae, return_prob=True)
    del pae_logits
    summary = {}
    summary["plddt"] = full_data["atom_plddt"].mean(dim=-1) * 100
    summary["gpde"] = (full_data["token_pair_pde"] * full_data["contact_probs"]).sum(dim=[-1, -2]) / full_data["contact_probs"].sum(dim=[-1, -2])
    summary["ptm"] = _ptm(hx, pae_prob, tok_sl=None, frame_sl=hx._gframe_sl, n_frame=hx.n_frame_all, n_d=hx.n_token)
    summary["iptm"] = _iptm(hx, pae_prob, token_asym_id, tok_sl=None, frame_sl=hx._gframe_sl, n_frame=hx.n_frame_all, n_d=hx.n_token)
    summary.update(_chain_based_gpde(hx, full_data["token_pair_pde"], full_data["contact_probs"]))
    summary.update(_chain_based_ptm(hx, pae_prob, token_asym_id))
    summary.update(_chain_based_plddt(hx, full_data["atom_plddt"]))
    del pae_prob
    summary["has_clash"] = _clash(hx, atom_coordinate, configs.metrics.clash.af3_clash_threshold)
    summary["num_recycles"] = torch.tensor(N_recycle, device=atom_coordinate.device)
    summary["disorder"] = torch.zeros_like(summary["ptm"])
    summary["ranking_score"] = 0.8 * summary["iptm"] + 0.2 * summary["ptm"] + 0.5 * summary["disorder"] - 100 * summary["has_clash"]
    summary = sc.break_down_to_per_sample_dict(summary, shared_keys=["num_recycles"])
    if return_full_data:
        full_data["token_has_frame"] = token_has_frame.clone()
        full_data["token_asym_id"] = token_asym_id.clone()
        full_data["atom_to_token_idx"] = atom_to_token_idx.clone()
        full_data["atom_is_polymer"] = atom_is_polymer.clone()
        full_data["atom_coordinate"] = atom_coordinate.clone()
        full_data = sc.break_down_to_per_sample_dict(full_data, shared_keys=["contact_probs", "token_has_frame", "token_asym_id", "atom_to_token_idx", "atom_is_polymer"])
        return summary, full_data
    return summary, [{}]


@torch.no_grad()
def compute_full_data_and_summary(configs, pae_logits, plddt_logits, pde_logits, contact_probs, token_asym_id, token_has_frame, atom_coordinate,
                                  atom_to_token_idx, atom_is_polymer, N_recycle, return_full_data: bool = False,
                                  interested_atom_mask: Optional[torch.Tensor] = None, mol_id: Optional[torch.Tensor] = None,
                                  elements_one_hot: Optional[torch.Tensor] = None):
    """Drop-in for sample_confidence.compute_full_data_and_summary (inference call shape); other shapes -> the stock function."""
    sc = sys.modules[TARGET]
    if interested_atom_mask is not None or mol_id is not None or elements_one_hot is not None:
        _ST["delegated"] += 1
        return _ST["orig"](configs=configs, pae_logits=pae_logits, plddt_logits=plddt_logits, pde_logits=pde_logits, contact_probs=contact_probs,
                           token_asym_id=token_asym_id, token_has_frame=token_has_frame, atom_coordinate=atom_coordinate,
                           atom_to_token_idx=atom_to_token_idx, atom_is_polymer=atom_is_polymer, N_recycle=N_recycle,
                           return_full_data=return_full_data, interested_atom_mask=interested_atom_mask, mol_id=mol_id, elements_one_hot=elements_one_hot)
    _ST["calls"] += 1
    N_sample = pae_logits.size(0)
    if contact_probs.dim() == 2:
        contact_probs = contact_probs.unsqueeze(dim=0).expand(N_sample, -1, -1)
    else:
        assert contact_probs.dim() == 3
    assert contact_probs.size(0) == plddt_logits.size(0) == pde_logits.size(0) == N_sample
    hx = HostIndex(sc, token_asym_id, token_has_frame, atom_to_token_idx, atom_is_polymer, sc.get_bin_params(configs.loss.pae))
    summary_confidence, full_data = [], []
    for i in range(N_sample):
        s_i, f_i = _one_sample(sc, configs, hx, pae_logits[i:i + 1], plddt_logits[i:i + 1], pde_logits[i:i + 1], contact_probs[i], token_asym_id,
                               token_has_frame, atom_coordinate[i:i + 1], atom_to_token_idx, atom_is_polymer, N_recycle, return_full_data)
        summary_confidence.extend(s_i); full_data.extend(f_i); _ST["samples"] += 1
    return summary_confidence, full_data


compute_full_data_and_summary._ptx_summary_host = True


# ------------------------------------------------------------------------------------------------------------------ install
EXPECTED_PARAMS = ("configs", "pae_logits", "plddt_logits", "pde_logits", "contact_probs", "token_asym_id", "token_has_frame", "atom_coordinate",
                   "atom_to_token_idx", "atom_is_polymer", "N_recycle", "return_full_data", "interested_atom_mask", "mol_id", "elements_one_hot")
MARK = "SUMHOST"                     # marker family: SUMHOST:on(...) | raises by name (the mode refuses)


def install() -> str:
    """Replace sample_confidence.compute_full_data_and_summary. Returns the marker text `on(...)`; raises RuntimeError naming the lever when the
    pinned stock surface is not the one this module restates (signature drift) — never a silent pass-through."""
    import importlib, inspect
    if _ST["installed"]:
        return "on(already)"
    sc = importlib.import_module(TARGET)
    cur = sc.compute_full_data_and_summary
    if getattr(cur, "_ptx_summary_host", False):
        _ST["installed"] = True; return "on(already)"
    got = tuple(inspect.signature(inspect.unwrap(cur)).parameters)
    if got != EXPECTED_PARAMS:
        raise RuntimeError(f"summary_hostidx: {TARGET}.compute_full_data_and_summary signature {got} != pinned {EXPECTED_PARAMS}")
    for name in ("calculate_normalization", "get_bin_centers", "get_bin_params", "logits_to_prob", "break_down_to_per_sample_dict"):
        if not hasattr(sc, name):
            raise RuntimeError(f"summary_hostidx: {TARGET}.{name} missing (pinned stock surface changed)")
    _ST["orig"] = cur
    sc.compute_full_data_and_summary = compute_full_data_and_summary
    _ST["installed"] = True
    if not _ST.get("atexit"):
        atexit.register(_atexit); _ST["atexit"] = True
    return "on(inference-call;d2h=2,h2d=2/item)"


def uninstall() -> None:
    if _ST["installed"] and _ST["orig"] is not None:
        sys.modules[TARGET].compute_full_data_and_summary = _ST["orig"]; _ST["installed"] = False


def report() -> dict:
    return {"lever": "summary_hostidx", **{k: v for k, v in _ST.items() if k != "orig"}}


def _atexit():
    r = report()
    try:
        print(f"[ptx_summary_host] SUMMARY calls={r['calls']} samples={r['samples']} delegated={r['delegated']} chains_max={r['chains_max']} "
              f"pairs={r['pairs']} d2h={r['d2h']} h2d={r['h2d']}", flush=True)
    except Exception:
        pass
    p = os.environ.get("PTX_LEVER_REPORT")
    if p:
        try:
            with open(p, "a") as fh:
                fh.write(json.dumps({"ptx_summary_host": r, "pid": os.getpid()}) + "\n")
        except Exception:
            pass


def apply():          # alias entry point: same as install()
    return install()
