"""rosettafold3_opt.tp_conf — RF3's confidence head under ``n_gpu > 1`` (rows, no replicated N^2 stage): P in {2, 3} multi-process gloo checks
against the DENSE head and RF3's own consumers.

The reference is RF3's real code from the kit's pinned stock archive (``stock/foundry-<commit>.tar.gz``): ``ConfidenceHead``
(``rf3/model/layers/af3_auxiliary_heads.py``) with a stand-in ``PairformerBlock`` (the real block's binding is the adapter's, checked there),
``compute_ptm`` / ``ComputeIPTM`` (``rf3/metrics/predicted_error.py``), ``unbin_logits`` / the mask builders (``rf3/metrics/metric_utils.py``)
and ``compile_af3_style_confidence_outputs`` (``rf3/utils/predicted_error.py``), exec'd as a throw-away ``rf3`` package with the third-party
modules they import stubbed when absent. What is asserted (fp32 CPU, per case): the five TM scalars (ptm, iptm, iptm_protein_protein /
protein_ligand / ligand_ligand) within 1e-6 of RF3's on the dense logits; the host-assembled expected PAE / PDE matrices within 1e-4 (values in
[0, 32]) — bitwise REPORTED; RF3's ``compile_af3_style_confidence_outputs`` on the expected matrices (``compile_on_expected``) equals RF3's on
the dense logits (summary numbers within 1e-3 — they are rounded to 2-4 places; the PAE matrix rows; identical keys); pLDDT /
exp_resolved logits within 1e-5, for the full call and for the ``X_pred_L=None`` early-stop probe (vs the stock probe); pair-derived keys
``None`` on ranks > 0 (and everywhere for the probe); NO rank ever allocates a tensor of ``I * I * 64`` elements or more (dispatch-mode numel census: no
``[I, I, 64]`` logits, no ``[I, I, c]`` pair, no ``[I, I]`` T on a device); the schedule census words; the P == 1 / no-group refusal and the
unbound-branch refusal by name; the config-of-record global LayerNorm (moments all-reduced); even and uneven I; with and without a ligand chain.

Run: ``python -m pytest opt/rosettafold3_opt/tests/test_tp_conf_cpu.py -q -rfE`` from ``rosettafold3/`` (torch + opt_core >= 0.4.3 rc2
importable, else skipped BY NAME) or ``python opt/rosettafold3_opt/tests/test_tp_conf_cpu.py`` (RESULT lines + SUMMARY).
"""
from __future__ import annotations

import glob
import hashlib
import io
import json
import os
import sys
import tarfile
import tempfile
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(os.path.dirname(HERE))                                  # rosettafold3/opt (the package's parent)
KIT = os.path.dirname(OPT)                                                    # rosettafold3/
if OPT not in sys.path:
    sys.path.insert(0, OPT)

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:  # noqa: BLE001
    HAVE_TORCH = False


def _have_core() -> bool:
    try:
        from rosettafold3_opt import _core
        D = _core.load("mem.rowpair.dist")
        return hasattr(D, "gather_rows_to_rank0_host")
    except Exception:  # noqa: BLE001
        return False


C_Z, C_S, C_SIN, NLAYERS = 32, 16, 449, 2
CFG = {"plddt": {"max_value": 1.0, "n_bins": 50}, "pae": {"max_value": 32, "n_bins": 64}, "pde": {"max_value": 32, "n_bins": 64},
       "exp_resolved": {"max_value": 1, "n_bins": 2}}                          # configs/trainer/loss/losses/confidence_loss.yaml
HEAD_KW = dict(n_pairformer_layers=NLAYERS, pairformer={}, n_bins_pae=64, n_bins_pde=64, n_bins_plddt=50, n_bins_exp_resolved=2,
               use_Cb_distances=False, use_af3_style_binning_and_final_layer_norms=True, symmetrize_Cb_logits=True)
STOCK_FILES = {                                                               # archive member suffix -> module name in the throw-away package
    "models/rf3/src/rf3/model/layers/af3_auxiliary_heads.py": "rf3.model.layers.af3_auxiliary_heads",
    "models/rf3/src/rf3/metrics/metric_utils.py": "rf3.metrics.metric_utils",
    "models/rf3/src/rf3/metrics/predicted_error.py": "rf3.metrics.predicted_error",
    "models/rf3/src/rf3/utils/predicted_error.py": "rf3.utils.predicted_error",
}
STANDIN_RF3_STRUCTURE = '''
import torch
import torch.nn as nn


def linearNoBias(dim_in, dim_out):
    return nn.Linear(dim_in, dim_out, bias=False)


class PairformerBlock(nn.Module):
    """Stand-in for the CPU check of the confidence-head binding: a row-local pair transition and a single-track update from the pair ROW
    means (exercises the driver seam: rows in / rows out, S carried and updated from sharded rows). The real block is bound by the adapter."""

    def __init__(self, c_s, c_z, **kw):
        super().__init__()
        self.ln = nn.LayerNorm(c_z)
        self.lin = nn.Linear(c_z, c_z)
        self.s_lin = nn.Linear(c_z, c_s)

    def z_update(self, Z):
        return self.lin(torch.relu(self.ln(Z)))

    def forward(self, S_I, Z_II):
        Z_II = Z_II + self.z_update(Z_II)
        if S_I is not None:
            S_I = S_I + self.s_lin(Z_II.mean(dim=-2))                          # S takes the pair operand's batch rank, as the real block's broadcasting does
        return S_I, Z_II
'''
STUB_MODULES = {                                                              # third-party names the stock files import; stubbed only when absent
    "foundry.metrics.metric": "class Metric:\n    pass\n",
    "rf3.chemical": "NHEAVY = 23\n",
}


# ============================================================================================================ the throw-away rf3 package
def _stock_archive() -> str:
    hits = sorted(glob.glob(os.path.join(KIT, "stock", "foundry-*.tar.gz")))
    if not hits:
        raise FileNotFoundError(f"no stock/foundry-*.tar.gz under {KIT}")
    return hits[0]


def _optional_stub(name: str):
    """Install a minimal stand-in for a third-party module the stock files import at module level, when it is not installed."""
    try:
        __import__(name)
        return
    except Exception:  # noqa: BLE001
        pass
    import typing
    mod = types.ModuleType(name)
    if name == "jaxtyping":
        class _Ann:
            def __class_getitem__(cls, item):
                return typing.Any
        mod.Float = mod.Bool = mod.Int = _Ann
    elif name == "beartype.typing":
        mod.Any = typing.Any
        parent = types.ModuleType("beartype")
        parent.typing = mod
        sys.modules.setdefault("beartype", parent)
    elif name == "biotite.structure":
        mod.AtomArray = type("AtomArray", (), {})
        mod.AtomArrayStack = type("AtomArrayStack", (), {})
        parent = types.ModuleType("biotite")
        parent.structure = mod
        sys.modules.setdefault("biotite", parent)
    elif name == "omegaconf":
        mod.DictConfig = dict
    elif name in ("einops", "tree", "pandas"):
        pass
    sys.modules[name] = mod


def load_rf3():
    """The throw-away ``rf3`` package: RF3's real files from the stock archive + the stand-in ``rf3.model.RF3_structure`` + stubs. Returns
    ``(AH, MU, ME, PE)`` = the auxiliary-heads, metric_utils, metrics.predicted_error, utils.predicted_error modules."""
    for name in ("jaxtyping", "beartype.typing", "biotite.structure", "omegaconf", "einops", "tree", "pandas"):
        _optional_stub(name)
    tgz = _stock_archive()
    h = hashlib.sha256(open(tgz, "rb").read())
    for modname in sorted({"rf3.model.RF3_structure": STANDIN_RF3_STRUCTURE, **STUB_MODULES}):   # the stand-in sources are part of the tree's bytes
        h.update(modname.encode() + b"\0" + {"rf3.model.RF3_structure": STANDIN_RF3_STRUCTURE, **STUB_MODULES}[modname].encode() + b"\0")
    root = os.path.join(tempfile.gettempdir(), f"rf3_tpconf_ref_{h.hexdigest()[:12]}")
    done = os.path.join(root, ".complete")
    if not os.path.exists(done):
        tmp = root + f".{os.getpid()}"
        with tarfile.open(tgz, "r:gz") as tf:
            members = {m.name: m for m in tf.getmembers()}
            for suffix, modname in STOCK_FILES.items():
                hit = [n for n in members if n.endswith(suffix)]
                assert len(hit) == 1, (suffix, hit)
                dst = os.path.join(tmp, *modname.split(".")) + ".py"
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst, "wb") as f:
                    f.write(tf.extractfile(members[hit[0]]).read())
        extra = {"rf3.model.RF3_structure": STANDIN_RF3_STRUCTURE, **STUB_MODULES}
        for modname, src in extra.items():
            dst = os.path.join(tmp, *modname.split(".")) + ".py"
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "w") as f:
                f.write(src)
        for d, _sub, _files in os.walk(tmp):
            if d != tmp and not os.path.exists(os.path.join(d, "__init__.py")):
                open(os.path.join(d, "__init__.py"), "w").close()
        open(os.path.join(tmp, ".complete"), "w").close()
        try:
            os.rename(tmp, root)
        except OSError:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
    for m in [m for m in sys.modules if m in ("rf3", "foundry") or m.startswith(("rf3.", "foundry."))]:
        if getattr(sys.modules[m], "__file__", "") and not str(getattr(sys.modules[m], "__file__", "")).startswith(root):
            sys.modules.pop(m, None)
    if root not in sys.path:
        sys.path.insert(0, root)
    import importlib
    importlib.invalidate_caches()
    AH = importlib.import_module("rf3.model.layers.af3_auxiliary_heads")
    MU = importlib.import_module("rf3.metrics.metric_utils")
    ME = importlib.import_module("rf3.metrics.predicted_error")
    PE = importlib.import_module("rf3.utils.predicted_error")
    return AH, MU, ME, PE


# ================================================================================================================= synthetic case
def synthetic(I: int, chains, ligand_chain, seed: int = 0, batched: bool = False, **head_kw):
    """Seeded inputs + a seeded real ConfidenceHead (stand-in blocks); identical on every rank. ``batched=False``: the shapes RF3 calls the head
    with at inference (RF3.py:493-507) — single reps ``[I, c]``, trunk pair ``[I, I, c_z]`` (a rank's shard ``[R, I, c_z]``), ``X_pred_L =
    X_L[i].unsqueeze(0)`` ``[1, L, 3]``; ``batched=True``: the ``[1, ...]`` forms."""
    import numpy as np
    import torch
    AH, MU, ME, PE = load_rf3()
    g = torch.Generator().manual_seed(seed)
    head = AH.ConfidenceHead(c_s=C_S, c_z=C_Z, **{**HEAD_KW, **head_kw})
    with torch.no_grad():
        for p_ in head.parameters():
            p_.copy_(torch.randn(p_.shape, generator=g) * (0.3 / max(1.0, float(p_.shape[-1])) ** 0.5))
        for ln in (head.layernorm_pae, head.layernorm_pde, head.layernorm_plddt, head.layernorm_exp_resolved):
            ln.weight.copy_(1.0 + 0.1 * torch.randn(ln.weight.shape, generator=g))
            ln.bias.copy_(0.1 * torch.randn(ln.bias.shape, generator=g))
    head.eval()
    sizes = list(chains)
    assert sum(sizes) == I
    asym = torch.cat([torch.full((n,), k, dtype=torch.long) for k, n in enumerate(sizes)])
    is_ligand = (asym == ligand_chain) if ligand_chain is not None else torch.zeros(I, dtype=torch.bool)
    names = [chr(ord("A") + k) for k in range(len(sizes))]
    chain_iid = np.array([names[int(a)] for a in asym])
    n_atoms = 2 * I
    X = torch.cumsum(torch.randn(1, n_atoms, 3, generator=g) * 2.2, dim=1)          # a random walk: pair distances span the 3.25..50.75 bins
    rep_atoms = torch.arange(I) * 2
    seq = torch.randint(0, 20, (I,), generator=g)
    is_real_atom = torch.rand(I, 23, generator=g) < 0.5                            # the engine's token x atom mask [I, 23] (confidence_feats)
    is_real_atom[:, 0] = True
    atom_array = types.SimpleNamespace(chain_id=np.array([names[int(asym[t])] for t in range(I) for _ in range(int(is_real_atom[t].sum()))]))
    d = {
        "head": head, "S_inputs": torch.randn(1, I, C_SIN, generator=g), "S_trunk": torch.randn(1, I, C_S, generator=g),
        "Z": torch.randn(1, I, I, C_Z, generator=g) * 1.5 + 0.2, "X": X, "rep_atoms": rep_atoms, "seq": seq, "asym": asym, "is_ligand": is_ligand,
        "batched": bool(batched),
        "chain_iid": chain_iid, "is_real_atom": is_real_atom, "atom_array": atom_array,
        "cfg": types.SimpleNamespace(**{k: types.SimpleNamespace(**v) for k, v in CFG.items()}),
    }
    if not batched:                                                                # drawn batched (the same numbers either way), then RF3's unbatched views
        for k in ("S_inputs", "S_trunk", "Z"):
            d[k] = d[k][0]
    return d, (AH, MU, ME, PE)


def dense_reference(d, mods):
    """RF3's own code on the dense tensors: head forward -> compute_ptm / ComputeIPTM on the logits -> compile on the logits."""
    import torch
    AH, MU, ME, PE = mods
    with torch.no_grad():
        out = d["head"](d["S_inputs"], d["S_trunk"], d["Z"], d["X"], d["seq"], d["rep_atoms"])
        pae_logits, pde_logits = out["pae_logits"], out["pde_logits"]
        tm = {"ptm": ME.compute_ptm(pae_logits, None)}
        iptm = ME.ComputeIPTM.compute(object.__new__(ME.ComputeIPTM), pae=pae_logits, asym_id=d["asym"], is_ligand=d["is_ligand"])
        tm["iptm"] = torch.as_tensor(iptm["iptm_0"]).reshape(1)
        for k in ("iptm_protein_protein", "iptm_protein_ligand", "iptm_ligand_ligand"):
            tm[k] = torch.as_tensor(iptm[f"{k}_0"]).reshape(1)
        ptm_metric = ME.ComputePTM.compute(object.__new__(ME.ComputePTM), pae=pae_logits, asym_id=d["asym"])
        comp = PE.compile_af3_style_confidence_outputs(out["plddt_logits"], pae_logits, pde_logits, d["chain_iid"], d["is_real_atom"],
                                                       d["atom_array"], d["cfg"], batch_idx=0)
    return {"out": out, "tm": tm, "metrics": {"ptm": ptm_metric, "iptm": iptm}, "compile": comp}


# ================================================================================================================ numel census
def _numel_guard():
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten

    class Guard(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.max_numel, self.where = 0, None

        def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            for t in tree_flatten(out)[0]:
                if isinstance(t, torch.Tensor) and t.numel() > self.max_numel:
                    self.max_numel, self.where = int(t.numel()), str(func)
            return out
    return Guard()


# ============================================================================================== the sharded entry (every rank runs it)
def entry_tpconf(I: int, chains, ligand_chain, rows, batched=False):
    import torch
    from rosettafold3_opt import _core, tp_conf
    D = _core.load("mem.rowpair.dist")
    T = _core.load("mem.rowpair.transition")
    C = _core.load("mem.rowpair.confidence")
    SH = _core.load("mem.rowpair.shard")
    P, r = D.world()
    lay = D.Layout.checked(I, P, r, 1)                                            # align 1: any I, uneven shards allowed
    d, mods = synthetic(I, chains, ligand_chain, batched=batched)
    AH, MU, ME, PE = mods
    head = d["head"]
    z_loc = SH.shard_rows(d["Z"], lay, dim=-3).clone()                            # this rank's trunk rows [R, I, c_z] ([1, R, I, c_z] batched) with their own storage

    def make_pair_stack_fn(probe: bool):                                           # the stand-in blocks on the pre-sharded rows (core seams only); S gains the
        def pair_stack_fn(z, S):                                                   # sample dim when the pass carried coordinates — the adapter's rule
            if not probe and S.dim() == 2:                                         # (rowpair._confidence_forward): stock broadcasts S against [1, I, I, c],
                S = S.unsqueeze(0)                                                 # which the pair operand is only once the distance term was added
            for blk in head.pairformer:
                T.transition_rows(blk.z_update, z, lay, rows, add=True)
                S = S + blk.s_lin(C.gather_single_rows(z.mean(dim=-2), lay))
            return z, S
        return pair_stack_fn
    pair_stack_fn = make_pair_stack_fn(False)

    rf3 = tp_conf.RF3Fns(MU.find_bin_midpoints, MU.unbin_logits, AH.discretize_distance_matrix)
    res = {"P": P, "I": I, "R": lay.R, "bounds": lay.bounds, "rows": rows, "checks": {}, "metrics": {}}
    guard = _numel_guard()
    t0 = time.monotonic()
    H = _core.load("mem.rowpair.heads")
    plan = H.ZTrunkPlan(z_loc, passes=1, free=False, park=False, name="z_trunk")       # the trunk shard resident for this pass (the placement forms are _run_ztrunk's)
    with guard:
        out = tp_conf.run_confidence_sharded(head, z_trunk_shard=z_loc, layout=lay, S_inputs_I=d["S_inputs"], S_trunk_I=d["S_trunk"],
                                             X_pred_L=d["X"], rep_atoms=d["rep_atoms"], pair_stack_fn=pair_stack_fn, asym_id=d["asym"],
                                             is_ligand=d["is_ligand"], rf3=rf3, pae=(CFG["pae"]["max_value"], CFG["pae"]["n_bins"]),
                                             pde=(CFG["pde"]["max_value"], CFG["pde"]["n_bins"]), rows=rows, plan=plan)
    plan.end(0, device_needed_next=False)
    plan.close()
    out.pop("moments", None)
    res["metrics"]["sharded_s"] = round(time.monotonic() - t0, 3)
    # ---- the numel census of THIS rank (reduced with max over ranks so rank 0 reports the worst)
    lim = I * I * 64
    mx = torch.tensor([float(guard.max_numel)], dtype=torch.float64)
    D.allreduce_(mx, "max")
    res["metrics"]["max_numel_any_rank"] = int(mx.item())
    res["metrics"]["max_numel_where_rank_local"] = guard.where
    res["metrics"]["numel_limit_IxIx64"] = lim
    res["checks"]["numel_below_IxIx64_every_rank"] = bool(mx.item() < lim)
    # ---- ranks > 0: pair-derived keys are None; replicated keys present (reduced to rank 0 as a count of violations)
    bad = 0
    if r > 0:
        bad += int(out["pae"] is not None) + int(out["pde"] is not None) + int(out["tm"] is not None)
    bad += int(out["plddt_logits"] is None) + int(out["exp_resolved_logits"] is None)
    badt = torch.tensor([float(bad)], dtype=torch.float64)
    D.allreduce_(badt, "sum")
    res["checks"]["rank_contract"] = bool(badt.item() == 0)
    EV = _core.load("mem.rowpair.evidence")
    res["census"] = {k: v for k, v in EV.schedule().items() if str(k).startswith(("conf", "host_gather"))}
    res["checks"]["census_words"] = bool(res["census"].get("conf") == "rows" and res["census"].get("conf_finish") == "rowstats"
                                         and res["census"].get("conf_ln") == "global" and int(res["census"].get("conf_gathers", -1)) == 0
                                         and res["census"].get("conf_collect") == "host" and res["census"].get("conf_probe") == "none"
                                         and "host_gather_cols" in res["census"])
    # ---- the recycle-0 early-stop PROBE (X_pred_L=None; RF3.py reads plddt_logits only): pLDDT on every rank, pair-derived keys None everywhere
    pplan = H.ZTrunkPlan(z_loc, passes=1, free=False, park=False, name="z_trunk.probe")
    with guard:
        pout = tp_conf.run_confidence_sharded(head, z_trunk_shard=z_loc, layout=lay, S_inputs_I=d["S_inputs"], S_trunk_I=d["S_trunk"],
                                              X_pred_L=None, rep_atoms=d["rep_atoms"], pair_stack_fn=make_pair_stack_fn(True), asym_id=d["asym"],
                                              is_ligand=d["is_ligand"], rf3=rf3, rows=rows, plan=pplan, last_use=False)
    pplan.end(0, device_needed_next=True)
    pplan.close()
    pout.pop("moments", None)
    pbad = int(pout["pae"] is not None) + int(pout["pde"] is not None) + int(pout["tm"] is not None) + int(pout["plddt_logits"] is None)
    pbadt = torch.tensor([float(pbad)], dtype=torch.float64)
    D.allreduce_(pbadt, "sum")
    res["checks"]["probe_rank_contract"] = bool(pbadt.item() == 0)
    res["checks"]["probe_census"] = bool(EV.schedule().get("conf_probe") == "plddt_only")
    mx2 = torch.tensor([float(guard.max_numel)], dtype=torch.float64)
    D.allreduce_(mx2, "max")
    res["checks"]["numel_below_IxIx64_every_rank"] = bool(res["checks"]["numel_below_IxIx64_every_rank"] and mx2.item() < lim)
    if r != 0:
        return None
    # ---- rank 0: RF3's own code on the dense tensors, then the comparisons
    ref = dense_reference(d, mods)
    tol_tm, tol_E, tol_summary, tol_s = 1e-6, 1e-4, 1e-3, 1e-5              # the global LayerNorm's moments are re-associated: not bitwise

    def cmp(name, a, b, tol):
        a = a.detach().double().cpu().reshape(-1)
        b = b.detach().double().cpu().reshape(-1)
        diff = float((a - b).abs().max()) if a.numel() else 0.0
        res["metrics"][name + "_maxdiff"] = diff
        res["metrics"][name + "_bitwise"] = bool(torch.equal(a, b))
        res["checks"][name] = bool(a.shape == b.shape and diff <= tol)

    for k in tp_conf.TM_KEYS:
        cmp("tm_" + k, out["tm"][k], ref["tm"][k], tol_tm)
    cmp("E_pae", out["pae"], ref["compile"]["pae"], tol_E)
    cmp("E_pde", out["pde"], ref["compile"]["pde"], tol_E)
    cmp("plddt_logits", out["plddt_logits"], ref["out"]["plddt_logits"], tol_s)
    cmp("exp_resolved_logits", out["exp_resolved_logits"], ref["out"]["exp_resolved_logits"], tol_s)
    with torch.no_grad():                                                      # the stock probe: ConfidenceHead.forward with X_pred_L=None
        pref = d["head"](d["S_inputs"], d["S_trunk"], d["Z"], None, d["seq"], d["rep_atoms"])
    cmp("probe_plddt_logits", pout["plddt_logits"], pref["plddt_logits"], tol_s)
    cmp("probe_exp_resolved_logits", pout["exp_resolved_logits"], pref["exp_resolved_logits"], tol_s)
    res["checks"]["E_on_host"] = bool(out["pae"].device.type == "cpu" and out["pde"].device.type == "cpu")
    res["checks"]["E_shape"] = bool(tuple(out["pae"].shape) == (1, I, I) and tuple(out["pde"].shape) == (1, I, I))
    # RF3's compile VERBATIM on the expected matrices vs RF3's compile on the dense logits
    comp = tp_conf.compile_on_expected(PE, out["plddt_logits"], out["pae"], out["pde"], d["chain_iid"], d["is_real_atom"], d["atom_array"],
                                       d["cfg"], batch_idx=0)
    res["checks"]["unbin_restored"] = bool(PE.unbin_logits is MU.unbin_logits)
    sc, sr = comp["summary_confidences"], ref["compile"]["summary_confidences"]
    res["checks"]["summary_keys"] = bool(sorted(sc) == sorted(sr))
    worst = 0.0

    def _flat(x):
        if isinstance(x, (list, tuple)):
            for y in x:
                yield from _flat(y)
        elif isinstance(x, bool) or x is None:
            yield x
        else:
            yield float(x)
    exact = True
    for k in sr:
        fa, fb = list(_flat(sc.get(k))), list(_flat(sr[k]))
        if len(fa) != len(fb):
            worst = float("inf")
            exact = False
            continue
        for x, y in zip(fa, fb):
            if isinstance(x, float) and isinstance(y, float):
                worst = max(worst, abs(x - y))
                exact = exact and x == y
            else:
                exact = exact and x == y
                if x != y:
                    worst = float("inf")
    res["metrics"]["summary_maxdiff"] = worst
    res["metrics"]["summary_identical"] = exact
    res["checks"]["summary_confidences"] = bool(worst <= tol_summary)
    pm, pr = comp["confidences"]["pae"], ref["compile"]["confidences"]["pae"]
    res["checks"]["confidences_pae_matrix"] = bool(len(pm) == len(pr) == I and max(abs(x - y) for ra, rb in zip(pm, pr) for x, y in zip(ra, rb)) <= 0.011)
    res["checks"]["confidences_keys"] = bool(sorted(comp["confidences"]) == sorted(ref["compile"]["confidences"]))
    mt = tp_conf.metrics_from_tm([out["tm"]])
    res["checks"]["metrics_keys"] = bool(sorted(mt["ptm"]) == sorted(ref["metrics"]["ptm"]) and sorted(mt["iptm"]) == sorted(ref["metrics"]["iptm"]))
    res["metrics"]["tm_sharded"] = {k: float(out["tm"][k][0]) for k in tp_conf.TM_KEYS}
    res["metrics"]["tm_dense"] = {k: float(ref["tm"][k][0]) for k in tp_conf.TM_KEYS}
    res["ok"] = all(res["checks"].values())
    return res


def entry_ztrunk(I: int, chains, ligand_chain, rows, batched=False):
    """Rank body of the trunk-shard placement checks (heads.ZTrunkPlan through tp_conf.run_confidence_sharded, the adapter's per-pass protocol:
    tp_conf begins the pass after the global-LayerNorm moments and before the pair input exists, the caller ends it): the head's outputs under
    resident | in place | parked are BITWISE identical on every rank; the parked form releases the shard's storage before the pair input is
    allocated, keeps ONE host copy across the item's passes (re-served, not re-parked) and leaves the storage released after the last pass; the
    in-place form (an fp32 shard at its last use) consumes the shard; the recycle-0 probe's park restores the shard bitwise-identical with the
    caller's views valid. Returns rank 0's verdicts (every boolean reduced over ranks)."""
    import torch
    from rosettafold3_opt import _core, tp_conf
    D = _core.load("mem.rowpair.dist")
    T = _core.load("mem.rowpair.transition")
    C = _core.load("mem.rowpair.confidence")
    SH = _core.load("mem.rowpair.shard")
    H = _core.load("mem.rowpair.heads")
    TP = _core.load("mem.rowpair.template")
    EV = _core.load("mem.rowpair.evidence")
    P, r = D.world()
    lay = D.Layout.checked(I, P, r, 1)
    d, mods = synthetic(I, chains, ligand_chain, batched=batched)
    AH, MU, ME, PE = mods
    head = d["head"]
    z_ref = SH.shard_rows(d["Z"], lay, dim=-3).clone()                            # [R, I, c_z] ([1, R, I, c_z] batched) fp32, its own storage
    nbytes = int(z_ref.untyped_storage().nbytes())
    rf3 = tp_conf.RF3Fns(MU.find_bin_midpoints, MU.unbin_logits, AH.discretize_distance_matrix)

    def make_pair_stack_fn(probe: bool):                                           # as entry_tpconf's: S gains the sample dim when the pass carried coordinates
        def pair_stack_fn(z, S):
            if not probe and S.dim() == 2:
                S = S.unsqueeze(0)
            for blk in head.pairformer:
                T.transition_rows(blk.z_update, z, lay, rows, add=True)
                S = S + blk.s_lin(C.gather_single_rows(z.mean(dim=-2), lay))
            return z, S
        return pair_stack_fn

    def run(z_loc, plan, i, last_use=None, moments=None, X="full"):
        return tp_conf.run_confidence_sharded(head, z_trunk_shard=z_loc, layout=lay, S_inputs_I=d["S_inputs"], S_trunk_I=d["S_trunk"],
                                              X_pred_L=(d["X"] if X == "full" else None), rep_atoms=d["rep_atoms"], pair_stack_fn=make_pair_stack_fn(X != "full"),
                                              asym_id=d["asym"], is_ligand=d["is_ligand"], rf3=rf3, pae=(CFG["pae"]["max_value"], CFG["pae"]["n_bins"]),
                                              pde=(CFG["pde"]["max_value"], CFG["pde"]["n_bins"]), rows=rows, plan=plan, pass_index=i, last_use=last_use, moments=moments)

    def same(a, b) -> bool:                                                          # bitwise, key by key (pair-derived keys are rank 0's; None elsewhere)
        ok = True
        for k in ("plddt_logits", "exp_resolved_logits", "pae", "pde"):
            x, y = a.get(k), b.get(k)
            if (x is None) != (y is None):
                return False
            if x is not None:
                ok = ok and bool(torch.equal(x, y))
        if a.get("tm") is not None:
            ok = ok and all(bool(torch.equal(a["tm"][k], b["tm"][k])) for k in a["tm"])
        return ok

    checks, facts = {}, {}
    # ---- resident (both levers off): the reference, two passes of one plan
    z = z_ref.clone(); plan = H.ZTrunkPlan(z, passes=2, free=False, park=False)
    o0 = run(z, plan, 0); plan.end(0); o1 = run(z, plan, 1, moments=o0["moments"]); plan.end(1)
    checks["resident_words"] = list(plan.words) == ["resident", "resident"]
    checks["resident_untouched"] = bool(torch.equal(z, z_ref)) and int(z.untyped_storage().nbytes()) == nbytes
    checks["passes_agree"] = same(o0, o1)                                          # the same sample twice: identical
    # ---- parked (park on, free off): storage released BEFORE the pair input, one host copy re-served, released after the last pass
    z = z_ref.clone(); plan = H.ZTrunkPlan(z, passes=2, free=False, park=True)
    parks0 = int(TP.ParkedStorage.STATS["parked"])
    p0 = run(z, plan, 0)
    checks["parked_released_in_pass0"] = int(z.untyped_storage().nbytes()) == 0 and plan.parked is not None
    plan.end(0)                                                                     # device_needed_next=False: the park stays live
    checks["parked_live_between_passes"] = plan.parked is not None and int(z.untyped_storage().nbytes()) == 0
    parks1 = int(TP.ParkedStorage.STATS["parked"])
    p1 = run(z, plan, 1, moments=p0["moments"])
    checks["parked_not_reparked"] = int(TP.ParkedStorage.STATS["parked"]) == parks1 and parks1 == parks0 + 1
    plan.end(1)
    checks["parked_consumed_after_last"] = bool(plan.consumed) and plan.parked is None and int(z.untyped_storage().nbytes()) == 0 and bool(plan.closed)
    checks["parked_words"] = [w.split(":")[0] for w in plan.words] == ["parked", "parked"] and all(str(w).startswith("parked:host") for w in plan.words)
    checks["parked_bitwise"] = same(p0, o0) and same(p1, o1)
    facts["parked_words"], facts["parked_host_gib"] = list(plan.words), plan.host_gib
    # ---- the item's order on the exported line: parked at the ROLL-OUT ENTRY (before any pass), pass 0's LayerNorm moments read from the park
    z = z_ref.clone(); plan = H.ZTrunkPlan(z, passes=2, free=True, park=True, name="z_trunk.entry")
    plan.park_now()
    checks["entry_parked_before_pass0"] = int(z.untyped_storage().nbytes()) == 0 and plan.parked is not None
    e0 = run(z, plan, 0); plan.end(0); e1 = run(z, plan, 1, moments=e0["moments"]); plan.end(1)
    checks["entry_moments_from_park_bitwise"] = all(bool(torch.equal(a, b)) for a, b in zip(e0["moments"], o0["moments"]))
    checks["entry_bitwise"] = same(e0, o0) and same(e1, o1) and int(z.untyped_storage().nbytes()) == 0 and bool(plan.consumed)
    # ---- in place (free on, park off): pass 0 is not the last use -> resident, named; pass 1 embeds into the shard's own storage
    z = z_ref.clone(); plan = H.ZTrunkPlan(z, passes=2, free=True, park=False)
    q0 = run(z, plan, 0); plan.end(0); q1 = run(z, plan, 1, moments=q0["moments"]); plan.end(1)
    checks["inplace_words"] = list(plan.words) == ["resident:free_declined:not_last", "inplace"]
    checks["inplace_consumed"] = bool(plan.consumed) and not bool(torch.equal(z, z_ref)) and int(z.untyped_storage().nbytes()) == nbytes
    checks["inplace_bitwise"] = same(q0, o0) and same(q1, o1)
    facts["inplace_words"] = list(plan.words)
    # ---- both levers (the exported line), two passes: park serves both (never restore-then-inplace); one pass (S=1): in place
    z = z_ref.clone(); plan = H.ZTrunkPlan(z, passes=2, free=True, park=True)
    b0 = run(z, plan, 0); plan.end(0); b1 = run(z, plan, 1, moments=b0["moments"]); plan.end(1)
    checks["both_two_passes_words"] = [w.split(":")[0] for w in plan.words] == ["parked", "parked"]
    checks["both_two_passes_bitwise"] = same(b0, o0) and same(b1, o1)
    z = z_ref.clone(); plan = H.ZTrunkPlan(z, passes=1, free=True, park=True)
    s0 = run(z, plan, 0); plan.end(0)
    checks["both_one_pass_inplace"] = list(plan.words) == ["inplace"] and same(s0, o0)
    # ---- a bf16-shaped rule on CPU: an fp64 shard embedded in fp32 declines in place BY NAME and parks
    z = z_ref.clone().double(); plan = H.ZTrunkPlan(z, passes=1, free=True, park=True)
    checks["dtype_declined_word"] = plan.decide(0, lead_out=(1,), out_dtype=torch.float32) == "parked"
    plan2 = H.ZTrunkPlan(z, passes=1, free=True, park=False)
    checks["dtype_declined_reason"] = plan2.decide(0, lead_out=(1,), out_dtype=torch.float32) == "resident:free_declined:dtype"
    plan.close(); plan2.close()
    # ---- a bf16 trunk shard (RF3's storage under the engine's autocast) embedded into the fp32 pair input: in place declined BY NAME (dtype), the
    #      park serves the bf16 rows, embed_rows is told out_dtype=float32 (a widening store, named) — bitwise equal to the resident bf16 run
    zb = z_ref.clone().to(torch.bfloat16)
    z = zb.clone(); plan = H.ZTrunkPlan(z, passes=1, free=True, park=True)
    h0 = run(z, plan, 0)
    checks["bf16_words"] = str(plan.words[0]).startswith("parked:host") and plan.decide(0, lead_out=(1,), out_dtype=torch.float32).startswith("parked")
    plan.end(0)
    checks["bf16_consumed"] = bool(plan.consumed) and int(z.untyped_storage().nbytes()) == 0
    z = zb.clone(); plan = H.ZTrunkPlan(z, passes=1, free=False, park=False)
    h1 = run(z, plan, 0); plan.end(0)
    checks["bf16_resident_word"] = list(plan.words) == ["resident"] and bool(torch.equal(z, zb))
    checks["bf16_bitwise"] = same(h0, h1) and h0["plddt_logits"].dtype == torch.float32
    z = zb.clone(); plan = H.ZTrunkPlan(z, passes=1, free=True, park=False)
    checks["bf16_free_declined"] = plan.decide(0, lead_out=(1,), out_dtype=torch.float32) == "resident:free_declined:dtype"
    plan.close()
    # ---- the recycle-0 probe under the exported line: parked for the pass, RESTORED after it (recycles 1.. read and write the shard)
    z = z_ref.clone(); view = z[0]                                                  # a view held across the probe, like the engine's references
    plan = H.ZTrunkPlan(z, passes=1, free=True, park=True, name="z_trunk.probe")
    pr = run(z, plan, 0, last_use=False, X=None)
    checks["probe_parked_during"] = int(z.untyped_storage().nbytes()) == 0 and str(plan.words[0]).startswith("parked:host")
    plan.end(0, device_needed_next=True)
    checks["probe_restored_bitwise"] = bool(torch.equal(z, z_ref)) and int(z.untyped_storage().nbytes()) == nbytes and not plan.consumed
    view.add_(1.0)                                                                  # the held view writes through into the restored storage
    checks["probe_views_valid"] = bool(torch.equal(z[0], z_ref[0] + 1.0))
    zr = z_ref.clone(); pref = run(zr, H.ZTrunkPlan(zr, passes=1, free=False, park=False), 0, last_use=False, X=None)
    checks["probe_bitwise"] = same(pr, pref) and pr["pae"] is None and pr["tm"] is None
    sched = EV.schedule()
    checks["census"] = ("conf_ztrunk" in sched and "conf_ztrunk_passes" in sched and sched.get("conf_pairstack") == "inplace"
                        and str(sched.get("conf_ztrunk_pass", "")).startswith("0:"))
    facts["census"] = {k: v for k, v in sched.items() if str(k).startswith("conf_ztrunk") or k == "conf_pairstack"}
    # ---- every rank agrees (a failed check anywhere fails rank 0's verdict)
    bad = torch.tensor([float(sum(1 for v in checks.values() if not v))], dtype=torch.float64)
    D.allreduce_(bad, "sum")
    if r != 0:
        return None
    return {"P": P, "I": I, "checks": checks, "facts": facts, "failed_anywhere": int(bad.item())}


def _run_ztrunk(P, *args):
    from rosettafold3_opt import _core
    launch = _core.load("mem.rowpair.launch")
    return launch.run_sharded(P, entry_ztrunk, *args, mode="big", backend="gloo", cpu_ok=True, nccl_timeout_s=180, run_timeout_s=900)


# ======================================================================================================================== the tests
CASES = [  # P, I, chains, ligand_chain, rows, batched — batched=False is RF3's inference call (single reps [I, c], the shard [R, I, c_z])
    (2, 256, (100, 100, 56), 2, 64, False),       # even I, explicit row block (several blocks per rank)
    (2, 301, (120, 121, 60), 2, None, False),     # uneven I, default row block (one block per rank)
    (3, 203, (90, 70, 43), None, 32, False),      # P=3, uneven shards, no ligand chain (the ligand masks are empty: RF3's 0/1e-6 rows)
    (3, 256, (128, 96, 32), 1, None, False),
    (2, 256, (100, 100, 56), 2, 64, True),        # the [1, ...] forms
]


def _run(P, *args):
    from rosettafold3_opt import _core
    launch = _core.load("mem.rowpair.launch")
    return launch.run_sharded(P, entry_tpconf, *args, mode="big", backend="gloo", cpu_ok=True, nccl_timeout_s=180, run_timeout_s=900)


def _skip_unless_ready():
    import pytest
    if not HAVE_TORCH:
        pytest.skip("BY NAME: torch not importable (the CPU logic check needs torch)")
    if not _have_core():
        pytest.skip("BY NAME: opt_core >= 0.4.3 rc2 (mem.rowpair.dist.gather_rows_to_rank0_host) not importable")


try:
    import pytest

    from . import _stubs

    @pytest.mark.parametrize("P,I,chains,ligand_chain,rows,batched", CASES)
    @_stubs.CORE_GLOO_DENIED
    def test_tp_conf_vs_dense(P, I, chains, ligand_chain, rows, batched):
        _skip_unless_ready()
        res = _run(P, I, tuple(chains), ligand_chain, rows, batched)
        print("RESULT " + json.dumps(res, default=str))
        failed = [k for k, v in res["checks"].items() if not v]
        assert not failed, (failed, res["metrics"])

    @pytest.mark.parametrize("P,batched", [(2, False), (3, False), (2, True)])
    @_stubs.CORE_GLOO_DENIED
    def test_ztrunk_plan_forms(P, batched):
        """ROWPAIR_FREE_ZTRUNK / ROWPAIR_CONF_PARK_ZTRUNK through tp_conf: resident | in place | parked give bitwise-identical head outputs at one and
        two passes per item; the park releases the shard's storage before the pair input and keeps one host copy across passes; the probe's park
        restores the shard bitwise-identical with views valid; the dtype rule declines in place by name."""
        _skip_unless_ready()
        res = _run_ztrunk(P, 256, (100, 100, 56), 2, 64, batched)              # batched=False: RF3's call — single reps [I, c] with the 3-D shard, through the
        print("RESULT " + json.dumps(res, default=str))                        # recycle-0 PROBE (no distance term) and the sample passes under every form
        failed = [k for k, v in res["checks"].items() if not v]
        assert not failed and res["failed_anywhere"] == 0, (failed, res["facts"])

    def test_p1_refused_by_name():
        _skip_unless_ready()
        import torch  # noqa: F811
        from rosettafold3_opt import _core, tp_conf
        RP = _core.load("mem.rowpair")
        D = _core.load("mem.rowpair.dist")
        d, mods = synthetic(64, (40, 24), 1)
        AH, MU, ME, PE = mods
        lay = D.Layout(64, 1, 0, 1)
        H = _core.load("mem.rowpair.heads")
        with pytest.raises(RP.RowpairRefused) as ei:
            tp_conf.run_confidence_sharded(d["head"], z_trunk_shard=d["Z"], layout=lay, S_inputs_I=d["S_inputs"], S_trunk_I=d["S_trunk"],
                                           X_pred_L=d["X"], rep_atoms=d["rep_atoms"], pair_stack_fn=lambda z, s: (z, s), asym_id=d["asym"],
                                           is_ligand=d["is_ligand"], rf3=tp_conf.RF3Fns(MU.find_bin_midpoints, MU.unbin_logits, AH.discretize_distance_matrix),
                                           plan=H.ZTrunkPlan(d["Z"], passes=1, free=False, park=False))
        assert "refused at n_gpu=1" in str(ei.value), str(ei.value)
        print("RESULT " + json.dumps({"case": "p1_refusal", "reason": str(ei.value)}))

    def test_p1_stock_branch_unchanged():
        """At n_gpu=1 nothing of tp_conf runs: the stock head (this file's reference) is the statement — a sanity check that the reference
        itself runs on the shipped config and that the branches tp_conf refuses are the ones the config leaves off."""
        _skip_unless_ready()
        d, mods = synthetic(48, (30, 18), 1)
        ref = dense_reference(d, mods)
        assert tuple(ref["out"]["pae_logits"].shape) == (1, 48, 48, 64)
        h = d["head"]
        assert h.use_af3_style_binning_and_final_layer_norms and not h.use_Cb_distances and not h.layer_norm_along_feature_dimension

    def test_compile_on_expected_restores_unbin_on_error():
        _skip_unless_ready()
        from rosettafold3_opt import tp_conf
        AH, MU, ME, PE = load_rf3()

        def boom(*a, **k):
            raise RuntimeError("inner")
        with pytest.raises(RuntimeError):
            tp_conf.compile_on_expected(PE, None, None, None, compile_fn=boom)
        assert PE.unbin_logits is MU.unbin_logits
except ImportError:                                                           # pytest absent: the __main__ runner below still works
    pass


if __name__ == "__main__":
    fails, ran = [], 0
    t_all = time.monotonic()
    for case in CASES:
        ran += 1
        t0 = time.monotonic()
        try:
            res = _run(case[0], *case[1:])
            print("RESULT " + json.dumps(res, default=str), flush=True)
            bad = [k for k, v in res["checks"].items() if not v]
            if bad:
                raise AssertionError(f"failed checks: {bad}")
            print(f"PASS case={case} ({time.monotonic() - t0:.1f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            fails.append((case, repr(e)))
            print(f"FAIL case={case}: {e!r}", flush=True)
    print(f"SUMMARY ran={ran} failed={len(fails)} wall={time.monotonic() - t_all:.1f}s", flush=True)
    sys.exit(1 if fails else 0)
