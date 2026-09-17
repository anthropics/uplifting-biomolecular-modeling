"""opt_core.mem.rowpair.trunk — the pair representation born row-sharded and carried across recycles (0.4.3).

In-process (no group): the row statements equal the dense statements' row slices BIT-EXACT (pure data movement / integer one-hots /
elementwise adds); the P = 1 refusal is exercised for real; ShardPark round trips bit-exact; block sizes resolve arg > env > ROWPAIR_ROWBLK_MB
(shard.choose_block_rows) and land in the schedule census. Multi-process (gloo on CPU through ``launch.run_sharded(P, entry, cpu_ok=True, backend="gloo")``, P in
{2, 3, 4}): a synthetic AF3-style input embedder (outer sum of two single projections + relpos one-hot linear + token-bond linear), the
recycling statement (parked and resident z_init, cycle 0 without a zeros shard), and the trunk driver over 3 cycles with row-local
synthetic template / MSA / pair-stack callables are compared with INDEPENDENT dense statements written in this file: fp32, one BLAS thread,
asserted max|diff| <= 1e-5, torch.equal REPORTED per piece (RESULT lines). The driver run is held to "no tensor of numel >= N*N*C is ever
produced on a rank" by a TorchDispatchMode census of every op's outputs.

Run: ``OMP_NUM_THREADS=1 python -m pytest tests/test_rowpair_trunk_043.py -q -rfE -s``.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import pytest  # noqa: E402

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:  # noqa: BLE001
    HAVE_TORCH = False

needs_torch = pytest.mark.skipif(not HAVE_TORCH, reason="torch not importable")
TOL32 = 1e-5
N_MP, B_MP, C_MP = 120, 16, 8            # N=120, B=16: every rank owns rows at P in {2, 3, 4} (bounds 64/48/32-row aligned)


# ================================================================================================================ synthetic engine (seeded)
class Synth(object):
    """A seeded AF3-style trunk stand-in, identical on every rank (CPU generator): token features of 3 chains, two single projections, relpos /
    bond / recycle / template / msa / pairstack weights. ``dense_*`` are the plain whole-tensor statements (the references); ``rows_fn`` etc.
    compose opt_core.mem.rowpair.trunk's row statements exactly as an adapter would."""

    def __init__(self, N: int, C: int, seed: int = 5, c_s: int = 6, S: int = 4, clip: int = 4, clip_chain: int = 2, lead=(1,)):
        import torch
        g = torch.Generator().manual_seed(seed)
        rnd = lambda *shape: torch.randn(shape, generator=g, dtype=torch.float32)  # noqa: E731
        self.N, self.C, self.c_s, self.S, self.clip, self.clip_chain, self.lead = N, C, c_s, S, clip, clip_chain, tuple(lead)
        n1, n2 = N // 3, 2 * (N // 3)
        L = tuple(lead)
        self.asym = torch.cat([torch.full((n1,), 1.0), torch.full((n2 - n1,), 2.0), torch.full((N - n2,), 3.0)]).reshape(L + (N,))
        self.entity = torch.cat([torch.full((n2,), 1.0), torch.full((N - n2,), 2.0)]).reshape(L + (N,))
        self.sym = torch.cat([torch.full((n1,), 1.0), torch.full((n2 - n1,), 2.0), torch.full((N - n2,), 1.0)]).reshape(L + (N,))
        self.res = torch.cat([torch.arange(n1), torch.arange(n2 - n1), torch.zeros(N - n2)]).float().reshape(L + (N,))
        self.tok = torch.arange(N).float().reshape(L + (N,))
        tb = torch.zeros(N, N)
        for i in range(n2, N - 1):
            tb[i, i + 1] = 1
            tb[i + 1, i] = 1
        self.token_bonds = tb.int().reshape(L + (N, N))
        self.token_mask = torch.ones(L + (N,))
        self.token_mask[..., n1 - 1] = 0.0
        self.s_input = rnd(*L, N, c_s)
        self.Wi, self.Wj = rnd(c_s, C) / c_s ** 0.5, rnd(c_s, C) / c_s ** 0.5
        self.nbins = 2 * (2 * clip + 2) + 1 + (2 * clip_chain + 2)
        self.Wr = rnd(self.nbins, C) / self.nbins ** 0.5
        self.Wb = rnd(1, C)
        self.ln = torch.nn.LayerNorm(C)
        self.ln_s = torch.nn.LayerNorm(c_s)
        with torch.no_grad():
            self.ln.weight.copy_(rnd(C)); self.ln.bias.copy_(rnd(C))
            self.ln_s.weight.copy_(rnd(c_s)); self.ln_s.bias.copy_(rnd(c_s))
        self.Wz = rnd(C, C) / C ** 0.5
        self.Ws = rnd(c_s, c_s) / c_s ** 0.5
        self.Wt = rnd(C, C) / C ** 0.5                  # "template": row-local update
        self.Wa, self.Wbb = rnd(3, 2) , rnd(3, 2)       # "msa": OPM-like a/b projections of m [S, N, 3] -> [S, N, 2]
        self.Wo = rnd(4, C) / 2.0                       # outer [2 x 2] -> C
        self.Wp = rnd(C, C) / C ** 0.5                  # "pairstack": row-local update
        self.a_i = self.s_input @ self.Wi               # [*, N, C]
        self.b_j = self.s_input @ self.Wj

    # ---- dense statements (the references; plain broadcasting over ALL rows) ------------------------------------------------------
    def dense_relpos_feat(self):
        import torch

        def onehot_rel(pos, cond, clip):
            off = pos[..., :, None] - pos[..., None, :]
            d = torch.clamp(off + clip, 0, 2 * clip)
            d = torch.where(cond, d, torch.full_like(d, 2 * clip + 1))
            return torch.nn.functional.one_hot(d.long(), 2 * clip + 2).float()
        same_chain = self.asym[..., :, None] == self.asym[..., None, :]
        same_res = self.res[..., :, None] == self.res[..., None, :]
        same_ent = self.entity[..., :, None] == self.entity[..., None, :]
        return torch.cat([onehot_rel(self.res, same_chain, self.clip), onehot_rel(self.tok, same_chain & same_res, self.clip),
                          same_ent[..., None].float(), onehot_rel(self.sym, same_ent, self.clip_chain)], dim=-1)      # [*, N, N, nbins]

    def dense_z_init(self):
        z = self.a_i[..., :, None, :] + self.b_j[..., None, :, :]
        z = z + self.dense_relpos_feat() @ self.Wr
        z = z + self.token_bonds[..., None].float() @ self.Wb
        return z

    def update(self, z_rows):                           # recycle: linear(LN(z_prev))  (row-local)
        return self.ln(z_rows) @ self.Wz

    def template(self, z_rows):                         # row-local "template embedder" update
        import torch
        return torch.relu(z_rows @ self.Wt) * 0.1

    def draw_m(self):                                   # the replicated random MSA draw (global RNG: identical streams -> identical m)
        import torch
        return torch.randn(self.lead + (self.S, self.N, 3))

    def opm(self, m, g0, g1):                           # OPM output rows [g0, g1): mean_s outer(a_si, b_sj) -> [*, rows, N, C]
        import torch
        a = (m @ self.Wa)[..., :, g0:g1, :]
        b = m @ self.Wbb
        o = torch.einsum("...sic,...sjd->...ijcd", a, b) / m.shape[-3]
        return o.reshape(o.shape[:-2] + (4,)) @ self.Wo

    def single(self, s):                                # s = s_init + linear_s(LN(s))
        return self.s_input + self.ln_s(s) @ self.Ws

    def pairstack_rows(self, s, z_rows):                # row-local pair update + replicated s update
        import torch
        return s + 0.01 * torch.tanh(s), z_rows + torch.tanh(z_rows @ self.Wp) * 0.1

    def dense_trunk(self, n_cycles):
        """The stock control flow on whole tensors (independent of the driver)."""
        import torch
        z_init = self.dense_z_init()
        s = torch.zeros_like(self.s_input)
        z = torch.zeros_like(z_init)
        for _cycle in range(n_cycles):
            z = z_init + self.update(z)
            z = z + self.template(z)
            m = self.draw_m()
            z = z + self.opm(m, 0, self.N)
            s = self.single(s)
            s, z = self.pairstack_rows(s, z)
        return s, z

    # ---- the adapter's row callables over opt_core.mem.rowpair.trunk ------------------------------------------------------------
    def rows_fn(self, g0, g1):
        import torch
        from opt_core.mem.rowpair import trunk as T
        z = T.outer_sum_rows(self.a_i, self.b_j, g0, g1)
        same_chain = T.same_rows(self.asym, g0, g1)
        same_res = T.same_rows(self.res, g0, g1)
        same_ent = T.same_rows(self.entity, g0, g1)
        feat = torch.cat([T.relpos_onehot_rows(self.res, g0, g1, self.clip, same_chain),
                          T.relpos_onehot_rows(self.tok, g0, g1, self.clip, same_chain & same_res),
                          same_ent[..., None].to(z.dtype),
                          T.relpos_onehot_rows(self.sym, g0, g1, self.clip_chain, same_ent)], dim=-1)
        z = z + feat @ self.Wr
        z = z + T.feature_rows(self.token_bonds, g0, g1, z.device)[..., None].to(z.dtype) @ self.Wb
        return z


def _metrics(got, ref):
    import torch
    maxabs = float((got.double() - ref.double()).abs().max().item()) if got.numel() else 0.0
    return {"equal": bool(torch.equal(got, ref)), "maxabs": maxabs, "shape": list(got.shape)}


# ================================================================================================================ in-process (no group)
@needs_torch
def test_row_statements_equal_dense_slices():
    import torch
    from opt_core.mem.rowpair import trunk as T
    sy = Synth(40, 8)
    feat_full = sy.dense_relpos_feat()
    z_full = sy.dense_z_init()
    pm_full = sy.token_mask[..., :, None] * sy.token_mask[..., None, :]
    for g0, g1 in [(0, 40), (0, 7), (7, 23), (23, 40), (39, 40), (5, 5)]:
        same_chain = T.same_rows(sy.asym, g0, g1)
        assert torch.equal(same_chain, (sy.asym[..., :, None] == sy.asym[..., None, :])[..., g0:g1, :])
        oh = T.relpos_onehot_rows(sy.res, g0, g1, sy.clip, same_chain)
        assert torch.equal(oh, feat_full[..., g0:g1, :, : 2 * sy.clip + 2]), (g0, g1)
        assert torch.equal(T.outer_sum_rows(sy.a_i, sy.b_j, g0, g1), (sy.a_i[..., :, None, :] + sy.b_j[..., None, :, :])[..., g0:g1, :, :])
        assert torch.equal(T.feature_rows(sy.token_bonds, g0, g1, "cpu"), sy.token_bonds[..., g0:g1, :])
        assert torch.equal(T.pair_mask_rows(sy.token_mask, g0, g1), pm_full[..., g0:g1, :])
        blk = sy.rows_fn(g0, g1)
        m = _metrics(blk, z_full[..., g0:g1, :, :])
        assert m["maxabs"] <= TOL32, (g0, g1, m)
    whole = _metrics(sy.rows_fn(0, 40), z_full)
    print("RESULT " + json.dumps({"case": "row_statements", "rows_fn_0_N_vs_dense": whole}))
    assert T.reloffset_rows(sy.res, 0, 3, sy.clip).dtype == torch.int64
    with pytest.raises(T.RowpairRefused):
        T.same_rows(sy.asym, 0, 41)


@needs_torch
def test_p1_refused_by_name_for_real():
    """The structural n_gpu=1 rule: the trunk schedules and the driver refuse a P == 1 layout by name (no monkeypatch, no group)."""
    import torch
    from opt_core.mem.rowpair import RowpairRefused
    from opt_core.mem.rowpair import trunk as T
    from opt_core.mem.rowpair.dist import Layout
    sy = Synth(24, 4)
    lay = Layout(24, 1, 0)
    with pytest.raises(RowpairRefused) as e1:
        T.init_pair_shard(lay, sy.rows_fn, sy.a_i, rows=8)
    assert "refused at n_gpu=1" in e1.value.reason, e1.value.reason
    z = sy.dense_z_init()
    with pytest.raises(RowpairRefused) as e2:
        T.recycle_shard_(torch.zeros_like(z), z, sy.update, lay, rows=8)
    assert "refused at n_gpu=1" in e2.value.reason
    with pytest.raises(RowpairRefused) as e3:
        T.run_trunk_sharded(lay, n_cycles=2, init_rows_fn=sy.rows_fn, init_like=sy.a_i, recycle_update_fn=sy.update)
    assert "run_trunk_sharded" in e3.value.reason and "refused at n_gpu=1" in e3.value.reason
    # a replicated layout (N < P*B) is refused the same way even with P > 1 asked
    with pytest.raises(RowpairRefused):
        T.init_pair_shard(Layout(24, 2, 0, 128), sy.rows_fn, sy.a_i, rows=8)


@needs_torch
def test_shard_park_round_trip_bitwise(monkeypatch):
    import torch
    from opt_core.mem.rowpair import RowpairRefused
    from opt_core.mem.rowpair import trunk as T
    g = torch.Generator().manual_seed(3)
    z = torch.randn((1, 37, 50, 6), generator=g)
    # parked (host copy of a CPU shard needs force_host; a CUDA shard parks into pinned memory and releases its device storage)
    p = T.ShardPark(z.clone(), park=True, force_host=True, name="z_init")
    r = p.record()
    assert r["parked"] and r["where"] == "host" and r["fallback"] is None and r["device_release"] is None, r
    for i0, i1 in [(0, 37), (0, 5), (5, 36), (36, 37), (10, 10)]:
        blk = p.block(i0, i1)
        assert torch.equal(blk, z[..., i0:i1, :, :]) and blk.data_ptr() != z.data_ptr()
    assert torch.equal(p.full(), z)
    with pytest.raises(RowpairRefused):
        p.block(0, 38)
    p.release()
    with pytest.raises(RowpairRefused):
        p.block(0, 1)
    # resident: block is a VIEW of the tensor (no copy), park flag off by default (ROWPAIR_PARK_ZINIT unset)
    monkeypatch.delenv("ROWPAIR_PARK_ZINIT", raising=False)
    q = T.ShardPark(z, name="z_init")
    assert not q.park and q.record()["where"] == "device"
    v = q.block(3, 9)
    assert torch.equal(v, z[..., 3:9, :, :]) and v.data_ptr() == z[..., 3:9, :, :].data_ptr()
    assert q.full() is z
    # the env flag is read when park=None; a CPU tensor without force_host stays resident (parking host->host frees nothing)
    monkeypatch.setenv("ROWPAIR_PARK_ZINIT", "1")
    assert T.ShardPark(z, name="z").park is False
    assert T.ShardPark(z.clone(), force_host=True, name="z").park is True
    with pytest.raises(RowpairRefused):
        T.ShardPark(z.transpose(-1, -2), park=True, force_host=True, name="strided")
    with pytest.raises(RowpairRefused):
        T.ShardPark(torch.zeros(3, 4), name="not_a_shard")


@needs_torch
def test_block_rows_resolution_and_census(monkeypatch):
    """rows given > ROWPAIR_<SCHEDULE>_ROWS (a row count) > ROWPAIR_ROWBLK_MB > default — through shard.choose_block_rows;
    value and source printed in the schedule census."""
    from opt_core.mem.rowpair import evidence, shard
    from opt_core.mem.rowpair import trunk as T
    for n in ("ROWPAIR_RECYCLE_ROWS", "ROWPAIR_INIT_ROWS", "ROWPAIR_ROWBLK_MB"):
        monkeypatch.delenv(n, raising=False)
    assert T._rows_for("trunk_recycle_rows", 1000, 128, 4, 17, "ROWPAIR_RECYCLE_ROWS") == 17
    assert evidence.schedule()["trunk_recycle_rows"] == 17 and evidence.schedule()["trunk_recycle_rows_source"] == "given"
    monkeypatch.setenv("ROWPAIR_RECYCLE_ROWS", "33")
    assert T._rows_for("trunk_recycle_rows", 1000, 128, 4, None, "ROWPAIR_RECYCLE_ROWS") == 33
    assert evidence.schedule()["trunk_recycle_rows_source"] == "env:ROWPAIR_RECYCLE_ROWS"
    assert T._rows_for("trunk_recycle_rows", 1000, 128, 4, 5, "ROWPAIR_RECYCLE_ROWS") == 5           # the argument wins over the env
    monkeypatch.delenv("ROWPAIR_RECYCLE_ROWS", raising=False)
    assert T._rows_for("trunk_recycle_rows", 1000, 128, 4, None, "ROWPAIR_RECYCLE_ROWS") == shard.choose_block_rows(1000, 128, 4)[0]
    assert evidence.schedule()["trunk_recycle_rows_source"] == "default"
    monkeypatch.setenv("ROWPAIR_ROWBLK_MB", "64")
    n = T._rows_for("trunk_recycle_rows", 1000, 128, 4, None, "ROWPAIR_RECYCLE_ROWS")
    assert n == (64 * 2 ** 20) // (1000 * 128 * 4) and evidence.schedule()["trunk_recycle_rows_source"] == "env:ROWPAIR_ROWBLK_MB", n
    n3 = T._rows_for("trunk_init_rows", 1000, 3 * 128, 4, None, "ROWPAIR_INIT_ROWS")
    assert n3 == (64 * 2 ** 20) // (1000 * 384 * 4) and evidence.schedule()["trunk_init_rows_source"] == "env:ROWPAIR_ROWBLK_MB"
    assert T._rows_for("trunk_init_rows", 1000, 384, 4, None, "ROWPAIR_INIT_ROWS", n_max=7) == 7            # never longer than the longest shard
    assert evidence.schedule()["trunk_init_rows_source"].endswith("+n_max")


# ================================================================================================================ multi-process (gloo, CPU)
def _mp(P, entry, *args, **kw):
    from opt_core.mem.rowpair import launch
    os.environ["OMP_NUM_THREADS"] = "1"                                         # inherited by the spawned ranks: one BLAS thread (M-invariance)
    return launch.run_sharded(P, entry, *args, mode="big", backend="gloo", cpu_ok=True, nccl_timeout_s=120, run_timeout_s=600, **kw)


def _gather_flags(d):
    """Every rank's dict to every rank (test plumbing)."""
    import torch.distributed as tdist
    allv = [None] * tdist.get_world_size()
    tdist.all_gather_object(allv, d)
    return allv


def _entry_init_recycle(N: int, C: int, B: int):
    import torch
    torch.set_num_threads(1)
    from opt_core.mem.rowpair import dist as D
    from opt_core.mem.rowpair import trunk as T
    from opt_core.mem.rowpair.shard import unshard_rows
    P, r = D.world()
    lay = D.Layout.checked(N, P, r, B)
    sy = Synth(N, C)
    res = {"case": "init_recycle", "P": P, "N": N, "C": C, "B": B, "rows": {}, "metrics": {}, "checks": {}}
    with torch.no_grad():
        z_ref = sy.dense_z_init()
        g = torch.Generator().manual_seed(9)
        z_prev = torch.randn(z_ref.shape, generator=g)
        rec_ref = z_ref + sy.update(z_prev)
        rec0_ref = z_ref + sy.update(torch.zeros_like(z_ref))
        mine = slice(lay.r0, lay.r1)
        # ---- seam pair-init: 7-row blocks, then the default schedule (one block at this N)
        z7 = T.init_pair_shard(lay, sy.rows_fn, sy.a_i, rows=7)
        zd = T.init_pair_shard(lay, sy.rows_fn, sy.a_i)
        res["metrics"]["init_rows7"] = _metrics(unshard_rows(z7, lay, dim=-3), z_ref)
        res["metrics"]["init_default"] = _metrics(unshard_rows(zd, lay, dim=-3), z_ref)
        res["metrics"]["init_rows7_local"] = _metrics(z7, z_ref[..., mine, :, :])
        # ---- seam recycling: parked z_init (host copy, row blocks staged back) and resident; in place; 5-row blocks and one block
        for tag, park in (("parked", True), ("resident", False)):
            zinit = T.ShardPark(zd.clone().contiguous(), park=park, force_host=True, name=f"z_init_{tag}")
            res["checks"][f"{tag}_where"] = zinit.record()["where"]
            for rows in (5, None):
                z_loc = z_prev[..., mine, :, :].clone().contiguous()
                ptr = z_loc.data_ptr()
                out = T.recycle_shard_(z_loc, zinit, sy.update, lay, rows=rows)
                res["checks"][f"{tag}_rows{rows}_inplace"] = bool(out is z_loc and out.data_ptr() == ptr)
                res["metrics"][f"recycle_{tag}_rows{rows}"] = _metrics(unshard_rows(out, lay, dim=-3), rec_ref)
                out0 = T.recycle_shard_(None, zinit, sy.update, lay, rows=rows)                    # cycle 0: zeros ROW BLOCK, no zeros shard
                res["metrics"][f"recycle0_{tag}_rows{rows}"] = _metrics(unshard_rows(out0, lay, dim=-3), rec0_ref)
            zinit.release()
        # a plain tensor as zinit is accepted (resident)
        out_t = T.recycle_shard_(None, zd, sy.update, lay, rows=11)
        res["metrics"]["recycle0_tensor_rows11"] = _metrics(unshard_rows(out_t, lay, dim=-3), rec0_ref)
    res["rows"] = {str(q): f"{a}:{b}" for q, (a, b) in enumerate(lay.bounds)}
    res["ok"] = all(m["maxabs"] <= TOL32 for m in res["metrics"].values()) and all(v for k, v in res["checks"].items() if k.endswith("inplace"))
    res["bitwise"] = {k: m["equal"] for k, m in res["metrics"].items()}
    return res


class _Recorder(object):
    """Duck-types rowpair.ckpt.TrunkCheckpointer: records the driver's calls; ``resume`` is what try_resume returns."""

    def __init__(self, resume=None):
        self.calls, self.resume = [], resume

    def try_resume(self, layout=None, num_cycles=None, device=None):
        self.calls.append(("try_resume", num_cycles))
        return self.resume

    def phase(self, name):
        self.calls.append(("phase", name))

    def maybe_save_cycle(self, cycle, layout=None, s=None, z_loc=None):
        self.calls.append(("maybe_save_cycle", int(cycle), tuple(z_loc.shape)))

    def save_trunk_final(self, layout=None, s_input=None, s=None, z_loc=None, cycle=None):
        self.calls.append(("save_trunk_final", int(cycle), tuple(z_loc.shape), s_input is not None))


def _numel_census(fn):
    """Run ``fn()`` under a TorchDispatchMode recording the largest tensor numel any op produced; returns (result, max_numel, n_ops)."""
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten
    box = {"max": 0, "ops": 0}

    class Census(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            box["ops"] += 1
            for o in tree_flatten(out)[0]:
                if isinstance(o, torch.Tensor):
                    box["max"] = max(box["max"], int(o.numel()))
            return out

    with Census():
        r = fn()
    return r, box["max"], box["ops"]


def _entry_driver(N: int, C: int, B: int, n_cycles: int, desync_rank: int = -1):
    import torch
    torch.set_num_threads(1)
    from opt_core.mem.rowpair import RowpairRefused
    from opt_core.mem.rowpair import dist as D
    from opt_core.mem.rowpair import evidence
    from opt_core.mem.rowpair import trunk as T
    from opt_core.mem.rowpair.shard import unshard_rows
    P, r = D.world()
    lay = D.Layout.checked(N, P, r, B)
    sy = Synth(N, C)
    res = {"case": "driver", "P": P, "N": N, "C": C, "B": B, "n_cycles": n_cycles, "metrics": {}, "checks": {}, "record": {}}
    SEED = 1234
    with torch.no_grad():
        torch.manual_seed(SEED)
        s_ref, z_ref = sy.dense_trunk(n_cycles)
    r0 = lay.r0

    def template_fn(z_loc, cycle):
        z_loc += sy.template(z_loc)
        return z_loc

    def msa_fn(z_loc, cycle):
        m = sy.draw_m()                                                         # the replicated draw (global RNG; guarded by the driver)
        z_loc += sy.opm(m, lay.r0, lay.r1)
        return z_loc

    def single_fn(s, cycle):
        return sy.single(s)

    def pairstack_fn(s, z_loc, cycle):
        return sy.pairstack_rows(s, z_loc)

    kw = dict(n_cycles=n_cycles, init_rows_fn=sy.rows_fn, init_like=sy.a_i, recycle_update_fn=sy.update, s_init=sy.s_input,
              single_recycle_fn=single_fn, template_fn=template_fn, msa_fn=msa_fn, pairstack_fn=pairstack_fn, s_input=sy.s_input,
              init_rows=9, recycle_rows=13)
    if desync_rank >= 0:
        # one rank's RNG stream out of lockstep: the guard must refuse by name on EVERY rank before msa_fn runs
        torch.manual_seed(SEED + (17 if r == desync_rank else 0))
        try:
            T.run_trunk_sharded(lay, park_zinit=False, guard="rng", **kw)
            res["checks"]["guard_refused"] = False
        except RowpairRefused as exc:
            res["checks"]["guard_refused"] = "trunk.cycle0.rng.cpu" in exc.reason
            res["reason"] = exc.reason[:200]
        res["ok"] = bool(res["checks"]["guard_refused"])
        res["all_ranks"] = [d["ok"] for d in _gather_flags({"ok": res["ok"]})]
        res["ok"] = all(res["all_ranks"])
        return res

    with torch.no_grad():
        # (1) gather="none" under the allocation census (no collective inside: guard off for this arm), parked z_init
        torch.manual_seed(SEED)
        (s1, z1, out1), max_numel, n_ops = _numel_census(lambda: T.run_trunk_sharded(lay, park_zinit=True, guard="off", gather="none", **kw))
        # NOTE park_zinit=True on a CPU shard without force_host stays resident by contract; the parked recycle path is covered by init_recycle
        res["checks"]["numel_max"] = int(max_numel)
        res["checks"]["numel_full_pair"] = int(z_ref.numel())
        res["checks"]["numel_shard"] = int(z1.numel())
        res["checks"]["no_full_pair_materialised"] = bool(max_numel < z_ref.numel()) and n_ops > 0
        res["checks"]["sharded_flag"] = bool(out1.sharded) and tuple(z1.shape[-3:]) == (lay.R, N, C)
        res["metrics"]["z_gather_none_local"] = _metrics(z1, z_ref[..., lay.r0:lay.r1, :, :])
        res["metrics"]["z_gather_none"] = _metrics(unshard_rows(z1, lay, dim=-3), z_ref)
        res["metrics"]["s_gather_none"] = _metrics(s1, s_ref)
        res["record"] = {k: (v if not isinstance(v, dict) else dict(v)) for k, v in out1.record.items()}
        sched = evidence.schedule()
        res["checks"]["census_fields"] = all(k in sched for k in ("trunk_init_rows", "trunk_init_rows_source", "trunk_recycle_rows",
                                                                  "trunk_guard", "trunk_gather", "trunk_replicated", "park_z_init"))
        res["schedule"] = {k: sched[k] for k in sorted(sched) if k.startswith("trunk") or k.startswith("park")}
        # (2) gather="all" with the RNG guard ON (identical streams on every rank -> passes) and a checkpoint recorder
        torch.manual_seed(SEED)
        ck = _Recorder()
        s2, z2, out2 = T.run_trunk_sharded(lay, guard="rng", gather="all", ckpt=ck, **kw)
        res["metrics"]["z_gather_all"] = _metrics(z2, z_ref)
        res["metrics"]["s_gather_all"] = _metrics(s2, s_ref)
        res["checks"]["gather_all_full"] = (not out2.sharded) and tuple(z2.shape) == tuple(z_ref.shape)
        names = [c[0] for c in ck.calls]
        res["checks"]["ckpt_calls"] = (names[0] == "try_resume" and names.count("maybe_save_cycle") == n_cycles - 1
                                      and names[-1] == "save_trunk_final" and ck.calls[-1][1] == n_cycles - 1 and ck.calls[-1][3] is True
                                      and ("phase", "cycle0.recycle") in ck.calls and ("phase", f"cycle{n_cycles - 1}.pairformer") in ck.calls)
        # (3) gather="rank0": the whole tensor on rank 0 only
        torch.manual_seed(SEED)
        s3, z3, out3 = T.run_trunk_sharded(lay, guard="rng", gather="rank0", **kw)
        if r == 0:
            res["metrics"]["z_gather_rank0"] = _metrics(z3, z_ref)
            res["checks"]["rank0_has_full"] = tuple(z3.shape) == tuple(z_ref.shape)
        else:
            res["checks"]["rank0_has_full"] = z3 is None
        # (4) resume from trunk_final: nothing is computed (init_rows_fn never called), the resumed tensors come back
        calls = {"n": 0}

        def counting_rows_fn(g0, g1):
            calls["n"] += 1
            return sy.rows_fn(g0, g1)
        ck2 = _Recorder(resume={"tag": "trunk_final", "cycle": n_cycles - 1, "s": s_ref, "z_loc": z1, "s_input": sy.s_input})
        kw4 = dict(kw, init_rows_fn=counting_rows_fn)
        s4, z4, out4 = T.run_trunk_sharded(lay, ckpt=ck2, gather="none", **kw4)
        res["checks"]["resume_trunk_final"] = bool(out4.resumed) and calls["n"] == 0 and (z4 is z1) and (s4 is s_ref)
        # (5) resume from cycle 0: continues at cycle 1 with the given carry (compare with a dense loop resumed the same way is the
        #     checkpointer's own test; here: start_cycle and the call count of the remaining cycles)
        torch.manual_seed(SEED)
        _ = sy.draw_m()                                                          # the dense stream after cycle 0's draw
        ck3 = _Recorder(resume={"tag": "cycle", "cycle": 0, "s": s1, "z_loc": z1.clone()})
        s5, z5, out5 = T.run_trunk_sharded(lay, ckpt=ck3, gather="none", **kw)
        res["checks"]["resume_cycle0_start"] = out5.start_cycle == 1 and [c for c in ck3.calls if c[0] == "maybe_save_cycle"] == \
            [("maybe_save_cycle", c, tuple(z1.shape)) for c in range(1, n_cycles - 1)]
    flags = _gather_flags({k: bool(v) for k, v in res["checks"].items() if isinstance(v, bool)})
    res["checks_all_ranks"] = {k: all(f.get(k, True) for f in flags) for k in flags[0]}
    res["ok"] = (all(m["maxabs"] <= TOL32 for m in res["metrics"].values()) and all(res["checks_all_ranks"].values())
                 and res["checks"]["no_full_pair_materialised"] and res["checks"]["census_fields"])
    res["bitwise"] = {k: m["equal"] for k, m in res["metrics"].items()}
    return res


def _report(res):
    print("RESULT " + json.dumps(res, sort_keys=True, default=str))
    assert res["ok"], res


@needs_torch
@pytest.mark.parametrize("P", [2, 3, 4])
def test_mp_init_recycle_park(P):
    _report(_mp(P, _entry_init_recycle, N_MP, C_MP, B_MP))


@needs_torch
@pytest.mark.parametrize("P", [2, 3, 4])
def test_mp_driver(P):
    _report(_mp(P, _entry_driver, N_MP, C_MP, B_MP, 3))


@needs_torch
def test_mp_driver_rng_guard_refuses_desync():
    _report(_mp(2, _entry_driver, N_MP, C_MP, B_MP, 2, 1))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-rfE", "-s"] + sys.argv[1:]))
