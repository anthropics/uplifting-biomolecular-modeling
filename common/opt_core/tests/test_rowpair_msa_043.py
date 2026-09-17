import re
"""opt_core.mem.rowpair.msa / msa_host — the MSA module on a row-sharded z: OPM output rows (budgeted, chunk-aligned, in place), the
pair-weighted averaging with row-local softmax weights, the MSA transition (replicated | token-sharded), the block/module drivers, host-parked
features with the rank0 broadcast mode, replicated draws. Multi-rank cases run P CPU ranks as THREADS on the family's threaded comm
(``opt_core.testing.run_ranks``); the references are independent DENSE synthetic modules (random weights, the 'zero-init' finals randomised)
evaluated once in the main thread. Row-local arithmetic is per-element identical to the dense statement only if the kernels are M-invariant, so
values are held to ``max|diff| <= 1e-5`` (fp32) with ``torch.equal`` REPORTED; every data-movement step (gathers, broadcasts, host round trips) is
held bitwise. P == 1 refuses by name (the kit's n_gpu=1 path runs the engine's own MSA module). The intra-op thread count is
process-global (``OMP_NUM_THREADS=1`` in the suite command): rank threads never change it, so the dense reference and the ranks run the
same CPU kernels."""
import pytest

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:                                             # noqa: BLE001
    HAVE_TORCH = False

needs_torch = pytest.mark.skipif(not HAVE_TORCH, reason="torch not importable")
TOL32 = 1e-5
P_CASES = [2, 3, 4]


def _run(P, fn, *args, **kw):
    from opt_core.testing import run_ranks
    return run_ranks(P, fn, *args, timeout_s=300.0, **kw)


def _diff(got, ref):
    import torch
    d = (got.double() - ref.double()).abs().max().item() if got.numel() else 0.0
    return {"max_abs_diff": float(d), "bitwise": bool(torch.equal(got, ref)), "shape": tuple(got.shape)}


# ================================================================================================================ synthetic dense engines
def _make_opm(c_m=6, c=4, C_z=5, eps=1e-3, seed=0):
    """An OuterProductMean in an engine's statement order (LN, linear_1/2, mask, transpose, einsum 'bac,dae->bdce', flatten, linear_out, mask
    norm) with the output projection RANDOMISED (an engine zero-inits it)."""
    import torch
    g = torch.Generator().manual_seed(seed)
    W = {"ln_w": torch.rand(c_m, generator=g) + 0.5, "ln_b": torch.randn(c_m, generator=g) * 0.1,
         "w1": torch.randn(c, c_m, generator=g) / c_m ** 0.5, "b1": torch.randn(c, generator=g) * 0.1,
         "w2": torch.randn(c, c_m, generator=g) / c_m ** 0.5, "b2": torch.randn(c, generator=g) * 0.1,
         "wo": torch.randn(C_z, c * c, generator=g) / c, "bo": torch.randn(C_z, generator=g) * 0.1}

    class OPM(object):
        def operands(self, m, mask):                      # replicated, dense M: [S, N, c] -> a, b as [N, S, c]
            mn = torch.nn.functional.layer_norm(m, (c_m,), W["ln_w"], W["ln_b"], 1e-5)
            mk = mask.unsqueeze(-1)
            a = (torch.nn.functional.linear(mn, W["w1"], W["b1"]) * mk).transpose(-2, -3)
            b = (torch.nn.functional.linear(mn, W["w2"], W["b2"]) * mk).transpose(-2, -3)
            return a, b

        def _opm(self, a, b):                             # stock _opm: [Na, S, c] x [Nb, S, c] -> [Na, Nb, C_z]
            outer = torch.einsum("...bac,...dae->...bdce", a, b)
            outer = outer.reshape(outer.shape[:-2] + (-1,))
            return torch.nn.functional.linear(outer, W["wo"], W["bo"])

        def rows(self, a_blk, b, g0, g1, mask, chunk):    # the engine statement for token rows [g0, g1): its own chunking over a rows + norm
            outs = [self._opm(a_blk[i:i + chunk], b) for i in range(0, int(a_blk.shape[0]), chunk)]
            outer = torch.cat(outs, dim=0)
            mk = mask.unsqueeze(-1)
            norm = torch.einsum("...abc,...adc->...bdc", mk[:, g0:g1], mk) + eps
            return outer / norm

        def forward(self, m, mask, chunk):                # dense reference: the global chunk grid from row 0
            a, b = self.operands(m, mask)
            return self.rows(a, b, 0, int(a.shape[0]), mask, chunk)

    return OPM(), C_z


def _make_pwa(c_m=6, C_z=5, H=3, c=4, inf=1e9, seed=1):
    """An MSAPairWeightedAveraging in an engine's statement order with linear_o RANDOMISED."""
    import torch
    g = torch.Generator().manual_seed(seed)
    W = {"lnm_w": torch.rand(c_m, generator=g) + 0.5, "lnm_b": torch.randn(c_m, generator=g) * 0.1,
         "lnz_w": torch.rand(C_z, generator=g) + 0.5, "lnz_b": torch.randn(C_z, generator=g) * 0.1,
         "wz": torch.randn(H, C_z, generator=g) / C_z ** 0.5,
         "wv": torch.randn(H * c, c_m, generator=g) / c_m ** 0.5,
         "wg": torch.randn(H * c, c_m, generator=g) / c_m ** 0.5, "bg": torch.randn(H * c, generator=g) * 0.1,
         "wo": torch.randn(c_m, H * c, generator=g) / (H * c) ** 0.5, "bo": torch.randn(c_m, generator=g) * 0.1}

    class PWA(object):
        heads = H
        chid = c

        def prep(self, z_rows, mask_rows):                # _prep_inputs on rows: [rows, N, C_z] -> logits [H, rows, N]
            zz = torch.nn.functional.layer_norm(z_rows, (C_z,), W["lnz_w"], W["lnz_b"], 1e-5)
            zz = torch.nn.functional.linear(zz, W["wz"]).permute(2, 0, 1)
            return zz + ((mask_rows - 1.0) * inf).unsqueeze(0)

        def values(self, m_chunk):                        # LN_m, linear_v (view/transpose), sigmoid(linear_g): dense M
            n, N = int(m_chunk.shape[0]), int(m_chunk.shape[1])
            mm = torch.nn.functional.layer_norm(m_chunk, (c_m,), W["lnm_w"], W["lnm_b"], 1e-5)
            v = torch.nn.functional.linear(mm, W["wv"]).view(n, N, H, c).transpose(-2, -3)          # [chunk, H, N, c]
            gate = torch.sigmoid(torch.nn.functional.linear(mm, W["wg"], W["bg"])).view(n, N, H, c)   # [chunk, N, H, c]
            return v, gate

        def attend(self, w, state, g0, g1):               # weights x values in the engine's layout, gate rows, flatten: [chunk, q, H*c]
            v, gate = state
            n = int(v.shape[0])
            o = torch.einsum("...hqk,...hkc->...qhc", w.unsqueeze(0).expand(n, *w.shape), v)
            o = o * gate[:, g0:g1]
            return o.reshape(n, g1 - g0, H * c)

        def out(self, o_full):                            # linear_o on the gathered slab: dense M
            return torch.nn.functional.linear(o_full, W["wo"], W["bo"])

        def forward(self, m, z, mask, s_chunk):            # dense reference (stock: softmax of the logits expanded over the sequence chunk)
            S, N = int(m.shape[0]), int(m.shape[1])
            logits = self.prep(z, mask)
            outs = []
            for s0 in range(0, S, s_chunk):
                s1 = min(S, s0 + s_chunk)
                v, gate = self.values(m[s0:s1])
                w = torch.softmax(logits.unsqueeze(0).expand(s1 - s0, H, N, N), dim=-1)
                o = torch.einsum("...hqk,...hkc->...qhc", w, v) * gate
                outs.append(self.out(o.reshape(s1 - s0, N, H * c)))
            return torch.cat(outs, dim=0)

    return PWA()


def _make_transition(c_m=6, n=4, seed=2):
    import torch
    g = torch.Generator().manual_seed(seed)
    W = {"ln_w": torch.rand(c_m, generator=g) + 0.5, "ln_b": torch.randn(c_m, generator=g) * 0.1,
         "w1": torch.randn(n * c_m, c_m, generator=g) / c_m ** 0.5, "w2": torch.randn(c_m, n * c_m, generator=g) / (n * c_m) ** 0.5}

    def fn(m_tok, g0, g1, mask):                          # per-token statement; mask [S, N] sliced by the global token range
        x = torch.nn.functional.layer_norm(m_tok, (c_m,), W["ln_w"], W["ln_b"], 1e-5)
        x = torch.nn.functional.linear(torch.relu(torch.nn.functional.linear(x, W["w1"])), W["w2"])
        return x * mask[:, g0:g1].unsqueeze(-1)
    return fn


def _inputs(N, S=7, c_m=6, C_z=5, seed=3, uneven_mask=True):
    import torch
    g = torch.Generator().manual_seed(seed)
    m = torch.randn(S, N, c_m, generator=g)
    z = torch.randn(N, N, C_z, generator=g)
    msa_mask = (torch.rand(S, N, generator=g) > (0.2 if uneven_mask else -1)).float()
    pair_mask = (torch.rand(N, N, generator=g) > 0.1).float()
    pair_mask.fill_diagonal_(1.0)                          # no fully-masked softmax row
    return m, z, msa_mask, pair_mask


def _layout(N, P, rank, policy, unit):
    from opt_core.mem.rowpair.dist import Layout
    if policy == "aligned":
        return Layout(N, P, rank, align=unit)
    return Layout.checked(N, P, rank, 16)                  # grid policy, B=16 (unit 4 divides it; N=100 -> 7 blocks: every rank of P in 2,3,4 owns rows)


# ================================================================================================================ pure / P == 1
def test_default_opm_rows_sources(monkeypatch):
    from opt_core.mem.rowpair import RowpairRefused
    from opt_core.mem.rowpair.msa import default_opm_rows, default_pwa_qblock, ENV_OPM_ROWS, ENV_PWA_QBLOCK, ENV_PWA_MAX_GB, ENV_ROWBLK_MB
    for name in (ENV_OPM_ROWS, ENV_PWA_QBLOCK, ENV_PWA_MAX_GB, ENV_ROWBLK_MB):
        monkeypatch.delenv(name, raising=False)
    per_row = 1000 * 2 * 128 * 4                                                     # [rows, N=1000, 2*C_z] fp32
    assert default_opm_rows(1000, 128, 250, elt_bytes=4, budget_bytes=10 * per_row) == (10, "budget")     # the agreed budget binds below the 512 MiB target
    assert default_opm_rows(1000, 128, 6, elt_bytes=4, budget_bytes=10 * per_row) == (6, "default+n_max")  # clipped to this rank's rows
    assert default_opm_rows(1000, 128, 250, elt_bytes=4, budget_bytes=1) == (1, "budget")                 # never 0 rows
    assert default_opm_rows(1000, 128, 250, elt_bytes=4) == (250, "default+n_max")                        # 512 MiB / per_row = 524 -> 250
    assert default_opm_rows(1000, 128, 900, elt_bytes=4) == (524, "default")
    monkeypatch.setenv(ENV_ROWBLK_MB, "4")                                                                # 4 MiB / per_row = 4 rows
    assert default_opm_rows(1000, 128, 900, elt_bytes=4) == (4, "env:ROWPAIR_ROWBLK_MB")
    monkeypatch.delenv(ENV_ROWBLK_MB)
    assert default_opm_rows(1000, 128, 250, elt_bytes=4, align=16, budget_bytes=37 * per_row) == (32, "budget+align")   # aligned DOWN to the chunk
    assert default_opm_rows(1000, 128, 250, elt_bytes=4, align=16, budget_bytes=3 * per_row) == (16, "budget+align")    # never below one chunk
    assert default_opm_rows(1000, 128, 250, elt_bytes=4, bytes_per_row=per_row // 2, budget_bytes=10 * per_row) == (20, "budget")
    assert default_opm_rows(70000, 128, 9000, elt_bytes=4, bytes_per_row=1) == (119, "default+cap")      # 2**31 // (70000*256) rows: int32 rule
    monkeypatch.setenv(ENV_OPM_ROWS, "32")
    assert default_opm_rows(1000, 128, 250) == (32, "env:ROWPAIR_OPM_ROWS")
    assert default_opm_rows(1000, 128, 20) == (20, "env:ROWPAIR_OPM_ROWS+n_max")
    assert default_opm_rows(1000, 128, 250, align=12) == (24, "env:ROWPAIR_OPM_ROWS+align")
    for bad in ("0", "-3", "many"):
        monkeypatch.setenv(ENV_OPM_ROWS, bad)
        with pytest.raises(RowpairRefused, match="ROWPAIR_OPM_ROWS"):
            default_opm_rows(1000, 128, 250)
    monkeypatch.delenv(ENV_OPM_ROWS)
    with pytest.raises(RowpairRefused, match="Rmax"):
        default_opm_rows(1000, 128, None)
    # the PWA query block: whole under the GiB threshold, 256 above it, env / given pins, int32 cap, clipped to R
    assert default_pwa_qblock(300, 1000, 8, 16) == (300, "whole")
    assert default_pwa_qblock(3000, 31140, 8, 16) == (256, "auto")                   # 16*8*3000*31140*4 B = 11.1 GiB > 8
    monkeypatch.setenv(ENV_PWA_MAX_GB, "100")
    assert default_pwa_qblock(3000, 31140, 8, 16) == (538, "whole+int32")             # 2**31 // (16*8*31140) = 538: the softmax launch cap
    assert default_pwa_qblock(300, 31140, 8, 16) == (300, "whole")
    monkeypatch.delenv(ENV_PWA_MAX_GB)
    assert default_pwa_qblock(300, 1000, 8, 16, q_block=7) == (7, "given")
    monkeypatch.setenv(ENV_PWA_QBLOCK, "64")
    assert default_pwa_qblock(300, 1000, 8, 16) == (64, "env")
    assert default_pwa_qblock(30, 1000, 8, 16) == (30, "env")
    monkeypatch.delenv(ENV_PWA_QBLOCK)
    assert default_pwa_qblock(4000, 2 ** 20, 8, 512) == (1, "auto+int32")             # 2**31 // (512*8*2**20) = 0 -> 1 row


@needs_torch
def test_p1_statements_refuse_by_name_and_draws_pass_through():
    """At P == 1 every sharded MSA statement / driver refuses BY NAME (the real refusal, no monkeypatch); the draw forms and the host module
    are the local statements."""
    import torch
    from opt_core.mem.rowpair import RowpairRefused
    from opt_core.mem.rowpair.dist import Layout
    from opt_core.mem.rowpair import msa, msa_host, transition
    lay = Layout(64, 1, 0)
    z = torch.zeros(64, 64, 4)
    m = torch.zeros(3, 64, 5)
    calls = {
        "opm_rows_budgeted": lambda: msa.opm_rows_budgeted(m, m, lay, lambda a, b: None, C_z=4),
        "opm_rows": lambda: transition.opm_rows(m, m, lay, lambda a, b: None),
        "pwa_rows": lambda: msa.pwa_rows(m, torch.zeros(2, 64, 64), lay, values_fn=lambda x: x, attend_fn=lambda w, s, g0, g1: w, out_fn=lambda o: o),
        "pwa_bias_rows": lambda: msa.pwa_bias_rows(lambda zr, g0, g1: zr, z, lay),
        "msa_transition_rows": lambda: msa.msa_transition_rows(lambda mm, g0, g1: mm, m, lay, shard_tokens=False),
        "msa_block_sharded": lambda: msa.msa_block_sharded(m, z, lay, opm=lambda m_, z_: z_, pair_block=lambda z_: z_),
        "msa_module_sharded": lambda: msa.msa_module_sharded(m, z, lay, []),
    }
    for name, call in calls.items():
        with pytest.raises(RowpairRefused) as ei:
            call()
        assert f"{name}: refused at n_gpu=1" in str(ei.value), str(ei.value)
    g = torch.Generator().manual_seed(0)
    d = msa.draw_replicated(lambda: torch.randperm(20, generator=g)[:5])
    assert d.shape == (5,) and not hasattr(msa, "guard_replicated")          # ONE guard in the family (rowpair.trunk); msa proves via dist.allreduce_checksum
    # host module at P == 1: park (mode rank0 degenerates to all), select rows, rows_to_device == index_select, residency census
    feats = {"msa": torch.randn(40, 9, 6, generator=g), "deletion": torch.randn(40, 9, generator=g), "small": torch.ones(2)}
    ref = feats["msa"].clone()
    facts = msa_host.park_features(feats, ["msa", "deletion", "absent"], mode="rank0", row_dims={"msa": 0, "deletion": 0})
    w = msa_host.where(feats["msa"])                                        # host_pinned with CUDA present, host otherwise: the park vocabulary
    assert facts == {"mode": "rank0", "parked": ["msa", "deletion"], "placeholders": [], "where": {"msa": w, "deletion": w}, "rows": {"msa": 40, "deletion": 40}}
    assert w in ("host", "host_pinned")
    assert msa_host.is_parked(feats["msa"]) and not msa_host.is_parked(feats["small"])
    ht = msa_host.HostTensor(feats["msa"], "msa", row_dim=0)
    idx = torch.tensor([3, 1, 39, 3])
    rows = msa_host.rows_to_device(lambda: ht.rows_to(idx, "cpu"), shape=(4, 9, 6), dtype=torch.float32, device="cpu", mode="rank0")
    assert torch.equal(rows, ref[idx]) and torch.equal(ht.select(idx), ref.index_select(0, idx))
    assert ht.placeholder().shape == (0, 9, 6) and msa_host.host_placeholder(feats["deletion"], -2).shape == (0, 9)
    with pytest.raises(RowpairRefused, match="declared"):
        msa_host.rows_to_device(lambda: ht.rows_to(idx, "cpu"), shape=(5, 9, 6), dtype=torch.float32, device="cpu", mode="all")
    assert msa_host.host_mode("0") is None and msa_host.host_mode("1") == "all" and msa_host.host_mode("rank0") == "rank0"
    with pytest.raises(RowpairRefused, match="ROWPAIR_MSA_HOST"):
        msa_host.host_mode("everywhere")
    assert msa_host.sync_host_features_(feats, ["msa"], "cpu", mode="all") is feats                      # no group: no-op
    assert torch.equal(msa_host.bcast_chunked_(rows, 0, chunk_gb=1e-9), ref[idx])
    text = msa_host.residency_text(feats, "p1")
    assert "[residency] p1:" in text and "msa" in text
    rr = msa_host.residency_rows(feats)
    assert rr[0][1] == "msa" and rr[0][4] in ("host", "pinned")



def _engine_tokens():
    """Engine names derived from the release tree beside common/ (empty when the core is checked out alone) — the hygiene suite's one reader."""
    try:
        from tests.test_instances_backend_hygiene import engine_tokens
    except Exception:  # noqa: BLE001
        return []
    return engine_tokens()


def _mentions(token: str, text: str) -> bool:
    """The hygiene suite's ONE word matcher (a token inside another word is not a mention)."""
    try:
        from tests.test_instances_backend_hygiene import mentions
    except Exception:  # noqa: BLE001
        return re.search(r"(?<![a-z0-9_])" + re.escape(token.lower()) + r"(?![a-z0-9_])", text.lower()) is not None
    return mentions(token, text)

def test_module_is_engine_free_and_imports_lazily():
    """msa.py / msa_host.py name no engine and import torch only through ._torch / the callables."""
    import ast
    import opt_core.mem.rowpair.msa as M
    import opt_core.mem.rowpair.msa_host as MH
    for mod in (M, MH):
        tree = ast.parse(open(mod.__file__).read())
        tops = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                tops.update(a.name.split(".")[0] for a in n.names)
            elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                tops.add(n.module.split(".")[0])
        for t in tops:
            assert not t.startswith(("torch", "openfold", "boltz", "protenix", "chai", "alphafold", "jax") + tuple(_engine_tokens())), (mod.__name__, t)
        src = open(mod.__file__).read().lower()
        for word in ("openfold", "boltz", "protenix") + tuple(_engine_tokens()):
            # the name as a WORD (not inside another identifier: a kit named like a substring of an ordinary word is not a mention)
            assert not _mentions(word, src), (mod.__name__, word)


# ================================================================================================================ multi-rank (threaded)
def _rank_opm(rank, P, N, policy, unit, m, msa_mask, z, rows):
    """OPM rows on this rank: out-of-place rows, in-place accumulation into the z shard (budgeted call, printed), census."""
    import torch
    from opt_core.mem.rowpair import msa as MS
    from opt_core.mem.rowpair.transition import opm_rows
    from opt_core.mem.rowpair.shard import shard_rows
    from opt_core.mem.rowpair.evidence import schedule
    lay = _layout(N, P, rank, policy, unit)
    opm, C_z = _make_opm()
    a, b = opm.operands(m, msa_mask)                                               # replicated, dense M
    outer = lambda a_blk, b_, g0, g1: opm.rows(a_blk, b_, g0, g1, msa_mask, unit)  # noqa: E731  the engine statement (own chunking + norm)
    mine = opm_rows(a, b, lay, outer, rows=rows, global_rows=True, row_dim=0, align=unit)                    # out of place [R, N, C_z]
    z_shard = shard_rows(z, lay).contiguous().clone()
    z_ref = z_shard + mine                                                          # out-of-place residual
    MS.opm_rows_budgeted(a, b, lay, outer, rows=rows, C_z=C_z, out=z_shard, add=True, align=unit, global_rows=True, row_dim=0)   # in place
    z_auto = shard_rows(z, lay).contiguous().clone()
    MS.opm_rows_budgeted(a, b, lay, outer, C_z=C_z, out=z_auto, add=True, align=unit, global_rows=True, row_dim=0,
                         budget_bytes=3 * N * 2 * C_z * 4)                            # budget source: 3 rows -> aligned to one chunk
    sched = dict(schedule())
    return {"r0": lay.r0, "r1": lay.r1, "rows": mine, "z_inplace": z_shard, "z_ref": z_ref, "z_auto": z_auto,
            "inplace_eq": bool(torch.equal(z_shard, z_ref)), "auto_eq": bool(torch.equal(z_auto, z_ref)), "sched": sched}


@needs_torch
@pytest.mark.parametrize("P", P_CASES)
@pytest.mark.parametrize("policy,N", [("grid", 100), ("aligned", 46)])
def test_opm_rows_equal_dense_slices(P, policy, N):
    """OPM output rows of every rank == dense[r0:r1] (an engine statement with its own chunk grid + mask norm; uneven N; both layout
    policies); the in-place accumulation into the z shard is BIT-EXACT the out-of-place residual; the budgeted call prints its schedule."""
    import torch
    unit = 4
    m, z, msa_mask, _pm = _inputs(N)
    opm, C_z = _make_opm()
    dense = opm.forward(m, msa_mask, unit)                                          # [N, N, C_z], global chunk grid
    outs = _run(P, _rank_opm, N, policy, unit, m, msa_mask, z, 8)
    seen = 0
    for r, o in enumerate(outs):
        d = _diff(o["rows"], dense[o["r0"]:o["r1"]])
        print(f"opm_rows P={P} {policy} N={N} rank={r} rows[{o['r0']}:{o['r1']}] max|diff|={d['max_abs_diff']:.3e} torch.equal={d['bitwise']}")
        assert d["max_abs_diff"] <= TOL32, d
        assert o["inplace_eq"], "in-place accumulation must equal the out-of-place residual bitwise"
        assert o["auto_eq"], "the budgeted (auto rows) in-place call must equal the residual bitwise"
        dz = _diff(o["z_inplace"], z[o["r0"]:o["r1"]] + dense[o["r0"]:o["r1"]])
        assert dz["max_abs_diff"] <= TOL32, dz
        seen += o["r1"] - o["r0"]
        assert o["sched"].get("opm_rows_source") in ("given", "given+n_max", "budget+align", "budget+align+n_max") and o["sched"].get("msa_m") == "replicated"
        assert o["sched"].get("opm_align") == unit and o["sched"].get("opm_inplace") is True
    assert seen == N, f"rows covered {seen} != N={N}"
    full = torch.cat([o["rows"] for o in outs], dim=0)
    print(f"opm_rows P={P} {policy} N={N} assembled torch.equal(dense)={bool(torch.equal(full, dense))}")


def _rank_pwa(rank, P, N, policy, unit, m, z, pair_mask, s_chunk, q_block):
    import torch
    from opt_core.mem.rowpair import msa as MS
    from opt_core.mem.rowpair.shard import shard_rows
    from opt_core.mem.rowpair.evidence import schedule
    lay = _layout(N, P, rank, policy, unit)
    pwa = _make_pwa()
    z_shard = shard_rows(z, lay).contiguous()
    bias = MS.pwa_bias_rows(lambda zr, g0, g1: pwa.prep(zr, pair_mask[g0:g1]), z_shard, lay)
    bias_blocked = MS.pwa_bias_rows(lambda zr, g0, g1: pwa.prep(zr, pair_mask[g0:g1]), z_shard, lay, rows=3)     # the LN-guard path forced
    upd = MS.pwa_rows(m, bias, lay, values_fn=pwa.values, attend_fn=pwa.attend, out_fn=pwa.out, s_chunk=s_chunk, q_block=q_block)
    sched = dict(schedule())
    upd_whole = MS.pwa_rows(m, bias, lay, values_fn=pwa.values, attend_fn=pwa.attend, out_fn=pwa.out, s_chunk=s_chunk)   # q = whole
    return {"r0": lay.r0, "r1": lay.r1, "upd": upd, "qblock_eq": bool(torch.equal(upd, upd_whole)),
            "qblock_maxdiff": float((upd - upd_whole).abs().max()),
            "bias_blocked_eq": bool(torch.equal(bias, bias_blocked)), "bias_shape": tuple(bias.shape), "sched": sched}


@needs_torch
@pytest.mark.parametrize("P", P_CASES)
@pytest.mark.parametrize("policy,N", [("grid", 100), ("aligned", 46)])
def test_pwa_rows_equal_dense(P, policy, N):
    """The pair-weighted-averaging m-update (row-local softmax weights, dense-M m-side GEMMs, token rows gathered before linear_o) == the dense
    module on every rank; q_block on/off identical within the M-invariance class (bit-exact reported); the LN-guarded (row-blocked) logits BIT-EXACT
    the one-call logits."""
    m, z, _mm, pair_mask = _inputs(N)
    pwa = _make_pwa()
    s_chunk = 3
    dense = pwa.forward(m, z, pair_mask, s_chunk)                                   # [S, N, c_m]
    outs = _run(P, _rank_pwa, N, policy, 4, m, z, pair_mask, s_chunk, 5)
    for r, o in enumerate(outs):
        d = _diff(o["upd"], dense)
        print(f"pwa_rows P={P} {policy} N={N} rank={r} rows[{o['r0']}:{o['r1']}] max|diff|={d['max_abs_diff']:.3e} torch.equal={d['bitwise']} "
              f"qblock_on_off_equal={o['qblock_eq']} ln_guard_blocked_equal={o['bias_blocked_eq']}")
        assert d["max_abs_diff"] <= TOL32, d
        # q_block changes only the query count M of the attend GEMMs: bit-exact iff the kernel is M-invariant (the stated class), else <= TOL32
        assert o["qblock_maxdiff"] <= TOL32, ("q_block must not change the values beyond the M-invariance class", o["qblock_maxdiff"], o["qblock_eq"])
        assert o["bias_blocked_eq"], "row-blocked logits (LN guard) must equal the one-call logits bitwise"
        assert o["bias_shape"] == (pwa.heads, o["r1"] - o["r0"], N)
        # the schedule census is process-global: under rank THREADS the last writer wins (this rank's q_block=5 call or a peer's whole call)
        assert o["sched"].get("pwa_qblock_source") in ("given", "whole") and o["sched"].get("pwa_s_chunk") == s_chunk


def _rank_transition(rank, P, N, policy, unit, m, msa_mask):
    import torch
    from opt_core.mem.rowpair import msa as MS
    lay = _layout(N, P, rank, policy, unit)
    fn0 = _make_transition()
    fn = lambda mt, g0, g1: fn0(mt, g0, g1, msa_mask)  # noqa: E731
    rep = MS.msa_transition_rows(fn, m, lay, shard_tokens=False)
    shd = MS.msa_transition_rows(fn, m, lay, shard_tokens=True)
    try:
        MS.msa_transition_rows(fn, m, lay, shard_tokens=None)
        refused = False
    except Exception as e:  # noqa: BLE001
        refused = "shard_tokens" in str(e)
    return {"rep": rep, "shd": shd, "explicit_only": refused}


@needs_torch
@pytest.mark.parametrize("P", P_CASES)
def test_msa_transition_sharded_equals_replicated(P):
    N = 46
    m, _z, msa_mask, _pm = _inputs(N)
    dense = _make_transition()(m, 0, N, msa_mask)
    outs = _run(P, _rank_transition, N, "aligned", 4, m, msa_mask)
    for r, o in enumerate(outs):
        assert bool(__import__("torch").equal(o["rep"], dense)), "replicated form is the dense call"
        d = _diff(o["shd"], dense)
        print(f"msa_transition P={P} rank={r} token_sharded max|diff|={d['max_abs_diff']:.3e} torch.equal={d['bitwise']}")
        assert d["max_abs_diff"] <= TOL32, d
        assert o["explicit_only"], "the replicated | token-sharded choice is an explicit argument (refused by name when missing)"


def _dense_msa_module(m, z, msa_mask, pair_mask, unit, s_chunk, plan):
    """The dense reference MSA module: per block z += OPM(m); [m += PWA(m, z); m += transition(m)]; z = pair_block(z) — orders per ``plan``."""
    import torch
    opm, _C = _make_opm()
    pwa = _make_pwa()
    tr = _make_transition()
    for opm_first, update in plan:                                                  # the engine statement order (late OPM inside the update)
        if opm_first:
            z = z + opm.forward(m, msa_mask, unit)
        if update:
            m = m + pwa.forward(m, z, pair_mask, s_chunk)
            m = m + tr(m, 0, int(m.shape[1]), msa_mask)
            if not opm_first:
                z = z + opm.forward(m, msa_mask, unit)
        z = _dense_pair_block(z)
    return m, z


def _dense_pair_block(z):
    import torch
    return z * 0.5 + torch.tanh(z)                                                  # a row-local stand-in for the trunk's pair block


def _rank_module(rank, P, N, unit, m, z, msa_mask, pair_mask, s_chunk, plan):
    import torch
    from opt_core.mem.rowpair import msa as MS
    from opt_core.mem.rowpair.shard import shard_rows, unshard_rows
    from opt_core.mem.rowpair.evidence import schedule
    lay = _layout(N, P, rank, "aligned", unit)
    opm, C_z = _make_opm()
    pwa = _make_pwa()
    tr = _make_transition()

    def opm_fn(m_, z_shard):
        a, b = opm.operands(m_, msa_mask)
        return MS.opm_rows_budgeted(a, b, lay, lambda ab, b_, g0, g1: opm.rows(ab, b_, g0, g1, msa_mask, unit), rows=8, C_z=C_z, out=z_shard,
                                    add=True, align=unit, global_rows=True, row_dim=0)

    def update_fn(m_, z_shard):
        bias = MS.pwa_bias_rows(lambda zr, g0, g1: pwa.prep(zr, pair_mask[g0:g1]), z_shard, lay)
        m_ = m_ + MS.pwa_rows(m_, bias, lay, values_fn=pwa.values, attend_fn=pwa.attend, out_fn=pwa.out, s_chunk=s_chunk)
        return m_ + MS.msa_transition_rows(lambda mt, g0, g1: tr(mt, g0, g1, msa_mask), m_, lay, shard_tokens=True)

    blocks = [{"opm": opm_fn, "msa_update": update_fn if upd else None, "pair_block": _dense_pair_block, "opm_first": first} for first, upd in plan]
    z_shard = shard_rows(z, lay).contiguous().clone()
    cleared = []
    z_out = MS.msa_module_sharded(m.clone(), z_shard, lay, blocks, between_blocks=cleared.append)
    return {"z_full": unshard_rows(z_out.contiguous(), lay), "between": list(cleared), "sched": dict(schedule())}


@needs_torch
@pytest.mark.parametrize("P", [2, 3])
def test_msa_module_driver_matches_dense(P):
    """The block/module drivers with engine callables (both OPM orders, a skipped last m-update under either order — the (opm_first=False,
    no update) block runs no OPM) reproduce the dense module's z."""
    import torch
    N, unit, s_chunk = 46, 4, 3
    plan = [(True, True), (False, True), (False, False), (True, False)]
    m, z, msa_mask, pair_mask = _inputs(N)
    _m_ref, z_ref = _dense_msa_module(m, z, msa_mask, pair_mask, unit, s_chunk, plan)
    outs = _run(P, _rank_module, N, unit, m, z, msa_mask, pair_mask, s_chunk, plan)
    for r, o in enumerate(outs):
        d = _diff(o["z_full"], z_ref)
        print(f"msa_module P={P} rank={r} z max|diff|={d['max_abs_diff']:.3e} torch.equal={d['bitwise']}")
        assert d["max_abs_diff"] <= 10 * TOL32, d
        assert o["between"] == [0, 1, 2, 3] and o["sched"].get("msa_blocks") == 4 and o["sched"].get("msa_m") == "replicated"
    assert all(torch.equal(o["z_full"], outs[0]["z_full"]) for o in outs), "every rank assembles the same z"


def _rank_host(rank, P, feats0, idx, chunk_gb):
    """msa_host across ranks: mode all (own host copies; sync from rank 0 repairs a corrupted rank), mode rank0 (placeholders + device
    broadcast of the selected rows), chunked broadcasts forced into many pieces, replicated draws proven / refused."""
    import torch
    from opt_core.mem.rowpair import RowpairRefused
    from opt_core.mem.rowpair import msa as MS, msa_host as MH
    res = {}
    ref_msa, ref_del = feats0["msa"], feats0["deletion"]
    n_sel, N = int(idx.shape[0]), int(ref_msa.shape[1])
    # --- mode all: every rank parks its own copy; a corrupted rank is repaired by the sync from rank 0 (meta + chunked pieces)
    feats = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in feats0.items()}
    if rank == 1:
        feats["msa"] = feats["msa"][:5] * 0 + 7.0                                    # wrong shape AND values on rank 1
        feats["deletion"] = feats["deletion"] + 1.0
    facts = MH.park_features(feats, ["msa", "deletion"], mode="all", row_dims=0)
    res["all_parked"] = facts["parked"]
    MH.sync_host_features_(feats, ["msa", "deletion"], "cpu", src=0, chunk_gb=chunk_gb, mode="all")
    res["sync_eq"] = bool(torch.equal(feats["msa"], ref_msa) and torch.equal(feats["deletion"], ref_del))
    hm, hd = MH.HostTensor(feats["msa"], "msa", 0), MH.HostTensor(feats["deletion"], "deletion", 0)
    build = lambda: torch.cat([hm.select(idx), hd.select(idx).unsqueeze(-1)], dim=-1).to("cpu", torch.float32).contiguous()  # noqa: E731
    ref_rows = torch.cat([ref_msa.index_select(0, idx), ref_del.index_select(0, idx).unsqueeze(-1)], dim=-1)
    rows_all = MH.rows_to_device(build, shape=(n_sel, N, ref_msa.shape[2] + 1), dtype=torch.float32, device="cpu", mode="all")
    res["rows_all_eq"] = bool(torch.equal(rows_all, ref_rows))
    # --- mode rank0: ranks > 0 hold zero-row placeholders; the selected rows arrive by the chunked device broadcast
    feats = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in feats0.items()}
    facts = MH.park_features(feats, ["msa", "deletion"], mode="rank0", row_dims={"msa": 0, "deletion": 0})
    res["rank0_placeholders"] = facts["placeholders"]
    res["rank0_rows"] = dict(facts["rows"])                                   # the parked row count, equal on every rank (placeholders carry it)
    res["rank0_shapes"] = (tuple(feats["msa"].shape), tuple(feats["deletion"].shape))
    hm, hd = MH.HostTensor(feats["msa"], "msa", 0, log=False), MH.HostTensor(feats["deletion"], "deletion", 0, log=False)
    build = lambda: torch.cat([hm.select(idx), hd.select(idx).unsqueeze(-1)], dim=-1).to("cpu", torch.float32).contiguous()  # noqa: E731
    rows_r0 = MH.rows_to_device(build, shape=(n_sel, N, ref_msa.shape[2] + 1), dtype=torch.float32, device="cpu", mode="rank0", chunk_gb=chunk_gb)
    res["rows_rank0_eq"] = bool(torch.equal(rows_r0, ref_rows))
    b = (torch.arange(24, dtype=torch.float32).reshape(4, 6) if rank == 0 else torch.zeros(4, 6))
    MH.bcast_chunked_(b, 0, chunk_gb=chunk_gb)
    res["bcast_eq"] = bool(torch.equal(b, torch.arange(24, dtype=torch.float32).reshape(4, 6)))
    flags = (torch.tensor([True, False, True]) if rank == 0 else torch.tensor([False, False, False]))
    MH.bcast_chunked_(flags, 0, chunk_gb=chunk_gb)
    res["bcast_bool_eq"] = flags.tolist() == [True, False, True]
    # --- replicated draws: identical generators pass the guard (checksums equal on every rank); a diverged rank is refused BY NAME
    g = torch.Generator().manual_seed(1234)
    same = torch.randperm(40, generator=g)[:n_sel]
    from opt_core.mem.rowpair.dist import allreduce_checksum
    res["guard"] = allreduce_checksum(same, name="msa_sel")                        # the primitive under the family's one guard
    g_bad = torch.Generator().manual_seed(1234 + rank)
    try:
        allreduce_checksum(torch.randperm(40, generator=g_bad)[:n_sel], name="msa_sel_bad")
        res["guard_refused"] = False
    except RowpairRefused as e:
        res["guard_refused"] = "msa_sel_bad" in str(e)
    g_r = torch.Generator().manual_seed(99 + rank)                                  # per-rank generators out of lockstep: rank 0's draw wins
    drawn = MS.draw_replicated(lambda: torch.randperm(40, generator=g_r)[:n_sel], name="msa_sel_draw")
    res["drawn"] = drawn
    res["residency"] = MH.residency_rows(feats)[0][1:]
    return res


@needs_torch
@pytest.mark.parametrize("P", P_CASES)
def test_msa_host_roundtrip_rank0_broadcast_and_draws(P):
    import torch
    g = torch.Generator().manual_seed(7)
    feats0 = {"msa": torch.randn(40, 9, 6, generator=g), "deletion": torch.randn(40, 9, generator=g), "token_index": torch.arange(9)}
    idx = torch.tensor([3, 1, 39, 3, 0])
    outs = _run(P, _rank_host, feats0, idx, 96 / 2 ** 30)                            # 96-byte pieces: every broadcast is chunked
    for r, o in enumerate(outs):
        assert o["all_parked"] == ["msa", "deletion"], o["all_parked"]
        assert o["sync_eq"], f"rank {r}: host copies must equal rank 0's after sync_host_features_"
        assert o["rows_all_eq"] and o["rows_rank0_eq"], f"rank {r}: selected rows must equal index_select of the reference (both modes)"
        assert o["bcast_eq"] and o["bcast_bool_eq"], f"rank {r}: chunked broadcast"
        assert (o["rank0_placeholders"] == [] if r == 0 else o["rank0_placeholders"] == ["msa", "deletion"]), (r, o["rank0_placeholders"])
        assert o["rank0_rows"] == outs[0]["rank0_rows"] and set(o["rank0_rows"]) == {"msa", "deletion"} and min(o["rank0_rows"].values()) > 0, (r, o["rank0_rows"])
        assert (o["rank0_shapes"] == ((40, 9, 6), (40, 9)) if r == 0 else o["rank0_shapes"] == ((0, 9, 6), (0, 9))), (r, o["rank0_shapes"])
        assert o["guard"] == outs[0]["guard"] and len(o["guard"]) == P, "identical draws: every rank's checksum, equal"
        assert o["guard_refused"], f"rank {r}: a diverged draw must be refused by name"
        assert torch.equal(o["drawn"], outs[0]["drawn"]), "draw_replicated: rank 0's draw on every rank"
        print(f"msa_host P={P} rank={r} sync/rows_all/rows_rank0/bcast bitwise=True checksums={o['guard'][0]}")


def _rank_token_sharded(rank, P, N, unit, m, z, msa_mask, pair_mask, s_chunk):
    """m_layout='token_sharded': OPM with a local `a` and the all-gathered `b`; PWA gathering only the values source; transition on the shard."""
    import torch
    from opt_core.mem.rowpair import RowpairRefused
    from opt_core.mem.rowpair import msa as MS
    from opt_core.mem.rowpair.shard import shard_rows, unshard_rows
    from opt_core.mem.rowpair.evidence import schedule
    lay = _layout(N, P, rank, "aligned", unit)
    opm, C_z = _make_opm()
    pwa = _make_pwa()
    tr = _make_transition()
    m_loc = m[:, lay.r0:lay.r1].contiguous()                                        # this rank's token shard [S, R, c_m]
    a_loc, b_loc = opm.operands(m_loc, msa_mask[:, lay.r0:lay.r1])                  # per-token statements on the shard: [R, S, c]
    b_full = unshard_rows(b_loc.contiguous(), lay, dim=0)                            # the NON-LOCAL operand only: [N, S, c]
    rows = MS.opm_rows_budgeted(a_loc, b_full, lay, lambda ab, b_, g0, g1: opm.rows(ab, b_, g0, g1, msa_mask, unit), rows=8, C_z=C_z,
                                align=unit, global_rows=True, row_dim=0, a_local=True)
    sched_opm = dict(schedule())
    z_shard = shard_rows(z, lay).contiguous()
    bias = MS.pwa_bias_rows(lambda zr, g0, g1: pwa.prep(zr, pair_mask[g0:g1]), z_shard, lay)
    upd_loc = MS.pwa_rows(m_loc, bias, lay, values_fn=pwa.values, attend_fn=pwa.attend, out_fn=pwa.out, s_chunk=s_chunk, m_layout="token_sharded")
    sched_pwa = dict(schedule())
    tr_loc = MS.msa_transition_rows(lambda mt, g0, g1: tr(mt, g0, g1, msa_mask), m_loc, lay, shard_tokens=True, m_layout="token_sharded")
    refusals = {}
    for name, call in {
        "layout_word": lambda: MS.pwa_rows(m_loc, bias, lay, values_fn=pwa.values, attend_fn=pwa.attend, out_fn=pwa.out, m_layout="sharded"),
        "replicated_m_given_shard": lambda: MS.pwa_rows(m_loc, bias, lay, values_fn=pwa.values, attend_fn=pwa.attend, out_fn=pwa.out),
        "transition_replicated_on_shard": lambda: MS.msa_transition_rows(lambda mt, g0, g1: mt, m_loc, lay, shard_tokens=False, m_layout="token_sharded"),
        "a_local_wrong_rows": lambda: MS.opm_rows_budgeted(a_loc, b_loc, lay, lambda ab, b_: ab, rows=8, C_z=C_z, row_dim=0, a_local=True),
    }.items():
        try:
            call()
            refusals[name] = None
        except RowpairRefused as e:
            refusals[name] = str(e)[:80]
    return {"r0": lay.r0, "r1": lay.r1, "rows": rows, "upd_loc": upd_loc, "tr_loc": tr_loc, "refusals": refusals,
            "msa_m_opm": sched_opm.get("msa_m"), "msa_m_pwa": sched_pwa.get("msa_m")}


@needs_torch
@pytest.mark.parametrize("P", P_CASES)
def test_token_sharded_m_layout(P):
    """The named ``m_layout='token_sharded'`` path: every rank's OPM rows, PWA m-update (its tokens) and transition equal the dense slices;
    layout words / mismatched operands refuse by name."""
    import torch
    N, unit, s_chunk = 46, 4, 3
    m, z, msa_mask, pair_mask = _inputs(N)
    opm, _C = _make_opm()
    dense_opm = opm.forward(m, msa_mask, unit)
    dense_pwa = _make_pwa().forward(m, z, pair_mask, s_chunk)
    dense_tr = _make_transition()(m, 0, N, msa_mask)
    outs = _run(P, _rank_token_sharded, N, unit, m, z, msa_mask, pair_mask, s_chunk)
    for r, o in enumerate(outs):
        r0, r1 = o["r0"], o["r1"]
        d1, d2, d3 = _diff(o["rows"], dense_opm[r0:r1]), _diff(o["upd_loc"], dense_pwa[:, r0:r1]), _diff(o["tr_loc"], dense_tr[:, r0:r1])
        print(f"token_sharded m P={P} rank={r} tokens[{r0}:{r1}] opm max|diff|={d1['max_abs_diff']:.3e} eq={d1['bitwise']} "
              f"pwa max|diff|={d2['max_abs_diff']:.3e} eq={d2['bitwise']} transition max|diff|={d3['max_abs_diff']:.3e} eq={d3['bitwise']}")
        assert d1["max_abs_diff"] <= TOL32 and d2["max_abs_diff"] <= TOL32 and d3["max_abs_diff"] <= TOL32, (d1, d2, d3)
        assert o["msa_m_opm"] == "token_sharded" and o["msa_m_pwa"] == "token_sharded"
        assert all(v is not None for v in o["refusals"].values()), o["refusals"]


@needs_torch
def test_msa_host_where_words_are_public():
    """is_pinned / where are public and park_features records per-key placement words (adapters never re-derive pinned|pageable)."""
    import torch
    from opt_core.mem.rowpair import evidence, msa_host as MH
    feats = {"msa": torch.randn(7, 5), "deletion_matrix": torch.randn(7, 5), "small": torch.zeros(0)}
    facts = MH.park_features(feats, ["msa", "deletion_matrix", "small", "absent"], mode="all", pin=False, log=False)
    assert facts["parked"] == ["msa", "deletion_matrix"] and facts["where"] == {"msa": "host", "deletion_matrix": "host"}
    assert MH.where(feats["msa"]) == "host" and MH.is_pinned(feats["msa"]) is False and "is_pinned" in MH.__all__ and "where" in MH.__all__
    assert evidence.schedule()["msa_host_where"] == "msa:host,deletion_matrix:host"
    ph = MH.host_placeholder(feats["msa"], 0)
    assert MH.where(ph) == "placeholder" and tuple(ph.shape) == (0, 5) and MH.parked_rows(ph) == 7 and MH.parked_rows(feats["msa"]) == 7
    assert facts["rows"] == {"msa": 7, "deletion_matrix": 7}
    assert MH.HostTensor(torch.randn(4, 3), name="x", pin=False, log=False).facts()["where"] == "host"
    assert MH.where("not a tensor") == "n/a"
