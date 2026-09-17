"""The package's ROWPIPE lever (rowpipe.py): on the stock Protenix modules loaded from the pinned source under stock/src (real torch on CPU,
seeded random weights, synthetic features) the row-slab relative-position encoding equals the stock module's plane; the row-slab pair path and
the whole slabbed prepare_cache match the hoist kit's whole-plane prepare tensor by tensor (fp32 reassociation class: within 1e-4, and
bitwise when one slab covers the plane — a partial slab is a GEMM of another M, whose last bit the CPU / cuBLAS kernel choice may move); two feature dicts in one process never share conditioning; apply_rowpipe
rebinds the kit's prepare by module global, idempotently; a shape-mode prepare is refused by name."""
import os
import sys
import types

import pytest

from pxdesign_opt import registry, rowpipe, sizeceil

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PROTENIX_SRC = os.path.join(TREE, "stock", "src", "Protenix")
HOIST_DIR = os.path.join(TREE, "opt", "forward", "hoist")
DIMS = dict(c_atom=32, c_atompair=8, c_token=64, c_s=48, c_z=24, c_s_inputs=40)
APT = 3                                                                  # atoms per token in the synthetic features


def _real_stack():
    """Real torch + the pinned protenix source importable (numpy, scipy: protenix/model/utils.py); else skip."""
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "arange") or not hasattr(torch, "Generator"):
        pytest.skip("torch in this interpreter is a stub")
    pytest.importorskip("numpy"); pytest.importorskip("scipy")
    for p in (PROTENIX_SRC, HOIST_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.pop("LAYERNORM_TYPE", None)                               # the torch LayerNorm (no CUDA extension build)
    from protenix.model.modules.diffusion import DiffusionModule
    import pxd_xattempt.hoist as H
    return torch, DiffusionModule, H


def _module(torch, DiffusionModule, seed=0):
    torch.manual_seed(seed)
    dm = DiffusionModule(**DIMS, atom_encoder={"n_blocks": 1, "n_heads": 2}, transformer={"n_blocks": 2, "n_heads": 4, "drop_path_rate": 0},
                         atom_decoder={"n_blocks": 1, "n_heads": 2}).eval()
    with torch.no_grad():                                                # the transitions' output Linear is zero-initialised: randomise so they count
        for p in dm.parameters():
            p.normal_(0.0, 0.3)
    return dm


def _feats(torch, seed, n):
    g = torch.Generator(); g.manual_seed(seed)
    n_atom = n * APT
    chains = torch.randint(1, 4, (n,), generator=g).cumsum(0) // max(2, n // 3)
    f = {"asym_id": chains.long(), "entity_id": (chains // 2).long(), "sym_id": (chains % 2).long(),
         "residue_index": (torch.arange(n) // 2 + torch.randint(0, 3, (n,), generator=g)).long(), "token_index": torch.arange(n).long(),
         "atom_to_token_idx": torch.arange(n).repeat_interleave(APT).long(),
         "ref_pos": torch.randn(n_atom, 3, generator=g), "ref_charge": torch.randn(n_atom, generator=g), "ref_mask": torch.ones(n_atom),
         "ref_element": torch.nn.functional.one_hot(torch.randint(0, 128, (n_atom,), generator=g), 128).float(),
         "ref_atom_name_chars": torch.nn.functional.one_hot(torch.randint(0, 64, (n_atom, 4), generator=g), 64).reshape(n_atom, 256).float(),
         "ref_space_uid": (torch.arange(n_atom) // APT).long()}
    return f, torch.randn(n, DIMS["c_s_inputs"], generator=g), torch.randn(n, DIMS["c_s"], generator=g), torch.randn(n, n, DIMS["c_z"], generator=g)


@pytest.mark.parametrize("n", [1, 7, 37, 64])
def test_relpe_rows_equal_the_stock_module(n):
    torch, DiffusionModule, _ = _real_stack()
    dm = _module(torch, DiffusionModule)
    feats, _, _, _ = _feats(torch, n, n)
    relpe = dm.diffusion_conditioning.relpe
    with torch.no_grad():
        ref = relpe(feats)                                               # RelativePositionEncoding.forward, eval path
        for R in (1, 5, 16, n):
            got = torch.cat([rowpipe.relpe_rows(relpe, feats, i0, min(n, i0 + R)) for i0 in range(0, n, R)], dim=-3)
            err = float((got - ref).abs().max())
            assert got.shape == ref.shape == (n, n, DIMS["c_z"]) and err <= 1e-5, (n, R, err)     # row slabs = the plane's GEMM at another M: fp32 reassociation class
            if R >= n:
                assert torch.equal(got, ref), (n, R, err)                                            # one slab = the plane's own call: bitwise


@pytest.mark.parametrize("n,rows", [(37, 5), (37, 16), (37, 37), (64, 256)])
def test_prepare_cache_slabs_match_the_kit_plane(n, rows, monkeypatch):
    torch, DiffusionModule, H = _real_stack()
    dm = _module(torch, DiffusionModule)
    feats, s_inputs, s_trunk, z_trunk = _feats(torch, 1, n)
    cond = dm.diffusion_conditioning
    monkeypatch.setattr(rowpipe, "ROWS", rows)
    with torch.no_grad():
        pz = torch.cat(tensors=[z_trunk, cond.relpe(feats)], dim=-1); pz = cond.linear_no_bias_z(cond.layernorm_z(pz))
        pz = pz + cond.transition_z1(pz); pz = pz + cond.transition_z2(pz)                       # DiffusionConditioning.forward pair lines
        got = torch.cat([rowpipe.pair_rows(cond, feats, z_trunk, i0, min(n, i0 + rows), False) for i0 in range(0, n, rows)], dim=-3)
        assert got.shape == pz.shape and float((got - pz).abs().max()) <= 1e-4
        C0 = H.prepare_cache(dm, feats, s_inputs, s_trunk, z_trunk, False, n_sample_shape=0)    # the kit's whole-plane prepare (rows mode)
        C1 = rowpipe.prepare_cache(dm, feats, s_inputs, s_trunk, z_trunk, False, n_sample_shape=0)
    f0, f1 = H._flatten_cache(C0), H._flatten_cache(C1)
    assert set(f0) == set(f1)
    assert tuple(C1["pair_z"].shape) == (rowpipe.CARRIER_ROWS, 0) and C1["pair_z"].dtype == torch.float32 and C1["n_sample_shape"] == 1
    assert [tuple(t.shape) for t in C1["tok_bias"]] == [tuple(t.shape) for t in C0["tok_bias"]] and all(t.is_contiguous() for t in C1["tok_bias"])
    worst = 0.0
    for k in f0:
        if k == "pair_z":
            continue
        assert f0[k].shape == f1[k].shape and f0[k].dtype == f1[k].dtype, k
        worst = max(worst, float((f0[k].float() - f1[k].float()).abs().max()))
    assert worst <= 1e-4, worst
    r = rowpipe.slab_rows(n)                                                                      # the balanced slab rows prepare_cache used
    if r >= n:
        assert worst == 0.0, worst                                                                # one slab = the plane's own calls: bitwise
    if rows >= n:
        assert torch.equal(got, pz)
    assert rowpipe.stats()["last"]["N_token"] == n and rowpipe.stats()["last"]["pair_z_resident_bytes"] == 0


def test_two_feature_dicts_never_share_conditioning(monkeypatch):
    torch, DiffusionModule, H = _real_stack()
    dm = _module(torch, DiffusionModule)
    monkeypatch.setattr(rowpipe, "ROWS", 16)
    A, B, Cc = _feats(torch, 1, 37), _feats(torch, 2, 37), _feats(torch, 3, 48)
    with torch.no_grad():
        CA = H._flatten_cache(rowpipe.prepare_cache(dm, *A, False)); CB = H._flatten_cache(rowpipe.prepare_cache(dm, *B, False))
        CC = rowpipe.prepare_cache(dm, *Cc, False); CA2 = H._flatten_cache(rowpipe.prepare_cache(dm, *A, False))
    assert all(not torch.equal(CA[k], CB[k]) for k in CA if k.startswith("tok_bias") or k == "enc_p_lm")
    assert all(torch.equal(CA[k], CA2[k]) for k in CA)                                          # nothing of B or C leaked into A's second prepare
    assert CC["tok_bias"][0].shape[-1] == 48 and tuple(CC["pair_z"].shape) == (rowpipe.CARRIER_ROWS, 0) == (1, 0)
    assert set(rowpipe._STATE) == {"prepares", "last"}                                           # counters only: no tensor state in the module


def test_resolve_cache_reads_only_the_carrier_shape(monkeypatch):
    torch, DiffusionModule, H = _real_stack()
    dm = _module(torch, DiffusionModule)
    feats, s_inputs, s_trunk, z_trunk = _feats(torch, 1, 23)
    monkeypatch.setattr(H, "prepare_cache", rowpipe.prepare_cache)
    lazy = {"_lazy": dict(dm=dm, input_feature_dict=feats, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk, inplace_safe=False, mode="rows"), "_by_ns": {}}
    with torch.no_grad():
        ent = H._resolve_cache(lazy, 5)
        assert tuple(ent["_z_dummy"].shape) == (1, 1) and ent["pair_z"].numel() == 0                   # z.shape[-2] = 1: DiffusionTransformer.forward does not empty_cache between blocks
        assert H._resolve_cache(lazy, 3) is ent                                                  # rows mode: one cache for every chunk N_sample


def test_apply_rebinds_the_kit_prepare_by_module_global():
    fake = types.ModuleType("fake_hoist")
    exec("def prepare_cache(*a, **k):\n    return 'stock'\n\ndef _resolve_cache(C, N_sample):\n    return prepare_cache   # the module global, as hoist._resolve_cache reads it\n", fake.__dict__)
    prepare_cache = fake.prepare_cache
    f1 = rowpipe.apply_rowpipe(fake)
    assert f1 == {"prepare_cache_rebound": True, "resolved_by_global": True}
    assert fake.prepare_cache is rowpipe.prepare_cache and fake._pxdesign_opt_stock_prepare_cache is prepare_cache
    f2 = rowpipe.apply_rowpipe(fake)                                                             # idempotent: no double wrap
    assert f2 == f1 and fake._pxdesign_opt_stock_prepare_cache is prepare_cache


def test_shape_mode_prepare_is_refused_by_name():
    with pytest.raises(RuntimeError, match="refused"):
        rowpipe.prepare_cache(None, {}, None, None, None, False, n_sample_shape=5)


def test_registry_row_matches_the_module():
    assert "rowpipe" in registry.PACKAGE_LEVERS and "rowpipe" in sizeceil.APPLIERS
    lv = registry.PACKAGE_LEVERS["rowpipe"]
    assert "rowpipe.py" in lv.code and lv.switch == "package_levers"
    assert sizeceil.REQUIRED_FIELDS["rowpipe"] == ("prepare_cache_rebound", "resolved_by_global")
    assert rowpipe.ROWS == 256 and rowpipe.GEMM_ROWS == 32768 and rowpipe.CARRIER_ROWS == 1 and rowpipe.HOIST_MODULE == "pxd_xattempt.hoist"
    assert rowpipe.GEMM_ROWS_BY_CC == {"9.0": 32768, "8.0": 65536} and rowpipe.GEMM_ROWS_BY_CC["9.0"] is rowpipe.GEMM_ROWS
    assert rowpipe.RELPE_FEATURES == ("asym_id", "residue_index", "entity_id", "sym_id", "token_index")


def test_gemm_rows_cap_is_keyed_by_compute_capability(monkeypatch):
    """gemm_rows_cap: the slab GEMM cap of the device's compute capability (GEMM_ROWS_BY_CC) — 9.0 runs GEMM_ROWS itself (the H100 schedule is
    the module constant's), 8.0 runs 65536, a capability outside the table or no CUDA device runs GEMM_ROWS; a CUDA device's capability is read
    through torch.cuda.get_device_capability (`capability`); the prepare line's tail names the cap, the capability, and a capability without a
    row. With the balanced schedule the 8.0 cap keeps every slab of a 145–1452-token plane above M 20 800 (the A100 size below which slab GEMMs
    leave stock's plane rows) and wide planes in slabs of at most 65536 // N_tok rows."""
    cap90, cap80 = rowpipe.gemm_rows_cap(cc=(9, 0)), rowpipe.gemm_rows_cap(cc=(8, 0))
    assert cap90 == rowpipe.GEMM_ROWS == 32768 and cap80 == 65536 == rowpipe.gemm_rows_cap(cc="8.0")
    assert rowpipe.gemm_rows_cap(cc=(8, 9)) == rowpipe.gemm_rows_cap(cc=(10, 0)) == rowpipe.GEMM_ROWS            # outside the table: the constant
    assert rowpipe.gemm_rows_cap() == rowpipe.gemm_rows_cap(device=types.SimpleNamespace(type="cpu")) == rowpipe.GEMM_ROWS
    assert rowpipe.capability(None) is None and rowpipe.capability(types.SimpleNamespace(type="cpu")) is None
    asked = []
    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(get_device_capability=lambda device=None: asked.append(device) or (8, 0))
    monkeypatch.setitem(sys.modules, "torch", fake)
    dev = types.SimpleNamespace(type="cuda", index=0)
    assert rowpipe.capability(dev) == "8.0" and rowpipe.gemm_rows_cap(device=dev) == 65536 and asked == [dev, dev]   # a CUDA device: its capability, through torch
    fake.cuda = types.SimpleNamespace(get_device_capability=lambda device=None: (9, 0))
    assert rowpipe.gemm_rows_cap(device=dev) == 32768
    assert rowpipe.gemm_rows_words("9.0") == "gemm_rows=32768 cc=9.0" and rowpipe.gemm_rows_words("8.0") == "gemm_rows=65536 cc=8.0"
    assert rowpipe.gemm_rows_words(None) == "gemm_rows=32768 cc=none" and rowpipe.gemm_rows_words("8.9") == "gemm_rows=32768 cc=8.9 (not in GEMM_ROWS_BY_CC: GEMM_ROWS)"
    assert rowpipe.slab_rows(191, gemm_rows=cap90) == 96 and rowpipe.slab_rows(191, gemm_rows=cap80) == 191           # pdl1's 191-token plane: two 96/95-row slabs on 9.0, one slab on 8.0
    assert rowpipe.slab_rows(275, gemm_rows=cap80) == 138 and rowpipe.slab_rows(700, gemm_rows=cap80) == 88 and rowpipe.slab_rows(5988, gemm_rows=cap80) == 10

    def slabs(n, cap):
        r = rowpipe.slab_rows(n, gemm_rows=cap)
        return [min(r, n - i0) for i0 in range(0, n, r)]                                                             # the slabs pair_derived evaluates
    for n in range(145, 1453):                                                                                        # every slab of the 8.0 schedule: M = rows × N_tok > 20 800
        assert all(k * n > 20800 and k <= rowpipe.ROWS and k * n <= max(cap80, n) for k in slabs(n, cap80)), (n, slabs(n, cap80))
    assert min(k * n for n in range(145, 801) for k in slabs(n, cap80)) == 21025 == 145 * 145                         # the minimum: the 145-token plane itself
    assert min(k * 1453 for k in slabs(1453, cap80)) < 20800                                                          # the first wider plane whose last slab falls under it
    two = [n for n in range(1, 1453) if len(slabs(n, cap80)) == 2]
    assert (two[0], two[-1]) == (257, 362) and min(k * n for n in two for k in slabs(n, cap80)) == 32896               # two slabs only from 257 tokens, M ≥ 32 896
    two90 = [n for n in range(1, 1453) if len(slabs(n, cap90)) == 2]
    assert (two90[0], two90[-1]) == (182, 256) and [n for n in two90 if min(k * n for k in slabs(n, cap90)) <= 20437] == list(range(182, 203))   # under 32768: 182–202-token planes hold a slab at M ≤ 20 437


def test_slab_rows_caps_hold_for_every_slab():
    """slab_rows(N): the cap is min(ROWS, GEMM_ROWS // N); the plane is cut into ceil(N / cap) slabs of ceil(N / n_slabs) rows, the last slab
    taking what remains (it may be smaller). Every slab of the schedule prepare_cache walks (range(0, N, r)) holds both caps: at most ROWS token
    rows and at most GEMM_ROWS token-pair rows (rows × N) per slab GEMM. The words of the design-suite shapes and of big's reach."""
    for n, expect_rows, expect_slabs in ((191, 96, 2), (275, 92, 3), (536, 60, 9), (1788, 18, 100), (2788, 11, 254), (5988, 5, 1198), (37, 37, 1), (1, 1, 1)):
        r = rowpipe.slab_rows(n)
        sizes = [min(r, n - i0) for i0 in range(0, n, r)]                       # the slabs pair_derived evaluates
        assert r == expect_rows and len(sizes) == expect_slabs and sum(sizes) == n, (n, r, len(sizes))
        assert all(1 <= k <= rowpipe.ROWS and k * n <= max(rowpipe.GEMM_ROWS, n) for k in sizes), (n, r, sizes)   # both caps, every slab (a plane wider than GEMM_ROWS runs one row per slab)
        assert all(k == r for k in sizes[:-1]) and 1 <= sizes[-1] <= r, (n, sizes[-3:])                             # equal slabs but the last, which takes the remaining rows
    assert [min(18, 1788 - i0) for i0 in range(0, 1788, 18)][-2:] == [18, 6]                                          # the docstring's example
    assert rowpipe.slab_rows(275, rows=256, gemm_rows=10**9) == 138 and rowpipe.slab_rows(275, rows=512, gemm_rows=10**9) == 275   # ROWS alone: 138+137; unbounded: the plane
    assert rowpipe.slab_rows(64, rows=256) == 64 and rowpipe.slab_rows(37, rows=5) == 5 and rowpipe.slab_rows(37, rows=16) == 13
