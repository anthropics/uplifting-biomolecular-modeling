"""The tp line's pair-stack binding equals STOCK openfold3 0.4.1 on CPU: a randomised ``PairBlock`` and a 2-block ``PairFormerStack`` (the real
classes from the pinned wheel) run dense in every rank as the reference, and through ``tp_rowpair.pairstack.pair_block_rows`` /
``pairformer_stack_rows`` on that rank's row shard of a P-way layout aligned to the attention chunk (opt_core >= 0.4.3's ONE pair-block driver
underneath: ring-streamed tri-mult, all-to-all transposed ending attention, row-local transition, local-query attention-pair-bias); the
unsharded result must equal the dense one to fp32 round-off (max |diff| <= 1e-5; ``torch.equal`` reported). Ranks are gloo processes
(``opt_core.mem.rowpair.launch.run_sharded(P, entry, backend="gloo")``), one BLAS thread each. Skipped when ``openfold3`` or ``torch`` are not
importable or the installed opt_core has no pair-block driver (< 0.4.3).
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("openfold3.core.model.latent.pairformer")
pytest.importorskip("opt_core.mem.rowpair.pairstack")

N, C_Z, C_S, H_APB, NBLK, CHUNK, ACHUNK = int(os.environ.get("TPT_N", "52")), 16, 24, 4, 2, 8, 4
TOL = 1e-5
F = dict(use_deepspeed_evo_attention=False, use_cueq_triangle_kernels=False, use_lma=False)


def _randomise(mod, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in mod.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.2)
    return mod.eval()


def build():
    """Stock modules + inputs, deterministic (every rank builds the same)."""
    from openfold3.core.model.latent.base_blocks import PairBlock
    from openfold3.core.model.latent.pairformer import PairFormerStack
    common = dict(c_hidden_mul=C_Z, c_hidden_pair_att=8, no_heads_pair=4, transition_type="swiglu", transition_n=2, pair_dropout=0.25, fuse_projection_weights=False, inf=1e9)
    blk = _randomise(PairBlock(c_z=C_Z, **common), 1)
    stack = _randomise(PairFormerStack(c_s=C_S, c_z=C_Z, c_hidden_pair_bias=C_S // H_APB, no_heads_pair_bias=H_APB, no_blocks=NBLK, blocks_per_ckpt=None,
                                       tune_chunk_size=False, **common), 3)
    g = torch.Generator().manual_seed(7)
    z = torch.randn(1, N, N, C_Z, generator=g)
    mask = (torch.rand(1, N, N, generator=g) > 0.03).float()
    s = torch.randn(1, N, C_S, generator=g)
    smask = (torch.rand(1, N, generator=g) > 0.05).float()
    return blk, stack, z, mask, s, smask


def references(blk, stack, z, mask, s, smask):
    with torch.no_grad():
        ref_blk = blk(z.clone(), pair_mask=mask, chunk_size=CHUNK, inplace_safe=True, _mask_trans=True, _attn_chunk_size=ACHUNK, **F)
        ref_s, ref_z = stack(s.clone(), z.clone(), smask, mask, chunk_size=CHUNK, inplace_safe=True, _mask_trans=True, **F)
    return ref_blk, ref_s, ref_z


def _entry(n: int):
    """One rank: shard, run the bindings, unshard, compare against the dense stock result computed in this process."""
    import torch
    torch.set_num_threads(1)
    from opt_core.mem.rowpair import dist as D
    from opt_core.mem.rowpair.shard import unshard_rows
    from openfold3_opt.tp_rowpair import pairstack as PS
    os.environ.setdefault("ROWPAIR_TRIMUL_SUB", "8")            # several b sub-blocks per rank at this N (the ring + deferral schedule exercised)
    P, r = D.world()
    lay = D.ctx(n, align=CHUNK)
    blk, stack, z, mask, s, smask = build()
    ref_blk, ref_s, ref_z = references(blk, stack, z, mask, s, smask)
    out = {"rank": r, "P": P, "rows": f"{lay.r0}:{lay.r1}", "metrics": {}}

    def rows(t):
        return t[..., lay.r0:lay.r1, :, :].clone().contiguous()

    def cmp(name, ref, got):
        d = float((ref - got).abs().max())
        out["metrics"][name] = {"maxabs": d, "equal": bool(torch.equal(ref, got)), "finite": bool(torch.isfinite(got).all())}

    with torch.no_grad():
        zb = PS.pair_block_rows(blk, rows(z), mask[..., lay.r0:lay.r1, :].clone(), lay, chunk_size=CHUNK, inplace_safe=True, _mask_trans=True, _attn_chunk_size=ACHUNK, **F)
        cmp("pairblock", ref_blk[0], unshard_rows(zb[0].contiguous(), lay, dim=0))
        s_out, z_out = PS.pairformer_stack_rows(stack, s.clone(), rows(z), smask, mask[..., lay.r0:lay.r1, :].clone(), lay, chunk_size=CHUNK, inplace_safe=True, _mask_trans=True, **F)
        cmp("stack.z", ref_z[0], unshard_rows(z_out[0].contiguous(), lay, dim=0))
        cmp("stack.s", ref_s, s_out)
    out["ok"] = all(m["maxabs"] <= TOL and m["finite"] for m in out["metrics"].values())
    print("TPT_RANK_RESULT", out, flush=True)
    return out


def _mp(P, entry, *args):
    from openfold3_opt.tests._stubs import pin_for_pickling
    pin_for_pickling(globals())          # fresh() in other test modules drops this module from sys.modules; multiprocessing pickles _entry by reference
    from opt_core.mem.rowpair import launch
    os.environ["OMP_NUM_THREADS"] = "1"
    return launch.run_sharded(P, entry, *args, mode="big", backend="gloo", cpu_ok=True, nccl_timeout_s=300, run_timeout_s=1200)


def _results(res):
    if isinstance(res, dict) and "metrics" in res:
        return [res]
    if isinstance(res, (list, tuple)):
        return [r for r in res if isinstance(r, dict)]
    if isinstance(res, dict):
        return [v for v in res.values() if isinstance(v, dict) and "metrics" in v] or [res]
    return [res]


@pytest.mark.parametrize("P", [2, 3])
def test_pairstack_equals_stock(P):
    res = _results(_mp(P, _entry, N))
    assert res, res
    for r in res:
        assert r.get("ok") is True, r


def _entry_kernels(n: int):
    """One rank with the tp line's kernels (flash_triattn, fpf_v4: constants of the binding) on CPU tensors: the triangle-multiplication provider declines every
    unit BY NAME below its size gate (``below_gate``) and the attention dispatch runs its torch statement under the explicit opt-out ``ROWPAIR_TRIATT_CORE=torch``
    (``kernel_torch``; without it the core refuses the flash kernel on CPU tensors by name, test_tp_kernels_cpu.py) — so the sharded results equal stock, and the
    census names both declines."""
    os.environ["ROWPAIR_TRIATT_CORE"] = "torch"
    from openfold3_opt.tp_rowpair import pairstack as PS
    out = _entry(n)
    a, m = PS.triatt_census(), PS.trimul_census()
    out["census"] = {"triatt": {k: a[k] for k in ("kernel", "bound", "calls", "served", "fallback", "fallback_by")}, "trimul": {k: m[k] for k in ("kernels", "bound", "served", "fallback", "fallback_by")}}
    out["ok"] = bool(out["ok"] and a["kernel"] == "flash_triattn" and a["bound"] > 0 and a["calls"] > 0 and a["served"] == 0 and a["fallback"] == a["calls"]
                     and set(a["fallback_by"]) == {"kernel_torch"} and m["kernels"] == "fpf_v4" and m["bound"] > 0 and m["served"] == 0 and set(m["fallback_by"]) == {"below_gate"})
    print("TPT_RANK_CENSUS", out["census"], flush=True)
    return out


def test_pairstack_with_the_kernel_words_equals_stock_on_cpu():
    """The tp line's fused-kernel words bound at P=2 on CPU: every unit declines by name (the size gate; the explicit attention opt-out) and the results equal stock."""
    pytest.importorskip("opt_core.mem.rowpair.trimul_fused")
    res = _results(_mp(2, _entry_kernels, N))
    assert res, res
    for r in res:
        assert r.get("ok") is True, r


def test_refuses_at_P1_by_name():
    """The structural n_gpu=1 rule end to end: the core's driver refuses a P=1 layout by name (no monkeypatch)."""
    from opt_core.mem.rowpair import RowpairRefused
    from opt_core.mem.rowpair import dist as D
    from openfold3_opt.tp_rowpair import pairstack as PS
    blk, stack, z, mask, s, smask = build()
    lay = D.ctx(N, align=CHUNK)                  # no process group: the calling process is rank 0 of P=1
    assert int(lay.P) == 1
    with pytest.raises(RowpairRefused):
        with torch.no_grad():
            PS.pair_block_rows(blk, z.clone(), mask.clone(), lay, chunk_size=CHUNK, inplace_safe=True, _mask_trans=True, _attn_chunk_size=ACHUNK, **F)


if __name__ == "__main__":                     # python test_tp_rowpair_pairstack_cpu.py [P ...] -> prints per-rank metrics
    for P in [int(a) for a in sys.argv[1:]] or [2]:
        print(_mp(P, _entry, N))
