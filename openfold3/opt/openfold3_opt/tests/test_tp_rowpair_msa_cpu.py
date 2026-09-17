"""tp_rowpair.msa vs the STOCK openfold3 0.4.1 MSA module on CPU: P ranks run as threads on opt_core's threaded communicator
(``opt_core.testing.run_ranks``); the stock modules (random weights, zero-inits randomised) run dense once in the main thread and every
rank's row shard is compared with the corresponding rows. Values are held to fp32 ``max|diff| <= 1e-5`` with ``torch.equal`` REPORTED
(row-local GEMMs differ from the dense call only in M); host round trips / broadcasts are held bitwise. Needs the ``openfold3`` package
(the kit's stock wheel, ``pip install --no-deps``) + opt_core >= 0.4.3; skipped by name otherwise.

Items: outer-product mean output rows (in place and out of place), pair-weighted averaging (local-row logits, gathered rows before
``linear_o``), MSA transition (replicated and token-sharded), the full ``MSAModuleStack`` (2 blocks incl. the pair stack through
``pairstack.pair_block_rows``), the MSA-embedder host hook (``msa.MSA_HOST`` rank0: m / mask bitwise vs stock, RNG stream position
identical, ranks > 0 hold zero rows in mode rank0), and a pair-shape guard (no rank materialises an ``[N, N, >=C_z]`` tensor)."""
import os
import threading

import pytest

try:
    import torch
    HAVE_TORCH = True
except Exception:                                             # noqa: BLE001
    HAVE_TORCH = False
try:
    import openfold3.core.model.latent.msa_module  # noqa: F401
    HAVE_OF3 = True
except Exception:                                             # noqa: BLE001
    HAVE_OF3 = False
try:
    import opt_core.mem.rowpair.msa  # noqa: F401
    from opt_core.testing import run_ranks  # noqa: F401
    HAVE_CORE = True
except Exception:                                             # noqa: BLE001
    HAVE_CORE = False

needs_stack = pytest.mark.skipif(not (HAVE_TORCH and HAVE_OF3 and HAVE_CORE), reason="needs torch + openfold3 0.4.1 + opt_core>=0.5.10 (rowpair.msa)")
TOL32 = 1e-5
P_CASES = [2, 3]
N_TOK, S_SEQ, CHUNK = 44, 7, 4                                 # N not a multiple of P*chunk: ragged last rank; chunk 4 = the layout's row unit
KW = dict(use_deepspeed_evo_attention=False, use_cueq_triangle_kernels=False, use_triton_triangle_kernels=False, use_lma=False)


# ================================================================================================================ shared helpers (the other tp_rowpair CPU tests import these)
def run(P, fn, *args, **kw):
    from opt_core.testing import run_ranks
    torch.set_num_threads(1)
    return run_ranks(P, fn, *args, timeout_s=900.0, **kw)


def diff(got, ref):
    d = (got.double() - ref.double()).abs().max().item() if got.numel() else 0.0
    return {"max_abs_diff": float(d), "bitwise": bool(torch.equal(got, ref)), "shape": tuple(got.shape)}


def randomise_(mod, seed, scale=0.2):
    """Random weights everywhere (OpenFold3 zero-initialises output projections / gates, which would make the comparison vacuous)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in mod.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * scale)
    return mod.eval()


def arch_config():
    import copy
    from openfold3.projects.of3_all_atom.config.model_config import model_config
    return copy.deepcopy(model_config).architecture


def dims():
    """``(c_m, c_z, c_s_input)`` of the stock architecture (read from the MSA module / embedder kwargs)."""
    arch = arch_config()
    msa_kw, emb_kw = dict(arch.msa.msa_module), dict(arch.msa.msa_module_embedder)
    return int(msa_kw["c_m"]), int(msa_kw["c_z"]), int(emb_kw["c_s_input"])


def layout(N, P, rank, chunk=CHUNK):
    from opt_core.mem.rowpair.dist import Layout
    return Layout(N, P, rank, align=chunk)


PAIRSTACK_MODES = ["oracle", "sharded"]      # oracle: the block's pair stack = stock dense blocks on the gathered slab, re-sliced (isolates THIS binding);
                                            # sharded: pairstack.pair_block_rows / pair_stack_rows (A-openfold3's binding of the core pair-block driver)


def oracle_pair_block_rows(blk, z_loc, pair_mask_loc, lay, chunk_size=None, inplace_safe=True, _mask_trans=True, _attn_chunk_size=None, **flags):
    """Test stand-in for ``pairstack.pair_block_rows``: gather the rows, run the STOCK block dense, return this rank's rows. Proves the
    caller's statements and control flow independently of the sharded pair-block driver (which has its own proof)."""
    from opt_core.mem.rowpair.shard import unshard_rows
    z = z_loc.pop() if isinstance(z_loc, list) else z_loc
    Z = unshard_rows(z.contiguous(), lay, dim=-3).contiguous()
    M = unshard_rows(pair_mask_loc.contiguous(), lay, dim=-2).contiguous()
    Z = blk(Z, M, chunk_size=chunk_size, inplace_safe=inplace_safe, _mask_trans=_mask_trans, _attn_chunk_size=_attn_chunk_size, **flags)
    return Z[..., lay.r0:lay.r1, :, :].contiguous()


def oracle_pair_stack_rows(blocks, z_loc, pair_mask_loc, lay, **kw):
    z = z_loc.pop() if isinstance(z_loc, list) else z_loc
    for b in blocks:
        z = oracle_pair_block_rows(b, z, pair_mask_loc, lay, **kw)
    return z


def oracle_pairformer_stack_rows(stack, s, z_loc, single_mask, pair_mask_loc, lay, chunk_size=None, inplace_safe=True, _mask_trans=True, _attn_chunk_size=None, **flags):
    """Test stand-in for ``pairstack.pairformer_stack_rows``: the STOCK ``PairFormerStack`` dense on the gathered pair rows; returns ``(s, rows)``."""
    from opt_core.mem.rowpair.shard import unshard_rows
    z = z_loc.pop() if isinstance(z_loc, list) else z_loc
    Z = unshard_rows(z.contiguous(), lay, dim=-3).contiguous()
    M = unshard_rows(pair_mask_loc.contiguous(), lay, dim=-2).contiguous()
    s, Z = stack(s, Z, single_mask, M, chunk_size=chunk_size, inplace_safe=inplace_safe, _mask_trans=_mask_trans, **flags)
    return s, Z[..., lay.r0:lay.r1, :, :].contiguous()


def use_pairstack(monkeypatch, mode):
    """``oracle``: route ``tp_rowpair.pairstack.pair_block_rows / pair_stack_rows / pairformer_stack_rows`` to the dense oracles for this test."""
    from openfold3_opt.tp_rowpair import pairstack as PS
    if mode == "oracle":
        monkeypatch.setattr(PS, "pair_block_rows", oracle_pair_block_rows)
        monkeypatch.setattr(PS, "pair_stack_rows", oracle_pair_stack_rows)
        monkeypatch.setattr(PS, "pairformer_stack_rows", oracle_pairformer_stack_rows)


class PairGuard(object):
    """Records every tensor an op returns while active whose shape carries two token-sized dims with >= ``min_c`` trailing channels — a rank
    that materialises ``[.., N, N, C]`` fails the sharding contract even when its rows compare equal."""

    def __init__(self, N, min_c):
        from torch.utils._python_dispatch import TorchDispatchMode
        self.N, self.min_c, self.hits = int(N), int(min_c), []
        guard = self

        class _Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = func(*args, **(kwargs or {}))
                for t in (out if isinstance(out, (tuple, list)) else (out,)):
                    if isinstance(t, torch.Tensor) and t.dim() >= 3:
                        s = tuple(t.shape)
                        for i in range(len(s) - 2):
                            if s[i] == guard.N and s[i + 1] == guard.N and s[-1] >= guard.min_c and i + 1 < len(s) - 1:
                                guard.hits.append((str(func), s))
                return out
        self._mode = _Mode()

    def __enter__(self):
        self._mode.__enter__()
        return self

    def __exit__(self, *a):
        return self._mode.__exit__(*a)


# per-THREAD default RNG for the engine's un-generatored draws (torch.randint / randperm in MSAModuleEmbedder): real ranks are processes with
# their own synchronous RNG stream; threads share one global generator, so the draws are routed to a thread-local generator seeded alike.
_tls = threading.local()
_ORIG = {}


def _gen():
    g = getattr(_tls, "g", None)
    if g is None:
        g = torch.Generator()
        g.manual_seed(getattr(_tls, "seed", 100))
        _tls.g = g
    return g


def reseed_thread_rng(seed=100):
    _tls.seed, _tls.g = seed, None


def install_thread_rng():
    if _ORIG:
        return
    _ORIG["randint"], _ORIG["randperm"] = torch.randint, torch.randperm

    def _randint(*a, **k):
        k.pop("generator", None)
        return _ORIG["randint"](*a, generator=_gen(), **k)

    def _randperm(n, **k):
        k.pop("generator", None)
        return _ORIG["randperm"](n, generator=_gen(), **k)
    torch.randint, torch.randperm = _randint, _randperm


def uninstall_thread_rng():
    if _ORIG:
        torch.randint, torch.randperm = _ORIG.pop("randint"), _ORIG.pop("randperm")


# ================================================================================================================ fixtures
def build_msa_stack(seed=0, no_blocks=2):
    from openfold3.core.model.latent.msa_module import MSAModuleStack
    kw = dict(arch_config().msa.msa_module)
    kw["no_blocks"] = no_blocks
    kw["tune_chunk_size"] = False
    stack = randomise_(MSAModuleStack(**kw), seed + 3)
    stack.chunk_size_tuner = None
    return stack


def msa_inputs(N=N_TOK, S=S_SEQ, seed=7):
    g = torch.Generator().manual_seed(seed)
    c_m, c_z, _ = dims()
    m0 = torch.randn(1, S, N, c_m, generator=g)
    z0 = torch.randn(1, N, N, c_z, generator=g) * 0.5
    msa_mask = (torch.rand(1, S, N, generator=g) > 0.1).float()
    token_mask = torch.ones(1, N)
    token_mask[0, N // 3] = 0.0
    pair_mask = token_mask[..., None] * token_mask[..., None, :]
    return m0, z0, msa_mask, pair_mask


# ================================================================================================================ tests
@needs_stack
@pytest.mark.parametrize("P", P_CASES)
def test_msa_statements_vs_stock(P, monkeypatch):
    """OPM output rows (in place / out of place), pair-weighted averaging, MSA transition (replicated + token-sharded) of block 0 vs stock."""
    from openfold3.core.utils.tensor_utils import add
    from openfold3_opt.tp_rowpair import msa as MSA
    monkeypatch.setenv("ROWPAIR_VERBOSE", "0")
    stack = build_msa_stack()
    blk = stack.blocks[0]
    m0, z0, msa_mask, pair_mask = msa_inputs()
    N, C_z = z0.shape[-2], z0.shape[-1]
    with torch.no_grad():
        z_opm_ref = add(z0.clone(), blk.outer_product_mean(m0, mask=msa_mask, chunk_size=CHUNK, inplace_safe=True), inplace=True)
        pwa_ref = blk.msa_att_row(m0, z=z0, mask=pair_mask, chunk_size=CHUNK)
        mt_ref = blk.msa_transition(m0, mask=msa_mask, chunk_size=CHUNK)

    def rank_fn(rank, P_, shard_trans):
        os.environ["ROWPAIR_MSA_TRANS_SHARD"] = "1" if shard_trans else "0"
        lay = layout(N, P_, rank)
        z_loc = z0[:, lay.r0:lay.r1].clone().contiguous()
        pm_loc = pair_mask[:, lay.r0:lay.r1].contiguous()
        out = {"rank": rank, "rows": (lay.r0, lay.r1)}
        ref_rows = z_opm_ref[:, lay.r0:lay.r1].contiguous()              # sliced OUTSIDE the guard (slicing the dense reference is an [N, N, C] op)
        with torch.no_grad(), PairGuard(N, C_z) as guard:
            z1 = MSA.opm_rows(blk.outer_product_mean, m0, msa_mask, z_loc.clone(), lay, chunk_size=CHUNK, inplace_safe=True)
            z2 = MSA.opm_rows(blk.outer_product_mean, m0, msa_mask, z_loc.clone(), lay, chunk_size=CHUNK, inplace_safe=False)
            pwa = MSA.pwa_rows(blk.msa_att_row, m0, z_loc, pm_loc, lay, chunk_size=CHUNK)
            mt = MSA.msa_transition_rows(blk, m0, msa_mask, lay, CHUNK, None)
        out["opm_inplace"], out["opm_outofplace"] = diff(z1, ref_rows), diff(z2, ref_rows)
        out["pwa"], out["msa_transition"] = diff(pwa, pwa_ref), diff(mt, mt_ref)
        out["pair_guard_hits"] = guard.hits
        return out

    report = []
    for shard_trans in (False, True):
        for r in run(P, rank_fn, shard_trans):
            for k in ("opm_inplace", "opm_outofplace", "pwa", "msa_transition"):
                report.append((P, shard_trans, r["rank"], k, r[k]))
                assert r[k]["max_abs_diff"] <= TOL32, (P, shard_trans, r["rank"], k, r[k])
            assert not r["pair_guard_hits"], (r["rank"], r["pair_guard_hits"][:3])
    print("\n".join(str(x) for x in report))


@needs_stack
@pytest.mark.parametrize("pairstack", PAIRSTACK_MODES)
@pytest.mark.parametrize("P", P_CASES)
def test_msa_module_stack_vs_stock(P, pairstack, monkeypatch):
    """The whole ``MSAModuleStack`` (2 blocks: OPM, PWA, transition and the pair stack via ``pairstack.pair_block_rows``) vs stock, per rank rows."""
    from openfold3_opt.tp_rowpair import msa as MSA
    monkeypatch.setenv("ROWPAIR_VERBOSE", "0")
    use_pairstack(monkeypatch, pairstack)
    monkeypatch.setenv("ROWPAIR_MSA_TRANS_SHARD", "0")
    stack = build_msa_stack()
    m0, z0, msa_mask, pair_mask = msa_inputs()
    N, C_z = z0.shape[-2], z0.shape[-1]
    with torch.no_grad():
        z_ref = stack(m0.clone(), z0.clone(), msa_mask=msa_mask, pair_mask=pair_mask, chunk_size=CHUNK, transition_ckpt_chunk_size=None,
                      inplace_safe=True, _mask_trans=True, **KW)

    def rank_fn(rank, P_):
        lay = layout(N, P_, rank)
        z_loc = z0[:, lay.r0:lay.r1].clone().contiguous()
        pm_loc = pair_mask[:, lay.r0:lay.r1].contiguous()
        with torch.no_grad(), PairGuard(N, C_z) as guard:
            z_out = MSA.msa_module_rows(stack, m0.clone(), z_loc, msa_mask=msa_mask, pair_mask_loc=pm_loc, lay=lay, chunk_size=CHUNK,
                                        transition_ckpt_chunk_size=None, inplace_safe=True, _mask_trans=True, **KW)
        return {"rank": rank, "z": diff(z_out, z_ref[:, lay.r0:lay.r1]), "pair_guard_hits": guard.hits if pairstack == "sharded" else []}

    for r in run(P, rank_fn):
        print(P, pairstack, r["rank"], "msa_module", r["z"])
        assert r["z"]["max_abs_diff"] <= TOL32, (P, r)
        assert not r["pair_guard_hits"], (r["rank"], r["pair_guard_hits"][:3])


@needs_stack
@pytest.mark.parametrize("P", P_CASES)
def test_msa_embedder_host_hook_vs_stock(P, monkeypatch):
    """The MSA host placement (``msa.MSA_HOST`` = rank0): the patched ``MSAModuleEmbedder.forward`` returns the stock ``(m, msa_mask)`` bitwise on every
    rank (same seeds), leaves the RNG stream where stock leaves it (next draw equal), and ranks > 0 hold ZERO MSA rows on the host.
    Two batches: many valid MSA rows (randperm branch) and fewer valid rows than the subsample (valid + permuted-invalid branch)."""
    from openfold3.core.model.feature_embedders.input_embedders import MSAModuleEmbedder
    from openfold3_opt.tp_rowpair import msa as MSA
    monkeypatch.setenv("ROWPAIR_VERBOSE", "0")
    monkeypatch.setattr(MSA, "FORCE_HOST", True)                # host and device are both cpu here: force the host gather path
    N, S_msa, SUB = 37, 40, 12
    monkeypatch.setenv("ROWPAIR_BCAST_CHUNK_GB", str(3 * N * 34 * 4 / 2**30))   # several broadcast pieces even at this size
    c_m, _c_z, c_s_input = dims()
    emb = randomise_(MSAModuleEmbedder(c_m_feats=34, c_m=c_m, c_s_input=c_s_input, subsample_main_msa=False, subsample_all_msa=True,
                                       min_subsampled_all_msa=SUB, max_subsampled_all_msa=SUB), 11)
    g = torch.Generator().manual_seed(3)

    def make_batch(n_valid):
        msa = torch.nn.functional.one_hot(torch.randint(0, 32, (1, S_msa, N), generator=g), 32).float()
        mask = torch.zeros(1, S_msa, N)
        mask[:, :n_valid] = (torch.rand(1, n_valid, N, generator=g) > 0.2).float()
        mask[:, :n_valid, 0] = 1.0
        return {"msa": msa, "has_deletion": (torch.rand(1, S_msa, N, generator=g) > 0.7).float(), "deletion_value": torch.rand(1, S_msa, N, generator=g),
                "msa_mask": mask, "num_paired_seqs": torch.tensor([0]), "asym_id": torch.ones(1, N)}

    batches = {"randperm": make_batch(30), "fill_invalid": make_batch(8)}
    s_input = torch.randn(1, N, c_s_input, generator=g)
    install_thread_rng()
    try:
        refs = {}
        for name, b in batches.items():
            reseed_thread_rng(100)
            with torch.no_grad():
                m_ref, mask_ref = emb(batch={k: v.clone() for k, v in b.items()}, s_input=s_input)
                refs[name] = (m_ref, mask_ref, torch.randint(0, 10**6, (3,)))

        def rank_fn(rank, P_):
            out = {"rank": rank}
            for name, b in batches.items():
                bb = {k: v.clone() for k, v in b.items()}
                MSA.place_batch_features(bb)
                held = int(bb["msa"].shape[-3])
                reseed_thread_rng(100)
                with torch.no_grad():
                    m, mask = MSA.msa_module_embedder_forward_host(emb, bb, s_input)
                    nxt = torch.randint(0, 10**6, (3,))
                m_ref, mask_ref, nxt_ref = refs[name]
                out[name] = {"m": diff(m, m_ref), "mask": diff(mask.float(), mask_ref.float()), "rng_next_equal": bool(torch.equal(nxt, nxt_ref)),
                             "host_rows_held": held}
            return out

        for r in run(P, rank_fn):
            for name in batches:
                x = r[name]
                print(P, r["rank"], name, x)
                assert x["m"]["bitwise"] and x["mask"]["bitwise"] and x["rng_next_equal"], (P, r["rank"], name, x)
                assert x["host_rows_held"] == (0 if r["rank"] != 0 else S_msa), (P, r["rank"], name, x)
    finally:
        uninstall_thread_rng()


@needs_stack
def test_refusals_by_name():
    """Non-predict MSA-embedder configuration and a batched z shard refuse BY NAME (MsaRefused), never fall back."""
    from openfold3.core.model.feature_embedders.input_embedders import MSAModuleEmbedder
    from openfold3_opt.tp_rowpair import msa as MSA
    saved, MSA.FORCE_HOST = MSA.FORCE_HOST, True
    try:
        emb = MSAModuleEmbedder(c_m_feats=34, c_m=8, c_s_input=5, subsample_main_msa=True, subsample_all_msa=False, min_subsampled_all_msa=2,
                                max_subsampled_all_msa=2).eval()
        b = {"msa": torch.zeros(1, 4, 6, 32), "has_deletion": torch.zeros(1, 4, 6), "deletion_value": torch.zeros(1, 4, 6), "msa_mask": torch.ones(1, 4, 6),
             "num_paired_seqs": torch.tensor([0]), "asym_id": torch.ones(1, 6)}

        def rank_fn(rank, P_):
            try:
                MSA.msa_module_embedder_forward_host(emb, b, torch.zeros(1, 6, 5))
            except MSA.MsaRefused as e:
                return str(e)
            return "no refusal"
        msgs = run(2, rank_fn)
        assert all("subsample_main_msa" in s for s in msgs), msgs
        lay = layout(8, 2, 0)
        with pytest.raises(MSA.MsaRefused):
            MSA._rows2(torch.zeros(2, lay.R, 8, 3), lay)
    finally:
        MSA.FORCE_HOST = saved
