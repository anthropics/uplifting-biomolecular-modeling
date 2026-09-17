"""tp_rowpair.diffusion vs the STOCK openfold3 0.4.1 ``SampleDiffusion`` / ``DiffusionModule`` on CPU: P rank PROCESSES on gloo
(``opt_core.mem.rowpair.launch.run_sharded(P, entry, backend="gloo", cpu_ok=True)`` — processes, not threads: the sampler draws from the
process-global RNG, which every rank must own) run ``diffusion.sample_diffusion_rows`` on their row shard of a tiny stock module (random
weights, zero-inits randomised; N=44 tokens (2 padding), 84 atoms, c_z 16, 2 DiT blocks, 8-query x 16-key atom windows so the band, the invalid keys and
the padded query slots are all exercised, 2 steps x 2 samples, fixed seed) and compare with the stock dense rollout run in the same process
from the same seed. Held: fp32 ``max|diff| <= 1e-5`` on the sampled positions (``torch.equal`` REPORTED), ``z_cond`` rows ``torch.equal`` the
dense conditioning rows (reported; held to 1e-5), the atom-pair term ``plm_z`` bitwise vs stock's ``convert_pair_rep_to_blocks``, the noise
guard refusing a desynchronised rank BY NAME (a real second rollout with rank-dependent seeds), the pair-shape census (no op output on any
rank has two token-sized dims, and the largest op output is smaller than ``N x N x c_z``), the sampler-hook loop with the no-op (Identity) hook
bitwise vs the stock loop, and ``P = 1`` refused by name. Needs ``openfold3`` (the kit's stock wheel, ``pip install --no-deps``) + opt_core >=
0.4.3 (``rowpair.diffusion``); skipped by name otherwise."""
import json
import os
import sys

import pytest

try:
    import torch
    HAVE_TORCH = True
except Exception:                                             # noqa: BLE001
    HAVE_TORCH = False
try:
    import openfold3.core.model.structure.diffusion_module  # noqa: F401
    HAVE_OF3 = True
except Exception:                                             # noqa: BLE001
    HAVE_OF3 = False
try:
    import opt_core.mem.rowpair.diffusion  # noqa: F401
    from opt_core.mem.rowpair import launch as _launch  # noqa: F401
    HAVE_CORE = True
except Exception:                                             # noqa: BLE001
    HAVE_CORE = False

needs_stack = pytest.mark.skipif(not (HAVE_TORCH and HAVE_OF3 and HAVE_CORE), reason="needs torch + openfold3 0.4.1 + opt_core>=0.5.10 (rowpair.diffusion)")
TOL32 = 1e-5
P_CASES = [(2, "auto"), (3, "auto"), (2, "0")]                    # (P, ROWPAIR_DIFF_BIAS_CACHE): the cached-bias and the recompute-per-step paths
N_TOK, CHUNK = 44, 4                                          # 11 row units of 4: P=2 -> 24/20 rows, P=3 -> 16/16/12 (ragged last rank)
STEPS, SAMPLES, SEED = 2, 2, 20260902
C_Z, C_S, C_S_INPUT, C_TOKEN, C_ATOM, C_ATOM_PAIR = 16, 32, 40, 64, 32, 8     # no channel count equals N_TOK (the pair-shape census counts dims == N)
N_QUERY, N_KEY = 8, 16
COND_ROWS_PIN = "4"                                           # ROWPAIR_DIFF_COND_ROWS: the relpos one-hot row slab [rows, N, 139] is the widest transient at c_z=16
KW = dict(use_deepspeed_evo_attention=False, use_cueq_triangle_kernels=False, use_triton_triangle_kernels=False, use_lma=False)


# ================================================================================================================ tiny stock modules + features
def tiny_config():
    """``model_config.architecture.diffusion_module`` (the ConfigDict ``DiffusionModule(config=)`` takes) at tiny dimensions, + the
    ``sample_diffusion`` / ``noise_schedule`` kwargs."""
    import copy
    from ml_collections import ConfigDict
    from openfold3.projects.of3_all_atom.config.model_config import model_config
    arch = copy.deepcopy(model_config).architecture
    d = arch.diffusion_module.to_dict()
    d["diffusion_module"].update(c_s=C_S, c_token=C_TOKEN)
    d["diffusion_conditioning"].update(c_s_input=C_S_INPUT, c_s=C_S, c_z=C_Z, tune_chunk_size=False)
    d["atom_attn_enc"].update(c_s=C_S, c_z=C_Z, c_atom=C_ATOM, c_atom_pair=C_ATOM_PAIR, c_token=C_TOKEN, c_hidden=C_ATOM // 4, no_heads=4, no_blocks=1,
                              n_query=N_QUERY, n_key=N_KEY, blocks_per_ckpt=None, ckpt_intermediate_steps=False)
    d["diffusion_transformer"].update(c_a=C_TOKEN, c_s=C_S, c_z=C_Z, c_hidden=C_TOKEN // 4, no_heads=4, no_blocks=2, blocks_per_ckpt=None)
    d["atom_attn_dec"].update(c_atom=C_ATOM, c_atom_pair=C_ATOM_PAIR, c_token=C_TOKEN, c_hidden=C_ATOM // 4, no_heads=4, no_blocks=1, n_query=N_QUERY, n_key=N_KEY,
                              blocks_per_ckpt=None)
    return ConfigDict(d), dict(arch.sample_diffusion.to_dict()), dict(arch.noise_schedule.to_dict())


def randomise_(mod, seed, scale=0.1):
    """Random weights everywhere (OpenFold3 zero-initialises output projections / gates, which would make the comparison vacuous)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in mod.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * scale)
    return mod.eval()


def build_modules(seed=SEED):
    from openfold3.core.model.structure.diffusion_module import DiffusionModule, SampleDiffusion
    cfg, sd_kw, ns_kw = tiny_config()
    torch.manual_seed(seed)
    dm = randomise_(DiffusionModule(config=cfg), seed)
    sd = SampleDiffusion(**sd_kw, diffusion_module=dm).eval()
    return sd, ns_kw


def make_batch(N=N_TOK, seed=SEED):
    """Synthetic features of one query WITHOUT the sample dim: 2 chains, 1 + (i % 3) atoms per real token, the last 2 tokens padding
    (token_mask 0, no atoms) — 84 atoms for N=44."""
    g = torch.Generator().manual_seed(seed + 1)
    token_mask = torch.ones(N)
    token_mask[-2:] = 0.0                                     # two padding tokens: they own no atoms (OpenFold3's featurizer pads tokens with num_atoms 0)
    napt = torch.tensor([1 + (i % 3) for i in range(N)], dtype=torch.long) * token_mask.long()
    n_atom = int(napt.sum())                                  # 84 for N=44: not a multiple of n_query (padded query slots) and < n_key windows at the ends (invalid keys)
    a2t = torch.repeat_interleave(torch.arange(N), napt)
    atom_mask = torch.ones(n_atom)
    half = N // 2
    asym_id = torch.cat([torch.zeros(half), torch.ones(N - half)])
    residue_index = torch.cat([torch.arange(half), torch.arange(N - half)]).float()
    batch = {
        "token_mask": token_mask, "atom_mask": atom_mask, "atom_to_token_index": a2t, "num_atoms_per_token": napt,             # int64, as featurized
        "asym_id": asym_id, "entity_id": torch.zeros(N), "sym_id": asym_id.clone(), "residue_index": residue_index, "token_index": torch.arange(N).float(),
        "ref_pos": torch.randn((n_atom, 3), generator=g) * 3.0, "ref_mask": torch.ones(n_atom), "ref_charge": torch.randint(-1, 2, (n_atom,), generator=g).float(),
        "ref_element": torch.nn.functional.one_hot(torch.randint(0, 119, (n_atom,), generator=g), 119).float(),
        "ref_atom_name_chars": torch.nn.functional.one_hot(torch.randint(0, 64, (n_atom, 4), generator=g), 64).float(),
        "ref_space_uid": a2t.float(),
    }
    return {k: v.unsqueeze(0) for k, v in batch.items()}       # batch dim 1


def make_trunk_outputs(N=N_TOK, seed=SEED):
    g = torch.Generator().manual_seed(seed + 2)
    return (torch.randn((1, N, C_S_INPUT), generator=g), torch.randn((1, N, C_S), generator=g), torch.randn((1, N, N, C_Z), generator=g))


def with_sample_dim(batch, si_input, si_trunk, z):
    """``model.py:656-664``: every tensor gains the sample dim before the rollout."""
    return {k: v.unsqueeze(1) for k, v in batch.items()}, si_input.unsqueeze(1), si_trunk.unsqueeze(1), z.unsqueeze(1)


def diff(got, ref):
    d = (got.double() - ref.double()).abs().max().item() if got.numel() else 0.0
    return {"max_abs_diff": float(d), "bitwise": bool(torch.equal(got, ref)), "shape": list(got.shape)}


class PairCensus(object):
    """Records, while active, every op output: ``offenders`` = tensors with >= 2 dims equal to N (a whole pair-shaped tensor), ``max_numel`` =
    the largest output (must stay below ``N x N x c_z``: nothing pair-sized is ever materialised on a rank)."""

    def __init__(self, N):
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.utils._pytree import tree_flatten
        self.N, self.offenders, self.max_numel, self.max_shape = int(N), [], 0, None
        cen = self

        class _Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = func(*args, **(kwargs or {}))
                for leaf in tree_flatten(out)[0]:
                    if isinstance(leaf, torch.Tensor):
                        if sum(1 for d in leaf.shape if int(d) == cen.N) >= 2:
                            cen.offenders.append((str(func), tuple(leaf.shape)))
                        n = int(leaf.numel())
                        if n > cen.max_numel:
                            cen.max_numel, cen.max_shape = n, (str(func), tuple(leaf.shape))
                return out
        self._mode = _Mode()

    def __enter__(self):
        self._mode.__enter__()
        return self

    def __exit__(self, *a):
        return self._mode.__exit__(*a)


# ================================================================================================================ the rank entry (one process per rank)
def _entry(N: int, chunk: int):
    from openfold3.core.model.structure.diffusion_module import create_noise_schedule
    from openfold3.core.utils.atom_attention_block_utils import convert_pair_rep_to_blocks
    from opt_core.mem.rowpair import RowpairRefused, dist as D, evidence as EV
    from opt_core.mem.rowpair.dist import Layout
    from openfold3_opt.tp_rowpair import diffusion as DIFF
    torch.set_num_threads(1)
    P, r = D.world()
    lay = Layout(N, P, r, align=chunk)
    res = {"rank": r, "P": P, "N": N, "R": int(lay.R), "rows": [int(lay.r0), int(lay.r1)], "checks": {}, "metrics": {}, "bitwise": {}, "facts": {}}
    chk, met, bit, facts = res["checks"], res["metrics"], res["bitwise"], res["facts"]

    sd, ns_kw = build_modules()
    dm = sd.diffusion_module
    batch, si_input, si_trunk, z = with_sample_dim(make_batch(N), *make_trunk_outputs(N))
    z_loc = z[:, :, lay.r0:lay.r1].contiguous()               # [1, 1, R, N, c_z]: this rank's shard (the trunk hands it over as such)
    with torch.no_grad():
        noise_schedule = create_noise_schedule(no_rollout_steps=STEPS, **ns_kw, dtype=si_input.dtype, device=si_input.device)
        # -------------------------------------------------------------------------------------------- dense references (stock, unpatched instances)
        torch.manual_seed(SEED)
        ref_pos = sd(batch=batch, si_input=si_input, si_trunk=si_trunk, zij_trunk=z, noise_schedule=noise_schedule, no_rollout_samples=SAMPLES, **KW)
        t0 = noise_schedule[0] * (sd.gamma_0 + 1)
        si_dense, zc_dense = dm.diffusion_conditioning(batch=batch, t=t0, si_input=si_input, si_trunk=si_trunk, zij_trunk=z, use_conditioning=True)
        npe = dm.atom_attn_enc.noisy_position_embedder
        plm_dense = convert_pair_rep_to_blocks(batch=batch, zij_trunk=npe.linear_z(npe.layer_norm_z(zc_dense)), n_query=dm.atom_attn_enc.n_query, n_key=dm.atom_attn_enc.n_key)
        chk["modules_stock_after_reference"] = all("forward" not in m.__dict__ for m in (dm, dm.diffusion_conditioning, dm.diffusion_transformer, npe))

        # -------------------------------------------------------------------------------------------- the binding: rollout on the shard, same seed
        torch.manual_seed(SEED)
        with PairCensus(N) as cen:
            pos = DIFF.sample_diffusion_rows(sd, batch=batch, si_input=si_input, si_trunk=si_trunk, z_loc=z_loc, lay=lay, noise_schedule=noise_schedule,
                                             no_rollout_samples=SAMPLES, use_conditioning=True, chunk_size=None, _mask_trans=True, **KW)
            cache = DIFF.build_rollout_cache(dm, batch, z_loc, lay, SAMPLES)           # the per-rollout state on its own, for the row / band checks
        chk["modules_stock_after_rollout"] = all("forward" not in m.__dict__ for m in (dm, dm.diffusion_conditioning, dm.diffusion_transformer, npe))
        m = diff(pos, ref_pos)
        met["positions"], bit["positions"] = m, m["bitwise"]
        chk["positions_shape"] = list(pos.shape) == [1, SAMPLES, int(batch["atom_mask"].shape[-1]), 3]
        chk["positions_within_1e-5"] = m["max_abs_diff"] <= TOL32 and bool(torch.isfinite(pos).all())
        m = diff(cache.z_cond_loc, zc_dense[:, :, lay.r0:lay.r1])
        met["z_cond_rows"], bit["z_cond_rows"] = m, m["bitwise"]
        chk["z_cond_rows_within_1e-5"] = m["max_abs_diff"] <= TOL32
        m = diff(cache.plm_z, plm_dense)
        met["plm_z_band"], bit["plm_z_band"] = m, m["bitwise"]
        chk["plm_z_band_bitwise_given_equal_zcond"] = m["bitwise"] or (not bit["z_cond_rows"] and m["max_abs_diff"] <= TOL32)   # the band is a bit copy of the projected rows
        chk["single_conditioning_bitwise"] = bool(torch.equal(DIFF.conditioning_single(dm.diffusion_conditioning, batch, t0, si_input, si_trunk), si_dense))
        # -------------------------------------------------------------------------------------------- census: nothing [N, N, *] on this rank
        res["pair_shaped_allocations"] = cen.offenders[:5]
        chk["never_whole_pair"] = len(cen.offenders) == 0
        met["max_op_numel"], met["max_op"], met["pair_numel"] = cen.max_numel, cen.max_shape, N * N * C_Z
        chk["max_numel_below_pair"] = cen.max_numel < N * N * C_Z
        sched = dict(EV.schedule_fields())
        facts.update({k: sched.get(k) for k in ("diff_cond_rows", "diff_bias_rows", "diff_q_rows", "diff_band_rows", "diff_bias_cache", "diff_rows_source", "diff_noise_sync",
                                                 "diff_noise_sync_calls", "diff_band_W", "diff_band_extra_rows", "diff_replicated")})
        facts.update({"band_W": int(cache.plan.W), "band_extra_rows": [int(i) for i in cache.plan.extra_rows], "bias_cache": bool(cache.schedule.bias_cache)})
        chk["schedule_recorded"] = str(sched.get("diff_rows_source", "")).find("cond:env") >= 0 and sched.get("diff_noise_sync") == "guard"
        chk["noise_guard_ran_every_step"] = f"guard:" in str(sched.get("diff_noise_sync_calls", "")) and int(str(sched.get("diff_noise_sync_calls")).split("guard:")[1].split(",")[0]) >= STEPS
        del cache

        # -------------------------------------------------------------------------------------------- the noise guard refuses a desynchronised rank BY NAME (real)
        torch.manual_seed(SEED + r)                           # rank-dependent stream: the initial draw differs across ranks
        try:
            DIFF.sample_diffusion_rows(sd, batch=batch, si_input=si_input, si_trunk=si_trunk, z_loc=z_loc, lay=lay, noise_schedule=noise_schedule,
                                       no_rollout_samples=SAMPLES, use_conditioning=True, chunk_size=None, _mask_trans=True, **KW)
            chk["guard_refuses_desynced_rank"] = False
            met["guard_message"] = "no refusal"
        except RowpairRefused as e:
            met["guard_message"] = str(e)[:300]
            chk["guard_refuses_desynced_rank"] = "diffusion.xl_noisy" in str(e)
        chk["modules_stock_after_refusal"] = all("forward" not in m.__dict__ for m in (dm, dm.diffusion_conditioning, dm.diffusion_transformer, npe))

    res["ok"] = all(chk.values())
    import torch.distributed as tdist                          # every rank's verdict reaches rank 0 (run_sharded returns rank 0's value)
    allres = [None] * P
    tdist.all_gather_object(allres, res)
    return {"P": P, "ranks": allres, "rows": [x["R"] for x in allres], "ok": all(x["ok"] for x in allres)}


def _mp(P, entry, *args, bias_cache="auto"):
    from openfold3_opt.tests._stubs import pin_for_pickling
    pin_for_pickling(globals())          # fresh() in other test modules drops this module from sys.modules; multiprocessing pickles _entry by reference
    from opt_core.mem.rowpair import launch
    keep = {k: os.environ.get(k) for k in ("ROWPAIR_TEST_DEVICE", "ROWPAIR_DIFF_COND_ROWS", "ROWPAIR_DIFF_BIAS_CACHE")}
    os.environ["ROWPAIR_TEST_DEVICE"] = "cpu"
    os.environ["ROWPAIR_DIFF_COND_ROWS"] = COND_ROWS_PIN
    os.environ["ROWPAIR_DIFF_BIAS_CACHE"] = bias_cache
    try:
        return launch.run_sharded(P, entry, *args, mode="big", backend="gloo", cpu_ok=True, nccl_timeout_s=180, run_timeout_s=1500)
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ================================================================================================================ tests
@needs_stack
@pytest.mark.parametrize("P,bias_cache", P_CASES)
def test_mp_sample_diffusion_rows_vs_stock(P, bias_cache):
    res = _mp(P, _entry, N_TOK, CHUNK, bias_cache=bias_cache)
    res["bias_cache_env"] = bias_cache
    print("RESULT " + json.dumps(res, sort_keys=True, default=str))
    assert all(x["facts"]["bias_cache"] == (bias_cache != "0") for x in res["ranks"]), [x["facts"]["bias_cache"] for x in res["ranks"]]
    assert len(res["ranks"]) == P and [x["rank"] for x in res["ranks"]] == list(range(P))
    assert len(set(res["rows"])) > 1, res["rows"]                                                  # the ragged layout really is ragged
    bad = {x["rank"]: [k for k, v in x["checks"].items() if not v] for x in res["ranks"] if not x["ok"]}
    assert not bad, (bad, [(x["rank"], x["metrics"], x["pair_shaped_allocations"]) for x in res["ranks"]])


@needs_stack
def test_p1_refused_by_name():
    """At P = 1 nothing of this module runs: the core's statements refuse a one-rank layout by name before any work."""
    from opt_core.mem.rowpair import RowpairRefused
    from opt_core.mem.rowpair.dist import Layout
    from openfold3_opt.tp_rowpair import diffusion as DIFF
    sd, ns_kw = build_modules()
    batch, si_input, si_trunk, z = with_sample_dim(make_batch(N_TOK), *make_trunk_outputs(N_TOK))
    lay = Layout(N_TOK, 1, 0, align=CHUNK)
    from openfold3.core.model.structure.diffusion_module import create_noise_schedule
    ns = create_noise_schedule(no_rollout_steps=STEPS, **ns_kw, dtype=si_input.dtype, device=si_input.device)
    with pytest.raises(RowpairRefused) as ei, torch.no_grad():
        DIFF.sample_diffusion_rows(sd, batch=batch, si_input=si_input, si_trunk=si_trunk, z_loc=z, lay=lay, noise_schedule=ns, no_rollout_samples=SAMPLES, **KW)
    assert "sample_diffusion_rows" in str(ei.value), str(ei.value)
    dm = sd.diffusion_module
    assert all("forward" not in m.__dict__ for m in (dm, dm.diffusion_conditioning, dm.diffusion_transformer, dm.atom_attn_enc.noisy_position_embedder))


@needs_stack
def test_refusals_by_name():
    from openfold3_opt.tp_rowpair import diffusion as DIFF
    from openfold3_opt.tp_rowpair.pairstack import PairstackRefused
    from opt_core.mem.rowpair.dist import Layout
    sd, _ = build_modules()
    lay = Layout(N_TOK, 2, 0, align=CHUNK)
    with pytest.raises(DIFF.DiffusionRefused):
        DIFF.sample_diffusion_rows(sd, batch={}, si_input=None, si_trunk=None, z_loc=None, lay=lay, noise_schedule=None, no_rollout_samples=1, use_conditioning=False, **KW)
    with pytest.raises(PairstackRefused):
        DIFF.sample_diffusion_rows(sd, batch={}, si_input=None, si_trunk=None, z_loc=None, lay=lay, noise_schedule=None, no_rollout_samples=1, use_lma=True)


def test_seams_registered():
    """The statements this binding calls are declared on the adapter's one resolver (``core.SEAMS``)."""
    sys.modules.pop("openfold3_opt.tp_rowpair.diffusion", None)
    try:
        from openfold3_opt.tp_rowpair import core as C, diffusion  # noqa: F401
    except ImportError as e:
        pytest.skip(f"adapter not importable here ({e})")
    for a in ("DiffusionSchedule", "pair_cond_rows", "PairBiasCache", "DiTBlockFns", "diffusion_transformer_sharded", "band_plan", "pair_band_rows", "band_lookup", "sync_replicated"):
        assert a in C.SEAMS["diffusion"], a
    assert "sampler_hook" not in C.SEAMS                                                    # the roll-out has no hook seam


if __name__ == "__main__":                                    # python test_tp_rowpair_diffusion_cpu.py [P]
    P = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    out = _mp(P, _entry, N_TOK, CHUNK)
    print("RESULT " + json.dumps(out, sort_keys=True, default=str))
    sys.exit(0 if out["ok"] else 1)
