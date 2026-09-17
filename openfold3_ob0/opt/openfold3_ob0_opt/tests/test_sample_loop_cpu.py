"""The sample_loop lever's OpenFold3 binding (openfold3_ob0_opt/sample_loop.py + tp_rowpair.diffusion's draws) on CPU: the plan from the environment
and its refusals by name; the loop's assembly of the rollout's nested outputs (sample-invariant confidence leaves kept once and checked equal,
per-sample leaves concatenated along dim 1, the census line on rank 0's stderr and in the schedule census); the kit surface (flag, lever tables);
and — on P = 2 gloo rank processes against the STOCK openfold3 SampleDiffusion — the chunked roll-out (k = 1 and k = 2 of S = 3 samples, draws
at the S shape sliced per pass) vs the one batched pass from the same seed: positions within 1e-5 (bitwise reported) and the generator's end
state EQUAL to the batched pass's on every rank."""
import json
import sys

import pytest

try:
    import torch
    HAVE_TORCH = True
except Exception:                                             # noqa: BLE001
    HAVE_TORCH = False

from openfold3_ob0_opt import sample_loop as SLB

try:
    from opt_core.mem import sample_loop as _SL  # noqa: F401
    HAVE_SEAM = True
except Exception:                                             # noqa: BLE001
    HAVE_SEAM = False

from openfold3_ob0_opt.tests import test_tp_rowpair_diffusion_cpu as TD    # the tiny stock modules / features / launch helpers (one definition)

needs_seam = pytest.mark.skipif(not (HAVE_TORCH and HAVE_SEAM), reason="needs torch + opt_core>=0.4.3 with mem.sample_loop")
needs_stack = pytest.mark.skipif(not (HAVE_TORCH and HAVE_SEAM and TD.HAVE_OF3 and TD.HAVE_CORE), reason="needs torch + openfold3 + opt_core>=0.4.3 (rowpair + sample_loop)")
S3 = 3


# ================================================================================================================ plan / env / surface
@needs_seam
def test_plan_is_one_sample_per_pass():
    p = SLB.plan_for(5)
    assert (p.chunk, p.source, p.passes, [(c.start, c.stop) for c in p.chunks]) == (1, "default", 5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    assert SLB.plan_for(1).passes == 1 and SLB.CHUNK == 1


def test_kit_surface_names_the_lever():
    from openfold3_ob0_opt import cli, modes, registry, stack
    assert "sample_loop" in stack._PROBES and "tp_shard_s" in stack._PROBES
    from openfold3_ob0_opt.tp_rowpair import env as ENVMAP
    assert "sample_loop" in registry.LEVERS and registry.LEVERS["sample_loop"].env_keys == ()
    assert registry.LEVERS["sample_loop"].marker and SLB.census_line({"samples": 2}).startswith(registry.LEVERS["sample_loop"].marker)
    assert "sample_loop" in modes.LINES[("big", "tp")].levers
    assert not ENVMAP.known("OF3TP_SAMPLE_CHUNK")                                    # no chunk switch: one sample per pass is the line
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["pred", "--mode", "big", "--n_gpu", "2", "--sample_chunk", "2", "--query-json", "q.json", "--output-dir", "o"])


@needs_seam
def test_run_rollout_assembles_nested_outputs_and_emits_the_census(capsys):
    torch.manual_seed(5)
    A, N = 12, 4
    contact = torch.rand(1, N, N)
    recorded = {}

    def diffuse(chunk, draws):
        return draws.draw(lambda: torch.randn(1, S3, A, 3), chunk)

    def heads(x, chunk):
        k = x.shape[1]
        conf = {"plddt": x.abs().sum(-1), "ptm": x[..., 0, 0] * 0 + chunk.start, "pae": torch.zeros(1, k, 0, 0), "contact_probs": contact.clone(),
                "chain_ptm": {"A": x[..., 0, 1]}, "chain_pair_iptm": {}}
        return {"plddt_logits": x.sum(-1, keepdim=True).expand(1, k, A, 5).contiguous(), "experimentally_resolved_logits": x[..., :2] * 1, "_tp_confidence": conf}
    SL = SLB.core()
    logged = []
    out = SLB.run_rollout(SL.plan(S3, 1), diffuse, heads, device=torch.device("cpu"), log=logged.append, rank=0, record_schedule=lambda **kw: recorded.update(kw))
    err = "\n".join(logged)                                                                   # the census line goes through the rank's logger (one emission)
    assert "[sample_loop] samples=3 chunk=1 source=given passes=3 draw_order=stock" in err
    assert recorded["samples"] == 3 and recorded["sample_chunk"] == 1 and recorded["sample_passes"] == 3
    torch.manual_seed(5)
    torch.rand(1, N, N)                                                   # the contact draw above
    ref_x = torch.randn(1, S3, A, 3)
    assert torch.equal(out["atom_positions_predicted"], ref_x)
    assert out["plddt_logits"].shape == (1, S3, A, 5) and out["experimentally_resolved_logits"].shape == (1, S3, A, 2)
    conf = out["_tp_confidence"]
    assert torch.equal(conf["plddt"], ref_x.abs().sum(-1)) and conf["ptm"].flatten().tolist() == [0.0, 1.0, 2.0]
    assert conf["pae"].shape == (1, S3, 0, 0) and torch.equal(conf["contact_probs"], contact) and conf["chain_pair_iptm"] == {}
    assert torch.equal(conf["chain_ptm"]["A"], ref_x[..., 0, 1])
    assert out["_sample_loop"]["sample_passes"] == 3
    from openfold3_ob0_opt import stack
    sys.modules[SLB.__name__] = SLB                                                            # other tests' fresh() (this directory) re-imports the package; the record reads sys.modules
    assert SLB.LAST["sample_passes"] == 3 and stack.lever_applied("sample_loop") is True      # the kit's activation record reads the lever's LAST fields

    def bad_heads(x, chunk):                                              # a "sample-invariant" leaf that changes between passes: refused by name
        h = heads(x, chunk)
        h["_tp_confidence"]["contact_probs"] = contact + chunk.start
        return h
    torch.manual_seed(5)
    with pytest.raises(SL.SampleLoopRefused):
        SLB.run_rollout(SL.plan(S3, 1), diffuse, bad_heads, device=torch.device("cpu"), log=lambda m: None, rank=1)


# ================================================================================================================ P = 2 rank processes: chunked vs batched
def _entry_chunked(N: int, chunk_align: int):
    from openfold3.core.model.structure.diffusion_module import create_noise_schedule
    from opt_core.mem import sample_loop as SL
    from opt_core.mem.rowpair import dist as D
    from opt_core.mem.rowpair.dist import Layout
    from openfold3_ob0_opt.tp_rowpair import diffusion as DIFF
    torch.set_num_threads(1)
    P, r = D.world()
    lay = Layout(N, P, r, align=chunk_align)
    res = {"rank": r, "P": P, "checks": {}, "metrics": {}, "bitwise": {}}
    chk, met, bit = res["checks"], res["metrics"], res["bitwise"]
    sd, ns_kw = TD.build_modules()
    batch, si_input, si_trunk, z = TD.with_sample_dim(TD.make_batch(N), *TD.make_trunk_outputs(N))
    z_loc = z[:, :, lay.r0:lay.r1].contiguous()
    kw = dict(use_conditioning=True, chunk_size=None, _mask_trans=True, **TD.KW)
    with torch.no_grad():
        noise_schedule = create_noise_schedule(no_rollout_steps=TD.STEPS, **ns_kw, dtype=si_input.dtype, device=si_input.device)
        torch.manual_seed(TD.SEED)
        ref = DIFF.sample_diffusion_rows(sd, batch=batch, si_input=si_input, si_trunk=si_trunk, z_loc=z_loc, lay=lay, noise_schedule=noise_schedule, no_rollout_samples=S3, **kw)
        ref_state = torch.get_rng_state()
        torch.manual_seed(TD.SEED)
        stock = sd(batch=batch, si_input=si_input, si_trunk=si_trunk, zij_trunk=z, noise_schedule=noise_schedule, no_rollout_samples=S3, **TD.KW)
        met["rows_vs_stock"] = TD.diff(ref, stock)
        chk["rows_vs_stock_within_1e-5"] = met["rows_vs_stock"]["max_abs_diff"] <= TD.TOL32
        for k in (1, 2):
            torch.manual_seed(TD.SEED)
            seen = []

            def rollout(chunk, draws, _seen=seen):
                _seen.append((chunk.start, chunk.stop))
                return DIFF.sample_diffusion_rows(sd, batch=batch, si_input=si_input, si_trunk=si_trunk, z_loc=z_loc, lay=lay, noise_schedule=noise_schedule,
                                                 no_rollout_samples=S3, draws=draws, chunk=chunk, **kw)
            out = SL.run(SL.plan(S3, k), rollout, None, sample_dim=1, host=True)
            pos = out.value
            m = TD.diff(pos, ref)
            met[f"chunk{k}_vs_batched"], bit[f"chunk{k}_vs_batched"] = m, m["bitwise"]
            chk[f"chunk{k}_shape"] = list(pos.shape) == list(ref.shape)
            chk[f"chunk{k}_within_1e-5"] = m["max_abs_diff"] <= TD.TOL32
            chk[f"chunk{k}_rng_end_state_equals_batched"] = bool(torch.equal(torch.get_rng_state(), ref_state))
            chk[f"chunk{k}_passes"] = seen == [(c.start, c.stop) for c in SL.plan(S3, k).chunks]
            chk[f"chunk{k}_census_gates"] = bool(out.record["census"]["consistent"]) and bool(out.record["census"]["end_states_equal"])
            met[f"chunk{k}_draws_per_pass"] = out.record["census"]["draws_per_chunk"]
            met[f"chunk{k}_fields"] = out.fields()
    res["ok"] = all(chk.values())
    import torch.distributed as tdist
    allres = [None] * P
    tdist.all_gather_object(allres, res)
    return {"P": P, "ranks": allres, "ok": all(x["ok"] for x in allres)}


@needs_stack
def test_mp_chunked_rollout_equals_batched_and_rng_ends_where_the_batched_pass_ends():
    from openfold3_ob0_opt.tests._stubs import pin_for_pickling
    pin_for_pickling(globals())          # fresh() in this directory's other tests drops this module from sys.modules; multiprocessing pickles _entry_chunked by reference
    res = TD._mp(2, _entry_chunked, TD.N_TOK, TD.CHUNK)
    print("RESULT " + json.dumps(res, sort_keys=True, default=str))
    bad = {x["rank"]: [k for k, v in x["checks"].items() if not v] for x in res["ranks"] if not x["ok"]}
    assert not bad, (bad, [(x["rank"], x["metrics"]) for x in res["ranks"]])
