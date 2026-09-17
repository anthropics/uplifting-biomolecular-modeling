"""``model.rollout_rows`` — the diffusion roll-out AND the confidence heads under the sample loop — on CPU rank PROCESSES (gloo, P = 2 and 3;
processes because the sampler draws from the process-global RNG) at S = 1 and S = 2 samples, with the trunk shard's placement the tp line runs
(``ROWPAIR_CONF_PARK_ZTRUNK=1 ROWPAIR_FREE_ZTRUNK=1``: the shard PARKED on the host at roll-out entry, its device storage released, the
diffusion conditioning and every confidence pass reading trunk rows from the plan's source by row block) against the RESIDENT placement (both
levers 0) from the same seed. Held: every output tensor ``torch.equal`` across the two placements (placement is not numerics); the roll-out's
``zij_trunk`` holds the sample dim (``[1, 1, n_loc, N, C]``) exactly as ``OpenFold3.forward`` hands it over, so the parked source serves the
diffusion conditioning's ``embed_fn`` the same 5-D row blocks the resident shard did; the host copy LIVES through the diffusion of every pass
and is dropped ONCE, inside the last confidence pass, with zero restores (asserted on the plan's own log lines in order); the census words.
Stock openfold3 modules (the pinned wheel) at the model's channel widths (``c_s`` 384, ``c_z`` 128, ``c_s_input`` 449; tiny DiT / atom attention), random
weights, N = 40 tokens (one atom each), 2 diffusion steps. Needs ``openfold3`` + opt_core >= 0.5.10 (``heads.ZTrunkPlan``); skipped by name
otherwise."""
import contextlib
import io
import os
import types

import pytest

try:
    import torch
    HAVE_TORCH = True
except Exception:                                             # noqa: BLE001
    HAVE_TORCH = False
try:
    import openfold3.core.model.structure.diffusion_module  # noqa: F401
    import openfold3.core.model.heads.head_modules  # noqa: F401
    HAVE_OF3 = True
except Exception:                                             # noqa: BLE001
    HAVE_OF3 = False
try:
    from opt_core.mem.rowpair import heads as _heads  # noqa: F401
    from opt_core.mem.rowpair import launch as _launch  # noqa: F401
    HAVE_CORE = hasattr(_heads, "ZTrunkPlan")
except Exception:                                             # noqa: BLE001
    HAVE_CORE = False

needs_stack = pytest.mark.skipif(not (HAVE_TORCH and HAVE_OF3 and HAVE_CORE), reason="needs torch + openfold3 + opt_core>=0.5.10 (rowpair.heads.ZTrunkPlan)")
N_TOK, CHUNK, STEPS, SEED = 40, 4, 2, 20260904          # 10 row units of 4: P=2 -> 20/20 rows, P=3 -> 16/12/12
C_TOKEN, C_ATOM, C_ATOM_PAIR, N_QUERY, N_KEY = 64, 32, 8, 8, 16
KW = dict(use_deepspeed_evo_attention=False, use_cueq_triangle_kernels=False, use_triton_triangle_kernels=False, use_lma=False)
PLACEMENTS = {"parked": ("1", "1"), "resident": ("0", "0")}   # -> (ROWPAIR_CONF_PARK_ZTRUNK, ROWPAIR_FREE_ZTRUNK); parked = the tp line's default


def randomise_(mod, seed, scale=0.1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in mod.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * scale)
    return mod.eval()


def build_model(S: int, seed=SEED):
    """A stand-in for the ``OpenFold3`` module carrying exactly what ``rollout_rows`` reads: ``shared`` / ``config`` (the stock model config with
    2 roll-out steps and ``S`` samples), ``sample_diffusion`` (stock SampleDiffusion around a DiffusionModule at the model's c_s / c_z / c_s_input),
    ``aux_heads`` (stock AuxiliaryHeadsAllAtom), the mode's memory settings, no offload."""
    import copy
    from ml_collections import ConfigDict
    from openfold3.core.model.heads.head_modules import AuxiliaryHeadsAllAtom
    from openfold3.core.model.structure.diffusion_module import DiffusionModule, SampleDiffusion
    from openfold3.projects.of3_all_atom.config.model_config import model_config
    cfg = ConfigDict(copy.deepcopy(model_config).to_dict())
    sh = cfg.architecture.shared
    sh.diffusion.no_full_rollout_steps, sh.diffusion.no_full_rollout_samples = STEPS, int(S)
    c_s, c_z, c_s_input = int(sh.c_s), int(sh.c_z), int(sh.c_s_input)
    d = cfg.architecture.diffusion_module.to_dict()
    d["diffusion_module"].update(c_s=c_s, c_token=C_TOKEN)
    d["diffusion_conditioning"].update(c_s_input=c_s_input, c_s=c_s, c_z=c_z, tune_chunk_size=False)
    d["atom_attn_enc"].update(c_s=c_s, c_z=c_z, c_atom=C_ATOM, c_atom_pair=C_ATOM_PAIR, c_token=C_TOKEN, c_hidden=C_ATOM // 4, no_heads=4, no_blocks=1,
                              n_query=N_QUERY, n_key=N_KEY, blocks_per_ckpt=None, ckpt_intermediate_steps=False)
    d["diffusion_transformer"].update(c_a=C_TOKEN, c_s=c_s, c_z=c_z, c_hidden=C_TOKEN // 4, no_heads=4, no_blocks=2, blocks_per_ckpt=None)
    d["atom_attn_dec"].update(c_atom=C_ATOM, c_atom_pair=C_ATOM_PAIR, c_token=C_TOKEN, c_hidden=C_ATOM // 4, no_heads=4, no_blocks=1, n_query=N_QUERY, n_key=N_KEY,
                              blocks_per_ckpt=None)
    torch.manual_seed(seed)
    dm = randomise_(DiffusionModule(config=ConfigDict(d)), seed)
    sd = SampleDiffusion(**dict(cfg.architecture.sample_diffusion.to_dict()), diffusion_module=dm).eval()
    ah = randomise_(AuxiliaryHeadsAllAtom(cfg.architecture.heads), seed + 1, 0.15)
    stk = ah.pairformer_embedding.pairformer_stack
    stk.tune_chunk_size = False
    stk.chunk_size_tuner = None
    mem = types.SimpleNamespace(chunk_size=4, **KW)
    model = types.SimpleNamespace(training=False, shared=sh, config=cfg, sample_diffusion=sd, aux_heads=ah, clear_autocast_cache=lambda: None,
                                  _get_mode_mem_settings=lambda: mem, _do_inference_offload=lambda seq_len, module_name: False)
    return model, (c_s, c_z, c_s_input)


def make_inputs(N, dims, seed=SEED):
    """The roll-out's inputs as ``OpenFold3.forward`` hands them over (``model.py:656-664``: the sample dim on every tensor): the batch of one query
    (3 chains, the third a 4-token ligand; every token one atom), ``si_input [1, 1, N, c_s_input]``, ``si_trunk [1, 1, N, c_s]``, ``z [1, 1, N, N, c_z]``."""
    c_s, c_z, c_s_input = dims
    g = torch.Generator().manual_seed(seed + 3)
    n1, n2 = N // 2, N - 4
    asym = torch.cat([torch.full((n1,), 1), torch.full((n2 - n1,), 2), torch.full((N - n2,), 3)]).float()
    residue_index = torch.cat([torch.arange(n1), torch.arange(n2 - n1), torch.arange(N - n2)]).float()
    is_ligand = (asym == 3)
    b = {"token_mask": torch.ones(N), "atom_mask": torch.ones(N), "atom_to_token_index": torch.arange(N), "num_atoms_per_token": torch.ones(N, dtype=torch.long),
         "start_atom_index": torch.arange(N), "asym_id": asym, "entity_id": (asym == 3).float(), "sym_id": torch.zeros(N), "residue_index": residue_index,
         "token_index": torch.arange(N).float(), "restype": torch.nn.functional.one_hot(torch.randint(0, 20, (N,), generator=g), 32).float(),
         "is_protein": (~is_ligand).long(), "is_rna": torch.zeros(N, dtype=torch.long), "is_dna": torch.zeros(N, dtype=torch.long), "is_ligand": is_ligand.long(),
         "is_atomized": torch.ones(N, dtype=torch.long),
         "ref_pos": torch.randn((N, 3), generator=g) * 3.0, "ref_mask": torch.ones(N), "ref_charge": torch.randint(-1, 2, (N,), generator=g).float(),
         "ref_element": torch.nn.functional.one_hot(torch.randint(0, 119, (N,), generator=g), 119).float(),
         "ref_atom_name_chars": torch.nn.functional.one_hot(torch.randint(0, 64, (N, 4), generator=g), 64).float(), "ref_space_uid": torch.arange(N).float()}
    batch = {k: v[None, None] for k, v in b.items()}
    batch["atom_array"] = [None]
    return (batch, torch.randn((1, 1, N, c_s_input), generator=g), torch.randn((1, 1, N, c_s), generator=g), torch.randn((1, 1, N, N, c_z), generator=g) * 0.5)


def tensors_of(tree, pfx=""):
    """``{dotted key: tensor}`` over a nested output dict (lists / non-tensors skipped)."""
    out = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            out.update(tensors_of(v, f"{pfx}{k}."))
    elif HAVE_TORCH and torch.is_tensor(tree):
        out[pfx.rstrip(".")] = tree
    return out


def _entry(N: int, S: int):
    from opt_core.mem.rowpair import dist as D, evidence as EV
    from opt_core.mem.rowpair.dist import Layout
    from openfold3_ob0_opt.tp_rowpair import core as C, model as M
    torch.set_num_threads(1)
    P, r = D.world()
    lay = Layout(N, P, r, align=CHUNK)
    res = {"rank": r, "P": P, "S": S, "R": int(lay.R), "checks": {}, "facts": {}, "unequal": [], "log": []}
    chk, facts = res["checks"], res["facts"]
    model, dims = build_model(S)
    model._tp_layout = lay
    batch, si_input, si_trunk, z = make_inputs(N, dims)
    comm = C.comm()
    comm.verbose = True                                                        # the plan's [park] / [conf] lines are this test's evidence

    def run(placement):
        park, free = PLACEMENTS[placement]
        os.environ["ROWPAIR_CONF_PARK_ZTRUNK"], os.environ["ROWPAIR_FREE_ZTRUNK"] = park, free
        zij_trunk = z[:, :, lay.r0:lay.r1].clone().contiguous()               # [1, 1, R, N, c_z]: this rank's shard, owning its storage (as the trunk hands it over)
        torch.manual_seed(SEED)
        err = io.StringIO()
        with torch.no_grad(), contextlib.redirect_stderr(err):
            out = M.rollout_rows(model, batch, si_input, si_trunk, zij_trunk, inplace_safe=True)
        return out, zij_trunk, [ln for ln in err.getvalue().splitlines() if ln.strip()], dict(EV.schedule_fields())

    out_p, zin_p, log_p, sched_p = run("parked")
    out_r, zin_r, log_r, sched_r = run("resident")
    res["log"] = [ln for ln in log_p if any(w in ln for w in ("[park]", "[conf]", "[model]", "[diffusion] rollout done"))][:60]
    # (i) placement is not numerics: every output tensor bit-identical across the two placements
    tp_, tr_ = tensors_of(out_p), tensors_of(out_r)
    chk["same_output_keys"] = set(tp_) == set(tr_)
    for k in sorted(set(tp_) & set(tr_)):
        if k == "zij_trunk":
            continue
        a, b = tp_[k], tr_[k]
        same = a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(torch.nan_to_num(a), torch.nan_to_num(b))) and bool(torch.equal(torch.isnan(a), torch.isnan(b)))
        if not same:
            res["unequal"].append([k, list(a.shape), list(b.shape), float((a.float() - b.float()).abs().max()) if a.shape == b.shape else -1.0])
    chk["outputs_equal_across_placements"] = not res["unequal"] and len(tp_) > 3
    x = out_p["atom_positions_predicted"]
    chk["positions_shape"] = list(x.shape) == [1, S, N, 3] and bool(torch.isfinite(x).all())
    chk["heads_present"] = all(k in out_p for k in ("plddt_logits", "experimentally_resolved_logits", "_tp_confidence"))   # the PAE / PDE logits are consumed row-wise into _tp_confidence (never materialised whole)
    chk["confidence_scores_present"] = isinstance(out_p.get("_tp_confidence"), dict) and (r != 0 or all(k in out_p["_tp_confidence"] for k in ("plddt", "pae", "pde", "ptm", "iptm")))
    # the consumed shard: the parked run hands back the zero-row stand-in, the resident run the shard itself
    chk["parked_zij_trunk_consumed"] = int(out_p["zij_trunk"].shape[-3]) == 0 and int(out_r["zij_trunk"].shape[-3]) == lay.R
    chk["resident_shard_untouched"] = bool(torch.equal(out_r["zij_trunk"], z[:, :, lay.r0:lay.r1]))
    chk["parked_shard_storage_released"] = int(zin_p.untyped_storage().nbytes()) == 0
    # (ii) the host copy's lifetime, from the plan's own lines in order
    idx = {key: [i for i, ln in enumerate(log_p) if pat in ln] for key, pat in
           (("park", "-> parked"), ("entry", "roll-out entry: z_trunk shard parked"), ("diff_done", "[diffusion] rollout done"), ("conf_done", "[model] confidence done"),
            ("drop", "host copy dropped"), ("restored", "restored to"), ("aborted", "roll-out aborted"))}
    facts["idx"] = idx
    passes = S                                                                 # the sample loop's default chunk of 1: one pass per sample
    chk["parked_once_at_entry"] = len(idx["park"]) == 1 and len(idx["entry"]) == 1 and idx["park"][0] < idx["entry"][0]
    chk["one_diffusion_and_confidence_per_pass"] = len(idx["diff_done"]) == passes and len(idx["conf_done"]) == passes
    chk["dropped_once_never_restored_not_aborted"] = len(idx["drop"]) == 1 and not idx["restored"] and not idx["aborted"]
    if chk["dropped_once_never_restored_not_aborted"] and chk["one_diffusion_and_confidence_per_pass"]:
        d = idx["drop"][0]
        chk["host_copy_alive_through_every_diffusion"] = all(i < d for i in idx["diff_done"])
        chk["dropped_inside_the_last_confidence_pass"] = all(i < d for i in idx["conf_done"][:-1]) and idx["conf_done"][-1] > d
    else:
        chk["host_copy_alive_through_every_diffusion"] = chk["dropped_inside_the_last_confidence_pass"] = False
    # census words
    facts["sched_parked"] = {k: sched_p.get(k) for k in ("conf_ztrunk_entry", "conf_ztrunk", "conf_ztrunk_passes", "conf_pairstack", "conf_samples", "sample_loop_passes")}
    facts["sched_resident"] = {k: sched_r.get(k) for k in ("conf_ztrunk_entry", "conf_ztrunk")}
    chk["words_parked"] = sched_p.get("conf_ztrunk_entry") == "parked:host" and str(sched_p.get("conf_ztrunk")) == ",".join(["parked:host"] * passes)
    chk["words_resident"] = sched_r.get("conf_ztrunk_entry") == "resident" and "parked" not in str(sched_r.get("conf_ztrunk"))
    res["ok"] = all(chk.values())
    import torch.distributed as tdist
    allres = [None] * P
    tdist.all_gather_object(allres, res)
    return {"P": P, "S": S, "ranks": allres, "ok": all(x["ok"] for x in allres)}


def _mp(P, entry, *args):
    from openfold3_ob0_opt.tests._stubs import pin_for_pickling
    pin_for_pickling(globals())
    from opt_core.mem.rowpair import launch
    keep = {k: os.environ.get(k) for k in ("ROWPAIR_TEST_DEVICE", "ROWPAIR_VERBOSE", "ROWPAIR_CONF_PARK_ZTRUNK", "ROWPAIR_FREE_ZTRUNK")}
    os.environ["ROWPAIR_TEST_DEVICE"] = "cpu"
    os.environ["ROWPAIR_VERBOSE"] = "1"
    try:
        return launch.run_sharded(P, entry, *args, mode="big", backend="gloo", cpu_ok=True, nccl_timeout_s=180, run_timeout_s=1500)
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@needs_stack
@pytest.mark.parametrize("S", [1, 2])
@pytest.mark.parametrize("P", [2, 3])
def test_rollout_rows_parked_trunk_shard_equals_resident(P, S):
    out = _mp(P, _entry, N_TOK, S)
    for r in out["ranks"]:
        print(f"P={P} S={S} rank {r['rank']} R={r['R']} ok={r['ok']} checks={r['checks']} facts={r['facts']} unequal={r['unequal'][:4]}")
        if not r["ok"]:
            print("\n".join(r["log"]))
    assert out["ok"], [(r["rank"], {k: v for k, v in r["checks"].items() if not v}, r["unequal"][:4], r["facts"]) for r in out["ranks"]]


@needs_stack
@pytest.mark.parametrize("n_samples,pocket,expect_passes,word", [
    (5, False, 5, None),                    # one sample per pass (sample_loop.CHUNK), no pocket features
    (5, True, 1, "one_pass_forced"),        # a pocket-conditioned query cannot be split (restart parents are chosen across samples): ONE batched pass, named
    (1, True, 1, "one_pass"),               # already one pass: named as such
    (5, "absent", 5, None),                 # the pocket features absent from the batch = not a pocket query
])
def test_sample_plan_pocket_query_runs_one_batched_pass(monkeypatch, n_samples, pocket, expect_passes, word):
    """``model.sample_plan``: the sample loop's plan, with a pocket-conditioned query (0.5.0 ``pocket_sampling_enabled``) forced to ONE batched
    pass and NAMED in the schedule census (``pocket_sampling=1 sample_loop_pocket=one_pass_forced|one_pass``) — never a silent schedule change."""
    import torch
    from openfold3_ob0_opt.tp_rowpair import core as C, model as M
    rec = {}
    monkeypatch.setattr(C.seam("evidence"), "record_schedule", lambda **kw: rec.update(kw))
    batch = {"token_mask": torch.ones(1, 8)}
    if pocket != "absent":
        batch["pocket_sampling_enabled"] = torch.tensor([[bool(pocket)]])
    plan = M.sample_plan(n_samples, batch)
    assert plan.passes == expect_passes, (plan, rec)
    assert sum(c.stop - c.start for c in plan.chunks) == n_samples
    assert rec.get("sample_loop_pocket") == word and (rec.get("pocket_sampling") == 1) == (word is not None), rec
