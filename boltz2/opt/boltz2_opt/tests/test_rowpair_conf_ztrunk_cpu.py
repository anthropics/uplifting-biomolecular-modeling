"""The trunk pair shard's placement from the roll-out entry through the confidence passes (``rowpair._ztrunk_plan`` binding the core's
``opt_core.mem.rowpair.heads.ZTrunkPlan``; ``rowpair_heads.diffusion_conditioning_rows`` parks at the roll-out entry and reads through
``plan.source()``, ``confidence_rows`` embeds each sample's pass through ``plan.begin(i)`` / ``plan.end(i)``) is PLACEMENT ONLY: on P gloo ranks
(the core's launcher, one process per rank — the driver of ``test_rowpair_heads_cpu.py``) the three forms — ``parked`` (ROWPAIR_CONF_PARK_ZTRUNK=1:
the shard copied to the host and its storage RELEASED before ``z_cond`` is allocated, rows served per block, dropped after the last pass), ``inplace``
(ROWPAIR_FREE_ZTRUNK=1, one sample, no live park: the confidence pair input overwrites the shard) and ``resident`` (both 0) — give conditioned
pair rows, sampled coordinates and every confidence output BIT-IDENTICAL to the resident run at 1 and 2 samples per forward, and the census
words / the released storage / ``dict_out["z"]`` (``rowpair.ConsumedRows``) say what happened. CPU, fp32 (a parked CPU shard is ``parked:host``;
the pinned-host word and the device byte proof are the GPU runs'). Skipped by name without the boltz 2.2.1 model tree.
"""
import os
import types

import pytest


@pytest.fixture(autouse=True)
def _cpu_ranks_only(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    for k in ("ROWPAIR_FREE_ZTRUNK", "ROWPAIR_CONF_PARK_ZTRUNK"):
        monkeypatch.delenv(k, raising=False)
    yield


torch = pytest.importorskip("torch")
pytest.importorskip("boltz.model.modules.confidencev2", reason="the pure-python boltz 2.2.1 model tree is required (pip install --no-deps boltz==2.2.1 einops scipy)")
from opt_core.mem.rowpair.heads import ZTrunkPlan  # noqa: E402  (the pinned core carries the statement; an older core FAILS here, never skips)

from boltz2_opt import rowpair  # noqa: E402
from boltz2_opt import rowpair_heads as RH  # noqa: E402
from boltz2_opt.tests.test_rowpair_heads_cpu import build, dense_heads, _abs, _rel, TOL  # noqa: E402

WORDS = {"park": {"ROWPAIR_CONF_PARK_ZTRUNK": "1", "ROWPAIR_FREE_ZTRUNK": "0"}, "free": {"ROWPAIR_CONF_PARK_ZTRUNK": "0", "ROWPAIR_FREE_ZTRUNK": "1"},
         "both": {"ROWPAIR_CONF_PARK_ZTRUNK": "1", "ROWPAIR_FREE_ZTRUNK": "1"}, "off": {"ROWPAIR_CONF_PARK_ZTRUNK": "0", "ROWPAIR_FREE_ZTRUNK": "0"}}
CONF_KEYS = ("ptm", "iptm", "ligand_iptm", "protein_iptm", "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde", "plddt", "pae", "pde", "plddt_logits", "resolved_logits")


def _flow(m, T, inp, feats, coords, S: int):
    """boltz2.py's order on the rows: conditioning (the roll-out entry) -> the sampler -> the confidence passes (one per sample)."""
    cond = RH.diffusion_conditioning_rows(m, T)
    released_after_cond = int(T.z_loc.untyped_storage().size()) == 0
    sample_kw = dict(s_trunk=inp["s"], s_inputs=inp["s_inputs"], feats=feats, num_sampling_steps=2, atom_mask=feats["atom_pad_mask"],
                     multiplicity=2, max_parallel_samples=None, steering_args=m.steering_args, diffusion_conditioning=cond)
    torch.manual_seed(11)
    struct = RH.sample_sharded(m, T, **sample_kw)
    conf = RH.confidence_rows(m, T, coords[:S].contiguous() if S == 1 else coords, S, True)
    return cond, struct, conf, released_after_cond


def _entry(N: int, S: int, form: str):
    torch.set_num_threads(1)
    os.environ.update(WORDS[form])
    m, feats, inp = build(N)
    with torch.no_grad():
        ref = dense_heads(m, feats, inp, seed=11)
        coords = ref["struct"]["sample_atom_coords"]                                # [2, NA, 3]: the reference coordinates (seams 16-17 isolated from the sampler)
        os.environ[rowpair.ENV_P] = os.environ["ROWPAIR_WORLD"]
        assert rowpair.apply() == ["rowpair_tp"]
        fake = types.ModuleType("fake_boltz2_model_module")
        fake.Boltz2 = type("Boltz2", (), {"forward": lambda self, feats: None})
        rowpair._install_trunk(fake)
        lay = rowpair._layout(N)
        r0, r1 = int(lay.r0), int(lay.r1)
        rowpair._pair_planes_to_host(feats)
        mask = feats["token_pad_mask"]

        def shard():
            return rowpair.TrunkShard(inp["s_inputs"], inp["s"], inp["z"][:, r0:r1].clone(), inp["pd"][:, r0:r1].clone(), mask,
                                      mask[:, :, None] * mask[:, None, :], lay, feats)
        # ---- the resident reference on the rows (both levers off, explicitly)
        T0 = shard()
        T0.zplan = ZTrunkPlan(T0.z_loc, passes=S, free=False, park=False, name="z_trunk")     # explicit kwargs win over the words: the resident form through the same statement
        cond0, struct0, conf0, rel0 = _flow(m, T0, inp, feats, coords, S)
        assert not rel0 and rowpair._ztrunk_close(T0) is T0.z_loc and T0.zplan.words and all(str(w).startswith("resident") for w in T0.zplan.words)
        from opt_core.mem.rowpair import evidence as EV
        EV.reset_schedule()
        # ---- the same flow under the form's words, the plan bound as _forward_sharded binds it
        T = shard()
        z_before = T.z_loc
        T.zplan = rowpair._ztrunk_plan(T, passes=S)
        assert T.zplan is not None and T.zplan.passes == S
        cond, struct, conf, released = _flow(m, T, inp, feats, coords, S)
        zout = rowpair._ztrunk_close(T)
        sched = dict(EV.schedule())
        rec = rowpair.report()["ztrunk"]
    # ---- bit-exactness across forms (placement only)
    equal = {"z_cond_rows": bool(torch.equal(cond["token_trans_bias"].z_loc, cond0["token_trans_bias"].z_loc)),
             "atom_enc_bias": bool(torch.equal(cond["atom_enc_bias"], cond0["atom_enc_bias"])),
             "sample_atom_coords": bool(torch.equal(struct["sample_atom_coords"], struct0["sample_atom_coords"]))}
    for k in CONF_KEYS:
        equal["conf." + k] = bool(torch.equal(conf[k], conf0[k])) and tuple(conf[k].shape) == tuple(conf0[k].shape)
    pc, pc0 = conf["pair_chains_iptm"], conf0["pair_chains_iptm"]
    equal["conf.pair_chains_iptm"] = all(torch.equal(pc[a][b], pc0[a][b]) for a in pc0 for b in pc0[a])
    # ---- and the resident rows equal the stock modules (tolerance class of test_rowpair_heads_cpu; S=1: sample 0 of the run_sequentially reference)
    cr = ref["conf"]
    diffs = {}
    for k in ("ptm", "iptm", "ligand_iptm", "protein_iptm", "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde"):
        diffs["conf." + k] = _abs(conf[k], cr[k][:S])
    diffs["conf.plddt"] = _rel(conf["plddt"], cr["plddt"][:S])
    if lay.rank == 0:
        assert tuple(conf["pae"].shape) == (S, N, N) and conf["pae"].device.type == "cpu"
        diffs["conf.pae"] = _rel(conf["pae"], cr["pae"][:S])
    else:
        assert tuple(conf["pae"].shape) == (S, r1 - r0, N)
        diffs["conf.pae_rows"] = _rel(conf["pae"], cr["pae"][:S, r0:r1])
    # ---- what the plan did, by form
    words, entry = list(rec["words"]), sched.get("conf_ztrunk_entry")
    facts = {"words": words, "entry": entry, "consumed": bool(rec["consumed"]), "released_after_cond": bool(released),
             "storage_now": int(z_before.untyped_storage().size()), "zout": type(zout).__name__, "sched_conf_ztrunk": sched.get("conf_ztrunk"),
             "conf_pairstack": sched.get("conf_pairstack"), "host_gib": rec["host_gib"], "free": rec["free"], "park": rec["park"]}
    if form in ("park", "both"):
        assert entry == "parked:host" and released, facts                            # parked AT THE ROLL-OUT ENTRY: storage released before z_cond was allocated (CPU shard: where=host)
        assert words == ["parked:host"] * S and facts["consumed"] and facts["storage_now"] == 0, facts   # every pass served by the live park; dropped, not restored
        assert isinstance(zout, rowpair.ConsumedRows), facts
    elif form == "free":
        assert str(entry).startswith("resident") and not released, facts             # no park lever: nothing moves at the roll-out entry
        assert words == (["inplace"] if S == 1 else ["resident:free_declined:not_last", "inplace"]), facts   # in place on the LAST pass only (one sample per pass)
        assert facts["consumed"] and isinstance(zout, rowpair.ConsumedRows) and facts["storage_now"] > 0, facts
    else:
        assert entry == "resident" and words == ["resident"] * S and not facts["consumed"] and zout is z_before and facts["storage_now"] > 0, facts
    if isinstance(zout, rowpair.ConsumedRows):
        with pytest.raises(rowpair.Refused, match="consumed by the confidence stage"):
            zout.cpu()
    assert sched.get("conf_ztrunk") == ",".join(words) and int(sched.get("conf_ztrunk_passes")) == S, facts
    assert sched.get("conf_pairstack") == "inplace", facts                            # the confidence Pairformer updates the pass's rows in place (no working copy)
    return {"rank": int(lay.rank), "P": int(lay.P), "S": S, "form": form, "equal": equal, "diffs": diffs, "facts": facts}


@pytest.mark.parametrize("S", [1, 2])
@pytest.mark.parametrize("form", ["park", "free", "both", "off"])
@pytest.mark.parametrize("P,N", [(2, 40), (3, 41)])                                  # (3, 41): ragged rows
def test_ztrunk_forms_are_placement_only(P, N, form, S):
    from opt_core.mem.rowpair import launch
    out, records = launch.run_sharded(P, _entry, N, S, form, mode=launch.MEMORY_MODE, backend="gloo", cpu_ok=True, run_timeout_s=1200, return_records=True)
    print("ZTRUNK", form, S, out["facts"])
    assert out["P"] == P and all(out["equal"].values()), (out["equal"], out["facts"])
    bad = {k: v for k, v in out["diffs"].items() if v > TOL}
    assert not bad, (bad, out["facts"])
    assert all((r.get("ok") if isinstance(r, dict) else r.ok) for r in records), records


def test_the_line_exports_both_words_and_the_lever_line_names_them():
    from boltz2_opt import modes, report, stack
    assert modes.TP_EXPORTS["ROWPAIR_CONF_PARK_ZTRUNK"] == "1" and modes.TP_EXPORTS["ROWPAIR_FREE_ZTRUNK"] == "1"
    env = stack.child_env("big", {"ROWPAIR_FREE_ZTRUNK": "0"}, n_gpu=2)
    assert env["ROWPAIR_CONF_PARK_ZTRUNK"] == "1" and env["ROWPAIR_FREE_ZTRUNK"] == "1"              # the row's words are the row's alone: a caller's copy is stripped
    assert "ROWPAIR_CONF_PARK_ZTRUNK" not in stack.child_env("big", {}, n_gpu=1)
    tr = dict(rowpair.report(), installed=True, schedule={"conf_ztrunk_entry": "parked:host_pinned", "conf_ztrunk": "parked:host_pinned,parked:host_pinned",
                                                          "conf_ztrunk_host_gib": 8.38, "conf_pairstack": "inplace", "park_z_init": "host_pinned", "msa_host": "rank0"})
    state, _why, pairs = report.lever_state("rowpair_tp", {"tp_report": tr}, {"n_gpu": 2})        # the LEVER line's pairs: the placement words as they ACTED (the core's census)
    assert state == "on" and {"conf_ztrunk_entry": "parked:host_pinned", "conf_ztrunk": "parked:host_pinned,parked:host_pinned", "conf_pairstack": "inplace",
                              "park_z_init": "host_pinned", "msa_host": "rank0", "conf_ztrunk_host_gib": 8.38}.items() <= dict(pairs).items(), pairs
    c = rowpair.ConsumedRows("z", "parked:host_pinned", (1, 5, 10, 4))
    assert "parked:host_pinned" in repr(c)
    with pytest.raises(rowpair.Refused, match="ROWPAIR_FREE_ZTRUNK=0 ROWPAIR_CONF_PARK_ZTRUNK=0"):
        c.cpu()
