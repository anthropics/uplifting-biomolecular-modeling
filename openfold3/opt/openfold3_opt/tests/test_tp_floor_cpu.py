"""The tp line's sharding floor (tp_rowpair/floor.py) — ONE predicate for the launcher (polymer residues) and the ranks (featurised tokens):
every rank of the run's row partition (opt_core's dist.row_parts at the pinned chunk) owns at least one full block of floor.BLOCK_UNIT (16) rows,
else the query is served unsharded by name (small_n_unsharded). CPU only."""
import io
import json
import os

import pytest

torch = pytest.importorskip("torch", reason="opt_core's row partition (dist.row_parts) lives in a torch module")
from openfold3_opt import tp                                          # noqa: E402
from openfold3_opt.tp_rowpair import floor as FLOOR                   # noqa: E402


def _query(tmp_path, n_res: int, name="q"):
    p = tmp_path / f"{name}_{n_res}.json"
    p.write_text(json.dumps({"queries": {name: {"chains": [{"molecule_type": "protein", "chain_ids": ["A"], "sequence": "A" * n_res}]}}}))
    return str(p)


def plan(tmp_path, n_res: int, world: int):
    """The launcher's decision for a query of `n_res` polymer residues at `world` ranks: (chunk, verdict)."""
    chunk, _how = tp.chunk_of_run(_query(tmp_path, n_res), environ={}, world=world)
    return chunk, FLOOR.verdict(n_res, world, chunk)


def test_the_incident_n33_p2_is_below_the_floor():
    """entity_ligand_smiles_aspirin_on_1l2y: 20 residues + 13 ligand atoms = 33 tokens at P=2. The rank tier (featurised count, pinned chunk 32 or 16)
    finds rank 1 with 1 row (32 | 1) -> below the floor; the launcher tier already stops at the 20 residues (16 | 4)."""
    for align in (32, 16):
        v = FLOOR.verdict(33, 2, align)
        assert not v["ok"] and v["parts"] == [(0, 32), (32, 33)] and v["min_rank"] == 1 and v["min_rows"] == 1, v
    v = FLOOR.verdict(20, 2, 16)
    assert not v["ok"] and v["rows"] == [16, 4], v
    assert "rank 1 would own 1 rows, fewer than one full block of 16" in FLOOR.words(FLOOR.verdict(33, 2, 32))


def test_n61_p2_stays_sharded_as_sealed(tmp_path):
    """template_gapped_1hdd_protein (61 tokens) at P=2: chunk 32, rows 32 | 29 — sharded (29 >= 16), the sealed row's layout unchanged."""
    chunk, v = plan(tmp_path, 61, 2)
    assert chunk == 32 and v["ok"] and v["parts"] == [(0, 32), (32, 61)], (chunk, v)


def test_rows_of_record_keep_their_plan(tmp_path):
    """The sealed rows' (residues, P) -> chunk and partition are what they ran with (their digests stay reproducible)."""
    for n, world, chunk_expected in ((400, 2, 128), (800, 2, 128), (1200, 2, 128), (2000, 2, 128), (1200, 8, 128), (2000, 8, 128), (6000, 4, 32), (6000, 8, 32), (16, 2, 16)):
        chunk, v = plan(tmp_path, n, world)
        assert chunk == chunk_expected, (n, world, chunk)
        assert v["ok"] == (n != 16), (n, world, v)                        # tiny16 (the sealed small-N case) stays unsharded; the rest sharded


def test_smallest_sharded_query_per_world(tmp_path):
    """The smallest polymer-residue count that stays sharded: 32 / 64 / 128 at P = 2 / 4 / 8 (P full blocks of 16 at the floor chunk); the plan's
    chunk is halved before a query is called below the floor, so from 16 x (2P - 1) residues up EVERY query is sharded (below that only the
    counts whose 16-row blocks split evenly are)."""
    smallest = {}
    for world in (2, 4, 8):
        oks = {n: plan(tmp_path, n, world)[1]["ok"] for n in range(1, 700)}
        smallest[world] = min(n for n, ok in oks.items() if ok)
        assert all(ok for n, ok in oks.items() if n >= 16 * (2 * world - 1)), (world, [n for n, ok in oks.items() if not ok and n >= 16 * (2 * world - 1)][:8])
    assert smallest == {2: 32, 4: 64, 8: 128}, smallest


def test_property_sharded_means_every_rank_owns_a_full_block(tmp_path):
    """Over N in [1, 300] x P in {2, 4, 8}: sharded => every rank owns >= 16 rows, boundaries on the chunk grid, rows sum to N; else below the floor
    (the launcher serves it unsharded by name). Also on the rank tier's inputs (any align of the plan: 16 / 32 / 64 / 128)."""
    for world in (2, 4, 8):
        for n in range(1, 301):
            chunk, v = plan(tmp_path, n, world)
            assert sum(v["rows"]) == n and len(v["rows"]) == world, (n, world, v)
            if v["ok"]:
                assert min(v["rows"]) >= FLOOR.BLOCK_UNIT and all(s % chunk == 0 for s, _ in v["parts"]), (n, world, chunk, v)
            else:
                assert min(v["rows"]) < FLOOR.BLOCK_UNIT, (n, world, chunk, v)
            for align in (16, 32, 64, 128):                             # the rank tier: the featurised count against the pinned chunk, whatever the plan pinned
                w = FLOOR.verdict(n, world, align)
                assert w["ok"] == (min(w["rows"]) >= FLOOR.BLOCK_UNIT) and sum(w["rows"]) == n


def test_rank_tier_exits_before_any_collective_on_the_featurised_count():
    """floor.check_featurised on a featurised batch: below the floor -> one named line + SystemExit(EXIT_BELOW_FLOOR) (the launcher serves the
    query unsharded on that exit); at N=61 / chunk 32 -> the verdict, no exit. The batch is read, never modified."""
    lines = []
    batch = {"token_mask": torch.ones(1, 33), "template_pseudo_beta_mask": torch.zeros(1, 4, 33)}      # 33 tokens (4 lazy-dummy template slots + a ligand: only the count matters)
    with pytest.raises(SystemExit) as ei:
        FLOOR.check_featurised(batch, 2, 32, log=lines.append)
    assert ei.value.code == FLOOR.EXIT_BELOW_FLOOR == 70 and len(lines) == 1
    assert lines[0].startswith("below the sharding floor: the featurised query's 33 tokens at P=2") and "exits 70 before any collective" in lines[0] and FLOOR.REASON in lines[0]
    assert FLOOR.check_featurised(batch := {"token_mask": torch.ones(1, 61)}, 2, 32)["parts"] == [(0, 32), (32, 61)]
    src = open(os.path.join(os.path.dirname(tp.__file__), "tp_rowpair", "model.py")).read()
    body = src[src.index("def transfer_batch_to_device_patched"):]
    body = body[:body.index("\ndef ", 10)]
    assert body.index("FLOOR.check_featurised(") < body.index("ensure_group()"), "the floor is checked before the rank joins the group"
    assert body.index('if "token_mask" in batch:') < body.index("FLOOR.check_featurised("), "a batch upstream failed to featurise (no token_mask) is not this check's"


def test_launcher_serves_unsharded_below_the_floor_and_on_a_rank_floor_exit(tmp_path, monkeypatch, capsys):
    """(1) 20 polymer residues at P=2 (16 | 4): served unsharded through the stock route before any rank starts — `FALLBACK small_n_unsharded=1`
    with the floor sentence, census word, exit line, rc = the stock call's. (2) A sharded plan whose ranks exit 70 (the featurised count was
    below the floor): the launcher serves the request unsharded the same way."""
    import argparse
    from openfold3_opt import cli
    calls = {}
    monkeypatch.delenv(tp.ENV_CHUNK, raising=False)
    monkeypatch.setattr(cli, "run_stock_subprocess", lambda ns, home, ckpt, query_json, **k: calls.setdefault("stock", (query_json, k, ns)) and 0)
    monkeypatch.setattr(tp, "gpu_census", lambda: pytest.fail("the floor comes before any GPU census"))
    monkeypatch.setattr(tp, "spawn_ranks", lambda *a_, **k: pytest.fail("no rank is spawned for an unsharded request"))
    HOME = os.path.abspath(os.path.join(os.path.dirname(tp.__file__), "..", ".."))
    a = argparse.Namespace(mode="big", n_gpu="2", query_json=_query(tmp_path, 20, "trpcage"), output_dir=str(tmp_path / "out"), det=1, runner_yaml=None,
                           upstream_fix=None)
    tp._LAST_CENSUS = {}
    rc = tp.launch(a, HOME, str(tmp_path / "w.pt"), {"path": "w.pt"}, 1, 2, (0, 1, 2, 3))
    err = capsys.readouterr().err
    assert rc == 0 and calls["stock"][2].mode == "off", (rc, calls, err)
    assert "FALLBACK small_n_unsharded=1: the query's 20 polymer residues at P=2 in row blocks aligned to the chunk 16 give rank rows [16, 4]" in err, err
    assert "fallbacks=tp:small_n_unsharded=1" in err and "exit mode=big line=tp n_gpu=2 served=unsharded fallbacks=tp:small_n_unsharded=1 -> 0" in err, err
    assert tp.fallbacks() == {"small_n_unsharded": 1}
    # (2) the rank floor exit: a fake pair of rank processes that exit 70
    calls.clear(); tp._LAST_CENSUS = {}
    a.query_json, a.output_dir = _query(tmp_path, 60, "sixty"), str(tmp_path / "out2")     # 60 residues: chunk 32, rows 32 | 28 -> the plan shards it

    class _P:
        def __init__(self, rc): self._rc, self.pid = rc, 4242
        def poll(self): return self._rc
        def wait(self, timeout=None): return self._rc
        def terminate(self): pass
        def kill(self): pass

    class _T:
        def join(self, timeout=None): pass

    def fake_spawn(argv_of_rank, out_dir, world, gpus, port, chunk, base_env=None):
        os.makedirs(os.path.join(out_dir, tp.RANK_DIR), exist_ok=True)
        return [{"rank": r, "proc": _P(FLOOR.EXIT_BELOW_FLOOR), "rc": None, "started": 0.0, "ended": None, "tee": _T(), "sink": io.BytesIO(), "gpu": str(r),
                 "log": os.path.join(out_dir, tp.RANK_DIR, f"rank{r}.log")} for r in range(world)]

    monkeypatch.setattr(tp, "gpu_census", lambda: ["0", "1"])
    monkeypatch.setattr(tp, "card_memory", lambda: None)
    monkeypatch.setattr(tp, "spawn_ranks", fake_spawn)
    monkeypatch.setattr(tp, "pinned_yaml", lambda base, chunk, yml, **k: {"chunk": chunk})
    monkeypatch.setattr(cli, "row_yaml", lambda home, ln, **k: str(tmp_path / "member.yml"))
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)                 # the launcher names no seed: the ranks get the default one (opt_core rankdata)
    monkeypatch.delenv("OF3TP_DATA_FORM", raising=False)                # the default data form (rank0_bcast) on the launch line
    rc = tp.launch(a, HOME, str(tmp_path / "w.pt"), {"path": "w.pt"}, 1, 2, (0, 1, 2, 3))
    err = capsys.readouterr().err
    assert rc == 0 and calls["stock"][2].mode == "off", (rc, calls, err)
    assert " hashseed=0 source=default ranks=2 data_form=rank0_bcast yaml=" in err, err     # the launch line's hash-seed and data-form census words
    assert "ranks 0,1 exited 70: the featurised query is below the sharding floor" in err and "FALLBACK small_n_unsharded=1: the ranks' featurised token count is below the sharding floor at the pinned chunk 32" in err, err
    assert "served=unsharded fallbacks=tp:small_n_unsharded=1 -> 0" in err
    tp._LAST_CENSUS = {}


def test_templates_are_judged_in_the_launcher_from_rank0s_record(tmp_path, monkeypatch, capsys):
    """The template guard on the sharded route: the launcher took the declaration (cli: TEMPLATES DECLARED) and judges it against rank 0's
    featurised record (its manifest), printing ONE `TEMPLATES FEATURISED` line per templated query that kept real slots — or `TEMPLATES DROPPED`
    + exit 5 (unless --allow-template-drop) + the census word when it lost them. Byte-identical text: cli.templates_guard prints it."""
    import argparse
    from openfold3_opt import templ_census
    q = tmp_path / "q.json"
    (tmp_path / "t.a3m").write_text(">x\nAAA\n")
    q.write_text(json.dumps({"queries": {"hdd": {"chains": [{"molecule_type": "protein", "chain_ids": ["C"], "sequence": "A" * 61,
                                                                "template_alignment_file_path": str(tmp_path / "t.a3m"), "template_entry_chain_ids": ["1hdd_C"]}]}}}))
    templ_census.record_file(str(q))
    declared_err = capsys.readouterr().err
    assert "TEMPLATES DECLARED" in declared_err and "chains_templated=1" in declared_err, declared_err
    a = argparse.Namespace(allow_template_drop=False)
    man = {"templates": {"featurised": {"hdd": {"slots": 4, "real_slots": 1, "key": "template_pseudo_beta_mask"}}}}
    block = {"fallbacks": {}}
    code, reason, guard = tp.judge_templates(a, man, 0, "ok", block)
    err = capsys.readouterr().err
    assert code == 0 and err.count("TEMPLATES FEATURISED query=hdd real_slots=1/4") == 1 and "DROPPED" not in err and block["fallbacks"] == {}, (code, err)
    assert guard["judged"] and not guard["dropped"]
    man0 = {"templates": {"featurised": {"hdd": {"slots": 4, "real_slots": 0, "key": "template_pseudo_beta_mask"}}}}
    code, reason, guard = tp.judge_templates(a, man0, 0, "ok", block)
    err = capsys.readouterr().err
    assert code == 5 and err.count("TEMPLATES DROPPED: query=hdd") == 1 and block["fallbacks"] == {"templates_dropped": 1} and "fallbacks=templates_dropped" in reason, (code, reason, err)
    code, reason, guard = tp.judge_templates(argparse.Namespace(allow_template_drop=True), man0, 0, "ok", {"fallbacks": {}})
    assert code == 0 and "accepted by --allow-template-drop" in reason
    code, reason, guard = tp.judge_templates(a, {"templates": {}}, 1, "runner rc 1", {"fallbacks": {}})     # a failed run is not judged (the exit rule's first clause)
    assert code == 1 and not guard["judged"]
