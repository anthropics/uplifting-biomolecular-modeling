"""design: the request copy (what is composed, what passes verbatim), the driver pass with its evidence read-back, partial / failed
verdicts, the ready line and the manifest, the dry run, the refusals before anything runs."""
import pytest
import json
import os
import re

import yaml

from genie3_opt import cli, registry, design, det, modes, report, stack, warm
from genie3_opt.tests import _stubs


def test_compose_request_sets_only_the_run_parameters(tmp_path, monkeypatch):
    b = _stubs.box(str(tmp_path), monkeypatch)
    raw = design.load_request(_stubs.request(str(tmp_path), n_sample=3))
    req, composed = design.compose_request(raw, str(tmp_path / "out"), b["weights"], det_level=1)
    assert req["paths"]["rootdir"] == str(tmp_path / "out") and req["paths"]["dataset"] == "data/design/binder_design/binderbench"
    assert req["experiment"]["seed"] == det.SEED == 0                                   # --det 1: the recipe's seed when the file sets none
    assert req["generation"]["base"]["checkpoint"] == os.path.join(b["weights"], "checkpoints", "step=600000.ckpt")
    assert req["generation"]["base"]["config"] == os.path.join(b["weights"], "config.yaml")
    assert req["generation"]["dataset"]["n_sample"] == 3 and req["generation"]["sampler"] == {"sampler": {"direction_scale": 0.0}}
    assert set(composed) == {"paths.rootdir", "experiment.seed", "generation.base.checkpoint", "generation.base.config"}
    assert raw["paths"]["rootdir"] == "examples/x"                                       # the input is never edited
    req0, composed0 = design.compose_request(raw, None, None)                             # no --out_dir, no GENIE3_WEIGHTS, --det 0: the file's own keys stand, nothing but the record
    assert req0 == dict(raw, experiment={"name": "t"}) and set(composed0) == {"experiment.seed"} and str(composed0["experiment.seed"]).startswith("absent")
    own = design.load_request(_stubs.request(str(tmp_path), name="own.yaml", extra={"generation.base.checkpoint": "/my/ckpt.ckpt", "generation.base.config": "/my/config.yaml"}))
    req1, composed1 = design.compose_request(own, str(tmp_path / "o1"), b["weights"])    # the request names its own weights: they stand, GENIE3_WEIGHTS composes nothing
    assert req1["generation"]["base"] == {"checkpoint": "/my/ckpt.ckpt", "config": "/my/config.yaml"} and "generation.base.checkpoint" not in composed1
    assert design.weights_paths(own, b["root"]) == ("/my/ckpt.ckpt", "/my/config.yaml") and design.weights_paths(raw, b["root"]) is None
    io = design.load_request(_stubs.request(str(tmp_path), name="io.yaml", extra={"generation.io.outdir": "somewhere"}))
    req2, composed2 = design.compose_request(io, str(tmp_path / "o2"), None)             # --out_dir is paths.rootdir, the one output directory: a disagreeing generation.io.outdir is removed, recorded
    assert "outdir" not in req2["generation"]["io"] and "generation.io.outdir" in composed2
    assert design.request_out_dir(io, "/ckout") == "/ckout/somewhere" and design.request_out_dir(raw, "/ckout") == "/ckout/examples/x" and design.request_out_dir({"generation": {}}, "/r") is None


def test_compose_request_seed_precedence_and_overrides(tmp_path, monkeypatch):
    b = _stubs.box(str(tmp_path), monkeypatch)
    raw = design.load_request(_stubs.request(str(tmp_path), seed=7))
    req, _ = design.compose_request(raw, str(tmp_path / "o"), b["weights"])
    assert req["experiment"]["seed"] == 7                                                # the file's seed is kept
    req, c = design.compose_request(raw, str(tmp_path / "o"), b["weights"], seed=11, n_sample=5, selections="03_il7ra,07_h1")
    assert req["experiment"]["seed"] == 11 and c["generation.dataset.n_sample"] == 5 and req["generation"]["dataset"]["selections"] == "03_il7ra,07_h1"
    assert design.request_problems(req, b["root"]) == ["03_il7ra", "07_h1"]


def test_det_default_is_off_and_the_recipe_seeds_only_an_unseeded_request(tmp_path, monkeypatch):
    """--det 0 (the default): experiment.seed exactly as the request file or --seed leaves it (upstream's shipped default is unseeded,
    config/models.py:18); --det 1: the recipe's seed 0 for a request with none — a file's own seed and an explicit --seed stand at either level."""
    b = _stubs.box(str(tmp_path), monkeypatch)
    assert det.DEFAULT_LEVEL == 0 and det.LEVELS == (0, 1)
    raw = design.load_request(_stubs.request(str(tmp_path), n_sample=3))
    req, composed = design.compose_request(raw, str(tmp_path / "o1"), b["weights"])                    # default level: unseeded stays unseeded
    assert "seed" not in req["experiment"] and str(composed["experiment.seed"]).startswith("absent (upstream's default, unseeded")
    req, composed = design.compose_request(raw, str(tmp_path / "o2"), b["weights"], det_level=1)
    assert req["experiment"]["seed"] == 0 and composed["experiment.seed"] == 0
    seeded = design.load_request(_stubs.request(str(tmp_path), seed=7, name="s.yaml"))
    for lvl in (0, 1):
        assert design.compose_request(seeded, None, None, det_level=lvl)[0]["experiment"]["seed"] == 7        # the file's seed is kept at both levels
        assert design.compose_request(seeded, None, None, seed=11, det_level=lvl)[0]["experiment"]["seed"] == 11
        assert design.compose_request(raw, None, None, seed=5, det_level=lvl)[0]["experiment"]["seed"] == 5   # --seed composes at both levels
    assert det.request_seed({}, det_level=0) is None and det.request_seed({}, det_level=1) == 0
    assert det.request_seed({"experiment": {"seed": 99}}, det_level=0) == 99 and det.request_seed({"experiment": {"seed": 99}}, seed_override=5, det_level=0) == 5


def test_only_an_unreadable_request_is_refused(tmp_path, monkeypatch):
    """The package refuses what it cannot read (a missing file, a file without upstream's generation.dataset section) and nothing upstream would
    run: dataset values (n_sample, source, …) pass verbatim to the line, which judges them as upstream does."""
    import pytest
    b = _stubs.box(str(tmp_path), monkeypatch)
    with pytest.raises(design.DesignError):
        design.load_request(_stubs.request(str(tmp_path)) + "-missing")
    p = str(tmp_path / "nods.yaml"); open(p, "w").write("experiment: {name: x}\n")
    with pytest.raises(design.DesignError):
        design.load_request(p)
    raw = design.load_request(_stubs.request(str(tmp_path), n_sample=0, name="z.yaml"))
    req, _ = design.compose_request(raw, str(tmp_path / "o"), b["weights"])
    assert req["generation"]["dataset"]["n_sample"] == 0                                 # passes verbatim


def test_request_problems_lists_the_dataset_when_no_selections(tmp_path, monkeypatch):
    b = _stubs.box(str(tmp_path), monkeypatch)
    raw = design.load_request(_stubs.request(str(tmp_path), selections=None))
    req, _ = design.compose_request(raw, str(tmp_path / "o"), b["weights"])
    probs = design.request_problems(req, b["root"])
    assert "04_pdl1" in probs and "03_il7ra" in probs and "07_h1" in probs and len(probs) >= 7


def test_driver_pass_exact_with_evidence(tmp_path, monkeypatch, capsys):
    b = _stubs.box(str(tmp_path), monkeypatch)
    out = str(tmp_path / "out_exact")
    rc, man = design.run(_stubs.request(str(tmp_path), n_sample=2), out, "exact", tag="t1", det_level=1, batch=8)
    assert rc == 0 and man["status"] == "ok", man
    assert man["mode"] == "exact" and man["batch_size"] == 8 and man["driver_pass"]["rc"] == 0        # the batched capture line at --batch_size 8
    ev = man["driver_pass"]["evidence"]
    assert ev["applied"] == ["L1", "L2", "L4", "L8", "L9", "L11", "L17", "L18", "L19"] and ev["missing"] == [] and ev["forbidden"] == []
    assert man["levers_evidenced"] == ["L1", "L2", "L4", "L8", "L9", "L11", "L17", "L18", "L19"]
    assert man["driver_pass"]["cwd"] == b["root"] and man["driver_pass"]["cmd"][-2:] == ["--timings", os.path.join(out, "timings.json")]
    assert man["driver_pass"]["cmd"][1] == os.path.join(b["tree"], "opt", "genie3_opt", "g3batch.py") and man["driver_pass"]["cmd"][2:6] == ["--config", os.path.join(out, "request.yaml"), "--outdir", out]
    assert man["driver_pass"]["cmd"][6:16] == ["--batch-size", "8", "--cuda-graphs", "--hoist", "--reuse-graphs", "16", "--lean-pair", "--wide-capture", "--alloc", "expandable"]
    T = json.load(open(os.path.join(out, "timings.json")))                                                 # the driver's evidence records
    assert T["patches"] and len(T["patches"]) == 2 and T["hoist"] is True and T["stock_batch"] == 8 and T["graph_cache"]["capacity"] == 16
    assert "--hoist" in T["argv"] and "--tf32" not in T["argv"] and T["lever"]["graph_captures"] == 1 and T["lever"]["hoist_fills"] == 1   # one capture (one batch of 2, one shape); the recipe adds no driver flag
    assert man["outputs"]["n_pdb"] == 2 and sorted(man["outputs"]["problems"]["04_pdl1"]) == ["04_pdl1_0", "04_pdl1_1"]
    assert man["request"]["composed"]["experiment.seed"] == 0 and man["recipe"]["seed"] == 0 and man["recipe"]["det"] == 1 and "declined" not in man and man["shard"] is None
    assert man["request"]["composed"]["generation.dataset.batch_size"] == 8                                    # --batch_size: composed into the request copy (every route reads the key) and the kit line's --batch-size
    assert os.path.isfile(os.path.join(out, "request.yaml")) and os.path.isfile(os.path.join(out, "opt_manifest.json")) and os.path.isfile(os.path.join(out, "design.log"))
    assert stack.status()["levers_applied"] == ["L1", "L2", "L4", "L8", "L9", "L11", "L17", "L18", "L19"] and stack.status()["partial"] is False
    err = capsys.readouterr().err
    assert "[genie3-opt] ACTIVE mode=exact attach=driver tier=1" in err and "--batch-size 8 --cuda-graphs --hoist --reuse-graphs 16" in err
    assert "[genie3-opt] RUN mode=exact problems=1 designs=2 seed=0" in err
    assert "[genie3-opt] EVIDENCE levers=L1,L2,L4,L8,L9,L11,L17,L18,L19 missing=none forbidden=0" in err
    assert "[genie3-opt] LEVER g3batch batch=8 cuda_graphs=1 hoist=1 reuse_graphs=16 pt_chunk=stock compile=none lean_pair=1 wide=1 alloc=expandable tf32=0 policy=genie3_stock_fp32 graph_captures=1 graph_reuses=0 cache_hits=0 cache_misses=1 hoist_fills=1 hoist_anomalies=0 patches=1 trimul=stock batches=1/1 finished=1 capacity=ok" in open(os.path.join(out, "design.log")).read()
    assert "[genie3-opt] ready mode=exact t=" in err and "first=04_pdl1/pdbs/04_pdl1_" in err
    assert report.exit_tally_line().startswith("[genie3-opt] EXIT pid=") and "n_designs=2" in report.exit_tally_line()


def test_batch_is_the_requests_own_key_on_both_routes(tmp_path, monkeypatch, capsys):
    """upstream's generation.dataset.batch_size: a kit line carries the request's value as the driver's --batch-size (the manifest's batch_size;
    upstream's default 1 when neither the file nor --batch_size sets it); the stock line passes the key verbatim (upstream's default 1 when unset)."""
    _stubs.box(str(tmp_path), monkeypatch)
    rc, man = design.run(_stubs.request(str(tmp_path), n_sample=6, batch=4), str(tmp_path / "k4"), "exact")
    assert rc == 0 and man["batch_size"] == 4 and man["driver_pass"]["cmd"][6:8] == ["--batch-size", "4"] and man["driver_pass"]["timings"]["batches"]["sizes"] == [4, 2]
    assert "generation.dataset.batch_size" not in man["request"]["composed"] and yaml_batch(str(tmp_path / "k4")) == 4 and man["activation"]["batch_size"] == 4
    assert "--batch-size 4 " in [l for l in capsys.readouterr().err.splitlines() if l.startswith("[genie3-opt] ACTIVE ")][0]   # the ACTIVE line spells the effective line
    stack.reset_for_tests()
    rc, man = design.run(_stubs.request(str(tmp_path), n_sample=6, batch=4, name="s4.yaml"), str(tmp_path / "s4"), "off")
    assert rc == 0 and man["batch_size"] == 4 and yaml_batch(str(tmp_path / "s4")) == 4
    assert "batch=4 precision=fp32" in capsys.readouterr().err
    stack.reset_for_tests()
    rc, man = design.run(_stubs.request(str(tmp_path), n_sample=2, name="s1.yaml"), str(tmp_path / "s1"), "off")   # unset: upstream's own default, the RUN line says `request`
    assert rc == 0 and man["batch_size"] is None and "batch=request" in capsys.readouterr().err
    stack.reset_for_tests()
    rc, man = design.run(_stubs.request(str(tmp_path), name="k0.yaml", batch=0), str(tmp_path / "k0"), "exact")
    assert rc == 3 and "batch" in man["reason"]


def yaml_batch(out_dir):
    import yaml
    return yaml.safe_load(open(os.path.join(out_dir, "request.yaml")))["generation"]["dataset"].get("batch_size")


def test_unconditional_shapes_capture_once_each(tmp_path, monkeypatch):
    """An unconditional request of two lengths at n_sample 8 and batch 8 = two batches of one shape each (dataset order is length-major): one
    capture per batch shape under graph reuse."""
    _stubs.box(str(tmp_path), monkeypatch)
    out = str(tmp_path / "u")
    rc, man = design.run(_stubs.request(str(tmp_path), n_sample=8, source="unconditional", batch=8), out, "exact")
    assert rc == 0 and man["status"] == "ok" and man["outputs"]["n_pdb"] == 16, man
    T = json.load(open(os.path.join(out, "timings.json")))
    assert len(T["graph_capture_s"]) == 2 and T["graph_cache"] == {"capacity": 16, "hits": 0, "misses": 2, "kept": 2, "captures": 2}
    stack.reset_for_tests()
    out = str(tmp_path / "u2")                                                                                 # n_sample 16 at batch 8: four batches, two shapes -> 2 captures + 2 reuses
    rc, man = design.run(_stubs.request(str(tmp_path), n_sample=16, source="unconditional", name="u2.yaml"), out, "exact", batch=8)
    T = json.load(open(os.path.join(out, "timings.json")))
    assert rc == 0 and T["graph_cache"]["hits"] == 2 and T["graph_cache"]["misses"] == 2 and [b["graph"] for b in T["per_batch"]] == ["capture", "reuse", "capture", "reuse"]


def test_driver_pass_fast_batches(tmp_path, monkeypatch, capsys):
    """fast = the exact line + L12 + L13 + L7: the driver runs with `--trimul fpf`, the KERNELS census line carries the kernel's `engaged` word (L7's
    evidence), the manifest names the provider."""
    _stubs.box(str(tmp_path), monkeypatch)
    rc, man = design.run(_stubs.request(str(tmp_path), n_sample=10, batch=8), str(tmp_path / "out_fast"), "fast")
    assert rc == 0 and man["batch_size"] == 8 and man["driver_pass"]["evidence"]["applied"] == ["L1", "L2", "L4", "L8", "L9", "L11", "L17", "L18", "L19", "L12", "L13", "L7", "L16"]
    assert man["trimul"] == "fpf" and man["levers_declined"] == [] and man["levers_missing"] == [] and man["driver_pass"]["evidence"]["declined"] == {}
    assert man["driver_pass"]["timings"]["batches"]["sizes"] == [8, 2]
    T = json.load(open(os.path.join(str(tmp_path / "out_fast"), "timings.json")))
    assert T["tf32"] is True and T["pt_chunk"]["chunk"] == "design" and T["compile"]["backend"] == "inductor" and "--tf32" in T["argv"] and "--compile" in T["argv"] and T["argv"][T["argv"].index("--pt-chunk") + 1] == "design" and T["argv"][T["argv"].index("--trimul") + 1] == "fpf"
    assert T["trimul"]["word"].startswith("engaged:fpf_trimul_v4@") and T["kernels"]["trimul"] == T["trimul"]["word"] and T["trimul"]["declined"] is None
    log = open(man["driver_pass"]["log"], encoding="utf-8").read()
    assert re.search(registry.LEVERS["L7"].evidence[1], log) and "[genie3-opt] EVIDENCE levers=L1,L2,L4,L8,L9,L11,L17,L18,L19,L12,L13,L7,L16 missing=none forbidden=0" in capsys.readouterr().err


def test_fast_declines_l7_by_name_under_the_kernel_floor(tmp_path, monkeypatch, capsys):
    """A request whose every pair extent is under the kernel's floor (101 tokens): no call the kernel could serve arrived — the census word reads
    `declined:below_min_tokens[…]`, the pass prints `LEVER lever=L7 state=skipped reason=below_min_tokens mode=fast …`, L7 is neither applied nor missing, the pass is
    NOT partial (exit 0) and the manifest lists it under levers_declined. An UNEXPECTED fallback reason still refuses the lever's gate: no evidence,
    the pass is partial (exit 3; --allow-partial records and proceeds)."""
    _stubs.box(str(tmp_path), monkeypatch)
    monkeypatch.setenv("STUB_FAIL", "trimuldeclined")
    rc, man = design.run(_stubs.request(str(tmp_path), n_sample=2), str(tmp_path / "d"), "fast")
    assert rc == 0 and man["status"] == "ok" and man["levers_declined"] == ["L7"] and man["levers_missing"] == [] and "L7" not in man["driver_pass"]["evidence"]["applied"]
    assert man["driver_pass"]["evidence"]["missing"] == [] and man["driver_pass"]["evidence"]["declined"] == {"L7": "below_min_tokens"} and stack.status()["levers_declined"] == ["L7"] and stack.status()["partial"] is False
    err = capsys.readouterr().err
    assert "[genie3-opt] LEVER lever=L7 state=skipped reason=below_min_tokens mode=fast served_by=module_forward" in err and "DECLINED" not in err
    assert "[genie3-opt] EVIDENCE levers=L1,L2,L4,L8,L9,L11,L17,L18,L19,L12,L13,L16 missing=none forbidden=0" in err
    T = json.load(open(os.path.join(str(tmp_path / "d"), "timings.json")))
    assert T["kernels"]["trimul"].startswith("declined:below_min_tokens[served=0,fallback=") and T["trimul"]["declined"] == "below_min_tokens"
    stack.reset_for_tests()
    monkeypatch.setenv("STUB_FAIL", "trimulrefused")
    capsys.readouterr()
    rc, man = design.run(_stubs.request(str(tmp_path), n_sample=2, name="r.yaml"), str(tmp_path / "r"), "fast")   # an UNEXPECTED fallback: the core gate refuses, the kernel could not serve —
    assert rc == 3 and man["status"] == "refused" and man["levers_missing"] == ["L7"] and man["levers_broken"] == [] and man["levers_declined"] == []   # L7 has no evidence and mode fast REFUSES BY NAME
    assert "fallback:" in json.load(open(os.path.join(str(tmp_path / "r"), "timings.json")))["kernels"]["trimul"]    # (a mode is all of its levers, never a subset under its name): exit 3
    err = capsys.readouterr().err
    assert report.levers_refused_line("fast", "L7") in err.splitlines() and "[genie3-opt] EVIDENCE levers=L1,L2,L4,L8,L9,L11,L17,L18,L19,L12,L13,L16 missing=L7 forbidden=0" in err and man["refused"] == "levers could not run: L7"


def test_batch_size_flag_sets_the_request_key_on_every_route(tmp_path, monkeypatch, capsys):
    """`design --batch_size B`: generation.dataset.batch_size for this pass on every mode — a kit line runs `--batch-size B` (the ACTIVE line
    spells it, the RUN line's batch=), the stock route hands upstream the key in the request copy; a batch below 1 is refused by name before
    anything runs (exit 3); without the flag the request's own key stands (upstream's default 1 when the file leaves it unset)."""
    _stubs.box(str(tmp_path), monkeypatch)
    rc, man = design.run(_stubs.request(str(tmp_path), n_sample=4), str(tmp_path / "b"), "exact", batch=2)
    assert rc == 0 and man["batch_size"] == 2 and man["flags"][:2] == ["--batch-size", "2"] and man["request"]["composed"]["generation.dataset.batch_size"] == 2
    assert man["driver_pass"]["timings"]["batches"]["sizes"] == [2, 2] and man["levers_planned"] == ["L1", "L2", "L4", "L8", "L9", "L11", "L17", "L18", "L19"]
    err = capsys.readouterr().err
    assert "--batch-size 2 " in [l for l in err.splitlines() if l.startswith("[genie3-opt] ACTIVE")][0] and " batch=2 " in [l for l in err.splitlines() if l.startswith("[genie3-opt] RUN")][0]
    stack.reset_for_tests()
    rc, man = design.run(_stubs.request(str(tmp_path), n_sample=2, name="o.yaml"), str(tmp_path / "o"), "off", batch=2)
    assert rc == 0 and man["batch_size"] == 2 and "batch_size: 2" in open(os.path.join(str(tmp_path / "o"), "request.yaml"), encoding="utf-8").read()
    stack.reset_for_tests()
    capsys.readouterr()
    rc, man = design.run(_stubs.request(str(tmp_path), n_sample=2, name="z.yaml"), str(tmp_path / "z"), "exact", batch=0)
    assert rc == 3 and man["status"] == "refused" and "the batch size is an integer >= 1" in man["reason"] and not os.path.exists(tmp_path / "z")
    assert capsys.readouterr().err.count("[genie3-opt] NOT ACTIVE: --batch_size 0") == 1
    stack.reset_for_tests()
    rc, man = design.run(_stubs.request(str(tmp_path), n_sample=2, name="d.yaml"), str(tmp_path / "d8"), "exact")
    assert rc == 0 and man["batch_size"] == 1 and man["flags"][:2] == ["--batch-size", "1"] and "generation.dataset.batch_size" not in man["request"]["composed"]


def test_a_lever_that_could_not_run_refuses_the_mode_by_name(tmp_path, monkeypatch, capsys):
    """A mode is ALL of its levers: a planned lever without its evidence could not run on this box, so the mode refuses BY NAME after the pass —
    `NOT ACTIVE: mode=exact refused — lever(s) L2 could not run …`, `levers_missing: [L2]` + `refused` in the manifest, status refused, exit 3 —
    never a subset under the mode's name, and no opt-out (--allow-partial is gone)."""
    _stubs.box(str(tmp_path), monkeypatch)
    monkeypatch.setenv("STUB_FAIL", "nopatch")
    rc, man = design.run(_stubs.request(str(tmp_path)), str(tmp_path / "o"), "exact")
    assert rc == 3 and man["status"] == "refused" and man["driver_pass"]["evidence"]["missing"] == ["L2"] and man["refused"] == "levers could not run: L2"
    assert man["levers_missing"] == ["L2"] and man["levers_broken"] == [] and man["forbidden_lines"] == 0 and "incomplete" not in man and "partial" not in man and "allow_partial" not in man
    assert stack.status()["levers_unavailable"] == ["L2"] and stack.status()["partial"] is True          # the activation report's fields
    err = capsys.readouterr().err
    assert report.levers_refused_line("exact", "L2") == ("[genie3-opt] NOT ACTIVE: mode=exact refused — lever(s) L2 could not run on this box (their LEVER / KERNELS / driver lines "
                                                       "above say why); a mode is all of its levers, never a subset under its name: exit 3 (the files this pass wrote are not mode "
                                                       "exact's; --mode off runs stock)") and report.levers_refused_line("exact", "L2") in err.splitlines()
    saved = json.load(open(tmp_path / "o" / "opt_manifest.json"))
    assert (saved["status"], saved["levers_missing"], saved["forbidden_lines"]) == ("refused", ["L2"], 0)
    stack.reset_for_tests()
    monkeypatch.setenv("STUB_FAIL", "nopatch,short")                                  # short outputs AND a lever that could not run: the refusal stands (3), the short count is recorded beside it
    rc, man = design.run(_stubs.request(str(tmp_path), n_sample=3), str(tmp_path / "e"), "exact")
    assert rc == 3 and man["status"] == "refused" and man["levers_missing"] == ["L2"] and man["incomplete"] == "2/3"


def test_no_allow_partial_switch_on_the_surface(tmp_path, monkeypatch, capsys):
    """The surface carries no --allow-partial and no GENIE3_OPT_ALLOW_PARTIAL: a mode is all of its levers or it refuses by name — there is no opt-out;
    argparse rejects the flag, the package has no reader of the variable."""
    _stubs.box(str(tmp_path), monkeypatch)
    with pytest.raises(SystemExit) as ex:
        cli.main(["design", "--mode", "exact", "--input", _stubs.request(str(tmp_path)), "--out_dir", str(tmp_path / "a"), "--allow-partial"])
    assert ex.value.code == 2
    with pytest.raises(SystemExit):
        cli.main(["warm", "--mode", "exact", "--out_dir", str(tmp_path / "w"), "--allow-partial"])
    assert not hasattr(stack, "allow_partial_env") and not hasattr(stack, "ENV_ALLOW_PARTIAL") and not hasattr(report, "partial_line") and not hasattr(report, "unserved_line") and not hasattr(design, "partial_exit")


def test_a_broken_lever_or_a_forbidden_line_fails_the_pass(tmp_path, monkeypatch, capsys):
    """Not a refusal: a lever that RAN and whose own record contradicts its contract (registry evidence kind `judged` — L8's hoist anomaly count)
    voids the pass (failed, 1, `levers_broken`), and so does a forbidden driver-log line (failed, 1, `forbidden_lines`); each with its FAILED line."""
    _stubs.box(str(tmp_path), monkeypatch)
    monkeypatch.setenv("STUB_FAIL", "hoiststale")
    rc, man = design.run(_stubs.request(str(tmp_path)), str(tmp_path / "h"), "exact")
    assert rc == 1 and man["status"] == "failed" and man["levers_broken"] == ["L8"] and man["levers_missing"] == ["L8"] and man["forbidden_lines"] == 0
    assert report.broken_line("L8") in capsys.readouterr().err.splitlines() and registry.LEVERS["L8"].evidence[0] == "judged"
    stack.reset_for_tests()
    monkeypatch.setenv("STUB_FAIL", "forbidden")                                                    # the RNG-bookkeeping divergence marker, rc 0 from the driver
    rc, man = design.run(_stubs.request(str(tmp_path)), str(tmp_path / "f"), "exact")
    assert rc == 1 and man["status"] == "failed" and man["forbidden_lines"] == 1 and man["levers_missing"] == [] and man["driver_pass"]["rc"] == 0
    assert report.forbidden_line(1) in capsys.readouterr().err.splitlines()


def test_short_outputs_are_incomplete_and_failures_keep_their_code(tmp_path, monkeypatch, capsys):
    """A PDB count short of the request is the named state `incomplete` (1), never `partial`; a driver that fails is `failed` (1) with
    the partial list recorded."""
    _stubs.box(str(tmp_path), monkeypatch)
    monkeypatch.setenv("STUB_FAIL", "short")
    rc, man = design.run(_stubs.request(str(tmp_path), n_sample=3), str(tmp_path / "o1"), "exact")
    assert rc == 1 and man["status"] == "incomplete" and man["incomplete"] == "2/3" and "expected 3 PDB files" in man["outputs_note"]
    assert man["levers_missing"] == [] and "[genie3-opt] incomplete: 2/3 PDB files for the request" in capsys.readouterr().err
    stack.reset_for_tests()
    monkeypatch.setenv("STUB_FAIL", "rc")
    rc, man = design.run(_stubs.request(str(tmp_path)), str(tmp_path / "o2"), "exact")
    assert rc == 1 and man["status"] == "failed" and man["driver_pass"]["rc"] == 1 and len(man["driver_pass"]["evidence"]["forbidden"]) >= 1
    assert man["levers_missing"] == sorted(["L1", "L2", "L4", "L8", "L9", "L11", "L17", "L18", "L19"]) and man["forbidden_lines"] >= 1   # a crashed driver left no evidence at all: recorded; the failure keeps its own code
    stack.reset_for_tests()
    monkeypatch.setenv("STUB_FAIL", "forbidden")
    rc, man = design.run(_stubs.request(str(tmp_path)), str(tmp_path / "o4"), "exact")
    assert rc == 1 and man["status"] == "failed" and man["forbidden_lines"] == 1 and man["driver_pass"]["evidence"]["missing"] == []   # a forbidden line voids the pass: failed, never partial / never 3


def test_warm_exits_as_its_design_does(tmp_path, monkeypatch, capsys):
    """warm's exit code is the design's own: a lever that could not run refuses the mode by name (3, `levers_missing=` on the WARM line); a short
    output set is 1 — never collapsed."""
    _stubs.box(str(tmp_path), monkeypatch)
    monkeypatch.setenv("STUB_FAIL", "nopatch")
    req = _stubs.request(str(tmp_path))
    rc = cli.main(["warm", "--mode", "exact", "--out_dir", str(tmp_path / "w1"), "--input", req])
    assert rc == 3
    err = capsys.readouterr().err
    assert "[genie3-opt] WARM refused mode=exact rc=3 first_design=1 levers_missing=L2 levers_declined=none out=" in err and report.levers_refused_line("exact", "L2") in err.splitlines()
    stack.reset_for_tests()
    monkeypatch.setenv("STUB_FAIL", "short")
    rc = cli.main(["warm", "--mode", "exact", "--out_dir", str(tmp_path / "w3"), "--input", req, "--n", "2"])
    assert rc == 1 and "[genie3-opt] WARM incomplete mode=exact rc=1 first_design=1 levers_missing=none levers_declined=none out=" in capsys.readouterr().err


def test_warm_line_is_composed_from_the_design_manifest(tmp_path, monkeypatch, capsys):
    """Every token of the WARM line is the design's own record (warm.warm_line over design.run's manifest): the mode the pass RAN, its status and
    exit code, the levers missing and the levers that declined the request by name — a fast warm whose every pair extent is under L7's token
    floor says `levers_declined=L7`, never a bare `ok`."""
    _stubs.box(str(tmp_path), monkeypatch)
    monkeypatch.setenv("STUB_FAIL", "trimuldeclined")
    res = warm.run("fast", str(tmp_path / "w"), request_path=_stubs.request(str(tmp_path)))
    assert res["rc"] == 0 and res["mode"] == "fast" and res["first_design"]["levers_declined"] == ["L7"]
    line = [l for l in capsys.readouterr().err.splitlines() if l.startswith("[genie3-opt] WARM ")]
    assert line == [warm.warm_line(res)] and " mode=fast rc=0 first_design=1 levers_missing=none levers_declined=L7 out=" in line[0], line
    assert warm.warm_line({"status": "ok", "mode": "off", "rc": 0, "out_dir": "/o", "first_design": {"n_pdb": 2}}) == "[genie3-opt] WARM ok mode=off rc=0 first_design=2 levers_missing=none levers_declined=none out=/o"


def test_refused_before_anything_runs(tmp_path, monkeypatch):
    _stubs.box(str(tmp_path), monkeypatch, smi=False)
    out = str(tmp_path / "never")
    rc, man = design.run(_stubs.request(str(tmp_path)), out, "exact")
    assert rc == 3 and man["status"] == "refused" and "no CUDA device" in man["reason"] and not os.path.exists(out)
    rc, man = design.run(_stubs.request(str(tmp_path)), out, "exact_shard")
    assert rc == 3 and "unknown mode 'exact_shard'" in man["reason"]


def test_an_output_directory_is_written_into_as_upstream_writes_into_it(tmp_path, monkeypatch):
    """A directory that already holds designs is not refused (upstream writes into it): the second pass runs, and its completeness count and
    outputs record are this pass's files (written since the pass began), not the earlier pass's."""
    _stubs.box(str(tmp_path), monkeypatch)
    out = str(tmp_path / "o")
    assert design.run(_stubs.request(str(tmp_path), n_sample=2), out, "exact")[0] == 0
    stack.reset_for_tests()
    rc, man = design.run(_stubs.request(str(tmp_path), n_sample=2), out, "exact")
    assert rc == 0 and man["status"] == "ok" and man["outputs"]["n_pdb"] == 2, man
    stack.reset_for_tests()
    import glob, time
    for f in glob.glob(os.path.join(out, "*", "pdbs", "*.pdb")):                          # the earlier pass's files, minutes old on a real box
        os.utime(f, (time.time() - 60, time.time() - 60))
    monkeypatch.setenv("STUB_FAIL", "short")                                            # a short pass over a full directory is still `incomplete`: the earlier files do not count
    rc, man = design.run(_stubs.request(str(tmp_path), n_sample=2), out, "exact")
    assert rc == 1 and man["status"] == "incomplete" and man["incomplete"] == "1/2", man


def test_env_dropped_from_the_driver_process(tmp_path, monkeypatch):
    _stubs.box(str(tmp_path), monkeypatch)
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "1")
    monkeypatch.setenv("GENIE3_OPT", "exact")
    rc, man = design.run(_stubs.request(str(tmp_path)), str(tmp_path / "o"), None)
    assert rc == 0 and man["mode"] == "exact" and man["env_dropped"] == ["GENIE3_OPT", "GENIE3_OPT_HOME", "NVIDIA_TF32_OVERRIDE"]


def test_unconditional_request_layout(tmp_path, monkeypatch):
    """An unconditional request: designs `<length>_<i>` in <out>/pdbs/ (postprocess.py:69-82), counted as lengths x n_sample."""
    _stubs.box(str(tmp_path), monkeypatch)
    rc, man = design.run(_stubs.request(str(tmp_path), n_sample=2, source="unconditional"), str(tmp_path / "u"), "exact")
    assert rc == 0 and man["status"] == "ok", man
    assert man["request"]["problems"] == ["unconditional"] and man["request"]["lengths"] == [50, 100] and man["request"]["designs_expected"] == 4
    assert man["outputs"]["n_pdb"] == 4 and sorted(man["outputs"]["problems"]["unconditional"]) == ["100_0", "100_1", "50_0", "50_1"]
    assert man["driver_pass"]["first_design"].startswith("pdbs/")


def test_l2_evidence_is_the_timings_patches_key(tmp_path, monkeypatch):
    """L2's record is the driver's timings key `patches`: the driver applies its patches with verbose=False (g3batch.py: g3fast_patches.apply), so
    no log line exists — a pass whose log carries no `[g3fast_patches]` line and whose timings say patches=true has L2 applied."""
    _stubs.box(str(tmp_path), monkeypatch)
    rc, man = design.run(_stubs.request(str(tmp_path), n_sample=1), str(tmp_path / "e"), "exact")
    assert rc == 0 and man["status"] == "ok", man
    ev = man["driver_pass"]["evidence"]
    assert "L2" in ev["applied"] and ev["missing"] == [], ev
    assert "[g3fast_patches]" not in open(tmp_path / "e" / "design.log").read()
    assert registry.LEVERS["L2"].evidence == ("timings", "patches")


def test_evidence_is_this_passs_only_in_a_reused_directory(tmp_path, monkeypatch, capsys):
    """The driver log is appended to across passes of one output directory: an earlier pass's `declined` / `engaged` census line is not this
    pass's evidence. declined-then-refused and engaged-then-refused into ONE out_dir both refuse mode fast by name for THIS pass, exit 3 (the earlier lines do not launder it)."""
    _stubs.box(str(tmp_path), monkeypatch)
    out = str(tmp_path / "same")
    monkeypatch.setenv("STUB_FAIL", "trimuldeclined")
    rc, man = design.run(_stubs.request(str(tmp_path), n_sample=2), out, "fast")
    assert rc == 0 and man["levers_declined"] == ["L7"]
    stack.reset_for_tests(); monkeypatch.setenv("STUB_FAIL", "trimulrefused")
    rc, man = design.run(_stubs.request(str(tmp_path), n_sample=2), out, "fast")
    assert rc == 3 and man["status"] == "refused" and man["levers_missing"] == ["L7"] and man["levers_declined"] == [] and "lever=L7" not in capsys.readouterr().err.split("EVIDENCE")[-1]
    out2 = str(tmp_path / "same2")
    stack.reset_for_tests(); monkeypatch.delenv("STUB_FAIL")
    rc, man = design.run(_stubs.request(str(tmp_path), n_sample=2), out2, "fast")
    assert rc == 0 and "L7" in man["driver_pass"]["evidence"]["applied"]
    stack.reset_for_tests(); monkeypatch.setenv("STUB_FAIL", "trimulrefused")
    rc, man = design.run(_stubs.request(str(tmp_path), n_sample=2), out2, "fast")
    assert rc == 3 and man["levers_missing"] == ["L7"] and "L7" not in man["driver_pass"]["evidence"]["applied"]      # the earlier engaged line does not launder this pass's refused gate
    assert man["driver_pass"]["evidence"]["declined"] == {}


def test_check_notes_a_card_without_a_cell_row(tmp_path, monkeypatch, capsys):
    """On a card whose compute capability has no row in fpf_cells.json the dry run NOTES that L7 cannot be served there (a fast pass would be
    partial) — noted, never refused; on a carded capability (9.0) nothing is noted. `check` takes no per-lever switch."""
    _stubs.box(str(tmp_path), monkeypatch)
    assert cli.main(["check", "--mode", "fast", "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)["activation"]
    assert rep["levers_planned"] == ["L1", "L2", "L4", "L8", "L9", "L11", "L17", "L18", "L19", "L12", "L13", "L7", "L16"] and rep["trimul"] == "fpf" and not [n for n in rep["notes"] if "fpf_cells.json" in n]   # cc 9.0: a row serves it
    stack.reset_for_tests()
    smi = [p for p in (os.path.join(d, "nvidia-smi") for d in os.environ["PATH"].split(os.pathsep)) if os.path.isfile(p)][0]
    open(smi, "w").write(_stubs.STUB_SMI.replace(", 9.0", ", 8.9").replace("H100 80GB HBM3", "L40S"))      # a card (cc 8.9) with no cell row
    assert cli.main(["check", "--mode", "fast", "--json"]) == 0
    notes = json.loads(capsys.readouterr().out)["activation"]["notes"]
    assert [n for n in notes if n.startswith("lever L7 (--trimul fpf): opt/genie3_opt/fpf_cells.json has no row for compute capability 8.9") and "refuses by name after the pass, exit 3" in n and n.endswith("(--mode exact does not plan the lever)")], notes
    stack.reset_for_tests()
    assert cli.main(["check", "--mode", "exact", "--json"]) == 0 and not [n for n in json.loads(capsys.readouterr().out)["activation"]["notes"] if "fpf_cells.json" in n]   # exact plans no L7: nothing to note
