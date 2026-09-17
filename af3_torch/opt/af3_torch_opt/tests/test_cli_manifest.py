"""`pred` composes the three model processes and accounts for every item (cli.last_run() = the record its lines are rendered from); the exit codes."""
import json
import os

from af3_torch_opt import cli, modes, registry, stack

from .conftest import stub_calls


def _inputs(tmp, names=("a", "b"), seeds=(1,)):
    d = tmp / "in"; d.mkdir(exist_ok=True)
    paths = []
    for n in names:
        p = d / f"{n}.json"; p.write_text(json.dumps({"name": n, "sequences": [], "modelSeeds": list(seeds)})); paths.append(str(p))
    return paths


def test_pred_chain_and_record(box, capsys):
    out = box["tmp"] / "out"
    ins = _inputs(box["tmp"], seeds=(1, 2))
    rc = cli.main(["pred", "--mode", "fast", "--json_path", ins[0], "--json_path", ins[1], "--output_dir", str(out)])
    assert rc == 0
    calls = stub_calls(box)
    scripts = [os.path.basename(c[1]) for c in calls]
    n_feat = scripts.count("featurise.py")                        # the streamed chain's lever feat_par runs the featurise step as two processes for two inputs (one each); the sequential chain / one input: one
    assert scripts[0] == "featurise.py" and n_feat in (1, 2) and sorted(scripts) == sorted(["featurise.py"] * n_feat + ["forward.py", "postprocess.py"])   # the streamed chain launches the writers beside the model process (their argv records may land in either order); the sequential chain runs them after it
    feat, fwd, post = (next(c for c in calls if os.path.basename(c[1]) == n) for n in ("featurise.py", "forward.py", "postprocess.py"))
    assert feat[0] == box["jax_py"] and post[0] == box["jax_py"] and fwd[0] == box["torch_py"]
    assert "--params_dir" not in post and post[post.index("--repo_dir") + 1] == stack.jax_repo()   # the model identifier rides result.npz (forward.py model_id)
    assert "--seed" not in feat and "--msa_free" not in feat and feat[feat.index("--buckets") + 1] == cli.bucket_row(modes.MODE_PADDING["fast"]) == "tile:64"
    assert feat[feat.index("--run_data_pipeline") + 1] == "1" and feat[feat.index("--run_inference") + 1] == "1" and "--db_dir" not in feat and "--repo_dir" in feat   # the stock CLI's flags and defaults, passed through; a path flag not given is not sent
    assert fwd[fwd.index("--package-levers") + 1] == ",".join(modes.MODE_PACKAGE_LEVERS["fast"]) and fwd[fwd.index("--padding") + 1] == modes.MODE_PADDING["fast"] == "kernel_tile" and "--canonical-buckets" not in fwd
    assert fwd[fwd.index("--levers") + 1] == ",".join(modes.resolve("fast")["levers"]) and "graph_pairformer" not in fwd[fwd.index("--levers") + 1] and fwd[fwd.index("--dtk") + 1] == "1"
    assert fwd[fwd.index("--kit") + 1] == stack.kit_home() and fwd[fwd.index("--params") + 1] == os.path.join(box["params"], stack.checkpoint())   # the exact pinned file, never the directory
    w = f"{out}/{cli.WORK_DIR}"
    assert [x for x in fwd if x.startswith("a=") or x.startswith("b=")] == [f"a=1={w}/a/seed-1/batch.npz={w}/a/seed-1/result.npz", f"a=2={w}/a/seed-2/batch.npz={w}/a/seed-2/result.npz",
                                                                            f"b=1={w}/b/seed-1/batch.npz={w}/b/seed-1/result.npz", f"b=2={w}/b/seed-2/batch.npz={w}/b/seed-2/result.npz"]
    M = cli.last_run()
    assert M["ok"] and [s["step"] for s in M["steps"]] == ["featurise", "forward", "postprocess"] and all(s["ok"] for s in M["steps"])
    assert [it["name"] for it in M["items"]] == ["a", "b"] and all(it["ok"] for it in M["items"])
    assert [(f["seed"], f["forward_s"], f["ok"]) for f in M["items"][0]["forwards"]] == [(1, 3.0, True), (2, 4.0, True)] and M["items"][0]["seeds"] == [1, 2]
    written = {os.path.relpath(os.path.join(dp, f), out) for dp, _, fs in os.walk(out) for f in fs}                 # a clean pred leaves the stock layout and nothing else: the work dir is gone
    assert written == {"a/seed-1_sample-0/a_seed-1_sample-0_model.cif", "a/seed-2_sample-0/a_seed-2_sample-0_model.cif", "a/a_model.cif", "a/ranking_scores.csv",
                       "b/seed-1_sample-0/b_seed-1_sample-0_model.cif", "b/seed-2_sample-0/b_seed-2_sample-0_model.cif", "b/b_model.cif", "b/ranking_scores.csv"}   # stock's files only: no side file beside them
    assert M["activation"]["mode"] == "fast" and M["activation"]["levers_applied"] == list(modes.resolve("fast")["levers"]) and M["stock_proof"]["ok"]
    err = capsys.readouterr().err
    assert "[af3-torch-opt] ACTIVE mode=fast" in err and err.count("[af3-torch-opt] COMMAND") == 2 + n_feat and "[af3-torch-opt] STOCK " in err
    if n_feat == 2:                                               # feat_par: the two featurisers' reports merged into the one featurise report, their step records into one
        assert M["reports"]["featurise"]["workers"] == 2 and M["steps"][0].get("workers") == 2 and M["activation"]["feat_par"]["workers"] == 2
        feats = [c for c in calls if os.path.basename(c[1]) == "featurise.py"]
        assert sorted(x.split("=")[0] for c in feats for i, x in enumerate(c) if c[i - 1] == "--item") == ["a", "b"]   # the inputs dealt between them
    assert "[af3-torch-opt] ITEM name=a seed=1 forward_s=3.0 tokens=100 bucket=256 peak_gb=1.5 ok=1" in err and err.count("[af3-torch-opt] ITEM ") == 4
    assert "[af3-torch-opt] PHASE item=a seed=1 lm_s=- trunk_s=1.0 sampler_s=1.5 conf_s=0.25 total_s=3.0" in err and err.count("[af3-torch-opt] PHASE ") == 4   # one PHASE line per (input, seed) that ran, after its ITEM line
    assert "[af3-torch-opt] DONE ok=1 rc=0 items=2/2 failed=none" in err


def test_pred_off_is_stock(box):
    out = box["tmp"] / "out"
    rc = cli.main(["pred", "--mode", "off", "--json_path", _inputs(box["tmp"], ("x",))[0], "--output_dir", str(out), "--norun_data_pipeline", "--db_dir", "/dbs", "--db_dir", "/dbs2", "--jackhmmer_n_cpu", "3"])
    assert rc == 0
    feat, fwd = [c for c in stub_calls(box) if os.path.basename(c[1]) in ("featurise.py", "forward.py")]
    assert "--seed" not in feat and feat[feat.index("--run_data_pipeline") + 1] == "0" and [feat[i + 1] for i, x in enumerate(feat) if x == "--db_dir"] == ["/dbs", "/dbs2"] and feat[feat.index("--jackhmmer_n_cpu") + 1] == "3" and "--repo_dir" not in feat   # the stock CLI's flags reach the featuriser as given
    assert any(x.startswith("x=1=") for x in fwd)                                                     # the fold input's own modelSeeds
    assert fwd[fwd.index("--levers") + 1] == "" and fwd[fwd.index("--dtk") + 1] == "0"
    M = cli.last_run()
    assert M["activation"]["levers"] == [] and M["activation"]["levers_applied"] == [] and M["stock_proof"]["env_present"] == []
    fj = M["reports"]["forward"]                                   # the forward step's own report, kept on the run record
    assert fj["env_prefixes_present"] == []                      # the stub recorded the model process's own environment


def test_pred_names_a_failed_item(box, monkeypatch, capsys):
    out = box["tmp"] / "out"
    monkeypatch.setenv("STUB_FAIL", "b")
    rc = cli.main(["pred", "--json_path", _inputs(box["tmp"])[0], "--json_path", _inputs(box["tmp"])[1], "--output_dir", str(out)])
    assert rc == 1
    M = cli.last_run()
    assert not M["ok"] and {it["name"]: it["ok"] for it in M["items"]} == {"a": True, "b": False}
    assert M["items"][1]["error"].startswith("featurise: stub: featurise refused")
    fwd = [c for c in stub_calls(box) if os.path.basename(c[1]) == "forward.py"][0]
    err = capsys.readouterr().err
    if M["activation"]["stream"]["engaged"]:                       # the streamed chain (lever prefetch): the model process is launched on every announced input and WITHDRAWS the one whose featurisation failed — no forward, no ITEM line, no writer row
        assert M["activation"]["prefetch"]["withdrawn"] == 1 and "ITEM name=b" not in err and M["items"][1].get("forwards") is None
    else:                                                          # the sequential chain: the failed item never reaches the model process
        assert not any(x.startswith("b=") for x in fwd)
    assert "DONE ok=0 rc=1 items=1/2 failed=b" in err


def test_pred_names_a_failed_seed(box, monkeypatch, capsys):
    out = box["tmp"] / "out"
    monkeypatch.setenv("STUB_FAIL", "forward:a:2")
    rc = cli.main(["pred", "--json_path", _inputs(box["tmp"], ("a",), seeds=(1, 2))[0], "--output_dir", str(out)])
    assert rc == 1
    M = cli.last_run()
    assert not M["items"][0]["ok"] and M["items"][0]["error"] == "forward: seed 2: stub: forward failed"
    assert [f["ok"] for f in M["items"][0]["forwards"]] == [True, False]
    err = capsys.readouterr().err
    assert "ITEM name=a seed=1 forward_s=3.0" in err and "ITEM name=a seed=2 forward_s=None tokens=None bucket=None peak_gb=None ok=0" in err
    post = [c for c in stub_calls(box) if os.path.basename(c[1]) == "postprocess.py"]
    assert len(post) == 1 and (out / "a" / "seed-1_sample-0" / "a_seed-1_sample-0_model.cif").is_file()     # the seed that ran is written out; the item stays failed
    assert M["items"][0]["files"] and "DONE ok=0 rc=1 items=0/1 failed=a" in err


def test_pred_all_seeds_failed_skips_postprocess(box, monkeypatch):
    out = box["tmp"] / "out"
    monkeypatch.setenv("STUB_FAIL", "forward:a")
    assert cli.main(["pred", "--json_path", _inputs(box["tmp"], ("a",), seeds=(1, 2))[0], "--output_dir", str(out)]) == 1
    M = cli.last_run()
    if "write_behind" in M["activation"]["stream"]["engaged"]:      # the streamed chain (lever write_behind): the writers run beside the model process and WITHDRAW the input that never got a result — no row, no output
        assert M["reports"]["postprocess"]["items"] == [] and M["reports"]["postprocess"]["write_behind"]["withdrawn"] == 1 and not (out / "a").exists()
    else:                                                          # the sequential chain: the writers are never launched
        assert not any(os.path.basename(c[1]) == "postprocess.py" for c in stub_calls(box))
    assert M["items"][0]["error"] == "forward: seed 1: stub: forward failed; seed 2: stub: forward failed"


def test_exit_codes(box, monkeypatch, capsys):
    out = str(box["tmp"] / "out")
    assert cli.main(["pred", "--output_dir", out]) == 2                                                    # no input
    assert cli.main(["pred", "--mode", "turbo", "--json_path", _inputs(box["tmp"])[0], "--output_dir", out]) == 2
    assert cli.main(["check", "--mode", "fast"]) == 0
    monkeypatch.setenv("AF3_TORCH_PARAMS_DIR", str(box["tmp"] / "nowhere"))
    assert cli.main(["check", "--mode", "fast"]) == 3
    assert cli.main(["pred", "--json_path", _inputs(box["tmp"])[0], "--output_dir", out]) == 3
    assert not stub_calls(box)                                                                            # nothing launched
    capsys.readouterr()


def test_check_json(box, capsys):
    assert cli.main(["check", "--mode", "off", "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["active"] and rep["mode"] == "off" and rep["levers"] == []


def test_distinct_stems_and_input_dir(box, capsys):
    out = str(box["tmp"] / "out")
    d1, d2 = box["tmp"] / "in1", box["tmp"] / "in2"; d1.mkdir(); d2.mkdir()
    for d in (d1, d2):
        (d / "same.json").write_text(json.dumps({"name": "same", "sequences": [], "modelSeeds": [1]}))
    assert cli.main(["pred", "--json_path", str(d1 / "same.json"), "--json_path", str(d2 / "same.json"), "--output_dir", out]) == 2
    (d1 / "a=b.json").write_text("{}")
    assert cli.main(["pred", "--json_path", str(d1 / "a=b.json"), "--output_dir", out]) == 2
    assert cli.main(["pred", "--input_dir", str(d1), "--output_dir", out]) == 2                     # a=b.json is among them
    os.remove(d1 / "a=b.json")
    assert cli.main(["pred", "--input_dir", str(d1), "--output_dir", out]) == 0
    assert [it["name"] for it in cli.last_run()["items"]] == ["same"]
    capsys.readouterr()


def test_partial_levers_is_refused_unless_allowed(box, monkeypatch, capsys):
    """Fail-loud: the model process applied fewer levers than the mode names -> PARTIAL, rc 3, outputs kept, the missing levers named;
    --allow-partial turns it into rc 0 with the opt-out recorded."""
    out = box["tmp"] / "out"
    fastest = modes.resolve("fast")["levers"]                                  # the levers fast builds with (the kit's fastest set minus registry.FAST_EXCLUDED)
    monkeypatch.setenv("STUB_LEVERS_APPLIED", ",".join(l for l in fastest if l != "trimul"))
    rc = cli.main(["pred", "--mode", "fast", "--json_path", _inputs(box["tmp"], ("a",))[0], "--output_dir", str(out)])
    assert rc == 3
    M = cli.last_run()
    assert M["ok"] and M["items"][0]["ok"] and (out / "a" / "a_model.cif").is_file()          # the run itself completed; it is not the mode it claims
    assert M["partial"] == {"events": [{"kind": "levers", "requested": list(fastest), "applied": [l for l in fastest if l != "trimul"], "missing": ["trimul"]}],
                            "kinds": ["levers"], "allow_partial": False}
    err = capsys.readouterr().err
    assert "[af3-torch-opt] LEVERS requested=" in err and "agree=0 missing=trimul" in err and "DONE ok=1 rc=3 items=1/1 failed=none partial=REFUSED:levers" in err
    out2 = box["tmp"] / "out2"
    rc = cli.main(["pred", "--mode", "fast", "--json_path", _inputs(box["tmp"], ("a",))[0], "--output_dir", str(out2), "--allow-partial"])
    assert rc == 0
    M = cli.last_run()
    assert M["partial"]["allow_partial"] is True and M["partial"]["events"][0]["missing"] == ["trimul"]
    assert "DONE ok=1 rc=0 items=1/1 failed=none partial=allowed:levers" in capsys.readouterr().err
    monkeypatch.delenv("STUB_LEVERS_APPLIED")
    out3 = box["tmp"] / "out3"
    assert cli.main(["pred", "--mode", "fast", "--json_path", _inputs(box["tmp"], ("a",))[0], "--output_dir", str(out3)]) == 0
    assert cli.last_run()["partial"] is None and "partial=none" in capsys.readouterr().err


def test_runtime_fallback_is_a_named_event(box, monkeypatch, capsys):
    """The kit's per-call fallback ladder (a kernel lever serving the stock path, or dying) is read from its census per item and is PARTIAL:
    the counters on the item's forwards[] record and the process census on the run record, a FALLBACK line, rc 3 unless --allow-partial."""
    monkeypatch.setenv("STUB_FALLBACK", "trimul:shape")
    out = box["tmp"] / "out"
    rc = cli.main(["pred", "--mode", "fast", "--json_path", _inputs(box["tmp"], ("a",))[0], "--output_dir", str(out)])
    assert rc == 3
    M = cli.last_run()
    assert M["ok"] and M["partial"]["kinds"] == ["fallback"] and M["partial"]["events"][0]["counts"] == {"trimul": {"fallback:shape": 3}} and M["partial"]["events"][0]["dead"] == {}
    assert M["items"][0]["forwards"][0]["fallbacks"] == {"trimul": {"fallback:shape": 3}} and M["items"][0]["forwards"][0]["dead"] == []
    assert M["census"]["trimul"]["fallback:shape"] == 3 and M["census"]["on"]
    err = capsys.readouterr().err
    assert "[af3-torch-opt] FALLBACK lever=trimul fallback_shape=3 expected=0 dead=0" in err and "partial=REFUSED:fallback" in err
    monkeypatch.setenv("STUB_FALLBACK", "trimul:c=64")                       # a declared coverage gate of the kit: named, counted, not a gate
    out0 = box["tmp"] / "out0"
    assert cli.main(["pred", "--mode", "fast", "--json_path", _inputs(box["tmp"], ("a",))[0], "--output_dir", str(out0)]) == 0
    M = cli.last_run()
    assert M["partial"] is None and M["fallbacks_expected"] == {"trimul": {"fallback:c=64": 3}} and M["items"][0]["forwards"][0]["fallbacks"] == {"trimul": {"fallback:c=64": 3}}
    assert "[af3-torch-opt] FALLBACK lever=trimul fallback_c=64=3 expected=1 dead=0" in capsys.readouterr().err
    monkeypatch.setenv("STUB_FALLBACK", "triattn:DEAD")
    out2 = box["tmp"] / "out2"
    assert cli.main(["pred", "--mode", "fast", "--json_path", _inputs(box["tmp"], ("a",))[0], "--output_dir", str(out2), "--allow-partial"]) == 0
    M = cli.last_run()
    assert M["partial"]["allow_partial"] is True and M["partial"]["events"][0]["dead"] == {"triattn": "RuntimeError('stub kernel error')"}
    assert "FALLBACK lever=triattn kernel_error=1 expected=0 dead=1" in capsys.readouterr().err
    for lever in ("stepgraph", "compile", "dtk"):                          # the runtime levers step aside the same way: dead by name, a FALLBACK line, dead=1 on their LEVER line, PARTIAL
        monkeypatch.setenv("STUB_FALLBACK", lever + ":DEAD")
        outl = box["tmp"] / ("out_" + lever)
        assert cli.main(["pred", "--mode", "fast", "--json_path", _inputs(box["tmp"], ("a",))[0], "--output_dir", str(outl)]) == 3
        M = cli.last_run()
        assert M["ok"] and M["partial"]["kinds"] == ["fallback"] and M["partial"]["events"][0]["dead"] == {lever: "RuntimeError('stub kernel error')"}
        err = capsys.readouterr().err
        assert f"FALLBACK lever={lever} kernel_error=1 expected=0 dead=1" in err, err
        (ll,) = [l for l in err.splitlines() if l.startswith(f"[af3-torch-opt] LEVER name={lever} state=on ")]
        assert ll.endswith(" dead=1"), ll
    monkeypatch.delenv("STUB_FALLBACK")
    out3 = box["tmp"] / "out3"
    assert cli.main(["pred", "--mode", "fast", "--json_path", _inputs(box["tmp"], ("a",))[0], "--output_dir", str(out3)]) == 0
    M = cli.last_run()
    assert M["partial"] is None and M["census"]["on"] and M["items"][0]["forwards"][0]["fallbacks"] == {}


def test_off_asserts_an_empty_census(box, monkeypatch, capsys):
    """The stock proof of off includes the kit's census being absent: a kernel census in an off run is PARTIAL (kind stock), rc 3."""
    out = box["tmp"] / "out"
    assert cli.main(["pred", "--mode", "off", "--json_path", _inputs(box["tmp"], ("a",))[0], "--output_dir", str(out)]) == 0
    M = cli.last_run()
    assert M["census"] is None and M["partial"] is None
    monkeypatch.setenv("STUB_STOCK_KERNELS", "1")
    out2 = box["tmp"] / "out2"
    assert cli.main(["pred", "--mode", "off", "--json_path", _inputs(box["tmp"], ("a",))[0], "--output_dir", str(out2)]) == 3
    M = cli.last_run()
    assert M["partial"]["kinds"] == ["stock"] and M["partial"]["events"][0]["kernels_enabled"] is True
    assert "[af3-torch-opt] STOCK census=present kernels_enabled=1 ok=0" in capsys.readouterr().err and "partial=REFUSED:stock" not in ""


def test_kernel_routes_and_lever_lines_are_named(box, monkeypatch, capsys):
    """fast: the forward argv names the core dir and the routes; one KERNELS line (every route ok, each kernel imported from the core copy) and
    one LEVER line per registry lever (on=1 with its run record); the run record carries kernel_routes. off: the same routes gated, no kernel
    imported (`:none`), every LEVER line on=0 evidence=absent. A refused route is the model process's rc 2: every item failed-with-reason, rc 1."""
    from af3_torch_opt import registry
    inp = _inputs(box["tmp"])[0]
    out = os.path.join(box["tmp"], "o-fast")
    assert cli.main(["pred", "--json_path", inp, "--output_dir", out, "--mode", "fast"]) == 0
    err = capsys.readouterr().err
    fwd = [a for a in stub_calls(box) if a and os.path.basename(a[1]) == "forward.py"][-1]
    assert fwd[fwd.index("--opt-core") + 1] == stack.core_dir() and fwd[fwd.index("--route") + 1] == "flash_triattn,fpf_triatt_k2b,fpf_triatt_epi,dtk_kernels,apb_attn,atom_window"
    assert "[af3-torch-opt] KERNELS routed=flash_triattn,fpf_triatt_k2b,fpf_triatt_epi,dtk_kernels,apb_attn,atom_window ok=1 flash_triattn=stub:core fpf_triatt_k2b=stub:core fpf_triatt_epi=stub:core dtk_kernels=stub:core" in err, err
    from af3_torch_opt import big
    levers = [l for l in err.splitlines() if l.startswith("[af3-torch-opt] LEVER ")]
    assert [l.split("name=")[1].split()[0] for l in levers] == list(registry.LEVERS) + list(big.LEVER_ORDER), levers   # kit levers, then big's memory levers
    assert all(l.split("strategy=")[1].split()[0] == registry.STRATEGY[l.split("name=")[1].split()[0]] for l in levers), levers   # strategy= is the canonical id
    assert all(" " not in v for l in levers for v in l[len("[af3-torch-opt] LEVER "):].split(" ")), levers                    # every value blank-free (k=v tokens)
    on = {l.split("name=")[1].split()[0] for l in levers if " state=on " in l and " arm=fast" in l}
    assert on == (set(modes.resolve("fast")["levers"]) | {"dtk"} | set(modes.MODE_PACKAGE_LEVERS["fast"])) - {"feat_par"}, (on, levers)   # hoist (graph_drop's) and the memory levers are off under fast; fast's package levers on — feat_par steps aside by name for a one-input pred
    assert any(l.startswith("[af3-torch-opt] LEVER name=feat_par state=skipped reason=skipped:single_input ") for l in levers), levers
    assert all((" state=off reason=not_in_arm_lever_set impl=" in l or " state=skipped reason=skipped:single_input impl=" in l) and " arm=fast" in l for l in levers if l.split("name=")[1].split()[0] not in on), levers   # feat_par: selected, skipped by name for one input
    assert "LEVER name=trimul state=on impl=opt_core.kernels.trimul origin=core strategy=F2.fpf_trimul_fast arm=fast served=1 fallback=0 kernel_error=0 dead=0" in err and "LEVER name=dtk state=on impl=dtk_kernels origin=core strategy=LOCAL.dit_fused_kernels arm=fast swap_s=0.1" in err, levers
    man = cli.last_run()
    assert set(man["kernel_routes"]) == set(registry.KERNEL_ROUTES) and man["kernel_routes"]["flash_triattn"]["imported_from"]
    out = os.path.join(box["tmp"], "o-off")
    assert cli.main(["pred", "--json_path", inp, "--output_dir", out, "--mode", "off"]) == 0
    err = capsys.readouterr().err
    assert "KERNELS routed=flash_triattn,fpf_triatt_k2b,fpf_triatt_epi,dtk_kernels,apb_attn,atom_window ok=1 flash_triattn=stub:none fpf_triatt_k2b=stub:none fpf_triatt_epi=stub:none dtk_kernels=stub:none" in err, err
    levers = [l for l in err.splitlines() if l.startswith("[af3-torch-opt] LEVER ")]
    assert len(levers) == len(registry.LEVERS) + len(big.LEVER_ORDER) and all(" state=off reason=not_in_arm_lever_set " in l and " arm=off" in l for l in levers), levers
    monkeypatch.setenv("STUB_ROUTE_FAIL", "flash_triattn")
    out = os.path.join(box["tmp"], "o-refused")
    assert cli.main(["pred", "--json_path", inp, "--output_dir", out, "--mode", "fast"]) == 1
    err = capsys.readouterr().err
    assert "KERNELS routed=flash_triattn,fpf_triatt_k2b,fpf_triatt_epi,dtk_kernels,apb_attn,atom_window ok=0" in err and "kernel route refused: flash_triattn: stub: bytes differ" in json.dumps(cli.last_run()), err


def test_big_pred_lines_and_partial(box, monkeypatch, capsys):
    """big: the featuriser gets the extended bucket row, the forward argv carries --big/--alloc/--kit-last-bucket, the BIG line names
    the levers in force, every F7 LEVER line is state=on with the model process's record, and the run record
    carries big + big_record. An allocator record that does not show expandable segments = LEVER skipped + PARTIAL rc 3 (--allow-partial: rc 0, recorded)."""
    from af3_torch_opt import big, registry
    inp = _inputs(box["tmp"])[0]
    out = os.path.join(box["tmp"], "o-big")
    assert cli.main(["pred", "--json_path", inp, "--output_dir", out, "--mode", "big"]) == 0
    err = capsys.readouterr().err
    calls = stub_calls(box)
    feat = [a for a in calls if a and os.path.basename(a[1]) == "featurise.py"][-1]; fwd = [a for a in calls if a and os.path.basename(a[1]) == "forward.py"][-1]
    assert feat[feat.index("--buckets") + 1] == cli.bucket_row(modes.MODE_PADDING["big"]) == "tile:64" and "--kit-last-bucket" not in fwd   # big pads on fast's open kernel-tile grid
    assert fwd[fwd.index("--big") + 1] == ",".join(big.LEVER_ORDER) and fwd[fwd.index("--alloc") + 1] == "expandable" and "--kit-last-bucket" not in fwd
    built = fwd[fwd.index("--levers") + 1].split(",")
    assert "graph_pairformer" not in built and "stepgraph" not in built and "hoist" in built and "--graph-drop-tokens" not in fwd   # NO graph in big: stepgraph dropped from the build, its hoist kept (gate 0)
    assert big.GRAPH_DROP_MIN_TOKENS == 0
    assert "[af3-torch-opt] BIG base=fastest levers={} disabled=none dropped=stepgraph kept=hoist graph_gate=none".format(",".join(big.LEVER_ORDER)) in err, err
    f7 = [l for l in err.splitlines() if l.startswith("[af3-torch-opt] LEVER ") and l.split("name=")[1].split()[0] in big.LEVER_ORDER]
    assert [l.split("name=")[1].split()[0] for l in f7] == list(big.LEVER_ORDER) and all(" state=on impl=" in l and " arm=big" in l for l in f7), f7
    assert " dropped=stepgraph kept=hoist gate_tokens=0" in f7[0], f7[0]                              # graph_drop in force: the graph lever absent from the build, its hoist kept
    assert "LEVER name=expandable_segments state=on impl=opt_core.mem.torch_alloc origin=core strategy=F7.expandable_segments arm=big alloc=expandable alloc_conf=expandable_segments:True effective=1" in err and "LEVER name=hoist state=on impl=diffusion_head.hoist origin=kit strategy=LOCAL.step_invariant_hoist arm=big record=levers_applied" in err, err   # big builds the hoist, never the step graph
    assert "graph_pairformer" not in err and " bucket_probe=" not in err and " pool_reset=" not in err and " diff_free=1" in err, err
    man = cli.last_run()
    assert man["big"]["levers"] == list(big.LEVER_ORDER) and "items_above_kit_row" not in man["big_record"] and man.get("partial") is None
    monkeypatch.setenv("STUB_ALLOC_INEFFECTIVE", "1")
    out = os.path.join(box["tmp"], "o-big-partial")
    assert cli.main(["pred", "--json_path", inp, "--output_dir", out, "--mode", "big"]) == 3
    err = capsys.readouterr().err
    assert "LEVER name=expandable_segments state=skipped reason=alloc_record_not_expandable(effective=False,source=stub) impl=opt_core.mem.torch_alloc origin=core strategy=F7.expandable_segments arm=big" in err and "BIG agree=0 skipped=expandable_segments" in err, err
    assert cli.last_run()["partial"]["kinds"] == ["big"]
    out = os.path.join(box["tmp"], "o-big-allowed")
    assert cli.main(["pred", "--json_path", inp, "--output_dir", out, "--mode", "big", "--allow-partial"]) == 0



def test_kernels_line_names_a_bypassed_route():
    """report.kernels_line: every route ok + imported from the core copy -> ok=1; a routed module imported from anywhere else (the kit's own
    third_party copy) is `OTHER:<path>` and ok=0; a route missing from the record or a refused route is ok=0; order of the record is immaterial."""
    from af3_torch_opt import report
    core = "/tree/common/opt_core/opt_core/kernels/fpf_trimul_v4"
    good = {"fpf_trimul_v4": {"ok": True, "version": "4.1.0", "core_copy": core, "imported_from": core + "/__init__.py"},
            "flash_triattn": {"ok": True, "version": "as carried (docstring header)", "core_copy": "/tree/common/opt_core/opt_core/kernels/flash_triattn.py", "imported_from": None}}
    names = ["fpf_trimul_v4", "flash_triattn"]
    assert report.kernels_line(good, names) == "[af3-torch-opt] KERNELS routed=fpf_trimul_v4,flash_triattn ok=1 fpf_trimul_v4=4.1.0:core flash_triattn=as:none"
    assert report.kernels_line(dict(reversed(list(good.items()))), names).split()[3] == "ok=1"          # the record's key order is immaterial
    bad = dict(good); bad["fpf_trimul_v4"] = dict(good["fpf_trimul_v4"], imported_from="/elsewhere/site-packages/fpf_trimul_v4/__init__.py")
    line_ = report.kernels_line(bad, names)
    assert " ok=0 " in line_ and "fpf_trimul_v4=4.1.0:OTHER:/elsewhere/site-packages/fpf_trimul_v4/__init__.py" in line_, line_
    assert " ok=0 " in report.kernels_line({"fpf_trimul_v4": good["fpf_trimul_v4"]}, names)                 # a route missing from the record
    assert " ok=0" in report.kernels_line({}, names) and " ok=0 " in report.kernels_line(dict(good, flash_triattn=dict(good["flash_triattn"], ok=False)), names)


def test_norun_inference_is_the_data_pipeline_only_run(box, capsys):
    """--norun_inference (the stock CLI's): the featuriser step runs (data pipeline as flagged) and writes <name>_data.json per input into
    the output dir; no model process, no writers; the pred is ok and leaves no work dir."""
    out = os.path.join(box["tmp"], "o-dp")
    rc = cli.main(["pred", "--mode", "fast", "--json_path", _inputs(box["tmp"], ("q",))[0], "--output_dir", out, "--norun_inference"])
    err = capsys.readouterr().err
    assert rc == 0 and err.count("[af3-torch-opt] COMMAND") == 1 and " ITEM " not in err and "DONE ok=1" in err, err
    M = cli.last_run()
    assert [s["step"] for s in M["steps"]] == ["featurise"] and M["items"][0]["ok"]
    feat = M["steps"][0]["argv"]
    assert feat[feat.index("--run_inference") + 1] == "0" and "--repo_dir" in feat and any(x == f"q={os.path.abspath(_inputs(box['tmp'], ('q',))[0])}={out}/q" for x in feat)
    assert os.path.isfile(os.path.join(out, "q", "q_data.json")) and not os.path.exists(os.path.join(out, cli.WORK_DIR))


def test_stock_usage_rule_and_kept_work_dir(box, capsys):
    """Both stock switches false is the stock CLI's own usage error; a failed pred keeps its work dir and names it on the DONE line."""
    out = os.path.join(box["tmp"], "o-usage")
    assert cli.main(["pred", "--json_path", _inputs(box["tmp"], ("v",))[0], "--output_dir", out, "--norun_inference", "--norun_data_pipeline"]) == cli.EXIT_USAGE
    assert "At least one of --run_inference or --run_data_pipeline must be set to true" in capsys.readouterr().err
    out2 = os.path.join(box["tmp"], "o-fail"); os.environ["STUB_FAIL"] = "w"
    try:
        rc = cli.main(["pred", "--json_path", _inputs(box["tmp"], ("w",))[0], "--output_dir", out2])
    finally:
        os.environ.pop("STUB_FAIL", None)
    err = capsys.readouterr().err
    assert rc == cli.EXIT_FAILED and f" work={out2}/{cli.WORK_DIR}" in err and os.path.isdir(os.path.join(out2, cli.WORK_DIR)), err

