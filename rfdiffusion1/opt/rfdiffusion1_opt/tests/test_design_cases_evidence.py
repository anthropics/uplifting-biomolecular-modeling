"""design: the input form (upstream's hydra overrides — one target, outputs at the typed prefix, the run record beside them) and warm's
bundled example cases (warm.load_cases: re-pointed in the package's own copy, handed to design.run as its rows), refusals by name (a request
the kit line cannot serve: NOT ACTIVE, exit 3, nothing runs), the served overrides composed on the driver line, the evidence read-back from a driver log, the exit tally, a full dry-run / stubbed pass writing opt_manifest.json, and OOM propagation on
the served paths — the core's one classifier (opt_core.oom.is_oom) is the package's only spelling of "out of memory"; a driver log line
it reads as an OOM is a forbidden line of design.read_evidence, so a pass whose driver met an OOM exits 3 through the design entry
instead of passing, and the whole-forward graph's reroute (a failed CUDA-graph capture -> eager) re-raises an out-of-memory before rerouting."""
import json
import os
import re
import sys
import tempfile

import pytest

from opt_core.oom import is_oom
from rfdiffusion1_opt import design, det, modes, report, stack, upstream_args, warm
from rfdiffusion1_opt.manifest import SCHEMA
from rfdiffusion1_opt.registry import KIT_BASE, LEVERS
from rfdiffusion1_opt.tests._stubs import EXACT_EVIDENCE, fake_driver_source, good_box, write_case_outputs

TREE = stack.tree_root()
PKG = os.path.join(TREE, "opt", "rfdiffusion1_opt")
FORWARD = os.path.join(TREE, "opt", "forward")
CAPTURE_REROUTE_SITES = ("fast_inference/drivers/rfd_fullgraph.py",)  # the whole-forward graph's reroute (W1: a failed CUDA-graph capture -> eager): it re-raises an out-of-memory first
PUB = os.path.join(stack.kit_dir(KIT_BASE), "tests", "cases_public.json")   # the kit's bundled example cases (warm's shapes): the tests' multi-target rows


def run_cases(cases_path, out_dir, mode, overrides=None, *, num_designs=None, **kw):
    """design.run over the rows of a cases file (warm's entry: warm.load_cases → design.run(cases=…, out_dir=…)) — the tests' multi-target pass."""
    return design.run(mode, overrides, cases=warm.load_cases(cases_path, num_designs), out_dir=out_dir, **kw)


@pytest.fixture(autouse=True)
def _reset():
    stack.reset_for_tests(); report.reset_for_tests()
    yield
    stack.reset_for_tests(); report.reset_for_tests()


def test_public_cases_are_repointed_in_the_packages_copy():
    p = PUB
    raw = json.load(open(p))
    assert raw[0]["pdb"] == "../inputs_public/insulin_target.pdb" and not any(os.path.isabs(c["pdb"]) for c in raw)   # as shipped: relative to the cases file
    cases = warm.load_cases(p)
    assert len(cases) == len(raw) and all(os.path.isfile(c["pdb"]) for c in cases)
    assert cases[0]["pdb"] == os.path.join(stack.kit_dir(KIT_BASE), "inputs_public", "insulin_target.pdb")
    assert json.load(open(p)) == raw                                                 # the kit file is untouched
    assert all("extra" not in c and "prefix" not in c for c in cases)               # warm's rows: no per-case overrides, outputs under <out_dir>/<name>/
    import inspect
    assert list(inspect.signature(warm.load_cases).parameters) == ["path", "num_designs", "input_dirs"]
    assert [c["num_designs"] for c in warm.public_cases()] == [1] * len(raw) and warm.PUBLIC_CASES == (KIT_BASE, "tests/cases_public.json")   # warm's shapes: the bundled cases at one design each
    assert not hasattr(design, "load_cases") and list(inspect.signature(design.run).parameters)[:2] == ["mode", "overrides"]   # the command line's one input form is upstream's overrides; a cases file is warm's internal affair


def test_a_case_with_per_case_overrides_is_refused_by_name(tmp_path):
    """A cases file carries no per-case hydra overrides (the resident driver composes none per case): a case naming `extra` is refused
    by name before anything runs — settings are typed on the command line."""
    base = json.load(open(PUB))[0]
    stray = tmp_path / "stray.json"
    stray.write_text(json.dumps([dict(base, extra=["diffuser.T=25"])]))
    with pytest.raises(design.DesignError, match=re.escape("case pub_ins_195: extra ['diffuser.T=25']: a cases file carries no per-case overrides")):
        warm.load_cases(str(stray))
    empty = tmp_path / "empty_extra.json"
    empty.write_text(json.dumps([dict(base, extra=[])]))
    assert "extra" not in warm.load_cases(str(empty))[0]                            # an empty list names nothing: the row loads, without the key


def test_case_validation(tmp_path):
    good = {"name": "a", "pdb": "insulin_target.pdb", "contigs": "[A1-115/0 80-80]", "hotspots": "[A59]", "num_designs": 2}
    f = tmp_path / "c.json"
    f.write_text(json.dumps([good, dict(good, name="a")]))
    with pytest.raises(design.DesignError, match="unique"):
        warm.load_cases(str(f))
    f.write_text(json.dumps([dict(good, name="bad name")]))
    with pytest.raises(design.DesignError, match="must be unique and match"):
        warm.load_cases(str(f))
    f.write_text(json.dumps([{"name": "a"}]))
    with pytest.raises(design.DesignError, match=re.escape("expected the keys ('name', 'pdb', 'contigs', 'num_designs')")):
        warm.load_cases(str(f))
    f.write_text(json.dumps([dict(good, pdb="/nowhere/none.pdb")]))
    with pytest.raises(design.DesignError, match="input pdb not found"):
        warm.load_cases(str(f))
    f.write_text("{}")
    with pytest.raises(design.DesignError, match="expected a non-empty JSON list of cases"):
        warm.load_cases(str(f))
    f.write_text(json.dumps([dict(good, startnum=5)]))
    c = warm.load_cases(str(f), num_designs=5)
    assert c[0]["num_designs"] == 5 and c[0]["startnum"] == 5 and c[0]["pdb"].endswith("inputs_public/insulin_target.pdb")
    nohot = {k: v for k, v in good.items() if k != "hotspots"}                      # hotspots is optional, as upstream's ppi.hotspot_res is (base.yaml: null)
    f.write_text(json.dumps([nohot, dict(nohot, name="b", hotspots=""), dict(nohot, name="c", hotspots=None)]))
    assert [x["hotspots"] for x in warm.load_cases(str(f))] == [design.NULL] * 3 == ["null"] * 3
    assert warm.CASE_KEYS == ("name", "pdb", "contigs", "num_designs")


HYDRA_CASE = ["contigmap.contigs=[A1-115/0 80-80]", "ppi.hotspot_res=[A59,A83,A91]", "inference.num_designs=2", "inference.design_startnum=3"]   # + inference.input_pdb / inference.output_prefix per test


def _hydra(tmp_path, prefix, *more):
    pdb = os.path.join(stack.kit_dir(KIT_BASE), "inputs_public", "insulin_target.pdb")
    return [f"inference.input_pdb={pdb}", f"inference.output_prefix={prefix}", *HYDRA_CASE, *more]


def test_hydra_form_is_one_case_at_the_typed_prefix(tmp_path, monkeypatch):
    """design.prepare on upstream's own overrides: one case whose `prefix` is the typed inference.output_prefix (absolute; upstream's default
    samples/design under the working directory when untyped), the run record beside it; no target at all is a usage error (exit 2), never a
    design refusal; a target key typed for warm (whose targets are the bundled cases) is a usage error by name."""
    ov = _hydra(tmp_path, tmp_path / "run" / "samples" / "binder")
    cases, out_dir = design.prepare(ov)
    assert len(cases) == 1 and out_dir == str(tmp_path / "run" / "samples")
    c = cases[0]
    assert (c["name"], c["prefix"], c["num_designs"], c["startnum"], c["hotspots"]) == ("binder", str(tmp_path / "run" / "samples" / "binder"), 2, 3, "[A59,A83,A91]") and "extra" not in c
    assert c["startnum_compose"] == "3"                                                                    # the typed inference.design_startnum token, verbatim: what the resident driver composes (driver_run.case_startnums)
    assert c["pdb"].endswith("inputs_public/insulin_target.pdb") and c["contigs"] == "[A1-115/0 80-80]"
    import inspect
    assert list(inspect.signature(design.prepare).parameters) == ["overrides"]                              # no --out_dir: the run record has no directory of its own — it lies beside the outputs
    monkeypatch.chdir(tmp_path)
    rel, out_rel = design.prepare(_hydra(tmp_path, "x")[0:1] + HYDRA_CASE[:1])      # no prefix typed: upstream's default, relative to the working directory
    assert rel[0]["prefix"] == str(tmp_path / "samples" / "design") and rel[0]["name"] == "design" and out_rel == str(tmp_path / "samples")
    assert (rel[0]["num_designs"], rel[0]["startnum"], rel[0]["hotspots"]) == (10, 0, "null")         # base.yaml: num_designs 10, design_startnum 0, hotspot_res null
    assert rel[0]["startnum_compose"] is None and upstream_args.TARGET_DEFAULTS == {"inference.num_designs": 10, "inference.design_startnum": 0, "inference.output_prefix": "samples/design"}   # untyped: nothing composed on the driver, as upstream's own line carries nothing
    with pytest.raises(design.UsageError, match="no target"):
        design.prepare(["inference.num_designs=3"])
    with pytest.raises(design.UsageError, match="no target"):
        design.prepare(None)
    for typed in (["inference.num_designs=3"], ["inference.input_pdb=x.pdb"], ["contigmap.contigs=[A1-10/0 5-5]"], ["inference.output_prefix=/o/p"], ["inference.design_startnum=2", "diffuser.T=25"]):
        with pytest.raises(design.UsageError, match=re.escape("typed for warm: the targets are the kit's bundled example cases (tests/cases_public.json)")):
            warm.public_cases(typed)                                                                     # warm's targets ARE the bundled cases: a typed target key beside them is a usage error (exit 2), never silently second
    ok = warm.public_cases(["inference.model_directory_path=/models/w", "inference.final_step=5", "inference.write_trajectory=False"], num_designs=7)
    assert {c["num_designs"] for c in ok} == {7}                                                             # the weights directory and any non-target key ride with warm's rows; the count is warm's
    res = warm.run("exact", str(tmp_path / "w"), ["inference.num_designs=3"])
    assert res["rc"] == 2 and res["status"] == "usage" and "typed for warm" in res["reason"]
    rc, man = design.run("exact", ["inference.num_designs=3"])
    assert rc == design.EXIT_USAGE == 2 and man == {"status": "usage", "reason": "no target: type upstream's overrides (inference.input_pdb=… 'contigmap.contigs=[…]' inference.output_prefix=… …) as scripts/run_inference.py takes them"}
    existing = tmp_path / "prev" / "d"
    write_case_outputs("", "", [0, 1, 4], prefix=str(existing))                                          # upstream's design_startnum=-1 rule: continue after the highest index on disk
    nxt, _ = design.prepare(_hydra(tmp_path, existing)[:2] + HYDRA_CASE[:1] + ["inference.design_startnum=-1"])
    assert nxt[0]["startnum"] == 5 and upstream_args.existing_indices(str(existing)) == [0, 1, 4]


def test_read_evidence(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("STACK torch 2.4.0\nfastpath levers: ['chain_breaks', 'full_graph', 'rbf', 'msa_index']\neinsum lever E: {'active': True}\n"
                   "fullgraph mode: applied = True\nprep lever: active = True\npdb writer lever IO1: armed (rfdiffusion.util.writepdb, writepdb_multi)\n")
    ev = design.read_evidence(str(log), list(modes.resolve("exact").levers))
    assert ev["applied"] == ["C1", "P", "E_einsum", "W1", "IO1"] and ev["missing"] == [] and ev["forbidden"] == []
    log.write_text("fastpath levers: []\nREFUSED: fullgraph needs dgl>=2.0\nfullgraph mode: applied = False\n")   # W1's line with applied = False is not its evidence
    ev = design.read_evidence(str(log), list(modes.resolve("exact").levers))
    assert ev["applied"] == ["C1"] and ev["missing"] == ["P", "E_einsum", "W1", "IO1"] and len(ev["forbidden"]) == 1
    log.write_text("fastpath levers: ['chain_breaks']\nprep lever: active = True\neinsum lever E: {}\nfullgraph mode: applied = True\npdb writer lever IO1: armed (rfdiffusion.util.writepdb, writepdb_multi)\n"
                   "[rfd_se3fast] v0.3.0 applied: mode=t2 scope=all se3_fn=se3_dense_forward_triton triton=1\ntriton LN lever (Tier 2): active = True\n{\n \"tag\": \"run\",\n \"tf32\": true,\n \"torch\": \"2.4.0\"\n}\n")
    ev = design.read_evidence(str(log), list(modes.resolve("fast").levers))                     # the fast row: T2's, K2's and TF32's own lines (TF32: the driver's run-manifest key)
    assert ev["missing"] == [] and ev["applied"][-2:] == ["K2", "TF32"], ev
    log.write_text("{\n \"tf32\": false,\n}\ntriton LN lever (Tier 2): active = False\n")
    ev = design.read_evidence(str(log), ["K2", "TF32"])
    assert ev["applied"] == [] and ev["missing"] == ["K2", "TF32"]


def test_evidence_regexes_match_the_drivers_print_lines():
    src = open(os.path.join(stack.kit_dir(KIT_BASE), "drivers", "rfd_bench.py"), encoding="utf-8").read()
    for lid, needle in (("C1", "print('fastpath levers:', FAST"), ("P", 'print("prep lever: active ="'),
                        ("E_einsum", 'print("einsum lever E:"'), ("K2", 'print("triton LN lever (Tier 2): active ="'), ("W1", 'print("fullgraph mode: applied ="'),
                        ("TF32", "tf32=bool(args.tf32)"), ("TF32", "print(json.dumps(manifest, indent=1), flush=True)")):      # TF32's evidence: the run manifest's `"tf32": true` key, one per line
        assert needle in src, (lid, needle)
    import re as _re
    assert _re.search(LEVERS["TF32"].evidence, ' "tf32": true,', _re.M) and not _re.search(LEVERS["TF32"].evidence, ' "tf32": false,', _re.M)
    assert _re.search(LEVERS["W1"].evidence, "fullgraph mode: applied = True (2 graphs)") and not _re.search(LEVERS["W1"].evidence, "fullgraph mode: applied = False")


def test_exit_tally(tmp_path, capsys):
    t = tmp_path / "run_timings.json"
    t.write_text(json.dumps({"fullgraph_final": {"n_replay": 1750, "n_capture": 2}, "prep_final": {"hits": 34}}))
    log = tmp_path / "run.log"
    log.write_text("fastpath levers: ['chain_breaks']\n")
    report.add_tally_source("run", str(t), str(log))
    line = report.exit_tally_line(pid=1)
    assert line.startswith("[rfdiffusion1-opt] EXIT pid=1 source=driver-files run:") and "fullgraph_final={n_replay=1750,n_capture=2}" in line and "prep_final={hits=34}" in line
    report.reset_for_tests()
    assert "no driver pass ran" in report.exit_tally_line(pid=1)


def test_dry_run_pass(monkeypatch, tmp_path, capsys):
    good_box(monkeypatch)
    p = os.path.join(stack.kit_dir(KIT_BASE), "tests", "cases_public.json")
    rc, man = run_cases(p, str(tmp_path / "out"), "exact", tag="t", dry_run=True, python="python")
    assert rc == 0 and man["status"] == "dry-run" and man["driver_cmd"][0] == "python"
    assert " ".join(man["driver_cmd"]).count(modes.K_LINE) == 1 and man["env_row"] == {"RFD_PDBIO": "1"} and "--compose" not in man["driver_cmd"]
    assert man["driver_cmd"][3:9] == ["--fixed", "inference.write_trajectory=True", "--fixed", "inference.cautious=True", "--fixed", "inference.deterministic=False"]
    assert man["det"] == 0 and man["compose_overrides"] == ["inference.write_trajectory=True", "inference.cautious=True", "inference.deterministic=False"] and "declined" not in man
    assert man["schema"] == "rfdiffusion1_opt.manifest/2" and not {"recipe", "host", "recipe_line"} & set(man)     # no recipe / host blocks: the deterministic recipe is the one `det` level
    assert man["weights"] == {"dir": "/models/rfdiffusion", "files": {"Complex_base_ckpt.pt": {"present": False}}}       # the checkpoint this pass loads, present or not — never digested, never gated
    assert not (tmp_path / "out").exists()                                                                          # a dry run touches nothing
    rc, man_d = run_cases(p, str(tmp_path / "out_d"), "exact", tag="t", dry_run=True, python="python", det_flag=True)   # --det 1: the seed composed on the driver line
    assert rc == 0 and man_d["det"] == 1 and man_d["driver_cmd"][7:9] == ["--fixed", "inference.deterministic=True"]
    rc, man_o = run_cases(p, str(tmp_path / "out2"), "off", dry_run=True, python="python")
    assert rc == 0 and len(man_o["stock_cmds"]) == len(man_o["cases"]) and man_o["stock_cmds"][0][1:4] == ["-s", "-m", "rfdiffusion1_opt.stock_cli"]
    assert det.SEED_OVERRIDE not in man_o["stock_cmds"][0] and man_o["compose_overrides"] == [] and man_o["driver_fixed"] is None
    rc, man_od = run_cases(p, str(tmp_path / "out3"), "off", dry_run=True, python="python", det_flag=True)
    assert rc == 0 and all(cmd.count(det.SEED_OVERRIDE) == 1 for cmd in man_od["stock_cmds"]) and man_od["det"] == 1
    # the command line's form: one target from upstream's overrides; outputs at the prefix, the record beside it
    ov = _hydra(tmp_path, tmp_path / "run" / "binder")
    rc, man_h = design.run("exact", ov, tag="t", dry_run=True, python="python")
    assert rc == 0 and man_h["out_dir"] == str(tmp_path / "run") and man_h["cases"][0]["prefix"] == str(tmp_path / "run" / "binder") and man_h["mode"] == "exact"
    assert man_h["overrides"] == ov and man_h["compose_overrides"] == ["inference.write_trajectory=True", "inference.cautious=True", "inference.deterministic=False"]   # the target keys are the case row's, not composed twice
    assert not any(t.startswith("--compose") for t in man_h["driver_cmd"])
    rc, man_ho = design.run("off", ov, dry_run=True, python="python")
    i = man_ho["stock_cmds"][0].index("--")
    assert rc == 0 and man_ho["stock_cmds"][0][i + 1:] == ov + ["inference.model_directory_path=/models/rfdiffusion"]   # the stock command line gets exactly what was typed (+ the weights directory it was not told, WEIGHTS)
    assert man_ho["stock_cmds"][0][man_ho["stock_cmds"][0].index("--proof-json") + 1] == str(tmp_path / "run" / "stock_env_proof.json")


UNSERVED_TAIL = " — refused by name, nothing ran; `--mode off` runs this request on upstream's command line (scripts/run_inference.py)"


def test_a_request_the_kit_line_cannot_serve_is_refused_by_name(monkeypatch, tmp_path, capsys):
    """What a kit mode cannot serve — symmetric oligomers, cyclic peptides, fold conditioning, sequence inpainting, a typed architecture /
    trained-schedule key, another --config-name, a Hydra application flag, a stray word — is refused by name before anything is resolved or run:
    ONE `NOT ACTIVE: mode=<m> cannot serve <feature [token]: mechanism; …>` line, exit 3, no manifest, no stock route (`--mode off` runs it,
    every token verbatim). The typed mode, the default mode, design, the packed line, check and warm share the one refusal (design.refuse_unserved)."""
    from rfdiffusion1_opt import serve
    good_box(monkeypatch)
    ov = _hydra(tmp_path, tmp_path / "run" / "binder")
    cases = (
        (["inference.symmetry=C3"], ["symmetric oligomers [inference.symmetry=C3]: the resident driver restates SelfConditioning.sample_step without upstream's symmetry branches and composes config/inference/base.yaml, not symmetry.yaml"]),
        (["--config-name", "symmetry"], ["primary config symmetry.yaml [--config-name symmetry]: the resident driver composes config/inference/base.yaml (symmetry.yaml is symmetric oligomer design)"]),
        (["-cn", "symmetry", "--resolve"], ["primary config symmetry.yaml [-cn symmetry]: the resident driver composes config/inference/base.yaml (symmetry.yaml is symmetric oligomer design)",
                                            "Hydra flag [--resolve]: it acts in hydra.main(), which the resident driver does not run (the configuration is composed through hydra.compose())"]),
        (["--cfg", "job"], ["Hydra flag [--cfg job]: it acts in hydra.main(), which the resident driver does not run (the configuration is composed through hydra.compose())"]),
        (["inference.cyclic=True", "inference.cyc_chains=a"], ["cyclic peptides [inference.cyclic=True]: upstream's per-call cyclic index loops in the positional embedding (torch.unique, mask writes) cannot run inside lever W1's captured forward graph"]),
        (["scaffoldguided.scaffoldguided=True", "scaffoldguided.scaffold_dir=/s"], ["fold-conditioned design [scaffoldguided.scaffoldguided=True]: unproven on the kit line: its checkpoints (Complex_Fold_base_ckpt.pt, InpaintSeq_Fold_ckpt.pt) are outside the kit's pinned weights"]),
        (["contigmap.inpaint_seq=[A1-10]", "contigmap.provide_seq=[5-9]"], ["sequence inpainting [contigmap.inpaint_seq=[A1-10]]: unproven on the kit line: upstream selects InpaintSeq_ckpt.pt for it, outside the kit's pinned weights",
                                                                            "sequence inpainting [contigmap.provide_seq=[5-9]]: unproven on the kit line: upstream selects InpaintSeq_ckpt.pt for it, outside the kit's pinned weights"]),
        (["diffuser.T=25", "++model.n_main_block=8"], ["the trained noise schedule [diffuser.T=25]: " + upstream_args.REFUSED_GROUPS["diffuser."][1], "the network's architecture keys [++model.n_main_block=8]: " + upstream_args.REFUSED_GROUPS["model."][1]]),
        (["inference.model_runner=default"], ["inference.model_runner=default [inference.model_runner=default]: " + upstream_args.REFUSED_VALUES["inference.model_runner"][1]]),
        (["inference.empty_cache_per_design=true", "logging.inputs=1", "hydra.run.dir=/x"], ["emptying the CUDA cache per design [inference.empty_cache_per_design=true]: " + upstream_args.REFUSED_SWITCHES["inference.empty_cache_per_design"][1],
                                                                                          "the model-input pickler [logging.inputs=1]: " + upstream_args.REFUSED_SWITCHES["logging.inputs"][1],
                                                                                          "Hydra's run settings [hydra.run.dir=/x]: " + upstream_args.REFUSED_GROUPS["hydra."][1]]),
        (["paper"], ["argument ['paper']: not a Hydra override (KEY=VALUE) or flag"]),
    )
    for extra, reasons in cases:
        for mode in ("exact", "fast", None):                                                   # no mode word = the default kit line (fast): it refuses alike, named as fast
            stack.reset_for_tests(); capsys.readouterr()
            m = mode or "fast"
            assert design.refusals(mode, ov + extra) == reasons and design.refusals("off", ov + extra) == [], (extra, design.refusals(mode, ov + extra))
            rc, man = design.run(mode, ov + extra, tag="t", dry_run=True, python="python")
            err = capsys.readouterr().err
            line = f"[rfdiffusion1-opt] NOT ACTIVE: mode={m} cannot serve {'; '.join(reasons)}{UNSERVED_TAIL}"
            assert rc == 3 and man == {"status": "refused", "reason": line.split("NOT ACTIVE: ", 1)[1], "mode": m, "refused": reasons}, (extra, mode, man)
            assert err.splitlines() == [line] and "DECLINED" not in err, (extra, mode, err)   # one line: nothing resolved (no DRY-RUN line), nothing run, no stock route
        stack.reset_for_tests(); capsys.readouterr()
        rc, man = serve.run("exact", ov + extra, k=2, dry_run=True, python="python")           # the packed line: the same refusal, before any slicing
        assert rc == 3 and man["refused"] == reasons and "serve" not in man and capsys.readouterr().err.count("NOT ACTIVE") == 1, extra
        stack.reset_for_tests(); capsys.readouterr()
        res = warm.run("exact", str(tmp_path / "w"), extra)                                   # warm: nothing of the kit's to warm for a request its line cannot serve — the same refusal, rc 3
        assert (res["status"], res["rc"], res["refused"], res["first_designs"]) == ("refused", 3, reasons, None) and capsys.readouterr().err.count("NOT ACTIVE") == 1, extra
        stack.reset_for_tests(); capsys.readouterr()
        rc, man = design.run("off", ov + extra, dry_run=True, python="python")                # off: upstream's command line, every token verbatim, nothing refused
        i = man["stock_cmds"][0].index("--")
        assert rc == 0 and man["mode"] == "off" and man["stock_cmds"][0][i + 1:i + 1 + len(ov + extra)] == ov + extra and "NOT ACTIVE" not in capsys.readouterr().err, extra
    stack.reset_for_tests(); capsys.readouterr()
    p = os.path.join(stack.kit_dir(KIT_BASE), "tests", "cases_public.json")
    rc, man = run_cases(p, str(tmp_path / "o"), "fast", ["inference.symmetry=C3"], dry_run=True, python="python")   # warm's rows refuse alike
    assert rc == 3 and man["status"] == "refused" and capsys.readouterr().err.count("NOT ACTIVE: mode=fast cannot serve symmetric oligomers") == 1
    assert not hasattr(design, "declined") and not hasattr(design, "DECLINED_FMT") and not hasattr(upstream_args, "DRIVER_KEYS") and not hasattr(upstream_args, "eligible")   # no decline-then-stock route exists


def test_refused_switches_are_read_as_upstream_reads_them_and_fail_closed(monkeypatch, tmp_path, capsys):
    """A boolean switch upstream tests for truth (`if conf.inference.cyclic:`) is off only for OmegaConf's falsy parses — false / 0 / null / the
    empty word; every other spelling (2, T, Y, yes, on, TRUE) is on upstream, so it is refused here, never served (upstream_args.switched_on).
    A valued switch (`inference.symmetry`, `contigmap.inpaint_seq`: tested `is not None`) is off for null alone. Each Hydra flag answers for its
    own value; a config-name flag without its value is refused with its usage."""
    good_box(monkeypatch)
    ov = _hydra(tmp_path, tmp_path / "run" / "binder")
    sw = upstream_args.switched_on
    for v in ("2", "T", "Y", "yes", "on", "1", "true", "True", "TRUE", "none", "~", "01"):
        assert sw("inference.cyclic", v) and sw("scaffoldguided.scaffoldguided", v), v
        for key, feature in (("inference.cyclic", "cyclic peptides"), ("scaffoldguided.scaffoldguided", "fold-conditioned design"), ("logging.inputs", "the model-input pickler"),
                             ("inference.empty_cache_per_design", "emptying the CUDA cache per design")):
            why = design.refusals("exact", ov + [f"{key}={v}"])
            assert len(why) == 1 and why[0].startswith(f"{feature} [{key}={v}]: "), (key, v, why)
    for v in ("false", "False", "FALSE", "0", "null", "NULL", "", " false "):
        assert not sw("inference.cyclic", v), repr(v)
        assert design.refusals("exact", ov + [f"inference.cyclic={v}", f"++scaffoldguided.scaffoldguided={v}", f"logging.inputs={v}"]) == [], repr(v)
    for v in ("null", "NULL", "Null"):
        assert design.refusals("fast", ov + [f"inference.symmetry={v}", f"contigmap.inpaint_seq={v}", f"contigmap.provide_seq={v}", f"contigmap.inpaint_str={v}"]) == [], v
    for v in ("none", "None", "", "~", "false", "0", "C3"):                                         # not null to upstream: `is not None` holds, the feature is on — refused
        why = design.refusals("fast", ov + [f"inference.symmetry={v}", f"contigmap.inpaint_seq={v}"])
        assert [w.split(" [", 1)[0] for w in why] == ["symmetric oligomers", "sequence inpainting"], (v, why)
    assert design.refusals("exact", ov + ["~inference.cyclic", "~inference.symmetry=C3"]) == []       # deleting a key switches nothing on
    stack.reset_for_tests(); capsys.readouterr()
    rc, man = design.run("exact", ov + ["inference.cyclic=2"], dry_run=True, python="python")         # the whole route: refused by name, exit 3, one line
    assert rc == 3 and man["refused"] == ["cyclic peptides [inference.cyclic=2]: " + upstream_args.REFUSED_SWITCHES["inference.cyclic"][1]]
    assert capsys.readouterr().err.count("NOT ACTIVE: mode=exact cannot serve cyclic peptides [inference.cyclic=2]") == 1
    # Hydra flags: each answers for its own value
    assert design.refusals("exact", ov + ["-cn", "base", "--config-name", "symmetry"]) == ["primary config symmetry.yaml [--config-name symmetry]: the resident driver composes config/inference/base.yaml (symmetry.yaml is symmetric oligomer design)"]
    assert design.refusals("exact", ov + ["--config-name=symmetry", "-cn", "base"]) == ["primary config symmetry.yaml [--config-name=symmetry]: the resident driver composes config/inference/base.yaml (symmetry.yaml is symmetric oligomer design)"]
    assert design.refusals("exact", ov + ["--config-name"]) == ["Hydra flag [--config-name] without its value (usage: --config-name <config>): the resident driver composes config/inference/base.yaml"]
    assert design.refusals("exact", ov + ["-cn=base"]) == [] and design.refusals("off", ov + ["--config-name"]) == []            # base served; off answers for nothing (upstream's parser does)
    # warm creates nothing for a request it refuses
    stack.reset_for_tests(); capsys.readouterr()
    res = warm.run("fast", str(tmp_path / "never"), ["inference.cyclic=T"])
    assert res["rc"] == 3 and res["status"] == "refused" and not (tmp_path / "never").exists() and capsys.readouterr().err.count("NOT ACTIVE") == 1


def test_served_overrides_ride_on_the_driver_line(monkeypatch, tmp_path, capsys):
    """Every override the kit line serves — guiding potentials, the noise schedules, partial diffusion, contigmap.length, inference.align_motif /
    final_step, an appending `+KEY`, a forcing `++KEY`, a deleting `~KEY`, a symmetry / cyclic switch typed OFF, `--config-name base` — is composed
    verbatim on the resident driver's configuration (driver_cmd --compose, in typed order, the +/++/~ form kept); the target keys and the
    driver's fixed keys are composed by their own rows; nothing is refused."""
    good_box(monkeypatch)
    ov = _hydra(tmp_path, tmp_path / "run" / "binder")
    served = ["potentials.guiding_potentials=[\"type:binder_ROG,weight:2\"]", "potentials.guide_scale=2", "potentials.guide_decay=quadratic", "denoiser.noise_scale_ca=0.5",
              "denoiser.final_noise_scale_ca=1", "denoiser.ca_noise_schedule_type=linear", "diffuser.partial_T=10", "contigmap.length=70-90", "inference.align_motif=False",
              "inference.final_step=5", "+inference.extra=1", "~ppi.hotspot_res", "inference.symmetry=null", "inference.cyclic=False", "inference.model_runner=SelfConditioning",
              "++inference.write_trajectory=False", "potentials.olig_intra_all=True"]
    for mode in ("exact", "fast"):
        stack.reset_for_tests(); capsys.readouterr()
        assert design.refusals(mode, ov + served + ["--config-name", "base"]) == []
        rc, man = design.run(mode, ov + served + ["--config-name", "base"], dry_run=True, python="python")
        err = capsys.readouterr().err
        assert rc == 0 and man["mode"] == mode and man["attach"] == "driver" and "NOT ACTIVE" not in err and "DECLINED" not in err, (mode, err)
        cmd = man["driver_cmd"]
        composed = [cmd[i + 1] for i, t in enumerate(cmd) if t == "--compose"]
        assert composed == [t for t in served if not t.startswith("++inference.write_trajectory")], (mode, composed)   # every served override verbatim, in typed order; the fixed key's ++ form is composed by its --fixed row
        assert cmd[3:5] == ["--fixed", "inference.write_trajectory=False"] and "--no-traj 1" in " ".join(cmd) and "--config-name" not in " ".join(cmd)
    stack.reset_for_tests(); capsys.readouterr()
    rc, man = design.run("exact", ov + ["inference.final_step=5", "denoiser.noise_scale_ca=0.5", "inference.write_trajectory=False"], dry_run=True, python="python")
    assert rc == 0 and man["mode"] == "exact" and "declined" not in man                                          # keys of the driver's: the kit line, composed
    assert man["driver_cmd"][man["driver_cmd"].index("--compose"):man["driver_cmd"].index("--compose") + 4] == ["--compose", "inference.final_step=5", "--compose", "denoiser.noise_scale_ca=0.5"]
    assert man["driver_cmd"][3:5] == ["--fixed", "inference.write_trajectory=False"] and "--no-traj 1" in " ".join(man["driver_cmd"])


def test_fold_conditioned_design_names_its_target_by_scaffoldguided_on_the_stock_route(monkeypatch, tmp_path, capsys):
    """A fold-conditioned request (scaffoldguided.*) types no inference.input_pdb / contigmap.contigs: on --mode off that is a target (upstream's
    command line gets every token verbatim), never the `no target` usage error; a kit mode refuses it by name."""
    good_box(monkeypatch)
    ov = ["scaffoldguided.scaffoldguided=True", "scaffoldguided.target_pdb=True", "scaffoldguided.target_path=/t/insulin.pdb", "scaffoldguided.scaffold_dir=/s",
          "ppi.hotspot_res=[A59,A83,A91]", f"inference.output_prefix={tmp_path / 'fc' / 'des'}", "inference.num_designs=2"]
    rc, man = design.run("off", ov, dry_run=True, python="python")
    i = man["stock_cmds"][0].index("--")
    assert rc == 0 and man["mode"] == "off" and man["stock_cmds"][0][i + 1:i + 1 + len(ov)] == ov and man["cases"][0]["name"] == "des"
    stack.reset_for_tests(); capsys.readouterr()
    rc, man = design.run("exact", ov, dry_run=True, python="python")
    assert rc == 3 and man["status"] == "refused" and "cannot serve fold-conditioned design [scaffoldguided.scaffoldguided=True]" in capsys.readouterr().err
    rc, man = design.run("off", ov[4:], dry_run=True, python="python")                                          # no target of any kind: the usage error, on every mode
    assert rc == 2 and man["status"] == "usage" and "no target" in man["reason"]


def test_refusals_before_anything_runs(monkeypatch, tmp_path):
    good_box(monkeypatch)
    p = os.path.join(stack.kit_dir(KIT_BASE), "tests", "cases_public.json")
    rc, man = run_cases(p, str(tmp_path / "o"), "turbo")                                                        # an unknown name: refused by name, exit 3
    assert rc == 3 and man["status"] == "refused" and man["reason"] == "unknown mode 'turbo' (expected off|exact|fast)"
    stray = tmp_path / "stray.json"
    stray.write_text(json.dumps([dict(json.load(open(p))[0], extra=["diffuser.T=25"])]))
    with pytest.raises(design.DesignError, match="a cases file carries no per-case overrides"):              # a bundled case naming per-case overrides: refused by name (warm.load_cases) before anything runs
        run_cases(str(stray), str(tmp_path / "o"), "exact")
    assert not (tmp_path / "o").exists()
    (tmp_path / "o" / "pub_ins_195").mkdir(parents=True)                                                         # an existing output directory is not a refusal: upstream writes into whatever exists (its cautious rule decides per file)
    rc, man = run_cases(p, str(tmp_path / "o"), "exact", dry_run=True, python="python")
    assert rc == 0 and man["status"] == "dry-run"


def test_stubbed_driver_pass_writes_manifest(monkeypatch, tmp_path, capsys):
    """A fake driver that prints every evidence line and writes the outputs: the pass is `ok`, the manifest complete, the ready line
    printed from the first design's mtime, the activation report's applied / partial keys filled."""
    good_box(monkeypatch, weights=str(tmp_path / "no_weights"))
    fake = tmp_path / "fake_driver.py"
    fake.write_text(fake_driver_source())
    monkeypatch.setenv("STUBS", os.path.dirname(__file__))
    monkeypatch.setattr(stack, "driver_command", lambda res, cases, out, tag, rfd=None, weights=None, python=None, compose_log=None, compose=None: [sys.executable, str(fake), "--cases", cases, "--out", out, "--tag", tag, "--no-traj", res.settings.driver_flags[1]])
    p = os.path.join(stack.kit_dir(KIT_BASE), "tests", "cases_public.json")
    rc, man = run_cases(p, str(tmp_path / "out"), "exact", tag="t", num_designs=2)
    assert rc == 0 and man["status"] == "ok", man.get("driver_passes")
    assert man["levers_evidenced"] == ["C1", "E_einsum", "IO1", "P", "W1"] and man["levers_missing"] == [] and man["forbidden_lines"] == 0
    dp = man["driver_passes"][0]
    assert dp["numerics"]["source"] == "torch" and dp["numerics"]["mismatch"] == [] and dp["numerics"]["policy"] == "fp32_strict"   # the driver's torch read-back equals exact's declaration
    assert not ({"ready_s", "wall_s", "first_design"} & set(dp))                                # the manifest carries no kit timing (no wall, no cold-start figure)
    assert dp["cmd"][dp["cmd"].index("--cases") + 1].startswith(tempfile.gettempdir()) and not (tmp_path / "out" / "cases.json").exists() and not (tmp_path / "out" / "_cases").exists()   # the driver's cases file lives in a scratch directory (design.scratch_dir), never among the outputs
    err = capsys.readouterr().err
    i_plan, i_ev, i_active = err.index("[rfdiffusion1-opt] PLAN mode=exact "), err.index("[rfdiffusion1-opt] EVIDENCE "), err.index("[rfdiffusion1-opt] ACTIVE mode=exact ")
    assert i_plan < i_ev < i_active and err.count("[rfdiffusion1-opt] ACTIVE ") == 1            # PLAN at activation; the ACTIVE documented line once, after the first pass's applied-lines
    assert man["activation"]["proven"] is True and man["activation"]["proven_by"].startswith("pass t: applied-lines ")
    assert "[rfdiffusion1-opt] ready " not in err                                                # the kit prints no cold-start / timing line
    rep = stack.status()
    assert rep["levers_applied"] == ["C1", "E_einsum", "IO1", "P", "W1"] and rep["levers_unavailable"] == [] and rep["partial"] is False
    assert man["outputs"]["n_pdb"] == 2 * len(man["cases"]) and man["outputs"]["n_traj"] == 4 * len(man["cases"])
    m = json.load(open(tmp_path / "out" / "opt_manifest.json"))
    assert m["schema"] == "rfdiffusion1_opt.manifest/2" == SCHEMA and m["mode"] == "exact" and m["line"].startswith("RFD_PDBIO=1 <fast_inference>/drivers/rfd_bench.py")
    assert m["det"] == 0 and not {"recipe", "recipe_line", "host", "declined"} & set(m)       # the deterministic recipe is the one `det` level; no host record, no recipe block
    assert m["weights"] == {"dir": str(tmp_path / "no_weights"), "files": {"Complex_base_ckpt.pt": {"present": False}}} and m["activation"]["gpu"]["class"] == "H100"   # a user's weights: recorded present or not, never digested, never refused
    assert all("sha256" not in json.dumps(m[k]) for k in ("weights", "outputs")) and "pinned" not in json.dumps(m["weights"])
    assert m["outputs"]["expected"] == 2 * len(man["cases"]) and set(m["outputs"]["cases"]) == {c["name"] for c in man["cases"]}
    per = m["outputs"]["cases"]["pub_ins_195"]
    assert per["prefix"] == str(tmp_path / "out" / "pub_ins_195" / "des") and per["indices"] == [0, 1] and per["designs"]["0"] == {"pdb": True, "trb": True, "traj": ["des_0_Xt-1_traj.pdb", "des_0_pX0_traj.pdb"]}
    assert os.path.isfile(tmp_path / "out" / "t.log") and os.path.isfile(tmp_path / "out" / "t_timings.json")
    assert m["activation"] == {k: v for k, v in man["activation"].items()} and "pins" not in m["activation"] and "forced" not in m["activation"]   # the activation report copied as is; the pins are check's report, not a gate
    import inspect
    assert "process_per_case" not in inspect.signature(design.run).parameters and "repro" not in inspect.signature(design.run).parameters   # one resident process per pass: no per-case switch, no recipe switch
    # the typed overrides reach the pass: no trajectories under inference.write_trajectory=False (the driver's --no-traj 1), the value composed on the driver line
    rc, man2 = run_cases(p, str(tmp_path / "out_nt"), "exact", ["inference.write_trajectory=False", "inference.cautious=False"], tag="t", num_designs=1)
    assert rc == 0 and len(man2["driver_passes"]) == 1 and man2["outputs"]["n_traj"] == 0 and man2["outputs"]["n_pdb"] == len(man2["cases"])
    assert man2["overrides"] == ["inference.write_trajectory=False", "inference.cautious=False"] and man2["compose_overrides"][:2] == ["inference.write_trajectory=False", "inference.cautious=False"]


def test_hydra_form_pass_writes_where_upstream_writes(monkeypatch, tmp_path, capsys):
    """The hydra form through a (stubbed) resident pass: the designs land at `<inference.output_prefix>_<i>.pdb` from the typed design_startnum,
    trajectories under `<dir>/traj/`, exactly upstream's layout — no `<case>/` directory, no `des_` renaming; the run record (manifest, log,
    timings) beside them; the ready line and the outputs census read the same prefix."""
    good_box(monkeypatch)
    fake = tmp_path / "fake_driver.py"
    fake.write_text(fake_driver_source())
    monkeypatch.setenv("STUBS", os.path.dirname(__file__))
    monkeypatch.setattr(stack, "driver_command", lambda res, cases, out, tag, rfd=None, weights=None, python=None, compose_log=None, compose=None: [sys.executable, str(fake), "--cases", cases, "--out", out, "--tag", tag, "--no-traj", res.settings.driver_flags[1]])
    run = tmp_path / "run"
    ov = _hydra(tmp_path, run / "binder")                                                     # inference.num_designs=2 inference.design_startnum=3
    rc, man = design.run("exact", ov, tag="t")
    err = capsys.readouterr().err
    assert rc == 0 and man["status"] == "ok" and man["out_dir"] == str(run), (man.get("status"), err[-600:])
    assert sorted(f for f in os.listdir(run) if not f.startswith("traj")) == ["binder_3.pdb", "binder_3.trb", "binder_4.pdb", "binder_4.trb", "opt_manifest.json", "t.log", "t_timings.json"]
    assert sorted(os.listdir(run / "traj")) == ["binder_3_Xt-1_traj.pdb", "binder_3_pX0_traj.pdb", "binder_4_Xt-1_traj.pdb", "binder_4_pX0_traj.pdb"]
    assert man["outputs"]["n_pdb"] == 2 and man["outputs"]["cases"]["binder"]["prefix"] == str(run / "binder") and man["outputs"]["cases"]["binder"]["indices"] == [3, 4]
    dp = man["driver_passes"][0]
    assert "first_design" not in dp and "[rfdiffusion1-opt] ready " not in err
    assert "[rfdiffusion1-opt] RUN mode=exact cases=1 designs=2 process=resident " in err and man["cases"][0]["prefix"] == str(run / "binder")
    m = json.load(open(run / "opt_manifest.json"))
    assert m["overrides"] == ov and m["cases"][0]["startnum"] == 3 and m["status"] == "ok"
    stack.reset_for_tests(); report.reset_for_tests(); capsys.readouterr()
    ov2 = _hydra(tmp_path, tmp_path / "run2" / "sub" / "d")                                  # a prefix in a directory that does not exist yet: made, as upstream makes it; the record beside the designs, nowhere else
    rc, man2 = design.run("exact", ov2, tag="t")
    assert rc == 0 and man2["status"] == "ok" and man2["out_dir"] == str(tmp_path / "run2" / "sub")
    assert sorted(f for f in os.listdir(tmp_path / "run2" / "sub") if f != "traj") == ["d_3.pdb", "d_3.trb", "d_4.pdb", "d_4.trb", "opt_manifest.json", "t.log", "t_timings.json"]
    stack.reset_for_tests(); report.reset_for_tests(); capsys.readouterr()
    monkeypatch.setenv("N_OUT", "1")                                                          # a pass that writes fewer designs than typed: incomplete, exit 1, counted at the prefix
    rc, man3 = design.run("exact", _hydra(tmp_path, tmp_path / "run3" / "d"), tag="t")
    assert rc == 1 and man3["status"] == "incomplete" and man3["incomplete"] == "1/2" and man3["outputs_note"] == "expected 2 designs, found 1"


def test_stubbed_driver_missing_evidence_is_partial(monkeypatch, tmp_path, capsys):
    """The exit rule: a lever without its evidence line is a partial activation — exit 3 with the family line, the partial levers by name in
    the manifest; `allow_partial` (the flag, or its environment form) turns the same pass into exit 0 with the allowance recorded and the
    PARTIAL allowed line; the incomplete verdict wins over the allowance."""
    good_box(monkeypatch)
    fake = tmp_path / "fake_driver.py"
    fake.write_text(fake_driver_source(evidence="print('fastpath levers: []')", final=""))          # C1's line only; the timings file carries the precision record and no lever counters
    monkeypatch.setenv("STUBS", os.path.dirname(__file__))
    monkeypatch.setattr(stack, "driver_command", lambda res, cases, out, tag, rfd=None, weights=None, python=None, compose_log=None, compose=None: [sys.executable, str(fake), "--cases", cases, "--out", out, "--tag", tag])
    p = os.path.join(stack.kit_dir(KIT_BASE), "tests", "cases_public.json")
    rc, man = run_cases(p, str(tmp_path / "out"), "exact", num_designs=1, tag="t")
    assert rc == report.EXIT_NOT_ACTIVE == 3 and man["status"] == "partial" and set(man["levers_missing"]) == {"P", "E_einsum", "W1", "IO1"} and "allow_partial" not in man
    assert man["partial"] == ["P", "E_einsum", "W1", "IO1"] and man["exit_code"] == 3 and man["incomplete"] is None
    assert man["partial_reason"] == "pass t: levers without their evidence line ['P', 'E_einsum', 'W1', 'IO1']; forbidden lines 0"
    err = capsys.readouterr().err
    line = "[rfdiffusion1-opt] NOT ACTIVE: partial activation — P,E_einsum,W1,IO1: pass t: levers without their evidence line ['P', 'E_einsum', 'W1', 'IO1']; forbidden lines 0; exit 3"
    assert line in err.splitlines() and report.NOT_ACTIVE_PARTIAL_RE.search(err).group("code") == "3"
    assert json.load(open(tmp_path / "out" / "opt_manifest.json"))["partial"] == ["P", "E_einsum", "W1", "IO1"]
    rep = stack.status()
    assert rep["partial"] is True and rep["levers_unavailable"] == ["E_einsum", "IO1", "P", "W1"] and rep["levers_applied"] == ["C1"]
    import inspect
    assert "allow_partial" not in inspect.signature(design.run).parameters                   # no opt-out: a partial pass is exit 3 with its outputs kept, never a recorded 0
    # outputs short of the request: incomplete (exit 1) wins over partial, the partial record kept beside it
    monkeypatch.setenv("N_OUT", "0")
    rc4, man4 = run_cases(p, str(tmp_path / "out4"), "exact", num_designs=1, tag="t")
    assert rc4 == 1 and man4["status"] == "incomplete" and man4["partial"] == ["P", "E_einsum", "W1", "IO1"] and man4["incomplete"] == f"0/{len(man4['cases'])}"
    err = capsys.readouterr().err
    assert "[rfdiffusion1-opt] PARTIAL recorded: P,E_einsum,W1,IO1: " in err and "(status incomplete, exit 1)" in err and "NOT ACTIVE" not in err


def test_warm_carries_the_design_pass_exit_code(monkeypatch, tmp_path, capsys):
    """warm's status and exit code are the design pass's own (partial → 3 with the family line, not a collapsed 'failed' 1); its
    --allow-partial is not a flag of warm (argparse refuses it, exit 2)."""
    from rfdiffusion1_opt import warm, cli
    good_box(monkeypatch)
    fake = tmp_path / "fake_driver.py"
    fake.write_text(fake_driver_source(evidence="print('fastpath levers: []')", final=""))
    monkeypatch.setenv("STUBS", os.path.dirname(__file__))
    monkeypatch.setattr(stack, "driver_command", lambda res, cases, out, tag, rfd=None, weights=None, python=None, compose_log=None, compose=None: [sys.executable, str(fake), "--cases", cases, "--out", out, "--tag", tag])
    res = warm.run("exact", str(tmp_path / "w"))
    assert res["status"] == "partial" and res["rc"] == 3 and res["first_designs"]["status"] == "partial" and "allow_partial" not in res
    assert res["first_designs"]["n_pdb"] == len(json.load(open(os.path.join(stack.kit_dir(KIT_BASE), "tests", "cases_public.json"))))   # no --input: the base kit's public cases, one design each
    assert report.NOT_ACTIVE_PARTIAL_RE.search(capsys.readouterr().err)
    a = cli.build_parser().parse_args(["warm", "--mode", "exact", "--out_dir", str(tmp_path / "w2")])
    assert cli.cmd_warm(a) == 3 and report.NOT_ACTIVE_PARTIAL_RE.search(capsys.readouterr().err)
    with pytest.raises(SystemExit) as ex:                                                                     # no opt-out: --allow-partial is not a flag of warm (argparse, exit 2)
        cli.build_parser().parse_args(["warm", "--mode", "exact", "--out_dir", str(tmp_path / "w3"), "--allow-partial"])
    assert ex.value.code == 2 and not (tmp_path / "w3").exists()
    stack.reset_for_tests(); report.reset_for_tests(); capsys.readouterr()
    res = warm.run("exact", str(tmp_path / "w4"))                                                            # the WARM line: status, mode, count, out
    assert res["status"] == "partial" and f"[rfdiffusion1-opt] WARM partial mode=exact first_designs={res['first_designs']['n_pdb']} out={tmp_path / 'w4'}" in capsys.readouterr().err


def test_driver_death_before_any_lever_is_not_active(monkeypatch, tmp_path, capsys):
    """A driver child that dies (an import failure) before any lever printed its applied-line: activation failed in the driver — the NOT ACTIVE
    line and exit 3, never a plain failure (exit 1) behind an ACTIVE plan line; a driver that dies AFTER its levers applied stays `failed` (1)."""
    good_box(monkeypatch)
    fake = tmp_path / "dies_at_import.py"
    fake.write_text("import sys\nsys.stderr.write('Traceback (most recent call last)\\nModuleNotFoundError: no module named dgl\\n'); sys.exit(1)\n")
    monkeypatch.setattr(stack, "driver_command", lambda res, cases, out, tag, rfd=None, weights=None, python=None, compose_log=None, compose=None: [sys.executable, str(fake), "--cases", cases, "--out", out, "--tag", tag])
    p = os.path.join(stack.kit_dir(KIT_BASE), "tests", "cases_public.json")
    rc, man = run_cases(p, str(tmp_path / "out"), "exact", num_designs=1, tag="t")
    assert rc == report.EXIT_NOT_ACTIVE == 3 and man["status"] == "failed" and man["exit_code"] == 3, man
    assert man["reason"].startswith("pass t: the driver exited 1 before any lever applied — activation failed in the driver (log ")
    err = capsys.readouterr().err
    assert "[rfdiffusion1-opt] NOT ACTIVE: pass t: the driver exited 1 before any lever applied" in err
    assert "[rfdiffusion1-opt] PLAN mode=exact " in err and "[rfdiffusion1-opt] ACTIVE " not in err      # a driver that never applied anything never leaves an ACTIVE line
    late = tmp_path / "dies_late.py"
    late.write_text("import sys\nprint('fastpath levers: []'); print('prep lever: active = True'); print('einsum lever E: {}'); print('cudagraph mode 1 : blocks wrapped = 36'); sys.exit(7)\n")
    monkeypatch.setattr(stack, "driver_command", lambda res, cases, out, tag, rfd=None, weights=None, python=None, compose_log=None, compose=None: [sys.executable, str(late), "--cases", cases, "--out", out, "--tag", tag])
    rc2, man2 = run_cases(p, str(tmp_path / "out2"), "exact", num_designs=1, tag="t")
    assert rc2 == report.EXIT_FAIL == 1 and man2["status"] == "failed" and man2["reason"] == "pass t: the driver process exited 7"


def test_weights_are_recorded_never_refused(monkeypatch, tmp_path, capsys):
    """A user's weights always run: manifest.checkpoint_record names the checkpoint directory and every checkpoint file the pass loads (the
    default one, a case's `ckpt` under the directory, a typed `inference.ckpt_override_path`) with `present` — no digest, no pin word, nothing
    announced or gated; a missing weights directory is the one refusal (stack's gate), a missing file inside it is upstream's to report."""
    from rfdiffusion1_opt import manifest
    zoo = tmp_path / "zoo"; zoo.mkdir()
    (zoo / "Complex_base_ckpt.pt").write_bytes(b"x"); (zoo / "mine.pt").write_bytes(b"y")
    assert manifest.default_checkpoint() == "Complex_base_ckpt.pt" == stack.pins()["weights"]["checkpoint"]
    rec = manifest.checkpoint_record(str(zoo), [{"name": "a", "ckpt": "mine.pt"}, {"name": "b", "ckpt": "gone.pt"}, {"name": "c"}, {"name": "d", "ckpt_path": str(zoo / "mine.pt")}])
    assert rec == {"dir": str(zoo), "files": {"Complex_base_ckpt.pt": {"present": True}, "gone.pt": {"present": False}, "mine.pt": {"present": True},
                                              str(zoo / "mine.pt"): {"present": True, "override_path": True}}}
    assert manifest.checkpoint_record(None, [{"name": "c"}]) == {"dir": None, "files": {"Complex_base_ckpt.pt": {"present": False}}}
    for gone in ("announce_weights", "pinned_weights"):
        assert not hasattr(manifest, gone), gone
    assert not hasattr(report, "weights_line")
    # through the design route: nothing about the weights is printed, and the pass proceeds (exit 0 on the dry run)
    good_box(monkeypatch, weights=str(zoo))
    p = os.path.join(stack.kit_dir(KIT_BASE), "tests", "cases_public.json")
    rc, man = run_cases(p, str(tmp_path / "o"), "exact", dry_run=True, python="python")
    assert rc == 0 and man["status"] == "dry-run" and man["weights"] == {"dir": str(zoo), "files": {"Complex_base_ckpt.pt": {"present": True}}}
    err = capsys.readouterr().err
    assert "weights" not in err.replace("--weights", "") and "NOT ACTIVE" not in err and not man["activation"].get("would_refuse")
    typed = _hydra(tmp_path, tmp_path / "r" / "d", f"inference.ckpt_override_path={zoo / 'mine.pt'}", f"inference.model_directory_path={zoo}")
    rc, man_h = design.run("exact", typed, dry_run=True, python="python")            # the hydra form: the typed checkpoint override recorded by path, composed onto the driver line verbatim
    assert rc == 0 and man_h["weights"]["files"][str(zoo / "mine.pt")] == {"present": True, "override_path": True} and man_h["weights"]["dir"] == str(zoo)
    assert f"inference.ckpt_override_path={zoo / 'mine.pt'}" in man_h["driver_cmd"] and man_h["activation"]["weights"] == str(zoo)


def test_counter_defects_gate_the_kits_final_counters(tmp_path):
    """design.COUNTER_RULES: W1's counters must show no eager fallback, no capture error, no replay mismatch and >= 1 capture; K2's must
    show the Triton kernel ran and no degraded-path reason; a counter absent or an unreadable file is a defect; levers without rules add none."""
    p = tmp_path / "t.json"
    ok = {"fullgraph_final": {"n_capture": 2, "capture_errors": [], "eager_fallback_calls": 0}, "triton_ln_final": {"n_triton": 812, "n_fallback": 40, "reason": ""}}
    p.write_text(json.dumps(ok))
    assert design.counter_defects(str(p), ["C1", "P", "E_einsum", "W1", "T2", "K2", "TF32"]) == []          # K2's n_fallback is routing (small LayerNorms stay on torch by rule), not degradation
    bad = json.loads(json.dumps(ok)); bad["fullgraph_final"].update(eager_fallback_calls=40, capture_errors=["fullgraph: RuntimeError: x"])
    p.write_text(json.dumps(bad))
    d = design.counter_defects(str(p), ["W1", "K2"])
    assert d == ["W1: forward calls served eagerly after a failed capture (fullgraph_final.eager_fallback_calls=40)", "W1: CUDA-graph capture errors (fullgraph_final.capture_errors=['fullgraph: RuntimeError: x'])"]
    bad = json.loads(json.dumps(ok)); bad["triton_ln_final"].update(n_triton=0, reason="import failed"); del bad["fullgraph_final"]["n_capture"]
    p.write_text(json.dumps(bad))
    d = design.counter_defects(str(p), ["W1", "K2"])
    assert d == ["W1: no graph was captured — counter absent (fullgraph_final.n_capture; the pass did not write its final counters)",
                 "K2: no LayerNorm call ran the Triton kernel (triton_ln_final.n_triton=0)", "K2: the lever recorded a degraded path (triton_ln_final.reason='import failed')"]
    assert design.counter_defects(str(tmp_path / "absent.json"), ["W1"])[0].startswith("W1: final counters unreadable (")
    assert design.counter_defects(str(tmp_path / "absent.json"), ["C1", "P"]) == [] and set(design.COUNTER_RULES) == {"W1", "K2"}


def test_numerics_declaration_follows_the_levers_and_the_torch_readback_is_gated():
    """fast declares tf32 (matmul + cuDNN TF32 on); a line without lever TF32 (exact, the served exact line) declares fp32_strict; the driver's per-case
    read-back must equal the declaration — a mismatch either way (TF32 on under exact, off under fast) is a defect line of the pass."""
    from rfdiffusion1_opt.tests._stubs import PRECISION_FP32, PRECISION_TF32
    fast, exact = modes.resolve("fast"), modes.resolve("exact")
    assert modes.numerics("fast", levers=fast.levers)["policy"] == "tf32" and modes.numerics("fast", levers=modes.resolve("exact").levers)["policy"] == "fp32_strict"
    n = modes.numerics("fast", dict(PRECISION_TF32), levers=fast.levers)
    assert n["source"] == "torch" and n["mismatch"] == [] and design.numerics_defects(n) == []
    n = modes.numerics("fast", dict(PRECISION_FP32), levers=fast.levers)                                        # --tf32 1 planned, torch says off: the lever did not act
    assert n["mismatch"] == ["matmul_tf32", "cudnn_tf32"] and design.numerics_defects(n) == ["numerics: matmul_tf32 recorded False, the line declares True", "numerics: cudnn_tf32 recorded False, the line declares True"]
    n = modes.numerics("exact", dict(PRECISION_TF32), levers=exact.levers)                                      # exact never runs TF32
    assert n["mismatch"] == ["matmul_tf32", "cudnn_tf32"] and design.numerics_defects(n) == ["numerics: matmul_tf32 recorded True, the line declares False", "numerics: cudnn_tf32 recorded True, the line declares False"]
    assert design.numerics_defects(modes.numerics("exact", dict(PRECISION_FP32), levers=exact.levers)) == []
    part = {k: v for k, v in PRECISION_FP32.items() if k != "cudnn_tf32"}                                       # a readable record that lacks a TF32 key: unproven, a defect (never silently green)
    n = modes.numerics("exact", part, levers=exact.levers)
    assert n["source"] == "torch" and n["missing"] == ["cudnn_tf32"] and design.numerics_defects(n) == ["numerics: cudnn_tf32 absent from the driver's torch record — the line's TF32 state is unproven"]
    assert design.numerics_defects(modes.numerics("fast", {"param_dtype": "torch.float32"}, levers=fast.levers)) == [
        "numerics: matmul_tf32 absent from the driver's torch record — the line's TF32 state is unproven", "numerics: cudnn_tf32 absent from the driver's torch record — the line's TF32 state is unproven"]
    assert design.numerics_defects(None) == ["numerics: the driver's torch numerics record is unreadable (no per-case `precision` block) — the line's TF32 state is unproven"]
    assert design.numerics_defects(modes.numerics("off")) == [] and design.numerics_defects(modes.numerics("exact")) == []      # a declaration alone (no torch record) is not judged here



def test_io1_counters_gate_and_the_io_census_line(tmp_path, monkeypatch, capsys):
    """Lever IO1's exit counters (driver_run's PDBIO_FINAL line in the driver log) are a gate of every pass that carries the lever: a byte
    mismatch against upstream's writer, no confirmed call, or the line absent is a forbidden line (partial, exit 3); a clean pass prints
    `IO pass=<tag> io=numpy confirmed=<n> mismatches=0 …` and records the counters in the manifest."""
    log = tmp_path / "d.log"
    log.write_text("x\n[rfdiffusion1-opt.driver_run] PDBIO_FINAL " + json.dumps(dict(armed=True, n_calls=9, n_verified=2, n_mismatch=0, mismatches=[], seconds=0.4)) + "\n")
    c = design.io_counters(str(log))
    assert c["n_verified"] == 2 and design.io_defects(c, ["C1", "IO1"]) == [] and design.io_defects(None, ["C1"]) == [] and design.io_defects(c, ["C1"]) == []
    assert design.io_defects(None, ["IO1"]) == ["IO1: final counters absent (PDBIO_FINAL; the pass did not reach the writer's exit line)"]
    bad = dict(c, n_mismatch=1, mismatches=["writepdb((8, 4, 3):torch.float32,...)"], n_verified=0)
    assert design.io_defects(bad, ["IO1"]) == ["IO1: bytes unlike rfdiffusion.util's writer (PDBIO_FINAL.n_mismatch=1 on ['writepdb((8, 4, 3):torch.float32,...)'])",
                                              "IO1: no call confirmed against rfdiffusion.util's writer (PDBIO_FINAL.n_verified=0)"]
    assert report.io_line("t", c) == "[rfdiffusion1-opt] IO pass=t io=numpy confirmed=2 mismatches=0 calls=9 seconds=0.40" and report.io_line("t", None) == "[rfdiffusion1-opt] IO pass=t io=unreadable"
    # through the design entry: a clean stub pass prints the IO line and records the counters; a stub reporting a mismatch is partial (exit 3) with the defect named
    good_box(monkeypatch)
    fake = tmp_path / "fake_driver.py"
    fake.write_text(fake_driver_source())
    monkeypatch.setenv("STUBS", os.path.dirname(__file__))
    monkeypatch.setattr(stack, "driver_command", lambda res, cases, out, tag, rfd=None, weights=None, python=None, compose_log=None, compose=None: [sys.executable, str(fake), "--cases", cases, "--out", out, "--tag", tag, "--no-traj", res.settings.driver_flags[1]])
    p = os.path.join(stack.kit_dir(KIT_BASE), "tests", "cases_public.json")
    rc, man = run_cases(p, str(tmp_path / "out"), "exact", tag="t", num_designs=1)
    err = capsys.readouterr().err
    assert rc == 0 and man["status"] == "ok" and man["driver_passes"][0]["io"]["n_verified"] == 2, (man.get("status"), err[-600:])
    assert "[rfdiffusion1-opt] IO pass=t io=numpy confirmed=2 mismatches=0 calls=3 seconds=0.10" in err.splitlines()
    stack.reset_for_tests(); report.reset_for_tests(); good_box(monkeypatch)
    fake.write_text(fake_driver_source().replace("n_mismatch=0", "n_mismatch=1").replace("mismatches=[]", "mismatches=['writepdb(...)']"))
    rc, man = run_cases(p, str(tmp_path / "out2"), "exact", tag="t", num_designs=1)
    err = capsys.readouterr().err
    assert rc == report.EXIT_NOT_ACTIVE == 3 and man["status"] == "partial" and man["forbidden_lines"] == 1, (man.get("status"), err[-600:])
    assert man["driver_passes"][0]["evidence"]["forbidden"] == ["IO1: bytes unlike rfdiffusion.util's writer (PDBIO_FINAL.n_mismatch=1 on ['writepdb(...)'])"]
    assert "[rfdiffusion1-opt] IO pass=t io=numpy confirmed=2 mismatches=1 calls=3 seconds=0.10" in err.splitlines()
    stack.reset_for_tests(); report.reset_for_tests()


# ------------------------------------------------------------------------------------------------------------------------------- OOM propagates
def test_is_oom_is_the_one_classifier():
    """The core's classifier, imported — never re-spelled: torch's class (by isinstance or by name), the CUDA message, MemoryError; not a plain error."""
    assert design.is_oom is is_oom
    standin = type("OutOfMemoryError", (RuntimeError,), {})                                  # torch.cuda.OutOfMemoryError's name, constructible without torch
    assert is_oom(standin("CUDA out of memory (mock)")) and is_oom(standin("")) and is_oom(MemoryError())
    assert is_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")) and is_oom(RuntimeError("x")) is False and is_oom(ValueError("capture failed")) is False
    try:
        import torch  # noqa: F401
    except ImportError:
        torch = None
    if torch is not None:                                                                     # the real class where torch is installed (constructible with a message, no GPU needed)
        assert is_oom(torch.cuda.OutOfMemoryError("CUDA out of memory (mock)"))
    wrapped = RuntimeError("capture failed")
    wrapped.__cause__ = standin("CUDA out of memory (mock)")
    assert is_oom(wrapped)                                                                    # one level down: a wrapper that re-raised an OOM as something else
    src = {f: open(os.path.join(PKG, f), encoding="utf-8").read() for f in os.listdir(PKG) if f.endswith(".py")}
    assert [f for f, s in src.items() if re.search(r"out of memory|OutOfMemoryError", s, re.I) and f != "design.py"] == []   # no second classifier anywhere in the package (serve.py names the launcher's own OOM-scan line, not a classifier)
    assert "from opt_core.oom import is_oom" in src["design.py"] and not re.search(r"def is_oom|cuda out of memory", src["design.py"], re.I)


def test_an_oom_line_in_a_driver_log_is_forbidden(tmp_path):
    log = tmp_path / "t.log"
    log.write_text("fastpath levers: ['chain_breaks']\nfullgraph mode: applied = True (2 graphs)\n"
                   "FGSTATS_FINAL t {\"n_replay\": 1, \"capture_errors\": [\"fullgraph: OutOfMemoryError: CUDA out of memory. Tried to allocate 1.2 GiB (mock)\"], \"eager_fallback_calls\": 40}\n"
                   "FGSTATS_FINAL u {\"capture_errors\": [\"fullgraph: RuntimeError: capture failed (mock)\"]}\n")
    ev = design.read_evidence(str(log), ["W1", "P"])
    assert ev["applied"] == ["W1"] and ev["missing"] == ["P"] and len(ev["forbidden"]) == 1 and "CUDA out of memory" in ev["forbidden"][0]


def _fake_oom_driver(tmp_path, capture_error):
    """A driver that prints every evidence line of the exact row, writes the outputs, and records one capture error in its final counters."""
    fake = tmp_path / "fake_driver.py"
    st = {"n_replay": 3, "n_capture": 1, "capture_errors": [capture_error], "eager_fallback_calls": 40}
    fake.write_text(fake_driver_source(
        final=f"fullgraph_final={st!r}",
        extra=(f"if {capture_error!r} == 'RAISE': raise type('OutOfMemoryError', (RuntimeError,), {{}})('CUDA out of memory. Tried to allocate 1.20 GiB (mock)')\n"
               f"print('FGSTATS_FINAL', tag, json.dumps({st!r}), flush=True)\n")))
    return fake


@pytest.mark.parametrize("capture_error, rc_expected, forbidden", [
    ("fullgraph: OutOfMemoryError: CUDA out of memory. Tried to allocate 1.20 GiB (mock)", 3, 3),   # an OOM the lever's handler rerouted to eager: the OOM line + the counters (capture_errors, eager_fallback_calls): never a pass
    ("fullgraph: RuntimeError: capture failed (mock)", 3, 2),                                        # a non-OOM capture error: the lever served the forwards eagerly — its final counters name the degraded path (design.COUNTER_RULES): partial, never silent
    ("RAISE", 1, 0),                                                                                 # an OOM the driver re-raised (the carried handlers' first statement): the pass fails
])
def test_oom_met_by_a_driver_propagates_through_the_design_entry(monkeypatch, tmp_path, capsys, capture_error, rc_expected, forbidden):
    stack.reset_for_tests(); report.reset_for_tests()
    good_box(monkeypatch, weights=str(tmp_path / "no_weights"))
    fake = _fake_oom_driver(tmp_path, capture_error)
    monkeypatch.setenv("STUBS", os.path.dirname(__file__))
    monkeypatch.setattr(stack, "driver_command", lambda res, cases, out, tag, rfd=None, weights=None, python=None, compose_log=None, compose=None:
                        [sys.executable, str(fake), "--cases", cases, "--out", out, "--tag", tag, "--no-traj", res.settings.driver_flags[1]])
    p = os.path.join(stack.kit_dir(KIT_BASE), "tests", "cases_public.json")
    rc, man = run_cases(p, str(tmp_path / "out"), "exact", tag="t", num_designs=2)
    err = capsys.readouterr().err
    assert rc == rc_expected, (man.get("status"), err[-600:])
    if rc_expected == 3:
        assert man["status"] == "partial" and man["forbidden_lines"] == forbidden and man["levers_missing"] == []
        assert "[rfdiffusion1-opt] NOT ACTIVE: partial activation" in err and f"forbidden lines {forbidden}" in err and f"[rfdiffusion1-opt] EVIDENCE levers=C1,P,E_einsum,W1,IO1 missing=none forbidden={forbidden}" in err
    elif rc_expected == 1:
        assert man["status"] == "failed" and man["forbidden_lines"] >= 2 and "NOT ACTIVE" not in err.split("[rfdiffusion1-opt] EVIDENCE")[0]   # the traceback and the OOM line; a failed process, exit 1
    else:
        assert man["status"] == "ok" and man["forbidden_lines"] == 0 and "NOT ACTIVE" not in err
    stack.reset_for_tests(); report.reset_for_tests()


def test_the_capture_reroute_re_raises_an_oom_first():
    """The whole-forward graph's reroute (W1: a failed CUDA-graph capture -> eager, recorded in capture_errors) guards its reroute with the
    core's classifier first: the module imports opt_core.oom.is_oom once and the capture handler's first statement is `if is_oom(e): raise`
    — an out-of-memory is never absorbed into an eager-fallback reroute."""
    for rel in CAPTURE_REROUTE_SITES:
        src = open(os.path.join(FORWARD, rel), encoding="utf-8").read().splitlines()
        assert sum(ln.strip() == "from opt_core.oom import is_oom" for ln in src) == 1, rel
        handlers = [n for n, ln in enumerate(src) if ln.strip() == "except Exception as e:" and any("capture_errors" in l2 for l2 in src[n + 1:n + 4])]
        assert len(handlers) == 1, (rel, handlers)                                                # one capture-failure handler in the module
        body = next(ln.strip() for ln in src[handlers[0] + 1:] if ln.strip() and not ln.strip().startswith("#"))
        assert body == "if is_oom(e): raise", (rel, body)
