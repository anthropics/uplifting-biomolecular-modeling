"""The command layer (usage, selection rules, check's exit codes), the exit tally (never silent on a partial application), the settings
presets (the settings rendered verbatim), the deterministic recipe, the query -> pack adapter (representable vs refused by
name) and the outputs index."""
import argparse
import json
import os
import sys
import types

import pytest

from opendde_opt import cli, det, inputs, modes, outputs, report, settings, stack, warm
from opendde_opt.tests import _stubs


@pytest.fixture
def clean(tmp_path, monkeypatch):
    site = _stubs.make_site(str(tmp_path))
    monkeypatch.syspath_prepend(site)
    import importlib
    importlib.invalidate_caches()
    for k in list(os.environ):
        if k.startswith(("ODDE_", "OPENDDE_OPT", "DIT_", "CUEQ_", "CUBLAS_", "PYTORCH_CUDA")):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("MODEL_OPT", _stubs.TREE)
    monkeypatch.setattr(stack, "_REPORT", None)
    return site


# ----------------------------------------------------------------------------------------------------------------- cli
def test_usage_and_unknown_command(capsys):
    assert cli.main([]) == cli.EXIT_USAGE
    assert cli.main(["help"]) == cli.EXIT_OK
    assert cli.main(["bogus"]) == cli.EXIT_USAGE
    assert "usage:" in capsys.readouterr().out


def test_selection_rules(monkeypatch):
    monkeypatch.delenv("OPENDDE_OPT", raising=False)
    assert cli.split_selection(["--mode", "exact", "-i", "q"]) == ("exact", ["-i", "q"])
    assert cli.split_selection(["--line=LSTAR2A"]) == (modes.DEFAULT_MODE, ["--line=LSTAR2A"])   # no line selector: an unknown flag, left for the verb's parser to refuse
    assert cli.split_selection([]) == (modes.DEFAULT_MODE, [])
    monkeypatch.setenv("OPENDDE_OPT", "exact")
    assert cli.split_selection(["--mode", "exact"]) == ("exact", [])
    with pytest.raises(cli.CliError):
        cli.split_selection(["--mode", "off"])                              # disagrees with OPENDDE_OPT
    monkeypatch.delenv("OPENDDE_OPT")
    with pytest.raises(ValueError):
        cli.split_selection(["--mode", "quick"])


def test_check_exit_codes(clean, capsys):
    assert cli.main(["check", "--mode", "fast", "--json"]) == cli.EXIT_OK
    rep = json.loads(capsys.readouterr().out)
    assert rep["dry_run"] and rep["line"].startswith("LSTAR2A(") and "ODDE_ARM_U=1" in rep["line"] and "ODDE_ARM_U2_TRIMUL=1" in rep["line"]
    stack._REPORT = None
    assert cli.main(["check", "--mode", "exact", "--json"]) == cli.EXIT_OK
    rep = json.loads(capsys.readouterr().out)
    assert rep["dry_run"] and rep["line"].startswith("S1(") and "ODDE_SERVED_LEVERS" not in os.environ
    stack._REPORT = None
    assert cli.main(["check", "--mode", "off"]) == cli.EXIT_OK
    stack._REPORT = None
    with pytest.raises(ValueError, match="unknown mode 'nope'"):                    # an unknown mode word (no line selector exists)
        cli.main(["check", "--mode", "nope"])


# ----------------------------------------------------------------------------------------------------------------- exit tally
def test_exit_tally_never_silent(monkeypatch):
    for m in ("odde_served_levers", "odde_arm_t", "odde_addon"):
        monkeypatch.delitem(sys.modules, m, raising=False)
    from opendde_opt import alloc                                                    # the premise: a process where no lever acted (the tree's own levers' state cleared too)
    monkeypatch.setattr(alloc, "STATE", {"decision": None})
    monkeypatch.setattr(stack, "_REPORT", None)
    assert "no lever counters" in report.exit_tally_line(1)
    m = types.ModuleType("odde_served_levers")
    m.STATE = {"version": "0.1.1", "active": True, "wrapped": True, "installs": [{"levers": {"ditfast": {"ok": 1}}}], "errors": {"odde_arm_t": "boom"}}
    monkeypatch.setitem(sys.modules, "odde_served_levers", m)
    monkeypatch.setattr(stack, "_REPORT", {"active": True, "partial": True, "levers_fallback": ["arm"]})
    line = report.exit_tally_line(1)
    assert line.startswith("[opendde-opt] EXIT pid=1 n_gpu=1 sharding=none served_levers={") and "n_installs=1" in line and "PARTIAL fallbacks=arm" in line


def test_activation_line_shapes():
    assert report.activation_line({"active": True, "mode": "exact", "line": "S(x)", "levers_applied": ["a", "b"], "package_version": "0.1.0"}) == \
        "[opendde-opt] ACTIVE mode=exact line=S(x) levers=a,b n_gpu=1 sharding=none package=0.1.0"
    assert report.activation_line({"active": False, "mode": "fast", "reason": "OPEN"}) == "[opendde-opt] NOT ACTIVE mode=fast reason=OPEN"
    assert report.activation_line({"active": False, "dry_run": True, "mode": "exact", "levers_planned": ["x"], "partial": True}).startswith("[opendde-opt] DRY RUN mode=exact levers=x PARTIAL")


# ----------------------------------------------------------------------------------------------------------------- settings / det
def test_upstreams_flags_pass_through_verbatim_and_absent_ones_stay_absent():
    """The upstream knobs on `pred`: stated -> handed to the engine exactly as stated, in settings.FLAGS order; not stated -> not passed
    (upstream's default applies, PINS cli_defaults); `effective` = stated, else the last pass-through occurrence, else upstream's default."""
    p = argparse.ArgumentParser(prog="t")
    cli._common_pred(p)
    base = ["-i", "q.json", "-o", "out"]
    a0 = p.parse_args(base)
    assert settings.stated(a0) == {} and settings.stated_args(a0) == []            # nothing stated: nothing passed
    d = settings.cli_defaults(_stubs.TREE)
    assert d["sample"] == 5 and d["dtype"] == "fp32" and d["need_atom_confidence"] is True and d["device"] == "auto" and d["seeds"] is None
    eff0 = settings.effective(a0, [], _stubs.TREE)
    assert eff0 == {"cycle": "10", "step": "200", "sample": "5", "dtype": "bf16", "model_name": "opendde_v1", "use_msa": "true", "use_template": "false",
                    "use_rna_msa": "false", "need_atom_confidence": "true", "trimul_kernel": "auto", "triatt_kernel": "auto",
                    "seeds": ""}   # upstream's own defaults + the stock base's dtype; seeds '' = the JSON's modelSeeds / random (upstream decides); the kernel selectors `auto`
    assert settings.base_args(a0) == ["--dtype", "bf16"] and settings.base_args(a0, ["--dtype", "fp32"]) == []   # the base's flag is passed unless the caller states --dtype (here after `--`)
    assert settings.fields(eff0, {}) == "flags=upstream_defaults dims=cycle:10,step:200,sample:5,use_msa:true"
    full = ["--cycle", "10", "--step", "200", "--sample", "5", "--dtype", "bf16", "--model_name", "opendde_v1", "--use_msa", "true",
            "--use_template", "false", "--use_rna_msa", "false", "--need_atom_confidence", "true", "--seeds", "101"]
    a1 = p.parse_args(base + full)
    assert settings.stated_args(a1) == full and settings.base_args(a1) == []       # verbatim, FLAGS order (seeds last); --dtype stated: nothing injected
    a2 = p.parse_args(base + ["--seeds", "101", "-e", "1", "-c", "1", "-p", "2"])      # upstream's short aliases; the order handed on is FLAGS order
    assert settings.stated_args(a2) == ["--cycle", "1", "--step", "2", "--sample", "1", "--seeds", "101"]
    assert settings.fields(settings.effective(a2, [], _stubs.TREE), settings.stated(a2)) == "flags=cycle,step,sample,seeds dims=cycle:1,step:2,sample:1,use_msa:true"
    a3 = p.parse_args(base + ["--use_msa", "false"])
    assert settings.effective(a3, ["--use_msa", "true", "--sample", "2"], _stubs.TREE)["use_msa"] == "false"   # stated beats a pass-through occurrence
    assert settings.effective(a0, ["--sample", "3", "--sample", "2"], _stubs.TREE)["sample"] == "2"          # the last pass-through occurrence, as click reads them
    a4 = p.parse_args(base + ["--triatt_kernel", "torch"])                              # upstream's triangle-kernel selectors: stated on pred or after `--`, verbatim; a value other
    assert settings.stated_args(a4) == ["--triatt_kernel", "torch"] and settings.stock_knobs(settings.stated(a4)) == {"triatt_kernel": "torch"}   # than auto is a stock knob
    assert settings.stock_knobs(settings.stated(a0, ["--trimul_kernel", "torch", "--triatt_kernel", "auto"])) == {"trimul_kernel": "torch"}
    assert settings.stated_in(["pred", "-i", "q.json", "--trimul_kernel=torch", "--triatt_kernel", "cuequivariance"]) == {"trimul_kernel": "torch", "triatt_kernel": "cuequivariance"}
    assert settings.stated(a0, ["--sample", "3", "-c", "4"]) == {"cycle": "4", "sample": "3"} and settings.stated_args(a0) == []   # stated after `--` counts as stated (PRED token, manifest); it travels in the pass-through list
    assert settings.fields(settings.effective(a0, ["--seeds", "7"], _stubs.TREE), settings.stated(a0, ["--seeds", "7"])) == "flags=seeds dims=cycle:10,step:200,sample:5,use_msa:true"
    desc = settings.describe(eff0, {}, _stubs.TREE, environ={})
    assert desc["stated"] == {} and desc["effective"] == eff0 and desc["base"] == "bf16_fastln" and desc["stock_env"] == {"LAYERNORM_TYPE": "fast_layernorm"}
    for flag, (short, line) in settings.UPSTREAM.items():                            # the help cites upstream's click option and default
        h = next(x for x in p._actions if x.dest == flag).help
        assert f"runner/batch_inference.py:{line}" in h and "default when absent" in h
    assert "[upstream flags: --seeds|-s --cycle|-c --step|-p --sample|-e --dtype --model_name|-n --use_msa --use_template --use_rna_msa --need_atom_confidence --trimul_kernel --triatt_kernel]" in cli.USAGE
    assert "--settings" not in cli.USAGE and "--preset" not in cli.USAGE and "--template_mmcif_dir" in cli.USAGE


def test_warm_launches_pred_with_its_own_explicit_upstream_flags(tmp_path, monkeypatch):
    """warm's `pred` child states its item's upstream flags explicitly (warm.WARM_FLAGS: one sample, no MSA/templates, bf16) — no settings name."""
    monkeypatch.setattr(warm.subprocess, "call", lambda cmd, **kw: 0)
    r = warm.run("off", str(tmp_path / "w"))
    i = r["cmd"].index("-o"); assert r["cmd"][i + 2:] == warm.WARM_FLAGS and "--settings" not in r["cmd"] and r["status"] == "FAIL"   # nothing ran: no manifest
    assert warm.WARM_FLAGS[warm.WARM_FLAGS.index("--sample") + 1] == "1" and warm.WARM_FLAGS[warm.WARM_FLAGS.index("--use_msa") + 1] == "false"
    assert warm.WARM_QUERY[0]["sequences"][0]["proteinChain"]["sequence"] == "ACDEFGHIK"


def test_det_recipe():
    assert det.cli_args(0) == [] and det.env(0) == {}
    assert det.cli_args(1) == ["--deterministic", "true"]
    assert det.env(1) == {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    with pytest.raises(ValueError):
        det.check_level(2)


# ----------------------------------------------------------------------------------------------------------------- inputs / outputs
def test_query_loader(tmp_path):
    q = tmp_path / "q.json"; q.write_text(json.dumps(warm.WARM_QUERY))
    assert inputs.describe_query(str(q))["items"] == ["tiny"]
    d = tmp_path / "d.json"; d.write_text(json.dumps({"blocks": []}))                            # a JSON object that is not a job: refused by name, no traceback later
    with pytest.raises(ValueError):
        inputs.load_query(str(d))


def test_index_cli_hashes_and_normalises(tmp_path):
    out = tmp_path / "out"
    p = out / "item1" / "seed_101" / "predictions"
    p.mkdir(parents=True)
    (p / "item1_sample_0.cif").write_text("data_item1\nx\n")
    (p / "item1_summary_confidence_sample_0.json").write_text('{"name": "item1", "iptm": 0.5}')
    for f in outputs.PACKAGE_FILES:
        (out / f).write_text("{}")                                                   # the package's own files are never indexed
    idx = outputs.index_cli(str(out))
    assert idx["n_files"] == 2
    keys = sorted(r["key"] for r in idx["files"])
    assert keys == ["ITEM/seed_101/predictions/NAME_sample_0.cif", "ITEM/seed_101/predictions/NAME_summary_confidence_sample_0.json"]
    r = next(x for x in idx["files"] if x["key"].endswith(".cif"))
    assert r["sha256"] != r["sha256_name_normalised"] and len(r["sha256"]) == 64


def test_pred_partial_exits_not_active_unless_allowed_and_recorded(clean, tmp_path, monkeypatch, capsys, weights_root):
    """A PARTIAL activation (a lever of the mode on the stock path) never reads as success on rc: `pred` exits 3 after writing its outputs
    and manifest; `--allow-partial` accepts the degraded row and is recorded in the manifest (`allow_partial`)."""
    q = tmp_path / "q.json"; q.write_text(json.dumps([{"name": "a", "sequences": [{"proteinChain": {"sequence": "MKT", "count": 1, "unpairedMsaPath": str(tmp_path / "a.a3m")}}]}]))   # a precomputed MSA path: the frozen-weights gate admits the chain under use_msa=true (the PARTIAL path is what this test asserts)
    active = {"active": True, "mode": "exact", "line": "S1(x)", "line_name": "S1", "levers_planned": ["a", "b"], "levers_applied": ["a", "b"],
              "levers_unavailable": [], "partial": False, "levers_fallback": [], "applied": {"exports": {}}, "notes": []}
    monkeypatch.setattr(stack, "activate", lambda *a, **k: dict(active))
    monkeypatch.setattr(cli, "run_stock_cli", _stubs.fake_stock_cli)   # writes the owed structure files, rc 0
    monkeypatch.setattr(stack, "applied", lambda: {})
    partial = dict(active, partial=True, levers_fallback=["fpf_trimul_exact"], levers_applied=["a"])
    monkeypatch.setattr(stack, "refresh", lambda predicted=False, facts=None: dict(partial))
    out = tmp_path / "o1"
    assert cli.main(["pred", "--mode", "exact", "-i", str(q), "-o", str(out)]) == cli.EXIT_NOT_ACTIVE
    err = capsys.readouterr().out
    assert "PRED PARTIAL refused: exit 3 fallbacks=fpf_trimul_exact" in err and "partial=True" in err
    m = json.load(open(out / "opt_manifest.json"))
    assert m["activation"]["partial"] is True and m["activation"]["levers_fallback"] == ["fpf_trimul_exact"] and m["allow_partial"] is False
    out2 = tmp_path / "o2"
    assert cli.main(["pred", "--mode", "exact", "-i", str(q), "-o", str(out2), "--allow-partial"]) == cli.EXIT_OK
    m2 = json.load(open(out2 / "opt_manifest.json"))
    assert m2["allow_partial"] is True and m2["activation"]["partial"] is True                 # accepted, and the record says so
    monkeypatch.setattr(stack, "refresh", lambda predicted=False, facts=None: dict(active))
    assert cli.main(["pred", "--mode", "exact", "-i", str(q), "-o", str(tmp_path / "o3")]) == cli.EXIT_OK   # no fallback: rc 0 without the flag


def test_tally_fields_prints_every_rowpair_key_id_first():
    """report.tally_fields is the core's non-truncating formatter (opt_core.report.tally_fields, id_keys=TALLY_ID_KEYS,
    max_fields=None): tp.py's rowpair STATS (61 scalars) prints installed / armed / n_gpu / rank / group first and EVERY key — no cap, no
    overflow marker (test_lever_mechanics.py holds the grammar on a synthetic module; this holds it on the adapter's real census)."""
    from opendde_opt import report, tp
    scal = {k: v for k, v in tp.STATS.items() if not isinstance(v, (dict, list))}
    assert len(scal) == 61 and {"installed", "armed", "n_gpu", "rank"} <= set(scal), sorted(scal)      # the row-sharded adapter's scalar census (tp.STATS)
    (field,) = report.tally_fields({"rowpair_tp": dict(tp.STATS)})
    assert field.startswith("rowpair_tp={installed="), field
    lead = [kv.split("=")[0] for kv in field[len("rowpair_tp={"):-1].split(",")[:5]]
    assert lead == [k for k in report.TALLY_ID_KEYS if k in tp.STATS][:5], lead                  # equality keys first, in TALLY_ID_KEYS order
    for k in scal:
        assert f",{k}=" in field or field.startswith(f"rowpair_tp={{{k}="), (k, field)                 # nothing dropped
    assert "…+" not in field and not hasattr(report, "TALLY_MAX_FIELDS")


def test_an_unreadable_query_is_a_named_refusal_before_any_gate(tmp_path, capsys):
    """`pred -i <missing|garbled>`: `reason=input_missing:<path>` / `input_unreadable:<path> (…)`, exit 2 — before
    the stack assertion, the version gate and the activation (a raw FileNotFoundError traceback is not a refusal)."""
    missing = str(tmp_path / "nope.json")
    assert cli.main(["pred", "--mode", "fast", "-i", missing, "-o", str(tmp_path / "o")]) == cli.EXIT_USAGE
    out = capsys.readouterr().out
    assert f"PRED refused: exit 2 reason=input_missing:{missing}" in out and "STACK" not in out and "Traceback" not in out
    bad = tmp_path / "bad.json"; bad.write_text("{not json")
    assert cli.main(["pred", "--mode", "off", "-i", str(bad), "-o", str(tmp_path / "o")]) == cli.EXIT_USAGE
    out = capsys.readouterr().out
    assert f"PRED refused: exit 2 reason=input_unreadable:{bad} (" in out


# ----------------------------------------------------------------------------------------------- the finiteness census (FAIL-LOUD: NaN is never complete)
def _tree(tmp_path, summary: str, cif: str = "data_x\n_atom_site.Cartn_x 1.0\n"):
    d = tmp_path / "o" / "item1" / "seed_101" / "predictions"; d.mkdir(parents=True)
    (d / "item1_sample_0.cif").write_text(cif)
    (d / "item1_summary_confidence_sample_0.json").write_text(summary)
    return outputs.index_cli(str(tmp_path / "o"))


@pytest.mark.parametrize("summary,cif,bad", [
    ('{"plddt": 81.2, "ptm": 0.61, "iptm": 0.5, "ranking_score": 0.55}', "data_x\n1.0 2.0 3.0\n", False),
    ('{"plddt": NaN, "ptm": 0.61, "iptm": 0.5, "ranking_score": 0.55}', "data_x\n", True),          # upstream's json.dump of a NaN tensor value
    ('{"plddt": 81.2, "ptm": null, "iptm": 0.5}', "data_x\n", True),                                # null where a number is required
    ('{"plddt": 81.2, "chain_ptm": [0.4, Infinity]}', "data_x\n", True),                            # nested, infinite
    ('{"plddt": 81.2, "ptm": 0.61}', "data_x\nATOM 1 C CA . ALA A 1 1 ? nan nan nan 1.0 0.0\n", True),   # NaN coordinates in the structure
    ('{"plddt": 81.2, "ptm": 0.61}', "data_x\nHETATM 1 N NAN . NAN A 1 1 ? 1.0 2.0 3.0\n", False),      # an upper-case residue code is not a NaN
])
def test_nonfinite_census_names_the_item_and_seed(tmp_path, summary, cif, bad):
    idx = _tree(tmp_path, summary, cif)
    assert outputs.nonfinite_cli(idx) == (["item1/seed_101"] if bad else [])


def test_pred_with_a_nan_structure_is_refused_exit_3_with_its_own_sentence(clean, tmp_path, monkeypatch, capsys, weights_root):
    """Every structure file written but NaN numbers inside: `pred` exits 3 with the finiteness gate's OWN sentence and token — `PRED NONFINITE
    refused: exit 3 nonfinite_output=<item>/seed_<S>` — and the manifest's `nonfinite_outputs`; it is not a lever fallback (no `PRED PARTIAL`,
    `partial=False`); the finite run of the same query exits 0."""
    q = tmp_path / "q.json"; q.write_text(json.dumps([{"name": "a", "sequences": [{"proteinChain": {"sequence": "MKT", "count": 1, "unpairedMsaPath": str(tmp_path / "a.a3m")}}]}]))
    active = {"active": True, "mode": "fast", "line": "LSTAR(x)", "line_name": "LSTAR", "levers_planned": ["a"], "levers_applied": ["a"],
              "levers_unavailable": [], "partial": False, "levers_fallback": [], "applied": {"exports": {}}, "notes": []}
    monkeypatch.setattr(stack, "activate", lambda *a_, **k: dict(active))
    monkeypatch.setattr(stack, "refresh", lambda predicted=False, facts=None: dict(active))
    monkeypatch.setattr(stack, "applied", lambda: {})

    def nan_cli(args):
        rc = _stubs.fake_stock_cli(args)
        d = os.path.join(args[args.index("-o") + 1], "a", "seed_101", "predictions")
        open(os.path.join(d, "a_summary_confidence_sample_0.json"), "w").write('{"plddt": NaN, "ptm": NaN, "iptm": NaN, "ranking_score": NaN}')
        return rc
    monkeypatch.setattr(cli, "run_stock_cli", nan_cli)
    out = tmp_path / "o1"
    assert cli.main(["pred", "--mode", "fast", "-i", str(q), "-o", str(out)]) == cli.EXIT_NOT_ACTIVE
    txt = capsys.readouterr().out
    assert "[opendde-opt] PRED NONFINITE refused: exit 3 nonfinite_output=a/seed_101 " in txt, txt[-600:]              # the exact token
    assert "PRED PARTIAL" not in txt and "partial=False" in txt and " nonfinite=1 " in txt
    m = json.load(open(out / "opt_manifest.json"))
    assert m["nonfinite_outputs"] == ["a/seed_101"] and m["activation"]["partial"] is False and not m["activation"]["levers_fallback"]
    monkeypatch.setattr(cli, "run_stock_cli", _stubs.fake_stock_cli)                              # the finite run: unchanged, exit 0
    assert cli.main(["pred", "--mode", "fast", "-i", str(q), "-o", str(tmp_path / "o2")]) == cli.EXIT_OK


def test_arm_u_evidence_surfaces_the_transition_adapters_cells(monkeypatch):
    """The U line's pair-transition adapter (fpf_transition_odde) keeps its own census; arm_u's LEVER line and the manifest's kit_stats carry
    transition_cells_stock / transition_cells_kernel / transition_stock_reasons from it — declared domain behaviour, never a partial."""
    m = types.ModuleType("fpf_transition_odde")
    m.STATS = {"calls": 96, "kernel_calls": 90, "fallback": 6, "fallback_reasons": {"cell-128-4": 6}, "chunks": 90, "residual_calls": 0}
    monkeypatch.setitem(sys.modules, "fpf_transition_odde", m)
    monkeypatch.setitem(sys.modules, "odde_arm_t", None)                          # this test reads the transition adapter alone (the arm's TriMul census has its own test)
    st = report.kit_stats()
    assert st["arm_u_transition"] == {"transition_calls": 96, "transition_cells_kernel": 90, "transition_cells_stock": 6, "transition_stock_reasons": {"cell-128-4": 6}}
    assert report.lever_evidence("arm_u", st) == {"transition_cells_stock": 6, "transition_cells_kernel": 90, "transition_stock_reasons": "cell-128-4:6"}
    rep = {"active": True, "mode": "fast", "line_name": "LSTAR", "levers_planned": ["arm_u"], "levers_applied": ["arm_u"], "route": "cli"}
    (ln,) = [x for x in report.lever_lines(rep, st) if " name=arm_u " in x]
    assert " state=on " in ln and " transition_cells_stock=6 transition_cells_kernel=90 transition_stock_reasons=cell-128-4:6" in ln, ln
