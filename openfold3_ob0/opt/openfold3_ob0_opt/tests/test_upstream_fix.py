"""The upstream-issues layer (opt_core.upstream_fix through the package's upstream_fix_registry): the flag's parsing, an unknown ID refused by name before any launch, the plumbing's reach,
and — with no confirmed issue at the pinned OpenFold3 — an empty table: the default path loads nothing from upstream_issues/.
"""
import json
import os
import re
import subprocess
import sys
import textwrap

import pytest

from openfold3_ob0_opt import UPSTREAM_ISSUES_MODULES, modes, cli, env, manifest, stock_pred, upstream_fix_registry
from opt_core import upstream_fix
from openfold3_ob0_opt.tests import _stubs
from openfold3_ob0_opt.tests.test_off_env_proof import STUB_CLI

HOME = _stubs.tree_home()
ID = "OF3-101"                                   # an ID of the form the layer parses; no issue file carries it in this tree


# ------------------------------------------------------------------------------------------------------------------ the flag ----
def test_parse_single_list_spaces_duplicates_and_none():
    assert upstream_fix.parse(None) == [] and upstream_fix.parse("") == [] and upstream_fix.parse(" , ") == []
    assert upstream_fix.parse("OF3-101") == ["OF3-101"]
    assert upstream_fix.parse("OF3-101,OF3-102") == ["OF3-101", "OF3-102"]
    assert upstream_fix.parse(" OF3-102 , OF3-101,OF3-102 ") == ["OF3-102", "OF3-101"]                 # order kept, duplicates dropped


def test_the_tree_carries_exactly_the_confirmed_issues_and_resolve_refuses_an_unknown_id_by_name():
    reg = upstream_fix_registry(HOME)
    assert reg.ids() == []                                                                                 # no confirmed upstream issue at the pinned OpenFold3: the layer is plumbing with an empty table
    assert reg.resolve([]) == []
    with pytest.raises(upstream_fix.UnknownUpstreamFix) as ei:
        reg.resolve(["OF3-199"])
    msg = str(ei.value)
    assert "'OF3-199'" in msg and "known: none" in msg and "--upstream-fix" in msg and "Known upstream issues" in msg
    assert cli.UPSTREAM_FIX_FLAG == upstream_fix.FLAG == "--upstream-fix"


def test_the_flag_exists_on_every_verb_that_runs_the_model_and_nowhere_else():
    ap = cli.build_parser()
    pred = ["pred", "--mode", "exact", "--query-json", "q.json", "--output-dir", "o"]
    assert ap.parse_args(pred).upstream_fix is None                                                      # default: none
    assert ap.parse_args(pred + ["--upstream-fix", "OF3-101"]).upstream_fix == "OF3-101"
    assert ap.parse_args(pred + ["--upstream-fix", "OF3-101,OF3-102"]).upstream_fix == "OF3-101,OF3-102"   # parsed later (upstream_fix.parse); the parser keeps the text
    assert ap.parse_args(["warm", "--mode", "fast", "--out", "w", "--upstream-fix", ID]).upstream_fix == ID
    with pytest.raises(SystemExit):
        ap.parse_args(["check", "--mode", "exact", "--upstream-fix", ID])                                      # the dry run runs no model: no such flag


def test_pred_refuses_an_unknown_id_by_name_before_any_launch(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("OPENFOLD3_OB0_OPT", raising=False)
    before = upstream_fix.loaded(UPSTREAM_ISSUES_MODULES)
    rc = cli.main(["pred", "--mode", "exact", "--ckpt", "/w.pt", "--query-json", "q.json", "--output-dir", str(tmp_path / "o"), "--upstream-fix", "OF3-101,OF3-404"])
    err = capsys.readouterr().err
    assert rc == cli.EXIT_USAGE and "unknown upstream fix 'OF3-101'" in err and "known: none" in err, err
    assert not os.path.exists(tmp_path / "o") and upstream_fix.loaded(UPSTREAM_ISSUES_MODULES) == before                       # nothing launched, nothing loaded


def test_the_manifest_records_the_ids_under_upstream_fix():
    off = {"active": False, "mode": "off", "reason": "stock"}
    assert manifest.build(off, command="pred")["upstream_fix"] == []                                   # the default: nothing applied
    assert manifest.build(off, command="pred", upstream_fix=[ID])["upstream_fix"] == [ID]
    assert manifest.build(off, command="warm", upstream_fix=[ID, "OF3-102"])["upstream_fix"] == [ID, "OF3-102"]


# ------------------------------------------------------------------------------------------------ the stock caller's seam ----
STUB_STRUCT = "def sample_templates(assembly_data, template_cache_directory, n_templates, take_top_k, chain_id, template_structure_array_directory, template_file_format, use_roda_monomer_format=False):\n    return {}\n"
STUB_PIPELINE = "from openfold3.core.data.primitives.structure.template import sample_templates  # noqa: F401\n\ndef process_template_structures_of3(*a, **k):\n    return {}\n"


def _stub_openfold3_with_template_modules(root):
    """A stub `openfold3` distribution: the run_openfold CLI stub of the stock-route tests plus the two template modules OF3-101 wraps."""
    files = {"openfold3/__init__.py": "", "openfold3/run_openfold.py": STUB_CLI,
             "openfold3/core/__init__.py": "", "openfold3/core/data/__init__.py": "", "openfold3/core/data/primitives/__init__.py": "",
             "openfold3/core/data/primitives/structure/__init__.py": "", "openfold3/core/data/primitives/structure/template.py": STUB_STRUCT,
             "openfold3/core/data/pipelines/__init__.py": "", "openfold3/core/data/pipelines/sample_processing/__init__.py": "",
             "openfold3/core/data/pipelines/sample_processing/template.py": STUB_PIPELINE}
    for rel, text in files.items():
        p = os.path.join(root, rel); os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            fh.write(text)
    return root


def _run_stock(tmp_path, own=(), stock_args=None):
    stub = _stub_openfold3_with_template_modules(str(tmp_path / "site"))
    out = str(tmp_path / "out")
    proof = str(tmp_path / "proof.json")
    pre = env.must_be_absent(HOME)
    e = env.strip(pre, dict(os.environ)); e["PYTHONPATH"] = _stubs.subprocess_pythonpath(HOME, stub); e["OPENFOLD3_OB0_CKPT"] = "/w.pt"
    cmd = [sys.executable, "-s", "-m", "openfold3_ob0_opt.stock_pred", "--proof-json", proof, "--env-absent", ",".join(pre), "--kit-dirs", os.pathsep.join(env.kit_dirs(HOME)),
           "--home", HOME] + list(own) + ["--"] + list(stock_args or ("--query-json", "q.json", "--output-dir", out))
    r = subprocess.run(cmd, env=e, capture_output=True, text=True)
    return r, (json.load(open(proof)) if os.path.isfile(proof) else None), out


def test_the_stock_caller_refuses_an_unknown_or_uninstallable_fix(tmp_path):
    r, proof, out = _run_stock(tmp_path, own=("--upstream-fix", "OF3-199"))
    assert r.returncode == stock_pred.EXIT_NOT_STOCK and "UPSTREAM-FIX NOT APPLIED" in r.stderr and "'OF3-199'" in r.stderr, r.stderr[-800:]
    assert proof["upstream_fix"] == [] and "upstream_fix_error" in proof and not os.path.exists(os.path.join(out, "argv.json"))   # refused before the stock call


def test_run_stock_subprocess_forwards_the_ids_to_the_stock_caller(monkeypatch, tmp_path):
    seen = {}

    def fake_call(cmd, env=None):
        seen["cmd"] = list(cmd)
        return 0

    def fake_gated_manifest(*args, **kw):
        seen["upstream_fix"] = kw.get("upstream_fix")
        return 0
    monkeypatch.setattr(subprocess, "call", fake_call)
    monkeypatch.setattr(cli, "gated_manifest", fake_gated_manifest)
    monkeypatch.setattr(cli, "ds4sci_stack_check", lambda *a, **k: None)
    q = tmp_path / "q.json"; q.write_text(json.dumps({"queries": {"q1": {"chains": []}}}))
    a = cli.build_parser().parse_args(["pred", "--mode", "off", "--query-json", str(q), "--output-dir", str(tmp_path / "o"), "--upstream-fix", "OF3-101"])
    cli.run_stock_subprocess(a, HOME, "/w.pt", str(q), det=0, upstream_fix=["OF3-101"])
    cmd = seen["cmd"]
    assert cmd[cmd.index("--upstream-fix") + 1] == "OF3-101" and cmd.index("--upstream-fix") < cmd.index("--")          # the stock caller's own argument, before the stock argv
    assert seen["upstream_fix"] == []                                                                                    # no proof written by the fake call: the manifest records what the child APPLIED, not what was asked
    seen.clear()
    cli.run_stock_subprocess(a, HOME, "/w.pt", str(q), det=0)
    assert "--upstream-fix" not in seen["cmd"]                                                                           # the default: the stock caller hears nothing of it


# ------------------------------------------------------------------------------------------------------------- default path ----
DEFAULT_PATH_PROBE = textwrap.dedent("""
    import json, os, sys
    from openfold3_ob0_opt.tests import _stubs
    from openfold3_ob0_opt import cli, modes
    home = _stubs.tree_home()
    d = _stubs.stub_dist()
    cli.build_parser().parse_args(["pred", "--mode", "fast", "--query-json", "q.json", "--output-dir", "o"])
    seen = []
    for (mode, line) in modes.LINES:                       # dry-run activation of every mode line: the whole resolve/gate path, nothing applied
        if mode == "off":
            continue
        _stubs.reset_package()
        from openfold3_ob0_opt import stack
        kw = dict(home=home, environ={}, dry_run=True)
        if mode == "big" and line == "tp": kw["n_gpu"] = 2
        stack.activate(mode, **kw)
        seen.append(f"{mode}/{line}")
    _stubs.reset_package(); _stubs.unstub_dist(d)
    root = os.path.realpath(os.path.join(home, "upstream_issues"))
    by_file = sorted(n for n, m in list(sys.modules.items()) if (getattr(m, "__file__", None) or "") and os.path.realpath(m.__file__).startswith(root + os.sep))
    by_name = sorted(n for n in sys.modules if n.startswith("openfold3_ob0_opt_upstream_issues"))
    print(json.dumps({"lines": seen, "by_file": by_file, "by_name": by_name, "core_module_loaded": "opt_core.upstream_fix" in sys.modules}))
""")


def test_the_default_path_loads_nothing_from_upstream_issues():
    """A fresh interpreter that builds the parser and dry-run-activates every mode line holds no module from upstream_issues/ (by file and by
    name): with the flag absent the fixes are unreachable code."""
    r = subprocess.run([sys.executable, "-c", DEFAULT_PATH_PROBE], capture_output=True, text=True,
                       env=dict(env.strip(env.must_be_absent(HOME), dict(os.environ)), PYTHONPATH=_stubs.subprocess_pythonpath(HOME)))
    assert r.returncode == 0, r.stderr[-1500:]
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert len(got["lines"]) == len([k for k in modes.LINES if k[0] != "off"]), got                     # every mode line was activated (dry run)
    assert got["by_file"] == [] and got["by_name"] == [] and got["core_module_loaded"] is False, got


LAYER_REACH = (re.compile(r"\bupstream_issues\b"),                                                       # the directory (loading by path)
               re.compile(r"(?m)^\s*(?:from\s+[\w.]*\s+import\s+[^\n]*\bupstream_fix\b|import\s+[^\n]*\bupstream_fix\b)"),   # importing the plumbing module
               re.compile(r"[\"']--upstream-fix[\"']"))                                                    # handling the flag itself


def test_no_lever_hook_mode_or_kit_file_reaches_the_upstream_issues_layer():
    """upstream_issues/ is reachable from the CLI plumbing only (cli, stock_pred, tp forwarding, manifest key, the package's upstream_fix_registry): no file of
    the add-ons (opt/forward/**), no hook, no lever module, no modes/registry/stack line loads the directory, imports the plumbing module or handles
    the flag."""
    pkg = os.path.join(HOME, "opt", "openfold3_ob0_opt")
    plumbing = {os.path.join(pkg, f) for f in ("cli.py", "stock_pred.py", "tp.py", "manifest.py", "__init__.py")}
    offenders = []
    for root in (os.path.join(HOME, "opt", "forward"), pkg):
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in ("tests", "__pycache__")]
            for f in filenames:
                p = os.path.join(dirpath, f)
                if p in plumbing or not f.endswith((".py", ".sh", ".yml", ".yaml", ".json")):
                    continue
                text = open(p, encoding="utf-8", errors="replace").read()
                if any(rx.search(text) for rx in LAYER_REACH):
                    offenders.append(os.path.relpath(p, HOME))
    assert offenders == [], offenders
