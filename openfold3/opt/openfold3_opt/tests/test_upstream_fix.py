"""``--upstream-fix <ID>[,<ID>…]`` (opt_core.upstream_fix through the package's upstream_fix_registry, <tree>/upstream_issues/): the flag's parsing and refusal by name, the one issue of
record OF3-001 on the STOCK ``sample_templates`` compiled from source (upstream ignores a templated inference chain; the fix consumes it;
untemplated chains and a caller's own cache directory are untouched; composition with the kit's template guard is truthful), the census
line iff requested, the manifest key, the stock caller's in-process seam, and the default path: with the flag absent nothing under
upstream_issues/ is loaded and no lever, hook or mode file names it."""
import json
import os
import re
import subprocess
import sys
import textwrap
import types

import pytest

from openfold3_opt import UPSTREAM_ISSUES_DIR, UPSTREAM_ISSUES_MODULES, cli, env, manifest, modes, stock_pred, templ_guard, upstream_fix_registry
from opt_core import upstream_fix
from openfold3_opt.tests import _stubs
from openfold3_opt.tests.test_off_env_proof import STUB_CLI
from openfold3_opt.tests.test_templ_guard import QUERY, FakeChain, stock_modules, structure_array_dir, write_cache_entry  # noqa: F401 — the fixture is used by name

HOME = _stubs.tree_home()
ID = "OF3-001"
FIX_FILE = "OF3-001_templates_not_consumed.py"


# ------------------------------------------------------------------------------------------------------------------ the flag ----
def test_parse_single_list_spaces_duplicates_and_none():
    assert upstream_fix.parse(None) == [] and upstream_fix.parse("") == [] and upstream_fix.parse(" , ") == []
    assert upstream_fix.parse("OF3-001") == ["OF3-001"]
    assert upstream_fix.parse("OF3-001,OF3-002") == ["OF3-001", "OF3-002"]
    assert upstream_fix.parse(" OF3-002 , OF3-001,OF3-002 ") == ["OF3-002", "OF3-001"]                 # order kept, duplicates dropped


def test_the_tree_carries_exactly_the_confirmed_issues_and_resolve_refuses_an_unknown_id_by_name():
    reg = upstream_fix_registry(HOME)
    assert reg.ids() == [ID] and reg.file_of(ID) == f"{UPSTREAM_ISSUES_DIR}/{FIX_FILE}"                # one file per confirmed issue; nothing speculative
    assert os.path.isfile(os.path.join(HOME, UPSTREAM_ISSUES_DIR, FIX_FILE))
    assert reg.resolve([]) == []
    assert reg.resolve([ID]) == [ID]
    with pytest.raises(upstream_fix.UnknownUpstreamFix) as ei:
        reg.resolve([ID, "OF3-999"])
    msg = str(ei.value)
    assert "'OF3-999'" in msg and "known: OF3-001" in msg and "--upstream-fix" in msg and "Known upstream issues" in msg
    assert cli.UPSTREAM_FIX_FLAG == upstream_fix.FLAG == "--upstream-fix"


def test_the_flag_exists_on_every_verb_that_runs_the_model_and_nowhere_else():
    ap = cli.build_parser()
    pred = ["pred", "--mode", "exact", "--query-json", "q.json", "--output-dir", "o"]
    assert ap.parse_args(pred).upstream_fix is None                                                      # default: none
    assert ap.parse_args(pred + ["--upstream-fix", "OF3-001"]).upstream_fix == "OF3-001"
    assert ap.parse_args(pred + ["--upstream-fix", "OF3-001,OF3-002"]).upstream_fix == "OF3-001,OF3-002"   # parsed later (upstream_fix.parse); the parser keeps the text
    assert ap.parse_args(["warm", "--mode", "fast", "--out", "w", "--upstream-fix", ID]).upstream_fix == ID
    with pytest.raises(SystemExit):
        ap.parse_args(["check", "--mode", "exact", "--upstream-fix", ID])                                      # the dry run runs no model: no such flag


def test_pred_refuses_an_unknown_id_by_name_before_any_launch(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("OPENFOLD3_OPT", raising=False)
    before = upstream_fix.loaded(UPSTREAM_ISSUES_MODULES)
    rc = cli.main(["pred", "--mode", "exact", "--ckpt", "/w.pt", "--query-json", "q.json", "--output-dir", str(tmp_path / "o"), "--upstream-fix", "OF3-001,OF3-404"])
    err = capsys.readouterr().err
    assert rc == cli.EXIT_USAGE and "unknown upstream fix 'OF3-404'" in err and "known: OF3-001" in err, err
    assert not os.path.exists(tmp_path / "o") and upstream_fix.loaded(UPSTREAM_ISSUES_MODULES) == before                       # nothing launched, nothing loaded


def test_of3_001_requires_use_templates_true_refused_by_name_before_any_launch(tmp_path, capsys, monkeypatch):
    """OF3-001 consumes PREPROCESSED template cache entries and upstream preprocesses templates only under --use-templates true: with templates
    off the set flag cannot be applied, so the run is refused by name at argument resolution (exit 2, nothing launched) — on pred given
    ``--use-templates false``, and on warm (its public query runs templates off); with the flag not given upstream's default (true) stands."""
    monkeypatch.delenv("OPENFOLD3_OPT", raising=False)
    words = "--upstream-fix OF3-001 requires --use-templates true"
    pred = ["pred", "--mode", "exact", "--ckpt", "/w.pt", "--query-json", "q.json", "--output-dir", str(tmp_path / "o"), "--upstream-fix", ID]
    for extra in (["--use-templates", "false"], ["--use-templates", "False"]):
        rc = cli.main(pred + extra)
        err = capsys.readouterr().err
        assert rc == cli.EXIT_USAGE and words in err and not os.path.exists(tmp_path / "o"), (extra, err)
    assert cli.main(["warm", "--mode", "fast", "--ckpt", "/w.pt", "--out", str(tmp_path / "w"), "--upstream-fix", ID]) == cli.EXIT_USAGE and words in capsys.readouterr().err
    ap = cli.build_parser()
    for ok in (ap.parse_args(pred + ["--use-templates", "true"]), ap.parse_args(pred)):                          # templates on — given, or upstream's default —: resolved (applied later by the model process)
        assert cli.effective_use_templates(ok) == "true" == cli.UPSTREAM_USE_TEMPLATES and cli.resolve_upstream_fix(ok, HOME, "pred") == [ID]
    assert cli.resolve_upstream_fix(ap.parse_args(pred[:-2]), HOME, "pred") == []                                # no flag: nothing resolved, nothing loaded by this call
    with pytest.raises(upstream_fix.UpstreamFixRefused):
        upstream_fix_registry(HOME).resolve([ID], run={"verb": "pred", "use_templates": "false"})
    assert upstream_fix_registry(HOME).resolve([ID], run={"verb": "pred", "use_templates": "TRUE "}) == [ID]
    assert upstream_fix_registry(HOME).resolve([ID]) == [ID]                          # no run words (the stock caller's own resolve): no precondition, the parent checked


# ------------------------------------------------------------------------------------------- OF3-001 on the stock function ----
KW = dict(n_templates=4, take_top_k=True, template_file_format="npz")


def _assembly(tmp_path, ids):
    npz = write_cache_entry(tmp_path / "entry", ids)
    arr = structure_array_dir(str(tmp_path / "arrays"), ids)
    return {"A": {"template_ids": ids, "cache_entry_file_path": npz}, "B": {"template_ids": [], "cache_entry_file_path": None}}, arr


def test_of3_001_makes_the_stock_sampler_consume_a_templated_inference_chain_and_nothing_else(stock_modules, tmp_path, capsys):
    struct, pipe = stock_modules                                                                         # STOCK sample_templates compiled from source, under the kit's template guard (a kit arm)
    ids = ["1abc_A", "2xyz_B", "3pqr_C"]
    assembly, arr = _assembly(tmp_path, ids)
    kw = dict(KW, template_structure_array_directory=arr)
    stock = struct.sample_templates.__wrapped__                                                          # upstream 0.4.1 itself
    assert stock(assembly_data=assembly, template_cache_directory=None, chain_id="A", **kw) == {}        # THE ISSUE: a templated inference chain, directory None -> nothing sampled
    assert struct.sample_templates(assembly_data=assembly, template_cache_directory=None, chain_id="A", **kw) == {}   # the default kit arm: exactly stock (named, not consumed)
    assert "not consumed at inference (upstream 0.4.1 behaviour)" in capsys.readouterr().err
    assert pipe.process_template_structures_of3({"A": (QUERY, {})}, assembly_data=assembly, **kw) == {"A": []}     # through stock's pipeline loop: no template slice reaches the featurizer
    capsys.readouterr()
    records = upstream_fix_registry(HOME).apply([ID], log=cli._err)                                      # --upstream-fix OF3-001 in this process
    err = capsys.readouterr().err
    assert [r["id"] for r in records] == [ID] and records[0]["file"] == f"upstream_issues/{FIX_FILE}"
    assert records[0]["targets"] == [f"{templ_guard.STRUCT_MODULE}.sample_templates", f"{templ_guard.PIPELINE_MODULE}.sample_templates"]
    assert f"[openfold3-opt] UPSTREAM-FIX {ID} applied file=upstream_issues/{FIX_FILE} targets=" in err, err   # the census line, printed by the process that applied it
    assert struct.sample_templates is pipe.sample_templates and getattr(struct.sample_templates, "_of3_upstream_fix") == ID
    got = struct.sample_templates(assembly_data=assembly, template_cache_directory=None, chain_id="A", **kw)
    assert sorted(got) == sorted(ids)[:3] and len(got) == 3                                             # FIXED: the chain's own cache entry is read, top-k ids sampled (k = min(3 ids, n_templates 4))
    err = capsys.readouterr().err
    assert f"UPSTREAM-FIX {ID} chain=A template_ids=3 sampled=3" in err and "not consumed" not in err, err   # the guard's 'not consumed' line does not fire: it would be false now
    assert sorted(got) == sorted(stock(assembly_data=assembly, template_cache_directory=tmp_path, chain_id="A", **kw))   # == what stock returns for a consuming caller (its own inference branch ran)
    slices = pipe.process_template_structures_of3({"A": (QUERY, {})}, assembly_data=assembly, **kw)["A"]
    assert len(slices) == 3                                                                              # through stock's pipeline loop (the call site's global): three template slices reach the featurizer
    capsys.readouterr()
    assert struct.sample_templates(assembly_data=assembly, template_cache_directory=None, chain_id="B", **kw) == {} and capsys.readouterr().err == ""   # untemplated chain: stock, silent
    mine = struct.sample_templates(assembly_data=assembly, template_cache_directory=tmp_path, chain_id="A", **kw)   # a caller's own directory: stock's arguments untouched, silent
    assert sorted(mine) == sorted(ids)[:3] and capsys.readouterr().err == ""
    assert upstream_fix_registry(HOME).apply([ID], log=cli._err) and struct.sample_templates.__wrapped__.__wrapped__ is stock   # idempotent: applied twice, wrapped once (fix ∘ guard ∘ stock)
    assert upstream_fix.loaded(UPSTREAM_ISSUES_MODULES) == [ID]


def test_the_guard_census_states_consumption_under_the_fix_and_is_byte_unchanged_by_default(monkeypatch, capsys):
    """Default: `TEMPLATES PARSED …: templates parsed; not consumed at inference (upstream 0.4.1 behaviour)` and templ_policy=ignore(upstream_0.4.1),
    byte for byte. After the CLI seam applies OF3-001 in this process the same census states the truth: `… consumed at inference under
    --upstream-fix OF3-001` and templ_policy=consume(OF3-001). (Run against the live package modules — the ones the CLI's own relative imports
    bind — so the seam and the census are observed on one module object whatever other tests re-imported.)"""
    import importlib
    live_cli, tg = importlib.import_module("openfold3_opt.cli"), importlib.import_module("openfold3_opt.templ_guard")
    from openfold3_opt.tests.test_templ_guard import stock_sample_function
    monkeypatch.setattr(tg, "_CONSUMED_BY", None)
    struct = types.ModuleType(tg.STRUCT_MODULE); struct.sample_templates = stock_sample_function()
    pipe = types.ModuleType(tg.PIPELINE_MODULE); pipe.sample_templates = struct.sample_templates
    monkeypatch.setitem(sys.modules, tg.STRUCT_MODULE, struct); monkeypatch.setitem(sys.modules, tg.PIPELINE_MODULE, pipe)
    monkeypatch.delitem(sys.modules, f"{UPSTREAM_ISSUES_MODULES}.{ID}", raising=False)
    preproc = types.ModuleType(tg.PREPROC_MODULE)

    class TemplatePreprocessor:                                                                          # stock's parse surface: one declared protein chain
        moltypes = ("PROTEIN",)

        def __init__(self):
            self.input_set = types.SimpleNamespace(queries={"q1": types.SimpleNamespace(chains=[FakeChain(["A"], "/aln/q", ids=["x1"])])})

        def _parse_inference_query_set(self):
            return None

        def _update_inference_query_set(self):
            return None
    preproc.TemplatePreprocessor = TemplatePreprocessor
    tg.patch_preproc_module(preproc)

    def parse_line():
        preproc.TemplatePreprocessor()._parse_inference_query_set()
        return capsys.readouterr().err
    assert parse_line() == "[openfold3-opt] TEMPLATES PARSED queries=1 chains=1: templates parsed; not consumed at inference (upstream 0.4.1 behaviour)\n"
    assert tg.policy() == ("ignore", "upstream_0.4.1") and tg.counters()["templ_policy"] == "ignore(upstream_0.4.1)" and tg.declared_words() == tg.NOT_CONSUMED
    records = live_cli.apply_upstream_fix([ID], HOME, "pred --mode exact")                               # the seam every arm runs after activation
    err = capsys.readouterr().err
    assert [r["id"] for r in records] == [ID] and records[0]["consumes_templates"] is True and f"UPSTREAM-FIX {ID} applied" in err
    assert tg.policy() == ("consume", ID) and tg.counters()["templ_policy"] == f"consume({ID})"
    assert parse_line() == f"[openfold3-opt] TEMPLATES PARSED queries=1 chains=1: templates parsed; consumed at inference under --upstream-fix {ID}\n"


def test_of3_001_applied_before_the_guard_is_left_alone_by_it(monkeypatch, tmp_path, capsys):
    """The stock arm's order cannot occur on a kit arm, but the contract holds either way: a function carrying the fix is never re-wrapped by
    templ_guard.install() (it carries the guard's marker), so no 'not consumed' line can contradict a consuming run."""
    import types
    from openfold3_opt.tests.test_templ_guard import stock_sample_function
    struct = types.ModuleType(templ_guard.STRUCT_MODULE); struct.sample_templates = stock_sample_function()
    struct.map_token_pos_to_template_residues = lambda *a, **k: None; struct.align_template_to_query = lambda *a, **k: []
    pipe = types.ModuleType(templ_guard.PIPELINE_MODULE); pipe.sample_templates = struct.sample_templates; pipe.align_template_to_query = struct.align_template_to_query
    pipe.process_template_structures_of3 = lambda *a, **k: {}
    monkeypatch.setitem(sys.modules, templ_guard.STRUCT_MODULE, struct); monkeypatch.setitem(sys.modules, templ_guard.PIPELINE_MODULE, pipe)
    monkeypatch.delitem(sys.modules, f"{UPSTREAM_ISSUES_MODULES}.{ID}", raising=False)
    upstream_fix_registry(HOME).apply([ID], log=cli._err)
    fixed = struct.sample_templates
    templ_guard.patch_struct_module(struct); templ_guard.patch_pipeline_module(pipe)                      # the guard arrives second
    assert struct.sample_templates is fixed and pipe.sample_templates is fixed
    ids = ["1abc_A"]; assembly, arr = _assembly(tmp_path, ids)
    assert sorted(struct.sample_templates(assembly_data=assembly, template_cache_directory=None, chain_id="A", template_structure_array_directory=arr, **KW)) == ids
    assert "not consumed" not in capsys.readouterr().err


def test_a_fix_that_cannot_be_installed_refuses_the_run_by_name(monkeypatch, capsys):
    """The stock modules cannot be imported in this process (forced: None in sys.modules, whether or not openfold3 is installed): apply raises,
    the CLI seam prints NOT ACTIVE: UPSTREAM-FIX NOT APPLIED and returns None (the verbs exit 3) — a requested fix never degrades to a run without it."""
    monkeypatch.setitem(sys.modules, templ_guard.STRUCT_MODULE, None); monkeypatch.setitem(sys.modules, templ_guard.PIPELINE_MODULE, None)   # import -> ModuleNotFoundError, on any interpreter
    monkeypatch.delitem(sys.modules, f"{UPSTREAM_ISSUES_MODULES}.{ID}", raising=False)
    assert cli.apply_upstream_fix([], HOME, "pred --mode exact") == [] and capsys.readouterr().err == ""   # nothing requested: nothing loaded, nothing printed
    assert upstream_fix.loaded(UPSTREAM_ISSUES_MODULES) == []
    assert cli.apply_upstream_fix([ID], HOME, "pred --mode exact") is None
    err = capsys.readouterr().err
    assert "NOT ACTIVE: UPSTREAM-FIX NOT APPLIED (pred --mode exact --upstream-fix OF3-001): ModuleNotFoundError" in err and "applied file=" not in err, err


def test_one_template_guard_finder_per_process_across_package_reimports():
    """templ_guard.install() keeps ONE finder of its kind on sys.meta_path: a stale instance left by a re-imported package (the tests'
    reset_package; never the runtime, which activates once) is dropped, and a finder never delegates a find to another of its kind — so an
    import of a stock module cannot bounce between two guard finders."""
    import importlib
    tg = importlib.import_module("openfold3_opt.templ_guard")
    stale = type("_TemplGuardFinder", (), {"find_spec": lambda self, *a, **k: (_ for _ in ()).throw(AssertionError("delegated to a stale guard finder"))})()
    sys.meta_path.insert(0, stale)
    try:
        tg.install()
        kinds = [f for f in sys.meta_path if type(f).__name__ == "_TemplGuardFinder"]
        assert kinds == [tg.FINDER]                                                                       # the stale instance is gone, the live one armed once
        sys.meta_path.insert(0, stale)                                                                    # even if one reappears, a find never reaches it
        assert tg.FINDER.find_spec(tg.STRUCT_MODULE) is None or True                                      # returns whatever the real finders say; no AssertionError from the stale one
    finally:
        while stale in sys.meta_path:
            sys.meta_path.remove(stale)
        tg.uninstall()


# ---------------------------------------------------------------------------------------------------------- the manifest key ----
def test_the_manifest_records_the_ids_under_upstream_fix():
    off = {"active": False, "mode": "off", "reason": "stock"}
    assert manifest.build(off, command="pred")["upstream_fix"] == []                                   # the default: nothing applied
    assert manifest.build(off, command="pred", upstream_fix=[ID])["upstream_fix"] == [ID]
    assert manifest.build(off, command="warm", upstream_fix=[ID, "OF3-002"])["upstream_fix"] == [ID, "OF3-002"]


# ------------------------------------------------------------------------------------------------ the stock caller's seam ----
STUB_STRUCT = "def sample_templates(assembly_data, template_cache_directory, n_templates, take_top_k, chain_id, template_structure_array_directory, template_file_format, use_roda_monomer_format=False):\n    return {}\n"
STUB_PIPELINE = "from openfold3.core.data.primitives.structure.template import sample_templates  # noqa: F401\n\ndef process_template_structures_of3(*a, **k):\n    return {}\n"


def _stub_openfold3_with_template_modules(root):
    """A stub `openfold3` distribution: the run_openfold CLI stub of the stock-route tests plus the two template modules OF3-001 wraps."""
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
    e = env.strip(pre, dict(os.environ)); e["PYTHONPATH"] = _stubs.subprocess_pythonpath(HOME, stub); e["OPENFOLD3_CKPT"] = "/w.pt"
    cmd = [sys.executable, "-s", "-m", "openfold3_opt.stock_pred", "--proof-json", proof, "--env-absent", ",".join(pre), "--kit-dirs", os.pathsep.join(env.kit_dirs(HOME)),
           "--home", HOME] + list(own) + ["--"] + list(stock_args or ("--query-json", "q.json", "--output-dir", out))
    r = subprocess.run(cmd, env=e, capture_output=True, text=True)
    return r, (json.load(open(proof)) if os.path.isfile(proof) else None), out


def test_the_stock_caller_applies_the_fix_inside_its_own_process_after_the_proof(tmp_path):
    r, proof, out = _run_stock(tmp_path, own=("--upstream-fix", ID))
    assert r.returncode == 0, r.stderr[-1200:]
    assert proof["ok"] and proof["ok_after"] and proof["exit_code"] == 0 and proof["n_cif"] == 1                          # the arm reads as it is: stock + the named fix (no kit module, no finder)
    assert proof["kit_modules_loaded_after"] == [] and proof["kit_hooks_installed_after"] == [] and proof["core_modules_loaded_after"] == []
    assert [rec["id"] for rec in proof["upstream_fix"]] == [ID] and proof["upstream_fix"][0]["file"] == f"upstream_issues/{FIX_FILE}"
    lines = r.stderr.splitlines()
    i_clean = next(i for i, l in enumerate(lines) if "ENV-CLEAN ok" in l)
    i_fix = next(i for i, l in enumerate(lines) if f"UPSTREAM-FIX {ID} applied" in l)
    i_call = next(i for i, l in enumerate(lines) if "stock: run_openfold predict" in l)
    assert i_clean < i_call < i_fix                                                                                      # after the proof and the call's announcement, right before the entry point runs
    assert json.load(open(os.path.join(out, "argv.json")))["args"][0] == "predict"                                      # the stock CLI ran, arguments unchanged
    r0, proof0, _ = _run_stock(tmp_path / "plain")                                                                      # the same call without the flag: nothing of this
    assert r0.returncode == 0 and proof0["upstream_fix"] == [] and "UPSTREAM-FIX" not in r0.stderr


def test_the_stock_caller_refuses_an_unknown_or_uninstallable_fix(tmp_path):
    r, proof, out = _run_stock(tmp_path, own=("--upstream-fix", "OF3-999"))
    assert r.returncode == stock_pred.EXIT_NOT_STOCK and "UPSTREAM-FIX NOT APPLIED" in r.stderr and "'OF3-999'" in r.stderr, r.stderr[-800:]
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
    a = cli.build_parser().parse_args(["pred", "--mode", "off", "--query-json", str(q), "--output-dir", str(tmp_path / "o"), "--upstream-fix", "OF3-001"])
    cli.run_stock_subprocess(a, HOME, "/w.pt", str(q), det=0, upstream_fix=["OF3-001"])
    cmd = seen["cmd"]
    assert cmd[cmd.index("--upstream-fix") + 1] == "OF3-001" and cmd.index("--upstream-fix") < cmd.index("--")          # the stock caller's own argument, before the stock argv
    assert seen["upstream_fix"] == []                                                                                    # no proof written by the fake call: the manifest records what the child APPLIED, not what was asked
    seen.clear()
    cli.run_stock_subprocess(a, HOME, "/w.pt", str(q), det=0)
    assert "--upstream-fix" not in seen["cmd"]                                                                           # the default: the stock caller hears nothing of it


# ------------------------------------------------------------------------------------------------------------- default path ----
DEFAULT_PATH_PROBE = textwrap.dedent("""
    import json, os, sys
    from openfold3_opt.tests import _stubs
    from openfold3_opt import cli, modes
    home = _stubs.tree_home()
    d = _stubs.stub_dist()
    cli.build_parser().parse_args(["pred", "--mode", "fast", "--query-json", "q.json", "--output-dir", "o"])
    seen = []
    for (mode, line) in modes.LINES:                       # dry-run activation of every line: the whole resolve/gate path, nothing applied
        if mode == "off":
            continue
        _stubs.reset_package()
        from openfold3_opt import stack
        kw = dict(home=home, environ=({"OF3TP_RANK": "0", "OF3TP_WORLD": "2"} if line == "tp" else {}), dry_run=True, n_gpu=("2" if line == "tp" else None))
        stack.activate(mode, **kw)
        seen.append(f"{mode}/{line}")
    _stubs.reset_package(); _stubs.unstub_dist(d)
    root = os.path.realpath(os.path.join(home, "upstream_issues"))
    by_file = sorted(n for n, m in list(sys.modules.items()) if (getattr(m, "__file__", None) or "") and os.path.realpath(m.__file__).startswith(root + os.sep))
    by_name = sorted(n for n in sys.modules if n.startswith("openfold3_opt_upstream_issues"))
    print(json.dumps({"lines": seen, "by_file": by_file, "by_name": by_name, "core_module_loaded": "opt_core.upstream_fix" in sys.modules}))
""")


def test_the_default_path_loads_nothing_from_upstream_issues():
    """A fresh interpreter that builds the parser and dry-run-activates every line holds no module from upstream_issues/ (by file and by
    name): with the flag absent the fixes are unreachable code."""
    r = subprocess.run([sys.executable, "-c", DEFAULT_PATH_PROBE], capture_output=True, text=True,
                       env=dict(env.strip(env.must_be_absent(HOME), dict(os.environ)), PYTHONPATH=_stubs.subprocess_pythonpath(HOME)))
    assert r.returncode == 0, r.stderr[-1500:]
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert len(got["lines"]) == len([k for k in modes.LINES if k[0] != "off"]), got                                    # every line was activated (dry run)
    assert got["by_file"] == [] and got["by_name"] == [] and got["core_module_loaded"] is False, got


LAYER_REACH = (re.compile(r"\bupstream_issues\b"),                                                       # the directory (loading by path)
               re.compile(r"(?m)^\s*(?:from\s+[\w.]*\s+import\s+[^\n]*\bupstream_fix\b|import\s+[^\n]*\bupstream_fix\b)"),   # importing the plumbing module
               re.compile(r"[\"']--upstream-fix[\"']"))                                                    # handling the flag itself


def test_no_lever_hook_mode_or_kit_file_reaches_the_upstream_issues_layer():
    """upstream_issues/ is reachable from the CLI plumbing only (cli, stock_pred, tp forwarding, manifest key, the package's upstream_fix_registry): no file of
    the add-ons (opt/forward/**), no hook, no lever module, no modes/registry/stack/templ_guard line loads the directory, imports the plumbing
    module or handles the flag (templ_guard is TOLD by the plumbing — consumed_by — and imports nothing of it)."""
    pkg = os.path.join(HOME, "opt", "openfold3_opt")
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
    tg = open(os.path.join(pkg, "templ_guard.py"), encoding="utf-8").read()
    assert "import" not in "".join(l for l in tg.splitlines(keepends=True) if "upstream" in l)               # the guard names the fix in words only


# --------------------------------------------------------------------------------------------- OF3-004, behind OF3-001: gap rows ----
def _fix_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("of3_001_under_test", os.path.join(HOME, UPSTREAM_ISSUES_DIR, FIX_FILE))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_of3_004_gap_rows_leave_the_sampled_entries_before_the_featurizer(capsys):
    """A predict-mode cache entry's idx_map keeps gap columns as -1 rows; the featurizer's residue mapper takes residue-residue rows only
    (OF3-004). The OF3-001 wrapper cuts every sampled entry to those rows, removes an entry with none left, leaves a gap-free entry's array
    untouched, and counts the rows on the chain's line."""
    import numpy as np
    mod = _fix_module()
    class Entry:
        def __init__(self, rows): self.idx_map = np.asarray(rows, dtype=int); self.index = 0; self.release_date = "2020-01-01"
    gapfree = Entry([[1, 11], [2, 12], [3, 13]])
    gapped = Entry([[1, 22], [2, 23], [3, -1], [4, -1], [-1, 24], [5, 25]])         # two template-gap rows, one query-gap row
    hopeless = Entry([[1, -1], [-1, 7]])
    sampled = {"1abc_A": gapfree, "2xyz_B": gapped, "3pqr_C": hopeless}
    kept_array = gapfree.idx_map
    assert mod.drop_gap_rows(sampled) == 5
    assert sorted(sampled) == ["1abc_A", "2xyz_B"]                                    # the entry with no residue pair is gone
    assert gapfree.idx_map is kept_array                                             # gap-free: the very same array (byte-unchanged)
    assert gapped.idx_map.tolist() == [[1, 22], [2, 23], [5, 25]]
    assert mod.drop_gap_rows({}) == 0 and mod.drop_gap_rows(None) == 0
    # through the wrapper: a templated inference chain with template_cache_directory=None -> stock's consuming branch, then the cut, then the line
    calls = []
    def stock(assembly_data, template_cache_directory, n_templates, take_top_k, chain_id, template_structure_array_directory, template_file_format, use_roda_monomer_format):
        calls.append(template_cache_directory)
        return {"9zzz_A": Entry([[1, 5], [2, -1], [3, 6]])}
    fixed = mod.wrap(stock, log=lambda s: calls.append(s))
    out = fixed({"A": {"template_ids": ["9zzz_A"], "cache_entry_file_path": "/x/entry.npz"}}, None, 4, True, "A", None, "cif", False)
    assert calls[0] == mod.SENTINEL_CACHE_DIR and out["9zzz_A"].idx_map.tolist() == [[1, 5], [3, 6]]
    assert calls[1] == "UPSTREAM-FIX OF3-001 chain=A template_ids=1 sampled=1 gap_rows_dropped=1"

