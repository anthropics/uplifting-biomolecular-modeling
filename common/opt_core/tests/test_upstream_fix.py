"""opt_core.upstream_fix: the ``--upstream-fix ID[,ID…]`` grammar (parse: order kept, duplicates dropped, no ``all`` word), the registry a
kit fills by injection (register / register_file), refusal of an unknown ID BY NAME listing the registered ones with the kit's words appended,
a fix's own run precondition (refuse words), apply in the order given with one census line per fix through the kit's logger, records completed
with id/file/extra words, nothing loaded or printed for no IDs, file-backed entries loaded lazily by path, and no engine names in the module."""
import os
import sys
import textwrap

import pytest

from opt_core import stock_proof, upstream_fix as U


def test_parse_keeps_order_drops_duplicates_and_blanks():
    assert U.parse(None) == [] and U.parse("") == [] and U.parse(" , ,") == []
    assert U.parse("A-001") == ["A-001"]
    assert U.parse("A-001,A-002") == ["A-001", "A-002"]
    assert U.parse(" A-002 , A-001,A-002 ") == ["A-002", "A-001"]
    assert U.parse("all") == ["all"]                                        # no keyword: 'all' is just an (unknown) ID, refused by name at resolve
    assert U.FLAG == "--upstream-fix" and U.NOT_APPLIED == "UPSTREAM-FIX NOT APPLIED"


def test_unknown_id_is_refused_by_name_listing_the_registered_ids_and_the_kits_words():
    reg = U.Registry(where=" — one file per issue under upstream_issues/ (README 'Known upstream issues (not fixed by default)')")
    assert reg.resolve([]) == [] and reg.known() == "none" and reg.ids() == []
    with pytest.raises(U.UnknownUpstreamFix) as ei:
        reg.resolve(["X-9"])
    assert str(ei.value) == "unknown upstream fix 'X-9' (--upstream-fix); known: none — one file per issue under upstream_issues/ (README 'Known upstream issues (not fixed by default)')"
    reg.register("B-002", lambda log=None: {}); reg.register("A-001", lambda log=None: {})
    assert reg.ids() == ["B-002", "A-001"] and reg.known() == "A-001, B-002"          # registration order kept; the refusal lists them sorted
    with pytest.raises(U.UnknownUpstreamFix) as ei:
        reg.resolve(["A-001", "Z-1", "B-002"])
    assert str(ei.value).startswith("unknown upstream fix 'Z-1' (--upstream-fix); known: A-001, B-002 — ")
    assert isinstance(ei.value, ValueError)
    assert reg.resolve(["A-001", "B-002"]) == ["A-001", "B-002"]


def test_registration_refuses_a_duplicate_a_bad_token_and_a_non_callable():
    reg = U.Registry()
    reg.register("A-001", lambda log=None: {})
    for bad in ("", "A,B", " A-002"):
        with pytest.raises(ValueError):
            reg.register(bad, lambda log=None: {})
    with pytest.raises(ValueError):
        reg.register("A-001", lambda log=None: {})                          # two entries under one ID
    with pytest.raises(ValueError):
        reg.register("A-003", "not callable")


def test_a_fix_whose_run_precondition_does_not_hold_is_refused_by_name():
    reg = U.Registry()
    reg.register("A-001", lambda log=None: {}, refuse=lambda run: None if run.get("use_templates") == "true" else "requires --use-templates true")
    reg.register("A-002", lambda log=None: {})                              # no precondition
    assert reg.resolve(["A-001", "A-002"]) == ["A-001", "A-002"]            # no run words (the process that applies): no precondition checked
    assert reg.resolve(["A-001"], run={"use_templates": "true"}) == ["A-001"]
    with pytest.raises(U.UpstreamFixRefused) as ei:
        reg.resolve(["A-002", "A-001"], run={"use_templates": "false"})
    assert str(ei.value) == "--upstream-fix A-001 requires --use-templates true" and isinstance(ei.value, ValueError)


def test_apply_installs_in_the_order_given_and_prints_one_census_line_per_fix_through_the_kits_logger():
    calls, lines = [], []
    reg = U.Registry()

    def fix(name, targets):
        def apply(log=None):
            calls.append(name); log(f"{name} detail words")               # a fix may print its own words through the same logger
            return {"summary": f"{name} summary", "targets": list(targets)}
        return apply
    reg.register("A-002", fix("two", []), file="upstream_issues/A-002_second.py")
    reg.register("A-001", fix("one", ["m.f", "n.g"]), file="upstream_issues/A-001_first.py")
    assert reg.apply([], log=lines.append) == [] and lines == [] and calls == []   # no IDs: nothing applied, nothing printed
    recs = reg.apply(["A-001", "A-002"], log=lines.append)
    assert calls == ["one", "two"]
    assert lines == ["one detail words", "UPSTREAM-FIX A-001 applied file=upstream_issues/A-001_first.py targets=m.f,n.g",
                     "two detail words", "UPSTREAM-FIX A-002 applied file=upstream_issues/A-002_second.py targets=-"]
    assert recs == [{"summary": "one summary", "targets": ["m.f", "n.g"], "id": "A-001", "file": "upstream_issues/A-001_first.py"},
                    {"summary": "two summary", "targets": [], "id": "A-002", "file": "upstream_issues/A-002_second.py"}]
    assert U.ids_of(recs) == ["A-001", "A-002"] and U.ids_of(None) == []
    assert U.applied_line({"id": "A-9", "targets": None}) == "UPSTREAM-FIX A-9 applied file=- targets=-"
    with pytest.raises(U.UnknownUpstreamFix):
        reg.apply(["A-001", "nope"], log=lines.append)                      # resolved first: nothing of A-001 ran again
    assert calls == ["one", "two"]

    def broken(log=None):
        raise RuntimeError("upstream's shape changed")
    reg.register("A-003", broken)
    with pytest.raises(RuntimeError):                                       # a fix that cannot be installed propagates: the caller refuses the run by name (NOT_APPLIED)
        reg.apply(["A-003"], log=lines.append)


ISSUE_FILE = textwrap.dedent('''
    """An issue file: ID, apply(log) -> record, refuse(run) -> words | None."""
    import sys
    ID = "T-001"
    SUMMARY = "test issue"
    FLAGGED = True
    sys.modules.setdefault("_t001_import_count", []).append(1)
    def refuse(run):
        return None if run.get("ok") else "requires ok"
    def apply(log=None):
        return {"id": ID, "summary": SUMMARY, "targets": ["pkg.mod.fn"]}
''')


def test_register_file_loads_lazily_by_path_validates_the_file_and_completes_the_record(tmp_path, monkeypatch):
    path = tmp_path / "T-001_test_issue.py"; path.write_text(ISSUE_FILE)
    monkeypatch.delitem(sys.modules, "kitx_upstream_issues.T-001", raising=False); monkeypatch.delitem(sys.modules, "_t001_import_count", raising=False)
    reg = U.Registry()
    reg.register_file("T-001", str(path), module_name="kitx_upstream_issues.T-001", file="upstream_issues/T-001_test_issue.py", summary="test_issue",
                      extra=lambda mod: {"flagged": bool(getattr(mod, "FLAGGED", False))})
    assert reg.ids() == ["T-001"] and reg.file_of("T-001") == "upstream_issues/T-001_test_issue.py" and reg.file_of("nope") is None
    assert reg.resolve(["T-001"]) == ["T-001"] and "kitx_upstream_issues.T-001" not in sys.modules and U.loaded("kitx_upstream_issues") == []   # registered and resolved without run words: NOT loaded
    with pytest.raises(U.UpstreamFixRefused):
        reg.resolve(["T-001"], run={"ok": False})                           # the precondition needs the file: loaded now, by path
    assert U.loaded("kitx_upstream_issues") == ["T-001"] and sys.modules["kitx_upstream_issues.T-001"].__file__ == str(path)
    lines = []
    recs = reg.apply(["T-001"], log=lines.append)
    assert recs == [{"id": "T-001", "summary": "test issue", "targets": ["pkg.mod.fn"], "file": "upstream_issues/T-001_test_issue.py", "flagged": True}]
    assert lines == ["UPSTREAM-FIX T-001 applied file=upstream_issues/T-001_test_issue.py targets=pkg.mod.fn"]
    reg.apply(["T-001"], log=lines.append)
    assert len(sys.modules["_t001_import_count"]) == 1                    # memoised: applied twice, imported once
    bad = tmp_path / "T-002_wrong_id.py"; bad.write_text(ISSUE_FILE)      # defines ID T-001, registered as T-002
    reg.register_file("T-002", str(bad), module_name="kitx_upstream_issues.T-002")
    with pytest.raises(ImportError):
        reg.apply(["T-002"], log=lines.append)
    assert "kitx_upstream_issues.T-002" not in sys.modules                # a file that fails validation leaves no module behind


def test_the_stock_proof_names_the_flags_module_as_the_one_addition_a_fixing_stock_process_may_hold():
    assert stock_proof.UPSTREAM_FIX_MODULE == U.__name__ == "opt_core.upstream_fix"
    assert stock_proof.CORE_ALLOWED_WITH_UPSTREAM_FIX == stock_proof.CORE_ALLOWED_IN_STOCK + ("opt_core.upstream_fix",)
    mods = {"opt_core": 1, "opt_core.stock_proof": 1, "opt_core.upstream_fix": 1}
    assert stock_proof.core_modules_loaded(mods) == ["opt_core.upstream_fix"]                                   # a plain stock process may not hold it
    assert stock_proof.core_modules_loaded(mods, allowed=stock_proof.CORE_ALLOWED_WITH_UPSTREAM_FIX) == []   # one that installed a requested fix may
    assert stock_proof.core_modules_loaded(dict(mods, **{"opt_core.gates": 1}), allowed=stock_proof.CORE_ALLOWED_WITH_UPSTREAM_FIX) == ["opt_core.gates"]


def test_the_module_is_standard_library_only():
    src = open(U.__file__, encoding="utf-8").read()
    imported = {l.split()[1].split(".")[0] for l in src.splitlines() if l.startswith(("import ", "from ")) and not l.startswith("from __future__")}
    assert imported <= {"importlib", "sys", "typing"}, imported
    assert os.path.basename(U.__file__) == "upstream_fix.py"
