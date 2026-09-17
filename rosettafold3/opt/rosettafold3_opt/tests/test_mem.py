"""The memory policy of the kit rows (mem.py; lever MEM): the allocator export and its refusals by name, the trunk-graph cache budget on tg
arms, the seams wrapped on the kit module with the allocator released at both, the counters in the tally, the line grammar."""
import os
import sys
import types

import pytest

from .. import mem, report, stack


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    saved = dict(mem.STATE)
    mem.STATE.update({"policy": None, "installed": False, "wrapped": False, "releases": 0, "freed_gb": 0.0, "max_reserved_gb": 0.0, "max_allocated_gb": 0.0,
                      "reserved_at_begin_gb": [], "reserved_at_end_gb": []})
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.delenv(mem.ENV_TG_MAX, raising=False)
    yield
    mem.STATE.update(saved)


def _fake_torch(monkeypatch, reserved):
    """A torch stub whose caching allocator reports ``reserved`` bytes and drops to half on empty_cache."""
    calls = []
    cuda = types.SimpleNamespace(is_available=lambda: True, synchronize=lambda: calls.append("sync"), memory_reserved=lambda: reserved[0],
                                 empty_cache=lambda: (calls.append("empty_cache"), reserved.__setitem__(0, reserved[0] // 2)),
                                 max_memory_reserved=lambda: 80_000_000_000, max_memory_allocated=lambda: 35_000_000_000)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    return calls


def test_declarations():
    from .. import registry
    assert mem.POLICY == "capped" and mem.ALLOC_CONF == "expandable_segments:True,garbage_collection_threshold:0.5" and mem.TG_BUDGET == "1"
    assert mem.EXPORTS == ("PYTORCH_CUDA_ALLOC_CONF", mem.ENV_TG_MAX) and callable(mem.apply) and callable(mem.report)
    lv = registry.LEVERS["mem"]
    assert lv.switch is None and lv.kit == registry.PACKAGE and lv.kit_files == ("mem.py",) and lv.cls == registry.CLASS_MEMORY and lv.probe == ("mem", "releases")


def test_install_records_the_policy_and_wrap_releases_at_both_seams(monkeypatch):
    rep = {}
    reserved = [60_000_000_000]
    calls = _fake_torch(monkeypatch, reserved)
    assert mem.apply(rep) == {"policy": "capped", "trigger": "rf3.graph_flags", "seams": ["hoist_begin", "hoist_end"], "applied": "deferred", "alloc_conf": mem.ALLOC_CONF, "fpf_tg_max": None}
    events = []
    gf = types.SimpleNamespace(hoist_begin=lambda n_calls=None: events.append(("begin", n_calls)), hoist_end=lambda: events.append("end") or "r")
    mem.wrap(gf)
    gf.hoist_begin(n_calls=199)
    assert gf.hoist_end() == "r"
    assert events == [("begin", 199), "end"] and calls == ["sync", "empty_cache", "sync", "empty_cache"]
    s = mem.report()
    assert s["policy"] == "capped" and s["installed"] and s["wrapped"] and s["releases"] == 2 and s["reserved_at_begin_gb"] == [60.0] and s["reserved_at_end_gb"] == [30.0]
    assert s["freed_gb"] == 45.0 and s["max_reserved_gb"] == 80.0 and s["max_allocated_gb"] == 35.0
    assert gf.hoist_begin.__wrapped__ is not None and gf.hoist_end.__wrapped__ is not None
    t = report.tally({"mode": "exact", "row": "exact"})
    assert t["mem"]["releases"] == 2                                            # the counters ride the tally JSON (pred reads it)


def test_line_grammar():
    assert mem.line() == mem.line("capped") == "[rosettafold3-opt] MEM policy=capped seams=rf3.graph_flags.hoist_begin,rf3.graph_flags.hoist_end alloc_conf=expandable_segments:True,garbage_collection_threshold:0.5"
    assert mem.line("capped", "1").endswith("garbage_collection_threshold:0.5 fpf_tg_max=1")


def test_the_fpf_trunk_graph_cache_is_budgeted_on_tg_arms_only(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_initialized=lambda: False)))
    assert mem.apply({"fpf": None})["fpf_tg_max"] is None and mem.ENV_TG_MAX not in os.environ          # exact: no trunk graph to budget
    assert mem.apply({"fpf": {"arm": "fast+gflash+ttr+apb+tg+res@L1"}})["fpf_tg_max"] == "1" and os.environ[mem.ENV_TG_MAX] == "1"
    monkeypatch.setenv(mem.ENV_TG_MAX, "4")
    assert mem.apply({"fpf": {"arm": "fast+tg@L1"}})["fpf_tg_max"] == "4" and os.environ[mem.ENV_TG_MAX] == "4"
    monkeypatch.delenv(mem.ENV_TG_MAX, raising=False)
    assert mem.apply({"fpf": {"arm": "fast+gflash+ttr+apb+res@L1", "components": ["fast", "gflash", "ttr", "apb", "res"]}})["fpf_tg_max"] is None and mem.ENV_TG_MAX not in os.environ   # the fast mode: no trunk graph     # the environment's own budget stands, recorded


def test_the_allocator_configuration_is_exported_or_refused_by_name(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_initialized=lambda: False)))
    assert mem.apply({})["alloc_conf"] == "expandable_segments:True,garbage_collection_threshold:0.5" == os.environ["PYTORCH_CUDA_ALLOC_CONF"]
    assert mem.apply({})["alloc_conf"] == mem.ALLOC_CONF                                     # the policy's own value already in place: accepted
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    with pytest.raises(mem.MemPolicyError, match="names another allocator configuration"):
        mem.apply({})
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_initialized=lambda: True)))
    with pytest.raises(mem.MemPolicyError, match="already initialised"):
        mem.apply({})


def test_release_without_cuda_is_a_counted_noop(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)))
    gf = types.SimpleNamespace(hoist_begin=lambda: None, hoist_end=lambda: None)
    mem.STATE["policy"] = mem.POLICY
    mem.wrap(gf); gf.hoist_begin(); gf.hoist_end()
    assert mem.report()["releases"] == 0 and mem.report()["wrapped"]


def test_wrap_refuses_a_module_without_the_seams():
    with pytest.raises(mem.MemPolicyError, match="has no hoist_begin, hoist_end: not the shipped kit module"):
        mem.wrap(types.ModuleType("rf3.graph_flags"))


def test_activation_refuses_by_name_when_the_allocator_export_is_declined(monkeypatch, patched_tree):
    """The policy is a GATE of stack.activate (beside undeclared_env, before the row is exported): an allocator configuration the core
    declines is the kit's NOT ACTIVE line + ActivationError under strict (the hook exits 3, the CLI its usage code); nothing is exported;
    the dry run reports it."""
    import io
    import rosettafold3_opt
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stderr", buf)
    with pytest.raises(rosettafold3_opt.ActivationError, match="memory policy capped: .*names another allocator configuration"):
        stack.activate("exact", strict=True)
    last = buf.getvalue().strip().splitlines()[-1]
    assert last.startswith("[rosettafold3-opt] NOT ACTIVE: memory policy capped: ") and "names another allocator configuration" in last, last
    assert "RF3_CUDAGRAPH" not in os.environ and stack.STATE["report"] is None
    rep = stack.activate("exact", dry_run=True)
    assert rep["reason"].startswith("memory policy capped: ") and rep["active"] is False
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    rep = stack.activate("exact", dry_run=True)
    assert rep["reason"] is None and rep["mem"] == {"policy": "capped", "trigger": "rf3.graph_flags", "seams": ["hoist_begin", "hoist_end"], "applied": "deferred", "alloc_conf": mem.ALLOC_CONF, "fpf_tg_max": "1"}


def test_check_prints_the_policy_on_the_dry_run(monkeypatch, patched_tree, capsys):
    """`check --mode exact` (the dry run) applies the policy and prints the MEM line beside the DRY-RUN line (stdout)."""
    from .. import cli
    monkeypatch.setattr(stack, "gpu_info", lambda: None)
    cli.main(["check", "--mode", "exact"])
    out = capsys.readouterr().out.splitlines()
    assert any(l == mem.line("capped", "1") for l in out), out                              # the policy on the dry run (exact carries the trunk graph: its cache budget on the line)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    assert cli.main(["check", "--mode", "exact"]) == cli.EXIT_NOT_ACTIVE
    out = capsys.readouterr().out
    assert "reason=memory policy capped: " in out and "names another allocator configuration" in out and mem.line("capped") not in out   # the dry run names it on its line


def test_watch_wraps_the_kit_module_once(monkeypatch, tmp_path):
    """stack's watch on rf3.graph_flags runs the APPLIED check and then wraps the seams."""
    import importlib
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_initialized=lambda: False)))
    rep = {"switches": {}, "mode": "exact", "row": "exact"}
    mem.apply(rep)
    monkeypatch.setattr(stack, "applied_check", lambda module, rep: rep.setdefault("applied_calls", []).append(module.__name__))
    pkg = tmp_path / "rf3"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "graph_flags.py").write_text("def hoist_begin(n_calls=None):\n    return 'b'\ndef hoist_end():\n    return 'e'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    for name in ("rf3", "rf3.graph_flags"):
        sys.modules.pop(name, None)
    stack._install_apply_watch(rep)
    _fake_torch(monkeypatch, [10_000_000_000])
    gf = importlib.import_module("rf3.graph_flags")
    assert rep["applied_calls"] == ["rf3.graph_flags"] and hasattr(gf.hoist_begin, "__wrapped__") and mem.STATE["wrapped"]
    assert gf.hoist_begin() == "b" and gf.hoist_end() == "e" and mem.report()["releases"] == 2
    for name in ("rf3", "rf3.graph_flags"):
        sys.modules.pop(name, None)
