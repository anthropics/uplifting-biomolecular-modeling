"""The ``--n_gpu P`` axis (tp.py / rowpair.py / report / stack / cli): the tokens and refusals are the core's words (opt_core.mem.ngpu), the
P set and the modes that take the axis are this kit's; nothing is installed at n_gpu = 1."""
import json
import re
import os
import subprocess
import sys

import pytest

from boltz2_opt import modes, registry, report as rep, stack, tp

HERE = os.path.dirname(os.path.abspath(__file__))
BIG_WORDS = "refused: n_gpu>1 requires --mode big (sharded reductions are not bitwise)"


def test_tokens_are_the_cores_exact_text():
    assert rep.ngpu_tokens(1) == "n_gpu=1 sharding=none"
    assert rep.ngpu_tokens(2) == "n_gpu=2 sharding=rowpair"
    assert rep.ngpu_tokens(4) == "n_gpu=4 sharding=rowpair"
    rep.reset_tally(); rep.arm_tally("big", "worker", n_gpu=2)
    assert " n_gpu=2 sharding=rowpair predictions=" in rep.tally_line()
    rep.reset_tally(); rep.arm_tally("exact", "worker")
    assert " n_gpu=1 sharding=none predictions=" in rep.tally_line()
    rep.reset_tally()
    line = rep.active_line({"mode": "big", "route": "worker", "gpu": "H100", "n_gpu": 2, "levers_applied": ["resid"], "levers_fallback": []})
    assert line.startswith("[boltz2-opt] ACTIVE mode=big route=worker gpu=H100 n_gpu=2 sharding=rowpair levers=resid"), line
    assert " n_gpu=1 sharding=none " in rep.active_line({"mode": "exact", "route": "worker", "levers_applied": []})
    assert " n_gpu=1 sharding=none " in rep.dry_run_line({"mode": "exact", "route": "worker", "levers": []})


def test_p1_is_accepted_for_every_mode_without_touching_the_box():
    for m in modes.MODE_NAMES:
        assert tp.check(1, m, visible=0) == 1 and tp.check("1", m) == 1 and tp.check(None, m) == 1


@pytest.mark.parametrize("mode", ["exact", "fast", "off"])
def test_p_above_1_outside_big_is_refused_with_the_cores_words(mode):
    with pytest.raises(tp.Refused) as e:
        tp.check(2, mode, visible=8)
    assert str(e.value) == BIG_WORDS, str(e.value)


def test_fewer_visible_gpus_than_p_is_refused_by_name_never_auto_sized():
    with pytest.raises(tp.Refused) as e:
        tp.check(2, "big", visible=1)
    assert str(e.value) == "refused: n_gpu=2 visible=1", str(e.value)
    with pytest.raises(tp.Refused) as e:
        tp.check(4, "big", visible=2)
    assert str(e.value) == "refused: n_gpu=4 visible=2"
    with pytest.raises(tp.Refused) as e:
        tp.check(2, "big", visible=None)
    assert "visible GPU count was not taken" in str(e.value)
    assert tp.check(2, "big", visible=2) == 2 and tp.check(4, "big", visible=8) == 4


def test_a_p_outside_the_kits_set_is_refused_naming_the_set():
    assert tp.SUPPORTED_P == (1, 2, 4, 8) and tp.TP_MODES == ("big",)
    with pytest.raises(tp.Refused) as e:
        tp.check(3, "big", visible=8)
    assert str(e.value) == "refused: n_gpu=3 not served: this kit's row-sharded pair stack serves n_gpu in {1,2,4,8} (boltz2_opt.tp.SUPPORTED_P)", str(e.value)
    for bad in ("0", "-1", "x", "2.5"):
        with pytest.raises(ValueError):
            tp.parse(bad)


def test_gate_carries_the_refusal_and_the_accepted_p():
    plan = stack.gate("exact", need_gpu=False, need_cache=False, check_pins=False, n_gpu=2)
    assert BIG_WORDS in plan["reasons"] and plan["n_gpu"] is None
    plan = stack.gate("off", need_gpu=False, need_cache=False, check_pins=False, n_gpu=2)
    assert plan["reasons"] == [BIG_WORDS]
    plan = stack.gate("off", need_gpu=False, need_cache=False, check_pins=False, n_gpu=1)
    assert plan["reasons"] == [] and plan["n_gpu"] == 1
    plan = stack.gate("big", need_gpu=False, need_cache=False, check_pins=False, n_gpu="3")
    assert any("serves n_gpu in {1,2,4,8}" in r for r in plan["reasons"]), plan["reasons"]
    plan = stack.gate("big", need_gpu=False, need_cache=False, check_pins=False, n_gpu=0)
    assert any(r.startswith("refused: n_gpu=0") for r in plan["reasons"]), plan["reasons"]


def test_worker_line_and_environment_gain_the_axis_only_above_1():
    c1, c2 = stack.worker_command("b.json", "big", 1), stack.worker_command("b.json", "big", 2)
    a1 = c1[c1.index("--attach") + 1].split(","); a2 = c2[c2.index("--attach") + 1].split(",")
    assert stack.TP_ATTACHMENT not in a1 and a2 == a1 + [stack.TP_ATTACHMENT]
    assert c1[c1.index("--") + 1:] == c2[c2.index("--") + 1:], "the worker script and its arguments are the same line at every P"
    sw = registry.LEVERS["rowpair_tp"]["switch"]
    assert sw == "BOLTZ_TP" and sw not in stack.child_env("big", {}, n_gpu=1) and stack.child_env("big", {}, n_gpu=2)[sw] == "2"
    sync = registry.LEVERS["rowpair_tp"]["sync_switch"]                     # the replicated-tensor sync policy word travels with the axis (explicit, det 0: bcast); the row's alone
    assert stack.child_env("big", {}, n_gpu=2)[sync] == "bcast" and stack.child_env("big", {sync: "guard"}, n_gpu=2)[sync] == "bcast" and sync not in stack.child_env("big", {}, n_gpu=1)
    for k, v in modes.TP_EXPORTS.items():                                       # the ×P line's placement words (the core's host-parking levers): set at P > 1 only, the row's values (a caller's copy is stripped)
        assert k.startswith("ROWPAIR_") and stack.child_env("big", {}, n_gpu=2)[k] == v and stack.child_env("big", {k: "0"}, n_gpu=2)[k] == v and k not in stack.child_env("big", {}, n_gpu=1)
        assert stack.tp_exports(stack.child_env("big", {}, n_gpu=2), 2)[k] == v and stack.tp_exports(stack.child_env("big", {}, n_gpu=1), 1) == {}
    assert modes.TP_EXPORTS["ROWPAIR_PARK_ZINIT"] == "recompute" and modes.TP_EXPORTS["ROWPAIR_HOST_SLAB"] == "lease"                          # the initial pair shard parked between recycles on the kit line
    assert modes.TP_EXPORTS["ROWPAIR_MSA_M_LAYOUT"] == "token_sharded"              # the MSA representation token-sharded across the ranks on the kit line
    from opt_core.mem.rowpair import launch
    assert stack.RANK_TAG_ENV == launch.ENV_TAG and stack.child_env("big", {}, n_gpu=2)[launch.ENV_TAG] == "boltz2-opt" and launch.ENV_TAG not in stack.child_env("big", {}, n_gpu=1)   # the launcher's and the ranks' line tag
    assert {k: v for k, v in stack.child_env("big", {}, n_gpu=2).items() if k not in (sw, sync, launch.ENV_TAG, *modes.TP_EXPORTS)} == stack.child_env("big", {}, n_gpu=1)


def test_rank_batch_writes_under_the_kit_dirs_rank_dirs(tmp_path):
    b = tmp_path / "pred_batch.json"
    b.write_text(json.dumps({"tag": "pred", "out_dir": str(tmp_path / "out"), "kit_dir": str(tmp_path / "out" / "_kit"), "items": [{"name": "x"}], "seeds": [0]}))
    p = stack.rank_batch(str(b), 1)
    d = json.load(open(p))
    rd = str(tmp_path / "out" / "_kit" / "ranks" / "rank1")
    assert p.endswith("pred_batch.rank1.json") and d["out_dir"] == d["kit_dir"] == rd == stack.rank_dir(str(tmp_path / "out" / "_kit"), 1) and os.path.isdir(rd) and d["items"] == [{"name": "x"}]
    assert stack.worker_log_path(rd, "pred") == os.path.join(rd, "pred_worker_log.json")


def test_the_adapter_installs_nothing_at_p1_and_says_so(monkeypatch):
    from boltz2_opt import rowpair
    monkeypatch.delenv("BOLTZ_TP", raising=False)
    with pytest.raises(rowpair.Refused) as e:
        rowpair.apply()
    assert "installs nothing at n_gpu=1" in str(e.value) and rowpair.report()["installed"] is False
    monkeypatch.setenv("BOLTZ_TP", "1")
    with pytest.raises(rowpair.Refused):
        rowpair.apply()
    assert rowpair.LEVERS == ("rowpair_tp",) and registry.LEVER_IDS["rowpair_tp"]["strategy"] == "F7.tensor_parallel"


TP_OK = {"installed": True, "applied": ["rowpair_tp"], "n_gpu": 2, "rank": 0, "sharded": ["PairformerModule", "PairformerNoSeqModule"], "replicated": ["MSAModule.pairformer_layer"],
         "layouts": {"398": "..."}, "rows_census": {"rows_tiled": True, "P": 2}, "calls": {"trunk_rows": 2, "zcond_rows": 2, "dit_blocks_sharded": 2, "conf_rows": 2, "layers": 300}, "peak_alloc_gib": 11.5, "errors": []}


def _big_log(**extra):
    log = {"env": {"BOLTZ_LEVERS": "resid,mask2"}, "lever_report": {"applied": True, "levers": ["resid", "mask2"]}, "per_item": [{"name": "x", "seed": 0}],
           "templ_report": {"installed": True, "items_checked": 1, "declared": 0, "live": 0, "refused": 0, "per_item": []},
           "trimul_report": {"installed": True, "levers": ["fpf_trimul"], "census": {"served": 10, "fallback": {}}, "gate": {"ok": True, "reason": "ok"}, "errors": []},
           "xl_report": {"installed": True, "levers": ["xl_trans", "xl_cond", "xl_free", "expandable_segments"], "disabled": [], "counters": {"trans_rowchunked_calls": 4, "cond_rowchunked_calls": 1, "free_events": 1},
                         "env": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}, "allocator": {"expandable_segments": True}, "errors": []}}
    log.update(extra); return log


SCHED_OK = {"park_z_init": "host_pinned", "msa_host": "rank0", "msa_host_where": "msa:host_pinned,has_deletion:host_pinned", "conf_ztrunk_entry": "parked:host_pinned",
            "conf_ztrunk": "parked:host_pinned,parked:host_pinned", "conf_ztrunk_passes": 2, "conf_pairstack": "inplace", "pairstack_transpose": "inplace",
            "msa_m": "token_sharded", "msa_cols": "0:2048/4076", "opm_a": "local"}


def test_placement_words_are_read_off_the_census_and_gate_the_exit():
    """The ×P placement words in force in a rank (tp_report.tp_exports) are READ off its schedule census (modes.placement_findings): acted ->
    nothing; a degraded form the core named (pageable park, resident shard) -> a FALLBACK (partial activation); a contradicted word -> a PROBLEM."""
    ex = dict(modes.TP_EXPORTS)
    assert modes.placement_findings(ex, SCHED_OK) == ([], [])
    assert modes.placement_findings({}, {}) == ([], [])                                       # no word in force: nothing expected
    assert modes.placement_findings(dict(ex, ROWPAIR_PARK_ZINIT="0", ROWPAIR_CONF_PARK_ZTRUNK="0", ROWPAIR_FREE_ZTRUNK="0", ROWPAIR_MSA_HOST="0", ROWPAIR_TRANSPOSE_INPLACE="0", ROWPAIR_MSA_M_LAYOUT="replicated"), dict(msa_m="replicated")) == ([], [])   # the caller's =0 parks: resident by request; the layout word has no =0 form (replicated is its opt-out)
    fb, pb = modes.placement_findings(dict(ex, ROWPAIR_MSA_M_LAYOUT="0"), SCHED_OK)             # a word that is not a layout: a problem here as at install (rowpair_msa.m_layout refuses it by name)
    assert fb == [] and len(pb) == 1 and "ROWPAIR_MSA_M_LAYOUT=0: not a layout" in pb[0]
    fb, pb = modes.placement_findings(ex, dict(SCHED_OK, msa_m="replicated"))               # the MSA representation's layout word is read off the census like the parks
    assert fb == [] and len(pb) == 1 and "msa_m=replicated under ROWPAIR_MSA_M_LAYOUT=token_sharded" in pb[0]
    assert modes.placement_findings(dict(ex, ROWPAIR_MSA_M_LAYOUT="replicated"), dict(SCHED_OK, msa_m="replicated")) == ([], [])
    fb, pb = modes.placement_findings(ex, {k: v for k, v in SCHED_OK.items() if k != "msa_m"})
    assert fb == [] and len(pb) == 1 and "msa_m=None" in pb[0]                                 # the MSA module never recorded its layout under the word: a problem
    fb, pb = modes.placement_findings(ex, dict(SCHED_OK, park_z_init="host_pageable:pin_alloc_failed"))
    assert len(fb) == 1 and "park_z_init=host_pageable:pin_alloc_failed" in fb[0] and pb == []
    fb, pb = modes.placement_findings(ex, dict(SCHED_OK, park_z_init="device"))
    assert len(fb) == 1 and "stayed on the device" in fb[0]
    fb, pb = modes.placement_findings(ex, dict(SCHED_OK, conf_ztrunk_entry="resident:not_parked:shared_storage", conf_ztrunk="resident:not_parked:shared_storage,resident:not_parked:shared_storage"))
    assert len(fb) == 2 and pb == []
    fb, pb = modes.placement_findings(ex, dict(SCHED_OK, msa_host="off"))
    assert fb == [] and len(pb) == 1 and "ROWPAIR_MSA_HOST=rank0" in pb[0]
    fb, pb = modes.placement_findings(ex, dict(SCHED_OK, conf_pairstack="copy"))
    assert len(pb) == 1 and "conf_pairstack=copy" in pb[0]
    fb, pb = modes.placement_findings(ex, dict(SCHED_OK, pairstack_transpose="p2p"))
    assert fb == [] and len(pb) == 1 and "ROWPAIR_TRANSPOSE_INPLACE=1" in pb[0]
    assert modes.placement_findings(dict(ex, ROWPAIR_TRANSPOSE_INPLACE="0"), dict(SCHED_OK, pairstack_transpose="p2p")) == ([], [])
    fb, pb = modes.placement_findings(ex, {k: v for k, v in SCHED_OK.items() if k != "pairstack_transpose"})
    assert fb == [] and len(pb) == 1 and "pairstack_transpose=None" in pb[0]                 # no pair block recorded its form under the word: a problem, never a pass
    fb, pb = modes.placement_findings(dict(ex, ROWPAIR_CONF_PARK_ZTRUNK="0"), dict(SCHED_OK, conf_ztrunk_entry="resident", conf_ztrunk="resident:free_declined:not_last,inplace"))
    assert (fb, pb) == ([], [])                                                                # FREE alone: in place on the last pass
    fb, pb = modes.placement_findings(dict(ex, ROWPAIR_CONF_PARK_ZTRUNK="0"), dict(SCHED_OK, conf_ztrunk_entry="resident", conf_ztrunk="resident:free_declined:dtype"))
    assert len(fb) == 1 and "did not embed in place" in fb[0]
    # through the evidence and the exit's partial census: rank 1 parked pageable -> a rowpair_tp fallback naming the rank; rank 0's msa_host off -> a problem
    ok = dict(TP_OK, tp_exports=ex, schedule=SCHED_OK)
    ev, problems = stack.evidence("big", _big_log(tp_report=ok), "", n_gpu=2, rank_worker_logs={1: _big_log(tp_report=dict(ok, rank=1))})
    assert not [p for p in problems if "n_gpu" in p] and ev["tp_placement_fallbacks"] == [] and not [f for f in stack.partial_activation("big", ev)[0] if f.startswith("rowpair_tp")]
    r1 = dict(ok, rank=1, schedule=dict(SCHED_OK, park_z_init="host_pageable:pin_alloc_failed"))
    ev, problems = stack.evidence("big", _big_log(tp_report=dict(ok, schedule=dict(SCHED_OK, msa_host="off"))), "", n_gpu=2, rank_worker_logs={1: _big_log(tp_report=r1)})
    assert any("rank 0: msa_host=off under ROWPAIR_MSA_HOST=rank0" in p for p in problems), problems
    assert ev["tp_placement_fallbacks"] == ["r1: park_z_init=host_pageable:pin_alloc_failed (pinned host refused: a pageable park, named by the core)"]
    fallbacks, _gates = stack.partial_activation("big", ev)
    assert any(f.startswith("rowpair_tp: r1: park_z_init=host_pageable") for f in fallbacks), fallbacks


def test_every_ranks_tp_report_is_required_evidence_above_p1():
    r1 = dict(TP_OK, rank=1, peak_alloc_gib=11.25)
    ev, problems = stack.evidence("big", _big_log(tp_report=TP_OK), "[boltz_flash_triattn_patch] " + json.dumps({"flash_calls": 8, "stock_calls": 0, "fallbacks": 0, "errors": 0}),
                                  n_gpu=2, rank_worker_logs={1: _big_log(tp_report=r1)})
    assert not [p for p in problems if "n_gpu" in p], problems
    assert ev["tp_peaks"] == "r0:11.5,r1:11.25"
    ev, problems = stack.evidence("big", _big_log(tp_report=TP_OK), "", n_gpu=2, rank_worker_logs={1: None})
    assert any("rank 1 has no tp_report" in p and p.startswith("reason=n_gpu_mismatch requested=2 active=1") for p in problems), problems
    ev, problems = stack.evidence("big", _big_log(tp_report=dict(TP_OK, calls={"trunk_rows": 0})), "", n_gpu=2, rank_worker_logs={1: _big_log(tp_report=r1)})
    assert any("ran no row-sharded trunk" in p for p in problems), problems
    ev, problems = stack.evidence("big", _big_log(tp_report=TP_OK), "", n_gpu=1)
    assert any("must install nothing at n_gpu=1" in p for p in problems), problems
    state = rep.lever_state("rowpair_tp", {"tp_report": TP_OK, "tp_peaks": "r0:11.5,r1:11.25"}, {"n_gpu": 2})
    assert state[0] == "on" and dict(state[2])["trunk_rows"] == 2 and dict(state[2])["gathers_named"] == 0
    assert rep.lever_state("rowpair_tp", {"tp_report": None}, {"n_gpu": 1})[:2] == ("skipped", "n_gpu=1")


def test_cli_refuses_n_gpu_above_1_outside_big_before_anything_runs(tmp_path):
    env = dict(os.environ); env.pop("BOLTZ2_OPT", None)
    for argv in (["pred", "--mode", "exact", "--n_gpu", "2", "--input", str(tmp_path / "a.yaml"), "--out_dir", str(tmp_path / "o")],
                 ["pred", "--mode", "off", "--n_gpu", "2", "--input", str(tmp_path / "a.yaml"), "--out_dir", str(tmp_path / "o")],
                 ["check", "--mode", "fast", "--n_gpu", "2"]):
        r = subprocess.run([sys.executable, "-m", "boltz2_opt"] + argv, capture_output=True, text=True, env=env, timeout=120)
        assert r.returncode == rep.EXIT_NOT_ACTIVE, (argv, r.returncode, r.stdout[-400:], r.stderr[-400:])
        assert BIG_WORDS in r.stdout, (argv, r.stdout[-600:])


def test_an_older_core_without_the_n_gpu_words_is_not_active_by_name(monkeypatch):
    monkeypatch.setattr(tp, "WORDS_MODULE", "opt_core.mem.no_such_ngpu_module")
    plan = stack.gate("exact", need_gpu=False, need_cache=False, check_pins=False, n_gpu=1)
    assert any(r.startswith("reason=core_missing:opt_core.mem.no_such_ngpu_module") for r in plan["reasons"]), plan["reasons"]
    assert rep.ngpu_tokens(1) == "n_gpu=<core_missing:opt_core.mem.no_such_ngpu_module>"


def _blocked_site(tmp_path, name, blocked):
    """A directory to put FIRST on PYTHONPATH whose ``sitecustomize`` installs a meta-path finder (ahead of every other finder, editable-install
    finders included) that makes the ``blocked`` modules unimportable: ``("opt_core",)`` = the core absent; ``("opt_core.mem.ngpu",)`` = an older
    core without the n_gpu words."""
    d = tmp_path / name; d.mkdir()
    (d / "sitecustomize.py").write_text(
        "import sys\nBLOCKED = %r\n\nclass _Blocked:\n    @staticmethod\n    def find_spec(name, path=None, target=None):\n"
        "        if any(name == b or name.startswith(b + '.') for b in BLOCKED):\n"
        "            raise ModuleNotFoundError(f\"No module named {name!r} (made unimportable for the test)\", name=name)\n        return None\n\n"
        "sys.meta_path.insert(0, _Blocked())\nfor m in [m for m in list(sys.modules) if any(m == b or m.startswith(b + '.') for b in BLOCKED)]:\n    del sys.modules[m]\n" % (tuple(blocked),))
    return str(d)


def _mismatched_site(tmp_path, name="mismatch", version="0.2.5"):
    """A directory to put FIRST on PYTHONPATH whose ``sitecustomize`` makes ``opt_core`` resolve (find_spec, ahead of every finder) to a
    shadow core of another version: ``<dir>/core/opt_core/__init__.py`` — the installed-but-wrong-version box."""
    d = tmp_path / name; core = d / "core" / "opt_core"; core.mkdir(parents=True)
    (core / "__init__.py").write_text(f'__version__ = "{version}"\n')
    (d / "sitecustomize.py").write_text(
        "import importlib.util, sys\nINIT = %r\n\nclass _Shadow:\n    @staticmethod\n    def find_spec(name, path=None, target=None):\n"
        "        if name == 'opt_core':\n            return importlib.util.spec_from_file_location('opt_core', INIT, submodule_search_locations=[INIT.rsplit('/', 1)[0]])\n"
        "        if name.startswith('opt_core.'):\n            raise ModuleNotFoundError(f\"No module named {name!r} (shadow core)\", name=name)\n"
        "        return None\n\nsys.meta_path.insert(0, _Shadow())\nfor m in [m for m in list(sys.modules) if m == 'opt_core' or m.startswith('opt_core.')]:\n    del sys.modules[m]\n" % (str(core / "__init__.py"),))
    return str(d)


def _kit_paths():
    pkg_parent = HERE
    while pkg_parent != "/" and not os.path.isdir(os.path.join(pkg_parent, "boltz2_opt")):
        pkg_parent = os.path.dirname(pkg_parent)
    kit = os.path.dirname(pkg_parent)
    assert os.path.isdir(os.path.join(pkg_parent, "boltz2_opt")) and os.path.isfile(os.path.join(kit, "run.sh")), (pkg_parent, kit)
    return pkg_parent, kit


def test_cli_route_refuses_by_name_at_an_absent_core_and_at_an_older_core(tmp_path):
    """`python -m boltz2_opt <verb>` with the shared core absent (-S with only opt/ on the path; and a meta-path block of opt_core over the
    installed one) or older (opt_core.mem.ngpu made unimportable): the NOT ACTIVE line names core_missing:<module>, exit 3, no traceback,
    nothing resolved."""
    pkg_parent, _ = _kit_paths()
    env = {k: v for k, v in os.environ.items() if not k.startswith("BOLTZ2_OPT") and k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-S", "-m", "boltz2_opt", "check", "--mode", "exact"], env={**env, "PYTHONPATH": pkg_parent}, capture_output=True, text=True, timeout=120)
    assert r.returncode == 3 and "[boltz2-opt] NOT ACTIVE: reason=core_missing:opt_core" in r.stderr and "Traceback" not in r.stderr, (r.returncode, r.stderr[-600:])
    base_pp = os.environ.get("PYTHONPATH", "")
    absent = _blocked_site(tmp_path, "absent", ("opt_core",)); older = _blocked_site(tmp_path, "older", ("opt_core.mem.ngpu",))
    import shutil
    script = shutil.which("boltz2-opt")                                                          # the console script (pyproject [project.scripts]) where the package is installed
    mism = _mismatched_site(tmp_path)
    for site_dir, word in ((absent, "core_missing:opt_core "), (older, "core_missing:opt_core.mem.ngpu "), (mism, "core_mismatch: opt_core pinned ")):
        pp = os.pathsep.join(p for p in (site_dir, pkg_parent, base_pp) if p)
        routes = [[sys.executable, "-m", "boltz2_opt", "check", "--mode", "exact"],
                  [sys.executable, "-m", "boltz2_opt", "pred", "--mode", "fast", "--input", str(tmp_path / "a.yaml"), "--out_dir", str(tmp_path / "o")],
                  [sys.executable, "-c", "import boltz2_opt; boltz2_opt.enable('exact'); print('reached')"]]      # the in-process route
        if script:
            routes.append([script, "check", "--mode", "exact"])
        for argv in routes:
            r = subprocess.run(argv, env={**env, "PYTHONPATH": pp}, capture_output=True, text=True, timeout=120)
            assert r.returncode == 3 and "[boltz2-opt] NOT ACTIVE: reason=" + word in r.stderr and "Traceback" not in r.stderr and "reached" not in r.stdout, (site_dir, argv[1:4], r.returncode, r.stdout[-300:], r.stderr[-600:])


def test_run_sh_verbs_refuse_by_name_at_an_absent_or_older_core(tmp_path):
    """`bash run.sh check` (the real entry behind every verb) with the core made absent or older by a meta-path block first on PYTHONPATH: rc 3
    and the NOT ACTIVE core_missing line. Runs where boltz is installed at the pin (run.sh checks the pins first); skipped by name elsewhere."""
    pkg_parent, kit = _kit_paths()
    pins = subprocess.run([sys.executable, "-I", os.path.join(kit, "stock", "check_pins.py"), "--quiet"], capture_output=True, text=True, timeout=120)
    if pins.returncode != 0:
        pytest.skip("boltz is not installed at the pin on this box (run.sh checks stock/PINS.json before the entry): the run.sh route is exercised on the stack image")
    env = {k: v for k, v in os.environ.items() if not k.startswith("BOLTZ2_OPT") and k != "PYTHONPATH"}
    base_pp = os.environ.get("PYTHONPATH", "")
    sites = (("absent", _blocked_site(tmp_path, "absent", ("opt_core",)), "core_missing:opt_core "),
             ("older", _blocked_site(tmp_path, "older", ("opt_core.mem.ngpu",)), "core_missing:opt_core.mem.ngpu "),
             ("mismatch", _mismatched_site(tmp_path), "core_mismatch: opt_core pinned "))
    for name, site_dir, word in sites:
        pp = os.pathsep.join(p for p in (site_dir, pkg_parent, base_pp) if p)
        r = subprocess.run(["bash", os.path.join(kit, "run.sh"), "check", "--mode", "exact"], env={**env, "PYTHONPATH": pp}, capture_output=True, text=True, timeout=180)
        assert r.returncode == 3 and "[boltz2-opt] NOT ACTIVE: reason=" + word in r.stderr and "Traceback" not in r.stderr, (name, r.returncode, r.stdout[-300:], r.stderr[-600:])


def test_the_requested_p_reaches_the_cores_launcher_and_a_short_rank_count_fails_closed(tmp_path, monkeypatch):
    """stack.run_worker at n_gpu=2 calls opt_core.mem.rowpair.launch.run_rank_processes with P=2 and one command line per rank (rank 1 on its
    own batch file); a launcher that stands fewer ranks than requested is NOT ACTIVE reason=n_gpu_mismatch (rc 3), never a pass."""
    from opt_core.mem.rowpair import launch as L
    seen = {}
    batch = tmp_path / "b.json"; batch.write_text(json.dumps({"tag": "pred", "out_dir": str(tmp_path / "o"), "kit_dir": str(tmp_path / "o" / "_kit"), "items": [], "seeds": [0]}))
    def fake(P, argv_of=None, **kw):
        seen["P"] = P; seen["argv"] = {r: list(argv_of(r)) for r in range(P)}; seen["kw"] = kw
        return [{"rank": r, "rc": 0} for r in range(P)]
    monkeypatch.setattr(L, "run_rank_processes", fake)
    rc = stack.run_worker("big", str(tmp_path), str(batch), str(tmp_path / "w.log"), env={"X": "1"}, n_gpu=2)
    assert rc == 0 and seen["P"] == 2 and seen["kw"]["isolate_devices"] is True
    b0 = seen["argv"][0][seen["argv"][0].index("--batch") + 1]; b1 = seen["argv"][1][seen["argv"][1].index("--batch") + 1]
    assert b0 == str(batch) and b1 != b0 and json.load(open(b1))["out_dir"].endswith("/ranks/rank1") and "--attach" in seen["argv"][1]
    assert "tp" in seen["argv"][0][seen["argv"][0].index("--attach") + 1].split(",")
    monkeypatch.setattr(L, "run_rank_processes", lambda P, argv_of=None, **kw: [{"rank": 0, "rc": 0}])      # a launcher that stood ONE rank for P=2
    rc = stack.run_worker("big", str(tmp_path), str(batch), str(tmp_path / "w2.log"), env={"X": "1"}, n_gpu=2)
    assert rc == 3 and "NOT ACTIVE: reason=n_gpu_mismatch requested=2 active=1" in open(tmp_path / "w2.log").read()
    ev, problems = stack.evidence("big", _big_log(tp_report=dict(TP_OK, n_gpu=1)), "", n_gpu=2, rank_worker_logs={1: _big_log(tp_report=dict(TP_OK, rank=1, n_gpu=1))})
    assert any(p.startswith("reason=n_gpu_mismatch requested=2 active=1") for p in problems), problems


def test_the_pth_route_refuses_by_name_at_a_mismatched_core(tmp_path):
    """The hook's own code path (the .pth line is `import boltz2_opt._autoload`) with BOLTZ2_OPT=<mode> and an importable opt_core of
    an older version: the kit's core pin gate prints `NOT ACTIVE: reason=core_mismatch: opt_core pinned >= v<want> at <path>, installed
    v0.2.5 at <root>` and the process exits 3. `-S`: no site, so an installed hook cannot have run first; the shadow finder is installed
    by hand."""
    pkg_parent, _ = _kit_paths()
    import opt_core
    core_parent = os.path.dirname(os.path.dirname(os.path.abspath(opt_core.__file__)))
    site_dir = _mismatched_site(tmp_path)
    env = {k: v for k, v in os.environ.items() if not k.startswith("BOLTZ2_OPT") and k != "PYTHONPATH"}
    code = (f"import sys; sys.path[:0] = [{site_dir!r}, {pkg_parent!r}, {core_parent!r}]; import sitecustomize; "
            "import os; os.environ['BOLTZ2_OPT'] = 'exact'; import boltz2_opt._autoload; print('reached')")
    r = subprocess.run([sys.executable, "-S", "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 3 and "[boltz2-opt] NOT ACTIVE: reason=core_mismatch: opt_core pinned " in r.stderr and "installed v0.2.5 at " in r.stderr and "reached" not in r.stdout and "Traceback" not in r.stderr, (r.returncode, r.stdout, r.stderr[-600:])


SEED_ARGV = [sys.executable, "-c", "import os; print('hashseed=%s' % os.environ.get('PYTHONHASHSEED'))"]


@pytest.mark.parametrize("case", ["inherited", "default"])
def test_every_rank_of_a_launch_starts_under_one_hash_seed(tmp_path, monkeypatch, capsys, case):
    """The ranks of a ×P run are separate interpreters: they must share ONE PYTHONHASHSEED (str-hash ordered bytes in upstream's data pipeline
    would differ otherwise). Driven through the kit's own launch wrapper, stack.run_worker at n_gpu=2 (its child_env, chdir and launcher call;
    the rank command is a seed printer here, the core's real run_rank_processes runs it on the CPU): an exported integer value reaches both
    ranks (`[boltz2-opt] RANKENV hashseed=123 source=inherited ranks=2` on stderr, under the kit's tag), an environment without one gives both ranks the launcher's `0`
    (`hashseed=0 source=default`). One place decides — the core's launcher; child_env sets no seed of its own."""
    from opt_core.mem.rowpair import launch as L
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    real = L.run_rank_processes
    monkeypatch.setattr(L, "run_rank_processes", lambda P, argv_of=None, **kw: real(P, argv_of=argv_of, **dict(kw, cpu_ok=True, isolate_devices=False)))
    monkeypatch.setattr(stack, "worker_command", lambda *a, **k: list(SEED_ARGV))
    batch = tmp_path / "b.json"; batch.write_text(json.dumps({"tag": "pred", "out_dir": str(tmp_path / "o"), "kit_dir": str(tmp_path / "o" / "_kit"), "items": [], "seeds": [0]}))
    if case == "inherited":
        monkeypatch.setenv("PYTHONHASHSEED", "123")
        env, want = None, "hashseed=123 source=inherited"                                   # run_worker builds child_env from this process's environment
        assert stack.child_env("big", n_gpu=2)["PYTHONHASHSEED"] == "123"
    else:
        env, want = stack.child_env("big", {"PATH": os.environ.get("PATH", "")}, n_gpu=2), "hashseed=0 source=default"   # an environment that exports no seed
        assert "PYTHONHASHSEED" not in env
    rc = stack.run_worker("big", str(tmp_path), str(batch), str(tmp_path / "w.log"), env=env, n_gpu=2)
    assert rc == 0
    err = capsys.readouterr().err
    assert f"[boltz2-opt] RANKENV {want} ranks=2" in err.splitlines(), err[-1500:]
    seeds = [re.findall(r"^hashseed=(\S*)$", open(tmp_path / "ranks" / f"rank{r}.log", errors="replace").read(), re.M) for r in (0, 1)]
    assert seeds == [[want.split()[0].split("=")[1]]] * 2, seeds


def test_the_xP_rows_words_are_the_rows_alone(monkeypatch):
    """The ×P line's words — BOLTZ_TP, the launcher tag, the sync policy word and every placement / kernel word of modes.TP_EXPORTS — are
    row words like BOLTZ_LEVERS: a caller's copy is stripped (stock/PINS.json must_be_absent_prefixes covers each name) and child_env assigns
    the row's value; an operational word the row does not own (ROWPAIR_NCCL_TIMEOUT_S) passes through; at n_gpu = 1 none is set."""
    monkeypatch.setenv("BOLTZ_CACHE", "/cache")
    strip = stack.load_pins()["stock_environment"]["must_be_absent_prefixes"]
    words = dict(modes.TP_EXPORTS, **{registry.LEVERS["rowpair_tp"]["sync_switch"]: registry.LEVERS["rowpair_tp"]["sync_default"],
                                      registry.LEVERS["rowpair_tp"]["switch"]: "2", stack.RANK_TAG_ENV: rep.TAG})
    for name in words:
        assert any(name.startswith(p) for p in strip), f"{name} is not under stock/PINS.json must_be_absent_prefixes"
    caller = {"PATH": "/usr/bin", "ROWPAIR_NCCL_TIMEOUT_S": "1800", **{k: ("off" if v != "off" else "0") for k, v in words.items()}}
    env2 = stack.child_env("big", base=dict(caller), n_gpu=2)
    assert {k: env2[k] for k in words} == words and env2["ROWPAIR_NCCL_TIMEOUT_S"] == "1800"
    env1 = stack.child_env("big", base=dict(caller), n_gpu=1)
    assert not any(k in env1 for k in words) and env1["ROWPAIR_NCCL_TIMEOUT_S"] == "1800"
    assert stack.tp_exports(env2, 2) == dict(modes.TP_EXPORTS) and stack.tp_exports(env1, 1) == {}
