"""The kit route's parent side without a GPU: the activation records read back (complete / partial / absent / another mode's), the
HOOK / ENV-KIT / KIT-RECORD / EXIT line grammar, the hook probe in an environment where the .pth is not installed (this test interpreter)."""
import json
import os
import sys

from complexa_opt import kit_design, modes, report, stack, stock_design


def _rec(d, pid, mode="exact", final=True, states=None, steps=2, fwd=800, peak=12.5):
    lev = {n: {"state": (states or {}).get(n, "on"), "tier": "exact", "counters": {"served": 3}} for n in modes.levers_of(mode)}
    doc = {"pid": pid, "mode": mode, "final": final, "levers": lev, "predict_steps": steps, "forwards": fwd, "peak_alloc_gib": peak}
    (d / f"kit_{pid}.json").write_text(json.dumps(doc))


def test_records_complete_partial_absent_wrong_mode(tmp_path):
    d = tmp_path / "kit_records"; d.mkdir()
    s = kit_design.read_records(str(d), "exact", modes.levers_of("exact"))
    assert s["state"] == "absent" and s["processes"] == 0 and s["partial"] == ["no_activation_record"] and s["levers_on"] == []
    _rec(d, 11)
    s = kit_design.read_records(str(d), "exact", modes.levers_of("exact"))
    assert s["state"] == "complete" and s["processes"] == 1 and s["partial"] == [] and s["levers_on"] == list(modes.levers_of("exact")) and s["predict_steps"] == 2 and s["forwards"] == 800 and s["peak_alloc_gib"] == 12.5
    _rec(d, 12, states={"loop_desync": "skipped"})
    s = kit_design.read_records(str(d), "exact", modes.levers_of("exact"))
    assert s["state"] == "partial" and s["processes"] == 2 and s["partial"] == ["loop_desync"] and "loop_desync" in s["levers_skipped"] and s["predict_steps"] == 4
    (d / "kit_12.json").unlink()
    _rec(d, 13, final=False)                                                          # a process that armed but never reported (died, or still running)
    s = kit_design.read_records(str(d), "exact", modes.levers_of("exact"))
    assert s["state"] == "partial" and s["final"] == 1 and s["processes"] == 2
    (d / "kit_13.json").unlink(); (d / "kit_11.json").unlink()
    _rec(d, 14, mode="fast")
    s = kit_design.read_records(str(d), "exact", modes.levers_of("exact"))
    assert s["state"] == "partial" and s["wrong_mode_pids"] == [14]
    line = report.kit_record_line(s)
    assert line.startswith("[complexa-opt] KIT-RECORD processes=1 mode=exact state=partial levers_on=") and " predict_steps=2 forwards=800 peak_alloc_gib=12.5" in line


def test_kit_lines_grammar():
    assert report.hook_line({"python": "/usr/bin/python3", "pth": "/sp/complexa_opt_autoload.pth", "package": "/t/complexa/opt/complexa_opt", "finder_armed": True}, "big") == \
        "[complexa-opt] HOOK ok python=/usr/bin/python3 pth=/sp/complexa_opt_autoload.pth package=/t/complexa/opt/complexa_opt finder=armed mode=big"
    removed = {"variables": ["COMPLEXA_OPT", "PYTHONSAFEPATH"], "pythonpath": []}
    ln = report.env_kit_line(removed, {"COMPLEXA_OPT": "fast", "COMPLEXA_OPT_RECORD": "/o/kit_records"}, ["CKPT_PATH", "COMPLEXA_OPT", "COMPLEXA_OPT_RECORD", "LOCAL_CODE_PATH"])
    assert ln == "[complexa-opt] ENV-KIT ok (2 variables removed: COMPLEXA_OPT,PYTHONSAFEPATH; exported: COMPLEXA_OPT=fast,COMPLEXA_OPT_RECORD=/o/kit_records; present: CKPT_PATH,COMPLEXA_OPT,COMPLEXA_OPT_RECORD,LOCAL_CODE_PATH)"
    x = report.exit_line(mode="exact", route="kit", rc=0, written=16, expected=16, wall=12.34, incomplete=None, exit_code=3, partial=["loop_desync"])
    assert x.endswith(" mode=exact route=kit rc=0 designs_written=16 designs_expected=16 wall=12.3 partial=loop_desync exit=3") and x.startswith("[complexa-opt] EXIT pid=")
    x = report.exit_line(mode="fast", route="kit", rc=0, written=16, expected=16, wall=1.0, incomplete=None, exit_code=0)
    assert " partial=" not in x and x.endswith(" exit=0")


def test_hook_probe_reports_what_this_interpreter_holds(tmp_path):
    """The probe child comes up in the kit child's environment and reports whether complexa_opt_autoload.pth armed the hook there: armed
    only if this interpreter has the kit INSTALLED (a tree-only PYTHONPATH import lays no .pth) — either way a report, and armed implies loaded."""
    env, _ = stock_design.clean_env(dict(os.environ, MODEL_OPT=stack.tree_home()), export={"COMPLEXA_OPT": "exact"}, keep_kit_paths=True)
    p = stack.hook_probe(env, python=sys.executable, cwd=str(tmp_path))
    assert p["rc"] == 0, p
    assert set(p) >= {"python", "pth", "autoload_loaded", "package", "finder_armed", "present", "rc", "stderr"}
    assert (not p["finder_armed"]) or (p["autoload_loaded"] and p["pth"] and p["package"])
    assert (p["pth"] is None) == (not p["autoload_loaded"])                    # the .pth is what loads it
    assert "COMPLEXA_OPT" in p["present"]


def test_kit_child_environment_is_the_clean_one_plus_the_two_exported_names(tmp_path):
    base = {"PATH": "/usr/bin", "HOME": "/h", "COMPLEXA_OPT": "big", "COMPLEXA_KIT_X": "1", "NVIDIA_TF32_OVERRIDE": "0", "PYTHONSAFEPATH": "1", "CKPT_PATH": "/w", "LOCAL_CODE_PATH": "/c",
            "MODEL_OPT": stack.tree_home()}
    base["PYTHONPATH"] = os.pathsep.join([os.path.join(stack.tree_home(), "opt"), "/elsewhere"])
    env, removed = stock_design.clean_env(base, export={"COMPLEXA_OPT": "exact", "COMPLEXA_OPT_RECORD": "/o/kit_records"}, keep_kit_paths=True)
    assert env["PYTHONPATH"] == base["PYTHONPATH"] and removed["pythonpath"] == []                 # the kit route keeps this tree on the child's path
    env_s, removed_s = stock_design.clean_env(base)
    assert env_s["PYTHONPATH"] == "/elsewhere" and removed_s["pythonpath"] == [os.path.join(stack.tree_home(), "opt")]   # the stock route strips it
    assert env["COMPLEXA_OPT"] == "exact" and env["COMPLEXA_OPT_RECORD"] == "/o/kit_records" and env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert "COMPLEXA_KIT_X" not in env and "NVIDIA_TF32_OVERRIDE" not in env and "PYTHONSAFEPATH" not in env
    assert removed["variables"] == ["COMPLEXA_KIT_X", "COMPLEXA_OPT", "NVIDIA_TF32_OVERRIDE", "PYTHONSAFEPATH"]
