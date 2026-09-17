"""The kit is checked present (the git commit identifies it, not a stored hash); the mode table is the kit's writer
table, read by AST; the gate refuses by name."""
import os
import sys

import pytest

from .. import kit, modes, registry, report
from ._fixtures import REAL_KIT, TREE_OPT, clean_env, make_root


def test_kit_is_present():
    v = kit.check_kit()
    assert v["ok"], v
    assert v["frozen"] == "v17"
    assert v["declared_frozen_dirs"] is not None and "v17" in v["declared_frozen_dirs"]
    assert os.path.basename(v["entry"]) == "borzoi_sad.py"
    assert os.path.isfile(os.path.join(REAL_KIT, "v17", "kitlib", "forward.py"))


def test_kit_missing_frozen_dir_is_refused(tmp_path, monkeypatch):
    """No stored hash to tamper with (the git commit identifies the kit) — the gate's negative case is a MISSING frozen dir/entry,
    not a modified one (kit.check_kit checks presence and shape, not bytes)."""
    r = make_root(tmp_path)
    clean_env(monkeypatch, r["env"])
    os.remove(os.path.join(r["kit_dir"], "v17", "borzoi_sad.py"))
    v = kit.check_kit()
    assert not v["ok"] and "kit entry not found" in v["reason"]
    rep = modes.resolve("exact", read_gpu=False)
    assert rep["ok"] is False and not rep["active"] and "kit entry not found" in rep["reason"]


def test_writer_table_is_read_from_the_kit_bytes():
    wt = kit.writer_table()
    assert wt["writer"] == "exact" and "tier 1" in wt["tier"].lower()   # the kit's one writer (v17/kitlib/writer.py WRITER, TIER), read by AST
    assert "SAD" in wt["stats"] and "D2" in wt["stats"] and "REF" not in wt["stats"]
    table = modes.kit_mode_table()
    assert list(table) == ["exact"]
    assert table["exact"]["env"] == kit.KIT_COMPOSITION == {"KIT_FWD": "1"} and table["exact"]["writer"] == wt["writer"]
    assert "tier 1" in table["exact"]["tier"].lower()


def test_composition_names_only_switches_the_kit_reads():
    names = kit.kit_env_names()
    assert set(kit.KIT_COMPOSITION) <= set(names)                    # KIT_FWD is read by v17/kitlib/forward.py
    assert set(names) == {"KIT_FWD", "KIT_STAMP_DIR"}                  # the literal KIT_* reads in the frozen dir: the forward switch + the stamp hand-off (by modes.kit_env_overrides)


def test_modes_and_default():
    assert modes.MODES == ("off", "exact") and modes.DEFAULT_MODE == "exact"            # the one kit mode is the default; off by name
    assert modes.KIT_MODES == ("exact",)
    assert modes.check_mode(None) == "exact" and modes.check_mode(" Exact ") == "exact" and modes.check_mode("off") == "off"
    for bad in ("turbo", "stock", "fast"):                        # no fast mode ships; "stock" is not a mode name
        with pytest.raises(modes.ActivationError):
            modes.check_mode(bad)


def test_resolve_exact_and_off_reports(tmp_path, monkeypatch):
    r = make_root(tmp_path)
    clean_env(monkeypatch, r["env"])
    rep = modes.resolve("exact", read_gpu=False)
    assert rep["ok"] and rep["active"] and rep["env"] == {"KIT_FWD": "1"} and rep["writer"] == kit.writer_table()["writer"] == "exact" and rep["det"] is False
    assert rep["entry"] == os.path.join(r["kit_dir"], "v17", "borzoi_sad.py") and rep["kit"]["name"] == "pipeline_tf.v17"
    assert rep["stock"]["pinned"] is True and rep["stock"]["entry"] == r["stock_sad"]
    assert set(rep["levers"]) == set(registry.by_mode("exact")) and {"copy_free_forward", "graph_forward", "post_writer"} <= set(rep["levers"])
    off = modes.resolve("off", read_gpu=False)
    assert off["ok"] and off["active"] is False and off["entry"] == r["stock_sad"] and off["kit"] is None and off["env"] == {}
    assert modes.status()["mode"] == "off"



def test_det_recipe_is_the_only_extra_environment(tmp_path, monkeypatch):
    r = make_root(tmp_path)
    clean_env(monkeypatch, r["env"])
    rep = modes.resolve("exact", read_gpu=False, det=True)
    assert rep["ok"] and rep["det"] is True and rep["env"] == {"KIT_FWD": "1", **modes.DET_RECIPE}
    off = modes.resolve("off", read_gpu=False, det=True)
    assert off["ok"] and off["env"] == dict(modes.DET_RECIPE)
    assert set(modes.DET_RECIPE) == {"NPY_DISABLE_CPU_FEATURES", "OPENBLAS_CORETYPE", "OPENBLAS_NUM_THREADS", "TF_CUDNN_USE_AUTOTUNE"} and modes.DET_RECIPE["TF_CUDNN_USE_AUTOTUNE"] == "0"


def test_kit_switch_in_the_environment_is_refused_by_name(tmp_path, monkeypatch):
    r = make_root(tmp_path)
    clean_env(monkeypatch, {**r["env"], "KIT_STAMP_DIR": "/tmp/x", "KIT_FWD": "1"})
    rep = modes.resolve("exact", read_gpu=False)
    assert rep["ok"] is False and rep["kit_env_overrides"] == {"KIT_FWD": "1", "KIT_STAMP_DIR": "/tmp/x"} and "KIT_FWD=1" in rep["reason"]
    monkeypatch.delenv("KIT_FWD"); monkeypatch.delenv("KIT_STAMP_DIR")
    assert modes.resolve("exact", read_gpu=False)["ok"] is True
    monkeypatch.setenv("KIT_FWD", "0")                                     # the composition's own switch set by the user: another composition, refused
    assert modes.resolve("exact", read_gpu=False)["ok"] is False
    monkeypatch.delenv("KIT_FWD")
    monkeypatch.setenv("KIT_HOME_VAR", "BORZOI_OPT_KIT"); monkeypatch.setenv("KIT_PYTHONPATH", "opt")    # another tool's KIT_-prefixed bookkeeping: not a switch of this kit
    rep2 = modes.resolve("exact", read_gpu=False)
    assert rep2["ok"] is True and rep2["kit_env_overrides"] == {}
    assert "KIT_HOME_VAR" not in kit.kit_env_names()


def test_stock_pin_mismatch_is_named(tmp_path, monkeypatch):
    r = make_root(tmp_path, pin_sha="0" * 64)
    clean_env(monkeypatch, r["env"])
    off = modes.resolve("off", read_gpu=False)
    assert off["ok"] is False and "not the pinned stock" in off["reason"]
    ex = modes.resolve("exact", read_gpu=False)
    assert ex["ok"] and any("stock entry" in n for n in ex["notes"])       # the kit mode runs the kit's entry; the stock's state is a note


def test_enable_installs_the_library_levers_in_process(tmp_path, monkeypatch):
    """The Python route: `borzoi_opt.enable("exact")` in a program that imports the stock library installs the kit's forward call on
    SeqNN.__call__ and the LUT one-hot on dna.dna_1hot (here a stand-in baskerville; a child interpreter so this process stays clean),
    before or after the library import; `off` enables nothing."""
    import subprocess, textwrap
    r = make_root(tmp_path)
    clean_env(monkeypatch, r["env"])
    lib = tmp_path / "lib" / "baskerville"; lib.mkdir(parents=True)
    (lib / "__init__.py").write_text("")
    (lib / "dna.py").write_text("def dna_1hot(seq, seq_len=None, n_uniform=False, n_sample=False):\n    return 'stock one-hot'\n")
    (lib / "seqnn.py").write_text("class SeqNN:\n    ensemble = None\n    model = None\n    def __call__(self, x, head_i=None, dtype='float32'):\n        return 'stock call'\n")
    env = {**os.environ, "MODEL_OPT": r["root"], "BORZOI_DIR": r["env"]["BORZOI_DIR"], "PYTHONPATH": os.pathsep.join([str(tmp_path / "lib"), TREE_OPT])}
    env.pop("BORZOI_OPT", None)
    for order in ("after", "before"):
        code = textwrap.dedent(f"""
            import sys, borzoi_opt
            if {order == 'after'!r}: from baskerville import dna, seqnn
            rep = borzoi_opt.enable("exact")
            from baskerville import dna, seqnn
            print("ACTIVE", rep["active"], rep["route"], rep.get("levers_installed"), dna.dna_1hot.__module__, seqnn.SeqNN.__call__.__module__)
            off = borzoi_opt.enable("off")
            print("OFF", off["active"], "nothing to enable" in off["reason"])
        """)
        p = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
        assert p.returncode == 0, p.stderr[-2000:]
        installed = "['baskerville.dna', 'baskerville.seqnn']" if order == "after" else "[]"
        assert f"ACTIVE True hook {installed} kitlib.onehot kitlib.forward" in p.stdout, (order, p.stdout, p.stderr[-1500:])
        assert "OFF False True" in p.stdout
        assert p.stderr.count("[borzoi-opt] ACTIVE mode=exact route=hook program=") == 1 and "levers=graph_forward,copy_free_forward,onehot_lut" in p.stderr, p.stderr
        assert "[borzoi-opt] EXIT mode=exact route=hook" in p.stderr and "onehot=LUT" in p.stderr


def test_registry_shape():
    for name, lv in registry.LEVERS.items():
        assert lv.name == name and lv.cls in ("datapath", "forward", "observability") and lv.file and lv.lines and lv.what and lv.switch
        assert lv.modes == ("exact",)
    assert set(registry.by_mode("exact")) == set(registry.LEVERS)
    assert registry.by_mode("fast") == {} and registry.by_mode("off") == {}
    for lv in registry.LEVERS.values():
        assert os.path.isfile(os.path.join(REAL_KIT, "v17", lv.file)), lv.file
    fwd = registry.LEVERS["copy_free_forward"]
    assert "KIT_FWD=1" in fwd.switch and fwd.tier == "exact"
    pw = registry.LEVERS["post_writer"]
    assert pw.switch.startswith("none") and pw.tier == "exact"
    gf = registry.LEVERS["graph_forward"]
    assert "KIT_FWD=1" in gf.switch and gf.tier == "exact" and "tf.function" in gf.what and "XLA" in gf.what


def test_environment_differences_are_named_not_refused(tmp_path, monkeypatch):
    """GPU class outside the tested ones, a GPU that is not the configuration's target, and stack versions that differ from stock/PINS.json
    are notes on the activation line of every mode (``notes=…``) — ``ok`` stays true; nothing is refused over the environment."""
    r = make_root(tmp_path)
    clean_env(monkeypatch, r["env"])
    import json as _json
    pins_path = os.path.join(r["root"], "stock", "PINS.json")
    pins = _json.load(open(pins_path))
    pins["python"] = "0.0.1"                                            # never the running interpreter
    pins["pins"] = {"tensorflow": "2.15.1", "numpy": "1.24.4"}
    _json.dump(pins, open(pins_path, "w"), indent=1)
    monkeypatch.setattr(modes, "stack_versions", lambda: {"tensorflow": None, "numpy": "9.9.9", "h5py": "3.10.0"})
    monkeypatch.setattr(modes, "gpu_info", lambda: {"name": "NVIDIA L4", "memory_mib": 23034, "cc": "8.9"})
    for mode in ("exact", "off"):
        rep = modes.resolve(mode, root=r["root"], environ={**os.environ, modes.TARGET_GPU_ENV: "H100"}, read_gpu=True)
        assert rep["ok"], rep.get("reason")
        notes = " | ".join(rep["notes"])
        assert "is not the configuration's target H100" in notes
        assert "compute capability 8.9 (NVIDIA L4) is outside the tested classes (9.0 H100, 8.0 A100)" in notes
        assert "python " in notes and "(tested 0.0.1)" in notes
        assert "tensorflow not installed (tested 2.15.1)" in notes and "numpy 9.9.9 (tested 1.24.4)" in notes
        assert "h5py" not in notes                                      # not in the fixture's pins table: nothing to compare
        assert rep["stack_drift"][0].startswith("python ")
        line = report.activation_line(rep)
        assert line.startswith("[borzoi-opt] " + ("OFF" if mode == "off" else "ACTIVE mode=" + mode)) and " notes=" in line and "outside the tested classes" in line
    # the tested card, no target, the pinned stack: no notes at all
    monkeypatch.setattr(modes, "stack_versions", lambda: {"tensorflow": "2.15.1", "numpy": "1.24.4"})
    monkeypatch.setattr(modes, "gpu_info", lambda: {"name": "NVIDIA H100", "memory_mib": 81559, "cc": "9.0"})
    monkeypatch.setattr(modes.platform, "python_version", lambda: "0.0.1")
    rep = modes.resolve("exact", root=r["root"], environ=dict(os.environ), read_gpu=True)
    assert rep["ok"] and rep["stack_drift"] == []
    assert not any(("tested" in n or "target" in n) for n in rep["notes"])   # (a machine without baskerville still notes that, rightly)
