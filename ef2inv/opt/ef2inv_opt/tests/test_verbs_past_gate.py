"""Every GPU verb runs past the card comparison to its subprocess call (the comparison stubbed, the subprocess stubbed): design and warm
each reach their launch with the arguments the verb composes — on the pinned card silently, on any other card after exactly ONE
``NOTE hardware …; proceeding`` line with the same arguments and the same exit code. The class of defect this guards is code after the
comparison that no test executes, and a card word that changes a route. The comparison itself (hardware.py: the card-reading adapter over
the shared core's torch-free probe, and the name+memory.total match against the pinned card) is tested directly at the bottom."""
import json
import os
import types

import pytest

from .. import cli, hardware as HW, modes as MD

OK_GATE = {"ok": True, "name": "NVIDIA H100 80GB HBM3", "nvidia_smi_memory_total_mib": 81559, "expected_name": "NVIDIA H100 80GB HBM3", "expected_mib": 81559, "reason": ""}
A100_GATE = dict(OK_GATE, ok=False, name="NVIDIA A100-SXM4-80GB", nvidia_smi_memory_total_mib=81920,
                 reason="hardware 'NVIDIA A100-SXM4-80GB, 81920' is not 'NVIDIA H100 80GB HBM3, 81559' (name, nvidia-smi memory.total MiB)")
UNREAD_GATE = dict(OK_GATE, ok=False, name=None, nvidia_smi_memory_total_mib=None,
                   reason="hardware 'nvidia-smi unavailable (FileNotFoundError)' is not 'NVIDIA H100 80GB HBM3, 81559' (name, nvidia-smi memory.total MiB)")


def note_lines(err: str):
    return [ln for ln in err.splitlines() if ln.startswith("[ef2inv-opt] NOTE ")]


@pytest.fixture
def tree(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_OPT", cli._model_opt())
    monkeypatch.setattr(cli.HW, "gate", lambda pins, query=None: dict(OK_GATE))
    return tmp_path


def _ns(**kw):
    return types.SimpleNamespace(**kw)


def test_warm_reaches_the_launch(tree, monkeypatch):
    calls = {}
    def run(argv, env, log, timeout=None):
        calls["run"] = {"argv": argv, "env": env}; return {"exit_code": 0}
    monkeypatch.setattr(cli.L, "run", run)
    monkeypatch.setattr(cli, "_cookbook_stock", lambda: "/nonexistent/binder_design.py")
    out = str(tree / "warm")
    tgt = tree / "target.fasta"; tgt.write_text(">t\nACDEFGHIK\n")
    rc = cli.cmd_warm(_ns(mode="fast", target_name="t", target_sequence="ACDEFGHIK", binder_len=80, out=out))
    assert rc == 0 and "--warm-only" in calls["run"]["argv"] and calls["run"]["env"].get("EF2_FAST_KIT") == "agk3"
    assert json.load(open(os.path.join(out, "warm_result.json")))["exit_code"] == 0


def test_warm_on_another_card_notes_once_and_reaches_the_launch(tree, monkeypatch, capsys):
    """The pinned card: no NOTE, the launch; another card: ONE NOTE line, the SAME launch (argv and env equal) and the same exit code."""
    runs = {}
    def run(argv, env, log, timeout=None):
        runs.setdefault("argv", []).append(argv); runs.setdefault("env", []).append(env); return {"exit_code": 0}
    monkeypatch.setattr(cli.L, "run", run)
    monkeypatch.setattr(cli, "_cookbook_stock", lambda: "/nonexistent/binder_design.py")
    tgt = tree / "target.fasta"; tgt.write_text(">t\nACDEFGHIK\n")
    rcs, notes = [], []
    for gate, sub in ((OK_GATE, "h100"), (A100_GATE, "a100")):
        monkeypatch.setattr(cli.HW, "gate", lambda pins, query=None, g=gate: dict(g))
        rcs.append(cli.cmd_warm(_ns(mode="fast", target_name="t", target_sequence="ACDEFGHIK", binder_len=80, out=str(tree / f"warm_{sub}"))))
        notes.append(note_lines(capsys.readouterr().err))
    assert rcs == [0, 0] and notes[0] == [] and len(notes[1]) == 1 and A100_GATE["reason"] in notes[1][0] and "untested card: memory limits may differ; proceeding" in notes[1][0]
    strip = lambda argv: [x for x in argv if "/warm_" not in x]                       # the two launches differ in --out only
    assert strip(runs["argv"][0]) == strip(runs["argv"][1]) and runs["env"][0] == runs["env"][1] and runs["env"][1].get("EF2_FAST_KIT") == "agk3"


def test_design_reaches_the_launch(tree, monkeypatch):
    calls = {}
    facts = {"problems": [], "esm": "3.4.0@d0207ea3", "torch": "x", "device": "NVIDIA H100 80GB HBM3", "kit_switch_var": "EF2_FAST_KIT", "hardware": dict(OK_GATE), "weights": {}}
    monkeypatch.setattr(cli, "_check_facts", lambda *a, **k: dict(facts))
    monkeypatch.setattr(cli, "_cookbook_stock", lambda: __file__)                      # any existing file stands in for the stock file here
    def run(argv, env, log, timeout=None):
        calls["run"] = {"argv": argv, "env": env}; return {"exit_code": 0, "wall_s": 1.0}
    def write(*a, **k):
        calls["write"] = k; return {"missing_outputs": [], "refusals": [], "evidence": "applied"}
    monkeypatch.setattr(cli.L, "run", run)
    monkeypatch.setattr(cli.M, "write", write)
    out = str(tree / "design")
    tgt = tree / "target.fasta"; tgt.write_text(">t\nACDEFGHIK\n")
    rc = cli.cmd_design(_ns(mode="off", levers=None, target_name="t", target_sequence="ACDEFGHIK", binder_len=80, seed=0, out=out, target_hotspot_ids=None,
                            det=0, allow_partial=False))
    assert rc == 0
    argv = calls["run"]["argv"]
    assert "--mode" in argv and argv[argv.index("--mode") + 1] == "off" and "--binder-len" in argv and argv[argv.index("--target-name") + 1] == "t" and argv[argv.index("--target-sequence") + 1] == "ACDEFGHIK" and "--tag" not in argv
    assert not any(k.startswith(("EF2_FAST_KIT", "EF2INV_OPT")) for k in calls["run"]["env"]) and calls["write"]["facts"] is not None


def test_design_refused_by_name_exits_3_and_allow_partial_opts_out(tree, monkeypatch, capsys):
    """The arm's named refusal (exit 3: a proof, a pin, a named fallback) is the design's exit 3 — no traceback in the parent — and
    `--allow-partial` is the one opt-out: the refusal is printed as allowed and the verb exits 0."""
    facts = {"problems": [], "esm": "3.4.0@d0207ea3", "torch": "x", "device": "NVIDIA H100 80GB HBM3", "kit_switch_var": "EF2_FAST_KIT", "hardware": dict(OK_GATE), "weights": {}}
    monkeypatch.setattr(cli, "_check_facts", lambda *a, **k: dict(facts))
    monkeypatch.setattr(cli, "_cookbook_stock", lambda: __file__)
    monkeypatch.setattr(cli.L, "run", lambda argv, env, log, timeout=None: {"exit_code": 3, "wall_s": 1.0})
    monkeypatch.setattr(cli.M, "write", lambda *a, **k: {"missing_outputs": ["design.pdb"], "refusals": ["target: 'x' is not a preset target; provide target_sequence"], "evidence": "stock"})
    ns = lambda allow: _ns(mode="off", levers=None, target_name="x", target_sequence=None, binder_len=80, seed=0, out=str(tree / f"refused_{int(allow)}"), target_hotspot_ids=None, det=0, allow_partial=allow)
    assert cli.cmd_design(ns(False)) == 3
    assert cli.cmd_design(ns(True)) == 0 and "partial by name, allowed" in capsys.readouterr().err


def _fake_torch(name="NVIDIA A100-SXM4-80GB"):
    m = types.ModuleType("torch"); m.__version__ = "2.11.0+cu128"
    m.cuda = types.SimpleNamespace(is_available=lambda: True, get_device_name=lambda i=0: name)
    m.version = types.SimpleNamespace(cuda="12.8")
    return m


def test_design_check_facts_word_another_card_as_a_note_not_a_problem(monkeypatch, tmp_path, capsys):
    """The design's launch facts (cli._check_facts, require_gpu) on a CUDA box whose card is not the pinned card: `problems` carries no
    hardware entry, ONE NOTE line is logged, the facts keep the line (`hardware.note`, which opt_manifest.json `stack.hardware_gate` records);
    on the pinned card: no line, no `note` key — the same facts as ever. Then `design` reaches its launch on both with equal argv and env."""
    import sys
    tree = tmp_path
    monkeypatch.setenv("MODEL_OPT", cli._model_opt())                                  # the real hardware.gate runs (no gate stub): the card reading is injected below
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout=json.dumps({"ok": True, "facts": {"esm_version": "3.4.0"}}), stderr="", returncode=0))
    monkeypatch.setattr(cli.MD, "jit_cache_key", lambda: "torch2.11.0-cu128-sm90")
    monkeypatch.setattr(cli, "_cookbook_stock", lambda: __file__)
    monkeypatch.setattr(cli, "CORE_FACTS", {"pinned": "stub", "installed": "stub"})
    pins = {"pinned_stack": {"gpu": {"name": "NVIDIA H100 80GB HBM3", "nvidia_smi_memory_total_mib": 81559}}, "upstream": {"commit": "d0207ea3"}}
    facts_by_card = {}
    for reading in (("NVIDIA H100 80GB HBM3", 81559), ("NVIDIA A100-SXM4-80GB", 81920), (None, None)):
        monkeypatch.setattr(cli.HW, "card", lambda probe=None, r=reading: (r[0], r[1], f"{r[0]}, {r[1]}" if r[0] else "nvidia-smi rc=9"))
        facts = cli._check_facts(MD.MODES["fast"], cli._model_opt(), pins, require_gpu=True, weights=False)
        notes = note_lines(capsys.readouterr().err)
        assert facts["problems"] == [] and facts["ok"], facts["problems"]                # no problem on any card: the design proceeds
        if reading[0] == "NVIDIA H100 80GB HBM3":
            assert notes == [] and "note" not in facts["hardware"] and facts["hardware"]["ok"]
        else:
            assert len(notes) == 1 and facts["hardware"]["note"] == notes[0][len("[ef2inv-opt] "):] and not facts["hardware"]["ok"]
            assert (f"'{reading[0]}, {reading[1]}'" if reading[0] else "'nvidia-smi rc=9'") in notes[0] and notes[0].endswith("; proceeding")
        facts_by_card[reading[0]] = facts
    # the design verb itself: the same launch on the pinned card and on another card
    runs = []
    monkeypatch.setattr(cli.L, "run", lambda argv, env, log, timeout=None: runs.append((argv, env)) or {"exit_code": 0, "wall_s": 1.0})
    monkeypatch.setattr(cli.M, "write", lambda *a, **k: {"missing_outputs": [], "refusals": [], "evidence": "applied"})
    tgt = tree / "target.fasta"; tgt.write_text(">t\nACDEFGHIK\n")
    rcs = []
    for card in ("NVIDIA H100 80GB HBM3", "NVIDIA A100-SXM4-80GB"):
        monkeypatch.setattr(cli, "_check_facts", lambda *a, _f=facts_by_card[card], **k: dict(_f))
        rcs.append(cli.cmd_design(_ns(mode="fast", levers=None, target_name="t", target_sequence="ACDEFGHIK", binder_len=80, seed=0, out=str(tree / "design_x"),
                                      target_hotspot_ids=None, det=0, allow_partial=False)))
    assert rcs == [0, 0] and len(runs) == 2 and runs[0] == runs[1] and runs[1][1].get("EF2_FAST_KIT") == "agk3"


# ---- hardware.py itself: the card reading adapter and the comparison the verbs above stub out
GATE_PINS = {"pinned_stack": {"gpu": {"name": "NVIDIA H100 80GB HBM3", "nvidia_smi_memory_total_mib": 81559}}}


def probe(name, mib, why="nvidia-smi"):
    return {"name": name, "memory_mib": mib, "probe": why}


def test_card_adapter(monkeypatch):
    """hardware.card() reshapes a probe dict to (name, mib, words-for-report); with no injection it runs the shared core's own torch-free
    probe (opt_core.gates.nvidia_smi_probe, stubbed here because the test host may carry no card)."""
    assert HW.card(probe("NVIDIA H100 80GB HBM3", 81559)) == ("NVIDIA H100 80GB HBM3", 81559, "NVIDIA H100 80GB HBM3, 81559")
    assert HW.card(probe(None, None, "nvidia-smi unavailable (FileNotFoundError)")) == (None, None, "nvidia-smi unavailable (FileNotFoundError)")
    from opt_core import gates as G
    monkeypatch.setattr(G, "nvidia_smi_probe", lambda *a, **k: probe("NVIDIA H100 80GB HBM3", 81559))
    assert HW.card() == ("NVIDIA H100 80GB HBM3", 81559, "NVIDIA H100 80GB HBM3, 81559")


def test_gate(monkeypatch):
    """hardware.gate() compares GPU 0 against the pinned name (containment either way, case-insensitive) and the pinned nvidia-smi
    memory.total (one reading, or the `|`-separated set a config names): ok on a match, not ok (reason naming both readings) on any mismatch — and an EF2INV_GPU / EF2INV_GPU_MIB /
    MODEL_OPT_TARGET_GPU override moves what is pinned (``H100`` names the real card ``NVIDIA H100 80GB HBM3``)."""
    for v in ("EF2INV_GPU", "EF2INV_GPU_MIB", "MODEL_OPT_TARGET_GPU"):
        monkeypatch.delenv(v, raising=False)
    assert HW.gate(GATE_PINS, probe("NVIDIA H100 80GB HBM3", 81559))["ok"]
    for bad in (probe("NVIDIA H100 PCIe", 81559), probe("NVIDIA H100 80GB HBM3", 81079), probe("NVIDIA A100-SXM4-80GB", 81920), probe(None, None, "nvidia-smi rc=9")):
        g = HW.gate(GATE_PINS, bad)
        assert not g["ok"] and g["reason"].startswith("hardware ") and "81559" in g["reason"]
    monkeypatch.setenv("EF2INV_GPU", "NVIDIA H200"); monkeypatch.setenv("EF2INV_GPU_MIB", "143771")
    assert HW.gate(GATE_PINS, probe("NVIDIA H200", 143771))["ok"] and not HW.gate(GATE_PINS, probe("NVIDIA H100 80GB HBM3", 81559))["ok"]
    monkeypatch.delenv("EF2INV_GPU"); monkeypatch.delenv("EF2INV_GPU_MIB"); monkeypatch.setenv("MODEL_OPT_TARGET_GPU", "H100")   # the short class name a launcher exports
    assert HW.gate(GATE_PINS, probe("NVIDIA H100 80GB HBM3", 81559))["ok"] and HW.gate(GATE_PINS, probe("nvidia h100 80gb hbm3", 81559))["ok"]
    assert not HW.gate(GATE_PINS, probe("NVIDIA H100 NVL", 95830))["ok"] and not HW.gate(GATE_PINS, probe("NVIDIA H200", 81559))["ok"]
    assert [HW.word(HW.gate(GATE_PINS, p)) for p in (probe("NVIDIA H100 80GB HBM3", 81559), probe("NVIDIA A100-SXM4-80GB", 81920), probe(None, None, "nvidia-smi rc=9"))] == ["pinned", "untested", "unread"]
    monkeypatch.setenv("MODEL_OPT_TARGET_GPU", "A100"); monkeypatch.setenv("EF2INV_GPU_MIB", "81920|40960")                      # configs/a100.env's expectation: the 80 GB form factors and the 40 GB part name it, an H100 does not
    assert [HW.word(HW.gate(GATE_PINS, probe(n, m))) for n, m in (("NVIDIA A100-SXM4-80GB", 81920), ("NVIDIA A100 80GB PCIe", 81920), ("NVIDIA A100-SXM4-40GB", 40960), ("NVIDIA H100 80GB HBM3", 81559), ("NVIDIA A100-SXM4-40GB", 81920))] == ["pinned", "pinned", "pinned", "untested", "pinned"]
    g = HW.gate(GATE_PINS, probe("NVIDIA A100-PCIE-40GB", 40536)); assert not g["ok"] and "81920|40960" in g["reason"] and g["expected_mib"] == [81920, 40960]   # a reading outside the set is worded with the whole set
    monkeypatch.setenv("EF2INV_GPU_MIB", "81920"); assert HW.gate(GATE_PINS, probe("NVIDIA A100-SXM4-80GB", 81920))["expected_mib"] == 81920                   # one reading: the record keeps the integer
