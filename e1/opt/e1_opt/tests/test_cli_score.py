"""The wrapper command end to end on the mocked box (in-process `cli.main`, the tool in a subprocess with the stub upstream): mode off writes
the tool's file through the clean stock runner; mode exact relays the kit's own KIT / LEVER lines and the KERNELS proof; the refusals by name
(mode/env disagreement, variant/env disagreement, pins drift, a --model-name off the pins, fast / big) print NOT ACTIVE / ERROR and exit
3 / 2, the EXIT line last; one assay per run in upstream's own flags, every other option handed to the tool verbatim; a refusal by name in
the tool's process (the kit's KitRefused) is the run's NOT ACTIVE, exit 3; a plain failure is exit 1."""
import os

import sys

from e1_opt import kit_score, cli, outputs, report, stack
from e1_opt.tests import _stubs

KIT_STUB = '''
import os, subprocess, sys
args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
assert not any(k.startswith(("E1_OPT", "E1_KIT", "MODEL_OPT")) or k == "E1_VARIANT" for k in os.environ), sorted(os.environ)   # the kit's child gets the stock's environment: no kit variable at all
print("[e1-opt] KIT vTEST mode=eager size=300m card=h100 gpu=\\"NVIDIA H100 80GB HBM3\\" pin=W8 levers=2/2", flush=True)
print("[e1-opt] LEVER name=rmsnorm_autotune_pin W=8 source=\\"size pin\\" state=on", flush=True)
print("[e1-opt] LEVER name=P1_precast state=on", flush=True)
{kernels_print}
rc = subprocess.call([sys.executable, "-m", "E1.tools.score", *args])
sys.exit(rc)
'''.replace("{kernels_print}", _stubs.KERNELS_PRINT)


def _box(monkeypatch, tmp_path, **kw):
    box = _stubs.setup_box(monkeypatch, tmp_path, **kw)
    _stubs.patch_pins_for_stub(monkeypatch, box["pins"], box["variant"], box["caches"])
    return box


def _stub_kit_child(monkeypatch, tmp_path, body=KIT_STUB):
    """The kit's child replaced by a stub script (the real child arms the real kit on a GPU)."""
    stub = tmp_path / "kit_child_stub.py"
    stub.write_text(body)
    monkeypatch.setattr(kit_score, "command", lambda python, tool_args, **kw: [python, str(stub), "--", *tool_args])
    return stub


def _run(capsys, argv):
    rc = cli.main(argv)
    out = capsys.readouterr()
    return rc, out.out.splitlines(), out.err


def _one(items_dir: str, item: str, out: str) -> list:
    """upstream's three flags for one item of a written items directory."""
    return ["--parent-path", os.path.join(items_dir, item, "parent.fasta"), "--mutants-path", os.path.join(items_dir, item, "mutants.fasta"),
            "--output-path", os.path.join(out, item, "scores.csv")]


def _exit(lines) -> dict:
    m = [report.RE_EXIT.match(s) for s in lines if report.RE_EXIT.match(s)]
    assert len(m) == 1 and report.RE_EXIT.match(lines[-1]), lines                # exactly one EXIT line, the last line of the run
    return m[0].groupdict()


def test_score_off_end_to_end(monkeypatch, tmp_path, capsys):
    box = _box(monkeypatch, tmp_path)
    items = _stubs.write_items(str(tmp_path / "in"), names=("item_a",))
    out = str(tmp_path / "out")
    rc, lines, err = _run(capsys, ["score", "--variant", box["variant"], "--mode", "off", *_one(items, "item_a", out)])
    assert rc == 0, (lines, err)
    assert report.RE_OFF.match(lines[0]) and report.RE_READY.match(lines[1]), lines[:2]
    m = report.RE_STRIPPED.match(lines[2])                     # the variables stripped from the tool's process, printed once (MODEL_OPT at least: the box sets it)
    assert m and "MODEL_OPT" in m.group("names").split(","), lines[2]
    kinds = [("env" if report.RE_ENV_CLEAN.match(s) else "hdr" if s == "STOCK_ENV_CHECK:" else "score" if report.RE_SCORE.match(s) else
              "kernels" if report.RE_KERNELS.match(s) else "exit" if report.RE_EXIT.match(s) else "?") for s in lines[3:]]
    assert kinds == ["env", "hdr", "kernels", "score", "exit"], lines   # the tool's ENV-CLEAN proof + STOCK_ENV_CHECK header, its KERNELS proof line before its first forward, the wall, EXIT last
    k = report.RE_KERNELS.match(lines[5])
    assert k.group("runner", "route") == ("stock", "stock") and k.group("flash_attn").startswith("engaged:flash_attn@") and "require=" not in lines[5], lines[5]
    assert [report.RE_SCORE.match(s).group("item") for s in lines if report.RE_SCORE.match(s)] == ["scores.csv"] and all(s.startswith("[e1-opt stock] score") for s in lines if report.RE_SCORE.match(s))
    lst = outputs.listing(outputs.scores_path(out, "item_a"))
    assert lst["present"] and lst["header"] == "id,context_id,score" and lst["n_rows"] == 3 and len(lst["sha256"]) == 64
    x = _exit(lines)
    assert (x["mode"], x["variant"], x["complete"], x["kernels_fallback"], x["rc"]) == ("off", box["variant"], "1", "none", "0"), lines[-1]
    assert os.listdir(out) == ["item_a"] and os.listdir(os.path.join(out, "item_a")) == ["scores.csv"]   # the tool's file only: the kit writes no file of its own




def test_score_exact_relays_the_kits_lines(monkeypatch, tmp_path, capsys):
    box = _box(monkeypatch, tmp_path)
    _stub_kit_child(monkeypatch, tmp_path)
    items = _stubs.write_items(str(tmp_path / "in"), names=("only",))
    out = str(tmp_path / "out")
    rc, lines, err = _run(capsys, ["score", "--variant", box["variant"], *_one(items, "only", out), "--det", "0"])
    assert rc == 0, (lines, err)
    m = report.RE_ACTIVE.match(lines[0])
    assert m and m.group("mode") == "exact" and m.group("variant") == box["variant"] and m.group("kit_mode") == "eager" and m.group("gpu") == box["gpu"][0]
    assert lines[0].endswith(" det=0 pins=ok route=score") and "items=" not in lines[0] and "multi=" not in lines[0]
    assert report.RE_READY.match(lines[1]) and report.RE_STRIPPED.match(lines[2])
    assert lines[3].startswith("[e1-opt] KIT vTEST mode=eager ") and lines[4].startswith("[e1-opt] LEVER name=rmsnorm_autotune_pin ") and lines[5] == "[e1-opt] LEVER name=P1_precast state=on"
    k = report.RE_KERNELS.match(lines[6])                                    # the kit child's KERNELS proof line, relayed verbatim after the kit's own lines
    assert k and k.group("runner", "route", "site") == ("exact", "exact", "kit_attn"), lines[6]
    assert report.RE_SCORE.match(lines[7]) and lines[7].startswith("[e1-opt] score scores.csv ")   # the assay is named by its output file
    lst = outputs.listing(os.path.join(out, "only", "scores.csv"))
    assert lst["present"] and lst["n_rows"] == 3 and os.listdir(os.path.join(out, "only")) == ["scores.csv"]   # the tool's file only
    x = _exit(lines)
    assert (x["mode"], x["complete"], x["kernels_fallback"], x["rc"]) == ("exact", "1", "none", "0"), lines[-1]
    # upstream's --model-name names the same pinned checkpoint as --variant: taken alone, and refused when it names anything else
    stack._reset_for_tests()
    rc, lines, err = _run(capsys, ["score", "--model-name", f"Profluent-Bio/E1-{box['variant']}", *_one(items, "only", str(tmp_path / "out2")), "--det", "0"])
    assert rc == 0 and report.RE_ACTIVE.match(lines[0]).group("variant") == box["variant"], (lines, err)
    rc, lines, err = _run(capsys, ["score", "--model-name", "someone/finetuned-E1", *_one(items, "only", str(tmp_path / "out3"))])
    assert rc == 2 and "--model-name 'someone/finetuned-E1': the kit serves the pinned checkpoints only" in err, err
    other = next(v for v in ("150m", "300m", "600m") if v != box["variant"])
    rc, lines, err = _run(capsys, ["score", "--model-name", f"Profluent-Bio/E1-{other}", "--variant", box["variant"], *_one(items, "only", str(tmp_path / "out4"))])
    assert rc == 2 and lines[0].startswith(f"[e1-opt] NOT ACTIVE: --model-name Profluent-Bio/E1-{other} disagrees with --variant {box['variant']}"), (lines, err)


def test_one_assay_in_upstreams_flags_and_everything_else_passes_through(monkeypatch, tmp_path, capsys):
    box = _box(monkeypatch, tmp_path)
    items = _stubs.write_items(str(tmp_path / "in"), names=("only",))
    out = str(tmp_path / "out")
    rc, _, err = _run(capsys, ["score", "--variant", box["variant"], "--parent-path", os.path.join(items, "only", "parent.fasta"), "--output-path", os.path.join(out, "s.csv")])
    assert rc == 2 and "missing: --mutants-path" in err, err
    for gone in ("--items", "--output-dir", "--kit-multi", "--allow-partial"):          # none of these is an option of the wrapper: each is handed to the tool, which refuses it (its own parser, rc 2) — the run fails, exit 1
        stack._reset_for_tests()
        rc, lines, err = _run(capsys, ["score", "--variant", box["variant"], "--mode", "off", *_one(items, "only", str(tmp_path / gone.strip("-"))), gone, "x"])
        assert rc == 1 and _exit(lines)["complete"] == "0" and "unrecognized arguments" in err, (gone, lines, err)
    seen = tmp_path / "seen.txt"
    stub = _stub_kit_child(monkeypatch, tmp_path, KIT_STUB.replace("rc = subprocess.call(", f"open({str(seen)!r}, 'w').write(repr(args)); rc = subprocess.call("))
    stack._reset_for_tests()
    rc, lines, err = _run(capsys, ["score", "--variant", box["variant"], *_one(items, "only", str(tmp_path / "o2")), "--context-path", "ctx.fasta",
                                 "--scoring-method", "masked_marginal", "--max-batch-tokens", "4096", "--context-reduction", "max"])
    args = eval(open(seen).read())
    assert args[0] == "--model-name" and args[2:8:2] == ["--parent-path", "--mutants-path", "--output-path"]        # the pinned snapshot, then upstream's three flags
    assert args[8:] == ["--context-path", "ctx.fasta", "--scoring-method", "masked_marginal", "--max-batch-tokens", "4096", "--context-reduction", "max"]   # then everything else, verbatim, in the order given
    assert stub.exists()


def test_the_kit_child_command_is_the_runner_around_upstreams_module():
    cmd = kit_score.command("py", ["--a", "1"], det=1, variant="300m", pins_path="/p.py")
    assert cmd[:3] == ["py", "-m", "e1_opt.kit_score"] and cmd[cmd.index("--det") + 1] == "1" and cmd[cmd.index("--variant") + 1] == "300m" and cmd[-3:] == ["--", "--a", "1"]
    assert "--script" not in cmd and "--mode" not in cmd and kit_score.TOOL_MODULE == "E1.tools.score"
    assert kit_score.RE_KIT_LINE.match("[e1-opt] LEVER name=P1_precast state=on") and kit_score.RE_KIT_LINE.match("[e1-opt] KIT v2.0 mode=eager size=600m") and not kit_score.RE_KIT_LINE.match("[e1-opt exact] KERNELS route=exact")
    assert not hasattr(kit_score, "resolved_mode")                                            # one kit mode: nothing to read back



def test_refusals_by_name(monkeypatch, tmp_path, capsys):
    box = _box(monkeypatch, tmp_path)
    items = _stubs.write_items(str(tmp_path / "in"), names=("x",))
    out = str(tmp_path / "out")
    base = ["score", "--variant", box["variant"], *_one(items, "x", out)]
    # mode vs env disagreement
    monkeypatch.setenv("E1_OPT", "off")
    rc, lines, _ = _run(capsys, base + ["--mode", "exact"])
    assert rc == 2 and lines[0].startswith("[e1-opt] NOT ACTIVE: mode exact disagrees with E1_OPT=off")   # a usage refusal (two sources), exit 2
    monkeypatch.delenv("E1_OPT")
    # variant vs env disagreement
    stack._reset_for_tests()
    monkeypatch.setenv("E1_VARIANT", "600m")
    rc, lines, _ = _run(capsys, base)
    assert rc == 2 and lines[0].startswith("[e1-opt] NOT ACTIVE: variant ") and "disagrees with E1_VARIANT=600m" in lines[0]
    monkeypatch.delenv("E1_VARIANT")
    # no variant at all
    stack._reset_for_tests()
    rc, lines, _ = _run(capsys, ["score", *_one(items, "x", out)])
    assert rc == 3 and lines[0].startswith("[e1-opt] NOT ACTIVE: no variant:") and report.RE_EXIT.match(lines[1]), lines
    assert _exit(lines)["rc"] == "3" and _exit(lines)["complete"] == "0"          # the EXIT line is the last line of a refused run too
    # a component variable set by the caller is dropped from the children but never a refusal; an unknown mode is a usage error
    stack._reset_for_tests()
    rc, lines, err = _run(capsys, base + ["--mode", "quick"])
    assert rc == 2 and "unknown mode" in err
    assert not os.listdir(os.path.join(out, "x")) if os.path.isdir(os.path.join(out, "x")) else True


def test_an_untested_card_runs_and_is_named_on_the_line(monkeypatch, tmp_path, capsys):
    """A GPU outside the tested classes, a memory reading off its class's nominal, a config targeting another class: NAMED on the
    activation line (`card=untested`, `notes="…"`, `pins=ok`), never refused — the run proceeds and scores."""
    name, (cls, mib) = next(iter(stack.CARDS.items()))
    box = _box(monkeypatch, tmp_path, gpu=("NVIDIA H100 NVL", 97871))
    kit_stub = tmp_path / "kit_score_stub.py"
    kit_stub.write_text(KIT_STUB)
    monkeypatch.setattr(kit_score, "command", lambda python, tool_args, **kw: [python, str(kit_stub), "--", *tool_args])
    items = _stubs.write_items(str(tmp_path / "in"), names=("x",))
    out = str(tmp_path / "out")
    rc, lines, err = _run(capsys, ["score", "--variant", box["variant"], *_one(items, "x", out), "--det", "0"])
    assert rc == 0, (lines, err)
    m = report.RE_ACTIVE.match(lines[0])
    assert m and " card=untested " in lines[0] and " pins=ok " in lines[0], lines[0]
    assert m.group("notes").startswith("card 'NVIDIA H100 NVL' (cc 9.0, 97871 MiB) is outside the tested classes (a100|h100|h200)"), lines[0]
    assert not any(report.RE_NOT_ACTIVE.match(s) for s in lines) and _exit(lines)["rc"] == "0" and outputs.listing(outputs.scores_path(out, "x"))["present"]
    stack._reset_for_tests()
    box = _box(monkeypatch, tmp_path / "b", gpu=(name, mib + int(0.03 * mib)))   # beyond the 2 % tolerance (a same-name card of another memory): named, runs
    rc, lines, _ = _run(capsys, ["score", "--variant", box["variant"], "--mode", "off", *_one(items, "x", str(tmp_path / "out2"))])
    off = report.RE_OFF.match(lines[0])
    assert rc == 0 and off and f"reports {mib + int(0.03 * mib)} MiB, its class {cls} has {mib} MiB" in off.group("notes"), lines[0]   # the off line carries the notes too
    stack._reset_for_tests()
    box = _box(monkeypatch, tmp_path / "c")
    monkeypatch.setenv(stack.ENV_TARGET_GPU, "A100")                          # a config targeting another class than this card's: named, the config's constants used as given
    rc, lines, _ = _run(capsys, ["score", "--variant", box["variant"], "--mode", "off", *_one(items, "x", str(tmp_path / "out3"))])
    assert rc == 0 and "the config targets a100 (MODEL_OPT_TARGET_GPU), this card is class h100" in report.RE_OFF.match(lines[0]).group("notes"), lines[0]


def test_another_stock_commit_refuses_and_the_rest_is_named(monkeypatch, tmp_path, capsys):
    """The ONE refusal by pin: the installed E1 is not the pinned stock (another commit changes what `stock` means) — NOT ACTIVE, exit 3.
    The stub box's placeholder kernel snapshot is not one: it is a note (`notes=`), as the placeholder weights' digest is the weights line's word."""
    box = _stubs.setup_box(monkeypatch, tmp_path, commit="0" * 40)   # E1 installed from another commit; placeholder weights / kernel unpatched
    items = _stubs.write_items(str(tmp_path / "in"), names=("x",))
    out = str(tmp_path / "out")
    rc, lines, err = _run(capsys, ["score", "--variant", box["variant"], "--mode", "off", *_one(items, "x", out)])
    assert rc == 3 and lines[0].startswith("[e1-opt] NOT ACTIVE: not the pinned stock: E1 installed from commit"), lines[0]
    assert "weights" not in lines[0] and "hub kernel snapshot" not in lines[0]         # the one refusal, by name — the placeholder kernel and weights are not refusals:
    rep = stack.status()                                                               # the kernel snapshot is a note, the weights digest is worded on the weights line (stderr)
    w = rep["pins"]["checks"]["weights"]
    assert w["ok"] and w["detail"]["word"] == box["pins"].WEIGHTS_WORDS[1] and w["detail"]["sha256"] == box["caches"]["weights_sha256"]
    assert err.splitlines()[0] == f"[e1-opt] {box['pins'].weights_words(w['detail'])}" and err.count(box["pins"].WEIGHTS_NOTICE) == 1, err
    assert len(rep["pins"]["refusals"]) == 1 and any(n.startswith("hub kernel snapshot") for n in rep["pins"]["notes"]) and rep["pins"]["checks"]["stack"]["ok"]
    assert _exit(lines)["rc"] == "3"
    stack._reset_for_tests()
    box = _stubs.setup_box(monkeypatch, tmp_path / "b")                              # the stock commit at its pin, the placeholder kernel still unpatched: named, the run proceeds
    rc, lines, err = _run(capsys, ["score", "--variant", box["variant"], "--mode", "off", *_one(items, "x", str(tmp_path / "out2"))])
    off = report.RE_OFF.match(lines[0])
    assert rc == 0 and off and off.group("notes").startswith("hub kernel snapshot ") and "layer_norm.py does not hash to the pin" in off.group("notes"), (lines, err)


def test_usage_errors_and_help(monkeypatch, tmp_path, capsys):
    box = _box(monkeypatch, tmp_path)
    items = _stubs.write_items(str(tmp_path / "in"), names=("x",))
    out = str(tmp_path / "out")
    rc, lines, _ = _run(capsys, ["score", "--variant", box["variant"], "--mode", "off", *_one(items, "x", out), "--max-batch-tokens", "4096", "--scoring-method", "masked_marginal"])
    assert rc == 0 and _exit(lines)["complete"] == "1"                                 # upstream's optional flags pass through to the tool (its parser took them: the assay scored)
    rc, lines, _ = _run(capsys, ["score", "--variant", box["variant"], "--mode", "big", *_one(items, "x", str(tmp_path / "o2"))])
    assert rc == 3 and lines[0] == "[e1-opt] NOT ACTIVE: mode big: not shipped by this kit (modes: off|exact)", lines   # the family's other words: refused by name
    rc, lines, _ = _run(capsys, ["check", "--variant", box["variant"], "--mode", "fast"])
    assert rc == 3 and lines[0].startswith("[e1-opt] NOT ACTIVE: mode fast: not shipped"), lines
    assert cli.main([]) == 2 and cli.main(["help"]) == 0 and cli.main(["bogus"]) == 2


# ------------------------------------------------------------------------------------------------------- the exit rule on a refusal
# The kit refuses BY NAME in the tool's process (its fail-loud class raised uncaught: the `KitRefused: <reason>` line on the child's stderr) when
# a lever of the set cannot be put in force there: the run's NOT ACTIVE line carries the reason, exit 3, no scores; a tool process that fails
# without a refusal is a plain failure (exit 1). The stub below refuses or fails by the assay's name (its output directory), read from knobs.txt.
REFUSING_KIT_STUB = '''
import os, subprocess, sys
args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
item = os.path.basename(os.path.dirname(args[args.index("--output-path") + 1]))
class KitRefused(RuntimeError):
    pass
knobs = dict(line.split() for line in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "knobs.txt")).read().splitlines() if line.strip())
if item in knobs:
    if knobs[item] == "refuse":
        raise KitRefused("kit vTEST: the command names size '300m' but the model's embedding width says '150m'")
    sys.exit(1)                                                              # a failure that is not a refusal: no scores.csv, no named line
print("[e1-opt] KIT vTEST mode=eager size=300m card=h100 gpu=\\"x\\" pin=W8 levers=1/1", flush=True)
{kernels_print}
rc = subprocess.call([sys.executable, "-m", "E1.tools.score", *args])
sys.exit(rc)
'''.replace("{kernels_print}", _stubs.KERNELS_PRINT)


def _refusing_box(monkeypatch, tmp_path, item: str, what: str = "refuse"):
    box = _stubs.setup_box(monkeypatch, tmp_path)
    _stubs.patch_pins_for_stub(monkeypatch, box["pins"], box["variant"], box["caches"])
    stub = tmp_path / "kit_score_stub.py"
    stub.write_text(REFUSING_KIT_STUB)
    (tmp_path / "knobs.txt").write_text(f"{item} {what}\n")
    monkeypatch.setattr(kit_score, "command", lambda python, tool_args, **kw: [python, str(stub), "--", *tool_args])   # the kit's child: the stub script directly
    return box


def _run_fresh(capsys, argv):
    stack._REPORT = None                                                              # one activation per process: each run here is its own
    rc = cli.main(argv)
    out = capsys.readouterr()
    return rc, out.out.splitlines(), out.err


def test_an_item_the_kit_refuses_by_name_is_the_runs_refusal_exit_3(monkeypatch, tmp_path, capsys):
    """An item whose command the kit refuses by name in its own process (its KitRefused line on stderr: a lever of the mode's set cannot run
    there, an input outside what it serves) is the RUN's refusal — a mode is all of its levers: the family's `NOT ACTIVE: item <item>:
    <reason>` line, exit 3, no scores.csv; never a run that goes on under the mode's name with fewer levers."""
    box = _refusing_box(monkeypatch, tmp_path, "item_a")                                        # the assay's item refuses by name
    items = _stubs.write_items(str(tmp_path / "in"), names=("item_a",))
    out = str(tmp_path / "out")
    rc, lines, err = _run_fresh(capsys, ["score", "--variant", box["variant"], *_one(items, "item_a", out), "--det", "0"])
    assert rc == cli.EXIT_NOT_ACTIVE, (lines, err)
    assert "KitRefused: kit vTEST: the command names size" in err                        # the child's stderr relayed verbatim (the traceback)
    na = [s for s in lines if report.RE_NOT_ACTIVE.match(s)]
    assert na == ["[e1-opt] NOT ACTIVE: kit vTEST: the command names size '300m' but the model's embedding width says '150m'"], lines   # the run's line: the kit's own reason, literal
    assert not os.path.exists(outputs.scores_path(out, "item_a"))
    x = _exit(lines)
    assert (x["complete"], x["rc"]) == ("0", "3"), lines[-1]


def test_a_failed_item_without_a_refusal_is_incomplete_exit_1_not_partial(monkeypatch, tmp_path, capsys):
    box = _refusing_box(monkeypatch, tmp_path, "item_b", what="fail")                            # the item exits 1 with no refusal line
    items = _stubs.write_items(str(tmp_path / "in"), names=("item_a", "item_b"))
    out = str(tmp_path / "out")
    rc, lines, err = _run_fresh(capsys, ["score", "--variant", box["variant"], *_one(items, "item_b", out), "--det", "0"])
    assert rc == cli.EXIT_FAIL, (lines, err)
    assert not any(report.RE_NOT_ACTIVE.match(s) for s in lines)                    # no refusal worded: a plain failure
    x = _exit(lines)
    assert (x["complete"], x["rc"]) == ("0", "1"), lines[-1]
    # a complete run words neither
    rc, lines, err = _run_fresh(capsys, ["score", "--variant", box["variant"], *_one(items, "item_a", out), "--det", "0"])
    assert rc == cli.EXIT_OK, (lines, err)
    x = _exit(lines)
    assert (x["complete"], x["rc"]) == ("1", "0") and outputs.listing(outputs.scores_path(out, "item_a"))["n_rows"] == 3




def test_refusal_is_read_from_the_childs_stderr(monkeypatch, tmp_path, capsys):
    """relay() collects the kit's uncaught refusal line from the child's stderr; refusal() reads the reason back."""
    box = _refusing_box(monkeypatch, tmp_path, "only")
    items = _stubs.write_items(str(tmp_path / "in"), names=("only",))
    idir = tmp_path / "out" / "only"
    idir.mkdir(parents=True)
    cmd = kit_score.command(sys.executable, ["--model-name", "/snap/E1-300m", "--parent-path", os.path.join(items, "only", "parent.fasta"),
                                             "--mutants-path", os.path.join(items, "only", "mutants.fasta"), "--output-path", str(idir / "scores.csv")], det=0, variant=box["variant"], pins_path=stack.pins_path())
    rc, wall, kit_lines = kit_score.relay(cmd, _stubs.child_env(box), cwd=str(idir))
    err = capsys.readouterr().err
    assert rc == 1 and "KitRefused" in err and kit_score.refusal(kit_lines) == "kit vTEST: the command names size '300m' but the model's embedding width says '150m'"
    assert kit_score.refusal(["[e1-opt] KIT vTEST mode=eager"]) is None and wall > 0


def test_kit_refused_name_is_the_kits_own_class():
    tree = _stubs.tree_or_skip()
    root = os.path.join(tree, "opt", "forward")
    if root not in sys.path:
        sys.path.insert(0, root)
    import ast
    src = open(os.path.join(root, "engines", "e1", "kits", "eager", "__init__.py"), encoding="utf-8").read()   # the kit's entry (not importable on the CPU bed: torch-side imports) re-exports v1's class
    assert "KitRefused = v1.KitRefused" in src
    v1src = ast.parse(open(os.path.join(root, "engines", "e1", "kits", "v1", "__init__.py"), encoding="utf-8").read())
    assert any(isinstance(n, ast.ClassDef) and n.name == kit_score.KIT_REFUSED for n in ast.walk(v1src))
    assert kit_score.RE_KIT_REFUSED.match("engines.e1.kits.v1.KitRefused: kit v2.0: x").group("reason") == "kit v2.0: x"
    assert kit_score.RE_KIT_REFUSED.match("KitRefused: y").group("reason") == "y" and kit_score.RE_KIT_REFUSED.match("RuntimeError: KitRefused: y") is None
