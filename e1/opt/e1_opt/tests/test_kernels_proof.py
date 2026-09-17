"""The KERNELS proof (accel.py): the one reader of the accelerators a model process BOUND and its line — on the mocked box of _stubs.py
(a stub upstream whose modules mirror E1's dispatch sites; no GPU, no torch).

  * the tables agree: the mode table states the kernels of every running mode, the reader's accelerator set, the line's field order
  * the line grammar round-trips through report.py's regexes; a line that names a fallback carries no refusal word
  * the reader reads OBJECTS: the bound varlen function, upstream's env switch, the class attribute the model calls through, the hub kernel
    module and its Autotuner, the compiled flex callable — each fallback worded, upstream's own warning trapped into the word
  * install() imports nothing of upstream's and fires once, at the first predictor construction; a kit mode's route is its own name
  * under a kit mode an accelerator that fell back is the mode's refusal by name at the first construction (NOT ACTIVE, SystemExit(3), before
    any forward); under mode off the stock child runs upstream's own fallback route, every KERNELS line words it and the run's EXIT line
    names it (`kernels_fallback=`), nothing is refused; the hub kernel's fallback is worded from upstream's own warning; the tool's options
    pass through verbatim; an exact child that printed no KERNELS line is the run's refusal; the stack-imports check judges flash_attn and
    kernels by importing them and names what does not import
"""
import logging
import os
import sys
import types

import pytest

from .. import _names, accel, cli, kit_score, modes, outputs, report, stack
from . import _stubs


def _one(items_dir, item, out):
    return ["--parent-path", os.path.join(items_dir, item, "parent.fasta"), "--mutants-path", os.path.join(items_dir, item, "mutants.fasta"), "--output-path", os.path.join(out, item, "scores.csv")]


def _box(monkeypatch, tmp_path, **kw):
    box = _stubs.setup_box(monkeypatch, tmp_path, **kw)
    _stubs.patch_pins_for_stub(monkeypatch, box["pins"], box["variant"], box["caches"])
    accel._reset_for_tests()
    return box


def _run(capsys, argv):
    stack._REPORT = None
    rc = cli.main(argv)
    out = capsys.readouterr()
    return rc, out.out.splitlines(), out.err


def _expected_words(pins):
    return {"flash_attn": f"engaged:flash_attn@{pins.STACK['flash_attn']}", "hub_layernorm": f"engaged:triton_layer_norm@{pins.KERNEL['rev'][:8]}",
            "flex_attention": f"engaged:torch.compile@{pins.STACK['torch_version_str']}"}


# ----------------------------------------------------------------------------------------------------------------- the tables
def test_tables_agree_and_every_running_mode_states_its_kernels():
    pins = _stubs.pins_or_skip()
    assert accel.ACCELERATORS == modes.KERNELS_ALL == report.KERNELS_ACCELERATORS == ("flash_attn", "hub_layernorm", "flex_attention")
    for name, e in modes.MODE_TABLE.items():                                            # every mode runs and states its kernels
        assert e.runner and set(e.kernels) <= set(accel.ACCELERATORS) and modes.kernels_expected(name) == e.kernels, name
    assert modes.kernels_expected("off") == modes.kernels_expected("exact") == modes.KERNELS_ALL      # E1: the stock engages all three; exact tracks stock
    assert accel.expected("off", pins) == accel.expected("exact", pins) == _expected_words(pins)
    assert not hasattr(_names, "EXIT_KERNELS") and len({_names.EXIT_OK, _names.EXIT_FAIL, _names.EXIT_USAGE, _names.EXIT_NOT_ACTIVE}) == 4   # no kernel exit code: a fallback is named, never refused
    assert accel.ROUTES == ("stock", "default", "exact") == accel.STOCK_ROUTES + modes.KIT_MODES == report.KERNELS_ROUTES
    assert accel.route_name("off") == "stock" and accel.route_name("exact") == "exact"
    assert not any(hasattr(accel, n) for n in ("allow_of", "require_word", "REQUIRE_ON", "REQUIRE_ESCAPE", "violations", "check_allow")) and not hasattr(report, "REQUIRE_WORDS")   # no escape, no guard: the line names, nothing refuses


def test_line_grammar_round_trips_and_carries_nothing_the_activation_grep_forbids():
    pins = _stubs.pins_or_skip()
    words = _expected_words(pins)
    for mode, route, site in (("off", "stock", "stock"), ("off", "default", "stock"), ("exact", "exact", "kit_attn")):
        line = report.kernels_line(mode, route, words, site, True)
        m = report.RE_KERNELS.match(line)
        assert m, line
        assert m.group("runner") == ("stock" if mode == "off" else mode) and m.group("route") == route and m.group("site") == site and "require=" not in line
        assert line.startswith(report.kernels_prefix(mode) + " ") and report.kernels_prefix(mode) == ("[e1-opt stock]" if mode == "off" else f"[e1-opt {mode}]")
        assert {a: m.group(a) for a in accel.ACCELERATORS} == words and m.group("upstream_says") == "True"
        if mode != "off":                                                                # the check's activation grep forbids these on the exact rows
            assert "refused" not in line and not line.startswith(report.STOCK_PREFIX) and "NOT ACTIVE" not in line and "KitRefused" not in line
    assert report.kernels_prefix("off") == report.STOCK_PREFIX == "[e1-opt stock]" and report.kernels_prefix("exact") == report.KIT_PREFIX == "[e1-opt exact]"
    fb_line = report.kernels_line("off", "stock", dict(words, flash_attn="fallback:flex(USE_FLASH_ATTN=0)"), "stock", False)   # a fallback: its word on the line, nothing else
    assert report.RE_KERNELS.match(fb_line).group("upstream_says", "flash_attn") == ("False", "fallback:flex(USE_FLASH_ATTN=0)") and "REFUSED" not in fb_line
    none_line = report.kernels_line("off", "stock", {}, "unknown(E1.model.attention_not_loaded)", None)
    assert report.RE_KERNELS.match(none_line).group("flash_attn", "upstream_says") == ("absent(not_read)", "none")
    assert not any(hasattr(report, n) for n in ("kernels_refused_line", "kernels_refused_run_line", "RE_KERNELS_REFUSED", "RE_KERNELS_REFUSED_RUN", "KERNELS_REFUSED_TAIL"))
    assert not any(hasattr(report, n) for n in ("kernels_unread_line", "item_refused_reason", "peak_line", "RE_PEAK")) and not hasattr(accel, "peak_line")   # one assay per run; no allocator lines (memory is the caller's meter)


# ----------------------------------------------------------------------------------------------------------------- the reader
def test_reader_reads_the_bound_objects(monkeypatch, tmp_path):
    box = _box(monkeypatch, tmp_path)
    pins = box["pins"]
    accel.trap_upstream_logs()
    import E1.predictor  # noqa: F401 — the stub upstream: predictor -> modeling -> model.attention -> flash_attention / flex_attention
    r = accel.read()
    assert r["words"] == _expected_words(pins) and r["site"] == "stock" and r["upstream_says"] is True, r
    assert r["details"]["flash_attn"]["flash_attn_2_cuda_loaded"] is True and r["details"]["hub_layernorm"]["rev"] == pins.KERNEL["rev"]
    assert accel.fell_back(r["words"], accel.expected("off", pins)) == []
    # upstream's own env switch, read as upstream reads it (per call): the varlen-flex route
    monkeypatch.setenv("USE_FLASH_ATTN", "0")
    r0 = accel.read()
    assert r0["words"]["flash_attn"] == "fallback:flex(USE_FLASH_ATTN=0)" and r0["upstream_says"] is False
    assert accel.fell_back(r0["words"], accel.expected("off", pins)) == [("flash_attn", "fallback:flex(USE_FLASH_ATTN=0)", f"engaged:flash_attn@{pins.STACK['flash_attn']}")]
    monkeypatch.delenv("USE_FLASH_ATTN")
    # the guarded import failed (upstream binds None): the package is on the path, so the word says the import failed, not that it is absent
    FA, A, M, FX = (sys.modules[n] for n in ("E1.model.flash_attention", "E1.model.attention", "E1.modeling", "E1.model.flex_attention"))
    monkeypatch.setattr(FA, "flash_attn_varlen_func", None)
    assert accel.read()["words"]["flash_attn"] == "fallback:flex(flash_attn_import_failed)" and accel.read()["upstream_says"] is False
    monkeypatch.undo(); _stubs.patch_pins_for_stub(monkeypatch, pins, box["variant"], box["caches"])
    FA, A, M, FX = (sys.modules[n] for n in ("E1.model.flash_attention", "E1.model.attention", "E1.modeling", "E1.model.flex_attention"))
    # a varlen function that is not flash_attn's own
    monkeypatch.setattr(FA, "flash_attn_varlen_func", lambda *a, **k: None)
    assert accel.read()["words"]["flash_attn"].startswith("fallback:unknown(varlen_func_is_")
    monkeypatch.undo(); _stubs.patch_pins_for_stub(monkeypatch, pins, box["variant"], box["caches"])
    FA, A, M, FX = (sys.modules[n] for n in ("E1.model.flash_attention", "E1.model.attention", "E1.modeling", "E1.model.flex_attention"))
    # the class attribute the model calls through, rebound by something that is neither upstream nor the kit's attn adapter
    def _flash_attn(self, *a, **k):
        return None
    monkeypatch.setattr(A.Attention, "_flash_attn", _flash_attn)
    assert accel.read()["site"].startswith("other(") and "test_kernels_proof" in accel.read()["site"]
    monkeypatch.undo(); _stubs.patch_pins_for_stub(monkeypatch, pins, box["variant"], box["caches"])
    FA, A, M, FX = (sys.modules[n] for n in ("E1.model.flash_attention", "E1.model.attention", "E1.modeling", "E1.model.flex_attention"))
    # the hub kernel fell back: upstream's own warning is the why
    monkeypatch.setattr(M, "layer_norm", None)
    logging.getLogger("E1.modeling").warning("Failed to load triton layer norm kernel: no snapshot; Will be using PyTorch RMSNorm implementation instead.")
    w = accel.read()["words"]["hub_layernorm"]
    assert w == "fallback:torch_rmsnorm(Failed_to_load_triton_layer_norm_kernel:_no_snapshot)", w
    monkeypatch.undo(); _stubs.patch_pins_for_stub(monkeypatch, pins, box["variant"], box["caches"])
    FA, A, M, FX = (sys.modules[n] for n in ("E1.model.flash_attention", "E1.model.attention", "E1.modeling", "E1.model.flex_attention"))
    # a hub module whose kernel object is not a triton Autotuner
    import types
    fake = types.ModuleType("fake_ln"); fake.__file__ = M.layer_norm.__file__
    exec("_layer_norm_fwd_1pass_kernel = object()\ndef rms_norm_fn(x, w, b, **k): return x", fake.__dict__)
    monkeypatch.setattr(M, "layer_norm", fake)
    assert accel.read()["words"]["hub_layernorm"] == "fallback:unknown(_layer_norm_fwd_1pass_kernel_is_object)"
    monkeypatch.undo(); _stubs.patch_pins_for_stub(monkeypatch, pins, box["variant"], box["caches"])
    FA, A, M, FX = (sys.modules[n] for n in ("E1.model.flash_attention", "E1.model.attention", "E1.modeling", "E1.model.flex_attention"))
    # flex: eager (not compiled) and absent
    monkeypatch.setattr(FX, "flex_attention", lambda *a, **k: None)
    assert accel.read()["words"]["flex_attention"] == "fallback:eager(cuda_not_available_at_import)"
    monkeypatch.setattr(FX, "flex_attention", None)
    assert accel.read()["words"]["flex_attention"] == "absent(torch_flex_attention_import_failed)"


def test_reader_before_upstream_is_loaded_reads_absent():
    saved = {n: sys.modules.pop(n) for n in list(sys.modules) if n == "E1" or n.startswith("E1.")}
    try:
        r = accel.read()
        assert r["words"] == {"flash_attn": "absent(E1.model.attention_not_loaded)", "hub_layernorm": "absent(E1.modeling_not_loaded)",
                              "flex_attention": "absent(E1.model.flex_attention_not_loaded)"} and r["upstream_says"] is None and r["site"].startswith("unknown(")
    finally:
        sys.modules.update(saved)


# ----------------------------------------------------------------------------------------------------------------- REQUIRE
def test_proof_names_a_fallback_off_runs_it_a_kit_mode_refuses_by_name(monkeypatch, tmp_path):
    box = _box(monkeypatch, tmp_path)
    pins = box["pins"]
    import E1.predictor  # noqa: F401
    lines = []
    rec = accel.proof("off", "stock", pins, emit=lines.append)                                    # everything engaged: one line, nothing fell back
    assert len(lines) == 1 and report.RE_KERNELS.match(lines[0]).group("route") == "stock" and rec["fell_back"] == [] and "require=" not in lines[0]
    lines.clear()
    rec = accel.proof("exact", "exact", pins, emit=lines.append)                                  # a kit mode with everything engaged: one line, no refusal
    assert len(lines) == 1 and report.RE_KERNELS.match(lines[0]).group("runner", "route") == ("exact", "exact") and rec["fell_back"] == []
    monkeypatch.setenv("USE_FLASH_ATTN", "0")                                                    # upstream's silent fallback route under mode OFF: NAMED by its word, the stock goes on
    lines.clear()
    rec = accel.proof("off", "default", pins, emit=lines.append)
    assert len(lines) == 1 and report.RE_KERNELS.match(lines[0]).group("route", "flash_attn", "upstream_says") == ("default", "fallback:flex(USE_FLASH_ATTN=0)", "False"), lines
    assert rec["fell_back"] == [["flash_attn", "fallback:flex(USE_FLASH_ATTN=0)", f"engaged:flash_attn@{pins.STACK['flash_attn']}"]] and accel.result() is rec
    monkeypatch.setattr(sys.modules["E1.modeling"], "layer_norm", None)                           # under a KIT mode two fallbacks: both worded on the line, then the MODE refuses by name (exit 3)
    lines.clear()
    with pytest.raises(SystemExit) as ex:
        accel.proof("exact", "exact", pins, emit=lines.append)
    assert ex.value.code == _names.EXIT_NOT_ACTIVE == 3
    assert len(lines) == 2 and report.RE_KERNELS.match(lines[0]).group("runner", "route") == ("exact", "exact") and "hub_layernorm=fallback:torch_rmsnorm(" in lines[0]
    na = report.RE_NOT_ACTIVE.match(lines[1])
    assert na and na.group("reason").startswith("mode exact: accelerator(s) not engaged in this process — flash_attn=fallback:flex(USE_FLASH_ATTN=0)") and "hub_layernorm=fallback:torch_rmsnorm(" in lines[1] and lines[1].endswith("exit 3")
    assert [f[0] for f in accel.result()["fell_back"]] == ["flash_attn", "hub_layernorm"] and "REFUSED" not in lines[0]
    with pytest.raises(ValueError):
        accel.install("off", "exact", pins)                                                      # a route belongs to its mode
    with pytest.raises(ValueError):
        accel.install("exact", "stock", pins)
    with pytest.raises(ValueError):
        accel.install("exact", "stock", pins)                                                    # a kit mode's route is its own name
    with pytest.raises(ValueError):
        accel.install("off", "exact", pins)


def test_install_imports_nothing_and_fires_once_at_the_first_construction(monkeypatch, tmp_path):
    box = _box(monkeypatch, tmp_path)
    pins = box["pins"]
    assert "E1.predictor" not in sys.modules
    lines = []
    accel.install("exact", "exact", pins, emit=lines.append)
    assert "E1.predictor" not in sys.modules and "torch" not in sys.modules and accel.armed() and lines == []   # armed through the import finder: nothing of upstream's loaded
    import E1.scorer as S                                                                        # upstream's own import order loads the predictor; the finder wraps it then
    assert getattr(sys.modules["E1.predictor"].E1Predictor.__init__, "_e1_opt_kernels", False) and lines == []
    S.E1Scorer(object(), method="masked_marginal")                                              # the scorer builds a predictor: the proof fires after the constructor returns
    assert len(lines) == 1 and report.RE_KERNELS.match(lines[0]).group("runner", "route") == ("exact", "exact"), lines
    S.E1Scorer(object(), method="masked_marginal")
    assert len(lines) == 1 and accel.result()["fell_back"] == [] and len(sys.modules["E1.predictor"].INSTANCES) == 2   # once per process; the constructor itself ran both times
    accel.install("exact", "exact", pins, emit=lines.append)                                     # idempotent
    assert sys.modules["E1.predictor"].E1Predictor.__init__.__wrapped__ is not None


def test_a_kit_modes_first_construction_refuses_by_name_when_an_accelerator_fell_back(monkeypatch, tmp_path):
    """The armed proof under a KIT mode: at the first scorer construction of a process where an accelerator the mode's levers run on fell back
    (upstream's own USE_FLASH_ATTN=0 here), the KERNELS line words it and the MODE refuses by name — the NOT ACTIVE line and SystemExit(3)
    before any forward; a mode is all of its levers, never a subset under its name. Under mode off the same construction goes on (named)."""
    box = _box(monkeypatch, tmp_path)
    pins = box["pins"]
    monkeypatch.setenv("USE_FLASH_ATTN", "0")
    lines = []
    accel.install("exact", "exact", pins, emit=lines.append)
    import E1.scorer as S
    with pytest.raises(SystemExit) as ex:
        S.E1Scorer(object(), method="masked_marginal")
    assert ex.value.code == _names.EXIT_NOT_ACTIVE
    assert len(lines) == 2 and report.RE_KERNELS.match(lines[0]).group("runner", "flash_attn") == ("exact", "fallback:flex(USE_FLASH_ATTN=0)")
    assert report.RE_NOT_ACTIVE.match(lines[1]).group("reason").startswith("mode exact: accelerator(s) not engaged in this process — flash_attn=fallback:flex(USE_FLASH_ATTN=0)")
    monkeypatch.setattr(accel, "_HOOK", dict(accel._HOOK, fired=False, result=None, config=None, armed=False)) if hasattr(accel, "_HOOK") else None
    lines.clear()
    rec = accel.proof("off", "stock", pins, emit=lines.append)                                    # mode off: the stock's own route, named, no exit
    assert len(lines) == 1 and rec["fell_back"][0][0] == "flash_attn"

# ----------------------------------------------------------------------------------------------------------------- live, through score
def test_stock_child_under_upstreams_fallback_route_runs_and_the_fallback_is_named(monkeypatch, tmp_path, capsys):
    """Upstream's own `USE_FLASH_ATTN=0` reaches the stock child (upstream's variable is not stripped): the varlen-flex route runs — exactly what
    stock does there — every model process words it on its KERNELS line and the run's EXIT line names the accelerator that fell back. Nothing
    is refused, nothing exits early: both items score."""
    box = _box(monkeypatch, tmp_path)
    items = _stubs.write_items(str(tmp_path / "in"))
    out = str(tmp_path / "out")
    monkeypatch.setenv("USE_FLASH_ATTN", "0")
    rc, lines, err = _run(capsys, ["score", "--variant", box["variant"], "--mode", "off", *_one(items, "item_a", out)])
    assert rc == 0, (lines, err)
    assert report.RE_OFF.match(lines[0]), lines[0]                                               # the off line: the stock route
    k = [report.RE_KERNELS.match(s) for s in lines if report.RE_KERNELS.match(s)]
    assert len(k) == 1 and all(m.group("runner", "route", "flash_attn", "upstream_says") == ("stock", "stock", "fallback:flex(USE_FLASH_ATTN=0)", "False") for m in k)   # one per item process
    assert all(m.group("hub_layernorm").startswith("engaged:") and m.group("flex_attention").startswith("engaged:") and "require=" not in m.string for m in k)
    assert not any("REFUSED" in s for s in lines)
    assert outputs.listing(outputs.scores_path(out, "item_a"))["present"]
    x = [report.RE_EXIT.match(s) for s in lines if report.RE_EXIT.match(s)][0]
    assert x.group("rc", "complete", "kernels_fallback") == ("0", "1", "flash_attn")               # the fallen-back accelerator named on the EXIT line
    monkeypatch.delenv("USE_FLASH_ATTN")                                                         # nothing fell back: nothing named
    rc, lines, err = _run(capsys, ["score", "--variant", box["variant"], "--mode", "off", *_one(items, "item_a", str(tmp_path / "out3"))])
    assert rc == 0, (lines, err)
    x = [report.RE_EXIT.match(s) for s in lines if report.RE_EXIT.match(s)][0]
    assert x.group("kernels_fallback") == "none" and all(report.RE_KERNELS.match(s).group("flash_attn").startswith("engaged:") for s in lines if report.RE_KERNELS.match(s))


def test_hub_kernel_fallback_is_named_from_upstreams_own_warning(monkeypatch, tmp_path, capsys):
    box = _box(monkeypatch, tmp_path)
    items = _stubs.write_items(str(tmp_path / "in"), names=("h",))
    out = str(tmp_path / "out")
    monkeypatch.setenv("STUB_HUB_KERNEL_FAILS", "1")                                             # the stub upstream's get_kernel raises: upstream logs its warning and binds None
    rc, lines, err = _run(capsys, ["score", "--variant", box["variant"], "--mode", "off", *_one(items, "h", out)])
    assert rc == 0, (lines, err)                                                                 # torch's rms_norm ran — upstream's own route there — named, not refused
    k = next(report.RE_KERNELS.match(s) for s in lines if report.RE_KERNELS.match(s))
    assert k.group("hub_layernorm").startswith("fallback:torch_rmsnorm(Failed_to_load_triton_layer_norm_kernel:_stub"), k.group("hub_layernorm")
    assert k.group("flash_attn").startswith("engaged:") and k.group("flex_attention").startswith("engaged:")
    x = [report.RE_EXIT.match(s) for s in lines if report.RE_EXIT.match(s)][0]
    assert x.group("rc", "complete", "kernels_fallback") == ("0", "1", "hub_layernorm") and outputs.listing(outputs.scores_path(out, "h"))["present"]


NO_KERNELS_KIT_STUB = '''
import os, subprocess, sys
args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
print("[e1-opt] KIT vTEST mode=eager size=300m card=h100 gpu=\\"x\\" pin=W8 levers=1/1", flush=True)
rc = subprocess.call([sys.executable, "-m", "E1.tools.score", *args])
sys.exit(rc)
'''


def test_an_exact_child_that_printed_no_kernels_line_is_the_runs_refusal(monkeypatch, tmp_path, capsys):
    box = _box(monkeypatch, tmp_path)
    stub = tmp_path / "kit_stub.py"
    stub.write_text(NO_KERNELS_KIT_STUB)
    monkeypatch.setattr(kit_score, "command", lambda python, tool_args, **kw: [python, str(stub), "--", *tool_args])   # a child outside the runner: no proof
    items = _stubs.write_items(str(tmp_path / "in"))
    out = str(tmp_path / "out")
    one = ["--parent-path", os.path.join(items, "item_a", "parent.fasta"), "--mutants-path", os.path.join(items, "item_a", "mutants.fasta"), "--output-path", os.path.join(out, "item_a", "scores.csv")]
    rc, lines, err = _run(capsys, ["score", "--variant", box["variant"], *one, "--det", "0"])
    assert rc == _names.EXIT_NOT_ACTIVE, (lines, err)                                            # the item completed but nothing PROVED its accelerators: the run refuses by name
    na = [s for s in lines if report.RE_NOT_ACTIVE.match(s)]
    assert na == ["[e1-opt] NOT ACTIVE: no KERNELS line from the completed process (the kit's hook did not fire: the tool built no model)"], lines
    assert sum(1 for s in lines if report.RE_SCORE.match(s)) == 1
    x = [report.RE_EXIT.match(s) for s in lines if report.RE_EXIT.match(s)][0]
    assert x.group("rc", "complete", "kernels_fallback") == ("3", "1", "none") and report.RE_EXIT.match(lines[-1])


def test_the_stack_imports_check_judges_flash_attn_and_kernels_by_import_and_names(monkeypatch, tmp_path, capsys):
    box = _box(monkeypatch, tmp_path)
    got = stack.stack_imports(box["pins"])                                                       # the stub packages import at the pins: nothing to name
    assert got["flash_attn_module"] == box["pins"].STACK["flash_attn"] and got["kernels_module"] == box["pins"].STACK["kernels"] and got["notes"] == []
    rc, lines, err = _run(capsys, ["check", "--variant", box["variant"], "--mode", "off"])
    assert rc == 0 and " pins=ok " in lines[0] and "would_refuse=none" in lines[0] and "untested=" not in lines[0], lines
    # a flash_attn distribution at the pinned version whose MODULE lacks the symbol the model imports: upstream's silent fallback route — NAMED, never refused
    open(os.path.join(box["site"], "flash_attn", "__init__.py"), "w").write(f"__version__ = {box['pins'].STACK['flash_attn']!r}\nflash_attn_func = None\n")
    for m in ("flash_attn", "flash_attn_2_cuda"):
        monkeypatch.delitem(sys.modules, m, raising=False)
    got = stack.stack_imports(box["pins"])
    assert got["flash_attn_module"] is None and len(got["notes"]) == 1 and got["notes"][0].startswith("flash_attn does not import (ImportError")
    for m in ("flash_attn", "flash_attn_2_cuda"):
        monkeypatch.delitem(sys.modules, m, raising=False)
    rc, lines, err = _run(capsys, ["check", "--variant", box["variant"], "--mode", "off"])
    m = report.RE_DRY_RUN.match(lines[0])
    assert rc == 0 and " pins=drift " in lines[0] and lines[0].endswith("would_refuse=none") and ' notes="flash_attn does not import (ImportError' in lines[0], lines
    # kernels without get_kernel: named too
    open(os.path.join(box["site"], "flash_attn", "__init__.py"), "w").write(_stubs.STUB_FLASH_ATTN.format(version=box["pins"].STACK["flash_attn"]))
    open(os.path.join(box["site"], "kernels", "__init__.py"), "w").write("__version__ = '0'\n")
    for m in ("flash_attn", "flash_attn_2_cuda", "kernels"):
        monkeypatch.delitem(sys.modules, m, raising=False)
    got = stack.stack_imports(box["pins"])
    assert got["kernels_module"] is None and any("kernels does not import or has no get_kernel" in n for n in got["notes"])


