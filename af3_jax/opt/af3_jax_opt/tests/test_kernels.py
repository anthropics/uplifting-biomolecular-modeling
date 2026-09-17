"""The KERNELS line and its REQUIRE guard (kernels.py + inprocess/kernels_probe.py): one line per pass on every route, the expected
reading derived from the mode table alone, and a pass whose model process resolves an accelerator implementation other than the route's —
absent, fallen back, a call site bound by other than the route's lever set, or no reading at all — refused by name with exit 5."""
import json
import os
import subprocess
import sys
import textwrap

import pytest

from af3_jax_opt import cli, kernels, modes, settings, stack, stock_pred
from af3_jax_opt.inprocess import kernels_probe

from .conftest import assert_no_markers, chain, ligand, render
import io
from af3_jax_opt import inputs, modes
from af3_jax_opt.inprocess import templates as T


def _run(argv, capsys):
    rc = cli.main(argv)
    out = capsys.readouterr()
    return rc, out.err, out.out


def _kernels_line(err):
    """The wrapper's ONE `KERNELS route=...` summary line (the model process's own KERNELS-PROBE / KERNELS REFUSED lines are echoed above it)."""
    hits = [ln for ln in err.splitlines() if ln.startswith("[af3-jax-opt] KERNELS route=")]
    assert len(hits) == 1, err
    return hits[0]


def _pred(box, capsys, mode="off", extra=(), name="a"):
    inp = box.input_json(name, seeds=(1,))
    out = os.path.join(box.root, f"out_{mode}_{name}")
    if mode != "off":
        box.warm_cache(mode=mode)
    rc, err, _ = _run(["pred", "--variant", "p2", "--mode", mode, "--json_path", inp, "--output_dir", out, *extra], capsys)
    return rc, err, out


# ---------------------------------------------------------------- the line, every route
@pytest.mark.parametrize("mode,route,flash,pair,trimul", [
    ("off", "stock", "engaged:triton@tokamax0.0.12", "tokamax", "stock"),
    ("exact", "exact", "engaged:triton@tokamax0.0.12:pinned(64x64x4x3)", "attncfg", "glut"),
    ("fast", "fast", "engaged:triton@tokamax0.0.12", "fpf", "fpf"),
])
def test_one_kernels_line_per_pass_on_every_route(box, capsys, mode, route, flash, pair, trimul):
    rc, err, out = _pred(box, capsys, mode)
    assert rc == 0, err
    ln = _kernels_line(err)
    m = kernels.KERNELS_RX.search(ln)
    assert m, ln
    assert (m.group("route"), m.group("mode")) == (route, mode) and "preset=" not in ln, ln
    assert m.group("flash_impl") == flash and m.group("pair_attn") == pair and m.group("trimul") == trimul, ln
    assert m.group("glu_impl") == "engaged:triton@tokamax0.0.12" and m.group("xla_flags") == "upstream", ln
    assert m.group("requested") == "triton:fork_default" and m.group("probe") == "ok" and m.group("verdict") == "ok", ln
    assert_no_markers(ln)                                                   # a conforming KERNELS line carries no log-scan failure marker
    assert "DONE status=ok" in err and m.group("route") == route
    acc = kernels.account(mode=mode, n_gpu=1, levers=modes.resolve(mode)["levers"], argv=box.stub_record()["argv"], env=box.stub_record()["env"], lines=err.splitlines(), rc=0, caller_environ={})
    assert acc["reasons"] == [] and acc["expected"] == kernels.expected(mode, modes.resolve(mode)["levers"], "triton")   # the account the KERNELS line was printed from


def test_the_stock_flags_ride_last_on_every_route(box, capsys):
    """The stock command line's own knobs pass through verbatim, after the composed flags, on off and on a kit mode alike; the package
    writes no model-shape flag of its own; a caller's --cache_dir under a kit mode is the class the mode gates."""
    stated = ("--num_recycles=3", "--num_diffusion_samples=1", "--buckets=256,512", "--noresolve_msa_overlaps")
    rc, err, out = _pred(box, capsys, "off", stated)
    assert rc == 0, err
    rec = box.stub_record()
    assert rec["script"] == "run_alphafold.py" and rec["launcher"] is None and tuple(rec["flags"][-4:]) == stated
    assert [f for f in rec["flags"][:-4] if f.startswith(("--num_recycles", "--num_diffusion_samples", "--flash_attention_implementation", "--resolve_msa"))] == []
    m = kernels.KERNELS_RX.search(_kernels_line(err))
    assert m and m.group("route") == "stock" and m.group("verdict") == "ok" and m.group("requested") == "triton:fork_default"
    assert "[af3-jax-opt] BUCKETS buckets=256,512 source=caller occurrences=2" in err and "predictions=1/1" in err
    stack._REPORT = None; stack._LAUNCHED.clear()                          # one (mode, variant) per process: reset as a fresh interpreter
    rc, err, out = _pred(box, capsys, "exact", stated, name="b")
    assert rc == 0, err
    rec = box.stub_record()
    assert rec["script"] == "run_alphafold_fast.py" and tuple(rec["flags"][-4:]) == stated and f"--cache_dir={box.cache_dir('exact')}" in rec["flags"]
    stack._REPORT = None; stack._LAUNCHED.clear()
    cold = os.path.join(box.root, "cold_class")
    rc, err, _ = _run(["pred", "--variant", "p2", "--mode", "exact", "--json_path", box.input_json("c", seeds=(1,)), "--output_dir", os.path.join(box.root, "out_x"), f"--cache_dir={cold}"], capsys)
    assert rc == 0 and "partial=cold_cache" in err and f"cache={cold}" in err, err                                                     # the caller's --cache_dir is the class the mode reads: a cold one is named (partial=cold_cache) by the same rule, the run proceeds


def test_a_stated_flash_impl_is_the_request(box, capsys):
    """--flash_attention_implementation=cudnn stated by the caller rides last; the request and the reading say cudnn, source `line`."""
    rc, err, out = _pred(box, capsys, "off", ("--flash_attention_implementation=cudnn",))
    assert rc == 0, err
    rec = box.stub_record()
    assert stock_pred.effective_flash_impl(["py"] + rec["argv"])["impl"] == "cudnn"
    m = kernels.KERNELS_RX.search(_kernels_line(err))
    assert m.group("flash_impl") == "engaged:cudnn@tokamax0.0.12" and m.group("requested") == "cudnn:line", m.group(0)


# ---------------------------------------------------------------- the REQUIRE guard: refused by name, exit 5
@pytest.mark.parametrize("stub_env,mode,flash_word,reason_key", [
    ({"STUB_KERNELS_DOWNGRADE": "1"}, "off", "fallback:cpu-auto-downgrade", "flag: expected triton got xla"),          # the fork's CPU auto-downgrade (run_alphafold.py:1126-1135)
    ({"STUB_KERNELS_ATTN_CLASS": "absent"}, "off", "absent:triton", "flash_impl absent"),                              # the image lacks the implementation
    ({"STUB_KERNELS_SUPPORTED": "0"}, "off", "fallback:triton-unsupported", "attn_supported: expected 1 got 0"),      # the bound class does not support the device
    ({"STUB_KERNELS_BACKEND": "cpu"}, "off", "engaged:triton@tokamax0.0.12", "backend: expected gpu got cpu"),
    ({"STUB_KERNELS_TRIMUL": "stock"}, "exact", "engaged:triton@tokamax0.0.12:pinned", "trimul_site: expected glut got stock"),        # refused at the first call: no census, so no tile tuple on the word   # a call site not owned by the route's lever
])
def test_a_reading_that_is_not_the_routes_is_refused_in_process(box, capsys, monkeypatch, stub_env, mode, flash_word, reason_key):
    for k, v in stub_env.items():
        monkeypatch.setenv(k, v)
    rc, err, out = _pred(box, capsys, mode)
    assert rc == cli.EXIT_KERNELS == 5, err
    ln = _kernels_line(err)
    m = kernels.KERNELS_RX.search(ln)
    assert m and m.group("flash_impl") == flash_word and m.group("probe") == "refused" and m.group("verdict").startswith("REFUSED:"), ln
    assert "[af3-jax-opt] KERNELS REFUSED in pid" in open(os.path.join(out, cli.LOG_NAME)).read()      # the process refused itself, before any timed item
    assert "Running model inference with seed" not in open(os.path.join(out, cli.LOG_NAME)).read()
    done = [x for x in err.splitlines() if x.startswith("[af3-jax-opt] DONE ")][0]
    assert "status=FAILED" in done and f"reason=kernels_refused:{kernels.route_name(mode)}: " in done and reason_key in done, done


def test_tokamax_glu_fallback_event_is_refused(box, capsys, monkeypatch):
    """tokamax's own logged GLU fallback (`Failed to run implementation`, gated_linear_unit/api.py:116) after a conforming first-call reading:
    the pass ran, the transcript names the event, the KERNELS line words glu_impl=fallback:<next impl>:<n> and the pass is refused (exit 5)."""
    monkeypatch.setenv("STUB_KERNELS_GLU_FALLBACK", "1")
    rc, err, out = _pred(box, capsys, "off")
    assert rc == 5, err
    m = kernels.KERNELS_RX.search(_kernels_line(err))
    assert m.group("glu_impl") == "fallback:msc:1" and m.group("probe") == "ok" and m.group("verdict") == "REFUSED:glu_fallback", m.group(0)


def test_no_probe_line_is_refused(box, capsys, monkeypatch):
    """No KERNELS-PROBE line (the hook did not load / no kernel call): the reading is missing — refused, never assumed."""
    monkeypatch.setenv("STUB_KERNELS", "none")
    rc, err, out = _pred(box, capsys, "off")
    assert rc == 5, err
    m = kernels.KERNELS_RX.search(_kernels_line(err))
    assert m.group("probe") == "missing" and m.group("flash_impl") == "unread:no_probe_line" and m.group("verdict").startswith("REFUSED:"), m.group(0)


# ---------------------------------------------------------------- the expectation is the mode table's
def test_expected_reading_follows_the_mode_table():
    for mode in modes.MODES:
        levers = [] if mode == "off" else modes.resolve(mode)["levers"]
        exp = kernels.expected(mode, levers, "triton")
        assert exp["flag"] == "triton" and exp["backend"] == "gpu" and exp["glu_head"] == kernels.GLU_HEAD == "triton"
        assert exp["dpa_site"] == ("attncfg" if "ATTNCFG" in levers and "FPF_TRIATT" not in levers else "tokamax")
        assert exp["triatt_site"] == ("fpf" if "FPF_TRIATT" in levers else "stock")
        assert exp["trimul_site"] == ("fpf" if "FPF_TRIMUL" in levers else ("glut" if "GLUT" in levers else "stock"))
        assert kernels_probe.parse_expect(kernels.expect_env(exp)[kernels.ENV_EXPECT]) == exp      # the env encoding round-trips into the probe's parser
    assert kernels.expected("off", [], "cudnn")["flag"] == "cudnn"                                # the request drives the expected implementation
    with pytest.raises(ValueError):
        kernels.expected("off", ["GLUT"], "triton")                                                   # the stock route carries no lever
    assert [kernels.route_name(*x) for x in (("off", 1), ("exact", 1), ("big", 2))] == ["stock", "exact", "big_x2"]
    assert kernels.EXIT_KERNELS == kernels_probe.EXIT_REFUSED == 5 and kernels.FLASH_IMPLS == ("triton", "cudnn", "xla")


def test_verdict_is_one_function():
    """kernels_probe.verdict — the comparison the model process runs and the wrapper re-runs on the probe line."""
    exp = kernels.expected("exact", modes.resolve("exact")["levers"], "triton")
    ok = {"flag": "triton", "dpa_site": "attncfg", "triatt_site": "stock", "trimul_site": "glut+w:TriangleMultiplicationChunked", "glu_head": "triton", "backend": "gpu",
          "attn_supported": "1", "attn_class": "PallasTritonFlashAttention", "attn_impls": "triton,xla"}
    assert kernels_probe.verdict(ok, exp) == []                                                     # a wrapped site: the base owns the kernel
    assert kernels_probe.verdict({**ok, "flag": "xla"}, exp) == ["flag: expected triton got xla"]
    assert kernels_probe.verdict({**ok, "attn_class": "absent"}, exp)[0].startswith("flash_impl absent")
    assert kernels_probe.verdict({**ok, "dpa_site": "tokamax"}, exp) == ["dpa_site: expected attncfg got tokamax"]
    assert kernels_probe.verdict({}, exp)                                                             # nothing read: every key refused, never vacuously ok
    assert kernels_probe.impl_name(None) == "None" and kernels_probe.impl_name("triton") == "triton"


def test_arm_puts_the_probe_beside_the_peak_instrument(tmp_path):
    from opt_core.mem import peak
    hook = peak.write_hook(str(tmp_path / "hook"))
    env = {"PYTHONPATH": hook["dir"]}
    exp = kernels.expected("off", [], "triton")
    h = kernels.arm(env, hook["dir"], exp)
    assert open(h["probe_py"], "rb").read() == open(kernels.PROBE_FILE, "rb").read()                 # byte for byte
    site = open(os.path.join(hook["dir"], "sitecustomize.py")).read()
    assert site.startswith(peak.SITECUSTOMIZE) and "kernels_probe.boot('sitecustomize')" in site       # ONE sitecustomize boots both instruments, the peak loader's text verbatim first
    assert env[kernels.ENV_EXPECT] == kernels.expect_env(exp)[kernels.ENV_EXPECT]


def test_the_probe_hooks_tokamax_import_and_refuses_at_the_first_call(tmp_path):
    """The real in-process route, without jax: a stand-in `tokamax` package on the path, the probe booted as the hook boots it, the first
    gated_linear_unit call prints KERNELS-PROBE and — the expectation contradicted (no gpu backend here) — exits 5 before returning."""
    pkg = tmp_path / "site" / "tokamax"; pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("__version__ = '0.0.12'\ndef dot_product_attention(q, k, v, *, implementation=None):\n    return 'dpa'\n"
                                     "def gated_linear_unit(x, weights, *, activation=None, implementation=None):\n    return 'glu'\n")
    hook = tmp_path / "hook"; hook.mkdir()
    import shutil
    shutil.copyfile(kernels.PROBE_FILE, hook / "kernels_probe.py")
    prog = textwrap.dedent(f"""
        import sys
        sys.path[:0] = [{str(hook)!r}, {str(tmp_path / 'site')!r}]
        import kernels_probe
        assert kernels_probe.boot('test') is True
        import tokamax
        assert getattr(tokamax.gated_linear_unit, '_kernels_probe_wrapped', False), 'post-import hook did not arm'
        print('calling', flush=True)
        r = tokamax.gated_linear_unit(1, 2)
        print('returned', r, flush=True)
    """)
    exp = kernels.expect_env(kernels.expected("off", [], "triton"))
    p = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True, env={**os.environ, **exp}, timeout=60)
    assert p.returncode == 5, (p.returncode, p.stdout, p.stderr)
    assert "[af3-jax-opt] KERNELS-PROBE " in p.stderr and "first=glu" in p.stderr and "verdict=REFUSED:" in p.stderr and "returned" not in p.stdout
    sc = kernels.scan(p.stderr.splitlines())
    assert sc["probe"]["tokamax"] == "0.0.12" and sc["probe"]["flag"].startswith("unread") and sc["refused_inprocess"]
    # without an expectation the probe is inert: no hook, the call returns
    p2 = subprocess.run([sys.executable, "-c", prog.replace("is True", "is False").replace("assert getattr", "assert not getattr")],
                        capture_output=True, text=True, env={k: v for k, v in os.environ.items() if k != kernels.ENV_EXPECT}, timeout=60)
    assert p2.returncode == 0 and "returned glu" in p2.stdout, (p2.returncode, p2.stdout, p2.stderr)




def test_probe_site_words_are_the_mode_tables_words():
    """The words the in-process probe gives the lever classes (kernels_probe.SITE_CLASSES, _site_word) are exactly the words kernels.expected
    derives from the mode table for every mode of the kit — stand-in classes carrying the add-ons' real module and class names (a class made inside a
    function, as the add-ons make them, keeps its name last in __qualname__)."""
    def cls(module, name, wrapped=None):
        def make():
            C = type(name, (), {})
            C.__module__ = module; C.__qualname__ = f"make.<locals>.{name}"
            if wrapped is not None:
                C.__wrapped__ = wrapped
            return C
        return make()
    stock_ga = type("GridSelfAttention", (), {"__module__": kernels_probe.STOCK_MODULES}); stock_ga.__qualname__ = "GridSelfAttention"
    stock_tm = type("TriangleMultiplication", (), {"__module__": kernels_probe.STOCK_MODULES}); stock_tm.__qualname__ = "TriangleMultiplication"
    fpf_ga, fpf_tm = cls("af3_flashpairformer.patch", "FlashGridSelfAttention"), cls("patch", "FlashTriangleMultiplication")
    glut_tm = cls("af3_pallas_levers", "TriangleMultiplication")
    chunk_tm = cls("af3_jax_opt.big_levers", "ChunkedTriangleMultiplication", wrapped=glut_tm)          # a memory lever's wrapper subclass over GLUT: base word + the wrapper named
    w = kernels_probe._site_word
    assert (w(stock_ga, "GridSelfAttention"), w(stock_tm, "TriangleMultiplication")) == ("stock", "stock")
    assert (w(fpf_ga, "GridSelfAttention"), w(fpf_tm, "TriangleMultiplication"), w(glut_tm, "TriangleMultiplication")) == ("fpf", "fpf", "glut")
    assert w(chunk_tm, "TriangleMultiplication") == "glut+w:ChunkedTriangleMultiplication"
    bound = {"exact": (stock_ga, glut_tm), "fast": (fpf_ga, fpf_tm), "big": (stock_ga, chunk_tm)}       # who binds the two pair sites under each mode's lever set (modes.KIT_MODES)
    for mode, (ga, tm) in bound.items():
        exp = kernels.expected(modes.effective_mode(mode), modes.resolve(mode)["levers"], "triton")
        reading = {"triatt_site": w(ga, "GridSelfAttention"), "trimul_site": w(tm, "TriangleMultiplication")}
        assert kernels_probe.verdict(reading, {k: exp[k] for k in ("triatt_site", "trimul_site")}) == [], (mode, exp, reading)


def test_refusal_ends_the_process_through_systemexit_not_a_hard_exit():
    """A refusal raises KernelsRefused (SystemExit code 5) after terminating the process's own worker children — never os._exit while workers
    hold the transcript pipe (the wrapper would wait on it forever)."""
    import multiprocessing
    started = multiprocessing.get_context("fork").Process(target=__import__("time").sleep, args=(30,)) if hasattr(os, "fork") else None
    if started: started.start()
    try:
        with pytest.raises(SystemExit) as ei:
            kernels_probe.refuse_exit()
        assert ei.value.code == kernels_probe.EXIT_REFUSED and isinstance(ei.value, kernels_probe.KernelsRefused)
        if started:
            started.join(5); assert not started.is_alive()                # the child was terminated before the raise
    finally:
        if started and started.is_alive(): started.kill()
        for t in __import__("threading").enumerate():                     # disarm the last-resort timer this test armed
            if isinstance(t, __import__("threading").Timer): t.cancel()


# --- merged from test_templates.py (file consolidation, tests unchanged) ---

P = "[af3-jax-opt]"


def test_untemplated_input_is_not_guarded_and_says_so():
    s = T.census([{"chain_id": "A", "declared": 0, "aligned": [], "skip": None}], {}, 4)
    assert s["verdict"] == "untemplated" and s["declared"] == 0 and s["lines"] == [f"{P} TEMPLATES census declared=0 kept=0 real=0 chains_templated=0 chains_all_dummy=0 verdict=untemplated"]
    T.emit(s, io.StringIO())


def test_real_and_partial_slots_are_counted_per_chain():
    chains = [{"chain_id": "A", "declared": 3, "aligned": [120, 118, 80], "skip": None}, {"chain_id": "B", "declared": 0, "aligned": [], "skip": None}]
    s = T.census(chains, {"A": [120, 0, 40, 0]}, 4)                       # slot 1 resolves nothing: dropped by name; slot 2 partial: real
    assert s["verdict"] == "ok" and s["real"] == 2 and s["declared"] == 3 and s["chains"][0]["dropped"] == [(1, "unresolved_aligned_residues=118/118")]
    assert s["lines"][0] == f"{P} TEMPLATES chain=A declared=3 kept=3 real=2/4 dropped=1:unresolved_aligned_residues=118/118"
    assert s["lines"][1] == f"{P} TEMPLATES census declared=3 kept=3 real=2 chains_templated=1 chains_all_dummy=0 verdict=ok"


def test_form_word_says_where_the_template_pair_features_live():
    """No `form=` word on one device (the line is the same as ever); `form=row_born` once the row-sharded pair stack is installed (`--n_gpu P`: big_launch
    sets it after the recipe's install — each device forms the template distogram / unit-vector / mask rows of its block only). The wrapper's parser reads it."""
    chains = [{"chain_id": "A", "declared": 4, "aligned": [306] * 4, "skip": None}, {"chain_id": "B", "declared": 4, "aligned": [306] * 4, "skip": None}]
    s = T.census(chains, {"A": [306] * 4, "B": [300] * 4}, 4, form="row_born")
    assert s["form"] == "row_born" and s["lines"][-1] == f"{P} TEMPLATES census declared=8 kept=8 real=8 chains_templated=2 chains_all_dummy=0 form=row_born verdict=ok"
    assert T.census(chains, {"A": [306] * 4, "B": [300] * 4}, 4)["lines"][-1].endswith(" chains_all_dummy=0 verdict=ok")          # one device: no form= word
    assert T._STATE["form"] is None and T.FORMS == (None, "row_born")
    T.set_form("row_born")
    try:
        assert T._STATE["form"] == "row_born"
        with pytest.raises(ValueError):
            T.set_form("sharded")
    finally:
        T.set_form(None)
    guard = f"{P} TEMPLATES guard=installed site=alphafold3.model.features.Templates.compute_features"
    rep = modes.templates_census([guard] + s["lines"], 8, "big")
    assert rep["verdict"] == "ok" and rep["censuses"][0]["form"] == "row_born" and rep["real"] == 8
    assert modes.templates_census([guard, f"{P} TEMPLATES census declared=2 kept=2 real=2 chains_templated=1 chains_all_dummy=0 verdict=ok"], 2, "fast")["censuses"][0]["form"] is None


def test_over_max_and_skipped_chains_are_named():
    chains = [{"chain_id": "A", "declared": 6, "aligned": [10] * 6, "skip": None}, {"chain_id": "L", "declared": 2, "aligned": [3, 3], "skip": "not_protein"}]
    s = T.census(chains, {"A": [10, 10, 10, 10], "L": [0, 0, 0, 0]}, 4)
    a, l = s["chains"]
    assert a["kept"] == 4 and a["real"] == 4 and a["dropped"] == [(4, "over_max_templates"), (5, "over_max_templates")]
    assert l["kept"] == 0 and l["real"] == 0 and l["all_dummy"] and l["dropped"] == [(0, "chain_skipped:not_protein"), (1, "chain_skipped:not_protein")]
    assert s["verdict"] == "all_dummy" and s["chains_all_dummy"] == 1 and "refusal" not in s                  # counted, never refused: the run proceeds as the stock script's does


def test_all_dummy_is_a_census_word_not_a_refusal():
    s = T.census([{"chain_id": "A", "declared": 4, "aligned": [200] * 4, "skip": None}], {"A": [0, 0, 0, 0]}, 4)   # the 8IO9 case: 4/4 slots zero-mask
    buf = io.StringIO()
    T.emit(s, buf)                                                            # prints the census and returns: no SystemExit, the model process continues
    out = buf.getvalue().splitlines()
    assert len(out) == 2 and not hasattr(T, "TemplatesRefused") and not hasattr(T, "EXIT_TEMPLATES_REFUSED")
    assert out[0] == f"{P} TEMPLATES chain=A declared=4 kept=4 real=0/4 dropped=0:unresolved_aligned_residues=200/200,1:unresolved_aligned_residues=200/200,2:unresolved_aligned_residues=200/200,3:unresolved_aligned_residues=200/200"
    assert out[1].endswith("chains_all_dummy=1 verdict=all_dummy")
    assert all(modes.is_evidence_line(ln) for ln in out)                 # the wrapper collects every one of them


def test_skip_reasons_follow_the_fork():
    assert T._skip_reason("polypeptide(L)", 100, True, "polypeptide(L)") is None
    assert T._skip_reason("polyribonucleotide", 100, True, "polypeptide(L)") == "not_protein"
    assert T._skip_reason("polypeptide(L)", 4, True, "polypeptide(L)") == "le_4_tokens" and T._skip_reason("polypeptide(L)", 5, False, "polypeptide(L)") == "not_in_crop"


def test_wrapper_account_of_the_templates_lines():
    guard = f"{P} TEMPLATES guard=installed site=alphafold3.model.features.Templates.compute_features"
    ok = [guard, f"{P} TEMPLATES chain=A declared=2 kept=2 real=2/4 dropped=none", f"{P} TEMPLATES census declared=2 kept=2 real=2 chains_templated=1 chains_all_dummy=0 verdict=ok"]
    t = modes.templates_census(ok, 2, "exact")
    assert t["verdict"] == "ok" and t["token"] == "2/2" and t["guard"] == "installed" and t["chains"][0]["chain"] == "A" and "refused" not in t
    assert modes.templates_census([guard, f"{P} TEMPLATES census declared=0 kept=0 real=0 chains_templated=0 chains_all_dummy=0 verdict=untemplated"], 0, "fast")["token"] == "none"
    dummy = ok[:1] + [f"{P} TEMPLATES chain=A declared=4 kept=4 real=0/4 dropped=0:unresolved_aligned_residues=9/9",
                      f"{P} TEMPLATES census declared=4 kept=4 real=0 chains_templated=1 chains_all_dummy=1 verdict=all_dummy"]
    t = modes.templates_census(dummy, 4, "exact")
    assert t["verdict"] == "all_dummy" and t["token"] == "all_dummy:0/4"                                            # counted in the DONE token and the manifest; the run's status is unaffected
    t = modes.templates_census([guard], 3, "fast")                        # a templated input, no census line: named in the token, nothing else
    assert t["verdict"] == "census_missing" and t["token"] == "census_missing:3"
    t = modes.templates_census([], 3, "off")                              # the stock script: no census hook
    assert t["verdict"] == "uncensused" and t["token"] == "uncensused:3"
    assert modes.templates_census([f"{P} TEMPLATES guard=failed:ImportError:x"], 0, "exact") == {**modes.templates_census([], 0, "exact"), "guard": "failed:ImportError:x"}


def test_templates_declared_counts_every_polymer_chain(tmp_path):
    j = render("t", [chain("A", "MKV" * 10), chain("B", "ACGU", kind="rna"), ligand("L", ["ATP"])])
    p = tmp_path / "in.json"; p.write_text(__import__("json").dumps(j))
    assert inputs.templates_declared(str(p)) == 0
    j["sequences"][0]["protein"]["templates"] = [{"mmcif": "x", "queryIndices": [0], "templateIndices": [0]}] * 3
    p.write_text(__import__("json").dumps(j))
    assert inputs.templates_declared(str(p)) == 3 and inputs.templates_declared(str(tmp_path)) == 3
