"""The command line on the stub box: the exit tally (0 ok, 1 failed, 2 usage, 3 not active), the stock proof (and what fails it), the exact
route's launcher, script, row and lever environment (and the re-copy of a stale kit script), the fast route's launcher and class, the
the pass's printed lines, warm's FAIL path, and the two switches' agreement rule."""
import hashlib
import json
import os
import time
import pytest
import re
import sys

from af3_jax_opt import carry, cli, modes, outputs, stack
from .conftest import FPF, KIT, PALLAS, PKG


def _run(argv, capsys):
    rc = cli.main(argv)
    out = capsys.readouterr()
    return rc, out.err, out.out


def _fast_script_sha():
    return hashlib.sha256(open(os.path.join(KIT, "patches", "patched_files", "run_alphafold.py"), "rb").read()).hexdigest()


def test_pred_off_is_the_stock_command_at_its_defaults(box, capsys):
    """No stock flag stated: the stock script's line carries the inputs, the outputs and the weights — the input-side flags
    (`--norun_data_pipeline`, the variant's two), the padding flag and the EMPTY cache class — and nothing else."""
    from af3_jax_opt import settings
    inp = box.input_json("a", seeds=(1, 2))
    out = os.path.join(box.root, "out")
    rc, err, _ = _run(["pred", "--variant", "p2", "--mode", "off", "--json_path", inp, "--output_dir", out], capsys)
    assert rc == 0, err
    rec = box.stub_record()
    assert rec["script"] == "run_alphafold.py" and rec["script_path"] == os.path.join(box.repo, "run_alphafold.py") and rec["launcher"] is None
    assert rec["argv"][0] == rec["script_path"]                                                                        # nothing between the interpreter and the stock script
    assert not [k for k in rec["env"] if k.startswith(("AF3_JAX_", "AF3P_", "AF3_FLASHPAIRFORMER", "AF3_DIFFUSION_HOIST"))]
    assert rec["env"]["XLA_FLAGS"] == "--xla_gpu_enable_triton_gemm=false"                                             # the image's own ENV line (stock/PINS.json image.env)
    assert rec["flags"] == ["--norun_data_pipeline", "--cache_dir=", "--of3_weights", f"--model_dir={box.params_root}/p2", *modes.pad_flags("off"), f"--output_dir={out}", f"--json_path={inp}"]   # off with nothing stated: the EMPTY cache class `--cache_dir=` in the composed cache slot (autotune skipped; JAX cache ./jax under the cwd)
    assert rec["cwd"] == out                                                                                             # … and that process runs IN the pass's output directory   # upstream's own --buckets: the ONE padding of every mode
    assert modes.pad_flags("off") == modes.pad_flags("exact") and modes.pad_flags("off")[0].startswith("--buckets=64,128,")
    assert "[af3-jax-opt] ACTIVE mode=off variant=p2" in err and "preset=" not in err and f"cache={settings.CWD_JAX_CACHE}" in err and "autotune=skipped(cache_dir empty)" in err
    assert "[af3-jax-opt] STOCK script=run_alphafold.py" in err and "proof=ok" in err
    assert "[af3-jax-opt] DONE status=ok rc=0 predictions=10/10" in err                                    # 2 seeds x upstream's 5 samples (read from the source)
    rep, cmd = stack._REPORT, [x for x in err.splitlines() if x.startswith("[af3-jax-opt] COMMAND ")][0]   # the activation the ACTIVE line printed; the argv launched
    assert rep["mode"] == "off" and f"CACHE when=after dir={os.path.join(out, 'jax')} " in err and re.search(r" --cache_dir=(\s|$)", cmd) and rep["stated"] == [] and "preset" not in rep and not rep["launcher"] and " launcher=none " in err   # off at its defaults: the EMPTY class flag; the census reads the stock script's own <output_dir>/jax
    assert "proof=ok" in err and "predictions=10/10" in err and carry.carry_check()["ok"] and carry.carry_check()["files"] == 13
    assert len(outputs.output_files(out)) == 30 and re.findall(r"\] PHASE item=\S+ seed=(\d+)", err) == ["1", "2"]
    assert int(re.search(r"CACHE when=after \S+ files=(\d+)", err).group(1)) >= 1 and os.path.isfile(os.path.join(out, cli.LOG_NAME)) and " launcher=none " in err
    assert sorted(os.listdir(out)) == sorted([cli.LOG_NAME, "jax"] + [d for d in os.listdir(out) if os.path.isdir(os.path.join(out, d)) and d != "jax"])   # the pass writes the transcript and nothing else of its own


def test_pred_off_with_a_callers_cache_dir_and_stated_defaults(box, capsys):
    """The stock flags stated ride last, verbatim: a --cache_dir naming exact's class (stock inside the kit's autotune class, the shared
    cache the bitwise rows name) replaces the EMPTY class, and upstream's three defaults written out change nothing but the words."""
    inp = box.input_json("a", seeds=(1, 2))
    out = os.path.join(box.root, "out_det")
    stated = ["--num_recycles=10", "--num_diffusion_samples=5", "--flash_attention_implementation=triton"]
    rc, err, _ = _run(["pred", "--variant", "p2", "--mode", "off", "--json_path", inp, "--output_dir", out, f"--cache_dir={box.cache_dir('exact')}", *stated], capsys)
    assert rc == 0, err
    rec = box.stub_record()
    assert rec["script"] == "run_alphafold.py" and rec["launcher"] is None and rec["argv"][0] == rec["script_path"]
    assert not [k for k in rec["env"] if k.startswith(("AF3_JAX_", "AF3P_", "AF3_FLASHPAIRFORMER", "AF3_DIFFUSION_HOIST"))]
    assert rec["env"]["XLA_FLAGS"] == "--xla_gpu_enable_triton_gemm=false" and rec["env"]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true" and rec["env"]["XLA_CLIENT_MEM_FRACTION"] == "0.95"
    assert rec["flags"] == ["--norun_data_pipeline", "--of3_weights", f"--model_dir={box.params_root}/p2", *modes.pad_flags("off"), f"--output_dir={out}", f"--json_path={inp}",
                            f"--cache_dir={box.cache_dir('exact')}", *stated]                                                    # the caller's flags last; no `--cache_dir=` EMPTY class beside theirs
    assert rec["cwd"] == box.repo                                                                                         # a named cache: the process runs in the repo dir, not in the output dir
    assert "[af3-jax-opt] ACTIVE mode=off variant=p2" in err and f"cache={box.cache_dir('exact')}" in err and "autotune=caller_cache_dir" in err and "proof=ok" in err
    assert "[af3-jax-opt] DONE status=ok rc=0 predictions=10/10" in err
    rep, cmd = stack._REPORT, [x for x in err.splitlines() if x.startswith("[af3-jax-opt] COMMAND ")][0]
    assert f"CACHE when=after dir={box.cache_dir('exact')} " in err and int(re.search(r"CACHE when=after \S+ files=(\d+)", err).group(1)) >= 1 and f"--cache_dir={box.cache_dir('exact')}" in cmd
    assert rep["stated"] == stated


def test_pred_exact_composes_the_kit_script_and_the_pallas_levers(box, capsys):
    box.warm_cache(mode="exact")
    inp = box.input_json("b", seeds=(1,))
    out = os.path.join(box.root, "out_exact")
    rc, err, _ = _run(["pred", "--variant", "p2", "--mode", "exact", "--json_path", inp, "--output_dir", out], capsys)
    assert rc == 0, err
    rec = box.stub_record()
    assert rec["launcher"] == "levers_launch.py" and rec["script"] == "run_alphafold_fast.py"
    assert rec["argv"][:3] == [os.path.join(PKG, "levers_launch.py"), os.path.join(PALLAS, "patches"), os.path.join(box.repo, "run_alphafold_fast.py")]
    fast = os.path.join(box.repo, "run_alphafold_fast.py")
    assert hashlib.sha256(open(fast, "rb").read()).hexdigest() == _fast_script_sha()                                   # the kit's patched file, copied beside stock
    flags = rec["flags"]
    assert [t for t in flags if t.startswith("--featurisation_")] == ["--featurisation_workers=3", "--featurisation_prefetch=4"]   # the default row: L1 prefetch, its flags as the FAST row writes them
    assert f"--cache_dir={box.cache_dir('exact')}" in flags
    assert rec["env"]["AF3P_GLU_T"] == "1" and rec["env"]["AF3P_ATTN_CFG"] == "64,64,4,3" and "AF3_FLASHPAIRFORMER" not in rec["env"]
    assert not [k for k in rec["env"] if k.startswith("AF3_JAX_")]
    assert "[af3-jax-opt] ACTIVE mode=exact variant=p2 script=run_alphafold_fast.py launcher=levers_launch.py alphafold3=v3.1.4@bc32b22f" in err
    assert "levers=FIX1+GLUT+ATTNCFG" in err and "partial=" not in err
    assert "[af3-jax-opt] SCRIPT path=" in err and "state=copied" in err
    assert "lever_env=AF3P_GLU_T=1 AF3P_ATTN_CFG=64,64,4,3" in err
    rep, rec = stack._REPORT, box.stub_record()
    assert rep["levers_applied"] == ["FIX1", "GLUT", "ATTNCFG", "L1", "WRITER"] and rep["launcher"][0].endswith("levers_launch.py")
    ev = modes.lever_evidence("exact", err.splitlines())
    assert ev["ok"] and ev["reported"] == ["L-ATTNCFG+L-GLUT", "Featurisation prefetch enabled: 3 worker process(es), 4 item(s) ahead.", "Output writer enabled: result extraction and output writing run on one writer thread behind the next fold job."]   # the LEVERS line + the row levers' own lines (L1, WRITER)
    assert {k: rec["env"].get(k) for k in modes.pallas_levers()} == modes.pallas_levers()                  # the lever switches reached the model process


def test_check_big_reports_its_region(box, capsys):
    """`check --mode big` with no input reports the reach region (the memory levers' program); ONE
    `BIG NOTE region=…` line follows the ACTIVE line, whose last tokens are `region=<r> n_est=<n|unknown> n_star=1408`."""
    rc, err, _ = _run(["check", "--variant", "p2", "--mode", "big", "--allow-partial"], capsys)    # a cold class: partial, recorded
    assert rc == 0 and "launcher=big_launch.py" in err and "region=reach n_est=unknown n_star=1408" in err and err.count("BIG NOTE region=reach n_est=unknown n_star=1408: memory levers engaged") == 1


def test_pred_big_below_n_star_is_the_fast_line(box, capsys):
    """`pred --mode big` on an input at or below the memory line's size boundary (region fast, decided up front from the input's token
    estimate): the fast line's launcher, levers, switches, kernel_tile padding and cache class, labelled mode big; the ACTIVE line ends
    `region=fast n_est=<n> n_star=1408` and ONE `BIG NOTE region=fast ...` line names the memory levers inactive by size."""
    box.warm_cache(mode="fast")                                           # a cache warmed for fast serves the memory mode's fast region (the same class)
    inp = box.input_json("g", seeds=(1,))
    out = os.path.join(box.root, "out_big_fast")
    rc, err, _ = _run(["pred", "--variant", "p2", "--mode", "big", "--json_path", inp, "--output_dir", out], capsys)
    assert rc == 0, err
    rec = box.stub_record()
    assert rec["launcher"] == "fpf_launch.py" and rec["script"] == "run_alphafold_fast.py" and rec["env"]["AF3_FLASHPAIRFORMER"] == "both" and rec["env"]["AF3_JAX_DATTN"] == "1" and rec["env"]["AF3_JAX_TTR"] == "fast"   # big region fast: fast's program, the provider's tier word the mode's own
    assert f"--cache_dir={box.cache_dir('fast')}" in rec["flags"] and modes.pad_flags("fast")[0] in rec["flags"]
    assert "ACTIVE mode=big variant=p2 script=run_alphafold_fast.py launcher=fpf_launch.py" in err and "levers=FIX1+FPF_TRIMUL+FPF_TRIATT+FPF_HOIST+TTR+DATTN+TRIATT_XLA+SAMPLER_BF16+ATOM_ATTN+TRIMUL_CD+HOIST_LOGITS+COND_SHARE+ATOM_COND_HOIST+LNP+L1" in err
    act = [ln for ln in err.splitlines() if " ACTIVE mode=big" in ln][0]
    assert act.endswith(" region=fast n_est=10 n_star=1408") and " n_gpu=1 sharding=none " in act and all("=" in tok for tok in act.split("] ACTIVE ")[1].split())   # whitespace-free key=value tokens only; the region tokens END the line
    assert err.count("[af3-jax-opt] BIG NOTE region=fast n_est=10 n_star=1408: memory levers transition_shard,cond_shard,trimul_chunk inactive by size") == 1
    rep = stack._REPORT
    assert rep["mode"] == "big" and rep["region"]["region"] == "fast" and rep["cache_class"] == "__fast" and rep["region"]["estimate"]["n_est"] == 10 and "DONE status=ok" in err


@pytest.mark.parametrize("mode", ["off", "fast"])
def test_pred_carries_the_peak_probe_on_both_routes(box, capsys, mode):
    """Every pred pass arms opt_core.mem.peak's hook on the model process (the hook dir LAST on PYTHONPATH, OPT_PEAK_MEM_PATH under
    <output_dir>/peak/) — the stock route included, its proof unchanged — and prints ONE PEAK line after the process exits (here the stub
    interpreter creates no backend: the absence is named, the pass's status does not read it)."""
    from af3_jax_opt import peakmem
    for extra in ([],):
        if mode == "fast":
            box.warm_cache(mode="fast")
        inp = box.input_json(f"pk_{mode}", seeds=(1,))
        out = os.path.join(box.root, f"out_peak_{mode}")
        rc, err, _ = _run(["pred", "--variant", "p2", "--mode", mode, "--json_path", inp, "--output_dir", out, *extra], capsys)
        assert rc == 0, err
        rec = box.stub_record()
        hook_dir = rec["env"]["PYTHONPATH"].split(os.pathsep)[-1]
        assert rec["env"]["OPT_PEAK_MEM_PATH"] == os.path.join(hook_dir, "records", "peak_mem.json") and rec["env"]["OPT_PEAK_MEM_PER_PROCESS"] == "1"   # one record per process, inside the hook directory
        assert os.path.basename(hook_dir).startswith("af3_jax_opt_hook_") and not hook_dir.startswith(out)      # LAST on the path (nothing of a kit's own order shadowed); a temporary directory, never the output directory
        peaks = [ln for ln in err.splitlines() if " PEAK-NOTE scope=pass " in ln]                             # the pass's line (the word PEAK alone is the per-item line's, report.peak_item_lines)
        assert len(peaks) == 1 and peakmem.PEAK_RX.search(peaks[0]), peaks
        assert not [ln for ln in err.splitlines() if ln.startswith("[af3-jax-opt] PEAK ") and " item=" not in ln]   # no PEAK line without item= (the cross-engine grammar)
        st = peakmem.PEAK_RX.search(peaks[0]).group("status")
        assert st in ("no_backend_record", "no_probe_record"), st                                            # the stub interpreter never creates a backend
        if mode == "off":
            assert "proof=ok" in err                                                                          # the stock proof reads kit prefixes, the script and kit-only flags: the probe's variables are none of them
        assert "DONE status=ok" in err and not os.path.exists(os.path.join(out, "peak")) and not os.path.exists(hook_dir)   # the records lived in the hook directory and left with it


def test_pred_fast_runs_through_the_addons_launcher_in_its_own_class(box, capsys):
    box.warm_cache(mode="fast")
    inp = box.input_json("f", seeds=(1,))
    out = os.path.join(box.root, "out_fast")
    rc, err, _ = _run(["pred", "--variant", "p2", "--mode", "fast", "--json_path", inp, "--output_dir", out], capsys)
    assert rc == 0, err
    rec = box.stub_record()
    assert rec["launcher"] == "fpf_launch.py" and rec["script"] == "run_alphafold_fast.py"
    assert rec["argv"][:3] == [os.path.join(PKG, "fpf_launch.py"), os.path.join(FPF, "run_alphafold_flashpairformer.py"), os.path.join(box.repo, "run_alphafold_fast.py")]
    assert rec["env"]["AF3_FLASHPAIRFORMER"] == "both" and rec["env"]["AF3_DIFFUSION_HOIST"] == "1" and rec["env"]["AF3_JAX_DATTN"] == "1" and rec["env"]["AF3_JAX_TTR"] == "fast" and "AF3P_GLU_T" not in rec["env"]
    assert f"--cache_dir={box.cache_dir('fast')}" in rec["flags"] and box.cache_dir("fast").endswith("__fast")
    assert "ACTIVE mode=fast variant=p2 script=run_alphafold_fast.py launcher=fpf_launch.py" in err
    assert "levers=FIX1+FPF_TRIMUL+FPF_TRIATT+FPF_HOIST+TTR+DATTN+TRIATT_XLA+SAMPLER_BF16+ATOM_ATTN+TRIMUL_CD+HOIST_LOGITS+COND_SHARE+ATOM_COND_HOIST+LNP+L1" in err and f"cache={box.cache_dir('fast')}" in err
    rep, rec = stack._REPORT, box.stub_record()
    want_env = modes.lever_env("fast", modes.KIT_MODES["fast"]["levers"])
    assert rep["cache_class"] == "__fast" and {k: rec["env"].get(k) for k in want_env} == want_env and rec["env"]["AF3_JAX_DATTN"] == "1"
    ev = modes.lever_evidence("fast", err.splitlines())
    assert ev["reported"][0] == "af3_flashpairformer: mode=both (TriangleMultiplication=FlashTriangleMultiplication, GridSelfAttention=FlashGridSelfAttention)" and len(ev["reported"]) == 4   # install line, L1 line, WRITER line, SERVED line
    assert "[af3-jax-opt] LEVER name=LOCAL.step_invariant_hoist state=on impl=jax origin=kit lever=FPF_HOIST mode=fast site=flashpairformer/af3_flashpairformer/diffusion_hoist.py evidence=hoist=1" in err and err.count("[af3-jax-opt] LEVER ") == 16
    assert "[af3-jax-opt] LEVER name=F1.flash_triatt state=on impl=cuda+triton_aot origin=kit lever=TRIATT_XLA mode=fast site=opt/af3_jax_opt/inprocess/triatt_xla.py evidence=txla=24_rows=cuda_sm90a:20,k2b_aot:4_aside=none" in err
    assert "[af3-jax-opt] LEVER name=F4.autocast_policy state=on impl=jax origin=kit lever=SAMPLER_BF16 mode=fast site=opt/af3_jax_opt/inprocess/sampler_bf16.py evidence=sbf16=16_sites=atom:12,token:4" in err
    assert "[af3-jax-opt] LEVER name=F2.fpf_trimul_fast state=on impl=pallas_triton origin=kit lever=FPF_TRIMUL mode=fast site=flashpairformer/af3_flashpairformer/patch.py served=3 fallback=0" in err   # kernel levers: the core's served=/fallback= keys
    assert ev["hoist"] == "1" and ev["served_via"] == "traced" and ev["reported"][3].endswith("dattn=53 dattn_sites=diffusion:5,pairformer_single:48 ttr=4 ttr_routed=2 ttr_fallback=none sbf16=16 sbf16_sites=atom:12,token:4 txla=24 txla_rows=cuda_sm90a:20,k2b_aot:4 txla_aside=none atomattn=7 atomattn_sites=diffusion_decoder:3,diffusion_encoder:3,evoformer_encoder:1 atomattn_fallback=none hlog=2 hlog_dtype=float32 cshare=2 achoist=2 achoist_sites=enc:2,dec:2,passed:2 achoist_aside=none cnoise=off cnoise_rule=none tcd=24 tcd_rows=cd_trimul:24 tcd_aside=none tcd_word=fast tcd_uncovered=af3_tmpl_c64_ch64_incoming:2")   # the fast row's SERVED line (after the install line and the two row levers' lines)
    assert ev["dattn"] == "53" and ev["ttr"] == "4" and ev["ttr_fallback"] == "none" and ev["cnoise"] == "off" and ev["txla"] == "24" and ev["txla_aside"] == "none" and ev["atomattn"] == "7" and ev["hlog"] == "2" and ev["hlog_dtype"] == "float32" and ev["cshare"] == "2" and ev["achoist"] == "2" and ev["tcd"] == "24"
    assert "[af3-jax-opt] LEVER name=LOCAL.fused_transition state=on impl=pallas_triton origin=kit lever=TTR mode=fast site=opt/af3_jax_opt/inprocess/ttr.py evidence=ttr=4_routed=2" in err
    assert "[af3-jax-opt] LEVER name=F5.flash_attn_dense state=on impl=opt_core.kernels.pallas origin=kit lever=DATTN mode=fast site=opt/af3_jax_opt/inprocess/dattn.py evidence=dattn=53_sites=diffusion:5,pairformer_single:48_word=fast_rows=tokamax@triton:53_aside=none_uncovered=none" in err   # served by the shared core's provider by tier word; the lever's census line quoted (word, the arms that served, asides / uncovered families by name)
    assert "[af3-jax-opt] DATTN word=fast served=53 rows=tokamax@triton:53 aside=none uncovered=none" in err
    assert ev["ok"] and "DONE status=ok" in err and "predictions=5/5" in err
    assert ev["served"] == {"FPF_TRIMUL": {"fused": 3, "fallback": 0}, "FPF_TRIATT": {"fused": 2, "fallback": 0}} and ev["fallback_shapes"] == []
    assert "[af3-jax-opt] SERVED trimul fused=3 fallback=0 triatt fused=2 fallback=0 fallback_shapes=none" in ev["reported"][-1]



def test_pred_fast_kernel_installed_but_not_served_at_a_declared_shape_is_a_named_aside_exit_0(box, capsys):
    """The add-on serves padded sizes that are a multiple of its tile only and runs the stock classes otherwise; the tree launcher's SERVED
    line DECLARES those shapes (`fallback_shapes=`): an input on which every call ran the stock class at a declared shape (an exact-size input
    above the largest bucket, a caller's own --buckets value) completed by name — `aside:pad64:<n>` on ONE PARTIAL line and the DONE line's
    partial= field, status=ok, rc 0 (a step-aside by name is never a refusal); a kernel that engaged nothing and declared nothing stays rc 3."""
    box.warm_cache(mode="fast")
    inp = box.input_json("g", seeds=(1,))
    out = os.path.join(box.root, "out_fast_fb")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("STUB_FPF_FALLBACK", "6000")
        rc, err, _ = _run(["pred", "--variant", "p2", "--mode", "fast", "--json_path", inp, "--output_dir", out], capsys)
    assert rc == 0 and "DONE status=ok" in err and "PARTIAL levers_short: FPF_TRIMUL aside:pad64:1 at trimul:6000x6000x128:bfloat16" in err, err[-900:]
    ev = modes.lever_evidence("fast", err.splitlines())
    assert not ev["ok"] and ev["softened"] and ev["served"]["FPF_TRIATT"] == {"fused": 0, "fallback": 1}
    assert ev["fallback_shapes"] == ["trimul:6000x6000x128:bfloat16", "triatt:6000x6000x128:bfloat16"]
    assert "allow_partial" not in err


def test_pred_fast_kernel_engaged_with_some_fallback_is_named_and_exit_0(box, capsys):
    """A kernel that ENGAGED (fused > 0) and also ran the stock class at some trace-time call sites is named — the levers_short words on ONE
    `PARTIAL levers_short: …; those call sites ran the stock class — recorded, exit 0` line and on the DONE line's partial= field — and the run
    keeps its own exit code (status=ok, rc 0). A kernel that engaged no call at a DECLARED shape is the named aside of the test above; undeclared stays rc 3."""
    box.warm_cache(mode="fast")
    inp = box.input_json("gp", seeds=(1,))
    out = os.path.join(box.root, "out_fast_pfb")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("STUB_FPF_PARTIAL", "6000")
        rc, err, _ = _run(["pred", "--variant", "p2", "--mode", "fast", "--json_path", inp, "--output_dir", out], capsys)
    words = "levers_short: FPF_TRIMUL not served: fused=3 fallback=1 at trimul:6000x6000x128:bfloat16; FPF_TRIATT not served: fused=2 fallback=1 at triatt:6000x6000x128:bfloat16"
    assert rc == 0 and "DONE status=ok" in err, err[-2000:]
    assert f"[af3-jax-opt] PARTIAL {words}; those call sites ran the stock class — recorded, exit 0" in err and f"partial={words}" in err and "allow_partial=1" not in err
    ev = modes.lever_evidence("fast", err.splitlines())                     # the evidence the wrapper's exit rule read off the same lines
    assert not ev["ok"] and ev["softened"] and ev["served"]["FPF_TRIMUL"] == {"fused": 3, "fallback": 1}


def test_exit_tally(box, capsys):
    inp = box.input_json("c", seeds=(1,))
    out = os.path.join(box.root, "o")
    assert _run(["pred", "--variant", "p2", "--mode", "turbo", "--json_path", inp, "--output_dir", out], capsys)[0] == 2        # no such mode
    assert _run(["pred", "--mode", "off", "--json_path", inp, "--output_dir", out], capsys)[0] == 2                              # variant required
    rc, err, _ = _run(["pred", "--variant", "p2", "--json_path", inp, "--output_dir", out], capsys)
    assert rc == 0 and "ACTIVE mode=fast" in err and "partial=cold_cache" in err and "NOT ACTIVE" not in err                    # --mode omitted = fast (the package default): a cold class is named (an untested shape, not a refusal), the run proceeds
    stack._REPORT = None; stack._LAUNCHED.clear()
    assert _run(["pred", "--variant", "p2", "--mode", "off", "--output_dir", out], capsys)[0] == 2                             # an input is required
    for flag in ("--of3_weights", "--use_msa_server", "--norun_data_pipeline"):             # the stock command line's own flags: never refused by the wrapper (they ride last: absl's last-wins order)
        assert _run(["pred", "--variant", "p2", "--mode", "off", "--json_path", inp, "--output_dir", out, flag], capsys)[0] != 2, flag
        stack._REPORT = None; stack._LAUNCHED.clear()
    rc, err, _ = _run(["pred", "--variant", "p2", "--mode", "exact", "--json_path", inp, "--output_dir", out, f"--cache_dir={out}_cold_class"], capsys)
    assert rc == 0 and "partial=cold_cache" in err and f"cache={out}_cold_class" in err, err                                                       # the caller's --cache_dir IS the class the mode reads: a cold one is named by the same rule
    stack._REPORT = None; stack._LAUNCHED.clear()
    rc, err, _ = _run(["pred", "--variant", "p2", "--mode", "off", "--json_path", inp, "--output_dir", out, "--featurisation_workers=3"], capsys)
    assert rc == 3 and "exact_only_flags=['--featurisation_workers=3'] proof=FAILED" in err, err                                                    # a kit-script flag on the stock route: the STOCK line names it, the route is not the stock one
    stack._REPORT = None; stack._LAUNCHED.clear()
    rc, err, _ = _run(["pred", "--variant", "p2", "--mode", "exact", "--json_path", inp, "--output_dir", out], capsys)
    assert rc == 0 and "ACTIVE mode=exact" in err and "partial=cold_cache" in err                                              # cold cache: named, runs
    stack._REPORT = None; stack._LAUNCHED.clear()
    os.environ["STUB_FAIL"] = "1"
    try:
        rc, err, _ = _run(["pred", "--variant", "p2", "--mode", "off", "--json_path", inp, "--output_dir", out], capsys)
    finally:
        del os.environ["STUB_FAIL"]
    assert rc == 1 and "DONE status=FAILED rc=7" in err                                                                        # the model process failed
    stack._LAUNCHED.clear(); stack._REPORT = None
    rc, err, _ = _run(["pred", "--variant", "p2", "--mode", "off", "--json_path", inp, "--output_dir", out], capsys)
    assert rc == 0


def test_switch_disagreement(box, capsys, monkeypatch):
    monkeypatch.setenv("AF3_JAX_OPT", "exact")
    rc, err, _ = _run(["check", "--variant", "p2", "--mode", "off"], capsys)
    assert rc == 2 and "disagrees" in err
    monkeypatch.setenv("AF3_JAX_VARIANT", "p3")
    rc, err, _ = _run(["check", "--variant", "p2", "--mode", "exact"], capsys)
    assert rc == 2 and "disagrees" in err
    monkeypatch.delenv("AF3_JAX_OPT"); monkeypatch.delenv("AF3_JAX_VARIANT")


def test_check_json_and_env_route(box, capsys, monkeypatch):
    rc, err, out = _run(["check", "--variant", "p2", "--json"], capsys)                      # --mode omitted: the package default is fast; its class is cold — named, active
    assert rc == 0 and json.loads(out)["mode"] == "fast" and json.loads(out)["active"] and json.loads(out)["partial_conditions"] == ["cold_cache"] and "partial=cold_cache" in err
    stack._REPORT = None
    rc, err, out = _run(["check", "--variant", "p2", "--mode", "off", "--json"], capsys)
    rep = json.loads(out)
    assert rc == 0 and rep["mode"] == "off" and rep["command"].startswith(box.py + " " + os.path.join(box.repo, "run_alphafold.py")) and rep["carry_all"]["ok"]
    monkeypatch.setenv("AF3_JAX_OPT", "exact"); monkeypatch.setenv("AF3_JAX_VARIANT", "p2")
    rc, err, out = _run(["check", "--json"], capsys)
    assert rc == 0 and json.loads(out)["mode"] == "exact" and json.loads(out)["variant"] == "p2" and json.loads(out)["cold_cache"] is True and "partial=cold_cache" in err
    box.warm_cache(mode="exact"); stack._REPORT = None
    rc, err, out = _run(["check", "--json"], capsys)
    rep = json.loads(out)
    assert rc == 0 and rep["active"] and rep["command"].startswith(box.py + " " + os.path.join(PKG, "levers_launch.py") + " " + os.path.join(PALLAS, "patches") + " ")
    assert "run_alphafold_fast.py" in rep["command"] and "--featurisation_workers=3" in rep["command"]


def test_late_activation_through_the_cli(box, capsys):
    inp = box.input_json("d", seeds=(1,))
    assert _run(["pred", "--variant", "p2", "--mode", "off", "--json_path", inp, "--output_dir", os.path.join(box.root, "o1")], capsys)[0] == 0
    box.warm_cache(mode="exact")
    rc, err, _ = _run(["pred", "--variant", "p2", "--mode", "exact", "--json_path", inp, "--output_dir", os.path.join(box.root, "o2")], capsys)
    assert rc == 3 and "late activation refused" in err


def test_warm_builds_each_modes_class(box, capsys):
    from af3_jax_opt import warm
    res = warm.warm("p2", mode="exact")
    assert res["status"] == "PASS", res
    assert not os.path.isdir(os.path.join(box.cache_dir("exact"), "executables")) and res["executables"] == [] and res["predictions"] == 25 == res["predictions_expected"]   # 5 seeds x 5 samples
    assert res["cache"]["before"] is None and res["cache"]["after"]["files"] >= 1 and box.stub_record()["launcher"] == "levers_launch.py"
    assert "WARM PASS mode=exact variant=p2" in warm.summary_line(res) and box.stub_record()["env"]["AF3P_GLU_T"] == "1"
    assert box.stub_record()["env"]["PYTHONPATH"].split(os.pathsep)[0] == stack.core_dir()                          # a lever mode's model process imports the core's carried kernels by path (warm as pred)
    assert [t for t in box.stub_record()["flags"] if t.startswith("--featurisation_")] == ["--featurisation_workers=3", "--featurisation_prefetch=4"]   # the row lever's flags
    stack._REPORT = None; stack._LAUNCHED.clear()
    res = warm.warm("p2", mode="fast")
    assert res["status"] == "PASS" and res["cache_dir"] == box.cache_dir("fast") and box.stub_record()["launcher"] == "fpf_launch.py"
    assert box.stub_record()["env"]["AF3_FLASHPAIRFORMER"] == "both" and box.stub_record()["env"]["PYTHONPATH"].split(os.pathsep)[0] == stack.core_dir()
    stack._REPORT = None; stack._LAUNCHED.clear()
    res = warm.warm("p2", mode="off", user_args=[f"--cache_dir={box.cache_dir('exact')}"])            # a stock warm of exact's class (the shared cache)
    assert res["status"] == "PASS" and box.stub_record()["script"] == "run_alphafold.py" and box.stub_record()["launcher"] is None
    assert box.stub_record()["env"].get("PYTHONPATH", "") == os.environ.get("PYTHONPATH", "")                       # off stays stock-pure: the tree adds nothing to its path
    assert res["cache_dir"] == box.cache_dir("exact") and f"--cache_dir={box.cache_dir('exact')}" in box.stub_record()["flags"] and "--cache_dir=" not in box.stub_record()["flags"]
    stack._REPORT = None; stack._LAUNCHED.clear()
    from af3_jax_opt import settings
    mine = os.path.join(box.root, "a_cache_of_my_own")
    res = warm.warm("p2", mode="off", user_args=[f"--cache_dir={mine}", "--num_recycles=3"])                 # any dir the caller names; the other stock flags ride along
    assert res["status"] == "PASS" and res["cache_dir"] == mine and os.path.isdir(mine) and box.stub_record()["flags"][-2:] == [f"--cache_dir={mine}", "--num_recycles=3"]
    assert res["activation"]["autotune"] == "caller_cache_dir" and res["activation"]["cache_dir"] == mine
    assert res["activation"]["stated"] == ["--num_recycles=3"]
    stack._REPORT = None; stack._LAUNCHED.clear()
    res = warm.warm("p2", mode="off")                                                                      # no --cache_dir: a fresh ./jax per pass — nothing to warm, said by name
    assert res["status"] == "SKIP" and res["cache_dir"] == settings.CWD_JAX_CACHE and "nothing to warm" in res["reason"]


def test_stock_proof_fails_on_a_dirty_environment_or_command():
    from af3_jax_opt import stock_pred
    clean = {"PATH": "/usr/bin", "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false"}
    argv = ["/venv/bin/python", "/repo/run_alphafold.py", "--norun_data_pipeline", "--cache_dir=/c", "--json_path=/i.json"]
    assert stock_pred.proof(clean, argv)["ok"]
    for k in ("AF3_JAX_OPT", "AF3P_GLU_T", "AF3_FLASHPAIRFORMER", "AF3_DIFFUSION_HOIST"):
        p = stock_pred.proof(dict(clean, **{k: "x"}), argv)
        assert not p["ok"] and p["env_present"] == [k], k                                                              # a set switch of the package or an add-on
    p = stock_pred.proof(clean, argv + ["--featurisation_workers=3"])
    assert not p["ok"] and p["exact_only_flags"] == ["--featurisation_workers=3"]                                      # a kit-row flag
    p = stock_pred.proof(clean, ["/venv/bin/python", "/repo/run_alphafold_fast.py"] + argv[2:])
    assert not p["ok"] and p["script"] == "run_alphafold_fast.py"                                                      # the kit script
    p = stock_pred.proof(clean, ["/venv/bin/python", "/x/levers_launch.py", "/p", "/repo/run_alphafold.py"] + argv[2:])
    assert not p["ok"] and p["script"] == "levers_launch.py"                                                           # a launcher in front of the stock script
    assert "proof=FAILED" in stock_pred.proof_line(p)


def test_stale_fast_script_is_recopied(box, capsys):
    box.warm_cache(mode="exact")
    inp = box.input_json("d", seeds=(1,))
    fast = os.path.join(box.repo, "run_alphafold_fast.py")
    with open(fast, "w") as f:
        f.write("# a stale copy, not the kit's bytes\n")
    rc, err, _ = _run(["pred", "--variant", "p2", "--mode", "exact", "--json_path", inp, "--output_dir", os.path.join(box.root, "o4")], capsys)
    assert rc == 0, err
    assert hashlib.sha256(open(fast, "rb").read()).hexdigest() == _fast_script_sha() and "state=copied" in err
    stack._LAUNCHED.clear(); stack._REPORT = None
    rc, err, _ = _run(["pred", "--variant", "p2", "--mode", "exact", "--json_path", inp, "--output_dir", os.path.join(box.root, "o5")], capsys)
    assert rc == 0 and "state=present" in err                                                                          # the kit's bytes are left alone


def test_warm_fail_path(box, capsys):
    from af3_jax_opt import warm
    os.environ["STUB_FAIL"] = "1"
    try:
        res = warm.warm("p2", mode="exact")
    finally:
        del os.environ["STUB_FAIL"]
    assert res["status"] == "FAIL" and res["exit_code"] == 7 and res["reason"].startswith("model process exited 7; predictions=0 expected=25")
    assert "WARM FAIL mode=exact variant=p2 key=" in warm.summary_line(res) and "reason=model process exited 7" in warm.summary_line(res)
    stack._REPORT = None; stack._LAUNCHED.clear()
    os.environ["STUB_FAIL"] = "1"
    try:
        rc, err, _ = _run(["warm", "--variant", "p2", "--mode", "exact"], capsys)
    finally:
        del os.environ["STUB_FAIL"]
    assert rc == 1 and "WARM FAIL" in err
    stack._REPORT = None; stack._LAUNCHED.clear()
    assert _run(["warm", "--variant", "p2", "--mode", "turbo"], capsys)[0] == 2                                         # no such mode: usage, no traceback
    stack._REPORT = None; stack._LAUNCHED.clear()
    rc, err, _ = _run(["warm", "--variant", "p2"], capsys)
    assert rc == 0 and "WARM PASS mode=fast" in err                                                                     # --mode omitted = the package default


def test_warm_partial_path(box, capsys):
    """The exit rule for warm, one code per condition: the cold class at activation is warm's purpose (recorded on the ACTIVE line and in the
    WARM line, never refused); levers reported short are PARTIAL — rc 3 by name, nothing waives it (a mode is all of its levers); a failed model
    process is FAIL (rc 1)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("STUB_NO_ROW_LINES", "1")                                # the kit script printed no L1 line: the row lever reported short
        rc, err, _ = _run(["warm", "--variant", "p2", "--mode", "exact"], capsys)
    assert rc == 3 and "WARM PARTIAL mode=exact" in err and "partial=levers_short" in err and "allow_partial" not in err and "partial=cold_cache" in err, err
    assert "record=" not in err and not os.path.exists(os.path.join(stack.cache_root(), "warm_records"))   # warm writes the class and prints its line; no record file
    stack._REPORT = None; stack._LAUNCHED.clear()
    rc, err, _ = _run(["warm", "--variant", "p2", "--mode", "off", "--cache_dir", os.path.join(box.root, "warm_mine")], capsys)
    assert rc == 0 and "WARM PASS mode=off" in err and "partial=" not in err                                             # off writes no executable: not a condition (a cache the caller named)
    stack._REPORT = None; stack._LAUNCHED.clear()
    rc, err, _ = _run(["warm", "--variant", "p2", "--mode", "off"], capsys)
    assert rc == 0 and "WARM SKIP mode=off" in err and "reason=mode_off_keeps_a_fresh_JAX_cache_per_pass" in err        # no --cache_dir: nothing to warm, by name
    stack._REPORT = None; stack._LAUNCHED.clear()
    mine = os.path.join(box.root, "my_class")
    rc, err, _ = _run(["warm", "--variant", "p2", "--mode", "exact", f"--cache_dir={mine}"], capsys)
    assert rc != 2 and f"cache={mine}" in err, err                                                                     # the caller's --cache_dir is the class warm builds
    stack._REPORT = None; stack._LAUNCHED.clear()
    rc, err, _ = _run(["warm", "--variant", "p2", "--mode", "exact"], capsys)
    warm_line = [ln for ln in err.splitlines() if "] WARM " in ln][0]
    assert rc == 0 and "WARM PASS mode=exact" in warm_line and "partial=" not in warm_line and "executables=0" in warm_line   # the class exists now (warmed above): nothing partial
    assert "partial=" not in err and "cold_cache" not in err
    stack._REPORT = None; stack._LAUNCHED.clear()


def test_lever_evidence_gates_the_run(box, capsys):
    """exact without the launcher's LEVERS line (or with fewer levers than the row sets) and fast without the add-on's install line (or at
    another mode) are FAILED runs with the reason named — never a silent stock run under a lever mode's label."""
    import pytest
    from af3_jax_opt import modes as _m
    box.warm_cache(mode="exact"); box.warm_cache(mode="fast")
    inp = box.input_json("g", seeds=(1,))
    out = os.path.join(box.root, "o_ev")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_m, "pallas_levers", lambda kit=None: {"AF3P_GLU_T": "1", "AF3P_ATTN_CFG": "64,64,4,3", "AF3P_EXTRA": "1"})   # the row sets 3, the process reports 2
        rc, err, _ = _run(["pred", "--variant", "p2", "--mode", "exact", "--json_path", inp, "--output_dir", out], capsys)
    assert rc == 3 and "DONE status=partial" in err and "levers_short: levers active=L-ATTNCFG+L-GLUT: 2 of the 3 the LEVERS row sets" in err
    assert "DONE status=partial" in err and "partial=levers_short: levers active=L-ATTNCFG+L-GLUT: 2 of the 3 the LEVERS row sets" in err and "allow_partial=1" not in [x for x in err.splitlines() if " DONE " in x][0]
    stack._REPORT = None; stack._LAUNCHED.clear()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_m, "fpf_env", lambda kit=None: {"AF3_FLASHPAIRFORMER": "trimul", "AF3_DIFFUSION_HOIST": "1"})
        mp.setattr(_m, "mode_env", lambda mode: {"AF3_FLASHPAIRFORMER": "both", "AF3_DIFFUSION_HOIST": "1"} if mode == "fast" else {})   # the process installs 'both', the tree wants 'trimul'
        rc, err, _ = _run(["pred", "--variant", "p2", "--mode", "fast", "--json_path", inp, "--output_dir", out], capsys)
    assert rc == 3 and "DONE status=partial" in err and "af3_flashpairformer mode=both, the tree sets trimul" in err
    stack._REPORT = None; stack._LAUNCHED.clear()
    L1 = "Featurisation prefetch enabled: 3 worker process(es), 4 item(s) ahead.\nOutput writer enabled: result extraction and output writing run on one writer thread behind the next fold job."   # the two row levers' lines (L1, WRITER) as one transcript element
    sub = lambda d: {k: d[k] for k in ("ok", "reported", "reason")}
    assert sub(modes.lever_evidence("exact", [L1])) == {"ok": False, "reported": [L1, L1], "reason": "no LEVERS line: the launcher did not run"}   # the one element carries both row levers' lines: reported once per row lever
    assert modes.lever_evidence("exact", [L1])["per_lever"]["GLUT"]["state"] == "skipped" and modes.lever_evidence("exact", [L1])["per_lever"]["L1"]["state"] == "on"
    assert modes.lever_evidence("exact", [])["reason"] == "no LEVERS line: the launcher did not run; L1 printed no 'Featurisation prefetch enabled:' line; WRITER printed no 'Output writer enabled:' line"
    assert sub(modes.lever_evidence("fast", ["x", L1])) == {"ok": False, "reported": [L1, L1], "reason": "no af3_flashpairformer install line: the add-on did not install"}
    assert modes.lever_evidence("exact", ["[af3-jax-opt] LEVERS active=none af3_pallas_levers.py=x script=y sha256=z", L1])["reason"] == "levers active=none: 0 of the 2 the LEVERS row sets"
    assert modes.lever_evidence("off", []) == {"ok": True, "reported": [], "reason": None, "per_lever": {}}


def test_prediction_count_is_gated_against_the_inputs(box, capsys):
    """expected = the inputs' seeds x the run's --num_diffusion_samples: the command line's value (detrecipe 5, kit 1) or, for the
    pass-through line that carries none, the flag's default read from the stock source (5) — never a guess."""
    from af3_jax_opt import outputs
    inp = box.input_json("h", seeds=(1, 2, 3))
    out = os.path.join(box.root, "o_cnt")
    rc, err, _ = _run(["pred", "--variant", "p2", "--mode", "off", "--json_path", inp, "--output_dir", out], capsys)
    assert rc == 0 and "predictions=15/15" in err
    assert "predictions=15/15" in err and "failures=" not in err and "DONE status=ok" in err
    stack._REPORT = None; stack._LAUNCHED.clear()
    out2 = os.path.join(box.root, "o_cnt_kit")
    rc, err, _ = _run(["pred", "--variant", "p2", "--mode", "off", "--json_path", inp, "--output_dir", out2, "--num_diffusion_samples=1"], capsys)
    assert rc == 0 and "predictions=3/3" in err                                                                         # the stated sample count, read off the composed line
    assert outputs.expected_predictions(["--num_diffusion_samples=4"], 3) == 12 and outputs.expected_predictions(["--x", "--num_diffusion_samples=1"], 7) == 7
    assert outputs.expected_predictions(["--num_recycles=10"], 3) == 15 == outputs.expected_predictions([], 3)          # the pass-through: upstream's default, read from the source


def test_dattn_sites_are_expected_under_their_declared_impl_and_a_reused_output_dir_counts_this_pass_only(tmp_path):
    """(a) `--flash_attention_implementation=xla` (upstream's enum) moves UPSTREAM's attention sites to xla; DATTN's own sites are served by the
    shared core's provider by tier word and the rows that served are READ from the lever's census line (inprocess/dattn.py census_line `rows=`;
    kernels.lever_impls -> the tokamax implementations those rows call under) — the kernels gate expects those calls by name: a completed fast
    pass with the flag conforms (was REFUSED:dpa_calls, rc 5); without DATTN in the composition, or without its census line, the same census
    is a refusal. (b) a second pass into the same --output_dir (upstream writes the new job into a timestamped sibling directory): the pass counts
    the models IT wrote, not the earlier job's (was predictions=4/2, rc 1)."""
    from af3_jax_opt import kernels, outputs
    from af3_jax_opt.inprocess import dattn
    exp = kernels.expected("fast", ["FPF_TRIATT", "FPF_TRIMUL", "DATTN"], "xla")
    reading = {**exp, "attn_impls": "cudnn,triton,xla"}                                                               # a probe reading that conforms key by key
    sc = {"probe": reading, "probes": 1, "traps": {k: [] for k in kernels.TRAPS}, "census": {"dpa": {"triton": 8}, "glu": {}, "lines": 1}, "refused_inprocess": False}
    lines = [dattn.census_line({"word": "fast", "traced": 53, "rows": {"tokamax@triton": 53}, "aside": {}, "uncovered": {}})]   # the lever's census line as its exit hook prints it
    assert lines[0] == "[af3-jax-opt] DATTN word=fast served=53 rows=tokamax@triton:53 aside=none uncovered=none"
    assert kernels.lever_impls(["FPF_TRIATT", "DATTN"], lines) == {"DATTN": ("triton",)} and kernels.lever_impls(["FPF_TRIATT"], lines) == {} and kernels.lever_impls(["DATTN"], []) == {"DATTN": ()}
    assert kernels.refusal(exp, sc, 0, lever_impls=kernels.lever_impls(["DATTN"], lines)) == []                      # DATTN's 8 triton calls under a requested xla: expected by name (the served row says triton)
    bare = kernels.refusal(exp, sc, 0)
    assert len(bare) == 1 and bare[0].startswith("dpa_calls: 8 attention call(s) under implementation=triton, the route requests xla")   # no lever declares them: refused
    assert kernels.refusal(exp, sc, 0, lever_impls=kernels.lever_impls(["DATTN"], [])) == bare                       # DATTN in the composition but no census line: nothing declared, refused the same
    sc2 = {**sc, "census": {"dpa": {"cudnn": 3}, "glu": {}, "lines": 1}}
    assert kernels.refusal(exp, sc2, 0, lever_impls=kernels.lever_impls(["DATTN"], lines))                            # an implementation NOBODY served or requested is still a refusal
    mixed = [dattn.census_line({"word": "fast", "traced": 53, "rows": {"tokamax@triton": 5, "tokamax@xla": 48}, "aside": {}, "uncovered": {}})]   # a small input: the single kind's 400-token cell names tokamax's xla path, the DiT's triton
    exp_t = kernels.expected("fast", ["FPF_TRIATT", "FPF_TRIMUL", "DATTN"], "triton")
    sc3 = {**sc, "probe": {**exp_t, "attn_impls": "cudnn,triton,xla"}, "census": {"dpa": {"triton": 5, "xla": 48}, "glu": {}, "lines": 1}}
    assert kernels.lever_impls(["DATTN"], mixed) == {"DATTN": ("triton", "xla")} and kernels.refusal(exp_t, sc3, 0, lever_impls=kernels.lever_impls(["DATTN"], mixed)) == []
    assert kernels.refusal(exp_t, sc3, 0, lever_impls=kernels.lever_impls(["DATTN"], lines))                          # the same census against a line that served triton only: the xla calls are nobody's, refused
    out = tmp_path / "out"; old = out / "job" / "seed-1_sample-0"; old.mkdir(parents=True)
    (old / "job_seed-1_sample-0_model.cif").write_text("x"); (out / "job" / "seed-1_sample-1").mkdir(); (out / "job" / "seed-1_sample-1" / "job_seed-1_sample-1_model.cif").write_text("x")
    before = outputs.prediction_files(str(out))
    assert len(before) == 2 and outputs.count_predictions(str(out)) == 2 and outputs.count_predictions(str(out), before=before) == 0   # nothing written since the snapshot
    new = out / "job_20260912_010203"                                                                                # upstream's timestamped sibling for the second pass
    for smp in (0, 1):
        d = new / f"seed-1_sample-{smp}"; d.mkdir(parents=True); (d / f"job_seed-1_sample-{smp}_model.cif").write_text("y")
    assert outputs.count_predictions(str(out)) == 4 and outputs.count_predictions(str(out), before=before) == 2       # this pass's two, not four
    time.sleep(0.01); (old / "job_seed-1_sample-0_model.cif").write_text("z")                                          # a model REWRITTEN after the snapshot (same path) is this pass's too
    assert outputs.count_predictions(str(out), before=before) == 3


def test_declared_full_fallback_is_a_named_aside_exit_0_and_norun_inference_exits_0(box, capsys):
    """A caller's own `--buckets 480` (or an exact-size input above the largest bucket): the padded size is not a multiple of the kernel tile, so
    EVERY fused FPF/TTR call runs the stock class and the launcher DECLARES the shapes — the run completed by name: `aside:pad64:<n>`, softened
    (a PARTIAL line, exit 0), never levers_short exit 3; an undeclared fused=0 stays a shortfall. `--norun_inference` (upstream's own flag) asks for
    no model inference: nothing for the census to judge, no prediction due — DONE ok, exit 0, named."""
    from af3_jax_opt import modes, settings
    inst = "af3_flashpairformer: mode=both (TriangleMultiplication=FlashTriangleMultiplication, GridSelfAttention=FlashGridSelfAttention), jax 0.10.2 backend gpu"
    L1 = "Featurisation prefetch enabled: 1 worker"
    wr = "Output writer enabled: result extraction"
    served480 = ("[af3-jax-opt] SERVED trimul fused=0 fallback=24 triatt fused=0 fallback=24 fallback_shapes=trimul:480x480x128:bfloat16,triatt:480x480x128:bfloat16 hoist=2 tiles=own:9.0 cc=9.0 "
                 "dattn=10 dattn_sites=diffusion:2,pairformer_single:8 ttr=14 ttr_routed=0 ttr_fallback=rows:480 sbf16=16 sbf16_sites=atom:12,token:4 txla=24 txla_rows=cuda_sm90a:20,k2b_aot:4 txla_aside=none "
                 "atomattn=6 atomattn_sites=diffusion_decoder:2,diffusion_encoder:2,evoformer_encoder:2 atomattn_fallback=none hlog=2 hlog_dtype=float32 cshare=5 achoist=5 achoist_sites=enc:5,dec:5,passed:0 achoist_aside=none "
                 "cnoise=off cnoise_rule=none tcd=24 tcd_rows=cd_trimul:24 tcd_aside=none tcd_word=fast tcd_uncovered=none") + "\n[af3-jax-opt] LNP served=150 word=fast rows=cd_ln:62,xla:88 units=msa:88,pair:58,tmpl:4 routed=adaptive:40,axis:2,channels:12 aside=none uncovered=none"
    ev = modes.lever_evidence("fast", [inst, L1, wr, served480], modes.resolve("fast")["levers"])
    assert not ev["ok"] and ev["softened"], ev["reason"]                                                                # softened = the exit rule's PARTIAL line, exit 0
    assert "FPF_TRIMUL aside:pad64:24 at trimul:480x480x128:bfloat16" in ev["fell_back"] and "FPF_TRIATT aside:pad64:24 at triatt:480x480x128:bfloat16" in ev["fell_back"] and "TTR aside at rows:480" in ev["fell_back"]
    undeclared = modes.lever_evidence("fast", [inst, L1, wr, served480.replace("fallback_shapes=trimul:480x480x128:bfloat16,triatt:480x480x128:bfloat16", "fallback_shapes=none")], modes.resolve("fast")["levers"])
    assert not undeclared["ok"] and not undeclared["softened"] and "FPF_TRIMUL not served: fused=0 fallback=24" in undeclared["reason"]   # fused=0 with nothing declared keeps exit 3
    assert settings.flag_false(["--norun_inference"], "run_inference") and settings.flag_false(["--run_inference=false"], "run_inference") and settings.flag_false(["--run_inference=0"], "run_inference")
    assert not settings.flag_false(["--run_inference=true"], "run_inference") and not settings.flag_false(["--norun_inference", "--run_inference"], "run_inference") and not settings.flag_false([], "run_inference")
    inp = box.input_json("noinf", seeds=(1,))
    out = os.path.join(box.root, "o_noinf")
    rc, err, _ = _run(["pred", "--variant", "p2", "--mode", "off", "--json_path", inp, "--output_dir", out, "--norun_inference"], capsys)
    assert rc == 0 and "no inference requested (--norun_inference)" in err and "DONE status=ok" in err and "partial=no_inference_requested" in err, err[-600:]


def test_expected_predictions_honours_num_seeds_and_both_flag_spellings(tmp_path):
    """The exit rule counts what the stock script writes for EVERY stock-accepted count flag: `--num_seeds N` expands each fold input to N
    seeds (run_alphafold.py: with_multiple_seeds — one seed in the JSON, N > 1, else upstream raises), so a one-seed JSON with
    `--num_seeds=3` and 5 samples writes 15 predictions, not 5; `--num_diffusion_samples` counts in either absl spelling; no flag =
    the JSON's own modelSeeds x upstream's default sample count."""
    from af3_jax_opt import inputs, settings
    one = tmp_path / "one.json"; one.write_text('{"dialect": "alphafold3", "version": 2, "name": "x", "modelSeeds": [7], "sequences": []}')
    d = tmp_path / "dir"; d.mkdir()
    for i in range(2):
        (d / f"in{i}.json").write_text('{"dialect": "alphafold3", "version": 2, "name": "y", "modelSeeds": [1], "sequences": []}')
    assert outputs.num_seeds(["--num_seeds=3"]) == 3 == outputs.num_seeds(["--json_path=x", "--num_seeds", "3"]) and outputs.num_seeds([]) is None
    assert inputs.seeds_in(str(one)) == 1 and inputs.seeds_in(str(one), outputs.num_seeds(["--num_seeds=3"])) == 3
    assert inputs.seeds_in(str(d), 4) == 8                                                                               # N per fold input, as upstream expands each
    argv = ["--num_seeds=3", "--num_diffusion_samples=5"]
    assert outputs.expected_predictions(argv, inputs.seeds_in(str(one), outputs.num_seeds(argv))) == 15                  # json seeds=1, --num_seeds=3, samples=5 -> 15
    assert outputs.expected_predictions([], inputs.seeds_in(str(one), outputs.num_seeds([]))) == 1 * int(settings.upstream_defaults()["num_diffusion_samples"])   # no flag: 1 seed x upstream's default
    assert outputs.expected_predictions(["--num_diffusion_samples", "2"], 3) == 6 == outputs.expected_predictions(["--num_diffusion_samples=2"], 3)   # both spellings
    assert settings.flag_int(["--num_seeds=2", "--num_seeds=4"], "--num_seeds") == 4 and settings.flag_int(["--num_seeds", "--other"], "--num_seeds") is None


def test_unpinned_weights_warn_and_run(unpinned_box, capsys):
    """Weights whose digest is not stock/PINS.json variants.p2.converted are NOT PINNED, named on one line, and RUN (rule: warn-and-run):
    check is active rc 0, pred runs rc 0; the digest is this process's own read of the file, remembered in the on-disk memo under the cache root."""
    from af3_jax_opt import variants
    rep = stack.check("off", "p2")
    assert rep["active"] and rep["params"]["pinned"] is False and rep["params"]["matches_pin"] is False and rep["reason"] is None
    assert len(rep["params"]["sha256"]) == 64 and rep["params"]["source"] == "variant"
    from af3_jax_opt import digest_memo
    assert rep["params"]["sha256"] == stack.sha256_file(rep["params"]["file"]) and rep["params"]["digest_cached_utc"] is None   # computed by this process
    table = json.load(open(os.path.join(unpinned_box.cache_root, digest_memo.MEMO_NAME)))          # the on-disk memo entry, written after the full hash
    assert table[digest_memo.stat_key(rep["params"]["file"])]["sha256"] == rep["params"]["sha256"]
    want = f"[af3-jax-opt] WEIGHTS weights sha256={rep['params']['sha256'][:12]} NOT PINNED — the kit's speed and output statements hold for the pinned weights only"
    stack._REPORT = None
    rc, err, _ = _run(["check", "--variant", "p2", "--mode", "off"], capsys)
    assert rc == 0 and want in err and "NOT ACTIVE" not in err, err
    for mode in ("off", "exact"):
        stack._REPORT = None; stack._LAUNCHED.clear()                     # one (mode, variant) per process: a fresh interpreter for the second pred
        if mode != "off":
            unpinned_box.warm_cache(mode=mode)
        inp = unpinned_box.input_json(f"unc_{mode}", seeds=(1,)); out = os.path.join(unpinned_box.root, f"out_unc_{mode}")
        rc, err, _ = _run(["pred", "--variant", "p2", "--mode", mode, "--json_path", inp, "--output_dir", out], capsys)
        assert rc == 0 and want in err and "predictions=" in err, (mode, err)
        assert stack._REPORT["params"]["pinned"] is False and stack._REPORT["params"]["sha256"] == rep["params"]["sha256"]


def test_check_hashes_afresh_and_pred_reads_the_memo(unpinned_box, capsys, monkeypatch):
    """The `check` verb digests with refresh=True (the file hashed afresh, its memo entry rewritten); `pred` / `warm` digest with refresh=False
    (an entry for the file's (realpath, size, mtime_ns, inode) serves the digest and the WEIGHTS line names the cached time); the memo is
    <AF3_JAX_CACHE_ROOT>/weights_digests.json; the digest still decides (the stat fields select an entry, never decide which weights these are)."""
    from af3_jax_opt import digest_memo, variants
    calls = []
    real_digest = digest_memo.digest
    monkeypatch.setattr(digest_memo, "digest", lambda path, memo, refresh=False, hasher=digest_memo.sha256_file: calls.append((refresh, memo)) or real_digest(path, memo, refresh=refresh, hasher=hasher))
    rc, err, _ = _run(["check", "--variant", "p2", "--mode", "off"], capsys)
    assert rc == 0 and calls == [(True, unpinned_box.cache_root)] and "(cached digest" not in err, (calls, err)
    assert os.path.isfile(os.path.join(unpinned_box.cache_root, digest_memo.MEMO_NAME))
    stack._REPORT = None; variants._DIGESTS.clear(); calls.clear()                 # a later process: pred reads the memo entry check wrote
    inp = unpinned_box.input_json("memo_off", seeds=(1,)); out = os.path.join(unpinned_box.root, "out_memo_off")
    rc, err, _ = _run(["pred", "--variant", "p2", "--mode", "off", "--json_path", inp, "--output_dir", out], capsys)
    assert rc == 0 and calls and calls[0] == (False, unpinned_box.cache_root), (calls, err)
    assert re.search(r"WEIGHTS weights sha256=[0-9a-f]{12} NOT PINNED — the kit\'s speed and output statements hold for the pinned weights only \(cached digest \S+\)", err), err
    w = os.path.join(unpinned_box.params_root, "p2", variants.PARAMS_FILE)
    same_size_other_bytes = b"\x28\xb5\x2f\xfd" + b"q2"                            # the digest decides: same size, other bytes -> another digest, hashed (a new stat key)
    with open(w, "wb") as f:
        f.write(same_size_other_bytes)
    variants._DIGESTS.clear()
    pc = variants.check_params("p2")
    assert pc["sha256"] == hashlib.sha256(same_size_other_bytes).hexdigest() and pc["pinned"] is False and pc["digest_cached_utc"] is None


def test_a_memo_dir_the_process_cannot_use_is_named_and_the_digest_computed_afresh(unpinned_box, capsys, monkeypatch, tmp_path):
    """A cache root the process cannot use at all (its parent is a file, not a directory — NotADirectoryError; the same path is unusable
    for any user, root included): `check` still digests the weights afresh, prints the WEIGHTS verdict and a named
    `WEIGHTS memo skipped: <path> is not available for caching this process; ...` line, and proceeds rc 0 — never a refusal. K14: no
    exception class name, no "Error:"/"Errno" text, no "unwritable"/"not writable" — a log scanner that greps stderr for generic error
    markers (the (Error|Exception): shape) must not match this line."""
    from af3_jax_opt import digest_memo, variants
    blocker = tmp_path / "not_a_dir"; blocker.write_text("x")                     # a regular file where the cache root's parent should be: unusable for any user, root included
    monkeypatch.setenv("AF3_JAX_CACHE_ROOT", str(blocker / "cache"))
    variants._DIGESTS.clear(); stack._REPORT = None
    rc, err, _ = _run(["check", "--variant", "p2", "--mode", "off"], capsys)
    real = stack.sha256_file(os.path.join(unpinned_box.params_root, "p2", variants.PARAMS_FILE))
    assert rc == 0 and f"WEIGHTS weights sha256={real[:12]} NOT PINNED" in err, err
    assert re.search(r"WEIGHTS memo skipped: \S+/cache/%s is not available for caching this process; every file above was digested in full, "
                      r"just not memoised" % re.escape(digest_memo.MEMO_NAME), err), err
    assert "NOT ACTIVE" not in err and "Traceback" not in err
    for bad in ("Error:", "Exception:", "Errno", "unwritable", "not writable", "not written"):
        assert bad not in err, (bad, err)                                        # K14: none of the exception-shaped substrings survive into stderr


def test_a_readonly_memo_mount_is_named_the_same_way_as_a_missing_one(unpinned_box, capsys, monkeypatch):
    """A DIFFERENT OSError subtype — PermissionError('Read-only file system'), the exact shape of a real read-only /jitcache mount (the
    original K14 report) — produces the identical wording as the NotADirectoryError case above: the message names the situation ("not
    available for caching"), never the specific OS reason. Both failure modes (a read-only mount, and a box that mounts no /jitcache
    at all) must be indistinguishable to a log scan — this test locks that down directly, by patching the memo
    write to raise each OSError subtype in turn rather than depending on real filesystem permissions."""
    from af3_jax_opt import digest_memo, variants
    for exc in (PermissionError("[Errno 30] Read-only file system: '/jitcache/af3_jax'"),
                FileNotFoundError("[Errno 2] No such file or directory: '/jitcache/af3_jax'")):
        def _raise(*_a, _exc=exc, **_kw):
            raise _exc
        monkeypatch.setattr(digest_memo, "digest", _raise)
        variants._DIGESTS.clear(); stack._REPORT = None
        rc, err, _ = _run(["check", "--variant", "p2", "--mode", "off"], capsys)
        assert rc == 0
        assert "WEIGHTS memo skipped:" in err and "is not available for caching this process" in err, (exc, err)
        for bad in ("Error:", "Exception:", "Errno", "Read-only file system", "No such file or directory", "unwritable", "not writable"):
            assert bad not in err, (exc, bad, err)


def test_pinned_weights_line(box, capsys):
    """The pinned digest prints `WEIGHTS weights=p2 sha256=<12> (pinned)` on check and pred."""
    pinned = stack.pins()["variants"]["p2"]["converted"]["sha256"]
    want = f"[af3-jax-opt] WEIGHTS weights=p2 sha256={pinned[:12]} (pinned)"
    rc, err, _ = _run(["check", "--variant", "p2", "--mode", "off"], capsys)
    assert rc == 0 and want in err, err
    stack._REPORT = None
    inp = box.input_json("cert", seeds=(1,)); out = os.path.join(box.root, "out_cert")
    rc, err, _ = _run(["pred", "--variant", "p2", "--mode", "off", "--json_path", inp, "--output_dir", out], capsys)
    assert rc == 0 and want in err and stack.check("off", "p2")["params"]["pinned"] is True


def test_missing_weights_refuse_by_name(box, capsys, tmp_path):
    """No parameters file = NOT ACTIVE by name, rc 3 — under the variant's directory and under a caller's --model_dir alike."""
    from af3_jax_opt import variants
    os.remove(os.path.join(box.params_root, "p2", variants.PARAMS_FILE))
    rc, err, _ = _run(["check", "--variant", "p2", "--mode", "off"], capsys)
    assert rc == 3 and "NOT ACTIVE" in err and f"no {variants.PARAMS_FILE} under {box.params_root}/p2" in err, err
    empty = tmp_path / "myweights"; empty.mkdir()
    stack._REPORT = None
    rc, err, _ = _run(["check", "--variant", "p2", "--mode", "exact", "--model_dir", str(empty)], capsys)
    assert rc == 3 and "NOT ACTIVE" in err and f"no {variants.PARAMS_FILE} under --model_dir {empty}" in err, err
    stack._REPORT = None
    inp = box.input_json("miss", seeds=(1,))
    rc, err, _ = _run(["pred", "--variant", "p2", "--mode", "off", f"--model_dir={empty}", "--json_path", inp, "--output_dir", str(tmp_path / "o")], capsys)
    assert rc == 3 and "NOT ACTIVE" in err and not os.path.exists(box.record), err          # nothing launched


def test_callers_model_dir_is_always_accepted(box, capsys, tmp_path):
    """A weights directory of the caller's own replaces <AF3_JAX_PARAMS_ROOT>/<variant> on the model command (both spellings; `--of3_weights`
    stays refused on argv): digested — here NOT PINNED — and run; the activation records the directory, digest and verdict."""
    from af3_jax_opt import variants
    mine = tmp_path / "mine"; mine.mkdir()
    (mine / variants.PARAMS_FILE).write_bytes(b"\x28\xb5\x2f\xfd" + b"my own weights")
    rc, err, _ = _run(["check", "--variant", "p2", "--mode", "off", "--model_dir", str(mine)], capsys)
    assert rc == 0 and "WEIGHTS weights sha256=" in err and "NOT PINNED" in err, err
    rep = stack.check("off", "p2", model_dir=str(mine))
    assert rep["active"] and rep["params"]["dir"] == str(mine) and rep["params"]["source"] == "model_dir" and rep["params"]["pinned"] is False
    stack._REPORT = None
    inp = box.input_json("mdir", seeds=(1,)); out = str(tmp_path / "out_mdir")
    rc, err, _ = _run(["pred", "--variant", "p2", "--mode", "off", "--json_path", inp, "--output_dir", out, f"--model_dir={mine}"], capsys)
    assert rc == 0 and "NOT PINNED" in err, err
    rec = box.stub_record()
    assert rec["flags"].count(f"--model_dir={mine}") == 1 and not [t for t in rec["flags"] if t.startswith("--model_dir=") and t != f"--model_dir={mine}"]
    assert "--of3_weights" in rec["flags"]
    assert stack._REPORT["params"]["dir"] == str(mine) and stack._REPORT["params"]["source"] == "model_dir" and stack._REPORT["params"]["pinned"] is False
    stack._REPORT = None
    rc, err, _ = _run(["pred", "--variant", "p2", "--mode", "off", "--json_path", inp, "--output_dir", out, "--of3_weights"], capsys)
    assert rc != 2, err                                                                          # the caller's --of3_weights rides last like any stock flag (no wrapper refusal)


@pytest.mark.parametrize("mode", ["off", "exact"])                          # the stock route AND a kit route: one (mode, variant) per interpreter (the late-activation rule)
def test_pred_prints_one_phase_line_per_item_on_every_route(box, capsys, mode):
    """The per-item PHASE timing line: one per run_inference call (seed) on the stock route and on a kit route alike, at the fork's own
    per-seed timer (total_s = its seconds), the three model phases NA with one PHASE-NOTE each per pass (one jit = one XLA program), lm_s=-;
    the DONE line, the prediction count and the per-item PHASE lines are what they were."""
    from af3_jax_opt import report
    if mode != "off":
        box.warm_cache(mode=mode)
    inp = box.input_json("a", seeds=(1, 2))
    out = os.path.join(box.root, "out_phase_" + mode)
    rc, err, _ = _run(["pred", "--variant", "p2", "--mode", mode, "--json_path", inp, "--output_dir", out], capsys)
    assert rc == 0, err
    phase = [ln for ln in err.splitlines() if ln.startswith("[af3-jax-opt] PHASE ")]
    assert phase == [f"[af3-jax-opt] PHASE item=a seed={s} lm_s=- trunk_s=NA sampler_s=NA conf_s=NA total_s=1.23" for s in (1, 2)], (mode, phase)   # item = the fold input's name (the plan item), seed its own token: the reducer's grammar
    notes = [ln for ln in err.splitlines() if ln.startswith("[af3-jax-opt] PHASE-NOTE ")]
    assert [ln.split()[2] for ln in notes] == list(report.NOT_SEPARABLE) and all(" not separable: " in ln for ln in notes), (mode, notes)
    assert err.index("PHASE-NOTE") < err.index("[af3-jax-opt] PHASE item=")                                 # named before the first item
    assert re.findall(r"\] PHASE item=\S+ seed=(\d+)", err) == ["1", "2"] and "[af3-jax-opt] DONE status=ok" in err


def test_phase_line_grammar():
    from af3_jax_opt import report
    assert report.phase_line(7, 83.456, job="2PV7 dimer") == "[af3-jax-opt] PHASE item=2PV7_dimer seed=7 lm_s=- trunk_s=NA sampler_s=NA conf_s=NA total_s=83.46"
    assert report.phase_line(7, 83.456).startswith("[af3-jax-opt] PHASE item=unnamed seed=7 ") and report.phase_line(7, 83.456).endswith("total_s=83.46") and report.PHASE_FIELDS == ("lm_s", "trunk_s", "sampler_s", "conf_s")
    assert cli.JOB_RX.search("\nRunning fold job 2PV7...\n").group(1) == "2PV7" and not cli.JOB_RX.search("Running fold job 2PV7 took 3 s")


def test_off_with_the_forks_own_cache_and_buckets_stated(box, capsys):
    """The fork as shipped, stated: its own cache path and its own --buckets list verbatim ride last and win (absl order) — no EMPTY
    class beside them, the process in the repo dir, ACTIVE says autotune=caller_cache_dir, BUCKETS says source=caller."""
    from af3_jax_opt import settings
    inp = box.input_json("dflt", seeds=(1,))
    out = os.path.join(box.root, "out_default")
    fork_cache = os.path.join(box.root, "tmp_alphafold_cache")                                                          # stands in for the fork's /tmp/alphafold_cache
    fork_buckets = "--buckets=" + ",".join(str(b) for b in settings.upstream_buckets())
    rc, err, _ = _run(["pred", "--variant", "p2", "--mode", "off", "--json_path", inp, "--output_dir", out, f"--cache_dir={fork_cache}", fork_buckets], capsys)
    assert rc == 0, err
    rec = box.stub_record()
    assert [f for f in rec["flags"] if f.startswith("--cache_dir")] == [f"--cache_dir={fork_cache}"] and rec["cwd"] == box.repo and rec["flags"][-1] == fork_buckets
    assert f"cache={fork_cache}" in err and "autotune=caller_cache_dir" in err and "preset=" not in err
    assert f"[af3-jax-opt] BUCKETS buckets={fork_buckets.split('=', 1)[1]} source=caller occurrences=2" in err     # the mode's flag, then the caller's: theirs wins


def test_peak_item_lines_grammar():
    """The cross-engine JAX grammar: ONE `PEAK item=<job> inuse_gib=<f>` when the pass had one input (GiB = 2^30 from the record's
    GB = 1e9 value); several inputs → ONE named PEAK-NOTE and no per-item line; no value → nothing."""
    from af3_jax_opt import report
    assert report.peak_item_lines({"status": "ok", "peak_alloc_gb": 42.949672960}, ["2PV7"]) == ["[af3-jax-opt] PEAK item=2PV7 inuse_gib=40.00"]
    many = report.peak_item_lines({"status": "ok", "peak_alloc_gb": 42.949672960}, ["a", "b"])
    assert len(many) == 1 and many[0].startswith("[af3-jax-opt] PEAK-NOTE scope=pass items=2 inuse_gib=40.00 per_item=unavailable(")
    assert report.peak_item_lines({"status": "no_backend_record"}, ["a"]) == []


import shutil  # noqa: E402  (K.39 route test)


def test_tokamax_line_names_the_class_table_on_every_route(box, capsys):
    """ONE `[af3-jax-opt] TOKAMAX ...` line per pass (K.39): warm on a cold class -> state=written autotune=saved (the model process wrote upstream's
    table); pred reading that class -> state=loaded autotune=loaded, table bytes untouched; a stack whose tokamax.autotune raises (the pinned one:
    STUB_TOKAMAX=unavailable) -> state=absent autotune=unavailable:RuntimeError, named, the pass still ok; off with the EMPTY class -> table=none."""
    from af3_jax_opt import det, warm
    res = warm.warm("p2", mode="exact")
    err = capsys.readouterr().err
    tk = os.path.join(box.cache_dir("exact"), "tokamax_autotune.json")
    assert res["status"] == "PASS" and res["tokamax"]["state"] == "written" and res["tokamax"]["autotune"] == "saved" and res["tokamax"]["entries"] == 1 and os.path.isfile(tk)
    assert f"[af3-jax-opt] TOKAMAX table={tk} state=written entries=1 sha256={det.tokamax_table(box.cache_dir('exact'))['sha256'][:16]} policy=heuristics autotune=saved" in err
    sha = det.tokamax_table(box.cache_dir("exact"))["sha256"]
    stack._REPORT = None; stack._LAUNCHED.clear()
    rc, err, _ = _run(["pred", "--variant", "p2", "--mode", "exact", "--json_path", box.input_json("a", seeds=(1,)), "--output_dir", os.path.join(box.root, "o1")], capsys)
    assert rc == 0 and f"[af3-jax-opt] TOKAMAX table={tk} state=loaded entries=1 sha256={sha[:16]} policy=heuristics autotune=loaded" in err
    assert f"state=loaded entries=1 sha256={sha[:16]}" in err.replace("entries=0", "entries=1") and det.tokamax_table(box.cache_dir("exact"))["sha256"] == sha   # the TOKAMAX line names the table the class holds
    shutil.rmtree(box.cache_dir("exact"))                                    # a cold class on a stack whose tokamax.autotune raises: absent, named, not a failure
    stack._REPORT = None; stack._LAUNCHED.clear()
    os.environ["STUB_TOKAMAX"] = "unavailable"
    try:
        rc, err, _ = _run(["pred", "--variant", "p2", "--mode", "exact", "--json_path", box.input_json("b", seeds=(1,)), "--output_dir", os.path.join(box.root, "o2"), "--allow-partial"], capsys)
    finally:
        del os.environ["STUB_TOKAMAX"]
    assert rc == 0 and f"[af3-jax-opt] TOKAMAX table={tk} state=absent entries=na sha256=none policy=heuristics autotune=unavailable:RuntimeError" in err
    stack._REPORT = None; stack._LAUNCHED.clear()
    rc, err, _ = _run(["pred", "--variant", "p2", "--mode", "off", "--json_path", box.input_json("c", seeds=(1,)), "--output_dir", os.path.join(box.root, "o3")], capsys)
    assert rc == 0 and "[af3-jax-opt] TOKAMAX table=none state=absent entries=na sha256=none policy=heuristics autotune=none" in err   # `--cache_dir=`: the fork keeps no table path


def test_warm_json_path_warms_that_input_not_the_kits_example(box, capsys):
    """0.3.27 (first-run rule): `warm --json_path X` hands the model process --json_path=<abs X> and NO --input_dir, so the class holds the
    executables `pred --json_path X` will load (before, --json_path fell through behind the default --input_dir, which the script reads first —
    the kit's example was warmed and pred compiled X itself); both flags together are refused (rc 2), as pred refuses them."""
    inp = box.input_json("wj", seeds=(1, 2))
    rc, err, _ = _run(["warm", "--variant", "p2", "--mode", "fast", "--json_path", inp], capsys)
    cmd = [ln for ln in err.splitlines() if ln.startswith("[af3-jax-opt] COMMAND ")][-1]
    assert rc == 0 and f"--json_path={os.path.abspath(inp)}" in cmd and "--input_dir=" not in cmd, cmd
    assert "WARM PASS" in err and " predictions=10 " in err, err[-1500:]                       # 2 seeds x 5 samples of THAT input (the kit's example has 1 seed)
    rc2, err2, _ = _run(["warm", "--variant", "p2", "--mode", "fast", "--json_path", inp, "--input_dir", os.path.dirname(inp)], capsys)
    assert rc2 == 2 and "at most one of --json_path / --input_dir" in err2


def test_pair_stack_switches_carry_the_modes_tier_word():
    """0.3.32: the pair stack's two provider bindings (TRIMUL_CD: serve.triangle_multiplication, TRIATT_XLA: serve.attention) are named by the MODE'S
    tier word — the switches carry it: ``1`` (= the modules' default word fast) under fast and in big's region fast (the fast line's own
    executables), ``big`` in big's region reach; no card, size or row word anywhere in the kit (the provider's cells decide per card)."""
    from af3_jax_opt import modes
    fast = modes.lever_env("fast", modes.levers_of("fast"))
    assert fast["AF3_JAX_TRIMUL_CD"] == "fast" and fast["AF3_JAX_TRIATT_XLA"] == "fast"   # fast names the tier word fast
    big = modes.resolve(modes.BIG, "/c", region="fast")                          # region fast IS the fast line (same executables, cache class __fast): fast's word
    assert big["env"]["AF3_JAX_TRIMUL_CD"] == "fast" and big["env"]["AF3_JAX_TRIATT_XLA"] == "fast", big["env"]   # big at or below N* runs the fast line: its executables, its cache class, its word
    reach = modes.resolve(modes.BIG, "/c", region="reach", size={"n_padded": 3072, "n_gpu": 1})   # region reach: TRIATT_XLA kept (big word); the triangle-multiplication site is TRIMUL_CHUNK's
    assert reach["env"].get("AF3_JAX_TRIATT_XLA") == "big" and "AF3_JAX_TRIMUL_CD" not in reach["env"], reach["env"]
    src_cd = open(os.path.join(os.path.dirname(modes.__file__), "inprocess", "trimul_cd.py")).read()
    src_tx = open(os.path.join(os.path.dirname(modes.__file__), "inprocess", "triatt_xla.py")).read()
    for src in (src_cd, src_tx):
        assert "CC_FLOOR" not in src and "ROW_WHEN_UNCOVERED" not in src and "compute_capability()" not in src
