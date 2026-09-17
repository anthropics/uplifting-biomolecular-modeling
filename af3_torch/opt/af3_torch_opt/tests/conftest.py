"""Stubs for the package tests (CPU, no GPU, no torch, no jax): two fake interpreters standing in for the image's venvs, a parameters dir
with an empty checkpoint file, and a clean environment. The fake interpreter is a shell script that dispatches on the script it is
given — featurise.py / forward.py / postprocess.py write the reports and files the package reads (the contract of cli.py), the
check_pins query answers with the pinned versions — and records every argv it saw under STUB_LOG."""
from __future__ import annotations

import json
import os
import stat
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))          # the tree root: af3_torch/
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))               # opt/ on the path: `import af3_torch_opt` without an install

STUB = r'''#!/usr/bin/env python3
import json, os, sys
log = os.environ.get("STUB_LOG")
if log:
    with open(log, "a") as f: f.write(json.dumps(sys.argv) + "\n")
args = sys.argv[1:]
if args and os.path.basename(args[0]) == "stock_launch.py":   # the stock route's launcher script: emulate run_alphafold.py's own layout under --output_dir
    out = args[args.index("--output_dir") + 1]; ins = [args[i + 1] for i, a in enumerate(args) if a == "--json_path"][-1:]   # the CLI's rule: --json_path is ONE string (absl keeps the last)
    if "--input_dir" in args: ins += sorted(os.path.join(args[args.index("--input_dir") + 1], f) for f in os.listdir(args[args.index("--input_dir") + 1]) if f.endswith(".json"))
    for p in ins:
        fi = json.load(open(p)); name = fi["name"].lower(); d = os.path.join(out, name); os.makedirs(d, exist_ok=True)
        print(f"Processing fold input {fi['name']}")                                        # run_alphafold.py's own lines (its format strings; the wrapper tees the seed clock)
        print(f"Featurising data for seeds {tuple(fi.get('modelSeeds') or [1])} took  0.50 seconds.")
        for sd in fi.get("modelSeeds") or [1]:
            print(f"Running model inference for seed {sd}...")
            print(f"Running model inference for seed {sd} took  {12.5 + sd:.2f} seconds.")
            print(f"Extracting output structures (one per sample) for seed {sd} took  0.25 seconds.")
            for k in range(5):
                sdir = os.path.join(d, f"seed-{sd}_sample-{k}"); os.makedirs(sdir, exist_ok=True)
                for suf in ("model.cif", "confidences.json", "summary_confidences.json"): open(os.path.join(sdir, suf), "w").write(f"{name} {sd} {k} {suf}\n")
        for suf in ("model.cif", "confidences.json", "summary_confidences.json"): open(os.path.join(d, f"{name}_{suf}"), "w").write(f"{name} best {suf}\n")
        open(os.path.join(d, "ranking_scores.csv"), "w").write("seed,sample,ranking_score\n"); open(os.path.join(d, f"{name}_data.json"), "w").write(json.dumps(fi))
    print("Done processing %d fold inputs." % len(ins)); sys.exit(0)
if args[:1] == ["-c"]:                                   # check_pins' version query: answer with the pinned versions
    pins = json.load(open(os.path.join(os.environ["STUB_HOME"], "stock", "PINS.json")))
    which = os.environ.get("STUB_WHICH", "torch_python")
    print(json.dumps({"python": "3.12.0", **pins["check_packages"][which]})); sys.exit(0)
script = os.path.basename(args[0]) if args else ""
def items(flag="--item"):
    out = []
    for i, a in enumerate(args):
        if a == flag: out.append(args[i + 1].split("=", 3))
    return out
fail = set((os.environ.get("STUB_FAIL") or "").split(","))
sys.path.insert(0, os.path.join(os.environ["STUB_HOME"], "opt", "af3_torch_opt")); import pipeline_stream as PS   # the streamed chain's hand-off protocol (package levers prefetch / write_behind): the stub plays each process's part of it
def report(path, rep):
    os.makedirs(os.path.dirname(path), exist_ok=True); json.dump(rep, open(path, "w"), indent=1)
if script == "featurise.py":
    rep = {"items": [], "ok": True}
    run_inference = args[args.index("--run_inference") + 1] != "0" if "--run_inference" in args else True
    stream = "--stream" in args and args[args.index("--stream") + 1] == "1" and run_inference       # the streamed chain: SEEDS line first, a hand-off marker per seed, ok / failed per item
    rep["stream"] = int(stream)
    if stream and os.environ.get("STUB_NO_SEEDS") is None: print(PS.seeds_line({name: json.load(open(src)).get("modelSeeds") or [1] for name, src, work in items()}), flush=True)   # STUB_NO_SEEDS: a featuriser that never announces (the wrapper goes on sequentially from the report)
    for name, src, work in items():
        if name in fail:
            rep["items"].append({"name": name, "ok": False, "error": "stub: featurise refused"}); rep["ok"] = False
            if stream: PS.touch(os.path.join(work, PS.ITEM_FAILED))
            continue
        seeds = json.load(open(src)).get("modelSeeds") or [1]
        if not run_inference:                                                   # --run_inference 0: the data json only, in the item's dir
            os.makedirs(work, exist_ok=True); open(os.path.join(work, f"{name}_data.json"), "w").write(open(src).read())
            rep["items"].append({"name": name, "ok": True, "seeds": seeds, "pipeline_s": 0.1}); continue
        for sd in seeds:
            d = os.path.join(work, f"seed-{sd}"); os.makedirs(d, exist_ok=True)
            open(os.path.join(d, "batch.npz"), "wb").write(b"NPZ" + name.encode()); open(os.path.join(d, "batch.pkl"), "wb").write(b"PKL")
            if stream: PS.touch(os.path.join(d, PS.BATCH_READY))
        if stream: PS.touch(os.path.join(work, PS.ITEM_OK))
        rep["items"].append({"name": name, "ok": True, "seeds": seeds, "n_tokens": 100, "bucket": 256, "featurise_s": 0.5, "msa_rows": [1, 100]})
    report(args[args.index("--report") + 1], rep); sys.exit(0 if rep["ok"] else 1)
if script == "forward.py":
    levers = args[args.index("--levers") + 1] if "--levers" in args else ""
    routes = [n for n in (args[args.index("--route") + 1] if "--route" in args else "").split(",") if n]
    core = args[args.index("--opt-core") + 1] if "--opt-core" in args else None                    # forward.py's route gate, emulated: the verdict record per routed kernel
    kernel_routes = {n: {"ok": os.environ.get("STUB_ROUTE_FAIL") != n, "reason": ("stub: bytes differ" if os.environ.get("STUB_ROUTE_FAIL") == n else None), "version": "stub",
                         "core_copy": f"{core}/opt_core/kernels/{n}", "exports": {"FPF_TRIMUL_V4_CELLS": os.environ.get("FPF_TRIMUL_V4_CELLS")} if n == "fpf_trimul_v4" else {}} for n in routes}
    if any(not v["ok"] for v in kernel_routes.values()):
        report(args[args.index("--report") + 1], {"kernel_routes": kernel_routes, "error": "kernel route refused: " + "; ".join(f"{n}: {v['reason']}" for n, v in kernel_routes.items() if not v["ok"]), "items": [], "ok": False}); sys.exit(2)
    applied = os.environ["STUB_LEVERS_APPLIED"] if os.environ.get("STUB_LEVERS_APPLIED") is not None else levers
    pkg = [l for l in (args[args.index("--package-levers") + 1] if "--package-levers" in args else "").split(",") if l]
    rep = {"graph_resets": 0, "graph_captures": 0, "levers_applied": [l for l in applied.split(",") if l], "dtk": args[args.index("--dtk") + 1] == "1", "build_s": 1.0, "device": "stub", "items": [], "ok": True,
           "package_levers": pkg, "package_levers_applied": list(pkg), "padding": args[args.index("--padding") + 1] if "--padding" in args else "af3_buckets",
           "template_dedupe": {"calls": 0, "slots": 0, "evaluated": 0}, "canonical_noise": {"trajectories": 0, "model_len": None, "canonical_len": None},
           "prefetch": {"streamed": int("--stream-root" in args), "items": 0, "waited": 0, "wait_s": 0.0, "first_wait_s": None, "withdrawn": 0}, "write_behind": {"streamed": int("--stream-root" in args), "published": 0}, "autotune_cache": {"knob": 1, "hits": 0, "benched": 0, "env": os.environ.get("TRITON_CACHE_AUTOTUNING") or "unset"},
           "env_prefixes_present": sorted(k for k in os.environ if k.startswith("AF3_TORCH_OPT"))}
    stream_root = args[args.index("--stream-root") + 1] if "--stream-root" in args else None      # the streamed chain: wait for each batch's hand-off (prefetch), hand each result over (write_behind)
    kernel_levers = [l for l in rep["levers_applied"] if l in ("trimul", "triattn", "transition", "apb", "resid_fold", "attn_epi", "tmpl_trimul", "pwa_lnl", "opm", "pwa_msa", "atom_window", "token_agg", "atom_rows")]   # atom_window: its census rides the kit's (served:calls / fallback:<word>)
    census = {l: {"served:x": 0} for l in kernel_levers} if kernel_levers or os.environ.get("STUB_STOCK_KERNELS") else None   # the kit's census shape
    if census is not None:
        census.update({"on": kernel_levers, "dead": {}, "mask_terms": 0, "arch": {}})
    rep["kernels_enabled"] = census is not None
    fb = os.environ.get("STUB_FALLBACK", "")                         # "lever:reason" — inject 3 per-call fallbacks on the FIRST item; "lever:DEAD" a kernel death
    buckets = dict(kv.split("=", 1) for kv in os.environ.get("STUB_BUCKETS", "").split(",") if kv)   # "NAME=n,..." — an item's padded token count (default 256)
    declared = dict(kv.split("=", 1) for i, a in enumerate(args) if a == "--templates-declared" for kv in [args[i + 1]])   # forward.py's template record, emulated: declared from the argv,
    live = dict(kv.split("=", 1) for kv in os.environ.get("STUB_TEMPLATES_LIVE", "").split(",") if kv)                     # live slots = STUB_TEMPLATES_LIVE "NAME=n,..." (default: every declared template featurised live)
    for name, seed, batch, out in items():
        seed = int(seed)
        if stream_root and "prefetch" in pkg:
            w = PS.await_batch(batch, stream_root)
            if not w["ready"]: rep["prefetch"]["withdrawn"] += 1; print(f"[forward] {name} seed={seed} withdrawn reason={w.get('reason')}", flush=True); continue
            rep["prefetch"]["items"] += 1; rep["prefetch"]["wait_s"] += w["wait_s"]
        if ("forward:" + name) in fail or (f"forward:{name}:{seed}") in fail:
            rep["items"].append({"name": name, "seed": seed, "ok": False, "error": os.environ.get("STUB_FAIL_ERROR", "stub: forward failed")}); rep["ok"] = False; continue   # STUB_FAIL_ERROR: the failed item's error words
        t_decl = int(declared.get(name, 0)); t_live = int(live.get(name, t_decl))
        os.makedirs(os.path.dirname(out), exist_ok=True); open(out, "wb").write(b"RESULT" + name.encode())
        if stream_root and "write_behind" in pkg: PS.touch(os.path.join(os.path.dirname(out), PS.RESULT_READY)); rep["write_behind"]["published"] += 1
        it = {"name": name, "seed": seed, "ok": True, "forward_s": 2.0 + seed, "peak_mem_gb": 1.5, "peak_reserved_gb": 2.5, "n_tokens": 100, "bucket": int(buckets.get(name, 256)), "fallbacks": {}, "dead": [], "rng": "seeded_per_item", "templates_declared": t_decl, "templates_live": t_live, "templates_slots": 4, "templates_slots_live": t_live, "phase_s": {"lm": None, "trunk": 1.0, "sampler": 0.5 + seed, "conf": 0.25}, "graph_reset": int(seed > 1), "pool_reset": int(seed > 1), "graph_capture": 1}
        if census is not None:
            for l in kernel_levers: census[l]["served:x"] += 1
            if fb and not rep["items"]:
                lever, reason = fb.split(":", 1)
                if reason == "DEAD":
                    census["dead"][lever] = "RuntimeError('stub kernel error')"; census.setdefault(lever, {})["kernel_error"] = 1; it["fallbacks"] = {lever: {"kernel_error": 1}}
                else:
                    census.setdefault(lever, {})["fallback:" + reason] = 3; it["fallbacks"] = {lever: {"fallback:" + reason: 3}}
            it["dead"] = sorted(census["dead"])
        rep["items"].append(it)
    rep["census"] = census
    rep["fallback_events"] = {l: {k: v for k, v in c.items() if k.startswith("fallback:") or k == "kernel_error"} for l, c in (census or {}).items() if isinstance(c, dict) and l not in ("dead", "mask_terms", "arch") and any(k.startswith("fallback:") or k == "kernel_error" for k in c)}
    rep["dead"] = dict((census or {}).get("dead") or {})
    big_levers = [n for n in (args[args.index("--big") + 1] if "--big" in args else "").split(",") if n]
    alloc = args[args.index("--alloc") + 1] if "--alloc" in args else ""
    gate = int(args[args.index("--graph-drop-tokens") + 1]) if "--graph-drop-tokens" in args else 0     # big's size-gated graph_drop: per item on the padded token count
    if big_levers or alloc:                                                                       # forward.py's big record, emulated: the facts the LEVER lines read
        for it in rep["items"]:
            if it.get("ok"):
                it["diff_free"] = int("diff_free" in big_levers); it["prev_free"] = int("prev_free" in big_levers)
        ok_items = [it for it in rep["items"] if it.get("ok")]
        rep["big"] = {"levers": big_levers, "alloc": ({"policy": alloc, "conf": "expandable_segments:True", "effective": os.environ.get("STUB_ALLOC_INEFFECTIVE") is None, "source": "stub"} if alloc else None),
                        "diff_free_calls": sum(it["diff_free"] for it in ok_items), "prev_free_calls": sum(it["prev_free"] for it in ok_items), "items_ok": len(ok_items),
                        "items": len(ok_items), "graph_drop_min_tokens": gate or None,
                        "paircond_rows": (int(args[args.index("--paircond-rows") + 1]) if "--paircond-rows" in args and "paircond_chunk" in big_levers else 0),
                        "paircond_items": len(ok_items) if "paircond_chunk" in big_levers else 0, "paircond_blocks": 2 * len(ok_items) if "paircond_chunk" in big_levers else 0,
                        "graph_drop_items": sum(1 for it in ok_items if gate and (it.get("bucket") or 0) >= gate) if "graph_drop" in big_levers else 0}
    used = {"fpf_trimul_v4": "trimul" in kernel_levers or "tmpl_trimul" in kernel_levers, "flash_triattn": "triattn" in kernel_levers, "fpf_triatt_k2b": "triattn" in kernel_levers, "fpf_triatt_epi": "attn_epi" in kernel_levers, "dtk_kernels": "--dtk" in args and args[args.index("--dtk") + 1] == "1"}      # a routed module is imported only by its lever's first call (dtk_kernels: the DTK swap; fpf_triatt_k2b: the K2B core, on the capabilities af3_arch_cells.json names — the stub's device is one)
    used["apb_attn"] = used["dtk_kernels"] and "sbatch" in levers.split(",")                                       # apb_attn: the sample-batched DTK step's attention (lever sbatch on the DTK route)
    used["atom_window"] = "atom_window" in levers.split(",")                                                        # atom_window: the atom transformers' window kernels (lever atom_window)
    for n, v in kernel_routes.items(): v["imported_from"] = (v["core_copy"] + ("/__init__.py" if n in ("fpf_trimul_v4", "fpf_triatt_k2b", "fpf_triatt_epi") else ".py")) if used.get(n) else None
    p_recv = int(args[args.index("--n-gpu") + 1]) if "--n-gpu" in args else 1                                     # the axis as RECEIVED on the command line (tests read it back from the report)
    rep["n_gpu_received"] = p_recv; rep["n_gpu"] = int(os.environ.get("STUB_NGPU_REPORT", p_recv)); rep["world"] = int(os.environ.get("STUB_WORLD_REPORT", rep["n_gpu"]))   # STUB_NGPU_REPORT / STUB_WORLD_REPORT: a model process that ran another P than asked
    tag = os.environ.get("ROWPAIR_TAG") or "rowpair"; rep["rowpair_tag"] = os.environ.get("ROWPAIR_TAG")   # the shared core's line tag as this process was handed it (the wrapper exports the kit's under --n-gpu > 1)
    if p_recv > 1:                                                  # the launcher's once-per-launch line, on stderr as the core prints it
        sys.stderr.write("[%s] RANKENV hashseed=0 source=default ranks=%d\n" % (tag, p_recv)); sys.stderr.flush()
    rep["kernel_routes"] = kernel_routes; rep["compiled"] = "compile" in rep["levers_applied"] or None
    f2spec = os.environ.get("STUB_ROWPAIR_F2")                     # "reason@n;...[;served@n]": rank 0's core F2.trimul_rows record under --n-gpu > 1 (declines by reason, kernel launches served)
    if p_recv > 1 and f2spec is not None:
        fb = {k: int(n) for k, n in (e.rsplit("@", 1) for e in f2spec.split(";") if e)}; served = fb.pop("served", 0)
        f2 = {"name": "F2.trimul_rows", "impl": "fpf_trimul_v4", "origin": "core", "state": "on" if served else "skipped", "served": served, "fallback": sum(fb.values()), "fallback_by": fb,
              "errors": {}, "calls": served + sum(fb.values()), "facts": {"min_tokens": 2048}}
        rep["rowpair"] = {"P": p_recv, "align": 32, "trimul": "rowpair_rows", "trimul_rows": f2,
                          "trimul_rows_line": "[%s] LEVER name=F2.trimul_rows state=%s served=%d fallback=%d" % (tag, f2["state"], served, f2["fallback"])}
    fastnn_on = int(args[args.index("--fastnn") + 1]) if "--fastnn" in args else 0                                 # forward.py's fastnn record, emulated: the three switches as the process would run them
    rep["fastnn"] = {k: ("triton" if fastnn_on else "torch") for k in ("layer_norm_implementation", "dot_product_attention_implementation", "gated_linear_unit_implementation")}
    if rep["dtk"]: rep["dtk_swap_s"] = 0.1
    report(args[args.index("--report") + 1], rep); sys.exit(0 if rep["ok"] else 1)
if script == "postprocess.py":
    rep = {"items": [], "ok": True}
    follow = args[args.index("--follow") + 1] if "--follow" in args else None                        # the streamed chain: take each item as the model process hands it over
    rep["write_behind"] = {"streamed": int(bool(follow)), "items": 0, "early": 0, "withdrawn": 0, "wait_s": 0.0}
    for name, work, out in items():
        if follow:
            w = PS.await_results(work, follow)
            if w["state"] != "ready": rep["write_behind"]["withdrawn"] += 1; print(json.dumps({"name": name, "withdrawn": w.get("reason")}), flush=True); continue
            rep["write_behind"]["items"] += 1; rep["write_behind"]["early"] += int(w["early"]); rep["write_behind"]["wait_s"] += w["wait_s"]
        seeds = sorted(int(d.split("-", 1)[1]) for d in os.listdir(work) if d.startswith("seed-") and os.path.isfile(os.path.join(work, d, "result.npz")))
        files = []
        for sd in seeds:
            d = os.path.join(out, f"seed-{sd}_sample-0"); os.makedirs(d, exist_ok=True)
            fn = f"{name}_seed-{sd}_sample-0_model.cif"
            open(os.path.join(d, fn), "w").write(f"data_{name}_{sd}\n"); files.append(f"seed-{sd}_sample-0/{fn}")
        open(os.path.join(out, name + "_model.cif"), "w").write("data_" + name + "\n"); open(os.path.join(out, "ranking_scores.csv"), "w").write("seed,sample,ranking_score\n")
        files += [name + "_model.cif", "ranking_scores.csv"]
        rep["items"].append({"name": name, "ok": True, "seeds": seeds, "files": files, "ranking": [{"seed": sd, "sample": 0, "ranking_score": 0.5} for sd in seeds], "top": {"seed": seeds[0], "sample": 0} if seeds else None})
    report(args[args.index("--report") + 1], rep); sys.exit(0)
sys.exit(0)
'''


def _write_exe(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)



def archive_or_skip():
    """The pinned upstream archive's path under stock/ (stock/PINS.json upstream.archive), or a skip BY NAME when this tree arrived without it
    (`run.sh install` fetches it: stock/fetch_upstream.py) — the archive-reading checks run where it is present; nothing else depends on it."""
    from af3_torch_opt import stack
    arch = os.path.join(stack.home(), "stock", stack.pins()["upstream"]["archive"]["file"])
    if not os.path.isfile(arch):
        pytest.skip(f"stock/{os.path.basename(arch)} is not in this tree (run.sh install fetches it: stock/fetch_upstream.py)")
    return arch

@pytest.fixture
def box(tmp_path, monkeypatch):
    """A fake box: stub torch/JAX interpreters, a fork checkout (run_alphafold.py), a parameters dir with the checkpoint file, a cache root, a clean environment."""
    torch_py, jax_py = tmp_path / "torch_venv" / "python", tmp_path / "jax_venv" / "python"
    torch_py.parent.mkdir(); jax_py.parent.mkdir()
    _write_exe(str(torch_py), STUB); _write_exe(str(jax_py), STUB)
    from af3_torch_opt import stack                                   # the checkpoint name from the pins (one spelling)
    params = tmp_path / "params"; params.mkdir(); (params / stack.checkpoint()).write_bytes(b"")
    stack._WEIGHTS.clear(); stack._WARNED.clear()                     # a fresh process per box: the checkpoint digest memo and the once-per-process UNPINNED line
    for k in list(os.environ):
        if k.startswith("AF3_TORCH"):
            monkeypatch.delenv(k, raising=False)
    stock_py = tmp_path / "stock_venv" / "python"; stock_py.parent.mkdir(); _write_exe(str(stock_py), STUB)
    jax_repo = tmp_path / "alphafold3"; jax_repo.mkdir(); (jax_repo / stack.RUN_ALPHAFOLD).write_text("# the fork's run_alphafold.py (stub)\n")
    monkeypatch.setenv("AF3_TORCH_PY", str(torch_py)); monkeypatch.setenv("AF3_TORCH_JAX_PY", str(jax_py)); monkeypatch.setenv("AF3_TORCH_STOCK_PY", str(stock_py))
    monkeypatch.setenv("AF3_TORCH_JAX_REPO", str(jax_repo))
    monkeypatch.setenv("AF3_TORCH_PARAMS_DIR", str(params)); monkeypatch.setenv("AF3_TORCH_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("STUB_HOME", HOME); monkeypatch.setenv("STUB_LOG", str(tmp_path / "stub.log"))
    monkeypatch.delenv("STUB_FAIL", raising=False)
    return {"tmp": tmp_path, "torch_py": str(torch_py), "jax_py": str(jax_py), "stock_py": str(stock_py), "jax_repo": str(jax_repo), "params": str(params), "log": str(tmp_path / "stub.log"), "home": HOME}


def stub_calls(box):
    try:
        with open(box["log"], encoding="utf-8") as f:
            return [json.loads(ln) for ln in f if ln.strip()]
    except OSError:
        return []
