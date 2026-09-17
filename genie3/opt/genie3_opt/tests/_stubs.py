"""Test fixtures: a temporary tree (the real stock/ pins + archive, the kit directory with a STUB driver in the driver's place), a
pinned checkout extracted from the stock archive, a weights directory, a stub
`nvidia-smi` and a stub `genie3` console script — everything the package gates on, without a GPU, torch or upstream.

The two carried modules (opt/forward/fast_inference/driver/g3fast.py, opt/forward/g3cap/g3cap.py) are libraries the batched capture line
imports; the package only checks that they are PRESENT (g3batch.kit_dirs), so the test tree holds a one-line placeholder for each. The stub batched
capture driver (STUB_G3BATCH, written to <tree>/opt/genie3_opt/g3batch.py; `box()` points the package's chain entry for g3batch.py at it —
the one package file a test tree stands in for, since the modes' driver is package code) does the same for opt/genie3_opt/g3batch.py's
contract (dataset-order batches of B, one capture per distinct batch shape under graph reuse, the hoist's fill per batch, the LEVER / BATCH
lines). The stub `genie3` reproduces the stock writer's layout and the generation_stats/ file.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import stat
import tarfile
import textwrap

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.abspath(os.path.join(HERE, "..", "..", ".."))                 # genie3/
PIN = "d77ae5ac04212ff1e8b29b585859a3244c614804"

STUB_CKPT = b"stub-ckpt"
STUB_CONFIG = b"model: {name: v1}\n"
STUB_DRIVER = "# placeholder for driver/g3fast.py in the test tree: present, never imported or run by the package's CPU tests\n"
STUB_G3CAP = "# placeholder for g3cap/g3cap.py in the test tree: present, never imported or run by the package's CPU tests\n"

STUB_G3BATCH = textwrap.dedent('''\
    #!/usr/bin/env python3
    """STUB of the batched capture driver opt/genie3_opt/g3batch.py for the package's CPU tests: the output contract only (upstream's batch
    semantics at --batch-size B: the designs in dataset order, B per batch; one graph capture per distinct batch shape under --reuse-graphs,
    one per batch without; the hoist's one fill per batch; the LEVER and BATCH lines; the timings keys design.py reads back). Computes nothing."""
    import argparse, json, os, sys, types, yaml
    FAIL = set(filter(None, (os.environ.get("STUB_FAIL") or "").split(",")))   # rc | short | nopatch | forbidden | graph | tf32leak | tf32off | noreadback | hoiststale | unfinished | capacity | trimulrefused | trimuldeclined, comma-joined
    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("--config", required=True); ap.add_argument("--outdir", required=True)
        ap.add_argument("--batch-size", type=int, default=8); ap.add_argument("--cuda-graphs", action="store_true")
        ap.add_argument("--hoist", action="store_true"); ap.add_argument("--reuse-graphs", type=int, default=0)
        ap.add_argument("--pt-chunk", choices=("design",), default=None); ap.add_argument("--compile", action="store_true"); ap.add_argument("--tf32", action="store_true")
        ap.add_argument("--lean-pair", action="store_true"); ap.add_argument("--wide-capture", action="store_true"); ap.add_argument("--alloc", choices=("expandable",), default=None)
        ap.add_argument("--limit", type=int, default=None); ap.add_argument("--seed", type=int, default=None)
        ap.add_argument("--shard-id", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1)   # upstream's dataset shards: the shard's share of every problem's n_sample, named from its offset
        ap.add_argument("--timings", default=None); ap.add_argument("--tag", default="")
        ap.add_argument("--trimul", choices=("stock", "fpf"), default="stock")          # L7: the real driver patches the TriMul class before the capture (genie3_opt.trimul.enable) and prints the KERNELS census line at exit
        a = ap.parse_args()
        req = yaml.safe_load(open(a.config))
        if "rc" in FAIL:
            print("Traceback (most recent call last):"); print("AssertionError: stub failure"); sys.exit(1)
        if "nopatch" not in FAIL:                                        # the real driver applies the kit's patches (g3fast_patches.apply); the stub plants the two markers stack.kit_levers_applied reads
            pair = types.ModuleType("genie3.generation.model.embedder.pair.v1")
            class V1PairFeatureNet: _g3fast_patched = True
            pair.V1PairFeatureNet = V1PairFeatureNet; sys.modules[pair.__name__] = pair
            geo = types.ModuleType("genie3.generation.utils.geo_utils")
            def batched_gather(*x): return x
            batched_gather.__module__ = "g3fast_patches"; geo.batched_gather = batched_gather; sys.modules[geo.__name__] = geo
        from genie3_opt.stack import kit_levers_applied
        markers = kit_levers_applied()
        ds = req["generation"]["dataset"]
        n = int(ds["n_sample"]); seed = req["experiment"].get("seed"); B = a.batch_size
        if ds.get("source") == "unconditional":
            problems = [str(L) for L in range(int(ds["min_length"]), int(ds["max_length"]) + 1, int(ds["length_step"]))]
        else:
            problems = [p.strip() for p in str(ds.get("selections") or "04_pdl1").split(",")]
        per = -(-n // a.num_shards) if a.num_shards > 1 else n                     # upstream's slicing, spelled here as loader.py _apply_shard_to_config spells it: ceil(n / M) per shard from K·ceil(n / M), clipped
        lo = min(a.shard_id * per, n) if a.num_shards > 1 else 0
        hi = min(lo + per, n)
        designs = [(p, f"{p}_{i}") for p in problems for i in range(lo, hi)]      # dataset order: problem-major, then design index; names from the shard's sample_index_offset
        batches = [designs[i:i + B] for i in range(0, len(designs), B)]
        seen, captures, reuses, per_batch = set(), 0, 0, []
        refused = None
        for bt in batches:
            shape = (len(bt), max(len(p) for p, _ in bt))                          # a batch's shape signature: (designs in it, longest problem name) stands in for (B, n_token)
            if "capacity" in FAIL:                                                  # the pre-run capacity guard refuses the first batch by name: nothing of it runs, exit 5
                refused = {"batch": len(bt), "n_token": shape[1], "regime": "graph", "probe_gb": 2.34, "need_gb": 75.9, "suggest_batch": 8}
                print(f"[genie3-opt] CAPACITY batch={len(bt)} n_token={shape[1]} regime=graph probe_gb=2.34 need_gb=75.9 resident_gb=0.7 total_gb=79.2 verdict=refused: set generation.dataset.batch_size <= 8 (or fewer kept graphs, --reuse-graphs k)", file=sys.stderr)
                break
            how = "eager"
            if a.cuda_graphs:
                if a.reuse_graphs > 0 and shape in seen:
                    how = "reuse"; reuses += 1
                else:
                    how = "capture"; captures += 1; seen.add(shape)
            for p, nm in bt:
                d = os.path.join(a.outdir, "pdbs") if ds.get("source") == "unconditional" else os.path.join(a.outdir, p, "pdbs"); os.makedirs(d, exist_ok=True)
                if "short" in FAIL and nm == designs[-1][1]: continue
                open(os.path.join(d, nm + ".pdb"), "w").write(f"REMARK stub seed={seed} B={B} graphs={a.cuda_graphs}\\nATOM {nm}\\nEND\\n")
            per_batch.append({"names": [nm for _, nm in bt], "n_token_pad": shape[1], "n_token_real": [len(p) for p, _ in bt], "wall_s": 1.0, "probe_s": 0.1, "graph": how, "hoist": "fill" if a.hoist else "off"})   # wall_s includes the capacity probe's probe_s
            print(f"[genie3-opt] BATCH {len(per_batch)}/{len(batches)} designs={len(bt)} n_token={shape[1]} graph={how} hoist={'fill' if a.hoist else 'off'} wall_s=1.000")
        ev = {"batch": B, "cuda_graphs": a.cuda_graphs, "hoist": a.hoist, "reuse_graphs": a.reuse_graphs, "pt_chunk": a.pt_chunk, "compile": ("inductor" if a.compile else None), "lean_pair": int(a.lean_pair), "wide": int(a.wide_capture), "alloc": a.alloc, "tf32": a.tf32, "trimul": a.trimul,
              "graph_captures": captures, "graph_reuses": reuses, "cache_hits": reuses if a.reuse_graphs else 0, "cache_misses": captures if a.reuse_graphs else 0,
              "hoist_fills": len(per_batch) if a.hoist else 0, "hoist_anomalies": 1 if "hoiststale" in FAIL else 0, "patches": len(markers) == 2, "batches": len(batches),
              "batches_done": len(per_batch) - (1 if "unfinished" in FAIL else 0), "finished": "unfinished" not in FAIL and refused is None, "policy": "fp32_tf32" if a.tf32 else "genie3_stock_fp32"}
        live_tf32 = (a.tf32 or "tf32leak" in FAIL) and "tf32off" not in FAIL          # the numerics READBACK the package judges (tf32leak: TF32 live on an fp32 line; tf32off: off on the tf32 line)
        rb = {"matmul": "high" if live_tf32 else "highest", "cudnn_tf32": True, "matmul_tf32": live_tf32, "cudnn_benchmark": False}
        T = {"argv": sys.argv[1:], "driver": "g3batch", "tag": a.tag, "seed": seed, "model_setup_s": 2.5, "featurize_s": 0.1, "tf32": live_tf32, "matmul_precision": rb["matmul"],
             "global_numerics_state": {"float32_matmul_precision": rb["matmul"], "cuda.matmul.allow_tf32": live_tf32, "cudnn.allow_tf32": True},
             "patches": len(markers) == 2 and markers, "n_designs": len(designs), "stock_batch": B, "batches": len(batches), "n_token": [b["n_token_pad"] for b in per_batch],
             "per_batch": per_batch, "sampling_loop_s": 1.0 * len(per_batch), "s_per_design_sampling": round(1.0 * len(per_batch) / max(1, len(designs)), 4),
             "write_s": 0.01, "total_s": 3.0, "D59_global_state_checks_passed": len(per_batch), "lever": ev}
        if refused: ev["capacity_refused"] = refused
        from genie3_opt import trimul as KT                                        # the census WORD is the package's (trimul.word / declined): the stub fakes the core census only
        served = 30 * captures if (a.trimul == "fpf" and not FAIL & {"trimulrefused", "trimuldeclined"}) else 0      # 3 eager calls (2 warm-ups + the capture) x 10 modules per captured shape
        if a.trimul == "fpf":
            n_calls = 30 * captures
            if "trimulrefused" in FAIL:                                                 # an UNEXPECTED fallback reason: the core gate refuses
                fb, gate = {"no_cell:cc(8, 0)": n_calls}, {"ok": False, "reason": "unexpected fallback reason(s) no_cell:cc(8, 0) (expected: below_min_tokens)"}
            elif "trimuldeclined" in FAIL:                                              # every call under the kernel's floor: the core gate reads `served 0 of n` as refused; the kit's word says declined
                fb, gate = {"below_min_tokens": n_calls}, {"ok": False, "reason": f"mode fast routed fpf_trimul_v4@4.2.0 but served 0 of {n_calls} calls"}
            else:
                fb, gate = {}, {"ok": True, "reason": None}
            cen = {"mode": "fpf", "strategy": "F2.fpf_trimul_fast", "kernel": "fpf_trimul_v4@4.2.0", "origin": "core", "served": served, "fallback": fb, "errors": {}}
            ev7 = {"mode": "fpf", "census": cen, "gate": gate}
            word = KT.word(ev7)
            T["trimul"] = dict(ev7, route={"kernel": "fpf_trimul_v4", "routed": True}, strategy="F2.fpf_trimul_fast", word=word, declined=KT.declined(ev7))
        else:
            word = KT.word({"mode": "stock"}); T["trimul"] = {"mode": "stock", "census": {"mode": "stock", "served": 0, "fallback": {}, "errors": {}}, "gate": {"ok": True, "reason": None}, "route": None, "strategy": "F2.fpf_trimul_fast", "word": word, "declined": None}
        T["kernels"] = {"route": "g3batch", "trimul": word, "triatt": "n/a-upstream:no-triangle-attention-in-architecture", "cueq": "n/a-upstream:not-imported", "deepspeed": "n/a-upstream:not-imported", "tf32_matmul": int(bool(a.tf32)), "served": served}
        if a.hoist and per_batch: T["hoist"] = True
        if "noreadback" not in FAIL: T["numerics_readback"] = rb; T["numerics_readback_end"] = rb
        else: T.pop("global_numerics_state", None)
        T["lever"] = ev
        if a.pt_chunk: T["pt_chunk"] = {"chunk": a.pt_chunk, "modules": 5, "rows": [100]}
        if a.compile: T["compile"] = {"backend": "inductor", "shapes": 1}
        if a.lean_pair: T["lean_pair"] = {"lt_release": 2, "trimul_lean": 0 if a.trimul == "fpf" else 1}
        if a.wide_capture: T["wide_capture"] = {"graphs": captures}
        if a.alloc: T["alloc"] = {"conf": "expandable_segments:True", "probe": "once-per-shape"}
        if a.cuda_graphs: T["graph_capture_s"] = [0.5] * captures
        if a.reuse_graphs > 0 and a.cuda_graphs: T["graph_cache"] = {"capacity": a.reuse_graphs, "hits": reuses, "misses": captures, "kept": len(seen), "captures": captures}
        if "forbidden" in FAIL: print("[D59] RNG stream bookkeeping diverged (stub)")
        if a.timings: json.dump(T, open(a.timings, "w"), indent=1)
        print(f"[genie3-opt] KERNELS route=g3batch trimul={T['kernels']['trimul']} triatt=n/a-upstream:no-triangle-attention-in-architecture cueq=n/a-upstream:not-imported deepspeed=n/a-upstream:not-imported tf32_matmul={int(bool(live_tf32))}", file=sys.stderr)
        print(f"[genie3-opt] LEVER g3batch batch={B} cuda_graphs={int(a.cuda_graphs)} hoist={int(a.hoist)} reuse_graphs={a.reuse_graphs} pt_chunk={ev['pt_chunk'] or 'stock'} compile={ev['compile'] or 'none'} lean_pair={ev['lean_pair']} wide={ev['wide']} alloc={ev['alloc'] or 'default'} tf32={int(live_tf32)} policy={ev['policy']} "
              f"graph_captures={captures} graph_reuses={reuses} cache_hits={ev['cache_hits']} cache_misses={ev['cache_misses']} hoist_fills={ev['hoist_fills']} hoist_anomalies={ev['hoist_anomalies']} "
              f"patches={int(ev['patches'])} trimul={a.trimul} batches={ev['batches_done']}/{len(batches)} finished={int(ev['finished'])} capacity={'refused' if refused else 'ok'}", file=sys.stderr)
        print(json.dumps({k: v for k, v in T.items() if k not in ("per_batch", "n_token")}))
        sys.exit(5 if refused else 0)
    if __name__ == "__main__":
        main()
    ''')


STUB_GENIE3 = textwrap.dedent('''\
    #!/usr/bin/env python3
    """STUB of upstream's `genie3` console script: `generate -c <yaml> [--log-dir D] [--verbose] [--num-devices N] [--shard-id K] [--num-shards M]`
    writing the stock layout; its argv is recorded in <out>/stub_argv.json."""
    import json, os, sys, yaml
    args = sys.argv[1:]
    assert args[0] == "generate" and args[1] == "-c", args
    req = yaml.safe_load(open(args[2]))
    out = (req["generation"].get("io") or {}).get("outdir") or req["paths"]["rootdir"]
    os.makedirs(out, exist_ok=True); json.dump(args, open(os.path.join(out, "stub_argv.json"), "w"))
    problems = [p.strip() for p in str(req["generation"]["dataset"].get("selections") or "04_pdl1").split(",")]
    n = int(req["generation"]["dataset"]["n_sample"]); seed = req["experiment"].get("seed")
    K = int(args[args.index("--shard-id") + 1]) if "--shard-id" in args else 0; M = int(args[args.index("--num-shards") + 1]) if "--num-shards" in args else 1
    lo, hi = 0, n
    if M > 1:                                                                     # upstream under shards (loader.py _apply_shard_to_config, workflow.py, runtime/shards.py): the shard's share of
        per = -(-n // M); lo = min(K * per, n); hi = min(lo + per, n)              # n_sample named from its sample_index_offset; a done shard (marker present) is skipped, exit 0; the marker
        def targets():                                                            # is written after the pass (or at once for a shard with no share) in every selected problem dir holding pdbs/, else out
            ds = [os.path.join(out, d) for d in sorted(os.listdir(out)) if os.path.isdir(os.path.join(out, d, "pdbs")) and d in problems]
            return ds or [out]
        mk = f"generate_shard_{K}_of_{M}.done"
        if hi > lo and all(os.path.isfile(os.path.join(t, ".shard_markers", mk)) for t in targets()):
            print(f"Shard {K}/{M} already complete (marker found). Skipping generation."); sys.exit(0)
    for p in problems:
        d = os.path.join(out, p, "pdbs"); os.makedirs(d, exist_ok=True)
        for i in range(lo, hi):
            open(os.path.join(d, f"{p}_{i}.pdb"), "w").write(f"REMARK stub seed={seed} B=1 graphs=False\\nATOM {p}_{i}\\nEND\\n")
    if M > 1:
        for t in targets():
            os.makedirs(os.path.join(t, ".shard_markers"), exist_ok=True); open(os.path.join(t, ".shard_markers", mk), "a").close()
    os.makedirs(os.path.join(out, "generation_stats"), exist_ok=True)
    json.dump({"stage": "main", "rank": 0, "sample_count": n * len(problems), "output_count": n * len(problems)}, open(os.path.join(out, "generation_stats", "main_rank_0.json"), "w"))
    if "--log-dir" in args:
        os.makedirs(args[args.index("--log-dir") + 1], exist_ok=True)
    print("Sampling done!")
    ''')

STUB_SMI = "#!/bin/sh\necho \"NVIDIA H100 80GB HBM3, 81559, 9.0\"\n"


def make_tree(tmp, kit_driver_text=STUB_DRIVER, kit_g3cap_text=None, kit_g3batch_text=None):
    """A temporary genie3/ tree: stock/ copied, the kit dir with the stub driver, PINS.json with the installed stack as its stack_check
    (no torch / lightning here). Returns the tree path."""
    tree = os.path.join(tmp, "genie3")
    shutil.copytree(os.path.join(TREE, "stock"), os.path.join(tree, "stock"))
    kit = os.path.join(tree, "opt", "forward", "fast_inference")
    os.makedirs(os.path.join(kit, "driver"), exist_ok=True)
    os.makedirs(os.path.join(kit, "tests"), exist_ok=True)
    open(os.path.join(kit, "driver", "g3fast.py"), "w").write(kit_driver_text)
    shutil.copyfile(os.path.join(TREE, "opt", "forward", "fast_inference", "driver", "g3fast_patches.py"), os.path.join(kit, "driver", "g3fast_patches.py"))
    cap = os.path.join(tree, "opt", "forward", "g3cap")                       # the capture-cell module's directory, its placeholder in the module's place
    os.makedirs(cap, exist_ok=True)
    open(os.path.join(cap, "g3cap.py"), "w").write(STUB_G3CAP if kit_g3cap_text is None else kit_g3cap_text)
    pkg = os.path.join(tree, "opt", "genie3_opt")                               # the batched capture driver's stand-in (package code in the real tree; box() routes the chain entry here)
    os.makedirs(pkg, exist_ok=True)
    open(os.path.join(pkg, "g3batch.py"), "w").write(STUB_G3BATCH if kit_g3batch_text is None else kit_g3batch_text)
    pins = json.load(open(os.path.join(tree, "stock", "PINS.json")))
    sc = {}
    for name in ("torch", "lightning", "numpy"):
        try:
            sc[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    pins["pinned_stack"]["stack_check"] = sc
    # the weights pins = the stub weight bytes make_weights writes (the package refuses any other sha256; a test that wants a refusal rewrites a file)
    pins["weights"]["step=600000.ckpt"]["sha256"] = hashlib.sha256(STUB_CKPT).hexdigest()
    pins["weights"]["config.yaml"]["sha256"] = hashlib.sha256(STUB_CONFIG).hexdigest()
    json.dump(pins, open(os.path.join(tree, "stock", "PINS.json"), "w"), indent=1)
    return tree


def make_checkout(tmp, tree):
    """The pinned checkout: the stock archive extracted (every member byte-identical); no .git."""
    arch = os.path.join(tree, "stock", "genie3-d77ae5ac.tar.gz")
    with tarfile.open(arch) as tf:
        tf.extractall(tmp, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
    root = os.path.join(tmp, "genie3-d77ae5ac")
    assert os.path.isfile(os.path.join(root, "setup.py"))
    return root


def make_weights(tmp):
    w = os.path.join(tmp, "weights")
    os.makedirs(os.path.join(w, "checkpoints"), exist_ok=True)
    open(os.path.join(w, "checkpoints", "step=600000.ckpt"), "wb").write(STUB_CKPT)
    open(os.path.join(w, "config.yaml"), "wb").write(STUB_CONFIG)
    return w


def make_bin(tmp, smi=True, genie3=True):
    """A bin/ with the stub nvidia-smi and the stub genie3 console script (both executable)."""
    b = os.path.join(tmp, "bin")
    os.makedirs(b, exist_ok=True)
    if smi:
        p = os.path.join(b, "nvidia-smi"); open(p, "w").write(STUB_SMI); os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)
    if genie3:
        p = os.path.join(b, "genie3"); open(p, "w").write(STUB_GENIE3); os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)
    return b


def request(tmp, selections="04_pdl1", n_sample=2, seed=None, name="req.yaml", source="target", batch=None, rootdir="examples/x", extra=None):
    req = {"experiment": {"name": "t"}, "paths": {"rootdir": rootdir, "dataset": "data/design/binder_design/binderbench"},
           "generation": {"dataset": {"source": source, "selections": selections, "n_sample": n_sample}, "sampler": {"sampler": {"direction_scale": 0.0}}}}
    if source == "unconditional":                                       # upstream's example shape (examples/unconditional/experiment.yaml)
        req["paths"] = {"rootdir": rootdir}
        req["generation"]["dataset"] = {"source": "unconditional", "min_length": 50, "max_length": 100, "length_step": 50, "n_sample": n_sample}
        req["generation"]["sampler"]["sampler"]["direction_scale"] = 0.8
    if seed is not None:
        req["experiment"]["seed"] = seed
    if batch is not None:
        req["generation"]["dataset"]["batch_size"] = batch
    for dotted, v in (extra or {}).items():                              # e.g. {"generation.inference.search": {...}}
        cur = req
        *head, last = dotted.split(".")
        for k in head:
            cur = cur.setdefault(k, {})
        cur[last] = v
    p = os.path.join(tmp, name)
    yaml.safe_dump(req, open(p, "w"), sort_keys=False)
    return p


def box(tmp, monkeypatch, smi=True, genie3=True):
    """The whole fixture: tree + checkout + weights + bin on PATH, GENIE3_OPT_HOME / GENIE3_ROOT / GENIE3_WEIGHTS set. Returns a dict."""
    from genie3_opt import report, stack
    tree = make_tree(tmp)
    root = make_checkout(tmp, tree)
    w = make_weights(tmp)
    b = make_bin(tmp, smi, genie3)
    monkeypatch.setenv("GENIE3_OPT_HOME", tree)
    monkeypatch.setenv("GENIE3_ROOT", root)
    monkeypatch.setenv("GENIE3_WEIGHTS", w)
    monkeypatch.setenv("PATH", b + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.delenv("GENIE3_OPT", raising=False)
    monkeypatch.delenv("NVIDIA_TF32_OVERRIDE", raising=False)
    monkeypatch.delenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", os.path.join(tmp, "cache"))     # the weights-digest memo (manifest.memo_dir) inside the fixture
    route_g3batch(monkeypatch, tree)
    stack.reset_for_tests()
    report.reset_for_tests()
    return {"tree": tree, "root": root, "weights": w, "bin": b}


def route_g3batch(monkeypatch, tree):
    """Point the package's one self-referencing chain entry (registry.PACKAGE, registry.G3BATCH — the modes' driver, package code) at the test
    tree's stand-in <tree>/opt/genie3_opt/g3batch.py; every other chain entry resolves as in the package (stack.chain_file)."""
    from genie3_opt import registry, stack
    real = stack.chain_file
    def chain_file(kit, rel):
        if kit == registry.PACKAGE and rel == registry.G3BATCH:
            return os.path.join(tree, "opt", "genie3_opt", registry.G3BATCH)
        return real(kit, rel)
    monkeypatch.setattr(stack, "chain_file", chain_file)
