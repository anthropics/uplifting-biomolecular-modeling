"""Stand-ins for the package tests: a stub driver with the real driver's I/O switches and file set, a stand-in tree whose pins the test
interpreter meets, and the environment that makes every gate pass — so the suite decides its own conditions and runs on a box without
jax, tensorflow or the weights (and on the pinned stack alike)."""
from __future__ import annotations

import importlib.metadata as md
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))          # af2ig/
KIT = os.path.join(TREE, "opt", "forward", "af2ig_kit")

STUB_DRIVER = r'''import argparse, glob, json, os, sys, time
p = argparse.ArgumentParser()
for f in ("-pdbdir", "-outpdbdir", "-scorefilename", "-checkpoint_name", "-af2_dir", "-timers", "-runlist"): p.add_argument(f, type=str, default="")
p.add_argument("-precompile", type=int, default=0); p.add_argument("-recycle", type=int, default=3); p.add_argument("-seed", type=int, default=0)
p.add_argument("-program_cache", type=str, default=""); p.add_argument("-prefetch", type=int, default=0); p.add_argument("-overlap_output", type=int, default=0)   # L15 look-ahead depth, L16 writer depth (patch 14)
#                                                      # L13: the program store directory (patch 13)
p.add_argument("-trimul_chunk", type=str, default=""); p.add_argument("-subbatch", type=int, default=0); p.add_argument("-tmpl_pointwise_sub", type=int, default=0)   # L18 (patch 15)      # L9: -subbatch; M1: the memory line's driver hook (patch 06)
for f in ("-fast", "-host_outputs", "-device_params", "-sort_by_length", "-flash_attn", "-fused_triattn", "-fused_trimul", "-opm_reassoc"): p.add_argument(f, action="store_true")
a = p.parse_args()
if a.fast: a.device_params = a.host_outputs = a.sort_by_length = True
SKIP = os.environ.get("STUB_DRIVER_SKIP_TAG", "")            # this design fails (a `failed` record, no score row): the run is incomplete
if os.environ.get("STUB_DRIVER_NO_TIMERS") == "1": a.timers = ""   # the driver writes no record at all
def emit(kind, **kw):
    if a.timers:
        kw.update(kind=kind, t=time.time(), pid=os.getpid())
        with open(a.timers, "a") as fh: fh.write(json.dumps(kw, default=str) + "\n")
emit("proc_start", argv=sys.argv, env={k: os.environ.get(k) for k in ("XLA_FLAGS", "XLA_PYTHON_CLIENT_MEM_FRACTION", "CUDA_VISIBLE_DEVICES")}, jax="stub", seed=a.seed,
     devices=["cuda:0"])
TRIMUL = tuple(int(x) for x in a.trimul_chunk.split(":")) if a.trimul_chunk else None
SUBB = set()
PS = {"eng": 0, "dis": 0}
def pairstack_census(final=False):                            # M1: the install's census as the real adapter writes it (af2ig_opt.pairstack.census)
    return {"installed": True,
            "trimul_chunk": None if TRIMUL is None else {"rows": TRIMUL[0], "min_residues": TRIMUL[1], "engaged_traces": PS["eng"], "disengaged_traces": PS["dis"], "shapes": {}, "producer": "opt_core.mem.rowpair_jax.rowchunk"}}
print("Found GPU and will use it to run AF2")
files = sorted(glob.glob(os.path.join(a.pdbdir, "*.pdb")))
emit("runner_ready", dt=0.01, n_inputs=len(files))
emit("params_loaded", dt=0.01, params_dir=a.af2_dir)
if a.device_params: emit("params_on_device", dt=0.01)
def length(f):
    return sum(1 for l in open(f) if l.startswith("ATOM") and l[12:16].strip() == "CA")
def compiled(L):
    return L                                                    # one program per distinct length (the real driver's rule at P = 1)
PROGS = set()                                                 # patch 13: signatures (lengths) whose program was prepared in this process
if a.precompile:
    shapes = sorted({compiled(length(f)) for f in files})
    for L in sorted(shapes):                                  # patch 13: the programs prepared ahead of the loop (AOT), one program record per new signature (loaded from -program_cache when STUB_PROGRAM_LOADED=1)
        src = "loaded" if (a.program_cache and os.environ.get("STUB_PROGRAM_LOADED") == "1") else "traced"
        emit("precompile_shape", L_compiled=L, dt=0.05, source=src)
        emit("program", tag=None, L_compiled=L, source=src, key="stubkey", **({"t_load": 0.05, "bytes": 123} if src == "loaded" else {"t_trace": 0.02, "t_lower": 0.01, "t_compile": 0.02, **({"t_store": 0.01, "bytes": 123} if a.program_cache else {})}))
        PROGS.add(L)
    emit("precompile_done", n_shapes=len(shapes), threads=a.precompile, dt=0.1); print("precompile: done in 0.1 s")
os.makedirs(a.outpdbdir, exist_ok=True)
hdr = ["binder_aligned_rmsd", "pae_binder", "pae_interaction", "pae_target", "plddt_binder", "plddt_target", "plddt_total", "target_aligned_rmsd", "time"]
with open(a.scorefilename, "a") as sc:
    sc.write("SCORE:     " + " ".join(hdr) + " description\n")
    for f in files:
        tag = os.path.basename(f)[:-4] + "_af2pred"
        if SKIP and tag[:-8] == SKIP:
            emit("failed", tag=f, err="RuntimeError('stub: skipped')"); print("Struct with tag %s failed" % f); continue
        with open(os.path.join(a.outpdbdir, tag + ".pdb"), "w") as fh: fh.write("ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00 90.00           N\n")
        sc.write("SCORE:     " + " ".join("%8.3f" % (1.0 + i) for i in range(8)) + " %8.3f        %s\n" % (0.5, tag))
        with open(a.checkpoint_name, "a") as fh: fh.write(tag[:-8] + "\n")
        L = length(f)
        if TRIMUL is not None:
            PS["eng" if L >= TRIMUL[1] else "dis"] += 2              # two TriangleMultiplication call sites traced per new program (stub count)
            emit("pairstack", L_compiled=L, **pairstack_census())
        if a.tmpl_pointwise_sub and not SUBB and L not in SUBB:   # L18: one tmpl_pointwise_sub record per process, as the real driver writes it at setup
            emit("tmpl_pointwise_sub", value=a.tmpl_pointwise_sub, stock_value=128)
        if a.subbatch and L not in SUBB:                         # L9: the shared policy's decision record per compiled length, as the real driver writes it (af2ig_opt.subbatch.record)
            SUBB.add(L); emit("subbatch", L_compiled=L, value=a.subbatch, source="requested", tokens=L, stock_value=4, estimated_bytes=None, device_bytes=None, requested=a.subbatch, details={}, policy="opt_core.jax_design.subbatch_policy")
        if (a.precompile or a.program_cache) and L not in PROGS:   # patch 13: a new signature met in the loop — its program record
            PROGS.add(L); emit("program", tag=tag[:-8], L_compiled=L, source="traced", key="stubkey", t_trace=0.02, t_lower=0.01, t_compile=0.02, **({"t_store": 0.01, "bytes": 123} if a.program_cache else {}))
        emit("design", tag=tag[:-8], L=L, L_compiled=L, t_model=1.5, t_output=0.1, t_total=1.7,
             **({} if os.environ.get("STUB_DRIVER_NO_PEAK") == "1" else {"peak_bytes_in_use": 5000000000}))   # patch 11: the allocator peak read after the design (STUB_DRIVER_NO_PEAK=1: a driver without it / a backend with no memory statistics)
        print("Tag: %s reported success in 1.7 seconds" % tag)
if a.flash_attn:                                              # L8: the adapter's census as the real driver writes it at exit (STUB_FLASH_SERVED=0: the kernel in no program; STUB_FLASH_UNEXPECTED=<reason>: one call left on the stock ops for an undeclared reason)
    served = int(os.environ.get("STUB_FLASH_SERVED", "12")); aunexp = os.environ.get("STUB_FLASH_UNEXPECTED", "")
    fb = ({"no_pair_bias": 4} if served else {"below_size_rule": 16}) | ({aunexp: 1} if aunexp else {})
    emit("flash_attn", L_compiled=None, final=True, name="F1.pallas_attn", impl="pallas_attn", origin="core", state="on" if served else "skipped", served=served, calls=served + sum(fb.values()), errors={},
         fallback=sum(fb.values()), fallback_by=fb, min_tokens=0, shapes={}, partial=not served, gate_source="default", all_calls=False, precision="tf32")
if a.fused_triattn:                                           # L10: the fused block's census as the real driver writes it at exit (STUB_FUSED_SERVED=0: the block in no program; STUB_FUSED_UNEXPECTED=<reason>: an undeclared fallback)
    fserved = int(os.environ.get("STUB_FUSED_SERVED", "4")); unexp = os.environ.get("STUB_FUSED_UNEXPECTED", "")
    emit("fused_triattn", L_compiled=None, final=True, name="F1.fpf_pallas_triattn", impl="fpf_pallas_f32", origin="core", state="on" if fserved else "skipped", served=fserved, bridge=("on:stubrow@min_len0/0" + ("+core_bfloat16" if os.environ.get("AF2IG_OPT_TRIATTN_CORE_DTYPE") == "bf16" else "")), providers=({"triattn_xla:stubrow": fserved} if fserved else {}),   # L19 (0.7.6): the census words the real adapter writes (bridge word + per-row counts)
      calls=fserved + (1 if unexp else 0) + (0 if fserved else 4), errors={},
         fallback=(1 if unexp else 0) + (0 if fserved else 4), fallback_by=({unexp: 1} if unexp else {}) | ({} if fserved else {"dtype_not_served": 4}), shapes={"N100xC128xH4": fserved} if fserved else {}, partial=not fserved, first=None, precision="tf32", key_mask="by_line", cc="9.0", tiles="own:9.0", jax="0.5.3")
if a.fused_trimul:                                            # L11: the fused triangle-multiplication block's census (STUB_FTRIMUL_SERVED=0: in no program — every call fell back; with STUB_FTRIMUL_FALLBACK=0 too: no call reached the block at all, the row-chunked body took them)
    mserved = int(os.environ.get("STUB_FTRIMUL_SERVED", "4")); mfallback = int(os.environ.get("STUB_FTRIMUL_FALLBACK", "0" if mserved else "4"))
    emit("fused_trimul", L_compiled=None, final=True, name="F2.fpf_pallas_trimul", impl="fpf_pallas_f32", origin="core", state="on" if mserved else "skipped", served=mserved, calls=mserved + mfallback, errors={},
         fallback=mfallback, fallback_by={"dtype_not_bf16": mfallback} if mfallback else {}, shapes={"N100xC128": mserved} if mserved else {}, partial=bool(mfallback) and not mserved, first=None, precision="tf32", cc="9.0", tiles="own:9.0", jax="0.5.3")
if a.opm_reassoc:                                             # L12: the re-associated OuterProductMean's census as af2ig_opt.opm.census writes it (STUB_OPM_SERVED=0: in no program)
    oserved = int(os.environ.get("STUB_OPM_SERVED", "2"))
    emit("opm_reassoc", L_compiled=None, final=True, name="LOCAL.af2ig.opm_reassoc", impl="opm_reassoc_jax", origin="kit", state="on" if oserved else "skipped", served=oserved, calls=oserved, errors={},
         fallback=0, fallback_by={}, shapes={"S5xN100xM256": oserved} if oserved else {}, partial=False, first=None, precision="default", backend="gpu", jax="0.5.3", lever="L12")
if TRIMUL is not None:
    emit("pairstack", L_compiled=None, final=True, **pairstack_census(final=True))
if a.program_cache:                                           # patch 13: the store census at exit + the L13 line
    os.makedirs(a.program_cache, exist_ok=True)
    nl = len(PROGS) if os.environ.get("STUB_PROGRAM_LOADED") == "1" else 0
    emit("program_store", loaded=nl, traced=len(PROGS) - nl, stored=len(PROGS) - nl, load_failed=0, store_failed=0, events=[], bytes_loaded=123 * nl, bytes_stored=123 * (len(PROGS) - nl), entries=len(PROGS), dir=a.program_cache, static_key="stubkey")
    print(f"[af2ig-opt] LEVER name=L13 state=on impl=serialize_executable origin=kit flag=-program_cache dir={a.program_cache} loaded={nl} traced={len(PROGS) - nl}", file=sys.stderr)
if a.prefetch:                                                # patch 14: the prefetcher's census at exit + the L15 line
    emit("prefetch", impl="thread_lookahead", depth=a.prefetch, workers=1, queued=len(files), taken=len(files), ready=max(0, len(files) - 1), waited=min(1, len(files)), skipped=0, unscheduled=0, errors=0, wait_s=0.2, work_s=0.5 * len(files), overlap_s=max(0.0, 0.5 * len(files) - 0.2), state="on")
    print(f"[af2ig-opt] LEVER name=L15 state=on impl=thread_lookahead origin=kit flag=-prefetch depth={a.prefetch} queued={len(files)} taken={len(files)}", file=sys.stderr)
if a.overlap_output:                                          # patch 14: the output writer's census at exit + the L16 line
    emit("output_writer", impl="thread_writer", depth=a.overlap_output, jobs=len(files), errors=0, busy_s=0.3 * len(files), lag_s=0.1, blocked_s=0.0, max_pending=1, state="on")
    print(f"[af2ig-opt] LEVER name=L16 state=on impl=thread_writer origin=kit flag=-overlap_output depth={a.overlap_output} jobs={len(files)}", file=sys.stderr)
emit("proc_end", n_done=len(files))
'''


from opt_core.gates import sha256_file as sha256          # noqa: E402 — used by other test modules for live file-vs-file comparisons (never a stored digest)


def make_tree(tmp, pins_ok=True, pins_absent=False):
    """A stand-in af2ig/ tree under `tmp`: stock/PINS.json whose named pins the test interpreter meets (pins_ok=False: installed at another
    version — drift; pins_absent=True: a required distribution that is not installed — the one pins refusal), the real
    stock/check_pins.py, a stub checkout (the driver only) and a stub weights directory pinned to the stub's bytes. Returns
    (tree, environment) — the environment makes every gate of the real package pass on it (AF2IG_OPT_KIT = the real kit)."""
    tree = os.path.join(tmp, "af2ig"); os.makedirs(os.path.join(tree, "stock")); os.makedirs(os.path.join(tree, "opt"))
    shutil.copyfile(os.path.join(TREE, "stock", "check_pins.py"), os.path.join(tree, "stock", "check_pins.py"))
    shutil.copyfile(os.path.join(TREE, "opt", "pyproject.toml"), os.path.join(tree, "opt", "pyproject.toml"))     # the core pin, read by the first gate
    real = json.load(open(os.path.join(TREE, "stock", "PINS.json"), encoding="utf-8"))
    ck = os.path.join(tmp, "checkout", "af2_initial_guess"); os.makedirs(ck)
    with open(os.path.join(ck, "predict_pdb.py"), "w") as fh:
        fh.write(STUB_DRIVER)
    params = os.path.join(tmp, "params"); os.makedirs(os.path.join(params, "params"))
    wp = os.path.join(params, "params", "params_model_1_ptm.npz")
    with open(wp, "wb") as fh:
        fh.write(b"stub weights")
    have = md.version("pip") if _installed("pip") else md.version("setuptools")
    name = "pip" if _installed("pip") else "setuptools"
    pins = dict(real)
    pins["pins"] = {name: have if pins_ok else "0.0.1"}
    if pins_absent:
        pins["pins"]["af2ig-no-such-distribution"] = "1.0"
    pins["pins_gpu_only"] = []
    pins["checkout"] = dict(real["checkout"], relpaths=["af2_initial_guess/predict_pdb.py"], patched_relpaths=[], n_files=1)   # patched_relpaths empty: the stub driver is not a copy of any kit patched_files/ file, so there is nothing to compare byte-for-byte
    pins["weights"] = dict(real["weights"], sha256=sha256(wp), bytes=os.path.getsize(wp))
    with open(os.path.join(tree, "stock", "PINS.json"), "w", encoding="utf-8") as fh:
        json.dump(pins, fh, indent=1)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AF2IG_OPT", "AF2_SUBBATCH", "JAX_COMPILATION", "JAX_PERSISTENT", "CUDA_MPS_", "MODEL_OPT_JIT_ROOT", "MODEL_OPT_LEVERS_OFF", "XLA_FLAGS"))}
    env.update({"AF2IG_OPT_HOME": tree, "AF2IG_OPT_KIT": KIT, "AF2IG_DIR": ck, "AF2_PARAMS": params, "AF2IG_OPT_CACHE_DIR": os.path.join(tmp, "cache"), "PYTHONDONTWRITEBYTECODE": "1"})
    env.pop("MODEL_OPT_TARGET_GPU", None)
    env = prepend_pythonpath(env, OPT)   # the package itself must be resolvable in every subprocess these tests spawn, regardless of how THIS
                                          # process found it (an editable install, a PYTHONPATH set by the runner, a sys.path insertion by a
                                          # caller — the last carries no PYTHONPATH at all): the pin gate is what the "stale core" tests are
                                          # exercising, so af2ig_opt's own presence must never be left to inheritance
    return tree, env


def _installed(name):
    try:
        md.version(name); return True
    except md.PackageNotFoundError:
        return False


def inputs(tmp, n=2):
    d = os.path.join(tmp, "in"); os.makedirs(d, exist_ok=True)
    src = os.path.join(KIT, "tests", "inputs", "pdbs")
    names = sorted(os.listdir(src))[:n]
    for nm in names:
        shutil.copyfile(os.path.join(src, nm), os.path.join(d, nm))
    return d, [nm[:-4] for nm in names]


def long_input(in_dir, name, n_res):
    """A synthetic complex of `n_res` residues (one CA per residue, two chains) under `in_dir`: the stub driver reads only the residue count."""
    os.makedirs(in_dir, exist_ok=True)
    with open(os.path.join(in_dir, name + ".pdb"), "w") as fh:
        for i in range(n_res):
            chain = "A" if i < n_res // 2 else "B"
            fh.write("ATOM  %5d  CA  GLY %s%4d    %8.3f%8.3f%8.3f  1.00 90.00           C\n" % (i + 1, chain, i % 9999 + 1, 3.8 * i, 0.0, 0.0))
    return name




OPT = os.path.join(TREE, "opt")                                 # the package's project directory (pyproject.toml; `python -m af2ig_opt` resolves from here)


def _stub_module(path_no_ext, mod):
    """Write a module (package dir or .py) at `path_no_ext` that raises ImportError naming `mod` — an absent sub-module no other finder can resupply."""
    if path_no_ext.endswith(".py"):
        os.makedirs(os.path.dirname(path_no_ext), exist_ok=True); stub = path_no_ext
    else:
        os.makedirs(path_no_ext, exist_ok=True); stub = os.path.join(path_no_ext, "__init__.py")
    with open(stub, "w", encoding="utf-8") as fh:
        fh.write(f'raise ImportError("{mod} absent from this core (shadowed by the test)", name="{mod}")\n')


def shadow_core(tmp, name, *, version=None, strip_version=False, drop=()):
    """A copy of the INSTALLED opt_core's package directory under tmp/<name>/, to PREPEND to PYTHONPATH (opt_core carries no MANIFEST.json
    to shadow: the pin gate reads only the package's own __version__ literal). `version` rewrites the copy's __init__.py __version__ to an
    OLDER version than the pin (the gate's core_mismatch: FLOOR semantics mean only an older version refuses). `strip_version` removes the
    __version__ literal entirely (an unversioned core — the gate's core_mismatch names it installed "v?"). `drop`: sub-modules/-packages
    replaced by ImportError stubs (a core lacking them, at whatever version — the package's own second guard on a missing sub-module)."""
    import re, opt_core
    pkg = os.path.dirname(opt_core.__file__); d = os.path.join(tmp, name)
    shutil.copytree(pkg, os.path.join(d, "opt_core"), ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    ini = os.path.join(d, "opt_core", "__init__.py")
    if strip_version:
        t = open(ini, encoding="utf-8").read()
        open(ini, "w", encoding="utf-8").write(re.sub(r'^__version__\s*=\s*[\'"][^\'"]+[\'"]\s*\n', "", t, count=1, flags=re.M))
    elif version:
        t = open(ini, encoding="utf-8").read()
        open(ini, "w", encoding="utf-8").write(re.sub(r'__version__\s*=\s*[\'"][^\'"]+[\'"]', f'__version__ = "{version}"', t, count=1))
    for rel in drop:
        q = os.path.join(d, "opt_core", rel)
        shutil.rmtree(q) if os.path.isdir(q) else (os.remove(q) if os.path.exists(q) else None)
        _stub_module(q, "opt_core." + rel.replace(".py", "").replace("/", "."))
    return d


def prepend_pythonpath(env, *dirs):
    e = dict(env); parts = list(dirs) + ([e["PYTHONPATH"]] if e.get("PYTHONPATH") else []); e["PYTHONPATH"] = os.pathsep.join(parts); return e


def run_cli(args, env, cwd=None, no_site=False):
    """`python -m af2ig_opt <args>`; no_site=True runs `python -S` (site-packages and their .pth finders off: only PYTHONPATH supplies packages —
    the way to make an installed opt_core ABSENT for one process; OPT is prepended so the package itself stays importable)."""
    import subprocess
    if no_site:
        env = prepend_pythonpath(env, OPT)
    r = subprocess.run([sys.executable] + (["-S"] if no_site else []) + ["-m", "af2ig_opt"] + list(args), env=env, capture_output=True, text=True, cwd=cwd or os.path.dirname(HERE))
    norm = lambda t: re.sub(r'(-program_cache(?:", "| ))([^" \n]*/programs)', lambda m: m.group(1) + "<programs>", t)   # the L13 directory (<cache root>/jit/<stack key>/programs) depends on the interpreter's jax and the temp root: normalized for the assertions; TestCcache covers the placement itself
    return r.returncode, norm(r.stdout), norm(r.stderr)


def records(stdout: str):
    """The driver's own -timers records as relayed to stdout by the package (cli.Relay): one JSON object per line, in order."""
    return [json.loads(l) for l in stdout.splitlines() if l.startswith("{")]


def run_files(out_dir: str):
    """What a run leaves under <out>: the driver's own outputs only (pdbs/, out.sc, check.point)."""
    return sorted(os.listdir(out_dir))

