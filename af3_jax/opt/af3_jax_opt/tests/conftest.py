"""Fixtures: a stand-in for the fork's interpreter and install, so every route runs on CPU without jax, a GPU or the weights.

`box` builds under a temp dir: a repo directory with the stock scripts of the archive (stock/src) and a site directory with the stock
module files the add-ons pin or patch (stock/PINS.json stock_files), a stub interpreter that (a) answers the package's probes (jax
build; the site directory) and check_pins.py's package probe (versions read from environment/requirements.lock) and (b) plays the model process
— whatever launcher and script it is handed, it records argv and environment to a JSON file, writes the fork's output file set for each
seed of each input, prints the kit script's row-lever and timing lines when the fast script is on argv, — a params root with fake converted parameters, a cache root, and a `bin/` with a fake nvidia-smi (one H100) and
a `python` that is this interpreter, for driving run.sh. Nothing here is imported by the package.
"""
from __future__ import annotations

LOG_FAILURE_MARKERS = ("OOM", "out of memory", "fallback", "WARN", "Error", "Traceback", "Could not load", "NOT ACTIVE", "refus")   # failure-marker substrings a log scanner keys on: a NOTE-class line of this package (BIG NOTE, NOTE xla_mem_fraction, PEAK, ROWPAIR NOTE) never carries one


def assert_no_markers(line: str) -> None:
    hit = [w for w in LOG_FAILURE_MARKERS if w in line]
    assert not hit, (hit, line)


import json
import os
import shutil
import stat
import sys
import textwrap
from typing import Optional

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
OPT = os.path.dirname(PKG)
TREE = os.path.dirname(OPT)
FORWARD = os.path.join(OPT, "forward")
KIT = os.path.join(FORWARD, "fast_inference")
PALLAS = os.path.join(FORWARD, "pallas_addon")


# ---- fold inputs the tests render (the fork's `alphafold3` dialect, version 2: unpairedMsa = the query alone, no paired MSA, no templates)
def chain(chain_id, sequence, kind="protein", unpaired_msa=None):
    body = {"id": chain_id, "sequence": sequence.strip().upper()}
    if kind == "protein":
        body.update({"unpairedMsa": unpaired_msa if unpaired_msa is not None else f">query\n{body['sequence']}\n", "pairedMsa": "", "templates": []})
    elif kind == "rna":
        body["unpairedMsa"] = unpaired_msa if unpaired_msa is not None else f">query\n{body['sequence']}\n"
    return {kind: body}


def ligand(chain_id, ccd_codes):
    return {"ligand": {"id": chain_id, "ccdCodes": list(ccd_codes)}}


def render(name, entities, seeds=(1,)):
    return {"name": name, "modelSeeds": list(seeds), "dialect": "alphafold3", "version": 2, "sequences": list(entities)}
FPF = os.path.join(FORWARD, "flashpairformer")
STOCK_SRC = os.path.join(TREE, "stock", "src")
CORE = os.path.dirname(os.path.dirname(os.path.abspath(__import__("opt_core").__file__)))   # the shared core beside the tree (common/opt_core), as imported
CHILD_PYTHONPATH = os.pathsep.join((OPT, CORE))                                              # a child interpreter finds the package and its core
HOOK_PTH = os.path.join(OPT, "af3_jax_opt_autoload.pth")                                      # the start-up hook the wheel ships at its root (opt/_build_backend.py)


def make_venv(dest: str, hook: bool, package: bool = True, hook_text: Optional[str] = None, core: bool = True, core_path: Optional[str] = None) -> str:
    """A venv of this interpreter (no pip, the system site visible) whose site carries, with ``package``, the package and the core the way an
    editable install does (`__editable__af3_jax_opt.pth`: the two paths) and, with ``hook``, the start-up hook .pth; returns its python. Two
    venvs prove run.sh's hook gate: with the hook (A) a kit mode passes the gate; importable alone (B) it is refused by name."""
    import subprocess
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", "--system-site-packages", dest], check=True, capture_output=True)
    import glob
    site_dir = glob.glob(os.path.join(dest, "lib", "python*", "site-packages"))[0]
    if package:
        with open(os.path.join(site_dir, "__editable__af3_jax_opt.pth"), "w") as f:
            f.write(OPT + "\n" + ((core_path or CORE) + "\n" if core else ""))                 # core=False: the package importable, the shared core absent; core_path: another core root (a stale one) in its place
    if hook or hook_text is not None:                                                     # hook_text: a stale copy (another import line) in place of the kit's
        with open(os.path.join(site_dir, os.path.basename(HOOK_PTH)), "w") as f:
            f.write(open(HOOK_PTH).read() if hook_text is None else hook_text)
    return os.path.join(dest, "bin", "python")


def python_on_path(bin_dir: str, py: str) -> None:
    """`python` in ``bin_dir`` = a wrapper that execs ``py`` by its own path (a symlink would lose the venv: the interpreter finds pyvenv.cfg
    beside the path it is invoked as)."""
    p = os.path.join(bin_dir, "python")
    with open(p, "w") as f:
        f.write(f'#!/bin/sh\nexec "{py}" "$@"\n')
    os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)


_VENV_A = {}


def venv_a_python() -> str:
    """The session's venv A (the hook in site), built once."""
    if "py" not in _VENV_A:
        import tempfile
        _VENV_A["dir"] = tempfile.mkdtemp(prefix="af3_jax_opt_venv_a_")
        _VENV_A["py"] = make_venv(_VENV_A["dir"], hook=True)
    return _VENV_A["py"]
GPU_H100 = {"name": "NVIDIA H100 80GB HBM3", "memory_mib": 81559, "compute_cap": 9.0, "count": 1}
BUILD = {"python": "3.12.1", "jax": "0.10.2", "jaxlib": "0.10.2", "executable": "stub"}

STUB = textwrap.dedent('''\
    #!/usr/bin/env python3
    """Stand-in for /alphafold3_venv/bin/python (tests only)."""
    import json, os, sys, time
    argv = sys.argv[1:]
    if argv[:2] == ["-I", "-c"]:
        code = argv[2]
        if "m.version('jax')" in code:
            print(json.dumps({"python": "3.12.1", "jax": "0.10.2", "jaxlib": "0.10.2", "alphafold3": None, "executable": sys.argv[0]})); sys.exit(0)
        if "import alphafold3" in code:
            print(os.environ["STUB_SITE"]); sys.exit(0)
        if "names = json.loads(sys.argv[1])" in code:                       # stock/check_pins.py's package probe: answer from the pinned freeze
            want = {}
            for ln in open(os.environ["STUB_FREEZE"], encoding="utf-8"):
                ln = ln.strip()
                if ln.startswith(("#", "-")) or not ln: continue
                if " @ " in ln:                                                     # a wheel pinned by URL: the version is the file name's (check_pins.freeze_versions' rule)
                    n, u = (x.strip() for x in ln.split(" @ ", 1)); want[n.lower().replace("_", "-")] = __import__("urllib.parse").parse.unquote(u.split("#", 1)[0].rsplit("/", 1)[-1]).split("-")[1]
                elif "==" in ln:
                    n, v = ln.split("==", 1); want[n.lower().replace("_", "-")] = v
            names = json.loads(argv[3])
            print(json.dumps({"python": "3.12.1", "packages": {n: want.get(n.lower().replace("_", "-")) for n in names}})); sys.exit(0)
        exec(code); sys.exit(0)
    MODEL_SCRIPTS = ("run_alphafold.py", "run_alphafold_fast.py")
    script = next((a for a in argv if os.path.basename(a) in MODEL_SCRIPTS), argv[0])
    launcher = os.path.basename(argv[0]) if os.path.basename(argv[0]) not in MODEL_SCRIPTS else None
    flags = [a for a in argv if a.startswith("--")]
    def val(name):
        for i, f in enumerate(argv):
            if f == name: return argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith("--") else "true"
            if f.startswith(name + "="): return f.split("=", 1)[1]
        return None
    rec = {"script": os.path.basename(script), "script_path": script, "launcher": launcher, "argv": argv, "flags": flags,
           "env": {k: v for k, v in os.environ.items()}, "cwd": os.getcwd()}
    with open(os.environ["STUB_RECORD"], "w") as f:
        json.dump(rec, f)
    if os.environ.get("STUB_FAIL"):
        print("stub: failing on request"); sys.exit(7)
    if launcher in ("levers_launch.py", "fpf_launch.py", "big_launch.py") and os.environ.get("STUB_CACHEKEY", "1") == "1":   # the tree's launchers install inprocess/portable_cache_key.py first (STUB_CACHEKEY=0: a process without it)
        print("[af3-jax-opt] CACHEKEY accelerator=device_kind jax=0.10.2 site=jax._src.cache_key._hash_accelerator_config")
    if launcher == "levers_launch.py":                                       # the tree's launcher line, from the lever variables as the add-on installs them
        names = (["L-GLUT"] if os.environ.get("AF3P_GLU_T") == "1" else []) + (["L-ATTNCFG"] if os.environ.get("AF3P_ATTN_CFG") else [])
        print("[af3-jax-opt] LEVERS active=" + ("+".join(sorted(names)) or "none") + " af3_pallas_levers.py=stub script=" + os.path.basename(script) + " sha256=stub")
    if launcher == "fpf_launch.py":                                          # the tree's launcher: the add-on's own install line (patch.py, logging.warning), the SERVED line at exit
        fmode = os.environ.get("AF3_FLASHPAIRFORMER", "both")
        print(f"af3_flashpairformer: mode={fmode} (TriangleMultiplication=FlashTriangleMultiplication, GridSelfAttention=FlashGridSelfAttention), jax 0.10.2 backend gpu")
        fb = os.environ.get("STUB_FPF_FALLBACK")                                # STUB_FPF_FALLBACK=<N>: the kernels fell back at an N x N pair activation (N % 128 != 0)
        pf = os.environ.get("STUB_FPF_PARTIAL")                                 # STUB_FPF_PARTIAL=<N>: the kernels ENGAGED (fused>0) and also fell back at an N x N activation
        hz = os.environ.get("STUB_FPF_HOIST", "1" if os.environ.get("AF3_DIFFUSION_HOIST") == "1" else "off")   # STUB_FPF_HOIST=0: installed, never traced
        fbn = fb or pf
        print(f"[af3-jax-opt] SERVED trimul fused={0 if fb else 3} fallback={1 if fbn else 0} triatt fused={0 if fb else 2} fallback={1 if fbn else 0} "
              f"fallback_shapes={f'trimul:{fbn}x{fbn}x128:bfloat16,triatt:{fbn}x{fbn}x128:bfloat16' if fbn else 'none'} hoist={hz} "
              f"tiles={os.environ.get('STUB_FPF_TILES', 'own')} cc=9.0 "
              f"dattn={os.environ.get('STUB_DATTN', '53' if os.environ.get('AF3_JAX_DATTN', '') not in ('', '0') else 'off')} dattn_sites={'diffusion:5,pairformer_single:48' if os.environ.get('AF3_JAX_DATTN', '') not in ('', '0') else 'none'} "   # DATTN: served through the shared core's provider by tier word (switch 1 = fast, big on the memory line); its own census line below
              f"ttr={os.environ.get('STUB_TTR', '4' if os.environ.get('AF3_JAX_TTR', '') not in ('', '0') else 'off')} ttr_routed={2 if os.environ.get('AF3_JAX_TTR', '') not in ('', '0') else 0} ttr_fallback={os.environ.get('STUB_TTR_FALLBACK', 'none')} "   # TTR: the switch carries the composition's tier word (fast | big)
              f"sbf16={os.environ.get('STUB_SBF16', '16' if os.environ.get('AF3_JAX_SAMPLER_BF16') == '1' else 'off')} sbf16_sites={'atom:12,token:4' if os.environ.get('AF3_JAX_SAMPLER_BF16') == '1' else 'none'} "
              f"txla={os.environ.get('STUB_TXLA', '24' if os.environ.get('AF3_JAX_TRIATT_XLA', '') not in ('', '0') else 'off')} txla_rows={'cuda_sm90a:20,k2b_aot:4' if os.environ.get('AF3_JAX_TRIATT_XLA', '') not in ('', '0') else 'none'} txla_aside=none "   # TRIATT_XLA: served traced calls / the core's rows / step-asides by name (an H100 transcript at 1,024 padded tokens)
                 # the words a fast pass prints on an H100 (3+3 atom-transformer sites and the DiT's attention + transition site)
              
              f"atomattn={os.environ.get('STUB_ATOMATTN', '7' if os.environ.get('AF3_JAX_ATOM_ATTN') == '1' else 'off')} atomattn_sites={'diffusion_decoder:3,diffusion_encoder:3,evoformer_encoder:1' if os.environ.get('AF3_JAX_ATOM_ATTN') == '1' else 'none'} atomattn_fallback={os.environ.get('STUB_ATOMATTN_FALLBACK', 'none')} "
              f"hlog={os.environ.get('STUB_HLOG', '2' if (os.environ.get('AF3_JAX_HOIST_LOGITS') == '1' and hz not in ('off', '0')) else ('skipped' if os.environ.get('AF3_JAX_HOIST_LOGITS') == '1' else 'off'))} hlog_dtype={('float32' if hz not in ('off', '0') else 'hoist_off') if os.environ.get('AF3_JAX_HOIST_LOGITS') == '1' else 'none'} "   # HOIST_LOGITS: step traces (skipped by name when the hoist is off)
              f"cshare={os.environ.get('STUB_CSHARE', '2' if os.environ.get('AF3_JAX_COND_SHARE') == '1' else 'off')} "
              f"achoist={os.environ.get('STUB_ACHOIST', ('2' if hz not in ('off', '0') else 'skipped') if os.environ.get('AF3_JAX_ATOM_COND_HOIST') == '1' else 'off')} achoist_sites={'enc:2,dec:2,passed:2' if os.environ.get('AF3_JAX_ATOM_COND_HOIST') == '1' else 'none'} achoist_aside={('none' if hz not in ('off', '0') else 'no_hoist_step:2') if os.environ.get('AF3_JAX_ATOM_COND_HOIST') == '1' else 'none'} "   # COND_SHARE / ATOM_COND_HOIST: sample calls traced / hoisted (the atom hoist steps aside by name without FPF_HOIST's step)
              f"cnoise=off cnoise_rule=none tcd={os.environ.get('STUB_TCD', '24' if os.environ.get('AF3_JAX_TRIMUL_CD', '') not in ('', '0') else 'off')} tcd_rows={'cd_trimul:24' if os.environ.get('AF3_JAX_TRIMUL_CD', '') not in ('', '0') else 'none'} tcd_aside=none tcd_word={('fast' if os.environ.get('AF3_JAX_TRIMUL_CD') == '1' else os.environ.get('AF3_JAX_TRIMUL_CD')) if os.environ.get('AF3_JAX_TRIMUL_CD') else 'none'} tcd_uncovered={'af3_tmpl_c64_ch64_incoming:2' if os.environ.get('AF3_JAX_TRIMUL_CD', '') not in ('', '0') else 'none'}")
        _lw = os.environ.get("AF3_JAX_LNP")                                     # LNP: the lever's own line (inprocess/lnp.py line(): served traced calls through the provider, the word, rows per unit; an H100 transcript's words)
        print(f"[af3-jax-opt] LNP served={os.environ.get('STUB_LNP', '150' if _lw else 'off')} word={('fast' if _lw == '1' else _lw) if _lw else 'none'} rows={'cd_ln:62,xla:88' if _lw else 'none'} units={'msa:88,pair:58,tmpl:4' if _lw else 'none'} routed={'adaptive:40,axis:2,channels:12' if _lw else 'none'} aside={os.environ.get('STUB_LNP_ASIDE', 'none')} uncovered=none")
        if os.environ.get('AF3_JAX_DATTN', '') not in ('', '0'):                 # the DATTN lever's own census line at exit (inprocess/dattn.py census_line): the provider word it named (1 = fast; big on the memory line), the arms that served (an H100 transcript above the 400-token bucket: tokamax@triton at both sites), step-asides / uncovered families by name
            dword = 'fast' if os.environ['AF3_JAX_DATTN'] == '1' else os.environ['AF3_JAX_DATTN']
            dserved = os.environ.get('STUB_DATTN', '53')
            print(f"[af3-jax-opt] DATTN word={dword} served={dserved if dserved.isdigit() else 0} rows={('tokamax@triton:' + dserved) if (dserved.isdigit() and int(dserved) > 0) else 'none'} aside={os.environ.get('STUB_DATTN_ASIDE', 'none')} uncovered=none")
    out = val("--output_dir"); cache = val("--cache_dir") or os.environ["STUB_DEFAULT_CACHE"]   # no --cache_dir: the fork caches at its flag's default
    if "--cache_dir=" in sys.argv[1:]: cache = os.path.join(os.getcwd(), "jax")            # `--cache_dir=` (empty): the fork keeps its JAX cache at ./jax under the cwd and skips the autotune cache (run_alphafold.py _CACHE_DIR handling)
    inputs = []
    if val("--json_path"): inputs = [val("--json_path")]
    elif val("--input_dir"): inputs = sorted(os.path.join(val("--input_dir"), f) for f in os.listdir(val("--input_dir")) if f.endswith(".json"))
    os.makedirs(cache, exist_ok=True)
    jc = os.path.join(cache, "jit__stub_apply_fn")                           # the JAX persistent compilation cache: an entry written on a cold cache (a miss), read (untouched) when present
    if not os.path.exists(jc):
        with open(jc, "w") as f: f.write("executable\\n")
    tk = os.path.join(cache, "tokamax_autotune.json")                       # upstream's tokamax table: loaded when the class has one, else the autotune attempt (STUB_TOKAMAX=unavailable: it raises, as on the pinned stack)
    if "--cache_dir=" in sys.argv[1:]: pass                                  # `--cache_dir=`: run_alphafold.py keeps no table path — no load, no attempt
    elif os.path.exists(tk): print(f"Loading tokamax autotune cache from {tk}")
    elif os.environ.get("STUB_TOKAMAX") == "unavailable": print("Tokamax autotune unavailable (RuntimeError: AttributeError: 'NameLoc' object has no attribute 'is_a_file'); kernel configs from tokamax's packaged tables, else tokamax_autotuning_cache_miss_fallback=heuristics.")
    else:
        with open(tk, "w") as f: f.write(json.dumps({"device_kind": "NVIDIA H100 80GB HBM3", "data": [[{"op": "stub"}, {"config": "stub"}]]}) + "\\n")
        print(f"Tokamax autotune cache saved to {tk}"); print("Subsequent runs will load this cache and skip autotuning.")
    if os.path.basename(script) == "run_alphafold_fast.py":                    # the kit script's own line for the row lever on its argv (L1 prefetch)
        if val("--featurisation_workers") and not os.environ.get("STUB_NO_ROW_LINES"): print(f"Featurisation prefetch enabled: {val('--featurisation_workers')} worker process(es), {val('--featurisation_prefetch')} item(s) ahead.")
        if "--output_writer" in argv and not os.environ.get("STUB_NO_ROW_LINES"): print("Output writer enabled: result extraction and output writing run on one writer thread behind the next fold job.")   # the WRITER row lever's line
    if os.environ.get("OPT_KERNELS_EXPECT") is not None and os.environ.get("STUB_KERNELS") != "none":   # the KERNELS probe (kernels_probe.py, carried in the hook dir LAST on PYTHONPATH): the stub plays the model
        import kernels_probe as kp                                            # process's first kernel call — a healthy reading for its launcher/levers unless STUB_KERNELS_* says otherwise
        if os.environ.get("STUB_KERNELS_DOWNGRADE"):                          # the fork's CPU auto-downgrade print (run_alphafold.py:1131-1134), then the flag as rewritten
            print("Setting --flash_attention_implementation=xla, required for CPU-only inference.")
        fpf = os.environ.get("AF3_FLASHPAIRFORMER", "")
        reading = {"flag": os.environ.get("STUB_KERNELS_FLAG") or ("xla" if os.environ.get("STUB_KERNELS_DOWNGRADE") else (val("--flash_attention_implementation") or "triton")),
                   "dpa_site": "attncfg" if os.environ.get("AF3P_ATTN_CFG") else "tokamax",
                   "triatt_site": "fpf" if fpf in ("both", "triatt") else "stock",
                   "trimul_site": os.environ.get("STUB_KERNELS_TRIMUL") or ("fpf" if fpf in ("both", "trimul") else ("glut" if os.environ.get("AF3P_GLU_T") == "1" else "stock")),
                   "transition_site": "stock", "attn_impls": "cudnn,msc_gpu,msc_tpu,triton,xla,xla_chunked",
                   "attn_class": os.environ.get("STUB_KERNELS_ATTN_CLASS", "PallasTritonFlashAttention"), "attn_supported": os.environ.get("STUB_KERNELS_SUPPORTED", "1"),
                   "glu_impls": "msc,triton,xla", "glu_chain": "triton,msc,xla", "glu_head": os.environ.get("STUB_KERNELS_GLU_HEAD", "triton"), "tokamax": "0.0.12", "jax": "0.10.2",
                   "backend": os.environ.get("STUB_KERNELS_BACKEND", "gpu"), "device": "NVIDIA_H100_80GB_HBM3", "cc": "9.0",
                   "xla_flags": (os.environ.get("XLA_FLAGS") or "unset").replace(" ", "|"), "prealloc": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "unset"),
                   "mem_fraction": os.environ.get("XLA_CLIENT_MEM_FRACTION", "unset"), "first": "glu", "pid": os.getpid()}
        expect = kp.parse_expect(os.environ["OPT_KERNELS_EXPECT"])
        reasons = kp.verdict(reading, expect)
        fields = {k: reading[k] for k in kp.READING_KEYS}
        fields["expect"] = ";".join(f"{k}={v}" for k, v in expect.items()) or "none"
        fields["verdict"] = ("REFUSED:" + "|".join(x.replace(" ", "_") for x in reasons)) if reasons else "ok"
        sys.stderr.write(kp.format_line("KERNELS-PROBE", fields) + "\\n")
        if os.environ.get("STUB_KERNELS_GLU_FALLBACK"):                       # tokamax's own GLU fallback event (gated_linear_unit/api.py:116)
            sys.stderr.write("E0904 03:00:00.000000 1 api.py:116] Failed to run implementation\\n")
        if reasons:
            sys.stderr.write(f"[af3-jax-opt] KERNELS REFUSED in pid {os.getpid()}: " + "; ".join(reasons) + " — exit 5 before the first timed item\\n"); sys.stderr.flush()
            sys.exit(kp.EXIT_REFUSED)
        dpa = ("obj:PallasTritonFlashAttention(64x64x4x3)" if reading["dpa_site"] == "attncfg" and reading["flag"] == "triton" else reading["flag"])
        sys.stderr.write(kp.format_line("KERNELS-CENSUS", {"pid": os.getpid(), "dpa_calls": f"{dpa}:4", "glu_calls": "None:6", "probe_lines": 1}) + "\\n"); sys.stderr.flush()
    n_samples = int(val("--num_diffusion_samples") or 5)                    # the stock script's default when the flag is absent
    for path in inputs:
        with open(path) as f: item = json.load(f)
        print("Running fold job " + item["name"] + "...")                     # the fork emits this per input (run_alphafold.py:1003) before its per-seed timers
        for s in item.get("modelSeeds", [1]):
            for k in range(n_samples):
                d = os.path.join(out, item["name"], f"seed-{s}_sample-{k}"); os.makedirs(d, exist_ok=True)
                body = "data_" + item["name"] + "\\n_ma_model_list.model_group_name 'AlphaFold-beta " + time.strftime("%Y%m%d%H%M%S%f") + "'\\n_atom_site.id 1\\n"
                open(os.path.join(d, item["name"] + "_model.cif"), "w").write(body)
                open(os.path.join(d, item["name"] + "_confidences.json"), "w").write(json.dumps({"seed": s, "sample": k, "atom_plddts": [90.0]}))
                open(os.path.join(d, item["name"] + "_summary_confidences.json"), "w").write(json.dumps({"iptm": 0.9, "ptm": 0.9, "ranking_score": 0.9}))
            print(f"Running model inference with seed {s} took 1.23 seconds.")
    sys.exit(0)
    ''')


def stock_files():
    """{relative path: absolute path under stock/src} for the stock files of stock/PINS.json."""
    pins = json.load(open(os.path.join(TREE, "stock", "PINS.json"), encoding="utf-8"))
    return {rel: os.path.join(STOCK_SRC, rel) for rel in pins["stock_files"]}


class Box:
    def __init__(self, root: str, monkeypatch, install: str = "stock", pinned_params: bool = True):
        self.root = root
        self.repo = os.path.join(root, "repo"); os.makedirs(self.repo)
        self.site = os.path.join(root, "site", "alphafold3"); os.makedirs(self.site)
        self.params_root = os.path.join(root, "params")
        self.cache_root = os.path.join(root, "cache")
        self.record = os.path.join(root, "stub_record.json")
        self.py = os.path.join(root, "python")
        with open(self.py, "w") as f:
            f.write(STUB)
        os.chmod(self.py, os.stat(self.py).st_mode | stat.S_IEXEC)
        from af3_jax_opt import modes, stack
        for rel, src in stock_files().items():                                 # the install: the archive's copies of the pinned stock files
            dst = os.path.join(self.site, rel[len(stack.SRC_PREFIX):]) if rel.startswith(stack.SRC_PREFIX) else os.path.join(self.repo, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
        if install == "missing":                                               # one stock file absent from the install (not run_alphafold.py:
            os.remove(os.path.join(self.repo, "convert_of3_weights.py"))       # its own absence is a separate, earlier gate — STOCK_SCRIPT)
        self.bin = os.path.join(root, "bin"); os.makedirs(self.bin)
        smi = os.path.join(self.bin, "nvidia-smi")
        with open(smi, "w") as f:
            f.write(f"#!/bin/sh\necho '{GPU_H100['name']}, {GPU_H100['memory_mib']}, {GPU_H100['compute_cap']}'\n")
        os.chmod(smi, os.stat(smi).st_mode | stat.S_IEXEC)
        python_on_path(self.bin, venv_a_python())                                       # run.sh's `python`: venv A — the package and its hook in site
        d = os.path.join(self.params_root, "p2"); os.makedirs(d)
        w = os.path.join(d, "of3_ported_weights.bin.zst")
        with open(w, "wb") as f:
            f.write(b"\x28\xb5\x2f\xfd" + b"p2")
        if pinned_params:                                                      # this process's hasher (variants._hasher) says the fake file hashes as the pin — in-process only; a child process digests the fake bytes as what they are (NOT PINNED, and runs) unless the memo entry this process wrote serves it
            from af3_jax_opt import digest_memo, variants
            pinned = json.load(open(os.path.join(TREE, "stock", "PINS.json"), encoding="utf-8"))["variants"]["p2"]["converted"]["sha256"]
            real_w = os.path.realpath(w)
            monkeypatch.setattr(variants, "_hasher", lambda p: pinned if os.path.realpath(p) == real_w else digest_memo.sha256_file(p))
        from af3_jax_opt import variants as _v
        monkeypatch.setattr(_v, "_DIGESTS", {})                                # a fresh in-process digest memo per box
        for k in ("AF3_JAX_OPT", "AF3_JAX_VARIANT", "AF3_JAX_OPT_HOME", "MODEL_OPT", "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_CLIENT_MEM_FRACTION",
                  "AF3P_GLU_T", "AF3P_ATTN_CFG", "AF3P_PAIR_CHUNK", "AF3_FLASHPAIRFORMER", "AF3_DIFFUSION_HOIST"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("AF3_JAX_REPO", self.repo)
        monkeypatch.setenv("AF3_JAX_PY", self.py)
        monkeypatch.setenv("AF3_JAX_PARAMS_ROOT", self.params_root)
        monkeypatch.setenv("AF3_JAX_CACHE_ROOT", self.cache_root)
        monkeypatch.setenv("STUB_SITE", self.site)
        self.upstream_default_cache = os.path.join(self.root, "upstream_default_cache")   # stands in for the fork's own --cache_dir default on this box:
        os.makedirs(self.upstream_default_cache, exist_ok=True)                       # these tests may run on a GPU box too and must never write the real one
        monkeypatch.setenv("STUB_DEFAULT_CACHE", self.upstream_default_cache)          # the stub caches there when no --cache_dir is on its line
        monkeypatch.setenv("STUB_RECORD", self.record)
        monkeypatch.setenv("STUB_FREEZE", os.path.join(TREE, "environment", "requirements.lock"))
        for k in ("STUB_FAIL", "STUB_FPF_FALLBACK", "STUB_FPF_PARTIAL", "STUB_FPF_HOIST", "STUB_FPF_TILES", "STUB_DATTN", "STUB_DATTN_ASIDE", "STUB_NO_ROW_LINES",
                  "STUB_KERNELS", "STUB_KERNELS_DOWNGRADE", "STUB_KERNELS_FLAG", "STUB_KERNELS_TRIMUL", "STUB_KERNELS_ATTN_CLASS", "STUB_KERNELS_SUPPORTED",
                  "STUB_KERNELS_GLU_HEAD", "STUB_KERNELS_BACKEND", "STUB_KERNELS_GLU_FALLBACK"):
            monkeypatch.delenv(k, raising=False)

    def key(self):
        from af3_jax_opt import stack
        return stack.cache_key(GPU_H100, BUILD)

    def cache_dir(self, mode: str = "off"):
        from af3_jax_opt import modes
        return os.path.join(self.cache_root, self.key() + modes.cache_class_of(mode))

    def warm_cache(self, variant: str = "p2", mode: str = "exact"):
        """A cache as `warm` leaves it for `mode`: the class directory with the autotune table."""
        d = self.cache_dir(mode); os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "tokamax_autotune.json"), "w") as f:
            f.write("{}\n")
        return d

    def run_sh(self, *args: str, env: dict = None, config: bool = True, unset: tuple = ()) -> tuple:
        """Drive the tree's run.sh as a caller would: venv A of this interpreter as `python` (the package and its start-up hook in site), the
        fake nvidia-smi first on PATH, the box's paths in the environment, the box root as the working directory (a caller's shell, never the
        package directory: `python -c "import af3_jax_opt"` must answer from the interpreter's site, not from the cwd); returns (rc, stdout, stderr)."""
        import subprocess
        e = {k: v for k, v in os.environ.items() if not k.startswith("STUB_FAIL")}
        e["PATH"] = self.bin + os.pathsep + e.get("PATH", "")
        e["PYTHONPATH"] = CHILD_PYTHONPATH
        e.pop("MODEL_OPT", None)
        e.update(env or {})
        for k in unset:                                                  # a deployment variable the caller leaves unset (the config must not fill it)
            e.pop(k, None)
        cmd = ["bash", os.path.join(TREE, "run.sh"), *(("--config", "h100") if config else ()), *args]
        p = subprocess.run(cmd, env=e, capture_output=True, text=True, timeout=300, cwd=self.root)
        return p.returncode, p.stdout, p.stderr

    def stub_record(self) -> dict:
        with open(self.record) as f:
            return json.load(f)

    def input_json(self, name="item1", seeds=(1, 2)):
        item = render(name, [chain("A", "MKTAYIAKQR")], seeds=seeds)
        p = os.path.join(self.root, f"{name}.json")
        with open(p, "w") as f:
            json.dump(item, f)
        return p


def _reset_stack():
    from af3_jax_opt import stack
    stack._REPORT = None
    stack._LAUNCHED.clear()
    stack._CACHE.clear()


@pytest.fixture
def box(tmp_path, monkeypatch):
    from af3_jax_opt import stack
    _reset_stack()
    b = Box(str(tmp_path), monkeypatch)
    monkeypatch.setattr(stack, "gpu_info", lambda index=0: dict(GPU_H100))
    yield b
    _reset_stack()


@pytest.fixture
def missing_stock_box(tmp_path, monkeypatch):
    """A box whose install is missing one stock file (convert_of3_weights.py)."""
    from af3_jax_opt import stack
    _reset_stack()
    b = Box(str(tmp_path), monkeypatch, install="missing")
    monkeypatch.setattr(stack, "gpu_info", lambda index=0: dict(GPU_H100))
    yield b
    _reset_stack()


@pytest.fixture
def unpinned_box(tmp_path, monkeypatch):
    """A box whose parameters file hashes as what it is (the fake bytes): its digest is not the pin's."""
    from af3_jax_opt import stack
    _reset_stack()
    b = Box(str(tmp_path), monkeypatch, pinned_params=False)
    monkeypatch.setattr(stack, "gpu_info", lambda index=0: dict(GPU_H100))
    yield b
    _reset_stack()


@pytest.fixture
def no_gpu(monkeypatch):
    from af3_jax_opt import stack
    monkeypatch.setattr(stack, "gpu_info", lambda index=0: None)
