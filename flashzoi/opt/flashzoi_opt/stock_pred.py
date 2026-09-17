#!/usr/bin/env python3
"""stock_pred.py — the ONE stock caller of the tree: the documented route of borzoi-pytorch, standalone (stdlib + numpy + torch +
borzoi_pytorch; it sits in the package's directory but imports nothing of flashzoi_opt or the kit, and drops its own directory from
sys.path first — the package's `pred --mode off` runs this script by path in a clean subprocess and forwards its lines and exit code).

    python flashzoi/opt/flashzoi_opt/stock_pred.py --input <dir of <item>.npy> --out <dir> [--det] [--items a,b] [--tracks SPEC]

The documented route (the README's call, under torch's own numerics — no switch set), verbatim:

    from borzoi_pytorch import Borzoi
    from borzoi_pytorch.pytorch_borzoi_helpers import predict_tracks
    models = [Borzoi.from_pretrained(f"johahi/flashzoi-replicate-{k}", revision=<pin>).to("cuda").eval() for k in range(4)]
    with torch.autocast("cuda"):
        y = predict_tracks(models, sequence_one_hot, slices)          # slices = every track -> (1, 4, 6144, 7611) float32; --tracks = those indices

Weights: the pinned revisions of the tree's `stock/PINS.json` ("weights": {<repo>: {"revision", "model.safetensors_sha256", "bytes"}},
"config_json_sha256"), resolved offline from the hub cache under HF_HOME (HF_HUB_OFFLINE=1); the resolved model.safetensors of each
replicate is hashed before use and the digest worded against its pin — `pinned` when sha256, byte count and config.json equal the pin,
else `not pinned`, said once per replicate as `weights sha256=<12> NOT PINNED — ...` and loaded either way (only a replicate with no pin, or
a file not in the cache, refuses). Inputs: `<item>.npy` = the (4, 524288) uint8 one-hot with
rows A, C, G, T (values 0/1; an N position is an all-zero column); `--items a,b` selects by stem; `--tracks SPEC` = the `slices` argument:
`all` (every track), or indices `i` / ranges `lo-hi` separated by commas, or `@file` with one such entry per line. Outputs: `<out>/<item>.npy` =
the call's return verbatim (np.save, float32 (1, 4, 6144, n_tracks)), `<out>/rows.jsonl` = one row per item {item, shape, dtype, wall_s (the call's
seconds), ok, error?} written row by row, `<out>/opt_manifest.json` = the ONE run record (mode off, the script, argv, det, numerics read back,
weights words, versions, gpu) carrying the environment proof under `stock_env_proof`.

The environment proof runs BEFORE torch or the model package is imported: no package variable and none of the kit's own switch names (FORBIDDEN_ENV below; CUBLAS_WORKSPACE_CONFIG
only under --det, exported by this process itself), no `engines` / `compare` (the kit's roots) module in sys.modules, no flashzoi_opt module
beyond the installed package's .pth shim (`flashzoi_opt`, `flashzoi_opt._autoload`) and no autoload finder on sys.meta_path, no kit root on
sys.path, the installed versions. A failed proof refuses (exit 3) — nothing stock runs in a doubtful environment.

`--det` = the deterministic recipe (a stock-side switch, never a mode): CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialises, seed 0
(inert: the call samples nothing), torch.use_deterministic_algorithms(True), cudnn.deterministic=True, cudnn.benchmark=False; the TF32
switches untouched (torch's defaults).

Lines (their format strings live HERE, at the top of the script): per item PRED_FMT, at exit EXIT_FMT, plus READY_FMT / PROOF_FMT / REFUSED_FMT.
Exit codes: 0 ok, 1 an item failed, 2 usage, 3 refused (the proof or the pins).
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time

PREFIX = "[flashzoi-stock]"
PRED_FMT = "{prefix} pred {item} s0 samples=1 {wall:.3f}s"
EXIT_FMT = "{prefix} EXIT items={items} ok={ok} failed={failed} wall={wall:.1f}s"
READY_FMT = "{prefix} ready replicates={n} t={t:.2f}s"
PROOF_FMT = "{prefix} stock environment proof {status} file={path}"
MANIFEST = "opt_manifest.json"                                            # the one run record beside the outputs on every route; the stock's carries the proof under stock_env_proof
REFUSED_FMT = "{prefix} REFUSED: {reason}"

HERE = os.path.dirname(os.path.abspath(__file__))                                              # flashzoi/opt/flashzoi_opt — the package's directory: dropped from sys.path, nothing of it importable here
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != HERE]
TREE = os.path.dirname(os.path.dirname(HERE))                                                   # flashzoi/
PINS_PATH = os.path.join(TREE, "stock", "PINS.json")
FORBIDDEN_ENV = ("FLASHZOI_OPT",)                                                                # the package's switch (the mode; the exact name); the kit itself reads no environment switch
DET_ENV, DET_VALUE, DET_ACCEPTED = "CUBLAS_WORKSPACE_CONFIG", ":4096:8", (":4096:8", ":16:8")
KIT_MODULE_ROOTS = ("engines", "compare")                      # the kit's closure roots
PACKAGE_ROOT = "flashzoi_opt"
PTH_SHIM_MODULES = ("flashzoi_opt", "flashzoi_opt._autoload")  # what the installed package's .pth imports at every interpreter start (no finder unless FLASHZOI_OPT is set)
KIT_PATH_MARKER = os.path.join("engines", "flashzoi", "kits")  # a sys.path entry carrying this directory is a kit root
PINNED_DISTS = ("borzoi-pytorch", "torch", "triton", "flash-attn", "transformers", "numpy")
HF_REPO_FMT = "johahi/flashzoi-replicate-{k}"
N_REPLICATES, SEQ_LEN, N_BINS, N_TRACKS = 4, 524288, 6144, 7611
OUTPUT_SHAPE, OUTPUT_DTYPE = (1, N_REPLICATES, N_BINS, N_TRACKS), "float32"
WEIGHTS_FILE, CONFIG_FILE = "model.safetensors", "config.json"
T0 = time.perf_counter()
ITEMS = {"items": 0, "ok": 0, "failed": 0}


def say(line):
    print(line, file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------------------------------------ the environment proof
def _forbidden(environ, spec):
    hits = set()
    for s in spec:
        for k in environ:
            if (k.startswith(s) if s.endswith("_") else k == s):
                hits.add(k)
    return sorted(hits)


def env_proof(det):
    """The proof dict (pass = every check holds); torch and the model package are NOT imported here."""
    env_hits = _forbidden(os.environ, FORBIDDEN_ENV)
    det_env = os.environ.get(DET_ENV)
    kit_mods = sorted(m for m in sys.modules if m.split(".")[0] in KIT_MODULE_ROOTS)
    pkg_mods = sorted(m for m in sys.modules if m == PACKAGE_ROOT or m.startswith(PACKAGE_ROOT + "."))
    pkg_beyond_shim = [m for m in pkg_mods if m not in PTH_SHIM_MODULES]
    finder = [type(f).__name__ for f in sys.meta_path if type(f).__module__.startswith(PACKAGE_ROOT)]
    kit_paths = sorted(p for p in sys.path if os.path.isdir(os.path.join(p or ".", KIT_PATH_MARKER)))
    versions = {}
    for d in PINNED_DISTS:
        try:
            versions[d] = importlib.metadata.version(d)
        except importlib.metadata.PackageNotFoundError:
            versions[d] = None
    reasons = []
    if env_hits:
        reasons.append(f"forbidden environment variables set: {env_hits}")
    if det_env is not None and not det:
        reasons.append(f"{DET_ENV}={det_env!r} is set without --det")
    if kit_mods:
        reasons.append(f"kit modules loaded: {kit_mods[:5]}")
    if pkg_beyond_shim:
        reasons.append(f"flashzoi_opt modules loaded beyond the .pth shim: {pkg_beyond_shim}")
    if finder:
        reasons.append(f"flashzoi_opt finder on sys.meta_path: {finder}")
    if kit_paths:
        reasons.append(f"kit roots on sys.path: {kit_paths}")
    return {"pass": not reasons, "reasons": reasons, "env_checked": list(FORBIDDEN_ENV), "env_hits": env_hits, "det": det, DET_ENV: det_env,
            "kit_modules_loaded": kit_mods, "flashzoi_opt_modules_loaded": pkg_mods, "pth_shim_modules": list(PTH_SHIM_MODULES), "finder_installed": finder,
            "kit_roots_on_sys_path": kit_paths, "sys_path": list(sys.path), "versions": versions, "python": sys.version.split()[0], "executable": sys.executable,
            "argv": list(sys.argv), "pid": os.getpid(), "script": os.path.abspath(__file__), "pins_path": PINS_PATH}


# ------------------------------------------------------------------------------------------------------------ inputs / outputs
def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def list_items(input_dir, only=None):
    if not os.path.isdir(input_dir):
        raise ValueError(f"--input {input_dir}: not a directory of <item>.npy files")
    names = sorted(f[:-4] for f in os.listdir(input_dir) if f.endswith(".npy"))
    if only:
        want = [s.strip() for s in only.split(",") if s.strip()]
        missing = [w for w in want if w not in names]
        if missing:
            raise ValueError(f"--items: not found under --input: {missing}")
        names = [n for n in names if n in want]
    if not names:
        raise ValueError(f"--input {input_dir}: no items")
    return [(n, os.path.join(input_dir, n + ".npy")) for n in names]


def load_onehot(path, np):
    x = np.load(path, allow_pickle=False)
    if x.shape != (4, SEQ_LEN):
        raise ValueError(f"{path}: shape {x.shape} != (4, {SEQ_LEN})")
    if x.dtype == np.bool_:
        x = x.astype(np.uint8)
    if not np.isin(x, (0, 1)).all():
        raise ValueError(f"{path}: values outside {{0, 1}}")
    u = np.ascontiguousarray(x.astype(np.uint8))
    if (u.sum(axis=0) > 1).any():
        raise ValueError(f"{path}: a column with more than one 1 (not a one-hot)")
    return u


def write_row(out_dir, row):
    with open(os.path.join(out_dir, "rows.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=str); fh.write("\n")
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------------------ weights
def resolve_cached(repo, filename, revision):
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(repo_id=repo, filename=filename, revision=revision)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"{repo}@{revision[:12]} {filename} is not in the offline hub cache (HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE')!r}, HF_HOME={os.environ.get('HF_HOME')!r}): {type(e).__name__}: {str(e)[:200]}") from e


WEIGHTS_WORDS = ("pinned", "not pinned")                               # the word recorded with each replicate's digest (opt_manifest.json / the proof "weights")
WEIGHTS_NOTICE = "NOT PINNED — the kit's statements hold for the pinned weights only"


def assert_pins(pins, repo):
    """Hash the pinned revision's files in the cache and word the digests against PINS.json: pinned when sha256, byte count and
    config.json equal the pin, else not pinned with each difference named — either one loads. Refuses (RuntimeError) only when there is
    no pin to resolve or a file is not in the cache."""
    w = (pins.get("weights") or {}).get(repo) or {}
    if not w.get("revision") or not w.get("model.safetensors_sha256"):
        raise RuntimeError(f"{PINS_PATH}: no complete weights pin for {repo} (revision + model.safetensors_sha256)")
    rev = w["revision"]
    p = resolve_cached(repo, WEIGHTS_FILE, rev)
    n = os.path.getsize(p)
    sha = sha256_file(p)
    rec = {"repo": repo, "revision": rev, "model.safetensors": os.path.realpath(p), "model.safetensors_sha256": sha, "pinned_sha256": w["model.safetensors_sha256"], "bytes": n}
    differs = []
    if w.get("bytes") is not None and int(w["bytes"]) != n:
        differs.append(f"{WEIGHTS_FILE}: {n} bytes != pinned {w['bytes']}")
    if sha != w["model.safetensors_sha256"]:
        differs.append(f"{WEIGHTS_FILE}: sha256 {sha[:16]} != pinned {w['model.safetensors_sha256'][:16]}")
    if pins.get("config_json_sha256"):
        c = resolve_cached(repo, CONFIG_FILE, rev)
        csha = sha256_file(c)
        rec["config.json_sha256"] = csha
        if csha != pins["config_json_sha256"]:
            differs.append(f"{CONFIG_FILE}: sha256 {csha[:16]} != pinned {pins['config_json_sha256'][:16]}")
    rec["differs"] = differs
    rec["word"] = WEIGHTS_WORDS[1] if differs else WEIGHTS_WORDS[0]
    if differs:
        say(f"{PREFIX} weights sha256={sha[:12]} {WEIGHTS_NOTICE} ({repo}@{rev[:12]} " + "; ".join(differs) + ")")
    else:
        say(f"{PREFIX} weights={repo}@{rev[:12]} sha256={sha[:12]} (pinned)")
    return rec


def load_models(pins, device, torch, Borzoi):
    """The four replicates in order 0..3: each digest worded against its pin, then the documented load at the pinned revision, .to(device).eval()."""
    models, recs = [], []
    for k in range(N_REPLICATES):
        repo = HF_REPO_FMT.format(k=k)
        rec = assert_pins(pins, repo)
        t0 = time.perf_counter()
        m = Borzoi.from_pretrained(repo, revision=rec["revision"]).to(device).eval()       # the documented load, at the pinned revision
        rec.update(replicate=k, load_s=round(time.perf_counter() - t0, 3))
        models.append(m); recs.append(rec)
    return models, recs


# ------------------------------------------------------------------------------------------------------------ numerics
def numerics_default(torch):
    """The stock's numerics: torch's own TF32 defaults, never set (cuDNN TF32 on, matmul TF32 off — read back into the manifest);
    the other switches are the stock defaults, set explicitly."""
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = False          # the stock defaults, set explicitly
    torch.use_deterministic_algorithms(False)


def numerics_det(torch):
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False       # TF32 untouched: torch's defaults under every recipe


def read_back(torch):
    return {"matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32), "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark), "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()), "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
            "autocast_gpu_dtype": str(torch.get_autocast_dtype("cuda")) if hasattr(torch, "get_autocast_dtype") else None, DET_ENV: os.environ.get(DET_ENV)}


# ------------------------------------------------------------------------------------------------------------ main
def parse_tracks(spec):
    """--tracks: `all` -> slice(None) (every track); else the sorted, de-duplicated int64 indices named by comma-separated `i` / `lo-hi`
    (inclusive) entries, or by `@file` with one entry per line (`#` lines and blanks skipped); an index outside 0..N_TRACKS-1, an empty or
    malformed spec is ValueError (exit 2). The result is the documented call's `slices` argument."""
    spec = (spec or "all").strip()
    if spec == "all":
        return slice(None)
    if spec.startswith("@"):
        with open(spec[1:], encoding="utf-8") as fh:
            entries = [ln.strip() for ln in fh if ln.strip() and not ln.strip().startswith("#")]
    else:
        entries = [e.strip() for e in spec.split(",") if e.strip()]
    idx = set()
    for e in entries:
        lo, sep, hi = e.partition("-")
        try:
            a, b = (int(lo), int(hi)) if sep else (int(e), int(e))
        except ValueError:
            raise ValueError(f"--tracks: {e!r} is not an index or a lo-hi range") from None
        if a > b or a < 0 or b >= N_TRACKS:
            raise ValueError(f"--tracks: {e!r} is outside 0..{N_TRACKS - 1}")
        idx.update(range(a, b + 1))
    if not idx:
        raise ValueError("--tracks: no track named")
    return sorted(idx)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python flashzoi/opt/flashzoi_opt/stock_pred.py", description="the stock caller: the documented borzoi-pytorch route, standalone")
    ap.add_argument("--input", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--det", action="store_true"); ap.add_argument("--items", default=None)
    ap.add_argument("--tracks", default="all", help="the documented call's `slices`: all | i,lo-hi,... | @file")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    os.makedirs(a.out, exist_ok=True)
    if a.det and os.environ.get(DET_ENV) not in DET_ACCEPTED:
        os.environ[DET_ENV] = DET_VALUE                                                     # before CUDA initialises
    proof = env_proof(a.det)
    try:
        tracks = parse_tracks(a.tracks)
    except (ValueError, OSError) as e:
        say(f"{PREFIX} usage: {e}"); return 2
    n_tracks = N_TRACKS if isinstance(tracks, slice) else len(tracks)
    ppath = os.path.join(a.out, MANIFEST); record = {"mode": "off", "route": "opt/flashzoi_opt/stock_pred.py", "script": os.path.abspath(__file__), "argv": list(sys.argv), "det": a.det,
                                                      "tracks": {"spec": a.tracks, "n": n_tracks}, "stock_env_proof": proof}
    write_json(ppath, record)                                                                 # the ONE run record beside the outputs (the proof inside it), rewritten below with the run's keys
    say(PROOF_FMT.format(prefix=PREFIX, status="PASS" if proof["pass"] else "FAIL", path=ppath))
    if not proof["pass"]:
        say(REFUSED_FMT.format(prefix=PREFIX, reason="; ".join(proof["reasons"])))
        say(EXIT_FMT.format(prefix=PREFIX, wall=time.perf_counter() - T0, **ITEMS))
        return 3
    try:
        items = list_items(a.input, a.items)
    except ValueError as e:
        say(f"{PREFIX} usage: {e}"); return 2
    import numpy as np
    import torch
    from borzoi_pytorch import Borzoi                                                        # the documented imports
    from borzoi_pytorch.pytorch_borzoi_helpers import predict_tracks
    numerics_default(torch)
    if a.det:
        numerics_det(torch)
    pins = json.load(open(PINS_PATH, encoding="utf-8"))
    t0 = time.perf_counter()
    try:
        models, wrecs = load_models(pins, "cuda", torch, Borzoi)
    except Exception as e:  # noqa: BLE001
        say(REFUSED_FMT.format(prefix=PREFIX, reason=f"stock load failed: {type(e).__name__}: {str(e)[:300]}"))
        say(EXIT_FMT.format(prefix=PREFIX, wall=time.perf_counter() - T0, **ITEMS))
        return 3
    torch.cuda.synchronize()
    say(READY_FMT.format(prefix=PREFIX, n=len(models), t=time.perf_counter() - t0))
    write_json(ppath, {**record, "device": "cuda",
                                                        "numerics_read_back": read_back(torch), "weights": wrecs, "versions": proof["versions"], "python": proof["python"],
                                                        "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None), "pins_path": PINS_PATH})
    slices = tracks if isinstance(tracks, slice) else np.asarray(tracks, dtype=np.int64)         # every track, or the --tracks indices
    expected = OUTPUT_SHAPE[:3] + (n_tracks,)
    rc = 0
    for item, path in items:
        ITEMS["items"] += 1
        try:
            x = torch.from_numpy(load_onehot(path, np)).to("cuda").float()                     # the (4, 524288) one-hot on the device, float32
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            with torch.autocast("cuda"):                                                      # THE DOCUMENTED CALL
                y = predict_tracks(models, x, slices)
            wall = time.perf_counter() - t1                                                    # the call incl. host materialisation (predict_tracks returns numpy)
            y = np.ascontiguousarray(y)
            if tuple(y.shape) != expected or str(y.dtype) != OUTPUT_DTYPE:
                raise RuntimeError(f"documented call returned {y.shape} {y.dtype}; expected {expected} {OUTPUT_DTYPE}")
            p = os.path.join(a.out, item + ".npy"); np.save(p + ".tmp.npy", y); os.replace(p + ".tmp.npy", p)
            row = {"item": item, "shape": list(y.shape), "dtype": str(y.dtype), "wall_s": round(wall, 4), "ok": True}
            write_row(a.out, row)
            say(PRED_FMT.format(prefix=PREFIX, item=item, wall=wall))
            ITEMS["ok"] += 1
            del y
        except Exception as e:  # noqa: BLE001 — a named row, never a missing one
            write_row(a.out, {"item": item, "ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"})
            ITEMS["failed"] += 1; rc = 1
    say(EXIT_FMT.format(prefix=PREFIX, wall=time.perf_counter() - T0, **ITEMS))
    return rc


if __name__ == "__main__":
    sys.exit(main())
