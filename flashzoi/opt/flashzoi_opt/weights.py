"""The ONE weights loader (the stock caller and the kit driver both call it): upstream's `Borzoi.from_pretrained(repo,
revision=<pin>)` from the offline hub cache (HF_HOME, HF_HUB_OFFLINE=1 — configs/h100.env), the pins read from stock/PINS.json
("weights": {<repo>: {"revision", "model.safetensors_sha256", "bytes"}}, "config_json_sha256"). Before a checkpoint is used, the
resolved `model.safetensors` of the pinned revision is hashed and the digest (with its byte count and config.json's sha256) is worded
against the pin: `pinned` when all equal the pin, else `not pinned` — said once per checkpoint on the weights line, recorded with
the digest, and loaded either way (WEIGHTS_WORDS, WEIGHTS_NOTICE). Only a repository with no pin to resolve or a file that is not in
the cache refuses, by name (WeightsError). Replicates load in the order asked (0..3), each `.to(device).eval()`.

`run.sh install --weights DIR` is this module run as a program (`python -m flashzoi_opt.weights DIR`, fetch() below): the one step that
fetches — each pinned replicate's config.json and model.safetensors into the hub cache DIR with huggingface_hub's `hf_hub_download` at the
pinned revision (the transfer `Borzoi.from_pretrained` itself resolves its files through), then the kit's one pin check over DIR
(stock/check_pins.py's weights check: byte count, sha256 of model.safetensors and of config.json per replicate). DIR is then the
FLASHZOI_WEIGHTS every route reads offline; a file already in DIR is kept and only checked; a digest off its pin is named and refused
(exit 1) with the file left in place.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from typing import Callable, List, Optional

from . import report as _report
from . import settings as _settings

WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"
WEIGHTS_WORDS = ("pinned", "not pinned")                               # the word recorded with each checkpoint's digest (opt_manifest.json "weights"[k]["word"])
WEIGHTS_NOTICE = "NOT PINNED — the kit's statements hold for the pinned weights only"   # the one notice for a digest that is not the pin's


class WeightsError(RuntimeError):
    """A checkpoint cannot be loaded: stock/PINS.json has no pin to resolve for the repository, or the pinned revision's file is not in
    the offline cache."""


def sha256_file(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def pin_for(pins: dict, repo: str) -> dict:
    w = (pins.get("weights") or {}).get(repo)
    if not w or not w.get("revision") or not w.get("model.safetensors_sha256"):
        raise WeightsError(f"stock/PINS.json has no complete weights pin for {repo} (revision + model.safetensors_sha256 needed)")
    return w


def resolve_file(repo: str, filename: str, revision: str) -> str:
    """The cached file of the pinned revision (offline resolution through huggingface_hub; nothing is fetched)."""
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(repo_id=repo, filename=filename, revision=revision)
    except Exception as e:  # noqa: BLE001
        raise WeightsError(f"{repo}@{revision[:12]} {filename} is not in the offline hub cache (HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE')!r}, HF_HOME={os.environ.get('HF_HOME')!r}): {type(e).__name__}: {str(e)[:200]}") from e


def assert_pins(pins: dict, repo: str) -> dict:
    """Hash the pinned revision's model.safetensors (and config.json) in the cache and word the digests against stock/PINS.json:
    ``pinned`` when sha256, byte count and config.json equal the pin, else ``not pinned`` with each difference named in
    ``differs`` — either one loads. Returns the record; WeightsError only when there is no pin to resolve or a file is not in the cache."""
    w = pin_for(pins, repo)
    rev = w["revision"]
    p = resolve_file(repo, WEIGHTS_FILE, rev)
    n = os.path.getsize(p)
    t0 = time.perf_counter()
    sha = sha256_file(p)
    rec = {"repo": repo, "revision": rev, "model.safetensors": os.path.realpath(p), "model.safetensors_sha256": sha, "pinned_sha256": w["model.safetensors_sha256"],
           "bytes": n, "hash_s": round(time.perf_counter() - t0, 3)}
    differs = []
    if w.get("bytes") is not None and int(w["bytes"]) != n:
        differs.append(f"{WEIGHTS_FILE}: {n} bytes != pinned {w['bytes']}")
    if sha != w["model.safetensors_sha256"]:
        differs.append(f"{WEIGHTS_FILE}: sha256 {sha[:16]} != pinned {w['model.safetensors_sha256'][:16]}")
    cfg_pin = pins.get("config_json_sha256")
    if cfg_pin:
        c = resolve_file(repo, CONFIG_FILE, rev)
        csha = sha256_file(c)
        rec["config.json_sha256"] = csha
        if csha != cfg_pin:
            differs.append(f"{CONFIG_FILE}: sha256 {csha[:16]} != pinned {cfg_pin[:16]}")
    rec["differs"] = differs
    rec["word"] = WEIGHTS_WORDS[1] if differs else WEIGHTS_WORDS[0]
    return rec


def weights_words(rec: dict) -> str:
    """The weights clause, one per checkpoint: ``weights=<repo>@<rev12> sha256=<12> (pinned)`` for the pinned digest, else
    ``weights sha256=<12> NOT PINNED — the kit's statements hold for the pinned weights only (<repo>@<rev12> <each difference>)``."""
    s = str(rec.get("model.safetensors_sha256"))[:12]
    name = f"{rec.get('repo')}@{str(rec.get('revision'))[:12]}"
    if rec.get("word") == WEIGHTS_WORDS[0]:
        return f"weights={name} sha256={s} (pinned)"
    return f"weights sha256={s} {WEIGHTS_NOTICE} ({name} " + "; ".join(rec.get("differs") or []) + ")"


def say_weights(rec: dict, stream=None) -> str:
    """Print the weights clause once on stderr with the package's prefix; returns the line."""
    return _report.emit(f"{_report.PREFIX} {weights_words(rec)}", stream)


def load_replicate(pins: dict, k: int, device: str = "cuda") -> tuple:
    """One replicate: its digest worded against the pin (the weights line), then upstream's from_pretrained at the pinned revision,
    .to(device).eval(). Returns (model, record)."""
    from borzoi_pytorch import Borzoi                                       # the documented import
    repo = _settings.HF_REPO_FMT.format(k=k)
    rec = assert_pins(pins, repo)
    say_weights(rec)                                                       # the pinned digest or another: both load, the line says which
    t0 = time.perf_counter()
    model = Borzoi.from_pretrained(repo, revision=rec["revision"]).to(device).eval()      # the documented load, at the pinned revision
    rec["load_s"] = round(time.perf_counter() - t0, 3)
    rec["replicate"] = k
    return model, rec


def load_replicates(pins: dict, replicates=None, device: str = "cuda", after_each=None) -> tuple:
    """The replicates in order (default 0..3); `after_each(model, record)` runs per model right after its load (the kit driver
    attaches there). Returns (models, records)."""
    reps = list(replicates if replicates is not None else _settings.DEFAULT.replicates)
    models: List = []
    records: List[dict] = []
    for k in reps:
        m, rec = load_replicate(pins, k, device)
        if after_each is not None:
            after_each(m, rec)
        models.append(m); records.append(rec)
    return models, records


# --------------------------------------------------------------------------------------------- run.sh install --weights DIR (the one fetch)
INSTALL_PREFIX = "[flashzoi-opt install]"
EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2
FETCHED_FILES = (CONFIG_FILE, WEIGHTS_FILE)                             # per replicate, in the order Borzoi.from_pretrained resolves them
_OFFLINE_TRUE = ("1", "ON", "YES", "TRUE")                              # huggingface_hub's truth values for HF_HUB_OFFLINE


def snapshot_file(hub_cache: str, repo: str, revision: str, filename: str) -> str:
    """Where the hub cache DIR holds `filename` of `repo` at `revision`: DIR/models--<org>--<name>/snapshots/<revision>/<filename> — the
    layout huggingface_hub writes and stock/check_pins.py --weights reads."""
    return os.path.join(hub_cache, "models--" + repo.replace("/", "--"), "snapshots", revision, filename)


def hub_downloader(hub_cache: str) -> Callable[[str, str, str], str]:
    """`download(repo, filename, revision)` -> the cached path: huggingface_hub's own `hf_hub_download(repo_id, filename, revision=…,
    cache_dir=DIR)` — it writes the blob under DIR/models--…/blobs and links snapshots/<revision>/<filename> to it, returns a file that is
    already there without a transfer, and keeps an interrupted transfer as a *.incomplete blob that the next run resumes. Refused by name
    when HF_HUB_OFFLINE is set in this environment (huggingface_hub would refuse the transfer itself, less legibly)."""
    if (os.environ.get("HF_HUB_OFFLINE") or "").strip().upper() in _OFFLINE_TRUE:
        raise RuntimeError(f"HF_HUB_OFFLINE={os.environ['HF_HUB_OFFLINE']} is set in this environment and huggingface_hub refuses transfers under it "
                           "— unset it for this step (nothing else in the kit fetches; the routes set it themselves at run time)")
    from huggingface_hub import hf_hub_download

    def download(repo: str, filename: str, revision: str) -> str:
        return hf_hub_download(repo_id=repo, filename=filename, revision=revision, cache_dir=hub_cache)
    return download


def pin_checker() -> Callable[[dict, Optional[str]], tuple]:
    """stock/check_pins.py's `check_weights(pins, hf_home)` — the kit's one pin check (`python stock/check_pins.py --weights`), loaded from
    this tree (stack.tree_home()); it reads the hub cache from HF_HUB_CACHE."""
    import importlib.util
    from . import stack as _stack
    path = os.path.join(_stack.tree_home(), "stock", "check_pins.py")
    spec = importlib.util.spec_from_file_location("flashzoi_stock_check_pins", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"stock/check_pins.py not found at {path} (set MODEL_OPT to the flashzoi/ directory)")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.check_weights


def fetch(hub_cache: str, pins: Optional[dict] = None, download: Optional[Callable[[str, str, str], str]] = None,
          checker: Optional[Callable[[dict, Optional[str]], tuple]] = None, out=None) -> int:
    """Fetch what is absent into the hub cache DIR, then check every replicate against stock/PINS.json. `pins` (stock/PINS.json), `download`
    (huggingface_hub's transfer) and `checker` (stock/check_pins.py's weights check) are injectable for the package tests."""
    out = out or sys.stdout
    hub_cache = os.path.abspath(hub_cache)
    os.makedirs(hub_cache, exist_ok=True)
    os.environ["HF_HUB_CACHE"] = hub_cache                               # what configs/<gpu>.env exports from FLASHZOI_WEIGHTS at run time; the check reads it
    if pins is None:
        from . import stack as _stack
        pins = _stack.read_pins()
    weights = pins.get("weights") or {}
    if not weights:
        print(f"{INSTALL_PREFIX} FAILED: stock/PINS.json pins no weights ('weights' is empty)", file=out, flush=True)
        return EXIT_FAIL
    for repo, w in weights.items():
        rev = pin_for(pins, repo)["revision"]
        for filename in FETCHED_FILES:
            target = snapshot_file(hub_cache, repo, rev, filename)
            state = "present" if os.path.isfile(target) else "fetching"
            size = f"{w.get('bytes', '?')} bytes, " if filename == WEIGHTS_FILE else ""
            print(f"{INSTALL_PREFIX} {repo}@{rev[:12]} {filename}: {state}" + (f" ({size}huggingface_hub hf_hub_download)" if state == "fetching" else ""), file=out, flush=True)
            if state == "present":
                continue
            try:
                if download is None:
                    download = hub_downloader(hub_cache)
                download(repo, filename, rev)
            except Exception as e:  # noqa: BLE001 — huggingface_hub's own error (network, disk, the repository, an offline switch), relayed by name; nothing is deleted
                print(f"{INSTALL_PREFIX} FAILED fetching {repo}@{rev[:12]} {filename}: {type(e).__name__}: {str(e)[:400]} — a partial transfer stays under "
                      f"{os.path.join(hub_cache, 'models--' + repo.replace('/', '--'), 'blobs')} as *.incomplete and resumes on the next run", file=out, flush=True)
                return EXIT_FAIL
            if not os.path.isfile(target):
                print(f"{INSTALL_PREFIX} FAILED: {filename} of {repo}@{rev[:12]} is not at {target} after the transfer", file=out, flush=True)
                return EXIT_FAIL
    if checker is None:
        checker = pin_checker()
    t0 = time.perf_counter()
    bad, detail = checker(pins, None)
    seconds = round(time.perf_counter() - t0, 1)
    for repo in weights:
        d = detail.get(repo) if isinstance(detail, dict) else None
        if isinstance(d, dict) and d.get("pinned"):
            print(f"{INSTALL_PREFIX} {repo}: pinned ({d.get('snapshot')})", file=out, flush=True)
    for b in bad:
        print(f"{INSTALL_PREFIX} {b}", file=out, flush=True)
    if bad:
        print(f"{INSTALL_PREFIX} REFUSED: {len(bad)} finding(s) above — {sum(1 for r in weights if not (isinstance(detail.get(r), dict) and detail[r].get('pinned')))} of "
              f"{len(weights)} replicates under {hub_cache} are not at their pins (stock/PINS.json 'weights', 'config_json_sha256'); the files are left in place — "
              f"remove the named snapshot (and its blob under blobs/) and re-run this step to fetch afresh", file=out, flush=True)
        return EXIT_FAIL
    print(f"{INSTALL_PREFIX} WEIGHTS OK: {len(weights)}/{len(weights)} replicates ({len(weights) * len(FETCHED_FILES)} files) under {hub_cache} have the pinned digests "
          f"({seconds} s hashing) — export FLASHZOI_WEIGHTS={hub_cache}", file=out, flush=True)
    return EXIT_OK


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1 or argv[0].startswith("-"):
        print("usage: python -m flashzoi_opt.weights DIR   (run.sh install --weights DIR: DIR = the hub cache FLASHZOI_WEIGHTS names)", file=sys.stderr)
        return EXIT_USAGE
    return fetch(argv[0])


if __name__ == "__main__":
    sys.exit(main())
