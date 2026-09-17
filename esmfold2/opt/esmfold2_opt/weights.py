"""`run.sh install --weights DIR` — fetch the pinned weight files of stock ESMFold2 into DIR with Hugging Face's own downloader and check
each one against ``stock/PINS.json`` ``weights`` (sha256). DIR is then the ``HF_HOME`` every route reads (README.md 'Setup'): the hub
cache holding the three pinned snapshots — ``biohub/ESMFold2``, ``biohub/ESMFold2-Fast`` and ``biohub/ESMC-6B`` (~27 GB in sixteen files: the checkpoints plus the config, index and tokenizer files the loaders read);
nothing is fetched at run time.

Upstream owns the transfer: the library loads every file through ``huggingface_hub`` (``from_pretrained(<repo>)`` and ``hf_hub_download``),
so ``huggingface_hub.hf_hub_download(repo_id, filename, revision=<the pinned snapshot commit>, cache_dir=DIR/hub)`` writes exactly the cache
entry the library reads offline — ``hub/models--<org>--<name>/snapshots/<commit>/<file>`` over ``blobs/`` — resumes an interrupted transfer
and never rewrites a file that is already there. The files come at the PINNED commit, not at the repository's current ``main``; the one
thing written besides them is each repository's ``refs/main`` (a 40-character pointer file of the cache layout), set to the pinned commit so
the library's offline ``from_pretrained(<repo>)`` — which asks for revision ``main`` — resolves to that snapshot. ``HF_HUB_OFFLINE`` /
``TRANSFORMERS_OFFLINE`` are lifted for this step only. A file whose digest is not the pin's is named and the step fails (exit 1) — the
file is left in place for inspection, never deleted. The comparison is ``stack.weight_files_check`` — the one the routes' activation line
reports — run afresh over every pinned file. ``python -m esmfold2_opt.weights DIR``.
"""
from __future__ import annotations

import os
import sys
from typing import Callable, List, Optional, Tuple

PREFIX = "[esmfold2-opt install]"
EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2
REF_NAME = "main"                                                    # the revision the library asks for (from_pretrained(<repo>) names no revision)


def repo_dir(hf_home: str, repo: str) -> str:
    """The hub cache directory of ``repo`` under ``hf_home`` (huggingface_hub's layout: hub/models--<org>--<name>)."""
    return os.path.join(hf_home, "hub", "models--" + repo.replace("/", "--"))


def pinned_files(pins: dict) -> List[Tuple[str, str, str, int]]:
    """(repo, snapshot commit, file name, size_bytes) of every pinned weight file, every repository (stock/PINS.json ``weights``)."""
    out = []
    for repo in sorted(pins["weights"]):
        w = pins["weights"][repo]
        for name in sorted(w["files"]):
            out.append((repo, w["snapshot_commit"], name, int(w["files"][name]["size_bytes"])))
    return out


def upstream_fetch(hf_home: str) -> Callable[[str, str, str], str]:
    """``fetch(repo, commit, name)`` → the local path: huggingface_hub's downloader into ``hf_home``'s hub cache at the pinned commit. Bound
    after the offline switches are lifted: ``huggingface_hub`` reads them when it is imported."""
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.pop(var, None)
    os.environ["HF_HOME"] = hf_home
    from huggingface_hub import hf_hub_download
    cache = os.path.join(hf_home, "hub")
    return lambda repo, commit, name: hf_hub_download(repo_id=repo, filename=name, revision=commit, cache_dir=cache)


def point_ref(hf_home: str, repo: str, commit: str) -> Optional[str]:
    """Set the cache's ``refs/main`` of ``repo`` to ``commit``; returns the previous pointer (None when there was none, ``commit`` when unchanged)."""
    ref = os.path.join(repo_dir(hf_home, repo), "refs", REF_NAME)
    prev = None
    if os.path.isfile(ref):
        with open(ref, encoding="utf-8") as fh:
            prev = fh.read().strip()
    if prev != commit:
        os.makedirs(os.path.dirname(ref), exist_ok=True)
        tmp = f"{ref}.tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(commit)
        os.replace(tmp, ref)
    return prev


def install_weights(hf_home: str, fetch: Optional[Callable[[str, str, str], str]] = None, pins: Optional[dict] = None, log=print) -> int:
    from . import stack
    pins = stack.pins() if pins is None else pins
    hf_home = os.path.abspath(hf_home)
    os.makedirs(hf_home, exist_ok=True)
    files = pinned_files(pins)
    total = sum(size for *_, size in files)
    log(f"{PREFIX} weights: {len(files)} pinned files of {len(pins['weights'])} repositories ({total / 1e9:.1f} GB) into HF_HOME={hf_home} (stock/PINS.json \"weights\")")
    fetch = fetch or upstream_fetch(hf_home)
    failed: List[str] = []
    for repo, commit, name, size in files:
        dest = os.path.join(repo_dir(hf_home, repo), "snapshots", commit, name)
        if os.path.isfile(dest):
            log(f"{PREFIX}   present  {repo}@{commit[:12]} {name}")
            continue
        log(f"{PREFIX}   fetching {repo}@{commit[:12]} {name} ({size / 1e6:.0f} MB)")
        try:
            fetch(repo, commit, name)
        except Exception as e:                                        # named and counted; the remaining files are still fetched
            failed.append(f"{repo} {name}: the transfer failed ({type(e).__name__}: {e})")
            log(f"{PREFIX}   FAILED   {repo} {name}: {type(e).__name__}: {e}")
    for repo in sorted(pins["weights"]):
        commit = pins["weights"][repo]["snapshot_commit"]
        prev = point_ref(hf_home, repo, commit)
        if prev != commit:
            log(f"{PREFIX}   refs/{REF_NAME} of {repo} -> {commit[:12]}" + (f" (was {prev[:12]})" if prev else ""))
    absent, unknown = stack.weight_files_check(hf_home, pins, None, afresh=True)   # the routes' own comparison, every repository, digested afresh
    for line in failed + [f"absent: {a}" for a in absent] + [f"off its pin: {u}" for u in unknown]:
        log(f"{PREFIX} WEIGHTS {line}")
    if failed or absent or unknown:
        log(f"{PREFIX} WEIGHTS FAILED: {len(absent)} absent, {len(unknown)} off their pin, of {len(files)} pinned files under {hf_home} "
            f"({len(failed)} transfer(s) failed; every file is left in place) — exit {EXIT_FAIL}")
        return EXIT_FAIL
    log(f"{PREFIX} WEIGHTS OK: {len(files)}/{len(files)} pinned files under {hf_home} match stock/PINS.json (sha256); export HF_HOME={hf_home} for pred / check / warm")
    return EXIT_OK


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1 or argv[0].startswith("-"):
        print(f"usage: python -m esmfold2_opt.weights DIR   (run.sh install --weights DIR; DIR = the HF_HOME to fill)", file=sys.stderr)
        return EXIT_USAGE
    return install_weights(argv[0])


if __name__ == "__main__":
    sys.exit(main())
