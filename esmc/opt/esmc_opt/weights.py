"""`run.sh install --weights DIR --variant V` — fetch the variant's weights snapshot into DIR with upstream's own downloader and check every
weight file against ``stock/PINS.json`` ``weights`` (sha256). DIR is the weights root of STOCK.md 'Weights': the Hugging Face cache is
``DIR/hf`` (``export HF_HOME=DIR/hf`` — the one variable every route reads) and the snapshot lands at
``DIR/hf/hub/models--biohub--ESMC-<SIZE>/snapshots/<commit>/``; nothing is fetched at run time.

Upstream owns the transfer: ``esm.models.hub.resolve_model_dir`` — the call ``ESMC.from_pretrained`` makes for a repo id — runs
``huggingface_hub.snapshot_download`` for the repo's ``*.json`` and ``*.safetensors`` at its default revision into that cache (``refs/main``
written, as a first ``from_pretrained`` leaves it); ``huggingface_hub`` reads ``HF_HOME`` when it is imported, so the variable is set first
and the cache it resolved is checked. Files already in the cache are kept (the hub client skips a complete file), so a populated DIR is only
checked. The digest comparison is ``stock/check_pins.py`` ``weights`` — the routine its ``--weights DIR --variant V`` switch runs — so the
two commands read one layout. A file off its pin, or a default revision that no longer names the pinned snapshot, is named and the step
fails (exit 1); files are left in place, never deleted (an interrupted transfer leaves ``*.incomplete`` blobs under ``DIR/hf/hub``; a re-run
resumes them). ``python -m esmc_opt.weights DIR VARIANT``.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Callable, List, Optional, Tuple

PREFIX = "[esmc-opt install]"
EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2
HF_SUBDIR = "hf"                     # stock/PINS.json weights_layout: the HF cache under the weights root (check_pins.py --weights reads the same)


def hf_home_of(root: str) -> str:
    """The HF_HOME under weights root ``root`` (stock/PINS.json ``weights_layout``: ``<root>/hf``)."""
    return os.path.join(os.path.abspath(root), HF_SUBDIR)


def snapshot_path(root: str, repo: str, commit: str) -> str:
    """Where the hub client keeps ``repo`` @ ``commit`` under weights root ``root``: ``<root>/hf/hub/models--<org>--<name>/snapshots/<commit>``."""
    return os.path.join(hf_home_of(root), "hub", "models--" + repo.replace("/", "--"), "snapshots", commit)


def upstream_route(hf_home: str) -> Callable[[str], str]:
    """``route(repo)`` → upstream's resolver call that fetches ``repo``'s snapshot into the cache under ``hf_home`` and returns the snapshot
    directory it resolved. Bound after ``HF_HOME`` is set: ``huggingface_hub`` reads the variable at import, and the cache it resolved must be
    the one under ``hf_home`` (a set ``HF_HUB_CACHE`` / ``HUGGINGFACE_HUB_CACHE``, or the module imported earlier in this process, would send
    the files elsewhere — refused by name)."""
    os.environ["HF_HOME"] = hf_home
    from huggingface_hub import constants                                  # noqa: E402 — after HF_HOME is set (read at import)
    cache, want = os.path.realpath(constants.HF_HUB_CACHE), os.path.realpath(os.path.join(hf_home, "hub"))
    if cache != want:
        raise RuntimeError(f"huggingface_hub resolves its cache to {cache}, not {want}: unset HF_HUB_CACHE / HUGGINGFACE_HUB_CACHE for this step "
                           f"(the routes read HF_HOME={hf_home}, whose cache is {want})")
    from esm.models.hub import resolve_model_dir                           # upstream's resolver: snapshot_download(repo_id, *.json + *.safetensors)

    def route(repo: str) -> str:
        return str(resolve_model_dir(repo))
    return route


def fetch(root: str, variant: str, pins: Optional[dict] = None, route: Optional[Callable[[str], str]] = None,
          checker: Optional[Callable[[dict, str, str], Tuple[List[str], int]]] = None, out=None) -> int:
    """Fetch what is missing of ``variant``'s snapshot under weights root ``root``, check every weight file's digest, report. Returns the exit code."""
    out = out or sys.stdout
    say = lambda s: print(f"{PREFIX} {s}", file=out, flush=True)
    if pins is None:
        from . import stack
        pins = stack.pins()
    variants = {k: v for k, v in pins.get("variants", {}).items() if isinstance(v, dict) and "hf_repo" in v}
    if variant not in variants:
        say(f"'{variant}' is not a variant ({'|'.join(variants)}: stock/PINS.json variants)"); return EXIT_USAGE
    repo = variants[variant]["hf_repo"]; w = pins["weights"][repo]
    commit, files, total = w["snapshot_commit"], list(w["weights_files"]), int(w.get("total_weights_bytes", 0))
    root = os.path.abspath(root); hf_home = hf_home_of(root); snap = snapshot_path(root, repo, commit)
    os.makedirs(hf_home, exist_ok=True)
    present = [f for f in files if os.path.isfile(os.path.join(snap, f))]
    say(f"{repo} @ {commit[:12]} ({variant}): {len(present)}/{len(files)} weight files present under {snap}"
        + ("" if len(present) == len(files) else f" — fetching with upstream's downloader into HF_HOME={hf_home} ({total / 1e9:.1f} GB in all)"))
    if len(present) < len(files):
        try:
            route = route or upstream_route(hf_home)
        except Exception as e:                                             # esm / huggingface_hub not importable, or the cache resolves elsewhere
            say(f"FAILED: {type(e).__name__}: {e}"); return EXIT_FAIL
        t0 = time.time()
        try:
            got = route(repo)
        except Exception as e:                                             # network, disk, an offline switch set (HF_HUB_OFFLINE=1): upstream's words
            say(f"FAILED fetching {repo}: {type(e).__name__}: {e} — what was transferred is left in place under {hf_home}/hub; re-run this step to resume")
            return EXIT_FAIL
        say(f"upstream's downloader returned {got} ({time.time() - t0:.0f} s)")
        if os.path.realpath(str(got)) != os.path.realpath(snap):
            say(f"FAILED: {repo}'s default revision resolved to {os.path.basename(str(got))}, not the pinned snapshot {commit} (stock/PINS.json weights) "
                f"— upstream re-published the repository; the fetched files are left in place, the pin names the tested ones")
            return EXIT_FAIL
    if checker is None:
        from . import stack
        checker = stack._check_pins_module().weights                       # stock/check_pins.py weights(pins, root, variant): every pinned file at its sha256
    t0 = time.time(); bad, n = checker(pins, root, variant); secs = time.time() - t0
    for line in bad: say(line)
    if bad or n != len(files):
        say(f"REFUSED: {len(bad)} finding(s) on the {len(files)} weight files of {repo} under {root} (stock/PINS.json weights: sha256) — the files are "
            f"left in place; remove the named ones and re-run this step to fetch them afresh")
        return EXIT_FAIL
    say(f"WEIGHTS OK: {n}/{len(files)} weight files of {repo} @ {commit[:12]} under {root} have the pinned digest ({secs:.0f} s hashing) — export HF_HOME={hf_home}")
    return EXIT_OK


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0].startswith("-") or argv[1].startswith("-"):
        print(f"usage: python -m esmc_opt.weights DIR VARIANT   (run.sh install --weights DIR --variant V: fetch VARIANT's weights snapshot into DIR/hf with "
              f"upstream's downloader and check it against stock/PINS.json; VARIANT = 300m|600m|6b)", file=sys.stderr)
        return EXIT_USAGE
    return fetch(argv[0], argv[1])


if __name__ == "__main__":
    sys.exit(main())
