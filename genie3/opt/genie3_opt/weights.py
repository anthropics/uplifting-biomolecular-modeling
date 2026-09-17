"""`run.sh install --weights DIR` — fetch upstream's ``pretrained/`` tree into DIR with upstream's own downloader, then check the two files the kit
reads against ``stock/PINS.json`` ``weights`` (sha256). ``GENIE3_WEIGHTS`` is then ``DIR/pretrained/v1`` (README.md Setup); nothing is fetched
at run time. ``python -m genie3_opt.weights DIR``.

Upstream owns the transfer: its downloader is ``scripts/setup/download.sh``, whose ``--weights`` branch runs huggingface_hub's command-line tool —
``hf download "$HF_REPO" --include <pattern> --local-dir <checkout>``. The repository id and the pattern are READ from that script (the copy under
``stock/src/``; nothing is restated here) and the same command runs with DIR as the local directory, so DIR receives upstream's own layout:
``pretrained/v1/checkpoints/step=600000.ckpt`` and ``pretrained/v1/config.yaml`` (the pinned pair every route reads, ``stack.WEIGHT_FILES``) beside
whatever else the pattern matches upstream (an earlier model under ``pretrained/legacy/`` that the kit does not read). DIR = ``$GENIE3_ROOT`` is
``bash scripts/setup/download.sh --weights`` exactly — the stock layout, with which ``GENIE3_WEIGHTS`` may stay unset. A pinned file already in
place is excluded from the download (``--exclude``) and only checked; when both are present nothing is downloaded. The comparison is
``manifest.weights_record`` — the census every route's activation line reports — hashed afresh, so its digest memo is warm for the first ``check``
/ ``warm``. A file whose digest is not the pin's, or that the download did not produce, is named and the step fails (exit 1); files are left in
place for inspection, never deleted.
Exit codes: 0 ok · 1 the download failed, or a pinned file is absent / off its pin · 2 usage.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import sysconfig
from typing import Callable, List, Optional, Sequence, Tuple

PREFIX = "[genie3-opt install]"
EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2
DOWNLOADER = "src/scripts/setup/download.sh"        # under stock/: upstream's downloader, the source of the repository id and the weights pattern
LOCAL_SUBDIR = os.path.join("pretrained", "v1")     # where that pattern lands the pinned pair under the local directory (stock/PINS.json weights: source; stack.weights_dir)


def upstream_constants(script_text: str) -> Tuple[str, str]:
    """(repository id, weights pattern) as upstream's download.sh spells them: ``HF_REPO="…"`` and the ``hf_download "<pattern>"`` call of its
    ``$DO_WEIGHTS`` branch. A script that no longer has that shape raises (the pin moved and this reader with it)."""
    repo = re.search(r'^HF_REPO="([^"]+)"', script_text, re.M)
    pattern = re.search(r'if \$DO_WEIGHTS; then.*?hf_download "([^"]+)"', script_text, re.S)
    if not repo or not pattern:
        raise RuntimeError(f"stock/{DOWNLOADER} does not have the expected shape (HF_REPO=\"…\" and the $DO_WEIGHTS branch's hf_download \"<pattern>\")")
    return repo.group(1), pattern.group(1)


def hf_command(repo: str, pattern: str, local_dir: str, exclude: Sequence[str] = ()) -> List[str]:
    """Upstream's command (download.sh hf_download) with ``local_dir`` as --local-dir and one --exclude per file already in place. ``hf`` is
    huggingface_hub's console script beside this interpreter (the pinned stack installs it)."""
    hf = os.path.join(sysconfig.get_path("scripts"), "hf")
    cmd = [hf, "download", repo, "--include", pattern]
    for rel in exclude:
        cmd += ["--exclude", rel]
    return cmd + ["--local-dir", local_dir]


def fetch(local_dir: str, runner: Optional[Callable[[List[str]], int]] = None, census: Optional[Callable[[str], dict]] = None,
          script_text: Optional[str] = None, out=None) -> int:
    """Download what is absent with upstream's command, then check every pinned file's sha256. ``runner`` (runs the command, returns its exit
    status), ``census`` (``manifest.weights_record``, afresh) and ``script_text`` (upstream's download.sh) are injectable for the package tests."""
    from . import stack
    out = out or sys.stdout
    local_dir = os.path.abspath(local_dir)
    weights_dir = os.path.join(local_dir, LOCAL_SUBDIR)
    if script_text is None:
        with open(os.path.join(stack.tree_root(), "stock", *DOWNLOADER.split("/")), encoding="utf-8") as fh:
            script_text = fh.read()
    if census is None:
        from . import manifest
        census = lambda d: manifest.weights_record(d, refresh=True)      # noqa: E731 — the routes' own census, hashed afresh (memo rewritten)
    if runner is None:
        def runner(cmd):                                                  # upstream's tool, writing its own progress; absent → named (no fallback)
            if not os.path.isfile(cmd[0]):
                raise FileNotFoundError(f"{cmd[0]} does not exist — huggingface_hub's `hf` tool is not installed beside {sys.executable} "
                                        f"(it is part of the pinned stack: environment/requirements.lock)")
            return subprocess.call(cmd)
    repo, pattern = upstream_constants(script_text)
    os.makedirs(local_dir, exist_ok=True)
    present = [rel for rel in stack.WEIGHT_FILES if os.path.isfile(os.path.join(weights_dir, *rel.split("/")))]
    for rel in stack.WEIGHT_FILES:
        print(f"{PREFIX} {LOCAL_SUBDIR}/{rel}: {'present — kept, checked below' if rel in present else 'absent — fetched below'}", file=out, flush=True)
    if len(present) == len(stack.WEIGHT_FILES):
        print(f"{PREFIX} every pinned file is present under {weights_dir}: nothing to download", file=out, flush=True)
    else:
        cmd = hf_command(repo, pattern, local_dir, exclude=[f"{LOCAL_SUBDIR}/{rel}" for rel in present])
        print(f"{PREFIX} upstream's downloader (scripts/setup/download.sh --weights) into {local_dir}: {' '.join(cmd)}", file=out, flush=True)
        try:
            rc = runner(cmd)
        except OSError as e:                                              # the tool could not be started (absent, not executable): named, no fallback
            print(f"{PREFIX} FAILED: the download did not run — {type(e).__name__}: {e}", file=out, flush=True)
            return EXIT_FAIL
        if rc != 0:
            print(f"{PREFIX} FAILED: the download exited {rc} (its own report is above); files already under {local_dir} are left as they are — re-run this step to resume", file=out, flush=True)
            return EXIT_FAIL
    rec = census(weights_dir)
    for line in rec.get("lines", []):
        print(f"{PREFIX} {line}", file=out, flush=True)
    missing = list(rec.get("missing", []))
    off = [rel for rel, f in rec.get("files", {}).items() if not f.get("missing") and not f.get("pinned")]
    if missing or off:
        found = [f"{rel} sha256={rec['files'][rel].get('sha256')} (pin {rec['files'][rel].get('pin_sha256')})" for rel in off]
        print(f"{PREFIX} REFUSED: under {weights_dir} — absent after the download: {', '.join(missing) or 'none'}; not at the pinned digest "
              f"(stock/PINS.json weights): {', '.join(found) or 'none'} — left in place; remove a bad file and re-run this step to fetch it afresh", file=out, flush=True)
        return EXIT_FAIL
    print(f"{PREFIX} WEIGHTS OK: {len(stack.WEIGHT_FILES)}/{len(stack.WEIGHT_FILES)} pinned files under {weights_dir} have the pinned digest — "
          f"export GENIE3_WEIGHTS={weights_dir}" + (" (or leave it unset: this is the stock layout under $GENIE3_ROOT)" if _is_checkout(local_dir) else ""), file=out, flush=True)
    return EXIT_OK


def _is_checkout(local_dir: str) -> bool:
    root = os.environ.get("GENIE3_ROOT")
    return bool(root) and os.path.realpath(root) == os.path.realpath(local_dir)


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1 or argv[0].startswith("-"):
        print("usage: python -m genie3_opt.weights DIR   (run.sh install --weights DIR)", file=sys.stderr)
        return EXIT_USAGE
    return fetch(argv[0])


if __name__ == "__main__":
    sys.exit(main())
