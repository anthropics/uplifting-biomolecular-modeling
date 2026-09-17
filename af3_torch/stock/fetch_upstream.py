#!/usr/bin/env python3
"""fetch_upstream.py [--archive FILE] — make the pinned upstream archive present under stock/: the step of `run.sh install` that lays xfold's
own bytes beside this file, whichever way the tree arrived.

The archive is stock/PINS.json `upstream.archive.file` (`xfold-22bdeed.tar.gz`, GitHub's archive of `upstream.commit`); the stock route
(`run.sh stock`: xfold's own run_alphafold.py, read out of the archive) and the kit's byte tests are its only readers — no run mode opens it.
  present  — the file is already under stock/: judged (a gzip tar with one root directory `xfold-<commit>/` holding run_alphafold.py) and
             reported, nothing fetched;
  absent   — fetched from `<upstream.repo>/archive/<upstream.commit>.tar.gz` (network needed), judged, then placed atomically; `--archive FILE`
             takes a copy fetched beforehand instead (a machine without network access) — judged the same way before it is placed.
The byte count is reported against PINS `upstream.archive.bytes` (the tested tree's copy) and not gated: the host's gzip framing of one commit's
tree is the host's; the members are what the readers use, and tests/test_kit_bytes.py holds the kit's xfold copy to them file for file.
One `[af3-torch-opt install] UPSTREAM ARCHIVE …` line on stdout; exit 0 present / fetched / placed, 1 a failed fetch or a refused file (the line
names the remedy), 2 usage. Standard library only.
"""
import argparse
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = "[af3-torch-opt install]"
EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2
CLI_MEMBER = "run_alphafold.py"        # xfold's CLI entry: the member the stock route extracts (af3_torch_opt/stock_cli.py ARCHIVE_MEMBER)


def load_pins():
    with open(os.path.join(HERE, "PINS.json"), encoding="utf-8") as f:
        return json.load(f)


def archive_path(pins):
    return os.path.join(HERE, pins["upstream"]["archive"]["file"])


def source_url(pins):
    u = pins["upstream"]
    return f"{u['repo'].rstrip('/')}/archive/{u['commit']}.tar.gz"


def judge(path, pins):
    """(ok, words): a file as the pinned upstream archive — a gzip tar whose members sit under ONE root directory `<name>-<commit>/` (GitHub's
    archive layout; the stock route names the commit from that path) and include the CLI entry. Byte count reported beside, not judged."""
    u = pins["upstream"]
    root = f"{u['name']}-{u['commit']}"
    try:
        with tarfile.open(path, "r:gz") as t:
            names = t.getnames()
    except (tarfile.TarError, OSError, EOFError) as e:
        return False, f"not a readable gzip tar ({e.__class__.__name__}: {e})"
    roots = sorted({n.split("/", 1)[0] for n in names})
    if roots != [root]:
        return False, f"root {roots[:3]} is not the one pinned directory {root}/"
    if f"{root}/{CLI_MEMBER}" not in names:
        return False, f"{root}/{CLI_MEMBER} is not among the {len(names)} members"
    n, want = os.path.getsize(path), u["archive"].get("bytes")
    size = f"{n} bytes" + ("" if want is None else (" = stock/PINS.json upstream.archive.bytes" if n == want else f" (stock/PINS.json upstream.archive.bytes says {want}: another gzip framing of the same commit; members are read by content)"))
    return True, f"members={len(names)} root={root}/ {size}"


def fetch(url, dest, opener=urllib.request.urlopen):
    """url -> dest (written to dest.part, renamed on success). Raises OSError / URLError on failure."""
    part = dest + ".part"
    with opener(url) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    os.replace(part, dest)


def main(argv=None, opener=urllib.request.urlopen, out=sys.stdout):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--archive", default=None, help="a copy of the upstream archive fetched beforehand (placed after the same judgement; nothing is downloaded)")
    a = ap.parse_args(argv)
    pins = load_pins()
    dest, url, rel = archive_path(pins), source_url(pins), "stock/" + pins["upstream"]["archive"]["file"]
    if os.path.isfile(dest):
        ok, words = judge(dest, pins)
        print(f"{PREFIX} UPSTREAM ARCHIVE {'present' if ok else 'REFUSED'} file={rel} {words}"
              + ("" if ok else f" — remove it and re-run this step (it fetches {url}), or pass --archive FILE"), file=out, flush=True)
        return EXIT_OK if ok else EXIT_FAIL
    tmpdir = tempfile.mkdtemp(prefix=".fetch_upstream.", dir=HERE)
    try:
        tmp = os.path.join(tmpdir, os.path.basename(dest))
        if a.archive:
            if not os.path.isfile(a.archive):
                print(f"{PREFIX} UPSTREAM ARCHIVE REFUSED --archive {a.archive}: no such file", file=out, flush=True)
                return EXIT_FAIL
            shutil.copyfile(a.archive, tmp); route = f"placed from={os.path.abspath(a.archive)}"
        else:
            try:
                fetch(url, tmp, opener)
            except (OSError, urllib.error.URLError) as e:
                print(f"{PREFIX} UPSTREAM ARCHIVE FETCH FAILED url={url} ({e.__class__.__name__}: {e}) — {rel} is not in this tree and could not be fetched; "
                      f"re-run this step with network access, or fetch that URL elsewhere and pass the file as --archive FILE", file=out, flush=True)
                return EXIT_FAIL
            route = f"fetched url={url}"
        ok, words = judge(tmp, pins)
        if not ok:
            print(f"{PREFIX} UPSTREAM ARCHIVE REFUSED {route} {words} — nothing was placed under stock/", file=out, flush=True)
            return EXIT_FAIL
        os.replace(tmp, dest)
        print(f"{PREFIX} UPSTREAM ARCHIVE {route} file={rel} {words}", file=out, flush=True)
        return EXIT_OK
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
