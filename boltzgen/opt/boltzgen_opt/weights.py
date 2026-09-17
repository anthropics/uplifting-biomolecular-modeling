"""`run.sh install --weights DIR` — fetch the six stock weight / data files into DIR with upstream's own downloader and check each one against
``stock/PINS.json`` ``weights.files`` (sha256). DIR is then the ``BOLTZGEN_CACHE`` every route reads (README.md 'Setup'); nothing is fetched
at run time (configs/h100.env sets HF_HUB_OFFLINE=1).

Upstream owns the transfer: ``boltzgen download`` resolves each ``huggingface:<repo>:<file>`` artifact through ``huggingface_hub.hf_hub_download``
into the Hugging Face cache layout under its ``--cache`` directory, and this step makes that same call — repository, file name and repository type
as PINS' cache path spells them (``models--boltzgen--boltzgen-1/snapshots/<snapshot>/<file>``, ``datasets--boltzgen--inference-data/…``), at the
pinned snapshot, ``cache_dir=DIR`` — for every file that is absent; a file already in DIR is kept and only checked. Each repository's ``refs/main``
under DIR is then pointed at the pinned snapshot, which is how upstream's offline lookups of ``main`` find these files at run time. A file whose
digest is not the pin's is named and the step fails (exit 1) — the file is left in place for inspection, never deleted; so is the ``*.incomplete``
blob of an interrupted transfer (huggingface_hub resumes it on the next run). The digest is the shared core's one chunked sha256
(``opt_core.gates.sha256_file``). ``python -m boltzgen_opt.weights DIR``.
"""
from __future__ import annotations

import os
import re
import sys
from typing import Callable, List, Optional

PREFIX = "[boltzgen-opt install]"
EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2
CACHE_PATH = re.compile(r"^(?P<kind>models|datasets)--(?P<org>[^/]+)--(?P<name>[^/]+)/snapshots/(?P<rev>[0-9a-f]{40})/(?P<filename>.+)$")   # PINS weights.files "file": the HF-cache path of one artifact
REPO_TYPES = {"models": "model", "datasets": "dataset"}


def parse_cache_path(rel: str) -> Optional[dict]:
    """PINS' cache path of a file → ``{"repo_id", "repo_type", "revision", "filename", "repo_dir"}`` (None when it is not an HF-cache path)."""
    m = CACHE_PATH.match(rel)
    if not m:
        return None
    return {"repo_id": f"{m['org']}/{m['name']}", "repo_type": REPO_TYPES[m["kind"]], "revision": m["rev"], "filename": m["filename"],
            "repo_dir": f"{m['kind']}--{m['org']}--{m['name']}"}


def upstream_route(cache_dir: str, download: Optional[Callable[..., object]] = None) -> Callable[[str], Optional[Callable[[], object]]]:
    """``route(rel)`` → the upstream call that fetches PINS' file ``rel`` into ``cache_dir`` (None for a path that names no Hugging Face artifact).
    The call is ``huggingface_hub.hf_hub_download`` — the one ``boltzgen download`` makes, ``library_name="boltzgen"`` included — at the pinned
    snapshot; ``download`` replaces it in the kit's unit tests."""
    if download is None:
        from huggingface_hub import hf_hub_download as download                # upstream's downloader (boltzgen.cli.boltzgen get_artifact_path)

    def route(rel: str):
        a = parse_cache_path(rel)
        if a is None:
            return None
        return lambda: download(a["repo_id"], a["filename"], repo_type=a["repo_type"], revision=a["revision"], library_name="boltzgen", cache_dir=cache_dir)
    return route


def point_refs(cache_dir: str, files: List[dict], out) -> None:
    """``<cache>/<repo_dir>/refs/main`` → the pinned snapshot, for every repository PINS names: upstream asks huggingface_hub for ``main`` with the
    hub offline, and that lookup reads this file. Written only when absent or naming another snapshot; the line says which."""
    seen = {}
    for w in files:
        a = parse_cache_path(w["file"])
        if a is not None:
            seen[a["repo_dir"]] = a["revision"]
    for repo_dir, rev in sorted(seen.items()):
        ref = os.path.join(cache_dir, repo_dir, "refs", "main")
        try:
            with open(ref, "r", encoding="utf-8") as fh:
                now = fh.read().strip()
        except OSError:
            now = None
        if now == rev:
            print(f"{PREFIX} {repo_dir}/refs/main: names the pinned snapshot {rev[:12]}", file=out, flush=True)
            continue
        os.makedirs(os.path.dirname(ref), exist_ok=True)
        with open(ref, "w", encoding="utf-8") as fh:
            fh.write(rev)
        print(f"{PREFIX} {repo_dir}/refs/main: {'written' if now is None else 'was ' + now[:12] + ', repointed'} → the pinned snapshot {rev[:12]}", file=out, flush=True)


def check(cache_dir: str, files: List[dict], digest: Optional[Callable[[str], str]] = None) -> dict:
    """Every PINS file under ``cache_dir`` by sha256: ``{"status": "pinned"|"unknown", "files", "unknown": [{"file", "sha256"|None}], "seconds"}``.
    An absent file counts as unknown (digest None). The digest is the core's ``opt_core.gates.sha256_file`` unless injected."""
    import time
    if digest is None:
        from opt_core.gates import sha256_file as digest
    t0 = time.time(); unknown = []
    for w in files:
        p = os.path.join(cache_dir, w["file"])
        found = digest(p) if os.path.isfile(p) else None
        if found != w["sha256"]:
            unknown.append({"file": w["file"], "sha256": found})
    return {"status": "unknown" if unknown else "pinned", "files": len(files), "unknown": unknown, "seconds": round(time.time() - t0, 1)}


def fetch(cache_dir: str, files: Optional[List[dict]] = None, route: Optional[Callable[[str], Optional[Callable[[], object]]]] = None,
          digest: Optional[Callable[[str], str]] = None, out=None) -> int:
    """Fetch what is absent, point the refs, then check every file's sha256 against the pin. ``files`` (PINS weights.files), ``route`` (upstream's
    downloader) and ``digest`` are injectable for the kit's unit tests."""
    out = out or sys.stdout
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    if files is None:
        from . import stack
        files = stack.pins()["weights"]["files"]
    try:
        if route is None:
            route = upstream_route(cache_dir)
    except Exception as e:  # noqa: BLE001 — huggingface_hub not importable here: named, nothing fetched
        print(f"{PREFIX} FAILED: {type(e).__name__}: {e}", file=out, flush=True)
        return EXIT_FAIL
    for w in files:
        rel = w["file"]; target = os.path.join(cache_dir, rel)
        state = "present" if os.path.isfile(target) else "fetching"
        print(f"{PREFIX} {rel}: {state}" + (" (upstream's downloader, the pinned snapshot)" if state == "fetching" else ""), file=out, flush=True)
        if state == "present":
            continue
        call = route(rel)
        if call is None:
            print(f"{PREFIX} FAILED: {rel} is not a Hugging Face cache path (<models|datasets>--<org>--<name>/snapshots/<snapshot>/<file>): stock/PINS.json weights.files names a file this step cannot fetch", file=out, flush=True)
            return EXIT_FAIL
        try:
            call()
        except Exception as e:  # noqa: BLE001 — upstream's own error (network, disk, an interrupted transfer), relayed by name; nothing is deleted
            print(f"{PREFIX} FAILED fetching {rel}: {type(e).__name__}: {e} — any *.incomplete blob under {cache_dir} is left in place (the next run resumes it)", file=out, flush=True)
            return EXIT_FAIL
        if not os.path.isfile(target):
            print(f"{PREFIX} FAILED: {rel} is not at {target} after upstream's fetch", file=out, flush=True)
            return EXIT_FAIL
    point_refs(cache_dir, files, out)
    wid = check(cache_dir, files, digest)
    for u in wid["unknown"]:
        print(f"{PREFIX} {u['file']}: sha256 {(u['sha256'] or 'absent')[:16]}… is not the pin's", file=out, flush=True)
    if wid["status"] != "pinned":
        bad = [u["file"] for u in wid["unknown"]]
        print(f"{PREFIX} REFUSED: {len(bad)} of {wid['files']} files under {cache_dir} do not have the pinned digest (stock/PINS.json weights.files): "
              f"{', '.join(bad)} — left in place; remove them and re-run this step to fetch afresh", file=out, flush=True)
        return EXIT_FAIL
    print(f"{PREFIX} WEIGHTS OK: {wid['files']}/{len(files)} files under {cache_dir} have the pinned digest ({wid['seconds']} s hashing) — export BOLTZGEN_CACHE={cache_dir}", file=out, flush=True)
    return EXIT_OK


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1 or argv[0].startswith("-"):
        print("usage: python -m boltzgen_opt.weights DIR   (run.sh install --weights DIR)", file=sys.stderr)
        return EXIT_USAGE
    return fetch(argv[0])


if __name__ == "__main__":
    sys.exit(main())
