"""`run.sh install --weights DIR` — populate DIR, the ``PROTENIX_ROOT_DIR`` every route reads, with the seven stock files (the checkpoint
``checkpoint/protenix-v2.pt`` and the six data caches under ``common/``) through upstream's own download routine, then check each one against
``stock/PINS.json`` (``checkpoint.sha256``, ``data_caches.files``). Nothing is fetched at run time afterwards: ``export PROTENIX_ROOT_DIR=DIR``
(plus ``PROTENIX_ROOT_FROZEN=1``, under which an absent file is a refusal by name instead of a download, README.md 'Install').

Upstream owns the transfer: ``configs.configs_data`` / ``configs.configs_inference`` read ``PROTENIX_ROOT_DIR`` when they are imported, and
``runner.inference.download_inference_cache(configs)`` writes every file that is absent — the four data caches, the two template caches (the configs
here have ``use_template`` on) — from upstream's own URL table (``protenix.web_service.dependency_url``); no URL is restated here and a file
already in DIR is kept and only checked. The checkpoint is the one file this step transfers itself, first (``prefetch_checkpoint``): upstream
torch-loads a checkpoint it has just downloaded, before any digest could be compared, so the file comes from upstream's URL for the pinned
model name into ``<checkpoint>.part``, its sha256 is compared with ``checkpoint.sha256``, and only a file with the pinned digest is renamed into
place — upstream's routine is handed a stand-in checkpoint directory in which that name already exists, so it transfers the data caches only
and never downloads or loads a checkpoint; a transfer with any other digest stays at ``.part``, named, never loaded, upstream's routine is
not run, and the step fails; a transfer the server refuses (README.md 'Setup') is named and the data caches are fetched regardless. A file whose
digest is not the pin's is named and the step fails (exit 1) — the file is left in place for inspection, never deleted; a file the transfer
leaves absent is named too, after every present file has been checked, so a partial transfer is finished by hand and re-run. The checkpoint comparison is
``manifest.weights_status`` — the one the routes' WEIGHTS line reports — run afresh, so its digest memo is warm for the first ``check`` / ``pred``;
the caches' is ``opt_core.gates.verify_sums`` over ``data_caches.files``. ``python -m protenix_opt.weights DIR``.
"""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from typing import Callable, List, Optional

PREFIX = "[protenix-opt install]"
EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2
ROOT_ENV = "PROTENIX_ROOT_DIR"


def wanted(pins: dict) -> List[str]:
    """The files of DIR, relative to it, in PINS order: the checkpoint (``checkpoint.file``) then the data caches (``data_caches.files``)."""
    return [pins["checkpoint"]["file"]] + list((pins.get("data_caches") or {}).get("files") or {})


def upstream_route(root: str) -> Callable[[], None]:
    """The upstream call that fetches every absent file of ``wanted`` into ``root``: ``runner.inference.download_inference_cache`` over the
    stock configs resolved for ``root`` (model name = the pinned checkpoint's, templates on). Bound after ``PROTENIX_ROOT_DIR`` is set:
    upstream's config modules read the variable at import, so an interpreter that imported them earlier is refused by name."""
    os.environ[ROOT_ENV] = root
    from configs import configs_data, configs_inference                    # stock's config modules (the protenix wheel's top-level `configs`)
    for mod in (configs_data, configs_inference):
        if os.path.realpath(str(mod.PROTENIX_ROOT_DIR)) != os.path.realpath(root):
            raise RuntimeError(f"{mod.__name__} resolved {ROOT_ENV}={mod.PROTENIX_ROOT_DIR!r} before {root!r} was set (it was imported earlier in this interpreter); run this step in a fresh interpreter")
    from ml_collections.config_dict import ConfigDict                     # the configs type upstream's runner reads (item and attribute access)
    from runner.inference import download_inference_cache                 # upstream's downloader: URL table, transfer, checkpoint load check
    from .manifest import CHECKPOINT_NAME
    import tempfile
    standin = tempfile.mkdtemp(prefix="protenix-opt-ckptdir-")             # a private directory holding an empty file with the checkpoint's name: upstream's routine
    open(os.path.join(standin, CHECKPOINT_NAME + ".pt"), "wb").close()     # sees the checkpoint "present" and transfers the data caches only — the checkpoint is this
    configs = ConfigDict({"data": configs_data.data_configs, "use_template": True,      # step's own transfer (prefetch_checkpoint), never upstream's download-and-load
                          "load_checkpoint_dir": standin, "model_name": CHECKPOINT_NAME})
    return lambda: download_inference_cache(configs)


def sha256_file(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def prefetch_checkpoint(root: str, pins: dict, retrieve: Optional[Callable[[str, str], object]] = None,
                        url_for: Optional[Callable[[str], str]] = None, log=print) -> Optional[str]:
    """The checkpoint, when absent, transferred by this step before upstream's routine runs: upstream's URL for the pinned model name
    (``protenix.web_service.dependency_url.URL``, the table ``download_inference_cache`` reads), ``urllib.request.urlretrieve`` (upstream's call)
    into ``<checkpoint>.part``, that file's sha256 compared with ``checkpoint.sha256``, then the rename to the checkpoint's name — so nothing reads
    the file as a checkpoint before its digest is the pin's. Returns None with the pinned checkpoint in place (or already present: ``check``
    judges it), else the line naming the refusal; the ``.part`` file is left where it is. ``retrieve`` / ``url_for`` are injectable for the tests."""
    rel = pins["checkpoint"]["file"]
    final = os.path.join(root, rel)
    if os.path.isfile(final):
        return None
    if url_for is None:
        from protenix.web_service.dependency_url import URL               # upstream's URL table, keyed by model name; no URL is restated here
        url_for = URL.__getitem__
    from .manifest import CHECKPOINT_NAME
    url = url_for(CHECKPOINT_NAME)
    part = final + ".part"
    os.makedirs(os.path.dirname(final), exist_ok=True)
    log(f"{PREFIX} {rel}: transferring {url} to {rel}.part — compared with the pin before it takes the checkpoint's name")
    (retrieve or urllib.request.urlretrieve)(url, part)
    found = sha256_file(part)
    if found != pins["checkpoint"]["sha256"]:
        return (f"{rel}.part sha256 {found} is not the pin {pins['checkpoint']['sha256']} (stock/PINS.json checkpoint.sha256): the transfer from {url} "
                f"is left at {part} for inspection and was never loaded — remove it and re-run this step")
    os.replace(part, final)
    log(f"{PREFIX} {rel}: sha256 {found[:12]} = the pin; in place before upstream's routine runs (it fetches the data caches and leaves a present checkpoint alone)")
    return None


def fetch(root: str, pins: dict, route: Optional[Callable[[], None]] = None, log=print,
          prefetch: Optional[Callable[..., Optional[str]]] = None) -> List[str]:
    """Ensure every ``wanted`` file exists under ``root``: present → kept; an absent checkpoint → ``prefetch`` (``prefetch_checkpoint``: compared
    with its pin before it is in place; a refusal is logged and upstream's downloader is not run); anything else absent → upstream's downloader
    runs once (it skips what is present). Returns the files still absent afterwards (named in the log; a failed transfer leaves the partial
    file, if any, in place). ``route`` / ``prefetch`` are injectable for the tests (an injected ``route`` writes every file unless a
    ``prefetch`` is injected with it)."""
    os.makedirs(root, exist_ok=True)
    absent = []
    for rel in wanted(pins):
        if os.path.isfile(os.path.join(root, rel)):
            log(f"{PREFIX} {rel}: present — kept, checked below")
        else:
            absent.append(rel); log(f"{PREFIX} {rel}: absent — upstream's downloader fetches it")
    if absent:
        try:
            if route is None:
                route = upstream_route(root)
                prefetch = prefetch_checkpoint if prefetch is None else prefetch
            refused = None
            if prefetch is not None and pins["checkpoint"]["file"] in absent:
                try:
                    refused = prefetch(root, pins, log=log)
                except Exception as e:  # noqa: BLE001 — the server refuses the checkpoint (README.md 'Setup') or the transfer broke: named; the data caches are still fetched
                    log(f"{PREFIX} {pins['checkpoint']['file']}: the transfer did not complete ({type(e).__name__}: {e}) — left absent (a partial .part file, if any, stays for inspection); "
                        f"place the file there yourself (README.md 'Setup'); the data caches are fetched regardless")
            if refused:
                log(f"{PREFIX} REFUSED: {refused}")
                return [rel for rel in wanted(pins) if not os.path.isfile(os.path.join(root, rel))]
            route()
        except Exception as e:                                            # noqa: BLE001 — upstream's failure is reported by name, the step goes on to name what is missing
            log(f"{PREFIX} upstream's download FAILED: {type(e).__name__}: {e}")
    return [rel for rel in wanted(pins) if not os.path.isfile(os.path.join(root, rel))]


def check(root: str, pins: dict, log=print) -> List[str]:
    """Every ``wanted`` file under ``root`` against its pin: the checkpoint through ``manifest.weights_status`` (hashed afresh, memo rewritten),
    the data caches through ``opt_core.gates.verify_sums``. Returns the files that are absent or not the pinned bytes (each named in the log)."""
    from . import _core  # noqa: F401  (opt_core importable: installed, else the pinned path)
    from . import manifest
    from opt_core import gates
    bad = []
    ws = manifest.weights_status(pins["checkpoint"]["sha256"], root_dir=root, refresh=True)
    log(f"{PREFIX} {ws['line']}")
    if ws.get("digest") != "pinned":
        bad.append(pins["checkpoint"]["file"])
    files = (pins.get("data_caches") or {}).get("files") or {}
    g = gates.verify_sums(root, {rel: e["sha256"] for rel, e in files.items()}, name="data_caches")
    for rel in g.details.get("missing", []):
        log(f"{PREFIX} {rel}: ABSENT"); bad.append(rel)
    for rel, d in (g.details.get("mismatched") or {}).items():
        log(f"{PREFIX} {rel}: sha256 {d['found']} is NOT the pin {d['expected']} (stock/PINS.json data_caches) — left in place, not the tested file"); bad.append(rel)
    if g.ok:
        log(f"{PREFIX} data caches pinned: {g.details['checked']}/{g.details['listed']} files under {os.path.join(root, 'common')} = stock/PINS.json data_caches")
    return bad


def main(argv: Optional[List[str]] = None, route: Optional[Callable[[], None]] = None, log=print, prefetch: Optional[Callable[..., Optional[str]]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1 or argv[0].startswith("-"):
        print(f"usage: python -m protenix_opt.weights DIR   (run.sh install --weights DIR): fetch the checkpoint and the stock data caches into DIR and check them against stock/PINS.json; DIR is then {ROOT_ENV}", file=sys.stderr)
        return EXIT_USAGE
    root = os.path.abspath(argv[0])
    from .stack import pins as read_pins
    pins = read_pins()
    n = len(wanted(pins))
    log(f"{PREFIX} weights root {root}: {n} files (stock/PINS.json checkpoint + data_caches), fetched by upstream's downloader when absent, then checked")
    missing = fetch(root, pins, route=route, log=log, prefetch=prefetch)
    bad = check(root, pins, log=log)                                      # every file's verdict — present and pinned, off its pin, absent — whatever the transfer left
    if missing or bad:
        if missing:
            log(f"{PREFIX} FAILED: {len(missing)} of {n} files absent under {root} after upstream's download: {missing} — place them there yourself (README.md 'Install') and re-run this step: it keeps and checks what is present")
        off = [rel for rel in bad if rel not in missing]
        if off:
            log(f"{PREFIX} REFUSED: {len(off)} of {n} files under {root} are not the pinned files: {off}")
        return EXIT_FAIL
    log(f"{PREFIX} WEIGHTS OK: {n}/{n} files under {root} are the pinned files (stock/PINS.json checkpoint.sha256 + data_caches.files) — export {ROOT_ENV}={root}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
