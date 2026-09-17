"""`run.sh install --weights DIR` — populate DIR, the ``PROTENIX_ROOT_DIR`` every route reads, with upstream's own boot-time downloader,
then judge it with the gate every verb runs (``kit.frozen_weights_check``): the checkpoint ``stock/PINS.json`` "stock".checkpoint against
its sha256 pin ("stock".checkpoint_sha256), the six ``common/`` data files of "stock".data_files by presence — and then, in this step,
each data file's sha256 against "stock".data_files_sha256 (``check_data_files``: a file off its pin is named, left in place, and the
step fails; upstream's URL table stays their only source). Nothing is fetched at run time (README.md 'Setup').

Upstream owns the transfer. ``PROTENIX_ROOT_DIR`` is exported first — ``configs/configs_inference.py`` and ``configs/configs_data.py`` read it
when imported and lay out ``<DIR>/checkpoint/`` and ``<DIR>/common/`` — then ``runner.inference.download_inference_cache(configs)`` (the
routine ``protenix pred`` calls before it builds its runner: runner/inference.py:291-347) fetches every file that is absent, from the URL
table of ``protenix/web_service/dependency_url.py``, with templates switched on so the two template caches come too: the four data caches,
``obsolete_to_successor.json`` + ``release_date_cache.json``. The checkpoint is the one file this step transfers itself, first
(``prefetch_checkpoint``): upstream torch-loads a checkpoint it has just downloaded, before any digest could be compared, so the file
comes from upstream's own URL for the pinned model name into ``<checkpoint>.part``, its sha256 is compared with the pin, and only a
file with the pinned digest is renamed into place — upstream's routine then finds it present and neither downloads nor loads it. A
transfer with any other digest stays at ``.part``, named, never loaded, and the step fails (exit 1). A file already in DIR is kept; no
URL is restated here. Importing
``runner.inference`` imports the model package: this step runs with ``LAYERNORM_TYPE=torch`` so that import neither loads nor compiles
stock's fast LayerNorm extension (the downloader needs no kernel). A checkpoint whose digest is not the pin's is named and the step fails
(exit 1) — the file is left in place for inspection, never deleted; so is the partial file of an interrupted transfer (urllib writes the
target directly: remove it by hand and re-run). ``python -m protenix_v1_opt.weights DIR``.
"""
from __future__ import annotations

import os
import sys
import urllib.request
from typing import Callable, List, Optional

PREFIX = "[protenix-v1-opt install]"
EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2


def expected_files(stock: Optional[dict] = None) -> List[str]:
    """The DIR-relative files of a complete weights root: "stock".checkpoint, then every "stock".data_files entry (the template pair included)."""
    if stock is None:
        from . import stack as S                                      # stack.pins: the one reader of stock/PINS.json
        stock = S.pins()["stock"]
    files = [stock["checkpoint"]]
    for v in stock["data_files"].values():
        files += list(v) if isinstance(v, list) else [v]
    return files


def upstream_configs(root: str):
    """Upstream's inference configuration bound to ``root``, templates on (so the template caches count as wanted). The two configs modules
    read PROTENIX_ROOT_DIR at import: a process that imported them under another root keeps that root — this runs in a fresh interpreter."""
    os.environ["PROTENIX_ROOT_DIR"] = root
    os.environ["LAYERNORM_TYPE"] = "torch"                            # the model package import below neither loads nor builds the fast LayerNorm extension
    from configs.configs_base import configs as configs_base         # noqa: E402 — after the variable is set, by design
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from protenix.config.config import parse_configs
    configs = parse_configs({**configs_base, "data": data_configs, **inference_configs}, arg_str="--use_template true", fill_required_with_null=True)
    bound = os.path.realpath(str(configs.load_checkpoint_dir))
    if os.path.dirname(bound) != os.path.realpath(root):
        raise RuntimeError(f"upstream's configs resolve the checkpoint directory to {bound}, not {root}/checkpoint: configs.configs_inference was "
                           f"imported before PROTENIX_ROOT_DIR was set in this process — run this step in a fresh interpreter (python -m protenix_v1_opt.weights DIR)")
    return configs


def upstream_fetch(root: str) -> None:
    """``download_inference_cache`` on those configs: each absent file fetched (a progress bar per file), present files kept."""
    configs = upstream_configs(root)
    from runner.inference import download_inference_cache            # upstream's routine (imports torch and the model package; loads no model)
    download_inference_cache(configs)


def prefetch_checkpoint(root: str, stock: dict, retrieve: Optional[Callable[[str, str], object]] = None,
                        url_for: Optional[Callable[[str], str]] = None, out=None) -> Optional[str]:
    """The checkpoint, when absent, transferred by this step before upstream's routine runs: upstream's URL for "stock".model_name (the
    ``URL`` table ``download_inference_cache`` reads), ``urllib.request.urlretrieve`` (upstream's call) into ``<checkpoint>.part``, the sha256
    of that file compared with "stock".checkpoint_sha256, then the rename to the checkpoint's name — so nothing reads the file as a checkpoint
    before its digest is the pin's. Returns None with the pinned checkpoint in place (or already present: the gate judges it), else the one
    line naming the refusal; the ``.part`` file is left where it is. ``retrieve`` / ``url_for`` are injectable for the package tests."""
    out = out or sys.stdout
    rel = stock["checkpoint"]
    final = os.path.join(root, rel)
    if os.path.isfile(final):
        return None
    if url_for is None:
        from protenix.web_service.dependency_url import URL            # upstream's URL table, keyed by model name; no URL is restated here
        url_for = URL.__getitem__
    url = url_for(stock["model_name"])
    part = final + ".part"
    os.makedirs(os.path.dirname(final), exist_ok=True)
    print(f"{PREFIX} {rel}: transferring {url} to {rel}.part — compared with the pin before it takes the checkpoint's name", file=out, flush=True)
    (retrieve or urllib.request.urlretrieve)(url, part)
    from .digest_memo import sha256_file                                  # the kit's one file hasher (the gate's)
    found = sha256_file(part)
    if found != stock["checkpoint_sha256"]:
        return (f"{rel}.part sha256 {found} is not the pin {stock['checkpoint_sha256']} (stock/PINS.json \"stock\".checkpoint_sha256): the transfer from {url} "
                f"is left at {part} for inspection and was never loaded — remove it and re-run this step")
    os.replace(part, final)
    print(f"{PREFIX} {rel}: sha256 {found[:12]} = the pin; in place before upstream's routine runs (it fetches the data files and leaves a present checkpoint alone)", file=out, flush=True)
    return None


def check_data_files(root: str, stock: dict, out=None) -> List[str]:
    """Every data file of "stock".data_files that "stock".data_files_sha256 pins, hashed under ``root`` and compared with its pin; returns the
    DIR-relative names of the files off their pins (each named on a line; nothing is deleted). Upstream's routine downloads these files without
    reading them, so the comparison here comes before any verb loads one."""
    out = out or sys.stdout
    from .digest_memo import sha256_file                                  # the kit's one file hasher (the gate's)
    sums = stock.get("data_files_sha256") or {}
    off = []
    for rel in expected_files(stock)[1:]:
        want = sums.get(rel)
        if not want:
            continue
        got = sha256_file(os.path.join(root, rel))
        if got != want:
            off.append(rel)
            print(f"{PREFIX} {rel}: sha256 {got} is not the pin {want} (stock/PINS.json \"stock\".data_files_sha256) — left in place", file=out, flush=True)
        else:
            print(f"{PREFIX} {rel}: sha256 {got[:12]} = the pin", file=out, flush=True)
    return off


def populate(root: str, fetch: Optional[Callable[[str], None]] = None, gate: Optional[Callable[..., dict]] = None, stock: Optional[dict] = None,
             out=None, prefetch: Optional[Callable[..., Optional[str]]] = None) -> int:
    """Fetch what is absent — an absent checkpoint first, by ``prefetch`` (``prefetch_checkpoint``: compared with the pin before it is in place),
    the rest by ``fetch`` (upstream's downloader) — then run the weights gate afresh. ``fetch``, ``gate`` (``kit.frozen_weights_check``),
    ``stock`` (PINS "stock") and ``prefetch`` are injectable for the package tests (an injected ``fetch`` writes every file unless a
    ``prefetch`` is injected with it)."""
    out = out or sys.stdout
    root = os.path.abspath(root)
    os.makedirs(root, exist_ok=True)
    if stock is None:
        from . import stack as S
        stock = S.pins()["stock"]
    files = expected_files(stock)
    missing = [f for f in files if not os.path.isfile(os.path.join(root, f))]
    for f in files:
        print(f"{PREFIX} {f}: {'fetching (upstream' + chr(39) + 's download_inference_cache)' if f in missing else 'present'}", file=out, flush=True)
    if missing:
        if prefetch is None and fetch is None:
            prefetch = prefetch_checkpoint
        try:
            refused = prefetch(root, stock, out=out) if prefetch is not None and stock["checkpoint"] in missing else None
            if refused:
                print(f"{PREFIX} REFUSED: {refused}", file=out, flush=True)
                return EXIT_FAIL
            (fetch or upstream_fetch)(root)
        except Exception as e:  # noqa: BLE001 — upstream not importable here, its root bound elsewhere, or its own transfer error: named, nothing deleted by this step
            print(f"{PREFIX} FAILED: {type(e).__name__}: {e}", file=out, flush=True)
            return EXIT_FAIL
        still = [f for f in missing if not os.path.isfile(os.path.join(root, f))]
        if still:
            print(f"{PREFIX} FAILED: absent after upstream's fetch: {', '.join(still)}", file=out, flush=True)
            return EXIT_FAIL
    if gate is None:
        from . import kit as K
        gate, refusal = K.frozen_weights_check, K.FrozenWeightsError
    else:
        refusal = RuntimeError
    try:
        w = gate(root, argv=[], announce=False, refresh=True)         # the gate every verb runs, hashed afresh (its digest memo is then warm for check / warm / pred)
    except refusal as e:
        print(f"{PREFIX} REFUSED: {e}", file=out, flush=True)
        return EXIT_FAIL
    if not w["pinned"]:
        print(f"{PREFIX} REFUSED: {w['checkpoint']} sha256 {w['sha256']} is not the pin {stock['checkpoint_sha256']} (stock/PINS.json \"stock\".checkpoint_sha256) "
              f"— left in place; remove it and re-run this step to fetch afresh", file=out, flush=True)
        return EXIT_FAIL
    off = check_data_files(root, stock, out=out)
    if off:
        print(f"{PREFIX} REFUSED: {len(off)} of {len(files) - 1} data files under {root} do not have the pinned digest: {', '.join(off)} "
              f"— left in place; remove them and re-run this step to fetch afresh", file=out, flush=True)
        return EXIT_FAIL
    pinned = sum(1 for rel in files[1:] if (stock.get("data_files_sha256") or {}).get(rel))
    how = (f"the {pinned} data files are at their pins (stock/PINS.json \"stock\".data_files_sha256)" if pinned == len(files) - 1 else
           f"{pinned} of the {len(files) - 1} data files carry a digest pin and are at it (stock/PINS.json \"stock\".data_files_sha256); the rest are pinned by name")
    print(f"{PREFIX} WEIGHTS OK: {len(files)}/{len(files)} files under {root}; {os.path.basename(w['checkpoint'])} sha256 {w['sha256'][:12]} (pinned, {w['checkpoint_bytes']} bytes); "
          f"{how} — export PROTENIX_ROOT_DIR={root}", file=out, flush=True)
    return EXIT_OK


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1 or argv[0].startswith("-"):
        print("usage: python -m protenix_v1_opt.weights DIR   (run.sh install --weights DIR)", file=sys.stderr)
        return EXIT_USAGE
    return populate(argv[0])


if __name__ == "__main__":
    sys.exit(main())
