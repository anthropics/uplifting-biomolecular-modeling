"""`run.sh install --weights DIR` — stage what stock E1 reads from the Hugging Face cache into DIR (an ``HF_HOME``) with upstream's own
downloaders, then check every pinned file against the kit's pins (``stock/PINS.json`` = the kit's pins module): the ``model.safetensors``
of each size (``Profluent-Bio/E1-150m|300m|600m`` at the pinned revisions) and the hub RMSNorm kernel snapshot
(``kernels-community/triton-layer-norm`` at the pinned revision). DIR is then the ``HF_HOME`` every route reads (README.md, Setup);
nothing is fetched at run time.

Upstream owns the transfers. The checkpoints come through ``huggingface_hub.snapshot_download(repo, revision=<pinned>)`` — the call
``E1ForMaskedLM.from_pretrained`` makes underneath for a repo id — into ``DIR/hub``, the library's own cache layout
(``models--Profluent-Bio--E1-<size>/snapshots/<rev>/model.safetensors``); a snapshot that already holds its ``model.safetensors`` is kept
and only checked. The kernel's package directory ships in the tree (``stock/hub_kernel/triton_layer_norm``: the files of
``build/torch-universal/triton_layer_norm`` at the pinned revision, ``SOURCE.md`` and ``LICENSE`` beside them) and is copied — no network —
into the hub cache layout ``kernels.install_kernel`` writes and ``E1.modeling``'s ``get_kernel`` reads at import
(``models--kernels-community--triton-layer-norm/snapshots/<rev>/build/torch-universal/triton_layer_norm``); upstream resolves the kernel
by its branch name ``main``, which an offline cache answers from ``refs/main``, so this step points that ref at the pinned revision (a ref
already naming another revision is reported and left alone: the step fails). An injected ``install`` callable (the package tests), or a
tree without the copy, goes through ``kernels.install_kernel(repo, revision=<pinned>)`` instead. A file whose digest is not the pin's is named and the step fails (exit 1) — the file is left in place for inspection, never
deleted; so is a fetch that upstream's code could not complete (its own error, relayed by name). The comparisons are the routes' own:
``pins.assert_weights`` (the weights line's word) and ``stack.kernel_snapshot`` (the kernel gate). For this process DIR is bound as
``HF_HOME`` / ``HF_HUB_CACHE`` and the offline switches are lifted, so the libraries read and write DIR exactly as the routes will read it.

    python -m e1_opt.weights DIR                     # DIR = an HF_HOME: the three checkpoints and the kernel snapshot under DIR/hub
    python -m e1_opt.weights --kernels-cache DIR     # DIR = a KERNELS_CACHE root: the kernel snapshot only, from the tree's copy (the container image keeps one at /opt/kernels_cache)
"""
from __future__ import annotations

import os
import shutil
import sys
from typing import Callable, List, Optional

from . import stack

PREFIX = "[e1-opt install]"
EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2
KERNEL_FILE = os.path.join("build", "torch-universal", "triton_layer_norm", "layer_norm.py")   # the pinned file of the kernel snapshot (stock/PINS.json hub_kernel.file)
VENDORED_KERNEL_RELDIR = os.path.join("stock", "hub_kernel", "triton_layer_norm")                  # the tree's copy of the snapshot's package directory (the files
                                                                                                # of build/torch-universal/triton_layer_norm at the pinned revision)
OFFLINE_SWITCHES = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")                                  # configs/<card>.env sets them for the routes; this step fetches


def bind_env(directory: str, kernels_cache: bool, out) -> None:
    """Bind DIR for this process the way the routes will read it: as HF_HOME (+ HF_HUB_CACHE = DIR/hub, the kernels package's own cache
    variables unset so the kernel lands and is looked for under DIR/hub) or, in the --kernels-cache form, as KERNELS_CACHE; the offline
    switches are lifted (named when they were set). Must run before huggingface_hub / kernels are imported: both read the variables at import."""
    lifted = [k for k in OFFLINE_SWITCHES if os.environ.pop(k, None) not in (None, "", "0")]
    if lifted:
        print(f"{PREFIX} {' and '.join(lifted)} lifted for this step (it fetches; the routes run offline)", file=out, flush=True)
    if kernels_cache:
        os.environ["KERNELS_CACHE"] = directory
        os.environ.pop("HF_KERNELS_CACHE", None)
    else:
        os.environ["HF_HOME"] = directory
        os.environ["HF_HUB_CACHE"] = os.path.join(directory, "hub")
        for k in stack.KERNEL_CACHE_ENVS:                                    # the kernels package's cache root variables
            os.environ.pop(k, None)


def kernel_repo_dir(cache_root: str, pins) -> str:
    return os.path.join(cache_root, "models--" + pins.KERNEL["repo"].replace("/", "--"))


def kernel_snapshot_dir(cache_root: str, pins) -> str:
    return os.path.join(kernel_repo_dir(cache_root, pins), "snapshots", pins.KERNEL["rev"])


def upstream_download():
    """``huggingface_hub.snapshot_download`` — the transfer transformers' ``from_pretrained`` runs for a repo id."""
    from huggingface_hub import snapshot_download
    return snapshot_download


def upstream_install_kernel():
    """``kernels.install_kernel`` — the transfer ``E1.modeling``'s ``get_kernel`` runs at import (it reads KERNELS_CACHE when the package is imported)."""
    from kernels import install_kernel
    return install_kernel


def fetch_checkpoints(hf_home: str, pins, download: Optional[Callable] = None, out=None) -> Optional[str]:
    """Fetch each size's snapshot that lacks its model.safetensors; returns None, or the failure worded by name."""
    hub = os.path.join(hf_home, "hub")
    for size, w in pins.WEIGHTS.items():
        snap = pins.weights_snapshot(size, hf_home)
        target = os.path.join(snap, "model.safetensors")
        if os.path.isfile(target):
            print(f"{PREFIX} {size} {w['repo']}@{w['rev'][:8]} model.safetensors: present", file=out, flush=True)
            continue
        print(f"{PREFIX} {size} {w['repo']}@{w['rev'][:8]} model.safetensors: fetching ({w['bytes']} bytes, huggingface_hub.snapshot_download)", file=out, flush=True)
        try:
            if download is None:
                download = upstream_download()
            download(repo_id=w["repo"], revision=w["rev"], cache_dir=hub)
        except Exception as e:  # noqa: BLE001 — upstream's own error (network, disk, a gated repository), relayed by name; nothing is deleted
            return f"FAILED fetching {w['repo']}@{w['rev']}: {type(e).__name__}: {e} — anything already under {hub} is left in place"
        if not os.path.isfile(target):
            return f"FAILED: {size}: no model.safetensors at {snap} after upstream's fetch"
    return None


def vendored_kernel_dir() -> str:
    """The kernel's package directory shipped in the tree (stock/hub_kernel/triton_layer_norm; SOURCE.md beside it names its origin)."""
    return os.path.join(stack.tree_home(), VENDORED_KERNEL_RELDIR)


def fetch_kernel(cache_root: str, pins, install: Optional[Callable] = None, out=None) -> Optional[str]:
    """Stage the kernel snapshot into the cache root when its layer_norm.py is absent — a copy of the tree's package directory into the
    layout the kernels package reads (no network), or, with an injected ``install`` / a tree without the copy, upstream's
    ``kernels.install_kernel`` — and point refs/main at the pinned revision; returns None, or the failure worded by name."""
    k = pins.KERNEL
    snap = kernel_snapshot_dir(cache_root, pins)
    target = os.path.join(snap, KERNEL_FILE)
    vendored = vendored_kernel_dir()
    if os.path.isfile(target):
        print(f"{PREFIX} hub kernel {k['repo']}@{k['rev'][:8]}: present", file=out, flush=True)
    elif install is None and os.path.isfile(os.path.join(vendored, "layer_norm.py")):
        print(f"{PREFIX} hub kernel {k['repo']}@{k['rev'][:8]}: staging the tree's copy ({VENDORED_KERNEL_RELDIR}) into {snap} — no network", file=out, flush=True)
        try:
            shutil.copytree(vendored, os.path.dirname(target), dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
        except OSError as e:
            return f"FAILED staging the hub kernel {k['repo']}@{k['rev']} into {snap}: {type(e).__name__}: {e}"
    else:
        print(f"{PREFIX} hub kernel {k['repo']}@{k['rev'][:8]}: fetching (kernels.install_kernel)", file=out, flush=True)
        try:
            if install is None:
                install = upstream_install_kernel()
            install(k["repo"], revision=k["rev"])
        except Exception as e:  # noqa: BLE001 — upstream's own error, relayed by name
            return f"FAILED fetching the hub kernel {k['repo']}@{k['rev']}: {type(e).__name__}: {e}"
        if not os.path.isfile(target):
            return (f"FAILED: {KERNEL_FILE} is not at {snap} after upstream's fetch (the kernels package read its cache root before this step bound "
                    f"{cache_root} — run this step in a fresh interpreter: python -m e1_opt.weights …)")
    ref = os.path.join(kernel_repo_dir(cache_root, pins), "refs", "main")
    have = open(ref, encoding="utf-8").read().strip() if os.path.isfile(ref) else None
    if have == k["rev"]:
        return None
    if have is not None:
        return (f"FAILED: {ref} names revision {have}, not the pinned {k['rev']} — upstream loads the kernel by the name `main`, which this cache answers "
                f"with that other revision; the ref is left as found (point it at the pinned revision, or use a cache root of its own)")
    os.makedirs(os.path.dirname(ref), exist_ok=True)
    with open(ref, "w", encoding="utf-8") as fh:
        fh.write(k["rev"])
    print(f"{PREFIX} hub kernel refs/main -> {k['rev'][:8]} (upstream asks for `main`; offline the cache answers with the pinned revision)", file=out, flush=True)
    return None


def check(pins, kernel_snapshot: Callable[[], dict], sizes: List[str], out) -> tuple:
    """The routes' own comparisons: pins.assert_weights per size (word pinned | not-pinned) and the kernel gate. Returns (n ok, n files, [names off the pin])."""
    bad, n = [], 0
    for size in sizes:
        n += 1
        try:
            rec = pins.assert_weights(size)
        except Exception as e:  # noqa: BLE001 — PinDrift: the file is not there
            bad.append(f"{size} model.safetensors ({e})"); continue
        if rec.get("word") != pins.WEIGHTS_WORDS[0]:
            print(f"{PREFIX} {size} model.safetensors: sha256 {rec.get('sha256', '')[:16]}… ({rec.get('bytes')} bytes) is not the pin's {rec.get('pinned_sha256', '')[:16]}…", file=out, flush=True)
            bad.append(f"{size} model.safetensors")
    n += 1
    kern = kernel_snapshot()
    if not kern.get("present"):
        bad.append(f"hub kernel snapshot {pins.KERNEL['rev'][:8]} (not found under {kern.get('searched')})")
    elif kern.get("layer_norm_py_sha256_ok") is not True:
        print(f"{PREFIX} hub kernel {kern.get('dir')}: {KERNEL_FILE} does not hash to the pin", file=out, flush=True)
        bad.append(f"hub kernel {KERNEL_FILE}")
    return n - len(bad), n, bad


def fetch(directory: str, kernels_cache: bool = False, pins=None, download: Optional[Callable] = None, install: Optional[Callable] = None,
          kernel_snapshot: Optional[Callable[[], dict]] = None, out=None) -> int:
    """Fetch what is absent, then check every pinned file. ``pins`` (the kit's pins module), ``download`` / ``install`` (upstream's downloaders)
    and ``kernel_snapshot`` (``stack.kernel_snapshot``) are injectable for the package tests."""
    out = out or sys.stdout
    directory = os.path.abspath(directory)
    os.makedirs(directory, exist_ok=True)
    bind_env(directory, kernels_cache, out)
    if pins is None:
        pins = stack.load_pins()
    if kernel_snapshot is None:
        kernel_snapshot = lambda: stack.kernel_snapshot(pins)          # noqa: E731 — the routes' kernel gate, reading the variables bound above
    cache_root = directory if kernels_cache else os.path.join(directory, "hub")
    why = None if kernels_cache else fetch_checkpoints(directory, pins, download, out)
    why = why or fetch_kernel(cache_root, pins, install, out)
    if why:
        print(f"{PREFIX} {why}", file=out, flush=True)
        return EXIT_FAIL
    ok, n, bad = check(pins, kernel_snapshot, [] if kernels_cache else list(pins.WEIGHTS), out)
    what, var = ("KERNEL", "KERNELS_CACHE") if kernels_cache else ("WEIGHTS", "HF_HOME")
    if bad:
        print(f"{PREFIX} REFUSED: {len(bad)} of {n} pinned files under {directory} do not check against the pins (stock/PINS.json): {', '.join(bad)} — "
              f"left in place; remove them and re-run this step to fetch afresh", file=out, flush=True)
        return EXIT_FAIL
    print(f"{PREFIX} {what} OK: {ok}/{n} pinned files under {directory} have the pinned digest — export {var}={directory}", file=out, flush=True)
    return EXIT_OK


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    kernels_cache = False
    if argv and argv[0] == "--kernels-cache":
        kernels_cache, argv = True, argv[1:]
    if len(argv) != 1 or argv[0].startswith("-"):
        print("usage: python -m e1_opt.weights [--kernels-cache] DIR   (run.sh install --weights DIR)", file=sys.stderr)
        return EXIT_USAGE
    return fetch(argv[0], kernels_cache=kernels_cache)


if __name__ == "__main__":
    sys.exit(main())
