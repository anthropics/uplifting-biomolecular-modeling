"""``run.sh install --weights DIR [--model_name evo2_7b|evo2_40b|evo2_1b_base]`` — fetch the model's checkpoint into DIR from the Hugging Face repository and
revision ``stock/PINS.json checkpoints.<model_name>`` pins (``huggingface_hub.hf_hub_download``, the library upstream's own loader uses; a sharded
checkpoint's ``.part<i>`` files are checked against their pins and concatenated in order as upstream does) and check its sha256 against the pin.
DIR is then ``EVO2_OPT_WEIGHTS`` (the route driver's default ``local_path``). A file already in DIR is kept and only checked; a mismatch is
REFUSED by name and left in place."""
import hashlib
import importlib.util
import os
import sys
from typing import Callable, List, Optional

PREFIX = "[evo2-opt install]"
EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2
CHUNK = 8192 * 1024                                                    # upstream's merge chunk (evo2/models.py)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def check_pins_module():
    """``stock/check_pins.py`` imported by path (stdlib only): its ``check_weights`` is the one digest check of a checkpoint against the pin."""
    from evo2_opt import pins
    p = os.path.join(pins.tree_root(), "stock", "check_pins.py")
    spec = importlib.util.spec_from_file_location("evo2_check_pins", p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def hub_download(repo_id: str, filename: str, revision: str, local_dir: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, local_dir=local_dir)


def fetch(weights_dir: str, model_name: str, pins: Optional[dict] = None, download: Optional[Callable[..., str]] = None,
          checker: Optional[Callable[[str, str, dict], List[str]]] = None, out=None) -> int:
    out = out or sys.stdout
    weights_dir = os.path.abspath(weights_dir); os.makedirs(weights_dir, exist_ok=True)
    if pins is None:
        from evo2_opt import pins as P
        pins = P.load()
    ck = (pins.get("checkpoints") or {}).get(model_name)
    if ck is None:
        print(f"{PREFIX} FAILED: no checkpoints.{model_name} in stock/PINS.json (pinned: {', '.join(sorted(pins.get('checkpoints') or {}))})", file=out, flush=True)
        return EXIT_USAGE
    checker = checker or check_pins_module().check_weights
    download = download or hub_download
    target = os.path.join(weights_dir, ck["file"]); parts = list(ck.get("parts") or [])
    if os.path.isfile(target):
        print(f"{PREFIX} {ck['file']}: present ({os.path.getsize(target)} bytes) — kept, checked below", file=out, flush=True)
    else:
        names = [p["file"] for p in parts] or [ck["file"]]
        print(f"{PREFIX} {ck['file']}: fetching {' + '.join(names)} from {ck['hf_repo']} @ {ck['hf_commit'][:12]} ({ck.get('bytes', '?')} bytes) into {weights_dir}", file=out, flush=True)
        for name in names:
            try:
                got = download(repo_id=ck["hf_repo"], filename=name, revision=ck["hf_commit"], local_dir=weights_dir)
            except Exception as e:  # noqa: BLE001 — the library's own error (network, disk), relayed by name; nothing is deleted
                print(f"{PREFIX} FAILED fetching {name}: {type(e).__name__}: {e}", file=out, flush=True); return EXIT_FAIL
            if not os.path.isfile(os.path.join(weights_dir, name)):
                print(f"{PREFIX} FAILED: {name} is not in {weights_dir} after the download (the library returned {got})", file=out, flush=True); return EXIT_FAIL
        if parts:
            for part in parts:
                digest = sha256_file(os.path.join(weights_dir, part["file"]))
                if digest != part["sha256"]:
                    print(f"{PREFIX} REFUSED: {part['file']} sha256 {digest[:16]}… is not the pin's {part['sha256'][:16]}… — left in place; remove it and re-run", file=out, flush=True)
                    return EXIT_FAIL
            tmp = target + ".merge_tmp"
            with open(tmp, "wb") as dst:
                for part in parts:
                    with open(os.path.join(weights_dir, part["file"]), "rb") as src:
                        for b in iter(lambda: src.read(CHUNK), b""):
                            dst.write(b)
            os.replace(tmp, target)
    bad = checker(model_name, target, pins)
    if bad:
        for b in bad:
            print(f"{PREFIX} REFUSED: {b} — {target} left in place; remove it and re-run to fetch afresh", file=out, flush=True)
        return EXIT_FAIL
    if parts and all(os.path.isfile(os.path.join(weights_dir, p["file"])) for p in parts):
        for part in parts:
            os.remove(os.path.join(weights_dir, part["file"]))
    print(f"{PREFIX} WEIGHTS OK: {ck['file']} in {weights_dir} has the pinned sha256 — export EVO2_OPT_WEIGHTS={weights_dir}", file=out, flush=True)
    return EXIT_OK


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    model_name = "evo2_7b"
    if "--model_name" in argv:
        i = argv.index("--model_name")
        if i + 1 >= len(argv):
            argv = []
        else:
            model_name = argv[i + 1]; del argv[i:i + 2]
    if len(argv) != 1 or argv[0].startswith("-"):
        print("usage: python -m evo2_opt.weights DIR [--model_name evo2_7b|evo2_40b|evo2_1b_base]   (run.sh install --weights DIR)", file=sys.stderr); return EXIT_USAGE
    return fetch(argv[0], model_name)


if __name__ == "__main__":
    sys.exit(main())
