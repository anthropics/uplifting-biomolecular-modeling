"""Weights resolution and comparison with stock/PINS.json (HF url + revision + sha256 + bytes per file).
Layout of a weights root: <root>/<hf-repo-name>/{config.json, README.md, weights/<file>.pth}; a directory that holds an extracted
``atlasfold_weights_v1.0.0/`` tree of that layout is accepted as the root too."""
import hashlib
import json
import os
from typing import Dict, Optional

ENV_DIR = "ATLASFOLD_WEIGHTS_DIR"
FILES = {   # stock model name -> (hf repo dir, relative file)
    "lm": ("atlaslm-3b-base", "weights/atlaslm_3b_base.pth"),
    "monomer": ("atlasfold-260703", "weights/atlasfold-260703.pth"),
    "multimer": ("atlasfold-m-260725", "weights/atlasfold-m-260725.pth"),
}


def kit_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))          # <engine>/opt/atlasfold_opt
    return os.path.dirname(os.path.dirname(here))               # <engine>


def pins_path() -> str:
    return os.path.join(kit_root(), "stock", "PINS.json")


def load_pins() -> dict:
    with open(pins_path()) as f:
        return json.load(f)


def weights_dir(cli_value: Optional[str] = None) -> Optional[str]:
    d = cli_value or os.environ.get(ENV_DIR)
    if d and os.path.isdir(os.path.join(d, "atlasfold_weights_v1.0.0")):   # a dir holding an extracted atlasfold_weights_v1.0.0/ tree
        d = os.path.join(d, "atlasfold_weights_v1.0.0")
    return d


def paths(root: str) -> Dict[str, str]:
    return {k: os.path.join(root, repo, rel) for k, (repo, rel) in FILES.items()}


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def compare(root: str, quick: bool = False) -> dict:
    """{ok, checked, missing, mismatched, files:{key:{path,bytes,sha256?}}} against PINS.json 'weights' entries (bytes always; sha256 unless quick)."""
    pins = {os.path.basename(w["file"]): w for w in load_pins().get("weights", []) if w["file"].endswith(".pth")}
    out = {"ok": True, "checked": 0, "missing": [], "mismatched": [], "files": {}}
    for key, p in paths(root).items():
        pin = pins.get(os.path.basename(p))
        if not os.path.isfile(p):
            out["missing"].append(p); out["ok"] = False; continue
        rec = {"path": p, "bytes": os.path.getsize(p)}
        if pin and rec["bytes"] != pin["bytes"]:
            out["mismatched"].append(f"{p}:bytes"); out["ok"] = False
        if not quick:
            rec["sha256"] = sha256_file(p)
            if pin and rec["sha256"] != pin["sha256"]:
                out["mismatched"].append(f"{p}:sha256"); out["ok"] = False
        out["files"][key] = rec; out["checked"] += 1
    return out
