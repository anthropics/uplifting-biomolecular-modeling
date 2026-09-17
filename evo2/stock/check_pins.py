"""The pin check of `run.sh install` (stdlib only; also runnable alone):

    python stock/check_pins.py [--weights CKPT --model_name evo2_7b|evo2_40b] [--gpu] [--quiet]

(1) evo2 and vtx are installed at the versions stock/PINS.json pins AND their installed files are byte-identical to the carried PyPI wheels
    (stock/*.whl RECORD digests) — anything else is another stock: REFUSED, exit 3;
(2) the library stack (torch / triton / flash-attn / numpy / transformer_engine) against the pins of the stack this python matches
    (Transformer Engine importable -> stacks.img_full, absent -> stacks.img_a100): a version off its pin is NAMED (unlisted), never refused;
(3) --weights: the checkpoint's sha256 against checkpoints.<model_name> (REFUSED on mismatch);
(4) --gpu: the visible device against gpus (named listed / unlisted).
"""
import argparse
import base64
import hashlib
import importlib.util
import json
import os
import sys
import zipfile
from importlib import metadata as md

HERE = os.path.dirname(os.path.abspath(__file__))
NAMED = ("torch", "triton", "flash_attn", "numpy", "transformer_engine")


def sha256_file(path, chunk=16 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def wheel_record(wheel_path):
    """{relative path: sha256 hex} from the wheel's RECORD (entries are 'sha256=<base64url>')."""
    rec = {}
    with zipfile.ZipFile(wheel_path) as z:
        name = [n for n in z.namelist() if n.endswith(".dist-info/RECORD")][0]
        for line in z.read(name).decode().splitlines():
            parts = line.split(",")
            if len(parts) >= 2 and parts[1].startswith("sha256="):
                rec[parts[0]] = base64.urlsafe_b64decode(parts[1][len("sha256="):] + "==").hex()
    return rec


def dist_version(name):
    for n in (name, name.replace("_", "-"), name + "_cu12"):
        try:
            return md.version(n)
        except md.PackageNotFoundError:
            continue
    return None


def check_dist(name, pin):
    """(1): the version pin and byte equality of the installed files against the carried wheel."""
    try:
        dist = md.distribution(name)
    except md.PackageNotFoundError:
        return [f"{name}: not installed (the stock is {name}=={pin['version']})"], None
    bad = []
    if dist.version != pin["version"]:
        bad.append(f"{name} {dist.version} is installed: the stock is {pin['version']}")
    wheel = os.path.join(HERE, os.path.basename(pin["archives"]["wheel"]["file"]))
    if not os.path.exists(wheel):
        return bad + [f"{name}: the carried wheel {wheel} is absent from stock/"], None
    n_ok, modified, missing = 0, [], []
    for rel, want in wheel_record(wheel).items():
        if ".dist-info/" in rel:
            continue
        p = dist.locate_file(rel)
        if not os.path.exists(p):
            missing.append(rel)
        elif sha256_file(p) == want:
            n_ok += 1
        else:
            modified.append(rel)
    if modified or missing:
        bad.append(f"{name}: installed files differ from the carried wheel — modified {modified[:5]}, missing {missing[:5]}")
    return bad, f"{name} {dist.version}: pinned ({n_ok} installed files byte-identical to {os.path.basename(wheel)})"


def check_stack(pins):
    """(2): the stack's library versions, named against the matching stack's pins."""
    te = dist_version("transformer_engine") if importlib.util.find_spec("transformer_engine") is not None else None
    sid = next((s for s, st in pins["stacks"].items() if (st["pins"].get("transformer_engine") != "absent") == (te is not None)), None)
    if sid is None:
        return [f"stack: no stacks entry for transformer_engine {'present' if te else 'absent'}"]
    notes = [f"stack {sid} ({'transformer_engine ' + te if te else 'transformer_engine absent'}): serves {', '.join(pins['stacks'][sid]['serves'])}"]
    for name in NAMED:
        want = pins["stacks"][sid]["pins"].get(name)
        got = te if name == "transformer_engine" else dist_version(name)
        if want in (None, "absent") or got is None:
            continue
        if str(got).split("+")[0] != str(want).split("+")[0]:
            notes.append(f"unlisted: {name} {got} != the pinned {want} (stacks.{sid}) — the kit engages and names it")
    return notes


def check_weights(model_name, path, pins):
    ck = (pins.get("checkpoints") or {}).get(model_name)
    if ck is None:
        return [f"--weights: no checkpoints.{model_name} pin (pinned: {sorted(pins.get('checkpoints') or {})})"]
    if not os.path.isfile(path):
        return [f"--weights: {path} is not a file"]
    got = sha256_file(path)
    return [] if got == ck["sha256"] else [f"{model_name}: {path} sha256 {got[:16]}… is not the pin's {ck['sha256'][:16]}… (checkpoints.{model_name})"]


def check_gpu(pins):
    try:
        import torch
    except ImportError:
        return ["--gpu: torch is not importable"]
    if not torch.cuda.is_available():
        return ["--gpu: no CUDA device visible"]
    notes = []
    for i in range(torch.cuda.device_count()):
        cc = torch.cuda.get_device_capability(i); sm = f"sm_{cc[0]}{cc[1]}"
        mib = torch.cuda.get_device_properties(i).total_memory // (1024 * 1024)
        cls = next((k for k, v in pins["gpus"].items() if v["sm"] == sm and abs(mib - v["memory_mib"]) <= 0.1 * v["memory_mib"]), None)
        notes.append(f"gpu {i}: {torch.cuda.get_device_name(i)} {sm} {mib} MiB — " + (f"listed ({cls})" if cls else "unlisted (not a class the kit was measured on: engaged and named)"))
    return notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    ap.add_argument("--model_name", default="evo2_7b")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    pins = json.load(open(os.path.join(HERE, "PINS.json")))
    bad, notes = [], []
    for name, pin in pins["upstream"].items():
        b, note = check_dist(name, pin)
        bad += b
        if note and not b:
            notes.append(note)
    notes += check_stack(pins)
    if a.weights:
        b = check_weights(a.model_name, a.weights, pins); bad += b
        if not b:
            notes.append(f"{a.model_name}: {a.weights} sha256 == checkpoints.{a.model_name}")
    if a.gpu:
        notes += check_gpu(pins)
    if not a.quiet:
        for n in notes:
            print(f"check_pins: {n}")
    if bad:
        for b in bad:
            print(f"check_pins: REFUSED {b}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
