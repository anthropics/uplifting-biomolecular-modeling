"""The parameters step of `run.sh install` (`run.sh install --weights DIR`, or AF3_JAX_PARAMS_ROOT set): the public OF3-p2 checkpoint
fetched and converted with the fork's own converter. Converted parameters
already under <root>/<variant>/ with the pinned digest are kept and reported (``CONVERT … status=PRESENT``), a checkpoint already in the
checkpoint directory is kept; nothing is deleted. The tree overlays nothing onto the install: the pinned stock is the pristine fork,
and the kit's script is copied beside the stock one by `pred`/`warm` themselves. The converter itself is the stock script, never a kit file;
the checkpoint's sha256 is checked against the pin before conversion and the converted ``of3_ported_weights.bin.zst`` is hashed and
compared with the pin afterward. Needs the fork's environment plus ``torch`` (CPU is fine; the converter's only extra dependency).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
from typing import List, Optional

from . import report as _report, stack, variants as _variants

CONVERTER = "convert_of3_weights.py"


def fetch(variant: str, out_dir: str) -> dict:
    ck = _variants.info(variant)["checkpoint"]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, ck["file"])
    res = {"variant": variant, "url": ck["url"], "path": path, "exit_code": 0}
    tmp = path + ".part"
    try:
        with urllib.request.urlopen(ck["url"]) as r, open(tmp, "wb") as f:
            for chunk in iter(lambda: r.read(1 << 22), b""):
                f.write(chunk)
        os.replace(tmp, path)
    except (OSError, urllib.error.URLError) as e:
        res.update(exit_code=1, ok=False, reason=f"fetch failed: {e}")
        return res
    res.update(check_checkpoint(path, variant))
    if not res["ok"]:
        res["reason"] = "checkpoint sha256/size differ from stock/PINS.json"
    return res


def check_checkpoint(path: str, variant: str) -> dict:
    ck = _variants.info(variant)["checkpoint"]
    size, digest = os.path.getsize(path), stack.sha256_file(path)
    return {"bytes": size, "sha256": digest, "expected_sha256": ck["sha256"], "expected_bytes": ck["bytes"],
            "ok": digest == ck["sha256"] and size == ck["bytes"]}


def converter_path(repo: Optional[str] = None) -> str:
    """The stock converter in the checkout (presence only): never a kit file."""
    repo = repo or stack.repo_dir()
    conv = os.path.join(repo, CONVERTER)
    if not os.path.isfile(conv):
        raise FileNotFoundError(f"no {CONVERTER} under {repo} (the fork checkout)")
    return conv


def convert(variant: str, checkpoint: str, out_dir: Optional[str] = None, repo: Optional[str] = None, py: Optional[str] = None) -> dict:
    repo, py = repo or stack.repo_dir(), py or stack.venv_python()
    out_dir = out_dir or _variants.params_dir(variant)
    if not out_dir:
        raise ValueError(f"{stack.ENV_PARAMS_ROOT} is not set and no --output_dir given")
    conv = converter_path(repo)
    res: dict = {"variant": variant, "checkpoint": os.path.abspath(checkpoint), "out_dir": out_dir, "converter": conv}
    res["checkpoint_check"] = check_checkpoint(checkpoint, variant)
    if not res["checkpoint_check"]["ok"]:
        res["status"] = "FAIL"; res["reason"] = "checkpoint sha256/size differ from stock/PINS.json"
        return res
    os.makedirs(out_dir, exist_ok=True)
    cmd = [py, conv, "--of3_checkpoint", os.path.abspath(checkpoint), "--output_dir", out_dir]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=repo, text=True, capture_output=True, env=stack.model_process_env())
    sys.stderr.write(p.stdout[-3000:] + p.stderr[-2000:])
    res.update(command=cmd, exit_code=p.returncode, wall_s=round(time.time() - t0, 1))
    res["params"] = _variants.check_params(variant, root=os.path.dirname(out_dir), digest=True) if os.path.basename(out_dir) == variant \
        else {"present": os.path.isfile(os.path.join(out_dir, _variants.PARAMS_FILE))}
    res["status"] = "PASS" if p.returncode == 0 and res["params"].get("present") and res["params"].get("matches_pin") else "FAIL"
    if res["status"] == "FAIL":
        res["reason"] = (f"converter exited {p.returncode}" if p.returncode else
                         (res["params"].get("reason") or "no converted parameters written"))
    return res


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m af3_jax_opt.convert", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fetch"); p.add_argument("--variant", required=True, choices=_variants.VARIANTS); p.add_argument("out_dir")
    p = sub.add_parser("convert"); p.add_argument("--variant", required=True, choices=_variants.VARIANTS)
    p.add_argument("--checkpoint", required=True); p.add_argument("--output_dir", default=None)
    p = sub.add_parser("install"); p.add_argument("--variant", action="append", choices=_variants.VARIANTS, default=None)
    p.add_argument("--checkpoint_dir", default=None, help="where the public checkpoint is (or is fetched to)")
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        r = fetch(a.variant, a.out_dir)
        _report.emit(_report.line("FETCH", variant=a.variant, path=r.get("path"), sha256_ok=r.get("ok"), rc=r["exit_code"], reason=r.get("reason")))
        return 0 if r.get("ok") else 1
    if a.cmd == "convert":
        r = convert(a.variant, a.checkpoint, a.output_dir)
        _report.emit(_report.line("CONVERT", variant=a.variant, status=r["status"], out_dir=r["out_dir"],
                                  sha256=(r.get("params") or {}).get("sha256"), matches_pin=(r.get("params") or {}).get("matches_pin"),
                                  reason=r.get("reason")))
        return 0 if r["status"] == "PASS" else 1
    if a.cmd == "install":
        rc = 0
        ck_dir = a.checkpoint_dir or os.path.join(stack.params_root() or ".", "checkpoints")
        for v in a.variant or list(_variants.VARIANTS):
            have = _variants.check_params(v, digest=True)                  # converted parameters already under <root>/<variant>/ with the pinned digest are kept and reported, not made again
            if have.get("present") and have.get("matches_pin"):
                _report.emit(_report.line("CONVERT", variant=v, status="PRESENT", out_dir=_variants.params_dir(v), sha256=have.get("sha256"), matches_pin=True))
                continue
            path = os.path.join(ck_dir, _variants.info(v)["checkpoint"]["file"])
            if not os.path.isfile(path):
                r = fetch(v, ck_dir)
                _report.emit(_report.line("FETCH", variant=v, path=r.get("path"), sha256_ok=r.get("ok"), rc=r["exit_code"], reason=r.get("reason")))
                if not r.get("ok"):
                    rc = 1; continue
            r = convert(v, path)
            _report.emit(_report.line("CONVERT", variant=v, status=r["status"], out_dir=r["out_dir"],
                                      sha256=(r.get("params") or {}).get("sha256"), matches_pin=(r.get("params") or {}).get("matches_pin"),
                                      reason=r.get("reason")))
            rc = rc or (0 if r["status"] == "PASS" else 1)
        return rc
    return 2


if __name__ == "__main__":
    sys.exit(main())
