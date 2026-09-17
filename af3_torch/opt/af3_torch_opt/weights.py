"""`run.sh install --weights DIR [--fetch]` — the weights step of the install: the converted checkpoint under DIR checked against stock/PINS.json,
and, with --fetch, made when DIR has none.

The kit's weights are the OpenFold3-preview2 parameters converted to this port's layout — stock/PINS.json `variants.p2.converted`
(of3_ported_weights.bin.zst) with `variants.p2.conventions` (of3_conventions.json, optional) beside it. The step judges DIR with the kit's ONE
digest judge, stock/check_pins.py `digest_report` (the words `run.sh check` prints too):
  pinned   — DIR holds the pinned bytes: WEIGHTS OK, exit 0;
  unpinned — DIR holds another parameters file in that layout (*.bin.zst | *.bin): WEIGHTS UNPINNED, exit 0 (it runs; the kit's tests and
             timings cover the pinned checkpoint only — the same rule every verb applies);
  missing  — no parameters file under DIR: WEIGHTS MISSING with the two ways to obtain it, exit 1 — unless --fetch: then the public
             OpenFold3-preview2 checkpoint (`variants.p2.checkpoint`: url, sha256) is downloaded into DIR/checkpoint/ (a copy already there with
             the pinned digest is kept), converted into DIR by the reference fork's own converter (`variants.p2.converter`: AF3_TORCH_JAX_REPO's
             convert_of3_weights.py on AF3_TORCH_JAX_PY — the JAX environment carries its torch (CPU) and zstandard), and DIR is judged again:
             WEIGHTS OK | WEIGHTS UNPINNED as above, or WEIGHTS FETCH FAILED / WEIGHTS CONVERT FAILED by name, exit 1. Nothing under DIR is deleted.
Without --fetch nothing is downloaded. DIR is then the value of AF3_TORCH_PARAMS_DIR (README.md 'Setup').

    python -m af3_torch_opt.weights DIR [--fetch]        (exit: 0 ok · 1 missing / failed · 2 usage)
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Callable, List, Optional

from . import stack

PREFIX = "[af3-torch-opt install]"
EXIT_OK, EXIT_MISSING, EXIT_USAGE = 0, 1, 2


def check(directory: str, pins: Optional[dict] = None, judge: Optional[Callable[[dict], dict]] = None, out=sys.stdout, fetch: bool = False,
          opener=urllib.request.urlopen) -> int:
    """Judge the checkpoint under `directory` against the pins and print ONE verdict line. `pins` defaults to stock/PINS.json, `judge` to
    stock/check_pins.py digest_report (it reads AF3_TORCH_PARAMS_DIR, which this step sets to `directory` for the judgement). `fetch`: a
    directory without a checkpoint gets one (fetch_and_convert) and is judged again."""
    pins = stack.pins() if pins is None else pins
    judge = stack.pins_tool().digest_report if judge is None else judge
    d = os.path.abspath(directory)
    os.environ["AF3_TORCH_PARAMS_DIR"] = d
    rep = judge(pins)
    spec = pins["variants"]["p2"]; pinned, conv = spec["converted"], spec.get("conventions")
    vrep = (rep.get("variants") or {}).get("p2") or {}
    crep = vrep.get("conventions") or {}
    conv_words = ""
    if conv:
        conv_words = (f"; {conv['file']} beside it " + ("has the pinned digest" if crep.get("ok") else "is present with another digest (reported, not judged)" if crep.get("present")
                      else "is absent (optional: the model steps pass it only when present)"))
    verdict = rep.get("verdict")
    if verdict == "pinned":
        print(f"{PREFIX} WEIGHTS OK: {rep['file']} is the pinned {spec['name']} checkpoint (sha256 {rep['sha256'][:16]}…, {pinned['bytes']} bytes){conv_words} "
              f"— export AF3_TORCH_PARAMS_DIR={d}", file=out, flush=True)
        return EXIT_OK
    if verdict == "unpinned":
        print(f"{PREFIX} WEIGHTS UNPINNED: {rep['file']} (sha256 {rep['sha256'][:16]}…) is not the pinned {pinned['file']} ({pinned['sha256'][:16]}…, {pinned['bytes']} bytes) "
              f"— it runs: {stack.UNPINNED_WORDS}{conv_words} — export AF3_TORCH_PARAMS_DIR={d}", file=out, flush=True)
        return EXIT_OK
    if fetch:
        rc = fetch_and_convert(d, pins, out=out, opener=opener)
        return rc if rc != EXIT_OK else check(d, pins, judge, out, fetch=False)
    ck = spec.get("checkpoint") or {}
    print(f"{PREFIX} WEIGHTS MISSING: no parameters file (*.bin.zst | *.bin) under {d}; nothing was downloaded. {pinned['file']} ({pinned['bytes']} bytes, "
          f"sha256 {pinned['sha256'][:16]}…) is the {spec['name']} checkpoint converted to this port's layout — {spec['source']}. Either re-run this step with --fetch "
          f"(`run.sh install --weights {d} --fetch`: the public checkpoint{' (%d MB)' % (ck['bytes'] // 10**6) if ck.get('bytes') else ''} is downloaded and converted here on the CPU), "
          f"or put a converted {pinned['file']} {'(and ' + conv['file'] + ') ' if conv else ''}in {d} and re-run it (STOCK.md 'weights').", file=out, flush=True)
    return EXIT_MISSING


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_and_convert(d: str, pins: dict, out=sys.stdout, opener=urllib.request.urlopen, runner=subprocess.run) -> int:
    """--fetch: the public checkpoint (PINS variants.p2.checkpoint) into <d>/checkpoint/ — kept when already there with the pinned digest,
    else downloaded and digest-checked — then the reference fork's converter (PINS variants.p2.converter) run on the JAX interpreter with the
    JAX-process environment, writing the converted parameters into <d>. ONE line per stage (FETCH, CONVERT); EXIT_OK when the converter
    returned 0 and wrote the file (the caller judges its digest), EXIT_MISSING otherwise with the reason by name. Deletes nothing."""
    spec = pins["variants"]["p2"]; ck, cv, pinned = spec["checkpoint"], spec["converter"], spec["converted"]
    jax_py, repo = stack.jax_python(), stack.jax_repo()
    script = os.path.join(repo or "", cv["script"])
    if not (jax_py and os.path.isfile(jax_py) and os.access(jax_py, os.X_OK)):
        print(f"{PREFIX} WEIGHTS CONVERT FAILED: AF3_TORCH_JAX_PY ({jax_py or 'not set'}) is not an executable interpreter — the converter runs on the JAX environment (README.md 'Variables')", file=out, flush=True)
        return EXIT_MISSING
    if not (repo and os.path.isfile(script)):
        print(f"{PREFIX} WEIGHTS CONVERT FAILED: {cv['script']} not found under AF3_TORCH_JAX_REPO ({repo or 'not set'}) — the reference fork checkout of the install (README.md 'Variables')", file=out, flush=True)
        return EXIT_MISSING
    ckdir = os.path.join(d, "checkpoint"); path = os.path.join(ckdir, ck["file"])
    os.makedirs(ckdir, exist_ok=True)
    if os.path.isfile(path) and _sha256(path) == ck["sha256"]:
        print(f"{PREFIX} FETCH kept file={path} sha256={ck['sha256'][:16]}… (already present with the pinned digest)", file=out, flush=True)
    else:
        t0 = time.time(); part = path + ".part"
        try:
            with opener(ck["url"]) as r, open(part, "wb") as f:
                for chunk in iter(lambda: r.read(1 << 22), b""):
                    f.write(chunk)
        except (OSError, urllib.error.URLError) as e:
            print(f"{PREFIX} WEIGHTS FETCH FAILED url={ck['url']} ({e.__class__.__name__}: {e}) — re-run with network access, or download that file into {ckdir} yourself and re-run", file=out, flush=True)
            return EXIT_MISSING
        got = _sha256(part)
        if got != ck["sha256"]:
            print(f"{PREFIX} WEIGHTS FETCH FAILED url={ck['url']} sha256={got[:16]}… is not the pinned {ck['sha256'][:16]}… ({os.path.getsize(part)} bytes; kept as {part} for inspection, not converted)", file=out, flush=True)
            return EXIT_MISSING
        os.replace(part, path)
        print(f"{PREFIX} FETCH downloaded url={ck['url']} file={path} bytes={os.path.getsize(path)} sha256={got[:16]}… (pinned) wall_s={time.time() - t0:.0f}", file=out, flush=True)
    cmd = [jax_py, script, "--of3_checkpoint", path, "--output_dir", d]
    t0 = time.time()
    print(f"{PREFIX} CONVERT running {' '.join(cmd)} (cwd {repo}; the fork's converter, CPU)", file=out, flush=True)
    rc = runner(cmd, cwd=repo, env=stack.model_process_env(jax=True)).returncode
    made = os.path.join(d, pinned["file"])
    if rc != 0 or not os.path.isfile(made):
        print(f"{PREFIX} WEIGHTS CONVERT FAILED rc={rc} — {cv['script']} did not leave {made} (its transcript is above)", file=out, flush=True)
        return EXIT_MISSING
    print(f"{PREFIX} CONVERT done file={made} bytes={os.path.getsize(made)} wall_s={time.time() - t0:.0f}", file=out, flush=True)
    return EXIT_OK


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    fetch = "--fetch" in argv; argv = [x for x in argv if x != "--fetch"]
    if len(argv) != 1 or argv[0].startswith("-"):
        print("usage: python -m af3_torch_opt.weights DIR [--fetch]   (run.sh install --weights DIR [--fetch])", file=sys.stderr)
        return EXIT_USAGE
    return check(argv[0], fetch=fetch)


if __name__ == "__main__":
    sys.exit(main())
