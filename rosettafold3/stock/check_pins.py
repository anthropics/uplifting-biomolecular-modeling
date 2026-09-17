#!/usr/bin/env python
"""stock/check_pins.py — is the upstream package in this interpreter the pinned one? (stdlib only; the one place the pin check lives)

`rc-foundry` is pinned by commit (stock/PINS.json "upstream"); the pin carries no tag and no PyPI release. Two install routes
are accepted, both read from pip's own record of the install (the dist's ``direct_url.json``, PEP 610):
  vcs      ``pip install "rc-foundry[rf3] @ git+https://github.com/RosettaCommons/foundry.git@<commit>"`` (or ``pip install -e .``
           of a checkout at the commit — an editable install records the checkout, whose HEAD is then read with git):
           ``vcs_info.commit_id`` must equal the pin;
  archive  ``pip install "stock/foundry-4010e3e2e.tar.gz[rf3]"``: ``archive_info.hash`` must equal ``sha256=<the tracked tarball's own sha256, hashed live>``
           (the archive is cut without ``.git``, so hatch-vcs gives it version 0.0.0 — the version string is recorded, not gated).
Anything else (no dist, no direct_url.json, another commit, another archive) fails with the reason. Exit 0 / 1; ``--json`` prints
the result dict as the last line; ``--python <interpreter>`` runs the same check in another interpreter.
"""
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DIST = "rc-foundry"


def pins():
    return json.load(open(os.path.join(HERE, "PINS.json"), "r", encoding="utf-8"))


def archive_sha():
    """The tracked archive's own sha256, hashed live from the file (no stored manifest of it)."""
    p = pins()
    name = p["archive"]["filename"]
    path = os.path.join(HERE, name)
    if not os.path.isfile(path):
        return None, name
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest(), name


def _git_head(path):
    try:
        r = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def check():
    p = pins()
    want = p["upstream"]["commit"]
    out = {"ok": False, "dist": DIST, "route": "none", "version": None, "commit": None, "archive_sha256": None, "pin": want, "detail": None}
    try:
        d = importlib.metadata.distribution(DIST)
    except importlib.metadata.PackageNotFoundError:
        out["detail"] = f"{DIST} is not installed in {sys.executable}"
        return out
    out["version"] = d.version
    du = d.read_text("direct_url.json")
    if not du:
        out["detail"] = f"{DIST} {d.version} has no direct_url.json (not installed from the pinned git commit nor from stock/{p['archive']['filename']})"
        return out
    du = json.loads(du)
    vcs = du.get("vcs_info") or {}
    if vcs.get("commit_id"):
        out["route"], out["commit"] = "vcs", vcs["commit_id"]
        out["ok"] = vcs["commit_id"] == want
        out["detail"] = "commit == pin" if out["ok"] else f"commit {vcs['commit_id'][:12]} != pin {want[:12]}"
        return out
    if du.get("dir_info", {}).get("editable") and du.get("url", "").startswith("file://"):
        head = _git_head(du["url"][7:])
        out["route"], out["commit"] = "editable", head
        out["ok"] = head == want
        out["detail"] = "checkout HEAD == pin" if out["ok"] else f"checkout HEAD {head} != pin {want[:12]}"
        return out
    ah = (du.get("archive_info") or {}).get("hash") or ""
    if ah:
        want_sha, name = archive_sha()
        got = ah.split("=", 1)[-1]
        out["route"], out["archive_sha256"] = "archive", got
        out["ok"] = bool(want_sha) and got == want_sha and du.get("url", "").endswith(name)
        out["detail"] = f"archive sha256 == SHA256SUMS ({name})" if out["ok"] else f"archive {du.get('url')} sha256 {got[:12]} != {str(want_sha)[:12]} ({name})"
        return out
    out["detail"] = f"direct_url.json of {DIST} names neither a git commit nor an archive hash: {du}"
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--python" in argv:
        py = argv[argv.index("--python") + 1]
        r = subprocess.run([py, os.path.abspath(__file__)] + [a for a in argv if a not in ("--python", py)], text=True)
        return r.returncode
    res = check()
    if "--json" in argv:
        print(json.dumps(res))
    else:
        print(f"[check_pins] {'OK' if res['ok'] else 'FAIL'} {DIST} {res['version']} route={res['route']}: {res['detail']}")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
