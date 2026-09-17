#!/usr/bin/env python3
"""Refuse unless the install matches stock/PINS.json: the upstream cookbook file, the esm checkout, the transformers fork, the weights.

usage: python -I stock/check_pins.py [--quiet] [--json] [--no-weights]     exit 0 = every pin holds; 3 = not (one line per failure on stderr)
       [--checks software,weights] selects the groups that run (default: both): `software` = the esm checkout + cookbook file, the
       transformers fork, the two pinned wheels; `weights` = the six snapshots under HF_HOME. `run.sh install` runs `--checks software`
       (the weights arrive by its --weights DIR step and are located by HF_HOME at `check` / `design`).

Reads package metadata and files only (no torch import): the esm distribution must be the editable checkout whose cookbook file hashes to
the tree's own vendored copy (stock/src/cookbook/tutorials/binder_design.py — PEP 610 direct_url.json dir_info.editable locates the
candidate, or the file at EF2INV_COOKBOOK_STOCK); the transformers distribution
must carry the fork commit (direct_url.json vcs_info.commit_id) AND its installed modeling files must hash to
``transformers_fork.installed_files_sha256`` — a code mismatch is a hard refusal. The pinned stack's two wheels (flash-attn, transformer_engine)
are warn-and-run (``dist_pins_check``: another version, or none, is one ``stack <key>=<got> (pinned <want>) NOT PINNED — …`` line and the run
proceeds on what is bound). Weights are warn-and-run (``weights_gate``): an unknown
checkpoint is NAMED and runs, never refused — the pinned snapshot when present, else the revision ``refs/main`` names, else the newest
snapshot of the repo under HF_HOME (--hf-home; no default: unset is a refusal by name) is digested (sha256 per file unless --no-weights, which reads presence and size only)
and reported in one line per snapshot, ``weights=<repo>@<commit> sha256=<12> (pinned)`` when it is the pinned revision with every file at
its pin, else ``weights=<repo>@<commit> sha256=<12> NOT PINNED — the speed and exactness figures apply to the pinned weights only`` (another
revision, or any differing file; exit code unchanged); the ONE refusal is weights genuinely missing — no snapshot of the repo at all, or a
pinned file absent from the snapshot that runs (the loop cannot load it). The sha pin is a note, not a gate; the design's
launch digests through ``--weights-cache PATH`` (``sha256_cached``: a (path, size, mtime_ns) memo — an unchanged file is hashed once per box,
never identified by size alone), ``check`` digests afresh;
the JSON report records the digest and the word per snapshot (``facts.weights_census``). Standard library only, so run.sh, the package and
the proof box run it on any python.
"""
import argparse
import hashlib
import importlib.metadata as md
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NOT_PINNED_NOTE = "NOT PINNED — the speed and exactness figures apply to the pinned weights only"


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def direct_url(name):
    try:
        d = md.distribution(name)
        return d, json.loads(d.read_text("direct_url.json") or "null")
    except Exception:
        return None, None


WEIGHTS_CACHE = os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"), "ef2inv_opt", "weights_sha256.json")   # the design launch's digest memo (sha256_cached)


def sha256_cached(path, cache_path):
    """sha256 of `path`, memoized in the JSON file `cache_path` keyed by (abspath, size, mtime_ns): an unchanged file is not re-hashed at the next
    launch; a change of size or mtime re-hashes. Never size-only — a cache miss, an unreadable or an unwritable cache digests the file."""
    st = os.stat(path); key = f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}"
    try:
        with open(cache_path, encoding="utf-8") as fh:
            cache = json.load(fh)
    except (OSError, ValueError):
        cache = {}
    cache = cache if isinstance(cache, dict) else {}
    if isinstance(cache.get(key), str) and len(cache[key]) == 64:
        return cache[key]
    digest = sha256(path); cache[key] = digest
    try:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        tmp = f"{cache_path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(cache, fh)
        os.replace(tmp, cache_path)
    except OSError:
        pass
    return digest


def repo_dir(hf, repo):
    return os.path.join(hf, "hub", "models--" + repo.replace("/", "--"))


def snapshot_dir(hf, repo, rec):
    return os.path.join(repo_dir(hf, repo), "snapshots", rec["snapshot_commit"])


def snapshot_found(hf, repo, rec):
    """(dir, commit) of the snapshot that runs: the PINNED revision when present; else the revision ``refs/main`` names, else the newest
    snapshot directory of the repo — an unknown checkpoint runs, worded unpinned (revision != pin); (None, None) when the repo holds no
    snapshot at all under ``hf`` (weights missing: the one refusal)."""
    d = snapshot_dir(hf, repo, rec)
    if os.path.isdir(d):
        return d, rec["snapshot_commit"]
    snaps = os.path.join(repo_dir(hf, repo), "snapshots"); ref = os.path.join(repo_dir(hf, repo), "refs", "main")
    if os.path.isfile(ref):
        c = open(ref, encoding="utf-8").read().strip()
        if c and os.path.isdir(os.path.join(snaps, c)):
            return os.path.join(snaps, c), c
    if os.path.isdir(snaps):
        cs = sorted((c for c in os.listdir(snaps) if os.path.isdir(os.path.join(snaps, c))), key=lambda c: os.path.getmtime(os.path.join(snaps, c)), reverse=True)
        if cs:
            return os.path.join(snaps, cs[0]), cs[0]
    return None, None


def weights_gate(weights_pins, hf, hash_files=True, cache=None):
    """Warn-and-run weights check. For every pinned repo the snapshot that runs (snapshot_found: the pinned revision, else refs/main's,
    else the newest) is always accepted and digested; another revision is worded unpinned (`revision <found> != pin <pinned>`), never
    refused; MISSING — no snapshot of the repo at all, or a pinned file absent from the snapshot that runs — is the one refusal by name;
    `cache` = a sha256_cached memo path (the design's launch), None = digest afresh (`check`);
    the file set is digested — sha256 over the sorted ``<sha256-or-size>  <rel>`` lines of its files (sha256 per
    file when ``hash_files``, else the sizes) — with the word ``pinned`` when every file is at its pin (size, and sha256 when hashed) or
    ``unpinned`` when any present file differs. Returns (fails, status, census, lines): ``status[repo]`` the one-line summary the check
    report shows, ``census[repo]`` {word, sha256, checked, snapshot_commit, files, ok, differs, missing}, ``lines`` the printed lines."""
    fails, status, census, lines = [], {}, {}, []
    for repo, rec in weights_pins.items():
        d, found = snapshot_found(hf, repo, rec); n = len(rec["files"]); at = f"{repo}@{(found or rec['snapshot_commit'])[:10]}"
        if d is None:
            fails.append(f"weights {repo}: snapshot {rec['snapshot_commit']} missing under {hf}")
            status[repo] = "missing"; census[repo] = {"word": "missing", "sha256": None, "checked": "sha256" if hash_files else "size", "snapshot_commit": rec["snapshot_commit"],
                                                       "files": n, "ok": 0, "differs": [], "missing": ["<snapshot dir>"]}
            continue
        missing, differs, manifest, n_ok = [], [], [], 0
        if found != rec["snapshot_commit"]:                                             # an unknown checkpoint (another HF revision): it runs, named unpinned — never refused
            differs.append(f"revision {found[:10]} != pin {rec['snapshot_commit'][:10]}")
        for rel, fr in sorted(rec["files"].items()):
            p = os.path.join(d, rel)
            if not os.path.isfile(p):
                missing.append(rel); fails.append(f"weights {repo}/{rel}: missing"); continue
            size = os.path.getsize(p)
            got = (sha256_cached(p, cache) if cache else sha256(p)) if hash_files else None
            manifest.append(f"{got if hash_files else size}  {rel}")
            if size != fr["size_bytes"]:
                differs.append(f"{rel}: size {size} != {fr['size_bytes']}"); continue
            if hash_files and got != fr["sha256"]:
                differs.append(f"{rel}: sha256 {got[:12]} != {fr['sha256'][:12]}"); continue
            n_ok += 1
        digest = hashlib.sha256(("\n".join(manifest) + "\n").encode()).hexdigest() if manifest else None
        checked = "sha256" if hash_files else "size"
        if missing:
            word = "missing"
        elif differs:
            word = "unpinned"
        else:
            word = "pinned"
        census[repo] = {"word": word, "sha256": digest, "checked": checked, "snapshot_commit": rec["snapshot_commit"], "files": n, "ok": n_ok, "differs": differs, "missing": missing}
        tag = f"sha256={digest[:12]}" if hash_files else f"sizes={digest[:12] if digest else '-'} (sha256 not computed: --no-weights)"
        if word == "missing":
            status[repo] = f"missing {len(missing)}/{n} files: {', '.join(missing)}"
            lines.append(f"weights={at} MISSING {', '.join(missing)} — refused")
        elif word == "unpinned":
            status[repo] = f"{n_ok}/{n} files ok NOT PINNED {tag} ({'; '.join(differs)})"
            lines.append(f"weights={at} {tag} {NOT_PINNED_NOTE} ({len(differs)} file(s) differ: {'; '.join(differs)})")
        else:
            status[repo] = f"{n_ok}/{n} files ok (pinned) {tag}" + ("" if hash_files else " (size only)")
            lines.append(f"weights={at} {tag} (pinned{'' if hash_files else ' by presence and size'})")
    return fails, status, census, lines


def dist_version(name):
    """The installed version of a distribution (importlib.metadata), None when absent."""
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


DIST_PINS = {"flash_attn": ("flash-attn", "flash_attn"), "transformer_engine": ("transformer_engine", "transformer-engine")}   # pinned_stack key -> distribution names


STACK_NOT_PINNED_NOTE = "NOT PINNED — the speed figures apply to the pinned stack only"


def dist_pins_check(pins, facts, version_of=None):
    """``pinned_stack.flash_attn`` / ``.transformer_engine`` vs the installed distributions (``version_of``: injection for the CPU
    tests) — the two sha256-pinned wheels of the pinned stack (``pinned_stack.image_recipe``). Warn-and-run, as the weights: returns the
    note lines (empty = met; a key absent from the pins is not checked), ``stack <key>=<got> (pinned <want>) NOT PINNED — …``, recorded as
    ``facts['stack_lines']`` by ``check()`` and never a fail — another version, or none, is named and the run proceeds on what is bound;
    records ``facts['<key>_version']``."""
    sor = pins.get("pinned_stack") or {}
    notes = []
    for key, names in DIST_PINS.items():
        want = sor.get(key)
        got = None
        for n in names:
            got = (version_of or dist_version)(n)
            if got:
                break
        facts[f"{key}_version"] = got
        if want and got != want:
            notes.append(f"stack {key}={got} (pinned {want}) {STACK_NOT_PINNED_NOTE} "
                         f"(the sha256-pinned {key} wheel of stock/PINS.json pinned_stack.image_recipe; STOCK.md §Pinned stack)")
    return notes


CHECK_GROUPS = ("software", "weights")   # --checks: the groups of pins a run checks (default: all); `software` alone is the install step's check
CHECKS = CHECK_GROUPS                      # the groups check() runs; main() narrows it from --checks


def check(pins, weights=True, hf_home=None, cookbook_path=None, cache=None):
    fails, facts = [], {}
    # esm: the editable checkout and the cookbook file
    dist, du = direct_url("esm")
    facts["esm_version"] = dist.version if dist else None
    facts["esm_direct_url"] = du
    if dist is None:
        fails.append("esm: not installed")
    cb = cookbook_path or os.environ.get("EF2INV_COOKBOOK_STOCK")
    if not cb and du and du.get("url", "").startswith("file://"):
        cb = os.path.join(du["url"][len("file://"):], "cookbook", "tutorials", "binder_design.py")
    facts["cookbook_path"] = cb
    ref = os.path.join(HERE, "src", "cookbook", "tutorials", "binder_design.py")   # the tree's own vendored copy: the git commit names the bytes; no stored digest
    if not cb or not os.path.isfile(cb):
        fails.append(f"upstream file not found ({cb!r}): set EF2INV_COOKBOOK_STOCK")
    else:
        got = sha256(cb); facts["cookbook_sha256"] = got
        want = sha256(ref)
        if got != want:
            fails.append(f"upstream file {cb} sha256 {got} != tree copy {ref} sha256 {want}")
    # transformers: the fork commit + the installed modeling files
    tdist, tdu = direct_url("transformers")
    facts["transformers_version"] = tdist.version if tdist else None
    want_commit = pins["transformers_fork"]["commit"]
    got_commit = (tdu or {}).get("vcs_info", {}).get("commit_id") if tdu else None
    facts["transformers_commit"] = got_commit
    if tdist is None:
        fails.append("transformers: not installed")
    elif got_commit != want_commit:
        fails.append(f"transformers: installed commit {got_commit} != {want_commit} (direct_url.json)")
    if tdist is not None:
        try:
            import transformers  # metadata only: locating the files
            root = os.path.dirname(os.path.dirname(transformers.__file__))
            for rel, want in pins["transformers_fork"]["installed_files_sha256"].items():
                p = os.path.join(root, rel.split("/", 1)[1]) if rel.startswith("src/") else os.path.join(root, rel)
                if not os.path.isfile(p):
                    fails.append(f"transformers file missing: {rel}")
                elif sha256(p) != want:
                    fails.append(f"transformers file differs from the pin: {rel}")
        except Exception as e:
            fails.append(f"transformers files: {e!r}")
    # flash-attn + transformer_engine: the two sha256-pinned wheels of the pinned stack (pinned_stack / image_recipe) — a box without
    # them, or with other versions, is NAMED (facts['stack_lines'], one NOT PINNED line each) and runs on what is bound, never refused
    facts["stack_lines"] = dist_pins_check(pins, facts)
    if "weights" not in CHECKS:   # --checks software (run.sh install): the software pins only; the weights are not looked for
        facts["hf_home"], facts["weights"], facts["weights_census"], facts["weights_lines"] = None, {}, {}, []
        return fails, facts
    # weights: warn-and-run (weights_gate) — missing is the one refusal; the weights root is the operator's HF_HOME (README.md §Weights), no default
    hf = hf_home or os.environ.get("HF_HOME")
    facts["hf_home"] = hf
    if not hf:
        fails.append("HF_HOME is not set — see README.md §Variables (the Hugging Face cache root that holds the six pinned biohub/* snapshots, README.md §Weights; the weights are located by it alone)")
        facts["weights"], facts["weights_census"], facts["weights_lines"] = {}, {}, []
        return fails, facts
    wfails, facts["weights"], facts["weights_census"], facts["weights_lines"] = weights_gate(pins["weights"], hf, hash_files=weights, cache=cache)
    fails.extend(wfails)
    return fails, facts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true"); ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-weights", action="store_true", help="presence and size of the weights only (no hashing of the large files)")
    ap.add_argument("--checks", default=",".join(CHECK_GROUPS), help="comma-separated groups to check: software (the esm checkout + cookbook file, the transformers fork, the two pinned wheels), weights (the snapshots under HF_HOME); default both")
    ap.add_argument("--pins", default=os.path.join(HERE, "PINS.json")); ap.add_argument("--hf-home"); ap.add_argument("--cookbook")
    ap.add_argument("--weights-cache", metavar="PATH", help="digest the weights through this (path, size, mtime_ns) sha256 memo (the design's launch: an unchanged file is hashed once per box)")
    a = ap.parse_args(argv)
    pins = json.load(open(a.pins))
    groups = [g.strip() for g in a.checks.split(",") if g.strip()]
    if not groups or [g for g in groups if g not in CHECK_GROUPS]:
        print(f"check_pins: --checks {a.checks!r}: the groups are {', '.join(CHECK_GROUPS)}", file=sys.stderr); return 2
    global CHECKS
    CHECKS = tuple(g for g in CHECK_GROUPS if g in groups)
    fails, facts = check(pins, weights=not a.no_weights, hf_home=a.hf_home, cookbook_path=a.cookbook, **({"cache": a.weights_cache} if a.weights_cache else {}))
    if a.json:
        print(json.dumps({"ok": not fails, "fails": fails, "facts": facts}, indent=1))
    elif not a.quiet:
        for line in facts.get("stack_lines", []) + facts.get("weights_lines", []):
            print("check_pins: " + line, file=sys.stderr)
        for f in fails:
            print("check_pins: " + f, file=sys.stderr)
        print("check_pins: " + ("OK" if not fails else f"FAIL ({len(fails)})"), file=sys.stderr)
    else:
        for line in facts.get("stack_lines", []) + facts.get("weights_lines", []):
            if NOT_PINNED_NOTE in line:
                print("check_pins: " + line, file=sys.stderr)              # --quiet still names unpinned weights: the one line that runs travel with
    return 0 if not fails else 3


if __name__ == "__main__":
    sys.exit(main())
