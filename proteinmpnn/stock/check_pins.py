#!/usr/bin/env python3
"""Refuse when what a pass runs is absent (the checkout, the weights); report the checkout against
its pin (stock/PINS.json) and the weights file a pass loads.

usage: python -I stock/check_pins.py [--variant soluble|vanilla] [--mpnn-dir DIR] [--model-name NAME] [--quiet] [--json]
       exit 0 = the pass can run; 3 = something it runs is absent (one line per finding on stderr); the notes and the weights line on stderr either way
       --checks package,weights runs the named subset of the checks (CHECKS below); `run.sh install` runs package — what is installed —
       and leaves weights to `--weights` and the routes

Variants soluble | vanilla (the two weight sets of one checkout): MPNN_DIR (or --mpnn-dir) is a directory holding protein_mpnn_run.py — absent, it is a finding (exit 3).
Its ``git rev-parse HEAD`` is read beside the pinned commit: a checkout at ANOTHER commit is a finding (exit 3) — another commit changes what
"stock" means, so no route runs it (check out the pin, or let `run.sh install --weights DIR` make the clone); a checkout without .git (git_hash=unknown
in protein_mpnn_run.py's .fa headers) cannot be read and runs as upstream runs it — a note (``notes`` in the --json record, ``check_pins: note: ...``
on stderr), never a refusal; the pinned commit names the tested bytes.
Weights: the one file the pass loads (``weights_file``: ``$MPNN_DIR/<variant>_model_weights/<model_name>.pt``, --model-name defaulting to
PINS.json "variants.<v>.model_name") must
exist — an absent file is a finding like any other — and is digested, never compared for refusal (``check_weights``): a PINS.json digest
prints ``weights=<name> sha256=<12 hex> (pinned)``, any other bytes (a fine-tuned checkpoint, say) print ``weights sha256=<12 hex>
NOT PINNED — proceeding (stock/PINS.json names the tested weights)`` and the exit code is unchanged; the record {file, sha256,
pinned, verdict, line} is "weights" in the --json detail.
Standard library only, so run.sh, the configs and the package all call this one file.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PINS_PATH = os.path.join(HERE, "PINS.json")
BASE_VARIANTS = ("soluble", "vanilla")
# The carried stock/src/ entry points: presence, not a re-hash — the tree's git commit names the bytes.
STOCK_SRC_FILES = ("protein_mpnn_run.py", "protein_mpnn_utils.py", "helper_scripts/parse_multiple_chains.py")
# --checks: the subset of checks a run of this file makes (default: all). package = the stock code (the checkout MPNN_DIR names); weights = the
# weights file the pass loads (implies the checkout checks, the file lives in it). Upstream declares no versions, so there is no stack check.
# SELECTED is narrowed by main() alone; importers (the kit package) get every check.
CHECKS = ("package", "weights")
SELECTED = set(CHECKS)


def load_pins(path=PINS_PATH):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(repo):
    """HEAD of a checkout, or None when the directory is not a git checkout (no .git)."""
    if not os.path.isdir(os.path.join(repo, ".git")):
        return None
    try:
        out = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


# ----------------------------------------------------------------------------------------------------------------- the weights a pass loads
NOT_PINNED = "NOT PINNED — proceeding (stock/PINS.json names the tested weights)"


def pinned_weights(pins):
    """{sha256: name} over every weights entry of PINS.json (the repository weight sets, "weights", named by path)."""
    names = {}
    for w in pins["weights"].values():
        names.setdefault(w["sha256"], w["path"])
    return names


def weights_file(pins, variant, mpnn_dir=None, model_name=None):
    """The one file a pass of ``variant`` loads: ``<MPNN_DIR>/<variant>_model_weights/<model_name>.pt`` — the directory of the
    variant's repository weights (PINS.json "weights.<v>.path") and upstream's --model_name (default PINS.json "variants.<v>.model_name";
    protein_mpnn_run.py:57 appends ``<model_name>.pt`` to --path_to_model_weights), so a second weight file in that directory is selected by its
    name."""
    v = pins["variants"][variant]
    return os.path.join(mpnn_dir or "", os.path.dirname(pins["weights"][v["weights"]]["path"]), f"{model_name or v['model_name']}.pt")


def weights_line(record):
    """The one weights line of a pass: ``weights=<name> sha256=<12 hex> (pinned)`` when the digest is a PINS.json entry, else
    ``weights sha256=<12 hex> NOT PINNED — proceeding (stock/PINS.json names the tested weights)``."""
    s12 = record["sha256"][:12]
    return f"weights={record['pinned']} sha256={s12} (pinned)" if record["pinned"] else f"weights sha256={s12} {NOT_PINNED}"


def check_weights(pins, path):
    """The weights file a pass loads, accepted whatever its bytes: (bad, record). An absent file is the one finding (refused by name, like every
    finding); a present file is digested and named, never refused — record = {file, sha256, pinned: the PINS.json name of that digest | None,
    verdict: pinned | not_pinned, line: ``weights_line``}."""
    if not os.path.isfile(path):
        return [f"{path}: the weights file is absent"], {"file": path, "sha256": None, "pinned": None, "verdict": None, "line": None}
    digest = sha256(path)
    pinned = pinned_weights(pins).get(digest)
    record = {"file": path, "sha256": digest, "pinned": pinned, "verdict": "pinned" if pinned else "not_pinned"}
    record["line"] = weights_line(record)
    return [], record


# ----------------------------------------------------------------------------------------------------------------- the pins
def check_base(pins, variant, mpnn_dir, model_name=None, weights_path=None):
    """The base checkout: MPNN_DIR holds upstream's entry point (absent: a finding); its HEAD beside the pinned commit (another commit: a finding —
    it changes what stock means; unreadable, no .git: a note, never a finding);
    the weights file of the pass (``check_weights`` on ``weights_path`` — the file the pass's own selectors name — else ``weights_file``: present,
    digested, named). Returns (bad, detail); ``detail["notes"]`` the reported facts."""
    bad, detail = [], {"mpnn_dir": mpnn_dir, "variant": variant, "notes": []}
    up = pins["upstream"]
    if not mpnn_dir:
        bad.append(f"MPNN_DIR is not set (README 'Variables'; the clone: {up['install']})")
        return bad, detail
    if not os.path.isdir(mpnn_dir):
        bad.append(f"MPNN_DIR={mpnn_dir!r} is not a directory (the clone: {up['install']})")
        return bad, detail
    if not os.path.isfile(os.path.join(mpnn_dir, "protein_mpnn_run.py")):
        bad.append(f"MPNN_DIR={mpnn_dir!r} does not hold protein_mpnn_run.py (the clone: {up['install']})")
    head = git_head(mpnn_dir)
    detail["head"] = head
    detail["pinned_commit"] = up["commit"]
    if head is None:
        detail["notes"].append(f"{mpnn_dir} is not a git checkout (no .git or git unavailable): {up['git_required']} — the pinned commit is {up['commit']}")
    elif head != up["commit"]:
        bad.append(f"{mpnn_dir} HEAD {head} is not the pinned commit {up['commit']}: another commit changes what stock means, so no route runs it — "
                   f"`git -C {mpnn_dir} checkout {up['commit']}` (or the clone `run.sh install --weights DIR` makes: {up['install']})")
    if "weights" not in SELECTED: return bad, detail
    wbad, detail["weights"] = check_weights(pins, weights_path or weights_file(pins, variant, mpnn_dir=mpnn_dir, model_name=model_name))
    bad += wbad
    return bad, detail


def check_snapshot(root=None):
    """The carried stock/src/ entry points are present at their expected paths — presence, not a re-hash (the tree's git commit names the
    bytes). Returns the list of missing paths, one ``stock/src/<rel>: missing`` finding per absent file."""
    root = root or HERE
    return [f"stock/src/{rel}: missing" for rel in STOCK_SRC_FILES if not os.path.isfile(os.path.join(root, "src", rel))]


def check(pins, variant="soluble", mpnn_dir=None, model_name=None, weights_path=None):
    return check_base(pins, variant, mpnn_dir if mpnn_dir is not None else os.environ.get("MPNN_DIR"), model_name, weights_path=weights_path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--variant", default="soluble", choices=BASE_VARIANTS)
    p.add_argument("--mpnn-dir", default=None, help="the ProteinMPNN checkout (default: $MPNN_DIR)")
    p.add_argument("--model-name", default=None, help="upstream's --model_name, the weights file <name>.pt of the pass (default: PINS.json variants.<v>.model_name)")
    p.add_argument("--weights-file", default=None, help="the weights file the pass loads when its own selectors name one (--path_to_model_weights / --use_soluble_model): digested and named instead of the variant's default")
    p.add_argument("--checks", default=",".join(CHECKS), help=f"the checks to run, comma-separated, of {','.join(CHECKS)} (default: all; run.sh install: package)")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--json", action="store_true", help="print the detail record on stdout")
    a = p.parse_args(argv)
    if not set(a.checks.split(",")) <= set(CHECKS) or not a.checks: p.error(f"--checks {a.checks!r}: names of {','.join(CHECKS)}, comma-separated")
    SELECTED.intersection_update(a.checks.split(","))
    pins = load_pins()
    bad, detail = check(pins, a.variant, a.mpnn_dir, a.model_name, weights_path=a.weights_file)
    notes = list(detail.get("notes") or [])
    if a.json:
        print(json.dumps({"pinned": not bad, "findings": bad, "notes": notes, "detail": detail}, indent=1))
    if not a.quiet:
        for line in notes:
            print(f"check_pins: note: {line}", file=sys.stderr)
    weights = detail.get("weights") or {}
    if weights.get("line") and not a.quiet:
        print(f"check_pins: {weights['line']}", file=sys.stderr)
    if bad:
        if not a.quiet:
            for line in bad:
                print(f"check_pins: {line}", file=sys.stderr)
        return 3
    if not a.quiet and not a.json:
        if SELECTED != set(CHECKS):
            print(f"check_pins: {a.variant}: checks {','.join(c for c in CHECKS if c in SELECTED)} pass")
            return 0
        print(f"check_pins: {a.variant} ready (ProteinMPNN {(detail['head'] or 'unknown')[:8]} at {detail['mpnn_dir']}; pinned commit {pins['upstream']['commit'][:8]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
