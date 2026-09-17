#!/usr/bin/env python3
"""Refuse unless the installed colabdesign is the pinned commit on the pinned stack (stock/PINS.json).

usage: python -I stock/check_pins.py [--quiet] [--no-stack] [--params-dir DIR] [--no-gpu] [--json]
       python -I stock/check_pins.py --fetch          (run.sh install: the fetch step alone, below; exit 0 = in place as pinned, 1 = not)
       exit 0 = every check passed; 3 = not (one line per finding on stderr)

Checks: (1) the installed `colabdesign` distribution's PEP 610 `direct_url.json` names the pinned commit (a VCS install; a wheel install
without one is compared by version only, and the three lever-touched source files — outside this repo's own git tree, in the installed
package — are hashed against PINS.json `source_sha256` when the package directory can be located without importing it: a missing file is
refused by name, a present-but-patched one is refused by digest); (2) with `--no-stack` absent, the installed distributions in PINS.json
`pins_asserted` equal the pinned stack and the interpreter is the pinned Python release series (major.minor; another series is refused,
another patch release of it is ONE stderr line `python=<v> NOT PINNED (pinned <v>) — this tree was tested with the pinned interpreter only`
and the run proceeds, exit 0 — `python_pin`, the one rule the package's ACTIVE line reads too) (`--no-stack`: the stock route on another
stack is a labelled run, not the pinned stock);
(3) with `--params-dir DIR`, the weights `DIR/params/params_model_{1..5}_multimer_v3.npz`: a MISSING file is refused by name; a present
file is always accepted — digested and worded `pinned` when its sha256 and byte count are PINS.json `weights`' (the line
`weights=<name> sha256=<12> (pinned)`), else `not pinned` with ONE stderr line per file `weights=<name> sha256=<12> NOT PINNED —
this tree was tested with the pinned weights only` and the run proceeds (exit 0; `--json` records digest + word); (4) with
`--no-gpu` absent, the card (nvidia-smi) is REPORTED, never a finding: no GPU visible (`fast` refuses by name at activation there — its
Pallas kernel lowers on the GPU backend only — `off` runs), a compute capability below PINS.json `gpu.cc_min` (the Pallas kernel's
floor: uncertified, the levers engage) or a card other than the tested one (`gpu` name, memory within 1 %) is ONE `NOTE` line on stderr and
the check proceeds; (5) the upstream source archives PINS.json `upstream.<name>.archive` (ColabDesign's — the offline install route, STOCK.md —
and BindCraft's) are REPORTED, never a finding: one `upstream archive <path>: included` line each, or `not included: …` in a tree cut
without them, naming how that upstream is installed there (the online route `upstream.colabdesign.install`, the same commit; the vendored
BindCraft tree, run as shipped); (6) the files this tree does not carry but `run.sh install` fetches (PINS.json
`upstream.<name>.fetched.files`: BindCraft's prebuilt DSSP executable, from the BindCraft repository at the pinned commit) are REPORTED —
`fetched file <path>: sha256=<12> (pinned)` — and a file absent or off its pin is ONE `NOTE` line on stderr, never a finding here: the
fetch step is their gate. `--fetch` is that step and nothing else: every such file that is absent is downloaded to a temporary file beside
its target and renamed into place only when its sha256 and byte count are the pin's (then given its pinned mode), else deleted and REFUSED
by name; a file already present is checked, never re-fetched — pinned: kept (its mode set), anything else: REFUSED by name and left in
place; exit 0 = every file in place as pinned, 1 = a refusal or a download that failed (named). Standard library only, nothing upstream is
imported: run.sh, the configs and the package all call this one file.
"""
import hashlib
import importlib.metadata as md
import json
import os
import subprocess
import sys
from urllib.request import urlopen

HERE = os.path.dirname(os.path.abspath(__file__))
DIST = "colabdesign"
LEVER_FILES = ("colabdesign/af/prep.py", "colabdesign/af/model.py", "colabdesign/af/alphafold/model/modules.py")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for part in iter(lambda: fh.read(chunk), b""):
            h.update(part)
    return h.hexdigest()


def installed(dist=DIST):
    """(version, direct_url dict or {}, package directory or None) of the installed distribution, without importing it."""
    try:
        d = md.distribution(dist)
    except md.PackageNotFoundError:
        return None, {}, None
    raw = d.read_text("direct_url.json")
    info = json.loads(raw) if raw else {}
    pkg_dir = None
    for f in d.files or []:
        if str(f).replace(os.sep, "/") == "colabdesign/__init__.py":
            pkg_dir = os.path.dirname(os.path.dirname(str(d.locate_file(f))))
            break
    return d.version, info, pkg_dir


def check_commit(pins):
    up = pins["upstream"][DIST]
    ver, du, pkg_dir = installed()
    detail = {"installed_version": ver, "direct_url": du, "package_dir": pkg_dir, "pinned_commit": up["commit"], "files": {}}
    bad = []
    if ver is None:
        return [f"{DIST}: not installed (PINS.json upstream.{DIST}.install)"], detail
    commit = (du.get("vcs_info") or {}).get("commit_id")
    detail["installed_commit"] = commit
    if commit:
        if commit != up["commit"]:
            bad.append(f"{DIST}: installed from commit {commit[:12]}, the pin is {up['commit'][:12]}")
    elif ver != up["version"]:
        bad.append(f"{DIST}: {ver} installed without a VCS record; the pin is {up['version']} @ {up['commit'][:12]}")
    if pkg_dir:
        for rel in LEVER_FILES:
            p = os.path.join(pkg_dir, rel)
            want = (pins.get("source_sha256") or {}).get("src/" + rel)
            if not os.path.isfile(p):
                detail["files"][rel] = "missing"
                bad.append(f"{DIST}: {rel} absent from the installed package at {pkg_dir}")
            elif want:
                got = sha256_file(p)
                detail["files"][rel] = "ok" if got == want else "differs"
                if got != want:
                    bad.append(f"{DIST}: {rel} differs from the pinned commit ({got[:12]} != {want[:12]}): a patched install is not stock")
            else:
                detail["files"][rel] = "ok"                      # present, no pin to check against
    return bad, detail


def base_version(v):
    return v.split("+", 1)[0] if isinstance(v, str) else v


def check_stack(pins):
    bad, detail = [], {}
    for name in pins.get("pins_asserted") or []:
        want = pins["pins"].get(name)
        if name == DIST:
            continue                                              # the commit check above is the colabdesign pin
        try:
            have = md.version(name)
        except md.PackageNotFoundError:
            have = None
        ok = base_version(have) == base_version(want)
        detail[name] = {"installed": have, "pinned": want, "ok": ok}
        if not ok:
            bad.append(f"stack: {name} {have} != pinned {want} (PINS.json pins)")
    d = python_pin(pins); detail["python"] = d
    if d["status"] == "refused":
        bad.append(f"stack: python {d['installed']} != pinned {d['pinned']} (another release series; another patch release of {_series(d['pinned'])} is named, not refused)")
    return bad, detail


PY_NOT_PINNED_NOTE = "this tree was tested with the pinned interpreter only"


def _series(v):
    return ".".join(str(v).split(".")[:2])


def python_pin(pins):
    """The interpreter against PINS.json `python` — the ONE rule (run.sh install through check_stack; the package's ACTIVE line through stack.stack_drift):
    {"installed", "pinned", "status", "ok"} with status `pinned` (equal), `not_pinned` (another patch release of the pinned major.minor series: named
    on ONE line — python_line — never a finding), `refused` (another series: a finding), `unpinned` (PINS.json names no interpreter)."""
    py, want = ".".join(str(x) for x in sys.version_info[:3]), pins.get("python")
    if not want:
        status = "unpinned"
    elif py == str(want):
        status = "pinned"
    elif _series(py) == _series(want):
        status = "not_pinned"
    else:
        status = "refused"
    return {"installed": py, "pinned": want, "status": status, "ok": status != "refused"}


def python_line(d):
    """The warn-and-run line of an interpreter at another patch release of the pinned series (python_pin status `not_pinned`), else None."""
    if (d or {}).get("status") != "not_pinned":
        return None
    return f"python={d['installed']} NOT PINNED (pinned {d['pinned']}) — {PY_NOT_PINNED_NOTE}"


NOT_PINNED_NOTE = "this tree was tested with the pinned weights only"
WEIGHTS_CACHE = os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"), "colabdesign_opt", "weights_sha256.json")   # the design start's digest memo (sha256_cached)


def sha256_cached(path, cache_path):
    """sha256 of `path`, memoized in the JSON file `cache_path` keyed by (abspath, size, mtime_ns): an unchanged file is not re-hashed at the next design
    start; a change of size or mtime re-hashes. Never size-only — a cache miss, an unreadable or an unwritable cache digests the file."""
    st = os.stat(path); key = f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}"
    try:
        with open(cache_path, encoding="utf-8") as fh:
            cache = json.load(fh)
    except (OSError, ValueError):
        cache = {}
    cache = cache if isinstance(cache, dict) else {}
    if isinstance(cache.get(key), str) and len(cache[key]) == 64:
        return cache[key]
    digest = sha256_file(path); cache[key] = digest
    try:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        tmp = f"{cache_path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(cache, fh)
        os.replace(tmp, cache_path)
    except OSError:
        pass
    return digest


def check_weights(pins, params_dir, cache=None):
    """Warn-and-run: every PRESENT weights file is accepted — digested (sha256, always: AF2 params share one byte count, so size says nothing), `pinned`
    when sha256 and byte count equal PINS.json `weights`, else `not_pinned` (recorded, never a finding); a MISSING file is the one refusal (by name).
    `cache` = a sha256_cached memo path (the design start: unchanged files are not re-hashed); None = digest now (the `check` route, authoritative)."""
    bad, detail = [], {}
    for name, w in (pins.get("weights") or {}).get("files", {}).items():
        p = os.path.join(params_dir, "params", name)
        if not os.path.isfile(p):
            detail[name] = "missing"; bad.append(f"weights: {p} missing"); continue
        size = os.path.getsize(p)
        got = sha256_cached(p, cache) if cache else sha256_file(p)
        pinned = got == w["sha256"] and (not w.get("bytes") or size == w["bytes"])
        detail[name] = {"sha256": got, "bytes": size, "pinned": pinned, "status": "pinned" if pinned else "not_pinned",
                        "pinned_sha256": w["sha256"], "pinned_bytes": w.get("bytes"), "ok": True}
    return bad, detail


def weights_lines(detail):
    """(pinned lines, not-pinned + absent lines) for the weights census: `weights=<name> sha256=<12> (pinned)` /
    `weights=<name> sha256=<12> NOT PINNED — this tree was tested with the pinned weights only` / `weights=<name> ABSENT — pinned params file not found`."""
    cert, unc = [], []
    for name, d in (detail or {}).items():
        if d == "missing":                                                       # a pinned params file absent from the params root: named on its own line (the design refuses it by name — bindcraft.refusals)
            unc.append(f"weights={name} ABSENT — pinned params file not found"); continue
        if not isinstance(d, dict):
            continue
        if d["pinned"]:
            cert.append(f"weights={name} sha256={d['sha256'][:12]} (pinned)")
        else:
            unc.append(f"weights={name} sha256={d['sha256'][:12]} NOT PINNED — {NOT_PINNED_NOTE}")
    return cert, unc


def nvidia_smi():
    """{name, memory_mib, cc} of GPU 0 (cc None when this nvidia-smi has no `compute_cap` field), or {} without a visible GPU."""
    for fields in ("name,memory.total,compute_cap", "name,memory.total"):
        try:
            r = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            return {}
        if r.returncode == 0 and r.stdout.strip():
            parts = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
            return {"name": parts[0], "memory_mib": int(float(parts[1])), "cc": (parts[2] or None) if len(parts) > 2 else None}
    return {}


def _cc(v):
    """'8.0' -> (8, 0); None for an absent / unparsable value."""
    try:
        return tuple(int(x) for x in str(v).split(".")) if v not in (None, "") else None
    except ValueError:
        return None


def check_gpu(pins, tolerance=0.01):
    """Never a finding — the card is named, not refused: no visible GPU (a mode whose lever set holds the Pallas kernel refuses by name at
    activation; `off` runs), and a card that is not the tested one (name, memory within `tolerance`) or below `gpu.cc_min` — the
    Pallas kernel's compute-capability floor — (uncertified: the levers engage) are notes (detail["notes"], printed as `check_pins: NOTE ...`).
    Returns ([], detail)."""
    want = pins.get("gpu") or {}
    got = nvidia_smi()
    detail = {"expected": want, "got": got, "notes": []}
    if not got:
        detail["notes"].append("gpu: no GPU visible (nvidia-smi) — `fast` refuses by name here (its Pallas kernel lowers on the GPU backend only); `off` runs: proceeding")
        return [], detail
    cc, cc_min = _cc(got.get("cc")), _cc(want.get("cc_min"))
    differs = []
    if want.get("name") and got["name"] != want["name"]:
        differs.append(f"card {got['name']!r} != {want['name']!r}")
    if want.get("memory_mib") and abs(got["memory_mib"] - want["memory_mib"]) > tolerance * want["memory_mib"]:
        differs.append(f"memory {got['memory_mib']} MiB outside {tolerance:.0%} of {want['memory_mib']} MiB")
    if cc_min is not None and cc is not None and cc < cc_min:
        differs.append(f"compute capability {got['cc']} < {want['cc_min']}, the Pallas kernel's floor")
    if differs:
        cap = ("" if (cc is not None and cc_min is not None and cc < cc_min)
               else (f"; compute capability {got['cc']} >= {want['cc_min']}, the Pallas kernel's floor" if cc is not None and cc_min is not None
                     else f"; compute capability not reported by nvidia-smi, the Pallas kernel's floor {want.get('cc_min')} unchecked"))
        detail["notes"].append(f"gpu: {'; '.join(differs)} — not the tested card{cap}: proceeding")
    return [], detail


def upstream_archives(pins, home=None):
    """The upstream source archives (PINS.json upstream.<name>.archive, relative to the kit directory) — REPORTED, never a finding:
    {name: {"path", "included", "install"}}. A tree cut without them installs stock ColabDesign by the online route only (upstream.colabdesign.install,
    the pinned commit from its repository) and runs the vendored BindCraft tree as shipped (stock/src/bindcraft)."""
    home = os.path.dirname(HERE) if home is None else home
    out = {}
    for name, up in (pins.get("upstream") or {}).items():
        rel = up.get("archive")
        if rel:
            out[name] = {"path": rel, "included": os.path.isfile(os.path.join(home, rel)), "install": up.get("install")}
    return out


def upstream_archive_lines(archives):
    """One stdout line per archive: `included`, or `not included: <how that upstream is installed here>` — the online pip route for a distribution,
    the vendored tree for a tree that runs from stock/src."""
    lines = []
    for name, d in archives.items():
        if d["included"]:
            lines.append(f"upstream archive {d['path']}: included")
        elif str(d.get("install") or "").startswith("vendored"):
            lines.append(f"upstream archive {d['path']}: not included: the vendored tree {str(d['install']).split()[1]} runs as shipped")
        else:
            lines.append(f"upstream archive {d['path']}: not included: online install route only, pip install --no-deps \"{d.get('install')}\"")
    return lines


FETCH_NOTE = "fetched by `run.sh install` from the upstream repository at the pinned commit (STOCK.md 'Pin')"


def fetched_files(pins):
    """[(key, record)] over PINS.json `upstream.<name>.fetched.files`: key = the file's path under stock/ (`src/bindcraft/functions/dssp`),
    record = {"url", "sha256", "bytes", "mode"} — files this tree does not carry; `run.sh install` fetches them (fetch, below)."""
    out = []
    for up in (pins.get("upstream") or {}).values():
        for key, rec in ((up.get("fetched") or {}).get("files") or {}).items():
            out.append((key, rec))
    return out


def check_fetched(pins, home=None):
    """REPORTED, never a finding: every install-fetched file against its pin — {key: {"path", "status": pinned | absent | not_pinned, "sha256",
    "bytes", "executable"}}. `home` = the kit directory (default: the one holding this stock/)."""
    stock = HERE if home is None else os.path.join(home, "stock")
    detail = {}
    for key, rec in fetched_files(pins):
        p = os.path.join(stock, key)
        if not os.path.isfile(p):
            detail[key] = {"path": p, "status": "absent"}; continue
        got, size = sha256_file(p), os.path.getsize(p)
        ok = got == rec["sha256"] and (not rec.get("bytes") or size == rec["bytes"])
        detail[key] = {"path": p, "status": "pinned" if ok else "not_pinned", "sha256": got, "bytes": size, "executable": os.access(p, os.X_OK),
                       "pinned_sha256": rec["sha256"], "pinned_bytes": rec.get("bytes")}
    return [], detail


def fetched_lines(detail):
    """(pinned lines, note lines): `fetched file <path>: sha256=<12> (pinned)` / `<path> absent — fetched by ...` / `<path> sha256=<12> is not the pinned <12> ...`."""
    ok, notes = [], []
    for key, d in (detail or {}).items():
        rel = "stock/" + key
        if d["status"] == "pinned":
            ok.append(f"fetched file {rel}: sha256={d['sha256'][:12]} (pinned){'' if d.get('executable') else ', NOT executable (run.sh install sets its mode)'}")
        elif d["status"] == "absent":
            notes.append(f"{rel} absent — {FETCH_NOTE}; BindCraft's design step cannot run its DSSP check without it")
        else:
            notes.append(f"{rel} sha256={d['sha256'][:12]} is not the pinned {d['pinned_sha256'][:12]} — not the file this tree was tested with ({FETCH_NOTE})")
    return ok, notes


def fetch(pins, home=None, opener=urlopen, out=sys.stdout):
    """`run.sh install`'s fetch step (`check_pins.py --fetch`): every install-fetched file (fetched_files) that is absent is downloaded from its pinned
    URL into a temporary file beside its target and accepted only when its sha256 and byte count are the pin's — then given its pinned mode and
    renamed into place; otherwise the temporary file is deleted and the file REFUSED by name. A file already present is checked, never re-fetched:
    pinned -> kept (its mode set when it lacks it), anything else -> REFUSED by name and left in place for inspection. `opener(url)` returns a
    readable binary stream (urllib's urlopen: https and file URLs). Returns the exit code: 0 = every file in place as pinned; 1 = a refusal or a
    download that failed (each named on its own line)."""
    stock = HERE if home is None else os.path.join(home, "stock")
    tag = f"[{pins.get('model', 'kit')}-opt install]"
    files = fetched_files(pins)
    bad = 0
    for key, rec in files:
        dest = os.path.join(stock, key); rel = "stock/" + key; mode = int(str(rec.get("mode", "755")), 8)
        if os.path.isfile(dest):
            got, size = sha256_file(dest), os.path.getsize(dest)
            if got == rec["sha256"] and (not rec.get("bytes") or size == rec["bytes"]):
                if os.stat(dest).st_mode & 0o777 != mode:
                    os.chmod(dest, mode)
                print(f"{tag} present  {rel}: sha256={got[:12]} (pinned; kept, mode {rec.get('mode', '755')})", file=out)
            else:
                bad += 1
                print(f"{tag} REFUSED {rel}: sha256 {got[:16]} ({size} bytes) is not the pinned {rec['sha256'][:16]} ({rec.get('bytes')} bytes) — left in place; "
                      f"not the file this tree was tested with (remove it and run the install again to fetch the pinned one from {rec['url']})", file=out)
            continue
        tmp = f"{dest}.part"
        print(f"{tag} fetching {rel} from {rec['url']}", file=out)
        try:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with opener(rec["url"]) as resp, open(tmp, "wb") as fh:
                for part in iter(lambda: resp.read(1 << 20), b""):
                    fh.write(part)
        except (OSError, ValueError) as e:                                 # urllib's URLError / HTTPError are OSErrors; an unwritable stock/ is one too
            bad += 1
            try:
                os.unlink(tmp)
            except OSError:
                pass
            print(f"{tag} FAILED  {rel}: the download did not complete ({type(e).__name__}: {e}); nothing was placed. By hand: fetch {rec['url']} "
                  f"(sha256 {rec['sha256']}), save it as {dest}, chmod {rec.get('mode', '755')}", file=out)
            continue
        got, size = sha256_file(tmp), os.path.getsize(tmp)
        if got != rec["sha256"] or (rec.get("bytes") and size != rec["bytes"]):
            bad += 1
            os.unlink(tmp)
            print(f"{tag} REFUSED {rel}: the bytes served at {rec['url']} have sha256 {got[:16]} ({size} bytes), not the pinned {rec['sha256'][:16]} "
                  f"({rec.get('bytes')} bytes) — discarded, nothing was placed", file=out)
            continue
        os.chmod(tmp, mode)
        os.replace(tmp, dest)
        print(f"{tag} placed   {rel}: sha256={got[:12]} (pinned), mode {rec.get('mode', '755')}", file=out)
    if bad:
        print(f"{tag} FETCH REFUSED: {bad}/{len(files)} file(s) fetched at install are not in place as pinned (stock/PINS.json upstream.*.fetched; the lines above)", file=out)
        return 1
    if files:
        print(f"{tag} FETCH OK: {len(files)}/{len(files)} file(s) fetched at install are in place as pinned (sha256 = stock/PINS.json upstream.*.fetched)", file=out)
    return 0


def check(pins, *, stack=True, params_dir=None, gpu=True):
    bad, detail = check_commit(pins)
    detail = {"commit": detail, "upstream_archives": upstream_archives(pins), "fetched": check_fetched(pins)[1]}
    if stack:
        b, d = check_stack(pins); bad += b; detail["stack"] = d
    if params_dir:
        b, d = check_weights(pins, params_dir); bad += b; detail["weights"] = d
    if gpu:
        b, d = check_gpu(pins); bad += b; detail["gpu"] = d
    return bad, detail


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    quiet, stack, gpu, as_json = "--quiet" in argv, "--no-stack" not in argv, "--no-gpu" not in argv, "--json" in argv
    params_dir = argv[argv.index("--params-dir") + 1] if "--params-dir" in argv else None
    pins = json.load(open(os.path.join(HERE, "PINS.json"), encoding="utf-8"))
    if "--fetch" in argv:                                                 # run.sh install's fetch step: the install-fetched files and nothing else
        sys.exit(fetch(pins))
    bad, detail = check(pins, stack=stack, params_dir=params_dir, gpu=gpu)
    if as_json:
        print(json.dumps({"ok": not bad, "findings": bad, "detail": detail}, indent=1, default=str))
    elif not quiet:
        c = detail["commit"]
        if c.get("installed_version"):
            print(f"{DIST} {c['installed_version']} @ {c['installed_commit'][:12] if c.get('installed_commit') else 'no VCS record'}: pin {c['pinned_commit'][:12]}"
                  f"{' (' + ', '.join(f'{k} {v}' for k, v in c['files'].items()) + ')' if c['files'] else ''}")
        for line in upstream_archive_lines(detail["upstream_archives"]):
            print(line)
        for line in fetched_lines(detail.get("fetched"))[0]:
            print(line)
        for name, d in (detail.get("stack") or {}).items():
            if d["ok"] and d.get("status", "pinned") == "pinned":
                print(f"{name} {d['installed']}: pinned stack")
        for line in weights_lines(detail.get("weights"))[0]:
            print(line)
        if detail.get("gpu", {}).get("got"):
            g = detail["gpu"]["got"]; pinned = not any(x.startswith("gpu:") for x in bad) and not detail["gpu"].get("notes")
            print(f"gpu {g['name']} {g['memory_mib']} MiB: {'the pinned card' if pinned else 'not the pinned card'}")
    for line in weights_lines(detail.get("weights"))[1]:                 # warn-and-run: a weights file that is not the pinned one is ONE line, never a finding (quiet or not)
        print(f"check_pins: {line}", file=sys.stderr)
    line = python_line((detail.get("stack") or {}).get("python"))        # warn-and-run: an interpreter at another patch release of the pinned series is ONE line, never a finding (quiet or not)
    if line:
        print(f"check_pins: {line}", file=sys.stderr)
    for line in (detail.get("gpu") or {}).get("notes") or []:           # note-and-run: a card other than the tested one is ONE line, never a finding (quiet or not)
        print(f"check_pins: NOTE {line}", file=sys.stderr)
    for line in fetched_lines(detail.get("fetched"))[1]:                 # note-and-run: an install-fetched file absent or off its pin is ONE line, never a finding here (run.sh install --fetch is its gate)
        print(f"check_pins: NOTE {line}", file=sys.stderr)
    if bad:
        for b in bad:
            print(f"check_pins: {b}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
