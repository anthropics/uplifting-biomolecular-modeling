#!/usr/bin/env python
"""Check that the installed pxdesign / protenix / pxdbench are the pins of stock/PINS.json — standard library only.

A package whose pin names a tree copy ("src": pxdesign, protenix) is pinned when its version equals the pin AND its installed source equals that copy under stock/src/<Repo>/<package>/
file for file (every .py/.cu/.cpp/.h/.hpp/.cuh, compared on every route — an editable checkout, a wheel, an archive or a git install
alike; sub-second). A differing or an extra file refuses on every route. A file of the tree's copy that is absent from the install is
tolerated only when pip's direct_url.json records the pinned git commit (a non-editable install ships what the package's setup.py
lists: Protenix's setup.py names `model/layer_norm/kernel/*` as package data and nothing under `openfold_local/utils/kernel/csrc/`);
on any other route a missing file refuses. The recorded commit is reported as provenance, never as the pass by itself. Anything
else — another commit, a modified checkout, a missing package — is refused (exit 3) with the pin printed. A package whose pin names no tree
copy (pxdbench: PXDesignBench is fetched from its repository at install and nothing of it is carried here) is pinned by version AND commit: the
commit pip recorded in direct_url.json, or, for an editable checkout, the commit its .git HEAD points at; no record of either is a refusal. The running stack (python,
torch, CUDA) is reported against pinned_stack.pinned (the pinned stack's versions); a different stack is a line in the report,
not a refusal, unless --strict-stack is given.

    python stock/check_pins.py [--quiet] [--json] [--strict-stack]     (rc 0 pinned | 3 not pinned)
Also importable: check(pins) -> (bad, detail) and stack_report(pins) -> dict, used by opt/pxdesign_opt/stack.py (one implementation).
"""
import hashlib
import importlib.metadata as md
import importlib.util
import json
import os
import platform
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
SOURCE_SUFFIXES = (".py", ".cu", ".cpp", ".h", ".hpp", ".cuh")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_files(root):
    out = {}
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in ("__pycache__", ".git")]
        for f in fn:
            if f.endswith(SOURCE_SUFFIXES):
                p = os.path.join(dp, f)
                out[os.path.relpath(p, root)] = p
    return out


def installed_root(package):
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.origin:
        return None
    return os.path.dirname(spec.origin)


def checkout_commit(package_root):
    """The commit an editable checkout stands at, read from its .git (HEAD, then the loose ref or packed-refs) without running git — None when
    the package directory is not inside a checkout or the files are unreadable."""
    d = package_root
    for _ in range(4):
        if d is None:
            return None
        g = os.path.join(d, ".git")
        if os.path.isfile(g):                                                         # a worktree/submodule pointer: "gitdir: <path>"
            line = open(g, encoding="utf-8", errors="replace").read().strip()
            g = os.path.normpath(os.path.join(d, line.split(":", 1)[1].strip())) if line.startswith("gitdir:") else None
        if g and os.path.isdir(g):
            try:
                head = open(os.path.join(g, "HEAD"), encoding="utf-8").read().strip()
                if head.startswith("ref:"):
                    ref = head.split(":", 1)[1].strip()
                    loose = os.path.join(g, ref)
                    if os.path.isfile(loose):
                        head = open(loose, encoding="utf-8").read().strip()
                    else:
                        head = ""
                        packed = os.path.join(g, "packed-refs")
                        for ln in (open(packed, encoding="utf-8") if os.path.isfile(packed) else []):
                            parts = ln.split()
                            if len(parts) == 2 and parts[1] == ref:
                                head = parts[0]
                return head.lower() if len(head) == 40 and all(c in "0123456789abcdef" for c in head.lower()) else None
            except OSError:
                return None
        parent = os.path.dirname(d)
        d = parent if parent != d else None
    return None


def content_check(package, src_dir):
    """Compare the installed package directory with stock/src/<Repo>/<package>/: returns (ok, summary)."""
    inst = installed_root(package)
    ref = os.path.join(src_dir, package)
    if inst is None:
        return False, {"installed_root": None, "reason": "not importable"}
    if not os.path.isdir(ref):
        return False, {"installed_root": inst, "reason": f"tree copy missing: {ref}"}
    a, b = source_files(inst), source_files(ref)
    missing = sorted(set(b) - set(a))
    extra = sorted(set(a) - set(b))
    differ = sorted(k for k in set(a) & set(b) if sha256(a[k]) != sha256(b[k]))
    ok = not (missing or extra or differ)
    return ok, {"installed_root": inst, "files_compared": len(set(a) & set(b)), "missing": missing[:20], "extra": extra[:20], "differ": differ[:20],
                "n_missing": len(missing), "n_extra": len(extra), "n_differ": len(differ)}


def check(pins, tree=TREE):
    bad, detail = [], {}
    for name, pin in pins.items():
        pkg = pin.get("package", name)
        try:
            dist = md.distribution(pkg)
        except md.PackageNotFoundError:
            bad.append(f"{pkg}: not installed (want {pin['repo']} @ {pin['commit']}; {pin['install']})")
            detail[pkg] = {"version": None, "commit": None, "pinned": False, "source": "not installed"}
            continue
        raw = dist.read_text("direct_url.json")
        info = json.loads(raw) if raw else {}
        commit = (info.get("vcs_info") or {}).get("commit_id")
        url = info.get("url")
        if not pin.get("src"):                                                        # fetched from the repository at install, no tree copy: version AND commit
            editable = bool((info.get("dir_info") or {}).get("editable"))
            head = checkout_commit(installed_root(pkg)) if not commit else None
            found = commit or head
            version_ok = dist.version == pin["version"]
            pinned = bool(version_ok and found and found == pin["commit"])
            how = (f"git commit {commit}" if commit else ((f"editable checkout {url} at commit {head}" if head else f"editable checkout {url} (no .git to read a commit from)") if editable else (url or "a non-git install")))
            detail[pkg] = {"version": dist.version, "commit": found, "pinned": pinned, "source": how, "by": "version+commit" if pinned else "none", "content": None}
            if not pinned:
                why = []
                if not version_ok:
                    why.append(f"version {dist.version} != {pin['version']}")
                if not found:
                    why.append("no commit on record (neither pip's direct_url.json nor a .git in the checkout)")
                elif found != pin["commit"]:
                    why.append(f"commit {found} != {pin['commit']}")
                bad.append(f"{pkg} {dist.version}: installed from {how}; {'; '.join(why)}; want {pin['repo']} @ {pin['commit']} ({pin['install']})")
            continue
        by_commit = bool(commit and commit == pin["commit"])                    # provenance: pip's record of the install, never the pass by itself
        by_content, content = content_check(pkg, os.path.join(tree, pin["src"]))
        c = content or {}
        if not by_content and by_commit and c.get("n_missing") and not (c.get("n_extra") or c.get("n_differ")):
            by_content = True                                                     # files setup.py does not ship are absent from a non-editable git install
            c["missing_tolerated"] = "install recorded at the pinned commit: the tree's files absent here are ones the package does not ship"
        version_ok = dist.version == pin["version"]
        pinned = bool(by_content and version_ok)
        how = (f"git commit {commit}" if commit else ("editable checkout " + url if (info.get("dir_info") or {}).get("editable") else (url or "a non-git install")))
        detail[pkg] = {"version": dist.version, "commit": commit, "pinned": pinned, "source": how,
                       "by": ("content" + (" (commit recorded)" if by_commit else "")) if by_content else "none", "content": content}
        if not pinned:
            why = []
            if not version_ok:
                why.append(f"version {dist.version} != {pin['version']}")
            if not by_content:
                why.append(f"source differs from {pin['src']} (missing {c.get('n_missing')}, extra {c.get('n_extra')}, differ {c.get('n_differ')}: {(c.get('differ') or c.get('missing') or c.get('extra') or [c.get('reason')])[:3]})")
            bad.append(f"{pkg} {dist.version}: installed from {how}; {'; '.join(why)}; want {pin['repo']} @ {pin['commit']} ({pin['install']})")
    return bad, detail


def stack_report(pins_all):
    want = pins_all["pinned_stack"]["pinned"]
    got = {"python": platform.python_version(), "torch": None, "cuda": None, "cudnn": None, "triton": None}
    try:
        got["torch"] = md.version("torch")
    except md.PackageNotFoundError:
        pass
    try:
        got["triton"] = md.version("triton")
    except md.PackageNotFoundError:
        pass
    if "torch" in sys.modules:                                        # never import torch here: the CUDA version comes from the wheel tag otherwise
        t = sys.modules["torch"]
        got["cuda"] = getattr(getattr(t, "version", None), "cuda", None)
        try:
            got["cudnn"] = str(t.backends.cudnn.version())
        except Exception:  # noqa: BLE001
            pass
    if got["cuda"] is None and got["torch"] and "+cu" in got["torch"]:
        tag = got["torch"].split("+cu", 1)[1]
        got["cuda"] = f"{tag[:-1]}.{tag[-1]}" if len(tag) >= 3 else tag
    diffs = {k: (got[k], want.get(k)) for k in ("python", "torch", "cuda") if got.get(k) and want.get(k) and got[k] != want[k]}
    return {"status": pins_all["pinned_stack"]["status"], "pinned_id": want.get("id"), "running": got, "pinned": want, "differs": diffs, "equal": not diffs}


def read_pins():
    return json.load(open(os.path.join(HERE, "PINS.json")))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    quiet, as_json, strict_stack = "--quiet" in argv, "--json" in argv, "--strict-stack" in argv
    pins_all = read_pins()
    bad, detail = check(pins_all["upstream"])
    stack = stack_report(pins_all)
    if as_json:
        print(json.dumps({"pinned": not bad, "bad": bad, "detail": detail, "stack": stack}, indent=1, default=str))
    elif not quiet:
        for name, d in detail.items():
            if d["pinned"]:
                print(f"{name} {d['version']}: pinned (by {d['by']}: {d['source']})")
        run = stack["running"]
        line = f"stack: python {run['python']} torch {run['torch']} cuda {run['cuda']} — pinned stack: {stack['status']}; pinned {stack['pinned_id']} (torch {stack['pinned']['torch']}): " + ("equal" if stack["equal"] else f"differs {stack['differs']}")
        print(line)
    if bad:
        for b in bad:
            print(f"check_pins: {b}", file=sys.stderr)
        sys.exit(3)
    if strict_stack and not stack["equal"]:
        print(f"check_pins: --strict-stack: running stack differs from the pinned one: {stack['differs']}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
