"""stock/: PINS.json, the archives and their unpacked copies agree; check_pins accepts the tree's own sources by content, and a package the
tree carries no copy of (pxdbench) by version and commit."""
import hashlib
import json
import os
import re
import tarfile

import pytest


def _sha(b):
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def pins(tree):
    return json.load(open(os.path.join(tree, "stock", "PINS.json")))


CARRIED = ("pxdesign", "protenix")            # pins with a reference archive and its extraction under stock/src/
FETCHED = ("pxdbench",)                        # pinned by version and commit; the tree carries no copy of it


def test_src_equals_archives(tree, pins):
    up = pins["upstream"]
    assert tuple(n for n in up if "archive" in up[n]) == CARRIED and tuple(n for n in up if "archive" not in up[n]) == FETCHED
    for name in FETCHED:
        assert "src" not in up[name] and up[name]["fetched"] and up[name]["commit"] in open(os.path.join(tree, "environment", "requirements.lock"), encoding="utf-8").read()
    stock = os.path.join(tree, "stock")
    assert sorted(f for f in os.listdir(stock) if f.endswith(".tar.gz")) == sorted(os.path.basename(up[n]["archive"]) for n in CARRIED)   # no archive but the two named
    assert sorted(os.listdir(os.path.join(stock, "src"))) == sorted(os.path.basename(up[n]["src"]) for n in CARRIED)                     # no extraction but the two named
    for name in CARRIED:
        pin = up[name]
        with tarfile.open(os.path.join(tree, pin["archive"])) as t:
            members = [m for m in t.getmembers() if m.isfile()]
            assert members
            in_archive = set()
            for m in members:
                rel = m.name.split("/", 1)[1]
                in_archive.add(rel)
                p = os.path.join(tree, pin["src"], rel)
                assert os.path.isfile(p), p
                assert _sha(t.extractfile(m).read()) == _sha(open(p, "rb").read()), rel
        src = os.path.join(tree, pin["src"])
        in_src = {os.path.relpath(os.path.join(dp, f), src) for dp, dn, fs in os.walk(src) if "__pycache__" not in dp.split(os.sep) for f in fs}
        assert in_src == in_archive, sorted(in_src ^ in_archive)                       # the extraction is the archive, file for file: nothing added, nothing left over (byte-code caches aside)
        for rel in pin.get("omitted", []):                                              # files the pin declares left out (STOCK.md Pin) are in neither
            assert rel not in in_archive and not os.path.exists(os.path.join(src, rel)), rel
        pkg_dir = os.path.join(tree, pin["src"], pin["package"])
        assert os.path.isdir(pkg_dir)
    assert up["pxdesign"]["omitted"] == ["pxdesign/pxd_server/Helvetica-Regular.ttf", "pxdesign/pxd_server/TimesNewRoman.ttf"] and "omitted" not in up["protenix"]


def test_pins_commits_and_versions(tree, pins):
    up = pins["upstream"]
    assert up["pxdesign"]["commit"] == "f788441313c84c3074fe9596ac2433f96b15c763" and up["pxdesign"]["version"] == "0.1.0"
    assert up["protenix"]["commit"] == "d18aa1daadd02a001b32bc7fa2278fc8a2f8f025" and up["protenix"]["version"] == "0.5.0+pxd"
    assert up["pxdbench"]["version"] == "0.1.2" and up["pxdbench"]["tag"] == "v0.1.2"
    assert pins["pinned_stack"]["status"] == "pinned" and pins["pinned_stack"]["pinned"]["torch"] == "2.3.1+cu121"
    raw = open(os.path.join(tree, "stock", "PINS.json"), encoding="utf-8").read()
    assert raw.count("im-") == 0 and isinstance(pins["pinned_stack"]["tested_on"], str)        # no container-image id anywhere in the pin card; tested_on is a free-text record nothing compares (check_pins reports status + pinned versions only)
    for f, sha in pins["weights"]["required_in_dir"]["files"].items():            # the companion checkpoints' pinned digests
        assert re.fullmatch(r"[0-9a-f]{64}", sha), f
    for f, sha in pins["ccd_cache"]["files"].items():                            # the pinned CCD digests
        assert re.fullmatch(r"[0-9a-f]{64}", sha), f
    assert pins["variants"]["checkpoint"] == "pxdesign_v0.1.0"


def test_check_pins_content_route(tree, pins, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("cp", os.path.join(tree, "stock", "check_pins.py"))
    cp = importlib.util.module_from_spec(spec); spec.loader.exec_module(cp)
    # an "installed" package whose files are the tree's own copy passes by content
    monkeypatch.setattr(cp, "installed_root", lambda pkg: os.path.join(tree, pins["upstream"][pkg]["src"], pkg))
    for pkg in CARRIED:
        ok, summ = cp.content_check(pkg, os.path.join(tree, pins["upstream"][pkg]["src"]))
        assert ok and summ["n_differ"] == 0 and summ["files_compared"] > 10, (pkg, summ)
    # a modified copy is refused
    monkeypatch.setattr(cp, "installed_root", lambda pkg: os.path.join(tree, "opt", "forward", "fast_inference", "stock", "PXDesign", "pxdesign"))
    ok, summ = cp.content_check("pxdesign", os.path.join(tree, pins["upstream"]["pxdesign"]["src"]))
    assert not ok and summ["n_missing"] > 0


def _fake_dist(version, commit=None, editable=False):
    class D:
        pass
    d = D(); d.version = version
    info = {"url": "https://example.invalid/x.git", "vcs_info": {"commit_id": commit, "vcs": "git"}} if commit else ({"url": "file:///x", "dir_info": {"editable": True}} if editable else {})
    d.read_text = lambda name: json.dumps(info) if name == "direct_url.json" else None
    return d


def _cp(tree, tag):
    import importlib.util
    spec = importlib.util.spec_from_file_location(tag, os.path.join(tree, "stock", "check_pins.py"))
    cp = importlib.util.module_from_spec(spec); spec.loader.exec_module(cp)
    return cp


def test_check_pins_content_is_checked_on_every_route(tree, pins, monkeypatch, tmp_path):
    """The recorded commit is provenance only: an install at the pinned commit with an edited file is refused; missing files are
    tolerated on that route alone (setup.py may not ship them), never a differing or an extra file; an editable checkout at no commit
    passes by content and is refused when a file is missing."""
    import shutil
    cp = _cp(tree, "cp3")
    pin = pins["upstream"]["pxdesign"]
    one = {"pxdesign": pin}
    src = os.path.join(tree, pin["src"], "pxdesign")
    monkeypatch.setattr(cp.md, "distribution", lambda pkg: _fake_dist("0.1.0", commit=pin["commit"]))
    # (a) pristine copy, recorded at the pinned commit: pinned by content, the commit reported
    monkeypatch.setattr(cp, "installed_root", lambda pkg: src)
    bad, detail = cp.check(one, tree=tree)
    assert not bad and detail["pxdesign"]["pinned"] and detail["pxdesign"]["by"] == "content (commit recorded)" and detail["pxdesign"]["content"]["n_differ"] == 0
    # (b) the same commit recorded, one file edited after the install: refused
    edited = tmp_path / "edited" / "pxdesign"; shutil.copytree(src, edited)
    with open(edited / "runner" / "inference.py", "a", encoding="utf-8") as fh:
        fh.write("\n# edited after install\n")
    monkeypatch.setattr(cp, "installed_root", lambda pkg: str(edited))
    bad, detail = cp.check(one, tree=tree)
    assert bad and not detail["pxdesign"]["pinned"] and "source differs" in bad[0] and detail["pxdesign"]["content"]["n_differ"] == 1
    # (c) the same commit recorded, one file the package does not ship absent: tolerated on this route
    thin = tmp_path / "thin" / "pxdesign"; shutil.copytree(src, thin); os.remove(thin / "runner" / "dumper.py")
    monkeypatch.setattr(cp, "installed_root", lambda pkg: str(thin))
    bad, detail = cp.check(one, tree=tree)
    assert not bad and detail["pxdesign"]["pinned"] and detail["pxdesign"]["content"]["n_missing"] == 1 and "missing_tolerated" in detail["pxdesign"]["content"]
    # (d) the same commit recorded, an extra source file: refused
    extra = tmp_path / "extra" / "pxdesign"; shutil.copytree(src, extra); (extra / "runner" / "added.py").write_text("x = 1\n")
    monkeypatch.setattr(cp, "installed_root", lambda pkg: str(extra))
    bad, detail = cp.check(one, tree=tree)
    assert bad and detail["pxdesign"]["content"]["n_extra"] == 1
    # (e) an editable checkout with no commit recorded: by content; a missing file refuses on this route
    monkeypatch.setattr(cp.md, "distribution", lambda pkg: _fake_dist("0.1.0", editable=True))
    monkeypatch.setattr(cp, "installed_root", lambda pkg: src)
    bad, detail = cp.check(one, tree=tree)
    assert not bad and detail["pxdesign"]["by"] == "content" and detail["pxdesign"]["commit"] is None
    monkeypatch.setattr(cp, "installed_root", lambda pkg: str(thin))
    bad, detail = cp.check(one, tree=tree)
    assert bad and not detail["pxdesign"]["pinned"]
    # (f) the pinned commit recorded but another version: refused
    monkeypatch.setattr(cp.md, "distribution", lambda pkg: _fake_dist("0.1.1", commit=pin["commit"]))
    monkeypatch.setattr(cp, "installed_root", lambda pkg: src)
    bad, detail = cp.check(one, tree=tree)
    assert bad and "version 0.1.1 != 0.1.0" in bad[0]


def test_check_pins_refuses_missing_package(tree, pins):
    import importlib.util
    spec = importlib.util.spec_from_file_location("cp2", os.path.join(tree, "stock", "check_pins.py"))
    cp = importlib.util.module_from_spec(spec); spec.loader.exec_module(cp)
    bad, detail = cp.check({"nosuchpkg_x": dict(pins["upstream"]["pxdesign"], package="nosuchpkg_x")}, tree=tree)
    assert bad and detail["nosuchpkg_x"]["pinned"] is False
    st = cp.stack_report(pins)
    assert st["status"] == "pinned" and st["pinned_id"] == "pxd-cu121"


def test_check_pins_fetched_package_by_version_and_commit(tree, pins, monkeypatch, tmp_path):
    """pxdbench has no copy in the tree: pinned when the installed version is the pin's AND the commit on record — pip's direct_url.json, or the
    HEAD of an editable checkout's .git — is the pinned one; anything else, or no record at all, refuses."""
    cp = _cp(tree, "cp4")
    pin = pins["upstream"]["pxdbench"]
    one = {"pxdbench": pin}
    pkg = tmp_path / "checkout" / "pxdbench"; pkg.mkdir(parents=True); (pkg / "__init__.py").write_text("")
    monkeypatch.setattr(cp, "installed_root", lambda name: str(pkg))
    # (a) a regular git install recorded at the pinned commit
    monkeypatch.setattr(cp.md, "distribution", lambda name: _fake_dist("0.1.2", commit=pin["commit"]))
    bad, detail = cp.check(one, tree=tree)
    assert not bad and detail["pxdbench"]["pinned"] and detail["pxdbench"]["by"] == "version+commit" and detail["pxdbench"]["content"] is None
    # (b) another commit recorded
    monkeypatch.setattr(cp.md, "distribution", lambda name: _fake_dist("0.1.2", commit="0" * 40))
    bad, detail = cp.check(one, tree=tree)
    assert bad and "commit " in bad[0] and not detail["pxdbench"]["pinned"]
    # (c) an editable checkout: the commit is read from its .git — detached HEAD, then a packed ref, then a loose ref
    monkeypatch.setattr(cp.md, "distribution", lambda name: _fake_dist("0.1.2", editable=True))
    bad, detail = cp.check(one, tree=tree)
    assert bad and "no commit on record" in bad[0]                                     # no .git yet
    git = tmp_path / "checkout" / ".git"; git.mkdir()
    (git / "HEAD").write_text(pin["commit"] + "\n")
    bad, detail = cp.check(one, tree=tree)
    assert not bad and detail["pxdbench"]["commit"] == pin["commit"] and "editable checkout" in detail["pxdbench"]["source"]
    (git / "HEAD").write_text("ref: refs/heads/main\n"); (git / "packed-refs").write_text("# pack-refs with: peeled fully-peeled sorted \n%s refs/heads/main\n" % pin["commit"])
    bad, detail = cp.check(one, tree=tree)
    assert not bad and detail["pxdbench"]["pinned"]
    (git / "refs" / "heads").mkdir(parents=True); (git / "refs" / "heads" / "main").write_text("1" * 40 + "\n")
    bad, detail = cp.check(one, tree=tree)
    assert bad and "commit " + "1" * 40 in bad[0]                                        # the loose ref wins over packed-refs, as git reads them
    # (d) the pinned commit but another version
    monkeypatch.setattr(cp.md, "distribution", lambda name: _fake_dist("0.1.3", commit=pin["commit"]))
    bad, detail = cp.check(one, tree=tree)
    assert bad and "version 0.1.3 != 0.1.2" in bad[0]
