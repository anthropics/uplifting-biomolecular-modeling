"""The gpn install probe (stack.gpn_install / gpn_gate): a VCS record at the pin, or modules byte-identical to the carried stock archive, both
stand for the pinned commit (gpn=<commit8>, no uncertainty word); installed files that differ from the archive are another stock (refused);
only an install with neither a commit record nor an archive to compare is NAMED gpn_commit=unrecorded(...). Fake distributions in a
subprocess (a dist-info + package on sys.path, a tiny synthetic archive) — no network, no real gpn."""
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import textwrap

from gpnstar_opt import report, stack
from gpnstar_opt.tests._paths import PY, env_clean

PIN = "6f28c81bcbfe7d65cb6d8ece9ce88f87ca583791"
FILES = {"gpn/__init__.py": b"# stock\n", "gpn/star/__init__.py": b"", "gpn/star/model.py": b"X = 1\n"}


def _site(tmp, direct_url, files=FILES):
    site = os.path.join(tmp, "site")
    for rel, data in files.items():
        p = os.path.join(site, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").write(data)
    di = os.path.join(site, "gpn-0.9.0.dist-info")
    os.makedirs(di, exist_ok=True)
    open(os.path.join(di, "METADATA"), "w").write("Metadata-Version: 2.1\nName: gpn\nVersion: 0.9.0\n")
    if direct_url is not None:
        open(os.path.join(di, "direct_url.json"), "w").write(json.dumps(direct_url))
    return site


def _archive(tmp, files=FILES):
    path = os.path.join(tmp, "gpn-6f28c81b.tar.gz")
    with tarfile.open(path, "w:gz") as t:
        for rel, data in list(files.items()) + [("README.md", b"readme")]:
            name = "gpn-6f28c81b/" + ("src/" + rel if rel.startswith("gpn/") else rel)
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            t.addfile(ti, io.BytesIO(data))
    return path


def _pins(archive_path=None, sha=None):
    return {"upstream": {"gpn": {"commit": PIN, "version": "0.9.0", "archive": "stock/gpn-6f28c81b.tar.gz", "package_dir": "src/gpn"}},
            "archive": {"sha256": sha if sha is not None else (hashlib.sha256(open(archive_path, "rb").read()).hexdigest() if archive_path else None)}, "pins": {}}


def _probe(site, pins, archive_path):
    code = textwrap.dedent(f"""
        import json, sys
        sys.path.insert(0, {site!r})
        from gpnstar_opt import stack
        g = stack.gpn_install(pins=json.loads({json.dumps(json.dumps(pins))}), archive_path={archive_path!r})
        print(json.dumps(g, default=str))
    """)
    p = subprocess.run([PY, "-c", code], capture_output=True, text=True, env=env_clean(), timeout=120)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout.strip().splitlines()[-1])


def test_a_vcs_record_at_the_pin_is_the_commit(tmp_path):
    arch = _archive(str(tmp_path))
    site = _site(str(tmp_path), {"url": "https://github.com/songlab-cal/gpn", "vcs_info": {"vcs": "git", "commit_id": PIN, "requested_revision": PIN}})
    g = _probe(site, _pins(arch), arch)
    assert g["version"] == "0.9.0" and g["commit"] == PIN and g["route"] == "vcs"
    refusals, words = stack.gpn_gate(g, _pins(arch), PIN)
    assert refusals == [] and words == []
    rep = {"word": "exact", "levers": ["devconst"], "gpn_commit8": g["commit"][:8], "words": words}
    assert " gpn=6f28c81b " in report.dry_run_line(rep) and "unrecorded" not in report.dry_run_line(rep)


def test_an_archive_install_byte_identical_to_the_stock_archive_is_the_commit(tmp_path):
    arch = _archive(str(tmp_path))
    site = _site(str(tmp_path), {"url": "file:///w/model-opt-release/gpnstar/stock/gpn-6f28c81b.tar.gz", "archive_info": {"hash": "sha256=00"}})
    g = _probe(site, _pins(arch), arch)
    assert g["commit"] == PIN and g["route"] == "archive", g
    assert g["archive"]["ok"] is True and g["archive"]["n_files"] == len(FILES)
    refusals, words = stack.gpn_gate(g, _pins(arch), PIN)
    assert refusals == [] and words == [], (refusals, words)                      # no gpn_commit=unrecorded word
    rep = {"word": "exact", "levers": ["devconst"], "gpn_commit8": g["commit"][:8], "words": words}
    assert " gpn=6f28c81b " in report.active_line(rep) and "unrecorded" not in report.active_line(rep)
    # the same holds with no direct_url.json at all (a plain local install of the archive's tree)
    g2 = _probe(_site(str(tmp_path / "b"), None), _pins(arch), arch)
    assert g2["commit"] == PIN and g2["route"] == "archive"


def test_an_install_whose_files_differ_from_the_archive_is_refused(tmp_path):
    arch = _archive(str(tmp_path))
    files = dict(FILES, **{"gpn/star/model.py": b"X = 2  # edited\n"})
    site = _site(str(tmp_path), {"url": "file:///w/gpnstar/stock/gpn-6f28c81b.tar.gz", "archive_info": {}}, files=files)
    g = _probe(site, _pins(arch), arch)
    assert g["commit"] is None and g["route"] == "differs" and "gpn/star/model.py" in g["detail"], g
    refusals, words = stack.gpn_gate(g, _pins(arch), PIN)
    assert len(refusals) == 1 and "differs from the stock archive" in refusals[0] and words == []


def test_nothing_to_compare_is_named_not_refused(tmp_path):
    arch = _archive(str(tmp_path))
    site = _site(str(tmp_path), {"url": "file:///elsewhere/gpn.tar.gz", "archive_info": {}})
    g = _probe(site, _pins(arch), str(tmp_path / "absent.tar.gz"))                 # the carried archive is not there to compare
    assert g["commit"] is None and g["route"].startswith("archive_url:unshown:archive_absent"), g
    refusals, words = stack.gpn_gate(g, _pins(arch), PIN)
    assert refusals == [] and len(words) == 1 and words[0].startswith("gpn_commit=unrecorded(")
    g = _probe(site, _pins(arch, sha="0" * 64), arch)                                # the archive is there but off PINS archive.sha256
    assert g["commit"] is None and "unshown:archive_sha256" in g["route"]
    vcs_off = _probe(_site(str(tmp_path / "c"), {"url": "https://x", "vcs_info": {"vcs": "git", "commit_id": "1" * 40}}), _pins(arch), arch)
    refusals, _ = stack.gpn_gate(vcs_off, _pins(arch), PIN)
    assert len(refusals) == 1 and "the levers are written against" in refusals[0]
