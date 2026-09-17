"""The carried kit bytes: what's under opt/forward is exactly the kit's runnable by file count (total accounting: 50 = 47 af3t +
3 dtk), no bytecode sits inside the carried dirs, and stock/PINS.json's kit_patches declares every patch by presence — a patch's `file`
exists under stock/, its `touches` name files that are actually carried. The kit's xfold copy is the archive's package plus exactly the port (PINS upstream.port) plus exactly the declared patches' touched files —
every other member byte-identical, a direct comparison of the archive's own bytes against the kit's on-disk bytes. The carried bytes are
named by the git commit (git log / git blame on opt/forward/), never by a re-hash against an in-repo manifest or a stored digest."""
import json
import os
import tarfile

import pytest

HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPT = os.path.join(HOME, "opt")
STOCK = os.path.join(HOME, "stock")
PINS = json.load(open(os.path.join(STOCK, "PINS.json"), encoding="utf-8"))
KIT_XFOLD = os.path.join(OPT, "forward", "af3t", "af3_torch", "xfold")
CARRIED = {"forward/af3t": 42, "forward/dtk": 2}          # the kit's runnable: the af3t v3 kit's model, api and kernel stack (+ the MSA-module adapter af3t_msa.py and its af3t_opm kernels; triangle attention, the triangle multiplication and the fused transition are the shared core providers' rows -- no copy, cell table or launch table of them carried here) + the DTK FusedDiT modules (dtk_modules.py; `dtk_kernels` is the shared core's routed module since 0.2.12) and the token aggregation kernel (af3t_token_agg.py)


def _present():
    out = []
    for d, _, fs in os.walk(os.path.join(OPT, "forward")):
        out += [os.path.relpath(os.path.join(d, f), OPT) for f in fs]
    return sorted(out)


def test_total_accounting():
    present = _present()
    assert len(present) == sum(CARRIED.values()) == 44          # forward/af3t: no cell table, no fused-transition copy or launch table (the shared core providers' cells); forward/dtk: dtk_modules.py + af3t_token_agg.py (dtk_kernels.py left the tree at 0.2.12: the core's routed copy is the only one)
    for d, n in CARRIED.items():
        assert sum(1 for rel in present if rel.startswith(d + "/")) == n, d


def test_no_bytecode_or_caches_in_the_carried_dirs():
    stray = [rel for rel in _present() if "__pycache__" in rel or rel.endswith((".pyc", ".pyo"))]
    assert not stray, stray


def test_declared_patches_match_the_tree():
    """stock/PINS.json "kit_patches": the root is a carried directory; each declared patch's diff file exists under stock/ and its `touches`
    name files the carried tree actually has, one declaration per file."""
    K = PINS["kit_patches"]
    rel = K["root"].rstrip("/").split("/", 1)[1]
    assert rel in CARRIED, K["root"]
    present = [p[len(rel) + 1:] for p in _present() if p.startswith(rel + "/")]
    touched = set()
    for patch in K["patches"]:
        assert patch["file"].startswith("patches/") and os.path.isfile(os.path.join(HOME, "stock", patch["file"])), patch["id"]
        assert patch["what"] and patch["applies"] and patch["id"] and patch["reason"], patch["id"]
        for path in patch["touches"]:
            assert path not in touched and path in present, (rel, path)                  # one declaration per file; the file is carried
            touched.add(path)


def _patches():
    """Every declared patch of the carried kit (stock/PINS.json kit_patches) with the carried root under HOME."""
    K = PINS["kit_patches"]
    return [(os.path.join(HOME, K["root"]), p) for p in K["patches"]]


def _patched_xfold_files():
    """{'xfold/<rel>'} of the declared patches' touched paths inside the kit's xfold copy."""
    out = set()
    for root, p in _patches():
        for path in p["touches"]:
            full = os.path.normpath(os.path.join(root, path))
            if full.startswith(KIT_XFOLD + os.sep):
                out.add("xfold/" + os.path.relpath(full, KIT_XFOLD))
    return out


def test_patch_diffs_exist_and_are_declared():
    for _, p in _patches():
        assert p["file"].startswith("patches/") and os.path.isfile(os.path.join(STOCK, p["file"])), p["file"]


def _archive_package():
    with tarfile.open(os.path.join(STOCK, PINS["upstream"]["archive"]["file"]), "r:gz") as t:
        prefix = f"xfold-{PINS['upstream']['commit']}/xfold/"
        return {m.name[len(prefix):]: t.extractfile(m).read() for m in t.getmembers() if m.isfile() and m.name.startswith(prefix)}


def test_kit_xfold_is_the_archive_plus_the_port():
    if not os.path.isfile(os.path.join(STOCK, PINS["upstream"]["archive"]["file"])):     # a tree that arrived without the archive: `run.sh install` fetches it (stock/fetch_upstream.py); skipped BY NAME until then
        pytest.skip(f"stock/{PINS['upstream']['archive']['file']} is not in this tree (run.sh install fetches it: stock/fetch_upstream.py)")
    up = _archive_package()
    kit = {}
    for d, dns, fs in os.walk(KIT_XFOLD):
        dns[:] = [x for x in dns if x != "__pycache__"]
        for f in fs:
            if not f.endswith(".pyc"):
                p = os.path.join(d, f); kit[os.path.relpath(p, KIT_XFOLD)] = open(p, "rb").read()
    changed = sorted(f"xfold/{k}" for k in up if k in kit and up[k] != kit[k])
    added = sorted(f"xfold/{k}" for k in kit if k not in up)
    removed = sorted(f"xfold/{k}" for k in up if k not in kit)
    port = PINS["upstream"]["port"]
    patched = _patched_xfold_files()
    assert not patched & set(port["changed_files"])                                     # a declared patch touches a file the port left pristine
    assert changed == sorted(set(port["changed_files"]) | patched), changed
    assert added == sorted(port["added_files"]), added
    assert removed == [], removed
    same = [k for k in up if k in kit and up[k] == kit[k]]
    assert len(same) == len(up) - len(changed)


def test_hardware_cells_are_the_kit_files():
    """hardware.kernel_cells names the kit's cell tables and their keys as the files carry them — never a cc the files do not key."""
    cells = PINS["hardware"]["kernel_cells"]
    assert set(cells) == set()                                                           # no kit cell table: triangle attention, TriMul and transition cells are the shared core providers' (hardware.triangle_attention)
    for name, c in cells.items():
        tab = json.load(open(os.path.join(HOME, c["file"]), encoding="utf-8"))
        assert sorted(k for k in tab if not k.startswith("_")) == sorted(c["keys"]), name
    assert "opt_core.kernels.triattn" in PINS["hardware"]["triangle_attention"] and not os.path.exists(os.path.join(HOME, "opt/forward/af3t/kernels/af3_arch_cells.json"))
