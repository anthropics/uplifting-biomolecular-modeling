"""The carried lever kit is present as its own directory at its expected file count, its warm-up inputs are in place, and the files NOT
carried (tables, manifests, history documents, drivers) are accounted for by name."""
import json
import os

from pxdesign_opt import registry, warm
from pxdesign_opt.registry import list_kit_files as kit_files

KITS = {"forward/hoist": registry.KIT_RELPATH}
N_CARRIED = {"forward/hoist": 12}


def test_every_carried_kit_is_present_at_its_expected_size(tree):
    """Presence + count, not bytes: git (or the sdist/wheel build) already guarantees the checked-out bytes match the commit."""
    opt = os.path.join(tree, "opt")
    assert registry.CARRIED_KITS == (registry.KIT_RELPATH,)
    for kit, relpath in KITS.items():
        present = set(kit_files(os.path.join(tree, relpath)))
        assert len(present) == N_CARRIED[kit], (kit, len(present))
    assert not os.path.exists(os.path.join(opt, "forward", "SHA256SUMS"))                # no stray hash list anywhere under opt/
    assert sorted(os.listdir(os.path.join(opt, "forward"))) == ["hoist"]                  # the lever kit is the one carried tree


def test_warm_inputs_are_vendored(tree):
    """warm's input set: the three tasks and the public structures they name, beside the lever (opt/forward/hoist/inputs/); SOURCES.json
    lists exactly the vendored files."""
    kit = os.path.join(tree, registry.KIT_RELPATH)
    tasks = json.load(open(os.path.join(kit, warm.TASKS_RELPATH), encoding="utf-8"))
    named = sorted({os.path.basename(x["condition"]["structure_file"]) for x in tasks})
    targets = os.path.join(kit, warm.TARGETS_RELPATH)
    assert named == ["1tnf.cif", "3di3.cif", "5o45.cif"]
    for name in named:
        assert os.path.isfile(os.path.join(targets, name)), name
    sources = json.load(open(os.path.join(targets, "SOURCES.json"), encoding="utf-8"))
    listed = sorted(e["file"] for e in (sources["files"] if isinstance(sources, dict) else sources))
    assert listed == named


def test_tables_and_drivers_not_carried(tree):
    fwd = os.path.join(tree, "opt", "forward")
    assert not os.path.exists(os.path.join(fwd, "hoist", "tables"))
    assert not os.path.exists(os.path.join(fwd, "fast_inference"))                          # no second kit under opt/forward
    for f in ("CHANGELOG_v1.2.2.md", "CHANGELOG_v1.2.2.diff", "UPSTREAM_PR.md", "MANIFEST.json", "tests", "tables"):   # the lever kit's history documents, manifests and script/table directories
        assert not os.path.exists(os.path.join(fwd, "hoist", f)), f
    assert not os.path.exists(os.path.join(tree, "opt", "serving"))                       # no serving add-on is carried (registry.CARRIED_KITS)
