"""install.sh --classify / --rebind / apply / --uninstall on stub site-packages trees: the OVERLAY_PRESTATE classes (pristine | current |
previous:<id> | foreign:<files>), the OVERLAY_APPLIED line, the overlay marker, and the rule that a foreign file is never touched by --rebind
(apply --force stays the manual escape)."""
import hashlib
import json
import os
import shutil
import subprocess

import pytest

from .. import stack, tree
from . import _stubs

MARKER_REL = os.path.join("rf3", ".rosettafold3_opt_overlay.json")
ID_RE = r"[0-9a-f]{12}"
NEW_FILES = ("rf3/graph_flags.py",)
pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash")


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _sh(tmp_path, site, *args, kit=None, tag="bin"):
    """install.sh <args> against `site` (a stub `python` with PYTHONPATH=<site> first on PATH, as the installer locates site-packages)."""
    kit = kit or stack.kit_home()
    py = _stubs.make_interpreter(str(tmp_path / tag), site)
    env = dict(os.environ, PATH=os.path.dirname(py) + os.pathsep + os.environ.get("PATH", ""), PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    r = subprocess.run(["bash", os.path.join(kit, "install.sh"), *args], env=env, capture_output=True, text=True, cwd=kit)
    return r.returncode, [l for l in r.stdout.splitlines() if l.strip()], r.stdout + r.stderr


def _classify(tmp_path, site, kit=None):
    rc, lines, both = _sh(tmp_path, site, "--classify", kit=kit)
    assert rc == 0 and len(lines) == 1 and lines[0].startswith("OVERLAY_PRESTATE: "), both
    return lines[0][len("OVERLAY_PRESTATE: "):]


def _newer_kit(tmp_path):
    """A copy of the kit whose patched/rf3/graph_flags.py carries one more tagged line: the tree of a LATER revision relative to what is
    installed (STOCK_SRC is passed so the copy still finds the pinned upstream's bytes)."""
    dst = str(tmp_path / "kit_next" / "rf3_xattempt_addon")
    shutil.copytree(stack.kit_home(), dst)
    with open(os.path.join(dst, "patched", "rf3", "graph_flags.py"), "a") as fh:
        fh.write("\n# [rf3_cudagraph] a later revision of this module\n")
    os.environ["STOCK_SRC"] = _stubs.stock_src()
    return dst


@pytest.fixture(autouse=True)
def _no_stock_src_leak():
    old = os.environ.get("STOCK_SRC")
    yield
    if old is None: os.environ.pop("STOCK_SRC", None)
    else: os.environ["STOCK_SRC"] = old


def test_pristine_rebind_installs_and_marks_then_current_is_a_no_op(tmp_path):
    site = _stubs.make_tree(str(tmp_path / "sp"), "stock")
    assert _classify(tmp_path, site) == "pristine"
    rc, lines, both = _sh(tmp_path, site, "--rebind")
    assert rc == 0 and lines[0] == "OVERLAY_PRESTATE: pristine" and len(lines) == 2, both
    assert lines[1].startswith("OVERLAY_APPLIED: 5 files id=") and len(lines[1].split("id=")[1]) == 12, both
    marker = json.load(open(os.path.join(site, MARKER_REL)))
    assert set(marker) == {"name", "overlay_id", "files"} and marker["name"] == "rosettafold3_opt patched rf3 files"
    assert lines[1].split("id=")[1] == marker["overlay_id"][:12]
    assert marker["files"] == {rel: _sha(os.path.join(stack.kit_home(), "patched", rel)) for rel in tree.RF3_FILES}
    for rel in tree.RF3_FILES:                                   # the sidecars: pristine backups for upstream's files, an absent-mark for the new one
        side = ".hoist_absent" if rel in NEW_FILES else ".hoist_orig"
        assert os.path.isfile(os.path.join(site, rel + side)), rel + side
    assert _classify(tmp_path, site) == "current"
    rc, lines, both = _sh(tmp_path, site, "--rebind")
    assert (rc, lines) == (0, ["OVERLAY_PRESTATE: current"]), both
    assert _sh(tmp_path, site, "--check")[0] == 0


def test_an_earlier_marked_install_is_previous_with_its_id_and_rebinds(tmp_path):
    site = _stubs.make_tree(str(tmp_path / "sp"), "stock")
    assert _sh(tmp_path, site, "--rebind")[0] == 0
    old = json.load(open(os.path.join(site, MARKER_REL)))
    nxt = _newer_kit(tmp_path)
    assert _classify(tmp_path, site, kit=nxt) == f"previous:{old['overlay_id'][:12]}"
    rc, lines, both = _sh(tmp_path, site, "apply", kit=nxt)      # plain apply refuses an earlier install by name (rc 4) and points at --rebind
    assert rc == 4 and "--rebind" in both, both
    rc, lines, both = _sh(tmp_path, site, "--rebind", kit=nxt)
    assert rc == 0 and lines[0] == f"OVERLAY_PRESTATE: previous:{old['overlay_id'][:12]}" and lines[1].startswith("OVERLAY_APPLIED: 5 files id="), both
    new = json.load(open(os.path.join(site, MARKER_REL)))
    assert new["overlay_id"] != old["overlay_id"] and lines[1].split("id=")[1] == new["overlay_id"][:12]
    assert _sha(os.path.join(site, "rf3/graph_flags.py")) == _sha(os.path.join(nxt, "patched", "rf3/graph_flags.py"))
    for rel in tree.RF3_FILES:                                   # the pristine backups survive a re-install: uninstall still leads back to upstream
        if rel not in NEW_FILES:
            assert _sha(os.path.join(site, rel + ".hoist_orig")) == _sha(os.path.join(_stubs.stock_src(), _stubs.PRISTINE[rel])), rel
    assert _classify(tmp_path, site, kit=nxt) == "current"
    assert _classify(tmp_path, site) == f"previous:{new['overlay_id'][:12]}"   # and this tree now sees the newer install as an earlier one of itself


def test_an_earlier_install_without_a_marker_is_previous_unmarked_by_its_sidecars_and_tags(tmp_path):
    site = _stubs.make_tree(str(tmp_path / "sp"), "stock")
    assert _sh(tmp_path, site, "--rebind")[0] == 0
    os.remove(os.path.join(site, MARKER_REL))                    # an install older than the marker
    nxt = _newer_kit(tmp_path)
    assert _classify(tmp_path, site, kit=nxt) == "previous:unmarked"
    rc, lines, both = _sh(tmp_path, site, "--rebind", kit=nxt)
    assert rc == 0 and lines[0] == "OVERLAY_PRESTATE: previous:unmarked" and lines[1].startswith("OVERLAY_APPLIED: 5 files id="), both
    assert _classify(tmp_path, site, kit=nxt) == "current"


def test_a_foreign_file_is_named_and_rebind_touches_nothing(tmp_path):
    site = _stubs.make_tree(str(tmp_path / "sp"), "stock")
    victim = os.path.join(site, "rf3/loss/loss.py")
    open(victim, "w").write("# some other edit of this file\n")
    snap = lambda: {rel: (_sha(os.path.join(site, rel)) if os.path.exists(os.path.join(site, rel)) else None) for rel in tree.RF3_FILES}
    before = snap()
    assert _classify(tmp_path, site) == "foreign:rf3/loss/loss.py"
    rc, lines, both = _sh(tmp_path, site, "--rebind")
    assert rc == 4 and lines[0] == "OVERLAY_PRESTATE: foreign:rf3/loss/loss.py" and not any(l.startswith("OVERLAY_APPLIED") for l in lines), both
    assert snap() == before and not os.path.exists(os.path.join(site, MARKER_REL))
    assert not any(os.path.exists(os.path.join(site, rel + s)) for rel in tree.RF3_FILES for s in (".hoist_orig", ".hoist_absent"))
    assert _sh(tmp_path, site, "apply")[0] == 4                  # plain apply refuses too …
    rc, lines, both = _sh(tmp_path, site, "apply", "--force")     # … --force is the manual escape: backs the foreign file up and installs over it
    assert rc == 0 and open(victim + ".hoist_orig").read() == "# some other edit of this file\n", both
    assert _classify(tmp_path, site) == "current"


def test_a_tagged_file_without_this_installers_sidecar_is_foreign(tmp_path):
    """The tags alone are not identification: an upstream file carrying `[xattempt_hoist]` text but no pristine .hoist_orig beside it is foreign."""
    site = _stubs.make_tree(str(tmp_path / "sp"), "stock")
    victim = os.path.join(site, "rf3/model/RF3_structure.py")
    with open(victim, "a") as fh:
        fh.write("\n# [xattempt_hoist] hand-edited\n")
    assert _classify(tmp_path, site) == "foreign:rf3/model/RF3_structure.py"
    shutil.copyfile(victim, victim + ".hoist_orig")             # a backup that is not upstream's bytes does not identify it either
    assert _classify(tmp_path, site) == "foreign:rf3/model/RF3_structure.py"


def test_a_hand_edit_of_a_marked_install_is_foreign_even_with_its_sidecar_and_tags(tmp_path):
    """Once an apply wrote the marker, a target file is identified by the marker's record only: an edited patched file (pristine backup beside
    it, tags inside it) is foreign — --rebind leaves it alone; the sidecar rule speaks only for installs older than the marker."""
    site = _stubs.make_tree(str(tmp_path / "sp"), "stock")
    assert _sh(tmp_path, site, "--rebind")[0] == 0
    with open(os.path.join(site, "rf3/graph_flags.py"), "a") as fh:
        fh.write("\n# [rf3_cudagraph] edited in place\n")
    assert _classify(tmp_path, site) == "foreign:rf3/graph_flags.py"
    assert _sh(tmp_path, site, "--rebind")[0] == 4
    os.remove(os.path.join(site, MARKER_REL))                    # the same bytes on a marker-less install read as an earlier revision (the sidecar rule)
    assert _classify(tmp_path, site) == "previous:unmarked"


def test_a_mix_of_patched_and_pristine_files_is_previous_partial_and_rebinds(tmp_path):
    site = _stubs.make_tree(str(tmp_path / "sp"), "patched")
    rel = "rf3/loss/loss.py"                                     # one file back to pristine, the rest == patched/: an interrupted apply / uninstall
    shutil.copyfile(os.path.join(_stubs.stock_src(), _stubs.PRISTINE[rel]), os.path.join(site, rel))
    assert _classify(tmp_path, site) == "previous:partial"
    rc, lines, both = _sh(tmp_path, site, "--rebind")
    assert rc == 0 and lines[0] == "OVERLAY_PRESTATE: previous:partial" and lines[1].startswith("OVERLAY_APPLIED: 5 files id="), both
    assert _classify(tmp_path, site) == "current"


def test_uninstall_restores_upstream_and_drops_the_marker(tmp_path):
    site = _stubs.make_tree(str(tmp_path / "sp"), "stock")
    assert _sh(tmp_path, site, "--rebind")[0] == 0
    rc, lines, both = _sh(tmp_path, site, "--uninstall")
    assert rc == 0 and not os.path.exists(os.path.join(site, MARKER_REL)), both
    assert _classify(tmp_path, site) == "pristine"


def test_the_overlay_id_is_a_digest_of_the_patched_bytes_not_a_stored_value(tmp_path):
    """No table in the tree carries the id: it is recomputed from patched/ (a tree with one more byte has another id)."""
    site = _stubs.make_tree(str(tmp_path / "sp"), "stock")
    rc, lines, both = _sh(tmp_path, site, "--rebind")
    ident = lines[1].split("id=")[1]
    lines_ = sorted(f"{_sha(os.path.join(stack.kit_home(), 'patched', rel))}  {rel}" for rel in tree.RF3_FILES)
    assert ident == hashlib.sha256(("\n".join(lines_) + "\n").encode()).hexdigest()[:12], (ident, both)
    nxt = _newer_kit(tmp_path)
    rc, lines, both = _sh(tmp_path, site, "--rebind", kit=nxt)
    assert lines[1].split("id=")[1] != ident, both
