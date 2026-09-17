"""Tree states by sha256, computed live: the add-on's patched/ files and the pinned upstream's files under stock/src; every state is named
or refused."""
import os

import pytest

from .. import stack, tree
from . import _stubs


@pytest.fixture
def lists():
    return stack.tree_digests()


def test_the_digests_are_the_checkouts_own_files(lists):
    kit, src = stack.kit_home(), stack.stock_src()
    assert set(lists.patched) == set(tree.RF3_FILES) and len(lists.patched) == 5
    assert lists.patched == {rel: tree.sha256_file(os.path.join(kit, "patched", rel)) for rel in tree.RF3_FILES}
    assert set(lists.stock) == set(tree.RF3_FILES[1:])                                       # graph_flags.py is absent upstream
    assert lists.stock == {rel: tree.sha256_file(os.path.join(src, "models", "rf3", "src", rel)) for rel in tree.RF3_FILES[1:]}
    assert all(lists.stock[rel] != lists.patched[rel] for rel in lists.stock)                 # the add-on changes every file it carries
    assert all(os.path.isfile(p) for p in lists.sources.values())
    with pytest.raises(FileNotFoundError, match="tree reference file missing"):
        tree.digests(kit, str(os.path.dirname(src)))                                          # a directory without the pinned files


def test_classify_names_every_state(tmp_path, lists):
    ts = tree.classify(_stubs.make_tree(str(tmp_path / "stock"), "stock"), lists)
    assert ts.state == "stock" and ts.rf3_count == "stock(5/5)" and ts.files["rf3/graph_flags.py"] == "absent"
    assert ts.line() == f"tree=stock(5/5) site-packages={ts.site_packages}"
    ts = tree.classify(_stubs.make_tree(str(tmp_path / "patched"), "patched"), lists)
    assert ts.state == "patched" and ts.rf3_count == "patched(5/5)"
    ts = tree.classify(_stubs.make_tree(str(tmp_path / "unknown"), "unknown"), lists)
    assert ts.state == "unknown" and ts.unknown == ["rf3/loss/loss.py"]
    with pytest.raises(RuntimeError, match="tree state is 'unknown'"):
        tree.expect(ts, "stock", "patched")
    # a patched tree with the graph_flags file removed is neither
    root = _stubs.make_tree(str(tmp_path / "mixed"), "patched")
    os.remove(os.path.join(root, "rf3", "graph_flags.py"))
    assert tree.classify(root, lists).state == "unknown"
    # an empty directory
    os.makedirs(tmp_path / "empty")
    assert tree.classify(str(tmp_path / "empty"), lists).state == "unknown"
    # a stock tree with a stray graph_flags.py is not stock
    r = _stubs.make_tree(str(tmp_path / "stray"), "stock")
    open(os.path.join(r, "rf3", "graph_flags.py"), "w").write("# stray\n")
    t3 = tree.classify(r, lists)
    assert t3.state == "unknown" and t3.unknown == ["rf3/graph_flags.py"]
    assert tree.STATES == ("stock", "patched")


def test_locate_of_names_each_package_where_it_lives(tmp_path, lists):
    """An editable checkout at the pin puts rf3 under models/rf3/src and foundry under src (stock/src/pyproject.toml): the tree state reads
    rf3 only; ``locate_of`` finds either package where it lives (the venv derivation copies both) and refuses by name when one is absent."""
    root = _stubs.make_tree(str(tmp_path / "models_src"), "stock")
    other = str(tmp_path / "src")
    os.makedirs(other)
    os.rename(os.path.join(root, "foundry"), os.path.join(other, "foundry"))
    assert tree.classify(root, lists).state == "stock"                          # foundry's location is not part of the state
    py = _stubs.make_interpreter(str(tmp_path / "bin"), os.pathsep.join([root, other]))
    assert os.path.realpath(tree.locate_of(py)) == os.path.realpath(root) and os.path.realpath(tree.locate_of(py, "foundry")) == os.path.realpath(other)
    assert tree.state_of(py, lists).state == "stock"
    py2 = _stubs.make_interpreter(str(tmp_path / "bin2"), root, isolated=True)          # no site-packages: absent on the proof box's interpreter too
    with pytest.raises(RuntimeError, match="foundry package is not installed"):
        tree.locate_of(py2, "foundry")


def test_state_of_second_interpreter(tmp_path, lists):
    root = _stubs.make_tree(str(tmp_path / "sp"), "stock")
    py = _stubs.make_interpreter(str(tmp_path / "bin"), root)
    ts = tree.state_of(py, lists)
    assert ts.state == "stock" and os.path.realpath(ts.site_packages) == os.path.realpath(root)


def test_state_of_this_interpreter():
    import importlib.util
    if importlib.util.find_spec("rf3") is None:
        with pytest.raises(RuntimeError, match="rf3 package is not installed"):
            tree.this_site_packages()
    else:
        assert os.path.isdir(os.path.join(tree.this_site_packages(), "rf3"))
