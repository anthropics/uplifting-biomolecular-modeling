"""chai1_eager/ts2eager.py reads an exported component's class files from the .pt archive itself.  A code tree already on disk (a temp or
shared path another user could have filled) is neither read nor written when a component loads, so no file found there is ever compiled;
the text parsed is the text extract_code_tree writes.  CPU only (torch for the module's import)."""
import ast, builtins, importlib.util, os, sys, zipfile

import pytest

pytest.importorskip("torch")
HERE = os.path.dirname(os.path.abspath(__file__))
EAGER = os.path.normpath(os.path.join(HERE, "..", "..", "forward", "eager_trunk", "chai1_eager"))
A = 'class Block(Module):\n  __parameters__ = ["weight", ]\n  __buffers__ = []\n  weight : Tensor\n  training : bool\n  sub : __torch__.sub.b.Leaf\n  def forward(self: __torch__.a.Block,\n    x: Tensor) -> Tensor:\n    return torch.add(x, self.weight)\n'
B = 'class Leaf(Module):\n  __parameters__ = []\n  __buffers__ = ["scale", ]\n  scale : Tensor\n  def forward(self: __torch__.sub.b.Leaf,\n    x: Tensor) -> Tensor:\n    return torch.mul(x, self.scale)\n'
PLANTED = 'class Planted(Module):\n  __parameters__ = []\n  def forward(self: __torch__.a.Planted,\n    x: Tensor) -> Tensor:\n    return x\n'


@pytest.fixture(scope="module")
def T():
    spec = importlib.util.spec_from_file_location("chai1_ts2eager_under_test", os.path.join(EAGER, "ts2eager.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def _archive(path, a=A):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("model/code/__torch__/a.py", a); z.writestr("model/code/__torch__/a.py.debug_pkl", b"\x80\x02]q\x00.")
        z.writestr("model/code/__torch__/sub/b.py", B); z.writestr("model/data.pkl", b"\x80\x02}q\x00."); z.writestr("model/constants.pkl", b"\x80\x02)q\x00.")
    return str(path)


def _records(tree):
    return {q: (c.name, list(c.params), list(c.buffers), dict(c.attrs), dict(c.methods_src)) for q, c in tree.classes.items()}


def test_archive_text_is_the_text_the_extracted_tree_holds(T, tmp_path):
    pt = _archive(tmp_path / "m.pt")
    src = T.archive_sources(pt)
    assert sorted(src) == ["__torch__/a.py", "__torch__/sub/b.py"] and src["__torch__/a.py"] == A     # class files only; the .debug_pkl beside them is no class file
    d = T.extract_code_tree(pt, str(tmp_path / "tree"))
    from_dir, from_archive = T.TSCodeTree(d), T.TSCodeTree(d, sources=src)
    assert _records(from_dir) == _records(from_archive) and sorted(from_archive.classes) == ["__torch__.a.Block", "__torch__.sub.b.Leaf"]
    assert from_archive.classes["__torch__.a.Block"].methods_src["forward"].startswith("def forward(self: __torch__.a.Block,")


def test_carriage_returns_read_as_a_text_file_reads_them(T, tmp_path):
    pt = _archive(tmp_path / "m.pt", a=A.replace("\n", "\r\n"))
    d = T.extract_code_tree(pt, str(tmp_path / "tree"))
    assert T.archive_sources(pt)["__torch__/a.py"] == A and _records(T.TSCodeTree(d)) == _records(T.TSCodeTree(d, sources=T.archive_sources(pt)))


def test_a_tree_already_on_disk_is_never_opened(T, tmp_path, monkeypatch):
    pt = _archive(tmp_path / "m.pt")
    planted = tmp_path / "shared" / "m.pt.code"; (planted / "__torch__").mkdir(parents=True)
    (planted / "__torch__" / "a.py").write_text(PLANTED); (planted / "__torch__" / "extra.py").write_text(PLANTED.replace("Planted", "Extra"))
    before = sorted(os.listdir(str(planted / "__torch__")))
    real_open = builtins.open

    def guarded(f, *a, **k):
        assert not os.path.abspath(str(f)).startswith(str(planted)), f"opened {f}"
        return real_open(f, *a, **k)
    monkeypatch.setattr(builtins, "open", guarded)
    tree = T.TSCodeTree(str(planted), sources=T.archive_sources(pt))
    assert sorted(tree.classes) == ["__torch__.a.Block", "__torch__.sub.b.Leaf"] and sorted(os.listdir(str(planted / "__torch__"))) == before
    monkeypatch.setattr(builtins, "open", real_open)
    assert "__torch__.a.Planted" in T.TSCodeTree(str(planted)).classes                                 # the directory form reads what it is given: it is for a caller's own tree


def test_a_component_load_parses_the_archive_and_extracts_nothing():
    src = open(os.path.join(EAGER, "ts2eager.py"), encoding="utf-8").read()
    fn = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "load_eager_component"][0]
    calls = {(c.func.id if isinstance(c.func, ast.Name) else c.func.attr) for c in ast.walk(fn) if isinstance(c, ast.Call)}
    assert "archive_sources" in calls and "TSCodeTree" in calls and not calls & {"extract_code_tree", "isdir", "exists", "makedirs"}, calls


def test_no_code_directory_default_is_shared_between_users():
    for rel in (os.path.join(EAGER, "stack.py"), os.path.join(EAGER, "ts2eager.py"), os.path.join(HERE, "_stubs.py")):
        assert '"/tmp/chai1_eager_code"' not in open(rel, encoding="utf-8").read(), rel
    stack = open(os.path.join(EAGER, "stack.py"), encoding="utf-8").read()
    assert 'code_root=None' in stack and '"chai1_eager_code-uid%d" % os.getuid()' in stack
