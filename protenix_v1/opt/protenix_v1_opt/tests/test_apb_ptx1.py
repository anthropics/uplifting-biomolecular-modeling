"""The sampler attention-with-pair-bias adapter (opt/forward/v05_addon/ptxfpf/apb_ptx1.py, lever words ditattn / ditattnfp16 / atomattn over
the shared core's carried kernel package opt_core/kernels/apb/fpf_apb): importable without torch; its routing decisions are pure functions of shapes, dtype names
and the card's capability — a call or a card the cells do not serve steps aside BY NAME (a `stock:<word>` / a named condition), never silently,
never as a refusal. No GPU needed; the one torch test runs where torch imports and skips itself on a CUDA machine."""
import ast
import os
import sys

import pytest

from .conftest import KIT

PTXFPF = os.path.join(KIT, "ptxfpf")
def _core_pkg(sub):
    """The shared core's carried package dir for an apb row word's subpackage (opt_core/kernels/apb/<sub>), located without importing torch."""
    import importlib.util
    spec = importlib.util.find_spec("opt_core")
    return os.path.join(os.path.dirname(spec.origin), "kernels", "apb", sub)


PKG = _core_pkg("fpf_apb")                            # rows fpf_apb / fpf_atom: the core's carried fpf_apb 0.2.2, bound by word since kit 0.2.21 


@pytest.fixture(scope="module")
def A():
    sys.path.insert(0, PTXFPF)
    try:
        sys.modules.pop("apb_ptx1", None)
        import apb_ptx1 as mod
    finally:
        sys.path.pop(0)
    return mod


def _literal(path, name):
    """The module-level `name = <literal>` of a source file (ast; no import)."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            try:
                return ast.literal_eval(node.value)
            except ValueError:                                                     # a table spelled with dict(...) calls: evaluated with `dict` as the only name
                return eval(compile(ast.Expression(node.value), path, "eval"), {"__builtins__": {}}, {"dict": dict})
    raise KeyError(name)


def test_adapter_imports_without_torch():
    with open(os.path.join(PTXFPF, "apb_ptx1.py"), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = {a.name.split(".")[0] for n in top for a in n.names} | {n.module.split(".")[0] for n in top if isinstance(n, ast.ImportFrom) and n.module}
    assert not names & {"torch", "triton", "protenix", "opt_core.kernels.apb.fpf_apb.apb_triton"}, names


def test_carried_package_files_and_provenance(A):
    for f in ("__init__.py", "apb_triton.py", "atom_triton.py", "install.py", "pf_triton.py", "NOTICE.md"):
        assert os.path.isfile(os.path.join(PKG, f)), f
    assert _literal(os.path.join(PKG, "__init__.py"), "__version__") == "0.2.2"
    for f in ("CELLS.json", "VECTORS.json", "loadcheck.py"):
        assert os.path.isfile(os.path.join(PKG, f)), f
    assert A.PACKAGE == "opt_core.kernels.apb.fpf_apb" and A.LEVERS == ("ditattn", "ditattnfp16", "atomattn")


def test_cells_mirror_the_carried_package(A):
    theirs = _literal(os.path.join(PKG, "install.py"), "CELLS")
    assert A.CELLS["dit"] == theirs["dit_attn"]
    assert A.CELLS["atom"] == theirs["atom_attn"]
    assert A.REFERENCE_CC in A.CELLS["dit"] and A.REFERENCE_CC in A.CELLS["atom"]


def test_card_cell_measured_named_and_aside(A):
    assert A.card_cell("dit", None) == (None, None, "no CUDA device")
    cell, key, named = A.card_cell("dit", (7, 5))
    assert cell is None and key is None and named.startswith("arch=sm_75")            # below Ampere: the lever installs on nothing, named
    assert A.card_cell("dit", (9, 0)) == ({"opd": "tf32x3", "bias_tma": True}, "9.0", None)
    assert A.card_cell("dit", (9, 0), fp16=True) == ({"opd": "fp16", "precast": True, "bias_tma": True}, "9.0", None)
    assert A.card_cell("atom", (9, 0)) == ({"opd": "tf32rn"}, "9.0", None)
    for cc in ((8, 0), (10, 0), (12, 0)):                                             # unmeasured cards >= 8.0 ENGAGE the 9.0 cell and are named
        cell, key, named = A.card_cell("atom", cc)
        assert cell == {"opd": "tf32rn"} and key == "9.0" and "sm_%d%d" % cc in named and "cell=9.0" in named
    cell, key, named = A.card_cell("dit", ("8", "6"), fp16=True)
    assert cell == {"opd": "fp16", "precast": True, "bias_tma": True} and key == "9.0" and "sm_86" in named


def test_geometry_words(A):
    assert A.geometry_word("dit", 16, 48, True) is None
    assert A.geometry_word("atom", 4, 32, True) is None
    assert A.geometry_word("dit", 16, 48, False) == "geometry=16x48:nogate"
    assert A.geometry_word("dit", 16, 40, True) == "geometry=16x40"
    assert A.geometry_word("dit", 12, 64, True) is None                               # another head count at a served width: per-head kernel, served
    assert A.geometry_word("atom", 8, 16, True) == "geometry=8x16"
    assert A.geometry_word("atom", 4, 32, False) == "geometry=4x32:nogate"


def test_dit_route_served_and_aside(A):
    N, S, H = 384, 5, 16
    q = (S, N, 768)
    assert A.dit_route(q, q, (1, H, N, N), H) is None
    assert A.dit_route(q, q, (H, N, N), H) is None
    assert A.dit_route(q, q, (1, 1, H, N, N), H) is None
    assert A.dit_route((1, S, N, 768), (1, S, N, 768), (1, 1, H, N, N), H) is None
    assert A.dit_route(q, q, (1, H, N + 8, N + 8), H) is None                           # a padded square bias (row pitch >= N) is read in place
    assert A.dit_route(q, q, (1, H, N, N), H, dtype="bfloat16") is None
    assert A.dit_route(q, q, (S, H, N, N), H) == "stock:bias_per_sample"               # a real sample dim on the bias: not the shared pair bias
    assert A.dit_route(q, q, (S, 1, N, N), H) == "stock:bias_per_sample"
    assert A.dit_route(q, q, (2, 1, H, N, N), H) == "stock:bias_per_sample"
    assert A.dit_route(q, q, None, H) == "stock:no_bias"
    assert A.dit_route(q, q, (1, 8, N, N), H) == "stock:bias_shape"
    assert A.dit_route(q, q, (1, H, N, N + 8), H) == "stock:bias_shape"
    assert A.dit_route(q, q, (1, H, N - 1, N - 1), H) == "stock:bias_shape"
    assert A.dit_route(q, (S, N + 32, 768), (1, H, N, N + 32), H) == "stock:kv_len"
    assert A.dit_route(q, (S + 1, N, 768), (1, H, N, N), H) == "stock:kv_len"
    assert A.dit_route(q, q, (1, H, N, N), H, local=True) == "stock:local_call"
    assert A.dit_route(q, q, (1, H, N, N), H, is_cuda=False) == "stock:device"
    assert A.dit_route(q, q, (1, H, N, N), H, dtype="float64") == "stock:dtype=float64"
    assert A.dit_route((768,), q, (1, H, N, N), H) == "stock:q_shape"


def test_atom_route_served_and_aside(A):
    S, H, NQ, NK = 5, 4, 32, 128
    for n_atom in (1000, 1024, 33, 32):
        T = -(-n_atom // NQ)
        q = (S, n_atom, 128)
        assert A.atom_route(q, q, None, (1, H, T, NQ, NK), H, NQ, NK) is None
        assert A.atom_route(q, q, None, (H, T, NQ, NK), H, NQ, NK) is None
        assert A.atom_route(q, q, None, (1, 1, H, T, NQ, NK), H, NQ, NK) is None
        assert A.atom_route(q, q, None, (S, H, T, NQ, NK), H, NQ, NK) == "stock:bias_per_sample"
        assert A.atom_route(q, q, None, (1, H, T + 1, NQ, NK), H, NQ, NK) == "stock:bias_shape"
    q = (S, 1000, 128); T = 32
    assert A.atom_route(q, q, None, (1, H, T, NQ, NK), H, None, None) == "stock:global_call"
    assert A.atom_route(q, q, None, (1, H, T, 16, 64), H, 16, 64) == "stock:window=16x64"
    assert A.atom_route(q, q, (1, H, 1000, 1000), (1, H, T, NQ, NK), H, NQ, NK) == "stock:dense_bias"
    assert A.atom_route(q, q, None, None, H, NQ, NK) == "stock:no_bias"
    assert A.atom_route(q, (S, 999, 128), None, (1, H, T, NQ, NK), H, NQ, NK) == "stock:kv_len"
    assert A.atom_route(q, q, None, (1, H, T, NQ, NK), H, NQ, NK, is_cuda=False) == "stock:device"
    assert A.atom_route(q, q, None, (1, H, T, NQ, NK), H, NQ, NK, dtype="float64") == "stock:dtype=float64"
    assert A.atom_route(q, q, None, (1, 2, T, NQ, NK), H, NQ, NK) == "stock:bias_shape"


def test_served_words(A):
    assert A.served_word("float32", "tf32x3") == "apb:tf32x3"
    assert A.served_word("float32", "fp16") == "apb:fp16"
    assert A.served_word("float32", "tf32rn") == "apb:tf32rn"
    assert A.served_word("bfloat16", "tf32x3") == "apb:bf16in"
    assert A.served_word("float16", "tf32rn") == "apb:fp16in"


def test_lever_facts_before_install_and_fp16_alone(A):
    for w in A.LEVERS:
        f = A.lever_facts(w)
        assert f["state"] == "skipped" and f["served"] == 0 and f["reason"], (w, f)
    assert A.lever_facts("ditattnfp16")["reason"] == "rides_ditattn"
    d = A.apply(None, ditattn=False, ditattnfp16=True, atomattn=False)                # the precision lever alone: nothing to install, named
    assert d["dit"]["engaged"] is False and "ditattnfp16 without ditattn" in d["dit"]["named"]
    assert A.lever_facts("ditattnfp16")["state"] == "skipped"
    assert A.lever_facts("ditattn")["state"] == "skipped"
    assert d["atom"]["engaged"] is False


def test_install_without_cuda_is_named_not_silent(A):
    torch = pytest.importorskip("torch", reason="the install path imports torch")
    if torch.cuda.is_available():
        pytest.skip("a CUDA device is present: the no-CUDA naming is this test's subject")
    f = A.install_ditattn(object())
    assert f["engaged"] is False and f["named"] == "no CUDA device" and f["installed_on"] == 0
    f = A.install_atomattn(object())
    assert f["engaged"] is False and f["named"] == "no CUDA device"
    assert A.lever_facts("atomattn") == dict(A.lever_facts("atomattn"), state="skipped", reason="no CUDA device")
    A.uninstall(None)
