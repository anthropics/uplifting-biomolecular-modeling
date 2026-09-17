"""tp_route's BIND table (the carried unit's trunk / relpos / template seams -> the kit's bindings on the shared core): every BIND target
is a ``protenix_opt.tp_bind`` module and defines every bound name; every bound name is a public definition of the carried module (by
source — the carried unit imports protenix/torch); every binding module imports only the shared core's rowpair package, torch, protenix
and its sibling bindings at module level (the carried unit's seam registry / instrumentation and the featuriser's template rows are
reached by function-local imports, listed here by name), and holds no collective and no bare Layout constructor of its own. CPU only."""
import ast
import importlib
import os

import pytest

from protenix_opt import tp, tp_route

HERE = os.path.dirname(os.path.abspath(__file__))
BIND_DIR = os.path.join(os.path.dirname(HERE), "tp_bind")
UNIT_PKG = os.path.join(tp.unit_dir(), "ptx_tp")
ALLOWED_TOP = ("os", "time", "typing", "torch", "opt_core.mem.rowpair", "protenix", "protenix_opt.tp_bind")
ALLOWED_LOCAL = {                                                        # function-local imports, by binding module
    "trunk": {"ptx_tp", "ptx_tp.trunk", "ptx_tp.template_real", "protenix.model", "protenix.utils.torch_utils"},
    "template": set(), "relpos": set(), "__init__": set(),
}
FORBIDDEN_CALLS = ("torch.distributed", "all_gather_rows(", "transpose_shards(", "alltoall_window(", "all_gather(", "broadcast(", "allreduce_(")


def _defs(path):
    tree = ast.parse(open(path).read())
    return {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))} | \
           {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}


def test_bind_targets_are_binding_modules_and_define_every_name():
    assert tp_route.BIND and set(tp_route.BOUND) == {"trunk", "template", "diffusion", "pairformer", "msa"}
    for carried, (target, names) in tp_route.BIND.items():
        assert carried.startswith("ptx_tp.") and target.startswith(tp_route.BIND_PKG + "."), (carried, target)
        defs = _defs(os.path.join(BIND_DIR, target.rsplit(".", 1)[1] + ".py"))
        missing = [n for n in names if n not in defs]
        assert not missing, (target, missing)


def test_every_bound_name_is_a_public_definition_of_the_carried_module():
    if not os.path.isdir(UNIT_PKG):
        pytest.skip("carried unit not in this tree")
    for carried, (_target, names) in tp_route.BIND.items():
        defs = _defs(os.path.join(UNIT_PKG, carried.split(".", 1)[1] + ".py"))
        missing = [n for n in names if n not in defs]
        assert not missing, (carried, missing)


def test_bind_rows_equal_the_bindings_route_rows():
    """tp_route.BIND is the union of the binding modules' ROUTE_ROWS (read by source: the bindings import torch)."""
    rows = {}
    for mod in ("trunk", "template", "pairstack"):
        tree = ast.parse(open(os.path.join(BIND_DIR, mod + ".py")).read())
        node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ROUTE_ROWS" for t in n.targets))
        rows.update(ast.literal_eval(node.value))
    rows["ptx_tp.diffusion"] = ("protenix_opt.tp_bind.diffusion", ("tp_prepare_cache", "tp_sample_diffusion", "report"))   # the diffusion binding names its row in tp_route
    assert {k: (v[0], tuple(v[1])) for k, v in rows.items()} == {k: (v[0], tuple(v[1])) for k, v in tp_route.BIND.items()}


def test_routes_and_bind_do_not_overlap_and_lines_differ():
    assert not set(tp_route.ROUTES) & set(tp_route.BIND)
    assert tp_route.LINE != tp_route.BIND_LINE == "TP-BIND"
    text = "[protenix-opt] TP-BIND ptx_tp.trunk -> protenix_opt.tp_bind.trunk names=7 core=?\n[protenix-opt] TP-ROUTE ptx_tp.dist -> x names=1\n"
    assert tp_route.bind_lines(text) == ["ptx_tp.trunk"] and tp_route.route_lines(text) == ["ptx_tp.dist"]


@pytest.mark.parametrize("mod", ["__init__", "relpos", "trunk", "template"])
def test_binding_imports_only_the_core_and_holds_no_collective(mod):
    src = open(os.path.join(BIND_DIR, mod + ".py")).read()
    tree = ast.parse(src)
    top, local = [], []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if isinstance(node, ast.ImportFrom) and node.level > 0:            # from . import x -> sibling binding
                names = ["protenix_opt.tp_bind"]
            (top if node.col_offset == 0 else local).extend(names)
    bad_top = [n for n in top if not any(n == a or n.startswith(a + ".") for a in ALLOWED_TOP)]
    assert not bad_top, (mod, bad_top)
    bad_local = [n for n in local if n not in ALLOWED_LOCAL[mod]]
    assert not bad_local, (mod, bad_local)
    hits = [c for c in FORBIDDEN_CALLS if c in src]
    assert not hits, (mod, hits)
    assert "Layout(" not in src.replace("Layout(N", "")                      # no bare Layout constructor (the refusing forms only)


def test_binding_modules_import_with_the_core_alone():
    try:
        import torch  # noqa: F401
        import opt_core.mem.rowpair  # noqa: F401
    except ImportError:
        pytest.skip("torch + the shared core required")
    for mod in ("relpos", "template", "trunk"):
        m = importlib.import_module(f"protenix_opt.tp_bind.{mod}")
        assert m.__name__ == f"protenix_opt.tp_bind.{mod}"
