"""The kit is carried once: it is identified in this repo by the git commit carrying it (no in-repo manifest restates it), no kernel the
shared core carries is carried twice (they are routed from opt_core/kernels), and the files the package reads (levers_ptx1.py, the detpatch pair, the PYTHONPATH directories, the
lever files) are among them."""
import os

from .conftest import KIT, n_kit_files
from protenix_v1_opt import kit as K


def test_kit_files_present_and_counted():
    n = n_kit_files()
    assert n > 0
    chk = K.check_files(KIT)
    assert chk == {"ok": True, "required": len(K.REQUIRED_RELPATHS), "missing": [], "files": n}


def test_files_the_package_reads_are_carried():
    assert os.path.isfile(os.path.join(KIT, K.LEVERS_RELPATH))
    for rel in K.DETPATCH_RELPATHS:
        assert os.path.isfile(os.path.join(KIT, rel)), rel
    assert os.path.isfile(os.path.join(KIT, "README.md"))
    for p in K.kit_sys_paths(KIT):
        assert os.path.isdir(p), p
    assert os.path.isfile(os.path.join(KIT, "inputs", "p995_1brs.json"))


CELLS = os.path.join("lib", "fpf_trimul_v4", "cells.json")                    # the kit's tuned cell table for fpf_trimul_v4: per-kit data the shared core does not carry (opt_core/kernels/SUMS/fpf_trimul_v4.json "not_carried")


def test_no_kernel_the_core_carries_is_carried_twice():
    """The kernels the levers reach live once, in opt_core/kernels (stack.ROUTED_KERNELS, routed by name): the kit directory carries
    no file their sums files list — at the package path or loose under lib/ — and no cell table of its own."""
    from opt_core import kernels as KR
    from protenix_v1_opt import stack
    carried_paths = set(K.tree_files(KIT))
    assert CELLS not in carried_paths and not os.path.exists(os.path.join(KIT, CELLS))   # kit 0.2.38: no kit TriMul cell table (the provider's own tables serve)
    twice = []
    for name in stack.ROUTED_KERNELS:
        doc = KR.sums(name)
        for rel in doc["files"]:
            p = os.path.join("lib", name, rel) if doc["kind"] == "package" else os.path.join("lib", rel)
            if p in carried_paths or os.path.exists(os.path.join(KIT, p)):
                twice.append(p)
    assert twice == [], twice


def test_no_bytecode_or_symlink_inside_the_kit():
    for root, dirs, names in os.walk(KIT):
        assert "__pycache__" not in dirs, root
        for n in names:
            p = os.path.join(root, n)
            assert not os.path.islink(p), p
            assert not n.endswith((".pyc", ".pyo")), p


def test_the_trimul_adapter_has_no_size_ceiling_in_any_mode():
    """`_trimul_forward` (levers_ptx1.py) compares the token count against its floor only (`N <= 100`, tiny items keep the stock statement):
    no `N > ...` / `N >= ...` condition in any mode — the exact construction and the fast cell apply at every size."""
    import ast
    with open(K.levers_file(), "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_trimul_forward"]
    assert len(fn) == 1, "levers_ptx1.py: no top-level _trimul_forward"
    ceilings = [ast.unparse(n) for n in ast.walk(fn[0]) if isinstance(n, ast.Compare) and isinstance(n.left, ast.Name) and n.left.id == "N"
                and any(isinstance(op, (ast.Gt, ast.GtE)) for op in n.ops)]
    assert ceilings == [], ceilings
