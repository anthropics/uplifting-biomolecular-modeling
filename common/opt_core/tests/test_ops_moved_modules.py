"""opt_core.ops (+ opt_core.tools.graph_audit) — the modules shared by more than one kit, moved here verbatim as the ONE copy (kits import them by module path).

Static, interpreter-only checks (no torch / triton import): the modules are present under their neutral package names, the packages
import nothing at package level (a kit that imports one module never drags the others' torch / triton imports in), the Triton kernels keep
64-bit row addressing and plain `@triton.jit` (no autotune state: launch-to-launch deterministic), the modules import only torch / triton /
the standard library (nothing from a kit tree), and the fused DiT / atom row kernels the same kits share are the pair-bias provider's carry
(kernels/apb/ditfast), imported by the kits from there — not duplicated under ops/.
"""
import ast
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, os.pardir, "opt_core")
OPS = os.path.join(PKG, "ops")
MODULES = {"msa_fused": "msa_triton.py"}
DIT_CARRY = os.path.join(PKG, "kernels", "apb", "ditfast")        # the DiT / atom Triton row kernels the same kits share: the pair-bias provider's carry (not duplicated under ops/)
TOOLS = os.path.join(PKG, "tools")
AUDIT = os.path.join(TOOLS, "graph_audit", "audit.py")     # the capture-safety instruments the same kits shared: tooling, torch imported on first use
TRITON_MODULES = ("msa_fused/msa_triton.py",)
STDLIB_OK = {"__future__", "collections", "contextlib", "hashlib", "os", "sys", "json", "math", "time", "typing", "functools", "itertools",
             "dataclasses", "threading", "traceback", "warnings", "weakref", "inspect", "types", "atexit", "random", "struct", "io", "re"}


def _src(rel):
    with open(os.path.join(OPS, rel), encoding="utf-8") as f:
        return f.read()


def test_the_moved_modules_are_present_under_neutral_package_names():
    assert os.path.isfile(os.path.join(OPS, "__init__.py")) and os.path.isfile(os.path.join(TOOLS, "__init__.py"))
    assert os.path.isfile(AUDIT) and os.path.isfile(os.path.join(TOOLS, "graph_audit", "__init__.py"))
    for pkg, mod in MODULES.items():
        d = os.path.join(OPS, pkg)
        assert os.path.isfile(os.path.join(d, "__init__.py")), pkg
        assert os.path.isfile(os.path.join(d, mod)), (pkg, mod)


def test_packages_import_nothing_at_package_level():
    inits = [os.path.join(OPS, "__init__.py")] + [os.path.join(OPS, p, "__init__.py") for p in MODULES] + [os.path.join(TOOLS, "__init__.py"), os.path.join(TOOLS, "graph_audit", "__init__.py")]
    for rel in inits:
        with open(rel, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        body = [n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant))]
        assert body == [], "%s must be a docstring only (found %s)" % (rel, [type(n).__name__ for n in body])


def test_modules_import_only_torch_triton_and_the_standard_library():
    for pkg, mod in MODULES.items():
        tree = ast.parse(_src(os.path.join(pkg, mod)))
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                roots = {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom):
                assert n.level == 0, "%s/%s: relative import (the module must stand alone)" % (pkg, mod)
                roots = {n.module.split(".")[0]}
            else:
                continue
            assert roots <= STDLIB_OK | {"torch", "triton"}, "%s/%s imports %s" % (pkg, mod, sorted(roots - STDLIB_OK - {"torch", "triton"}))


def test_graph_audit_imports_torch_on_first_use_only():
    with open(AUDIT, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    roots = {a.name.split(".")[0] for n in top if isinstance(n, ast.Import) for a in n.names} | {n.module.split(".")[0] for n in top if isinstance(n, ast.ImportFrom)}
    assert roots <= STDLIB_OK, "graph_audit/audit.py imports %s at module top level (torch / numpy belong inside the functions that use them)" % sorted(roots - STDLIB_OK)


def test_triton_row_kernels_are_plain_jit_with_int64_row_addressing():
    for rel in TRITON_MODULES:
        s = _src(rel)
        assert "@triton.jit" in s and "@triton.autotune" not in s, rel
        assert ".to(tl.int64)" in s, rel                              # row offsets formed in int64 (rows x stride can pass 2^31 at large N x samples)


def test_dit_row_kernels_are_the_pair_bias_providers_carry_and_not_duplicated_here():
    for name in ("kernels.py", "atom_kernels.py"):
        assert os.path.isfile(os.path.join(DIT_CARRY, name)), name                # kits import opt_core.kernels.apb.ditfast.{kernels, atom_kernels}
    assert not os.path.isdir(os.path.join(OPS, "dit_fused")) and not os.path.isdir(os.path.join(OPS, "dit_atom")), "one copy inside the core"
