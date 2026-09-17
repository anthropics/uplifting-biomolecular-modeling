"""Every attribute this kit reads off the eager stack's modules INSIDE a function body resolves against the carried source.

The surface test (test_lever_surfaces) checks module-level names a registry line cites; it cannot see a runtime lookup such as
``from chai1_eager import kernels as K; K.CFG_sdpa()`` inside ``big._triattn_chunked`` (the chunked triangle-attention path engages only
at crop >= 1536, so no CPU activation test reaches it).  This test parses the kit modules' function bodies statically: for every local import of
``chai1_eager.<mod>`` (``import ... as X`` or ``from chai1_eager.<mod> import a, b``) it asserts that each ``X.<attr>`` read and each imported
name is defined at module level in the carried file ``opt/forward/eager_trunk/chai1_eager/<mod>.py`` (def / class / assignment / __all__ entry)."""
import ast
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.normpath(os.path.join(HERE, ".."))                                   # opt/chai1_opt
EAGER = os.path.normpath(os.path.join(KIT, "..", "forward", "eager_trunk", "chai1_eager"))
KIT_MODULES = ("big.py", "pairtrack.py", "stack.py", "triattn_core.py", "trunk_n.py")


def module_level_names(path):
    tree = ast.parse(open(path).read())
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        names.add(n.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, (ast.If, ast.Try, ast.With)):
            for sub in ast.walk(node):
                if isinstance(sub, (ast.FunctionDef, ast.ClassDef)):
                    names.add(sub.name)
                elif isinstance(sub, ast.Assign):
                    for t in sub.targets:
                        if isinstance(t, ast.Name):
                            names.add(t.id)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def eager_references(path):
    """[(func, eager_module, attr, lineno)] for every attribute read / imported name of a chai1_eager module inside a function body."""
    tree = ast.parse(open(path).read()); out = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        aliases = {}                                                                # local alias -> eager module name
        for node in ast.walk(fn):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "chai1_eager":
                parts = node.module.split(".")
                if len(parts) == 1:                                                # from chai1_eager import kernels as K
                    for a in node.names:
                        aliases[a.asname or a.name] = a.name
                else:                                                              # from chai1_eager.trunk import BF, bfw
                    for a in node.names:
                        out.append((fn.name, parts[1], a.name, node.lineno))
            elif isinstance(node, ast.Import):
                for a in node.names:
                    parts = a.name.split(".")
                    if parts[0] == "chai1_eager" and len(parts) == 2 and a.asname:
                        aliases[a.asname] = parts[1]
        for node in ast.walk(fn):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases:
                out.append((fn.name, aliases[node.value.id], node.attr, node.lineno))
    return out


class TestEagerAttributeLookupsResolve(unittest.TestCase):
    def test_every_runtime_lookup_names_something_the_carried_stack_defines(self):
        defined = {}
        missing = []
        checked = 0
        for km in KIT_MODULES:
            path = os.path.join(KIT, km)
            if not os.path.isfile(path):
                continue
            for func, emod, attr, line in eager_references(path):
                src = os.path.join(EAGER, emod + ".py")
                if not os.path.isfile(src):                                        # a package or a module this checkout does not carry: not this test's subject
                    continue
                if emod not in defined:
                    defined[emod] = module_level_names(src)
                checked += 1
                if attr not in defined[emod]:
                    missing.append(f"{km}:{line} {func}(): chai1_eager.{emod}.{attr}")
        self.assertGreater(checked, 0, "no chai1_eager lookups found — the test's parser lost its subject")
        self.assertEqual(missing, [], "runtime lookups of names the carried eager stack no longer defines:\n  " + "\n  ".join(missing))

    def test_the_chunked_triangle_attention_path_reads_no_removed_plug(self):
        """big._triattn_chunked serves crops >= the trunk_chunk gate: its per-block attention is the module's own SDPA statement — no
        CFG_sdpa / low-copy composition."""
        src = open(os.path.join(KIT, "big.py")).read()
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_triattn_chunked")
        body = ast.get_source_segment(src, fn)
        for gone in ("CFG_sdpa", "triattn_lowcopy", "CFG_lowcopy"):
            self.assertNotIn(gone, body)
        self.assertIn("scaled_dot_product_attention", body)


if __name__ == "__main__":
    unittest.main()
