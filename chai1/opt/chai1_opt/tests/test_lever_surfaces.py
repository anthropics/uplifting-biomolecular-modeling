"""Every lever's declared surface exists in the kit's source — CPU-only, static (AST), no import of the lever modules (no torch / triton /
CUDA touch): for each lever in ``registry.LEVERS`` the file ``kit_file`` names exists, every ``<file>.py:<name>[,<name>…]`` item of its
``lines`` field resolves to a file under opt/ whose module level defines those names (``Class.method[/method…]`` checks the class body), and
every module that declares a literal ``SURFACE = (...)`` tuple defines each name in it. A helper dropped from a lever module (say the Router's
``admits`` / ``_op``) fails here, before a fold meets a NameError. Data-driven: a new lever is covered by its registry ``lines``;
a module opts into the stricter check by adding a ``SURFACE`` tuple."""
import ast
import os
import re
import unittest

from chai1_opt import registry

OPT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))          # <kit>/opt
SKIP_DIRS = ("stock", "tests", "__pycache__", ".eggs", "build")
IDENT = re.compile(r"^\s*([A-Za-z_]\w*)((?:\.\w+)?(?:/\w+)*)\s*(?:\(.*\))?\s*$")             # name | Class.method | Class.m1/m2 | Name(...)
# Registry `lines` text another track owns that names a helper its module no longer defines: tolerated BY NAME here (reported, not failed)
# until that owner refreshes the text; an entry whose name exists again fails the test (remove it then).
KNOWN_STALE = {}


def py_index():
    idx = {}
    for root, dirs, files in os.walk(OPT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.endswith(".egg-info")]
        for f in files:
            if f.endswith(".py"):
                p = os.path.join(root, f); rel = os.path.relpath(p, OPT)
                idx.setdefault(f, []).append(rel)
                parts = rel.split(os.sep)
                if len(parts) >= 2:
                    idx.setdefault("/".join(parts[-2:]), []).append(rel)
    return idx


def module_names(path, _cache={}):
    """top-level names a module defines (def / class / assign / annotated assign / import-as / global-decl in functions ignored) + {class: {methods}}"""
    if path in _cache:
        return _cache[path]
    tree = ast.parse(open(os.path.join(OPT, path), encoding="utf-8").read(), filename=path)
    names, classes, surface = set(), {}, None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
            classes[node.name] = {n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))} | \
                                 {t.id for n in node.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
            for b in node.bases:                                   # methods inherited from a base defined in the same module count
                if isinstance(b, ast.Name) and b.id in classes:
                    classes[node.name] |= classes[b.id]
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        names.add(n.id)
                if isinstance(t, ast.Name) and t.id == "SURFACE" and isinstance(node.value, (ast.Tuple, ast.List)):
                    surface = tuple(e.value for e in node.value.elts if isinstance(e, ast.Constant) and isinstance(e.value, str))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, (ast.If, ast.Try)):                  # names bound under a module-level if/try (optional imports, fallbacks)
            for sub in ast.walk(node):
                if isinstance(sub, (ast.FunctionDef, ast.ClassDef)):
                    names.add(sub.name)
                elif isinstance(sub, ast.Assign):
                    for t in sub.targets:
                        if isinstance(t, ast.Name):
                            names.add(t.id)
                elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                    for a in sub.names:
                        names.add((a.asname or a.name).split(".")[0])
    for node in ast.walk(tree):                                    # `global X` assignments inside functions define module names too
        if isinstance(node, ast.Global):
            names.update(node.names)
    _cache[path] = (names, classes, surface)
    return _cache[path]


class TestLeverSurfaces(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.idx = py_index()

    def resolve(self, fname, near=None):
        """a `lines` file token -> path relative to opt/: next to the lever's kit_file first, else the unique match in the tree"""
        if near:
            cand = os.path.normpath(os.path.join(os.path.dirname(near), fname))
            if os.path.exists(os.path.join(OPT, cand)):
                return cand
            top = near.split("/")[0]
            same = [p for p in self.idx.get(fname, []) if p.split(os.sep)[0] == top or ("/" + top + "/") in ("/" + p)]
            if len(same) == 1:
                return same[0]
        hits = self.idx.get(fname, [])
        self.assertTrue(hits, f"{fname}: no such file under opt/ (registry lines / kit_file)")
        if len(hits) > 1 and near:
            pref = [h for h in hits if os.path.dirname(h).endswith(os.path.dirname(near)) or near.split("/")[0] in h.split(os.sep)]
            if len(pref) == 1:
                return pref[0]
        self.assertEqual(len(hits), 1, f"{fname} is ambiguous under opt/ ({hits}): qualify it in the registry lines")
        return hits[0]

    def kit_file_path(self, lever):
        kf = (lever.kit_file or "").strip()
        if not kf.endswith(".py"):
            return None
        hits = self.idx.get(kf) or self.idx.get(os.path.basename(kf)) or []
        hits = [h for h in hits if h.endswith(kf.replace("/", os.sep))] or hits
        self.assertTrue(hits, f"lever {lever.name}: kit_file {kf} not found under opt/")
        return hits[0]

    def missing(self, path, dotted):
        """None if `dotted` (name | Class.method[/method…]) is defined in the module at `path`, else the words of what is missing."""
        names, classes, _ = module_names(path)
        head, _, tail = dotted.partition(".")
        if head not in names:
            return f"{path} does not define `{head}` at module level"
        for meth in (tail.split("/") if tail else ()):
            if head not in classes:
                return f"{head} in {path} is not a class (wanted .{meth})"
            if meth not in classes[head]:
                return f"class {head} in {path} has no `{meth}`"
        return None

    def assert_defined(self, path, dotted, ctx):
        why = self.missing(path, dotted)
        self.assertIsNone(why, f"{ctx}: {why}")

    def test_registry_lines_resolve_to_defined_names(self):
        checked, problems, stale_seen = 0, [], set()
        for name, lever in registry.LEVERS.items():
            near = self.kit_file_path(lever)
            for item in (lever.lines or "").split(";"):
                item = item.strip()
                if not item:
                    continue
                fname, _, rest = item.partition(":")
                fname = fname.strip()
                if not fname.endswith(".py"):
                    continue
                path = self.resolve(fname, near)
                for piece in rest.split(","):
                    m = IDENT.match(piece)
                    if not m:                                       # prose ("W1 block", "the per-input loop"): the file's existence is the check
                        continue
                    dotted = m.group(1) + m.group(2)                 # name | Class.m1[/m2…]
                    head, _, meths = dotted.partition(".")
                    for d in ([head] if not meths else [f"{head}.{mm}" for mm in meths.split("/")]):
                        why = self.missing(path, d)
                        key = (name, d)
                        if why is None:
                            self.assertNotIn(key, KNOWN_STALE, f"{key} is defined again: remove it from KNOWN_STALE")
                        elif key in KNOWN_STALE:
                            stale_seen.add(key)
                        else:
                            problems.append(f"lever {name} lines `{item}`: {why}")
                    checked += 1
        self.assertEqual(problems, [], "\n".join(problems))
        self.assertEqual(stale_seen, set(KNOWN_STALE), f"KNOWN_STALE entries not met (remove them): {set(KNOWN_STALE) - stale_seen}")
        self.assertGreater(checked, 20, checked)

    def test_declared_SURFACE_tuples_hold(self):
        seen = 0
        for key, rels in self.idx.items():
            for rel in rels:
                if "/" in key:                                      # each file once (the basename key)
                    continue
                names, classes, surface = module_names(rel)
                if surface is None:
                    continue
                seen += 1
                for dotted in surface:
                    self.assert_defined(rel, dotted, f"{rel} SURFACE")
        self.assertGreaterEqual(seen, 4, "the SAMPLER modules declare SURFACE (chai1_eager/hoist.py, chai1_eager/stack.py, chai1_fastln/dit_attn.py, stackx.py)")
