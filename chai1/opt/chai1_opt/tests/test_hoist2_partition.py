"""hoist2 (chai1_eager/hoist.py, the value-taint hoister behind the `hoist2` DSTEP lever): its partition rules on hand-written traced-style
statements — CPU, no torch (the pure functions are compiled from the module's own source; the module itself imports torch).
GPU truth (torch.equal vs the base hoister / the un-hoisted call; byte-identical CIFs vs --mode off under --det 1) is the GPU-host acceptance,
not this file."""
import ast, os, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
HOIST_PY = os.path.normpath(os.path.join(HERE, "..", "..", "forward", "eager_trunk", "chai1_eager", "hoist.py"))
PURE = ("_names", "value_reads", "inplace_targets", "partition_value")


def _pure_namespace():
    src = open(HOIST_PY, encoding="utf-8").read()
    mod = ast.parse(src)
    keep = [n for n in mod.body if (isinstance(n, ast.FunctionDef) and n.name in PURE)
            or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("INPLACE_EXTRA", "META_TORCH", "META_PRIM", "HOISTERS") for t in n.targets))]
    ns = {"ast": ast}
    exec(compile(ast.Module(body=keep, type_ignores=[]), HOIST_PY, "exec"), ns)   # noqa: S102  the module's own pure functions
    return ns


def _stmts(src):
    return ast.parse(src).body


class TestHoist2Partition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _pure_namespace()

    def test_names_present(self):
        for n in PURE:
            self.assertIn(n, self.ns)
        self.assertEqual(self.ns["HOISTERS"], ("base", "hoist2"))

    def test_metadata_read_does_not_taint(self):
        body = _stmts(
            "n = int(torch.size(coords, 1))\n"          # metadata of a variant input: step-invariant under hoist2 (the base hoister taints it)
            "e = torch.expand(cond, [n, 4])\n"           # reads n only: invariant
            "d = torch.new_empty(coords, [n])\n"         # metadata only: invariant
            "y = torch.add(coords, e)\n"                 # value read of coords: per-step
            "z = torch.mul(y, sigma)\n")                 # derived: per-step
        is_var, info = self.ns["partition_value"](body, ("coords", "sigma"))
        self.assertEqual(is_var, [False, False, False, True, True]); self.assertEqual(info["demoted"], [])
        self.assertEqual(info["meta_only_reads"], {"coords": 2})
        vals, metas = self.ns["value_reads"](body[0])
        self.assertNotIn("coords", vals); self.assertEqual(metas, {"coords"})
        vals, metas = self.ns["value_reads"](_stmts("p = ops.prim.device(coords)")[0])
        self.assertEqual(metas, {"coords"})

    def test_inplace_on_a_tainted_name_taints_and_demotion(self):
        body = _stmts(
            "a = torch.zeros([4])\n"                     # 0 pre
            "b = torch.add(coords, 1)\n"                 # 1 step
            "c = torch.mul(a, 2)\n"                      # 2 pre (reads a before the in-place below)
            "r = torch.add(b, a)\n"                      # 3 step, value-reads a BEFORE statement 4 mutates it
            "_0 = torch.add_(a, 5)\n"                    # 4 in-place on a: precompute-only by taint, but a per-step statement before it read a -> DEMOTED to per-step
            "s = torch.mul(a, 3)\n")                     # 5 reads a after a per-step in-place -> per-step
        is_var, info = self.ns["partition_value"](body, ("coords", "sigma"))
        self.assertEqual(info["demoted"], [4]); self.assertEqual(is_var, [False, True, False, True, True, True])
        self.assertEqual(self.ns["inplace_targets"](body[4]), {"a"})
        self.assertEqual(self.ns["inplace_targets"](_stmts("o = torch.matmul(x, w, out=buf)")[0]), {"buf"})
        self.assertEqual(self.ns["inplace_targets"](_stmts("o = torch.masked_fill_(t, m, 0.)")[0]), {"t"})

    def test_no_step_reader_before_an_invariant_inplace_keeps_it_hoisted(self):
        body = _stmts(
            "a = torch.zeros([4])\n"
            "_0 = torch.add_(a, 5)\n"                    # in-place on an invariant name with no per-step reader before it: stays precompute-only
            "y = torch.add(coords, a)\n")
        is_var, info = self.ns["partition_value"](body, ("coords", "sigma"))
        self.assertEqual(is_var, [False, False, True]); self.assertEqual(info["demoted"], [])


if __name__ == "__main__":
    unittest.main()
