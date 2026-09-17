"""Every bundled example input (tests/inputs/*.a3m) parses with STOCK colabfold's own a3m functions, taken verbatim from the pinned wheel
(stock/colabfold-1.6.1-py3-none-any.whl: colabfold/input.py `parse_fasta` + `get_queries`, colabfold/batch.py `normalize_a3m` +
`unserialize_msa`; alphafold's template mock is stubbed — it is not part of parsing): one query per file, the header's chain lengths and
cardinalities, one paired row per chain, and at least one UNPAIRED row per chain — a paired+unpaired complex a3m without its unpaired
query rows leaves stock with an empty per-chain MSA and the prediction fails downstream."""
import ast
import glob
import os
import re
import typing
import unittest
import zipfile
from pathlib import Path

TREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
STOCK_WHEEL = os.path.join(TREE, "stock", "colabfold-1.6.1-py3-none-any.whl")
INPUTS = os.path.join(TREE, "tests", "inputs")
EXPECTED = {"1BRS_AD": ([110, 89], [1, 1]), "1A3N_hemoglobin": ([141, 146], [2, 2])}   # file stem -> (chain lengths, cardinalities)


def stock_functions():
    """The four stock functions, compiled from the wheel's source text (no colabfold import: its package pulls the GPU stack)."""
    z = zipfile.ZipFile(STOCK_WHEEL)
    ns = {n: getattr(typing, n) for n in ("List", "Optional", "Tuple", "Union", "Dict", "Any")}
    ns.update(Path=Path, re=re, logging=__import__("logging"), random=__import__("random"), MolType=object,
              mk_mock_template=lambda q: {"stub_for": len(q)})                     # alphafold's template mock: featurisation, not parsing
    for member, names in (("colabfold/input.py", ("parse_fasta", "get_queries")), ("colabfold/batch.py", ("normalize_a3m", "unserialize_msa"))):
        src = z.read(member).decode("utf-8"); tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                exec(compile(ast.Module(body=[node], type_ignores=[]), member, "exec"), ns)
        for n in names:
            assert n in ns, f"{member} lacks {n}"
    return ns


class TestBundledInputs(unittest.TestCase):
    def test_every_bundled_a3m_parses_with_stock(self):
        F = stock_functions()
        files = sorted(glob.glob(os.path.join(INPUTS, "*.a3m")))
        self.assertEqual(sorted(Path(f).stem for f in files), sorted(EXPECTED))            # every shipped input is accounted for here
        queries, is_complex = F["get_queries"](INPUTS)
        self.assertEqual(len(queries), len(files)); self.assertTrue(is_complex)
        for name, query_sequence, a3m_lines, *_ in queries:
            with self.subTest(input=name):
                lens, cards = EXPECTED[name]
                unpaired, paired, seqs, cardinality, templates = F["unserialize_msa"](a3m_lines, query_sequence)
                self.assertEqual([len(s) for s in seqs], lens); self.assertEqual(cardinality, cards)
                self.assertEqual(len(query_sequence), sum(lens))                                # the paired query row is every chain once
                self.assertEqual([p.count(">") for p in paired], [1] * len(lens))               # one paired row per chain
                self.assertTrue(all(u.count(">") >= 1 for u in unpaired), unpaired)              # the unpaired query row per chain stock needs
                for u, s in zip(unpaired, seqs):
                    self.assertIn(s, u.replace("-", ""))                                        # each chain's unpaired row carries that chain
                self.assertEqual(len(templates), len(lens))


if __name__ == "__main__":
    unittest.main()
