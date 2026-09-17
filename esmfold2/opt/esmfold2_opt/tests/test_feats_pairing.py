"""ef2_feats (lever `fz`) — the vectorised taxonomy pairing (`construct_paired_msa_vec` / `_pairing_tables`) reproduces upstream's
`paired_msa.construct_paired_msa` row table exactly.

Two layers: (1) CPU-only hand cases of `_pairing_tables` whose expected tables were derived by hand from upstream's rules (group order =
distinct-chain count descending then first appearance; member chains cycle their group rows; non-member chains draw their unpaired rows
FIFO or -1; the trailing unpaired block; the max_pairs / max_total / max_seqs cuts) — these run everywhere; (2) a randomized differential
test against the installed upstream function itself (skipped where `esm` is not importable: it runs in the kit's environment), covering
None / empty / shared MSAs, `key=-1` and key-less headers, single-chain duplicate taxa, small caps, chains without tokens, modified-residue
tokens past the query length, gapped / 1-based / unordered residue ids.
"""
import os, random, sys, unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.normpath(os.path.join(HERE, "..", "..", "forward", "fast_inference", "driver"))
FZ = None


def setUpModule():
    global FZ
    import importlib.util
    spec = importlib.util.spec_from_file_location("ef2_feats", os.path.join(DRIVER, "ef2_feats.py"))
    FZ = importlib.util.module_from_spec(spec)
    sys.modules["ef2_feats"] = FZ
    spec.loader.exec_module(FZ)


def tearDownModule():
    sys.modules.pop("ef2_feats", None)


def _tables(chain_taxa, **caps):
    taxs = [np.asarray([-1 if t is None else t for t in taxa], dtype=np.int64) for taxa in chain_taxa]
    kw = dict(max_pairs=8192, max_total=16384, max_seqs=16384); kw.update(caps)
    M, pair, flag = FZ._pairing_tables(taxs, kw["max_pairs"], kw["max_total"], kw["max_seqs"])
    return M, [p.tolist() for p in pair], [f.astype(int).tolist() for f in flag]


class TestPairingTables(unittest.TestCase):
    def test_two_chains_groups_leftovers(self):
        # chain 0 rows: query, key=5, key=5, key=9, key=-1, key=3 ; chain 1 rows: query, key=9, key=5, (no key), key=7, key=7
        # groups kept: 5 (3 entries, 2 chains), 9 (2 entries, 2 chains), 7 (2 entries, chain 1 only); 3 is a singleton -> its row is unpaired
        M, pair, flag = _tables([[None, 5, 5, 9, -1, 3], [None, 9, 5, None, 7, 7]])
        self.assertEqual(M, 7)
        self.assertEqual(pair[0], [0, 1, 2, 3, 4, 5, -1]); self.assertEqual(flag[0], [1, 1, 1, 1, 0, 0, 0])
        self.assertEqual(pair[1], [0, 2, 2, 1, 4, 5, 3]);  self.assertEqual(flag[1], [1, 1, 1, 1, 1, 1, 0])

    def test_three_chains_order_by_distinct_chains(self):
        # taxon 6 spans 3 chains -> first; then 4 (chains 0, 2), then 2 (chains 1, 2); the query row's own key (1) is never grouped
        M, pair, flag = _tables([[1, 4, 4, 6], [None, 6, 2], [None, 6, 4, 2, 2]])
        self.assertEqual(M, 6)
        self.assertEqual(pair[0], [0, 3, 1, 2, -1, -1]); self.assertEqual(flag[0], [1, 1, 1, 1, 0, 0])
        self.assertEqual(pair[1], [0, 1, -1, -1, 2, 2]); self.assertEqual(flag[1], [1, 1, 0, 0, 1, 1])
        self.assertEqual(pair[2], [0, 1, 2, 2, 3, 4]);   self.assertEqual(flag[2], [1, 1, 1, 1, 1, 1])

    def test_caps(self):
        M, pair, flag = _tables([[None, 5, 5, 9, -1, 3], [None, 9, 5, None, 7, 7]], max_pairs=3, max_total=5, max_seqs=4)
        self.assertEqual(M, 4)
        self.assertEqual(pair[0], [0, 1, 2, 4]); self.assertEqual(flag[0], [1, 1, 1, 0])
        self.assertEqual(pair[1], [0, 2, 2, 3]); self.assertEqual(flag[1], [1, 1, 1, 0])

    def test_dummy_chain(self):
        # a chain without an MSA is one query row with taxonomy -1: gap rows everywhere below the query
        M, pair, flag = _tables([[None, 3, 3], [None]])
        self.assertEqual(M, 3)
        self.assertEqual(pair[0], [0, 1, 2]);   self.assertEqual(flag[0], [1, 1, 1])
        self.assertEqual(pair[1], [0, -1, -1]); self.assertEqual(flag[1], [1, 0, 0])

    def test_no_groups_all_unpaired(self):
        M, pair, flag = _tables([[None, -1, None], [None, 8]])
        self.assertEqual(M, 3)
        self.assertEqual(pair[0], [0, 1, 2]);  self.assertEqual(flag[0], [1, 0, 0])
        self.assertEqual(pair[1], [0, 1, -1]); self.assertEqual(flag[1], [1, 0, 0])

    def test_max_seqs_zero_and_tiny_max_pairs(self):
        M, pair, _ = _tables([[None, 5, 5], [None, 5]], max_seqs=0)
        self.assertEqual(M, 0); self.assertEqual(pair[0], [])
        M, pair, flag = _tables([[None, 5, 5], [None, 5]], max_pairs=1)     # the first group row is appended before the cap is tested
        self.assertEqual(pair[0], [0, 1]); self.assertEqual(pair[1], [0, 1]); self.assertEqual(flag[0], [1, 1])


class TestAgainstUpstream(unittest.TestCase):
    """Randomized differential test against the installed upstream function (skipped where `esm` is absent; nothing of upstream is
    imported before this class runs; the imported upstream modules stay in sys.modules as any import would — removing them behind
    modules that hold references breaks later tests)."""
    AA = "ACDEFGHIKLMNPQRSTVWY-X"

    @classmethod
    def setUpClass(cls):
        try:
            import esm.models.esmfold2.paired_msa as P
            from esm.utils.msa.msa import MSA
        except Exception as e:             # noqa: BLE001 — not installed here (the kit environment has it)
            raise unittest.SkipTest(f"esm (upstream ESMFold2) not importable here ({type(e).__name__}): the differential test runs in the kit image")
        cls.P, cls.MSA = P, MSA
        ref = cls.P.construct_paired_msa                          # captured BEFORE any enable() of this process could patch it
        if getattr(ref, "__module__", "") == "ef2_feats":          # another test enabled the lever: the original is what it saved
            ref = sys.modules["ef2_feats"]._ORIG["pair"]
        cls.ref = staticmethod(ref)                               # a plain function on the class would bind `self` on lookup
        cls.L2R = cls.P.protein_letter_to_res_type()
        cls.Entry = getattr(sys.modules[cls.MSA.__module__], "FastaEntry")   # MSA(entries) holds FastaEntry(header, sequence) rows (esm.utils.msa.msa)

    def _msa(self, rng, L, depth, pool, p_nokey, p_neg):
        E = self.Entry
        rows = [E("query" + (" key=%d" % rng.choice(pool) if rng.random() < 0.5 else ""), "".join(rng.choice(self.AA[:20]) for _ in range(L)))]
        for i in range(1, depth):
            s = []
            for _ in range(L):
                if rng.random() < 0.05:
                    s.append("".join(rng.choice("acgt") for _ in range(rng.randint(1, 3))))
                s.append(rng.choice(self.AA))
            r = rng.random()
            h = "s%d" % i if r < p_nokey else ("s%d key=-1" % i if r < p_nokey + p_neg else "tr|s%d key=%d OS=x" % (i, rng.choice(pool)))
            rows.append(E(h, "".join(s)))
        return self.MSA(rows)

    def _case(self, rng, small_caps):
        nC = rng.randint(1, 6); pool = list(range(1, rng.choice([3, 8, 30, 200])))
        cm, qr, asym, res, shared = {}, {}, [], [], None
        for c in range(nC):
            L = rng.randint(1, 25); k = rng.random()
            if k < 0.15:
                m = None
            elif k < 0.3 and shared is not None and shared[0] == L:
                m = shared[1]
            else:
                m = self._msa(rng, L, rng.randint(1, rng.choice([2, 5, 40, 120])), pool, rng.choice([0.0, 0.2, 0.6]), rng.choice([0.0, 0.2])); shared = (L, m)
            cm[c] = m; qr[c] = np.array([rng.randrange(0, 21) for _ in range(L)], dtype=np.int64)
            mode = rng.random()
            if mode < 0.55:   idxs = list(range(L))
            elif mode < 0.65: idxs = []
            elif mode < 0.75: idxs = list(range(L + rng.randint(1, 3)))
            elif mode < 0.85: idxs = sorted(rng.sample(range(L), rng.randint(1, L))) if L > 1 else [0]
            elif mode < 0.93: idxs = [i + 1 for i in range(L)]
            else:             idxs = [rng.randrange(0, L + 2) for _ in range(rng.randint(1, L + 2))]
            asym += [c] * len(idxs); res += idxs
        caps = dict(max_pairs=rng.choice([1, 2, 3, 5, 9, 8192]), max_total=rng.choice([2, 4, 7, 12, 30, 16384]), max_seqs=rng.choice([0, 1, 3, 6, 25, 16384])) if small_caps else {}
        return cm, qr, np.array(asym, dtype=np.int64), np.array(res, dtype=np.int64), caps

    def test_randomized_identical_arrays(self):
        FZ._ORIG.setdefault("pair", self.ref)
        FZ._ORIG.setdefault("fn", self.P.msa_to_res_type_and_deletions if getattr(self.P.msa_to_res_type_and_deletions, "__module__", "") != "ef2_feats" else FZ._ORIG.get("fn"))
        FZ._PAIR["checked"] = True; FZ._PAIR["off"] = False
        n_cases = int(os.environ.get("KIT_TEST_PAIRING_CASES", "400"))
        for seed, small in ((1, False), (2, True)):
            rng = random.Random(seed)
            for i in range(n_cases):
                cm, qr, ta, tr, caps = self._case(rng, small)
                ref = self.ref(cm, qr, ta, tr, self.L2R, **caps)
                got = FZ.construct_paired_msa_vec(cm, qr, ta, tr, self.L2R, **caps)
                for name, a, b in zip(("msa_residues", "deletion_value", "is_paired"), got, ref):
                    self.assertEqual((a.dtype, a.shape), (b.dtype, b.shape), f"case {seed}/{i} {name} caps={caps}")
                    self.assertTrue(np.array_equal(a, b), f"case {seed}/{i} {name} differs caps={caps}")


if __name__ == "__main__":
    unittest.main()
