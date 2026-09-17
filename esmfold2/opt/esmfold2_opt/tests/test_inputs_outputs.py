"""inputs.py: the kit's items convert with the driver's chain conventions and upstream inputs pass through; outputs.py: the one schema
is reproducible byte for byte (sorted keys, no volatile field, savez_compressed stable across time)."""
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import unittest

from esmfold2_opt import inputs, outputs
from esmfold2_opt.tests import _stubs


class TestInputs(unittest.TestCase):
    def test_upstream_schema_roleless_heteromer_and_monomer(self):
        """Upstream's own input schema, chains as given, no roles: a two-protein complex, a monomer, a protein+ligand input and the kit's
        public slice all normalise unchanged (ids kept, order kept, every chain's msa as named); an extra top-level key is ignored; a dict
        that is not an upstream input ({'sequences': [...]}) is not an input."""
        het = {"id": "het", "sequences": [{"type": "protein", "id": "A", "sequence": "MKVLAT", "msa": None}, {"type": "protein", "id": "D", "sequence": "GGSAAW", "msa": "D.a3m"}], "ligand_chains": []}
        mono = {"sequences": [{"type": "protein", "id": "A", "sequence": "MKVL", "msa": None}]}
        lig = {"id": "l", "sequences": [{"type": "protein", "id": "A", "sequence": "MKVL", "msa": None}, {"type": "ligand", "id": "LIG1", "ccd": ["ATP"]}]}
        out = inputs.normalise([het, mono, lig], base_dir="/base")
        self.assertEqual([e["id"] for e in out], ["het", "item001", "l"])
        self.assertEqual([[c["id"] for c in e["sequences"]] for e in out], [["A", "D"], ["A"], ["A", "LIG1"]])
        self.assertEqual(out[0]["sequences"][1]["msa"], "/base/D.a3m"); self.assertIsNone(out[0]["sequences"][0]["msa"])
        self.assertEqual(out[0]["ligand_chains"], [])                                           # carried, never read
        self.assertEqual(inputs.normalise(mono)[0]["id"], inputs.DEFAULT_ID)
        for bad in ({"complex_id": "k", "binder_seqs": ["MK"], "target_seqs": ["MKV"]}, [{"complex_id": "k", "target_seqs": ["MKV"]}], {"chains": []}):
            with self.assertRaises(inputs.InputError):
                inputs.normalise(bad)
        kit = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(inputs.__file__)))), "opt", "forward", "fast_inference")
        items_path = os.path.join(kit, "tests", "w4_public_slice.json")
        if os.path.isfile(items_path):
            sl = inputs.load(items_path)
            self.assertEqual([e["id"] for e in sl][:1], ["1BRS_barstar1_barnase1"]); self.assertTrue(all(c["msa"] is None for e in sl for c in e["sequences"]))

    def test_one_sample_count_per_run(self):
        """inputs.run_samples: --num_diffusion_samples when given (an input naming another count refused by name), else the inputs' common
        num_diffusion_samples key (two counts, or a named beside an unnamed input, refused), else the default; item_samples = an input's own
        key else the run's count; counts below 1 refused."""
        a1 = {"id": "a", "sequences": [], "num_diffusion_samples": 1}; b5 = {"id": "b", "sequences": [], "num_diffusion_samples": 5}
        c5 = {"id": "c", "sequences": [], "num_diffusion_samples": 5}; plain = {"id": "p", "sequences": []}
        self.assertEqual(inputs.run_samples([plain, plain], flag=None, default=1), (1, "default"))
        self.assertEqual(inputs.run_samples([plain, plain]), (None, "default"))                          # the library's: not known here, resolved with the settings
        self.assertEqual(inputs.run_samples([plain], flag=5, default=1), (5, "flag"))
        self.assertEqual(inputs.run_samples([b5, c5], flag=5, default=1), (5, "flag"))                  # the key agreeing with the flag is no disagreement
        self.assertEqual(inputs.run_samples([b5, c5], flag=None, default=1), (5, "inputs"))
        with self.assertRaises(inputs.InputError) as cm:                                                # an input against the flag: named (id, its count, the flag's)
            inputs.run_samples([b5, a1], flag=5, default=1)
        self.assertIn("'a'", str(cm.exception)); self.assertIn("num_diffusion_samples=1", str(cm.exception)); self.assertIn("--num_diffusion_samples 5", str(cm.exception))
        with self.assertRaises(inputs.InputError) as cm:                                                # two inputs against each other
            inputs.run_samples([a1, b5], flag=None, default=1)
        self.assertIn("'a'", str(cm.exception)); self.assertIn("'b'", str(cm.exception))
        with self.assertRaises(inputs.InputError) as cm:                                                # a named beside an unnamed input
            inputs.run_samples([b5, plain], flag=None, default=1)
        self.assertIn("'p'", str(cm.exception)); self.assertIn("names none", str(cm.exception))
        with self.assertRaises(inputs.InputError):
            inputs.run_samples([plain], flag=0, default=1)
        with self.assertRaises(inputs.InputError):
            inputs.run_samples([{"id": "z", "sequences": [], "num_diffusion_samples": 0}], flag=None, default=1)
        self.assertEqual(inputs.item_samples(b5, 1), 5); self.assertEqual(inputs.item_samples(plain, 3), 3)
        self.assertIn(inputs.SAMPLES_KEY, inputs.PACKAGE_KEYS)                                          # the key never reaches upstream's deserializer

    def test_normalise_forms(self):
        one = {"sequences": [{"type": "protein", "id": "A", "sequence": "MKV"}]}
        self.assertEqual(inputs.normalise(one)[0]["id"], inputs.DEFAULT_ID)
        many = [dict(one, id="x"), dict(one, id="y")]
        self.assertEqual([i["id"] for i in inputs.normalise(many)], ["x", "y"])
        with self.assertRaises(inputs.InputError):
            inputs.normalise([dict(one, id="x"), dict(one, id="x")])
        with self.assertRaises(inputs.InputError):
            inputs.normalise([one, {"complex_id": "k", "target_seqs": ["MKV"]}])           # not an upstream input
        rel = inputs.normalise({"sequences": [{"type": "protein", "id": "A", "sequence": "MKV", "msa": "m.a3m"}]}, base_dir="/tmp/x")
        self.assertEqual(rel[0]["sequences"][0]["msa"], "/tmp/x/m.a3m")

    def test_build_spi_with_stub_upstream(self):
        tmp = tempfile.mkdtemp()
        try:
            site = _stubs.write_stub_upstream(tmp)
            _stubs.install_stub_modules(site)
            a3m = os.path.join(tmp, "B.a3m")
            open(a3m, "w").write(">q\nMKVL\n>h1\nMKaVL\n>h2\nM-VL\n")
            item = inputs.normalise({"sequences": [{"type": "protein", "id": "A", "sequence": "MK"},
                                                   {"type": "protein", "id": "B", "sequence": "MKVL", "msa": a3m},
                                                   {"type": "ligand", "id": "L1", "ccd": ["ATP"]}]})[0]
            spi = inputs.build_spi(item, use_msa=True, msa_read_depth=2)
            self.assertEqual(inputs.msa_depths(spi), [0, 2, 0])                       # read to depth 2, insertions removed
            spi2 = inputs.build_spi(item, use_msa=False)
            self.assertEqual(inputs.msa_depths(spi2), [0, 0, 0])                      # the variant decides: no MSA on fast / full_nomsa
            bad = inputs.normalise({"sequences": [{"type": "protein", "id": "B", "sequence": "MKVX", "msa": a3m}]})[0]
            with self.assertRaises(inputs.InputError):
                inputs.build_spi(bad, use_msa=True)
        finally:
            _stubs.forget_stub_modules(site)
            shutil.rmtree(tmp, ignore_errors=True)


class _Arr:
    def __init__(self, a): self._a = a
    def numpy(self): return self._a
    def detach(self): return self
    def cpu(self): return self


class _FakeResult:
    """What upstream's result looks like to the kit's writer: pae / plddt / pair_chains_iptm tensors, the complex's mmCIF, the ptm / iptm scalars."""
    def __init__(self, seed, L=8):
        import numpy as np
        rng = np.random.default_rng(seed)
        self.plddt = _Arr(rng.random(L, dtype=np.float32)); self.pae = _Arr(rng.random((L, L), dtype=np.float32) * 30)
        self.ptm = 0.5; self.iptm = 0.25; self.pair_chains_iptm = _Arr(rng.random((2, 2), dtype=np.float32)); self.pde = None

        class _C:
            def to_mmcif(_self):
                return f"data_pred\n# seed={seed}\n"
        self.complex = _C()


class _Feats:
    """Token features as upstream's prepare_input returns them (asym_id / mol_type / token_attention_mask), 4 + 4 tokens, two chains."""
    def __init__(self):
        import numpy as np
        self.d = {"asym_id": np.array([[1] * 4 + [2] * 4]), "mol_type": np.zeros((1, 8), dtype=np.int64), "token_attention_mask": np.ones((1, 8), dtype=np.int64)}

    def __getitem__(self, k):
        return _Arr2(self.d[k])


class _Arr2(_Arr):
    def __getitem__(self, i): return _Arr2(self._a[i])
    def sum(self): return _Arr2(self._a.sum())
    def item(self): return self._a.item()


class TestOutputs(unittest.TestCase):
    """The file set through the kit's own writer (outputs.write_rows = ef2_server.write_outputs), and the stock arm's staging round trip."""

    @classmethod
    def setUpClass(cls):
        cls.kit = _stubs.require_kit()
        try:
            import numpy  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("numpy not installed")
        cls.tmp = tempfile.mkdtemp()
        cls.site = _stubs.write_stub_upstream(cls.tmp, _stubs.pins_or_none())
        cls.stub_kit = _stubs.write_stub_kit(cls.tmp, cls.kit)
        sys.path.insert(0, cls.site); _stubs.install_stub_modules(cls.site)
        cls.server = outputs.kit_server(cls.stub_kit)
        cls.item = inputs.normalise({"id": "c1", "sequences": [{"type": "protein", "id": "A", "sequence": "MKVL"}, {"type": "protein", "id": "B", "sequence": "AAAA"}]})[0]

    @classmethod
    def tearDownClass(cls):
        for m in [m for m in sys.modules if m in ("ef2_server", "run_ef2_om")]:
            sys.modules.pop(m, None)
        _stubs.forget_stub_modules(cls.site)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_kit_writer_file_set_and_rows(self):
        d = tempfile.mkdtemp()
        try:
            meta = outputs.spi_meta([0, 0], _Feats())
            self.assertEqual((meta[0], meta[3]), ([0, 0], 8))
            rows = outputs.write_rows(self.server, d, "pred", self.item, "full_nomsa", 3, [_FakeResult(0), _FakeResult(1)], meta, 1.5, "stub-gpu")
            self.assertEqual([r["pred_file"] for r in rows], ["c1__full__s3_x0.cif", "c1__full__s3_x1.cif"])   # the driver's stem: the server's variant name
            self.assertEqual(sorted(os.listdir(os.path.join(d, "cif_all"))), ["c1__full__s3_x0.cif", "c1__full__s3_x0_pae.npz", "c1__full__s3_x1.cif", "c1__full__s3_x1_pae.npz"])
            self.assertEqual([json.loads(l)["sample_index"] for l in open(os.path.join(d, "pred_rows.jsonl"))], [0, 1])
            r = rows[0]
            self.assertEqual(set(r), {"complex_id", "variant", "seed", "sample_index", "status", *self.server.ROW_RESULT_FIELDS, "n_tokens", "msa_depths", *outputs.META_KEYS})   # upstream's scalars + bookkeeping: nothing derived
            self.assertEqual((r["complex_id"], r["variant"], r["seed"], r["n_tokens"], r["iptm"], r["ptm"], r["msa_depths"], r["gpu"]), ("c1", "full", 3, 8, 0.25, 0.5, [0, 0], "stub-gpu"))
            self.assertTrue(set(outputs.RESULT_ATTRS) >= set(self.server.ROW_RESULT_FIELDS) | set(self.server.NPZ_RESULT_FIELDS))   # the stock arm stages every attribute the writer reads
            import numpy as np
            z = np.load(os.path.join(d, "cif_all", "c1__full__s3_x0_pae.npz"))
            self.assertEqual(set(z.files), {"pae", "plddt", "asym_id", "mol_type", "pair_chains_iptm"})
            self.assertEqual((z["pae"].dtype, z["plddt"].dtype, z["pair_chains_iptm"].dtype), (np.float16, np.float16, np.float32))
            self.assertTrue(np.array_equal(z["pair_chains_iptm"], _FakeResult(0).pair_chains_iptm.numpy()))
            self.assertEqual(outputs.written_files(d, rows)[0], os.path.join(d, "cif_all", "c1__full__s3_x0.cif"))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_staging_round_trip_equals_the_direct_write(self):
        """The stock arm's path (stage in the subprocess, finalise in the pred process) writes the same files and rows as the direct call."""
        d1, d2 = tempfile.mkdtemp(), tempfile.mkdtemp()
        try:
            meta = outputs.spi_meta([0, 0], _Feats())
            res = [_FakeResult(0), _FakeResult(1)]
            direct = outputs.write_rows(self.server, d1, "pred", self.item, "fast", 7, res, meta, 2.0, "g")
            stage = os.path.join(d2, "staged")
            stems = outputs.stage_result(stage, self.item, "fast", 7, res, [0, 0], _Feats(), 2.0, "g")
            self.assertEqual(stems, ["c1__fast__s7_x0", "c1__fast__s7_x1"])
            staged = outputs.finalise_staged(self.server, stage, d2, "pred")
            self.assertFalse(os.path.exists(stage))
            self.assertEqual(sorted(os.listdir(os.path.join(d1, "cif_all"))), sorted(os.listdir(os.path.join(d2, "cif_all"))))
            for fn in os.listdir(os.path.join(d1, "cif_all")):
                if fn.endswith(".cif"):
                    self.assertEqual(open(os.path.join(d1, "cif_all", fn)).read(), open(os.path.join(d2, "cif_all", fn)).read())
            import numpy as np
            for fn in os.listdir(os.path.join(d1, "cif_all")):
                if fn.endswith(".npz"):
                    self.assertEqual(open(os.path.join(d1, "cif_all", fn), "rb").read(), open(os.path.join(d2, "cif_all", fn), "rb").read(), fn)   # byte-identical run to run
                    a, b = np.load(os.path.join(d1, "cif_all", fn)), np.load(os.path.join(d2, "cif_all", fn))
                    self.assertEqual(set(a.files), set(b.files))
                    for k in a.files:
                        self.assertTrue(np.array_equal(a[k], b[k]), f"{fn}:{k}"); self.assertEqual(a[k].dtype, b[k].dtype, f"{fn}:{k}")
            skip = set(outputs.META_KEYS)
            self.assertEqual([{k: v for k, v in r.items() if k not in skip} for r in direct], [{k: v for k, v in r.items() if k not in skip} for r in staged])
            self.assertEqual(outputs.listing(d1), outputs.listing(d2))                  # the listing: equal
        finally:
            shutil.rmtree(d1, ignore_errors=True); shutil.rmtree(d2, ignore_errors=True)

    def test_a_single_chain_input_is_written_like_any_other(self):
        """One protein chain (no second chain, no roles of any kind) is a valid input end to end: normalised, folded elsewhere, written here."""
        item = inputs.normalise({"id": "mono", "sequences": [{"type": "protein", "id": "A", "sequence": "MKVLAGWA"}]})[0]
        self.assertEqual(set(item), {"id", "sequences"})
        d = tempfile.mkdtemp()
        try:
            one = _FakeResult(5); one.pair_chains_iptm = None; one.iptm = None                # what upstream returns for one chain: no chain pair
            rows = outputs.write_rows(self.server, d, "pred", item, "fast", 0, one, outputs.spi_meta([0], _Feats()), 1.0, "g")
            self.assertEqual((len(rows), rows[0]["complex_id"], rows[0]["iptm"], rows[0]["status"]), (1, "mono", None, "ok"))
            import numpy as np
            self.assertEqual(set(np.load(os.path.join(d, "cif_all", "mono__fast__s0_x0_pae.npz")).files), {"pae", "plddt", "asym_id", "mol_type"})
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
