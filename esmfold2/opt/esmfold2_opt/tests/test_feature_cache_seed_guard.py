"""The feature-cache seed guard (stack.guard_feature_cache): the kit's `fc` lever caches ESMFold2InputBuilder.prepare_input per (content,
device) — seed-blind — while upstream seeds the SMILES conformer generation, so an input with a SMILES-ligand chain must never be served
another seed's features. The guard routes such inputs to the builder class's own prepare_input (uncached, per seed), prints one named line
per input, counts, and leaves every other input on the kit's cache. Stand-ins here: a builder class whose prepare_input returns features
that carry the seed, and the kit's cache as the seed-blind instance wrapper install_feature_cache installs (keyed by content, first seed
wins) — no torch, no kit."""
import io
import types
import unittest
from unittest import mock

from esmfold2_opt import stack


class Chain:
    def __init__(self, id, sequence=None, smiles=None, ccd=None):
        self.id, self.sequence, self.smiles, self.ccd = id, sequence, smiles, ccd


class Input:
    def __init__(self, *chains):
        self.sequences = list(chains)


class Builder:
    """Upstream's builder: prepare_input featurizes per seed (the SMILES conformer is seeded)."""
    calls = 0

    def prepare_input(self, input, seed=None, device=None):
        Builder.calls += 1
        return {"seed": seed, "chains": [c.id for c in input.sequences]}, "chain_infos"


def install_kit_cache(builder):
    """The kit's install_feature_cache as it keys: (content, device), never the seed — the first seed's features serve every later seed."""
    orig = builder.prepare_input
    store = {}

    def prepare_input_cached(input, seed=None, device=None):
        key = (tuple((c.id, c.sequence, c.smiles, c.ccd) for c in input.sequences), str(device))
        if key not in store:
            store[key] = orig(input, seed=seed, device=device)
        return store[key]

    builder.prepare_input = prepare_input_cached
    builder._ef2opt_fc = True
    return store


class TestSeedGuard(unittest.TestCase):
    def setUp(self):
        Builder.calls = 0
        for k in stack.SEED_GUARD:
            stack.SEED_GUARD[k] = 0

    def test_smiles_input_is_featurized_per_seed_with_one_named_line(self):
        b = Builder(); install_kit_cache(b)
        self.assertTrue(stack.guard_feature_cache(b))
        self.assertFalse(stack.guard_feature_cache(b))                                  # idempotent per builder
        lig = Input(Chain("A", sequence="MKV"), Chain("L", smiles="CCO"))
        err = io.StringIO()
        with mock.patch("sys.stderr", err):
            feats = [b.prepare_input(lig, seed=s, device="cpu")[0]["seed"] for s in (0, 1, 2)]
        self.assertEqual(feats, [0, 1, 2])                                              # every seed its own features, none served from the cache
        self.assertEqual(Builder.calls, 3)
        lines = [l for l in err.getvalue().splitlines() if "feature_cache: bypassed for SMILES ligand chain(s) L" in l]
        self.assertEqual(len(lines), 1, err.getvalue())                                 # one named line per input
        self.assertTrue(lines[0].startswith("[esmfold2-opt] "), lines[0])
        self.assertEqual((stack.SEED_GUARD["installed"], stack.SEED_GUARD["smiles_inputs"], stack.SEED_GUARD["smiles_calls"]), (1, 1, 3))
        self.assertEqual(stack.smiles_chain_ids(lig), ["L"])

    def test_protein_and_ccd_inputs_keep_the_kit_cache(self):
        b = Builder(); store = install_kit_cache(b); stack.guard_feature_cache(b)
        prot = Input(Chain("A", sequence="MKV"), Chain("B", sequence="MKVL"))
        ccd = Input(Chain("A", sequence="MKV"), Chain("L", ccd="ATP"))               # a CCD ligand takes a dictionary conformer: no RNG, cached
        err = io.StringIO()
        with mock.patch("sys.stderr", err):
            self.assertEqual([b.prepare_input(prot, seed=s, device="cpu")[0]["seed"] for s in (0, 1)], [0, 0])   # the kit's cache: seed 0's features served
            self.assertEqual([b.prepare_input(ccd, seed=s, device="cpu")[0]["seed"] for s in (0, 1)], [0, 0])
        self.assertEqual(Builder.calls, 2)
        self.assertEqual(len(store), 2)
        self.assertEqual(err.getvalue(), "")
        self.assertEqual(stack.SEED_GUARD["smiles_inputs"], 0)
        self.assertEqual(stack.smiles_chain_ids(prot), []); self.assertEqual(stack.smiles_chain_ids(ccd), [])

    def test_no_kit_cache_or_no_builder_installs_nothing(self):
        self.assertFalse(stack.guard_feature_cache(None))
        b = Builder()                                                                   # no fc lever on this builder (no _ef2opt_fc)
        self.assertFalse(stack.guard_feature_cache(b))
        self.assertFalse(getattr(b, "_esmfold2_opt_seed_guard", False))
        self.assertEqual(stack.SEED_GUARD["installed"], 0)

    def test_status_carries_the_guard_record(self):
        self.assertEqual(set(stack.SEED_GUARD), {"installed", "smiles_inputs", "smiles_calls"})
        with mock.patch.object(stack, "_REPORT", {"active": True, "mode": "fast"}):
            self.assertEqual(stack.status()["seed_guard"], {"installed": 0, "smiles_inputs": 0, "smiles_calls": 0})


if __name__ == "__main__":
    unittest.main()
