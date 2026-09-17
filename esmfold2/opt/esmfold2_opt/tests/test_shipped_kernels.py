"""No kit module ships a CUDA cubin in this tree: the pair TriMul and the transitions are the shared core's provider rows under the
modes' tier words.  CPU tests: the (empty) shipped set checks clean through ef2_nvjit, the manifest carries no cubin entry, and the
trimul group is the provider word (lever tx), individually switchable."""
import os
import sys
import unittest

KIT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "forward", "fast_inference")
DRIVER_MODULES = ("ef2_nvjit",)
J = None


def setUpModule():
    global J
    sys.path.insert(0, os.path.join(KIT, "driver"))
    import ef2_nvjit as J            # noqa: E402


def tearDownModule():
    for name in DRIVER_MODULES:
        sys.modules.pop(name, None)
    try:
        sys.path.remove(os.path.join(KIT, "driver"))
    except ValueError:
        pass


class ShippedSet(unittest.TestCase):
    def test_no_kit_cubin_ships_and_the_empty_set_checks_clean(self):
        self.assertEqual(J.PREBUILD_MODULES, ())
        self.assertEqual(J.kernel_specs(), [])
        self.assertEqual(J.check_shipped(), [])                                                          # the `cubins` step of run.sh install: an empty set, no defect
        man = J.shipped_manifest()
        self.assertEqual(man.get("cubins"), {})
        for gone in ("ef2_trimul_v6_k3.cubin", "ef2_transition_cute.cubin"):
            self.assertFalse(os.path.exists(os.path.join(J.shipped_dir(), gone)), gone)


class TrimulGroupIsTheProviderWord(unittest.TestCase):
    def test_tx_is_the_trimul_group_and_individually_switchable(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("ef2_server_under_test", os.path.join(KIT, "driver", "ef2_server.py"))
        srv = importlib.util.module_from_spec(spec); spec.loader.exec_module(srv)
        base = {"t15", "t15msa", "t10"}
        self.assertEqual(srv.check_composition("opt14_msa", off=base)["trimul"], ["tx"])
        self.assertEqual(srv.check_composition("opt7x")["trimul"], [])                                    # the exact set's pair TriMul is the upstream fused statement itself
        self.assertEqual(srv.check_composition("opt14_msa", off=base | {"tx"})["trimul"], [])           # tx is individually switchable
        self.assertEqual(srv.TRIMUL_TIER, {"opt7x": "exact", "opt14_msa": "fast"})
        for gone in ("t9", "sigmoid", "incnt", "formtab", "k3cute", "lnfold"):                          # TriMul words the kit does not have: the provider's rows serve
            self.assertNotIn(gone, srv.TRIMUL_LEVERS)
        env = dict(os.environ)
        try:
            os.environ.pop(srv.TRIMUL_TIER_ENV, None); os.environ.pop(srv.PACKAGE_MODE_ENV, None)
            self.assertEqual(srv.trimul_tier("opt7x"), "exact"); self.assertEqual(srv.trimul_tier("opt14_msa"), "fast")
            os.environ[srv.PACKAGE_MODE_ENV] = "big"; self.assertEqual(srv.trimul_tier("opt14_msa"), "big")
            os.environ[srv.PACKAGE_MODE_ENV] = "fast"; self.assertEqual(srv.trimul_tier("opt14_msa"), "fast")
            os.environ[srv.TRIMUL_TIER_ENV] = "native"; self.assertEqual(srv.trimul_tier("opt14_msa"), "native")   # an A/B by row word
        finally:
            os.environ.clear(); os.environ.update(env)


if __name__ == "__main__":
    unittest.main()
