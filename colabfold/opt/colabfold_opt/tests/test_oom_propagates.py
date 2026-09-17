"""Out-of-memory is never rerouted (opt_core.oom.is_oom asked first in every broad handler on the served path): a device
out-of-memory raised inside a lever's installation propagates out of `enable` — no NOT ACTIVE record, no stock run — and out of the
deployment lever's placement; the non-OOM routes of the same handlers keep their named words. The out-of-memory is a stand-in class
named like jaxlib's (`XlaRuntimeError`, a RuntimeError) carrying XLA's text `RESOURCE_EXHAUSTED: Out of memory`. No jax, no GPU."""
import io
import unittest
from contextlib import redirect_stderr
from unittest import mock

import colabfold_opt
from colabfold_opt import stack, trimul_pallas, xla_cache
from colabfold_opt.tests import _stubs
from colabfold_opt.tests.test_activation import Base


class XlaRuntimeError(RuntimeError):
    """Named like jaxlib.xla_extension.XlaRuntimeError; the message is XLA's allocator text."""


OOM = XlaRuntimeError("RESOURCE_EXHAUSTED: Out of memory while trying to allocate 77.50GiB.")


class TestOomPropagates(Base):
    def test_oom_in_a_lever_install_propagates_out_of_enable(self):
        with mock.patch.object(trimul_pallas, "enable", side_effect=OOM), redirect_stderr(io.StringIO()) as err:
            with self.assertRaises(XlaRuntimeError):
                colabfold_opt.enable("fast", queries=_stubs.queries(600))
        self.assertNotIn("NOT ACTIVE", err.getvalue())                               # no refusal record was printed on the way out: the error is the answer

    def test_non_oom_install_error_keeps_its_named_route(self):
        with mock.patch.object(trimul_pallas, "enable", side_effect=ValueError("kernel module unreadable")), redirect_stderr(io.StringIO()):
            rep = colabfold_opt.enable("fast", queries=_stubs.queries(600), strict=False)
        self.assertFalse(rep["active"]); self.assertTrue(rep["reason"].startswith("enable() failed: TRIMUL_PALLAS: ValueError("), rep["reason"])   # the refusing lever is named

    def test_oom_in_the_deployment_lever_propagates(self):
        with mock.patch.object(xla_cache, "apply", side_effect=OOM):
            with self.assertRaises(XlaRuntimeError):
                stack._place_xla_cache({})
        rep = {}
        with mock.patch.object(xla_cache, "apply", side_effect=OSError("read-only cache root")):
            stack._place_xla_cache(rep)
        self.assertEqual(rep["xla_cache"]["reason"], "placement_error")

    def test_memoryerror_counts_as_oom(self):
        with mock.patch.object(xla_cache, "apply", side_effect=MemoryError()):
            with self.assertRaises(MemoryError):
                stack._place_xla_cache({})


if __name__ == "__main__":
    unittest.main()
