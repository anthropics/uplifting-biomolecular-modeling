"""`--mode big --n_gpu P`: the axis' tokens, its refusals by name, the P=1 no-op, the pad rule — CPU, no GPU, no jax.

The refusal sentences and the ACTIVE-line tokens belong to the shared core (opt_core.mem.ngpu, opt_core.mem.rowpair_jax.evidence): the tests
assert the exact text callers grep for. Against a core WITHOUT those modules (a pin older than the n_gpu core) the one refusal is
`core_missing:<module>` — asserted by name in that case, so these tests are green on both cores and silent on neither."""
import io
import contextlib
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from colabfold_opt import cli, modes, report, stack

try:
    from opt_core.mem import ngpu as _ngpu  # noqa: F401
    from colabfold_opt import big
    HAVE_AXIS = True
except ImportError:
    big = None
    HAVE_AXIS = False

REFUSE_MODE = "refused: n_gpu>1 requires --mode big (sharded reductions are not bitwise)"


def run_cli(argv, count=1):
    err, out = io.StringIO(), io.StringIO()
    with mock.patch.object(stack, "gpu_info", return_value={"name": "NVIDIA H100 80GB HBM3", "memory_mib": 81559, "compute_cap": "9.0", "count": count}), \
            redirect_stderr(err), redirect_stdout(out):
        rc = cli.main(list(argv))
    return rc, err.getvalue() + out.getvalue()


class TestPoolFraction(unittest.TestCase):
    """`pred --mode big --n_gpu 8` starts the model process with jax's memory pool at 0.90 of each card (modes.N_GPU_MEM_FRACTION: NCCL's
    communicators allocate outside the pool; the P = 8 line runs at 0.90); P = 1, 2, 4 keep the environment's value; a fraction the caller set —
    XLA_PYTHON_CLIENT_MEM_FRACTION at anything but the image's preset (stock/PINS.json image.env 0.95), or XLA_CLIENT_MEM_FRACTION at all — wins."""

    def test_the_rule(self):
        F, ALT = modes.MEM_FRACTION_ENV, modes.MEM_FRACTION_ENV_ALT
        self.assertEqual((modes.N_GPU_MEM_FRACTION, stack.pins()["image"]["env"][F]), ({8: "0.90"}, "0.95"))
        self.assertEqual([cli.n_gpu_mem_fraction(8, e) for e in ({}, {F: ""}, {F: "0.95"}, {F: ".950"})], ["0.90"] * 4)   # unset or the image's preset: 0.90
        self.assertEqual([cli.n_gpu_mem_fraction(8, e) for e in ({F: "0.85"}, {F: "0.97"}, {ALT: "0.9"}, {F: "0.95", ALT: "0.9"})], [None] * 4)   # the caller's choice wins
        self.assertEqual([cli.n_gpu_mem_fraction(p, {}) for p in (1, 2, 4)], [None] * 3)


class TestAxisGate(unittest.TestCase):
    def test_p1_outside_big_imports_nothing(self):
        self.assertIsNone(stack.gate_n_gpu("fast", 1, None))
        self.assertIsNone(stack.gate_n_gpu("exact", 1, 4))

    @unittest.skipIf(HAVE_AXIS, "core carries the n_gpu modules")
    def test_core_without_the_axis_refuses_by_name(self):
        g = stack.gate_n_gpu("big", 1, None)
        self.assertTrue(g.startswith("core_missing:opt_core.mem"), g)
        rc, text = run_cli(["check", "--mode", "big"])
        self.assertEqual(rc, cli.EXIT_NOT_ACTIVE); self.assertIn("NOT ACTIVE: core_missing:opt_core.mem", text)

    @unittest.skipUnless(HAVE_AXIS, "needs opt_core.mem.ngpu + opt_core.mem.rowpair_jax")
    def test_refusals_by_name(self):
        self.assertEqual(stack.gate_n_gpu("fast", 2, 8), REFUSE_MODE)
        self.assertEqual(stack.gate_n_gpu("exact", 4, 8), REFUSE_MODE)
        self.assertEqual(stack.gate_n_gpu("off", 2, 8), REFUSE_MODE)
        self.assertEqual(stack.gate_n_gpu("big", 3, 8), "refused: n_gpu=3 not in {1,2,4,8} (the P set this kit ships, modes.N_GPU_SUPPORTED)")
        self.assertEqual(stack.gate_n_gpu("big", 4, 2), "refused: n_gpu=4 visible=2")
        self.assertIsNone(stack.gate_n_gpu("big", 2, 2)); self.assertIsNone(stack.gate_n_gpu("big", 1, None)); self.assertIsNone(stack.gate_n_gpu("big", 8, None))

    @unittest.skipUnless(HAVE_AXIS, "needs opt_core.mem.ngpu + opt_core.mem.rowpair_jax")
    def test_cli_refuses_before_anything_runs(self):
        rc, text = run_cli(["check", "--mode", "fast", "--n_gpu", "2"], count=8)
        self.assertEqual(rc, cli.EXIT_NOT_ACTIVE); self.assertIn(f"[colabfold-opt] NOT ACTIVE: {REFUSE_MODE} (mode=fast n_gpu=2)", text)
        rc, text = run_cli(["pred", "--mode", "exact", "--n_gpu", "2", "x", "y"], count=8)
        self.assertEqual(rc, cli.EXIT_NOT_ACTIVE); self.assertIn(REFUSE_MODE, text)
        rc, text = run_cli(["check", "--mode", "big", "--n_gpu", "4"], count=2)
        self.assertEqual(rc, cli.EXIT_NOT_ACTIVE); self.assertIn("NOT ACTIVE: refused: n_gpu=4 visible=2 (mode=big n_gpu=4)", text)
        rc, text = run_cli(["check", "--mode", "big", "--n_gpu", "6"], count=8)
        self.assertEqual(rc, cli.EXIT_NOT_ACTIVE); self.assertIn("refused: n_gpu=6 not in {1,2,4,8}", text)
        rc, text = run_cli(["check", "--mode", "big", "--n_gpu", "0"], count=8)
        self.assertEqual(rc, cli.EXIT_USAGE); self.assertIn("--n_gpu 0: a positive GPU count is required", text)

    @unittest.skipUnless(HAVE_AXIS, "needs opt_core.mem.ngpu + opt_core.mem.rowpair_jax")
    def test_active_line_tokens(self):
        self.assertEqual(big.active_fields(1), [("n_gpu", 1), ("sharding", "none")])
        self.assertEqual(big.active_fields(2), [("n_gpu", 2), ("sharding", "rowpair")])
        rep = {"active": True, "mode": "big", "levers_applied": ["DEVICE_RESIDENT", "SUBBATCH", "TRIMUL_PALLAS", "AF_PALLAS_ATTN", "PALLAS_MSA", "TRIATTN_XLA", "MSA_COL_CUDNN", "TEMPL_DEDUP", "TRANSITION", "ROWPAIR"], "tokens_min": 996, "tokens_max": 996,
               "queries": 1, "colabfold_version": "1.6.1", "alphafold_colabfold_version": "2.3.13", "jax_version": "0.5.3", "key": "9.0|0.5.3",
               "gpu": {"name": "NVIDIA H100 80GB HBM3"}, "n_gpu": 2, "n_gpu_fields": big.active_fields(2)}
        line = report.activation_line(rep)
        self.assertTrue(line.startswith("[colabfold-opt] ACTIVE mode=big levers=DEVICE_RESIDENT,SUBBATCH,TRIMUL_PALLAS,AF_PALLAS_ATTN,PALLAS_MSA,TRIATTN_XLA,MSA_COL_CUDNN,TEMPL_DEDUP,TRANSITION,ROWPAIR "), line)
        self.assertTrue(line.endswith(" gpu=NVIDIA_H100_80GB_HBM3 n_gpu=2 sharding=rowpair"), line)
        rep.update(n_gpu=1, n_gpu_fields=big.active_fields(1), levers_applied=["DEVICE_RESIDENT", "SUBBATCH", "TRIMUL_PALLAS", "AF_PALLAS_ATTN", "PALLAS_MSA", "TRIATTN_XLA", "MSA_COL_CUDNN", "TEMPL_DEDUP", "TRANSITION"])
        self.assertTrue(report.activation_line(rep).startswith("[colabfold-opt] ACTIVE mode=big levers=DEVICE_RESIDENT,SUBBATCH,TRIMUL_PALLAS,AF_PALLAS_ATTN,PALLAS_MSA,TRIATTN_XLA,MSA_COL_CUDNN,TEMPL_DEDUP,TRANSITION tokens=996-996 "))
        self.assertTrue(report.activation_line(rep).endswith(" n_gpu=1 sharding=none"))
        self.assertNotIn("n_gpu=", report.activation_line({**rep, "mode": "fast", "n_gpu_fields": None}))      # fast / exact lines carry no axis tokens (their lines unchanged)

    @unittest.skipUnless(HAVE_AXIS, "needs opt_core.mem.ngpu + opt_core.mem.rowpair_jax")
    def test_p1_installs_nothing(self):
        from colabfold_opt import device_resident
        self.addCleanup(big.reset_for_tests)
        st = big.apply(1)
        self.assertEqual((st["enabled"], st["n_gpu"], st["sharding"], st["pad_multiple"]), (False, 1, "none", 1))     # P=1 installs no sharding: the process's lever set is fast's
        self.assertIsNone(big._STATE["patches"]); self.assertIsNone(big._STATE["rmesh"]); self.assertIsNone(device_resident._MESH["rmesh"])
        line = big.exit_line()
        self.assertIn("LEVER name=ROWPAIR state=off reason=n_gpu=1 impl=opt_core.mem.rowpair_jax@", line); self.assertIn("origin=core strategy=F7.tensor_parallel n_gpu=1 sharding=none", line)
        self.assertEqual(tuple(modes.resolve("big")["levers"]), ("DEVICE_RESIDENT", "SUBBATCH", "TRIMUL_PALLAS", "AF_PALLAS_ATTN", "PALLAS_MSA", "TRIATTN_XLA", "MSA_COL_CUDNN", "TEMPL_DEDUP", "TRANSITION", "ROWPAIR"))    # the table names the axis; activate() drops it at P=1

    @unittest.skipUnless(HAVE_AXIS, "needs opt_core.mem.ngpu + opt_core.mem.rowpair_jax")
    def test_pad_len_raised_to_a_multiple_of_p(self):
        seen = []
        padded = big._pad_to_multiple(lambda **kw: seen.append(kw["pad_len"]) or kw["pad_len"], 2)
        self.addCleanup(big.reset_for_tests)
        self.assertEqual(padded(pad_len=199, sequences_lengths=[110, 89]), 200)
        self.assertEqual(padded(pad_len=220, sequences_lengths=[110, 89]), 220)
        self.assertEqual(big._pad_to_multiple(lambda **kw: kw["pad_len"], 8)(pad_len=210, sequences_lengths=[110, 89]), 216)
        self.assertEqual(big._pad_to_multiple(lambda **kw: kw["pad_len"], 4)(pad_len=None, sequences_lengths=[110, 89]), 200)
        self.assertEqual(seen, [200, 220]); self.assertEqual(big._STATE["pad_events"][0], {"route": "monomer", "asked": 199, "seq_len": 199, "pad_len": 200})

    def test_declared_variable(self):
        from colabfold_opt import _autoload
        self.assertIn("COLABFOLD_OPT_N_GPU", _autoload.ENV_NAMES); self.assertEqual(modes.ENV_N_GPU, "COLABFOLD_OPT_N_GPU")
        self.assertEqual(_autoload.undeclared({"COLABFOLD_OPT_N_GPU": "2", "COLABFOLD_OPT": "big"}), [])


if __name__ == "__main__":
    unittest.main()
