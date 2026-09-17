"""The STACK REFUSED gate (cli.ds4sci_stack_check): an arm whose runner yaml leaves upstream's DS4Sci evoformer attention on runs only where
deepspeed's evoformer_attn op is pre-built (stack.evoformer_attn_op reads deepspeed's build record without importing deepspeed); a
kernels-off yaml is never gated."""
import importlib.machinery
import os
import sys

from openfold3_ob0_opt.tests import _stubs
from openfold3_ob0_opt import cli, modes, stack

HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _fake_deepspeed(tmp_path, record):
    pkg = tmp_path / "deepspeed"; pkg.mkdir()
    (pkg / "__init__.py").write_text("raise RuntimeError('the probe must not import deepspeed')\n")
    if record is not None:
        (pkg / stack.DS4SCI_OPS_RECORD).write_text(record)
    return importlib.machinery.ModuleSpec("deepspeed", None, origin=str(pkg / "__init__.py"))


def test_probe_reads_the_build_record_without_importing(tmp_path):
    spec = _fake_deepspeed(tmp_path, "version='0.19.2'\ngit_hash='unknown'\ninstalled_ops={'async_io': False, 'evoformer_attn': True}\ntorch_info={'version': '2.10'}\n")
    had = "deepspeed" in sys.modules                                                     # on an image that ships deepspeed another test may have imported it already
    ok, detail = stack.evoformer_attn_op(spec)
    assert ok is True and "0.19.2" in detail and ("deepspeed" in sys.modules) == had        # the probe reads the build record; it imports nothing


def test_probe_verdicts(tmp_path):
    d1 = tmp_path / "a"; d1.mkdir()
    assert stack.evoformer_attn_op(_fake_deepspeed(d1, "installed_ops={'evoformer_attn': False}\n"))[0] is False
    d2 = tmp_path / "b"; d2.mkdir()
    assert stack.evoformer_attn_op(_fake_deepspeed(d2, "installed_ops={'async_io': True}\n"))[0] is False          # the op absent from the table = not built
    d3 = tmp_path / "c"; d3.mkdir()
    ok, detail = stack.evoformer_attn_op(_fake_deepspeed(d3, None))
    assert ok is None and stack.DS4SCI_OPS_RECORD in detail                                                        # no build record = unknown, gated as not installed
    d4 = tmp_path / "d"; d4.mkdir()
    assert stack.evoformer_attn_op(_fake_deepspeed(d4, "installed_ops = oops(\n"))[0] is None                       # unreadable = unknown
    assert stack.evoformer_attn_op(importlib.machinery.ModuleSpec("deepspeed", None, origin=None))[0] is None


def test_gate_matrix(tmp_path):
    ds4sci, kernels_off, shipped = (_stubs.kernel_yaml(tmp_path, "d.yml", ds4sci=True), os.path.join(HOME, modes.KERNELS_OFF_YAML), os.path.join(HOME, modes.SHIPPED_YAML))
    absent, built, missing = (lambda: (None, "deepspeed is not installed")), (lambda: (True, "built")), (lambda: (False, "not built"))
    assert cli.ds4sci_stack_check(HOME, kernels_off, "pred --mode fast", probe=absent) is None                       # kernels off: never gated, the probe not even asked
    assert cli.ds4sci_stack_check(HOME, shipped, "pred --mode off", probe=absent) is None                             # as shipped (0.5.0): the DS4Sci attention off, never gated
    assert cli.ds4sci_stack_check(HOME, ds4sci, "pred --mode off", probe=built) is None                               # a DS4Sci candidate on a stack with the op
    for probe in (absent, missing):
        for what in ("pred --mode off", "pred --mode exact"):
            msg = cli.ds4sci_stack_check(HOME, ds4sci, what, probe=probe)                                             # a DS4Sci candidate without the op: refused by name, the shipped and kernels-off yamls named
            assert msg.startswith(cli.STACK_REFUSED) and "use_deepspeed_evo_attention on" in msg and modes.SHIPPED_YAML in msg and modes.KERNELS_OFF_YAML in msg
    assert cli.ds4sci_stack_check(HOME, os.path.join(HOME, modes.BIG_BF16_C16_YAML), "pred --mode big", probe=missing) is None      # the big line's yaml: DS4Sci off, never gated


def test_ds4sci_load_words(monkeypatch):
    """stack.ds4sci_load: the op LOADS (True, its file) or the reason (False, …) — a raising builder is a named failure, never an exception out of the gate."""
    from openfold3_ob0_opt import stack
    class Loaded:
        __file__ = "/site/deepspeed/ops/evoformer_attn_op.so"
    class OK:
        def load(self, verbose=False): return Loaded()
    class Broken:
        def load(self, verbose=False): raise RuntimeError("Unable to JIT load the evoformer_attn op due to it not being compatible")
    assert stack.ds4sci_load(OK) == (True, Loaded.__file__)
    ok, why = stack.ds4sci_load(Broken)
    assert ok is False and why.startswith("RuntimeError: Unable to JIT load")
    assert stack.DS4SCI_NOT_LOADED == "DS4Sci evoformer attention did not load"
