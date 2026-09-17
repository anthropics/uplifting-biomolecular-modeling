"""The STACK REFUSED gate (cli.ds4sci_stack_check): an arm whose runner yaml leaves upstream's DS4Sci evoformer attention on runs only where
deepspeed's evoformer_attn op is pre-built (stack.evoformer_attn_op reads deepspeed's build record without importing deepspeed); a
kernels-off yaml is never gated."""
import importlib.machinery
import importlib.util
import os
import sys

import pytest

from openfold3_opt import cli, modes, stack

no_deepspeed_only = pytest.mark.skipif(importlib.util.find_spec("deepspeed") is not None,
                                       reason="deepspeed is installed on this interpreter: the `not imported` assertion is decidable only where deepspeed is absent "
                                              "(the probe reads deepspeed's build record and module file without importing it; on the full stack another import may already hold it)")

HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _fake_deepspeed(tmp_path, record):
    pkg = tmp_path / "deepspeed"; pkg.mkdir()
    (pkg / "__init__.py").write_text("raise RuntimeError('the probe must not import deepspeed')\n")
    if record is not None:
        (pkg / stack.DS4SCI_OPS_RECORD).write_text(record)
    return importlib.machinery.ModuleSpec("deepspeed", None, origin=str(pkg / "__init__.py"))


@no_deepspeed_only
def test_probe_reads_the_build_record_without_importing(tmp_path):
    spec = _fake_deepspeed(tmp_path, "version='0.19.2'\ngit_hash='unknown'\ninstalled_ops={'async_io': False, 'evoformer_attn': True}\ntorch_info={'version': '2.10'}\n")
    ok, detail = stack.evoformer_attn_op(spec)
    assert ok is True and "0.19.2" in detail and "deepspeed" not in sys.modules


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


def test_gate_matrix():
    stock, stock_det, kernels_off = os.path.join(HOME, modes.STOCK_YAML), os.path.join(HOME, modes.STOCK_DET_YAML), os.path.join(HOME, modes.KERNELS_OFF_YAML)
    absent, built, missing = (lambda: (None, "deepspeed is not installed")), (lambda: (True, "built")), (lambda: (False, "not built"))
    assert cli.ds4sci_stack_check(HOME, kernels_off, "pred --mode fast", probe=absent) is None                       # kernels off: never gated, the probe not even asked
    assert cli.ds4sci_stack_check(HOME, stock_det, "pred --mode off --det 1", probe=absent) is None                  # stock under the det recipe: the DS4Sci attention off, never gated
    assert cli.ds4sci_stack_check(HOME, stock, "pred --mode off", probe=built) is None                               # stock at det 0 on a stack with the op
    for probe in (absent, missing):
        for what in ("pred --mode off", "pred --mode exact"):
            msg = cli.ds4sci_stack_check(HOME, stock, what, probe=probe)                                              # stock at det 0 without the op: refused by name, the det recipe and the kernels-off yaml named
            assert msg.startswith(cli.STACK_REFUSED) and "use_deepspeed_evo_attention on" in msg and modes.STOCK_DET_YAML in msg and modes.KERNELS_OFF_YAML in msg
    assert cli.ds4sci_stack_check(HOME, os.path.join(HOME, modes.OFFLOAD_SHIPPED_C16_YAML), "pred --mode big", probe=missing).startswith(cli.STACK_REFUSED)
    assert cli.ds4sci_stack_check(HOME, os.path.join(HOME, modes.OFFLOAD_C16_YAML), "pred --mode big", probe=missing) is None


def test_ds4sci_load_words(monkeypatch):
    """stack.ds4sci_load: the op LOADS (True, its file) or the reason (False, …) — a raising builder is a named failure, never an exception out of the gate."""
    from openfold3_opt import stack
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


# ---- the device-code half of the gate: the op's embedded cubin / PTX architectures against the GPU's compute capability ----
def _fatbin(entries):
    """A CUDA fat binary container holding `entries` = [(kind, sm)] (kind 1 = PTX, 2 = cubin) with 16-byte dummy payloads — the layout
    stack.fatbin_code reads (container: magic u32, version u16, header size u16, payload size u64; entry: kind u16 @0, header size u32 @4,
    padded payload size u64 @8, sm u32 @28)."""
    import struct
    body = b""
    for kind, sm in entries:
        hdr = bytearray(64)
        struct.pack_into("<HHI", hdr, 0, kind, 0x0101, 64)
        struct.pack_into("<Q", hdr, 8, 16)
        struct.pack_into("<I", hdr, 28, sm)
        body += bytes(hdr) + b"\xAB" * 16
    return struct.pack("<IHHQ", stack._FATBIN_MAGIC, 1, 16, len(body)) + body


def _elf_with_fatbin(section: bytes) -> bytes:
    """A minimal ELF64 little-endian image with one `.nv_fatbin` section (plus the null section and .shstrtab) — enough for stack._elf_sections."""
    import struct
    shstr = b"\0.nv_fatbin\0.shstrtab\0"
    data_off = 64
    shstr_off = data_off + len(section)
    shoff = shstr_off + len(shstr)
    shoff += -shoff % 8
    ehdr = bytearray(64)
    ehdr[:4] = b"\x7fELF"; ehdr[4] = 2; ehdr[5] = 1; ehdr[6] = 1
    struct.pack_into("<HHIQQQIHHHHHH", ehdr, 16, 3, 62, 1, 0, 0, shoff, 0, 64, 0, 0, 64, 3, 2)   # e_type ET_DYN, x86-64, …, e_shoff, e_ehsize 64, e_shentsize 64, e_shnum 3, e_shstrndx 2
    def sh(name, typ, off, size):
        h = bytearray(64); struct.pack_into("<IIQQQQ", h, 0, name, typ, 0, 0, off, size); return bytes(h)
    body = bytes(ehdr) + section + shstr
    body += b"\0" * (shoff - len(body))
    return body + sh(0, 0, 0, 0) + sh(1, 1, data_off, len(section)) + sh(12, 3, shstr_off, len(shstr))


def test_fatbin_code_lists_cubin_and_ptx_architectures():
    two_units = _fatbin([(2, 80), (2, 90), (2, 100), (1, 100)]) + _fatbin([(2, 80), (2, 90), (2, 100), (1, 100)])   # the multi-arch op: two translation units, each sm_80/90/100 cubins + compute_100 PTX
    assert stack.fatbin_code(two_units) == ([80, 90, 100], [100])
    assert stack.fatbin_code(_fatbin([(2, 90)])) == ([90], [])                                                       # the H100-only build (DS4SCI_ARCHS=9.0)
    assert stack.fatbin_code(b"\0" * 40) == ([], [])                                                                # no container = no code (the caller's `unknown`)


@no_deepspeed_only
def test_op_code_is_read_from_the_module_file_without_importing(tmp_path):
    spec = _fake_deepspeed(tmp_path, "installed_ops={'evoformer_attn': True}\n")
    ops = tmp_path / "deepspeed" / "ops"; ops.mkdir()
    assert stack.evoformer_attn_op_code(spec)[:2] == (None, None)                                                   # installed per the record but no module file: unknown, named
    (ops / "evoformer_attn_op.cpython-311-x86_64-linux-gnu.so").write_bytes(_elf_with_fatbin(_fatbin([(2, 90)])))
    cubin, ptx, detail = stack.evoformer_attn_op_code(spec)
    assert (cubin, ptx) == ([90], []) and "cubin sm_90" in detail and "PTX none" in detail and "deepspeed" not in sys.modules
    (ops / "evoformer_attn_op.cpython-311-x86_64-linux-gnu.so").write_bytes(b"not an elf")
    assert stack.evoformer_attn_op_code(spec)[:2] == (None, None)                                                   # an unreadable module: unknown, never a verdict


def test_device_code_serves_follows_cuda_compatibility():
    multi, h100_only = ([80, 90, 100], [100]), ([90], [])
    for cc in ("8.0", 8.0, (8, 0), "8.6", "9.0", "10.0", "10.3", "12.0"):
        assert stack.device_code_serves(cc, *multi) is True                                                         # sm_80 cubin serves 8.x, sm_90 9.0, sm_100 10.0, compute_100 PTX 10.3 and above
    assert stack.device_code_serves("8.0", *h100_only) is False                                                     # the trap: an sm_90-only op on an A100
    assert stack.device_code_serves("8.9", *h100_only) is False
    assert stack.device_code_serves("9.0", *h100_only) is True
    assert stack.device_code_serves("10.0", *h100_only) is False                                                    # a cubin never serves another major architecture; no PTX to JIT
    assert stack.device_code_serves("7.5", *multi) is False
    assert stack.device_code_serves(None, *multi) is None and stack.device_code_serves("8.0", None, None) is None   # no GPU / no code list: no verdict
    assert stack.device_code_serves("n/a", *multi) is None


def test_gate_refuses_an_op_without_this_gpus_device_code_by_name():
    stock = os.path.join(HOME, modes.STOCK_YAML)
    built = lambda: (True, "built")
    h100_only = lambda: ([90], [], "evoformer_attn_op.so: cubin sm_90; PTX none")
    multi = lambda: ([80, 90, 100], [100], "evoformer_attn_op.so: cubin sm_80,sm_90,sm_100; PTX compute_100")
    unknown = lambda: (None, None, "no evoformer_attn_op*.so")
    assert cli.ds4sci_stack_check(HOME, stock, "pred --mode off", probe=built, code=multi, cc="8.0") is None        # A100 on the multi-arch image
    assert cli.ds4sci_stack_check(HOME, stock, "pred --mode off", probe=built, code=h100_only, cc="9.0") is None    # H100 on its own image: unchanged
    assert cli.ds4sci_stack_check(HOME, stock, "pred --mode off", probe=built, code=unknown, cc="8.0") is None      # no code list readable: no verdict, the run proceeds
    assert cli.ds4sci_stack_check(HOME, stock, "pred --mode off", probe=built, code=h100_only, cc=None) is None     # no GPU (a CPU box's check): no verdict
    for what in ("pred --mode off", "pred --mode exact", "check --mode off"):
        msg = cli.ds4sci_stack_check(HOME, stock, what, probe=built, code=h100_only, cc="8.0")                     # A100 on the sm_90-only image: refused by name before the run
        assert msg.startswith(cli.STACK_REFUSED) and "no device code for this GPU (compute capability 8.0" in msg and "cubin sm_90" in msg
        assert "DS4SCI_ARCHS" in msg and modes.STOCK_DET_YAML in msg and modes.KERNELS_OFF_YAML in msg
    assert cli.DS4SCI_STACK["evoformer_attn_op_code"][0] is False
    assert cli.ds4sci_stack_check(HOME, os.path.join(HOME, modes.KERNELS_OFF_YAML), "pred --mode fast", probe=built, code=h100_only, cc="8.0") is None   # kernels off: never judged
