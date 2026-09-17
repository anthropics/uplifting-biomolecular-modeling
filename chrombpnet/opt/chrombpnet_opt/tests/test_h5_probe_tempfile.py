"""h5_fast._guess_chunks reads h5py's auto-chunk shape off a throwaway file: that file is a private tempfile.mkstemp one (0600, O_EXCL,
unpredictable name), removed on return — never a fixed or pid-derived name in the shared temporary directory. No TF, no GPU."""
import os, sys, stat, tempfile
import pytest
KIT_HO_TF = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "kit_ho", "tf")   # opt/kit_ho/tf: the fast kit's host-side package lives there
h5py = pytest.importorskip("h5py"); np = pytest.importorskip("numpy")
if not os.path.isdir(os.path.join(KIT_HO_TF, "chrombpnet_fastkit")):
    pytest.skip("opt/kit_ho is not in this tree", allow_module_level=True)
if KIT_HO_TF not in sys.path:
    sys.path.insert(0, KIT_HO_TF)
from chrombpnet_fastkit import h5_fast  # noqa: E402


def test_probe_file_is_a_private_mkstemp_file_removed_on_return(monkeypatch, tmp_path):
    made = []
    real = tempfile.mkstemp
    def spy(*a, **k):
        fd, p = real(*a, **k); made.append((p, stat.S_IMODE(os.stat(p).st_mode))); return fd, p
    monkeypatch.setattr(tempfile, "mkstemp", spy)
    monkeypatch.setenv("TMPDIR", str(tmp_path)); tempfile.tempdir = None
    try:
        ch = h5_fast._guess_chunks((1000, 300), np.float64)
    finally:
        tempfile.tempdir = None
    assert isinstance(ch, tuple) and len(ch) == 2
    assert len(made) == 1, made                                            # the probe file came from mkstemp — nothing else was opened by name
    p, mode = made[0]
    assert os.path.dirname(p) == str(tmp_path) and os.path.basename(p).startswith("fastkit_h5_probe_") and p.endswith(".h5")
    assert mode == 0o600
    assert not os.path.exists(p)                                           # removed on return
    assert os.listdir(str(tmp_path)) == []


def test_no_fixed_temporary_name_in_the_writer_source():
    src = open(h5_fast.__file__, encoding="utf-8").read()
    assert '"/tmp/' not in src and "'/tmp/" not in src
