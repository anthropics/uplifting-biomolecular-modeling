"""The deterministic recipe's carrier census (det.xla_caches_state / autotune_entries / detclass_line): the class carries XLA's per-fusion
autotune results under jax's default; a JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES that does not name the autotune directory empties it and the
DETCLASS line says so by name (never a refusal)."""
import os

from af3_jax_opt import det
from af3_jax_opt.tests.conftest import assert_no_markers


def test_xla_caches_state_words():
    assert det.xla_caches_state({}) == {"word": "default", "carried": True}
    assert det.xla_caches_state({det.XLA_CACHES_ENV: "all"}) == {"word": "all", "carried": True}
    assert det.xla_caches_state({det.XLA_CACHES_ENV: "xla_gpu_per_fusion_autotune_cache_dir"})["carried"] is True
    assert det.xla_caches_state({det.XLA_CACHES_ENV: "xla_gpu_kernel_cache_file,xla_gpu_per_fusion_autotune_cache_dir"})["carried"] is True
    assert det.xla_caches_state({det.XLA_CACHES_ENV: "none"}) == {"word": "none", "carried": False}
    assert det.xla_caches_state({det.XLA_CACHES_ENV: ""}) == {"word": "blank", "carried": False}
    assert det.xla_caches_state({det.XLA_CACHES_ENV: "xla_gpu_kernel_cache_file"})["carried"] is False


def test_autotune_entries_counts_the_carrier(tmp_path):
    cls = str(tmp_path / "class")
    assert det.autotune_entries(None) is None and det.autotune_entries(cls) is None
    d = os.path.join(cls, det.AUTOTUNE_SUBDIR)
    os.makedirs(os.path.join(d, "tmp"))                                    # XLA keeps a tmp/ directory beside the entries: not an entry
    for i in range(3):
        open(os.path.join(d, f"e{i}.textproto"), "w").close()
    assert det.autotune_entries(cls) == 3


def test_detclass_line_carried_and_not(tmp_path):
    cls = str(tmp_path / "class")
    os.makedirs(os.path.join(cls, det.AUTOTUNE_SUBDIR))
    open(os.path.join(cls, det.AUTOTUNE_SUBDIR, "a.textproto"), "w").close()
    ln = det.detclass_line(cls, {})
    assert_no_markers(ln)
    assert ln.startswith("[af3-jax-opt] DETCLASS carrier=xla_autotune_results dir=" + os.path.join(cls, det.AUTOTUNE_SUBDIR) + " entries=1 xla_caches=default state=carried") and "note=" not in ln
    ln = det.detclass_line(cls, {det.XLA_CACHES_ENV: "none"})
    assert " xla_caches=none state=not_carried note=exact==off_not_expected(" in ln and det.XLA_CACHES_ENV + "=none" in ln and " " not in ln.split("note=", 1)[1]
    assert " entries=absent " in det.detclass_line(str(tmp_path / "fresh"), {}) and " entries=absent " in det.detclass_line(None, {})
