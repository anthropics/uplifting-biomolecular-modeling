"""opt_core.seq.hostenv: the core count witnesses agree with the OS, the pool rule is the stated expression, the override is recorded."""
import os

from opt_core.seq import hostenv
from opt_core import report


def test_quota_record_shape_and_witness():
    q = hostenv.cgroup_cpu_quota()
    assert set(q) == {"cpus", "source", "cpu_count"}
    assert q["cpus"] >= 1 and q["cpu_count"] == (os.cpu_count() or 1)
    assert q["source"] in ("cgroup v2 cpu.max", "cgroup v1 cfs quota", "sched_getaffinity", "os.cpu_count")


def test_deliverable_is_min_of_witnesses():
    d = hostenv.deliverable_cores()
    assert d["cores"] == max(1, min(d["affinity"], d["quota"]))
    assert d["affinity"] == hostenv.affinity_count() >= 1


def test_pool_rule_floor_margin_cap():
    d = hostenv.deliverable_cores()
    r = hostenv.pool_size(margin=2, floor=2, environ={})
    assert r["threads"] == max(2, d["cores"] - 2) and r["override"] is False and r["override_error"] is None
    assert r["form"] == "max(2, min(inf, cores=%d) - 2)" % d["cores"]
    r1 = hostenv.pool_size(margin=0, floor=1, cap=1, environ={})
    assert r1["threads"] == 1 and r1["form"].startswith("max(1, min(1, cores=")
    for k in ("cores", "affinity", "quota", "quota_source", "cpu_count"):
        assert r[k] == d[k]


def test_override_recorded_never_silent():
    r = hostenv.pool_size(margin=2, floor=2, override_env="KIT_WRITER_THREADS", environ={"KIT_WRITER_THREADS": "7"})
    assert r["threads"] == 7 and r["override"] is True and r["form"] == "KIT_WRITER_THREADS=7"
    bad = hostenv.pool_size(margin=2, floor=2, override_env="KIT_WRITER_THREADS", environ={"KIT_WRITER_THREADS": "many"})
    assert bad["override"] is False and "not an integer" in bad["override_error"] and bad["threads"] == hostenv.pool_size(margin=2, floor=2, environ={})["threads"]
    unset = hostenv.pool_size(margin=2, floor=2, override_env="KIT_WRITER_THREADS", environ={})
    assert unset["override"] is False and unset["override_error"] is None


def test_line_fields_feed_report_kv():
    r = hostenv.pool_size(margin=2, floor=2, override_env="KIT_WRITER_THREADS", environ={"KIT_WRITER_THREADS": "3"})
    f = hostenv.line_fields(r, key="writer_threads")
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in f.items())
    line = report.kv(**f)
    assert line.startswith("writer_threads=3 cores=%d quota=%d(" % (r["cores"], r["quota"]))
    assert " affinity=%d" % r["affinity"] in line and line.endswith("writer_threads_override=KIT_WRITER_THREADS")
    plain = report.kv(**hostenv.line_fields(hostenv.pool_size(margin=2, floor=2, environ={})))
    assert plain.split()[0].startswith("pool=") and "override" not in plain


def test_pool_size_has_no_default_margin_or_floor():
    import pytest
    with pytest.raises(TypeError):
        hostenv.pool_size()                      # the kit passes its tested margin / floor; there is no default
    with pytest.raises(TypeError):
        hostenv.pool_size(2, 2)                  # keyword-only
