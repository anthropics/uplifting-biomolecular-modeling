"""The tp launcher's host-memory census (tp.HostSampler): under `pss` the per-rank figure is the rank tree's Pss sum; under `rss_own`
(sandboxed /proc without smaps_rollup) it is the rank PROCESS's own VmRSS, and the tree's VmRSS sum — which counts a shared page once per
process, so a pool forked from a 30-GB rank reads workers x 30 GB — rides beside it as `host_tree_peak_mib`, never in its place."""
from openfold3_opt import tp


def _sampler(method, kib_by_pid, tree):
    hs = tp.HostSampler({0: tree[0]}, period_s=1.0)
    hs.method, hs.tree = method, "children"
    hs._tree = lambda pid, children=None: list(tree)
    hs._kib = lambda pid: kib_by_pid[pid]
    hs._used_kib = lambda: 120 * 1024 * 1024
    return hs


def test_rss_own_reports_the_rank_process_not_the_forked_tree_sum():
    gib = 1024 * 1024                                                                    # kB per GiB
    kib = {100: 30 * gib, 101: 30 * gib, 102: 30 * gib, 103: 5 * gib}                    # a 30-GiB rank, two forked pool workers reading its pages again, one loader worker
    hs = _sampler("rss_own", kib, [100, 101, 102, 103])
    hs.round(); rep = hs.report()
    assert rep["method"] == "rss_own" and rep["samples"] == 1 and rep["nproc_peak"] == {"0": 4}
    assert rep["host_rss_peak_mib"] == {"0": 30 * 1024}                                  # MiB: the rank process's own VmRSS
    assert rep["host_tree_peak_mib"] == {"0": 95 * 1024}                                 # the tree's summed reading, named for what it is
    assert rep["host_used_peak_mib"] == 120 * 1024


def test_pss_reports_the_tree_sum_as_the_figure_of_record():
    gib = 1024 * 1024
    hs = _sampler("pss", {100: 30 * gib, 101: 2 * gib, 102: 2 * gib}, [100, 101, 102])   # Pss splits shared pages: the sum is the charge
    hs.round(); hs.round(); rep = hs.report()
    assert rep["method"] == "pss" and rep["samples"] == 2
    assert rep["host_rss_peak_mib"] == {"0": 34 * 1024} == rep["host_tree_peak_mib"]


def test_the_platform_decides_the_method_word():
    import os
    hs = tp.HostSampler({0: os.getpid()})
    assert hs.method in ("pss", "rss_own") and hs.tree in ("children", "ppid_scan")
    hs.round(); rep = hs.report()
    assert rep["host_rss_peak_mib"]["0"] > 0 and rep["host_tree_peak_mib"]["0"] >= rep["host_rss_peak_mib"]["0"]
