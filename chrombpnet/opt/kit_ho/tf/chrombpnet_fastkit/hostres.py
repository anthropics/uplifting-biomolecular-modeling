"""Worker pools are sized from the CONTAINER's CPU quota, never the host's count. The kit's own reader (nothing outside the kit is imported):
cgroup v2 cpu.max ('quota period' or 'max'), cgroup v1 cpu.cfs_quota_us / cpu.cfs_period_us, the scheduler affinity mask, and os.cpu_count()
as the host count. effective_cpus() = the minimum of the defined ones (>= 1); pool_form() = the stamp a run record carries."""
import os


def _read(p):
    try: return open(p).read().strip()
    except Exception: return None


def cgroup_cpu_quota():
    """CPUs granted by the cgroup (float) or None when unlimited / unreadable."""
    # the kit's own reader: nothing outside the kit is imported
    v2 = _read("/sys/fs/cgroup/cpu.max")
    if v2:
        parts = v2.split()
        if parts and parts[0] != "max":
            try: return float(parts[0]) / float(parts[1])
            except Exception: pass
    q, per = _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if q and per:
        try:
            if int(q) > 0: return int(q) / int(per)
        except Exception: pass
    return None


def effective_cpus():
    cands = []
    q = cgroup_cpu_quota()
    if q: cands.append(max(1, int(q)))
    try: cands.append(len(os.sched_getaffinity(0)))
    except Exception: pass
    cands.append(os.cpu_count() or 1)
    return max(1, min(cands))


def pool_form(requested=None):
    """The stamp: how a pool was sized on this host."""
    q = cgroup_cpu_quota()
    try: aff = len(os.sched_getaffinity(0))
    except Exception: aff = None
    eff = effective_cpus()
    return {"pool_form": "cgroup_quota" if q else ("sched_affinity" if aff else "os.cpu_count"), "cgroup_cpu_quota": q, "sched_affinity": aff, "host_cpu_count": os.cpu_count(), "cpus_effective": eff,
            "requested": requested, "used": (min(requested, eff) if requested else eff), "home": "engines.core.cgroup_cpu_quota (never imported on the user path: v0.12.36)" if _has_home() else "chrombpnet_fastkit.hostres (the same sources; mirror when the tree is not on the path)"}


def _has_home():
    return False   # the kit never reaches outside itself; pool_form() names this module as the reader


def cap(requested):
    """A pool size capped by the container's CPUs (never below 1)."""
    return max(1, min(int(requested), effective_cpus()))
