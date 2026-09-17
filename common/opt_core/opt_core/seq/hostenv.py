"""Host environment as found: the CPU cores this process can actually use, and the size of a host worker pool derived from them.

Contract. :func:`cgroup_cpu_quota` reads the box's CPU quota in the order cgroup v2 ``cpu.max`` -> cgroup v1 ``cfs_quota`` ->
scheduler affinity -> ``os.cpu_count()`` and names the witness it used. :func:`deliverable_cores` is the count a pool may be sized
from: the scheduler affinity count capped by that quota (either witness alone over-counts inside a container: a ``cpu=8`` container
can report a quota of 24 with an affinity of 8, or the host's full affinity with a quota of 8). :func:`pool_size` turns the count into
a thread / worker count, ``max(floor, min(cap, cores) - margin)``, with an optional environment override that is recorded, never
silent. Every function returns a plain dict (the record a kit stamps into its manifest); :func:`line_fields` projects a
:func:`pool_size` record to the ``{key: value}`` fields of the kit's activation line (``opt_core.report.kv(**fields)``). Standard library
only; never raises on a box without cgroup files.
"""
from __future__ import annotations

import os
from typing import Dict, Mapping, Optional


def cgroup_cpu_quota() -> dict:
    """``{"cpus", "source", "cpu_count"}``: the CPU quota of this process's container rounded up to whole CPUs, the witness that
    produced it (``"cgroup v2 cpu.max"`` | ``"cgroup v1 cfs quota"`` | ``"sched_getaffinity"`` | ``"os.cpu_count"``), and the host's
    ``os.cpu_count()`` for the record."""
    n_host = os.cpu_count() or 1
    try:
        with open("/sys/fs/cgroup/cpu.max") as fh:
            v = fh.read().split()
        if v and v[0] != "max":
            return {"cpus": max(1, -(-int(v[0]) // int(v[1]))), "source": "cgroup v2 cpu.max", "cpu_count": n_host}
    except (OSError, ValueError, IndexError):
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fh:
            q = int(fh.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fh:
            per = int(fh.read())
        if q > 0 and per > 0:
            return {"cpus": max(1, -(-q // per)), "source": "cgroup v1 cfs quota", "cpu_count": n_host}
    except (OSError, ValueError):
        pass
    try:
        return {"cpus": len(os.sched_getaffinity(0)), "source": "sched_getaffinity", "cpu_count": n_host}
    except (AttributeError, OSError):
        return {"cpus": n_host, "source": "os.cpu_count", "cpu_count": n_host}


def affinity_count() -> int:
    """The scheduler affinity count of this process (``nproc``); ``os.cpu_count()`` where the platform has no affinity call."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def deliverable_cores() -> dict:
    """``{"cores", "affinity", "quota", "quota_source", "cpu_count"}``: ``cores`` = ``max(1, min(affinity, quota))`` — the cores this
    process can run threads on at once; the other keys are the two witnesses and the host count, for the record."""
    q = cgroup_cpu_quota()
    aff = affinity_count()
    return {"cores": max(1, min(aff, q["cpus"])), "affinity": aff, "quota": q["cpus"], "quota_source": q["source"], "cpu_count": q["cpu_count"]}


def pool_size(*, margin: int, floor: int, cap: Optional[int] = None, override_env: Optional[str] = None,
              environ: Optional[Mapping[str, str]] = None) -> dict:
    """The size of a host worker pool (writer threads, loader workers) on this box.

    ``threads`` = ``max(floor, min(cap, cores) - margin)`` over :func:`deliverable_cores` (``cap`` None = no cap). ``margin`` (cores left
    to the main thread and the device driver) and ``floor`` (the smallest useful pool) are the kit's tested values — required, no defaults. When ``override_env`` names an environment
    variable set to an integer, ``threads`` is that integer and ``override`` is True (a value that is not an integer is ignored and
    named in ``override_error``). The dict carries ``threads``, ``form`` (the expression that produced it, as text), ``override``,
    ``override_env``, ``override_error`` and every key of :func:`deliverable_cores`."""
    env = os.environ if environ is None else environ
    d = deliverable_cores()
    usable = d["cores"] if cap is None else min(int(cap), d["cores"])
    n = max(int(floor), usable - int(margin))
    form = "max({f}, min({c}, cores={k}) - {m})".format(f=int(floor), c="inf" if cap is None else int(cap), k=d["cores"], m=int(margin))
    override, override_error = False, None
    raw = env.get(override_env) if override_env else None
    if raw is not None and raw != "":
        try:
            n = int(raw)
            override = True
            form = "{name}={value}".format(name=override_env, value=n)
        except ValueError:
            override_error = "{name}={value!r} is not an integer; ignored".format(name=override_env, value=raw)
    out = {"threads": max(0, n), "form": form, "override": override, "override_env": override_env, "override_error": override_error}
    out.update(d)
    return out


def line_fields(record: Mapping, key: str = "pool") -> Dict[str, str]:
    """The activation-line fields of a :func:`pool_size` record — ``{<key>: <threads>, "cores": <n>, "quota": "<n>(<witness>)", "affinity":
    <n>}`` plus ``<key>_override: <VAR>`` when the environment sized the pool and ``<key>_override_error`` when an override was ignored; every
    value a str. Feed to ``opt_core.report.kv(**fields)`` (insertion order is the field order)."""
    fields = {key: str(record["threads"]), "cores": str(record["cores"]),
              "quota": "{q}({s})".format(q=record["quota"], s=str(record["quota_source"]).replace(" ", "_")),
              "affinity": str(record["affinity"])}  # type: Dict[str, str]
    if record.get("override"):
        fields[key + "_override"] = str(record.get("override_env"))
    if record.get("override_error"):
        fields[key + "_override_error"] = str(record["override_error"])
    return fields
