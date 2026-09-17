"""The PEAK line — the allocator peak of a pred pass, on every route. ONE instrument, ``opt_core.mem.peak`` (reads the process's own allocator
counters from a daemon thread; imports neither ``jax`` nor ``torch``), rides every model process — stock included — via a ``sitecustomize`` hook
placed on ``PYTHONPATH``: no kit code runs in the stock model process. A measurement, never a gate — the pass's status and exit code never read
it. The per-process records live in the pass's temporary start-up hook directory and leave with it. Under JAX's preallocated pool, nvidia-smi shows the pool, not the peak, so no smi sampler
runs here — the allocator's own high-water is the true peak."""
from __future__ import annotations

import os
import tempfile
import re

from opt_core.mem import peak as _peak

from . import report as _report

RECORD = "peak_mem.json"
PEAK_RX = re.compile(re.escape(_report.PREFIX) + r" PEAK-NOTE scope=pass status=(?P<status>\w+)(?: peak_alloc_gib=(?P<peak_alloc_gib>[\d.]+) kind=(?P<kind>\S+) bytes_limit_gib=(?P<bytes_limit_gib>[\d.]+|none)"
                     r" device=(?P<device>\S+) processes=(?P<processes>\d+) backend_records=(?P<backend_records>\d+) loaders=(?P<loaders>\S+))?(?: detail=(?P<detail>\S+))?\s*$")
_WORDS_RX = re.compile(r"[^A-Za-z0-9_.,:/=+-]+")
def arm(env: dict) -> dict:
    """Write the hook into a temporary directory and arm `env` (the model process's, in place): the hook dir LAST on PYTHONPATH (nothing of a
    kit's own path order is shadowed), one peak record per process under ``<hook dir>/records/``. Returns the hook description (paths,
    instrument sha). The directory is the pass's: cli removes it once collect() has read it — nothing of it lands in the output directory."""
    hook = _peak.write_hook(tempfile.mkdtemp(prefix="af3_jax_opt_hook_"))
    d = os.path.join(hook["dir"], "records")
    os.makedirs(d, exist_ok=True)
    env["PYTHONPATH"] = (env["PYTHONPATH"] + os.pathsep if env.get("PYTHONPATH") else "") + hook["dir"]
    env[_peak.ENV_PATH] = os.path.join(d, RECORD)
    env[_peak.ENV_PER_PROCESS] = "1"
    return {"dir": d, "hook": hook["dir"], "instrument_sha256": hook["sha256"], "record": env[_peak.ENV_PATH], "loader": "sitecustomize"}


def _words(s: str) -> str:
    return _WORDS_RX.sub("_", str(s)).strip("_")[:240] or "none"


def collect(hook: dict) -> dict:
    """Reduce the pass's per-process records (``hook``: what arm() returned) and summarise: status ok with the allocator peak, or a named absence
    (no_probe_record: no process wrote a record — the loader did not run; no_backend_record: records, but no process created a backend;
    unreadable: records none of which read)."""
    d = hook["dir"]
    if not os.path.isdir(d) or not _peak.per_process_files(d, RECORD.rsplit(".", 1)[0]):
        return {"status": "no_probe_record", "detail": _words("no per-process record - the loader did not run in the model interpreter")}
    try:
        _, doc = _peak.reduce_dir(d)
    except _peak.PeakRefusal as e:                                        # records on disk, none readable / well-formed: named
        return {"status": "unreadable", "detail": _words(e)}
    rec = {"status": "ok" if doc.get("peak_alloc_gb") is not None and doc.get("backend_records") else "no_backend_record",
           "peak_alloc_gb": doc.get("peak_alloc_gb"), "kind": doc.get("peak_alloc_kind"), "bytes_limit_gb": doc.get("jax_bytes_limit_gb"),
           "device": doc.get("device_name") or doc.get("device"), "processes": doc.get("records"), "backend_records": doc.get("backend_records"),
           "loaders": doc.get("loaders") or [], "window": doc.get("window"), "samples": doc.get("samples"),
           "xla_effective": doc.get("xla_effective"), "instrument": (doc.get("instrument") or {}).get("sha256")}
    if rec["status"] != "ok":
        rec["detail"] = _words(doc.get("error") or "no process of the pass created a backend")
    return rec


def line(rec: dict) -> str:
    if rec.get("status") == "ok":
        return _report.line("PEAK-NOTE", scope="pass", status="ok", peak_alloc_gib=f"{gib(rec['peak_alloc_gb']):.2f}", kind=rec.get("kind") or "none",
                            bytes_limit_gib=f"{gib(rec['bytes_limit_gb']):.2f}" if rec.get("bytes_limit_gb") is not None else "none", device=_words(rec.get("device") or "none"),
                            processes=rec.get("processes") or 0, backend_records=rec.get("backend_records") or 0, loaders=",".join(rec.get("loaders") or []) or "none")
    return _report.line("PEAK-NOTE", scope="pass", status=rec.get("status") or "unreadable", detail=rec.get("detail") or "none")


def gib(gb) -> float:
    """The core record's GB (1e9 bytes) as GiB (2**30 bytes): the report lines' unit."""
    return float(gb) * 1e9 / 2 ** 30
