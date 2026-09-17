"""ptx_tp.mirror -- output-directory mirror + memory-peak / phase logger (no protenix imports; torch optional).

Two pieces:
 1. PhaseLog (inside each RANK process):  PL = PhaseLog.from_env()  (env PTX_TP_PHASE_LOG=<dir or file>; default out/phases_rank<r>.jsonl)
      PL.phase("trunk_start", cycle=0)  -> appends {"t","utc","rank","phase","extra","cuda":{"alloc_GiB","reserved_GiB","max_alloc_GiB","max_reserved_GiB"}}
      and touches the mirror trigger file (env PTX_TP_MIRROR_TRIGGER) so the mirror thread syncs within ~5 s (= "at each phase boundary").
 2. MirrorThread (inside the LAUNCHER / job-script process): copies watched files/dirs + an nvidia-smi memory snapshot + heartbeat to the
    mirror dir every `every_s` (<= 300 s) and whenever the trigger file is touched; finish(rc) does a final copy of out/ and writes the
    rc marker LAST.  CLI wrapper (the job's main command):
      python -m ptx_tp.mirror --mirror <persistent dir>/<label> --watch out --watch logs --every 300 -- <command ...>
    runs <command>, mirrors while it runs, then copies out/ -> <mirror>/out and writes <mirror>/rc (= "job's LAST step").
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from typing import List, Optional


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class PhaseLog:
    def __init__(self, path: str, trigger: Optional[str] = None, rank: Optional[int] = None):
        self.rank = int(os.environ.get("RANK", "0")) if rank is None else rank
        if path.endswith(".jsonl"):
            self.path = path
        else:
            os.makedirs(path, exist_ok=True)
            self.path = os.path.join(path, f"phases_rank{self.rank}.jsonl")
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        self.trigger = trigger if trigger is not None else os.environ.get("PTX_TP_MIRROR_TRIGGER", "")
        self.t0 = time.time()

    @classmethod
    def from_env(cls) -> "PhaseLog":
        return cls(os.environ.get("PTX_TP_PHASE_LOG", "out"))

    @staticmethod
    def cuda_mem(reset_peak: bool = False) -> dict:
        try:
            import torch
            if not torch.cuda.is_available():
                return {}
            G = float(2 ** 30)
            d = {"alloc_GiB": round(torch.cuda.memory_allocated() / G, 3), "reserved_GiB": round(torch.cuda.memory_reserved() / G, 3),
                 "max_alloc_GiB": round(torch.cuda.max_memory_allocated() / G, 3), "max_reserved_GiB": round(torch.cuda.max_memory_reserved() / G, 3)}
            free_b, total_b = torch.cuda.mem_get_info()
            d["dev_free_GiB"], d["dev_total_GiB"] = round(free_b / G, 3), round(total_b / G, 3)
            if reset_peak:
                torch.cuda.reset_peak_memory_stats()
            return d
        except Exception as e:  # torch absent or no device
            return {"err": repr(e)[:120]}

    def phase(self, name: str, reset_peak: bool = False, **extra) -> dict:
        rec = {"t": round(time.time(), 3), "dt": round(time.time() - self.t0, 3), "utc": _utc(), "rank": self.rank, "phase": name,
               "extra": extra, "cuda": self.cuda_mem(reset_peak=reset_peak)}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        print(f"[ptx_tp phase r{self.rank}] {name} +{rec['dt']:.1f}s cuda={rec['cuda']} {extra if extra else ''}", file=sys.stderr, flush=True)
        if self.trigger:
            try:
                with open(self.trigger, "a"):
                    os.utime(self.trigger, None)
            except Exception:
                pass
        return rec


def nvidia_smi_snapshot() -> str:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        return r.stdout.strip()
    except Exception as e:
        return f"ERR {e!r}"


class MirrorThread(threading.Thread):
    """Daemon thread: every `every_s` seconds, or within `poll_s` of the trigger file's mtime changing, copy `watch` paths into mirror_dir."""

    def __init__(self, mirror_dir: str, watch: List[str], every_s: int = 300, poll_s: float = 5.0, trigger: Optional[str] = None,
                 max_file_mb: int = 512, gpu: bool = True):
        super().__init__(daemon=True)
        self.mirror_dir, self.watch, self.every_s, self.poll_s = mirror_dir, list(watch), min(int(every_s), 300), poll_s
        self.trigger = trigger
        self.max_file_mb, self.gpu = max_file_mb, gpu
        self._stop = threading.Event()
        self.n_sync = 0
        self.last_err = None
        os.makedirs(mirror_dir, exist_ok=True)

    def _copy_path(self, src: str):
        if not os.path.exists(src):
            return
        dst = os.path.join(self.mirror_dir, os.path.basename(os.path.normpath(src)))
        if os.path.isdir(src):
            for root, _dirs, files in os.walk(src):
                rel = os.path.relpath(root, src)
                droot = os.path.join(dst, rel) if rel != "." else dst
                os.makedirs(droot, exist_ok=True)
                for fn in files:
                    s = os.path.join(root, fn)
                    try:
                        st = os.stat(s)
                        if st.st_size > self.max_file_mb * 2 ** 20:
                            continue
                        d = os.path.join(droot, fn)
                        if os.path.exists(d):
                            dstst = os.stat(d)
                            if dstst.st_size == st.st_size and dstst.st_mtime >= st.st_mtime:
                                continue
                        shutil.copy2(s, d)
                    except Exception as e:
                        self.last_err = repr(e)
        else:
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                self.last_err = repr(e)

    def sync(self, reason: str = "periodic"):
        t0 = time.time()
        for w in self.watch:
            self._copy_path(w)
        hb = {"utc": _utc(), "t": time.time(), "reason": reason, "n_sync": self.n_sync, "last_err": self.last_err}
        if self.gpu:
            snap = nvidia_smi_snapshot()
            hb["nvidia_smi"] = snap
            with open(os.path.join(self.mirror_dir, "nvidia_smi_mem.csv"), "a") as f:
                for line in snap.splitlines():
                    f.write(f"{hb['utc']},{line}\n")
        hb["sync_s"] = round(time.time() - t0, 2)
        with open(os.path.join(self.mirror_dir, "heartbeat.json"), "w") as f:
            json.dump(hb, f, indent=1)
        self.n_sync += 1

    def run(self):
        last_sync = 0.0
        last_trig = 0.0
        while not self._stop.is_set():
            now = time.time()
            trig = 0.0
            if self.trigger and os.path.exists(self.trigger):
                try:
                    trig = os.path.getmtime(self.trigger)
                except Exception:
                    trig = 0.0
            if now - last_sync >= self.every_s:
                self._safe_sync("periodic"); last_sync = time.time()
            elif trig > last_trig:
                last_trig = trig
                self._safe_sync("phase"); last_sync = time.time()
            self._stop.wait(self.poll_s)

    def _safe_sync(self, reason):
        try:
            self.sync(reason)
        except Exception as e:
            self.last_err = repr(e)

    def stop(self):
        self._stop.set()

    def finish(self, rc: int, out_dir: Optional[str] = "out"):
        """Final step of a job: stop the thread, copy out/ fully, write the rc marker LAST."""
        self.stop()
        try:
            self.sync("final")
        except Exception as e:
            self.last_err = repr(e)
        if out_dir and os.path.isdir(out_dir):
            self._copy_path(out_dir)
        with open(os.path.join(self.mirror_dir, "rc"), "w") as f:
            f.write(f"{rc}\n{_utc()}\n")
        try:
            os.sync()
        except Exception:
            pass


def run_with_mirror(cmd: List[str], mirror_dir: str, watch: List[str], every_s: int = 300, out_dir: str = "out",
                    log_path: Optional[str] = None) -> int:
    """Run cmd (stdout+stderr tee'd to log_path, default out/job.log), mirroring while it runs; returns rc after finish(rc)."""
    os.makedirs(out_dir, exist_ok=True)
    trigger = os.path.join(out_dir, ".mirror_trigger")
    open(trigger, "a").close()
    env = dict(os.environ)
    env.setdefault("PTX_TP_MIRROR_TRIGGER", os.path.abspath(trigger))
    env.setdefault("PTX_TP_PHASE_LOG", os.path.abspath(out_dir))
    log_path = log_path or os.path.join(out_dir, "job.log")
    mt = MirrorThread(mirror_dir, watch=list(dict.fromkeys(list(watch) + [out_dir])), every_s=every_s, trigger=trigger)
    mt.start()
    mt._safe_sync("start")
    t0 = time.time()
    with open(log_path, "ab") as logf:
        logf.write(f"[mirror {_utc()}] start cmd={cmd} mirror={mirror_dir}\n".encode())
        logf.flush()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        assert p.stdout is not None
        for line in iter(p.stdout.readline, b""):
            stamp = f"{time.time():.3f} ".encode()
            sys.stdout.buffer.write(line); sys.stdout.flush()
            logf.write(stamp + line); logf.flush()
        rc = p.wait()
        logf.write(f"[mirror {_utc()}] end rc={rc} wall_s={time.time() - t0:.1f}\n".encode())
    with open(os.path.join(out_dir, "job_wall.txt"), "a") as f:
        f.write(f"rc={rc} wall_s={time.time() - t0:.1f} utc={_utc()}\n")
    mt.finish(rc, out_dir)
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run a command with a periodic volume mirror (logs + memory peaks + outputs) and a final rc marker.")
    ap.add_argument("--mirror", required=True, help="mirror directory (created; never deletes anything)")
    ap.add_argument("--watch", action="append", default=[], help="file or directory to mirror (repeatable); out/ is always mirrored")
    ap.add_argument("--every", type=int, default=300, help="seconds between periodic syncs (capped at 300)")
    ap.add_argument("--out", default="out", help="output dir copied in full at the end")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("no command given after --")
    return run_with_mirror(cmd, a.mirror, a.watch, every_s=a.every, out_dir=a.out)


if __name__ == "__main__":
    sys.exit(main())
