"""png_workers — the `-bw` finish stage's two PNG tasks (the stock's own counts_metrics: the density-scatter PNG + spearman/pearson/mse; the stock's own
plot_histogram: the profile-JSD PNG) in two helper processes SPAWNED at process start: fresh interpreters (subprocess.Popen of this module's
helper loop), never a fork of the main process. The hazard this shape removes: fork() in a threaded parent copies only the forking thread — a
lock any other thread holds at that instant (an allocator arena, a logging/IO lock, a TF/cuDNN/torch runtime mutex) stays locked in the child
forever, and a child that never exits is caught only by the main process's bounded join (a loud failure minutes later). A fork-before-the-runtimes point
does not exist on the deterministic arm (the recipe's seed hook imports TensorFlow at interpreter start, before any script line), so the helpers
are spawned: they share no memory with the main process, import numpy / scipy / matplotlib / the stock's metrics module themselves — once, at start,
while the main process loads its model — and receive per item the same float64 arrays the serial stock calls take (length-prefixed pickle frames over two pipes per
helper: jobs down, replies up; stdout/stderr = the main process's log). Two helpers keep the two PNGs concurrent with the main process's h5 write. A helper error, a
dead helper or a bounded wait raises in the main process (no fallback); a helper exits on a None job (stop()) or on its job pipe's EOF (the main process's
exit_fast or death). The helper's environment = the main process's minus the deterministic recipe's seed-hook switch (it runs no TensorFlow op).
multiprocessing's `spawn` is not used: it re-imports the CLI script as __mp_main__. Stdlib only at import."""
import os, sys, pickle, select, struct, subprocess

KINDS = ("counts", "jsd")
STATE = {"workers": None, "note": None}
_HERE = os.path.dirname(os.path.abspath(__file__))


def _write_all(fd, data):
    view = memoryview(data); n = 0
    while n < len(view):
        n += os.write(fd, view[n:])


def _read_exact(fd, n):
    chunks = []; got = 0
    while got < n:
        b = os.read(fd, n - got)
        if not b:
            raise EOFError
        chunks.append(b); got += len(b)
    return b"".join(chunks)


def send_frame(fd, obj):
    """One frame = 8-byte big-endian length + the pickle (whole frames: a pipe read/write may be short past the pipe buffer, 64 KiB)."""
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    _write_all(fd, struct.pack("!Q", len(data)) + data)


def recv_frame(fd):
    (n,) = struct.unpack("!Q", _read_exact(fd, 8))
    return pickle.loads(_read_exact(fd, n))


def helper_main(kind, fd_in, fd_out):
    """The helper's loop (a fresh interpreter): read job frames from fd_in, run the stock's own call, write reply frames to fd_out.
    The helper lowers its own CPU priority first (its imports overlap the main process's model load; the main process's load wins the contention) and
    leaves with os._exit after the loop (the PNGs are on disk; no interpreter teardown for the main process's stop() to wait on)."""
    try:
        os.nice(5)
    except Exception:
        pass
    try:
        sys.stdin = open(os.devnull, "r")
    except Exception:
        pass
    reply = lambda obj: send_frame(fd_out, obj)
    try:
        import chrombpnet.training.metrics as metrics                         # the stock's metrics module (numpy, scipy, matplotlib): the helper's own import
    except BaseException as e:  # noqa: BLE001 — reported to the main process (it raises), never a traceback of the helper's own
        reply(("err", "the helper could not import the stock's metrics module: %s: %s" % (type(e).__name__, str(e)[:200]))); return
    reply(("ready", kind))
    while True:
        try:
            job = recv_frame(fd_in)
        except EOFError:
            os._exit(0)
        if job is None:
            os._exit(0)
        try:
            if kind == "counts":
                labels, preds, output_prefix, title = job
                sp, pe, mse = metrics.counts_metrics(labels, preds, output_prefix, title)
                reply(("ok", (float(sp), float(pe), float(mse))))                 # float64 exact through the pipe (no text round-trip)
            else:
                jsd_pw, jsd_rnd, output_prefix, title = job
                metrics.plot_histogram(jsd_pw, jsd_rnd, output_prefix, title)
                reply(("ok", None))
        except BaseException as e:  # noqa: BLE001 — the main process decides (it raises)
            reply(("err", "%s: %s" % (type(e).__name__, str(e)[:300])))


class _Helper:
    def __init__(self, kind, env):
        self.kind = kind
        r_job, w_job = os.pipe(); r_rep, w_rep = os.pipe()
        code = "import sys; sys.path.insert(0, %r); from chrombpnet_fastkit import png_workers; png_workers.helper_main(%r, %d, %d)" % (os.path.dirname(_HERE), kind, r_job, w_rep)
        self.proc = subprocess.Popen([sys.executable, "-c", code], pass_fds=(r_job, w_rep), env=env, stdin=subprocess.DEVNULL, close_fds=True)
        os.close(r_job); os.close(w_rep)
        self.fd_job = w_job; self.fd_rep = r_rep
        self.pending = 0; self.ready = False

    def _recv(self, timeout):
        if not select.select([self.fd_rep], [], [], timeout)[0]:
            raise RuntimeError("[png_workers] the %s helper gave no reply within %.0f s (alive=%s) — refused (no fallback)" % (self.kind, timeout, self.proc.poll() is None))
        try:
            return recv_frame(self.fd_rep)
        except EOFError:
            raise RuntimeError("[png_workers] the %s helper closed its pipe (exit %s) — refused (no fallback)" % (self.kind, self.proc.poll()))

    def submit(self, args):
        if self.proc.poll() is not None:
            raise RuntimeError("[png_workers] the %s helper is not alive (exit %s) — refused (no fallback)" % (self.kind, self.proc.returncode))
        if not self.ready:
            status, payload = self._recv(900.0)
            if status != "ready":
                raise RuntimeError("[png_workers] the %s helper did not report ready (%s: %s) — refused (no fallback)" % (self.kind, status, payload))
            self.ready = True
        send_frame(self.fd_job, tuple(args)); self.pending += 1

    def result(self, timeout=900.0):
        if not self.pending:
            raise RuntimeError("[png_workers] result() without a pending %s job" % self.kind)
        status, payload = self._recv(timeout); self.pending -= 1
        if status != "ok":
            raise RuntimeError("[png_workers] the %s helper failed: %s — refused (no fallback)" % (self.kind, payload))
        return payload

    def stop(self):
        try:
            if self.fd_job >= 0:
                send_frame(self.fd_job, None); os.close(self.fd_job); self.fd_job = -1
        except Exception:
            pass
        try:
            self.proc.wait(5)
        except Exception:
            self.proc.kill()
        try:
            os.close(self.fd_rep)
        except Exception:
            pass


def start():
    """Spawn the two helpers now (idempotent). Their environment: the main process's minus the deterministic recipe's seed-hook switch."""
    if STATE["workers"] is not None:
        return STATE["workers"]
    env = dict(os.environ); env.pop("CHROMBPNET_DET_SUBPROCESS", None)
    STATE["workers"] = {k: _Helper(k, env) for k in KINDS}
    STATE["note"] = "two helpers spawned at process start (fresh interpreters, no fork of this process; pids %s)" % ",".join(str(w.proc.pid) for w in STATE["workers"].values())
    return STATE["workers"]


def require():
    """The helpers, or a refusal: the finish stage never forks after the runtimes exist."""
    if STATE["workers"] is None:
        raise RuntimeError("[png_workers] the PNG helpers were not started at process start (pred_bw_fast.py starts them with -bw) — refused (no fork after the runtimes)")
    return STATE["workers"]


def stop():
    """The end-of-process None to BOTH helpers first, then the waits (they leave in parallel)."""
    w = STATE["workers"]
    if w is None:
        return
    for h in w.values():
        try:
            send_frame(h.fd_job, None); os.close(h.fd_job); h.fd_job = -1
        except Exception:
            pass
    for h in w.values():
        h.stop()
    STATE["workers"] = None


def stamp():
    w = STATE["workers"]
    return {"shape": "two helpers spawned at process start (fresh interpreters; no fork of the main process)", "started": STATE["note"],
            "alive": ({k: (h.proc.poll() is None) for k, h in w.items()} if w else None)}
