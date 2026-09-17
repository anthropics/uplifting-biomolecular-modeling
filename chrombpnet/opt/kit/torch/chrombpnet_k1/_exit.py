"""chrombpnet_k1._exit — exit_fast(): leave the process with os._exit(0) once EVERY output and the stamp are on disk (re-opened, non-empty and
fsync'd here). What it skips is the interpreter's teardown of a multi-GB process — object GC, CUDA context destruction, shared-library unload —
which is pure wall time after the results are durable. The call belongs ONLY on the all-outputs-written success path, as the caller's last
statement; every failure path (an excepthook, a writer-join sentinel, any non-zero rc) reaches the shell exactly as without it.
K1_EXIT_TEARDOWN=1 keeps the interpreter's normal exit (a debug arm, not a tuning knob)."""
import os, sys, json

def exit_fast(stamp_path, outputs=(), env="K1_EXIT_TEARDOWN"):
    """Checks first: the stamp parses as JSON from disk; every path in `outputs` (and the stamp) exists, is non-empty and is fsync'd through a
    read-only descriptor, so the data is durable before the exit. Then: env == '1' -> print and return 'teardown' (the interpreter exits
    normally); else flush stdout/stderr and os._exit(0). A missing, empty or truncated file RAISES (the process then leaves through the
    caller's error path with rc != 0, never through _exit)."""
    with open(stamp_path) as f: json.load(f)                                       # a truncated stamp raises here (JSONDecodeError)
    for p in list(outputs) + [stamp_path]:
        if not os.path.isfile(p) or os.path.getsize(p) == 0: raise RuntimeError(f"[chrombpnet_k1] exit_fast: output missing or empty before the exit: {p}")
        fd = os.open(p, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    if os.environ.get(env, "0") == "1":
        print(f"[chrombpnet_k1] exit: interpreter teardown kept ({env}=1; debug arm)", flush=True); return "teardown"
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
