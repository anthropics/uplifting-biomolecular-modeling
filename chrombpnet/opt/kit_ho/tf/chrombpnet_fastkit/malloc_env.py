"""The malloc env arm: three glibc malloc tunables applied to THIS process through mallopt (ctypes; M_MMAP_MAX=-4, M_TRIM_THRESHOLD=-1,
M_TOP_PAD=-2 — glibc's own constants) and exported as MALLOC_MMAP_MAX_ / MALLOC_TRIM_THRESHOLD_ / MALLOC_TOP_PAD_ so any exec'd child (the
spawned writer) reads the same values at its start; a forked child inherits the parent's arenas as tuned. No re-exec (a re-exec loses the
interpreter's own flags and a -c argv). Allocation behaviour only — no numerics are touched. Every run record carries the malloc_env stamp.
Pure stdlib; imported FIRST by the entry script, before numpy / TensorFlow."""
import os, sys

TUNABLES = {"MALLOC_MMAP_MAX_": "0", "MALLOC_TRIM_THRESHOLD_": "-1", "MALLOC_TOP_PAD_": "536870912"}   # 512 MB = 536870912 bytes
_MALLOPT = {"MALLOC_MMAP_MAX_": (-4, 0), "MALLOC_TRIM_THRESHOLD_": (-1, -1), "MALLOC_TOP_PAD_": (-2, 536870912)}   # glibc malloc.h: M_TRIM_THRESHOLD -1, M_TOP_PAD -2, M_MMAP_MAX -4

def plan():
    present = {k: os.environ.get(k) for k in TUNABLES}
    if all(present[k] == v for k, v in TUNABLES.items()):
        return {"active": True, "present": present, "source": "explicit env at process start (the user's own knob) + mallopt re-applied by the launcher"}
    return {"active": True, "present": present, "source": "launcher: mallopt in-process + the env exported for exec'd children"}

def ensure():
    """Apply the tunables (mallopt) and export the env; refuse by name if mallopt is unavailable (no silent fallback). Returns the stamp."""
    p = plan()
    if not p["active"]:
        return {"tunables": p["present"], "active": False, "source": p["source"], "mallopt": None}
    import ctypes, ctypes.util
    libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
    libc.mallopt.argtypes = (ctypes.c_int, ctypes.c_int); libc.mallopt.restype = ctypes.c_int
    rets = {}
    for k, (param, val) in _MALLOPT.items():
        rets[k] = int(libc.mallopt(param, val))
        if rets[k] != 1: raise SystemExit(f"[chrombpnet_fastkit.malloc_env] mallopt({param}, {val}) returned {rets[k]} — the malloc arm cannot be applied on this libc (refused)")
    for k, v in TUNABLES.items(): os.environ[k] = v
    return {"tunables": dict(TUNABLES), "active": True, "source": p["source"], "mallopt": rets}

def stamp():
    p = plan(); return {"tunables": p["present"], "active": p["active"], "source": p["source"]}
