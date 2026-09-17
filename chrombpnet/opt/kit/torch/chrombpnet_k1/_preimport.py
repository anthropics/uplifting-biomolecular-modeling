"""chrombpnet_k1._preimport — import the stage-2 module chain (scipy.stats, matplotlib + pyplot on the Agg backend, chrombpnet.training.metrics,
chrombpnet's data_utils) on a DAEMON THREAD while stage 1 waits on the GPU / IO (torch's C++ ops release the GIL), so the single-process CLI's
stage 2 finds them already in sys.modules. Imports only — no output is touched. A module that fails to import is RECORDED in STATE["errors"], never
raised into the main thread (stage 2 raises its own ImportError if it truly needs the module). start() imports numpy and pandas on the main thread
first so the two threads never wait on each other's import locks; join() records whether the chain finished under stage 1 ('overlapped') and its wall."""
import sys, time, threading, importlib
STAGE2_CHAIN = ("scipy.stats", "matplotlib", "matplotlib.pyplot", "chrombpnet.training.metrics", "chrombpnet.training.utils.data_utils")
STATE = {"started": None, "done": None, "walls_s": {}, "errors": {}, "thread": None, "overlapped": None}

def _run(modules):
    for m in modules:
        t0 = time.time()
        try:
            if m == "matplotlib.pyplot":
                import matplotlib; matplotlib.use("Agg", force=False)               # the headless backend the metrics stage uses; no display
            importlib.import_module(m); STATE["walls_s"][m] = round(time.time() - t0, 3)
        except Exception as e: STATE["errors"][m] = f"{type(e).__name__}: {str(e)[:120]}"
    STATE["done"] = time.time()

def start(modules=STAGE2_CHAIN):
    """Idempotent. Starts the daemon thread; returns STATE."""
    if STATE["thread"] is not None: return STATE
    for root in ("numpy", "pandas"): importlib.import_module(root)             # the SHARED roots imported on the MAIN thread first: stage 1 never waits on a module lock the thread holds while the thread waits on one stage 1 holds (importlib's cross-thread deadlock)
    STATE["started"] = time.time(); t = threading.Thread(target=_run, args=(tuple(modules),), name="chrombpnet_k1_preimport", daemon=True); STATE["thread"] = t; t.start(); return STATE

def join(timeout=None):
    """Called at stage-2 start: waits for the thread (bounded by timeout), records 'overlapped' (= the chain had finished before this call) and a
    one-line summary in STATE['line'] for the kit's status line."""
    t = STATE["thread"]
    if t is None: STATE["line"] = "preimport: not started"; return STATE
    STATE["overlapped"] = STATE["done"] is not None
    t.join(timeout); tot = round((STATE["done"] or time.time()) - STATE["started"], 3)
    STATE["line"] = (f"preimport: {len(STATE['walls_s'])}/{len(STATE['walls_s']) + len(STATE['errors'])} stage-2 modules on a thread in {tot} s, "
                     + ("finished under stage 1 (overlapped)" if STATE["overlapped"] else "still running at stage 2 (partially overlapped)")
                     + (f"; import errors {STATE['errors']}" if STATE["errors"] else ""))
    return STATE
