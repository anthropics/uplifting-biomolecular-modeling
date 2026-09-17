"""``warm``: one small documented-line job on a staged input — the K1 route's Triton cache and the TF route's driver cache warmed on
this machine, and the whole pipeline exercised end to end. The job is `pred_bw --mode fast` (cli.py) over the first N rows of a regions file
(`--regions`, default CHROMBPNET_OPT_REGIONS; N = `--n`, default the stock batch size, stock parsers.py:223) with the config's model,
genome, chrom sizes and observed bigWig (CHROMBPNET_OPT_MODEL / _GENOME / _CHROM_SIZES / _BIGWIG). Prints one `WARM PASS|FAIL` line.
The exit follows the pred_bw step's rule: its 3 (not active, or a partial activation) is warm's 3.
"""
import os
import subprocess
import sys
import tempfile
import time
from typing import Optional

from . import report as _report, stack

DATA_ENV = {"model": "CHROMBPNET_OPT_MODEL", "genome": "CHROMBPNET_OPT_GENOME", "regions": "CHROMBPNET_OPT_REGIONS",
            "chrom_sizes": "CHROMBPNET_OPT_CHROM_SIZES", "bigwig": "CHROMBPNET_OPT_BIGWIG"}   # the config's data variables (configs/*.env)

STOCK_BATCH_DEFAULT = 64                      # stock chrombpnet/parsers.py:223 (-bs default)
OUTPUTS = ("_chrombpnet.bw", "_chrombpnet_preds.bed")   # the outputs every job writes (pred_bw_fast.py's module text)


def run(mode: str = "fast", regions: Optional[str] = None, n: Optional[int] = None, out_dir: Optional[str] = None, timeout: float = 1800.0,
        stream=None) -> dict:
    stream = stream or sys.stderr
    t0 = time.perf_counter()
    try:
        stack.kit_home()
    except stack.ActivationError as e:
        _report.print_mode_line({"active": False, "mode": mode, "reason": str(e)}); return {"ok": False, "inactive": True, "reason": str(e)}
    data = {k: os.environ.get(v) for k, v in DATA_ENV.items()}
    regions = regions or data.get("regions")
    missing = [DATA_ENV[k] for k in ("model", "genome", "chrom_sizes") if not data.get(k)] + ([] if regions else [DATA_ENV["regions"]])
    if missing:
        print("{} WARM FAIL reason=unset: {}".format(_report.PREFIX, ",".join(missing)), file=stream, flush=True)
        return {"ok": False, "reason": "unset: " + ",".join(missing)}
    n = int(n or STOCK_BATCH_DEFAULT)
    out_dir = os.path.abspath(out_dir or tempfile.mkdtemp(prefix="chrombpnet_opt_warm_"))
    os.makedirs(out_dir, exist_ok=True)
    bed = os.path.join(out_dir, "warm_regions.bed")
    opener = __import__("gzip").open if regions.endswith(".gz") else open
    with opener(regions, "rt") as fh, open(bed, "w") as out:
        for i, line in enumerate(fh):
            if i >= n:
                break
            out.write(line)
    prefix = os.path.join(out_dir, "warm")
    argv = [sys.executable, "-m", "chrombpnet_opt", "pred_bw", "--mode", mode] + \
           ["-cm", data["model"], "-r", bed, "-g", data["genome"], "-c", data["chrom_sizes"], "-op", prefix, "-os", prefix + "_stats"] + \
           (["-bw", data["bigwig"]] if data.get("bigwig") else [])
    env = dict(os.environ); env.pop(stack.ENV, None)               # the mode is in argv
    try:
        rc = subprocess.run(argv, env=env, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        rc = None
    ok = rc == 0 and all(os.path.isfile(prefix + s) and os.path.getsize(prefix + s) > 0 for s in OUTPUTS)
    inactive = rc == 3                                            # the pred_bw step's own rule: not active (refused by name), or a partial activation
    line = "{} WARM {} mode={} n={} rc={} wall={:.0f}s out={}".format(_report.PREFIX, "PASS" if ok else "FAIL", mode, n, rc if rc is not None else "timed_out",
                                                                  time.perf_counter() - t0, out_dir)
    print(line, file=stream, flush=True)
    return {"ok": ok, "inactive": inactive, "rc": rc, "out_dir": out_dir, "line": line, "argv": argv}
