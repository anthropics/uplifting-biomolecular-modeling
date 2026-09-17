#!/usr/bin/env python3
"""make_cells.py -- writes ../CELLS.json: the MEASURED cells of this package's router, one record per measured case, keyed by compute
capability.  Nothing is re-measured here; the inputs are timing result files (schema triattn_bench/v2: env, args, contenders, rows).

    python tools/make_cells.py carry  --from <CELLS.json of the previous generation>          start ../CELLS.json from a previous generation's cells
                                                                                              (records copied as they are; used for the cc-9.0 cells,
                                                                                              which generation 11 does not re-measure)
    python tools/make_cells.py append --cc 8.0 --runs a.json b.json ... [--route-census r.json] [--replace] [--pkg-contender sm80] [--strip-prefix <p>,<q>]
                                                                                              add the cells of one cc from harness result files made on
                                                                                              a device of that cc (--replace drops that cc's cells first)
    python tools/make_cells.py show                                                           one line per unmasked starting-node scoreboard cell

Per cell: key (cc, dtype, H, D, B, N pair rows, S keys, node start | end (transposed operands made contiguous by the harness) | endstrided (transposed
views, no copy), mask kind + pad fraction, bias class, grid = the result file's name, variant), the route the package takes for the case (from a
route census file when given, else from the routing table), per contender the CUDA-graph-replay median / min ms (`timing`, the primary
timing) and the eager median (`timing_eager`), status, and error vs the fp64 reference (max_abs, ulpe_max = max error in excess of the
bf16 output-rounding floor, in ulp); ratios contender_ms / package_ms (> 1 = the package is faster); the source file name, case id and run time.
Contender names: pkg (= contender `best` in the result files, the package's router), cueq (cuequivariance_ops_torch triangle_attention), ours_k2b (the Triton
kernel fpf_triatt_k2b of opt_core), ours_v9 (opt_core flash_triattn v9), sdpa
(torch scaled_dot_product_attention per pair row) -- whichever the result files hold.  Machine identifiers in the result files are not copied.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(PKG, "CELLS.json")
SCHEMA = "triattn_pkg_cells/v2"
CONTENDERS = {"best": "pkg", "cueq": "cueq", "ours_k2b": "ours_k2b", "ours_v9": "ours_v9", "sdpa": "sdpa", "tri": "tri"}   # result-file name -> cell name; + k2b_<cell> tuned Triton variants, kept under their own names (see _contender_name)
RATIO_OVER_PKG = ("cueq", "ours_k2b", "ours_v9", "sdpa", "tri")                  # + every k2b_<cell> present
ERR_KEYS = ("max_abs", "rel_fro", "ulp_max", "ulpe_max", "ulpe_rms", "n_nonfinite")
WITHHELD = "withheld"


def r3(x):
    return None if x is None else round(float(x), 4)


def _contender_name(hname: str, pkg_alias: str):
    """cell name of a harness contender: the package's route measured under `pkg_alias` (default `best`; on a device whose every routed cell is one
    kernel the harness may time that kernel's own plug-in, e.g. `sm80` on cc 8.0) -> 'pkg'; tuned K2B cells `k2b_<cell>` keep their names."""
    if hname == pkg_alias:
        return "pkg"
    if hname.startswith("k2b_"):
        return hname
    return CONTENDERS.get(hname)


def _grid_name(path: str, strip: str):
    g = os.path.splitext(os.path.basename(path))[0]
    for pre in [x for x in (strip or "").split(",") if x]:
        if g.startswith(pre):
            g = g[len(pre):]
    return g


def _package():
    sys.path.insert(0, PKG)
    from triattn_pkg._face import BEST_COMMIT, PINS                       # no torch needed for these
    import triattn_pkg
    return {"version": triattn_pkg.__version__, "best_commit": BEST_COMMIT, "pins": {k: list(v) for k, v in PINS.items()}, "harness_contender": "best (candidate:best:bf16bias)"}


def _empty():
    return {"schema": SCHEMA,
            "what": "Measured triangle-attention forward cells for this package's router (`best`) and its contenders, keyed by compute capability (cc 9.0 = H100, "
                    "cc 8.0 = A100); ms = CUDA-graph-replay median per call (ms_min: min; ms_eager: back-to-back eager median, warm L2); ratio = contender ms / package ms "
                    "(> 1: package faster); err_vs_fp64 from the harness's fp64 reference (ulpe_max = max error above the bf16 rounding floor, in output ulp). One record per "
                    "harness case; nothing here is re-measured. by_cc holds the hardware, contender versions and source files per cc.",
            "package": _package(), "by_cc": {}, "n_cells": 0, "cells": []}


def _load():
    return json.load(open(OUT)) if os.path.isfile(OUT) else _empty()


def _write(d):
    d["package"] = _package()
    d["n_cells"] = len(d["cells"])
    for cc, sec in d["by_cc"].items():
        sec["n_cells"] = sum(1 for c in d["cells"] if c["key"]["cc"] == cc)
    json.dump(d, open(OUT, "w"), indent=1); open(OUT, "a").write("\n")
    print("CELLS.json: %d cells (%s)" % (d["n_cells"], ", ".join("cc %s: %d" % (cc, s["n_cells"]) for cc, s in sorted(d["by_cc"].items()))))


def cmd_carry(a):
    src = json.load(open(a.src))
    d = _empty()
    if src.get("schema") == "triattn_pkg_cells/v1":                        # one cc per file
        cc = src["hardware"]["cc"]
        d["by_cc"][cc] = {"hardware": src["hardware"], "contenders": src["contenders"], "sources": src["sources"], "harness_contender": src["package"].get("harness_contender"),
                          "generation": src["package"]["version"], "note": "carried from generation %s's CELLS.json unchanged (that generation's router on these cells == this one's)" % src["package"]["version"]}
        d["cells"] = list(src["cells"])
    elif src.get("schema") == SCHEMA:
        d["by_cc"] = src["by_cc"]; d["cells"] = list(src["cells"])
    else:
        sys.exit("make_cells carry: unknown schema %r" % src.get("schema"))
    _write(d)


def _static_route(cc, case):
    """The router's table restated for cells without a route census (cc 8.0: bf16 D 16/32/64 -> cuda_80, else tri)."""
    if cc == "8.0":
        return "cuda_80" if (case["dtype"] == "bf16" and case["D"] in (16, 32, 64)) else "tri"
    return "not in route census"


def cmd_append(a):
    d = _load()
    if d.get("schema") != SCHEMA:
        sys.exit("make_cells append: ../CELLS.json has schema %r; run carry first" % d.get("schema"))
    cc = a.cc
    if a.replace:
        d["cells"] = [c for c in d["cells"] if c["key"]["cc"] != cc]
    census = json.load(open(a.route_census)) if a.route_census else {}
    sec = d["by_cc"].setdefault(cc, {"hardware": None, "contenders": {}, "sources": {}, "harness_contender": "best (candidate:best:bf16bias)", "generation": _package()["version"]})
    if a.pkg_contender != "best":
        sec["harness_contender"] = (a.pkg_contender + " (" + (a.pkg_contender_desc or "the route's kernel plug-in, timed directly; the router adds a device-capability lookup and shape checks per call")
                                    + "); cells the router sends to tri carry the tri plug-in's numbers as pkg")
    have = {(c["key"]["cc"], c["source"]["grid_file"], c["source"]["case_id"]) for c in d["cells"]}
    n_new = 0
    for path in a.runs:
        r = json.load(open(path))
        if r.get("schema") != "triattn_bench/v2":
            sys.exit("make_cells append: %s has schema %r (want triattn_bench/v2)" % (path, r.get("schema")))
        env = r["env"]; grid = _grid_name(path, a.strip_prefix)
        rcc = "%d.%d" % tuple(env["cc"])
        if rcc != cc:
            sys.exit("make_cells append: %s was made on cc %s, not %s" % (path, rcc, cc))
        sec["sources"][grid] = {"file": grid + ".json", "time_utc": env.get("time_utc"), "kernel": r["args"].get("kernel"), "n_rows": len(r["rows"]), "clocks_overall": r.get("clocks_overall"),
                                "protocol": {k: r["args"].get(k) for k in ("protocol", "rounds", "block_s", "min_total", "warmup", "eager_iters", "flush_l2", "seed", "check", "full_ref_max_n", "sample_rows")}}
        if sec["hardware"] is None:
            sec["hardware"] = {"cc": rcc, "gpu": env.get("gpu"), "nvidia_smi": env.get("nvidia_smi"), "torch": env.get("torch"), "torch_cuda": env.get("torch_cuda"),
                               "python": env.get("python"), "triton": env.get("triton"), "packages": env.get("packages") or env.get("versions")}
        for name, c in (r.get("contenders") or {}).items():
            pname = _contender_name(name, a.pkg_contender)
            if pname and pname not in sec["contenders"]:
                sec["contenders"][pname] = {k: c.get(k) for k in ("desc", "version", "takes_strided", "prefix_masks_only", "bias_staging", "lossy_fp32_bias", "notes")}
        for row in r["rows"]:
            case = row["case"]; res = row["results"]
            if (cc, grid + ".json", case["id"]) in have:
                continue
            S = case.get("S") or case["N"]
            ms, ms_min, ms_eager, status, err = {}, {}, {}, {}, {}
            route_static = _static_route(cc, case)
            pkg_alias = a.pkg_contender
            if a.pkg_contender != "best" and route_static == "tri" and "tri" in res:
                pkg_alias = "tri"                                        # the router sends this cell to tri: the package's number is the tri plug-in's, not the cc's CUDA kernel's
            to_pkg = [h for h in res if _contender_name(h, pkg_alias) == "pkg" or (h == "tri" and pkg_alias == "tri")]
            if len(to_pkg) > 1:                                          # e.g. a run that timed both `best` and the --pkg-contender plug-in: name one, never merge silently
                raise SystemExit(f"append: case {case['id']} in {os.path.basename(path)}: contenders {to_pkg} would all be the package column (--pkg-contender {a.pkg_contender}); "
                                 "re-run with the one that is the package's route on this cc, or drop the other from the result file")
            for hname, x in res.items():
                pname = _contender_name(hname, pkg_alias)
                if hname == "tri" and pkg_alias == "tri":
                    pname = "pkg"
                if pname is None or x is None:
                    continue
                status[pname] = str(x.get("status")) + ((": " + x["note"]) if x.get("note") else "")
                tm = x.get("timing") or {}
                ms[pname] = r3(tm.get("median_ms")); ms_min[pname] = r3(tm.get("min_ms")); ms_eager[pname] = r3((x.get("timing_eager") or {}).get("median_ms"))
                e = (x.get("check") or {}).get("err") or {}
                err[pname] = {k: (r3(e.get(k)) if k != "n_nonfinite" else e.get(k)) for k in ERR_KEYS} if e else None
                if x.get("check") and "deterministic" in x["check"]:
                    err[pname] = dict(err[pname] or {}, deterministic=x["check"]["deterministic"])
            p = ms.get("pkg")
            ratios = {o + "_over_pkg": (r3(ms[o] / p) if (p and ms.get(o)) else None) for o in list(RATIO_OVER_PKG) + sorted(k for k in ms if k.startswith("k2b_")) if o in ms}
            cen = census.get(case["id"]) or {}
            route = cen.get("route") or (res.get("best") or {}).get("route") or _static_route(cc, case)
            d["cells"].append({
                "key": {"cc": cc, "dtype": case["dtype"], "H": case["H"], "D": case["D"], "B": case["B"], "N": case["N"], "S": S, "node": case["node"],
                        "mask": case["mask"], "pad_frac": case.get("pad_frac") if case["mask"] != "none" else None, "bias": case["bias"], "grid": grid, "variant": case.get("variant", "plain")},
                "route": route,
                "ms": ms, "ms_min": ms_min, "ms_eager": ms_eager, "ratio": ratios, "status": status, "err_vs_fp64": err,
                "source": {"grid_file": grid + ".json", "case_id": case["id"], "box_id": WITHHELD, "time_utc": env.get("time_utc"), "ref": row.get("ref")},
            })
            n_new += 1
    print("append: %d new cc-%s cells from %d file(s)" % (n_new, cc, len(a.runs)))
    _write(d)


def cmd_show(a):
    d = _load()
    for c in d["cells"]:
        k = c["key"]
        if k["grid"] in ("scoreboard",) or (k["mask"] == "none" and k["node"] == "start" and k["B"] == 1):
            if k["mask"] == "none" and k["node"] == "start":
                rt = c.get("ratio", {})
                print("  cc %s %s H%d D%d N=%-5d S=%-5d route %-8s pkg %8s ms  cueq x%-6s k2b x%-6s cute x%-6s" % (k["cc"], k["dtype"], k["H"], k["D"], k["N"], k["S"], c["route"],
                      c["ms"].get("pkg"), rt.get("cueq_over_pkg"), rt.get("ours_k2b_over_pkg")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("carry"); c.add_argument("--from", dest="src", required=True)
    p = sub.add_parser("append"); p.add_argument("--cc", required=True); p.add_argument("--runs", nargs="+", required=True); p.add_argument("--route-census", default=None); p.add_argument("--replace", action="store_true")
    p.add_argument("--pkg-contender", default="best", help="harness contender that is the package's route on this cc (default best; e.g. sm80 when the harness timed the cuda_80 kernel's own plug-in)")
    p.add_argument("--pkg-contender-desc", default="", help="one line recorded as by_cc[cc].harness_contender")
    p.add_argument("--strip-prefix", default="", help="comma-separated file-name prefixes dropped from grid names (result files named <prefix><grid>.json)")
    sub.add_parser("show")
    a = ap.parse_args()
    {"carry": cmd_carry, "append": cmd_append, "show": cmd_show}[a.cmd](a)


if __name__ == "__main__":
    main()
