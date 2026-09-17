#!/usr/bin/env python3
"""test_pkg.py -- checks triattn_pkg against the shipped test vectors (testvectors/manifest.json + <case>.expected.pt + <case>.ref_rows.pt).

Every case in testvectors/cases.py names the compute capabilities it is a vector for; this script runs the cases of THIS device's cc (cc 9.0: the
17 H100 vectors; cc 8.0: the a80_* A100 vectors) and never compares a device against another architecture's bytes.

    python test_pkg.py                       every case of this device's cc: inputs regenerated (digests must match the manifest), output of
                                             triattn_pkg.triangle_attention (1) BITWISE == the expected bytes, (2) GATE vs the fp64 reference recomputed
                                             here: finite, and max |out - ref| <= 1.5 x the error recorded at generation, (3) route == the recorded
                                             route; stored fp64 reference rows must equal the recomputed ones to 1e-6 relative (reference integrity).
                                             A case of this cc whose expected vector is not recorded yet FAILS by name (`expected_not_recorded`).
                                             Exit 0 = all pass.
    python test_pkg.py --only s1024_none,s2048_prefixvar      a subset (ids of this device's cc)
    python test_pkg.py --list [--cc 8.0]                      the case table (all, or one cc) with what the manifest records per case; no GPU needed
    python test_pkg.py --impl dispatch:<dir>                  run another implementation of the same contract against the vectors: <dir>/candidate.py `best`
                                                              (a live router tree) or --impl <module>:<function> importable from --path entries
    TRIATTN_PKG_PREBUILT=never python test_pkg.py             the package with its CUDA extensions JIT-built here instead of the prebuilt binaries
    python test_pkg.py --generate [--only IDS] [--note TEXT]  (maintainers) rewrite the vectors OF THIS DEVICE'S cc from the package's outputs: expected
                                                              outputs, reference rows, manifest records; the other cc's records and files are left untouched
                                                              and ids of another cc are refused
    python test_pkg.py --record-inputs                        (maintainers, any machine with torch) record the input digests of cases the manifest does not
                                                              know yet (no outputs; marks them `expected: null` until --generate runs on their device)
    python test_pkg.py --record-loadcheck [--ext NAME,...]    (maintainers, on the device of the extension) fill the `loadcheck` field of prebuilt build
                                                              records of this stack that say "pending": {case id: sha256/16 of the output bytes}, after
                                                              checking those bytes equal the expected vectors (TRIATTN_PKG_PREBUILT=always)

Bitwise equality is expected on the device class a vector was made on with the routed kernels of this package; with another implementation or another
triton the gate and the digests still apply and a bitwise difference is reported per case, not hidden.  Without a CUDA device only the input digests
are checked (named skip).
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TV = os.path.join(HERE, "testvectors")
sys.path.insert(0, HERE)                       # triattn_pkg
sys.path.insert(0, TV)                         # cases
import torch  # noqa: E402

import cases as C  # noqa: E402

GATE_FACTOR = 1.5
REF_INTEGRITY_RTOL = 1e-6
SCHEMA = "triattn_pkg_testvectors/v2"
MANIFEST = os.path.join(TV, "manifest.json")
# the byte-gate cases a prebuilt extension's build record names in its `loadcheck` field (cases the extension itself serves on its device)
LOADCHECK_CASES = {"triattn_sm90_ext": ("s384_endstrided",), "triattn_m1_ext": ("s512_prefixvar", "s1024_prefix"), "triattn_mw_ext_g3x4": ("s3584_prefix",),
                   "triattn_sm80_ext": ("a80_s512_prefixvar", "a80_s1024_endstrided", "a80_d64_s512_none")}
EXT_CC = {"triattn_sm90_ext": "9.0", "triattn_m1_ext": "9.0", "triattn_mw_ext_g3x4": "9.0", "triattn_sm80_ext": "8.0"}


def load_impl(spec, paths):
    """-> (callable(q,k,v,bias,mask,scale), route callable or None, label)."""
    for p in paths:
        if p and p not in sys.path:
            sys.path.insert(0, p)
    if spec == "pkg":
        import triattn_pkg as P
        return P.triangle_attention, P.route, "triattn_pkg v%s (best %s) %s" % (P.__version__, P.BEST_COMMIT, os.environ.get("TRIATTN_PKG_PREBUILT", "auto"))
    kind, _, rest = spec.partition(":")
    if kind == "dispatch":
        s = importlib.util.spec_from_file_location("dispatch_candidate_live", os.path.join(rest, "candidate.py"))
        m = importlib.util.module_from_spec(s); sys.modules["dispatch_candidate_live"] = m; s.loader.exec_module(m)
        return (lambda q, k, v, bias, mask=None, scale=None: m.best(q, k, v, bias, mask=mask, scale=scale)), m.route, "dispatch:%s" % rest
    m = importlib.import_module(kind)
    return getattr(m, rest), None, spec


def env_record():
    """The interpreter / device facts recorded with generated vectors (versions and the device name; no machine identifiers)."""
    rec = {"torch": torch.__version__, "torch_cuda": torch.version.cuda, "python": sys.version.split()[0], "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        import triton; rec["triton"] = triton.__version__
    except Exception:  # noqa: BLE001
        rec["triton"] = None
    if torch.cuda.is_available():
        rec["gpu"] = torch.cuda.get_device_name(0); rec["cc"] = list(torch.cuda.get_device_capability(0))
        try:
            rec["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,clocks.max.sm", "--format=csv,noheader"],
                                               capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:  # noqa: BLE001
            pass
    try:
        import triattn_pkg as P
        rec["stack_tag"] = P.stack_tag(); rec["pkg_version"] = P.__version__; rec["best_commit"] = P.BEST_COMMIT; rec["pins"] = P.PINS
    except Exception as e:  # noqa: BLE001
        rec["stack_tag"] = "unavailable: %s" % e
    try:
        rec["tree_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True, timeout=20).stdout.strip() or None
    except Exception:  # noqa: BLE001
        rec["tree_commit"] = None
    return rec


def expected_path(cid):
    return os.path.join(TV, cid + ".expected.pt")


def ref_path(cid):
    return os.path.join(TV, cid + ".ref_rows.pt")


def load_manifest():
    man = json.load(open(MANIFEST))
    if man.get("schema") != SCHEMA:
        sys.exit("test_pkg: manifest schema %r, expected %r" % (man.get("schema"), SCHEMA))
    return man


def write_manifest(man):
    order = [c.id for c in C.CASES]
    man["cases"] = sorted(man["cases"], key=lambda c: order.index(c["id"]) if c["id"] in order else len(order))
    json.dump(man, open(MANIFEST, "w"), indent=1)
    open(MANIFEST, "a").write("\n")


def this_cc():
    return C.device_cc(0) if torch.cuda.is_available() else None


def run_case(case, fn, route_fn, device):
    t = C.make_inputs(case, device)
    dig = C.input_digests(t)
    torch.cuda.synchronize()
    out = fn(t["q"], t["k"], t["v"], t["bias"], t["mask"], t["scale"])
    torch.cuda.synchronize()
    out2 = fn(t["q"], t["k"], t["v"], t["bias"], t["mask"], t["scale"])          # run-to-run determinism in this process
    torch.cuda.synchronize()
    route = None
    if route_fn is not None:
        try:
            route = route_fn(t["q"], t["k"], t["v"], t["bias"], t["mask"])
        except Exception as e:  # noqa: BLE001
            route = "error:%s" % e
    return t, dig, out, bool(torch.equal(out, out2)), route


def cmd_list(a):
    man = load_manifest() if os.path.isfile(MANIFEST) else {"cases": []}
    by_id = {c["id"]: c for c in man["cases"]}
    ccs = [a.cc] if a.cc else C.CCS
    for cc in ccs:
        print("cc %s:" % cc)
        for case in C.cases_for_cc(cc):
            rec = by_id.get(case.id)
            state = "not in manifest" if rec is None else ("inputs only (expected: pending)" if not rec.get("expected") else
                                                           "route %-7s max|err| %.2e sha %s" % (rec.get("route"), rec["err_vs_fp64"]["max_abs"], rec["out_sha256"][:12]))
            print("  %-26s N=%-4d S=%-5d H=%-2d D=%-3d B=%d %-4s mask %-9s bias %-6s %-10s | %s" % (case.id, case.N, case.S, case.H, case.D, case.B, case.dtype, case.mask, case.bias, case.layout, state))
    return 0


def cmd_record_inputs(a):
    man = load_manifest()
    known = {c["id"] for c in man["cases"]}
    n = 0
    for case in C.CASES:
        if case.id in known:
            continue
        t = C.make_inputs(case, "cpu")
        man["cases"].append(dict(C.case_dict(case), inputs_sha256=C.input_digests(t), expected=None))
        n += 1
        print("[record-inputs] %-26s q %s .. mask %s" % (case.id, C.tensor_digest(t["q"])[:12], (C.tensor_digest(t["mask"]) or "-")[:12]), flush=True)
    man.setdefault("inputs_recorded", {})["torch"] = torch.__version__
    write_manifest(man)
    print("[record-inputs] %d new case(s); manifest has %d" % (n, len(man["cases"])))
    return 0


def generate(a):
    import triattn_pkg as P
    cc = this_cc()
    if cc is None:
        sys.exit("test_pkg --generate needs a CUDA device")
    mine = {c.id for c in C.cases_for_cc(cc)}
    only = set(a.only.split(",")) if a.only else None
    if only and (only - mine):
        sys.exit("test_pkg --generate: %s are not cc %s cases; their vectors are made on their own device class and are not touched here" % (sorted(only - mine), cc))
    todo = [c for c in C.cases_for_cc(cc) if not only or c.id in only]
    fn, route_fn, label = load_impl("pkg", [])
    dev = torch.device("cuda")
    man = load_manifest()
    by_id = {c["id"]: c for c in man["cases"]}
    made = env_record(); n = 0
    for case in todo:
        t0 = time.time()
        t, dig, out, det, route = run_case(case, fn, route_fn, dev)
        old = by_id.get(case.id)
        if old and old.get("inputs_sha256") and old["inputs_sha256"] != dig:
            sys.exit("test_pkg --generate: %s: inputs regenerated here differ from the recorded input digests (torch RNG changed?) -- refusing to write" % case.id)
        ref = C.reference_fp64(t)
        err = C.error_stats(out, ref)
        rec = dict(C.case_dict(case), inputs_sha256=dig, route=route, deterministic=det, out_shape=list(out.shape), out_dtype=str(out.dtype).replace("torch.", ""),
                   out_strides=list(out.stride()), out_sha256=C.tensor_digest(out), err_vs_fp64=err)
        if case.store == "full":
            torch.save(out.detach().contiguous().cpu(), expected_path(case.id)); rec["expected"] = {"file": case.id + ".expected.pt", "rows": "all"}
        else:
            torch.save(out.detach()[:, :C.DIGEST_ROWS].contiguous().cpu(), expected_path(case.id)); rec["expected"] = {"file": case.id + ".expected.pt", "rows": C.DIGEST_ROWS}
        torch.save(ref[:, :C.REF_ROWS].float().cpu(), ref_path(case.id)); rec["ref"] = {"file": case.id + ".ref_rows.pt", "rows": C.REF_ROWS, "dtype": "float32 (from fp64)"}
        rec["gen_s"] = round(time.time() - t0, 2)
        by_id[case.id] = rec; n += 1
        print("[generate] %-26s route %-7s det %s  max|err| %.3e (%.2f ulp@max)  sha %s  %.1fs" % (case.id, route, det, err["max_abs"], err["max_abs_in_ulp_at_max_ref"], rec["out_sha256"][:12], rec["gen_s"]), flush=True)
        del t, out, ref; torch.cuda.empty_cache()
    made["prebuilt_status"] = _clean_status(P.prebuilt_status()["resolved"])
    man["cases"] = list(by_id.values())
    man.setdefault("made", {})[cc] = dict(made, note=a.note, n_cases_written=n)
    man["package"] = {"version": P.__version__, "best_commit": P.BEST_COMMIT, "pins": P.PINS}
    write_manifest(man)
    print("[generate] cc %s: wrote %d case(s); manifest has %d cases" % (cc, n, len(man["cases"])))
    return 0


def _clean_status(resolved):
    """prebuilt_status()['resolved'] with the binary path made package-relative (the record itself is the shipped .json)."""
    out = {}
    for name, st in resolved.items():
        b = st.get("binary", "")
        if b.startswith("prebuilt:"):
            b = "prebuilt:" + os.path.relpath(b[len("prebuilt:"):], HERE)
        out[name] = {"binary": b, "record": st.get("record")}
    return out


def verify(a):
    man = load_manifest()
    by_id = {c["id"]: c for c in man["cases"]}
    paths = [p for p in (a.path or "").split(":") if p]
    have_gpu = torch.cuda.is_available()
    fn = route_fn = None; label = "no CUDA device: digests only"
    cc = this_cc()
    if have_gpu:
        fn, route_fn, label = load_impl(a.impl, paths)
        todo = C.cases_for_cc(cc)
        if not todo:
            print("[test_pkg] no test vectors for cc %s in this package (vectors exist for cc %s); nothing to compare on this device" % (cc, ", ".join(C.CCS)))
            return 1
    else:
        todo = list(C.CASES)
    dev = torch.device("cuda") if have_gpu else torch.device("cpu")
    only = set(a.only.split(",")) if a.only else None
    results = []; ok_all = True
    made = (man.get("made") or {}).get(cc or "", {})
    print("[test_pkg] impl = %s | device cc %s | vectors of this cc made on %s (torch %s, triton %s) | here: %s" % (label, cc, made.get("gpu"), made.get("torch"), made.get("triton"),
          json.dumps({k: v for k, v in env_record().items() if k in ("gpu", "torch", "triton", "stack_tag")})), flush=True)
    for case in todo:
        if only and case.id not in only:
            continue
        rec = by_id.get(case.id)
        r = {"id": case.id, "checks": {}}
        t = C.make_inputs(case, dev)
        dig = C.input_digests(t)
        if rec is None or not rec.get("inputs_sha256"):
            r["checks"]["input_digest"] = "not_in_manifest"
        else:
            r["checks"]["input_digest"] = "ok" if dig == rec["inputs_sha256"] else "input_digest_mismatch:%s" % ",".join(n for n in dig if dig[n] != rec["inputs_sha256"].get(n))
        if not have_gpu:
            r["checks"]["run"] = "skipped:no_cuda_device"
        elif r["checks"]["input_digest"] != "ok":
            r["checks"]["run"] = "skipped:inputs_differ"
        elif not rec.get("expected"):
            r["checks"]["expected"] = "expected_not_recorded (run --generate on a cc %s device)" % cc
        else:
            try:
                _, _, out, det, route = run_case(case, fn, route_fn, dev)
            except Exception as e:  # noqa: BLE001
                r["checks"]["run"] = "raised:%s: %s" % (type(e).__name__, str(e)[:300]); out = None
            if out is not None:
                r["checks"]["deterministic"] = "ok" if det else "run_to_run_differs"
                exp = torch.load(expected_path(case.id), map_location="cpu")
                got = out.detach().contiguous().cpu() if rec["expected"]["rows"] == "all" else out.detach()[:, :int(rec["expected"]["rows"])].contiguous().cpu()
                sha = C.tensor_digest(out)
                bit = got.shape == exp.shape and got.dtype == exp.dtype and bool(torch.equal(got, exp)) and (sha == rec["out_sha256"])
                if bit:
                    r["checks"]["bitwise"] = "ok"
                else:
                    d = (got.float() - exp.float()).abs() if got.shape == exp.shape else torch.tensor([float("nan")])
                    r["checks"]["bitwise"] = "DIFFERS: %d/%d stored elements differ, max |d| %.3e; full-output sha %s vs %s" % (int((got != exp).sum()) if got.shape == exp.shape else -1, exp.numel(), d.max().item(), sha[:12], rec["out_sha256"][:12])
                if route_fn is not None:
                    r["checks"]["route"] = "ok" if route == rec["route"] else "route %s vs recorded %s" % (route, rec["route"])
                ref = C.reference_fp64(t)
                err = C.error_stats(out, ref); r["err_vs_fp64"] = err
                lim = GATE_FACTOR * rec["err_vs_fp64"]["max_abs"]
                r["checks"]["gate"] = "ok" if (err["n_nonfinite"] == 0 and err["max_abs"] <= lim) else "FAIL: max|err| %.3e > %.3e (= %.1f x recorded %.3e) or non-finite %d" % (err["max_abs"], lim, GATE_FACTOR, rec["err_vs_fp64"]["max_abs"], err["n_nonfinite"])
                sref = torch.load(ref_path(case.id), map_location="cpu").double()
                rr = ref[:, :sref.shape[1]].cpu()
                rel = ((rr.float().double() - sref).abs().max() / rr.abs().max().clamp_min(1e-300)).item()
                r["checks"]["reference_integrity"] = "ok" if rel <= REF_INTEGRITY_RTOL else "stored fp64 reference rows differ from the recomputed ones: rel %.2e" % rel
                del out, ref
        r["pass"] = all(v == "ok" for v in r["checks"].values())
        ok_all &= r["pass"]
        print("[test_pkg] %-26s %s  %s" % (case.id, "PASS" if r["pass"] else "FAIL", "; ".join("%s=%s" % kv for kv in r["checks"].items() if kv[1] != "ok") or
                                            ("route %s, max|err| %.2e (recorded %.2e)" % (rec["route"], r.get("err_vs_fp64", {}).get("max_abs", float("nan")), rec["err_vs_fp64"]["max_abs"]))), flush=True)
        results.append(r)
        del t; torch.cuda.empty_cache() if have_gpu else None
    summary = {"impl": label, "cc": cc, "here": env_record(), "n": len(results), "n_pass": sum(r["pass"] for r in results), "all_pass": bool(ok_all), "results": results}
    if a.json:
        json.dump(summary, open(a.json, "w"), indent=1)
    print("[test_pkg] cc %s: %d/%d cases pass -- %s" % (cc, summary["n_pass"], summary["n"], "OK" if ok_all else "FAILED"))
    return 0 if ok_all else 1


def cmd_record_loadcheck(a):
    """Fill `loadcheck` (+ `loadcheck_bitwise`, `device`) in prebuilt/<stack tag>/<ext>.json records of this stack that say "pending", for extensions
    whose routes belong to this device's cc: run the LOADCHECK_CASES of the extension through the package with TRIATTN_PKG_PREBUILT=always (a fresh
    interpreter), require output bytes == the expected vectors, and record sha256/16 of those bytes."""
    import triattn_pkg as P
    cc = this_cc()
    if cc is None:
        sys.exit("test_pkg --record-loadcheck needs a CUDA device")
    d = os.path.join(P._face.PREBUILT_DIR, P.stack_tag())
    exts = [e for e in (a.ext.split(",") if a.ext else LOADCHECK_CASES) if EXT_CC.get(e) == cc]
    man = load_manifest(); by_id = {c["id"]: c for c in man["cases"]}
    todo = []
    for ext in exts:
        p = os.path.join(d, ext + ".json")
        if not os.path.isfile(p):
            print("[record-loadcheck] %s: no record %s for this stack -- skipped" % (ext, os.path.relpath(p, HERE))); continue
        rec = json.load(open(p))
        if rec.get("loadcheck") != "pending" and not a.ext:
            print("[record-loadcheck] %s: loadcheck %s -- skipped (name it with --ext to (re)record)" % (ext, "already recorded" if rec.get("loadcheck") else "not a field of this record (built by an earlier generation)")); continue
        todo.append((ext, p, rec))
    if not todo:
        print("[record-loadcheck] nothing to do on this device (cc %s, stack %s)" % (cc, P.stack_tag())); return 0
    ids = sorted({cid for ext, _, _ in todo for cid in LOADCHECK_CASES[ext]})
    missing = [cid for cid in ids if not (by_id.get(cid) or {}).get("expected")]
    if missing:
        sys.exit("test_pkg --record-loadcheck: expected vectors not recorded for %s -- run --generate on this device first" % missing)
    # a fresh interpreter with TRIATTN_PKG_PREBUILT=always: the bytes recorded are the PREBUILT's, whatever this process did before
    res_path = os.path.join(HERE, ".loadcheck_%d.json" % os.getpid())
    env = dict(os.environ, TRIATTN_PKG_PREBUILT="always")
    rc = subprocess.run([sys.executable, os.path.abspath(__file__), "--only", ",".join(ids), "--json", res_path], cwd=HERE, env=env).returncode
    rep = json.load(open(res_path)); os.remove(res_path)
    byres = {r["id"]: r for r in rep["results"]}
    if rc != 0 or not all(byres.get(cid, {}).get("pass") for cid in ids):
        sys.exit("test_pkg --record-loadcheck: the byte gate did not pass with the prebuilt binaries (%s) -- nothing recorded" % {cid: byres.get(cid, {}).get("checks") for cid in ids})
    for ext, p, rec in todo:
        rec["loadcheck"] = {cid: by_id[cid]["out_sha256"][:16] for cid in LOADCHECK_CASES[ext]}
        rec["loadcheck_bitwise"] = {cid: True for cid in LOADCHECK_CASES[ext]}
        rec["device"] = torch.cuda.get_device_name(0)
        json.dump(rec, open(p, "w"), indent=1, sort_keys=True); open(p, "a").write("\n")
        print("[record-loadcheck] %s: %s" % (ext, rec["loadcheck"]))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impl", default="pkg", help="pkg (default) | dispatch:<dir with candidate.py> | <module>:<function>")
    ap.add_argument("--path", default="", help="extra sys.path entries for --impl, ':'-separated")
    ap.add_argument("--only", default="", help="comma-separated case ids")
    ap.add_argument("--cc", default="", help="--list: restrict to this compute capability, e.g. 8.0")
    ap.add_argument("--json", default="", help="write the per-case results here")
    ap.add_argument("--list", action="store_true", help="print the case table and what the manifest records; no GPU needed")
    ap.add_argument("--generate", action="store_true", help="rewrite the test vectors of this device's cc from the package's outputs")
    ap.add_argument("--record-inputs", action="store_true", help="record input digests of cases the manifest does not know yet")
    ap.add_argument("--record-loadcheck", action="store_true", help="fill pending `loadcheck` fields of this stack's prebuilt records on this device")
    ap.add_argument("--ext", default="", help="--record-loadcheck: extension names, comma-separated (default: every pending record of this device's cc)")
    ap.add_argument("--note", default="", help="free text recorded in the manifest (--generate)")
    a = ap.parse_args()
    if a.list:
        sys.exit(cmd_list(a))
    if a.record_inputs:
        sys.exit(cmd_record_inputs(a))
    if a.record_loadcheck:
        sys.exit(cmd_record_loadcheck(a))
    sys.exit(generate(a) if a.generate else verify(a))


if __name__ == "__main__":
    main()
