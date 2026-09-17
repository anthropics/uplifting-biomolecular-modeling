"""Pre-stamp verb for the sealed trimul_native payload (kernels.trimul rows native / native_exact / native:f32in).

    python -m opt_core.kernels.trimul.native stamp [--verify] [--device N] [--json]      (also stamps the fpf_trimul_v4 warm probe's verdicts)
        Make sure THIS installed tree x device class x stack holds a verification stamp: when an intact stamp exists the payload is only
        digest-verified + loaded + load-checked (prints stamp=hit); otherwise the byte gate replays the sealed test vectors of the device class
        and the stamp is written (stamp=written).  --verify replays the gate even over an intact stamp and re-writes it.  Run it from a kit's
        `run.sh warm`, or at overlay / image-bake time ON A GPU OF THE SAME CLASS: the stamp key holds the payload SHA256SUMS digest, the
        package path, cc + device name, driver, CUDA and torch versions -- a stamp baked on cc X is honoured at run time on cc X (same device
        name / driver / torch / tree path) and ignored (the gate runs once, then stamps) on any other.  Exit 0 = stamped / verified; 2 = refused.
    python -m opt_core.kernels.trimul.native status [--json]
        The stamp directory chain (kind=path, writable?) and the stamps present; no GPU work.
"""
import json
import os
import sys


def _token(rep):
    st = str(rep.get("verdict_stamp", "-")); ran = "yes" if rep.get("gate_ran", True) else "no"
    return "[opt_core] NATIVE_STAMP trimul native@%s pkg=%s cc=%s stamp=%s gate_ran=%s gate_s=%.2f install_s=%.2f dir=%s" % (
        str(rep.get("payload_version", "?")).split()[-1], rep.get("pkg", "?"), rep.get("cc", rep.get("device_class", "?")), st, ran,
        float(rep.get("gate_s") or 0.0), float(rep.get("install_s") or 0.0), rep.get("verdict_stamp_dir_kind") or "-")


def main(argv=None):
    import argparse
    from . import (ACTIVE_PKG, VERSION, STAMP_ENV, Unavailable, install, payload_version, stamp_dir, stamp_dirs, verify_digests)
    ap = argparse.ArgumentParser(prog="python -m opt_core.kernels.trimul.native", description="verification stamp for the sealed trimul_native payload")
    sub = ap.add_subparsers(dest="verb", required=True)
    p = sub.add_parser("stamp", help="verify + load the payload on this GPU; replay the byte gate unless an intact stamp exists; write the stamp")
    p.add_argument("--verify", action="store_true", help="replay the byte gate even when an intact stamp exists, then re-write it")
    p.add_argument("--device", type=int, default=None, help="CUDA device index (default: current)")
    p.add_argument("--json", action="store_true", help="print the install report as JSON after the token line")
    q = sub.add_parser("status", help="the stamp directory chain and the stamps present (no GPU work)")
    q.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.verb == "status":
        chain = [{"kind": k, "path": d, "writable": bool(os.access(d, os.W_OK | os.X_OK)),
                  "stamps": sorted(f for f in os.listdir(d) if f.startswith("gate-") and f.endswith(".json"))} for k, d in stamp_dirs()]
        facts = {"pkg": ACTIVE_PKG, "version": VERSION, "env": os.environ.get(STAMP_ENV), "write_dir": stamp_dir(), "chain": chain}
        if a.json:
            print(json.dumps(facts, indent=1))
        else:
            print("trimul_native %s (pkg/%s); %s=%s; write dir: %s" % (VERSION, ACTIVE_PKG, STAMP_ENV, os.environ.get(STAMP_ENV), stamp_dir()))
            for c in chain:
                print("  %-13s %s  writable=%s  stamps=%d %s" % (c["kind"], c["path"], c["writable"], len(c["stamps"]), " ".join(c["stamps"][:4])))
            if not chain:
                print("  (no stamp directory: disabled by env or nothing creatable -- the byte gate runs in every process)")
        return 0
    try:
        try:
            import torch                                                     # the payload binds the driver through the framework's context
        except ImportError as e:
            print("[opt_core] NATIVE_STAMP trimul native@%s pkg=%s stamp=disabled:no_framework (%s; nothing stamped)" % (payload_version().split()[-1], ACTIVE_PKG, str(e)[:80]))
            return 2
        if not torch.cuda.is_available():
            print("[opt_core] NATIVE_STAMP trimul native@%s pkg=%s stamp=disabled:no_device (no CUDA device in this process; nothing stamped)" % (payload_version().split()[-1], ACTIVE_PKG))
            return 2
        dev = a.device if a.device is not None else torch.cuda.current_device()
        torch.cuda.set_device(dev)
        verify_digests()
        rep = install(device=dev, gate=True, force_gate=bool(a.verify))
        try:                                                                 # one bake command stamps both: the byte gate above and the fpf_trimul_v4 warm probe's
            from ..fpf_trimul_v4 import generic as _G                        # verdict for every shape class that package serves on this device (leaf fpf_trimul_v4)
            _G.probe_all_classes(("cuda:%d" % dev) if isinstance(dev, int) else (dev if dev is not None else "cuda"), log=print)
        except (ImportError, RuntimeError, OSError, AttributeError, KeyError, ValueError, TypeError) as e:   # that package / triton not importable or not servable here: said, not fatal for the gate stamp
            print("[fpf_trimul_v4 probe] not stamped: %s: %s" % (type(e).__name__, str(e)[:120]))
    except Unavailable as e:
        print("[opt_core] NATIVE_STAMP trimul native@%s pkg=%s stamp=refused:%s (%s)" % (payload_version().split()[-1], ACTIVE_PKG, e.kind, str(getattr(e, "detail", ""))[:160]))
        return 2
    print(_token(rep))
    if a.json:
        print(json.dumps({k: v for k, v in rep.items() if k != "units"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
