"""CPU tests (no GPU, no TF, no torch): the CACHE-PINS PRECONDITION on the K1 route. (1) k1_cache_pins(kit_root, arch) reads this arch's shipped
Triton cache's JIT_IDENTITY.json kernels pin vs sha256(kernels.py)[:16] from the kit's own bytes; (2) resolve() on a K1-route (class, mode) with a
MISMATCH -> K1 with the cold JIT named in the reason (the torch stack faked present); with a MATCH -> k1; (3) (removed with the --forward flag: no explicit route)
runs k1 with the cold JIT stated; (4) the real vendored package: its pins read exactly as the tree's bytes say (a datum printed, never asserted)."""
import os, sys, json, shutil, tempfile, hashlib, subprocess, textwrap
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); KIT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "kit")   # the carried kit (opt/kit: torch/ + cache/) = fastdefault.resolve's kit_root, as pred_bw_fast.py's _BASE_KIT
from chrombpnet_fastkit import fastdefault as fd
def _fake_kit(pin_matches, arch_dir="triton_cache_sm90", arch="cuda-90-32"):
    d = tempfile.mkdtemp(); t = os.path.join(d, "torch"); os.makedirs(os.path.join(t, "chrombpnet_k1")); os.makedirs(os.path.join(t, arch_dir))
    kern = b"def k(): pass  # fake kernels\n"; open(os.path.join(t, "chrombpnet_k1", "kernels.py"), "wb").write(kern)
    ksha = hashlib.sha256(kern).hexdigest()[:16]
    json.dump({"entries": [{"arch": arch, "dir": arch_dir, "device_classes": ["NVIDIA H100"], "stack": {"triton": "3.0.0", "torch": "2.4.1+cu124"}}]}, open(os.path.join(t, "triton_cache_of_record.json"), "w"))
    json.dump({"kernels_sha256_16": ksha if pin_matches else "0000000000000000", "triton": "3.0.0", "torch": "2.4.1+cu124"}, open(os.path.join(t, arch_dir, "JIT_IDENTITY.json"), "w"))
    shutil.copy(os.path.join(HERE, "chrombpnet_fastkit", "fastdefault.py"), os.path.join(d, "fastdefault_copy.py"))
    return d, ksha
def test_pins_reader():
    d, ksha = _fake_kit(True); r = fd.k1_cache_pins(d, "cuda-90"); assert r["ok"] is True and r["kernels_sha256_16"] == ksha == r["cache_pin"], r
    d, ksha = _fake_kit(False); r = fd.k1_cache_pins(d, "cuda-90"); assert r["ok"] is False and "pinned to other kernel bytes" in r["note"], r
    r = fd.k1_cache_pins(d, "cuda-100"); assert r["ok"] is False and "no shipped cache" in r["note"], r
def _resolve(kit_root, cls, det=False):
    env = dict(os.environ); env["CHROMBPNET_FASTKIT_GPU_CLASS_OVERRIDE"] = cls
    if det: env.update({"TF_DETERMINISTIC_OPS": "1", "TF_USE_DEFAULT_CONV_ALGO": "1"})
    code = textwrap.dedent("""
    import sys, json; sys.path.insert(0, %r)
    from chrombpnet_fastkit import fastdefault as fd
    fd._torch_stack = lambda kit_root: (True, {"torch": "2.4.1+cu124", "triton": "3.0.0", "source": "fake", "present": True})
    r = fd.resolve(kit_root=%r)
    print("RES " + json.dumps({"forward": r["forward"], "why": r["forward_reason"][:400], "pins": r.get("k1_cache_pins"), "skipped": r["skipped"][:4]}))
    """) % (HERE, kit_root)
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, env=env, timeout=180)
    lines = [l for l in out.stdout.split("\n") if l.startswith("RES ")]; assert lines, out.stdout[-600:] + out.stderr[-900:]
    return json.loads(lines[-1][4:])
def test_mismatch_runs_k1_with_the_cold_jit_named():
    # a cache-pin miss is a missing shipped cache, not a missing lever — the same K1 forward runs with its kernels compiling in this process, said on one line
    d, _ = _fake_kit(False); r = _resolve(d, "H100")
    assert r["forward"] == "k1" and "K1 cold JIT: cache pins do not match this stack" in r["why"] and "pinned to other kernel bytes" in r["why"], r
    assert r["pins"] and r["pins"]["ok"] is False, r
    assert not any("HELD" in s for s in r["skipped"]), r["skipped"]
def test_match_takes_k1():
    d, _ = _fake_kit(True); r = _resolve(d, "H100"); assert r["forward"] == "k1", r; assert r["pins"]["ok"] is True, r
def test_real_vendored_package_pins_datum():
    for arch in ("cuda-90", "cuda-80", "cuda-89"):
        r = fd.k1_cache_pins(KIT, arch); print("   vendored package pins", arch, json.dumps(r)[:240])
if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"): fn(); print("ok", name)
    print("V01270 CPU TESTS OK")
