"""Per-mode CPU tests: resolve() on an L40S stand-in reads the MODE from the DET recipe env — PROD -> the stock-route branch (keras_predict_fileorder;
'[L40S/prod]' on the line; no K1 probe); DET -> the K1 route (k1 where the stack imports; here torch is absent -> the TF route with the reason;
'[L40S/det]'); H100 both modes k1-default; A100 both modes stock; route_mode(); class_route() on string and dict rows, an unlisted class with a
device (K1 engages, said) and no device (K1 cannot run)."""
import os, sys, json, subprocess, textwrap
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); KIT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "kit")   # the carried kit (opt/kit: torch/ + cache/) = fastdefault.resolve's kit_root, as pred_bw_fast.py's _BASE_KIT
def _res(cls, extra):
    env = dict(os.environ)
    for k in ("TF_USE_DEFAULT_CONV_ALGO", "TF_DETERMINISTIC_OPS", "CHROMBPNET_DET_SUBPROCESS"): env.pop(k, None)
    env.update(extra); env["CHROMBPNET_FASTKIT_GPU_CLASS_OVERRIDE"] = cls
    code = textwrap.dedent("""
    import sys, json; sys.path.insert(0, %r)
    from chrombpnet_fastkit import fastdefault as fd
    r = fd.resolve(kit_root=%r); print("RES " + json.dumps({"forward": r["forward"], "why": r["forward_reason"][:700], "mode": r.get("mode"), "probe": r["torch_stack"].get("probe"), "precision": r.get("precision"), "k1_wanted": r.get("k1_wanted"), "why_tail": r["forward_reason"][-160:]}))
    """) % (HERE, KIT)
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, env=env, timeout=180)
    lines = [l for l in out.stdout.split("\n") if l.startswith("RES ")]; assert lines, out.stdout[-500:] + out.stderr[-900:]
    return json.loads(lines[-1][4:])
def test_l40s_prod_is_the_stock_route():
    r = _res("L40S", {}); assert r["forward"] == "keras_predict_fileorder" and r["mode"] == "prod" and "[L40S/prod]" in r["why"], r
    assert "skipped" in str(r["probe"]), r
def test_l40s_det_is_the_k1_route_of_record():
    r = _res("L40S", {"TF_DETERMINISTIC_OPS": "1", "CHROMBPNET_DET_SUBPROCESS": "1"}); assert r["mode"] == "det" and "[L40S/det]" in r["why"], r
    assert r["forward"] in ("k1", "tf_function"), r      # k1 where the stack imports; the TF route (the tf.function DET form) with the reason where it does not
    assert "skipped" not in str(r["probe"]), r            # the decision needed the stack: probed
def test_other_classes_both_modes():
    for cls, extra, want_k1 in (("H100", {}, True), ("H100", {"TF_DETERMINISTIC_OPS": "1"}, True), ("A100", {}, True), ("A100", {"TF_DETERMINISTIC_OPS": "1"}, False)):   # A100: K1 (tensor-core convs) at shipped numerics, stock's own graph under the recipe
        r = _res(cls, extra)
        if want_k1:
            assert "skipped" not in str(r["probe"]) and r["forward"] in ("k1", "tf_function") and r["k1_wanted"] is True, (cls, extra, r)
            if r["forward"] == "k1": assert r["precision"] == ("ieee" if extra else "tf32") and ("stock's deterministic order" if extra else "tensor cores") in r["why_tail"], (cls, extra, r)   # the conv arithmetic follows the mode
        else: assert r["forward"] in ("keras_predict_fileorder", "tf_function") and "skipped" in str(r["probe"]) and r["precision"] is None, (cls, extra, r)
def test_route_mode_and_rows():
    from chrombpnet_fastkit import fastdefault as fd
    for k in ("TF_USE_DEFAULT_CONV_ALGO", "TF_DETERMINISTIC_OPS", "CHROMBPNET_DET_SUBPROCESS"): os.environ.pop(k, None)
    assert fd.route_mode() == "prod"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    try: assert fd.route_mode() == "det"
    finally: del os.environ["TF_DETERMINISTIC_OPS"]
    assert fd.class_route("H100", "9.0", "prod")[0] is True and fd.class_route("H100", "9.0", "det")[0] is True
    assert fd.class_route("L40S", "8.9", "prod")[0] is False and fd.class_route("L40S", "8.9", "det")[0] is True
    assert fd.class_route("A100", "8.0", "prod")[0] is True and fd.class_route("A100", "8.0", "det")[0] is False and fd.class_route("B200", "10.0", "det")[0] is False
    k1, why = fd.class_route("unknown", "8.6", "det"); assert k1 is True and "cold JIT" in why and "not measured on this card" in why, why   # an unlisted class with a device: K1 engages, said
    k1, why = fd.class_route("unknown", None, "prod"); assert k1 is False and "cannot run" in why, why                                    # no device class: K1 cannot run
    assert fd.arch_default_route(KIT, "cuda-86", "prod") is None                                                                           # an unlisted arch: class_route decides
if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"): fn(); print("ok", name)
    print("PER-MODE CPU TESTS OK")
