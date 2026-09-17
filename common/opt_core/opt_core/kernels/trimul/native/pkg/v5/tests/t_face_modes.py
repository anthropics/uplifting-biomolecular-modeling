#!/usr/bin/env python3
"""face.serve under every execution mode a caller may be in: torch.inference_mode(), torch.no_grad(), autograd-enabled eager (incl. inputs
that require grad), a non-default stream, and CUDA-graph capture + replay (incl. replay after an in-place input swap) -- all must give the
bytes of the plain eager call; mixed sequences (inference-mode call then no_grad call on the same cache, and back) must not raise.  Also
measures the host cost per serve (us of CPU per call, cache warm, no synchronisation inside the loop) next to the device time per call.

    PYTHONPATH=python python tests/t_face_modes.py [--c 128] [--n 256] [--forms bf16,f32z] [--out /tmp/t_face_modes.json]
"""
import argparse, hashlib, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
import torch                                                  # noqa: E402
from trimul_native import face as F, vectors as V              # noqa: E402


def sha(t):
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def run_form(c, n, form, direction, dev, rec):
    case = {"c_z": c, "c_hidden": c, "n": n, "batch": 1, "form": form, "mask": "tail3", "direction": direction, "residual": True, "variant": "fast"}
    z, mask, w = V.closed_form_inputs(case, dev)
    key = "c%d_n%d_%s_%s" % (c, n, form, direction)
    out = {}
    ac = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if form == "f32z" else (lambda: torch.autocast("cuda", enabled=False))
    kw = dict(direction=direction, weights=w, residual=True)
    cache = {}
    # 1. plain eager (grad mode on, inputs without grad)
    with ac():
        ref = F.serve(z, mask, cache=cache, **kw)
    torch.cuda.synchronize()
    out["eager"] = sha(ref)
    # 2. no_grad
    with torch.no_grad(), ac():
        o = F.serve(z, mask, cache=cache, **kw)
    out["no_grad"] = sha(o)
    # 3. inference_mode with inference-tensor inputs (fresh tensors created inside the region) on the SAME cache, then eager again
    with torch.inference_mode(), ac():
        zi, mi, wi = V.closed_form_inputs(case, dev)
        oi = F.serve(zi, mi, direction=direction, weights=wi, residual=True, cache=cache)
        out["inference_mode_fresh_inputs"] = sha(oi)
        oi2 = F.serve(z, mask, cache=cache, **kw)                            # normal tensors passed inside inference mode
        out["inference_mode_normal_inputs"] = sha(oi2)
    with ac():
        o = F.serve(z, mask, cache=cache, **kw)                              # back to grad mode on the same cache (workspace namespaces)
    out["eager_after_inference"] = sha(o)
    with torch.no_grad(), ac():
        o = F.serve(zi, mi, direction=direction, weights=wi, residual=True, cache=cache)   # inference tensors used under no_grad
    out["no_grad_on_inference_tensors"] = sha(o)
    # 4. inputs that require grad (the forward is not differentiable; outputs must not require grad and nothing may raise)
    zg = z.clone().requires_grad_(True) if z.dtype == torch.float32 else z.clone().float().requires_grad_(True).to(z.dtype)
    wg = {k: v.clone().requires_grad_(True) for k, v in w.items()}
    with ac():
        og = F.serve(zg if z.dtype == torch.float32 else z, mask, direction=direction, weights=wg, residual=True, cache={})
    out["grad_inputs"] = sha(og)
    out["grad_inputs_requires_grad"] = bool(og.requires_grad)
    # 5. non-default stream
    s = torch.cuda.Stream(device=dev)
    s.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(s), ac():
        os_ = F.serve(z, mask, cache=cache, **kw)
    torch.cuda.current_stream(dev).wait_stream(s)
    torch.cuda.synchronize()
    out["side_stream"] = sha(os_)
    # 6. CUDA graph: capture on static inputs (cache warm from the calls above), replay, swap input bytes in place, replay again
    zs = z.clone(); ms = mask.clone()
    with ac():
        F.serve(zs, ms, cache=cache, **kw)                                   # warm this exact (address, N) plan before capture
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), ac():
        og_static = F.serve(zs, ms, cache=cache, **kw)
    g.replay(); torch.cuda.synchronize()
    out["graph_replay"] = sha(og_static)
    case2 = dict(case, mask="none")
    z2, _, _ = V.closed_form_inputs(dict(case, form=form), dev)
    z2 = (z2.float() * 0.5).to(z.dtype)                                    # different input bytes, same address after copy_
    with ac():
        want2 = F.serve(z2, ms, cache={}, **kw)
    zs.copy_(z2); g.replay(); torch.cuda.synchronize()
    out["graph_replay_after_swap"] = sha(og_static)
    out["graph_replay_after_swap_want"] = sha(want2)
    t_rep = []
    for _ in range(20):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record(); e1.synchronize(); t_rep.append(e0.elapsed_time(e1))
    out["graph_replay_ms_median"] = sorted(t_rep)[len(t_rep) // 2]
    # verdicts
    same = ["no_grad", "inference_mode_fresh_inputs", "inference_mode_normal_inputs", "eager_after_inference", "no_grad_on_inference_tensors", "grad_inputs", "side_stream", "graph_replay"]
    out["ok"] = all(out[k] == out["eager"] for k in same) and out["graph_replay_after_swap"] == out["graph_replay_after_swap_want"] and not out["grad_inputs_requires_grad"]
    out["mismatches"] = [k for k in same if out[k] != out["eager"]] + ([] if out["graph_replay_after_swap"] == out["graph_replay_after_swap_want"] else ["graph_replay_after_swap"])
    # 7. host cost per serve (cache warm): CPU us per call without synchronisation, and device ms per call (events around 20 calls)
    with ac():
        for _ in range(10):
            F.serve(z, mask, cache=cache, **kw)
        torch.cuda.synchronize()
        nloop = 200
        t0 = time.perf_counter()
        for _ in range(nloop):
            F.serve(z, mask, cache=cache, **kw)
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(20):
            F.serve(z, mask, cache=cache, **kw)
        e1.record(); e1.synchronize()
    out["host_us_per_serve"] = round((t1 - t0) / nloop * 1e6, 1)
    out["wall_us_per_serve_incl_drain"] = round((t2 - t0) / nloop * 1e6, 1)
    out["device_ms_per_serve_stream_of_20"] = round(e0.elapsed_time(e1) / 20, 4)
    rec[key] = out
    print("[t_face_modes] %-22s ok=%s eager=%s mismatches=%s host=%.1fus/serve device=%.3fms/serve graph_replay=%.3fms" % (
        key, out["ok"], out["eager"], out["mismatches"], out["host_us_per_serve"], out["device_ms_per_serve_stream_of_20"], out["graph_replay_ms_median"]))
    return out["ok"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=int, default=128)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--forms", default="bf16,f32z")
    ap.add_argument("--directions", default="outgoing,incoming")
    ap.add_argument("--out", default="/tmp/t_face_modes.json")
    a = ap.parse_args()
    dev = torch.device("cuda", torch.cuda.current_device())
    print("precision: allow_tf32=%s float32_matmul_precision=%s; torch %s; device %s cc %d.%d" % ((torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision(), torch.__version__, torch.cuda.get_device_name(dev)) + torch.cuda.get_device_capability(dev)))
    rep = F.check(dev)
    print("[t_face_modes] check: arch %s binding %s units %s gate %s" % (rep["arch"], rep["binding"], sorted(rep["units"]), rep["gate"]))
    rec = {"torch": torch.__version__, "device": torch.cuda.get_device_name(dev), "check": {"arch": rep["arch"], "binding": rep["binding"], "gate": rep["gate"]}, "cases": {}}
    ok = True
    cc = torch.cuda.get_device_capability(dev)
    for form in a.forms.split(","):
        for d in a.directions.split(","):
            word = F.admits(cc, "bf16", a.c, a.c, a.n, d, residency=("fp32" if form == "f32z" else None), residual=True)
            if word is not None:                                            # a form this release does not serve at this width: a named refusal, not a failure
                rec["cases"]["c%d_n%d_%s_%s" % (a.c, a.n, form, d)] = {"ok": True, "skipped": word}
                print("[t_face_modes] c%d_n%d_%s_%s skipped by name: %s" % (a.c, a.n, form, d, word))
                continue
            try:
                ok &= run_form(a.c, a.n, form, d, dev, rec["cases"])
            except Exception as e:                                           # record and continue: the report names the failing mode
                import traceback
                rec["cases"]["c%d_n%d_%s_%s" % (a.c, a.n, form, d)] = {"ok": False, "error": "%s: %s" % (type(e).__name__, str(e)[:300]), "tb": traceback.format_exc()[-1500:]}
                print("[t_face_modes] c%d_n%d_%s_%s ERROR %s: %s" % (a.c, a.n, form, d, type(e).__name__, str(e)[:200]))
                ok = False
    rec["ok"] = bool(ok)
    json.dump(rec, open(a.out, "w"), indent=1, default=str)
    print("[t_face_modes] %s -> %s" % ("OK" if ok else "FAIL", a.out))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
