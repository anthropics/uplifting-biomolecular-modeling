#!/usr/bin/env python3
"""Discriminates call paths for one case: vectors' _serve_case vs face.serve(config f32_resident_ok) vs autocast, before/after face.check()."""
import json, os, sys, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "python"))

def child(mode):
    import torch
    from trimul_native import face as F, vectors as V
    case = {"c_z": 256, "c_hidden": 256, "n": 403, "batch": 1, "direction": "outgoing", "form": "f32z", "residual": True, "mask": "tail3", "variant": "fast"}
    z, m, w = V.closed_form_inputs(case, "cuda")
    ref = V.fp64_reference(z, m, w, "outgoing", True)
    out = {}
    def rec(tag, o):
        d = (o.double() - ref).abs().max().item(); out[tag] = (V.tensor_sha256(o)[:12], str(o.dtype).replace("torch.", ""), round(d, 5))
    if mode == "gate_first":
        F.check()
    with torch.no_grad():
        rec("1_serve_case(vectors path)", V._serve_case(F, case, z, m, w, {}))
        rec("2_face f32_resident_ok cfg", F.serve(z, m, direction="outgoing", weights=w, residual=True, cache={}, config={"f32_resident_ok": True}))
        rec("3_face f32_resident_ok + variant fast + gate False", F.serve(z, m, direction="outgoing", weights=w, residual=True, cache={}, config={"variant": "fast", "gate": False, "f32_resident_ok": True}))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rec("4_face under autocast", F.serve(z, m, direction="outgoing", weights=w, residual=True, cache={}))
        rec("5_serve_case again", V._serve_case(F, case, z, m, w, {}))
        z2 = z.clone(); rec("6_face cfg on a cloned z", F.serve(z2, m, direction="outgoing", weights=w, residual=True, cache={}, config={"f32_resident_ok": True}))
        c1 = {}; rec("7a shared cache call1", F.serve(z, m, direction="outgoing", weights=w, residual=True, cache=c1, config={"f32_resident_ok": True}))
        rec("7b shared cache call2", F.serve(z, m, direction="outgoing", weights=w, residual=True, cache=c1, config={"f32_resident_ok": True}))
    tv = {c["id"]: c["expected"]["sm90"].get("sha256", "")[:12] for c in V.load_manifest()["cases"]}
    out["expected(testvectors)"] = tv["c256x256_n403_out_f32z_res1_mtail3_fast"]
    print(json.dumps({"mode": mode, "results": out}, indent=1))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        child(sys.argv[1])
    else:
        for mode in ("plain", "gate_first"):
            subprocess.run([sys.executable, __file__, mode])
