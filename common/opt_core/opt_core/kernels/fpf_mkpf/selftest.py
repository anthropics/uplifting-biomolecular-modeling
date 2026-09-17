#!/usr/bin/env python3
"""fpf_mkpf from-tarball selftest on REAL dumps: composes exactly like the CLI (FPF env.sh ARM=T|E + PTX_GLUE_V2 + PTX_MK_PF), imports protenix pairformer (-> levers apply),
calls install(), then for block 0/47 at each N: runs the BLK2 tri-att statement helper (ptx_trunk2_levers._triatt_block_pro_epi) with MK live vs with MK disabled (tested path),
reports err-ratio vs fp64 (stock error = tested path error since tested == stock bits), n_diff, r2r x3, and CUDA-graph replay of the fused prologue; F3 likewise via fn_residual.
Exit code 0 = all checks pass.  Usage: python -m fpf_mkpf.selftest --dumps <dir>[:<dir>] --sizes 356,705 [--json out.json]"""
import os, sys, json, argparse, math, time
import torch
ap = argparse.ArgumentParser(); ap.add_argument("--dumps", default="dumps"); ap.add_argument("--sizes", default="356,705")
ap.add_argument("--keys", default="pf_c1_b0,pf_c10_b47"); ap.add_argument("--json", default=""); ap.add_argument("--ratio-bar", type=float, default=1.10)
a = ap.parse_args(); dev = "cuda"; bf = torch.bfloat16
import protenix.model.modules.pairformer as PF          # -> bundle sitecustomize -> levers (BLK2, glue) applied
import ptx_trunk2_levers as LEV, fpf_mkpf, fpf_mkpf.kernels as MK
import fpf_triatt_pro.prologue as PRO, fpf_triatt_epi.epilogue as EPI, fpf_transition.transition as TR
st = fpf_mkpf.install(verbose=True)
print(json.dumps({"selftest": "fpf_mkpf", "device": torch.cuda.get_device_name(0), "cc": list(torch.cuda.get_device_capability(0)), "torch": torch.__version__, "stats": st}, default=str), flush=True)
import fpf.reference as R
model = R.build_runner().model.eval(); PB = model.pairformer_stack.blocks; BLK = {"pf_c1_b0": 0, "pf_c10_b0": 0, "pf_c1_b47": 47, "pf_c10_b47": 47}
def load(N):
    for base in a.dumps.split(":"):
        if not os.path.isdir(base): continue
        for fn in sorted(os.listdir(base)):
            if (fn.startswith("acts_") and fn.endswith(f"_{N}tok.pt")) or fn == f"acts_{N}tok.pt": return torch.load(os.path.join(base, fn), map_location="cpu", weights_only=False)
    raise FileNotFoundError(f"FATAL no dump N={N} under {a.dumps}")
checks = []
def check(name, ok, **kw):
    checks.append(dict(check=name, PASS=bool(ok), **kw)); print(("PASS " if ok else "FAIL ") + name, json.dumps(kw, default=str), flush=True)
epi_k = EPI.triatt_epilogue; pro = PRO.triatt_prologue; gc = PRO.get_cache
for N in [int(x) for x in a.sizes.split(",") if x]:
    D = load(N)
    for key in a.keys.split(","):
        if key not in D: continue
        blk = PB[BLK[key]]; z0 = D[key]["z"].to(dev, bf).contiguous()
        with torch.no_grad(), torch.autocast("cuda", dtype=bf):
            for ending in (False, True):
                mod = blk.tri_att_end if ending else blk.tri_att_start; tag = f"N={N} {key} {'end' if ending else 'start'}"
                sel = ("F1E" in st["levers"]) if ending else ("F1S" in st["levers"])
                # tested statement: force the original LN mode hook
                z_ref = z0.clone(); lnm_inst = LEV._blk_ln_mode; LEV._blk_ln_mode = getattr(lnm_inst, "__wrapped__", lnm_inst)
                LEV._triatt_block_pro_epi(mod, z_ref, ending, epi_k, pro, gc); LEV._blk_ln_mode = lnm_inst
                c0 = dict(fpf_mkpf._STATE["calls"])
                outs = []
                for rep in range(3):
                    zc = z0.clone(); LEV._triatt_block_pro_epi(mod, zc, ending, epi_k, PRO.triatt_prologue, gc); outs.append(zc)
                c1 = dict(fpf_mkpf._STATE["calls"])
                used = (c1["f1_end"] - c0["f1_end"]) if ending else (c1["f1_start"] - c0["f1_start"])
                r2r = bool(torch.equal(outs[0], outs[1])) and bool(torch.equal(outs[0], outs[2]))
                if not sel:
                    check(f"F1 not selected for this node -> statement bitwise == certified [{tag}]", bool(torch.equal(outs[0], z_ref)) and used == 0, mk_calls=used); continue
                # fp64 reference of the whole statement is expensive (attention); use the tested output as the reference class carrier: report n_diff + max|d| + relative rms
                d = (outs[0].float() - z_ref.float()); nd = int((outs[0] != z_ref).sum()); rel = float(d.pow(2).mean().sqrt() / z_ref.float().pow(2).mean().sqrt())
                exact = fpf_mkpf._STATE["cells"].get("ln_arith") == "welford"
                check(f"F1 statement vs certified: r2r + engaged + " + ("BITWISE (EXACT cell)" if exact else "deviation class (Tier-2 cell)") + f" [{tag}]", r2r and used == 3 and ((nd == 0) if exact else rel < 1e-3) and bool(torch.isfinite(outs[0]).all()), mk_calls=used, r2r=r2r, n_diff=nd, frac=nd / z_ref.numel(), max_abs=float(d.abs().max()), rel_rms=rel)
                # op-level: fused prologue vs tested prologue on the same x: err ratio vs fp64
                x = z0.transpose(-2, -3) if ending else z0; x_ln = mod.layer_norm(x).to(bf).contiguous()
                ref = getattr(pro, "__wrapped__", pro)(mod, x_ln, ending=False, ln_mode="stock", x_ln=x_ln) if True else None
                ARITH = fpf_mkpf._STATE["cells"].get("ln_arith", "fused"); FMA = tuple(bool(int(f)) for f in fpf_mkpf._STATE["cells"].get("fma_flags", [1, 1, 1]))
                cand = MK.prologue_ln(mod, x, gc(mod, z0.device), st["cells"]["f1"], ending=False, ln_arith=ARITH, fma_flags=FMA)
                lw, lb = mod.layer_norm.weight.double(), mod.layer_norm.bias.double(); x64 = torch.nn.functional.layer_norm(x.double(), (256,), lw, lb, mod.layer_norm.eps)
                g64 = x64 @ mod.mha.linear_g.weight.double().t()
                ec = (cand[3].double() - g64); er = (ref[3].double() - g64)
                ratio_rms = float(ec.pow(2).mean().sqrt() / er.pow(2).mean().sqrt()); ratio_max = float(ec.abs().max() / er.abs().max())
                check(f"F1 prologue g-projection err ratio vs fp64 (cand/certified) <= {a.ratio_bar} [{tag}]", ratio_rms <= a.ratio_bar and ratio_max <= a.ratio_bar, ratio_rms=round(ratio_rms, 4), ratio_max=round(ratio_max, 4))
                del x64, g64, ec, er, ref, cand
                # graph replay on new input
                try:
                    zs = z0.clone(); xs = zs.transpose(-2, -3) if ending else zs
                    MK.prologue_ln(mod, xs, gc(mod, z0.device), st["cells"]["f1"], ending=False, ln_arith=ARITH, fma_flags=FMA); torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g): res = MK.prologue_ln(mod, xs, gc(mod, z0.device), st["cells"]["f1"], ending=False, ln_arith=ARITH, fma_flags=FMA)
                    znew = (z0.float() * 1.003 + 0.01).to(bf); zs.copy_(znew); g.replay(); torch.cuda.synchronize()
                    got = [t.clone() for t in res]; xn = znew.transpose(-2, -3) if ending else znew
                    eager = MK.prologue_ln(mod, xn, gc(mod, z0.device), st["cells"]["f1"], ending=False, ln_arith=ARITH, fma_flags=FMA)
                    check(f"F1 CUDA-graph replay on NEW z == eager [{tag}]", all(bool(torch.equal(p, q)) for p, q in zip(got, eager)))
                except Exception as e:
                    check(f"F1 CUDA-graph replay [{tag}]", False, error=repr(e)[:200])
            if "F3" in st["levers"]:
                fw = TR._forward; TR._forward = getattr(fw, "__wrapped__", fw); ref = TR.fn_residual(blk.pair_transition, z0); TR._forward = fw
                c0 = fpf_mkpf._STATE["calls"]["f3"]; outs = [TR.fn_residual(blk.pair_transition, z0) for _ in range(3)]; used = fpf_mkpf._STATE["calls"]["f3"] - c0
                d = (outs[0].float() - ref.float()); nd = int((outs[0] != ref).sum())
                check(f"F3 transition vs certified [N={N} {key}]", used == 3 and bool(torch.equal(outs[0], outs[1])) and bool(torch.equal(outs[0], outs[2])) and float(d.abs().max()) < 8.1, mk_calls=used, n_diff=nd, frac=nd / ref.numel(), max_abs=float(d.abs().max()))
        del z0; torch.cuda.empty_cache()
summary = {"selftest": "fpf_mkpf", "n_checks": len(checks), "n_fail": sum(1 for c in checks if not c["PASS"]), "failed": [c["check"] for c in checks if not c["PASS"]], "stats": fpf_mkpf.stats(), "PASS": all(c["PASS"] for c in checks) and len(checks) > 0}
print("SELFTEST SUMMARY", json.dumps(summary, default=str), flush=True)
if a.json: json.dump({"summary": summary, "checks": checks}, open(a.json, "w"), indent=1, default=str)
sys.exit(0 if summary["PASS"] else 1)
