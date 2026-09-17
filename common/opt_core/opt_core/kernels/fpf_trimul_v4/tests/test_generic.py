"""fpf_trimul_v4 v4.1.0 generic-entry tests (GPU).  Sections:
 A  Protenix equality: torch.equal(v4.1.0 fn (thin wrapper over generic), v4.0.4r1 fn) on real dumps 356/705 x {pf_c1_b0, pf_c10_b47} x out/in x residual on/off (+ mask=ones);
    also generic.trimul called with raw module tensors == fn.  Needs the Protenix model (numkit FPF_SPEC_v0) and the v4.0.4r1 package importable as `fpf_trimul_v404`.
 B  Unit tests vs pure-torch references, (C,D) in {(128,128),(256,256),(128,256),(256,128)} x N in {546,705} x out/in: seeded random weights WITH biases, random mask (p=0.9),
    z = real-dump channel slice (bf16); err vs fp64 reference for: v4 (bf16 in), v4 (fp32 in), torch fp32 (TF32 off), torch bf16-autocast; ratios printed; r2r (5 calls) bit-exact;
    CUDA-graph capture + replay on a NEW input == eager; mask=None == mask=ones bit-exact; finite; fused w_proj spec == four-matrix spec bit-exact.
 C  Speed: v4 (update only) vs torch-bf16-autocast vs torch-fp32 per (C,D,N), CUDA events median of 20, prereg printed beside.
Usage: python -m fpf_trimul_v4.tests.test_generic --dumps DIR --out OUT --tag TAG [--sections A,B,C]"""
import os, sys, json, time, argparse, itertools, importlib
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--dumps", required=True); ap.add_argument("--out", default="out"); ap.add_argument("--tag", default="v41")
ap.add_argument("--sections", default="A,B,C"); ap.add_argument("--sizes", default="546,705"); ap.add_argument("--nrep", type=int, default=20)
ap.add_argument("--v404-pkg", default="fpf_trimul_v404", help="import name of the shipped v4.0.4r1 package for section A")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
dev = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
import triton
print("GPU", torch.cuda.get_device_name(), "cc", torch.cuda.get_device_capability(), "torch", torch.__version__, "triton", triton.__version__, flush=True)
import fpf_trimul_v4 as PKG, fpf_trimul_v4.generic as G, fpf_trimul_v4.kernels as K, fpf_trimul_v4.trimul as V41
print("fpf_trimul_v4", PKG.__version__, "from", os.path.dirname(PKG.__file__), flush=True)
RES = {"gpu": torch.cuda.get_device_name(), "cc": torch.cuda.get_device_capability(), "torch": torch.__version__, "triton": triton.__version__, "version": PKG.__version__, "A": [], "B": [], "C": []}
SECTIONS = set(args.sections.split(","))

def dump(N):
    p = os.path.join(args.dumps, "acts_%dtok.pt" % N)
    return torch.load(p, map_location="cpu", weights_only=False)

def save():
    json.dump(RES, open(os.path.join(args.out, args.tag + "_generic.json"), "w"), indent=1, default=str)

def timeit(f, nrep, warm=5):
    for _ in range(warm): f()
    torch.cuda.synchronize(); ev = [torch.cuda.Event(enable_timing=True) for _ in range(nrep + 1)]; ev[0].record()
    for k in range(nrep): f(); ev[k + 1].record()
    torch.cuda.synchronize(); ts = sorted(ev[k].elapsed_time(ev[k + 1]) for k in range(nrep)); return ts[len(ts) // 2]

def errs(c, r):
    e = (c.double() - r.double()); return {"max_abs": float(e.abs().max()), "rms": float(e.pow(2).mean().sqrt())}

# ------------------------------------------------------------------ A: Protenix path equality vs shipped v4.0.4r1
if "A" in SECTIONS:
    sys.path.insert(0, os.environ["FPF_SPEC_DIR"])
    import fpf, fpf.reference as R
    R.install_fp64_layernorm_fallback(); t0 = time.time(); runner = R.build_runner(); model = runner.model.eval()
    print("[A] model built in %.0fs" % (time.time() - t0), flush=True)
    pb = model.pairformer_stack.blocks[0]; MODS = {"out": pb.tri_mul_out, "in": pb.tri_mul_in}
    OLD = importlib.import_module(args.v404_pkg + ".trimul"); OLDPKG = importlib.import_module(args.v404_pkg)
    print("[A] old package", OLDPKG.__version__, "from", os.path.dirname(OLDPKG.__file__), flush=True)
    assert OLDPKG.__version__ == "4.0.4", OLDPKG.__version__
    def call(fnx, m, z, mask, residual):
        with torch.no_grad(), R.amp():
            return fnx(m, z, mask=mask, inplace_safe=residual, _add_with_inplace=residual, _inplace_chunk_size=256, triangle_multiplicative="cuequivariance")
    nfail = 0
    for N in (356, 705):
        d = dump(N)
        for key in ("pf_c1_b0", "pf_c10_b47"):
            z0 = d[key]["z"].to(dev).contiguous()
            ones = torch.ones(N, N, device=dev, dtype=torch.float32)
            for dname, m in MODS.items():
                for residual in (True, False):
                    for mk, mask in (("none", None), ("ones", ones)):
                        a = call(V41.fn, m, z0.clone(), mask, residual); b = call(OLD.fn, m, z0.clone(), mask, residual)
                        row = {"N": N, "key": key, "dir": dname, "residual": residual, "mask": mk, "equal_v410_fn_vs_v404r1_fn": bool(torch.equal(a, b)), "n_diff": int((a != b).sum())}
                        if mk == "none":
                            # generic.trimul with raw tensors taken from the module (module-agnostic call) must equal fn too
                            with torch.no_grad(), R.amp():
                                g = G.trimul(z0.clone(), None, direction="outgoing" if m._outgoing else "incoming", residual=residual,
                                             ln_in_w=m.layer_norm_in.weight, ln_in_b=m.layer_norm_in.bias, w_ag=m.linear_a_g.weight, w_ap=m.linear_a_p.weight,
                                             w_bg=m.linear_b_g.weight, w_bp=m.linear_b_p.weight, ln_out_w=m.layer_norm_out.weight, ln_out_b=m.layer_norm_out.bias,
                                             w_o=m.linear_z.weight, w_og=m.linear_g.weight)
                            row["equal_generic_raw_vs_v404r1_fn"] = bool(torch.equal(g, b))
                        ok = row["equal_v410_fn_vs_v404r1_fn"] and row.get("equal_generic_raw_vs_v404r1_fn", True)
                        nfail += 0 if ok else 1
                        RES["A"].append(row); print("[A]", json.dumps(row), flush=True)
            del z0
        del d; save()
    print("[A] SUMMARY: %d/%d cells identical (v4.1.0 fn & generic == v4.0.4r1 fn)" % (len(RES["A"]) - nfail, len(RES["A"])), flush=True)
    print("[A] v4.1.0 COUNTS", json.dumps(V41.COUNTS, default=str)); print("[A] v4.0.4 COUNTS", json.dumps(OLD.COUNTS, default=str)); print("[A] generic COUNTS", json.dumps(G.COUNTS, default=str), flush=True)
    torch.cuda.empty_cache()

# ------------------------------------------------------------------ B/C: module-agnostic unit tests + speed
PREREG = {(256, 256, 705): 1.85, (256, 256, 546): 1.15, (128, 128, 705): 0.75, (128, 128, 546): 0.45, (128, 256, 705): 1.25, (128, 256, 546): 0.75, (256, 128, 705): 1.20, (256, 128, 546): 0.72}
def make_weights(C, D, seed, bias=True):
    g = torch.Generator(device="cpu").manual_seed(seed)
    def lin(o, i, scale): return (torch.randn(o, i, generator=g) * scale / i ** 0.5).to(dev)
    def vec(n, scale, base=0.0): return (base + torch.randn(n, generator=g) * scale).to(dev)
    W = {"ln_in_w": vec(C, 0.1, 1.0), "ln_in_b": vec(C, 0.1), "w_ag": lin(D, C, 1.0), "w_ap": lin(D, C, 1.0), "w_bg": lin(D, C, 1.0), "w_bp": lin(D, C, 1.0),
         "ln_out_w": vec(D, 0.1, 1.0), "ln_out_b": vec(D, 0.1), "w_o": lin(C, D, 1.0), "w_og": lin(C, C, 1.0)}
    if bias:
        W.update({"b_ag": vec(D, 0.5), "b_ap": vec(D, 0.2), "b_bg": vec(D, 0.5), "b_bp": vec(D, 0.2), "b_o": vec(C, 0.2), "b_og": vec(C, 0.5)})
    return W

def torch_ref(z, mask, direction, W, dtype, residual=False, autocast_bf16=False):
    with torch.no_grad():
        if autocast_bf16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return G.reference_torch(z.float(), mask, direction=direction, residual=residual, dtype=torch.float32, **W)
        return G.reference_torch(z, mask, direction=direction, residual=residual, dtype=dtype, **W)

if "B" in SECTIONS or "C" in SECTIONS:
    sizes = [int(s) for s in args.sizes.split(",")]
    combos = [(128, 128), (256, 256), (128, 256), (256, 128)]
    for N in sizes:
        d = dump(N); zfull = d["pf_c10_b47"]["z"].to(dev).contiguous(); del d          # real pairformer activation (std ~60); channel slice for C=128
        gmask = (torch.rand(N, N, generator=torch.Generator(device="cpu").manual_seed(7)) < 0.9).float().to(dev)
        for (C, D) in combos:
            z = zfull[..., :C].contiguous()
            W = make_weights(C, D, seed=1000 + C + D)
            wp = G.pack_weights(**W)
            for direction in ("outgoing", "incoming"):
                row = {"N": N, "C": C, "D": D, "dir": direction, "bias": True, "mask": "random_p0.9"}
                try:
                    if "B" in SECTIONS:
                        ref64 = torch_ref(z, gmask, direction, W, torch.float64)
                        v_bf = G.trimul(z, gmask, direction=direction, weights=wp)
                        v_f32 = G.trimul(z.float(), gmask, direction=direction, weights=wp)
                        t_f32 = torch_ref(z, gmask, direction, W, torch.float32)
                        t_bf = torch_ref(z, gmask, direction, W, None, autocast_bf16=True)
                        e = {"v4_bf16": errs(v_bf, ref64), "v4_fp32in": errs(v_f32, ref64), "torch_fp32": errs(t_f32, ref64), "torch_bf16_autocast": errs(t_bf, ref64)}
                        row["err_vs_fp64"] = e; row["ref64_rms"] = float(ref64.pow(2).mean().sqrt())
                        for k in ("v4_bf16", "v4_fp32in"):
                            row["ratio_%s_vs_torch_bf16_autocast" % k] = {"max": e[k]["max_abs"] / e["torch_bf16_autocast"]["max_abs"], "rms": e[k]["rms"] / e["torch_bf16_autocast"]["rms"]}
                            row["ratio_%s_vs_torch_fp32" % k] = {"max": e[k]["max_abs"] / max(e["torch_fp32"]["max_abs"], 1e-30), "rms": e[k]["rms"] / max(e["torch_fp32"]["rms"], 1e-30)}
                        row["finite"] = bool(torch.isfinite(v_bf).all() and torch.isfinite(v_f32).all())
                        # r2r
                        row["r2r_bitwise_5"] = all(torch.equal(G.trimul(z, gmask, direction=direction, weights=wp), v_bf) for _ in range(5))
                        # mask None vs ones
                        ones = torch.ones(N, N, device=dev)
                        row["mask_none_eq_ones"] = bool(torch.equal(G.trimul(z, None, direction=direction, weights=wp), G.trimul(z, ones, direction=direction, weights=wp)))
                        # residual path (bf16 + fp32 in)
                        v_res = G.trimul(z, gmask, direction=direction, weights=wp, residual=True); ref64r = torch_ref(z, gmask, direction, W, torch.float64, residual=True)
                        t_bfr = torch_ref(z, gmask, direction, W, None, residual=True, autocast_bf16=True).to(torch.bfloat16)   # a bf16 module would hold z+update in bf16
                        er = errs(v_res, ref64r); etr = errs(t_bfr, ref64r); row["residual_ratio_vs_torch_bf16"] = {"max": er["max_abs"] / etr["max_abs"], "rms": er["rms"] / etr["rms"]}
                        # fused projection spec == four-matrix spec (bit-exact)
                        order = ("ap", "bp", "ag", "bg")
                        wfused = torch.cat([W["w_" + k] for k in order], 0); bfused = torch.cat([W["b_" + k] for k in order], 0)
                        wp2 = G.pack_weights(ln_in_w=W["ln_in_w"], ln_in_b=W["ln_in_b"], w_proj=wfused, b_proj=bfused, proj_split=order, ln_out_w=W["ln_out_w"], ln_out_b=W["ln_out_b"],
                                             w_o=W["w_o"], w_og=W["w_og"], b_o=W["b_o"], b_og=W["b_og"])
                        row["fused_spec_eq_4mat"] = bool(torch.equal(G.trimul(z, gmask, direction=direction, weights=wp2), v_bf))
                        # no-bias variant numerics (HAS_BIAS=False path at this (C,D))
                        Wnb = {k: v for k, v in W.items() if not k.startswith("b_")}; wpnb = G.pack_weights(**Wnb)
                        enb = errs(G.trimul(z, gmask, direction=direction, weights=wpnb), torch_ref(z, gmask, direction, Wnb, torch.float64)); etb = errs(torch_ref(z, gmask, direction, Wnb, None, autocast_bf16=True), torch_ref(z, gmask, direction, Wnb, torch.float64))
                        row["nobias_ratio_vs_torch_bf16"] = {"max": enb["max_abs"] / etb["max_abs"], "rms": enb["rms"] / etb["rms"]}
                        # CUDA graph: capture on z, replay on a NEW input (permuted z) == eager on that input
                        try:
                            zs = z.clone(); ms = gmask.clone()
                            G.trimul(zs, ms, direction=direction, weights=wp); torch.cuda.synchronize()
                            g = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(g):
                                og = G.trimul(zs, ms, direction=direction, weights=wp)
                            g.replay(); torch.cuda.synchronize(); same_in = bool(torch.equal(og, v_bf))
                            perm = torch.randperm(N, generator=torch.Generator(device="cpu").manual_seed(3)).to(dev)
                            znew = z[perm][:, perm].contiguous(); mnew = gmask[perm][:, perm].contiguous()
                            zs.copy_(znew); ms.copy_(mnew); g.replay(); torch.cuda.synchronize()
                            eager_new = G.trimul(znew, mnew, direction=direction, weights=wp)
                            row["graph_same_input_eq"] = same_in; row["graph_new_input_eq_eager"] = bool(torch.equal(og, eager_new))
                            del g, og, zs, ms
                        except Exception as ex:
                            row["graph_error"] = repr(ex)[:200]
                        del ref64, v_f32, t_f32, t_bf, v_res, ref64r, t_bfr
                    if "C" in SECTIONS:
                        f_v4 = lambda: G.trimul(z, gmask, direction=direction, weights=wp)
                        f_tbf = lambda: torch_ref(z, gmask, direction, W, None, autocast_bf16=True)
                        f_t32 = lambda: torch_ref(z, gmask, direction, W, torch.float32)
                        zf = z.float(); f_v4f = lambda: G.trimul(zf, gmask, direction=direction, weights=wp)
                        row["ms_v4_bf16"] = timeit(f_v4, args.nrep); row["ms_v4_fp32in"] = timeit(f_v4f, args.nrep); row["ms_torch_bf16_autocast"] = timeit(f_tbf, args.nrep); row["ms_torch_fp32"] = timeit(f_t32, max(5, args.nrep // 2))
                        row["x_vs_torch_bf16"] = row["ms_torch_bf16_autocast"] / row["ms_v4_bf16"]; row["x_vs_torch_fp32"] = row["ms_torch_fp32"] / row["ms_v4_bf16"]
                        pre = PREREG.get((C, D, N)); row["prereg_ms"] = pre; row["prereg_HIT"] = None if pre is None else bool(row["ms_v4_bf16"] <= pre)
                        # stage split
                        Np = K.ceil_to(N, 16); cfg = G._check(z, gmask, wp, None, None); k1c, k3c = K.resolve_cfg(cfg, C, D, True)
                        z4 = z.unsqueeze(0)                                      # the raw launches take the BATCHED buffers of kernels.trimul_v4_forward (B = 1 here): z [B, N, N, C],
                        ab = torch.empty((2, 1, D, Np, Np), dtype=torch.bfloat16, device=dev); x = torch.empty((1, D, Np, Np), dtype=torch.bfloat16, device=dev); o = torch.empty_like(z4)   # ab [2, B, D, Np, Np], x [B, D, Np, Np], one [N, N] mask = MASK_SHARED
                        row["stage_ms"] = {"k1": timeit(lambda: K.launch_k1(z4, wp, gmask, ab, N, Np, k1c, 1e-5, K.MASK_SHARED), args.nrep), "bmm": timeit(lambda: K.launch_bmm(ab, x, direction == "outgoing"), args.nrep),
                                           "k3": timeit(lambda: K.launch_k3(x, z4, wp, o, N, Np, k3c, False, True), args.nrep)}
                        row["cell"] = dict(zip(("k1", "k3"), K.resolve_cfg(cfg, C, D, True)))
                        del ab, x, o, z4, zf
                except Exception as ex:
                    row["ERROR"] = repr(ex)[:400]
                    import traceback; traceback.print_exc()
                RES["B" if "B" in SECTIONS else "C"].append(row)
                short = {k: v for k, v in row.items() if k not in ("err_vs_fp64", "cell")}
                print("[BC]", json.dumps(short, default=lambda o: round(o, 4) if isinstance(o, float) else str(o)), flush=True)
                save()
            del z, wp
        del zfull; torch.cuda.empty_cache()
    print("[generic COUNTS]", json.dumps(G.COUNTS, default=str), flush=True)

# ------------------------------------------------------------------ P: pad=8 planes (ESMFold2 port datapoint): numerics, r2r, graph, pad8-vs-pad16 bits, bmm + module ms
if "P" in SECTIONS:
    RES["P"] = []
    for N in (546, 705):
        d = dump(N); zfull = d["pf_c10_b47"]["z"].to(dev).contiguous(); del d
        gmask = (torch.rand(N, N, generator=torch.Generator(device="cpu").manual_seed(7)) < 0.9).float().to(dev)
        for (C, D, bias) in [(256, 256, False), (128, 128, True)]:
            z = zfull[..., :C].contiguous(); W = make_weights(C, D, seed=1000 + C + D, bias=bias); wp = G.pack_weights(**W)
            for direction in ("outgoing", "incoming"):
                row = {"N": N, "C": C, "D": D, "bias": bias, "dir": direction}
                try:
                    ref64 = torch_ref(z, gmask, direction, W, torch.float64); t_bf = torch_ref(z, gmask, direction, W, None, autocast_bf16=True); etb = errs(t_bf, ref64)
                    outs = {}
                    for pad in (16, 8):
                        o = G.trimul(z, gmask, direction=direction, weights=wp, pad=pad); e = errs(o, ref64)
                        row["Np_pad%d" % pad] = K.ceil_to(N, pad); row["ratio_pad%d_vs_torch_bf16" % pad] = {"max": e["max_abs"] / etb["max_abs"], "rms": e["rms"] / etb["rms"]}
                        row["r2r_pad%d" % pad] = all(torch.equal(G.trimul(z, gmask, direction=direction, weights=wp, pad=pad), o) for _ in range(3)); outs[pad] = o
                        Np = K.ceil_to(N, pad); cfg = G._check(z, gmask, wp, None, None); k1, k3 = K.resolve_cfg(cfg, C, D, bias)
                        z4 = z.unsqueeze(0); ab = torch.empty((2, 1, D, Np, Np), dtype=torch.bfloat16, device=dev); x = torch.empty((1, D, Np, Np), dtype=torch.bfloat16, device=dev); oo = torch.empty_like(z4)   # batched raw-launch buffers (B = 1)
                        K.launch_k1(z4, wp, gmask, ab, N, Np, k1, 1e-5, K.MASK_SHARED)
                        row["ms_pad%d" % pad] = {"module": timeit(lambda: G.trimul(z, gmask, direction=direction, weights=wp, pad=pad), args.nrep), "k1": timeit(lambda: K.launch_k1(z4, wp, gmask, ab, N, Np, k1, 1e-5, K.MASK_SHARED), args.nrep),
                                                 "bmm": timeit(lambda: K.launch_bmm(ab, x, direction == "outgoing"), args.nrep), "k3": timeit(lambda: K.launch_k3(x, z4, wp, oo, N, Np, k3, False, True), args.nrep)}
                        try:
                            from torch.profiler import profile, ProfilerActivity
                            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                                K.launch_bmm(ab, x, direction == "outgoing"); torch.cuda.synchronize()
                            row["bmm_kernel_pad%d" % pad] = [e_.key for e_ in prof.key_averages() if e_.device_time_total > 0 and any(s in e_.key.lower() for s in ("nvjet", "cutlass", "gemm", "xmma"))][:3]
                        except Exception as ex:
                            row["bmm_kernel_pad%d" % pad] = repr(ex)[:80]
                        if pad == 8:      # graph replay on new input at pad 8
                            zs = z.clone(); ms_ = gmask.clone(); G.trimul(zs, ms_, direction=direction, weights=wp, pad=8); torch.cuda.synchronize()
                            g = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(g):
                                og = G.trimul(zs, ms_, direction=direction, weights=wp, pad=8)
                            perm = torch.randperm(N, generator=torch.Generator(device="cpu").manual_seed(5)).to(dev); znew = z[perm][:, perm].contiguous(); mnew = gmask[perm][:, perm].contiguous()
                            zs.copy_(znew); ms_.copy_(mnew); g.replay(); torch.cuda.synchronize()
                            row["graph_new_input_eq_eager_pad8"] = bool(torch.equal(og, G.trimul(znew, mnew, direction=direction, weights=wp, pad=8))); del g, og, zs, ms_
                        del ab, x, oo
                    row["pad8_bitwise_eq_pad16"] = bool(torch.equal(outs[8], outs[16])); row["pad8_vs_pad16_n_diff"] = int((outs[8] != outs[16]).sum())
                    del outs, ref64, t_bf
                except Exception as ex:
                    row["ERROR"] = repr(ex)[:400]; import traceback; traceback.print_exc()
                RES["P"].append(row); print("[P]", json.dumps(row, default=lambda o: round(o, 4) if isinstance(o, float) else str(o)), flush=True); save()
            del z, wp
        del zfull; torch.cuda.empty_cache()
save(); print("DONE")
