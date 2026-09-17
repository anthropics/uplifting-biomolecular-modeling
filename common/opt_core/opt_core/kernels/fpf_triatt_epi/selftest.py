#!/usr/bin/env python3
"""selftest.py — FPF TriAttEpi: numerics + timing driver (one process, model built once).
   python -m fpf_triatt_epi.selftest --dumps $FPF_DUMPS --sizes 356,546,705,813 --out out/ [--sweep] [--keys pf_c1_b0,pf_c10_b47]
Writes: out/EPI_RESULTS.json, out/EPI_TABLE.md, appends out/OPTABLE_epi.csv (fpf.optable row format + engine_stock_class etc.)."""
import os, sys, json, time, argparse, math, csv, copy
import torch, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--dumps", default=os.environ.get("FPF_DUMPS", "dumps")); ap.add_argument("--sizes", default="356,546,705,813")
ap.add_argument("--keys", default="pf_c1_b0,pf_c10_b47"); ap.add_argument("--out", default="out"); ap.add_argument("--sweep", action="store_true")
ap.add_argument("--reps", type=int, default=20); ap.add_argument("--spec", default=os.environ.get("FPF_SPEC_DIR", "")); ap.add_argument("--no-model", action="store_true")
ap.add_argument("--prereg", default="356:0.14,546:0.31,705:0.52,813:0.69")     # block-mode (starting node) pre-registered targets, N:value
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
if a.spec: sys.path.insert(0, a.spec)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fpf, fpf.reference as R, fpf.optable as T


def _install_fp64_layernorm_fallback():
    """fast_layernorm has no fp64 kernel: a float64 reference call raises, or a script silently uses stock as its reference. Patch FusedLayerNorm.forward
    to F.layer_norm for float64 inputs ONLY; bf16/fp32 stock paths untouched."""
    from protenix.model.layer_norm.layer_norm import FusedLayerNorm
    if getattr(FusedLayerNorm, "_fpf_fp64_patched", False): return
    _orig = FusedLayerNorm.forward
    def forward(self, input):
        if input.dtype == torch.float64:
            w = self.weight.to(torch.float64) if self.weight is not None else None
            b = self.bias.to(torch.float64) if self.bias is not None else None
            return F.layer_norm(input, self.normalized_shape, w, b, self.eps)
        return _orig(self, input)
    FusedLayerNorm.forward = forward; FusedLayerNorm._fpf_fp64_patched = True
    print("[selftest] fp64 LayerNorm fallback installed (ref64 only)", flush=True)


_install_fp64_layernorm_fallback()
from fpf_triatt_epi import epilogue as E
from protenix.model.utils import permute_final_dims, flatten_final_dims
import protenix.model.triangular.layers as TL

GPU = torch.cuda.get_device_name()
print("GPU", GPU, "torch", torch.__version__, "triton", __import__("triton").__version__, "cfg_sha256", E.config_sha256()[:16], flush=True)
PREREG = dict((int(k), float(v)) for k, v in (x.split(":") for x in a.prereg.split(",")))
RES = {"gpu": GPU, "torch": torch.__version__, "config_sha256": E.config_sha256(), "pinned_config": {repr(k): v for k, v in E.PINNED_CONFIG.items()},
       "env_cfg_override": os.environ.get("FPF_TRIATT_EPI_CFG", ""), "cases": [], "timing": [], "sweep": [], "started": time.strftime("%FT%TZ", time.gmtime())}

t0 = time.time()
runner = R.build_runner(); model = runner.model.eval(); D = fpf.dims(model); pb = model.pairformer_stack.blocks[0]
print("model built %.0fs" % (time.time() - t0), "dims", json.dumps(D["pairformer"]), flush=True)
RES["dims"] = D["pairformer"]
MODS = {"triatt_start": pb.tri_att_start, "triatt_end": pb.tri_att_end}
print("tri_att_start.starting =", pb.tri_att_start.starting, " tri_att_end.starting =", pb.tri_att_end.starting, " type:", type(pb.tri_att_end).__name__, flush=True)
RES["tri_att_end_starting_attr"] = bool(pb.tri_att_end.starting)


def evms(fn, reps=a.reps, warm=3):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
    for _ in range(reps): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / reps


def load(N):
    p = os.path.join(a.dumps, f"acts_{N}tok.pt")
    if os.path.exists(p):
        return torch.load(p, map_location="cuda"), "real"
    torch.manual_seed(N)
    d = {"pf_c1_b0": {"z": (torch.randn(N, N, 256, device="cuda") * 8).to(torch.bfloat16), "s": (torch.randn(N, 384, device="cuda") * 20).to(torch.bfloat16)},
         "pf_c10_b47": {"z": (torch.randn(N, N, 256, device="cuda") * 25).to(torch.bfloat16), "s": (torch.randn(N, 384, device="cuda") * 60).to(torch.bfloat16)}}
    return d, "synthetic"


@torch.no_grad()
def stock_pieces(module, x):
    """Run stock TriangleAttention pieces exactly as layers.Attention.forward, returning intermediates (o cuEq layout, g_lin, stock update u)."""
    mha = module.mha
    with R.amp():
        mask = x.new_ones(x.shape[:-1])
        xl = module.layer_norm(x)
        mask_bias = (module.inf * (mask - 1))[..., :, None, None, :]
        tb = permute_final_dims(module.linear(xl), (2, 0, 1)).unsqueeze(-4)
        q, k, v = mha._prep_qkv(xl, xl, apply_scale=False)
        scale = 1.0 / math.sqrt(mha.c_hidden)
        o = TL.cuequivariance_triangular_attn(q, k, v, tb.float(), (mask_bias == 0).bool(), scale)[0]
        o_t = o.transpose(-2, -3)
        u = mha._wrap_up(o_t, xl)                       # stock epilogue (incl. linear_g GEMM + sigmoid + mul + flatten + linear_o)
        g_lin = mha.linear_g(xl)
    return o, g_lin, u, xl


@torch.no_grad()
def stock_epi_pieces_timer(module, o, xl, g_lin, z, ending):
    """time the STOCK epilogue pieces this kernel replaces: sigmoid + mul + flatten + linear_o + residual (+ transpose.contiguous for ending). g GEMM excluded."""
    mha = module.mha
    def f():
        with R.amp():
            g = torch.sigmoid(g_lin)
            g = g.view(g.shape[:-1] + (mha.no_heads, -1))
            oo = o.transpose(-2, -3) * g
            oo = flatten_final_dims(oo, 2)
            u = mha.linear_o(oo)
            zz = z  # in-place add into a scratch z
            zz += u
            if ending:
                zz = zz.transpose(-2, -3).contiguous()
        return zz
    return f


def bits_equal(x, y): return bool(torch.equal(x, y))
def maxabs(x, y): return float((x.float() - y.float()).abs().max())
def n_diff(x, y): return int((x != y).sum())


hog_A = None
def with_hog(fn):
    global hog_A
    if hog_A is None: hog_A = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        B_ = hog_A
        for _ in range(30): B_ = (B_ @ hog_A.T) * 1e-3
    out = fn(); torch.cuda.synchronize(); return out


rows_csv = []
HDR = T.HDR + ["real_dumps", "gpu", "dump_key", "mode", "config_sha256", "cotenancy_stream", "n_elem_diff_vs_stock", "maxabs_vs_stock"]
for N in [int(s) for s in a.sizes.split(",")]:
    d, src = load(N)
    for key in a.keys.split(","):
        if key not in d: print("missing key", key); continue
        z0 = d[key]["z"].cuda().contiguous()
        for opname, module in MODS.items():
            ending = (opname == "triatt_end")
            # ---- the frame stock runs this module in: start: x = z ; end: caller passes zT (contiguous) and (since module.starting==True in v2.0.0) x = zT
            zT = z0.transpose(-2, -3).contiguous()
            x_in = zT if ending else z0
            o, g_lin, u_stock, xl = stock_pieces(module, x_in)
            if N == int(a.sizes.split(",")[0]) and key == a.keys.split(",")[0]:
                print(f"[{opname}] o shape {tuple(o.shape)} strides {o.stride()} dtype {o.dtype}; g {tuple(g_lin.shape)} {g_lin.stride()}; u {tuple(u_stock.shape)} {u_stock.dtype}", flush=True)
            wo16 = E._cache(module, x_in)["wo16"]
            # ---- (A) op mode: epilogue vs stock _wrap_up on bit-identical (o, g)
            with R.amp():
                u_mine = E.triatt_epilogue(o, g_lin, wo16, ending=False, residual=False)
            u_mine = u_mine.reshape(u_stock.shape)
            caseA = {"N": N, "key": key, "op": opname, "test": "op_mode_vs_stock_wrapup", "bitwise": bits_equal(u_mine, u_stock), "n_diff": n_diff(u_mine, u_stock),
                     "numel": u_stock.numel(), "maxabs": maxabs(u_mine, u_stock)}
            # torch emulation of the stock math (sanity: tells whether a mismatch is in the elementwise part or the GEMM)
            with R.amp():
                u_emul = E.torch_reference_epilogue(o, g_lin, wo16)
            caseA["torch_emulation_bitwise_vs_stock"] = bits_equal(u_emul.reshape(u_stock.shape), u_stock)
            # gated intermediate check: my elementwise == stock elementwise?  (stock: sigmoid(linear_g) ; o*g ; flatten)
            with R.amp():
                gs = torch.sigmoid(g_lin).view(g_lin.shape[:-1] + (module.mha.no_heads, -1)); gated_stock = flatten_final_dims(o.transpose(-2, -3) * gs, 2)
                u_from_stock_gated = module.mha.linear_o(gated_stock)
            caseA["stock_pieces_reassembled_bitwise_vs_wrapup"] = bits_equal(u_from_stock_gated, u_stock)
            # r2r x5 + cotenancy
            with R.amp():
                outs = [E.triatt_epilogue(o, g_lin, wo16).reshape(u_stock.shape) for _ in range(5)]
            caseA["r2r5_bitwise"] = all(bits_equal(q, outs[0]) for q in outs)
            with R.amp():
                co = with_hog(lambda: E.triatt_epilogue(o, g_lin, wo16).reshape(u_stock.shape))
            caseA["cotenancy_stream_bitwise"] = bits_equal(co, outs[0])
            # token-major o input (stretch: SDPA-style [I,J,H,D]) must give identical bits
            with R.amp():
                o_tok = o.transpose(-2, -3).contiguous()
                u_tok = E.triatt_epilogue(o_tok, g_lin, wo16, o_layout="ijhd").reshape(u_stock.shape)
            caseA["o_layout_ijhd_bitwise"] = bits_equal(u_tok, u_stock); del o_tok, u_tok
            # row invariance (op is row-separable given o,g): first I/2 rows
            Ih = o.shape[-4] // 2
            with R.amp():
                half = E.triatt_epilogue(o[..., :Ih, :, :, :], g_lin[..., :Ih, :, :], wo16)
            caseA["row_invariance_bitwise"] = bits_equal(half.reshape(-1, u_stock.shape[-1]), outs[0][..., :Ih, :, :].reshape(-1, u_stock.shape[-1]))
            # err vs fp64 of the EPILOGUE given identical (o,g): ref = fp64 math on the same bf16 operands
            o64 = o.transpose(-2, -3).double(); g64 = torch.sigmoid(g_lin.double()); ref = (o64 * g64.view(g64.shape[:-1] + (module.mha.no_heads, -1))).reshape(g64.shape) @ wo16.double().T
            es = (u_stock.double() - ref); ec = (outs[0].double() - ref)
            caseA["epi_err"] = {"stock_max": float(es.abs().max()), "stock_rms": float(es.pow(2).mean().sqrt()), "cand_max": float(ec.abs().max()), "cand_rms": float(ec.pow(2).mean().sqrt())}
            caseA["epi_err"]["ratio_max"] = caseA["epi_err"]["cand_max"] / max(caseA["epi_err"]["stock_max"], 1e-30); caseA["epi_err"]["ratio_rms"] = caseA["epi_err"]["cand_rms"] / max(caseA["epi_err"]["stock_rms"], 1e-30)
            RES["cases"].append(caseA); print(json.dumps(caseA), flush=True)
            # ---- (B) registry fn (whole op) vs stock forward: bit-exact? + timing + err ratios vs fp64 module reference
            kw = dict(mask=None, triangle_attention="cuequivariance", inplace_safe=True, chunk_size=None)
            with R.amp():
                full_stock = R.stock_call(module, "forward", x_in, **kw)
                full_mine = E.fn(module, x_in, **kw)
            caseB = {"N": N, "key": key, "op": opname, "test": "registry_fn_vs_stock_forward", "bitwise": bits_equal(full_mine, full_stock), "n_diff": n_diff(full_mine, full_stock), "maxabs": maxabs(full_mine, full_stock)}
            try:
                r64 = R.ref64_call(module, "forward", x_in, **kw)
                err = R.err_table(full_mine, full_stock, r64); caseB.update({k: err[k] for k in ("ratio_max", "ratio_rms", "cand_max_abs", "stock_max_abs", "cand_rms", "stock_rms")})
                assert err["stock_max_abs"] > 0 and err["stock_rms"] > 0, "ref64 sanity: stock error vs fp64 must be non-zero"
                del r64
            except Exception as ex:
                caseB["ref64_error"] = repr(ex)[:200]; err = None
            with R.amp():
                fo = [E.fn(module, x_in, **kw) for _ in range(5)]
            caseB["r2r5_bitwise"] = all(bits_equal(q, fo[0]) for q in fo)
            with R.amp():
                co = with_hog(lambda: E.fn(module, x_in, **kw))
            caseB["cotenancy_stream_bitwise"] = bits_equal(co, fo[0])
            del fo, co
            if key == a.keys.split(",")[0]:
                ms_stock = evms(lambda: R.stock_call(module, "forward", x_in, **kw)); ms_mine = evms(lambda: (lambda: E.fn(module, x_in, **kw))() if True else None)
                with R.amp():
                    ms_mine = evms(lambda: E.fn(module, x_in, **kw))
                caseB["ms_stock_forward"] = ms_stock; caseB["ms_fn_forward"] = ms_mine
                dd = dict(D["pairformer"]); fl, by = T.flops_bytes("triatt", N, dd)
                lab = "EXACT" if caseB["bitwise"] else ("TIER2-sameclass" if err and err["same_class(<=1.5x)"] else "TIER2?")
                rows_csv.append(T.row(opname, "stock(cuEq/bf16)", N, ms_stock, fl, by, err={"ratio_max": 1.0, "ratio_rms": 1.0, "bitwise_vs_stock": True}, r2r=True, binv="n/a", prereg=None, label=f"inputs={src}",
                                      engine_stock_class="Protenix v2 TriangleAttention.forward (cuEq path) — STOCK")
                                + [src == "real", GPU, key, "whole-op", "", "", 0, 0.0])
                rows_csv.append(T.row(opname, "fpf_triatt_epi.epilogue:fn", N, ms_mine, fl, by, err=(err or {}) | {"bitwise_vs_stock": caseB["bitwise"]}, r2r=caseB["r2r5_bitwise"], binv=caseA["row_invariance_bitwise"], prereg={356: 1.8, 546: 5.0, 705: 9.7, 813: 14.0}.get(N), label=lab,
                                      engine_stock_class=E.ENGINE_STOCK_CLASS)
                                + [src == "real", GPU, key, "whole-op (stock prologue+cuEq + fused epilogue, op mode)", E.config_sha256()[:16], caseB["cotenancy_stream_bitwise"], caseB["n_diff"], caseB["maxabs"]])
            RES["cases"].append(caseB); print(json.dumps(caseB), flush=True)
            del full_stock, full_mine
            # ---- (C) block mode: in-place residual (+ transposed scatter for ending) vs the stock statements on the UNtransposed z
            z_ref = z0.clone()
            with R.amp():
                if ending:
                    zz = z_ref.transpose(-2, -3).contiguous(); zz += R.stock_call(module, "forward", zz, **kw); z_ref = zz.transpose(-2, -3).contiguous(); del zz
                else:
                    z_ref += R.stock_call(module, "forward", z_ref, **kw)
            z_mine = z0.clone()
            with R.amp():
                E.fn_block_residual(module, z_mine, ending=ending)
            caseC = {"N": N, "key": key, "op": opname, "test": "block_mode_inplace_vs_stock_statements", "ending": ending, "bitwise": bits_equal(z_mine, z_ref), "n_diff": n_diff(z_mine, z_ref), "maxabs": maxabs(z_mine, z_ref)}
            # block mode with the exact stock (o,g): isolates the kernel's residual/scatter from prologue differences
            z_k = z0.clone()
            with R.amp():
                E.triatt_epilogue(o, g_lin, wo16, z_k, ending=ending, residual=True)
            # reference for that: z0 (+ transposed) u_stock
            z_k_ref = z0.clone()
            if ending:
                zt = z_k_ref.transpose(-2, -3); zt += u_stock      # values: z[j,i] += u[i,j]
            else:
                z_k_ref += u_stock
            caseC["kernel_residual_given_stock_og_bitwise"] = bits_equal(z_k, z_k_ref); caseC["kernel_residual_n_diff"] = n_diff(z_k, z_k_ref)
            with R.amp():
                zs = [None] * 5
                for r_ in range(5):
                    zs[r_] = z0.clone(); E.triatt_epilogue(o, g_lin, wo16, zs[r_], ending=ending, residual=True)
            caseC["r2r5_bitwise"] = all(bits_equal(q, zs[0]) for q in zs); del zs
            RES["cases"].append(caseC); print(json.dumps(caseC), flush=True)
            del z_ref, z_mine, z_k, z_k_ref
            # ---- (D) timing: epilogue alone (op mode / block mode) vs stock epilogue pieces; only on first key
            if key == a.keys.split(",")[0]:
                zscr = z0.clone()
                with R.amp():
                    t_op = evms(lambda: E.triatt_epilogue(o, g_lin, wo16))
                    t_blk = evms(lambda: E.triatt_epilogue(o, g_lin, wo16, zscr, ending=ending, residual=True))
                    zscr2 = (zT if ending else z0).clone()
                    t_stock = evms(stock_epi_pieces_timer(module, o, xl, g_lin, zscr2, ending))
                    # individual stock pieces
                    gsig = torch.sigmoid(g_lin); gv = gsig.view(g_lin.shape[:-1] + (module.mha.no_heads, -1)); o_t = o.transpose(-2, -3)
                    t_sig = evms(lambda: torch.sigmoid(g_lin)); t_mul = evms(lambda: o_t * gv); om = o_t * gv
                    t_flat = evms(lambda: flatten_final_dims(om, 2)); of = flatten_final_dims(om, 2)
                    t_lin = evms(lambda: module.mha.linear_o(of)); ul = module.mha.linear_o(of)
                    t_add = evms(lambda: zscr2.add_(ul) if not ending else zscr2.add_(ul)); t_tr = evms(lambda: zscr2.transpose(-2, -3).contiguous()) if ending else 0.0
                zbytes = N * N * 256 * 2
                tim = {"N": N, "op": opname, "key": key, "inputs": src, "ms_epilogue_op_mode": t_op, "ms_epilogue_block_mode": t_blk, "ms_stock_epilogue_pieces": t_stock,
                       "stock_pieces_ms": {"sigmoid": t_sig, "mul": t_mul, "flatten_copy": t_flat, "linear_o": t_lin, "residual_add": t_add, "transpose_contig": t_tr},
                       "prereg_block_ms": PREREG.get(N), "speedup_block_vs_stock_pieces": t_stock / t_blk,
                       "GBs_block_mode(4 z-passes)": 4 * zbytes / t_blk / 1e6, "GBs_op_mode(3 z-passes)": 3 * zbytes / t_op / 1e6, "TFLOPs_block": 2 * N * N * 256 * 256 / t_blk / 1e9}
                RES["timing"].append(tim); print("TIMING", json.dumps(tim), flush=True)
                fl_e = 2 * N * N * 256 * 256 + 4 * N * N * 256; by_e = 4 * zbytes + 256 * 256 * 2
                rows_csv.append(T.row(opname + "_epilogue", "stock pieces (sigmoid+mul+flatten+linear_o+residual" + ("+transpose)" if ending else ")"), N, t_stock, fl_e, by_e, err={"ratio_max": 1.0, "ratio_rms": 1.0, "bitwise_vs_stock": True}, r2r=True, binv="n/a", prereg={356: 0.45, 546: 1.0, 705: 1.7, 813: 2.2}.get(N) if not ending else {356: 0.55, 546: 1.25, 705: 2.1, 813: 2.8}.get(N), label=f"inputs={src}",
                                      engine_stock_class="Protenix v2 layers.Attention._wrap_up tail (sigmoid, mul, flatten, linear_o) + PairformerBlock residual add" + (" + transpose.contiguous" if ending else "") + " — STOCK")
                                + [src == "real", GPU, key, "epilogue-only", "", "", 0, 0.0])
                rows_csv.append(T.row(opname + "_epilogue", "fpf_triatt_epi.epilogue:triatt_epilogue(block mode)", N, t_blk, fl_e, by_e, err={"ratio_max": caseA["epi_err"]["ratio_max"], "ratio_rms": caseA["epi_err"]["ratio_rms"], "bitwise_vs_stock": caseC["kernel_residual_given_stock_og_bitwise"]}, r2r=caseC["r2r5_bitwise"], binv=caseA["row_invariance_bitwise"], prereg=PREREG.get(N),
                                       label=("EXACT" if caseC["kernel_residual_given_stock_og_bitwise"] else "TIER2?"), engine_stock_class=E.ENGINE_STOCK_CLASS)
                                + [src == "real", GPU, key, "epilogue-only block mode (residual%s)" % ("+transposed scatter" if ending else ""), E.config_sha256()[:16], caseA["cotenancy_stream_bitwise"], caseC["kernel_residual_n_diff"], caseC["maxabs"]])
                # ---- optional config sweep (engineering only)
                if a.sweep:
                    for (KV, BI, BJ, W, S) in [(2, 16, 8, 8, 1), (2, 16, 8, 8, 2), (2, 8, 16, 8, 2), (1, 16, 8, 8, 1), (1, 8, 8, 4, 1), (1, 16, 4, 4, 1), (2, 32, 8, 8, 1), (2, 16, 16, 8, 1)]:
                        cfg = dict(KVER=KV, BI=BI, BJ=BJ, num_warps=W, num_stages=S, EXP="libdevice")
                        try:
                            with R.amp():
                                uu = E.triatt_epilogue(o, g_lin, wo16, cfg=cfg).reshape(u_stock.shape); ok = bits_equal(uu, u_stock)
                                tt = evms(lambda: E.triatt_epilogue(o, g_lin, wo16, zscr, ending=ending, residual=True, cfg=cfg), reps=10)
                                to = evms(lambda: E.triatt_epilogue(o, g_lin, wo16, cfg=cfg), reps=10)
                            sw = {"N": N, "op": opname, "cfg": [KV, BI, BJ, W, S], "ms_block": tt, "ms_op": to, "bitwise_vs_stock": ok}
                        except Exception as ex:
                            sw = {"N": N, "op": opname, "cfg": [KV, BI, BJ, W, S], "error": repr(ex)[:150]}
                        RES["sweep"].append(sw); print("SWEEP", json.dumps(sw), flush=True)
                del zscr, zscr2
            del o, g_lin, u_stock, xl, u_mine, outs
            torch.cuda.empty_cache()
        del z0
    del d; torch.cuda.empty_cache()
    json.dump(RES, open(os.path.join(a.out, "EPI_RESULTS.json"), "w"), indent=1)

# ---------------- extra cells: MSA pair-stack tri-att (c=256,H=8) and TEMPLATE-embedder pair-stack tri-att (c=64,H=2,D=32) — check or document fallback
os.environ["FPF_TRIATT_EPI_ALLOW_UNVERIFIED"] = "1"      # checking job: allow pinned-but-unchecked cells to run the kernel so we can qualify them
extra = {}
try:
    mb = model.msa_module.blocks[0].pair_stack; extra["msa_triatt_start"] = mb.tri_att_start; extra["msa_triatt_end"] = mb.tri_att_end
except Exception as ex: print("no msa pair stack:", repr(ex)[:100])
try:
    te = model.template_embedder
    if te.n_blocks > 0:
        tb = te.pairformer_stack.blocks[0]; extra["templ_triatt_start"] = tb.tri_att_start; extra["templ_triatt_end"] = tb.tri_att_end
except Exception as ex: print("no template stack:", repr(ex)[:100])
RES["extra_cells"] = []
for name, module in extra.items():
    mha = module.mha; c_in = mha.linear_o.weight.shape[0]; H = mha.no_heads; Dh = mha.c_hidden
    for N in [int(x) for x in a.sizes.split(",")][:3]:
        d, src = load(N)
        for key in a.keys.split(","):
            if c_in == 256:
                x_in = d[key]["z"].cuda().contiguous(); srcx = src
            else:
                tz = d.get("templ", {}).get("z") if isinstance(d.get("templ"), dict) else None
                if tz is not None:
                    x_in = tz.reshape((-1,) + tuple(tz.shape[-3:]))[0].cuda().to(torch.bfloat16).contiguous(); srcx = "real(templ)"
                else:
                    torch.manual_seed(N + 7); x_in = (torch.randn(N, N, c_in, device="cuda") * (8 if key == "pf_c1_b0" else 25)).to(torch.bfloat16); srcx = "synthetic(templ z absent in dump)"
            kw = dict(mask=None, triangle_attention="cuequivariance", inplace_safe=True, chunk_size=None)
            try:
                with R.amp():
                    st = R.stock_call(module, "forward", x_in, **kw); mine = E.fn(module, x_in, **kw)
                    outs = [E.fn(module, x_in, **kw) for _ in range(3)]
                row = {"cell": name, "c": c_in, "H": H, "D": Dh, "N": N, "key": key, "inputs": srcx, "bitwise": bits_equal(mine, st), "n_diff": n_diff(mine, st), "maxabs": maxabs(mine, st),
                       "r2r_bitwise": all(bits_equal(q, mine) for q in outs), "kernel_calls": dict(E._STATS)}
                if key == a.keys.split(",")[0]:
                    with R.amp():
                        row["ms_stock"] = evms(lambda: R.stock_call(module, "forward", x_in, **kw), reps=10); row["ms_fn"] = evms(lambda: E.fn(module, x_in, **kw), reps=10)
            except Exception as ex:
                row = {"cell": name, "c": c_in, "H": H, "D": Dh, "N": N, "key": key, "error": repr(ex)[:300]}
            RES["extra_cells"].append(row); print("EXTRA", json.dumps(row), flush=True)
            del x_in
        del d; torch.cuda.empty_cache()
RES["finished"] = time.strftime("%FT%TZ", time.gmtime())
json.dump(RES, open(os.path.join(a.out, "EPI_RESULTS.json"), "w"), indent=1)
p = os.path.join(a.out, "OPTABLE_epi.csv"); new = not os.path.exists(p)
with open(p, "a", newline="") as f:
    w = csv.writer(f)
    if new: w.writerow(HDR)
    w.writerows(rows_csv)
# markdown summary
with open(os.path.join(a.out, "EPI_TABLE.md"), "w") as f:
    f.write(f"# TriAttEpi results  GPU={GPU}  cfg_sha={E.config_sha256()[:16]}  env_override='{RES['env_cfg_override']}'\n\n")
    f.write("| N | op | key | test | bitwise | n_diff | maxabs | extra |\n|---|---|---|---|---|---|---|---|\n")
    for c in RES["cases"]:
        extra = {k: v for k, v in c.items() if k not in ("N", "op", "key", "test", "bitwise", "n_diff", "maxabs")}
        f.write(f"| {c['N']} | {c['op']} | {c['key']} | {c['test']} | {c['bitwise']} | {c.get('n_diff')} | {c.get('maxabs')} | {json.dumps(extra)[:300]} |\n")
    f.write("\n## timing (ms/call, CUDA events)\n| N | op | epilogue op-mode | epilogue block-mode | prereg block | stock epilogue pieces | speedup | GB/s block | stock pieces breakdown |\n|---|---|---|---|---|---|---|---|---|\n")
    for t in RES["timing"]:
        f.write(f"| {t['N']} | {t['op']} | {t['ms_epilogue_op_mode']:.4f} | {t['ms_epilogue_block_mode']:.4f} | {t['prereg_block_ms']} | {t['ms_stock_epilogue_pieces']:.4f} | {t['speedup_block_vs_stock_pieces']:.2f}x | {t['GBs_block_mode(4 z-passes)']:.0f} | {json.dumps({k: round(v, 4) for k, v in t['stock_pieces_ms'].items()})} |\n")
    if RES.get("extra_cells"):
        f.write("\n## extra cells (MSA pair stack c=256/H=8; template pair stack c=64/H=2/D=32) — registry fn vs stock forward\n| cell | c,H,D | N | key | inputs | bitwise | n_diff | r2r | ms stock | ms fn |\n|---|---|---|---|---|---|---|---|---|---|\n")
        for r_ in RES["extra_cells"]:
            f.write(f"| {r_['cell']} | {r_['c']},{r_['H']},{r_['D']} | {r_['N']} | {r_['key']} | {r_.get('inputs','')} | {r_.get('bitwise', r_.get('error','ERR')[:60])} | {r_.get('n_diff','')} | {r_.get('r2r_bitwise','')} | {r_.get('ms_stock','')} | {r_.get('ms_fn','')} |\n")
    if RES["sweep"]:
        f.write("\n## sweep (engineering)\n| N | op | cfg BI,BJ,W,S | ms block | ms op | bitwise |\n|---|---|---|---|---|---|\n")
        for s_ in RES["sweep"]:
            f.write(f"| {s_['N']} | {s_['op']} | {s_['cfg']} | {s_.get('ms_block', 'ERR')} | {s_.get('ms_op', '')} | {s_.get('bitwise_vs_stock', s_.get('error', ''))} |\n")
print("| " + " | ".join(HDR) + " |")
for r_ in rows_csv: print("| " + " | ".join(str(x) for x in r_) + " |")
n_fail = sum(1 for c in RES["cases"] if not c["bitwise"])
print("SUMMARY: cases", len(RES["cases"]), "non-bitwise", n_fail, flush=True)
