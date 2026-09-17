#!/usr/bin/env python3
"""fpf_glue_v2/selftest.py — EXACT-vs-shipped unit test on REAL Protenix-v2 activation dumps (acts_{N}tok.pt), 2 sizes, both regimes, start+end.

  source $FPF_HOME/env.sh   (ARM E or T; only PYTHONPATH/PROTENIX_ROOT_DIR/LAYERNORM_TYPE matter here)
  python -m fpf_glue_v2.selftest --dumps <dump dir>[:<dump dir> …] --sizes 356,705[,1493] [--keys pf_c1_b0,pf_c10_b47] [--graph]

For every (N, key, node): builds exactly what the BLK2 block path feeds the kernels (stock fast_layernorm -> x_ln; shipped prologue -> q,k,v,g,bias; cuEq attention with the
stock all-True mask -> o) and asserts torch.equal( glue-v2 kernel output , shipped kernel output ) for prologue (5 tensors), epilogue (block-mode z in place, start
and ending scatter; + op mode), transition (_launch with residual; + without), each run 3x (r2r); + prologue_v4_padded vs shipped triatt_prologue_padded on full ceil8-padded buffers when fpf_triatt_pro provides it.  --graph additionally captures each glue kernel in a CUDA graph, replays it on NEW
inputs (copied into the captured buffers) and asserts equality with eager (CUDA-graph replay safety).  Exit code 0 = all PASS.  Prints one JSON line per check and a summary.
"""
import os, sys, json, math, argparse, time
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--dumps", default=os.environ.get("FPF_DUMPS", "dumps"))
ap.add_argument("--sizes", default="356,705")
ap.add_argument("--keys", default="pf_c1_b0,pf_c10_b47")
ap.add_argument("--graph", action="store_true")
ap.add_argument("--json", default="")
a = ap.parse_args()

dev, bf = "cuda", torch.bfloat16
os.environ.setdefault("PTX_GLUE_V2", "1")
import ptx_trunk2_levers as LEV                       # merges $FPF_HOME/CELLS.json into the shipped pinned tables (current cells per arch)
from fpf_triatt_pro import prologue as PRO
from fpf_triatt_epi import epilogue as EPI
from fpf_transition import transition as TR
def _orig(f): return getattr(f, "__wrapped__", f)
SHIPPED = dict(pro=_orig(PRO.triatt_prologue), epi=_orig(EPI.triatt_epilogue), launch=_orig(TR._launch), pro_pad=(_orig(PRO.triatt_prologue_padded) if hasattr(PRO, "triatt_prologue_padded") else None))     # the shipped callables (unwrapped if the overlay hook already patched them); pro_pad only when fpf_triatt_pro provides triatt_prologue_padded
import fpf_glue_v2 as G2
cells = G2.install()
from fpf_glue_v2 import kernels as GK
import protenix.model.triangular.layers as TL
import fpf.reference as R

print(json.dumps(dict(selftest="fpf_glue_v2", device=torch.cuda.get_device_name(0), cc=torch.cuda.get_device_capability(0), torch=torch.__version__,
                      triton=__import__("triton").__version__, cells={k: v["cfg"] for k, v in cells.items()}, why=G2.stats()["why"])), flush=True)
if not cells:
    print("SELFTEST: no glue-v2 cells for this (cc, triton) -> nothing to test (shipped kernels stay in place). PASS (vacuous)"); sys.exit(0)

runner = R.build_runner(); model = runner.model.eval()
PB = model.pairformer_stack.blocks
BLK_OF_KEY = {"pf_c1_b0": 0, "pf_c10_b0": 0, "pf_c1_b47": 47, "pf_c10_b47": 47}
def amp(): return torch.autocast("cuda", dtype=bf)
RES = []; FAIL = []

def check(name, ok, **kw):
    r = dict(check=name, PASS=bool(ok), **kw); RES.append(r); print(json.dumps(r, default=str), flush=True)
    if not ok: FAIL.append(name)

def tri_inputs(module, z, ending):
    with torch.no_grad(), amp():
        x = z.transpose(-2, -3) if ending else z
        x_ln = module.layer_norm(x)
        if x_ln.dtype != bf: x_ln = x_ln.to(bf)
        if not x_ln.is_contiguous(): x_ln = x_ln.contiguous()
        q, k, v, g, bias = SHIPPED["pro"](module, x_ln, ending=False, ln_mode="stock", x_ln=x_ln)
        scale = 1.0 / math.sqrt(module.mha.c_hidden)
        mask = torch.ones((q.shape[0], 1, 1, k.shape[-2]), dtype=torch.bool, device=z.device)
        o = TL.cuequivariance_triangular_attn(q, k, v, bias.unsqueeze(0), mask, scale)
        o = o[0] if isinstance(o, (tuple, list)) else o
        if o.dim() == 5: o = o[0]
    wo16 = module.mha.linear_o.weight.detach().to(bf).contiguous()
    return dict(x_ln=x_ln, q=q, k=k, v=v, g=g, bias=bias, o=o.contiguous(), wo16=wo16)

def graph_check(name, make_inputs, run):
    """capture run(bufs) with bufs = make_inputs(0); replay after copying make_inputs(1) into bufs; compare with eager run on inputs(1)."""
    bufs = make_inputs(0)
    run(bufs); torch.cuda.synchronize()                               # warm/compile outside capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs_g = run(bufs)
    new = make_inputs(1)
    for kk in bufs:
        if torch.is_tensor(bufs[kk]) and kk in new: bufs[kk].copy_(new[kk])
    g.replay(); torch.cuda.synchronize()
    got = [t.clone() for t in (outs_g if isinstance(outs_g, (tuple, list)) else [outs_g]) if torch.is_tensor(t)] + [bufs[kk].clone() for kk in sorted(bufs) if kk.startswith("inout_")]
    eager = run(new); torch.cuda.synchronize()
    exp = [t for t in (eager if isinstance(eager, (tuple, list)) else [eager]) if torch.is_tensor(t)] + [new[kk] for kk in sorted(new) if kk.startswith("inout_")]
    ok = len(got) == len(exp) and all(torch.equal(a_, b_) for a_, b_ in zip(got, exp))
    check(name, ok, n_tensors=len(got))

for N in [int(s) for s in a.sizes.split(",") if s]:
    path = None
    for base in [p_ for p_ in a.dumps.split(":") if p_]:                       # --dumps may be a ':'-separated list of dirs; file = acts_<N>tok.pt or acts_*_<N>tok.pt (large-systems naming)
        for fn in (sorted(os.listdir(base)) if os.path.isdir(base) else []):
            if fn.startswith("acts_") and fn.endswith(f"_{N}tok.pt") or fn == f"acts_{N}tok.pt": path = os.path.join(base, fn); break
        if path: break
    assert path, f"no acts dump for N={N} under {a.dumps}"
    print(f"dump N={N}: {path}", flush=True)
    d = torch.load(path, map_location="cuda")
    for key in a.keys.split(","):
        if key not in d: check(f"dump_key_{N}_{key}", False, error="missing key"); continue
        blk = PB[BLK_OF_KEY[key]]; z0 = d[key]["z"].to(dev, bf).contiguous()
        for ending in (False, True):
            mod = blk.tri_att_end if ending else blk.tri_att_start
            ti = tri_inputs(mod, z0, ending)
            tag = f"N={N} key={key} node={'end' if ending else 'start'}"
            if "prologue" in cells and not ending or ("prologue" in cells and ending):
                ref = SHIPPED["pro"](mod, ti["x_ln"], ending=False, ln_mode="stock", x_ln=ti["x_ln"])
                cch = PRO.get_cache(mod, ti["x_ln"].device)
                outs = [PRO.triatt_prologue(mod, ti["x_ln"], ending=False, ln_mode="stock", x_ln=ti["x_ln"]) for _ in range(3)]
                used = G2.stats()["calls"]["prologue_v4"]
                ok = all(torch.equal(o_, r_) for o_, r_ in zip(outs[0], ref)); r2r = all(torch.equal(o_, p_) for rep in outs[1:] for o_, p_ in zip(outs[0], rep))
                lay = all(o_.shape == r_.shape and o_.stride() == r_.stride() and o_.dtype == r_.dtype for o_, r_ in zip(outs[0], ref))
                check(f"prologue_v4 torch.equal(q,k,v,g,bias) vs shipped [{tag}]", ok and r2r and lay and used > 0, r2r=r2r, layouts_equal=lay, glue_calls=used)
                # padded entry (present only when fpf_triatt_pro provides triatt_prologue_padded): torch.equal vs shipped triatt_prologue_padded on FULL padded buffers, P = ceil8(N)
                if SHIPPED.get("pro_pad") is not None:
                    H_, D_ = cch["H"], cch["D"]; HD_ = H_ * D_; P = ((N + 7) // 8) * 8
                    def bufs():
                        b_ = {"q": torch.zeros((P, H_, P, D_), dtype=bf, device=dev), "k": torch.zeros((P, H_, P, D_), dtype=bf, device=dev), "v": torch.zeros((P, H_, P, D_), dtype=bf, device=dev),
                              "bias": torch.zeros((1, H_, P, P), dtype=torch.float32, device=dev), "g": torch.zeros((P, P, HD_), dtype=bf, device=dev)}
                        if N < P: b_["bias"][..., N:] = -1e9
                        return b_
                    rb_ = bufs(); refp = SHIPPED["pro_pad"](mod, ti["x_ln"], rb_, ending=False, ln_mode="stock", x_ln=ti["x_ln"])
                    c0 = G2.stats()["calls"].get("prologue_v4_padded", 0)
                    obs = []
                    for _ in range(3):
                        cb_ = bufs(); op_ = PRO.triatt_prologue_padded(mod, ti["x_ln"], cb_, ending=False, ln_mode="stock", x_ln=ti["x_ln"]); obs.append((cb_, op_))
                    usedp = G2.stats()["calls"].get("prologue_v4_padded", 0) - c0
                    okp = all(torch.equal(obs[0][0][nm], rb_[nm]) for nm in ("q", "k", "v", "bias", "g")) and all(torch.equal(o_, r_) for o_, r_ in zip(obs[0][1], refp))
                    r2rp = all(torch.equal(obs[0][0][nm], obs[i][0][nm]) for i in (1, 2) for nm in ("q", "k", "v", "bias", "g"))
                    layp = all(o_.shape == r_.shape and o_.stride() == r_.stride() and o_.dtype == r_.dtype for o_, r_ in zip(obs[0][1], refp))
                    check(f"prologue_v4_padded torch.equal(full padded q,k,v,g,bias; P={P}) vs shipped triatt_prologue_padded [{tag}]", okp and r2rp and layp and usedp > 0, r2r=r2rp, layouts_equal=layp, glue_calls=usedp)
                    del rb_, obs
            if "epilogue" in cells:
                zr = z0.clone(); SHIPPED["epi"](ti["o"], ti["g"], ti["wo16"], zr, ending=ending, residual=True)
                outs = []
                for _ in range(3):
                    zz = z0.clone(); EPI.triatt_epilogue(ti["o"], ti["g"], ti["wo16"], zz, ending=ending, residual=True); outs.append(zz)
                ok = torch.equal(outs[0], zr); r2r = all(torch.equal(outs[0], o_) for o_ in outs[1:])
                mx = float((outs[0].float() - zr.float()).abs().max())
                uref = SHIPPED["epi"](ti["o"], ti["g"], ti["wo16"], ending=False, residual=False); u = EPI.triatt_epilogue(ti["o"], ti["g"], ti["wo16"], ending=False, residual=False)
                check(f"epilogue_v3 block-mode z torch.equal vs shipped [{tag}]", ok and r2r and G2.stats()["calls"]["epilogue_v3"] > 0, r2r=r2r, maxabs=mx, opmode_equal=bool(torch.equal(u, uref)))
                if not torch.equal(u, uref): FAIL.append("epilogue opmode " + tag)
            if a.graph and key == a.keys.split(",")[0] and N == int(a.sizes.split(",")[0]):
                if "epilogue" in cells:
                    ti2 = tri_inputs(mod, (z0 * 0.5).contiguous(), ending)
                    def mk(i, ti=ti, ti2=ti2):
                        s = ti if i == 0 else ti2
                        return dict(o=s["o"].clone(), g=s["g"].clone(), inout_z=(z0 if i == 0 else (z0 * 0.5)).contiguous().clone(), wo16=ti["wo16"])
                    graph_check(f"graph replay epilogue_v3 [{tag}]", mk, lambda b: EPI.triatt_epilogue(b["o"], b["g"], b["wo16"], b["inout_z"], ending=ending, residual=True))
                if "prologue" in cells:
                    ti2 = tri_inputs(mod, (z0 * 0.5).contiguous(), ending)
                    def mkp(i, ti=ti, ti2=ti2): return dict(x_ln=(ti if i == 0 else ti2)["x_ln"].clone())
                    graph_check(f"graph replay prologue_v4 [{tag}]", mkp, lambda b: PRO.triatt_prologue(mod, b["x_ln"], ending=False, ln_mode="stock", x_ln=b["x_ln"]))
            del ti; torch.cuda.empty_cache()
        if "transition" in cells:
            mt = blk.pair_transition
            with torch.no_grad(), amp():
                x2 = z0.reshape(-1, 256); y = mt.layernorm1(x2).to(bf).contiguous(); cache = TR._weights(mt, x2.device)
                ref = torch.empty_like(x2); SHIPPED["launch"](y, cache, ref, res2d=x2, ln_mode=0)
                refu = torch.empty_like(x2); SHIPPED["launch"](y, cache, refu, res2d=None, ln_mode=0)
                outs = []
                for _ in range(3):
                    ob = torch.empty_like(x2); TR._launch(y, cache, ob, res2d=x2, ln_mode=0); outs.append(ob)
                obu = torch.empty_like(x2); TR._launch(y, cache, obu, res2d=None, ln_mode=0)
                full = TR.fn_residual(mt, z0)                                             # public entry (uses the patched _launch)
                full_ref = None
            ok = torch.equal(outs[0], ref); r2r = all(torch.equal(outs[0], o_) for o_ in outs[1:]); oku = torch.equal(obu, refu)
            check(f"transition(+WS) torch.equal vs shipped, residual & op mode [N={N} key={key}]", ok and r2r and oku and G2.stats()["calls"]["transition_ws"] > 0,
                  r2r=r2r, opmode_equal=bool(oku), fn_residual_matches_launch=bool(torch.equal(full.reshape(-1, 256), outs[0])), maxabs=float((outs[0].float() - ref.float()).abs().max()))
            if a.graph and key == a.keys.split(",")[0] and N == int(a.sizes.split(",")[0]):
                y2 = mt.layernorm1((z0 * 0.5).reshape(-1, 256)).to(bf).contiguous()
                def mkt(i): return dict(y=(y if i == 0 else y2).clone(), res=(x2 if i == 0 else (z0 * 0.5).reshape(-1, 256)).clone(), inout_out=torch.zeros_like(x2))
                graph_check(f"graph replay transition [N={N} key={key}]", mkt, lambda b: TR._launch(b["y"], cache, b["inout_out"], res2d=b["res"], ln_mode=0))
        del z0; torch.cuda.empty_cache()

summary = dict(selftest="fpf_glue_v2", n_checks=len(RES), n_fail=len(FAIL), failed=FAIL, calls=G2.stats()["calls"], PASS=(len(FAIL) == 0 and len(RES) > 0))
print("SELFTEST SUMMARY", json.dumps(summary), flush=True)
if a.json:
    json.dump(dict(summary=summary, checks=RES), open(a.json, "w"), indent=1, default=str)
sys.exit(0 if summary["PASS"] else 1)
