"""Lever `atom_window` (fast, big): the atom transformers' sequence-local attention on the support library's window kernels
(opt_core.kernels.atom_window) — wired through the kit's tables, routed by name, counted in the census, stepping aside by name; and (GPU,
AF3T_GPU_TESTS=1) the windowed DiffusionCrossAttTransformer against xfold's own statements: inside the bf16 route's distance from the fp32
module, sample s of a batched call bitwise the single-sample call, padding blocks left out."""
import os

import pytest

from af3_torch_opt import registry, stack

HERE = os.path.dirname(os.path.abspath(__file__))
NN = os.path.join(stack.forward_dir(), "af3t", "af3_torch", "xfold", "nn")


def _src(*rel):
    return open(os.path.join(*rel), encoding="utf-8").read()


def test_lever_is_wired_through_the_kits_tables():
    src = _src(stack.forward_dir(), "af3t", "af3_torch", "af3_torch_api.py")
    fastest = src[src.index('"fastest": ('):src.index(")", src.index('"fastest": ('))]
    assert '"atom_window"' in fastest and fastest.index('"atom_window"') > fastest.index('"stepgraph"')     # built after the hoist / graph levers it rides
    assert 'model._atom_window_state = enable_atom_window(model) if "atom_window" in levers else None' in src
    assert 'return "needs_hoist"' in src and "_atom_window_warmup(AW, C, H, has_bias, ATOM_WINDOW_PRECISION, ln_kw)" in src
    assert 'ATOM_WINDOW_CELLS = {(9, 0): {}, (8, 0): {"BLOCK_R": 32}}' in src            # the ln_qkvg row tile per compute capability (cc 8.0: 163 KB shared memory)
    L = registry.LEVERS["atom_window"]
    assert L["kind"] == "kernel" and L["family"] == "F5" and "TF32" in L["numerics"] and "not bitwise" in L["numerics"]
    assert "atom_window" in registry.NOT_BITWISE and "atom_window" not in registry.EXACT
    assert registry.EVIDENCE["atom_window"] == "census" and registry.IMPL["atom_window"] == ("atom_window", "core")
    assert registry.STRATEGY["atom_window"] == "LOCAL.af3_torch.atom_window"
    assert registry.KERNEL_ROUTES["atom_window"] == {"levers": ("atom_window",), "exports": {}, "kit_copy": None}
    assert "atom_window" in stack.kernel_routes()


def test_the_step_asides_are_named():
    fwd = _src(HERE, "..", "forward.py")
    assert '("atom_window", "skipped:rowpair_hoist")' in fwd                                                   # big --n_gpu P: the row schedule replaces the hoist
    assert 'census["atom_window"] = {"served:calls": int(c["calls"])' in fwd and 'out["atom_window"] = aw' in fwd   # census served/fallback:<word>; dead word at build
    rpx = _src(HERE, "..", "rowpair_xfold.py")
    assert '_lever("atom_window", "skipped:rowpair_hoist")' in rpx and "model.diffusion_head.use_atom_window = False" in rpx
    dt = _src(NN, "diffusion_transformer.py")
    for word in ('"kernel_absent"', '"device_cpu"', '"window_geom_%dx%d"', '"head_geom"', '"rank_%d"'):
        assert word in dt, word                                                                                 # DiffusionCrossAttTransformer.window_geometry_ok refusal words
    assert "def forward_windowed(" in dt and "def window_operands(" in dt and 'window["rows_host"]' in dt
    dh = _src(NN, "diffusion_head.py")
    assert 'new["aw.rows_host"] = int(new["awE.rows"].item())' in dh and 'enc["dec_window"]' in dh            # hoisted once per trajectory, one read-back outside any capture
    aca = _src(NN, "atom_cross_attention.py")
    assert 'window=static.get("enc_window")' in aca and "window=window," in aca


_GPU_SCRIPT = r"""
import os, sys, math, torch
sys.path[:0] = KITPATH; sys.path.insert(0, CORE)
from opt_core.kernels import route
route("atom_window")
import atom_window as AW
from xfold.nn import diffusion_transformer as DT, atom_layout
torch.manual_seed(0)
dev = torch.device("cuda", 0)
ns, nq, nk, C, H, S, cp = 10, 32, 128, 128, 4, 3, 16
A = ns * nq; n_real = 300
m = DT.DiffusionCrossAttTransformer(c_query=C, c_single_cond=C, c_pair_cond=cp, num_blocks=3, num_head=H).to(dev)
with torch.no_grad():
    for p_ in m.parameters():
        p_.copy_(torch.randn_like(p_) * (0.5 / math.sqrt(p_.shape[-1]) if p_.dim() == 2 else 0.2))
m.eval()
qmask = (torch.arange(A, device=dev) < n_real).reshape(ns, nq)
starts = [min(max(i * nq + nq // 2 - nk // 2, 0), n_real - nk) for i in range(ns)]
gidx = torch.tensor([[st + j for j in range(nk)] for st in starts], device=dev)
q2k = atom_layout.GatherInfo(gather_idxs=gidx, gather_mask=torch.ones_like(gidx, dtype=torch.bool), input_shape=torch.tensor([ns, nq]))
kmask = atom_layout.convert(q2k, qmask, layout_axes=(-2, -1))
x = torch.randn(S, ns, nq, C, device=dev); cond = torch.randn(ns, nq, C, device=dev); pc = torch.randn(ns, nq, nk, cp, device=dev) * 0.5
kcond = atom_layout.convert(q2k, cond, layout_axes=(-3, -2))
def run(mod, xin, window=None):
    outs = []
    with torch.no_grad():
        pl = mod.compute_pair_logits(pc.to(next(mod.parameters()).dtype))
        for s_ in range(xin.shape[0]) if window is None else [None]:
            xi = xin[s_].clone() if s_ is not None else xin.clone()
            outs.append(mod(xi, qmask, q2k, kmask, cond.to(xi.dtype) if window is None else cond, kcond.to(xi.dtype) if window is None else kcond, pc, pair_logits=pl, window=window))
    return torch.stack(outs) if window is None else outs[0]
ref = run(m, x).float()                                                     # xfold's statements, fp32 weights, no autocast: the reference
mb = m.to(torch.bfloat16)                                                   # the fast route: bf16 weights under bf16 autocast
with torch.autocast("cuda", dtype=torch.bfloat16):
    bf = run(mb, x.to(torch.bfloat16)).float()
    import af3_torch_api as API
    cc = tuple(torch.cuda.get_device_capability()); kw = dict(API.ATOM_WINDOW_CELLS.get(cc, API.ATOM_WINDOW_CELLS[(8, 0)] if cc[0] == 8 else {}))
    ca = mb.cross_attention[0]; hb = tuple(l.bias is not None for l in (ca.q_projection, ca.k_projection, ca.v_projection, ca.gating_query, ca.adaptive_zero_init.transition2))
    DT.DiffusionCrossAttTransformer.WINDOW_KERNEL = AW; DT.DiffusionCrossAttTransformer.WINDOW_LN_KW = kw
    with torch.no_grad():
        API._atom_window_warmup(AW, C, H, hb, "tf32rn", kw)
    with torch.no_grad():
        w = mb.window_operands(cond.to(torch.bfloat16), qmask)
    w["rows_host"] = int(w["rows"].item())
    assert mb.window_geometry_ok(x, q2k) is None
    win = run(mb, x.to(torch.bfloat16), window=w).float()                   # all S samples in one call
    win1 = run(mb, x[:1].to(torch.bfloat16), window=w).float()              # sample 0 alone
    # kernel level: sample 0 of a 3-sample launch vs the 1-sample launch, same programs -> same bits (the block's transition GEMMs are torch's: class, not bits)
    a3 = x.reshape(S, A, C).float().contiguous(); ca0 = mb.cross_attention[0]
    with torch.no_grad():
        q3 = AW.ln_qkvg(a3, w["gq0"], w["lsq0"], w["gk0"], w["lsk0"], ca0.q_projection, ca0.k_projection, ca0.v_projection, ca0.gating_query, 1e-5, float(ca0.q_scale), precision="tf32rn", **kw)
        q1 = AW.ln_qkvg(a3[:1], w["gq0"], w["lsq0"], w["gk0"], w["lsk0"], ca0.q_projection, ca0.k_projection, ca0.v_projection, ca0.gating_query, 1e-5, float(ca0.q_scale), precision="tf32rn", **kw)
        pl0 = mb.compute_pair_logits(pc.to(torch.bfloat16))[0].float().contiguous()
        y3 = AW.window_attn(q3, a3, pl0, w["ks"], w["n_real"], w["amask"], w["zg0"], ca0.adaptive_zero_init.transition2.weight, None, H, nq, nk, 1e9, precision="tf32rn")
        y1 = AW.window_attn(q1, a3[:1], pl0, w["ks"], w["n_real"], w["amask"], w["zg0"], ca0.adaptive_zero_init.transition2.weight, None, H, nq, nk, 1e9, precision="tf32rn")
    kern_same = int(torch.equal(q3[:1], q1) and torch.equal(y3[:1], y1))
real = qmask.reshape(-1)
d = lambda a_, b_: float((a_.reshape(a_.shape[0], -1, C)[:, real] - b_.reshape(b_.shape[0], -1, C)[:, real]).abs().max())
cnt = DT.DiffusionCrossAttTransformer.WINDOW_COUNTS
print("RESULT", d(bf, ref), d(win, ref), kern_same, w["rows_host"], cnt["calls"], cnt["blocks_real"], cnt["blocks_total"], float(ref.abs().max()), d(win[:1], win1))
with torch.autocast("cuda", dtype=torch.bfloat16):                          # lever 'atom_rows': the rows-only restatement vs forward_windowed and vs the fp32 module
    DT.DiffusionCrossAttTransformer.WINDOW_ROWS_ONLY = True
    win_rows = run(mb, x.to(torch.bfloat16), window=w).float()
    DT.DiffusionCrossAttTransformer.WINDOW_ROWS_ONLY = False
rc = DT.DiffusionCrossAttTransformer.ROWS_COUNTS
print("RESULT2", d(win_rows, ref), d(win_rows, win), rc["calls"], sum(rc["refused"].values()))
"""


@pytest.mark.skipif(os.environ.get("AF3T_GPU_TESTS") != "1", reason="GPU numerics test: AF3T_GPU_TESTS=1 on a CUDA box with the kit's torch venv")
def test_windowed_atom_transformer_is_inside_the_bf16_routes_class_on_gpu():
    """The windowed DiffusionCrossAttTransformer (bf16 weights, autocast, window kernels TF32) vs xfold's statements in fp32: its max |err| on the
    real atoms is at most the bf16 route's own; at the kernel level sample 0 of a 3-sample launch is bitwise the 1-sample launch (the whole block
    across batch sizes is the class: the transition's GEMMs are torch's); the ceil(300 / 32) = 10 real blocks of 10 are handed to the kernels, counted."""
    import subprocess
    import sys
    script = _GPU_SCRIPT.replace("KITPATH", repr(stack.kit_sys_path())).replace("CORE", repr(stack.core_dir()))
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=900)
    line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
    line2 = [l for l in r.stdout.splitlines() if l.startswith("RESULT2 ")]
    assert r.returncode == 0 and line and line2, (r.returncode, r.stdout[-2000:], r.stderr[-4000:])
    _, e_bf, e_win, kern_same, rows, calls, breal, btot, scale, d_s01 = line[-1].split()
    _, e_rows, d_rows_win, rows_calls, rows_refused = line2[-1].split()
    assert float(e_rows) <= 1.1 * float(e_bf) and float(d_rows_win) <= float(e_bf), (line, line2)   # lever atom_rows: the rows-only path is inside the same class, next to forward_windowed
    assert int(rows_calls) == 1 and int(rows_refused) == 0, line2
    assert float(e_win) <= 1.1 * float(e_bf), line                     # op level: the window route is at or inside the bf16 route's distance from the fp32 module
    assert kern_same == "1", line                                       # the kernels: sample 0 of a 3-sample launch is bitwise the 1-sample launch (same programs per sample)
    assert float(d_s01) <= float(e_bf), line                            # the whole block across batch sizes: the class (torch's transition GEMMs pick algorithms by row count)
    assert int(rows) == 320 and int(calls) == 2 and int(breal) == 20 and int(btot) == 20, line


def test_atom_rows_and_prologue_are_wired_through_the_kits_tables():
    """Levers atom_rows / prologue: registered where the kit reads levers, in the fast set, switchable, stepping aside by name."""
    for name, impl in (("atom_rows", "diffusion_transformer.forward_windowed_rows"), ("prologue", "diffusion_head.augment_and_noise_batched")):
        assert name in registry.LEVERS and name in registry.NOT_BITWISE and name not in registry.EXACT, name
        assert registry.IMPL[name] == (impl, "kit") and registry.STRATEGY[name].startswith("LOCAL."), name
    api = _src(stack.forward_dir(), "af3t", "af3_torch", "af3_torch_api.py")
    fastest = api[api.index('"fastest": ('):api.index("\n", api.index('"fastest": ('))]
    assert '"atom_rows"' in fastest and '"prologue"' in fastest
    assert "def enable_atom_rows(model, levers)" in api and 'return "needs_atom_window"' in api and '"needs_sbatch"' in api
    dtp = _src(stack.forward_dir(), "af3t", "af3_torch", "xfold", "nn", "diffusion_transformer.py")
    assert "WINDOW_ROWS_ONLY = False" in dtp and "def forward_windowed_rows(self" in dtp and 'ROWS_COUNTS["refused"]' in dtp
    dhp = _src(stack.forward_dir(), "af3t", "af3_torch", "xfold", "nn", "diffusion_head.py")
    assert "def augment_and_noise_batched(" in dhp and "plan.seek(s0 + j, step_idx)" in dhp
    fwd = _src(HERE, "..", "forward.py")
    assert '("atom_rows", "skipped:rowpair_hoist")' in fwd and '("prologue", "skipped:rowpair_sampler")' in fwd and 'census["atom_rows"]' in fwd and 'out["prologue"] = pl' in fwd
