#!/usr/bin/env python
"""CPU test: the per-arch counts form — (1) dense_groups_bfly reproduces, bitwise, the reference outputs of the shipped synthetic 64-row fixture
(the cuda-89 (L40S) dense order computed by the independent float32 reference in chrombpnet_k1/fixtures/README.md), and a different group size does not; (2) exact_counts dispatches on the form; (3) the form is selected by arch and an unknown form name
is refused; (4) K1ChromBPNet carries the form into both sub-models (also when constructed directly) and the stamp names it. No GPU."""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chrombpnet_k1 import kernels as K, forward as F
fx = np.load(os.path.join(os.path.dirname(K.__file__), "fixtures", "l40s_counts_head_64rows.npz")); cases = 0
for head in ("nobias", "bias"):
    g = torch.from_numpy(fx[f"{head}_gap_out"]); w = torch.from_numpy(fx[f"{head}_W"]); b = torch.tensor(float(fx[f"{head}_b"][0]))
    lc = K.dense_groups_bfly(g, w, b, group=128).numpy().reshape(-1); tgt = fx[f"{head}_logcounts"].reshape(-1)
    n = int((lc.view(np.uint32) == tgt.view(np.uint32)).sum()); assert n == len(tgt) == 64, (head, n)
    # a different group size is NOT the L40S order (the fixture discriminates)
    lc2 = K.dense_groups_bfly(g, w, b, group=g.shape[1]).numpy().reshape(-1) if g.shape[1] > 128 else None
    if lc2 is not None: assert int((lc2.view(np.uint32) == tgt.view(np.uint32)).sum()) < 64
    cases += 1
f89, line89 = F.select_counts_form("cuda-89"); assert f89 == {"form": "groups_prod_bfly_seq", "group": 128} and "cuda-89 stock's OWN dense order" in line89 and "stride-halving butterfly" in line89, (f89, line89); cases += 1
for a in ("cuda-90", "cuda-80", "cuda-100"):
    f, line = F.select_counts_form(a); assert f is None and "H100 order" in line, (a, f, line)
cases += 1
class _Lin(torch.nn.Module):
    def __init__(self, C): super().__init__(); self.weight = torch.nn.Parameter(torch.from_numpy(fx["nobias_W"]).reshape(1, -1)); self.bias = torch.nn.Parameter(torch.from_numpy(fx["nobias_b"]))
lin = _Lin(512); h = torch.from_numpy(fx["nobias_gap_out"])[:, :, None].repeat(1, 1, 3)   # (N, C, L=3) with every position = the pooled value -> gap_serial mean = the value
try:
    K.exact_counts(h, lin, {"form": "no_such_form"}); raise SystemExit("an unknown form was accepted")
except ValueError as e: assert "unknown counts form" in str(e); cases += 1
try:
    r = K.exact_counts(h, lin, {"form": "groups_prod_bfly_seq", "group": 128})                                        # dispatch to the L40S form (gap_serial may need Triton -> skip on CPU)
    assert (r.numpy().reshape(-1).view(np.uint32) == fx["nobias_logcounts"].reshape(-1).view(np.uint32)).sum() == 64; cases += 1
except Exception as e:
    if any(k in repr(e).lower() for k in ("triton", "cuda", "active drivers")): print("   (exact_counts dispatch needs the Triton pool: no GPU driver here, the dispatch case skipped)")
    else: raise
class _Port:                                                                                                          # a stand-in port for K1ChromBPNet
    accessibility = None; bias = None
_orig = K.K1BPNetV3.__init__
def _fake_init(self, bpnet, head_pad=16, counts_mode="exact", head="serial", counts_form=None): torch.nn.Module.__init__(self); self.counts_mode = counts_mode; self.head = head; self.counts_form = counts_form
K.K1BPNetV3.__init__ = _fake_init; F.K1BPNetV3 = K.K1BPNetV3
os.environ["CHROMBPNET_K1_STREAMS"] = "0"; torch.cuda.get_device_capability = lambda i=0: (8, 9)                                   # no CUDA here: the serial stream form for the construction case
m = F.K1ChromBPNet(_Port(), tile="64x512x16x16x3", counts_form=f89); assert m.acc.counts_form == f89 and m.bias.counts_form == f89 and m.counts_form == f89; cases += 1
m2 = F.K1ChromBPNet(_Port(), tile="64x512x16x16x3"); assert m2.counts_form == f89 and m2.acc.counts_form == f89 and "cuda-89 stock's OWN dense order" in m2.counts_line, (m2.counts_form, m2.counts_line); cases += 1   # the direct-constructor path resolves the arch's form itself
torch.cuda.get_device_capability = lambda i=0: (9, 0); m3 = F.K1ChromBPNet(_Port(), tile="64x512x16x16x3"); assert m3.counts_form is None and "H100 order" in m3.counts_line; cases += 1                    # cuda-90: the H100 order
dr = json.load(open(F.ARCH_TILES_PATH))["entries"]["cuda-89"]["default_route"]; assert dr["det"]["route"] == "k1" and "bitwise equal to the L40S stock's own deterministic run" in dr["det"]["basis"] and dr["prod"]["route"] == "tf" and "byte-identical to the L40S stock's own deterministic run" in json.load(open(F.ARCH_TILES_PATH))["entries"]["cuda-89"]["counts_form"]["certificate"]; cases += 1   # the per-mode entry — det k1 (with its basis text), prod tf
print(f"PASS test_counts_form_cpu: {cases} cases (the L40S dense order bitwise on the 64-row fixture for both heads; the arch selection; the dispatch + refusal; the form carried into both sub-models)")
