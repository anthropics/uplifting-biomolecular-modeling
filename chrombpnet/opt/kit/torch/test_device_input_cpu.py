#!/usr/bin/env python
"""CPU test: predict() with a DEVICE-RESIDENT tensor input (read in place: no H2D copy, no host cast) gives the same bytes as the host-array path in
both the (N, L, 4) and the (N, 4, L) layout, pads the remainder the same way, refuses any other shape, prints the contract line once, and treats a
tensor that is not on the model's device as host input. The fake model lives on the CPU, so a CPU tensor IS resident on the model's device; the
host path's .cuda() is stubbed."""
import os, sys, io, contextlib, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.Tensor.cuda = lambda self, *a, **k: self; torch.cuda.synchronize = lambda *a, **k: None; torch.cuda.get_device_name = lambda i=0: "cpu-test"
from chrombpnet_k1 import forward as F
class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.w = torch.nn.Parameter(torch.linspace(0.5, 1.5, F.OUTPUT_LEN)); self.tile = "fake"; self.head = "serial"; self.counts_mode = "exact"
    def forward(self, x):                        # x: (B, 4, L) float32
        assert x.dtype == torch.float32 and x.shape[1] == 4 and x.is_contiguous(), (x.dtype, x.shape, x.is_contiguous())
        p = x[:, 0, :F.OUTPUT_LEN] * self.w + x[:, 1, :F.OUTPUT_LEN] * 0.25 - x[:, 2, :F.OUTPUT_LEN]; c = x.sum(dim=(1, 2)).reshape(-1, 1) * 0.001
        return p, c
rng = np.random.default_rng(0); N = 130
X = np.zeros((N, F.INPUT_LEN, 4), dtype=np.int8); X[np.arange(N)[:, None], np.arange(F.INPUT_LEN)[None, :], rng.integers(0, 4, (N, F.INPUT_LEN))] = 1
m = FakeModel(); cases = 0
buf = io.StringIO()
with contextlib.redirect_stdout(buf): lg_h, lc_h, st_h = F.predict(m, X, batch_size=64)                              # the host path (numpy)
with contextlib.redirect_stdout(buf): lg_d, lc_d, st_d = F.predict(m, torch.from_numpy(X), batch_size=64)            # resident (N, L, 4) int8, in place
assert st_h["input"]["kind"] == "host array" and st_d["input"]["kind"].startswith("device-resident tensor") and st_d["input"]["layout"] == "NL4", (st_h["input"], st_d["input"])
assert lg_d.shape == (N, F.OUTPUT_LEN) and (lg_d.view(np.uint32) == lg_h.view(np.uint32)).all() and (lc_d.view(np.uint32) == lc_h.view(np.uint32)).all(); cases += 1   # (1) bitwise
Xm = torch.from_numpy(X).permute(0, 2, 1).float().contiguous()                                                          # (N, 4, L) float32 = the model layout
with contextlib.redirect_stdout(buf): lg_m, lc_m, st_m = F.predict(m, Xm, batch_size=64)
assert st_m["input"]["layout"] == "N4L" and (lg_m.view(np.uint32) == lg_h.view(np.uint32)).all() and (lc_m.view(np.uint32) == lc_h.view(np.uint32)).all(); cases += 1   # (2) the model layout, bitwise
assert st_d["padded_rows"] == st_h["padded_rows"] == 64 * 3 - N and st_d["forward_calls"] == 3; cases += 1              # (3) the remainder padded on the device the same way
try: F.predict(m, torch.zeros((N, 7, 9), dtype=torch.int8)); raise SystemExit("a bad shape was accepted")
except ValueError as e: assert "device-resident input must be" in str(e); cases += 1                                    # (4) refused loudly
out = buf.getvalue(); assert out.count("read IN PLACE (no H2D copy, no host cast; CONTRACT v2)") == 1 and "layout NL4" in out, out; cases += 1   # (5) the contract line once
with contextlib.redirect_stdout(buf): lg_t, lc_t, st_t = F.predict(FakeModelNoParams := type("NP", (), {"tile": "f", "head": "s", "counts_mode": "e", "__call__": lambda self, x: m(x)})(), torch.from_numpy(X), batch_size=64)
assert st_t["input"]["kind"] == "host tensor" and (lg_t.view(np.uint32) == lg_h.view(np.uint32)).all(); cases += 1      # (6) a tensor not resident on the model's device = the host path, same bytes
print(f"PASS test_device_input_cpu: {cases} cases (a device-resident tensor is read in place == the host array bitwise, both layouts; the padded remainder; a bad shape refused; the contract line once; a non-resident tensor takes the host path)")
