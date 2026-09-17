#!/usr/bin/env python
"""CPU test: predict() splits any batch above K1_MAX_BATCH (1024) with the cap printed once and stamped, a batch at/below the cap is untouched,
a direct forward past the kernels' int32 index bound raises ValueError and the process stays usable afterwards, and the first predict() of the
process records its wall time. No GPU (cuda stubbed)."""
import os, sys, io, contextlib, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chrombpnet_k1.forward as F
class FakeModel:
    tile = "test"; head = "serial"; counts_mode = "exact"; seen = []
    def __call__(self, x):
        FakeModel.seen.append(int(x.shape[0])); return torch.zeros((x.shape[0], F.OUTPUT_LEN), dtype=torch.float32), torch.zeros((x.shape[0], 1), dtype=torch.float32)
torch.Tensor.cuda = lambda self, *a, **k: self; torch.cuda.synchronize = lambda *a, **k: None; torch.cuda.get_device_name = lambda i=0: "cpu-test"; torch.backends.cudnn.version = lambda: 0
cases = 0
X = np.zeros((2100, F.INPUT_LEN, 4), np.int8)
buf = io.StringIO()
with contextlib.redirect_stdout(buf): lg, lc, st = F.predict(FakeModel(), X, batch_size=4096)
assert st["forward_calls"] == 3 and st["batch_size"] == 1024 and st["batch_size_requested"] == 4096 and st["batch_cap"] and max(FakeModel.seen) == 1024, st
assert "int32 index bound" in buf.getvalue() and lg.shape == (2100, F.OUTPUT_LEN), buf.getvalue()[:200]; cases += 1                       # (1) the split + the cap printed
FakeModel.seen.clear(); buf2 = io.StringIO()
with contextlib.redirect_stdout(buf2): lg, lc, st = F.predict(FakeModel(), X, batch_size=4096)
assert buf2.getvalue() == "" and st["batch_cap"], "the cap is printed ONCE per process"; cases += 1                                        # (2) printed once
FakeModel.seen.clear(); lg, lc, st = F.predict(FakeModel(), X[:1100], batch_size=1024)
assert st["batch_cap"] is None and st["batch_size"] == 1024 and st["forward_calls"] == 2 and max(FakeModel.seen) == 1024, st; cases += 1   # (3) at the cap: untouched
lg, lc, st = F.predict(FakeModel(), X[:130], batch_size=64); assert st["batch_cap"] is None and st["forward_calls"] == 3, st; cases += 1     # (4) below: untouched
try: F.check_batch_bound(torch.zeros((4096, 4, F.INPUT_LEN))); raise SystemExit("no raise at B=4096")
except ValueError as e: assert "int32 index bound" in str(e); cases += 1                                                                  # (5) the forward guard raises loudly
F.check_batch_bound(torch.zeros((1024, 4, F.INPUT_LEN))); F.check_batch_bound(torch.zeros((1984, 4, F.INPUT_LEN))); cases += 1             # (6) at/below the bound: passes (1984 = 2^31 // (512*2114))
try: F.check_batch_bound(torch.zeros((1985, 4, F.INPUT_LEN))); raise SystemExit("no raise at 1985")
except ValueError: cases += 1                                                                                                              # (7) just past the bound: raises
# (8) the first predict() of the process records its wall time (_FIRST_FORWARD, reported by cache_witness())
assert F._FIRST_FORWARD.get("wall_s") is not None and F._FIRST_FORWARD.get("n") is not None and F._FIRST_FORWARD["wall_s"] >= 0.0, F._FIRST_FORWARD; cases += 1
# (9) after a refused direct forward (past the bound) the process is ALIVE and a valid predict() still works; the stamp's batch_bound text exists
try: F.check_batch_bound(torch.zeros((2048, 4, F.INPUT_LEN))); raise AssertionError("2048 must refuse")
except ValueError as e: assert "int32 index bound" in str(e)
lg, lc, st = F.predict(FakeModel(), X[:130], batch_size=64); assert st["forward_calls"] == 3 and lg.shape == (130, F.OUTPUT_LEN); assert "batch bound" in open(F.__file__).read(); cases += 1
print(f"PASS test_batch_bound_cpu: {cases} cases (predict() splits > 1024 with the cap printed once + stamped; the forward guard raises past n*C*L >= 2^31)")
