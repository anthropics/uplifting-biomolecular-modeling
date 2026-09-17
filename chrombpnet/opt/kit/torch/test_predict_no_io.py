#!/usr/bin/env python
"""CPU test: predict()'s per-call host work does NO file I/O and NO whole-array hashing — os.walk / open /
os.stat / os.listdir / os.scandir raise inside predict(); the stamp's md5 fields are None unless stamp_md5=True. Runs without a GPU (cuda stubbed)."""
import os, sys, builtins, hashlib, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chrombpnet_k1.forward as F

class FakeModel:
    tile = "test"; head = "serial"; counts_mode = "exact"
    def __call__(self, x): return torch.zeros((x.shape[0], F.OUTPUT_LEN), dtype=torch.float32), torch.zeros((x.shape[0], 1), dtype=torch.float32)

torch.Tensor.cuda = lambda self, *a, **k: self; torch.cuda.synchronize = lambda *a, **k: None; torch.cuda.get_device_name = lambda i=0: "cpu-test"
torch.backends.cudnn.version = lambda: 0
X = np.zeros((130, 2114, 4), np.int8)
def boom(*a, **k): raise AssertionError("file I/O inside predict()")
saved = (os.walk, builtins.open, os.stat, os.listdir, os.scandir, hashlib.md5)
os.walk = boom; builtins.open = boom; os.stat = boom; os.listdir = boom; os.scandir = boom
hashlib.md5 = lambda *a, **k: (_ for _ in ()).throw(AssertionError("md5 inside predict() by default"))
try:
    lg, lc, st = F.predict(FakeModel(), X, batch_size=64)
finally:
    os.walk, builtins.open, os.stat, os.listdir, os.scandir, hashlib.md5 = saved
assert lg.shape == (130, F.OUTPUT_LEN) and lc.shape == (130, 1) and st["forward_calls"] == 3 and st["padded_rows"] == 62, st
assert st["md5_logits"] is None and st["md5_logcts"] is None and "not read per call" in str(st["triton_cache"]), st
lg2, lc2, st2 = F.predict(FakeModel(), X, batch_size=64, stamp_md5=True); assert st2["md5_logits"] == hashlib.md5(np.ascontiguousarray(lg2).tobytes()).hexdigest()
print("PASS test_predict_no_io: 1 case (predict() does no file I/O / hashing per call; md5 only with stamp_md5=True)")
