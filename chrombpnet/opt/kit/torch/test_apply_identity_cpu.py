#!/usr/bin/env python
"""CPU test: apply() keys the shipped cache on target (arch) + backend_hash + the IR/PTX identity key — a record built under a different device
name of the same arch is served; a differing key, target or stage-1 pin takes the JIT path with the reason printed and no exception; an unlisted
arch gets a tile by shared memory; require_device only prints. Runs without a GPU: torch.cuda and the Triton identity are stubbed, the model
construction is stubbed."""
import os, sys, json, tempfile, types, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.cuda.get_device_name = lambda i=0: "NVIDIA H200"; torch.cuda.get_device_capability = lambda i=0: (9, 0); torch.cuda.synchronize = lambda *a, **k: None
torch.backends.cudnn.version = lambda: 90100
class _P: shared_memory_per_block_optin = 232448
torch.cuda.get_device_properties = lambda i=0: _P()
import chrombpnet_k1.forward as F
F._triton_identity = lambda: {"triton_key": "tk", "backend_hash": "bh", "target": "cuda-90-32"}
F._triton_version = lambda: "3.0.0"; F._kernels_sha = lambda: "ks"
import triton.compiler.compiler as _tc; _tc.triton_key = lambda: "tk"
class _Model:
    def __init__(self, port, counts_mode="exact", tile=None, head="serial", tile_selected_by=None, counts_form=None, precision="ieee"): self.tile = tile; self.tile_selected_by = "test"; self.precision = precision
F.K1ChromBPNet = _Model
class _Port:
    @staticmethod
    def from_chrombpnet(a, b): return _Port()
    def cuda(self): return self
    def eval(self): return self
try:
    import bpnetlite.chrombpnet as _bp; _bp.ChromBPNet = _Port
except ModuleNotFoundError:                     # no bpnetlite here: a stub package (apply() imports bpnetlite.chrombpnet.ChromBPNet lazily)
    _pkg = types.ModuleType("bpnetlite"); _mod = types.ModuleType("bpnetlite.chrombpnet"); _mod.ChromBPNet = _Port; _pkg.chrombpnet = _mod; sys.modules["bpnetlite"] = _pkg; sys.modules["bpnetlite.chrombpnet"] = _mod
def mk_cache(dev="NVIDIA H100 80GB HBM3", target="cuda-90-32", ptx="ptx same"):
    d = tempfile.mkdtemp(); e = "k" * 64; os.makedirs(os.path.join(d, e))
    open(os.path.join(d, e, "k.ttir"), "w").write("module"); open(os.path.join(d, e, "k.ptx"), "w").write(ptx); open(os.path.join(d, e, "k.json"), "w").write('{"shared": 1}'); open(os.path.join(d, e, "k.cubin"), "wb").write(b"SASS")
    json.dump({"child_paths": {"k.cubin": os.path.join(d, e, "k.cubin")}}, open(os.path.join(d, e, "__grp__k.json"), "w"))
    rec = {"triton": "3.0.0", "torch": torch.__version__, "kernels_sha256_16": "ks", "triton_key": "tk", "backend_hash": "bh", "target": target, "device": dev, "cudnn": 8900, "entries": {e: {"kit": "k1"}}}
    rec.update(F.cache_identity_key(d, rec)); json.dump(rec, open(os.path.join(d, "JIT_IDENTITY.json"), "w"))
    with open(os.path.join(d, "SHA256SUMS"), "w") as sf:   # the files an install copies, each a line of the cache's SHA256SUMS (apply() holds them to it before installing)
        for f in sorted(os.listdir(os.path.join(d, e))): sf.write("%s  %s/%s\n" % (hashlib.sha256(open(os.path.join(d, e, f), "rb").read()).hexdigest(), e, f))
    return d, rec
os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp()
# case A: the record's device name differs (H100 record on an H200), the arch + key equal -> SERVED, the datum on the line
F._CACHE_SKIP.clear(); d, rec = mk_cache(); model, st = F.apply("b", "nb", cache_dir=d, tile="64x512x16x16x3")
assert st["not_applicable"] is None and "key equal, served" in st["r8_line"] and "record built on NVIDIA H100 80GB HBM3" in st["r8_line"], st["r8_line"]
print("A served with the device datum:", st["r8_line"][-160:])
# case B: a differing IR/PTX key -> the JIT path with the reason, no exception
F._CACHE_SKIP.clear(); d2, rec2 = mk_cache(ptx="ptx OTHER"); json.dump(dict(rec2, identity_key="0000000000000000"), open(os.path.join(d2, "JIT_IDENTITY.json"), "w"))
model, st = F.apply("b", "nb", cache_dir=d2, tile="64x512x16x16x3")
assert st["not_applicable"] and "NOT APPLICABLE" in st["not_applicable"] and "identity_key" in st["not_applicable"], st
print("B key mismatch -> JIT path said:", st["not_applicable"][:140])
# case C: a differing target (an sm80 record on this sm90 device) -> the JIT path said
F._CACHE_SKIP.clear(); d3, rec3 = mk_cache(target="cuda-80-32"); model, st = F.apply("b", "nb", cache_dir=d3, tile="64x512x16x16x3")
assert st["not_applicable"] and "target" in st["not_applicable"], st
print("C target mismatch -> JIT path said:", st["not_applicable"][:120])
# case D: a stage-1 pin mismatch (kernels bytes) -> the JIT path said
F._CACHE_SKIP.clear(); d4, rec4 = mk_cache(); json.dump(dict(rec4, kernels_sha256_16="zz"), open(os.path.join(d4, "JIT_IDENTITY.json"), "w")); model, st = F.apply("b", "nb", cache_dir=d4, tile="64x512x16x16x3")
assert st["not_applicable"] and "stage-1 pins" in st["not_applicable"], st
print("D pin mismatch -> JIT path said:", st["not_applicable"][:120])
# case E: an unlisted arch -> a tile by smem, said (never a refusal)
torch.cuda.get_device_capability = lambda i=0: (10, 0); tile, how = F.select_tile(None); assert tile == "64x512x16x16x3" and "unlisted class" in how, (tile, how)
_P.shared_memory_per_block_optin = 101376; tile, how = F.select_tile(None); assert tile == "64x256x16x8x3", (tile, how); torch.cuda.get_device_capability = lambda i=0: (9, 0); _P.shared_memory_per_block_optin = 232448
print("E unlisted arch -> tile by smem, said")
# case F: require_device mismatch -> a datum, no exception
F._CACHE_SKIP.clear(); model, st = F.apply("b", "nb", cache_dir=None, require_device="NVIDIA H100", tile="64x512x16x16x3"); assert st["device_expectation"] and "expected NVIDIA H100" in st["r8_line"], st
print("F require_device -> datum:", st["device_expectation"])
print("PASS test_apply_identity_cpu: 6 cases")
