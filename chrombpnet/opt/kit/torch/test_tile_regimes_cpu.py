#!/usr/bin/env python
"""CPU test: the tile-by-batch regimes and the per-arch route — regime_tile() picks the first by_batch entry with B <= max_B, else the arch tile;
an arch without by_batch gets one tile at every B; K1ChromBPNet switches the tile per call, counts calls per tile, prints the policy once and keeps
the kernels' tile globals in step; a tile passed by apply() with its selection reason keeps the regimes on, an explicit tile turns them off;
default_route() answers per (arch, mode), takes the mode from TF_DETERMINISTIC_OPS when none is passed and refuses any other mode."""
import os, sys, json, io, contextlib, types, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chrombpnet_k1.forward as F
cases = 0
bb = [{"max_B": 2, "tile": "32x256x16x8x3"}, {"max_B": 32, "tile": "64x256x16x8x3"}]
assert [F.regime_tile(B, "64x512x16x16x3", bb) for B in (1, 2, 3, 4, 16, 32, 33, 64, 1024)] == ["32x256x16x8x3", "32x256x16x8x3", "64x256x16x8x3", "64x256x16x8x3", "64x256x16x8x3", "64x256x16x8x3", "64x512x16x16x3", "64x512x16x16x3", "64x512x16x16x3"]; cases += 1
assert F.regime_tile(1, "64x512x16x16x3", []) == "64x512x16x16x3"; cases += 1                                   # (2) no regimes -> the arch tile at every B
m = json.load(open(F.ARCH_TILES_PATH)); e = m["entries"]["cuda-90"]
assert [(r["max_B"], r["tile"]) for r in e["by_batch"]] == [(2, "32x256x16x8x3"), (32, "64x256x16x8x3")] and "chosen per call from the batch size" in e["by_batch_basis"] and "by_batch" not in m["entries"]["cuda-89"] and "by_batch" not in m["entries"]["cuda-80"]; cases += 1
# (4) the model's per-call switch: a K1ChromBPNet with stubbed sub-models (no GPU): counts per tile, prints once, kernels globals follow the tile
torch.cuda.get_device_capability = lambda i=0: (9, 0); torch.cuda.get_device_name = lambda i=0: "cpu-test"; torch.cuda.Stream = lambda *a, **k: None
class _Sub(torch.nn.Module):
    def forward(self, x): return torch.zeros((x.shape[0], 1, 1000)), torch.zeros((x.shape[0], 1))
orig_init = F.K1ChromBPNet.__init__
def fake_init(self, port, counts_mode="exact", tile=None, head="serial", tile_selected_by=None):
    torch.nn.Module.__init__(self)
    if tile is None: tile, self.tile_selected_by = F.select_tile(None)
    else: self.tile_selected_by = tile_selected_by or "explicit"
    self.acc = _Sub(); self.bias = _Sub(); self.tile = tile; self.counts_mode = counts_mode; self.head = head; self.streams = False; self._s2 = None
    self.by_batch = [] if self.tile_selected_by == "explicit" else F.batch_regime_tiles(); self.tile_calls = {}; self._regime_printed = False
    self.regime_policy = ("explicit tile (no regimes)" if self.tile_selected_by == "explicit" else (("tile by batch: " + ", ".join(f"B<={r['max_B']} {r['tile']}" for r in self.by_batch) + f", else {tile}") if self.by_batch else f"one tile at every B ({tile})"))
F.K1ChromBPNet.__init__ = fake_init
from chrombpnet_k1 import kernels as KN
model = F.K1ChromBPNet(None)
assert model.tile == "64x512x16x16x3" and model.by_batch == e["by_batch"], (model.tile, model.by_batch)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    for B in (1, 1, 4, 64, 1024, 2): model(torch.zeros((B, 4, 2114)))
assert model.tile_calls == {"32x256x16x8x3": 3, "64x256x16x8x3": 1, "64x512x16x16x3": 2}, model.tile_calls
assert buf.getvalue().count("[chrombpnet_k1] tile by batch") == 1 and (KN._BM, KN._BN, KN._NW) == (32, 256, 8), (buf.getvalue(), KN._BM, KN._BN); cases += 1
model.tile_calls = {}; model2 = F.K1ChromBPNet(None, tile="64x512x16x16x3"); model2(torch.zeros((1, 4, 2114))); assert model2.tile_calls == {"64x512x16x16x3": 1} and model2.by_batch == []; cases += 1   # (5) explicit tile: no regimes
# (6) the apply() path: the arch-selected tile is passed WITH its selection reason -> the regimes engage (a resolved tile passed as if explicit would turn them off)
t6, how6 = F.select_tile(None); m6 = F.K1ChromBPNet(None, tile=t6, tile_selected_by=how6); assert m6.by_batch == e["by_batch"] and "tile by batch" in m6.regime_policy, (how6, m6.regime_policy)
m6(torch.zeros((1, 4, 2114))); assert m6.tile_calls == {"32x256x16x8x3": 1}; cases += 1
# (7) the per-arch default route: cuda-80 -> k1 at shipped numerics, stock's own graph under the recipe; cuda-90 -> k1; cuda-89 -> the TensorFlow route at shipped numerics, k1 under the recipe; an unlisted arch -> 'unlisted' (the kit's class table decides; said)
assert F.default_route("cuda-80", "prod")[0] == "k1" and "tensor-core" in F.default_route("cuda-80", "prod")[1] and F.default_route("cuda-90", "prod")[0] == "k1" and F.default_route("cuda-89", "det")[0] == "k1" and F.default_route("cuda-75", "prod")[0] == "unlisted"; cases += 1   # cuda-80 -> k1 at shipped numerics (tensor cores; stock's own graph under the recipe); cuda-89 -> k1 under the recipe; an unlisted arch -> the class table decides
# the route under the deterministic recipe, per arch (asserted below)
assert F.default_route("cuda-90", "det")[0] == "k1" and F.default_route("cuda-89", "det")[0] == "k1" and F.default_route("cuda-80", "det")[0] == "stock" and F.default_route("cuda-100", "det")[0] == "unlisted" and F.default_route("cuda-89", "prod")[0] == "tf" and F.default_route("cuda-90", "prod")[0] == "k1" and "faster than K1 on this card" in F.default_route("cuda-89", "prod")[1], {a: F.default_route(a)[0] for a in ("cuda-90", "cuda-89", "cuda-80", "cuda-100")}
assert "bitwise equal to the L40S stock's own deterministic run" in F.default_route("cuda-89", "det")[1] and "every output file" in F.default_route("cuda-89", "det")[1]; cases += 1
# the mode from TF_DETERMINISTIC_OPS when the caller passes none; anything else refused
os.environ["TF_DETERMINISTIC_OPS"] = "1"; assert F.recipe_mode() == "det" and F.default_route("cuda-89")[0] == "k1"; os.environ["TF_DETERMINISTIC_OPS"] = "0"; assert F.recipe_mode() == "prod" and F.default_route("cuda-89")[0] == "tf"
try: F.default_route("cuda-89", "fast"); raise SystemExit("a bad mode was accepted")
except ValueError as e: assert "'prod' or 'det' only" in str(e)
cases += 1   # the cuda-89 basis text on the line
print(f"PASS test_tile_regimes_cpu: {cases} cases (regime_tile thresholds; the arch map's measured cuda-90 regimes only; per-call counts + printed once; explicit tile = no regimes)")
