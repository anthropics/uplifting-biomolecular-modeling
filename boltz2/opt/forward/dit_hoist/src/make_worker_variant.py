#!/usr/bin/env python3
"""make_worker_variant.py — derive the BZ2DIT worker VARIANT from the Boltz-2 trunk add-on's persistent worker (its `src/bz_worker_lev*.py`,
or its `src/bz_worker_stock.py`) by inserting three anchored blocks; nothing else in the base file is touched, so the variant is exactly
"their worker + the diffusion-sampler levers" (diff: patches/01_bz_worker_dit_vs_trunk_addon_worker.diff).

  block 1 (after the base worker's `ev("imports_done", ...)` line / its trunk-lever line): import + apply
      boltz_graph_patch (kit lever B1, CUDA-graph denoiser step; env BOLTZ_GRAPH_DIFFUSION=graph|predraw, inert when unset/0) and
      boltz_dit_hoist (this add-on; env BOLTZ_DIT_HOIST=1|2; inert when unset/0), in that order (hoist outermost);
      then a CUDA-synchronised wall timer around AtomDiffusion.sample (2 syncs per prediction; numerics untouched) -> per-item sampler_s.
  block 2 (inside the base worker's predict_step digest wrapper, after model_s is recorded): per-item report columns
      graph_* (sampler_mode, n_replay, n_capture) and hoist_* (level, net_calls, cache_builds/refreshes,
      captured_with_cache, capture_stock_fallbacks, cache_mb, n_caches) + sampler_s.
  block 3 (nothing to insert: the base worker already dumps LOG incl. per_item to <out_dir>/<tag>_worker_log.json).

usage: python make_worker_variant.py <base_worker.py> <out.py>        (prints the sha256 of base and variant; exit 2 if an anchor is missing)
"""
import sys, hashlib, re

BLOCK1 = '''# ---- BZ2DIT: diffusion-sampler levers (inert unless the env vars are set; stock bytes otherwise). Order: CUDA-graph patch first, hoist outermost. ----
_gd_mode = os.environ.get("BOLTZ_GRAPH_DIFFUSION", "0").lower()
_BGP = _DH = None
if _gd_mode not in ("0", "off", ""):
    sys.path.insert(0, os.environ.get("BOLTZ_OPT_WORKDIR") or os.getcwd()); import boltz_graph_patch as _BGP; LOG["env"]["graph_diffusion"] = _BGP.apply(_gd_mode); ev("graph_patch", mode=LOG["env"]["graph_diffusion"])
if os.environ.get("BOLTZ_DIT_HOIST", "0").lower() not in ("0", "off", "", "false"):
    sys.path.insert(0, os.environ.get("BOLTZ_OPT_WORKDIR") or os.getcwd()); import boltz_dit_hoist as _DH; LOG["env"]["dit_hoist"] = _DH.apply(); ev("dit_hoist", level=LOG["env"]["dit_hoist"])
from boltz.model.modules.diffusionv2 import AtomDiffusion as _AD
_SAMP = {"s": None}; _AD._bz2dit_timed_inner = _AD.sample
def _bz2dit_timed_sample(self, *a, **k):
    torch.cuda.synchronize(); _t = time.time(); r = _AD._bz2dit_timed_inner(self, *a, **k); torch.cuda.synchronize(); _SAMP["s"] = round(time.time() - _t, 4); return r
_bz2dit_timed_sample._dit_hoist = getattr(_AD.sample, "_dit_hoist", False); _bz2dit_timed_sample._bgp_patched = getattr(_AD.sample, "_bgp_patched", False)
_AD.sample = _bz2dit_timed_sample
'''
BLOCK2 = '''    d["sampler_s"] = _SAMP["s"]
    if _BGP is not None: d.update({"graph_" + k: v for k, v in (_BGP.STATS.get("last") or {}).items() if k in ("sampler_mode", "n_replay", "n_capture", "captures_total", "capture_s_last", "capture_pool_gb_last", "n_tokens", "release_min_tokens", "released", "headroom_gated", "projected_gib", "transient_gib", "need_gib", "free_gib")})
    else: d["graph_sampler_mode"] = "stock"
    if _DH is not None: d.update({"hoist_" + k: v for k, v in (_DH.STATS.get("last") or {}).items() if k in ("hoist_level", "net_calls", "cache_builds", "cache_refreshes", "captured_with_cache", "capture_stock_fallbacks", "cache_mb", "n_caches", "cache_released", "headroom_gated", "projected_cache_gib", "token_gated", "max_tokens")})
'''

def main(base, out):
    src = open(base).read().split("\n")
    # anchor 1: the trunk-lever apply line if present (variant keeps their levers), else the imports_done event line
    a1 = [i for i, l in enumerate(src) if "import boltz_trunk_levers as _LEV" in l]
    if a1:
        # insert after the atexit line that follows it (if any)
        i1 = a1[0] + (1 if a1[0] + 1 < len(src) and src[a1[0] + 1].startswith("import atexit as _ax") else 0)
    else:
        a1 = [i for i, l in enumerate(src) if l.startswith('ev("imports_done"')]
        if not a1: print("anchor 1 (the imports_done event line) not found in", base); sys.exit(2)
        i1 = a1[0]
    # anchor 2: the digest wrapper's model_s line
    a2 = [i for i, l in enumerate(src) if re.match(r'\s+d\["model_s"\] = round\(time\.time\(\) - t, 3\)', l)]
    if len(a2) != 1: print("anchor 2 (the model_s line inside the predict_step digest wrapper) not found exactly once in", base, a2); sys.exit(2)
    i2 = a2[0]
    assert i1 < i2
    out_lines = src[:i1 + 1] + BLOCK1.rstrip("\n").split("\n") + src[i1 + 1:i2 + 1] + BLOCK2.rstrip("\n").split("\n") + src[i2 + 1:]
    txt = "\n".join(out_lines)
    compile(txt, out, "exec")
    open(out, "w").write(txt)
    h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    print(f"worker variant written: {out} (sha256 {h(out)}) = {base} (sha256 {h(base)}) + BZ2DIT blocks after lines {i1 + 1} and {i2 + 1}")

if __name__ == "__main__":
    if len(sys.argv) != 3: print(__doc__); sys.exit(1)
    main(sys.argv[1], sys.argv[2])
