
# bg_inproc.py — `inproc`: run the configured BoltzGen pipeline steps (steps.yaml) in ONE process instead of one subprocess per step. argv: <run_dir> <seed> [step1,step2,...]. The module body below IS the run (no __main__ guard); xa_run.py executes this file's code by path.
# Seeding contract (the kit's, every mode): before step i, pl.seed_everything(seed+i, workers=True); random/np/torch seeded.
# Each step still builds its own model from its checkpoint (stock RNG consumption order preserved); what is saved is the per-step python
# start-up/import cost. With BG_GRAPH=graph the design step uses the graphed sampler (released before the next step).
import os, sys, time, json, random, importlib.util
import numpy as np, torch, yaml
import pytorch_lightning as pl
out_dir, seed = sys.argv[1], int(sys.argv[2])
only = sys.argv[3].split(",") if len(sys.argv) > 3 else None
import boltzgen as _bg
main_py = os.path.join(os.path.dirname(_bg.__file__), "resources", "main.py")
spec = importlib.util.spec_from_file_location("bg_main", main_py); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with open(os.path.join(out_dir, "steps.yaml")) as fh:
    steps = yaml.safe_load(fh)["steps"]
times = {}
# Process-global torch state that a fresh subprocess starts with (stock: one subprocess per step). Each Predict.run() may set
# torch.set_float32_matmul_precision(cfg.matmul_precision) (design: "high" = TF32; inverse_folding: null = untouched = "highest"),
# so between steps the defaults are restored -- otherwise the inverse-folding GEMMs after a design step would silently run in TF32 and its
# sequences could differ from stock's. Restored: matmul precision, cuDNN/cuBLAS TF32 flags, cudnn benchmark/deterministic, grad mode, autocast cache.
_STOCK_STATE = dict(matmul=torch.get_float32_matmul_precision(), cudnn_tf32=torch.backends.cudnn.allow_tf32, cublas_tf32=torch.backends.cuda.matmul.allow_tf32,
                    cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_det=torch.backends.cudnn.deterministic, grad=torch.is_grad_enabled())
def _restore_stock_state():
    torch.set_float32_matmul_precision(_STOCK_STATE["matmul"])
    torch.backends.cudnn.allow_tf32 = _STOCK_STATE["cudnn_tf32"]; torch.backends.cuda.matmul.allow_tf32 = _STOCK_STATE["cublas_tf32"]
    torch.backends.cudnn.benchmark = _STOCK_STATE["cudnn_benchmark"]; torch.backends.cudnn.deterministic = _STOCK_STATE["cudnn_det"]
    torch.set_grad_enabled(_STOCK_STATE["grad"])
    try: torch.clear_autocast_cache()
    except Exception: pass
for i, s in enumerate(steps):
    if only and s["name"] not in only: continue
    _restore_stock_state()
    os.environ["BOLTZGEN_PIPELINE_STEP"] = s["name"]; os.environ["BOLTZGEN_PIPELINE_PROGRESS"] = "Step %d/%d" % (i + 1, len(steps))
    step_seed = seed + i
    pl.seed_everything(step_seed, workers=True)
    random.seed(step_seed); np.random.seed(step_seed % (2**32)); torch.manual_seed(step_seed)
    t0 = time.time()
    m.main(os.path.join(out_dir, s["config_file"]), [])
    if torch.cuda.is_available(): torch.cuda.synchronize()
    times[s["name"]] = {"wall_s": round(time.time() - t0, 3), "seed": step_seed}
    try:
        import bg_graph_patch as g; g.release()
    except Exception: pass
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    times[s["name"]]["matmul_precision_during_step"] = torch.get_float32_matmul_precision()
    _restore_stock_state()
    print("inproc step %s wall_s=%.1f seed=%d" % (s["name"], times[s["name"]]["wall_s"], step_seed), flush=True)
json.dump(times, open(os.path.join(out_dir, "inproc_times.json"), "w"))
