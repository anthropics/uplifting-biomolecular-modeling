# RFD1-001 — the first design of a process is not reproducible at another process position (doc-only)

**What happens in stock.** With `inference.deterministic=True`, design *i* is seeded with *i* (`scripts/run_inference.py` `make_deterministic(i_des)`), so one
expects `inference.design_startnum=i inference.num_designs=1` to regenerate design *i* of a longer run. It does not: the first design any process makes
differs from the same seed made at a later position of another invocation, while later
positions agree bitwise with one another.

**Cause.** Not the random state — torch (CPU and CUDA, Philox offset included), numpy and Python RNG states are bitwise equal at seeding, after
`sample_init`, after step 1 and at the end, and x_T is identical. The first differing operation is the TorchScript-scripted `NormSE3` of the first SE(3)
layer in step 1: PyTorch's profiling executor runs the generic graph on a scripted function's first calls and the specialised, fused graph afterwards,
and the last-bit difference between the two is amplified by the 50-step stochastic trajectory. With the profiling executor disabled, first == later
bitwise. The same effect recurs at the first design of a new (contig, length) shape inside one long-lived process.

**Status here.** Not fixed and no flag: every mode runs upstream's behaviour, and the kit's resident driver keeps the same position classes (its first
design of a shape is stock's first-of-process class, later ones the warmed class), which is why `exact` equals `off` design position for design position.

**Proposed upstream change** (two throw-away denoising steps right after the sampler is built and before the per-design re-seeding; they draw no
random number that survives the re-seed; the cost is two extra steps per process; with it, design *i* is bitwise the same at every position and equals stock's later-position class):

```diff
--- a/config/inference/base.yaml
+++ b/config/inference/base.yaml
@@ -19,6 +19,8 @@ inference:
   symmetric_self_cond: True
   final_step: 1
   deterministic: False
+  jit_warmup: True
+  jit_warmup_steps: 2
   trb_save_ckpt_path: null
--- a/scripts/run_inference.py
+++ b/scripts/run_inference.py
@@ -35,6 +35,18 @@ def make_deterministic(seed=0):
     random.seed(seed)


+def _warmup_sampler(sampler, log, n_steps=2):
+    """Run n_steps denoising steps on a throw-away trajectory (RNG state is re-seeded per design afterwards)."""
+    t0 = time.time()
+    x_init, seq_init = sampler.sample_init()
+    x_t, seq_t = torch.clone(x_init), torch.clone(seq_init)
+    t_start = int(sampler.t_step_input)
+    for t in range(t_start, max(t_start - n_steps, sampler.inf_conf.final_step - 1), -1):
+        _px0, x_t, seq_t, _plddt = sampler.sample_step(t=t, x_t=x_t, seq_init=seq_t, final_step=sampler.inf_conf.final_step)
+    torch.cuda.synchronize()
+    log.info(f"JIT warm-up done in {time.time() - t0:.1f}s")
+
+
 @hydra.main(version_base=None, config_path="../config/inference", config_name="base")
 def main(conf: HydraConfig) -> None:
@@ -53,6 +65,9 @@ def main(conf: HydraConfig) -> None:
     # Initialize sampler and target/contig.
     sampler = iu.sampler_selector(conf)

+    if conf.inference.get("jit_warmup", True) and torch.cuda.is_available():
+        _warmup_sampler(sampler, log, n_steps=int(conf.inference.get("jit_warmup_steps", 2)))
+
     # Loop over number of designs to sample.
```
