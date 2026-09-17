# RFD1-002 — the IGSO(3) schedule cache is written in place (doc-only)

**What happens in stock.** `rfdiffusion/diffusion.py` caches the IGSO(3) schedule (`<inference.schedule_directory_path or the checkout's schedules/>/T_50_…_schedule_linear.pkl`)
on first use with a plain `open(path, "wb")` + `pickle.dump`. Several processes started together on an empty cache directory can observe a partially written
file and stop with `EOFError` / `UnpicklingError`. It has not been observed in the kit's runs; it follows from the code.
The cached table is computed with host-ISA-dependent CPU math (numpy / scipy), so its exact bits depend on the machine that first created it and are reused
afterwards — relevant only for bitwise reproduction across CPU host classes (ship the cache file with the checkout, or type a fresh `inference.schedule_directory_path=`
per host class).

**Status here.** Not fixed and no flag; every mode uses upstream's cache code as shipped. Avoid the race by running one design before starting many processes on a
fresh cache directory (`run.sh warm` does that), or by giving concurrent first runs their own `inference.schedule_directory_path=`.

**Proposed upstream change** (atomic write; recompute when an existing cache cannot be unpickled):

```diff
--- a/rfdiffusion/diffusion.py
+++ b/rfdiffusion/diffusion.py
@@ -128,9 +128,12 @@ class EuclideanDiffuser:
 def write_pkl(save_path: str, pkl_data):
-    """Serialize data into a pickle file."""
-    with open(save_path, "wb") as handle:
+    """Serialize data into a pickle file, atomically (several workers sharing one schedule directory
+    otherwise race on a half-written cache file)."""
+    tmp_path = f"{save_path}.tmp.{os.getpid()}"
+    with open(tmp_path, "wb") as handle:
         pickle.dump(pkl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
+    os.replace(tmp_path, save_path)
@@ -228,10 +231,14 @@ class IGSO3:
+        igso3_vals = None
         if os.path.exists(cache_fname):
             self._log.info("Using cached IGSO3.")
-            igso3_vals = read_pkl(cache_fname)
-        else:
+            try:
+                igso3_vals = read_pkl(cache_fname)
+            except (EOFError, pickle.UnpicklingError):
+                self._log.warning(f"IGSO3 cache {cache_fname} is unreadable (partial write?) - recomputing.")
+        if igso3_vals is None:
             self._log.info("Calculating IGSO3.")
```
