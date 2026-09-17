# KNOWN_ISSUES.md — reproducibility facts of upstream Chai-1 (chai_lab 0.6.1) and the kit's deterministic recipe

## 1. Stock Chai-1 is not run-to-run bitwise on GPU at default numerics
Two `run_inference` calls with the same seed on the same GPU give different bits (cuBLAS / SDPA kernel nondeterminism); output files
differ between two stock runs, and the differences grow on flexible regions. A bitwise comparison with stock is therefore only meaningful
under the deterministic recipe: `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` + cuDNN deterministic
(`CHAI_DETERMINISTIC=1`, the kit's `--det 1`). The exported TorchScript modules run under it without errors; two stock processes
then write byte-identical CIF and score files. It costs speed, so it is off by default. The recipe does not cover every input: an input
carrying a modified residue (e.g. phosphoserine) or a tRNA chain (with or without an MSA) can still give different bytes from two stock
processes under it, so on such inputs a comparison with stock — `exact` included — is meaningful only within stock's own run-to-run spread.

## 2. First-call effect of the traced ESM2-3B fp16 module (TorchScript profiling executor)
`chai_lab/data/dataset/embeddings/esm.py` loads `traced_sdpa_esm2_t36_3B_UR50D_fp16.pt` and calls it once per unique sequence. Under
PyTorch's default profiling executor the first forward(s) of a freshly loaded traced module run the un-optimised graph and later forwards
the optimised graph; in fp16 the two give different bits for the same input. Within one stock process, `run_inference(seed=0)` (the first
ESM call) and later calls therefore embed the same sequence differently, and a driver that embeds each sequence once (this kit's W2 / W5)
matches stock's later calls but not its first one bit for bit — a difference of fp16 rounding order, inside stock's own run-to-run spread
at default numerics. `torch._C._jit_set_profiling_mode(False)` before the first scripted call makes every call identical; it is part of the
deterministic recipe on both the stock and the kit route (it disables TorchScript fusion, so it is not applied at default numerics).

## 3. Python `random` / NumPy / torch-CPU RNG are consumed during featurisation
Between `set_seed([seed])` and the first diffusion noise draw, stock advances the torch-CPU, NumPy and Python `random` generators inside
its collate / feature-generator code path while the CUDA generator is untouched. None of these draws influences a feature value at
inference settings (collated tensors are identical across seeds), but any re-implementation that skipped or reordered these steps would
change the RNG state at the first diffusion draw and hence the noise. The kit runs featurisation and input embedding per seed exactly as
stock, so every generator state at the first draw — and every noise tensor — is stock's.

## 4. Weights download
chai-lab 0.6.x downloads its weights from its CDN on first use into `CHAI_DOWNLOADS_DIR`; the kit's `run.sh install --weights DIR`
fetches and checks them once, and the kit modes refuse to start when a pinned file is missing rather than fetching at run time.
