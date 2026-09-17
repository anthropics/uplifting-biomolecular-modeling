# libtriatt_procuda_sm90a.so — provenance

* **Origin:** compiled from `../csrc/triatt_procuda_sm90.cu`, which was written in this repository for this kit. No third-party source is
  included or linked: the kernel uses only the CUDA toolkit headers (`cuda.h`, `cuda_runtime.h`, `cuda_bf16.h`) and inline PTX
  (mbarrier, cp.async, cp.async.bulk.tensor, wgmma, ldmatrix/stmatrix). No CUTLASS / CuTe / cuDNN / cuEquivariance or other third-party library code, headers or
  binaries are used. The LayerNorm arithmetic it reproduces is the kit's own F1 prologue cell (opt_core `fpf_mkpf`, 'welford' emulation of the
  upstream Protenix `fast_layernorm` extension's arithmetic); the projection arithmetic is the hardware bf16 tensor-core product with fp32
  accumulation in ascending-k order, as in that cell.
* **Build:** `python build.py --nvcc /usr/local/cuda/bin/nvcc` in the package directory, on the kit's stack image:
  `nvcc -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a --shared -Xcompiler -fPIC -Xcompiler -ffile-prefix-map=<csrc>=. -o prebuilt/libtriatt_procuda_sm90a.so triatt_procuda_sm90.cu`
  (run with the source directory as working directory so only the relative file name is recorded; no `-lineinfo`, no fast-math).
  Toolchain of this binary: `Build cuda_13.0.r13.0/compiler.36424714_0`; CUDA runtime linked statically (nvcc default); loads through `ctypes`.
* **Identity:** `manifest.json` records `so_sha256` and `src_sha256`; `binding.load()` refuses to load when either differs from the files present.
  This binary: so_sha256 `e9009ba57e432421cf1943cbbe5c22777bc1a674ca0423448ac2f06b9baecbcf`, src_sha256
  `fc120646461cd993a7d9fc94886a137c74873887da251fa538b1701f7880c1c1`, built 2026-09-11.
* **Rebuilds:** nvcc output is not byte-reproducible across toolchains; a rebuilt binary gets a new manifest and must reproduce the F1 cell's outputs
  bitwise (the package compares the first call of every new shape against that cell in-process and refuses by name on any difference).
* **Target:** sm_90a only (Hopper); other architectures cannot load it and the lever refuses by name at install.
