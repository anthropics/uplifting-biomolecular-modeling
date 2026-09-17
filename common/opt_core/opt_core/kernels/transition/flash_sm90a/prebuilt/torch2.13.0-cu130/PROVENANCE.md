# flash_transition_sm90a.so — provenance

Built with `csrc/build.py` (torch.utils.cpp_extension, ninja) on an NVIDIA H100 80GB HBM3 GPU under the kit's stack image (torch 2.13.0+cu130,
CUDA 13.0, nvcc 13.0 r13.0/compiler.36424714_0, python 3.11) against NVIDIA CUTLASS 4.2.0 headers (`CUTLASS_INC`), flags in manifest.json
(`-gencode=arch=compute_90a,code=sm_90a`), compiled from a staging copy of csrc/ under a neutral path so the binary embeds no site-specific path.  manifest.json records sha256 of this .so and of the csrc files it was built from; the loader refuses a
tree whose csrc no longer hashes to the manifest.  Rebuild for another stack: `CUTLASS_INC=<include dir> python csrc/build.py` inside that stack
(writes prebuilt/<torch>-cu<cuda>/).
