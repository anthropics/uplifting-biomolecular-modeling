# LNSTREAM — shipped binaries of the current-stream fused-LayerNorm extension

`prebuilt/<torch>-<cuda>-<cpython>-<cxx11abi>/fast_layer_norm_cuda_v2_cs.so` + `manifest.json`: upstream OpenDDE's own fused-LayerNorm extension
source with its kernel launches placed on the current CUDA stream (`opt/opendde_opt/lnstream.py`), compiled once per stack for sm_90 and sm_80 in
one binary. The kit loads it only when the manifest names the digest of the source it would otherwise compile, the running device's architecture
and the file's sha256, and the `SHA256SUMS` line beside the `.so` matches the file; anything else is named on the lever's LEVER line
(`prebuilt=<reason>`) and the extension is built with nvcc as before (also what `run.sh warm` pre-pays). Re-make after a torch / CUDA / upstream
change: `python -m opendde_opt.lnstream prebuild`.
