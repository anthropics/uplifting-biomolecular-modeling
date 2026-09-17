fast_layer_norm_cuda_v2_stream.so — Protenix 2.0.0's fast_layernorm CUDA extension rebuilt for torch 2.13.0+cu130 / CUDA 13.0 / Python 3.11 /
sm_90 (NVIDIA H100) with its kernels launched on the current CUDA stream: the protenix 2.0.0 layer_norm kernel sources plus the stream-argument
launch patches in patched_src/ (arithmetic untouched), built by third_party/fastln_prebuilt.build_prebuilt_package(). manifest.json records the
torch / cuda / python / gpu it was built for, so_sha256, source_sha256 and patched_source_sha256, patched_launches, and the build-time checks
(bitwise_at_build over n_cases against the stock extension, side_stream_at_build). Infrastructure, not a lever: the loader
(fastln_prebuilt.install_prebuilt_fastln, require_bitwise=True) re-checks the torch version, the so digest and the bitwise / side-stream /
graph-replay properties in every process and refuses by name otherwise (the source-built extension then serves; outputs identical).
