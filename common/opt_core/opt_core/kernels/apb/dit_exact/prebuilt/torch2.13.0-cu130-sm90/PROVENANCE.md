# dit_attn_exact prebuilt (torch2.13.0-cu130-sm90)

- built from `csrc/dit_attn_exact.cu` (sha256 27af200d33676ad05fa1615bf1b3aa18b28a5c0dd211ed6ea60636c98ab8bc7c) by `build_prebuilt.py`
- torch 2.13.0+cu130, CUDA 13.0, Build cuda_13.0.r13.0/compiler.36424714_0; flags: -O3 -std=c++17 -gencode=arch=compute_90a,code=sm_90a --expt-relaxed-constexpr -Xptxas=-O3 -Xcompiler -ffile-prefix-map=<build dir>/=./; arch sm_90; python 3.11.5
- device at build: NVIDIA H100 80GB HBM3; built 2026-09-11T05:54:47Z in 38.2 s
- dit_attn_exact.so sha256 f95aea477b48cd92c960b761c581b816c93a84cf24b6545706490cc25054c9ed
- load-time check: 3 cases bit-identical to torch.nn.functional.scaled_dot_product_attention (fp32, memory-efficient kernel) at build; output digests recorded in manifest.json
