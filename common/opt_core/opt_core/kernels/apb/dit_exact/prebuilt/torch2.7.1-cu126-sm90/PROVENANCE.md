# dit_attn_exact prebuilt (torch2.7.1-cu126-sm90)

- built from `csrc/dit_attn_exact.cu` (sha256 27af200d33676ad05fa1615bf1b3aa18b28a5c0dd211ed6ea60636c98ab8bc7c) by `build_prebuilt.py`
- torch 2.7.1+cu126, CUDA 12.6, Build cuda_12.6.r12.6/compiler.35059454_0; flags: -O3 -std=c++17 -gencode=arch=compute_90a,code=sm_90a --expt-relaxed-constexpr -Xptxas=-O3 -Xcompiler -ffile-prefix-map=<build dir>/=./; arch sm_90; python 3.11.5
- device at build: NVIDIA H100 80GB HBM3; built 2026-09-11T10:42:03Z in 43.3 s
- dit_attn_exact.so sha256 405933ef29262dd15ec073c1c28e93d091381017eaaf6e40b46f115238ddee2e
- load-time check: 3 cases bit-identical to torch.nn.functional.scaled_dot_product_attention (fp32, memory-efficient kernel) at build; output digests recorded in manifest.json
