protenix_fpf_msa — the MSA-module lever package behind lever words opm_fused / pwa_fused, served to protenix 1.1.0 through the adapter
`ptxfpf/trunk2_ptx1.py`.

Kernels: the shared core's `opt_core.ops.msa_fused.msa_triton` (ln_linear / opm_out / pwa_ln_vg / pwa_out2), re-exported by `__init__.py`.
trunk2_ptx1.py installs them on protenix 1.1.0's OuterProductMean / MSAPairWeightedAveraging modules of the MSAModule with cells sized for this
model's geometry (c_m 64, c_hidden 32, 8 heads x 32, c_z 128); `install.py` holds the package's own instance-level installs
(install_opm_fused / install_pwa_fused / report), which this kit's adapter does not call. pwa_fused's z-path pair-bias producer is
`opt_core.kernels.apb.fpf_apb.pf_triton.pf_bias`.
Load check: `python -m protenix_fpf_msa.loadcheck` with this lib/ directory and the shared core importable — e.g. from the kit root
`PYTHONPATH=opt/forward/v05_addon/lib:../common/opt_core "$STACK_PYTHON" -m protenix_fpf_msa.loadcheck` (needs a CUDA device; compares
against VECTORS.json). CELLS.json / NOTICE.md beside this file.
PROVENANCE: written for this kit; the Triton kernels it binds live in the shared core (`opt_core.ops.msa_fused`); no third-party source is included.
