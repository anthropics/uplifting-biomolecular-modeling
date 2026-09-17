"""protenix_fpf_msa — fused kernels for the Protenix-v2 MSA module (by the fpf_apb authors, 2026-09).
Levers (TOLERANCE class, fast + big lines): `opm_fused` (OuterProductMean) and `pwa_fused` (MSAPairWeightedAveraging).
install_opm_fused(model) / install_pwa_fused(model) / report(); see install.py. Provenance: NOTICE.md (original code, no third-party source)."""
__version__ = "0.1.0"
from opt_core.ops.msa_fused.msa_triton import ln_linear, opm_out, pwa_ln_vg, pwa_out2  # noqa: F401
from .install import install_opm_fused, install_pwa_fused, report, CELLS, STATE  # noqa: F401
