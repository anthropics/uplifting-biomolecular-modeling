"""The lever names per route (the words of the APPLIED line) and the levers that are not applicable on a route, by name."""
ROUTES = ("torch-conv", "vortex-kernels")

LEVERS_BY_ROUTE = {
    "torch-conv": ("L1b_hcl_filter_cache", "E10_persistent_padded_fft_buffer", "L1c_hcm_filter_spectrum_cache", "E8_bf16_depthwise_fir", "E7_triton_fir3_featurizer",
                   "E11_fused_rmsnorm_tail", "E9a_hcl_fftconv_epilogue", "E9b_hcm_fftconv_epilogue", "E59E60_direct_cufft_plans", "C_SPEC_compact_half_spectrum",
                   "C2R_OUT_c2r_output_aliases_spectrum_storage", "L2_interleave_folded_into_te_master_weight", "W1_fp8_weight_workspace_cache",
                   "E50_out_filter_dense_matmul_on_transposed_view", "E56_fused_bias_residual", "E65_attention_residual_norm", "E66_residual_into_next_pre_norm", "T9_cublaslt_config_selection", "E61_general_gemm_out_buffer", "E63_pre_norm_fp8_emit", "R1_rotary_table_reuse", "GATE_forward_gate_shape_manager", "G1_forward_graph_replay"),
    "vortex-kernels": ("L1d_hcl_filter_spectrum_cache", "L1c_hcm_filter_spectrum_cache", "E10_persistent_padded_fft_buffer",
                       "E59E60_direct_cufft_plans", "C2R_OUT_c2r_output_aliases_spectrum_storage", "E9k_strided_conv_epilogues", "E7_triton_fir3_featurizer", "E7k_featurizer_fir_inline_hcl_hcm",
                       "E62_fused_hcs_gate_conv", "E11_fused_rmsnorm_tail",
                       "L2_interleave_folded_into_te_master_weight", "W1_fp8_weight_workspace_cache", "E50_out_filter_dense_matmul_on_transposed_view", "E56_fused_bias_residual", "E65_attention_residual_norm", "E66_residual_into_next_pre_norm", "T9_cublaslt_config_selection",
                       "E61_general_gemm_out_buffer", "E63_pre_norm_fp8_emit", "E64_channels_first_projection", "R1_rotary_table_reuse", "GATE_forward_gate_shape_manager", "G1_forward_graph_replay"),
}
NOT_APPLICABLE = {
    "torch-conv": {},
    "vortex-kernels": {**{n: "the vortex kernels (hcs_conv / hcm_fft_conv / hcl_fft_conv) compute this stage on the use_kernels=True route: the stock's own kernel runs"
                          for n in ("E8_bf16_depthwise_fir", "E9a_hcl_fftconv_epilogue", "E9b_hcm_fftconv_epilogue")},
                       "C_SPEC_compact_half_spectrum": "hcl_fft_conv / hcm_fft_conv already transform one-sided (torch.fft.rfft): there is no full spectrum to compact",
                       "L1b_hcl_filter_cache": "the long filter is consumed once per (layer, length) by the spectrum fill (L1d) and never stored on this route"},
}
