"""Triangle attention with pair bias and key mask, forward AND backward (jax.custom_vjp), Pallas (Triton lowering): the AF2-design
kit's `triatt` lever kernels carried for programs that differentiate triangle attention (see NOTICE; META: kernels/META/pallas_triatt.json).
Modules: triatt_attn (forward kernel, the op factory make_attention / attention / reference_attention), attbwd_dkdv (the backward kernels)."""
