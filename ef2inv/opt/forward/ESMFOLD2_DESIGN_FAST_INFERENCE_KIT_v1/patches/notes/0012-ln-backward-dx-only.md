# LayerNorm backward without dgamma / dbeta reductions (frozen LN parameters)
**What:** in the design loop all weights are frozen; the vendored `_ln_bwd` still reduces per-tile dgamma / dbeta partials with two extra
kernels per LayerNorm backward. agk's frozen-LayerNorm backwards (`LNFrozen`, `TransitionRefround`, through `agk._ln_bwd_dx_only`) launch the
vendored `_ln_bwd_kernel` directly for dx and skip the two reductions. `ef2_fused_ln` may rebind that helper to its own row-major dx-only kernel
(`k/README.md`).
**Numerics class:** exact — bitwise (the identical kernel binary computes dx; unit test `test_ln_dx_only_backward_is_bitwise_equal_to_vendored`).
