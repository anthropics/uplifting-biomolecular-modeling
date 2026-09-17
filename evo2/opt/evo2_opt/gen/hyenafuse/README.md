# evo2_opt.gen.hyenafuse — fused Hyena decode step (generation member; modes exact and fast)

In cached generation (`Evo2.generate`) every Hyena block updates its recurrent state once per token in vortex's
`HyenaCascade.sequential_forward`: a 3-tap FIR over its three projection rows, then the inner FIR (hcs 7 / hcm 128 taps) or the 16-pole
IIR in pole/residue form (hcl), and the gating products. This module replaces that method, per block instance, with ONE Triton launch
(`kernel.py`) doing the stock's arithmetic in the stock's rounding order per element (fp32 FIR/IIR with separately rounded products and
sums, bf16 gate products, ATen's `torch.sum` order over the taps): block output and next states are bit-identical to the stock step's by
construction. Prefill, attention, norms, projections, MLP and the generation loop stay the stock's; only instance attributes are set.

    from evo2_opt.gen.hyenafuse import arm      # the kit arms it on every constructed model; by hand:
    h = arm(model)                              # [evo2-gen hyenafuse] ARMED: …  (HyenaFuseRefused after a REFUSED: <reason> line; nothing patched)
    h.status; h.describe(); h.uninstall()       # blocks per kind, fused_steps, fallback_steps; uninstall puts the stock step back

A prompt shorter than a block's FIR length makes the stock grow that block's state with `torch.cat`; those steps take the stock step for
that block until its state is full (one FALLBACK line, counted in `status["fallback_steps"]`); `arm(model, strict=True)` raises instead.

Requires CUDA + triton and vtx 1.1.0's layout (`HyenaCascade` inside `ParallelGatedConvBlock`, `vortex.model.engine`); `short_filter_length`
3, no short-filter bias, a bf16 `[3H,1,3]` short filter; inner FIR length 2..129 with bf16 filters, the bf16 gated bias `D` at ≥128 taps and
no bias below; IIR `state_size` a power of two ≤ 32 with fp32 `log_poles` / `residues` and bf16 `D`; `print_activations` off; `interleave`,
`column_split_hyena`, `hyena_flip_x1x2` in any setting (the row map comes from vortex's `interleave` / `column_split`). Anything else: REFUSED by name.
