# Fused pair-track kernels for RoseTTAFold3 (foundry `rf3` @ 4010e3e2e) — the FPF adapter

`rf3fpf/fpf_rf3_adapter.py` re-binds, at run time and class-wide (no file or weight edits), the pair-track modules of `rf3` to the
shared core's kernel providers (`opt_core.kernels`: `trimul`, `triattn`, `transition`, `apb`, `ln`; `opt_core.ops.msa_*`), imported
through `rosettafold3_opt.stack.kernels_route`. Inference only (eval mode, bf16-mixed autocast as rf3 runs).

An arm is `<trimul>[+<component>...][@L1[.warm]]`, applied once per process by `apply_arm(...)`; `describe_v2()` returns the
per-component served / fallback counters the kit prints in its LEVER and census lines. The kit's modes choose the arm
(`rosettafold3_opt/modes.py`). Components (a `.<word>` sub-word names a provider row or tier word; without it the component's default
word applies):

- trimul head `stock` | `fast[.<word>]` — the fused triangle multiplicative update; `xmul[.<word>]` — the stock statement with its one
  library call served by the provider's exact row (`fpf_rf3_trimul_rows.py`);
- `gflash[.<word>]` — fused triangle attention: Triton LayerNorm/cast/transpose/bias prologue, one q|k|v|g projection, the provider's
  attention row, fused gate/transpose epilogue; `xatt[.<word>]` — the stock statement with its library call served by the exact row
  (`fpf_rf3_triattn.py`);
- `ttr` — the pair- and MSA-track transitions through the transition provider under the arm's tier word; `res` — residual adds fused
  into the trimul / transition epilogues;
- `apb[.<word>]` — Triton LayerNorm + pair-bias producer of the pairformer's attention-pair-bias, the attention core per the sub-word
  (`fpf_rf3_apb_rows.py`); `sapb` — the stock attention-pair-bias in a capture-safe form (bitwise);
- `tg` — the 48-block pairformer stack captured as one CUDA graph per token count and replayed (needs `sapb` or `apb`; the graph pool is
  bounded by `FPF_RF3_TG_MAX`); it replays exactly the kernels the eager arm launches;
- `dattn` — the diffusion transformer's dense attention seam, through which the kit's `dtk` lever installs its kernel;
- `msa[.<word>]` / `smsa` — the MSA module's outer-product-mean and pair-weighted averaging on fused cells (`fpf_rf3_msa_rows.py`);
- `xln[.<word>]` — `torch.nn.LayerNorm.forward` through the LayerNorm provider (`fpf_rf3_ln_rows.py`);
- `@L1[.warm]` — the arm runs with the kit's patched-file flags on (`RF3_CUDAGRAPH=1 RF3_HOIST=1`, `../rf3_xattempt_addon/README.md`);
  `.warm` sets the sampler's warm-up lever.

Numerics: `xmul`, `xatt` and `xln` under their default words, `sapb` and `smsa` keep stock's bits; the other components are
tolerance-class (bf16 operands, fp32 accumulation and LayerNorm statistics, a different summation order), identical from run to run.
A call a component cannot serve (token floor, width, card, a row's refusal) runs the stock statement for that call, counted by name
in the census — never silently.
