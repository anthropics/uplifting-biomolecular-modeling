# ESM C kit — what changes vs stock

Stock = ESM C, `esm` 3.4.0 at the pin (STOCK.md). ESM C is a Python API, so the kit patches the loaded model rather than a command:
activation (`ESMC_OPT=exact` at `import esm`, or `esmc_opt.enable("exact")`) applies the datapath lever and wraps
`ESMC.from_pretrained`; each client that call builds is loaded through the datapath lever and handed to the forward lever's
`apply(client.model)`. `off` imports and applies nothing. A mode is all of its levers. Lever names are the ones printed on the
`ACTIVE … levers=…` and `APPLIED … levers_applied=…` lines (`opt/esmc_opt/registry.py`); each is a subpackage of `esmc_opt.kits`
applied only by its own `apply()`. Every kernel the kit adds lives under `opt/esmc_opt/kits/`; nothing is bound from the shared core.

## exact — outputs identical to stock

- `pipe` — the datapath around the forward (`kits/pipe`, composing `kits/boot`). `boot` serves `ESMC.from_pretrained` with its two
  parts, `U1_device_loader` and `U1_skip_meta_init`, named in its record: a device-side safetensors reader (the header parsed, the data region read in large `os.preadv` chunks from reader threads into pinned
  host slots, one asynchronous host-to-device copy per tensor) and a meta-device init that skips the throwaway random initialisation;
  each tensor's fp32 file bytes stay on the device only until its last byte range lands, then are cast in place to the dtype the stock
  load casts to next (bf16 off the CPU; a direct `EsmcForMaskedLM.from_pretrained(dtype=…)`: that dtype) and the staging is released,
  so the device never holds the fp32 checkpoint beside the bf16 model. `tok` runs the one-sequence call
  `tokenizer([seq], return_tensors="pt", padding=True)` through an exact character table (`<cls>` + ids + `<eos>`); any other call
  form or character goes to the stock tokenizer, counted. Both patch at the load, before or after `import esm` alike.
  Numerics: bitwise (no forward kernel or output tensor is touched; the parameter bits are the file's bits after stock's own cast).
  Steps aside: never on its own.
- `fused` — the eager fused forward (`kits/fused`, composing `kits/residual_ln`, `kits/qk_rotary`, `kits/thin`), engaged on every
  model `ESMC.from_pretrained` builds while the mode is active, in both call shapes (one protein per call, or a padded batch).
  The base patch (`_patch.py`) computes the varlen attention metadata once per forward instead of once per layer (one host sync per
  forward), makes the fp32 LayerNorm weight copies once at engage instead of per call, writes the normalised q/k into the QKV slots
  without a per-layer stack, turns unpad/pad into views at batch 1, builds `ESMC.logits` hidden states only when the `LogitsConfig`
  asks for them, and writes collected hidden states into the padded fp32 buffer with one scatter-with-cast Triton launch
  (`_hs_write.py`). Composed items (`COMPOSED_ITEMS`, selected per hidden size by `composition_for`, named on the kit's line):
  `residual_ln` — `x + y / sf` in one launch (ATen's opmath form) and residual + LayerNorm + qkv as CUDA C++ kernels from one source
  (`kernels.cu`; recipe `build.py`, flags `EXTENSION.json`), the LayerNorm mirroring the Transformer Engine forward template TE's own
  dispatch selects for the hidden size (one warp per row for 512 < d ≤ 2048, four warps per row for 2048 < d ≤ 8192) with the same
  contraction set; compiled once per machine by torch's extension builder with `nvcc` into
  `~/.cache/esmc_sdkfused/<ABI key>__<source+recipe sha>/` (by `run.sh install`, else at the first apply) and imported from there by
  every later process. `qk_rotary` — q/k LayerNorm + rotary embedding as one Triton kernel in ATen's vectorized LayerNorm thread
  order for any width divisible by 4 (the ragged last pass masked as ATen's strided loop leaves it); its compile classes are warmed at
  apply into `TRITON_CACHE_DIR`, so no compile lands inside a caller's forward. `thin` — launchers that call Transformer Engine's
  `layernorm_fwd` / `generic_gemm` / `swiglu` bindings, flash-attn's `varlen_fwd` and the cached Triton rotary launcher directly with
  the stock's own arguments, skipping the Python wrappers above them; templates are recorded by one short priming forward at apply,
  and a call outside the recorded class falls back to the stock callable. Serves the three ESM C configurations (`SERVED_CONFIGS`:
  hidden 960 · 30 layers · 15 heads, 1152 · 36 · 18, 2560 · 80 · 40; head dim 64); `apply(model)` takes no options and needs
  flash-attn importable.
  Numerics: bitwise — same kernels in the same order with fewer launches, and the two added kernels reproduce the arithmetic they
  replace in the same reduction order. Dependencies: flash-attn 2.7.4.post1 and Transformer Engine 2.15.0 (upstream's `accel` extra),
  whose bindings the launchers call.
  Steps aside: a model of any other configuration (`fused` refuses by name before anything is patched); the `residual_ln` extension
  not built and no `nvcc` / `ninja` to build it. Either way the `APPLIED` line carries `levers_fallback=fused`,
  `[esmc-opt] NOT ACTIVE: partial activation — fused: <reason>` is printed and `ESMC.from_pretrained` raises
  `esmc_opt.ActivationError` — nothing runs on a subset under the mode's name.

## Every mode

- No stock exception and no upstream fix: `esm` 3.4.0 runs as shipped on every mode.
- Activation refuses before any model is built (`[esmc-opt] NOT ACTIVE: <reason>`; exit 3 under `ESMC_OPT=exact`, an inactive report
  from `esmc_opt.enable`) for: an unknown mode word (under `ESMC_OPT` the word is named and the process runs stock), `esm` absent or
  installed from a commit other than the pinned one, no visible GPU, the marker variable `ESMC_KIT` already set, an `ESMC` client
  built before activation, or a second activation naming a different mode.
- Named, never a reason to step aside: torch / triton / flash-attn / Transformer Engine off their pinned versions
  (`drift=<dist>:<installed>(pin_<pinned>)` on the ACTIVE line); an accelerator the built model does not engage (a `KERNELS SHORT …`
  line after the model's `KERNELS route=… flash_attn=… rotary=… te=… xformers=…` line, which states what the bound objects engage);
  a Triton cache miss (compiled at apply, counted on the kit's line).
- Out of memory is never rerouted: every fallback branch re-raises it (`_oom.py`).
- Cards: nothing reads a card name or memory size, so H100, H200 and A100 (80 GB, 40 GB) run the identical code path and lever set.
  `run.sh install` builds the `residual_ln` extension for the visible GPU's capability (sm80 and sm90, `SERVED_SM`, when none is
  visible); `stock/PINS.json` `stacks.accel.jit_key` names sm90 (H100, H200), sm80 (A100 80 GB) and sm100 (B200) keys, while the
  pinned flash-attn / Transformer Engine wheels are upstream's `py312-pt211-cu13-sm80-90` set.

## Switches

- `ESMC_OPT=off|exact` per process, or `esmc_opt.enable(mode)` from code; `run.sh check --variant V [--mode M]` prints the same
  resolution as a `DRY-RUN` line with nothing applied. There is no switch that drops a lever from a mode and no kit variable that
  changes lever behaviour (variables: STOCK.md).
