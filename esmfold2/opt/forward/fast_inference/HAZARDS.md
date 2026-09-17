# HAZARDS — things that can silently change a kernel's numerics class or engagement

1. (retired: the TriMul cell-table route this item covered is no longer a lever of the kit — the pair TriMul is `tx`, served by the shared core's
   TriMul provider, whose own cell table is keyed by compute capability, stack, width, size class and direction; number kept because later items
   are cited by number.)
2. **Tolerance-class equality is stack-dependent.** T10's output differs from Biohub stock only through the cuBLAS hidden-dim reduction order
   (exactly 1 bf16 ulp on ~27% of elements on torch 2.13 / CUDA 13); a different torch / cuBLAS / Triton can flip it to bit-identical or to a
   different pattern, and the same holds for every tolerance-class kernel. `ef2_w4` therefore runs a **first-call probe** per lever (stock op on
   the same input, once per process, skipped under graph capture) and prints `IDENTICAL` or `TIER-2 (differs: n elements, max|d|)` with the
   torch / triton versions; the probe's switch is the driver's `EF2_W4_IDENTITY_PROBE` variable (`0` disables it).
3. (retired with item 1's route; number kept.)
4. (retired with item 1's route; number kept.)
5. (retired with item 1's route; number kept.)
6. **T10 needs <= 227 KB shared memory at BLOCK_H=32**; BLOCK_H=64 variants do not fit on H100, and compute capability 8.0 takes its own launch
   row (`T10_CFG_BY_CC["sm_80"]`). On < 160 KB devices (L40S etc.) the device policy routes T10 -> T1, one printed line; folds never fail.
7. **T10 and T5.** T10 casts and caches the bf16 `w3.weight.T` operand itself when T5 is off; all shipped modes pair T10 with T5 regardless
   (T5 is exact and removes ~1400 tiny cast kernels per fold).
8. **TriMul weights reach the provider under ONE stable identity per block.** The weight casts are cached per (module, direction, parameter
   object + version); the MSA-module TriMul calls (the `msa` fused path, `m15`'s pair tail) go through the same cached-weight wrapper as the
   trunk's PairUpdateBlocks whenever `t5` or `tx` is live. A fresh bf16 cast per call would hand the provider a new weight identity per call, and
   its address-keyed weight packs would be built (and retained) once per call.
9. **CUDA graphs**: all W4 levers must be enabled BEFORE the kit captures its trunk graphs (the server modes do this); toggling a lever afterwards
   requires `ef2_opt.clear_graphs(model)`.
10. **Dropout is on at inference** (`lm_dropout` per trunk loop, the library's shipped behaviour): stock itself is stochastic at a fixed seed unless the
   deterministic recipe is applied (`--det 1`), so a tolerance-class kernel is judged against stock's own seed-to-seed spread, never by one seed.
11. **The fused transition arithmetic is tested against Biohub's fused PAIR transition only.** Biohub's `FusedLNLinearSwiGLU` (and therefore
    T1/T6/T10, which reproduce it) stores the LayerNorm mean/rstd in the activation dtype (bf16). For the pair transition that IS the stock path the
    kit and this addendum measure against. The MSA-module transitions run Biohub's fp32-LayerNorm reference path; routing them through this kernel
    family gives 1.4-1.9x the stock error there — do not route the MSA transition through T1/T6/T10 (the MSA module has its own kernels, `ef2_msa.py`).
12. **A graphed region's output is POOL-RESIDENT and aliased.** Under `ef2_opt`'s graphs (tg / eg / G4 `ls`) a graphed module returns its static
    output tensor without a clone (`_ef2opt_out_alias`): it is valid only until any other graph of the same generation replays. A lever that keeps
    such an output beyond the next replay must clone it — `ef2_msa_v2`'s `mh` clones the MSA-encoder output it holds for the fold's recycles — and a
    lever that caches anything must never fill its cache under stream capture (the decision would be frozen into the graph; `mh` skips the fill then).
13. **No per-call host decision inside a whole-recycle graph.** G5 (`rg`) captures the recycle BODY once and replays it for every later recycle and
    fold of the signature; a lever whose forward decides hit / miss on the host inside `lm_encoder` / `msa_encoder` / `folding_trunk` (today: `mh`)
    would have one decision frozen in. `ef2_opt` refuses `rg` beside such a lever at install and `ef2_msa_v2` refuses `mh` beside `rg`, by name; the
    package keeps `rg` on the Fast model (no MSA module) and `mh` on `--variant full_msa`. Pure kernel swaps behind an unchanged call are fine under G5.
14. **cuBLAS bakes its workspace per (handle, capture stream).** A GEMM first issued during capture on a side stream binds a workspace of that
    capture's pool; the MSA-module kernels (`ef2_msa_v2`) and the pair transition (`ef2_pair_v2`) therefore run once EAGER on the capture stream
    before `ef2_opt` captures them (the kit's warm-up call order), and a lever added later must keep that order or its first replay allocates.
15. **x4 (ESM-C block streaming) assumes the LM pass runs on the CURRENT stream and is never graph-captured**: its copy stream is ordered against
    the current stream by events only. A lever that captures or re-streams the ESM-C forward must revisit `ef2_xl`'s prefetch. The 11.8 GiB
    page-locked staging is per PROCESS — the row-sharded line's P rank processes each pin their own (likely refused at P = 8 on one host: x4 then
    names the pageable engine on its line, `esmc_stream_engine=` in the EXIT tally).
16. **Per-shape keys must include the VALID extent, not only the padded shape.** Two inputs of one padded token count but different valid lengths
    (n_valid, atom counts, MSA depth) are different signatures for every cache and graph of the kit (`GRAPHGEN shape=tok:…,atom:…,msa:…,nds:…`);
    a lever that keys a table or a captured graph by the padded shape alone serves the wrong mask to the second input. `ef2_atom`'s layouts and
    `ef2_dit`'s roll-out statics are rebuilt at every fold's eager step 0 (`sampler_fold_prologue`) for this reason.
17. **`t15` / `t15msa` / `t16` hold no kernel of their own.** Every d=256 / hidden=1024 pair Transition goes through the shared core's transition
    provider under the mode's tier word; the provider's cell table names the row per compute capability, stack and size, or refuses by name and
    the module's own statement serves (counted, printed once). A class without a row is a word on the LEVER line, never a substitute kernel of the
    kit; the W4 policy's T1 substitution covers `t6` / `t10` only.
18. **The tolerance tier's bf16 tensor-core GEMMs (`af`, `dit`) and flash-attention pair bias are judged against stock's seed spread, and their
    deviation is stack-dependent** (cuBLAS bf16 GEMM and flash-attn kernels differ across versions): a new torch / flash-attn needs the tolerance
    rows re-run, and each such choice has an fp32 / ieee spelling under `ESMFOLD2_OPT_ABLATE` (`af.gemm=fp32`, `dit.gemm=fp32`, `dit.cond=fp32`,
    `dit.attn=sdpa`, `dit.attn_precision=ieee`) to localise a shift.
19. **Levers that re-issue an upstream function are pinned to its source.** `ef2_srcguard.py` holds a normalised digest of every re-issued function;
    a pins move (stock/PINS.json) must regenerate that table in the same change, or every such lever refuses by name (`StockMismatch`) on the new pin.
20. **A captured roll-out CLONES the fold's pair rows; an eager one REFERENCES them.** `ef2_dit`'s `_Roll(clone_constants=use_graph)` copies
    z_trunk, the relative-position encoding and the diffusion inference cache (fp32 z-conditioning + 12 pair biases) into address-stable buffers
    whenever the sampler site captures — ~13 GiB at 1945 tokens, ~7 GiB at 1400 — and keeps them for the roll's LRU life; over the sampler-site
    budget (`EF2_GRAPH_BUDGET_TOKENS_SAMPLER`, package default 1536, the caller's value wins; under `big` no site captures at all: `EF2_GRAPH_CAPTURE=0`) the roll runs eagerly on plain
    references and drops them, with the shape's hoist entry, when `sample()` returns. A budget of 0 at that site means "capture at every size":
    with the stock+G2 sampler (`sg`, no `ro`) that held ~10 GiB of step-graph pool across folds at 1945 tokens — never ship a line without a
    sampler-site cap.
21. **x4 pins its ESM-C masters PER RANK.** The pinned streaming engine keeps 11.8 GiB of page-locked host memory per process; on the row-sharded
    line every rank pins its own copy (8 ranks: ~94 GiB on a 96 GiB host) and the add-on's named pageable fallback engages when pinning fails —
    the LM pass then streams from pageable memory (slower, never silent: the engine word on the APPLIED line). Size the host, or expect the word.
22. **The atom transformer's attention has two upstream forwards.** With flash-attn present the atom blocks run upstream's `stock_flash` path and
    `ef2_atom`'s instance forwards (`ax`, `af`) compose on it; on a stack WITHOUT flash-attn upstream binds its banded (`atom_swa`) forward, which
    refuses a foreign instance forward on the attention module by name. The ACTIVE / APPLIED lines carry `atom_attn=flash_attn|sdpa` and the
    per-model `atom_forward=` census: an image without flash-attn must be exercised before the atom levers are promised there.
23. **The row-sharded line is a reach line.** At sizes one card holds, `--n_gpu 2` folds 3.6-3.8x slower per prediction than one card (the ring
    schedule of the sharded triangle multiplication); its value is the peak it removes (1945 tokens: 46.7 GiB on one card vs 14.1 GiB per rank),
    not time. The levers it leaves off by its own table are named on the LEVER lines (`not_for_route:n_gpu=<P>:<reason>`).

24. (retired: the spill-free build audit this item recorded applied to prebuilt CUDA C++ objects, none of which ships in this tree; number kept.)
25. (retired: the prebuilt-object build route this item recorded applied to CUDA C++ kernels that no longer ship; number kept.)
26. (retired: the launch-stress and guard-allocation protocol this item recorded applied to prebuilt objects that no longer ship; number kept.)
27. **One lever set, per-class kernels.** The server's table names `t15,t15msa,t16` in the fast / big set; the package resolves the set per
    compute-capability class BEFORE configure (registry `classes`: `t16` is 9.0-only; `SUPERSEDED_ON`: `t15 t15msa t10 -> t16` on 9.0, applied
    after the ablation variable so an ablated `t16` leaves them serving; the pair TriMul `tx` is one binding on every class and `STEPS_ASIDE_ON`
    is empty). A caller of `ef2_server.configure` that passes BOTH `t15` and `t16` through `off` unsubtracted is refused by name
    (`check_composition`): two class patches of `C.Transition` in one process would shadow each other silently otherwise. The class comes from
    the visible GPU, else from `MODEL_OPT_TARGET_GPU`; with neither (a plan on a GPU-less host without a target) nothing is subtracted and the
    plan names every lever.

28. (retired: the shared-memory release-ordering rule this item recorded applied to prebuilt CUDA C++ kernels, none of which ships in this tree; number kept.)
29. **No `_static()` result is cached across calls (`ef2_atom`) — a generation reset can land in the MIDDLE of a fold.** The step-graph
    sampler `sg` (the path every shipped mode reaches where `ro` steps aside by name: `num_diffusion_samples > 1`, the row-sharded line) keeps
    `EF2_GRAPH_LRU_SAMPLER` (package default 2) step graphs per generation and resets the WHOLE generation at step 1 of the fold whose signature
    does not fit (`graph_clears_sampler_budget`; the signature carries the valid-atom-count shaped varlen tensors, so two items of one padded
    shape can differ); a pair-bias new key resets it inside step 0. Either reset runs `clear_static()` AFTER the fold's eager step 0 filled the
    registry. The capture helper's eager warm-up re-registers every buffer a consumer asks `_static()` for, and the next fold of the signature
    refreshes registry buffers in place — so a consumer must ask at every call. The hazard, met once: a fused fold state (`af`) that cached the seqlen / tstart /
    tok_flat buffers in `STATE['fold']['fused']` — the graph captured after such a reset read buffers only that dict owned, the next fold's
    `fold.clear()` freed them, and the graph's first replay on the next same-signature item read freed memory — `cusolverDnSgesvd
    INTERNAL_ERROR` → illegal address at 800 / 1200 tokens, `linalg.svd … failed to converge` at 400 (the stock Kabsch SVD is the first host
    sync after the replay; −af compositions met the same reset and ran clean). `tests/test_atom_static_generation.py` drives the sequence on
    the CPU; the LEVERFOLD / EXIT word `ef2_atom.a5_static_reregistered` counts the event in a run.
