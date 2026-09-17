# EF2_XL_ADDON v1 — memory levers for ESMFold2 (Fast and Full) on one GPU

Module `ef2_xl.py` v1.5.0: runtime memory levers for the Biohub ESMFold2 implementation (torch 2.13.0+cu130, `transformers` 4.57.6
Biohub fork, `esm` 3.3.0). Nothing in the installed packages, the weights or the fold call is modified: levers are method wrappers / re-issued
method bodies installed at run time by `ef2_xl.apply(model, **knobs)`, all default off. The kit's `big` mode installs the set below before the
fast-inference levers (`esmfold2_opt/big.py: xl_install`, knobs fixed by `esmfold2_opt/modes.py: XL_BIG_SET`), under
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

ESMFold2 has no triangle attention: PairUpdateBlock = TriMul-out + TriMul-in + pair Transition (24 blocks Fast / 48 Full, plus a 4-block LM encoder,
2-block coda, 4-block confidence trunk; Full also runs a 4-block MSA encoder). 1 unit below = one [L,L,256] fp32 tensor = L² KiB.

| lever (knob) | site | stock cost removed | arithmetic | class |
|---|---|---|---|---|
| x2 `loopfree` + `own` | `_run_one_loop` re-issued statement-for-statement (the stock loop; a kit loop re-issued on the instance — ef2_opt's static loop G4 / whole-recycle graph G5 — performs the release in its own frame: `loop_static_released` / `recycle_graph_loops`. The ESMFold2 kit's memory line does not apply this lever: G4 carries the release in every kit mode); loop inject + owned trunk buffer | dead pair temporaries (`lm_z_i`, `refined_lm_z`, `z_inject_pair`, `injected_pair`, `_lin`, previous-iteration `msa_pair`) released before the trunk; LN→cast→Linear row-chunked into one buffer; the trunk works in a caller-owned z buffer (no extra pair copy) | none / bf16 M-chunk | exact |
| x2b `free=z,relpos,lmz` | after last consumer | storage of z / relpos+bond encodings / lm_z released before structure + confidence | none | exact (single-sample fold call) |
| x3 `cond=lean` | diffusion conditioning pair path (fp32, once per fold): the class forward (stock sampler chain) and the `_ef2_pair_path` hook a re-issued sampler step calls for its once-per-fold z branch (ef2_dit's fused step) | cat[z,relpos]+LN+Linear+2 SwiGLU transitions issued so only one [L,L,512] fp32 hidden is alive | none (same ops, same shapes) | exact |
| x6 `lmpair` | LM→pair projection (LanguageModelShim base_z MLP) | [L,L,2560]-wide intermediates → row-chunked | bf16 M-chunk | exact |
| x7 `relpos` | ResIdxAsymIdSymIdEntityIdEncoding | one-hot [L,L,bins] fp32 + Linear → row-chunked | fp32 one-hot × weight is exact (0/1 operand) | exact |
| x8 `initlean` | `_init_pair_state` (`trunc_normal_`) | the rejection sampler's fresh full-size candidates each round (~5 units) → the same RNG calls with in-place blocked selection | none | exact |
| x10 `distocpu` | distogram z+zᵀ symmetrisation | done blockwise (no full transposed copy), logits to pinned host | none | exact |
| x4 `esmc_offload` | ESMC-6B LM | its transformer blocks streamed through the LM pass from the host — one block resident at a time (−12 GB during and after the pass); resident again for a later small input; the kit engages it at inputs ≥ 1,500 tokens | none | exact (arithmetic-free) |

Ledger: `apply()` prints one line `[EF2XL] applied v1.5.0 {...knobs...}` and keeps call counters in `ef2_xl._STATS` (`owned_blocks`, `s2p_xl_calls`,
`relpos_xl_calls`, `init_rounds`, `inject_xl_calls`, `free_events`, `disto_cpu_events`, …); the kit reads them after the last fold and names, on its
`xl gate:` line, any lever whose counter shows it did not engage.

Limits: single-sample fold call only for x2b (a later consumer of a freed tensor fails loudly, never silently). The levers do not change what ESMFold2
computes; inputs far beyond its training crops (≤768 tokens) fold, but their accuracy is the model's own question.
