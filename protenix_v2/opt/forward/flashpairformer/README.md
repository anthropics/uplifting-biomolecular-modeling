# FLASHPAIRFORMER — fused Pairformer trunk for Protenix v2 (2.0.0 @ 2475421)

Two arms for the stock `protenix pred` CLI (protenix-v2 at its CLI defaults: bf16, cuEquivariance triangle ops,
LAYERNORM_TYPE=fast_layernorm), selected by sourcing `env.sh`, the unit's one switch table. Nothing about the model changes: same
weights, inputs, MSAs, recycles, diffusion steps, seeds, output files; only how the trunk executes. `protenix_opt` sources `env.sh` for
its modes (`exact` = ARM E; `fast` and `big` = ARM T) and sets the per-card switches of the table below around it.

## Arms

| arm | what runs | class |
|---|---|---|
| **ARM E (default)** | tuned cuEquivariance TriMul tiles + template distinct-evaluation + DEADSKIP + the fused block path (row-chunked above the memory policy's token threshold) + the pair-stack TriMul and the transitions through the shared core by the word `exact` + CUDA-graph replay of the 48-block stack for small inputs | EXACT: equal to stock bitwise under the deterministic recipe |
| **ARM T (opt-in)** | ARM E + the K2B flash triangle-attention kernel in the block core (or the shared core's triangle-attention provider by the mode's tier word, `PTX_T_ATT`) + the pair-stack TriMul and the transitions by the tier word (`PTX_TRIMUL=tier`) | Tier-2: bf16-rounding class, not bitwise |

## Switches around env.sh

* ARM E composition: `ARM=E source env.sh; export PTX_E_PAD8=1 PTX_GLUE_V2=1`.
  - PTX_E_PAD8=1 (`fpf_cueq_pad8exact`): the stock cuEquivariance triangle attention on a copy padded to a multiple of 8, cc 8.x/9.x
    only, when N%8 != 0 and N >= PTX_E_PAD8_MIN_TOKENS; the provider refuses kv_len < 104 and checks torch.equal against the unpadded
    call on its first call.
  - PTX_GLUE_V2=1 (`fpf_glue_v2`, served from the shared core): restructured prologue_v4 (+ padded) / epilogue_v3 kernels; cells keyed
    cc|triton in the core package's own table; a (cc, triton) without a cell refuses by name and the unit's own prologue / epilogue
    kernels run.
* ARM T composition: `ARM=T source env.sh; export PTX_GLUE_V2=1`; on the cc-9.0 and cc-8.0 rows also `export PTX_T_ATT=fast` before
  sourcing (`big` exports `PTX_T_ATT=big`): the mode's tier word, handed to the shared core's triangle-attention provider
  `opt_core.kernels.triattn` (lever `triattn_native`); the provider's cell table names the kernel per card, head count and N, and the
  K2B kernel serves where it has no cell.
  - PTX_TRIMUL=exact|tier (exported by env.sh): the pair-stack TriMul (c_z 256) through the shared core's one provider
    `opt_core.kernels.trimul` by tier word — `exact` under ARM E, the mode's `fast` | `big` under ARM T (`src/ptx_trimul_routes.py`);
    the unit carries no TriMul kernel and pins no row.
  - PTX_GLUE_V2=1: as in ARM E (the prologue / epilogue kernels serve the K2B path too).
* PTX_MK_PF=F1|F1,F3 (`fpf_mkpf`): fused LN-in-registers triangle-attention prologue (F1 start/end incl. the PAD8
  padded producer) and transition (F3, cc 10.0 only); cells keyed cc|triton; a stack without a cell refuses by name.
* PTX_LAZY_INIT=1 (`src/ptx_lazy_init`): skips the random parameter init at model construction (start-up only; the weights come from the
  checkpoint).

Order of env operations: `export PTX_T_ATT=fast|big` (ARM T, on the cc-9.0 and cc-8.0 rows) -> `ARM=<E|T> source env.sh` -> `export
PTX_E_PAD8=1 PTX_GLUE_V2=1 [PTX_MK_PF=...] [PTX_E_PAD8_MIN_TOKENS=512]` (the row's post switches).

## Compositions per (cc | triton)

| stack (cc \| triton) | E\* (EXACT) | T\* (Tier-2 arm) | MK-PF | notes |
|---|---|---|---|---|
| 9.0 \| 3.3 (H100: torch 2.7.1 / triton 3.3.1) | `ARM=E; PTX_E_PAD8=1 PTX_GLUE_V2=1 PTX_E_PAD8_MIN_TOKENS=512 PTX_DIT_ATTN_EXACT=1 PTX_TRIATT_EXACT=1 PTX_ATOM_ATTN_EXACT=1 PTX_TRIATT_PROCUDA=1` (dit_attn_exact: the exact composition's DiT attention kernel, sm_90 prebuilt — this cc only; triatt_exact: the pair stacks' triangle attention through opt_core.kernels.triattn by the tier word exact — the provider's bit-identical row where its cell table vouches for this stack, the library op by name elsewhere) | `PTX_T_ATT=fast; ARM=T; PTX_GLUE_V2=1 PTX_DIT_ATTN=1 PTX_DIT_ATTN_FP16=1 PTX_ATOM_ATTN=1 PTX_PF_ATTN=1 PTX_OPM_FUSED=1 PTX_PWA_FUSED=1 PTX_COND_DEDUPE=1 PTX_DIT_FAST=1 PTX_DIT_LOWP=fp16 PTX_ATOM_FAST=1 PTX_TRIATT_PROCUDA=1` (the sampler attention levers dit_attn / dit_attn_fp16 / atom_attn: fpf_apb, cells measured on cc 9.0 with triton 3.7 — another triton engages and is named) | PTX_MK_PF=F1 (END only) is a TIER-2 OPT-IN here (two-pass LN, 1 bf16-ulp class; honours PTX_T_MIN_TOKENS) | welford (EXACT) arithmetic does not compile in bounded time on triton 3.3.1 |
| 9.0 \| 3.7 (H100: torch 2.13 / triton 3.7.1) | as above + `PTX_MK_PF=F1` (F1 start+end incl. the PAD8 padded producer, EXACT) | as above + `PTX_MK_PF=F1` | F1 both nodes EXACT; graph replay of F1 needs the stream-correct LayerNorm, which fpf_stackgraph enforces before arming | by_triton['3.7'] pure-config cells also apply here |
| 9.0 \| * (H100, a triton not listed) | `ARM=E; PTX_E_PAD8=1 PTX_GLUE_V2=1 PTX_E_PAD8_MIN_TOKENS=512 PTX_DIT_ATTN_EXACT=1 PTX_TRIATT_EXACT=1 PTX_ATOM_ATTN_EXACT=1 PTX_MK_PF=F1 PTX_TRIATT_PROCUDA=1` (the 9.0 / 3.7 composition; each kernel keeps its own cc-and-triton cell gate and refuses by name without cells) | `PTX_T_ATT=fast; ARM=T; PTX_GLUE_V2=1 PTX_MK_PF=F1 PTX_DIT_ATTN=1 PTX_DIT_ATTN_FP16=1 PTX_ATOM_ATTN=1 PTX_PF_ATTN=1 PTX_OPM_FUSED=1 PTX_PWA_FUSED=1 PTX_COND_DEDUPE=1 PTX_DIT_FAST=1 PTX_DIT_LOWP=fp16 PTX_ATOM_FAST=1 PTX_TRIATT_PROCUDA=1` | as its cells allow | the broad row: an exact key above is an override, never the only way |
| 8.0 \| 3.7 (A100: torch 2.13 / triton 3.7.1) | `ARM=E; PTX_GLUE_V2=1 PTX_MK_PF=F1 PTX_TRIATT_EXACT=1` (triatt_exact: the pair stacks' triangle attention through opt_core.kernels.triattn by the tier word exact — the provider's bit-identical row where its cell table vouches for this card's stack, else the library op by name; the transitions through opt_core.kernels.transition and the TriMul through opt_core.kernels.trimul by the tier word, the BLK2 block path on its 8.0 cells; the block core's fused tri-attention statement engages from CELLS.json blk2_triatt_min_tokens (400 tokens) up — MK-PF F1 prologue, the stock attention kernel, GLUE_V2 epilogue_v3, on the shared core's 8.0 cells — and steps aside by name to the stock statement below the floor (one `[FPF] BLK: fused tri-attention statement -> stock ...` line, tally blk_triatt_below_tokens); PAD8 is not set on this card) | `PTX_T_ATT=fast; ARM=T; PTX_GLUE_V2=1 PTX_MK_PF=F1 PTX_DIT_ATTN=1 PTX_DIT_ATTN_FP16=1 PTX_ATOM_ATTN=1` (the TriMul by the tier word on the provider's 8.0 cells; the shared core's tri-attention provider by the tier word fast (lever triattn_native) is the attention core of the fused statement from the floor up where it has a cell for this card, K2B otherwise, and the stock kernel below the floor; the sampler attention levers dit_attn / dit_attn_fp16 / atom_attn on the attention-with-pair-bias provider's 8.0 cells) | F1 both nodes EXACT from the floor up (this unit's 8.0 cell, welford arithmetic) | the stock fast-LayerNorm extension must carry sm_80 code (kit STOCK.md 'Stack') |
| 8.0 \| * (A100, a triton not listed) | `ARM=E; PTX_GLUE_V2=1 PTX_MK_PF=F1 PTX_TRIATT_EXACT=1` (the 8.0 / 3.7 composition; each kernel keeps its own cc-and-triton cell gate: the MK-PF 8.0 cell applies and is named as untested on that triton, GLUE_V2 refuses by name without a tested triton) | `PTX_T_ATT=fast; ARM=T; PTX_GLUE_V2=1 PTX_MK_PF=F1 PTX_DIT_ATTN=1 PTX_DIT_ATTN_FP16=1 PTX_ATOM_ATTN=1` (the 8.0 / 3.7 compositions — another triton engages the same bindings) | as 8.0 / 3.7 | as 8.0 / 3.7 |
| 10.0 \| 3.7 (B200) | `ARM=E; PTX_GLUE_V2=1 PTX_MK_PF=F1,F3` (PAD8 is OFF on cc>=10 by design) | `ARM=T; PTX_GLUE_V2=1 PTX_MK_PF=F1,F3` | F1 both + F3 EXACT | T\* on cc 10.0 is Tier-2-within-T |
| 10.0 \| * (B200, a triton not listed) | `ARM=E; PTX_GLUE_V2=1 PTX_MK_PF=F1,F3` (the 10.0 / 3.7 composition) | `ARM=T; PTX_GLUE_V2=1 PTX_MK_PF=F1,F3` | as its cells allow | as 10.0 / 3.7 |
| 10.3 \| 3.7 (B300: torch 2.13 / triton 3.7.1) | `ARM=E` (the transitions through opt_core.kernels.transition by tier word and the BLK2 block path — fused tri-attention prologue / epilogue — serve their sm103 cells; PAD8 is OFF on cc>=10 by design; GLUE_V2 / MK-PF have no 10.3 cells: unset here, they refuse by name if set) | `ARM=T` (K2B on its package-default cells) | refuses (no cell) | as 10.0 / 3.7 |
| 10.3 \| * (B300, a triton not listed) | `ARM=E` (the 10.3 / 3.7 composition) | `ARM=T` | as 10.3 / 3.7 | as 10.3 / 3.7 |
| other (a cc not listed: L40S 8.9, sm_120, …) | ARM E defaults; the switches above refuse or fall back loudly (cells keyed cc\|triton) | ARM T defaults | refuses (no cell) | never set FPF_GLUE_V2_ALLOW_TRITON / FPF_MKPF_ALLOW_TRITON in an exact arm |

**torch 2.13 / cu130 note:** the stream-correct LayerNorm ships prebuilt for cu130 (`third_party/fastln_prebuilt_cu130`, selected by
env.sh from the installed torch); on a torch with no matching prebuilt, `fpf_stackgraph.ensure_stream_ln()` source-rebuilds it at the
first armed process (needs `nvcc`; once per TORCH_EXTENSIONS_DIR); with neither, graphs stay off (exact eager path) with a printed
reason.

## Limits

* cu13 cuEquivariance builds on cc 10.x: PAD8 is off there by design (the table above).
* Large N: the row-chunked block path (`PTX_BLK_CHUNKED=1`, default; threshold `PTX_FPF_CHUNK_TOK`) keeps ARM E active at any size the
  device holds; the stack-graph cache serves inputs up to `PTX_BLK_GRAPH_MAXTOK` tokens and is bounded by a memory budget (`PTX_BLK_GRAPH_MAX=auto`, `PTX_BLK_GRAPH_MEM_GB`);
  signatures beyond it run eager (exact, slower; counted).
* Worker start-up: persist the Triton JIT cache (`TRITON_CACHE_DIR`) and `TORCH_EXTENSIONS_DIR` across processes; a cold cache compiles every
  kernel again at the first armed forward.
