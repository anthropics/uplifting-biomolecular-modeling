# forward/pairfuse — the PAIRFUSE layer driver (fast tier)

`bz_pairfuse.py` drives every C=128 pair stack of Boltz-2 2.2.1 (trunk PairformerModule 64 blocks × 4 passes, the confidence
module's 8 blocks, the MSA module's PairformerNoSeqLayer per MSALayer) on ONE resident copy of the pair tensor z per stack call:
each sub-layer reads z once (LayerNorm inside the core's prologue) and writes it once (the residual added inside the core's
epilogue, in place), the ending-node triangle attention addressed by strides (no transposes), the sequence-attention pair bias
(LN + Linear 128→16 + rearrange) in one Triton pass over the resident z.  Cores (read only): `fpf_trimul_v4` (routed),
`opt_core.attn.pair_fused` (prologue v3 / epilogue v2 / transition v1 cells, `ln='fused'`), K2B flash triangle attention.

Adapter: `boltz2_opt/pairfuse.py` (attach name `pairfuse`, switch `BOLTZ_PAIRFUSE=bf16|fp32`, report key `pairfuse_report`).
Tests: `boltz2_opt/tests/test_pairfuse_gpu.py` (in-place == out-of-place bitwise per site; site error vs the stock statements;
pair-bias kernel vs LayerNorm+Linear; whole layer + RNG-offset equality).

## Why every in-place site is safe (out aliases the resident z) — and how it is checked

The driver writes only into the buffer it allocated at stack entry (never the caller's tensor: TemplateModule's `v = v + pairformer(v)`
aliases its input).  Within that buffer every provider call stores an output tile only after every read of the aliased input that the
tile depends on has completed:

| site | kernel structure | why `out is z` is safe |
|---|---|---|
| TriMul (`trimul_v4_forward(residual=True, out=z)`) | K1 (`_k1t`/`_k1c`) reads ALL of z and writes the a/b planes to the workspace; the batched GEMM reads only the planes; K3 (`_k3c`) per token tile loads the z tile once for the output gate's LN_in (registers), then per channel block loads the residual slice and stores the same slice | K1 and the GEMM complete before K3 launches (stream order); inside K3 each program touches only its own tile, loads precede the store of the same addresses in program order, `num_stages=1` (no software pipelining across channel blocks), no other program reads that tile |
| triangle attention start (`tri_attn_block(residual=True)`, prologue v3 → K2B/cueq core → epilogue v2) | the prologue reads z (LN in-kernel) and writes q/k/v/g/bias to workspaces; the core reads only those; the epilogue per row tile reads o, g, the residual tile of z and stores that tile | three kernels in stream order; the epilogue is tile-local (each program reads and writes only its own [rows, C] tile, read before write) |
| triangle attention end (the same with `ending=True`) | the prologue reads z^T BY STRIDES (no copy); the epilogue scatters tile (i,j) of the update to z[j,i] and reads the residual at z[j,i] | the residual read and the store address the SAME element z[j,i] inside one program; no program reads an element another program stores (the epilogue reads z only at its own store addresses; every read of z proper happened in the prologue, two kernels earlier) |
| transition (`transition(residual=True, out=z)`) | one kernel; each program owns BM rows: loads x rows (LN), loops the hidden dimension on chip, loads the residual rows (same addresses), stores the rows | row-local: a program's loads and its store cover the same rows and nothing else; loads precede the store in program order |
| pair bias (`pair_bias`) | reads z, writes a separate [H,N,N] fp32 tensor | no aliasing |

The checks in `boltz2_opt/tests/test_pairfuse_gpu.py`: in place == out of place BITWISE for the two sites whose provider has an out-of-place residual path
(trimul K3 `out=`, fused transition `out=`: `test_trimul_inplace_bitwise`, `test_transition_inplace_bitwise`); the tri-att epilogue's residual mode is in place
by construction in opt_core, so it is checked ELEMENTWISE to 2 bf16 ulps against bf16(z + u) with u from the epilogue's op mode, both nodes
(`test_triatt_inplace_within_rounding`); the whole driven layer bitwise run-to-run over 20 repetitions at N=384 and N=397 (not a multiple of any tile) in both
residencies (`test_repeated_runs_bitwise`) — a race in an in-place store or a nondeterministic reduction breaks it.

## Capture contract (graph levers)

After one eager served call of a shape class (module, N, residency, device) with a given pair-mask tensor, the served path issues no host
synchronisation and takes no data-dependent host decision (eligibility cached per module, providers probed, pair-bias modules installed,
the all-ones mask predicate cached per tensor object); per layer only fixed-geometry launches, allocator calls and four `torch.rand` draws
remain.  `mask_is_trivial(pair_mask)` decides the predicate once per tensor (one sync); `with mask_predicate(True|False):` lets a caller
that replays captured graphs on static buffers state it from the host side (no sync at all — and it must key its graph on the value);
`primed(module, z, pair_mask)` answers, host-only, whether a call is sync-free before it is made; `STATS['host_syncs']` counts every
sync the driver issues.  STATS count body executions: a body captured once and replayed by a graph lever is counted once here and N times
in that lever's census; the fail-closed check never compares its counts with the number of calls the model made.  GPU test: `test_no_host_sync_after_first_call`
(torch's sync debug mode raises on any synchronizing op during the second call).
