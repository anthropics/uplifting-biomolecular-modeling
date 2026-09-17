# triattn_exact — triangle attention with results bitwise identical to the library op (the provider's `triattn_exact` row)

What it is: a CUDA triangle-attention forward (`q/k/v [B,N,H,S,D]`, pair bias `[B,1,H,S,S]`, key mask `[B,N,1,1,S]`) written independently in
this tree against the observable numerics of the cuEquivariance triangle-attention op (inputs in, outputs compared; nothing here describes or
depends on how the library computes them). Its results are bitwise identical to that op's for the supported shapes: every (library version,
ops build, card class, dtype, head dim, shape class, environment) cell that `CELLS.json` marks `proven`. Everything else is refused BY NAME
(`triattn_exact.Refused`, raised before any launch); the caller decides what serves a refused call — in this tree the provider serves it with
the kit's stock op, counted per reason.

Contract
- Import name `triattn_exact` (routed to this directory by `opt_core.kernels.route`); the one entry point is
  `triattn_exact.face.triangle_attention(q, k, v, bias, mask=None, scale=None, kv_lengths=None)` — the library's signature and semantics,
  5-D output. Forward only; inputs that require grad are refused by name. Opaque to `torch.compile` (one graph break, eager kernel).
- First call per (process, device, library version, route) self-checks a few small shapes against the live library and poisons the route
  for the process on any mismatch. The provider runs this at resolve time, outside any served call or captured region.
- Rows of `S <= CUEQ_TRIATTN_FALLBACK_THRESHOLD` (a documented library setting, default 100) are refused by name and the library itself
  serves them; no bitwise cell is claimed for that range.
- How the member (`cuda_mma`, sm_80 code path, runs on cc 8.0 and 9.0 parts) reaches the card: a cell that carries `binary_fingerprint`
  is served from exactly one of the certified PREBUILT binaries under `_prebuilt/blobs/` (loaded through the CUDA driver; no compiler in the
  image, no torch C++ ABI; the toolkit build is chosen by the running driver); a cell certified on source (`route_fingerprint`) compiles
  `csrc/cuda_mma/triattn_v3.cu` with `nvcc` at first use (`$CUDA_HOME/bin/nvcc` or `nvcc` on PATH), cached under `$TRIATTN_EXACT_CACHE`,
  and refuses by name when the carried source no longer matches the cell.

Environment: `TRIATTN_EXACT_CELLS` (cell table; default this directory's), `TRIATTN_EXACT_CACHE` (build cache), `TRIATTN_EXACT_ARCHS`
(default `80;90`), `TRIATTN_EXACT_SKIP_SELFCHECK=1` (logged loudly; not for production), `TRIATTN_EXACT_LIB_VERSION` /
`TRIATTN_EXACT_OPS_BUILD` (override the detected library; not recommended), `TRIATTN_EXACT_ALLOW_JIT=1` (compile from source on a card
without a certified binary; development only), `TRIATTN_EXACT_IGNORE_FINGERPRINT=1` (development only).

Files: `__init__.py` (the typed refusal), `_paths.py` (layout), `face.py` (cell key, table match, binary / source fingerprint guards, self-check, the entry point),
`cuda_mma/__init__.py` (build, launch, capability check), `_prebuilt/` (driver-API launcher `cuda_mma.py`, `driver.py`, `manifest.json`,
`blobs/*.cubin.xz`), `csrc/cuda_mma/triattn_v3.cu`, `CELLS.json` (proven rows of the kernel route, consumed
keys only: `cells[*].{id,status,route,when,binary_fingerprint,route_fingerprint}`; `route_fingerprint` is stamped over this directory's
own bytes), `UPSTREAM.json` (checkpoint id, per-file sha256, carried routes, prebuilt builds, route fingerprints — the tests hold the
directory to it). Updating the package = replacing these files from a newer checkpoint and rewriting `UPSTREAM.json`; the provider's row
record and vouch keys live in `kernels/triattn/TRIATTN_CELLS.json`.
