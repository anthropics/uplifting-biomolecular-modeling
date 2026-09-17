# trimul_native — triangle multiplication as arch-keyed CUDA cubins behind one provider face

The pair-stack triangle multiplicative update (outgoing and incoming), forward, for bf16 pair tensors and for fp32-resident pair
tensors under a bf16 autocast region, on compute capability 9.0 (sm_90a cubins) and 8.x (sm_80 cubins, binary-compatible with 8.6 / 8.7 /
8.9). The kernel family is parameterised over channel widths c_z / c_hidden in {64, 128, 256, 384} and token counts 16–4096 (ragged sizes
included); the widths, pairings, forms and variants a given build SERVES are exactly those recorded in `build/manifest.json` (`face.describe()`
/ `face.admits()` state them; anything else is refused by name, never substituted).

Memory envelope (analytic, per call at batch 1: the op's own tensors z + update plus the workspaces — two bf16 planes [2·c_hidden, N16, N16]
and the contraction result [c_hidden, N16, N16]; ≈ 10·c bytes·N² for bf16 z, ≈ 14·c bytes·N² for fp32-resident z at c_z = c_hidden = c):
c 128 bf16 reaches N 4096 in ≈ 20 GiB; c 256 bf16 needs ≈ 40 GiB at N 4096 (an 80 GB part; a 40 GB part tops out near N ≈ 3950, fp32-resident
near N ≈ 3350); c 384 bf16 tops out near N ≈ 3200 on 40 GB and needs ≈ 60 GiB at N 4096; c 64 fits N 4096 in ≈ 10 GiB. Workspaces are cached
per (N, width) in the caller's cache dict and count once per live shape.

```
z [B,N,N,c_z] = PRE-LayerNorm pair tensor, mask [B,N,N] 0/1 or None
x = LN_in(z); a = mask·sigmoid(x W_agᵀ)·(x W_apᵀ); b = mask·sigmoid(x W_bgᵀ)·(x W_bpᵀ)
X[i,j,:] = Σ_k a[i,k,:] b[j,k,:] (outgoing) | Σ_k a[k,i,:] b[k,j,:] (incoming)
update = sigmoid(x W_ogᵀ) · (LN_out(X) W_oᵀ);  serve() returns update, or z + update with residual=True
```

Structure: prologue kernel (LN_in + four gated projections + mask → channel-major bf16 planes) | tensor-core contraction (cuBLAS through
the framework) | epilogue kernel (LN_out + W_o + output gate (+ residual)). One kernel family; tiling and staging are parameters
(c_z, c_hidden, N, direction, form, architecture) read from a tile table, not separate code paths.

## Layout

```
python/trimul_native/      the package: face.py (serve / check / pack_weights / admits / Refusal), launch.py (module cache, argument
                           packing incl. by-value parameter structs and CUtensorMap, launches on the framework's current stream),
                           _driver.py (CUDA driver binding: ctypes over libcuda.so.1, or the cuda.bindings wheel — same bytes either way),
                           manifest.py (build record + sha256 verification), build.py (nvcc → cubins + manifest)
csrc/                      kernel sources; one .cu = one unit = one cubin per architecture (csrc/dev_probe/: the route probe, development only)
build/<arch>/<unit>.cubin  the built objects + <unit>.ptxas.txt (register / shared-memory report); build/manifest.json records cubin and
                           source digests, the nvcc release, flags, ELF facts (ABI generation, toolkit line) and per-kernel resources
testvectors/               closed-form test vectors: expected output digests per device class + witness tensors, replayed by face.check() before the first serve
tests/                     t_face_modes.py (execution modes, streams, CUDA graphs, host cost per serve) and the members' own checks
pkg/vN/                    sealed releases (see SEALING.md)
```

## Binary route

The kernels ship as cubins keyed by GPU architecture, not by framework or interpreter ABI: `cuModuleLoadData` into the framework's primary
context, `cuLaunchKernel` on `torch.cuda.current_stream()`, workspaces from the framework's caching allocator, tensor-map descriptors encoded
host-side with `cuTensorMapEncodeTiled`. No host synchronisation in a serve; CUDA-graph capture records the launches once the modules are
loaded (call `face.check()` or one eager serve before capturing). A cubin built by a CUDA 13.0 toolkit loads on drivers of the CUDA 13 line
(R580) and newer; `build/manifest.json` records the toolkit line of every cubin (`elf.toolkit_version`) so a deployment can compare it with
`cuDriverGetVersion` before loading, and a refused load is reported by name (`load:<driver error>`).

## Commands

```
python -m trimul_native.build --all && python -m trimul_native.vectors make   # rebuild every unit + regenerate the vectors at a new source state (nvcc image, sm_90a device)
python -m trimul_native.build --archs sm_90a,sm_80 --jobs 8   # writes build/<arch>/<unit>.cubin + build/manifest.json (incl. the served table decided by kernel.py / sm80_ops.py naming)
python -m trimul_native.build --list                          # the unit table (per-width sm_90a units, the sm_80 units, development units)
python -m trimul_native.vectors make                          # on a device of each class: testvectors/ (closed-form inputs, expected digests, witnesses)
python -m trimul_native.vectors replay --which gate|all       # the byte gate by hand
python -c "from trimul_native import face; print(face.check())"             # driver -> arch -> manifest-verified loads -> load check -> byte gate
python tests/t_face_modes.py                                  # serve under inference_mode / no_grad / grad / side stream / CUDA graph + host us per serve
python -m trimul_native.seal --version X.Y.Z --out pkg/vN --cells CELLS.json  # cut a sealed release (SEALING.md)
python csrc/dev_probe/probe_test.py --binding ctypes          # route probe: cp.async (sm_80+) and TMA (sm_90a) tile kernels, bytes vs torch, CUDA graph replay
```

`PYTHONPATH` must include `python/`. Environment: `TRIMUL_NATIVE_DRIVER=auto|ctypes|cuda_bindings` selects the driver binding (default auto:
the wheel when importable with the tensor-map API, else ctypes).

## Refusals (by name, before any launch; never a silent substitute)

`no_cubin:<arch>` · `driver_unavailable:<why>` · `cc_unsupported:<cc>` · `loadcheck_failed:<case>` · `manifest:<what>` · `load:<driver error>` ·
`variant:<what>` · `vectors:<what>` · `shape:` / `dtype:` / `direction:` / `weights:` envelope words · `not_built:<what>` (development trees only).

## Numerics

Default builds keep `--use_fast_math` off; sigmoid and LayerNorm are evaluated in fp32 with IEEE division and `expf`, products rounded to
bf16 once where the reference op rounds. The reference for the byte gate and the tolerance class is cuequivariance_torch 0.11.1
`triangle_multiplicative_update` (class bound: max-abs ≤ 1 bf16 ulp of the output range, rel-RMS ≤ 4.5e-4); an exact configuration
reproduces the reference bit for bit where the schedule allows it.

## Provider integration: measured cells and the uncovered-key policy

A release ships `CELLS.json` (package root; `TRIMUL_NATIVE_CELLS` overrides the path): the measured rows keyed
(cc, form bf16 | f32z, class fast | exact, c_z, c_hidden, direction, residual, N-bucket), optionally each with the `config` that won there
(`variant`, `incoming_mode`, `k1_cfg`, `k3_cfg`, `sm80_tiles`; absent = the kernel tile table's own default). On the first call of a key
`face.serve` resolves the cell (`face.resolve_cell(cc, form, c_z, c_hidden, direction, N, klass, residual)`, pure). "Nearest" is taken ONLY in
size: among rows of the same card (exact cc), form, numerics class (an exact request never inherits a tolerance-class row nor the reverse),
width pair and call form (direction, residual; a row recorded with `*` matches both), the N-bucket nearest in log distance is the cell:

1. N within ×1.5 of that bucket: covered — the row's configuration is served silently;
2. N farther than that: served with the nearest bucket's configuration, and the face emits the information token
   `UNCOVERED_CELL:<cc>|<form>|<class>|z<c_z>_h<c_hidden>|<direction>|res<0|1>|N<n>` followed by `nearest=<the row used>` (and, on sm_90a,
   `tiles=<the tile-table key the kernel used>`), once per key per process — on the `trimul_native` logger at INFO, to every sink registered
   with `face.add_notice_sink(fn)`, and in `face.notices()` / `face.describe()["notices"]` for a provider's status line; never an exception;
3. no row of that (cc, form, class, width pair, direction, residual) at all: refusal by name `no_cell:<key>` — nothing is inherited from
   another width, class, card or call form (an sm_90a cell is never served on cc 8.x nor the reverse; a width the build does not serve is
   the `face.admits` refusal as before);
4. no cells table at all (a development tree): served with the kernel tile table's defaults, notice `NO_CELLS_TABLE` once.

A caller's explicit `config=` overrides the cell's configuration key by key (tuning and tests); `config={"cells": False}` serves without
consulting the table (the byte gates — `face.check`, `python -m trimul_native.vectors` — are table-independent).
