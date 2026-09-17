# RFD3_XATTEMPT_ADDON — the ten-file overlay on `rfd3/model` that the kit's modes switch on

**What it is.** Ten full files for the installed `rfd3` package of RosettaCommons foundry at commit `4010e3e2`: eight of upstream's own files with small guarded
hunks (`inference_sampler.py`, `RFD3_diffusion_module.py`, `layers/{attention,block_utils,blocks,encoders,layer_utils,pairformer_layers}.py`) and two new
modules (`hoist.py`, `cudagraph_sampler.py`). Every added branch is keyed on one of the switches below, read once when the module is imported or the sampler is
constructed; with every switch unset the files take upstream's code path statement for statement. The package `rfdiffusion3_opt` exports a mode's switches
before `import rfd3.model` (`rfdiffusion3/CHANGES.md` is the mode table); nothing here is meant to be set by hand.

| switch | file(s) | what it does |
|---|---|---|
| `RFD3_HOIST` | `hoist.py`; call sites in `layers/attention.py`, `RFD3_diffusion_module.py`, `inference_sampler.py` | a per-roll-out cache of the tensors a denoiser call recomputes although their inputs are the roll-out's constants: the atom pair bias `to_b(P_LL)` of the 3 atom-encoder and 3 atom-decoder blocks and `downcast_c(C_L, S_I)` once per roll-out; the dense `(D, L, L)` key mask of `dense_sdpa_pairbias_attention` once per denoiser call; the masked bias written in one `torch.where(…, out=)` pass; the decoder blocks' masked bias kept across the call's two recycles. Same values, same layouts as upstream's statements (`docs/INVARIANCE_NOTES.md`); the cache is opened at the start of a roll-out and dropped in a `finally` at its end |
| `RFD3_FZT` | `layers/layer_utils.py` | `Transition` served by the shared core's fused SwiGLU kernel (`opt_core.attn.pair_fused.transition`, impl fpf): the module's own RMSNorm output is the kernel's input, upstream's bf16 rounding points are kept, the hidden activation is never written to memory; served at the row counts of the running card's range (`FZT_ROWS_BY_CC`: every row count on compute capability 9.0; floors and a 1048544-row ceiling on 8.0, outside which upstream's statement runs by name, counted `rows<MIN` / `rows>MAX`); a cell the core's table does not carry for the GPU falls back to upstream's statement by name |
| `RFD3_CUDAGRAPH` | `cudagraph_sampler.py` (installed at the sampler's construction in `inference_sampler.py`); `graph_mode()` hunks in `layers/*.py` | the default branch of `sample_diffusion_like_af3` with the per-step denoiser call replaced by the replay of one CUDA graph captured per (diffusion batch, design signature): the Gaussian draws taken in upstream's order at upstream's points, the motif index-put through integer indices, `gamma` from a host copy of the schedule, the sequence-entropy read-back moved after the loop, the duplicate-index assertion checked once after the roll-out; the origin jitter, the per-step realignment (with the motif-fix schedule) and the partial-diffusion schedule run from upstream's own statements; the two roll-out kinds the graph does not drive (classifier-free guidance, chunked `P_LL`) run upstream's loop, counted and logged; the symmetry sampler keeps its own loop (the kit's modes refuse it by name) |
| `RFD3_COMPILE` | `hoist.py` (`compile_install`, once per process) | `torch.compile` (inductor) of upstream's four compile targets — `encoder`, `diffusion_token_encoder`, `diffusion_transformer`, `decoder` — on the module the sampler runs, inside the captured step; the cache accessors, the graph sampler's memo helpers, `use_dense_sdpa_pairbias`, `create_attention_indices` and the fused transition are dynamo boundaries. Inductor's summation order, not eager's |
| `RFD3_TOKEN_SDPA` | `layers/attention.py` | the token transformer's pair-bias attention through upstream's own `dense_sdpa_pairbias_attention` (scaled-dot-product attention over the same keys and the same `-inf` mask) instead of the einsum path. SDPA's summation order |

`RFD3_GATHER_ATTN` (the atom index-set attention through the shared core's gather kernel) is installed by the package at the engine's initialisation, not by
these files. `rfdiffusion3/CHANGES.md` says which switches each mode sets and what each mode guarantees; `KNOWN_ISSUES.md` lists the overlay's own limits.

**Install state.** `install.sh` (run by `python -m rfdiffusion3_opt.install apply|status|classify|rebind|check-only|uninstall`) classifies each of the ten target
files in the active interpreter's `site-packages/rfd3` by sha256 — `patched` (equal to `patched/`), `stock` (upstream at the pin, `STOCK_SHA256_upstream_4010e3e.txt`),
`stock-head` (upstream's later `layer_utils.py`, `STOCK_SHA256_upstream_faf4299_extra.txt`), `absent-ok` (a new file not yet present) or `unknown` —, backs each
original up once as `<file>.hoist_orig` (an empty `<file>.hoist_absent` for a new file), copies `patched/` over, checks the result, and records the install in
`site-packages/rfd3/.xattempt_addon_overlay.json` (kit name, version, overlay id — a digest of the installed `patched/` bytes —, each file's sha256);
`--uninstall` restores the backups and drops that marker. `--classify` names the whole tree's state on one line, `OVERLAY_PRESTATE: pristine | current |
previous:<id> | foreign:<files>`: `previous` is an earlier install of THIS overlay, identified by that marker (`<id>` = `<version>/<overlay id, 12 hex>`) or,
for an install older than the marker, by the sidecars plus the `[RFD3_` hunk tags (`<id>` = `unmarked`; `partial` for a mix of `patched/` and pristine files).
`--rebind` prints that line and re-bases a pristine or previous state on this tree's overlay (`OVERLAY_APPLIED: 10 files id=<version>/<overlay id>`; the
pristine `.hoist_orig` backups are kept), leaves `current` alone, and exits 4 without touching a `foreign` file; plain `apply` takes pristine / current only
(exit 4 otherwise, unless `--force`: the manual escape that backs unknown files up and installs over them). The kit installs it into its second interpreter only; the pristine interpreter that runs `--mode off` never sees these files.

Layout: `patched/rfd3/model/…` (the ten files) · `install.sh` · `MANIFEST.json` (the file listing the package checks the tree against) · `STOCK_SHA256_*.txt` ·
`KNOWN_ISSUES.md` · `docs/INVARIANCE_NOTES.md` · `LICENSE`, `NOTICE`, `LICENSE_foundry_BSD-3-Clause.md` (BSD-3-Clause, as foundry).
