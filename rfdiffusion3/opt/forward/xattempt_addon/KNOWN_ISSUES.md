# KNOWN_ISSUES — the overlay's own limits
1. **Memory of the roll-out cache.** With `RFD3_HOIST` the decoder blocks' masked bias — up to three `(D, 4, L, L)` bf16 tensors — stays alive for the duration
   of a denoiser call, and the hoisted pair bias for the duration of a roll-out (it grows with L²·D). With `RFD3_INIT_CHUNK` (both modes) the pass's peak
   is no longer the token initialiser's L_atom²-sized fp32 transient but the roll-out itself: cache + graph pool + the step's live tensors.
2. **The graph pool persists across a pass.** With `RFD3_CUDAGRAPH` the captured step's private memory pool lives until the process ends and the next
   design's initialiser runs beside it; `rfdiffusion3/configs/h100.env` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (placement only,
   no effect on numerics) so the two pools do not fragment each other.
3. **Default numerics are not bitwise run to run in upstream itself** (CUDA atomics in its scatter reductions): two upstream runs at one seed give different,
   equally valid designs. "Identical designs" for `RFD3_HOIST` / `RFD3_FZT` / `RFD3_CUDAGRAPH` is therefore stated under the deterministic recipe
   (`rfdiffusion3/CHANGES.md`, `design --det 1`), where upstream reproduces its own bits.
4. **Upstream's later `layer_utils.py`.** At upstream commit `faf4299` that file differs from the pin by one line (a static RMSNorm size for `torch.compile`);
   `install.sh` accepts either as stock and installs the pin's version of the file, so that later line is not carried.
5. **Low-memory / chunked-`P_LL` and sparse attention.** The pair-bias part of the cache engages only on upstream's full-`P_LL`, dense-attention path; upstream's
   `RFD3_LOW_MEMORY_MODE=1` / `RFD3_DENSE_SDPA_ATTENTION=0` select paths the overlay does not serve (the kit's modes refuse them by name; `--mode off` runs them).
6. **Roll-outs the graph sampler does not drive.** Classifier-free guidance (a second denoiser pass per step) and the chunked `P_LL` of the low-memory mode run
   upstream's own loop under `RFD3_CUDAGRAPH`, counted (`declined`, logged `[RFD3_CUDAGRAPH] declined by design`); the kit's modes refuse both by name before a run.
   The symmetry sampler keeps its own loop (the graph wraps the default sampler only); the kit's modes refuse it by name likewise. Realignment, origin jitter, the motif-fix schedule and partial diffusion are
   driven by the graph roll-out with upstream's statements. `allow_realignment=true` on a design with fixed motif atoms fails in upstream itself at this pin
   (`weighted_rigid_align`: `torch.linalg.svd` has no bf16 CUDA kernel, `NotImplementedError`), on every route.
7. While the overlay is installed into an editable (`pip install -e`) foundry checkout, `git status` there shows the two new files, the `*.hoist_orig` backups,
   the `.hoist_absent` markers and `rfd3/.xattempt_addon_overlay.json` as untracked and the eight edited files as modified; `bash install.sh --uninstall`
   restores a clean tree.
8. An install made by an `install.sh` older than the overlay marker is recognised as `previous:unmarked` by its sidecars and hunk tags; a hand-edited overlay
   file beside an intact pristine `.hoist_orig` reads the same way there (a marked install tells the two apart: the edited file is `foreign`).
