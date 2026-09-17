# patches/upstream_diffs/ — diffs against the pinned upstream (ProteinDesignLab/caliby @ 41d31560c3c73d7980d94f40f3c852b90bfab5c0)

`patches/0002-potts-fast-sampler.patch` (caliby/model/seq_denoiser/denoisers/seq_design/potts.py, 2 hunks) and `patches/0003-potts-params-gpu-path.patch` (caliby/model/seq_denoiser/denoisers/atom_mpnn_denoiser.py, 4 hunks)
ARE the upstream diffs (`a/caliby/... b/caliby/...`, `patch -p1` from the repo root): both files fetched from GitHub at the pin are byte-identical to the originals carried in `../xattempt_addon/stock/`;
0002 and 0003 apply cleanly, and the results are the base of the shipped `../xattempt_addon/fast/potts.py` and `atom_mpnn_denoiser.py` (that add-on's
`patches/X001-….vs_kit0002.patch` and `X006-….vs_kit0003.patch` are exactly the diffs from these results to the shipped files).
Upstream-PR readiness: 0002/0003 are performance levers (host-sync-free / CUDA-graph Potts sweep; GPU-side parameter aggregation) behind env switches, default-off in upstream terms — proposable as features, not bug fixes.
