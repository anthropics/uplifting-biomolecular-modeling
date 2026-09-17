# CALIBY_FAST_INFERENCE_KIT_v1 (README)

Two inference-performance levers for Caliby / SolubleCaliby sequence design (ProteinDesignLab/caliby @ 41d31560), each a whole-file
replacement of one upstream module behind an environment switch: the Potts sampler (`CALIBY_FAST_SAMPLER`, patch 0002) and the Potts
parameter path (`CALIBY_FAST_POTTS_PARAMS`, patch 0003). Same model, same weights, same inputs, same 500 DLMC sweeps, same seeds, same
outputs; with both switches unset (or 0) the patched files run the stock path byte-for-byte (the fast paths are reached only when
`CALIBY_FAST_*` > 0). See HOWTO.md (files, install, switch values) and NOTICE.md / licenses/ (licences).

DEFAULT configuration (Tier 1: stock's outputs): `CALIBY_FAST_SAMPLER=2 CALIBY_FAST_POTTS_PARAMS=1` (= "P1").
OPT-IN: `CALIBY_FAST_POTTS_PARAMS=2` (= "P2", sparse 48-neighbour J, no N x N densification): reordered accumulation over the same
coupling set — the same sequences as stock, Potts energies U with last-ulp float32 differences. A user who needs
stock's energies U to the last bit picks P1, or adds the add-on `opt/forward/xattempt_addon` (lever `CALIBY_X_SPARSE_EXACT`), which makes
P2's energies equal stock's; the
package's kit modes (`caliby_opt`, `run.sh design --mode fast|exact`) run both kits' levers through that add-on's rows.

This directory is the record of the two patches (`patches/0002`, `0003`) and their licences; the patched files themselves ship once, in the add-on
(`../xattempt_addon/fast/potts.py`, `atom_mpnn_denoiser.py` = upstream + 0002 / 0003 + the add-on's patches; the upstream originals in
`../xattempt_addon/stock/`), which is what the release package (`caliby_opt`) serves by import hook over the pinned installed tree. Nothing is copied
into site-packages by any package verb.
