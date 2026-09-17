# Third-party notices — `af2ig`

This kit wraps the `af2_initial_guess` step of dl_binder_design, which itself carries a copy of AlphaFold 2. The table lists every
third-party component that files of this kit adapt, restate, patch or ship, with the upstream location, the licence, the copyright line
exactly as printed upstream, the kit files concerned, and where the verbatim licence text sits (`third_party_licenses/` in this directory;
each text there is a byte-for-byte copy of the upstream file named in its row). Source files keep their original headers unedited; this
file carries the attribution and the statement of changes. The AlphaFold 2 model parameters are not part of this repository or of any image
built from it: `./run.sh install --weights DIR` fetches `params_model_1_ptm.npz` from DeepMind's `alphafold_params_2022-12-06.tar`, which
DeepMind publishes under CC BY 4.0 (https://github.com/google-deepmind/alphafold#model-parameters-license). Kernels of the shared library
(`common/opt_core`) that this kit calls carry their own notices in that tree.
The kit's own code is licensed under the Apache License 2.0 (LICENSE in this directory and at the top of the tree); this file concerns the third-party portions only.

Third-party code outside `stock/`. `run.sh install` unpacks `stock/dl_binder_design-cafa3853.tar.gz` to `./dl_binder_design` and applies, in
order, the patch series `opt/forward/af2ig_kit/patches/[0-9][0-9]_*.diff` listed in `stock/PINS.json` (`checkout.patches`). The directory
`opt/forward/af2ig_kit/patches/patched_files/af2_initial_guess/` carries the five files that series creates or modifies, exactly as they read
after it (`stock/check_pins.py --checkout` compares an installed tree against these copies): `predict_pdb.py` — created by patch `00` as an
adaptation of upstream's `af2_initial_guess/predict.py` (dl_binder_design, MIT) and extended by patches `05`–`16`; `af2_util.py` — upstream's file
(dl_binder_design, MIT) modified by patch `00`; `alphafold/data/mmcif_parsing.py` — AlphaFold 2 (Apache-2.0) modified by patch `00`;
`alphafold/model/config.py` and `alphafold/model/modules.py` — AlphaFold 2 (Apache-2.0) modified by patch `05`. No other patch of the series
touches an AlphaFold file, and no other file outside `stock/` and `third_party_licenses/` contains third-party code; `opt/af2ig_opt/levers/04_seed_switch.diff`
is a further unified diff against `predict_pdb.py` only.

| Component | Upstream | Licence | Copyright line (as printed upstream) | Kit files concerned | Verbatim licence text |
|---|---|---|---|---|---|
| dl_binder_design | https://github.com/nrbennet/dl_binder_design at commit `cafa3853ac94dceb1b908c8d9e6954d71749871a` | MIT | `Copyright (c) 2023 Nathaniel R. Bennett` | `stock/dl_binder_design-cafa3853.tar.gz` (the commit's files, unmodified); `opt/forward/af2ig_kit/patches/patched_files/af2_initial_guess/predict_pdb.py` (a PDB-directory front end adapted from upstream `af2_initial_guess/predict.py`); `opt/forward/af2ig_kit/patches/patched_files/af2_initial_guess/af2_util.py` (upstream file, modified); the patch series `opt/forward/af2ig_kit/patches/[0-9][0-9]_*.diff` and `opt/af2ig_opt/levers/04_seed_switch.diff` (unified diffs against the upstream checkout; they are the complete record of every change) | `third_party_licenses/dl_binder_design.MIT.txt` (= archive member `dl_binder_design/LICENSE`; also carried as `opt/forward/af2ig_kit/upstream/LICENSE.dl_binder_design.MIT.txt`) |
| AlphaFold 2 (vendored by dl_binder_design under `af2_initial_guess/alphafold/`) | https://github.com/google-deepmind/alphafold | Apache-2.0 | `Copyright 2021 DeepMind Technologies Limited` (per-file headers) | Modified upstream files, shipped as patched copies with their original headers: `opt/forward/af2ig_kit/patches/patched_files/af2_initial_guess/alphafold/model/config.py` and `.../alphafold/model/modules.py` (changed by patch `05_flash_attention.diff`: an optional flash-attention branch in `Attention.__call__` and its configuration keys), `.../alphafold/data/mmcif_parsing.py` (changed by patch `00_pdb_frontend_envport.diff`: import-time portability of the parsing module for the pinned stack). Statement of changes (Apache-2.0 §4(b)): these three files were modified by this repository; the exact modifications are the hunks of the two patches named. All other AlphaFold files are used unmodified from the archive. | `third_party_licenses/AlphaFold.Apache-2.0.txt` (= archive member `dl_binder_design/af2_initial_guess/LICENSE`, identical to the upstream repository's `LICENSE`; `opt/forward/af2ig_kit/upstream/LICENSE.alphafold.Apache-2.0.txt` carries the same terms behind a short attribution preamble) |
| silent_tools (shipped inside the dl_binder_design archive as `dl_binder_design/include/silent_tools/`, without its licence file) | https://github.com/bcov77/silent_tools | MIT | `Copyright (c) 2023 Brian Coventry` | none of this kit's own files; present only inside `stock/dl_binder_design-cafa3853.tar.gz` and the checkout unpacked from it (this kit's PDB front end does not use silent files) | `third_party_licenses/silent_tools.MIT.txt` (the upstream repository's `LICENSE`) |
| ProteinMPNN (the archive's `dl_binder_design/mpnn_fr/` wrapper is, per the dl_binder_design README, adapted in part from ProteinMPNN code; shipped without the ProteinMPNN licence file; no ProteinMPNN weights are included) | https://github.com/dauparas/ProteinMPNN | MIT | `Copyright (c) 2022 Justas Dauparas` | none of this kit's own files; present only inside `stock/dl_binder_design-cafa3853.tar.gz` and the checkout unpacked from it (not run by this kit) | `third_party_licenses/ProteinMPNN.MIT.txt` (the upstream repository's `LICENSE`) |
| RCSB PDB entries 1BRS, 5JDS, 6M0J, 1YY9 (test inputs) | https://www.rcsb.org | wwPDB/RCSB usage policy (CC0) | — | `opt/forward/af2ig_kit/tests/inputs/pdbs/*.pdb` (processed coordinate files; provenance in `opt/forward/af2ig_kit/tests/inputs/PUBLIC_INPUTS_README.md`) | policy: https://www.rcsb.org/pages/usage-policy |
| OpenStructure stereo-chemical parameter table | https://git.scicore.unibas.ch/schwede/openstructure (`modules/mol/alg/src/stereo_chemical_props.txt`), as committed in AlphaFold 2 and carried from there by dl_binder_design | LGPL-3.0 (GNU Lesser General Public License, version 3) | none printed in the file (a data table; OpenStructure's licence names the project) | inside `stock/dl_binder_design-cafa3853.tar.gz` only: member `dl_binder_design/af2_initial_guess/alphafold/common/stereo_chemical_props.txt`, unmodified (sha256 `24510899eeb49167cffedec8fa45363a4d08279c0c637a403b452f7d0ac09451`, the bytes of that OpenStructure file at commit `7102c63615b64735c4941278d92b554ec94415f8`); installed with the checkout and read by AlphaFold 2's `alphafold/common/residue_constants.py` at run time; no file of this kit outside `stock/` copies or adapts it | `third_party_licenses/OpenStructure.LGPL-3.0.txt` (the LGPL-3.0 text; the archive itself carries no copy) |

Data files. Outside `stock/`: `opt/forward/af2ig_kit/tests/inputs/pdbs/*.pdb` (six files) are processed coordinates of the wwPDB entries 1BRS,
5JDS, 6M0J and 1YY9 (experimental structures; wwPDB archive data, CC0 1.0 / RCSB usage policy), with the processing and the per-file entry
table in `opt/forward/af2ig_kit/tests/inputs/PUBLIC_INPUTS_README.md` and `index.csv` beside them. Inside the carried archive, as committed
upstream and unmodified: `af2_initial_guess/alphafold/common/testdata/2rbg.pdb` (wwPDB entry 2RBG, CC0) and
`af2_initial_guess/alphafold/relax/testdata/*.pdb` (test structures of the AlphaFold 2 repository, Apache-2.0), `examples/inputs/`
(dl_binder_design's example design models and Rosetta silent files, part of the MIT-licensed repository), `include/silent_tools/in.silent`
(silent_tools example, MIT) and the 85 `.pyc` files upstream committed beside its sources (byte-compiled copies of those same MIT / Apache-2.0
files). The one copyleft-licensed item anywhere in the kit is the OpenStructure table of the last table row (LGPL-3.0, a data file carried
unmodified inside the archive). The tree carries no model parameters, no sequence databases and no multiple-sequence alignments; the front end runs AlphaFold 2 in
single-sequence mode with the target structure as template, so no alignment data is read or shipped.

The older per-directory notice list `opt/forward/af2ig_kit/LICENSE` stays in place and points here.
