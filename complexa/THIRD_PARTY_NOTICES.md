# Third-party notices — `complexa` kit

The kit's own files (`opt/`, `configs/`, `run.sh`, `environment/`, the three documents) are new add-on code and text except as listed
below. `stock/` redistributes the upstream release verbatim at its pin (`stock/proteina-complexa-916eaaed.tar.gz`, upstream's `git archive`
of the commit less one pre-compiled third-party executable, `src/proteinfoundation/result_analysis/sc`, that upstream's `README.md` states is
not distributed with the repository — STOCK.md 'Stock exceptions'; upstream's `LICENSE` and `licenses/` inside; byte copies of `LICENSE` and
`licenses/license_weights.txt` under `stock/src/`); no upstream file is modified. Verbatim licence texts that concern the kit's own files are
reproduced under `third_party_licenses/`, byte-for-byte from the upstream release at the pin; one further text there,
`OpenStructure.LGPL-3.0.txt`, accompanies a third-party file that upstream's archive carries without its licence (section below). No model weights are included in this kit or in
images built from `environment/`; `run.sh install --weights DIR` fetches them from their publisher under the NVIDIA Open Model License
Agreement (`stock/src/licenses/license_weights.txt`). Paths are relative to this directory.

| component | upstream | licence (as named upstream) | copyright line (exactly as printed upstream) | kit files concerned | verbatim licence text |
|---|---|---|---|---|---|
| Proteina-Complexa (Python distribution `proteinfoundation` 1.1.0) | https://github.com/NVIDIA-BioNeMo/Proteina-Complexa at commit `916eaaedce5b07c205efb6ef32370c01d366591e` | Apache License, Version 2.0 (upstream's `licenses/license_code.txt`; upstream's root `LICENSE` reads "This repository contains multiple components covered by different licenses. See the licenses/ directory for details."; upstream ships no NOTICE file) | `Copyright [2026] [NVIDIA]` (the notice line of upstream's `licenses/license_code.txt`) | `opt/complexa_opt/levers.py` restates, with the changes its docstrings and `CHANGES.md` describe, upstream method bodies: `_cpf_forward` restates `ConcatPairFeaturesFactory.forward` (upstream `src/proteinfoundation/nn/feature_factory/concat_pair_feature_factory.py`); `_pba_forward` restates `PairBiasAttention.forward` and `_attn_sdpa` replaces `PairBiasAttention._attn` (upstream `src/proteinfoundation/nn/modules/pair_bias_attn.py`, next row); lever `loop_desync` re-creates `ProductSpaceFlowMatcher.full_simulation`, `RDNFlowMatcher.simulation_step`, `vf_to_score` and `score_to_vf` (upstream `src/proteinfoundation/flow_matching/`) at run time from the installed upstream's own source, one statement inserted or rewritten each — the file carries only the single-statement anchors it matches; `_bin_and_one_hot_f32` is a new implementation with the signature of upstream's `feature_utils.bin_and_one_hot`. No other kit file restates upstream code. | `third_party_licenses/Proteina-Complexa.Apache-2.0.txt` (= upstream `licenses/license_code.txt`; also inside the archive under `stock/`) |
| protein-docking code carried inside Proteina-Complexa (upstream file `src/proteinfoundation/nn/modules/pair_bias_attn.py` keeps its own header) | https://github.com/MattMcPartlon/protein-docking, as named in upstream's `licenses/license_third_party.txt` ("protein-docking (https://github.com/MattMcPartlon/protein-docking), MIT license") | MIT License (the header of that upstream file, lines 1-21, and upstream's `licenses/license_third_party.txt`) | `Copyright (c) 2022 MattMcPartlon` | `opt/complexa_opt/levers.py`: `_pba_forward` and `_attn_sdpa` (previous row) restate / replace the two methods of `PairBiasAttention` defined in that file | `third_party_licenses/Proteina-Complexa.license_third_party.txt` (= upstream `licenses/license_third_party.txt`, section *protein-docking*: the MIT text with this copyright line); the same text heads the upstream file inside the archive under `stock/` |
| ColabDesign copy carried inside Proteina-Complexa (`community_models/colabdesign/`) | https://github.com/sokrypton/ColabDesign, as named in upstream's `licenses/license_third_party.txt` | named "Beer-ware license" by upstream's `licenses/license_third_party.txt` (which reproduces that notice) and "MIT License" by upstream's `community_models/README.md` ("Each package retains its original license: colabdesign: MIT License"); the directory itself carries no licence file inside the archive | as printed in upstream's `licenses/license_third_party.txt`, section *ColabDesign*: `"THE BEER-WARE LICENSE" (Revision 42)` notice signed `Sergey Ovchinnikov` | none of the kit's own files adapts or restates ColabDesign code; `environment/Dockerfile` and STOCK.md §Stack install that copy, unmodified, from the unpacked archive as upstream's second editable install (`environment/requirements.lock` names the same directory of the same commit) | `third_party_licenses/Proteina-Complexa.license_third_party.txt` (section *ColabDesign*) |

The other components upstream lists in `licenses/license_third_party.txt` (openfold — Apache 2.0; alphafold3-pytorch, ProteinMPNN,
LigandMPNN — MIT) and its dataset terms (`licenses/license_datasets.txt`, CC BY 4.0) travel inside the archive under `stock/` with
upstream's copies of their licences; no kit file adapts them. `opt/complexa_opt/_core_gate.py` and `opt/_build_backend.py` are byte copies of
this release's own shared-library templates (`../common/opt_core/kit_template/`), not third-party code.

## OpenStructure stereo-chemical table carried inside the archive (LGPL-3.0)

The archive member `community_models/openfold/resources/stereo_chemical_props.txt` (9,119 bytes, sha256 `24510899eeb49167cffedec8fa45363a4d08279c0c637a403b452f7d0ac09451`) is
OpenStructure's table of reference bond lengths and angles (`modules/mol/alg/src/stereo_chemical_props.txt` of
https://git.scicore.unibas.ch/schwede/openstructure), which OpenFold's and AlphaFold's `residue_constants.load_stereo_chemical_props` read;
upstream carries it inside its OpenFold copy without a licence note. OpenStructure is licensed under the GNU Lesser General Public License,
version 3 (LGPL-3.0); the licence text is reproduced as `third_party_licenses/OpenStructure.LGPL-3.0.txt` (the LGPL-3.0 supplements the GNU
General Public License version 3, https://www.gnu.org/licenses/gpl-3.0.txt). The file is carried unmodified, exactly as upstream released
it inside the archive; the same bytes ship in ColabFold's and AlphaFold's Python distributions. No file of this kit reads, copies or adapts it
(upstream's `src/proteinfoundation/datasets/transforms.py` can load it through the OpenFold copy for motif tasks the kit's passes do not run).

## Data files carried inside `stock/proteina-complexa-916eaaed.tar.gz`

The archive is upstream's repository at the pin, so it carries upstream's `assets/` and `community_models/` data files listed below,
unmodified. The kit's generation passes read the target structures under `assets/target_data/` (`DATA_PATH` in `configs/*.env`; the README's
example pass names `bindcraft_targets/PD-L1.pdb`); the other files are not read by any kit route. Upstream attaches its dataset terms,
`licenses/license_datasets.txt` (CC BY 4.0), to the data it releases; where a file reproduces or derives from records of the wwPDB archive,
those records are CC0 1.0 and are cited by entry identifier. Theoretical models are not wwPDB entries and are marked as such.

| Archive members | What they hold | Source and terms |
|---|---|---|
| `assets/target_data/alpha_proteo_targets/*` (19 files: `<id>_cropped.pdb`, `_cropped_fixed.pdb`, `_repacked.pdb`, one `.cif`, one editor backup `.pdb~` and one empty file, as released) | binder-design target structures for wwPDB entries 1BJ1, 1TNF, 1WWW, 2WH6, 3DI3, 4HSA, 4ZXB, 5O45, 5VLI and 6M0J, cropped to the target region and, in the `_fixed` / `_repacked` variants, completed or side-chain repacked by upstream | coordinates derived from the named wwPDB entries (CC0 1.0); upstream's processing under its dataset terms (CC BY 4.0) |
| `assets/target_data/bindcraft_targets/*.pdb` (14 files: BBF-14, BetV1, CD45, CLDN1, CbAgo NPIWI and PAZ crops, DerF21, DerF7, HER2 crop, IFNAR2, PD-L1, PD1, Sas6, sCas9 crop) | single-chain target models named after the design targets of the BindCraft study, renumbered from residue 1 with occupancy and B-factor columns zeroed and no header records: prepared models, not wwPDB entries as deposited (upstream does not record per file whether a model was cut from a deposited structure or predicted) | upstream's data release (CC BY 4.0); to the extent a model derives from a deposited structure, that structure is wwPDB archive data (CC0 1.0) |
| `assets/target_data/ame_input_structures/M*_<pdb id>*.pdb` (46 files) | catalytic-site motifs of a few residues each, named by a Mechanism and Catalytic Site Atlas entry (`M0024` ...) and the wwPDB entry the residues come from (`1nzy` ...); `_fixA` / `_fixC` / `_v3` are upstream's edited variants | residues excerpted from the named wwPDB entries (CC0 1.0); selection and edits upstream's (CC BY 4.0) |
| `assets/target_data/ligand_targets/{5SDV,7BKC,7C7M,7v11}_ligand_centered.pdb` and the marker file `MAKE_SURE_LIGAND_IS_CHAIN_A_FOR_COMPLEXA` | the small-molecule ligand records (HETATM) of wwPDB entries 5SDV, 7BKC, 7C7M and 7V11, re-centred by upstream | wwPDB archive data (CC0 1.0); upstream's processing (CC BY 4.0) |
| `assets/target_data/check_pdb.ipynb` | upstream's notebook that counts residues per chain of the target files | upstream (its code licence, `licenses/license_code.txt`) |
| `assets/data/pdb_multimer.csv`, `assets/data/plinder_valid_dataset.csv`, `assets/pipeline_figure.png` | Git LFS pointer files only (three lines of text each); the tables and the figure themselves are not in the archive | nothing to license beyond the pointer text |
| `community_models/LigandMPNN/training/{train,valid,test_metal,test_nucleotide,test_small_molecule}.json`, `community_models/LigandMPNN/inputs/*.json` | LigandMPNN's lists of wwPDB entry identifiers for its training and test splits, and its example residue-selection inputs (identifiers and settings only, no coordinates) | LigandMPNN (MIT, `community_models/LigandMPNN/LICENSE` in the archive) |
| `community_models/colabdesign/rf/blueprint.css`, `blueprint.js`, `community_models/colabdesign/mpnn/legacy/example.ipynb` | web assets and an example notebook of the ColabDesign copy (row above) | ColabDesign's licence notice as carried by upstream |

No model weights, sequence databases or alignments are inside the archive.
