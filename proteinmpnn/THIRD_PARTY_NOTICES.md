# Third-party notices — `proteinmpnn/`

This kit carries, adapts and restates code from one third-party component, listed below with the files concerned. No model
weights and no third-party structure files are included in this tree or in the container images built from it: `run.sh install
--weights DIR` fetches the upstream repository (code and weights) at the pinned commit into `DIR`. Per-directory notices remain
beside the code they describe (`opt/forward/mpnn_pdb_parser/NOTICE.md`, `opt/forward/mpnn_exact_worker/NOTICE`).

## ProteinMPNN

- Upstream: https://github.com/dauparas/ProteinMPNN at commit `8907e6671bfbfc92303b5f79c4b5e6ce47cdef57` (the pin in `stock/PINS.json`)
- Licence: MIT License
- Copyright line, exactly as printed in the upstream `LICENSE`: `Copyright (c) 2022 Justas Dauparas`
- Licence text in this tree: `third_party_licenses/ProteinMPNN.MIT.txt`, a byte-identical copy of the upstream `LICENSE` at that commit
  (the same text is also carried at `opt/forward/mpnn_pdb_parser/licenses/dauparas__ProteinMPNN__LICENSE`)

| Kit file | Relation to the upstream code |
|---|---|
| `stock/src/protein_mpnn_run.py`, `stock/src/protein_mpnn_utils.py`, `stock/src/helper_scripts/parse_multiple_chains.py` | unmodified copies of the upstream files of the same names at the pinned commit, carried for reading (the option table); `stock/` is never edited |
| `opt/forward/mpnn_pdb_parser/kit/fast_parse.py` | modified copy of upstream `helper_scripts/parse_multiple_chains.py` (each PDB file is read once; the output is identical); described in `opt/forward/mpnn_pdb_parser/NOTICE.md` |
| `opt/forward/mpnn_exact_worker/addon/mpnn_worker2.py` | re-implements the control flow of `ProteinMPNN.sample()` and `ProteinMPNN.forward()` from upstream `protein_mpnn_utils.py` for batched execution and imports the user's unmodified `protein_mpnn_utils` at run time; described in `opt/forward/mpnn_exact_worker/NOTICE` |
| `opt/proteinmpnn_opt/lowmem.py` | restates and rewrites in low-memory form these sites of upstream `protein_mpnn_utils.py`: `ProteinFeatures._dist`, `ProteinFeatures._get_rbf`, `ProteinFeatures.forward`, and the decoding-order mask of `ProteinMPNN.forward` / `ProteinMPNN.sample`; at run time its text is inlined into the staged copy of `mpnn_worker2.py` |

The copyright notice above and the MIT permission notice (full text in `third_party_licenses/ProteinMPNN.MIT.txt`) accompany these
copies and adaptations.

## Data files and binaries

None are carried: this kit ships no structure, sequence, alignment, weight or compiled file, in `stock/` or outside it (`stock/` holds the three
upstream Python files above, `PINS.json` and `check_pins.py`); the tests build their inputs in code. The ProteinMPNN weights arrive only with the
clone that `run.sh install --weights DIR` makes, under the MIT licence quoted above, and the one run-time compile (the `exact` mode's Triton
kernel) happens on the user's machine. Run-time dependencies (PyTorch, NumPy, Triton, Biopython and the rest of `environment/requirements-mpnn.lock`)
are installed from their package indexes under their own licences and are not redistributed here.
