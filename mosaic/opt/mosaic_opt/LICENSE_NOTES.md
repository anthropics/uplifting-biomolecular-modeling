# LICENSE_NOTES — per component (see NOTICE.md for the table and licenses/ for verbatim texts)

- **mosaic** @70fec525: MIT, from the repository LICENSE file at that commit (GitHub licence API: spdx MIT). The upstream archive kept under
  stock/ (and the package pip installs from the same commit) also carries third-party material under its own terms, each with upstream's
  notice beside it: AlphaFold code (Apache-2.0, DeepMind), ProteinMPNN weights and model code (MIT, Dauparas), AbMPNN weights (CC BY 4.0,
  Zenodo record 8164693, re-serialised by upstream), a UniRef50-derived data table (CC BY 4.0, UniProt), a stability-model data notice (CC BY
  4.0 dataset / MIT split; no model file), esm2quinox code portions (Apache-2.0) and a simplex-projection routine whose source header reads
  BSD 3 clause (used by `simplex_APGM`). Of the weight and data files, only the ProteinMPNN `v_48_020` weights are read on this kit's path.
- **joltz** @ed0f0425: MIT (repository LICENSE; GitHub licence API: MIT).
- **boltz** (escalante-bio fork @1acc397b of jwohlwend/boltz): MIT (repository LICENSE; identical text to upstream boltz v2.2.1). Its source
  embeds third-party portions that keep their own terms: OpenFold-derived layers, a score_sde_pytorch class, a pdbeccdutils routine and an NVIDIA
  NeMo-derived callback (Apache-2.0); alphafold3-pytorch-derived modules, a ColabFold-derived MSA client and Bio-Diffusion code (MIT); PyTorch3D
  quaternion helpers (BSD-3-Clause) — member paths and printed copyright lines in the kit-level `THIRD_PARTY_NOTICES.md` §3.
- **Boltz-2 weights**: MIT per the Boltz README ("All the code and weights are provided under MIT license"); `boltz2_conf.ckpt` (digest-checked),
  `boltz2_aff.ckpt` (fetched by the same upstream call, never read by this kit, no digest recorded) and the CCD molecule library (`mols.tar`,
  digest-checked; boltz's processed form of the wwPDB Chemical Component Dictionary) are downloaded by
  the user at install / first run; none of them is in the release tree, the archives or the container recipes.
- **JAX / jaxlib / jax-cuda12-plugin 0.10.2**: Apache-2.0.
- **PyTorch 2.7.1 (CPU)**: BSD-3-Clause. **equinox**: Apache-2.0. **optax / dm-haiku / ml-collections**: Apache-2.0. **jaxtyping**: MIT. **esm2quinox**: Apache-2.0
  (its PyPI metadata and the header of mosaic's `losses/esm.py`). **gemmi**: MPL-2.0 (installed from PyPI, not carried here).
- **NVIDIA cu12 runtime wheels**: NVIDIA proprietary licence permitting redistribution of the runtime; installed from PyPI by the user, not vendored here.
- **PDB 1BRS**: wwPDB entries are CC0; only an 89-residue sequence string derived from it is inlined.
Commercial use: every component above permits it (MIT / BSD / Apache-2.0 / MPL-2.0 / CC BY 4.0 with attribution; NVIDIA runtime EULA). This note is a factual reading of the published
licence files, not legal advice.
