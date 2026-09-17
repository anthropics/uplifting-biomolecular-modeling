# LICENSE_NOTES

| path | origin | licence |
|---|---|---|
| `patched/rf3/diffusion_samplers/inference_sampler.py`, `patched/rf3/model/RF3_structure.py`, `patched/rf3/model/layers/af3_diffusion_transformer.py`, `patched/rf3/loss/loss.py` | RosettaCommons **foundry** (`models/rf3/src/rf3/...`) at commit 4010e3e2e7350edada3e25a45c908c6bf407df4d, modified as marked in each file | BSD 3-Clause (foundry's, `LICENSE_foundry_BSD-3-Clause.md`); the modifications BSD 3-Clause (`LICENSE`) |
| `LICENSE`, `NOTICE`, `LICENSE_foundry_BSD-3-Clause.md` | licence of these files; attribution; verbatim copy of foundry's licence file | — |
| `public_inputs/1brs_tiles.json` | amino-acid sequences of PDB entry 1BRS (barnase / barstar), polymer entities 1 and 2, from the RCSB PDB Data API | PDB archive data are available under CC0 1.0 |

Model weights: none are included. The kit runs with the RF3 checkpoint distributed by the foundry project; the foundry repository does not
state separate licence terms for the weights beyond the repository licence — check the foundry project's current terms before redistributing them.
