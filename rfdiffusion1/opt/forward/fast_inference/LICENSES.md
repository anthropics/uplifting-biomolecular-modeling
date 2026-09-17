# LICENSES.md — origin and third-party licence notes for the files shipped in this kit (`LICENSE`: the upstream RFdiffusion licence that covers the derivative files; attribution summary in `NOTICE`)

| path in kit | origin | licence |
|---|---|---|
| `drivers/rfd_bench.py`, `rfd_fastpath.py`, `rfd_fullgraph.py`, `rfd_prep.py`, `rfd_einsum.py`, `tests/*`, `*.md`, `*.json` | written for this kit. `rfd_fastpath.py`, `rfd_fullgraph.py` and `rfd_prep.py` contain transcriptions of control flow from `RoseTTAFoldModel.py`, `Track_module.py` and `inference/model_runners.py` (RFdiffusion, BSD 3-Clause) and are therefore derivative works | the transcribed portions: RFdiffusion's licence ("BSD License", Copyright (c) 2023 University of Washington; the full upstream LICENSE text is reproduced below) |
| `inputs_public/insulin_target.pdb`, `inputs_public/5TPN.pdb` | copied unmodified from RFdiffusion `examples/input_pdbs/` (upstream repository, BSD 3-Clause); `insulin_target.pdb` is a theoretical model (`EXPDTA THEORETICAL MODEL`), not a wwPDB entry; 5TPN coordinates originate from the Protein Data Bank (entry 5TPN; PDB data are available under CC0 1.0) — per file: `inputs_public/SOURCES.md` | as upstream |
| NVIDIA SE3Transformer (`env/SE3Transformer` in the RFdiffusion repo) | **not shipped, not patched** — used as installed from the upstream repository | MIT (Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES) |
| DGL, PyTorch, Triton, opt_einsum, e3nn, hydra | **not shipped** — installed from their package indexes (`environment/requirements.lock` of the kit) | Apache-2.0 / BSD-style / MIT / MIT / MIT / MIT respectively |
| RFdiffusion model weights (`Complex_base_ckpt.pt`) | **not shipped** — download per the upstream README | as upstream (the RFdiffusion BSD licence states it covers "both the source code and model"; check the upstream repository for the current terms) |

If you redistribute the derivative files, keep the upstream LICENSE text below and the attribution headers.

## RFdiffusion LICENSE (verbatim; applies to the derivative files)

BSD License

Copyright (c) 2023 University of Washington. Developed at the Institute for
Protein Design by Joseph Watson, David Juergens, Nathaniel Bennett, Brian Trippe
and Jason Yim. This copyright and license covers both the source code and model
weights referenced for download in the README file.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this
list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

Neither the name of the University of Washington nor the names of its
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE UNIVERSITY OF WASHINGTON AND CONTRIBUTORS “AS
IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY OF WASHINGTON OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
