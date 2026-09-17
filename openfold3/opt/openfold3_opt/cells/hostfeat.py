"""Cell `hostfeat` — this kit's binding of `of3_hostfeat` (exact class, in bytes): two host-side featurisation statements of the engine's data
pipeline re-stated in numpy — the a3m deletion matrix + aligned letter matrix (`openfold3.core.data.io.sequence.msa.parse_a3m`: a per-character
Python loop over every MSA row) and the MSA letter -> alphabet index mapping (`openfold3.core.data.resources.residues.map_str_array_to_idx_array`:
one full-array comparison per alphabet letter; a 256-entry table filled by the engine's own function instead) — so item 1's featurisation, which
the forward waits for in every process, is seconds shorter at typical input sizes, every feature identical (integer / string arrays, compared equal
by the kit tests). Switch `OPENFOLD3_OPT_HOSTFEAT=1` (the `exact`, `fast` and `big` lines export it); knob `OPENFOLD3_OPT_HOSTFEAT_PARTS=a3m,msaidx`
(default both; each alone). Installed by the package at activation (openfold3_opt.stack.activate, modes.ACTIVATION_CELLS): the data pipeline is
engine plumbing every line shares; the DataLoader workers fork from the patched process."""
from . import of3_hostfeat as _core

ENV = "OPENFOLD3_OPT_HOSTFEAT"
ENV_PARTS = "OPENFOLD3_OPT_HOSTFEAT_PARTS"

_core.configure(ENV=ENV, ENV_PARTS=ENV_PARTS, PREFIX="[openfold3-opt/hostfeat]", PARTS=("a3m", "msaidx"),
                M_MSA_IO="openfold3.core.data.io.sequence.msa", M_RESIDUES="openfold3.core.data.resources.residues",
                REBIND={"a3m": ("openfold3.core.data.tools.colabfold_msa_server",),
                        "msaidx": ("openfold3.core.data.primitives.featurization.msa", "openfold3.core.data.primitives.sequence.msa")},
                DIGESTS={"parse_a3m": {"32b3d1f5ca0f391c": "v041"},                      # OpenFold3 0.4.1 core/data/io/sequence/msa.py:96-141
                         "map_str_array_to_idx_array": {"e433a3331b908883": "v041"}})   # OpenFold3 0.4.1 core/data/resources/residues.py

STATE = _core.STATE
VALUES = _core.VALUES
KNOWN_PARTS = _core.KNOWN_PARTS
requested, parts, install, uninstall, census_line, serving = _core.requested, _core.parts, _core.install, _core.uninstall, _core.census_line, _core.serving
