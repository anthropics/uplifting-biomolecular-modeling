"""The `hostfeat` lever — this kit's binding of `cells/of3_hostfeat.py` (exact class, in bytes): the a3m deletion matrix + aligned letter matrix
of the engine's MSA reader (`openfold3.core.data.io.sequence.msa.parse_a3m`: a per-character Python loop over every MSA row, then translate + a
row-by-row '<U1' fill) computed from the byte view of the rows, then OpenFold3 0.5.0's own tail (`MsaArray.from_parsed` + `truncate(…, inplace=True)`)
— integer / string arrays only, every feature identical (the kit tests against the engine's text); item 1's featurisation, which every process's
first forward waits for, is shorter. Part `msaidx` (the MSA letter -> index table) is not bound on this engine: OpenFold3
0.5.0 already ships the byte-view table (`residues._get_residue_idx_lut`) — n/a: upstream. Switch `OPENFOLD3_OB0_OPT_HOSTFEAT=1` (every kit line but
`off` exports it); knob `OPENFOLD3_OB0_OPT_HOSTFEAT_PARTS=a3m`. Installed by the package's activation in every active process (stack.ACTIVATION_MODULES;
registry kit `package`) before the engine imports its data pipeline; the DataLoader workers fork from the patched process. Marker
`[openfold3_ob0-opt/hostfeat] installed`; exit census `[openfold3_ob0-opt/hostfeat] LEVER name=hostfeat state=on|refused|off parts=a3m a3m_calls=<n>
a3m_rows=<n> a3m_fallback=<n> msaidx_calls=0 msaidx_cells=0 msaidx_fallback=0 aside=<part:reason|none>`."""
from .cells import of3_hostfeat as _core

ENV = "OPENFOLD3_OB0_OPT_HOSTFEAT"
ENV_PARTS = "OPENFOLD3_OB0_OPT_HOSTFEAT_PARTS"
TARGET = "openfold3.core.data.io.sequence.msa"

_core.configure(ENV=ENV, ENV_PARTS=ENV_PARTS, PREFIX="[openfold3_ob0-opt/hostfeat]", PARTS=("a3m",),
                M_MSA_IO=TARGET, M_RESIDUES="openfold3.core.data.resources.residues",
                REBIND={"a3m": ("openfold3.core.data.tools.colabfold_msa_server",), "msaidx": ()},
                DIGESTS={"parse_a3m": {"1d613d6180b100f5": "v050"},                      # OpenFold3 0.5.0 core/data/io/sequence/msa.py parse_a3m
                         "map_str_array_to_idx_array": {}})                              # not bound (upstream 0.5.0 ships the table)

STATE = _core.STATE
VALUES = _core.VALUES
KNOWN_PARTS = _core.KNOWN_PARTS
requested, parts, install, uninstall, census_line, serving = _core.requested, _core.parts, _core.install, _core.uninstall, _core.census_line, _core.serving
