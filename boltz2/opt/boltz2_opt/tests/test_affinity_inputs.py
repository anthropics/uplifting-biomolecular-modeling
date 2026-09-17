"""worker.affinity_inputs: which inputs declare upstream's affinity property (`properties: - affinity: {binder: <chain>}`) — their units are
complete only with affinity_<name>.json, which the worker's run of upstream's affinity leg writes (test_affinity_leg.py)."""
import os

from .. import worker

AFFINITY_YAML = """version: 1
sequences:
- protein:
    id: A
    sequence: NLYIQWLKDGGPSSGRPPPS
    msa: empty
- ligand:
    id: LIG1
    smiles: CC(=O)Oc1ccccc1C(=O)O
properties:
- affinity:
    binder: LIG1
"""


def _write(d, name, text):
    p = os.path.join(str(d), f"{name}.yaml"); open(p, "w").write(text); return p


def test_affinity_inputs_names_exactly_the_inputs_declaring_the_property(tmp_path):
    aff = _write(tmp_path, "withaff", AFFINITY_YAML)
    lig = _write(tmp_path, "ligonly", AFFINITY_YAML.split("properties:")[0])
    odd = _write(tmp_path, "odd", "version: 1\nproperties: notalist\nsequences: []\n")
    bad = _write(tmp_path, "bad", "version: 1\nsequences: [\n")           # unparseable: upstream's to name (Failed to process … Skipping), not this check's
    items = worker.items_from_yamls([aff, lig, odd, bad])
    assert worker.affinity_inputs(items) == ["withaff"]
