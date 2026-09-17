from mosaic.structure_prediction import (
    StructurePredictionModel,
    TargetChain,
    StructurePrediction,
)

from mosaic.losses.structure_prediction import IPTMLoss

from mosaic.losses.boltz2 import (
    load_boltz2 as lb,
    load_features_and_structure_writer,
    set_binder_sequence,
    boltz2_trunk,
    boltz2_forward_from_trunk,
    Boltz2Loss,
    MultiSampleBoltz2Loss,
)

from pathlib import Path
from jaxtyping import Array, Float, PyTree
import equinox as eqx
import jax
import jax.numpy as jnp

import numpy as np
import gemmi

from tempfile import NamedTemporaryFile

def pad_atom_features(features: dict, pad_to: int):

    n_atoms = features["atom_pad_mask"].shape[-1]
    assert pad_to >= n_atoms

    def pad(v):
        pad_width = tuple((0, pad_to-n_atoms) if d == n_atoms else (0,0) for d in v.shape)
        return jnp.pad(v, pad_width)

    return jax.tree.map(pad, features)

def _prefix():
    return """version: 1
sequences:"""


def chain_yaml(chain_name: str, chain: TargetChain) -> str:
    raw = f"""  - {chain.polymer_type.lower()}:
        id: [{chain_name}]
        sequence: {chain.sequence}"""
    if not chain.use_msa:
        raw += """
        msa: empty"""

    return raw


def target_only_features(chains: list[TargetChain]):
    yaml = "\n".join(
        [_prefix()]
        + [
            chain_yaml(chain_id, c)
            for chain_id, c in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", chains)
        ]
    )

    tf, template_yaml = build_template_yaml("ABCDEFGHIJKLMNOPQRSTUVWXYZ", chains)
    if tf is not None:
        yaml += template_yaml

    features, writer = load_features_and_structure_writer(yaml)
    if tf is not None: # make sure we actually got a template
        assert np.sum(features["template_mask"]) > 0
    return (features, writer)



def build_template_yaml(chain_names: str, chains: list[TargetChain]):
    # boltz wants perfect .cifs :( 
    templates = {
        chain_id: c.template_chain
        for chain_id, c in zip(chain_names, chains)
        if c.template_chain != None
    }
    if len(templates) > 0:
        st = gemmi.Structure()
        model = gemmi.Model("0")
        entities = []

        for chain_id, chain in templates.items():
            chain.name = chain_id
            ent = gemmi.Entity(chain_id)
            ent.entity_type = gemmi.EntityType.Polymer
            ent.polymer_type = gemmi.PolymerType.PeptideL
            ent.subchains = [chain_id]
            ent.full_sequence = [r.name for r in chain]
            entities.append(ent)
            for r in chain:
                r.subchain = chain_id
            model.add_chain(chain)

        st.add_model(model)
        st.entities = gemmi.EntityList(entities)
        st.assign_subchains()
        st.setup_entities()
        st.ensure_entities()
        st.assign_label_seq_id()

        tf = NamedTemporaryFile(suffix=".cif")

        template_yaml = f"""
        
templates:
  - cif: {tf.name}
    chain_id: [{', '.join(k for k in templates)}]
    template_id: [{', '.join(k for k in templates)}]
"""
        
        st.setup_entities()
        doc = st.make_mmcif_document()
        doc.write_file(tf.name)
        return tf, template_yaml
    else:
        return None, None

def binder_features(binder_length, chains: list[TargetChain]):
    return target_only_features([TargetChain(sequence = "X"* binder_length, use_msa=False)] + chains)


class Boltz2(StructurePredictionModel):
    model: eqx.Module

    def __init__(self, cache_path: Path | None = None):
        self.model = lb(cache_path) if cache_path is not None else lb()

    @staticmethod
    def target_only_features(chains: list[TargetChain]):
        return target_only_features(chains)

    @staticmethod
    def binder_features(binder_length, chains: list[TargetChain]):
        return binder_features(binder_length, chains)

    def build_loss(self, *, loss, features, recycling_steps=1, sampling_steps=None):
        return Boltz2Loss(
            joltz2=self.model,
            features=features,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps if sampling_steps is not None else 25,
            loss=loss,
            deterministic=True,
        )

    def build_multisample_loss(self, *, loss, features, recycling_steps=1, num_samples: int = 4, sampling_steps=None, reduction=jnp.mean):
        return MultiSampleBoltz2Loss(
            joltz2=self.model,
            features=features,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps if sampling_steps is not None else 25,
            loss=loss,
            deterministic=True,
            num_samples=num_samples,
            reduction=reduction,
        )


    @eqx.filter_jit
    def model_output(
        self,
        *,
        PSSM: None | Float[Array, "N 20"] = None,
        features: PyTree,
        recycling_steps=1,
        sampling_steps=None,
        key,
    ):
        if PSSM is not None:
            features = set_binder_sequence(PSSM, features)

        initial_embedding, trunk_state = boltz2_trunk(
            self.model, features,
            recycling_steps=recycling_steps,
            deterministic=True,
            key=key,
        )
        return boltz2_forward_from_trunk(
            self.model, features, initial_embedding, trunk_state,
            num_sampling_steps=sampling_steps if sampling_steps is not None else 25,
            deterministic=True,
            key=key,
        )

    def predict(
        self,
        *,
        PSSM: None | Float[Array, "N 20"] = None,
        features: PyTree,
        writer: any,
        recycling_steps=1,
        sampling_steps=None,
        key,
    ):
        output = self.model_output(
            PSSM=PSSM,
            features=features,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            key=key,
        )
        seq = PSSM if PSSM is not None else jnp.zeros((0, 20))
        iptm = -IPTMLoss()(seq, output, key=jax.random.key(0))[0]
        return StructurePrediction(st=writer(output.structure_coordinates), plddt=output.plddt, pae=output.pae, iptm=iptm, model_output=output)
