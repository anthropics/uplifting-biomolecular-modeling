import dataclasses

from typing_extensions import Self

# For mmCIF parsing
# See mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_exptl.method.html
CRYSTALLIZATION_METHODS = {
    "ELECTRON CRYSTALLOGRAPHY",
    "FIBER DIFFRACTION",
    "NEUTRON DIFFRACTION",
    "POWDER DIFFRACTION",
    "X-RAY DIFFRACTION",
}
NMR_METHODS = {
    "SOLUTION NMR",
    "SOLID-STATE NMR",
}
EM_METHODS = {
    "ELECTRON MICROSCOPY",
}
OTHER_METHODS = {
    "FLUORESCENCE TRANSFER",
    "INFRARED SPECTROSCOPY",
    "SOLUTION SCATTERING",
}
ALL_EXPERIMENT_METHODS = (
    CRYSTALLIZATION_METHODS | NMR_METHODS | EM_METHODS | OTHER_METHODS
)


@dataclasses.dataclass(slots=True, kw_only=True)
class JsonSerializable:
    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values."""
        data = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        return {k: v for k, v in data.items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Create ExperimentRecord from dictionary."""
        return cls(**data)


# TODO: do we need separate rcsb from experiment record?
@dataclasses.dataclass(slots=True, kw_only=True)
class ExperimentRecord(JsonSerializable):
    """Metadata record from RCSB PDB."""

    pdb_id: str
    release_date: str
    method: str
    resolution: float | None = None  # NMR: None

    @property
    def is_nmr_structure(self) -> bool:
        return self.method in NMR_METHODS

    @property
    def is_crystal_structure(self) -> bool:
        return self.method in CRYSTALLIZATION_METHODS

    @property
    def is_em_structure(self) -> bool:
        return self.method in EM_METHODS

    def __repr__(self) -> str:
        return (
            f"ExperimentRecord("
            f"pdb_id={self.pdb_id}, "
            f"method={self.method}, "
            f"resolution={self.resolution}, "
            ")"
        )


@dataclasses.dataclass(slots=True, kw_only=True)
class PredictionRecord(JsonSerializable):
    """Metadata record from structure prediction."""

    # TODO: add more fields if necessary
    model: str | None = None  # e.g., "AlphaFold2"
    plddt: float | None = None


@dataclasses.dataclass(slots=True, kw_only=True)
class Metadata(JsonSerializable):
    id: str  # User/Author-defined name
    num_residues: int
    label_asym_id: str | None = None  # starts from 1
    auth_asym_id: str | None = None
    entity_id: int | None = None  # starts from 1
    asym_id: int | None = None  # starts from 1
    sym_id: int | None = None  # starts from 1
    cluster_id: str | None = None
    cluster_size: int | None = None
    is_low_homology: bool = False
    exp: ExperimentRecord | None = None
    pred: PredictionRecord | None = None

    def to_dict(self) -> dict:
        data = {
            "id": self.id,
            "num_residues": self.num_residues,
        }
        # Add optional fields if they are not None
        for field in [
            "cluster_id",
            "cluster_size",
            "label_asym_id",
            "auth_asym_id",
            "entity_id",
            "asym_id",
            "sym_id",
        ]:
            if getattr(self, field) is not None:
                data[field] = getattr(self, field)
        if self.is_low_homology:
            data["is_low_homology"] = self.is_low_homology
        # Add nested records if they are not None
        for field in ["exp", "pred"]:
            if getattr(self, field) is not None:
                data[field] = getattr(self, field).to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        # Create nested records if they are present
        if data.get("exp", None):
            exp = ExperimentRecord.from_dict(data["exp"])
        else:
            exp = None
        if data.get("pred", None):
            pred = PredictionRecord.from_dict(data["pred"])
        else:
            pred = None

        return cls(
            id=data["id"],
            num_residues=data["num_residues"],
            label_asym_id=data.get("label_asym_id", None),
            auth_asym_id=data.get("auth_asym_id", None),
            entity_id=data.get("entity_id", None),
            asym_id=data.get("asym_id", None),
            sym_id=data.get("sym_id", None),
            cluster_id=data.get("cluster_id", None),
            cluster_size=data.get("cluster_size", None),
            is_low_homology=data.get("is_low_homology", False),
            exp=exp,
            pred=pred,
        )


@dataclasses.dataclass(slots=True, kw_only=True)
class InterfaceMetadata(JsonSerializable):
    chain_ids: tuple[int, int]  # (chain id of chain A, chain id of chain B)
    cluster_id: str | None = None
    cluster_size: int | None = None
    is_low_homology: bool = False

    def to_dict(self) -> dict:
        data: dict = {"chain_ids": self.chain_ids}
        if self.cluster_id is not None:
            data["cluster_id"] = self.cluster_id
        if self.cluster_size is not None:
            data["cluster_size"] = self.cluster_size
        if self.is_low_homology:
            data["is_low_homology"] = self.is_low_homology
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls(
            chain_ids=tuple(data["chain_ids"]),
            cluster_id=data.get("cluster_id", None),
            cluster_size=data.get("cluster_size", None),
            is_low_homology=data.get("is_low_homology", False),
        )


@dataclasses.dataclass(slots=True)
class MultimerMetadata(JsonSerializable):
    id: str  # User/Author-defined name
    chains: list[Metadata]  # List of metadata for each chain
    interfaces: list[InterfaceMetadata]  # List of metadata for each interface
    exp: ExperimentRecord | None = None  # Optional experimental record
    pred: PredictionRecord | None = None  # Optional prediction record

    @property
    def num_chains(self) -> int:
        return len(self.chains)

    @property
    def num_residues(self) -> int:
        return sum(chain.num_residues for chain in self.chains)

    def to_dict(self) -> dict:
        data = {
            "id": self.id,
            "num_chains": self.num_chains,
            "num_residues": self.num_residues,
            "chains": [chain.to_dict() for chain in self.chains],
            "interfaces": [iface.to_dict() for iface in self.interfaces],
        }
        # Add metadata fields
        for field in ["exp", "pred"]:
            if getattr(self, field) is not None:
                data[field] = getattr(self, field).to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        chains = [Metadata.from_dict(chain_data) for chain_data in data["chains"]]
        interfaces = [
            InterfaceMetadata.from_dict(iface_data) for iface_data in data["interfaces"]
        ]

        if data.get("exp", None):
            exp = ExperimentRecord.from_dict(data["exp"])
        else:
            exp = None
        if data.get("pred", None):
            pred = PredictionRecord.from_dict(data["pred"])
        else:
            pred = None

        return cls(
            id=data["id"],
            chains=chains,
            interfaces=interfaces,
            exp=exp,
            pred=pred,
        )
