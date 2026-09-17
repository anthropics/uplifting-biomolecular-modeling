"""Inputs: the target, the shape (target, binder length, target copies, the staged MSA) and the files of a shape under the cache root.

The driver's default target is the kit's public one — barstar, PDB 1BRS chain D SEQRES (89 aa), single-sequence (`msa: empty`) — inlined in
`tools/fetch_public_inputs.py` and read from there by ``public_target()``; `--target-fasta` names any other protein target (the FASTA's one
record; a multi-record file is refused by name unless `--first-record` takes record 1) and `--msa` its precomputed alignment (upstream's
`TargetChain(use_msa=True)` with upstream's MSA-server fetch done ahead of time; no server is ever contacted: tools/recipe.py refuses a chain that asks
for an MSA without a staged file); `--epitope 12,15,40-48` restricts the loss's binder→target contact term to those target residues (1-based
along the target sequence, every copy; tools/recipe.py `epitope`). Every one of them is resolved through the ONE recipe module. A shape is
(target, `--binder-length L`, `--target-copies N`, the MSA, the epitope) = target residues·N + L tokens (169 at L=80, N=1 on the public target;
the `tools/public_design_run.py` docstring), keyed `L<L>_c<N>` for the public target (the kit's own way: `features_L<L>_c<N>.npz`), `<target
id>[-msa<sha8>]_L<L>_c<N>` for a FASTA target (id = tools/recipe.py target_id), with `ep<sha8>` (sha256 of the epitope's canonical spelling)
joined in front of `L<L>` when an epitope is given: the epitope changes the traced loss, so the compiled executables (P1) are the epitope's own.
Under `$MOSAIC_OPT_CACHE_ROOT/<key>/` the shape keeps its frozen features (P3; sha256 beside them) and its P1 directory `xla_cache/`
(compilation cache + `xla_autotune_results.pb`).
"""
import hashlib
import os
import re
from dataclasses import dataclass
from typing import Optional

FETCH_RELPATH = os.path.join("tools", "fetch_public_inputs.py")
P1_DIRNAME = "xla_cache"                                        # one P1 directory per shape


@dataclass(frozen=True)
class Shape:
    """A design shape: no defaults here — the driver's own argparse defaults are the values used (shape_from_args). `target` is the
    recipe's target id (tools/recipe.py target_id: <FASTA record id>-<sha256(sequence)[:8]>) when a FASTA names the target, None for the
    public target; `target_fasta` / `msa` the files the driver's --target-fasta / --msa read; `first_record` / `fasta_records` the driver's
    --first-record and the FASTA's record count; `epitope` the canonical spelling of --epitope (tools/recipe.py epitope_spec) or None. The key
    names the target first, so two targets at one (L, copies) never share a features file or a P1 directory; a staged MSA is part of it
    (`-msa<sha8>`): the featurized inputs — and so the compiled executable — differ with and without the alignment; an epitope is part of it
    (`ep<sha8>`): the loss it restricts is another executable."""
    binder_length: int
    target_copies: int
    target: Optional[str] = None
    target_fasta: Optional[str] = None
    target_length: Optional[int] = None
    msa: Optional[str] = None
    msa_sha8: Optional[str] = None
    first_record: bool = False
    fasta_records: Optional[int] = None
    epitope: Optional[str] = None

    @property
    def epitope_sha8(self) -> Optional[str]:
        return hashlib.sha256(self.epitope.encode()).hexdigest()[:8] if self.epitope else None

    @property
    def key(self) -> str:
        base = f"L{self.binder_length}_c{self.target_copies}"
        head = []
        if self.target is not None:
            head.append(f"{self.target}{'-msa' + self.msa_sha8 if self.msa_sha8 else ''}")
        if self.epitope:
            head.append(f"ep{self.epitope_sha8}")
        return "_".join(head + [base])

    @property
    def features_name(self) -> str:
        return f"features_{self.key}.npz"

    def tokens(self, public_target_length: int) -> int:
        """Token count: target residues × copies + L — the shape's own target length when a FASTA names the target, else the public target's."""
        return (self.target_length if self.target_length is not None else public_target_length) * self.target_copies + self.binder_length

    def record(self) -> dict:
        """The shape as the package's manifests and reports carry it: {key, binder_length, target_copies, target, target_length, msa, first_record,
        fasta_records, epitope}."""
        return {"key": self.key, "binder_length": self.binder_length, "target_copies": self.target_copies, "target": self.target, "target_length": self.target_length, "msa": self.msa,
                "first_record": self.first_record, "fasta_records": self.fasta_records, "epitope": self.epitope}

    def flags(self) -> list:
        out = ["--binder-length", str(self.binder_length), "--target-copies", str(self.target_copies)]
        if self.target_fasta:
            out += ["--target-fasta", self.target_fasta]
        if self.first_record:
            out += ["--first-record"]
        if self.msa:
            out += ["--msa", self.msa]
        if self.epitope:
            out += ["--epitope", self.epitope]
        return out


def shape_from_args(binder_length: Optional[int], target_copies: Optional[int], driver_defaults: dict, target_fasta: Optional[str] = None,
                    msa: Optional[str] = None, kit_home: Optional[str] = None, first_record: bool = False, epitope: Optional[str] = None) -> Shape:
    """The shape from the command line, the driver's own defaults where unset (settings.driver_defaults). The target, its MSA, `--first-record` and
    `--epitope` are resolved through the kit's recipe module — the ONE parser the driver runs too (recipe.target / recipe.epitope: the target's id,
    the MSA file's digest, the epitope's canonical spelling) — and refused by name there: no record, not protein, more than one record without
    `--first-record` (`fasta_multi_record`), `--first-record` or `--msa` without a FASTA (the public target is one single-sequence chain), MSA file
    absent, an epitope position outside the target (`epitope_out_of_range`) or misspelled (`epitope_format`); the refusal reaches the caller as a
    ValueError carrying the recipe's words (the command line's usage exit), every other error propagates as itself."""
    L = int(binder_length) if binder_length is not None else int(driver_defaults["--binder-length"])
    N = int(target_copies) if target_copies is not None else int(driver_defaults["--target-copies"])
    if L < 1 or N < 1:
        raise ValueError(f"binder length and target copies must be >= 1 (got L={L}, N={N})")
    from . import recipe
    M = recipe.module(kit_home)
    try:
        T = M.target(fasta=target_fasta or None, copies=N, msa=msa, first_record=bool(first_record))  # the public target when no FASTA is given
        E = M.epitope(epitope, T) if epitope is not None else None
    except M.RecipeError as e:                                             # the recipe's refusal by name is a usage error here (its words unchanged); anything else the module raises is a failure, not usage
        raise ValueError(str(e)) from e
    return shape_of(T, L, N, E)


def shape_of(target_rec: dict, binder_length: int, target_copies: int, epitope_rec: Optional[dict] = None) -> Shape:
    """The Shape of a RESOLVED target record (tools/recipe.py target()) and epitope record (epitope(), or None): the one place a target becomes
    a shape — the command line (shape_from_args) and the package accessor (recipe.recipe) both come here, so the driver's input flags are
    composed once, by Shape.flags(). The public target keeps the kit's own key (`L<L>_c<N>`, no target field)."""
    ep = epitope_rec["spec"] if epitope_rec else None
    L, N = int(binder_length), int(target_copies)
    if target_rec["source"] == "public":
        return Shape(binder_length=L, target_copies=N, epitope=ep)
    return Shape(binder_length=L, target_copies=N, target=target_rec["id"], target_fasta=target_rec["fasta"], target_length=target_rec["length"], msa=target_rec["msa"],
                 msa_sha8=(target_rec["msa_sha256"][:8] if target_rec["msa_sha256"] else None), first_record=bool(target_rec["first_record"]),
                 fasta_records=target_rec["fasta_records"], epitope=ep)


def public_target(kit_home: str) -> dict:
    """The kit's inlined target: {"pdb", "chain", "sequence", "length", "sha256", "pdb_sha256"} read from tools/fetch_public_inputs.py
    (the constants `PDB_ID`, `BARSTAR_1BRS_D`, `PDB_SHA256`; `BARSTAR_SHA256` is the kit's sha256 of the sequence string, recomputed here)."""
    path = os.path.join(kit_home, FETCH_RELPATH)
    src = open(path, "r", encoding="utf-8").read()

    def const(name):
        m = re.search(rf'^{name}\s*=\s*["\']([^"\']*)["\']', src, re.M)
        if not m:
            raise ValueError(f"{name} not found in {path}")
        return m.group(1)
    seq = const("BARSTAR_1BRS_D")
    if not re.fullmatch(r"[A-Z]+", seq):
        raise ValueError(f"BARSTAR_1BRS_D in {path} is not a plain sequence: {seq[:20]!r}")
    if not re.search(r"^BARSTAR_SHA256\s*=\s*hashlib\.sha256\(BARSTAR_1BRS_D\.encode\(\)\)\.hexdigest\(\)", src, re.M):
        raise ValueError(f"BARSTAR_SHA256 in {path} is not sha256(BARSTAR_1BRS_D) as expected")
    return {"pdb": const("PDB_ID"), "chain": "D", "sequence": seq, "length": len(seq), "sha256": hashlib.sha256(seq.encode()).hexdigest(),
            "pdb_sha256": const("PDB_SHA256")}


def shape_dir(cache_root: str, shape: Shape) -> str:
    return os.path.join(cache_root, shape.key)


def features_path(cache_root: str, shape: Shape) -> str:
    return os.path.join(shape_dir(cache_root, shape), shape.features_name)


def p1_dir(cache_root: str, shape: Shape, mode: Optional[str] = None) -> str:
    """The shape's P1 directory under the cache root: `<root>/<shape>/xla_cache` — the pinned files `warm` populates (exact) — or, for a
    mode whose row carries P1 in its transparent form (fast, big), its own `<root>/<shape>/xla_cache_<mode>`: filled by the first process
    of the shape, never in the way of the pinned directory's populate-from-empty rule."""
    return os.path.join(shape_dir(cache_root, shape), P1_DIRNAME + (f"_{mode}" if mode else ""))


def sha256_file(path: str, block: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(block), b""):
            h.update(b)
    return h.hexdigest()


def features_state(path: Optional[str]) -> dict:
    """{"path", "present", "sha256"} for a frozen-features file (the sha256 is what the row's `--features-sha` carries)."""
    if not path:
        return {"path": None, "present": False, "sha256": None}
    present = os.path.isfile(path)
    return {"path": path, "present": present, "sha256": sha256_file(path) if present else None}
