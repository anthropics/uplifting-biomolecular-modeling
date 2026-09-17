"""Device-resident scalars (lever ``dev_scalars``): the trunk pass's constant scalars live on the model's device instead of being copied
from the host at every use — each such copy is a host→device transfer the CUDA stream synchronises on, several per Evoformer pass.

Two stock sites, restated statement for statement but for where the scalar comes from:

* ``xfold/geometry.py`` ``Vec3Array.norm`` clips the squared norm with ``torch.maximum(norm2, torch.tensor(epsilon**2, dtype=…,
  device=…))`` — a fresh device tensor built from a host float on EVERY call (three calls per template slot evaluation: the unit vector
  and the backbone frame's two axes). Here that tensor is built once per (epsilon, dtype, device) and process by the very same
  statement (``eps2_tensor``) and reused: ``torch.maximum`` sees the same operand bits — bitwise by construction.
* ``xfold/alphafold3.py`` ``Evoformer._embed_bonds`` writes the bond contact matrix with ``contact_matrix[i, j] = 1.0`` (twice under
  the OF3 weights' symmetric bonds) and ``contact_matrix[0, 0] = 0.0`` — each a host scalar copied to the device. Here the values are
  the device scalars ``contact_matrix.new_ones(())`` / ``.new_zeros(())`` (fill kernels, no transfer): the same 1.0 / 0.0 written at
  the same indices — bitwise by construction.

Composed by ``install(model)`` (forward.py ``--package-levers dev_scalars``, every mode but ``off``): ``Vec3Array.norm`` is rebound on
the class (every caller in the process), ``_embed_bonds`` on the model's Evoformer instance. A tensor is never cached while a CUDA graph
is being captured (its memory would belong to the graph's pool): the stock statement runs there instead. The census (``COUNTS``:
norm calls, eps tensors built, bond-matrix calls; ``take()`` per item) rides forward.json (``dev_scalars``) and the LEVER line.
tests/test_kit_statement_mirrors.py holds both restatements to the kit's source, live.
"""
from __future__ import annotations

import types

COUNTS = {"norm_calls": 0, "eps_tensors": 0, "bond_calls": 0}
_EPS2 = {}                      # (epsilon, dtype, device) -> the device-resident epsilon**2 tensor (the stock statement's value, built once)


def take() -> dict:
    """The census since the last take (and reset it): {'norm_calls', 'eps_tensors', 'bond_calls'}."""
    out = dict(COUNTS)
    for k in COUNTS:
        COUNTS[k] = 0
    return out


def eps2_tensor(epsilon, dtype, device):
    """``torch.tensor(epsilon**2, dtype=dtype, device=device)`` — the stock statement — built once per (epsilon, dtype, device) and
    process; never cached while the current stream is capturing a CUDA graph (then it is the stock per-call tensor)."""
    import torch
    key = (epsilon, dtype, device)
    t = _EPS2.get(key)
    if t is None:
        t = torch.tensor(epsilon**2, dtype=dtype, device=device)
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            return t
        _EPS2[key] = t; COUNTS["eps_tensors"] += 1
    return t


def norm(self, epsilon: float = 1e-6):
    """Compute Norm of Vec3Array, clipped to epsilon."""
    import torch
    COUNTS["norm_calls"] += 1
    # To avoid NaN on the backward pass, we must use maximum before the sqrt
    norm2 = self.dot(self)
    if epsilon:
        norm2 = torch.maximum(norm2, eps2_tensor(epsilon, norm2.dtype, norm2.device))   # stock: torch.tensor(epsilon**2, dtype=norm2.dtype, device=norm2.device) built here, per call
    return torch.sqrt(norm2)


def _embed_bonds(self, batch, pair_activations):
    """Embeds bond features and merges into pair activations."""
    import torch
    from xfold import of3
    COUNTS["bond_calls"] += 1
    # Construct contact matrix.
    num_tokens = batch.token_features.token_index.shape[0]
    contact_matrix = torch.zeros(
        (num_tokens, num_tokens), dtype=pair_activations.dtype, device=pair_activations.device)

    tokens_to_polymer_ligand_bonds = (
        batch.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds
    )
    gather_idxs_polymer_ligand = tokens_to_polymer_ligand_bonds.gather_idxs
    gather_mask_polymer_ligand = (
        tokens_to_polymer_ligand_bonds.gather_mask.prod(dim=1).to(
            dtype=gather_idxs_polymer_ligand.dtype)[:, None]
    )
    # If valid mask then it will be all 1's, so idxs should be unchanged.
    gather_idxs_polymer_ligand = (
        gather_idxs_polymer_ligand * gather_mask_polymer_ligand
    )

    tokens_to_ligand_ligand_bonds = (
        batch.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds
    )
    gather_idxs_ligand_ligand = tokens_to_ligand_ligand_bonds.gather_idxs
    gather_mask_ligand_ligand = tokens_to_ligand_ligand_bonds.gather_mask.prod(
        dim=1
    ).to(dtype=gather_idxs_ligand_ligand.dtype)[:, None]
    gather_idxs_ligand_ligand = (
        gather_idxs_ligand_ligand * gather_mask_ligand_ligand
    )

    gather_idxs = torch.concatenate(
        [gather_idxs_polymer_ligand, gather_idxs_ligand_ligand]
    )
    one = contact_matrix.new_ones(())               # the device scalars: stock writes the host floats 1.0 / 0.0 (a host->device copy per statement)
    zero = contact_matrix.new_zeros(())
    contact_matrix[
        gather_idxs[:, 0], gather_idxs[:, 1]
    ] = one
    if of3.OF3:
        # OF3 weights were trained with a symmetric bond matrix
        contact_matrix[gather_idxs[:, 1], gather_idxs[:, 0]] = one

    # Because all the padded index's are 0's.
    contact_matrix[0, 0] = zero

    bonds_act = self.bond_embedding(contact_matrix[:, :, None])

    return pair_activations + bonds_act


def install(model) -> dict:
    """Rebind ``xfold.geometry.Vec3Array.norm`` (the class: every caller) and ``model.evoformer._embed_bonds`` (the instance) to the
    restatements above. Idempotent. {'installed', 'already'}."""
    import importlib
    evo = model.evoformer
    if getattr(evo, "_dev_scalars", False):
        return {"installed": True, "already": True}
    for name in ("_embed_bonds", "bond_embedding"):
        if not hasattr(evo, name):
            raise RuntimeError(f"dev_scalars: {type(evo).__name__} has no {name!r} (the kit's Evoformer changed shape)")
    geometry = importlib.import_module("xfold.geometry")
    V = getattr(geometry, "Vec3Array", None)
    if V is None or not callable(getattr(V, "norm", None)) or not callable(getattr(V, "dot", None)):
        raise RuntimeError("dev_scalars: xfold.geometry.Vec3Array has no norm/dot (the kit's geometry changed shape)")
    if not getattr(V, "_dev_scalars", False):
        V._stock_norm = V.norm
        V.norm = norm
        V._dev_scalars = True
    evo._embed_bonds = types.MethodType(_embed_bonds, evo)
    evo._dev_scalars = True
    return {"installed": True, "already": False}
