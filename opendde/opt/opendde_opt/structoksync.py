"""structok_sync — the tree's own lever on the structural-token expander's per-role-pair host round-trips (exact class: the same rows reach
the same projections in the same order; only how the rows of a role pair are enumerated changes).

Stock OpenDDE 1.1.1 (`opendde/model/modules/structural_tokens.py`, `StructuralTokenExpander._pair_project_by_role_full{,_chunk,_tile}`)
projects the structural pair tensor role pair by role pair: for each of the n_roles² (role_i, role_j) pairs it builds a boolean row mask,
asks `torch.any(mask)` — a device-to-host read (`structural_tokens.py:311`, 343 of them per item at 384 tokens, 741 at 768) — and then
gathers / scatters the pair's rows by boolean-mask indexing (two more reads per taken pair, `:312-313`): several hundred host
synchronisations inside the trunk phase of every item.

The lever serves the three methods with ONE host read per call: the rows' pair ids (`role_i * n_roles + role_j`, the stock tensors) are
counted on the device (`bincount`) and read once; a stable sort of the pair ids enumerates, pair after pair in the stock order, exactly the
rows the stock mask selects, in the same ascending order; each taken pair gathers those rows by index (`flat_z[..., idx, :]` — the op the
mask indexing lowers to), runs the SAME projection module on the same [rows, c_z] operand, and writes them back by index assignment
(`flat_delta[..., idx, :] = ...`, the stock statement's own index_put); an empty pair adds the stock dummy term. Operands, shapes, modules and
statement order are the stock ones, so the result is bitwise the stock result (CPU and CUDA, with or without deterministic algorithms —
`tests/test_structoksync.py` holds it against the stock methods). Nothing else of the expander is touched.

Evidence: `STATS` — `calls` (projection calls served), `pairs` (role pairs projected), `empty` (dummy terms), `host_reads` (= calls: one
per call instead of n_roles² + 2 x taken pairs), per method counts; the LEVER row prints them; `ran.COUNTERS["structok_sync"]` = calls.
Left out by name with `MODEL_OPT_LEVERS_OFF=structok_sync`.
"""
from __future__ import annotations

import json

LEVER = "structok_sync"
TAG = "opendde-opt"
MODULE, CLASS = "opendde.model.modules.structural_tokens", "StructuralTokenExpander"
METHODS = ("_pair_project_by_role_full", "_pair_project_by_role_full_chunk", "_pair_project_by_role_full_tile")

_STATS0 = {"installed": False, "patches": 0, "calls": 0, "pairs": 0, "empty": 0, "host_reads": 0, "by_method": {}, "failures": 0, "failure": None}
STATS = json.loads(json.dumps(_STATS0))
_PATCH: dict = {}


class ActivationError(RuntimeError):
    pass


def project_rows(expander, flat_z, role_i, role_j):
    """The stock per-role-pair projection of `flat_z` ([..., rows, c_z]) for the rows' roles `role_i` / `role_j` ([rows] each), with one host
    read: returns (flat_delta, dummy_use) exactly as the stock loop leaves them."""
    import torch
    n_roles = int(expander.n_roles)
    flat_delta = torch.zeros_like(flat_z)
    dummy_input = flat_z[..., :1, :]
    dummy_use = flat_z.new_zeros(())
    pair = role_i.to(torch.long) * n_roles + role_j.to(torch.long)                 # [rows] pair id, role_i major (the stock loop order)
    counts = torch.bincount(pair, minlength=n_roles * n_roles).tolist()             # the ONE host read of this call
    STATS["host_reads"] += 1
    order = torch.sort(pair, stable=True).indices if pair.numel() else pair         # pair after pair, ascending row within a pair = the mask's own order
    start = 0
    for proj_idx in range(n_roles * n_roles):
        projection = expander.pair_block_proj[proj_idx]
        n = int(counts[proj_idx]) if proj_idx < len(counts) else 0
        if n:
            idx = order[start:start + n]
            flat_delta[..., idx, :] = projection(flat_z[..., idx, :])                 # the stock statement on the stock rows (index form of the mask)
            STATS["pairs"] += 1
        else:
            dummy_use = dummy_use + projection(dummy_input).sum() * 0.0              # the stock dummy term (DDP: every projection used)
            STATS["empty"] += 1
        start += n
    return flat_delta, dummy_use


def _full(orig):
    def _pair_project_by_role_full(self, z, role):
        n_struct = role.shape[-1]
        batch_shape = z.shape[:-3]
        flat_z = z.reshape(*batch_shape, n_struct * n_struct, self.c_z)
        role_i = role[:, None].expand(n_struct, n_struct).reshape(-1)
        role_j = role[None, :].expand(n_struct, n_struct).reshape(-1)
        flat_delta, dummy_use = project_rows(self, flat_z, role_i, role_j)
        _count("_pair_project_by_role_full")
        return flat_delta.reshape(*batch_shape, n_struct, n_struct, self.c_z) + dummy_use
    _pair_project_by_role_full._orig = orig
    _pair_project_by_role_full._structok_sync = True
    return _pair_project_by_role_full


def _chunk(orig):
    def _pair_project_by_role_full_chunk(self, z, role, row_index):
        n_struct = role.shape[-1]
        chunk_len = row_index.numel()
        batch_shape = z.shape[:-3]
        flat_z = z.reshape(*batch_shape, chunk_len * n_struct, self.c_z)
        row_role = role.index_select(dim=0, index=row_index)
        role_i = row_role[:, None].expand(chunk_len, n_struct).reshape(-1)
        role_j = role[None, :].expand(chunk_len, n_struct).reshape(-1)
        flat_delta, dummy_use = project_rows(self, flat_z, role_i, role_j)
        _count("_pair_project_by_role_full_chunk")
        return flat_delta.reshape(*batch_shape, chunk_len, n_struct, self.c_z) + dummy_use
    _pair_project_by_role_full_chunk._orig = orig
    _pair_project_by_role_full_chunk._structok_sync = True
    return _pair_project_by_role_full_chunk


def _tile(orig):
    def _pair_project_by_role_full_tile(self, z, role, row_index, col_index):
        row_len = row_index.numel()
        col_len = col_index.numel()
        batch_shape = z.shape[:-3]
        flat_z = z.reshape(*batch_shape, row_len * col_len, self.c_z)
        row_role = role.index_select(dim=0, index=row_index)
        col_role = role.index_select(dim=0, index=col_index)
        role_i = row_role[:, None].expand(row_len, col_len).reshape(-1)
        role_j = col_role[None, :].expand(row_len, col_len).reshape(-1)
        flat_delta, dummy_use = project_rows(self, flat_z, role_i, role_j)
        _count("_pair_project_by_role_full_tile")
        return flat_delta.reshape(*batch_shape, row_len, col_len, self.c_z) + dummy_use
    _pair_project_by_role_full_tile._orig = orig
    _pair_project_by_role_full_tile._structok_sync = True
    return _pair_project_by_role_full_tile


FACTORIES = {"_pair_project_by_role_full": _full, "_pair_project_by_role_full_chunk": _chunk, "_pair_project_by_role_full_tile": _tile}


def _count(method: str) -> None:
    STATS["calls"] += 1
    STATS["by_method"][method] = STATS["by_method"].get(method, 0) + 1


def _sync() -> None:
    ps = list(_PATCH.values())
    STATS["patches"] = sum(1 for p in ps if getattr(p, "state", None) == "installed")
    STATS["installed"] = bool(ps) and STATS["patches"] == len(ps)


def install() -> None:
    """Patch the expander's three role-pair projection methods (now, or at the module's import). Idempotent."""
    from opt_core import autoload
    if not _PATCH:
        try:
            for m in METHODS:
                _PATCH[m] = autoload.patch_attr_at_import(MODULE, f"{CLASS}.{m}", FACTORIES[m], tag=TAG, name=LEVER)
        except autoload.PatchError as e:
            raise ActivationError(str(e)) from None
    _sync()


def armed() -> bool:
    return bool(_PATCH)


def _reset() -> None:
    """Test support: counters back to import time (the patches stay: one AttrPatch per site per process)."""
    STATS.clear(); STATS.update(json.loads(json.dumps(_STATS0)))


def fallbacks(planned) -> list:
    if LEVER not in planned or not STATS["failure"]:
        return []
    return [f"{LEVER}:{STATS['failure']}"]


def kit_stats() -> dict:
    _sync()
    return json.loads(json.dumps(STATS, default=str))
