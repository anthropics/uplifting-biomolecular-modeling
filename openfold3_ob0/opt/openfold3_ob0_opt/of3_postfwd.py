"""The `postfwd_mem` lever (exact class; kit-namespaced): the runner's confidence scoring AFTER the model
forward — `aggregate_confidence_ranking.get_confidence_scores` on the forward's outputs — stated so that no [S, N, N, bins] or
[S, N_atom, N_atom(, 3)] temporary is ever materialised on the device, without changing one output byte.

Why. The engine scores the S diffusion samples of an item in one batched call (below its per-sample atom cutoff, 10 000 atoms): the
statements `torch.softmax(pae_logits)` / `probs * centers` ([S, N, N, 64] fp32 each, twice: pde and pae), `compute_ptm`'s
`logits[:, mask][..., mask, :]` + softmax + `probs * weight` (three more [S, n, n, 64] pieces per call: the full complex twice, every
chain pair, every chain) and `get_token_frame_atoms`' all-atom difference tensor `x[..., None, :] - x[..., None, :, :]`
([S, N_atom, N_atom, 3] fp32, plus its square and the [S, N_atom, N_atom] distances) set the PROCESS peak of every
line after `OpenFold3.forward` has returned — above the forward's own peak.

What. Three engine functions are re-stated (their text at the pin: DIGESTS, per kit adapter; a different text refuses the lever by name — empty = unchecked):
  * `_get_confidence_scores`: pde / pae expectations and the distogram contact probabilities computed per (sample, row block) —
    `softmax` over the bins + the engine's own `probs_to_expected_error` / masked bin sum on the block — and COPIED into full-size result
    tensors of the engine's shape and dtype; gpde's two `sum(dim=[-2, -1])` statements then run verbatim on those full tensors.
  * `compute_ptm`: the engine's preamble verbatim; `ptm_ij` ([S, n, n]) assembled per (sample, row block) from `softmax` + `sum(probs *
    bin_weight, -1)` on a [rows, n, 64] slice (a view when the mask is all-true, else ONE fused 2-d gather instead of two boolean-index
    copies); the engine's tail (`(ptm_ij * pair_mask).sum(-1) / …`, `masked_fill`, `max`) verbatim on the assembled tensor.
  * `get_token_frame_atoms`: the pair mask, distances and `topk(k=3)` per (sample, row block) of atoms, the [S, N_atom, 3] index tensor
    assembled by copy; everything after it verbatim.
Bitwise argument (structural): every restated reduction is over the LAST dim
(64 bins, 3 coordinates, <= 64 contact bins) — one output element per row, no input vectorisation (extent < 128), thread-block shape
saturated (outputs per launch >= 16, MIN_ROWS) — so each output is computed by the same instruction sequence whatever the number of rows in
the launch; softmax(dim=-1) at 64 columns is one warp per row; elementwise statements and copies are exact; every reduction over a LONG
dim (N, N_atom, N^2) runs on a full-size tensor of the engine's shape, i.e. the engine's own launch. `topk` differs from a batched launch
only in how it may break EXACT ties at the k-th place; ties occur only among masked entries (distance := inf), whose frames the engine's
own atom-mask / same-chain tests reject whichever index is returned (the positions it also returns are discarded by the caller).
Below MIN_TOKENS tokens / MIN_ATOMS atoms, with `return_probs` configured, or on tensors of an unexpected rank the engine's statements run
(counted `fallback=<reason>:n`); under `<KIT>_POSTFWD_MEM_MIB=0` every call does (`reason=mib_0`, `fallback=mib_0:n`: the ablation switch).

Census (exit): `<PREFIX> LEVER name=postfwd_mem state=<on|off|refused> block_mib=<b> items=<n> fwd_peak_mib=<max allocated when scoring
began, per item> post_growth_mib=<growth of the process max while scoring, per item> post_alloc_mib=<allocated MiB sampled at block
boundaries while scoring, max per item> lean=<group:n,…> fallback=<reason:n,…|none>`.
Switches (the kit adapter's `configure(ENV=…, ENV_MIB=…)`): <KIT>_POSTFWD_MEM=1 arms; <KIT>_POSTFWD_MEM_MIB=<int> the per-block transient
budget (default BLOCK_MIB; 0 = the engine's statements on every call, by name).
"""
from __future__ import annotations

import atexit
import hashlib
import inspect
import math
import os
import sys
from typing import Any, Dict, Optional, Tuple

PREFIX = "[opt_core/of3_post.postfwd_mem]"
ENV: Optional[str] = None                                # <KIT>_POSTFWD_MEM=1
ENV_MIB: Optional[str] = None                            # <KIT>_POSTFWD_MEM_MIB=<int MiB> (0 = off by name)
VALUES = ("1",)
BLOCK_MIB = 256                                          # transient budget per block statement group (MiB)
MIN_ROWS = 16                                            # rows per launch floor (thread-block shape of the last-dim reductions saturates at 16 outputs)
MIN_TOKENS = 64                                          # below: the engine's statements (nothing to save)
MIN_ATOMS = 64
M_ACR: Optional[str] = None                              # ….core.metrics.aggregate_confidence_ranking
M_SR: Optional[str] = None                               # ….core.metrics.sample_ranking
M_CONF: Optional[str] = None                             # ….core.metrics.confidence
M_ATOMIZE: Optional[str] = None                          # ….core.utils.atomize_utils
DIGESTS: Dict[str, Tuple[str, ...]] = {}                 # engine function name -> accepted sha256[:16] of its source (empty = not checked)
CONFIGURABLE = ("PREFIX", "ENV", "ENV_MIB", "BLOCK_MIB", "M_ACR", "M_SR", "M_CONF", "M_ATOMIZE", "DIGESTS")

STATE: Dict[str, Any] = {"installed": False, "state": "off", "reason": "", "block_mib": None, "items": 0, "lean": {}, "fallback": {},
                         "fwd_peak_mib": [], "post_growth_mib": [], "post_alloc_mib": [], "digest": {}}
ORIG: Dict[str, Any] = {}                                # the engine's functions, by name (restored by uninstall(); the fallbacks call them)
_CUR: Dict[str, int] = {"alloc": 0}


def configure(**kw) -> None:
    """The kit adapter's binding: reassigns this module's engine words before install; unknown names raise."""
    for k, v in kw.items():
        if k not in CONFIGURABLE:
            raise KeyError(f"{__name__}.configure: unknown setting {k!r} (known: {', '.join(CONFIGURABLE)})")
        globals()[k] = v


def _log(msg: str) -> None:
    sys.stderr.write(f"{PREFIX} {msg}\n")


def _count(d: dict, k, n: int = 1) -> None:
    d[k] = d.get(k, 0) + n


def requested(environ=None) -> bool:
    environ = os.environ if environ is None else environ
    if not ENV:
        return False
    v = (environ.get(ENV) or "").strip()
    if not v:
        return False
    if v not in VALUES:
        raise ValueError(f"{ENV}={v!r} is not one of {'|'.join(VALUES)}")
    return True


def block_mib(environ=None) -> int:
    """The per-block transient budget in MiB: <KIT>_POSTFWD_MEM_MIB when set (a non-negative integer; 0 = off), else BLOCK_MIB."""
    environ = os.environ if environ is None else environ
    v = (environ.get(ENV_MIB) or "").strip() if ENV_MIB else ""
    if not v:
        return int(BLOCK_MIB)
    try:
        n = int(v)
    except ValueError:
        raise ValueError(f"{ENV_MIB}={v!r} is not a non-negative integer (MiB; 0 = the engine's statements)") from None
    if n < 0:
        raise ValueError(f"{ENV_MIB}={v!r} is negative (MiB; 0 = the engine's statements)")
    return n


def serving() -> bool:
    return STATE["state"] == "on"


def census_line() -> str:
    j = lambda xs: ",".join(str(int(x)) for x in xs) or "-"                                   # noqa: E731
    kv = lambda d: ",".join("%s:%d" % kv for kv in sorted(d.items())) or "none"            # noqa: E731
    return (f"{PREFIX} LEVER name=postfwd_mem state={STATE['state']}" + (f" reason={STATE['reason']}" if STATE["reason"] else "") +
            f" block_mib={STATE['block_mib']} items={STATE['items']} fwd_peak_mib={j(STATE['fwd_peak_mib'])} post_growth_mib={j(STATE['post_growth_mib'])}"
            f" post_alloc_mib={j(STATE['post_alloc_mib'])} lean={kv(STATE['lean'])} fallback={kv(STATE['fallback'])}")


def digest(fn) -> str:
    """sha256[:16] of a function's source text (the engine statements this lever re-states are pinned by it)."""
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return "unreadable"
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]


# ----------------------------------------------------------------------------------------------------------------------------------
# block planning + memory sampling
def _rows(n_rows: int, row_bytes: int) -> int:
    """Rows per block for a statement group whose transient costs `row_bytes` per row under the budget; >= MIN_ROWS."""
    budget = int(STATE["block_mib"] or BLOCK_MIB) << 20
    return max(MIN_ROWS, min(n_rows, budget // max(1, row_bytes)))


def _blocks(n: int, r: int):
    """[r0, r1) row ranges of size r covering range(n); a trailing range shorter than MIN_ROWS is merged into its predecessor."""
    out = []
    r0 = 0
    while r0 < n:
        r1 = min(n, r0 + r)
        if n - r1 < MIN_ROWS and r1 < n:
            r1 = n
        out.append((r0, r1))
        r0 = r1
    return out


def _sample(t=None) -> None:
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            _CUR["alloc"] = max(_CUR["alloc"], int(torch.cuda.memory_allocated() >> 20))
    except Exception:  # noqa: BLE001
        pass


# ----------------------------------------------------------------------------------------------------------------------------------
# the lean statement groups
def expected_error_lean(logits, probs_to_expected_error, cfg: dict):
    """`probs_to_expected_error(torch.softmax(logits, -1), **cfg)` for logits [*, N1, N2, B] without the [*, N1, N2, B] probabilities:
    per (leading index, row block) softmax + the engine's expectation on the block, copied into the [*, N1, N2] result."""
    import torch
    *lead, n1, n2, nb = logits.shape
    x = logits.reshape(-1, n1, n2, nb)
    r = _rows(n1, n2 * nb * 4 * 3)                                                          # fp32 upcast + probs + product per row
    out = None
    for s in range(x.shape[0]):
        for r0, r1 in _blocks(n1, r):
            e = probs_to_expected_error(torch.softmax(x[s, r0:r1], dim=-1), **cfg)         # [rows, N2]: the engine's statements on the block
            if out is None:
                out = torch.empty((x.shape[0], n1, n2), dtype=e.dtype, device=e.device)
            out[s, r0:r1].copy_(e)
            del e
        _sample()
    return out.reshape(*lead, n1, n2)


def gpde_lean(pde, logits, bin_min, bin_max, no_bins, eps=1e-8, orig=None, **kwargs):
    """`compute_global_predicted_distance_error`: the distogram contact probabilities per row block (softmax + the masked bin sum on the
    block, copied into the full [*, N, N] tensor), then the engine's two gpde statements verbatim on the full tensors."""
    import torch
    device = logits.device
    *lead, n1, n2, nb = logits.shape
    x = logits.reshape(-1, n1, n2, nb)
    distogram_bin_ends = torch.linspace(bin_min, bin_max, no_bins + 1, device=device)[1:]  # verbatim
    distogram_bins_8A = distogram_bin_ends <= 8.0                                           # verbatim
    r = _rows(n1, n2 * nb * 4 * 3)
    contact_probs = None
    for s in range(x.shape[0]):
        for r0, r1 in _blocks(n1, r):
            probs = torch.softmax(x[s, r0:r1], dim=-1)
            c = torch.sum(probs[..., distogram_bins_8A], dim=-1)                          # verbatim on the block
            if contact_probs is None:
                contact_probs = torch.empty((x.shape[0], n1, n2), dtype=c.dtype, device=c.device)
            contact_probs[s, r0:r1].copy_(c)
            del probs, c
        _sample()
    contact_probs = contact_probs.reshape(*lead, n1, n2)
    gpde = torch.sum(contact_probs * pde, dim=[-2, -1]) / (                              # verbatim, full tensors
        torch.sum(contact_probs, dim=[-2, -1]) + eps
    )
    _sample()
    return gpde, contact_probs


def make_compute_ptm(get_bin_centers, orig):
    """The engine's `compute_ptm` with `ptm_ij` assembled per (sample, row block); preamble and tail verbatim."""
    import torch

    def compute_ptm(logits, has_frame, bin_min, bin_max, no_bins, mask_i, asym_id=None, interface=False, eps=1e-8):
        if logits.dim() != 4 or not STATE["block_mib"]:
            if STATE["block_mib"]:
                _count(STATE["fallback"], "ptm_rank")
            return orig(logits, has_frame, bin_min, bin_max, no_bins, mask_i, asym_id=asym_id, interface=interface, eps=eps)
        device, dtype = logits.device, logits.dtype
        mask_i = mask_i.to(device=device, dtype=torch.bool)

        if interface and asym_id is None:
            raise ValueError("asym_id is required when interface=True")

        n_all = int(mask_i.shape[0])
        n = int(mask_i.sum())                                                               # one host read (the engine's boolean indexing below syncs likewise)
        if n < MIN_TOKENS:                                                                  # small contexts (ligand / short chains): the engine's statements
            _count(STATE["fallback"], "ptm_lt_min_tokens")
            return orig(logits, has_frame, bin_min, bin_max, no_bins, mask_i, asym_id=asym_id, interface=interface, eps=eps)

        if asym_id is not None:
            asym_id = asym_id[mask_i].to(device=device)

        # Compute bin weights (verbatim)
        num_tokens_considered = mask_i.sum().clamp_min(1).to(dtype)
        clipped = torch.maximum(
            num_tokens_considered, torch.tensor(19.0, device=device, dtype=dtype)
        )
        d0 = 1.24 * (clipped - 15.0).clamp_min(0).pow(1.0 / 3.0) - 1.8

        bin_centers = get_bin_centers(bin_min, bin_max, no_bins, device, dtype)
        bin_weight = 1.0 / (1.0 + (bin_centers / d0) ** 2)

        # Subset to token mask — lean: ptm_ij [S, n, n] assembled per (sample, row block)
        has_frame = has_frame[:, mask_i].bool()
        full = n == n_all
        idx = None if full else torch.nonzero(mask_i, as_tuple=False).squeeze(1)
        S, nb = int(logits.shape[0]), int(logits.shape[-1])
        r = _rows(n, n * nb * 4 * 3 + (0 if full else n * nb * logits.element_size()))
        ptm_ij = None
        for s in range(S):
            ls = logits[s]
            for r0, r1 in _blocks(n, r):
                blk = ls[r0:r1] if full else ls[idx[r0:r1].unsqueeze(1), idx.unsqueeze(0)]   # [rows, n, B]: a view, or one gather (the engine: two boolean-index copies)
                probs = torch.softmax(blk, dim=-1)
                pij = torch.sum(probs * bin_weight, dim=-1)                                  # verbatim on the block
                if ptm_ij is None:
                    ptm_ij = torch.empty((S, n, n), dtype=pij.dtype, device=pij.device)
                ptm_ij[s, r0:r1].copy_(pij)
                del blk, probs, pij
            _sample()
        _count(STATE["lean"], "ptm")

        # Subset tokens j to different chain from i if interface=True (verbatim tail, full tensors)
        if interface:
            pair_mask = asym_id.unsqueeze(-1) != asym_id.unsqueeze(-2)
            tm_i = (ptm_ij * pair_mask).sum(dim=-1) / pair_mask.sum(dim=-1).clamp_min(eps)
        else:
            tm_i = ptm_ij.sum(dim=-1) / num_tokens_considered

        tm_i = tm_i.masked_fill(~has_frame, 0.0)
        return tm_i.max(dim=-1).values

    compute_ptm.__wrapped__ = orig
    compute_ptm._of3opt_postfwd_mem = True
    return compute_ptm


def make_token_frame_atoms(A, orig):
    """The engine's `get_token_frame_atoms` with the pair mask / distances / top-3 per (sample, row block) of atoms; the rest verbatim.
    `A` is the engine's atomize_utils module (broadcast_token_feat_to_atoms, get_token_atom_index_offset)."""
    import torch

    def get_token_frame_atoms(batch, x, atom_mask, angle_threshold=25.0, eps=1e-8, inf=1e9):
        if not STATE["block_mib"]:
            return orig(batch, x, atom_mask, angle_threshold=angle_threshold, eps=eps, inf=inf)
        n_atom = int(x.shape[-2])
        atom_asym_id = A.broadcast_token_feat_to_atoms(                                     # verbatim (hoisted: the row blocks read it)
            token_mask=batch["token_mask"],
            num_atoms_per_token=batch["num_atoms_per_token"],
            token_feat=batch["asym_id"],
        )
        lead = tuple(x.shape[:-2])
        ok = (x.dim() >= 2 and x.shape[-1] == 3 and n_atom >= MIN_ATOMS and atom_mask.shape[-1] == n_atom and atom_asym_id.shape[-1] == n_atom
              and atom_mask.numel() == n_atom and atom_asym_id.numel() == n_atom)          # one mask / chain row for all samples (the inference form)
        if not ok:
            _count(STATE["fallback"], "frames_shape" if n_atom >= MIN_ATOMS else "frames_lt_min_atoms")
            return orig(batch, x, atom_mask, angle_threshold=angle_threshold, eps=eps, inf=inf)
        am = atom_mask.reshape(n_atom)
        aid = atom_asym_id.reshape(n_atom)
        xs = x.reshape(-1, n_atom, 3)
        r = _rows(n_atom, n_atom * (3 * 4 * 3 + 4 * 4))                                    # diff / square / eps-sum pieces + distances and masks per row
        closest = torch.empty((xs.shape[0], n_atom, 3), dtype=torch.long, device=x.device)
        for s in range(xs.shape[0]):
            for r0, r1 in _blocks(n_atom, r):
                # Create pairwise atom mask, restricted to atoms within the same chain (the engine's statements, on rows r0:r1)
                pair_mask = am[r0:r1][..., None] * am[..., None, :]
                atom_asym_id_mask = aid[r0:r1][..., None] == aid[..., None, :]
                pair_mask = pair_mask * atom_asym_id_mask
                # Compute distance matrix rows [rows, N_atom]
                d = torch.sum(eps + (xs[s, r0:r1][..., None, :] - xs[s][..., None, :, :]) ** 2, dim=-1) ** 0.5
                d = d * pair_mask + inf * (1 - pair_mask)
                _, ci = torch.topk(d, k=3, dim=-1, largest=False)
                closest[s, r0:r1].copy_(ci)
                del pair_mask, atom_asym_id_mask, d, ci
            _sample()
        closest_atom_index = closest.reshape(*lead, n_atom, 3)
        _count(STATE["lean"], "frames")

        # ---- the engine's statements from here on (verbatim) ----
        # Find indices of two closest atoms for start atoms
        # [*, N_token]
        start_atom_index = batch["start_atom_index"].long()
        start_atom_index = start_atom_index.expand(
            *x.shape[:-2], start_atom_index.shape[-1]
        )
        # (the engine's `_, closest_atom_index = torch.topk(d, k=3, dim=-1, largest=False)` is the blockwise loop above)
        a_index = torch.gather(closest_atom_index[..., 1], dim=-1, index=start_atom_index)
        c_index = torch.gather(closest_atom_index[..., 2], dim=-1, index=start_atom_index)

        # Construct indices of atoms used for frame construction
        # [*, N_token]
        is_standard_protein = batch["is_protein"] * (1 - batch["is_atomized"])
        is_standard_nucleotide = (batch["is_dna"] + batch["is_rna"]) * (
            1 - batch["is_atomized"]
        )

        restype = batch["restype"]
        n_atom_index_offset, n_atom_mask = A.get_token_atom_index_offset(
            atom_name="N", restype=restype
        )
        ca_atom_index_offset, ca_atom_mask = A.get_token_atom_index_offset(
            atom_name="CA", restype=restype
        )
        c_atom_index_offset, c_atom_mask = A.get_token_atom_index_offset(
            atom_name="C", restype=restype
        )
        c3p_atom_index_offset, c3p_atom_mask = A.get_token_atom_index_offset(
            atom_name="C3'", restype=restype
        )
        c1p_atom_index_offset, c1p_atom_mask = A.get_token_atom_index_offset(
            atom_name="C1'", restype=restype
        )
        c4p_atom_index_offset, c4p_atom_mask = A.get_token_atom_index_offset(
            atom_name="C4'", restype=restype
        )
        frame_atoms = {
            "a": {
                "index": (
                    a_index * batch["is_atomized"]
                    + (start_atom_index + n_atom_index_offset) * is_standard_protein
                    + (start_atom_index + c3p_atom_index_offset) * is_standard_nucleotide
                ),
                "token_atom_mask": (
                    batch["is_atomized"]
                    + n_atom_mask * is_standard_protein
                    + c3p_atom_mask * is_standard_nucleotide
                ),
            },
            "b": {
                "index": (
                    start_atom_index * batch["is_atomized"]
                    + (start_atom_index + ca_atom_index_offset) * is_standard_protein
                    + (start_atom_index + c1p_atom_index_offset) * is_standard_nucleotide
                ),
                "token_atom_mask": (
                    batch["is_atomized"]
                    + ca_atom_mask * is_standard_protein
                    + c1p_atom_mask * is_standard_nucleotide
                ),
            },
            "c": {
                "index": (
                    c_index * batch["is_atomized"]
                    + (start_atom_index + c_atom_index_offset) * is_standard_protein
                    + (start_atom_index + c4p_atom_index_offset) * is_standard_nucleotide
                ),
                "token_atom_mask": (
                    batch["is_atomized"]
                    + c_atom_mask * is_standard_protein
                    + c4p_atom_mask * is_standard_nucleotide
                ),
            },
        }

        # Extract coordinates
        for key in frame_atoms:
            frame_atoms[key].update(
                {
                    "atom_positions": torch.gather(
                        x,
                        dim=-2,
                        index=frame_atoms[key]["index"]
                        .unsqueeze(-1)
                        .expand(*(x.shape[:-2] + (frame_atoms[key]["index"].shape[-1], 3)))
                        .long(),
                    ),
                    "asym_id": torch.gather(
                        atom_asym_id.expand(*x.shape[:-2], atom_asym_id.shape[-1]),
                        dim=-1,
                        index=frame_atoms[key]["index"].long(),
                    ),
                    "atom_mask": torch.gather(
                        atom_mask.expand(*x.shape[:-2], atom_mask.shape[-1]),
                        dim=-1,
                        index=frame_atoms[key]["index"].long(),
                    )
                    * batch["token_mask"]
                    * frame_atoms[key]["token_atom_mask"],
                }
            )

        # Compute cosine of angles
        u = frame_atoms["a"]["atom_positions"] - frame_atoms["b"]["atom_positions"]
        v = frame_atoms["c"]["atom_positions"] - frame_atoms["b"]["atom_positions"]
        uv = torch.einsum("...i,...i->...", u, v)
        u_norm = (eps + torch.sum(u**2, dim=-1)) ** 0.5
        v_norm = (eps + torch.sum(v**2, dim=-1)) ** 0.5
        cos_angle = uv / (u_norm * v_norm)

        # Compute valid frame mask from angle constraints
        # (for ligand and non-standard residues)
        cos_angle_min_bound = math.cos((180 - angle_threshold) * math.pi / 180)
        cos_angle_max_bound = math.cos(angle_threshold * math.pi / 180)
        valid_frame_mask_angle = (cos_angle < cos_angle_max_bound) * (
            cos_angle > cos_angle_min_bound
        )
        valid_frame_mask_angle = (
            valid_frame_mask_angle * batch["is_atomized"]
            + torch.ones_like(valid_frame_mask_angle) * (1 - batch["is_atomized"])
        ) * batch["token_mask"]

        # Compute valid frame mask from atom mask constraints
        valid_frame_mask_atom = (
            frame_atoms["a"]["atom_mask"]
            * frame_atoms["b"]["atom_mask"]
            * frame_atoms["c"]["atom_mask"]
        )

        # Compute valid frame mask from chain constraints
        valid_frame_mask_asym_id = (
            frame_atoms["a"]["asym_id"] == frame_atoms["b"]["asym_id"]
        ) * (frame_atoms["b"]["asym_id"] == frame_atoms["c"]["asym_id"])

        # Compute final valid frame mask
        valid_frame_mask = (
            valid_frame_mask_angle * valid_frame_mask_atom * valid_frame_mask_asym_id
        )
        phi = (
            frame_atoms["a"]["atom_positions"],
            frame_atoms["b"]["atom_positions"],
            frame_atoms["c"]["atom_positions"],
        )

        return phi, valid_frame_mask

    get_token_frame_atoms.__wrapped__ = orig
    get_token_frame_atoms._of3opt_postfwd_mem = True
    return get_token_frame_atoms


def make_get_confidence_scores(acr, orig):
    """The engine's `_get_confidence_scores` with the pde / pae / contact statement groups on the lean forms above; the rest verbatim
    (the sample-ranking / chain metrics reach the lean `compute_ptm` through their module's binding). `acr` is the engine module."""
    import torch

    def _get_confidence_scores(batch: dict, outputs: dict, config) -> dict:
        if not STATE["block_mib"]:
            _count(STATE["fallback"], "mib_0")
            return orig(batch=batch, outputs=outputs, config=config)
        pae = outputs.get("pae_logits")
        ref = pae if pae is not None else outputs["pde_logits"]
        n_tok = int(ref.shape[-2])
        if ref.dim() < 3 or n_tok < MIN_TOKENS:
            _count(STATE["fallback"], "scores_lt_min_tokens" if ref.dim() >= 3 else "scores_rank")
            return orig(batch=batch, outputs=outputs, config=config)
        cuda = ref.is_cuda
        if cuda:
            entry_peak = int(torch.cuda.max_memory_allocated(ref.device) >> 20)
            _CUR["alloc"] = int(torch.cuda.memory_allocated(ref.device) >> 20)
        confidence_scores = {}
        confidence_scores["plddt"] = (
            acr.probs_to_expected_error(
                torch.softmax(outputs["plddt_logits"], dim=-1), **config.confidence.plddt
            )
            * 100.0
        )

        if config.confidence.pde.return_probs:                                           # the probabilities themselves are an output: the engine's statements
            pde_probs = torch.softmax(outputs["pde_logits"], dim=-1)
            confidence_scores["pde"] = acr.probs_to_expected_error(
                pde_probs, **config.confidence.pde
            )
            confidence_scores["pde_probs"] = pde_probs
            _count(STATE["fallback"], "pde_return_probs")
        else:
            confidence_scores["pde"] = expected_error_lean(outputs["pde_logits"], acr.probs_to_expected_error, dict(**config.confidence.pde))
            _count(STATE["lean"], "pde")

        if config.confidence.distogram.return_contact_probs:
            confidence_scores["gpde"], contact_probs = acr.compute_global_predicted_distance_error(
                pde=confidence_scores["pde"],
                logits=outputs["distogram_logits"],
                **config.confidence.distogram,
            )
            confidence_scores["contact_probs"] = contact_probs
            _count(STATE["fallback"], "contact_return_probs")
        else:
            confidence_scores["gpde"], contact_probs = gpde_lean(
                pde=confidence_scores["pde"],
                logits=outputs["distogram_logits"],
                **config.confidence.distogram,
            )
            del contact_probs
            _count(STATE["lean"], "contact")

        if config.architecture.heads.pae.enabled:
            if config.confidence.pae.return_probs:
                pae_probs = torch.softmax(outputs["pae_logits"], dim=-1)
                confidence_scores["pae"] = acr.probs_to_expected_error(
                    pae_probs, **config.confidence.pae
                )
                confidence_scores["pae_probs"] = pae_probs
                _count(STATE["fallback"], "pae_return_probs")
            else:
                confidence_scores["pae"] = expected_error_lean(outputs["pae_logits"], acr.probs_to_expected_error, dict(**config.confidence.pae))
                _count(STATE["lean"], "pae")

            _, valid_frame_mask = acr.get_token_frame_atoms(                                # the module's binding: the lean form while installed
                batch=batch,
                x=outputs["atom_positions_predicted"],
                atom_mask=batch["atom_mask"],
            )

            valid_frame_mask = valid_frame_mask.bool()

            confidence_scores.update(
                acr.full_complex_sample_ranking_metric(
                    batch=batch,
                    output=outputs,
                    has_frame=valid_frame_mask,
                    **config.confidence.sample_ranking.full_complex,
                    **config.confidence.ptm,
                )
            )
            _sample()

            if config.confidence.sample_ranking.chain_pair_iptm.enabled:
                confidence_scores.update(
                    acr.compute_chain_pair_iptm(
                        batch=batch,
                        logits=outputs["pae_logits"],
                        has_frame=valid_frame_mask,
                        **config.confidence.ptm,
                    )
                )
                _sample()

            if config.confidence.sample_ranking.chain_ptm.enabled:
                confidence_scores.update(
                    acr.compute_chain_ptm(
                        batch=batch,
                        outputs=outputs,
                        has_frame=valid_frame_mask,
                        **config.confidence.ptm,
                    )
                )
                _sample()

        STATE["items"] += 1
        if cuda:
            _sample()
            exit_peak = int(torch.cuda.max_memory_allocated(ref.device) >> 20)
            STATE["fwd_peak_mib"].append(entry_peak)
            STATE["post_growth_mib"].append(max(0, exit_peak - entry_peak))
            STATE["post_alloc_mib"].append(_CUR["alloc"])
        return confidence_scores

    _get_confidence_scores.__wrapped__ = orig
    _get_confidence_scores._of3opt_postfwd_mem = True
    return _get_confidence_scores


# ----------------------------------------------------------------------------------------------------------------------------------
def install(environ=None) -> dict:
    """Rebind the engine's `_get_confidence_scores` (M_ACR), `get_token_frame_atoms` (M_ACR's binding of M_ATOMIZE's) and `compute_ptm`
    (M_CONF's, and M_SR's binding of it). Idempotent. An engine without the functions, or whose function text is not a pinned one, is
    refused by name (state=refused, nothing patched); <KIT>_POSTFWD_MEM_MIB=0 leaves the engine as it is (state=off reason=mib_0)."""
    environ = os.environ if environ is None else environ
    if STATE["installed"] or not requested(environ):
        return STATE
    if not (M_ACR and M_SR and M_CONF and M_ATOMIZE):
        raise RuntimeError(f"{PREFIX} not bound to an engine: the kit adapter must configure(M_ACR=, M_SR=, M_CONF=, M_ATOMIZE=) before install")
    mib = block_mib(environ)
    STATE["block_mib"] = mib                                                                # 0: installed, and every call runs the engine's statements (fallback=mib_0 — the ablation switch)
    import importlib
    try:
        acr = importlib.import_module(M_ACR); sr = importlib.import_module(M_SR); conf = importlib.import_module(M_CONF); A = importlib.import_module(M_ATOMIZE)
        found = {"_get_confidence_scores": acr._get_confidence_scores, "compute_ptm": conf.compute_ptm, "get_token_frame_atoms": A.get_token_frame_atoms,
                 "probs_to_expected_error": conf.probs_to_expected_error, "compute_global_predicted_distance_error": conf.compute_global_predicted_distance_error}
        assert sr.compute_ptm is conf.compute_ptm and acr.get_token_frame_atoms is A.get_token_frame_atoms and acr.probs_to_expected_error is conf.probs_to_expected_error
        _ = conf.get_bin_centers, A.broadcast_token_feat_to_atoms, A.get_token_atom_index_offset, acr.full_complex_sample_ranking_metric
    except Exception as e:  # noqa: BLE001
        STATE.update(installed=True, state="refused", reason=f"engine_surface:{type(e).__name__}")
        _log(f"REFUSED: the engine's confidence scoring surface is not the pinned one ({type(e).__name__}: {str(e)[:160]}) — its statements run as they are")
        atexit.register(lambda: sys.stderr.write(census_line() + "\n"))
        return STATE
    for name, fn in found.items():
        d = digest(getattr(fn, "__wrapped__", fn))
        STATE["digest"][name] = d
        acc = DIGESTS.get(name) or ()
        if acc and d not in acc:
            STATE.update(installed=True, state="refused", reason=f"engine_source:{name}:{d}")
            _log(f"REFUSED: {name}'s source digest {d} is not a pinned one ({'|'.join(acc)}) — the engine's statements run as they are")
            atexit.register(lambda: sys.stderr.write(census_line() + "\n"))
            return STATE
    if not getattr(acr._get_confidence_scores, "_of3opt_postfwd_mem", False):
        ORIG.update(found)
        ptm = make_compute_ptm(conf.get_bin_centers, conf.compute_ptm)
        conf.compute_ptm = ptm; sr.compute_ptm = ptm
        frames = make_token_frame_atoms(A, A.get_token_frame_atoms)
        acr.get_token_frame_atoms = frames                                                 # the scorer's binding only: the loss module's (training) keeps the engine's
        acr._get_confidence_scores = make_get_confidence_scores(acr, acr._get_confidence_scores)
    STATE.update(installed=True, state="on", reason="mib_0" if mib == 0 else "")
    if mib == 0:
        _log(f"installed, {ENV_MIB}=0: every call runs the engine's own scoring statements (counted fallback=mib_0)")
        atexit.register(lambda: sys.stderr.write(census_line() + "\n"))
        return STATE
    _log(f"installed: confidence scoring after the forward per (sample, row block) under a {mib} MiB block budget — pde/pae expectations, "
         f"contact probabilities, compute_ptm's ptm_ij and the frame-atom distances/top-3 assembled blockwise, every long reduction on the "
         f"engine's full tensors (digests {','.join('%s=%s' % kv for kv in sorted(STATE['digest'].items()))})")
    atexit.register(lambda: sys.stderr.write(census_line() + "\n"))
    return STATE


def uninstall() -> None:
    """Restore the engine's functions (tests)."""
    if not ORIG:
        return
    import importlib
    acr = importlib.import_module(M_ACR); sr = importlib.import_module(M_SR); conf = importlib.import_module(M_CONF)
    acr._get_confidence_scores = ORIG["_get_confidence_scores"]; acr.get_token_frame_atoms = ORIG["get_token_frame_atoms"]
    conf.compute_ptm = ORIG["compute_ptm"]; sr.compute_ptm = ORIG["compute_ptm"]
    ORIG.clear()
    STATE.update(installed=False, state="off", reason="")
