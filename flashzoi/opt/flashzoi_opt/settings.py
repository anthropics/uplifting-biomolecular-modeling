"""The settings for one prediction — the documented call of the upstream package under torch's own numerics:

    import torch
    from borzoi_pytorch import Borzoi
    from borzoi_pytorch.pytorch_borzoi_helpers import predict_tracks
    models = [Borzoi.from_pretrained(f"johahi/flashzoi-replicate-{k}").to("cuda").eval() for k in range(4)]
    with torch.autocast("cuda"):
        y = predict_tracks(models, sequence_one_hot, slices)      # (1, 4, 6144, n_tracks) float32

`sequence_one_hot` is the (4, 524288) float32 one-hot (rows A,C,G,T) on the device; `slices` = every track (slice(None)); batch 1
(predict_tracks adds the batch axis); the four replicates loaded in order 0..3; autocast at torch's CUDA default dtype (float16 on the
pinned stack — recorded in the manifest as read back from torch, never assumed). The TF32 switches are torch's own defaults and are
never set (cuDNN TF32 on, matmul TF32 off — read back into the manifest); the other framework switches are the stock defaults, set
explicitly: cudnn.benchmark False, cudnn.deterministic False, deterministic algorithms off. `--det` adds the deterministic recipe
(det.py); the call itself is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple

HF_REPO_FMT = "johahi/flashzoi-replicate-{k}"
N_REPLICATES = 4
SEQ_LEN = 524288
N_TRACKS = 7611
N_BINS = 6144
ONEHOT_ROWS = ("A", "C", "G", "T")
OUTPUT_SHAPE = (1, N_REPLICATES, N_BINS, N_TRACKS)     # the documented call's return for one window and every track
OUTPUT_DTYPE = "float32"


@dataclass(frozen=True)
class Settings:
    replicates: Tuple[int, ...] = tuple(range(N_REPLICATES))
    slices: str = "all"                                  # the requested tracks (pred --tracks): "all" = slice(None) (every track), else `i,lo-hi,...` or `@file`
    batch: int = 1
    autocast: str = "cuda"                               # torch.autocast("cuda") at its default dtype
    device: str = "cuda"
    matmul_allow_tf32: Optional[bool] = None             # None = torch's own default, never set (False on the pinned stack; read back into the manifest)
    cudnn_allow_tf32: Optional[bool] = None              # None = torch's own default, never set (True on the pinned stack)
    cudnn_benchmark: bool = False                        # the stock default, set explicitly
    cudnn_deterministic: bool = False                    # the stock default, set explicitly
    deterministic_algorithms: bool = False               # the stock default, set explicitly
    seed: int = 0                                        # inert: the documented call samples nothing; recorded on every row as s0

    def repos(self):
        return [HF_REPO_FMT.format(k=k) for k in self.replicates]

    def track_slice(self):
        """The documented call's `slices` argument: slice(None) for "all", else the int64 index array parse_tracks names."""
        idx = parse_tracks(self.slices)
        if isinstance(idx, slice):
            return idx
        import numpy as np
        return np.asarray(idx, dtype=np.int64)

    def n_tracks(self) -> int:
        idx = parse_tracks(self.slices)
        return OUTPUT_SHAPE[3] if isinstance(idx, slice) else len(idx)

    def output_shape(self, n_models: int = N_REPLICATES) -> tuple:
        """The documented call's return shape for one window: (1, replicates, bins, tracks requested)."""
        return (OUTPUT_SHAPE[0], int(n_models), OUTPUT_SHAPE[2], self.n_tracks())

    def as_dict(self) -> dict:
        d = asdict(self)
        d["replicates"] = list(self.replicates)
        d["repos"] = self.repos()
        d["output_shape"] = list(self.output_shape(len(self.replicates))); d["output_dtype"] = OUTPUT_DTYPE; d["n_tracks"] = self.n_tracks()
        return d


def parse_tracks(spec):
    """--tracks: `all` -> slice(None) (every track); else the sorted, de-duplicated indices named by comma-separated `i` / `lo-hi`
    (inclusive) entries, or by `@file` with one entry per line (`#` lines and blanks skipped); an index outside 0..N_TRACKS-1, an empty or
    malformed spec is ValueError. The same grammar as the stock caller's --tracks (stock_pred.parse_tracks)."""
    spec = (spec or "all").strip()
    if spec == "all":
        return slice(None)
    if spec.startswith("@"):
        with open(spec[1:], encoding="utf-8") as fh:
            entries = [ln.strip() for ln in fh if ln.strip() and not ln.strip().startswith("#")]
    else:
        entries = [e.strip() for e in spec.split(",") if e.strip()]
    idx = set()
    for e in entries:
        lo, sep, hi = e.partition("-")
        try:
            a, b = (int(lo), int(hi)) if sep else (int(e), int(e))
        except ValueError:
            raise ValueError(f"--tracks: {e!r} is not an index or a lo-hi range") from None
        if a > b or a < 0 or b >= N_TRACKS:
            raise ValueError(f"--tracks: {e!r} is outside 0..{N_TRACKS - 1}")
        idx.update(range(a, b + 1))
    if not idx:
        raise ValueError("--tracks: no track named")
    return sorted(idx)


DEFAULT = Settings()


def apply_numerics(settings: Settings = DEFAULT) -> dict:
    """Set the framework switches (the stock defaults; the TF32 pair is left at torch's own defaults unless a value is given); returns what torch reads back."""
    import torch
    if settings.matmul_allow_tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = settings.matmul_allow_tf32
    if settings.cudnn_allow_tf32 is not None:
        torch.backends.cudnn.allow_tf32 = settings.cudnn_allow_tf32
    torch.backends.cudnn.benchmark = settings.cudnn_benchmark
    torch.backends.cudnn.deterministic = settings.cudnn_deterministic
    torch.use_deterministic_algorithms(settings.deterministic_algorithms)
    return read_back()


def read_back() -> dict:
    """torch's own switches as they stand (the manifest records these, never the requested values)."""
    import torch
    return {"matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32), "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark), "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
            "autocast_gpu_dtype": str(torch.get_autocast_dtype("cuda")) if hasattr(torch, "get_autocast_dtype") else str(torch.get_autocast_gpu_dtype())}
