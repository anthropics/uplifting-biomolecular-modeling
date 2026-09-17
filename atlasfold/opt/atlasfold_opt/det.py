"""The --det recipe: det=1 makes stock and kit runs reproducible run-to-run on one card (the precondition for comparing outputs byte for byte).
Stock atlasfold already forks the torch RNG per seed (runner.py seed_context); this adds the library switches: cuDNN deterministic, TF32 off,
deterministic algorithms (warn_only), the cuBLAS workspace variable. torch's ``fill_uninitialized_memory`` (a debugging aid deterministic mode
switches on: one fill kernel per uninitialised allocation) is switched back off — the stock subprocess's recipe (env_for_subprocess: the cuBLAS
workspace variable only) does not have it either, so the fill plays no part in byte identity with stock and would only cost a memset per allocation."""
import os


def apply(det: int, tag: str = "atlasfold-opt") -> dict:
    facts = {"det": int(det or 0)}
    if not det:
        return facts
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        facts["deterministic_algorithms"] = "warn_only"
    except Exception as e:  # noqa: BLE001
        facts["deterministic_algorithms"] = f"unavailable:{type(e).__name__}"
    try:
        torch.utils.deterministic.fill_uninitialized_memory = False          # deterministic algorithms without the per-allocation fill (see the module docstring)
        facts["fill_uninitialized_memory"] = bool(torch.utils.deterministic.fill_uninitialized_memory)
    except Exception as e:  # noqa: BLE001
        facts["fill_uninitialized_memory"] = f"unavailable:{type(e).__name__}"
    facts["cublas_workspace"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    return facts


def env_for_subprocess(det: int) -> dict:
    """What a clean stock subprocess needs for the same recipe (mode off under --det 1)."""
    return {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"} if det else {}          # the whole stock-subprocess recipe (stock forks its own RNG per seed)
