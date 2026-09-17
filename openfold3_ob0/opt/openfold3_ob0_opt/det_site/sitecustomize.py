"""The det site: first on the stock child's PYTHONPATH under ``--det 1`` (det.py DET_SITE; opt_core.stock_proof's ``det_exception``),
so the interpreter runs this file at start-up, before ``run_openfold`` imports anything. Under ``OF3_DETERMINISTIC=1`` it applies the
recipe's statements through the tree's one implementation (opt_core.precision.recipe.apply_torch: torch.use_deterministic_algorithms(True),
cuDNN deterministic on / benchmark off; the cuBLAS workspace variable is already exported by the recipe's environment). Without the switch it
does nothing. It installs no finder and touches no OpenFold3 module; the stock proof notes that this file ran, that it loaded torch, and
which core modules it holds (env.DET_CORE_MODULES).
"""
import os as _os
import sys as _sys

if _os.environ.get("OF3_DETERMINISTIC") == "1":
    import torch as _torch
    from opt_core.precision.recipe import apply_torch as _apply_torch

    DET_RECORD = _apply_torch(1, torch=_torch)
    _sys.stderr.write("[openfold3_ob0-opt] det site: deterministic_algorithms=%s cudnn_deterministic=%s cudnn_benchmark=%s cublas_workspace=%s\n" % (
        DET_RECORD["deterministic_algorithms"], DET_RECORD["cudnn_deterministic"], DET_RECORD["cudnn_benchmark"], DET_RECORD["cublas_workspace"]))
