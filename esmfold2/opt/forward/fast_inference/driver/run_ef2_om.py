#!/usr/bin/env python3
"""The kit's constants module (imported by driver/ef2_server.py as ``KIT``):
  NUM_LOOPS, NUM_STEPS, NSAMP, MSA_MAX the fast-inference kit's own fold-call constants (the package's `pred` passes the library's
                                      settings instead: esmfold2_opt/settings.py; configure() cites MSA_MAX in one NOTE line);
  fget(result, name)                  a result attribute as a float (or a short list), None when absent (write_outputs' score row).
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np

NUM_LOOPS, NUM_STEPS, NSAMP, MSA_MAX = 10, 68, 1, 2048


def fget(r, name):
    try:
        v = getattr(r, name, None)
    except Exception:
        return None
    if v is None: return None
    try:
        if hasattr(v, "numpy"): v = v.detach().cpu().numpy() if hasattr(v, "detach") else v.numpy()
        if isinstance(v, np.ndarray):
            return float(v) if v.size == 1 else (v.tolist() if v.size <= 64 else None)
        return float(v)
    except Exception:
        return None
