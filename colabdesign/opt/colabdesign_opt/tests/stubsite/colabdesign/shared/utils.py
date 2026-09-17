import copy


def copy_dict(d):
    return copy.deepcopy(d)


def to_float(x):
    """floats out of numpy scalars / arrays, recursively (colabdesign.shared.utils)."""
    if isinstance(x, dict):
        return {k: to_float(v) for k, v in x.items()}
    if hasattr(x, "tolist"):
        return x.tolist()
    return float(x) if isinstance(x, (int, float)) else x


def to_list(x):
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]
