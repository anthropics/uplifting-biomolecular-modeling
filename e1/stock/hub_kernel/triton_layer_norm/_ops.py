import torch
ops = torch.ops._triton_layer_norm_9b61b27_dirty

def add_op_namespace_prefix(op_name: str):
    """
    Prefix op by namespace.
    """
    return f"_triton_layer_norm_9b61b27_dirty::{op_name}"