"""Lever `oneread_mmap`: the stock create_model() with the waste removed, the installed tensors unchanged.

Stock, ``--fp16 true`` (create_model: transformers 4.16.2 modeling_utils.PreTrainedModel.from_pretrained with torch_dtype=float16,
low_cpu_mem_usage=True): (1) torch.load(pytorch_model.bin) ONCE to list the keys, then `del state_dict`; (2) construct the module under
the torch_dtype default (nn.Linear/Embedding/LayerNorm run their own kaiming/normal reset_parameters — no_init_weights only disables the
model's _init_weights); (3) _load_state_dict_into_model_low_mem: every param → meta, torch.load AGAIN, setattr(Parameter(state_dict[k]))
— the checkpoint tensors are installed AS STORED (fp32 for small/medium, fp16 for the others); (4) model.eval().
Stock, ``--fp16 false`` (create_model: from_pretrained(ckpt), no torch_dtype, no low_cpu_mem_usage): the module is constructed under
the process default dtype (float32) and _load_state_dict_into_model COPIES every checkpoint tensor into the constructed one
(Module._load_from_state_dict: param.copy_(stored)) — every parameter and persistent buffer takes the constructed dtype: an exact
widening for the fp16-stored sizes, the same values for the fp32-stored ones.

This loader: construct on the meta device (no allocation, no reset_parameters) under the same default dtype, torch.load ONCE
(mmap=True: the file is paged in exactly once, by the H2D copy), install the SAME tensors by the SAME names — ``fp16``: as stored
(Parameter(state_dict[k]), no cast, no copy); ``fp16=False``: cast to the constructed tensor's dtype (`.to(dtype)`: the copy_'s own
conversion, bit for bit; a no-op for a tensor stored in it) — buffers not in the checkpoint (`attn.bias`, `attn.masked_bias` ARE in the
checkpoint; none are missing) re-created from the config as the stock constructor makes them, eval().  Equal to the stock's load by
construction on the CUDA route, where the generation load uses it (on a CPU box the stock's discarded init draws advance the CPU RNG, so the
stock loader stays there).
"""
import os

import torch


def create_model_oneread(ckpt: str, fp16: bool = True):
    from models.progen.modeling_progen import ProGenForCausalLM
    from models.progen.configuration_progen import ProGenConfig
    config = ProGenConfig.from_pretrained(ckpt)
    path = os.path.join(ckpt, "pytorch_model.bin")
    default_dtype = torch.float16 if fp16 else torch.float32  # the stock's construction dtype (scale_attn is created under it)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(default_dtype)
    try:
        with torch.device("meta"):
            model = ProGenForCausalLM(config)
    finally:
        torch.set_default_dtype(prev)
    sd = torch.load(path, map_location="cpu", mmap=True)  # the stock call (torch.load(file, map_location="cpu")) + mmap: the file is paged in once, by the H2D copy
    # the constructor's one non-parameter tensor attribute (modeling_progen.py L83:
    # `self.scale_attn = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(torch.get_default_dtype())`)
    # was made on meta — remade with the same expression on the CPU under the default dtype the stock constructed it with
    torch.set_default_dtype(default_dtype)
    try:
        for block in model.transformer.h:
            block.attn.scale_attn = torch.sqrt(torch.tensor(block.attn.head_dim, dtype=torch.float32)).to(torch.get_default_dtype())
    finally:
        torch.set_default_dtype(prev)
    missing, unexpected = [], []
    names = {n for n, _ in model.named_parameters()} | {n for n, _ in model.named_buffers()}
    for k, v in sd.items():
        if k not in names:
            unexpected.append(k); continue
        mod_name, _, leaf = k.rpartition(".")
        sub = model.get_submodule(mod_name)
        if leaf in sub._parameters:
            sub._parameters[leaf] = torch.nn.Parameter(v if fp16 else v.to(sub._parameters[leaf].dtype))  # fp16: as the stock low-mem loader installs it; fp32: as param.copy_ leaves it
        else:
            sub._buffers[leaf] = v if fp16 else v.to(sub._buffers[leaf].dtype)
    for n in names:
        if n not in sd:
            missing.append(n)
    assert not missing and not unexpected, f"checkpoint/key mismatch: missing={missing[:5]} unexpected={unexpected[:5]}"
    model.eval()
    return model

