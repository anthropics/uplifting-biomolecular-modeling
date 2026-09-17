import torch
import logging

log = logging.getLogger(__name__)


def get_dim_for_local_rank(dim: int, world_size: int, local_rank: int, multiple_of: int = 1) -> int:
    """Get the dim for the local rank derived from splitting dim on world_size processes.

    The split may not be even across the world_size processes.
    """
    multiple = dim // multiple_of
    div = multiple // world_size
    mod = multiple % world_size
    local_multiple = div + int(local_rank < mod)
    return local_multiple * multiple_of


def grab_first_if_tuple(x):
    if x.__class__.__name__ == "tuple":
        return x[0]
    else:
        return x


def interleave(z_pre):
    if len(z_pre.shape) == 3:  # non-cached
        x1 = z_pre[:, 0::3, :]
        x2 = z_pre[:, 1::3, :]
        v = z_pre[:, 2::3, :]
        z_pre = torch.cat([x1, x2, v], dim=1)
        return z_pre
    else:
        x1 = z_pre[..., 0::3]
        x2 = z_pre[..., 1::3]
        v = z_pre[..., 2::3]
        z_pre = torch.concat([x1, x2, v], dim=-1)
        return z_pre


def column_split(x, num_heads, head_size):
    """Split a tensor with `num_heads` alongside the head dimension, instead of
    across heads. Fixed to three projections
    """
    # FIXME: merge cases
    if len(x.shape) == 2:
        x_reshaped = x.reshape(
            x.shape[0],
            num_heads,
            3 * head_size,
        )

        x2, x1, v = (
            x_reshaped[..., :head_size],
            x_reshaped[..., head_size : 2 * head_size],
            x_reshaped[..., 2 * head_size :],
        )
        x2, x1, v = (
            x2.reshape(x2.shape[0], -1),
            x1.reshape(x1.shape[0], -1),
            v.reshape(v.shape[0], -1),
        )
        return x2, x1, v
    else:
        x = x.reshape(
            x.shape[0],
            num_heads,
            3 * head_size,
            x.shape[2],
        )
        x2, x1, v = (
            x[:, :, :head_size],
            x[
                :,
                :,
                head_size : 2 * head_size,
            ],
            x[:, :, 2 * head_size :],
        )
        x2, x1, v = (
            x2.reshape(x2.shape[0], -1, x2.shape[-1]),
            x1.reshape(x1.shape[0], -1, x1.shape[-1]),
            v.reshape(v.shape[0], -1, v.shape[-1]),
        )
        return x2, x1, v


def load_checkpoint(model, checkpoint_path):
    if checkpoint_path is None:
        log.warning("Using random weights (dry-run)")
        return
    log.info(f"Loading {checkpoint_path}")

    # We must allowlist BytesIO, as fp8-enabled checkpoints store this type
    # in Transformer Engine layers' _extra keys. If not, weights_only=True
    # will not be happy.
    import io

    safe_globals = [io.BytesIO]
    try:
        from transformer_engine.common.recipe import DelayedScaling, Format, _FormatHelper
        safe_globals.extend([DelayedScaling, Format, _FormatHelper])
    except ImportError:
        pass

    torch.serialization.add_safe_globals(safe_globals)

    with torch.inference_mode():
        state = torch.load(
            checkpoint_path,
            # Make sure we override device location that is specified in the
            # checkpoint dictionary (e.g. checkpoints may have "cuda:0"
            # as a location for all layers, which then wouldn't work for
            # multi-GPU case.)
            map_location="cpu",
            # This is an optimization: with that, we don't actually read
            # whole checkpoints dictionary from disk to CPU memory in one
            # go; instead, pytorch would only load relevant layers to CPU
            # memory when we are about to copy them to GPU.
            mmap=True,
            # Make sure PyTorch is not issuing a warning regarding potential
            # security issues.
            weights_only=True,
        )
        model.to_bfloat16_except_pr_lc(to_float32=True)

        model.custom_load_state_dict(state)

        model.to_bfloat16_except_pr_lc()


def move_to_device(module, device):
    """Recursively moves all parameters and buffers to the specified device."""
    for child in module.children():
        move_to_device(child, device)

    for param in module.parameters(recurse=False):
        if param.device != device:
            param.data = param.data.to(device)

    for buf in module.buffers(recurse=False):
        if buf.device != device:
            buf.data = buf.data.to(device)

    module.to(device)


def fixup_fp8_extra_states(module):
    """Recursively fixes device location of TE's Linear fp8 extra states.
    No-op when Transformer Engine is not installed."""
    if not hasattr(module, "fp8_meta"):
        for child in module.children():
            fixup_fp8_extra_states(child)
        return

    for child in module.children():
        fixup_fp8_extra_states(child)

    log.debug(f"Reloading fp8 extra state to a proper device for {module}")

    # Must set to false, otherwise set_extra_state will be no-op
    module.fp8_meta_tensors_initialized = False

    # TE Linear uses default "cuda" device to load extra state, which causes
    # trouble when the layer is moved to another GPU. Instead, this is how
    # TE Linear should load extra_state: using parameters' device.
    device = next(module.parameters()).device
    with torch.cuda.device(device):
        module.set_extra_state(module.get_extra_state())

    # Make sure we actually fixed everything we wanted.
    for k in ["scaling_fwd", "scaling_bwd"]:
        for attr in ["amax_history", "scale"]:
            tensor = getattr(module.fp8_meta[k], attr)
            assert tensor.device == device, (k, tensor, device)


def fixup_te_workspace():
    """TE uses single workspace tensor for all calls, disregarding that inputs
    may be on separate GPUs. This patches TE's Linear module to use per-device
    workspaces. No-op when Transformer Engine is not installed."""
    try:
        import transformer_engine  # noqa: F401
    except ImportError:
        return

    from functools import lru_cache

    @lru_cache
    def te_cublas_get_workspace_per_device(device):
        log.info(f"Fixup applied: Allocating cublas workspace for {device=}")
        import transformer_engine.pytorch.module.base as tebase

        with torch.cuda.device(device):
            tebase._cublas_workspace = None  # Force get_workspace() to reallocate tensor
            return tebase.get_workspace()

    def get_workspace():
        return te_cublas_get_workspace_per_device(torch.cuda.current_device())

    import transformer_engine.pytorch.module.linear as telinear

    telinear.get_workspace = get_workspace


def get_init_from_string(init_str):
    if type(init_str) == str:
        if init_str == "torch.nn.init.zeros_":
            return torch.nn.init.zeros_
        elif init_str == "torch.nn.init.xavier_uniform_":
            return torch.nn.init.xavier_uniform_
        elif init_str == "torch.nn.init.xavier_normal_":
            return torch.nn.init.xavier_normal_
        else:
            raise ValueError(f"Unrecognized init {init_str}")


def print_rank_0(message, debug=False, end="\n"):
    """Print from rank 0 only."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True, end=end)
    else:
        print(message, flush=True, end=end)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


class Lambda(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class VocabUtility:
    """Split the vocabulary into `world_size` chunks amd return the
    first and last index of the vocabulary belonging to the `rank`
    partition: Note that indices in [first, last]"""

    @staticmethod
    def vocab_range_from_per_partition_vocab_size(per_partition_vocab_size, rank, world_size):
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(global_vocab_size, rank, world_size):
        per_partition_vocab_size = divide(global_vocab_size, world_size)
        return VocabUtility.vocab_range_from_per_partition_vocab_size(per_partition_vocab_size, rank, world_size)
