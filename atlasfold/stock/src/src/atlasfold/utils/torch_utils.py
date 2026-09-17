import torch


def add(x: torch.Tensor, y: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """Add two tensors together, optionally in-place.

    Parameters
    ----------
    x : torch.Tensor
        The first tensor to add.
    y : torch.Tensor
        The second tensor to add.
    inplace : bool, optional
        Whether to perform the addition in-place on `x`, by default False.

    Returns
    -------
    torch.Tensor
        The result of adding `x` and `y`.
    """
    if inplace:
        return x.add_(y)
    else:
        return x + y


def expand_dim(
    tensor: torch.Tensor,
    n_repeat: int,
    dim: int,
    add_new_dim: bool = True,
) -> torch.Tensor:
    """Expanding a tensor with new shape"""
    if add_new_dim:
        tensor = tensor.unsqueeze(dim)
    shape = tensor.shape
    to_expand = [-1] * len(shape)
    to_expand[dim] = n_repeat
    return tensor.expand(*to_expand)


def repeat_dim(
    tensor: torch.Tensor,
    n_repeat: int,
    dim: int,
    add_new_dim: bool = True,
) -> torch.Tensor:
    """Repeat a tensor with new shape"""
    if add_new_dim:
        tensor = tensor.unsqueeze(dim)
    shape = tensor.shape
    to_repeat = [1] * len(shape)
    to_repeat[dim] = n_repeat
    return tensor.repeat(*to_repeat)


def pad_dim(
    tensor: torch.Tensor,
    dim: int,
    max_len: int,
    pad_value: float | int | bool = 0.0,
) -> torch.Tensor:
    """Pad a tensor with new shape"""
    current_len = tensor.shape[dim]
    if current_len > max_len:
        raise ValueError(
            f"Cannot pad tensor of shape {tensor.shape} to max_len {max_len} "
            f"along dim {dim}"
        )
    if current_len == max_len:
        return tensor
    shape = list(tensor.shape)
    shape[dim] = max_len - current_len
    pad_tensor = torch.full(shape, pad_value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad_tensor], dim=dim)


def gather_dim(
    tensor: torch.Tensor,
    dim: int,
    index: torch.Tensor,
) -> torch.Tensor:
    """Helper function to gather values from `tensor` along `dim` using `index`."""
    dim = dim % tensor.ndim  # Normalize negative dims to positive
    assert tensor.ndim == index.ndim
    expand_shape = [-1 if i == dim else d for i, d in enumerate(tensor.shape)]

    mask = index < 0  # Mask for out-of-bounds indices
    index_clipped = index.clamp(min=0)
    out = tensor.gather(dim, index=index_clipped.expand(expand_shape))
    return out.masked_fill_(mask, 0)


def index_select_dim(
    tensor: torch.Tensor,
    dim: int,
    index: torch.Tensor,
) -> torch.Tensor:
    """Helper function to index select values from `tensor` along `dim` using `index`."""
    dim = dim % tensor.ndim
    # Pad the index with trailing dimensions until ndims match
    while index.ndim < tensor.ndim:
        index = index.unsqueeze(-1)
    return gather_dim(tensor, dim, index).squeeze(dim)


def get_one_hot_from_boundaries(
    tensor: torch.Tensor, bounds: torch.Tensor
) -> torch.Tensor:
    """Get one-hot encoding of a tensor based on the provided boundaries.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor to be one-hot encoded of shape (*,).
    bounds : torch.Tensor
        A tensor containing the boundaries of the bins for one-hot encoding
        of shape (num_bins - 1,).

    Returns
    -------
    torch.Tensor
        A one-hot encoded tensor of shape (*, num_bins).
    """
    num_bins = bounds.shape[0] + 1
    indices = (tensor[..., None] > bounds).sum(dim=-1).long()
    return torch.nn.functional.one_hot(indices, num_classes=num_bins)


def get_one_hot_from_bins(
    tensor: torch.Tensor, bin_centers: torch.Tensor
) -> torch.Tensor:
    """Get one-hot encoding of a tensor based on the provided bins.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor to be one-hot encoded of shape (*,).
    bin_centers : torch.Tensor
        A tensor containing the centers of the bins for one-hot encoding
        of shape (num_bins,).

    Returns
    -------
    torch.Tensor
        A one-hot encoded tensor of shape (*, num_bins).
    """
    num_bins = bin_centers.shape[0]
    d = torch.abs(tensor[..., None] - bin_centers)  # [*, num_bins]
    indices = d.argmin(dim=-1)  # [*]
    return torch.nn.functional.one_hot(indices, num_classes=num_bins)


def get_context_dtype(device_type: str | None = None) -> torch.dtype:
    """Get the current context dtype for autocast."""
    if device_type is None:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"

    if torch.is_autocast_enabled(device_type):
        return torch.get_autocast_dtype(device_type)
    else:
        return torch.float32
