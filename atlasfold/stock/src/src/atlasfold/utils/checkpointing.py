# Copyright 2026 Seonghwan Seo (KAIST), MIT LICENSE
# Copyright 2025 AlQuraishi Laboratory (LICENSE-2.0)

from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.utils.checkpoint

checkpoint_fn = torch.utils.checkpoint.checkpoint

BLOCK_ARG = Any
BLOCK_ARGS = Sequence[BLOCK_ARG]
LAYER_ARG = Any
LAYER_ARGS = dict[str, Sequence[LAYER_ARG]]
STATIC_KWARG = Any
STATIC_KWARGS = dict[str, STATIC_KWARG]


def wrap(a: BLOCK_ARG) -> BLOCK_ARGS:
    return (a,) if type(a) is not tuple else a


def checkpoint_blocks(
    blocks: list[Callable],
    args: BLOCK_ARGS,
    blocks_per_ckpt: int | None,
    static_args: STATIC_KWARGS | None = None,
    layer_args: LAYER_ARGS | None = None,
    use_reentrant: bool = False,
) -> BLOCK_ARGS:
    """
    Chunk a list of blocks and run each chunk with activation
    checkpointing. We define a "block" as a callable whose only inputs are
    the outputs of the previous block.

    Parameters
    ----------
    blocks: list[Callable]
        List of blocks to execute sequentially.
    args: Sequence[Any]
        Tuple of arguments for the first block.
    blocks_per_ckpt: int | None
        Size of each chunk. A higher value corresponds to fewer
        checkpoints, and trades memory for speed. If None, no checkpointing
        is performed.
    static_args: dict[str, Any] | None
        Optional dictionary mapping argument names to static keyword arguments
        that are passed to all blocks. If not provided, no static keyword
        arguments will be passed to the blocks.
    layer_args: dict[str, Sequence[Any]] | None
        Optional dictionary mapping argument names to lists of arguments for
        each block. If provided, the arguments for block i will be passed as
        keyword arguments to block i. If not provided, no keyword arguments
        will be passed to the blocks.
    use_reentrant: bool
        Whether to use reentrant checkpointing.

    Returns:
        The output of the final block
    """
    # Add static keyword arguments if not provided
    static_args = static_args or {}

    # Add layer arguments if not provided
    layer_args = layer_args or {}
    for k, v in layer_args.items():
        if len(v) != len(blocks):
            raise ValueError(
                f"Length of layer arguments for {k} must match number of blocks"
            )
    layer_kwargs_list = [
        {k: v[i] for k, v in layer_args.items()} for i in range(len(blocks))
    ]

    # Add block indices
    blocks: list[tuple[int, Callable]] = list(enumerate(blocks))

    def exec(b: list[tuple[int, Callable]], a: BLOCK_ARGS) -> BLOCK_ARGS:
        for i, _b in b:
            la = layer_kwargs_list[i]
            a = wrap(_b(*a, **la, **static_args))
        return a

    if not torch.is_grad_enabled() or blocks_per_ckpt is None:
        return exec(blocks, args)

    if blocks_per_ckpt < 1:
        raise ValueError("blocks_per_ckpt must be at least 1")

    def chunker(s, e):
        return lambda *a: exec(blocks[s:e], a)

    for s in range(0, len(blocks), blocks_per_ckpt):
        e = s + blocks_per_ckpt
        args = checkpoint_fn(chunker(s, e), *args, use_reentrant=use_reentrant)
        args = wrap(args)

    return args
