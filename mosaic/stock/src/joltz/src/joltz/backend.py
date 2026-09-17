# See `joltz.py` for an introduction.

from dataclasses import fields
from functools import singledispatch

import time
import einops
import equinox as eqx
import jax
import torch
from jax import numpy as jnp
from jaxtyping import Array, Float, Int
from functools import partial
import numpy as np
from jax import tree


@singledispatch
def from_torch(x):
    raise NotImplementedError(f"from_torch not implemented for {type(x)}: {x}")


# basic types
from_torch.register(torch.Tensor, lambda x: np.array(x.detach()))
from_torch.register(int, lambda x: x)
from_torch.register(float, lambda x: x)
from_torch.register(bool, lambda x: x)
from_torch.register(type(None), lambda x: x)
from_torch.register(tuple, lambda x: tuple(map(from_torch, x)))
from_torch.register(dict, lambda x: {k: from_torch(v) for k, v in x.items()})
from_torch.register(torch.nn.ReLU, lambda _: jax.nn.relu)
from_torch.register(torch.nn.Sigmoid, lambda _: jax.nn.sigmoid)
from_torch.register(torch.nn.SiLU, lambda _: jax.nn.silu)
from_torch.register(torch.nn.ModuleList, lambda x: [from_torch(m) for m in x])


class AbstractFromTorch(eqx.Module):
    """
    Default implementation of `from_torch` for equinox modules.
    This checks that the fields of the equinox module are present in the torch module and constructs the equinox module from the torch module by recursively calling `from_torch` on the children of the torch module.
    Allows for missing fields in the torch module if the corresponding field in the equinox module is optional.

    """

    @classmethod
    def from_torch(cls, model: torch.nn.Module):
        # assemble arguments to `cls` constructor from `model`

        field_to_type = {field.name: field.type for field in fields(cls)}
        kwargs = {
            child: from_torch(child_module)
            for child, child_module in model.named_children()
        } | {
            parameter_name: from_torch(parameter)
            for parameter_name, parameter in model.named_parameters(recurse=False)
        }

        # add fields that are not child_modules or parameters
        for field_name, field_type in field_to_type.items():
            if not hasattr(model, field_name):
                if not isinstance(None, field_type):
                    raise ValueError(
                        f"Field {field_name} for {cls} is not optional but is missing from torch model {model}"
                    )
                else:
                    kwargs[field_name] = None
            else:
                kwargs[field_name] = from_torch(getattr(model, field_name))

        # check we're not passing any additional properties
        torch_not_equinox = kwargs.keys() - field_to_type.keys()
        if torch_not_equinox:
            raise ValueError(
                f"Properties in torch model not found in equinox module {cls}: {torch_not_equinox}"
            )

        return cls(**kwargs)


def register_from_torch(torch_module_type):
    """Class decorator to register an equinox module for conversion from a torch module."""

    def decorator(cls):
        from_torch.register(torch_module_type, cls.from_torch)
        return cls

    return decorator


# this isn't very jax-y
def _vmap(f, tensor, *args):
    for _ in range(len(tensor.shape) - 1):
        f = jax.vmap(f)
    return f(tensor, *args)


def vmap_to_last_dimension(f):
    return partial(_vmap, f)


@register_from_torch(torch.nn.Linear)
class Linear(eqx.Module):
    """Linear layer that matches pytorch semantics"""

    weight: Float[Array, "Out In"]
    bias: Float[Array, "Out"] | None

    def __call__(self, x: Float[Array, "... In"]) -> Float[Array, "... Out"]:
        o = einops.einsum(x, self.weight, "... In, Out In -> ... Out")
        if self.bias is not None:
            o = o + jnp.broadcast_to(self.bias, x.shape[:-1] + (self.bias.shape[-1],))
        return o

    @staticmethod
    def from_torch(l: torch.nn.Linear):
        return Linear(weight=from_torch(l.weight), bias=from_torch(l.bias))



@register_from_torch(torch.nn.modules.linear.Identity)
class Identity(eqx.Module):

    def __call__(self, x: Float[Array, "... In"]) -> Float[Array, "... Out"]:
        return x

    @staticmethod
    def from_torch(_: torch.nn.modules.linear.Identity):
        return Identity()


@register_from_torch(torch.nn.LayerNorm)
class LayerNorm(eqx.Module):
    """LayerNorm that matches pytorch semantics"""

    weight: Float[Array, "Out"] | None
    bias: Float[Array, "Out"] | None
    eps: float

    def __call__(self, x: Float[Array, "... Out"]) -> Float[Array, "... Out"]:
        ln = eqx.nn.LayerNorm(
            shape=x.shape[-1],
            eps=self.eps,
            use_weight=self.weight is not None,
            use_bias=self.bias is not None,
        )
        ln = eqx.tree_at(
            lambda l: (l.weight, l.bias),
            ln,
            (self.weight, self.bias),
            is_leaf=lambda x: x is None,
        )

        return vmap_to_last_dimension(ln)(x)

    @staticmethod
    def from_torch(l: torch.nn.LayerNorm):
        return LayerNorm(
            weight=from_torch(l.weight), bias=from_torch(l.bias), eps=l.eps
        )


@register_from_torch(torch.nn.Sequential)
class Sequential(eqx.Module):
    _modules: dict[
        str, AbstractFromTorch
    ]  # IMHO this is a fairly wild design choice, but this is really how pytorch works.

    def __call__(self, x):
        for idx in range(len(self._modules)):
            x = self._modules[str(idx)](x)
        return x

    @staticmethod
    def from_torch(module: torch.nn.Sequential):
        return Sequential(_modules=from_torch(module._modules))

@register_from_torch(torch.nn.modules.sparse.Embedding)
class Embedding(eqx.Module):
    weight: Float[Array, "V D"]

    def __call__(self, tokens: Int[Array, "..."]) -> Float[Array, "... D"]:
        return self.weight[tokens]

    @staticmethod
    def from_torch(m: torch.nn.modules.sparse.Embedding):
        return Embedding(weight=from_torch(m.weight))


# Useful for testing
class TestModule(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.mod = module
        self.j_m = eqx.filter_jit(from_torch(self.mod))

    def forward(self, *args, **kwargs):
        torch_start = time.time()
        torch_output = self.mod(*args, **kwargs)
        torch_end = time.time()

        jax_start = time.time()
        with jax.default_matmul_precision("float32"):
            jax_output = self.j_m(*from_torch(args), **from_torch(kwargs))
        tree.map(lambda v: v.block_until_ready(), jax_output)
        jax_end = time.time()

        errors = tree.map(
            lambda a, b: jnp.abs(jnp.array(a) - b).max()
            if isinstance(b, jnp.ndarray)
            else None,
            torch_output,
            jax_output,
            is_leaf=eqx.is_inexact_array,
        )
        print(f"max abs error {type(self.mod)}: ", errors)
        print(
            f"torch time: {torch_end - torch_start : .3f}s, jax time: {jax_end - jax_start : .3f}s"
        )
        return torch_output
