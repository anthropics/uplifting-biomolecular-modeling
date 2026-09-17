# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md


import dataclasses
from typing import Self

import torch


def unstack(value: torch.Tensor, dim: int = -1) -> list[torch.Tensor]:
    return [
        torch.squeeze(v, dim=dim)
        for v in torch.chunk(value, value.shape[dim], dim=dim)
    ]


@dataclasses.dataclass
class Vec3Array:
    """Vec3Array in 3 dimensional Space implemented as struct of arrays.
    """

    x: torch.Tensor
    y: torch.Tensor
    z: torch.Tensor

    def __add__(self, other: Self) -> Self:
        return Vec3Array(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Self) -> Self:
        return Vec3Array(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other: float | torch.Tensor) -> Self:
        return Vec3Array(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other: float | torch.Tensor) -> Self:
        return self * other

    def __truediv__(self, other: float | torch.Tensor) -> Self:
        return Vec3Array(self.x / other, self.y / other, self.z / other)

    def __neg__(self) -> Self:
        return Vec3Array(-self.x, -self.y, -self.z)

    def __pos__(self) -> Self:
        return Vec3Array(self.x, self.y, self.z)

    @classmethod
    def from_array(cls, array):
        return cls(*unstack(array))

    def cross(self, other: Self) -> Self:
        """Compute cross product between 'self' and 'other'."""
        new_x = self.y * other.z - self.z * other.y
        new_y = self.z * other.x - self.x * other.z
        new_z = self.x * other.y - self.y * other.x
        return Vec3Array(new_x, new_y, new_z)

    def dot(self, other: Self) -> torch.Tensor:
        """Compute dot product between 'self' and 'other'."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def norm(self, epsilon: float = 1e-6) -> torch.Tensor:
        """Compute Norm of Vec3Array, clipped to epsilon."""
        # To avoid NaN on the backward pass, we must use maximum before the sqrt
        norm2 = self.dot(self)
        if epsilon:
            norm2 = torch.maximum(norm2, torch.tensor(
                epsilon**2, dtype=norm2.dtype, device=norm2.device))
        return torch.sqrt(norm2)

    def normalized(self, epsilon: float = 1e-6) -> Self:
        """Return unit vector with optional clipping."""
        return self / self.norm(epsilon)


@dataclasses.dataclass
class Rot3Array:
    """Rot3Array Matrix in 3 dimensional Space implemented as struct of arrays."""

    xx: torch.Tensor
    xy: torch.Tensor
    xz: torch.Tensor
    yx: torch.Tensor
    yy: torch.Tensor
    yz: torch.Tensor
    zx: torch.Tensor
    zy: torch.Tensor
    zz: torch.Tensor

    def inverse(self) -> Self:
        """Returns inverse of Rot3Array."""
        return Rot3Array(
            *(self.xx, self.yx, self.zx),
            *(self.xy, self.yy, self.zy),
            *(self.xz, self.yz, self.zz),
        )

    def apply_to_point(self, point: Vec3Array) -> Vec3Array:
        """Applies Rot3Array to point."""
        return Vec3Array(
            self.xx * point.x + self.xy * point.y + self.xz * point.z,
            self.yx * point.x + self.yy * point.y + self.yz * point.z,
            self.zx * point.x + self.zy * point.y + self.zz * point.z,
        )

    @classmethod
    def from_two_vectors(cls, e0: Vec3Array, e1: Vec3Array) -> Self:
        """Construct Rot3Array from two Vectors.

        Rot3Array is constructed such that in the corresponding frame 'e0' lies on
        the positive x-Axis and 'e1' lies in the xy plane with positive sign of y.

        Args:
        e0: Vector
        e1: Vector

        Returns:
        Rot3Array
        """
        # Normalize the unit vector for the x-axis, e0.
        e0 = e0.normalized()
        # make e1 perpendicular to e0.
        c = e1.dot(e0)
        e1 = (e1 - c * e0).normalized()
        # Compute e2 as cross product of e0 and e1.
        e2 = e0.cross(e1)
        # pytype: disable=wrong-arg-count  # trace-all-classes
        return cls(e0.x, e1.x, e2.x, e0.y, e1.y, e2.y, e0.z, e1.z, e2.z)


@dataclasses.dataclass
class Rigid3Array:
    """Rigid Transformation, i.e. element of special euclidean group."""

    rotation: Rot3Array
    translation: Vec3Array

    def inverse(self) -> Self:
        """Return Rigid3Array corresponding to inverse transform."""
        inv_rotation = self.rotation.inverse()
        inv_translation = inv_rotation.apply_to_point(-self.translation)
        return Rigid3Array(inv_rotation, inv_translation)

    def apply_to_point(self, point: Vec3Array) -> Vec3Array:
        """Apply Rigid3Array transform to point."""
        return self.rotation.apply_to_point(point) + self.translation
