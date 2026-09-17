"""
Affine transformation utilities.

Provides functions for SE(3) transformations and rigid body operations.
"""

# Adapted from OpenFold
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from typing import Tuple, List

# torch.backends.cuda.preferred_linalg_library(backend='magma')


# According to DeepMind, this prevents rotation compositions from being
# computed on low-precision tensor cores. I'm personally skeptical that it
# makes a difference, but to get as close as possible to their outputs, I'm
# adding it.
def rot_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two rotation matrices using explicit element-wise operations.

    This prevents rotation compositions from being computed on low-precision
    tensor cores, following DeepMind's implementation for numerical stability.

    Args:
        a: Rotation matrix of shape [..., 3, 3]
        b: Rotation matrix of shape [..., 3, 3]

    Returns:
        torch.Tensor: Product rotation matrix of shape [..., 3, 3]
    """
    e = ...
    row_1 = torch.stack(
        [
            a[e, 0, 0] * b[e, 0, 0] + a[e, 0, 1] * b[e, 1, 0] + a[e, 0, 2] * b[e, 2, 0],
            a[e, 0, 0] * b[e, 0, 1] + a[e, 0, 1] * b[e, 1, 1] + a[e, 0, 2] * b[e, 2, 1],
            a[e, 0, 0] * b[e, 0, 2] + a[e, 0, 1] * b[e, 1, 2] + a[e, 0, 2] * b[e, 2, 2],
        ],
        dim=-1,
    )
    row_2 = torch.stack(
        [
            a[e, 1, 0] * b[e, 0, 0] + a[e, 1, 1] * b[e, 1, 0] + a[e, 1, 2] * b[e, 2, 0],
            a[e, 1, 0] * b[e, 0, 1] + a[e, 1, 1] * b[e, 1, 1] + a[e, 1, 2] * b[e, 2, 1],
            a[e, 1, 0] * b[e, 0, 2] + a[e, 1, 1] * b[e, 1, 2] + a[e, 1, 2] * b[e, 2, 2],
        ],
        dim=-1,
    )
    row_3 = torch.stack(
        [
            a[e, 2, 0] * b[e, 0, 0] + a[e, 2, 1] * b[e, 1, 0] + a[e, 2, 2] * b[e, 2, 0],
            a[e, 2, 0] * b[e, 0, 1] + a[e, 2, 1] * b[e, 1, 1] + a[e, 2, 2] * b[e, 2, 1],
            a[e, 2, 0] * b[e, 0, 2] + a[e, 2, 1] * b[e, 1, 2] + a[e, 2, 2] * b[e, 2, 2],
        ],
        dim=-1,
    )

    return torch.stack([row_1, row_2, row_3], dim=-2)


def rot_vec_mul(r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Apply rotation matrix to a vector using explicit element-wise operations.

    Args:
        r: Rotation matrix of shape [..., 3, 3]
        t: Vector of shape [..., 3]

    Returns:
        torch.Tensor: Rotated vector of shape [..., 3]
    """
    x = t[..., 0]
    y = t[..., 1]
    z = t[..., 2]
    return torch.stack(
        [
            r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 2] * z,
            r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 2] * z,
            r[..., 2, 0] * x + r[..., 2, 1] * y + r[..., 2, 2] * z,
        ],
        dim=-1,
    )


class T:
    """
    Rigid transformation in SE(3) represented by rotation and translation.

    Represents a rigid body transformation combining a 3x3 rotation matrix
    and a 3D translation vector. Supports composition, inversion, and
    application to points.

    Attributes:
        rots: Rotation matrices of shape [..., 3, 3]
        trans: Translation vectors of shape [..., 3]
    """

    def __init__(self, rots: torch.Tensor, trans: torch.Tensor):
        """
        Initialize a rigid transformation.

        Args:
            rots: Rotation matrices [..., 3, 3] or None for identity rotation
            trans: Translation vectors [..., 3] or None for zero translation

        Raises:
            ValueError: If both rots and trans are None or shapes are incompatible
        """
        self.rots = rots
        self.trans = trans

        if self.rots is None and self.trans is None:
            raise ValueError("Only one of rots and trans can be None")
        elif self.rots is None:
            self.rots = T.identity_rot(
                self.trans.shape[:-1], self.trans.dtype, self.trans.device
            )
        elif self.trans is None:
            self.trans = T.identity_trans(
                self.rots.shape[:-2], self.rots.dtype, self.rots.device
            )

        if (
            self.rots.shape[-2:] != (3, 3)
            or self.trans.shape[-1] != 3
            or self.rots.shape[:-2] != self.trans.shape[:-1]
        ):
            raise ValueError("Incorrectly shaped input")

    def __getitem__(self, index):
        """
        Index into the batch dimensions of the transformation.

        Args:
            index: Index or tuple of indices for batch dimensions

        Returns:
            T: Indexed transformation with preserved rotation and translation structure
        """
        if type(index) != tuple:
            index = (index,)
        return T(
            self.rots[index + (slice(None), slice(None))],
            self.trans[index + (slice(None),)],
        )

    def __eq__(self, obj) -> bool:
        """
        Check equality of two transformations.

        Args:
            obj: Another T object to compare with

        Returns:
            bool: True if rotations and translations are element-wise equal
        """
        return torch.all(self.rots == obj.rots) and torch.all(self.trans == obj.trans)

    def __mul__(self, right: torch.Tensor):
        """
        Scale transformation by a scalar (right multiplication).

        Args:
            right: Scalar tensor to multiply with

        Returns:
            T: Scaled transformation
        """
        rots = self.rots * right[..., None, None]
        trans = self.trans * right[..., None]

        return T(rots, trans)

    def __rmul__(self, left: torch.Tensor):
        """
        Scale transformation by a scalar (left multiplication).

        Args:
            left: Scalar tensor to multiply with

        Returns:
            T: Scaled transformation
        """
        return self.__mul__(left)

    @property
    def shape(self) -> torch.Size:
        """
        Get the batch shape of the transformation.

        Returns:
            torch.Size: Batch dimensions (excluding final rotation/translation dims)
        """
        s = self.rots.shape[:-2]
        return s if len(s) > 0 else torch.Size([1])

    def compose(self, t):
        """
        Compose this transformation with another (self ∘ t).

        Args:
            t: Another T object to compose with

        Returns:
            T: Composed transformation equivalent to applying t then self
        """
        rot_1, trn_1 = self.rots, self.trans
        rot_2, trn_2 = t.rots, t.trans

        rot = rot_matmul(rot_1, rot_2)
        trn = rot_vec_mul(rot_1, trn_2) + trn_1

        return T(rot, trn)

    def apply(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Apply transformation to points (rotate then translate).

        Args:
            pts: Points of shape [..., 3]

        Returns:
            torch.Tensor: Transformed points of shape [..., 3]
        """
        r, t = self.rots, self.trans
        rotated = rot_vec_mul(r, pts)
        return rotated + t

    def invert_apply(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Apply inverse transformation to points.

        Args:
            pts: Points of shape [..., 3]

        Returns:
            torch.Tensor: Inverse-transformed points of shape [..., 3]
        """
        r, t = self.rots, self.trans
        pts = pts - t
        return rot_vec_mul(r.transpose(-1, -2), pts)

    def invert(self):
        """
        Compute the inverse transformation.

        Returns:
            T: Inverse transformation such that self.compose(self.invert()) is identity
        """
        rot_inv = self.rots.transpose(-1, -2)
        trn_inv = rot_vec_mul(rot_inv, self.trans)

        return T(rot_inv, -1 * trn_inv)

    def unsqueeze(self, dim: int):
        """
        Add a singleton dimension to the batch shape.

        Args:
            dim: Dimension to unsqueeze

        Returns:
            T: Transformation with added dimension

        Raises:
            ValueError: If dimension is out of range
        """
        if dim >= len(self.shape):
            raise ValueError("Invalid dimension")
        rots = self.rots.unsqueeze(dim if dim >= 0 else dim - 2)
        trans = self.trans.unsqueeze(dim if dim >= 0 else dim - 1)

        return T(rots, trans)

    @staticmethod
    def identity_rot(
        shape: Tuple,
        dtype: torch.dtype,
        device: torch.device,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        """
        Create identity rotation matrices.

        Args:
            shape: Batch shape for the rotations
            dtype: Data type for the tensor
            device: Device to create tensor on
            requires_grad: Whether to track gradients

        Returns:
            torch.Tensor: Identity rotation matrices of shape [..., 3, 3]
        """
        rots = torch.eye(3, dtype=dtype, device=device, requires_grad=requires_grad)
        rots = rots.view(*((1,) * len(shape)), 3, 3)
        rots = rots.expand(*shape, -1, -1)

        return rots

    @staticmethod
    def identity_trans(
        shape: Tuple,
        dtype: torch.dtype,
        device: torch.device,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        """
        Create zero translation vectors.

        Args:
            shape: Batch shape for the translations
            dtype: Data type for the tensor
            device: Device to create tensor on
            requires_grad: Whether to track gradients

        Returns:
            torch.Tensor: Zero translation vectors of shape [..., 3]
        """
        trans = torch.zeros(
            (*shape, 3), dtype=dtype, device=device, requires_grad=requires_grad
        )
        return trans


_quat_elements = ["a", "b", "c", "d"]
_qtr_keys = [l1 + l2 for l1 in _quat_elements for l2 in _quat_elements]
_qtr_ind_dict = {key: ind for ind, key in enumerate(_qtr_keys)}


def _to_mat(pairs: List[Tuple[str, float]]) -> torch.Tensor:
    """
    Helper function to create quaternion transformation matrix from pairs.

    Args:
        pairs: List of (key, value) pairs for matrix construction

    Returns:
        torch.Tensor: 4x4 matrix
    """
    mat = torch.zeros((4, 4))
    for pair in pairs:
        key, value = pair
        ind = _qtr_ind_dict[key]
        mat[ind // 4][ind % 4] = value

    return mat


_qtr_mat = torch.zeros((4, 4, 3, 3))
_qtr_mat[..., 0, 0] = _to_mat([("aa", 1), ("bb", 1), ("cc", -1), ("dd", -1)])
_qtr_mat[..., 0, 1] = _to_mat([("bc", 2), ("ad", -2)])
_qtr_mat[..., 0, 2] = _to_mat([("bd", 2), ("ac", 2)])
_qtr_mat[..., 1, 0] = _to_mat([("bc", 2), ("ad", 2)])
_qtr_mat[..., 1, 1] = _to_mat([("aa", 1), ("bb", -1), ("cc", 1), ("dd", -1)])
_qtr_mat[..., 1, 2] = _to_mat([("cd", 2), ("ab", -2)])
_qtr_mat[..., 2, 0] = _to_mat([("bd", 2), ("ac", -2)])
_qtr_mat[..., 2, 1] = _to_mat([("cd", 2), ("ab", 2)])
_qtr_mat[..., 2, 2] = _to_mat([("aa", 1), ("bb", -1), ("cc", -1), ("dd", 1)])


def quat_to_rot(quat: torch.Tensor) -> torch.Tensor:  # [*, 4]
    """
    Convert quaternion to rotation matrix.

    Args:
        quat: Quaternion tensor of shape [..., 4] with components [a, b, c, d]

    Returns:
        torch.Tensor: Rotation matrix of shape [..., 3, 3]
    """
    # [*, 4, 4]
    quat = quat[..., None] * quat[..., None, :]

    # [*, 4, 4, 3, 3]
    shaped_qtr_mat = _qtr_mat.view((1,) * len(quat.shape[:-2]) + (4, 4, 3, 3))
    quat = quat[..., None, None] * shaped_qtr_mat.to(quat.device)

    # [*, 3, 3]
    return torch.sum(quat, dim=(-3, -4))


def rot_to_quat(
    rot: torch.Tensor,
) -> torch.Tensor:
    """
    Convert rotation matrix to quaternion representation.

    Args:
        rot: Rotation matrix of shape [..., 3, 3]

    Returns:
        torch.Tensor: Quaternion of shape [..., 4]

    Raises:
        ValueError: If rotation matrix is not 3x3
    """
    if rot.shape[-2:] != (3, 3):
        raise ValueError("Input rotation is incorrectly shaped")

    rot = [[rot[..., i, j] for j in range(3)] for i in range(3)]
    [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = rot

    k = [
        [
            xx + yy + zz,
            zy - yz,
            xz - zx,
            yx - xy,
        ],
        [
            zy - yz,
            xx - yy - zz,
            xy + yx,
            xz + zx,
        ],
        [
            xz - zx,
            xy + yx,
            yy - xx - zz,
            yz + zy,
        ],
        [
            yx - xy,
            xz + zx,
            yz + zy,
            zz - xx - yy,
        ],
    ]

    k = (1.0 / 3.0) * torch.stack([torch.stack(t, dim=-1) for t in k], dim=-2)

    #_, vectors = torch.linalg.eigh(k)
    vectors = batched_eigh(k)
    return vectors[..., -1]

def batched_eigh(k: torch.Tensor) -> torch.Tensor:
    '''
    fix to avoid overflow when calling cusolverDnXsyevBatched_bufferSize

    cfr. https://github.com/pytorch/pytorch/issues/166004
    '''
    max_allowed_4x4 = 31291
    if k[...,0,0].nelement() <= max_allowed_4x4:
        _, vectors = torch.linalg.eigh(k)
        return vectors
    subvectors = []
    for k_small in torch.split(k.reshape(-1, 4, 4), max_allowed_4x4, dim=0):
        _, vectors_small = torch.linalg.eigh(k_small)
        subvectors.append(vectors_small)
    vectors = torch.cat(subvectors, dim=0)
    return vectors.reshape(*k.shape)
