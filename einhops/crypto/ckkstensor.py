import math
import numpy
import torch
import desilofhe
from dataclasses import dataclass
from .backend import (
    fhe_encode, fhe_encrypt, fhe_decrypt, fhe_decode,
    fhe_add, fhe_mul, fhe_rotate, fhe_level_down,
    SLOT_COUNT, MAX_LEVEL
)

@dataclass
class CKKSTensor:
    slots: desilofhe.Ciphertext | torch.Tensor
    shape: torch.Size
    fhe_shape: torch.Size
    ndim: int
    level: int

    def __str__(self):
        return f"CKKSTensor(slots={type(self.slots)}, shape={self.shape}, fhe_shape={self.fhe_shape}, ndim={self.ndim}, level={self.level})"
    
    def __repr__(self):
        return self.__str__()
   
    def __add__(self, other):
        self._validate_shape(other)

        if isinstance(other, torch.Tensor):
            packed = tensor_to_packed_vector(other)
            other = CKKSTensor(packed, other.shape, other.shape, other.ndim, self.level)

        res_add = fhe_add(self.slots, other.slots)
        return CKKSTensor(res_add, self.shape, self.fhe_shape, self.ndim, self.level)
    
    def __mul__(self, other):
        self._validate_shape(other)

        if isinstance(other, torch.Tensor):
            packed = tensor_to_packed_vector(other)
            other = CKKSTensor(packed, other.shape, other.shape, other.ndim, self.level)

        res_mul = fhe_mul(self.slots, other.slots)
        new_level = res_mul.level if isinstance(res_mul, desilofhe.Ciphertext) else self.level - 1
        return CKKSTensor(res_mul, self.shape, self.fhe_shape, self.ndim, new_level)

    def rotate(self, steps):
        """Rotate slots by given number of steps"""

        res_rot = fhe_rotate(self.slots, steps)
        return CKKSTensor(res_rot, self.shape, self.fhe_shape, self.ndim, self.level)

    def level_down(self, level):
        self.slots = fhe_level_down(self.slots, level)
        self.level = level
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def _validate_shape(self, other):
        assert self.shape == other.shape, "CKKSTensors must have the same shape"
    
# functions for translating arbitrary tensors -> FHE tensors -> CKKS vectors
def next_power_of_two(n):
    """
    Return the smallest power of two >= n.
    >>> next_power_of_two(9)
    16
    """

    if n < 1:
        raise ValueError("n must be >= 1")
    return 1 << (math.ceil(math.log2(n)))

def power_of_two_dims(sizes):
    """
    Pads each dimension to the closest power of two
    >>> power_of_two_dims(torch.Size([2, 3]))
    [2, 4]
    >>> power_of_two_dims(torch.Size([3, 17]))
    [4, 32]
    """

    return torch.Size(list(map(next_power_of_two, sizes)))


def tensor_to_packed_vector(clear_tensor):
    """
    Converts a clear tensor to a packed vector.

    Args:
        clear_tensor (torch.Tensor): The clear tensor to convert.

    Returns:
        torch.Tensor: The packed vector.
    """

    clear_shape = clear_tensor.shape
    fhe_shape = power_of_two_dims(clear_shape)
    assert fhe_shape.numel() <= SLOT_COUNT, "FHE tensor must be smaller than SLOT_COUNT"

    fhe_tensor = torch.zeros(fhe_shape)
    fhe_idx = tuple(slice(0, s) for s in clear_shape)
    fhe_tensor[fhe_idx] = clear_tensor

    slots = torch.zeros(SLOT_COUNT)
    slots[:fhe_tensor.numel()] = fhe_tensor.flatten()

    return slots


def tensor_to_fhe(clear_tensor, level=MAX_LEVEL):
    """
    Converts a clear tensor to a CKKSTensor.

    Args:
        clear_tensor (torch.Tensor): The clear tensor to convert.
        level (int): The level to encrypt the tensor to.

    Returns:
        CKKSTensor: The FHE tensor
    """

    clear_shape = clear_tensor.shape
    fhe_shape = power_of_two_dims(clear_shape)
    assert fhe_shape.numel() <= SLOT_COUNT, "FHE tensor must be smaller than SLOT_COUNT"

    fhe_tensor = torch.zeros(fhe_shape)
    fhe_idx = tuple(slice(0, s) for s in clear_shape)
    fhe_tensor[fhe_idx] = clear_tensor

    slots = torch.zeros(SLOT_COUNT)
    slots[:fhe_tensor.numel()] = fhe_tensor.flatten()

    encrypted_tensor = fhe_encrypt(slots, level)

    return CKKSTensor(
        slots=encrypted_tensor,
        shape=clear_shape,
        fhe_shape=fhe_shape,
        ndim=len(clear_shape),
        level=level
    )

def fhe_to_tensor(fhe_tensor):
    """
    Performs both decrypting and decoding.

    Args:
        fhe_tensor (CKKSTensor): The FHE tensor to convert.

    Returns:
        torch.Tensor: The clear tensor.
    """

    ptxt_tensor = fhe_decrypt(fhe_tensor.slots)
    clear_tensor = fhe_decode(ptxt_tensor)
    if isinstance(clear_tensor, numpy.ndarray):
        clear_tensor = torch.from_numpy(clear_tensor).to(torch.float32)

    # edge-case: zero (unit tensors) or one dimensional (1-d arrays)
    if fhe_tensor.ndim == 0:
        return clear_tensor[0]
    elif fhe_tensor.ndim == 1:
        return clear_tensor[:fhe_tensor.shape.numel()]

    clear_shape = fhe_tensor.shape
    clear_strides = torch.tensor(torch.empty(fhe_tensor.fhe_shape).stride())
    clear_tensor_indices = torch.cartesian_prod(*[torch.arange(shape) for shape in clear_shape])
    clear_indices = torch.sum(clear_tensor_indices * clear_strides, dim=1)
    clear_tensor = clear_tensor.flatten()[clear_indices]
    clear_tensor = clear_tensor.reshape(clear_shape)
    return clear_tensor
