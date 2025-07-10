import math
import torch
import logging
from .crypto.backend import SLOT_COUNT
from .crypto.bsgs import bsgs_matrix_mult

logger = logging.getLogger("einhops")

def expand_dimensions_torch(operand, src_dim, dst_dim, dim_sizes):
    """
    Broadcasts a torch tensor from src_dim to dst_dim.

    Args:
        operand (torch.Tensor): The tensor to expand.
        src_dim (list): The dimensions of the source tensor.
        dst_dim (list): The dimensions of the destination tensor.
        dim_sizes (dict): The dimension sizes.

    Returns:
        torch.Tensor: Broadcasted tensor.
    """

    logger.debug(f'Expanding dimensions (torch.Tensor): {src_dim} -> {dst_dim}')

    # no expansion needed
    if src_dim == dst_dim:
        return operand

    # get the missing dimensions
    missing_dim = set(dst_dim) - set(src_dim)
    expanded_src_dim = src_dim + "".join(missing_dim)

    operand = operand.view(*operand.shape, *([1]*len(missing_dim)))            # add singleton dims
    operand = torch.einsum(f'{expanded_src_dim}->{dst_dim}', operand)          # re-order dims
    operand = torch.broadcast_to(operand, [dim_sizes[dim] for dim in dst_dim]) # broadcast
    return operand


def expand_dimensions(operand, src_dim, dst_dim, dim_sizes, fhe_dim_sizes):
    """
    Broadcasts a CKKSTensor from src_dim to dst_dim.

    Args:
        operand (CKKSTensor | torch.Tensor): The tensor to expand.
        src_dim (list): The dimensions of the source tensor.
        dst_dim (list): The dimensions of the destination tensor.
        dim_sizes (dict): The dimension sizes.
        fhe_dim_sizes (dict): The FHE dimension sizes.

    Returns:
        CKKSTensor | torch.Tensor: The broadcasted tensor.
    """

    if isinstance(operand, torch.Tensor):
        return expand_dimensions_torch(operand, src_dim, dst_dim, dim_sizes)

     # no expansion needed
    logger.debug(f'Expanding dimensions (CKKSTensor): {src_dim} -> {dst_dim}')
    if src_dim == dst_dim:
        return operand

    # get the stride of each dim in the src tensor
    src_shape = [fhe_dim_sizes[dim] for dim in src_dim]
    src_stride = torch.tensor(torch.empty(*src_shape).stride())

    # get the stride of each dim in the dst tensor
    dst_shape = [fhe_dim_sizes[dim] for dim in dst_dim]
    dst_stride = torch.tensor(torch.empty(*dst_shape).stride())

    # map src dimensions to their new stride in dst
    src_to_dst_indices = [dst_dim.index(d) for d in src_dim]
    new_stride = dst_stride[src_to_dst_indices]

    # get flattened indices of src tensor
    src_tensor_idxs = torch.cartesian_prod(*[torch.arange(fhe_dim_sizes[dim]) for dim in src_dim])
    if src_tensor_idxs.ndim == 1: # handling 1-d arrays
        src_tensor_idxs = src_tensor_idxs.unsqueeze(1)
    src_idxs = torch.sum(src_tensor_idxs * src_stride, dim=1)
    new_idxs = torch.sum(src_tensor_idxs * new_stride, dim=1)

    # create linear transformation to re-arrange the src to the expanded dst tensor
    if not torch.equal(new_idxs, src_idxs):
        T = torch.zeros((SLOT_COUNT, SLOT_COUNT), dtype=torch.float32)
        T[new_idxs, src_idxs] = 1

        # perform linear transform
        out = bsgs_matrix_mult(T, operand)
    else:
        out = operand
    
    # replicate the missing dimensions
    for dim in reversed(dst_dim):
        if dim not in src_dim:
            stride = dst_stride[dst_dim.index(dim)].item()
            num_rots = int(math.log2(fhe_dim_sizes[dim])) 
            for rep in range(num_rots):
                rotated = out.rotate(stride)
                out = out + rotated
                stride *= 2

    # set new shapes 
    out_shape = torch.Size([dim_sizes[dim] for dim in dst_dim])
    out_fhe_shape = torch.Size([fhe_dim_sizes[dim] for dim in dst_dim])
    out_ndim = len(out_fhe_shape)

    out.shape = out_shape
    out.fhe_shape = out_fhe_shape
    out.ndim = out_ndim
    return out


def multiply_broadcasted_inputs(expanded_inputs):
    """
    Performs an element-wise multiplication between all broadcasted inputs.
    Uses a tree-based reduction to limit multiplicative level. 
    For inputs A, B, C, D:
    Level 1: out1 = A * B
    Level 1: out2 = C * D
    Level 2: out = out1 * out2

    Args:
        expanded_inputs (list): The expanded inputs (each is either a CKKSTensor or a Torch Tensor).

    Returns:
        CKKSTensor: The partial product of the broadcasted inputs.
    """

    if len(expanded_inputs) == 1:
        return expanded_inputs[0]
    
    multiplied_inputs = []
    for i in range(0, len(expanded_inputs), 2):
        if i + 1 < len(expanded_inputs):
            multiplied_inputs.append(expanded_inputs[i] * expanded_inputs[i+1])
        else:
            multiplied_inputs.append(expanded_inputs[i])
    return multiply_broadcasted_inputs(multiplied_inputs)


def reduce_dimensions(partial_product, r_dims, o_dims, dim_sizes, fhe_dim_sizes):
    """
    Performs a reduction over the contraction dimensions.

    Args:
        partial_product (CKKSTensor): The partial product of the broadcasted inputs.
        r_dims (list): The dimensions over which to reduce.
        o_dims (list): The resulting output dimensions.
        dim_sizes (dict): The dimension sizes.
        fhe_dim_sizes (dict): The FHE dimension sizes.

    Returns:
        CKKSTensor: The reduced tensor.
    """

    if not r_dims:  # No reduction needed
        return partial_product

    # get the starting rotation amount and the number of rotations needed
    rot_amount = math.prod([fhe_dim_sizes[dim] for dim in o_dims])
    num_repeated_o_dims = math.prod([fhe_dim_sizes[dim] for dim in r_dims])
    num_rots = int(math.log2(num_repeated_o_dims))

    # perform the reduction
    result = partial_product
    for i in range(num_rots):
        logger.debug(f'Reduction {i}: {rot_amount=}, {num_rots=}')
        rotated = result.rotate(-rot_amount)
        result = result + rotated
        rot_amount *= 2

    # set the new shapes by including only the output dimensions
    out_shape = torch.Size([dim_sizes[dim] for dim in o_dims])
    out_fhe_shape = torch.Size([fhe_dim_sizes[dim] for dim in o_dims])
    out_ndim = len(out_fhe_shape)

    result.shape = out_shape
    result.fhe_shape = out_fhe_shape
    result.ndim = out_ndim
    return result


def mask_slots(cipher):
    """
    Masks the slots of a CKKSTensor to only include the output dimensions.

    Args:
        cipher (CKKSTensor): The CKKSTensor to mask.

    Returns:
        CKKSTensor: All unused slots are set to 0.
    """

    mask = torch.ones(cipher.shape, dtype=torch.float32)
    out = cipher * mask
    return out
