import torch
import logging
import desilofhe
from tqdm import tqdm
from .backend import SLOT_COUNT, select_bsgs_factors, fhe_encrypt, fhe_hoisted_rotate, fhe_add, fhe_rotate_fixed, fhe_mul
from .ckkstensor import CKKSTensor


logger = logging.getLogger("einhops")

def bsgs_matrix_mult(matrix, ciphertext):
    """
    Performs a matrix multiplication using the single-hoisted BSGS algorithm.

    Args:
        matrix (torch.Tensor): The matrix to multiply.
        ciphertext (CKKSTensor | torch.Tensor): The ciphertext to multiply.

    Returns:
        CKKSTensor: The result of the matrix multiplication.
    """

    # ensure_keys_initialized()
    shape, fhe_shape, ndim = ciphertext.shape, ciphertext.fhe_shape, ciphertext.ndim

    N1, N2 = select_bsgs_factors(SLOT_COUNT)
    set_rots_needed = get_required_rotation_amounts(matrix, N1, N2)
   
    # hoisted rotations
    logging.debug('Generating pre-rotated ciphertexts. ')
    set_rots = list(sorted([i for i in set_rots_needed]))
    rotated_inputs = fhe_hoisted_rotate(ciphertext.slots, set_rots)
    rotated_inputs_map = {rot_amount: rotated_inputs[i] for i, rot_amount in enumerate(set_rots)}
    logging.debug('Complete.')

    # starting indices for the N1 diagonals
    idxs = torch.arange(N1).unsqueeze(1) + torch.arange(SLOT_COUNT).unsqueeze(0)  # (N1, slots)
    idxs = idxs % SLOT_COUNT


    result = fhe_encrypt(torch.zeros(SLOT_COUNT))
    for j in tqdm(range(N2), disable=logger.level != logging.DEBUG):
        diags = matrix[range(SLOT_COUNT), idxs]

        non_zero_diags = (diags.sum(dim=1) != 0).nonzero(as_tuple=False).squeeze(1)
        if non_zero_diags.numel() > 0:
            curr_block = fhe_encrypt(torch.zeros(SLOT_COUNT))
            for i in non_zero_diags.tolist():
                diag_i = torch.roll(diags[i], shifts=N1 * j)
                prod = fhe_mul(rotated_inputs_map[i], diag_i)
                curr_block = fhe_add(curr_block, prod)
            rotated_block = fhe_rotate_fixed(curr_block, -N1*j)
            result = fhe_add(result, rotated_block)

        # update the indexing for extracting the next set of N1 diagonals   
        idxs = (idxs + N1) % SLOT_COUNT

    level = result.level if isinstance(result, desilofhe.Ciphertext) else ciphertext.level - 1
    return CKKSTensor(result, shape, fhe_shape, ndim, level)


def get_required_rotation_amounts(matrix, N1, N2):
    """
    Determines which of the N1 rotations are actually required
    based upon the structure of the diagonal matrix.

    Args:
        matrix (torch.Tensor): The matrix to multiply.
        N1 (int): Baby-step size
        N2 (int): Giant-step size

    Returns:
        list: The required rotation amounts.
    """
    # starting indices for the N1 diagonals
    idxs = torch.arange(N1).unsqueeze(1) + torch.arange(SLOT_COUNT).unsqueeze(0)
    idxs = idxs % SLOT_COUNT

    set_rots_needed = set()
    for j in range(N2):
        diags = matrix[range(SLOT_COUNT), idxs]

        # find which rotations actually contribute
        nz_rows = (diags.sum(dim=1) != 0).nonzero(as_tuple=False).squeeze(1)
        set_rots_needed.update(nz_rows.tolist())

        # update the indexing for extracting the next set of N1 diagonals   
        idxs = (idxs + N1) % SLOT_COUNT
    
    return list(sorted(set_rots_needed))
    