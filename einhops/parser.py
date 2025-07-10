import opt_einsum


def parse_dims(equation, *args):
    """
    Parse the labeled dimensions of the inputs using opt_einsum (https://github.com/dgasmith/opt_einsum).

    Args:
        equation (str): The equation to parse.
        *args (CKKSTensor | torch.Tensor): The input tensors.

    Returns:
        tuple: The input dimensions (list), output dimensions (str), and reduction dimensions (str).

    >>> import torch
    >>> i_dims, o_dims, r_dims = parse_dims("ij,jk->ik", torch.empty(2,3), torch.empty(3,4))
    >>> i_dims
    ['ij', 'jk']
    >>> ''.join(sorted(o_dims))
    'ik'
    >>> ''.join(sorted(r_dims))
    'j'
    """

    input_subs, o_dims, _ = opt_einsum.parser.parse_einsum_input((equation, *args))
    i_dims = input_subs.strip().split(",")

    seen = set()
    r_dims = "".join(
        d for d in "".join(i_dims)
        if d not in o_dims and not (d in seen or seen.add(d))
    )

    return i_dims, o_dims, r_dims


def get_dim_sizes(input_dims, *args):
    """
    Associates all labeled dimensions with their corresponding size.

    Args:
        input_dims (list): The input dimensions.
        *args (CKKSTensor | torch.Tensor): The input tensors.

    Returns:
        dict: The dimension sizes.

    >>> import torch
    >>> i_dims, _, _ = parse_dims("ij,jk->ik", torch.empty(2,3), torch.empty(3,4))
    >>> dim_sizes = get_dim_sizes(i_dims, torch.empty(2,3), torch.empty(3,4))
    >>> dim_sizes
    {'i': 2, 'j': 3, 'k': 4}
    """

    dim_sizes = {}
    for input_dim, arg in zip(input_dims, args):
        for i, dim in enumerate(input_dim):
            dim_sizes[dim] = arg.shape[i]
    return dim_sizes


def get_fhe_dim_sizes(input_dims, *args):
    """
    Associates all labeled dimensions with their corresponding FHE tensor size.
    For EinHops, all FHE tensors have power-of-two sizes in order to enable 
    log-based rotation-and-summation.

    Args:
        input_dims (list): The input dimensions.
        *args (CKKSTensor | torch.Tensor): The input tensors.

    Returns:
        dict: The FHE dimension sizes.

    >>> import torch
    >>> i_dims, _, _ = parse_dims("ij,jk->ik", torch.empty(2,3), torch.empty(3,4))
    >>> fhe_dim_sizes = get_fhe_dim_sizes(i_dims, torch.empty(2,3), torch.empty(3,4))
    >>> fhe_dim_sizes
    {'i': 2, 'j': 4, 'k': 4}
    """

    next_power_of_two = lambda n: 1 << ((n - 1).bit_length())

    fhe_dim_sizes = {}
    for input_dim, arg in zip(input_dims, args):
        for i, dim in enumerate(input_dim):
            fhe_dim_sizes[dim] = next_power_of_two(arg.shape[i])
    return fhe_dim_sizes
