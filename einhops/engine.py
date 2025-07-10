import sys
import math
import torch
import logging
import opt_einsum
from .parser import parse_dims, get_dim_sizes, get_fhe_dim_sizes
from .operations import expand_dimensions, multiply_broadcasted_inputs, reduce_dimensions, mask_slots, SLOT_COUNT


class EinsumEngine:
    """EinHops: Einsum Notation for Expressive Homomorphic Operations on RNS-CKKS Tensors.

    This class is the main driver for EinHops. Steps performed:
    - Validating and parsing an einsum expression
    - Pre-processing each input (i.e. match dimensions)
    - Performing an element-wise multiplication
    - Reducing the output over the contraction dimensions
    """

    def __init__(self):
        self.logger = logging.getLogger("einhops")

    def einsum(self, equation, *args):
        """
        Performs an einsum operation on the inputs.

        Args:
            equation (str): The einsum equation to perform.
            *args: The inputs to the einsum operation (can either be CKKS Tensors or Torch Tensors)

        Returns:
            The result of the einsum operation.
        """

        if self._all_torch_inputs(*args):
            self.logger.warning("All inputs are Torch Tensors. Falling back to opt_einsum.")
            return opt_einsum.contract(equation, *args)

        # Validate the einsum call using opt_einsum.
        self.logger.info("Step 1 (Validation)")
        out_clear_shape = self._validate_expression(equation, *args)

        # Parse the einsum string and get the input, output, and reduction dims.
        self.logger.info("Step 2 (Dimension Semantics)")
        i_dims, o_dims, r_dims = parse_dims(equation, *args)
        broadcasted_dims = r_dims + o_dims

        # Associate each label with its corresponding dimension size.
        dim_sizes = get_dim_sizes(i_dims, *args)
        fhe_dim_sizes = get_fhe_dim_sizes(i_dims, *args)
        self._validate_slot_usage(fhe_dim_sizes)

        # Expand the dimensions of each input to match the desired broadcasted dimension.
        self.logger.info("Step 3 (Match Dimensions)")
        expanded_inputs = []
        for (src_dim, arg) in zip(i_dims, args):
            expanded = expand_dimensions(arg, src_dim, broadcasted_dims, dim_sizes, fhe_dim_sizes)
            expanded_inputs.append(expanded)

        # Perform the multiplication; all inputs have the same dimensions.
        self.logger.info("Step 4 (Multiplication)")
        partial_product = multiply_broadcasted_inputs(expanded_inputs)

        # Reduce over the contraction dimensions.
        self.logger.info("Step 5 (Reduction)")
        out_sum = reduce_dimensions(partial_product, r_dims, o_dims, dim_sizes, fhe_dim_sizes)

        # Mask any slots that are not part of the output dimensions.
        self.logger.info("Step 6 (Gathering the output)")
        out = mask_slots(out_sum)

        assert out.shape == out_clear_shape, f"Error! Output shape mismatch: {out.shape} != {out_clear_shape}"
        return out

    def _validate_expression(self, equation, *args):
        """
        Validates the einsum expression using opt_einsum. 
        If the expression is valid, returns the expected shape of the output.
        """

        if "..." in equation:
            raise ValueError("Ellipsis are not supported in EinHops.")

        if '->' not in equation:
            raise ValueError("Equation must contain ->")
        
        shapes = [arg.shape for arg in args]
        output = opt_einsum.contract(equation, *[torch.empty(shape) for shape in shapes])
        return output.shape
    
    def _all_torch_inputs(self, *args):
        """
        Checks if all inputs to the einsum call are Torch Tensors.
        """

        return all(isinstance(arg, torch.Tensor) for arg in args)
    
    def _validate_slot_usage(self, fhe_dim_sizes):
        """
        Ensures that all data fits within the slots of a CKKSTensor.
        Program will exit if we exceed the slot count.
        """

        num_slots_required = math.prod(fhe_dim_sizes.values())
        if num_slots_required > SLOT_COUNT:
            self.logger.critical(f"Total number of slots ({num_slots_required}) exceeds the maximum number of slots ({SLOT_COUNT}).")
            sys.exit(1)
